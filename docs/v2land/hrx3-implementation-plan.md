# HRX3 Implementation Plan

Date: 2026-06-23

This document is the working plan for re-architecting the llama.cpp HRX backend
around Loom kernels, JIT specialization, and offline machine-specific tuning.
It is based on the HRX2 Loom dogfood handoff, the Loom authoring guide, and the
current workspace state.

The goal is not to polish the HRX2 dogfood branch. HRX2 proved that Loom can
match and beat Vulkan on a specific model and machine when the schedule,
runtime submission path, and route selection are controlled. HRX3 should carry
forward the kernel corpus, catalog metadata, tuning evidence, and hard lessons,
while replacing the route-heavy C++ backend shape that made HRX1/HRX2 hard to
generalize.

## Current Workspace State

The root repository tracks workspace metadata only. Implementation changes
belong in independent source repositories under `sources/`.

Current checkout state from `tools/status.py`:

```text
sources/hrx-system: branch main, clean
sources/llama.cpp: branch hrx-integration, clean
sources/llama.cpp-ref: branch users/benvanik/hrx2/loom-dogfood-handoff-20260622, clean
rocm -> /srv/vm-shared/rocm/rocm-head
```

Active planning and handoff documents:

```text
docs/v2land/hrx2-loom-dogfood-handoff-20260622.md
docs/v2land/loom-authoring-optimization-guide-20260622.md
```

Relevant source surfaces:

```text
sources/llama.cpp/ggml/src/ggml-hrx/
  Current hrx-integration backend. It contains runtime plumbing plus many
  checked-in HIP C++ kernels, generated HSACO catalog plumbing, route logic,
  and HIP microbench tools.

sources/llama.cpp-ref/ggml/src/ggml-hrx2/
  HRX2 reference corpus. It contains Loom kernels, split catalog JSON, catalog
  assembly/link/validation tools, embedded artifact generation, and the
  llama.cpp-side Loom JIT shim.

sources/hrx-system/
  HRX runtime and Loom compiler/runtime dependency. Track main unless runtime
  or compiler changes are explicitly approved.
```

## Product Direction

HRX3 is a production rewrite of the llama.cpp HRX integration around these
constraints:

1. C++ backend routes should encode structural eligibility only. Kernel choice,
   launch parameters, concrete shape specializations, evidence, and tuning
   provenance belong in data.
2. Kernel research and tuning must stay route-free and Loom-native until a
   candidate has earned integration. llama.cpp is a validation gate, not the
   workbench where kernels first become correct.
3. Launch parameters and concrete problem sizes must travel with the JIT
   specialization and evidence chain. They must not be split between hard-coded
   C++ routes and a separate kernel catalog.
4. Environment variables may remain as coarse process knobs, but backend code
   must read them once into an options/config snapshot. Repeated env queries in
   graph or dispatch hot paths are not acceptable.
5. Start by JIT-specializing everything to static constraints representing the
   actual launch. Future dynamic relaxation is allowed only after the compiler
   and runtime provide enough metadata and control to make it tractable.
6. The first integrated kernel must be a hard, representative case. The initial
   hero target is `MUL_MAT` `Q4_K x F32 -> F32`, especially the Q4_K_M prompt
   path and related static-shape launch variants.
7. The offline tuning flow must generate catalog metadata for the model basket
   and representative problem-size spreads before the backend grows a large
   route set.

## Explicit Non-Goals For This Sprint

- Do not create a broad catalog of new kernels before the HRX3 architecture is
  stable.
- Do not port HRX2's env-var route forcing maze as the production interface.
- Do not preserve HIP C++ kernels or generated HSACO plumbing in the HRX backend
  after the strip phase, except as temporary reference material outside the new
  runtime path.
- Do not claim performance from route-free microbenchmarks without model-level
  route selection, correctness, and same-session timing evidence.
- Do not modify `sources/hrx-system` unless an HRX/Loom runtime or compiler
  change is clearly required and approved.

## Architecture Principles

### Data-Driven Routes

The backend should answer two questions in C++:

```text
Can this ggml op or graph window be represented by an HRX3 route family?
What concrete shape, layout, and binding facts should be passed to the catalog?
```

It should not hard-code the concrete provider decision. The catalog should own:

- route family and route id
- supported ggml op or graph window
- input and output type/layout contract
- shape domain and guards
- target key
- source id and artifact id
- root symbol and export name
- ABI and binding layout
- dispatch geometry
- JIT config bindings
- tuning parameters
- evidence metadata
- fallback policy

The immediate implementation can still have family-specific shape extraction in
C++ where ggml's tensor semantics require it. That extraction should produce a
small typed shape record consumed by a generic route matcher instead of directly
selecting kernels.

### Structural C++ Constraints

C++ support predicates should be conservative and structural:

- ggml op kind or recognized graph window
- tensor type compatibility
- contiguous or supported stride/layout constraints
- aliasing and view constraints
- shape representability in the HRX ABI
- supported quantization block structure
- required auxiliary buffers or staging availability

They should avoid algorithm-specific choices such as "use route X for this
exact K/rows/cols because it won on gfx1100." That belongs in catalog rows
generated by tuning.

### One Backend Configuration Snapshot

The backend should parse env vars once at device/backend initialization into a
configuration object. Hot paths should read fields from that object. The config
should also be included in trace/evidence output.

Required early fields:

```text
catalog_dir
evidence_dir
trace_jsonl_path
trace_routes
trace_graph
trace_fusion_gates
dispatches_per_submit
max_mul_mat_bytes_per_submit
disable_submit_batching
sync_after_dispatch
dependency_stream_barriers
async_graph_compute
skip_backend_api_synchronize
disable_async_graph_exit_barrier
visible_device_ordinal
provider_cache_policy
fallback_policy
debug_dump controls
```

The current HRX and HRX2 code call env helpers from graph compute, fusion gates,
debug paths, submission batching, and dispatch paths. That pattern should be
removed during the HRX3 import rather than cleaned up later.

### Explicit Fallback

Fallback must be observable and policy-controlled.

Initial policy:

- Default bringup mode may let unsupported ops fall back to CPU so a clean graph
  run can be established after legacy kernel removal.
- Evidence and trace output must distinguish unsupported op fallback from failed
  route selection, JIT compile failure, provider cache failure, and explicit CPU
  scheduling.
- Tuning and performance runs must have a "no silent fallback" mode that fails
  loudly if a covered family misses route selection or falls back to CPU.

The HRX2 handoff preserved a failure where a p64 timing row was actually CPU
Q4_K fallback, not a Loom kernel datapoint. HRX3 should make that impossible to
miss.

### Route-Free Kernel Workbench First

Every serious kernel candidate should start outside llama.cpp:

```text
extract exact ggml/reference shape
  -> write or update one parameterized .loom source
  -> add check.case and check.benchmark rows
  -> compile with reports and retained artifacts
  -> inspect target listing/ISA against a hypothesis
  -> benchmark with dispatch_complete and batch-size=1
  -> preserve result and evidence
  -> integrate only after the candidate earns it
```

llama.cpp integration begins only after the route-free kernel is correct and has
a documented reason to be fast.

### Static JIT Specialization First

The first HRX3 route catalog should assume the JIT sees exact concrete launch
constraints:

- concrete `k`, `rows`, `cols`, token count, head count, or KV length
- concrete workgroup size and dispatch geometry
- exact tile counts and masks
- fixed trip counts when known
- target key and processor
- selected tuning parameters

Use one parameterized Loom source per schedule family where possible. Do not
manually duplicate sources per shape unless the algorithm really changes.

Dynamic generality should come through config-driven specialization and catalog
rows, not C++ route forks. Future work can relax static constraints after Loom
compiler report metadata, cache economics, and runtime dispatch support are
strong enough.

## HRX2 Assets To Carry Forward

The HRX2 branch is source material and evidence, not the architecture to copy.

Carry forward:

- `.loom` kernel bodies in `ggml/src/ggml-hrx2/kernels/`
- split catalog format and route metadata concepts in `ggml/src/ggml-hrx2/catalog/`
- catalog assembly, validation, artifact linking, and embedding tools
- `loom-jit/` llama.cpp-side JIT shim, adapted to HRX3 naming and config
- selected diagnostics, especially root-selected `loom-compile` reproducer
  command generation
- trace/evidence concepts: JSONL route/provider events, provider cache keys,
  compile reports, manifests, and artifact bundle paths
- negative-result records that prevent repeated bad fusions

Do not carry forward unchanged:

- `ggml-hrx2.cpp` as the production architecture
- route forcing through long env-var allow/prefer lists
- command-mode policy embedded in graph matching code
- fusion gates that repeatedly query env vars in hot paths
- dogfood-only graph matchers unless rewritten as data-driven route families
- benchmark phase markers as public API

## HRX2 Kernel Lessons To Preserve

Positive motifs:

- `mul_mat_q4_k_f32.loom` improved when the literal two-iteration inner WMMA
  loop was explicitly unrolled. The lesson is not "unroll everything"; it is
  that source must expose schedule facts the compiler should not guess.
- ROPE matched Vulkan when the host-computed `theta_scale` push-constant value
  was passed into Loom instead of recomputing from `freq_base`.
- The p021 f16 matvec benefited from spelling exact lane terms for the
  `ncols=128`, wave64 decode case.
- RMS 3072-wide routes used vector width 16.
- The Q4 Vulkan-clone matvec needed explicit leader-lane demand for subgroup
  reduction, with bias and store sunk into the `%lane == 0` region.
- Positive fusions preserved useful work decomposition, such as q4/SwiGLU,
  q5 V-cache, and q5 Q-scale-before-RoPE.

Negative motifs:

- Softmax+KQV reduced dispatches but regressed live KV512/KV768 decode because
  it serialized V dot products inside the workgroup.
- Add+RMS and RoPE+scale looked attractive in compile reports but collapsed
  producer parallelism or lengthened SFU-heavy kernels.
- Dispatch barrier skipping was fast and wrong; it corrupted decode.
- Batch-size 16 route-free benchmarks overstated per-dispatch model relevance
  for some rows.
- Route coverage bugs can dominate timing and must be traced separately from
  kernel performance.

## Catalog And Evidence Model

### Catalog Shape

The HRX3 catalog should be generated from split source files and assembled into
a runtime-consumable form. The source-of-truth format should be friendly to
offline tuning tools and code review.

Suggested top-level areas:

```text
catalog/
  metadata.json
  targets.json or targets/
  sources.json
  artifacts.json
  families.json
  routes/
  fusions/
  tuning/
  evidence/
```

Route rows should include enough information to reproduce why a route exists:

```text
identity:
  route id
  family
  op or graph window
  target key
  priority or selection score

source:
  source id
  artifact id
  root symbol
  export name
  loader format

shape:
  domain
  guards
  dynamic/static specialization mode

abi:
  binding count
  parameter count
  constant byte length
  binding layout

dispatch:
  workgroup size
  workgroup count formula or concrete mapping
  rows/cols/elements per workgroup
  subgroup size if required

jit_config:
  bindings from shape facts
  bindings from tuning facts
  bindings from target facts

evidence:
  source of candidate
  tuning run id
  benchmark row id
  correctness fixture id
  compile report id
  target listing id
  known caveats
```

### Kernel Passport

Every hero kernel, tuned route family, and fusion candidate should have a
passport. It can live in generated evidence JSON, a tuning summary, or a
reviewable markdown note, but it must contain:

```text
identity:
  production op/fusion
  model and shape
  reference backend and shader/kernel
  workgroup count
  workgroup size
  subgroup/wave size
  push constants / config values
  binding layout

oracle:
  reference source/SPIR-V/ISA
  captured input tensors or synthetic fixture
  expected output tensors

loom workbench:
  .loom source
  root symbol
  check.case rows
  check.benchmark rows
  target key

measurement:
  route-free benchmark command
  benchmark result
  compile report
  target listing
  timing warnings

integration:
  catalog route
  HRX3 trace proving selection
  completion/PPL gate
  same-session timing pair
```

### Evidence Levels

Every performance or correctness claim should state the highest evidence level
it reached:

| Level | Evidence | Meaning |
| --- | --- | --- |
| 1 | `loom-compile` succeeds | Syntax/lowering exists |
| 2 | Full-output `check.case` passes | Fixture semantics are correct |
| 3 | Route-free benchmark wins | Kernel body can be fast in isolation |
| 4 | Report/listing supports hypothesis | Compiler emitted intended schedule |
| 5 | HRX3 trace selects route | llama.cpp executed intended provider |
| 6 | Completion/PPL passes | Integrated route did not corrupt model |
| 7 | Same-session model timing wins | User-visible performance moved |

Compile reports are evidence, not a scoreboard. Better static counts do not
override correctness and model timing.

## Offline Tuning Workflow

The tuning workflow should be built as the "mini me" version of future kernel
research infrastructure. It should be scriptable, reproducible, and independent
from llama.cpp backend experimentation except for shape collection and final
integration validation.

### Required Inputs

- target machine and device identity
- git hashes and dirty markers for `sources/hrx-system` and `sources/llama.cpp`
- ROCm path and relevant runtime versions
- model basket and GGUF metadata
- representative prompt/decode fixture list
- shape rows collected from llama.cpp graphs
- candidate route families and tuning knobs
- reference backend rows where available, typically Vulkan

Representative spreads should include:

- small, medium, and large prefill
- decode shapes
- power-of-two and non-power-of-two token counts
- tile-boundary cliffs, not just powers of two
- model widths and KV lengths from the target model basket
- batch and ubatch settings that reflect intended llama.cpp use

### Required Outputs

A tuning run should emit one directory under `cache/` or `.tmp/` containing:

```text
run.json
env.json
git.json
stdout.log
stderr.log
shapes.jsonl
candidates.jsonl
results.jsonl or results.json
compile_reports/
manifests/
target_listings/
artifact_bundles/
route_traces/
scheduler_traces/
summary.json
catalog_delta/
```

The summary should identify:

- winning route rows
- rejected route rows and reasons
- correctness failures
- compile failures
- suspicious timing warnings
- route-free wins that failed integration
- route coverage gaps
- compiler/runtime issues needing standalone reproducers

### Workflow Stages

1. Collect shapes from representative llama.cpp runs with trace output enabled.
2. Normalize shapes into route-family candidate records.
3. Generate candidate JIT configs and tuning knobs.
4. Run route-free Loom correctness and benchmarks.
5. Reduce results into target-specific winners and rejected examples.
6. Emit catalog metadata, evidence links, and artifact references.
7. Integrate the catalog into HRX3.
8. Prove route selection in llama.cpp trace.
9. Run correctness gates.
10. Run same-session HRX3/reference timing.

The existing HRX2 workspace flow is a useful starting sketch:

```text
collect shapes
  -> generate candidates
  -> run loom sweep
  -> reduce tuning
  -> emit catalog
```

HRX3 should tighten this flow around provenance and no-silent-fallback checks.

## Initial Hero Kernel

The first integrated kernel should be `MUL_MAT` `Q4_K x F32 -> F32`, focused on
the Q4_K_M hard path. This is intentionally not a toy:

- it is central to quantized model performance
- HRX1 had hard-coded machine-specific behavior in this area
- HRX2 has useful Loom source, route metadata, and negative/positive evidence
- it exercises static JIT shape binding, dispatch geometry, provider caching,
  correctness fixtures, and model-level timing

Initial source material:

```text
sources/llama.cpp-ref/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom
sources/llama.cpp-ref/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q4_k_f32.json
```

Initial route families to evaluate:

- prompt WMMA split-K route for fixed large rows and cols spread
- q8_1 x4 mmq matvec/decode route
- split-K reducer internal routes where still structurally required

Initial acceptance criteria:

- route-free full-output correctness for the selected production shapes
- route-free dispatch benchmark with `dispatch_complete`, batch size 1, compile
  report, manifest, and target listing retained
- catalog row generated from tuning metadata, not hand-written C++ logic
- HRX3 trace proves route id, root symbol, cache key, target key, JIT config,
  and dispatch geometry
- deterministic completion matches reference backend
- PPL smoke is finite and stable against reference on the same corpus
- same-session timing is recorded for mixed p512/n64, prefill p512, and decode
  n64 or the nearest agreed model-basket equivalents

## Migration Phases

### Phase 0: Baseline And Inventory

Objective: make the starting point explicit before destructive backend changes.

Tasks:

- Record source hashes and dirty state for all source checkouts.
- Inventory current `ggml-hrx` HIP kernels, generated catalog plumbing, runtime
  plumbing, env vars, route gates, and fallback behavior.
- Inventory HRX2 reference assets and classify them as:
  - carry-forward runtime/JIT substrate
  - carry-forward Loom source
  - carry-forward catalog/evidence concept
  - diagnostic-only dogfood artifact
  - negative result to preserve
- Identify current HRX runtime plumbing that should survive:
  - backend/device registration
  - buffer types and allocation
  - stream creation and synchronization
  - queue copy/fill paths
  - submission batching
  - staging arena
  - graph compute skeleton

Exit criteria:

- Written inventory or checklist committed as docs or spike notes.
- No code changes required yet.

### Phase 1: Strip Legacy HIP Kernels And Routes

Objective: remove HRX1 HIP kernels and route-specific plumbing while preserving
the HRX runtime integration shell.

Tasks:

- Remove HIP kernel sources and generated HSACO build rules from the active HRX
  backend.
- Remove HIP microbench build targets unless preserved outside the production
  backend as archived reference tools.
- Remove route dispatch implementations tied directly to HIP kernels.
- Preserve buffer management, stream plumbing, backend/device registration,
  copy/fill paths, submission batching, staging, and CPU fallback behavior.
- Make unsupported compute routes fall back cleanly and observably.
- Add trace/scheduler output that can prove all compute kernels are falling
  back during this phase.

Exit criteria:

- HRX backend builds without HIP C++ kernel compilation.
- A llama.cpp run can initialize the backend and execute with CPU fallback for
  all formerly HRX-routed kernels.
- Trace output distinguishes unsupported fallback from runtime errors.

### Phase 2: Import HRX2 JIT And Catalog Substrate

Objective: bring in the Loom JIT path and catalog toolchain without enabling
old HRX2 route behavior wholesale.

Tasks:

- Import/adapt the `loom-jit` shim.
- Adapt CMake to find `hrx` and `loomc` packages from `sources/hrx-system`
  builds.
- Add catalog assembly, validation, linking, and embedding tools.
- Define HRX3 naming and directory structure. Prefer `ggml-hrx` if this is a
  replacement backend; avoid carrying `hrx2` names into production APIs unless
  required for incremental build mechanics.
- Add runtime loading of embedded catalog and optional source catalog directory.
- Add provider cache keyed by route id, target key, source/artifact id, root
  symbol, JIT config, and relevant backend options.
- Add compile evidence dump support:
  - provider JSON
  - compile report JSON
  - manifest JSON
  - root-selected compile command

Exit criteria:

- Backend builds and links against HRX/Loom.
- Catalog can be assembled and validated in the build.
- Runtime can load an empty or no-op catalog.
- All compute still falls back cleanly to CPU.
- No hot-path env var queries are introduced.

### Phase 3: Backend Options Snapshot

Objective: replace ad hoc env-var querying with a production-quality flag path.

Tasks:

- Define `ggml_backend_hrx_options` or equivalent C++ config structure.
- Parse env vars once at backend/device/context initialization.
- Thread config through route matching, graph compute, submission batching,
  tracing, evidence dumping, and debug helpers.
- Convert existing retained env-controlled behavior to config fields.
- Include config in trace/evidence output.
- Add warnings for invalid env values at parse time only.

Exit criteria:

- No route/fusion/dispatch hot path calls `std::getenv`.
- Config defaults are documented.
- Trace output records effective config.

This phase may happen before or alongside Phase 2 if it reduces import churn.

### Phase 4: HRX3 Route Matcher Skeleton

Objective: create the data-driven route selection path with no hero kernel yet.

Tasks:

- Define typed shape records for supported families, starting with quantized
  `MUL_MAT`.
- Build generic catalog matching over family, op, type/layout support, shape
  domain, shape guards, target key, priority, and fallback policy.
- Bind JIT config values from shape and tuning metadata.
- Emit trace events for candidate rejection and selected route.
- Ensure provider cache hit/miss is visible.
- Add no-silent-fallback mode for covered families.

Exit criteria:

- Synthetic or empty catalog tests can prove route matching behavior.
- Trace shows why a route was rejected or selected.
- Unsupported ops still fallback as expected.

### Phase 5: Wire The Q4_K_M Hero Kernel

Objective: integrate one hard Loom kernel end to end.

Tasks:

- Import the relevant Q4 Loom source and catalog seed metadata.
- Run route-free correctness and benchmark rows for target shapes.
- Generate or adapt catalog route rows from evidence.
- Wire shape extraction for `Q4_K x F32 -> F32` `MUL_MAT`.
- JIT compile the selected root with exact static shape/tuning bindings.
- Dispatch through HRX runtime with catalog-provided geometry.
- Emit provider evidence and route trace.
- Run deterministic completion, PPL smoke, and timing gates.

Exit criteria:

- At least one Q4_K_M production shape selects the HRX3 Loom provider.
- CPU fallback is absent for the covered hero family in no-silent-fallback mode.
- Correctness gates pass.
- Same-session timing is recorded against reference backend.
- The evidence packet is sufficient for another agent to reproduce the route.

### Phase 6: Offline Tuning Script Helper

Objective: materialize the mini tuning workflow that produces catalog metadata
for the model basket and shape spreads.

Tasks:

- Add shape collection helper or adapt HRX2 scripts.
- Add candidate generation for Q4 route family.
- Add Loom sweep runner with retained artifacts.
- Add reducer that chooses target-specific catalog rows and preserves rejects.
- Add catalog emitter that writes split metadata suitable for review.
- Add summary output designed for agent consumption.

Exit criteria:

- A tuning run can generate a catalog delta for Q4 shapes.
- Output includes enough provenance to reconstruct commands and inputs.
- Rejected candidates and warnings are preserved.
- The emitted catalog can be validated, built, and used by HRX3.

### Phase 7: Broaden Catalog Deliberately

Objective: expand only after the hero path proves the architecture.

Candidate families, in rough order:

- Q5_K and Q6_K quantized matmuls
- RMS/RMS_MUL
- quantize Q8_1
- RoPE theta-scale
- softmax variants
- contiguous/copy/set rows
- f16 batched/p021 matvecs
- selected fusions with preserved decomposition evidence

Each family must follow the same route-free evidence and integration gates.

## Build And Runtime Expectations

Required HRX/Loom build flags in `sources/hrx-system` remain:

```text
LOOM_TARGET_AMDGPU=ON
LOOM_EMIT_AMDGPU=ON
LOOM_EXECUTE_IREE_HAL=ON
```

If HRX runtime or Loom compiler files change, rebuild `sources/hrx-system`
before rebuilding llama.cpp. A stale static dependency can invalidate the entire
investigation.

Workspace-local paths should remain preferred:

```text
ROCM_PATH=$LLAMACPP_DEVWS/rocm
GGML_HRX_ROCM_PATH=$LLAMACPP_DEVWS/rocm
build trees under build/
scratch and run output under cache/ or .tmp/
```

Timing runs must record device mapping:

```text
ROCR_VISIBLE_DEVICES
HIP_VISIBLE_DEVICES
GPU_DEVICE_ORDINAL
HRX visible device ordinal
VULKAN_BENCH_DEVICE
```

Barrier and command-buffer settings are correctness-sensitive. Any change to
stream barriers, dependency barriers, async graph behavior, PM4/AQL mode, or
backend API synchronization requires deterministic completion and PPL gates
before timing numbers are trusted.

## Correctness And Performance Gates

### Route-Free Gate

- `check.case` full-output correctness passes for the production shape or an
  explicitly justified fixture.
- `check.benchmark` times the same dispatch that correctness validated.
- Benchmark uses `dispatch_complete` and `batch-size=1` for per-dispatch claims.
- Compile report, manifest, and target listing are retained.
- Timing warnings are recorded.

### Integration Gate

- HRX3 route trace proves selected route id, family, target key, root symbol,
  provider cache key, concrete shape, JIT config, and dispatch geometry.
- Scheduler trace proves no CPU fallback for covered route families.
- Provider evidence contains compile report and manifest or points to retained
  artifacts.

### Model Gate

- Deterministic completion matches the reference backend for fixed seed and
  sampling parameters.
- PPL smoke is finite and stable against the same reference/corpus.
- Same-session timing is recorded against reference backend with identical
  model, device pinning, and benchmark flags.

Historical HRX2 timing rows are baseline evidence only. They are not HRX3
claims until rerun on the current source pair.

## Compiler And Runtime Handoff Protocol

When HRX3 exposes a compiler/runtime problem, produce a standalone route-free
reproducer. It should include:

```text
absolute .loom path
root symbol
target key and processor
exact command
expected behavior
actual behavior
check.case or benchmark name
config values
compile_report.json
manifest.json
target listing or disassembly
input/output artifact if correctness-related
smallest known failing variant
larger production context
```

Avoid "llama.cpp is slow/wrong" handoffs. Compiler work moved quickly in HRX2
when the reproducer was focused and route-free.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Recreating HRX2 route maze | Keep C++ structural; move selection and tuning to catalog |
| Env-var overhead in decode | Parse once into backend config; forbid hot-path env queries |
| Silent CPU fallback | Add no-silent-fallback mode and scheduler trace |
| Stale catalog or artifacts | Make catalog assembly/link/validation explicit build steps |
| Provider cache hides changes | Include config/source/root/shape/tuning in cache key and trace |
| Wrong root selected | Emit root-selected compile reproducer |
| Microbenchmark false positive | Require model-level route trace, correctness, and timing |
| Fusion loses parallelism | Require fusion packet with producer/consumer grid economics |
| Device ordinal mismatch | Record and pin device env for every run |
| Barrier shortcut corrupts decode | Correctness gates after any synchronization change |
| Overfitting to gfx1100 | Target-key catalog rows plus separate runs for adjacent machines |
| Static specialization cache explosion | Track provider cache cardinality and compile latency in tuning evidence |

## Immediate Spike Backlog

1. Inventory and strip plan for `sources/llama.cpp/ggml/src/ggml-hrx`.
2. HRX3 backend options snapshot design and env-var audit.
3. Minimal HRX3 catalog schema draft using HRX2 split catalog as input.
4. Minimal Loom JIT import spike that builds but selects no providers.
5. Route matcher skeleton for `MUL_MAT` structural shape extraction.
6. Q4_K_M hero kernel passport from HRX2 source, route metadata, and current
   target machine.
7. Route-free Q4_K_M benchmark/correctness reproduction on current tip-of-tree
   Loom.
8. Tuning run directory schema and summary format.
9. No-silent-fallback trace mode for model-level runs.
10. First integrated Q4_K_M HRX3 route.

## Definition Of Done For This Initiative Slice

This slice is complete when:

- The legacy HIP C++ kernel path has been removed from the active HRX backend.
- HRX runtime plumbing remains intact and can run with explicit CPU fallback.
- HRX3 builds against HRX/Loom and has a catalog/JIT provider path.
- Effective backend config is parsed once and traced.
- The Q4_K_M hero route is integrated through Loom JIT and catalog metadata.
- Offline tuning can generate reviewable catalog metadata for representative
  Q4_K_M shape spreads.
- Correctness and timing evidence exists for the hero route.
- The process for broadening the catalog is documented and repeatable.

