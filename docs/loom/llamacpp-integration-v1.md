# llama.cpp HRX2 Loom Integration V1

Date: 2026-06-11

## Purpose

HRX2 is a clean replacement track for the current llama.cpp HRX backend. The
current HIP catalog remains useful as workload evidence and reference code, but
HRX2 should not inherit its ad hoc search residue, route names, or hardcoded
kernel-policy structure.

The goal is a Loom-authored, toolchain-tuned, data-driven backend:

```text
model basket and ggml graph facts
  -> missing-kernel and fusion workload inventory
  -> Loom kernel families and provider/config axes
  -> standalone correctness and benchmark evidence
  -> generated target/shape/fusion catalog
  -> minimal llama.cpp HRX2 planner and dispatcher
```

llama.cpp owns graph recognition, tensor/buffer ownership, route construction,
and final HRX dispatch. Loom owns kernel source, provider selection,
configuration materialization, target lowering, compile reports, benchmark
rows, HSACO emission, and artifact manifests. Offline tooling owns tuning and
winner selection.

## Design Principles

- Start from scratch as `HRX2`. Keep the existing HIP HRX backend available for
  reference and fallback until HRX2 wins, then delete/rename later.
- Make kernel families the unit of work. A family names the operation or
  fusion contract, ABI, shape domain, target domain, provider variants,
  correctness policy, benchmark policy, and evidence.
- Keep llama.cpp data-driven. Runtime code should answer "which measured plan
  applies?" rather than encode tuning decisions in C++ branches.
- Use `gfx_id` as the v1 target key. Keep `target_variant` in the catalog schema
  for future user-selected variants such as cooling, power cap, clock policy,
  or site-specific deployment.
- Treat fusions as measured catalog entries. A fusion is accepted only when it
  is measurably faster than the sum of its selected unfused parts on the same
  target and shape class.
- Compile on build, install, model warmup, or cache miss. Never compile in the
  per-token hot dispatch path.
- Prefer explicit source-level schedule intent. If wide vector loads, packed
  dot forms, wave policy, or layout ownership matter, express them in Loom
  source/provider config instead of hoping a compiler recovers them.

## V1 Model Basket

The first basket should fit on a 48 GB W7900 while covering popular dense,
MoE, small, medium, large, reasoning, coding, multimodal-adjacent, and quant
families. The coverage profile intentionally includes public Llama and Gemma
repos because they are representative of real llama.cpp usage.

Default destination:

```text
shared/models/llamacpp-hrx2-basket-v1/
```

Recommended coverage profile:

| Model | GGUF file | Size | Why it is in the basket |
| --- | --- | ---: | --- |
| Qwen3 30B A3B Instruct | `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf` | 16.48 GiB | Popular MoE, current Unsloth default-style quant, exercises `MUL_MAT_ID`, routing, and MoE fusions |
| Qwen3 30B A3B Instruct | `Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf` | 23.37 GiB | Higher-quality K-quant for quant-sensitivity and memory-bandwidth behavior |
| Qwen3 Coder 30B A3B | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` | 17.28 GiB | Very popular coding MoE shape with different graph pressure |
| Llama 3.1 8B Instruct | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.58 GiB | Extremely common dense baseline |
| Llama 3.1 8B Instruct | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` | 7.95 GiB | Dense high-precision quant path, exposes Q8 behavior |
| Llama 3.2 3B Instruct | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 1.88 GiB | Small model overhead sensitivity and dispatch/fusion threshold signal |
| Qwen2.5 Coder 7B | `qwen2.5-coder-7b-instruct-q5_k_m.gguf` | 5.07 GiB | Official GGUF, Q5_K coverage, coding dense workload |
| DeepSeek R1 Distill Qwen 14B | `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` | 8.37 GiB | Reasoning graph, Qwen-derived dense shape, popular public GGUF |
| Mistral Small 3.2 24B | `Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf` | 13.35 GiB | Large dense 24B architecture, long-context and attention pressure |
| Gemma 3 27B IT | `google_gemma-3-27b-it-Q4_K_M.gguf` | 15.41 GiB | Gemma architecture coverage and non-Llama graph corners |
| Phi-4 Mini Instruct | `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` | 2.32 GiB | Small dense corner case and overhead sensitivity |

Observed public metadata on 2026-06-11 showed these repos are public and
ungated. W7900 capacity is based on AMD's published 48 GB GDDR6 specification.
Use the downloader in `tools/download_hrx2_model_basket.py` to materialize the
basket and record exact repo/file metadata.

## HRX2 Backend Shape

The implementation should create a new backend subtree rather than modifying
the current HIP catalog in place:

```text
ggml/src/ggml-hrx2/
  ggml-hrx2.cpp                 runtime, planner, provider registry
  catalog/                      generated catalog inputs/outputs
  loom/                         checked-in Loom source or bytecode packages
  tools/                        catalog generation and validation helpers
```

The public backend can initially remain internal or experimental. If it is
exposed through the normal backend registry, use a distinct name such as
`HRX2` while the old backend still exists.

Runtime data flow:

```text
HRX2 device init
  -> query HRX device architecture
  -> normalize gfx_id
  -> read embedded generated catalog
  -> load matching HSACO or compile/cache Loom artifact at warmup
  -> validate manifest against HRX export metadata
  -> register providers by family, shape domain, and fusion id
```

Planning data flow:

```text
ggml graph node/subgraph
  -> extract op, tensor, layout, quant, and regime facts
  -> lookup exact or bucketed catalog record by gfx_id
  -> optionally lookup target_variant if user specified one
  -> claim route only when catalog evidence and ABI validation match
  -> otherwise fall through to smaller HRX2 route or another backend
```

Minimum runtime records:

```json
{
  "target_key": "gfx1151",
  "target_variant": null,
  "family": "mul_mat_vec_q5_k",
  "shape_key": "decode:k4096:rows4096:cols1:q5_k",
  "root_symbol": "@mul_mat_vec_q5_k_decode",
  "provider": "dot4_vec4_wg32",
  "config": {"workgroup_size": 128, "load_width": 4},
  "artifact": {"format": "amdgpu-hsaco", "fingerprint": "..."},
  "manifest": {
    "binding_count": 3,
    "parameter_count": 0,
    "constant_byte_length": 64,
    "static_workgroup_size": [128, 1, 1],
    "wavefront_size": 32,
    "sgpr_count": 40,
    "vgpr_count": 72
  },
  "evidence": {
    "benchmark_p50_ns": 12345,
    "benchmark_p90_ns": 12700,
    "compile_report": "..."
  }
}
```

Fusion records add the measured comparison:

```json
{
  "fusion_id": "rms_norm_mul_rope",
  "target_key": "gfx1151",
  "shape_key": "decode:ncols4096:rows1",
  "fused_p50_ns": 18000,
  "unfused_sum_p50_ns": 22000,
  "speedup": 1.22,
  "accept": true
}
```

Default fusion acceptance for v1:

- require correctness pass for the fused family and each unfused part;
- require same `gfx_id`, same shape key, same benchmark method;
- require at least three stable benchmark repetitions;
- accept only if fused p50 is at least 3% faster than unfused summed p50 and
  p90/p50 spread is not worse enough to hide the win;
- keep rejected fusions in the tuning evidence store, not in the runtime
  dispatch catalog.

## Catalog Format And Packaging

llama.cpp already vendors `nlohmann/json.hpp` and uses it in `common/`,
`tools/`, and tests. HRX2 should use JSON as the canonical catalog table
format without adding a new dependency.

Use an isomorphic dev and production representation:

```text
dev catalog directory
  catalog.json
  families/*.json
  targets/*.json
  evidence/*.jsonl
  artifacts/*.loombc or *.hsaco

production generated C++
  embedded catalog JSON string
  embedded artifact byte arrays
  generated artifact index
```

The same JSON schema should drive both modes. In dev mode, the backend can load
exploded tables from a catalog directory such as `GGML_HRX2_CATALOG_DIR`. In
production mode, a generator embeds the JSON text in a C++ translation unit and
parses it at device initialization. That keeps the production binary hermetic
while preserving the exact table format used by authoring and tuning tools.

The current HRX backend already generates C++ for embedded artifacts and a
small lookup table. HRX2 should keep that packaging idea, but the generated C++
should not be the schema. It should be a storage vehicle for:

- one canonical JSON blob containing families, targets, shape domains, routes,
  fusion comparisons, compile-report summaries, and manifest facts;
- byte arrays for Loom bytecode, HSACO, or other binary artifacts;
- a generated artifact table that maps JSON `artifact_id` values to embedded
  bytes without requiring JSON to carry large binary payloads.

Runtime parsing should immediately normalize JSON into typed C++ structs and
indexes. The hot path should never traverse arbitrary JSON objects; it should
perform typed lookups by target, family, shape domain, and fusion id.

Minimum top-level JSON shape:

```json
{
  "schema": "ggml-hrx2-catalog-v1",
  "generated_at": "2026-06-11T00:00:00Z",
  "targets": [
    {"target_key": "gfx1151", "target_variant": null}
  ],
  "families": [],
  "routes": [],
  "fusions": [],
  "artifacts": [
    {
      "artifact_id": "mul_mat_vec_q5_k_decode_gfx1151_001",
      "storage": "embedded",
      "format": "amdgpu-hsaco",
      "fingerprint": "..."
    }
  ]
}
```

## Validation Contract

Use two validation layers with different jobs.

Loom-level validation is the fast authoring and tuning gate. It should run
inside the standalone Loom flow with deterministic random seeds, representative
synthetic distributions, boundary cases, and benchmark cases. Do not benchmark
all-zero, all-one, or other fixed-pattern inputs as the only performance
signal; some devices and kernels behave differently on real-ish data. A Loom
candidate that fails synthetic correctness, manifest validation, or performance
sanity is rejected before it reaches llama.cpp.

ggml validation is the source of truth before acceptance. Every accepted HRX2
route must pass focused llama.cpp tests that compare against CPU reference
behavior. The first implementation should mirror the current
`tests/test-backend-hrx.cpp` kernel-level style for HRX2-specific route checks,
and also use `tests/test-backend-ops.cpp` where the normal backend op harness
already compares with the CPU backend. A candidate is not accepted into the
runtime catalog merely because Loom validation passed.

Acceptance rule:

```text
accepted catalog row =
  Loom synthetic correctness pass
  + Loom benchmark evidence
  + manifest/export validation
  + focused ggml CPU-reference correctness pass
  + model-basket smoke coverage where applicable
```

## Shape Evidence

Static shape lists are not enough. HRX2 needs a shape evidence pipeline that
combines observed traces, analytical seed cases, bucket sweeps, and local
breakpoint refinement.

Collect observed shape facts from model-basket runs:

- model file, graph node or subgraph, op/fusion id, quant/layout, dtype;
- target key and optional target variant;
- regime: decode, grouped decode, multi-token prediction, prefill, or batch;
- dimensions such as rows, columns/tokens, K, heads, experts, route count, and
  batch/microbatch facts;
- dispatch count, latency contribution, and current backend owner.

Seed the tuning grid with explicit regime buckets rather than one flat list:

- decode: `n_tokens == 1`;
- very narrow: `n_tokens` in `2..8`, including multi-token prediction;
- narrow/medium: `9..64`;
- prefill: powers of two and non-powers such as `96`, `192`, `384`, `768`,
  `1536`, plus common context and batch split boundaries;
- tail and alignment cases around vector width, tile width, quant block size,
  expert count, and route count;
- MoE route-density cases, including empty, sparse, and high-contention expert
  selections.

Implementations may diverge completely by regime. The catalog should allow
different providers for exact decode, very narrow grouped decode, medium
prefill, and wide prefill. Shape domains can overlap only when priority and
guards are explicit, so the runtime can make a deterministic choice.

Tuning should sweep coarse buckets first, then run denser local sweeps around
breakpoints where winners change. The reducer records the winning provider and
the tested shape range. For unmeasured shapes, selection should use an explicit
bucket rule or a conservative fallback route, not a hidden assumption that one
implementation generalizes continuously.

Minimum shape evidence row:

```json
{
  "shape_key": "mul_mat:q5_k:decode:k4096:rows4096:cols1",
  "source": "observed+seed+sweep",
  "observed_count": 18240,
  "models": ["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"],
  "tested_range": {"cols": [1, 1], "k": [4096, 4096], "rows": [4096, 4096]},
  "winner": "dot4_vec4_wg32",
  "fallback": "mul_mat_q5_k_generic"
}
```

## Tuning Automation

The agent can guide tuning, but broad tuning should be owned by tools rather
than by LLM-coordinated individual benchmark runs. The agent's job is to read
reduced evidence, diagnose gaps or local maxima, propose new provider variants,
and request the next structured sweep.

Tooling should own:

- candidate matrix generation from declared provider/config axes;
- randomized seed selection and synthetic input generation;
- run scheduling and GPU exclusivity;
- correctness gates before benchmarking;
- benchmark result ingestion;
- compile-report and manifest capture;
- Pareto pruning and fusion-vs-unfused reduction;
- catalog JSON generation.

Expected tool sequence:

```text
tools/hrx2_collect_shapes.py
  -> cache/hrx2/shapes/*.jsonl
tools/hrx2_generate_candidates.py
  -> cache/hrx2/candidates/*.jsonl
tools/hrx2_run_loom_sweep.py
  -> cache/hrx2/runs/<run-id>/
tools/hrx2_reduce_tuning.py
  -> cache/hrx2/reduced/<target-key>/*.json
tools/hrx2_emit_catalog.py
  -> ggml/src/ggml-hrx2/catalog/generated/
```

The reducer should produce human-readable summaries for agent review, but its
machine-readable output is what enters the catalog. Agent-authored notes can
explain why the next family/provider variant should exist; they should not be
the tuning database.

## Curriculum

### Phase 0: Instrumented Baseline

Before writing HRX2 kernels, run the model basket through CPU, old HRX, and
other useful GPU backends where available. Capture:

- graph/op inventory and tensor facts;
- backend/provider ownership;
- CPU fallback list;
- dispatch counts and per-dispatch timing;
- decode and prefill regimes;
- correctness smoke outputs.

CPU fallback means a compute graph node or claimed subgraph that must run on the
CPU because HRX2 lacks support. Host-side sampling, tokenization, file I/O, and
normal CPU orchestration are out of scope.

### Phase 1: Unfused Kernel Coverage

For the model basket, identify every unfused compute op that falls back to CPU
and create a discrete HRX2/Loom family for it. Proceed until the basket has no
unexplained compute-kernel CPU fallback.

Output per family:

- Loom family source or bytecode package;
- ABI note and shape domain;
- `check.case` correctness coverage;
- `check.benchmark` rows for decode and prefill samples;
- compile report and artifact manifest;
- generated catalog record;
- llama.cpp route that is selected only by catalog data.

Phase 1 favors broad correctness and coverage over peak speed. The discrete
kernel is still tuned enough to avoid obvious pathologies such as scalarizing a
wide packed load that the author intended to be vectorized.

### Phase 2: Fusion Selection

Select fusion candidates from:

- observed high dispatch count or tiny-kernel overhead;
- dataflow adjacency in ggml graphs;
- CUDA, Vulkan, HIP, and CPU backend patterns;
- analytical reasoning about memory traffic and redundant materialization;
- model-basket frequency and latency impact.

Author fusions as separate Loom families or root wrappers over existing
provider libraries. A fusion is wired into HRX2 only if the standalone evidence
answers:

```text
is fused(target, shape) measurably faster than
sum(best_unfused_parts(target, shape))?
```

Fusion routing should never encode "probably faster" in C++. If the catalog has
no measured win for the active target and shape, do not fuse.

### Phase 3: Aggressive Tuning

Run the offline Loom tuning loop across:

- decode shapes: `n_tokens = 1`, grouped decode, MoE route patterns;
- prefill shapes: powers of two, non-powers of two, boundary sizes, and batch
  splits likely to occur in llama.cpp;
- quant families: Q4_K, Q5_K, Q6_K, Q8_0, IQ/UD variants from the basket;
- target keys: initially `gfx1151`, later additional gfx ids as available;
- fusions and unfused kernels together, so the catalog can select the best
  whole-plan route.

The tuner emits a compact generated catalog. llama.cpp is rebuilt against that
catalog. Live JIT can be introduced later, but v1 should keep deployment
simple: offline bulk tuning, generated data, deterministic runtime lookup.

## Tooling Artifacts

### Model Basket Downloader

`tools/download_hrx2_model_basket.py` downloads GGUFs into `shared/` with
resume and size checks:

```bash
python3 tools/download_hrx2_model_basket.py --dry-run
python3 tools/download_hrx2_model_basket.py --profile coverage
python3 tools/download_hrx2_model_basket.py --dest shared/models/hrx2-smoke
```

The script writes `basket_manifest.json` with repo, file, quant, source URL,
expected size, local path, and profile.

### Draft Skills

- `skills/hrx2-coverage-audit`: run basket workloads and turn provider/CPU
  fallback traces into a prioritized kernel-family backlog.
- `skills/hrx2-loom-family-authoring`: author one Loom family with ABI,
  correctness cases, benchmarks, compile reports, and manifest validation.
- `skills/hrx2-fusion-tuning-catalog`: run offline tuning, compare fused versus
  unfused plans, and emit generated catalog inputs.

## Goal Prompts

Use these as copy-paste starting points for a goal/Ralph loop.

### Phase 0: Interactive Bringup Spike Goal

```text
Goal: Build an interactive HRX2 bringup spike for one unfused op:
GGML_OP_RMS_NORM.

Read AGENTS.md, docs/loom/llamacpp-integration-v1.md, and the
hrx2-loom-family-authoring skill. Use tools/status.py before modifying source
checkouts. Keep this goal interactive: stop and discuss design changes before
making broad backend, catalog, build-system, or Loom-flow decisions that would
set precedent for Phase 1.

Scope:
- create the smallest viable HRX2 backend skeleton needed to claim exactly one
  route for GGML_OP_RMS_NORM;
- author one Loom RMS_NORM family with a semantic baseline and one or two
  explicit provider/config variants;
- create a minimal catalog JSON schema instance for this single op;
- support dev loading from exploded JSON/artifacts, and stub or prototype the
  production path where the same JSON is embedded into generated C++;
- parse catalog JSON at device initialization into typed C++ records, then use
  typed lookup in the planning/dispatch path;
- validate Loom manifest/export facts before route registration;
- add focused ggml CPU-reference correctness coverage for RMS_NORM, modeled on
  the existing HRX kernel-level tests and/or test-backend-ops behavior.

Why RMS_NORM:
- it is common in every basket model;
- it is simple enough to isolate backend/catalog mechanics;
- it still exercises row-wise reduction, constants, shape domains, and
  performance sensitivity across decode and prefill widths;
- it is a likely input to later fusions such as RMS_NORM + MUL + ROPE.

Interactive checkpoints:
- before choosing the permanent HRX2 source tree and CMake layout;
- before finalizing the catalog JSON schema fields;
- before deciding how dev exploded artifacts map to production embedded
  artifacts;
- before adding a general planner/provider abstraction beyond what RMS_NORM
  needs;
- before accepting any workaround caused by current Loom limitations.

Success criteria:
- HRX2 can initialize, load or parse the one-op catalog, and decline all ops
  except RMS_NORM;
- the RMS_NORM route passes focused ggml CPU-reference correctness;
- Loom synthetic validation and at least a small benchmark run are captured as
  JSON/JSONL evidence;
- compile-report and manifest facts are recorded for the accepted candidate;
- the spike produces a short note listing design refinements needed before
  unattended Phase 1 work.
```

### Phase 1: Unfused Coverage Goal

```text
Goal: Drive HRX2 Phase 1 unfused kernel coverage for the HRX2 model basket.

Read AGENTS.md, docs/loom/llamacpp-integration-v1.md, and the
hrx2-coverage-audit and hrx2-loom-family-authoring skills. Use
tools/status.py before modifying source checkouts.

Build or reuse the HRX2 instrumentation needed to run decode and prefill
workloads over the downloaded basket. Collect shape and backend ownership
evidence under cache/hrx2/. Classify every CPU execution as compute fallback,
host orchestration, sampler/tokenizer/I/O, or environment issue.

For the highest-priority missing unfused compute families, author Loom HRX2
families from scratch rather than porting old HIP policy. Use Loom synthetic
validation and benchmarks for fast rejection, then require focused ggml
CPU-reference tests before accepting a catalog route. The production catalog
format is JSON; dev mode may load exploded JSON tables, and production should
embed the same JSON into generated C++.

Success criteria:
- the basket has an auditable compute-fallback backlog with shape evidence;
- each implemented family has ABI notes, Loom checks, benchmark evidence,
  compile-report/manifest facts, and a generated catalog JSON row;
- accepted routes pass ggml CPU-reference correctness;
- no tuning decisions are hardcoded directly into llama.cpp C++.
```

### Phase 2: Fusion Selection Goal

```text
Goal: Build the HRX2 Phase 2 fusion selection and validation pipeline.

Read AGENTS.md, docs/loom/llamacpp-integration-v1.md, and the
hrx2-fusion-tuning-catalog skill. Start from observed Phase 1 traces, ggml
dataflow adjacency, and useful patterns from other llama.cpp backends.

Create a ranked fusion backlog from measured dispatch overhead, memory traffic,
materialization savings, model-basket frequency, and backend precedent. Author
fusions as Loom families or wrappers over reusable provider libraries. Run
Loom synthetic validation and benchmarks first, then focused ggml
CPU-reference tests for every accepted fusion.

For each candidate, measure fused performance against the sum of the best
measured unfused HRX2 parts on the same target key, target variant if present,
shape key, and benchmark method. Record rejected fusions in evidence but do
not emit runtime routes for them.

Success criteria:
- every accepted fusion has same-target, same-shape fused-vs-unfused evidence;
- every accepted fusion passes ggml correctness against CPU reference behavior;
- rejected fusions remain available for later analysis outside the runtime
  catalog;
- llama.cpp routing remains data-driven from catalog JSON.
```

### Phase 3: Tuning And Catalog Goal

```text
Goal: Run HRX2 Phase 3 aggressive Loom tuning and emit the deployable catalog.

Read AGENTS.md, docs/loom/llamacpp-integration-v1.md, and the
hrx2-fusion-tuning-catalog skill. Use tools to run broad sweeps and reducers;
do not coordinate individual benchmark runs through LLM messages.

Collect observed shape distributions from the model basket and combine them
with analytical seed buckets for decode n_tokens=1, grouped decode,
multi-token prediction, narrow/medium token counts, prefill powers and
non-powers, quant block boundaries, tile/vector alignment boundaries, and MoE
route-density corners. Sweep coarse buckets first, then densify around
breakpoints where provider winners change.

Use compile reports, manifest facts, benchmark statistics, and ggml correctness
results to reduce candidates. Generate an isomorphic catalog: exploded JSON for
dev use and the same JSON embedded in production C++ with artifact byte arrays
or artifact indices. Keep hot dispatch lookups typed and pre-indexed after
startup parsing.

Success criteria:
- every route in the generated catalog has correctness, benchmark, manifest,
  and shape-domain evidence;
- narrow, decode, grouped decode, and prefill regimes can choose different
  provider implementations when evidence supports it;
- fusions are included only when they beat the measured sum of their parts;
- the catalog can be regenerated from Loom sources, configs, evidence, and
  reducer policy without hand-editing runtime C++.
```

## Feedback For Loom

Keep accumulating product feedback for the Loom author as implementation
proceeds. Current high-value items:

- stable, documented artifact manifest schema and versioning;
- compile-report population guarantees per AMDGPU backend path;
- benchmark JSON cleanup for tiny-kernel dispatch timing;
- first-class target variant metadata for tuning identity, even if compiler
  legality remains keyed by `gfx_id`;
- ergonomic generated-wrapper flow for exact-shape JIT or offline bulk tuning;
- clear guidance for lane-varying values, VGPR/SGPR intent, and explicit wide
  vector memory operations.

## Acceptance Criteria

HRX2 v1 is successful when:

- the coverage basket runs with no unexplained compute-kernel CPU fallback;
- every runtime route comes from generated catalog data and manifest-validated
  artifacts;
- every accepted fusion has same-target, same-shape evidence against the sum of
  its best unfused parts;
- standalone Loom benchmark evidence predicts llama.cpp route choices;
- old HIP HRX can be disabled without losing the HRX2 coverage target;
- the system can regenerate the catalog from Loom source, benchmark data, and
  tuning policy without hand-editing C++ route decisions.

## References

- Loom authoring guide: `docs/loom/llamacpp-hrx-authoring.md`
- Loom current-answers note: `docs/loom/llamacpp-hrx-stella-questions-2026-06-11.md`
- Prior catalog design note: `docs/spike/analysis/llamacpp_hrx_loom_kernel_catalog_design_2026-06-11.md`
- AMD W7900 product page: https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html
- Hugging Face GGUF repos:
  - https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
  - https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
  - https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  - https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
  - https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
  - https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF
  - https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF
  - https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF
  - https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF
