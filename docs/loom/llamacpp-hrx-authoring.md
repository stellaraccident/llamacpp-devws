# Using Loom For llama.cpp HRX Kernels

This guide starts after Loom has been built. It is written for a llama.cpp HRX
kernel author who wants to replace HIP C++ catalog kernels with Loom source,
checked Loom bytecode, offline HSACO artifacts, or an embedded `loomc` JIT.

The useful unit is a kernel family rather than one source file. A family names
the graph operation, buffer and constants ABI, shape domain, target domain,
provider variants, correctness policy, benchmark policy, and the evidence that
explains why one variant should be selected.

```text
llama.cpp graph and tensor facts
  + HRX target/device facts
  + user or environment policy
    -> Loom family
    -> provider/config candidates
    -> link, specialize, compile
    -> correctness and benchmark evidence
    -> executable cache lookup/fill
    -> HRX or raw HSA dispatch
```

Loom owns source-level provider variants, config specialization, target
lowering, executable generation, diagnostics, correctness fixtures, benchmark
rows, and target artifacts. llama.cpp and HRX continue to own graph recognition,
scratch allocation, buffer ownership, constants packing, dependency ordering,
and the final runtime dispatch path.

## Current Contract

The checked implementation supports these paths today:

| Path | Status | Primary references |
| --- | --- | --- |
| Author `.loom` kernels with `check.case` and `check.benchmark` | Available | `loom/src/loom/test/corpus/authoring/README.md`, `loom/src/loom/test/corpus/authoring/*.loom` |
| Format text and bytecode | Available | `loom-format --help` |
| Link families and provider libraries | Available | `loom-link --help` |
| Run pass pipelines and pass-boundary IR tracing | Available | `loom-opt --help`, `loom/src/loom/tooling/pass/trace_cli.h` |
| Compile offline AMDGPU artifacts | Available | `loom-compile --help`, `loom/src/loom/tools/loom-compile/loom-compile-amdgpu.test.json` |
| Plan, run, compare, and profile benchmarks | Available | `iree-benchmark-loom --help`, `iree-benchmark-loom --agents_md` |
| Execute correctness cases through the HAL path | Available | `iree-test-loom --help` |
| Embed Loom through the C API | Available | `loom/binding/c/doc/mainpage.md`, `loom/binding/c/include/loomc` |
| Emit AMDGPU HSACO through `loomc` without HSA | Available | `loom/binding/c/example/emit_amdgpu_offline.c` |
| Compile, load, and launch a Loom HSACO through raw HSA from embedded text, `.loom`, or `.loombc` | Available | `loom/binding/c/example/emit_amdgpu_hsa.c` |
| Integrate the cache/selector into llama.cpp HRX | Application integration | Use the offline or `loomc` artifacts described here from the HRX catalog or JIT cache code |

The AMDGPU CLI backend name is `amdgpu-hal` because the tool path shares the
HAL artifact-provider abstraction used by `iree-test-loom` and
`iree-benchmark-loom`. The artifact bytes produced by the AMDGPU provider are
raw AMDGPU HSACO ELF code object bytes. In current `loom-compile` output, the
primary executable data and optional `--emit-target-artifact` sidecar are the
same HSACO bytes for AMDGPU.

The C API names the same native artifact format as
`LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO`, whose string value is `amdgpu-hsaco`.

## Checked Source Map

Start from checked files rather than scratch experiments.

| Path | Why it matters |
| --- | --- |
| `loom/README.md` | Build targets, current product slice, authoring corpus, C API examples |
| `loom/src/loom/test/corpus/authoring/README.md` | Source authoring rules for helpers, providers, cases, and benchmarks |
| `loom/src/loom/test/corpus/authoring/memset_i8.loom` | Small dynamic-extent kernel with `check.case` and benchmark rows |
| `loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom` | Quantized gate/up SwiGLU family shape with provider selection |
| `loom/src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom` | Parameterized case and benchmark samples over realistic rows |
| `loom/binding/c/doc/mainpage.md` | `loomc` embedding contract and ownership model |
| `loom/binding/c/include/loomc/compile.h` | Prepared compiler, per-invocation config, compile roots, artifacts |
| `loom/binding/c/include/loomc/link.h` | Programmatic linking and provider library composition |
| `loom/binding/c/include/loomc/target.h` | Target profiles, selections, and target pipelines |
| `loom/binding/c/include/loomc/target/amdgpu.h` | Public AMDGPU target aggregation header |
| `loom/binding/c/include/loomc/target/amdgpu/profile.h` | AMDGPU processor profiles, HSA ISA normalization, targetless kernel assignment |
| `loom/binding/c/include/loomc/target/amdgpu/emit.h` | AMDGPU HSACO emission option extension |
| `loom/binding/c/example/emit_amdgpu_offline.c` | Minimal offline `loomc` compile/emit to HSACO |
| `loom/binding/c/example/emit_amdgpu_hsa.c` | Raw HSA load and launch of a Loom-produced HSACO from embedded text, `.loom`, or `.loombc` |
| `loom/binding/c/example/emit_spirv_vulkan.c` | Parallel raw-runtime example for SPIR-V/Vulkan |
| `loom/src/loom/tools/iree-benchmark-loom/help.c` | Dense benchmark output schema and `jq` recipes |
| `loom/src/loom/tooling/pass/trace_cli.h` | Shared pass IR trace flags for `loom-opt` and `loom-compile` |

For older experiments, copy ideas and schedules, not syntax. The checked
authoring corpus and tool help are the source of truth for current spelling and
workflow.

## Invocation Style

When working inside the source tree, the wrapper form builds the tool and keeps
the command root-relative:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-compile:loom-compile -- \
  path/to/kernel.loom \
  --help
```

When binaries are already on `PATH`, use the short form:

```bash
loom-compile path/to/kernel.loom --help
```

Every public Loom tool now has a curated `--help` surface. The complete linked
IREE/runtime flag registry is still available through `--help=all`:

```bash
loom-compile --help
loom-compile --help=all
```

`iree-test-loom` and `iree-benchmark-loom` also consume inherited HAL runtime
flags for device selection and profiling. The curated help describes the
Loom-facing workflow; `--help=all` is the inventory for flags such as
`--device` and the device profiling mode controls.

`--agents_md` is the standard agent-snippet spelling where a tool offers it.
Today that is most useful on `iree-benchmark-loom` and `loom-check`:

```bash
iree-benchmark-loom --agents_md
loom-check --agents_md
```

## Family Anatomy

A family note should be short, concrete, and close to the source. For a Q4/Q8
MoE gate/up SwiGLU replacement, name at least the following.

Operation contract:

```text
dst[token, route, row] =
  swiglu(dot_q4_q8(gate_weight[expert, row], activation[token]),
         dot_q4_q8(up_weight[expert, row], activation[token]))
```

ABI contract:

```text
bindings:
  0: gate weights
  1: up weights
  2: activation or quantized activation scratch
  3: route ids or compact route records
  4: output

constants:
  k
  rows
  n_ids
  n_tokens
  source strides
  output strides
  route-group and tile parameters
```

Shape facts:

```text
k: reduction dimension
rows: expert output rows
n_ids: routed experts per token
n_tokens: prompt/decode shape regime
```

Provider variants:

```text
semantic baseline
direct q4/q8 dot
Q8 register reuse
explicit vector<4xi32> packed loads
LDS staging probe
WG32 or WG64 variants
route-grouped variants
tile-shape variants
```

Evidence records:

```text
compile report
pass IR trace
target listing
HSACO artifact
benchmark JSON or JSONL rows
profile bundle summaries or counters
llama.cpp/HRX focused validation result
```

The pressure to keep this family note precise is practical: every provider
variant, offline artifact, and JIT cache entry must agree on the same ABI,
shape facts, correctness cases, and benchmark rows.

## Author `.loom` Source

Source describes semantics and reusable implementation contracts. Provider
libraries describe schedule variants and target-oriented choices. Configuration
binds the family to one exact candidate.

```text
family source:
  kernel ABI, operation contract, helper contracts, check.case, check.benchmark

provider library:
  func.template implementations, vectorization choices, tile shapes,
  route grouping, target-specific schedules

configuration:
  selected provider, shape specialization, target key, tuning choices
```

Common authoring features:

| Feature | Role |
| --- | --- |
| `kernel.def` | Exported kernel ABI and launch contract |
| `func.call` | Exact helper call when one symbol is required |
| `func.apply<K>` | Implementation demand for provider selection |
| `func.template` | Provider body that can satisfy an implementation contract |
| `inline` | Boundary intent for helpers that should disappear before target lowering |
| `noinline` | Boundary intent for callables that should survive as real symbols |
| `check.case` | Correctness workload and oracle |
| `check.param.choice` | Named concrete sample choices for shapes/config |
| `check.benchmark<@case>` | Timing rows tied to one correctness case |

The authoring corpus is the best first reference:

```bash
python dev.py bazel test \
  //loom/src/loom/test/corpus/authoring:ffn_gate_up_swiglu_q6q8_plan_test \
  //loom/src/loom/test/corpus/authoring:mlp_down_projection_residual_bf16_plan_test
```

Use the checked corpus when validating tools and local setup. Commands below
that mention `q4_moe_swiglu` describe the HRX family shape to build; commands
that mention `memset_i8` are copy-paste smoke tests against checked source.

## Format And Package Source

`loom-format` normalizes text and converts between text and Loom bytecode:

```bash
loom-format input.loom \
  --from=auto \
  --to=text \
  --output=input.formatted.loom

loom-format input.loom \
  --from=auto \
  --to=bytecode \
  --output=input.loombc
```

`loom-link` builds archives, links roots against provider libraries, binds
configuration, strips check records for deployment artifacts, and prints plans
or symbol tables:

```bash
loom-link family.loom \
  --library=providers.loombc \
  --root=@q4_moe_swiglu \
  --config=tile_m=32 \
  --config-file=q4_moe_gfx1100.json \
  --to=bytecode \
  --strip-check \
  --require-resolved-config \
  --output=q4_moe_swiglu.linked.loombc

loom-link family.loom \
  --library=providers.loombc \
  --root=@q4_moe_swiglu \
  --print-plan

loom-link family.loom \
  --library=providers.loombc \
  --list-symbols
```

This is the command-line version of the same composition a llama.cpp JIT uses
through `loomc`: load source or bytecode modules, bind config facts, link
selected providers, compile an exact executable, and cache the result.

Runnable packaging smoke over the checked `memset_i8` source:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-format:loom-format -- \
  loom/src/loom/test/corpus/authoring/memset_i8.loom \
  --from=auto \
  --to=bytecode \
  --output=/tmp/loom-hrx-memset_i8.loombc

python dev.py bazel run //loom/src/loom/tools/loom-link:loom-link -- \
  /tmp/loom-hrx-memset_i8.loombc \
  --from=auto \
  --list-symbols

python dev.py bazel run //loom/src/loom/tools/loom-link:loom-link -- \
  /tmp/loom-hrx-memset_i8.loombc \
  --from=auto \
  --root=@memset_i8 \
  --to=bytecode \
  --strip-check \
  --output=/tmp/loom-hrx-memset_i8.linked.loombc
```

## Correctness And Benchmark Loop

Dry-run planning catches source, case, benchmark, sample, and config mistakes
without requiring a GPU:

```bash
iree-benchmark-loom family.loom \
  --dry-run \
  --output=plan.json
```

Run correctness cases through the HAL-backed execution path:

```bash
iree-test-loom family.loom \
  --case=@smoke \
  --sample=0 \
  --pipeline=default
```

Run timing with debug artifacts:

```bash
iree-benchmark-loom family.loom \
  --device=amdgpu \
  --benchmark=@q4_moe_swiglu_prompt \
  --measure=dispatch_complete \
  --sample-compilation=once \
  --batch-size=64 \
  --warmup-iterations=4 \
  --iterations=16 \
  --min-time-ms=100 \
  --artifact-bundle-dir=loom-run \
  --artifact-bundle-policy=debug \
  --output-format=jsonl \
  --output=loom-run/results.jsonl
```

Compare two candidates with interleaved dispatch timing:

```bash
iree-benchmark-loom family.loom \
  --device=amdgpu \
  --compare=@baseline,@candidate \
  --interleave=ABABA \
  --repetitions=5 \
  --sample=0 \
  --measure=dispatch_complete \
  --batch-size=64 \
  --profile-final-batch=false \
  --output-format=jsonl \
  --output=ababa.jsonl
```

Collect final-batch profile evidence outside the measured timing window:

```bash
iree-benchmark-loom family.loom \
  --device=amdgpu \
  --benchmark=@q4_moe_swiglu_prompt \
  --measure=dispatch_complete \
  --batch-size=64 \
  --profile-final-batch=true \
  --profile-data=dispatch-events,executable-metadata \
  --artifact-bundle-dir=profile-run \
  --artifact-bundle-policy=debug \
  --output-format=jsonl
```

Tiny kernels are often dominated by host and queue overhead when measured one
dispatch at a time. `--batch-size`, `--input-ring-min-bytes`, and
`--input-ring-count` make that policy explicit. Use `--input-ring-count=1` for
deliberate hot-reuse measurements and the default auto ring for cache-thwarting
dispatch timing.

Useful JSONL queries:

```bash
jq 'select(.row=="compile" and .diagnostics) | .diagnostics[]?' results.jsonl
jq 'select(.row=="compile" and .static_summary) | {candidate_id,code:.static_summary.code_byte_count,spills:.static_summary.allocation_spill_count,local:.static_summary.local_memory_bytes}' results.jsonl
jq 'select(.row=="compile" and .compile_report_path) | {candidate_id,path:.compile_report_path}' results.jsonl
jq 'select(.row=="compile" and .target_artifact_path) | {candidate_id,target:.target_artifact_path,listing:.target_listing_path,hal:.hal_executable_path}' results.jsonl
jq 'select(.row=="benchmark") | .benchmark_result | {benchmark,status,p50:.operation_timing_ns.p50,data_cache}' results.jsonl
jq 'select(.row=="comparison") | {candidate_id,baseline_candidate_id,ratio_p50,speedup_p50,ratio_p90,speedup_p90}' results.jsonl
```

## Pass And Compile Debugging

`loom-opt` and `loom-compile` share pass-boundary IR trace flags:

```bash
--dump-ir-before=<pass-or-stage>
--dump-ir-after=<pass-or-stage>
--dump-ir-before-all
--dump-ir-after-all
--dump-ir-format=text|jsonl
--dump-ir-output=stderr|stdout|-|<file>|<directory>/
```

Human-readable stderr trace:

```bash
loom-opt family.loom \
  --pass=canonicalize \
  --pass=dce \
  --dump-ir-after=dce \
  --dump-ir-format=text \
  --dump-ir-output=stderr
```

Agent-queryable JSONL trace bundle:

```bash
loom-compile family.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --compile-root=@q4_moe_swiglu \
  --output=q4_moe_swiglu.hsaco \
  --dump-ir-after-all \
  --dump-ir-format=jsonl \
  --dump-ir-output=trace/
```

Directory output writes a `trace.jsonl` index plus per-event `ir/*.loom`
artifacts. That shape lets agents filter by pass key, stage, symbol, status, or
event ordinal without scraping a terminal transcript.

Runnable bytecode-to-HSACO smoke with a compile report and trace bundle:

```bash
python dev.py bazel run //loom/src/loom/tools/loom-compile:loom-compile -- \
  /tmp/loom-hrx-memset_i8.linked.loombc \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --compile-root=@memset_i8 \
  --module-name=memset_i8 \
  --output=/tmp/loom-hrx-memset_i8.hsaco \
  --emit-target-artifact=/tmp/loom-hrx-memset_i8.native.hsaco \
  --compile-report=details \
  --compile-report-output=/tmp/loom-hrx-memset_i8.compile_report.json \
  --compile-report-row-limit=32 \
  --dump-ir-after-all \
  --dump-ir-format=jsonl \
  --dump-ir-output=/tmp/loom-hrx-memset_i8_trace/

file /tmp/loom-hrx-memset_i8.hsaco
jq -r 'select(.point=="after") | [.stage,.pass,.artifact_path] | @tsv' \
  /tmp/loom-hrx-memset_i8_trace/trace.jsonl
```

Use the `gfx*` processor matching the HRX target host. The raw HSA `loomc`
example shown later demonstrates deriving that processor key from the live HSA
agent instead of spelling it on the command line.

Compile reports complement pass traces:

```bash
loom-compile family.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --compile-root=@q4_moe_swiglu \
  --output=q4_moe_swiglu.hsaco \
  --compile-report=details \
  --compile-report-output=q4_moe_swiglu.compile_report.json \
  --compile-report-row-limit=64
```

Supported report requests are `summary`, `details`, `json`,
`json-summary`, `json-details`, `text`, `text-summary`, `text-details`, empty,
and `none`. JSON is the default shape to hand agents. Text is useful while
working in a terminal.

When a compile, correctness, or benchmark run fails, collect the smallest
bundle that preserves mechanism:

```text
source .loom or linked .loombc
config JSON and direct --config bindings
full command line
pass trace bundle
compile report details
benchmark JSONL or snapshot
artifact bundle manifest
target listing
HSACO bytes when emission succeeded
```

## Offline HRX Catalog Path

The baseline integration path is intentionally boring:

```text
Loom family source/provider package
  -> loom-link selected provider/config
  -> loom-compile --backend=amdgpu-hal --target=<gfx>
  -> raw AMDGPU HSACO ELF bytes
  -> generated HRX catalog entry
  -> existing llama.cpp provider selection and dispatch
```

Compile a targetless kernel family for one AMDGPU processor:

```bash
loom-compile q4_moe_swiglu.linked.loombc \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --compile-root=@q4_moe_swiglu \
  --module-name=q4_moe_swiglu \
  --output=q4_moe_swiglu_gfx1100.hsaco \
  --emit-target-artifact=q4_moe_swiglu_gfx1100.native.hsaco \
  --compile-report=details \
  --compile-report-output=q4_moe_swiglu_gfx1100.compile_report.json
```

For AMDGPU today:

```text
--output=<path>                 raw HSACO ELF bytes used as executable data
--emit-target-artifact=<path>   optional raw HSACO sidecar for catalogs/debug
--target=<gfx*>                 assigns targetless kernel.def ops before codegen
--compile-root=@symbol          scopes root-sensitive target behavior
```

An HRX catalog record should preserve at least:

```json
{
  "name": "q4_moe_swiglu_loom_vec4",
  "family": "q4_moe_swiglu",
  "provider": "q8_reg_reuse_vec4_load",
  "gfx_target": "gfx1100",
  "artifact_format": "amdgpu-hsaco",
  "artifact_path": "q4_moe_swiglu_gfx1100.hsaco",
  "entry_point": "q4_moe_swiglu",
  "binding_count": 5,
  "constants_size": 64,
  "workgroup_size": [64, 1, 1],
  "shape_domain": {
    "k": 2048,
    "rows": 512,
    "n_ids": 8,
    "n_tokens": [1, 8]
  },
  "evidence": {
    "compile_report": "q4_moe_swiglu_gfx1100.compile_report.json",
    "benchmark_rows": "q4_moe_swiglu_gfx1100.results.jsonl"
  }
}
```

The HRX loader side should validate the artifact by loading the bytes, querying
the export metadata, checking binding and parameter counts, and checking the
selected workgroup size before registering the provider. The runtime selector
can then choose between existing static HIP-produced HSACO entries and
Loom-produced HSACO entries with the same shape policy.

For direct HRX executable dispatch, the public API surface is:

```text
hrx_executable_load_file or hrx_executable_load_data
hrx_executable_lookup_export_by_name
hrx_executable_export_info
hrx_stream_dispatch or hrx_queue_dispatch
```

The observed metadata is part of the provider contract. A one-buffer smoke
kernel should report an export name, `constant_byte_length = 0`,
`binding_count = 1`, and the workgroup size that the dispatch code will use.
For real catalog kernels, mismatch between the manifest and
`hrx_executable_export_info` is a loader failure, not a reason to guess at
runtime.

## Embedded `loomc` JIT Path

The `loomc` flow mirrors the CLI path but keeps source, modules, diagnostics,
and artifacts in memory:

```text
llama.cpp starts
  -> initialize HRX/HSA runtime
  -> query selected GPU agent
  -> derive AMDGPU processor, such as gfx1100 or gfx942
  -> create Loom AMDGPU target environment and context
  -> load family source and provider bytecode
  -> bind graph/tensor/config facts
  -> link the exact root
  -> assign targetless kernels to the selected AMDGPU profile
  -> run the prepared target pipeline
  -> emit LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO
  -> load bytes through HRX or raw HSA
  -> cache by source hash, provider/config, target, ABI, and shape facts
```

The public AMDGPU target headers are intentionally free of HSA, HIP, ROCm, and
IREE HAL types. The host runtime adapter queries its own device API and passes
only normalized target facts into Loom.

Offline `loomc` example:

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_offline -- \
  gfx1100 \
  /tmp/targetless_store_i32.hsaco
```

This example:

```text
creates an AMDGPU target environment
creates a context with AMDGPU dialects registered
creates a processor profile from gfx*
assigns targetless kernel.def ops
creates a prepared-low target pipeline
compiles the module
emits an executable artifact with format amdgpu-hsaco
writes HSACO bytes when an output path is supplied
```

Raw HSA example:

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_hsa
```

Successful launch ends with:

```text
launched targetless_store_i32 via raw HSA: output=42
```

The same example can start from a `.loom` or `.loombc` file. This is the
minimal catalog/JIT shape: checked source is formatted or linked to bytecode,
then an embedder reads those bytes and compiles for the live HSA target.

```bash
python dev.py bazel run //loom/binding/c/example:emit_amdgpu_hsa -- \
  /tmp/loom-hrx-targetless_store_i32.loombc \
  @targetless_store_i32 \
  targetless_store_i32 \
  targetless_store_i32
```

The positional arguments are source path, compile-root symbol, emitted HSA
kernel symbol, and module name. The sample launch path intentionally assumes the
one-buffer `targetless_store_i32` ABI and checks that the kernel writes `42`.
Use it to validate the `.loom`/`.loombc -> loomc -> HSACO -> HSA` flow before
adapting the code to a real HRX kernel ABI.

Set `LOOMC_HSA_RUNTIME_PATH` when the HSA runtime is not discoverable through
the default dynamic loader search:

```bash
LOOMC_HSA_RUNTIME_PATH=/opt/rocm/lib \
  python dev.py bazel run //loom/binding/c/example:emit_amdgpu_hsa
```

The raw HSA example dynamically loads only HSA symbols. It does not include the
IREE HAL. Its shape is the embedding proof llama.cpp/HRX agents should copy:

```text
load HSA runtime
find CPU and GPU agents
read HSA ISA target id
normalize to Loom AMDGPU processor
compile and emit HSACO through loomc
create HSA code object reader from memory
load and freeze executable
query kernel symbol metadata
allocate output and kernarg memory
write one AQL dispatch packet
ring the queue doorbell
wait for completion
check output value
```

The `emit_spirv_vulkan.c` example is the same architectural pattern for
SPIR-V/Vulkan: query the live raw runtime, build a Loom target profile, compile,
emit, and hand bytes directly to that runtime. The AMDGPU/HSA example is the
analog the HRX integration can adapt directly.

## Specialization Patterns

There are two useful specialization styles.

Config-bound authored family:

```text
family.loom declares config-sensitive choices
provider libraries define candidates
CLI or loomc binds config values
link/compile materializes the selected source
```

CLI shape:

```bash
loom-link family.loom \
  --library=providers.loombc \
  --root=@q4_moe_swiglu \
  --config=provider=q8_reg_reuse_vec4_load \
  --config=tile_m=32 \
  --config-file=shape_gfx1100.json \
  --to=bytecode \
  --output=q4_moe_swiglu.specialized.loombc
```

Generated-wrapper family:

```text
checked provider library contains reusable functions/templates
llama.cpp generates a tiny in-memory .loom wrapper with one kernel.def
wrapper hard-codes shape or ABI facts for the exact graph instance
loomc links wrapper against provider bytecode
loomc compiles and emits HSACO
```

This wrapper approach is attractive for JITs because the host can keep the
checked provider library stable while generating only the small root that
reflects live graph shape, tensor layout, target, and policy facts.

## llama.cpp Selector Shape

Keep the existing HRX selector facts visible:

```text
graph family
k
rows
n_ids
n_tokens
strides
route count and route layout
q8 scratch availability
target processor
wavefront behavior
supported dot forms
LDS and register pressure envelope
environment overrides
cached evidence threshold
```

The provider record can point at either:

```text
precompiled HSACO catalog entry
JIT cache entry compiled from the same family/provider/config package
```

Dispatch remains runtime-owned:

```text
route scratch allocation
route count clearing
route compaction
optional Q8 activation packing
HRX buffer binding
constant packing
kernel submission and dependency ordering
```

Loom should receive those choices as explicit source, config, target, binding,
or dispatch facts. It should not rediscover command ordering or buffer
dependencies by inspecting kernel arguments.

## Representative Family Backlog

Q4 MoE SwiGLU is a strong pilot because it exercises packed quantized data,
route-aware expert selection, gate/up reuse, nonlinear epilogue, target-specific
provider choice, correctness fixtures, benchmark rows, and artifact packaging.

The same guide structure applies to the rest of the HRX catalog:

| Family | Loom pressure |
| --- | --- |
| Quantized matvec and mat-id | Q4/Q5/Q6/Q8 layouts, routed MoE, q8 packing helpers, row tile variants |
| Dense matvec | F16/BF16/F32 paths, epilogue fusions, prompt/decode shapes |
| Attention decode and prefill | Split/reduce decode, prefill, K/V formats, softmax policy |
| Row and norm operations | RMS norm, RMS norm plus multiply, RoPE fused forms, row reductions |
| Elementwise fusions | Add/multiply/divide chains, SiLU/SwiGLU/softplus epilogues |
| Data movement | Copy, concat, get/set rows, conversion, quantize/dequantize helpers |
| MoE routing | Top-k, compact routes, route counting, route scratch layout |
| Stateful kernels | SSM conv/update and gated delta net |

Each family should get one semantic baseline, one realistic correctness case,
one dry-run benchmark plan, one debug-artifact benchmark run, and one offline
HSACO package before schedule variants multiply.

## Feedback Package

When a miscompile, unsupported feature, performance cliff, or integration bug
appears, send a compact reproducer bundle with the evidence that keeps the
mechanism visible:

```text
source .loom files or linked .loombc package
provider libraries used by the root
config JSON and direct config bindings
exact command line or loomc option structs
target processor and HSA ISA name when AMDGPU is involved
pass trace bundle with --dump-ir-format=jsonl
compile report details
benchmark JSONL rows and artifact bundle manifest
target listing
HSACO bytes when emission succeeded
loader/export metadata observed by HRX or HSA
expected and actual output for correctness failures
```

For tool-shape requests, name the evidence row or artifact that was missing.
Examples:

```text
compile report needs a per-kernel wavefront mode row
JSONL compile row needs a direct path to the selected provider config
pass trace needs a stable stage key before target assignment
HRX loader needs an artifact manifest field for constants_size
benchmark rows need a cache-hot/cache-thwarting label
```

Concrete capture command for a compile or emission failure:

```bash
loom-compile repro.linked.loombc \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --compile-root=@repro_kernel \
  --module-name=repro_kernel \
  --output=repro_kernel.hsaco \
  --emit-target-artifact=repro_kernel.native.hsaco \
  --compile-report=details \
  --compile-report-output=repro_kernel.compile_report.json \
  --compile-report-row-limit=128 \
  --dump-ir-after-all \
  --dump-ir-format=jsonl \
  --dump-ir-output=repro_kernel_trace/

file repro_kernel.hsaco
llvm-nm --defined-only repro_kernel.hsaco
jq -r 'select(.point=="after") | [.stage,.pass,.artifact_path] | @tsv' \
  repro_kernel_trace/trace.jsonl
```

For HRX or raw-HSA loader failures, keep the loader evidence next to the
compiler evidence: target processor, HSA ISA name, artifact format string,
export name, export ordinal, constant byte length, binding count, parameter
count, workgroup size, and the exact status returned by
`hrx_executable_load_*`, `hrx_executable_export_info`,
`hsa_executable_load_agent_code_object`, or
`hsa_executable_get_symbol_by_name`.

That is the product loop Loom is built for: source, compiler facts, artifacts,
runtime loader evidence, correctness, and benchmark data stay close enough that
humans and agents can repair the next layer without guessing.
