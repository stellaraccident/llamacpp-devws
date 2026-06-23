# Loom Kernel Authoring And Optimization Guide - 2026-06-22

Audience: agents and humans trying to author Loom kernels or fusions that need
to match a production GPU backend, especially HRX2/llama.cpp-style routes where
there is a known Vulkan/HIP/CUDA oracle.

This guide is distilled from the Phi-4-mini Q4_K_M dogfood run. The punchline
is uncomfortable but useful: writing Loom arithmetic was not the expensive
part. Most time disappeared into proving that the measured artifact was the
intended artifact, proving that the correctness oracle encoded the same
semantics as the reference backend, and separating compiler issues from stale
HRX2/llama.cpp integration state.

The successful loop made those hidden states explicit and then kept moving.
Compiler engineers were productive because they got standalone reproducers,
not vague reports that llama.cpp was slow. Kernel work made progress because a
blocked kernel became a bead/reproducer and the author moved to another route
instead of waiting.

## Working Position

Loom is WYSIWYG kernel authoring with target specialization. The compiler is
allowed to lower, allocate, packetize, and specialize from the facts in the
source/config. It should not be expected to discover a completely different
whole-kernel schedule that the source did not express.

That changes the authoring contract. If the Vulkan shader is fast because it
uses a particular work partition, subgroup shape, tile size, LDS staging layout,
dot primitive, split-K decomposition, or push-constant specialization, the Loom
source needs to express that schedule class. A source that merely computes the
same mathematical function can be correct and still lose by an order of
magnitude.

The model-level stack is a gate, not the inner loop:

```text
trace/reference backend
  -> route-free .loom workbench
  -> full-output correctness
  -> dispatch benchmark and compile report
  -> target listing / ISA comparison
  -> integrate one route
  -> prove selected route
  -> deterministic completion / PPL / same-session benchmark
```

llama.cpp, HRX2, and route tables are noisy integration machinery. They are
allowed to confirm or reject a kernel. They should not be the place where the
kernel is first made correct.

## Evidence Hierarchy

Every claim should say which level it reached.

| Level | Evidence | What It Proves | What It Does Not Prove |
| --- | --- | --- | --- |
| Source compiles | `loom-compile` succeeds | The syntax/lowering path exists | Runtime correctness or speed |
| Full-output check passes | `check.case` or benchmark correctness passes | The selected input/output contract is right for that fixture | Production route selection |
| Route-free benchmark wins | `iree-benchmark-loom` beats baseline on the checked case | The kernel body can be fast in isolation | Model wall-time movement |
| Compile report/listing matches hypothesis | report + disassembly show expected resources/instructions | The compiler emitted the intended schedule motifs | That the schedule is globally useful |
| HRX2 route trace selects it | `GGML_HRX2_TRACE_JSONL` dispatch rows show route/cache key | The model stack executed the intended provider | Correctness or performance |
| Completion/PPL passes | deterministic text and PPL smoke are stable | The integrated route did not obviously corrupt the model | Performance |
| Same-session model timing wins | HRX2/Vulkan or variant/baseline ABABA rows | User-visible movement | Generality across models/devices |

Compile reports are evidence, not a scoreboard. Some losing fusions had better
instruction counts and fewer dispatches. The report explains mechanisms; the
benchmark decides whether the mechanism mattered.

## Kernel Passport

Before a serious optimization starts, create a small "passport" for the kernel
or fusion family. A future compiler engineer or route integrator should be able
to reproduce the work from this packet without reading the whole session log.

```text
identity:
  production op/fusion:
  model and shape:
  reference backend and shader/kernel:
  workgroup count:
  workgroup size:
  subgroup/wave size:
  push constants / config values:
  binding layout:

oracle:
  reference source/SPIR-V/ISA:
  captured input tensors:
  expected output tensors:
  scalar or exact synthetic fixture:

loom workbench:
  .loom source:
  root symbol:
  check.case rows:
  check.benchmark rows:
  target key:

measurement:
  route-free benchmark command:
  benchmark result:
  compile report:
  target listing:
  timing warnings:

integration:
  catalog route:
  HRX2 trace proving selection:
  completion/PPL gate:
  same-session timing pair:
```

This packet is not ceremony. It is the guardrail that prevents the author from
optimizing a stale provider, the wrong route, a synthetic-only semantic, or a
measurement artifact.

## First Hour Procedure

Start with the production row rather than the source file.

1. Name the target row: model, prompt/decode shape, backend pair, exact device,
   timing boundary, and acceptance rule.

2. Trace the reference backend and HRX2 before editing. The HRX2 trace should
   include route id, root symbol, cache key, target key, workgroup geometry,
   provider cache hit/miss, and evidence artifact paths. The Vulkan side should
   give shader identity, push constants, SPIR-V, and RADV/ACO ISA when possible.

3. Reduce the route into a standalone `.loom` workbench. The workbench should
   have one parameterized kernel or fusion family with named benchmark rows for
   production shapes. Related specializations belong in the same file when they
   share an ABI and semantic contract.

4. Build correctness before performance. A full-output exact fixture is better
   than probes. A real captured tensor is better when exact synthetic values do
   not exercise the production rounding/indexing path.

5. Benchmark the same dispatch that the correctness case validated. Setup or
   reference kernels may exist in the check, but the timed benchmark row should
   isolate the candidate dispatch unless the product claim is a whole fused
   schedule.

6. Inspect compile report and target listing against a concrete hypothesis:
   unrolled fixed loops, wide LDS packets, subgroup reductions, split-K,
   occupancy, spills, waitcnt trains, bank conflicts, or instruction mix.

7. Integrate only after the route-free kernel is correct and has a reason to be
   fast. After integration, the first question is whether the model selected
   the intended provider, not whether wall time moved.

8. Run deterministic completion/PPL and same-session timing. If the route is
   correct but wall time regresses, keep the negative result. It is often more
   useful than a local microbenchmark win.

## Correctness Fixtures

The best fixture is the smallest one that proves the production semantic.

| Fixture type | Strength | Trap |
| --- | --- | --- |
| Zero output / no-op | Catches stale writes and shape holes | Broken math can still pass |
| Tiny scalar fixture | Easy to reason about | Often has the wrong f16/f32 accumulation policy |
| Exact synthetic full output | Great first promotion gate | Needs values chosen for crisp rounding |
| Captured real tensor | Production-faithful | Requires trace/extraction tooling |
| End-to-end text/PPL | Catches large integration corruption | Too coarse for kernel debugging |

For quantized GEMM, exact synthetic values were powerful. Example: q4 payload
bytes set to a known value, scale chosen so the f16acc result is exactly
representable, RHS filled with `1.0`, and the expected output filled with a
constant. That produced a full-output check without depending on a scalar
emulation of RADV's accumulation order.

For fusions, the check must prove the same I/O contract as the original graph
window. It is easy to write a mathematically plausible fusion that changes when
rounding happens, changes aliasing behavior, duplicates a producer, or collapses
parallelism that the split schedule relied on.

## Benchmarking Rules

Use `iree-benchmark-loom` as the kernel loop. It can run correctness and timing
from the same `.loom` file, emit compile reports, retain artifact bundles, and
produce JSON rows that another agent can query.

For per-dispatch claims, start with dispatch-complete timing and batch size 1:

```bash
build_tools/bin/iree-bazel-run //loom/src/loom/tools/iree-benchmark-loom -- \
  path/to/workbench.loom \
  --device=amdgpu \
  --benchmark=@candidate_bench \
  --measure=dispatch_complete \
  --iterations=50 \
  --warmup-iterations=10 \
  --batch-size=1 \
  --input-ring-count=1 \
  --compile-report=details \
  --artifact-bundle-dir=.notes/loom-dogfood/<work>/artifacts \
  --artifact-bundle-policy=debug \
  --output=.notes/loom-dogfood/<work>/results.json \
  --output-format=json
```

Then run a batch-size sweep when timing is near the noise floor or when the
reference measurement is throughput-like. During the dogfood run, batch-size 16
made some isolated kernels look much faster because completion cost was
amortized over multiple independent dispatches. That is useful for throughput
analysis and dangerous for per-model-dispatch parity claims.

Warnings are part of the result. A row with too few physical dispatches,
unstable p90/p50, an active benchmark lock conflict, or an unexpected debug
build state is diagnostic evidence, not a number to quote.

For serious local numbers:

- Build optimized binaries.
- Hold `~/.dotfiles/bin/benchmark-lock` when possible.
- Run HRX2 and Vulkan in the same machine state.
- Record device ordinals and physical GPU mapping.
- Compare ABABA-style when deltas are small.
- Keep profile/trace runs separate from no-profile timing claims.

## Compile Reports And Listings

The useful report read starts with a hypothesis.

Good pressure questions:

```text
Did the intended root compile, or did the benchmark summarize a helper?
Are there spills or scratch?
What limits occupancy: VGPRs, SGPRs, LDS, barriers, or launch shape?
Did a source change materialize copies or release them?
Did wide vector stores become wide LDS packets?
Did fixed-trip loops remain rolled when the schedule needs them explicit?
Did a fusion reduce dispatches while collapsing producer grid parallelism?
Did target resources move in the direction predicted by the source edit?
```

Strong patterns from this run:

- Literal fixed-trip inner loops should be made visible. The q4 prompt route
  knew it had two inner WMMA iterations; explicitly unrolling that literal loop
  reduced VGPR pressure and materialized copies enough to move model time.
- Contiguous f16 LDS staging should be authored as wide vector stores when the
  layout contract permits it. The q4/q5 wide staging edits reduced LDS packets,
  waits, code size, and pressure.
- Subgroup reductions should make leader-lane demand explicit when only one
  lane needs the result. The q4 Vulkan-clone matvec became cleaner only after
  bias and store moved under `%lane == 0`.
- Direct scale-byte loads beat LDS staging for the q6 decode result-output
  shape. More staging was not automatically better.
- Dynamic size specialization should be tile-count aware. Odd sizes inside the
  same tile were fine; crossing a tile boundary caused the visible cliff.

Strong negative patterns:

- Better static counts did not make add+RMS, RoPE+scale, or specialized always
  add variants faster.
- Fewer dispatches did not make softmax+KQV live-KV faster; the fusion serialized
  V dot products that the split graph kept decomposed.
- Route-free microbenchmarks did not always predict integration when graph
  aliasing, route gates, or provider selection changed.

## Fusions

A fusion is a schedule change, not a string concatenation of kernels.

The first question is what parallelism or memory traffic the boundary currently
has. Removing a dispatch can lose when the producer ran over many workgroups and
the consumer ran over one reduction workgroup. That was the add+RMS failure:
the fused kernel saved a launch but pulled a cheap global add into the narrower
RMS row-reduction grid.

A fusion packet should record:

```text
original graph window:
  nodes:
  tensors materialized:
  dispatch count:
  producer grid:
  consumer grid:
  memory traffic:
  aliasing:

candidate fused schedule:
  work ownership:
  repeated work:
  new live state:
  output rounding point:
  removed memory traffic:
  lost parallelism:
  expected win mechanism:
```

Positive fusions in this run had a clear preserved decomposition: q4/SwiGLU,
q5 V-cache, and q5 Q-scale-before-RoPE. Negative fusions usually looked good in
one scalar metric and bad in the schedule economics.

## Dynamic Shapes And JIT Specialization

The right Loom shape is one source with config-driven specialization, not one
manually duplicated file per size. Use `check.param.choice`, `scf.if`,
`scf.for`, helper functions, and forced inlining to express the family. Let the
JIT specialize per concrete shape when that shape is worth caching.

The dynamic odd-size experiments changed the intuition:

- Sub-workgroup odd sizes have masking overhead, but dispatch overhead dominates
  tiny isolated pointwise kernels.
- Row-tiled work is tile-count sensitive rather than power-of-two sensitive.
  `63` and `64` can be effectively the same; `64 -> 65` is the cliff because it
  doubles tiles per row.
- For normal llama prefill, token row count varies more often than hidden width.
  The expensive model widths were fixed clean multiples; the dynamic question is
  often how many token rows or KV rows are active.

This is exactly where Loom should be strong. The source remains generic, while
the runtime config gives the compiler exact values for launch geometry, masks,
and fixed-trip loops.

## Integration Traps

These failure modes consumed most of the lost time:

| Trap | Symptom | Stabilizing Mechanism |
| --- | --- | --- |
| Stale generated catalog | Source changed but route still old | Regenerate `catalog.json` before rebuild |
| Stale embedded artifacts | Runtime loads old bytecode | Rebuild `ggml-hrx2` and provider artifacts |
| Provider cache hit | Correct source exists but old executable runs | Trace cache key and compile/cache event |
| Wrong root selected | Report/listing belongs to another function | Use root-selected compile command |
| Route not selected | Kernel work has no model effect | Dispatch trace must show route id/cache key |
| CPU fallback | Wall time explodes or profiles lie | Scheduler trace with zero CPU compute fallback |
| Device ordinal mismatch | HRX2/Vulkan ratios become absurd | Pin and record visible devices |
| Debug/profiling perturbation | Trace run looks slower | Separate structural traces from timing rows |
| AQL/PM4 mode drift | Prefill/decode conclusions conflict | Record command mode and compare lanes separately |
| Barrier shortcuts | Run is fast but wrong | Completion/PPL gate after sync changes |

The practical fix is scripts, not memory. A run script should emit raw stdout,
stderr, JSON, env, git hash, dirty marker, route trace, scheduler trace, and
summary in one directory. Reconstructing benchmark flags from shell history was
one of the most expensive self-inflicted wounds.

## Compiler Engineer Handoff

The compiler engineer can move fast when the reproducer is route-free. A good
handoff includes:

```text
absolute .loom path:
root symbol:
target key and processor:
exact command:
expected behavior:
actual behavior:
check.case / benchmark name:
config values:
compile_report.json:
manifest.json:
target listing / disassembly:
input/output artifact if correctness-related:
smallest known failing variant:
larger production context:
```

The strongest reproducers were not "HRX2 is wrong." They were things like:

```text
This standalone .loom check fails on gfx1100 when a uniform branch has two arms
with different workgroup-reduction counts. Q rows are correct; K/V rows corrupt.
The same file has passing controls for each arm alone. Here is the root-selected
compile report and the expected/actual tensor mismatch.
```

That kind of report lets a compiler agent fix source-to-low, allocation,
sanitizer lowering, vector legalization, or report quality without installing
the llama.cpp experiment stack.

When a compiler bug blocks one kernel, file the reproducer and move to another
kernel family. The Phi run only progressed because compiler work was batched:
allocation diagnostics, range proofs, dot accumulators, f16 stores, ASAN,
subgroup reductions, and blocking codegen fixes all landed while kernel work
continued elsewhere.

## What To Teach Future Agents

The winning behavior was not that the author guessed better kernels. The winning
behavior was the discipline of converting every confusion into either a
route-free workbench, a branch snapshot, or a compiler reproducer.

The loop to teach is:

```text
start with the production row
extract the exact reference dispatch
write the check before trusting the benchmark
benchmark the same checked dispatch
read reports against a hypothesis
integrate only after the workbench earns it
prove the route selected
gate correctness at model level
time against the reference in the same session
preserve negative results
file standalone compiler reproducers
move to the next boulder while fixes land
```

This is the process that made Loom look good again. It is also the process our
tools should eventually make boring: trace-to-workbench generation, loud
artifact provenance, focused compiler diagnostics, compact benchmark summaries,
and a TLDR that tells an agent which next action is supported by the evidence.
