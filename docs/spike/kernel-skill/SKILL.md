---
name: hrx-hip-kernel-optimization
description: Use this when optimizing HRX pure-HIP GPU kernels, especially llama.cpp/ggml-hrx LLM inference kernels on AMD Radeon or ROCm. It covers profiler-first workflow, Tracy/IREE runtime profiling, rocprof/ATT/ISA kernel tooling, correctness gates, Vulkan comparison, wave32/wave64 decisions, and safe promotion of exact or approximate kernel routes.
metadata:
  short-description: Optimize HRX HIP kernels with profiler and correctness guardrails
---

# HRX HIP Kernel Optimization

This skill is an entrypoint. Load only the reference file needed for the task:

- General process and optimization catalog:
  `docs/spike/kernel-skill/kernel_optimization_guide.md`
- AMD RDNA3 / wavefront / ISA gotchas:
  `docs/spike/kernel-skill/amd_rdna3_wavefront_isa_gotchas.md`
- llama.cpp-specific provider, profiling, and correctness workflow:
  `docs/spike/kernel-skill/llamacpp_optimization_skill.md`

## Non-Negotiables

- Do not optimize from wall tok/s alone. Use provider trace, Tracy plus
  `iree-profile` dispatch buckets, and correctness gates.
- Separate prefill from decode. They need different kernels.
- Do not run performance/profiling jobs in parallel when comparing candidates.
  Single-run noise is common; rerun before making a consequential route or
  kernel decision.
- Treat D2D copies, `CPY`/`CONT`/`CONCAT`, and CPU fallbacks as blockers until
  explained.
- When isolating backend behavior, run focused backend op unit tests before
  full integration/model tests. Use integration tests after the op-level path is
  understood, not as the first debugging boundary.
- For Loom kernels, treat the source as WYSIWYG. Explicitly encode vector
  widths, packed load widths, tile shapes, address-range assumptions, and dot
  forms; do not expect the compiler to infer those choices from scalar-looking
  code. Use compile reports and ISA/resource checks to verify the spelling.
- Spell packed integer dot signedness from the mathematical data, not from the
  storage type. Q4_K codes are unsigned 4-bit values widened into i8 lanes and
  Q8_1 activations are signed i8 values, so Q4_K x Q8_1 dot products should be
  `vector.dot4i<u8s8>`. `s8s8` may compile and may be arithmetically identical
  for 0..15 inputs, but it is the wrong WYSIWYG contract and can select the
  wrong target form.
- For broad performance cliffs, especially prompt matmul, packed quantization,
  attention, and fusions, do prior-driven engineering before local tuning. Mine
  HRX1, Vulkan, CUDA/HIP, and other backend kernels; run, disassemble, or
  compile them when possible; compare actual schedule facts against the Loom
  candidate; then implement the missing algorithmic class. Do not burn time on
  knob tweaks when the current kernel is not in the same schedule family as a
  known-good prior.
- Prior search must leave a written schedule ledger before coding: source path
  or symbol, tile shape, wave/subgroup shape, lane ownership, per-lane outputs,
  vector/load widths, packed layout, dot primitive, staging/barriers, reduction
  and writeback policy, resource/ISA facts, and the shape regime where it won.
  A prior is not "used" until those facts are compared against the current
  candidate and the comparison is recorded.
- Before inventing a new schedule, generate analytical alternatives from the
  prior matrix: preserve the winning dataflow but vary one meaningful axis at a
  time, such as BM/BN/BK, WG size, wave32/wave64, vector width, Q8 packing,
  A/B staging, unroll depth, output ownership, or tail strategy. Do not jump
  from "Vulkan is faster" to a guessed Loom rewrite without this comparison.
- Adjacent schedule probes are allowed and useful, including probes without
  strong direct evidence, but only as bracketing around a documented schedule
  family. Define the pivot axis, bounds, and expected signal before coding,
  then run those variants in a focused kernel or backend-op benchmark sweep.
  Do not use full model integration runs as the first screen for speculative
  tile/vector/unroll variants, and do not promote them into the production
  catalog until the sweep selects a useful winner. Exploration is a kernel
  benchmark activity first; integration is the acceptance path after the sweep
  has produced evidence.
- For HRX2 prefill gaps of many multiples, aim at broad structural misses first.
  A useful kernel pass should be able to say which prior schedule it is
  matching, which emitted ISA/resource facts agree or differ, and what evidence
  remains if performance still does not move. Wall tok/s alone is not enough
  for that judgment.
- In the HRX2 Phase 2a goal loop, environmental failures are owned by the
  implementing agent. Fix stale builds, bad cache state, missing env wiring,
  benchmark harness issues, and runtime interop regressions before escalating.
  HRX1 runtime behavior is a primary reference for stream/submission/copy
  semantics.
- Before default-enabling a provider, run focused correctness and the full Qwen
  gate.
- For approximate prompt kernels, require rollback env vars, chat/loop guards,
  and a clear rationale tied to established GPU behavior.
- Verify final ISA/resource facts from built HSACOs when CMake uses per-source
  flags.

## Default Workflow

1. Establish a clean baseline:
   - HRX wall: `llama-bench`
   - HRX runtime/system: Tracy-enabled HRX plus `iree-profile`
   - HRX kernel/device time: `iree-profile` dispatch buckets
   - Vulkan reference: `GGML_VK_PERF_LOGGER=1`
   - Provider ownership: `GGML_HRX_TRACE_PROVIDERS=1`
2. Rank kernel families with `iree-profile dispatch --format=jsonl` grouped by
   export name.
3. Pick one hot family and one regime: prefill or decode.
4. Read the matching section in `kernel_optimization_guide.md` and the
   reusable prior-art ledger.
5. If the gap is large, produce a prior matrix before coding: backend/source,
   tile shape, lane ownership, per-lane outputs, vector/dot primitive, staging,
   packing, barriers, reduction and writeback policy, activation constraints,
   shape regime, and any ISA/resource evidence.
6. From that matrix, write the short list of analytical schedule alternatives
   to test and why each differs from the priors. Each implementation attempt
   should correspond to one row in that list, not to an ungrounded guess.
7. If the task involves wavefront size, WMMA, dot, LDS, or spills, read
   `amd_rdna3_wavefront_isa_gotchas.md`.
8. Implement a narrow provider or source change with an opt-out or opt-in knob.
9. Run focused backend op/unit correctness for the touched route before any
   model-level integration test.
10. Run inner-loop correctness:
   `reproducers/qwen_hrx_inner_loop.sh`.
11. Capture Tracy plus `.ireeprof`; compare dispatch counts, export route, and
   dispatch time buckets before attributing a regression to a kernel.
12. Run the full milestone gate before promotion:
   `reproducers/qwen_hrx_correctness_gate.sh`.
13. Record accepted and rejected results in the analysis log.

## Quick Commands

Environment:

```bash
export ROCM_PATH="$PWD/rocm"
export GGML_HRX_ROCM_PATH="$ROCM_PATH"
export HRX_RUNTIME_INSTALL="$PWD/build/hrx-rocm713-install"
export LLAMA_BUILD="$PWD/build/llama-hrx-rocm713"
export MODEL="$PWD/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$HRX_RUNTIME_INSTALL/lib64:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export GGML_HRX_KERNEL_PROVIDER=pure_hip
```

Build:

```bash
cmake --build build/llama-hrx-rocm713 \
  --target llama-bench llama-cli test-backend-hrx test-backend-ops export-graph-ops hrx-kernel-bench \
  -j"$(nproc)"
```

Provider trace:

```bash
GGML_HRX_TRACE_PROVIDERS=1 "$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 0 -n 8 -b 512 -ub 512 -fa 0 -r 1 \
  -o json --no-warmup -ngl 99 -dev HRX0 \
  > build/trace.json 2> build/trace.log
```

Fused Tracy + IREE profile smoke:

```bash
export HRX_RUNTIME_INSTALL="$PWD/build/hrx-rocm713-tracy-install"
export LD_LIBRARY_PATH="$HRX_RUNTIME_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export IREE_TRACY_CAPTURE="$PWD/build/iree-tracy-tools/tracy/iree-tracy-capture"
OUT=build/hrx-tracy-fused-smoke
mkdir -p "$OUT"
HRX_PROFILE_FILE="$OUT/run.ireeprof" \
  sources/iree/build_tools/tracing/iree_tracy_capture.py \
    --output-dir "$OUT" --name llama-p32n8 \
    -- "$LLAMA_BUILD/bin/llama-bench" \
      -m "$MODEL" -p 32 -n 8 -b 512 -ub 512 -r 1 \
      -o json --no-warmup -ngl 99 -dev HRX0 \
    > "$OUT/llama-bench.json"
build/iree-rt/tools/iree-profile summary "$OUT/run.ireeprof"
```

Use Tracy-enabled HRX with `IREE_TRACING_MODE=1` for scripted runtime captures.
Default `HRX_PROFILE_MODE=queue`; `dispatch`/`all` are opt-in until the new
AMDGPU profiling paths are correctness-clean for backend unit tests.

IREE profile prefill:

```bash
OUT=build/hrx-profile-prefill-p512-fa1
mkdir -p "$OUT"
HRX_PROFILE_FILE="$OUT/run.ireeprof" HRX_PROFILE_MODE=all \
  "$LLAMA_BUILD/bin/llama-bench" \
    -m "$MODEL" -p 512 -n 0 -fa 1 -b 2048 -ub 2048 -r 1 \
    -o json --no-warmup -ngl 99 -dev HRX0 \
  > "$OUT/llama-bench.json"
build/iree-rt/tools/iree-profile summary "$OUT/run.ireeprof"
build/iree-rt/tools/iree-profile dispatch --format=jsonl "$OUT/run.ireeprof" \
  > "$OUT/dispatch.jsonl"
```

For non-power-of-two prefill testing, set `-ub` high enough to keep the prompt
in one graph when the goal is kernel shape validation. For example, p513 with
the default `-ub 512` is a p512 graph plus a p1 residual graph, which is useful
for scheduler/microbatch behavior but not a pure p513 kernel-tail test. Use
`-b 2048 -ub 2048` for single-graph p513/p768/p1024 comparisons on the current
Qwen topology.

Inner-loop correctness:

```bash
FOCUS=q6 RESULTS_DIR=build/hrx-inner-loop-q6-candidate \
  reproducers/qwen_hrx_inner_loop.sh
```

Full correctness:

```bash
RESULTS_DIR=build/hrx-correctness-qwen-candidate \
CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1 \
LOOP_SEEDS=1,5 LOOP_PREDICT=384 \
reproducers/qwen_hrx_correctness_gate.sh
```

## Decision Rules

- If a tiny kernel is dominated by kernarg/scalar waits, fuse/eliminate it.
- If a K-quant prompt kernel is still F32-RHS scalar matvec-like, port the
  Vulkan-style packed Q8_1 matrix tile before doing local tweaks.
- If a decode kernel is `cols=1`, do not send it through prompt MMQ. Build a
  skinny DMMV or exact-shape provider.
- If a source change lowers VGPR but duplicates RHS reads, be skeptical.
- If a source change matches Vulkan visually but regresses device time, reject
  it and document the number.
- If a bucket is already ahead of Vulkan, guard it. Do not force a Vulkan-shaped
  rewrite into a winning path.
- If generation starts looping or producing question marks, suspect routing,
  approximate prompt math, or long-context hidden-state poisoning before
  declaring it a sampler issue.
- If a native HIP microbench shows a win but full-model `iree-profile` regresses
  the same route, reject the change until the discrepancy is explained.

## Current Known Priorities

### HRX2 Phase 2a checkpoint

For the current HRX2/Loom work in `sources/llama.cpp`, use Vulkan on the same
machine as the throughput baseline and `shared/models/llamacpp-hrx2-basket-v1`
as the model basket. The fresh reduced baseline artifact after the accepted F16
batched-attention route and accepted Phi V-cache `CONT_SET_ROWS` fusion is:

```text
cache/hrx2/phase2a/current-reduced-after-cont-setrows-20260615-235800/
```

Decode is currently about 0.33x-0.39x Vulkan with zero CPU compute fallback.
Prefill is still the bulk blocker at about 0.016x-0.05x Vulkan. Treat this as a
structural problem until evidence says otherwise. The current top boulders are:

- Q4_K prompt matmul schedule.
- Q5/Q6 prompt matmul routes.
- Attention-chain/fusion candidates.

The F16 batched-attention cols8 route is already committed in llama.cpp at
`37d5417ab hrx2: add p512 f16 attention cols8 route`. It is deliberately
limited to the c512/p512 domain because p64/narrow prompt smoke regressed.
Treat it as a small accepted lift, not as the remaining bulk prefill answer.

The Phi V-cache `CONT -> SET_ROWS` fusion is already committed in llama.cpp at
`bf1c90d8d hrx2: fuse cont into v-cache set rows`. It adds route
`cont_set_rows_f32_f16_n128_wg256`, trace op `CONT_SET_ROWS`, and rollback
`GGML_HRX2_DISABLE_CONT_SET_ROWS_FUSION=1`. Same-binary A/B improved Phi
prefill by about 2-3% and reduced dispatch count, so treat it as accepted. Do
not remove its large-KV guard (`set_rows.ne1 > 1048576`) without widening and
testing the Loom config ranges; a default-cache CLI smoke previously exposed a
provider config failure for huge `SET_ROWS` shapes.

The Q4_K Q8_1/x4 MMQ path is opt-in and correctness-failing. The latest
narrowing removed NaNs by making Q8_1 `d/s` metadata single-writer and f32 in
LDS, but the route still fails focused Q4_K rows with finite mismatch around
`ERR ~= 1.0`. The non-x4 Q8_1 cols4 route passes the same rows, so do not blame
the generic Q4_K/Q8_1 plumbing. Do not benchmark or promote x4 until focused
`test-backend-ops` rows pass. The next serious Q4_K attempt should be a clean
HRX1/Vulkan-style tiled MMQ schedule with packed Q8_1 RHS, cooperative A/B
staging, dot4 inner loops, and multi-output lane ownership, or a tiny
diagnostic consumer that precisely proves the current x4 layout bug.

For the current HRX2 Phase 2a prefill gap, use the fresh reduced basket
artifact `cache/hrx2/phase2a/current-reduced-after-cont-setrows-20260615-235800/`
as the recent checkpoint. Decode is roughly one third of Vulkan with no CPU
compute fallback, while prefill is still only about 0.016x-0.05x Vulkan and the
final stream sync is carrying real device work. This means kernel/fusion
quality is the broad issue, not just launch overhead. Prioritize Q4_K prompt
matmul, Q5/Q6 prompt matmuls, and attention-chain/fusion candidates. For each
candidate, first extract a known-good schedule from HRX1, Vulkan, CUDA/HIP, or
emitted ISA; then spell the missing vector widths, load widths, dot forms,
staging, tile shape, unroll policy, and bounds handling in Loom.

The latest fresh reduced rerun is
`cache/hrx2/phase2a/current-reduced-20260616-000512/`. It reconfirmed zero CPU
compute fallback, decode around one third of Vulkan, and prefill around
0.016x-0.05x Vulkan. If this differs from a newer artifact, prefer the newer
same-run HRX2/Vulkan basket and update this note.

Current next boulders: Q4_K/Q5_K/Q6_K prompt matmul quality and attention-chain
fusions. For Q4_K, use either a standalone Loom/low-level integer-LDS
reproducer for the Loom author, or a different staging spelling that can match
the HRX1/Vulkan tiled packed-MMQ schedule. Do not continue local knob sweeps on
the known BM64/BN8 Loom source until the integer-LDS correctness issue is
explained.

For attention-chain work, remember that the current basket baseline uses
`--flash-attn 0` and therefore exposes unfused KQ matmul, SOFT_MAX, and KQV
matmul routes. Before writing another small attention matmul tweak, check
whether a `fa1-prefill-probe-*` artifact exists and compare HRX2/Vulkan
`--flash-attn 1` behavior. If Vulkan moves to a much better flash-attention path
while HRX2 falls back or stays unfused, the right boulder is HRX2
`GGML_OP_FLASH_ATTN_EXT` support or an equivalent fusion, starting from HRX1
flash-attention kernels and Vulkan flash-attention shaders.

When adding prompt-specialized routes, keep the shape domain tight until the
data proves it generalizes. The F16 batched attention cols8 route is a current
example: c512 backend-op gates passed and p512 smoke improved modestly, but
p64 smoke regressed, so the route must stay p512/cols512-only unless further
tuning produces a separate narrow-shape route.

After the April 21 decode-final grind, the best current W7900/Qwen decode
candidate is around the 115 tok/s target in untraced release runs, with
correctness gates passing. Do not assume this means all kernel work is done:
the remaining decode work is now small-percent kernel quality, route quality,
and real dispatch elimination.

Current grind order:

1. Q6_K decode and Q6_K SILU/MUL. Exact-K native probes looked promising but
   regressed in full-model profile, so the next pass needs ATT/ISA evidence,
   not another selector-only route.
2. Q5_K decode matvec. Current `q5_wg32` still beat obvious rows2/rows4/dot16
   native variants at both normal and giant-row decode shapes; compare packing
   and waits against Vulkan/CUDA before changing it.
3. Q4_K MoE packed paths. Current decode routes remain locally best:
   packed WG32 for SWIGLU and rows2_x16 WG32 for Q4 MUL. Further work should
   be a packed-kernel true-up, not route roulette.
4. BF16 SWIGLU/dense tail and TopK/MoE. These are smaller buckets but still
   plausible sources of the last few percent.
5. Dispatch elimination audit. Remove real materializations, copies, or tiny
   dispatches; do not use submission-boundary changes as correctness or
   stability fixes.
