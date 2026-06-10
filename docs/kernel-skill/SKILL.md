---
name: hrx-hip-kernel-optimization
description: Use this when optimizing HRX pure-HIP GPU kernels, especially llama.cpp/ggml-hrx LLM inference kernels on AMD Radeon or ROCm. It covers profiler-first workflow, Tracy/IREE runtime profiling, rocprof/ATT/ISA kernel tooling, correctness gates, Vulkan comparison, wave32/wave64 decisions, and safe promotion of exact or approximate kernel routes.
metadata:
  short-description: Optimize HRX HIP kernels with profiler and correctness guardrails
---

# HRX HIP Kernel Optimization

This skill is an entrypoint. Load only the reference file needed for the task:

- General process and optimization catalog:
  `docs/kernel-skill/kernel_optimization_guide.md`
- AMD RDNA3 / wavefront / ISA gotchas:
  `docs/kernel-skill/amd_rdna3_wavefront_isa_gotchas.md`
- llama.cpp-specific provider, profiling, and correctness workflow:
  `docs/kernel-skill/llamacpp_optimization_skill.md`

## Non-Negotiables

- Do not optimize from wall tok/s alone. Use provider trace, Tracy plus
  `iree-profile` dispatch buckets, and correctness gates.
- Separate prefill from decode. They need different kernels.
- Do not run performance/profiling jobs in parallel when comparing candidates.
  Single-run noise is common; rerun before making a consequential route or
  kernel decision.
- Treat D2D copies, `CPY`/`CONT`/`CONCAT`, and CPU fallbacks as blockers until
  explained.
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
4. Read the matching section in `kernel_optimization_guide.md`.
5. If the task involves wavefront size, WMMA, dot, LDS, or spills, read
   `amd_rdna3_wavefront_isa_gotchas.md`.
6. Implement a narrow provider or source change with an opt-out or opt-in knob.
7. Run inner-loop correctness:
   `reproducers/qwen_hrx_inner_loop.sh`.
8. Capture Tracy plus `.ireeprof`; compare dispatch counts, export route, and
   dispatch time buckets before attributing a regression to a kernel.
9. Run the full milestone gate before promotion:
   `reproducers/qwen_hrx_correctness_gate.sh`.
10. Record accepted and rejected results in the analysis log.

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
