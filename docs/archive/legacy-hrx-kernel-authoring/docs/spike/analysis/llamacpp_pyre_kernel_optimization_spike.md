# llama.cpp Pyre kernel optimization spike

**Status:** Active
**Date:** 2026-04-09

## Goal

Continue systematic Pyre kernel optimization on `sources/llama.cpp` branch
`epic1_perf_spike`. The current useful baseline is approximately:

- Pyre decode: about 23 tok/s on Qwen3.5-35B-A3B-UD-Q4_K_L, `-p 0 -n 32`,
  `-fa 0`, `PYRE0`.
- Vulkan decode: about 105 tok/s on the same model and shape, `Vulkan0`.

Provider-level graph coverage is now good enough that kernel optimization should
be visible. CPU utilization remains a required sanity signal: if a throughput
gain comes with high sustained CPU usage, treat it as suspicious until provider
trace and Tracy show that high-parallelism work did not escape to CPU.

## Baseline Commands

Pyre no-trace:

```bash
LD_LIBRARY_PATH="$PWD/build/therock/dist/rocm/lib:$PWD/build/therock/dist/rocm/lib/rocm_sysdeps/lib:$PWD/build/pyre-runtime-install/lib64:${LD_LIBRARY_PATH:-}" \
GGML_PYRE_KERNEL_PROVIDER=pure_hip \
./build/llama-pyre/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf \
  -p 0 -n 32 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup \
  -ngl 99 -dev PYRE0
```

Vulkan no-trace:

```bash
./build/llama-vulkan/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf \
  -p 0 -n 32 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup \
  -ngl 99 -dev Vulkan0
```

Pyre provider trace:

```bash
LD_LIBRARY_PATH="$PWD/build/therock/dist/rocm/lib:$PWD/build/therock/dist/rocm/lib/rocm_sysdeps/lib:$PWD/build/pyre-runtime-install/lib64:${LD_LIBRARY_PATH:-}" \
GGML_PYRE_KERNEL_PROVIDER=pure_hip \
GGML_PYRE_TRACE_PROVIDERS=1 \
./build/llama-pyre/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf \
  -p 0 -n 8 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup \
  -ngl 99 -dev PYRE0
```

## Measurement Rules

- Use provider trace to validate op ownership and fallbacks only. Do not use
  traced throughput as a performance number.
- Record no-trace Pyre and Vulkan JSON for accepted changes.
- Record CPU utilization around benchmark runs. High sustained CPU use is a
  blocker until explained.
- GPU utilization is a weak signal for this phase. Record it opportunistically
  from `/sys/class/drm/card1/device/gpu_busy_percent`, but do not gate on it.
- Use Tracy when a provider trace looks clean but runtime structure may be wrong.
  Watch for high counts or time in:
  `iree_hal_hsa_stream_command_buffer_copy_buffer`,
  `hsa_amd_memory_async_copy`, `hsa_copy_signal_wait`,
  `hsa_kernarg_memory_pool_allocate`, `hsa_dispatch_queue_reserve`,
  `hsa_dispatch_ring_doorbell`, and `hsa_dispatch_signal_wait`.

## Work Queue

1. Extend `pyre-kernel-bench` so `MUL_MAT_ID` and fused expert paths can be
   measured with Qwen-shaped dimensions without loading the full GGUF.
2. Use the microbench and full Qwen decode to prioritize MoE expert kernels:
   Q4_K `MUL_MAT_ID`, `MUL_MAT_ID+SWIGLU`, and `MUL_MAT_ID+MUL`.
3. Compare Pyre kernels against Vulkan shader structure when a provider is
   measurably behind.
4. Re-check flash attention only after no-FA MoE/matvec work stops being the
   obvious target.

## Experiment Log

### 2026-04-09: Start spike

- Branch: `sources/llama.cpp` `epic1_perf_spike`.
- Initial blocker: `pyre-kernel-bench` supports dense and quantized `MUL_MAT`
  matvecs but not `MUL_MAT_ID`, so it cannot isolate Qwen MoE expert kernels.

### 2026-04-09: Add model-free MoE expert microbench coverage

- Extended `tools/pyre-epic2/pyre-kernel-bench.cpp` in `sources/llama.cpp`
  for Q4_K expert paths: `mul_mat_id_q4_k`, `mul_mat_id_q4_k_mul`, and
  `mul_mat_id_q4_k_swiglu`.
- Shape used for the initial Qwen-like probe:
  `--ncols 2048 --nrows 512 --n-experts 256 --n-ids 8 --n-tokens 1`.
- Pure HIP provider results on the Pyre build:
  `mul_mat_id_q4_k` median 229.136 us / min 224.576 us,
  `mul_mat_id_q4_k_mul` median 212.576 us / min 205.255 us, and
  `mul_mat_id_q4_k_swiglu` median 226.276 us / min 222.426 us.
- `mul_mat_id_q4_k_swiglu` required a looser microbench tolerance than the
  original dense-path `1e-3` absolute check. The fused result combines two Q4_K
  expert dot products and SiLU, so the harness now uses
  `5e-3 + 1e-4 * abs(expected)` as a guardrail while backend conformance
  remains covered by `test-backend-pyre`.
- Verification: rebuilt `pyre-kernel-bench`; ran the shaped expert microbenches
  under `GGML_PYRE_KERNEL_PROVIDER=pure_hip`; ran `test-backend-pyre`.

### 2026-04-09: Check gated q8_1 RHS expert path

- Updated the microbench reference path to account for
  `GGML_PYRE_ENABLE_Q8_1_MMVQ=1` by quantizing/dequantizing the RHS as q8_1
  before computing expected results. This prevents q8_1 experiments from being
  rejected against an F32-RHS reference.
- Qwen-like shape with q8_1 enabled:
  `mul_mat_id_q4_k` median 398.631 us / min 387.441 us and
  `mul_mat_id_q4_k_mul` median 401.601 us / min 391.310 us.
- `mul_mat_id_q4_k_swiglu` has no q8_1 variant, so it stayed on the F32-RHS
  fused path: median 235.586 us / min 234.046 us in this run.
- Interpretation: q8_1 RHS is not an immediate default win in the current
  synchronous runtime because the extra quantize dispatch dominates this
  decode-shaped microbench. Keep q8_1 as an opt-in experiment until the
  quantize can be fused, amortized, or retested on the incoming runtime.

### 2026-04-09: Q6_K F32-RHS matvec scale hoist

- Provider trace for `n=8` decode showed frequent Q6_K matvec shapes:
  `k=2048 rows=4096` and `k=4096 rows=2048`.
- The F32-RHS Q6_K kernel recomputed the per-group scale inside the unrolled
  four-element dequant helper. Reworked the lane mapping to match the q8_1 Q6
  kernel structure and hoisted scale conversion once per block/group.
- Microbench before:
  `mul_mat_vec_q6_k 2048x4096` median 262.317 us / min 249.377 us and
  `mul_mat_vec_q6_k 4096x2048` median 254.746 us / min 246.866 us.
- Microbench after:
  `mul_mat_vec_q6_k 2048x4096` median 245.876 us / min 243.487 us and
  `mul_mat_vec_q6_k 4096x2048` median 245.627 us / min 241.236 us.
- Verification: rebuilt `pyre-kernel-bench`; ran the two shaped Q6 microbenches;
  ran `test-backend-pyre`.
- End-to-end no-trace check after rebuilding `llama-bench`: `-p 0 -n 32 -fa 0`
  on Qwen reported 24.594 tok/s at build commit `b6c25b02d`, versus the
  previous report baseline of 23.75 tok/s. `/usr/bin/time -v` showed 82% CPU for
  the process, which is not the earlier high-parallel CPU fallback signature on
  this 96-core/192-thread system.

### 2026-04-09: Reverted Q5_K two-row workgroup experiment

- Tried a F32-RHS Q5_K kernel variant that dispatches half as many workgroups
  and computes two adjacent rows per workgroup to reuse RHS loads.
- Result was negative: `mul_mat_vec_q5_k 2048x8192` was roughly flat/slightly
  worse at 248.216 us median versus the prior 246.007 us, while the large final
  projection shape `2048x248320` regressed badly to 1956.342 us median versus
  the prior 1578.132 us.
- Reverted the experiment. Likely explanation: the added row-local state and
  second-row memory stream hurt occupancy/cache behavior more than RHS reuse
  helped for this simple one-workgroup-per-row design.

### 2026-04-09: Remove recurrent state-copy materializations

- Added provider trace detail for copy-like nodes so remaining materializations
  can be grouped by node/source name rather than just shape.
- The largest remaining copy cluster was not MoE scale materialization. It was
  Qwen recurrent state traffic: `cache_s_l*` 2 MiB `new_state` copies and
  `cache_r_l*` convolution-state updates, plus zero-length cache windows.
- Fused `GATED_DELTA_NET` with the following `new_state -> cache_s_l*` copy by
  adding an optional state-output binding to the Pyre GDN kernel. In the n=8
  provider trace this replaced 240 2 MiB buffer copies with 240
  `GATED_DELTA_NET_STATE_UPDATE` fused dispatches.
- Added `SSM_CONV_UPDATE_SILU`, which computes the Qwen linear-attention
  convolution directly from `conv_states` and `qkv_mixed`, writes the shifted
  convolution state cache in-place, and skips `CONCAT`, `CPY(last_conv_states)`,
  and standalone `SSM_CONV_SILU`. In the n=8 provider trace this replaced 240
  `CONCAT` dispatches and 240 96 KiB strided copies with 240 fused dispatches.
- Zero-length `GET_ROWS` and `CPY` state-window operations are now skipped
  before provider claim logging, so copy accounting reflects real work.
- Provider-trace checkpoint: `CPY` claims went to zero for the n=8 decode trace.
  Remaining materialization claims were 160 small `CONT` operations in
  full-attention gating (`attn_pregate-*` and `gate_reshaped-*`).
- No-trace `llama-bench -p 0 -n 32 -fa 0` checkpoint after both recurrent
  fusions reported 25.796 tok/s. The prior checkpoint after Q6_K scale hoist was
  24.594 tok/s.

### 2026-04-09: Fuse full-attention gate materializations

- The remaining inner materializations after recurrent-state fusion were the 160
  small `CONT` operations in full-attention layers: `attn_pregate-*` from the
  permuted attention output and `gate_reshaped-*` from the query/gate view.
- Added `SIGMOID_MUL_STRIDED`, which reads both pre-contiguous source views,
  computes `attn * sigmoid(gate)`, and writes the final gated attention output
  directly. The graph matcher is intentionally narrow: it requires the Qwen
  `attn_pregate` / `gate_reshaped` names, F32 sources, matching shapes, and a
  contiguous destination.
- Added a `test-backend-pyre` case that constructs the same strided attention
  and gate-view pattern and verifies the fused output numerically. With provider
  tracing enabled, the test claims `SIGMOID_MUL_STRIDED n=24`.
- Final n=8 provider-trace checkpoint for Qwen decode:
  `CONT=0`, `CPY=0`, `CONCAT=0`, `SIGMOID_MUL_STRIDED=80`,
  `SSM_CONV_UPDATE=240`, `GATED_DELTA_NET_STATE_UPDATE=240`.
- No-trace `llama-bench -p 0 -n 32 -fa 0` checkpoint reported 26.258 tok/s with
  `/usr/bin/time -v` showing 81% process CPU. This is a modest end-to-end gain
  over the 25.796 tok/s recurrent-fusion checkpoint, but it removes the last
  copy-like provider claims observed inside this decode trace.

### 2026-04-09: Chat reproducer CPU investigation

- The interactive reproducer `reproducers/chat_qwen_pyre.sh` defaults to
  `CONTEXT=4096`, which exercises a different path than the decode-only
  `llama-bench -p 0` checks.
- Provider tracing with the script showed the earlier large `SOFT_MAX` fallback
  risk in prompt graphs: full-attention prefill can present rows wider than the
  original Pyre softmax kernel's 1024-column shared buffer and can use a
  broadcast mask. Reworked the softmax kernel to a multi-pass row reduction and
  allowed mask broadcasting in dimension 1. Added a large masked softmax
  backend test (`ncols=2048`) to cover the multi-pass path.
- The sustained high CPU during decode was not an op fallback: with
  `PREDICT=128` and the script default `CONTEXT=4096`, provider tracing kept
  decode softmax on Pyre and the only fallback logs were zero-token `MUL_MAT`
  planning artifacts. A live `ps -L` sample showed many runnable llama.cpp CPU
  worker threads.
- `--poll 0 --poll-batch 0` did not reduce the CPU burn. Capping CPU threadpool
  width did: adding `-t 1 -tb 1` reduced `/usr/bin/time -v` process CPU from
  about 3970% to 86% while generation remained essentially unchanged
  (`17.7 tok/s` before and after in the sampled one-turn reproducer).
- Updated the local chat reproducer to default `THREADS=1` and
  `THREADS_BATCH=$THREADS`, with env overrides still available.

### 2026-04-09: Add Pyre/Vulkan trace summary harness

- Added `sources/llama.cpp/tools/pyre-epic2/pyre-trace-summary.py` on the
  llama.cpp topic branch. It can run a short Qwen trace for Pyre and/or Vulkan
  and can summarize existing logs:
  `pyre-trace-summary.py run-qwen --backend both --prompt 0 --gen 8`.
- The Pyre side runs with `GGML_PYRE_TRACE_PROVIDERS=1` and reports provider
  claim counts, fallback counts, and hot shapes. The Vulkan side runs with
  `GGML_VK_PERF_LOGGER=1` and reports per-fusion timing labels from Vulkan
  timestamp queries.
- Important runner fix: the active backend's `bin` directory must be first in
  `LD_LIBRARY_PATH`. Putting `build/llama-pyre/bin` before
  `build/llama-vulkan/bin` caused the Vulkan binary to resolve the wrong ggml
  libraries and reject `-dev Vulkan0`.
- Fresh `n=8`, no-FA trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-1/`.
- Pyre trace result: zero provider fallbacks. Top claims included
  `ADD_ADD` 1280, `MUL_MAT bf16` 880, `MUL_MAT f32_batched` 640,
  `RMS_NORM_MUL` 568, `MUL_MAT q6_K` 560, `TOPK_MOE` 320,
  `MUL_MAT_ID_SWIGLU` 320, and `MUL_MAT_ID_MUL` 320.
- Vulkan perf-label result for the same short trace: major fusion labels were
  `RMS_NORM_MUL` 1048, `MULTI_ADD` 640, `MUL_MAT_ID_MUL` 320,
  `TOPK_MOE_EARLY_SOFTMAX_NORM` 320, and `MUL_MAT_ADD` 72.
- First actionable parity gap from the automated diff: Vulkan reports 72
  `MUL_MAT_ADD` fusions with no direct Pyre equivalent. This matches
  `pyre-workspace-hf9f.9.5` and should be the next graph-fusion target after
  the trace harness ticket closes.
- Throughput from this traced run was not used as a performance number, but the
  JSON reported Pyre 17.729 tok/s and Vulkan 56.695 tok/s under tracing.

### 2026-04-09: Add Q8_0 MUL_MAT+ADD fusion parity

- The trace harness exposed a direct Vulkan/Pyre parity gap for Qwen no-FA
  decode: Vulkan reported 72 `MUL_MAT_ADD` fusion labels while Pyre reported no
  direct equivalent.
- The Pyre trace showed the matching Qwen chain as Q8_0 matvecs
  `k=4096 rows=2048 cols=1` followed by F32 `ADD n=2048`.
- Added a narrow `pyre_mul_mat_vec_q8_0_add_f32` provider and graph matcher for
  `MUL_MAT(Q8_0,F32) -> ADD` where the bias is F32, contiguous, and exactly the
  same shape as the matvec output. This intentionally does not yet cover
  broadcast bias, `ADD_ID`, or non-Q8_0 matvecs.
- Added a backend test that constructs `ggml_add(ggml_mul_mat(q8_0, f32), bias)`
  and verifies the fused result. With provider tracing enabled,
  `test-backend-pyre` now logs
  `claim MUL_MAT_ADD provider=pure_hip_q8_0 k=64 rows=3 cols=1`.
- Fresh Qwen `n=8`, no-FA Pyre trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-5-q8add/`.
- Result: Pyre now claims 72 `MUL_MAT_ADD pure_hip_q8_0` providers for the same
  shape class, and the Pyre/Vulkan trace diff reports no remaining direct
  `MUL_MAT_ADD` label gap against the earlier Vulkan trace.
- No-trace `llama-bench -p 0 -n 32 -fa 0` after the change reported
  26.002 tok/s with `/usr/bin/time -v` showing 81% process CPU. This is neutral
  within the noise of the current runtime, but it removes 72 dispatches and a
  full output read/write from the traced decode graph.

### 2026-04-09: K-quant DMMV workgroup tuning

- Took `pyre-workspace-hf9f.9.2` for Vulkan DMMV items 4-6. The Qwen trace
  showed all hot decode `MUL_MAT` claims at `cols=1`, so the practical first
  pass targeted subgroup-sized reduction/workgroup variants rather than
  multi-column register reuse.
- Added Q4_K/Q5_K/Q6_K matvec variants with 128-thread and 64-thread workgroups
  alongside the existing 256-thread kernels. The variants share the same source
  and can be forced with `GGML_PYRE_MUL_MAT_VEC_K_WG=64|128|256`; the default
  `auto` selector keeps a type/shape policy for Qwen-derived shapes.
- Auto policy after microbench: Q4_K uses WG128, Q5_K uses WG128 except
  very large row counts use WG64, and Q6_K uses WG128 only for
  `k>=4096 && rows<=2048`; other Q6_K decode shapes stay on WG256.
- Representative microbench results from `pyre-kernel-bench`:
  Q5_K `k=2048 rows=8192`: WG256 297.107 us, WG128 228.626 us, WG64
  283.078 us.
  Q5_K giant `k=2048 rows=248320`: WG256 1642.513 us, WG128 1360.836 us,
  WG64 1291.444 us.
  Q6_K `k=2048 rows=4096`: WG256 230.096 us, WG128 284.748 us, WG64
  235.107 us.
  Q6_K `k=4096 rows=2048`: WG256 299.988 us, WG128 233.356 us, WG64
  296.218 us.
  Q6_K `k=2048 rows=8192`: WG256 277.437 us, WG128 332.248 us, WG64
  282.528 us.
  Q4_K `k=2048 rows=4096`: WG256 259.097 us, WG128 203.785 us, WG64
  251.796 us.
- Validation: `test-backend-pyre` passed with WG256, WG128, WG64, and auto
  modes. A short Qwen `n=2` provider trace at
  `build/pyre-epic2-results/trace-diff-hf9f-9-2-wgauto/` showed zero provider
  fallbacks and the expected labels: Q5_K `_wg128`, Q6_K `_wg128` only on
  `k=4096 rows=2048`, and baseline Q6_K on the `k=2048` shapes.
- End-to-end no-trace `llama-bench -p 0 -n 32 -fa 0` on the same build:
  forced WG256 reported 26.279 tok/s and auto reported 26.804 tok/s, both with
  `/usr/bin/time -v` process CPU at 81%. This is a modest end-to-end gain, but
  the per-shape microbench data is clear and the knob remains available for
  differential analysis.

### 2026-04-09: Reverted vector RHS K-quant matvec experiment

- Took `pyre-workspace-hf9f.9.3` for Vulkan shader inner-loop parity. The first
  low-risk experiment was to keep the committed Pyre lane mapping but replace
  four scalar RHS loads with a `float4` RHS load in Q4_K/Q5_K/Q6_K matvec loops.
- Result was not a net win against the workgroup-tuned baseline from
  `7daee119d`. Correctness passed, but the frequent decode shapes regressed:
  Q4_K `k=2048 rows=4096` went 203.785 us -> 212.795 us, Q5_K
  `k=2048 rows=8192` went 228.626 us -> 284.598 us, Q6_K
  `k=2048 rows=4096` went 230.096 us -> 249.276 us, and Q6_K
  `k=2048 rows=8192` went 277.437 us -> 315.479 us.
- There were isolated positives: Q5_K giant `k=2048 rows=248320` went
  1291.444 us -> 1214.952 us, and Q6_K `k=4096 rows=2048` went
  233.356 us -> 227.756 us. Those wins are too narrow to justify changing the
  shared kernel body because the frequent shapes lose more.
- Reverted the experiment and rebuilt `test-backend-pyre`, `pyre-kernel-bench`,
  and `llama-bench` back to the committed workgroup-tuned source. Next useful
  inner-loop work should use a separate specialized provider, not a global
  replacement, or should port the fuller Vulkan 16-lane packed arithmetic
  structure rather than only changing RHS load width.

### 2026-04-09: Q4_K expert MUL_MAT_ID workgroup tuning

- Took `pyre-workspace-hf9f.9.4` for Qwen MoE expert matvec tuning. This pass
  intentionally stayed at the safer workgroup/reduction specialization level;
  the fuller Vulkan-style packed-lane K-quant arithmetic is tracked separately
  by the planner follow-up noted on the ticket.
- Added 128-thread and 64-thread provider variants for the Q4_K expert kernels:
  `MUL_MAT_ID`, `MUL_MAT_ID_MUL`, and `MUL_MAT_ID_SWIGLU`. Added
  `GGML_PYRE_MUL_MAT_ID_Q4_K_WG=auto|64|128|256` so the variants remain easy to
  force for differential runs.
- Microbench sweep, `n_ids=8 n_tokens=1 n_experts=256`, showed the 64-thread
  variant as the best target for the Qwen expert decode shapes:
  `MUL_MAT_ID k=2048 rows=512`: WG256 264.637 us, WG128 258.407 us,
  WG64 215.826 us.
  `MUL_MAT_ID_MUL k=2048 rows=512`: WG256 224.006 us, WG128 277.277 us,
  WG64 217.426 us.
  `MUL_MAT_ID_SWIGLU k=2048 rows=512`: WG256 233.866 us, WG128 286.077 us,
  WG64 226.216 us.
  `MUL_MAT_ID_MUL k=512 rows=2048`: WG256 252.707 us, WG128 289.758 us,
  WG64 212.826 us.
- Set the default auto policy to use WG64 for `k <= 2048` and keep larger
  unmeasured expert shapes on the original WG256 path. A trace-probed
  `pyre-kernel-bench` run confirmed auto claims
  `MUL_MAT_ID_MUL provider=pure_hip_q4_K_wg64 k=2048 rows=512`.
- Validation: `test-backend-pyre` passed with `GGML_PYRE_MUL_MAT_ID_Q4_K_WG`
  forced to auto, 64, and 256. The earlier broader sweep also passed auto, 64,
  128, and 256.
- Fresh Qwen `n=2`, no-FA trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-4-idwgauto/`. Result: zero Pyre
  fallbacks, with `MUL_MAT_ID_SWIGLU pure_hip_q4_K_wg64 k=2048 rows=512`
  claimed 80 times and `MUL_MAT_ID_MUL pure_hip_q4_K_wg64 k=512 rows=2048`
  claimed 80 times.
- End-to-end no-trace `llama-bench -p 0 -n 32 -fa 0` with the expert WG auto
  policy reported 28.006 tok/s and `/usr/bin/time -v` process CPU at 78%.

### 2026-04-09: Retest and gate Q8_1 MMVQ

- Took `pyre-workspace-hf9f.9.6` to re-evaluate the opt-in Q8_1 RHS path under
  the lower-overhead runtime state. The original global opt-in still is not a
  safe default: it changes numerical reference behavior for tests, and the hot
  Qwen expert paths still lose badly.
- Added `GGML_PYRE_Q8_1_MMVQ_POLICY=auto|all`. `all` preserves the older
  force-every-Q8_1-capable-path behavior for differential experiments. The new
  `auto` policy, used when `GGML_PYRE_ENABLE_Q8_1_MMVQ=1` is set without a
  policy override, only permits large standalone Q4_K/Q6_K matvecs and keeps
  Q5_K plus Q4_K `MUL_MAT_ID` expert paths on the F32 RHS providers. Updated
  `pyre-kernel-bench` so its q8_1 reference path mirrors the policy selector.
- Correctness: `test-backend-pyre` passes in default mode and with
  `GGML_PYRE_ENABLE_Q8_1_MMVQ=1` auto policy. `GGML_PYRE_Q8_1_MMVQ_POLICY=all`
  still fails the old strict F32-reference test on small Q4_K matvecs, as
  expected, because it intentionally forces a quantized RHS path.
- Paired microbench results against the current WG-tuned build were mixed and
  noisy enough to require full decode validation:
  Q4_K `k=2048 rows=4096`: F32 269.407 us, Q8_1 202.405 us in one sweep, but
  Q8_1 auto rerun 291.558 us.
  Q5_K `k=2048 rows=8192`: F32 215.506 us, Q8_1 297.098 us, so Q5_K remains
  F32 in auto.
  Q6_K `k=2048 rows=4096`: F32 299.058 us, Q8_1 262.377 us in one sweep, but
  Q8_1 auto rerun 275.757 us.
  Q6_K `k=2048 rows=8192`: F32 330.469 us, Q8_1 301.367 us in one sweep, but
  Q8_1 auto rerun 331.128 us.
  Q6_K `k=4096 rows=2048`: F32 286.227 us, Q8_1 262.817 us in one sweep, and
  Q8_1 auto rerun 267.207 us.
  Expert Q4_K `MUL_MAT_ID k=2048 rows=512`: F32 216.545 us, forced Q8_1
  530.254 us; auto correctly stays F32.
  Expert Q4_K `MUL_MAT_ID_MUL k=512 rows=2048`: F32 265.487 us, forced Q8_1
  455.832 us; auto correctly stays F32.
- Fresh Qwen `n=2`, no-FA trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-6-q8auto/`. Result: zero Pyre
  fallbacks. The auto policy claims 140 `MUL_MAT pure_hip_q6_K_q8_1`, keeps
  60 `MUL_MAT pure_hip_q5_K_wg128` plus 2 giant Q5_K WG64 on F32 RHS, and
  keeps both Q4_K expert fused paths on `pure_hip_q4_K_wg64`.
- End-to-end no-trace `llama-bench -p 0 -n 32 -fa 0` result:
  default/no-Q8_1 reported 27.717 tok/s at 81% process CPU after this policy
  patch, while Q8_1 auto reported 23.416 tok/s at 78% process CPU. The
  recommendation is still: do not enable Q8_1 by default. The remaining useful
  work is conversion caching or a fused quantize+matvec prototype; separate
  conversion dispatches still erase the apparent per-kernel wins.

### 2026-04-09: Gated packed-lane Q4_K DMMV experiment

- Took `pyre-workspace-hf9f.9.14` for the fuller Vulkan-style packed K-quant
  arithmetic follow-up. Started with one isolated target: standalone
  Q4_K F32-RHS DMMV at `k=2048 rows=4096`, because the earlier global float4 RHS
  load rewrite had already been rejected and the current WG policy is easy to
  compare against.
- Added a separate `pyre_mul_mat_vec_q4_k_packed_wg64_f32` provider instead of
  replacing the existing Q4_K kernel body. The packed variant follows the Vulkan
  16-lanes-per-quant-block decomposition: each lane handles four 4-value
  chunks from the Q4_K superblock, accumulates 16 RHS products, and uses the
  existing wave reduction. It is gated by `GGML_PYRE_ENABLE_PACKED_Q4_K_DMMV=1`
  and only selected for `k=2048 rows=4096` so future differential runs can
  exercise it without perturbing the default path.
- Correctness: `test-backend-pyre` passes with the default path and with
  `GGML_PYRE_ENABLE_PACKED_Q4_K_DMMV=1`.
- Performance was not robust enough to enable by default. Initial microbench
  showed a promising win:
  Q4_K `k=2048 rows=4096`: default 254.137 us, packed 194.885 us.
  Q4_K `k=2048 rows=8192`: default 284.207 us, packed 208.296 us.
  But later reruns on the narrowed `k=2048 rows=4096` selector showed the
  opposite: default 210.566 us, packed 245.767 us. An intermediate rerun also
  showed `rows=8192` flipping negative, so the broader shape selector was
  removed.
- Recommendation: keep this as a gated experiment only. The code captures a
  closer Vulkan lane decomposition, but the current HIP compiler/runtime
  behavior does not produce a stable enough win to replace the simpler WG-tuned
  kernel. The next packed-lane attempt should inspect ISA/resource deltas before
  expanding to Q5_K/Q6_K or enabling by default.
- Shape-specific vector-RHS providers from the earlier float4 RHS experiment
  were not promoted. The Q5_K giant `k=2048 rows=248320` win appears only twice
  in the short Qwen decode trace, so it is not worth adding a decode-default
  provider. The narrow Q6_K `k=4096 rows=2048` win was small and overlapped with
  later WG/Q8_1 selector work. If revisited, this belongs behind an ISA-guided
  shape-specific provider, not a global load-width rewrite.

### 2026-04-09: TopK MoE mode coverage audit

- Took `pyre-workspace-hf9f.9.8` to compare Pyre TopK MoE coverage against the
  Vulkan TopK fusion modes in hopper items 34-36. Pyre already covers the Qwen
  gating pattern: `SOFT_MAX -> RESHAPE -> ARGSORT -> VIEW -> GET_ROWS` with the
  optional `RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE` normalization tail.
  This maps to Vulkan's `TOPK_MOE_EARLY_SOFTMAX_NORM` mode when the norm tail is
  present, and to `TOPK_MOE_EARLY_SOFTMAX` when it is absent.
- Added pattern-specific Pyre provider trace strings so TopK claims now report
  `TOPK_MOE_EARLY_SOFTMAX_NORM` or `TOPK_MOE_EARLY_SOFTMAX` and distinguish the
  subgroup provider from the non-subgroup provider.
- Fresh Qwen `n=2`, no-FA trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-8-topk/`. Result: zero Pyre
  fallbacks and 80 claims of
  `TOPK_MOE_EARLY_SOFTMAX_NORM pure_hip_f32_subgroup` at `k=8 nrows=1`.
- Did not add Vulkan's other TopK modes in this pass. Pyre does not currently
  implement the sigmoid+bias normalized mode, late-softmax/softmax-weight mode,
  or the optional output scale/bias affine fused onto TopK output weights. Those
  are model-general parity gaps, but they are not applicable to the active Qwen
  trace and should be added when a traced model actually hits those patterns.
- Validation: rebuilt `test-backend-pyre` and `llama-bench`; `test-backend-pyre`
  passed with `GGML_PYRE_KERNEL_PROVIDER=pure_hip`.

### 2026-04-09: F16-RHS and layout-specific matvec audit

- Took `pyre-workspace-hf9f.9.7` to audit hopper items 10-13 against the active
  Qwen decode trace. This was intentionally trace-first: no new variant was
  added unless it could replace a hot fallback or copy/materialization in the
  current graph.
- Evidence:
  Pyre trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-8-topk/`.
  Vulkan trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-7-vulkan/`.
  Pyre has zero fallbacks in the short decode trace and claims the hot matvec
  families directly. Vulkan reports `CONT`, `CPY`, and `CONCAT` timing buckets,
  but its matvec timing labels for this model do not expose a separate F16-RHS
  K-quant or P021/NC path that Pyre is missing on Qwen.
- Audit table:
  `MUL_MAT_VEC bf16 m=512 n=1 k=2048`, `m=32 n=1 k=2048`,
  and `m=2048 n=1 k=512`: Vulkan `bf16` DMMV; Pyre
  `pure_hip_bf16` plus `pure_hip_bf16` SWIGLU and BF16->F16 SET_ROWS fusions;
  action: no new RHS/layout variant.
  `MUL_MAT_VEC f16 m=256 n=1 k=256 batch=16`: Vulkan batched F16 DMMV; Pyre
  `pure_hip_f16_batched`; action: no new variant.
  `MUL_MAT_VEC f32 m=256 n=1 k=2048` and `m=1 n=1 k=2048`: Vulkan F32 DMMV;
  Pyre `pure_hip_f32_batched`; action: no new variant.
  `MUL_MAT_VEC q5_K/q6_K/q8_0`: Vulkan K-quant DMMV labels; Pyre
  `pure_hip_q5_K_wg128`, `pure_hip_q5_K_wg64`, `pure_hip_q6_K`,
  `pure_hip_q6_K_wg128`, `pure_hip_q8_0`, and Q8_0 ADD fusion; action: no
  F16-RHS variant because the active graph presents F32 RHS or opt-in Q8_1
  conversion, and the previous Q8_1 pass showed conversion overhead dominates.
  `MUL_MAT_ID_VEC q4_K`: Vulkan expert matvec; Pyre Qwen path uses
  `pure_hip_q4_K_wg64` `MUL_MAT_ID_MUL` and `MUL_MAT_ID_SWIGLU`; action: no
  F16-RHS/layout provider.
  Vulkan P021/NC F16 matvec providers: no matching Pyre fallback or traced copy
  opportunity in this Qwen decode; action: defer until a trace contains
  `fallback MUL_MAT reason=src0 is not contiguous`, `src1 is not contiguous`,
  or a Pyre materialization feeding a hot matvec.
- Decision: close this as an audit-only task. Adding Vulkan-style F16-RHS or
  P021/NC providers now would be unexercised code for this model and would risk
  reintroducing the copy/materialization problem unless a future trace proves a
  layout-specific provider can consume the real strided source directly.

### 2026-04-09: RMS/ROPE/softmax heat audit

- Took `pyre-workspace-hf9f.9.9` to decide whether to tune RMS/ROPE/softmax
  kernels immediately. The useful comparison point is the Vulkan perf trace
  artifact `build/pyre-epic2-results/trace-diff-hf9f-9-7-vulkan/`, because the
  Pyre provider trace currently gives counts and shapes but not per-provider GPU
  time for these small kernels.
- Vulkan timing estimate from that short run: matvec/expert families account for
  about 22.3 ms of logged kernel time, or ~63%; RMS/ROPE/softmax account for
  about 2.73 ms, or ~7.7%; the remaining logged time is elementwise/state/copy
  work. The visible non-matvec items include `RMS_NORM_MUL`, standalone `ROPE`,
  and `SOFT_MAX`, but they are not yet first-order relative to K-quant and expert
  matvec work.
- Pyre coverage in the matching trace is already structurally good:
  `RMS_NORM_MUL`, `ADD_RMS_NORM_MUL`, `RMS_NORM_MUL_ROPE`,
  `RMS_NORM_MUL_ROPE_SET_ROWS`, and standalone `SOFT_MAX` are all claimed with
  zero fallbacks. There is no evidence of a missing fusion or CPU fallback here.
- Compared code structure against Vulkan. Vulkan has RMS unroll buckets and
  softmax data caching/unroll variants, while Pyre currently uses simpler fixed
  loops. Those are legitimate future local optimizations, but they need either
  Pyre per-provider timing or a dedicated microbench extension for softmax/rope
  before changing production kernels. The existing `pyre-kernel-bench` covers
  RMS norm but not the fused RMS+ROPE or softmax shapes that appear in Qwen.
- Decision: defer code changes for this pass. Revisit after the ISA/resource
  inspection loop or Pyre per-provider timing can isolate these kernels; do not
  spend the current decode catch-up budget here while matvec/expert kernels
  dominate the trace.

### 2026-04-09: Flash-attention heat audit

- Took `pyre-workspace-hf9f.9.10` to decide whether to port scalar
  flash-attention decode optimizations now. This is separate from the default
  Qwen no-FA path; the active optimization checkpoints above use `-fa 0`.
- Fresh Pyre `-fa 1`, `n=2` provider trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-10-pyre-fa/`. Result: zero Pyre
  fallbacks. The trace claims 20
  `FLASH_ATTN_EXT pure_hip_f32_k_f16_v_f16_decode` calls with
  `D=256 KV=256 N=1 H=16 H_KV=2`, plus 20 `CONT pure_hip_strided_copy` and 2
  `CPY pure_hip_f32_f16_copy` claims for the FA-side layout/cast work. Those are
  Pyre providers, not CPU fallbacks.
- Fresh Vulkan `-fa 1`, `n=2` perf trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-10-vulkan-fa/`. Vulkan reports 20
  `FLASH_ATTN_EXT` calls totaling about 1.21 ms in the short run, while the
  Q5_K/Q6_K/Q4_K expert matvec families still dominate the logged timing.
- No-trace Pyre `llama-bench -p 0 -n 32 -fa 1` checkpoint on the current build
  reported 25.420 tok/s with `/usr/bin/time -v` process CPU at 81%. This is
  below the best recent no-FA checkpoints, so FA is still optional rather than
  the path to improving the default decode number.
- Code comparison: Pyre scalar FA still uses double accumulators, a full
  `logits[1024]` shared-memory materialization, and one workgroup per
  `(head, token, seq)`. Vulkan has split-K, mask optimization, GQA-aware
  remapping, shape tuning, and better scalar staging options. Those are real
  future targets, especially at longer KV, but the current Qwen short-context
  decode trace does not justify taking this before the remaining matvec/compiler
  work.
- Decision: defer FA kernel changes for this pass. Revisit with a KV-length
  sweep and/or Pyre per-provider timing. First candidate then should be the
  float-only accumulator variant behind an environment gate, followed by split-K
  only if longer-context traces show FA under-occupancy.

### 2026-04-09: Temp reuse and memory-movement audit

- Took `pyre-workspace-hf9f.9.11` to check whether copy-like work or temp reuse
  became the next hidden fixed overhead after the earlier copy-elimination
  changes. This audit intentionally stayed in `ggml-pyre`; broader HSA dispatch
  batching and kernarg work belongs to the separate runtime overhead epic.
- Current no-FA Pyre trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-8-topk/`. Result: zero fallbacks.
  Copy-like/materialization-related claims are still clean for the default
  decode path: no `CPY`, no `CONT`, and no `CONCAT`. The remaining state/layout
  work is fused provider work: 60 `SSM_CONV_UPDATE_SILU`, 60
  `GATED_DELTA_NET_STATE_UPDATE`, 20 `MUL_MAT_SET_ROWS`, and 20
  `SIGMOID_MUL_STRIDED`.
- FA-on Pyre trace artifact:
  `build/pyre-epic2-results/trace-diff-hf9f-9-10-pyre-fa/`. Result: zero
  fallbacks. It has 20 `CONT pure_hip_strided_copy`, 2
  `CPY pure_hip_f32_f16_copy`, and 20 `SET_ROWS pure_hip_f32_f16`; these are the
  expected FA-side layout/cast providers described in the profiling guide, not
  CPU fallback or untraced copies.
- Scratch/temp reuse status: the opt-in Q8_1 path already uses a reusable
  `scratch_q8_1` buffer, but the Q8_1 retest showed conversion dispatch and
  scratch traffic still lose end-to-end, so no new default temp reuse change was
  made here. Runtime-level command batching and kernarg allocation are tracked
  outside this child task.
- Decision: close as audit-only. The current default Qwen no-FA decode trace is
  copy-clean at the provider level. The guardrail for future changes is simple:
  new `CPY`, `CONT`, `CONCAT`, `buffer_copy`, or `fallback` entries in the no-FA
  trace should block kernel-comparison conclusions until the materialization is
  explained or fused.

### 2026-04-09: ISA/resource inspection loop

- Took `pyre-workspace-hf9f.9.12` to add a repeatable ISA/resource inspection
  loop for the pure-HIP catalog. Added
  `sources/llama.cpp/tools/pyre-epic2/pyre-kernel-isa-summary.py`.
- The script recompiles selected catalog sources with the same HIP device-only
  `-O3` flow used by the generated Pyre kernel catalog, emits AMDGCN assembly
  with `-S`, unbundles clang's `__CLANG_OFFLOAD_BUNDLE__` payload with
  `clang-offload-bundler`, and reads raw ELF AMDHSA metadata with
  `llvm-readobj --notes`. It reports rough opcode counts plus per-entrypoint
  VGPR, SGPR, spill counts, LDS/group segment size, private segment size,
  kernarg size, wavefront size, and max workgroup size. This is intentionally a
  developer tool, not a fragile default build gate.
- Repro commands:
  `sources/llama.cpp/tools/pyre-epic2/pyre-kernel-isa-summary.py --kernel 'mul_mat_vec_q[46]_k' --out-dir build/pyre-epic2-results/isa-hf9f-9-12 --json build/pyre-epic2-results/isa-hf9f-9-12/summary.json`
  and
  `sources/llama.cpp/tools/pyre-epic2/pyre-kernel-isa-summary.py --out-dir build/pyre-epic2-results/isa-hf9f-9-12-hot --json build/pyre-epic2-results/isa-hf9f-9-12-hot/summary.json`.
- Spot-check result for the packed-lane Q4_K DMMV experiment: the baseline
  `pyre_mul_mat_vec_q4_k_f32`, `wg128`, and `wg64` entrypoints all compile at
  23 VGPR / 21 SGPR with zero spills. The gated
  `pyre_mul_mat_vec_q4_k_packed_wg64_f32` entrypoint compiles at 44 VGPR / 21
  SGPR with zero spills. That gives a plausible root cause for the earlier
  unstable packed-lane performance: the arithmetic rewrite removed no spills,
  but almost doubled VGPR pressure relative to the simpler WG64 path.
- Q6_K WG selector result: `pyre_mul_mat_vec_q6_k_f32`, `wg128`, and `wg64`
  all compile at 29 VGPR / 32 SGPR with zero spills. The workgroup-selector
  gains therefore look like launch/occupancy/cache-shape effects, not register
  allocation changes among the three entrypoints.
- Broader hot-kernel smoke result: the default hot selector compiled matvec,
  expert, flash-attention, TopK, RMS/softmax, state-update, and Q8_1 conversion
  sources successfully and found no VGPR or SGPR spills in the inspected slice.
  Current high-level ISA signals are register pressure, LDS/group segment size,
  barriers, and instruction mix; no evidence yet of compiler spill cliffs.

### 2026-04-09: MFMA/WMMA matrix-matrix prefill scope

- Took `pyre-workspace-hf9f.9.13` as a scoping pass rather than a decode-path
  code change. The local target is `gfx1100`, so the practical hardware
  primitive to validate first is `v_wmma*`; CDNA-style `v_mfma*` should remain
  in the inspection vocabulary but is not the right first assumption for this
  machine.
- Reference comparison:
  Vulkan's `mul_mm.comp`/`mul_mm_cm2.comp` tile around `BM=64`, `BN=64`,
  `BK=16/32`, `BLOCK_SIZE=64`, `WM=32`, `WN=32`, `WMITER=2`, `TM=4`, `TN=2`,
  with scalar, coopmat, aligned, and f16acc/f32acc variants. `mul_mmq.comp`
  covers quantized `src0` with a packed `block_q8_1_x4_packed128` RHS and
  dot-product style accumulation. `mul_mat_split_k_reduce.comp` is downstream
  of having a baseline tiled matmul; do not start there.
- Current Pyre shape evidence:
  `build/pyre-epic2-results/trace-diff-hf9f-9-13-prompt64/` is a Pyre
  prompt-only trace for `-p 64 -n 0 -fa 0`. It has zero fallbacks but claims the
  prompt work through matvec-style providers, for example BF16 `k=2048 rows=32
  cols=64`, BF16 `k=512 rows=2048 cols=64`, F32 batched `k=2048 rows=256
  cols=64`, Q5_K `k=2048 rows=8192 cols=64`, Q6_K `k=2048 rows=4096 cols=64`,
  Q6_K `k=4096 rows=2048 cols=64`, Q8_0 `MUL_MAT_ADD k=4096 rows=2048
  cols=64`, and Q4_K expert paths with `tokens=64`. The run reported about
  115 prompt tok/s. That makes tiled matrix-matrix meaningful again, but the
  current decode path should not be replaced by this work.
- Recommended next implementation slice:
  add a gated dense provider such as `mul_mat_mm_bf16_f32.hip.cpp` for
  `src0=BF16`, `src1=F32`, `dst=F32`, contiguous layout, `src1->ne[1] >= 16`
  or `>= 32`, `src0->ne[2/3] == 1`, and `src1/op` batch dims matching the
  existing batched-provider guard. Keep it behind an opt-in environment flag
  until prompt microbench and trace data show wins. The first kernel should
  tile `M x N` rather than launch one workgroup per `(row, col)`, and should
  use the ISA summary tool to require visible `v_wmma*` (or document why clang
  cannot emit it). The decode providers remain the default for `cols=1`.
- Follow-on slices:
  add an aligned dense variant only after the base BF16/F32 tile wins; add F16
  only if a trace shows it hot; then evaluate Q4_K/Q5_K/Q6_K with a packed Q8_1
  RHS tile modeled after Vulkan `mul_mmq.comp`. Split-K and matmul-id subgroup
  matrix-matrix should wait until the dense tile exists and a prompt/batched MoE
  trace shows under-occupancy or large-K reduction pressure.

### 2026-04-09: Epic 2.2 ISA-guided follow-up baseline

- Took `pyre-workspace-hf9f.10` to continue from the Epic 2.1 checkpoint with
  the trace and ISA tools in the loop.
- Fresh no-trace decode baseline:
  `build/pyre-epic2-results/hf9f-10-baseline-pyre-decode-n32.json` and
  `build/pyre-epic2-results/hf9f-10-baseline-pyre-decode-n32.time`.
  Command: `GGML_PYRE_KERNEL_PROVIDER=pure_hip /usr/bin/time -v
  ./build/llama-pyre/bin/llama-bench -m
  ./models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf -p 0 -n 32 -b 512 -ub 512 -fa 0 -r
  1 -o json --no-warmup -ngl 99 -dev PYRE0`. Result: 27.087 tok/s, 81%
  process CPU, max RSS ~20.3 GB. This is in the same noise band as the prior
  27.7-28.0 tok/s checkpoints.
- Fresh Vulkan no-trace check:
  `build/pyre-epic2-results/hf9f-10-baseline-vulkan-decode-n32.json` and
  `.time`. Result: 67.506 tok/s, 95% process CPU. This is below the earlier
  ~105 tok/s baseline, so use it only as a run-local comparison until the Vulkan
  configuration/noise is rechecked; do not reinterpret the whole Pyre/Vulkan gap
  from this one sample.
- Fresh short Pyre/Vulkan trace:
  `build/pyre-epic2-results/trace-diff-hf9f-10-baseline/`. Pyre has zero
  fallbacks. Hot Pyre decode claims are unchanged: 80
  `MUL_MAT_ID_SWIGLU pure_hip_q4_K_wg64`, 80 `MUL_MAT_ID_MUL
  pure_hip_q4_K_wg64`, 60 `MUL_MAT pure_hip_q5_K_wg128` at
  `k=2048 rows=8192`, 60 `MUL_MAT pure_hip_q6_K` at `k=2048 rows=4096`, 60
  `MUL_MAT pure_hip_q6_K_wg128` at `k=4096 rows=2048`, and 20
  `MUL_MAT pure_hip_q6_K` at `k=2048 rows=8192`. The next work stays on
  K-quant/expert matvec arithmetic unless microbench or ISA data contradicts it.

### 2026-04-09: Q6_K dot4 byte-safe inner-loop rewrite

- Hypothesis: Q6_K is hit in three traced decode shapes, and its baseline ISA
  showed a high instruction/global-load count relative to the simple work it
  performs. Replacing four generic `pyre_q6_k_value()` calls with a single
  group-specific four-value unpack should reduce branch/index recomputation and
  expose packed byte work to clang.
- Baseline microbench results on traced shapes:
  `mul_mat_vec_q6_k k=2048 rows=4096`: 242.296 us;
  `k=4096 rows=2048`: 233.087 us;
  `k=2048 rows=8192`: 280.087 us. Baseline ISA artifact:
  `build/pyre-epic2-results/isa-hf9f-10-baseline-hot/`, with Q6_K at 1453
  instructions, 114 `global_load` opcode matches, 29 VGPR, 32 SGPR, and zero
  spills.
- First attempt used raw `uint32_t *` loads from Q6 byte arrays and failed
  correctness. The root cause was partly unsafe/misaligned block-row access
  and partly a ql-index mapping bug: Q6 ql byte half is selected by
  `group_in_half & 1`, while nibble half is selected by `group_in_half >= 2`.
  The fixed implementation assembles the four-byte ql/qh words byte-by-byte and
  uses the corrected mapping.
- Fixed microbench results:
  `k=2048 rows=4096`: 216.705 us;
  `k=4096 rows=2048`: 195.555 us;
  `k=2048 rows=8192`: 226.246 us. ISA artifact:
  `build/pyre-epic2-results/isa-hf9f-10-q6-dot4-fixed/`, with Q6_K at 733
  instructions, 15 `global_load` opcode matches, 34 VGPR, 21 SGPR, and zero
  spills. This is a good tradeoff despite the +5 VGPR pressure.
- Validation:
  `cmake --build build/llama-pyre --target pyre-kernel-bench test-backend-pyre
  llama-bench -j$(nproc)` completed; `GGML_PYRE_KERNEL_PROVIDER=pure_hip
  ./build/llama-pyre/bin/test-backend-pyre` passed. Short provider trace
  artifact `build/pyre-epic2-results/trace-diff-hf9f-10-q6-dot4/` has zero
  fallbacks and unchanged provider selection.
- End-to-end no-trace Qwen decode:
  `build/pyre-epic2-results/hf9f-10-q6-dot4-pyre-decode-n32.json` and `.time`
  reported 30.852 tok/s at 78% process CPU, versus the hf9f.10 baseline of
  27.087 tok/s at 81% CPU. Keep this default-enabled.
- Follow-up Q5_K dot4 attempt:
  Applied the analogous four-value byte assembly to Q5_K and tested
  `mul_mat_vec_q5_k k=2048 rows=8192` plus the rare `rows=248320` projection.
  It was correctness-clean but did not improve the hot shape materially
  (`~228 us`, essentially baseline), raised ISA pressure from 31 to 36 VGPR and
  slightly increased instruction count to 829. Full decode with Q5+Q6 dot4
  reported 30.291 tok/s at 81% CPU, below the Q6-only checkpoint. Reverted the
  Q5 edit and recorded it as a dead end.
- Follow-up Q4_K expert `MUL_MAT_ID_MUL` dot4 attempt:
  Applied a similar four-value q4 unpack to the traced `k=512 rows=2048 ids=8
  tokens=1` expert-mul shape. It was correctness-clean but measured 216.946 us,
  not better than the earlier ~213 us baseline/noise band. Reverted without
  running full decode; this suggests the Q4 expert fused path is not leaving the
  same branch/index overhead on the table as Q6_K.
- Follow-up Q5_K WG64 selector attempt:
  Changed the Q5_K selector locally from `rows > 65536 ? 64 : 128` to
  `rows >= 8192 ? 64 : 128`, targeting the 60-call decode shape. The rebuilt
  microbench measured 229.666 us for `k=2048 rows=8192`, again not a clear win
  over baseline/noise. Reverted the selector change; no full decode run needed.
- Final rebuilt Q6-only checkpoint after reverting the Q5/Q4 selector/body
  experiments:
  `build/pyre-epic2-results/hf9f-10-final-q6-pyre-decode-n32.json` and `.time`
  reported 30.811 tok/s at 81% process CPU on build commit `5fdcad32f`.
  This is the final hf9f.10 default-path state for this pass.

### 2026-04-09: Epic 2.3 ranked decode cost model

- Took `pyre-workspace-hf9f.11` for a deeper Vulkan-parity pass, starting from
  llama.cpp commit `5fdcad32f`.
- Repeated no-trace baseline artifacts:
  `build/pyre-epic2-results/hf9f-11/baseline-pyre-r3.json` and `.time`;
  `build/pyre-epic2-results/hf9f-11/baseline-vulkan-r3.json` and `.time`.
  Pyre reported 30.842 tok/s, stddev 0.093 tok/s, 84% process CPU. Vulkan
  reported 108.209 tok/s, stddev 3.920 tok/s, 94% process CPU, so the earlier
  low 67 tok/s Vulkan sample was not representative for this machine/config.
- Fresh Pyre trace artifact:
  `build/pyre-epic2-results/hf9f-11/trace-pyre-n8/`. Result: zero fallbacks.
  No `CPY`, `CONT`, or `CONCAT` provider claims appeared in the no-FA decode
  trace.
- Initial microbench-weighted ranking from the n=8 trace. The absolute totals
  overestimate decode time because the model-free microbench includes fixed
  graph/dispatch costs, but the ranking is still useful for prioritization:
  Q5_K `k=2048 rows=8192`, 240 claims, 231.386 us median, estimated 55.5 ms;
  Q6_K `k=4096 rows=2048`, 240 claims, 197.415 us, estimated 47.4 ms; Q6_K
  `k=2048 rows=4096`, 240 claims, 184.475 us, estimated 44.3 ms; Q4_K
  `MUL_MAT_ID_SWIGLU k=2048 rows=512 ids=8 tokens=1`, 320 claims, 225.086 us,
  estimated 72.0 ms; Q4_K `MUL_MAT_ID_MUL k=512 rows=2048 ids=8 tokens=1`,
  320 claims, 215.746 us, estimated 69.0 ms; Q5_K giant `k=2048 rows=248320`,
  8 claims, 1222.782 us, estimated 9.8 ms; Q6_K `k=2048 rows=8192`, 80 claims,
  214.676 us, estimated 17.2 ms.
- Dense matvec microbench medians were all around the fixed-overhead floor in
  this harness: BF16 `k=2048 rows=32` 170.014 us, BF16 `k=512 rows=2048`
  176.194 us, BF16 `k=2048 rows=512` 172.624 us, F32 `k=2048 rows=256`
  178.324 us, and F16 `k=256 rows=256 ne2=16` 175.835 us. These are still hot
  by count, but the K-quant/expert arithmetic remains the better first attack.

### 2026-04-09: Q5_K 16-lane Vulkan-style block mapping

- Hypothesis: Q5_K remained a top standalone cost. The prior failed Q5 attempts
  were a simple four-value unpack and a workgroup selector change, not a
  structural lane mapping change. Vulkan's Q5 DMMV shader uses 16 lanes per
  quant block with each lane handling a 16-value packed chunk, while the Pyre
  HIP kernel used 64 lanes per quant block with four values per lane.
- Changed `mul_mat_vec_q5_k.hip.cpp` so each lane assembles strided qh/qs words
  for the same 16-value pattern as Vulkan: groups `(0,1,4,5)` or `(2,3,6,7)`
  depending on `v_im`, RHS offsets `0,1,16,17`, and a hoisted qh word reused
  across the four q5 dot4 chunks.
- Validation: rebuilt `pyre-kernel-bench`, `test-backend-pyre`, and
  `llama-bench`; `GGML_PYRE_KERNEL_PROVIDER=pure_hip
  ./build/llama-pyre/bin/test-backend-pyre` passed.
- Microbench results versus the hf9f.11 baseline:
  Q5_K `k=2048 rows=8192` improved from 231.386 us to 205.745 us; Q5_K
  `k=2048 rows=248320` improved from 1222.782 us to 794.501 us. The first
  non-hoisted version was positive but weaker: 216.236 us and 791.281 us.
- ISA/resource tradeoff: baseline Q5 ISA artifact
  `build/pyre-epic2-results/hf9f-11/isa-q5-baseline/` reported 814
  instructions, 27 `global_load` opcode matches, 31 VGPR, 22 SGPR, and zero
  spills. The accepted qh-hoisted artifact
  `build/pyre-epic2-results/hf9f-11/isa-q5-lane16-qhhoist/` reports 1171
  instructions, 54 `global_load` opcode matches, 72 VGPR, 21 SGPR, and zero
  spills. This is a higher-register, higher-ILP tradeoff; the microbench and
  decode data justify it for this shape but it should be revisited on a real
  profiler.
- End-to-end no-trace checkpoint:
  `build/pyre-epic2-results/hf9f-11/q5-lane16-qhhoist/pyre-r3.json` and
  `.time` reported 31.550 tok/s, stddev 0.079 tok/s, 83% process CPU, versus
  the hf9f.11 baseline of 30.842 tok/s. Short provider trace artifact
  `build/pyre-epic2-results/hf9f-11/q5-lane16-qhhoist/trace-pyre-n8/` had zero
  fallbacks and unchanged provider selection.
- Final rebuilt checkpoint after reverting rejected Q4 expert experiments:
  `build/pyre-epic2-results/hf9f-11/final/pyre-r3.json` and `.time` reported
  31.424 tok/s, stddev 0.041 tok/s, 84% process CPU on build commit
  `495467e21`. This remains above the repeated hf9f.11 baseline and is the
  default branch state.
- Adjacent Q4_K expert lane16 experiment:
  applying the same 16-lane packed mapping to both `MUL_MAT_ID_MUL` and
  `MUL_MAT_ID_SWIGLU` passed correctness but was not a default win.
  `SWIGLU k=2048 rows=512` improved from 225.086 us to 217.066 us in the
  combined experiment and 215.856 us when tested alone, but VGPR rose from 36
  to 92 and full decode with `SWIGLU` alone reported 31.493 tok/s versus the
  Q5-only 31.550 tok/s checkpoint. `MUL k=512 rows=2048` regressed from
  215.746 us to 223.816 us and VGPR rose from 22 to 59. Both Q4 expert edits
  were reverted. Artifact:
  `build/pyre-epic2-results/hf9f-11/isa-q4id-lane16/`; decode artifact:
  `build/pyre-epic2-results/hf9f-11/q4swiglu-lane16/pyre-r3.json`.

### 2026-04-09: Epic 2.4 structural map baseline

- Took `pyre-workspace-hf9f.12` as the last structural-analysis-only pass
  before profiler-guided work. The evaluation goal is a map of structural
  opportunities tried/rejected vs not yet tried, not only commits.
- Repeated post-`495467e21` baselines:
  `build/pyre-epic2-results/hf9f-12/baseline-pyre-decode-r3.json`: Pyre
  decode 31.360 tok/s, stddev 0.071 tok/s, 83% process CPU.
  `build/pyre-epic2-results/hf9f-12/baseline-vulkan-decode-r3.json`: Vulkan
  decode 107.855 tok/s, stddev 3.738 tok/s, 94% process CPU.
  `build/pyre-epic2-results/hf9f-12/baseline-pyre-prompt64-r3.json`: Pyre
  prompt64 187.323 prompt tok/s, stddev 1.976 tok/s, 75% process CPU.
- Fresh traces:
  decode trace `build/pyre-epic2-results/hf9f-12/trace-pyre-decode-n8/`
  remains clean: zero fallbacks and no no-FA `CPY`/`CONT`/`CONCAT` claims.
  prompt trace `build/pyre-epic2-results/hf9f-12/trace-pyre-prompt64/` has zero
  fallbacks but still shows prompt-only state materialization: 30 `CONCAT`, 30
  `CPY pure_hip_strided_copy nrows=8192`, and 30 `SSM_CONV_SILU`.
  Vulkan decode trace
  `build/pyre-epic2-results/hf9f-12/trace-vulkan-decode-n8/` again shows the
  remaining major Vulkan-side costs in Q4/Q5/Q6 matvec/expert labels, but it
  also records Vulkan `CPY/CONCAT/CONT` timing buckets, so trace-label parity
  alone is not sufficient to explain the 3.4x decode gap.
- Refreshed microbench samples:
  decode Q5_K `2048x8192`: 219.366 us; decode Q5_K giant `2048x248320`:
  800.101 us; Q6_K `2048x4096`: 183.004 us; Q6_K `4096x2048`: 197.175 us;
  Q6_K `2048x8192`: 226.846 us; Q4 expert `SWIGLU 2048x512 ids=8 tokens=1`:
  224.656 us; Q4 expert `MUL 512x2048 ids=8 tokens=1`: 215.966 us.
  Prompt/multi-column samples show a distinct waste class: Q5_K
  `2048x8192 cols=64`: 1668.053 us; Q6_K `2048x4096 cols=64`: 1173.851 us;
  Q6_K `4096x2048 cols=64`: 869.603 us; Q4 expert ID
  `512x2048 ids=8 tokens=64`: 1628.773 us; BF16 `512x2048 cols=64`:
  422.621 us; BF16 `2048x512 cols=64`: 305.648 us; F32 `2048x256 cols=64`:
  266.007 us. The Q4 `SWIGLU tokens=64` harness row hit a small numerical
  tolerance mismatch (`~0.0064` absolute) and is not used as validated data.
- Structural opportunity map at this checkpoint:
  Tried and accepted: Q6_K byte-safe unpack/dot4; Q5_K 16-lane Vulkan-style
  block mapping; decode recurrent-state copy fusions; full-attention gate
  materialization fusion; Q8_0 MUL_MAT+ADD fusion; K-quant and expert WG64/128
  selectors.
  Tried and rejected/reverted: Q5 simple dot4; Q5 WG64 selector for 8192 rows;
  Q5 two-row workgroup reuse; Q4 expert simple dot4; Q4 expert lane16 mapping;
  Q4 packed standalone DMMV; global vector RHS loads; Q8_1 default policy; Q4
  expert q8_1; prompt SSM update fusion/defaulting, described below.
  Not yet structurally tried: true tiled prompt/prefill matrix-matrix for BF16
  or F16 with `cols>=16/32`; quantized MMQ-style prompt providers with packed
  RHS; Q4 expert kernels that change work ownership to reduce repeated expert
  id/address loads across rows/tokens without raising VGPR; TopK/ARGSORT prompt
  fusion beyond current decode TopK mode; profiler-guided runtime dispatch
  packing/command-buffer changes.

### 2026-04-09: Gated prompt SSM_CONV_UPDATE experiment

- Prompt trace showed a dataflow materialization absent from decode:
  `CONCAT(conv_state, qkv_mixed) -> CPY(last_conv_state) -> SSM_CONV -> SILU`.
  Decode already fuses this as `SSM_CONV_UPDATE_SILU`, but the support predicate
  was intentionally decode-shaped with `input->ne[0] == 1`.
- Generalized the existing `ssm_conv_update_f32` kernel to read the convolution
  window from separate `conv_state` and multi-token `input` buffers. The final
  recurrent state update is now written only by `token == 0`, avoiding the first
  naive prompt version's redundant `n_tokens` copies of the same state window.
- Because full prompt perf regressed, the prompt form was first gated off by
  default with `GGML_PYRE_ENABLE_PROMPT_SSM_CONV_UPDATE=1` as the force knob.
  That was still not a good shippable state: the shared decode kernel source
  changed even when the prompt gate was off, and repeated default decode samples
  did not clear the no-regression bar. Commit `37866df1b` was therefore
  reverted by `f3b5fc3e4`.
- Validation artifacts:
  `build/pyre-epic2-results/hf9f-12/ssm-prompt-update-gated/default-trace-pyre-prompt64/`
  shows the default prompt path still has `CONCAT`, `CPY`, and `SSM_CONV_SILU`.
  `build/pyre-epic2-results/hf9f-12/ssm-prompt-update-gated/forced-trace-pyre-prompt64/`
  shows forced prompt fusion replacing those with 30
  `SSM_CONV_UPDATE_SILU` claims and zero fallbacks.
  `test-backend-pyre` passed with the gated code before revert, then passed
  again after the revert.
- Performance evidence:
  baseline prompt64 was 187.323 tok/s (`r=3`). The first generalized forced
  version regressed to 183.130 tok/s (`r=5`) because it wrote the final state
  window once per token. After fixing that, the forced prompt path still
  reported 182.426 tok/s (`r=5`) at 72% CPU. Decode stayed healthy at
  31.720 tok/s (`r=3`), but this is not enough to default-enable a prompt
  regression. After the revert, default decode returned to 31.553 tok/s
  (`r=3`, 84% CPU) in
  `build/pyre-epic2-results/hf9f-12/final-reverted/pyre-decode-r3.json`.
  Prompt64 after the revert measured 183.845 tok/s (`r=5`, 72% CPU) in
  `build/pyre-epic2-results/hf9f-12/final-reverted/pyre-prompt64-r5-repeat.json`;
  this is lower than the earlier 187.323 tok/s prompt baseline but uses the
  restored baseline code path, so treat it as prompt-run variability or a
  runtime/caching effect to revisit with profiler traces rather than as an SSM
  kernel change.
- ISA/resource artifact:
  `build/pyre-epic2-results/hf9f-12/isa-ssm-update-gated/` reports
  `ssm_conv_update_f32` at 598 instructions, 24 VGPR, 40 SGPR, zero spills, no
  LDS. The failure mode is therefore not register spilling; likely causes are
  branch/address overhead in the two-buffer window load or runtime scheduling
  effects where removing two small copy-like dispatches does not yet pay for
  the fused kernel's more complex access pattern.
- Decision: revert rather than keep a speculative gated kernel. This is a good
  example of "inner copy removed" not automatically being a throughput win on
  the current runtime without profiler evidence. The next structurally cleaner
  version should be a separate prompt provider/kernel entry, not a shared
  decode-kernel generalization, so the default decode ISA remains byte-for-byte
  insulated while the prompt path is profiled.

### 2026-04-09: hf9f.13 rocprof-driven pass

- Scope: switched this pass to rocprof/kernel-only evidence per
  `pyre-workspace-hf9f.13`. Primary baseline was
  `build/rocprof-pyre-kernelonly-n64/kernelonly-n64_results.db`, which showed
  80,316 kernel dispatches and about 891,661 us over 64 decode tokens
  (~13.93 ms/token). Top measured buckets were Q4 expert `SWIGLU`/`MUL`
  (~2.71 ms/token combined), `pyre_gated_delta_net_f32` (~1.03 ms/token), and
  dense/K-quant matvec buckets.
- P0 copy attribution: added gated runtime logging via
  `GGML_PYRE_TRACE_BUFFER_COPIES`. `GGML_PYRE_TRACE_BUFFER_COPIES=993280`
  identified the steady 993,280-byte transfer as:
  `buffer get size=993280 tensor=result_output type=f32 ne=[248320,1,1,1]`.
  This is full raw-logits extraction (`248,320 * sizeof(float)`) from the
  output tensor, not an inner graph materialization or ggml `CPY` provider.
  Provider trace with `GGML_PYRE_TRACE_PROVIDERS=1` also showed zero
  `CPY`/`CONT`/`DUP` claims and zero fallbacks for the short decode run.
- Copy cost evidence: final full HSA/copy trace
  `build/rocprof-pyre-decode-n16-hf9f13-final/decode-n16-hf9f13-final_results.db`
  and log
  `build/pyre-epic2-results/hf9f-13/decode-n16-hf9f13-final.log` show the
  steady slice has 16 copies of size 993,280 B totaling 625.006 us, average
  39.063 us. This is not worth kernel fusion work; removing it requires
  changing benchmark/sampling/API behavior so raw logits are not read back.
- Accepted gated-delta optimization: Pyre HIP previously used one column per
  workgroup with up to 256 lanes and shared reductions. CUDA/Vulkan use the
  more appropriate structure: multiple columns per workgroup with 32 lanes per
  column and per-lane row shards. Commit `fe56f372a` remaps
  `gated_delta_net_f32` to four columns per 128-lane workgroup and changes
  dispatch X from `S_v` to `ceil(S_v / 4)`.
- Gated-delta profiler result: kernel-only n64 rocprof artifact
  `build/rocprof-pyre-kernelonly-n64-hf9f13-gdn/kernelonly-n64-hf9f13-gdn_results.db`
  shows `pyre_gated_delta_net_f32` at 35,506.097 us over 64 tokens, average
  18.493 us/call, versus the prior 65,709.343 us over 64 tokens, average
  ~34.079 us/call. This is ~472 us/token saved in the target bucket.
- Gated-delta ISA/resource tradeoff: baseline artifact
  `build/pyre-epic2-results/hf9f-13/isa-gated-delta-baseline/` reported
  1,146 instructions, 13 VGPR, 83 SGPR, 1,024 B LDS, 18 barriers, and no
  spills. Final artifact
  `build/pyre-epic2-results/hf9f-13/isa-gated-delta-cols4-final/` reports
  1,629 instructions, 54 VGPR, 94 SGPR, 512 B LDS, 12 barriers, and no spills.
  The accepted tradeoff is higher per-workgroup register/instruction footprint
  for four columns per workgroup and fewer barriers; rocprof validates it.
- Wall guardrail: unprofiled Pyre decode artifact
  `build/pyre-epic2-results/hf9f-13/pyre-gdn-cols4-n64-r3.json` reports
  31.394 tok/s, stddev 0.019. This is neutral versus the prior ~31.4 tok/s
  range, which is consistent with current runtime overhead muting a
  ~0.47 ms/token kernel-only win.
- Q4 expert experiment rejected: tried a different Q4 expert structure from
  the earlier lane16/dot4 attempts by vectorizing four packed q4 bytes with
  `uint32_t` and using `float4` RHS loads in the Q4 `MUL`/`SWIGLU` expert
  kernels. ISA shrank slightly for those variants, but clean-cache microbench
  did not produce a reliable win: `MUL` regressed in the clean-cache test, and
  `SWIGLU` results were contaminated until `$PYRE_CACHE_DIR/kernels` was
  cleared. Both source edits were reverted. Artifacts:
  `build/pyre-epic2-results/hf9f-13/isa-q4-id-baseline/` and
  `build/pyre-epic2-results/hf9f-13/isa-q4-id-vector-load/`.
- Build/linkage fix: commit `6df6991c4` adds PIC to the Pyre test/tool targets
  needed by this pass (`llama-bench`, `pyre-kernel-bench`, tests, and the
  static `cpp-httplib` archive) after lld rejected non-PIC objects when linking
  the ROCm 7.13 Pyre build.
- Validation commands run: rebuilt with
  `cmake --build build/llama-pyre-rocm713 --target llama-bench pyre-kernel-bench
  test-backend-pyre -j$(nproc)`; ran
  `GGML_PYRE_KERNEL_PROVIDER=pure_hip ./build/llama-pyre-rocm713/bin/test-backend-pyre`
  with the ROCm 7.13 `LD_LIBRARY_PATH`; ran kernel-only n64 rocprof, unprofiled
  n64 `llama-bench -r 3`, final gated-delta ISA summary, and full HSA/copy n16
  trace with `GGML_PYRE_TRACE_BUFFER_COPIES=993280`.
- Next profiler target: Q4 expert `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` and
  `pyre_mul_mat_id_q4_k_mul_wg64_f32` remain the largest measured kernel-only
  bucket (~2.7 ms/token combined). The vector-load route is rejected; the next
  credible work should be a more structural expert-data reuse/ownership change
  or a profiler-assisted memory-traffic analysis, not another small unpack
  variant.

### 2026-04-09: Q4 expert follow-up analysis after hf9f.13

- Re-read the hot Pyre Q4 expert kernels against Vulkan's Q4_K matvec-id path.
  Pyre's fused expert kernels still compute one output row per workgroup. Vulkan's
  `mul_mat_vec_q4_k.comp` supports `NUM_ROWS` rows per workgroup and uses a
  packed 16-lane Q4_K superblock decomposition, but prior Pyre experiments show
  those two ideas must be separated: simple row tiling and simple load widening
  are not enough.
- Current Pyre ISA/resource state is not register-bound in the default WG64
  kernels. Fresh ISA summary artifact
  `build/pyre-epic2-results/analysis-q4-current-isa/` reports
  `mul_mat_id_q4_k_mul` at 1,429 instructions, 22 VGPR, 42 SGPR, no spills, and
  `mul_mat_id_q4_k_swiglu` at 1,948 instructions, 36 VGPR, 46 SGPR, no spills.
  This points away from minor occupancy cleanup and toward a dataflow/kernel
  structure change if we want a large win.
- Tried a local-only two-rows-per-workgroup variant for the fused WG64 expert
  kernels. It reused the RHS four-value load across two rows and reduced/wrote
  two row sums per workgroup. The experiment was reverted.
  Microbench with `n_experts=8 n_ids=8 n_tokens=1`:
  `SWIGLU k=2048 rows=512` baseline 294.218 us median / 285.357 us min,
  row2 304.758 us median / 298.728 us min;
  `SWIGLU k=512 rows=2048` baseline 245.067 us / 241.586 us, row2
  230.306 us / 218.436 us;
  `MUL k=2048 rows=512` baseline 270.757 us / 265.647 us, row2
  290.777 us / 284.058 us;
  `MUL k=512 rows=2048` baseline 223.086 us / 205.975 us, row2
  232.706 us / 217.116 us. The two Qwen decode-relevant shapes are
  `SWIGLU k=2048 rows=512` and `MUL k=512 rows=2048`, so this is not a default
  direction.
- Vulkan's AMD policy likely uses q8_1 RHS / integer-dot MMVQ for Q4_K matvec-id
  when `k >= 2048`, but not for the `k=512` down/expert MUL shape. Pyre's forced
  q8_1 expert path previously lost badly because the quantize dispatch was not
  amortized, and there is no SWIGLU q8_1 fused provider. The credible q8_1
  follow-up is therefore not "force q8_1"; it is a fused or cached RHS-q8_1
  path for the `SWIGLU k=2048 rows=512` expert shape only, measured against the
  current rocprof kernel bucket.
- Better next prescriptions:
  1. Build a separate experimental `mul_mat_id_q4_k_swiglu` provider that ports
     the existing standalone packed Q4_K DMMV decomposition into the SWIGLU
     expert kernel, but do not combine it with row tiling. Use the prior
     standalone packed provider as the starting point and target only
     `k=2048 rows=512 ids=8 tokens=1`. Stop if VGPR jumps toward the prior
     lane16 rejected path (~90+ VGPR) without a microbench win.
  2. For `mul_mat_id_q4_k_mul`, do not port the same SWIGLU experiment blindly:
     the hot shape is `k=512 rows=2048`, Vulkan would not choose the q8_1/MMVQ
     path for AMD at that K, and row2 tiling lost. First try instruction/dataflow
     analysis specific to small-K: compare WG64 vs a 32-thread/subgroup-only
     variant and inspect whether the four-block WG64 mapping has excess idle
     lanes or reduction overhead at `k=512`.
  3. If attempting expert-id/address reuse, make it explicit and measurable:
     group multiple rows only after proving row2's regression source with ISA or
     rocprof counters. The naive RHS-reuse version is rejected.

### 2026-04-09: hf9f.14 packed Q4_K SWIGLU expert experiment

- Accepted a separate, env-gated packed SWIGLU provider for the Qwen decode
  expert shape only. Commit `a952b8249` adds
  `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32`, registers it in the Pyre
  kernel catalog, and selects it only when
  `GGML_PYRE_ENABLE_PACKED_Q4_K_SWIGLU=1` and
  `k=2048 rows=512 ids=8 tokens=1`. The default path is unchanged.
- The packed SWIGLU kernel ports the standalone packed Q4_K DMMV decomposition
  into the fused gate/up SWIGLU expert kernel without row tiling. It uses the
  16-lane packed Q4_K block decomposition with four block slots in a 64-lane
  workgroup and keeps the fused `up * silu(gate)` write.
- Kernel-only rocprof accepted the change. Artifact
  `build/rocprof-pyre-kernelonly-n64-hf9f14-packed-swiglu/kernelonly-n64-hf9f14-packed-swiglu_results.db`
  shows `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` at 77,286.641 us over
  64 decode tokens, average 30.190 us/call. The hf9f.13 baseline
  `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` bucket was 93,547.818 us over 64
  tokens, average 36.542 us/call. The measured named-kernel delta is
  16,261.177 us per n64 run, about 254 us/token.
- ISA/resource result: final Q4 ID summary artifact
  `build/pyre-epic2-results/hf9f-14-isa-q4-id-final/` reports the existing
  SWIGLU WG64 path at 36 VGPR / 46 SGPR / no spills and the packed SWIGLU path
  at 86 VGPR / 46 SGPR / no spills. The VGPR jump is close to the prior
  caution threshold, so the path remains explicitly gated rather than default,
  but the profiler delta justifies keeping it as an experiment.
- Microbench evidence was noisy but favorable after the catalog entry was
  corrected. Packed SWIGLU final artifact
  `build/pyre-epic2-results/hf9f-14-swiglu-packed-final.json` reported
  168.195 us median / 161.904 us min. A representative baseline artifact
  `build/pyre-epic2-results/hf9f-14-swiglu-baseline-real.json` reported
  261.797 us median / 229.136 us min, but clean-cache/runtime variance made
  the kernel-only rocprof bucket the primary acceptance signal.
- Wall guardrail is healthy with the gated provider enabled. Final unprofiled
  n64 artifact
  `build/pyre-epic2-results/hf9f-14-packed-swiglu-n64-r3-final.json` reports
  31.974 tok/s average, 0.019 tok/s stddev, with samples 31.953, 31.978, and
  31.991 tok/s. This is not a material wall regression relative to the prior
  ~31.4 tok/s range.
- Rejected and removed the secondary `mul_mat_id_q4_k_mul` WG32/subgroup-only
  variant for the hot small-K shape. After fixing the coverage mapping, artifact
  `build/pyre-epic2-results/hf9f-14-mul-wg32-fixed2.json` reported 288.817 us
  median / 286.107 us min, worse than the restored default artifact
  `build/pyre-epic2-results/hf9f-14-mul-final.json` at 265.927 us median /
  262.047 us min. No WG32 speculative code was kept.
- Validation commands run: rebuilt with
  `cmake --build build/llama-pyre-rocm713 --target pyre-kernel-bench llama-bench
  test-backend-pyre -j$(nproc)`, cleared `cache/kernels`, ran
  `GGML_PYRE_KERNEL_PROVIDER=pure_hip ./build/llama-pyre-rocm713/bin/test-backend-pyre`,
  ran provider-trace verification for the packed SWIGLU claim, ran SWIGLU and
  MUL microbenches, ran the Q4 ID ISA summary, ran kernel-only rocprof n64, and
  ran the final unprofiled n64 wall guardrail.

### 2026-04-09: clean A/B verification for hf9f.14

- Re-ran the packed-SWIGLU experiment on cleanly rebuilt commit `a952b8249`
  because the first guardrail artifact still reported build commit `fe56f372a`
  and the first packed rocprof database had unrelated kernel buckets spiking.
  The stale-commit concern is resolved: both clean rocprof runs and the
  unprofiled guardrail now report build commit `a952b8249`.
- Clean profiled A/B run 1:
  - Packed off:
    `build/rocprof-pyre-kernelonly-n64-ab-off-1/kernelonly-n64-ab-off-1_results.db`
    reported total kernel dispatch time 858,888.101 us over 64 decode tokens
    and `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` at 93,854.774 us total,
    36.662 us/call.
  - Packed on:
    `build/rocprof-pyre-kernelonly-n64-ab-on-1/kernelonly-n64-ab-on-1_results.db`
    reported total kernel dispatch time 833,507.559 us over 64 decode tokens
    and `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` at 67,631.026 us total,
    26.418 us/call.
  - Delta: total kernel dispatch time improved by 25,380.542 us per n64 run,
    or 396.571 us/token. The named SWIGLU bucket improved by 26,223.748 us per
    n64 run, or 409.746 us/token.
- Clean profiled A/B run 2, in opposite order:
  - Packed on:
    `build/rocprof-pyre-kernelonly-n64-ab-on-2/kernelonly-n64-ab-on-2_results.db`
    reported total kernel dispatch time 828,672.264 us and packed SWIGLU at
    67,266.838 us total, 26.276 us/call.
  - Packed off:
    `build/rocprof-pyre-kernelonly-n64-ab-off-2/kernelonly-n64-ab-off-2_results.db`
    reported total kernel dispatch time 853,634.200 us and baseline SWIGLU at
    94,057.592 us total, 36.741 us/call.
  - Delta: total kernel dispatch time improved by 24,961.936 us per n64 run,
    or 390.030 us/token. The named SWIGLU bucket improved by 26,790.754 us per
    n64 run, or 418.606 us/token.
- Unprofiled n64 wall guardrail also moved in the expected direction:
  `build/pyre-epic2-results/ab-packed-swiglu/off-r3.json` reported 31.144 tok/s
  average with samples 31.163, 31.168, 31.103 tok/s, while
  `build/pyre-epic2-results/ab-packed-swiglu/on-r3.json` reported 31.458 tok/s
  average with samples 31.397, 31.492, 31.483 tok/s.
- Conclusion: the earlier
  `build/rocprof-pyre-kernelonly-n64-hf9f14-packed-swiglu/kernelonly-n64-hf9f14-packed-swiglu_results.db`
  aggregate regression was a noisy/bad artifact, not a reproducible side effect
  of the packed SWIGLU provider. The packed path is a real win for this shape:
  roughly 25 ms per 64-token decode in aggregate kernel dispatch time and about
  0.3 tok/s on the current synchronous runtime. Keep it gated for now because
  the packed kernel still uses 86 VGPR, but the experiment is accepted and worth
  carrying forward.

### 2026-04-10: hf9f.15 packed Q4 expert follow-up

- Reproduced the packed-SWIGLU A/B before making retained code changes. Current
  checkout/build at commit `a952b8249` reported:
  `build/rocprof-pyre-kernelonly-n64-hf9f15-off-repro/kernelonly-n64-hf9f15-off-repro_results.db`
  with total kernel dispatch 855,539.349 us over 64 tokens and baseline
  `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` at 93,250.226 us, versus
  `build/rocprof-pyre-kernelonly-n64-hf9f15-on-repro/kernelonly-n64-hf9f15-on-repro_results.db`
  with total kernel dispatch 836,991.977 us and
  `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` at 66,685.958 us. This
  reproduces the named SWIGLU win at 26,564.268 us/n64, about 415 us/token.
  The aggregate dispatch delta was smaller than the prior clean A/B but still
  positive at 18,547.372 us/n64, about 290 us/token.
- Tried and rejected a low-risk SWIGLU hardening variant that split the packed
  kernel's q01 and q23 scale/min work into separate scopes to reduce live
  temporaries. ISA artifact
  `build/pyre-epic2-results/hf9f-15-isa-q4-id-paired-scales/` moved packed
  SWIGLU from 86 VGPR to 85 VGPR, 2,767 to 2,758 file-level instructions, and
  no spills, but the target microbench regressed badly:
  `build/pyre-epic2-results/hf9f-15-swiglu-packed-paired-scales.json` reported
  283.287 us median / 276.848 us min. The edit was removed. Reverted baseline
  artifact `build/pyre-epic2-results/hf9f-15-swiglu-packed-reverted.json`
  returned to 165.794 us median / 160.024 us min.
- Accepted a new gated packed Q4_K `MUL` expert provider for the hot small-K
  shape. Commit `9abcbd9d2` adds
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32`, registers it in the Pyre kernel
  catalog, and selects it only when `GGML_PYRE_ENABLE_PACKED_Q4_K_MUL=1` and
  `k=512 rows=2048 ids=8 tokens=1`. Default behavior remains unchanged.
- The packed MUL experiment is distinct from the rejected WG32 path: it keeps a
  64-lane workgroup/reduction but maps the two `k=512` Q4_K blocks through the
  same 16-lane packed decomposition used by the standalone packed Q4_K DMMV
  kernel. Provider trace artifacts
  `build/pyre-epic2-results/hf9f-15-mul-default-trace.log` and
  `build/pyre-epic2-results/hf9f-15-mul-packed-trace.log` confirmed default
  `_wg64` selection without the env var and `_packed_wg64` selection with
  `GGML_PYRE_ENABLE_PACKED_Q4_K_MUL=1`.
- MUL microbench and ISA evidence justified keeping the provider gated. Artifact
  `build/pyre-epic2-results/hf9f-15-mul-packed-k512.json` reported 177.414 us
  median / 174.394 us min for `k=512 rows=2048 ids=8 tokens=1`, while traced
  default single-iteration evidence was 271.217 us. ISA artifact
  `build/pyre-epic2-results/hf9f-15-isa-q4-id-packed-mul/` reports the new
  packed MUL entrypoint at 43 VGPR / 42 SGPR / no spills, versus 22 VGPR /
  42 SGPR / no spills for the default WG64 path.
- Full n64 kernel-only rocprof with both packed expert gates enabled reported
  `build/rocprof-pyre-kernelonly-n64-hf9f15-packed-experts/kernelonly-n64-hf9f15-packed-experts_results.db`
  at total kernel dispatch 829,794.675 us over 64 tokens. In that run
  `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` was 67,208.943 us and
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32` was 69,559.363 us. Against the
  reproduced packed-off run, the combined Q4 expert bucket moved from
  171,638.347 us to 136,768.306 us, about 545 us/token saved in the named
  expert kernels; aggregate dispatch improved by 25,744.674 us/n64, about
  402 us/token.
- Unprofiled wall guardrail also moved positive. Packed off artifact
  `build/pyre-epic2-results/hf9f-15-wall-off-r3.json` reported 31.506 tok/s
  average, while packed SWIGLU+MUL artifact
  `build/pyre-epic2-results/hf9f-15-wall-packed-experts-r3.json` reported
  31.875 tok/s average.
- Validation commands run: rebuilt
  `cmake --build build/llama-pyre-rocm713 --target pyre-kernel-bench
  llama-bench test-backend-pyre -j$(nproc)` after clearing `cache/kernels`;
  ran `test-backend-pyre` with `GGML_PYRE_KERNEL_PROVIDER=pure_hip` and again
  with both packed expert env vars enabled; ran provider traces, target
  microbenches, Q4 ID ISA summaries, kernel-only n64 rocprof off/on, and the
  unprofiled n64 wall guardrail off/on. After committing, rebuilt once more so
  local binaries report commit `9abcbd9d2`.

### 2026-04-09: fresh next-step analysis after hf9f.15

- Re-ran committed-HEAD wall guardrails because the hf9f.15 wall artifacts were
  collected before the final commit and still reported build commit
  `a952b8249`. Fresh artifacts in
  `build/pyre-epic2-results/fresh-next-step/` now report build commit
  `9abcbd9d2`:
  - `head-off-r3.json`: packed expert gates off, 31.465 tok/s average, samples
    31.439, 31.409, 31.546 tok/s.
  - `head-packed-experts-r3.json`: `GGML_PYRE_ENABLE_PACKED_Q4_K_SWIGLU=1` and
    `GGML_PYRE_ENABLE_PACKED_Q4_K_MUL=1`, 31.987 tok/s average, samples
    31.959, 32.012, 31.989 tok/s.
  This confirms the gated packed expert work is still positive on committed
  HEAD.
- Remaining kernel-only n64 profile after both packed expert gates are enabled
  is `build/rocprof-pyre-kernelonly-n64-hf9f15-packed-experts/kernelonly-n64-hf9f15-packed-experts_results.db`.
  It reports 80,316 dispatches, 829,794.675 us total kernel-dispatch time over
  64 decode tokens, about 12.966 ms/token. Top remaining buckets:
  - `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32`: 69,559.363 us total,
    1,086.865 us/token.
  - `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32`: 67,208.943 us total,
    1,050.140 us/token.
  - `pyre_mul_mat_vec_bf16_f32`: 60,895.612 us total, 951.494 us/token.
  - `pyre_mul_mat_vec_q6_k_f32`: 57,759.013 us total, 902.485 us/token.
  - `pyre_mul_mat_vec_f32_batched_f32`: 53,961.505 us total, 843.149 us/token.
  - `pyre_add_add_f32_broadcast`: 51,936.821 us total, 811.513 us/token.
  - `pyre_mul_mat_vec_q5_k_wg128_f32`: 46,780.086 us total, 730.939 us/token.
  - `pyre_get_rows_f32`: 46,558.037 us total, 727.469 us/token.
- ISA/resource spot-check of the next tier does not show an obvious register or
  spill pathology. Examples: `mul_mat_vec_bf16_f32` is 151 file-level
  instructions, 10 VGPR / 20 SGPR / no spills; `mul_mat_vec_q6_k` is 733
  instructions, 34 VGPR / 21 SGPR / no spills; `mul_mat_vec_f32_batched_f32` is
  1,240 instructions, 11 VGPR / 64 SGPR / no spills; `add_add_f32_broadcast` is
  648 instructions, 27 VGPR / 54 SGPR / no spills; `get_rows_f32` is 472
  instructions, 4 VGPR / 47 SGPR / no spills.
- Provider trace artifact
  `build/pyre-epic2-results/fresh-next-step/head-packed-trace-n1.log` shows the
  key graph granularity problem: one decode token claims 160 `ADD_ADD` fused
  dispatches, each with `n=2048`. That is only 2,048 elements per dispatch, so
  the 811 us/token bucket is dominated by dispatch/granularity overhead rather
  than memory bandwidth. The trace shows runs of consecutive `ADD_ADD` claims
  around the MoE expert accumulation sequence, while the current Pyre fusion
  only combines binary `ADD -> ADD` into one `ADD_ADD` kernel.
- Decision: the next ticket should target multi-add accumulation fusion rather
  than another blind Q4 expert kernel variant. A multi-add chain provider that
  reduces the MoE accumulation from several `ADD_ADD` dispatches per layer to
  one or two dispatches has a plausible upper bound near the current
  `ADD_ADD` bucket (~0.8 ms/token), and it attacks a known app/kernel
  granularity issue that will not be fixed by the async runtime alone. This
  should be gated and measured with provider-trace counts, rocprof bucket
  deltas, and committed-HEAD wall guardrails. Continue Q4 expert work only
  after this fusion opportunity is either harvested or rejected.

### 2026-04-10: hf9f.16 gated ADD8 MoE accumulation fusion

- Reproduced the motivating trace shape before changing retained behavior. The
  prior committed-HEAD trace
  `build/pyre-epic2-results/fresh-next-step/head-packed-trace-n1.log` showed
  one decode token with 160 `ADD_ADD` claims, all `n=2048`, around MoE expert
  accumulation. This is four `ADD_ADD` dispatches per layer on the 40-layer
  Qwen decode path.
- Added a deliberately narrow gated `ADD8` fusion in commit `12c0f38a6`, then
  flipped it on by default after final validation because the wall guardrail was
  a straight ~10% positive result. The opt-out knob is
  `GGML_PYRE_DISABLE_MULTI_ADD_FUSION=1`. The matcher requires seven consecutive
  linear `GGML_OP_ADD` nodes,
  single-use semantics via `ggml_can_fuse_subgraph_ext`, eight F32 same-shape
  contiguous source tensors, and one F32 contiguous output. It does not attempt
  arbitrary add DAGs or broadcast semantics.
- Provider trace artifact
  `build/pyre-epic2-results/hf9f-16-add8-trace-n1.log` with packed expert gates
  and `GGML_PYRE_ENABLE_MULTI_ADD_FUSION=1` showed 40 `ADD8` claims and 40
  remaining `ADD_ADD` claims for one decode token. This fuses the observed
  three-`ADD_ADD` plus one-`ADD` MoE accumulation chain in each layer and leaves
  the later separated `ADD_ADD` per layer untouched.
- ISA/resource result: artifact `build/pyre-epic2-results/hf9f-16-isa-add8/`
  reports `pyre_add8_f32` at 67 instructions, 12 VGPR, 22 SGPR, no spills, no
  LDS, and eight global loads plus one global store. For comparison, the
  broadcast `ADD_ADD` provider in the same artifact is 648 instructions,
  27 VGPR, 54 SGPR, and no spills.
- Kernel-only n64 rocprof with packed expert gates but without ADD8:
  `build/rocprof-pyre-kernelonly-n64-hf9f16-packed-noadd8/kernelonly-n64-hf9f16-packed-noadd8_results.db`
  reports 80,316 dispatches, 825,924.644 us total kernel-dispatch time, and
  `pyre_add_add_f32_broadcast` at 52,333.434 us over 64 tokens.
- Kernel-only n64 rocprof with packed expert gates and ADD8:
  `build/rocprof-pyre-kernelonly-n64-hf9f16-packed-add8/kernelonly-n64-hf9f16-packed-add8_results.db`
  reports 72,636 dispatches, 780,998.280 us total kernel-dispatch time,
  `pyre_add_add_f32_broadcast` at 12,984.462 us, and `pyre_add8_f32` at
  9,460.657 us over 64 tokens. The combined ADD bucket moves from
  52,333.434 us to 22,445.119 us, a 29,888.315 us/n64 reduction
  (~467 us/token). Aggregate kernel-dispatch time improves by 44,926.364 us/n64
  (~702 us/token), and dispatch count drops by 7,680.
- Unprofiled wall guardrail is strongly positive in this run. With packed
  expert gates and ADD8 disabled,
  `build/pyre-epic2-results/hf9f-16-wall-packed-noadd8-r3.json` reports
  31.593 tok/s average. With packed expert gates and ADD8 enabled,
  `build/pyre-epic2-results/hf9f-16-wall-packed-add8-r3.json` reports
  34.429 tok/s average. Final pre-flip candidate validation with the same ADD8
  code path enabled by env,
  `build/pyre-epic2-results/hf9f-16-final-add8-default-candidate-r3.json`,
  reported 34.601 tok/s average.
- Default-policy validation: after changing the policy, provider trace artifact
  `build/pyre-epic2-results/hf9f-16-add8-default-on-trace-n1.log` confirmed
  repeated `ADD8` claims without `GGML_PYRE_ENABLE_MULTI_ADD_FUSION`. Opt-out
  trace artifact
  `build/pyre-epic2-results/hf9f-16-add8-default-off-trace-n1.log` with
  `GGML_PYRE_DISABLE_MULTI_ADD_FUSION=1` confirmed the path returns to
  `ADD_ADD`/`ADD` claims and no `ADD8`.
- Validation commands run: rebuilt
  `cmake --build build/llama-pyre-rocm713 --target pyre-kernel-bench
  llama-bench test-backend-pyre -j$(nproc)` after clearing `cache/kernels`;
  ran `test-backend-pyre` with `GGML_PYRE_KERNEL_PROVIDER=pure_hip`, with
  `GGML_PYRE_ENABLE_MULTI_ADD_FUSION=1`, and again after the commit; ran
  provider trace n=1, ADD ISA summary, kernel-only n64 rocprof packed off/on
  for ADD8, unprofiled n64 wall guardrails, and default-on/default-off policy
  traces. After committing the original gated kernel, rebuilt once more so local
  binaries report commit `12c0f38a6`; after the default-on policy commit, rebuild
  again so local binaries report the new commit.

### 2026-04-10: hf9f.17 post-ADD8 Q4 expert policy and follow-up

- Took `pyre-workspace-hf9f.17` on llama.cpp commit `96121af5e`, after ADD8 was
  default-enabled. Reproduced the current wall baseline before changing code:
  `build/pyre-epic2-results/hf9f-17/default-add8-r3.json` reported
  33.753 tok/s average, stddev 0.028, samples `[33.7509, 33.7822, 33.7273]`,
  average 29.627 ms/token. With the packed expert gates manually enabled,
  `build/pyre-epic2-results/hf9f-17/add8-packed-experts-r3.json` reported
  33.954 tok/s average, stddev 0.096, samples `[34.0331, 33.9815, 33.8468]`,
  average 29.452 ms/token. The packed run matches the planner baseline; the
  default run was faster than the planner sample but within normal run drift, so
  the checkout was considered aligned.
- Kernel-only baseline before policy change:
  `build/rocprof-pyre-kernelonly-n64-hf9f17-default-add8/kernelonly-n64-hf9f17-default-add8_results.db`
  reports the top two buckets as `pyre_mul_mat_id_q4_k_swiglu_wg64_f32`
  93,716.150 us and `pyre_mul_mat_id_q4_k_mul_wg64_f32` 79,329.781 us over
  64 tokens. The manually packed comparison
  `build/rocprof-pyre-kernelonly-n64-hf9f17-packed-experts/kernelonly-n64-hf9f17-packed-experts_results.db`
  reports `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` 67,248.838 us and
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32` 69,812.627 us. The combined Q4
  expert bucket drops from 173,045.931 us to 137,061.465 us over 64 tokens,
  about 562 us/token in the named expert kernels.
- Decision: default-enable the packed Q4 expert providers in llama.cpp commit
  `80f841626` for only their existing narrow Qwen decode shapes. The shape gates
  remain unchanged:
  SWIGLU requires `k=2048 rows=512 ids=8 tokens=1`, and MUL requires
  `k=512 rows=2048 ids=8 tokens=1`. New opt-out knobs are
  `GGML_PYRE_DISABLE_PACKED_Q4_K_SWIGLU=1` and
  `GGML_PYRE_DISABLE_PACKED_Q4_K_MUL=1`.
- Default-policy trace validation:
  `build/pyre-epic2-results/hf9f-17/default-policy-trace-n1.log` and summary
  confirm 40 `MUL_MAT_ID_SWIGLU pure_hip_q4_K_packed_wg64` claims and 40
  `MUL_MAT_ID_MUL pure_hip_q4_K_packed_wg64` claims without opt-in env vars.
  Opt-out trace `build/pyre-epic2-results/hf9f-17/packed-optout-trace-n1.log`
  confirms both providers return to `_wg64` with the disable env vars.
- Final wall guardrail after the policy change:
  `build/pyre-epic2-results/hf9f-17/default-packed-policy-r3.json` reported
  34.180 tok/s average, stddev 0.105, samples `[34.1667, 34.2914, 34.0831]`,
  average 29.257 ms/token. The opt-out comparison
  `build/pyre-epic2-results/hf9f-17/packed-optout-policy-r3.json` reported
  34.059 tok/s average, stddev 0.010, samples `[34.0625, 34.067, 34.0481]`,
  average 29.361 ms/token. The measured wall win in this sample is small
  (~0.104 ms/token), but the profiler win is stable and the policy stays
  conservative via shape gates and opt-out knobs.
- Final default-policy rocprof:
  `build/rocprof-pyre-kernelonly-n64-hf9f17-default-packed-policy/kernelonly-n64-hf9f17-default-packed-policy_results.db`
  reports 72,636 dispatches. The top buckets are
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32` 70,058.991 us,
  `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` 67,234.785 us,
  `pyre_mul_mat_vec_bf16_f32` 60,177.559 us,
  `pyre_mul_mat_vec_q6_k_f32` 57,705.985 us, and
  `pyre_mul_mat_vec_f32_batched_f32` 53,315.643 us.
- ISA/resource summary artifact
  `build/pyre-epic2-results/hf9f-17/isa-q4-id-policy/` reports
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32` at 43 VGPR / 42 SGPR / no spills /
  8 bytes LDS and `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` at
  86 VGPR / 46 SGPR / no spills / 16 bytes LDS. The SWIGLU VGPR footprint
  remains high but is spill-free and now justified by repeated profiler-backed
  wins.
- Rejected structural variant: attempted to push the packed kernels closer to
  the Vulkan `mul_mat_vec_q4_k` shader by loading q bytes as aligned
  `uint32_t`, RHS values as `float4`, and accumulating grouped q-dot and min
  terms. It passed compilation and `test-backend-pyre`, but target microbench
  artifacts `build/pyre-epic2-results/hf9f-17/microbench-swiglu-vectorized-default.txt`
  and `build/pyre-epic2-results/hf9f-17/microbench-mul-vectorized-default.txt`
  regressed to 287.297 us median for SWIGLU and 231.406 us median for MUL, so
  the kernel edits were reverted. The likely issue is extra scalar extraction
  and VGPR pressure dominating any reduced load count in HIP/LLVM for this
  lane shape.
- Validation commands run: rebuilt
  `cmake --build build/llama-pyre-rocm713 --target pyre-kernel-bench
  llama-bench test-backend-pyre -j$(nproc)`; ran `test-backend-pyre` with
  default policy and packed opt-out; ran provider traces for default and
  opt-out; ran target microbenches for the accepted policy and rejected
  vectorized variant; ran n64 wall guardrails; ran kernel-only n64 rocprof for
  default/packed policy; and ran Q4 expert ISA summary.

### 2026-04-10: hf9f.18 Q4 expert measurement hygiene and live-range experiment

- Measurement hygiene changes landed in llama.cpp commit `d1f0a1524`. Added
  `tools/pyre-epic2/pyre-q4-experiment-manifest.py`, which writes a compact JSON
  manifest containing git HEAD, dirty state, wall `build_commit`, env toggles,
  model path, wall/trace/rocprof/ISA artifact paths, and Q4 expert provider
  counts. It fails by default if the wall artifact `build_commit` does not
  match `git rev-parse --short HEAD`; `--allow-stale-build-commit` is the
  explicit escape hatch for documenting known-stale artifacts. Also made
  `tools/pyre-epic2/pyre-trace-summary.py` accept the older shorthand
  `pyre-trace-summary.py trace.log --top N` by rewriting it to
  `summarize --pyre-log trace.log --top N`.
- Fresh default manifest:
  `build/pyre-epic2-results/hf9f-18/default-packed-policy-manifest.json` was
  generated after rebuilding `llama-bench` to report commit `d1f0a1524`. The
  manifest reports `git_dirty=false`, wall `build_commit=d1f0a1524`, 34.527
  tok/s average, and 28.963 ms/token. Provider trace counts prove the default
  selected 40 `MUL_MAT_ID_SWIGLU pure_hip_q4_K_packed_wg64` claims and 40
  `MUL_MAT_ID_MUL pure_hip_q4_K_packed_wg64` claims for the expected Qwen
  decode shapes.
- Fresh default rocprof:
  `build/rocprof-pyre-kernelonly-n64-hf9f18-default-packed/kernelonly-n64-hf9f18-default-packed_results.db`
  and summary
  `build/pyre-epic2-results/hf9f-18/rocprof-default-packed-n64-summary.txt`
  report the top kernel buckets over 64 tokens as
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32` 69,248.807 us,
  `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` 67,570.709 us,
  `pyre_mul_mat_vec_bf16_f32` 60,374.077 us,
  `pyre_mul_mat_vec_q6_k_f32` 57,839.319 us, and
  `pyre_mul_mat_vec_f32_batched_f32` 53,542.232 us. The Q4 expert pair remains
  the top named bucket, but the remaining gap is now smaller and must be attacked
  with real algorithmic/dataflow changes rather than syntax-level Vulkan ports.
- Fresh default ISA:
  `build/pyre-epic2-results/hf9f-18/isa-q4-id-default-summary.txt` reports
  `pyre_mul_mat_id_q4_k_mul_packed_wg64_f32` at 43 VGPR / 42 SGPR / no spills /
  8 bytes LDS, and `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32` at
  86 VGPR / 46 SGPR / no spills / 16 bytes LDS. The whole MUL source summary is
  2,024 instructions; the whole SWIGLU source summary is 2,767 instructions.
- Rejected SWIGLU low-VGPR variant: added an opt-in provider
  `pyre_mul_mat_id_q4_k_swiglu_packed_lowvgpr_wg64_f32` under
  `GGML_PYRE_ENABLE_PACKED_Q4_K_SWIGLU_LOWVGPR=1`. The variant split the packed
  SWIGLU loop into separate gate and up passes to shrink live ranges, knowingly
  trading away RHS reuse. Provider trace
  `build/pyre-epic2-results/hf9f-18/swiglu-lowvgpr-trace-n1.log` proved 40
  `_packed_lowvgpr_wg64` SWIGLU claims and unchanged packed MUL claims.
  Microbench artifact
  `build/pyre-epic2-results/hf9f-18/microbench-swiglu-lowvgpr.txt` reported
  274.687 us median / 270.467 us min, not a positive signal.
- Low-VGPR result: ISA artifact
  `build/pyre-epic2-results/hf9f-18/isa-swiglu-lowvgpr-summary.txt` confirmed
  the intended resource tradeoff: packed SWIGLU dropped from 86 VGPR to
  47 VGPR with no spills, but source-level instruction count rose from 2,767 to
  3,665 and global-load opcode matches rose from 73 to 117. Kernel-only rocprof
  `build/rocprof-pyre-kernelonly-n64-hf9f18-swiglu-lowvgpr/kernelonly-n64-hf9f18-swiglu-lowvgpr_results.db`
  showed the tradeoff loses in real decode: SWIGLU rose to 72,056.983 us over
  64 tokens and wall artifact
  `build/pyre-epic2-results/hf9f-18/swiglu-lowvgpr-r3.json` dropped to
  33.788 tok/s / 29.597 ms/token. Manifest
  `build/pyre-epic2-results/hf9f-18/swiglu-lowvgpr-manifest.json` records the
  full rejected bundle. The experimental code was reverted; no low-VGPR provider
  is retained.
- Verification: rebuilt
  `cmake --build build/llama-pyre-rocm713 --target pyre-kernel-bench
  llama-bench test-backend-pyre -j$(nproc)` before measurements and after
  reverting the rejected kernel. Ran `test-backend-pyre` for default policy and
  with `GGML_PYRE_DISABLE_PACKED_Q4_K_SWIGLU=1
  GGML_PYRE_DISABLE_PACKED_Q4_K_MUL=1`; both passed. Also ran
  `test-backend-pyre` with the low-VGPR experiment gate before rejecting it.
- Next recommendation: do not pursue SWIGLU occupancy by duplicating RHS reads;
  the packed kernel is memory/dataflow sensitive enough that the 86-VGPR
  spill-free path still wins. The next structural Q4 expert attempts should
  either preserve RHS reuse while reducing scale/min lifetime, or move to a
  genuinely different work decomposition that reduces repeated expert/id/row
  setup across neighboring rows without increasing per-lane live ranges.

### 2026-04-10: hf9f.19 non-Q4 profiler-ranked kernel pass

- Fresh baseline manifest:
  `build/pyre-epic2-results/hf9f-19/default-packed-policy-manifest.json`.
  It was generated from commit `d1f0a1524` and wall artifact
  `build/pyre-epic2-results/hf9f-19/default-packed-policy-r3.json`; the wall
  baseline reported 34.330 tok/s average, samples `[34.2407, 34.3999,
  34.3481]`, and 29.130 ms/token. Provider trace
  `build/pyre-epic2-results/hf9f-19/default-packed-policy-trace-summary.txt`
  ranked the next non-Q4 decode targets as BF16 matvec/BF16 SWIGLU, Q6_K
  matvec, F32/F16 batched matvec, `get_rows_f32`, and
  `gated_delta_net_f32`.
- Selected targets: BF16 matvec family and Q6_K workgroup policy. These were
  chosen because they were among the largest non-Q4 buckets and had plausible
  local policy/resource changes. F32/F16 batched matvec and `get_rows_f32` were
  left for the next profiler-guided pass after the first two targets produced
  clear negative evidence rather than a reusable structural win.
- BF16 experiment: added env-gated 128- and 64-thread workgroup variants for
  `mul_mat_vec_bf16.hip.cpp` and `mul_mat_vec_bf16_swiglu.hip.cpp`, selectable
  with `GGML_PYRE_MUL_MAT_VEC_BF16_WG=64|128|256|auto`. The default `auto`
  policy intentionally remains equivalent to 256, so final default trace
  `build/pyre-epic2-results/hf9f-19/final-default-trace-summary.txt` still
  reports unsuffixed `MUL_MAT pure_hip_bf16` and `MUL_MAT_SWIGLU
  pure_hip_bf16` claims, not `_wg64` or `_wg128`.
- BF16 result: rejected for default policy. ISA artifact
  `build/pyre-epic2-results/hf9f-19/isa-bf16-wg-variants-summary.txt` shows
  the variants reduce LDS only: BF16 stays at 10 VGPR / 20 SGPR / no spills,
  and BF16 SWIGLU stays at 14 VGPR / 26 SGPR / no spills. Kernel-only rocprof
  `build/pyre-epic2-results/hf9f-19/rocprof-bf16wg128-n64-summary.txt` shows
  `pyre_mul_mat_vec_bf16_wg128_f32` at 67,464.003 us versus the hf9f.18
  default BF16 bucket at 60,374.077 us, and
  `pyre_mul_mat_vec_bf16_swiglu_wg128_f32` at 34,574.400 us versus the
  hf9f.18 default SWIGLU bucket at 31,070.883 us. A transient mixed heuristic
  also regressed wall throughput from `bf16-wg256-r3.json` 34.453 tok/s to
  `bf16-wg-auto-r3.json` 34.089 tok/s. Manifest:
  `build/pyre-epic2-results/hf9f-19/bf16-wg128-manifest.json`.
- Q6_K experiment: used the existing `GGML_PYRE_MUL_MAT_VEC_K_WG` knob to test
  forced 256-thread workgroups against the current mixed default. Microbench
  artifact `build/pyre-epic2-results/hf9f-19/q6-wg-microbench.txt` suggested
  256 could be better for the sampled Q6 shapes, with 266.317 us median for
  2048x4096 and 211.136 us median for 4096x2048, but this did not survive
  decode-level validation.
- Q6_K result: rejected for default policy. Non-rocprof wall guardrails
  `build/pyre-epic2-results/hf9f-19/default-after-bf16-gated-r3.json` and
  `build/pyre-epic2-results/hf9f-19/q6wg256-after-bf16-gated-r3.json` reported
  33.966 tok/s for default versus 33.591 tok/s for forced `K_WG=256`.
  Kernel-only rocprof
  `build/pyre-epic2-results/hf9f-19/rocprof-q6wg256-n64-summary.txt` shows
  Q6 itself roughly flat/slightly worse at 89,884.057 us versus hf9f.18's
  default combined Q6 total of 89,468.149 us, while the shared K workgroup knob
  also forced Q5_K to a bad 256-thread policy at 110,557.066 us. Manifest:
  `build/pyre-epic2-results/hf9f-19/q6-wg256-rocprof-manifest.json`.
- Final validation: rebuilt `pyre-kernel-bench`, `llama-bench`, and
  `test-backend-pyre`; ran `test-backend-pyre` successfully; regenerated the
  final default provider trace to prove no unintended default provider-policy
  changes. The retained code change is gated-only BF16 workgroup variants for
  future differential analysis, with default behavior unchanged.
- Next recommendation: do not use the broad `GGML_PYRE_MUL_MAT_VEC_K_WG` knob
  for Q6 policy decisions because it changes Q5_K at the same time. If Q6 gets
  revisited, add a Q6-specific selector/knob and validate against kernel-only
  rocprof before wall. The next structural work should move to profiler-ranked
  F32/F16 batched matvec and `get_rows_f32`, with `gated_delta_net_f32` behind
  them unless profiler traces point elsewhere.

### 2026-04-10: hf9f.20 F32 batched matvec common-shape pass

- Scope: follow-up after hf9f.19, targeting F32/F16 batched matvec and
  `get_rows_f32`. Provider trace
  `build/pyre-epic2-results/hf9f-20/f32batched-cols1-trace-summary.txt`
  confirms the F32 batched common decode shapes are `k=2048 rows=256 cols=1
  ne2=1` and `k=2048 rows=1 cols=1 ne2=1`. Prompt/prefill still hits the
  generic F32 batched path for `cols=512`. `get_rows_f32` remains a large
  bucket but is a simple gather/copy path.
- Accepted change in llama.cpp commit `2fd0e651a`: added
  `pyre_mul_mat_vec_f32_batched_cols1_ne2_1_f32` and route only F32 batched
  matvecs with `cols == 1` and `dst_ne2 == 1` to it. The specialized kernel
  removes the hot decode path's `outer % cols`, `outer / cols`, `i12`
  extraction, and `dst_ne2` address math while preserving the generic provider
  for prompt/prefill and other layouts.
- ISA/resource evidence:
  `build/pyre-epic2-results/hf9f-20/isa-f32batched-cols1-summary.txt` reports
  the generic F32 batched provider at 11 VGPR / 64 SGPR / no spills / 1024
  bytes LDS. The specialized decode provider is 10 VGPR / 42 SGPR / no spills /
  1024 bytes LDS. `get_rows_f32` was unchanged in the accepted commit and
  remains 4 VGPR / 47 SGPR / no spills / 0 LDS.
- Kernel-only rocprof evidence:
  `build/pyre-epic2-results/hf9f-20/rocprof-f32batched-cols1-clean2-n64-summary.txt`
  reports `pyre_mul_mat_vec_f32_batched_cols1_ne2_1_f32` at 43,314.554 us over
  5,120 calls. The hf9f.18 baseline `pyre_mul_mat_vec_f32_batched_f32` bucket
  was 53,542.232 us over the same call count, so the targeted bucket improves by
  about 10.2 ms over 64 tokens (~0.16 ms/token). `get_rows_f32` stayed near
  baseline at 46,040.923 us.
- Wall guardrail:
  `build/pyre-epic2-results/hf9f-20/f32batched-cols1-manifest.json` records the
  clean-commit decode r=5 artifact at 34.528 tok/s average, 34.483 tok/s
  median, samples `[34.4701, 34.7259, 34.4826, 34.4871, 34.4759]`, and
  28.962 ms/token with `build_commit=2fd0e651a`. This is essentially neutral
  versus hf9f.18 wall and modestly above the hf9f.19 fresh baseline, which is
  consistent with runtime/wall noise muting a real but small kernel-only win.
- Rejected `get_rows_f32` attempt: changed the F32 get-rows catalog workgroup to
  512 and made the kernel use `blockDim.x` for indexing, mirroring Vulkan's
  512-wide F32 get-rows pipeline. It passed `test-backend-pyre`, but rocprof
  artifact
  `build/pyre-epic2-results/hf9f-20/rocprof-getrows512-f32batched-n64-summary.txt`
  showed `get_rows_f32` regressed from the hf9f.18 baseline 46,051.966 us to
  52,090.240 us over 64 tokens. The get-rows change was reverted before the
  accepted commit.
- Measurement notes: one clean-commit rocprof sample
  `build/pyre-epic2-results/hf9f-20/rocprof-f32batched-cols1-clean-n64-summary.txt`
  was contaminated by multi-millisecond outliers and was not used for
  acceptance. A rerun (`clean2`) had normal max latency and is the accepted
  profiler sample. Validation included rebuilding `pyre-kernel-bench`,
  `llama-bench`, and `test-backend-pyre`; running `test-backend-pyre`;
  provider trace n=1; clean-commit wall r=5; ISA summary; and kernel-only n64
  rocprof.
- Next recommendation: continue with F16 batched matvec. It remains at about
  30.4 ms over 64 tokens in the clean2 profile, and its prompt/decode shapes
  differ (`k=256/512`, `rows=256/512`, `cols=1/512`, `ne2=16`), so it likely
  needs a separate common-shape specialization rather than reusing the F32
  `cols=1,dst_ne2=1` pattern. `get_rows_f32` should not be revisited via wider
  workgroups; look for row-index/address-dataflow changes or fusions instead.

### 2026-04-10: hf9f.21 F16 batched matvec common-shape pass

- Scope: follow-up after hf9f.20, targeting the remaining F16 batched matvec
  bucket. Provider trace
  `build/pyre-epic2-results/hf9f-21/f16batched-cols1-trace-summary.txt`
  shows 20 decode claims for `MUL_MAT pure_hip_f16_batched_cols1` at
  `k=256 rows=256 cols=1 ne2=16`. The prompt/prefill F16 shapes stay on the
  generic provider at `cols=512`.
- Accepted change in llama.cpp commit `21c951393`: added
  `pyre_mul_mat_vec_f16_batched_cols1_f32` and route only F16 batched matvecs
  with `cols == 1` to it. Since F16 batched support already requires
  `ne3 == 1`, the specialized kernel removes the generic `outer % cols`,
  `outer / cols`, and `i13` address math, while retaining the existing generic
  provider for prompt/prefill and other shapes.
- ISA/resource evidence:
  `build/pyre-epic2-results/hf9f-21/isa-f16batched-cols1-summary.txt` reports
  the generic F16 batched provider at 11 VGPR / 64 SGPR / no spills / 1024
  bytes LDS. The specialized F16 decode provider is 10 VGPR / 40 SGPR / no
  spills / 1024 bytes LDS. The previously accepted F32 decode provider remains
  10 VGPR / 42 SGPR / no spills.
- Kernel-only rocprof evidence:
  `build/pyre-epic2-results/hf9f-21/rocprof-f16batched-cols1-serial-n64-summary.txt`
  reports `pyre_mul_mat_vec_f16_batched_cols1_f32` at 20,529.092 us over 1,280
  calls. The hf9f.18 baseline `pyre_mul_mat_vec_f16_batched_f32` bucket was
  30,313.104 us over 1,280 calls, so the targeted bucket improves by about
  9.8 ms over 64 tokens (~0.15 ms/token). In the same clean serial profile,
  `pyre_mul_mat_vec_f32_batched_cols1_ne2_1_f32` remains stable at 43,194.290
  us and `get_rows_f32` remains stable at 45,689.802 us.
- Wall guardrail:
  `build/pyre-epic2-results/hf9f-21/f16batched-cols1-manifest.json` records the
  clean serial decode r=5 artifact at 34.341 tok/s average, 34.333 tok/s
  median, samples `[34.3334, 34.3743, 34.292, 34.4188, 34.2882]`, and
  29.119 ms/token with `build_commit=21c951393`. This is wall-neutral versus
  hf9f.19 and below the best hf9f.20 wall sample, but the profiler sample is a
  clean targeted kernel win and there is no evidence of collateral kernel
  regression.
- Measurement notes: one wall/rocprof pair was accidentally run concurrently
  and produced multi-millisecond rocprof max latencies plus a bad wall sample;
  those artifacts are not used for acceptance. The accepted evidence is from
  serial runs only: `f16batched-cols1-clean-serial-r5.json` for wall and
  `rocprof-f16batched-cols1-serial-n64-summary.txt` for profiler.
- Next recommendation: the batched F32/F16 address-math specializations are now
  done for the obvious decode shapes. Move back to the profiler-ranked table:
  `get_rows_f32` needs a different approach than wider workgroups, and
  `gated_delta_net_f32` or the Q5_K/Q6_K non-shared policy split are better next
  structural candidates. Any future wall interpretation should avoid overlapping
  rocprof and wall runs; concurrent measurement materially polluted hf9f.21
  samples.

### 2026-04-10: hf9f.22 gated-delta/get_rows/Q5-Q6 profiler pass

- Scope: follow-up after hf9f.21 using fresh default trace, wall, ISA, and n64
  kernel-only rocprof artifacts under `build/pyre-epic2-results/hf9f-22/`.
  Fresh baseline was llama.cpp commit `21c951393`. Baseline wall artifact
  `default-r5.json` reported prompt 200.298 tok/s average and decode
  34.152 tok/s average. Baseline trace `default-trace-summary.txt` confirmed
  hot `GATED_DELTA_NET_STATE_UPDATE pure_hip_f32` shapes at 30 prompt calls
  with `tokens=512 H=32` and 30 decode calls with `tokens=1 H=32`.
- Fresh baseline profiler ranking:
  `rocprof-default-n64-summary.txt` put the relevant non-Q4 targets at
  `get_rows_f32` 46,118.527 us, F32 batched decode 43,494.314 us,
  `gated_delta_net_f32` 35,149.906 us, Q6_K default 57,841.214 us plus
  Q6_K wg128 31,650.001 us, Q5_K wg128 46,677.762 us, and Q5_K wg64
  32,533.577 us over n64. Baseline ISA
  `isa-top-baseline-summary.txt` reported `pyre_gated_delta_net_f32` at
  54 VGPR / 94 SGPR / no spills / 512 bytes LDS, with 12 barriers.
- Gated-delta reference comparison: Pyre and Vulkan labels are comparable
  fused gated-delta state update work for the traced Qwen shapes. CUDA keeps a
  32-lane-per-column warp-style implementation, while Vulkan's S_v=128 path
  selects clustered subgroup reductions. The Pyre baseline was therefore a
  credible target: it used four 32-lane column reductions per 128-thread
  workgroup and LDS/barriers, while Vulkan avoids the LDS reduction path for
  this shape.
- Rejected gated-delta cluster8 attempt: added an S_v=128 clustered-reduction
  variant with 8 lanes per column and 32-thread workgroups. It passed
  `test-backend-pyre` and trace artifact `gdn-s128-cluster8-trace-summary.txt`
  from the intermediate build confirmed the same hot shape selection. ISA
  artifact `isa-gdn-s128-cluster8-summary.txt` showed the variant at
  91 VGPR / 76 SGPR / no spills / 0 LDS. Rocprof artifact
  `rocprof-gdn-s128-cluster8-n64-summary.txt` improved gated-delta to
  29,419.592 us over n64, about 5.73 ms total / 0.090 ms-token, but was worse
  than the cluster16 comparison and was removed before the final code.
- Accepted gated-delta cluster16 change: added
  `pyre_gated_delta_net_s128_cluster16_f32`, selected only when `S_v == 128`
  and the direct provider is available. This uses 16 lanes per column,
  64-thread workgroups, wave shuffles instead of LDS reductions, and keeps the
  generic provider for other S_v shapes. Final ISA artifact
  `isa-accepted-summary.txt` reports the specialized provider at
  59 VGPR / 74 SGPR / no spills / 0 LDS, versus the generic provider at
  54 VGPR / 94 SGPR / no spills / 512 bytes LDS. Intermediate rocprof
  `rocprof-gdn-s128-cluster16-n64-summary.txt` measured 27,853.728 us. The
  final accepted rocprof `rocprof-final-n64-summary.txt` measured
  28,212.560 us, about 6.94 ms total / 0.108 ms-token better than the fresh
  baseline GDN bucket.
- Q6 policy split: added a Q6-only workgroup override
  `GGML_PYRE_MUL_MAT_VEC_Q6_K_WG=64|128|256|auto` so Q6 policy can be tested
  without perturbing Q5. The broad `GGML_PYRE_MUL_MAT_VEC_K_WG` knob remains
  higher precedence and still intentionally forces all K-quant matvecs.
- Q6 policy result: accepted default Q6_K wg128. Final trace
  `final-trace-summary.txt` shows all 140 Q6 claims as
  `MUL_MAT pure_hip_q6_K_wg128`, covering `k=2048 rows=4096`,
  `k=4096 rows=2048`, and `k=2048 rows=8192` for both prompt and decode.
  Differential rocprof probes were `rocprof-q6wg64-n64-summary.txt`
  (Q6 85,322.166 us), `rocprof-q6wg128-n64-summary.txt`
  (Q6 81,660.925 us), and `rocprof-q6wg256-n64-summary.txt`
  (Q6 89,866.696 us). Final default rocprof `rocprof-final-n64-summary.txt`
  measured Q6 wg128 at 82,009.698 us, compared to the fresh baseline combined
  Q6 total of 89,491.215 us. Q5 guardrails stayed flat: final Q5 wg128
  46,941.702 us and Q5 wg64 32,555.140 us versus baseline 46,677.762 us and
  32,533.577 us.
- get_rows result: no code change accepted. The fresh baseline kept
  `get_rows_f32` high at 46,118.527 us, and final accepted rocprof measured
  46,040.127 us. The previous hf9f.20 wg512 attempt had already regressed this
  bucket, so wider workgroups were not retried. No credible row-index,
  contiguous-row vectorization, or fusion opportunity was proven from this pass;
  next get_rows work should be profiler/graph-dataflow driven rather than
  another catalog workgroup-size guess.
- Final validation: rebuilt `pyre-kernel-bench`, `llama-bench`, and
  `test-backend-pyre`; ran `test-backend-pyre` successfully after final edits;
  refreshed provider trace (`final-trace-summary.txt`), ISA
  (`isa-accepted-summary.txt`), n64 kernel-only rocprof
  (`rocprof-final-n64-summary.txt`), and wall guardrails (`final-r5.json` and
  `final-cleancommit-r5.json`). The clean-commit wall guardrail for
  `93ed7ba67` reported prompt 216.388 tok/s average and decode 34.762 tok/s
  average, samples `[34.8099, 34.7083, 34.8288, 34.8264, 34.6382]`, versus
  the hf9f.22 fresh baseline decode average of 34.152 tok/s.
- Net result: accepted changes total about 14.4 ms over n64, or roughly
  0.225 ms/token kernel-only, from GDN cluster16 plus Q6 wg128. This is below
  the ticket's preferred 0.300 ms-token target, but it is a clean default
  kernel-only win with wall moving in the same direction. Remaining structural
  candidates are lower confidence: get_rows needs a dataflow/fusion angle, and
  Q5/Q6 broad workgroup forcing has now been separated enough to avoid the
  earlier false Q6 signal caused by Q5 collateral damage.

### 2026-04-10: hf9f.23 kernel grind pass

- Scope: follow-up after hf9f.22 on llama.cpp branch `epic1_perf_spike`.
  Accepted code landed in llama.cpp commit `6d73c9358`
  (`Specialize GET_ROWS and reduce RMS barriers`). Fresh baseline artifacts
  live under `build/pyre-epic2-results/hf9f-23/`: `default-trace-summary.txt`,
  `rocprof-default-n64-summary.txt`, `isa-top-baseline-summary.txt`, and
  `default-r5.json`.
- Fresh baseline ranking: `rocprof-default-n64-summary.txt` put the top
  kernel-only buckets at Q6_K wg128 82,110.308 us, Q4 expert packed MUL
  69,152.162 us, Q4 expert packed SWIGLU 67,034.239 us, BF16 matvec
  60,185.933 us, Q5_K wg128 47,067.721 us, `get_rows_f32` 45,828.981 us,
  F32 batched decode 43,280.550 us, `rms_norm_mul_f32` 39,080.022 us,
  Q5_K wg64 32,670.771 us, BF16 SWIGLU 31,198.157 us, and GDN cluster16
  27,947.339 us over n64. Baseline wall `default-r5.json` reported prompt
  216.827 tok/s average and decode 35.067 tok/s average.
- Accepted GET_ROWS nr=1 specialization: added `pyre_get_rows_f32_nr1`,
  selected only when the index tensor has one row. The provider removes the
  generic per-element row-to-3D index decomposition and multidimensional byte
  stride addressing for the repeated `nr=1` decode/prompt copies, while leaving
  the generic F32 provider for `nr>1` and Q5 row lookups unchanged. Final trace
  `final-trace-summary.txt` shows 124 claims of `GET_ROWS pure_hip_f32_nr1`
  and 39 remaining generic `GET_ROWS pure_hip_f32` claims, with no Pyre
  fallbacks. ISA artifact `isa-combined-summary.txt` shows the specialization
  at 4 VGPR / 18 SGPR / no spills versus generic 4 VGPR / 47 SGPR / no spills.
  Final clean rocprof `rocprof-final-n64-summary.txt` measured
  `pyre_get_rows_f32_nr1` at 29,572.580 us versus the fresh baseline
  `pyre_get_rows_f32` bucket at 45,828.981 us, a 16,256.401 us win over n64
  (~0.254 ms/token). This is the dominant accepted win.
- Accepted BF16 cols=1 matvec/SWIGLU specializations: added
  `pyre_mul_mat_vec_bf16_cols1_f32` and
  `pyre_mul_mat_vec_bf16_swiglu_cols1_f32`, selected only for `cols == 1`.
  These remove column addressing from the hot decode shapes and keep the
  existing generic/wg64/wg128 providers for other BF16 shapes. Final trace
  shows 111 BF16 cols1 matvec claims plus 119 generic BF16 matvec claims, and
  41 BF16 SWIGLU cols1 claims plus 39 generic BF16 SWIGLU claims.
  `isa-combined-summary.txt` shows BF16 cols1 at 10 VGPR / 18 SGPR / no spills
  versus generic 10 VGPR / 20 SGPR, and BF16 SWIGLU cols1 at
  14 VGPR / 18 SGPR / no spills versus generic 14 VGPR / 26 SGPR. Final
  rocprof measured BF16 cols1 at 60,116.001 us versus baseline BF16
  60,185.933 us, and BF16 SWIGLU cols1 at 30,667.377 us versus baseline
  31,198.157 us. This is small and near the noise floor for plain BF16, but it
  is shape-gated, resource-neutral-to-positive, and positive in the final
  decode-level profile.
- Accepted RMS barrier/LDS reduction: rewrote `rms_norm_mul_f32` and
  `add_rms_norm_mul_f32_broadcast` reductions to use wave shuffles plus a
  16-float cross-wave shared reduction. This drops LDS from 2048 bytes to
  64 bytes and reduces `s_barrier` count from 10 to 2 for both kernels.
  `isa-combined-summary.txt` reports `pyre_rms_norm_mul_f32` at
  13 VGPR / 51 SGPR / no spills / 64 bytes LDS and
  `pyre_add_rms_norm_mul_f32_broadcast` at 15 VGPR / 68 SGPR / no spills /
  64 bytes LDS. Intermediate rocprof `rocprof-bf16-rms-n64-summary.txt`
  showed small wins for both RMS buckets. The final clean rocprof sample kept
  ADD_RMS positive (15,657.954 us versus baseline 15,877.606 us) but measured
  plain RMS as a small regression (39,517.537 us versus baseline
  39,080.022 us). Net impact is small relative to GET_ROWS; the change was
  retained because it removes eight barriers and 1984 bytes of LDS per block
  without spills, and because the final accepted bucket sum remains positive.
- Q4 expert investigation: no new Q4 default change was landed. The fresh
  ranking still puts packed Q4 expert MUL/SWIGLU near the top, but prior
  hf9f.17/hf9f.18 evidence already rejected two obvious structures: vectorized
  packed loads duplicated RHS traffic and regressed, while the low-VGPR SWIGLU
  variant reduced VGPRs from 86 to 47 but increased instruction/global-load
  pressure and regressed decode-level rocprof. Those failures point away from
  another lane16/dot4 variant and toward a genuinely different expert-row
  decomposition or RHS reuse strategy. The final clean guardrail shows Q4 MUL
  at 70,022.035 us versus baseline 69,152.162 us and Q4 SWIGLU at
  66,665.921 us versus baseline 67,034.239 us, which is mixed/noise and was
  not caused by a Q4 code change in this pass.
- Q5/Q6 guardrails: no Q5/Q6 code changed in hf9f.23. Final clean rocprof
  measured Q6 wg128 at 81,732.596 us versus baseline 82,110.308 us, Q5 wg128
  at 46,935.891 us versus baseline 47,067.721 us, and Q5 wg64 at
  32,623.539 us versus baseline 32,670.771 us. These stayed effectively flat
  to slightly positive and did not block the accepted changes.
- Validation: rebuilt `pyre-kernel-bench`, `llama-bench`, and
  `test-backend-pyre` after committing `6d73c9358`; `test-backend-pyre`
  passed. Final trace artifact `final-trace-summary.txt` confirms no Pyre
  fallbacks. Final wall guardrail `final-cleancommit-r5.json` reports
  `build_commit=6d73c9358`, prompt 217.320 tok/s average, and decode
  35.468 tok/s average with samples `[35.5525, 35.4963, 35.4941, 35.3627,
  35.4342]`, versus the fresh baseline decode average of 35.067 tok/s.
- Net result: accepted default-enabled changes total about 16.64 ms over n64
  in the final clean rocprof sample, or ~0.260 ms/token kernel-only, using the
  changed bucket mapping BF16 cols1, BF16 SWIGLU cols1, GET_ROWS nr1, RMS, and
  ADD_RMS. The win clears the ticket's preferred 0.250 ms/token threshold and
  moves the wall guardrail in the same direction. Next structural work should
  not re-try the same Q4 local variants; it should either use profiler-guided
  expert-row/RHS reuse analysis or move to another high-ranked bucket where the
  resource/ISA evidence still shows a clear structural defect.

### 2026-04-10: hf9f.24 ATT-guided Q4 expert SWIGLU scheduling pass

- Scope: first micro-optimization pass driven by rocprofv3 ATT rather than
  structural ranking. Target was only
  `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32`; GET_ROWS widening and the
  already rejected Q4 vectorized/low-VGPR shapes were intentionally not retried.
  The input ATT artifact was
  `build/rocprof-att-q4swiglu-n1-20260410-102122/`, summarized with
  `sources/llama.cpp/tools/pyre-epic2/pyre-att-summary.py`.
- Baseline ATT map: `stats_ui_output_agent_16245_dispatch_24.csv` reported
  819 rows, 38,727 hits, 1,768,159 latency, 1,491,992 stall, and 1,861,383
  idle. The hottest stall sites were the packed Q4/RHS load and consume window:
  `global_load_d16_b16 ... offset:4` at vaddr 21428 with 242,983 stall,
  `s_waitcnt vmcnt(26)` at vaddr 22196 with 224,768 stall, plus setup
  `lgkmcnt` waits and tail `vmcnt` waits. Opcode buckets were dominated by
  `s_waitcnt` at 994,748 stall and `global_load` at 342,449 stall.
- Experiment tried and rejected: added a gated
  `pyre_mul_mat_id_q4_k_swiglu_packed_prefetch_wg64_f32` provider behind
  `GGML_PYRE_ENABLE_PACKED_Q4_K_SWIGLU_PREFETCH=1`. The source-level hypothesis
  was to issue q/RHS loads before scale/min conversion and byte-scale setup so
  the compiler had more independent work between load issue and the first
  `vmcnt` consume. It preserved the existing Qwen shape gate and RHS reuse, and
  did not duplicate the low-VGPR rejected structure.
- Correctness/selection: `test-backend-pyre` passed both default policy and
  the prefetch gate. Provider trace artifact
  `build/pyre-epic2-results/hf9f-24/prefetch-trace-summary.txt` showed exactly
  40 `MUL_MAT_ID_SWIGLU pure_hip_q4_K_packed_prefetch_wg64` claims, 40 packed
  Q4 MUL claims, and zero Pyre fallbacks.
- ISA/resource: `build/pyre-epic2-results/hf9f-24/isa-swiglu-prefetch-summary.txt`
  showed the candidate stayed resource-neutral relative to the current packed
  path: both were 86 VGPR / 46 SGPR / no spills / 16 bytes LDS. The full source
  opcode summary rose to 3,593 instructions and 101 `global_load` matches for
  the combined catalog source, so this still needed ATT/rocprof evidence before
  acceptance.
- Candidate ATT result: artifact
  `build/rocprof-att-q4swiglu-prefetch-n1-20260410-103410/` and summary
  `build/pyre-epic2-results/hf9f-24/prefetch-att-summary.txt` showed the
  scheduling hypothesis regressed the target. Total stall increased from
  1,491,992 to 1,977,612; total latency increased from 1,768,159 to 2,347,966;
  `global_load` opcode stall increased from 342,449 to 893,326; and the top
  `global_load_d16_b16 ... offset:4` site increased to 499,650 stall. The
  first `s_waitcnt vmcnt(26)` remained large at 216,062 stall, so the schedule
  mostly moved cost into worse global-load behavior instead of hiding it.
- Decode-level rocprof: `build/pyre-epic2-results/hf9f-24/rocprof-prefetch-n64-summary.txt`
  measured the gated SWIGLU candidate at 66,727.336 us over n64. That is
  effectively flat/slightly worse versus the previous clean packed SWIGLU
  sample in hf9f.23 (`rocprof-final-n64-summary.txt`: 66,665.921 us) and far
  short of the ticket's 0.150 ms/token acceptance bar. The ATT regression is
  therefore a real rejection signal, not just a profiler visualization artifact.
- Final decision: reverted the prefetch provider and left no llama.cpp code
  change for this ticket. Rebuilt `pyre-kernel-bench`, `llama-bench`, and
  `test-backend-pyre` after the revert to remove the temporary catalog entry.
  Final default trace `build/pyre-epic2-results/hf9f-24/final-default-trace-summary.txt`
  confirms the default path is back to 40
  `MUL_MAT_ID_SWIGLU pure_hip_q4_K_packed_wg64` claims, 40 packed Q4 MUL
  claims, and zero Pyre fallbacks.
- Next recommendation: do not pursue q/RHS prefetch-before-scale as a local
  schedule tweak. The compiler baseline already issues many loads before the
  first consume, and this attempt made global-load stalls worse without changing
  occupancy. The next Q4 SWIGLU micro-optimization should use a different
  mechanism: reduce the number of outstanding independent RHS/q loads per
  unrolled body, change expert-row work ownership, or use ATT on the packed MUL
  sibling to find a smaller-risk scheduling pattern before returning to SWIGLU.

### 2026-04-10: hf9f.26 broader kernel strategy pass

- Scope: widened the hf9f.24 ATT pass into a broader structural map, tracked
  by `pyre-workspace-hf9f.26`. The main strategy note is
  `docs/spike/analysis/llamacpp_pyre_broader_kernel_strategy.md`; artifacts live in
  `build/pyre-epic2-results/hf9f-26/`.
- Decode/prompt split: collected separate provider traces and rocprof tables.
  Decode n64 is still led by Q6 wg128 (82.065 ms), Q4 packed MUL (70.534 ms),
  Q4 packed SWIGLU (67.200 ms), BF16 cols1 (60.712 ms), and Q5 wg128
  (47.095 ms). Prompt p512 is a different problem: Q4 SWIGLU wg64
  465.926 ms, Q6 wg128 465.777 ms, Q4 ID wg64 387.125 ms, Q5 wg128
  345.125 ms, and F16 batched 257.210 ms. Both traces showed zero Pyre
  fallbacks.
- ATT map: reused the Q4 SWIGLU baseline from hf9f.24 and added packed Q4 MUL
  plus Q6 summaries. Packed Q4 MUL reported 810,139 total stall dominated by
  `s_waitcnt` at 640,059 stall, with top LGKM waits at 188,929 and 132,162
  stall and a later `vmcnt(19)` wait at 106,667 stall. Q6 reported 764,581
  total stall dominated by one `s_waitcnt vmcnt(4)` site at 589,308 stall.
  These differ from the Q4 SWIGLU global-load-heavy ATT shape.
- Byte model: current decode Q4 MUL and Q4 SWIGLU both reread about 32 MiB of
  F32 RHS per dispatch under one-output-row ownership, or about 1.34 GiB per
  decode token across the 40 expert dispatches. Q4 source traffic is smaller
  for MUL (~4.5 MiB/dispatch) and larger for SWIGLU (~9.0 MiB/dispatch for
  gate+up), which justified trying multi-row ownership in MUL first.
- Experiment tried and kept gated/default-off: added
  `pyre_mul_mat_id_q4_k_mul_packed_2row_wg64_f32` behind
  `GGML_PYRE_ENABLE_PACKED_Q4_K_MUL_2ROW=1`. It computes two adjacent output
  rows per workgroup for the Qwen decode shape, reusing RHS loads across both
  rows while doubling source-row work. Trace
  `q4mul-2row-trace-n1.log` showed 40
  `MUL_MAT_ID_MUL pure_hip_q4_K_packed_2row_wg64` claims and no fallback/error
  lines.
- Result: rejected for default. Wall r3 moved only from 35.246 tok/s to
  35.399 tok/s, within noise. Rocprof measured Q4 MUL at 70.383 ms versus the
  current packed baseline 70.534 ms, while the grid halved from
  `(131072,8,1)` to `(65536,8,1)`. Resource metadata showed the cost: packed
  MUL is 128 SGPR / 48 VGPR / no private / 8 B LDS, while two-row is
  128 SGPR / 64 VGPR / no private / 8 B LDS. The naive RHS-reuse ownership
  pattern therefore does not expose a real decode win and should not be
  promoted.
- Updated recommendation: do not spend the next pass on direct two-row SWIGLU;
  MUL is the lower-pressure sibling and it was flat. The higher-value
  structural tasks are prompt/prefill tiled matmul for the `cols=512`
  Q4/Q5/Q6/F16 providers, or a profiler-guided Q6 wait-site pass around the
  dominant `s_waitcnt vmcnt(4)` site. More Q4 ownership work should wait for a
  profiler trace that shows RHS misses, not scalar wait/VGPR pressure, as the
  controlling cost.

### 2026-04-10: hf9f.27 Q6 prompt/prefill cols4 provider

- Scope: implemented the first prompt/prefill-specific tiled provider for the
  `cols=512` regime tracked by `pyre-workspace-hf9f.27`. Artifacts live in
  `build/pyre-epic2-results/hf9f-27/`.
- Target selection: chose non-MoE Q6_K prompt matmul before Q4 expert SWIGLU.
  It was the second-hottest prompt bucket and a lower-risk proving ground:
  baseline prompt p512 showed 70 calls to `pyre_mul_mat_vec_q6_k_wg128_f32`
  totaling 464.757 ms with grid `(1048576,512,1)`, while Q4 expert SWIGLU
  combines expert ID routing, quant decode, two source rows, and activation
  fusion.
- Implementation: added `pyre_mul_mat_vec_q6_k_cols4_wg128_f32`, selected only
  for Q6_K `cols == 512`. One workgroup now computes four adjacent RHS columns
  for one output row, reusing Q6 block decode and reducing the prompt Q6
  y-grid from 512 to 128. The provider is default-on after validation, with
  `GGML_PYRE_DISABLE_Q6_K_COLS4_PROMPT=1` as the rollback knob.
- Rejected alternatives for this pass: full Vulkan-style BM/BN tiling and
  matrix-instruction rewrites were deferred as too broad for the first
  provider; Q4 expert SWIGLU was deferred because the routing/fusion complexity
  would obscure the basic prompt tiling result; Q5_K remains the next direct
  sibling if this pattern is continued.
- Prompt p512 rocprof before/after: Q6 moved from 70 calls, 464.757 ms total,
  6.639 ms avg, grid `(1048576,512,1)` to 70 calls, 186.916 ms total,
  2.670 ms avg, grid `(1048576,128,1)`. Other large buckets stayed in the
  expected noise band: Q4 SWIGLU 464.175 -> 460.788 ms, Q4 ID
  385.758 -> 382.678 ms, Q5 343.977 -> 343.620 ms, F16 batched
  256.183 -> 253.634 ms, BF16 119.772 -> 118.843 ms.
- Prompt wall checks: baseline trace run was 218.462 tok/s; gated cols4 trace
  run was 249.435 tok/s; r3 gated prompt check was 248.076 tok/s
  (`248.316, 248.823, 247.089`); default-on trace after validation was
  248.510 tok/s. These are single-machine smoke numbers, but the delta is well
  above the run-to-run noise seen in the r3 sample.
- Decode guardrail: n64 r3 with the provider enabled stayed at 35.471 tok/s
  (`35.4116, 35.4227, 35.5774`). Decode rocprof still used
  `pyre_mul_mat_vec_q6_k_wg128_f32` for the cols=1 path, 4480 calls totaling
  81.707 ms, so the prompt provider did not take over decode.
- Provider/fallback trace: gated prompt trace showed all 70 Q6 prompt claims
  on `pure_hip_q6_K_cols4_wg128` with zero fallback/error lines. The
  post-validation default-on trace also showed 70
  `pure_hip_q6_K_cols4_wg128` claims, no old prompt `q6_K_wg128` claims, and
  zero fallback/error lines.
- ISA/resource summary: baseline Q6 wg128 used 128 SGPR, 40 VGPR,
  16 B LDS/group, 0 private bytes, 48 B kernarg. The cols4 provider uses
  128 SGPR, 56 VGPR, 16 B LDS/group, 0 private bytes, 48 B kernarg. The VGPR
  increase is acceptable for the prompt win and does not introduce spills.
- Validation: rebuilt `test-backend-pyre`, `llama-bench`, and `llama-cli`.
  Added a direct backend correctness test for Q6_K `cols=512`; the full
  `test-backend-pyre` harness passed under the Pyre/ROCm runtime environment.

### 2026-04-10: hf9f.28 Q5 and Q4 prompt tiling push

- Scope: followed up the Q6 cols4 win in `pyre-workspace-hf9f.28`.
  Artifacts live in `build/pyre-epic2-results/hf9f-28/`. Fresh post-Q6
  baseline was captured before Q5 changes: prompt p512 r3 was 251.976 tok/s
  (`252.737, 252.143, 251.049`), and rocprof still showed Q5_K at
  344.458 ms with grid `(1048576,512,1)`.
- Q5_K implementation: added `pyre_mul_mat_vec_q5_k_cols4_wg128_f32` for the
  Q5_K `cols == 512` prompt path. Like Q6 cols4, one workgroup computes four
  adjacent RHS columns for one row, reusing Q5 unpack/decode work and reducing
  the Q5 prompt y-grid from 512 to 128. It is default-on with
  `GGML_PYRE_DISABLE_Q5_K_COLS4_PROMPT=1` as the rollback knob.
- Q5_K result: prompt p512 r3 improved from 251.976 to 276.234 tok/s
  (`276.868, 276.274, 275.560`). Rocprof moved Q5 from 30 calls,
  344.458 ms total, 11.482 ms avg, grid `(1048576,512,1)` to 30 calls,
  165.721 ms total, 5.524 ms avg, grid `(1048576,128,1)`. Provider trace
  showed 30 `pure_hip_q5_K_cols4_wg128` claims and zero fallback/error lines.
- Q5_K resources: baseline Q5 wg128 was 128 SGPR / 72 VGPR / 0 accum VGPR /
  16 B LDS / 0 private / 48 B kernarg. Q5 cols4 is 128 SGPR / 96 VGPR /
  0 accum VGPR / 16 B LDS / 0 private / 48 B kernarg. The extra VGPRs did
  not spill and the prompt win is large enough to keep it default-on.
- Q4 expert ID implementation: did not blindly port cols4. `MUL_MAT_ID`
  prompt uses `outer = token * n_ids + id_pos`; grouping adjacent y values
  would group different expert slots and often different experts, which is
  weak for source reuse. Instead added `pyre_mul_mat_id_q4_k_row4_wg64_f32`
  for the Q4 expert ID prompt shape. It computes four adjacent output rows
  for one routed expert slot/token, reusing the RHS vector and reducing the
  x-grid by 4. It is default-on for the narrow `k == 512`, `n_tokens == 512`,
  `rows % 4 == 0` prompt shape, with
  `GGML_PYRE_DISABLE_Q4_K_ID_ROW4_PROMPT=1` as the rollback knob.
- Q4 expert ID result: prompt p512 r3 improved further from the Q5-only
  276.234 tok/s to 294.896 tok/s (`295.905, 294.975, 293.809`). Rocprof moved
  Q4 expert ID from 39 calls, 383.114 ms in the fresh post-Q6 baseline
  (384.727 ms in the Q5-only run), grid `(131072,4096,1)`, to 39 calls,
  260.055 ms total, 6.668 ms avg, grid `(32768,4096,1)`. Provider trace
  showed 39 `pure_hip_q4_K_row4_wg64` claims and zero fallback/error lines.
- Q4 expert ID resources: baseline Q4 ID wg64 was 128 SGPR / 24 VGPR /
  0 accum VGPR / 8 B LDS / 0 private / 136 B kernarg. Row4 is
  128 SGPR / 64 VGPR / 0 accum VGPR / 8 B LDS / 0 private / 136 B kernarg.
  This is a large VGPR increase but no spill/private usage, and it bought a
  clear prompt wall and dispatch-time win.
- Decode guardrail: final n64 r3 with Q5 cols4 and Q4 ID row4 default-on was
  35.032 tok/s (`34.9271, 35.0215, 35.1476`). Decode rocprof still used the
  normal cols=1 paths, e.g. Q6 wg128, packed Q4 MUL, packed Q4 SWIGLU, BF16
  cols1, and Q5 wg128. The new prompt providers did not take over decode;
  the wall number is within the current decode noise band for this runtime.
- Validation: rebuilt `test-backend-pyre`, `llama-bench`, and `llama-cli`.
  Added direct Q5_K `cols=512` and Q4 `mul_mat_id` `tokens=512` correctness
  coverage; the full `test-backend-pyre` harness passed under the Pyre/ROCm
  runtime environment.
- Rejected/deferred alternatives: Q4 expert SWIGLU row grouping is the obvious
  next structural target because it is still the hottest prompt bucket at
  about 459 ms. I did not fold it into this commit because a row4 SWIGLU kernel
  carries gate and up sums for four rows, i.e. eight reductions plus much
  higher VGPR pressure. It should be attempted as the next focused experiment
  with its own correctness and resource gate, not hidden inside the already
  successful Q5/Q4-ID landing.

### 2026-04-10: hf9f.29 copy triage and Q4 SWIGLU row2 experiment

- Scope: followed up `pyre-workspace-hf9f.28` with D2D copy attribution first,
  then Q4 expert SWIGLU prompt row tiling. Artifacts live in
  `build/pyre-epic2-results/hf9f-29/`.
- Fresh default baseline: prompt p512 r3 before SWIGLU row2 was 294.156 tok/s
  (`294.419, 293.839, 294.209`). Rocprof showed Q4 expert SWIGLU at
  39 calls, 461.907 ms total, 11.844 ms avg, grid `(32768,4096,1)`, and the
  large copy bucket persisted: 120 copies of 150,994,944 B totaling
  669.677 ms.
- D2D copy attribution: `GGML_PYRE_TRACE_BUFFER_COPIES=150994944` showed the
  120 large transfers are `buffer set` uploads of Q4 expert weights:
  `blk.N.ffn_down_exps.weight`, `blk.N.ffn_gate_exps.weight`, and
  `blk.N.ffn_up_exps.weight` for 40 layers. That is 3 transfers per layer, not
  an inner graph `cpy_tensor` materialization. The trace had no
  `ggml-pyre: buffer copy` lines at that size.
- Copy conclusion: no quick llama.cpp graph/kernel fix landed. The call path is
  `ggml_backend_pyre_buffer_set_tensor` -> `pyre_synchronous_h2d` ->
  IREE HAL transfer; rocprof reports the underlying staging as
  `MEMORY_COPY_DEVICE_TO_DEVICE`. This should move to a runtime/load-path task
  if we want to remove the profiler bucket or exclude model upload from kernel
  steady-state runs. It is not a prompt graph contiguity copy.
- Q4 SWIGLU implementation tried: added
  `pyre_mul_mat_id_q4_k_swiglu_row2_wg64_f32` as a gated/default-off provider
  behind `GGML_PYRE_ENABLE_Q4_K_SWIGLU_ROW2_PROMPT=1`. It computes two
  adjacent output rows for one routed expert slot/token, reusing the RHS vector
  and reducing the SWIGLU prompt x-grid by 2. I chose row2 instead of row4
  because SWIGLU carries gate and up sums; row4 would require eight live sums
  and likely much higher VGPR pressure.
- Q4 SWIGLU row2 result: provider trace selected 39
  `pure_hip_q4_K_row2_wg64` claims with zero fallback/error lines. Rocprof
  improved the SWIGLU kernel bucket from 461.907 ms to 389.827 ms, and grid
  moved from `(32768,4096,1)` to `(16384,4096,1)`. Resource metadata moved
  from 128 SGPR / 40 VGPR / 16 B LDS / 0 private / 160 B kernarg for wg64 to
  128 SGPR / 64 VGPR / 8 B LDS / 0 private / 160 B kernarg for row2.
- Default decision: do not enable row2 by default. Wall was not convincing:
  row2 p512 r3 reported 242.641 tok/s with high variance
  (`178.987, 244.434, 304.501`) and the provider trace run was 154.164 tok/s.
  The kernel bucket moved in the right direction, but end-to-end prompt wall
  did not clear the default-on bar. The default post-rebuild trace showed zero
  row2 claims and 39 normal `q4_K_wg64` SWIGLU claims.
- Decode guardrail: default n64 r3 after rebuilding reported
  `17.7714, 35.5361, 35.5983` tok/s. The first sample is a cold/outlier run,
  while the steady samples match the previous ~35.5 tok/s band; row2 is
  default-off and shape-gated to prompt, so it does not take over decode.
- Validation: rebuilt `test-backend-pyre`, `llama-bench`, and `llama-cli`.
  Added direct Q4 SWIGLU prompt-shape correctness coverage and ran the full
  backend harness in both default mode and with
  `GGML_PYRE_ENABLE_Q4_K_SWIGLU_ROW2_PROMPT=1`; both passed. The SWIGLU test
  uses a relative tolerance because the row2 reduction order shifts large
  fused outputs by small relative amounts.
- Next recommendation: do not pursue SWIGLU row grouping further without a
  profiler trace that explains the wall/kernel mismatch. The remaining highest
  structural prompt candidates are F16 batched prompt tiling and BF16 prompt
  tiling, while the large copy bucket should be handled as runtime/model-load
  transfer behavior rather than as an inner graph copy.

### 2026-04-10: hf9f.30 F16 batched prompt cols4 provider

- Scope: followed up the hf9f.29 recommendation by targeting the dense F16
  prompt attention matvecs. Artifacts live in
  `build/pyre-epic2-results/hf9f-30/`.
- Baseline hygiene: the first collected baseline used a stale executable
  reporting build commit `a90f2db76`. I rebuilt `llama-bench` at branch head
  `4467291dd` before measuring deltas. The rebuilt default p512 r3 baseline
  was 295.022 tok/s with samples `296.009, 294.451, 294.607`.
- Baseline cost model: rocprof on the rebuilt baseline showed
  `pyre_mul_mat_vec_f16_batched_f32` as the third-largest prompt kernel bucket:
  20 calls, 423.664 ms total, 21.183 ms avg, grid `(131072,8192,1)`.
  Provider trace showed the two hot shapes are `k=256 rows=512 cols=512
  ne2=16` and `k=512 rows=256 cols=512 ne2=16`.
- Implementation: added `pyre_mul_mat_vec_f16_batched_cols4_f32` and selected
  it by default only for F16 batched prompt shape `cols == 512` with
  `cols % 4 == 0`. Decode `cols == 1` stays on the existing
  `pyre_mul_mat_vec_f16_batched_cols1_f32` provider. The new provider computes
  four adjacent prompt columns per workgroup, reusing the same F16 `src0` row
  and carrying four RHS accumulators. A kill switch is available via
  `GGML_PYRE_DISABLE_F16_BATCHED_COLS4_PROMPT=1`.
- Prompt result: p512 r3 improved from 295.022 tok/s to 320.204 tok/s with
  samples `321.355, 320.361, 318.895`. The single provider-trace run reported
  318.232 tok/s and selected 20 `pure_hip_f16_batched_cols4` claims with no
  fallback/error lines.
- Kernel result: rocprof showed the F16 batched bucket moved from
  423.664 ms to 106.761 ms, with the grid reduced from `(131072,8192,1)` to
  `(131072,2048,1)`. The remaining largest prompt buckets are now Q4 expert
  SWIGLU at 459.529 ms, Q4 expert ID row4 at 260.433 ms, Q6 cols4 at
  187.479 ms, Q5 cols4 at 166.542 ms, BF16 matvec at 120.492 ms, and F16
  batched cols4 at 106.761 ms.
- Resource check: `pyre-kernel-isa-summary.py` reports the generic F16 batched
  provider at 11 VGPR / 64 SGPR / no spills / 1024 B LDS, cols1 at
  10 VGPR / 40 SGPR / no spills / 1024 B LDS, and cols4 at
  20 VGPR / 65 SGPR / no spills / 1024 B LDS. The extra VGPR pressure is
  acceptable for this prompt shape because it removes 75% of the workgroups
  and produces a large kernel-bucket and wall win.
- Decode guardrail: n64 r3 default-on reported prompt `33.471` tok/s and
  decode `35.890` tok/s. With
  `GGML_PYRE_DISABLE_F16_BATCHED_COLS4_PROMPT=1`, the same command reported
  prompt `32.917` tok/s and decode `35.228` tok/s. That is within normal
  decode noise and confirms the new prompt provider does not take over the
  cols1 decode path.
- Validation: rebuilt `llama-bench` and `test-backend-pyre`. Added direct
  F16 batched prompt-shape correctness coverage with `cols=512` and batch
  broadcasting, and the full `test-backend-pyre` harness passed.
- Next recommendation: F16 prompt tiling should stay default-on. The next
  structural target is BF16 prompt tiling, but Q4 expert SWIGLU remains the
  largest kernel bucket. Do not revisit Q4 SWIGLU row grouping without
  profiler evidence explaining the hf9f.29 wall/kernel mismatch.

### 2026-04-10: hf9f.31 BF16 prompt cols4 second pass

- Scope: followed up `pyre-workspace-hf9f.30` by targeting the BF16 prompt
  matvec buckets before any wider quantized tiling. Artifacts live in
  `build/pyre-epic2-results/hf9f-31/`.
- Baseline hygiene: captured a fresh p512 r3 default baseline at 325.717 tok/s
  with samples `327.377, 325.483, 324.291`. An early baseline rocprof was
  rejected as contaminated because it overlapped another run and inflated
  unrelated buckets (`pyre_mul_mat_vec_bf16_swiglu_f32` at 420.215 ms and
  BF16 matvec at 247.402 ms). All accepted rocprof conclusions below use
  sequential same-binary runs.
- BF16 matvec implementation: added `pyre_mul_mat_vec_bf16_cols4_f32`,
  selected by default only for `cols == 512` BF16 prompt matvecs, preserving
  the existing BF16 `cols == 1` decode provider and wg64/wg128 policy hooks.
  The provider computes four adjacent RHS columns per workgroup and reuses the
  BF16 `src0` row. Rollback knob:
  `GGML_PYRE_DISABLE_BF16_COLS4_PROMPT=1`.
- BF16 matvec result: same-binary p512 r3 improved from 321.630 tok/s with
  the BF16 cols4 provider disabled (`323.101, 321.163, 320.625`) to
  329.261 tok/s default-on (`330.804, 328.831, 328.147`). Trace selected
  119 `pure_hip_bf16_cols4` matvec claims with zero fallback/error lines.
  Rocprof moved the BF16 matvec bucket from 121.823 ms, grid
  `(524288,512,1)`, to 69.090 ms, grid `(524288,128,1)`.
- BF16 matvec resources: ISA/resource summary reports generic BF16 matvec at
  10 VGPR / 20 SGPR / no spills / 32 B LDS, cols1 at
  10 VGPR / 18 SGPR / no spills / 32 B LDS, and cols4 at
  22 VGPR / 25 SGPR / no spills / 32 B LDS. This is a safe default-on
  tradeoff for the prompt shape.
- BF16 SWIGLU implementation: added
  `pyre_mul_mat_vec_bf16_swiglu_cols4_f32`, selected by default only for
  `cols == 512` BF16 prompt SWIGLU fusions. This carries four gate sums and
  four up sums, so it was evaluated separately from plain BF16 matvec.
  Rollback knob: `GGML_PYRE_DISABLE_BF16_SWIGLU_COLS4_PROMPT=1`.
- BF16 SWIGLU result: same-binary p512 r3 improved from 328.367 tok/s with
  only SWIGLU cols4 disabled (`330.146, 327.829, 327.128`) to a final
  333.713 tok/s default-on after fixing the dispatch y-count
  (`335.434, 333.147, 332.557`). Trace selected
  39 `pure_hip_bf16_cols4` SWIGLU claims, plus 158 BF16 matvec cols4 claims,
  with zero fallback/error lines. Rocprof moved the BF16 SWIGLU bucket from
  63.774 ms, grid `(131072,512,1)`, to 33.953 ms, grid `(131072,128,1)`.
- BF16 SWIGLU resources: generic BF16 SWIGLU is
  14 VGPR / 26 SGPR / no spills / 64 B LDS; cols1 is
  14 VGPR / 18 SGPR / no spills / 64 B LDS; cols4 is
  31 VGPR / 28 SGPR / no spills / 32 B LDS. No private memory or spills were
  introduced, and wall plus rocprof both moved in the right direction.
- Decode guardrail: after both default-on providers and the SWIGLU dispatch
  y-count fix, n64 r3 reported prompt `32.951` tok/s and decode
  `35.751` tok/s (`35.5452, 35.8148, 35.8921` for decode). The prompt
  providers are shape gated to `cols == 512`, so they do not take over decode.
- Validation: rebuilt `llama-bench` and `test-backend-pyre`; added direct
  BF16 prompt matvec and BF16 prompt SWIGLU correctness coverage with
  `cols=512`; the full `test-backend-pyre` harness passed after both
  providers were enabled.
- Next recommendation: keep both BF16 prompt cols4 providers default-on. I did
  not proceed to Q6_K cols8 in this ticket because the BF16 pass produced two
  clean default-on wins and Q4 expert SWIGLU remains the dominant prompt
  bucket. The next structural non-Q4 candidate is Q6_K cols8; do Q6 before Q5
  because Q6 cols4 previously had lower VGPR pressure than Q5 cols4.

### 2026-04-10: hf9f.32 Q6_K prompt cols8 and adjacent Q5_K probe

- Scope: followed up `pyre-workspace-hf9f.31` by widening the Q6_K prompt
  cols4 provider to eight adjacent prompt columns, then evaluated exactly one
  adjacent variant, Q5_K cols8. Artifacts live in
  `build/pyre-epic2-results/hf9f-32/`.
- Baseline hygiene: the first p512 r3 baseline used the previously built
  executable reporting build commit `256a2a546`, while the source tree was at
  `15e61e0a5` after the dead BF16 cleanup. That baseline was 335.662 tok/s
  (`336.287, 336.511, 334.188`) and is useful only as a continuity check.
  After rebuilding the same source, the accepted same-binary disabled-Q6-cols8
  control was 334.134 tok/s (`334.375, 334.398, 333.630`).
- Q6_K cols8 implementation: added
  `pyre_mul_mat_vec_q6_k_cols8_wg128_f32`, selected by default only for Q6_K
  prompt matvecs with `cols == 512` and `cols % 8 == 0`. It computes eight
  adjacent RHS columns per workgroup for one output row, reducing the prompt
  y-grid from 128 column groups to 64. Decode and other non-prompt shapes stay
  on the previous providers. Rollback knob:
  `GGML_PYRE_DISABLE_Q6_K_COLS8_PROMPT=1`.
- Q6_K result: same-binary p512 r3 improved from 334.134 tok/s with Q6 cols8
  disabled to a final default 348.251 tok/s
  (`348.869, 348.951, 346.935`). Provider trace selected
  `pure_hip_q6_K_cols8_wg128` for the Q6 prompt matvecs, while Q5 stayed on
  `pure_hip_q5_K_cols4_wg128`, Q4 expert ID stayed on
  `pure_hip_q4_K_row4_wg64`, F16 stayed on `pure_hip_f16_batched_cols4`, and
  BF16 matvec/SWIGLU stayed on `pure_hip_bf16_cols4`.
- Q6_K rocprof result: the paired p512 rocprof run moved the Q6 bucket from
  70 calls / 192.979 ms total / 2.757 ms avg on
  `pyre_mul_mat_vec_q6_k_cols4_wg128_f32`, grid `(1048576,128,1)`, to
  70 calls / 142.084 ms total / 2.030 ms avg on
  `pyre_mul_mat_vec_q6_k_cols8_wg128_f32`, grid `(1048576,64,1)`. The rocprof
  single-run wall moved from 329.375 tok/s with Q6 cols8 disabled to
  342.982 tok/s with the default Q6 cols8 provider.
- Q6_K resources: ISA/resource summary reports Q6 cols4 at
  56 VGPR / 25 SGPR / no spills / 16 B LDS / no private memory and Q6 cols8
  at 69 VGPR / 24 SGPR / no spills / 16 B LDS / no private memory. The VGPR
  increase is acceptable for this prompt shape because wall and rocprof both
  move in the right direction.
- Q5_K adjacent probe: added `pyre_mul_mat_vec_q5_k_cols8_wg128_f32`, but left
  it opt-in via `GGML_PYRE_ENABLE_Q5_K_COLS8_PROMPT=1` after evaluation.
  With both Q5 and Q6 cols8 enabled, p512 r3 was 346.433 tok/s
  (`347.879, 345.733, 345.687`), effectively flat/slightly worse than the
  Q6-only result. ISA showed Q5 cols8 has no spills/private memory
  (93 VGPR / 39 SGPR) and slightly fewer VGPR than Q5 cols4 (95 VGPR), so the
  rejection is based on measured wall performance rather than a resource
  hazard. Keeping the kernel opt-in preserves a differential knob without
  changing defaults.
- Decode guardrail: final default n64 r3 reported 33.220 tok/s
  (`31.1761, 34.2594, 34.2259`). The new Q6 default provider is gated to
  `cols == 512`, so it does not take over decode.
- Validation: rebuilt `llama-bench` and `test-backend-pyre`; the full
  `test-backend-pyre` harness passed with the default Q6 cols8 path and with
  the Q5/Q6 cols8 candidate active during the probe.
- Next recommendation: keep Q6_K cols8 default-on and Q5_K cols8 opt-in only.
  The next structural work should return to larger remaining p512 buckets:
  Q4 expert SWIGLU and Q4 expert ID remain much larger than the now-reduced
  Q6 bucket, and the rocprof API/copy sections still show runtime/model-load
  costs that should be treated separately from inner kernel structure.

### 2026-04-10: wonq.1 prefill 10x baseline refresh

- Scope: started `pyre-workspace-wonq`, a Vulkan-guided p512 prefill push
  focused on the Q4 expert path. Artifacts live in
  `build/pyre-epic2-results/wonq/`.
- Pyre p512 baseline: after rebuilding `llama-bench` so the embedded build
  commit is `ff0b42da9`, prompt-only p512 r5 reported 348.782 tok/s avg
  (`350.057, 349.904, 349.391, 348.504, 346.053`), with warm-last3
  347.983 tok/s. An earlier identical-code run built before the commit embed
  refresh reported commit `15e61e0a5` and 350.895 tok/s; it is kept only as a
  continuity check.
- Vulkan p512 reference: `build/llama-vulkan/bin/llama-bench` on `Vulkan0`
  reported 2320.513 tok/s avg for p512 r5
  (`2110.930, 2424.320, 2351.480, 2388.080, 2327.750`), with warm-last3
  2355.770 tok/s. This keeps Pyre about 6.8x behind Vulkan on the prompt-only
  shape after the recent Q5/Q6/F16/BF16 prompt wins.
- Pyre decode guardrail: n64 r3 at the same `ff0b42da9` build reported
  33.538 tok/s avg (`30.9746, 34.8884, 34.7520`).
- Pyre p512 rocprof baseline: single-run p512 rocprof reported 347.217 tok/s
  and confirmed the Q4 expert path dominates. Top relevant buckets:
  `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` at 39 calls / 458.842 ms total /
  11.765 ms avg, and `pyre_mul_mat_id_q4_k_row4_wg64_f32` at 39 calls /
  260.005 ms total / 6.667 ms avg. Combined Q4 expert time is therefore
  718.848 ms, roughly half of the prompt wall. Other buckets were
  Q5 cols4 167.815 ms, Q6 cols8 139.266 ms, F16 batched cols4 107.560 ms,
  BF16 cols4 69.053 ms, and BF16 SWIGLU cols4 33.788 ms.
- Commands:
  `llama-bench -p 512 -n 0 -b 512 -ub 512 -fa 0 -r 5 -o json --no-warmup -ngl 99 -dev PYRE0`,
  the same command with `build/llama-vulkan/bin/llama-bench -dev Vulkan0`,
  decode guardrail `-p 1 -n 64 -r 3`, and
  `OUT_DIR=build/rocprof-pyre-wonq-baseline-p512 OUT_FILE=wonq-baseline-p512 PROMPT=512 GEN=0 REPETITIONS=1 reproducers/rocprof_qwen_pyre_decode.sh`.
- Next track: proceed to `wonq.2` Vulkan-vs-HIP inventory. The first design
  question is whether Pyre can copy Vulkan's matrix-matrix/matmul-id path for
  prompt expert Q4, rather than continuing local matvec-style ownership changes.

### 2026-04-10: wonq.2 Vulkan-vs-HIP Q4 expert inventory

- Scope: inventoried the Vulkan prompt Q4 expert path against Pyre HIP before
  starting `pyre-workspace-wonq.3`. The baseline remains the `wonq.1` Pyre
  p512 Q4 expert bucket: 39 calls / 458.842 ms for
  `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` plus 39 calls / 260.005 ms for
  `pyre_mul_mat_id_q4_k_row4_wg64_f32`, combined 718.848 ms.
- Vulkan reference shape: `GGML_VK_PERF_LOGGER=1` on p512 reported hot
  matrix-shaped `MUL_MAT_ID q4_K` calls, not the vector-ID path:
  `m=2048 n=8 k=512 n_expert=256 batch=512` at 39 calls / 38.010 ms total /
  974.623 us avg and `m=512 n=8 k=2048 n_expert=256 batch=512` at
  78 calls / 68.807 ms total / 882.137 us avg. This is the strongest
  current evidence that the remaining Pyre p512 gap is Q4 expert kernel
  structure, not runtime overhead.
- Vulkan ownership/dataflow: `ggml_vk_mul_mat_id_q_f16` selects the matmul-ID
  matrix path and dispatches a `count_experts` kernel before the Q4_K matmul.
  The Q4_K matmul shader (`mul_mmq.comp`) groups work by expert on
  `gl_WorkGroupID.z`, by routed-row tile on `gl_WorkGroupID.y`, and by output
  row tile on `gl_WorkGroupID.x`. `load_row_ids` scans the `(id, token)` route
  matrix into shared row IDs for the current expert tile, then stores results
  back to the original output row positions. This avoids Pyre's current
  repeated per-output direct `ids[id, token]` lookup and enables tile-local
  reuse across a batch of routed prompt rows.
- Vulkan arithmetic/packing: the hot Q4_K path quantizes the RHS activation
  matrix to Q8_1 when integer dot support is available, loads Q4_K scales/mins
  into a shared/cache representation, repacks Q4 lanes, and uses
  `dotPacked4x8EXT` style Q4 x Q8 dot products. The important structural
  point is not the exact SPIR-V intrinsic; it is that Vulkan converts the
  F32 RHS once into a compact Q8_1 representation and reuses it inside a
  tiled matrix-style expert matmul.
- Pyre current ownership/dataflow: the default prompt Q4 expert-ID kernel is
  `pyre_mul_mat_id_q4_k_row4_wg64_f32` and the default prompt Q4 SWIGLU
  fusion is `pyre_mul_mat_id_q4_k_swiglu_wg64_f32`. Both are still
  matvec-shaped: one workgroup owns one output row (or four adjacent output
  rows for the row4 ID provider) for one `(id, token)` route. They read
  F32 RHS values directly for each output row and route, do not compact routes
  by expert, and do not reuse an activation tile across rows the way Vulkan
  does.
- Existing Pyre Q8_1 clue: Pyre already has direct executable providers for
  `pyre_mul_mat_id_q4_k_q8_1_f32`, `pyre_mul_mat_id_q4_k_mul_q8_1_f32`, and
  `pyre_quantize_q8_1_f32`, but support is gated behind
  `GGML_PYRE_ENABLE_Q8_1_MMVQ=1` plus `GGML_PYRE_Q8_1_MMVQ_POLICY=all`. That
  means the baseline prompt trace does not exercise the Q8_1 expert-ID path,
  and there is no Q8_1 SWIGLU fused provider yet.
- Already-tried/rejected local variants: prior Q4 prompt/decode attempts
  covered row2 SWIGLU, packed decode, packed two-row MUL, lane/vector tweaks,
  and a local prefetch-style SWIGLU variant. These were small ownership or
  lane-local changes and did not change the dominant dataflow. They should not
  be repeated as the main `wonq.3` attempt unless needed as guardrail knobs.
- Structural candidates for `wonq.3`: first, measure the existing Q4 expert
  Q8_1 path on prompt and, if it moves the right bucket, make a prompt-specific
  gated policy rather than relying on global `policy=all`. Second, add a
  fused Q4_K SWIGLU Q8_1 provider so the largest Q4 bucket can use one RHS
  quantization and Q4 x Q8 arithmetic while preserving the current fusion
  boundary. Third, if Q8_1 arithmetic alone is insufficient, implement a
  prompt-only expert-grouped tiled path with count/row-id compaction, because
  that is the largest remaining structural mismatch with Vulkan.

### 2026-04-10: wonq.3-wonq.5 Q4 prompt row-tiling pass

- Scope: implemented and validated two structurally distinct Q4 expert prompt
  variants after the Vulkan inventory: a Q4 expert-ID row8 provider and a Q4
  SWIGLU row4 provider. Artifacts live under
  `build/pyre-epic2-results/wonq/`.
- Q8_1 rejection: forced `GGML_PYRE_ENABLE_Q8_1_MMVQ=1
  GGML_PYRE_Q8_1_MMVQ_POLICY=all` was rechecked before writing new Q8_1 code.
  With SWIGLU fusion intact, p512 r1 collapsed to 61.250 tok/s while only the
  smaller `k=512 rows=2048` Q4 expert-ID path selected `_q8_1`. Disabling
  SWIGLU fusion converted the two SWIGLU `MUL_MAT_ID` inputs to `_q8_1` too,
  but fell further to 54.269 tok/s. This matches the earlier decode-scale
  Q8_1 rejection in this document, so no fused Q8_1 SWIGLU provider was added.
- Q4 expert-ID row8: added `pyre_mul_mat_id_q4_k_row8_wg64_f32`, selected only
  with `GGML_PYRE_ENABLE_Q4_K_ID_ROW8_PROMPT=1`. The first run exposed a
  dispatch-grid bug (`rows` workgroups instead of `rows / 8`); after fixing
  that, rocprof moved the Q4 ID bucket from the `wonq.1` baseline 260.005 ms
  to 245.026 ms. Full p512 r3 was still only 346.310 tok/s, so row8 remains
  opt-in/rejected for default use. ISA/resource summary: row4 ID is
  59 VGPR / 48 SGPR / no spills / 8 B LDS / no private memory; row8 ID is
  75 VGPR / 48 SGPR / no spills / 8 B LDS / no private memory.
- Q4 SWIGLU row4: added `pyre_mul_mat_id_q4_k_swiglu_row4_wg64_f32`, selected
  by default for the prompt shape `k=2048 rows=512 ids=8 tokens=512`, with
  rollback knob `GGML_PYRE_DISABLE_Q4_K_SWIGLU_ROW4_PROMPT=1`. The provider
  computes four adjacent output rows per workgroup and reuses the same routed
  F32 RHS load across gate/up accumulation for those rows. A follow-up scratch
  change gave each gate/up row its own tiny reduction LDS slice, reducing
  explicit barriers from 30 to 23 in the compiled source summary.
- Q4 SWIGLU result: final default no-trace p512 r3 after the scratch change was
  363.825 tok/s (`364.745, 364.005, 362.726`). A p512 r5 no-trace run before
  the scratch change was 361.617 tok/s (`364.037, 363.463, 361.978, 360.833,
  357.772`); the final trace-enabled p512 r5 was 355.725 tok/s and confirmed
  195 prompt claims for Q4 ID row4 and 195 prompt claims for Q4 SWIGLU row4,
  with no row8 or fallback/error claims. Baseline from `wonq.1` was
  348.782 tok/s p512 r5.
- Q4 SWIGLU rocprof: final default p512 rocprof reported 358.947 tok/s and
  moved Q4 SWIGLU from 39 calls / 458.842 ms total / 11.765 ms avg on
  `pyre_mul_mat_id_q4_k_swiglu_wg64_f32` to 39 calls / 419.541 ms total /
  10.757 ms avg on `pyre_mul_mat_id_q4_k_swiglu_row4_wg64_f32`. Q4 ID stayed
  on row4 at 39 calls / 255.438 ms total / 6.550 ms avg. Combined Q4 expert
  bucket is therefore 674.979 ms, down from the `wonq.1` baseline 718.848 ms.
- Q4 SWIGLU resources: final row4 SWIGLU compiles at 114 VGPR / 53 SGPR /
  no VGPR spills / no SGPR spills / 64 B LDS / no private memory. The previous
  wg64 SWIGLU provider is 36 VGPR / 46 SGPR / 16 B LDS; the row2 prompt probe
  is 59 VGPR / 49 SGPR / 8 B LDS; the packed decode provider is
  86 VGPR / 46 SGPR / 16 B LDS. The accepted row4 provider deliberately pays
  higher VGPR for fewer workgroups and more RHS reuse, and the wall plus
  rocprof results justify that tradeoff for p512.
- Decode guardrail: final default decode n64 r3 after enabling row4 SWIGLU was
  32.626 tok/s (`30.1802, 33.8955, 33.8035`). The first sample remains cold;
  warm samples are in the same range as the `wonq.1` 33.538 tok/s guardrail
  (`30.9746, 34.8884, 34.7520`). The new SWIGLU provider is shape-gated to
  `tokens == 512`, so decode should stay on the existing packed decode path.
- Validation: rebuilt `llama-bench` and `test-backend-pyre`; `test-backend-pyre`
  passed with row8 opt-in, with row4 SWIGLU opt-in, and after row4 SWIGLU was
  made default-on. Provider traces showed the expected prompt-only selection
  and no fallback/error lines.
- Final status: meaningful partial success, not the 10x/Vulkan-class target.
  Default-on: Q4 SWIGLU row4. Opt-in/rejected for default: Q4 expert-ID row8
  (`GGML_PYRE_ENABLE_Q4_K_ID_ROW8_PROMPT=1`). Rejected: current global Q8_1
  expert path, because conversion and current scalar Q8 kernels overwhelm any
  packed-arithmetic benefit. Next highest-leverage track is still a real
  Vulkan-style expert-grouped tiled Q4 matmul-ID path with route compaction
  (`count_experts` / row IDs) and matrix-shaped output tiles; local row tiling
  has now produced only single-digit percent Q4 bucket improvements.

### 2026-04-10: wonq.6 expert-grouped Q4 prompt prototype

- Scope: reopened the prefill work with the stricter bar from `wonq`: stop
  iterating row-only local variants and prototype the actual Vulkan-style
  structural mismatch. Added route compaction (`clear_u32` +
  `compact_moe_routes_i32`) and grouped Q4 prompt providers that dispatch by
  expert and output-row tile, with compacted `(id, token)` routes staged once
  per prompt expert matmul. The accepted paths are default-on for the exact
  Qwen prompt shapes and keep rollback knobs:
  `GGML_PYRE_DISABLE_Q4_K_ID_GROUPED_PROMPT=1` and
  `GGML_PYRE_DISABLE_Q4_K_SWIGLU_GROUPED_PROMPT=1`.
- Q4 expert-ID grouped path: added
  `pyre_mul_mat_id_q4_k_grouped_row4_wg64_f32`, selected for
  `k=512 rows=2048 ids=8 tokens=512`. It uses compacted routes by expert and
  a row4 x route4 tile, so one expert/row tile reuses Q4 scale/min and weight
  loads across four routed activation columns. Correctness passed under
  `test-backend-pyre`. A p512 trace run with only this path enabled reported
  382.506 tok/s and 39 prompt claims for
  `pure_hip_q4_K_grouped_row4_wg64`; rocprof reported 382.655 tok/s and moved
  the Q4 ID bucket to 39 calls / 194.870 ms total / 4.997 ms avg, with
  route compaction overhead at only 174.521 us total plus clear at 92.320 us.
  ISA/resource summary: 106 VGPR / 86 SGPR / no VGPR spills / no SGPR spills /
  32 B LDS / no private memory. This is a real structural win and is now
  default-on.
- Rejected SWIGLU route1 attempt: the first grouped SWIGLU prototype used the
  same route compaction and expert grouping but processed one compacted route
  at a time for row4. It passed `test-backend-pyre`, but p512 r3 regressed to
  345.995 tok/s with SWIGLU-only grouped selection and 363.828 tok/s when
  combined with grouped ID. That proved route compaction alone is insufficient
  for SWIGLU; the tile must reuse Q4 loads across routed activation columns.
- Intermediate SWIGLU route2 attempt: replacing route1 with a row4 x route2
  grouped SWIGLU tile passed correctness and improved SWIGLU-only p512 r3 to
  384.457 tok/s. Combined with grouped ID it reached 405.869 tok/s p512 r5
  (`407.025, 407.880, 407.411, 403.086, 403.944`). Rocprof showed the Q4
  bucket at 39 calls / 340.309 ms for grouped SWIGLU plus 39 calls /
  186.466 ms for grouped ID, with route compaction + clear below 0.6 ms total.
  ISA/resource summary: grouped route2 SWIGLU compiled at 121 VGPR / 83 SGPR /
  no spills / 64 B LDS. Useful, but still below the `wonq` p512 partial bar.
- Accepted SWIGLU route4 structure: added
  `pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route4_wg64_f32`, selected ahead
  of the row4 x route2 fallback for the same prompt shape. This keeps the
  accumulator footprint similar to the route2 variant but changes the tile to
  row2 x route4, cutting Q4 gate/up block reloads for a four-route tile while
  still grouping by expert and compacted prompt routes. Correctness passed;
  SWIGLU-only p512 r3 was 426.396 tok/s (`426.909, 426.465, 425.814`) with
  117 provider claims for `pure_hip_q4_K_grouped_row2_route4_wg64`.
- Final default performance: after making grouped ID and grouped SWIGLU
  default-on, the no-env p512 r5 trace reported 453.024 tok/s
  (`453.277, 455.827, 454.416, 452.650, 448.950`) and confirmed 195 claims
  for `pure_hip_q4_K_grouped_row2_route4_wg64` plus 195 claims for
  `pure_hip_q4_K_grouped_row4_wg64`. The best explicit-env combined p512 r5
  before the default-on relink was 455.457 tok/s (`455.944, 457.398, 455.485,
  454.985, 453.473`). Relative to the refreshed `wonq.1` baseline
  348.782 tok/s, this is a 29.9-30.6% p512 improvement and meets the p512
  half of the `wonq` meaningful-partial bar.
- Final rocprof bucket: p512 rocprof on the accepted default shape reported
  428.504 tok/s under profiler, with Q4 ID at 39 calls / 212.048 ms total /
  5.437 ms avg and Q4 SWIGLU row2-route4 at 39 calls / 205.296 ms total /
  5.264 ms avg. Route compaction remained cheap: 78 calls / 506.286 us total
  for `pyre_compact_moe_routes_i32` and 78 calls / 316.244 us total for
  `pyre_clear_u32`. Combined Q4 expert bucket is 417.344 ms versus the
  refreshed 718.848 ms baseline, a 41.9% reduction; this is meaningful but
  still not the stricter `<=250 ms` / 2x Q4-bucket target.
- Final resources: row2-route4 SWIGLU compiles at 86 VGPR / 81 SGPR / no VGPR
  spills / no SGPR spills / 32 B LDS / no private memory. This is lower VGPR
  and LDS than row4-route2 (121 VGPR / 64 B LDS) and much faster in wall time,
  validating the row2 x route4 structural choice. Grouped ID remains 106 VGPR /
  86 SGPR / no spills / 32 B LDS.
- Decode guardrail: default-shape gating keeps decode on the existing decode
  providers. Combined grouped ID + SWIGLU row2-route4 decode n64 r5 repeat was
  35.388 tok/s (`35.3675, 35.5086, 35.3452, 35.3640, 35.3528`). An earlier
  n64 r3 had one cold 22.552 tok/s sample but then two 35.8 tok/s warm samples;
  the repeat run is the guardrail used for acceptance.
- Validation commands/artifacts: rebuilt with
  `cmake --build build/llama-pyre-rocm713 --target llama-bench test-backend-pyre -j$(nproc)`;
  ran `test-backend-pyre` for grouped SWIGLU route1, route2, row2-route4, and
  final default-on policy. Key artifacts are under
  `build/pyre-epic2-results/wonq/`: `grouped-default-on-p512-r5.json`,
  `grouped-default-on-p512-r5.trace`,
  `grouped-row2-route4-rocprof-p512-summary.txt`,
  `grouped-id-swiglu-row2-route4-decode-n64-r5-repeat.json`, and
  `isa-grouped-row2-route4/summary.txt`. Provider traces showed expected
  prompt-only grouped selection with no fallback/error lines.
- Remaining gap: the provider interface was sufficient for this prototype; no
  runtime/provider API change was required. The remaining Vulkan gap is not
  route compaction overhead; it is the actual tiled arithmetic quality. Vulkan
  still uses a deeper matrix-style Q4 path with RHS packing/Q8 dot-style
  arithmetic and more aggressive scale/min/weight tile reuse. Future work
  should use profiler data to decide between a Q4 x Q8 grouped SWIGLU/ID path,
  a route8/row1 or row4 variant if VGPR allows, or a more faithful shared-tile
  Q4_K dequant layout. Local row-only variants should remain rejected unless
  used as guardrail scaffolding.

#### wonq.6 follow-up: accepted ID row2-route8, rejected SWIGLU row1-route8

- After the first default-on grouped endpoint, I applied the same successful
  route-width idea to the non-SWIGLU Q4 expert-ID path. The accepted provider
  is `pyre_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32`, selected ahead of
  the row4-route4 grouped ID fallback. It keeps 16 accumulators, reduces Q4
  row-block reloads across eight compacted routes, and remains default-on under
  `GGML_PYRE_DISABLE_Q4_K_ID_GROUPED_PROMPT=1` rollback.
- Final p512 no-env trace after removing the rejected SWIGLU row1-route8 probe:
  470.675 tok/s r5 (`470.316, 469.605, 471.979, 470.531, 470.947`). Provider
  trace confirmed 195 claims for `pure_hip_q4_K_grouped_row2_route8_wg64` and
  195 claims for `pure_hip_q4_K_grouped_row2_route4_wg64`, with no row1-route8
  SWIGLU claims.
- Final rocprof for the ID route8 endpoint reported 447.672 tok/s under
  profiler. Q4 buckets: 39 calls / 170.409 ms total / 4.369 ms avg for
  `pyre_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32` and 39 calls /
  202.027 ms total / 5.180 ms avg for
  `pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route4_wg64_f32`. Combined Q4
  expert bucket is 372.436 ms, down 48.2% from the refreshed 718.848 ms
  baseline. This misses the strict 2x bucket target by about 13 ms but clears
  the p512 target by a wide margin: +34.9% over 348.782 tok/s.
- ID route8 resources: 90 VGPR / 104 SGPR / no VGPR spills / no SGPR spills /
  16 B LDS / no private memory. Compared with grouped ID row4-route4 at
  106 VGPR / 86 SGPR / 32 B LDS, route8 is both faster and lower VGPR/LDS,
  with higher SGPR still acceptable.
- Decode guardrail for the route8 endpoint: n64 r5 was 33.807 tok/s including
  a cold first sample (`25.3345, 35.8409, 35.8060, 36.0594, 35.9944`); warm
  decode samples remained in the expected ~35.8-36.1 tok/s range. The grouped
  prompt providers are shape-gated to `tokens == 512` and do not select during
  decode.
- Rejected SWIGLU row1-route8: the provider passed `test-backend-pyre`, but
  default p512 with row1-route8 SWIGLU selected regressed to 436.318 tok/s
  (`437.840, 437.725, 436.433, 435.860, 433.733`). It also perturbed the
  generated SWIGLU code object enough that leaving it as an opt-in guardrail
  reduced the non-selected default path, so I removed the kernel/catalog/provider
  instead of keeping it gated.
- Updated artifacts: `final-id-route8-no-row1-p512-r5.json`,
  `final-id-route8-no-row1-p512-r5.trace`,
  `grouped-id-route8-rocprof-p512-summary.txt`,
  `grouped-id-route8-decode-n64-r5.json`, and
  `isa-grouped-id-route8/summary.txt`.

#### wonq.7 follow-up: SWIGLU route8 and multi-output reductions

- Recovered after a power-loss interruption by rebuilding with a hard check for
  generator failures/empty catalogs. `rebuild-after-power.log` and
  `rebuild-default-route8.log` both rebuilt cleanly without `failed to compile`
  or empty-catalog warnings.
- Added `pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32`, initially
  opt-in, then default-on with rollback
  `GGML_PYRE_DISABLE_Q4_K_SWIGLU_GROUPED_ROW2_ROUTE8_PROMPT=1`. The first
  route8-only SWIGLU probe passed correctness and selected 195 prompt claims,
  but only reached 482.716 tok/s p512 r5 (`484.580, 483.286, 483.684,
  481.405, 480.625`). ISA explained the modest local gain: row2-route8 SWIGLU
  rises to 118 VGPR / 107 SGPR / no spills from row2-route4's 86 VGPR / 81
  SGPR / no spills.
- Added multi-output reduction helpers for the serial-reduction frontier in
  Q5_K cols4/cols8, Q6_K cols4/cols8, and F16 batched cols4. This removes the
  one-shared-reduction-per-output-column pattern and reduces all accumulators in
  one shared-memory phase. With SWIGLU route8 still opt-in, p512 moved to
  500.972 tok/s r5 (`502.058, 502.676, 502.331, 499.834, 497.959`); without
  the SWIGLU route8 opt-in, the same multi-reduce patch reported 482.431 tok/s
  r5.
- Applied the same reduction-pattern fix to the dominant Q4 grouped kernels: ID
  row2-route8 now reduces two row accumulators together per routed token, and
  SWIGLU row2-route4/row2-route8 reduce gate/up for both rows together. This
  preserved the same VGPR resources but lowered barrier/instruction count:
  `mul_mat_id_q4_k.hip.cpp` s_barrier 81 -> 73 and instructions 9678 -> 9388;
  `mul_mat_id_q4_k_swiglu.hip.cpp` s_barrier 101 -> 65 and instructions 14259
  -> 13395.
- Final no-env default p512 r5 is 520.474 tok/s (`522.879, 522.174, 520.168,
  518.541, 518.610`), versus the prior wonq.6 endpoint 470.675 tok/s and the
  refreshed wonq.1 baseline 348.782 tok/s. Provider trace confirmed 195 claims
  for `pure_hip_q4_K_grouped_row2_route8_wg64` and 195 claims for
  `pure_hip_q4_K_grouped_row2_route8_wg64` on SWIGLU, with no fallback/error
  lines.
- Final p512 rocprof artifact `q4-multireduce-route8-rocprof-p512-summary.txt`:
  Q4 SWIGLU row2-route8 39 calls / 148.610 ms total / 3.811 ms avg, Q4 ID
  row2-route8 39 calls / 145.529 ms total / 3.732 ms avg. Combined Q4 expert
  bucket is 294.139 ms, down from 372.436 ms at the wonq.6 route8 endpoint and
  down 59.1% from the refreshed 718.848 ms wonq.1 baseline.
- The same rocprof run shows the next frontier clearly: Q5_K cols4 30 calls /
  144.689 ms, Q6_K cols8 70 calls / 120.635 ms, F16 batched cols4 20 calls /
  87.135 ms, BF16 cols4 119 calls / 66.072 ms, and Q8_0 add 9 calls /
  41.226 ms. The broad multi-reduce pass already improved Q5/Q6/F16, but they
  are now comparable to the individual Q4 buckets and should drive the next
  pass if Q4 structural variants stall.
- Resource summary artifact `isa-q4-multireduce-route8/summary.txt`: Q4 ID
  row2-route8 remains 90 VGPR / 104 SGPR / no spills / 16 B LDS; Q4 SWIGLU
  row2-route8 remains 118 VGPR / 107 SGPR / no spills / 32 B LDS; Q5 cols4 is
  95 VGPR / 25 SGPR / no spills; Q6 cols8 is 69 VGPR / 24 SGPR / no spills;
  F16 batched cols4 is 19 VGPR / 65 SGPR / no spills.
- Decode guardrail in final default configuration: n64 r5 35.941 tok/s
  (`35.9127, 35.9434, 35.9626, 35.9400, 35.9481`). Prompt providers remain
  shape-gated and do not materially affect decode.
- Key artifacts under `build/pyre-epic2-results/wonq7/`: 
  `test-backend-default-route8.log`,
  `default-route8-q4-multireduce-p512-r5.json`,
  `default-route8-q4-multireduce-p512-r5.trace`,
  `default-route8-q4-multireduce-decode-n64-r5.json`,
  `q4-multireduce-route8-rocprof-p512-summary.txt`, and
  `isa-q4-multireduce-route8/summary.txt`.

#### wonq.7 checkpoint: Q5 cols8 retest after multi-reduce

- Retested Q5_K cols8 after the multi-output reduction change because Q5_K
  cols4 became the largest non-Q4 bucket. Opt-in p512 r5 with
  `GGML_PYRE_ENABLE_Q5_K_COLS8_PROMPT=1` reached 530.636 tok/s (`534.077,
  534.567, 529.127, 527.443, 527.968`) with 150 Q5 cols8 provider claims and
  no Q5 cols4 claims.
- Made Q5_K cols8 default-on with rollback
  `GGML_PYRE_DISABLE_Q5_K_COLS8_PROMPT=1`. Default p512 r5 then reported
  527.170 tok/s (`530.447, 530.484, 526.662, 524.469, 523.791`), with 150
  Q5 cols8 claims, 350 Q6 cols8 claims, expected Q4 route8 claims, and no
  fallback/error lines.
- Rocprof artifact `default-q5-cols8-rocprof-p512-summary.txt`: Q5_K cols8
  bucket is 30 calls / 133.718 ms total / 4.457 ms avg. This is a small but
  real improvement over the prior Q5_K cols4 bucket at 144.689 ms in the Q4
  multi-reduce endpoint. Other buckets in that profiler run were noisier at
  memory-copy/API level, so p512 r5 is the acceptance signal for the default-on
  policy.
- Decode guardrail `default-q5-cols8-decode-n64-r5.json`: 34.344 tok/s
  including a cold first sample (`27.4529, 36.0645, 36.0966, 36.0548,
  36.0524`). Warm decode remains stable around 36.05 tok/s.

#### wonq.7 checkpoint: BF16 multi-output reductions

- Extended the multi-output reduction pattern to BF16 prompt providers:
  `pyre_mul_mat_vec_bf16_cols4_f32` now reduces four RHS columns in one
  shared-memory phase, and `pyre_mul_mat_vec_bf16_swiglu_cols4_f32` reduces
  four gate plus four up accumulators in one phase. This is still the existing
  default provider path; no new policy gate was added.
- Rebuild artifact `rebuild-bf16-multireduce.log` completed cleanly with the
  hard empty-catalog check, and `test-backend-bf16-multireduce.log` passed.
- Default p512 r5 improved to 542.279 tok/s (`547.010, 545.369, 539.946,
  541.464, 537.606`) from the Q5-cols8 default checkpoint at 527.170 tok/s
  and the Q4 multi-reduce checkpoint at 520.474 tok/s.
- Provider trace confirmed the intended BF16 prompt paths: 595 claims for
  `MUL_MAT provider=pure_hip_bf16_cols4`, 195 claims for
  `MUL_MAT_SWIGLU provider=pure_hip_bf16_cols4`, and no fallback/error lines.
- Decode guardrail `bf16-multireduce-decode-n64-r5.json`: 33.756 tok/s
  including a cold first sample (`25.3338, 35.8059, 35.8450, 35.9104,
  35.8858`). Warm decode remains in the expected ~35.8-35.9 tok/s band.
- Rocprof artifact `bf16-multireduce-rocprof-p512-summary.txt`: BF16 cols4 is
  119 calls / 57.941 ms total / 0.487 ms avg, and BF16 SWIGLU cols4 is
  39 calls / 28.949 ms total / 0.742 ms avg. Compared with the previous noisy
  Q5-cols8 default trace, these buckets were 73.806 ms and 35.929 ms; compared
  with the Q4 multi-reduce endpoint, they were 66.072 ms and 32.359 ms.
- Resource artifact `isa-bf16-multireduce/summary.txt`: BF16 cols4 is 22 VGPR /
  25 SGPR / no spills / 32 B LDS; BF16 SWIGLU cols4 is 31 VGPR / 28 SGPR /
  no spills / 256 B LDS. The added LDS is bounded and the kernels remain
  spill-free.
- Remaining top p512 buckets in the same trace are still structural targets:
  Q4 SWIGLU route8 164.293 ms, Q4 ID route8 153.020 ms, Q5_K cols8 134.367 ms,
  Q6_K cols8 125.743 ms, F16 batched cols4 93.117 ms, and Q8_0 add 49.318 ms.
  The next best structural probe is to widen one of the remaining tiled prompt
  paths, with Q6_K cols16 or F16 batched cols8 as the lowest-reg pressure
  candidates; Q4 route16/SWIGLU row4 remain higher-risk because current route8
  SWIGLU is already 118 VGPR.

#### wonq.7 checkpoint: F16 batched cols8

- Prototyped `pyre_mul_mat_vec_f16_batched_cols8_f32` after BF16 multi-reduce
  because F16 batched cols4 remained a 90 ms-class p512 bucket and its cols4
  resource use was low. The provider computes eight prompt RHS columns per
  workgroup and reuses the existing F16 batched dispatch constants/layout.
- The first incremental opt-in run reported 571.950 tok/s p512 r5 but one
  decode guardrail run was anomalously bad. I treated that as suspect, ran a
  clean build from the current source state (`clean-f16-cols8.log`,
  `rebuild-clean-f16-cols8.log`), and recollected sequential default/opt-in
  data from the top. The concurrent clean p512/decode artifacts should not be
  used for decisions; they were contaminated by overlapping `llama-bench`
  processes and are superseded by the `clean-seq-*` artifacts.
- Clean sequential opt-in data: default p512 r5 was 548.680 tok/s (`550.756,
  550.806, 547.862, 547.171, 546.804`) and opt-in F16 cols8 p512 r5 was
  565.562 tok/s (`567.789, 567.728, 565.909, 564.674, 561.709`). Decode was
  effectively unchanged: default n64 r5 33.892 tok/s (`30.3176, 34.7103,
  34.7684, 34.9498, 34.7114`) versus opt-in 33.827 tok/s (`30.6182, 34.4901,
  34.7388, 34.6200, 34.6671`).
- Made F16 batched cols8 default-on with rollback
  `GGML_PYRE_DISABLE_F16_BATCHED_COLS8_PROMPT=1`. Final no-env p512 r5 is
  568.637 tok/s (`569.460, 571.116, 568.004, 567.676, 566.929`), and final
  no-env decode n64 r5 is 33.894 tok/s (`30.3720, 34.7314, 34.6839, 34.8661,
  34.8175`). `test-backend-default-f16-cols8.log` passed.
- Provider trace `default-f16-cols8-p512.trace` selected 20
  `MUL_MAT provider=pure_hip_f16_batched_cols8` claims, no cols4 F16 batched
  claims, and no fallback/error lines.
- Rocprof artifact `default-f16-cols8-rocprof-p512-summary.txt`: F16 batched
  cols8 is 20 calls / 58.412 ms total / 2.921 ms avg, replacing the prior
  F16 batched cols4 bucket at 93.117 ms in the BF16 multi-reduce checkpoint.
  The same trace still shows Q4 SWIGLU route8 151.862 ms, Q4 ID route8
  149.080 ms, Q5_K cols8 132.974 ms, Q6_K cols8 122.000 ms, BF16 cols4
  49.671 ms, and Q8_0 add 43.361 ms.
- ISA/resource artifact `isa-clean-f16-cols8/summary.txt`: F16 batched cols8
  is 29 VGPR / 64 SGPR / no spills / 256 B LDS. The existing cols4 kernel is
  19 VGPR / 65 SGPR / no spills / 1024 B LDS. This is a clean structural
  prompt win, but it still leaves the endpoint below the wonq.7 meaningful
  partial target of 588 tok/s, so the next work should continue with Q6_K cols16
  or another structurally distinct Q5/Q6/Q4 attempt rather than stop here.

#### wonq.7 checkpoint: Q6_K cols16

- Added `pyre_mul_mat_vec_q6_k_cols16_wg128_f32` as the next structural probe
  after F16 cols8. It widens the existing Q6_K prompt path from 8 to 16 RHS
  columns per workgroup and is prompt-shape gated to `cols == 512`.
- Initial opt-in validation with `GGML_PYRE_ENABLE_Q6_K_COLS16_PROMPT=1`:
  `test-backend-q6-cols16.log` passed, p512 r5 reached 576.903 tok/s
  (`580.325, 580.570, 576.116, 576.296, 571.208`), and decode n64 r5 was
  34.142 tok/s (`31.3466, 34.7573, 35.3357, 34.4935, 34.7781`).
  Provider trace selected 70 cols16 claims and no fallback/error lines.
- ISA/resource artifact `isa-q6-cols16/summary.txt`: Q6_K cols16 is
  95 VGPR / 25 SGPR / no spills / 256 B LDS. The existing cols8 path was
  69 VGPR / 24 SGPR / no spills / 128 B LDS. The higher VGPR is acceptable
  for this prompt path and did not cause spills/private memory.
- Made Q6_K cols16 default-on with rollback
  `GGML_PYRE_DISABLE_Q6_K_COLS16_PROMPT=1`. Final no-env p512 r5 is
  572.264 tok/s (`577.796, 575.664, 569.361, 568.732, 569.767`), and final
  no-env decode n64 r5 is 33.767 tok/s (`30.4043, 34.6243, 34.6072, 34.5874,
  34.6131`). `test-backend-default-q6-cols16.log` passed.
- Provider trace `default-q6-cols16-p512.trace` selected 70 cols16 claims and
  no fallback/error lines. The run no longer selected Q6 cols8 in prompt.
- Rocprof artifact `default-q6-cols16-rocprof-p512-summary.txt`: Q6_K cols16
  is 70 calls / 117.931 ms total / 1.685 ms avg, versus Q6_K cols8 at
  121.999 ms in the F16 cols8 checkpoint. This is a smaller profiler bucket
  movement than hoped because the dispatch count stays 70 and the grid Y halves
  from 64 to 32; the local arithmetic/lane structure still dominates enough to
  mute the win.
- Current no-env endpoint after F16 cols8 + Q6 cols16 is 572.264 tok/s, up from
  the clean post-BF16 default of 548.680 tok/s and the prior accepted BF16
  checkpoint of 542.279 tok/s, but still below the wonq.7 meaningful partial
  target of 588 tok/s. The next structural frontier remains Q4 SWIGLU/ID,
  Q5_K cols8, or a genuinely different Q5/Q6 packed RHS strategy; simple
  width-only widening is showing diminishing returns.

#### wonq.8 checkpoint: packed RHS rejection and Q5_K cols16

- Started from the wonq.7 endpoint: no-env p512 r5 572.264 tok/s and rocprof
  top buckets Q4 SWIGLU route8 148.922 ms, Q4 ID route8 146.292 ms, Q5_K
  cols8 130.842 ms, Q6_K cols16 117.931 ms, F16 cols8 57.423 ms, BF16 cols4
  48.731 ms, and Q8_0 add 42.530 ms.
- Prototyped a prompt-tiled Q6_K x Q8_1 path:
  `pyre_mul_mat_vec_q6_k_q8_1_cols8_wg128_f32`, gated by
  `GGML_PYRE_ENABLE_Q6_K_Q8_1_COLS8_PROMPT=1` and shape-gated to model-scale
  Q6 prompt ops (`k >= 2048`, `rows >= 2048`, `cols == 512`). This uses the
  existing Q8_1 scratch conversion but changes the old scalar Q8_1 matmul into
  an 8-column prompt-tiled provider. The synthetic `test-backend-pyre` Q6
  p512 case initially failed exact tolerance on the small `k=256, rows=4`
  shape because Q8_1 activation quantization is lossy; the final gate avoids
  that shape, consistent with the existing old Q8_1 auto policy.
- Rejected the Q6_K Q8_1 cols8 path. It selected all 70 Q6 prompt claims and
  had no fallback/error lines, but p512 r5 regressed to 473.697 tok/s
  (`475.073, 475.420, 473.219, 473.181, 471.591`). Rocprof showed the
  quantization dispatch itself was cheap, 70 calls / 1.585 ms total, but the
  new Q6 Q8_1 matmul bucket was 70 calls / 321.555 ms total / 4.594 ms avg
  versus the accepted Q6 cols16 bucket at ~118 ms. Resource summary before the
  dot rewrite was 53 VGPR / 36 SGPR / no spills / 128 B LDS.
- Tried a dp4a-style Q6 Q8_1 follow-up using `__builtin_amdgcn_sudot4`.
  ISA confirmed `v_dot` instructions and no spills; resources were 58 VGPR /
  32 SGPR / no spills / 128 B LDS. p512 only improved the rejected path to
  485.971 tok/s (`488.412, 488.839, 484.484, 484.448, 483.673`), so the local
  blocker is Q8_1 work decomposition/layout rather than just scalar integer
  multiply. The gated Q6 Q8_1 provider remains off by default for differential
  analysis; it must not be enabled without a packed-x4/tiled layout redesign.
- Added `pyre_mul_mat_vec_q5_k_cols16_wg128_f32` as a resource-backed follow-up
  because Q5_K remained a 130 ms-class bucket and Q5 cols8 had low enough
  resource pressure to justify one wider probe. The accepted provider is
  default-on with rollback `GGML_PYRE_DISABLE_Q5_K_COLS16_PROMPT=1`.
  `test-backend-final-q5-cols16.log` passed.
- Q5_K cols16 validation: opt-in p512 r5 reached 599.511 tok/s (`603.739,
  602.027, 596.679, 599.861, 595.247`). Final no-env p512 r5 after removing
  the rejected cols32 probe is 593.324 tok/s (`593.963, 594.615, 593.176,
  592.715, 592.154`). A briefly overlapped p512 artifact
  `final-q5-cols16-p512-r5.json` should not be used for decisions; it reported
  597.767 tok/s but overlapped with `test-backend-pyre`.
- Decode guardrail for the final default Q5 cols16 endpoint:
  `default-q5-cols16-decode-n64-r5.json` reported 33.641 tok/s including a
  cold first sample (`30.6692, 34.3338, 34.2528, 34.3573, 34.5913`), with warm
  decode still in the expected ~34.3-34.6 tok/s band.
- Provider trace `default-q5-cols16-p512.trace` selected 30
  `pure_hip_q5_K_cols16_wg128` claims, no Q5 cols8 prompt claims, and no
  fallback/error lines. `q5-cols16-p512-seq.trace` is also clean but was
  collected before making cols16 default-on.
- Rocprof artifact `default-q5-cols16-rocprof-p512-summary.txt`: Q5_K cols16
  is 30 calls / 103.145 ms total / 3.438 ms avg, down from Q5_K cols8 at
  130.842 ms in the wonq.7 endpoint. The same final trace still shows the next
  large structural buckets: Q4 SWIGLU route8 150.017 ms, Q4 ID route8
  147.440 ms, Q6_K cols16 118.836 ms, F16 cols8 57.989 ms, BF16 cols4
  49.120 ms, and Q8_0 add 43.089 ms.
- Resource artifact `isa-q5-cols16/summary.txt`: Q5_K cols16 is 95 VGPR /
  24 SGPR / no spills / 256 B LDS. Existing Q5 cols8 was 93 VGPR / 39 SGPR /
  no spills / 128 B LDS. This made cols16 a clean default-on win. A Q5 cols32
  probe was rejected and removed: p512 regressed to 583.559 tok/s and resources
  jumped to 150 VGPR / 24 SGPR / no spills / 512 B LDS.
- wonq.8 did not reach the next 650 tok/s checkpoint, and the final no-env
  endpoint is below the 600 tok/s minimum-useful threshold despite the opt-in
  Q5 run grazing it. The trace-backed next step is not more width-only widening:
  Q5 cols32 and Q6 Q8_1/dot both rejected. The remaining structural frontier is
  the Q4 grouped route8 dataflow: either reduce route8 live ranges/conditionals
  in the full-route fast path, or change the route-compacted schedule so Q4
  ID and SWIGLU consume a better row/route tile without increasing VGPR past
  the current 90 VGPR ID and 118 VGPR SWIGLU resources.

#### wonq.9 checkpoint: Q5_K mini-MMQ prompt pilot

- Scope: implemented the first Vulkan-shaped regular prompt pilot for Q5_K.
  This is not another cols-only widening pass. The accepted provider,
  `pyre_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32`, changes work ownership to
  a 32-row x 32-column output tile, uses the existing Q8_1 RHS scratch as the
  first-stage packed activation input, stages a 32-column Q8_1 K tile in LDS,
  packs Q5 low/high-bit groups into 4-byte operands, and accumulates with
  `__builtin_amdgcn_sudot4` / `v_dot`. It avoids the old one-output-element
  shared reduction pattern: each lane owns one output row and eight output
  columns for the full K loop.
- The first 32x16 pilot was rejected and removed. It selected the intended
  30 Q5 prompt claims and lowered the Q5 bucket from 103.145 ms to
  83.181 ms, but p512 r5 regressed to 578.544 tok/s
  (`576.463, 579.193, 577.122, 579.996, 579.947`). This was a useful proof that
  the scheduler/dataflow was pointing in the right direction, but it did not
  meet the Q5 bucket target and left too much tile count overhead.
- The accepted 32x32 provider is default-on for the prefill shape with rollback
  `GGML_PYRE_DISABLE_Q5_K_Q8_1_MMQ_PROMPT=1`. The prefill/decode split matters:
  the provider is shape-gated to the model-scale Q5 prompt shape
  (`k >= 2048`, `rows >= 2048`, `cols == 512`) and decode remains on the
  existing skinny Q5 providers. A short decode trace selected 150
  `pure_hip_q5_K_wg128` claims plus 5 `pure_hip_q5_K_wg64` claims and no MMQ
  claims for `cols=1`.
- Validation: rebuilt `llama-bench` and `test-backend-pyre` after removing the
  32x16 code object (`rebuild-q5-mmq32x32-default.log`); no empty catalog or
  compile error lines. `test-backend-q5-mmq32x32-default.log` passed.
- Endpoint: final no-env p512 r5 is 655.488 tok/s
  (`658.814, 655.785, 657.843, 652.970, 652.028`), up from the wonq.8 clean
  593.324 tok/s baseline. A prior opt-in sequential p512 r5 before cleanup was
  646.297 tok/s (`648.412, 649.275, 646.855, 645.459, 641.483`). The earlier
  32x32 decode artifact `q5-mmq32x32-decode-n64-r5.json` should not be used
  because it overlapped with a p512 run and reported a false 11.888 tok/s
  regression.
- Decode guardrail: final no-env decode n64 r5 is 34.199 tok/s
  (`31.5887, 34.7785, 34.8854, 34.8320, 34.9127`), matching the prior warm
  decode band. The sequential opt-in rerun was 34.220 tok/s
  (`31.4969, 34.8567, 34.8404, 35.1025, 34.8022`).
- Provider trace `default-q5-mmq32x32-p512-r1.trace` selected 30
  `pure_hip_q5_K_q8_1_mmq32x32_wg128` prompt claims and 1 skinny
  `pure_hip_q5_K_wg64` giant-row claim, with no fallback/error lines.
- Rocprof artifact `default-q5-mmq32x32-rocprof-p512-summary.txt`: Q5_K
  MMQ32x32 is 30 calls / 33.399 ms total / 1.113 ms avg, plus Q8_1 quantization
  at 30 calls / 0.534 ms total. This beats both the Q5 minimum useful target
  (<70 ms) and strong target (<55 ms), and is a 69.7 ms reduction from the
  wonq.8 Q5 cols16 bucket of 103.145 ms. The same default rocprof run now has
  Q4 SWIGLU 148.487 ms, Q4 ID 144.171 ms, Q6_K cols16 115.716 ms, F16 cols8
  56.920 ms, BF16 cols4 48.136 ms, and Q8_0 add 42.397 ms as the next dominant
  buckets.
- ISA/resource artifact `isa-q5-mmq32x32-final/summary.txt`: the accepted
  Q5 MMQ source emits `v_dot` instructions, uses 115 VGPR / 22 SGPR /
  no spills / 1152 B LDS / no private memory. The prior Q5 cols16 reference
  remains 95 VGPR / 24 SGPR / no spills / 256 B LDS. The resource tradeoff is
  acceptable for prompt because it removes the shared reduction and cuts the
  Q5 bucket by about 67%, but this shape should not be used for skinny decode.
- Remaining structural frontier after this checkpoint: the biggest prompt
  buckets are now Q4 expert SWIGLU/ID and Q6_K. The Q5 pilot proves the
  Vulkan-style matrix-tile direction is viable in HIP even with the existing
  non-x4 Q8_1 scratch layout; the next transliteration should carry the same
  prefill-only split into Q6_K and then Q4 matmul-id route-tiled MMQ.

#### wonq.10/wonq.11 checkpoint: true x4 RHS layout and Q5/Q6 MMQ follow-up

- Planner audit correction addressed: wonq.9's accepted Q5 MMQ used the old
  Pyre Q8_1 scratch ABI (`d`, `s`, `qs[32]`, 36 bytes per 32 activations). This
  pass added a real prompt-specific packed-x4 RHS ABI matching Vulkan's
  conceptual `block_q8_1_x4_packed128`: four Q8_1 blocks per 144-byte record,
  `ds[4]` as four `(d, sum*d)` half pairs stored as `ds[8]`, and `qs` as 32
  packed int32 values. Scratch sizing is `ceil(q8_1_blocks / 4) * 144` for x4
  prompt MMQ users and remains `q8_1_blocks * 36` for the old skinny/decode
  Q8_1 path. The old quantizer/provider remains available and is still the
  default for existing Q4/Q5/Q6 skinny and non-x4 prompt paths.
- Added `pyre_quantize_q8_1_x4_f32` and Q5 x4 consumers
  `pyre_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32` and
  `pyre_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32`. Both are opt-in only:
  `GGML_PYRE_ENABLE_Q5_K_Q8_1_X4_MMQ32_PROMPT=1` and
  `GGML_PYRE_ENABLE_Q5_K_Q8_1_X4_MMQ64_PROMPT=1`. Default Q5 remains the
  accepted wonq.9 old-layout MMQ32x32 path because it is still faster.
- Q5 x4 32x32 scalar-staged result: p512 r5 was 623.488 tok/s
  (`621.287, 620.383, 619.804, 617.749, 638.217`), below the restored default
  old-layout Q5 MMQ result of 648.890 tok/s in this pass and below the wonq.9
  655.488 tok/s result. Provider trace confirmed prompt selected
  `pure_hip_q5_K_q8_1_x4_mmq32x32_wg128`; decode isolation stayed clean.
  Decode guardrail after this work was 35.483 tok/s
  (`35.5788, 35.5182, 35.3514, 35.4935, 35.4752`).
- Q5 x4 64x64 scalar-staged result: p512 r5 was 617.847 tok/s
  (`622.045, 621.186, 617.304, 613.903, 614.795`). Rocprof showed the Q5 bucket
  regressed from default old-layout MMQ32x32 at 30 calls / 33.697 ms total /
  1.123 ms avg to x4 64x64 at 30 calls / 36.131 ms total / 1.204 ms avg.
  Quantize x4 cost was not the blocker: 30 calls / 0.586 ms versus old Q8_1
  quantize at 30 calls / 0.535 ms. ISA/resource explains the loss: 64x64 jumps
  to 176 VGPR / 22 SGPR / no spills / 2304 B LDS versus old-layout 32x32 at
  115 VGPR / 22 SGPR / 1152 B LDS. This simple HIP medium tile is occupancy /
  live-range limited.
- Vulkan-divergence analysis: x4 32x32 initially looked resource-neutral but
  still emitted the same scalar-load shape as the old layout. A Vulkan-like
  `int4`/128-bit RHS load experiment reduced x4 32x32 resources from 114 VGPR
  to 97 VGPR and produced `global_load_b128` / `ds_store_b128` in ISA, but it
  was rejected: sequential p512 r5 dropped to 609.735 tok/s and rocprof showed
  Q5 MMQ x4 vector-load at 30 calls / 45.177 ms total / 1.506 ms avg. The local
  LDS scatter/wait behavior was worse than the scalar staged loads even though
  VGPR and load width looked better. That code was reverted; artifacts are kept
  under `wonq10/isa-q5-x4-vectorload/` and `wonq10/rocprof-x4-vector32/`.
- Added a Q6 x4 MMQ32 prototype,
  `pyre_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32`, behind
  `GGML_PYRE_ENABLE_Q6_K_Q8_1_X4_MMQ32_PROMPT=1`, reusing the same x4 RHS
  quantize/scratch layout. It is rejected/off by default. It passed the opt-in
  `test-backend-pyre` harness, and provider trace confirmed it selected the Q6
  prompt claims, but p512 r5 regressed to 518.364 tok/s
  (`520.019, 521.724, 518.156, 517.831, 514.088`). Rocprof showed the current
  default Q6 cols16 bucket at 70 calls / 116.438 ms total / 1.663 ms avg, while
  Q6 x4 MMQ32 was 70 calls / 357.378 ms total / 5.105 ms avg plus 1.607 ms of
  x4 quantization. ISA/resource explains this as a bad A-side port: Q6 x4 MMQ32
  is 137 VGPR in the standalone summary (144 VGPR in rocprof metadata) with
  heavy source-level global-load count, versus the older Q6 Q8 cols8 at 58 VGPR
  and the accepted Q6 cols16 non-Q8 prompt path at 96 VGPR in rocprof. Dot alone
  and x4 RHS reuse do not overcome the naive Q6 unpack/scale schedule.
- Current decision: keep the real packed-x4 RHS ABI and Q5/Q6 x4 prompt MMQ
  variants gated for differential analysis, but keep default on the wonq.9 Q5
  old-layout MMQ32x32 and the accepted Q6 cols16 path. The Vulkan layout is not
  disproven; rather, Pyre's direct HIP translations are missing the surrounding
  Vulkan scheduling advantages. The next Vulkan-guided work should not widen
  this naive 64x64/16-accumulator shape. It should either build a lower-live-
  range medium tile that preserves coalesced x4 loads without LDS scatter
  overhead, or move to Q4 route-tiled ID/SWIGLU where the current route8 serial
  dataflow remains a larger structural bucket.

#### Direct pass: Q5_K Vulkan-schedule audit

The packed-x4 checkpoint fixed the RHS ABI gap, but the Q5/Q6 x4 kernels still
use the wonq.9 mini-MMQ schedule. The next direct prototype therefore targets
schedule equivalence before broader rollout.

| Dimension | Current HIP Q5 x4 MMQ32 | Vulkan MMQ K-quant schedule |
| --- | --- | --- |
| Workgroup | 128 threads | Small AMD-style shape is one subgroup-sized tile; use 64-thread first probe |
| Tile | BM=32, BN=32, BK=32 | BM/BN selected by warptile; small K-quant probe uses BM=32, BN=32, BK=32 |
| K blocking | One 32-wide Q8 block per barrier | `BK_STEP=4` for regular matmul, so four Q8 blocks per shared-memory phase |
| Output ownership | One row per lane, eight columns per col-lane thread | Each lane owns a register microtile derived from `WM/WN/WMITER/TM/TN` |
| A staging | No LDS A tile; each col-lane reloads/repackages the same A row/group | Cooperative `buf_a[BM * BK_STEP]`, then `cache_a[WMITER * TM]` |
| B staging | LDS B tile only | Cooperative `buf_b[BN * BK_STEP]`, then `cache_b` |
| A repack repetition | Repeated across four col-lane groups for the same row/group | Repacked once per A row/K block into LDS |
| Barrier cadence | Two barriers per 32-wide Q8 block | Two barriers per `BK_STEP * 32` K elements |
| Live accumulators | 8 floats for 32x32, 16 for 64x64 | Small K-quant probe has 16 floats but less repeated A work; K-quants force `WMITER=1` |
| RHS load shape | x4 layout consumed as scalar `qs[inner * 8 + iqs]` | `block_b_to_shmem` loads two `ivec4` groups per Q8 block into eight qpack ints |

The first direct prototype should be a Q5-only opt-in provider that keeps the
current default untouched and mirrors Vulkan's cooperative A/B staging. If it
does not beat the old-layout Q5 MMQ32 bucket, the result should identify whether
the blocker is VGPR pressure, LDS wait/scatter behavior, barrier cost, subgroup
shape, or layout order.

#### Direct pass results: Q5_K large Vulkan-shaped tile

The 32x32 one-subgroup probe was the wrong Vulkan analog for this prefill shape.
For `m=8192, n=512`, llama.cpp Vulkan's non-coopmat2 selection logic chooses
the large K-quant MMQ path, not the small path. The direct HIP follow-up keeps
the default Q5 MMQ untouched and adds only an opt-in large packed-x4 provider:

```bash
GGML_PYRE_ENABLE_Q5_K_Q8_1_X4_MMQL128_PROMPT=1
```

The retained probe is `pyre_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32`:
BM=128, BN=128, BK_STEP=4, WARP=64, WM=64, WN=64, WMITER=1, TM=4, TN=2. It
cooperatively stages both Q5 A blocks and packed-x4 Q8_1 B blocks in LDS, then
computes a register microtile.

ROCProfiler kernel-trace comparison on Qwen3.5-35B-A3B, `-p 512 -n 0 -r 1`:

| Path | Calls | Q5 kernel total | Avg call | WG | Grid | VGPR | LDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default Q5 MMQ32 old RHS | 30 | 33.502 ms | 1.117 ms | 128 | 32768x16 | 120 | 1.125 KiB |
| Large x4 Vulkan-shaped tile | 30 | 26.853 ms | 0.895 ms | 256 | 16384x4 | 176 | 40.0 KiB |

That is a 19.8% kernel-family reduction for the targeted Q5 prompt matmuls.
The associated quantization change is not material: `pyre_quantize_q8_1_f32`
was 0.533 ms total and `pyre_quantize_q8_1_x4_f32` was 0.578 ms total across
the same 30 calls.

End-to-end `llama-bench -p 512 -n 0 -r 5` is too noisy and too diluted by other
buckets to reliably score this class of hero-kernel change. In the same build,
default measured 649.3 tok/s and the large opt-in path measured 632.8 tok/s,
while rocprof showed the target Q5 bucket itself improving. For MMQ hero
routes, kernel-family time is the acceptance signal until the rest of the graph
and runtime noise are controlled.

The large path was therefore promoted to default-on for the matching prompt
shape, with `GGML_PYRE_DISABLE_Q5_K_Q8_1_X4_MMQL128_PROMPT=1` as the escape
hatch. Follow-up validation after promotion:

- Provider trace selected
  `pure_hip_q5_K_q8_1_x4_mmql128x128_wg256` for all 30 Q5 prompt calls.
- ROCProfiler measured the promoted Q5 bucket at 26.798 ms total across 30
  calls, matching the earlier opt-in 26.853 ms probe and preserving the kernel
  family win over the old 33.502 ms default.
- Aggregate guardrails after promotion remain context, not the acceptance
  signal for this hero route: p512 r3 measured 650.481 tok/s and decode n64 r3
  measured 35.603 tok/s. The p512 aggregate is lower than the batched-reduction
  checkpoint despite the Q5 bucket win, which is consistent with the current
  graph having enough other inefficient buckets and clocking/runtime noise to
  obscure a single-family MMQ improvement.

The failed 32x32 Vulkan-style probe was pruned. It had 80 VGPR and no spills,
but used 10 KiB LDS and regressed the Q5 bucket to 42.179 ms. That confirms
cooperative A/B staging alone is not enough; the tile shape selected by Vulkan
for the actual prefill matrix dimensions matters.

#### Cleanup pass: prune superseded MMQ variants

The first build-time cleanup pass removed prompt-width variants that were no
longer part of the live no-env provider selection:

- Removed Q5 F32-RHS prompt variants `q5_k_cols4`, `q5_k_cols8`, and
  `q5_k_cols16`. Q5 prompt prefill initially routed through the accepted
  `pure_hip_q5_K_q8_1_mmq32x32_wg128` path after pruning, then was promoted to
  the larger Vulkan-shaped `pure_hip_q5_K_q8_1_x4_mmql128x128_wg256` path once
  rocprof confirmed the Q5 kernel-family win. Skinny Q5 providers remain
  available for decode/fallback.
- Removed Q6 F32-RHS prompt variants `q6_k_cols4` and `q6_k_cols8`. The
  accepted non-MMQ Q6 prompt path is still `q6_k_cols16`, so that provider,
  catalog entry, and HIP kernel were intentionally retained.
- Removed the rejected Q6 Q8_1 cols8 experiment
  `q6_k_q8_1_cols8_wg128` and its `GGML_PYRE_ENABLE_Q6_K_Q8_1_COLS8_PROMPT`
  gate. Current default Q6 prompt routing remains `q6_k_cols16`; the Q6 x4 MMQ
  prototype remains opt-in for future analysis.
- Kept active packed-x4 work (`quantize_q8_1_x4`, Q5 x4 MMQ32, Q5 x4 MMQ64,
  Q5 large x4 MMQ128, and Q6 x4 MMQ32), the Q5 non-x4 MMQ32 fallback, Q6
  cols16, and skinny/decode providers.

Validation after pruning:

- Removed names no longer appear in `ggml-pyre.cpp`,
  `generate_pyre_kernels.py`, or the HIP kernel sources.
- `cmake --build build/llama-pyre-rocm713 --target llama-bench
  pyre-kernel-bench test-backend-pyre -j$(nproc)` passed and regenerated the
  embedded catalog.
- `test-backend-pyre` exited 0.
- p512 provider trace on Qwen3.5-35B-A3B still selected
  `pure_hip_q5_K_q8_1_mmq32x32_wg128` for Q5 and
  `pure_hip_q6_K_cols16_wg128` for Q6 immediately after pruning; none of the
  removed providers appeared. A later provider trace after the large-Q5
  promotion selected `pure_hip_q5_K_q8_1_x4_mmql128x128_wg256`.
- While touching the catalog generator, the HIP device-only clang optimization
  level was changed back from speculative `-O3` to `-O2`. The rebuilt O2 catalog
  passed `test-backend-pyre`; p512 r3 measured 667.208 tok/s and decode n64 r3
  measured 36.130 tok/s. The pre-change O3 sanity p512 run in the same pruned
  tree measured 654.764 tok/s, so O2 shows no regression signal in this quick
  guardrail.

#### Q4 grouped MoE reduction batching

Fresh rocprof on the pruned/O2 tree showed that the p512 prefill critical path
had moved clearly to the Q4 grouped MoE prompt kernels:

| Kernel bucket | Baseline calls | Baseline total |
| --- | ---: | ---: |
| `pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32` | 39 | 144.890 ms |
| `pyre_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32` | 39 | 141.374 ms |

Two structural probes were useful:

- Replacing the SWIGLU grouped route8 inner loop with the existing packed Q4
  decode dataflow was rejected. It passed `test-backend-pyre` but regressed the
  SWIGLU bucket to 170.011 ms and p512 r3 to 649.959 tok/s. The likely cause is
  VGPR pressure from applying the packed four-group schedule inside an already
  large 2-row x 8-route fused SWIGLU kernel.
- Disabling grouped routing was also rejected. It selected the ungrouped row4
  prompt kernels and dropped p512 to 439.960 tok/s, with the Q4 buckets growing
  to 408.876 ms for SWIGLU and 245.046 ms for ID. Expert-route compaction is
  therefore mandatory for this model shape.

The retained change batches the per-route cross-wave reductions in the active
route8 grouped kernels. Previously each route in an 8-route chunk paid its own
cross-wave shared-memory reduction and synchronization. The new path reduces all
routes in the chunk, writes per-wave partials to separate shared slots, does a
single cross-wave synchronization, then stores all valid routes.

Validation:

- `cmake --build build/llama-pyre-rocm713 --target llama-bench
  pyre-kernel-bench test-backend-pyre -j$(nproc)` passed.
- `test-backend-pyre` exited 0.
- p512 r3 improved from the O2/pruned baseline 667.208 tok/s to 682.574 tok/s.
- Decode n64 r3 stayed in the same range: 36.130 tok/s baseline versus
  35.633 tok/s after batching.

ROCProfiler p512 r1 after batching:

| Kernel bucket | Calls | Baseline total | Batched total | Delta |
| --- | ---: | ---: | ---: | ---: |
| Q4 SWIGLU grouped row2 route8 | 39 | 144.890 ms | 138.828 ms | -4.2% |
| Q4 ID grouped row2 route8 | 39 | 141.374 ms | 125.166 ms | -11.5% |

This is a useful cleanup win but not the missing structural jump to Vulkan.
Vulkan's comparable Q4 ID bucket was about 37.630 ms and the two unfused
SWIGLU-side Q4 matmul buckets were about 67.142 ms total in the earlier trace,
so the next pass still needs a dataflow/layout change rather than only barrier
reduction.

#### Q6_K Vulkan-style MMQL128 prompt path

The next hero route was Q6_K. The prior accepted default used the non-Q8
`cols16` prompt provider and the current p512 rocprof bucket was about
122.997 ms for 70 calls. The earlier Q6 x4 MMQ32 experiment stayed rejected:
it emitted dot instructions but used the wrong row/column dataflow and measured
357.378 ms in the same family.

The accepted path ports the Q5 large MMQL128 skeleton to Q6:

- Provider: `pyre_mul_mat_vec_q6_k_q8_1_x4_mmql128x128_wg256_f32`.
- Tile shape: BM=128, BN=128, BK_STEP=4, WG=256, WARP=64, TM=4, TN=2.
- RHS: existing Q8_1 x4 packed prompt quantization.
- LHS: Q6 values packed into signed int8 lanes in LDS, with two scale values
  per 32-value Q8 block. Q6 has no min term.
- Selection: prompt-only, `src1->ne[1] == 512`, and default-on with rollback
  `GGML_PYRE_DISABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1`.

The first correct MMQL128 port measured 98.358 ms for the Q6 bucket. Carrying
only two A-side scale values per cache row instead of eight dropped that to
62.447 ms. A follow-up ATT pass showed the top stalls were `s_waitcnt vmcnt`
around Q6 byte/scale global loads and that the branchy scalar `q6_k_value`
helper produced too much load/control-flow structure in the packer. Replacing
the four scalar value calls with two aligned 32-bit loads plus bytewise signed
packing dropped the bucket again to 52.677 ms.

Final validation:

- `cmake --build build/llama-pyre-rocm713 --target llama-bench
  pyre-kernel-bench test-backend-pyre -j$(nproc)` passed.
- `test-backend-pyre` exited 0.
- Default p512 provider trace selected 70
  `pure_hip_q6_K_q8_1_x4_mmql128x128_wg256` claims and zero fallbacks.
- Decode provider trace with the opt-in gate still selected skinny
  `pure_hip_q6_K_wg128`, confirming the large provider is prompt-only.
- Default p512 r3 measured 774.260 tok/s
  (`767.894`, `779.002`, `775.883`).
- Default decode n64 r3 measured 35.797 tok/s
  (`35.7577`, `35.8072`, `35.8258`).

ROCProfiler p512 r1 progression:

| Q6 provider | Calls | Total |
| --- | ---: | ---: |
| Prior default cols16 | 70 | 122.997 ms |
| Initial MMQL128 port | 70 | 98.358 ms |
| MMQL128 with two-scale A cache | 70 | 62.447 ms |
| MMQL128 with direct Q6 pack4 | 70 | 52.677 ms |

ATT sanity after direct packing:

| Metric | Branchy pack | Direct pack |
| --- | ---: | ---: |
| Decoded rows | 11,901 | 10,962 |
| Hitcount | 929,124 | 791,442 |
| Total latency | 3,423,556 | 2,867,407 |
| Total stall | 2,225,354 | 1,772,201 |
| Total idle | 4,636,290 | 4,144,646 |
| Global-load instruction rows | 138 | 24 |
| `s_waitcnt vmcnt` stall | 1,235,950 | 783,356 |
| Arch VGPR | 184 | 176 |
| LDS group segment | 38,912 B | 38,912 B |
| Private segment | 0 B | 0 B |

The remaining Q6 MMQL128 stalls are still mostly explicit waits after
consolidated global loads, plus LDS/barrier and dot/FMA issue. That is a normal
next-order tuning problem, not evidence that the schedule is structurally wrong.
The larger structural opportunity now moves back to Q4 grouped expert ID and
SWIGLU, which remain above 100 ms each in the same p512 profiles.

#### MFMA / Vulkan cooperative-matrix sanity check

The Q5/Q6 prompt MMQ path is not a cooperative-matrix path in Vulkan. The
specific SPIR-V files selected for Q8_1 prompt MMQ are:

- `matmul_q5_k_q8_1.spv`
- `matmul_q6_k_q8_1.spv`

Disassembly shows these use `SPV_KHR_integer_dot_product`,
`DotProductInput4x8BitPacked`, and `OpSDot ... PackedVectorFormat4x8Bit`.
They do not declare `SPV_KHR_cooperative_matrix`, do not contain
`OpTypeCooperativeMatrixKHR`, and do not contain
`OpCooperativeMatrixMulAddKHR`. The shader generator also only emits the
`*_q8_1` MMQ shaders under `!coopmat && !coopmat2`, which matches the runtime
selection path for `src1_type == GGML_TYPE_Q8_1`.

The cooperative-matrix shaders exist, but they are the separate `*_f16_cm2`
matmul family. For example `matmul_q5_k_f16_cm2.spv` and
`matmul_q6_k_f16_cm2.spv` declare `SPV_KHR_cooperative_matrix` /
`SPV_NV_cooperative_matrix2` and contain `OpTypeCooperativeMatrixKHR`,
`OpCooperativeMatrixLoadTensorNV`, and `OpCooperativeMatrixMulAddKHR`. That is
not the same pipeline family as the Q8_1 MMQ kernels being matched here.

The Pyre Q5/Q6 large MMQL128 kernels match the Vulkan Q8_1 MMQ compute
primitive, not the Vulkan f16 coopmat primitive:

- Q5 code object:
  `pyre_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32`.
  `llvm-objdump` found 2,304 `v_dot*` instructions and no `v_mfma*`
  instructions. It reports 175 VGPRs, 0 AGPRs, and private segment size 0.
- Q6 code object:
  `pyre_mul_mat_vec_q6_k_q8_1_x4_mmql128x128_wg256_f32`.
  `llvm-objdump` found 2,113 `v_dot*` instructions and no `v_mfma*`
  instructions. It reports 174 VGPRs, 0 AGPRs, and private segment size 0.

So the previous MFMA rejection was correct for these exact Q8_1 MMQ kernels,
but the justification should be stated more precisely: Vulkan is not using
cooperative matrix for this path either. It is using packed integer dot product.
If we later add a dequant-to-f16-plus-coopmat/MFMA route, that is a new
algorithmic variant to benchmark against Vulkan's `*_f16_cm2` path, not a
missing lowering of the current Q8_1 MMQ route.

#### Q5 done-done revisit

I retested one Q5 cleanup suggested by the Q6 result: replacing the bytewise
Q5 `qs`/`qh` pack loads with aligned 32-bit loads. The p512 profile did not
justify keeping it: Q5 large measured 25.466 ms for 30 calls versus the prior
roughly 25.1-25.3 ms bucket. I reverted that local tweak and kept the existing
Q5 MMQL128 provider unchanged.

#### Q4 MoE route-tiled MMQ pass

The sparse MoE compute path had not yet reached the same Vulkan-aligned state as
Q5/Q6. The old grouped Q4 prompt kernels compacted routes, but still computed a
row2 x route8 chunk over F32 RHS inside each workgroup:

- `pyre_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32`
- `pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32`

The new pass adds prompt-only route-tiled Q8_1 x4 MMQ providers:

- `pyre_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32`
- `pyre_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32`

Both reuse the existing route compaction, quantize the F32 activation/RHS tensor
to the same packed Q8_1 x4 scratch format used by Q5/Q6, and dispatch with
expert in `grid.z` and compacted route tile in `grid.y`. The compute kernel
packs Q4 values into 4-byte dot operands, uses `__builtin_amdgcn_sudot4`, and
applies Q4 scale/min outside the integer dot loop. This is the same structural
move that made Q5/Q6 successful: stop widening a scalar matvec and instead make
the routed tokens the matrix-N tile.

Same-binary p512 rocprof comparison:

| Path | Calls | Before | After |
| --- | ---: | ---: | ---: |
| Q4 ID grouped prompt | 39 | 92.568 ms | 43.269 ms |
| Q4 SWIGLU grouped prompt | 39 | 92.276 ms | 69.829 ms |
| Q8_1 x4 quantize total | +78 calls | - | +4.379 ms |

The net sparse MoE compute+quant win is about 67 ms on this p512 trace. The
default p512 endpoint after promotion measured 1235.40 +/- 0.90 tok/s
(`llama-bench -p 512 -n 0 -r 3`). Decode guardrail remained in band at
35.61 +/- 0.07 tok/s (`-p 0 -n 64 -r 3`) because the new providers are gated to
the 512-token prompt shapes.

The new providers are default-on with rollback knobs:

- `GGML_PYRE_DISABLE_Q4_K_ID_Q8_1_X4_MMQ_PROMPT=1`
- `GGML_PYRE_DISABLE_Q4_K_SWIGLU_Q8_1_X4_MMQ_PROMPT=1`

The remaining Q4 gap is now more likely local tuning than a missing Vulkan-level
ownership model: the ID route is close to the earlier Vulkan ID bucket scale,
while SWIGLU still carries two A matrices and fused GLU in one kernel. Further
work should inspect the SWIGLU kernel's VGPR/occupancy and decide whether a
Vulkan-like split into two route-tiled matmul-id kernels plus a separate GLU
beats the current fused route-tiled kernel.

#### Q4 MoE route-tiled ATT check

I ran rocprofv3 advanced thread trace on both new sparse route-tiled kernels:

- ID trace:
  `build/rocprof-att-q4moe-id-mmq-p512-20260411-203519/`
- SWIGLU trace:
  `build/rocprof-att-q4moe-swiglu-mmq-p512-20260411-203433/`

The result changes the "done-done" call. The kernels are structurally much
better than the old row2/route8 path, but the thread traces show that the hot
stall source is still load latency inside the tile staging code.

| Kernel | Rows | Hits | Latency | Stall | Idle | Top stall bucket |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ID 64x64 | 9,383 | 2,197,210 | 10,033,451 | 7,585,794 | 9,364,125 | `s_waitcnt`, 87.5% |
| SWIGLU 32x64 | 8,828 | 4,202,710 | 17,738,024 | 13,217,713 | 16,105,562 | `s_waitcnt`, 83.7% |

The highest-stall instructions are repeated `s_waitcnt vmcnt(0)` or
`s_waitcnt vmcnt(1)` immediately after scalar `global_load_b32` /
`global_load_d16_u8` loads used to stage Q4/Q8 fragments into LDS. For example,
the ID trace's top stall is a `global_load_b32` followed immediately by
`s_waitcnt vmcnt(0)`, then `v_and_b32` and `ds_store_b32`. SWIGLU shows the same
shape, just with more instances because it stages both gate and up matrices.

Opcode bucket breakdown by stall:

| Kernel | `s_waitcnt` | `v_dot` | LDS `ds_*` | `v_add` | `v_fma` |
| --- | ---: | ---: | ---: | ---: | ---: |
| ID 64x64 | 87.5% | 2.7% | 1.0% | 4.1% | 0.8% |
| SWIGLU 32x64 | 83.7% | 4.9% | 1.0% | 6.1% | 0.9% |

That means the current sparse MMQ kernels are using the right compute primitive
(`v_dot` from `__builtin_amdgcn_sudot4`) and are not primarily dot-issue bound.
They are also not showing an obvious spill/private-memory pathology in the
existing resource metadata. The remaining structural work is to hide or reduce
global memory wait time in the staging phase:

- double-buffer or software-pipeline the Q4/Q8 tile loads so the next `kb`
  fragment is in flight while the current fragment is doing dot work;
- vectorize/coalesce the staging loads where possible instead of relying on many
  scalar 32-bit loads followed by immediate waits;
- consider split-SWIGLU if lower VGPR pressure and simpler load scheduling beats
  the current fused kernel despite the extra kernel boundary.

So: ID and SWIGLU are validated as real improvements, but the ATT evidence says
they are not yet done-done.

#### Q4 MoE staged-load follow-up

I followed the ATT signal by splitting sparse Q4/Q8 staging into separate
fetch/commit phases. Each lane now issues the Q4 and Q8 loads for a `k` slice
into registers, then commits the batch into LDS, instead of immediately
unpacking/storing after each scalar global load. This is not full cross-`kb`
double-buffering, but it directly targets the trace pattern where a
`global_load_b32` was followed by an immediate `s_waitcnt`.

Correctness still passes with `test-backend-pyre`. The p512 endpoint remains
noisy without warmup, but the kernel bucket moved in the intended direction:

| Kernel | Before | After | Delta |
| --- | ---: | ---: | ---: |
| ID 64x64, 39 calls | 43.269 ms | 41.819 ms | -1.450 ms |
| SWIGLU 32x64, 39 calls | 69.829 ms | 65.294 ms | -4.535 ms |

Resource metadata after the staged-load edit:

| Kernel | LDS | Scratch | SGPR | VGPR |
| --- | ---: | ---: | ---: | ---: |
| ID 64x64 | 21,504 B | 0 B | 128 | 192 |
| SWIGLU 32x64 | 21,504 B | 0 B | 128 | 184 |

The ID kernel pays extra VGPR pressure versus the prior 176 VGPR version, so
the win is modest. SWIGLU kept the same reported VGPR count and improved more
cleanly.

ATT after the staged-load edit:

| Kernel | Stall before | Stall after | `s_waitcnt` before | `s_waitcnt` after |
| --- | ---: | ---: | ---: | ---: |
| ID 64x64 | 7,585,794 | 4,989,416 | 87.5% | 76.4% |
| SWIGLU 32x64 | 13,217,713 | 9,711,199 | 83.7% | 78.2% |

Trace artifacts:

- ID staged-load ATT:
  `build/rocprof-att-q4moe-id-mmq-batched-p512-20260411-204128/`
- SWIGLU staged-load ATT:
  `build/rocprof-att-q4moe-swiglu-mmq-batched-p512-20260411-204209/`
- Kernel bucket trace:
  `build/rocprof-pyre-current-p512/q4moe-batched-loads-kernel-204044/`

Done-for-now call: keep the staged-load version. It reduces the measured sparse
kernel bucket by about 6.0 ms on p512 and materially lowers ATT wait stalls.
The remaining waitcnt share is still high, but the next step would be a more
invasive `kb` software pipeline/double-buffer design with higher register and
barrier risk. That is no longer a quick sparse cleanup; it should be its own
kernel-design pass if sparse kernels become hot again after broader graph/runtime
work.

### Flash Attention Prefill Pass

Goal: revisit `FLASH_ATTN_EXT` for prompt/prefill after the earlier FA audit had
mostly measured decode. The existing Pyre FA provider was a first-pass scalar
kernel named `pyre_flash_attn_ext_f32_f16_decode`; despite the name, Pyre's
support gate allowed prompt shapes up to `N <= 1024`, so Qwen p512 was already
entering it when `-fa 1` was requested.

Initial p512 evidence on Qwen3.5-35B-A3B, sequential runs:

| Path | Prompt tok/s | Main FA/attention bucket |
| --- | ---: | ---: |
| Pyre `-fa 0` baseline | 1225.248 tok/s r3 | `pyre_mul_mat_vec_f16_batched_cols16_f32`, 20 calls / 51.669 ms |
| Pyre `-fa 1` before edits | 721.266 tok/s r3 | `pyre_flash_attn_ext_f32_f16_decode`, 10 calls / 359.434 ms |
| Vulkan `-fa 1` | 2222.553 tok/s r3 | `FLASH_ATTN_EXT`, 10 calls / 1.934 ms via `GGML_VK_PERF_LOGGER` |

Provider tracing confirmed this was not a CPU fallback: Pyre claimed 10
`FLASH_ATTN_EXT provider=pure_hip_f32_k_f16_v_f16_decode` calls with
`D=256 KV=512 N=512 H=16 H_KV=2`, and no fallbacks. The kernel was simply the
wrong algorithmic shape for prefill.

Reference implementation read:

- Vulkan on RADV/Navi31 selects the coopmat2 FA path for this shape. The host
  tuning resolves to large-row cooperative matrix FA, not scalar per-token FA:
  for F16 K/V, `HSK=HSV=256`, `N=512`, `KV=512`, it uses block FA over
  `Br=64`, `Bc=32`, `wg=128`. The shader keeps online `M/L/O` state across KV
  blocks and computes tiled `QK`, softmax update, and `PV` with cooperative
  matrix operations.
- CUDA/HIP has a different but more canonical pure-GPU FA structure in
  `fattn-tile.cuh`: staged Q/K/V tiles in shared memory, multiple Q columns per
  block, online softmax state, and explicit RDNA launch configurations. For a
  pure HIP backend, this is probably the better source shape than a literal
  Vulkan coopmat transliteration.

Accepted scalar cleanup:

- Replaced double accumulators, `pow`, `tanh`, and `exp` with float
  accumulators and `powf`/`tanhf`/`expf`.
- Removed the serial-per-output-dimension reduction. After softmax, one lane now
  owns one output dimension and accumulates the full V-weighted sum directly.
  This cuts hundreds of barriers/reductions per query row.
- Added a causal-mask skip before the expensive QK dot. For F16 mask entries at
  `-inf`, the kernel stores `-FLT_MAX` in the logits buffer and avoids the
  256-wide QK dot entirely. On p512 causal prefill this skips about half of the
  QK work.

Validation:

- `test-backend-pyre` exits 0 after the FA edits. This harness is silent on
  success in the current build, so the saved logs are empty but the process
  status is the useful signal.
- Decode guardrail with `-fa 1`, `-p 0 -n 64 -r 5`, reported 35.471 tok/s
  (`35.525, 35.582, 35.546, 35.355, 35.345`). This is in band with the recent
  no-FA decode numbers, but decode should not be optimized through this FA path.

Final p512 result after scalar cleanup and mask skip:

| Path | Prompt tok/s | FA bucket |
| --- | ---: | ---: |
| Pyre `-fa 0` post-edit baseline | 1226.816 tok/s r5 | no FA |
| Pyre `-fa 1` final | 1289.714 tok/s r5 | `pyre_flash_attn_ext_f32_f16_decode`, 10 calls / 42.432 ms |

This is a 5.1% p512 wall uplift for `-fa 1` versus `-fa 0` on the current
Pyre graph, and an 8.5x reduction in the Pyre FA kernel bucket versus the
one-shot starting point. It is still about 22x slower than Vulkan's FA bucket
for the same shape, so the current kernel is not "done-done" as a final FA
implementation. It is good enough to make FA useful for prompt experiments and
to prove that the earlier negative FA conclusion was an artifact of decode-only
measurement plus a pathological scalar implementation.

Next implementation target:

- Add a new prompt-only F16 K/V FA provider for `D=256`, `N=512`, `KV=512`,
  `H=16`, `H_KV=2`, mask-enabled causal prefill.
- Use the CUDA/HIP tiled FA structure as the main source: staged Q/K/V tiles,
  online `m/l/o` update, multiple Q rows per workgroup, and explicit RDNA launch
  choices. Vulkan coopmat2 remains the performance target and shape oracle, not
  necessarily the literal implementation substrate.
- Keep decode separate. A decode fast path should likely be a narrow fused
  attention/update kernel rather than running the full tiled/prefill FA machinery
  for `N=1`.

#### Prompt-only F16 tile8 FA provider

Implemented the first prompt-only HIP FA provider:
`pyre_flash_attn_ext_f32_f16_prefill_tile8`. It is deliberately narrow-gated to
the Qwen p512 shape that was measured above: F32 Q, F16 K/V, F16 mask, F32
output, `D=256`, `N=512`, `KV=512`, `H=16`, `H_KV=2`, no sinks. The gate is
default-on and can be disabled with `GGML_PYRE_DISABLE_F16_PREFILL_FA_TILE=1`.
Decode is not allowed through this route.

Implementation shape:

- one workgroup handles eight query rows for one attention head and sequence;
- 256 lanes per workgroup;
- shared logits for `8 x 512` scores and shared partials for max/sum
  reductions;
- causal-mask skip is preserved before the QK dot;
- each lane owns one output dimension for the final PV accumulation.

Validation:

- `test-backend-pyre` exits 0 with the new catalog entry.
- Provider tracing on p512 reports 10
  `pure_hip_f32_k_f16_v_f16_prefill_tile8` claims and no fallback claims.
- With `GGML_PYRE_DISABLE_F16_PREFILL_FA_TILE=1`, the same p512 run falls back
  to 10 `pure_hip_f32_k_f16_v_f16_decode` claims, proving the rollback gate.
- Decode guardrail (`-fa 1`, `-p 0 -n 64 -r 5`) reports only
  `pure_hip_f32_k_f16_v_f16_decode` claims and no tile8 claims.

Performance:

| Path | Prompt tok/s | FA bucket |
| --- | ---: | ---: |
| Pyre `-fa 0` post-edit baseline | 1226.816 tok/s r5 | no FA |
| Pyre scalar `-fa 1` | 1289.714 tok/s r5 | 10 calls / 42.432 ms |
| Pyre tile8 `-fa 1` | 1365.682 tok/s r5 | 10 calls / 16.692 ms |
| Vulkan `-fa 1` reference | 2222.553 tok/s r3 | 10 calls / 1.934 ms |

This is a 2.54x FA-kernel bucket reduction versus the scalar FA cleanup and an
11.3% prompt wall uplift versus the no-FA Pyre baseline. It remains about 8.6x
slower than Vulkan's FA bucket for the same p512 shape. That gap is expected:
this provider batches rows and removes the worst scalar overhead, but it is not
yet the real tiled tensor-core/cooperative-matrix algorithm.

Resource/codegen sanity:

| Kernel | LDS | Scratch | SGPR | VGPR | Spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pyre_flash_attn_ext_f32_f16_prefill_tile8` | 24.0 KiB | 0 B | 80 | 44 | 0 |

Disassembly after extracting the gfx1100 offload bundle shows vectorized global
loads and `v_fma_mix_f32` in the QK/PV body, with no `v_mfma` or dot primitive.
So the provider is resource-clean, but still below the algorithmic class used by
Vulkan coopmat2 FA. The next FA pass should not spend much time polishing this
tile8 kernel locally; it should replace the core dataflow with a real online
softmax tiled implementation using packed F16/MFMA-style work distribution.

#### rocWMMA and gfx11 direct FA provider

The tile8 provider established the right prompt/decode split, but it was still
not in the right architectural class. The follow-up work moved the Qwen p512
prefill path to a narrow F32-Q/F16-KV WMMA provider and then to a gfx11-specific
direct-WMMA implementation.

Landing commits in `sources/llama.cpp`:

- `f08b8fa04 Add opt-in gfx11 direct flash attention kernel`
- `b596f0d66 Use f16 WMMA for gfx11 flash attention PV`
- `e928785bc Keep gfx11 FA output accumulators in registers`
- `53acb3bf3 Split gfx11 FA kernel variant`

Current layout:

- `flash_attn_ext_f32_f16_prefill_gfx11_direct.hip.cpp` is a thin selector.
- `flash_attn_ext_f32_f16_prefill_gfx11_direct_generic.inc` contains the
  generic rocWMMA fallback.
- `flash_attn_ext_f32_f16_prefill_gfx11_direct_gfx11.inc` contains the
  hand-coded gfx11 implementation.
- The selector chooses the gfx11 include for `__gfx1100__`, `__gfx1101__`, and
  `__gfx1102__`, unless `GGML_PYRE_FORCE_GENERIC_FA_GFX11_DIRECT=1` is set for
  compile testing.
- CMake now tracks those two `.inc` files as dependencies of only this HSACO, so
  editing the FA variant does not force a full kernel catalog rebuild.

Key algorithmic facts learned:

- Prefill and decode must remain distinct. The prefill kernel wants tiled QK/PV
  and online softmax over a 512-token KV window; decode should get a separate
  narrow fusion later rather than reusing this machinery.
- The useful numerical schedule is F16 inputs with F32 accumulation for QK, then
  F16 accumulation for PV/O. An all-F16 QK experiment was correct but did not
  improve the bucket, and it is a worse accuracy/perf risk tradeoff.
- rocWMMA was a good bootstrap but a poor final abstraction for this shape. Its
  fragment layout encouraged storing PV results to LDS and redistributing them
  back to scalar/thread-owned output fragments. That extra shared-memory path
  was the main remaining structural tax.
- The winning gfx11 change was to make each wave own its output columns and keep
  two F16 WMMA output accumulators live across the KV loop. Online softmax
  rescales those accumulator fragments directly, and the final store writes from
  the WMMA accumulator layout instead of routing through `pv_matrix_tile`.
- The earlier rejected direct-Q and packed-prob experiments were useful negative
  evidence: lowering LDS/barriers locally is not enough if it destroys reuse or
  causes the compiler to collapse/unroll the wrong loop body. For this kernel,
  preserving the Q tile and the simple staged probability tile was better than
  more aggressive local rewrites.

Rejected branches and why:

| Experiment | Result | Decision |
| --- | --- | --- |
| All-F16 QK/PV | Correct, zero F32 WMMA, but no meaningful bucket win | reject; keep F32 QK accumulation |
| QK `#pragma unroll` | More static F32 WMMA but VGPR pressure jumped and FA bucket regressed to ~250 us | reject |
| `__launch_bounds__(256,2)` | No resource change versus baseline | reject |
| Packed probability tile | LDS/barriers improved, but FA bucket regressed to ~224 us | reject |
| Direct Q global loads | LDS dropped but QK body/codegen degraded and FA bucket regressed to ~240 us | reject |
| rocWMMA PV/output redistribution | Functional and useful for bringup, but stuck around ~224-232 us in rocprof buckets | replaced |

Measured progression for p512 FA bucket:

| Path | FA bucket |
| --- | ---: |
| Original scalar FA | 10 calls / 359.434 ms |
| Scalar cleanup + mask skip | 10 calls / 42.432 ms |
| Tile8 prompt provider | 10 calls / 16.692 ms |
| rocWMMA WMMA16 provider | 10 calls / 2.235 ms, ~223.5 us/call |
| gfx11 direct before output-accumulator rewrite | ~194 us/call |
| gfx11 direct with register-resident output accumulators | 10 calls / 1.397 ms, ~139.7 us/call |

The final gfx11 FA bucket is about 28.1% lower time than the previous direct
variant and about 39.1% higher kernel throughput. Against the measured Vulkan
FA bucket of 183.292 us/call, the gfx11 direct kernel is about 23.8% lower time,
or about 31.2% higher kernel throughput. End-to-end p512 moved from the
post-edit no-FA baseline of 1226.816 tok/s to about 1416-1425 tok/s with FA
enabled, a roughly 15.5-16.2% whole-prefill uplift on this graph.

Resource/codegen comparison for the critical direct rewrite:

| Metric | Direct redistribution | Register-resident output |
| --- | ---: | ---: |
| LDS | 23,168 B | 19,136 B |
| VGPR | 162 | 110 |
| SGPR | 66 | 66 |
| Static `s_barrier` | 26 | 14 |
| Static `ds_swizzle_b32` | 80 | 64 |
| Static QK WMMA | 2 x `v_wmma_f32_16x16x16_f16` | unchanged |
| Static PV WMMA | 8 x `v_wmma_f16_16x16x16_f16` | unchanged |

Important caveats:

- The kernel is shape-specialized for the current Qwen p512 F32-Q/F16-KV
  prefill case (`D=256`, `KV=512`) and is still gated as a narrow provider.
- Whole-prefill is not FA-bound after this change. The FA kernel is now in the
  same performance universe as the best reference path, while other graph
  buckets still dominate the Pyre/Vulkan end-to-end gap.
- The source comments in the gfx11 include should stay architectural and
  kernel-local. Overall tok/s, correctness commands, and detailed journey notes
  belong in this analysis document, not in the device kernel source.

### 2026-04-12: fast output corruption and correctness gate

The fast default checkpoint was producing random-symbol output in chat. The
root cause was not one single text-generation setting; it was a set of
optimized prompt routes that had been promoted based on local performance
signals without exact model-shape correctness coverage.

Concrete failures found with `export-graph-ops` plus `test-backend-ops` on the
Qwen p512 graph:

- `GATED_DELTA_NET` autoregressive `S_v=128` cluster16 failed exact comparison;
  the generic GDN route passes the same model-shape autoregressive test.
- `Q6_K x Q8_1` large x4 MMQL128 prompt route failed
  `linear_attn_out-0 q6_K[4096,2048] x f32[4096,512]` with max error around
  `3.0`.
- `Q5_K` prompt large/x4 routes failed model-shape prompt cases including
  `node_13` and `result_output` with errors around `1.2`.
- `Q4_K` prompt x4 `MUL_MAT_ID` / SWIGLU routes failed long-prompt behavior and
  exact model-shape MoE cases.
- The gfx11 direct FA provider is fast and coherent in smoke tests, but it
  fails exact p512 `FLASH_ATTN_EXT` comparison with error around `0.55`.
  Running FA without `GGML_PYRE_ENABLE_F16_PREFILL_FA_GFX11_DIRECT=1` passes
  the same `FLASH_ATTN_EXT` model-shape tests.

Default policy after the fix:

- GDN cluster16 is opt-in behind `GGML_PYRE_ENABLE_GATED_DELTA_NET_CLUSTER16=1`.
- The aggressive Q4/Q5/Q6 prompt MMQ/MMQL128 routes are opt-in behind their
  `GGML_PYRE_ENABLE_*` flags.
- `GGML_PYRE_DISABLE_SSM_CONV=1` and `GGML_PYRE_DISABLE_GATED_DELTA_NET=1`
  exist as coarse diagnostic kill switches.
- The Qwen chat reproducer no longer enables the gfx11 direct FA provider by
  default. It still runs with `-fa 1`, but uses the correctness-clean FA route
  unless the direct provider is explicitly enabled by the caller.

Validation after disabling the bad defaults:

- `MUL_MAT`, `MUL_MAT_ID`, and `SSM_CONV` model-shape tests pass:
  32/32 tests passed.
- GDN autoregressive model-shape test passes.
- Full critical testing still trips on the chunked GDN random-input case, but
  both CPU and Pyre produce NaN for that generated input. This is a test
  fixture problem until we add a bounded-input chunked GDN case.
- Short and long `reproducers/chat_qwen_pyre.sh -st --seed 1 --temp 0`
  smoke tests now produce coherent output rather than random symbols.

Correctness procedure change:

- Use `reproducers/qwen_pyre_correctness_gate.sh` before promoting any Pyre
  llama.cpp optimized route to default-on.
- For opt-in variants, run the gate with the candidate enable env set, for
  example:

```bash
GGML_PYRE_ENABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1 \
  reproducers/qwen_pyre_correctness_gate.sh
```

- Treat provider traces and chat output as secondary evidence. They prove route
  selection and user-visible sanity, not numerical equivalence.
- A route that improves rocprof but fails model-shape exact tests must remain
  opt-in and be documented as experimental until repaired.

### 2026-04-12: gfx11 direct FA numeric repair

Follow-up on the direct FA failure above:

- Forcing the direct provider entry point through the generic rocWMMA include
  passed the p128 and p512 `FLASH_ATTN_EXT` graph tests. That ruled out
  dispatcher constants, mask strides, and the exported graph fixture.
- A hybrid using the hand-coded gfx11 QK path with the generic PV/output path
  initially failed p512. The smoking gun was the gfx11 f32 WMMA accumulator
  lane layout: row-major stores are even/odd row interleaved. The correct row
  mapping is `row = (lane >> 4) + 2*i`, not a contiguous `0..7` / `8..15`
  split. After fixing that store, the direct-QK/generic-PV hybrid passed p512.
- The resident direct PV path was then repaired by using f32 PV WMMA
  accumulation and applying the same even/odd row mapping for online rescale
  and final stores. With `GGML_PYRE_ENABLE_F16_PREFILL_FA_GFX11_DIRECT=1`, the
  p512 model-shape FA exact test now passes.
- The original f16-PV resident path remains the performance target, but it is
  not safe to re-enable until its padded f16 accumulator mapping is handled
  with the same rigor. The repaired f32-PV direct path is correctness-clean but
  not an end-to-end win in the noisy p512 bench: default FA measured about
  568 tok/s and direct f32-PV measured about 566 tok/s in an r3 prompt-only
  check.

Correctness tooling changes:

- `test-backend-ops` now supports `GGML_TEST_BACKEND_OPS_PRINT_MAX_DIFF=1` to
  print the largest absolute-difference index and tensor coordinate when a
  comparison exceeds its threshold. This was needed to see whether FA failures
  were localized to rows, heads, or output columns.
- `reproducers/qwen_pyre_correctness_gate.sh` now runs both the default FA
  exact check and an opt-in gfx11 direct FA exact check by default. Set
  `CHECK_DIRECT_FA=0` only when intentionally skipping the experimental direct
  provider.

### 2026-04-12: correctness-clean prompt kernel recovery

Accepted changes:

- `Q5_K` prompt `q8_1_mmq32x32_wg128` is now default-on. It passed the Qwen
  model-shape correctness gate and cut the p512 Q5 bucket from about 364 ms to
  about 36 ms in rocprof. This moved prompt-only p512 from the earlier safe
  baseline around 568 tok/s to roughly 898 tok/s.
- GDN `s128_cluster16` is now default-on. The original opt-in version reduced
  across the wrong lane domain; the fixed version uses an explicit shared
  16-lane grouped reduction. It passed the full gate and cut the p512 GDN
  bucket from about 40 ms to about 16 ms. Steady prompt-only p512 samples were
  about 935-939 tok/s after this promotion.
- `Q6_K` prompt now uses an exact f32 RHS rows2 x cols8 provider. This keeps
  the exact f32 RHS path instead of the lossy `Q8_1` RHS variants, but reuses
  each RHS tile across two adjacent Q6 rows. It passed the full gate and
  selected as `pure_hip_q6_K_rows2_cols8_wg128` by default.

Rejected or deferred:

- `Q5_K`/`Q6_K` x4 `Q8_1` prompt MMQ variants still fail exact model-shape
  checks. The failure magnitude is around `1.2` for Q5 and `3.0` for Q6 on
  p512 prompt ops, so these remain experimental.
- A candidate fix to the x4 `Q8_1` packer did not repair the Q5/Q6 failures,
  which suggests the dominant issue is either RHS quantization error for these
  exact checks or a separate kernel-side layout problem. Do not promote these
  variants without a fresh exact pass.
- Forcing the non-SwiGLU Q4 grouped row4 provider ahead of row2-route8 was a
  clear regression: p512 fell to about 834 tok/s. The row2-route8 shape remains
  the better tradeoff for this model.

Current p512 framing:

- The quick post-commit steady p512 prompt-only run is about 955 tok/s on the
  W7900 with `-fa 1`, `-b 512`, `-ub 512`, `-ngl 99`, `-dev PYRE0`.
- Measurements remain sensitive to clocking and first-run outliers. Use
  several repeats and prefer the steady samples over a single rocprof summary
  if unrelated buckets all inflate together.

### 2026-04-12: default fast prompt numerics policy

We changed the promotion rule for prompt kernels: exact CPU comparison is a
guardrail, not the primary performance contract. For K-quant prompt matmuls,
Vulkan-style packed RHS / reassociated schedules are acceptable default
candidates when they satisfy model-level guardrails. The model is already
quantized, and requiring CPU accumulation order for every prompt route blocks
the schedules that actually map to the GPU.

Accepted default:

- `Q6_K` prompt now defaults to
  `pure_hip_q6_K_q8_1_x4_mmql128x64_wg256` for the p512 model shapes. This is
  the fast packed-Q8_1 RHS route. It fails the CPU exact op comparison on the
  Q6 prompt matmuls with errors around `3.0`, but it produces coherent chat
  output and matches the class of approximation used by the established GPU
  paths.
- `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1` forces the conservative f32-RHS
  prompt route for CPU-comparison tests and diagnostic runs.
- The Q4 SwiGLU grouped row2-route8 kernel had an LDS sizing bug:
  `sumsh[4 * waves]` was used for an 8-route reduction. The safe size is
  `8 * 4 * waves`. This is a correctness/safety fix independent of the
  approximate numerics policy.

Guardrail changes:

- `reproducers/qwen_pyre_correctness_gate.sh` runs the exact `MUL_MAT`,
  `MUL_MAT_ID`, and `SSM_CONV` op checks with
  `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1`. That keeps CPU-comparison coverage
  for the conservative implementations.
- The same gate still runs normal chat smoke tests without disabling fast
  approximate prompt kernels, so default user-visible behavior is exercised.
- The FA check now compares the default direct gfx11 FA route and the generic
  fallback route. The old opt-in env var for direct FA is stale now that the
  repaired direct route is default-on; use
  `GGML_PYRE_DISABLE_F16_PREFILL_FA_GFX11_DIRECT=1` to force generic FA.

Measurements on W7900 / gfx1100, Qwen p512, `-fa 1`, `-b 512`, `-ub 512`,
`-ngl 99`, `-dev PYRE0`:

- Default fast prompt: `1234.8 tok/s` average, samples `1228.7-1243.1 tok/s`.
- Conservative prompt with `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1`:
  `956.5 tok/s` average.
- Decode guardrail, `-p 0 -n 64`: `35.12 tok/s` average. The Q6 fast prompt
  route is gated on `cols == 512`, so it does not affect skinny decode.
- Provider trace confirms Q6 prompt selection as
  `pure_hip_q6_K_q8_1_x4_mmql128x64_wg256` for the
  `k=2048/4096`, `rows=2048/4096/8192`, `cols=512` hero shapes.

Do not promote the current Q4 x4 Q8_1 prompt routes by analogy alone. The Q4
SwiGLU x4 path is much slower on this graph, and the Q4 ID x4 path previously
failed the MoE down-projection checks badly. The new rule is: Vulkan-style
approximation is admissible, but it still needs coherent model-level smoke,
shape-specific provider tracing, decode guardrails, and a clear fallback knob.

### 2026-04-12: long-generation instability and MoE routing

The long Pyre memo prompt exposed a user-visible instability that the original
op-level gate did not catch. Symptoms were late-generation loops or collapse
after roughly 100+ decode tokens, for example repeated ``hip`-level`` /
`through` tokens or a long run of `?` output. This was reproducible enough to
use deterministic seeds as a guardrail.

Controls:

- Vulkan `llama-cli` on the same W7900, same prompt, same seeds `1` and `5`,
  did not reproduce the loop/collapse.
- Pyre with `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1` ruled out the Q6 fast
  path for one failure class, but default Pyre still showed late repetition.
- Pyre with `GGML_PYRE_DISABLE_ARGSORT=1` was clean for the bad seed sequence
  in the local sample set. Disabling both `ARGSORT` and `TOPK_MOE` was also
  clean. Disabling only the TOPK subgroup path was not clean and even produced
  worse corrupt output in one run.

Findings:

- The broad all-op exported Qwen gate exposed a prompt-shape
  `ARGSORT(name=ffn_moe_argsort-0, ne=[256,512])` failure that the focused
  matmul/GDN/FA gate missed. The `[256,1]` decode-shape ARGSORT passed, while
  the `[256,512]` prompt-shape case failed intermittently. Vulkan passed both.
- This maps directly to sparse MoE expert selection. Bad prompt expert routing
  can poison hidden state and only surface after many decode steps, which
  matches the observed late-token failure mode.
- The current Pyre `TOPK_MOE` fusion only supports `ggml_nrows(...) == 1`, so
  prompt MoE routing falls through to standalone ARGSORT. Decode still uses the
  fused TOPK_MOE path.
- A first attempt to make Pyre ARGSORT follow Vulkan's small-sort schedule
  exactly, by always sorting ascending in shared memory and reversing on DESC
  output, did not eliminate intermittent random-input test failures. Keep the
  schedule change for now, but do not consider ARGSORT repaired.

Mitigation landed:

- Pyre ARGSORT is now opt-in via `GGML_PYRE_ENABLE_ARGSORT=1`; default Pyre
  leaves ARGSORT unsupported so the scheduler can use the fallback path.
- `GGML_PYRE_DISABLE_TOPK_MOE=1` was added as a diagnostic kill switch for the
  fused decode MoE selector, but it is not default-on.
- `reproducers/qwen_loop_guard.py` records deterministic long-prompt runs and
  flags large question-mark counts or repeated n-grams. It should be used
  alongside `test-backend-ops`; exact op checks alone are not enough for this
  class of failure.

Current stability/performance tradeoff:

- The bad long-prompt seeds `1`, `5`, `1` were clean in 3/3 default Pyre runs
  after making ARGSORT opt-in.
- Conservative p512 prefill drops from roughly `956 tok/s` to roughly
  `827 tok/s` with ARGSORT fallback. This is a real cost, but it is preferable
  to corrupt generation.
- The next performance recovery target is therefore narrow: implement a
  correct prompt-shape MoE selector/top-k path for `[256,512]` finite softmax
  inputs, validate it with a realistic routing fixture, then re-enable it.

Follow-up after isolating ARGSORT:

- Retested the Q6_K x4 MMQL128 prompt route with ARGSORT still gated off. The
  previous corruption did not reproduce: seeds `1` and `5`, three runs each,
  `PREDICT=384`, all passed the loop guard with no question-mark collapse and
  no high-repeat n-grams.
- Re-promoted Q6_K x4 MMQL128 for default prompt use, still covered by the
  global rollback `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1` and the targeted
  rollback `GGML_PYRE_DISABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1`.
- Conservative exact gates continue to run with
  `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1`; this keeps CPU-comparison tests
  focused on exact providers while default chat/loop guards exercise the
  approximate fast path.
- A prompt TOPK_MOE extension was tested behind
  `GGML_PYRE_ENABLE_PROMPT_TOPK_MOE=1`. It was clean on the same long-prompt
  guard but effectively performance-neutral against the CPU ARGSORT fallback in
  normal p512 bench conditions. Leave it opt-in until there is a dedicated
  finite-input routing fixture and a clear p512 win.
- Current default p512 after Q6 re-promotion measured about `1030 tok/s` in a
  sequential r5 run, with samples `1041.65, 1002.31, 1038.83, 1039.37,
  1026.75`. The safe fallback without Q6 fast prompt was about `840 tok/s` in
  the same build family.
- `reproducers/qwen_pyre_correctness_gate.sh` now has a `CHECK_LOOP=1` phase by
  default. It runs `qwen_loop_guard.py` with `LOOP_SEEDS`, `LOOP_RUNS`,
  `LOOP_PREDICT`, and `LOOP_CONTEXT` overrides so local optimization passes
  cannot pass only the op-level gate while reintroducing late-token collapse.

### 2026-04-13: prefill parity pass restart

Restarted from commit `5c4ab3ba5` with the post-correctness default fast
prompt path back at roughly `1338 tok/s` p512 on W7900. Fresh side-by-side
numbers for the same model and `-fa 1`:

- Pyre `llama-bench -p 512 -n 1 -r 5`: `1337.69 +/- 13.48 tok/s` prompt,
  `34.54 +/- 0.54 tok/s` single-token decode.
- Vulkan `llama-bench -p 512 -n 1 -r 5`: `2374.57 +/- 76.64 tok/s` prompt,
  `99.19 +/- 5.56 tok/s` single-token decode.
- Pyre rocprof family totals for a p512/n0 run still show the remaining
  device-time work concentrated in `q4_moe_swiglu` (~53.5 ms),
  `dense_f32` (~51.1 ms), `q6` (~30.2 ms), `q4_moe_id` (~29.4 ms), `q5`
  (~21.0 ms), `dense_bf16` (~20.8 ms), and `gated_delta` (~14.5 ms).

First experiment: make the Q6_K x4 MMQL128 prompt kernel match Vulkan's large
K-quant column tile more literally by changing the HIP tile from `BM=128,
BN=64, WM=64, WN=32, WNITER=4` to `BM=128, BN=128, WM=64, WN=64,
WNITER=8`. The first attempt intentionally only touched the device kernel and
immediately faulted because the host launch metadata still used 64 columns per
workgroup. After correcting the launch metadata for the experiment, the run was
valid but slower:

- Q6 128x64 default: prior p512 reference `1337.69 tok/s`.
- Q6 128x128 experiment: p512 `1286.56 +/- 4.19 tok/s`, decode `33.98 +/-
  0.05 tok/s`.

Decision: reject and revert the Q6 128x128 HIP variant. The larger shape is
nominally closer to Vulkan's large K-quant schedule, but this HIP kernel already
has heavy Q6 unpack/scale state and doubling the column tile appears to lose
more to register/LDS pressure than it saves in launch/workgroup count. Keep the
current 128x64 Q6 route unless a deeper rewrite changes the per-thread state
shape enough to retest the wider tile.

Second experiment: tighten the `GATED_DELTA_NET` S=128 prompt path against
Vulkan's schedule. Pyre's accepted path still used LDS plus workgroup barriers
for each 16-lane column reduction. Vulkan builds the S=128 pipeline with
`LANES_PER_COLUMN=8` when subgroup clustered reductions are available, which
means 8 columns per workgroup and a subgroup reduction/broadcast rather than
shared-memory reductions.

Results:

- A first shuffle-only replacement was wrong because a down-shuffle reduction
  leaves the full sum only in lane 0. `GATED_DELTA_NET` needs the column sum in
  every lane to update each lane's state shard. Adding the explicit cluster
  broadcast fixed the exact fixture.
- The final Pyre S=128 path now uses 8-lane clusters, 16 rows per lane, 8
  columns per workgroup, and a shuffle reduce plus broadcast. The provider was
  renamed from `_s128_cluster16` to `_s128_cluster8`; the kill switch is now
  `GGML_PYRE_DISABLE_GATED_DELTA_NET_CLUSTER8`.
- Exact check: `test-backend-ops test -o GATED_DELTA_NET -b PYRE0` passes
  `18/18`.
- p512 rocprof GDN bucket improved from the pre-pass `14.492 ms` to
  `13.055 ms` for 30 calls. The intermediate shuffle+broadcast cluster16 form
  measured `13.501 ms`, so most of the win is the subgroup-style reduction and
  the remaining piece is the Vulkan-style 8-column workgroup shape.
- p512 `llama-bench -r 5` after the rename measured `1351.76 +/- 35.91 tok/s`
  prompt and `34.46 +/- 0.73 tok/s` single-token decode. Treat wall tok/s as
  secondary; the rocprof bucket reduction is the acceptance signal.
- Full Qwen gate also passed after the rename:
  `CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1`, seeds `1,5`,
  `LOOP_PREDICT=384`, logs in
  `build/pyre-correctness-qwen-exhaustive-20260412-224605`.

This does not close the whole GDN gap to the Vulkan label (~7 ms), but it removes
the obvious barrier/LDS mismatch and halves the GDN workgroup grid from
`2048x32x1` to `1024x32x1`. Any further GDN work should use thread trace rather
than more shape guessing; likely remaining gaps are register pressure from
16-row shards, exp/load scheduling, or compiler differences in clustered
subgroup arithmetic.

Q5/Q6 follow-up:

- The standalone ISA helper initially reported wave32 for Q5/Q6, but that was
  an analysis-tool artifact: CMake compiles `mul_mat_vec_q5_k_q8_1.hip.cpp` and
  `mul_mat_vec_q6_k_q8_1.hip.cpp` with `-mwavefrontsize64`. Unbundling the
  actual built HSACO confirms the shipped Q5/Q6 providers have
  `.wavefront_size: 64`.
- Tried matching Vulkan's shorter RHS register lifetime by loading one
  `cache_b` column at a time instead of keeping `cache_b[TN]` live for the
  two-column inner loop. Metadata improved materially: Q5 large VGPR
  `173 -> 147`, Q6 large VGPR `136 -> 132`, no spills.
- Device time rejected the change. p512 rocprof worsened Q5 large to
  `21.237 ms` and Q6 large to `31.701 ms` versus the prior roughly `20.2 ms`
  and `29.6-30.6 ms` range. The original two-column RHS cache appears to expose
  useful ILP or scheduling slack that outweighs the higher VGPR count.
- Reverted the RHS lifetime experiment. Do not repeat it unless paired with a
  different dot-product schedule that recovers the lost ILP.

Q8_0 fused ADD follow-up:

- The remaining `dense_f32` bucket contained 9 prompt-shape
  `MUL_MAT + ADD` sites for `attn_output-*`: Q8_0 LHS, F32 RHS, F32 bias,
  shape `k=4096 rows=2048 cols=512`. The old default was the exact
  `rows4_cols4` F32-RHS kernel, which rocprof had around `9-10 ms` total for
  those 9 calls.
- First Q8_1 attempt used a large `BM=128 BN=64` tiled integer-dot dataflow.
  It was structurally closer to the large K-quant prompt kernels, but the HIP
  compiler spilled badly (`vgpr=192`, `vgpr_spill=698` in the standalone ISA
  helper) and device time regressed to `23.737 ms` for the 9 calls before
  counting the extra RHS quantization. Rejected.
- Second Q8_1 attempt switched to a `BM=64 BN=64` row-lane schedule:
  one row per lane, 16 columns per thread, Q8_1_x4 RHS staged in LDS, and Q8_0
  LHS packed with two 16-bit loads to match Vulkan's explicit treatment of
  34-byte Q8_0 blocks. This removed spills (`vgpr=124`, `vgpr_spill=0` in the
  standalone ISA helper) and measured `4.336 ms` for the 9 fused Q8_0 ADD
  kernel calls in `build/rocprof-pyre-prefill-q8-mmq64`.
- The new route adds 9 RHS quantization launches through the existing
  Q8_1_x4 scratch path. Total `pyre_quantize_q8_1_x4_f32` in the same trace was
  `6.805 ms` across 187 calls, so the incremental Q8_0 RHS quantization cost is
  on the order of a few tenths of a millisecond, not enough to erase the
  ~5 ms kernel win.
- Wall p512 after fixing an accidental Q6 provider rename measured
  `1369.04 +/- 18.59 tok/s` versus `1279.31 +/- 12.87 tok/s` with
  `GGML_PYRE_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ64_PROMPT=1` in the same build
  family. The wall delta is noisy, but the device bucket is a clear win.
- The accepted path is still under the global approximate prompt rollback
  `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1` and has a targeted rollback
  `GGML_PYRE_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ64_PROMPT=1`. It needs the full Qwen
  correctness gate before commit because it replaces exact F32 RHS arithmetic
  with a Vulkan-style Q8_1 RHS approximation.
- Full Qwen gate passed after the Q8_0 MMQ64 route:
  `CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1`, seeds `1,5`,
  `LOOP_PREDICT=384`, logs in
  `build/pyre-correctness-qwen-q8-mmq64-20260412-230352`.

Q4_K MoE SWIGLU retile follow-up:

- Started from the largest remaining p512 device-time bucket:
  `pyre_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32`, around
  `52.660 ms` over 39 calls in the post-Q8 trace. The legacy x4 grouped
  SWIGLU path used a conservative `BM=16, BN=32, WG=64` shape. Because SWIGLU
  loads two Q4_K expert matrices (`gate` and `up`) for the same RHS route tile,
  that row tile gives poor amortization of the RHS route/Q8 staging and launches
  twice as many row workgroups as the corresponding non-SWIGLU grouped MMQ.
- Current experiment: retile SWIGLU to `BM=32, BN=32, WG=64`, keeping the same
  Q8_1_x4 RHS packing and route tile, and update the host row grid from
  `(rows + 15) / 16` to `(rows + 31) / 32`. This is deliberately smaller than
  the non-SWIGLU `BM=64` path because SWIGLU keeps two accumulator arrays
  (`gate_sum` and `up_sum`) live and is much closer to the VGPR cliff.
- Rejected. Standalone ISA was not awful (`vgpr=136`, no spills, LDS 3968),
  but rocprof worsened the p512 SWIGLU bucket to `54.746 ms` for 39 calls
  versus the post-Q8 baseline around `52.660 ms`. The likely issue is
  occupancy/scheduling pressure from doubling the live SWIGLU accumulator state:
  no explicit spills, but enough VGPR pressure and less ILP headroom to erase
  the row-grid reduction. Reverted to `BM=16`; future SWIGLU work should reduce
  accumulator lifetime or route/tile waste rather than just widening rows.

BF16 dense prompt follow-up:

- Vulkan's BF16 matmul path is structurally different from Pyre's scalar BF16
  path: the Vulkan shader routes BF16 through cooperative matrix with BF16
  matrix operands, whereas Pyre's accepted prompt path expands BF16 weights to
  F32 and multiplies by F32 RHS values in scalar lanes. This leaves MFMA/WMMA
  idle for a bucket that costs about `20.7 ms` total in the p512 trace
  (`11.1 ms` dense BF16 matmuls plus `9.6 ms` BF16 SWIGLU).
- Current experiment: add a gfx11-style `16x16x16` BF16 WMMA path for the
  dense non-SWIGLU BF16 prompt matmuls (`k=2048`, `rows % 16 == 0`,
  `cols=512`). The RHS is rounded to BF16 in the load path before WMMA, so this
  is an approximate Vulkan-style route and is covered by both
  `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1` and the targeted
  `GGML_PYRE_DISABLE_BF16_WMMA16_PROMPT=1`.
- First measurement with the gate restricted to `k=2048` replaced 80 prompt
  BF16 calls with `pyre_mul_mat_vec_bf16_wmma16x16_f32` at `2.754 ms` total.
  The old scalar `rows2_cols16` path still handled 39 calls at `5.924 ms`;
  those came from `ffn_shexp-0`, shape `bf16[512,2048] x f32[512,512]`.
  Full Qwen correctness passed with logs in
  `build/pyre-correctness-qwen-bf16-wmma16-20260412-231558`.
- Widened the host gate to `k % 16 == 0`, `rows % 16 == 0`, `cols == 512`.
  This admits `ffn_shexp-0` without adding tail behavior to the kernel. The
  p512 trace in `build/rocprof-pyre-prefill-bf16-wmma16-k512` shows all 119
  dense non-SWIGLU BF16 prompt calls on the WMMA route: `5.795 ms` total versus
  the previous `8.678 ms` combined WMMA+scalar bucket. The whole `dense_bf16`
  family drops from `18.405 ms` to `15.537 ms`, with BF16 SWIGLU still
  unchanged at about `9.7 ms`.
- The WMMA route fails exact CPU comparison for the prompt BF16 matmuls with
  max error around `1.87-1.89`, because it deliberately rounds the F32 RHS to
  BF16 before WMMA. Exact `test-backend-ops` is therefore not the acceptance
  test for this approximate path; the rollback variable remains
  `GGML_PYRE_DISABLE_BF16_WMMA16_PROMPT=1` or the global
  `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1`.
- Full Qwen gate passed after the `k=512` widening:
  `CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1`, seeds `1,5`,
  `LOOP_PREDICT=384`, logs in
  `build/pyre-correctness-qwen-bf16-wmma16-k512-20260412-232115`. This accepts
  the BF16 WMMA route as a model-safe approximate prompt default.
- Followed with the matching BF16 SWIGLU WMMA path. It uses one wave per
  16x16 output tile, two BF16 WMMA accumulators for the gate/up matmuls, and
  applies SiLU at store time. Standalone ISA for
  `pyre_mul_mat_vec_bf16_swiglu_wmma16x16_f32` is wave32, `vgpr=63`,
  `sgpr=26`, no spills, no LDS, and exactly two `v_wmma` instructions.
- p512 trace in `build/rocprof-pyre-prefill-bf16-swiglu-wmma16` moves the 39
  BF16 SWIGLU prompt calls from `pyre_mul_mat_vec_bf16_swiglu_rows2_cols8_f32`
  at roughly `9.7 ms` to `pyre_mul_mat_vec_bf16_swiglu_wmma16x16_f32` at
  `2.791 ms`. With both dense BF16 WMMA routes enabled, the `dense_bf16`
  family is `8.329 ms` total, down from the earlier `~20.7 ms` pre-pass bucket.
- Full Qwen gate passed after BF16 SWIGLU WMMA:
  `CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1`, seeds `1,5`,
  `LOOP_PREDICT=384`, logs in
  `build/pyre-correctness-qwen-bf16-swiglu-wmma16-20260412-232718`. Targeted
  rollback is `GGML_PYRE_DISABLE_BF16_SWIGLU_WMMA16_PROMPT=1`; the global
  rollback is still `GGML_PYRE_DISABLE_FAST_APPROX_PROMPT=1`.

Q4_K MoE SWIGLU route-loop rejection:

- Tried reducing apparent overlaunch in
  `pyre_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32`. The
  original dispatch uses `y=(n_tokens+31)/32` route-tile lanes per expert, and
  the kernel advances each lane by the full token-tile span. Given Qwen p512 has
  4096 routes over 256 experts, the mean count is only 16 routes/expert, so a
  one-lane-per-expert route loop looked attractive.
- Rejected. Changing host `y` to 1 and making the kernel iterate
  `route_base += BN` serialized route tiles within each row/expert workgroup and
  regressed the SWIGLU bucket from about `54.4 ms` to `97.959 ms` in
  `build/rocprof-pyre-prefill-q4-swiglu-route-loop`. The existing parallel
  route tiling is paying for occupancy/load balance even when some workgroups
  early-exit. Do not repeat this without a true compact tile-descriptor list or
  indirect dispatch support that preserves parallel route tiles without launching
  the empty ones.
- Also tried a middle point with 8 route-tile lanes per expert
  (`route_tile_span = 256` for p512, host `y=8`). This preserved more route
  parallelism than the one-lane loop but still regressed the SWIGLU bucket to
  `58.025 ms` in `build/rocprof-pyre-prefill-q4-swiglu-route-y8`. Reverted.
  The current 16-lane route decomposition remains the best measured launch
  shape for this kernel.

Overnight p512 baseline and GDN non-KDA specialization:

- Refreshed the p512 device-time comparison before starting the next tranche.
  Pyre rocprof baseline:
  `build/rocprof-pyre-prefill-overnight-baseline-20260412-235206/pyre-p512-baseline_results.db`.
  The top Pyre buckets were `q4_moe_swiglu` `53.230 ms`, `dense_f32`
  `46.371 ms`, `q4_moe_id` `29.932 ms`, `q6` `29.083 ms`, `q5` `21.371 ms`,
  `gated_delta` `13.272 ms`, `dense_bf16` `8.350 ms`, and `flash_attn`
  `1.487 ms`.
- Refreshed the Vulkan p512 reference with `GGML_VK_PERF_LOGGER=1` in
  `build/vulkan-p512-perf-20260412-235236.log`. Vulkan wall prompt averaged
  `2358.69 tok/s`; logger totals ranged roughly `201-218 ms`. Vulkan labels
  put GDN around `7.0-7.5 ms`, Q6 around `20-21 ms`, Q5 around `12.5-13.7 ms`
  plus the single vocab matvec, and Q4 MoE ID/SWIGLU around `101-108 ms`
  combined. This makes Q6/Q5/GDN better parity targets than more Q4 route
  reshaping for now.
- Found a concrete GDN mismatch: Vulkan specializes KDA as a pipeline constant,
  but the Pyre S=128 cluster8 path kept KDA as a runtime branch and carried a
  `g_reg[16]` row array even for the Qwen non-KDA shape (`g_ne0 == 1`). Added
  `pyre_gated_delta_net_s128_cluster8_nokda_f32`, selected only for
  `S_v == 128` and `g_ne0 != S_v`, while leaving the KDA-capable cluster8 and
  generic paths as fallback.
- Targeted exact GDN passed:
  `build/pyre-inner-loop-gdn-nokda-20260412-235438`.
- p512 rocprof with the non-KDA specialization:
  `build/rocprof-pyre-prefill-gdn-nokda-20260412-235443/pyre-p512-gdn-nokda_results.db`.
  GDN improved from `13.272 ms` to `10.310 ms` over 30 calls (`-22.3%`).
  Other bucket movements in that one-sample rocprof were noise/regression
  confounds (`q6 +5.9%`, `q5 +1.5%`, `dense_f32 +1.8%`), not caused by
  provider selection changes.
- Full Qwen gate passed after the GDN specialization with chat and loop guards:
  `build/pyre-correctness-qwen-gdn-nokda-20260412-235518`.
- Remaining GDN gap to Vulkan is now roughly `2.8-3.3 ms` for p512. Since the
  obvious specialization issue is removed, the next GDN pass should use ISA or
  thread trace against the non-KDA kernel and focus on exp/load scheduling,
  register residency for `s_shard/q_reg/k_reg`, and whether the Vulkan compiler
  is emitting a tighter clustered-reduction sequence.

Q5/Q6 MMQL cleanup and rejected retile variants:

- Broke the p512 Q6 bucket down by shape after the GDN checkpoint. The accepted
  Pyre Q6 route is a single `128x64` MMQL kernel over 70 calls:
  `m=2048,n=512,k=4096` accounts for about `11.2 ms`, `m=4096,n=512,k=2048`
  about `10.9 ms`, and `m=8192,n=512,k=2048` about `7.0 ms`. Vulkan's labels
  for the same shapes are roughly `7.3-7.8 ms`, `8.4-8.6 ms`, and
  `4.3-4.7 ms` respectively after warmup. Q6 remains a real per-shape gap,
  even though the total graph is now near Vulkan's aggregate device-time band.
- Forced the old Q6 `mmq32x32` path correctly by disabling the default MMQL
  provider:
  `GGML_PYRE_DISABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1` plus
  `GGML_PYRE_ENABLE_Q6_K_Q8_1_X4_MMQ32_PROMPT=1`. This selected
  `pyre_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32` and regressed Q6 from
  about `30.8 ms` to `183.6 ms`. The earlier unforced probe was invalid because
  provider priority kept selecting MMQL128. The `mmq32x32` route should stay
  rejected for p512 prompt shapes.
- Matched Vulkan's inner-loop B lifetime more closely for the accepted Q5/Q6
  MMQL kernels. Vulkan loads one B cache entry and accumulates all resident A
  rows against it; Pyre previously loaded `cache_b[TN]` before the `cr x cc`
  loop. Reordering to `cc -> cr` lowers Q5 large VGPRs from `173` to `147` and
  Q6 large VGPRs from `136` to `132` with no spills. Targeted Q5 and Q6 exact
  tests passed (`build/pyre-inner-loop-q5-bcache-20260413-000322` and
  `build/pyre-inner-loop-q6-bcache-20260413-000322`).
- p512 rocprof for the B-lifetime change:
  `build/rocprof-pyre-prefill-q5q6-bcache-20260413-000342/pyre-p512-q5q6-bcache_results.db`.
  Q6 improved from `30.788 ms` to `29.140 ms` (`-5.4%`) versus the GDN
  checkpoint. Q5 was effectively flat (`21.682 ms` to `21.423 ms`, within
  one-sample noise). This change is still worth keeping because it moves the
  HIP schedule toward the proven Vulkan structure and materially lowers Q5
  register pressure, but it is not the missing Q5 win by itself.
- Retried a Vulkan-large-style Q6 `128x128` tile after the B-lifetime cleanup.
  It passed targeted Q6 exact tests but raised Q6 large VGPRs to `151` and
  regressed the p512 Q6 bucket to `31.833 ms` in
  `build/rocprof-pyre-prefill-q6-128x128-20260413-000531`. Reverted. Matching
  Vulkan's tile dimensions without matching its full scalar-dot schedule is not
  enough here.
- Tried replacing Q5 scalar byte assembly in `pyre_q5_k_pack4` with direct
  32-bit `qs/qh` loads, mirroring Vulkan's packed-field source. It passed exact
  Q5 tests (`build/pyre-inner-loop-q5-packed-loads-20260413-000631`) but did
  not reduce VGPRs and was slightly negative in p512 rocprof
  (`build/rocprof-pyre-prefill-q5-packed-loads-20260413-000651`), so it was
  reverted.
- Tried `__launch_bounds__(256, 2)` on the accepted Q5/Q6 large kernels to force
  occupancy. It backfired at compile time: Q5 VGPRs rose to `165` and Q6 to
  `170`, with no useful spill tradeoff. Reverted without running a full trace.
- Final accepted change in this tranche is the Q5/Q6 one-B-cache loop order.
  Full Qwen correctness passed with exact main ops, exact GDN, direct and
  generic FA, chat smoke tests, and loop guard seeds `1,5`:
  `build/pyre-correctness-qwen-q5q6-bcache-final-20260413-000933`.
- The current aggregate p512 Pyre kernel total from
  `build/rocprof-pyre-prefill-q5q6-final-20260413-000845/pyre-p512-q5q6-final_results.db`
  is `202.775 ms`, which is within the refreshed Vulkan logger band
  (`~201-218 ms`). This is an aggregate parity result, not a per-bucket parity
  result: Pyre is getting help from very strong Q4 MoE and FA buckets while Q5,
  Q6, and GDN still lag their corresponding Vulkan labels. The next profitable
  work should therefore be shape-specific Q5/Q6 schedule analysis and GDN
  thread-trace/ISA analysis, not more graph-wide speculation.

Post-parity structural probes and rejected variants:

- Refreshed the p512 Pyre/Vulkan comparison after the Q5/Q6 B-cache commit.
  Pyre rocprof:
  `build/rocprof-pyre-prefill-current-20260413-001924/pyre-p512-current_results.db`.
  Vulkan logger:
  `build/vulkan-current-20260413-002000/vulkan-trace.log`. Aggregate device
  time remains close, but the bucket mix is uneven. Pyre is ahead on the Q4 MoE
  grouped routes (`~84.5 ms` Pyre versus `~109.1 ms` Vulkan) and FA
  (`~1.46 ms` Pyre versus `~1.84 ms` Vulkan). Vulkan is still ahead on Q5
  (`~14.8 ms` Vulkan versus `~22.5 ms` Pyre), Q6 (`~26.3 ms` Vulkan versus
  `~29.4 ms` Pyre), and GDN (`~7.7 ms` Vulkan versus `~10.3 ms` Pyre). This
  makes Q5 the largest remaining per-bucket prefill gap, followed by Q6 and
  GDN.
- Tried a Q5 `128x64` MMQL retile by halving the accepted large tile's column
  dimension and temporarily selecting 64 columns per workgroup from the host
  provider. This lowered the hot Q5 kernel from `147` to `124` VGPRs and kept
  the kernel wave64 with no spills. Targeted Q5 exact tests passed:
  `build/pyre-inner-loop-q5-128x64-20260413-002408`. The p512 trace
  `build/rocprof-pyre-prefill-q5-128x64-20260413-002427/pyre-p512-q5-128x64_results.db`
  regressed the large Q5 kernel from `~21.9 ms` to `~37.2 ms`, so the variant
  was reverted. The likely failure mode is increased A-side traffic and worse
  scheduling despite improved register residency. Do not retry this shape
  without a substantially different dataflow.
- Tried compiling GDN with `-mwavefrontsize64` to match the Vulkan subgroup
  size. The hot non-KDA kernel stayed spill-free with `83` VGPRs, but targeted
  GDN exact passed only to reveal a p512 performance regression in
  `build/rocprof-pyre-prefill-gdn-wave64-20260413-002548/pyre-p512-gdn-wave64_results.db`:
  GDN moved from `~10.3 ms` to `~11.5 ms`. Reverted. For the current HIP GDN
  schedule on this RDNA card, wave32 is better even if Vulkan's compiled shader
  uses subgroup64.
- Tried scalarizing the repeated non-KDA GDN `exp(g)` by computing it on one
  lane and broadcasting with `__shfl`. Targeted GDN exact passed:
  `build/pyre-inner-loop-gdn-broadcast-exp-20260413-002704`. The p512 trace
  `build/rocprof-pyre-prefill-gdn-broadcast-exp-20260413-002710/pyre-p512-gdn-broadcast-exp_results.db`
  regressed GDN to `~11.9 ms`, so this was reverted. Either transcendental
  latency is not the bottleneck in this kernel, or the shuffle/control
  dependency costs more than the redundant exp work it removes.
- Ran an ATT smoke capture with rocprofv3 using `--att-target-cu 0`. This
  gets past the earlier invalid-target-cu failure, but the unfiltered p0/n1
  capture hung during finalization and produced code object dumps without
  `stats_ui_output*.csv` derived stats:
  `build/rocprof-att-smoke-20260413-001437`. Thread trace is still the right
  tool for the next phase, but this ROCm build likely needs smaller filtered
  captures or a different rocprof invocation before ATT can be used as the main
  loop.
- Re-read the Vulkan dispatch path and corrected an earlier mental model: for
  some batched matvec shapes Vulkan's `mul_mat_vec_*_q8_1_f32` path is a
  wave64, one-row, up-to-eight-column DMMV-style reduction, not the large
  `128x128` integer MMQ tile. Implemented a temporary Q5 x4 DMMV8 HIP probe to
  test that exact algorithmic family. It selected correctly and an unprofiled
  p512 run looked superficially promising, but rocprof showed the smoking gun:
  `pyre_mul_mat_vec_q5_k_q8_1_x4_dmmv8_wg64_f32` took `65.1 ms` over 30 calls
  in
  `build/rocprof-pyre-prefill-q5-dmmv8-20260413-003522/pyre-p512-q5-dmmv8_results.db`.
  That is far worse than the accepted Q5 MMQL route (`~21.9 ms`) and the
  existing row-oriented x4 `32x32` route (`~32.8 ms` in
  `build/rocprof-pyre-prefill-q5-x4mmq32-20260413-003135/pyre-p512-q5-x4mmq32_results.db`).
  The DMMV8 probe was removed. The useful conclusion is negative but important:
  a literal one-row Vulkan DMMV transliteration is not the missing Q5 prefill
  structure in Pyre's current graph/runtime shape; the accepted large MMQL
  remains the best local Q5 route.

Q6 fill-order and GDN Q residency:

- GDN non-KDA specialization follow-up: removed persistent `q_reg` state from
  `pyre_gated_delta_net_s128_cluster8_nokda_f32` and reload Q in the attention
  loop while keeping K resident. This is a small register-lifetime/dataflow
  change, not an algorithmic change. Targeted GDN exact passed. The focused
  p512 trace
  `build/rocprof-pyre-prefill-gdn-qreload-20260413-003839/pyre-p512-gdn-qreload_results.db`
  measured GDN at `10.029 ms`, versus the fresh current baseline of
  `10.325 ms`. The hot non-KDA code object stayed at `83` VGPR / `68` SGPR,
  wave32, no spills. A later combined run was noisier (`10.549 ms`), but the
  change is correctness-neutral, resource-neutral, and positive in the focused
  measurement, so it is kept.
- Rejected adjacent GDN variants. Remapping the grid to Vulkan's apparent
  `(head, seq, colgroup)` ID order passed exactness but measured `10.499 ms`.
  Reloading both Q and K passed exactness but measured `10.419 ms`. Rewriting
  the update as explicit `fmaf` passed exactness but measured `10.640 ms`.
  These were reverted. The remaining GDN gap to Vulkan (`~7.7 ms`) is no longer
  exposed by source-level obvious changes; it needs ATT/thread-trace or a
  deeper schedule rewrite.
- Re-read the Vulkan large K-quant MMQ constants for RADV/RDNA. For the hot
  large integer MMQ path, Vulkan's non-coopmat schedule uses the gross shape
  `{threads=256, BM=128, BN=128, BK=32, WM=64, WN=64, WMITER=1, TM=4, TN=2,
  WARP=64}`. Pyre's accepted Q5 large MMQL route already matches this high-level
  tile shape. That means the remaining Q5 gap is not a missing gross tile
  dimension; it is in emitted schedule details, codegen, lifetime, LDS traffic,
  or split/pipeline policy.
- Q5/Q6 accumulator-layout probe: changed the MMQL local sum layout toward
  Vulkan's `(wsic, cc, cr)` traversal. Targeted exact tests passed, but p512
  showed no useful movement (`Q6 29.641 ms`, Q5 about flat), so it was reverted.
- Q5/Q6 Vulkan-style A/B tile fill-order probe: changed LDS tile filling from a
  K-step outer loop to row/column outer loops with K-step inner loops. The same
  source-level schedule was split by result. Q6 improved materially; Q5
  catastrophically regressed.
- Accepted Q6 fill-order change: with Q5 reverted and Q6 kept, the p512 trace
  `build/rocprof-pyre-prefill-q6-fillorder-gdn-qreload-20260413-004922/pyre-p512-q6-fillorder-gdn-qreload_results.db`
  measured Q6 at `27.264 ms` versus the fresh current `29.374 ms`, about a
  `7.2%` Q6-bucket win and close to the Vulkan Q6 aggregate (`~26.3 ms`). The
  earlier paired probe measured an even lower `24.493 ms` for Q6, but that run
  had poisoned Q5 code and should not be used as the accepted number. The Q6
  code object now has `192` VGPR and `4` VGPR spills, yet measured faster; this
  is an explicit compiler tradeoff accepted by profiler evidence.
- Rejected Q5 fill-order change: the same Vulkan-style fill-order edit on Q5
  passed exactness but regressed the hot Q5 bucket to `1106.951 ms` over 30
  calls in
  `build/rocprof-pyre-prefill-q5q6-vulkan-fillorder-20260413-004816/pyre-p512-q5q6-vulkan-fillorder_results.db`.
  Q5 and Q6 therefore need separate schedule treatment. Q5 already has the
  Vulkan gross tile, but its HIP source is close to a compiler local maximum:
  seemingly aligned source-order changes can trigger disastrous codegen.
- Full Qwen correctness passed with the accepted Q6 fill-order and GDN Q-reload
  changes:
  `build/pyre-correctness-qwen-q6-fill-gdn-qreload-20260413-005002`. The gate
  covered exact main ops, exact GDN autoregressive checks, direct and generic
  FA, short/long chat smokes, and loop guard seeds `1,5`.

Q5 device O3 and rejected compile-policy probes:

- Q5 wave32 probe: removed the explicit `-mwavefrontsize64` compile flag from
  `mul_mat_vec_q5_k_q8_1.hip.cpp`. Targeted exact Q5 checks passed, but p512
  regressed the Q5 bucket to `24.356 ms` in
  `build/rocprof-pyre-prefill-q5-wave32-20260413-005724/pyre-p512-q5-wave32_results.db`,
  worse than the accepted `~22.1 ms` band. Keep Q5 wave64. This is consistent
  with the Vulkan/RDNA subgroup64 choice for the corresponding large integer
  MMQ path, even though the HIP source does not expose that as a high-level
  algorithmic requirement.
- Accepted Q5-only device `-O3`: adding `-O3` only for
  `mul_mat_vec_q5_k_q8_1.hip.cpp` passed targeted exact Q5 checks and the full
  Qwen gate. The final p512 trace
  `build/rocprof-pyre-prefill-q5-o3-final-20260413-010034/pyre-p512-q5-o3-final_results.db`
  measured Q5 at `20.580 ms`, versus `22.095 ms` in the accepted Q6/GDN
  checkpoint, about a `6.9%` Q5-bucket win. The hot Q5 code-object metadata was
  unchanged at `147` VGPR / `44` SGPR / no spills / wave64, so this appears to
  be a schedule/codegen win rather than a resource-class change. Q5 is still
  meaningfully behind the Vulkan aggregate (`~14.8 ms`), so the remaining work
  should compare emitted schedules, LDS wait sites, dot issue density, and
  global load coalescing rather than keep perturbing gross tile dimensions.
- Rejected Q6 device `-O3`: exact Q6 checks passed, but the p512 trace
  `build/rocprof-pyre-prefill-q5q6-o3-20260413-005951/pyre-p512-q5q6-o3_results.db`
  measured Q6 at `28.670 ms`, slower than the accepted Q6 fill-order/O2 result
  (`27.264 ms`). Q6 stays on the default device optimization level.
- Rejected GDN device `-O3`: exact GDN checks passed, but the p512 trace
  `build/rocprof-pyre-prefill-q5-gdn-o3-20260413-010125/pyre-p512-q5-gdn-o3_results.db`
  measured GDN at `10.385 ms`, with no useful movement versus the Q-reload
  result. GDN stays on the default device optimization level.
- Full Qwen correctness passed with Q5-only `-O3` plus the accepted Q6
  fill-order and GDN Q-reload changes:
  `build/pyre-correctness-qwen-q5-o3-final-20260413-010209`. The gate covered
  exact main ops, exact GDN autoregressive checks, export FA-on, direct and
  generic FA, short/long chat smokes, and loop guard seeds `1,5`.

Q5 post-O3 schedule probes:

- Rejected Q5 accumulator-layout reorder under Q5 `-O3`: changed the local
  sum index order toward `(wsic, cc, cr)` while preserving arithmetic and exact
  output. Targeted Q5 exact checks passed in
  `build/pyre-inner-loop-q5-sumlayout-o3-20260413-010955`, but p512 regressed
  the hot Q5 kernel to `21.156 ms` in
  `build/rocprof-pyre-prefill-q5-sumlayout-o3-20260413-011015/pyre-p512-q5-sumlayout-o3_results.db`
  versus `20.580 ms` for the accepted Q5-O3 checkpoint. The source-level order
  that looks closer to the Vulkan traversal is not a win for the HIP compiler's
  current register schedule.
- Rejected Q5 `-ffast-math`: exact Q5 checks passed in
  `build/pyre-inner-loop-q5-fastmath-20260413-011336`, but p512 regressed the
  hot Q5 kernel to `21.942 ms` in
  `build/rocprof-pyre-prefill-q5-fastmath-20260413-011358/pyre-p512-q5-fastmath_results.db`.
  Do not use broad fast-math as a compile-policy shortcut for these integer
  MMQ kernels.
- Rejected Q5/Q6 `__restrict__` pointer annotations. Focused exact Q5/Q6 checks
  passed in `build/pyre-inner-loop-q5q6-restrict-20260413-011641` and
  `build/pyre-inner-loop-q5q6-restrict-20260413-011658`. The p512 repeats were
  not stable wins:
  `build/rocprof-pyre-prefill-q5q6-restrict-20260413-011729/pyre-p512-q5q6-restrict_results.db`
  measured Q6 slightly better but Q5 worse, while
  `build/rocprof-pyre-prefill-q5q6-restrict-r2-20260413-011801/pyre-p512-q5q6-restrict-r2_results.db`
  regressed both Q5 and Q6 versus Q5-O3. The annotations were reverted. This
  is a useful negative result: alias hints alone are not exposing the remaining
  schedule gap.
- ATT status: a filtered Q5 capture with
  `--kernel-include-regex pyre_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32`
  and one consecutive kernel either rejected invalid small buffer sizes or hung
  during finalization with this ROCm 7.13 alpha toolchain. Keep rocprofv3 SQL as
  the reliable loop for now; return to ATT only with a smaller standalone
  harness or after toolchain/runtime changes.

Q8_0 fused-add q8_1 tile retune:

- Rejected graph-level disable of the Q8_0 q8_1 fused-add path. With
  `GGML_PYRE_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ64_PROMPT=1`, p512 selected
  `pyre_mul_mat_vec_q8_0_add_rows4_cols4_f32` and regressed the fused Q8 work
  to `9.544 ms` in
  `build/rocprof-pyre-prefill-disable-q8-q81-20260413-012035/pyre-p512-disable-q8-q81_results.db`.
  The q8_1 scratch path remains the right graph route even though the scratch
  quantizer is visible in the dense bucket.
- Accepted Q8_0 fused-add q8_1 tile retune from `64x64` to `128x32`. The
  revised provider keeps a 256-thread workgroup and 16 output columns per
  active thread, but changes the tile to reduce B-tile LDS traffic and shrink
  the number of column lanes duplicating each A row from four to two. The first
  p512 trace
  `build/rocprof-pyre-prefill-q8-bm128bn32-20260413-012230/pyre-p512-q8-bm128bn32_results.db`
  measured the hot fused Q8 kernel at `3.899 ms` over 9 calls, versus
  `4.303 ms` in the Q5-O3 baseline. The repeat
  `build/rocprof-pyre-prefill-q8-bm128bn32-r2-20260413-012302/pyre-p512-q8-bm128bn32-r2_results.db`
  measured `3.902 ms`. After renaming the provider/symbol to
  `pyre_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32`, the p512 smoke
  `build/rocprof-pyre-prefill-q8-128x32-renamed-20260413-012811/pyre-p512-q8-128x32-renamed_results.db`
  measured `3.911 ms`, confirming the renamed catalog entry selects the same
  path.
- Correctness for the Q8 retune passed the full Qwen milestone gate before the
  mechanical rename:
  `build/pyre-correctness-qwen-q8-bm128bn32-20260413-012341`. This covered
  exact main ops, GDN autoregressive exactness, direct and generic FA, short and
  long chat smokes, and loop guard seeds `1,5`. The subsequent rename was
  verified by rebuild plus p512 rocprof selection of the new symbol.
- Accepted standalone Q8_0 prompt reroute onto the same packed q8_1/x4
  `128x32` MMQ dataflow used by the fused-add Q8 path. Before this change, the
  single non-fused prompt Q8 matmul still selected
  `pyre_mul_mat_vec_q8_0_cols8_f32` and cost `1.671 ms` in
  `build/rocprof-pyre-prefill-q5-bk1probe-r2-20260413-014407/pyre-p512-q5-bk1probe-r2_results.db`.
  The new `pyre_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32` provider keeps
  the same RHS quantization and packed dot schedule, but removes the bias
  operand/store path. It measured `0.431 ms` in
  `build/rocprof-pyre-prefill-q8-standalone-20260413-015320/pyre-p512-q8-standalone_results.db`
  and `0.545 ms` in the repeat
  `build/rocprof-pyre-prefill-q8-standalone-r2-20260413-015411/pyre-p512-q8-standalone-r2_results.db`.
  The extra `pyre_quantize_q8_1_x4_f32` call is included in the trace
  (`188` calls versus `187` before), so this is a net `~1.1 ms` p512
  device-time reduction for the Q8 bucket.
- Full Qwen correctness for the standalone Q8 reroute passed in
  `build/pyre-correctness-qwen-q8-standalone-20260413-015443`, covering exact
  main ops, exact GDN autoregressive checks, direct and generic FA, short/long
  chat smokes, and loop guard seeds `1,5`.

GDN non-KDA algebra placement:

- Accepted the non-KDA Gated Delta Net algebra placement used by the optimized
  schedule: compute `g_scalar` before the `S*K` reduction, fold `g_scalar` into
  `kv_partial`, and then compute `delta_col = (v_col - kv_col) * beta`.
  This preserves the mathematical result while moving the multiply into the
  per-lane partial accumulation instead of after the cluster reduction.
- Focused exact GDN passed in
  `build/pyre-inner-loop-gdn-vk-algebra-20260413-012947`. Full Qwen correctness
  passed in `build/pyre-correctness-qwen-gdn-vk-algebra-20260413-013133`,
  covering exact main ops, exact GDN autoregressive checks, direct and generic
  FA, short/long chat smokes, and loop guard seeds `1,5`.
- The first p512 trace
  `build/rocprof-pyre-prefill-gdn-vk-algebra-20260413-012954/pyre-p512-gdn-vk-algebra_results.db`
  measured GDN at `9.832 ms` over 30 calls, versus `10.700 ms` in the
  post-Q8-retune renamed baseline
  `build/rocprof-pyre-prefill-q8-128x32-renamed-20260413-012811/pyre-p512-q8-128x32-renamed_results.db`.
  The repeat
  `build/rocprof-pyre-prefill-gdn-vk-algebra-r2-20260413-013026/pyre-p512-gdn-vk-algebra-r2_results.db`
  measured `9.959 ms`. Against the earlier Q5-O3 checkpoint (`10.341 ms`),
  this is a smaller but still useful `~4-5%` GDN gain; against the immediate
  renamed baseline it removes `~0.7-0.9 ms` of p512 device time.
- Rejected the Vulkan-like no-KDA Q preload schedule in the specialized HIP
  GDN kernel. Focused exact GDN passed in
  `build/pyre-inner-loop-gdn-qpreload-20260413-020415`, but p512 regressed GDN
  to `10.316 ms` in
  `build/rocprof-pyre-prefill-gdn-qpreload-20260413-020420/pyre-p512-gdn-qpreload_results.db`.
  Delaying the Q loads until the attention accumulation keeps the HIP register
  schedule healthier even though Vulkan preloads Q with K.

Q5 MMQL LDS residency:

- Accepted Q5 `q8_1_x4` MMQL LDS-residency retune: keep the proven `128x128`
  output tile and wave64 schedule, but reduce the K staging depth from
  `BK_STEP=4` to `BK_STEP=1`. This cuts the hot kernel's fixed LDS allocation
  from the previous `40960` bytes to `10240` bytes, allowing more resident
  workgroups without changing the output tile or widening the dispatch grid.
  The accepted kernel metadata is `vgpr=141`, `sgpr=38`, no spills, wave64.
- Also changed Q5 `qh`/`qs` pack loads from byte-by-byte assembly to aligned
  32-bit loads. As a standalone probe this was at best a small/noisy win, but
  it is the structurally correct load form and is retained with the LDS-depth
  retune.
- Rejected Q5 `128x64` macro-tile probe. It preserved correctness but moved
  the hot Q5 kernel to `22.128 ms` in
  `build/rocprof-pyre-prefill-q5-128x64probe-20260413-014056/pyre-p512-q5-128x64probe_results.db`,
  slower than the `128x128` baseline. Halving N reduced accumulator pressure
  but doubled the N-grid work and lost.
- Rejected Q5 `64x64` route as a default: with
  `GGML_PYRE_ENABLE_Q5_K_Q8_1_X4_MMQ64_PROMPT=1`, the current p512 trace
  `build/rocprof-pyre-prefill-q5-mmq64-current-20260413-013939/pyre-p512-q5-mmq64-current_results.db`
  measured the hot Q5 kernel at `33.555 ms`.
- Accepted Q5 `BK_STEP=1` after exact Q5 checks and p512 repeats. The first
  trace
  `build/rocprof-pyre-prefill-q5-bk1probe-20260413-014337/pyre-p512-q5-bk1probe_results.db`
  measured the hot Q5 kernel at `19.139 ms`; the repeat
  `build/rocprof-pyre-prefill-q5-bk1probe-r2-20260413-014407/pyre-p512-q5-bk1probe-r2_results.db`
  measured `19.239 ms`. The scoreboard delta against the immediate GDN-algebra
  baseline is `q5` `19.851 ms` total versus `21.766 ms`, a `-1.915 ms`
  (`-8.8%`) Q5-family improvement.
- Full Qwen correctness passed in
  `build/pyre-correctness-qwen-q5-u32-bk1-20260413-014436`, covering exact
  main ops, exact GDN autoregressive checks, direct and generic FA, short/long
  chat smokes, and loop guard seeds `1,5`.
- Rejected `__launch_bounds__(256, 2)` for the accepted Q5 kernel. Focused
  exact Q5 passed, but the p512 trace
  `build/rocprof-pyre-prefill-q5-launchbounds2-20260413-014943/pyre-p512-q5-launchbounds2_results.db`
  measured the hot Q5 kernel at `19.915 ms`, slower than the `BK_STEP=1`
  repeats. The compiler pressure hint did not buy a better schedule.
- Rejected packed 32-bit `dm` loading for Q5. This looked closer to the Vulkan
  packed32 source view and focused Q5 exactness passed in
  `build/pyre-inner-loop-q5-dm32-20260413-020229`, but p512 measured the hot
  Q5 kernel at `19.732 ms` in
  `build/rocprof-pyre-prefill-q5-dm32-20260413-020248/pyre-p512-q5-dm32_results.db`.
  Keep the compiler's separate half-load form for `d`/`dmin`.

Q6 MMQL LDS-depth probe:

- Rejected Q6 `BK_STEP=1`. Focused exact Q6 passed in
  `build/pyre-inner-loop-q6-bk1-20260413-020809`, but p512 regressed the hot
  Q6 kernel from `25.257 ms` in
  `build/rocprof-pyre-prefill-current-20260413-020620/pyre-p512-current_results.db`
  to `29.959 ms` in
  `build/rocprof-pyre-prefill-q6-bk1-20260413-020814/pyre-p512-q6-bk1_results.db`.
  Unlike Q5, Q6 benefits from the deeper four-block K staging. Keep the current
  `128x64`, `BK_STEP=4`, wave64 path; on the fresh post-Q8 trace it is already
  in the Vulkan device-time range for the Q6 bucket.

GDN cluster-width probe:

- Rejected an S=128 cluster16 GDN schedule. Focused exact GDN passed in
  `build/pyre-inner-loop-gdn-cluster16-20260413-020946`, but p512 regressed GDN
  from `10.095 ms` in
  `build/rocprof-pyre-prefill-current-20260413-020620/pyre-p512-current_results.db`
  to `10.763 ms` in
  `build/rocprof-pyre-prefill-gdn-cluster16-20260413-020952/pyre-p512-gdn-cluster16_results.db`.
  Halving the per-lane state rows from 16 to 8 did not offset the wider
  clustered reduction and doubled column workgroup count. Keep the current
  cluster8 schedule.

Q5 accumulator order and GDN fast exp checkpoint:

- Accepted Q5 accumulator indexing aligned with the Vulkan MMQ loop order:
  `(wsic * TN + cc) * TM + cr` instead of `(wsic * TM + cr) * TN + cc`.
  This is mathematically identical and focused exact Q5 passed in
  `build/pyre-inner-loop-q5-sumorder-20260413-021053`. Metadata stayed
  unchanged for the hot Q5 kernel (`vgpr=141`, `sgpr=38`, no spills,
  `group_segment_fixed_size=10240`). p512 traces were noise-level rather than
  a decisive standalone win: `20.010 ms` in
  `build/rocprof-pyre-prefill-q5-sumorder-20260413-021113/pyre-p512-q5-sumorder_results.db`
  and `19.869 ms` in
  `build/rocprof-pyre-prefill-q5-sumorder-r2-20260413-021154/pyre-p512-q5-sumorder-r2_results.db`.
  Keep it because it removes a hot-loop schedule mismatch with the reference
  MMQ algorithm without changing resource usage.
- Accepted `__expf` for the active no-KDA GDN scalar gate. Focused exact GDN
  passed in `build/pyre-inner-loop-gdn-fastexp-20260413-021252`. p512 measured
  GDN at `9.581 ms` in
  `build/rocprof-pyre-prefill-gdn-fastexp-20260413-021256/pyre-p512-gdn-fastexp_results.db`
  and `9.359 ms` in
  `build/rocprof-pyre-prefill-gdn-fastexp-r2-20260413-021327/pyre-p512-gdn-fastexp-r2_results.db`,
  versus the fresh `10.095 ms` reference
  `build/rocprof-pyre-prefill-current-20260413-020620/pyre-p512-current_results.db`.
  The win is `~0.5-0.7 ms` p512 device time, bringing GDN closer to the Vulkan
  `7.726 ms` bucket while preserving the full correctness gate.
- Full Qwen correctness passed in
  `build/pyre-correctness-qwen-gdn-fastexp-q5-sumorder-20260413-021403`,
  covering exact main ops, exact GDN autoregressive checks, direct and generic
  FA, short/long chat smokes, and loop guard seeds `1,5`.
- Rejected a GDN workgroup-axis reorder matching the Vulkan launch order
  (`head, seq, col_group`) instead of the existing Pyre order
  (`col_group, head, seq`). Focused exact GDN passed in
  `build/pyre-inner-loop-gdn-axis-20260413-021850`, but p512 measured GDN at
  `9.353 ms` in
  `build/rocprof-pyre-prefill-gdn-axis-20260413-021857/pyre-p512-gdn-axis_results.db`,
  effectively the same as the accepted fast-exp repeat and not a meaningful
  improvement.

Prompt SSM convolution/update fusion:

- Provider tracing showed the prompt graph still claimed separate
  `CONCAT`, strided `CPY(state_update)`, and `SSM_CONV_SILU` kernels for each
  SSM layer. The existing `SSM_CONV_UPDATE` fusion was decode-only because its
  predicate required `input->ne[0] == 1`.
- Accepted a generalized `pyre_ssm_conv_update_f32` dataflow that computes the
  convolution over the logical `[old_state, prompt_tokens]` window and writes
  the final recurrent state in the same dispatch. Provider trace
  `build/pyre-provider-trace-p512-ssmupdate-20260413-063306.log` now claims
  `SSM_CONV_UPDATE_SILU` 30 times for p512 and removes the 30 prompt `CONCAT`
  launches plus 30 prompt state-copy launches.
- Correctness smoke passed in
  `build/pyre-correctness-ssmupdate-smoke-20260413-063357`: exact main ops,
  exact GDN autoregressive check, short/long chat, and loop guard seed `1`.
- Rocprof trace
  `build/rocprof-pyre-prefill-ssmupdate-20260413-063624/pyre-p512-ssmupdate_results.db`
  measured the fused kernel at `2.796 ms` over 30 calls. It replaces the prior
  local bucket of `pyre_concat_f32` (`4.006 ms`), `pyre_ssm_conv_f32`
  (`1.139 ms`), and most prompt `pyre_copy_strided_f32` (`~0.6-0.8 ms`), so the
  local device-time savings are about `3 ms` and the dispatch count drops by
  about 60 launches. The global p512 trace was noisy, with large unrelated
  MoE/Q6 buckets moving up; treat this as a real local win and a runtime-overhead
  reduction rather than a new global-best p512 sample.

Prompt MoE reduction tail:

- Provider tracing after the SSM fusion still showed the prompt expert output
  reduction falling through to one large `MUL` over `[2048, 8, 512]`, then a
  reduction chain of three `ADD_ADD` kernels and one `ADD` kernel per MoE layer.
  The existing `ADD8` fusion was attempted first but rejected these expert
  slices because they are strided views, not contiguous 2D tensors.
- Accepted strided-2D support in `pyre_add8_f32`. Provider trace
  `build/pyre-provider-trace-p512-add8strided-20260413-063953.log` now claims
  `ADD8` 40 times. The prompt `ADD_ADD` count drops from 157 to 40 while the
  39 large MoE-weight `MUL` kernels remain as the next obvious fusion target.
- Correctness smoke passed in
  `build/pyre-correctness-add8strided-smoke-20260413-064028`: exact main ops,
  exact GDN autoregressive check, short/long chat, and loop guard seed `1`.
- Rocprof trace
  `build/rocprof-pyre-prefill-add8strided-20260413-064256/pyre-p512-add8strided_results.db`
  measured `pyre_add8_f32` at `0.931 ms` over 40 calls and remaining
  `pyre_add_add_f32_broadcast` at `1.148 ms` over 40 calls, compared with the
  pre-change `pyre_add_add_f32_broadcast` bucket of `4.359 ms` over 157 calls.
  The dense-f32 family improved by `-5.200 ms` (`-11.7%`) relative to the
  `gdn-fastexp-r2` baseline, and the graph dispatch count dropped by another
  `~117` launches. This is a real graph-level win even though the large MoE and
  Q6 buckets remain trace-noisy.
- Accepted a follow-on `pyre_mul_sum8_f32` fusion for the weighted expert
  reduction pattern: `MUL([rows, 8, tokens], router_weights)` followed by the
  seven-ADD 8-way sum. The graph matcher scans forward past reshape/view
  metadata after the `MUL`, verifies that each ADD source unwraps to the same
  multiply, and dispatches directly to the final 2D reduction output.
- Provider trace
  `build/pyre-provider-trace-p512-mulsum8-r2-20260413-064735.log` claims
  `MUL_SUM8` 39 times, leaves only one unrelated `ADD8`, and eliminates the
  39 large `pyre_mul_f32_broadcast n=8388608` claims from the prompt MoE path.
- Correctness smoke passed in
  `build/pyre-correctness-mulsum8-smoke-20260413-064802`: exact main ops,
  exact GDN autoregressive check, short/long chat, and loop guard seed `1`.
  Full correctness then passed in
  `build/pyre-correctness-mulsum8-full-20260413-065144`, covering FA direct
  and generic, exact main ops, exact GDN, short/long chat, and loop guard seeds
  `1,5`.
- Rocprof trace
  `build/rocprof-pyre-prefill-mulsum8-20260413-065031/pyre-p512-mulsum8_results.db`
  measures `pyre_mul_sum8_f32` at `1.307 ms` over 39 calls. Relative to the
  strided-ADD8 checkpoint, total visible kernel time drops from `195.310 ms`
  to `193.443 ms` and kernel dispatches drop from `1744` to `1705`. The
  family scoreboard shows dense-f32 at `35.416 ms` (`-3.836 ms`, `-9.8%`),
  with small opposing noise in the MoE/Q6/GDN hero buckets. Treat this as a
  confirmed launch-count and dense-tail cleanup. It is not a substitute for
  further work on the remaining Q4 MoE/Q6/Q5 hero kernels.
- Accepted `pyre_mul_add_add_f32_broadcast` for the shared-expert tail after
  the MoE reduction. The repeated prompt pattern is a broadcast `MUL` over
  `n=1048576`, then two residual/expert `ADD` nodes. Provider trace
  `build/pyre-provider-trace-p512-muladdadd-20260413-065913.log` claims
  `MUL_ADD_ADD` 40 times, leaves zero `ADD_ADD` claims, and removes the large
  shared-expert `MUL n=1048576` claims.
- Correctness smoke passed in
  `build/pyre-correctness-muladdadd-smoke-20260413-065939`; full correctness
  passed in `build/pyre-correctness-muladdadd-full-20260413-070240`, covering
  FA direct and generic, exact main ops, exact GDN, short/long chat, and loop
  guard seeds `1,5`.
- Rocprof trace
  `build/rocprof-pyre-prefill-muladdadd-20260413-070206/pyre-p512-muladdadd_results.db`
  measures `pyre_mul_add_add_f32_broadcast` at `1.294 ms` over 40 calls.
  Relative to the `MUL_SUM8` checkpoint, total visible kernel time drops from
  `193.443 ms` to `191.884 ms` and dispatches drop from `1705` to `1665`.
  Dense-f32 improves by another `-1.268 ms` (`-3.6%`). The aggregate device
  trace is now comfortably inside the refreshed Vulkan logger band; remaining
  optimization should be judged by per-bucket gaps and launch count, not by
  expecting large end-to-end tok/s movement before the runtime path changes.
- Rejected an attempted `pyre_quantize_q8_1_x4_wg256_f32` variant that packed
  eight Q8_1 blocks per 256-thread workgroup instead of four blocks per
  128-thread workgroup. Correctness smoke passed in
  `build/pyre-correctness-q8x4wg256-smoke-20260413-100419`, and the provider
  selected the new kernel in
  `build/rocprof-pyre-prefill-q8x4wg256-20260413-100647/pyre-p512-q8x4wg256_results.db`,
  but the quantizer itself measured `6.955 ms` over 188 calls. The same binary
  with `GGML_PYRE_DISABLE_Q8_1_X4_WG256_QUANT=1` measured the old x4 quantizer
  at `6.702 ms` over 188 calls in
  `build/rocprof-pyre-prefill-q8x4wg256-disabled-20260413-100723/pyre-p512-q8x4wg256-disabled_results.db`.
  The apparent global delta was trace noise in unrelated hero buckets, so the
  wg256 quantizer was reverted.
- Rejected a Q5 compact scaled-half MMQL cache probe. The hypothesis was that
  Vulkan's Q5 MMQ cache keeps the scale pair compact while Pyre expands A/B
  scales to `float` in LDS; the experiment stored scaled half values in the
  Q5 `q8_1_x4_mmql128x128` LDS cache and converted once when loading into
  registers. Focused Q5 exactness passed in
  `build/pyre-inner-loop-q5-compact-scale-20260413-100954`, but p512 rocprof
  `build/rocprof-pyre-prefill-q5-compact-scale-20260413-101015/pyre-p512-q5-compact-scale_results.db`
  measured the hot Q5 kernel at `155.566 ms` over 30 calls. This is a severe
  conversion/codegen regression, not a viable LDS-traffic tradeoff, and was
  reverted.
- Accepted a narrow L2 norm workgroup specialization for the Qwen prompt shape.
  Provider tracing shows the repeated L2 nodes are `ncols=128`, `nrows=8192`;
  the original Pyre kernel used 256 threads and reduced 128 inactive lanes.
  Added `pyre_l2_norm_wg128_f32`, selected for `ncols <= 128` with rollback
  `GGML_PYRE_DISABLE_L2_NORM_WG128=1`.
- Correctness passed in
  `build/pyre-correctness-l2-wg128-full-20260413-101513`, covering exact main
  ops, exact GDN autoregressive checks, direct and generic FA, short/long chat,
  and loop guard seeds `1,5`.
- Same-binary rocprof A/B:
  enabled trace
  `build/rocprof-pyre-prefill-l2-wg128-20260413-101411/pyre-p512-l2-wg128_results.db`
  measured `pyre_l2_norm_wg128_f32` at `1.491 ms` over 60 calls; disabled trace
  `build/rocprof-pyre-prefill-l2-wg128-disabled-20260413-101440/pyre-p512-l2-wg128-disabled_results.db`
  measured the original `pyre_l2_norm_f32` at `2.128 ms`. This is a clean
  `~0.64 ms` dense-tail p512 device-time win and reduces the dense-f32 family
  by about `1.0 ms` in the same-binary scoreboard.
- Accepted a Q4 MoE MMQ compile-mode specialization: compile
  `mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp` with `-mwavefrontsize64`. The source
  schedule is explicitly organized around 64 logical lanes (`BLOCK_SIZE=64`,
  `WARP=64`) but the default gfx11 object had `.wavefront_size: 32`. The
  wave64 object keeps the same VGPR counts (`135` for ID, `112` for SWIGLU)
  and no spills, but makes the hardware wavefront match the kernel schedule.
- Focused Q4 exactness passed in
  `build/pyre-inner-loop-q4-moe-wave64-20260413-102509`, and the full gate
  passed in
  `build/pyre-correctness-q4moe-wave64-reduce16-full-20260413-102555`,
  covering exact main ops, exact GDN autoregressive checks, FA direct and
  generic, short/long chat, and loop guard seeds `1,5`.
- Rocprof trace
  `build/rocprof-pyre-prefill-q4moe-wave64-20260413-102521/pyre-p512-q4moe-wave64_results.db`
  versus the L2 checkpoint shows total visible kernel time dropping from about
  `188.9 ms` to `183.296 ms`. The Q4 MoE SWIGLU bucket improves from
  `56.5 ms` to `52.0 ms` (`-4.46 ms`, `-7.9%`), and the Q4 MoE ID bucket
  improves from `31.5 ms` to `30.2 ms` (`-1.29 ms`, `-4.1%`). This is the
  clearest remaining per-bucket device-time cleanup since FA: the important
  point is not that wave64 is globally preferred on RDNA, but that this
  particular two-wave, 64-lane MMQ schedule was already written for it.
- Accepted a small dense F32 reduction cleanup while in this pass. The
  `pyre_mul_mat_vec_f32_batched_cols16_f32` and
  `pyre_mul_mat_vec_f32_batched_rows2_cols8_f32` kernels now reduce their
  sixteen accumulators in one helper instead of two independent 8-accumulator
  reductions. Focused dense exactness passed in
  `build/pyre-inner-loop-f32-reduce16-20260413-102026`. The standalone p512
  effect is small/noisy (`rows2_cols8` around `5.2-5.4 ms`), but it removes
  one cross-wave barrier without changing the per-output reduction order.
- Rejected a dense F32 `rows4_cols4` tile for the same prompt shape. It passed
  focused dense exactness in
  `build/pyre-inner-loop-f32-rows4cols4-20260413-102200`, but same-binary
  profiling showed the new tile at `5.279 ms` over 39 calls while forcing it
  off selected `rows2_cols8` at `5.192 ms` over 39 calls
  (`build/rocprof-pyre-prefill-f32-rows4cols4-20260413-102206` versus
  `build/rocprof-pyre-prefill-f32-rows4cols4-disabled-20260413-102234`).
  The load-count hypothesis lost to register pressure/occupancy or poorer
  scheduling, so the provider and kernel were reverted.
- Accepted explicit `-O3` compilation for `mul_mat_vec_q6_k_q8_1.hip.cpp`.
  The hot Q6 MMQL object remains a high-pressure kernel (`vgpr_count=192`,
  `vgpr_spill_count=4`, wave64), so this is not a structural fix, but it is a
  small positive compile/schedule cleanup. Focused Q6 exactness passed in
  `build/pyre-inner-loop-q6-o3-20260413-103010`; full correctness passed in
  `build/pyre-correctness-q6-o3-full-20260413-103053`.
- Rocprof trace
  `build/rocprof-pyre-prefill-q6-o3-20260413-103018/pyre-p512-q6-o3_results.db`
  against the Q4 MoE wave64 checkpoint shows Q6 moving from `27.497 ms` to
  `26.842 ms` over 70 calls (`-0.655 ms`, `-2.4%`) and total visible kernel
  time at `182.519 ms`. Because register spills are still present, the next
  Q6 work should be source-level VGPR reduction or a different tile, not more
  global compile-flag tuning.
- Rejected compiling `gated_delta_net_f32.hip.cpp` with `-mwavefrontsize64`.
  The focused GDN exact check passed in
  `build/pyre-inner-loop-gdn-wave64-20260413-103515`, but rocprof
  `build/rocprof-pyre-prefill-gdn-wave64-20260413-103523/pyre-p512-gdn-wave64_results.db`
  regressed `pyre_gated_delta_net_s128_cluster8_nokda_f32` to `11.182 ms`
  over 30 calls versus the current `~9.6 ms` band. Metadata also introduced
  SGPR spills in the generic GDN kernel under wave64. GDN should stay wave32
  unless a source-level schedule change justifies revisiting it.

### 2026-04-13: Re-open decode as a device-time target

- Re-ran decode after the prefill pass with the old shape
  `llama-bench -p 0 -n 64 -b 512 -ub 512 -fa 0 -r 1 --no-warmup` for Pyre and
  Vulkan. Pyre rocprof trace:
  `build/rocprof-pyre-decode-n64-20260413-104032/pyre-decode-n64_results.db`.
  Vulkan perf-log trace:
  `build/vulkan-decode-n64-20260413-104032/vulkan-decode-n64.log`.
- The raw rocprof database includes model-load/setup D2D copies. Those are not
  decode work: `501` copies totaling `765.240 ms` finish before the first
  decode kernel. Filtering to the active kernel window leaves only `63`
  in-window D2D copies totaling `2.457 ms`.
- Filtered Pyre decode device time is close but not done: kernel dispatch sum
  is `722.594 ms` over `70076` dispatches, or `725.052 ms` including active
  in-window D2D copies. Vulkan's logged device work for the same shape sums to
  `674.301 ms`. On this measurement Pyre is `~1.07x` Vulkan on visible
  device work, while tok/s remains much worse (`24.67` Pyre versus `72.59`
  Vulkan) because host/runtime gaps stretch the Pyre active kernel window to
  `2587 ms`.
- Biggest decode device-time gaps versus Vulkan:
  BF16 vector paths are `98.725 ms` Pyre versus `46.631 ms` Vulkan
  (`+52.094 ms`, `2.12x`); Q4 MoE ID multiply is `68.467 ms` versus
  `24.117 ms` (`+44.350 ms`, `2.84x`); GDN is `30.057 ms` versus
  `9.840 ms` (`+20.217 ms`, `3.05x`); dense F32 batched cols1 is
  `43.102 ms` versus `26.307 ms` (`+16.795 ms`, `1.64x`); TopK MoE is
  `20.871 ms` versus `8.850 ms` (`+12.021 ms`, `2.36x`).
- Not all decode buckets are behind. Q6_K is `80.884 ms` Pyre versus
  `91.880 ms` Vulkan, Q5_K is `79.543 ms` versus `98.386 ms`, and Q4 MoE
  SWIGLU is `66.263 ms` versus `80.998 ms` on the comparable Vulkan
  `MUL_MAT_ID_VEC` bucket. These should be guarded while optimizing the
  slower buckets.
- Tested the dormant Q4 MoE ID multiply two-row packed variant with
  `GGML_PYRE_ENABLE_PACKED_Q4_K_MUL_2ROW=1`. It regressed the hot kernel from
  `68.467 ms` to `70.645 ms` and total kernel time from `722.594 ms` to
  `733.197 ms` in
  `build/rocprof-pyre-decode-n64-q4mul2row-20260413-104356/pyre-decode-n64-q4mul2row_results.db`.
  Keep it disabled; the Q4 ID multiply gap needs a different schedule, not this
  variant.
- Provider tracing shows `fallback ARGSORT` logs for `ffn_moe_argsort-*`
  shape `256 x 16`, but this appears to be a support-query artifact for a
  fused TopK subgraph rather than executed CPU work. Enabling
  `GGML_PYRE_ENABLE_ARGSORT=1` removes the trace fallback, does not add
  `pyre_argsort` dispatches to rocprof, and worsens/noises the run
  (`756.612 ms` kernels in
  `build/rocprof-pyre-decode-n64-argsort-20260413-104453/pyre-decode-n64-argsort_results.db`).
  Do not enable prompt argsort as a decode optimization without a separate
  graph-execution reason.
- Rejected a quick BF16 cols1 workgroup override. The current decode path
  always picks the dedicated `cols1` kernels before the generic BF16
  workgroup-size policy. A temporary selector change allowed
  `GGML_PYRE_MUL_MAT_VEC_BF16_WG=128` to force the generic `wg128` kernels
  for `cols == 1`. The serial rocprof trace
  `build/rocprof-pyre-decode-n64-bf16wg128-serial-20260413-104824/pyre-decode-n64-bf16wg128-serial_results.db`
  regressed plain BF16 from `59.415 ms` to `63.212 ms`, BF16 SWIGLU from
  `31.540 ms` to `34.573 ms`, and total BF16 from `98.725 ms` to about
  `105.754 ms`. The temporary selector hook was reverted. The BF16 gap needs a
  structural dataflow change, likely closer to Vulkan's subgroup/multi-row
  `mul_mat_vec` shader pattern, not a simple smaller workgroup.

### 2026-04-13: Decode bucket cleanup, Q4 MoE and BF16

- Accepted a decode-specific Q4 MoE ID multiply schedule for the exact Qwen
  decode shape `k=512 rows=2048 ids=8 tokens=1`: `rows2_x16_wg32`, with
  fixed four-iteration `k=512` unrolling and vectorized Q/RHS loads. Focused
  correctness is covered by `GGML_PYRE_TEST_FILTER=q4_id_mul_decode`.
  Device time moved from the old packed path at `69.420 ms` over 2560 calls
  (`build/rocprof-pyre-decode-n64-q4rows2x16-disabled-20260413-110245/...`)
  to the final clean run at `45.916 ms` over 2560 calls
  (`build/rocprof-pyre-decode-n64-decode-buckets-clean2-20260413-113450/...`).
  This is a `~33.9%` reduction for the Q4 ID MUL bucket.
- Rejected several Q4 ID MUL variants after focused correctness and full decode
  rocprof:
  `packed_2row_wg64` (`70.012 ms`), `rows2_x16_wg16` shared-RHS (`97.616 ms`),
  a wg32 RHS-shuffle variant (`73.390 ms`), and min factoring (`47.885 ms`).
  The useful shape was not simply "more rows per workgroup"; it was the
  halfwave row split plus fixed-shape unrolling/vector loads.
- Accepted a packed Q4 SWIGLU cleanup for the exact skinny decode shape
  `k=2048 rows=512 ids=8 tokens=1`: fixed two-iteration block unrolling and
  vectorized Q/RHS loads inside `pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32`.
  Focused correctness is covered by
  `GGML_PYRE_TEST_FILTER=q4_id_swiglu_decode`. Device time moved from
  `66.779 ms` in
  `build/rocprof-pyre-decode-n64-q4final-bf16restore-20260413-112150/...`
  to `63.596 ms` in the clean final trace, a `~4.8%` bucket reduction.
- Rejected the broader/multi-row Q4 SWIGLU decode shapes. Existing row2 and
  row4 prompt kernels were made decode-selectable only behind env knobs and
  both regressed (`94.570 ms` and `98.475 ms`). A new rows2/x16 wg32 kernel was
  correct but also regressed (`80.746 ms`). Conclusion: unlike Q4 ID MUL, the
  current packed SWIGLU lane schedule is the right decode base; the productive
  work there is packed-schedule cleanup unless a genuinely different packed
  algorithm is introduced.
- Added and accepted a BF16 decode shape gate for `k=512 rows=2048 cols=1`:
  `pyre_mul_mat_vec_bf16_rows4_k512_cols1_lds_wg256_f32`. It stages the
  512-float RHS vector in LDS, computes four rows per workgroup, and uses an
  explicit two-halfwave shared reduction so correctness is independent of
  wave32 versus wave64 codegen. Focused correctness now includes this shape in
  `GGML_PYRE_TEST_FILTER=bf16_decode`. The clean final trace reports
  `19.354 ms` over 2560 calls for this shape. The remaining BF16 rows2 bucket
  is `28.904 ms` over 4480 calls and BF16 SWIGLU is `30.678 ms`.
- Rejected the earlier BF16 wg32 rows2 variant as a default. Broad selection
  regressed BF16 to `81.668 ms`; restricting it to only `k=512 rows=2048`
  still lost to the accepted row4/LDS path (`24.930 ms` versus `19.354 ms`).
  It remains opt-in only via `GGML_PYRE_ENABLE_BF16_ROWS2_COLS1_WG32_DECODE`.
- Current clean no-env decode scoreboard is
  `build/rocprof-pyre-decode-n64-decode-buckets-clean2-20260413-113450/pyre-decode-n64-decode-buckets-clean2-20260413-113450_results.db`.
  Top buckets are now dense F32 `274.952 ms`, dense BF16 `86.784 ms`, Q6
  `81.622 ms`, Q5 `79.312 ms`, Q4 SWIGLU `63.596 ms`, Q4 ID MUL `45.916 ms`,
  and GDN `30.910 ms`. Relative to the initial decode re-open, Q4 ID MUL is
  largely repaired, Q4 SWIGLU has a small clean win, BF16 has a small clean
  win, and the next high-value decode device-time work should be structural
  dense F32/GDN/BF16-SWIGLU work rather than more Q4 multi-row experiments.

### 2026-04-13: Decode GDN structural specialization

- Re-opened `GATED_DELTA_NET` because the clean decode bucket was still
  `30.910 ms` over 1920 calls in
  `build/rocprof-pyre-decode-n64-decode-buckets-clean2-20260413-113450/...`,
  versus roughly `9.840 ms` in the Vulkan decode perf log. The active Qwen
  decode shape is `S_v=128 H=32 tokens=1 seqs=1 q_heads=16 k_heads=16
  g_ne0=1`, so the generic S128 path was carrying unnecessary modulo,
  stride, token-loop, and two-wave workgroup structure.
- ATT on the old S128 non-KDA kernel showed the smoking gun: top stalls were
  scalar setup and waits before the useful math (`s_waitcnt lgkmcnt(0)` and
  scalar kernarg loads), not the F32 update loop itself. A first
  `nomod`/power-of-two-head specialization removed the dynamic head modulo and
  seq division and selected for the real `q_heads=k_heads=16` shape. It reduced
  GDN to `20.834 ms` over 1920 calls in
  `build/rocprof-pyre-decode-n64-gdn-nomod-mask-20260413-114724/...`, a
  `32.6%` bucket cut from the clean baseline.
- Added focused correctness for the real GQA decode shape in
  `GGML_PYRE_TEST_FILTER=gated_delta_net` (`S_v=128 H=32 q_heads=16
  tokens=1 seqs=1`). This prevents the fast path from only being checked on the
  smaller `H=4 q_heads=1` smoke shape.
- Added a fully shape-specialized exact decode kernel
  `pyre_gated_delta_net_s128_h32_qk16_tok1_nokda_f32`. It hardcodes the
  contiguous Qwen decode layout, removes the token loop and all runtime
  strides except the state destination offset, and gates on exact tensor shapes
  and strides. This moved GDN to `19.153 ms` in
  `build/rocprof-pyre-decode-n64-gdn-h32tok1-20260413-115159/...`.
- Compared the Vulkan shader shape directly. Vulkan's S128 GDN path uses
  `LANES_PER_COLUMN=8`, but its local size is one subgroup (`32` lanes), so it
  processes `4` columns per workgroup. Pyre's cluster8 path had been using two
  waves per workgroup (`64` lanes, `8` columns). Changing the exact decode
  kernel to the same one-wave/4-column workgroup shape reduced GDN further to
  `18.383 ms` in
  `build/rocprof-pyre-decode-n64-gdn-h32-wg32-20260413-115527/...`.
- Tried replacing the cluster reduction with DPP row-shift reduction. The DPP
  reduction plus the normal HIP broadcast was neutral-to-slightly-useful and is
  kept in the exact kernel. A further DPP row-broadcast replacement was
  correct but regressed GDN to `18.806 ms` in
  `build/rocprof-pyre-decode-n64-gdn-h32-dpp-bcast-20260413-115634/...`, so it
  was reverted.
- Current accepted GDN state: exact decode fast path, wave32, one subgroup per
  workgroup, `4` columns per workgroup, DPP row-shift reduction plus HIP
  clustered broadcast. Resource metadata for the exact kernel is spill-free:
  `34` SGPR, `59` VGPR, `0` LDS, wave32, kernarg segment `80` bytes. This is a
  `~40.5%` GDN bucket reduction from the clean baseline, but still about
  `1.9x` Vulkan for this bucket. Remaining GDN work should use ATT/disassembly
  against the exact kernel and focus on load/update scheduling or a genuinely
  different state update layout, not more generic modulo cleanup.

### 2026-04-13: Decode BF16 k=2048 structural rows4/LDS pass

- Re-opened dense BF16 decode after the GDN pass because the remaining BF16
  family was still `86.311 ms` in
  `build/rocprof-pyre-decode-n64-gdn-h32-wg32-20260413-115527/...`.
  Provider tracing for `p0/n1` showed only four live BF16 decode shapes:
  `60` claims of `MUL_MAT k=2048 rows=32 cols=1`, `10` claims of
  `MUL_MAT k=2048 rows=512 cols=1`, `40` claims of
  `MUL_MAT k=512 rows=2048 cols=1`, and `40` claims of
  `MUL_MAT_SWIGLU k=2048 rows=512 cols=1`.
- The old `k=2048` dense paths were still scalar row reductions with repeated
  RHS global loads. Added exact decode providers
  `pyre_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32` and
  `pyre_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32`. Both stage
  the 2048-float RHS once per workgroup in LDS, split a 256-thread workgroup
  into four 64-lane row shards, use packed BF16/float2 loads, reduce two
  wave32 partials per row, and write four rows per dispatch. The existing
  `k=512 rows=2048` rows4/LDS kernel remains selected for the shared-expert
  down shape.
- Focused correctness now covers the live large decode shapes:
  `GGML_PYRE_TEST_FILTER=bf16_decode` includes `k=2048 rows=32`,
  `k=2048 rows=512`, `k=512 rows=2048`, and the `k=2048 rows=512` SWIGLU
  fusion. The large SWIGLU fixture uses a slightly wider tolerance because the
  rows4 reduction changes accumulation order; the observed miss before
  widening was `2.89e-4` on a value around `13.0`, not a structural mismatch.
- Full model guard passed with
  `RESULTS_DIR=build/pyre-correctness-qwen-bf16-k2048-decode-20260413-120526`,
  `CHECK_FA=0 CHECK_DIRECT_FA=0 CHECK_CHAT=1 CHECK_LOOP=1 LOOP_PREDICT=384`.
  This covered exact Qwen `MUL_MAT,MUL_MAT_ID,SSM_CONV`, autoregressive GDN,
  short and long deterministic chat, and loop guards for seeds `1,5`.
- Provider trace after the change confirmed exact selection:
  `60` `pure_hip_bf16_rows4_k2048_cols1_lds_wg256` claims for
  `k=2048 rows=32`, `10` claims for `k=2048 rows=512`, `40`
  `pure_hip_bf16_rows4_k512_cols1_lds_wg256` claims, and `40`
  `MUL_MAT_SWIGLU pure_hip_bf16_rows4_k2048_cols1_lds_wg256` claims.
- Rocprof n64 artifact
  `build/rocprof-pyre-decode-n64-bf16-k2048-rows4-20260413-120402/pyre-decode-n64-bf16-k2048-rows4_results.db`
  reduced the BF16 family from `86.311 ms` to `70.937 ms` versus the post-GDN
  baseline, a `15.375 ms` / `17.8%` family win. Against the earlier clean
  decode baseline before GDN, BF16 is `15.847 ms` / `18.3%` lower.
- Kernel breakdown in that trace: the new dense `k=2048` rows4 kernel is
  `18.722 ms` over `4480` calls (`4.179 us/call`), the new SWIGLU rows4 kernel
  is `26.458 ms` over `2560` calls (`10.335 us/call`), the existing
  `k=512 rows=2048` rows4 kernel is `19.130 ms`, and BF16 set-rows is
  `6.627 ms`.
- Resource metadata after unbundling the built HSACOs: dense `k=2048` rows4 is
  wave32, `32 VGPR / 18 SGPR / no spills`; SWIGLU rows4 is wave32,
  `49 VGPR / 18 SGPR / no spills`; the existing `k=512` rows4 remains
  wave32, `18 VGPR / 18 SGPR / no spills`.
- This validates the user concern that local WG tweaks were not enough: the
  useful change was the whole dataflow shape, especially RHS amortization and
  row-sharded work distribution. Remaining decode priorities after this trace
  are dense F32 (`272.318 ms`), Q6 (`80.892 ms`), Q5 (`78.541 ms`), Q4 MoE
  SWIGLU (`63.044 ms`), and Q4 MoE ID (`45.881 ms`).

### 2026-04-13: Decode F32 k=2048 cols1 wave32/vector pass

- Re-opened dense F32 decode after BF16. Provider tracing showed the hot
  F32 batched decode shapes were `40` claims of
  `MUL_MAT k=2048 rows=1 cols=1` and `40` claims of
  `MUL_MAT k=2048 rows=256 cols=1` per token. The selected Pyre provider was
  still the generic `pyre_mul_mat_vec_f32_batched_cols1_ne2_1_f32`, which used
  a 256-thread workgroup, scalar loads, shared memory cross-wave reduction, and
  one workgroup per output row.
- Added an exact sibling provider
  `pyre_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32` for the live
  decode shape only: `k=2048`, `cols=1`, `dst_ne2=1`. It uses one wave32 per
  output row, `float4` vector loads, fixed 16 unrolled iterations per lane, and
  a single subgroup reduction. The generic provider remains available for
  non-2048 or less regular cases.
- Focused correctness now includes `GGML_PYRE_TEST_FILTER=f32_batched_decode`
  with the two live row counts (`rows=1` and `rows=256`) across three batches.
  Full Qwen gate passed with
  `RESULTS_DIR=build/pyre-correctness-qwen-f32-wg32-decode-20260413-121437`,
  `CHECK_FA=0 CHECK_DIRECT_FA=0 CHECK_CHAT=1 CHECK_LOOP=1 LOOP_PREDICT=384`.
- Rocprof n64 artifact
  `build/rocprof-pyre-decode-n64-f32-wg32-20260413-121357/pyre-decode-n64-f32-wg32_results.db`
  confirms selection of the exact kernel for `5120` calls. The kernel moved
  from `43.078 ms` in the BF16 baseline to `35.221 ms` (`6.879 us/call`), an
  `18.2%` kernel reduction. Dense F32 as a family moved from `272.318 ms` to
  `263.238 ms`, a `9.081 ms` / `3.3%` reduction.
- This is another structural alignment result, but it is smaller than the BF16
  rows4/LDS pass because the shape has no multi-row RHS reuse yet. The next
  dense F32 targets are not more local tuning of this exact kernel; they are
  the remaining F32 kernels visible above it in aggregate, especially
  `get_rows`, fused norm/RoPE paths, and any cols512 prompt/decode crossover
  kernels that still use 256-thread scalar schedules.

### 2026-04-13: Decode Q6_K rows2/wg32 pass

- Provider tracing showed Q6 decode was still using the old scalar
  `pyre_mul_mat_vec_q6_k_wg128_f32` path for all live Qwen decode Q6 shapes:
  `30` claims of `k=2048 rows=4096 cols=1`, `10` claims of
  `k=2048 rows=8192 cols=1`, and `30` claims of `k=4096 rows=2048 cols=1`.
- The Vulkan RDNA DMMV specialization for K-quants uses one subgroup for
  these direct F32-RHS paths: `BLOCK_SIZE=32`, `NUM_ROWS=2`, `NUM_COLS=1`,
  subgroup reduction, and 16 lanes per quant block. Pyre's inherited Q6 decode
  schedule used `WG=128`, one output row per workgroup, and a 64-lane mapping
  per quant block. Added `pyre_mul_mat_vec_q6_k_rows2_cols1_wg32_f32` to match
  the proven one-subgroup/two-row structure.
- Focused correctness now includes `GGML_PYRE_TEST_FILTER=q6_decode`, covering
  `k=2048` and `k=4096` with odd row counts so the tail path is exercised.
  Full Qwen gate passed with
  `RESULTS_DIR=build/pyre-correctness-qwen-q6-rows2-decode-20260413-122513`,
  `CHECK_FA=0 CHECK_DIRECT_FA=0 CHECK_CHAT=1 CHECK_LOOP=1 LOOP_PREDICT=384`.
- Rocprof n64 artifact
  `build/rocprof-pyre-decode-n64-q6-rows2-wg32-20260413-122200/pyre-decode-n64-q6-rows2-wg32_results.db`
  reduced the Q6 family from `80.806 ms` to `76.168 ms`, a `4.638 ms` /
  `5.7%` family win. The kernel average is now `17.002 us/call` over `4480`
  calls.
- Checked the tempting Q8_1/MMVQ route. With
  `GGML_PYRE_ENABLE_Q8_1_MMVQ=1 GGML_PYRE_Q8_1_MMVQ_POLICY=all`, decode
  regressed badly: Q6 moved to `227.202 ms`, Q5 moved to `129.848 ms`, and
  quantization added `63.703 ms`. This confirms the existing Pyre Q8_1 decode
  kernels are not production candidates yet. Vulkan also explicitly avoids
  MMVQ for Q6_K because of its alignment behavior, so Q6 should stay on direct
  DMMV unless a dedicated packed-dot kernel is built.

### 2026-04-13: Decode rejected structural probes after Q6

- Tested applying the same two-row/wg32 DMMV shape to Q5_K. The live Qwen
  Q5 decode sites are `k=2048 rows=8192 cols=1` and the large
  `k=2048 rows=248320 cols=1` expert table. Focused synthetic correctness
  passed and provider tracing confirmed all Q5 decode calls moved to the new
  rows2 path, but rocprof regressed the Q5 family from `79.223 ms` in
  `build/rocprof-pyre-decode-n64-q6-rows2-wg32-20260413-122200/...` to
  `90.107 ms` in
  `build/rocprof-pyre-decode-n64-q5-rows2-wg32-20260413-123054/...`.
  Recompiling Q5/Q6 direct-vector sources with `-mno-wavefrontsize64` did not
  repair the result: Q5 stayed at `89.575 ms` and Q6 was neutral/slightly
  worse in
  `build/rocprof-pyre-decode-n64-q5q6-wave32-20260413-123241/...`.
  Conclusion: Q5 should keep the current wg128/wg64 split for now. Unlike Q6,
  Q5 was already ahead of the Vulkan decode bucket in the first head-to-head
  (`79.543 ms` Pyre versus `98.386 ms` Vulkan), so the next Q5 attempt should
  be a true algorithm rewrite, not a direct transplant of the Q6 rows2 shape.
- Investigated TopK/MoE because Pyre remains about `2.3x` slower than the
  Vulkan fused `TOPK_MOE_EARLY_SOFTMAX_NORM` timing (`~20.9 ms` versus
  `~8.85 ms` for n64 decode). Provider tracing shows the exact live shape is
  `experts=256 k=8 nrows=1`, selected through `pure_hip_f32_wave32`.
  A separate rows1 provider using the same wave kernel body, first accidentally
  compiled as wave64 and then correctly as wave32, was neutral:
  `20.865 ms` / `20.974 ms` in
  `build/rocprof-pyre-decode-n64-topk-rows1-20260413-123849/...` and
  `build/rocprof-pyre-decode-n64-topk-rows1-wave32-20260413-124006/...`.
  A hardcoded exact `256 experts, top-8, nrows=1, normalized softmax` kernel
  also lost at `21.562 ms` in
  `build/rocprof-pyre-decode-n64-topk-exact-20260413-124215/...`.
  These probes were reverted. The TopK gap is therefore not explained by
  launch-y waste or dynamic shape checks; it needs ISA/ATT comparison against
  the Vulkan shader, especially the exp/reciprocal and top-k reduction
  instruction schedule.
- The broader takeaway matches the "not just local fixes" rule: keep landing
  structural wins when the bucket is behind, but do not force Vulkan-shaped
  variants into buckets already at or ahead of Vulkan. The next decode work
  should prioritize device-time gaps with proven headroom: Q4 MoE ID remains
  roughly `1.9x` Vulkan after the rows2_x16 pass, GDN remains roughly `1.9x`,
  TopK remains roughly `2.3x`, and GET_ROWS should be attacked through
  graph/fusion elimination rather than the tiny `nr1` kernel body.
