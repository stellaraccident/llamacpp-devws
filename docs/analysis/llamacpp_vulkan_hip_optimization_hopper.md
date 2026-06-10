# llama.cpp Vulkan-to-HIP optimization hopper

**Status:** Draft analysis
**Date:** 2026-04-09

## Purpose

This is a backlog of Vulkan backend optimizations and implementation patterns
to port, verify, or deliberately reject in the Pyre pure-HIP backend. It is
biased toward decode on Qwen3.5-35B-A3B-UD-Q4_K_L, because that is the current
model and shape under active measurement, but it calls out prefill/batched
items where Vulkan has a different class of optimization.

Current active measurements in the companion spike log put Pyre no-FA decode at
about 26 tok/s and Vulkan at about 105 tok/s on the same model/shape. The point
of this list is not to pretend any single item is the whole gap; Vulkan has a
large body of accumulated graph and shader tuning, and Pyre needs to catch up in
bulk.

Impact estimates are directional:

- **Very high:** plausible double-digit end-to-end decode or major prefill gap.
- **High:** likely visible end-to-end on this model if provider trace shows the
  path is hot.
- **Medium:** likely visible in microbenchmarks; may matter after larger gaps
  are cleared.
- **Low:** worth tracking for parity or cleanliness, not a first-order decode
  bet.

## Immediate Verification Harness Work

1. **Provider trace diff against Vulkan fusion labels**
   - Vulkan pattern: graph compute records fusion names such as
     `MUL_MAT_ADD`, `MUL_MAT_ID_ADD_ID_MUL`, `RMS_NORM_MUL_ROPE`, and
     `TOPK_MOE_*` in `ggml-vulkan.cpp`.
   - Pyre status: provider tracing exists, but the easy comparison is still
     manual.
   - Impact: high as triage infrastructure.
   - Verify: run one-token or short `n=8` traces with Pyre and Vulkan perf
     labels enabled, then tabulate matched, missing, and fallback subgraphs.

2. **Per-provider timing in the Pyre trace path**
   - Vulkan pattern: timestamp query support exists around graph compute.
   - Pyre status: dispatch counts exist, but not all provider claims carry GPU
     time by op family.
   - Impact: high as triage infrastructure.
   - Verify: add coarse optional timestamps around dispatches or use Tracy zones
     to group `MUL_MAT`, `MUL_MAT_ID`, `TOPK_MOE`, FA, and state update fusions.

3. **Kernel microbench coverage for every hot provider**
   - Vulkan pattern: many shaders have internal matmul test hooks and many
     shape-specialized variants.
   - Pyre status: `pyre-kernel-bench` now covers some expert paths, but not all
     fused graph paths and not all Qwen shapes.
   - Impact: high.
   - Verify: ensure microbench cases exist for Q4_K/Q5_K/Q6_K matvec, Q4_K
     `MUL_MAT_ID`, `MUL_MAT_ID_MUL`, `MUL_MAT_ID_SWIGLU`, TopK MoE, FA decode,
     RMS+ROPE+SET_ROWS, and recurrent-state fusions.

## Decode Matvec and Expert Kernels

4. **Port Vulkan DMMV `NUM_COLS` specialization**
   - Vulkan pattern: `mul_mat_vec_max_cols = 8`; shader variants specialize
     `NUM_COLS=1..8` and carry multiple output columns in registers.
   - Pyre status: generic kernels accept `cols` and launch one workgroup per
     `(row, col)`.
   - Impact: high for decode matvec and expert paths.
   - Savings: fewer workgroups, better RHS reuse across adjacent columns, fewer
     reductions and stores.
   - Verify: specialize `cols=1,2,4,8` first and microbench Qwen shapes.

5. **Port Vulkan subgroup-only reduction variant**
   - Vulkan pattern: DMMV has regular, subgroup, and subgroup-no-shmem variants.
   - Pyre status: reductions use wave shuffles plus shared memory for 256-thread
     reductions.
   - Impact: high/medium depending shape and occupancy.
   - Savings: fewer barriers and shared-memory transactions.
   - Verify: add 64-thread and 128-thread subgroup-only variants; compare against
     current 256-thread reduction for Q4_K/Q5_K/Q6_K and F16/BF16 dense matvec.

6. **Tune workgroup size instead of hard-coding 256**
   - Vulkan pattern: DMMV selects subgroup or larger workgroup by device/vendor
     and dimensions.
   - Pyre status: most matvec kernels default to 256 threads.
   - Impact: high/medium.
   - Savings: less over-reduction overhead for small K; better occupancy for
     large row counts.
   - Verify: generate 64/128/256 variants and run the full Qwen-shaped grid.

7. **Match Vulkan Q4_K lane mapping and unrolling**
   - Vulkan pattern: `mul_mat_vec_q4_k.comp` processes 16 threads per block,
     vectorized RHS loads, unpacked scale/min groups, and fully unrolled
     per-block arithmetic.
   - Pyre status: Q4_K kernels use a structurally similar group/lane mapping but
     not the same register/vector load pattern or multi-column handling.
   - Impact: high for MoE expert matvecs if Q4_K dominates.
   - Savings: fewer scalar loads and better compiler scheduling.
   - Verify: port the exact `v_im`, `q_offset`, `y_offset`, vectorized RHS load
     structure and microbench before/after.

8. **Repeat Q4_K work for Q5_K**
   - Vulkan pattern: dedicated `mul_mat_vec_q5_k.comp`.
   - Pyre status: Q5_K exists, and a two-row experiment regressed, so shape
     reuse must be more careful.
   - Impact: medium/high if provider trace shows many Q5_K rows.
   - Savings: same class as Q4_K but model-dependent.
   - Verify: port inner loop only first; avoid changing row/workgroup mapping
     until local instruction count improves.

9. **Repeat Q4_K work for Q6_K**
   - Vulkan pattern: dedicated `mul_mat_vec_q6_k.comp`.
   - Pyre status: one scale-hoist optimization already produced a small win.
   - Impact: medium/high.
   - Savings: incremental instruction reduction and better vectorization.
   - Verify: continue shader-structure parity from the now-improved Q6 kernel.

10. **Add Vulkan-style F16 RHS DMMV variants**
    - Vulkan pattern: matvec variants cover `B_TYPE=float` and
      `B_TYPE=float16_t`.
    - Pyre status: many Pyre matvecs assume F32 RHS in the current path.
    - Impact: medium, possibly high if graph can keep RHS in F16.
    - Savings: half RHS bandwidth, less cache pressure.
    - Verify: first confirm whether the Vulkan baseline is actually using F16
      RHS on the hot Qwen decode matvecs; then add HIP F16-RHS variants.

11. **Use explicit vector loads in dense F16/BF16 matvec**
    - Vulkan pattern: shader generator uses vector types and aligned variants.
    - Pyre status: dense F16/BF16 kernels do scalar loads in a simple loop.
    - Impact: medium.
    - Savings: fewer load instructions and better memory coalescing.
    - Verify: `half2`/`float2`/`float4` experiments on dense FFN and attention
      projection shapes.

12. **Add non-contiguous/P021 matvec equivalents**
    - Vulkan pattern: `mul_mat_vec_p021.comp` and `mul_mat_vec_nc.comp`.
    - Pyre status: Pyre supports some batched F16/F32 layouts but has stricter
      support predicates and often requires contiguity.
    - Impact: medium if current copy eliminations leave strided matvec fallbacks.
    - Savings: avoids materialization and preserves graph fusion.
    - Verify: provider trace should show whether any `MUL_MAT` fallback or
      deferred copy still exists for non-contiguous operands.

13. **Add 64-bit-indexing equivalent only where needed**
    - Vulkan pattern: selects 64-bit indexing pipelines when buffers exceed
      max storage-buffer range.
    - Pyre status: HIP pointer arithmetic already uses 64-bit-ish C++ types in
      several kernels, but coverage is inconsistent.
    - Impact: low for performance, medium for correctness on huge tensors.
    - Verify: static audit of all kernels using `int`/`uint32_t` offsets.

14. **Re-evaluate Q8_1 MMVQ after runtime changes**
    - Vulkan pattern: auto-uses Q8_1 RHS when integer dot is available and
      heuristics pass.
    - Pyre status: opt-in `GGML_PYRE_ENABLE_Q8_1_MMVQ`; current microbench shows
      q8_1 expert path slower because quantize dispatch dominates.
    - Impact: medium now, high if quantize is fused or amortized.
    - Savings: lower RHS bandwidth and integer dot style arithmetic.
    - Verify: rerun with new runtime and with quantize time separated from matvec
      time.

15. **Cache Q8_1 RHS conversions more aggressively**
    - Vulkan pattern: `prealloc_y_last_pipeline_used` and
      `prealloc_y_last_tensor_used` avoid repeated conversion of the same tensor.
    - Pyre status: scratch buffer exists, but q8_1 conversion is dispatched in
      the matvec path for each eligible op.
    - Impact: medium/high if the same RHS tensor is reused in grouped paths.
    - Savings: removes quantize dispatch and memory traffic.
    - Verify: trace tensor pointers/names for q8_1 quantize calls; cache when
      source and shape match.

16. **Fused quantize+matvec experiment**
    - Vulkan pattern: separate quantize plus matvec, but Vulkan runtime overhead
      is low enough and has caching.
    - Pyre status: separate dispatch is expensive in current runtime.
    - Impact: medium/high if q8_1 arithmetic is locally faster.
    - Savings: removes one dispatch and scratch write/read.
    - Verify: for one Q4_K shape, fuse per-block RHS quantization into the dot
      kernel and compare against F32 RHS and separate Q8_1.

17. **Use integer dot or packed dot intrinsics explicitly if compiler misses**
    - Vulkan pattern: integer-dot SPIR-V variants exist for Q8_1 RHS.
    - Pyre status: earlier disassembly did not show `v_dot*`; current code uses
      scalar integer multiplies.
    - Impact: medium; prior Vulkan disable-int-dot test says this may not be
      first-order for current decode.
    - Savings: fewer integer ALU instructions.
    - Verify: disassemble q8_1 HIP object and test AMDGCN intrinsics or packed
      dot builtins in isolation.

18. **Expert ID subgroup variant**
    - Vulkan pattern: `matmul_id_subgroup` variants exist for matrix-matrix, and
      matvec-id variants have subgroup/no-shmem forms.
    - Pyre status: `MUL_MAT_ID` exists but follows the generic 256-thread
      reduction pattern.
    - Impact: high for Qwen MoE.
    - Savings: fewer barriers, better small-expert occupancy.
    - Verify: specialize `n_ids=8`, `n_tokens=1`, Q4_K expert path.

19. **Expert bias/add fusion parity**
    - Vulkan pattern: `MUL_MAT_ID + ADD_ID` and
      `MUL_MAT_ID + ADD_ID + MUL` are graph fusions.
    - Pyre status: `MUL_MAT_ID + MUL` exists; `ADD_ID` family does not appear
      mirrored.
    - Impact: high if Qwen graph contains `ADD_ID`.
    - Savings: one or two dispatches and full output read/write per expert.
    - Verify: provider trace for `ADD_ID`; if present, port Vulkan fusion.

20. **Dense `MUL_MAT + ADD(+ADD)` fusion**
    - Vulkan pattern: fuses matvec result bias adds when shapes/strides match.
    - Pyre status: not obviously present; Pyre has `ADD_ADD` but not this
      matvec-output fusion.
    - Impact: medium/high depending graph.
    - Savings: eliminate output write/read and add dispatches.
    - Verify: provider trace for `ADD` immediately after `MUL_MAT`.

## Flash Attention

21. **Split-K decode FA**
    - Vulkan pattern: dynamically splits KV when workgroup count is too low,
      writes partial O/M/L to `prealloc_split_k`, then reduces.
    - Pyre status: decode FA is a single workgroup per `(head, token, seq)`.
    - Impact: high at long context, low/medium at short context.
    - Savings: more CUs occupied when `N` is tiny and KV is large.
    - Verify: microbench KV sweep; add split-K only after FA is proven hot.

22. **Mask optimization prepass**
    - Vulkan pattern: `flash_attn_mask_opt.comp` skips all-zero or all-neg-inf
      mask blocks for sufficiently large masks.
    - Pyre status: decode FA checks mask inside the main loop.
    - Impact: medium for masked long-context attention.
    - Savings: skip K/V tiles and mask loads.
    - Verify: run with mask shapes from Qwen trace; count skipped blocks.

23. **GQA-aware FA workgroup remapping**
    - Vulkan pattern: for small N and GQA, changes the N dimension to the GQA
      ratio and reduces workgroups in Y.
    - Pyre status: current decode FA maps head/token/seq directly.
    - Impact: medium.
    - Savings: better occupancy and less redundant addressing.
    - Verify: compare head grouping and launch geometry for Qwen.

24. **FA shape tuning table**
    - Vulkan pattern: chooses `Br`, `Bc`, `D_split`, `row_split`,
      `SubGroupSize`, `SHMEM_STAGING`, f32/f16 accumulation, and occupancy
      limiter flags by shape/device.
    - Pyre status: one simpler decode kernel per K/V type.
    - Impact: medium/high once FA is hot.
    - Savings: better register/shared-memory balance.
    - Verify: add compile-time variants for the Qwen head size and KV sweep.

25. **Vectorized Q/K/V staging**
    - Vulkan pattern: uses `vec4`/`f16vec4` loads and optional shared-memory
      staging for K/V.
    - Pyre status: decode FA performs scalar loads and writes shared `logits`.
    - Impact: medium.
    - Savings: fewer memory instructions and better coalescing.
    - Verify: vectorize Q/K/V load loops for F16 and Q8/Q4 variants.

26. **Avoid double math in FA**
    - Vulkan pattern: scalar FA uses float accumulator configuration unless
      explicitly using f16/f32 variants.
    - Pyre status: `flash_attn_ext_f32_f16_decode` uses `double` for slope,
      score, local sums, and reductions.
    - Impact: medium/high if FA is hot.
    - Savings: cheaper arithmetic and lower register pressure.
    - Verify: implement float-only variant gated by accuracy tests.

27. **Streamed softmax to avoid full logits array**
    - Vulkan pattern: tiled flash attention keeps running `M` and `L` rather
      than materializing all logits for the full KV when possible.
    - Pyre status: stores `logits[1024]`, limiting KV and doing multiple passes.
    - Impact: high for longer KV and correctness scale-out.
    - Savings: less shared memory, no KV<=1024 limitation, fewer passes.
    - Verify: port online softmax structure from Vulkan scalar FA.

28. **Coopmat/MFMA FA for prefill**
    - Vulkan pattern: coopmat FA exists but scalar is preferred at `N==1`.
    - Pyre status: no MFMA/WMMA emission in representative kernels.
    - Impact: very high for prefill/batched, low for current decode.
    - Verify: defer until decode matvec and scalar FA are closer.

## Graph Fusion and Scheduling

29. **Graph reordering to preserve fusions**
    - Vulkan pattern: graph optimizer avoids pulling nodes out of supported
      fusion patterns and tries to keep real/view nodes ordered for fusion.
    - Pyre status: graph compute scans linearly and claims narrow patterns.
    - Impact: medium/high if candidate fusions are missed due to ordering.
    - Savings: more fusions fire without adding kernels.
    - Verify: compare node order around missed fusion opportunities.

30. **Multi-add fusion beyond two ADDs**
    - Vulkan pattern: `multi_add.comp` supports fusing multiple ADDs and optional
      RMS partial behavior.
    - Pyre status: `ADD_ADD` exists; broader multi-add not obvious.
    - Impact: medium.
    - Savings: collapse chains of elementwise adds.
    - Verify: provider trace for repeated ADD sequences.

31. **RMS_NORM partials path**
    - Vulkan pattern: has `rms_norm_partials` and `rms_norm_mul_partials` for
      add-RMS partial accumulation cases.
    - Pyre status: has RMS and RMS+MUL, plus ADD+RMS+MUL broadcast fusion.
    - Impact: low/medium unless those partial cases appear.
    - Verify: trace `ADD_RMS_NORM_MUL` and any fallback RMS partial-like graph.

32. **RMS+ROPE+SET_ROWS shape parity**
    - Vulkan pattern: supports `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` when shared
      memory and rope mode constraints pass.
    - Pyre status: has `pyre_rms_norm_mul_rope_set_rows_f32_f16`.
    - Impact: already captured; remaining work is tuning.
    - Savings: reduce shared memory, pow/cos/sin overhead, vectorize stores.
    - Verify: microbench and compare against Vulkan for Qwen rope dimensions.

33. **ROPE mode coverage**
    - Vulkan pattern: normal, NeoX, MROPE in some fusions; vision has separate
      handling.
    - Pyre status: current fused code handles a subset.
    - Impact: low for Qwen if current modes are covered; medium for model
      generality.
    - Verify: audit `mode` conditions in support predicates versus Vulkan.

34. **TopK MoE sigmoid/bias mode parity**
    - Vulkan pattern: `TOPK_MOE_SIGMOID_NORM_BIAS` handles sigmoid gates with
      bias, norm, and optional scale.
    - Pyre status: TopK kernel has softmax-like behavior and clamp/norm; mode
      parity needs verification.
    - Impact: high if Qwen uses sigmoid/bias gating pattern.
    - Savings: collapse softmax/sigmoid, argsort/get_rows/scale chains.
    - Verify: trace which `TOPK_MOE` pattern Pyre claims and compare to Vulkan.

35. **TopK MoE softmax-weight late path**
    - Vulkan pattern: supports early-softmax and late-softmax modes plus
      `GATING_FUNC_SOFTMAX_WEIGHT`.
    - Pyre status: not obviously mode-complete.
    - Impact: medium, model-dependent.
    - Verify: add pattern-specific trace strings to Pyre TopK fusion.

36. **Output scale/bias in TopK MoE**
    - Vulkan pattern: TopK shader applies `output_scale` and `output_bias`.
    - Pyre status: constants include `scale`, clamp, norm; output affine parity
      should be checked.
    - Impact: low/medium.
    - Verify: search provider trace for SCALE/GET_ROWS after TopK.

37. **Fusion overlap safety checks**
    - Vulkan pattern: disables fusion if outputs overlap sources unsafely; has a
      special single-row TopK case.
    - Pyre status: fusions are narrower, but overlap rules may block future
      broader fusions.
    - Impact: correctness enabler.
    - Verify: port overlap checks before adding broader graph fusions.

38. **Batch submit / command batching policy**
    - Vulkan pattern: batches nodes and submits after about 100 nodes or a
      matmul-byte threshold to overlap CPU command generation and GPU execution.
    - Pyre status: runtime path is different; dispatch overhead was previously a
      major concern.
    - Impact: high if runtime still serializes each dispatch heavily.
    - Verify: Tracy dispatch timeline before/after new runtime.

## Matrix-Matrix and Prefill

39. **Coopmat/MFMA tiled matmul**
    - Vulkan pattern: `mul_mm.comp`, `mul_mm_cm2.comp`, and `mul_mmq.comp` cover
      scalar, coopmat, aligned, f16acc/f32acc, quantized, and q8_1 variants.
    - Pyre status: representative HIP kernels did not emit MFMA/WMMA; current
      work is decode matvec-heavy.
    - Impact: very high for prompt processing and batched decode.
    - Savings: orders of magnitude local versus scalar matvec-like approaches.
    - Verify: write one pure-HIP MFMA tile for F16/BF16 matrix-matrix, then Q4_K
      dequant tile.

40. **Aligned matrix-matrix load variants**
    - Vulkan pattern: generator emits `_aligned` matmul variants with larger
      load vectors.
    - Pyre status: not represented.
    - Impact: high for prefill, low for current decode.
    - Verify: inspect prompt/prefill provider trace after decode path stabilizes.

41. **Split-K matmul reduce**
    - Vulkan pattern: `mul_mat_split_k_reduce.comp` for large matmul reductions.
    - Pyre status: absent.
    - Impact: high for large prefill or low-workgroup shapes.
    - Verify: add only after baseline MFMA matmul exists.

42. **Matmul ID subgroup matrix-matrix**
    - Vulkan pattern: generates `matmul_id_subgroup` variants.
    - Pyre status: expert work is matvec-focused.
    - Impact: high for batched MoE/prefill.
    - Verify: revisit when batch/token count grows beyond decode `N=1`.

## Memory Movement and Layout

43. **Prealloc temp buffer parity**
    - Vulkan pattern: `prealloc_x`, `prealloc_y`, `prealloc_split_k`, and
      sync-needed flags avoid per-op allocation and redundant conversions.
    - Pyre status: scratch exists for q8_1; copy eliminations have removed many
      materializations.
    - Impact: medium/high if Tracy still shows allocation/sync per dispatch.
    - Verify: trace buffer allocation counts during decode.

44. **Copy-to-contiguous avoidance for more layouts**
    - Vulkan pattern: specialized non-contig matvec and copy pipelines.
    - Pyre status: current spike removed observed copy-like claims, but support
      predicates are still stricter than Vulkan in many cases.
    - Impact: medium for model portability; low if Qwen trace stays clean.
    - Verify: keep provider trace gates in CI-like benchmark scripts.

45. **Copy transpose variants**
    - Vulkan pattern: `copy_transpose.comp` has 16/32-bit variants.
    - Pyre status: no direct transpose-copy catalog entry.
    - Impact: low for current Qwen decode unless fallback appears.
    - Verify: provider trace for transpose/copy nodes on other models.

46. **Quantized set_rows parity**
    - Vulkan pattern: copy-to-quant/from-quant and set-rows style paths are broad.
    - Pyre status: has F32 to F16/F32/Q4_0/Q8_0 set_rows.
    - Impact: low/medium.
    - Verify: trace KV-cache write types; add missing quant formats only when
      observed.

47. **Skip zero-length work before dispatch**
    - Vulkan pattern: empty/view/metadata handling avoids many no-op kernels.
    - Pyre status: spike log says zero-length GET_ROWS/CPY are now skipped.
    - Impact: already captured; keep as regression guard.
    - Verify: provider trace should remain `CONT=0`, `CPY=0`, `CONCAT=0` for
      current no-FA Qwen decode.

## Elementwise, Norm, and Softmax Micro-Optimizations

48. **Subgroup reductions in softmax**
    - Vulkan pattern: softmax variants use unrolled dimensions and shared memory;
      some paths use subgroup arithmetic.
    - Pyre status: softmax kernel is simpler.
    - Impact: low/medium for no-FA; higher if standalone softmax remains hot.
    - Verify: provider trace count and timing for `SOFT_MAX`.

49. **Large softmax decomposition**
    - Vulkan pattern: `soft_max_large1/2/3` handles large rows in staged passes.
    - Pyre status: no equivalent large-softmax family.
    - Impact: low for current decode if FA/topk fusions cover hot softmax.
    - Verify: fallback or poor timing on long rows.

50. **RMS norm unroll buckets**
    - Vulkan pattern: `rms_norm.comp` instantiates several `num_iters` buckets
      to help unroll by row width.
    - Pyre status: RMS kernels use a fixed loop over 512-thread blocks.
    - Impact: medium for RMS-heavy segments.
    - Verify: specialize common `ncols` values and microbench.

51. **ROPE trigonometric precompute/recurrence**
    - Vulkan pattern: rope shader code is mature and has several mode-specific
      paths.
    - Pyre status: fused RMS+ROPE uses `powf`, `cosf`, and `sinf` per pair.
    - Impact: medium if fused rope timing is visible.
    - Savings: reduce transcendental overhead.
    - Verify: use recurrence or precomputed frequency tensor when available.

52. **Fast math audit**
    - Vulkan pattern: GLSL compiler `-O` is used for most non-coopmat/non-bf16
      shaders; some shaders avoid `spirv-opt` due to known issues.
    - Pyre status: HIP compiler flags may not be tuned per kernel.
    - Impact: medium.
    - Verify: inspect generated ISA and try fast-math-like flags only on kernels
      with acceptable numerical slack.

## ISA and Compiler Verification

53. **Automated disassembly check**
    - Vulkan pattern: shader backend eventually lowers through the same AMDGPU
      compiler stack, but SPIR-V source structure differs.
    - Pyre status: manual disassembly showed no MFMA/WMMA/dot in sampled kernels.
    - Impact: high as guardrail.
    - Verify: script `llvm-objdump` checks for `v_mfma`, `v_wmma`, `v_dot`,
      barriers, spills, and VGPR count for every hot kernel build.

54. **VGPR/spill tracking per kernel revision**
    - Vulkan pattern: many shader variants are partly about register/shared
      memory balance.
    - Pyre status: impacts are being logged, but compiler resource counters are
      not part of the loop.
    - Impact: medium/high.
    - Verify: record SGPR/VGPR/spill counts in the experiment log per kernel.

55. **Explicit launch-bounds audit**
    - Vulkan pattern: local size and subgroup size are specialization constants.
    - Pyre status: HIP kernels rely on dispatch workgroup size and compiler
      inference.
    - Impact: medium.
    - Verify: try `__launch_bounds__` on hot kernels and compare occupancy.

56. **Subgroup size control**
    - Vulkan pattern: can force subgroup size in selected pipelines.
    - Pyre status: uses `warpSize` and AMD wave assumptions but dispatch config
      sets subgroup size to 0.
    - Impact: medium.
    - Verify: set subgroup size where Pyre runtime supports it and compare wave32
      versus wave64 behavior on RDNA3.

## Priority Stack

The likely best near-term order is:

1. Add trace/timing diff tools so misses are visible.
2. Port DMMV `NUM_COLS` plus subgroup/no-shmem variants for Q4_K/Q5_K/Q6_K.
3. Add missing `MUL_MAT(+ADD)` and `MUL_MAT_ID+ADD_ID` fusions if the Qwen trace
   shows them.
4. Retest Q8_1 under the incoming runtime; if still slow, try fused
   quantize+matvec for one Q4_K shape.
5. Tune `MUL_MAT_ID` expert kernels specifically; Qwen MoE makes these hot.
6. Only then spend serious effort on scalar FA split-K/tuning unless traces show
   FA already dominates.
7. Treat MFMA matrix-matrix as a separate prefill/batched epic, not the first
   decode catch-up step.
