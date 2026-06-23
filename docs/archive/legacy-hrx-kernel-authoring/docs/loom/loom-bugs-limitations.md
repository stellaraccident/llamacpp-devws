# Loom Bugs And Limitations Ledger

This is the durable issue ledger for HRX2/Loom bringup. Keep entries concrete:
date, affected source, exact symptom, impact, workaround, and current owner.
General design requests and nice-to-have author feedback belong in
`docs/loom/loom-author-feedback.md`; items here are things that can invalidate
kernel implementation, measurement, or backend integration work if forgotten.

Update this file whenever an issue changes what can safely be accepted into the
catalog, invalidates benchmark evidence, blocks a kernel source from compiling,
or requires a runtime workaround. Do not bury those items in the general author
feedback log.

## 2026-06-16: Q6_K p64 cols64 needs prior-matched HIP staging; Loom MMQ64x32 is not the reference optimum

- **Area:** HRX2 Q6_K prompt matmul schedule selection.
- **Affected sources:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q6_k_f32.loom`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp`,
  and `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q6_k_f32.json`.
- **Observed case:** Llama 3.2 3B Q4_K_M p64 Q6_K prompt rows
  `k3072 rows1024 cols64` and `k8192 rows3072 cols64`.
- **Symptom:** The existing Loom `q8_1_x4_mmq64x32` route remained much slower
  than the Vulkan target and was also slower than a prior-matched HIP bridge on
  the large Q6 cols64 row. A naive Q6 wave32 port with `BK_STEP=1` regressed the
  small-row case:

  ```text
  k3072 rows1024 cols64: Loom 178.834 us, wave32 BK1 206.558 us
  k8192 rows3072 cols64: Loom 561.460 us, wave32 BK4 336.034 us
  ```

- **Impact:** Future Q6 prompt work should not use the current Loom
  `mmq64x32` route as the presumed schedule optimum for cols64 p64 shapes. The
  working prior is the packed Q8_1-x4 MMQ schedule with wave32 BM64/BN64 and
  `BK_STEP=4` staging. This is a schedule-authoring limitation, not a confirmed
  Loom compiler defect.
- **Workaround/current policy:** Keep the accepted narrow HIP bridge for the
  two p64 cols64 Q6 rows. If a Loom version is authored later, it must spell the
  same staging, wave/tile ownership, packed load widths, and dot form before
  comparing performance.
- **Evidence:**
  - Focused op gate:
    `cache/hrx2/phase2a/q6-w32-vkm64x64-opgate-20260616-140920/`.
  - Reduced basket:
    `cache/hrx2/phase2a/q6-w32-vkm64x64-reduced-20260616-141320/`.
- **Owner:** HRX2 llama.cpp route work.

## 2026-06-16: High-level Q4_K BK_STEP4 spelling compiled and passed but regressed

- **Area:** HRX2/Loom prompt MMQ authoring methodology.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`.
- **Observed case:** Temporary rewrite of
  `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static` to stage four 32-wide K blocks
  per barrier while preserving the existing 64x32 output tile and one-row by
  eight-column per-thread ownership.
- **Symptom:** The corrected route compiled, selected, and passed focused
  backend-op correctness, but regressed every Q4_K hot perf row:

  ```text
  k3072 rows3072 cols512:  1146.188 us -> 1343.000 us
  k8192 rows3072 cols512:  3422.688 us -> 3853.750 us
  k3072 rows8192 cols512:  3253.188 us -> 3677.563 us
  k3072 rows16384 cols512: 6454.625 us -> 7194.688 us
  ```

- **Impact:** Do not assume that copying a single Vulkan schedule knob into the
  current high-level Loom route is enough. The accepted Q4_K route already
  emits the correct `v_dot4_i32_iu8` dot form and has no spills; the remaining
  gap is probably tile/work-ownership/lifetime structure, not just barrier
  count.
- **Workaround/current policy:** Keep the accepted MMQ64x32 route. The next
  Q4_K rewrite should be a new prior-matched route that changes work ownership
  and tile shape together, using Vulkan/CUDA MMQ as the reference, rather than
  another local BK_STEP-only mutation.
- **Evidence:**
  - Correct-but-regressed gate:
    `cache/hrx2/phase2a/q4-bkstep4-fixed-opgate-20260616-083353/`.
  - Saved rejected patch:
    `cache/hrx2/phase2a/q4-bkstep4-fixed-opgate-20260616-083353/saved/q4-bkstep4-correct-but-regressed.patch`.
  - Post-revert baseline gate:
    `cache/hrx2/phase2a/q4-post-bkstep-revert-opgate-20260616-083505/`.
- **Owner:** HRX2 llama.cpp route work. This is not currently a Loom compiler
  bug; it is schedule feedback for future agents.

## 2026-06-15: llama-cli can hit unsupported HRX2 stride domains that llama-bench basket avoids

- **Area:** HRX2 integration validation / route shape domains.
- **Affected sources:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/set_rows_f32.loom`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_f16_f32_batched.loom`,
  and their route/domain metadata.
- **Observed case:** Llama 3.2 3B Q4_K_M `llama-cli` smoke with prompt
  `Q: What is 2+2? A:` and HRX2 full offload. The first attempt used CLI
  defaults; the second constrained the run to the basket-style
  `-b 512 -ub 512 -fa 0`.
- **Symptom:** The CLI path failed before it could validate the new
  `ADD -> RMS_NORM -> MUL` fusion:

  ```text
  CONFIG/INVALID: config 'hrx2.shape.set_rows.ne1' value 134217728 violates constraint 'range'
  HRX2: SET_ROWS provider is not available for nc=1 nr=2048 dst=f16
  CONFIG/INVALID: config 'hrx2.shape.mul_mat_f16.src0_stride_ne3' value 268435456 violates constraint 'range'
  HRX2: MUL_MAT F16/F32 provider is not available for k=128 rows=256 cols=46 dst_ne2=24 dst_ne3=1
  ```

- **Impact:** A failed `llama-cli` smoke is not necessarily evidence against a
  touched kernel/fusion. It may be a broader route-domain problem exposed by
  the interactive/speculative compatibility path. For kernel/fusion admission,
  continue to use focused backend op gates first, then `llama-bench` graph
  traces on the agreed basket shapes, and only use CLI as an additional smoke
  after its route domains are covered.
- **Workaround:** For current Phase 2a route work, validate with
  `tools/hrx2_phase2a_benchmark.py` and trace route selection. Do not claim
  CLI correctness for a change unless the CLI smoke itself completes.
- **Evidence:**
  - Default CLI attempt:
    `cache/hrx2/phase2a/add-rms-norm-mul-decode-only-20260615/cli-smoke/`.
  - Basket-constrained CLI attempt:
    `cache/hrx2/phase2a/add-rms-norm-mul-decode-only-20260615/cli-smoke-b512/`.
  - Repeat during FA0 attention-chain validation:
    `cache/hrx2/phase2a/fa0-fusion-correctness-20260616-111156/`. The failure
    again involved `hrx2.shape.set_rows.ne1` and
    `hrx2.shape.mul_mat_f16.src0_stride_ne3` violating configured ranges in a
    short-prompt CLI path; bounded p64/p512 `llama-bench` fa0 prefill traces
    had no provider failures.
- **Owner:** HRX2 route-domain follow-up in llama.cpp. The likely fix is to
  tighten route applicability before JIT or broaden the affected Loom config
  ranges only if the kernels are actually valid for those large byte strides.

## 2026-06-15: HRX2 benchmark traces report compile-report size but do not persist the report

- **Area:** HRX2 JIT/tuning evidence capture.
- **Affected sources:** HRX2 JIT/runtime trace path in
  `sources/llama.cpp/ggml/src/ggml-hrx2/` and the Phase 2a benchmark flow.
- **Observed case:** Q5_K decode dot16 route
  `mul_mat_q5_k_f32_dot16_k256_32768_r1_262144_c1_wg32`.
- **Symptom:** The provider trace records that JIT compilation succeeded and
  includes `compile_report_bytes`, for example:

  ```text
  "route_id":"mul_mat_q5_k_f32_dot16_k256_32768_r1_262144_c1_wg32",
  "compile_report_bytes":4320
  ```

  but the benchmark artifact directory contains only `hrx2.jsonl`,
  `sched.jsonl`, `llama-bench.json`, and `stderr.txt`. The structured compile
  report JSON is not persisted alongside the run.
- **Impact:** Future agents cannot inspect register pressure, spills, lowering
  choices, or resource summaries from accepted model-level evidence without
  reproducing the JIT compile or rebuilding a separate standalone tuning case.
  This weakens the intended compile-report-guided optimization workflow.
- **Workaround:** For now, use standalone Loom tuning artifacts when resource
  evidence must be preserved, and treat model traces as route/tok/s evidence
  only. Do not claim spill/register facts from a model run unless the structured
  compile report was explicitly captured.
- **Evidence:**
  - Q5_K op gate:
    `cache/hrx2/phase2a/q5-dot16-20260615-154659/op-gate/`.
  - Q5_K basket comparison:
    `cache/hrx2/phase2a/q5-dot16-decode-20260615-154740/`.
- **Owner:** HRX2 llama.cpp tooling follow-up. Add an opt-in artifact directory
  for JIT compile reports keyed by route/cache key, or extend the benchmark
  tool to ask the JIT layer to dump reports per provider compile.

## 2026-06-15: Q8_1 quantizer needed non-obvious AMDGPU spelling workarounds

- **Area:** Loom AMDGPU lowering / HRX2 kernel authoring.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/quantize_q8_1.loom`.
- **Observed case:** Target-generic F32 to `block_q8_1` Loom kernel for the
  Q4_K prompt RHS backplane.
- **Symptoms and workarounds:**
  - `scalar.roundf` / `scalar.roundevenf` did not have an AMDGPU target-low
    contract in this flow. The accepted source spells ggml's round-away path
    explicitly with sign-dependent `+/-0.5` followed by `scalar.fptosi`.
  - f32 compare-to-zero lowered to invalid inline operand forms such as
    `v_cmp_ogt_f32.src1_inline {rhs = 0}`. The source now bitcasts the f32
    value to i32 and checks the bit pattern for zero where that is semantically
    enough.
  - explicit integer clamps selected `v_max_i32` / `v_min_i32` forms with SGPR
    constants where the chosen descriptor wanted a VGPR RHS. The clamps were
    removed because ggml's CPU Q8_1 reference does not clamp and the math
    constrains the result to the valid range.
  - AMDGPU u32 address constraints required explicit `index.assume` facts on
    launch ids and address intermediates. A fully dynamic spelling failed
    lowering because target-low could not prove address widths.
- **Impact:** Loom is WYSIWYG here: do not expect the compiler to recover
  missing vectorization, rounding, address-range, or operand-form intent. Future
  agents should spell the desired load width, dot width, address bounds, and
  rounding behavior directly, then inspect the compile report/ISA before
  accepting the route.
- **Evidence:**
  - Standalone configured compile:
    `/tmp/quantize_q8_1.hsaco` from
    `/tmp/hrx2-artifacts-q8/artifacts/quantize_q8_1_f32.loombc`.
  - Focused opt-in route gate:
    `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/serial2/`.
- **Owner:** Workarounds live in llama.cpp; report missing lowering coverage to
  the Loom author if these forms should lower directly.

## 2026-06-15: Direct Q4_K x Q8_1 prompt route validated plumbing but regressed throughput

- **Area:** HRX2 route acceptance and packed-matmul design.
- **Affected sources:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/quantize_q8_1.loom`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`, and
  `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`.
- **Observed case:** Llama 3.2 3B Q4_K_M, p64 prompt, opt-in
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`.
- **Symptom:** The route was correctness-clean at the focused backend-op level
  and selected in a real graph, but it lost to the accepted F32-RHS cols4 path:

  ```text
  default cols4:       77.000 tok/s, 566 dispatches
  opt-in direct q8_1:  69.187 tok/s, 678 dispatches
  ```

- **Root cause class:** The direct Q8_1 variant adds a quantize dispatch for
  each prompt matmul, but its one-row/one-column direct-dot schedule does not
  reuse the packed RHS enough to pay for conversion. It proves scratch-backed
  RHS conversion and route selection, not the final Q4_K prompt algorithm.
- **Rule:** Do not promote Q8_1 conversion unless the paired matmul is a real
  packed/MMQ schedule that reuses the RHS tile across enough rows/columns. For
  this family, the next accepted candidate should be compared against the
  default F32-RHS cols4 path and should reduce both repeated Q4_K dequant work
  and prompt dispatch/runtime overhead.
- **Evidence:**
  - Backend op gate:
    `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/serial2/`.
  - Default model smoke:
    `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/llama32-3b-q4-p64-default/`.
  - Opt-in model smoke:
    `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/llama32-3b-q4-p64-optin/`.
- **Owner:** HRX2 Q4_K packed/MMQ follow-up in llama.cpp. Keep the direct route
  opt-in as a backplane test until the real packed route replaces it.

## 2026-06-15: Q4_K x Q8_1 cols4 prompt route is mixed and must remain opt-in

- **Area:** HRX2 route acceptance and prompt packed-matmul design.
- **Affected sources:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom` and
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q4_k_f32.json`.
- **Observed case:** Opt-in route
  `mul_mat_q4_k_q8_1_f32_cols4_k256_32768_c4_512_wg256`, enabled with
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`.
- **Symptom:** Focused `MUL_MAT` backend-op validation passed and the route
  selected correctly, but full-model prefill A/B was mixed:

  ```text
  Phi p64:       62.115 -> 59.141 tok/s, 0.952x
  Phi p512:      65.356 -> 65.125 tok/s, 0.996x
  Llama 3.2 p64: 73.664 -> 71.503 tok/s, 0.971x
  Llama 3.2 p512:79.973 -> 82.557 tok/s, 1.032x
  Llama 3.1 p64: 31.103 -> 31.855 tok/s, 1.024x
  Llama 3.1 p512:31.936 -> 33.512 tok/s, 1.049x
  ```

- **Impact:** Replacing F32 cols4 with Q8_1 cols4 is not a defaultable Q4_K
  prompt solution. It adds one quantize dispatch per Q4_K prompt matmul and
  only partially amortizes that cost. Larger Llama prompt shapes can benefit,
  but smaller prompts and Phi do not.
- **Rule:** Keep the route opt-in as a backplane probe. Do not default it or
  treat it as the Phase 2a Q4_K prompt bulk lift. The next accepted candidate
  must be a true MMQ-style tile that reuses Q8_1 RHS packing and Q4_K dequant
  across a larger row/column tile.
- **Evidence:**
  - Backend op gate:
    `cache/hrx2/phase2a/q4-q8-cols4-20260615-160225/op-gate/`.
  - Full prefill default:
    `cache/hrx2/phase2a/q4-q8-cols4-full-default-20260615-1605/`.
  - Full prefill opt-in:
    `cache/hrx2/phase2a/q4-q8-cols4-full-optin-20260615-1605/`.
- **Owner:** HRX2 Q4_K MMQ follow-up in llama.cpp.

## 2026-06-16: Q4_K x4 MMQ64x8 correctness improved, but schedule is still too slow

- **Area:** HRX2 Q4_K prompt matmul schedule selection.
- **Affected sources:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom` and
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q4_k_f32.json`.
- **Observed case:** Diagnostic route
  `mul_mat_q4_k_q8_1_x4_mmq64x8_k256_32768_r1_32768_c8_512_wg64`, tested
  after the latest hrx-system AMDGPU memory/codegen fixes.
- **Symptom change:** A focused exported Llama p512 Q4_K op gate passed:

  ```text
  cache/hrx2/phase2a/q4k-mmq64x8-after-addtidfix-opgate-20260616-004334/
  ```

  but the reduced two-model run regressed badly:

  ```text
  llama32 p512: 94.9 -> 72.0 tok/s
  llama32 p64:  83.2 -> 22.3 tok/s
  phi4 p512:    85.0 -> 67.0 tok/s
  phi4 p64:     74.4 -> 27.8 tok/s
  ```

- **Impact:** Do not keep treating this Q4 MMQ variant as blocked purely on a
  compiler correctness bug. At least one spelling is now correctness-clean, but
  it is not a viable schedule. The next Q4 MMQ attempt needs a new
  prior-matched schedule, not local promotion of the MMQ64x8 diagnostic route.
- **Workaround/current policy:** Q4_K direct Q8_1 prompt is now default-on with
  rollback `GGML_HRX2_DISABLE_Q4_K_Q8_1_PROMPT=1`; the x4 MMQ path remains
  behind `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`.
- **Owner:** HRX2 Q4_K MMQ follow-up in llama.cpp. Use the HRX1/Vulkan/CUDA
  prior matrix before authoring a new route.

## 2026-06-16: Q4_K x4 MMQ32x32 looped spelling exposed integer CFG/select lowering gaps

- **Area:** Loom AMDGPU lowering / HRX2 prompt-MMQ authoring.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`.
- **Observed case:** Rewriting
  `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static` from the old eight-way cloned Q4
  group topology into the Q6-shaped loop over Q8_1 sub-blocks.
- **Symptoms:**
  - Standalone `loom-compile` failed in `source-to-low` with:

    ```text
    AMDGPU branch argument materializer selected for an unsupported type
    ```

    when an `scf.if` yielded decoded Q4 scale/min values as `i32`.
  - Replacing the branch with `scf.select` on `i32` also failed because this
    route did not have a target-low contract for that select form.
  - The llama.cpp JIT trace only reported `provider_compile` failure; the
    actionable diagnostic came from standalone `loom-compile`.
- **Impact:** A valid schedule can appear blocked or silently fall back unless
  agents inspect provider traces and reproduce failing providers with standalone
  compile reports. Do not accept a focused CSV pass without proving the intended
  route compiled and dispatched.
- **Workaround:** Keep integer control values out of AMDGPU branch arguments in
  this path. The accepted Q4 source yields already-scaled `f32` scale/min
  values from the branch and uses explicit shift/mask nibble selection instead
  of `scf.select<i32>`.
- **Evidence:**
  - Failing looped compile attempts:
    `cache/hrx2/phase2a/q4-x4-mmq-kb-loop-branchless-compile-20260616-045348/`
    and
    `cache/hrx2/phase2a/q4-x4-mmq-kb-loop-branchless-compile2-20260616-045407/`.
  - Accepted compile report:
    `cache/hrx2/phase2a/q4-x4-mmq-kb-loop-report-20260616-045914/report.json`.
  - Accepted default/rollback focused gates:
    `cache/hrx2/phase2a/q4-x4-mmq-default-on-gates-20260616-050208/`.
- **Owner:** Workaround lives in llama.cpp. Report the missing integer branch
  argument and `scf.select<i32>` lowering coverage to the Loom author if these
  forms should be legal on the AMDGPU path.

## 2026-06-15: HRX2 scheduler event copy rotation changed wait counts but not throughput

- **Area:** llama.cpp scheduler/runtime measurement.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`
  event prototype and experimental llama.cpp scheduler-copy wiring.
- **Observed case:** Llama 3.2 3B Q4_K, HRX2 full offload, decode p1/n64.
- **Symptom:** Enabling HRX2 events and scheduler input-copy rotation reduced
  traced full stream synchronizes from 595 to 205, but added 390 event
  synchronizes and left tok/s unchanged at about 42 tok/s.
- **Root cause:** The remaining split inputs are tiny dynamic control tensors
  (`inp_tokens`, ROPE positions, KV row ids, attention mask, output ids).
  The normal decode loop still needs graph completion for logits/sampling, and
  dispatch count plus device work dominate. Reclassifying waits from full
  stream synchronizes to event synchronizes does not remove the dependency.
- **Impact:** Do not accept runtime changes based only on lower
  `hrx_stream_synchronize` counts. Compare tok/s and trace all wait classes.
  For Phase 2a, prioritize dispatch elimination, fusions, and hero kernel
  throughput over standalone scheduler-copy rotation.
- **Evidence:**
  - Baseline: `cache/hrx2/phase2a/rope-probe-fix-compare-20260615/`.
  - Event-only no-op:
    `cache/hrx2/phase2a/hrx2-events-compare-20260615-rerun/`.
  - Pipeline-parallel event attempt:
    `cache/hrx2/phase2a/hrx2-sched-events-compare-20260615/`.
  - Decoupled copy-rotation attempt:
    `cache/hrx2/phase2a/hrx2-sched-copy-rotation-compare-20260615/`.
- **Owner:** Rejected for Phase 2a as a standalone optimization. Backend event
  support can still be useful if a future multi-stream/multi-device path proves
  an actual throughput improvement.

## 2026-06-15: Q4_K rows2 decode route reduced workgroups but regressed throughput

- **Area:** HRX2 route acceptance and decode optimization methodology.
- **Affected sources:** Rejected local candidate in
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom` and
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q4_k_f32.json`.
- **Observed case:** Llama 3.2 3B Q4_K_M decode, `p1 n64`, Q4_K direct
  `cols=1` route. The candidate computed two output rows per workgroup for
  even row counts.
- **Symptom:** Focused `test-backend-ops` correctness passed and the route
  selected in the model trace, but model throughput regressed badly:

  ```text
  baseline direct Q4_K decode: 41.370 tok/s, 36790 dispatches
  rows2 decode candidate:       9.124 tok/s, 36790 dispatches
  ```

- **Impact:** Workgroup-count reductions are not reliable decode evidence.
  This candidate left graph dispatch count unchanged and apparently worsened
  kernel schedule/occupancy enough to dominate any per-workgroup arithmetic
  savings.
- **Rule:** Do not accept rows-per-workgroup or multi-output decode variants
  without same-binary model throughput evidence, route trace evidence, and a
  plausible data-reuse or dispatch-elimination mechanism. Small-model decode
  is runtime-backpressure sensitive, but at about 30B more runtime slop can be
  hidden by heavier device work; both regimes still need explicit measurement.
- **Evidence:**
  - Backend op gate:
    `cache/hrx2/phase2a/q4k-rows2-decode-20260615/op-gate/`.
  - Regressing model smoke:
    `cache/hrx2/phase2a/q4k-rows2-decode-20260615/llama32-3b-decode/default/`.
- **Owner:** Rejected and removed from llama.cpp. Future decode work should
  target proven HRX1-style direct schedules, dispatch-eliminating fusions, or
  dynamic-input removal rather than this rows2 Q4_K shape.

## 2026-06-15: ROPE weight-buffer probe used n_dims=0 and misplaced shared factors on CPU

- **Area:** llama.cpp model tensor placement / HRX2 scheduler interop.
- **Affected source:** `sources/llama.cpp/src/llama-model-loader.cpp`,
  `weight_buft_supported(..., GGML_OP_ROPE)`.
- **Observed case:** Llama 3.2 3B Q4_K_M, HRX2 full offload, decode p1/n64.
- **Symptom:** Even with all compute nodes on HRX2 and input embeddings
  offloaded, scheduler traces showed `HRX20#rope_freqs.weight#0` as a repeated
  split input for ROPE: 672 copies in the 64-token decode smoke.
- **Root cause:** The loader chooses a tensor's buffer type by building a
  representative op and calling backend `supports_op`. The ROPE representative
  called `ggml_rope_ext` with `n_dims=0`. HRX2 rejected that invalid synthetic
  op, so the shared `rope_freqs.weight` tensor fell back to CPU even though
  the actual decode ROPE shapes were supported by catalog routes.
- **Impact:** CPU compute fallback can be zero while recurrent split-input
  copies still create decode runtime backpressure. Do not infer clean runtime
  placement from zero CPU `sched_node` counts alone.
- **Fix:** The representative ROPE op now derives `n_dims` from the factor
  tensor (`w->ne[0] * 2`) and uses a one-token shape. The fix removes
  `rope_freqs.weight` from the HRX2 split-input list without broadening HRX2
  `supports_op`.
- **Evidence:**
  - Manual override proof:
    `cache/hrx2/phase2a/rope-freqs-override-20260615/`.
  - Accepted normal run:
    `cache/hrx2/phase2a/rope-probe-fix-compare-20260615/`.
- **Owner:** Resolved in llama.cpp; remaining decode split inputs are true
  dynamic leaves (`inp_pos`, KV row ids, attention mask) and need separate
  runtime/fusion work.

## 2026-06-15: llama-cli smoke hit existing F16/F32 attention route config range rejection

- **Area:** HRX2 integration smoke coverage.
- **Affected source:** Existing `mul_mat_f16_f32_batched` route/config domain,
  not the Q4_K SWIGLU fusion work.
- **Observed case:** Deterministic `llama-cli` smoke on
  `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` after Q4_K SWIGLU fusion landed.
- **Symptom:** The run failed in decode before producing useful correctness
  evidence:

  ```text
  CONFIG/INVALID: config 'hrx2.shape.mul_mat_f16.src0_stride_ne3' value 268435456 violates constraint 'range'
  HRX2: MUL_MAT F16/F32 provider is not available for k=128 rows=256 cols=2 dst_ne2=24 dst_ne3=1
  ```

- **Impact:** Do not use this failed CLI smoke as evidence against the Q4_K
  SWIGLU fusion routes. It exposes a separate route-domain bug for a small
  attention-shaped F16/F32 matmul path used by the interactive CLI graph.
- **Workaround:** Use the standard `llama-bench` slice and focused backend-op
  gates for the Q4_K SWIGLU fusion evidence until the F16/F32 route domain is
  fixed. Re-run deterministic CLI after that fix.
- **Evidence:** `cache/hrx2/phase2a/q4k-swiglu-cli-smoke-20260615-113551/`.
- **Owner:** HRX2 attention/F16 route-domain follow-up.

## 2026-06-15: Fused Q4_K SWIGLU route metadata initially had wrong parameter_count

- **Area:** HRX2 catalog ABI metadata.
- **Affected source:** Initial candidate route
  `mul_mat_q4_k_swiglu_f32_direct_k256_32768_c1_512_wg256`.
- **Observed case:** Llama 3.2 Q4 p64 graph fusion attempted to JIT the split
  FFN route for `k=3072`, `rows=8192`, `cols=64`.
- **Symptom:** The graph recognizer fired, but the provider was unavailable.
  Standalone configured `loom-compile` showed the kernel manifest ABI was
  `binding_count=4`, `parameter_count=4`, `constant_byte_length=0`; the route
  JSON had `parameter_count=3`.
- **Impact:** Runtime ABI validation rejects providers when route metadata is
  stale even if Loom source and standalone compilation are correct.
- **Fix:** Corrected the route ABI to `parameter_count=4`. The fused route then
  dispatched successfully and was accepted.
- **Evidence:**
  - Failing provider trace:
    `cache/hrx2/phase2a/q4k-swiglu-smoke-20260615-112134/`.
  - Standalone manifest:
    `cache/hrx2/phase2a/q4k-swiglu-standalone-manifest-20260615-112352/`.
  - Passing fused smoke:
    `cache/hrx2/phase2a/q4k-swiglu-smoke-20260615-112416/`.
- **Owner:** Resolved for this route; keep as a warning to always compare
  route ABI against Loom manifest facts before accepting a new provider.

## 2026-06-15: p512 masked SOFT_MAX route passed correctness but regressed model throughput

- **Area:** HRX2 Phase 2a route admission and performance acceptance.
- **Affected source:** Candidate route additions in
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/routes/soft_max_f32.json`.
- **Observed case:** Phase 2a p512 prefill for
  `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` with `llama-bench -p 512 -n 0
  -b 512 -ub 512 -fa 0`.
- **Symptom:** Adding an ncols=512 masked softmax route made focused
  `test-backend-ops` pass for the exact benchmark graph rows, but model
  throughput regressed:

  ```text
  route-coverage-smoke-20260615-103535:          42.733 tok/s, CPU fallback 576
  route-coverage-softmax512-smoke-20260615-103731: 36.870 tok/s, CPU fallback 384
  ```

- **Impact:** Correctness and CPU-fallback elimination are not sufficient
  acceptance criteria for tiny or scalar p512 routes. The standalone masked
  SOFT_MAX kernel can be slower than leaving the op on CPU in this regime.
- **Fix/Workaround:** Rejected and removed the p512 masked softmax route.
  Keep p512 softmax CPU-owned until there is a faster standalone implementation
  or an attention fusion that proves fused-vs-unfused improvement on the same
  target and shape bucket.
- **Evidence:**
  - Passing focused route test before rejection:
    `cache/hrx2/phase2a/p512-c512-op-export-20260615-103625/hrx2-after-route-coverage-test.csv`.
  - Regressing model smoke:
    `cache/hrx2/phase2a/route-coverage-softmax512-smoke-20260615-103731/`.
  - Accepted post-rejection p512 comparison:
    `cache/hrx2/phase2a/route-coverage-p512-comparison-20260615-103954/`.
- **Owner:** HRX2 route admission and future attention/softmax kernel work.

## 2026-06-13: Scheduler reducer counted split events as compute fallback nodes

- **Area:** HRX2 Phase 1 coverage accounting.
- **Affected source:** `tools/hrx2_reduce_sched_trace.py`.
- **Observed case:** Route-slice-44 summaries included `sched_split_begin`
  compute events in the same reduction as `sched_node` graph nodes.
- **Symptom:** CPU-owned graph islands could inflate top fallback counts with
  split-level ROPE rows, making the report look like more individual graph
  nodes were CPU fallbacks than the node trace alone proved.
- **Impact:** Route-slice summaries produced before this fix are still useful
  for ranking, but their `node_count`, class totals, and top fallback counts
  should be treated as mixed scheduler-event accounting rather than pure graph
  node coverage.
- **Fix:** The reducer now filters to `event == "sched_node"` before
  classifying coverage. Split-level diagnostics should be reduced separately
  if needed.
- **Evidence:** During the route-slice-45 Mistral ROPE audit, direct
  `sched_node` inspection and HRX2 dispatch traces proved the missing route was
  NORMAL no-frequency ROPE, while mixed-event reductions also included CPU
  split events.
- **Owner:** HRX2 coverage tooling.

## 2026-06-13: Normal-frequency ROPE h32/p64 strict tolerance was sensitive to theta spelling

- **Area:** HRX2 ROPE route admission and Loom AMDGPU transcendental numeric
  parity.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rope_neox_f32_freq.loom`,
  route `rope_normal_f32_freq_n128_d128_h32_t1_64_wg256` in
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json`.
- **Observed case:** Llama 3.1 normal-mode ROPE with F32 frequency factors,
  `mode=0`, `ncols=128`, `n_dims=128`, `nheads=32`, `ntokens=64`.
- **Symptom:** The h32 route compiles and dispatches, but focused
  `test-backend-ops` replay of the p64 graph row failed strict CPU-reference
  tolerance:

  ```text
  [ROPE] ERR = 0.000007262 > 0.000000100
  ```

- **Impact:** Numeric spelling matters for ROPE acceptance. Do not treat two
  algebraically equivalent transcendental expressions as interchangeable unless
  the focused ggml CPU-reference gate covers the target token bucket.
- **Fix:** Route slice 48 rewrote the NORMAL frequency-source root to match the
  CPU recurrence: compute one `theta_scale` and multiply `theta` forward once
  per pair before dividing by the frequency-factor buffer. The route now admits
  `ntokens=1..64`.
- **Evidence:**
  - Failing focused p1/p16/p64 replay:
    `cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/focused-20260613-015635`.
  - Passing split-domain focused replay:
    `cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/focused-split-t1-t16-20260613-015955`.
  - Passing recurrence focused replay:
    `cache/hrx2/phase1_0/route-slice-48-rope-normal-h32-p64/focused-final-20260613-044740`.
  - Passing final basket with zero unexplained compute fallbacks:
    `cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836`.
- **Owner:** Resolved in HRX2 route slice 48; keep this entry as a regression
  warning for future ROPE rewrites.

## 2026-06-13: Multi-family route lookup discarded earlier ROPE providers

- **Area:** HRX2 runtime catalog loading and route admission.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`.
- **Observed case:** Qwen3 30B A3B UD-Q4 exported ROPE rows with no frequency
  factors, `mode=2` / NEOX, `ncols=128`, `nheads=4 or 32`, and `ntokens=1 or
  64`.
- **Symptom:** The structural C++ support predicate accepted the rows, but
  focused `test-backend-ops` reported the rows unsupported. Debug route
  tracing showed `device_context->rope_routes` contained only
  `rope_normal_f32_freq_n128_d128_h8_24_t1_64_wg256`; all NEOX providers were
  absent, so route selection could not find a matching binding/domain.
- **Root cause:** `ggml_backend_hrx2_catalog_find_routes` clears the output
  vector before adding matches. Device initialization called it once for
  `rope_neox_f32` and then again for `rope_f32` using the same
  `device_context->rope_routes` vector. The second call discarded the earlier
  NEOX routes.
- **Impact:** Multi-family ggml ops can appear correctly authored and
  cataloged but be unavailable at runtime if route loading reuses a vector
  across clearing lookup calls. Focused route validation catches this; source
  and catalog validation alone do not.
- **Fix:** Load all ROPE providers with one op-wide catalog lookup
  (`family=null`, `op=ROPE`) so the vector contains every ROPE family and the
  existing route-selection metadata chooses the applicable provider.
- **Evidence:**
  - Failing/exported rows:
    `cache/hrx2/phase1_0/route-slice-37-rope-mode-export-current/qwen3_ud_q4_rope_ops.txt`.
  - Passing focused CPU-reference replay:
    `cache/hrx2/phase1_0/route-slice-37-rope-loader-fix-current`.
  - Passing Qwen3 decode/narrow/prefill64 model smoke:
    `cache/hrx2/phase1_0/route-slice-37-rope-loader-fix-current/qwen3-smoke`.
- **Owner:** HRX2 runtime catalog loading.

## 2026-06-12: AMDGPU target-low lacks scalar.powf contract

- **Area:** Loom AMDGPU math lowering for HRX2 production kernels.
- **Affected source:** Initial
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rope_neox_f32.loom`.
- **Observed case:** Phase 1 route slice 29 attempted to compile no-`src2`
  NEOX F32 ROPE with:

  ```text
  %theta_pow = scalar.powf<afn> %freq_base, %exponent : f32
  ```

- **Symptom:** Focused `test-backend-ops` route validation failed provider
  JIT compilation with:

  ```text
  TARGET/001: target 'amdgpu-rdna3' export 'hrx2_rope_neox_f32'
  config 'amdgpu.rdna3.core' has no target-low contract for 'scalar.powf'
  ```

- **Impact:** Agents should not use `scalar.powf<afn>` in accepted AMDGPU
  HRX2 Loom sources without first proving the current branch lowers it. A
  route can pass source formatting/build-bytecode steps and still fail at JIT
  compile time for the target.
- **Workaround:** Rewrite `pow(freq_base, exponent)` as
  `exp(log(freq_base) * exponent)` using `scalar.logf<afn>` and
  `scalar.expf<afn>`. Loom's AMDGPU math legalization tests cover those
  approximate forms, and the corrected ROPE source passed focused ggml
  CPU-reference validation for all accepted route rows.
- **Evidence:**
  - Failing focused run:
    `cache/hrx2/phase1_0/route-slice-29-rope-focused-20260612-214203`.
  - Passing focused run after rewrite:
    `cache/hrx2/phase1_0/route-slice-29-rope-focused-20260612-214501`.
- **Owner:** Loom AMDGPU math lowering; HRX2 keeps the workaround in source
  until `scalar.powf` lowering is available and validated.

## 2026-06-12: Catalog validator and C++ loader had stale pointwise ABI assumptions

- **Area:** HRX2 catalog validation and embedded catalog loading.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/tools/validate_hrx2_catalog.py`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2-catalog.cpp`.
- **Observed case:** Route slice 26 added pointwise layout config sources:
  `shape.pointwise.src0_row_stride` and `shape.pointwise.src1_ncols`.
- **Symptom:** The Python validator rejected the new binding sources. After
  that was fixed, the C++ catalog loader printed:

  ```text
  HRX2: invalid route entry in catalog
  ```

  because it still required `parameter_count > 0`, even though existing
  buffer-only kernels can legitimately have no scalar launch parameters.
- **Impact:** A source/catalog change could pass build artifact generation but
  fail at runtime catalog load, or force agents to avoid adding necessary
  layout-specialization axes.
- **Fix:** The validator now accepts the new pointwise binding sources and
  treats ABI `parameter_count` as present/nonnegative. The C++ loader now
  requires nonempty route/source/root/export and positive binding count, but
  allows zero scalar parameters.
- **Evidence:** Route slice 26 focused validation passed with the intended
  pointwise route IDs selected:
  `cache/hrx2/phase1_0/route-slice-26-20260612-201843/test-focused-rerun`.
- **Owner:** HRX2 catalog tooling.

## 2026-06-12: Llama 3.1 8B Q8_0 basket smoke aborted before q8 dispatch

- **Area:** HRX2 model-level runtime validation and q8_0 route admission.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`,
  q8_0 matmul route family in `sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json`
- **Observed case:** Coverage-basket smoke for
  `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` in decode, narrow, and prefill64
  regimes under:
  `cache/hrx2/phase1_0/basket-smoke-fixed-20260612-190821`.
- **Symptom:** All three regimes abort prompt decode with:

  ```text
  test_prompt: failed to decode prompt batch, res = -3
  main: error: failed to run prompt
  ```

  The HRX2 route trace for the decode run shows successful compile and
  dispatch of:

  ```text
  rms_norm_f32_n4096_r1_vector_vw4_wg512
  mul_f32_generic_wg256
  ```

  The scheduler trace then assigns the first q8_0 `MUL_MAT` (`Qcur-0`,
  `src0=q8_0`, shape `4096x1x1x1`) to HRX2, but there is no corresponding
  HRX2 dispatch trace before the abort.
- **Root cause:** The q8_0 `MUL_MAT` support predicate still included
  allocated-pointer non-overlap guards. Scheduler probing can run before the
  tensors have concrete data pointers, where those guards returned true. The
  same predicate was then re-run during allocated graph execution, where the
  pointer checks rejected the split graph before q8 dispatch. HRX1 had already
  learned not to use these alias guards in route support predicates.
- **Impact:** Model-level validation could assign q8_0 matmul to HRX2 and then
  fail before dispatch. This invalidated Q8_0 basket coverage even though the
  standalone q8 route still passed focused `test-backend-ops` validation.
- **Fix:** Removed the q8_0 non-overlap guards from
  `ggml_backend_hrx2_supports_mul_mat_q8_0`, matching the scheduler-visible
  shape/type/layout contract. Kept low-noise q8 dispatch-failure trace events
  for buffer binding, shape, and stream-dispatch failures.
- **Evidence:**
  - Focused q8 route tests passed before and after the fix:
    `cache/hrx2/phase1_0/q8-testfile-20260612-192030` and
    `cache/hrx2/phase1_0/q8-testfile-after-overlap-fix-20260612-192724`.
  - Scheduler split trace isolated the failure to HRX2 split execution before
    q8 dispatch:
    `cache/hrx2/phase1_0/q8-split-trace-20260612-192500`.
  - Temporary HRX2 node trace showed the abort at `MUL_MAT Qcur-0` before the
    q8 dispatch path:
    `cache/hrx2/phase1_0/q8-hrx2-node-trace-20260612-192615`.
  - Model-level Q8_0 decode passed after the fix:
    `cache/hrx2/phase1_0/q8-model-after-overlap-fix-20260612-192735`.
  - Clean production Q8_0 decode, narrow, and prefill64 smokes passed after
    cleanup:
    `cache/hrx2/phase1_0/q8-three-regimes-after-overlap-fix-20260612-192957`.
  - Full 11-model coverage basket passed 33/33 after the fix:
    `cache/hrx2/phase1_0/basket-smoke-after-q8-overlap-fix-20260612-193159`.
- **Owner:** HRX2 runtime/q8_0 route admission.

## 2026-06-12: Broad phase0 RMS_NORM route overclaimed unsupported shapes

- **Area:** HRX2 catalog route admission.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json`
- **Observed case:** Coverage-basket smoke on Qwen3 30B A3B selected
  `rms_norm_f32_contiguous` for `ncols=2048` and `nrows=1/16/64`.
- **Symptom:** The scheduler assigned `RMS_NORM` to HRX2, then provider JIT
  compilation failed and prompt decode aborted with `res = -3`. HRX2 trace:

  ```text
  provider_compile status=failed route_id=rms_norm_f32_contiguous
  cache_key=...ncols=2048|nrows=1...
  provider_unavailable op=RMS_NORM route_id=rms_norm_f32_contiguous
  ```

- **Root cause:** A broad phase0 route used the dynamic ABI root
  `@hrx2_rms_norm_f32` with a shape domain of `ncols=1..65536` and
  `nrows=1..1048576`, but the route was not validated across that domain. It
  could claim new model shapes before we had compile/correctness evidence.
- **Impact:** Broad basket smokes can fail before producing fallback evidence.
  This also proves route admission must be exact-shape or bucket-evidence
  driven; "generic" routes are dangerous unless the whole advertised domain is
  tested or the JIT compile path can fail closed during support probing.
- **Fix/workaround:** Removed `rms_norm_f32_contiguous` from the production
  catalog. Keep only measured exact RMS_NORM routes until a tuned generic or
  bucketed route passes compile and focused ggml CPU-reference validation for
  every admitted bucket.
- **Owner:** HRX2 catalog tooling/admission policy.

## 2026-06-12: Model smoke can hide a supported standalone route inside CPU islands

- **Area:** HRX2 route acceptance and llama.cpp scheduler evidence.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json`
- **Observed case:** `SWIGLU` f32 for Phi-4 prompt/decode shapes
  `ncols=8192`, `nrows=1` and `nrows=16`.
- **Symptom:** The HRX2 support predicate admits the model-shape `SWIGLU`
  node and scheduler trace lists `HRX20` in `supported_by`, but the node is
  still assigned to CPU because the adjacent quantized matmul nodes remain CPU
  fallbacks. HRX2 dispatch traces therefore show no selected `SWIGLU` route in
  the full model smoke even though focused `test-backend-ops` coverage proves a
  smaller `SWIGLU` route is compilable and correct.
- **Impact:** Full-model smoke traces are not sufficient route-selection
  evidence for standalone ops embedded in CPU-owned graph islands. A route can
  be correct and supported yet remain unselected until neighboring prerequisite
  ops are offloaded.
- **Workaround:** For every newly admitted route, collect focused backend-op or
  scratch-graph evidence for the intended exact shape, in addition to the model
  smoke. Count the model smoke as route evidence only when the HRX2 dispatch
  trace actually contains the route.
- **Owner:** HRX2 acceptance tooling.

## 2026-06-12: SET_ROWS f32->f16 global store address lowering fails

- **Area:** Loom target lowering for HRX2 production kernel authoring.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/set_rows_f32.loom`
- **Current status:** Resolved on `sources/hrx-system` main at
  `7189ff975 Improve Loom AMDGPU codegen coverage and diagnostics (#95)`.
  Keep this entry as a regression warning.
- **Repro shape:** `test-backend-ops test -b HRX20 -o SET_ROWS -p 'type=f16,type_idx=i64,ne=\[256,5,1,3\],nr23=\[1,1\],r=1,v=0' --output csv`
- **Symptom:** JIT provider compilation rejects `@hrx2_set_rows_f32_f16` with:

  ```text
  TARGET/003: target 'amdgpu-rdna3' export 'hrx2_set_rows_f32_f16'
  config 'amdgpu.rdna3.core' rejected 'index.shli' address-width 'u32':
  constraint 'amdgpu.address.u32' is not satisfied
  ```

- **What was tried:** The kernel was rewritten to use typed element views
  instead of byte-offset addressing, with explicit `index.assume` ranges before
  all address products/adds. The f16 store was also rewritten through an `xi16`
  destination view using `scalar.fptrunc f32 to f16` followed by
  `scalar.bitcast f16 to i16`.
- **Previous impact:** HRX2 could not rely on the Loom f32->f16 unfused
  SET_ROWS provider, which is a scheduler prerequisite for model-level traces
  with KV updates.
- **Previous workaround:** HRX2 had a deliberately slow host-mediated SET_ROWS
  fallback so model-level evidence collection could continue. The fallback was
  used by default; `GGML_HRX2_ENABLE_SET_ROWS_LOOM=1` exercised the current
  Loom SET_ROWS providers and reproduced the lowering failure. This was not an
  optimized-kernel substitute and was not counted as done-done kernel coverage.
- **Resolution evidence:** On 2026-06-15, exact exported Phi p512 SET_ROWS rows
  passed with the Loom provider selected by default:
  `cache/hrx2/phase2a/set-rows-default-20260615-110043/test.csv`. The route
  trace contains `set_rows_f32_f16_generic` for all four rows.
- **llama.cpp follow-up:** HRX2 now defaults to the Loom SET_ROWS provider when
  a route is available and keeps `GGML_HRX2_DISABLE_SET_ROWS_LOOM=1` as a
  regression/repro escape hatch for the old host fallback.
- **Owner:** Resolved; HRX2 keeps focused SET_ROWS replay as a regression gate.

## 2026-06-12: HRX2 support predicates diverged from allocated graph execution

- **Area:** llama.cpp HRX2 backend integration, scheduler contract.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`
- **Symptom:** The scheduler assigned `RMS_NORM` and `SET_ROWS` nodes to HRX2
  because support probing happened before concrete graph allocation, but
  `graph_compute` rejected the allocated split graph with:

  ```text
  HRX2: unsupported RMS_NORM shape/type/layout
  HRX2: unsupported SET_ROWS shape/type/layout
  ```

- **Root cause:** HRX2 re-ran support predicates during execution that included
  address/alias assumptions not present during scheduler probing. HRX1 had
  already learned to avoid these false non-overlap guards for these ops.
- **Impact:** Model-level prompt decode failed before useful kernel coverage
  evidence could be collected, even for shapes that the scheduler correctly
  considered HRX2-supported.
- **Fix/workaround:** HRX2 now aligns `RMS_NORM` and `SET_ROWS` support checks
  with HRX1-style shape/type/layout predicates and emits detailed tensor
  summaries when a graph-compute validation failure still occurs.
- **Owner:** HRX2 backend.

## 2026-06-12: I64 SET_ROWS index load lacks a clean high-level path to index

- **Area:** Loom source spelling for ggml `SET_ROWS` with `I64` row indices.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/set_rows_f32.loom`
- **Symptom:** Loading an `i64` row index and converting it to `index` was not
  available in the current target path used by HRX2.
- **Impact:** Directly spelling ggml's `I64` index semantics in Loom is blocked.
- **Workaround:** The current Loom source reads the low `i32` lane from the
  `I64` index buffer via an `xi32` view and casts that to `index`. This matches
  current llama.cpp row-index ranges in practice, but it is a narrowing
  assumption and should be removed once Loom supports the direct path.
- **Owner:** Loom lowering/API gap; HRX2 must keep the assumption documented if
  it ships before the direct path exists.

## 2026-06-12: Catalog validation does not verify route configs against source declarations

- **Area:** HRX2 catalog tooling and Loom JIT integration.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/pointwise_f32.loom`
- **Symptom:** A `SCALE` route included a binding for
  `@hrx2.shape.pointwise.src1_row_stride`, but the `SCALE` provider source does
  not declare or use that config. Catalog validation passed, but runtime JIT
  failed with:

  ```text
  CONFIG/INVALID: unknown config 'hrx2.shape.pointwise.src1_row_stride'
  ```

- **Impact:** A catalog can pass static validation and still fail only when a
  route is first JIT-compiled. This is easy to miss in shape-driven routes that
  are not covered by focused unit tests.
- **Workaround/fix:** The bad `SCALE` binding was removed and f32 `SCALE`
  correctness now passes through `test-backend-ops`. Keep focused
  `test-backend-ops` coverage or model-smoke coverage for every newly added
  route until validation checks route bindings against the selected source
  module.
- **Owner:** HRX2 catalog tooling.

## 2026-06-12: Pointwise f32 coverage is contiguous/simple-row-broadcast only

- **Area:** HRX2 phase 1 unfused pointwise coverage.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/pointwise_f32.loom`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`
- **Symptom:** `ADD` and `MUL` f32 CPU-reference tests pass for supported
  contiguous cases, including same-shape RHS and simple row-broadcast RHS, but
  generated ggml test cases with higher-rank broadcast/permuted RHS layouts
  remain unsupported by the backend support predicate.
- **Impact:** The generic pointwise route is correct for the current Phi-4
  smoke shapes and gives unit-test coverage for common contiguous cases, but it
  is not complete general ggml pointwise coverage.
- **Workaround:** Keep the support predicate narrow. Add broader broadcast/view
  variants only when a traced model shape proves they are needed, and validate
  each admitted layout with `test-backend-ops`.
- **Owner:** HRX2 backend/kernel catalog.

## 2026-06-12: Generic flat GET_ROWS f32 Loom candidate fails target lowering and correctness

- **Area:** Loom target lowering and HRX2 unfused coverage.
- **Affected attempted source:** `cache/hrx2/phase1_0/rejected-get-rows/get_rows_f32.loom`
- **Repro command:** `test-backend-ops test -b HRX20 -o GET_ROWS -p 'type=f32' --output csv`
- **Primary symptom:** Many `ncols=256` f32 GET_ROWS cases failed provider
  compilation with:

  ```text
  TARGET/003: target 'amdgpu-rdna3' export 'hrx2_get_rows_f32'
  config 'amdgpu.rdna3.core' rejected 'index.shli' address-width 'u32'
  constraint 'amdgpu.address.u32' is not satisfied
  ```

- **Secondary symptom:** Some `ncols=256` cases that reached execution produced
  numeric mismatches with maximum absolute error around `2.0`, so this is not
  only a target-proof problem.
- **Update 2026-06-13:** Route slice 46 accepted a narrower compact-dense F32
  `GET_ROWS` family. The passing source uses a 2D dense view spelling
  `src0[row_index, col] -> dst[row, col]` plus an explicit `src0_nrows`
  config, and the runtime predicate admits only dense F32 source rows, I32
  indices, and dense F32 destination rows. Focused graph-op validation and the
  full basket passed at:

  ```text
  cache/hrx2/phase1_0/route-slice-46-get-rows-f32/focused-existing-exports-20260613-032516
  cache/hrx2/phase1_0/route-slice-46-get-rows-f32/focused-phi4-3072-20260613-032546
  cache/hrx2/phase1_0/basket-smoke-route-slice-46-20260613-032627
  ```

- **Update 2026-06-13:** Route slice 47 accepted separate quantized embedding
  `GET_ROWS` families for `q4_K`, `q5_K`, `q6_K`, and `q8_0` sources. Focused
  graph-op validation passed 41/41 exact rows and the full basket passed at:

  ```text
  cache/hrx2/phase1_0/route-slice-47-get-rows-quant/focused-final-20260613-042127
  cache/hrx2/phase1_0/basket-smoke-route-slice-47-offload-hook-20260613-042340
  ```

- **Remaining limitation:** The original flat/generic GET_ROWS spelling is
  still rejected and should not be generalized back into production coverage
  without a focused Loom reproduction. Accepted F32 and quantized embedding
  routes are narrow, traced-layout route families rather than generic ggml
  `GET_ROWS`.
- **Additional route-admission finding:** Unused Loom `config.decl` entries can
  be pruned from the linked source. If the catalog still binds such a pruned
  key, HRX2 JIT fails with a diagnostic like:

  ```text
  CONFIG/INVALID: unknown config 'hrx2.shape.get_rows.src0_row_stride'
  ```

  Keep catalog specialization bindings limited to configs consumed by the
  selected root, or deliberately consume the config in source.
- **Impact:** HRX2 may advertise the accepted compact-dense F32 and quantized
  embedding route domains, but broader strided/dynamic/generic GET_ROWS remains
  unsupported.
- **Workaround:** Keep the support predicate narrow and use exact graph-op rows
  for validation. Do not use generic `test-backend-ops -o GET_ROWS` as the sole
  admission gate for this family.
- **Owner:** Loom lowering investigation plus HRX2 route admission.

## 2026-06-13: Supported quantized embedding GET_ROWS stayed CPU-assigned without offload_op

- **Area:** HRX2 scheduler placement and CPU/host-seeded embedding gathers.
- **Affected source:** `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`.
- **Observed case:** Route slice 47 after adding quantized `GET_ROWS` routes
  for Qwen, Llama, Mistral, Gemma, and Phi hidden-width buckets.
- **Symptom:** Focused CPU-reference validation accepted the quantized routes,
  but full-basket scheduler traces still assigned 396 quantized embedding
  `GET_ROWS` graph nodes to CPU. The trace showed the nodes as
  `supported_by=[HRX20,CPU]`, so this was placement, not route support.
- **Root cause:** llama.cpp's scheduler uses backend `offload_op` to move
  CPU/host-weight seeded ops to a higher-priority backend in this path. HRX2
  had `offload_op = nullptr`, so CPU-seeded token embedding gathers stayed in
  the CPU island even when `supports_op` was true.
- **Fix:** Added a conservative HRX2 device `offload_op` hook that delegates to
  `supports_op` only for `GGML_OP_GET_ROWS`.
- **Impact:** Future route families can pass focused validation yet fail to
  move model-level coverage if their source placement relies on scheduler
  offload hooks. Check `cpu_assigned_but_hrx_supported` in the reduced basket
  summary before claiming model-level coverage.
- **Evidence:**

  ```text
  cache/hrx2/phase1_0/basket-smoke-route-slice-47-loader-20260613-040651
  cache/hrx2/phase1_0/route-slice-47-get-rows-quant/offload-hook-qwen-q4-p1-20260613-042034
  cache/hrx2/phase1_0/basket-smoke-route-slice-47-offload-hook-20260613-042340
  ```

- **Owner:** HRX2 backend scheduler integration.

## 2026-06-12: MoE downstream support routes remain CPU-assigned until top-k/gather is offloaded

- **Area:** HRX2 scheduler placement and Phase 1 MoE coverage.
- **Evidence:** Full basket route slice 27:
  `cache/hrx2/phase1_0/basket-smoke-route-slice-27-20260612-204250`.
- **Symptom:** Focused `test-backend-ops` validates new HRX2 routes for
  `SUM_ROWS`, `CLAMP`, and `DIV`, but the full basket still reports those ops
  as CPU compute fallbacks. Scheduler traces show:

  ```text
  ARGSORT supported_by=CPU
  GET_ROWS supported_by=CPU
  SUM_ROWS/CLAMP/DIV supported_by=HRX20,CPU but assigned CPU
  ```

- **Update from route slice 28:** Narrow MoE `ARGSORT` and MoE weight
  `GET_ROWS` routes are now accepted in focused validation and show
  `supported_by=HRX20,CPU` in the full basket, but the whole MoE island is
  still CPU-assigned because `MUL_MAT_ID` remains CPU-only:

  ```text
  ARGSORT supported_by=HRX20,CPU assigned CPU
  GET_ROWS supported_by=HRX20,CPU assigned CPU
  SUM_ROWS/CLAMP/DIV supported_by=HRX20,CPU assigned CPU
  MUL_MAT_ID supported_by=CPU assigned CPU
  ```

- **Impact:** Downstream MoE support kernels and existing GLU routes cannot
  reduce model-level fallback counts while `MUL_MAT_ID` gate/up/down paths
  keep the island on CPU. This is not a validation failure for the accepted
  support routes, but it is a Phase 1 coverage blocker.
- **Workaround:** Prioritize `MUL_MAT_ID` and related quantized matmul
  coverage before spending more effort on model-level impact for the MoE
  support chain.
- **Owner:** HRX2 backend route coverage, especially `MUL_MAT_ID`.

## 2026-06-12: Generic SOFT_MAX test coverage misses HRX2 attention route domain

- **Area:** HRX2 focused validation workflow.
- **Observed case:** Route slice 30 accepts attention softmax rows with
  `ncols=256`, F32 masks, and broadcast head/token shapes from the model
  basket.
- **Symptom:** `test-backend-ops -o SOFT_MAX` generates useful generic
  softmax cases, but not the exact `ncols=256` route domain. A support run over
  generated SOFT_MAX cases therefore reports no supported rows even though the
  model-basket scheduler reports the accepted rows as `supported_by=HRX20,CPU`.
- **Evidence:**

  ```text
  cache/hrx2/phase1_0/route-slice-30-softmax-focused-20260612-current/test-focused-masked-generator
  cache/hrx2/phase1_0/route-slice-30-softmax-focused-20260612-current/test-focused-masked-manual
  ```

- **Impact:** Agents must not use generic `-o SOFT_MAX` coverage alone to
  decide whether Phase 1 attention softmax routes are unsupported.
- **Workaround:** Use exact graph-op rows from model export when available, or
  construct a focused test file with `GGML_OP_SOFT_MAX` rows matching the route
  domain and run `test-backend-ops --test-file` against HRX2. Route slice 30
  used manual rows for `256x1x{24,32,40}x1` masked attention and exported rows
  for `128x{1,16,64}` unmasked MoE probabilities.
- **Owner:** HRX2 validation workflow. A future production helper should
  generate exact graph-op test files from route metadata and observed
  scheduler rows.

## 2026-06-12: Generic MUL_MAT tests miss p021 F16 attention matvec layouts

- **Area:** HRX2 focused validation workflow.
- **Observed case:** Route slice 31 accepts F16/F32 attention `MUL_MAT` rows
  where `src0` has p021-style byte strides and grouped-head broadcast, for
  example `src0 ne=[128,256,4,1] nb=[2,1024,256,262144]`.
- **Symptom:** Generic `test-backend-ops -o MUL_MAT` coverage does not provide
  the exact strided/batched attention layouts needed to validate this family.
  A generic pass is therefore not sufficient evidence that the route is
  covered or uncovered.
- **Evidence:**

  ```text
  cache/hrx2/phase1_0/route-slice-31-f16-attn-focused-current/mul_mat_f16_f32_attention_ops.txt
  cache/hrx2/phase1_0/route-slice-31-f16-attn-focused-current
  ```

- **Impact:** Agents must derive exact graph-op rows from model scheduler
  traces or exported graph-op files for attention matvec validation.
- **Workaround:** Route slice 31 generated focused `GGML_OP_MUL_MAT` rows
  directly from the basket scheduler metadata and replayed them with
  `test-backend-ops --test-file` against ggml CPU reference.
- **Update 2026-06-13:** The same validation trap appeared for the F32/F32
  MoE-logits matmul. `test-backend-ops -o MUL_MAT -p
  type_a=f32,type_b=f32,m=...,n=...,k=...` exited successfully with only a CSV
  header and no HRX2 trace. Route slice 42 used exact exported graph-op rows
  instead.
- **Owner:** HRX2 validation workflow. A reusable route-domain-to-test-file
  helper should be added before the broad kernel sweep grows much larger.

## 2026-06-12: ARGSORT bitonic/LDS candidates compile but GPU-fault under HRX2 raw dispatch

- **Area:** Loom source/runtime interaction for workgroup-memory ARGSORT.
- **Affected attempted sources:** Phase 1 route slice 28 scratch candidates
  under `cache/hrx2/phase1_0/route-slice-28-current`.
- **Observed case:** MoE `ARGSORT` for `f32 -> i32`, DESC, `ncols=128`, rows
  `1`, `16`, and `64`.
- **Symptoms:**
  - A dynamic bitonic source using `index.div`/`index.shrui` loop control hit
    an AMDGPU target diagnostic around address-width proof.
  - Static/unrolled bitonic sources using workgroup scratch compiled, but
    focused `test-backend-ops` runs GPU-faulted on dispatch with a page-not-
    present/supervisor-privilege memory access fault.
- **Evidence:**

  ```text
  cache/hrx2/phase1_0/route-slice-28-current/test-focused-vector-scratch
  cache/hrx2/phase1_0/route-slice-28-current/test-focused-static-argsort
  ```

- **Impact:** The natural one-workgroup bitonic/LDS prior cannot currently be
  accepted for HRX2 production. Agents should not assume "compiled" means this
  source is safe to dispatch.
- **Workaround:** Route slice 28 uses a rank-count no-LDS `ARGSORT` fallback
  for phase-one coverage. It is correct for the traced `ncols=128` MoE support
  shape and has clean compile reports, but it is not a final general sort/top-k
  algorithm.
- **Owner:** Loom lowering/runtime investigation, with HRX2 keeping the
  rank-count route narrow until the LDS fault is explained.

## 2026-06-13: Shape-only ROPE tests can validate the wrong pairing mode

- **Area:** HRX2 focused validation workflow.
- **Observed case:** Route slice 33 initially added a frequency-factor ROPE
  route for NeoX pairing and validated it with synthetic rows. The Llama 3.2
  Q4_K graph uses the same visible tensor shapes and frequency-factor source,
  but raw op params show `GGML_ROPE_TYPE_NORMAL`.
- **Symptom:** The NeoX route passed focused CPU-reference tests for its own
  synthetic rows but did not move the model scheduler because real model ROPE
  rows had `mode=0` and were therefore correctly rejected.
- **Evidence:**

  ```text
  cache/hrx2/phase1_0/route-slice-33-rope-freq-export-current/llama32-q4k-rope-ops.txt
  cache/hrx2/phase1_0/route-slice-33-rope-normal-focused-after-family-cleanup
  ```

- **Impact:** Agents must not author or validate mode-sensitive kernels from
  shape/type evidence alone. For ROPE, normal and NeoX have identical visible
  shapes but different pair ownership.
- **Workaround:** Export exact graph-op rows and replay them with
  `test-backend-ops --test-file`. Preserve mode in catalog metadata
  (`supports.mode`) and make route matching filter on the raw op mode.
- **Owner:** HRX2 validation workflow. A future graph-row reducer should
  surface semantic op params in fallback summaries for mode-sensitive ops.

## 2026-06-13: AMDGPU path cannot lower `scf.select` in Q6 unpacking

- **Area:** Loom AMDGPU target lowering for scalar/select control forms.
- **Affected source:** Phase 1 route slice 35
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q6_k_f32.loom`.
- **Observed case:** The first Q6 source used `scf.if` to choose the high or
  low ql nibble based on `part >= 2`. Rewriting it to `scf.select` avoided the
  original internal error but hit a missing target-low contract.
- **Symptoms:**

  ```text
  INTERNAL; AMDGPU branch argument materializer selected for an unsupported type
  ```

  followed after the `scf.select` rewrite by:

  ```text
  TARGET/001: target 'amdgpu-rdna3' export 'hrx2_mul_mat_q6_k_f32_static'
  config 'amdgpu.rdna3.core' has no target-low contract for 'scf.select'
  ```

- **Impact:** Tiny integer unpack choices inside production kernels should not
  be spelled as `scf.if` returning an integer or as `scf.select` until this
  target path is proven fixed. A source can pass `loom-link` and fail only
  during HRX2 JIT target lowering.
- **Workaround:** Spell Q6 low-nibble selection as direct packed-bit
  arithmetic:

  ```text
  ((ql_byte >> ((part / 2) * 4)) & 0xf)
  ```

  The branchless source passed focused ggml CPU-reference validation for 10
  real-trace Q6 rows and emitted clean compile reports.
- **Evidence:**
  - Failing `scf.if` run:
    `cache/hrx2/phase1_0/route-slice-35-q6-focused-current/test.csv`.
  - Failing `scf.select` run:
    `cache/hrx2/phase1_0/route-slice-35-q6-focused-current/test-select.csv`.
  - Passing branchless run:
    `cache/hrx2/phase1_0/route-slice-35-q6-focused-current/test-branchless.csv`.
- **Owner:** Loom AMDGPU lowering; HRX2 keeps the branchless source spelling.

## 2026-06-13: AMDGPU path rejects scalar `andi` on predicate values

- **Area:** Loom AMDGPU target lowering for boolean predicate composition.
- **Affected source:** Phase 1 route slice 38
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_id_q4_k_f32.loom`.
- **Observed case:** The first Q4_K `MUL_MAT_ID` source combined
  `expert >= 0` and `expert < nexperts` with scalar boolean `andi` before a
  guarded store.
- **Symptom:**

  ```text
  TARGET/003 ... rejected 'scalar.andi' field 'lhs' ...
  constraint 'low_register_class.amdgpu.scc' is not satisfied
  ```

- **Impact:** Predicate composition that looks natural at Loom high level can
  fail only during target lowering. This is easy to rediscover in boundary
  checks for ids, tails, and optional stores.
- **Workaround:** Spell combined predicates as nested `scf.if` regions. Slice
  38 uses nested expert-valid and lane-zero guards and passes focused
  CPU-reference validation.
- **Owner:** Loom AMDGPU lowering. HRX2 authors should preserve the nested
  control-flow spelling until the boolean lowering path is fixed.

## 2026-06-13: llama.cpp loader probes 512-token weight compatibility

- **Area:** HRX2/llama.cpp integration, not Loom codegen.
- **Observed case:** Q4_K `MUL_MAT_ID` passed focused `test-backend-ops`
  validation but Qwen3 model graphs still placed Q4_K matmuls on CPU.
- **Root cause:** `llama_model_loader::select_weight_buft` checks whether a
  weight can live in a backend buffer by building a synthetic representative
  op. For `MUL_MAT` and `MUL_MAT_ID`, the synthetic RHS uses 512 columns or
  tokens. Routes capped at 64 therefore make the backend reject its own
  load-bearing weights, even when all real runtime shapes are within the
  smaller domain.
- **Evidence:** Before widening, verbose Qwen3 UD-Q4 load showed
  `CPU_Mapped model buffer size = 16757.27 MiB`,
  `CPU_REPACK model buffer size = 14429.25 MiB`, and
  `HRX20 model buffer size = 0.80 MiB`. After widening Q4_K route domains to
  512, the same load showed `HRX20 model buffer size = 14430.05 MiB` and q4
  `MUL_MAT_ID` dispatches on HRX20.
- **Impact:** Focused op tests are insufficient for phase 1. Every
  load-bearing route must also pass a model-load placement smoke that proves
  tensors are allocated in HRX2 buffers.
- **Workaround:** Include the loader's 512-token probe in route domains for
  quantized matmul-family weights, or change the backend/model-loader contract
  deliberately. Do not narrow production route metadata below the loader probe
  just because the current benchmark basket only uses smaller prefill buckets.
- **Owner:** HRX2 integration workflow. Future catalog authoring should treat
  load selector compatibility as a required acceptance gate.

## 2026-06-13: AMDGPU path lacks `scalar.tanhf` target-low contract

- **Area:** Loom AMDGPU target math lowering.
- **Affected source:** Phase 1 route slice 44 GEGLU prototype in
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/swiglu_f32.loom`.
- **Observed case:** A direct GEGLU spelling using
  `scalar.geluf<tanh> %x : f32` passed bytecode generation but failed during
  HRX2 JIT to AMDGPU HSACO.
- **Symptom:**

  ```text
  TARGET/001: target 'amdgpu-rdna3' export 'hrx2_geglu_f32_split'
  config 'amdgpu.rdna3.core' has no target-low contract for 'scalar.tanhf'
  in '@hrx2_geglu_f32_split'
  ```

- **Impact:** The high-level GELU op is available, but current AMDGPU lowering
  can expose an unsupported `scalar.tanhf` after math legalization. This is
  easy to rediscover for GEGLU and any tanh-family activation.
- **Workaround:** Spell tanh-GELU through the logistic identity:

  ```text
  gelu_tanh(x) = x * logistic(2 * sqrt(2/pi) * x * (1 + 0.044715*x*x))
  ```

  The accepted GEGLU route uses `scalar.logisticf`, passes 15 focused exact
  graph-op rows including 3 GEGLU rows, and passes the full 33-run basket.
- **Evidence:**
  - Failing direct GELU run:
    `cache/hrx2/phase1_0/route-slice-44-glu-large/test.csv`.
  - Passing logistic run:
    `cache/hrx2/phase1_0/route-slice-44-glu-large/test-trace-logistic.csv`.
  - Route trace:
    `cache/hrx2/phase1_0/route-slice-44-glu-large/hrx2-logistic.jsonl`.
- **Owner:** Loom AMDGPU math lowering. HRX2 authors should use the logistic
  identity until `scalar.geluf<tanh>` lowers directly on AMDGPU.

## 2026-06-14: AMDGPU binding materialization reused kernarg SGPRs across SMEM loads

- **Area:** Loom AMDGPU binding materialization and HRX direct executable
  dispatch.
- **Affected route:** HRX2 route
  `mul_mat_f16_f32_batched_attention_wg256`, export
  `hrx2_mul_mat_f16_f32_batched`.
- **Observed case:** Focused backend op validation for the active F16/F32
  attention `MUL_MAT` route times out with the route enabled. The active JIT
  instance is:

  ```text
  k=128 rows=256 cols=1 dst_ne2=32 dst_ne3=1
  workgroup_size=256x1x1 workgroup_count=256x32x1
  constants=0 bindings=3
  ```

- **Symptom:** A minimal HRX runtime runner that loads the dumped HSACO with
  `hrx_executable_load_data`, submits `hrx_stream_dispatch`, and waits with
  `hrx_stream_synchronize` reaches:

  ```text
  dispatch
  synchronize
  ```

  and then does not return for the full production grid. A small `1x32` grid
  completes and verifies, so the failure is not a simple load/export/binding
  rejection.
- **Root cause:** Loom's AMDGPU binding materialization allowed a final
  same-width kernarg load to reuse the live kernarg segment pointer SGPR pair as
  its result. The hung HSACO prologue starts:

  ```text
  s_load_b64  s[0:1], s[0:1], 0x10
  s_load_b128 s[4:7], s[0:1], null
  ...
  s_waitcnt lgkmcnt(0)
  ```

  The second load was meant to use the original kernarg pointer, but its base
  SGPR pair is also the pending destination of the previous SMEM load. That
  relies on source operand timing across an SMEM RAW hazard; when the second
  load observes the newly loaded binding pointer, it reads bogus source
  pointers from the destination buffer and the full-grid kernel can hang. The
  fixed prologue keeps the kernarg pointer in `s[0:1]` and writes binding
  pointers to distinct SGPRs:

  ```text
  s_load_b64  s[4:5], s[0:1], 0x10
  s_load_b128 s[8:11], s[0:1], null
  ...
  s_waitcnt lgkmcnt(0)
  ```
- **Fix:** Disable
  `LOOM_AMDGPU_HAL_BINDING_LOAD_FLAG_REUSE_KERNARG_STORAGE` in
  `sources/hrx-system/loom/src/loom/target/arch/amdgpu/hal/binding_materialization.c`.
  A future narrower fix could keep the optimization only when no later kernarg
  loads can use the reused SGPR pair as a base before an explicit wait/copy.
- **Evidence:** Scratch repro:
  `cache/hrx2/attention-route-hrx-runtime-repro/`.
  Run from the workspace root:

  ```bash
  cache/hrx2/attention-route-hrx-runtime-repro/run.sh \
    cache/hrx2/attention-route-hrx-runtime-repro/kernel.active-route.patched.hsaco \
    256 32
  ```

  Expected current behavior: timeout at `hrx_stream_synchronize`. Passing
  reduced-grid check:

  ```bash
  cache/hrx2/attention-route-hrx-runtime-repro/run.sh \
    cache/hrx2/attention-route-hrx-runtime-repro/kernel.active-route.patched.hsaco \
    1 32
  ```

  observed `checked=32 bad=0`.
  Passing fixed full-grid repro:

  ```bash
  cache/hrx2/attention-route-hrx-runtime-repro/run.sh \
    cache/hrx2/attention-route-hrx-runtime-repro/kernel.binding-materialization-fix.hsaco \
    256 32
  ```

  observed `checked=8192 bad=0`. Focused backend op validation also passes with
  the route enabled and evidence dumped under
  `cache/hrx2/compiler-fix-f16-attn-focused-binding-only-20260615/`.
- **Important workflow note:** Do not disable this route to hide the issue.
  Backend op unit tests are the first isolation gate; integration/model tests
  come after the op-level failure is understood.
- **Owner:** Loom AMDGPU compiler. Earlier local workgroup-reduce publication
  and LDS-wait experiments did not fix the backend op timeout and should not be
  treated as causal for this hang.

## HRX2 Runtime: GET_ROWS Op-Offload Copied CPU Embeddings Every Decode Graph

- **Status:** Fixed in llama.cpp by removing HRX2 `offload_op`.
- **Symptom:** `llama-cli` / `llama-bench` decode used a full CPU core and low
  effective GPU throughput even with zero scheduler compute fallback.
- **Root cause:** HRX2 advertised `offload_op` for supported `GET_ROWS`. The
  llama.cpp model loader intentionally keeps the input embedding table on CPU.
  With default `op_offload`, the scheduler selected HRX2 for the one-token
  embedding lookup and then copied the full CPU-resident `token_embd.weight`
  into the HRX split input for each decode graph.
- **Evidence:** In
  `cache/hrx2/phase2a/cpu-underfeed-diagnostic-20260615-114026/`,
  Llama 3.2 3B Q4_K `p0 n16` showed 17 split-input copies of
  `token_embd.weight`, each 323 MB, for about 5.49 GB of CPU-sourced input
  traffic. Throughput was about 3 tok/s. The same binary with `-nopo 1` reached
  30.0 tok/s. After removing HRX2 `offload_op`, default `op_offload` reached
  29.6 tok/s and traced CPU split-input traffic dropped to 348 KB with no large
  inputs:
  `cache/hrx2/phase2a/offload-policy-fix-20260615-114809/`.
- **Rule:** Do not use scheduler `offload_op` for host-resident `GET_ROWS`
  unless the backend can gather from the host source without recurring full
  source materialization. HRX1 avoided this by not providing `offload_op`.
  CUDA/Vulkan avoid it by assigning `GET_ROWS` zero batch size in their
  op-offload heuristic; CANN excludes `GET_ROWS` explicitly.
- **Follow-up cleanup:** HRX2 full-offload runs can instead place
  `token_embd.weight` on the first HRX2 device at model load time. That removes
  the residual embedding CPU compute fallback without recurring full-table split
  input copies. Llama 3.2 3B Q4_K smoke: `decode-p1n64` CPU compute `12 -> 0`
  and `prefill-p64n0` CPU compute `6 -> 0` with neutral/slightly positive
  throughput in
  `cache/hrx2/phase2a/hrx2-input-embd-smoke-20260615/`.

## HRX2 Runtime: Decode Needs HRX1 Stream Interop, Not Just Kernels

- **Status:** Partially fixed in llama.cpp by porting HRX1-style submit
  batching to HRX2. Remaining runtime interop work is open.
- **Symptom:** After the `offload_op` fix, decode still used substantial CPU
  and had low effective GPU utilization even with only 12 CPU scheduler compute
  nodes in the basket trace.
- **Root cause:** HRX2 had not carried forward several solid HRX1 runtime
  interop mechanisms. The first proven missing piece was submit batching:
  HRX1 flushed streams by real dispatch count and matmul-byte progress, while
  HRX2 flushed only at graph end. This left many tiny decode dispatches waiting
  behind poor stream-progress cadence.
- **Fix landed:** HRX2 now uses HRX1-style submit batching under
  `GGML_HRX2_DISPATCHES_PER_SUBMIT`,
  `GGML_HRX2_MAX_MUL_MAT_BYTES_PER_SUBMIT`, and
  `GGML_HRX2_DISABLE_SUBMIT_BATCHING=1`. Reverse-order Llama 3.2 3B Q4_K
  decode A/B showed disabled 16.067 tok/s versus enabled 22.690 tok/s. Stream
  synchronization trace time dropped from about 1.489 s to 0.124 s with the
  same 36725 dispatches.
- **Evidence:** `cache/hrx2/phase2a/submit-batching-ab-20260615-120956/`,
  `cache/hrx2/phase2a/submit-batching-ab-reverse-20260615-121032/`, and
  `cache/hrx2/phase2a/submit-batching-three-model-decode-20260615-121056/`.
  Three-model decode speedup versus the prior Phase 2a baseline was 1.38x to
  1.74x.
- **Remaining limitations:** HRX2 now has the first tranche of HRX1-style
  staging/copy interop, but decode can still be split-input bound even when CPU
  compute fallback is zero. Agents should keep treating runtime backpressure as
  a Phase 2a boulder, especially for small models. Prefill remains dominated by
  kernel/fusion quality and attention route coverage; submit batching helps
  decode more than prompt prefill.

## HRX2 Runtime: Decode Can Be Split-Input Bound With Zero CPU Fallback

- **Status:** Open.
- **Symptom:** After embedding placement removed CPU compute fallback, Llama 3.2
  3B Q4_K `decode-p1n64` still ran at about `41.4 tok/s` versus Vulkan at
  about `140.9 tok/s`.
- **Evidence:** In
  `cache/hrx2/phase2a/hrx2-input-embd-smoke-20260615/`, the HRX2 trace had
  `260` scheduler splits for `65` decode graphs, `455` split-input copies, and
  `725` stream synchronizations totaling about `1.01 s` inside a `1.55 s`
  benchmark row.
- **Root cause class:** CPU leaf tensors for token ids, ROPE positions/freqs,
  KV row indices, and attention masks keep breaking an otherwise HRX2 graph
  into split-input copies. Zero CPU compute fallback is therefore necessary but
  not sufficient for decode runtime health.
- **Rejected experiment:** Assigning all graph inputs to the first backend via
  a scheduler env knob made llama.cpp abort in
  `llama_kv_cache::set_input_k_idxs`, because KV-cache input setters currently
  assert host buffers and fill them directly.
- **Rule:** Do not use a blind global scheduler placement change for graph
  inputs. Prefer HRX2 route/fusion specialization that removes hot CPU leaf
  dependencies, or first redesign llama.cpp input setters to support selected
  device-resident graph inputs through backend tensor set APIs.

## HRX2 Catalog: Route Domains Must Be Derived From Real Graph Split Shapes

- **Status:** Partially fixed in llama.cpp for p512 F16 attention route
  coverage.
- **Symptom:** p512 prefill showed hundreds of CPU `MUL_MAT` and `SOFT_MAX`
  nodes even though catalog metadata appeared to have nearby attention and
  softmax routes.
- **Root cause:** The first p512 route attempt reasoned from nominal attention
  shapes instead of the scheduler's real split graph. The actual p512 basket
  shapes included `kq` with `rows=512`, `cols={1,16,512}`; softmax with
  `ncols=512`, `nrows=24..16384`; and `kqv` with `k=512`, `rows=128`,
  `cols={1,16,512}`. The route domain covered only `rows<=256`, `cols<=64`,
  and `k<=256`, so HRX2 correctly declined those ops and the scheduler kept
  them on CPU.
- **Fix landed:** The generic F16/F32 batched attention route now covers
  `k<=512`, `rows<=512`, and `cols<=512`; masked p512 softmax uses a
  target-generic `ncols=512` / `wg512` route because the kernel maps one
  workitem to one softmax column.
- **Evidence:** `cache/hrx2/phase2a/p512-attn-k512-three-model-20260615-122321/`
  reduced p512 CPU compute from 510-582 nodes down to 6 embedding `GET_ROWS`
  nodes across the three-model basket. Throughput improved only 1.05x to
  1.13x, so the remaining Phase 2a gap is kernel/fusion quality rather than
  this route hole.
- **Rule:** Always derive route-domain changes from `sched.jsonl` and
  `hrx2.jsonl` for the exact model/prompt bucket. Generic backend op tests are
  necessary but not sufficient because they often do not include the exact
  graph split shapes that matter.

## HRX2 Diagnostics: Trace I/O Can Corrupt Decode Benchmarks

- **Status:** Fixed in llama.cpp by keeping HRX2 and scheduler trace files open
  for the process lifetime.
- **Symptom:** With `GGML_HRX2_TRACE_JSONL` and `GGML_SCHED_TRACE_JSONL`
  enabled, Llama 3.2 3B Q4_K_M decode measured about 22.7 tok/s while the same
  run without traces measured about 40.6 tok/s.
- **Root cause:** The trace writers reopened and appended the JSONL file for
  each event. Decode emits tens of thousands of dispatch/provider events and
  thousands of scheduler events, so diagnostic I/O became part of the measured
  runtime.
- **Fix landed:** `ggml-hrx2.cpp` and `ggml-backend.cpp` now cache the trace
  file handle by path. The corrected traced decode result is about 39.5 tok/s,
  within roughly 2% of the no-trace run.
- **Rule:** Continue to use traced runs for Phase 2a evidence, but if a decode
  number looks suspicious, verify against a no-trace run before attributing the
  gap to backend runtime or kernel quality.

## HRX2 Fusion: Dispatch Count Reductions Are Not Automatically Boulders

- **Status:** RMS_NORM+MUL exists as an opt-in experiment, disabled by default.
- **Symptom:** Enabling the fused route reduced Llama 3.2 3B Q4_K_M decode
  dispatches from 36725 to 33020 but did not improve throughput.
- **Evidence:** `cache/hrx2/phase2a/rms-norm-mul-fusion-ab-20260615-124513/`
  measured decode 40.251 tok/s disabled versus 39.892 enabled, and p512 60.997
  disabled versus 60.025 enabled.
- **Rule:** Treat fusions as accepted only when same-run evidence beats the
  unfused path for the relevant shape/regime. A fusion that merely reduces
  dispatch count can still lose if the model is dominated by synchronization,
  memory traffic elsewhere, or the fused kernel has worse local codegen.

## HRX2 Catalog: Matmul Routes Need Explicit Column-Multiple Guards

- **Status:** Fixed in llama.cpp by adding `cols_multiple_of` route guards and
  applying it to the Q4_K cols4 prompt route.
- **Symptom:** A new Q4_K prompt route that computes four RHS columns per
  workgroup was safe and faster for real prompt columns (`cols=16/64/512`), but
  a synthetic `cols=5` backend-op probe first tried to JIT that route and Loom
  failed HSACO emission with `low schedule dependency cycle in block 22`.
  HRX2 then fell back to the old direct route and the test passed, but relying
  on failed JIT as route selection is not acceptable.
- **Root cause:** The catalog could express `cols_min/max` and dispatch
  `cols_per_workgroup`, but generic matmul matching had no way to require
  `shape.cols % N == 0`. The route was therefore considered applicable to
  shapes outside the intended schedule family.
- **Fix landed:** `ggml_backend_hrx2_kernel_route` now parses
  `shape_guards.cols_multiple_of`; generic `MUL_MAT`, `MUL_MAT_ID`, and
  F16/F32 matmul route matching apply it to `cols` or `nselected` as
  appropriate. The Q4_K cols4 route declares `cols_multiple_of: 4`.
- **Evidence:** `cache/hrx2/phase2a/q4k-cols4-oddcols-guarded-*` selected the
  old direct route for `cols=5` with no failed cols4 compile. The model-derived
  Q4_K gate in `cache/hrx2/phase2a/q4k-cols4-focused-guarded-*` selected the
  old direct route for `cols=1` and the cols4 route for `cols=16/64`.
- **Rule:** Any prompt-tiled matmul route whose schedule assumes a column tile
  must encode the tile divisibility in route metadata. Do not rely on in-kernel
  bounds checks or failed JIT fallback for route applicability.

## Loom AMDGPU: Q4_K Cols4 Odd-Column Compile Failure

- **Status:** Avoided in llama.cpp via `cols_multiple_of=4`; underlying Loom
  AMDGPU lowering issue remains worth reporting if odd-column tails need this
  exact schedule.
- **Repro:** Q4_K cols4 Loom export
  `hrx2_mul_mat_q4_k_f32_cols4_static` with `k=256`, `rows=16`, `cols=5`,
  `workgroup_size=256`.
- **Observed diagnostic:** `AMDGPU HSACO emission failed: EMIT/TARGET: low
  schedule dependency cycle in block 22`.
- **Impact:** No production impact after the route guard because current
  prompt buckets use multiples of four and odd columns route to the old direct
  kernel. The diagnostic matters for future tail-general routes and should not
  be confused with a correctness failure in the accepted `cols=16/64/512`
  schedule.

## HRX2 Runtime: HRX1 Copy/Fill Parity Is Necessary But Not A Decode Boulder

- **Status:** Accepted in llama.cpp HRX2 as runtime parity.
- **Context:** HRX1 uses timeline-mediated direct queue copy/fill helpers for
  synchronous buffer operations and supports graph-level `CPY` including
  contiguous `F32 -> F16` conversion. HRX2 now carries these runtime surfaces:
  queue fill/copy helpers, same-type graph CPY, F32 strided CPY through the
  `CONT` route when possible, row-copy fallback, and a Loom
  `copy_f32_f16_generic_wg256` provider.
- **Evidence:** The final focused gate is
  `cache/hrx2/phase2a/runtime-parity-copy-f32-f16-20260615-164621/`.
  `CPY`/`CONT` produced 462 rows, 0 fail-like rows, and 119 supported rows.
  The new F32->F16 provider compiled and ran with no spills or private/local
  memory in its compile report.
- **Model impact:** Basket decode traces currently contain zero graph `CPY`
  nodes, and split-input counts remain unchanged, so this parity work does not
  materially move decode throughput by itself. That is expected.
- **Rule:** Keep HRX1 runtime semantics in HRX2, but do not mistake runtime
  parity for the Phase 2a bulk lift. The decode boulders remain split-input
  backpressure, direct ROPE/VIEW/SET_ROWS KV-cache writes, attention/cache
  fusion, and quantized hero-kernel quality.

## HRX2 Runtime: Scratch Growth Must Retire, Not Synchronize And Free

- **Status:** Fixed in llama.cpp HRX2.
- **Symptom:** HRX2's persistent Q8_1 prompt scratch buffer synchronized the
  compute stream and released the old scratch buffer whenever the requested
  capacity grew.
- **Why it matters:** HRX1 avoided this by retiring old persistent scratch and
  reclaiming it after normal backend synchronization. A mid-graph synchronize in
  a scratch allocator is a runtime cliff, even if it only appears on uncommon
  shape transitions or currently env-gated prompt routes.
- **Fix landed:** `ggml_backend_hrx2_device_scratch` now tracks retired buffers;
  growth moves the old buffer into the retired list, and
  `ggml_backend_hrx2_synchronize` releases retired scratch after synchronizing
  the graph stream.
- **Evidence:** `cache/hrx2/runtime-parity-audit-20260615-165658/` has 1788
  focused `MUL_MAT`/`CPY`/`CONT` rows with no backend error rows. The
  scratch-exercising prompt smoke
  `cache/hrx2/phase2a/runtime-parity-q8scratch-smoke-20260615-165752/` selected
  the Q8_1 prompt path 112 times and had zero CPU compute nodes.
- **Rule:** Scratch/persistent temporary growth in HRX2 should follow the HRX1
  retire-and-reclaim model. Do not introduce stream synchronizes inside
  allocation growth paths unless the graph has no possible live users.

## HRX2 Runtime: HRX1 Parity Still Does Not Mean True Scheduler Async

- **Status:** Known limitation inherited from HRX1.
- **Context:** HRX1 and HRX2 both advertise async-capable GPU devices but leave
  the backend tensor async callbacks and ggml event callbacks null. Internally
  they use HRX streams, timeline semaphores, staging arenas, and queue copy/fill
  helpers, but llama.cpp's scheduler cannot use backend events for real
  split-to-split overlap.
- **Fix landed:** HRX2 now matches the rest of HRX1's runtime contract more
  closely: transient scratch pool, persistent route scratch, transfer-stream
  initial upload behavior, graph-compute synchronization by default, global
  fusion disable, and graph tracing.
- **Evidence:** Full HRX2 backend op gate
  `cache/hrx2/phase2a/runtime-parity-opgate-20260615-171403/` produced 320
  supported rows with 0 supported failures. Phi decode smoke
  `cache/hrx2/phase2a/runtime-parity-smoke-20260615-171512/` completed with
  zero CPU compute nodes. A later focused runtime-surface recheck
  `cache/hrx2/phase2a/runtime-parity-recheck-20260615-181232/` covered
  `RMS_NORM,MUL_MAT,SET_ROWS,CPY,CONT` and produced 211 supported HRX2 rows
  with 0 supported errors. The trace included stream CPY routes, `cont_f32`,
  F32->F16 copy conversion, SET_ROWS, scratch-backed quantized matmuls, and
  RMS_NORM.
- **Impact:** Future runtime work should distinguish "HRX1 parity" from
  "modern scheduler async." HRX1 parity is now available for Phase 2a route
  work, but reducing decode backpressure further likely requires fewer split
  inputs, fewer dispatches, or real ggml event/copy callback support rather
  than more queue-copy parity patches.
- **Rule:** When evaluating runtime changes, report tok/s and all wait classes.
  Do not accept a runtime patch solely because it changes the number of traced
  `hrx_stream_synchronize` calls.

## HRX2 JIT Provider Failures Need First-Class Diagnostics

- **Status:** Open limitation in llama.cpp HRX2 tooling/runtime.
- **Symptom:** While bringing up `ROPE_SET_ROWS`, the graph route matched but
  provider compilation failed. The JSON trace recorded
  `provider_compile status=failed` and `provider_unavailable`, but it did not
  include the Loom diagnostic. The model-level symptom was only
  `failed to decode prompt batch`.
- **Root cause found manually:** Reproducing with `loom-compile` showed the
  actual compiler diagnostic: AMDGPU address lowering rejected destination
  address arithmetic because dynamic row-derived addresses lacked the required
  bounded `index.assume` facts.
- **Impact:** Future route bring-up can waste time in integration tests unless
  agents immediately reproduce the exact route config with `loom-compile` or
  the runtime persists failure diagnostics.
- **Desired fix:** Include the first Loom diagnostic/status message in the HRX2
  `provider_compile` failure trace event and optionally persist a failed-provider
  evidence directory with config bindings and diagnostic text.
- **Rule:** If a provider is unavailable after route selection, do not guess.
  Reproduce the exact `cache_key` config with `loom-compile --backend=amdgpu-hal
  --target=<gfx> --compile-report=details` and fix the source or route.

## Large Prompt Shapes Can Make Generic `test-backend-ops perf` Too Expensive

- **Status:** Tooling limitation / workflow hazard.
- **Symptom:** A focused `test-backend-ops perf` run over four exported Llama
  3.2 p512 prompt matmul rows was allowed to run for about 46 seconds and was
  still busy on HRX2, while a full Vulkan p512 model pass reports about 80-90 ms
  total on the same model. The run was terminated manually.
- **Artifact:** `cache/hrx2/phase2a/q4k-op-perf-testfile-20260615-183317/`.
  The partial trace confirms route selection and JIT for
  `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256` at `k=3072, rows=3072,
  cols=512` and `k=8192, rows=3072, cols=512`.
- **Cause:** The generic backend perf harness is not currently a safe default
  for very large, slow HRX2 prompt shapes. It may run a long inner-loop timing
  schedule before emitting CSV, so it is poor for exploratory isolation when a
  route is already suspected to be pathological.
- **Rule:** For large prompt matmul isolation, first use model-level warm
  runs plus HRX2 route traces and Vulkan per-op logger. Use `test-backend-ops`
  in `test`/`support` mode for correctness and route coverage. Only use
  `perf` on large exported shapes when the iteration count/runtime is known or
  the kernel is already near target.

## Loom AMDGPU Lowering Hazards On Q4_K x Packed Q8_1 MMQ Route

- **Status:** Partially worked around in llama.cpp; keep as a regression note
  and WYSIWYG guidance for future kernels.
- **Route attempted:** `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`, an
  HRX1-style 32-row x 32-column Q4_K x packed Q8_1 tile with 128 workitems and
  packed RHS x4 scratch.
- **Runtime route state:** The route metadata and WIP Loom export are present
  in llama.cpp so the failing shape can be reproduced in the production control
  plane, but route selection is behind a dedicated
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1` opt-in in addition to the broader
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1` prompt opt-in. Do not default-enable
  this route until the compiler issue below is fixed and model-derived prompt
  shapes pass the backend op gate.
- **Original failure:** JIT compilation failed during `source-to-low` with
  `INTERNAL; AMDGPU branch argument materializer selected for an unsupported type`
  at `hrx-system/loom/src/loom/target/arch/amdgpu/lower/control.c:339`.
- **Evidence:** The production model route attempted real p512 shapes such as
  `k=3072, rows=3072, cols=512`, `k=3072, rows=1024, cols=512`, and
  `k=8192, rows=3072, cols=512`, then recorded `provider_compile failed` and
  `provider_unavailable` before falling through. Standalone `loom-compile`
  repro artifacts:
  `cache/hrx2/phase2a/q4k-q8x4-mmq32-compile-repro-20260615-193610/` and
  `cache/hrx2/phase2a/q4k-q8x4-mmq32-compile-repro-ir-20260615-193621/`.
  The IR dump contains the export at about line 714 under
  `kernel.def export("hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static")`.
  Earlier synthetic repros also hit the same internal at
  `cache/hrx2/phase2a/q4k-x4mmq-opgate-20260615-190045/` and
  `cache/hrx2/phase2a/q4k-x4mmq-modelshape-opgate-20260615-190155/`.
- **Isolation gate:** `cache/hrx2/phase2a/q4k-q8-prompt-gated-opgate-20260615-194009/`
  proves that `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1` alone still passes the
  focused `MUL_MAT` backend op gate with 49 supported rows, 0 supported errors,
  and 0 `q8_1_x4` route events. This keeps the blocked MMQ candidate from
  contaminating the older direct/cols4 Q8_1 prompt experiment.
- **Attempts already made:** The initial row-tail `scf.if` was replaced with
  branchless row clamping, and the divergent two-iteration RHS load `scf.for`
  was rewritten as two explicit per-lane loads. The same lowering internal
  remained, so do not spend time rediscovering those variants as fixes.
- **Current workaround:** The accepted source workaround is to make the Q4_K
  group value naturally constant at the lowering point: use an outer Q4 block
  loop and an inner `scf.for %group = [0 to 8 step 1] ... unroll(%eight)`.
  With this spelling, the `group < 4` scale/min branch folds and
  `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static` compiles at representative
  prompt shapes. Artifact:
  `cache/hrx2/phase2a/q4k-x4mmq-unrolled-group-repro-20260615-201222/`.
- **Additional quantizer failure found:** The paired x4 Q8_1 RHS quantizer used
  `kernel.subgroup.shuffle` with computed lane ids. AMDGPU lowering rejected it
  with `source-to-low constraint 'subgroup_shuffle.exact_lane' is not
  satisfied`. Artifact:
  `cache/hrx2/phase2a/q8-1-x4-quantize-compile-repro-20260615-201446/`.
- **Current quantizer workaround:** Store one quantized byte per lane into the
  packed x4 byte layout instead of using shuffle-gathered `vector<4xi8>` stores.
  This preserves the dword load layout consumed by the MMQ kernel and compiles
  cleanly. Artifact:
  `cache/hrx2/phase2a/q8-1-x4-quantize-byte-store-compile-20260615-201538/`.
- **Impact:** The compiler blockers are no longer the immediate reason the x4
  route fails to move Llama 3.2 3B prefill. The current blocker is schedule and
  algorithm quality: the functional x4 route is still slower than the default
  F32-RHS cols8 route on W7900.
- **Rule:** For Loom WYSIWYG packed kernels, avoid data-dependent branch values
  and computed subgroup shuffle lanes in the hot path unless a compile-report
  artifact proves the lowering accepts them. Prefer meta-programmed/unrolled
  exact-lane or exact-group spellings when the value is naturally drawn from a
  small fixed algorithmic set.

## 2026-06-15: Q4_K x Packed Q8_1 x4 route is not a correctness-clean Phase 2a candidate

- **Area:** HRX2 packed prompt matmul route acceptance.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  route `mul_mat_q4_k_q8_1_x4_mmq32x32_k256_32768_r1_32768_c32_512_wg128`.
- **Observed case:** Exported Llama 3.2 3B Q4_K_M p512 prompt matmul rows:
  `cache/hrx2/phase2a/q4k-op-perf-testfile-20260615-183317/llama32-q4k-p512-ops.txt`.
- **Symptom:** With `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1` and
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`, `test-backend-ops test -b HRX20`
  selects the x4 route but fails CPU-reference comparison with NaNs at index
  0 on the Q4_K rows. The default no-env route on the same test file selects
  `mul_mat_q4_k_f32_cols8...` and passes all four exported rows.
- **Follow-up experiment:** A local, reverted probe tried to add a Q4_K A-tile
  workgroup cache to the x4 route while preserving the existing 32x32 control
  plane. It compiled, but failed the same op gate with about 1.0 relative error.
  Variants that cached Q4 payload as `vector<1xi32>`, cached it as
  `vector<4xi8>`, and added explicit `scf.for ... unroll(%eight)` on the
  producer loop all failed. Recomputing scale/min globally did not fix it, so
  the failed probe should not be rediscovered as a quick optimization path.
- **Impact:** The x4 route must remain opt-in and must not be used as evidence
  for Phase 2a throughput until it has a model-derived backend op gate with 0
  supported errors. The p512 boulder is still quantized prompt matmul quality,
  but the next credible implementation should be a clean HRX1/Vulkan-inspired
  tiled route or HIP reference, not incremental promotion of the current x4
  route.
- **Rule:** For every hero route, run backend op unit tests before integration
  tests or model timing. A model-level tok/s number from an opt-in route is not
  acceptance evidence unless the exact model-derived op rows pass against ggml's
  CPU reference first.
- **Narrowing update:** The packed x4 quantizer/layout itself is likely not the
  root cause. A temporary diagnostic route that consumed packed x4 RHS directly
  without MMQ/LDS tiling passed the same model-derived Q4_K op gate:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-diagnostic-20260615-210115/`.
  The MMQ route still failed when Q8 scale/sum were loaded directly from global
  instead of LDS:
  `cache/hrx2/phase2a/q4k-x4-mmq-global-ds-diagnostic-20260615-210230/`.
  It also failed when Q8 payload words and scale/sum were all loaded directly
  from global:
  `cache/hrx2/phase2a/q4k-x4-mmq-global-payload-diagnostic-20260615-210331/`.
  Treat the remaining bug as MMQ row/column lane mapping, accumulator/control
  spelling, or Q4/Q8 arithmetic structure until proven otherwise.

## 2026-06-15: Q4_K/F32 cols16 direct route passed op gate but regressed model throughput

- **Area:** HRX2 prompt matmul route acceptance.
- **Affected source:** Reverted local probe in
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom` and
  `catalog/routes/mul_mat_q4_k_f32.json`.
- **Observed case:** Mechanical widening of the accepted
  `mul_mat_q4_k_f32_cols8...` route to a new
  `mul_mat_q4_k_f32_cols16...` route.
- **Correctness:** The route passed the focused backend op gate on exported
  Llama 3.2 3B p512 Q4_K prompt rows:
  `cache/hrx2/phase2a/q4k-op-perf-testfile-20260615-183317/llama32-q4k-p512-ops.txt`.
- **Performance artifact:**
  `cache/hrx2/phase2a/q4k-cols16-hrx2-20260615-204347/`.
- **Symptom:** Full model prefill throughput regressed or stayed flat:
  Llama 3.2 p64 `77.481 -> 71.253 tok/s`, Llama 3.2 p512
  `83.530 -> 82.723 tok/s`, Llama 3.1 p512 `32.892 -> 32.412 tok/s`,
  Phi p512 `67.874 -> 66.501 tok/s`.
- **Impact:** Do not repeat a simple F32-RHS direct-cols widening as the Q4_K
  prefill bulk-lift path. The issue is schedule class, not just the number of
  columns per workgroup. The next candidate must materially change data reuse
  or arithmetic form: packed RHS, A/B tile reuse, a known-good HIP reference, or
  Vulkan/HRX1-inspired MMQ structure.
- **Rule:** Passing backend op tests is necessary but not sufficient. For hero
  prompt routes, require same-basket model A/B before promotion, and reject
  correctness-clean variants that preserve the wrong schedule class and do not
  move tok/s.

## 2026-06-15: F16 batched route family was not priority sorted

- **Area:** llama.cpp HRX2 catalog route selection.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/ggml-hrx2.cpp`.
- **Symptom:** A higher-priority experimental
  `mul_mat_f16_f32_batched_attention_cols4_wg256` route was present in the
  generated catalog and matched prompt `cols=512` shapes, but dispatch still
  selected the lower-priority `mul_mat_f16_f32_batched_attention_wg256`
  fallback. Trace artifact:
  `cache/hrx2/phase2a/f16-cols4-op-test-20260615-205158/`.
- **Cause:** `mul_mat_f16_f32_routes` was collected into the device context but
  was missing from the route-family priority sort block. The route vector
  therefore stayed in catalog-index order.
- **Fix:** Sort `mul_mat_f16_f32_routes` with the same `route_less` comparator
  used for the other matmul route families.
- **Validation:** After the fix, the focused F16 attention backend-op gate
  selected the cols4 route for four `cols=512` rows and the scalar route for
  four `cols=1` rows, with all rows passing. After rejecting and removing the
  cols4 route, the same focused op file still passed:
  `cache/hrx2/phase2a/f16-route-sort-regression-20260615-205433/`.
- **Rule:** When adding a new route family or new multi-versioned routes, verify
  with provider traces that the intended route wins for a matching shape before
  trusting performance results. A catalog entry being present is not enough.

## 2026-06-15: F16 batched cols4 direct route was correctness-clean but not a bulk lift

- **Area:** HRX2 prompt attention matmul route acceptance.
- **Affected source:** Reverted local probe in
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_f16_f32_batched.loom`
  and `catalog/routes/mul_mat_f16_f32_batched.json`.
- **Observed case:** Direct scalar F16/F32 batched attention route computing
  four adjacent prompt columns per workgroup, guarded by `cols_multiple_of=4`.
- **Correctness:** Passed model-derived backend-op rows from
  `cache/hrx2/phase2a/p512-c512-op-export-20260615-103625/basket-p512-c512-f16-attention-ops.txt`.
- **Performance artifact:**
  `cache/hrx2/phase2a/f16-batched-cols4-hrx2-20260615-205222/`.
- **Symptom:** p512 improved only 1.6-3.4%, while p64 regressed on Phi and
  Llama 3.2. Decode was unchanged because the route correctly did not apply to
  `cols=1`.
- **Impact:** This route is not a Phase 2a bulk-lift candidate. It confirms that
  adjacent-column scalar widening alone is too small and can hurt the shorter
  prompt bucket. Future F16 attention work should use a materially different
  schedule or a fusion route that removes memory traffic/dispatches.

## 2026-06-15: Q4_K packed Q8_1 x4 MMQ route fails correctness in scale/metadata path

- **Area:** HRX2 Q4_K prompt matmul / packed Q8_1 x4 RHS route.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/quantize_q8_1.loom`, and
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`.
- **Observed case:** Focused model-derived Llama 3.2 p512 Q4_K rows with:

  ```bash
  GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1
  GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
    --test-file cache/hrx2/phase2a/q4k-op-perf-testfile-20260615-183317/llama32-q4k-p512-ops.txt \
    --output csv
  ```

- **Symptom:** The Q4_K rows select the x4 MMQ route and fail with NaNs at
  output index 0. The neighboring Q6_K row passes.
- **Diagnostic evidence:**
  - Current repro after rebuild:
    `cache/hrx2/phase2a/q4k-x4-repro-20260615-211932/`.
  - Forcing Q4 scale/min finite moved, but did not remove, NaNs:
    `cache/hrx2/phase2a/q4k-x4-diag-q4scale1-20260615-212014/`.
  - Forcing both Q4 and Q8 scale/sum finite removed NaNs and left ordinary
    numerical mismatch, proving the dot payload/destination path can produce
    finite values:
    `cache/hrx2/phase2a/q4k-x4-diag-allscale1-20260615-212051/`.
  - Loading Q8 `d/s` directly from global instead of LDS and rewriting the Q4
    scale-byte fetch from unaligned i32 views to explicit i8 loads did not fix
    the NaN:
    `cache/hrx2/phase2a/q4k-x4-direct-ds-20260615-212302/`,
    `cache/hrx2/phase2a/q4k-x4-byte-scale-direct-ds-20260615-212434/`.
  - The temporary diagnostic source diff was reverted from llama.cpp and saved
    as `cache/hrx2/phase2a/q4k-x4-diagnostic-edits.patch`.
- **Impact:** Keep the x4 MMQ route opt-in. Do not use it for model
  performance evidence until a focused op gate is zero-error. The issue is not
  simply "Loom cannot emit dot4"; the route emits dot instructions. It is a
  metadata/scale spelling or packed-layout interaction in this specific kernel.
- **Rule:** The next Q4_K packed route should be rewritten as a clean
  HRX1/Vulkan-style schedule with a correctness-first metadata path. Avoid
  further patching of the current 32x32 route unless the experiment directly
  proves which metadata value becomes NaN.
- **Latest narrowing update:** A later opt-in focused gate on the model-derived
  Q4_K rows moved the failure boundary:
  - Making Q8_1 `d/s` metadata single-writer in LDS and storing it as f32
    removed the NaNs, but the route still fails with finite CPU-reference
    mismatch around `ERR ~= 1.0`.
  - Requesting subgroup size 32 for the x4 quantizer did not materially change
    the mismatch.
  - Packing the x4 Q8 payload with an explicit single i32 store per four bytes
    also did not materially change the mismatch.
  - Running the same focused op file with only the non-x4 Q8_1 prompt route
    enabled passes all rows, which isolates the remaining issue to the x4 MMQ
    consumer/layout/schedule path rather than the generic Q4_K/Q8_1 plumbing.
  - The next in-flight diagnostic was to replace unaligned i32 scale/min byte
    loads with explicit i8 loads in the x4 consumer; test it before trusting
    any source state that includes that edit.
- **Latest artifacts:**
  - NaNs after single-writer `d/s`:
    `cache/hrx2/phase2a/q4k-x4-single-writer-ds-20260615-215839/`.
  - Finite mismatch after f32 `d/s` LDS:
    `cache/hrx2/phase2a/q4k-x4-f32-ds-20260615-220035/`.
  - Subgroup-size probe:
    `cache/hrx2/phase2a/q4k-x4-f32-ds-wave32-20260615-220227/`.
  - Explicit packed Q8 store probe:
    `cache/hrx2/phase2a/q4k-x4-packed-q8-store-20260615-220513/`.
  - Passing non-x4 control on the same op file:
    `cache/hrx2/phase2a/q4k-q8-cols4-same-opfile-20260615-220557/`.
- **Rejected patch bundle:** The full local experiment combining
  single-writer/f32 `d/s`, explicit packed Q8 i32 stores, dispatch subgroup
  size 32, and explicit i8 Q4 scale/min loads still failed the two c64 Q4_K
  rows with finite mismatch around `ERR ~= 1.0`. It was saved and reverted:
  `cache/hrx2/phase2a/q4k-x4-failed-experiment-20260615-221511/`.
- **Standalone compile evidence:** The current artifacts compile as wave32.
  `hrx2_quantize_q8_1_x4_f32` reports wavefront size 32 with 12 VGPR and no
  spills. `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static` reports wavefront size
  32, emits `v_dot4_i32_iu8`, uses 1280 bytes LDS, and reports 153 VGPR with
  no spills. Artifact:
  `cache/hrx2/phase2a/q4k-x4-standalone-compile-20260615-221313/`.
- **Current conclusion:** Do not continue patching this 32x32/wg128 MMQ source
  as the main Phase 2a path. The evidence no longer supports a missing dot form
  or wide-subgroup explanation. The next credible path is a clean HRX1/Vulkan
  style BM64/BN16-or-BN32/TM4/TN1-or-TN2 wg64 route, or a tiny x4-layout
  diagnostic consumer that precisely identifies the remaining lane/layout bug.
- **Direct x4 consumer update:** The tiny x4-layout diagnostic consumer was
  implemented, tested, saved, and reverted. Route
  `hrx2_mul_mat_q4_k_q8_1_x4_direct_cols4_static` consumed packed Q8_1 x4 RHS
  directly, with no MMQ/LDS tiling, and passed all eight model-derived Q4_K
  backend-op rows when compiled with `@hrx2.tuning.workgroup_size=256`.
  Artifact:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-wg256-diag-20260615-222302/`.
  Saved patch:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-passing-diagnostic-20260615-222414/passing-diagnostic.patch`.
  This shifts suspicion away from the x4 quantizer/layout and toward the
  current MMQ consumer's lane mapping, staged metadata use, or schedule
  structure. Do not restore the diagnostic route as a performance candidate; it
  shadows the MMQ route under the same opt-in gate and is direct/slow by
  construction.
- **Dot signedness update:** The Q4_K x Q8_1 Loom dot operations were corrected
  from `vector.dot4i<s8s8>` to `vector.dot4i<u8s8>`, matching unsigned Q4
  codes times signed Q8_1 activations. Control artifact
  `cache/hrx2/phase2a/q4k-u8s8-control-20260615-222811/` passed all eight
  focused rows, so the accepted non-x4 Q8_1 routes tolerate the correction.
  Opt-in x4 artifact `cache/hrx2/phase2a/q4k-u8s8-x4mmq-20260615-222824/`
  still fails the c64 rows with NaNs, so signedness spelling was necessary
  WYSIWYG cleanup, not the root cause of the current MMQ failure.

## HRX2 Profiling: `HRX_PROFILE_MODE=all` Requires Capture Filter

- **Status:** Workflow limitation on current `hrx-system` main.
- **Symptom:** Running llama.cpp HRX2 with `HRX_PROFILE_FILE=...` and
  `HRX_PROFILE_MODE=all` fails during device initialization:
  `AMDGPU executable trace profiling requires a capture filter; use an function
  pattern, command buffer/id, physical device, or queue filter to avoid tracing
  every dispatch`.
- **Artifact:** `cache/hrx2/phase2a/profile-llama32-p512-20260615-191148/`.
- **Current confirmation:** Retested on the Llama 3.2 3B Q4_K_M p512 run after
  the ROPE_SET_ROWS route-domain cleanup. `HRX_PROFILE_MODE=all` still fails
  before HRX2 device registration:
  `cache/hrx2/phase2a/rope-setrows-route512-profile-all-20260615-200107/`.
- **Impact:** Do not use `all` as the default profiling mode for Phase 2a
  model runs. It can make the backend report `invalid device: HRX20` because
  profiling begin failed before the device was available.
- **Missing wrapper control:** IREE exposes capture filter flags such as
  `--device_profiling_filter_function`, `--device_profiling_filter_command_buffer`,
  `--device_profiling_filter_command_index`, physical device, and queue. HRX's
  environment wrapper currently exposes only `HRX_PROFILE_FILE` and
  `HRX_PROFILE_MODE`, so llama.cpp HRX2 cannot request capture-filtered
  executable traces without an HRX wrapper change.
- **Working path:** `HRX_PROFILE_MODE=queue` is safe and captures queue events;
  `HRX_PROFILE_MODE=dispatch` runs on the same p512 case and captures executable
  metadata plus queue-device records, but in this configuration
  `iree-profile dispatch` reports zero dispatch events. Use these modes only
  for queue/device-span evidence, not per-route device timing. Recent queue
  artifact:
  `cache/hrx2/phase2a/rope-setrows-route512-profile-queue-20260615-200237/`.
  For per-route attribution, a diagnostic
  `GGML_HRX2_DISPATCHES_PER_SUBMIT=1` run can align the post-model-load queue
  event suffix with HRX2 route dispatch order, but that is an attribution aid,
  not a production throughput measurement.
- **Rule:** Agents should default to trace JSONL plus queue/dispatch profile
  evidence until a capture-filtered executable/ATT workflow is explicitly set
  up. Do not interpret an `HRX_PROFILE_MODE=all` failure as a model or route
  failure.

## Q4_K BM64/BN8 MMQ: Flattened Loop Lowering and Integer LDS Payload Staging

- **Status:** Active Phase 2a limitation/bug candidate; do not promote the
  diagnostic routes below.
- **Context:** A clean HRX1/Vulkan-inspired Q4_K x packed-Q8_1-x4 route was
  generated as `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x8_static` to replace the
  failed 32x32/wg128 consumer with a BM64/BN8/TM4/wg64 shape.
- **Compiler lowering boundary:** The first spelling flattened Q4 block and
  group into a single `%kb` loop carrying eight or sixteen f32 accumulators.
  `loom-compile --backend=amdgpu-hal` failed in `source-to-low` with
  `AMDGPU branch argument materializer selected for an unsupported type`.
  Reducing BN16 to BN8 did not fix it. Rewriting the loop topology to match
  the old compile-friendly form, outer `%q4_block_iter` plus inner
  `scf.for %group ... unroll(%eight)`, compiled cleanly. Artifacts:
  `cache/hrx2/phase2a/q4k-mmq64x16-compile-repro-20260615-224644/`,
  `cache/hrx2/phase2a/q4k-mmq64x8-compile-repro-20260615-225145/`,
  `cache/hrx2/phase2a/q4k-mmq64x8-nested-group-compile-repro-20260615-225606/`.
- **Correctness boundary:** The nested BM64/BN8 route JIT-compiled and was
  selected by focused backend-op traces, but the fully staged A+B integer
  payload path failed with finite `ERR ~= 1.0`. Removing the Q4_K min
  correction did not materially change the error, so the failure is not
  primarily the dequant correction term.
- **Staging isolation evidence:**
  - A+B payload and Q8 metadata direct from global memory passed all eight
    model-derived Q4_K focused backend-op rows:
    `cache/hrx2/phase2a/q4k-x4-mmq64x8-a-b-global-diag-20260615-230050/`.
  - A staged through LDS with B direct from global failed with
    `ERR ~= 0.985`:
    `cache/hrx2/phase2a/q4k-x4-mmq64x8-a-scalar-lds-b-global-diag-20260615-230303/`.
  - A direct from global with B staged through LDS also failed with
    `ERR ~= 1.003-1.006`:
    `cache/hrx2/phase2a/q4k-x4-mmq64x8-a-global-b-lds-diag-20260615-230410/`.
  - Changing A payload LDS from `vector.store/load vector<1xi32>` to scalar
    `view.store/load i32` did not fix correctness.
- **Performance boundary:** The correct A+B-global diagnostic is not a
  promotable route. On Llama 3.2 3B Q4_K_M p64/n0 it measured about
  `22.39 tok/s` versus `79.58 tok/s` for the accepted non-x4 Q8_1 fallback:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-smoke-20260615-230153/`.
- **Saved patches:** Current diagnostic source and generator patches were
  saved at
  `cache/hrx2/phase2a/q4k-mmq64x8-diagnostic-patches-20260615-230451/`.
- **Current conclusion:** The route/math/lane ownership can be made correct,
  but integer payload reuse through Loom workgroup memory is not currently
  trustworthy for this Q4_K/Q8_1 MMQ shape. Future work should either produce a
  smaller standalone integer-LDS reproducer for the Loom author, or use a
  different staging spelling/low-level path before attempting more performance
  tuning. Do not commit the slow A+B-global diagnostic as a Phase 2a bulk lift.

## HRX2 CONT_SET_ROWS Fusion Route Domain Must Guard Large KV Cache Shapes

- **Status:** llama.cpp-side limitation handled by route applicability guard.
- **Date:** 2026-06-16.
- **Context:** The Phase 2a `CONT -> SET_ROWS` V-cache fusion originally
  selected for a `llama-cli` run with a large default KV cache. That produced a
  JIT compile failure because `@hrx2.shape.set_rows.ne1` was `134217728`, while
  the Loom source declared `range(%value, 1, 1048576)`.
- **Symptom artifact:**
  `cache/hrx2/phase2a/cont-setrows-fusion-cli-smoke-20260615-234931/`.
- **Resolution:** The fusion extractor now declines shapes with
  `set_rows.ne1 > 1048576`, causing large-cache contexts to use the existing
  unfused path instead of failing the graph. The accepted route remains active
  for the phase2a benchmark basket where the configured KV-cache row count is
  within range.
- **Lesson:** Fusions that specialize on view-expanded cache tensors must guard
  every config range before provider selection. Do not rely on Loom compile
  failure as a route-selection filter; the fusion dispatcher has no safe
  automatic unfused fallback once it has skipped the producer node.

## Q5_K Packed-Q8_1 x4 MMQ wg128 Produces NaN While Direct x4 Passes

- **Status:** Active Phase 2a correctness blocker for the Q5_K prompt MMQ
  boulder. Do not promote the wg128 MMQ spelling without a reducer/fix.
- **Date:** 2026-06-16.
- **Context:** HRX1 has a Q5_K/Q8_1-x4 MMQ32x32/wg128 prior with B staged in
  shared memory, one row per `tid & 31`, eight output columns per col lane, and
  `sudot4(false,true)` unsigned-Q5 by signed-Q8 dot4. HRX2 attempted a Loom
  port as `hrx2_mul_mat_q5_k_q8_1_x4_mmq32x32_static`.
- **Failure artifact:**
  `cache/hrx2/phase2a/q5k-x4-mmq32x32-opgate-20260616-005220/`.
- **Observed behavior:** The provider JIT-compiled successfully, selected for
  `q5_K[3072,5120] x f32[3072,512]`, dispatched the existing
  `quantize_q8_1_x4_f32_generic_wg128` RHS quantizer, then failed the
  backend-op CPU reference with `NaN at index 0`.
- **Isolation evidence:** A direct packed-Q8_1-x4 Q5 route with the same Q5
  high-bit packing and RHS x4 quantizer passed the same exported Phi p512 op
  file:
  `cache/hrx2/phase2a/q5k-x4-direct-clean-opgate-20260616-005818/`.
- **Current conclusion:** Q5 high-bit packing and packed x4 RHS layout are
  correct. The NaN is specific to the MMQ schedule, most likely the wg128
  workgroup-memory staging pattern or a schedule/lowering hazard similar to the
  Q4_K packed-MMQ limitations above. Future work should reduce the failing
  MMQ to the smallest B-LDS or accumulator reproducer before spending more
  tuning time on Q5_K.

## Q6_K Packed-Q8_1 x4 Direct Route Is Correct but Not a Final Prompt-MMQ Schedule

- **Status:** llama.cpp route accepted as default-on prompt substrate with
  opt-out rollback; performance limitation remains.
- **Date:** 2026-06-16.
- **Context:** HRX2 added
  `mul_mat_q6_k_q8_1_x4_direct_cols4_k256_32768_r1_262144_c4_512_wg256` to
  consume packed Q8_1 x4 RHS activations for prompt Q6_K matmuls.
- **Correctness evidence:** Focused model-derived Q6 rows passed with the new
  route selected for prompt and the existing rows2 route selected for decode:
  `cache/hrx2/phase2a/q6k-x4-direct-opgate-20260616-011732/`. A default-off
  focused gate selected only existing Q6 routes:
  `cache/hrx2/phase2a/q6k-x4-direct-default-off-opgate-20260616-011921/`.
- **Performance evidence:** With Q5+Q6 x4 direct routes default-on, the
  reduced two-model prefill run improved over the old default by about +6.0%
  to +20.8% with zero CPU compute fallback:
  `cache/hrx2/phase2a/q5-q6-packed-default-on-two-model-20260616-012557/`.
  Focused opt-out coverage is preserved by:
  `cache/hrx2/phase2a/q6k-packed-optout-opgate-20260616-012703/`.
- **Limitation:** The route is direct global-load work, one output row by four
  columns per workgroup. It validates Q6 signed unpacking, Q8_1 x4 layout, and
  `vector.dot4i<s8s8>` spelling, but it adds 31 extra Q8_1 x4 quantize
  dispatches on the Phi prefill path and does not reuse RHS tiles across output
  rows. It should not be treated as the final Q6_K prompt matmul schedule.
- **Next step:** Build a staged MMQ/tiled Q6_K x packed-Q8_1-x4 route using the
  HRX1/Vulkan prompt-matmul priors. If workgroup-memory staging fails the same
  way as the Q4_K/Q5_K MMQ attempts, reduce that as a Loom/compiler issue
  instead of continuing local knob sweeps on the direct route.

## HRX2 JIT Rejects Unused Config Bindings Even When Standalone loom-compile Succeeds

- **Status:** llama.cpp-side authoring limitation with a known fix.
- **Date:** 2026-06-16.
- **Context:** The p512 F16 KQV+CONT fusion route
  `mul_mat_f16_f32_batched_attention_cols8_contiguous_wg256` initially failed
  provider JIT in the model run, while standalone `loom-compile
  --backend=amdgpu-hal --target=gfx1100` with the same visible shape config
  succeeded.
- **Symptom artifact:**
  `cache/hrx2/phase2a/f16-kqv-cont-ab-20260616-015759/`.
- **Standalone compile artifact:**
  `cache/hrx2/phase2a/f16-kqv-cont-standalone-20260616-020155/`.
- **Root cause:** HRX2 JIT sets `LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
  LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED`. The route copied the normal
  F16/F32 matmul specialization bindings and included
  `@hrx2.shape.mul_mat_f16.dst_stride_col`,
  `@hrx2.shape.mul_mat_f16.dst_stride_ne2`, and
  `@hrx2.shape.mul_mat_f16.dst_stride_ne3`, but the contiguous export no
  longer consumed those config keys because it writes dense post-`CONT` layout.
  Standalone `loom-compile` ignores unused direct bindings, so it hid the
  stricter runtime failure.
- **Fix:** Remove unused config bindings from the route metadata. After the
  fix, the provider compiled and selected cleanly:
  `cache/hrx2/phase2a/f16-kqv-cont-after-config-fix-20260616-020955/`.
- **Rule:** Every runtime catalog route must pass exactly the config keys its
  selected Loom root consumes. Do not copy a route's specialization bindings
  after changing layout or address derivation without checking for unused
  keys. If standalone compile succeeds but HRX2 provider JIT fails before
  HSACO load, immediately compare the route binding list against `config.get`
  usage under the selected export.

## Q4_K x4 MMQ Block-Staged RHS Produces NaNs Despite Cleaner Barrier Schedule

- **Status:** Active Phase 2a correctness blocker for the Q4_K tiled prompt
  MMQ path.
- **Date:** 2026-06-16.
- **Context:** The opt-in Q4_K `q8_1_x4_mmq32x32` route originally had a poor
  schedule shape relative to HRX1/Vulkan: many global loads and sixteen
  workgroup barriers per Q4_K block. A diagnostic rewrite staged all eight
  Q8_1 sub-blocks for one Q4_K block into workgroup memory before the dot loop.
- **Positive compile evidence:** The rewrite moved the compile report from
  roughly `global_memory=348, barrier=16, local=1152` to
  `global_memory=156, barrier=2, local=9216`, proving that the WYSIWYG staging
  spelling affected the intended schedule class.
- **Failure evidence:**
  - `cache/hrx2/phase2a/q4k-x4-mmq32x32-blockstaged-model-opgate-20260616-023618/`
  - `cache/hrx2/phase2a/q4k-x4-mmq32x32-blockstaged-stageunroll-opgate-20260616-023736/`
  - rejected patch:
    `cache/hrx2/phase2a/q4k-x4-mmq32x32-blockstaged-rejected-20260616-023755/`
- **Observed behavior:** The provider selected and dispatched with no
  `provider_unavailable` fallback, but three model-derived Q4_K rows returned
  NaNs at index 0. Unrolling the staging loop did not repair correctness.
- **Current conclusion:** This is not a catalog route-selection issue. It is a
  Loom/source-lowering/staging issue in the workgroup-memory MMQ spelling, or a
  remaining source-level hazard not exposed by the direct packed route. Keep the
  production source on the accepted non-x4 Q8_1 direct route. Future work should
  build a smaller reducer around staged Q8_1 payload plus f16 `d/s` metadata, or
  move to a low-level/bridge spelling that can match the HRX1/Vulkan tiled
  packed-MMQ schedule without this correctness failure.

## test-backend-ops Perf CSV Needed Timing Fields and Runtime Caps

- **Status:** Fixed in llama.cpp test tooling.
- **Date:** 2026-06-16.
- **Context:** Phase 2a needs focused backend-op timing before full integration
  runs. `test-backend-ops perf` computed useful timing fields but omitted them
  from CSV, and its default one-second-per-case plus large graph duplication
  made model-derived op files too slow for candidate A/Bs.
- **Fix:** CSV now includes `time_us`, `flops`, `bandwidth_gb_s`, `memory_kb`,
  and `n_runs`. The env vars `GGML_TEST_BACKEND_OPS_PERF_MIN_US` and
  `GGML_TEST_BACKEND_OPS_PERF_MAX_RUNS` control the perf timing window and
  graph duplicate count.
- **Validation artifact:**
  `cache/hrx2/phase2a/test-backend-ops-perf-csv-capped-20260616-024525/`.
- **Usage rule:** For HRX2 kernel candidate loops, prefer a small model-derived
  op file plus `GGML_TEST_BACKEND_OPS_PERF_MIN_US=10000` and
  `GGML_TEST_BACKEND_OPS_PERF_MAX_RUNS=16` to get quick relative timings. Keep
  full-model HRX2/Vulkan runs for accepted batches, because this perf mode still
  measures graph-compute time around duplicated ops rather than a pure hardware
  timestamp per provider.

## Q4_K x4 MMQ Payload LDS Is Correct, But f16 Q8 Metadata LDS Is Unsafe In Current Spelling

- **Status:** Narrowed Phase 2a limitation. Keep Q4_K x4 MMQ opt-in until the
  Q8_1 x4 backplane is faster or a shape-specific route table proves net wins.
- **Date:** 2026-06-16.
- **Context:** Earlier Q4_K `q8_1_x4_mmq32x32` attempts that staged both packed
  Q8 payload and f16 Q8_1 `d/s` metadata through Loom workgroup memory failed
  focused model-derived backend-op rows with NaNs.
- **New isolation evidence:** Reading Q8 `d/s` metadata directly from global
  packed-x4 storage while staging only the packed integer payload in LDS passes
  the same rows:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-global-ds-opgate-20260616-025517/`.
  After removing the unused f16 LDS staging entirely, the cleaned opt-in route
  also passed:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-payloadlds-globalds-clean-20260616-025707/`.
- **Performance evidence:** The cleaned route is a 3x-class backend-op win on
  Q4 prompt rows in both p64 and p512 exported op buckets, but model-level
  A/B remains mixed:
  - full-op perf:
    `cache/hrx2/phase2a/q4k-p64-p512-fullop-perf-ab-20260616-030132/`;
  - reduced one-repetition model A/B:
    `cache/hrx2/phase2a/q4k-x4-mmq32x32-globalds-prefill-ab-20260616-025741/`;
  - repeated Llama A/B:
    `cache/hrx2/phase2a/q4k-x4-mmq-repeat3-20260616-030241-x4/`.
- **Current conclusion:** Do not describe this as a generic workgroup-memory
  failure. Packed integer payload LDS staging is now correctness-clean. The
  unsafe piece is f16 Q8_1 metadata staging in the current Loom/source spelling
  or lowering. Separately, the x4 model-level path is limited by quantizer and
  backplane amortization; p512 Llama is only a small win and p64 Llama remains
  negative even after repeated runs.
- **Workaround/current policy:** The source keeps the cleaned opt-in
  payload-LDS/global-metadata route for continued tuning. The default remains
  the non-x4 Q8_1 cols4 route. Future work should either make x4 quantization
  much cheaper/fused/reused, or add data-driven shape selection that avoids
  p64 while preserving any proven p512+ wins.

## Q8_1 x4 Quantizer Is Not Compiler-Limited; Backplane Reuse Is The Limitation

- **Status:** Active Phase 2a architecture/performance limitation.
- **Date:** 2026-06-16.
- **Context:** After the cleaned Q4_K x4 MMQ route became correctness-clean and
  faster at backend-op level, model traces moved the visible blocker to
  `quantize_q8_1_x4_f32_generic_wg128`.
- **Compile/ISA evidence:**
  `cache/hrx2/phase2a/q4k-x4-evidence-reports-20260616-030624/`.
  The x4 quantizer reports 168 instructions, peak live units 18, no spills, no
  private memory, and no local memory. ISA shows one `global_store_b8` payload
  store per lane and two f16 metadata stores per subgroup. This is WYSIWYG but
  not obviously bad compiler codegen.
- **Trace evidence:** The existing quantized-RHS cache is one-entry last-use.
  Llama gets some adjacent reuse, but Phi gets zero cache hits in the reduced
  p64/p512 basket. Enabling Q4 x4 MMQ changes layout requests enough to reduce
  Llama quantizer dispatches, but still leaves 83 x4 quantize dispatches for a
  single prefill and does not improve Phi.
- **Current conclusion:** Do not spend the next pass trying to shave a few
  instructions from the quantizer kernel. The structural problem is repeated
  per-consumer RHS quantization and incompatible per-route layout requests.
  Future work should design either a multi-entry/per-graph quantized-RHS cache,
  a producer fusion, or a planner-level activation-cluster layout decision.
  A multi-entry cache is not a trivial one-line extension because the current
  cache owns one reusable scratch buffer; multiple live cached RHS values need
  retained buffers or scratch-arena offsets with graph lifetime handling.

## Q4_K x4 MMQ32x32 Current Source Fails Re-Gate; Q8 Scale Path Produces NaNs

- **Status:** Active correctness blocker for the Q4_K tiled prompt MMQ path.
  This supersedes the earlier “payload LDS is correct” acceptance claim until a
  new clean gate proves otherwise.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`.
- **Observed case:** Focused model-derived Q4_K p512 op gate with
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`.
- **Fresh evidence:** Current committed source selects the x4 MMQ32x32 route
  and returns NaNs:
  `cache/hrx2/phase2a/q4k-x4-current-regate-20260616-032128/`.
  Therefore the older
  `q4k-x4-mmq32x32-global-ds-opgate-20260616-025517/` and
  `q4k-x4-mmq32x32-payloadlds-globalds-clean-20260616-025707/` artifacts are
  not valid acceptance evidence for the current source.
- **Isolation evidence:** Temporary diagnostics are saved under
  `cache/hrx2/phase2a/q4k-x4-current-diagnostics-20260616-032727/` and
  individual artifacts:
  - scalar i32 LDS payload spelling:
    `q4k-x4-scalar-lds-current-20260616-032241/`, NaNs remain;
  - direct global Q8 payload with no LDS stores/barriers:
    `q4k-x4-direct-payload-nolds-current-20260616-032515/`, NaNs remain;
  - Q8 sum correction forced to zero:
    `q4k-x4-zero-bs-nolds-current-20260616-032553/`, NaNs remain;
  - Q8 scale forced to one and sum forced to zero:
    `q4k-x4-one-bd-zero-bs-nolds-current-20260616-032633/`, NaNs disappear and
    become finite expected mismatches (`ERR ~= 0.996`).
  - packed i32 metadata load plus bitcast/extract:
    `q4k-x4-packed-ds-i32-opgate-20260616-033311/`, NaNs remain;
  - explicit bounded `index.assume` facts for all Q8 metadata f16 indices:
    `q4k-x4-ds-index-assume-opgate-20260616-033629/`, NaNs remain;
  - direct x4 cols4 consumer re-applied against the current quantizer/runtime:
    `q4k-x4-direct-cols4-current-quantizer-opgate-20260616-033411/`, passes
    with the direct route selected.
- **Current conclusion:** The NaN source is the Q8 scale (`d`) load/use path in
  the current MMQ32x32 route spelling. It is not isolated to Q8 sum correction,
  vector-vs-scalar LDS, LDS staging, packed-vs-f16 metadata load width, or
  missing f16 metadata index range facts. The direct x4 consumer passes with the
  current quantizer/runtime, so the basic x4 quantizer layout is not disproven.
- **Workaround/current policy:** Keep Q4_K x4 MMQ disabled by default. Do not
  benchmark, tune, or promote this route until a focused backend-op gate passes
  with the provider selected. The next useful artifact is a minimal reducer for
  the MMQ32x32 Q8 scale-load pattern, or a cleaner HRX1/Vulkan-shaped rewrite
  validated from correctness before adding staging/perf work.

## Workgroup Memory Is Not Categorically Blocked For Packed MMQ

- **Status:** Clarification from Phase 2a Q6_K work.
- **Date:** 2026-06-16.
- **Evidence:** The accepted Q6_K x Q8_1 x4 MMQ32x32 Loom route stages the
  packed RHS payload and Q8 scale metadata through workgroup memory and passes
  focused model-derived backend-op rows:
  `cache/hrx2/phase2a/q6-mmq32x32-fix-opgate-20260616-040708/`.
- **Current conclusion:** Do not generalize the Q4_K x4 MMQ NaN issue into a
  blanket Loom LDS/workgroup-memory bug. Q6_K proves the basic workgroup-memory
  path can be correctness-clean for packed RHS tiles. The remaining Q4_K issue
  is specific to that route/source shape, metadata path, or lowering context
  until a standalone reducer proves a broader compiler defect.

## Q5_K MMQ32x32 Needs Packed-Word QL/QH Decode And f32 LDS Metadata

- **Status:** Active authoring constraint for HRX2 packed K-quant MMQ routes.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q5_k_f32.loom`,
  export `hrx2_mul_mat_q5_k_q8_1_x4_mmq32x32_static`.
- **Evidence:** Accepted gate and perf:
  `cache/hrx2/phase2a/q5-mmq32x32-final-gates-20260616-053952/`.
  Evidence dump:
  `cache/hrx2/phase2a/q5-mmq32x32-evidence-20260616-054240/`.
- **Problematic spellings found:**
  - Per-byte dynamic Q5 low-nibble shifts compiled but failed strict
    CPU-reference correctness with finite `ERR ~= 1.1`.
  - `scf.if` yielding i32 nibble values in the larger looped MMQ shape failed
    source-to-low with `AMDGPU branch argument materializer selected for an
    unsupported type`; the passing CSV from that run was fallback, not the
    candidate provider.
  - `scf.select` and branchless arithmetic nibble selection compiled but
    produced NaNs in this MMQ source shape.
  - f16 Q8_1 `d/s` values staged through LDS compiled but failed strict
    correctness with finite `ERR ~= 0.55`.
- **Accepted spelling:**
  - Decode Q5 `qs` by loading four low-nibble bytes as one aligned i32, shifting
    the packed word by `0` or `4`, masking `0x0f0f0f0f`, and unpacking byte
    lanes.
  - Decode Q5 `qh` by loading four high-bit bytes as one aligned i32, shifting
    by the group index, masking `0x01010101`, shifting into bit 4, and unpacking
    byte lanes.
  - Stage Q8_1 `d/s` metadata in LDS as f32 rather than f16 for this route.
- **Current conclusion:** For packed K-quant MMQ authoring, prefer packed-word
  bit extraction over per-byte dynamic shifts or integer CFG/selects. Treat f16
  LDS metadata as suspect in larger MMQ source shapes until a smaller reducer
  proves whether this is a Loom lowering bug or an authoring constraint.

## Direct Fused Epilogues Are Shape-Regime Specific

- **Status:** Active HRX2 route-selection constraint.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q4_k_swiglu_f32.json`.
- **Evidence:** Same-binary A/B after Q4/Q5/Q6 prompt MMQ acceptance:
  `cache/hrx2/phase2a/q4k-swiglu-current-default-20260616-061015/` versus
  `cache/hrx2/phase2a/q4k-swiglu-current-disabled-20260616-061049/`. Disabling
  the direct Q4_K SWIGLU fusion improved p512 from about `95-105` tok/s to
  about `618-632` tok/s on the reduced Phi/Llama slice.
- **Root cause:** The fused route removed dispatches but preserved a direct
  Q4_K x F32 RHS schedule. Once the unfused path had packed Q4_K x Q8_1 x4
  MMQ prompt routes, the "unfused" composition became much faster for prompt
  despite extra dispatches and the standalone SWIGLU epilogue.
- **Current policy:** Keep the direct Q4_K SWIGLU fused routes single-column
  only. Prompt SWIGLU should remain separate until a true packed-MMQ SWIGLU
  fusion exists and beats the separate packed matmuls plus SWIGLU for the
  target shape bucket.
- **Process rule:** Fusion acceptance is per shape regime and per best
  available route composition. Do not promote a fusion only because it reduces
  dispatch count or beats an older weaker unfused baseline.

## F16 Attention Rows2 Probe Shows KQ Needs True Tiled Matmul

- **Status:** Active Phase 2a performance limitation with accepted intermediate
  mitigation.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_f16_f32_batched.loom`,
  export `hrx2_mul_mat_f16_f32_batched_rows2_cols8`.
- **Evidence:** Focused gate
  `cache/hrx2/phase2a/f16-rows2-cols8-wg128-opgate-20260616-063143/`;
  perf gate
  `cache/hrx2/phase2a/f16-rows2-cols8-wg128-perf-20260616-063202/`;
  reduced model rerun
  `cache/hrx2/phase2a/f16-rows2-cols8-wg128-reduced-20260616-063239/`.
- **Finding:** Row tiling over the KQ p512 shape is profitable only when the
  workgroup size matches the static K dimension. A WG256 rows2 probe passed
  correctness but was mixed because `k=128` left half the workgroup idle while
  doubling per-workgroup reductions. The accepted WG128 route cuts p512 KQ
  rows from roughly 4.8-4.9 ms to 2.0-2.7 ms and improves p512 model
  throughput by about 7-8%.
- **Remaining limitation:** This is still not the known-good Vulkan schedule.
  Vulkan uses a true tiled matmul family with many output rows/columns per
  workgroup and shared staging. HRX2 still has separate KQ, SOFT_MAX, KQV, and
  layout traffic under `--flash-attn 0`, and KQV remains on the older cols8
  route. The next attention pass should either implement a proper tiled
  F16/F32 matmul in Loom or replace the chain with a fused attention route.
- **Process rule:** For F16 attention prompt shapes, do not keep adding local
  row/column-count variants after the cheap KQ row-tiling win. The schedule
  gap to Vulkan is now clearly algorithmic.

## F16 Attention KQV Rows2 Contiguous Probe Did Not Move Model Throughput

- **Status:** Rejected Phase 2a tuning path.
- **Date:** 2026-06-16.
- **Affected attempted source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_f16_f32_batched.loom`,
  attempted export `hrx2_mul_mat_f16_f32_batched_rows2_cols8_contiguous`.
- **Evidence:** Raw attention op gate
  `cache/hrx2/phase2a/f16-kqv-rows2-cols8-wg256-opgate-trace-20260616-064358/`
  passed, but did not exercise the fused contiguous provider. Real graph route
  proof in
  `cache/hrx2/phase2a/f16-kqv-rows2-cols8-wg256-phi-p512-smoke-20260616-064429/`
  selected the attempted route 32 times. Reduced comparison in
  `cache/hrx2/phase2a/f16-kqv-rows2-cols8-wg256-reduced-20260616-064458/`
  showed p512 moving only from `674.326` to `677.450` tok/s for Phi and from
  `690.675` to `688.392` tok/s for Llama 3.2, while Vulkan also moved within
  one-run noise.
- **Current conclusion:** KQV is still far behind Vulkan at the focused-op
  level, but simply copying the KQ rows2 dot-per-output idea into the
  contiguous KQV fusion is not enough. Treat further attention work as a
  schedule-family problem: implement a true tiled F16/F32 matmul or a fused
  attention chain. Do not reintroduce local rows2 KQV variants unless compile
  reports/ISA show a materially different schedule.

## Q4_K A-Side Workgroup Staging Still Fails Correctness In MMQ32x32

- **Status:** Active Loom/source-spelling limitation for Q4_K prompt MMQ.
- **Date:** 2026-06-16.
- **Affected attempted source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`.
- **Evidence:** Scalar staged payload attempt:
  `cache/hrx2/phase2a/q4-a-staged-opgate-20260616-071646/`; vector
  staged payload attempt:
  `cache/hrx2/phase2a/q4-a-staged-vector-opgate-20260616-071840/`; rejected
  patch:
  `cache/hrx2/phase2a/q4-a-staged-rejected-20260616-071952/q4-a-staged-rejected.patch`.
- **Finding:** Staging decoded Q4_K A payload plus f32 scale/min metadata into
  workgroup memory compiled and selected the intended provider, but all Q4_K
  rows in the focused K-quant gate failed strict CPU-reference checks with
  small finite errors. Q5_K and Q6_K rows in the same gate stayed correct, and
  Q6_K already stages packed RHS payloads successfully.
- **Current conclusion:** Do not assume high-level A-side integer/f32 LDS
  staging is correctness-clean for Q4_K MMQ32x32. The schedule direction is
  still prior-backed because Vulkan stages A and B tiles, but future work needs
  a minimal reducer or a lower-level spelling for the Q4_K A tile before this
  optimization can be promoted.

## Q6_K Output Projection Benefits Slightly From Larger Row Tiles

- **Status:** Accepted HRX2 Phase 2a route constraint.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q6_k_f32.loom`,
  export `hrx2_mul_mat_q6_k_q8_1_x4_mmq64x32_static`.
- **Evidence:** Focused gate/perf:
  `cache/hrx2/phase2a/q6-mmq64x32-final-opgate-20260616-072527/`; reduced
  basket:
  `cache/hrx2/phase2a/q6-mmq64x32-final-reduced-20260616-072605/`.
- **Finding:** Doubling the Q6_K x Q8_1 x4 MMQ row tile from 32 to 64 rows
  and using WG256 improved the large output projection rows by about 3-4% and
  moved p512 model throughput by about 0.7-1.3%, while small-row Q6 FFN rows
  were mixed.
- **Current policy:** Keep the Q6_K packed prompt route at MMQ64x32/WG256 for
  now. This is a small lift, not a proof that row-tile increases are generally
  profitable; use focused row evidence before applying the same knob to Q4_K or
  Q5_K.

## Q4_K Packed Prompt MMQ Also Benefits Slightly From 64-Row Tiles

- **Status:** Accepted HRX2 Phase 2a route constraint, not a final Q4_K
  prompt-matmul solution.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static`.
- **Evidence:** Focused gate/perf:
  `cache/hrx2/phase2a/q4-mmq64x32-final-opgate-20260616-073350/`; reduced
  basket:
  `cache/hrx2/phase2a/q4-mmq64x32-final-reduced-20260616-073443/`.
- **Finding:** Doubling the Q4_K x packed-Q8_1 x4 prompt MMQ row tile from
  32 to 64 rows and using WG256 passed all hot K-quant backend-op rows and
  improved the reduced p64/p512 basket by roughly 1-3%. Focused op rows were
  mixed: `Qcur` and `ffn_up` improved, while `ffn_gate` and `ffn_out`
  regressed slightly.
- **Current policy:** Keep the Q4_K packed prompt route at MMQ64x32/WG256 for
  now. This preserves a small measured lift, but the real remaining boulder is
  still the schedule-family gap versus HRX1/Vulkan: cooperative A-side Q4_K
  staging, fewer repeated Q4_K decodes, and/or a lower-level spelling that can
  reproduce the known-good tiled schedule without correctness failures.

## Q5_K Packed Prompt MMQ64x32 Row-Tile Probe Regressed

- **Status:** Rejected Phase 2a tuning path.
- **Date:** 2026-06-16.
- **Affected attempted source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q5_k_f32.loom`,
  attempted schedule change inside
  `hrx2_mul_mat_q5_k_q8_1_x4_mmq32x32_static`.
- **Evidence:** Focused gate/perf:
  `cache/hrx2/phase2a/q5-mmq64x32-wg256-opgate-20260616-073750/`; rejected
  patch:
  `cache/hrx2/phase2a/q5-mmq64x32-rejected-20260616-073834/q5-mmq64x32-rejected.patch`.
- **Finding:** The Q5_K WG256/MMQ64x32 probe passed correctness but regressed
  the hot `wqkv` row from about `2172 us` to about `2247 us`.
- **Current policy:** Do not blindly apply the Q4/Q6 64-row tile to Q5_K.
  Keep Q5_K on MMQ32x32/WG128 and require a schedule-specific prior or compile
  report/ISA finding before changing its tile shape again.

## Q4_K A-Side Staging Is Correctness-Clean In The 64-Row Spelling

- **Status:** Repaired Phase 2a limitation for the default Q4_K packed prompt
  route.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static`.
- **Evidence:** Correctness probe:
  `cache/hrx2/phase2a/q4-a-staged64-optest-20260616-074238/`; final
  focused gate/perf:
  `cache/hrx2/phase2a/q4-a-staged64-final-opgate-20260616-074455/`; reduced
  basket after commit `dcb8958c5`:
  `cache/hrx2/phase2a/q4-a-staged64-final-reduced-20260616-074651/`.
- **Finding:** The earlier A-side staging rejection was not a fundamental
  Loom blocker. Reapplying the schedule to the accepted MMQ64x32/WG256 route
  and sizing the staged A tile for 64 rows (`512xi32` payload plus `64xf32`
  scale/min vectors) passed all hot K-quant backend-op rows and improved Q4_K
  prompt op timings by about 5-10%.
- **Current policy:** Keep A-side staging in the default Q4_K packed prompt
  route. Do not reuse the old MMQ32x32 rejection as evidence that workgroup
  staging is unsafe; it was a stale spelling and tile-size problem. Still use
  focused backend-op correctness for any further A/B staging changes.

## Q6_K A-Side Staging Did Not Transfer From Q4_K

- **Status:** Rejected Phase 2a tuning path with one correctness concern.
- **Date:** 2026-06-16.
- **Affected attempted source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q6_k_f32.loom`,
  export `hrx2_mul_mat_q6_k_q8_1_x4_mmq64x32_static`.
- **Evidence:** Payload plus scale staging gate:
  `cache/hrx2/phase2a/q6-a-staged64-opgate-20260616-080255/`; payload-only
  staging gate:
  `cache/hrx2/phase2a/q6-a-payload-staged64-opgate-20260616-080424/`;
  guarded baseline proof:
  `cache/hrx2/phase2a/q6-rowguard-baseline-opgate-20260616-080608/`.
- **Finding:** Staging both decoded Q6 payload and f32 per-half scale in LDS
  compiled and selected the intended provider, but failed one large
  `result_output` row by `ERR=0.000646920 > 0.0005`. Staging only the Q6
  payload passed all hot rows but regressed the large output rows to about
  `63.6 ms` and `99.9 ms`, compared with the guarded baseline at about
  `60.3 ms` and `95.3 ms`.
- **Current policy:** Do not assume Q4_K's successful A-side reuse pattern
  transfers to Q6_K. The next Q6_K output-projection optimization should return
  to the HRX1 MMQL prior or a lower-level tiled schedule that changes lane
  ownership and output reuse, not just LDS-caching the current one-row/eight
  column-lane schedule.
- **Retained guard:** The Q6_K MMQ64x32 route now has `rows_multiple_of: 64`,
  backed by generic catalog validator/parser/dispatcher support. This prevents
  partial-row shapes from entering a kernel whose barriers are inside a
  per-row bounds region.

## Direct HIP Bridge Kernels Should Use HRX2 u32 Shape ABI Wrappers

- **Status:** Active HRX2 bridge/refutation guidance.
- **Date:** 2026-06-16.
- **Affected source:** Direct embedded `amdgpu-hsaco` bridge routes under
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/`.
- **Evidence:** Raw HRX1 Q6_K HIP export
  `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_f32` loaded through HRX2
  direct-HSACO routing and matched the reported 3-buffer/3-scalar ABI, but
  focused backend-op rows failed with NaNs/Infs:
  `cache/hrx2/phase2a/q6-hip-bridge-opgate-20260616-085933/`.
  A llama.cpp-local wrapper export using `uint32_t k, rows, cols` and calling
  the same HRX1 device implementation passed the same focused gate and
  produced the expected speedup:
  `cache/hrx2/phase2a/q6-hip-bridge-u32-opgate-20260616-090550/`.
- **Finding:** For bridge/refutation kernels, do not route raw legacy HRX1 HIP
  exports with 64-bit by-value shape parameters unless there is a focused
  reducer proving that ABI path. Use explicit HRX2 wrapper exports with the
  same 12-byte u32 shape constants as Loom mul-mat routes. This avoids mixing
  legacy HIP by-value packing assumptions with HRX2 catalog dispatch.
- **Current policy:** Keep bridge kernels local to llama.cpp unless they become
  general HRX-system APIs. Treat direct HSACO loading as useful production
  infrastructure, but treat raw external HIP ABI reuse as diagnostic-only until
  backend-op correctness proves it.

## Q5/Q6 HRX1 HIP Bridge Routes Need Catalog Workgroup Size

- **Status:** Repaired in llama.cpp Phase 2a; retained as launch-contract
  guidance.
- **Date:** 2026-06-16.
- **Affected sources:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q5_k_q8_1_wave64.hip.cpp`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp`.
- **Evidence:** Initial u32 wrapper gates appeared correct and fast:
  `cache/hrx2/phase2a/q6-hip-bridge-u32-opgate-20260616-090550/` and
  `cache/hrx2/phase2a/q5q6-hip-bridge-opgate-20260616-091045/`. A later
  standalone isolation failed repeatedly:
  `cache/hrx2/phase2a/q4-a-pack2-isolation-20260616-092634/q5q6-only/` and
  `.../q5q6-only-second/`. Disabling x4 prompt bridge routes made the same
  rows pass on safe Loom routes:
  `cache/hrx2/phase2a/q5q6-fallback-check-20260616-092804/`.
- **Finding:** The schedule and Q8_1_x4 physical layout were not the root
  problem. The HRX2 direct-HSACO dispatch path trusted
  `export_info.workgroup_size[0]` before route metadata. The embedded HIP
  bridge HSACOs reported local size `1`, while the catalog route specified
  `256`. Launching the HRX1-derived kernels with one workitem left most LDS
  tile entries uninitialized and produced NaN/Inf mismatches. For `_hip_`
  bridge routes, llama.cpp now dispatches with the catalog workgroup size.
- **Accepted evidence:** Old failing rows fixed:
  `cache/hrx2/phase2a/q5q6-bridge-wgfix2-opgate-20260616-094441/`; final
  mixed K-quant gate:
  `cache/hrx2/phase2a/q5q6-bridge-final-opgate-20260616-094957/`; rollback
  gate:
  `cache/hrx2/phase2a/q5q6-bridge-final-disable-opgate-20260616-095030/`;
  reduced default HRX2/Vulkan run:
  `cache/hrx2/phase2a/q5q6-bridge-default-reduced-20260616-094832/`.
- **Current policy:** Q5/Q6 HIP bridge routes are default-enabled and can be
  disabled with `GGML_HRX2_DISABLE_Q5_Q6_HIP_BRIDGE_PROMPT=1`. Future
  direct-HSACO bridge routes must either use catalog local-size metadata for
  dispatch or reject/log export metadata that disagrees with the catalog.

## Q4_K Vulkan-Style Packed A Payload Is Correctness-Clean And Faster

- **Status:** Accepted HRX2 Phase 2a Q4_K packed prompt route improvement.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static`.
- **Evidence:** Schedule ledger:
  `docs/loom/llamacpp-hrx2-q4k-schedule-ledger.md`; focused gate:
  `cache/hrx2/phase2a/q4-a-pack2-safe-routes-opgate-20260616-093021/`;
  final rebuilt-source gate:
  `cache/hrx2/phase2a/q4-a-pack2-final-opgate-20260616-093421/`;
  focused perf:
  `cache/hrx2/phase2a/q4-a-pack2-safe-routes-perf-20260616-093114/`;
  reduced basket:
  `cache/hrx2/phase2a/q4-a-pack2-safe-routes-reduced-20260616-093151/`.
- **Finding:** Packing two Q4_K nibble groups into one staged A payload word,
  matching Vulkan's `vals0 | (vals1 << 4)` layout, passed all hot K-quant rows
  and improved the four focused Q4_K rows by roughly 13-16%.
- **Current policy:** Keep the packed-A spelling. This is a valid example of
  bracketed prior-driven probing: it changed one axis around the documented
  Vulkan schedule family and was screened at the backend-op level before the
  reduced model benchmark.

## Q4_K HRX1 HIP Bridge Is The Current Prompt Route To Beat

- **Status:** Accepted HRX2 Phase 2a target-specific bridge route.
- **Date:** 2026-06-16.
- **Affected source:**
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q4_k_q8_1_wave64.hip.cpp`.
- **Evidence:** Focused correctness:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-opgate-20260616-100100/`; focused
  perf:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-perf-20260616-100151/`; default gate:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-default-opgate-20260616-100354/`;
  rollback gate:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-disable-opgate-20260616-100430/`;
  reduced model run:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-optin-reduced/`.
- **Finding:** A dense prompt wrapper of the HRX1 Q4_K packed-Q8_1/x4
  MMQL64x32 wave64 schedule passed all focused mixed K-quant rows and improved
  the four hot Q4_K rows by about `1.9x-2.5x` versus the previous accepted Loom
  MMQ64x32 route. Reduced prefill moved from roughly `0.14x-0.22x` Vulkan to
  roughly `0.20x-0.33x` Vulkan on the two-model p64/p512 slice.
- **Current policy:** The bridge is default-enabled for the gfx1100 HSACO route
  and can be disabled with `GGML_HRX2_DISABLE_Q4_HIP_BRIDGE_PROMPT=1`. Any new
  Q4_K Loom route must beat this bridge, not the older Loom MMQ route. Future
  direct-HSACO bridge routes must use HRX2 u32 shape ABI wrappers, catalog
  workgroup size, focused backend-op correctness/perf, route-trace proof, and a
  rollback env before default promotion.
- **Loom authoring note:** This does not prove Loom cannot express the schedule;
  it proves the current high-level Loom spelling was not yet saying the same
  thing as the HRX1 schedule. A Loom rewrite should preserve the same concrete
  schedule facts first: BM64/BN32, wave64, four rows by eight columns per lane,
  explicit packed Q4/Q8 load widths, staged A/B tiles, and unsigned-Q4 by
  signed-Q8 dot form.

## F16 Attention Standalone Tiled Matmul Bridge Was Rejected

- **Status:** Rejected HRX2 Phase 2a attention probe.
- **Date:** 2026-06-16.
- **Artifacts:** KQ failing gate
  `cache/hrx2/phase2a/f16-hip-tile-opgate-20260616-102340/`; KQ writeback fix
  still failing
  `cache/hrx2/phase2a/f16-hip-tile-opgate-20260616-102446/`; KQV non-fused
  math gate failing
  `cache/hrx2/phase2a/f16-hip-tile-kqv-opgate-20260616-102730/`.
- **Finding:** A simple exact-shape HIP tiled-GEMM bridge for p512 attention
  selected correctly but failed CPU-reference for both KQ and KQV. The KQ live
  tensor has non-obvious destination strides, and fixing the first contiguous
  writeback mistake did not make the route correct. KQV then failed with the
  same shared-tile dataflow, so this is not a defaultable bridge shortcut.
- **Current policy:** Do not spend more Phase 2a time on standalone F16
  rows/cols variants around the current dot-per-output Loom matmul unless a
  fresh prior explains the exact tensor-layout contract. The better boulder is
  fused streaming attention: use Vulkan `flash_attn.comp` online-softmax
  dataflow and HRX1 gfx11 flash-attention lane/output ownership, targeting
  D=128 p512 first and treating p64 separately.
## Rejected Q6_K HRX1 BM128/BN64 Bridge Pivot

Date: 2026-06-16.

After the accepted Q4/Q5/Q6 HIP bridge work, a fresh reduced run from commit
`e909aef98` was captured at
`cache/hrx2/phase2a/current-fresh-20260616-103513/`:

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | CPU compute | Dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | `328.750` | `1447.030` | `0.2272` | 0 | 735 |
| `phi4-mini-q4` | p512/n0 | `1447.725` | `4300.615` | `0.3366` | 0 | 703 |
| `llama32-3b-q4` | p64/n0 | `324.308` | `1547.918` | `0.2095` | 0 | 698 |
| `llama32-3b-q4` | p512/n0 | `1507.337` | `4815.075` | `0.3130` | 0 | 670 |

A focused hot-op rerun showed the largest Q6_K output rows remain expensive:
`cache/hrx2/phase2a/current-hot-op-perf-20260616-103627/`.

A bounded schedule pivot exposed the sibling HRX1 Q6_K packed-Q8_1 schedule
`BM128/BN64` (`hrx_mul_mat_vec_q6_k_q8_1_x4_mmql_wg256_impl<128,64>`) in HRX2
alongside the current accepted `BM64/BN128` route. This was a bracket around an
existing HRX1 prior, not a blind schedule guess.

Focused correctness passed all p512 K-quant rows plus p64 Q5/Q6 rows:
`cache/hrx2/phase2a/q6-mmql128x64-opgate-20260616-103911/`. The new route
selected for six Q6 rows and had no provider-unavailable events.

Focused perf rejected it as a production default:
`cache/hrx2/phase2a/q6-mmql128x64-perf-20260616-103957/`.

| Row | Current BM64/BN128 | Candidate BM128/BN64 | Change |
| --- | ---: | ---: | ---: |
| Q6_K result_output rows200064 cols512 | `80147.9 us` | `89848.9 us` | `0.89x` |
| Q6_K result_output rows128256 cols512 | `48445.3 us` | `55133.6 us` | `0.88x` |
| Q6_K ffn_out rows3072 cols512 | `2689.8 us` | `2858.9 us` | `0.94x` |
| Q6_K result_output rows200064 cols64 | `10962.4 us` | `8899.1 us` | `1.23x` |
| Q6_K ffn_out rows3072 cols64 | `511.1 us` | `740.9 us` | `0.69x` |

Decision: reject and remove the route. The p64 huge-output win is too narrow to
justify a production-catalog entry while p512 regresses the dominant output
projection rows. Keep the current `BM64/BN128` Q6 bridge. Future Q6 work needs a
new prior or deeper schedule change, not this adjacent pivot.

## Q4_K Pack2 BK_STEP4 Is A Schedule Refutation, Not A Loom Bug

- **Status:** Rejected HRX2 Phase 2a schedule pivot.
- **Date:** 2026-06-16.
- **Artifact:** `cache/hrx2/phase2a/q4-pack2-bkstep4-opgate-20260616-122640/`.
- **Finding:** An opt-in HIP bridge variant kept the accepted Q4_K
  BM64/BN32 wave64 pack2 topology but staged four Q8 blocks per barrier to
  bracket the Vulkan `BK_STEP=4` prior. The route selected with no
  provider-unavailable events and passed p64 backend-op correctness, but p64
  focused timings regressed about `5.8x-11.8x` versus the accepted pack2 route.
- **Conclusion:** Do not report this as a Loom limitation. `BK_STEP=4` is not
  profitable inside the current single-wave BM64/BN32 pack2 dataflow. Future
  Q4_K work should move to a genuinely different Vulkan-style tile/ownership
  family if pursuing the remaining gap.

## Backend Library Contamination Can Invalidate HRX2/Vulkan Comparisons

- **Status:** Benchmark harness limitation, not a Loom compiler bug.
- **Date:** 2026-06-16.
- **Invalid artifact:**
  `cache/hrx2/phase2a/repeated-prefill-hrx2-vulkan-20260616-123059/`.
- **Finding:** A one-off repeated prefill runner appeared to show HRX2 and
  Vulkan at parity, but the saved `bench.json` files under the `vulkan/`
  directory report `backends=HRX2`. The runner used a mixed `LD_LIBRARY_PATH`
  with HRX2 libraries before Vulkan libraries, so the comparison was not
  apples-to-apples.
- **Current control:** Use `tools/hrx2_phase2a_benchmark.py`, which sets
  backend-specific library paths and now reports cold and steady-state samples.
  Verify `backends` in every `llama-bench` JSON before using a run for KPI
  decisions.

## HIP Wave32 Flag And Broad-Wave32 Q4 Scope

- **Status:** Build/tooling note plus schedule-scoping limitation.
- **Date:** 2026-06-16.
- **Finding:** ROCm clang in this workspace accepts `-mwavefrontsize64` for
  wave64 and `-mno-wavefrontsize64` for wave32. It rejects
  `-mwavefrontsize32`.
- **Evidence:** The accepted HRX2 Q4_K narrow wave32 HIP bridge artifact
  `mul_mat_vec_q4_k_q8_1_wave32.hsaco` reports `.wavefront_size: 32` for
  `hrx2_mul_mat_vec_q4_k_q8_1_x4_vkm64x64_pack2_wg128_w32_u32`.
- **Schedule lesson:** The broad wave32 version of the Vulkan-medium Q4_K tile
  improved p64 focused rows but regressed most p512 rows and some model shapes.
  It became profitable only after route metadata narrowed it to proven cols64
  domains. Do not treat wave32 as a blanket fix; preserve wavefront size as a
  tunable schedule axis and promote only shape domains that win.

## Generic Loom Q5_K MMQ32x32 Is Not The Reference p64 Schedule

- **Status:** HRX2 route quality limitation, not a confirmed Loom compiler bug.
- **Date:** 2026-06-16.
- **Evidence:** `cache/hrx2/phase2a/q5-w32-vkm64x64-perf-20260616-142752/`.
- **Finding:** On the Phi p64 hot Q5 row (`k3072 rows5120 cols64`), the generic
  Loom `mul_mat_q5_k_q8_1_x4_mmq32x32...` route measured `310.158 us`, while a
  prior-led HIP bridge spelling the Vulkan/HRX1 packed-Q8_1-x4 wave32
  `BM64/BN64` schedule measured `194.608 us` and passed CPU-reference backend
  op testing.
- **Conclusion:** For Q5 cols64 prompt rows, future Loom work should not start
  from the generic `mmq32x32` route as though it were near-optimal. Use the
  accepted wave32 schedule as the reference dataflow and compare emitted
  compile reports/ISA if rewriting it in Loom.
