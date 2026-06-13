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
- **Impact:** HRX2 cannot currently rely on the Loom f32->f16 unfused SET_ROWS
  provider, which is a scheduler prerequisite for model-level traces with KV
  updates.
- **Workaround:** HRX2 has a deliberately slow host-mediated SET_ROWS fallback
  so model-level evidence collection can continue. The fallback is used by
  default; set `GGML_HRX2_ENABLE_SET_ROWS_LOOM=1` to exercise the current Loom
  SET_ROWS providers and reproduce the lowering failure. This is not an
  optimized-kernel substitute and must not be counted as done-done kernel
  coverage.
- **Owner:** Loom lowering investigation, with HRX2 fallback owned in
  llama.cpp until fixed.

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

## 2026-06-12: GET_ROWS f32 high-level Loom candidate fails target lowering and correctness

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
- **Impact:** HRX2 must not advertise GET_ROWS f32 support yet. Leaving a route
  selectable causes the scheduler to assign GET_ROWS to HRX2 and either fail
  JIT compilation or fail CPU-reference validation.
- **Workaround:** The GET_ROWS candidate was removed from the production
  catalog and runtime route discovery. Keep GET_ROWS on CPU until the address
  proof and indexing semantics are corrected, preferably with a smaller
  standalone Loom reproduction before re-adding route metadata.
- **Owner:** Loom lowering investigation plus HRX2 route admission.
