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
