# HRX2 Phase 1.0 Route Slice 40: Q5/Q6 Loader Domains And Indexed Matmul

Date: 2026-06-13

## Scope

This checkpoint closes the remaining quantized expert-matmul gap exposed by
Qwen3 after slice 38:

- widen direct Q4_K/Q5_K/Q6_K `MUL_MAT` source and route column domains to
  include llama.cpp's 512-token load-time weight probe;
- add target-neutral direct `MUL_MAT_ID` baselines for Q5_K and Q6_K expert
  weights;
- share HRX2 C++ `MUL_MAT_ID` shape/support/dispatch logic across Q4_K,
  Q5_K, and Q6_K while keeping catalog families and route vectors separate.

The indexed Q5/Q6 routes use the same baseline topology as Q4:
one workgroup per `(row, selected_expert, token)`, JIT-specialized expert
planes, RHS selected/token strides, and destination token stride.

## Evidence

Focused CPU-reference validation:

- Q5 real Qwen3 rows:
  `cache/hrx2/phase1_0/route-slice-40-mul-mat-id-q5-q6-focused-current/q5-test`
- Q6 synthetic expert-down rows with Q6_K block strides:
  `cache/hrx2/phase1_0/route-slice-40-mul-mat-id-q5-q6-focused-current/q6-synthetic-test`

Qwen3 UD-Q4 placement smoke:

- `cache/hrx2/phase1_0/route-slice-40-mul-mat-id-q5-q6-focused-current/qwen3-p1`
- HRX20 model buffer increased to `16650.36 MiB`.
- q4 `MUL_MAT_ID`: `1560` HRX20 scheduler placements.
- q5 `MUL_MAT_ID`: `168` HRX20 scheduler placements.
- q5 indexed dispatches: `28`.

Full 11-model basket smoke:

- `cache/hrx2/phase1_0/basket-smoke-route-slice-40-20260613-010859`
- Result: `33/33` passed across decode `p=1`, narrow `p=16`, and prefill64
  `p=64`.
- Aggregate:
  - nodes: `516270`
  - accelerated: `426182`
  - host orchestration: `34986`
  - infrastructure blocker: `32112`
  - compute fallback: `22990`
  - HRX20 compute: `458294`
  - CPU compute: `22990`

## Remaining Backlog

The top compute fallback families after this slice are no longer quantized
expert matmul. Next high-value phase 1 targets:

- F32/F32 attention matmul: `MUL_MAT f32,f32` shapes
  `128x{1,16,64}x1x1`.
- Q8_0 direct matmul wider FFN/output rows:
  `14336x{1,16,64}x1x1` and `4096x{1,16,64}x1x1`.
- ROPE with frequency source operand:
  `ROPE f32,i32,f32` and remaining no-frequency normal/NeoX variants for
  `128x32x{1,16,64}x1`.
- GLU/SWIGLU width coverage:
  `13824`, `14336`, `18944`, `21504`, `32768` across decode/narrow/prefill64.
- GET_ROWS coverage for F32 and Q4_K embeddings/state rows.

## Process Notes

- Load-placement validation is mandatory. The direct Q5/Q6 `MUL_MAT` kernels
  were already source-valid, but the model loader would not place weights in
  HRX2 buffers until the route and source domains accepted 512 columns.
- Q6 indexed validation used synthetic op rows because no cached basket export
  contained a real Q6_K `MUL_MAT_ID` row. The rows preserve expert/down
  topology and use correct Q6_K block strides, so this is a route correctness
  test, not basket frequency evidence.
- Future agents should keep q4/q5/q6 indexed catalog families separate even
  though C++ dispatch is shared; tuning and performance refutation will differ
  by quant format.

