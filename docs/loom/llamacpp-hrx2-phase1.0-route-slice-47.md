# HRX2 Phase 1.0 Route Slice 47

Date: 2026-06-13

This slice adds quantized embedding `GET_ROWS` coverage for the coverage-basket
fallbacks left after compact dense F32 `GET_ROWS`.

## Accepted Routes

```text
get_rows_q4_k_f32_n2048_r1_512_wg256
get_rows_q4_k_f32_n4096_r1_512_wg256
get_rows_q4_k_f32_n5120_r1_512_wg256
get_rows_q5_k_f32_n3584_r1_512_wg256
get_rows_q6_k_f32_n2048_r1_512_wg256
get_rows_q6_k_f32_n3072_r1_512_wg256
get_rows_q6_k_f32_n5376_r1_512_wg256
get_rows_q8_0_f32_n4096_r1_512_wg256
```

## Implementation Notes

- Added target-neutral Loom sources for `q4_K`, `q5_K`, `q6_K`, and `q8_0`
  embedding row gather into F32.
- The kernels map one output element per lane, load the selected row index,
  decode the source quant block element, and store the dequantized F32 output.
- Route metadata remains target-specific through `target_key`, but the Loom
  sources do not use source-level target attributes.
- Runtime support is deliberately narrow: source type must match the route
  family, source indices are I32, destination is F32, source and destination
  are compact embedding-row layouts, and the hidden width must match an
  accepted route bucket.
- Added a conservative HRX2 `offload_op` hook for `GET_ROWS`. Without it,
  quantized embedding gathers could be `supported_by=[HRX20,CPU]` but still
  CPU-assigned because token-embedding weights enter the graph from CPU/host
  placement.

## Focused Validation

Focused graph-op rows:

```text
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/get_rows_all_existing_exports_ops.txt
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/get_rows_phi4_3072_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-47-get-rows-quant/focused-final-20260613-042127
```

Result: 41/41 exact graph-op rows passed against ggml CPU reference.

Dispatch summary:

| Count | Route |
| ---: | --- |
| 5 | `get_rows_q4_k_f32_n5120_r1_512_wg256` |
| 3 | `get_rows_q4_k_f32_n2048_r1_512_wg256` |
| 3 | `get_rows_q5_k_f32_n3584_r1_512_wg256` |
| 3 | `get_rows_q8_0_f32_n4096_r1_512_wg256` |
| 3 | `get_rows_q6_k_f32_n5376_r1_512_wg256` |
| 2 | `get_rows_q6_k_f32_n3072_r1_512_wg256` |

No `provider_unavailable`, `dispatch_failed`, or unsupported focused rows were
observed.

## Scheduler Placement Fix

Before this slice, a full basket smoke with the quantized routes present still
left 396 quantized embedding `GET_ROWS` graph nodes on CPU:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-47-loader-20260613-040651
```

The scheduler trace showed those nodes as HRX2-supported but CPU-assigned. A
single-model smoke after adding `offload_op` proved placement changed for the
Qwen3 UD Q4 token embedding gather:

```text
cache/hrx2/phase1_0/route-slice-47-get-rows-quant/offload-hook-qwen-q4-p1-20260613-042034
```

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-47-offload-hook-20260613-042340
```

Result: 33/33 prompt runs passed.

Aggregate after slice 47:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| HRX20 compute nodes | 481092 |
| CPU compute fallbacks | 192 |
| infrastructure blockers | 32112 |

Delta versus slice 46: `compute_fallback -396`, `HRX20 compute +396`.

Full-basket quantized `GET_ROWS` dispatches:

| Count | Route |
| ---: | --- |
| 12 | `get_rows_q4_k_f32_n2048_r1_512_wg256` |
| 12 | `get_rows_q6_k_f32_n3072_r1_512_wg256` |
| 12 | `get_rows_q4_k_f32_n5120_r1_512_wg256` |
| 6 | `get_rows_q6_k_f32_n2048_r1_512_wg256` |
| 6 | `get_rows_q4_k_f32_n4096_r1_512_wg256` |
| 6 | `get_rows_q8_0_f32_n4096_r1_512_wg256` |
| 6 | `get_rows_q5_k_f32_n3584_r1_512_wg256` |
| 6 | `get_rows_q6_k_f32_n5376_r1_512_wg256` |

Remaining compute fallback:

```text
ROPE f32 <- f32,i32,f32, shape 128x32x64x1, count 192
```

`cpu_assigned_but_hrx_supported` is empty in the reduced basket summary.

## Decision

Slice 47 is accepted as quantized embedding `GET_ROWS` Phase 1 coverage. The
next Phase 1 coverage item is the remaining h32/p64 normal-frequency ROPE row,
which is already known to need numeric-parity work before the route can be
admitted.
