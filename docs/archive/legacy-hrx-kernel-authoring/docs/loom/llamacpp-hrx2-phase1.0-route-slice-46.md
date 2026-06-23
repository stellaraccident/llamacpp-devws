# HRX2 Phase 1.0 Route Slice 46

Date: 2026-06-13

This slice adds compact dense F32 `GET_ROWS` coverage for the remaining
non-quantized row-gather fallbacks in the coverage basket.

## Accepted Routes

```text
get_rows_f32_n2048_r1_64_wg256
get_rows_f32_n3072_r1_64_wg256
get_rows_f32_n3584_r1_64_wg256
get_rows_f32_n4096_r1_64_wg256
get_rows_f32_n5120_r1_64_wg256
get_rows_f32_n5376_r1_64_wg256
```

## Implementation Notes

- Added target-neutral `kernels/get_rows_f32.loom`.
- The accepted source uses a 2D dense view spelling:
  `src0[row_index, col] -> dst[row, col]`.
- Runtime support is deliberately narrow:
  F32 source, I32 indices, F32 destination, 2D compact dense rows, contiguous
  output, and `src0.ne[2..3] == 1`.
- Quantized embedding `GET_ROWS` rows remain unsupported and are the next
  row-gather family.
- The first flat 1D source attempted for this slice reproduced the earlier
  AMDGPU address-width target-low rejection. The 2D dense view spelling avoided
  the issue and matched the intended schedule more directly.

## Focused Validation

Focused graph-op rows:

```text
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/get_rows_all_existing_exports_ops.txt
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/get_rows_phi4_3072_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/focused-existing-exports-20260613-032516
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/focused-phi4-3072-20260613-032546
```

The combined focused replays passed. Unsupported quantized embedding rows in
the same files stayed unsupported. HRX2 dispatched all six dense F32 route
buckets with no `provider_unavailable` and no `dispatch_failed` events.

## Compile Reports

| Route | HSACO | Inst | Global mem | Spills | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: |
| `get_rows_f32_n2048_r1_64_wg256` | 9184 | 23 | 3 | 0 | 8 |
| `get_rows_f32_n3072_r1_64_wg256` | 9184 | 19 | 2 | 0 | 10 |
| `get_rows_f32_n3584_r1_64_wg256` | 9184 | 30 | 3 | 0 | 8 |
| `get_rows_f32_n4096_r1_64_wg256` | 9184 | 23 | 3 | 0 | 8 |
| `get_rows_f32_n5120_r1_64_wg256` | 9184 | 27 | 3 | 0 | 8 |
| `get_rows_f32_n5376_r1_64_wg256` | 9184 | 30 | 3 | 0 | 8 |

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-46-20260613-032627
```

Result: 33/33 prompt runs passed.

Aggregate after slice 46:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| HRX20 compute nodes | 480696 |
| CPU compute fallbacks | 588 |
| infrastructure blockers | 32112 |

Delta versus slice 45: `compute_fallback -792`, `HRX20 compute +792`.

New full-basket route dispatches:

| Count | Route | Shape facts |
| ---: | --- | --- |
| 36 | `get_rows_f32_n2048_r1_64_wg256` | `ncols=2048`, `nrows={1,16,64}` |
| 24 | `get_rows_f32_n4096_r1_64_wg256` | `ncols=4096`, `nrows={1,16,64}` |
| 24 | `get_rows_f32_n3072_r1_64_wg256` | `ncols=3072`, `nrows={1,16,64}` |
| 24 | `get_rows_f32_n5120_r1_64_wg256` | `ncols=5120`, `nrows={1,16,64}` |
| 12 | `get_rows_f32_n3584_r1_64_wg256` | `ncols=3584`, `nrows={1,16,64}` |
| 12 | `get_rows_f32_n5376_r1_64_wg256` | `ncols=5376`, `nrows={1,16,64}` |

Remaining compute fallbacks:

- Known unclaimed Llama 3.1 normal-frequency ROPE h32/p64 row:
  `ROPE f32 <- f32,i32,f32`, shape `128x32x64x1`, count `192`.
- Quantized embedding `GET_ROWS`:
  `q4_K`, `q5_K`, `q6_K`, and `q8_0` sources for hidden widths
  `2048`, `3072`, `3584`, `4096`, `5120`, and `5376`, with row buckets
  `1`, `16`, and `64`.

## Decision

Slice 46 is accepted as compact dense F32 `GET_ROWS` Phase 1 coverage. The next
row-gather slice should implement quantized `GET_ROWS` dequantizing to F32,
using the Vulkan/Metal quant gather priors rather than widening this dense F32
route.
