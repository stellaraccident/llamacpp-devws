# HRX2 Phase 1.0 Route Slice 45

Date: 2026-06-13

## Scope

Slice 45 adds Mistral NORMAL-mode F32 ROPE coverage without an external
frequency tensor:

```text
rope_normal_f32_n128_d128_h8_t1_64_wg256
rope_normal_f32_n128_d128_h32_t1_64_wg256
```

The source is target-neutral Loom in:

```text
sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rope_normal_f32.loom
```

It uses the NORMAL adjacent-pair layout, binds `n_dims`, and keeps route
selection in catalog JSON. It does not add a source-level target attribute.

## Evidence

Mistral graph exports:

```text
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-p1-ops.txt
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-p16-ops.txt
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-p64-ops.txt
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-rope-focused-ops.txt
```

Focused CPU-reference validation:

```text
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/focused-20260613-025336
```

Result: 4/4 exact ROPE rows passed against ggml CPU reference. HRX2 selected
both new routes, and provider compilation succeeded.

Compile-report summary:

| Route | Reports | Artifact bytes | Code bytes | Inst | Spills | Private | Local | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rope_normal_f32_n128_d128_h8_t1_64_wg256` | 2 | 9192 | 284 | 59-60 | 0 | 0 | 0 | 10 |
| `rope_normal_f32_n128_d128_h32_t1_64_wg256` | 2 | 9192 | 284 | 59-60 | 0 | 0 | 0 | 10 |

Targeted Mistral smoke:

```text
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-smoke-20260613-025422
```

All three basket regimes passed. Fixed graph-node reduction:

| Class | Count |
| --- | ---: |
| accelerated | 28908 |
| infrastructure blocker | 2880 |
| compute fallback | 108 |

Remaining Mistral compute fallback is only GET_ROWS.

Full 11-model basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-45-20260613-025611
```

All 33 runs passed. Fixed graph-node reduction:

| Class | Count |
| --- | ---: |
| accelerated | 447792 |
| infrastructure blocker | 32112 |
| compute fallback | 1380 |

Compute backend counts:

| Backend | Count |
| --- | ---: |
| HRX20 | 479904 |
| CPU | 1380 |

Delta versus slice 44 after re-reducing slice 44 with the fixed graph-node
reducer:

| Metric | Delta |
| --- | ---: |
| HRX20 compute nodes | +2880 |
| CPU compute fallbacks | -2880 |

New route dispatch counts in the full basket:

| Count | Op | Route |
| ---: | --- | --- |
| 240 | `ROPE` | `rope_normal_f32_n128_d128_h32_t1_64_wg256` |
| 240 | `ROPE` | `rope_normal_f32_n128_d128_h8_t1_64_wg256` |

There were no `provider_unavailable` events.

## Remaining Work

The only remaining ROPE fallback in the full basket is the previously
documented Llama 3.1 NORMAL frequency-source h32/p64 case:

```text
ROPE f32 src f32,i32,f32 shape 128x32x64x1 count 192
```

It remains deliberately unclaimed because the h32/p64 route failed strict
ggml CPU-reference tolerance in route slice 43. The rest of the remaining
compute fallback inventory is GET_ROWS embedding lookups across F32 and
quantized embedding tables.
