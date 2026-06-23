# HRX2 Phase 1.0 Route Slice 30

Date: 2026-06-12

This checkpoint adds F32 `SOFT_MAX` coverage for the Phase 1 attention
softmax rows and the narrow MoE probability rows observed in the 11-model
basket. It deliberately excludes sinks, `max_bias > 0`, F16 masks, non-F32
destinations, and untraced column counts.

## Accepted Scope

Accepted family:

```text
soft_max_f32
```

Accepted route shape grid:

```text
masked attention:
  ncols = 256
  nrows = 24, 28, 32, 40, 384, 448, 512, 640, 1536, 1792, 2048, 2560
  workgroup_size = 256

unmasked MoE probabilities:
  ncols = 128
  nrows = 1, 16, 64
  workgroup_size = 128
```

The Loom source is target-neutral. The route rows use `target_key=gfx1100`
only as measured catalog metadata.

ABI:

```text
launch params:
  scale: f32
bindings:
  unmasked:
    0: src0 f32
    1: dst f32
  masked:
    0: src0 f32
    1: mask f32
    2: dst f32
constants:
  scale
```

Algorithm:

- one workgroup owns one row;
- one lane owns one column, so accepted routes require `workgroup_size == ncols`;
- reduce row max, exponentiate `src0 * scale + optional mask`, reduce row sum,
  and normalize;
- mask indexing is specialized from element strides and broadcast dimensions.

## Focused Validation

Focused test files:

```text
cache/hrx2/phase1_0/route-slice-30-softmax-focused-20260612-current/soft_max_f32_ops.txt
cache/hrx2/phase1_0/route-slice-30-softmax-focused-20260612-current/soft_max_f32_mask_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-30-softmax-focused-20260612-current/test-focused-unmasked-param-fixed
cache/hrx2/phase1_0/route-slice-30-softmax-focused-20260612-current/test-focused-masked-manual
```

Result: 6/6 focused rows passed against ggml CPU reference. The unmasked rows
covered `128x{1,16,64}`. The masked rows covered attention-style broadcast
masks for `256x1x{24,32,40}x1`. HRX2 selected the intended route IDs,
JIT-compiled all six providers, dispatched all six routes, and had no provider
failures.

## Compile Report Guardrails

| Route | HSACO | Schedule nodes | Spills | Private | Local | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soft_max_f32_n128_r1_wg128` | 9184 | 104 | 0 | 0 | 32 | 7 |
| `soft_max_f32_n128_r16_wg128` | 9184 | 117 | 0 | 0 | 32 | 9 |
| `soft_max_f32_n128_r64_wg128` | 9184 | 117 | 0 | 0 | 32 | 9 |
| `soft_max_f32_mask_n256_r24_wg256` | 9192 | 129 | 0 | 0 | 64 | 11 |
| `soft_max_f32_mask_n256_r32_wg256` | 9192 | 129 | 0 | 0 | 64 | 11 |
| `soft_max_f32_mask_n256_r40_wg256` | 9192 | 129 | 0 | 0 | 64 | 11 |

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-30-softmax-20260612-221001-fixed
```

Result: 33/33 prompt runs passed.

Scheduler reduction versus route slice 29:

| Metric | Slice 29 | Slice 30 | Delta |
| --- | ---: | ---: | ---: |
| CPU compute fallbacks | 218232 | 202176 | -16056 |
| HRX20 compute nodes | 263052 | 279108 | +16056 |
| Infrastructure blockers | 32112 | 32112 | 0 |

Route dispatch counts in the full basket:

| Count | Route |
| ---: | --- |
| 1240 | `soft_max_f32_mask_n256_r32_wg256` |
| 310 | `soft_max_f32_mask_n256_r512_wg256` |
| 310 | `soft_max_f32_mask_n256_r2048_wg256` |
| 240 | `soft_max_f32_mask_n256_r24_wg256` |
| 192 | `soft_max_f32_mask_n256_r40_wg256` |
| 112 | `soft_max_f32_mask_n256_r28_wg256` |
| 60 | `soft_max_f32_mask_n256_r384_wg256` |
| 60 | `soft_max_f32_mask_n256_r1536_wg256` |
| 48 | `soft_max_f32_mask_n256_r640_wg256` |
| 48 | `soft_max_f32_mask_n256_r2560_wg256` |
| 28 | `soft_max_f32_mask_n256_r448_wg256` |
| 28 | `soft_max_f32_mask_n256_r1792_wg256` |

There were 504 provider compiles, 41,166 provider cache hits, 46,518 HRX2
dispatches, and no provider failures in the full basket run.

## Placement Finding

The unmasked MoE `SOFT_MAX` routes pass focused CPU-reference validation and
report `supported_by=HRX20,CPU` in the full basket, but they remain CPU-assigned
because their logits are produced by CPU-only MoE matmul paths:

```text
ffn_moe_logits -> SOFT_MAX -> ARGSORT -> GET_ROWS -> SUM_ROWS/CLAMP/DIV
```

This is the same island-placement issue recorded for route slices 27 and 28.
It is not a `SOFT_MAX` provider failure. The next MoE-impacting route work
must address `MUL_MAT_ID` or split the island.

## Checkpoint Decision

This route slice is accepted as Phase 1 unfused attention softmax coverage. It
has target-neutral Loom source, embedded bytecode, catalog metadata, focused
CPU-reference correctness for masked and unmasked roots, compile reports,
manifests, selected-route traces, and full-basket pass evidence.
