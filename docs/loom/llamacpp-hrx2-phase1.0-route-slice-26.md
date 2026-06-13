# HRX2 Phase 1.0 Route Slice 26

Date: 2026-06-12

This checkpoint adds the next phase-one coverage slice after
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-15.md`.

## Accepted Scope

Accepted route IDs:

```text
rms_norm_f32_n128_r4_vector_vw4_wg64
rms_norm_f32_n128_r16_vector_vw4_wg64
rms_norm_f32_n128_r32_vector_vw4_wg64
rms_norm_f32_n128_r64_vector_vw4_wg64
rms_norm_f32_n128_r256_vector_vw4_wg64
rms_norm_f32_n128_r512_vector_vw4_wg64
rms_norm_f32_n128_r1024_vector_vw4_wg64
rms_norm_f32_n128_r2048_vector_vw4_wg64
mul_f32_n2048_r8_rhscolbroadcast_wg256
mul_f32_n2048_r128_rhscolbroadcast_wg256
add_f32_n2048_r16_lhsrhsrowstride_wg256
```

The pointwise source was widened from contiguous-linear inputs to a
layout-explicit row kernel. The JIT config now includes:

```text
@hrx2.shape.pointwise.src0_row_stride
@hrx2.shape.pointwise.src1_row_stride
@hrx2.shape.pointwise.src1_ncols
```

This admits the traced MoE shapes without claiming arbitrary ggml broadcasting:

- `MUL` with contiguous source and RHS column broadcast, e.g.
  `src0=[2048,8,tokens]`, `src1=[1,8,tokens]`;
- `ADD` with row-strided F32 sources and contiguous destination, e.g.
  `src nb1=65536` for `ne0=2048`.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-26-20260612-201843/phase1_route_slice_26_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-26-20260612-201843/test-focused-rerun
```

Result: 12/12 focused rows passed against ggml CPU reference. The HRX2 trace
selected all intended route IDs and had no `provider_unavailable` events.

Compile-report guardrails:

| Route | HSACO | Inst | Spills | Private | Local | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `add_f32_n2048_r16_lhsrhsrowstride_wg256` | 9176 | 25 | 0 | 0 | 0 | 3 |
| `add_f32_n2048_r64_rhsrowstride_wg256` | 9176 | 25 | 0 | 0 | 0 | 3 |
| `mul_f32_n2048_r8_rhscolbroadcast_wg256` | 9176 | 20 | 0 | 0 | 0 | 3 |
| `mul_f32_n2048_r128_rhscolbroadcast_wg256` | 9176 | 20 | 0 | 0 | 0 | 3 |
| `rms_norm_f32_n128_r4_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r16_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r32_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r64_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r256_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r512_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r1024_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |
| `rms_norm_f32_n128_r2048_vector_vw4_wg64` | 9224 | 117 | 0 | 0 | 8 | 8 |

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-26-20260612-202007
```

Result: 33/33 prompt runs passed.

Scheduler reduction versus route slice 15:

| Metric | Slice 15 | Slice 26 | Delta |
| --- | ---: | ---: | ---: |
| CPU compute fallbacks | 264240 | 242502 | -21738 |
| HRX20 compute nodes | 217044 | 238782 | +21738 |
| Infrastructure blockers | 32112 | 32112 | 0 |

New route dispatch counts in the full basket:

| Count | Op | Route |
| ---: | --- | --- |
| 1269 | `ADD` | `add_f32_n2048_r16_lhsrhsrowstride_wg256` |
| 824 | `RMS_NORM` | `rms_norm_f32_n128_r32_vector_vw4_wg64` |
| 582 | `MUL` | `mul_f32_n2048_r8_rhscolbroadcast_wg256` |
| 576 | `RMS_NORM` | `rms_norm_f32_n128_r4_vector_vw4_wg64` |
| 248 | `RMS_NORM` | `rms_norm_f32_n128_r16_vector_vw4_wg64` |
| 206 | `RMS_NORM` | `rms_norm_f32_n128_r256_vector_vw4_wg64` |
| 206 | `RMS_NORM` | `rms_norm_f32_n128_r512_vector_vw4_wg64` |
| 206 | `RMS_NORM` | `rms_norm_f32_n128_r2048_vector_vw4_wg64` |
| 144 | `RMS_NORM` | `rms_norm_f32_n128_r64_vector_vw4_wg64` |
| 141 | `MUL` | `mul_f32_n2048_r128_rhscolbroadcast_wg256` |
| 62 | `RMS_NORM` | `rms_norm_f32_n128_r1024_vector_vw4_wg64` |

The top-100 fallback list no longer contains `ncols=128` RMS_NORM or the
traced MoE pointwise ADD/MUL layouts. Remaining top fallbacks are still
dominated by attention `MUL_MAT`/`SOFT_MAX`, quantized matmuls, `ROPE`,
`MUL_MAT_ID`, and MoE support ops (`ARGSORT`, `GET_ROWS`, `SUM_ROWS`,
`CLAMP`, `DIV`).

## Checkpoint Decision

The route-admission slice is accepted as phase-one coverage. It is not a
done-done performance pass for RMS_NORM or pointwise ops, but all new admitted
routes have focused CPU-reference correctness, compile reports, manifests,
route traces, and full-basket evidence.
