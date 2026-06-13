# HRX2 Phase 1.0 Route Slice 27

Date: 2026-06-12

This checkpoint adds the next phase-one route slice after
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-26.md`.

## Accepted Scope

Accepted route IDs:

```text
div_f32_n8_r1_rhscolbroadcast_wg256
div_f32_n8_r16_rhscolbroadcast_wg256
div_f32_n8_r64_rhscolbroadcast_wg256
clamp_f32_n1_r1_contiguous_wg256
clamp_f32_n1_r16_contiguous_wg256
clamp_f32_n1_r64_contiguous_wg256
sum_rows_f32_n8_r1_wg32
sum_rows_f32_n8_r16_wg32
sum_rows_f32_n8_r64_wg32
rms_norm_f32_n3072_r64_vector_vw4_wg512
rms_norm_f32_n3584_r1_vector_vw4_wg512
rms_norm_f32_n3584_r16_vector_vw4_wg512
rms_norm_f32_n3584_r64_vector_vw4_wg512
rms_norm_f32_n4096_r16_vector_vw4_wg512
rms_norm_f32_n4096_r64_vector_vw4_wg512
```

The new pointwise coverage reuses the target-neutral `pointwise_f32.loom`
source. `DIV` covers the traced MoE RHS column-broadcast form:
`src0=[8,tokens]`, `src1=[1,tokens]`, `dst=[8,tokens]`. `CLAMP` covers the
traced contiguous one-column scalar rows. `SUM_ROWS` uses a new target-neutral
`sum_rows_f32.loom` source with one workgroup per row and WG32 reduction over
the eight MoE weights.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-27-20260612-current/phase1_route_slice_27_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-27-20260612-current/test-focused
```

Result: 15/15 focused rows passed against ggml CPU reference. The HRX2 trace
selected all intended route IDs and had no `provider_unavailable` events.

Compile-report guardrails:

| Route group | HSACO | Inst | Spills | Private | Local | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `div_f32_n8_r{1,16,64}_rhscolbroadcast_wg256` | 9176 | 22-24 | 0 | 0 | 0 | 8-10 |
| `clamp_f32_n1_r{1,16,64}_contiguous_wg256` | 9176 | 23 | 0 | 0 | 0 | 10 |
| `sum_rows_f32_n8_r{1,16,64}_wg32` | 9184 | 46-56 | 0 | 0 | 0 | 8-15 |
| `rms_norm_f32_n{3072,3584,4096}_r*_vector_vw4_wg512` | 9224 | 131-134 | 0 | 0 | 64 | 13 |

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-27-20260612-204250
```

Result: 33/33 prompt runs passed.

Scheduler reduction versus route slice 26:

| Metric | Slice 26 | Slice 27 | Delta |
| --- | ---: | ---: | ---: |
| CPU compute fallbacks | 242502 | 238536 | -3966 |
| HRX20 compute nodes | 238782 | 242748 | +3966 |
| Infrastructure blockers | 32112 | 32112 | 0 |

New residual RMS routes dispatched in the full basket, including:

```text
rms_norm_f32_n4096_r16_vector_vw4_wg512
rms_norm_f32_n4096_r64_vector_vw4_wg512
rms_norm_f32_n3584_r1_vector_vw4_wg512
rms_norm_f32_n3584_r16_vector_vw4_wg512
rms_norm_f32_n3584_r64_vector_vw4_wg512
rms_norm_f32_n3072_r64_vector_vw4_wg512
```

The MoE support routes (`SUM_ROWS`, `CLAMP`, `DIV`) are accepted as standalone
coverage but did not dispatch in the full basket. Scheduler traces show they
remain in a CPU island because upstream `ARGSORT` and `GET_ROWS` nodes are
CPU-only:

```text
ARGSORT supported_by=CPU
GET_ROWS supported_by=CPU
SUM_ROWS/CLAMP/DIV supported_by=HRX20,CPU but assigned CPU
```

This is a placement dependency, not a kernel validation failure. The next MoE
coverage slice must solve `ARGSORT` and `GET_ROWS` before expecting the newly
accepted `SUM_ROWS`/`CLAMP`/`DIV` routes or existing GLU routes to reduce
model-level fallback counts.

## Checkpoint Decision

The route-admission slice is accepted as phase-one coverage. All new admitted
routes have focused CPU-reference correctness, compile reports, manifests,
route traces, and full-basket pass evidence. The basket-effecting win in this
slice is residual RMS_NORM coverage; MoE support route impact is blocked by
the still-CPU top-k/gather island.
