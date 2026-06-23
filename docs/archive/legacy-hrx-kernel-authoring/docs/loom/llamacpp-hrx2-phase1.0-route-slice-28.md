# HRX2 Phase 1.0 Route Slice 28

Date: 2026-06-12

This checkpoint adds the MoE top-k support routes needed after route slice 27:
small DESC `ARGSORT` over expert probabilities and the narrow MoE weight
`GET_ROWS` gather from top-k expert indices.

## Accepted Scope

Accepted route IDs:

```text
argsort_f32_i32_n128_r1_desc_wg128
argsort_f32_i32_n128_r16_desc_wg128
argsort_f32_i32_n128_r64_desc_wg128
get_rows_moe_weights_f32_ne128_k8_t1_wg256
get_rows_moe_weights_f32_ne128_k8_t16_wg256
get_rows_moe_weights_f32_ne128_k8_t64_wg256
```

The `ARGSORT` source is a target-neutral rank-count DESC implementation for
the traced MoE shape `ncols=128`. It is intended as phase-one coverage for the
small support op, not as a final general-purpose sort kernel. The intended
bitonic/LDS path is still blocked by a GPU fault under HRX2 raw dispatch.

The `GET_ROWS` source is deliberately narrow: it covers the MoE weights gather
from `src0=[1,128,tokens]` and top-k index view `idx=[8,tokens]` into
`dst=[1,8,tokens]`, preserving padded token strides as JIT config facts.
This is not the rejected generic `GET_ROWS` route.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-28-current/phase1_route_slice_28_get_rows_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-28-current
```

Result: 6/6 focused rows passed against ggml CPU reference. The HRX2 trace
selected all intended route IDs, JIT-compiled six providers, dispatched six
routes, and had no `provider_unavailable` events.

Compile-report guardrails:

| Route group | HSACO | Inst | Spills | Private | Local | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `argsort_f32_i32_n128_r{1,16,64}_desc_wg128` | 9200 | 52-59 | 0 | 0 | 0 | 8-9 |
| `get_rows_moe_weights_f32_ne128_k8_t{1,16,64}_wg256` | 9208 | 34-39 | 0 | 0 | 0 | 4-5 |

## Rejected Candidates

Two `ARGSORT` bitonic candidates were tried before accepting the rank-count
coverage route:

- a dynamic bitonic source using `index.div`/`index.shrui` loop control, which
  hit an AMDGPU address-width lowering diagnostic;
- a static/unrolled bitonic source with workgroup scratch, which compiled but
  GPU-faulted on dispatch.

Evidence directories:

```text
cache/hrx2/phase1_0/route-slice-28-current/test-focused-vector-scratch
cache/hrx2/phase1_0/route-slice-28-current/test-focused-static-argsort
```

The faulted bitonic/LDS route is recorded in
`docs/loom/loom-bugs-limitations.md`. Keep the rank-count route as the
phase-one coverage workaround until the LDS path is understood.

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-28-20260612-211843
```

Result: 33/33 prompt runs passed.

Scheduler reduction versus route slice 27:

| Metric | Slice 27 | Slice 28 | Delta |
| --- | ---: | ---: | ---: |
| CPU compute fallbacks | 238536 | 238536 | 0 |
| HRX20 compute nodes | 242748 | 242748 | 0 |
| Infrastructure blockers | 32112 | 32112 | 0 |

The lack of fallback-count movement is expected after inspecting scheduler
placement. The new `ARGSORT` and `GET_ROWS` routes are now reported as
`supported_by=HRX20,CPU`, but the MoE island still remains CPU-assigned because
`MUL_MAT_ID` gate/up/down paths are CPU-only:

```text
ARGSORT supported_by=HRX20,CPU assigned CPU
GET_ROWS supported_by=HRX20,CPU assigned CPU
SUM_ROWS/CLAMP/DIV supported_by=HRX20,CPU assigned CPU
MUL_MAT_ID supported_by=CPU assigned CPU
```

This is route prerequisite coverage, not a model-level fallback win. The next
MoE-impacting slice must offload `MUL_MAT_ID` or otherwise break the CPU graph
island.

## Checkpoint Decision

The route-admission slice is accepted as phase-one prerequisite coverage. All
new admitted routes have target-neutral Loom source, embedded bytecode,
catalog metadata, focused CPU-reference correctness, compile reports,
manifests, route traces, and full-basket pass evidence. The next fallback
slice should prioritize `MUL_MAT_ID`/quantized matmuls or attention
`ROPE`/`SOFT_MAX`/f16 matmuls, because those are the remaining model-level
placement blockers.
