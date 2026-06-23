# HRX2 Phase 1.0 Route Slice 31

Date: 2026-06-12

This checkpoint adds target-neutral F16/F32 batched attention `MUL_MAT`
coverage for the 11-model Phase 1 basket. It covers the observed KQ and KQV
matvec layouts where `src0` is F16 p021-style attention memory, `src1` is F32,
and the destination is F32.

## Accepted Scope

Accepted family:

```text
mul_mat_f16_f32_batched
```

Accepted route shape domain:

```text
k    = 128, 256
rows = 128, 256
cols = 1, 16, 64
heads = 24, 28, 32, 40
src0 grouped heads = 4, 8, 16
workgroup_size = 256
```

The Loom source is target-neutral. The accepted route has no target-specific
`target_key`; it specializes all dimensions and byte strides through
`jit_config` bindings.

ABI:

```text
bindings:
  0: src0 f16
  1: src1 f32
  2: dst f32
constants:
  none
```

Algorithm:

- one workgroup owns one output element `(row, col, head, batch)`;
- each lane accumulates a strided dot-product segment over `k`;
- the workgroup reduces the F32 partials and lane 0 writes the result;
- `src0` grouped-head broadcast is specialized through `src0_ne2/src0_ne3`
  and byte strides, matching the old HRX batched F16 attention matvec layout.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-31-f16-attn-focused-current/mul_mat_f16_f32_attention_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-31-f16-attn-focused-current
```

Result: 12/12 focused rows passed against ggml CPU reference. The rows cover
both KQ and KQV shapes for `cols=1` and `cols=16`, plus head counts
`24,28,32,40`. HRX2 selected `mul_mat_f16_f32_batched_attention_wg256`, JIT
compiled 12 specializations, dispatched all rows, and had no provider
failures.

## Compile Report Guardrails

Across the 12 focused specializations:

| Metric | Range |
| --- | ---: |
| HSACO bytes | 9208 |
| Schedule nodes | 94-121 |
| Code bytes | 472-612 |
| Spills | 0 |
| Private bytes | 0 |
| Peak live units | 11-20 |

The largest register-pressure case was `k=128, rows=256, cols=16`,
`dst_ne2=24/32`, with peak live units at 20. No accepted specialization had
spills or private memory.

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-31-f16-attn-current
```

Result: 33/33 prompt runs passed.

Route dispatch counts in the full basket:

| Count | Route |
| ---: | --- |
| 2676 | `mul_mat_f16_f32_batched_attention_wg256` |

There were no provider failures in the full basket run. The scheduler
reduction shows no remaining F16/F32 `MUL_MAT` compute fallbacks:

| Metric | Slice 31 |
| --- | ---: |
| CPU compute fallbacks | 85032 |
| HRX20 compute nodes | 155610 |
| F16/F32 `MUL_MAT` CPU fallbacks | 0 |
| F16/F32 `MUL_MAT` HRX20 nodes | 16056 |

The total scheduler node count changed substantially versus slice 30 because
moving the attention matmuls to HRX2 changes graph partitioning and removes a
large amount of CPU island/copy structure. For this slice, the reliable direct
acceptance signals are: no residual F16/F32 `MUL_MAT` fallbacks, 2676 route
dispatches, no provider failures, and 33/33 model-regime passes.

Top remaining compute fallback families after this slice are K-quant
`MUL_MAT`, `MUL_MAT_ID` MoE paths, F32/F32 MoE logits, and `ROPE` variants with
`src2` frequency factors.

## Prior Art

The accepted source follows the old HRX batched F16 attention matvec family:

```text
sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_f16_batched.hip.cpp
```

The old HRX file contains scalar cols=1, rows2/cols1 vectorized, and
cols4/8/16 variants. Slice 31 intentionally accepts the scalar one-output
baseline first to close Phase 1 coverage, while preserving the family boundary
needed to tune rows-per-workgroup and cols-per-workgroup variants later.

## Checkpoint Decision

This route slice is accepted as Phase 1 unfused F16/F32 attention matmul
coverage. It has target-neutral Loom source, embedded bytecode, catalog
metadata, focused CPU-reference correctness, compile reports, manifests,
selected-route traces, and full-basket pass evidence.
