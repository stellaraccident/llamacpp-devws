# HRX2 Phase 1.0 Route Slice 15

Date: 2026-06-12

llama.cpp checkpoint:

```text
20db711cc hrx2: add phase one route slice
```

This checkpoint adds 15 unfused HRX2 routes selected from the coverage-basket
fallback audit. The scope is intentionally breadth-first: RMS_NORM hidden-size
coverage, row-strided ADD, and split-input SWIGLU. It is not a done-done
performance pass.

## Accepted Routes

| Family | Routes | Why |
| --- | ---: | --- |
| `rms_norm_f32` | 9 | Covers dominant dense hidden sizes `2048`, `5120`, and `5376` across decode, narrow, and prefill64 rows. |
| `add_f32` | 2 | Covers Qwen MoE residual/add rows where RHS has padded row stride. |
| `swiglu_f32` | 4 | Covers split gate/up SWIGLU rows from Qwen MoE and Llama 3.1 Q8 decode. |

Route ids:

```text
rms_norm_f32_n2048_r1_vector_vw4_wg512
rms_norm_f32_n2048_r16_vector_vw4_wg512
rms_norm_f32_n2048_r64_vector_vw4_wg512
rms_norm_f32_n5120_r1_vector_vw4_wg512
rms_norm_f32_n5120_r16_vector_vw4_wg512
rms_norm_f32_n5120_r64_vector_vw4_wg512
rms_norm_f32_n5376_r1_vector_vw4_wg512
rms_norm_f32_n5376_r16_vector_vw4_wg512
rms_norm_f32_n5376_r64_vector_vw4_wg512
add_f32_n2048_r16_rhsrowstride_wg256
add_f32_n2048_r64_rhsrowstride_wg256
swiglu_f32_split_n768_r8_wg256
swiglu_f32_split_n768_r128_wg256
swiglu_f32_split_n768_r512_wg256
swiglu_f32_split_n14336_r1_wg256
```

## Implementation Notes

- Added a target-neutral `hrx2_swiglu_f32_split` Loom root with three bindings:
  `src0`, `src1`, and `dst`.
- Extended generic SWIGLU support to admit either packed single-source SWIGLU
  or split-source SWIGLU; route selection rejects ABI mismatches by binding
  count.
- Extended pointwise ADD/MUL admission to allow same-shape F32 RHS tensors with
  padded row stride for 2D rows, using the existing
  `@hrx2.shape.pointwise.src1_row_stride` config.
- Added exact catalog rows for this checkpoint. The Loom sources remain
  target-neutral; route rows are selected from JSON metadata.

## Focused Validation

Exact graph-op rows were exported under:

```text
cache/hrx2/phase1_0/route-slice-export-20260612-194632
```

Focused CPU-reference test:

```bash
LD_LIBRARY_PATH="$PWD/build/llama-hrx2/bin:$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}" \
  GGML_HRX2_TRACE_JSONL="$OUT/hrx2.jsonl" \
  GGML_HRX2_EVIDENCE_DIR="$OUT/evidence" \
  GGML_HRX2_DUMP_COMPILE_REPORT=1 \
  GGML_HRX2_DUMP_MANIFEST=1 \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase1_0/route-slice-export-20260612-194632/phase1_route_slice_15_ops.txt \
  --output csv
```

Evidence:

```text
cache/hrx2/phase1_0/route-slice-export-20260612-194632/test-phase1-route-slice-15
```

Result: 15 focused rows passed against ggml CPU reference, 15 unique HRX2 route
ids were dispatched, and there were no `provider_unavailable` events.

Focused perf evidence:

```text
cache/hrx2/phase1_0/route-slice-export-20260612-194632/perf-console-phase1-route-slice-15
```

These timings are harness/launch dominated and should not be used as final
performance refutation evidence. They are sufficient phase-1 route evidence.

## Compile Report Summary

| Route | HSACO | Code bytes | Inst | Spills | Private | Local | Peak live | Global mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `add_f32_n2048_r16_rhsrowstride_wg256` | 9176 | 108 | 24 | 0 | 0 | 0 | 8 | 3 |
| `add_f32_n2048_r64_rhsrowstride_wg256` | 9176 | 108 | 24 | 0 | 0 | 0 | 8 | 3 |
| `rms_norm_f32_n2048_r16_vector_vw4_wg512` | 9224 | 532 | 129 | 0 | 0 | 64 | 14 | 3 |
| `rms_norm_f32_n2048_r1_vector_vw4_wg512` | 9224 | 488 | 109 | 0 | 0 | 64 | 9 | 3 |
| `rms_norm_f32_n2048_r64_vector_vw4_wg512` | 9224 | 532 | 129 | 0 | 0 | 64 | 14 | 3 |
| `rms_norm_f32_n5120_r16_vector_vw4_wg512` | 9224 | 632 | 134 | 0 | 0 | 64 | 13 | 3 |
| `rms_norm_f32_n5120_r1_vector_vw4_wg512` | 9224 | 608 | 131 | 0 | 0 | 64 | 13 | 3 |
| `rms_norm_f32_n5120_r64_vector_vw4_wg512` | 9224 | 632 | 134 | 0 | 0 | 64 | 13 | 3 |
| `rms_norm_f32_n5376_r16_vector_vw4_wg512` | 9224 | 632 | 134 | 0 | 0 | 64 | 13 | 3 |
| `rms_norm_f32_n5376_r1_vector_vw4_wg512` | 9224 | 608 | 131 | 0 | 0 | 64 | 13 | 3 |
| `rms_norm_f32_n5376_r64_vector_vw4_wg512` | 9224 | 632 | 134 | 0 | 0 | 64 | 13 | 3 |
| `swiglu_f32_split_n14336_r1_wg256` | 9192 | 104 | 25 | 0 | 0 | 0 | 8 | 3 |
| `swiglu_f32_split_n768_r128_wg256` | 9192 | 104 | 25 | 0 | 0 | 0 | 8 | 3 |
| `swiglu_f32_split_n768_r512_wg256` | 9192 | 104 | 25 | 0 | 0 | 0 | 8 | 3 |
| `swiglu_f32_split_n768_r8_wg256` | 9192 | 104 | 25 | 0 | 0 | 0 | 8 | 3 |

## Basket Validation

Full coverage basket after this route slice:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-15-20260612-195539
```

Result: 33/33 prompt runs passed.

Scheduler reduction versus the prior clean baseline:

| Metric | Previous | This slice | Delta |
| --- | ---: | ---: | ---: |
| CPU compute fallbacks | 300420 | 264240 | -36180 |
| HRX20 compute nodes | 180864 | 217044 | +36180 |
| Infrastructure blockers | 32112 | 32112 | 0 |

New route dispatch counts in the full basket:

| Count | Op | Route |
| ---: | --- | --- |
| 1176 | `RMS_NORM` | `rms_norm_f32_n2048_r1_vector_vw4_wg512` |
| 1128 | `ADD` | `add_f32_n2048_r16_rhsrowstride_wg256` |
| 1128 | `ADD` | `add_f32_n2048_r64_rhsrowstride_wg256` |
| 1004 | `RMS_NORM` | `rms_norm_f32_n5376_r1_vector_vw4_wg512` |
| 720 | `RMS_NORM` | `rms_norm_f32_n5120_r1_vector_vw4_wg512` |
| 285 | `RMS_NORM` | `rms_norm_f32_n2048_r16_vector_vw4_wg512` |
| 285 | `RMS_NORM` | `rms_norm_f32_n2048_r64_vector_vw4_wg512` |
| 245 | `RMS_NORM` | `rms_norm_f32_n5376_r16_vector_vw4_wg512` |
| 245 | `RMS_NORM` | `rms_norm_f32_n5376_r64_vector_vw4_wg512` |
| 174 | `RMS_NORM` | `rms_norm_f32_n5120_r16_vector_vw4_wg512` |
| 174 | `RMS_NORM` | `rms_norm_f32_n5120_r64_vector_vw4_wg512` |

Split SWIGLU routes are selected in focused ggml validation but remain
CPU-assigned in full model graphs because adjacent matmul/MoE nodes still form
CPU graph islands. This is route-selection context, not a correctness failure.

## Remaining Top Fallbacks

The basket is now dominated by attention and matmul/MoE families:

- f16 attention `MUL_MAT` and `SOFT_MAX`;
- quantized `MUL_MAT` for Q4_K/Q5_K/Q6_K;
- `ROPE` decode and narrow shapes;
- `MUL_MAT_ID` MoE paths;
- small MoE support ops: `ARGSORT`, `GET_ROWS`, `SUM_ROWS`, `CLAMP`, `DIV`;
- RMS_NORM attention-head shapes such as `128x32`, `128x4`, and prefill
  variants.

## Checkpoint Decision

Phase 1 can continue unattended at a larger scale. This slice did not expose a
new infrastructure blocker. The next route slice should prioritize attention
support (`ROPE`, `SOFT_MAX`, f16 matmuls) or quant/MoE matmuls, and should keep
the same acceptance gates: focused CPU-reference rows, compile reports,
manifest capture, route traces, and full basket smoke.
