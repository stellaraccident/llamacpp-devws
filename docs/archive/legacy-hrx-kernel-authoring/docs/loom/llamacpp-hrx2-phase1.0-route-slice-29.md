# HRX2 Phase 1.0 Route Slice 29

Date: 2026-06-12

This checkpoint adds no-`src2` NEOX F32 `ROPE` coverage for the attention
shapes observed in the 11-model Phase 1 basket. It deliberately does not claim
ROPE variants with a frequency-factor `src2`, non-NEOX layouts, YaRN/ext-factor
behavior, or destination type conversions.

## Accepted Scope

Accepted family:

```text
rope_neox_f32
```

Accepted route shape grid:

```text
ncols = 128
nheads = 4, 8, 16, 28, 32, 40
ntokens = 1, 16, 64
workgroup_size = 256
```

The Loom source is target-neutral. The route rows use `target_key=gfx1100`
only as measured catalog metadata.

ABI:

```text
launch params:
  freq_base: f32
  freq_scale: f32
  attn_factor: f32
bindings:
  0: src0 f32
  1: pos i32
  2: dst f32
constants:
  freq_base, freq_scale, attn_factor
```

Algorithm:

- one workitem owns one NEOX pair for one `(token, head)` row;
- pair offsets are `pair` and `pair + ncols / 2`;
- angle uses `pos[token] * freq_base^(-i / ncols) * freq_scale`;
- `powf` is spelled as `expf(logf(freq_base) * exponent)` because
  `scalar.powf<afn>` currently lacks an AMDGPU target-low contract.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-29-rope-focused-20260612-214501/rope_neox_f32_ops.txt
```

Focused CPU-reference evidence:

```text
cache/hrx2/phase1_0/route-slice-29-rope-focused-20260612-214501/test-focused
```

Command:

```bash
LD_LIBRARY_PATH="$PWD/build/llama-hrx2/bin:$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}" \
  GGML_HRX2_TRACE_JSONL="$RUN/hrx2.jsonl" \
  GGML_HRX2_EVIDENCE_DIR="$RUN/evidence" \
  GGML_HRX2_DUMP_COMPILE_REPORT=1 \
  GGML_HRX2_DUMP_MANIFEST=1 \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file "$OUT/rope_neox_f32_ops.txt" --output csv
```

Result: 18/18 focused rows passed against ggml CPU reference. The HRX2 trace
selected all intended route IDs, JIT-compiled 18 providers, dispatched 18
routes, and had no `provider_unavailable` events.

## Compile Report Guardrails

| Route | HSACO | Code bytes | Inst | Spills | Private | Local | Peak live | Global mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rope_neox_f32_n128_h16_t16_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h16_t1_wg256` | 9184 | 280 | 61 | 0 | 0 | 0 | 10 | 4 |
| `rope_neox_f32_n128_h16_t64_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h28_t16_wg256` | 9184 | 312 | 65 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h28_t1_wg256` | 9184 | 328 | 68 | 0 | 0 | 0 | 10 | 4 |
| `rope_neox_f32_n128_h28_t64_wg256` | 9184 | 312 | 65 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h32_t16_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h32_t1_wg256` | 9184 | 280 | 61 | 0 | 0 | 0 | 10 | 4 |
| `rope_neox_f32_n128_h32_t64_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h40_t16_wg256` | 9184 | 296 | 62 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h40_t1_wg256` | 9184 | 312 | 65 | 0 | 0 | 0 | 10 | 4 |
| `rope_neox_f32_n128_h40_t64_wg256` | 9184 | 296 | 62 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h4_t16_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h4_t1_wg256` | 9184 | 268 | 57 | 0 | 0 | 0 | 9 | 4 |
| `rope_neox_f32_n128_h4_t64_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h8_t16_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h8_t1_wg256` | 9184 | 280 | 61 | 0 | 0 | 0 | 10 | 4 |
| `rope_neox_f32_n128_h8_t64_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |

## Basket Validation

Full coverage basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-29-rope-20260612-214635
```

Result: 33/33 prompt runs passed.

Scheduler reduction versus route slice 28:

| Metric | Slice 28 | Slice 29 | Delta |
| --- | ---: | ---: | ---: |
| CPU compute fallbacks | 238536 | 218232 | -20304 |
| HRX20 compute nodes | 242748 | 263052 | +20304 |
| Infrastructure blockers | 32112 | 32112 | 0 |

Route dispatch counts in the full basket:

| Count | Route |
| ---: | --- |
| 824 | `rope_neox_f32_n128_h32_t1_wg256` |
| 688 | `rope_neox_f32_n128_h4_t1_wg256` |
| 248 | `rope_neox_f32_n128_h16_t1_wg256` |
| 206 | `rope_neox_f32_n128_h32_t16_wg256` |
| 206 | `rope_neox_f32_n128_h32_t64_wg256` |
| 192 | `rope_neox_f32_n128_h40_t1_wg256` |
| 192 | `rope_neox_f32_n128_h8_t1_wg256` |
| 172 | `rope_neox_f32_n128_h4_t16_wg256` |
| 172 | `rope_neox_f32_n128_h4_t64_wg256` |
| 112 | `rope_neox_f32_n128_h28_t1_wg256` |
| 62 | `rope_neox_f32_n128_h16_t16_wg256` |
| 62 | `rope_neox_f32_n128_h16_t64_wg256` |
| 48 | `rope_neox_f32_n128_h40_t16_wg256` |
| 48 | `rope_neox_f32_n128_h8_t16_wg256` |
| 48 | `rope_neox_f32_n128_h40_t64_wg256` |
| 48 | `rope_neox_f32_n128_h8_t64_wg256` |
| 28 | `rope_neox_f32_n128_h28_t16_wg256` |
| 28 | `rope_neox_f32_n128_h28_t64_wg256` |

There were 449 provider compiles, 38,041 provider cache hits, 43,842 HRX2
dispatches, and no provider failures in the full basket run.

## Remaining Top Fallbacks

The no-`src2` ROPE rows were removed from the top fallback table. The largest
remaining families are now:

- attention `MUL_MAT` F16/F32 and `SOFT_MAX` shapes;
- quantized `MUL_MAT` for Q4_K/Q5_K/Q6_K;
- `MUL_MAT_ID` MoE paths, which keep accepted MoE support ops in CPU islands;
- `ROPE` with `src2` frequency factors, especially Gemma/Phi-style shapes;
- supported-but-CPU-assigned MoE support ops and GLU routes until the upstream
  `MUL_MAT_ID` island is broken.

## Checkpoint Decision

This route slice is accepted as Phase 1 unfused attention coverage. It has
target-neutral Loom source, embedded bytecode, catalog metadata, focused
CPU-reference correctness, compile reports, manifests, selected-route traces,
and full-basket pass evidence. The next route slice should prioritize
attention F16/F32 matmuls plus `SOFT_MAX`, or the `MUL_MAT_ID`/quantized
matmul families that currently block MoE model-level placement.
