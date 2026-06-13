# HRX2 Phase 1.0 Route Slice 37

Date: 2026-06-13

llama.cpp checkpoint: pending at write time.

## Scope

This slice fixes HRX2 runtime loading for existing ROPE catalog routes. No new
Loom source or catalog row was required.

## Root Cause

`ggml_backend_hrx2_catalog_find_routes` clears its output vector before adding
matches. HRX2 device initialization called it twice for `ROPE`, once for
`rope_neox_f32` and once for `rope_f32`, with the same
`device_context->rope_routes` vector. The second call discarded all NEOX
routes and left only the normal/frequency route loaded.

The symptom was that focused exported Qwen3 ROPE rows were structurally
supported by the C++ predicate but every no-`src2` NEOX row was rejected at
route-selection time because the relevant providers were absent from the
runtime route vector.

## Fix

Load all `ROPE` catalog families in one op-wide lookup:

```text
family = null
op = ROPE
```

This preserves both NEOX and normal/frequency ROPE families in the same
runtime route vector while keeping route selection data-driven by catalog
metadata.

## Focused Validation

Exported Qwen3 UD-Q4 ROPE rows:

```text
cache/hrx2/phase1_0/route-slice-37-rope-mode-export-current/qwen3_ud_q4_rope_ops.txt
```

Focused CPU-reference replay:

```text
cache/hrx2/phase1_0/route-slice-37-rope-loader-fix-current
```

Result: all four focused rows passed `test-backend-ops` against the ggml CPU
reference and selected the intended HRX2 NEOX routes:

| Shape | Route |
| --- | --- |
| `ncols=128,nheads=4,ntokens=1` | `rope_neox_f32_n128_h4_t1_wg256` |
| `ncols=128,nheads=4,ntokens=64` | `rope_neox_f32_n128_h4_t64_wg256` |
| `ncols=128,nheads=32,ntokens=1` | `rope_neox_f32_n128_h32_t1_wg256` |
| `ncols=128,nheads=32,ntokens=64` | `rope_neox_f32_n128_h32_t64_wg256` |

There were no `provider_unavailable` events.

## Compile Report Summary

| Route | HSACO | Code bytes | Inst | Spills | Private | Local | Peak live | Global mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rope_neox_f32_n128_h32_t1_wg256` | 9184 | 280 | 61 | 0 | 0 | 0 | 10 | 4 |
| `rope_neox_f32_n128_h32_t64_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |
| `rope_neox_f32_n128_h4_t1_wg256` | 9184 | 268 | 57 | 0 | 0 | 0 | 9 | 4 |
| `rope_neox_f32_n128_h4_t64_wg256` | 9184 | 280 | 60 | 0 | 0 | 0 | 10 | 5 |

## Model Smoke

Qwen3 30B A3B UD-Q4 decode, narrow, and prefill64 smoke:

```text
cache/hrx2/phase1_0/route-slice-37-rope-loader-fix-current/qwen3-smoke
```

Result: all three regimes passed.

Route dispatch counts:

| Regime | Provider unavailable | ROPE dispatches |
| --- | ---: | --- |
| decode | 0 | 96 `h32_t1`, 96 `h4_t1` |
| narrow | 0 | 48 `h32_t16`, 48 `h4_t16`, 48 `h32_t1`, 48 `h4_t1` |
| prefill64 | 0 | 48 `h32_t64`, 48 `h4_t64`, 48 `h32_t1`, 48 `h4_t1` |

Scheduler reduction for the Qwen3 smoke:

| Metric | Count |
| --- | ---: |
| graph nodes | 83886 |
| HRX20 compute nodes | 45000 |
| CPU compute fallbacks | 26064 |
| infrastructure blockers | 3456 |

The top remaining fallbacks in this model are now MoE `MUL_MAT_ID`, quantized
`MUL_MAT`, attention `MUL_MAT`/`SOFT_MAX`, and MoE support ops. No NEOX ROPE
fallback remains for the covered Qwen3 shapes.

## Checkpoint Decision

This is a validated runtime-loader fix and should be committed separately from
new kernel-family work. It also establishes a rule for future multi-family ops:
if a runtime vector intentionally stores every provider for one ggml op, load
by op in a single catalog lookup or append explicitly; do not repeatedly call a
clearing helper with the same destination vector.
