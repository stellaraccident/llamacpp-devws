# HRX2 Phase 1 Final Report

Date: 2026-06-13

## Status

Phase 1 unfused compute-kernel coverage is complete for the HRX2 11-model
coverage basket. The final full basket run passed all decode, narrow, and
prefill64 regimes and the graph-node scheduler reduction shows no unexplained
CPU compute fallbacks.

Final evidence:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836
```

Summary:

| Metric | Count |
| --- | ---: |
| basket runs | 33 |
| valid `llama-bench` JSON outputs | 33 |
| graph nodes | 481284 |
| HRX20 compute nodes | 481284 |
| CPU compute fallbacks | 0 |
| infrastructure blockers | 32112 |

`top_compute_fallbacks` is empty and `cpu_assigned_but_hrx_supported` is empty.

## Accepted Route Surface

The production catalog contains 162 HRX2 route rows:

| Family | Routes |
| --- | ---: |
| `add_f32` | 6 |
| `argsort_f32_i32` | 3 |
| `clamp_f32` | 3 |
| `cont_f32` | 5 |
| `div_f32` | 3 |
| `get_rows_f32` | 6 |
| `get_rows_moe_weights_f32` | 3 |
| `get_rows_q4_k_f32` | 3 |
| `get_rows_q5_k_f32` | 1 |
| `get_rows_q6_k_f32` | 3 |
| `get_rows_q8_0_f32` | 1 |
| `mul_f32` | 5 |
| `mul_mat_f16_f32_batched` | 1 |
| `mul_mat_f32_f32` | 1 |
| `mul_mat_id_q4_k_f32` | 1 |
| `mul_mat_id_q5_k_f32` | 1 |
| `mul_mat_id_q6_k_f32` | 1 |
| `mul_mat_q4_k_f32` | 1 |
| `mul_mat_q5_k_f32` | 1 |
| `mul_mat_q6_k_f32` | 1 |
| `mul_mat_q8_0_f32` | 10 |
| `rms_norm_f32` | 32 |
| `rope_f32` | 4 |
| `rope_neox_f32` | 33 |
| `scale_f32` | 3 |
| `set_rows_f32` | 2 |
| `soft_max_f32` | 15 |
| `sum_rows_f32` | 3 |
| `swiglu_f32` | 10 |

Route-slice docs under `docs/loom/llamacpp-hrx2-phase1.0-route-slice-*.md`
record the focused validation, compile reports, manifests, route traces, and
basket deltas for the accepted slices.

## Final Slice

The final accepted route is:

```text
rope_normal_f32_freq_n128_d128_h32_t1_64_wg256
```

Focused evidence:

```text
cache/hrx2/phase1_0/route-slice-48-rope-normal-h32-p64/focused-final-20260613-044740
```

Compile report summary:

| `ntokens` | HSACO bytes | Code bytes | Inst | Spills | Private | Local | Peak live | Global mem |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9200 | 352 | 76 | 0 | 0 | 0 | 15 | 5 |
| 16 | 9200 | 352 | 75 | 0 | 0 | 0 | 15 | 6 |
| 64 | 9200 | 352 | 75 | 0 | 0 | 0 | 15 | 6 |

This route resolves the earlier normal-frequency ROPE h32/p64 numeric parity
failure by spelling the per-pair theta update as a CPU-like recurrence.

## Remaining CPU Work

The remaining scheduler class is `infrastructure_blocker`, currently 32112
graph nodes. In the final basket dispatch trace this corresponds to the
deferred `SET_ROWS` host path, visible as `host_fallback_set_rows_f32_f16`.
This is treated as host orchestration/deferred infrastructure for Phase 1, not
as an unexplained missing unfused compute kernel.

Phase 2 may choose to eliminate or fuse this traffic, especially around
attention-cache update paths, but it does not block Phase 1 compute coverage.

## Rejected Or Deferred Candidates

- Broad generic RMS_NORM route: removed because it overclaimed unvalidated
  shapes and failed at JIT compile time during basket smoke.
- Q8_0 pointer-overlap support guards: removed from the scheduler-visible
  predicate after model-level execution proved they could reject allocated
  graphs after scheduler assignment.
- Normal-frequency ROPE h32/p64 independent theta expression: rejected after
  strict ggml CPU-reference failure; replaced by recurrence spelling in slice
  48.
- Q8_0 larger-shape performance: route coverage is accepted, but performance
  refutation remains a separate optimization task for the Phase 2/performance
  pass.
- High/low Loom interop and low-ASM escape paths: documented as tooling work
  for kernels that cannot yet express the desired schedule in high-level Loom.

## Validation Commands

Catalog and artifact validation:

```bash
python3 sources/llama.cpp/ggml/src/ggml-hrx2/tools/validate_hrx2_catalog.py \
  --catalog build/llama-hrx2/ggml/src/ggml-hrx2/generated/catalog.json \
  --source-root sources/llama.cpp/ggml/src/ggml-hrx2 \
  --artifact-root build/llama-hrx2/ggml/src/ggml-hrx2/generated/catalog \
  --require-artifacts
```

Build validation:

```bash
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench -j$(nproc)
```

Focused final route validation:

```bash
GGML_HRX2_TRACE_JSONL="$OUT/hrx2.jsonl" \
GGML_HRX2_EVIDENCE_DIR="$OUT/evidence" \
GGML_HRX2_DUMP_COMPILE_REPORT=1 \
GGML_HRX2_DUMP_MANIFEST=1 \
build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/rope-normal-freq-h32-ops.txt \
  --output csv
```

Scheduler reduction:

```bash
python3 tools/hrx2_reduce_sched_trace.py \
  cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836/*/sched.jsonl \
  --json-out cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836/summary.json \
  --md-out cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836/summary.md
```

## Phase 2 Readiness

Phase 2 fusion selection can begin unattended from a coverage standpoint. The
backend now has standalone unfused routes for the basket's compute kernels, so
fusion candidates can be evaluated against measured selected unfused parts
instead of against CPU islands. The next pass should focus on performance
refutation, representative device-time benchmarking, and measured fusion
acceptance rather than expanding Phase 1 coverage.
