# HRX2 Phase 1.0 Route Slice 36

Date: 2026-06-13

## Scope

This slice adds Q5_K/F32 unfused `MUL_MAT` coverage for the Phi-4 attention
projection bucket seen in the coverage basket. It is a correctness and
route-coverage baseline, not a final quantized-matmul performance refutation.

Accepted route:

```text
mul_mat_q5_k_f32_direct_k256_32768_r1_262144_c1_64_wg256
```

The Loom source is target-neutral. The route is selected by catalog metadata
and JIT-specialized by `k`, `rows`, `cols`, and workgroup size.

## Implementation Notes

- Added `kernels/mul_mat_q5_k_f32.loom`.
- Added embedded bytecode artifact `mul_mat_q5_k_f32_static.loombc`.
- Added runtime Q5_K route loading, support probing, shape extraction, JIT
  config binding, and dispatch.
- The source reuses the Q4_K scale/min decoding pattern and explicitly adds
  Q5 high-bit extraction from `qh`.
- Low-nibble selection is branchless:
  `((q_byte >> ((group % 2) * 4)) & 0xf)`.
- The admitted row domain is up to `262144`, matching the Q5/Q6 coverage
  baseline and avoiding an artificial cap for future output rows.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-36-q5-focused-current/mul_mat_q5_k_f32_ops.txt
```

It contains 3 real-trace Q5 rows covering Phi-4 `wqkv`:

- `k=3072`, `rows=5120`, `cols=1`;
- `k=3072`, `rows=5120`, `cols=16`;
- `k=3072`, `rows=5120`, `cols=64`.

Support validation:

```text
cache/hrx2/phase1_0/route-slice-36-q5-focused-current/support.csv
```

Result: 3/3 rows supported by HRX2.

CPU-reference validation:

```text
cache/hrx2/phase1_0/route-slice-36-q5-focused-current/test.csv
```

Result: 3/3 rows passed against ggml CPU reference. HRX2 trace selected the
intended Q5 route for all rows.

## Compile Report Summary

Evidence root:

```text
cache/hrx2/phase1_0/route-slice-36-q5-focused-current/evidence
```

Across the 3 focused shapes:

| Field | Values |
| --- | --- |
| HSACO bytes | `9208` |
| Emitted instructions | `204-209` |
| Code bytes | `1080-1108` |
| Spills | `0` |
| Private memory | `0` |
| Local memory | `32` |
| Peak live units | `25` |

These are acceptable for a phase-one coverage baseline.

## Model Smoke

Phi-4 three-regime smoke:

```text
cache/hrx2/phase1_0/route-slice-36-q5-smoke-phi4-current
```

Regimes:

| Regime | Prompt | Decode | Status |
| --- | ---: | ---: | ---: |
| decode | 1 | 1 | 0 |
| narrow | 16 | 1 | 0 |
| prefill64 | 64 | 1 | 0 |

Scheduler reduction:

| Metric | Count |
| --- | ---: |
| graph nodes | 32226 |
| HRX20 compute nodes | 16200 |
| CPU compute fallbacks | 8208 |
| infrastructure blockers | 2304 |

The full model scheduler reports Q5_K matmuls as `supported_by=HRX20,CPU` but
keeps them CPU-assigned because adjacent Q4_K/Q6_K/GLU graph islands remain on
CPU. This is not a Q5 route correctness failure; focused validation is the
acceptance evidence for this standalone route.

## Follow-Up

- Expand Q4_K route coverage for the observed Phi-4 row buckets so Q5/Q6
  support can become model-selected in the same graph island.
- Re-run a full basket after the next major quantized-matmul slice to surface
  any Q5 shapes outside the Phi-4 `wqkv` bucket.
- Revisit Q5 performance after Phase 1 coverage closes. The direct route is
  intentionally simple; future axes should include packed/vectorized RHS,
  scale reuse, rows/cols per workgroup, and comparison against CUDA/HIPified,
  Metal, Vulkan, and old HRX quantized-matmul references.
