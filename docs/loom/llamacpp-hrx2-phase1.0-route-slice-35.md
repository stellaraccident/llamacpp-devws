# HRX2 Phase 1.0 Route Slice 35

Date: 2026-06-13

## Scope

This slice adds Q6_K/F32 unfused `MUL_MAT` coverage for the dense Q6 paths
seen in the coverage basket. It is a correctness and route-coverage baseline,
not a final quantized-matmul performance refutation.

Accepted route:

```text
mul_mat_q6_k_f32_direct_k256_32768_r1_262144_c1_64_wg256
```

The Loom source is target-neutral. The route is selected by catalog metadata
and JIT-specialized by `k`, `rows`, `cols`, and workgroup size.

## Implementation Notes

- Added `kernels/mul_mat_q6_k_f32.loom`.
- Added embedded bytecode artifact `mul_mat_q6_k_f32_static.loombc`.
- Added runtime Q6_K route loading, support probing, shape extraction, JIT
  config binding, and dispatch.
- The Q6 unpack uses branchless packed-bit arithmetic for low-nibble
  selection. `scf.if` and `scf.select` both exposed current AMDGPU lowering
  gaps for this tiny integer choice.
- The admitted row domain is up to `262144` to cover vocabulary/output rows
  such as Phi-4 `200064x{1,16,64}`.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-35-q6-focused-current/mul_mat_q6_k_f32_ops.txt
```

It contains 10 real-trace Q6 rows covering:

- attention K/V: `k=2048, rows=512, cols=1/16/64`;
- attention/FFN: `k=3072, rows=1024, cols=1/16/64`;
- output projection: `k=3072, rows=200064, cols=1`;
- FFN down: `k=8192, rows=3072, cols=1/16/64`.

Support validation:

```text
cache/hrx2/phase1_0/route-slice-35-q6-focused-current/support.csv
```

Result: 10/10 rows supported by HRX2.

CPU-reference validation:

```text
cache/hrx2/phase1_0/route-slice-35-q6-focused-current/test-branchless.csv
```

Result: 10/10 rows passed against ggml CPU reference. HRX2 trace selected the
intended Q6 route for all rows.

## Compile Report Summary

Evidence root:

```text
cache/hrx2/phase1_0/route-slice-35-q6-focused-current/evidence-branchless
```

Across the 10 focused shapes:

| Field | Values |
| --- | --- |
| HSACO bytes | `9208` |
| Emitted instructions | `154-159` |
| Code bytes | `784-816` |
| Spills | `0` |
| Private memory | `0` |
| Local memory | `32` |
| Peak live units | `19-20` |

These are acceptable for a phase-one coverage baseline.

## Model Smoke

Phi-4 three-regime smoke:

```text
cache/hrx2/phase1_0/route-slice-35-q6-smoke-phi4-current
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

The full model scheduler reports Q6 matmuls as `supported_by=HRX20,CPU` but
keeps them CPU-assigned because adjacent Q4_K/Q5_K/GLU graph islands remain on
CPU. This is not a Q6 route correctness failure; focused validation is the
acceptance evidence for this standalone route.

## Follow-Up

- Add Q5_K coverage next; it remains a top Phi-4 fallback and blocks model
  placement around the same FFN islands.
- Add or retune Q4_K route coverage for the observed Phi-4 row buckets so Q6
  routes can become model-selected rather than merely supported.
- Revisit Q6 performance after Phase 1 coverage closes. The direct route is
  intentionally simple; future axes should include packed/vectorized RHS,
  scale reuse, rows/cols per workgroup, and comparison against CUDA/HIPified
  and old HRX quantized-matmul references.
