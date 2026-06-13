# HRX2 Phase 1.0 Route Slice 34: Llama 3.2 Split SWIGLU Route

Date: 2026-06-13

## Scope

This slice adds catalog coverage for Llama 3.2 split-source SWIGLU rows:

```text
SWIGLU f32[8192, {1,64}, 1, 1] = f32[8192, {1,64}, 1, 1], f32[8192, {1,64}, 1, 1]
```

No new Loom kernel was required. The existing `@hrx2_swiglu_f32_split` root
already implements split-source SWIGLU. The missing piece was a route domain
for `ncols=8192`, rows `1..64`, and `binding_count=3`.

## Evidence

Exact graph-op rows:

```text
cache/hrx2/phase1_0/route-slice-34-glu-diagnose-current/llama32-q4k-glu-ops.txt
```

Focused replay:

```text
cache/hrx2/phase1_0/route-slice-34-swiglu-focused-current
```

Result: 2/2 exact rows passed HRX2 support and ggml CPU-reference validation.

Compile-report summary:

- `spill_count = 0`
- `spill_plan_count = 0`
- `private_bytes = 0`
- `local_bytes = 0`
- instruction count: 25
- peak live units: 8

## Model Smoke

Llama 3.2 Q4_K_M smoke:

```text
cache/hrx2/phase1_0/route-slice-34-swiglu-smoke-llama32-current
```

Regimes:

- decode: `p=1`, `n=1`
- narrow: `p=16`, `n=1`
- prefill64: `p=64`, `n=1`

All three runs passed. GLU rows moved from CPU-only support to
`supported_by=HRX20,CPU`, but remain CPU-assigned with the larger Q4_K/Q6_K
matmul island. No additional SWIGLU kernel work is indicated for this model
slice.

## Follow-Up

The next effective slice should target the remaining matmul island. The top
fallbacks are Q4_K/Q6_K `MUL_MAT`, Q-head ROPE, and split SWIGLU, but the last
two are now HRX-supported and stay on CPU because the surrounding graph is
still CPU-placed.
