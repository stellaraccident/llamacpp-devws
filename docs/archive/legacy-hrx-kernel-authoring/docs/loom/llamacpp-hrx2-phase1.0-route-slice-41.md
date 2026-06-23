# HRX2 Phase 1.0 Route Slice 41: Wider Q8_0 Direct Matmul Domains

Date: 2026-06-13

## Scope

This checkpoint closes the Q8_0 direct `MUL_MAT` fallback shapes that remained
after indexed quantized matmul coverage landed in slice 40.

The accepted change widens the existing target-neutral Q8_0/F32 Loom family and
catalog domains:

- `k`: `32..8192` to `32..32768`
- `rows`: `1..8192` to `1..262144`

No new target-specific source was added. The source still relies on metadata
and route config for target and shape selection.

## Evidence

Focused Llama 3.1 Q8_0 smoke after widening both row and K domains:

```text
cache/hrx2/phase1_0/route-slice-41-q8-wide-rows-current/llama31-after-k-widen
```

Result:

- decode, narrow, and prefill64 regimes passed;
- q8_0 direct `MUL_MAT` scheduler placements moved to HRX2;
- no q8_0 direct `MUL_MAT` CPU fallback remained in the focused trace.

Full 11-model basket smoke:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-41-20260613-012314
```

Result: `33/33` passed across decode `p=1`, narrow `p=16`, and prefill64
`p=64`.

Aggregate:

- nodes: `515064`
- accelerated: `430444`
- host orchestration: `33780`
- infrastructure blocker: `32112`
- compute fallback: `18728`
- HRX20 compute: `462556`
- CPU compute: `18728`

Compared with slice 40, CPU compute fallbacks dropped from `22990` to `18728`.
Q8_0 direct `MUL_MAT` no longer appears in the aggregate fallback list.

## Remaining Backlog

Top remaining compute fallback families after this slice:

- F32/F32 matmul: `MUL_MAT f32,f32` at `128x{1,16,64}x1x1`;
- ROPE with frequency source operand:
  `ROPE f32,i32,f32` at `128x32x{1,16,64}x1`;
- GLU/SWIGLU split coverage for widths
  `13824`, `14336`, `18944`, `21504`, and `32768`;
- remaining no-frequency ROPE variants at `nheads=8` and `nheads=32`;
- GET_ROWS coverage for F32 and quantized state/embedding rows.

## Process Notes

- The first focused attempt widened only `rows`; Llama 3.1 Q8_0 still fell
  back because the model uses `k=14336`, outside the old `k_max=8192` route
  domain.
- The fix is a route/source applicability correction, not a new performance
  claim for the whole widened domain. Q8_0 performance refutation remains
  governed by the existing Q8 notes and common-runner evidence.
- Keep this family target-neutral unless future variants use genuinely
  target-specific primitives or ABI.
