# HRX2 Phase 1.0 Route Slice 44: Large Split GLU Coverage

## Summary

This slice adds target-neutral split-source GLU coverage for the remaining
large FFN activation rows in the Phase 1 basket:

- SWIGLU: `ncols=13824/14336/18944/32768`, `nrows=1/16/64`
- GEGLU: `ncols=21504`, `nrows=1/16/64`

The runtime now carries `supports.glu_op` from catalog JSON into route
selection, so SWIGLU and GEGLU can share the same GLU family/ABI without route
ambiguity.

## Implementation

llama.cpp changes:

- `ggml_backend_hrx2_kernel_route::supports_glu_op`
- catalog parser support for `supports.glu_op`
- GLU shape extraction records the ggml GLU op
- GLU route matching filters on `supports_glu_op`
- GLU provider cache keys and dispatch traces include `glu_op`
- `swiglu_f32.loom` adds `hrx2_geglu_f32_split`
- catalog adds:
  - `swiglu_f32_split_large_n13824_32768_r1_64_wg256`
  - `geglu_f32_split_n21504_r1_64_wg256`

The GEGLU source originally used `scalar.geluf<tanh>`, but the AMDGPU lowering
path emitted a `scalar.tanhf` target-low error. The accepted source spells the
same tanh-GELU formula as:

```text
gelu_tanh(x) = x * logistic(2 * sqrt(2/pi) * x * (1 + 0.044715*x*x))
```

This preserves ggml's tanh-GELU semantics while using the supported Loom
`scalar.logisticf` path.

## Focused Validation

Focused test file:

```text
cache/hrx2/phase1_0/route-slice-44-glu-large/glu-large-focused-ops.txt
```

Final focused CPU-reference run:

```text
cache/hrx2/phase1_0/route-slice-44-glu-large/test-trace-logistic.csv
```

Result:

- 15/15 exact exported graph-op rows passed against ggml CPU reference.
- HRX2 dispatch trace selected:
  - 12 `swiglu_f32_split_large_n13824_32768_r1_64_wg256` rows
  - 3 `geglu_f32_split_n21504_r1_64_wg256` rows

Trace:

```text
cache/hrx2/phase1_0/route-slice-44-glu-large/hrx2-logistic.jsonl
```

Compile-report summary:

| Route | Rows | Peak live | Instructions | Private | Local |
| --- | ---: | ---: | ---: | ---: | ---: |
| `swiglu_f32_split_large_n13824_32768_r1_64_wg256` | 12 | 8 | 15 | 0 | 0 |
| `geglu_f32_split_n21504_r1_64_wg256` | 3 | 8 | 20 | 0 | 0 |

Evidence:

```text
cache/hrx2/phase1_0/route-slice-44-glu-large/evidence-trace-logistic
```

## Model Smoke

Targeted smoke set:

```text
cache/hrx2/phase1_0/route-slice-44-glu-large/model-smoke-20260613-current
```

Models/regimes:

- DeepSeek Q4_K_M, p=1/16/64
- Qwen2.5 Q5_K_M, p=1/16/64
- Llama 3.1 Q8_0, p=1/16/64
- Mistral Q4_K_M, p=1/16/64
- Gemma Q4_K_M, p=1/16/64

All 15 runs passed. The targeted scheduler reduction has no GLU compute
fallback. HRX2 GLU dispatch count was 1260 across the new routes.

## Full Basket

Full 11-model basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-44-20260613-current
```

All 33 runs passed.

Aggregate delta from route slice 43:

| Metric | Slice 43 | Slice 44 | Delta |
| --- | ---: | ---: | ---: |
| accelerated | 437740 | 444912 | +7172 |
| compute fallback | 11432 | 4260 | -7172 |
| HRX20 compute | 469852 | 477024 | +7172 |
| CPU compute | 11432 | 4260 | -7172 |

The full-basket route trace shows 2676 GLU dispatches. No GLU row remains in
`top_compute_fallbacks`; remaining compute fallbacks are now dominated by
normal ROPE and GET_ROWS shapes.

## Notes

- The broad SWIGLU route intentionally supersedes older exact split rows for
  large dense FFN widths. Existing MoE `ncols=768` and Llama 3.2 `ncols=8192`
  routes are unchanged.
- This is coverage-quality GLU source. The per-element schedule is acceptable
  for Phase 1 unfused coverage, but Phase 2 fusion work should normally absorb
  GLU into FFN matmul/fusion candidates.
