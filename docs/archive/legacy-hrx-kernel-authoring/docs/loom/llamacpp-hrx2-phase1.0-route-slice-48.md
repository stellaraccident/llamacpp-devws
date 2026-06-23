# HRX2 Phase 1.0 Route Slice 48

Date: 2026-06-13

This slice closes the final unexplained compute fallback in the 11-model HRX2
Phase 1 basket: Llama 3.1 NORMAL frequency-source F32 ROPE with
`ncols=128`, `n_dims=128`, `nheads=32`, and `ntokens=64`.

## Accepted Route

```text
rope_normal_f32_freq_n128_d128_h32_t1_64_wg256
```

The route widens the previous h32 normal-frequency domain from
`ntokens=1..16` to `ntokens=1..64`:

```text
ncols=128
n_dims=128
nheads=32
ntokens=1..64
workgroup_size=256
bindings=4: src0, positions, frequency factors, dst
parameters=7
```

The Loom source remains target-neutral. The catalog row is still selected by
runtime/catalog metadata for the measured gfx target; the source does not carry
a target attribute.

## Implementation Notes

The rejected slice-43 version recomputed the per-pair theta scale with an
independent `exp(log(freq_base) * exponent)` expression. That was close enough
for decode and narrow rows but failed strict ggml CPU-reference validation at
`ntokens=64`:

```text
[ROPE] ERR = 0.000007262 > 0.000000100
```

The accepted source matches ggml's CPU recurrence more closely:

```text
theta = pos
theta_scale = powf(freq_base, -2.0f / n_dims)
for k in 0..pair:
  theta *= theta_scale
theta /= freq_factor[pair]
theta *= freq_scale
```

The source spells `theta_scale` with `logf`/`expf` because current Loom AMDGPU
lowering still does not accept `scalar.powf`, but it uses a single scale and a
pair-index recurrence instead of recomputing each pair independently.

## Focused Validation

Command:

```bash
GGML_HRX2_TRACE_JSONL="$OUT/hrx2.jsonl" \
GGML_HRX2_EVIDENCE_DIR="$OUT/evidence" \
GGML_HRX2_DUMP_COMPILE_REPORT=1 \
GGML_HRX2_DUMP_MANIFEST=1 \
build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/rope-normal-freq-h32-ops.txt \
  --output csv
```

Evidence:

```text
cache/hrx2/phase1_0/route-slice-48-rope-normal-h32-p64/focused-final-20260613-044740
```

Result:

| Row | Supported | Route | Workgroups |
| ---: | --- | --- | ---: |
| `ntokens=1` | yes | `rope_normal_f32_freq_n128_d128_h32_t1_64_wg256` | 8 |
| `ntokens=16` | yes | `rope_normal_f32_freq_n128_d128_h32_t1_64_wg256` | 128 |
| `ntokens=64` | yes | `rope_normal_f32_freq_n128_d128_h32_t1_64_wg256` | 512 |

Trace events: three provider-cache misses, three successful provider compiles,
and three dispatches. There were no provider failures.

## Compile Report Summary

| `ntokens` | HSACO bytes | Code bytes | Inst | Spills | Private | Local | Peak live | Global mem |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9200 | 352 | 76 | 0 | 0 | 0 | 15 | 5 |
| 16 | 9200 | 352 | 75 | 0 | 0 | 0 | 15 | 6 |
| 64 | 9200 | 352 | 75 | 0 | 0 | 0 | 15 | 6 |

Manifest summary: `amdgpu-rdna3`, export `hrx2_rope_normal_f32_freq`, 4
bindings, 7 parameters, 12 constant bytes, workgroup size `256x1x1`, subgroup
size 32.

## Model And Basket Validation

Targeted Llama 3.1 p64 smoke:

```text
cache/hrx2/phase1_0/route-slice-48-rope-normal-h32-p64/llama31-p64-smoke-20260613-044755
```

Result: no compute fallbacks. The scheduler reducer reports 8520 graph nodes,
all compute nodes on HRX20, with 768 deferred infrastructure blockers.

Full 11-model basket:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836
```

Result: 33/33 `llama-bench` JSON outputs are valid and the reduced scheduler
summary reports:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| accelerated | 449172 |
| infrastructure blockers | 32112 |
| HRX20 compute nodes | 481284 |
| CPU compute fallbacks | 0 |

The new route dispatched 384 times in the full basket.

## Decision

Accepted. This route closes the final unexplained CPU compute fallback in the
Phase 1 basket. The earlier h32/p64 numeric-parity limitation is resolved by
matching the CPU recurrence spelling.
