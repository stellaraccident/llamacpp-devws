# HRX2 Phase 1.0 Route Slice 33: Normal ROPE With Frequency Factors

Date: 2026-06-13

## Scope

This slice fixes a ROPE coverage miss discovered while trying to unblock the
Q4_K matmul route in Llama 3.2 3B Q4_K_M. The earlier frequency-factor route
covered NeoX layout, but the exported Llama 3.2 graph uses
`GGML_ROPE_TYPE_NORMAL` with `src2` frequency factors:

```text
cache/hrx2/phase1_0/route-slice-33-rope-freq-export-current/llama32-q4k-rope-ops.txt
```

The accepted route adds a target-neutral Loom root
`@hrx2_rope_normal_f32_freq` in the same ROPE frequency-factor source and a
new `rope_f32` catalog family. Runtime route matching now parses
`supports.mode` from catalog JSON and filters normal versus NeoX in generic
C++ route plumbing.

## Prior Art

CUDA, CPU, Metal, and Vulkan all treat normal and NeoX ROPE as different
pairing layouts. Normal mode rotates adjacent pairs:

```text
x[i0 + 0], x[i0 + 1]
```

NeoX mode rotates split-half pairs:

```text
x[i0 / 2], x[i0 / 2 + n_dims / 2]
```

The model export showed `mode=0`, `n_dims=128`, `src2=f32[64]`,
`freq_base=500000`, `freq_scale=1`, and `ext_factor=0`.

## Validation

Focused exact graph-op replay:

```text
cache/hrx2/phase1_0/route-slice-33-rope-normal-focused-after-family-cleanup
```

Result: 4/4 exact exported rows passed `test-backend-ops test -b HRX20 -o
ROPE --test-file ...` against ggml CPU reference.

Compile-report summary:

- `spill_count = 0`
- `spill_plan_count = 0`
- `private_bytes = 0`
- `local_bytes = 0`
- instruction count: 65-70
- peak live units: 12

Build validation:

```text
python3 sources/llama.cpp/ggml/src/ggml-hrx2/tools/validate_hrx2_catalog.py \
  --catalog build/llama-hrx2/ggml/src/ggml-hrx2/generated/catalog.json \
  --source-root sources/llama.cpp/ggml/src/ggml-hrx2

cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops -j$(nproc)
```

## Model Smoke

Final Llama 3.2 Q4_K_M smoke:

```text
cache/hrx2/phase1_0/route-slice-33-rope-normal-smoke-after-family-cleanup
```

Regimes:

- decode: `p=1`, `n=1`
- narrow: `p=16`, `n=1`
- prefill64: `p=64`, `n=1`

All three runs passed. Scheduler reduction:

- HRX20 compute nodes: 13,176
- CPU compute fallbacks: 9,216
- K-head ROPE `128x8x{1,16,64}x1` now dispatches through
  `rope_normal_f32_freq_n128_d128_h8_24_t1_64_wg256`.
- Q-head ROPE `128x24x{1,16,64}x1` is now listed as `supported_by=HRX20,CPU`
  but remains CPU-assigned with the Q4_K matmul island.

## Follow-Up

The remaining top fallbacks for this model are Q4_K and Q6_K `MUL_MAT`, GLU,
and GET_ROWS. Q4_K rows are already HRX-supported in the scheduler but still
CPU-assigned, so the next slice should focus on breaking the CPU island rather
than adding another ROPE variant.
