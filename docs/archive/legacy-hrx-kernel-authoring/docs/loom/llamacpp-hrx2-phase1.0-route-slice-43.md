# HRX2 Phase 1.0 Route Slice 43: Normal ROPE Freq H32

Date: 2026-06-13

## Scope

This slice adds the missing Llama 3.1 normal-mode frequency-factor ROPE bucket
for 32 attention heads in decode and narrow prompt regimes.

Accepted route:

```text
rope_normal_f32_freq_n128_d128_h32_t1_16_wg256
```

The route reuses the existing target-neutral
`kernels/rope_neox_f32_freq.loom` source and the
`@hrx2_rope_normal_f32_freq` root. The source stays portable; the catalog
route records the measured `gfx1100` applicability.

## Why This Was Needed

The slice-42 basket aggregate still had frequency-source normal ROPE as the
top compute fallback:

```text
ROPE f32 <- f32,i32,f32 128x32x1  count 1536
ROPE f32 <- f32,i32,f32 128x32x16 count 576
ROPE f32 <- f32,i32,f32 128x32x64 count 192
```

The existing normal-frequency route covered heads 8 through 24. Llama 3.1
exports `mode=0` (`GGML_ROPE_TYPE_NORMAL`) with 32 heads and an F32 frequency
factor tensor, so the NEOX h32 routes were correctly ignored.

## Validation

Focused CPU-reference replay used exact exported Llama 3.1 rows:

```text
cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/rope-normal-freq-h32-t1-t16-ops.txt
```

Passing focused run:

```text
cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/focused-split-t1-t16-20260613-015955
```

Both rows passed:

```text
128x32x1  mode=0 src2=f32
128x32x16 mode=0 src2=f32
```

The route selected in real Llama 3.1 Q4 and Q8 model smokes:

```text
cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/llama31-smoke-split-20260613-020010
```

Dispatch evidence:

```text
Q4 p1:  64 dispatches h32/t1
Q4 p16: 32 dispatches h32/t16, 32 dispatches h32/t1
Q8 p1:  64 dispatches h32/t1
Q8 p16: 32 dispatches h32/t16, 32 dispatches h32/t1
```

Compile-report summary for both accepted specializations:

```text
HSACO bytes: 9200
instructions: 66
code bytes: 304
spills: 0
private bytes: 0
local bytes: 0
peak live units: 12
```

Full 11-model coverage basket passed 33/33:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-43-20260613-020246
```

Aggregate movement versus slice 42:

```text
compute_fallback: 13544 -> 11432
accelerated:       435628 -> 437740
HRX20 compute:     467740 -> 469852
CPU compute:       13544 -> 11432
```

## Rejected Shape

The h32/p64 row selected and compiled, but failed strict ggml CPU-reference
correctness:

```text
cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/focused-20260613-015635
[ROPE] ERR = 0.000007262 > 0.000000100
```

The catalog therefore does not admit `ntokens=64` for h32 normal-frequency
ROPE. The original h8-24/t1-64 route remains intact, so this slice does not
regress Llama 3.1 h8 p64 ROPE coverage.

Remaining aggregate fallback after this slice:

```text
ROPE f32 <- f32,i32,f32 128x32x64 count 192
```

## Notes

- Do not widen normal-frequency h32 to p64 without either tightening numeric
  parity against ggml CPU or intentionally relaxing the op tolerance with a
  documented policy decision.
- This is another example where focused graph-op replay is stronger than
  shape-only synthetic testing: the route compiles and runs, but the p64
  numeric row is not acceptable under the current ggml test gate.
