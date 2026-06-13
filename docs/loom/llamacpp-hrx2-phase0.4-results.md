# HRX2 Phase 0.4 Mini Tuning Results

Date: 2026-06-11

This pass intentionally reduced scope to a tiny, bounded Loom benchmark loop
instead of attempting a broad phase 1-style sweep.

## What Ran

The new miniature harness is:

```bash
./tools/hrx2_mini_tune.py \
  --run-id phase0.4-mini \
  --workgroup-sizes 128,256,512 \
  --iterations 5 \
  --warmup-iterations 1 \
  --timeout 15
```

It reads the checked-in RMS_NORM Loom source, emits generated workgroup-size
variants under `cache/hrx2/mini_tune/<run-id>/variants/`, then runs
`iree-benchmark-loom --measure=dispatch_complete` with
`--sample-compilation=per_sample` for decode and small-prefill benchmark cases.

The run writes:

- `cache/hrx2/mini_tune/phase0.4-mini/results.jsonl`
- `cache/hrx2/mini_tune/phase0.4-mini/summary.json`
- `cache/hrx2/mini_tune/phase0.4-mini/summary.md`
- per-candidate raw Loom benchmark JSONL under
  `cache/hrx2/mini_tune/phase0.4-mini/benchmark_jsonl/`

## Result

The mini run completed 6/6 candidates successfully.

| Benchmark | WG | p50 ns | p90 ns | Inst | Code bytes | Spills | Peak live |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hrx2_rms_norm_f32_decode` | 128 | 26631 | 27181 | 102 | 500 | 0 | 11 |
| `hrx2_rms_norm_f32_decode` | 256 | 23771 | 24960 | 107 | 524 | 0 | 11 |
| `hrx2_rms_norm_f32_decode` | 512 | 22290 | 23391 | 112 | 548 | 0 | 11 |
| `hrx2_rms_norm_f32_small_prefill` | 128 | 25151 | 42171 | 126 | 564 | 0 | 14 |
| `hrx2_rms_norm_f32_small_prefill` | 256 | 22430 | 23901 | 131 | 588 | 0 | 14 |
| `hrx2_rms_norm_f32_small_prefill` | 512 | 22520 | 39161 | 136 | 612 | 0 | 14 |

Winners from this small run:

- Decode: WG 512 at 22290 ns p50
- Small prefill: WG 256 at 22430 ns p50

These timings are not a production decision. The run uses hot input reuse
(`--input-ring-count=1`) and a very small sample count so that the loop is fast
and safe for interactive bringup.

## Lessons

`dispatch_complete` plus `per_sample` is the viable standalone GPU timing path
for the current HRX2 RMS_NORM source. It gives us the control loop we wanted:
generate variants, run Loom validation and timing, collect compile summaries,
and reduce to a candidate choice without llama.cpp in the timing loop.

The Q8_0/F32 Loom benchmark source is not yet ready for the same loop. Its
named benchmark plan currently reports zero dispatch samples even though the
case has a nonzero parameter cartesian product. That should be fixed before
using Q8 as the next miniature target.
