# Q8_0/F32 MUL_MAT Delta Root Cause: gfx1100

Date: 2026-06-12

Correction: the remaining focused WG64 gap described here was later closed for
`k=512`, `rows=64`, `cols=8`, `rows_per_workgroup=1`, `WG64` by the exact-shape
unrolled Loom source documented in
`docs/loom/q8-0-f32-wg64-parity-followup-gfx1100.md`. The measurement-domain
correction and listing evidence below remain useful, but the final focused
WG64 parity result is in that follow-up.

## Bottom Line

The large Q8_0/F32 Loom-vs-HIP delta was primarily a measurement/runner
artifact, not a broad Loom codegen failure. Loading the same Loom target ELF
through the HIP module API and measuring it with `rocprofv3 --kernel-trace`
puts the best Loom WG128 kernel at p50 2.04 us, essentially tied with the HIP
WG128 reference at p50 2.00 us for `k=512`, `rows=64`, `cols=8`.

The intermediate state at the time of this note still had a real WG64 delta:
Loom WG64 was p50 2.00 us vs HIP WG64 p50 1.88 us, about 6%. A no-reduction
isolation kept most of that gap, so the remaining issue looked like
inner-loop/address/scheduling detail, not the workgroup reduction alone. The
later exact-shape unrolled Loom source closed that specific focused gap.

## Why The Earlier Conclusion Was Wrong

The previous decisive-looking comparison mixed timing domains:

- Loom was measured through `iree-benchmark-loom` / IREE HAL dispatch-event
  machinery.
- HIP was measured through a standalone HIP runner and `rocprofv3`.

For sub-5us kernels, that difference was large enough to make a Loom kernel
look roughly 3.3 us when the same code object, loaded through the same HIP
module path as the HIP reference, profiles around 2.0 us. The earlier
WYSIWYG/codegen lessons still stand, but the "best high-level Loom is 1.6-1.8x
behind HIP" performance conclusion does not.

## Common Runner

New harness:

- `tools/hrx2_q8_0_f32_module_runner.py`

The harness generates a single HIP module runner that can load either:

- a Loom target artifact ELF with kernel `q8_0_f32_candidate`; or
- a HIP HSACO with `refute_q8_0_rows1_wg{64,128}_f32`.

It uses one fixture generator, one buffer layout, one launch path
(`hipModuleLaunchKernel`), and one correctness check. The only intentional ABI
difference is that the HIP reference accepts `k`, `rows`, and `cols` as kernel
arguments while the Loom artifacts are exact-shape-specialized.

Smoke command:

```bash
./tools/hrx2_q8_0_f32_module_runner.py \
  --run-id q8-common-smoke-20260612 \
  --iters 100 \
  --warmup 10 \
  --repeats 3
```

The runner's HIP event timings showed all full kernels around 4.6-4.8 us. Those
numbers are useful as a smoke check, but they are not precise enough for this
question. The decisive numbers below are from `rocprofv3 --kernel-trace` on the
same runner.

## Full-Kernel Device-Time Results

All rows use `k=512`, `rows=64`, `cols=8`, 100 warmup launches, and 1000 timed
launches. The DBs contain 1100 kernel dispatches because warmup launches are
included in the trace.

| Variant | Runner | Workgroup | Blocks x cols | p50 | mean | p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Loom `word4_bitunpack_rhsvec_dotf` | HIP module API | 128 | 64 x 8 | 2.04 us | 2.364 us | 2.56 us |
| HIP reference | HIP module API | 128 | 64 x 8 | 2.00 us | 2.503 us | 2.76 us |
| Loom `word4_bitunpack_rhsvec_dotf` | HIP module API | 64 | 64 x 8 | 2.00 us | 2.047 us | 2.12 us |
| HIP reference | HIP module API | 64 | 64 x 8 | 1.88 us | 1.897 us | 1.96 us |

Artifacts:

- Loom WG128 DB:
  `cache/hrx2/q8_0_f32_common_runner/q8-common-smoke-20260612/rocprof_all/q8_common_all_results.db`
- HIP WG128 DB:
  `cache/hrx2/q8_0_f32_common_runner/q8-common-smoke-20260612/rocprof_seq_hip_wg128/q8_common_hip_wg128_results.db`
- Loom WG64 DB:
  `cache/hrx2/q8_0_f32_common_runner/q8-common-smoke-20260612/rocprof_seq_loom_wg64/q8_common_loom_wg64_results.db`
- HIP WG64 DB:
  `cache/hrx2/q8_0_f32_common_runner/q8-common-smoke-20260612/rocprof_seq_hip_wg64/q8_common_hip_wg64_results.db`

Do not use the non-`seq` HIP profile directories from this run; those were
accidentally collected concurrently and were discarded.

## Launch Geometry Check

`rocprofv3` reports `grid_size_x` as total workitems, not block count. The
equivalent launch shape is therefore:

| Variant | `workgroup_size_x` | `grid_size_x` | `grid_size_x / workgroup_size_x` | `grid_size_y` |
| --- | ---: | ---: | ---: | ---: |
| Loom WG128 | 128 | 8192 | 64 blocks | 8 |
| HIP WG128 | 128 | 8192 | 64 blocks | 8 |
| Loom WG64 | 64 | 4096 | 64 blocks | 8 |
| HIP WG64 | 64 | 4096 | 64 blocks | 8 |

This rules out launch geometry as the source of the observed difference.

## Listing Comparison

The updated Loom branch can emit the important packed/mixed inner-loop shape:

- `global_load_b32` for four Q8 bytes;
- `global_load_d16_b16` for the f16 scale;
- `global_load_b128` for four RHS floats;
- `v_bfe_i32` for byte extraction;
- `v_fma_mix_f32` and `v_fmac_f32` in the scale/multiply/accumulate path.

WG64 Loom listing:

- `cache/hrx2/q8_0_f32_tune/q8-stella-focused-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep2/target_listings/r00052f84794d3ae8_c0_per_sample_sample0_target_listing.amdgpu-assembly`
- Compile report: no spills, local memory 8 bytes, 122 instructions, 600 code
  bytes, register-pressure peak 16 live units.
- Function-level instruction highlights: `global_load_b32=1`,
  `global_load_d16_b16=1`, `global_load_b128=1`, `v_bfe_i32=3`,
  `v_fma_mix_f32=4`, `v_fmac_f32=3`, `ds_bpermute_b32=2`,
  `s_barrier=2`.

HIP WG64 listing:

- `cache/hrx2/q8_0_f32_refute/gfx1100-q8-hip-rerun-20260612/hip_asm/q8_0_f32_refute.s`
- Function: `refute_q8_0_rows1_wg64_f32`.
- Function-level instruction highlights: `global_load_b32=1`,
  `global_load_d16_b16=1`, `global_load_b128=1`, `v_bfe_i32=3`,
  `v_fma_mix_f32=4`, `v_fmac_f32=4`, `ds_bpermute_b32=10`,
  `s_barrier=1`, plus explicit scheduling/control instructions such as
  `s_clause` and `s_delay_alu`.

The remaining WG64 gap is not explained by missing packed loads or missing
mixed FMA form. It is now schedule parity: address arithmetic, wait placement,
byte extraction details, and control/reduction shape.

## No-Reduction Isolation

To separate the inner packed loop from the reduction, both kernels were patched
to store a lane-local partial and skip the workgroup reduction. Correctness was
not checked for this experiment because the output is intentionally not the
real dot product.

Scratch artifacts:

- Loom source:
  `cache/hrx2/q8_delta_noreduce/loom_q8_wg64_noreduce.loom`
- Loom ELF:
  `cache/hrx2/q8_delta_noreduce/loom_q8_wg64_noreduce.elf`
- HIP source:
  `cache/hrx2/q8_delta_noreduce/q8_hip_refute_noreduce.hip.cpp`
- HIP HSACO:
  `cache/hrx2/q8_delta_noreduce/q8_hip_refute_noreduce.hsaco`

Results:

| Variant | Workgroup | p50 | mean | p90 |
| --- | ---: | ---: | ---: | ---: |
| Loom no-reduce | 64 | 1.92 us | 2.352 us | 2.56 us |
| HIP no-reduce | 64 | 1.72 us | 2.148 us | 2.52 us |

No-reduce DBs:

- `cache/hrx2/q8_delta_noreduce/rocprof_loom_wg64_noreduce/q8_loom_wg64_noreduce_results.db`
- `cache/hrx2/q8_delta_noreduce/rocprof_hip_wg64_noreduce/q8_hip_wg64_noreduce_results.db`

This keeps roughly 0.20 us of p50 delta even after removing the reduction. The
full-kernel WG64 p50 delta is roughly 0.12 us. So reduction/control is not the
whole issue; the inner packed loop still needs schedule inspection if we want
to chase the last few percent.

## Current Interpretation

Strong claims that survived:

- Loom is WYSIWYG. The source must explicitly say the intended packed/vector
  schedule. Scalar source should not be expected to become a packed Q8 schedule.
- `vector.bitunpacks<8>` is now a viable high-level way to express the Q8
  packed unpack path on this branch.
- Compile reports are useful guardrails: spills, local/private memory,
  register pressure, code bytes, instruction mix, and lowered target metadata
  all helped keep the investigation bounded.

Strong claims that changed:

- The Q8 packed high-level Loom route is not fundamentally 1.6-1.8x behind HIP.
  Under a common HIP module runner and `rocprofv3`, WG128 is effectively tied.
- The primary remaining issue is not "cannot emit dot form" or "cannot emit
  mixed FMA." Those forms are present.
- For this tiny-kernel class, IREE/HAL dispatch-event measurements cannot be
  the final refutation methodology unless calibrated against an equivalent
  code-object runner.

## Next Actions

1. Treat the common HIP module runner, or an equivalent code-object-level
   runner, as mandatory for decisive sub-5us Loom-vs-native comparisons.
2. Ask the Loom author to inspect why the same target artifact looks materially
   slower through `iree-benchmark-loom` dispatch-event profiling than through
   direct HIP module launch and `rocprofv3`.
3. If we chase the final WG64 delta, focus on schedule parity rather than new
   algorithm invention: compare address arithmetic, wait placement, `s_clause`
   / `s_delay_alu`, byte extraction, and reduction/control lowering.
4. Keep low/asm interop in reserve for kernels where the target listing cannot
   be made to match the intended schedule, but do not invoke it as a blanket
   explanation here. The high-level Loom code is already close for this case.
