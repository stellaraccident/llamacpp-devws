# Q8_0/F32 WG64 Parity Follow-Up: gfx1100

Date: 2026-06-12

## Current Status

The focused goal-shape WG64 Loom kernel now reaches HIP WG64 parity with the
exact-shape unrolled Loom variant:

| Variant | Runs | p50 | Mean range | Launch metadata |
| --- | ---: | ---: | ---: | --- |
| HIP WG64 reference | 3 | 1.96 us | 2.224-2.362 us | WG64, `grid_size_x=4096`, `grid_size_y=8`, 1100 calls |
| Loom WG64 unrolled bitunpack | 3 | 1.84-1.96 us | 1.868-2.296 us | WG64, `grid_size_x=4096`, `grid_size_y=8`, 1100 calls |
| Loom WG64 `scf.for unroll(%factor)` bitunpack | 3 | 1.96 us | 2.233-2.357 us | WG64, `grid_size_x=4096`, `grid_size_y=8`, 1100 calls |

This satisfies the narrow parity objective for:

```text
k=512, rows=64, cols=8, rows_per_workgroup=1, cols_per_group=1, WG64
```

The previous looped-source same-runner baseline was:

| Variant | Runs | p50 | Mean range | Launch metadata |
| --- | ---: | ---: | ---: | --- |
| HIP WG64 reference | 3 | 1.96 us | 2.262-2.362 us | WG64, `grid_size_x=4096`, `grid_size_y=8`, 1100 calls |
| Loom WG64 bitunpack | 3 | 2.12 us | 2.392-2.465 us | WG64, `grid_size_x=4096`, `grid_size_y=8`, 1100 calls |

That reproduced a real same-runner gap of about 8% p50 on that machine state.
The unrolled variant closes the narrow goal-shape gap without changing ABI,
fixture, launch geometry, or semantics.

However, this tiny `512x64x8` shape is too close to the profiler/timer floor to
be a representative final target. A larger exact-shape rerun at `4096x128x8`
shows a much clearer throughput gap:

| Variant | Runs | p50 | Mean range | Launch metadata |
| --- | ---: | ---: | ---: | --- |
| HIP WG64 reference, `4096x128x8` | 3 | 4.12-4.16 us | 4.267-4.299 us | WG64, `grid_size_x=8192`, `grid_size_y=8`, 300 calls |
| Loom WG64 bitunpack, `4096x128x8` | 3 | 7.88 us | 7.959-7.973 us | WG64, `grid_size_x=8192`, `grid_size_y=8`, 300 calls |

This means the small-shape 8% gap should be treated as a schedule microscope,
not as the final performance judgment. For representative large shapes, the
same high-level Loom schedule is currently about 1.9x slower than the HIP
schedule.

The Q8 tuner default basket now uses larger acceptance shapes by default:

```text
512x64x8,4096x128x1,4096x128x8,4096x512x1,4096x512x8,8192x128x8
```

Use `512x64x8` for listing/schedule diagnosis only. Use `4096x128x8` or larger
as the first acceptance target, because the tiny shape is close enough to timer
floor and profiler granularity that it can understate real throughput gaps.

The production HRX2 source is intentionally target-neutral. The gfx1100 labels
in this note and in the accepted catalog rows are route/evidence metadata, not
source applicability constraints. Do not add `target(@...)`,
`amdgpu.target<...>`, or a gfx-specific source attribute unless a future Q8
variant uses a genuinely target-specific primitive, layout, or low/rocasm
block. For the current packed high-level Q8 schedule, the source stays portable
and the catalog `target_key` selects measured winners.

Benchmark interpretation has two distinct domains:

- Same-runner hot-loop device time is a root-cause tool. It launches Loom and
  HIP code objects through the same path over the same buffers, so it is good
  for emitted-code parity, but it can benefit from hot kernargs and cached data.
- Rotated-buffer Loom/tool timing is closer to an acceptance benchmark. Loom's
  benchmark tooling intentionally rotates buffers to avoid common benchmark
  mistakes. Its default operation timestamps may include host round trips, so
  tables must name whether they are using device kernel timestamps or
  tool/operation intervals.

For hill climbing among Loom candidates, Loom's own benchmark/profile path is
the right tool as long as comparisons stay inside that self-consistent
measurement universe. For Loom-vs-existing-HIP refutation, the formal comparison
should stay entirely inside the HIP tooling universe: compile Loom to a target
artifact/code object, compile HIP C++ to HSACO, load both with the HIP module API
in one runner, launch equivalent geometry and fixtures, and measure both with
`rocprofv3 --kernel-trace`. Loom timestamps should not be cross-compared
directly with HIP event or rocprof timings.

## Baseline Commands And Artifacts

The common runner now has an explicit `loom64` case:

```bash
./tools/hrx2_q8_0_f32_module_runner.py \
  --run-id q8-wg64-parity-20260612-smoke \
  --cases=loom64,hip64 \
  --iters 10 \
  --warmup 2 \
  --repeats 2
```

This smoke passed correctness for both full kernels.

The profiled baseline used the generated runner:

```text
cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-20260612-smoke/q8_common_module_runner
```

and ran three independent non-concurrent `rocprofv3 --kernel-trace` captures
for each candidate, with `--no-check` in the profiled region after correctness
had already passed.

DBs:

- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-baseline-20260612/rocprof_loom64_run1/loom64_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-baseline-20260612/rocprof_loom64_run2/loom64_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-baseline-20260612/rocprof_loom64_run3/loom64_run3_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-baseline-20260612/rocprof_hip64_run1/hip64_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-baseline-20260612/rocprof_hip64_run2/hip64_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-baseline-20260612/rocprof_hip64_run3/hip64_run3_results.db`

Per-run stats:

| DB | Symbol | p50 | p90 | Mean |
| --- | --- | ---: | ---: | ---: |
| `hip64_run1_results.db` | `refute_q8_0_rows1_wg64_f32` | 1.96 us | 2.56 us | 2.262 us |
| `hip64_run2_results.db` | `refute_q8_0_rows1_wg64_f32` | 1.96 us | 2.68 us | 2.362 us |
| `hip64_run3_results.db` | `refute_q8_0_rows1_wg64_f32` | 1.96 us | 2.561 us | 2.336 us |
| `loom64_run1_results.db` | `q8_0_f32_candidate` | 2.12 us | 2.84 us | 2.465 us |
| `loom64_run2_results.db` | `q8_0_f32_candidate` | 2.12 us | 2.56 us | 2.424 us |
| `loom64_run3_results.db` | `q8_0_f32_candidate` | 2.12 us | 2.56 us | 2.392 us |

## Larger-Shape Evidence

Exact-shape Loom artifact generation:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id q8-wg64-parity-large-k4096-r128-c8-20260612 \
  --shapes 4096x128x8 \
  --workgroup-sizes 64 \
  --rows-per-workgroup 1 \
  --cols-per-workgroup 1 \
  --algorithms word4_bitunpack_rhsvec_dotf \
  --iterations 1 \
  --warmup-iterations 1 \
  --repetitions 1 \
  --timeout 45
```

The artifact compiled and passed Loom correctness:

```text
cache/hrx2/q8_0_f32_tune/q8-wg64-parity-large-k4096-r128-c8-20260612/bundles/q8_0_f32_word4_bitunpack_rhsvec_dotf_k4096_r128_c8_rpg1_cpg1_wg64_rep0/target_artifacts/r0005325a8eee7dec_c0_per_sample_sample0_target.elf
```

The common-runner correctness smokes also passed for both Loom and HIP at
`k=4096`, `rows=128`, `cols=8`.

Profile DBs:

- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-large-k4096-r128-c8-20260612/rocprof_loom64_run1/loom64_large_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-large-k4096-r128-c8-20260612/rocprof_loom64_run2/loom64_large_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-large-k4096-r128-c8-20260612/rocprof_loom64_run3/loom64_large_run3_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-large-k4096-r128-c8-20260612/rocprof_hip64_run1/hip64_large_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-large-k4096-r128-c8-20260612/rocprof_hip64_run2/hip64_large_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-large-k4096-r128-c8-20260612/rocprof_hip64_run3/hip64_large_run3_results.db`

Per-run stats:

| DB | Symbol | p50 | p90 | Mean |
| --- | --- | ---: | ---: | ---: |
| `hip64_large_run1_results.db` | `refute_q8_0_rows1_wg64_f32` | 4.16 us | 4.84 us | 4.299 us |
| `hip64_large_run2_results.db` | `refute_q8_0_rows1_wg64_f32` | 4.12 us | 4.80 us | 4.267 us |
| `hip64_large_run3_results.db` | `refute_q8_0_rows1_wg64_f32` | 4.12 us | 4.84 us | 4.285 us |
| `loom64_large_run1_results.db` | `q8_0_f32_candidate` | 7.881 us | 8.48 us | 7.973 us |
| `loom64_large_run2_results.db` | `q8_0_f32_candidate` | 7.880 us | 8.52 us | 7.959 us |
| `loom64_large_run3_results.db` | `q8_0_f32_candidate` | 7.880 us | 8.52 us | 7.960 us |

Additional larger-shape hot-loop device-time check at `8192x128x8`:

| DB | Symbol | p50 | p90 | Mean |
| --- | --- | ---: | ---: | ---: |
| `loom8192_results.db` | `q8_0_f32_candidate` | 14.080 us | 14.760 us | 14.178 us |
| `hip8192_results.db` | `refute_q8_0_rows1_wg64_f32` | 6.480 us | 7.161 us | 6.641 us |

DBs:

- `cache/hrx2/q8_0_f32_common_runner/q8-shape8192-device-20260612/rocprof_loom64/loom8192_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-shape8192-device-20260612/rocprof_hip64/hip8192_results.db`

## Shape-Targeted Runner

The common runner wrapper now exposes the C++ runner's shape arguments:

```bash
./tools/hrx2_q8_0_f32_module_runner.py \
  --run-id q8-shape-targeting-wrapper-smoke-20260612 \
  --cases=loom64,hip64 \
  --loom-module-wg64 cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k4096-r128-c8-20260612/bundles/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k4096_r128_c8_rpg1_cpg1_wg64_rep0/target_artifacts/r00053299d3de2023_c0_per_sample_sample0_target.elf \
  --k 4096 \
  --rows 128 \
  --cols 8 \
  --iters 10 \
  --warmup 2 \
  --repeats 2
```

The wrapper smoke passed correctness for both Loom and HIP and recorded the
actual shape in the emitted command/result rows.

## Focused DoD Result: Exact-Shape Unrolled WG64

Tool/source change:

- Added `word4_bitunpack_unrolled_rhsvec_dotf` to
  `tools/hrx2_q8_0_f32_tune.py`.
- Preserved the winning source at
  `docs/loom/q8-0-f32-kernels/loom_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_wg64.loom`.

Compile command:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id q8-wg64-parity-unrolled-k512-r64-c8-20260612 \
  --shapes 512x64x8 \
  --workgroup-sizes 64 \
  --rows-per-workgroup 1 \
  --cols-per-workgroup 1 \
  --algorithms word4_bitunpack_unrolled_rhsvec_dotf \
  --iterations 1 \
  --warmup-iterations 1 \
  --repetitions 1 \
  --timeout 60
```

Correctness:

- Loom `check.case` passed during the tuning command.
- Common-runner full-kernel correctness passed for both `loom64` and `hip64`:

```bash
./tools/hrx2_q8_0_f32_module_runner.py \
  --run-id q8-wg64-unrolled-k512-smoke-20260612 \
  --cases=loom64,hip64 \
  --loom-module-wg64 cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/target_artifacts/r00053313161878c6_c0_per_sample_sample0_target.elf \
  --k 512 \
  --rows 64 \
  --cols 8 \
  --iters 20 \
  --warmup 5 \
  --repeats 2
```

Compile report:

- 150 instructions
- 748 code bytes
- 0 private memory bytes
- 0 allocation spills
- 8 bytes LDS
- peak live units 19
- expected packed forms present:
  `global_load_b32`, `global_load_d16_b16`, `global_load_b128`,
  `v_bfe_i32`, `v_fma_mix_f32`, `v_fmac_f32`

Artifacts:

- Source:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k512-r64-c8-20260612/variants/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64.loom`
- Durable source copy:
  `docs/loom/q8-0-f32-kernels/loom_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_wg64.loom`
- Target artifact:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/target_artifacts/r00053313161878c6_c0_per_sample_sample0_target.elf`
- Target listing:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/target_listings/r00053313161878c6_c0_per_sample_sample0_target_listing.amdgpu-assembly`
- Compile report:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/compile_reports/r00053313161878c6_c0_per_sample_sample0_compile_report.json`

Same-runner `rocprofv3` DBs:

- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-unrolled-k512-dod-20260612/rocprof_loom64_run1/loom_unrolled_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-unrolled-k512-dod-20260612/rocprof_loom64_run2/loom_unrolled_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-unrolled-k512-dod-20260612/rocprof_loom64_run3/loom_unrolled_run3_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-unrolled-k512-dod-20260612/rocprof_hip64_run1/hip_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-unrolled-k512-dod-20260612/rocprof_hip64_run2/hip_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-unrolled-k512-dod-20260612/rocprof_hip64_run3/hip_run3_results.db`

Per-run device-time stats from `rocpd_kernel_dispatch`:

| Run | Variant | Symbol | p50 | p90 | Mean | Min | Max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Loom unrolled WG64 | `q8_0_f32_candidate` | 1.960 us | 2.560 us | 2.288 us | 1.440 us | 12.440 us |
| 1 | HIP WG64 | `refute_q8_0_rows1_wg64_f32` | 1.960 us | 2.640 us | 2.362 us | 1.440 us | 12.360 us |
| 2 | Loom unrolled WG64 | `q8_0_f32_candidate` | 1.960 us | 2.560 us | 2.296 us | 1.600 us | 12.080 us |
| 2 | HIP WG64 | `refute_q8_0_rows1_wg64_f32` | 1.960 us | 2.560 us | 2.358 us | 1.440 us | 12.440 us |
| 3 | Loom unrolled WG64 | `q8_0_f32_candidate` | 1.840 us | 1.960 us | 1.868 us | 1.360 us | 3.960 us |
| 3 | HIP WG64 | `refute_q8_0_rows1_wg64_f32` | 1.960 us | 2.560 us | 2.224 us | 1.440 us | 12.480 us |

Launch/resource metadata from rocprof run 1:

| Variant | Workgroup | `grid_size_x` | `grid_size_y` | Dispatches | SGPR count | Arch VGPR count | LDS | Private |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Loom unrolled WG64 | 64 | 4096 | 8 | 1100 | 128 | 24 | 8 B | 0 B |
| HIP WG64 | 64 | 4096 | 8 | 1100 | 128 | 24 | 8 B | 0 B |

Decision: accepted for the focused DoD shape. The unrolled Loom candidate is
at parity or better in all three independent same-runner `rocprofv3` runs.
This does not mean the same exact unroll is the right general Q8 strategy: at
`4096x128x8`, exact-shape unrolling improved baseline Loom but remained behind
HIP, so unrolling should be a shape/config axis rather than a universal rule.

## Preferred Spelling: `scf.for unroll(%factor)`

The manual unrolled source was replaced with a cleaner high-level spelling that
uses the Loom SCF unroll annotation:

```text
config.def @hrx2.tuning.q8_0_f32.unroll_factor = 2 : index
...
%q8_unroll_factor = config.get @hrx2.tuning.q8_0_f32.unroll_factor : index
%trip_count_0 = index.constant 2 : index
%sum_0 = scf.for %unroll_iter_0 = [%zero_index to %trip_count_0 step %one](
    %acc_0 = %zero_f32_0 : f32) -> (f32) unroll(%q8_unroll_factor) {
  %block_iter_step_0 = index.mul %unroll_iter_0, %block_step : index
  %block_idx_raw_0 = index.add %block_slot, %block_iter_step_0 : index
  ...
  scf.yield %next_0 : f32
}
```

Important authoring detail: the natural Q8 loop lower bound is `%block_slot`,
which is lane-dependent. The current unroll pass needs exact static loop
bounds, so the accepted spelling loops over a static ordinal range and computes
the dynamic `block_idx` inside the loop body. With `unroll_factor=2`, the target
listing contains no residual loop.

The configuration knob is real in the compile path. A direct `loom-compile`
run with `--config=hrx2.tuning.q8_0_f32.unroll_factor=0` succeeded and produced
a looped artifact with residual branches:

```bash
build/hrx-install/bin/loom-compile \
  docs/loom/q8-0-f32-kernels/loom_word4_bitunpack_scfunroll_rhsvec_dotf_k512_r64_c8_rpg1_wg64.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --config=hrx2.tuning.q8_0_f32.unroll_factor=0 \
  --emit-target-artifact=cache/hrx2/q8_0_f32_tune/q8-wg64-scfunroll-config0-20260612/q8_scf_unroll0.hsaco \
  --compile-report=json-summary \
  --compile-report-output=cache/hrx2/q8_0_f32_tune/q8-wg64-scfunroll-config0-20260612/compile_report_unroll0.json \
  --output=cache/hrx2/q8_0_f32_tune/q8-wg64-scfunroll-config0-20260612/q8_scf_unroll0.vmfb
```

That report has 119 instructions, 580 code bytes, peak live units 14, no
spills, and the disassembly contains the expected loop control. The current
`iree-benchmark-loom` help does not advertise direct `--config` bindings, so
automated config sweeps in the benchmark tool still need either wrapper support
or a generated-source/compile step per config.

Tool/source change:

- Added `word4_bitunpack_scfunroll_rhsvec_dotf` to
  `tools/hrx2_q8_0_f32_tune.py`.
- Preserved the preferred source at
  `docs/loom/q8-0-f32-kernels/loom_word4_bitunpack_scfunroll_rhsvec_dotf_k512_r64_c8_rpg1_wg64.loom`.

## Production HRX2 Route

The production route now uses the same algorithm family but keeps the tuner
shape as a bucketed route rather than an exact one-off. The accepted source root
is target-neutral:

```text
kernel.def export("hrx2_mul_mat_q8_0_f32_static_packed_scf_unroll")
  @hrx2_mul_mat_q8_0_f32_static_packed_scf_unroll()
```

The catalog route selects it for gfx1100 evidence with:

```text
route: mul_mat_q8_0_f32_packed_scfunroll_k256_8192_c1_16_wg64
shape domain: k=256..8192, rows=1..8192, cols=1..16
guard: k_multiple_of=256
workgroup: 64x1x1
config:
  @hrx2.shape.k = shape.k
  @hrx2.shape.rows = shape.rows
  @hrx2.shape.cols = shape.cols
  @hrx2.tuning.workgroup_size = 64
  @hrx2.tuning.q8_0_f32.unroll_factor = shape.q8_full_unroll_factor
```

That `shape.q8_full_unroll_factor` is derived as:

```text
(k / 32) / (workgroup_size / 8)
```

The guard keeps this route on shapes where the current full-unroll strategy is
well-defined. Lower-priority looped packed routes cover the wider shape space
without forcing full unroll.

Focused production validation on 2026-06-12:

```bash
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops -j"$(nproc)"

OUT=cache/hrx2/q8-prod-20260612-163416
LD_LIBRARY_PATH="$PWD/build/llama-hrx2/bin:$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}" \
GGML_HRX2_TRACE_JSONL="$OUT/events.jsonl" \
GGML_HRX2_TRACE_ROUTES=1 \
GGML_HRX2_EVIDENCE_DIR="$OUT/evidence" \
build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32,m=16,n=[1-9],k=256' --output csv
```

Result: pass. The trace selected
`mul_mat_q8_0_f32_packed_scfunroll_k256_8192_c1_16_wg64` for columns 1-9 and
captured detailed compile reports of 2398-2401 bytes with allocation, schedule,
pressure, memory, and static instruction-mix sections. This proves the
production JIT/link/export/config/evidence path for the bucket. It does not by
itself prove large-shape parity; the larger-shape HIP refutation results above
remain the acceptance bar for future Q8 schedule work.

Compile command:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id q8-wg64-scfunroll-k512-r64-c8-20260612 \
  --shapes 512x64x8 \
  --workgroup-sizes 64 \
  --rows-per-workgroup 1 \
  --cols-per-workgroup 1 \
  --algorithms word4_bitunpack_scfunroll_rhsvec_dotf \
  --iterations 1 \
  --warmup-iterations 1 \
  --repetitions 1 \
  --timeout 60
```

Compile report:

- 145 instructions
- 748 code bytes
- 0 private memory bytes
- 0 allocation spills
- 8 bytes LDS
- peak live units 23
- expected packed forms present:
  `global_load_b32`, `global_load_d16_b16`, `global_load_b128`,
  `v_bfe_i32`, `v_fma_mix_f32`, `v_fmac_f32`

The instruction counts for key operations matched the manual-unrolled source,
but the listing is not byte-identical and register scheduling differs. The
manual source reported peak live units 19; the SCF-unrolled source reports 23.
This did not affect the focused device-time result.

Artifacts:

- Source:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-scfunroll-k512-r64-c8-20260612/variants/q8_0_f32_word4_bitunpack_scfunroll_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64.loom`
- Durable source copy:
  `docs/loom/q8-0-f32-kernels/loom_word4_bitunpack_scfunroll_rhsvec_dotf_k512_r64_c8_rpg1_wg64.loom`
- Target artifact:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-scfunroll-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_scfunroll_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/target_artifacts/r0005343cd97b39b9_c0_per_sample_sample0_target.elf`
- Target listing:
  `cache/hrx2/q8_0_f32_tune/q8-wg64-scfunroll-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_scfunroll_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/target_listings/r0005343cd97b39b9_c0_per_sample_sample0_target_listing.amdgpu-assembly`

Same-runner `rocprofv3` DBs:

- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-scfunroll-k512-dod-20260612/rocprof_loom64_run1/loom64_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-scfunroll-k512-dod-20260612/rocprof_loom64_run2/loom64_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-scfunroll-k512-dod-20260612/rocprof_loom64_run3/loom64_run3_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-scfunroll-k512-dod-20260612/rocprof_hip64_run1/hip64_run1_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-scfunroll-k512-dod-20260612/rocprof_hip64_run2/hip64_run2_results.db`
- `cache/hrx2/q8_0_f32_common_runner/q8-wg64-scfunroll-k512-dod-20260612/rocprof_hip64_run3/hip64_run3_results.db`

Per-run device-time stats from `rocpd_kernel_dispatch`:

| Run | Variant | Symbol | p50 | p90 | Mean | Min | Max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Loom SCF-unrolled WG64 | `q8_0_f32_candidate` | 1.960 us | 2.640 us | 2.357 us | 1.440 us | 12.400 us |
| 1 | HIP WG64 | `refute_q8_0_rows1_wg64_f32` | 1.960 us | 2.520 us | 2.262 us | 1.440 us | 12.440 us |
| 2 | Loom SCF-unrolled WG64 | `q8_0_f32_candidate` | 1.960 us | 2.560 us | 2.246 us | 1.400 us | 12.081 us |
| 2 | HIP WG64 | `refute_q8_0_rows1_wg64_f32` | 1.960 us | 2.640 us | 2.330 us | 1.440 us | 12.281 us |
| 3 | Loom SCF-unrolled WG64 | `q8_0_f32_candidate` | 1.960 us | 2.560 us | 2.233 us | 1.520 us | 12.361 us |
| 3 | HIP WG64 | `refute_q8_0_rows1_wg64_f32` | 1.960 us | 2.560 us | 2.277 us | 1.440 us | 12.241 us |

Decision: accepted as the preferred source spelling for the focused DoD shape.
It keeps loop form explicit, allows the unroll amount to be represented as a
configuration value, and matches HIP p50 in three independent same-runner
`rocprofv3` captures.

## Listing Delta

The accepted Loom WG64 listing is:

```text
cache/hrx2/q8_0_f32_tune/q8-wg64-parity-unrolled-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_unrolled_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep0/target_listings/r00053313161878c6_c0_per_sample_sample0_target_listing.amdgpu-assembly
```

The previous looped Loom WG64 listing is:

```text
cache/hrx2/q8_0_f32_tune/q8-stella-focused-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep2/target_listings/r00052f84794d3ae8_c0_per_sample_sample0_target_listing.amdgpu-assembly
```

The most useful HIP parity template is now the static same-ABI control:

```text
cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-static-hip-20260612/q8_static_candidate.s
```

Its exported function is also `q8_0_f32_candidate(src0, src1, dst)` and it is
launched through the runner's Loom-style three-argument path.

Material deltas:

| Area | Accepted Loom unrolled WG64 | Static HIP same-ABI control | Interpretation |
| --- | --- | --- | --- |
| Loads | Two unrolled `global_load_b128`, `global_load_b32`, and `global_load_d16_b16` groups | One load group in the loop body | Loom specializes the two loop iterations for this exact shape. |
| Load scheduling | Two load groups are issued before the single `s_waitcnt vmcnt(0)` | `s_clause`, staged waits, and explicit delay scheduling | Loom still lacks HIP-style wait scheduling, but unrolling increases available independent memory work for this shape. |
| Delay scheduling | No explicit `s_clause` or `s_delay_alu` | Uses `s_clause` and many `s_delay_alu` packets | Remaining listing difference, but no longer a performance blocker on this tiny shape. |
| Byte extraction | `v_bfe_i32` plus `v_ashrrev_i32` for high byte, twice | `v_ashrrev_i16`, `v_bfe_i32`, `v_ashrrev_i32` in loop | Different spelling, same exact semantics. |
| Accumulate form | Eight `v_fma_mix_f32`, one `v_fma_f32`, seven `v_fmac_f32` | Four `v_fma_mix_f32`, four `v_fmac_f32` per loop iteration | Accepted Loom form carries two Q blocks through one straight-line body. |
| Reduction | DPP within wave, two barriers, two LDS writes/reads, two `ds_bpermute` | HIP shuffle-style `ds_bpermute` tree, one barrier | Still different, but hidden by the improved straight-line body on the focused shape. |
| Resources | Compile report: 19 peak live units, 8 B LDS, 0 private/spills. Rocprof metadata: 24 arch VGPR, 128 SGPR, 8 B LDS | Rocprof metadata: 24 arch VGPR, 128 SGPR, 8 B LDS | Same profiled resource metadata; occupancy/resource shape is not the limiter. |

## Experiments

### Source Order: Scale Before RHS

Tool change:

- Added `word4_bitunpack_scalefirst_rhsvec_dotf` to
  `tools/hrx2_q8_0_f32_tune.py`.

Command:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id q8-wg64-parity-scalefirst-20260612 \
  --shapes 512x64x8 \
  --workgroup-sizes 64 \
  --rows-per-workgroup 1 \
  --cols-per-workgroup 1 \
  --algorithms word4_bitunpack_scalefirst_rhsvec_dotf \
  --iterations 1 \
  --warmup-iterations 1 \
  --repetitions 1 \
  --timeout 30
```

Result:

- Compiled and passed Loom correctness.
- Compile report was unchanged: 122 instructions, 600 code bytes, no spills, 8
  bytes LDS, peak live units 16.
- Target listing was identical to the baseline WG64 Loom function.

Decision: rejected as a no-op. Loom canonicalized/scheduled this source order
back to the same emitted code.

### Manual Packed-Word Unpack

Existing algorithm:

- `word4_rhsvec_dotf`

Command:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id q8-wg64-parity-manualunpack-20260612 \
  --shapes 512x64x8 \
  --workgroup-sizes 64 \
  --rows-per-workgroup 1 \
  --cols-per-workgroup 1 \
  --algorithms word4_rhsvec_dotf \
  --iterations 1 \
  --warmup-iterations 1 \
  --repetitions 1 \
  --timeout 30
```

Same-runner profile:

```text
cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-exp-20260612/rocprof_manualunpack_wg64/manualunpack_wg64_results.db
```

Result:

| Variant | Correctness | p50 | Mean | Static notes |
| --- | --- | ---: | ---: | --- |
| Loom manual unpack WG64 | passed | 2.16 us | 2.357 us | 126 instructions, 600 code bytes, no `v_fma_mix_f32` |

Decision: rejected. Manual byte extraction is worse than `vector.bitunpacks<8>`
and does not close the gap.

### Exact-Shape Unrolled Loop

Tool change:

- Added `word4_bitunpack_unrolled_rhsvec_dotf` to
  `tools/hrx2_q8_0_f32_tune.py`.

Focused shape command and result are recorded above in
`Focused DoD Result: Exact-Shape Unrolled WG64`. The same source axis was also
tested on a larger representative shape.

Large-shape command:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id q8-wg64-parity-unrolled-k4096-r128-c8-20260612 \
  --shapes 4096x128x8 \
  --workgroup-sizes 64 \
  --rows-per-workgroup 1 \
  --cols-per-workgroup 1 \
  --algorithms word4_bitunpack_unrolled_rhsvec_dotf \
  --iterations 1 \
  --warmup-iterations 1 \
  --repetitions 1 \
  --timeout 90
```

Compile report:

- 613 instructions
- 3640 code bytes
- 0 private memory bytes
- 0 allocation spills
- 8 bytes LDS
- peak live units 68
- expected load/unpack/MACC forms present

Same-runner profile DB:

```text
cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-unrolled-k4096-r128-c8-20260612/rocprof_unrolled_large/unrolled_large_results.db
```

Result:

| Variant | Correctness | p50 | p90 | Mean | Launch |
| --- | --- | ---: | ---: | ---: | --- |
| Loom unrolled WG64, `4096x128x8` | passed | 6.120 us | 6.760 us | 6.251 us | WG64, `grid_x=128`, `grid_y=8`, 300 calls |

Decision: accepted for the focused goal shape, but keep the large-shape result
as a limit on generalization. Exact-shape unrolling reduced the large-shape gap
versus baseline Loom (`~7.88 us` p50 to `6.12 us` p50), but HIP remains around
`4.12-4.16 us` p50 on the same representative shape. Treat unrolling as a
shape/config axis rather than a universal Q8 algorithm.

### Static HIP Same-ABI Control

Scratch source:

```text
cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-static-hip-20260612/q8_static_candidate.hip.cpp
```

This adds:

```c++
extern "C" __global__ void q8_0_f32_candidate(
    const block_q8_0 * src0, const float * src1, float * dst) {
  q8_0_rows_impl<64, 1>(src0, src1, dst, 512, 64, 8);
}
```

and launches it through the runner's `kind=loom` path with the same three
kernel arguments as the Loom artifact.

Profile DB:

```text
cache/hrx2/q8_0_f32_common_runner/q8-wg64-parity-static-hip-20260612/rocprof_static_candidate/static_candidate_results.db
```

Result:

| Variant | Correctness | p50 | Mean | Launch |
| --- | --- | ---: | ---: | --- |
| Static HIP `q8_0_f32_candidate` | passed | 1.96 us | 2.256 us | Same three-argument ABI as Loom |

Decision: this proves the runner ABI, exact-shape specialization, and exported
symbol shape are not the reason Loom is slower. A HIP/assembly schedule with the
same ABI reaches HIP parity.

## Conclusion

The focused WG64 goal shape is closed: the exact-shape unrolled Loom kernel is
at HIP parity or better in three independent same-runner `rocprofv3` captures.
The preferred durable source now expresses that result with
`scf.for ... unroll(%factor)` rather than manual duplication.

What changed: the source stopped asking Loom to compile a two-iteration loop for
this tiny shape and instead presented the two Q8 block iterations as one
straight-line body. That gave the lowering more independent memory and
arithmetic work before the reduction without changing semantics.

What did not change: high-level Loom still does not expose HIP-style staged
wait scheduling, `s_clause`/`s_delay_alu` placement, or the exact one-barrier
HIP reduction schedule. Those remain relevant for larger Q8 shapes, where the
same unrolled source axis helps but does not reach HIP parity.

Next concrete engineering action for the broader Q8 family:

1. Keep `loop_form=looped|exact_unrolled` as an explicit tuning axis keyed by
   shape and target.
2. For large K, look for a schedule that increases independent work without
   explosive code size, such as partial unroll factors or software-pipelined
   block groups.
3. If high-level source cannot request the staged wait/reduction schedule, use
   low/asm interop when available and use the HIP same-ABI assembly as the
   parity template.
