# Q8_0/F32 MUL_MAT Performance Refutation: gfx1100

Date: 2026-06-12

Correction: the final performance conclusion in this note is superseded by
`docs/loom/q8-0-f32-delta-root-cause-gfx1100.md`. The WYSIWYG source lessons
and HIP reference artifacts remain useful, but a later common HIP module runner
showed the best Loom WG128 target artifact at p50 2.04 us versus HIP WG128 p50
2.00 us under the same launch/profiling path. The large Loom-vs-HIP delta below
was primarily a measurement/runner artifact.

## Claim Tested

The current HRX2 Loom Q8_0/F32 standalone scalar route was tested against
independent exact-semantics native HIP baselines and then rechecked with true
device timing. The question was whether the Loom route was close enough to keep
optimizing within the same source shape, or whether there is clear headroom
that refutes the current implementation.

This refutation only compares exact F32 RHS kernels. Q8_1 packed RHS, prompt
MMQ, and `MUL_MAT + ADD` fusion remain separate algorithm/fusion families.

## Method

New reusable skill:

- `skills/hrx2-performance-refutation/SKILL.md`

New harness:

- `tools/hrx2_q8_0_f32_refute.py`

The harness generates a standalone HIP C++ benchmark under `cache/`, compiles it
with workspace ROCm, and runs deterministic non-constant Q8_0/F32 fixtures.
The tested exact baseline is a rows-per-workgroup schedule derived from old HRX
and the Metal/OpenCL row-grouping pattern:

- one, two, or four Q8 rows owned by a workgroup;
- workgroup sizes 32, 64, 128, and 256;
- each active lane processes a `float4` RHS chunk and four Q8 quants per block;
- wave/workgroup reductions write the same F32 output layout as the Loom route.

Command:

```bash
./tools/hrx2_q8_0_f32_refute.py \
  --run-id gfx1100-q8-0-f32-refute-exact-20260612 \
  --iters 1000 \
  --warmup 100 \
  --repeats 5
```

Raw artifacts:

- `cache/hrx2/q8_0_f32_refute/gfx1100-q8-0-f32-refute-exact-20260612/manifest.json`
- `cache/hrx2/q8_0_f32_refute/gfx1100-q8-0-f32-refute-exact-20260612/results.jsonl`
- `cache/hrx2/q8_0_f32_refute/gfx1100-q8-0-f32-refute-exact-20260612/summary.md`

The initial HIP numbers in that summary are HIP event timings. They are useful
for comparing HIP candidates to each other, but they are not equivalent to
Loom's `dispatch_complete` host/dispatch timing. For the decisive comparison,
use:

- HIP device time from `rocprofv3 --kernel-trace` on the standalone HIP
  executable;
- Loom device time from `iree-benchmark-loom --profile-final-batch=true
  --profile-data=dispatch-events,executable-metadata`.

`rocprofv3` does not currently work under HRX/Loom in this workspace, so Loom's
own dispatch-event profile is the source of truth for HRX/Loom device timing.

## Results

Host/dispatch timing for the focused `k512_r64_c8`, rows1/WG64 route:

| Variant | Timing source | Median/p50 |
| --- | --- | ---: |
| Loom scalar | `dispatch_complete` | 22.05 us |
| Loom block4 | `dispatch_complete` | 21.71 us |
| HIP rows1/WG64 | HIP events | 4.54 us |

Those host/event numbers are not apples-to-apples. True device timing for the
same shape is:

| Variant | Timing source | Device median |
| --- | --- | ---: |
| Loom scalar | Loom dispatch-event profile, 5 repeats | 4.44 us |
| Loom block4 scalar-load schedule | Loom dispatch-event profile, 5 repeats | 3.44 us |
| Loom block4 + RHS vector + dotf | Loom dispatch-event profile, 5 repeats | 3.36 us |
| Loom packed Q word + RHS vector + dotf | Loom dispatch-event profile, 5 repeats | 3.44 us |
| Loom packed Q word + bitunpack + RHS vector + dotf, WG128 | Loom dispatch-event profile, 5 repeats | 3.24-3.32 us |
| Loom packed Q word + bitunpack + subgroup reduce, WG32 | Loom dispatch-event profile, 5 repeats | 3.44 us |
| HIP rows1/WG64 | `rocprofv3 --kernel-trace`, 1100 calls | p50 1.88 us, mean 1.91 us |
| HIP rows1/WG128 | `rocprofv3 --kernel-trace`, 1100 calls | p50 2.00 us, mean 2.40 us |

The earlier 4-5x claim was mostly a measurement-stack artifact. The refuted
scalar Loom source is about 2.36x slower than the native HIP device-time
baseline for this focused shape. Rewriting the Loom source to express the Q8
block schedule directly improved the device time by about 22.5%. The current
best high-level bitunpack route is still roughly 1.6-1.8x slower than the
native HIP kernel, depending on whether the HIP WG64 p50 or WG128 mean is used
as the reference.

## Interpretation

The current Loom Q8_0/F32 scalar route is refuted, but the corrected reason is
more precise than the initial report. Loom is WYSIWYG: the scalar source said
"one quant per lane iteration", so the emitted code performed one Q byte load,
one RHS scalar load, and one scale load per quant. The compiler was not expected
to infer the HIP row/block schedule.

The better Loom source shape is:

- map each lane to a four-quant chunk inside a Q8_0 block;
- loop over Q8 blocks with `block_idx = lane / 8` and `step = WG / 8`;
- load/amortize one f16 block scale per four Q values;
- load four RHS floats for the same chunk;
- accumulate four products before the workgroup reduction.

That `block4` source compiles and is faster. A fully packed `chunk4` source that
uses `vector.load -> vector<4xi8>` for the Q payload and `vector.load ->
vector<4xf32>` for RHS is blocked today by AMDGPU target-lowering coverage:

- `vector.sitofp vector<4xi8> -> vector<4xf32>` is rejected by the AMDGPU target
  contract;
- replacing it with scalar extracts then fails because the AMDGPU target has no
  `vector.extract` lowering contract in this path.

The direct packed-word workaround proved that #1 and #2 are expressible without
waiting on `vector<4xi8>` lowering:

- `block4_rhsvec` emits `global_load_b128` for RHS and four scalar Q byte loads;
- `word4_rhsvec` emits `global_load_b128` for RHS and `global_load_b32` for the
  four Q bytes;
- `*_dotf` variants emit `v_fma_f32`/`v_fmac_f32` for RHS accumulation;
- `word4_bitunpack_rhsvec_dotf` uses `vector.bitunpacks<8>` on the packed Q
  word and is the fastest profiled Loom variant so far.

However, the packed Q word path still does not close the gap. The updated Loom
branch now emits `global_load_d16_b16`, `v_bfe_i32`, `v_fma_mix_f32`, and
`v_fmac_f32` from the high-level `vector.bitunpacks<8>` form, so the old
"missing mixed scale multiply" diagnosis is obsolete. The remaining difference
is schedule parity: HIP and Loom still choose different byte-extract details
and different reduction/control schedules, and the Loom dispatch-event timing
has sub-5us profiling ambiguity. A batch-64 Loom profile for the WG128 winner
reported benchmark p50 around 3.38 us, `dispatch_function` mean around 2.76 us,
and individual operation rows alternating around 1.7 and 3.4 us. That does not
prove parity with HIP, but it does mean future refutation should compare
listings and timing domains carefully before attributing every sub-microsecond
difference to source quality.

The low-code escape hatch is partly proven but not yet integrated into this
kernel family. A pure `low.kernel.def` AMDGPU smoke compiles and runs correctly
through the explicit low prep pipeline. The author indicated that today's
interoperability model is declaration/link based: high code uses `func.decl`,
target-low code provides `low.func.decl`/definitions, the high module is
lowered to low, and then the modules are linked. In this branch, AMDGPU
`source-to-low` does not appear to lower imported `func.decl` symbols yet
because the AMDGPU lowering policy has no nonzero `import_decl_kind`; a minimal
`func.decl import("rocasm", ...) target(@target)` smoke remained unchanged
after `loom-opt --pass=source-to-low`. Treat that as toolchain feedback rather
than a reason to abandon the low path.

## Decision

Status: refuted, with a concrete next schedule/measurement target.

Do not promote the current Loom Q8_0/F32 implementation as done-done. Keep
`scalar`, `block4`, `block4_rhsvec`, `word4_rhsvec`, `word4_bitunpack`, dotf,
and reduction variants as explicit algorithm families in the tuning space. The
next Loom task is no longer "make packed loads happen" or "make `v_fma_mix`
happen"; both are now demonstrated. The hard target is HIP-like full schedule
parity and a robust device-time comparison method.

After the exact HIP-equivalent Loom variant is working, run the normal tuning
flow again and only then evaluate packed Q8_1 RHS and `MUL_MAT + ADD` fusion
as separate route families.
