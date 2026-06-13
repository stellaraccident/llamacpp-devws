# Q8_0/F32 MUL_MAT: HIP Reference vs Loom on gfx1100

Date: 2026-06-12

Correction: the final performance conclusion in this note is superseded by
`docs/loom/q8-0-f32-delta-root-cause-gfx1100.md`. The HIP reference, Loom
source attempts, and assembly notes are still useful. The later common HIP
module runner showed the best Loom WG128 target artifact at p50 2.04 us versus
HIP WG128 p50 2.00 us under the same launch/profiling path, so the large
Loom-vs-HIP delta below was primarily a measurement/runner artifact.

## Scope

This note is specific to the exact-semantics Q8_0/F32 `MUL_MAT` kernel family:
Q8_0 packed rows on the left, F32 RHS, F32 output. It records the standalone HIP
reference, the Loom attempts to match that reference, the assembly/target-low
escape-hatch status, and the evidence that led us to the current diagnosis.

Focused shape for the deepest comparison:

```text
k = 512, rows = 64, cols = 8, rows_per_workgroup = 1
```

This is not the final shape universe. It is the smallest useful refutation
shape where we can compare device timing and target listings without llama.cpp
runtime effects dominating the result.

## Raw Evidence

Primary artifacts:

- Durable source copies:
  `docs/loom/q8-0-f32-kernels/`
- HIP refutation harness: `tools/hrx2_q8_0_f32_refute.py`
- Loom tuning harness: `tools/hrx2_q8_0_f32_tune.py`
- HIP run summary:
  `cache/hrx2/q8_0_f32_refute/gfx1100-q8-hip-rows1-wg-sweep-20260612/summary.md`
- HIP disassembly:
  `cache/hrx2/q8_0_f32_refute/gfx1100-q8-hip-rerun-20260612/hip_asm/q8_0_f32_refute.s`
- Loom bitunpack WG128 listing:
  `cache/hrx2/q8_0_f32_tune/q8-stella-focused-k512-r64-c8-20260612/bundles/q8_0_f32_word4_bitunpack_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg128_rep2/target_listings/r00052f870d700453_c0_per_sample_sample0_target_listing.amdgpu-assembly`
- Loom profile bundles:
  `cache/hrx2/q8_0_f32_tune/q8-stella-focused-k512-r64-c8-20260612/`
  `cache/hrx2/q8_0_f32_tune/q8-stella-subgroup-k512-r64-c8-20260612/`
  `cache/hrx2/q8_0_f32_tune/q8-stella-batch64-wg128-20260612/results.jsonl`

Important measurement rule: compare device time to device time. HIP event timing
and Loom `dispatch_complete` host timing were useful during exploration but were
not apples-to-apples. For sub-5us decisive comparisons, the current preferred
path is a common code-object runner that loads both Loom and HIP artifacts
through the same API and measures both with the same profiler. Loom
dispatch-event profiles remain useful diagnostics:

```bash
iree-benchmark-loom ... \
  --profile-final-batch=true \
  --profile-data=dispatch-events,executable-metadata \
  --artifact-bundle-policy=full
```

The common runner in `tools/hrx2_q8_0_f32_module_runner.py` can load the Loom
target ELF directly through the HIP module API, which avoids the HRX/Loom
profiling limitation for this comparison.

## Timing Summary

Focused `k512_r64_c8`, rows1 comparison:

| Variant | Timing source | Device time |
| --- | --- | ---: |
| Loom scalar source | Loom dispatch-event profile, 5 repeats | 4.44 us |
| Loom block4 scalar-load schedule | Loom dispatch-event profile, 5 repeats | 3.44 us |
| Loom block4 + RHS vector + dotf | Loom dispatch-event profile, 5 repeats | 3.36 us |
| Loom packed Q word + RHS vector + dotf | Loom dispatch-event profile, final batch | 3.36 us |
| Loom packed Q word + bitunpack + RHS vector + dotf, WG128 | Loom dispatch-event profile, 5 repeats | 3.24-3.32 us |
| Loom packed Q word + bitunpack + subgroup reduce, WG32 | Loom dispatch-event profile, 5 repeats | 3.44 us |
| HIP rows1/WG64 reference | `rocprofv3 --kernel-trace`, 1100 calls | p50 1.88 us, mean 1.91 us |
| HIP rows1/WG128 reference | `rocprofv3 --kernel-trace`, 1100 calls | p50 2.00 us, mean 2.40 us |

The table above is historical dispatch-event evidence. The later common-runner
comparison supersedes it for final route judgment: WG128 is essentially tied
with HIP under the same HIP module launch path, while WG64 has only a small
remaining gap. The current Loom branch is expressing the broad memory schedule
and the key mixed-scale arithmetic instructions; any remaining work is schedule
parity and measurement calibration, not a broad inability to express the kernel.

## HIP Reference Shape

The HIP reference is a straightforward exact-semantics row/workgroup kernel:

- one output row per workgroup for this focused shape;
- each lane owns four Q8 values within a Q8_0 block;
- each lane loads one packed Q word for four adjacent Q bytes;
- each lane loads one f16 scale for the Q8_0 block;
- each lane loads four F32 RHS values as a wide vector;
- each lane accumulates four products before the workgroup reduction.

The winning inner ISA shape from the HIP disassembly is:

```text
global_load_b32       packed Q payload for four Q bytes
global_load_d16_b16   f16 Q8_0 scale
global_load_b128      four F32 RHS values
v_ashrrev_i16 / v_bfe_i32 / v_ashrrev_i32
v_cvt_f32_i32
v_fma_mix_f32         f16 scale * converted Q, with zero addend
v_fmac_f32            RHS multiply-accumulate
```

Representative HIP listing excerpt:

```text
global_load_b32 v2, v[11:12], off
global_load_d16_b16 v15, v[3:4], off
global_load_b128 v[11:14], v[5:6], off
v_ashrrev_i16 v16.l, 8, v2.l
v_bfe_i32 v17, v2, 0, 8
v_bfe_i32 v18, v2, 16, 8
v_ashrrev_i32_e32 v2, 24, v2
v_cvt_f32_i32_e32 v17, v17
v_fma_mix_f32 v17, v15, v17, neg(0) op_sel_hi:[1,0,0]
v_fmac_f32_e32 v9, v11, v17
```

The important point is not that HIP is magic. It is explicitly doing the same
algorithm we want Loom to say: packed load, extract bytes, convert signed Q
values, multiply by the f16 block scale through a mixed instruction, then F32
accumulate with RHS.

## Loom Attempt History

### Scalar Baseline

The first Loom source was semantically correct but did not encode the desired
machine schedule. It effectively said "one logical quant per lane iteration",
so the listing performed scalar Q load, scalar RHS load, and scale reload at
that granularity. Device time was about 4.44 us for the focused shape.

Diagnosis: this was a WYSIWYG authoring failure, not a mysterious optimizer
failure. Loom did what the source said.

### `block4`

The next source mapped each lane to four Q8 values inside a Q8_0 block and
amortized one scale load across those four values. This matched the row/block
schedule conceptually and improved device time to about 3.44 us.

This was the first useful evidence that the right workflow is to express the
natural algorithm directly, then inspect target listings. It also showed that
we should not expect Loom to infer a four-value packed schedule from scalar
iteration.

### RHS Vector and Dot Forms

The `block4_rhsvec` and `block4_rhsvec_dotf` variants used `vector.load ->
vector<4xf32>` for the RHS and `vector.dotf` for the four-lane accumulation.
These variants emit the intended `global_load_b128` RHS load and FMA-style
accumulation. Best focused device time was about 3.36 us.

### Packed Q Word

The `word4_rhsvec*` variants used a byte-offset `view<1xi32>` over the Q
payload to force a packed 32-bit Q load. This successfully emitted
`global_load_b32` for the Q payload and `global_load_b128` for RHS.

This disproved the early suspicion that the primary gap was just missing packed
loads. Loom can express both key load widths today.

### `vector.bitunpacks<8>`

The `word4_bitunpack_rhsvec_dotf` variant used:

```text
%q_packed = vector.load ... -> vector<1xi32>
%q_i32 = vector.bitunpacks<8> %q_packed : vector<1xi32> -> vector<4xi32>
%q_f32 = vector.sitofp %q_i32 : vector<4xi32> to vector<4xf32>
```

This is the best high-level Loom spelling so far. After the 2026-06-12 Loom
update, it produces the broad load shape and the desired mixed scale multiply
shape:

```text
global_load_b32
global_load_d16_b16
global_load_b128
v_bfe_i32
v_fma_mix_f32
v_fmac_f32
```

Representative WG128 listing excerpt:

```text
global_load_b32 v2, v2, s[8:9] offset:0
global_load_d16_b16 v3, v5, s[8:9] offset:0
global_load_b128 v[8:11], v4, s[6:7] offset:0
v_bfe_i32 ...
v_cvt_f32_i32 ...
v_fma_mix_f32 ...
v_fmac_f32 ...
```

This fixes the previous main diagnosis. The load plan and inner mixed
unpack/scale/macc plan are now close. The remaining gap moved to the exact
schedule, reduction/control flow, and measurement parity.

### Subgroup Reduction Probe

The `word4_bitunpack_rhsvec_dotf_subgroup` variant replaces:

```text
kernel.workgroup.reduce<addf>
```

with:

```text
kernel.subgroup.reduce<addf>
```

and is constrained to WG32 so the lane-0 store observes the only subgroup's
complete result. It compiled and passed correctness, but did not improve the
focused shape:

```text
WG32 workgroup reduce: 3440 ns
WG32 subgroup reduce:  3440 ns
WG64 workgroup reduce: 3440 ns
WG128 workgroup reduce: 3320 ns
```

This means the existing high-level reduction selector is not the missing lever
for this shape. There is also no current source attribute to choose the
workgroup reduction schedule; Loom selects that internally from workgroup size,
wave size, tails, and available descriptors.

## Current Schedule Difference

The previous decisive difference was scale/unpack arithmetic. That is no
longer true on the updated branch. Current differences are:

| Concern | HIP reference | Best Loom source today |
| --- | --- | --- |
| Q payload load | `global_load_b32` | `global_load_b32` |
| RHS load | `global_load_b128` | `global_load_b128` |
| scale load | `global_load_d16_b16` | `global_load_d16_b16` |
| byte extraction | `v_bfe_i32`, `v_ashrrev_i16/i32` | `v_bfe_i32` plus high-byte shift |
| scale multiply | `v_fma_mix_f32` using f16 scale operand | `v_fma_mix_f32` using f16 scale operand |
| RHS accumulation | `v_fmac_f32` | `v_fmac_f32` |
| reduction | HIP shuffle/LDS schedule from HIP C++ lowering | Loom DPP plus LDS/barrier workgroup schedule |
| resource shape | 19 VGPR / 19 SGPR for rows1 WG64/WG128 | 14 VGPR / 14 SGPR for WG128 |

So the bottom line is:

1. Loom can now say the important load widths and mixed arithmetic.
2. The best Loom listing is still not identical to HIP, especially around
   byte extraction details and the multi-wave reduction/control schedule.
3. The remaining ~3.3 us Loom vs 1.88-2.00 us HIP gap is real enough to keep
   the route refuted, but the next hypothesis is schedule/reduction and
   profiling parity, not missing packed/mixed arithmetic.
4. Batch-64 Loom profiling shows timing ambiguity at this scale: benchmark
   dispatch p50 was about 3.38 us, the profiled `dispatch_function` mean was
   about 2.76 us, and individual operation rows alternated around 1.7 and
   3.4 us. This is not enough to declare parity, but it is enough to require
   caution before overfitting a sub-microsecond delta.

## ASM and Target-Low Interop Status

### Inline LLVM asm from high source

An attempted high-level escape hatch using `llvmir.inline_asm` parsed but failed
AMDGPU target lowering:

```text
ERR_TARGET_001: has no target-low contract for 'llvmir.inline_asm'
```

This path is not usable today for this kernel.

### Pure target-low kernel

A pure AMDGPU low kernel does compile and run if we skip source-to-low and use
the explicit low prep pipeline:

```bash
build/hrx-install/bin/iree-run-loom cache/hrx2/q8_lowasm_probe/min_low_kernel.loom \
  --backend=amdgpu --compile-root=@min_store \
  --pipeline=amdgpu-materialize-hal-kernel-abi,canonicalize,cse,low-select-operand-forms,low-dce,low-materialize-allocation \
  --workgroup-count=1,1,1 \
  --binding=1xf32=0 --expected-binding=1xf32=1 \
  --compile-report=summary \
  --emit-target-artifact=cache/hrx2/q8_lowasm_probe/min_low_kernel_run_lowprep.hsaco
```

That smoke wrote `1.0` to a single F32 binding and passed correctness. The
compile report showed a tiny zero-spill low kernel.

Limitation: `iree-benchmark-loom` did not benchmark the pure `low.kernel.def`
smoke because it does not currently derive static workgroup counts for low-only
kernels the way it does for `kernel.def`.

### High/low interop

The author clarified that the intended current bridge is declaration/linking
based:

- high module uses `func.decl`;
- low module provides `low.func.decl` or low definitions;
- lower the high module to low;
- link the lowered high module with the low module.

`low.invoke` is expected to be the friendlier spelling later.

Current branch status from inspection:

- `func.decl` and `low.func.decl` both exist in the IR;
- `source-to-low` has code to lower target-bound external declarations into
  `low.func.decl`;
- that path only runs when the target lowering policy has nonzero
  `import_decl_kind`;
- IREEVM sets `LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM`;
- AMDGPU's current `kAmdgpuLowLowerPolicy` does not set `import_decl_kind`.

A minimal AMDGPU smoke:

```text
amdgpu.target<gfx1100> @target

func.decl import("rocasm", "test.add_i32") target(@target)
  @test_add_i32(%lhs: i32, %rhs: i32) -> (i32)
```

remained a `func.decl` after:

```bash
build/hrx-install/bin/loom-opt ... --pass=source-to-low
```

That is consistent with the AMDGPU policy skipping imported declarations today.
It may be missing target plumbing, a different intended command sequence, or a
not-yet-landed piece. For this kernel, it means the low/rocasm escape hatch is
promising but not yet a turnkey authoring path.

### Descriptor/lowering status

The updated branch now emits `v_fma_mix_f32` from the high-level
`vector.bitunpacks<8>` plus f16-scale vector multiply path. That removes the
old request for a basic mixed-scale lowering. Remaining low/asm needs are more
specific:

- a way to author or select a known-good reduction schedule when
  `kernel.workgroup.reduce` is not the desired one;
- a reliable high/low or rocasm bridge for instruction-level experiments that
  are too target-specific for portable high source;
- compile-report schedule evidence fine-grained enough to explain why two
  listings with similar inner load/macc shapes differ by a material device-time
  margin.

## Current Conclusion

Loom is viable enough to express the broad algorithmic schedule for this kernel:
lane ownership, packed Q load, RHS vector load, scale amortization, bitunpack,
mixed f16-scale multiply, and dot-like accumulation. The gap is no longer "we
cannot say packed loads" or "we cannot emit `v_fma_mix_f32`".

The blocker for matching HIP on this specific kernel is now lower-level
schedule parity: exact byte-extract choices, reduction/control schedule, wait
placement, and possibly profiler/domain differences when comparing HRX/Loom
against standalone HIP. To close the gap, the next work should target exact
listing convergence and better device-time measurement, using low/rocasm only
if high-level source cannot select the needed schedule.

Until that is resolved, keep Q8_0/F32 marked as refuted, not done-done.
