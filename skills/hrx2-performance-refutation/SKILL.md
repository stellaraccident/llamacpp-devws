---
name: hrx2-performance-refutation
description: Use this when deciding whether an accepted HRX2 Loom kernel route is close enough to done or still has meaningful optimization headroom, by building independent exact-semantics native baselines, probing prior-art algorithm families, and documenting refutation evidence.
metadata:
  short-description: Refute HRX2 kernel performance claims
---

# HRX2 Performance Refutation

Use this skill after a Loom family has a measured candidate and before declaring
the op or fusion done.

## Required Context

Read first:

- the family report under `docs/loom/`;
- `docs/loom/backend-prior-art-algorithms.md`;
- the active HRX2 catalog route rows in `sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json`.

## Workflow

1. State the route claim to refute: op/fusion, target key, shape bucket, numeric
   policy, and current Loom p50/p90.
2. Build independent exact-semantics references first. Prefer native HIP C++
   transliterations of old HRX, CUDA/HIPified, Vulkan, OpenCL, or Metal
   algorithms with the same ABI, layout, fixtures, and output semantics.
3. Benchmark with deterministic non-constant fixtures and the same shape basket
   as the Loom sweep. The basket must include at least one representative
   large-enough shape whose median device time is comfortably above profiler
   tick/timer noise; keep tiny shapes as schedule microscopes, not acceptance
   gates. Keep warmup, repeats, raw samples, and command lines.
4. Keep timing domains comparable. For hill climbing among Loom candidates, use
   Loom's own benchmark/tuning tools and stay inside that self-consistent
   measurement universe. The HIP module API runner is only required when
   refuting against an existing native/static kernel: compile Loom to a target
   artifact/code object, compile the native reference to HSACO, load both with
   one HIP module API benchmark runner, launch both with equivalent geometry and
   fixtures, and measure both with `rocprofv3 --kernel-trace`. Treat this as the
   gold-standard emitted-code comparison against native references because it
   avoids crossing Loom/IREE benchmark timing with HIP/native timing. Keep Loom
   `--profile-final-batch=true
   --profile-data=dispatch-events,executable-metadata` as useful diagnostic and
   tuning-tool evidence, but do not use it directly against HIP event timing or
   rocprof timing without calibration. Record `dispatch_function`,
   `dispatch_command_operation`, benchmark score, profile clock uncertainty,
   launch metadata, and common runner DB paths separately.
   Treat common-runner hot loops as emitted-code parity tests, not final
   application realism: repeated dispatch over identical buffers can benefit
   from hot kernargs and cached data. For acceptance, also run the Loom/tool
   rotated-buffer path or an equivalent rotating-buffer runner, and label
   whether a number is device kernel timestamp, HIP event time, or tool
   operation interval.
5. When a remaining delta is small but persistent, build a same-ABI native
   control before blaming kernel ABI or runtime setup. For Loom kernel
   comparisons, this can be a native HIP/assembly artifact exporting the same
   symbol and taking the same kernel arguments as the Loom artifact, launched
   through the common runner's Loom path. If that control reaches the native
   reference timing, the remaining issue is emitted schedule or Loom
   expressibility, not the runner ABI.
6. Compare exact references adversarially:
   - if any exact reference is stably more than 5-10% faster, the Loom route is
     refuted and needs a new Loom algorithm/config axis;
   - if exact references are within 5%, treat the route as not refuted unless
     the bucket dominates model-level time;
   - if results are noisy, increase repetitions before deciding.
7. Probe storage-changing, approximate, or fused algorithms separately. A win
   there opens a new route family or fusion candidate; it does not refute the
   standalone exact route.
8. Use analytical lower bounds only as sanity checks: bytes moved, ops,
   dispatch floor, occupancy, register pressure, spills, and memory coalescing.
9. Record reusable algorithm lessons in the prior-art ledger and write a
   family-specific refutation report.

## WYSIWYG Refutation Loop

When a native reference beats a Loom candidate, inspect both listings before
calling it generic codegen quality:

- first verify that the Loom source is saying the intended schedule explicitly;
  if the source is scalar, the scalar listing is not a compiler surprise;
- identify the exact source-level schedule difference: load width, block/tile
  ownership, scale reuse, packed dot form, LDS staging, tail handling, or
  reduction structure;
- write the corresponding Loom source/config axis explicitly and remeasure;
- if the explicit Loom form cannot compile, record the exact target diagnostic
  as Loom-author feedback;
- if it compiles but the listing still lacks the intended load/arithmetic form,
  record the emitted instruction mismatch and keep the route refuted.

Do not ask Loom to discover a wide-vector schedule from scalar code. A scalar
source is already a claim about the schedule.
Likewise, do not make portable source target-specific just to reproduce a
measured target route. Target-specific source is reserved for real target-only
primitives or layouts; otherwise the refutation should manipulate source/config
axes and let metadata select target winners.

Current Q8_0/F32 lesson: on the updated Stella branch, high-level Loom can emit
the important packed/mixed inner sequence (`global_load_b32`,
`global_load_d16_b16`, `global_load_b128`, `v_bfe_i32`, `v_fma_mix_f32`,
`v_fmac_f32`) from `vector.bitunpacks<8>`. A common HIP module runner later
showed the WG128 Loom code object at p50 2.04 us versus HIP WG128 p50 2.00 us
for `k512_r64_c8`; the older 3.24-3.32 us Loom number was not an
apples-to-apples final codegen result. For the focused `k512_r64_c8` WG64
parity case, exact-shape unrolling of the two block iterations closed the p50
gap: three same-runner `rocprofv3` captures measured Loom at 1.96, 1.96, and
1.84 us versus HIP at 1.96, 1.96, and 1.96 us. That tiny shape is still too
close to timer floor for broad-family acceptance. On `k4096_r128_c8`, HIP WG64
measured around 4.12-4.16 us p50, baseline Loom WG64 around 7.88 us p50, and
exact-shape unrolled Loom around 6.12 us p50. On a larger `k8192_r128_c8`
hot-loop device-time check, baseline Loom WG64 measured 14.08 us p50 versus HIP
WG64 6.48 us p50. Use the tiny shape for listing/schedule diagnosis and
specific parity closure; use representative larger shapes for the actual
family-level done/not-done decision. If a route is still slower than a HIP
reference, investigate full schedule parity, reduction/control flow, loop
form/unroll factor, and measurement domain before claiming the primitive cannot
be expressed.

## Output

Produce:

```text
route claim
shape basket
exact-reference benchmark table
algorithmic-headroom benchmark table, if any
correctness result
resource/compile notes
refuted / not-refuted / open-new-family decision
next Loom axes to implement
```

## Guardrails

- Do not compare exact Loom kernels against approximate or packed-RHS kernels
  as if they had the same contract.
- Do not compare host dispatch timing against device kernel timing as a
  performance conclusion. Keep the timing domain in every table.
- Do not use IREE/HAL benchmark timing as the sole refutation evidence for
  sub-5us kernels. Build a common code-object runner or explain why that is not
  possible.
- Do not use all-zero/all-one benchmark data unless the op requires it.
- Do not benchmark competing kernels concurrently on the same GPU.
- Do not turn a native reference gap into a source-level target attribute unless
  the missing path is genuinely target-specific.
- Do not keep optimizing a route after repeated exact references fail to beat
  it by a material margin, unless model-level profiling says the bucket matters.
