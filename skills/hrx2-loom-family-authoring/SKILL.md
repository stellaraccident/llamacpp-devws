---
name: hrx2-loom-family-authoring
description: Use this when authoring or revising a Loom kernel family for the llama.cpp HRX2 backend, including prior-art harvest, ABI definition, provider/config axes, target metadata, randomized correctness fixtures, benchmarks, compile reports, and HRX manifest validation.
metadata:
  short-description: Author HRX2 Loom kernel families
---

# HRX2 Loom Family Authoring

Use this skill when creating or revising one Loom family for HRX2.

## Required Context

Read first:

- `docs/loom/llamacpp-integration-v1.md`
- `docs/loom/llamacpp-hrx-authoring.md`
- `docs/loom/backend-prior-art-algorithms.md`
- `docs/loom/llamacpp-hrx-stella-questions-2026-06-11.md` for current Loom target/provider/report details
- For an exemplar completed op, `docs/loom/rms-norm-standalone-done-gfx1100.md`

## Workflow

1. Harvest prior art first from CUDA/HIPified, Vulkan, OpenCL/Metal, old HRX,
   and current HRX2 Loom seeds. Record reusable algorithms in the ledger.
   Prior search is not complete until the schedule facts are written down:
   source/symbol, shape regime, tile/workgroup/subgroup shape, lane ownership,
   per-lane outputs, vector/packed load width, layout, dot/WMMA/ALU primitive,
   A/B staging, barriers, unroll, reduction/writeback, emitted resource facts,
   and known win/regression constraints.
2. Define the family contract: ggml op or fusion, ABI, bindings, constants,
   shape domain, numeric policy, layout/aliasing constraints, and target policy.
3. Decide target structure before coding:
   - target-neutral Loom source for portable algorithms and fallbacks;
   - separate source/artifact entries for truly target-specific kernels;
   - route `target_key` for measured winners, not for source portability guesses.
4. Keep portable `.loom` files free of `target(@...)`/`amdgpu.target<...>`
   attributes unless the source itself uses a target-specific primitive,
   lowering contract, ABI, or layout. For ordinary vector/load/reduction
   schedules, let the runtime/catalog target key select the tuned route and let
   Loom assign the target at compile time.
5. Mark production roots with `kernel.def export("stable_symbol")` before
   selective linking. Targetless roots without explicit export metadata can
   fail AMDGPU compilation with no compatible target record after linking.
6. Write a semantic baseline first. Add provider contracts with
   `func.apply`/`func.template` only where schedule choices need tuning.
7. Express schedule intent explicitly. Loom is WYSIWYG: do not expect it to
   infer wide loads, packed dot forms, scale amortization, tile ownership,
   lane/block mapping, or mixed-precision instruction selection from scalar
   code. Say the machine schedule you want in the source/config axes.
8. Add `check.case` coverage for representative decode, narrow multi-token,
   prefill, odd/tail, and boundary shapes. Use deterministic non-constant
   random or representative synthetic inputs; never benchmark all-zero/all-one
   patterns unless the op semantics require it.
9. Keep production catalog sources separate from check-only sources until
   `loom-link --strip-check` is reliable for the family. Current Q8 production
   packaging removes check roots from the embedded source because strip-check
   can remove a check symbol that is still considered required.
10. Compile one root/candidate at a time for initial tuning. Multi-root bytecode
   is fine for runtime packaging after root/config interactions are understood.
11. Request emit-stage detailed compile reports and artifact manifests. Treat
   spills, unexpected private/local memory, pathological code size, pressure
   rows, memory facts, and instruction mix as decision data.
12. Run Loom correctness for fast iteration, then focused ggml CPU-reference
   correctness before accepting any route into the HRX2 runtime catalog.
13. Before adding a production catalog row, write the candidate matrix row it
   came from. Each candidate must name the prior row or analytical schedule it
   follows, the single pivot axis, sweep bounds, expected signal, correctness
   gate, timing gate, and decision. Adjacent probes without direct prior
   evidence are valid only as brackets around an explicit schedule family.
14. Record accepted and rejected provider/config variants with JSON/JSONL evidence.

## WYSIWYG Schedule Rules

Before benchmarking a Loom candidate, inspect the source and ask whether it
literally says the expected hardware work:

- **Load width:** use `vector.load`, packed integer views, or explicit scalar
  loads according to the intended memory transaction. If the target ISA should
  show `global_load_b32`, `global_load_b128`, or a packed dot input, encode that
  shape directly instead of hoping adjacent scalar loads combine.
- **Lane ownership:** encode how workitems map to rows, blocks, tiles,
  subgroups, chunks, and tails. For quant blocks, make block-local structure
  explicit, such as "lane owns four Q8 values in one block" rather than
  iterating over individual logical elements.
- **Reuse/amortization:** load scales, zero-points, row constants, and LDS tiles
  at the intended granularity. A scalar loop that reloads a block scale for
  every quant will generally emit exactly that.
- **Arithmetic form:** encode `vector.dot*`, `vector.mulf`, `vector.reduce`,
  `vector.fmaf`, or scalar ops according to the desired ISA form. Treat missing
  target lowering as author feedback, not as a reason to fall back silently.
- **Packed transforms:** use source primitives such as `vector.bitunpacks<N>`
  when the intended dataflow is a packed word becoming lanes. This is better
  than scalar hand-unpack, but still inspect the listing. On the current
  Stella branch, Q8 `vector.bitunpacks<8>` plus f16-scale vector multiply can
  emit `global_load_d16_b16` and `v_fma_mix_f32`; do not keep hand-rolling that
  path unless the listing proves the generated form regressed.
- **Reduction schedule:** `kernel.workgroup.reduce` currently does not expose a
  source-level schedule knob. Try subgroup-only forms when a one-wave algorithm
  is semantically valid, but treat workgroup size, subgroup use, and any future
  low/rocasm helper as explicit algorithm axes rather than hidden heuristics.
- **Tail policy:** preserve algorithm choices as config axes unless the general
  source naturally degenerates to the special case when a shape fact is known.
- **Loop form/unroll factor:** when a static shape implies only a small number
  of loop iterations, make `looped` versus `exact_unrolled` or a bounded unroll
  factor an explicit tuning axis. Do not promote full unrolling as a general
  heuristic: inspect code size, live values, spills, and larger-shape behavior.
  Prefer `scf.for ... unroll(%factor)` over hand-duplicated source when the
  loop trip count is static. If the natural loop has dynamic lower/upper bounds
  such as a lane-dependent start, rewrite it as a static ordinal loop and derive
  the dynamic induction value inside the body. Verify the target listing has no
  residual loop. On the current branch, partial unroll factors are not proven;
  full-trip-count unroll worked for the focused two-iteration Q8 case. Direct
  `loom-compile --config=name=value` can select the unroll factor; if the
  benchmark CLI cannot bind that config directly, make the sweep wrapper
  generate or compile one config at a time and record that limitation.

After compile, verify WYSIWYG with the compile report and target listing:

- expected global/LDS load instruction widths are present;
- scale/control loads occur at the planned granularity;
- dot/MFMA/WMMA/vector arithmetic appears only when intentionally requested;
- spills, private memory, code size, and register pressure match the design;
- use true device-time evidence for kernel comparisons. `dispatch_complete`
  p50 is useful for runtime overhead, but kernel refutation needs a calibrated
  device-time path. For sub-5us kernels, prefer a common code-object runner that
  launches Loom and native references through the same API, such as a HIP module
  runner measured by `rocprofv3 --kernel-trace`. Use `iree-benchmark-loom
  --profile-final-batch=true
  --profile-data=dispatch-events,executable-metadata` as diagnostic evidence,
  and record benchmark score, `dispatch_function`,
  `dispatch_command_operation`, profile-summary clock uncertainty, launch
  metadata, and runner/DB paths separately;
- any target-lowering blocker is recorded in `docs/loom/loom-author-feedback.md`
  with the exact diagnostic, source snippet, and target listing evidence.

For Q8_0/F32, the failure mode to remember is concrete: scalar Loom source
emitted scalar Q/RHS/scale work and measured slower. A better `block4` source
that explicitly maps lanes to four Q8 values per block improved device time.
Direct packed-word forms can emit `global_load_b32` for Q and `global_load_b128`
for RHS, and `vector.bitunpacks<8>` is a viable source spelling for unpacking a
packed Q word. The updated Stella branch can also emit `global_load_d16_b16`
and `v_fma_mix_f32` for the Q8 scale path. A later common HIP module runner
showed the WG128 Loom target artifact essentially tied with the HIP WG128
reference under `rocprofv3`; the remaining WG64 gap is small and appears to be
schedule parity, especially address/wait placement, byte-extract details, and
workgroup reduction/control schedule. For the focused `k512_r64_c8` WG64 case,
an exact-shape unrolled `word4_bitunpack_unrolled_rhsvec_dotf` source closed
the same-runner rocprof p50 gap to HIP parity. The preferred spelling is now
`word4_bitunpack_scfunroll_rhsvec_dotf`, which uses `scf.for
unroll(%q8_unroll_factor)` over a static ordinal loop and preserves the unroll
amount as a config value. The same unroll axis did not close larger Q8 shapes,
so preserve it as a tunable shape-specific axis instead of hard-coding it as
the Q8 default.

## Low-Code Escape Hatch

Use low code when the desired instruction sequence cannot be represented by
target-neutral source or the AMDGPU descriptor/lowering set is missing a key
operation.

- Pure `low.kernel.def` AMDGPU kernels can compile and run today through the
  explicit low prep pipeline:
  `amdgpu-materialize-hal-kernel-abi,canonicalize,cse,low-select-operand-forms,low-dce,low-materialize-allocation`.
- `iree-benchmark-loom` currently does not benchmark pure `low.kernel.def`
  kernels through the same static workgroup-count path as `kernel.def`; use
  `iree-run-loom` for correctness and document benchmark gaps.
- Current author guidance for high/low interop is declaration/link based:
  high modules use `func.decl`, low modules provide `low.func.decl` or low
  definitions, the high module is lowered to low, then the low modules are
  linked. `low.invoke` is the intended friendlier ABI bridge when available.
- In the current branch, AMDGPU `source-to-low` does not appear to lower
  imported `func.decl` symbols because the AMDGPU lowering policy has no
  nonzero `import_decl_kind`. Treat this as toolchain feedback and record the
  exact smoke result before depending on this path.

## Output

For each family, produce:

```text
family source or bytecode package
ABI note
candidate axes
correctness results
benchmark results
compile report
artifact manifest
HRX2 catalog record draft
Loom author feedback, if tool limitations were found
```

## Guardrails

- Do not hand-pick a winner without benchmark evidence.
- Do not start from a blind schedule guess. Start from a documented prior
  schedule or an analytical alternative derived from the prior matrix.
- Do not integrate speculative tile/vector/unroll/staging pivots directly into
  llama.cpp. Run them as standalone Loom/kernel/backend-op sweeps first, then
  promote only a measured winner with a clear shape domain.
- Do not rely on the compiler to infer wide packed loads, dot forms, or tail
  strategies that source/config axes can spell.
- Do not bury tunable algorithm choices in hard-coded heuristics unless a
  superset naturally degenerates into the subset by shape.
- Do not mix true target-specific Loom code into a portable source file. Keep
  target-specific files separate and select them with metadata. Scalar/vector
  baselines should usually stay target-neutral; target-specific source is for
  target-fiddly primitives such as chip-specific WMMA layouts.
- Treat the production catalog as embedded JSON parsed by llama.cpp, not as
  hand-authored C++ route policy.
- Do not put kernel-specific tuner scripts in llama.cpp. Keep scratch/specialized
  tuners in the workspace; production llama.cpp should contain generic catalog,
  embed, validation, runtime, and JIT plumbing.
- Keep feedback for the Loom author in the relevant `docs/loom/` feedback note.
