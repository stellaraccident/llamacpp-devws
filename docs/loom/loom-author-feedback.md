# Loom Author Feedback

Date: 2026-06-11

Concrete bugs and limitations that affect correctness, measurement, route
admission, or required workarounds are tracked separately in
`docs/loom/loom-bugs-limitations.md`. Keep this file for author-facing design
feedback, diagnostics, ergonomics, and future tool requests.

## Phase 0 HRX2 Bringup

1. `iree-test-loom` rejects dynamic workgroup counts for HAL actual
   invocation, even when check.case samples bind concrete values:

   ```text
   HAL actual invocation requires a statically resolved workgroup count after sample constants are applied
   ```

   `iree-benchmark-loom --sample-compilation=per_sample` works for the same
   source. It would be helpful if the fast correctness runner either used the
   same specialization path or suggested that flag/tool in the diagnostic.

2. The Loom manifest and HRX runtime export metadata disagree on the name/value
   of `parameter_count` for the RMS_NORM kernel:

   - Loom manifest: `binding_count=2`, `parameter_count=11`,
     `constant_byte_length=44`
   - HRX runtime export info: `binding_count=2`, `parameter_count=13`,
     `constant_byte_length=44`

   The difference appears to be scalar launch parameters only vs scalar plus
   buffer parameters. A clearly named pair, or a client-facing ABI summary for
   HRX dispatch, would avoid guesswork.

3. Loom's public artifact format for the emitted executable is
   `amdgpu-hsaco`, while HRX/IREE executable loading wants either format
   inference or the target code-object string such as
   `amdgcn-amd-amdhsa--gfx1100`. This is manageable, but the integration point
   needs an explicit "loader format" field or guidance.

4. AMDGPU lowering diagnostics for dynamic addressing were useful and precise.
   The required `index.assume` facts are reasonable, but this is important
   enough for authoring docs: dynamic flat/global loads need non-negative
   32-bit address proofs, and examples should show the intended pattern.

5. The compile report already exposes useful tuning data for the HRX2 catalog
   flow: spills, local/private bytes, instruction mix, code bytes, and register
   pressure rows. The compact `static_summary` emitted by
   `iree-benchmark-loom` is a good shape for automated sweeps.

## Phase 0.1 HRX2 Catalog Pipeline

1. `loom-link --mode=selective` correctly pulls provider implementations for
   `func.apply<contract>` roots out of the Q8 test kernel. The printed plan
   reports `provider` as the live reason, which is exactly the kind of signal a
   catalog authoring pipeline can use.

2. `loom-link --strip-check` currently fails for phase0.1 sources even when the
   `check.case` symbols are not public:

   ```text
   NOT_FOUND; required symbol '@hrx2_mul_mat_q8_0_f32_zero_case' was stripped
   ```

   The same source links, bytecode-compiles, and runs correctly without
   `--strip-check`. For production catalog artifacts we need either a fixed
   strip mode or guidance that checks should live in a separate source/library.

3. The address-proof diagnostics were useful for the Q8 kernel. A broad dynamic
   domain failed until the route and `index.assume` facts bounded `k`, `rows`,
   and `cols` tightly enough. That is good behavior, but examples should show
   how catalog shape-domain metadata maps to Loom range assumptions.

4. Target-aware providers are usable in the intended integration shape:
   `func.template ... target(@gfx1100)` participates for gfx1100, the untargeted
   fallback remains in source for portability, and selective linking can produce
   route-specific bytecode before runtime target specialization.

## Phase 0.2 Exact-Shape JIT Config

1. The intended final-form flow works: `loom-link --mode=selective` can emit
   bytecode with unresolved `config.decl` symbols, and the C API can later
   compile that bytecode with per-invocation config bindings. This let HRX2
   compile exact-shape Q8_0/F32 kernels at provider creation time from embedded
   bytecode.

2. It would help catalog automation if the compile report exposed the resolved
   config binding set and the emitted dispatch ABI summary (`binding_count`,
   runtime scalar/direct-arg count, constant bytes) in the same report object.
   Today the report confirms root/target/export/artifact facts, but HRX2 still
   has to load the HSACO and query runtime export metadata to validate ABI.

3. I moved the experimental JIT shim source into the llama.cpp HRX2 backend so
   it is not a required HRX runtime API. For a clean standalone build without
   a transitional HRX wrapper target, Loom needs to install/export the raw C API
   CMake targets (`loom::binding::c::loomc` and the AMDGPU target binding) or
   provide an equivalent package-level target.

## Phase 0.4 Mini Tuning

1. `iree-benchmark-loom --measure=dispatch_complete` is the right path for
   standalone GPU kernel timing in this workspace. `case_end_to_end` on an
   AMDGPU kernel failed with:

   ```text
   no actual invocation provider is configured
   ```

   This is fine once known, but the docs/examples should steer GPU kernel
   tuning toward `dispatch_complete`.

2. `--sample-compilation=per_sample` was required for HRX2 RMS_NORM dispatch
   benchmarks so concrete check.case parameters could resolve the workgroup
   count. With that flag, the tool produced plan, compile, sample, benchmark,
   and summary JSONL rows, including `static_summary`.

3. The HRX2 Q8_0/F32 benchmark declarations currently plan zero
   `dispatch_complete` samples even for named benchmarks that bind all three
   case parameters:

   ```text
   benchmark_sample_count: 0
   case_cartesian_sample_count: 8
   ```

   This may be user error in the benchmark shape or a limitation around cases
   that pass sampled scalar values as actual invocation arguments. Either way,
   a diagnostic that says why the benchmark dictionary selected no samples
   would save time.

4. The dispatch benchmark loop is usable for small automated tuning runs when
   bounded explicitly with low `--iterations`, `--max-batches`, and a command
   timeout. The JSONL schema had enough information to reduce candidates by
   p50 latency while carrying spills, code size, register pressure, and memory
   use as secondary decision data.

## RMS_NORM Standalone Tuning

1. The AMDGPU diagnostic for unsupported cache policies was precise and carried
   the right source location. A `vector.load` with
   `{cache_scope = device, cache_temporal = non_temporal}` failed with
   `ERR_AMDGPU_024` because device/non-temporal global memory is not encodable
   by descriptor set `amdgpu.rdna3.core`.

2. It would help autotuners if cache-policy axes could be queried from the
   target descriptor set before generating candidates. The current diagnostic
   is good after the fact, but a cheap capability query would let a sweep prune
   invalid memory-policy combinations up front.

3. `kernel.workgroup.reduce<addf>` generated zero-spill RMS_NORM candidates
   close to the copy/traffic floor across the tested gfx1100 shapes. This is a
   good default authoring primitive for first-pass row reductions.

4. A unified RMS_NORM bytecode library containing both generic and config-driven
   roots exposed a root pruning/config-resolution rough edge. Compiling the
   generic `@hrx2_rms_norm_f32` root from the multi-root `.loombc` failed until
   HRX2 supplied otherwise-unused config bindings for the static roots:

   ```text
   unresolved config '@hrx2.tuning.rms_norm.workgroup_size' remains for final compilation
   ```

   The route-level workaround is to bind every config declared in the bytecode
   package, even for roots that do not use those configs. The nicer behavior for
   catalog integration would be root-scoped pruning before final config
   validation, or a compile option that only requires configs reachable from the
   selected root.

## Q8_0/F32 Refutation

1. The first exact native HIP refutation for Q8_0/F32 initially looked like a
   4-5x gap, but that compared HIP event timing against Loom
   `dispatch_complete` timing. A later code-object-level comparison changed the
   conclusion again: loading the same Loom target ELF through the HIP module API
   and measuring it with `rocprofv3 --kernel-trace` gives p50 2.04 us for the
   WG128 Loom `word4_bitunpack_rhsvec_dotf` kernel versus p50 2.00 us for the
   HIP WG128 reference on `k512_r64_c8`.

   This suggests the earlier 3.24-3.32 us Loom number came from the Loom/IREE
   benchmark/profile path rather than the code object under an equivalent launch
   path. Please treat this as a request for guidance on sub-5us dispatch-event
   profiling: either the dispatch-event rows are measuring a different interval,
   the final-batch batching profile needs a different interpretation, or there
   is a runner overhead/clock attribution issue that can mislead kernel authors.

2. This is a good WYSIWYG authoring example. The original Loom source said
   "one quant per lane iteration", so the listing did exactly that: scalar Q
   load, scalar RHS load, and scale reload per quant. A better source maps each
   lane to a four-quant chunk inside a Q8_0 block, loops over blocks with
   `block_idx = lane / 8`, loads one scale per four Q values, and accumulates
   four products before the workgroup reduction. That source-level algorithm
   change alone improved device time from 4.44 us to 3.44 us.

3. The currently blocked fully packed form is concrete. This Loom source shape:

   ```text
   %q_i8 = vector.load %src0_i8_view[%q_byte] : view<...xi8, #dense> -> vector<4xi8>
   %rhs = vector.load %src1_view[%src1_index] : view<...xf32, #dense> -> vector<4xf32>
   %q_f32 = vector.sitofp %q_i8 : vector<4xi8> to vector<4xf32>
   ```

   fails AMDGPU target validation with:

   ```text
   ERR_TARGET_003: Target contract guard constraint is not satisfied.
   target 'amdgpu-rdna3' ... rejected 'vector.sitofp' type 'vector<i32>'
   constraint 'amdgpu.arithmetic.vector_i32' is not satisfied
   ```

   Replacing vector conversion with scalar extracts then fails with:

   ```text
   ERR_TARGET_001: Target has no lowering contract for an operation.
   target 'amdgpu-rdna3' ... has no target-low contract for 'vector.extract'
   ```

   This may just be missing AMDGPU lowering coverage. It is worth fixing because
   the HIP baseline's winning ISA shape is specific: packed Q load
   (`global_load_b32` for four Q bytes), wide RHS load, one f16 scale load, and
   four products.

4. A direct workaround can express the load widths without waiting for
   `vector<4xi8>` lowering:

   - create a byte-offset `view<1xi32>` at the Q payload address and load it to
     get `global_load_b32`;
   - use `vector.load -> vector<4xf32>` for RHS to get `global_load_b128`;
   - use `vector.dotf` to request RHS FMA accumulation.

   This compiles and the listing matches the requested load shape. A later
   source variant using `vector.bitunpacks<8>` on the packed word also compiles.
   On the updated branch it emits `global_load_d16_b16`, `v_bfe_i32`,
   `v_fma_mix_f32`, and `v_fmac_f32`, so the old "missing mixed scale multiply"
   diagnosis is obsolete. Under a common HIP module runner the best WG128 Loom
   artifact is essentially tied with the HIP WG128 reference, so the older
   3.24-3.32 us dispatch-event profile should not be treated as final codegen
   evidence.

5. The residual WG64 gap is now small and specific: Loom p50 2.00 us vs HIP p50
   1.88 us under the same HIP module runner. Removing the reduction keeps most
   of the gap (Loom p50 1.92 us vs HIP p50 1.72 us), so the remaining issue is
   likely inner-loop address/schedule/wait placement rather than a missing
   high-level packed form. The listings both contain `global_load_b32`,
   `global_load_d16_b16`, `global_load_b128`, `v_bfe_i32`,
   `v_fma_mix_f32`, and `v_fmac_f32`; HIP still has different explicit
   scheduling/control shape (`s_clause`, `s_delay_alu`, wait placement, and
   different LDS/reduction structure).

6. The next specific AMDGPU codegen/authoring gap is full schedule parity, not
   the broad unpack/scale arithmetic shape. HIP and Loom now share the important
   packed/mixed instructions, but still differ in byte-extract details,
   reduction/control flow, wait placement, and measurement domain. Compile
   report detail that explains these schedule choices would help agents decide
   whether to keep manipulating source axes or move to low/rocasm.

7. The gap is not explained by Q8_1 packed RHS or target-specific dot forms.
   The refutation baseline preserves exact F32 RHS semantics, so any remaining
   same-runner delta should be treated as the next Loom authoring/codegen target
   before moving on to approximate packed-RHS families.

8. Inline LLVM assembly is not currently a viable escape hatch from high-level
   AMDGPU source. An attempted `llvmir.inline_asm` source variant parsed but
   failed target lowering with:

   ```text
   ERR_TARGET_001: has no target-low contract for 'llvmir.inline_asm'
   ```

   Pure target-low AMDGPU kernels do work when compiled through the low prep
   pipeline. This smoke test parsed descriptor-backed low asm, emitted HSACO,
   and passed through `iree-run-loom`:

   ```bash
   build/hrx-install/bin/iree-run-loom cache/hrx2/q8_lowasm_probe/min_low_kernel.loom \
     --backend=amdgpu --compile-root=@min_store \
     --pipeline=amdgpu-materialize-hal-kernel-abi,canonicalize,cse,low-select-operand-forms,low-dce,low-materialize-allocation \
     --workgroup-count=1,1,1 \
     --binding=1xf32=0 --expected-binding=1xf32=1 \
     --compile-report=summary \
     --emit-target-artifact=cache/hrx2/q8_lowasm_probe/min_low_kernel_run_lowprep.hsaco
   ```

   `iree-benchmark-loom` could not benchmark that pure `low.kernel.def` because
   it does not currently derive or accept static workgroup counts for low-only
   kernels the same way it does for `kernel.def`.

9. Per the author, the intended current low-code bridge is declaration/linking
   based: use `func.decl` on the high side and `low.func.decl`/low definitions
   on the target-low side; lower the high module to low, then link it with the
   low module. `low.invoke` is the friendlier planned spelling for this ABI
   transition.

   The installed AMDGPU policy in this branch does not appear to enable imported
   declaration lowering yet: `kAmdgpuLowLowerPolicy` has no
   `import_decl_kind`, and `source-to-low` skips imported declarations when the
   policy import kind is zero. A minimal
   `func.decl import("rocasm", "test.add_i32") target(@target) ...` smoke
   therefore remains a `func.decl` after `loom-opt --pass=source-to-low`.
   IREEVM does set `LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM`, so this may simply be
   missing AMDGPU policy plumbing or the wrong current CLI flow.

10. The 2026-06-12 Stella branch update fixed the most important stale Q8
   lowering complaint: the high-level `vector.bitunpacks<8>` path now emits the
   expected packed/mixed inner shape (`global_load_b32`, `global_load_d16_b16`,
   `global_load_b128`, `v_bfe_i32`, `v_fma_mix_f32`, `v_fmac_f32`). The broad
   high-level form is now close enough that the remaining ask is narrower: help
   explain or control schedule differences and timing evidence when the listing
   has the right broad load/macc form.

11. `kernel.workgroup.reduce<addf>` currently has no source-level schedule
    selector. A WG32 `kernel.subgroup.reduce<addf>` variant compiled and passed
    but did not improve the focused Q8 shape. For hard reductions, it would be
    useful to have either a schedule/config knob, compile-report details that
    make the selected reduction plan obvious, or documented guidance for when
    to drop to low/rocasm.

12. The new dispatch-event profiling evidence is useful but noisy at this
    scale. For the Q8 WG128 winner, a batch-64 run reported benchmark p50
    around 3.38 us, `dispatch_function` mean around 2.76 us, and individual
    operation rows alternating around 1.7 and 3.4 us. The profile summary also
    reports clock uncertainty around 4-9 us while the kernel itself is shorter
    than that. This may be expected for the current profiler, but agents need
    guidance on which row to use for sub-5us kernels and how much confidence to
    assign to cross-method comparisons against rocprof/HIP.

13. A focused WG64 parity attempt reproduced the remaining same-runner gap:
    three independent `rocprofv3` runs measured HIP WG64 at p50 1.96 us and
    Loom WG64 at p50 2.12 us with identical launch metadata. A static HIP
    control exporting `q8_0_f32_candidate(src0, src1, dst)` with hard-coded
    `k=512`, `rows=64`, `cols=8` also measured p50 1.96 us through the runner's
    Loom-style three-argument path. This rules out the runner ABI and
    exact-shape specialization as explanations.

14. The specific high-level scheduling blocker is now sharper. The Loom source
    variant that moved scale work before RHS load compiled to an identical
    target listing, still with a single `s_waitcnt vmcnt(0)` before unpack and
    MACC. The HIP parity template instead uses `s_clause`, `s_waitcnt vmcnt(2)`
    before Q byte extraction, `s_waitcnt vmcnt(1)` before scale FMAs, and
    `s_waitcnt vmcnt(0)` before RHS MACCs, with explicit `s_delay_alu`
    scheduling. If Loom has a way to express or preserve that staged wait
    schedule from high-level source, it is not obvious from the current docs.

15. The tiny `k512_r64_c8` Q8 shape is useful as a schedule microscope but is
    too close to the timer/profiler floor for final acceptance. A larger
    same-runner `rocprofv3` rerun at `k4096_r128_c8` measured HIP WG64 at
    p50 4.12-4.16 us and baseline Loom WG64 at p50 7.88 us. An exact-shape
    unrolled Loom variant compiled without spills and improved to p50 6.12 us,
    but remained materially behind HIP. For this kernel, large-shape evidence
    suggests the remaining issue is real schedule/code shape rather than
    measurement noise.

16. The benchmark methodology needs explicit labels in reports. Same-runner
    hot-loop device-time measurements are useful for Loom-vs-HIP emitted-code
    parity because both code objects get the same hot kernarg/buffer reuse. They
    should not be treated as final application realism. Loom's rotated-buffer
    benchmark design is the right defense against those benchmark mistakes, but
    if operation timestamps include host round trips, agents need a documented
    way to extract or label pure device kernel time versus tool operation time.
    A larger hot-loop `k8192_r128_c8` check still showed a large device-time
    gap, Loom p50 14.08 us versus HIP p50 6.48 us, so the current Q8 issue is
    not only timer-floor noise.

17. For cross-tool parity/refutation claims against existing native kernels, the
    cleanest methodology is to keep the final measurement in one tooling
    universe: emit a Loom target artifact/code object, emit a HIP C++ HSACO,
    load both with a HIP module API runner, and measure both with `rocprofv3
    --kernel-trace`. This should not be required for ordinary Loom hill
    climbing; a self-consistent Loom benchmark loop is fine there. Loom
    benchmark timing can be self-consistent and HIP/rocprof timing can be
    self-consistent while still not being directly cross-consistent. The docs
    should encourage this code-object runner pattern specifically for
    native-reference comparisons.

18. The focused `k512_r64_c8` Q8 WG64 gap was closed by presenting the two
    block iterations as a straight-line exact-shape unrolled Loom body. Three
    same-runner `rocprofv3` captures measured unrolled Loom at p50 1.96, 1.96,
    and 1.84 us versus HIP WG64 at p50 1.96, 1.96, and 1.96 us. This is good
    evidence that loop/control form can be a first-class tuning axis. It does
    not eliminate the larger-shape scheduling request: exact unrolling improved
    but did not close `k4096_r128_c8`, where Loom remained around p50 6.12 us
    versus HIP around p50 4.12-4.16 us. For larger K, agents still need a way
    to express or inspect staged wait scheduling, partial unroll/software
    pipeline choices, and reduction schedule choices without exploding code
    size.

19. The focused Q8 source can now be spelled with `scf.for` and an SSA unroll
    factor instead of manual body duplication. The accepted source uses:

    ```text
    config.def @hrx2.tuning.q8_0_f32.unroll_factor = 2 : index
    %q8_unroll_factor = config.get @hrx2.tuning.q8_0_f32.unroll_factor : index
    %sum = scf.for %i = [0 to 2 step 1](%acc = %zero : f32) -> (f32)
        unroll(%q8_unroll_factor) {
      ...
    }
    ```

    The ordinal loop form matters. The natural loop
    `[block_slot to blocks_per_row step block_step]` has the right runtime
    semantics but does not expose exact static bounds to the current unroll
    pass. Rewriting it as `[0 to trip_count step 1]` and computing
    `block_idx = block_slot + i * block_step` inside the loop let the standard
    target pipeline erase the loop. Three same-runner `rocprofv3` captures
    measured this SCF-unrolled source at p50 1.96 us versus HIP WG64 p50 1.96
    us on every run.

20. The current `scf.for` unroll transform appears to accept only no-op factors
    and exact full-trip-count unrolls. The test
    `loom/src/loom/transforms/test/scf_unroll.loom-test` rejects a partial
    dynamic factor. That is fine for the focused two-iteration Q8 microscope,
    but it leaves the large-K Q8 path without a clean bounded partial-unroll
    spelling. For kernel-catalog tuning, agents need either supported partial
    unroll factors or a documented alternative for "group N loop iterations
    per software-pipeline body without fully exploding code size."

21. `loom-compile` can specialize the SCF unroll factor with
    `--config=hrx2.tuning.q8_0_f32.unroll_factor=0`; that compile produced a
    looped artifact and report as expected. `iree-benchmark-loom --help` does
    not currently advertise an equivalent direct `--config` flag, even though
    benchmark dictionaries are the intended tuning substrate. If benchmark-time
    config sweeps are supported under another spelling, that should be
    documented; otherwise agents will need wrapper-generated sources or a
    compile step per config value.

22. Production HRX2 targetless Q8 roots needed explicit export metadata:

    ```text
    kernel.def export("hrx2_mul_mat_q8_0_f32_static_packed_scf_unroll")
      @hrx2_mul_mat_q8_0_f32_static_packed_scf_unroll()
    ```

    Without the `export(...)` attribute, a root-only selective link followed by
    AMDGPU compile failed with:

    ```text
    TARGET/011: module contains no function with a target record compatible
    with target pipeline 'AMDGPU HAL-native'
    ```

    Adding `amdgpu.target<gfx11-generic>` or `target(@...)` to the root also
    makes compilation work, but that is the wrong default for portable HRX2
    sources. The correct integration shape is target-neutral source plus
    explicit export metadata, with the runtime/catalog selecting the measured
    target route. It would help if Loom documented this requirement or produced
    a diagnostic that suggests adding an export record when compiling a
    targetless exported root.

23. The current C API path for llama.cpp had to mirror the CLI as:

    ```text
    source/bytecode -> link index -> loomc_link_module(root, target_selection)
      -> loomc_compile_module(config, target_selection)
      -> loomc_emit_module(AMDGPU HSACO, compile_report(details), manifest)
    ```

    The installed headers do not have the previously discussed
    `compile_root_symbol` or targetless-assignment helper in the public path I
    could use from llama.cpp. Root selection through `loomc_link_module` works
    and is probably the right layering, but examples should show this exact
    selective-link-before-compile flow for embedded JIT clients.

24. Detailed compile reports for embedded JIT are emitted from the emit stage,
    not from compile-stage artifact flags. The working C API chain uses
    `loomc_compile_report_options_t { mode = DETAILS }` on
    `loomc_emit_options_t::next`, then reads an artifact with kind `REPORT` and
    format `LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON`. It would be easy for
    clients to miss this because the compile object and the CLI name both imply
    "compile report"; a minimal C example would prevent that.

25. The production Q8 source still cannot keep check roots in the same source
    while building stripped catalog artifacts. `loom-link --strip-check`
    previously failed with a required check symbol being stripped. HRX2 removed
    check cases from the production Q8 source for now and relies on separate
    Loom validation plus ggml CPU-reference tests. A fixed strip-check mode
    would let one family file carry both authoring checks and production roots.

## Phase 1.0 Coverage Bringup

1. A first high-level f32 GET_ROWS candidate hit the same broad address-proof
   family as SET_ROWS, but on an `index.shli` in the load/gather path:

   ```text
   TARGET/003: target 'amdgpu-rdna3' export 'hrx2_get_rows_f32'
   config 'amdgpu.rdna3.core' rejected 'index.shli' address-width 'u32'
   constraint 'amdgpu.address.u32' is not satisfied
   ```

   The attempted source is preserved at
   `cache/hrx2/phase1_0/rejected-get-rows/get_rows_f32.loom`. The source
   already includes many explicit `index.assume` facts around the computed
   source, index, and destination offsets, so this is a useful concrete case
   for documenting the intended high-level spelling for ggml gather/copy ops
   with dynamic row indices.

2. The same GET_ROWS candidate also showed numeric mismatches on some
   `ncols=256` cases when it reached execution, with errors around `2.0`.
   That may be a bug in the attempted indexing formula rather than Loom, but it
   is a reminder that address-lowering fixes are not enough: route admission
   still needs `test-backend-ops` CPU-reference coverage for every admitted
   layout.
