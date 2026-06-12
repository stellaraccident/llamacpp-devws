# Loom Author Feedback

Date: 2026-06-11

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
