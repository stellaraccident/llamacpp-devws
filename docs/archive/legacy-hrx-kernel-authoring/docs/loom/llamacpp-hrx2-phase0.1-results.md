# HRX2 Phase 0.1 Results

Date: 2026-06-11

## Implemented

- Converted the HRX2 catalog to `ggml-hrx2-catalog-v1` with explicit sources,
  bytecode artifacts, routes, target keys, ABI metadata, shape domains, and
  dispatch metadata.
- Added CMake-time `loom-link --mode=selective` generation of per-route
  `.loombc` artifacts and embedded those artifacts into `ggml-hrx2`.
- Added lazy runtime provider compilation from embedded or exploded
  `loom-bytecode`, with Loom text fallback for development.
- Added a second proof kernel: contiguous 2D `GGML_OP_MUL_MAT` for
  `Q8_0 x F32 -> F32`.
- Verified target-aware Loom provider selection on the Q8 kernel via selective
  linking; the link plan reports provider dependencies.

## Validation

- `cmake --build build/llama-hrx2 --target ggml-hrx2 -j$(nproc)`: pass.
- `cmake --build build/llama-hrx2 --target test-backend-ops -j$(nproc)`: pass.
- Embedded catalog:
  - `test-backend-ops test -b HRX20 -o RMS_NORM --output csv`: supported
    non-inplace cases pass.
  - `test-backend-ops test -b HRX20 -o MUL_MAT --output csv`: supported
    `q8_0 x f32` contiguous cases pass.
- Exploded catalog:
  - `GGML_HRX2_CATALOG_DIR=/tmp/hrx2-dev-catalog` with generated `.loombc`
    artifacts compiles and runs both RMS_NORM and Q8 MUL_MAT from disk
    bytecode.

## Known Limits

- The Q8 kernel is an infrastructure proof, not a performance candidate. It is
  one output row/column per workgroup with scalar per-lane accumulation and a
  workgroup reduction.
- The Q8 route intentionally only accepts simple contiguous 2D layouts. Batched,
  permuted, F16 RHS, and large Qwen-ish `n=4096` shapes remain unsupported in
  this phase.
- `loom-link --strip-check` is not used because it currently strips check
  symbols that the planner still treats as required.
