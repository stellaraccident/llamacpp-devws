# HRX2 Phase 0 Bringup Results

Date: 2026-06-11

## Scope

Phase 0 implemented the final-form path for one unfused op:
`GGML_OP_RMS_NORM`.

The path is intentionally narrow:

- new `GGML_HRX2` backend option in llama.cpp;
- normal ggml backend registry/device/buffer integration;
- embedded JSON route metadata with a dev override via
  `GGML_HRX2_CATALOG_DIR`;
- embedded Loom source plus exploded source file at
  `ggml/src/ggml-hrx2/kernels/rms_norm_f32.loom`;
- HRX-owned `hrx_loom_jit` wrapper library in `hrx-system`;
- runtime Loom JIT compile to HSACO for the detected HRX GPU architecture;
- HRX executable load/lookup/dispatch;
- focused ggml CPU-reference validation through `test-backend-ops`.

## Implemented Route

Route id: `rms_norm_f32_contiguous`

Supported shape/layout:

- `GGML_OP_RMS_NORM`
- source and destination type `GGML_TYPE_F32`
- contiguous source and destination
- non-view destination
- non-overlapping input/output storage
- `ne[0] <= 65536`
- `ggml_nrows(src0) <= 1048576`

Rejected in Phase 0:

- non-contiguous/view inputs;
- in-place RMS_NORM;
- non-F32 types;
- fused RMS_NORM variants.

The Loom source uses dynamic `ncols`/`nrows` and explicit `index.assume`
range facts so AMDGPU lowering can prove non-negative 32-bit addresses.

## Evidence

Loom compile smoke:

```bash
build/hrx-install/bin/loom-compile \
  sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rms_norm_f32.loom \
  --backend=amdgpu-hal \
  --target=gfx1100 \
  --compile-root=@hrx2_rms_norm_f32 \
  --output=/tmp/rms_norm_f32_gfx1100.hsaco \
  --emit-artifact-manifest=/tmp/rms_norm_f32_gfx1100.manifest.json \
  --compile-report=details \
  --compile-report-output=/tmp/rms_norm_f32_gfx1100.report.json
```

Key compile facts for the generic `gfx1100` artifact:

- HSACO size: 9184 bytes
- wavefront size: 32
- workgroup size: 512x1x1
- SGPR count: 14
- VGPR count: 12
- instruction count: 165
- spills: 0
- local memory: 64 bytes

Loom correctness-gated dispatch benchmark:

```bash
LD_LIBRARY_PATH="$PWD/build/hrx-install/lib:$PWD/rocm/lib:$LD_LIBRARY_PATH" \
build/hrx-install/bin/iree-benchmark-loom \
  sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rms_norm_f32.loom \
  --device=amdgpu \
  --benchmark=@hrx2_rms_norm_f32_decode \
  --sample=0 \
  --sample-compilation=per_sample \
  --measure=dispatch_complete \
  --compile-report=details \
  --output=cache/loom/hrx2-phase0/rms_norm_decode.json
```

Result:

- correctness failures: 0
- p50 dispatch timing: 20431 ns for `ncols=4096,nrows=1`
- per-sample emitted instruction count: 112
- per-sample spills: 0
- per-sample local memory: 64 bytes

ggml CPU-reference validation:

```bash
LD_LIBRARY_PATH="/srv/vm-shared/projects/llamacpp-devws/build/hrx-install/lib:/srv/vm-shared/projects/llamacpp-devws/rocm/lib:$LD_LIBRARY_PATH" \
../../build/llama-hrx2/bin/test-backend-ops \
  test -b HRX20 -o RMS_NORM --output csv
```

Result:

- accepted contiguous non-in-place F32 RMS_NORM rows passed;
- non-contiguous `v=1` rows were reported as `not supported`;
- in-place row was reported as `not supported`.

## Notes

The HRX runtime export metadata reports `parameter_count=13` for the Loom
HSACO while the Loom manifest reports `parameter_count=11`. Bindings and
constant bytes agree: 2 bindings and 44 constant bytes. HRX2 treats HRX runtime
export metadata as the dispatch ABI source of truth.

The standalone `iree-test-loom` path currently needs static workgroup counts
for HAL actual invocation. `iree-benchmark-loom --sample-compilation=per_sample`
worked for the Phase 0 correctness-gated benchmark and is the useful path for
dynamic-shape tuning today.
