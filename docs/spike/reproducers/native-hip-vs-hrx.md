# Native HIP vs HRX Reproducer

These commands reproduce the native HIP comparison build used for `docs/spike/reports/native-hip-vs-hrx.md`.

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
export ROCM_PATH=$ROOT/rocm
export PATH=$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}
export MODEL=$ROOT/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf

cmake -S $ROOT/sources/llama.cpp-hip -B $ROOT/build/llama-hip-compare -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM_PATH/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DGGML_HRX=ON \
  -DGGML_HRX_ROCM_PATH=$ROCM_PATH \
  -DGGML_HRX_AMDGPU_TARGETS=gfx1100 \
  -DGGML_HRX_BUILD_HIP_BENCHES=OFF \
  -DGGML_NATIVE=OFF

cmake --build $ROOT/build/llama-hip-compare \
  --target test-backend-hrx test-backend-ops llama-cli llama-completion llama-bench \
  -j$(nproc)
```

Correctness:

```bash
OPS=RMS_NORM,ADD,MUL,DIV,SCALE,CPY,CONT,SET_ROWS,GET_ROWS,MUL_MAT,FLASH_ATTN_EXT,CONCAT,SOFT_MAX,ARGSORT,ROPE,UNARY,GLU,SUM_ROWS,L2_NORM,CLAMP,SSM_CONV,GATED_DELTA_NET,MUL_MAT_ID

GGML_HIP_COMPARE_SUBMISSION=stream \
  $ROOT/build/llama-hip-compare/bin/test-backend-ops test -b HRX0 -o "$OPS"

GGML_HIP_COMPARE_SUBMISSION=graph \
  $ROOT/build/llama-hip-compare/bin/test-backend-ops test -b HRX0 -o "$OPS"
```

Decode benchmark:

```bash
GGML_HIP_COMPARE_SUBMISSION=stream \
  $ROOT/build/llama-hip-compare/bin/llama-bench \
  -m "$MODEL" -dev HRX0 -ngl 99 --no-host 1 -fa 1 \
  -p 0 -n 64 -b 512 -ub 512 -r 7

GGML_HIP_COMPARE_SUBMISSION=graph \
  $ROOT/build/llama-hip-compare/bin/llama-bench \
  -m "$MODEL" -dev HRX0 -ngl 99 --no-host 1 -fa 1 \
  -p 0 -n 64 -b 512 -ub 512 -r 7
```

Prefill sweep:

```bash
GGML_HIP_COMPARE_SUBMISSION=stream \
  $ROOT/build/llama-hip-compare/bin/llama-bench \
  -m "$MODEL" -dev HRX0 -ngl 99 --no-host 1 -fa 1 \
  -p 2,3,31,32,33,127,128,129,255,256,257,512,1023,1024,1025 \
  -n 0 -b 512 -ub 512 -r 5

GGML_HIP_COMPARE_SUBMISSION=graph \
  $ROOT/build/llama-hip-compare/bin/llama-bench \
  -m "$MODEL" -dev HRX0 -ngl 99 --no-host 1 -fa 1 \
  -p 2,3,31,32,33,127,128,129,255,256,257,512,1023,1024,1025 \
  -n 0 -b 512 -ub 512 -r 5
```

Current HRX comparison from the existing integration build:

```bash
export HRX_INSTALL=$ROOT/build/hrx-rocm713-install
export LD_LIBRARY_PATH=$ROOT/build/llama-hrx-integration/bin:$HRX_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}
export IREE_HAL_AMDGPU_LIBHSA_PATH=$ROCM_PATH/lib/libhsa-runtime64.so
unset GGML_HIP_COMPARE_SUBMISSION

$ROOT/build/llama-hrx-integration/bin/llama-bench \
  -m "$MODEL" -dev HRX0 -ngl 99 --no-host 1 -fa 1 \
  -p 0 -n 64 -b 512 -ub 512 -r 7

$ROOT/build/llama-hrx-integration/bin/llama-bench \
  -m "$MODEL" -dev HRX0 -ngl 99 --no-host 1 -fa 1 \
  -p 2,3,31,32,33,127,128,129,255,256,257,512,1023,1024,1025 \
  -n 0 -b 512 -ub 512 -r 5
```
