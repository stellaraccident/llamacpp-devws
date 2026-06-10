# llama.cpp HRX AMDGPU handoff - 2026-04-16

Current workspace: `/srv/vm-shared/projects/pyre-workspace`.

## Branches

| Repo | Path | Branch | HEAD | Notes |
| --- | --- | --- | --- | --- |
| HRX | `sources/hrx` | `users/awoloszyn/amdgpu` | `fca8f59cded2026bb7362bd7d89b62bc4c054fd3` | AMDGPU runtime branch |
| IREE | `sources/iree` | `users/benvanik/amdgpu-wip` | `eddd269435b7ce7168d33028fbb5cf368dd2492c` | direct AMDGPU HSACO branch |
| llama.cpp | `sources/llama.cpp` | `hrx_backend` | `3dd809890` | includes raw-HSACO HRX kernel support |

HRX now lives at `sources/hrx`, not `sources/pyre-runtime`.

The latest llama.cpp commit is `3dd809890 Support raw HSACO HRX kernels`. That commit makes the HRX kernels emit raw HSACO, removes the old explicit `FPIH` executable format from generated kernel metadata, and shrinks the fused RMS/ROPE/set_rows by-value constants block from 256 to 248 bytes.

## Paths

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
ROCM_PATH=$ROOT/rocm
HRX_BUILD=$ROOT/build/hrx-rocm713
HRX_INSTALL=$ROOT/build/hrx-rocm713-install
LLAMA_BUILD=$ROOT/build/llama-hrx-rocm713
MODEL=$ROOT/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf
```

`$ROOT/rocm` points at the ROCm 7.13 alpha install used for this setup.

## Model

Local model: `models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf`.

Local size is about 19 GiB.

Likely source:

- Hugging Face repo: <https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF>
- Direct file URL: <https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf>

Download command:

```bash
cd /srv/vm-shared/projects/pyre-workspace
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-Q4_K_L.gguf \
  --local-dir models
```

## Build HRX

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace

cmake -S "$ROOT/sources/hrx" -B "$ROOT/build/hrx-rocm713" \
  -DHRX_IREE_SOURCE_DIR="$ROOT/sources/iree" \
  -DCMAKE_PREFIX_PATH="$ROOT/rocm" \
  -DHRX_BUILD_CTS=OFF \
  -DHRX_BUILD_PASSTHROUGH=OFF \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -GNinja

cmake --build "$ROOT/build/hrx-rocm713" --target hrx hrx-info -j"$(nproc)"
cmake --install "$ROOT/build/hrx-rocm713" --prefix "$ROOT/build/hrx-rocm713-install"
```

Smoke test:

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
export ROCM_PATH="$ROOT/rocm"
export HRX_INSTALL="$ROOT/build/hrx-rocm713-install"
export LD_LIBRARY_PATH="$HRX_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"

"$ROOT/build/hrx-rocm713/hrx-info"
"$ROOT/build/hrx-rocm713/hrx-info" --device=gpu:0
```

Expected: AMD Radeon Pro W7900 GPU device, local-task CPU device, and `All tests PASSED` for `--device=gpu:0`.

## Build llama.cpp

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace

cmake -S "$ROOT/sources/llama.cpp" -B "$ROOT/build/llama-hrx-rocm713" \
  -DGGML_VULKAN=OFF \
  -DGGML_HIP=OFF \
  -DGGML_HRX=ON \
  -DGGML_HRX_ROCM_PATH="$ROOT/rocm" \
  -DGGML_HRX_AMDGPU_TARGET=gfx1100 \
  -DCMAKE_PREFIX_PATH="$ROOT/build/hrx-rocm713-install" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -GNinja

cmake --build "$ROOT/build/llama-hrx-rocm713" \
  --target llama-cli llama-bench test-backend-hrx export-graph-ops hrx-kernel-bench \
  -j"$(nproc)"
```

Smoke test:

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
export ROCM_PATH="$ROOT/rocm"
export HRX_INSTALL="$ROOT/build/hrx-rocm713-install"
export LLAMA_BUILD="$ROOT/build/llama-hrx-rocm713"
export LD_LIBRARY_PATH="$LLAMA_BUILD/bin:$HRX_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"

"$LLAMA_BUILD/bin/test-backend-hrx"
ctest --test-dir "$LLAMA_BUILD" -R '^test-backend-hrx$' --output-on-failure
```

`test-backend-hrx` exits 0 and normally prints no output.

## Chat

The wrapper still has the old `pyre` name, but it points at HRX paths:

```bash
cd /srv/vm-shared/projects/pyre-workspace
GGML_HRX_KERNEL_PROVIDER=pure_hip ./reproducers/chat_qwen_pyre.sh
```

Equivalent direct command:

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
export ROCM_PATH="$ROOT/rocm"
export HRX_INSTALL="$ROOT/build/hrx-rocm713-install"
export LLAMA_BUILD="$ROOT/build/llama-hrx-rocm713"
export MODEL="$ROOT/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$LLAMA_BUILD/bin:$HRX_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
export GGML_HRX_KERNEL_PROVIDER=pure_hip

"$LLAMA_BUILD/bin/llama-cli" \
  -m "$MODEL" \
  -p "You are a helpful assistant." \
  -ngl 99 \
  -dev HRX0 \
  -c 4096 \
  -n 512 \
  -t 1 \
  -tb 1 \
  -fa 1 \
  --reasoning off \
  --reasoning-budget 0 \
  -cnv
```

For deterministic short checks, append `--seed 1 --temp 0 -n 64`.

## Benchmarks

Use the same environment as the chat command, then:

```bash
OUT="$ROOT/build/bench-hrx-amdgpu-rerun-$(date +%Y%m%d)"
mkdir -p "$OUT"

"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -dev HRX0 -ngl 99 \
  -p 0 -n 64 -fa 0 \
  -ctk f16 -ctv f16 \
  -t 1 -tb 1 \
  -r 3 -o json \
  > "$OUT/hrx-decode-n64-r3.json"

"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -dev HRX0 -ngl 99 \
  -p 512 -n 0 -fa 1 \
  -ctk f16 -ctv f16 \
  -t 1 -tb 1 \
  -r 3 -o json \
  > "$OUT/hrx-prefill-p512-n0-r3.json"

"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -dev HRX0 -ngl 99 \
  -pg 512,0 -pg 0,1 -fa 1 \
  -ctk f16 -ctv f16 \
  -t 1 -tb 1 \
  -r 3 -o json \
  > "$OUT/hrx-prefill-p512-n1-r3.json"
```

Provider trace check:

```bash
GGML_HRX_TRACE_PROVIDER=1 "$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -dev HRX0 -ngl 99 \
  -p 0 -n 16 -fa 0 \
  -ctk f16 -ctv f16 \
  -t 1 -tb 1 \
  -r 1 -o json \
  > "$OUT/hrx-trace-n16.json" \
  2> "$OUT/hrx-trace-n16.log"

rg -n 'fallback|claim' "$OUT/hrx-trace-n16.log" | head
```

Kernel microbenchmarks:

```bash
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op rms_norm --ncols 3584 --nrows 1
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op mul_mat_vec_f16 --ncols 3584 --nrows 3584
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op mul_mat_vec_q4_k --ncols 3584 --nrows 3584
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op mul_mat_vec_q5_k --ncols 3584 --nrows 3584
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op mul_mat_vec_q6_k --ncols 3584 --nrows 3584
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op mul_mat_vec_q8_0 --ncols 3584 --nrows 3584
"$LLAMA_BUILD/bin/hrx-kernel-bench" --op mul_mat_id_q4_k --ncols 2048 --nrows 512 --n-experts 256 --n-ids 8 --n-tokens 1
```

## Current baseline

Latest local artifacts: `build/bench-hrx-amdgpu-rerun-20260415/`.

| File | Backend | Shape | Flash attention | KV cache | Average |
| --- | --- | --- | --- | --- | --- |
| `hrx-decode-n16-r3.json` | HRX | `p=0 n=16` | off | K=f16 V=f16 | 54.15 tok/s |
| `hrx-decode-n64-r3.json` | HRX | `p=0 n=64` | off | K=f16 V=f16 | 54.92 tok/s |
| `hrx-prefill-p512-n0-r3.json` | HRX | `p=512 n=0` | on | K=f16 V=f16 | 2183.47 tok/s |
| `hrx-prefill-p512-n1-r3.json` | HRX | `p=512 n=0` | on | K=f16 V=f16 | 2170.03 tok/s |
| `hrx-prefill-p512-n1-r3.json` | HRX | `p=0 n=1` | on | K=f16 V=f16 | 51.36 tok/s |

Comparison Vulkan results from the same artifact directory:

| File | Backend | Shape | Flash attention | KV cache | Average |
| --- | --- | --- | --- | --- | --- |
| `vulkan-decode-n64-r3.json` | Vulkan | `p=0 n=64` | off | K=f16 V=f16 | 108.66 tok/s |
| `vulkan-prefill-p512-n0-r3.json` | Vulkan | `p=512 n=0` | on | K=f16 V=f16 | 2292.72 tok/s |

Provider trace artifact `hrx-trace-n16.log` had zero fallback lines and about 18k provider claim lines, so the benchmarked path was using HRX provider kernels rather than silently falling back.

Kernel microbenchmark highlights from `hrx-kernel-bench-summary.txt`:

| Op | Shape | Median |
| --- | --- | --- |
| `rms_norm` | `ncols=3584 nrows=1` | 35.261 us |
| `mul_mat_vec_f16` | `3584 x 3584` | 59.812 us |
| `mul_mat_vec_q4_k` | `3584 x 3584` | 45.201 us |
| `mul_mat_vec_q5_k` | `3584 x 3584` | 40.071 us |
| `mul_mat_vec_q6_k` | `3584 x 3584` | 37.121 us |
| `mul_mat_vec_q8_0` | `3584 x 3584` | 40.241 us |
| `mul_mat_id_q4_k` | `ncols=2048 nrows=512 n_experts=256 n_ids=8 n_tokens=1` | 48.052 us |

## rocprof status

The ROCm profiler binary supplied by the user is `/srv/vm-shared/shared/rocm-7.13alpha/bin/rocprofv3`.

As of this handoff, `rocprofv3 --kernel-trace` is not reliable for the new HRX AMDGPU runtime. It has crashed both full llama.cpp model runs and a minimal `hrx-info --device=gpu:0` smoke run inside IREE AMDGPU host queue code. Treat JSON benchmark output and provider trace logs as the reliable checks until the rocprof interaction is fixed.

## Notes

- Current benchmark runs use f16 KV cache (`-ctk f16 -ctv f16`).
- `reproducers/chat_qwen_pyre.sh` still has the old name but is currently the useful HRX chat wrapper.
- HRX install consumed by llama.cpp is `build/hrx-rocm713-install`.
