# llama.cpp HRX runtime overhead analysis

Date: 2026-04-15

This is the current HRX/AMDGPU-runtime rerun of
`llamacpp_pyre_runtime_overhead_analysis.md`. The old note is still useful as a
baseline, but the source checkout has been renamed to HRX and now uses the new
AMDGPU runtime instead of the earlier HSA driver path.

## Test setup

Model:

```bash
models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf
```

Builds:

```bash
build/hrx-rocm713-install
build/llama-hrx-rocm713/bin/llama-bench
build/llama-vulkan/bin/llama-bench
rocm/ -> /srv/vm-shared/shared/rocm-7.13alpha
```

Common HRX environment:

```bash
export ROCM_PATH=/srv/vm-shared/projects/pyre-workspace/rocm
export HRX_RUNTIME_INSTALL=/srv/vm-shared/projects/pyre-workspace/build/hrx-rocm713-install
export LLAMA_BUILD=/srv/vm-shared/projects/pyre-workspace/build/llama-hrx-rocm713
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$HRX_RUNTIME_INSTALL/lib64:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export GGML_HRX_KERNEL_PROVIDER=pure_hip
export GGML_HRX_ENABLE_ARGSORT=1
```

Decode command shape:

```bash
$LLAMA_BUILD/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf \
  -p 0 -n 64 -b 512 -ub 512 -fa 0 -r 3 -o json --no-warmup \
  -ngl 99 -dev HRX0
```

## Summary

HRX endpoint decode has improved over the old Pyre run, but the same class of
runtime overhead remains. Kernel quality is now in the right range for this
optimized decode path: rocprof sees about `9.63 ms/token` of kernel body time on
`n_gen=64`, implying a kernel-only floor of about `104 tok/s`. The unprofiled HRX
endpoint only reaches `38.23 tok/s`, so roughly two thirds of decode wall time is
still outside the kernel bodies.

The current Vulkan baseline on the same model and shape is `108.57 tok/s`. That
is close to the HRX kernel-only floor, which reinforces the earlier conclusion:
the remaining gap is primarily runtime scheduling/dispatch behavior, not the
optimized HIP kernel bodies.

The steady decode HSA trace still shows one host wait per kernel-scale operation:
`16,425 hsa_signal_wait_scacquire` calls in the sliced `n_gen=16` trace, totaling
`445.0 ms` with a `27.1 us` average. D2D copy time is negligible. Alloc/free
churn is visible but secondary.

## Endpoint measurements

Unprofiled `llama-bench`, `-r 3`, no warmup:

| Backend | Shape | avg_ms | tok/s | samples tok/s |
| --- | --- | ---: | ---: | --- |
| HRX | `p=0 n=16 fa=0` | 431.087 | 37.117 | 36.838, 37.231, 37.281 |
| HRX | `p=0 n=64 fa=0` | 1673.914 | 38.234 | 38.274, 38.235, 38.193 |
| Vulkan | `p=0 n=16 fa=0` | 154.829 | 103.554 | 97.034, 106.192, 107.437 |
| Vulkan | `p=0 n=64 fa=0` | 589.560 | 108.569 | 106.890, 109.493, 109.323 |
| HRX | `p=512 n=0 fa=1` | 242.724 | 2109.801 | 2078.380, 2149.350, 2101.670 |
| Vulkan | `p=512 n=0 fa=1` | 223.663 | 2292.817 | 2184.060, 2408.520, 2285.870 |

Compared with the 2026-04-09 Pyre note, HRX `n_gen=64` improved from
`30.68 tok/s` to `38.23 tok/s`. The old Vulkan decode point was about
`106 tok/s`; the current Vulkan rerun is `108.57 tok/s`.

## Kernel-only profile

`rocprofv3 --kernel-trace` on HRX `p=0 n=64`, one repetition:

```text
70,076 kernel dispatches
616,604.157 us total kernel time
9.634 ms/token
103.79 tok/s kernel-only floor
```

Top kernel totals:

```text
4480 calls  74081.664 us  avg=16.536 us  hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32
2560 calls  62537.899 us  avg=24.429 us  hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32
1920 calls  45863.332 us  avg=23.887 us  hrx_mul_mat_vec_q5_k_wg128_f32
2560 calls  45120.955 us  avg=17.625 us  hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32
5120 calls  33133.739 us  avg= 6.471 us  hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32
  64 calls  32483.225 us  avg=507.550 us  hrx_mul_mat_vec_q5_k_wg64_f32
3968 calls  27795.482 us  avg= 7.005 us  hrx_get_rows_f32_nr1
```

The old Pyre kernel-only number was about `13.93 ms/token`. So the kernel side
is materially better, but endpoint decode only moved from `30.68 tok/s` to
`38.23 tok/s` because runtime overhead remains dominant.

## Runtime profile

`rocprofv3 --kernel-trace --hsa-trace --memory-copy-trace
--memory-allocation-trace` on HRX `p=0 n=16`, sliced from the first steady
`993280 B` copy:

```text
16,410 kernel dispatches       147,637.739 us   9.227 ms/token
16,425 hsa_signal_wait calls   445,000.392 us  27.093 us/call
16 D2D copies                      624.888 us   0.039 ms/token
125 alloc/free records          24,703.981 us   1.544 ms/token
```

The API/region totals overlap with kernel execution and should not be summed as
wall time. They are useful for attribution: the main fixed cost is still the
large number of host waits around small kernel dispatches. Allocation churn is
worth addressing, but it is not the first-order decode gap.

## Provider coverage

`GGML_HRX_TRACE_PROVIDERS=1` on HRX `p=0 n=16` reported:

```text
claim_lines    18464
fallback_lines 0
```

The hottest claimed shapes are the optimized decode kernels:

```text
1280 RMS_NORM_MUL pure_hip_f32
1280 MUL_MAT      pure_hip_f32_batched_cols1_ne2_1_k2048_wg32
1120 MUL_MAT      pure_hip_bf16_rows4_k2048_cols1_lds_wg256
1120 SIGMOID      pure_hip_f32
1120 MUL_MAT      pure_hip_q6_K_rows2_cols1_wg32
 992 GET_ROWS     pure_hip_f32_nr1
```

So this is not a fallback-coverage regression.

## Artifacts

```text
build/bench-hrx-runtime-overhead/hrx-decode-n16-r3.json
build/bench-hrx-runtime-overhead/hrx-decode-n64-r3.json
build/bench-hrx-runtime-overhead/hrx-prefill-p512-r3.json
build/bench-hrx-runtime-overhead/vulkan-decode-n16-r3.json
build/bench-hrx-runtime-overhead/vulkan-decode-n64-r3.json
build/bench-hrx-runtime-overhead/vulkan-prefill-p512-r3.json
build/bench-hrx-runtime-overhead/hrx-trace-n16.log
build/bench-hrx-runtime-overhead/hrx-kernel-n16-summary.txt
build/bench-hrx-runtime-overhead/hrx-kernel-n64-summary.txt
build/bench-hrx-runtime-overhead/hrx-sys-n16-since-993280-summary.txt
build/bench-hrx-runtime-overhead/rocprof-kernel-n16/hrx-kernel-n16_results.db
build/bench-hrx-runtime-overhead/rocprof-kernel-n64/hrx-kernel-n64_results.db
build/bench-hrx-runtime-overhead/rocprof-sys-n16/hrx-sys-n16_results.db
```

