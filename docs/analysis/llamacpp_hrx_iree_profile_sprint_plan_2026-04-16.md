# llama.cpp HRX/IREE profiling checkpoint and next sprint plan

Date: 2026-04-16

This note supersedes the rocprof-centered plan from the earlier runtime overhead analysis. For the next sprint, use two complementary profiling lenses:

- Tracy for the system/runtime timeline from HRX and IREE.
- `iree-profile` for the dispatch-level HAL view.

`rocprof` is not the right default tool for this phase. Keep it available for targeted hardware-counter or GPU-kernel questions, but do not drive runtime overhead work from rocprof traces.

Near-term expectation from Ben: ATT, counters, and related low-level GPU data
are expected to land in `iree-profile` shortly. Treat any rocprof-based
counter/ATT workflow as temporary stopgap tooling and prefer extending the
IREE profile pipeline once those records are available.

Read this after:

- `docs/analysis/llamacpp_pyre_runtime_overhead_analysis.md`
- `docs/analysis/llamacpp_hrx_runtime_overhead_analysis_2026-04-15.md`

The older Pyre note diagnosed the first structural problem: the reference HSA path waited on the host around every command. The 2026-04-15 HRX rerun showed better kernels but still a large endpoint/runtime gap. The current toolchain lets us inspect that gap in runtime terms: Tracy gives the process/system timeline, and IREE profile files give command-buffer/export/dispatch detail.

## Current branches and build state

Relevant source branches:

```text
sources/hrx        users/awoloszyn/amdgpu
sources/iree       users/benvanik/amdgpu-wip
sources/llama.cpp  hrx_backend
```

Relevant builds:

```text
build/hrx-rocm713
build/hrx-rocm713-install
build/hrx-rocm713-tracy
build/hrx-rocm713-tracy-install
build/iree-rt
build/iree-tracy-tools
build/llama-hrx-rocm713
```

Model:

```text
models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf
```

Common environment for traced llama.cpp runs:

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
export ROCM_PATH="$ROOT/rocm"
export HRX_RUNTIME_INSTALL="$ROOT/build/hrx-rocm713-tracy-install"
export LLAMA_BUILD="$ROOT/build/llama-hrx-rocm713"
export MODEL="$ROOT/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
export LD_LIBRARY_PATH="$HRX_RUNTIME_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export GGML_HRX_KERNEL_PROVIDER=pure_hip
export IREE_TRACY_CAPTURE="$ROOT/build/iree-tracy-tools/tracy/iree-tracy-capture"
```

## Tracing build

HRX now has a Tracy-enabled build:

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
cmake -S sources/hrx -B build/hrx-rocm713-tracy \
  -DHRX_IREE_SOURCE_DIR="$ROOT/sources/iree" \
  -DHRX_ENABLE_TRACY=ON \
  -DIREE_TRACING_MODE=1 \
  -DHRX_BUILD_CTS=OFF -DHRX_BUILD_PASSTHROUGH=OFF \
  -DCMAKE_PREFIX_PATH="$ROOT/rocm" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja

cmake --build build/hrx-rocm713-tracy --target hrx hrx-info -j"$(nproc)"
cmake --install build/hrx-rocm713-tracy \
  --prefix build/hrx-rocm713-tracy-install
```

Use `IREE_TRACING_MODE=1` for the current system-lens build. Mode 2 enables Tracy allocation tracking, but the capture tool currently trips an allocator verification failure during llama startup:

```text
Instrumentation failure: Memory allocation event was reported for an address that is already tracked and not freed.
```

That is useful signal, but it blocks scripted full-run captures. For now, leave allocation cadence analysis to IREE profile metadata and explicit HRX/IREE instrumentation rather than Tracy allocation events.

The Tracy capture CLI is built separately:

```bash
cmake -S sources/iree -B build/iree-tracy-tools \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_DRIVER_VULKAN=OFF \
  -DIREE_BUILD_TRACY=ON \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_TRACING_PROVIDER=tracy \
  -DTRACY_NO_VERIFY=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja

cmake --build build/iree-tracy-tools --target iree-tracy-capture -j"$(nproc)"
```

## IREE profile support

HRX supports HAL profile files:

```bash
HRX_PROFILE_FILE=/tmp/run.ireeprof "$LLAMA_BUILD/bin/llama-bench" ...
build/iree-rt/tools/iree-profile summary /tmp/run.ireeprof
build/iree-rt/tools/iree-profile dispatch --format=jsonl /tmp/run.ireeprof
```

`HRX_PROFILE_MODE` defaults to `queue`. Supported values are:

```text
queue
dispatch
executable
all
```

Use `queue` by default on this AMDGPU branch. `dispatch` and `all` are exposed, but treat them as opt-in until profiling correctness is clean across broader tests.

## Fused smoke run

This command captures the temporary fused view:

```bash
ROOT=/srv/vm-shared/projects/pyre-workspace
OUT="$ROOT/build/hrx-tracy-fused-smoke"
mkdir -p "$OUT"

export ROCM_PATH="$ROOT/rocm"
export HRX_RUNTIME_INSTALL="$ROOT/build/hrx-rocm713-tracy-install"
export LLAMA_BUILD="$ROOT/build/llama-hrx-rocm713"
export MODEL="$ROOT/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
export LD_LIBRARY_PATH="$HRX_RUNTIME_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export GGML_HRX_KERNEL_PROVIDER=pure_hip
export HRX_PROFILE_FILE="$OUT/run.ireeprof"
export IREE_TRACY_CAPTURE="$ROOT/build/iree-tracy-tools/tracy/iree-tracy-capture"

"$ROOT/sources/iree/build_tools/tracing/iree_tracy_capture.py" \
  --output-dir "$OUT" \
  --name llama-p32n8 \
  -- "$LLAMA_BUILD/bin/llama-bench" \
    -m "$MODEL" -p 32 -n 8 -r 1 -o json --no-warmup -ngl 99 -dev HRX0 \
  > "$OUT/llama-bench.json"

"$ROOT/build/iree-rt/tools/iree-profile" summary "$OUT/run.ireeprof" \
  > "$OUT/iree-profile-summary.txt"
"$ROOT/build/iree-rt/tools/iree-profile" dispatch --format=jsonl "$OUT/run.ireeprof" \
  > "$OUT/iree-dispatch.jsonl"
```

Validation from the current smoke run:

```text
build/hrx-tracy-fused-smoke/20260416-174106-llama-p32n8.tracy  17M
build/hrx-tracy-fused-smoke/run.ireeprof                      1.1M
build/hrx-tracy-fused-smoke/iree-dispatch.jsonl                 22K
build/hrx-tracy-fused-smoke/llama-bench.json                   2.5K
```

Endpoint numbers from that single traced smoke:

| Shape | avg_ns | tok/s | KV cache | Flash attention |
| --- | ---: | ---: | --- | --- |
| `p32 n0` | 404,747,965 | 79.062 | F16/F16 | off |
| `p0 n8` | 96,844,400 | 82.607 | F16/F16 | off |

These are instrumentation-smoke numbers, not performance gates.

`iree-profile summary` from the same run:

```text
records: file=10 session_begin=1 chunks=8 session_end=1 unknown=0
chunks: devices=1 queues=1 executables=1 executable_exports=1 command_buffers=1 clock_correlations=2 dispatch_events=1 unknown=0 truncated=0
metadata_records: executables=158 executable_exports=862 command_buffers=934
dispatches=10552 valid=10552 invalid=0
dispatch_time_ns: min=1520.006 avg=20512.525 max=946083.867 total=216448164.704
```

## Current interpretation

1. The profiling stack is usable.

   HRX with IREE Tracy mode 1 can run llama-bench under `iree_tracy_capture.py` and simultaneously emit a parseable `.ireeprof` file.

2. The old reference-HSA diagnosis is no longer sufficient.

   The new AMDGPU runtime is structurally different. Runtime overhead now needs to be examined in terms of command-buffer cadence, allocation/fence cadence, host scheduler behavior, and dispatch mix.

3. The dispatch lens is already clean enough for triage.

   The profile file contains valid export, command-buffer, queue, and dispatch records. Use this for grouping by export, counting dispatches, and tracking total HAL dispatch time.

4. The system lens is now available but should stay mode 1 for scripted runs.

   Mode 2 allocation tracking currently fails Tracy capture verification during startup. Do not block runtime work on that; capture mode 1 timelines and use IREE profile files for dispatch-level detail.

5. The known profiling correctness caveat remains.

   `test-backend-hrx` under `HRX_PROFILE_FILE` has previously failed an argsort check while normal `test-backend-hrx` passed. Profiled llama-bench works, but do not treat profiled backend-test failures as benchmark evidence until that path is understood.

## Profile readout

Additional focused captures were taken under:

```text
build/hrx-profile-samples/decode-n16
build/hrx-profile-samples/prefill-p512-fa1
build/hrx-profile-samples/unprofiled-baseline
```

The first useful split is endpoint time versus IREE dispatch/queue time:

| Sample | Endpoint | Queue submissions | Dispatches | Queue span | Aggregate dispatch time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `p0 n16 fa0` traced | 201.843 ms | 16 | 18,464 | 114.794 ms | 159.293 ms | exactly 1,154 dispatches/token |
| `p0 n16 fa0` unprofiled | 189.278 ms | n/a | n/a | n/a | n/a | 84.541 tok/s, stable across 3 reps |
| `p512 n0 fa1` traced | 477.341 ms | 40 | 1,665 | 125.546 ms | 326.241 ms | matches cold first baseline sample |
| `p512 n0 fa1` unprofiled | 322.230 ms avg | n/a | n/a | n/a | n/a | samples were 476.450, 246.086, 244.156 ms |

Interpretation:

- Decode has a structural dispatch-count problem. `n16` produces 18,464 dispatches, or 1,154 per token. Even if each dispatch is small, the command and host envelope becomes a first-order term.
- Decode endpoint overhead over aggregate IREE dispatch time is about 42.6 ms for 16 tokens in the traced run, roughly 2.7 ms/token. Endpoint overhead over queue span is about 87.0 ms, roughly 5.4 ms/token.
- Prefill profiling is contaminated by cold-start effects unless the run is warmed. The traced `p512` sample matched the first unprofiled sample, while the next two unprofiled samples were about 245 ms. Future prefill captures need an explicit warmup or a repeated run where only the warm repetition is analyzed.
- IREE's aggregate dispatch time can exceed queue span because the current projection sums dispatch durations and does not provide a pure non-overlapped queue occupancy metric. Use queue span for device elapsed time and aggregate dispatch time for export-family weighting.

Top decode `n16` dispatch-time buckets:

| Export | Count | Total |
| --- | ---: | ---: |
| `hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32` | 1,120 | 28.989 ms |
| `hrx_mul_mat_vec_q5_k_wg128_f32` | 480 | 11.051 ms |
| `hrx_swiglu_f32` | 480 | 10.322 ms |
| `hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32` | 640 | 8.412 ms |
| `hrx_mul_mat_vec_q5_k_wg64_f32` | 16 | 8.190 ms |
| `hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32` | 640 | 8.028 ms |
| `hrx_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32` | 640 | 7.299 ms |
| `hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32` | 1,280 | 7.268 ms |

Top decode `n16` dispatch-count buckets:

| Export | Count | Avg |
| --- | ---: | ---: |
| `hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32` | 1,280 | 5.678 us |
| `hrx_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32` | 1,120 | 4.505 us |
| `hrx_sigmoid_f32` | 1,120 | 2.224 us |
| `hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32` | 1,120 | 25.883 us |
| `hrx_get_rows_f32_nr1` | 992 | 4.372 us |
| `hrx_scale_f32` | 960 | 3.077 us |
| `hrx_l2_norm_wg128_f32` | 960 | 3.921 us |
| `hrx_rms_norm_mul_f32` | 800 | 6.387 us |

This suggests the first runtime attack should be dispatch-count reduction and graph/provider fusion in decode, not single-kernel micro-optimization. The kernel-time buckets still matter, but there are many thousands of sub-10us dispatches where launch/command overhead is likely comparable to or larger than the useful work.

## Next sprint approach

### 1. Establish a repeatable measurement matrix

For every benchmark run, write a self-contained result directory with:

- unprofiled llama-bench JSON;
- Tracy `.tracy` from the mode 1 HRX/IREE build;
- IREE `.ireeprof`;
- `iree-profile summary`;
- `iree-profile dispatch --format=jsonl`;
- environment and branch metadata.

Minimum matrix:

| Regime | Shape | Purpose |
| --- | --- | --- |
| Prefill | `p512 n0 fa1 r3` | compare against prior optimization checkpoint |
| Decode short | `p0 n16 fa0 r1` | manageable timeline/profile inspection |
| Decode target | `p0 n64 fa0 r3` | stable endpoint comparison |
| Mixed smoke | `p32 n8 r1` | fast fused-profile sanity |

Do not use `/tmp` as the durable result location.

### 2. Add a small fused-profile runner

Add a script that creates one result directory and runs:

- unprofiled llama-bench;
- Tracy-captured llama-bench with `HRX_PROFILE_FILE`;
- `iree-profile summary`;
- `iree-profile dispatch --format=jsonl`;
- a compact metadata file with branches, build paths, model path, env vars, and command lines.

The script should fail clearly if the requested HRX install is not Tracy-enabled or if `iree-tracy-capture` is missing.

### 3. Add an IREE dispatch summarizer

Consume:

```bash
iree-profile dispatch --format=jsonl run.ireeprof
```

Emit:

- total dispatches;
- total dispatch time;
- top exports by total time;
- top exports by count;
- p50/p90/p99 when raw per-dispatch durations are available;
- command-buffer count and dispatches per command buffer;
- optional diff against a previous profile.

This is the main dispatch lens until IREE profile data and Tracy are fused upstream.

### 4. Correlate Tracy timeline with IREE profile data

For each result directory:

- compare llama-bench wall time with IREE aggregate dispatch time;
- identify large host-side gaps between dispatch groups in Tracy;
- map spikes in Tracy zones to command-buffer/export groups in the IREE JSONL;
- separate startup/model-load effects from steady-state decode.

The expected outcome is not exact time equality. The expected outcome is a repeatable explanation of where endpoint time goes.

### 5. Quantify allocation and command-buffer cadence

The current working hypothesis is:

```text
endpoint wall gap = dispatch/queue overhead
                  + allocation/fence cadence
                  + host scheduler behavior
                  + residual kernel/fusion imbalance
```

For decode, measure:

- command buffers per token;
- dispatches per token;
- host waits/fences per token;
- transient allocations per token;
- whether HRX/llama.cpp can hoist or reuse allocations across token steps.

The conservative `queue_alloca` wait added for transient-buffer correctness is still on the suspect list. It may be correct for bringup but too expensive at decode cadence.

### 6. Keep rocprof out of the default loop

Use rocprof only when a specific question requires GPU counters or a hardware timeline cross-check that is not yet available through `iree-profile`. Ben expects ATT/counter data to arrive in `iree-profile` soon, so do not build new long-lived rocprof infrastructure unless the gap is urgent. The default sprint loop should be:

```text
llama-bench JSON -> Tracy timeline -> IREE profile summary/dispatch JSONL
```

This keeps the investigation aligned with the runtime/HAL questions we need to answer before climbing toward SOTA.
