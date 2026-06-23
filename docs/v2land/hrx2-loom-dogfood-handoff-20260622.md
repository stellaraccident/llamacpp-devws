# HRX2 Loom Dogfood Handoff - 2026-06-22

This branch is a reproducibility snapshot of the Phi-4-mini Q4_K_M
HRX2/Loom dogfood run on a gfx1100 machine. It is not a production HRX2 design.
The value in this branch is the checked-in Loom kernel corpus, the route
metadata that activates it, the measured kernel/process lessons, and the exact
run recipe that lets another person rebuild the same hacked stack and compare
against Vulkan.

The durable product direction is still a replacement HRX2/HRX3 integration.
Treat `ggml-hrx2.cpp` in this branch as a runnable shim that proves the kernels
and scheduling ideas can hit the model contract. The env-var maze, route
forcing, command-mode policy, and llama.cpp graph matching are preservation
scaffolding rather than the design to carry forward.

## Source Pair

This llama.cpp branch must be paired with `hrx-systems` main. The branch carries
the HRX2-side kernels, catalog metadata, route forcing, and trace hooks; the
Loom compiler/runtime dependency comes from `hrx-systems`.

The branch state at the time of this report was:

```text
hrx-systems dependency: main
llama.cpp base snapshot: 486ee68 [HRX] Snapshot fusion dogfood scratch state
```

The measured rows below were taken while the supporting compiler/runtime
changes were moving into `hrx-systems` main. If the timing changes after
rebasing either tree, rerun the correctness gates and all three timing rows
before making a new performance claim.

## What Is In This Branch

The useful artifacts live in these places:

```text
ggml/src/ggml-hrx2/kernels/
ggml/src/ggml-hrx2/catalog/
ggml/src/ggml-hrx2/catalog.json
ggml/src/ggml-hrx2/ggml-hrx2-catalog.cpp
ggml/src/ggml-hrx2/ggml-hrx2.cpp
tools/llama-bench/llama-bench.cpp
```

The kernel corpus contains the relevant Loom sources for the current Phi run:

- Quantized prompt and decode matmuls: `mul_mat_q4_k_f32.loom`,
  `mul_mat_q5_k_f32.loom`, `mul_mat_q6_k_f32.loom`, and the Vulkan-clone
  matvec sources.
- Decode fusions: packed q4/SwiGLU, q5 WQKV V-cache, q5 Q-scale-before-RoPE,
  q6 direct-scale decode/down/output variants, and the rejected-but-preserved
  softmax+KQV source.
- Attention and pointwise support: f16 batched/p021 matvecs, softmax variants,
  RoPE theta-scale routes, RMS/RMS_MUL routes, quantize, contiguous, and row
  update kernels.
- Route metadata that makes HRX2 select the tuned sources for the Phi shapes.

`ggml-hrx2-catalog.cpp` was also adjusted to print a reproducible
`loom-compile` command with the selected root symbol. That is a keeper
diagnostic improvement: when a provider is slow or wrong, the trace can hand a
compiler engineer a focused root-selected reproducer instead of a whole catalog.

`tools/llama-bench/llama-bench.cpp` contains phase trace markers for aligning
Vulkan/HRX2 runs. They are useful during this dogfood pass, but they are not a
public benchmark API.

## Required Environment

The setup that produced the reported rows was:

```text
GPU: gfx1100/RDNA3
ROCm/TheRock: a ROCm tree with HSA, AMDGPU, and Vulkan runtime support
LLVM: clang/clang++ suitable for both hrx-systems and llama.cpp CMake builds
model: microsoft Phi-4-mini-instruct Q4_K_M GGUF
```

The critical authoring build flags are `LOOM_TARGET_AMDGPU=ON`,
`LOOM_EMIT_AMDGPU=ON`, and `LOOM_EXECUTE_IREE_HAL=ON`. Missing any of those can
leave the build looking healthy while the HRX2 provider path cannot compile or
execute the AMDGPU Loom artifacts.

## Rebuild Recipe

The commands below assume separate checkouts for `hrx-systems` and this
llama.cpp branch. Replace the placeholder paths with checkout, build,
toolchain, and model locations.

```bash
export HRX_SYSTEMS=/path/to/hrx-systems
export LLAMA_SRC=/path/to/llama.cpp
export BUILD_ROOT=/path/to/build/root
export HRX_BUILD=$BUILD_ROOT/hrx-systems
export LLAMA_BUILD=$BUILD_ROOT/llama-hrx2
export ROCM_PATH=/path/to/rocm
export LLVM_ROOT=/path/to/llvm
export MODEL=/path/to/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf
```

Build the Loom tools used by the route-free loop:

```bash
$HRX_SYSTEMS/build_tools/bin/iree-bazel-build \
  //loom/src/loom/tools/loom-compile:loom-compile \
  //loom/src/loom/tools/iree-test-loom:iree-test-loom \
  //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom
```

Configure and build the `hrx-systems` HRX/Loom dependency:

```bash
cmake -S "$HRX_SYSTEMS" -B "$HRX_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$LLVM_ROOT/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_BUILD_BENCHMARKS=ON \
  -DIREE_ENABLE_ASSERTIONS=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_DRIVER_VULKAN=ON \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_DRIVER_NULL=ON \
  -DIREE_ROCM_PATH="$ROCM_PATH" \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned \
  -DLOOM_BUILD=ON \
  -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_EMIT_AMDGPU=ON \
  -DLOOM_EXECUTE_IREE_HAL=ON

cmake --build "$HRX_BUILD" \
  --target hrx loomc_shared loom-link loom-compile iree-test-loom iree-benchmark-loom \
  -j 16
```

Regenerate the source-side HRX2 catalog after every route or source metadata
edit. The runtime loads this source catalog when `GGML_HRX2_CATALOG_DIR`
points at the source tree, so stale `catalog.json` files can absolutely produce
stale performance claims.

```bash
python3 "$LLAMA_SRC/ggml/src/ggml-hrx2/tools/assemble_hrx2_catalog.py" \
  --catalog-dir "$LLAMA_SRC/ggml/src/ggml-hrx2/catalog" \
  --out "$LLAMA_SRC/ggml/src/ggml-hrx2/catalog.json"
```

Configure and build llama.cpp HRX2:

```bash
cmake -S "$LLAMA_SRC" -B "$LLAMA_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER="$LLVM_ROOT/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DGGML_HRX2=ON \
  -DGGML_HRX=OFF \
  -DGGML_VULKAN=ON \
  -DGGML_HIP=OFF \
  -DGGML_OPENMP=ON \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_F16C=OFF \
  -DGGML_FMA=OFF \
  -DGGML_HRX_ROCM_PATH="$ROCM_PATH" \
  -DROCM_PATH="$ROCM_PATH" \
  -Dhrx_DIR="$HRX_BUILD/libhrx/cmake/hrx" \
  -Dloomc_DIR="$HRX_BUILD/loom/binding/c/cmake/loomc" \
  -DGGML_HRX2_LOOM_LINK_EXECUTABLE="$HRX_BUILD/loom/src/loom/tools/loom-link/loom-link" \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON

cmake --build "$LLAMA_BUILD" \
  --target ggml-hrx2 llama-bench llama-completion llama-perplexity test-backend-ops \
  -j 16
```

If any `hrx-systems` HRX runtime file changed, rebuild the HRX dependency
before rebuilding llama.cpp. Rebuilding only `ggml-hrx2` can preserve a stale
static HRX/Loom dependency and send the investigation into nonsense.

## Blessed Runtime Environment

This is the safe recipe for the 2026-06-22 milestone. The barrier settings are
intentional. Disabling dispatch barriers was fast and wrong; it corrupted
decode and is only useful as a synchronization bug reproducer.

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export HRX2_VISIBLE_DEVICE_ORDINAL=0
export VULKAN_BENCH_DEVICE=Vulkan1
export GGML_HRX2_CATALOG_DIR=$LLAMA_SRC/ggml/src/ggml-hrx2

export GGML_HRX2_AUTO_COMMAND_MODE=1
export GGML_HRX2_PREFER_SOURCE=soft_max_f32_mask_n512_r24_16384_wg64_lanes8,soft_max_f32_mask_n768_r24_16384_wg1024,soft_max_f32_mask_n256_r1536_wg64_lanes4,vk_clone,mul_mat_q4_k_q8_1_x4_mmq64x32,mul_mat_q4_k_f32,mul_mat_q5_k_f32,mul_mat_q6_k_f32,rope_neox_f32_freq
export GGML_HRX2_DISABLE_Q4K_SWIGLU_FUSION=0
export GGML_HRX2_ENABLE_SOFTMAX_KQV_FUSION=0
export GGML_HRX2_ENABLE_Q5_WQKV_VCACHE_FUSION=1
export GGML_HRX2_ENABLE_Q5_WQKV_QSCALE_FUSION=1
export GGML_HRX2_DISPATCHES_PER_SUBMIT=18
export GGML_HRX2_MAX_MUL_MAT_BYTES_PER_SUBMIT=0
export HRX_SKIP_STREAM_DISPATCH_BARRIERS=0
export GGML_HRX2_DEPENDENCY_STREAM_BARRIERS=0
export HRX_AMDGPU_COMMAND_BUFFER_MODE=auto
export HRX_AMDGPU_EXPERIMENTAL_PM4_COMMAND_BUFFERS=1
export HRX_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE=host-async-copy
export GGML_HRX2_ASYNC_GRAPH_COMPUTE=1
export GGML_HRX2_SKIP_BACKEND_API_SYNCHRONIZE=1
export GGML_HRX2_DISABLE_ASYNC_GRAPH_EXIT_BARRIER=1
```

The oracle flags below are valid for HRX2/Vulkan parity investigation. They
force the HRX2 backend to stay on the Vulkan-equivalent route families and
remove extra buffers that hide route coverage problems. Use them for the dogfood
reproduction rows unless deliberately testing non-oracle routing.

```bash
export GGML_HRX2_ORACLE_VULKAN=1
export GGML_HRX2_FORCE_Q4_K=1
export GGML_HRX2_DISABLE_EXTRA_BUFTS=1
```

## Timing Commands

Use the same binary, model, device pinning, and environment for HRX2 and Vulkan.
The three core benchmark cases are:

```bash
export LLAMA_BENCH=$LLAMA_BUILD/bin/llama-bench

# Mixed: p512 prefill followed by 64 decode tokens.
$LLAMA_BENCH -m "$MODEL" -ngl 99 -b 512 -ub 512 -r 3 -o json -dev HRX20 -p 0 -n 0 -pg 512,64
$LLAMA_BENCH -m "$MODEL" -ngl 99 -b 512 -ub 512 -r 3 -o json -dev Vulkan1 -p 0 -n 0 -pg 512,64

# Pure prefill.
$LLAMA_BENCH -m "$MODEL" -ngl 99 -b 512 -ub 512 -r 3 -o json -dev HRX20 -p 512 -n 0
$LLAMA_BENCH -m "$MODEL" -ngl 99 -b 512 -ub 512 -r 3 -o json -dev Vulkan1 -p 512 -n 0

# Decode.
$LLAMA_BENCH -m "$MODEL" -ngl 99 -b 512 -ub 512 -r 3 -o json -dev HRX20 -p 0 -n 64
$LLAMA_BENCH -m "$MODEL" -ngl 99 -b 512 -ub 512 -r 3 -o json -dev Vulkan1 -p 0 -n 64
```

Keep HRX2 and Vulkan on the same physical GPU. Earlier p64 data was polluted by
ordinal mismatches and made good kernels look wildly bad. Record
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`,
`HRX2_VISIBLE_DEVICE_ORDINAL`, and `VULKAN_BENCH_DEVICE` with every run.

## Correctness Gates

The timing rows are only meaningful after a correctness gate passes. The smoke
gate used during this ratchet was deterministic completion plus a small
perplexity run.

The deterministic completion shape is:

```bash
$LLAMA_BUILD/bin/llama-completion \
  -m "$MODEL" \
  -ngl 99 \
  -b 512 \
  -ub 512 \
  -c 512 \
  -dev HRX20 \
  -p "The quick brown fox" \
  -n 32 \
  -s 42 \
  --temp 0 \
  --top-k 1 \
  --top-p 1 \
  --min-p 0 \
  --repeat-penalty 1 \
  --presence-penalty 0 \
  --frequency-penalty 0 \
  -no-cnv \
  --no-display-prompt \
  --no-warmup \
  --simple-io
```

Run the same command against `-dev Vulkan1` and compare output text. For route
work that only executes under oracle forcing, include the oracle flags above
and trace the selected route.

The PPL smoke used in the milestone reported `ppl=1.0011` for a 512-token,
one-chunk run. The exact smoke corpus was scratch data, so the branch-level
reproduction requirement is finite, stable PPL on the same corpus for HRX2 and
Vulkan, plus matching deterministic completion output for the fixed prompt.

## Reported Numbers

The strongest complete safe triplet from the 2026-06-22 milestone was:

| case | HRX2 avg ms | Vulkan1 avg ms | HRX2/Vulkan time | HRX2/Vulkan tok/s |
| --- | ---: | ---: | ---: | ---: |
| mixed p512/n64 | 468.607840 | 527.435855 | 0.888464 | 1.12554 |
| prefill p512/n0 | 100.826112 | 111.150167 | 0.907116 | 1.10220 |
| decode p1/n64 | 345.475134 | 396.888469 | 0.870459 | 1.14886 |

After the q4 subgroup-leader follow-up, the live scratch state improved mixed
and decode again:

| case | baseline HRX2 avg ms | q4 subgroup HRX2 avg ms | adjacent Vulkan avg ms | reading |
| --- | ---: | ---: | ---: | --- |
| mixed p512/n64 | 468.901 | 464.297 | 523.057 | about 4.6 ms better than the restored baseline |
| decode p1/n64 | 345.475 | 344.707 | 396.888 | flat/slightly positive |

The q4 subgroup addendum did not rerun a full clean prefill triplet. Until that
is repeated on a fresh branch handoff run, use the complete safe triplet as the
baseline claim and the subgroup row as a follow-on optimization point.

## Keeper Kernel Lessons

The biggest single accepted win was in `mul_mat_q4_k_f32.loom`: the q4 128x128
prompt route now explicitly unrolls the literal two-iteration inner WMMA loop.
This was not a generic "unroll more" lesson. RADV exposed the two iterations
statically and the Loom source already had literal bounds; the source was
failing to tell the compiler what the schedule knew.

The report movement for the large q4 provider was:

| metric | rolled inner loop | fixed two-iteration unroll |
| --- | ---: | ---: |
| static WMMA instructions | 16 | 32 |
| vector register count | 192 | 136 |
| vector pressure peak live units | 246 | 182 |
| resident subgroups per SIMD | 5 | 7 |
| materialized copies | 16 | 0 |
| coalesced copies | 0 | 32 |

Other accepted motifs:

- ROPE matches Vulkan by passing the host-computed `theta_scale` push-constant
  value into Loom instead of recomputing from `freq_base` inside the kernel.
- The p021 f16 matvec spells the exact two lane terms for the `ncols=128`,
  wave64 decode case and beats the Vulkan perf-log p021 row.
- RMS 3072-wide routes use vector width 16.
- The q4 Vulkan-clone matvec uses a leader-lane
  `kernel.subgroup.reduce<addf>` schedule for the final dot reduction, with
  bias and store sunk into the `%lane == 0` region. The mechanism matters: a
  direct subgroup rewrite failed until the source made leader-lane demand
  explicit.
- q5 WQKV V-cache and Q-scale-before-RoPE fusions were positive when they
  preserved the useful decomposition of the work.

## Negative Results To Preserve

These are as important as the wins because they explain why route-free checks
and model-level gates both matter.

- Softmax+KQV was a positive tiny route-free fusion for old KV256-like shapes,
  but live KV512/KV768 decode variants regressed. The fused schedule serialized
  V dot products inside the workgroup; removing dispatches was not enough.
  Keep `GGML_HRX2_ENABLE_SOFTMAX_KQV_FUSION=0` in the blessed recipe.
- Add+RMS and RoPE+scale fusions looked attractive in compile reports and
  reduced dispatch count, but they collapsed producer parallelism or lengthened
  SFU-heavy kernels. Benchmarks beat theory here.
- Dispatch-barrier skipping was fast and wrong. It silently corrupted decode.
- Some route-free benchmark rows with `batch-size=16` looked much faster than
  full-stack dispatch profile rows because they measured throughput-amortized
  batches, not one model dispatch. Use `batch-size=1` or retained dispatch
  profiles for per-dispatch parity claims.
- Route coverage bugs can dominate everything. One p64 800 ms disaster was CPU
  Q4_K fallback, not a Loom kernel datapoint.

## How To Continue From Here

The next agent should start from route-free checked Loom files, not from
llama.cpp routing. The productive loop was:

```text
extract exact Vulkan/ggml shape
  -> write one parameterized .loom source with check.case and check.benchmark rows
  -> compile with reports and target listing
  -> compare against RADV/ACO ISA where useful
  -> benchmark with dispatch_complete and batch-size=1 for per-dispatch claims
  -> integrate only after correctness and benchmark evidence survive
  -> trace the full llama.cpp route to prove it selected the intended provider
  -> run deterministic completion, PPL smoke, and same-session HRX2/Vulkan timing
```

When a compiler issue appears, preserve a standalone `.loom` reproducer with
absolute source paths, the root symbol, the target key, compile config, report,
manifest, target listing, and the observed wrong output or diagnostic. The
compiler side moved quickly when the handoff was a focused reproducer; it
stalled when the handoff was "llama.cpp is slow."

For HRX3, the real deliverables from this branch are:

- The `.loom` kernel bodies and the checked motifs above.
- The route catalog data needed to instantiate them for Phi Q4_K_M shapes.
- The evidence that Loom can match and beat Vulkan for this model on one
  architecture when the kernel schedule and runtime submission path are
  controlled.
- The negative results showing that compile-report hill climbing and blind
  fusion can mislead agents unless every candidate has correctness and model
  timing gates.
