---
name: hrx-llamacpp-kernel-optimization
description: Skill-like reference for future agents working on llama.cpp ggml-hrx optimizations. Use for provider selection, HRX/Vulkan profiling, rocprof scoreboards, Qwen correctness gates, graph export tests, long-generation loop guards, kernel catalog edits, and safe default policy decisions.
metadata:
  short-description: llama.cpp ggml-hrx optimization workflow
---

# llama.cpp HRX Optimization Skill

This document is written for future agents working specifically in
`sources/llama.cpp` on `ggml-hrx`.

Use it with:

- `docs/kernel-skill/kernel_optimization_guide.md`
- `docs/kernel-skill/amd_rdna3_wavefront_isa_gotchas.md`
- historical analysis logs under `docs/analysis/` may predate the HRX rename;
  treat them as background records, not command references.

## Index

- [A. Repo and Build Rules](#a-repo-and-build-rules)
- [B. Provider Selection](#b-provider-selection)
- [C. Benchmark Shapes](#c-benchmark-shapes)
- [D. Profiling Workflow](#d-profiling-workflow)
- [E. Correctness Workflow](#e-correctness-workflow)
- [F. Kernel Catalog Workflow](#f-kernel-catalog-workflow)
- [G. Env Gates and Policy](#g-env-gates-and-policy)
- [H. Vulkan Comparison](#h-vulkan-comparison)
- [I. Common Failure Modes](#i-common-failure-modes)
- [J. Current Baseline](#j-current-baseline)
- [K. Search Map](#k-search-map)

## A. Repo and Build Rules

Work in:

```text
sources/llama.cpp
```

Do not commit or branch in workspace root unless the human explicitly changes
the rule. The workspace root is human-managed.

Normal build:

```bash
cmake --build build/llama-hrx-rocm713 \
  --target llama-bench llama-cli test-backend-hrx test-backend-ops export-graph-ops hrx-kernel-bench \
  -j"$(nproc)"
```

For integration work in `sources/llama.cpp-integrate`, use the matching
integration build directory instead:

```bash
export LLAMA_BUILD="$PWD/build/llama-hrx-integration"
cmake --build "$LLAMA_BUILD" \
  --target llama-bench llama-cli test-backend-hrx test-backend-ops \
  -j"$(nproc)"
```

Useful runtime environment:

```bash
export ROCM_PATH="$PWD/rocm"
export GGML_HRX_ROCM_PATH="$ROCM_PATH"
export HRX_RUNTIME_INSTALL="$PWD/build/hrx-rocm713-install"
export LLAMA_BUILD="$PWD/build/llama-hrx-rocm713"
export MODEL="$PWD/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$HRX_RUNTIME_INSTALL/lib64:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export GGML_HRX_KERNEL_PROVIDER=pure_hip
```

Check loaded libraries:

```bash
ldd "$LLAMA_BUILD/bin/llama-bench" | rg 'libhsa|librocprofiler-register|libhrx|therock'
```

If `therock` appears in the active profiling build, stop and fix the
environment.

### Tail-Generalization Discipline

When broadening a provider that was originally profiled at one prompt length,
preserve the old full-tile path and add tail handling only on actual edge
tiles. A guard helper that looks equivalent can still perturb codegen or expose
latent memory bugs on the full p512 route. This happened with the BF16 WMMA
prompt kernel: routing full 16x16 tiles through guarded helpers caused later
FA checks to fail intermittently, while using the original unguarded helpers for
full tiles and guarded helpers only for tails restored stability.

Before landing tail generalization:

- add kernel-level boundary tests for exact, odd, and one-past-tile shapes;
- run `test-backend-hrx` repeatedly after the boundary tests, not only once;
- run `test-backend-ops -b HRX0 -o MUL_MAT` and relevant op suites;
- compare p2, p3, p31, p32, p33 and p511, p512, p513 full-model prefill.

Do not accept a tail kernel that only passes its own sampled output but corrupts
a later test. Treat later FA/RMS/MUL failures after a new kernel test as evidence
of memory corruption until proven otherwise.

When a hot kernel needs tail support, prefer a full-tile/edge-tile split over
moving every access through guarded helpers. The full-tile path should preserve
the original unguarded access pattern and codegen; only the actual edge tiles
should pay bounds checks or zero-fill. On April 19 this pattern recovered a
large p512 prefill bucket for `mul_mat_vec_f32_batched_rows2_cols8` while still
passing p2, odd-size, and one-past-tile full-model prefill tests.

## B. Provider Selection

Primary files:

```text
sources/llama.cpp/ggml/src/ggml-hrx/ggml-hrx.cpp
sources/llama.cpp/ggml/src/ggml-hrx/kernels/
sources/llama.cpp/ggml/src/ggml-hrx/kernels/generate_hrx_kernels.py
sources/llama.cpp/ggml/src/ggml-hrx/kernels/hrx_kernel_catalog.h
```

Provider trace:

```bash
GGML_HRX_TRACE_PROVIDERS=1 "$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 0 -n 8 -b 512 -ub 512 -fa 0 -r 1 \
  -o json --no-warmup -ngl 99 -dev HRX0 \
  > build/hrx-trace.json 2> build/hrx-trace.log
```

Summarize:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-trace-summary.py summarize \
  --hrx-log build/hrx-trace.log --top 50
```

Provider trace tells you:

- which op/provider pair was selected;
- live shapes;
- fallbacks or support-query misses;
- whether a gate is actually active.

Provider trace does **not** tell you device time and should not be used for
tok/s conclusions.

Do not compare raw dispatch counts from different profilers without normalizing
the measurement window. HRX `iree-profile` captures can include warmup plus the
measured benchmark run, while the Vulkan perf logger reports only its logger
scope. If one side appears to have exactly double the route count, first align
warmup/repetition windows before concluding that provider selection changed.

### Provider naming conventions

Use names that encode the key schedule:

```text
pure_hip_q6_K_rows2_cols1_wg32
pure_hip_bf16_rows4_k2048_cols1_lds_wg256
pure_hip_q4_K_q8_1_x4_mmq64x64_wg64
pure_hip_f32_k_f16_v_f16_prefill_gfx11_direct
```

For shape-exact providers, gate narrowly in `ggml-hrx.cpp`. A correct provider
with a broad gate can regress decode or prompt.

### Fusion Selection Discipline

Pick fusion candidates from data movement and locality, not just graph
adjacency. Indexing loads/stores (`GET_ROWS`, `SET_ROWS`, narrow gathers, state
updates) are usually best fused into the closest producer/consumer hero op that
can naturally own the augmented access pattern. Standalone fusion of a small
indexing op with another small elementwise op often only reshuffles dispatch
overhead and can miss the larger benefit.

For each fusion candidate, write down:

- which intermediate tensor write/read is removed;
- approximate bytes avoided per token/prompt;
- whether the hero op can absorb the addressing without hurting occupancy or
  coalescing;
- which correctness gate covers the affected hidden state;
- whether a new provider should be opt-in until measured.

Example: `SCALE -> GET_ROWS -> SSM_CONV_UPDATE` on recurrent state is a
consumer-side gather candidate, because `SSM_CONV_UPDATE` can load the selected
state row directly. The experimental provider is gated by
`GGML_HRX_ENABLE_SSM_STATE_GATHER_FUSION`; leave it opt-in until the data
movement savings beat the extra addressing cost. The default-on optimization is
only to skip zero-element `SCALE` submissions, which is dispatch cleanup rather
than semantic fusion.

## C. Benchmark Shapes

Use these as the standard comparison unless a task says otherwise.

Prefill:

```bash
"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 512 -n 1 -b 512 -ub 512 -fa 1 -r 3 \
  -o json --no-warmup -ngl 99 -dev HRX0
```

Decode:

```bash
"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 0 -n 64 -b 512 -ub 512 -fa 0 -r 3 \
  -o json --no-warmup -ngl 99 -dev HRX0
```

Vulkan:

```bash
build/llama-vulkan/bin/llama-bench \
  -m "$MODEL" -p 512 -n 1 -b 512 -ub 512 -fa 1 -r 3 \
  -o json --no-warmup -ngl 99 -dev Vulkan0

build/llama-vulkan/bin/llama-bench \
  -m "$MODEL" -p 0 -n 64 -b 512 -ub 512 -fa 0 -r 3 \
  -o json --no-warmup -ngl 99 -dev Vulkan0
```

Interactive HRX smoke:

```bash
reproducers/chat_qwen_hrx.sh -st --seed 1 --temp 0
```

If CPU usage spikes, first run provider trace. If there are no real fallbacks,
check runtime/host behavior and thread count before assuming a CPU op.

### HRX transfer rule

Do not use `hrx_synchronous_h2d` or `hrx_synchronous_d2h` in llama.cpp backend
runtime paths. Those APIs are bringup/test conveniences and create hidden
allocate/wait behavior that can dominate decode.

Use stream-ordered staging instead:

- H2D: copy host bytes into a mapped host-local/device-visible staging buffer,
  then enqueue `hrx_stream_copy_buffer` to the device buffer on the backend
  stream.
- D2H: enqueue `hrx_stream_copy_buffer` from the device buffer to mapped
  staging, then `hrx_stream_synchronize` only at the API boundary where the
  host must observe the bytes.
- Reuse staging allocations across decode steps; reset/reclaim only after the
  stream has synchronized. The current arena size knob is
  `GGML_HRX_STAGING_ARENA_SIZE`.

### HRX graph submission rule

The HRX backend should partially submit command buffers while llama.cpp is still
walking the graph. This mirrors the Vulkan backend's hard-fought heuristic and
overlaps host command-buffer construction with GPU execution during decode.

Default policy:

- After each non-forced dispatch, enqueue an async `hrx_stream_flush` when either
  100 graph nodes have been submitted or the graph has accumulated enough
  matmul input bytes.
- Seed the matmul byte threshold from the previous graph:
  `min(100MB, last_total_mul_mat_bytes / 40)`, doubling it for the first three
  submits in the next graph.
- Do not synchronize on these partial submits, and do not reset staging storage
  until the normal graph synchronization point.

Diagnostic knobs:

```text
GGML_HRX_DISABLE_FEATHER_SUBMIT=1
GGML_HRX_FEATHER_NODES_PER_SUBMIT=100
GGML_HRX_FEATHER_MAX_MUL_MAT_BYTES_PER_SUBMIT=100000000
```

April 17 checkpoint: on Qwen decode `-p 0 -n 64 -r 3`, feathered submits moved
HRX from roughly 84 tok/s to 108 tok/s with f16/f16 KV cache.

## D. Profiling Workflow

For runtime overhead, use the temporary fused Tracy plus `iree-profile` flow.
Tracy is the system lens; `iree-profile` is the dispatch lens. Keep rocprof for
kernel bucket ranking, ATT setup, and targeted hardware-counter questions.
Do not run benchmark or profiling jobs in parallel when comparing candidates.
Several W7900/Qwen runs showed enough noise that consequential decisions should
be based on repeated same-binary measurements.

### IREE HAL profile files

Use HRX's built-in IREE profile sink when investigating runtime/queue overhead:

```bash
HRX_PROFILE_FILE=build/llama-bench-p32n8.ireeprof \
  "$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 32 -n 8 -b 512 -ub 512 -r 1 \
  -o json --no-warmup -ngl 99 -dev HRX0

build/iree-rt/tools/iree-profile summary build/llama-bench-p32n8.ireeprof
```

Default `HRX_PROFILE_MODE` is `queue`, which produced complete llama-bench
profiles on the April 16 AMDGPU branch. `HRX_PROFILE_MODE=dispatch` and `all`
exercise newer counter paths; verify correctness before using them for numbers.
At this checkpoint, `test-backend-hrx` can trip an argsort assertion under HRX
profiling even though normal `test-backend-hrx` and profiled `llama-bench` pass.

### Step 1: provider trace

Run HRX provider trace and check:

```bash
rg -c 'fallback' build/hrx-trace.log
rg 'claim ' build/hrx-trace.log | sort | uniq -c | sort -nr | head -50
```

No-FA decode should generally have no `CPY`, `CONT`, or `CONCAT` provider
claims in the steady path. If they appear, explain or fuse before kernel tuning.

### Step 2: rocprof HRX kernel/device time

Use this for kernel bucket ranking, not as the default runtime-overhead lens.

Prefill:

```bash
OUT_DIR=build/rocprof-hrx-prefill-candidate OUT_FILE=hrx-p512 \
PROMPT=512 GEN=0 FA=1 REPETITIONS=1 \
reproducers/rocprof_qwen_hrx_prefill.sh
```

Decode:

```bash
OUT_DIR=build/rocprof-hrx-decode-candidate OUT_FILE=hrx-decode-n64 \
GEN=64 REPETITIONS=1 \
reproducers/rocprof_qwen_hrx_decode.sh
```

Family scoreboard:

```bash
reproducers/hrx_rocprof_scoreboard.py \
  build/rocprof-hrx-prefill-candidate/hrx-p512_results.db \
  --baseline build/rocprof-hrx-prefill-baseline/hrx-p512_results.db \
  --top 20 --top-kernels 20
```

Raw detail:

```bash
sources/llama.cpp/tools/hrx-epic2/rocprof-rocpd-summary.py \
  build/rocprof-hrx-decode-candidate/hrx-decode-n64_results.db \
  --top 40
```

Decode HSA/copy traces include setup copies. Do not compare raw copy totals
unless you filter to the active decode window.

### Step 2b: IREE dispatch bucket profile

For current HRX integration work, the quickest full-model kernel-route sanity
check is a short `iree-profile` dispatch run. This is not a replacement for
rocprof/ATT when studying instruction stalls, but it is the best first pass for
answering "did this route actually reduce the model bucket?"

```bash
OUT=build/hrx-profile-decode-current
mkdir -p "$OUT"

HRX_PROFILE_FILE="$OUT/run.ireeprof" HRX_PROFILE_MODE=all \
  "$LLAMA_BUILD/bin/llama-bench" \
    -m "$MODEL" -p 0 -n 16 -b 512 -ub 512 -fa 1 -r 1 \
    -o json -ngl 999 -dev HRX0 \
  > "$OUT/llama-bench.json" 2> "$OUT/stderr.log"

build/iree-rt/tools/iree-profile dispatch --format=jsonl \
  --dispatch_events "$OUT/run.ireeprof" > "$OUT/events_detail.jsonl"

jq -r 'select(.type=="dispatch_event") |
  [.duration_ns, .key, (.workgroup_count|tostring), (.workgroup_size|tostring)] |
  @tsv' "$OUT/events_detail.jsonl" |
awk -F'\t' '
  { key=$2" wg="$3" sz="$4; total[key]+=$1; count[key]++ }
  END {
    for (k in total) {
      printf "%9.3f ms %5d %9.3f us %s\n",
        total[k]/1e6, count[k], total[k]/count[k]/1000, k
    }
  }' | sort -nr | head -60
```

Use this profile before and after any candidate that came from a native HIP
harness. If the native harness wins but this profile shows the active bucket
regressed, reject or keep investigating before changing the default route.

### Step 3: Vulkan perf logger

```bash
GGML_VK_PERF_LOGGER=1 build/llama-vulkan/bin/llama-bench \
  -m "$MODEL" -p 512 -n 0 -b 512 -ub 512 -fa 1 -r 1 \
  -o json --no-warmup -ngl 99 -dev Vulkan0 \
  > build/vulkan-p512.json 2> build/vulkan-p512.log
```

Use `hrx-trace-summary.py` to compare fusion labels and counts.

### Step 4: ISA/resource inspection

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-kernel-isa-summary.py \
  --kernel 'q5_k|q6_k|q4_k|bf16|gated_delta|flash_attn|topk' \
  --out-dir build/isa-candidate \
  --json build/isa-candidate/summary.json
```

Record:

- wavefront size;
- VGPR/SGPR and spills;
- LDS;
- `v_dot`, `v_wmma`, `v_mfma`;
- `s_barrier`, `ds_*`, `global_load`.

If CMake has per-source compile flags, inspect the built HSACO directly.

### Step 5: ATT/thread trace

Use only after rocprof identifies a hot kernel and source-level hypotheses are
exhausted. Filter to one kernel with `--kernel-include-regex`.

ATT can hang on this ROCm alpha. Smaller captures are safer:

- `--att-target-cu 0`
- `--att-serialize-all 1`
- one-token decode or one p512 prompt with one kernel regex

Summarize:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-att-summary.py \
  build/rocprof-att-candidate/stats_ui_output_agent_*_dispatch_*.csv \
  --top 40
```

## E. Correctness Workflow

### Fast focused loop

Generate graph op files if missing:

```bash
CHECK_CHAT=0 CHECK_LOOP=0 CHECK_FA=1 CHECK_DIRECT_FA=1 \
  reproducers/qwen_hrx_correctness_gate.sh
```

Then run focused tests:

```bash
FOCUS=q4 RESULTS_DIR=build/hrx-inner-loop-q4-candidate \
  reproducers/qwen_hrx_inner_loop.sh

FOCUS=q6 RESULTS_DIR=build/hrx-inner-loop-q6-candidate \
  reproducers/qwen_hrx_inner_loop.sh

FOCUS=gdn RESULTS_DIR=build/hrx-inner-loop-gdn-candidate \
  reproducers/qwen_hrx_inner_loop.sh

FOCUS=fa RESULTS_DIR=build/hrx-inner-loop-fa-candidate \
  reproducers/qwen_hrx_inner_loop.sh
```

For candidate env vars:

```bash
FOCUS=prompt CANDIDATE_ENV='GGML_HRX_ENABLE_EXPERIMENT=1' \
  RESULTS_DIR=build/hrx-inner-loop-experiment \
  reproducers/qwen_hrx_inner_loop.sh
```

### Full milestone gate

```bash
RESULTS_DIR=build/hrx-correctness-qwen-candidate \
CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1 \
LOOP_SEEDS=1,5 LOOP_PREDICT=384 \
reproducers/qwen_hrx_correctness_gate.sh
```

Run this before default-on promotion.

### Exact vs approximate

Conservative exact checks run with:

```bash
GGML_HRX_DISABLE_FAST_APPROX_PROMPT=1
```

Approximate prompt routes may fail CPU exactness if they intentionally match
Vulkan-style GPU approximations, such as Q8_1 RHS packing or BF16 WMMA. They
still need:

- focused route selection trace;
- full Qwen gate;
- loop guard;
- rollback env var;
- no decode regression.

### Long-generation loop guard

```bash
reproducers/qwen_loop_guard.py \
  --backend hrx \
  --seeds 1,5 \
  --runs 1 \
  --predict 384 \
  --context 4096 \
  --out-dir build/hrx-loop-check/candidate
```

Use for:

- prompt routing;
- ARGSORT/TOPK;
- approximate prompt matmuls;
- FA accumulator/layout changes;
- any change that affects hidden state before decode.

## F. Kernel Catalog Workflow

Kernel sources live in:

```text
sources/llama.cpp/ggml/src/ggml-hrx/kernels/
```

Rules:

- Add a new `.hip.cpp` for a meaningfully different schedule.
- For device-family variants, prefer one `.hip.cpp` selector with `.inc` files
  for generic/family implementations.
- Put architectural comments in family `.inc` files, not benchmark tok/s.
- Add the provider to `generate_hrx_kernels.py` / catalog machinery.
- Add provider selection and env gate in `ggml-hrx.cpp`.
- Add focused test coverage in `test-backend-hrx` or `test-backend-ops`
  export path before measuring.
- Keep prompt-only providers out of decode by shape gate.
- Keep decode exact-shape providers out of prompt by shape gate.

When changing templates or device compile flags, clear:

```bash
rm -rf "$HRX_CACHE_DIR/kernels"
```

## G. Env Gates and Policy

Search current knobs:

```bash
rg 'GGML_HRX_(ENABLE|DISABLE|FORCE)' sources/llama.cpp/ggml/src/ggml-hrx
```

Policy:

- Experimental route: `GGML_HRX_ENABLE_*`.
- Default-on approximate or risky route: `GGML_HRX_DISABLE_*`.
- Global conservative prompt rollback:
  `GGML_HRX_DISABLE_FAST_APPROX_PROMPT=1`.
- Diagnostic kill switches are acceptable, but do not use them as performance
  fixes.

Knobs that matter historically:

```text
GGML_HRX_DISABLE_FAST_APPROX_PROMPT
GGML_HRX_DISABLE_F16_PREFILL_FA_GFX11_DIRECT
GGML_HRX_DISABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT
GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ64_PROMPT
GGML_HRX_DISABLE_BF16_WMMA16_PROMPT
GGML_HRX_DISABLE_BF16_SWIGLU_WMMA16_PROMPT
GGML_HRX_ENABLE_ARGSORT
GGML_HRX_BARRIER_EACH_DISPATCH
GGML_HRX_DISABLE_SIGMOID_MUL_ADD_ADD_FUSION
GGML_HRX_DISABLE_L2_NORM_PAIR_FUSION
GGML_HRX_DISABLE_GATED_DELTA_NET_BETA_SIGMOID_FUSION
GGML_HRX_ENABLE_SSM_STATE_GATHER_FUSION
GGML_HRX_DISABLE_TOPK_MOE
GGML_HRX_DISABLE_GATED_DELTA_NET_CLUSTER8
GGML_HRX_DISABLE_FEATHER_SUBMIT
GGML_HRX_FEATHER_NODES_PER_SUBMIT
GGML_HRX_FEATHER_MAX_MUL_MAT_BYTES_PER_SUBMIT
```

This list can drift. Always `rg`.

## H. Vulkan Comparison

Useful source files:

```text
sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp
sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl
sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl
sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp
sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q5_k.comp
sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp
```

Shader tools available:

```bash
glslc
glslangValidator
spirv-dis
```

Use Vulkan as:

- graph-fusion oracle;
- gross tile/dataflow oracle;
- subgroup/wavefront clue;
- device-time reference.

For MoE routes, validate expert/route counts before using spike-era data as an
oracle. A hard-coded or degenerate route distribution can make a grouped kernel
look much better or worse than it is on real model routing. Compare the live
route histogram, Vulkan dispatch family, and HRX provider trace before changing
expert-loop structure.

April 19 integration triage found exactly this failure mode: the spike branch
reported p512 Q4 `MUL_MAT_ID`/SWIGLU MMQ buckets with the same HSACO text and
same workgroup grid as integration, but route-count tracing showed the first
Q4 MoE layers putting all 4096 token/id assignments into only 8 experts
(`max=512` each), while the integration branch routed the same shape across
roughly 150-220 experts. The same grouped MMQ kernel is much faster on the
degenerate dense-per-expert distribution than on broad sparse routing. Treat
spike-era MoE prompt throughput as evidence of a schedule under that route
distribution, not as proof that the integration route or kernel is slow by
itself. When optimizing Q4 MoE prompt kernels, collect:

- `iree-profile` Q4 bucket times;
- provider trace proving the same kernel route;
- route-count histograms for the same prompt/model;
- correctness evidence that any route-distribution change is intentional.

The simple experiment of launching one row/expert workgroup and looping route
tiles inside the Q4 x4 MMQ kernel was correct but slower on the Qwen p512/p1024
integration branch. Do not retry that shape without a new reason; if broad
sparse routing remains hot, design a schedule that preserves parallel route
tiles while reducing empty expert/tile overhead.

Do not blindly port:

- tile dimensions without full schedule;
- source loop order that triggers worse HIP codegen;
- one-row DMMV to prompt shapes where large MMQ is already better;
- wave64 into kernels whose HIP schedule is wave32-friendly.

## I. Common Failure Modes

### "invalid argument: -i"

llama-cli flags drift. Use the current reproducer scripts instead of old manual
interactive commands.

### Wrong backend library loaded

If `LD_LIBRARY_PATH` puts the wrong build `bin` first, Vulkan and HRX binaries
can resolve incompatible ggml libraries. Put the active backend build directory
first.

### Provider fallback logs that are not executed

Some fallbacks happen during support checks for a fused subgraph. Confirm with
rocprof kernel list and provider claim counts before chasing a false CPU path.
For Qwen routing, `ARGSORT` support-query fallbacks historically showed
`ncols=256` decode-router shapes while the graph was actually claimed by
`TOPK_MOE_EARLY_SOFTMAX_NORM`. Current HRX supports bounded decode-router
ARGSORT by default for `nrows<=64` to remove that ambiguity; broader prompt
ARGSORT still requires `GGML_HRX_ENABLE_ARGSORT`.

### CPU use but no fallback

Can be runtime overhead, llama.cpp worker threads, polling, or model load/page
faulting. Provider trace first; then inspect runtime/copy traces.

### "Exact test passed, chat corrupts"

Use the loop guard. Past issues included prompt ARGSORT/routing, approximate
prompt kernels, and FA accumulator layout. Hidden-state poison can surface only
after many decode tokens.

### "v_dot/v_wmma appears, but performance regressed"

Instruction presence is necessary but not sufficient. The rejected Q6/Q8_1 and
Q5 schedule probes emitted the desired primitives but had the wrong dataflow,
register pressure, or conversion overhead.

### WMMA layout guessed instead of proven

Do not write FA/BF16/WMMA stores from a row-major assumption. The gfx11 direct
FA bug cluster came from treating accumulator fragments as simple row-major
tiles when the actual lane mapping was interleaved. Use exact model-shape tests
with max-diff coordinates, a small controlled fixture if needed, and disassembly
to prove the mapping for the exact `v_wmma` instruction and accumulator type.
Short coherent chat is not enough for this class of bug.

### "Lower VGPR should be faster"

Not always. Low-VGPR variants often duplicated RHS reads or removed useful ILP.
Use rocprof and ATT.

## J. Current Baseline

April 17, 2026 end-of-spike checkpoint on W7900/gfx1100,
Qwen3.5-35B-A3B `Q4_K_L.gguf`, f16/f16 KV cache, HRX pure HIP provider.
Use untraced `Release` builds for scoreboard numbers. Tracy builds are useful
for timeline inspection but are not zero overhead.

llama.cpp source checkpoint:

```text
55b4e3afa Feather HRX graph submissions during decode
```

Final release scoreboard:

| Regime | Shape | HRX | Vulkan | Notes |
| --- | --- | ---: | ---: | --- |
| Decode | `p0 n64 fa=0 b512 ub512 r3` | `108.36 tok/s` | `108-109 tok/s` | HRX is effectively at Vulkan wall time for this shape. |
| Decode, feather disabled | `p0 n64 fa=0 b512 ub512 r3` | `83.55 tok/s` | n/a | `GGML_HRX_DISABLE_FEATHER_SUBMIT=1`; isolates the submission-overlap gain. |
| Prefill | `p512 n0 fa=1 b512 ub512 r3` | `2685.8 tok/s` | `2231.0 tok/s` | Warm HRX samples were about `2866 tok/s`. |
| Prefill + one decode | `p512 n1 fa=1 b512 ub512 r3`, prompt row | `2701.9 tok/s` | `2236.6 tok/s` | Standard prefill comparison in this doc. |
| Prefill + one decode | `p512 n1 fa=1 b512 ub512 r3`, decode row | `87.53 tok/s` | `86.27 tok/s` | First sample cold; warm HRX about `103.7 tok/s`. |
| Prefill, no FA | `p512 n0 fa=0 b512 ub512 r3` | `2047.1 tok/s` | `2223.2 tok/s` | HRX still trails Vulkan by roughly `7-8%` on warm samples. |

Key artifacts:

```text
build/hrx-prefill-current-20260417-075645/
build/hrx-correctness-feather-submit-smoke/
build/hrx-loop-check/feather-submit-smoke/
build/hrx-tracy-feather-submit-decode-p0n64-r1-20260417-075010/20260417-075010-llama-decode-p0n64-r1-feather-submit.tracy
```

Provider-traced runs are much slower and are only valid for route/count
inspection. The default decode path skips zero-element `SCALE` submissions; a
p0/n8 provider trace showed 30 live `SCALE` claims after the skip, versus the
larger historical count polluted by zero-work state scales. No fallbacks were
present in the traced decode path.

Default-on decode fusions:

- `SIGMOID_MUL_ADD_ADD`
- `L2_NORM_PAIR`
- `GATED_DELTA_NET_BETA_SIGMOID`

Rollback knobs are `GGML_HRX_DISABLE_SIGMOID_MUL_ADD_ADD_FUSION=1`,
`GGML_HRX_DISABLE_L2_NORM_PAIR_FUSION=1`, and
`GGML_HRX_DISABLE_GATED_DELTA_NET_BETA_SIGMOID_FUSION=1`.

Experimental but not default:

- `GGML_HRX_ENABLE_SSM_STATE_GATHER_FUSION=1` fuses the recurrent-state
  `SCALE`/`GET_ROWS` load into `SSM_CONV_UPDATE` via
  `hrx_ssm_conv_update_gather_f32`.
- This hit only the first decode sample's recurrent-state gather in the current
  llama-bench shape and did not yet justify default promotion.
- A first attempt at fusing the larger `cache_s` `GET_ROWS` into
  `GATED_DELTA_NET_STATE_UPDATE` faulted at the first fused dispatch and was
  reverted before commit. The opportunity is still high value, but needs an
  isolated state-index fixture before another end-to-end attempt.
- Wiring the existing Q6 `rows4_cols1_wg64` and `rows8_cols1_wg128` kernels
  behind `GGML_HRX_Q6_K_COLS1_ROWS_PER_WG` did not beat the current auto
  `rows2_cols1_wg32` route on p0/n64 and was reverted.

Meaning:

- Decode is no longer primarily a runtime-overhead gap versus Vulkan at the
  benchmarked shape. The staged-transfer and feathered-submit changes recovered
  the major host-side losses.
- The remaining decode work is mostly kernel/fusion quality, with runtime
  checks guided by Tracy if a new timeline regression appears.
- Flash-attention prefill is competitive or ahead on this model/shape. Non-FA
  prefill remains behind Vulkan and is the cleaner prefill gap if that path
  matters.
- Prefer hero-op fusions that remove real intermediate traffic. Keep opt-in
  gates until no-trace tok/s and full Qwen gates both justify promotion.

April 19 small-prefill note:

- For Qwen Q4 `MUL_MAT_ID` on W7900, the grouped ID providers are not the best
  small-prefill route below p32. A later direct rerun on the generalized branch
  showed the rows2_x16 WG32 route beating forced row4 and row8 across
  p2/p3/p4/p5/p6/p7/p8/p9/p11/p13/p15/p16 by roughly 3-8%; keep rows2_x16
  below the p32 grouped threshold unless new same-binary evidence says
  otherwise.
- Do not infer the same threshold for Q4 SwiGLU or Q5/Q6 matvecs. Q4 SwiGLU
  grouped row2 starts paying for itself at p8, and disabling Q5/Q6 cols8 prompt
  routes regressed p8/p15/p16/p17 even though odd tails show wasted column-tile
  work.
- Q5/Q6 rows2 skinny prompt matvecs benefit from exact-width entry points at
  very small prefill sizes. The W7900/Qwen route evidence favored cols2..8 for
  p2..p8, a cols3 specialization for p9, and the original cols8 schedule for
  p10 and above. A padded-work heuristic that selected cols5/cols6 for p15/p17
  was slower despite doing less nominal work; keep the explicit p9-only tail
  rule unless new per-kernel evidence proves a broader selector.
- Q6 rows2/cols2..8 got a real small-prefill win from a wg32 family derived
  from the existing cols1 decode-style `dot16` path. The analogous-looking Q5
  wg32 probe regressed against Q6-only, and BF16/F32 rows2/cols4 probes either
  regressed p4/p5 or were below noise. Do not promote symmetry changes without
  A/B runs that include odd sizes; rerun before acting on single-run dips.
- Q4 SwiGLU packed prefers the narrower WG32 split for p2..p8 in the latest
  W7900/Qwen skinny-prefill profile. Repeated p2..p5 reruns kept the WG32 edge
  positive, while forcing row2/grouped routes regressed p2..p8 by roughly
  5-13%; keep p1 on WG64 and p2..p8 on packed WG32 unless new repeated
  evidence says otherwise.
- BF16 and F32 exact-width skinny matvecs are useful selectively. BF16/BF16
  SwiGLU exact cols2/3/5/6/7 were positive at p2/p3/p5/p6/p7 and neutral at
  p4/p8. F32 exact cols3/4/5/6/7 were positive but smaller, with the clearest
  wins at p3 and p6. Do not keep unused fallback variants: an F32 cols8
  fallback was removed because exact-width providers beat it and it was not a
  default route.
- BF16 SwiGLU p8 should use the single-row `cols8` route, not
  `rows2_cols8`. Repeated W7900/Qwen A/B of disabling `rows2_cols8` showed p8
  improving from about 338.4 to 350.1 tok/s (+3.5%), and the committed selector
  check measured p8 at 356.9 tok/s with IREE profile showing
  `hrx_mul_mat_vec_bf16_swiglu_cols8_f32`.
- Prompt TopK/MoE fusion became safe to enable for p2..p4 only with the
  `shared4` route. The compact normalized-weight output aliases the logits
  allocation in the Qwen graph; `shared4` is safe for p2..p4 because one
  workgroup reads all rows into registers before any output write. A later
  `shared8` route applies the same one-workgroup property to p5..p8; this is
  safe because the kernel reads all logits before writing compact outputs, not
  because p5+ aliasing is generally safe. Do not broaden the same memory-safety
  exception beyond the one-workgroup variants without a different kernel
  strategy, because multiple workgroups can race on the compact output/logits
  alias. The p2..p4 default route moved W7900/Qwen p2 from roughly 157.8 to
  170.2 tok/s, p3 from 213.9 to 229.6 tok/s, and p4 from 264.0 to
  281.2 tok/s. Repeated p5..p8 A/B runs with `shared8` versus forced
  `shared4`/unfused prompt TopK showed p5 +5.1%, p6 +4.4%, p7 +3.5%, and
  p8 +5.4%; p9 remained unfused in the profile.
- After these skinny primitive improvements, HRX still trails Vulkan at p3..p8
  while beating Vulkan at p2 and p9+. IREE profiling of p7 showed about
  16.9 ms/run summed dispatch device time and about 20.0 ms/run queue span
  across 98 submissions/run, matching the normal ~20.5 ms wall time. Treat the
  remaining p3..p8 gap as both device-kernel and many-small-dispatch overhead;
  rerun before acting on single-run dips.
- Two later symmetry/tiling probes were rejected and reverted:
  - Q6_K rows4/cols2..8 WG64 kept the existing dot16 algorithm but processed
    four rows per workgroup; it passed `test-backend-hrx` and 467/467
    `MUL_MAT`, but regressed p2..p11 and p13..p20 by roughly 4-10% on the
    W7900/Qwen p2..p20 sweep. Do not revisit row grouping without a different
    algorithm, not just a larger row tile.
  - F32 batched rows2/cols2..7 exact-width providers also passed correctness,
    but regressed the intended p2..p7 range by roughly 0.2-2.5%. The existing
    generic F32 batched route is better for p2..p7 despite doing more nominal
    workgroups; do not promote exact-width F32 rows2 solely by analogy with
    quantized skinny kernels.
- April 19 p7/p8 follow-up rejected more selector-only changes after TopK and
  BF16 p8 routing:
  - `GGML_HRX_DISPATCHES_PER_SUBMIT=8` had a narrow p8-looking win in one
    sweep, but the broad p2/p3/p4/p5/p6/p7/p8/p9/p11/p13/p15/p16 A/B regressed
    p7, p8, p9, p13, and p16; keep the default submit threshold.
  - Disabling Q5 rows2 prompt providers lost roughly 5-10% on p5..p8, and
    forcing Q5 WG128 was neutral-to-negative after rerun.
  - Disabling Q6 WG32 rows2 providers lost on p7/p8; WG64/WG128 fallback is not
    an improvement.
  - Disabling Q4 SwiGLU packed routes lost about 9-14% on p7/p8, and disabling
    WG32 alone still lost about 1.5-2%; packed WG32 remains the right skinny
    Q4 SwiGLU route.

April 21 decode-final grind notes:

- The useful optimization loop was: full-model bucket profile, provider route
  proof, native HIP harness for the isolated kernel, full-model re-profile,
  focused correctness, and full Qwen/loop guard before promotion.
- Native HIP harnesses are necessary but not sufficient. A Q6 exact-K variant
  improved the isolated harness and passed `test-backend-hrx`, but regressed
  the full-model Q6 dispatch buckets. It was backed out. Do not promote
  microbench-only wins.
- Current Q4 MoE decode choices were rechecked in the native harness:
  `swiglu_packed_wg32` beat WG64 and rows2 variants for the Qwen decode
  SWIGLU shape, and `mul_rows2_x16_wg32` decisively beat the packed/WG64 Q4
  MUL alternatives.
- Current Q5 decode route was rechecked against obvious rows2/rows4/dot16 and
  WG-size variants at both normal and giant-row decode shapes. `q5_wg32`
  remained the best tested route, despite Q5 being the largest aggregate
  decode bucket in the short profile. Future Q5 work should compare packing,
  wait behavior, and instruction mix against Vulkan/CUDA rather than trying
  the same row-group variants again.
- The retained decode wins came from route/fusion changes that removed real
  work or made the live route more exact: GDN direct gather, SSM decode/direct
  update, FA decode reduce shared-scale, one-sided gate strided sigmoid/mul,
  Q6 SILU/MUL, and BF16 vector `MUL_MAT -> SET_ROWS`.
- BF16 `MUL_MAT -> SET_ROWS` is the model example of a good fusion: it removed
  the standalone `SET_ROWS` dispatch and avoided an intermediate F32 write/read
  while preserving a provider expectation test.
- Submission-boundary tweaks are not stability fixes. If changing batching or
  flush boundaries changes correctness or hangs, treat that as evidence of a
  real bug elsewhere and root-cause it.
- If generation produces repeated question marks, garbage, or late-loop
  collapse, stop optimizing and bisect/RCA. These failures have repeatedly
  been hidden-state poison from route, fusion, FA, or approximate-math issues,
  not sampler behavior.
- Current remaining decode grind list after reaching roughly 115 tok/s on
  W7900/Qwen: Q6_K decode/SILU buckets first, Q5_K decode packing/wait
  analysis second, then Q4_K MoE packed true-up, BF16 SWIGLU/dense tail,
  TopK/MoE, and real dispatch/materialization elimination.

## K. Search Map

Useful `rg` terms:

```text
GGML_HRX_TRACE_PROVIDERS
GGML_HRX_DISABLE_FAST_APPROX_PROMPT
q8_1_x4
MMQL128
rows2_cols1_wg32
rows2_x16_wg32
rows4_k2048_cols1_lds
cluster8_nokda
s128_h32_qk16_tok1_nokda
gfx11_direct
TOPK_MOE_EARLY_SOFTMAX_NORM
ARGSORT
hrx_rocprof_scoreboard
qwen_hrx_correctness_gate
qwen_loop_guard
GGML_VK_PERF_LOGGER
```

Primary analysis logs:

```text
docs/analysis/ historical llama.cpp optimization logs
docs/analysis/llamacpp_vulkan_hip_optimization_hopper.md
```
