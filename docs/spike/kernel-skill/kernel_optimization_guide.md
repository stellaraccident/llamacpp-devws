# HRX HIP Kernel Optimization Guide

Status: working guide from the llama.cpp Epic 2 optimization spike.

Primary target so far: `Qwen3.5-35B-A3B-UD-Q4_K_L.gguf` on Radeon Pro W7900
(`gfx1100`, RDNA3) through the llama.cpp `ggml-hrx` pure-HIP provider.

## Index

- [A. Operating Model](#a-operating-model)
- [B. Measurement Stack](#b-measurement-stack)
- [C. Tool Reference](#c-tool-reference)
- [D. Correctness Gates](#d-correctness-gates)
- [E. Optimization Catalog](#e-optimization-catalog)
- [F. Accepted Patterns](#f-accepted-patterns)
- [G. Rejected Patterns](#g-rejected-patterns)
- [H. Current Scoreboard](#h-current-scoreboard)
- [I. Search Terms](#i-search-terms)

## A. Operating Model

### Optimize regimes separately

Do not mix prompt/prefill and decode conclusions.

- **Prefill**: `cols` / token count is wide. The right algorithms are tiled
  matrix-matrix, Q8_1 RHS packing, route-tiled MoE, BF16/WMMA, and prefill
  flash attention.
- **Decode**: `cols=1`. The right algorithms are DMMV-style K-quant matvec,
  sparse MoE selector/compute, small dense matvec, recurrent/state fusions, and
  very narrow exact shape specializations.
- **Wall clock**: currently still includes HRX runtime overhead. Use tok/s as a
  guardrail. For runtime overhead, capture Tracy plus IREE profile files. For
  kernel work, use device-time buckets first.
- **Device time**: compare HRX rocprof kernel sums against Vulkan perf logger
  label sums. The methods are not identical, but they are good enough for
  kernel bucket ranking and head-to-head sanity. Do not use rocprof as the
  default system/runtime lens.

### Preferred loop

1. **Classify the graph** with HRX provider trace and Vulkan perf labels.
2. **Choose the profiling lens**: Tracy plus `iree-profile` for runtime
   overhead, rocprof kernel-trace SQLite for kernel bucket ranking.
3. **Pick one hot bucket** with a proven Vulkan/reference gap or graph
   granularity issue.
4. **Mine known priors before coding** when the gap is broad. For HRX2/Loom
   work this means HRX1 HIP, Vulkan shaders and dispatch, CUDA/HIP MMQ/DMMV,
   and any relevant OpenCL/Metal kernels. Run or disassemble the prior when
   possible and extract the real schedule: tile dimensions, lane ownership,
   vector load width, packed data layout, dot primitive, LDS/shared-memory
   staging, reduction/writeback structure, and activation constraints.
5. **Write the schedule ledger.** Prior search is incomplete until the exact
   schedule facts are recorded in the analysis log or prior-art doc: source
   path and symbol, shape regime, BM/BN/BK or equivalent tile shape,
   wave/subgroup width, lane ownership, per-lane output count, vector and
   packed load width, quant layout, dot primitive, A/B staging, barrier cadence,
   reduction/writeback policy, emitted resource facts, and any known failure
   mode. A prior that is only mentioned but not decomposed is not evidence.
6. **Generate analytical alternatives before coding.** Use the prior ledger to
   form a short list of schedule hypotheses that preserve known-good dataflow
   while varying one meaningful axis at a time: tile dimensions, wave32 versus
   wave64, Q8_1 packing, A-side or B-side staging, vector width, unroll depth,
   output ownership, tail strategy, or fusion boundary. Do not move directly
   from "backend X is faster" to a guessed kernel rewrite.
7. **Bracket adjacent schedules in the kernel loop.** It is valid to probe
   nearby schedules that do not have strong direct prior evidence, but they
   must be pivots around a documented schedule family: name the pivot axis,
   sweep range, and expected signal. Run those variants through focused
   kernel/backend-op correctness and timing first. Promote to full integration
   only after the sweep shows a material bucket-level win or a useful
   refutation.
8. **Compare emitted code, not just source.** For Loom use compile reports and
   HSACO disassembly; for Vulkan use generated SPIR-V/disassembly plus perf
   labels; for HIP/CUDA use object/HSACO/SASS/ISA where available. If HRX2 is
   orders of magnitude behind, expect an algorithmic schedule mismatch before
   assuming a compiler/codegen limitation.
9. **Write the smallest exact provider gate** for the target shape.
10. **Run focused correctness**, then the full Qwen gate before defaulting.
11. **Measure device time**, not just wall tok/s.
12. **Inspect ISA/resources** if a change is surprising.
13. **Use ATT/thread trace** only when source-level structure is no longer
   enough and the capture can be filtered to one kernel.
14. **Record rejected variants** with numbers. Many plausible Vulkan-shaped
   changes regressed.

Do not iterate on local knobs when the candidate is not in the same schedule
family as a known-good prior. First port the missing schedule class, then tune.
Once the schedule family is grounded, bracket nearby variants with a controlled
kernel benchmark sweep instead of scattered full-model probes.

### HRX2 Phase 2a boulder rule

For the current HRX2/Loom throughput lift, Vulkan is the same-machine baseline
and the target basket lives under `shared/models/llamacpp-hrx2-basket-v1`.
When HRX2 is still far behind Vulkan, treat the problem as structural until
proven otherwise. Own the full stack needed to prove that: environment, build
state, stale kernel caches, benchmark harness, route metadata, runtime interop,
and model-shape evidence are all part of the optimization task.

Current Phase 2a evidence puts decode at roughly 0.23x-0.38x Vulkan with no CPU
compute fallback, while prefill remains about 0.015x-0.05x Vulkan. Do not spend
multi-hour loops on tiny knobs in that state. First rule out the big classes:
unsupported route domains, CPU fallback, graph materialization/copy traffic,
missing HRX1 runtime behavior, weak packed quantized matmul schedules, missing
attention/fusion paths, and measurement artifacts.

Before coding a large kernel rewrite, produce a prior matrix from HRX1, Vulkan,
CUDA/HIP, and any relevant backend: exact source/symbol, shape regime, tile
sizes, lane ownership, per-lane output count, vector load width, packing
format, dot primitive, LDS/shared-memory staging, reduction and writeback
strategy, barriers, and emitted ISA/resource facts. Then add an analytical
alternatives table derived from that matrix. Each candidate should say which
prior it follows, which axis it changes, and what outcome would confirm or
refute the hypothesis. Only then implement the missing schedule class in Loom
or as a bridge with explicit WYSIWYG vectorization and validate it with focused
backend op tests before model benchmarks.

For the current prefill gap, prefer one well-grounded boulder at a time over a
large scatter of speculative tweaks. A pass is not complete until it has either
moved the relevant bucket materially, or produced a concrete refutation: the
Loom source matches the prior schedule, the emitted ISA/resource facts have
been compared, focused op tests pass, and the remaining delta is narrowed to a
specific compiler/runtime/measurement hypothesis.

When the current candidate is far from target, schedule guessing is a failure
mode. If a candidate cannot point to a prior ledger row or a derived analytical
alternative, stop and do that research first. The correct output of prior
search is a reusable schedule-shape document, not just a vague assertion that
another backend "does MMQ" or "uses wave64."

This does not forbid exploratory pivots. It forbids blind pivots. A nearby
schedule can be worth testing even when no prior says it will win, provided it
is framed as a bracket around the schedule being pursued. For example, after
selecting a Vulkan-style packed-MMQ schedule, it is reasonable to sweep BN, BK
step, A/B staging choice, vector load width, and unroll depth around that
schedule. The sweep should run as a kernel or focused backend-op benchmark with
correctness gates and route traces, producing a compact table that either
selects a variant or rejects the axis. Only then should the variant move to
llama-bench or chat/integration testing.

### Schedule ledger and pivot protocol

For every broad kernel gap, create or update a short ledger before code
changes. The ledger is the contract that separates useful exploration from
blind guessing:

```text
Prior row:
- source/symbol/backend:
- shape regime and evidence artifact:
- tile/workgroup/subgroup:
- lane ownership and per-lane outputs:
- vector/packed load widths:
- quant or element layout:
- dot/WMMA/ALU primitive and signedness:
- A/B staging, barriers, unroll, reduction, writeback:
- emitted resource facts:
- known win/regression/constraint:

Candidate row:
- follows prior row:
- pivot axis:
- sweep range:
- expected signal:
- correctness gate:
- timing gate:
- decision:
```

Adjacent schedules can lack a direct prior only when they are written as a
candidate row that pivots around an existing prior row or an explicitly chosen
analytical schedule. Examples: BM64 versus BM128 around the same MMQ dataflow;
BK_STEP 1/2/4/8 around the same cooperative staging pattern; vector load width
4/8/16 around the same packed layout; tail strategy variants for the same
ownership map. These should be benchmark-sweep inputs, not one-off integrated
routes or production-catalog entries.

Do not use full model runs to discover whether a speculative tile, vector
width, unroll, or staging axis is promising. First run the focused kernel or
backend-op sweep and record route selection, correctness, device timing, compile
report deltas, and emitted ISA/resource facts. Move the winning variant into
the llama.cpp production catalog only after it has bucket-level evidence and a
clear shape domain.

### Decode grind loop that held up

The April 2026 decode-final pass converged only after using this stricter loop:

1. Start from a no-trace full-model run to keep endpoint tok/s honest.
2. Capture a short full-model `iree-profile` dispatch run and rank buckets by
   export name, count, total time, and average time.
3. Use provider trace or provider expectation tests to prove the intended route
   is live. Disable generic fallbacks in the test when a fusion must fire.
4. Build a native HIP harness only for the bucket being studied. Compare the
   current route, obvious variants, and Vulkan/CUDA-inspired packing choices.
5. Re-profile the full model before keeping the change. Native microbench wins
   can regress the real graph due to launch shape, memory residency, codegen,
   or interaction with adjacent dispatches.
6. Run focused correctness and the full Qwen gate before promotion. For routing,
   FA, TopK, approximate prompt math, or recurrent state, also run long
   generation guards.
7. Record both accepted and rejected variants. Do not re-run rejected shapes
   without a new hypothesis.

This loop caught a misleading Q6 exact-K native win: the native harness improved
the isolated kernel, but the full-model dispatch profile regressed the active
Q6 buckets, so the production route was backed out. Treat that as the default
standard for future microbench-driven changes.

### Acceptance hierarchy

Use this order unless the task explicitly says otherwise:

1. No provider fallback for the intended path.
2. Focused correctness for the actual model shape.
3. Full correctness gate including chat and loop guard.
4. Target bucket improves in rocprof by more than noise.
5. Adjacent hot buckets and decode/prefill guardrails do not regress.
6. Wall tok/s is neutral or positive.

For approximate prompt kernels, exact CPU comparison is a diagnostic, not the
final contract. The route must still be model-stable and have rollback knobs.

## B. Measurement Stack

### HRX provider trace

Purpose: route selection, fallbacks, shape discovery.

Signals:

- `claim OP provider=...`: HRX accepted and dispatched a provider.
- `fallback OP ...`: a support check or actual fallback happened. Confirm
  whether it executed before treating it as a CPU path.
- `CPY`, `CONT`, `CONCAT`, `buffer_copy`: suspect materialization unless
  explicitly expected.

Do not use provider-traced tok/s as a performance number.

### rocprofv3 kernel trace

Purpose: HRX kernel device-time bucket ranking and targeted counter work.

Use kernel trace for kernel optimization. Do not use rocprof as the default
runtime-overhead tool; use Tracy plus `iree-profile` for that. Add HSA/copy
trace only for a specific hardware/copy question because it perturbs runs and
can include setup/model-load work.

For decode, filter setup copies. A raw database can show hundreds of D2D copies
before the first decode kernel. Those are not steady decode work.

### Vulkan perf logger

Purpose: reference bucket labels and graph-fusion comparison.

Vulkan labels are shader/fusion labels, not the same table as rocprof kernels.
Still, they are the best local reference for where the established Radeon path
spends device time.

### ISA/resource summary

Purpose: check wavefront size, VGPR/SGPR, spills, LDS, and primitive emission.

Always inspect the **built HSACO** for final conclusions when CMake uses
per-source flags. The standalone helper is useful, but it can miss compile flags
such as `-mwavefrontsize64` or per-kernel `-O3`.

### ATT/thread trace

Purpose: instruction-level stall evidence once coarse profiling stops being
enough.

Use filtered one-kernel captures. Unfiltered ATT can hang or produce unusable
artifacts on the ROCm 7.13 alpha toolchain.

## C. Tool Reference

### Environment

Use the ROCm tree in the workspace:

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

Verify linkage before profiling:

```bash
ldd "$LLAMA_BUILD/bin/llama-bench" | rg 'libhsa|librocprofiler-register|libhrx|therock'
```

Expected: `libhsa-runtime64.so.1` and `librocprofiler-register.so.0` from
`rocm/lib`, `libhrx.so.0` from `build/hrx-rocm713-install/lib64`, and
no `therock`.

### Build

```bash
cmake --build build/llama-hrx-rocm713 \
  --target llama-bench llama-cli test-backend-hrx test-backend-ops export-graph-ops hrx-kernel-bench \
  -j"$(nproc)"
```

If provider source or compile flags changed, clear the HRX kernel cache before
measuring:

```bash
rm -rf "$HRX_CACHE_DIR/kernels"
```

### Baseline wall runs

HRX prefill:

```bash
"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 512 -n 1 -b 512 -ub 512 -fa 1 -r 3 \
  -o json --no-warmup -ngl 99 -dev HRX0
```

HRX decode:

```bash
"$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 0 -n 64 -b 512 -ub 512 -fa 0 -r 3 \
  -o json --no-warmup -ngl 99 -dev HRX0
```

Vulkan equivalents:

```bash
build/llama-vulkan/bin/llama-bench \
  -m "$MODEL" -p 512 -n 1 -b 512 -ub 512 -fa 1 -r 3 \
  -o json --no-warmup -ngl 99 -dev Vulkan0

build/llama-vulkan/bin/llama-bench \
  -m "$MODEL" -p 0 -n 64 -b 512 -ub 512 -fa 0 -r 3 \
  -o json --no-warmup -ngl 99 -dev Vulkan0
```


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

### Tracy plus IREE profile files

Purpose: temporary fused runtime view until IREE profile data and Tracy are
upstream-fused. Tracy gives the HRX/IREE system timeline; `iree-profile` gives
logical HAL queues, command buffers, exports, and dispatches.

Use the Tracy-enabled HRX install built with `IREE_TRACING_MODE=1`; mode 2
allocation tracking currently trips Tracy capture verification during llama
startup.

```bash
export HRX_RUNTIME_INSTALL="$PWD/build/hrx-rocm713-tracy-install"
export LD_LIBRARY_PATH="$HRX_RUNTIME_INSTALL/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$LLAMA_BUILD/bin:${LD_LIBRARY_PATH:-}"
export IREE_TRACY_CAPTURE="$PWD/build/iree-tracy-tools/tracy/iree-tracy-capture"
OUT=build/hrx-tracy-fused-smoke
mkdir -p "$OUT"

HRX_PROFILE_FILE="$OUT/run.ireeprof" \
  sources/iree/build_tools/tracing/iree_tracy_capture.py \
    --output-dir "$OUT" --name llama-p32n8 \
    -- "$LLAMA_BUILD/bin/llama-bench" \
      -m "$MODEL" -p 32 -n 8 -b 512 -ub 512 -r 1 \
      -o json --no-warmup -ngl 99 -dev HRX0 \
    > "$OUT/llama-bench.json"

build/iree-rt/tools/iree-profile summary "$OUT/run.ireeprof" \
  > "$OUT/iree-profile-summary.txt"
build/iree-rt/tools/iree-profile dispatch --format=jsonl "$OUT/run.ireeprof" \
  > "$OUT/iree-dispatch.jsonl"
```


### Provider trace

```bash
GGML_HRX_TRACE_PROVIDERS=1 "$LLAMA_BUILD/bin/llama-bench" \
  -m "$MODEL" -p 0 -n 8 -b 512 -ub 512 -fa 0 -r 1 \
  -o json --no-warmup -ngl 99 -dev HRX0 \
  > build/trace.json 2> build/trace.log
```

Summarize HRX/Vulkan traces:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-trace-summary.py summarize \
  --hrx-log build/trace.log \
  --vulkan-log build/vulkan-trace.log \
  --top 40
```

Or run both:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-trace-summary.py run-qwen \
  --backend both --prompt 0 --gen 8 --output-dir build/trace-diff
```

### rocprof device-time runs

Prefill:

```bash
OUT_DIR=build/rocprof-hrx-prefill-current \
OUT_FILE=hrx-p512 \
PROMPT=512 GEN=0 REPETITIONS=1 FA=1 \
reproducers/rocprof_qwen_hrx_prefill.sh
```

Decode:

```bash
OUT_DIR=build/rocprof-hrx-decode-current \
OUT_FILE=hrx-decode-n64 \
GEN=64 REPETITIONS=1 \
reproducers/rocprof_qwen_hrx_decode.sh
```

Summarize raw rocprof:

```bash
sources/llama.cpp/tools/hrx-epic2/rocprof-rocpd-summary.py \
  build/rocprof-hrx-prefill-current/hrx-p512_results.db \
  --top 30
```

Summarize family scoreboard:

```bash
reproducers/hrx_rocprof_scoreboard.py \
  build/rocprof-hrx-prefill-current/hrx-p512_results.db \
  --baseline build/rocprof-hrx-prefill-baseline/hrx-p512_results.db \
  --top 20 --top-kernels 20
```

### Vulkan perf logger

```bash
GGML_VK_PERF_LOGGER=1 build/llama-vulkan/bin/llama-bench \
  -m "$MODEL" -p 512 -n 0 -b 512 -ub 512 -fa 1 -r 1 \
  -o json --no-warmup -ngl 99 -dev Vulkan0 \
  > build/vulkan-p512.json 2> build/vulkan-p512.log
```

The logger lines look like:

```text
MUL_MAT q6_K m=4096 n=512 k=2048: 30 x ... us = ... us
```

### ISA/resource summary

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-kernel-isa-summary.py \
  --kernel 'mul_mat_vec_q[56]_k_q8_1|gated_delta|flash_attn' \
  --out-dir build/isa-current \
  --json build/isa-current/summary.json
```

Fields to record:

- `wavefront_size`
- `vgpr_count`, `sgpr_count`
- `vgpr_spill_count`, `sgpr_spill_count`
- `group_segment_fixed_size` / LDS
- opcode counts for `v_dot`, `v_wmma`, `v_mfma`, `s_barrier`, `ds_*`,
  `global_load`

### Built-HSACO inspection

When CMake flags matter, inspect the built object rather than only the helper.
Useful commands:

```bash
"$ROCM_PATH/lib/llvm/bin/llvm-readobj" --notes path/to/kernel.hsaco
"$ROCM_PATH/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx1100 path/to/kernel.hsaco \
  | rg 'v_wmma|v_mfma|v_dot|s_barrier|s_waitcnt|global_load|ds_'
```

### ATT/thread trace

Filtered one-kernel capture shape:

```bash
OUT_DIR=build/rocprof-att-one-kernel
mkdir -p "$OUT_DIR"

"$ROCM_PATH/bin/rocprofv3" \
  --rocm-root "$ROCM_PATH" \
  --att \
  --att-library-path "$ROCM_PATH/lib" \
  --att-target-cu 0 \
  --att-simd-select 0x0 \
  --att-shader-engine-mask 0x1 \
  --att-buffer-size 0x6000000 \
  --att-serialize-all 1 \
  --kernel-include-regex 'hrx_kernel_name_here' \
  --output-format json \
  -d "$OUT_DIR" \
  -o att \
  -- "$LLAMA_BUILD/bin/llama-bench" \
    -m "$MODEL" -p 0 -n 1 -b 512 -ub 512 -fa 0 -r 1 \
    -o json --no-warmup -ngl 99 -dev HRX0
```

Summarize decoded CSV:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-att-summary.py \
  "$OUT_DIR"/stats_ui_output_agent_*_dispatch_*.csv \
  --top 40
```

Interpretation:

- High `s_waitcnt` after `global_load`: load schedule or immediate consume.
- High `s_waitcnt lgkmcnt`: scalar/kernarg/LDS dependency.
- Tiny kernels dominated by kernarg waits are graph/fusion targets, not kernel
  body targets.
- A reduction change must lower wait/barrier cost in ATT, not just look cleaner
  in source.

## D. Correctness Gates

### Inner loop

Use this before expensive full gates:

```bash
FOCUS=q6 RESULTS_DIR=build/hrx-inner-loop-q6-candidate \
  reproducers/qwen_hrx_inner_loop.sh
```

Valid `FOCUS` values include `prompt`, `q4`, `q5`, `q6`, `dense`, `gdn`, `fa`,
and `moe`.

For approximate candidates:

```bash
FOCUS=prompt APPROX=1 CHECK_LOOP=1 \
  CANDIDATE_ENV='GGML_HRX_ENABLE_MY_APPROX=1' \
  RESULTS_DIR=build/hrx-inner-loop-my-approx \
  reproducers/qwen_hrx_inner_loop.sh
```

### Milestone gate

Run before default-enabling or committing a meaningful kernel change:

```bash
RESULTS_DIR=build/hrx-correctness-qwen-candidate \
CHECK_CHAT=1 CHECK_LOOP=1 CHECK_FA=1 CHECK_DIRECT_FA=1 \
LOOP_SEEDS=1,5 LOOP_PREDICT=384 \
reproducers/qwen_hrx_correctness_gate.sh
```

The gate covers:

- conservative exact `MUL_MAT`, `MUL_MAT_ID`, `SSM_CONV`
- autoregressive model-shape `GATED_DELTA_NET`
- default and generic `FLASH_ATTN_EXT`
- short and long chat smoke
- deterministic loop/collapse guard

### Long-generation guard

Use this for prompt routing, approximate math, ARGSORT/TOPK, FA, or any change
that can poison hidden state:

```bash
reproducers/qwen_loop_guard.py \
  --backend hrx \
  --seeds 1,5 \
  --runs 1 \
  --predict 384 \
  --context 4096 \
  --out-dir build/hrx-loop-check/candidate
```

Loop symptoms seen in the spike included repeated token phrases and long runs of
question marks after O(100+) decode tokens. Vulkan controls did not reproduce
those failures.

## E. Optimization Catalog

### Graph and materialization

Accepted:

- Decode recurrent-state fusions: `GATED_DELTA_NET_STATE_UPDATE` and
  `SSM_CONV_UPDATE_SILU`.
- Decode tiny-dispatch fusions available as opt-ins:
  `SIGMOID_MUL_ADD_ADD`, `L2_NORM_PAIR`, and
  `SIGMOID_BETA_GATED_DELTA_NET_STATE_UPDATE`. In the 2026-04-17 stress sample,
  the combined opt-in reduced decode dispatches from 17,504 to 15,904 over
  16 tokens and improved `p0 n64` from about 70.85 to 72.22 tok/s, but loop
  guard still showed latent nondeterministic question-mark failures both with
  and without the opt-ins. Do not promote these by default until the latent
  instability is understood.
- Full-attention gate materialization fusion: `SIGMOID_MUL_STRIDED`.
- `Q8_0 MUL_MAT + ADD` fusion for decode and prompt variants.
- `ADD8` MoE accumulation fusion. This removed thousands of tiny add dispatches
  and produced one of the clearest wall wins in decode.
- Zero-length `GET_ROWS` / `CPY` skip before provider claim logging.

Gotchas:

- D2D copies are usually app/kernel bugs, not runtime issues.
- Raw logits readback is expected in llama-bench and can appear as a steady
  993,280-byte copy for the full vocab output.
- Provider trace fallbacks can be support-query artifacts for fused paths; prove
  they execute before assigning runtime cost.
- HRX already records command-buffer execution barriers in the low-level stream
  API after dispatch/copy/fill/update. Do not add duplicate llama-side barriers
  as a default fix. In the 2026-04-17 question-mark loop RCA,
  `GGML_HRX_BARRIER_EACH_DISPATCH=1` happened to pass 12/12 while normal
  execution failed 1-2 times per 12, but that knob adds a second barrier on top
  of the HRX default and should be treated as diagnostic timing evidence, not a
  principled fix.

### K-quant decode

Accepted:

- Q6 direct F32-RHS decode: `rows2_cols1_wg32` with one wave/subgroup and two
  rows per workgroup. This follows the Vulkan DMMV shape and reduced Q6.
- Q5 direct decode: keep current `wg128` / giant-row `wg64` split. It was
  already ahead of Vulkan in the current decode comparison.
- Q4 MoE ID decode: exact `rows2_x16_wg32` with fixed `k=512` unroll and
  vectorized loads. This was a large bucket win.
- Q4 MoE SWIGLU decode: keep packed `wg64` base with targeted unroll/vector-load
  cleanup. Broader multi-row variants lost.

Rejected:

- Q8_1/MMVQ decode default. Quantization and current packed-dot kernels regressed
  Q5/Q6 badly in decode.
- Applying Q6 `rows2_wg32` shape to Q5. Correct but slower.
- Q4 ID two-row packed `wg64`, shared-RHS `wg16`, RHS shuffle, and min factoring
  variants. Only the `rows2_x16_wg32` schedule won.

### K-quant prefill / prompt

Accepted:

- Q5 large MMQL: Q8_1 x4 RHS, `128x128`, `wg256`, wave64. Later best state used
  `BK_STEP=1`, aligned Q5 q-byte loads, and Q5-only device `-O3`.
- Q6 large MMQL: Q8_1 x4 RHS, `128x64`, `BK_STEP=4`, wave64, direct Q6 pack,
  Vulkan-style fill order, and Q6-specific `-O3`.
- Q4 MoE route-tiled MMQ: Q8_1 x4 RHS, route tiles in matrix-N dimension, Q4
  packed dot, staged-load follow-up, and wave64 compile for the prompt MMQ
  kernels.
- Q8_0 prompt/fused-add reroute to Q8_1 x4 packed MMQ `128x32`.

Rejected:

- Q5/Q6 `32x32` prompt MMQ for p512. The gross tile was wrong for the real
  shape.
- Q6 `128x128` retile. More Vulkan-looking at a glance, slower in HIP.
- Q5 `128x64` and `64x64` macro-tiles. Lower register pressure did not beat the
  work/grid increase.
- Q5 fill-order change that helped Q6. It catastrophically regressed Q5.
- Q5 fast-math, broad `restrict`, `launch_bounds`, and packed `dm` loads.
- Q4 SWIGLU route-loop / reduced route-lane count. It serialized useful route
  parallelism and regressed.

### Dense prompt

Accepted:

- BF16 WMMA16 prompt for dense and SWIGLU. This uses RDNA `v_wmma`, rounds RHS
  to BF16, and is guarded as a fast approximate prompt route.
- Dense F32 prompt reduce16 cleanup.

Gotchas:

- BF16 WMMA fails strict CPU exactness because the RHS is rounded to BF16. It is
  accepted only through model-level guardrails and rollback knobs.
- Do not infer MFMA/WMMA is relevant to Q5/Q6 Q8_1 MMQ; Vulkan uses integer dot
  for those paths.

### Dense decode

Accepted:

- BF16 `k=512 rows=2048 cols=1`: rows4/LDS provider.
- BF16 `k=2048` dense and SWIGLU decode: rows4/LDS providers with four row
  shards and repeated RHS amortization.
- F32 batched `k=2048 cols=1 dst_ne2=1`: wave32/float4 fixed-shape provider.

Rejected:

- Simple BF16 `wg128` override. The useful change was dataflow/RHS staging, not
  workgroup size.
- Dense F32 `rows4_cols4` prompt tile. Correct but slightly slower.

### Gated Delta Net

Accepted:

- Prompt `S=128` non-KDA specialization, cluster8, wave32.
- GDN algebra placement cleanup and Q-reload lifetime cleanup.
- Decode exact `s128_h32_qk16_tok1_nokda`, one wave32 workgroup, four columns
  per workgroup, DPP row-shift reduction plus broadcast.

Rejected:

- GDN wave64 on this RDNA3 HIP schedule.
- Broadcast-only exp scalarization.
- Vulkan-like Q preload in the HIP non-KDA kernel.
- DPP row-broadcast replacement.

### Flash attention

Accepted path progression:

1. Scalar cleanup and causal mask skip: made `-fa 1` useful for prompt.
2. Prompt-only tile8: reduced the pathological scalar bucket.
3. rocWMMA bootstrap: got into the right architectural class.
4. gfx11 direct WMMA provider: hand-coded fragment/load/output schedule.

Critical gotchas:

- Prefill FA and decode FA must stay separate.
- rocWMMA helped bootstrap but hid fragment/layout costs.
- The gfx11 accumulator lane mapping is nontrivial. Exact p512 FA tests are
  mandatory after any change.
- Never assume WMMA accumulator fragments are row-major. The gfx11 direct FA
  repair required resolving the actual even/odd interleaved lane mapping; a
  guessed store layout was the source of a real correctness bug cluster.
  Re-establish the lane-to-output-coordinate map whenever changing WMMA
  accumulator type, wave mode, or output path.
- Use F32 accumulation for QK unless a new exact and model-level test proves
  otherwise.

### Routing / MoE selector

Accepted:

- Decode `TOPK_MOE_EARLY_SOFTMAX_NORM` fused provider for the Qwen `nrows=1`
  path.
- Bounded Qwen decode-router ARGSORT support for `ncols=256, nrows<=64`.
  This removes support-query fallback ambiguity and is covered by standalone
  `test-backend-hrx` ARGSORT cases for 8/16/64 rows. In the Qwen decode graph
  this is still consumed by `TOPK_MOE_EARLY_SOFTMAX_NORM`; it does not add
  standalone ARGSORT dispatches.
- Broader prompt ARGSORT remains opt-in because the previous prompt-shape
  ARGSORT path caused late-generation instability.

Rejected:

- Hardcoded exact TopK rows1 kernel. Correct but slower.
- Broadly enabling HRX prompt ARGSORT to remove decode support-query fallback
  logs. It did not improve device time and is not evidence of executed CPU
  work.

## F. Accepted Patterns

- Prefer **shape-exact siblings** over widening a generic provider.
- Keep **prompt and decode gates separate**. A prefill win can destroy decode.
- Use **rollback env vars** for every approximate or narrow default route.
- Keep **Vulkan as a map**, not a law. Port gross dataflow first; local source
  order may need to diverge for HIP/LLVM.
- Prefer **device bucket deltas** to noisy endpoint tok/s for hero kernels.
- Prefer **full-model bucket confirmation** over native microbench-only wins.
  Native HIP harnesses are the right way to inspect a kernel, but they are not
  sufficient evidence for default route changes.
- Prefer **hero-op fusions that remove real data movement**. BF16
  `MUL_MAT -> SET_ROWS` worked because it eliminated a dispatch and an
  intermediate write/read. Fusions that only reshuffle small operations need
  stronger evidence.
- Use **provider expectation tests** for route-sensitive fusions and kernels.
  A passing generic fallback test does not prove the optimized provider ran.
- Treat **decode-specific recurrent fusions** as separate kernels from prefill.
  Attempts to share SSM/GDN state-update fusions across regimes produced fragile
  graph predicates and correctness risk.
- Treat **high VGPR without spills** as a tradeoff, not a failure. Some winning
  kernels are high pressure.
- Treat **spills as serious but not automatically fatal**. Q6 fill-order won
  despite small VGPR spills; profiler evidence decides.
- Use **model-level loop guards** for approximate prompt, routing, FA, and MoE
  work. Exact op tests missed some hidden-state poison.

## G. Rejected Patterns

Avoid repeating these without new evidence:

- "Just vectorize loads" globally. `float4` RHS and aligned q-byte changes often
  raised scalar extraction/VGPR pressure and regressed.
- "Just reduce VGPR." Low-VGPR Q4 SWIGLU duplicated RHS reads and lost.
- "Just match Vulkan tile dimensions." Q6 `128x128`, Q5 `128x64`, and DMMV8
  probes showed that gross shape without matching full schedule can regress.
- "Just force wave64." It helped Q4 prompt MMQ and Q5/Q6 large MMQ, but hurt
  GDN and was neutral/negative elsewhere.
- "Just use Q8_1 everywhere." Q8_1 is key for prompt MMQ, but current decode
  Q8_1/MMVQ is not production-worthy.
- "Tiny kernel body tuning." GET_ROWS `nr1` ATT showed mostly kernarg/global
  wait. Eliminate or fuse tiny dispatches instead.
- "Microbench won, therefore route it." A Q6 exact-K probe improved isolated
  HIP timing and then lost in the full model. Keep the harness result as a
  clue, not a promotion decision.
- "Submission-boundary stabilization." Moving command-buffer boundaries can
  hide or expose bugs, but it is not a correctness fix. All submit boundaries
  must produce correct results.
- "Trust chat only." Random-symbol and late-loop failures required exact op
  export plus deterministic long-generation guards.

## H. Current Scoreboard

Checkpoint from April 13, 2026, W7900/gfx1100, Qwen 35B A3B Q4_K_L.

Device-time methodology:

- HRX: rocprof kernel dispatch sum.
- Vulkan: `GGML_VK_PERF_LOGGER` label sum.

| Regime | Shape | HRX Device | Vulkan Device | HRX Wall | Vulkan Wall |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill | `p512`, `fa=1` | `182.5 ms` | `226.7 ms` | `1462.8 tok/s` | `2312.0 tok/s` |
| Decode | `n64`, `fa=0` | `650.4 ms` | `674.3 ms` | `38.2 tok/s` | `107.4 tok/s` |

Later integration/decode-final notes supersede these wall-clock numbers for the
current branch, but the table remains useful as historical evidence for the
device-vs-runtime split at that checkpoint.

Interpretation:

- HRX visible device time is now roughly Vulkan-class in aggregate.
- Wall-clock remains behind, especially decode, because runtime/host overhead is
  still much worse than Vulkan.
- Aggregates hide bucket imbalance. HRX is ahead in some buckets and still
  behind in TopK/GDN/Q4 ID/GET_ROWS-like tails.

## I. Search Terms

Use these terms with `rg` to pull only relevant context:

- `q8_1_x4`
- `MMQL128`
- `rows2_x16`
- `rows4_k2048_cols1_lds`
- `s128_h32_qk16_tok1_nokda`
- `cluster8_nokda`
- `gfx11_direct`
- `v_wmma`
- `v_dot`
- `wave64`
- `GGML_HRX_DISABLE_FAST_APPROX_PROMPT`
- `qwen_hrx_correctness_gate`
- `hrx_rocprof_scoreboard`
- `GGML_VK_PERF_LOGGER`
- `TOPK_MOE_EARLY_SOFTMAX_NORM`
- `ARGSORT`
