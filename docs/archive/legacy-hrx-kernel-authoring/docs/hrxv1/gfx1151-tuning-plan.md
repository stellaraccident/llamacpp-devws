# HRX v1 gfx1151 Tuning Plan

Date: 2026-06-17

## Purpose

This document is the goal anchor for productionizing the HRX v1 HIP C++ kernel
catalog on `gfx1151` and tuning it to reach or exceed same-machine Vulkan
throughput.

The working direction is to keep the original HRX v1 HIP C++ pipeline, but make
the kernel library behave like a real fusion library:

- many candidate kernels per family;
- CMake/Ninja builds all HIP C++ candidates in parallel;
- route selection and promotion are data driven;
- gfx1100 legacy routes remain the initial accepted baseline;
- gfx1151 gets its own tuning database and promotion evidence;
- odd, tail, and very narrow prefill shapes are tested explicitly before any
  route becomes default.

This is not a replacement of HRX v1 with Loom. HRX2/Loom is prior art for the
catalog shape, tuning discipline, and evidence model.

## How This Differs From The Loom-Centric Plan

Follow the spirit of the HRX2/Loom workflow: evidence first, prior-driven
schedules, explicit shape domains, correctness before promotion, and recorded
rejections. The implementation mechanics differ because this spike stays in
HRX v1 HIP C++.

Differences for this HIP C++ spike:

- Source of truth is HIP C++ plus CMake-built HSACO artifacts, not Loom source
  plus `loom-compile` reports.
- Candidate enumeration happens through catalog JSON and CMake source entries,
  not `func.template` provider expansion.
- Tuners and catalog assemblers may select, validate, and record candidates,
  but they must not compile kernels. Ninja owns compilation.
- Compile evidence comes from HIP compiler output, HSACO metadata, disassembly,
  resource usage, and runtime profile buckets.
- Shape guards remain in C++ initially, then migrate family by family into
  catalog policy. Do not block useful kernel work on a full runtime selector
  rewrite.
- Kernel source can use manual C++ templates, macros, and separate `.hip.cpp`
  files when that makes parallel CMake builds clearer.
- The catalog may contain duplicate-looking HIP sources or route rows if that
  keeps builds explicit and makes tuning evidence easier to attribute.

What should not change:

- do not guess schedules blindly;
- mine HRX1, Vulkan, CUDA/HIP, and HRX2 notes before broad rewrites;
- write a schedule ledger before serious route work;
- bracket candidate pivots around a documented schedule family;
- run focused correctness and route evidence before model benchmarks;
- keep rejected variants and their evidence.

Current plan advice after the first Q4_K/Q6_K probes:

- Use aggregate HRX/Vulkan numbers to rank boulders and accept or reject a
  production route, not as the inner optimization loop.
- Optimize candidate fusion and prompt matmul schedules primarily through
  exported model op rows, route traces, CPU-reference gates, focused timing,
  and HSACO metadata.
- The first useful gfx1151 schedule correction is the opt-in Q4_K/Q6_K x Q8_1
  x4 MMQ64 dense prompt route family:
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1` and
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- Q4 and Q6 MMQ64 are now guarded at `cols >= 32`, not merely "prompt",
  because focused p2 timing rejected the route and p8/p16 were mixed. The
  p32, p33, p512, and p513 rows are the accepted current shape regime.
- Q5 small-row prompt was tested as a selector-only opt-in and rejected for
  default promotion: one focused row improved, but same-runner p512 model A/B
  slightly regressed.
- The first useful F16 attention-chain correction is the opt-in Vulkan-DMMV
  shaped route `GGML_HRX_ENABLE_F16_BATCHED_ROWS2_COLS16_WG32_PROMPT=1`.
  It uses one wave32 workgroup for two rows by sixteen columns, passed p512,
  p33, and p513 focused CPU-reference gates, and improved same-binary HRX p512
  by about 1.31x on top of the Q4/Q6 candidates.
- The current best opt-in stack on Qwen3 30B Q4_K_XL no-FA prefill is still
  below Vulkan at production prompt widths, but Q4_K and Q5_K MoE ID now have
  opt-in grouped Q8_1 x4 candidates: with
  `GGML_HRX_ENABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1` and
  `GGML_HRX_ENABLE_Q5_K_ID_Q8_1_X4_MMQ16_PROMPT=1`, p33 is 0.91x Vulkan, p512
  is 0.53x, and p513 is 0.54x in
  `cache/hrxv1/gfx1151/current-best-q4q5-id-vulkan-r3-20260617-162608/`.
- The wide Q4 dense prompt candidate
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1` is accepted as an opt-in
  gfx1151 route for production-width prefill only. It raises same-run HRX to
  p512 635.9 tok/s and p513 539.1 tok/s in
  `cache/hrxv1/gfx1151/q4-mmql128-model-ab-20260617-165235/`, about 0.56x
  and 0.57x the prior Vulkan rows. It must stay guarded at `cols >= 128`
  because focused p33 A/B is faster on the existing Q4 MMQ64 route.
- The split D128 flash-attention prefill-direct route
  `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1` now covers valid D128 GQA
  prompt shapes beyond Qwen3 H32/HKV4. Broader basket p512 A/B improved newly
  covered rows by `1.16x-1.63x`, and p513 smoke improved representative rows
  by `1.22x-1.60x`. The generalized policy must stay guarded to
  `prompt_tokens >= 128` except for the original H32/HKV4 shape, because Llama
  3.2 3B p33 regressed slightly on prefill-direct. Evidence:
  `cache/hrxv1/gfx1151/fa1-d128-gqa-model-ab-p512-r3-20260617-175433/`,
  `cache/hrxv1/gfx1151/fa1-d128-gqa-focused-odd-tail-20260617-175638/`, and
  `cache/hrxv1/gfx1151/fa1-d128-gqa-policy-sanity-20260617-175858/`.
- Llama 3.1 8B Q8_0 exposed a gfx1151 prompt-matmul policy miss: Q/V/K
  projections with `k=4096, rows=1024, cols>=32` stayed on
  `hrx_mul_mat_vec_q8_0_cols8_f32` while larger Q8_0 prompt rows already used
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`. Focused CPU-reference
  gates passed and the isolated p512 K/V row improved from `2110 us` to
  `546 us` under the packed route. The accepted gfx1151 policy lowers only the
  Q8_0 auto row threshold from 2048 to 1024 while preserving the existing
  `cols >= 32` guard because earlier p31 Q8_1 MMVQ forcing was rejected.
  Model A/B on Llama 3.1 8B Q8_0 improved p512 `201.4 -> 219.0 tok/s`, p513
  `192.0 -> 200.6 tok/s`, and p33 was flat/slightly positive. Evidence:
  `cache/hrxv1/gfx1151/q8_0-kv1024-focused-20260617-180540/`,
  `cache/hrxv1/gfx1151/q8_0-mmvq1024-odd-tail-focused-20260617-181012/`,
  `cache/hrxv1/gfx1151/q8_0-mmvq1024-model-ab-p512-r3-20260617-180920/`,
  and
  `cache/hrxv1/gfx1151/q8_0-mmvq1024-model-ab-odd-tail-r3-20260617-181206/`.
- Qwen3 30B Q6_K exposed a structural MoE prompt coverage miss: HRX had dense
  Q6 prompt routes but no Q6_K `MUL_MAT_ID` route, while Vulkan spent major
  time in Q6_K MoE `MUL_MAT_ID` buckets. The opt-in candidate
  `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1` adds
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32` using the accepted
  Q5 grouped Q8_1 x4 MMQ16 schedule family. Focused CPU-reference gates passed
  and model A/B improved p512 `217.8 -> 509.3 tok/s` and p513
  `220.6 -> 483.9 tok/s`. p33 regressed `123.1 -> 108.9 tok/s`, so the route
  is guarded to production-width prefill with `n_tokens >= 128`. Evidence:
  `cache/hrxv1/gfx1151/q6-mul-mat-id-q8x4-focused-20260617-183451/`,
  `cache/hrxv1/gfx1151/q6-id-model-ab-p512-fa1-r3-20260617-183602/`,
  `cache/hrxv1/gfx1151/q6-id-model-ab-odd-tail-r3-20260617-183942/`, and
  `cache/hrxv1/gfx1151/q6-id-postguard-focused-20260617-184103/`.
- A later threshold-32 Q6 ID experiment confirmed why p33 needs a separate
  medium/narrow policy. Lowering the grouped route threshold from 128 to 32
  eliminated the hidden p33 scheduler split cliff caused by HRX-resident expert
  weights being copied back to CPU, and focused p33/p512 Q6 ID rows passed.
  However, same-runner Qwen3 30B Q6_K p33 still regressed versus the current
  disabled/narrow policy (`105.18 tok/s` vs `112.94 tok/s`) and remained far
  below Vulkan (`182.23 tok/s`). The experiment is rejected as a default route
  policy and recorded in
  `cache/hrxv1/gfx1151/q6-id-threshold32-experiment-20260618-182836/`.
  The useful conclusion is architectural: fix p33 either with a true
  Vulkan-medium Q6 ID schedule or by decoupling expert-weight placement from
  production-width route support; do not force the production grouped route
  into the narrow regime.
- Llama 3.1 8B Q8_0 is now the clearest dense prompt-matmul boulder. The
  wave64-only compile probe for the current Q8_0 packed `128x32` schedule
  (`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X32_WAVE64_PROMPT=1`) passed focused
  CPU-reference rows and selected correctly, but regressed every p512 focused
  row. In particular, `ffn_gate` regressed `21138.7 -> 35777.0 us` and
  `ffn_out` regressed `12100.7 -> 18948.8 us`. Evidence:
  `cache/hrxv1/gfx1151/q8_0-wave64-focused-p512-20260617-185605/`. The route
  is rejected as a wavefront-only fix; the next Q8_0 attempt must change tile
  shape/output ownership and/or shared A staging toward the Vulkan integer-MMQ
  dataflow. Schedule ledger:
  `docs/hrxv1/q8_0-prompt-schedule-ledger.md`.
- The Q8_0 `BM64/BN64` tile/output-ownership pivot
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1` is accepted as an opt-in
  gfx1151 candidate. Focused p33, p512, and p513 CPU-reference gates passed,
  focused p512 total time over the five exported Q8_0 rows improved
  `239.2 ms -> 104.2 ms`, and model A/B on Llama 3.1 8B Q8_0 improved p33
  `92.3 -> 194.5 tok/s`, p512 `209.7 -> 394.2 tok/s`, and p513
  `207.0 -> 378.0 tok/s`. Same-run Vulkan in the same artifact was
  `196.6 tok/s` at p33, `884.0 tok/s` at p512, and `837.2 tok/s` at p513, so
  narrow Q8_0 is nearly at Vulkan but production-width Q8_0 is still around
  `0.45x`. Evidence:
  `cache/hrxv1/gfx1151/q8_0-mmq64x64-focused-p512-20260617-190744/` and
  `cache/hrxv1/gfx1151/q8_0-mmq64x64-model-ab-20260617-191149/`.
  The next Q8_0 work should mine the remaining Vulkan schedule delta, with
  shared A staging and smaller per-lane output/register tiles as the leading
  axes.
- The naive Q8_0 cooperative A+B LDS staging probe
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_AB_PROMPT=1` is rejected. It kept the
  accepted `BM64/BN64` tile and added shared A staging, passed focused p512
  correctness, and selected correctly, but regressed every p512 focused row
  versus direct-A BM64/BN64, with total time `104.9 ms -> 115.3 ms`. Evidence:
  `cache/hrxv1/gfx1151/q8_0-mmq64x64-ab-focused-p512-20260617-191938/`.
  Future A-reuse work should not repeat this spelling; it needs to be coupled
  to a different per-lane output tile/register-tiling strategy.
- The simple Q8_0 smaller-output-tile probe
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ32X64_PROMPT=1` is rejected. It reduced
  the public route VGPR count from 127 to 78 and passed focused p512
  correctness, but regressed total p512 focused time versus BM64/BN64
  `105.7 ms -> 117.0 ms`. Evidence:
  `cache/hrxv1/gfx1151/q8_0-mmq32x64-focused-p512-20260617-192604/`.
  The loss shows that simply halving per-thread columns while doubling row
  workgroups is the wrong local tradeoff for production-width Q8_0.
- The simple Q8_0 column-widening probe
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X64_PROMPT=1` is rejected on compile
  resources before runtime testing. It compiled as wave32 but spilled heavily
  (`vgpr_count=192`, `vgpr_spill_count=47`, private segment 192 bytes), so it
  is not a useful production candidate. Future column-widening work needs a
  different register tile rather than a direct `COLS_PER_THREAD=32` extension.
- Current-best p512/fa1 basket evidence after accepting Q8_0 BM64/BN64 is
  still well below Vulkan overall: eight downloaded rows geomean `0.489x`
  Vulkan in
  `cache/hrxv1/gfx1151/current-best-q8bm64-basket-p512-fa1-r1-20260617-193412/`.
  The worst rows are Llama 3.1 8B Q4_K_M (`0.395x`), Qwen2.5 Coder 7B Q5_K_M
  (`0.396x`), DeepSeek R1 Qwen 14B Q4_K_M (`0.407x`), Qwen3 30B Q6_K
  (`0.427x`), and Llama 3.1 8B Q8_0 (`0.450x`). This makes dense quantized
  prompt matmul schedule quality the next primary boulder. Aggregate numbers
  should only rank that boulder; candidate work should continue at the
  exported-kernel row level with route traces, CPU-reference gates, focused
  timing, HSACO metadata, and Vulkan/HRX1 schedule deltas. The immediate next
  goal is to build the Vulkan oracle capture hooks and use RADV/SPIR-V evidence
  before adding more dense prompt HIP candidates:
  `docs/hrxv1/gfx1151-vulkan-oracle-goal.md`.
- The first Vulkan oracle capture is now available for Llama 3.1 8B Q4_K_M
  `p512/n0/fa1`:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/`.
  It identifies `matmul_q4_k_f32_f16acc_aligned_l` as the dominant dense Q4_K
  prompt pipeline and records a Q4_K Vulkan-vs-HRX schedule comparison in
  `docs/hrxv1/q4k-vulkan-oracle-schedule-ledger.md`. The immediate lesson is
  that matching `BM128/BN128/WG256/WAVE64` is not sufficient: current HRX
  already does that, while RADV uses a much larger no-spill LDS/VGPR budget.
  The next dense prompt candidate must name a staging/dataflow hypothesis from
  that ledger before it is added to the catalog.
- The full first p512/fa1 Vulkan oracle matrix is also captured for the five
  worst rows named in `docs/hrxv1/gfx1151-vulkan-oracle-goal.md`. The
  consolidated schedule ledger is
  `docs/hrxv1/quantized-prompt-vulkan-oracle-ledger.md`. Across Q4_K, Q5_K,
  Q6_K, and Q8_0, the dominant Vulkan dense prompt route uses the same large
  aligned family: `spec=[256,128,128,32,64,64,2,16,16,16,64]`,
  `wg_denoms=[128,128,1]`, `workgroup=256x1x1`, `LDS=22528`, `VGPR=192`,
  and no spills. The next HIP C++ family should target that shared schedule
  directly, with odd/tail gates retained before promotion.
- The MoE expert path is no longer the obvious unsupported-route blocker for
  the active Qwen3 p33/p512/p513 rows. Route tracing showed the default Q4 rows
  selected `hrx_mul_mat_id_q4_k_wg64_f32` with grouped=0/q8_1_x4=0, and Q5 was
  unsupported. Q4 wide-K and Q5 grouped Q8_1 x4 ID candidates now pass focused
  and model A/B gates. Remaining work is basket coverage, target-specific
  default policy, and a fresh boulder ranking from route/device evidence.
- Remaining Q5 prompt shapes and promotion-policy plumbing are still required
  after the MoE path is understood, especially to make proven gfx1151 routes
  target-specific without perturbing gfx1100 legacy behavior.

## Production Target

Primary target:

- machine: current gfx1151 workspace host;
- ROCm: `/srv/vm-shared/rocm/rocm-head`;
- source repo: `sources/llama.cpp`;
- backend: HRX v1 HIP C++ provider;
- baseline: same-machine Vulkan backend from the same llama.cpp source state;
- starting model basket:
  `/srv/vm-shared/projects/llamacpp-devws/shared/models/llamacpp-hrx2-basket-v1`;
- target: HRX v1 steady-state throughput at or above Vulkan on the production
  model basket, without CPU compute fallback or hidden backend contamination.

Parity claims must be made per regime:

- prefill and decode are separate targets;
- flash-attention-on and flash-attention-off paths are separate targets when
  Vulkan performance differs materially;
- cold and steady-state samples must be reported separately;
- JSON `backends` values must be checked before comparing rows.

Models may still be downloading during early work. Start with whichever model
rows are locally available, but label those artifacts as partial-basket
evidence. Partial-basket results can rank boulders, validate route selection,
and reject bad schedules. They cannot promote a broad default route until the
full production basket has arrived and passed the required gates.

Currently visible starting subset:

```text
shared/models/llamacpp-hrx2-basket-v1/
  unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/
    Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf
    Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf
  unsloth__Qwen3-Coder-30B-A3B-Instruct-GGUF/
    Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```

Use these rows immediately for baseline and boulder ranking. As additional
GGUFs finish downloading into the basket, add them to the same benchmark matrix
and mark older subset artifacts as superseded, not invalid.

## Non-Negotiables

- Do not compile HIP C++ kernels inside tuner or assembler helper scripts.
  Candidate `.hip.cpp` files must be declared in CMake and built by Ninja.
- Do not promote a route from full-model tok/s alone. Route trace, focused
  correctness, focused timing, and same-runner model A/B are all required.
- Do not treat gfx1100 route choices as authoritative for gfx1151. Treat them
  as accepted legacy seeds until gfx1151 evidence replaces them.
- Do not broaden a route without odd-size and tail coverage.
- Do not tune prefill and decode as one problem. Their winning schedules are
  often different.
- Do not hide fallback behind a passing benchmark. Inspect provider traces and
  CPU fallback evidence.
- Do not erase rejected variants. Rejections are part of the tuning database.
- On this unified-memory GPU, `amd-smi` process/residency readings are useful
  sanity checks but not proof of kernel throughput or memory pressure. Use
  route traces, focused timing, profile buckets, and same-runner JSON timings
  for tuning decisions.

## Catalog And Tuning Artifacts

The catalog should be split and data driven:

```text
ggml/src/ggml-hrx/catalog/
  metadata.json
  sources.json
  artifacts.json
  families.json
  routes/
  fusions/
  tuning/
    gfx1151/
      shapes.json
      results/
      promotions.json
      rejections.json
```

Route rows should carry at least:

- route id, family, op, source id, artifact id, export name;
- target key, target prefixes, priority, status;
- ABI metadata and workgroup size;
- shape domain and explicit shape guards;
- config axes such as tile sizes, wave policy, vector width, staging policy,
  route grouping, Q8 packing mode, and tail strategy;
- evidence summary and rollback or disable knob when defaulted.

Result rows should carry at least:

- source commit and catalog revision;
- ROCm path and target listing;
- CMake build directory and build flags;
- exact benchmark or test command;
- route trace path proving the selected provider;
- correctness result and error tolerance;
- focused timing result with variance;
- model-level HRX/Vulkan comparison;
- compile report, disassembly, or resource facts when relevant;
- decision: candidate, rejected, accepted, diagnostic, or retired.

Keep the human-readable experiment trail in
`docs/hrxv1/experiments-log.md`. Every baseline, focused sweep, rejection, and
promotion candidate should get a log entry with commands, artifact paths,
correctness, timing, decision, and the concrete lesson learned.

Keep family-specific schedule ledgers under `docs/hrxv1/`. The initial prefill
ledger is `docs/hrxv1/qwen3-30b-q4xl-prefill-ledger.md`.

## Baseline Procedure

1. Verify environment.

   Record:

   - `readlink -f rocm`;
   - ROCm compiler version;
   - `rocminfo` target facts;
   - `ldd` for HRX and Vulkan benchmark binaries;
   - CMake cache for HRX and Vulkan builds.

2. Build HRX v1 and Vulkan from the same source state.

   HRX must use:

   ```bash
   -DGGML_HRX=ON
   -DGGML_HRX_AMDGPU_TARGETS=gfx1151
   -DGGML_HRX_ROCM_PATH="$PWD/rocm"
   ```

   HIP candidates must appear as Ninja build edges, one HSACO per
   source/target pair.

3. Run the baseline basket.

   If only part of the model basket is downloaded, run the available subset and
   save it under a path that makes the subset explicit. Re-run the same
   baseline procedure as each missing model becomes available. Do not compare
   subset runs as if they were full-basket parity results.

   Capture for each row:

   - HRX `llama-bench` JSON;
   - Vulkan `llama-bench` JSON;
   - HRX provider or route trace;
   - HRX profile dispatch buckets;
   - Vulkan perf logger rows;
   - cold and steady-state summary.

   GPU process and utilization telemetry must use `amd-smi`, not `rocm-smi`.
   This host is a unified-memory GPU, so memory counters and residency need
   careful interpretation. Treat `amd-smi` memory output as a sanity signal for
   process residency, pressure, and gross leaks, not as a standalone VRAM-style
   proof of hot working-set size. A process can remain resident on the GPU
   after a benchmark, utilization can fluctuate normally, and reported memory
   can reflect several layers of unified-memory state. Promotion evidence must
   still come from backend JSON, route traces, focused op timings, profile or
   perf buckets, correctness gates, and same-runner A/B throughput.

   For each missing model, record:

   - expected model path;
   - unavailable reason, such as still downloading;
   - benchmark rows skipped because of that model;
   - date when the row should be refreshed.

4. Rank boulders.

   Rank by device time and dispatch count, not only wall time. Classify each
   boulder as one of:

   - missing route;
   - weak schedule family;
   - bad shape guard;
   - CPU fallback;
   - copy/contiguous traffic;
   - runtime submission overhead;
   - benchmark or library contamination.

## Candidate Gate

Before editing or promoting a serious route, fill this block in the relevant
result artifact:

```markdown
### Candidate Gate

- Production target:
- Baseline command:
- Variant command:
- Same-runner comparison method:
- Route trace path:
- Scheduler/per-op trace path:
- Focused CPU-reference command:
- Compile report path:
- Target listing path:
- Prior-art schedule source:
- Odd-size and tail gate:
- Promotion rule:
```

If any line is unknown, the work is exploratory. It may produce probes and
rejections, but it must not produce a default route.

## Prior-Driven Kernel Workflow

For each hot family:

1. Mine priors before coding.

   Use HRX1 HIP C++, Vulkan shaders, CUDA/HIP llama.cpp kernels, and HRX2/Loom
   notes where useful.

2. Write a schedule ledger.

   Record:

   - source path and symbol;
   - shape regime;
   - tile/workgroup/subgroup shape;
   - wave32 or wave64 policy;
   - lane ownership and per-lane outputs;
   - vector and packed load widths;
   - quantization and packing layout;
   - dot, WMMA, or ALU primitive and signedness;
   - LDS/shared-memory staging;
   - barrier cadence;
   - reduction and writeback policy;
   - resource facts: VGPR, SGPR, LDS, spills, occupancy;
   - known win, regression, or constraint.

3. Define candidate pivots.

   Valid pivots include:

   - tile shape;
   - wave size;
   - vector width;
   - Q8_1 packing mode;
   - A-side or B-side staging;
   - LDS layout;
   - unroll depth;
   - output ownership;
   - route grouping;
   - full-tile versus tail tile strategy.

4. Add candidates to the catalog and CMake.

   The tuner selects from built candidates. It does not compile them.

5. Run focused sweeps.

   A sweep row must prove:

   - intended route selected;
   - CPU-reference correctness passed;
   - no provider fallback;
   - focused timing is better or diagnostically useful;
   - generated ISA/resource facts are plausible for the schedule.

6. Promote only after same-binary model A/B.

   Promotion requires a production-shape win and no regression on the required
   boundary matrix.

## Odd-Size And Tail Gate

Odd sizes are a production requirement, especially for prefill. Prior HRX work
needed many specialty routes and parameters for odd, tail, and very narrow
prefill shapes. A gfx1151 route is not production-ready until it has explicit
coverage for those cases.

Every prefill candidate must declare its tail strategy:

- exact full-tile only;
- guarded tail in the same kernel;
- separate tail kernel;
- separate very-narrow route;
- fallback to an older safe route;
- unsupported, with an explicit guard.

The default preferred pattern is full-tile fast path plus isolated edge/tail
handling. Do not route every full-tile access through generalized guarded
helpers if that perturbs the hot path or hides memory bugs.

Minimum prefill boundary matrix:

```text
Prompt/token columns:
  1, 2, 3, 4, 7, 8, 15, 16, 17,
  31, 32, 33, 63, 64, 65,
  127, 128, 129, 255, 256, 257,
  511, 512, 513, 767, 768, 769,
  1023, 1024, 1025

K dimensions:
  exact tile K,
  K - one vector group,
  K + one vector group,
  model-derived K values.

Rows/expert routes:
  1, 2, 3, 4, 7, 8, 16,
  model-derived MoE route densities,
  empty or near-empty route groups when applicable.

Batch and microbatch:
  p513 with ub >= 513 for one graph,
  p513 with ub 512 to expose residual graph behavior,
  p31/p32/p33 and p511/p512/p513 for scheduler boundaries.
```

Minimum decode boundary matrix:

```text
Tokens/columns:
  n = 1, 2, 3, 7, 8, 16, 32, 64

Context/KV:
  0, 1, 2, 15, 16, 17,
  511, 512, 513,
  8191, 8192, 8193 where route policy allows.

Attention:
  GQA ratios present in the model basket,
  mask and no-mask cases,
  sink and no-sink cases when supported.
```

Odd-size tests must include both focused backend-op rows and model-level smoke
rows. A candidate that passes sampled kernel output but corrupts a later op is
a failed candidate.

## Initial gfx1151 Worklist

1. Catalog mechanics.

   - Keep split JSON catalog assembled at build time.
   - Preserve CMake/Ninja HIP compilation for all candidates.
   - Add gfx1151 tuning result files and promotion metadata.
   - Make runtime route selection consume catalog policy for one family before
     expanding to all families.

2. Q4_K/Q5_K/Q6_K prompt matmul.

   - Start from Vulkan and HRX1 packed-MMQ priors.
   - Sweep tile shape, wave policy, Q8_1 packing, vector width, and staging.
   - Require odd prefill matrix before broad promotion.

3. Attention path.

   - Compare Vulkan flash-attention-on and flash-attention-off rows.
   - Decide whether HRX v1 needs a stronger flash-attention route, a split
     decode route, or an unfused-chain cleanup.
   - Test long KV and odd prompt boundaries explicitly.

4. MoE and SWIGLU fusions.

   - Reuse route grouping and packed Q8 priors.
   - Treat sparse, empty, and narrow route groups as correctness boundaries.
   - Do not promote the current Q4_K `MUL_MAT_ID` route for p512: focused
     timing is much slower than Vulkan. Use the opt-in wide-K grouped Q8_1 x4
     route as the Q4 candidate for Qwen3-style `k=2048/768` expert rows while
     Q5 support and broader coverage are developed.
   - Use `GGML_HRX_TRACE_ROUTES=1` and
     `GGML_HRX_EXPECT_MUL_MAT_ID_PROVIDER=<symbol>` for every focused ID route
     A/B. The current Qwen3 p512 rows select
     `hrx_mul_mat_id_q4_k_wg64_f32` by default and
     `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32` under
     `GGML_HRX_ENABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`.
   - Q5_K `MUL_MAT_ID` now has an opt-in grouped Q8_1 x4 route for the active
     Qwen3 p33/p512/p513 shapes:
     `GGML_HRX_ENABLE_Q5_K_ID_Q8_1_X4_MMQ16_PROMPT=1`.
   - Keep route-density histograms in the artifact. Real Qwen p512 routing is
     broad and sparse, so dense-per-expert spike priors are not sufficient.

5. Copy, contiguous, get/set rows, and small fusions.

   - Fuse only when it removes meaningful traffic or dispatch overhead.
   - Preserve route traces proving dispatch reduction and no hidden fallback.

6. Decode runtime overhead.

   - Keep submission batching, staging reuse, and stream-ordered transfers in
     the evidence path.
   - Do not spend kernel effort on decode before ruling out runtime overhead
     buckets.

## Promotion Rule

A gfx1151 route can become default only when all are true:

- focused CPU-reference gate passes;
- route trace proves the intended route selected;
- no CPU compute fallback;
- focused timing beats the current HRX route for the declared shape class;
- same-binary model A/B beats or matches the current HRX route;
- Vulkan comparison is at parity or the route materially closes the top
  remaining gap;
- odd-size and tail matrix passes for the declared shape domain;
- rollback or disable knob exists for risky approximate or shape-specialized
  routes;
- result row is recorded under the gfx1151 tuning database.

Accepted gfx1151 routes should not silently replace gfx1100 policy. gfx1100
legacy routes remain accepted for that target until separately retuned.

## Current Open Questions

- Which model basket is authoritative for HRX v1 gfx1151 parity?
- Which Vulkan build directory is the canonical same-source baseline?
- Should ROCWMMA-dependent routes remain excluded with `rocm-head`, or should a
  compatible ROCWMMA header/package be added to the custom ROCm install?
- Which family should be the first runtime catalog-policy migration target:
  Q4_K prompt matmul, attention, or MoE/SWIGLU?
- What threshold defines "exceed parity": steady-state median, p50/p95, or
  worst-row no-regression across the basket?
