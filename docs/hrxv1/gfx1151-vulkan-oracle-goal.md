# HRX v1 gfx1151 Vulkan Oracle Goal

Date: 2026-06-17

## Goal

Before adding more HRX v1 HIP C++ dense prompt-matmul candidates, build the
Vulkan oracle capture path and use it to drive schedule ports from same-machine
RADV/Vulkan evidence.

The objective is to stop optimizing from aggregate HRX/Vulkan token rates or
local HIP schedule guesses. Aggregate basket results identify the boulder; the
Vulkan oracle identifies the schedule to clone or intentionally deviate from.

## Current Motivation

The current-best HRX v1 gfx1151 stack reaches only `0.489x` Vulkan geomean on
the downloaded p512/fa1 basket after accepting the Q8_0 BM64/BN64 route.

Worst rows:

- Llama 3.1 8B Q4_K_M: `0.395x` Vulkan;
- Qwen2.5 Coder 7B Q5_K_M: `0.396x` Vulkan;
- DeepSeek R1 Qwen 14B Q4_K_M: `0.407x` Vulkan;
- Qwen3 30B Q6_K: `0.427x` Vulkan;
- Llama 3.1 8B Q8_0: `0.450x` Vulkan.

This points at dense quantized prompt matmul schedule quality. The Q8_0
BM64/BN64 route proved that schedule shape matters, but the rejected wave64,
A+B LDS staging, BM32/BN64, and BM128/BN64 probes show that local pivots are
not enough.

## Required Instrumentation

Add Vulkan backend hooks equivalent to the workflow in:

```text
cache/radv-spirv-dump-guide-20260617.md
```

The active `sources/llama.cpp` branch currently has `GGML_VK_PERF_LOGGER`, but
does not yet have the required oracle hooks.

Required environment variables:

- `GGML_VK_TRACE_JSONL`: write JSONL rows for pipeline creation, dispatch,
  graph/node metadata, bindings, push constants, workgroups, tensor shapes, and
  optional joined timing.
- `GGML_VK_TRACE_SPV_DIR`: write one raw `.spv` file per unique created Vulkan
  pipeline variant.
- `GGML_VK_TRACE_RADV_PIPELINE_LABELS=1`: bracket `vkCreateComputePipelines`
  stderr with `GGML_VK_RADV_PIPELINE_BEGIN {...}` and
  `GGML_VK_RADV_PIPELINE_END {...}` labels so `RADV_DEBUG=shaders,shaderstats`
  output can be split by exact pipeline identity.

Pipeline identity must include at least:

- pipeline name and entrypoint;
- generated shader source label;
- SPIR-V hash and byte size;
- specialization constants;
- push constant size;
- parameter count;
- workgroup denominators;
- subgroup size, full-subgroup requirement, and robustness policy.

Kernel name alone is not sufficient.

## First Capture Matrix

Capture Vulkan oracle artifacts for p512/fa1 first:

- Llama 3.1 8B Q4_K_M;
- Qwen2.5 Coder 7B Q5_K_M;
- DeepSeek R1 Qwen 14B Q4_K_M;
- Qwen3 30B Q6_K;
- Llama 3.1 8B Q8_0.

Then add odd and tail prefill rows for the selected kernel family before any
route promotion:

- p33;
- p513;
- any exported odd/narrow backend-op rows already present for that model.

Decode is a separate regime and should get separate captures after the prefill
boulder is understood.

## Artifact Contract

Each oracle capture directory should contain:

- `command.txt`;
- `stdout.json`;
- `stderr.log` or `radv-stderr.log`;
- `vulkan.jsonl`;
- `spv/*.spv`;
- `spvasm/*.spvasm`;
- `radv/full/*.radv.txt`;
- `radv/isa/*.amdgcn.txt`;
- `radv/stats/*.stats.txt`;
- `inventory/pipeline_inventory.json`;
- `inventory/dispatch_signature_inventory.json`;
- `inventory/normalized_shape_inventory.json`;
- `inventory/dispatches_full.jsonl`;
- `inventory/kernel_inventory.md`.

Use a fresh output directory and disable the Mesa shader cache for compiler
dump runs:

```bash
MESA_SHADER_CACHE_DISABLE=true
RADV_DEBUG=shaders,shaderstats
GGML_VK_TRACE_JSONL="$OUT/vulkan.jsonl"
GGML_VK_TRACE_SPV_DIR="$OUT/spv"
GGML_VK_TRACE_RADV_PIPELINE_LABELS=1
GGML_VK_PERF_LOGGER=1
```

The committed helper for this procedure is:

```bash
sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py
```

It runs `llama-bench`, writes `command.txt`, `stdout.json`,
`radv-stderr.log`, raw SPIR-V, SPIR-V asm, split RADV output, and the
inventory files.

## Porting Rule

For each high-time Vulkan pipeline, open artifacts in this order:

1. dispatch signature and tensor shape;
2. SPIR-V assembly;
3. RADV full compiler output;
4. RADV AMDGCN ISA;
5. RADV shader stats;
6. matching HRX HIP source and built HSACO metadata/ISA.

Before coding a new HRX candidate, write a schedule-ledger row comparing Vulkan
against the current HRX provider:

- tile shape and workgroup geometry;
- subgroup/wavefront size;
- lane ownership;
- per-lane output count;
- vector load width and alignment assumptions;
- packed Q8_1 layout;
- integer dot primitive and signedness;
- A/B staging and LDS footprint;
- barrier and wait strategy;
- reduction and writeback policy;
- VGPR/SGPR/LDS/spill facts;
- shape regime where the Vulkan pipeline wins.

The first HRX candidate for a boulder should preserve the winning Vulkan
dataflow as closely as HIP C++ allows. Benchmark-driven deviations are allowed
only after the clone is correct and the ISA/resource comparison explains the
remaining gap.

## Current Checkpoint

The first oracle capture is complete:

```text
cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/
```

It captured Llama 3.1 8B Q4_K_M `p512/n0/fa1` on Vulkan0 with
`backends=Vulkan`, 16 pipeline compile rows, 517 dispatch rows, 16 SPIR-V
files, 16 SPIR-V asm files, 16 split RADV ISA/stats blocks, and 27 normalized
shape signatures.

Top dense quantized prompt pipelines:

- Q4_K: `matmul_q4_k_f32_f16acc_aligned_l`, hash `0x5666175250529efb`,
  190 dispatches, spec `[256,128,128,32,64,64,2,16,16,16,64]`,
  RADV stats `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills.
- Q6_K: `matmul_q6_k_f32_f16acc_aligned_l`, hash `0x6eebdfb4c3043b23`,
  31 dispatches, same spec family, RADV stats `SGPR=108`, `VGPR=192`,
  `LDS=22528`, no spills.

The Q4_K schedule comparison against the current HRX v1 provider is recorded
in:

```text
docs/hrxv1/q4k-vulkan-oracle-schedule-ledger.md
```

The leading delta is not the headline `BM128/BN128/WG256/WAVE64` tuple; the
current HRX Q4 route already matches that family. The exposed delta is resource
and dataflow detail: RADV uses far more LDS/VGPR budget without spilling,
while current HRX stages one K-quant block step at a time with 8192 bytes LDS.
The next HIP candidate should test a named staging/dataflow hypothesis from the
ledger, not another blind tile-shape sweep.

The full first p512/fa1 matrix is also captured:

| Row | Artifact |
| --- | --- |
| Llama 3.1 8B Q4_K_M | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/` |
| Qwen2.5 Coder 7B Q5_K_M | `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p512-fa1-20260617-200349/` |
| DeepSeek R1 Qwen 14B Q4_K_M | `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p512-fa1-20260617-200426/` |
| Qwen3 30B Q6_K | `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/` |
| Llama 3.1 8B Q8_0 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/` |

The consolidated cross-quant schedule ledger is:

```text
docs/hrxv1/quantized-prompt-vulkan-oracle-ledger.md
```

That ledger shows the dominant Q4_K, Q5_K, Q6_K, and Q8_0 production prompt
pipelines all use the same Vulkan large aligned family:
`spec=[256,128,128,32,64,64,2,16,16,16,64]`,
`wg_denoms=[128,128,1]`, `workgroup=256x1x1`, `LDS=22528`,
`VGPR=192`, and no spills.

Odd/tail oracle captures for the selected Llama 3.1 8B Q4_K_M family are also
complete:

| Row | Artifact | Vulkan route |
| --- | --- | --- |
| p33/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p33-fa1-20260617-200738/` | medium aligned `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills |
| p513/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p513-fa1-20260617-200751/` | large aligned `spec=[256,128,128,32,64,64,2,16,16,16,64]`, fifth workgroup column for the tail, `LDS=22528`, `VGPR=192`, no spills |

This confirms the narrow p33 route and production-width tail route should be
handled as separate policy regimes.

The same odd/tail oracle coverage is now complete for the active Llama 3.1 8B
Q8_0 boulder:

| Row | Artifact | Vulkan route |
| --- | --- | --- |
| p33/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-194300/` | medium aligned `matmul_q8_0_f32_f16acc_aligned_m`, `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills |
| p513/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-193949/` | large aligned `matmul_q8_0_f32_f16acc_aligned_l`, `spec=[256,128,128,32,64,64,2,16,16,16,64]`, fifth workgroup column for the tail, `LDS=22528`, `VGPR=192`, no spills |

The Q8_0 route policy should follow the same split: p33 is a medium/narrow
regime, while p512/p513 are large aligned production-width routes. The next
Q8_0 large-route candidate must mechanically target the remaining RADV/HIP ISA
delta, especially cooperative-matrix global store/lane ownership, before it is
allowed to enter model-level promotion.

The earlier Q8_0 p33 artifact
`cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-continued/`
reported `backends=Vulkan,HRX` and is superseded by the clean Vulkan-only p33
capture above.

The same odd/tail oracle coverage is now complete for the active Qwen3 30B
Q6_K boulder:

| Row | Artifact | Vulkan route |
| --- | --- | --- |
| p33/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p33-fa1-20260618-061613/` | medium aligned `matmul_q6_k_f32_f16acc_aligned_m`, `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills |
| p513/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p513-fa1-20260618-061619/` | large aligned `matmul_q6_k_f32_f16acc_aligned_l`, `spec=[256,128,128,32,64,64,2,16,16,16,64]`, fifth workgroup column for the tail, `LDS=22528`, `VGPR=192`, no spills, plus `split_k_reduce` tail reductions |

The p33 capture confirms Q6 narrow prefill is the same medium-route regime seen
for Q4_K and Q8_0. The p513 capture adds a stronger requirement: Vulkan keeps
the large aligned Q6 route for production-width tails and also emits
`split_k_reduce` dispatches for tail reduction buckets. HRX Q6 work should
therefore be judged first by exact schedule convergence on p512 and p513 large
rows, with p33 kept as a separate medium/narrow policy row.

The same odd/tail oracle coverage is now complete for the active Qwen2.5 Coder
7B Q5_K_M boulder:

| Row | Artifact | Vulkan route |
| --- | --- | --- |
| p33/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-20260618-063510/` | medium aligned `matmul_q5_k_f32_f16acc_aligned_m`, `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills |
| p513/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p513-fa1-20260618-063522/` | large aligned `matmul_q5_k_f32_f16acc_aligned_l`, `spec=[256,128,128,32,64,64,2,16,16,16,64]`, fifth workgroup column for the tail, `LDS=22528`, `VGPR=192`, no spills, plus `split_k_reduce` tail reductions |

This closes the first odd/tail oracle matrix for Q4_K, Q5_K, Q6_K, and Q8_0.
The repeated pattern is now evidence, not a guess: p33 is a medium aligned
schedule, while p512/p513 are large aligned schedules, and p513 can require a
separate split-K reduction path. HRX route policy should encode that split
explicitly instead of treating prompt matmul as one aggregate performance row.

The first Q6 p33 medium-route promotion is complete:

- route: `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`;
- default policy: gfx1151 Q6_K prompt rows with `16 <= cols <= 64`;
- rollback:
  `GGML_HRX_DISABLE_Q6_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_PROMPT=1`;
- artifact:
  `cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/`;
- focused p33 improvement: Vcur `678.27 -> 397.77 us`, ffn_out
  `3175.03 -> 2239.18 us`, result_output `18046.03 -> 11744.51 us`;
- Qwen3 30B Q6_K p33/fa1 model A/B: `96.68 -> 109.61 tok/s`.

This is a schedule-led odd-size lift from the Vulkan medium route, not an exact
RADV parity claim. The HIP route matches the medium-route LDS footprint but
still differs in visible WMMA/load/store shape, so large-route p512/p513 work
must continue to mechanically close the cooperative-matrix schedule delta.

## Direct Basket KPI Checkpoint

The captured Vulkan-oracle rows below remain useful as schedule evidence, but
they are no longer the throughput KPI. The direct same-source, same-machine
`llama-bench` basket with JSON-confirmed backend identity is the KPI. The
pre-Q6-ID baseline was:

```text
cache/hrxv1/gfx1151/basket-head-full-commitaligned-20260618-200300/
```

This artifact was built from `sources/llama.cpp` commit `4e2d724d1` for both
HRX and Vulkan. It reports `backends=HRX` for HRX rows and `backends=Vulkan`
for Vulkan rows. The full downloaded basket result before the Q6 ID default
was:

- average geomean HRX/Vulkan: `0.433x`;
- steady-state geomean HRX/Vulkan: `0.422x`;
- rows below parity: `24/24`.

Worst steady rows in that direct basket are dominated by Qwen3 MoE prompt
matmul, especially Q6_K and Q4_K_XL:

| Row | HRX steady tok/s | Vulkan steady tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| Qwen3 30B Q6_K p512 | `206.594` | `984.058` | `0.210x` |
| Qwen3 30B Q6_K p513 | `202.137` | `942.842` | `0.214x` |
| Qwen3 30B Q4_K_XL p512 | `281.452` | `1190.790` | `0.236x` |
| Qwen3 30B Q4_K_XL p513 | `279.168` | `1134.165` | `0.246x` |

Therefore the active parity mission was not complete. Vulkan-oracle captures
are now schedule priors; direct same-binary basket rows are the KPI. The
immediate accepted Q6_K structural fix was defaulting the existing grouped Q6
`MUL_MAT_ID` provider on gfx1151 for `n_tokens >= 32`:

- route: `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`;
- default policy: gfx1151 Q6_K `MUL_MAT_ID`, `k % 256 == 0`,
  `rows % 64 == 0`, `n_ids == 8`, `n_tokens >= 32`;
- rollback: `GGML_HRX_DISABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1`;
- focused gate:
  `cache/hrxv1/gfx1151/q6-id-threshold32-default-regate-20260618-201739/`;
- model A/B:
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold32-default-20260618-201756/`
  versus
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold32-rollback-20260618-201820/`.

The accepted Q6 policy moved the Qwen3 30B Q6_K p512/p513 rows from about
`0.23x` Vulkan to about `0.60x` Vulkan:

| Row | Default HRX | Rollback HRX | Vulkan | Default/Vulkan |
| --- | ---: | ---: | ---: | ---: |
| p33 | `97.017` | `114.288` | `170.298` | `0.570x` |
| p512 | `588.052` | `238.257` | `984.058` | `0.598x` |
| p513 | `573.081` | `215.994` | `942.842` | `0.608x` |

The three-row steady geomean improved from `0.334x` Vulkan with rollback to
`0.591x` Vulkan with the default route. This is a production improvement, not
parity. The known cost is p33: the grouped ID route is still slower than
rollback on the narrow row, but a `n_tokens >= 64` policy reproduced the severe
`~15 tok/s` placement cliff while selecting no visible Q6 ID route. The next
Q6 work should either fix expert-weight placement so narrow declined routes do
not split badly, or build a true Vulkan-medium Q6 ID schedule for p33.

The clean post-promotion direct basket is:

```text
cache/hrxv1/gfx1151/basket-after-q6-id-default-commitaligned-20260618-203035/
```

Both HRX and Vulkan rows report build commit `07167d398`. The full downloaded
basket result after the Q6 ID default is:

- average geomean HRX/Vulkan: `0.485x`;
- steady-state geomean HRX/Vulkan: `0.478x`;
- rows below parity: `24/24`.

The Q6_K p512/p513 rows are no longer the worst rows:

| Row | HRX steady tok/s | Vulkan steady tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| Qwen3 30B Q4_K_XL p512 | `281.264` | `1202.565` | `0.234x` |
| Qwen3 30B Q4_K_XL p513 | `274.572` | `1129.760` | `0.243x` |
| Qwen3-Coder 30B Q4_K_M p512 | `390.842` | `1126.665` | `0.347x` |
| Qwen3-Coder 30B Q4_K_M p513 | `381.238` | `1059.255` | `0.360x` |
| Qwen2.5 Coder 7B Q5_K_M p33 | `136.317` | `359.810` | `0.379x` |

The next parity boulder should be the Qwen3/Qwen3-Coder Q4_K MoE prompt path:
use the already captured Vulkan `matmul_id_subgroup_q4_k_f32_f16acc_aligned_m`
and dense Q4_K oracle schedules, then mechanically compare them against the
current `hrx_mul_mat_id_q4_k_wg64_f32`/B-quad dense routes before adding more
HIP C++ candidates.

Strict generated-catalog artifact validation currently has a pre-existing
hygiene issue: `hrx_catalog.json` references
`flash_attn_ext_f32_f16_prefill_wmma16.hsaco`, while CMake excludes
`flash_attn_ext_f32_f16_prefill_wmma16.hip.cpp` from generated embedding. The
catalog structure validates, and the Q6 ID HSACO exists, but a future catalog
hygiene pass should reconcile excluded sources with metadata so
`validate_hrx_catalog.py --require-artifacts` can be used as a hard gate.

## Current Parity Checkpoint

This section is retained as captured Vulkan-oracle schedule history. It is
superseded for KPI decisions by the direct basket checkpoint above.

The current same-build HRX scoreboard after accepting Q5_K B-quad large-tail
policy is:

```text
cache/hrxv1/gfx1151/current-scoreboard-after-q5tail-20260618-151637/
```

Clean Vulkan-oracle comparisons in that artifact show:

| Row | HRX tok/s | Vulkan tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B Q8_0 p512 | `333.327` | `423.308` | `0.787x` |
| DeepSeek R1 Qwen 14B Q4_K_M p512 | `245.593` | `290.539` | `0.845x` |
| Qwen3 30B Q6_K p512 | `175.810` | `206.049` | `0.853x` |
| Qwen3 30B Q6_K p513 | `174.771` | `203.362` | `0.859x` |
| DeepSeek R1 Qwen 14B Q4_K_M p513 | `212.575` | `246.390` | `0.863x` |
| Llama 3.1 8B Q4_K_M p512 | `443.915` | `371.807` | `1.194x` |
| Llama 3.1 8B Q4_K_M p513 | `409.247` | `347.622` | `1.177x` |
| Qwen2.5 Coder 7B Q5_K_M p512 | `454.516` | `365.256` | `1.244x` |
| Qwen2.5 Coder 7B Q5_K_M p513 | `407.551` | `342.041` | `1.192x` |

Interpretation:
the original goal motivation numbers are stale for the current source state.
Q4_K and Q5_K p512/p513 are now at or above the captured Vulkan oracle rows.
This scoreboard predates the latest Q8_0 recheck, so its Q8_0 row is no longer
the active gap. The clean same-runner Q8_0 recheck at source `4e2d724d1`
showed:

| Row | HRX tok/s | Vulkan tok/s | HRX/Vulkan | HRX route |
| --- | ---: | ---: | ---: | --- |
| Llama 3.1 8B Q8_0 p33 | `203.790` | `44.606` | `4.569x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32` |
| Llama 3.1 8B Q8_0 p512 | `458.352` | `394.089` | `1.163x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` |
| Llama 3.1 8B Q8_0 p513 | `420.915` | `399.650` | `1.053x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` |

Subsequent current-HEAD rechecks at source `4e2d724d1` also superseded the
older DeepSeek/Q6 gaps for the captured p512/p513 oracle matrix:

| Row | HRX tok/s | Vulkan tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B Q4_K_M p512 | `660.925` | `371.807` | `1.778x` |
| Llama 3.1 8B Q4_K_M p513 | `579.520` | `347.622` | `1.667x` |
| Qwen2.5 Coder 7B Q5_K_M p512 | `606.467` | `365.256` | `1.660x` |
| Qwen2.5 Coder 7B Q5_K_M p513 | `518.754` | `342.041` | `1.517x` |
| DeepSeek R1 Qwen 14B Q4_K_M p512 | `376.315` | `290.539` | `1.295x` |
| DeepSeek R1 Qwen 14B Q4_K_M p513 | `303.596` | `246.390` | `1.232x` |
| Qwen3 30B Q6_K p512 | `221.641` | `206.049` | `1.076x` |
| Qwen3 30B Q6_K p513 | `213.658` | `203.362` | `1.051x` |
| Llama 3.1 8B Q8_0 p512 | `458.352` | `394.089` | `1.163x` |
| Llama 3.1 8B Q8_0 p513 | `420.915` | `399.650` | `1.053x` |

The ten-row p512/p513 geomean is `1.323x` Vulkan. The current HEAD p33 exact
matrix is also above Vulkan:

| Row | HRX tok/s | Vulkan tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B Q4_K_M p33 | `177.656` | `36.504` | `4.867x` |
| DeepSeek R1 Qwen 14B Q4_K_M p33 | `108.931` | `33.010` | `3.300x` |
| Qwen3 30B Q6_K p33 | `112.305` | `14.958` | `7.508x` |
| Qwen2.5 Coder 7B Q5_K_M p33 | `136.513` | `36.787` | `3.711x` |
| Llama 3.1 8B Q8_0 p33 | `203.790` | `44.606` | `4.569x` |

This is a parity checkpoint for the captured Vulkan-oracle rows, not a reason
to stop validating. Continue with full-basket reruns as more models finish
downloading, focused backend-op correctness for any touched route, and more
odd/narrow/tail rows before broadening any selector. Aggregate/model rows
remain boulder ranking and promotion guardrails; candidate work should remain
focused on backend-op rows, route traces, and schedule/ISA evidence.

## 2026-06-21 Current Basket Refresh

After accepting the Q5 motif192 ASMWAIT small-projection route in
`sources/llama.cpp` commit
`4f30f87cd hrx: promote q5 motif192 asmwait smallproj`, the downloaded basket
was re-run with:

```bash
python3 tools/hrxv1_basket_benchmark.py \
  --tag basket-current-4f30f87cd-r1 \
  --cases p33,p512,p513 \
  --backends hrx,vulkan \
  --repetitions 1 \
  --flash-attn 1 \
  --timeout 1200
```

Artifact:

```text
cache/hrxv1/gfx1151/basket-current-4f30f87cd-r1/
```

Audit:

- all rows exited status `0`;
- HRX rows reported HRX backend labels;
- Vulkan rows reported Vulkan backend labels;
- HRX fallback lines: `0`.

Result:

- average geomean HRX/Vulkan: `0.603x`;
- steady geomean HRX/Vulkan: `0.603x`;
- rows below parity: `23/24`.

Worst current rows:

| Row | HRX tok/s | Vulkan tok/s | HRX/Vulkan | Top HRX route |
| --- | ---: | ---: | ---: | --- |
| Qwen3 30B Q6_K p33 | `49.189` | `169.685` | `0.290x` | `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32` |
| Llama 3.1 8B Q8_0 p512 | `437.947` | `885.747` | `0.494x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bk2_wave64_chunk8_wg256_f32` |
| Llama 3.1 8B Q8_0 p513 | `422.585` | `816.090` | `0.518x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` |
| Qwen2.5 Coder 7B Q5_K_M p513 | `533.129` | `967.391` | `0.551x` | `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_wg256_f32` |
| Qwen3 30B Q6_K p512 | `557.366` | `1009.853` | `0.552x` | `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32` |

Interpretation:

The older parity-looking checkpoint is stale for current KPI decisions. The
same-run downloaded basket is clean and still materially below Vulkan. The next
production boulder is Qwen3 30B Q6_K p33.

However, the Q6 p33 route history already rejects the obvious local pivots:

- H4LOAD is the accepted floor, but its static shape is still only `8` WMMA,
  `20` LDS reads, and `16` global stores, versus the RADV medium oracle's
  `16` WMMA, `48` `ds_load_b64`, `96` `buffer_store_b32`, `64`
  `ds_store_b16`, and `64` `ds_load_u16_d16`.
- Padladder-expwait transferred the explicit first-WMMA wait ladder and passed
  focused correctness, but regressed same-runner focused timing.
- RADV96 duplicate-output transferred the 96-store surface and passed focused
  correctness, but also regressed same-runner focused timing.
- Padladder faststage/mixedstage bench controls preserve more RADV-like issue
  and selected-half staging, but bench timing was still about `1.30x-1.34x`
  slower than accepted VK64.

Therefore the next Q6 p33 source change should not be another selector-only,
prefetch, bufferstore, expwait, duplicate-output, faststage, or mixedstage
replay. It needs a new lower-cost selected-output ownership/store primitive,
or a different dataflow that can beat H4LOAD in focused p33 backend-op timing
while preserving correctness and route evidence.

DeepSeek R1 Qwen 14B Q4_K_M odd/tail oracle coverage is now also captured:

| Row | Artifact | Vulkan route |
| --- | --- | --- |
| p33/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p33-fa1-20260618-192733/` | medium aligned Q4/Q6 routes: `matmul_q4_k_f32_f16acc_aligned_m` and `matmul_q6_k_f32_f16acc_aligned_m`, `spec=[128,64,64,32,64,32,2,16,16,16,64]`, one workgroup column, no split-K |
| p513/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p513-fa1-20260618-192702/` | large aligned Q4/Q6 routes: `matmul_q4_k_f32_f16acc_aligned_l` and `matmul_q6_k_f32_f16acc_aligned_l`, `spec=[256,128,128,32,64,64,2,16,16,16,64]`, fifth workgroup column |

The p513 Vulkan row ran at `246.390 tok/s`, giving the existing HRX p513
scoreboard row `212.575 tok/s` a measured `0.863x` Vulkan ratio. This confirms
DeepSeek's p512 gap generalizes to the odd production-width tail regime, but
it does not change the schedule family: p33 stays medium and p512/p513 stay
large aligned. The previous Q6 rollback probe showed that reverting embedded
Q6 rows to old rows2/MMQL64 routes is not viable, so any DeepSeek policy work
should target the existing large aligned Q4/Q6 schedule gap rather than route
rollback.

The DeepSeek p33 row exposed a separate narrow-policy bug: HRX was leaving most
Q4_K p33 work on old scalar/Q8_1 routes while Vulkan selected the medium
aligned Q4_K pipeline. Existing HRX `MMQ64` was tested as the matching narrow
packed route:

| Gate | Default/Rollback | MMQ64 Default |
| --- | ---: | ---: |
| Focused Q4_K p33 rows | `30.213 ms` | `3.652 ms` |
| DeepSeek p33/fa1 model | `15.689 tok/s` | `104.827 tok/s` |
| Vulkan p33/fa1 oracle | `33.010 tok/s` | `33.010 tok/s` |

Accepted policy:
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmq64x64_wg256_f32` is default on gfx1151 for
Q4_K prompt rows with `32 <= cols < 128`, `rows % 64 == 0`, and packed Q8_1 x4
available. Rollback:
`GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`. Evidence:
`cache/hrxv1/gfx1151/deepseek-q4-p33-mmq64-default-regate-20260618-193411/`.

Qwen3/Qwen3-Coder Q4_K MoE prompt rows then became the worst current
post-Q6 basket gap. The same-machine Vulkan oracle for those rows selects the
medium subgroup ID route:
`matmul_id_subgroup_q4_k_f32_f16acc_aligned_m`,
`spec=[128,64,64,32,64,32,2,16,16,16,64]`,
`wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills. The existing HRX
wide-K grouped Q8_1 x4 route was re-gated on exact p33, p512, and p513 MoE
exports and promoted from opt-in to gfx1151 default:
`hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32`. Rollback:
`GGML_HRX_DISABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`. Evidence:
`cache/hrxv1/gfx1151/q4-id-widek-current-regate-20260618-204002/`,
`cache/hrxv1/gfx1151/q4-id-widek-default-postedit-regate-20260618-204720/`,
`cache/hrxv1/gfx1151/q4-id-widek-current-default-20260618-204154/`,
`cache/hrxv1/gfx1151/q4-id-widek-current-optin-20260618-204241/`, and
`cache/hrxv1/gfx1151/q4-id-widek-current-vulkan-20260618-204346/`.
Post-promotion full-basket KPI:
`cache/hrxv1/gfx1151/basket-after-q4-id-widek-default-commitaligned-20260618-205025/`.

Same-runner Qwen3/Qwen3-Coder Q4 MoE p33/p512/p513 geomean improved `1.941x`
over default and moved from `0.338x` to `0.657x` Vulkan. This is a real
promotion, but not the end state: HRX still uses only `3264` bytes LDS and
`107` VGPR for the selected wide-K route, so the next MoE work should
mechanically close the remaining medium-subgroup dataflow/resource gap rather
than switch back to aggregate-only exploration.

The full basket after this promotion improved from the prior post-Q6
commit-aligned checkpoint `0.478x` steady Vulkan geomean to `0.573x`.

A newer current-head full-basket KPI at llama.cpp commit `77e389e06` is:

```text
cache/hrxv1/gfx1151/basket-current-head-77e389e06-20260619-121528/
```

Both HRX and Vulkan rows report build commit `77e389e06`, HRX rows report
`backends=HRX`, Vulkan rows report `backends=Vulkan`, and HRX fallback count is
zero for all rows. This artifact supersedes the older post-Q4-ID basket for
current KPI ranking:

- average geomean HRX/Vulkan: `0.610x`;
- steady-state geomean HRX/Vulkan: `0.609x`;
- rows below parity: `22/24`.

Current worst steady rows are:

| Row | HRX steady tok/s | Vulkan steady tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B Q8_0 p512 | `457.292` | `913.835` | `0.500x` |
| Qwen3 30B Q4_K_XL p512 | `640.505` | `1238.080` | `0.517x` |
| Qwen2.5 Coder 7B Q5_K_M p512 | `610.729` | `1175.300` | `0.520x` |
| Qwen3 30B Q6_K p33 | `94.192` | `179.372` | `0.525x` |
| Qwen3 30B Q6_K p512 | `551.976` | `1049.760` | `0.526x` |

The mission remains active. Aggregate rows should be used only to rank the next
boulder; route work still needs focused schedule/ISA/correctness evidence.

## Operating Rule

Until HRX v1 is much closer to Vulkan parity, aggregate HRX/Vulkan token rates
are only boulder selectors and regression guardrails. They are not route
promotion evidence.

The primary development loop is:

1. choose the high-time Vulkan pipeline for an exact model/op/shape row;
2. extract the Vulkan schedule, SPIR-V, RADV ISA, and resource facts;
3. compare them mechanically against the current HRX HIP C++ route and HSACO;
4. implement the smallest named schedule delta that moves HRX toward the
   winning schedule;
5. run focused CPU-reference, route-trace, and same-runner A/B gates on p33,
   p512, and p513 before any model-level basket;
6. promote only when the focused evidence is correct, the intended route
   selected, odd/tail behavior is covered, and the schedule delta has a written
   acceptance reason.

If a HIP C++ source-visible spelling cannot reproduce a RADV behavior, record
that as a lowering or primitive blocker and pivot to a lower-level implementation
path, matrix-fragment API, or packed-route prior rather than continuing blind
aggregate exploration.

## Acceptance For This Goal

This goal is satisfied when:

- Vulkan oracle hooks are implemented in `sources/llama.cpp`;
- `build/vulkan-gfx1151` builds with the hooks;
- at least one worst-row p512/fa1 capture produces JSONL, SPIR-V, SPIR-V asm,
  split RADV ISA/stats, and inventory artifacts;
- the top dense quantized prompt pipeline has a written schedule comparison
  against the current HRX v1 provider;
- the next HRX candidate gate references that oracle schedule comparison
  before any new HIP kernel is promoted.

## Non-Goals

- Do not replace the HRX v1 HIP C++ pipeline with Loom.
- Do not compile HIP candidates in helper scripts; CMake/Ninja still owns HIP
  C++ kernel compilation.
- Do not promote from Vulkan token rate alone.
- Do not treat UMA memory-utilization readings as kernel evidence.
- Do not skip odd/tail prefill rows after a p512 win.
