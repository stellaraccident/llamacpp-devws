# HRX v1 gfx1151 Q6_K Vulkan Oracle Schedule Ledger

Date: 2026-06-17

## Scope

This ledger compares the Vulkan oracle Q6_K dense prompt schedule against the
current HRX v1 Q6_K prompt routes and the first HIP C++ WMMA diagnostic.

## Artifacts

- Vulkan oracle:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/`
- Vulkan SPIR-V asm:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/spvasm/matmul_q6_k_f32_f16acc_aligned_l__main__0x6eebdfb4c3043b23.spvasm`
- Vulkan RADV ISA/stats:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/radv/isa/matmul_q6_k_f32_f16acc_aligned_l__main__6eebdfb4c3043b23.amdgcn.txt`
  and
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/radv/stats/matmul_q6_k_f32_f16acc_aligned_l__main__6eebdfb4c3043b23.stats.txt`
- Exact Llama Q4_K_M Q6 rows:
  `cache/hrxv1/gfx1151/llama31-q4km-q6-output-rerank-20260617-214220/`
- HRX WMMA diagnostic:
  `cache/hrxv1/gfx1151/q6-wmma16-f16acc-focused-20260617-215803/`

## Vulkan Pipeline Facts

- Pipeline: `matmul_q6_k_f32_f16acc_aligned_l`
- Hash: `0x6eebdfb4c3043b23`
- Spec: `[256,128,128,32,64,64,2,16,16,16,64]`
- Workgroup denominators: `[128,128,1]`
- Workgroup size: `256 x 1 x 1`
- Dispatch count in Qwen3 30B Q6_K p512/fa1 graph: `192`
- Resource facts: `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills,
  `Instructions=3979`, `VALU=1833`, `VMEM=228`, `SMEM=102`
- ISA fact: the hot loop emits `v_wmma_f16_16x16x16_f16`.

Representative normalized p512 shape buckets:

| Dispatches | Src0 | Src1 | Dst | Workgroups |
| ---: | --- | --- | --- | --- |
| 96 | q6_K `[2048,512]` | f32 `[2048,512]` | f32 `[512,512]` | `[8,4,1]` |
| 48 | q6_K `[2048,4096]` | f32 `[2048,512]` | f32 `[4096,512]` | `[32,4,1]` |
| 48 | q6_K `[4096,2048]` | f32 `[4096,512]` | f32 `[2048,512]` | `[16,4,1]` |

## Current HRX Route Facts

Fastest current exact Llama Q4_K_M Q6 output path:
`hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x64_wg256_f32`, opt-in
`GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.

Focused exact-row timing:

| Row | Current MMQ64 |
| --- | ---: |
| `Vcur-0` | `2254.51 us` |
| `ffn_out-0` | `56728.72 us` |
| `result_output` | `379448.50 us` |

Compile facts for the direct MMQ64 object:
`wavefront_size=64`, `VGPR=181`, `SGPR=42`, `LDS=2176`, no spills.
The wider `MMQ64x128` and staged `MMQL128x128` Q6 variants select correctly
but are slower or spill, and are recorded as rejected in the tuning DB.

## WMMA Diagnostic

Candidate:
`hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_f32`, opt-in
`GGML_HRX_ENABLE_Q6_K_WMMA16_F16ACC_PROMPT=1`.

What it tests:

- Direct Q6_K dequantization to f16 for A.
- Direct F32 RHS cast to f16 for B.
- One 32-thread wave per 16x16 output tile.
- `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`.

Compile facts:

- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q6_k_wmma16.hsaco`
- `wavefront_size=32`, `SGPR=24`, `VGPR=35`, `LDS=0`, no spills.
- ISA emits `v_wmma_f16_16x16x16_f16`.

Correctness:

- p512 exact Llama Q4_K_M Q6 rows passed CPU-reference focused testing.
- Synthesized p33 and p513 odd/tail rows passed CPU-reference focused testing.
- Route traces selected `hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_f32`.

Timing:

- The current MMQ64 baseline completed on the exact p512 rows.
- The WMMA diagnostic was stopped after several minutes with an empty perf CSV
  while repeatedly selecting the intended provider on the large `ffn_out`
  shape. This is a performance-timeout rejection.

## Schedule Comparison

| Axis | Vulkan oracle | Current HRX Q6 | WMMA diagnostic |
| --- | --- | --- | --- |
| Main math | `v_wmma_f16_16x16x16_f16` | `v_dot4_i32_iu8` | `v_wmma_f16_16x16x16_f16` |
| RHS representation | F32 graph input lowered inside shader | Separate packed Q8_1 x4 route | Direct F32 cast to f16 |
| Workgroup | 256 threads | 256 threads | 32 threads |
| Tile | 128x128 denominator family | 64x64 prompt tile | 16x16 tile |
| LDS | 22528 bytes | 2176 bytes | 0 bytes |
| VGPR | 192, no spills | 181, no spills | 35, no spills |
| Status | prior to clone | current fastest exact path | rejected diagnostic |

## Conclusion

The useful Q6 direction is not another existing packed-Q8_1 tile flip. Vulkan's
winning Q6 pipeline is a staged 128x128 cooperative-matrix dataflow with large
LDS and high VGPR budget but no spills. The direct 16x16 HIP WMMA diagnostic
proves Q6 dequant lane mapping and odd/tail correctness, but it is far too slow
without Vulkan-style staging and 256-thread tile aggregation.

The follow-up staged HIP candidate
`hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_wg256_f32` validates that direction:

- wave32, workgroup 256, 64x32 output tile, 16-wide K panel;
- LDS `3072` bytes for staged f16 A/B panels;
- SGPR `36`, VGPR `45`, no spills;
- emits `v_wmma_f16_16x16x16_f16`;
- exact p512 Q6 rows plus synthesized p33/p513 odd-tail rows passed
  CPU-reference focused testing.

Focused p512 timing on exact Llama 3.1 8B Q4_K_M Q6 rows was mixed:

| Row | Current MMQ64 | Staged WG256 | Decision |
| --- | ---: | ---: | --- |
| `Vcur-0` | `2288.11 us` | `2034.99 us` | staged wins |
| `ffn_out-0` | `70762.23 us` | `52681.14 us` | staged wins |
| `result_output` | `373409.03 us` | `486642.92 us` | keep MMQ64 |

The accepted policy is therefore row-bounded: enable staged WMMA for Q6 dense
prompt rows with `rows <= 4096`, while keeping the large vocabulary projection
on the current packed-Q8_1 MMQ64 route. With that guard, Qwen3 30B Q6_K p512
same-binary no-trace r3 model A/B improved from `492.92 tok/s` to
`520.82 tok/s`; the traced variant selected the staged route 192 times.

Next Q6 work should build a staged WMMA candidate that preserves the Vulkan
family more closely:

- 256-thread workgroup over a 128x128 output tile;
- staged f16 A/B fragments in LDS or an equivalent register/LDS pipeline;
- explicit p33 medium-route policy kept separate from p512/p513 large routes;
- focused p33, p512, and p513 correctness before any model A/B.

## VK128 Default And B64GROUP Probe

The later accepted gfx1151 default is
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`.
It is the first Q6_K route that preserves the Vulkan oracle's broad shape:
BM128/BN128/BK32, 256-thread workgroup, wave64, f16 WMMA accumulation, direct
F32 RHS, and p33/p512/p513 tail correctness. It is default on gfx1151 with
rollback
`GGML_HRX_DISABLE_Q6_K_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.

Focused p512 timing versus the prior current-best route:

| Row | Prior current best | VK128 padded W64 |
| --- | ---: | ---: |
| `Vcur-0` | `2024.14 us` | `1261.13 us` |
| `ffn_out-0` | `45578.26 us` | `22575.58 us` |
| `result_output` | `382338.19 us` | `185386.36 us` |

Compile/resource facts for the accepted default:

- wave64, SGPR `36`, VGPR `156`, LDS `20480`, no spills;
- 32 static `v_wmma_f16_16x16x16_f16` sites;
- 32 `ds_load_b128`, 64 `global_store_b32`, 2 barriers.

The remaining RADV delta is now very specific: the Vulkan oracle still has
LDS `22528`, VGPR `192`, 64 `ds_load_b64`, 128 `ds_load_u16_d16`, 128 LDS
`b16` stores, and 192 `buffer_store_b32`.

The follow-up B64GROUP diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`
tested one exact RADV axis by grouping A/B fragment reads before each 4x4 WMMA
block. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-focused-20260618-050234/`.

Compile/resource facts:

- wave64, SGPR `40`, VGPR `197`, LDS `20480`, no spills;
- 32 `v_wmma_f16_16x16x16_f16`;
- 64 `ds_load_b64`, 0 `ds_load_b128`, 64 `global_store_b32`, 2 barriers.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- route traces selected the B64GROUP provider on all three Q6 rows for each
  size.

Same-runner focused timing rejected the probe:

| Size | Row | Default | B64GROUP | Ratio |
| --- | --- | ---: | ---: | ---: |
| p33 | `Vcur-0` | `678.80 us` | `798.20 us` | `1.176x` |
| p33 | `ffn_out-0` | `3360.94 us` | `3581.94 us` | `1.066x` |
| p33 | `result_output` | `17916.01 us` | `22896.35 us` | `1.278x` |
| p512 | `Vcur-0` | `1231.62 us` | `2428.77 us` | `1.972x` |
| p512 | `ffn_out-0` | `22822.28 us` | `29021.04 us` | `1.272x` |
| p512 | `result_output` | `184755.28 us` | `227402.03 us` | `1.231x` |
| p513 | `Vcur-0` | `1032.27 us` | `1157.09 us` | `1.121x` |
| p513 | `ffn_out-0` | `11579.62 us` | `17035.95 us` | `1.471x` |
| p513 | `result_output` | `111122.29 us` | `134430.43 us` | `1.210x` |

Decision: reject for production promotion. Matching RADV's `64 ds_load_b64`
count alone is insufficient in this HIP C++ spelling, especially with VGPR
pressure rising past the RADV oracle (`197` vs `192`) while the LDS footprint,
`ds_load_u16_d16`, LDS stores, and 192-store writeback are still missing. The
next Q6 direct-WMMA attempt should target those remaining cooperative-matrix
LDS/writeback deltas together, or pivot back to packed-path schedule work.

## VK128 Full-Tile Store Probe

The follow-up fullstore diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_fullstore_f16acc_wg256_f32`
tested the store-side RADV delta directly. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q6-wmma-vk128-fullstore-focused-20260618-055935/`.

Compile/resource facts:

- wave64, SGPR `36`, VGPR `156`, LDS `20480`, no spills;
- 32 `v_wmma_f16_16x16x16_f16`;
- 32 `ds_load_b128`, 128 `global_store_b32`, 2 barriers.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- route traces selected the fullstore provider on all three Q6 rows for each
  size.

Same-runner focused timing was mixed:

| Size | Row | Default | Fullstore | Ratio |
| --- | --- | ---: | ---: | ---: |
| p33 | `Vcur-0` | `679.31 us` | `679.32 us` | `1.000x` |
| p33 | `ffn_out-0` | `3204.62 us` | `3223.13 us` | `1.006x` |
| p33 | `result_output` | `18061.42 us` | `17961.53 us` | `0.994x` |
| p512 | `Vcur-0` | `1246.93 us` | `1201.46 us` | `0.963x` |
| p512 | `ffn_out-0` | `22209.12 us` | `22331.66 us` | `1.006x` |
| p512 | `result_output` | `185590.17 us` | `183399.28 us` | `0.988x` |
| p513 | `Vcur-0` | `1032.30 us` | `1020.43 us` | `0.989x` |
| p513 | `ffn_out-0` | `11798.46 us` | `11725.39 us` | `0.994x` |
| p513 | `result_output` | `113830.84 us` | `113984.53 us` | `1.001x` |

Decision: reject for production promotion. The isolated full-tile store split
moved the static writeback shape from 64 to 128 stores without increasing
register pressure or spilling, but the runtime signal is only small and mixed.
It still misses RADV's `22528` byte LDS footprint, `64 ds_load_b64`,
`128 ds_load_u16_d16`, `128` LDS b16 stores, and `192` buffer-store
cooperative-matrix writeback. This confirms that we should continue mechanically
zeroing in on the exact winning schedule, but the next attempt needs to combine
the remaining LDS/writeback/lane-ownership deltas rather than treating the store
guard split as an independent promotion axis.

## VK128 Store-Stage Probe

The store-stage diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_store_stage_f16acc_wg256_f32`
tested the remaining LDS-footprint/writeback axis by staging the f16
accumulator tile through LDS before scalar global writeback. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_STORE_STAGE_F16ACC_WG256_PROMPT=1`.

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-store-stage-compile-20260618/`;
- focused:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-store-stage-focused-20260618/`.

Compile/resource facts:

- wave64, SGPR `36`, VGPR `148`, LDS `22528`, no spills;
- 32 `v_wmma_f16_16x16x16_f16`;
- 66 `ds_store_b16`, 64 `ds_load_u16`, 64 `global_store_b32`;
- 34 barriers and 315 `s_waitcnt`.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on the accepted VK64 narrow route;
- p512 and p513 selected the store-stage provider on all three Q6 rows.

Same-runner focused timing was mixed:

| Size | Row | Default | Store-stage | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `1223.75 us` | `1233.23 us` | `1.008x` |
| p512 | `ffn_out-0` | `22550.84 us` | `23574.10 us` | `1.045x` |
| p512 | `result_output` | `187339.61 us` | `187393.19 us` | `1.000x` |
| p513 | `Vcur-0` | `1032.48 us` | `1027.26 us` | `0.995x` |
| p513 | `ffn_out-0` | `11686.69 us` | `11651.95 us` | `0.997x` |
| p513 | `result_output` | `113641.02 us` | `108827.76 us` | `0.958x` |

Decision: reject for broad production promotion. The probe now matches RADV's
large-route LDS footprint and remains correct, but the HIP source spelling pays
for that with many barriers and does not recover the low-barrier
cooperative-matrix writeback. The p513 `result_output` signal is useful and
should be preserved for a future tail-only policy test, but p512 is flat or
regressive and is the main production-width gate.

## Rejected VK128 Store-Stage Tail-Only Policy

Route:
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_store_stage_f16acc_wg256_f32`

Temporary policy tested:
default store-stage only for production-width tails with `cols >= 512` and
`cols % 128 != 0`. Rollback:
`GGML_HRX_DISABLE_Q6_K_WMMA16_VK128_PADDED_W64_STORE_STAGE_TAIL_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q6-store-stage-tail-default-20260618-235052/`

Why this was worth testing:
the broad store-stage probe matched RADV's `22528` byte LDS footprint and
regressed p512, but it had a focused p513 tail win, especially on
`result_output`. This policy isolated that positive tail signal while keeping
p33 on VK64 and full-width rows on the accepted VK128 route.

Focused route/correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p33 route traces stayed on
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`;
- p512 route traces stayed on
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`;
- p513 route traces selected the store-stage provider.

p513 focused default vs rollback:

| Row | Store-stage tail | Rollback VK128 | Delta |
| --- | ---: | ---: | ---: |
| `Vcur-0-p513` | 1028.65 us | 1030.81 us | -0.21% |
| `ffn_out-0-p513` | 11660.75 us | 11757.25 us | -0.82% |
| `result_output-p513` | 110853.92 us | 111678.94 us | -0.74% |

Same-binary Qwen3 30B Q6_K p513/fa1 model A/B:

| Store-stage tail | Rollback VK128 | Ratio |
| ---: | ---: | ---: |
| 537.657 tok/s | 542.453 tok/s | 0.991x |

Decision:
reject and remove the temporary selector. The focused p513 signal is real but
too small and does not survive the model-level gate. Keep the store-stage route
opt-in as evidence for the RADV LDS-footprint axis; do not promote it without a
lower-barrier cooperative writeback path or a model-level win.

## VK128 B64GROUP Plus Full-Tile Store Probe

The combined diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_fullstore_f16acc_wg256_f32`
tested the two source-visible deltas together: grouped `ds_read_b64` fragment
loads and full-tile stores for interior tiles. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_FULLSTORE_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-fullstore-focused-20260618-060839/`.

Compile/resource facts:

- wave64, SGPR `40`, VGPR `196`, LDS `20480`, no spills;
- 32 `v_wmma_f16_16x16x16_f16`;
- 64 `ds_load_b64`, 128 `global_store_b32`, 2 barriers.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- route traces selected the combined provider on all three Q6 rows for each
  size.

Same-runner focused timing rejected the probe:

| Size | Row | Default | B64GROUP+Fullstore | Ratio |
| --- | --- | ---: | ---: | ---: |
| p33 | `Vcur-0` | `678.53 us` | `798.70 us` | `1.177x` |
| p33 | `ffn_out-0` | `3266.17 us` | `3836.85 us` | `1.175x` |
| p33 | `result_output` | `17918.60 us` | `23491.26 us` | `1.311x` |
| p512 | `Vcur-0` | `1229.96 us` | `2383.51 us` | `1.938x` |
| p512 | `ffn_out-0` | `22244.78 us` | `28722.42 us` | `1.291x` |
| p512 | `result_output` | `180519.25 us` | `232795.11 us` | `1.290x` |
| p513 | `Vcur-0` | `1031.38 us` | `1138.76 us` | `1.104x` |
| p513 | `ffn_out-0` | `12337.31 us` | `16923.53 us` | `1.372x` |
| p513 | `result_output` | `112612.90 us` | `135186.16 us` | `1.200x` |

Decision: reject for production promotion. The combined source-visible clone
matches the RADV oracle on the `64 ds_load_b64` and 32-WMMA axes and moves the
store count from 64 to 128, but it is slower on every focused row and still
does not reproduce RADV's `22528` byte LDS footprint, `128 ds_load_u16_d16`,
`128` LDS b16 stores, or `192` buffer-store cooperative writeback. This is a
strong indication that the remaining Q6 direct-WMMA parity gap is in the
cooperative-matrix load/store/lane-ownership lowering, not in these two isolated
or combined HIP C++ source pivots.

## Q6 Odd/Tail Vulkan Oracle Coverage

The p512 oracle and direct-WMMA probes are not sufficient for Q6 promotion.
Qwen3 30B Q6_K odd and tail captures are now available:

| Row | Artifact | Backend validation |
| --- | --- | --- |
| p33/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p33-fa1-20260618-061613/` | `stdout.json` reports `backends=Vulkan`, `n_prompt=33`, `flash_attn=true` |
| p513/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p513-fa1-20260618-061619/` | `stdout.json` reports `backends=Vulkan`, `n_prompt=513`, `flash_attn=true` |

p33 inventory facts:

- top dense Q6 route: `matmul_q6_k_f32_f16acc_aligned_m`;
- hash: `0x6eebdfb4c3043b23`;
- spec: `[128,64,64,32,64,32,2,16,16,16,64]`;
- workgroup denominators: `[64,64,1]`;
- dispatch count: `192`;
- RADV resources: `SGPR=108`, `VGPR=144`, `LDS=11264`, no spills;
- ISA counts: `16` WMMA, `48 ds_load_b64`, `64 ds_load_u16_d16`,
  `64 ds_store_b16`, `96 buffer_store_b32`, `2` barriers, no
  `ds_load_b128`.

Representative p33 normalized shape buckets:

| Dispatches | Pipeline | Src0 | Src1 | Dst | Workgroups |
| ---: | --- | --- | --- | --- | --- |
| 96 | `matmul_q6_k_f32_f16acc_aligned_m` | q6_K `[2048,512]` | f32 `[2048,33]` | f32 `[512,33]` | `[8,1,1]` |
| 48 | `matmul_q6_k_f32_f16acc_aligned_m` | q6_K `[2048,4096]` | f32 `[2048,33]` | f32 `[4096,33]` | `[64,1,1]` |
| 48 | `matmul_q6_k_f32_f16acc_aligned_m` | q6_K `[4096,2048]` | f32 `[4096,33]` | f32 `[2048,33]` | `[32,1,1]` |

p513 inventory facts:

- top dense Q6 route: `matmul_q6_k_f32_f16acc_aligned_l`;
- hash: `0x6eebdfb4c3043b23`;
- spec: `[256,128,128,32,64,64,2,16,16,16,64]`;
- workgroup denominators: `[128,128,1]`;
- dispatch count: `192`;
- RADV resources: `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills;
- ISA counts: `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `192 buffer_store_b32`, `2` barriers, no
  `ds_load_b128`;
- `split_k_reduce` tail reductions appear in the graph: `143` dispatches with
  representative workgroups `[257,1,1]`, `SGPR=108`, `VGPR=12`, no LDS, no
  spills, `83` instructions.

Representative p513 normalized shape buckets:

| Dispatches | Pipeline | Src0 | Src1 | Dst | Workgroups |
| ---: | --- | --- | --- | --- | --- |
| 96 | `matmul_q6_k_f32_f16acc_aligned_l` | q6_K `[2048,512]` | f32 `[2048,513]` | f32 `[512,513]` | `[8,5,1]` |
| 96 | `split_k_reduce` | q6_K `[2048,512]` | f32 `[2048,513]` | f32 `[512,513]` | `[257,1,1]` |
| 48 | `matmul_q6_k_f32_f16acc_aligned_l` | q6_K `[2048,4096]` | f32 `[2048,513]` | f32 `[4096,513]` | `[32,5,1]` |
| 48 | `matmul_q6_k_f32_f16acc_aligned_l` | q6_K `[4096,2048]` | f32 `[4096,513]` | f32 `[2048,513]` | `[16,5,1]` |

Policy implication:

- p33 is a medium/narrow route family, not a downscaled p512 large route;
- p512 and p513 share the large aligned route family, but p513 also requires
  tail-reduction behavior to be considered when judging parity;
- the rejected Q6 HIP C++ B64GROUP, fullstore, and combined probes remain valid
  negative evidence against isolated source-visible pivots;
- the next Q6 route should either reproduce the missing RADV
  cooperative-matrix LDS/load/store/lane-ownership behavior more directly, or
  move to a different prior such as a packed route, matrix-fragment API, or
  lower-level implementation path.

## p513 Split-K Reduce Oracle Extract

Artifact:

```text
cache/hrxv1/gfx1151/split-k-reduce-oracle-summary-20260618-162944/
```

The source-controlled extractor is:

```text
sources/llama.cpp/tools/vulkan-oracle/extract_split_k_reduce.py
```

It pairs every Vulkan `split_k_reduce` dispatch in the Qwen3 30B Q6_K p513/fa1
oracle trace with the preceding producer dispatch by matching the producer
output binding to the reduce input binding. All `143` reductions paired
successfully:

| Count | Producer | Src0 | Src1 | Dst | Producer WG | Reduce WG | Output elems | Factor | Scratch bytes |
| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 96 | `matmul_q6_k_f32_f16acc_aligned_l` | q6_K `[2048,512]` | f32 `[2048,513]` | f32 `[512,513]` | `[8,5,1]` | `[257,1,1]` | 262656 | 2 | 2101248 |
| 47 | `matmul_f32_f32_aligned_l` | f32 `[2048,128]` | f32 `[2048,513]` | f32 `[128,513]` | `[8,5,1]` | `[65,1,1]` | 65664 | 8 | 2101248 |

The K/V-style Q6 tail contract matches the Q5 p513 tail contract:
`output_elems * factor * sizeof(float)` scratch and a separate reduce. Qwen3
also exposes an F32 MoE-logit split-K row with factor `8`, so a general HRX
runtime hook should not hard-code factor `2` or K-quant-only producers.
Production Q6 p513 parity should be gated on the `96` Q6_K K/V rows first,
then checked against the F32 MoE-logit rows if the runtime path is generalized.

## VK64 Padded44 Medium-Route Promotion

The accepted narrow/odd Q6 route is
`hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`.
It is default on gfx1151 for `Q6_K x F32 -> F32` prompt rows with
`16 <= cols <= 64`, `rows >= 16`, and `k % 256 == 0`. Rollback:
`GGML_HRX_DISABLE_Q6_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/`.

What it clones from the p33 Vulkan oracle:

- BM64/BN64/BK32, 256-thread workgroup, wave64;
- direct F32 RHS, f16 WMMA accumulation;
- 44-half LDS stride, giving `(64 + 64) * 44 * sizeof(f16) = 11264` bytes
  LDS, matching the RADV medium route footprint.

Focused p33 evidence:

| Row | Prior VK128 default | VK64 padded44 | Ratio |
| --- | ---: | ---: | ---: |
| `Vcur-0-p33` | `678.27 us` | `397.77 us` | `0.586x` |
| `ffn_out-0-p33` | `3175.03 us` | `2239.18 us` | `0.705x` |
| `result_output-p33` | `18046.03 us` | `11744.51 us` | `0.651x` |

Qwen3 30B Q6_K p33/fa1 no-trace model A/B improved from `96.68 tok/s` to
`109.61 tok/s`. Default p33 CPU-reference routes select VK64; rollback selects
the previous VK128 route. A post-promotion p512 focused gate still selects
VK128, so this promotion does not steal production-width rows.

Remaining schedule delta:

- HIP VK64 emits 8 visible WMMA sites, 16 `global_store_b32`, 2 barriers, and
  the matching 11264-byte LDS footprint;
- RADV medium emits 16 WMMA sites, 48 `ds_load_b64`, 64 `ds_load_u16_d16`,
  64 LDS b16 stores, 96 `buffer_store_b32`, 2 barriers, and 11264-byte LDS.

Decision: accept the route as a narrow-prompt default because focused and model
evidence are strong and the route follows the p33 Vulkan medium regime. Do not
treat it as exact Vulkan schedule parity; future Q6 work still needs to close
the visible cooperative-matrix load/store/lane-ownership delta.

## VK64 Padded44 H4LOAD Probe

The diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32`
keeps the accepted p33 medium-route shape but spells LDS fragment reads as
half4 chunks. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_H4LOAD_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q6-vk64-h4load-static-20260618-225444/`.

Static triage rejected it before runtime timing. The emitted HSACO preserved
the accepted VK64 route's resource contract and tracked ISA shape:

- wave64, SGPR `36`, VGPR `59`, LDS `11264`, no spills;
- 8 visible `v_wmma_f16_16x16x16_f16`;
- 20 `ds_load_2addr_b64`, 0 plain `ds_load_b64`,
  0 `ds_load_u16_d16`;
- 2 `ds_store_b16`, 16 `global_store_b32`;
- 2 barriers and 58 `s_waitcnt`.

Decision: reject as compile-equivalent. This confirms that local C++ half4
load spelling is not enough to recover the RADV medium topology. The next Q6
p33 work needs a materially different cooperative load/store or lower-level
primitive, not another scalar/half-vector fragment-load wrapper around the same
VK64 source.

## Current-Head Q6 KPI Recheck

Artifact:
`cache/hrxv1/gfx1151/qwen3-q6-current-head-r1-20260619-continued/`.

Command:

```bash
python3 tools/hrxv1_basket_benchmark.py \
  --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k \
  --cases p33,p512,p513 \
  --backends hrx,vulkan \
  --repetitions 1 \
  --timeout 1200 \
  --tag qwen3-q6-current-head-r1-20260619-continued
```

The source checkout was clean at `24fc9766e`, but HRX `llama-bench` still
reported build commit `c888948cf` because generated build metadata was not
refreshed after the opt-in B-pair diagnostic commit. Treat this as
default-equivalent evidence, not a strict commit-aligned basket.

Route evidence:

- p33 selected
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32` for
  dense Q6 rows and
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32` for MoE Q6 ID rows.
- p512 and p513 selected
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32` for dense
  Q6 rows plus the same grouped Q6 ID provider.
- HRX fallback lines were zero.

Timing:

| Row | HRX tok/s | Vulkan tok/s | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| p33 | `84.683` | `150.088` | `0.564x` |
| p512 | `547.507` | `994.327` | `0.551x` |
| p513 | `545.891` | `979.095` | `0.558x` |

Decision: Q6 remains a first-tier parity gap. Selector ownership is not the
main unknown: the expected dense and ID providers are selected and correct
enough for model execution. The next Q6 candidate must either:

- build a true Vulkan-medium Q6 `MUL_MAT_ID` route for p33, using
  `matmul_id_subgroup_q6_k_f32_f16acc_aligned_m` as the schedule prior; or
- introduce a lower-level cooperative load/WMMA/store primitive for dense Q6
  that can express RADV's 16-WMMA, 48 `ds_load_b64`, 64 `ds_load_u16_d16`,
  64 `ds_store_b16`, 96 `buffer_store_b32`, and two-barrier medium path, plus
  the matching large-route 32-WMMA / 192-store production-width path.

## Rejected VK128 B64GROUP Buffer-Store Probe

Route:
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifacts:

- focused:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-bufferstore-focused-20260619-074930/`
- static:
  `cache/hrxv1/gfx1151/q6-b64group-bufferstore-static-20260619-074858/`

What changed:

- preserved the Q6 VK128 padded W64 B64GROUP direct-F32 WMMA math path;
- added the gfx11 raw-buffer descriptor/writeback primitive already validated
  in Q5/Q8;
- kept p33 on the accepted VK64 narrow route while forcing p512/p513 large
  rows to the new provider.

Static facts:

- wave64, SGPR `40`, VGPR `196`, LDS `20480`, no spills;
- `32` `v_wmma_f16_16x16x16_f16`;
- `64` LDS reads, `2` LDS writes, `128` VMEM stores, `2` barriers.

Focused correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 selected
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`;
- p512 and p513 selected the new B64GROUP buffer-store provider.

Same-runner focused timing rejected the route:

| Size | Row | Default | B64GROUP buffer-store | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `1264.654 us` | `2432.407 us` | `1.923x` |
| p512 | `ffn_out-0` | `22716.399 us` | `29124.536 us` | `1.282x` |
| p512 | `result_output` | `187888.389 us` | `227737.083 us` | `1.212x` |
| p513 | `Vcur-0-p513` | `1030.436 us` | `1140.124 us` | `1.106x` |
| p513 | `ffn_out-0-p513` | `11604.048 us` | `16796.850 us` | `1.447x` |
| p513 | `result_output-p513` | `111050.510 us` | `137889.980 us` | `1.242x` |

Decision:
reject for production promotion. This closes the simple Q6 raw-buffer writeback
axis: like Q5/Q8, buffer stores plus grouped LDS fragment reads are not enough
without RADV's cooperative halfword LDS load/store and accumulator
lane-ownership topology. The useful next dense-Q6 path is a lower-level
cooperative store/load primitive or a different schedule family, not another
isolated source-visible writeback wrapper.
  64 LDS-store, 96-buffer-store medium route without the correctness failures
  seen in GROUPK2.

Do not spend the next Q6 loop on grouped-ID threshold policy or local half-load
spelling. Those axes are already rejected or insufficient.

## VK64 Padded44 GROUPK2 Probe

The diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_groupk2_f16acc_wg256_f32`
keeps the accepted p33 medium-route tile shape but changes the live fragment
topology. Each wave preloads both K tiles' four A fragments plus one B fragment
with nowait b64 LDS reads before issuing the eight WMMAs. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_GROUPK2_F16ACC_WG256_PROMPT=1`.

Artifacts:

- static:
  `cache/hrxv1/gfx1151/q6-vk64-groupk2-static-20260618-230019/`;
- focused:
  `cache/hrxv1/gfx1151/q6-vk64-groupk2-focused-20260618-230100/`.

Static triage moved toward the RADV medium route on the LDS load axis:

| Route | VGPR | LDS | Spills | `ds_load_b64` | First-window loads | First-window WMMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RADV medium | 144 | 11264 | 0 | 48 | 48 | 16 |
| accepted VK64 | 59 | 11264 | 0 | 0 | 0 | 8 |
| GROUPK2 | 101 | 11264 | 0 | 40 | 20 | 8 |

The probe still does not recover RADV's schedule. It emits final
`lgkmcnt(0)`, keeps only 8 visible WMMAs, and does not introduce RADV's
`ds_load_u16_d16` plus LDS-store/writeback topology.

Correctness rejected it before timing:

- p33 CPU-reference failed all three Q6 rows with NaNs while route traces
  confirmed GROUPK2 selection;
- p512 and p513 CPU-reference non-steal gates passed and stayed on the
  existing VK128 provider.

Decision: reject. Source-level nowait grouped preloading can force plain
`ds_load_b64`, but this spelling is not semantically safe and still falls short
of RADV's cooperative issue/writeback schedule. The next Q6 p33 attempt should
not be another grouped scalar-fragment wrapper; it needs a safer explicit wait
ladder/dependency model or a lower-level implementation that can preserve
RADV-like scheduling without racing fragment data.

## VK64 Padded44 GROUPK2_WAIT Probe

The diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_groupk2_wait_f16acc_wg256_f32`
keeps GROUPK2's two-K grouped fragment topology but replaces nowait LDS
fragment reads with the existing helper that waits after each b64 fragment
read. It is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_GROUPK2_WAIT_F16ACC_WG256_PROMPT=1`.

Artifacts:

- static:
  `cache/hrxv1/gfx1151/q6-vk64-groupk2-wait-static-20260618-230620/`;
- focused:
  `cache/hrxv1/gfx1151/q6-vk64-groupk2-wait-focused-20260618-230642/`.

Static triage:

| Route | VGPR | LDS | Spills | `ds_load_b64` | `s_waitcnt` | First-window loads | First-window WMMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RADV medium | 144 | 11264 | 0 | 48 | 115 | 48 | 16 |
| GROUPK2 | 101 | 11264 | 0 | 40 | 10 | 20 | 8 |
| GROUPK2_WAIT | 101 | 11264 | 0 | 40 | 50 | 20 | 4 |

Correctness:

- p33 route traces confirmed GROUPK2_WAIT selection;
- `Vcur-0-p33` passed;
- `ffn_out-0-p33` failed with finite `ERR 0.001619838`;
- `result_output-p33` failed with finite `ERR 0.007942211`;
- p512 and p513 non-steal gates passed and stayed on the existing VK128
  provider.

Decision: reject. The per-load wait helper removes the NaN failure mode from
the nowait GROUPK2 probe but does not make the grouped scalar-fragment topology
exact, and it moves the first-WMMA schedule farther from RADV. The next useful
Q6 p33 step should either prove the exact WMMA fragment/lane mapping in a small
fixture or use a lower-level/cooperative-store implementation path; more
wrappers around the same scalar fragment preload are exhausted.

## VK128 B64GROUP Plus Store-Stage Probe

The combined diagnostic
`hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_store_stage_f16acc_wg256_f32`
tested grouped LDS fragment reads and explicit LDS output staging together. It
is opt-in behind
`GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_STAGE_F16ACC_WG256_PROMPT=1`.

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-store-stage-compile-20260618/`;
- focused:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-store-stage-focused-20260618/`.

Compile/resource facts:

- wave64, SGPR `40`, VGPR `196`, LDS `22528`, no spills;
- 32 `v_wmma_f16_16x16x16_f16`;
- 64 `ds_load_b64`, 64 `ds_load_u16`, 66 `ds_store_b16`,
  64 `global_store_b32`;
- 34 barriers and 367 `s_waitcnt`.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on the accepted VK64 narrow route;
- p512 and p513 selected the combined b64group-store-stage provider.

Same-runner focused timing rejected the probe:

| Size | Row | Default | B64GROUP+Store-stage | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `1253.70 us` | `2442.08 us` | `1.948x` |
| p512 | `ffn_out-0` | `22306.27 us` | `28932.14 us` | `1.297x` |
| p512 | `result_output` | `182744.78 us` | `230756.64 us` | `1.263x` |
| p513 | `Vcur-0` | `1041.04 us` | `1151.76 us` | `1.106x` |
| p513 | `ffn_out-0` | `11754.13 us` | `16996.90 us` | `1.446x` |
| p513 | `result_output` | `110050.98 us` | `136412.67 us` | `1.240x` |

Decision: reject for production promotion. The source-visible direct-WMMA clone
now matches the broad tile, LDS footprint, 32 WMMA, and 64 `ds_load_b64` axes,
but it is still not RADV's cooperative-matrix load/store/lane-ownership
schedule: the HIP spelling has 34 barriers, half the `ds_load_u16_d16`/LDS
store shape, and only 64 global stores. Further Q6 work should stop combining
these same HIP C++ scalar-store pivots and should either use a real matrix
fragment/cooperative-store path, a lower-level implementation, or a different
packed-route prior.

## HSACO Family Static Triage

Artifact:
`cache/hrxv1/gfx1151/q6_k-hsaco-family-summary-20260618-162005/`.

Tool:
`sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py`.

Key static facts:

| Route HSACO | Wave | VGPR | LDS | VGPR spills | Hot op | Static signal |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mul_mat_vec_q6_k_q8_1.hsaco` | 32 | 136 | 1088 | 0 | `v_dot4_i32_iu8` | current small packed route, not Vulkan large |
| `mul_mat_vec_q6_k_q8_1_wave64.hsaco` | 64 | 192 | 28672 | 553 | `v_dot4_i32_iu8` | heavy spill cliff |
| `mul_mat_vec_q6_k_q8_1_x4_mmql128.hsaco` | 64 | 192 | 9728 | 398 | `v_dot4_i32_iu8` | heavy spill cliff |
| `mul_mat_vec_q6_k_q8_1_x4_wave64_direct.hsaco` | 64 | 192 | 4352 | 135 | `v_dot4_i32_iu8` | spill cliff |
| `mul_mat_vec_q6_k_wmma16_vk128_padded_w64_b64group_store_stage_wg256.hsaco` | 64 | 196 | 22528 | 0 | `v_wmma_f16_16x16x16_f16` | closest LDS-footprint clone, but wrong barriers/store topology |
| `mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_wg256.hsaco` | 64 | 59 | 11264 | 0 | `v_wmma_f16_16x16x16_f16` | accepted medium/narrow p33 route |

Decision:
the Q6 large-route gap is not a missing compile flag or simple wave64/packed
variant. Existing packed large routes hit spill cliffs, while the direct-WMMA
large clones that avoid spills still miss the RADV cooperative load/store and
barrier topology. Keep the VK64 p33 default, keep production-width large rows
on the current default until a materially different schedule primitive exists,
and treat `split_k_reduce` p513 parity as a separate future requirement.

## MUL_MAT_ID Direct-F32 WMMA Diagnostic

Route:
`hrx_mul_mat_id_q6_k_wmma16x16_direct_f16acc_wg32_f32`, enabled only with
`GGML_HRX_ENABLE_Q6_K_ID_WMMA16_DIRECT_PROMPT=1`.

Artifacts:

- focused correctness:
  `cache/hrxv1/gfx1151/q6-id-wmma16-direct-focused-20260619-003510/`;
- focused perf:
  `cache/hrxv1/gfx1151/q6-id-wmma16-direct-perf-20260619-003759/`;
- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_id_q6_k_wmma16_direct.hsaco`.

Static facts:

- wave32, SGPR `54`, VGPR `52`, LDS `0`, no private segment, no spills;
- one visible `v_wmma_f16_16x16x16_f16` site and scalar global stores;
- no RADV-like LDS staging or cooperative writeback.

Correctness and route evidence:

- p33, p512, and p513 Qwen3 Q6_K `MUL_MAT_ID` focused CPU-reference gates
  passed;
- default traces selected
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`;
- opt-in traces selected the direct-F32 provider with `direct_f32=1`.

Same-runner focused timing rejected the probe:

| Size | Row | Default grouped Q8_1/x4 | Direct-F32 WMMA | Ratio |
| --- | --- | ---: | ---: | ---: |
| p33 | `ffn_moe_gate-0` | `300.02 us` | `2347.06 us` | `7.82x` |
| p33 | `ffn_moe_down-0` | `262.13 us` | `2153.97 us` | `8.22x` |
| p512 | `ffn_moe_gate-0` | `2399.79 us` | `15890.97 us` | `6.62x` |
| p512 | `ffn_moe_down-0` | `2319.38 us` | `19716.80 us` | `8.50x` |
| p513 | `ffn_moe_gate-0` | `2495.16 us` | `16175.99 us` | `6.48x` |
| p513 | `ffn_moe_down-0` | `2363.23 us` | `20285.69 us` | `8.58x` |

Decision: reject for production and skip model A/B. The diagnostic proves the
raw F32 RHS contract and Q6 direct-WMMA lane mapping can be made correct for
grouped MoE ID, but without packed Q8_1 reuse or RADV's cooperative staging and
writeback it is an order-of-magnitude-losing schedule. Future Q6 ID work should
target the Vulkan medium subgroup schedule directly or move to a lower-level
cooperative primitive; do not continue with direct-F32 wrapper variants.
