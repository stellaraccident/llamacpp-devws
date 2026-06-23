# HRX v1 Q5_K Vulkan Oracle Schedule Ledger

Date: 2026-06-18

## Target

Active gap: Q5_K dense prompt `MUL_MAT` on gfx1151.

Focused rows are reused from:
`cache/hrxv1/gfx1151/q5-wmma-vk128-focused-20260618-025116/`.

Rows:

- `Kcur-0`: Q5_K `[3584,512]` x F32 `[3584,p]` -> F32 `[512,p]`
- `Qcur-0`: Q5_K `[3584,3584]` x F32 `[3584,p]` -> F32 `[3584,p]`
- `ffn_out-3`: Q5_K `[18944,3584]` x F32 `[18944,p]` -> F32 `[3584,p]`
- `ffn_gate-0`: Q5_K `[3584,18944]` x F32 `[3584,p]` -> F32 `[18944,p]`

The current production HRX route remains the packed-Q8_1 path for production
widths, with narrow prompts on rows2/cols8. Direct-F32 WMMA probes are
diagnostic until they beat that route.

## Rejected p33 Combined96 Store Contract

Source change:
`sources/llama.cpp` adds a CMake-built diagnostic mode to
`hrx-hip-bench-wmma-f16-lane-map`:

```bash
build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-wmma-f16-lane-map \
  --mode=combined96-store-contract
```

Artifact:
`cache/hrxv1/gfx1151/q5-combined96-store-contract-20260619-031500/`

Result:

| cols | written | duplicate coords | unexpected second-half writes | valid |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 2048 | 2048 | 0 | no |
| 33 | 2112 | 2050 | 64 | no |
| 64 | 4096 | 2112 | 2048 | no |

Interpretation:
the p33 combined96 route is structurally invalid before Q5 math is considered.
It emits 24 logical writeback groups from only eight accumulator vectors using
`group & 7`. Groups `16..23` alias earlier output coordinates, and at the odd
p33 boundary the stage path writes column `32` from accumulators that only
represent the first two 16-column output tiles. This explains why matching the
RADV p33 medium total opcode counts was not enough: the diagnostic fixture
matched a store-count surface, but the catalog route does not have a valid
medium-tile output ownership contract. Do not continue tuning this combined96
catalog route; the next p33 candidate must first compute and own the full
medium `64x64` output tile or explicitly narrow its `BN` contract.

## Rejected p33 Full64 Catalog Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_FULL64_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-p33-full64-catalog-20260619-032340/`

Store-contract artifact:
`cache/hrxv1/gfx1151/q5-full64-store-contract-20260619-035618/`

What changed:

- kept the p33 medium BM64/BN64/BK32/WG256/wave64/padded44 shape;
- repaired the combined96 output-ownership bug by computing all `16` medium
  output tile groups in wave0;
- wrote only groups `0..15`, avoiding the duplicate `group & 7` writeback and
  second-half aliasing found by the store-contract diagnostic.

Focused correctness:

- route trace selected the full64 provider for all four Qwen2.5 Coder Q5_K_M
  p33 rows;
- CPU-reference failed all four rows with NaNs.

Store-coordinate contract:

- `hrx-hip-bench-wmma-f16-lane-map --mode=full64-store-contract` was added as
  a CMake-built diagnostic.
- The full64 mapping wrote every target coordinate exactly once with no NaNs
  for `cols=32`, `cols=33`, and `cols=64`.
- Therefore the full64 catalog failure is not another output-coordinate alias
  like combined96. The failure is now narrowed to WMMA math/fragment dependency
  or lane-value mapping in the real Q5 ABI.

WMMA/LDS fragment diagnostics:

- artifacts:
  `cache/hrxv1/gfx1151/wmma-lds-k2-contract-20260619-040138/`,
  `cache/hrxv1/gfx1151/wmma-lds-repeat-scale-20260619-040346/`, and
  `cache/hrxv1/gfx1151/wmma-prodstride-contract-20260619-040530/`.
- current validation sweep:
  `cache/hrxv1/gfx1151/wmma-lds-diagnostics-current-20260619-041040/`.
- one-K, two-K, and production stride44 LDS-fed fragment fixtures can produce
  NaNs in inactive odd accumulator slots;
- adding per-load `lgkmcnt(0)` waits did not clear those inactive-slot NaNs;
- selected even output slots remained finite in the generic, long-repeat, and
  production-stride fixtures.

Interpretation:
the full64 catalog route's real output NaNs are not explained by missing or
duplicate output coordinates, simple LDS wait placement, or the stride44
fragment load shape alone. The next useful reproducer needs the exact Q5
dequant/shared-layout source context and register-pressure shape, or a
lower-level cooperative-matrix spelling closer to RADV.

Exact-kernel reproducer:

- source:
  `hrx-hip-bench-q5-wmma-full64-repro`, a CMake-built wave64 bench that
  includes and launches the real
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32`
  catalog kernel on synthetic Q5 blocks and F32 RHS. It also contains
  reduced-active-group diagnostic kernels that keep the same Q5 dequant,
  stride44 LDS staging, b64 fragment loads, and store contract while varying
  the number of live full64 output groups.
- artifact:
  `cache/hrxv1/gfx1151/q5-full64-exact-repro-profiles-20260619-041650/`.
- active-group artifact:
  `cache/hrxv1/gfx1151/q5-full64-active-groups-repro-20260619-042113/`.
- batched4 artifact:
  `cache/hrxv1/gfx1151/q5-full64-batched4-repro-20260619-042415/`.
- result:
  selected outputs do reproduce the real failure. The small-scale profile
  reported NaNs with no Infs for the p33 cases `rows=64, cols=33, k=256`,
  `rows=64, cols=33, k=512`, and `rows=64, cols=33, k=3584`; the stress
  profile also reported NaNs/Infs for the same p33 and p64 cases.
- active-group result:
  on `rows=64, cols=33, k=3584` with the small-scale profile, `active1` and
  `active4` were finite, `active8` stayed finite but had a large mismatch, and
  `active12`/`active16` reproduced selected-output NaNs.
- batched4 result:
  computing all sixteen output groups four at a time produced no NaNs/Infs for
  the p33 small-scale case and matched the finite error scale of `active4`.

Interpretation:
the exact Q5 dequant/shared-layout source context plus the full64 live
accumulator topology is sufficient to reproduce selected-output NaNs without
requiring stress-scale overflow. The failure starts before or at the high
live-accumulator regime rather than at store-coordinate ownership. The next
candidate should stop scaling one HIP wave to 12+ live f16 WMMA accumulators
and instead pursue low live-accumulator group batching or the RADV medium
schedule's smaller WMMA count, staged halfword writeback, and
cooperative-matrix lane ownership.

Static comparison to the p33 RADV medium oracle:

- SGPR `64` vs RADV `108`;
- VGPR `198` vs RADV `144`;
- LDS `11264` matches;
- no spills;
- `32` WMMA vs RADV `16`;
- `64 ds_load_b64` vs RADV `48`;
- `64 buffer_store_b32` vs RADV `96`;
- no `ds_load_u16_d16` store stage vs RADV `64`;
- `3` barriers vs RADV `2`;
- first-window `24` pre-WMMA `ds_load_b64` and final `lgkmcnt(0)` vs RADV
  `48` and `lgkmcnt(40)`.

Decision:
reject before timing/model tests. Full output ownership alone is not enough;
the HIP spelling moves to high VGPR pressure, doubles the visible WMMA count,
and still does not reproduce RADV's cooperative load/store schedule. The next
p33 route should not keep scaling live f16 WMMA accumulators in one wave; it
should isolate the RADV-style fragment/lane dependency path with valid full
medium-tile output ownership.

## Rejected p33 Batched4 Low-Live-Accumulator Catalog Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_batched4_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_BATCHED4_F16ACC_WG256_PROMPT=1`

Artifacts:

- correctness:
  `cache/hrxv1/gfx1151/q5-p33-batched4-catalog-20260619-043426/`
- focused timing:
  `cache/hrxv1/gfx1151/q5-p33-batched4-focused-perf-20260619-043508/`
- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_batched4_wg256.hsaco`

What changed:

- kept the p33 medium BM64/BN64/BK32/WG256/wave64/padded44 shape;
- kept full medium-tile output ownership;
- computed all sixteen output groups four at a time, matching the finite
  exact-kernel diagnostic and avoiding the full64 high-live-accumulator
  topology.

Focused correctness:

- route trace selected the batched4 provider for all four Qwen2.5 Coder Q5_K_M
  p33 rows;
- CPU-reference passed all four rows.

Static selected-symbol facts:

- SGPR `51`, VGPR `186`, LDS `11264`, no spills, wave64;
- `32` WMMA;
- `256 ds_load_b64`;
- `64 buffer_store_b32`;
- `12` barriers.

Focused timing versus default rows2/cols8:

| Row | Default | Batched4 | Decision |
| --- | ---: | ---: | --- |
| Kcur | `72.754 us` | `1386.552 us` | reject |
| Qcur | `259.498 us` | `1771.570 us` | reject |
| ffn_out | `1535.251 us` | `12762.069 us` | reject |
| ffn_gate | `1715.737 us` | `8246.900 us` | reject |

Interpretation:
the live-accumulator bracket is real: reducing live f16 WMMA accumulators fixes
the selected-output NaNs in the real catalog ABI. But batched4 fixes it by
restaging A/B four times, so it explodes LDS loads and barriers and is not a
performance candidate. The next Q5 p33 candidate should preserve the low-live
property without the reload waste, or should pursue RADV's cooperative
matrix-store/writeback topology directly.

## Rejected p33 Wave4row Low-Live-Accumulator Catalog Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_wave4row_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_WAVE4ROW_F16ACC_WG256_PROMPT=1`

Artifacts:

- nowait correctness failure:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-catalog-20260619-044305/`
- waited-load correctness pass:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-wait-catalog-20260619-044416/`
- focused timing:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-focused-perf-20260619-044505/`
- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_wave4row_wg256.hsaco`

What changed:

- kept the p33 medium BM64/BN64/BK32/WG256/wave64/padded44 shape;
- kept full medium-tile output ownership;
- used all four waves in the workgroup: each wave owns one 16-row stripe and
  four column groups;
- kept only four live f16 WMMA accumulators per wave;
- staged A/B once per K tile, avoiding batched4's four full restaging passes.

Correctness:

- the first `nowait` B64 LDS fragment-load spelling selected for all focused
  rows but failed CPU-reference with finite errors on Kcur/Qcur/ffn_gate and a
  NaN on ffn_out;
- switching to waited B64 fragment-load helpers passed all four focused p33
  rows with route evidence.

Static selected-symbol facts for the waited form:

- SGPR `32`, VGPR `102`, LDS `11264`, no spills, wave64;
- `8` visible WMMA;
- `40 ds_load_b64`;
- `16 buffer_store_b32`;
- `2` barriers;
- `51 s_waitcnt`.

Focused timing versus default rows2/cols8:

| Row | Default | Wave4row | Decision |
| --- | ---: | ---: | --- |
| Kcur | `72.210 us` | `398.540 us` | reject |
| Qcur | `259.574 us` | `474.473 us` | reject |
| ffn_out | `1538.760 us` | `3406.255 us` | reject |
| ffn_gate | `1721.514 us` | `2380.085 us` | reject |

Interpretation:
wave4row is the first direct-F32 Q5 p33 WMMA catalog route in this thread that
is both correctness-clean and structurally close to the RADV medium route. It
uses all four waves, avoids the full64 live-accumulator cliff, and avoids
batched4's restaging explosion. It still loses to the accepted rows2/cols8
route, so it is not a production candidate. The next Q5 p33 path should stop
reshuffling direct-F32 WMMA ownership and instead target the remaining RADV
cooperative fragment/store contract or a packed-Q8_1 medium schedule.

## Vulkan Oracle

Artifact:
`cache/hrxv1/gfx1151/q5-radv-vs-hrx-wmma-vk128-padded-w64-isa-20260618/`.

RADV pipeline:

- name: `matmul_q5_k_f32_f16acc_aligned_l`
- hash: `0x0ee599afb33ff07b`
- spec: `[256,128,128,32,64,64,2,16,16,16,64]`
- workgroup denominators: `[128,128,1]`
- tile shape: BM128/BN128/BK32, WG256, wave64-style subgroup policy
- resources: SGPR `108`, VGPR `192`, LDS `22528`, no spills
- opcodes: `32` `v_wmma_f16_16x16x16_f16`, `64` `ds_load_b64`,
  `128` `ds_load_u16_d16`, `128` `ds_store_b16`, `192`
  `buffer_store_b32`, `2` barriers

Odd/tail oracle artifacts:

- p33/fa1:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-20260618-063510/`
- p513/fa1:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p513-fa1-20260618-063522/`

p33 medium route:

- name: `matmul_q5_k_f32_f16acc_aligned_m`
- hash: `0x0ee599afb33ff07b`
- spec: `[128,64,64,32,64,32,2,16,16,16,64]`
- workgroup denominators: `[64,64,1]`
- dispatches: `166`
- resources: SGPR `108`, VGPR `144`, LDS `11264`, no spills
- opcodes: `16` `v_wmma_f16_16x16x16_f16`, `48` `ds_load_b64`,
  `64` `ds_load_u16_d16`, `64` `ds_store_b16`, `96`
  `buffer_store_b32`, `2` barriers
- SPIR-V: `OpCapability CooperativeMatrixKHR` with
  `OpCooperativeMatrixLoadKHR`, `OpCooperativeMatrixMulAddKHR`, and
  `OpCooperativeMatrixStoreKHR`

p513 large/tail route:

- name: `matmul_q5_k_f32_f16acc_aligned_l`
- hash: `0x0ee599afb33ff07b`
- spec: `[256,128,128,32,64,64,2,16,16,16,64]`
- workgroup denominators: `[128,128,1]`
- dispatches: `166`
- representative workgroups: `[28,5,1]` for Qcur and `[148,5,1]` for
  FFN gate/up
- resources: SGPR `108`, VGPR `192`, LDS `22528`, no spills
- opcodes: `32` `v_wmma_f16_16x16x16_f16`, `64` `ds_load_b64`,
  `128` `ds_load_u16_d16`, `128` `ds_store_b16`, `192`
  `buffer_store_b32`, `2` barriers
- tail reduction: `split_k_reduce`, `56` dispatches total across Q5/Q6
  narrow-width rows, representative workgroups `[257,1,1]`, SGPR `108`,
  VGPR `12`, LDS `0`, no spills, `83` instructions

Conclusion:
p33, p512, and p513 are not one route. Vulkan uses the medium aligned family
for p33, returns to the large aligned family for p512/p513, and uses a separate
split-K reduction path for production-width tails. HRX Q5 promotion must be
gated on all three regimes.

## Rejected B-quad Split-K wg1024 Diagnostic

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_splitk_part_wg256_f32` plus
`hrx_split_k_reduce_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BQUAD_SPLITK_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-bquad-splitk-reduce-wg1024-focused-20260618-172949/`

What changed:

- preserved the accepted Q5 B-quad MMQL128 partial producer
- kept the opt-in two-part split-K dispatcher for large p513 Q5 rows
- changed only `hrx_split_k_reduce_f32` dispatch geometry from 256 to 1024
  threads per workgroup, matching the RADV `split_k_reduce`
  `wg_denoms=[1024,1,1]`

Focused correctness:

- p513 CPU-reference gate passed all four Q5 rows
- Kcur stayed on `hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32`
- Qcur, ffn_out, and ffn_gate each selected two split-K partial producers
  followed by `hrx_split_k_reduce_f32`
- reduce grids changed to the expected 1024-thread geometry:
  `n=1838592 -> [1796,1,1]` and `n=9718272 -> [9491,1,1]`

Focused timing:

| Row | Default | Split-K wg1024 | Ratio |
| --- | ---: | ---: | ---: |
| Kcur p513 | 896.241 us | 891.726 us | 0.995x |
| Qcur p513 | 1507.874 us | 1643.134 us | 1.090x |
| ffn_out p513 | 8943.630 us | 8887.670 us | 0.994x |
| ffn_gate p513 | 8384.141 us | 11830.322 us | 1.411x |
| summed rows | 19731.885 us | 23252.851 us | 1.178x |

Decision:
reject for production. The wg1024 change is the correct Vulkan-oracle launch
geometry and reduces the earlier rejected split-K row sum from `54.15 ms` to
`23.25 ms`, but the diagnostic still loses to the default `19.73 ms` row sum.
The remaining issue is not reduce workgroup count; it is that the HRX spelling
does two full global partial writes and a separate global F32 reduce instead of
the locality and scheduling used by the winning Vulkan path.

## Rejected Direct-F32 VK128 Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`

Evidence:

- artifact:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-focused-20260618-025116/`
- ISA artifact:
  `cache/hrxv1/gfx1151/q5-radv-vs-hrx-wmma-vk128-padded-w64-isa-20260618/`
- emitted shape: wave64, SGPR `32`, VGPR `164`, LDS `20480`, no spills,
  `32` WMMA sites, `32 ds_load_b128`, `64 global_store_b32`
- correctness: p512 and p513 passed; p33 stayed on the narrow rows2 route
- timing p512 regressed versus current packed-Q8_1 routing:
  Kcur `866.67 -> 971.27 us`, Qcur `1215.98 -> 2476.18 us`,
  ffn_out `7191.96 -> 17643.78 us`, ffn_gate
  `6332.38 -> 14489.87 us`

Decision:
reject for production. This route matches the headline BM128/BN128/BK32/WG256
and WMMA count, but not the RADV LDS/read/writeback schedule.

## Rejected B64GROUP Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-focused-20260618-054536/`

What changed:

- preserved the direct-F32 VK128 padded wave64 route shape
- changed LDS fragment reads from compiler-selected `ds_load_b128` to grouped
  `ds_read_b64` loads before each 4x4 WMMA block
- kept this as an opt-in catalog route, not a default

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_wg256.hsaco`
- wave64, SGPR `32`, VGPR `199`, LDS `20480`, no spills
- emitted `32` WMMA sites, `64 ds_load_b64`, `0 ds_load_b128`, `2` barriers,
  `64 global_store_b32`

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed
- p33 correctly stayed on `hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32`
- p512 and p513 selected the B64GROUP candidate

Focused timing:

| Shape | Row | Current route | B64GROUP | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | Kcur | 887.78 us | 1046.75 us | 1.18x |
| p512 | Qcur | 1232.95 us | 2471.40 us | 2.00x |
| p512 | ffn_out | 7308.67 us | 17817.09 us | 2.44x |
| p512 | ffn_gate | 6662.22 us | 16905.31 us | 2.54x |
| p513 | Kcur | 891.04 us | 964.61 us | 1.08x |
| p513 | Qcur | 1594.35 us | 3375.88 us | 2.12x |
| p513 | ffn_out | 9362.91 us | 23583.29 us | 2.52x |
| p513 | ffn_gate | 8558.62 us | 20055.20 us | 2.34x |

Decision:
reject for production. Matching RADV's `64 ds_load_b64` and `32` WMMA axes is
not enough. The remaining large deltas are still the LDS footprint, `ds_load_u16`
path, LDS b16 stores, cooperative/global writeback count, and lane ownership.

## Rejected Fullstore Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_fullstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk128-fullstore-focused-20260618-065217/`

What changed:

- preserved the direct-F32 VK128 padded wave64 route shape
- removed guarded per-element edge checks for full in-bounds 16x16 output
  tiles
- kept edge fallback for tails, so p513 remains covered
- kept this as an opt-in catalog route, not a default

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_wmma16_vk128_padded_w64_fullstore_wg256.hsaco`
- wave64, SGPR `32`, VGPR `164`, LDS `20480`, no spills
- emitted `32` WMMA sites, `128 global_store_b32`, and `2` barriers

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed
- p33 correctly stayed on `hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32`
- p512 and p513 selected the fullstore candidate

Focused timing:

| Shape | Row | Current route | Fullstore | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | Kcur | 865.99 us | 965.08 us | 1.11x |
| p512 | Qcur | 1237.64 us | 2344.02 us | 1.89x |
| p512 | ffn_out | 7204.77 us | 16963.43 us | 2.35x |
| p512 | ffn_gate | 6912.03 us | 14368.33 us | 2.08x |
| p513 | Kcur | 890.41 us | 887.88 us | 1.00x |
| p513 | Qcur | 1595.89 us | 2510.60 us | 1.57x |
| p513 | ffn_out | 9548.74 us | 18271.78 us | 1.91x |
| p513 | ffn_gate | 8779.65 us | 16906.24 us | 1.93x |

Decision:
reject for production. Moving the HIP route from `64` to `128` global stores
does not recover the Vulkan schedule. The remaining mismatch is structural:
RADV is using cooperative matrix load/store and lane ownership that produces
`192 buffer_store_b32` plus the LDS load/store pattern. The next direct-WMMA
attempt should target that exact writeback mechanism, or use a lower-level
source form that can emit it.

## Rejected VK64 GROUPK2 Probes

Routes:

- `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_groupk2_f16acc_wg256_f32`
- `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_groupk2_wait_f16acc_wg256_f32`

Gates:

- `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_GROUPK2_F16ACC_WG256_PROMPT=1`
- `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_GROUPK2_WAIT_F16ACC_WG256_PROMPT=1`

Artifacts:

- ISA:
  `cache/hrxv1/gfx1151/q5-vk64-groupk2-isa-20260619-023058/`
- focused p33 correctness/route trace:
  `cache/hrxv1/gfx1151/q5-vk64-groupk2-focused-20260619-023220/`

What changed:

- ported the prior Q6 VK64 GROUPK2 diagnostic into Q5
- kept BM64/BN64/BK32/WG256/wave64/padded44 and 11264-byte LDS
- each wave owns one 16-column tile and the four row subtiles
- preloads both K tiles of four A fragments plus one B fragment through
  explicit b64 LDS reads before the eight WMMAs
- added both nowait and per-load-wait variants

Static ISA:

| Metric | RADV p33 | GROUPK2 | GROUPK2_WAIT |
| --- | ---: | ---: | ---: |
| SGPR | 108 | 32 | 32 |
| VGPR | 144 | 123 | 123 |
| LDS | 11264 | 11264 | 11264 |
| WMMA sites | 16 | 8 | 8 |
| `ds_load_b64` | 48 | 40 | 40 |
| pre-WMMA `ds_load_b64` | 48 | 24 | 24 |
| final pre-WMMA `lgkmcnt` | 40 | 0 | 0 |
| `ds_load_u16_d16` | 64 | 0 | 0 |
| `ds_store_b16` | 64 | 2 | 2 |
| `buffer_store_b32` | 96 | 0 | 0 |
| `global_store_b32` | 0 | 16 | 16 |
| `s_waitcnt` | 111 | 11 | 51 |
| `s_waitcnt_depctr` | 2 | 42 | 42 |

Focused correctness:

- default p33 CPU-reference passed all four Q5 rows
- GROUPK2 selected on all four rows and failed all four:
  Kcur `ERR=0.293269939`, Qcur `ERR=0.117790458`, ffn_out NaN, and ffn_gate
  `ERR=0.129065323`
- GROUPK2_WAIT selected on all four rows and passed Kcur/Qcur, but failed
  ffn_out `ERR=0.000916021` and ffn_gate `ERR=0.004372403`

Decision:
reject before timing/model tests. This branch proves that the Q6-style grouped
fragment spelling can force b64 LDS traffic in Q5, but it does not reach the
RADV issue shape and is not numerically correct. The next Q5 direct-WMMA step
should not be another local VK64 grouped-load variant; it should either build a
lower-level cooperative matrix load/store/lane-ownership fixture or a different
source form that can emit 16 visible WMMA sites, high pre-WMMA `lgkmcnt`, the
d16 LDS load/store/writeback topology, and the RADV-style buffer-store path.

## Rejected VK64 Padded44 Medium Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk64-padded44-medium-focused-20260618-064232/`

What changed:

- parameterized the existing Q5 VK128 direct-F32 WMMA source for BM/BN
- added a BM64/BN64 wrapper with padded44 LDS rows and wave64 compile flags
- kept the route opt-in and constrained to `16 <= cols <= 64`

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_wg256.hsaco`
- wave64, SGPR `32`, VGPR `75`, LDS `11264`, no spills
- emitted `8` WMMA sites, `2` `ds_store_b16`, `16` global stores, and `2`
  barriers

Focused correctness:

- p33 CPU-reference gate passed and selected the VK64 candidate
- p512 and p513 CPU-reference gates passed and did not select VK64, preserving
  the existing production-width route

Focused timing:

| Row | Current rows2_cols8 | VK64 padded44 | Ratio |
| --- | ---: | ---: | ---: |
| Kcur p33 | 71.92 us | 347.35 us | 4.83x |
| Qcur p33 | 331.19 us | 458.33 us | 1.38x |
| ffn_out p33 | 2583.19 us | 3082.48 us | 1.19x |
| ffn_gate p33 | 2330.04 us | 1742.43 us | 0.75x |

Decision:
reject for production. This is a useful negative result because it proves that
matching the medium route's BM64/BN64/LDS footprint is not enough. The emitted
HIP schedule is still far from RADV: `8` versus `16` WMMA sites, `16` versus
`96` stores, and much lower VGPR pressure. The next Q5 medium attempt should
not be another wrapper-only clone; it must address cooperative-matrix
store/lane ownership or use a different source form that emits the RADV-like
store and fragment schedule.

## Rejected MMQL128 B-Pair Packed Probe

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bpair_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BPAIR_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-mmql128-bpair-focused-20260618/`

What changed:

- preserved the current Q5 packed-Q8_1/x4 MMQL128 dataflow: BM128, BN128,
  wave64, BK_STEP1, TM4, TN2, WNITER8;
- bracketed the Q4_K B-cache clustering win with the lower-live-state neighbor
  of rejected Q5 B-quad;
- preloaded only the two B-cache rows for each TN=2 micro-iteration before dot
  consumption;
- kept the selector opt-in and full-tile only, so p33 and p513 remained on
  existing routes during the first gate.

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_q8_1_x4_mmql128_bpair.hsaco`;
- wave64, SGPR `50`, VGPR `149`, LDS `10240`, no spills.

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p512 selected B-pair for the three packed Q5 rows;
- p33 stayed on `rows2_cols8`/MMQ64;
- exact p513 tails fell back to current MMQL128.

Focused p512 timing:

| Row | Current | B-pair | Ratio |
| --- | ---: | ---: | ---: |
| Kcur | 870.47 us | 876.16 us | 1.007x |
| Qcur | 1220.50 us | 1268.61 us | 1.039x |
| ffn_out | 7342.03 us | 7576.46 us | 1.032x |
| ffn_gate | 6940.84 us | 6729.86 us | 0.970x |

Decision:
reject for production. The Q4 B-cache read-clustering axis does not transfer
to Q5_K at pair or quad cluster size. B-pair lowers VGPR pressure versus the
rejected B-quad (`149` vs `169`) but still regresses three of four p512 rows.

## Rejected MMQL128 CR Issue-Order Probe

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_cr_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_CR_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-mmql128-cr-focused-20260618/`

What changed:

- preserved the current Q5 packed-Q8_1/x4 MMQL128 dataflow: BM128, BN128,
  WG256, wave64, BK_STEP1, TM4, TN2, WNITER8;
- changed only the inner dot issue order to consume one Q5 row cache in
  `cr`-major order across columns before advancing to the next row-cache slot;
- kept the selector opt-in and full-tile only, so p33 stayed on existing
  narrow routes and exact p513 tails fell back to current MMQL128.

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_q8_1_x4_mmql128_cr.hsaco`;
- wave64, SGPR `50`, VGPR `192`, LDS `10240`;
- private segment `452`, VGPR spills `135`;
- preserved `512 v_dot4_i32_iu8`, `64 global_store_b32`, and `2` barriers,
  but raised wait count pressure to `230 s_waitcnt`.

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p512 selected CR for Qcur, ffn_out, and ffn_gate;
- p33 stayed on `rows2_cols8`/MMQ64;
- exact p513 tails fell back to current MMQL128.

Focused p512 timing:

| Row | Current | CR | Ratio |
| --- | ---: | ---: | ---: |
| Kcur | 876.79 us | 876.52 us | 1.000x |
| Qcur | 1225.57 us | 3832.51 us | 3.127x |
| ffn_out | 7453.69 us | 21017.01 us | 2.820x |
| ffn_gate | 7161.45 us | 19333.78 us | 2.700x |

Decision:
reject for production. The CR-major issue-order axis explodes live range and
spills without moving the packed path toward the Vulkan oracle. Do not continue
row-major issue order unless it is paired with a different B ownership or
prefetch strategy that eliminates the spill wall.
The next Q5 packed-path attempt should be Q5-specific, not another local
B-cache cluster-size clone.

## Next Q5_K Work

Do not repeat standalone LDS-load reshaping or local B-cache cluster-size
clones. The next Q5 direct-WMMA attempt needs to address the remaining RADV
deltas together, especially cooperative matrix global store/lane ownership or
a lower-level equivalent. The standalone `ds_load_b64`, fullstore, medium p33
wrapper, B-quad, and B-pair axes have now been tested and rejected. Return to
the cooperative-matrix store/lane ownership problem, or to a Q5-specific
packed-Q8_1 schedule axis with focused p33/p512/p513 gates.

New mechanical target from the p513 SPIR-V/ISA inspection:

## Rejected p33 MMQL64 BK2 TN4/BQUAD Ownership Probe

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_tn4_bquad_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL64_BK2_TN4_BQUAD_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-mmql64-bk2-tn4-bquad-focused-20260619-051644/`

What changed:

- preserved the accepted p33 packed route's BM64/BN64/BK_STEP=2/WG256/wave64
  dataflow;
- kept the BQUAD issue window;
- changed only per-lane output ownership from `TM4/TN2` to `TM2/TN4`, testing
  a Q5-specific packed RHS/column-ownership axis after BK1, B-pair, B-quad,
  and CR-major issue-order probes.

Compile evidence:

- TN4/BQUAD: wave64, SGPR `62`, VGPR `192`, VGPR spills `183`,
  private segment `736`, LDS `10240`;
- accepted BQUAD: wave64, SGPR `62`, VGPR `124`, no spills, private segment
  `0`, LDS `10240`;
- TN4 preserved `256` `v_dot` sites and `2` barriers but increased
  `ds_load` count `40 -> 50` and `s_waitcnt` count `119 -> 249`.

Decision:
reject before focused correctness/perf. This ownership pivot reaches the
gfx1151 register cliff and is not a viable packed-medium schedule. The Q5 p33
packed path has now rejected BK1, B-pair, TN4/BQUAD, CR-major issue order, and
the broad B-cache window itself. The remaining packed-route work needs a
different contract than local output ownership or B-cache live-window changes.

- full in-bounds path:
  RADV loads the half accumulator cooperative matrix, converts it to a float
  cooperative matrix, and emits
  `OpCooperativeMatrixStoreKHR` directly to the output buffer with `rows` as
  the matrix stride.
- non-aligned in-bounds path:
  RADV emits `OpCooperativeMatrixStoreKHR` to a `coopmat_stage` LDS buffer,
  barriers, `ds_load_u16_d16`, half-to-float conversion, and scalar
  `buffer_store_b32`.
- edge path:
  RADV uses the same LDS staging path but predicates scalar stores for partial
  rows/columns.

The current HIP C++ direct-WMMA route has only manual scalar stores from the
WMMA accumulator. The next exact-clone attempt should therefore not be another
manual per-tile store-stage route: the Q5 B64GROUP+store-stage probe below
matches the LDS footprint and some halfword staging but pays too many barriers.

## Rejected B64GROUP Store-Stage Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_store_stage_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_STAGE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-store-stage-focused-20260618-110958/`

What changed:

- preserved the direct-F32 VK128 padded wave64 route shape;
- preserved grouped LDS fragment reads from the rejected B64GROUP probe;
- added a half accumulator LDS store-stage before writing F32 output;
- kept the route opt-in and built through CMake/Ninja.

Compile evidence:

- wave64, SGPR `32`, VGPR `198`, LDS `22528`, no spills;
- emitted `32` `v_wmma_f16_16x16x16_f16`;
- emitted `64 ds_load_b64`, `64 ds_load_u16_d16`, `66 ds_store_b16`,
  `64 global_store_b32`, and `34` barriers.

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on `rows2_cols8`;
- p512 and p513 selected the store-stage candidate.

Focused timing:

| Shape | Row | Current | Store-stage | Ratio |
| --- | ---: | ---: | ---: | ---: |
| p512 | Kcur | 865.46 us | 1060.91 us | 1.23x |
| p512 | Qcur | 1224.57 us | 4819.52 us | 3.94x |
| p512 | ffn_out | 7227.94 us | 19946.42 us | 2.76x |
| p512 | ffn_gate | 6906.12 us | 30089.16 us | 4.36x |
| p513 | Kcur | 887.97 us | 1018.55 us | 1.15x |
| p513 | Qcur | 1577.88 us | 5254.41 us | 3.33x |
| p513 | ffn_out | 9533.36 us | 23344.87 us | 2.45x |
| p513 | ffn_gate | 9214.98 us | 33125.13 us | 3.59x |

Decision:
reject for production. This is the closest HIP C++ direct-WMMA clone on
resource shape so far: it reaches RADV's `22528` byte LDS footprint, `32`
WMMA count, and `64 ds_load_b64`. It still misses RADV's low-barrier
cooperative-matrix store/lane ownership: RADV has `2` barriers and `192`
`buffer_store_b32`, while this HIP C++ spelling has `34` barriers and only
manual scalar stores. The next exact-clone attempt needs a lower-level
cooperative-store/lane-ownership spelling, not another source-level manual
stage.

## Rejected B64GROUP Fullstore Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_fullstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_FULLSTORE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-fullstore-focused-20260618-111743/`

What changed:

- preserved the direct-F32 VK128 padded wave64 route shape;
- preserved grouped LDS fragment reads from the rejected B64GROUP probe;
- added full in-bounds scalar direct stores, avoiding the barrier-heavy manual
  store-stage path;
- kept the route opt-in and built through CMake/Ninja.

Compile evidence:

- wave64, SGPR `32`, VGPR `198`, LDS `20480`, no spills;
- emitted `32` `v_wmma_f16_16x16x16_f16`;
- emitted `64 ds_load_b64`, `128 global_store_b32`, and `2` barriers.

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on `rows2_cols8`;
- p512 and p513 selected the fullstore candidate.

Focused timing:

| Shape | Row | Current | B64GROUP fullstore | Ratio |
| --- | ---: | ---: | ---: | ---: |
| p512 | Kcur | 867.19 us | 1058.19 us | 1.22x |
| p512 | Qcur | 1230.26 us | 4903.45 us | 3.99x |
| p512 | ffn_out | 7238.35 us | 19727.06 us | 2.73x |
| p512 | ffn_gate | 6733.11 us | 29961.79 us | 4.45x |
| p513 | Kcur | 898.34 us | 1013.67 us | 1.13x |
| p513 | Qcur | 1587.39 us | 5277.83 us | 3.33x |
| p513 | ffn_out | 9597.82 us | 23037.40 us | 2.40x |
| p513 | ffn_gate | 9032.34 us | 32848.22 us | 3.64x |

Decision:
reject for production. This rules out the cheap low-barrier approximation of
RADV's full-tile path. Even with RADV-like grouped `ds_load_b64`, full-tile
direct stores, only `2` barriers, and no spills, the manual HIP C++ accumulator
ownership still loses badly to the packed-Q8_1 production route. The remaining
direct-F32 target is specifically cooperative-matrix store/lane ownership and
RADV's `192 buffer_store_b32` writeback shape.

## rocWMMA Cooperative Store Probe

Artifact:
`cache/hrxv1/gfx1151/rocwmma-coopstore-probe-20260618/`

Purpose:
test whether the installed HIP C++/rocWMMA stack exposes a source-level route
to the missing RADV cooperative accumulator store/lane-ownership path.

Probe source:
`rocwmma_coopstore_variants.hip.cpp`

Compile matrix:

- default accumulator store:
  `fragment<accumulator, 16, 16, 16, float, row_major>` compiled with
  `-mwavefrontsize64`;
- cooperative row accumulator store:
  `fragment<accumulator, 32, 32, 16, float, row_major,
  coop_row_major_2d<2,2>>` failed template instantiation;
- cooperative col accumulator store:
  `fragment<accumulator, 32, 32, 16, float, row_major,
  coop_col_major_2d<2,2>>` failed template instantiation;
- `single<2,2,0>` compiled, but is only one-wave ownership.

Header evidence:

- `rocwmma/internal/io_layout.hpp` has an accumulator `MaxVWSelector`
  assertion that non-interleaved accumulator I/O is not cooperative.
- Interleaved accumulator layouts can instantiate the cooperative scheduler
  path, but the row/col 2x2 forms resolve an invalid `SplitK=0` in the
  current headers.
- `rocwmma/internal/opaque_store.hpp` implements public stores through
  vector/scalar memory stores from fragment access, not through a
  cooperative-matrix store intrinsic.

ISA evidence:

- default store compiled as wave64, SGPR `4`, VGPR `4`, LDS `0`, no spills;
  emitted `8 global_store_b32`, no `buffer_store_b32`, no `ds_store`, no
  barrier, and no WMMA.
- `single<2,2,0>` compiled as wave64, SGPR `4`, VGPR `5`, LDS `0`, no spills;
  emitted the default kernel plus a guarded single-wave store kernel with
  another `8 global_store_b32`.
- cooperative row/col accumulator forms failed at compile time with
  `ColInlineInt<32, 32, float, 16, 0>` and division-by-zero `SplitK`
  diagnostics.

Decision:
do not build a production Q5 route on rocWMMA public accumulator stores. The
probe did not expose the Vulkan `OpCooperativeMatrixStoreKHR` lowering that
RADV uses for this direct-F32 oracle path. The exact direct-F32 clone now
requires either a lower-level compiler/IR mechanism that can express
cooperative matrix store semantics for AMDGPU, or a deliberate pivot back to
the existing packed-Q8_1 production dataflow with Q5-specific schedule axes.

## Rejected B64GROUP Store-Batch8 Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_store_batch8_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_BATCH8_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-store-batch8-20260618/`

What changed:

- preserved the rejected B64GROUP store-stage direct-F32 WMMA math/load path;
- kept BM128/BN128/BK32/WG256, wave64, grouped `ds_load_b64`, and f16 WMMA;
- changed writeback staging from one LDS stage plus a barrier pair per tile to
  one LDS stage plus a barrier pair per eight tiles;
- kept the route opt-in and built through CMake/Ninja.

Compile evidence:

- wave64, SGPR `32`, VGPR `214`, LDS `36864`, no spills;
- emitted `32` `v_wmma_f16_16x16x16_f16`;
- emitted `64 ds_load_b64`, `64 ds_load_u16_d16`, `66 ds_store_b16`,
  `64 global_store_b32`, and `6` barriers.

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on `rows2_cols8`;
- p512 and p513 selected the store-batch8 candidate for all rows.

Focused timing:

| Shape | Row | Current | Store-batch8 | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | Kcur | 864.14 us | 1043.49 us | 1.208x |
| p512 | Qcur | 1223.22 us | 2499.40 us | 2.043x |
| p512 | ffn_out | 7229.01 us | 17649.52 us | 2.441x |
| p512 | ffn_gate | 6903.14 us | 16915.11 us | 2.450x |
| p513 | Kcur | 886.35 us | 960.72 us | 1.084x |
| p513 | Qcur | 1585.01 us | 3389.07 us | 2.138x |
| p513 | ffn_out | 9601.19 us | 23409.53 us | 2.438x |
| p513 | ffn_gate | 8720.37 us | 20101.28 us | 2.305x |

Decision:
reject for production. This probe confirms that reducing explicit
store-stage barriers from `34` to `6` is not enough. The route still misses
RADV's cooperative-matrix store/lane ownership, `192 buffer_store_b32`
writeback shape, and full `128/128` halfword LDS load/store topology while
raising LDS to `36864` and VGPR to `214`. The next direct-WMMA route should
not be another manual LDS-stage approximation; it needs a lower-level
cooperative-store spelling or a different schedule family.

## Rejected B64GROUP Bufferstore Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-bufferstore-20260618/`

What changed:

- preserved the B64GROUP direct-F32 WMMA math/load path;
- ported the fixed gfx11 raw-buffer-store descriptor from the Q8_0 probe;
- replaced scalar global stores with raw `buffer_store_b32` output stores;
- kept the route opt-in and built through CMake/Ninja.

Compile evidence:

- wave64, SGPR `32`, VGPR `198`, LDS `20480`, no spills;
- emitted `32` `v_wmma_f16_16x16x16_f16`;
- emitted `64 ds_load_b64`, `0 ds_load_u16_d16`, `2 ds_store_b16`,
  `128 buffer_store_b32`, `0 global_store_b32`, and `2` barriers.

Focused correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on `rows2_cols8`;
- p512 and p513 selected the bufferstore candidate for all rows.

Focused timing:

| Shape | Row | Current | Bufferstore | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | Kcur | 871.11 us | 1038.82 us | 1.193x |
| p512 | Qcur | 1232.52 us | 2388.60 us | 1.938x |
| p512 | ffn_out | 7322.86 us | 17957.57 us | 2.452x |
| p512 | ffn_gate | 7126.12 us | 16776.44 us | 2.354x |
| p513 | Kcur | 897.54 us | 959.48 us | 1.069x |
| p513 | Qcur | 1590.42 us | 3315.05 us | 2.084x |
| p513 | ffn_out | 9404.53 us | 23052.67 us | 2.451x |
| p513 | ffn_gate | 8409.39 us | 20001.36 us | 2.378x |

Decision:
reject for production. This is the Q5 analogue of the rejected Q8_0
B64GROUP-bufferstore exact-schedule probe. Raw buffer stores are correct and
source-visible, but matching `buffer_store_b32` plus grouped LDS loads still
does not recover RADV's cooperative-matrix store/lane ownership or halfword LDS
topology.

## Packed-Q5 Hot-Op Issue-Window Score

Artifact:
`cache/hrxv1/gfx1151/q5-packed-hotop-score-20260618-145056/`

Tool update:
`sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py` now records a
generic first hot-op score in addition to the WMMA-specific score. For packed
Q5 paths, the first hot op is `v_dot4_i32_iu8`, so this score exposes LDS-load
cadence before the first dot issue window.

Compared variants:

- default `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32`;
- rejected B-pair;
- rejected B-quad;
- rejected CR-major.

Score:

| Variant | VGPR | Spills | First-window pre-dot LDS loads | Final pre-dot `lgkmcnt` | Dot ops in window |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | `141` | `0` | `13` | `2` | `58` |
| B-pair | `149` | `0` | `15` | `4` | `60` |
| B-quad | `169` | `0` | `20` | `9` | `60` |
| CR-major | `192` | `135` | `28` | `17` | `48` |

Interpretation:
the rejected B-cache clustering variants did move the intended local schedule
metric: they widen the LDS-read window before the first dot block and raise
the pre-dot `lgkmcnt`. The problem is that Q5 timing still regressed, and the
CR-major extreme reaches the VGPR cliff and spills. This rules out simply
making the pre-dot B-cache window larger as the missing Q5_K parity axis.

Decision:
future Q5 packed-path work should not repeat B-pair/B-quad/CR-style read
clustering. The next Q5-specific packed route should change a different
contract, such as packed RHS ownership/layout, split-K or tail policy, or a
lower-level cooperative-store-capable path. Use this hot-op score as a
compile-evidence guard: a bigger pre-dot window is only useful if it does not
raise live state enough to lose focused p512/p513 timing.

## Rejected VK128 Packstage Fast-Half Selected Store Probe

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_selected_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_FAST_HALF_SELECTED_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q5-packstage-fast-half-selected-compile-20260618-212450/`
- focused correctness:
  `cache/hrxv1/gfx1151/q5-packstage-fast-half-selected-focused-20260618-212617/`

What changed:

- ported the Q8_0 direct-WMMA pack-stage/fast-half selected-store controls to
  Q5_K;
- kept the VK128 padded W64 B64GROUP direct-F32 WMMA math path;
- staged A/B into LDS with `ds_write_b32`;
- staged selected accumulator halves through LDS with b16 store/load;
- used raw `buffer_store_b32` output stores;
- kept the route opt-in and built through CMake/Ninja.

Compile evidence:

- wave64, SGPR `37`, VGPR `199`, LDS `22528`, no spills;
- emitted `32` `v_wmma_f16_16x16x16_f16`;
- emitted `64 ds_load_b64`, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  `128 buffer_store_b32`, and `2` barriers.

Focused correctness:

- p33 passed and stayed off the VK128 route:
  rows2 for Kcur/Qcur/ffn_out and the accepted VK64 narrow FFN-gate route for
  ffn_gate;
- p512 selected the candidate for all four Q5 rows and failed CPU-reference:
  Kcur ERR `1.9417`, Qcur ERR `3.9573`, ffn_out ERR `3.9666`, and ffn_gate
  ERR `3.8717`.

Decision:
reject before perf/model testing. This proves HIP C++ can now reach the RADV
large-route LDS and halfword opcode classes for Q5, but the selected-half
accumulator store/lane mapping is not semantically valid in this spelling, and
the route still emits `128` rather than RADV's `192` buffer stores. The next
direct-WMMA attempt must solve accumulator lane ownership/store semantics, not
only force the halfword LDS topology.

## BHALF Packed-Q8_1 RHS Scale Cache Probe

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bhalf_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BHALF_PROMPT=1`

Artifacts:

- static score:
  `cache/hrxv1/gfx1151/q5-bhalf-hotop-score-20260618-145959/`
- focused:
  `cache/hrxv1/gfx1151/q5-mmql128-bhalf-focused-20260618-150104/`
- model A/B:
  `cache/hrxv1/gfx1151/q5-mmql128-bhalf-model-ab-20260618-150334/`

Hypothesis:
after B-pair, B-quad, and CR-major ruled out simply widening the pre-dot
B-cache issue window, test a smaller Q5-specific packed RHS cache-layout axis.
BHALF preserves the current MMQL128 BM128/BN128 packed-Q8_1 dataflow and dot
issue order, but stores Q8_1 RHS `d/s` in LDS as original half payloads and
converts after shared load. This reduces LDS footprint and tests whether the
scale/sum cache layout is contributing to the remaining Q5 gap.

Implementation note:
the first wrapper build accidentally set both B-pair and B-quad prefetch
macros to `0`, while the shared `bquad` template had no no-prefetch/default
accumulation branch. The compiler optimized that source into a zero-fill
kernel with `LDS=0`, `VGPR=15`, and no `v_dot4_i32_iu8`. The template now has
an explicit default branch, and the rebuilt BHALF HSACO is a real MMQL kernel.

Compile evidence after the fix:

- wave64, SGPR `49`, VGPR `140`, LDS `9728`, no spills;
- `512 v_dot4_i32_iu8`;
- first hot-op score: `12` pre-dot LDS loads, final pre-dot `lgkmcnt=1`,
  `54` dot ops in the first window.

For comparison, default Q5 MMQL128 scores SGPR `50`, VGPR `141`, LDS `10240`,
no spills, `13` pre-dot LDS loads, final `lgkmcnt=2`, and `58` dot ops in the
first window. BHALF slightly reduces footprint but does not improve the local
dot issue window.

Focused correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on `rows2_cols8`;
- p512 selected BHALF for the three large packed Q5 rows;
- p513 stayed on the existing MMQL tail path.

Focused p512 timing:

| Row | Current | BHALF | Delta |
| --- | ---: | ---: | ---: |
| Kcur | 870.36 us | 884.22 us | +1.59% |
| Qcur | 1230.52 us | 1207.13 us | -1.90% |
| ffn_out | 7293.25 us | 7098.09 us | -2.68% |
| ffn_gate | 6711.83 us | 6739.96 us | +0.42% |

The four-row p512 sum improved by `1.10%`, but the model smoke did not hold
that gain. Same-binary Qwen2.5 Coder 7B Q5_K_M p512/fa1 model A/B:

| Current | BHALF | Ratio |
| ---: | ---: | ---: |
| 455.164 tok/s | 454.457 tok/s | 0.998x |

Decision:
reject for gfx1151 promotion. The RHS half-scale cache layout is a useful
negative result because it proves a small LDS/VGPR footprint reduction is not
enough, and it does not address the p513 large-tail route. Future packed-Q5
work should target a stronger schedule axis: tail/split-K policy, different
packed RHS ownership, or a lower-level cooperative-store-capable path informed
by the RADV oracle rather than another local cache-footprint tweak.

## B-Quad Large-Tail Policy

The p513 tail follow-up tested the direct analogue of the accepted Q4_K
B-quad tail policy. This is not a new kernel; it exposes the existing
`full_tile`/edge-tile split in
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_wg256_f32` for large Q5_K
tails while keeping full p512 tiles on the current MMQL128 route.

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_wg256_f32`

Tail rollback:
`GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL128_BQUAD_TAIL_PROMPT=1`

Artifacts:

- opt-in probe:
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-probe-20260618-150846/`
- opt-in model A/B:
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-model-ab-20260618-150954/`
- default regate:
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-default-regate-20260618-151118/`
- default model A/B:
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-default-model-ab-20260618-151218/`

Focused route/correctness:

- p33 passed and stayed on `rows2_cols8`;
- p512 passed and stayed on current MMQL128;
- p513 passed and selected B-quad for the three large Q5 rows;
- targeted rollback selected current MMQL128 for p513.

p513 default vs rollback focused timing:

| Row | Default B-quad Tail | Tail Rollback MMQL128 | Delta |
| --- | ---: | ---: | ---: |
| Kcur | 899.07 us | 900.48 us | -0.16% |
| Qcur | 1535.85 us | 1628.67 us | -5.70% |
| ffn_out | 9160.55 us | 9957.13 us | -8.00% |
| ffn_gate | 8266.31 us | 8560.03 us | -3.43% |

The four-row p513 focused sum improved by `5.63%`.

Same-binary Qwen2.5 Coder 7B Q5_K_M p513/fa1 default-vs-rollback A/B:

| Default B-quad Tail | Tail Rollback MMQL128 | Ratio |
| ---: | ---: | ---: |
| 405.465 tok/s | 393.579 tok/s | 1.030x |

Decision:
accept B-quad for gfx1151 large Q5_K prompt tails with `cols >= 512` and
`cols % 128 != 0`. Keep p33 on the narrow route and keep p512/full tiles on
current MMQL128, because the full-tile Q5 B-quad transfer was already rejected.
This is a schedule-led tail-policy lift, not an exact RADV split-K clone:
Vulkan p513 still shows `split_k_reduce` dispatches that HRX does not yet
reproduce. Future Q5 tail work should mine whether that reduction path can be
matched directly after broader large-tail model coverage is available.

## p513 Split-K Reduce Oracle Extract

Artifact:

```text
cache/hrxv1/gfx1151/split-k-reduce-oracle-summary-20260618-162944/
```

The source-controlled extractor is:

```text
sources/llama.cpp/tools/vulkan-oracle/extract_split_k_reduce.py
```

It pairs every Vulkan `split_k_reduce` dispatch in the Qwen2.5 Coder 7B
Q5_K_M p513/fa1 oracle trace with the preceding producer dispatch by matching
the producer output binding to the reduce input binding. All `56` reductions
paired successfully:

| Count | Producer | Src0 | Src1 | Dst | Producer WG | Reduce WG | Output elems | Factor | Scratch bytes |
| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 42 | `matmul_q5_k_f32_f16acc_aligned_l` | q5_K `[3584,512]` | f32 `[3584,513]` | f32 `[512,513]` | `[8,5,1]` | `[257,1,1]` | 262656 | 2 | 2101248 |
| 14 | `matmul_q6_k_f32_f16acc_aligned_l` | q6_K `[3584,512]` | f32 `[3584,513]` | f32 `[512,513]` | `[8,5,1]` | `[257,1,1]` | 262656 | 2 | 2101248 |

The scratch byte count equals `output_elems * factor * sizeof(float)`, so the
Vulkan contract is explicit: the matmul producer writes two F32 partials per
output element to scratch and `split_k_reduce` writes the final output. The
current HRX B-quad p513 tail route is an accepted local tail-policy lift, but
it is not a split-K clone. A future Q5 p513 parity candidate must either add
equivalent HRX split-K scratch/reduce behavior or document why a fused/single
pass substitute beats the Vulkan two-dispatch schedule on focused p513 gates.

## Rejected p33 BN48 Packed-Q8_1 Probe

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x48_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQ64X48_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q5-mmq64x48-focused-20260618-213624/`

Hypothesis:
the current p33 Q5 packed route uses the MMQ64 narrow path with
`BN64/COLS_PER_THREAD16`, so `cols=33` still stages and carries work for
columns 49-64. A BN48 variant preserves the existing row ownership and
integer-dot path while reducing that wasted narrow-column work.

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_q8_1_x4_mmq64x48.hsaco`;
- wave64, SGPR `38`, VGPR `132`, LDS `1728`, no private segment;
- opcode summary: `96` `v_dot4_i32`, `52` LDS reads, `12` stores, `2`
  barriers.

Focused correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 selected BN48 only for Qcur and ffn_out;
- p512 stayed on current MMQL128;
- p513 stayed on current B-quad tail routing;
- Kcur stayed on rows2 and ffn_gate stayed on the accepted VK64 narrow route.

p33 focused timing:

| Row | Current | BN48 | Delta |
| --- | ---: | ---: | ---: |
| Kcur | 73.13 us | 75.36 us | +3.04% |
| Qcur | 330.61 us | 318.00 us | -3.81% |
| ffn_out | 2586.46 us | 2758.25 us | +6.64% |
| ffn_gate | 1757.01 us | 1756.91 us | -0.01% |

Decision:
reject for production. Smaller BN is a real p33 Qcur improvement, but it
regresses the larger-K ffn_out row by more than it saves. Future packed-path
p33 work should not keep shrinking BN blindly; it needs a K-dependent
issue/staging hypothesis or a selector/kernel split that improves Qcur without
hurting ffn_out.

## Rejected Qcur-Only BN48 Selector Split

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x48_wg256_f32`

Temporary policy tested:
default BN48 only for the exact Qwen2.5 Coder Q5_K_M Qcur geometry
`k=3584`, `rows=3584`, and `32 <= cols <= 48`, with rollback
`GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ64X48_QCUR_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q5-mmq64x48-qcur-default-20260618-234556/`

Why this was worth testing:
the broad BN48 probe improved Qcur against the older route while regressing
ffn_out. After accepting MMQL64/BK2/BQUAD as the narrow default, the obvious
non-blind follow-up was to isolate BN48 to Qcur and keep ffn_out on BQUAD.

Focused route/correctness:

- p33 CPU-reference passed;
- p33 selected BN48 only for Qcur;
- Kcur stayed on rows2;
- ffn_out stayed on MMQL64/BK2/BQUAD;
- ffn_gate stayed on the accepted VK64 narrow route;
- p512 and p513 CPU-reference gates passed and did not steal the current
  MMQL128/B-quad large/tail policy.

p33 default selector vs rollback to current BQUAD:

| Row | Qcur-only BN48 | Rollback BQUAD | Delta |
| --- | ---: | ---: | ---: |
| Kcur | 74.95 us | 75.69 us | -0.97% |
| Qcur | 317.72 us | 259.21 us | +22.57% |
| ffn_out | 1582.74 us | 1574.78 us | +0.51% |
| ffn_gate | 1771.46 us | 1759.16 us | +0.70% |

Decision:
reject and remove the temporary selector. The current BQUAD route is now much
faster than BN48 on Qcur, so the earlier BN48 win was only relative to the
pre-BQUAD baseline. This closes the simple smaller-BN/Qcur-specific route
split. The next Q5 p33 attempt needs a different schedule contract, not another
BN shrink.

## Accepted p33 MMQL64 BK2 Packed-Q8_1 Route

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_wg256_f32`

Default policy:
gfx1151 Q5_K packed-Q8_1 prompt rows with `rows % 64 == 0`,
`k % 256 == 0`, and `32 <= cols <= 64`.

Rollback:
`GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL64_BK2_PROMPT=1`

Artifacts:

- opt-in focused gate:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-focused-20260618-214600/`
- opt-in model A/B:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-model-ab-sourcebuild-20260618-214835/`
- default regate:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-default-regate-20260618-215108/`
- default model A/B:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-default-model-ab-20260618-215207/`

Schedule:
the route preserves BM64/BN64/WG256/wave64 and packed integer-dot math, but
changes the old MMQ64 direct-A/staged-B loop into an MMQL-style staged A+B
loop with `BK_STEP=2` and four WN16 wave slices. This targets the large-K
ffn_out gap exposed by the BN48 rejection.

Compile evidence:

- built through CMake/Ninja as
  `mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hsaco`;
- wave64, SGPR `62`, VGPR `97`, LDS `10240`, no private segment;
- opcode summary: `256` `v_dot4_i32`, `40` LDS reads, `24` LDS writes,
  `16` stores, `2` barriers.

Focused route/correctness:

- p33, p512, and p513 CPU-reference gates passed;
- p33 default selects BK2 for Qcur and ffn_out;
- p512 stays on current MMQL128;
- p513 stays on current B-quad tail routing;
- rollback returns p33 Qcur and ffn_out to MMQ64.

p33 default vs rollback focused timing:

| Row | Default BK2 | Rollback MMQ64 | Delta |
| --- | ---: | ---: | ---: |
| Kcur | 76.84 us | 76.13 us | +0.94% |
| Qcur | 277.69 us | 331.58 us | -16.25% |
| ffn_out | 1578.48 us | 2520.58 us | -37.38% |
| ffn_gate | 1755.22 us | 1794.70 us | -2.20% |

Same-binary Qwen2.5 Coder 7B Q5_K_M p33/fa1 default-vs-rollback A/B:

| Default BK2 | Rollback MMQ64 | Ratio |
| ---: | ---: | ---: |
| 168.205 tok/s | 153.206 tok/s | 1.098x |

Decision:
accept as the gfx1151 default for Q5_K narrow packed prompt rows. This is a
schedule-led packed-path lift, not an exact RADV cooperative-matrix clone; the
Qwen2.5 Coder Q5 p33 row remains below the captured Vulkan row, so further Q5
work should continue from RADV medium schedule deltas.

## Current Q5 ISA Matrix After Fullstore Diagnostic

Artifact:
`cache/hrxv1/gfx1151/q5-current-isa-score-20260619-021745/`

Generated with:
`sources/llama.cpp/tools/vulkan-oracle/summarize_isa_compare_matrix.py`
over current CMake-built HSACOs and the Qwen2.5 Coder Q5_K_M RADV p33/p512
oracle ISA/stats.

Key matrix rows:

| Candidate | Math | LDS b64 | d16 loads | Stores | First-window loads | Final lgkmcnt | Hot ops in window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RADV p33 medium | 16 WMMA | 48 | 64 | 96 buffer | 48 | 40 | 16 |
| HRX p33 VK64 default | 8 WMMA | 0 | 0 | 16 global | 0 | 2 | 8 |
| HRX p33 VK64 fullstore diagnostic | 8 WMMA | 0 | 0 | 32 global | 0 | 2 | 8 |
| HRX p33 MMQL64 BK2/BQUAD | 256 dot | 0 | 0 | 16 global | 20 | 9 | 60 |
| RADV p512 large | 32 WMMA | 64 | 128 | 192 buffer | 32 | 24 | 14 |
| HRX p512 MMQL128/BQUAD | 512 dot | 0 | 0 | 64 global | 20 | 9 | 60 |

Decision:
do not continue local store-count variants around the existing VK64 narrow
route. The fullstore diagnostic proved the isolated writeback axis is too small
and noisy, and the matrix shows the broader structural miss: HRX VK64 emits
half the visible WMMA sites and no plain `ds_load_b64`/`ds_load_u16_d16`
cooperative-load topology, while the packed MMQL routes are an entirely
different dot-product family. The next Q5 candidate gate should first improve
the first-window schedule toward RADV's p33 signature: at least 16 visible
WMMA sites, nonzero grouped `ds_load_b64`, high pre-WMMA `lgkmcnt`, and a
store topology closer to RADV's 96 buffer stores. If the compiler cannot expose
that from HIP C++ source, move the effort toward a lower-level or inline-asm
fixture before adding another production route.

## p33 Medium Fixture Contract

Artifact:
`cache/hrxv1/gfx1151/q5-p33-fixture-medium-contract-20260619-024558/`

New CMake-built HIP bench modes:

- `hrx-hip-bench-coopmat-store-contract --mode=radv-mixed96`
- `hrx-hip-bench-wmma-issue-window --mode=mediumfrag12`

Purpose:
split the Q5 p33 RADV medium schedule into two lower-level compile contracts
before attempting another production route. This keeps the experiment out of
the kernel catalog while proving which opcode surfaces are reachable from HIP
C++/inline asm.

Results:

| Fixture | WMMA | LDS b64 | d16 loads | LDS stores | Stores | Barriers | First-window loads | Final lgkmcnt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RADV p33 medium | 16 | 48 | 64 | 64 | 96 buffer | 2 | 48 | 40 |
| `radv-mixed96` store fixture | 0 | 0 | 64 | 64 | 96 buffer | 2 | 0 | n/a |
| `mediumfrag12` issue fixture | 16 | 48 | 0 | 0 | 0 | 0 | 48 | 40 |

Decision:
accept these as diagnostic infrastructure only. Together they prove HIP C++ can
emit the p33 RADV store-count/topology surface and the p33 RADV first-WMMA
load window, but only as separate controlled fixtures. The production Q5 route
still lacks the combined cooperative-matrix load/store/lane-ownership lowering
inside one arithmetic kernel. The next useful implementation step is to fuse
these two fixture contracts into a minimal arithmetic/lane-mapping reproducer
before adding another opt-in catalog route.

## p33 Medium Combined Fixture

Artifact:
`cache/hrxv1/gfx1151/q5-p33-combined96-padded-fixture-20260619-025352/`

Mode:
`hrx-hip-bench-wmma-issue-window --mode=mediumfrag12-combined96`

Purpose:
combine the p33 medium issue-window and writeback fixture contracts in one
CMake-built HIP kernel before trying to move the shape into the Q5 catalog.

Static score against RADV `matmul_q5_k_f32_f16acc_aligned_m`:

| Metric | RADV p33 medium | Combined fixture |
| --- | ---: | ---: |
| SGPR | 108 | 14 |
| VGPR | 144 | 142 |
| LDS | 11264 | 11264 |
| WMMA sites | 16 | 16 |
| `ds_load_b64` | 48 | 48 |
| `ds_load_u16_d16` | 64 | 64 |
| `ds_store_b16` | 64 | 64 |
| `buffer_store_b32` | 96 | 96 |
| Barriers | 2 | 3 |
| First-window `ds_load_b64` | 48 | 48 |
| Final pre-WMMA `lgkmcnt` | 40 | 40 |
| Hot ops in scored window | 16 | 16 |

Smoke:
the mode ran through `hrx-hip-bench-wmma-issue-window` with finite output
(`nan=0`).

Decision:
accept as the current lower-level p33 direct-WMMA reproducer. It proves that
HIP C++ plus inline asm can carry the RADV medium opcode/resource contract in
one kernel, modulo one extra barrier and unlike SGPR setup. This is still not a
production route because it does not consume real Q5/Q8 data or prove
accumulator lane mapping against `MUL_MAT`. The next catalog-facing step should
port this exact load/writeback skeleton into a Q5 p33 provider with focused
CPU-reference and route traces; if correctness fails, debug lane ownership from
this fixture before changing schedule axes.

## p33 Combined96 Catalog Probe

Artifact:
`cache/hrxv1/gfx1151/q5-p33-combined96-catalog-20260619-030619/`

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_combined96_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_COMBINED96_F16ACC_WG256_PROMPT=1`

Purpose:
move the combined fixture skeleton into the real Q5_K direct-F32 ABI as a
p33-only opt-in provider, then reject or keep it based on static schedule and
focused CPU-reference evidence.

Static score against RADV `matmul_q5_k_f32_f16acc_aligned_m`:

| Metric | RADV p33 medium | Combined96 catalog |
| --- | ---: | ---: |
| SGPR | 108 | 55 |
| VGPR | 144 | 142 |
| LDS | 11264 | 11264 |
| Private/scratch | 0 | 0 |
| WMMA sites | 16 | 16 |
| `ds_load_b64` | 48 | 48 |
| `ds_load_u16_d16` | 64 | 64 |
| `ds_store_b16` | 64 | 66 |
| `buffer_store_b32` | 96 | 96 |
| Barriers | 2 | 4 |
| First-window `ds_load_b64` | 48 | 20 |
| Final pre-WMMA `lgkmcnt` | 40 | 0 |
| Hot ops in scored window | 16 | 16 |

Focused CPU-reference:
`test-backend-ops test -b HRX0 -o MUL_MAT --test-file q5_prompt_p33.txt`
selected the combined96 route for all four Qwen2.5 Coder Q5_K_M p33 rows and
failed all four with NaNs.

Decision:
reject before timing/model tests. The real catalog ABI can now hit the broad
p33 RADV opcode-count surface, but it did not preserve the fixture issue
window and it does not have valid lane/output ownership. The next useful step
is a lower-level dependency/lane-mapping reproducer, not another local
selector or timing sweep around this provider.

## p33 Combined96 Real-Q5 Reproducer

Artifact:
`cache/hrxv1/gfx1151/q5-combined96-repro-20260619-052854/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now includes a `combined96` synthetic
Q5_K/F32 RHS variant alongside the existing full64, active-group, and batched4
controls. It builds through the normal CMake/Ninja HIP bench target.

Purpose:
reproduce the rejected combined96 catalog behavior with the exact Q5 dequant,
stride44 LDS staging, b64 fragment loads, RADV-like halfword staged writeback,
and CPU reference comparison, but outside `test-backend-ops` and without
changing runtime route selection.

Static facts for `q5_combined96_repro_kernel`:

- wave64, SGPR `55`, VGPR `142`, LDS `11264`, no spills;
- `16` `v_wmma_f16_16x16x16_f16`;
- `48` `ds_load_b64`;
- `64` `ds_load_u16_d16`;
- `66` `ds_store_b16`;
- `96` `buffer_store_b32`;
- `3` barriers.

Runtime result:

- `combined96` reproduces NaNs and large finite mismatches even on the small
  profile: p33/k256 `nan=42`, p33/k512 `nan=28`, p33/k3584 `nan=26`, and
  p64/k3584 `nan=28 inf=1`;
- stress p33 also reproduces `nan=32 inf=30` at k256 and `nan=24 inf=30` at
  k3584;
- low-live controls remain finite for the same small p33/k3584 case:
  `active4 nan=0 inf=0 max_abs=0.0996094`, and `batched4 nan=0 inf=0
  max_abs=0.0996094`.

Decision:
keep combined96 rejected and keep this as diagnostic infrastructure. The
failure is now reproduced with the real Q5 dequant/shared-layout context while
preserving the broad RADV p33 opcode/resource surface. The next route-facing
work should not change tile shape or selector policy; it should isolate the
accumulator lane/value dependency contract that makes the low-live controls
finite but makes the combined RADV-like writeback topology numerically invalid.

## p33 Combined96 Raw8 and Wait0 Dependency Probe

Artifact:
`cache/hrxv1/gfx1151/q5-combined96-wait0-repro-20260619-054018/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now includes:

- `combined96-raw8`: same explicit combined96 fragment load and WMMA issue
  order, but stores only the first eight raw accumulator groups;
- `combined96-raw8-wait0`: same raw8 path with `lgkmcnt(0)` instead of the
  RADV-like `lgkmcnt(40)` before WMMA issue;
- `combined96-wait0`: the full staged combined96 path with `lgkmcnt(0)`.

Purpose:
separate staged/aliased second-half writeback from first-eight accumulator
value corruption, and test whether the RADV-like relaxed LDS wait is the HIP
correctness hazard.

Runtime result on the small profile:

| Variant | rows | cols | k | NaNs | first-eight NaNs | Infs | Sentinel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined96 | 64 | 33 | 3584 | 26 | 26 | 0 | 0 |
| combined96-raw8 | 64 | 33 | 3584 | 104 | 104 | 1 | 0 |
| combined96-raw8-wait0 | 64 | 33 | 3584 | 144 | 144 | 4 | 0 |
| combined96-wait0 | 64 | 33 | 3584 | 30 | 30 | 0 | 0 |
| active8 control | 64 | 33 | 3584 | 0 | 0 | 0 | 0 |
| batched4 control | 64 | 33 | 3584 | 0 | 0 | 0 | 0 |

Static selected-symbol facts:

| Variant | WMMA | `ds_load_b64` | `ds_load_u16_d16` | `ds_store_b16` | `buffer_store_b32` | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined96-raw8 | 16 | 48 | 0 | 2 | 32 | 2 | 23 |
| combined96-raw8-wait0 | 16 | 48 | 0 | 2 | 32 | 2 | 23 |
| combined96 | 16 | 48 | 64 | 66 | 96 | 3 | 25 |
| combined96-wait0 | 16 | 48 | 64 | 66 | 96 | 3 | 25 |
| active8 control | 16 | 64 | 0 | 2 | 32 | 2 | 24 |

Decision:
the failure is in the first eight accumulator values, not in the staged or
aliased second-half writeback. Strengthening the pre-WMMA LDS wait to
`lgkmcnt(0)` does not fix the same 48-load path, so a missing wait is not the
root cause. The remaining live bracket is the fragment-load/dependency shape:
the finite `active8` control emits the same WMMA count and raw store count but
uses `64` B64 fragment loads, while both explicit combined paths use the
RADV-like `48` B64 fragment-load surface and fail. The next diagnostic should
test whether extra B-fragment materialization/padding or a different inline
asm constraint is required for HIP C++ to preserve the cooperative-matrix lane
contract; do not promote another route on this family until that is resolved.

## p33 Combined96 B-Fragment Padding Probe

Artifact:
`cache/hrxv1/gfx1151/q5-combined96-bpad-repro-20260619-054349/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now includes:

- `combined96-raw8-bpad`: same raw first-eight combined96 path, plus forced
  materialization of the otherwise-unused B fragments for column tiles 2 and 3;
- `combined96-bpad`: same full staged combined96 path, plus the same forced B
  fragment materialization.

Purpose:
test the remaining difference between failing explicit combined96 and the
finite `active8` control: `active8` emits `64` B64 LDS fragment loads, while
combined96 previously emitted `48`.

Static facts:

| Variant | SGPR | VGPR | WMMA | `ds_load_b64` | `ds_load_u16_d16` | `ds_store_b16` | `buffer_store_b32` | Barriers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined96-raw8 | 55 | 142 | 16 | 48 | 0 | 2 | 32 | 2 |
| combined96-raw8-bpad | 65 | 178 | 16 | 64 | 0 | 2 | 32 | 2 |
| combined96 | 55 | 142 | 16 | 48 | 64 | 66 | 96 | 3 |
| combined96-bpad | 65 | 178 | 16 | 64 | 64 | 66 | 96 | 3 |
| active8 control | 64 | 159 | 16 | 64 | 0 | 2 | 32 | 2 |

Runtime result on the small p33/k3584 row:

| Variant | NaNs | first-eight NaNs | Infs | Sentinel | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| combined96-raw8 | 112 | 112 | 0 | 0 | fail |
| combined96-raw8-bpad | 136 | 136 | 0 | 0 | fail |
| combined96 | 32 | 32 | 0 | 0 | fail |
| combined96-bpad | 38 | 38 | 0 | 0 | fail |
| active8 control | 0 | 0 | 0 | 0 | finite but inaccurate |

Decision:
reject B-fragment materialization as the missing correctness condition. The
padded variants match `active8`'s `64` B64 load count but still produce
first-eight NaNs. The remaining difference is now more specific than load
count or wait placement: the explicit combined96 value/operand dependency
spelling itself differs from the finite array-loop `active8` topology. The
next diagnostic should generate an array-loop version with combined96's
reduced B-column use, or an explicit combined96 version that exactly follows
the active8 loop body order, before attempting another catalog route.

## p33 Array-Loop Reduced-B Probe

Artifact:
`cache/hrxv1/gfx1151/q5-array8-b2-repro-20260619-054730/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now includes `array8-b2`, a diagnostic
that preserves the finite `active8` array-loop WMMA body and first-eight raw
store contract, but loads only B column tiles 0 and 1 like combined96.

Purpose:
separate the active8 loop topology from active8's full four-column B fragment
materialization. This answers whether the reduced-B fragment set alone is
enough to trigger the first-eight NaNs.

Static facts:

| Variant | SGPR | VGPR | LDS | WMMA | `ds_load_b64` | `buffer_store_b32` | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| array8-b2 | 52 | 142 | 11264 | 16 | 48 | 32 | 2 | 24 |
| combined96-raw8 | 55 | 142 | 11264 | 16 | 48 | 32 | 2 | 23 |
| active8 control | 64 | 159 | 11264 | 16 | 64 | 32 | 2 | 24 |

Runtime result on the small profile:

| Variant | rows | cols | k | NaNs | first-eight NaNs | Infs | Sentinel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| array8-b2 | 64 | 33 | 256 | 160 | 160 | 2 | 0 |
| array8-b2 | 64 | 33 | 3584 | 136 | 136 | 1 | 0 |
| array8-b2 | 64 | 64 | 3584 | 112 | 112 | 0 | 0 |
| active8 control | 64 | 33 | 3584 | 0 | 0 | 0 | 0 |
| batched4 control | 64 | 33 | 3584 | 0 | 0 | 0 | 0 |

Decision:
reject reduced-B array-loop as a correctness path. The first-eight NaNs appear
even when the WMMA body uses active8's array-loop topology and `lgkmcnt(0)`.
The stable condition now appears to require both the active8-style array-loop
topology and full four-column B fragment materialization. Because the B-padding
probe showed full B materialization does not rescue explicit combined96, the
next diagnostic should keep all four B fragments live in the active8 loop body
while varying only the compute loop shape, such as a no-`if` `col_sub < 2`
active8 variant, before attempting another route.

## p33 Array-Loop Full-B No-If Probe

Artifact:
`cache/hrxv1/gfx1151/q5-array8-fullb-noif-repro-20260619-055054/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now includes
`array8-fullb-noif`, a diagnostic that:

- loads all four B column fragments per K tile, like finite `active8`;
- forces the unused B fragments to stay materialized;
- computes only first-eight output groups with a branch-free `col_sub < 2`
  loop;
- stores only the first-eight raw output groups.

Purpose:
test whether active8's `if (tile < active_groups)` branch is required for the
finite behavior, or whether the full-B materialization plus simple branch-free
first-eight compute is enough.

Static facts:

| Variant | SGPR | VGPR | LDS | WMMA | `ds_load_b64` | `buffer_store_b32` | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| array8-fullb-noif | 64 | 170 | 11264 | 16 | 64 | 32 | 2 | 24 |
| array8-b2 | 52 | 142 | 11264 | 16 | 48 | 32 | 2 | 24 |
| active8 control | 64 | 159 | 11264 | 16 | 64 | 32 | 2 | 24 |

Runtime result on the small profile:

| Variant | rows | cols | k | NaNs | Infs | max_abs | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| array8-fullb-noif | 64 | 33 | 256 | 0 | 0 | 0.0107422 | first 32 cols only |
| array8-fullb-noif | 64 | 33 | 3584 | 0 | 0 | 0.0996094 | first 32 cols only |
| array8-fullb-noif | 64 | 64 | 3584 | 0 | 0 | 0.0996094 | first 32 cols only |
| array8-b2 | 64 | 33 | 3584 | 104 | 1 | 61117.7 | fail |
| active8 control | 64 | 33 | 3584 | 0 | 0 | 9.81689 | finite, less accurate |
| batched4 control | 64 | 33 | 3584 | 0 | 0 | 0.0996094 | full p33 coverage |

Decision:
accept as the current positive first-eight dependency contract. The active8
branch is not required; full four-column B fragment materialization plus the
array-loop fragment topology is sufficient to avoid NaNs on the first 32
columns, and the branch-free form improves the active8 finite error scale.
This is still not a route because p33 needs the tail column at `col=32`
(`groups 8..11`). The next route-facing diagnostic should compute the first
eight groups with `array8-fullb-noif` and handle groups `8..11` in a separate
low-live phase, without restaging all K four times like `batched4`.

## p33 Array-Loop Tail Phase Probes

Artifacts:

- `cache/hrxv1/gfx1151/q5-array8-tail4-2dispatch-repro-20260619-055405/`
- `cache/hrxv1/gfx1151/q5-array8-tail8-2dispatch-repro-20260619-055502/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now includes:

- `array8-tail4-2dispatch`: dispatches the accepted first-eight phase, then a
  low-live `groups 8..11` phase for the p33 tail column;
- `array8-tail8-2dispatch`: dispatches the accepted first-eight phase, then a
  paired `groups 8..15` phase to test whether the tail also needs a two-column
  group pair.

Purpose:
turn the positive first-eight contract into full p33 coverage without
batched4's four full K-restaging passes.

Static facts:

| Variant | SGPR | VGPR | WMMA | `ds_load_b64` | `buffer_store_b32` | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first-eight phase `<0,0,2,8>` | 64 | 170 | 16 | 64 | 32 | 2 | 24 |
| tail4 phase `<8,2,1,4>` | 55 | 146 | 8 | 64 | 16 | 2 | 24 |
| tail8 phase `<8,2,2,8>` | 64 | 170 | 16 | 64 | 32 | 2 | 24 |
| batched4 control | 51 | 186 | 32 | 256 | 64 | 12 | 108 |

Runtime result on the small profile:

| Variant | rows | cols | k | NaNs | Infs | max_abs | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| array8-tail4-2dispatch | 64 | 33 | 3584 | 0 | 0 | 5649.78 | reject |
| array8-tail8-2dispatch | 64 | 33 | 3584 | 0 | 0 | 3642.56 | reject |
| array8-tail8-2dispatch | 64 | 64 | 3584 | 24 | 0 | 43105.8 | reject |
| batched4 control | 64 | 33 | 3584 | 0 | 0 | 0.0996094 | pass |

Decision:
reject the simple two-dispatch tail repair. It removes p33 NaNs but tail-column
values are wildly wrong, and the paired tail8 phase still produces NaNs on
p64. The static schedule is low-live and spill-free, so the failure is not
resource pressure. The only full-p33 accurate control remains batched4, which
suggests groups `8..11` depend on the exact batched4 source/codegen context or
on a lane/value contract still not captured by the phase kernels. Do not
promote the two-dispatch phase path; the next useful route-facing step is to
lift the exact batched4 group-base loop into a p33-only catalog probe and
measure whether a two-phase or four-phase version can beat the current packed
Q8_1 route after correctness is preserved.

## p33 Pruned Batched4 Probe

Artifact:
`cache/hrxv1/gfx1151/q5-batched4-p33-repro-20260619-055914/`

Source:
`hrx-hip-bench-q5-wmma-full64-repro` now templates the correct batched4
diagnostic on `group_base_end`:

- `batched4`: original four phases, groups `0..15`;
- `batched4-p33`: p33-pruned three phases, groups `0..11`, skipping the
  unused `group_base=12` phase for `cols=33`.

Purpose:
preserve the exact batched4 group-base source/codegen contract that is known
to be p33-correct, while removing only work that cannot contribute to a p33
output tile.

Static facts:

| Variant | SGPR | VGPR | WMMA | `ds_load_b64` | `ds_store_b16` | `buffer_store_b32` | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batched4-p33 | 51 | 182 | 24 | 192 | 6 | 48 | 9 | 80 |
| batched4 | 51 | 186 | 32 | 256 | 8 | 64 | 12 | 108 |

Runtime result on the small profile:

| Variant | rows | cols | k | NaNs | Infs | max_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batched4-p33 | 64 | 33 | 256 | 0 | 0 | 0.0107422 |
| batched4-p33 | 64 | 33 | 3584 | 0 | 0 | 0.0996094 |
| batched4 | 64 | 33 | 3584 | 0 | 0 | 0.0996094 |

Decision:
accept as the current correctness-preserving p33 direct-WMMA baseline, but not
as a production route. It saves exactly one batched4 phase and preserves the
tail-column value contract, unlike the two-dispatch phase probes. However it
still performs three full K-restaging passes and emits `192` B64 LDS loads for
one p33 output tile, so it is unlikely to beat the current packed-Q8_1 route.
The next production candidate should not promote this directly; it should use
`batched4-p33` as the correctness oracle while trying to fuse the first two
phases or otherwise reuse staged A/B without breaking the batched4 lane/value
contract.

## p33 Pruned Batched4 Catalog Rejection

Artifact:
`cache/hrxv1/gfx1151/q5-p33-batched4-p33-catalog-20260619-060808/`

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_batched4_p33_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_BATCHED4_P33_F16ACC_WG256_PROMPT=1`

Purpose:
test whether the p33-pruned batched4 source contract survives the real catalog
ABI and focused model-derived Q5 rows.

Focused result:

| Row | Shape | Result |
| --- | --- | --- |
| Kcur | `k=3584 rows=512 cols=33` | pass |
| Qcur | `k=3584 rows=3584 cols=33` | pass |
| ffn_out | `k=18944 rows=3584 cols=33` | fail, `ERR=0.043113960` |
| ffn_gate | `k=3584 rows=18944 cols=33` | pass |

Static selected-symbol facts:

| SGPR | VGPR | LDS | WMMA | `ds_load_b64` | `buffer_store_b32` | Barriers | Waits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 51 | 182 | 11264 | 24 | 192 | 48 | 9 | 80 |

Decision:
reject before timing/model tests. The standalone repro only validated
`batched4-p33` at `k=256` and `k=3584`; the real focused wide-K `ffn_out`
row at `k=18944` exposes a finite correctness failure. Simple p33 phase
pruning is therefore not a safe catalog route. The full four-phase batched4
route remains the direct-WMMA p33 correctness oracle, while production work
should continue from the accepted packed-Q8_1 medium path or from a
correctness-preserving full-contract wide-K direct-WMMA variant.

## p33 MMQL64 BK2 BQUAD BHALF Default

Artifact:
`cache/hrxv1/gfx1151/q5-bquad-bhalf-default-postpromotion-20260619-062244/`

Route:
`hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_bquad_bhalf_wg256_f32`

Policy:
default on `gfx1151` for Q5_K packed-Q8_1/x4 prompt rows with
`32 <= cols <= 64`, `rows % 64 == 0`, `k % 256 == 0`, and contiguous tensors.
Rollback:
`GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL64_BK2_BQUAD_BHALF_PROMPT=1`.

Purpose:
return to the accepted packed-Q8_1 medium path after the direct-WMMA p33 route
failed to become a production route, and bracket one remaining packed-path
axis: B-cache footprint and scale-conversion pressure. The schedule preserves
MMQL64 BM64/BN64/WG256/wave64/BK_STEP=2/BQUAD issue order, but stores Q8_1
RHS `d/s` values in LDS as half payloads instead of widened floats.

Static selected-symbol comparison:

| Variant | SGPR | VGPR | LDS | `v_dot4_i32_iu8` | Global loads | Global stores | Barriers | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BQUAD | 62 | 124 | 10240 | 256 | 39 | 16 | 2 | 119 |
| BHALF | 61 | 122 | 9728 | 256 | 39 | 16 | 2 | 117 |

Focused post-promotion result:

| Row | Default BHALF r2 | Rollback BQUAD r2 | Ratio |
| --- | ---: | ---: | ---: |
| Kcur | 75.400940 us | 73.245789 us | 1.029424 |
| Qcur | 257.528592 us | 259.448791 us | 0.992599 |
| ffn_out | 1541.924964 us | 1582.095238 us | 0.974609 |
| ffn_gate | 1160.300144 us | 1171.256133 us | 0.990646 |
| sum | 3035.154640 us | 3086.045951 us | 0.983509 |

Route evidence:
p33 selects BHALF for Qcur, ffn_out, and ffn_gate; Kcur remains on
`hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32`. p512 remains on MMQL128 and p513
remains on MMQL128 BQUAD, so the new default does not steal production-width
or tail rows.

Model smoke:
Qwen2.5 Coder 7B Q5_K_M p33 `llama-bench -r 5` improved from rollback
`209.982618 tok/s` to default `211.406107 tok/s` with `backends=HRX`.

Decision:
promote as a small, evidence-backed p33 default with rollback. This is not the
Vulkan parity route: the winning Vulkan p33 medium oracle still has a
different cooperative-matrix-style dataflow (`LDS=11264`, `VGPR=144`, `16`
WMMA, `48` `ds_load_b64`, `64` `ds_load_u16_d16`, `96` stores, `2` barriers).
BHALF is only a local packed-route cache/layout improvement while the direct
RADV clone remains unsolved.

## p33 VK64 Wave4row Batch-Wait Rejection

Artifact:
`cache/hrxv1/gfx1151/q5-p33-wave4row-batchwait-20260619-064007/`

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_wave4row_batchwait_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_WAVE4ROW_BATCHWAIT_F16ACC_WG256_PROMPT=1`

Purpose:
preserve the correctness-clean `wave4row` ownership contract while testing one
remaining RADV-like load-window axis. The prior `wave4row` route used all four
wave64s, one 16-row stripe per wave, four live accumulators, and one A/B LDS
staging pass per K tile, but waited after every `ds_load_b64`. This probe
issues the four B64 LDS reads for one fragment, waits once, then forms the
WMMA operand. It tests whether reducing wait granularity moves the direct-F32
WMMA route toward the RADV p33 medium oracle.

Correctness and route evidence:
p33 focused CPU-reference passed all four Qwen2.5 Coder Q5_K_M rows, and route
traces selected the batch-wait provider for Kcur, Qcur, ffn_out, and ffn_gate.
p512 and p513 focused non-steal gates passed; traces show p512 remained on
MMQL128 and p513 remained on MMQL128 BQUAD.

Selected-symbol static facts:

| SGPR | VGPR | LDS | WMMA | `ds_load_b64` | `buffer_store_b32` | Barriers | Waits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 102 | 11264 | 8 | 40 | 16 | 2 | 36 |

Focused timing:

| Row | Default BHALF | Batch-wait | Ratio |
| --- | ---: | ---: | ---: |
| Kcur | 71.558960 us | 359.587036 us | 5.025046 |
| Qcur | 258.398009 us | 462.957041 us | 1.791643 |
| ffn_out | 1552.670996 us | 3140.975469 us | 2.022950 |
| ffn_gate | 1155.688312 us | 2261.818182 us | 1.957118 |
| sum | 3038.316277 us | 6225.337728 us | 2.048943 |

Decision:
reject before model tests. The load-window change is correct and improves the
prior `wave4row` wait count (`51 -> 36`) and timing modestly, but the route is
still about 2x slower than the current BHALF packed default on the focused p33
sum. This closes the simple batch-wait direct-WMMA pivot. Further Q5 p33 work
should either find a lower-level way to reproduce RADV's cooperative
store/writeback surface (`ds_load_u16_d16`/halfword stage/96 stores) or return
to packed-Q8_1 medium schedule work; another direct-F32 WMMA ownership reshuffle
is unlikely to reach parity.

## Accepted p512/p513 Motif192 Small-Projection Default

Route:
`hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_motif192_bufferstore_f16acc_wg256_f32`

Policy:
default on `gfx1151` for Q5_K small projection prompt rows with
`k % 256 == 0`, `k <= 3584`, `128 <= rows <= 3584`, and `cols >= 512`.
Rollback:
`GGML_HRX_DISABLE_Q5_K_WMMA16_VK128_MOTIF192_SMALLPROJ_PROMPT=1`.

Purpose:
promote the closest current RADV-like Q5 large-route HSACO only for the
Kcur/Vcur-style rows where it consistently wins, while leaving p33 and large
Q/FFN rows on the accepted packed-Q8_1 routes. This is a selector-level
production lift around an existing direct-WMMA route; it is not a full RADV
clone because the motif192 static contract still differs from RADV in the
first-WMMA/load window and store clustering.

Static evidence:
`cache/hrxv1/gfx1151/q5-current-contract-refresh-08451f1de-20260620-165448/`
shows motif192 is the closest current RADV-like Q5 p512 HSACO: wave64, no
spills, LDS `22528`, `32` WMMA, and `192` VMEM stores. The accepted packed
p512 default remains a different `v_dot` family, so this promotion is
deliberately limited to the small-projection row where the direct-WMMA route
has measured benefit.

Focused committed-source opt-in evidence:
`cache/hrxv1/gfx1151/q5-motif192-smallproj-focused-committed-20260620-170715/`
and
`cache/hrxv1/gfx1151/q5-motif192-smallproj-model-ab-committed-20260620-170557/`.
At commit `18e3bceb4`, p33 selected no motif192 routes. p512/p513 selected
motif192 only for `k=3584, rows=512, cols>=512`; Qcur and FFN rows stayed on
MMQL128 BHALF/BQUAD. Same-binary Qwen2.5 Coder 7B Q5_K_M model A/B improved
p512 steady by `1.0198x` and p513 steady by `1.0114x`.

Post-edit default-vs-rollback evidence:
`cache/hrxv1/gfx1151/q5-motif192-smallproj-default-regate-20260620-171010/`,
`cache/hrxv1/gfx1151/q5-motif192-smallproj-default-model-ab-20260620-171203/`,
and repeat
`cache/hrxv1/gfx1151/q5-motif192-smallproj-default-model-ab-repeat-20260620-171312/`.
Final source commit:
`85edc5327 hrx: default q5 motif192 small projection route`.

Focused CPU-reference passed p33, p512, and p513 for both default and rollback.
Route traces prove:

- p33 is unchanged under default and rollback;
- p512 moves only `k=3584, rows=512, cols=512` from `rows2_cols8` to motif192;
- p513 moves only `k=3584, rows=512, cols=513` from `rows2_cols8` to motif192;
- Qcur and FFN rows remain on the accepted MMQL128 packed routes.

Focused default-vs-rollback timing:

| Case | Default sum | Rollback sum | Default/Rollback |
| --- | ---: | ---: | ---: |
| p33 | `3016.907 us` | `3030.743 us` | `0.995x` |
| p512 | `16118.840 us` | `16429.413 us` | `0.981x` |
| p513 | `19720.294 us` | `20362.338 us` | `0.968x` |

Model guardrails:

| Artifact | Case | Default/Rollback steady |
| --- | --- | ---: |
| first default A/B, r5 | p33 | `0.997x` |
| first default A/B, r5 | p512 | `0.999x` |
| first default A/B, r5 | p513 | `1.018x` |
| rollback-first repeat, r7 | p512 | `1.007x` |
| rollback-first repeat, r7 | p513 | `1.009x` |

Decision:
accept the narrow default with rollback. The focused evidence is clean and the
repeat model guardrail is positive for the rows that actually route to
motif192. The p512 model effect is small and noisy, so future broad Q5 work
should not treat this as closing the Q5 large-route boulder. Continue using
RADV static evidence to pursue the remaining direct-WMMA load/store contract,
or move to larger basket boulders if Q5 is no longer the worst row.
