# HRX v1 gfx1151 Experiments Log

This log records evidence for the HRX v1 HIP C++ `gfx1151` tuning effort. Keep
entries append-only unless correcting a factual error. Supersede stale evidence
with a new entry instead of rewriting history.

Required fields for every experiment:

- date;
- source commit and dirty-state summary;
- build directory and relevant CMake flags;
- model and shape;
- route or kernel candidate;
- baseline command;
- variant command;
- route trace path;
- profile or timing artifact path;
- correctness result;
- timing result;
- decision;
- notes.

Partial-basket results are useful for boulder ranking, route validation, and
rejecting bad schedules. They are not sufficient for broad route promotion until
the full production basket is available and the odd-size/tail gates pass.

## 2026-06-21 - F16 WMMA LLVM MC compact form screen

- source:
  `sources/llama.cpp` at `4e34cbae6-dirty`, adding
  `tools/vulkan-oracle/run_wmma_mc_compact_probe.py`.
- build:
  no HIP kernel build; this probes the workspace ROCm assembler
  `/srv/vm-shared/projects/llamacpp-devws/rocm/llvm/bin/llvm-mc`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-mc-compact-current-4e34cbae6-20260621-013523/`.
- model and shape:
  lower-level primitive screen for RADV-style compact f16 WMMA accumulator
  ownership used by Q8_0/Q6_K direct-WMMA oracle candidates.
- route:
  not a production route; assembler primitive screen only.
- baseline command:
  `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1151 --show-encoding width8.s`.
- variant command:
  `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1151 --show-encoding width4.s`.
- route trace path:
  not applicable.
- profile or timing artifact path:
  artifact `summary.json`, `summary.md`, `width8.stdout`, and
  `width4.stderr`.
- correctness result:
  no runtime correctness run; this is before object generation.
- static result:
  width8 form
  `v_wmma_f16_16x16x16_f16 v[0:7], v[0:7], v[0:7], v[0:7]`
  assembled and emitted encoding
  `[0x00,0x40,0x42,0xcc,0x00,0x01,0x02,0x1c]`. Width4 form
  `v_wmma_f16_16x16x16_f16 v[0:3], v[0:3], v[0:3], v[0:3]`
  failed with `operands are not valid for this GPU or mode`.
- timing result:
  not applicable.
- decision:
  reject straightforward LLVM MC as a path to emit RADV compact width4 f16
  WMMA dst/C operands on gfx1151.
- notes:
  the compact accumulator blocker is not only a HIP C++ inline-asm constraint
  problem. A true exact RADV f16-WMMA clone likely needs a compiler/target
  description change, a non-LLVM code-object path, or must be replaced with a
  measured non-compact/packed-Q8 dataflow.

## 2026-06-21 - Q8_0 current-head compact accumulator screen

- source:
  `sources/llama.cpp` at `86d94bd97-dirty`, adding
  `tools/vulkan-oracle/run_wmma_compact_screen.py` and refreshing the current
  Q8 split59 static comparison.
- build:
  CMake/Ninja targets
  `ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_baseoff_split59_bufferstore_wg256.hsaco`
  and `hrx-hip-bench-wmma-f16-lane-map`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8-compact-screen-current-86d94bd97-20260621-012828/`.
- model and shape:
  static comparison against Llama 3.1 8B Q8_0 p512 Vulkan large oracle
  `matmul_q8_0_f32_f16acc_aligned_l`.
- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_baseoff_split59_bufferstore_f16acc_wg256_f32`.
- baseline command:
  RADV ISA from
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/radv/isa/matmul_q8_0_f32_f16acc_aligned_l__main__72d309e22f889977.amdgcn.txt`.
- variant command:
  `sources/llama.cpp/tools/vulkan-oracle/run_wmma_compact_screen.py --radv-isa ... --hip-hsaco build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_baseoff_split59_bufferstore_wg256.hsaco --out-dir ...`.
- route trace path:
  not applicable; static screen only, using the already rejected opt-in route.
- profile or timing artifact path:
  artifact `wmma-ownership.md`, `wmma-ownership.json`, `summary.json`, and
  `compact-check.stderr.log`.
- correctness result:
  no runtime correctness run in this screen; prior split59 focused CPU-reference
  p33/p512/p513 already passed in
  `cache/hrxv1/gfx1151/q8-baseoff-split59-realdata-67045a7c93d0-dirty-20260621-002755/`.
- static result:
  RADV and HIP both show `32` f16 WMMAs, `64 ds_load_b64`, and
  `192 buffer_store_b32`, but the compact accumulator screen fails for HIP:
  RADV has `dst`/`C` width4 counts `32/32`, while HIP has `dst`/`C` width8
  counts `32/32` and zero width4 accumulator operands.
- timing result:
  no new timing; prior split59 focused timing was rejected at `1.51x` slower
  on p512 and `1.413x` slower on p513 versus the packed-Q8_1 default.
- decision:
  keep Q8 split59 rejected and require future Q8 large f16-WMMA clone
  candidates to pass `run_wmma_compact_screen.py` before focused timing, unless
  they are explicitly framed as measured width8 deviations.
- notes:
  this turns the RADV compact dst/C operand mismatch into a reusable current
  build screen, so the next Q8 parity attempt should be a different primitive
  or lowering path rather than another issue-window-only route variant.

## 2026-06-21 - Q6_K MUL_MAT_ID MMQ64x32 p33 rejection

- source:
  `sources/llama.cpp` at `cd0f064bd-dirty`, adding an opt-in
  packed-Q8_1/x4 `BN=32` Q6 `MUL_MAT_ID` route.
- build:
  CMake/Ninja target `test-backend-ops` and generated HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_id_q6_k_q8_1_x4_mmq.hsaco`,
  built in `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-mmq64x32-focused-cd0f064bd-dirty-20260621-012104/`.
- model and shape:
  focused Qwen3 Q6_K `MUL_MAT_ID` p33 rows from
  `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt`.
- route:
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x32_wg64_f32`.
- baseline command:
  default focused perf with
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1`.
- variant command:
  same focused perf with
  `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ32_PROMPT=1` and
  `GGML_HRX_EXPECT_MUL_MAT_ID_Q6_PROVIDER=hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x32_wg64_f32`.
- route trace path:
  artifact `test.stderr.log`, `default-perf.stderr.log`, and
  `mmq32-perf.stderr.log`.
- profile or timing artifact path:
  artifact `compare.md`, `compare.json`, and `hsaco-summary.md`.
- correctness result:
  focused CPU-reference passed `ffn_moe_gate-0` and `ffn_moe_down-0`; route
  traces proved `route_tile_n=32` with workgroups `[12,2,128]` and
  `[32,2,128]`.
- static result:
  wave64, SGPR `68`, VGPR `126`, LDS `3840`, no scratch/spills, `256`
  `v_dot4_i32_iu8`, `62` LDS-read ops, `26` LDS-write ops, `32` global
  stores, and two barriers.
- timing result:
  reject. Same-runner p33 total regressed from `580.217 us` default MMQ16 to
  `606.418 us` MMQ32. `ffn_moe_gate-0` regressed `308.434 -> 327.414 us`;
  `ffn_moe_down-0` regressed `271.782 -> 279.005 us`.
- decision:
  reject before p512/p513 or model A/B.
- notes:
  the intermediate output-ownership point closes the bracket between accepted
  MMQ16 and rejected MMQ64 for the current packed-Q8_1/x4 schedule. Reducing
  p33 Y groups from three to two does not recover Vulkan parity and does not
  even preserve the current p33 floor. Future Q6 ID work should not continue
  route-tile-only packed probes; it needs either a real compact f16-WMMA
  lowering path or a different packed dataflow/packing-cost hypothesis.

## 2026-06-21 - Q6_K MUL_MAT_ID p33 floor refresh

- source:
  `sources/llama.cpp` at `cd0f064bd`, clean after committing the Q6 dense
  p33-stage12 rejection probe.
- build:
  CMake/Ninja target `test-backend-ops`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`; the build regenerated the HRX catalog and
  all touched gfx1151 HSACOs through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-current-floor-p33-cd0f064bd-20260621-011706/`.
- model and shape:
  focused Qwen3 Q6_K `MUL_MAT_ID` p33 rows from
  `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt`.
- route:
  default `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- baseline command:
  previous current-head floor artifact
  `cache/hrxv1/gfx1151/q6-id-current-head-p33-focused-c012bdbf0-20260620-222329/`.
- variant command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops perf -b HRX0 -o MUL_MAT_ID --test-file cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt --output csv`.
- route trace path:
  artifact `perf.stderr.log`.
- profile or timing artifact path:
  artifact `perf.stdout.csv`, `compare-vs-c012bdbf0.md`, and
  `compare-vs-c012bdbf0.json`.
- correctness result:
  perf rows reported `passed=1` for `ffn_moe_gate-0` and `ffn_moe_down-0`.
- route result:
  both rows stayed on
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32` with
  `route_tile_n=16` and `wg_count=[12,3,128]`.
- timing result:
  refreshed total `555.855 us` versus prior current-head total `584.362 us`
  (`1.051x` prior/current). Row timings: `ffn_moe_gate-0` `291.312 us`,
  `ffn_moe_down-0` `264.542 us`.
- decision:
  keep as the current accepted Q6 ID p33 floor.
- notes:
  this confirms the next Q6 ID experiment should not be another selector or
  Y-grid-only change. The prior `MMQ64x64` Y-grid reduction was rejected, and
  the compact f16-WMMA accumulator screen shows the exact RADV clone is blocked
  by width-4 dst/C operand availability in the current HIP C++/LLVM path. The
  next useful route-facing candidate must either pass that compact accumulator
  static screen via a genuinely different lowering path or declare itself as a
  packed-Q8_1/x4 deviation with a concrete output-ownership/packing-cost
  hypothesis tested against this floor on p33, p512, and p513.

## 2026-06-21 - Q6_K VK64 depwait p33-stage12 rejection

- source:
  `sources/llama.cpp` at `c9dc2a616-dirty`, adding a standalone
  p33-stage12 depwait diagnostic.
- build:
  CMake/Ninja targets
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33stage12-repro` and
  generated gfx1151 HSACO in `build/hrx-v1-catalog-gfx1151`, with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-p33stage12-c9dc2a616-dirty-20260621-011022/`.
- model and shape:
  standalone Q6_K p33 diagnostic rows covering `64x33` at k256/k512/k3584 and
  `128x33` at k3584, plus the known stress calibration row.
- route:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_p33stage12_f16acc_wg256_f32`.
- baseline command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-vk64-repro`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33stage12-repro`.
- route trace path:
  not applicable; standalone probe only.
- profile or timing artifact path:
  artifact `stdout.txt`, `depwait.stdout.txt`, `stagefull.stdout.txt`,
  `accepted-vk64.stdout.txt`, `static/radv-compare.md`, and
  `static/hsaco-summary.md`.
- correctness result:
  normal p33 rows passed with `bad_gt_0p25=0`; the stress row remains in the
  known VK64 calibration failure band.
- static result:
  the CMake-built HSACO matches the RADV p33 medium `96 buffer_store_b32`,
  `16` WMMA, `48 ds_load_b64`, `LDS=11264`, two barriers, no spills, and the
  first-WMMA `lgkmcnt(40)` window. It still misses the RADV halfword cluster
  topology and lower pressure: HIP has `VGPR=169`, `50 ds_store_b16`,
  `48 ds_load_u16_d16`, and three store clusters, versus RADV's `VGPR=144`,
  `64 ds_store_b16`, `64 ds_load_u16_d16`, and eleven store clusters.
- timing result:
  reject. Same-session p33-stage12 timing was `413.900 us` for
  `64x33 k3584` and `414.057 us` for `128x33 k3584`, versus accepted VK64
  `308.252 us` and `326.901 us`. Depwait and stagefull comparators were in
  the same slow band, around `409-428 us`.
- decision:
  reject before route selection or model A/B.
- notes:
  matching the `96 buffer_store_b32` headline is insufficient. The next Q6 p33
  candidate should target RADV's real halfword store/loadback cluster topology
  and lower accumulator/output live range, not another store-count-only or
  wait-ladder-only spelling.

## 2026-06-21 - Q8_0 production split59 issue-window route

- source:
  `sources/llama.cpp` at `67045a7c93d0-dirty`, adding a production-ABI
  opt-in Q8_0 split59 route.
- build:
  CMake/Ninja target `llama-bench test-backend-ops`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8-baseoff-split59-realdata-67045a7c93d0-dirty-20260621-002755/`.
- model and shape:
  exported focused Q8_0 `MUL_MAT` rows covering p33, p512, and odd-tail p513.
- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_baseoff_split59_bufferstore_f16acc_wg256_f32`.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT` on the exported p512/p513
  files with the default packed-Q8_1 route.
- variant command:
  same focused perf command with
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_ASMWMMA_MOTIF192_K2_BASEOFF_SPLIT59_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- route trace path:
  artifact `focused/p33/route-trace.txt`,
  `focused/p512/route-trace.txt`, and `focused/p513/route-trace.txt`.
- profile or timing artifact path:
  artifact `perf/p512/compare.md`, `perf/p512/compare.json`,
  `perf/p513/compare.md`, and `perf/p513/compare.json`.
- correctness result:
  focused CPU-reference passed p33 `10/10`, p512 `5/5`, and p513 `10/10`;
  see artifact `focused/correctness-summary.json`.
- static result:
  built HSACO passes the accepted Q8 large RADV-style issue-window screen:
  wave64, SGPR `56`, VGPR `205`, LDS `22528`, no scratch/spills, one
  32-WMMA region, final `s_waitcnt lgkmcnt(51)`, and exactly `59` immediate
  LDS loads. See artifact `static/issue-window.md` and
  `static/hsaco-summary.md`.
- route result:
  p33 stayed on existing narrow/default routes. p512 selected split59 only
  for `ffn_out-0`, `ffn_gate-0`, and `result_output`; `Vcur-0` and `Qcur-0`
  stayed on packed Q8_1 routes. p513 selected split59 only for the wide
  513-column ffn/result rows while cols=1 residual rows stayed default.
- timing result:
  reject. p512 default total `65392.050 us`, split59 total `98726.973 us`,
  ratio `1.510x` slower. p513 default total `81068.259 us`, split59 total
  `114518.954 us`, ratio `1.413x` slower. Selected ffn/result rows regressed
  by `1.35x-1.61x`.
- decision:
  keep opt-in only and reject for promotion/default.
- notes:
  this is an important negative result. The production route mechanically
  reproduced the visible RADV split59 LDS issue-window contract and remained
  correctness-clean on odd/tail coverage, but still lost badly to the current
  packed-Q8_1 route. Follow-up RADV-vs-split59 ownership artifact
  `cache/hrxv1/gfx1151/q8-split59-radv-ownership-dc8406cb1-20260621-003703/`
  shows the harder mismatch: RADV Q8 large uses compact width-4 dst/C
  accumulator operands for all `32` f16 WMMAs, while HIP split59 uses width-8
  dst/C operands for all `32`. Further Q8 parity work should stop treating
  wait-window equivalence as sufficient; a f16-WMMA candidate should pass the
  compact accumulator screen before focused p512/p513 timing unless it
  deliberately pivots to a measured packed route.

## 2026-06-20 - Q6_K MUL_MAT_ID WMMA operand-width analysis

- source:
  `sources/llama.cpp` at `67da9c87c-dirty`, extending
  `tools/vulkan-oracle/extract_wmma_ownership.py`.
- build:
  no kernel rebuild required. The analysis consumes the existing
  CMake/Ninja-built `bankedcompact` device objdump and RADV Q6 ID ISA.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-bankedcompact-width-analysis-67da9c87c-dirty-20260620-234925/`.
- purpose:
  separate raw register naming from the actual primitive mismatch after the
  `bankedcompact` fixture matched RADV on A/B cardinality and wait ladder but
  still reported different dst/C banks.
- result:
  the enhanced extractor reports operand-width histograms. RADV Q6 ID uses
  width-8 A and B operands for all `16` WMMAs, but width-4 dst/C operands for
  all `16` WMMAs. HIP `bankedcompact` uses width-8 A/B and also width-8 dst/C
  operands for all `16` WMMAs.
- decision:
  treat dst/C width as a hard static screen for future RADV-style f16-WMMA Q6
  ID candidates. The remaining mismatch is concrete: RADV's accumulator
  operand class is compact width-4 while the current HIP C++ builtins and
  inline-asm spellings emit width-8 dst/C. A route-facing candidate should not
  proceed to p33/p512/p513 gates unless it either fixes that primitive mismatch
  or intentionally pivots to packed Q8_1/x4 with focused timing wins.

## 2026-06-20 - F16 WMMA tied builtin primitive probe

- source:
  `sources/llama.cpp` at `0eba9b41c-dirty`, extending
  `hrx-hip-bench-wmma-f16-lane-map`.
- build:
  CMake/Ninja target `hrx-hip-bench-wmma-f16-lane-map`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-tied-lane-map-ownership-0eba9b41c-dirty-20260620-234412/`.
- purpose:
  test whether Clang's
  `__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64` exposes the compact
  dst/C operand behavior seen in the RADV Q6 `MUL_MAT_ID` oracle. LLVM's
  intrinsic metadata says the tied form preserves the non-selected half of the
  accumulator, so it was the next candidate primitive after the Q6
  `bankedcompact` fixture still emitted wide/overlapping dst/C ranges.
- result:
  `--mode=tied-basic` validates the same lane-map behavior as the existing
  builtin: `opsel=0` changes `256` even slots and `opsel=1` changes `256`
  odd slots, with no coordinate-map failures.
- static evidence:
  the tied op_sel0 probe emits one
  `v_wmma_f16_16x16x16_f16 v[9:16], v[1:8], v[1:8], v[9:16]`, two
  `global_store_b128`, wave64, SGPR `12`, VGPR `13`, no spills, and no
  scratch. The emitted dst/C operand form matches the ordinary HIP builtin's
  wide range, not RADV's compact examples such as `v[52:55]`.
- decision:
  reject the tied builtin as the missing compact dst/C primitive. It may still
  be useful for preserving the non-selected half semantically, but it does not
  by itself produce the RADV-like operand ownership needed for the Q6 ID clone.
- follow-up:
  the generated Clang builtin table also exposes
  `__builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12` with a compact
  `V4xV4xV4xV4x` signature. A transient CMake/Ninja build probe rejected it
  for `gfx1151` with:
  `needs target feature wmma-128b-insts,wavefrontsize64`. That rules out the
  obvious gfx12 compact builtin path for this target unless the target feature
  model changes.

## 2026-06-20 - Q6_K MUL_MAT_ID banked compact-accumulator fixture

- source:
  `sources/llama.cpp` at `0a468b9fd-dirty`, extending
  `hrx-hip-bench-q6-id-subgroup-contract`.
- build:
  CMake/Ninja target `hrx-hip-bench-q6-id-subgroup-contract`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-subgroup-contract-bankedcompact-0a468b9fd-dirty-20260620-233607/`.
- purpose:
  test whether replacing the banked fixture's `_Float16 x8` accumulator with
  a packed `uint32_t x4` inline-asm operand forces HIP toward RADV's eight
  compact four-register dst/C WMMA banks while preserving the banked A/B
  ownership.
- result:
  all rows validated over the fixture's 2048 output values. Same executable
  timing over 2000 reps:
  direct `2.263697 us`, staged `3.363540 us`, loaddeep `3.539424 us`,
  minstore `3.449291 us`, banked `3.764099 us`, bankedcompact
  `3.774900 us`.
- static evidence:
  bankedcompact emits wave64, SGPR `14`, VGPR `110`, LDS `28672`,
  private segment `0`, no spills, `16` WMMA, `52 ds_load_b64`,
  `32 ds_load_u16_d16`, `32 ds_store_b16`, `24 ds_store_b128`,
  `32 global_store_b32`, `2` barriers, `46 s_waitcnt`, and
  `2 s_waitcnt_depctr`. The ownership extractor confirms `8` A banks and
  `4` B banks and now preserves RADV's visible wait ladder:
  `[40, 36, 32, 28, 24, 16, 12, 8, 4, 0]`. The destination/C result is worse
  than the target: only `5` extracted wide/overlapping banks instead of RADV's
  eight compact four-register banks.
- decision:
  reject as a route candidate and keep only as a static fixture result. The
  full wait ladder is expressible in HIP C++, but the packed accumulator
  spelling does not solve accumulator ownership or store inflation. The next
  route-facing Q6 ID attempt should preserve the banked A/B ownership and this
  wait ladder while changing the accumulator/store primitive, or intentionally
  pivot back to a packed Q8_1/x4 route with focused gate/up plus down wins.

## 2026-06-20 - Q6_K MUL_MAT_ID banked ownership fixture

- source:
  `sources/llama.cpp` at `4bbd6ce54-dirty`, extending
  `hrx-hip-bench-q6-id-subgroup-contract`.
- build:
  CMake/Ninja target `hrx-hip-bench-q6-id-subgroup-contract`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-subgroup-contract-banked-deps-4bbd6ce54-dirty-20260620-232846/`.
- purpose:
  test the next mechanical RADV Q6 `MUL_MAT_ID` schedule delta after minstore
  matched the opcode surface but collapsed all WMMA operands into one A/B
  bank. The `banked` row uses twelve distinct LDS fragments so the 16 WMMAs
  have the RADV cardinality: eight A banks and four B banks.
- result:
  all rows validated over the fixture's 2048 output values. Same executable
  timing over 2000 reps:
  direct `2.266737 us`, staged `3.361962 us`, loaddeep `3.539264 us`,
  minstore `3.454530 us`, banked `3.766118 us`.
- static evidence:
  banked emits wave64, SGPR `14`, VGPR `110`, LDS `12288`, private segment
  `0`, no spills, `16` WMMA, `52 ds_load_b64`, `32 ds_load_u16_d16`,
  `32 ds_store_b16`, `24 ds_store_b128`, `32 global_store_b32`, `2`
  barriers, and `46 s_waitcnt`. The ownership extractor confirms `8` A banks
  and `4` B banks, matching RADV's operand-bank cardinality. The destination
  banks are still `7` wide/overlapping half-vector ranges rather than RADV's
  eight four-register accumulator banks, and the visible wait ladder is
  `[0, 36, 32, 28, 24, 16, 8, 4]` instead of RADV's
  `[40, 36, 32, 28, 24, 16, 12, 8, 4, 0]`.
- decision:
  keep as an accepted static fixture prior and reject as a route candidate.
  It proves the 8xA/4xB ownership cardinality is expressible in HIP C++, but
  it is slower than minstore/staged and still misses the exact wait ladder,
  accumulator packing, and RADV's `2 ds_store_b128` store surface. The next
  route-facing Q6 ID attempt should preserve the banked operand cardinality
  while solving wait placement and store inflation before p33/p512/p513 gates.

## 2026-06-20 - Q6_K MUL_MAT_ID WMMA ownership extraction

- source:
  `sources/llama.cpp` at `68bd4a40e-dirty`, adding
  `tools/vulkan-oracle/extract_wmma_ownership.py`.
- build:
  no kernel rebuild required. The tool consumes existing RADV ISA text and
  CMake/Ninja-built HIP fixture objdump text from prior artifacts.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-wmma-ownership-68bd4a40e-dirty-20260620-232047/`.
- purpose:
  mechanically compare the p33 RADV Q6 `MUL_MAT_ID` WMMA operand ownership
  against the HIP `minstore` subgroup-contract fixture. This addresses the
  remaining ambiguity after minstore matched the headline static opcode
  surface but still represented synthetic one-fragment work.
- result:
  RADV and HIP minstore both show `16` WMMA and `52 ds_load_b64`, but the
  operand maps are different. RADV uses eight A fragment banks
  (`v[32:39]`, `v[72:79]`, `v[80:87]`, `v[88:95]`, `v[96:103]`,
  `v[120:127]`, `v[128:135]`, `v[136:143]`) and four B banks
  (`v[40:47]`, `v[64:71]`, `v[104:111]`, `v[112:119]`) with a
  `lgkmcnt` ladder `40,36,32,28,24,16,12,8,4,0`. HIP minstore uses one A
  and one B bank (`v[4:11]`) for all 16 WMMAs and only two explicit
  `lgkmcnt(0)` wait anchors.
- decision:
  accept the extractor and ownership report as the next Q6 ID schedule ledger
  anchor. The minstore fixture is confirmed as a static-surface match only,
  not an operand-ownership match. The next route-facing candidate should be a
  banked subgroup-ID clone preserving RADV's 8xA/4xB operand reuse and wait
  ladder, then validated against p33/p512/p513 focused rows before any model
  A/B.

## 2026-06-20 - Q6_K MUL_MAT_ID minstore static-surface fixture

- source:
  `sources/llama.cpp` at `dcba2f11e-dirty`, extending
  `hrx-hip-bench-q6-id-subgroup-contract`.
- build:
  CMake/Ninja target `hrx-hip-bench-q6-id-subgroup-contract`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-subgroup-contract-minstorepad-dcba2f11e-dirty-20260620-231235/`.
- purpose:
  test whether HIP C++ can hit the Q6 `MUL_MAT_ID` RADV ID headline static
  surface after loaddeep matched `52 ds_load_b64` but still emitted
  `8 ds_store_b128`. The `minstore` row reuses one operand fragment bank for
  all thirteen 4x64-bit LDS loads and pads LDS allocation back to 12 KiB
  without adding memory operations.
- result:
  direct, staged, loaddeep, and minstore rows all validated over 2048 output
  values. Same executable timing over 2000 reps:
  direct `2.221468 us`, staged `3.375103 us`, loaddeep `3.533869 us`,
  minstore `3.448519 us`.
- static evidence:
  extracted AMDGPU object shows the minstore probe at wave64, SGPR `12`, VGPR
  `35`, LDS `12288`, private segment `0`, `16` WMMA, `52 ds_load_b64`,
  `32 ds_store_b16`, `32 ds_load_u16_d16`, `2 ds_store_b128`,
  `32 buffer_store_b32`, `2` barriers, and `39 s_waitcnt`.
- decision:
  accept as the current low-level static contract prior, not as a route
  candidate. This proves the RADV-like static surface is expressible in HIP
  C++, but the synthetic one-fragment ownership is slower than the simpler
  staged row and does not represent useful real-tile work. The next Q6 ID
  production attempt must transfer the actual RADV lane/operand ownership into
  the route ABI, or switch to a packed Q8_1/x4 route with focused wins on both
  gate/up and down rows.

## 2026-06-20 - Q6_K MUL_MAT_ID load-depth contract fixture

- source:
  `sources/llama.cpp` at `2f1322537-dirty`, extending
  `hrx-hip-bench-q6-id-subgroup-contract`.
- build:
  CMake/Ninja target `hrx-hip-bench-q6-id-subgroup-contract`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-subgroup-contract-loaddeep-2f1322537-dirty-20260620-230734/`.
- purpose:
  isolate the next RADV Q6 `MUL_MAT_ID` delta after the subgroup-contract
  fixture matched the 12 KiB/16-WMMA/32-halfword surface but only emitted
  `16 ds_load_b64`. The new `loaddeep` row deliberately issues thirteen
  4x64-bit LDS fragment loads to test whether HIP C++ can preserve RADV's
  `52 ds_load_b64` operand-load surface in the same fixture.
- result:
  direct, staged, and loaddeep rows all validated over 2048 output values.
  Same executable timing over 2000 reps:
  direct `2.241520 us`, staged `3.378919 us`, loaddeep `3.566414 us`.
- static evidence:
  extracted AMDGPU object shows the loaddeep probe at wave64, SGPR `14`, VGPR
  `61`, LDS `12288`, private segment `0`, `16` WMMA, `52 ds_load_b64`,
  `32 ds_store_b16`, `32 ds_load_u16_d16`, `32 buffer_store_b32`, `2`
  barriers, and `39 s_waitcnt`. It still emits `8 ds_store_b128` versus RADV
  ID's `2 ds_store_b128`.
- decision:
  reject operand-load depth alone as a route direction. It matches RADV's
  `52 ds_load_b64` count but slows the fixture and still lacks the real
  lane/operand ownership and wait-overlap contract. A direct-F32 Q6 ID route
  should not be attempted until that ownership map is understood, or else the
  next productive path is a packed Q8_1/x4 candidate with a named deviation
  from RADV and focused gate/up plus down wins.

## 2026-06-20 - Q6_K MUL_MAT_ID subgroup-contract fixture

- source:
  `sources/llama.cpp` commit
  `4dbcc52c9 hrx: add q6 id subgroup contract bench`; artifact directory was
  captured just before commit and therefore carries the `a93bb23ff-dirty`
  label.
- build:
  added CMake/Ninja target `hrx-hip-bench-q6-id-subgroup-contract`, built in
  `build/hrx-v1-catalog-gfx1151` with ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-id-subgroup-contract-bench-a93bb23ff-dirty-20260620-225935/`.
- purpose:
  isolate the Q6 `MUL_MAT_ID` RADV subgroup-ID store/load contract before
  trying another route-facing direct-F32 WMMA bridge. The staged probe targets
  the source-visible RADV facts that the rejected staged VK64 route missed:
  wave64, 12 KiB LDS, 16 WMMA, 32 halfword LDS stores, 32 halfword LDS
  loadbacks, 32 global stores, and no spills.
- result:
  both direct and staged fixture rows validated over 2048 output values. Same
  executable timing over 2000 reps was `2.228195 us` for direct writeback and
  `3.366400 us` for the staged halfword-loadback motif.
- static evidence:
  extracted AMDGPU object shows the staged probe at wave64, SGPR `14`, VGPR
  `50`, LDS `12288`, private segment `0`, `16` visible WMMA,
  `32 ds_store_b16`, `32 ds_load_u16_d16`, `32 buffer_store_b32`, `2`
  barriers, and `39 s_waitcnt`. It still only emits `16 ds_load_b64` and
  `8 ds_store_b128`, versus RADV ID's `52 ds_load_b64` and `2 ds_store_b128`.
- decision:
  accept as a low-level fixture, not as a route candidate. This closes the
  simple "make the staged ID writeback look like RADV's halfword surface"
  subproblem but does not explain parity. The next route-facing Q6 ID attempt
  must also reproduce the operand-load/lane-ownership depth, or else pivot back
  to a packed Q8_1/x4 schedule that improves both gate/up and down rows.

## 2026-06-20 - Q6_K MUL_MAT_ID staged VK64 WMMA rejection

- source:
  `sources/llama.cpp` at
  `c95718b4e hrx: reject q6 id mmq64 token route` plus local dirty changes
  adding the opt-in
  `hrx_mul_mat_id_q6_k_wmma16x16_staged_vk64_f16acc_wg256_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`; the new HSACO was built through the normal
  CMake/Ninja target
  `ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_id_q6_k_wmma16_staged_vk64.hsaco`,
  followed by `ggml-hrx` and `test-backend-ops`.
- model/shape:
  Qwen3 30B Q6_K MoE `MUL_MAT_ID` focused rows from
  `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/`:
  p33 and p513 `ffn_moe_gate-0` / `ffn_moe_down-0`.
- route or kernel candidate:
  staged direct-F32 WMMA ID bridge: wave64, WG256, BM64/BN64/BK32,
  `route_tile_n=64`, 48-half shared stride, and 12 KiB LDS. Env:
  `GGML_HRX_ENABLE_Q6_K_ID_WMMA16_STAGED_VK64_PROMPT=1`.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT_ID --test-file <p33|p513>/focused/moe_qk_prompt.txt --output csv`.
- variant command:
  `GGML_HRX_ENABLE_Q6_K_ID_WMMA16_STAGED_VK64_PROMPT=1 GGML_HRX_EXPECT_MUL_MAT_ID_Q6_PROVIDER=hrx_mul_mat_id_q6_k_wmma16x16_staged_vk64_f16acc_wg256_f32 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT_ID --test-file <p33|p513>/focused/moe_qk_prompt.txt --output csv`.
- route trace path:
  `cache/hrxv1/gfx1151/q6-id-wmma16-staged-vk64-focused-c95718b4e-dirty-20260620-224319/staged/test.stderr.log`
  and
  `cache/hrxv1/gfx1151/q6-id-wmma16-staged-vk64-focused-c95718b4e-dirty-20260620-224319/staged-p513/test.stderr.log`.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-id-wmma16-staged-vk64-focused-c95718b4e-dirty-20260620-224319/`.
- correctness result:
  passed p33 and p513 focused CPU-reference for both rows. p33 route traces
  selected the new provider with workgroups `[12,1,128]` and `[32,1,128]`,
  matching the Vulkan ID oracle's one-Y-group p33 geometry.
- timing result:
  rejected versus the accepted grouped Q8_1/x4 route. p33 `ffn_moe_gate-0`
  regressed `290.697 us -> 342.517 us`, p33 `ffn_moe_down-0` regressed
  `261.210 us -> 359.243 us`, p513 gate regressed
  `2474.333 us -> 2901.407 us`, and p513 down regressed
  `2331.777 us -> 3232.209 us`.
- decision:
  reject for promotion and keep opt-in only. Matching Vulkan's p33 Y-grid plus
  a staged 12 KiB wave64 WMMA surface is not enough; the next Q6 ID candidate
  needs the actual subgroup-ID lane ownership/LDS halfword contract, or a
  packed route that improves gate/up and down together.
- notes:
  static metadata: wave64, SGPR `67`, VGPR `113`, LDS `12288`, private segment
  `0`, no spills. Static opcode counts include `8` visible `v_wmma`,
  `16` LDS-read-class ops, `2` LDS-write-class ops, `16` global/buffer loads,
  `16` global/buffer stores, `2` barriers, and `66` `s_waitcnt`.

## 2026-06-20 - Q6_K MUL_MAT_ID MMQ64x64 p33 token-group rejection

- source:
  `sources/llama.cpp` at
  `c012bdbf0 hrx: reject q6 explicit wait route` plus local dirty changes
  adding the opt-in
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x64_wg64_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`; CMake/Ninja generated the updated
  `mul_mat_id_q6_k_q8_1_x4_mmq.hsaco` while building `ggml-hrx` and
  `test-backend-ops`.
- model/shape:
  Qwen3 30B Q6_K MoE `MUL_MAT_ID` p33 focused rows from
  `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt`.
- route or kernel candidate:
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x64_wg64_f32`, enabled only with
  `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ64_PROMPT=1`.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT_ID --test-file cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt --output csv`.
- variant command:
  `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ64_PROMPT=1 GGML_HRX_EXPECT_MUL_MAT_ID_Q6_PROVIDER=hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x64_wg64_f32 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT_ID --test-file cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt --output csv`.
- route trace path:
  `cache/hrxv1/gfx1151/q6-id-mmq64x64-focused-c012bdbf0-dirty-20260620-223040/default/test.stderr.log`
  and
  `cache/hrxv1/gfx1151/q6-id-mmq64x64-focused-c012bdbf0-dirty-20260620-223040/mmq64/test.stderr.log`.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-id-mmq64x64-focused-c012bdbf0-dirty-20260620-223040/`.
- correctness result:
  passed both focused CPU-reference rows. Default route traces selected
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32` with
  `route_tile_n=16` and workgroups `[12,3,128]` / `[32,3,128]`; opt-in traces
  selected the new provider with `route_tile_n=64` and workgroups
  `[12,1,128]` / `[32,1,128]`, matching the Vulkan oracle's p33 Y-grid.
- timing result:
  rejected overall. `ffn_moe_gate-0` regressed `311.808 us -> 377.120 us`,
  while `ffn_moe_down-0` improved `282.727 us -> 224.608 us`. The two-row
  sum regressed slightly, `594.534 us -> 601.728 us`, and the model has two
  gate/up-shaped rows for each down row, so the likely integrated p33 mix is
  worse than the two-row sum suggests.
- decision:
  reject for promotion and keep opt-in only. Matching Vulkan's p33 one-Y-group
  launch inside the current packed-dot schedule is not sufficient; the next
  Q6 ID route needs a true staged WMMA/RADV-like dataflow or a packed schedule
  that improves gate/up as well as down.
- notes:
  static metadata for the new export is wave64, SGPR `72`, VGPR `163`, LDS
  `5120`, private segment `0`, no spills. Symbol-sliced ISA shows `512`
  `v_dot4`, `114` LDS-load-class ops, `28` LDS-store-class ops, `64`
  global-store-class ops, and `2` barriers. This moves the p33 launch geometry
  toward RADV but not the RADV math/staging family.

## 2026-06-20 - Q6_K MUL_MAT_ID current-head p33 focused baseline

- source:
  `sources/llama.cpp` clean at
  `c012bdbf0 hrx: reject q6 explicit wait route`.
- build:
  `build/hrx-v1-catalog-gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Qwen3 30B Q6_K MoE `MUL_MAT_ID` p33 focused rows from
  `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt`.
- route or kernel candidate:
  current default
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test -b HRX0 -o MUL_MAT_ID --test-file cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt --output csv`.
- variant command:
  none; this is a current-head baseline refresh for the Q6 ID Vulkan-oracle
  ledger.
- route trace path:
  `cache/hrxv1/gfx1151/q6-id-current-head-p33-focused-c012bdbf0-20260620-222329/test.stderr.log`
  and
  `cache/hrxv1/gfx1151/q6-id-current-head-p33-focused-c012bdbf0-20260620-222329/perf.stderr.log`.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-id-current-head-p33-focused-c012bdbf0-20260620-222329/`.
- correctness result:
  passed both focused CPU-reference rows. Route traces selected the grouped
  Q8_1/x4 Q6 ID provider for both `ffn_moe_gate-0` and `ffn_moe_down-0`.
- timing result:
  `ffn_moe_gate-0` ran in `306.466 us`; `ffn_moe_down-0` ran in
  `277.896 us`.
- decision:
  use this as the current p33 grouped-ID throughput floor for future Q6 ID
  route candidates. The schedule comparison against RADV is recorded in
  `docs/hrxv1/q6k-mul-mat-id-vulkan-oracle-schedule-ledger.md`.
- notes:
  p33 HRX launches `[12,3,128]` and `[32,3,128]`, while Vulkan's
  `matmul_id_subgroup_q6_k_f32_f16acc_aligned_m` p33 oracle launches
  `[12,1,128]` and `[32,1,128]`. A future candidate needs to explain that
  token-grouping delta before model-level A/B.

## 2026-06-20 - Q6_K VK64 padladder explicit-wait route rejection

- source:
  `sources/llama.cpp` at
  `7af9c70e1 hrx: reject q6 radv96 duplicate route` plus local dirty changes
  adding the opt-in
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_padladder_expwait_f16acc_wg256_f32`
  route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; CMake/Ninja generated the new HSACO as
  `generated/hsaco/gfx1151/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_padladder_expwait_wg256.hsaco`
  while building target `ggml-hrx test-backend-ops`.
- model/shape:
  focused Qwen3 30B Q6_K p33 prompt rows exported in
  `cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/q6_prompt_p33.txt`:
  `Vcur-0-p33`, `ffn_out-0-p33`, and `result_output-p33`.
- route or kernel candidate:
  route-facing transfer of the accepted fixture
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperexpwait` first-WMMA
  wait ladder. This candidate keeps the in-bounds normal 64-store output path
  and changes only the padladder WMMA issue waits to the explicit
  `lgkmcnt(40),36,32,28,24,20,16,12,8,4,0` ladder. It is an axis isolation
  probe, not a full RADV96 store-contract clone.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops perf -b HRX0 -o MUL_MAT --test-file cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/q6_prompt_p33.txt --output csv`.
- variant command:
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_RING96_K2_KLOOP_ASM_COPYAB_PADLADDER_EXPWAIT_F16ACC_WG256_PROMPT=1 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/q6_prompt_p33.txt --output csv`.
- route trace path:
  correctness route trace:
  `cache/hrxv1/gfx1151/q6-vk64-padladder-expwait-focused-routetrace-7af9c70e1-dirty-20260620-221508/expwait-test-p33.routes.log`;
  perf route traces:
  `cache/hrxv1/gfx1151/q6-vk64-padladder-expwait-perf-7af9c70e1-dirty-20260620-221524/default-perf-p33.routes.log`
  and
  `cache/hrxv1/gfx1151/q6-vk64-padladder-expwait-perf-7af9c70e1-dirty-20260620-221524/expwait-perf-p33.routes.log`.
- profile or timing artifact path:
  static ISA/resource artifact:
  `cache/hrxv1/gfx1151/q6-vk64-padladder-expwait-static-7af9c70e1-dirty-20260620-221355/`;
  focused timing artifact:
  `cache/hrxv1/gfx1151/q6-vk64-padladder-expwait-perf-7af9c70e1-dirty-20260620-221524/`.
- correctness result:
  passed all three focused p33 CPU-reference rows with the expwait provider
  selected for every row.
- timing result:
  rejected versus the accepted H4LOAD default. `Vcur-0-p33` regressed
  `397.144 us -> 514.373 us` (`1.295x` slower), `ffn_out-0-p33` regressed
  `2405.729 us -> 2922.757 us` (`1.215x` slower), and
  `result_output-p33` regressed `11301.292 us -> 20686.143 us`
  (`1.830x` slower). Total focused row time regressed
  `14.104 ms -> 24.123 ms` (`1.710x` slower).
- decision:
  reject as a performance route and keep opt-in only. The built HSACO proves
  HIP can express the RADV-like first-WMMA `lgkmcnt(40)` window in the
  route-facing kernel, but that wait ladder alone does not close the Q6 p33
  gap. The next useful Q6 axis should stop retesting explicit wait ladders and
  instead target the remaining structural RADV deltas: store-contract/output
  ownership without duplicate stores, lower VGPR than 170, and avoiding the
  high-cost K2/padladder dataflow that amplifies `result_output`.

## 2026-06-20 - Current-head Q6_K parity checkpoint after RADV96 rejections

- source:
  `sources/llama.cpp` at
  `7af9c70e1 hrx: reject q6 radv96 duplicate route`, clean worktree.
- build:
  `build/hrx-v1-catalog-gfx1151` and `build/vulkan-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; both rebuilt through CMake/Ninja and both
  report build commit `7af9c70e1`.
- model/shape:
  Qwen3 30B Q6_K from
  `shared/models/llamacpp-hrx2-basket-v1`, prefill `p33`, `p512`, and `p513`
  with `--flash-attn 1`, three repetitions.
- route or kernel candidate:
  no new route. This is a commit-aligned KPI/boulder checkpoint after the Q6
  RADV96 sidecar and duplicate-output probes were rejected.
- baseline command:
  `python3 tools/hrxv1_basket_benchmark.py --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --flash-attn 1 --tag q6-current-head-7af9c70e1-r3`.
- variant command:
  after noticing the Vulkan binary was stale at commit `85edc5327`, rebuilt
  `build/vulkan-gfx1151` and reran
  `python3 tools/hrxv1_basket_benchmark.py --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k --cases p33,p512,p513 --backends vulkan --repetitions 3 --flash-attn 1 --tag q6-current-head-7af9c70e1-vulkan-r3`.
- route trace path:
  HRX route traces are in
  `cache/hrxv1/gfx1151/q6-current-head-7af9c70e1-r3/hrx/unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k/p33/stderr.log`,
  `.../p512/stderr.log`, and `.../p513/stderr.log`.
- profile or timing artifact path:
  HRX artifact:
  `cache/hrxv1/gfx1151/q6-current-head-7af9c70e1-r3/`;
  commit-aligned Vulkan artifact:
  `cache/hrxv1/gfx1151/q6-current-head-7af9c70e1-vulkan-r3/`.
- correctness result:
  not a CPU-reference route gate. This checkpoint confirms backend identity:
  HRX JSON rows report `backends=HRX`, Vulkan JSON rows report
  `backends=Vulkan`, and both report build commit `7af9c70e1`. HRX fallback
  line count is zero for all three rows.
- timing result:
  corrected steady ratios using HRX from the first artifact and commit-aligned
  Vulkan from the second artifact:

  | Row | HRX steady tok/s | Vulkan steady tok/s | HRX/Vulkan |
  | --- | ---: | ---: | ---: |
  | p33 | `91.027` | `188.376` | `0.483x` |
  | p512 | `581.472` | `1083.740` | `0.537x` |
  | p513 | `580.711` | `1024.460` | `0.567x` |

  The three-row steady geomean is `0.528x` Vulkan.
- decision:
  Q6 remains a stable active boulder. The selected HRX dense Q6 routes are
  still `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32`
  for p33 and
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32` for
  p512/p513; grouped Q6 ID remains selected separately. Do not promote another
  Q6 route unless it changes a named RADV schedule delta and passes focused
  p33/p512/p513 CPU-reference, route, static, and same-runner timing gates.
- notes:
  the first combined HRX/Vulkan artifact remains useful for HRX route evidence
  but its Vulkan rows were built from stale commit `85edc5327`; use the
  separate `q6-current-head-7af9c70e1-vulkan-r3` artifact for current Vulkan
  ratios.

## 2026-06-20 - Q6_K VK64 padladder RADV96 duplicate-output rejection

- source:
  `sources/llama.cpp` at
  `e0071be01 hrx: add q6 radv96 explicit wait fixture` plus local opt-in
  route `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_padladder_radv96_duplicate_f16acc_wg256_f32`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; built through CMake/Ninja target
  `ggml-hrx test-backend-ops`.
- model/shape:
  Qwen3 30B Q6_K dense prompt p33 medium rows exported in
  `cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/q6_prompt_p33.txt`.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_padladder_radv96_duplicate_f16acc_wg256_f32`.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <q6_prompt_p33.txt> --output csv`.
- variant command:
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_RING96_K2_KLOOP_ASM_COPYAB_PADLADDER_RADV96_DUPLICATE_F16ACC_WG256_PROMPT=1 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <q6_prompt_p33.txt> --output csv`.
- route trace path:
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-duplicate-focused-e0071be01-dirty-20260620-215636/test.stderr.log`
  and
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-duplicate-perf-e0071be01-dirty-20260620-215702/duplicate.stderr.log`.
- profile or timing artifact path:
  static artifact
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-duplicate-static-e0071be01-dirty-20260620-215319/`;
  focused correctness artifact
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-duplicate-focused-e0071be01-dirty-20260620-215636/`;
  focused perf artifact
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-duplicate-perf-e0071be01-dirty-20260620-215702/`.
- correctness result:
  focused p33 CPU-reference passed all three rows while route traces selected
  the duplicate provider for Vcur, ffn_out, and result_output. This proves the
  sidecar sentinel failure can be avoided by mapping the extra RADV96 groups
  back to in-bounds logical output tiles.
- timing result:
  rejected. Current default H4LOAD times were `397.009 us`, `2467.540 us`,
  and `11597.065 us`; duplicate-output times were `515.121 us`,
  `2794.594 us`, and `20654.221 us`. Total focused p33 row time regressed
  `14.462 ms -> 23.964 ms`, and result_output regressed `1.781x`.
- static evidence:
  duplicate-output preserved the same useful static headline as the rejected
  sidecar no-merge probe without out-of-bounds writes: `16` WMMA,
  `48 ds_load_b64`, first-WMMA wait `lgkmcnt(40)`,
  `64 ds_load_u16_d16`, `96 buffer_store_b32`, no `buffer_store_b128`, and no
  spills. It still diverged from RADV on the important remaining costs:
  `LDS=14336` versus RADV `11264`, `VGPR=170` versus `144`, `8` barriers
  versus `2`, `17` depctrs versus `2`, and only `2 ds_store_b16` versus RADV
  `64`.
- decision:
  keep as opt-in diagnostic only and reject before model A/B. This closes the
  hypothesis that the route-facing sidecar failure was merely an output bounds
  issue: in-bounds duplicate writes are correct but much slower than the
  accepted H4LOAD route. The next Q6 p33 route-facing attempt should not add
  duplicate output traffic; it needs lower-live output ownership or a lower
  barrier/staging topology that creates RADV's store surface as useful work.

## 2026-06-20 - Q6_K VK64 padladder RADV96 sidecar route rejection

- source:
  `sources/llama.cpp` at
  `e0071be01 hrx: add q6 radv96 explicit wait fixture` plus local
  route-facing catalog/provider wrappers for upper-stage and RADV96 sidecar
  Q6_K VK64 padladder probes. The wrappers were removed after rejection and
  are not production route material.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; built through CMake/Ninja target
  `ggml-hrx test-backend-ops`.
- model/shape:
  Qwen3 30B Q6_K dense prompt p33 medium rows exported in
  `cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/q6_prompt_p33.txt`.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_padladder_radv96_sidecar_f16acc_wg256_f32`.
- baseline command:
  current default H4LOAD route selected by
  `test-backend-ops test -b HRX0 -o MUL_MAT --test-file <q6_prompt_p33.txt> --output csv`.
- variant command:
  `GGML_HRX_DISABLE_Q6_K_WMMA16_VK64_PADDED44_W64_H4LOAD_F16ACC_WG256_PROMPT=1 GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_RING96_K2_KLOOP_ASM_COPYAB_PADLADDER_RADV96_SIDECAR_F16ACC_WG256_PROMPT=1 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test -b HRX0 -o MUL_MAT --test-file <q6_prompt_p33.txt> --output csv`.
- route trace path:
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-sidecar-nomerge-forced-focused-e0071be01-dirty-20260620-214655/test.stderr.log`.
- profile or timing artifact path:
  static artifacts
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-padladder-radv96-sidecar-static-e0071be01-dirty-20260620-214354/`
  and
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-padladder-radv96-sidecar-nomerge-static-e0071be01-dirty-20260620-214518/`;
  focused correctness artifact
  `cache/hrxv1/gfx1151/q6-vk64-padladder-radv96-sidecar-nomerge-forced-focused-e0071be01-dirty-20260620-214655/`.
- correctness result:
  the unforced run passed but selected the existing
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32`
  provider, so it was not evidence for the candidate. With H4LOAD disabled,
  the candidate selected for all three p33 rows and failed all rows with
  `sentinel mismatch: sent_3`, proving the sidecar writes outside the logical
  output tensor in the real route ABI.
- timing result:
  skipped. A route that corrupts the sentinel buffer is not timing material.
- static evidence:
  the coalesced sidecar build preserved the intended first-WMMA wait window
  (`16` WMMA, `48 ds_load_b64`, final `lgkmcnt(40)`) but still emitted only
  `64 buffer_store_b32` plus `buffer_store_b128` sidecar coalescing. Adding the
  no-merge store spelling forced the headline RADV-like surface:
  `16` WMMA, `48 ds_load_b64`, `64 ds_load_u16_d16`, `96 buffer_store_b32`,
  no `buffer_store_b128`, no scratch/private segment, and first wait
  `lgkmcnt(40)`. It still diverged from RADV on route-safety and resource
  shape: `LDS=14336` versus RADV `11264`, `VGPR=170` versus RADV `144`,
  `8` barriers versus RADV `2`, `17` depctrs versus RADV `2`, and only
  `2 ds_store_b16` versus RADV `64`.
- decision:
  reject before timing or model A/B and remove the route-facing catalog/provider
  additions. The experiment proves the strongest fixture prior does not
  transfer directly through a sidecar into the real Q6_K route ABI. The next
  useful Q6 p33 work must make the extra ownership/store surface real without
  out-of-bounds sidecar writes, or pivot to a lower-level/lane-ownership
  implementation that can reproduce RADV's 96-store topology under the actual
  output contract.

## 2026-06-20 - Q6_K VK64 p33 staged-tail rejection

- source:
  `sources/llama.cpp` at
  `da3ac5ad8 hrx: reject q6 p33 branch-store probe` plus the local p33
  staged-tail diagnostic.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33stagetail-repro` and the
  matching generated HSACO target.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium narrow prompt schedule probe for Qwen3
  30B Q6_K. The harness intentionally skips the `64x64` row because this
  diagnostic compiles a `cols=33` output branch policy.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_p33stagetail_f16acc_wg256_f32`.
- baseline command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-repro`
  and
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33branch-repro`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33stagetail-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-p33stagetail-da3ac5ad8-dirty-20260620-204240/`.
- correctness result:
  normal p33 focused rows passed with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, and `128x33 k3584`. The stress row remains in
  the known VK64 calibration failure band.
- timing result:
  variant `64x33 k3584` was `407.462 us` and `128x33 k3584` was `412.732 us`.
  Same-run depwait baseline was `406.415 us` and `412.703 us`; p33branch was
  `412.387 us` and `407.618 us`. This is noise/flat, with no production-sized
  timing signal.
- static evidence:
  the candidate preserves the RADV first-WMMA/load window exactly:
  `59` pre-hot loads, `48` immediate LDS loads, final `lgkmcnt(40)`, and
  `16` WMMA. The tail-only LDS staging changed the store surface from
  p33branch's `48 buffer_store_b32`, `2 ds_store_b16`, `1 ds_load_u16_d16`,
  `VGPR=169` to `48 buffer_store_b32`, `18 ds_store_b16`,
  `16 ds_load_u16_d16`, `VGPR=170`. RADV p33 medium remains
  `96 buffer_store_b32`, `64 ds_store_b16`, `64 ds_load_u16_d16`, and
  `VGPR=144`.
- decision:
  reject before route selector or model A/B. The experiment confirms that
  adding the odd-tail halfword staging primitive by itself does not recover the
  RADV writeback family. It increases static LDS tail traffic and slightly
  increases VGPR pressure without improving focused timing. The next Q6 p33
  work should stop patching the tail drain around the same live accumulator
  ownership and instead test a lower-live output ownership split or a direct
  port of RADV's direct/staged/tail branch ladder per tile.

## 2026-06-20 - Q6_K VK64 p33 branch-store rejection

- source:
  `sources/llama.cpp` at
  `eff506276 hrx: reject q6 depwait stagefull probe` plus the local
  p33-specialized branch-store diagnostic.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33branch-repro` and the
  matching generated HSACO target.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium narrow prompt schedule probe for Qwen3
  30B Q6_K. The harness intentionally skips the `64x64` row because this
  diagnostic compiles a `cols=33` output branch policy.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_p33branch_f16acc_wg256_f32`.
- baseline command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-repro`
  and
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-stagefull-repro`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-p33branch-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-p33branch-eff506276-dirty-20260620-203715/`.
- correctness result:
  normal p33 focused rows passed with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, and `128x33 k3584`. The stress row remains in
  the known VK64 calibration failure band.
- timing result:
  variant `64x33 k3584` was `404.506 us` and `128x33 k3584` was `414.557 us`.
  Same-run depwait baseline was `404.773 us` and `408.095 us`; stagefull was
  `413.687 us` and `408.657 us`. This is flat at the primary p33 row and
  regressive for `128x33`.
- static evidence:
  the candidate preserves the RADV first-WMMA/load window exactly:
  `59` pre-hot loads, `48` immediate LDS loads, final `lgkmcnt(40)`, and
  `16` WMMA. It also keeps `LDS=11264`, `2` barriers, and no spills. The
  branch specialization compiles away most unused writeback surface:
  `48 buffer_store_b32`, `2 ds_store_b16`, and `VGPR=169`, versus RADV p33
  medium's `96 buffer_store_b32`, `64 ds_store_b16`, and `VGPR=144`.
- decision:
  reject before route selector or model A/B. The experiment proves that
  removing generic full/tail store duplication alone is not the missing
  schedule; it does not reduce VGPR pressure and does not beat the current
  depwait timing. The next Q6 p33 axis should target output value lifetime and
  RADV's staged/tail scalar-store ownership more directly, rather than simply
  deleting statically unused column-tile branches.

## 2026-06-20 - Q6_K VK64 depwait stagefull rejection

- source:
  `sources/llama.cpp` at
  `169147f5e hrx: reject q6 depwait upper-stage probe` plus the local
  per-tile staged-full diagnostic.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-stagefull-repro` and the
  matching generated HSACO target.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium schedule probe for Qwen3 30B Q6_K
  narrow prompt rows.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_stagefull_f16acc_wg256_f32`.
- baseline command:
  prior direct depwait and depwait upper-stage artifacts:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-direct-1d2fabbb1-dirty-20260620-195910/`
  and
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-upper-stage-5684b4705-dirty-20260620-202159/`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-stagefull-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-stagefull-169147f5e-dirty-20260620-202900/`.
- correctness result:
  normal focused rows passed with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, `64x64 k3584`, and `128x33 k3584`.
  The stress row remains in the known VK64 calibration failure band.
- timing result:
  `64x33 k3584` was `412.707 us`, `64x64 k3584` was `453.952 us`, and
  `128x33 k3584` was `412.762 us`, still materially slower than the accepted
  VK64 calibration and flat versus the recent depwait family.
- static evidence:
  this is the first route-facing Q6 p33 diagnostic to compose the preserved
  RADV first-WMMA issue window with low-barrier halfword staged-full output
  traffic without increasing LDS: `LDS=11264`, `2` barriers,
  `64 ds_load_u16_d16`, `66 ds_store_b16`, and no spills. It still misses the
  RADV p33 medium output policy: HIP emits `VGPR=169` and
  `128 buffer_store_b32`, while RADV is `VGPR=144` and
  `96 buffer_store_b32`. The static store surface is duplicated by the generic
  full/tail branch structure rather than matching RADV's eight direct
  full-tile blocks, eight staged full-tile blocks, and partial scalar-store
  alternatives.
- decision:
  reject before route selector or model A/B. The halfword primitive itself is
  no longer the blocker; HIP C++ can force it with the right LDS footprint and
  barrier count. The next Q6 p33 axis should reduce duplicated store/branch
  surface and live output values toward RADV's branch policy, or build a
  p33-specialized branch schedule that separates the first two full column
  tiles from the odd/tail column instead of compiling both full and guarded
  stores for every logical tile.

## 2026-06-20 - Q6_K VK64 depwait upper-stage rejection

- source:
  `sources/llama.cpp` at
  `5684b4705 hrx: reject q6 clause4 depwait probe` plus the local
  depwait+upper-stage diagnostic.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-upper-stage-repro` and the
  matching generated HSACO target.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium schedule probe for Qwen3 30B Q6_K
  narrow prompt rows.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_upper_stage_f16acc_wg256_f32`.
- baseline command:
  prior direct depwait, upper-stage, and clause4 artifacts:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-direct-1d2fabbb1-dirty-20260620-195910/`,
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-upper-stage-repro-8c096ba3d-20260620-180931/`, and
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-clause4-fulltile-wired-ec1656f3a-dirty-20260620-201418/`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-upper-stage-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-upper-stage-5684b4705-dirty-20260620-202159/`.
- correctness result:
  normal focused rows passed with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, `64x64 k3584`, and `128x33 k3584`.
  The stress row remains in the known VK64 calibration failure band.
- timing result:
  `64x33 k3584` was `413.089 us`, `64x64 k3584` was `456.204 us`, and
  `128x33 k3584` was `412.020 us`, still slower than the accepted VK64
  calibration and flat/regressive versus the recent depwait probes.
- static evidence:
  the RADV first-WMMA issue window still composes with the upper-stage drain:
  `59` pre-hot loads, `48` immediate LDS loads, final `lgkmcnt(40)`,
  `16` WMMA, and no spills. The writeback surface remains wrong:
  `VGPR=170`, `LDS=15360`, `4` barriers, `64 buffer_store_b32`,
  `32 ds_load_u16_d16`, and only `2 ds_store_b16`. RADV p33 medium remains
  `VGPR=144`, `LDS=11264`, `2` barriers, `96 buffer_store_b32`,
  `64 ds_store_b16`, and `64 ds_load_u16_d16`.
- decision:
  reject before route selector or model A/B. Combining the preserved RADV
  depwait issue window with the upper-half staged drain does not recover the
  RADV medium output-ownership family. The next Q6 p33 work should stop
  composing these route-facing RING96 drain pieces and instead target the RADV
  per-tile full-staged branch primitive directly: four `ds_store_b16`, four
  `ds_load_u16_d16`, and four scalar global stores per 16x16 tile, with
  low-barrier branch selection rather than a post-hoc staged drain.

## 2026-06-20 - RADV large-route store branch-path audit

- source:
  `sources/llama.cpp` at commit `a4abc2c0f`, dirty after adding
  `tools/vulkan-oracle/analyze_radv_store_branch_paths.py` and source tuning
  evidence.
- build:
  no HIP build; Python oracle-analysis tool only.
- model/shape:
  Qwen3 30B Q6_K p512 Vulkan-large and Llama 3.1 8B Q8_0 p512 Vulkan-large.
- route or kernel candidate:
  no HRX candidate. This identifies the scalar branch ladder selecting RADV's
  direct, staged, and partial/tail writeback paths.
- baseline command:
  `tools/vulkan-oracle/analyze_radv_store_branch_paths.py` over the Q6 and Q8
  RADV ISA dumps from the p512/fa1 Vulkan oracle captures.
- variant command:
  not applicable.
- route trace path:
  not applicable; no HRX selector changed.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/radv-store-branch-paths-20260620-continue/`.
- correctness result:
  not run; no executable HRX candidate was introduced.
- timing result:
  not run.
- static evidence:
  Q6 and Q8 share the same branch ladder. The direct full-tile store block is
  reached by conditional fallthrough from `s_cmp_lg_i32 s19, 0;
  s_cbranch_scc0 BB7`, so the direct path corresponds to `s19 != 0`. The
  staged fallback is the next fallthrough from `s_cmp_lg_i32 s15, 0;
  s_cbranch_scc0 BB9`. The partial/tail entry is guarded by row/column bounds
  checks and then writes through exec-masked scalar-store blocks.
- decision:
  accept as static branch-target evidence. The next full-aligned p512 Q6/Q8
  HIP fixture should target the direct branch predicate and compact `s_clause
  0x3` four-store block, not just the aggregate static store count. p513 tail
  behavior remains a separate gate.
- notes:
  this is the bridge between the store-path count audit and a real C++ fixture:
  the next fixture must prove or replace the direct-branch lane ownership first,
  then add staged/tail behavior only after the full-tile path is
  correctness-clean and schedule-close.

## 2026-06-20 - RADV large-route store path audit

- source:
  `sources/llama.cpp` at commit `299d3d307`, dirty only after adding source
  tuning evidence for this audit.
- build:
  no new HIP build; static RADV ISA classification only.
- model/shape:
  Qwen3 30B Q6_K p512 Vulkan-large and Llama 3.1 8B Q8_0 p512 Vulkan-large.
- route or kernel candidate:
  no HRX candidate. This supersedes the assumption that RADV's static
  `192 buffer_store_b32` count is the full-aligned p512 writeback target.
- baseline command:
  `tools/vulkan-oracle/classify_radv_store_paths.py` over the Q6 and Q8 RADV
  ISA dumps from the p512/fa1 Vulkan oracle captures.
- variant command:
  not applicable.
- route trace path:
  not applicable; no HRX selector changed.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/radv-store-path-audit-20260620-continue/`.
- correctness result:
  not run; no executable HRX candidate was introduced.
- timing result:
  not run.
- static evidence:
  Q6 static totals are `192` buffer stores, `134` LDS stores, `192` LDS loads,
  `32` WMMA, and `2` barriers. Q8 static totals are `192` buffer stores,
  `132` LDS stores, `192` LDS loads, `32` WMMA, and `2` barriers. For both
  quant families, the store blocks classify into `64` direct global-store ops,
  `64` full-tile staged-store ops, and `64` partial/tail scalar-store ops, plus
  separate partial-stage entry blocks.
- decision:
  accept as a static target correction. The next full-aligned p512 Q6/Q8 HIP
  candidate should not chase `192` static stores by default; it should first
  validate and target the active RADV branch, likely the direct `64`-store path
  for full tiles. Staged and partial paths remain required for p513/tail gates.
- notes:
  the corrected Q6 dual-stage rejection is now interpreted as chasing the wrong
  staged writeback family for p512 and paying `34` barriers. The next large
  route work should focus on direct writeback lane ownership, B64 load issue
  window, and resource pressure instead of adding more per-tile staged drains.

## 2026-06-20 - Q8_0 mixed192 stage-one18 reproducer

- source:
  `sources/llama.cpp` at commit `e23846da8`, dirty after adding two
  low-level `coopmat_store_contract` modes.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-coopmat-store-contract`.
- model/shape:
  standalone Q8_0 cooperative-store/lane-ownership fixture for the Llama 3.1
  8B Q8_0 p512/p513 large-route writeback gap.
- route or kernel candidate:
  no route candidate. Added fixture modes `wmma-lds-k2-stage-one18` and
  `wmma-lds-k2-stage-one18-nodirect`.
- baseline command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-direct192-raw --flags=0x31004000`.
- variant command:
  same binary with `--mode=wmma-lds-k2-radv-mixed192`,
  `--mode=wmma-lds-k2-stage-one18`, and
  `--mode=wmma-lds-k2-stage-one18-nodirect`.
- route trace path:
  not applicable; standalone fixture only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-coopstore-stage-one18-e23846da8-20260620-183120/`.
- correctness result:
  `direct192-raw` passed with `bad=0`. Full `radv-mixed192` failed with
  `bad=2112`, first bad `group=18 slot=3 lane=0`. Both new `stage-one18`
  modes passed with `bad=0`. Existing bracket modes show `mixed128` and
  `mixed160` fail, while `mixed160-splitstage` passes.
- static evidence:
  `stage-one18` with direct accumulator groups is no-spill at SGPR `14`,
  VGPR `162`, LDS `24576`, with `32` WMMA, `64 ds_load_b64`,
  `4 ds_load_u16_d16`, `4 ds_store_b16`, and `68 buffer_store_b32`.
  `stage-one18-nodirect` has WMMA optimized dead, so it only validates the raw
  group18 stage mapping.
- decision:
  keep as a compiler/ownership reproducer. A single staged group after live
  WMMA is not the bug; the failing condition needs a wider staged writeback
  surface/order.
- notes:
  do not add another full mixed halfword-stage production wrapper unless it
  changes batching/order or moves below this HIP C++ staging contract.

## 2026-06-20 - Q6_K RADV96 sidecar inline-asm store rejection

- source:
  `sources/llama.cpp` at commit `0386c32fc`, dirty after adding the
  inline-asm sidecar wrapper, CMake kernel source, and HIP bench target.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-radv96-sidecar-inlineasm-repro`.
- model/shape:
  standalone Q6_K VK64/RADV96 p33-medium schedule probe for Qwen3 30B Q6_K
  narrow prompt rows; includes odd/narrow `64x33` and stress coverage.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_radv96_sidecar_inlineasm_f16acc_wg256_f32`.
- baseline command:
  value-barrier/no-merge sidecar comparators from the previous entries, plus
  accepted VK64 calibration.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=1000 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-radv96-sidecar-inlineasm-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-radv96-sidecar-inlineasm-repro-0386c32fc-20260620-182504/`.
- correctness result:
  normal rows `64x33 k256`, `64x33 k512`, `64x33 k3584`,
  `64x64 k3584`, and `128x33 k3584` passed with `bad_gt_0p25=0` and
  `sidecar_written=2048`. The known stress row failed with
  `bad_gt_0p25=2044/2112`, matching accepted VK64 stress behavior.
- timing result:
  `64x33 k3584` was `409.072 us`; `64x64 k3584` was `450.754 us`.
  This remains materially slower than accepted VK64 calibration
  (`309.269 us` and `352.721 us`).
- static evidence:
  inline asm emitted the exact scalar store surface, `96 buffer_store_b32` and
  zero `buffer_store_b128`, with wave64, SGPR `42`, VGPR `153`, LDS `14336`,
  no spills, `16` WMMA, `96 ds_load`, `10 ds_store`, `100 s_waitcnt`, and
  `8` barriers. This only improves the previous exact-store barrier variants
  by four waits; it does not recover the coalesced sidecar's lower
  `68 ds_load`/`76 s_waitcnt` shape or the accepted VK64 pressure profile.
- decision:
  reject for production and skip model A/B. Explicit `buffer_store_b32`
  sequencing alone is not the missing Vulkan schedule.
- notes:
  the next Q6 p33 candidate should change accumulator/store ownership or the
  LDS writeback primitive. More barriers or scalar-store forcing around the
  same sidecar topology is now a closed axis unless it also reduces the
  96-load/8-barrier wait surface.

## 2026-06-19 - Q8_0 K2 direct-wait inline-WMMA compile-contract rejection

- source:
  `sources/llama.cpp` dirty after adding
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_asmwmma_packstage_bufferstore_wg256.hip.cpp`
  and registering it in CMake.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja as a generated
  HSACO target.
- model/shape:
  compile-contract probe for the Llama 3.1 8B Q8_0 p512 Vulkan-large family;
  no model runtime gate because the static gate failed.
- route or kernel candidate:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_asmwmma_packstage_bufferstore_f16acc_wg256_f32`.
- baseline command:
  prior K2 direct-wait compile probe
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_packstage_bufferstore_f16acc_wg256_f32`.
- variant command:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_asmwmma_packstage_bufferstore_wg256.hsaco -j$(nproc)`.
- route trace path:
  not applicable; rejected before runtime selection.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8_0-k2-directwait-asmwmma-compile-20260619-182519/`.
- correctness result:
  not run; static pressure rejected the candidate before p33/p512/p513 gates.
- timing result:
  not run.
- static evidence:
  wave64, SGPR `29`, VGPR `256`, LDS `20480`, private segment `76`,
  `18` VGPR spills, `32` inline `v_wmma_f16_16x16x16_f16`, `64`
  `ds_load_b64`, `128 buffer_store_b32`, `2 ds_store_b32`, and `2` barriers.
  The issue window improves versus the older direct-wait probe: `64`
  pre-WMMA `ds_load_b64` and final pre-WMMA `lgkmcnt(51)`. The older route was
  no-spill at VGPR `196` but collapsed to final `lgkmcnt(0)`.
- decision:
  reject before route promotion or runtime gating. Inline WMMA recovers the
  visible issue-window contract only by reintroducing a spill cliff.
- notes:
  this is a useful bracket: builtin WMMA keeps the direct-wait ABI pressure low
  but loses the RADV issue window; inline WMMA keeps the issue window but spills.
  The next Q8 candidate needs lower live operand/accumulator pressure or a
  different cooperative writeback primitive, not another direct toggle between
  builtin and inline WMMA.

## 2026-06-19 - Q8_0 raw K2 inline-WMMA compile-contract rejection

- source:
  `sources/llama.cpp` dirty after adding
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_raw_asmwmma_packstage_bufferstore_wg256.hip.cpp`
  and registering it in CMake.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja as a generated
  HSACO target.
- model/shape:
  compile-contract probe for the Llama 3.1 8B Q8_0 p512 Vulkan-large family;
  no model runtime gate because the static gate failed.
- route or kernel candidate:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_raw_asmwmma_packstage_bufferstore_f16acc_wg256_f32`.
- baseline command:
  prior raw K2 compile probe
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_raw_packstage_bufferstore_f16acc_wg256_f32`.
- variant command:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_raw_asmwmma_packstage_bufferstore_wg256.hsaco -j$(nproc)`.
- route trace path:
  not applicable; rejected before runtime selection.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8_0-depwait-k2-raw-asmwmma-compile-20260619-182212/`.
- correctness result:
  not run; static pressure rejected the candidate before p33/p512/p513 gates.
- timing result:
  not run.
- static evidence:
  wave64, SGPR `29`, VGPR `256`, LDS `20480`, private segment `172`,
  `42` VGPR spills, `32` inline `v_wmma_f16_16x16x16_f16`, `64`
  `ds_load_b64`, `128 buffer_store_b32`, `2 ds_store_b32`, and `2` barriers.
  The older raw K2 compile probe was already rejected at private segment `124`
  and `30` VGPR spills, so the inline-WMMA spelling worsens the production
  pressure cliff.
- decision:
  reject before route promotion or runtime gating.
- notes:
  standalone raw direct192 inline asm repairs the operand/lane correctness
  contract, but that repair does not transfer through the current source-visible
  K2 depwait production ABI. The next Q8 path should reduce ABI/live-range
  pressure before exposing both K tiles or move to a lower-level cooperative
  matrix/writeback primitive.

## 2026-06-19 - Q8_0 BM128 direct192 inline-asm contract probe

- source:
  `sources/llama.cpp` at commit
  `6a168db0c hrx: add q8 ktilefrag schedule probe`, dirty only for catalog
  evidence metadata after the run.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q8-wmma-repro`.
- model/shape:
  standalone real-Q8 BM128/BN128 contract rows `rows=128`, `k=4096`,
  `cols=128`, plus odd/narrow `cols=33`.
- route or kernel candidate:
  BM128 direct192 one-dispatch contract variants using explicit inline
  `v_wmma_f16_16x16x16_f16` operand spelling:
  `contract-bm128-direct192-raw-asm`,
  `contract-bm128-direct192-bcopy-upper-asm`, and
  `contract-bm128-direct192-abcopy-bhoist-asm`.
- baseline command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-bm128-direct192-raw`,
  with non-asm copy controls in the same artifact.
- variant command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-bm128-direct192-raw-asm`
  plus the two copy/hoist inline-asm modes above.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-bm128-direct192-asm-contract-20260619-181343/`.
- correctness result:
  the non-asm raw control reproduced the known failure on p128 and p33,
  first bad at group `16`, max_abs `7.57994`. Raw inline asm passed p128 and
  p33 with `bad=0`, no NaNs, no infinities, no sentinels, and max_abs
  `0.00268994`/`0.00262882`. Upper-B-copy inline asm also passed tightly.
  A+B-copy/B-hoist inline asm passed only loosely with max_abs `0.109229`.
- timing result:
  not run; this remains a standalone contract diagnostic.
- static evidence:
  raw inline asm compiles to wave64, SGPR `50`, VGPR `195`, LDS `20480`,
  private segment `0`, 32 WMMA, 64 `ds_load_b64`, and 192
  `buffer_store_b32`. The extracted disassembly summary still reports scratch
  mnemonics despite `private_segment=0`, so production promotion needs focused
  HSACO review rather than this summary alone.
- decision:
  accept as a positive standalone diagnostic, not a route promotion. The raw
  direct192 semantics are repaired by inline WMMA operand spelling, not by
  fragment copies.
- notes:
  next Q8 work should port the raw inline-asm operand contract into a
  route-facing production ABI that preserves the required store/lane surface,
  then run p33/p512/p513 CPU-reference, route, static, and same-runner timing
  gates against the current packed-Q8_1 default.

## 2026-06-19 - Q8_0 BM128 upper B-copy contract rejection

- source:
  `sources/llama.cpp` dirty after adding `copy_b_min_col_sub` to the
  standalone BM128 direct192 repro and adding
  `contract-bm128-direct192-bcopy-upper` plus
  `contract-bm128-direct192-bcopy-upper-hoist` modes.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q8-wmma-repro`.
- model/shape:
  standalone real-Q8 BM128/BN128 contract rows `rows=128`, `k=4096`,
  `cols=128`, plus odd/narrow `cols=33`.
- route or kernel candidate:
  one-dispatch direct192 contract with explicit B materialization only for
  upper-column fragments (`col_sub >= 2`), with and without B-copy hoisting.
- baseline command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-bm128-direct192-raw` and
  `--mode contract-bm128-direct192-abcopy-bhoist`.
- variant command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-bm128-direct192-bcopy-upper`
  and `--mode contract-bm128-direct192-bcopy-upper-hoist`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-bm128-upper-bcopy-contract-20260619-141902/`.
- correctness result:
  non-hoisted upper B-copy passed p128 and p33 with `bad=0` but loose max_abs
  `0.109595` and `0.0130464`. Hoisted upper B-copy failed p128 and p33 with
  thousands of bad values and NaNs. Raw and A+B-copy/B-hoist controls reproduced
  the prior rejected/passing results in the same artifact.
- timing result:
  not run; this remains a standalone contract diagnostic.
- static evidence:
  non-hoisted upper B-copy preserves 32 WMMA, 64 `ds_load_b64`, and 192
  `buffer_store_b32`, but compiles to wave64, SGPR `50`, VGPR `256`, LDS
  `20480`, private segment `128`, and `31` VGPR spills. Hoisted upper B-copy
  is wave64, SGPR `50`, VGPR `256`, LDS `20480`, private segment `0`, and no
  spills, but fails correctness. Raw remains VGPR `195` with no spills; full
  A+B-copy/B-hoist remains correct but spills with private segment `80` and
  `19` VGPR spills.
- decision:
  reject both upper-only modes for route promotion. Non-hoisted upper B-copy
  repairs the raw upper-column contract only by worsening the spill cliff;
  hoisted upper B-copy proves the no-spill surface still has the wrong live
  B-fragment/operand placement.
- notes:
  this narrows the next useful Q8 path to a lower-level B operand/lane
  primitive or inline WMMA operand spelling. Another HIP C++ fragment-copy
  placement axis is unlikely to reach the RADV no-spill surface.

## 2026-06-19 - Q8_0 contract192 A+B-copy with B-hoist diagnostic

- source:
  `sources/llama.cpp` dirty after extending
  `q8_contract_direct192_repro_kernel` with a `hoist_b_copy` template axis and
  adding `contract-direct192-bcopy-hoist` plus
  `contract-direct192-abcopy-bhoist` modes.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q8-wmma-repro`.
- model/shape:
  standalone real-Q8 contract rows `rows=64`, `k=4096`, `cols=64` and
  odd/narrow `cols=33`.
- route or kernel candidate:
  direct192 one-dispatch contract with B fragment materialization hoisted once
  per `k_tile/col_sub`; final useful probe is
  `q8_contract_direct192_repro_kernel<true,true,true>`.
- baseline command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-direct192-abcopy`.
- variant command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-direct192-bcopy-hoist` and
  `--mode contract-direct192-abcopy-bhoist`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-contract192-abcopy-bhoist-20260619-140145/`.
- correctness result:
  raw still failed with NaNs. B-copy-hoist passed the harness threshold but had
  loose active-output max_abs `0.109229`. A+B-copy/B-hoist passed both p64 and
  p33 with tight max_abs `0.00302361` and `0.00262882`, comparable to the old
  A+B-copy path.
- timing result:
  not run; this remains a standalone contract diagnostic, not a real catalog
  route.
- static evidence:
  A+B-copy/B-hoist preserves the 32-WMMA, 64 `ds_load_b64`, 192
  `buffer_store_b32`, two-barrier contract. It is wave64, SGPR `62`,
  VGPR `256`, LDS `10240`, private segment `28`, and `6` VGPR spills. This
  improves over old A+B-copy's private segment `60` and `14` spills, and over
  B-copy's private segment `280` and `85` spills. Raw remains resource-clean at
  VGPR `196` and no spills but is semantically wrong.
- decision:
  accept as a positive standalone diagnostic only. Do not promote until the
  spelling is transferred to the real Q8_0 catalog ABI and passes focused
  p33/p512/p513 backend-op correctness, route evidence, and same-runner timing.
- notes:
  the next route-facing candidate should port the A+B-copy/B-hoist ordering,
  then verify whether the production BM128/BN128/four-wave ABI preserves the
  lower spill surface and avoids the previous finite `ERR ~= 0.25` catalog
  failure.

## 2026-06-19 - Q8_0 contract192 copy-axis rejection

- source:
  `sources/llama.cpp` dirty after adding standalone
  `contract-direct192-raw` and `contract-direct192-bcopy` modes beside the
  existing `contract-direct192-abcopy` repro.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q8-wmma-repro`.
- model/shape:
  standalone real-Q8 contract rows `rows=64`, `k=4096`, `cols=64` and
  odd/narrow `cols=33`.
- route or kernel candidate:
  `q8_contract_direct192_repro_kernel<false,false>`,
  `<false,true>`, and `<true,true>`.
- baseline command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-direct192-abcopy`.
- variant command:
  `hrx-hip-bench-q8-wmma-repro --mode contract-direct192-raw` and
  `--mode contract-direct192-bcopy`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-contract192-copy-axis-20260619-135406/`.
- correctness result:
  raw failed both p64 and p33 with `bad=256` and NaNs. B-copy and A+B-copy
  passed both shapes with `bad=0` and max_abs about `0.00269`.
- timing result:
  not run; static resource evidence rejected the candidate before timing.
- static evidence:
  raw is wave64, SGPR `62`, VGPR `196`, LDS `10240`, private segment `0`, no
  spills, `32` WMMA, `64 ds_load_b64`, and `192 buffer_store_b32`. B-copy is
  wave64, SGPR `62`, VGPR `256`, LDS `10240`, private segment `280`, and
  `85` VGPR spills. A+B-copy is wave64, SGPR `62`, VGPR `256`, LDS `10240`,
  private segment `60`, and `14` VGPR spills.
- decision:
  reject B-copy as a production route template. It proves A-copy is not needed
  for standalone correctness, but it worsens the compiler spill cliff compared
  with A+B-copy.
- notes:
  the next Q8 parity work should move away from local fragment-copy pivots and
  toward lower-level cooperative-store/lane ownership or another spelling that
  preserves the raw variant's resource profile while restoring the B-fragment
  correctness contract.

## 2026-06-19 - Q8_0 phase96 backend-op dump/replay

- source:
  `sources/llama.cpp` dirty after adding diagnostic
  `GGML_TEST_BACKEND_OPS_DUMP_DIR` support and `--dump-dir` replay to
  `hrx-hip-bench-q8-wmma-repro`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  Llama 3.1 8B Q8_0 focused p512 `ffn_out`
  `MUL_MAT(q8_0[14336,4096], f32[14336,512])`.
- route or kernel candidate:
  catalog phase96 phase0+phase1 route versus standalone specialized
  BM128 phase96 replay.
- baseline command:
  `test-backend-ops test -b HRX0 -o MUL_MAT --test-file ffn_out-only.txt --output csv`
  with
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PHASE96_ABCOPY_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- variant command:
  `hrx-hip-bench-q8-wmma-repro --mode phase96-bm128-abcopy --dump-dir <dump>`.
- route trace path:
  focused backend-op artifact
  `cache/hrxv1/gfx1151/q8-phase96-backendop-dump-20260619-132943/`.
- profile or timing artifact path:
  replay artifact
  `cache/hrxv1/gfx1151/q8-phase96-backendop-replay-20260619-132954/`.
- correctness result:
  catalog route failed the dumped row with `ERR = 0.250678102`. Replaying the
  exact dumped Q8_0 lhs, F32 rhs, and CPU reference output through the
  standalone BM128 phase96 kernel passed the NMSE threshold with
  `nmse=0.000168671`, no NaNs, no infinities, and no sentinels.
- timing result:
  not run; this was a correctness-contract isolation.
- decision:
  reject the earlier data-distribution hypothesis. The blocker is the
  catalog/generic-kernel phase96 spelling, not model-derived tensors, BM128
  ownership, large K, row tile index, or CPU-reference contract.
- notes:
  next Q8_0 parity work should port the passing specialized BM128 phase96
  spelling into a real catalog provider, or use a lower-level cooperative
  writeback/lane-store primitive that preserves this proven replay contract.

## 2026-06-19 - Q8_0 BM128 phase96 catalog perf rejection

- source:
  `sources/llama.cpp` dirty after adding opt-in BM128 phase96 catalog
  providers:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_bm128phase96_abcopy_bufferstore_phase0_f16acc_wg256_f32`
  and `phase1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, providers compiled through CMake/Ninja.
- model/shape:
  Llama 3.1 8B Q8_0 focused p512, p33, and p513 `MUL_MAT` rows.
- route or kernel candidate:
  two-dispatch specialized BM128 phase96 port of the standalone replay-clean
  repro spelling. Runtime opt-in gate:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_BM128PHASE96_ABCOPY_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <focused-row-file> --output csv`
  without the opt-in gate.
- variant command:
  the same `test-backend-ops perf` command with the BM128 phase96 opt-in gate
  set.
- route trace path:
  `cache/hrxv1/gfx1151/q8-bm128phase96-focused-20260619-133943/`.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-bm128phase96-focused-20260619-133943/perf-compare/`.
- correctness result:
  passed focused CPU-reference gates for p512 and p513 large rows; p33 was
  correctly guarded out and stayed on existing narrow routes. Route traces show
  phase0+phase1 selected only for ffn_out, ffn_gate, and result_output in
  p512/p513, while Vcur/Qcur stayed on packed Q8_1.
- timing result:
  rejected before model tests. p512 total regressed from `72144.483 us` to
  `238515.070 us` (`3.306x`). p513 total regressed from `81933.650 us` to
  `265724.231 us` (`3.243x`). p33 was neutral at `1.010x` because the new route
  did not select.
- static evidence:
  each target phase is wave64, SGPR `50`, VGPR `175`, LDS `20480`,
  private segment `0`, dynamic stack `false`, with `16` WMMA,
  `64 ds_load_b64`, `32 buffer_store_b32`, and two barriers.
- decision:
  keep the route opt-in only as negative evidence. The specialized spelling
  fixes the catalog correctness problem but is far slower than the current
  packed Q8_1 split-qsum baseline, so it is not a promotion candidate.
- notes:
  next Q8 parity work should target RADV cooperative store/lane ownership or a
  lower-level writeback primitive. Another two-dispatch HIP C++ WMMA phase clone
  is unlikely to close the Vulkan gap.

## 2026-06-19 - Q8_0 group12 selected-only stage rejection

- source:
  `sources/llama.cpp` dirty after adding selected-only stage modes to
  `hrx-hip-bench-q8-wmma-repro`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  standalone real-Q8 repro rows `rows=64`, `k=4096`, `cols=64` and `cols=33`.
- route or kernel candidate:
  `single-group8-bcopy-stage-selected`,
  `single-group12-bcopy-stage-selected`, and
  `single-group12-abcopy-stage-selected`.
- baseline command:
  existing split-selected staged modes from
  `cache/hrxv1/gfx1151/q8-wmma-single-group-stage-20260619-110613/`.
- variant command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q8-wmma-repro --mode <selected-stage-mode>`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-wmma-selected-only-stage-20260619-111309/`.
- correctness result:
  group 8 selected-only passed p64 and p33. Group 12 selected-only still
  failed finite p64 correctness: bcopy `bad=16/256`, `max_abs=0.346383`;
  abcopy `bad=16/256`, `max_abs=0.348336`; no NaNs or infinities.
- timing result:
  not run; diagnostic correctness boundary only.
- decision:
  reject the dummy-other-write hypothesis.
- notes:
  selected-symbol facts confirm the intended reduced stage surface: wave64, no
  spills, LDS `10752`, `8` WMMA, `64` B64 LDS reads, `6` halfword LDS stores,
  `4` halfword reloads, `4` buffer stores, and `81` `v_mov_b32` for bcopy
  selected-only. The group12 failure follows the `col_sub=3` B operand,
  accumulator, or store lane contract even when only selected halfwords are
  staged.

## 2026-06-19 - Q8_0 single-group split-selected stage contract

- source:
  `sources/llama.cpp` dirty after adding diagnostic modes to
  `hrx-hip-bench-q8-wmma-repro`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  standalone real-Q8 repro rows `rows=64`, `k=4096`, `cols=64` and `cols=33`.
- route or kernel candidate:
  `single-group0-bcopy-stage`, `single-group8-bcopy-stage`,
  `single-group8-abcopy-stage`, `single-group12-bcopy-stage`, and
  `single-group12-abcopy-stage`.
- baseline command:
  prior passing raw-store fragment-copy modes
  `single-group8-bcopy`, `single-group8-abcopy`, `single-group12-bcopy`, and
  `single-group12-abcopy`.
- variant command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q8-wmma-repro --mode <stage-mode>`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-wmma-single-group-stage-20260619-110613/`.
- correctness result:
  group 0 and group 8 staged bcopy/abcopy modes passed p64 and p33. Group 12
  staged bcopy and abcopy failed finite p64 correctness with `bad=10/256`,
  no NaNs, and `max_abs ~= 0.346-0.348`. Group 12 has no p33 active outputs
  because `cols=33` does not reach `col_sub=3`.
- timing result:
  not run; diagnostic correctness boundary only.
- decision:
  reject split-selected staged writeback as a generally valid production
  transfer until the `col_sub=3` contract is understood.
- notes:
  extracted selected-symbol facts show wave64, no spills, LDS `10752`, `8`
  WMMA, `64` B64 LDS reads, `10` halfword LDS stores, `8` halfword reloads,
  `4` buffer stores, and `81` `v_mov_b32` for both group8 and group12 bcopy
  staged modes. This narrows the remaining Q8 direct-WMMA semantic failure:
  the helper is not universally bad, and group8/`col_sub=2` can pass; the
  isolated group12/`col_sub=3` path still fails.

## 2026-06-19 - Q8_0 mediumfrag12 combined96 catalog transfer rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_mediumfrag12_combined96_packstage_bufferstore_f16acc_wg256_f32`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  Llama 3.1 8B Q8_0 focused p512 prompt rows.
- route or candidate:
  opt-in catalog port of the standalone
  `wmma-issue-window --mode=mediumfrag12-combined96` ladder rung.
- baseline command:
  `test-backend-ops test -b HRX0 -o MUL_MAT --test-file q8_0_prompt.txt --output csv`
  with current default routing.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_MEDIUMFRAG12_COMBINED96_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- route trace path:
  `cache/hrxv1/gfx1151/q8_0-mediumfrag12-combined96-focused-20260619-110113/`;
  control `cache/hrxv1/gfx1151/q8_0-current-control-focused-20260619-110151/`.
- profile or timing artifact path:
  not run; CPU-reference failed before timing.
- correctness result:
  the opt-in route selected for `ffn_out`, `ffn_gate`, and `result_output`
  and failed all three with NaNs. The same rows passed on the current default
  packed-Q8_1 provider.
- timing result:
  not applicable.
- decision:
  reject as a production route and direct transfer template.
- notes:
  static ISA preserved the target probe surface: wave64, 16 WMMA, 48 B64 LDS
  reads, 64 halfword LDS stores, 64 halfword LDS reloads, 96 buffer stores, and
  the intended `lgkmcnt(40)`-class issue window. This is an important negative:
  HIP C++ can preserve this narrower RADV-like schedule in the catalog build,
  but the partial combined96 output/accumulator contract is not correct for
  real model-derived rows. Next Q8 work should target the exact upper-column
  B-fragment and accumulator lane contract, or use a lower-level
  cooperative-matrix/writeback primitive.

## 2026-06-18 - q8_0 VK128 direct-F32 WMMA hybrid

- source:
  `sources/llama.cpp` with
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`,
  guarded by
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  Llama 3.1 8B Q8_0 p512/fa1, plus focused p33/p513 odd/tail rows.
- route or candidate:
  direct-F32 Q8_0 WMMA route anchored to the Vulkan oracle
  `matmul_q8_0_f32_f16acc_aligned_l`: BM128, BN128, BK32, WG256, wave64,
  padded 40-half LDS rows, Q8_0 dequant to f16 LDS, F32 RHS cast to f16 LDS,
  and f16acc WMMA.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file q8_0_prompt_all.txt --output csv`
  with current default Q8_0 routing.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-wmma16-vk128-padded-w64-focused-20260618-002300/`.
- profile/timing:
  `cache/hrxv1/gfx1151/q8_0-wmma16-vk128-hybrid-model-ab-20260618-002821/`.
- correctness:
  CPU-reference p512, p33, and p513 gates passed. The final selector keeps
  Vcur/Qcur and odd p33/p513 on packed Q8_1, and selects VK128 only for
  `cols >= 128 && (rows >= 8192 || k >= 8192)`.
- timing:
  focused p512 summed Q8_0 row time improved
  `267564.84 -> 119227.42 us`. Same-binary HRX model A/B improved
  `182.704 -> 280.436 tok/s`. Same-machine Vulkan measured `903.038 tok/s`,
  so the hybrid is about `0.31x` Vulkan on this row.
- decision:
  superseded by current-best comparison below. This route improved over the
  old/default Q8 path but was not compared against the stronger BN96/BN64
  current-best Q8 policy in this entry.
- notes:
  The first broad probe showed direct VK128 regressed Vcur/Qcur, so the hybrid
  selector is required.

## 2026-06-18 - q8_0 VK128 direct-F32 WMMA versus current-best BN96

- source:
  `sources/llama.cpp` at `7ba57d8ca` plus catalog evidence correction.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Llama 3.1 8B Q8_0 focused p512 rows and eight-model p512/fa1 basket.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`
  compared against the accepted Q8_0 BN96/BN64 packed-Q8_1 policy.
- baseline command:
  focused `test-backend-ops perf` with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1` and
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- variant command:
  same plus
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-vk128-vs-bn96-focused-20260618-003553/`.
- profile/timing:
  focused artifact above and basket artifact
  `cache/hrxv1/gfx1151/current-best-q8vk128hybrid-basket-p512-fa1-r1-20260618-003443/`.
- correctness:
  focused p512 and odd p33/p513 CPU-reference gates passed.
- timing:
  focused p512 rows all regressed versus BN96/BN64:
  `Vcur 651.63 -> 682.40 us`, `Qcur 2026.25 -> 2069.85 us`,
  `ffn_out 8269.05 -> 9696.54 us`, `ffn_gate 7371.01 -> 10308.36 us`,
  and `result_output 66339.24 -> 92056.48 us`. Basket geomean regressed
  `0.5354x -> 0.5212x` Vulkan, and the Q8_0 row regressed
  `422.15 -> 356.70 tok/s`.
- decision:
  reject for current-best promotion. Keep as an opt-in diagnostic only.
- notes:
  This is the useful correction from the mechanical Vulkan clone: direct-F32
  VK128 wave64 WMMA is closer to RADV by opcode/tile, but the accepted packed
  Q8_1 BN96/BN64 policy is still the stronger production Q8 path. The next
  dense-row target should return to the worst current basket row, DeepSeek
  Q4_K, and mine the remaining packed-path schedule deltas instead of using
  aggregate Q8 gains from a weaker baseline.

## Entry Template

```markdown
## YYYY-MM-DD - short-title

- source:
- build:
- model/shape:
- route or candidate:
- baseline command:
- variant command:
- route trace:
- profile/timing:
- correctness:
- timing:
- decision:
- notes:
```

## 2026-06-19 - Q8_0 LDS K2 live-WMMA mixed-store contract rejection

- source:
  `sources/llama.cpp` at `ab41b8701-dirty`, adding explicit
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-radv-mixed192`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  standalone gfx1151 cooperative-matrix store contract fixture, targeting the
  Llama 3.1 8B Q8_0 p512 RADV large prompt-matmul topology.
- route or candidate:
  two LDS-loaded WMMA phases accumulated into live groups `0..15`, direct raw
  buffer stores for those groups, and synthetic halfword LDS stage/load/store
  controls for groups `16..47`.
- baseline command:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-radv-mixed192` and
  `--mode=wmma-lds-radv-mixed192`.
- variant command:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-radv-mixed192`.
- route trace:
  not applicable; this is a standalone CMake-built HIP bench diagnostic.
- profile/timing:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-live-wmma-mixed-probe-20260619-080339/`.
- correctness:
  controls passed twice with `bad=0`. The K2 mode failed twice with
  `bad=2112 max_abs=50625 first_bad=4800 actual=2048 expected=12803`.
- static evidence:
  `compare-k2.md` shows the fixture reaches the target opcode counts:
  wave64, SGPR `14`, VGPR `162`, LDS `32768`, no spills, `32` WMMA,
  `64 ds_load_b64`, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  `192 buffer_store_b32`, three barriers, and `135 s_waitcnt`. The current
  RADV Q8_0 p512 large oracle is SGPR `108`, VGPR `192`, LDS `22528`, no
  spills, the same `32/64/128/128/192` WMMA/LDS/store opcode counts, two
  barriers, and `169 s_waitcnt`.
- decision:
  reject. Matching the RADV opcode-count surface from HIP C++ is insufficient;
  the remaining issue is the exact cooperative-matrix lane ownership/topology.
- notes:
  This closes the cheap "just make HIP emit the same counts" path for Q8_0.
  The next Q8_0 candidate should use a lower-level primitive or a different
  source form that proves the lane contract in the fixture before touching a
  production route.

## 2026-06-19 - Q8_0 LDS K2 direct-only split

- source:
  `sources/llama.cpp` after
  `58394a965 hrx: test q8 k2 mixed-store fixture`, with dirty source adding
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-direct64`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  standalone gfx1151 cooperative-matrix store contract fixture, targeting the
  direct-store subset of the Llama 3.1 8B Q8_0 p512 RADV large topology.
- route or candidate:
  two LDS-loaded WMMA phases accumulated into live groups `0..15`, then direct
  raw buffer-stored without any synthetic halfword LDS stage groups.
- baseline command:
  same binary with `--mode=wmma-lds-k2-radv-mixed192`, which remains the known
  failing mixed halfword-stage contract.
- variant command:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-direct64`.
- route trace:
  not applicable; standalone CMake-built HIP bench diagnostic.
- profile/timing:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-direct64-probe-20260619-081232/`.
- correctness:
  `wmma-lds-k2-direct64` passed with `bad=0`; the same binary still failed
  `wmma-lds-k2-radv-mixed192` at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`.
- static evidence:
  `compare-direct64.md` reports wave64, SGPR `14`, VGPR `153`, LDS `16384`,
  no spills, `32` WMMA, `64 ds_load_b64`, `64 buffer_store_b32`, two
  barriers, and no halfword LDS load/store path.
- decision:
  accept as a diagnostic primitive, not a production route. K2 live
  accumulator math plus direct raw buffer-store is correct in isolation.
- notes:
  The next probe should isolate the halfword-stage corruption in the mixed
  surface or replace that stage with a lower-level cooperative-matrix store
  primitive. Retesting K2 direct stores alone is now low value.

## 2026-06-19 - Q8_0 LDS K2 stage-first mixed-store rejection

- source:
  `sources/llama.cpp` after
  `6811fbbc0 hrx: split q8 k2 direct-store fixture`, with dirty source adding
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-stagefirst-mixed192`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  standalone gfx1151 cooperative-matrix store contract fixture, targeting the
  full Q8_0 p512 RADV opcode-count surface.
- route or candidate:
  synthetic halfword LDS stage stores issued before two LDS-loaded WMMA phases
  and live direct raw buffer stores, with synthetic stage load/store after the
  barrier.
- baseline command:
  same binary with `--mode=wmma-lds-k2-direct64` and
  `--mode=wmma-lds-k2-radv-mixed192`.
- variant command:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-stagefirst-mixed192`.
- route trace:
  not applicable; standalone CMake-built HIP bench diagnostic.
- profile/timing:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-stagefirst-mixed-probe-20260619-081501/`.
- correctness:
  direct-only K2 passed with `bad=0`; both stage-first mixed and original mixed
  failed at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`.
- static evidence:
  `compare-stagefirst.md` reports wave64, SGPR `14`, VGPR `162`, LDS `32768`,
  no spills, `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `192 buffer_store_b32`, three barriers, and
  `135 s_waitcnt`.
- decision:
  reject. The failure is not fixed by issuing the synthetic halfword stage
  before the K2 WMMA/direct-store work.
- notes:
  Continue by reducing the mixed halfword surface to identify the corruption
  threshold, or move below this HIP C++ halfword-stage spelling.

## 2026-06-19 - Q8_0 LDS K2 mixed-stage threshold

- source:
  `sources/llama.cpp` after
  `3efe1336b hrx: isolate q8 k2 mixed-store ordering`, with dirty source adding
  `wmma-lds-k2-mixed96`, `wmma-lds-k2-mixed128`,
  `wmma-lds-k2-mixed160-lo`, and `wmma-lds-k2-mixed160-hi`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  standalone gfx1151 cooperative-matrix store contract fixture, bracketing the
  halfword-stage surface needed by the Q8_0 p512 RADV large topology.
- route or candidate:
  two LDS-loaded WMMA phases accumulated into direct groups `0..15`, plus
  increasing synthetic halfword LDS stage/load/store groups.
- baseline command:
  same binary with `--mode=wmma-lds-k2-direct64`,
  `--mode=wmma-lds-k2-mixed128`, and
  `--mode=wmma-lds-k2-radv-mixed192`.
- variant command:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-mixed96`,
  `--mode=wmma-lds-k2-mixed160-lo`, and
  `--mode=wmma-lds-k2-mixed160-hi`.
- route trace:
  not applicable; standalone CMake-built HIP bench diagnostic.
- profile/timing:
  first sweep:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-mixed-threshold-probe-20260619-081846/`;
  second sweep and ISA:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-mixed-threshold2-probe-20260619-082030/`.
- correctness:
  `direct64`, `mixed96`, and `mixed128` passed with `bad=0`.
  `mixed160-lo`, `mixed160-hi`, and `mixed192` failed at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`.
- static evidence:
  passing `mixed128` emitted wave64, SGPR `14`, VGPR `162`, no spills,
  `32` WMMA, `64 ds_load_b64`, `64 ds_store_b16`, `64 ds_load_u16_d16`,
  and `128 buffer_store_b32`. Failing `mixed160-lo` kept the same WMMA and
  fragment-load shape but expanded to `96 ds_store_b16`,
  `96 ds_load_u16_d16`, and `160 buffer_store_b32`.
- decision:
  accept `mixed128` as the last correctness-clean diagnostic surface; reject
  the `mixed160` expansion axis.
- notes:
  The failure is triggered by adding any extra staged block beyond `16..31`,
  not by the specific `32..39` versus `40..47` address slice. This is now
  strong evidence to stop trying to reach the RADV `192` store surface through
  this scalarized HIP C++ halfword-stage expansion.

## 2026-06-19 - Q8_0 LDS K2 mixed160 tight/raw controls

- source:
  `sources/llama.cpp` after
  `cd59c1f67 hrx: bracket q8 k2 mixed-stage threshold`, with dirty source
  adding `wmma-lds-k2-mixed160-lo-tight`,
  `wmma-lds-k2-mixed160-hi-tight`, and `wmma-lds-k2-direct160-raw`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  standalone gfx1151 cooperative-matrix store contract fixture.
- route or candidate:
  K2 live-WMMA direct stores plus either raw synthetic direct stores or a
  tight 24-group halfword LDS stage allocation for 160-store surfaces.
- baseline command:
  `--mode=wmma-lds-k2-mixed128`,
  `--mode=wmma-lds-k2-mixed160-lo`, and
  `--mode=wmma-lds-k2-mixed160-hi`.
- variant command:
  `--mode=wmma-lds-k2-direct160-raw`,
  `--mode=wmma-lds-k2-mixed160-lo-tight`, and
  `--mode=wmma-lds-k2-mixed160-hi-tight`.
- route trace:
  not applicable; standalone CMake-built HIP bench diagnostic.
- profile/timing:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-mixed160-tight-raw-control-20260619-082600/`.
- correctness:
  `direct160-raw` passed with `bad=0`; both tight mixed160 modes failed at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`.
  `mixed128` passed repeatedly after rerun.
- static evidence:
  `direct160-raw` emitted `160 buffer_store_b32`, no halfword LDS stage,
  SGPR `14`, VGPR `153`, no spills. The tight mixed160 modes emitted
  `160 buffer_store_b32`, `96 ds_store_b16`, `96 ds_load_u16_d16`, SGPR `14`,
  VGPR `162`, no spills, and `group_segment_fixed_size=28672`.
- decision:
  reject the LDS-allocation-size and raw-store-count explanations. The failure
  belongs to the expanded scalarized halfword LDS stage in the K2 mixed
  context.
- notes:
  The remaining route-relevant conclusion is stronger: do not try to clone
  RADV's full `192` store surface by adding more HIP C++ halfword-stage groups.
  Use a smaller compiler reproducer or a lower-level cooperative-store
  primitive instead.

## 2026-06-19 - Q8_0 LDS K2 mixed128 padded32 control

- source:
  `sources/llama.cpp` after
  `134a4d304 hrx: isolate q8 mixed160 halfword stage`, with dirty source
  adding `wmma-lds-k2-mixed128-padded32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  standalone gfx1151 cooperative-matrix store contract fixture.
- route or candidate:
  same operations as passing `mixed128`, but with a 32-group stage allocation
  matching the full mixed192 footprint.
- baseline command:
  `--mode=wmma-lds-k2-mixed128`,
  `--mode=wmma-lds-k2-mixed160-lo-tight`, and
  `--mode=wmma-lds-k2-direct160-raw`.
- variant command:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-mixed128-padded32`.
- route trace:
  not applicable; standalone CMake-built HIP bench diagnostic.
- profile/timing:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-mixed128-padded32-probe-20260619-083017/`.
- correctness:
  `mixed128` passed, `mixed128-padded32` passed, `mixed160-lo-tight` failed,
  and `direct160-raw` passed.
- static evidence:
  `mixed128-padded32` emitted `group_segment_fixed_size=32768`, SGPR `14`,
  VGPR `162`, no spills, `32` WMMA, `64 ds_load_b64`, `64 ds_store_b16`,
  `64 ds_load_u16_d16`, and `128 buffer_store_b32`.
- decision:
  reject total LDS footprint as the trigger. The first failing boundary remains
  halfword-stage expansion from `64` to `96` store/load pairs.
- notes:
  This closes the cheap LDS-padding/occupancy explanation. The remaining useful
  work is either a smaller compiler reproducer for the halfword-stage expansion
  or a lower-level cooperative-store path.

## 2026-06-18 - Q4_K MoE wide-K ID default promotion

- source:
  `sources/llama.cpp` at `07167d398` for the focused and model A/B evidence;
  follow-up source change defaults
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32` unless
  `GGML_HRX_DISABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; the route is built through CMake/Ninja in
  `mul_mat_id_q4_k_q8_1_x4_mmq.hsaco`.
- model/shape:
  Qwen3 30B Q4_K_XL and Qwen3-Coder 30B Q4_K_M MoE `MUL_MAT_ID` p33, p512,
  and p513 prefill rows.
- route or candidate:
  wide-K grouped Q8_1 x4 Q4_K MoE ID route, guarded to `k % 256 == 0`,
  `rows % 64 == 0`, `n_ids == 8`, and `n_tokens >= 32`.
- baseline command:
  `tools/hrxv1_basket_benchmark.py --models <qwen3-q4xl,qwen3-coder-q4km>
  --cases p33,p512,p513 --backends hrx --repetitions 3 --flash-attn 1`.
- variant command:
  same command with
  `GGML_HRX_ENABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`; after promotion,
  rollback is
  `GGML_HRX_DISABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-id-widek-current-regate-20260618-204002/` and
  `cache/hrxv1/gfx1151/q4-id-widek-current-optin-20260618-204241/`.
  Post-edit default and rollback smoke:
  `cache/hrxv1/gfx1151/q4-id-widek-default-postedit-regate-20260618-204720/`.
- profile/timing:
  focused perf CSVs under the regate artifact plus model artifacts
  `cache/hrxv1/gfx1151/q4-id-widek-current-default-20260618-204154/`,
  `cache/hrxv1/gfx1151/q4-id-widek-current-optin-20260618-204241/`, and
  `cache/hrxv1/gfx1151/q4-id-widek-current-vulkan-20260618-204346/`.
  Post-promotion full basket:
  `cache/hrxv1/gfx1151/basket-after-q4-id-widek-default-commitaligned-20260618-205025/`.
- correctness:
  exact p33, p512, and p513 CPU-reference gates passed for the two Q4 rows in
  each export. The mixed Q5 row remains unsupported by this Q4-specific route.
  Route assertion selected
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32` for all supported
  rows.
- timing:
  focused Q4 MoE rows measured about `261/245 us` at p33,
  `2061/1976 us` at p512, and `2073/2083 us` at p513. Same-runner model A/B
  lifted Qwen3 Q4_XL steady tok/s from `97.338 -> 155.184` on p33,
  `289.451 -> 630.757` on p512, and `287.163 -> 632.876` on p513. Qwen3-Coder
  lifted `102.083 -> 152.696`, `403.337 -> 876.437`, and `396.676 -> 853.230`.
  The subset geomean improved `1.941x` over default and moved from `0.338x`
  to `0.657x` Vulkan. The post-promotion full basket improved from the prior
  post-Q6 checkpoint `0.478x` steady Vulkan geomean to `0.573x`; all 24 rows
  remain below parity.
- decision:
  promote as gfx1151 default with rollback env. This is not parity yet, but it
  removes the worst Q4 MoE default-policy miss and keeps exact odd/tail
  coverage.
- notes:
  The Vulkan prior is
  `matmul_id_subgroup_q4_k_f32_f16acc_aligned_m` from the Qwen3/Qwen3-Coder
  p512 oracle captures: `spec=[128,64,64,32,64,32,2,16,16,16,64]`,
  `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills. The HRX route is
  still structurally lighter (`LDS=3264`, `VGPR=107`, wave64), so the next Q4
  MoE work should target the remaining medium-subgroup dataflow/resource delta
  instead of treating the 0.657x Vulkan result as done.

## 2026-06-18 - Q5_K VK128 B64GROUP store-stage rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_store_stage_f16acc_wg256_f32`
  as an opt-in CMake-built HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; full CMake/Ninja build completed and
  generated the new HSACO.
- model/shape:
  focused Qwen2.5 Coder 7B Q5_K_M-derived p33, p512, and p513 prompt rows.
- route or candidate:
  direct-F32 VK128 padded wave64 WMMA route combining grouped `ds_load_b64`
  fragment reads with a half accumulator LDS store-stage.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file
  q5_prompt_p{512,513}.txt --output csv` with current Q5 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_STAGE_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-store-stage-focused-20260618-110958/`.
- profile/timing:
  same focused artifact; ISA/resource files are `llvm-readobj-notes.txt`,
  `isa.amdgcn.txt`, and `isa-counts.txt`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on rows2; p512
  and p513 selected the new route.
- timing:
  rejected versus current packed-Q8_1 routing. p512 rows regressed:
  Kcur `865.46 -> 1060.91 us`, Qcur `1224.57 -> 4819.52 us`, ffn_out
  `7227.94 -> 19946.42 us`, ffn_gate `6906.12 -> 30089.16 us`. p513 rows
  regressed: Kcur `887.97 -> 1018.55 us`, Qcur `1577.88 -> 5254.41 us`,
  ffn_out `9533.36 -> 23344.87 us`, ffn_gate `9214.98 -> 33125.13 us`.
- decision:
  reject production promotion.
- notes:
  This is useful negative evidence. The route matches RADV's `22528` byte LDS
  footprint, `32` WMMA count, and `64 ds_load_b64`, but emits `34` barriers
  and only manual scalar stores. The next exact schedule attempt needs a
  lower-level cooperative-matrix store/lane-ownership spelling that reaches
  RADV's low-barrier `192 buffer_store_b32` writeback, not another manual HIP
  C++ LDS store-stage.

## 2026-06-18 - Q5_K VK128 B64GROUP fullstore rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_fullstore_f16acc_wg256_f32`
  as an opt-in CMake-built HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; full CMake/Ninja build completed and
  generated the new HSACO.
- model/shape:
  focused Qwen2.5 Coder 7B Q5_K_M-derived p33, p512, and p513 prompt rows.
- route or candidate:
  direct-F32 VK128 padded wave64 WMMA route combining grouped `ds_load_b64`
  fragment reads with low-barrier full in-bounds scalar direct stores.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file
  q5_prompt_p{512,513}.txt --output csv` with current Q5 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-fullstore-focused-20260618-111743/`.
- profile/timing:
  same focused artifact; ISA/resource files are `llvm-readobj-notes.txt`,
  `isa.amdgcn.txt`, and `isa-counts.txt`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on rows2; p512
  and p513 selected the new route.
- timing:
  rejected versus current packed-Q8_1 routing. p512 rows regressed:
  Kcur `867.19 -> 1058.19 us`, Qcur `1230.26 -> 4903.45 us`, ffn_out
  `7238.35 -> 19727.06 us`, ffn_gate `6733.11 -> 29961.79 us`. p513 rows
  regressed: Kcur `898.34 -> 1013.67 us`, Qcur `1587.39 -> 5277.83 us`,
  ffn_out `9597.82 -> 23037.40 us`, ffn_gate `9032.34 -> 32848.22 us`.
- decision:
  reject production promotion.
- notes:
  This route emits the desired low-barrier shape for the tested axes:
  `32` WMMA, `64 ds_load_b64`, `128 global_store_b32`, `2` barriers, and no
  spills. Since it still regresses badly, the direct-F32 clone needs the actual
  cooperative-matrix store/lane ownership and RADV's `192 buffer_store_b32`
  writeback, not scalar full stores.

## 2026-06-18 - Q6_K ID threshold-32 p33 policy rejection

- source:
  `sources/llama.cpp` clean after commit
  `a5c52c84e hrx: record q6 id p33 policy rejection`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; `llama-bench` and `test-backend-ops`
  rebuilt successfully after the catalog evidence commit.
- model/shape:
  Qwen3 30B Q6_K p33, p512, and focused Q6_K `MUL_MAT_ID` exported rows.
- route or candidate:
  diagnostic lowering of the grouped Q6_K `MUL_MAT_ID` threshold from
  `n_tokens >= 128` to `n_tokens >= 32` for
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- baseline command:
  same-runner HRX p33 with the accepted production-width Q6 ID policy and the
  grouped route disabled for p33.
- variant command:
  same-runner HRX p33 with
  `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1` and the threshold-32
  diagnostic source change.
- route trace:
  `cache/hrxv1/gfx1151/q6-id-threshold32-experiment-20260618-182836/`.
- profile/timing:
  same artifact; Vulkan p33 reference:
  `cache/hrxv1/gfx1151/qwen3-q6-p33-clean-vulkan-20260618-182942/`.
- correctness:
  focused p33 and p512 Q6 ID rows passed and selected the grouped route where
  expected; p1 residual rows remained unsupported.
- timing:
  threshold-32 fixed the catastrophic scheduler split cliff
  (`~15 tok/s -> 105.18 tok/s`) by allowing HRX-resident expert weights to be
  consumed on HRX. It still lost to the accepted narrow policy on p33
  (`105.18 tok/s` versus `112.94 tok/s`) and remained far below Vulkan
  (`182.23 tok/s`). p512 stayed strongly positive with Q6 ID enabled
  (`581.74 tok/s` versus `232.48 tok/s` disabled).
- decision:
  reject threshold-32 as a default policy and keep Q6 ID guarded to
  production-width rows. The source-controlled rejection is now recorded in
  `ggml/src/ggml-hrx/catalog/tuning/gfx1151/rejections.json` and
  `ggml/src/ggml-hrx/catalog/tuning/gfx1151/results/index.json`.
- notes:
  This is useful scheduler evidence, not a route promotion. The p33 fix should
  be either a true Vulkan-medium Q6 ID schedule or a scheduler/placement change
  that prevents expert weights from moving to HRX for CPU-only p33 rows.

## 2026-06-18 - next Q8_0 parity gate

- source:
  `sources/llama.cpp` clean at `a5c52c84e`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Llama 3.1 8B Q8_0 p512/p513/fa1; p33 remains an odd/narrow guard row.
- route or candidate:
  next candidate must start from the accepted packed-Q8_1 default
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` or a lower-level
  cooperative-store probe. Do not start from another direct-F32 VK128 HIP C++
  scalar-store clone unless the compile contract explains how it will reach
  RADV's aligned cooperative-matrix store/lane mapping.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT` on exported Q8_0 p33,
  p512, and p513 rows with current default routing.
- variant command:
  same focused rows with a single opt-in route gate, followed by same-binary
  `llama-bench` only if focused p512 hot rows win.
- route trace:
  required `GGML_HRX_TRACE_ROUTES=1` artifact proving intended route selection
  and p33 fallback behavior.
- profile/timing:
  required focused timing artifact plus HSACO metadata/ISA comparison against
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/`.
- correctness:
  required CPU-reference focused gates for p33, p512, and p513 before any
  model A/B.
- timing:
  promotion must beat current Q8_0 default focused p512 hot rows and close the
  current clean model gap (`333.33 tok/s` HRX versus `423.31 tok/s` Vulkan in
  `cache/hrxv1/gfx1151/current-scoreboard-after-q5tail-20260618-151637/`).
- decision:
  superseded by the clean source-HEAD serial recheck on the accepted packed
  Q8_0 routes. The old model gap in this row came from stale scoreboard
  evidence and should not drive new Q8_0 work unless a future full-basket
  rerun reopens it.
- notes:
  Current direct-WMMA evidence says HIP C++ source-visible clones can match
  isolated RADV facts but still miss the schedule that matters: low-barrier
  cooperative-matrix load/store lane ownership, `192 buffer_store_b32`, and the
  RADV first-WMMA issue window without a VGPR cliff. The next productive probe
  should either import one of those facts into the packed-Q8_1 route without
  reintroducing spills, or move the cooperative-store experiment below ordinary
  HIP C++ source spelling.

## 2026-06-18 - rocWMMA scratch probe for Vulkan direct-WMMA store parity

- source:
  `sources/llama.cpp` clean at `38b5d0177` before adding a source-controlled
  rejection note; no production kernel source was changed.
- build:
  scratch compile with active `/srv/vm-shared/rocm/rocm-head` compiler and
  rocWMMA headers from `/srv/vm-shared/rocm/rocm-7.14.0a20260610/include`.
- model/shape:
  no model run; this was a compiler/ISA probe for the remaining Q8_0/Q6_K
  direct-WMMA Vulkan schedule delta.
- route or candidate:
  minimal rocWMMA default accumulator store and f16 WMMA fixture, plus a failed
  cooperative accumulator-fragment compile attempt.
- baseline command:
  compare against existing HRX direct-WMMA helpers and RADV oracle ISA ledgers.
- variant command:
  scratch `amdclang++ -x hip --offload-arch=gfx1151 -O3 -std=c++17
  -fno-gpu-rdc -I/srv/vm-shared/rocm/rocm-7.14.0a20260610/include`.
- route trace:
  not applicable.
- profile/timing:
  `cache/hrxv1/gfx1151/rocwmma-probe-20260618/`.
- correctness:
  not run; no production route was implemented.
- timing:
  not run.
- decision:
  do not implement a rocWMMA production route from this probe. The headers can
  compile against `rocm-head`, but the default accumulator store is wave32 and
  matches the existing HRX wave32 lane map; cooperative accumulator fragments
  failed at template compile time.
- notes:
  Extracted ISA emitted `v_wmma_f32_16x16x16_f16`, `global_load_b128`, and
  eight `global_store_b32` per 16x16 accumulator wave. This does not provide
  the missing large aligned RADV cooperative-matrix global store/lane-ownership
  lowering for the 128x128 family.

## 2026-06-18 - q4 mmql128 b-oct compile gate rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_boct_wg256_f32` as an opt-in
  diagnostic route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; compiled through CMake/Ninja.
- model/shape:
  compile-gate only for Q4_K dense prompt full tiles; no runtime model row
  executed because compile evidence failed the no-spill gate.
- route or candidate:
  B-oct packed-Q8_1/x4 MMQL128 probe. It preserves the accepted B-quad
  production dataflow and preloads all eight WNITER B-cache positions before
  dot consumption.
- baseline command:
  CMake/Ninja build plus HSACO metadata extraction for accepted B-quad prior.
- variant command:
  `ninja -C build/hrx-v1-catalog-gfx1151 -j$(nproc)` after adding the B-oct
  source/catalog route.
- route trace:
  not run; rejected at compile gate.
- profile/timing:
  `cache/hrxv1/gfx1151/q4-mmql128-boct-compile-20260618/`.
- correctness:
  not run; compile evidence failed the no-spill production acceptance row.
- timing:
  not run. HSACO metadata: wave64, SGPR `53`, VGPR `192`,
  `vgpr_spill_count=94`, private segment `376`, LDS `8192`.
- decision:
  rejected at compile gate. Keep accepted B-quad as the current no-spill Q4_K
  packed-path default.
- notes:
  This is still schedule-led: it brackets the positive B-pair/B-quad
  B-cache-read clustering axis. The result says the full B window is too much
  live state on gfx1151, so the next packed-path attempt should alter wait or
  issue order with B-quad-level live state rather than holding all B-cache rows.

## 2026-06-18 - q4 mmql128 b-quad-cr focused rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_cr_wg256_f32` as an opt-in
  diagnostic route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; compiled through CMake/Ninja.
- model/shape:
  DeepSeek-derived Q4_K focused p512 rows plus exact p33 and p513 correctness
  rows from `q4-wmma16-wg256-focused-20260617-223548`.
- route or candidate:
  B-quad-CR packed-Q8_1/x4 MMQL128 probe. It keeps B-quad's two-WNITER
  B-cache cluster and changes only dot consume order so each A micro-row drains
  the loaded B cluster.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 test-backend-ops perf -b HRX0 -o MUL_MAT --test-file q4_prompt_p512.txt --output csv`
  with current default B-quad routing.
- variant command:
  same focused p512 perf command with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_CR_PROMPT=1`; correctness used
  the same env on p33, p512, and p513.
- route trace:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-focused-20260618/`.
- profile/timing:
  same artifact, plus compile metadata in
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-compile-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p512 selected B-quad-CR,
  p33 stayed narrow, and p513 stayed on accepted B-quad tail.
- timing:
  p512 Kcur `982.78 -> 962.51 us` and Qcur `3904.55 -> 3893.61 us` improved
  slightly, but hot FFN rows regressed: ffn_out `13024.67 -> 14032.85 us` and
  ffn_gate `11422.31 -> 12004.63 us`.
- decision:
  rejected for production. Keep accepted B-quad as the current no-spill Q4_K
  packed-path default.
- notes:
  Compile facts were good: wave64, SGPR `53`, VGPR `167`, no spills, LDS
  `8192`. The schedule signal is row-family specific; this consume order is
  not a model-level default candidate unless future policy splits K/Q from FFN.

## 2026-06-18 - qwen2.5 coder 7b q5km vulkan oracle odd tail rows

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, clean at source checkpoint
  `bd0b591ed`; workspace root dirty only with documentation and cache
  artifacts.
- build:
  `build/vulkan-gfx1151`, Release, Vulkan backend, RADV STRIX_HALO;
  workspace ROCm symlink points to `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/Qwen__Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q5_k_m.gguf`,
  `p33/n0/fa1` and `p513/n0/fa1`, `b=1024`, `ub=1024`, one no-warmup
  repetition.
- route or candidate:
  Vulkan oracle prior for Q5_K odd/tail prompt matmul; not an HRX route
  promotion.
- baseline command:
  `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench build/vulkan-gfx1151/bin/llama-bench --model <model> --out-dir <out> --prompt <33-or-513> --gen 0 --batch 1024 --ubatch 1024 --flash-attn 1 --repetitions 1 --device Vulkan0`.
- variant command:
  not applicable; this is a Vulkan prior capture.
- route trace:
  - `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-20260618-063510/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p513-fa1-20260618-063522/vulkan.jsonl`
- profile/timing:
  each artifact contains `stdout.json`, `spv/`, `spvasm/`, `radv/`, and
  `inventory/kernel_inventory.md`.
- correctness:
  both benchmark rows completed and reported `backends=Vulkan`; these are
  schedule-oracle captures, not CPU-reference HRX correctness gates.
- timing:
  one-sample oracle captures only: p33 `35.285692 tok/s`, p513
  `342.041341 tok/s`.
- decision:
  accepted as Q5_K odd/tail oracle evidence. p33 uses the medium aligned
  `matmul_q5_k_f32_f16acc_aligned_m` route; p513 uses the large aligned
  `matmul_q5_k_f32_f16acc_aligned_l` route with fifth-column workgroups and
  `split_k_reduce` tail reductions.
- notes:
  - p33: `spec=[128,64,64,32,64,32,2,16,16,16,64]`,
    `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills, `16` WMMA,
    `48` `ds_load_b64`, `64` `ds_load_u16_d16`, `64` `ds_store_b16`, `96`
    `buffer_store_b32`, `2` barriers.
  - p513: `spec=[256,128,128,32,64,64,2,16,16,16,64]`,
    `wg_denoms=[128,128,1]`, `LDS=22528`, `VGPR=192`, no spills, `32`
    WMMA, `64` `ds_load_b64`, `128` `ds_load_u16_d16`, `128`
    `ds_store_b16`, `192` `buffer_store_b32`, `2` barriers.
  - The repeated Q4/Q5/Q6/Q8 pattern is now confirmed: odd/narrow p33 and
    production-width p512/p513 need separate route policies, and p513 can need
    an explicit split-K reduction path.

## 2026-06-18 - q5 vk64 padded44 medium direct wmma probe

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32` as an
  opt-in diagnostic route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; compiled through CMake/Ninja.
- model/shape:
  focused Qwen2.5 Coder 7B Q5_K_M exported p33 rows, plus p512/p513
  non-steal correctness checks.
- route or candidate:
  Q5_K direct-F32 WMMA medium clone of Vulkan
  `matmul_q5_k_f32_f16acc_aligned_m`: BM64, BN64, BK32, WG256, wave64,
  padded44 LDS rows, f16acc WMMA.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file q5_prompt_p33.txt --output csv`
  with current default routing.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-wmma-vk64-padded44-medium-focused-20260618-064232/`.
- profile/timing:
  focused artifact above, including `perf-p33.csv`, p512/p513 non-steal route
  logs, `vk64-padded44.readobj.txt`, `vk64-padded44.objdump.txt`, and
  `isa-counts.txt`.
- correctness:
  p33 CPU-reference passed and selected VK64; p512 and p513 CPU-reference
  passed and selected the existing rows2 route, so the opt-in does not steal
  production-width rows.
- timing:
  mixed and mostly regressive versus current rows2 p33: Kcur
  `71.92 -> 347.35 us`, Qcur `331.19 -> 458.33 us`, ffn_out
  `2583.19 -> 3082.48 us`, ffn_gate `2330.04 -> 1742.43 us`.
- decision:
  rejected for production; keep as opt-in diagnostic only.
- notes:
  the built HSACO uses wave64, SGPR `32`, VGPR `75`, LDS `11264`, no spills,
  `8` WMMA sites, `2` `ds_store_b16`, `16` global stores, and `2` barriers.
  RADV p33 medium uses the same LDS footprint but SGPR `108`, VGPR `144`, `16`
  WMMA, `48` `ds_load_b64`, `64` `ds_load_u16_d16`, `64` `ds_store_b16`, and
  `96` buffer stores. The missing schedule is still cooperative-matrix
  store/lane ownership, not just BM/BN/LDS shape.

## 2026-06-18 - llama31 8b q8_0 vulkan oracle odd tail rows

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, clean at the latest source
  checkpoint `6f76bc17f` before documentation updates in the workspace root.
- build:
  `build/vulkan-gfx1151`, Release, Vulkan backend, RADV STRIX_HALO;
  workspace ROCm symlink points to `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`,
  `p33/n0/fa1` and `p513/n0/fa1`, `b=1024`, `ub=1024`, one no-warmup
  repetition.
- route or candidate:
  Vulkan oracle prior for Q8_0 odd/tail prompt matmul; not an HRX route
  promotion.
- baseline command:
  `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench sources/llama.cpp/build/hrx-v1-catalog-gfx1151/bin/llama-bench --model <model> --out-dir <out> --prompt <33-or-513> --gen 0 --batch 1024 --ubatch 1024 --flash-attn 1 --repetitions 1 --device Vulkan0`.
- variant command:
  not applicable; this is a Vulkan prior capture.
- route trace:
  - `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-continued/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-continued/vulkan.jsonl`
- profile/timing:
  each artifact contains `stdout.json`, `spv/`, `spvasm/`, `radv/`, and
  `inventory/kernel_inventory.md`.
- correctness:
  both benchmark rows completed and reported `backends=Vulkan`; these rows are
  schedule-oracle captures, not CPU-reference HRX correctness gates.
- timing:
  timings are one-sample oracle captures and are not used for promotion
  directly. They are used to identify route family and emitted schedule.
- decision:
  accepted as Q8_0 odd/tail oracle evidence. p33 uses the medium aligned
  `matmul_q8_0_f32_f16acc_aligned_m` route; p513 uses the large aligned
  `matmul_q8_0_f32_f16acc_aligned_l` route with a fifth column workgroup.
- notes:
  - p33: `spec=[128,64,64,32,64,32,2,16,16,16,64]`,
    `wg_denoms=[64,64,1]`, `LDS=11264`, `VGPR=144`, no spills.
  - p513: `spec=[256,128,128,32,64,64,2,16,16,16,64]`,
    `wg_denoms=[128,128,1]`, `LDS=22528`, `VGPR=192`, no spills.
  - This reinforces that aggregate p512 Q8 work is only a boulder selector.
    Production route promotion must be driven by focused p512/p513 large-route
    evidence and must preserve p33 on a medium/narrow route.
  - The next exact Q8_0 clone should target the remaining RADV/HIP delta:
    cooperative-matrix global store/lane ownership and RADV-like resource
    shape, not another BM/BN rename.

## 2026-06-17 - llama31 8b q4km p512 fa1 vulkan oracle capture

- source: `sources/llama.cpp` `0668b9ee5` on `hrx-kernel-lib-v1`, dirty with
  HRX v1 catalog/tuning work and Vulkan oracle hooks.
- build: `build/vulkan-gfx1151`, Release, Vulkan backend, RADV STRIX_HALO;
  `rocm -> /srv/vm-shared/rocm/rocm-head`; `spirv-dis` from SPIRV-Tools
  v2026.1 installed and used for post-processing.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`,
  `p512/n0`, `fa=1`, `b=1024`, `ub=1024`, one no-warmup repetition.
- route or candidate: Vulkan oracle prior for dense quantized prompt matmul;
  top pipeline is `matmul_q4_k_f32_f16acc_aligned_l`.
- baseline command:
  `MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders,shaderstats GGML_VK_TRACE_JSONL=<out>/vulkan.jsonl GGML_VK_TRACE_SPV_DIR=<out>/spv GGML_VK_TRACE_RADV_PIPELINE_LABELS=1 GGML_VK_PERF_LOGGER=1 build/vulkan-gfx1151/bin/llama-bench -m <model> -p 512 -n 0 -b 1024 -ub 1024 -fa 1 -r 1 -o json --no-warmup -ngl 99 -dev Vulkan0`.
- variant command: not applicable; this is a Vulkan prior capture.
- route trace:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/vulkan.jsonl`.
- profile/timing:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/stdout.json`;
  inventory:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/inventory/kernel_inventory.md`.
- correctness: benchmark completed and JSON reported `backends=Vulkan`; no
  CPU-reference comparison was run because this row is a schedule oracle, not
  an HRX route promotion.
- timing: `371.806637 tok/s`, `avg_ns=1377059873`, one no-warmup sample.
- decision: accepted as the first Vulkan oracle artifact for the gfx1151
  prompt-matmul schedule effort.
- notes:
  - Capture produced 16 pipeline compile rows, 517 dispatch rows, 16 SPIR-V
    files, 16 SPIR-V asm files, and 16 split RADV ISA/stats blocks.
  - Normalized inventory reduced the dispatches to 27 schedule-relevant shape
    signatures.
  - Top dense Q4_K pipeline:
    `matmul_q4_k_f32_f16acc_aligned_l`, hash `0x5666175250529efb`,
    spec `[256,128,128,32,64,64,2,16,16,16,64]`, 190 dispatches,
    `BM128/BN128` workgroup-denominator family, RADV stats
    `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills.
  - Top dense Q6_K pipeline:
    `matmul_q6_k_f32_f16acc_aligned_l`, hash `0x6eebdfb4c3043b23`,
    same spec family, 31 dispatches, RADV stats `SGPR=108`, `VGPR=192`,
    `LDS=22528`, no spills.
  - Schedule comparison against current HRX Q4_K provider is recorded in
    `docs/hrxv1/q4k-vulkan-oracle-schedule-ledger.md`.

## 2026-06-17 - p512 fa1 vulkan oracle first worst-row matrix

- source: `sources/llama.cpp` `4e368666c` on `hrx-kernel-lib-v1`, dirty with
  HRX v1 catalog/tuning work and the new Vulkan oracle runner.
- build: `build/vulkan-gfx1151`, Release, Vulkan backend, RADV STRIX_HALO;
  `rocm -> /srv/vm-shared/rocm/rocm-head`.
- model/shape: first p512/fa1 worst-row matrix from
  `docs/hrxv1/gfx1151-vulkan-oracle-goal.md`, all with `p512/n0`, `fa=1`,
  `b=1024`, `ub=1024`, one no-warmup repetition.
- route or candidate: same-machine Vulkan prior for dense quantized prompt
  matmul across Q4_K, Q5_K, Q6_K, and Q8_0.
- baseline command:
  `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench build/vulkan-gfx1151/bin/llama-bench --model <model> --out-dir <out> --prompt 512 --gen 0 --batch 1024 --ubatch 1024 --flash-attn 1 --repetitions 1 --device Vulkan0`.
- variant command: not applicable; this is a Vulkan prior matrix.
- route trace:
  - `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p512-fa1-20260617-200349/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p512-fa1-20260617-200426/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/vulkan.jsonl`
- profile/timing: each artifact contains `stdout.json`, `spv/`, `spvasm/`,
  `radv/`, and `inventory/`.
- correctness: all benchmark rows completed and JSON reported
  `backends=Vulkan`; these are schedule-oracle rows, not HRX correctness gates.
- timing:
  - Llama 3.1 8B Q4_K_M: `371.806637 tok/s`
  - Qwen2.5 Coder 7B Q5_K_M: `365.256092 tok/s`
  - DeepSeek R1 Qwen 14B Q4_K_M: `290.538702 tok/s`
  - Qwen3 30B Q6_K: `206.048605 tok/s`
  - Llama 3.1 8B Q8_0: `423.307681 tok/s`
- decision: accepted as the first complete p512/fa1 Vulkan oracle matrix for
  dense prompt schedule work.
- notes:
  - Artifact counts: Llama Q4 has 16 SPIR-V/asm/RADV blocks and 27 normalized
    shapes; Qwen2.5 Q5 has 18 and 31; DeepSeek Q4 has 17 and 29; Qwen3 Q6
    has 28 and 39; Llama Q8 has 13 and 24.
  - Dominant dense pipelines share the same large aligned family:
    `spec=[256,128,128,32,64,64,2,16,16,16,64]`,
    `wg_denoms=[128,128,1]`, `workgroup=256x1x1`, `LDS=22528`,
    `VGPR=192`, and no spills.
  - Consolidated schedule ledger:
    `docs/hrxv1/quantized-prompt-vulkan-oracle-ledger.md`.

## 2026-06-17 - llama31 8b q4km vulkan oracle odd tail rows

- source: `sources/llama.cpp` `bde9a1ba4` on `hrx-kernel-lib-v1`, dirty with
  HRX v1 catalog/tuning work.
- build: `build/vulkan-gfx1151`, Release, Vulkan backend, RADV STRIX_HALO.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`,
  `p33/n0` and `p513/n0`, `fa=1`, `b=1024`, `ub=1024`, one no-warmup
  repetition.
- route or candidate: Vulkan odd/tail prior for the selected Q4_K dense prompt
  family.
- baseline command:
  `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench build/vulkan-gfx1151/bin/llama-bench --model <model> --out-dir <out> --prompt <33|513> --gen 0 --batch 1024 --ubatch 1024 --flash-attn 1 --repetitions 1 --device Vulkan0`.
- variant command: not applicable.
- route trace:
  - `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p33-fa1-20260617-200738/vulkan.jsonl`
  - `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p513-fa1-20260617-200751/vulkan.jsonl`
- profile/timing: each artifact contains `stdout.json`, `spv/`, `spvasm/`,
  `radv/`, and `inventory/`.
- correctness: both benchmark rows completed and JSON reported
  `backends=Vulkan`.
- timing:
  - p33: `36.504047 tok/s`
  - p513: `347.622295 tok/s`
- decision: accepted as odd/tail Vulkan prior evidence for the selected Q4_K
  prompt family.
- notes:
  - p33 uses `matmul_q4_k_f32_f16acc_aligned_m` with
    `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`,
    `VGPR=144`, `LDS=11264`, no spills.
  - p513 uses `matmul_q4_k_f32_f16acc_aligned_l` with the same large
    `spec=[256,128,128,32,64,64,2,16,16,16,64]` as p512, but workgroups use
    a fifth column for the tail, for example `[112,5,1]`.
  - This supports separate HRX policy for narrow p33 and production-width
    p512/p513 tails.

## 2026-06-17 - q8_0 vulkan oracle wmma schedule identification

- source: `sources/llama.cpp` `e219c99ec` on `hrx-kernel-lib-v1`, clean;
  root docs dirty with this evidence entry.
- build: scratch compiler probe only, using
  `/srv/vm-shared/projects/llamacpp-devws/rocm` ->
  `/srv/vm-shared/rocm/rocm-head`; production HRX kernels remain required to
  build through CMake/Ninja.
- model/shape:
  Vulkan prior from
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/`,
  Llama 3.1 8B Q8_0, `p512/n0`, `fa=1`.
- route or candidate: Vulkan `matmul_q8_0_f32_f16acc_aligned_l` versus HRX
  scalar-dot Q8 probes; scratch HIP WMMA builtin probe.
- baseline command:
  `spirv-dis <q8_0 spv>` and RADV ISA/stats inspection from the Vulkan oracle
  artifact.
- variant command:
  `rocm/bin/amdclang++ -x hip --offload-arch=gfx1151 -O3 -c .tmp/wmma-probe/wmma_probe.hip.cpp -o .tmp/wmma-probe/wmma_probe.o`,
  then `llvm-objdump --offloading` and `llvm-objdump -d --arch-name=amdgcn
  --mcpu=gfx1151` on the extracted gfx1151 offload image.
- route trace:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/vulkan.jsonl`.
- profile/timing:
  Vulkan Q8_0 oracle timing in
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/stdout.json`;
  scratch probe disassembly in `.tmp/wmma-probe/`.
- correctness: no runtime correctness gate; this is schedule identification
  and compiler capability evidence only.
- timing: Vulkan Q8_0 prior `423.307681 tok/s`; no HRX WMMA timing yet.
- decision: accepted as evidence that the next serious HRX Q8_0 parity
  candidate should be WMMA/cooperative-matrix based, not another scalar-dot
  tile reshuffle.
- notes:
  - Q8_0 SPIR-V uses `OpCapability CooperativeMatrixKHR`,
    `OpCooperativeMatrixLoadKHR`, `OpCooperativeMatrixMulAddKHR`, and
    `OpCooperativeMatrixStoreKHR`.
  - RADV ISA for the hot Q8_0 pipeline emits
    `v_wmma_f16_16x16x16_f16`; stats are `SGPR=108`, `VGPR=192`,
    `LDS=22528`, no spills.
  - The committed scalar-dot probe
    `hrx_mul_mat_vec_q8_0_q8_1_x4_mmql128x128_wg256_f32` is rejected before
    runtime because its HSACO metadata reports `vgpr_count=192`,
    `vgpr_spill_count=472`, and `private_segment_fixed_size=1892`.
  - The scratch HIP builtin probe emitted
    `v_wmma_f32_16x16x16_f16` and `v_wmma_f16_16x16x16_f16` with
    `vgpr_count=25`, `sgpr_count=8`, no LDS, and no spills. This proves the
    compiler path is available in HIP C++ without ROCWMMA headers.

## 2026-06-17 - q8_0 direct wmma16 diagnostic

- source: `sources/llama.cpp` `e219c99ec` on `hrx-kernel-lib-v1`, dirty with
  Q8_0 WMMA diagnostic source/catalog/runtime changes.
- build: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`; built through CMake/Ninja target
  `ggml-hrx`.
- model/shape:
  model-derived Q8_0 prompt rows from Llama 3.1 8B Q8_0, focused
  `k=4096 rows=1024 cols=512` plus synthetic odd/tail `cols=33` and
  `cols=513`.
- route or candidate: `hrx_mul_mat_vec_q8_0_wmma16x16_f32`, direct Q8_0
  dequant-to-f16 and F32 RHS cast-to-f16 using
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`.
- baseline command:
  `test-backend-ops perf --test-file q8_0_prompt_rows1024.txt --output csv`
  with default HRX policy.
- variant command:
  same command with `GGML_HRX_ENABLE_Q8_0_WMMA16_PROMPT=1` and
  `GGML_HRX_TRACE_ROUTES=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-wmma16-focused-20260617-203009/test-nobackend.stderr`,
  `test-odd.stderr`, `perf-default-p512.stderr`, and
  `perf-wmma16-p512.stderr`.
- profile/timing:
  `cache/hrxv1/gfx1151/q8_0-wmma16-focused-20260617-203009/`;
  HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16.hsaco`.
- correctness: passed CPU-reference focused p512 row and p33/p513 odd/tail
  rows; route traces selected `hrx_mul_mat_vec_q8_0_wmma16x16_f32`.
- timing: p512 focused row regressed from `520.688696 us` on current default
  packed Q8_1 route to `734.751304 us` on direct WMMA16.
- decision: rejected for production promotion; keep as opt-in diagnostic
  because it proves the HIP WMMA path, output mapping, and odd/tail handling.
- notes:
  - HSACO metadata: `wavefront_size=32`, `sgpr_count=20`, `vgpr_count=34`,
    `group_segment_fixed_size=0`, `private_segment_fixed_size=0`, no spills.
  - ISA contains `v_wmma_f32_16x16x16_f16`.
  - The next WMMA attempt should not use this one-wave direct global-load
    schedule as the performance target. It should stage/reuse A and B toward
    the Vulkan 128x128 cooperative-matrix family while preserving the verified
    lane/output mapping.

## 2026-06-17 - gfx1151 catalog build baseline

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`,
  `rocm -> /srv/vm-shared/rocm/rocm-head`.
- model/shape: no model row; build and catalog validation only.
- route or candidate: generated HRX v1 HIP C++ catalog from split JSON.
- baseline command: not applicable.
- variant command:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx --parallel "$(nproc)"`.
- route trace: not applicable.
- profile/timing: not applicable.
- correctness:
  - Python compile checks passed for catalog generator and tools.
  - Catalog assembly and validation passed.
  - `ggml-hrx` and `test-backend-hrx` targets built.
- timing: not measured.
- decision: accepted as build-system baseline for further tuning work.
- notes:
  - HIP C++ kernel compilation remains owned by CMake/Ninja.
  - `rocm-head` does not currently expose `rocwmma/rocwmma.hpp`; the
    ROCWMMA-dependent FA prefill WMMA source is excluded from this build.
  - The generated catalog currently has 72 sources, 72 artifacts, 227 families,
    and 227 routes before ROCWMMA source exclusion.

## 2026-06-17 - focused backend test status

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build: `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`.
- model/shape: backend unit-test shapes.
- route or candidate: current HRX v1 catalog providers.
- baseline command: not applicable.
- variant command:
  `LD_LIBRARY_PATH=build/hrx-install/lib:rocm/lib:rocm/lib64:rocm/lib/rocm_sysdeps/lib build/hrx-v1-catalog-gfx1151/bin/test-backend-hrx`.
- route trace: not captured.
- profile/timing: not captured.
- correctness: test aborts at
  `flash_attn_ext_decode_f16_d128_n1_h32_hkv8_kv16384_support_unsupported`.
- timing: not measured.
- decision: unresolved test expectation or route-policy mismatch; not a route
  promotion blocker yet, but must be resolved before using full
  `test-backend-hrx` as a green gate.
- notes:
  - The failing case expects `d=128 n=1 h=32 h_kv=8 kv=16384` to be
    unsupported, while the current runtime reports support.
  - Keep focused op gates for touched routes; do not hide this inside
    full-model testing.

## 2026-06-17 - available model subset discovery

- source: workspace root `130360d79ecc` on `main`; docs dirty with HRX v1 plan
  and this log.
- build: not applicable.
- model/shape:
  - `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf`
    25092532640 bytes.
  - `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`
    17690497440 bytes.
  - `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
    5006938086 bytes.
- route or candidate: not applicable.
- baseline command:
  `find shared/models/llamacpp-hrx2-basket-v1 -maxdepth 3 -type f -name '*.gguf' -printf '%p\t%s\n' | sort`.
- variant command: not applicable.
- route trace: not applicable.
- profile/timing: not applicable.
- correctness: not applicable.
- timing: not applicable.
- decision: use these three GGUFs as the initial partial basket for baseline
  and boulder ranking.
- notes:
  - Partial-basket evidence can reject bad candidates and prioritize boulders.
  - Broad default promotion must wait for the full intended basket and
    odd-size/tail coverage.

## 2026-06-17 - qwen3-coder q4 load probe while downloading

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build: `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`,
  attempted `p32/n0`.
- route or candidate: current HRX v1 catalog providers.
- baseline command: not applicable.
- variant command:
  `GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p 32 -n 0 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev HRX0`.
- route trace:
  `cache/hrxv1/gfx1151/hrx-smoke-qwen3-coder-q4-p32n0-20260617-134514/trace.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/hrx-smoke-qwen3-coder-q4-p32n0-20260617-134514/llama-bench.json`.
- correctness: failed before graph execution; `llama-bench` could not load the
  model.
- timing: not measured.
- decision: do not use this GGUF yet; it was still changing on disk after the
  failed probe.
- notes:
  - The file had a valid `GGUF` header, but its size changed after the failed
    run, consistent with an in-progress download.
  - Re-probe after the model file stops changing.

## 2026-06-17 - qwen3 30b q4xl hrx p32 smoke

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build: `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`;
  `llama-bench` and `test-backend-ops` built successfully.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`,
  `p32/n0`, `fa=0`, `b=512`, `ub=512`.
- route or candidate: current HRX v1 catalog providers on `gfx1151`.
- baseline command: not yet run; Vulkan build/baseline is still missing for
  this source state.
- variant command:
  `GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_GRAPH=1 build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p 32 -n 0 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev HRX0`.
- route trace:
  `cache/hrxv1/gfx1151/hrx-smoke-qwen3-30b-q4xl-p32n0-20260617-134624/trace.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/hrx-smoke-qwen3-30b-q4xl-p32n0-20260617-134624/llama-bench.json`;
  route histogram:
  `cache/hrxv1/gfx1151/hrx-smoke-qwen3-30b-q4xl-p32n0-20260617-134624/route-histogram.txt`.
- correctness: benchmark completed and JSON reported `backends=HRX`; no
  CPU-reference comparison was run.
- timing: `91.676323 tok/s`, one repetition, no warmup; this is a smoke row,
  not promotion evidence.
- decision: useful partial-basket route-evidence smoke; not a performance
  conclusion until repeated and compared against Vulkan.
- notes:
  - Route histogram from trace: 337 `MUL_MAT` routes and 49 `GET_ROWS` routes.
  - Top providers: `hrx_mul_mat_vec_q4_k_f32` 118,
    `hrx_mul_mat_vec_f16_batched_cols16_f32` 96,
    `hrx_mul_mat_vec_f32_batched_rows2_cols8_f32` 47,
    `hrx_get_rows_f32` 47,
    `hrx_mul_mat_vec_q4_k_q8_1_f32` 42.
  - This row points to prompt matmul, F16 attention-chain matmuls, and MoE
    routing as immediate boulder-ranking candidates, but device-time profiling
    and Vulkan comparison are still required.

## 2026-06-17 - qwen3 30b q4xl p32 hrx-vulkan smoke comparison

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build:
  - HRX: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
    `GGML_HRX_AMDGPU_TARGETS=gfx1151`.
  - Vulkan: `build/vulkan-gfx1151`, Release, `GGML_VULKAN=ON`,
    `GGML_HRX=OFF`; Vulkan loader `1.4.341`; device `Vulkan0` is RADV
    STRIX_HALO on AMD Radeon 8060S Graphics.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`,
  `p32/n0`, `fa=0`, `b=512`, `ub=512`, one repetition, no warmup.
- route or candidate: current HRX v1 catalog providers versus same-source
  Vulkan backend.
- baseline command:
  `GGML_VK_PERF_LOGGER=1 build/vulkan-gfx1151/bin/llama-bench -m <model> -p 32 -n 0 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev Vulkan0`.
- variant command:
  `GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_GRAPH=1 build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p 32 -n 0 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev HRX0`.
- route trace:
  `cache/hrxv1/gfx1151/hrx-smoke-qwen3-30b-q4xl-p32n0-20260617-134624/trace.log`.
- profile/timing:
  - HRX JSON:
    `cache/hrxv1/gfx1151/hrx-smoke-qwen3-30b-q4xl-p32n0-20260617-134624/llama-bench.json`.
  - Vulkan JSON:
    `cache/hrxv1/gfx1151/vulkan-smoke-qwen3-30b-q4xl-p32n0-20260617-134908/llama-bench.json`.
  - Vulkan perf labels:
    `cache/hrxv1/gfx1151/vulkan-smoke-qwen3-30b-q4xl-p32n0-20260617-134908/vulkan-perf.log`.
  - summary:
    `cache/hrxv1/gfx1151/qwen3-30b-q4xl-p32n0-smoke-comparison.json`.
- correctness: both benchmark rows completed; JSON reported `backends=HRX` and
  `backends=Vulkan` respectively. No CPU-reference correctness was run.
- timing:
  - HRX: `91.676323 tok/s`.
  - Vulkan: `52.728147 tok/s`.
  - HRX/Vulkan: `1.73866x` on this smoke row.
- decision: harness accepted for first partial-basket smoke comparison; not
  promotion evidence because it is one repetition, no warmup, and `p32` only.
- notes:
  - This row suggests HRX is already ahead of Vulkan for this narrow prefill
    smoke shape, so the next useful comparison should move to production
    shapes such as `p64`, `p512`, odd/tail rows, and decode rows.
  - Vulkan perf labels expose possible baseline-specific slow buckets,
    including F16 attention-chain matmul and small Q4_K rows, but HRX still
    needs HRX-side device/profile buckets before ranking its own bottlenecks.

## 2026-06-17 - qwen3 30b q4xl prefill odd and production matrix

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build:
  - HRX: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
    `GGML_HRX_AMDGPU_TARGETS=gfx1151`.
  - Vulkan: `build/vulkan-gfx1151`, Release, `GGML_VULKAN=ON`,
    `GGML_HRX=OFF`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`,
  `fa=0`, one repetition, no warmup.
- route or candidate: current HRX v1 catalog providers versus same-source
  Vulkan.
- baseline command:
  `GGML_VK_PERF_LOGGER=1 build/vulkan-gfx1151/bin/llama-bench -m <model> -p <p> -n 0 -b <b> -ub <ub> -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev Vulkan0`.
- variant command:
  `GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p <p> -n 0 -b <b> -ub <ub> -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev HRX0`.
- route trace:
  `cache/hrxv1/gfx1151/prefill-matrix-qwen3-30b-q4xl-20260617-135156/hrx-p*/trace.log`
  and
  `cache/hrxv1/gfx1151/prefill-tail-matrix-qwen3-30b-q4xl-20260617-135329/hrx-*/trace.log`.
- profile/timing:
  - primary matrix:
    `cache/hrxv1/gfx1151/prefill-matrix-qwen3-30b-q4xl-20260617-135156/comparison.json`.
  - tail matrix:
    `cache/hrxv1/gfx1151/prefill-tail-matrix-qwen3-30b-q4xl-20260617-135329/comparison.json`.
  - per-row HRX route histograms:
    `route-histogram.txt` beside each HRX row.
  - Vulkan perf labels:
    `vulkan-perf.log` beside each Vulkan row.
- correctness: all rows completed and JSON backend identity matched the
  intended backend. No CPU-reference correctness was run.
- timing:

  ```text
  shape          HRX tok/s   Vulkan tok/s   HRX/Vulkan
  p31 ub512        21.985       254.725       0.086
  p32 ub512        94.945       265.337       0.358
  p33 ub512        93.533       104.767       0.893
  p64 ub512       109.602       365.065       0.300
  p512 ub512      121.471       817.758       0.149
  p513 ub512      119.420       938.295       0.127
  p513 ub1024     110.023       933.304       0.118
  ```

- decision: accepted as partial-basket baseline and odd-size evidence; not
  route promotion evidence.
- notes:
  - `p31` exposes a severe HRX route-selection cliff. Its route histogram has
    160 `hrx_mul_mat_vec_q4_k_f32` uses and no Q4/Q5/Q6 packed Q8_1 prompt
    routes, while `p32/p33/p64/p512` select packed Q8_1 variants.
  - `p512` and `p513` are the current production-regime boulders. HRX remains
    around `0.12x-0.15x` of Vulkan there despite using packed prompt routes.
  - `p513 ub512` splits into residual graph behavior and shows 672 HRX
    `MUL_MAT` route lines, while `p513 ub1024` stays at 337 `MUL_MAT` route
    lines. Both remain far below Vulkan, so the p512/p513 gap is not only
    residual-graph overhead.
  - The first concrete follow-up is a schedule-ledger pass for Q4/Q5/Q6 prompt
    matmul and MoE `MUL_MAT_ID`, using the Vulkan perf labels and HRX route
    histograms as the shape evidence.

## 2026-06-17 - hrx profile mode check for qwen3 30b q4xl p512

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the initial split-catalog refactor.
- build: `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`;
  profile analyzer:
  `build/hrx-system/runtime/src/iree/tools/iree-profile/iree-profile`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`,
  `p512/n0`, `fa=0`, `b=512`, `ub=512`.
- route or candidate: current HRX v1 catalog providers.
- baseline command: not applicable.
- variant command:
  `HRX_PROFILE_FILE=<out>/run.ireeprof HRX_PROFILE_MODE=dispatch build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p 512 -n 0 -b 512 -ub 512 -fa 0 -r 1 -o json --no-warmup -ngl 99 -dev HRX0`.
- route trace: not captured for this profiled row.
- profile/timing:
  `cache/hrxv1/gfx1151/hrx-profile-dispatch-qwen3-30b-q4xl-p512n0-20260617-135530/`.
- correctness: benchmark completed and JSON reported `backends=HRX`.
- timing: profiled HRX row reported `120.201185 tok/s`.
- decision: useful runtime/profile-mode diagnostic, but not sufficient for
  kernel bucket ranking.
- notes:
  - `HRX_PROFILE_MODE=all` makes this HRX-v1 binary report no available HRX
    devices, so it cannot be used as-is for this spike.
  - `HRX_PROFILE_MODE=dispatch` creates a profile bundle and queue-device
    timing, but no per-dispatch events. `iree-profile dispatch` reports zero
    dispatches.
  - The profile statistics row reports queue execute timing with 2126 samples,
    10041 operations, and about 4.21 seconds of device queue duration, matching
    the llama-bench wall row. For kernel bucket ranking, use route traces plus
    rocprofv3 or fix/profile the HRX dispatch-event path.

## 2026-06-17 - qwen3 30b q4xl p31 q8_1 forced-route control

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with the split-catalog refactor.
- build:
  - HRX: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
    `GGML_HRX_AMDGPU_TARGETS=gfx1151`.
  - Vulkan: `build/vulkan-gfx1151`, Release, `GGML_VULKAN=ON`,
    `GGML_HRX=OFF`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`,
  `p31/n0`, `fa=0`, `b=512`, `ub=512`, three repetitions, no warmup.
- route or candidate:
  forced existing packed Q8_1 prompt route selection with
  `GGML_HRX_Q8_1_MMVQ=all`.
- baseline command:
  `GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p 31 -n 0 -b 512 -ub 512 -fa 0 -r 3 -o json --no-warmup -ngl 99 -dev HRX0`.
- variant command:
  `GGML_HRX_Q8_1_MMVQ=all GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p 31 -n 0 -b 512 -ub 512 -fa 0 -r 3 -o json --no-warmup -ngl 99 -dev HRX0`.
- route trace:
  - baseline:
    `cache/hrxv1/gfx1151/p31-q8-1-mmvq-forced-r3-20260617-140229/hrx-p31-baseline/trace.log`.
  - forced:
    `cache/hrxv1/gfx1151/p31-q8-1-mmvq-forced-r3-20260617-140229/hrx-p31-q8all/trace.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/p31-q8-1-mmvq-forced-r3-20260617-140229/summary.json`;
  Vulkan reference:
  `cache/hrxv1/gfx1151/p31-q8-1-mmvq-forced-r3-20260617-140229/vulkan-p31/llama-bench.json`.
- correctness:
  all rows completed and JSON reported the intended `backends` value. No
  CPU-reference correctness was run.
- timing:
  - HRX baseline: `94.479619 tok/s` average, samples
    `[93.8861, 95.6614, 93.8914]`, steady average `94.7764`.
  - HRX forced packed Q8_1: `63.511767 tok/s` average, samples
    `[62.8932, 64.4519, 63.1902]`, steady average `63.82105`.
  - Vulkan reference: `260.583814 tok/s` average, samples
    `[257.865, 273.696, 250.19]`, steady average `261.943`.
- decision:
  reject broad relaxation of `GGML_HRX_Q8_1_MMVQ_AUTO_COLS_MIN` for this p31
  row.
- notes:
  - The forced route selected 160 `hrx_mul_mat_vec_q4_k_q8_1_f32`, 22
    `hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32`, and 11
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32` routes.
  - The unforced route selected the legacy narrow providers and was about
    `1.49x` faster than forced packed Q8_1.
  - The original single-sample `p31` cliff is superseded for this guard
    hypothesis. The remaining p31 gap versus Vulkan needs either a different
    narrow schedule or device-time evidence, not a blind packed-route gate
    change.

## 2026-06-17 - qwen3 30b q4xl focused p512 op-shape export

- source:
  - root docs/tooling dirty with `tools/hrxv1_focus_exported_ops.py`.
  - `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with
    split-catalog refactor and a `test-backend-ops` CSV timing-field patch.
- build:
  `build/hrx-v1-catalog-gfx1151`; `export-graph-ops` and `test-backend-ops`
  built through CMake/Ninja.
- model/shape:
  `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`, `p512/n0`, `fa=0`,
  `b=512`, `ub=512`.
- route or candidate:
  model-derived focused op extraction for schedule A/B, not a kernel candidate.
- baseline command:
  `build/hrx-v1-catalog-gfx1151/bin/export-graph-ops -m <model> -p 512 -n 0 -b 512 -ub 512 -fa 0 -o <out>/ops.txt`.
- variant command:
  `python3 tools/hrxv1_focus_exported_ops.py --input <out>/ops.txt --out-dir <out>/focused`.
- route trace: not applicable.
- profile/timing:
  `cache/hrxv1/gfx1151/model-op-shapes-qwen3-30b-q4xl-p512-20260617-140526/`.
- correctness:
  export completed and produced 78 unique graph op rows.
- timing:
  not a performance row.
- decision:
  accepted as the focused schedule-test source for p512 Q4/Q5/Q6 prompt, MoE
  prompt, and F16 attention rows.
- notes:
  - Focused files produced 8 Q4/Q5/Q6 prompt `MUL_MAT` rows, 8 decode
    `MUL_MAT` rows, 3 MoE prompt `MUL_MAT_ID` rows, and 4 F16 attention-chain
    prompt rows.
  - `test-backend-ops --test-file` is the immediate correctness/perf harness
    for these rows.
  - `test-backend-ops` CSV output was patched to include `passed`, `time_us`,
    `flops`, `bandwidth_gb_s`, `memory_kb`, and `n_runs`, because this branch
    previously hid the timing fields in CSV output.

## 2026-06-17 - qwen3 30b q4xl focused qk prompt schedule A/B

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with
  split-catalog refactor and `test-backend-ops` CSV timing-field patch.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`.
- model/shape:
  exact p512 exported Q4/Q5/Q6 prompt `MUL_MAT` rows from
  `cache/hrxv1/gfx1151/model-op-shapes-qwen3-30b-q4xl-p512-20260617-140526/focused/qk_prompt.txt`.
- route or candidate:
  current default route policy versus forced existing packed Q8_1 policy
  (`GGML_HRX_Q8_1_MMVQ=all`).
- baseline command:
  `GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <qk_prompt.txt> --output csv`.
- variant command:
  `GGML_HRX_Q8_1_MMVQ=all GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <qk_prompt.txt> --output csv`.
- route trace:
  - default correctness:
    `cache/hrxv1/gfx1151/focused-qk-prompt-opgate-20260617-140712/test-trace.log`.
  - default perf:
    `cache/hrxv1/gfx1151/focused-qk-prompt-perf-timing-20260617-141110/perf-trace.log`.
  - forced Q8:
    `cache/hrxv1/gfx1151/focused-qk-prompt-q8all-20260617-141500/`.
- profile/timing:
  - default timing:
    `cache/hrxv1/gfx1151/focused-qk-prompt-perf-timing-20260617-141110/perf.csv`.
  - forced timing:
    `cache/hrxv1/gfx1151/focused-qk-prompt-q8all-20260617-141500/perf/perf.csv`.
- correctness:
  all 8 focused rows passed CPU-reference comparison for both default and
  forced-Q8 route policies.
- timing:

  ```text
  shape          default us    q8all us    q8all/default
  Vcur-0           2168.82      2511.85        1.16x
  Vcur-1            540.18       249.14        0.46x
  node_32         16165.07     19907.09        1.23x
  node_100          955.90      1130.07        1.18x
  node_372        10753.86     23340.47        2.17x
  Qcur-0          22548.38     30107.66        1.34x
  Qcur-1            995.79      1210.79        1.22x
  result_output 1117752.89   1131429.98        1.01x
  ```

- decision:
  reject blanket packed-Q8_1 forcing for p512 QK prompt. Keep `Vcur-1` as a
  narrow Q5 route-policy candidate; treat Q4 and Q6 as schedule-work targets.
- notes:
  - This is the correct optimization granularity for the kernel catalog:
    model-derived op rows, route evidence, CPU-reference correctness, and
    focused timing before full-model A/B.
  - The current Q4 Q8_1 schedule is not production quality on these p512 rows.
  - The current Q6 x4 MMQL route is especially weak for the exported
    `node_372` and `result_output` rows. Candidate HIP schedule variants should
    be built by CMake/Ninja and tested against these rows first.

## 2026-06-17 - q4-k x4 mmq64 focused candidate

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with split catalog refactor, focused CSV timing patch, and the opt-in Q4_K
  x4 MMQ64 candidate.
- build: `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`,
  `rocm -> /srv/vm-shared/rocm/rocm-head`.
- model/shape:
  `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`; focused p512 exported
  Q4/Q5/Q6 `MUL_MAT` rows plus HRX-only model smokes at `p33`, `p512`, and
  `p513 ub1024`.
- route or candidate:
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmq64x64_wg256_f32`, opt-in with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <qk_prompt.txt> --output csv`
  using current default route policy.
- variant command:
  same command with `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`; model
  smokes used same-binary `llama-bench -p <p> -n 0 -fa 0 -r 1 --no-warmup`.
- route trace:
  `cache/hrxv1/gfx1151/focused-qk-prompt-q4x4mmq64-opgate-20260617-142923/test.log`,
  `cache/hrxv1/gfx1151/focused-qk-prompt-q4x4mmq64-perf-20260617-142946/perf.log`,
  and `cache/hrxv1/gfx1151/q4x4mmq64-model-ab-*/**/trace.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/focused-qk-prompt-q4x4mmq64-perf-20260617-142946/perf.csv`,
  `cache/hrxv1/gfx1151/q4x4mmq64-model-ab-p512-20260617-143156/`,
  and `cache/hrxv1/gfx1151/q4x4mmq64-model-ab-odd-20260617-143237/`.
- correctness:
  focused CPU-reference `test-backend-ops test` passed all 8 exported p512
  rows. Route trace showed the new provider selected for the three Q4_K rows
  only; Q5/Q6 route families were unchanged.
- timing:
  focused Q4 rows improved:
  `Vcur-0 2168.82 -> 202.02 us`, `node_32 16165.07 -> 1304.93 us`,
  `Qcur-0 22548.38 -> 1193.69 us`.
  Same-binary HRX model smokes improved:
  `p33 91.57 -> 120.00 tok/s`, `p512 119.60 -> 182.49 tok/s`,
  `p513 ub1024 108.98 -> 158.75 tok/s`.
- decision:
  keep as opt-in candidate; do not default-promote yet. Next gates are repeated
  HRX/Vulkan model rows, focused odd exported-op rows, full available basket,
  and a Q6 schedule pass.
- notes:
  HSACO facts for the candidate: wavefront 64, VGPR 192, SGPR 40, no spills,
  LDS 2304 bytes, 128 `v_dot4_i32_iu8` instructions. This validates the
  schedule-side correction: Q4 needed a packed tiled MMQ route, not aggregate
  route-policy tuning.

## 2026-06-17 - q6-k x4 mmq64 prompt candidate and narrow threshold

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with split catalog refactor, focused CSV timing patch, opt-in Q4 MMQ64, and
  opt-in Q6 MMQ64.
- build: `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`,
  `rocm -> /srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Qwen3 30B Q4_K_XL focused `MUL_MAT` Q/K prompt rows exported at p2, p32,
  p33, p512, and p513; model smokes at p33, p512, and p513.
- route or candidate:
  `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x64_wg256_f32`, opt-in with
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`; after threshold testing both
  Q4 and Q6 MMQ64 prompt candidates are guarded at `cols >= 32`.
- baseline command:
  `test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <focused qk_prompt.txt> --output csv`
  using current default route policy.
- variant command:
  same command with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1 GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- route trace:
  - Q6 p512 correctness:
    `cache/hrxv1/gfx1151/focused-q6-prompt-mmq64-opgate-20260617-145034/`.
  - Q6 p512 timing A/B:
    `cache/hrxv1/gfx1151/focused-q6-prompt-mmql-vs-mmq64-20260617-145055/`.
  - exact p33/p513 odd correctness:
    `cache/hrxv1/gfx1151/focused-qk-prompt-q4q6mmq64-odd-exact-opgate-20260617-145554/`.
  - p2/p33/p513 pre-threshold correctness:
    `cache/hrxv1/gfx1151/focused-qk-prompt-q4q6mmq64-p2-p33-p513-opgate-20260617-145708/`.
  - p2 perf rejection:
    `cache/hrxv1/gfx1151/focused-qk-prompt-p2-q4q6mmq64-perf-20260617-145749/`.
  - p8/p16/p32 threshold sweep:
    `cache/hrxv1/gfx1151/focused-qk-prompt-narrow-threshold-q4q6mmq64-20260617-145835/`.
  - final p2/p32 threshold correctness:
    `cache/hrxv1/gfx1151/focused-qk-prompt-q4q6mmq64-threshold-final-opgate-20260617-150129/`.
- profile/timing:
  Q6 p512 focused timing:

  ```text
  row             default us   q6 mmq64 us   speedup
  node_372          10475.60        2682.39    3.91x
  result_output    437618.52       98076.84    4.46x
  ```

  Narrow threshold focused timing:

  ```text
  p2:  Q4 MMQ64 rejected; Vcur-0 11.04 -> 86.21 us, node_32 65.06 -> 216.43 us.
  p8:  mixed; Vcur-0 loses 38.88 -> 86.51 us, node_32 wins 251.39 -> 214.24 us.
  p16: mixed; Vcur-0 loses 76.08 -> 87.10 us, node_32 wins 500.88 -> 213.25 us.
  p32: accepted threshold; Vcur-0 141.99 -> 88.68 us, node_32 1023.93 -> 219.25 us,
       node_372 1731.58 -> 275.50 us, result_output 51337.06 -> 12636.86 us.
  ```

- model A/B:

  ```text
  shape   default tok/s   q4+q6 mmq64 tok/s   speedup
  p33          90.409             128.015       1.42x
  p512        118.151             185.620       1.57x
  p513        107.658             161.665       1.50x
  ```

- correctness:
  CPU-reference focused gates passed for exact p33 and p513 rows with Q4/Q6
  MMQ64 selected. After tightening the guard to `cols >= 32`, p2 passed with
  zero MMQ64 route selections and p32 passed with five MMQ64 route selections.
- decision:
  accept Q6 MMQ64 as an opt-in gfx1151 prompt candidate for `cols >= 32`;
  reject the candidate for p2/narrow prompt. Keep Q4/Q6 MMQ64 opt-in until
  repeated HRX/Vulkan rows and broader basket gates are recorded.
- notes:
  - Existing selected Q6 MMQL providers spill heavily on gfx1151; the new Q6
    MMQ64 HSACO reports wavefront 64, VGPR 181, SGPR 42, and zero spills.
  - The no-spill legacy Q6 `mmq32x32` route was rejected earlier for p512
    because focused Q6 timing was slower than the spilling large MMQL tile.
  - Aggregate model speedups confirmed the focused schedule win, but did not
    drive the guard. The guard was set from focused p2/p8/p16/p32 schedule A/B.

## 2026-06-17 - q4/q6 mmq64 repeated HRX/Vulkan comparison and policy probes

- source: `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty
  with split catalog refactor and opt-in Q4/Q6/Q5 selector candidates.
- build:
  - HRX: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
    `GGML_HRX_AMDGPU_TARGETS=gfx1151`.
  - Vulkan: `build/vulkan-gfx1151`, Release, `GGML_VULKAN=ON`.
- model/shape:
  `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`, `fa=0`, `n=0`,
  `b=1024`, `ub=1024`, repeated `r=3`.
- route or candidate:
  current HRX default versus Q4/Q6 MMQ64 opt-in:
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1` and
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- baseline command:
  HRX default:
  `build/hrx-v1-catalog-gfx1151/bin/llama-bench -m <model> -p <p> -n 0 -b 1024 -ub 1024 -fa 0 -r 3 -o json --no-warmup -ngl 99 -dev HRX0`.
- variant command:
  same HRX command with Q4/Q6 MMQ64 env vars, and same-source Vulkan command:
  `build/vulkan-gfx1151/bin/llama-bench ... -dev Vulkan0` with
  `GGML_VK_PERF_LOGGER=1`.
- route/profile artifacts:
  - repeated HRX/Vulkan:
    `cache/hrxv1/gfx1151/q4q6-mmq64-hrx-vulkan-r3-20260617-150414/`.
  - attempted rocprof:
    `cache/hrxv1/gfx1151/rocprof-q4q6-p512-20260617-150654/` and
    `cache/hrxv1/gfx1151/rocprof-q4q6-p512-bool-20260617-150731/`.
- correctness:
  `llama-bench` completed and JSON reported `backends=HRX` for HRX rows and
  `backends=Vulkan` for Vulkan rows.
- timing:

  ```text
  shape   HRX default   HRX q4+q6   Vulkan     q4+q6/default   q4+q6/Vulkan
  p33        92.230      127.378    218.193        1.38x          0.58x
  p512      117.057      181.212   1095.640        1.55x          0.17x
  p513      104.277      156.052    922.711        1.50x          0.17x
  ```

- decision:
  Q4/Q6 MMQ64 remains a strong opt-in candidate but is not default-promoted.
  The gap to Vulkan is still structural; next work should attack remaining
  F16 attention, MoE/support traffic, or other high-count buckets with focused
  evidence.
- notes:
  - HRX p512 route counts after Q4/Q6 cleanup still show 288 F16 batched
    attention-chain dispatches, 141 F32 MoE-logits dispatches, 141 GET_ROWS,
    and Q5 rows.
  - Vulkan p512 perf labels still show large `MUL_MAT_ID` MoE and F16
    attention-chain buckets.
  - `rocprofv3 --kernel-trace` ran the app but produced no output files on this
    HRX path, even with explicit boolean flags. For now, use route traces plus
    focused backend-op timing unless the profiler path is repaired.

## 2026-06-17 - f16 attention and moe-logits route-policy sweeps

- source/build/model:
  same source and HRX build as above; focused rows exported from
  `cache/hrxv1/gfx1151/model-op-shapes-qwen3-30b-q4xl-p512-20260617-140526/`.
- route or candidate:
  existing selector policies only; no new HIP kernel source.
- F16 attention baseline/variant command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <attention_f16_prompt.txt> --output csv`,
  with env vars disabling `F16_BATCHED_COLS16`, then `COLS8`, then `COLS4`.
- F32 MoE logits baseline/variant command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <f32_moe_logits.txt> --output csv`,
  with env vars disabling `F32_BATCHED_ROWS2_COLS8`, then `COLS16`, then
  `COLS8`.
- artifacts:
  - F16:
    `cache/hrxv1/gfx1151/focused-f16-attention-policy-sweep-20260617-151010/`.
  - F32:
    `cache/hrxv1/gfx1151/focused-f32-moe-logits-policy-sweep-20260617-151838/`.
- correctness:
  F16 attention CPU-reference gate passed all 4 rows. F32 MoE-logits
  CPU-reference gate passed both rows.
- timing:

  F16 attention selected rows:

  ```text
  policy    kqv p512 us   kq p512 us       decision
  cols16      3147673      7649239         baseline; total roughly flat vs cols8
  cols8       2188563      8650358         helps KQV, hurts KQ
  cols4       2151651     13579146         reject
  generic    15706807     55454001         reject
  ```

  F32 MoE logits prompt row:

  ```text
  policy             p512 us    decision
  rows2_cols8         182.31    keep default
  cols8               238.48    reject
  cols16              508.07    reject
  generic             710.19    reject
  ```

- decision:
  no selector-policy promotion for F16 or F32. F16 needs a real schedule or
  fusion change rather than choosing an existing alternate provider. F32
  MoE-logits is already on the best existing route for this focused shape.
- notes:
  The F16 focused times are huge because `test-backend-ops` replays the full
  batched tensor shape; use the relative policy result, not the absolute value,
  for this decision.

## 2026-06-17 - q5 small-row mmql128 opt-in diagnostic

- source/build/model:
  same source and HRX build as above.
- route or candidate:
  new selector-only opt-in:
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_SMALL_PROMPT=1`, using the existing
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32` provider for
  `rows == 512`, `cols >= 32`.
- focused baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <q5_prompt.txt> --output csv`
  using default route policy.
- focused variant command:
  same command with `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_SMALL_PROMPT=1`.
- model A/B command:
  same-binary HRX `llama-bench -p 512 -n 0 -b 1024 -ub 1024 -fa 0 -r 3`
  comparing Q4/Q6 MMQ64 against Q4/Q5/Q6 opt-ins.
- artifacts:
  - focused:
    `cache/hrxv1/gfx1151/focused-q5-small-mmql128-opgate-20260617-152125/`.
  - model:
    `cache/hrxv1/gfx1151/q5small-model-ab-p512-20260617-152157/`.
- correctness:
  focused CPU-reference gate passed all 3 Q5 rows.
- timing:

  ```text
  focused row   default us   q5small us   speedup
  Vcur-1           533.16       234.86      2.27x
  node_100         932.85       935.12      1.00x
  Qcur-1           887.70       874.75      1.01x

  model p512:
  q4+q6      184.09 tok/s
  q4+q5+q6   180.18 tok/s
  ```

- decision:
  keep as an opt-in diagnostic only. Do not default-promote: focused timing
  improved the isolated small Q5 row, but the same-runner model A/B was noisy
  and slightly regressive with 36 small-Q5 MMQL route selections.
- notes:
  This is a useful example of why focused wins still need model integration
  evidence. The next Q5 work should be a real fused/schedule candidate or a
  broader MoE-path change, not this narrow policy flip.

## 2026-06-17 - f16 attention rows2 cols16 wg32 candidate

- source/build/model:
  same HRX build tree, rebuilt after adding
  `hrx_mul_mat_vec_f16_batched_rows2_cols16_wg32_f32` to
  `mul_mat_vec_f16_batched.hip.cpp`, generated catalog metadata, and the
  opt-in selector path.
- route or candidate:
  `GGML_HRX_ENABLE_F16_BATCHED_ROWS2_COLS16_WG32_PROMPT=1`. This follows the
  Vulkan F16 DMMV prior for the attention-chain `MUL_MAT` rows: one wave32
  workgroup, two rows, sixteen columns, wave-local reduction, F16 A times F32
  RHS with F32 accumulation.
- focused baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <attention_f16_prompt.txt> --output csv`
  using default F16 batched cols16 routing.
- focused variant command:
  same command with
  `GGML_HRX_ENABLE_F16_BATCHED_ROWS2_COLS16_WG32_PROMPT=1`.
- model A/B command:
  same-binary HRX `llama-bench -n 0 -b 1024 -ub 1024 -fa 0 -r 3`
  comparing Q4/Q6 MMQ64 opt-ins against Q4/Q6 plus the F16 candidate.
- artifacts:
  - p512 correctness:
    `cache/hrxv1/gfx1151/focused-f16-rows2-cols16-wg32-opgate-20260617-153054/`.
  - p512 focused timing:
    `cache/hrxv1/gfx1151/focused-f16-rows2-cols16-wg32-perf-20260617-153215/`.
  - p33/p513 odd correctness:
    `cache/hrxv1/gfx1151/focused-f16-rows2-cols16-wg32-odd-opgate-20260617-153411/`.
  - p512 model A/B:
    `cache/hrxv1/gfx1151/f16-rows2-cols16-model-ab-p512-20260617-153540/`.
  - p33/p513 model A/B:
    `cache/hrxv1/gfx1151/f16-rows2-cols16-model-ab-odd-20260617-153620/`.
- correctness:
  p512 focused gate passed 4/4 rows. p33+p513 focused gate passed 8/8 rows.
  Route traces show the new provider selected only for prompt-width F16 rows;
  existing cols1 rows stayed on
  `hrx_mul_mat_vec_f16_batched_rows2_cols1_x8_wg32_f32`.
- compile evidence:
  built by CMake/Ninja into `mul_mat_vec_f16_batched.hsaco`. HSACO metadata for
  the new symbol reports wavefront 32, VGPR 67, SGPR 83, no spills, and zero
  LDS.
- timing:

  ```text
  focused p512 row   default us    variant us   speedup
  cols1 row              3680.01      3686.51     1.00x
  kqv p512            3177985.00   1565222.50     2.03x
  cols1 row             36576.81     36583.12     1.00x
  kq p512             7474460.50   1673102.50     4.47x

  model p33:
  q4+q6                 128.17 tok/s
  q4+q6+f16             146.21 tok/s  1.14x

  model p512:
  q4+q6                 188.87 tok/s
  q4+q6+f16             246.46 tok/s  1.31x

  model p513:
  q4+q6                 161.65 tok/s
  q4+q6+f16             227.96 tok/s  1.41x
  ```

- decision:
  keep as an opt-in gfx1151 candidate pending broader basket coverage and
  target-specific policy migration. It satisfies focused correctness,
  route-selection, compile-resource, odd-size, and same-binary model A/B gates
  on the current Qwen3 Q4_K_XL subset, but should not silently affect gfx1100
  legacy routing.
- notes:
  This candidate validates the kernel/schedule A/B workflow: the aggregate
  p512 Vulkan gap ranked attention as a boulder, but the winning change came
  from matching the Vulkan subgroup DMMV work ownership at the focused row
  level.

## 2026-06-17 - current best opt-in stack versus Vulkan

- source/build/model:
  `sources/llama.cpp` `0668b9ee5` on `hrx-kernel-lib-v1`, dirty with split
  catalog, Q4/Q6 MMQ64, F16 rows2-cols16, and CSV-output changes.
  HRX build `build/hrx-v1-catalog-gfx1151`; Vulkan build `build/vulkan-gfx1151`.
  Model:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`.
- route or candidate:
  Q4 MMQ64, Q6 MMQ64, and F16 rows2-cols16 all opt-in:
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`,
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`, and
  `GGML_HRX_ENABLE_F16_BATCHED_ROWS2_COLS16_WG32_PROMPT=1`.
- command:
  same-machine `llama-bench -n 0 -b 1024 -ub 1024 -fa 0 -r 3 -o json
  --no-warmup -ngl 99`, HRX `-dev HRX0` versus Vulkan `-dev Vulkan0` with
  backend-specific `LD_LIBRARY_PATH`; Vulkan also used `GGML_VK_PERF_LOGGER=1`.
- artifacts:
  `cache/hrxv1/gfx1151/current-best-hrx-vulkan-r3-20260617-154104/`.
- correctness:
  benchmark rows completed; JSON reports `backends=HRX` for HRX rows and
  `backends=Vulkan` for Vulkan rows. Focused correctness for the opt-in routes
  is recorded in the earlier Q4/Q6 and F16 entries.
- timing:

  ```text
  shape   HRX tok/s   Vulkan tok/s   HRX/Vulkan
  p33       146.55        211.98        0.691x
  p512      251.34       1113.21        0.226x
  p513      233.30        940.49        0.248x
  ```

- decision:
  current opt-in stack is a real HRX improvement, but it is not close enough
  for production parity at p512/p513. Use this artifact for boulder ranking,
  not as an aggregate-only optimization target.
- notes:
  HRX p512 route histogram is dominated by Q4/Q5/Q6 packed prompt matmuls,
  F16 attention-chain rows, F32 MoE-logit matmuls, and `GET_ROWS`. Vulkan
  p512 labels show large MoE `MUL_MAT_ID` time, so the next evidence target is
  the MoE expert path and its schedule/fusion boundary.

## 2026-06-17 - moe mul_mat_id and decomposed path focused evidence

- source/build/model:
  same source, build, and Qwen3 Q4_K_XL p512 exported-op artifact as above.
- route or candidate:
  existing HRX Q4_K `MUL_MAT_ID` providers versus the decomposed HRX model
  path. This was an investigation, not a promotion candidate.
- focused commands:
  `test-backend-ops support/test/perf -b HRX0 -o MUL_MAT_ID --test-file <moe_qk_prompt.txt> --output csv`
  and
  `test-backend-ops test/perf -b HRX0 -o MUL_MAT --test-file <f32_moe_logits.txt> --output csv`.
- artifacts:
  - `cache/hrxv1/gfx1151/focused-moe-mul-mat-id-opgate-20260617-154557/`.
  - `cache/hrxv1/gfx1151/focused-moe-decomposed-opgate-20260617-154752/`.
- correctness:
  Q4_K `MUL_MAT_ID` focused rows passed CPU-reference support/test. Q5_K
  `MUL_MAT_ID` is unsupported. Decomposed F32 MoE-logit `MUL_MAT` rows passed.
- timing:

  ```text
  focused row                         HRX time
  MUL_MAT_ID q4 gate p512             20840 us
  MUL_MAT_ID q4 down p512             28468 us
  MUL_MAT_ID q5 down p512             unsupported
  F32 MoE logits p512                 174 us
  F32 MoE logits cols1                3.1 us

  Vulkan p512 labels from current-best:
  MUL_MAT_ID q4 m=2048 n=8 k=768      ~2007-2084 us
  MUL_MAT_ID q4 m=768 n=8 k=2048      ~1882-1938 us
  MUL_MAT_ID q5 m=2048 n=8 k=768      ~2202-2297 us
  ```

- decision:
  do not promote or force the existing HRX `MUL_MAT_ID` route. The useful next
  work is a prior-driven Q4/Q5 ID or equivalent expert-fusion schedule that
  matches Vulkan's tiled packed-Q8 MMQ shape and supports Q5, with route-density
  evidence for real Qwen p512.
- notes:
  Full HRX p512 `llama-bench` route trace does not dispatch `MUL_MAT_ID`; it
  decomposes MoE into ordinary packed `MUL_MAT`, F32 logits, and `GET_ROWS`.
  That is a structural route/fusion boundary, not just a local kernel knob.

## 2026-06-17 - plain mul_mat_id route tracing and provider pinning

- source/build/model:
  same HRX v1 source, `build/hrx-v1-catalog-gfx1151`, and Qwen3 30B Q4_K_XL
  p512 exported MoE focus file as above. GPU process state was checked with
  `amd-smi monitor -u -m -v -q -w 1 -i 1 --json`; no running processes were
  reported before the focused run.
- route or candidate:
  instrumentation-only change for plain `MUL_MAT_ID`: `GGML_HRX_TRACE_ROUTES`
  now prints selected provider, type, K, rows, ids, tokens, experts, grouping,
  Q8_1 x4 status, workgroup grid, and destination. The new
  `GGML_HRX_EXPECT_MUL_MAT_ID_PROVIDER` assertion mirrors the existing
  SWIGLU provider pinning hook.
- artifacts:
  - failed expected-provider probe:
    `cache/hrxv1/gfx1151/focused-moe-mul-mat-id-trace-20260617-155829/`.
  - clean current-provider probe:
    `cache/hrxv1/gfx1151/focused-moe-mul-mat-id-trace-20260617-155927/`.
- correctness:
  with `GGML_HRX_EXPECT_MUL_MAT_ID_PROVIDER=hrx_mul_mat_id_q4_k_wg64_f32`,
  Q4_K `ffn_moe_gate-0` and `ffn_moe_down-0` passed CPU-reference test.
  Q5_K `ffn_moe_down-1` remains unsupported.
- timing:

  ```text
  focused row                         provider                        time
  q4 gate p512                        hrx_mul_mat_id_q4_k_wg64_f32    20852 us
  q4 down p512                        hrx_mul_mat_id_q4_k_wg64_f32    28244 us
  q5 down p512                        unsupported                     n/a
  ```

- route evidence:

  ```text
  q4 gate: k=2048 rows=768 n_ids=8 n_tokens=512 n_experts=128 grouped=0 q8_1_x4=0 wg_count=[768,4096,1]
  q4 down: k=768 rows=2048 n_ids=8 n_tokens=512 n_experts=128 grouped=0 q8_1_x4=0 wg_count=[2048,4096,1]
  ```

- decision:
  this is accepted as productionization/evidence plumbing, not as a
  performance promotion. A deliberately pinned grouped-Q8 provider failed,
  proving the existing grouped x4 ID route is not selected for these Qwen3
  expert shapes. The next MoE candidate needs a new Q4/Q5 expert schedule for
  `k=2048/768`, not a selector flip to the existing `k==512` grouped family.

## 2026-06-17 - q4 moe id wide-k grouped q8_1 x4 opt-in

- source/build/model:
  same HRX v1 source, `build/hrx-v1-catalog-gfx1151`, and Qwen3 30B Q4_K_XL
  no-FA prefill target. Change is a selector-only opt-in that allows the
  existing grouped Q4_K ID Q8_1 x4 MMQ16 provider for wider K expert rows.
- route or candidate:
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32` under
  `GGML_HRX_ENABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`.
  Guard:
  `k % 256 == 0`, `rows % 64 == 0`, `n_ids == 8`, `n_tokens >= 32`, and
  grouped-route scratch/quantization providers available.
- focused artifacts:
  - p512 correctness/timing:
    `cache/hrxv1/gfx1151/focused-moe-q4-id-widek-mmq16-20260617-160209/`.
  - p33/p513 exported shape files:
    `cache/hrxv1/gfx1151/model-op-shapes-qwen3-30b-q4xl-odd-batch-20260617-160547/`.
  - p33/p513 correctness:
    `cache/hrxv1/gfx1151/focused-moe-q4-id-widek-mmq16-odd-20260617-160622/`.
  - p33/p513 timing:
    `cache/hrxv1/gfx1151/focused-moe-q4-id-widek-mmq16-odd-perf-20260617-160640/`.
- model artifacts:
  - same-run p512 A/B:
    `cache/hrxv1/gfx1151/q4-id-widek-model-ab-p512-20260617-160312/`.
  - same-run p33/p513 A/B:
    `cache/hrxv1/gfx1151/q4-id-widek-model-ab-odd-20260617-160345/`.
  - repeated current-best HRX:
    `cache/hrxv1/gfx1151/current-best-with-q4-id-widek-r3-20260617-160734/`.
- correctness:
  Q4_K `MUL_MAT_ID` rows passed CPU-reference tests for p33, p512, and p513
  with `GGML_HRX_EXPECT_MUL_MAT_ID_PROVIDER` pinned to the grouped Q8_1 x4
  provider. Q5_K `MUL_MAT_ID` remains unsupported.
- focused timing:

  ```text
  shape   q4 gate      q4 down      default comparison
  p33       258 us       252 us      odd focused baseline not rerun
  p512     2094 us      1986 us      20852 us / 28244 us default wg64
  p513     2138 us      2098 us      odd focused baseline not rerun
  ```

- model timing:

  ```text
  shape   baseline HRX   wide-k HRX   lift    Vulkan r3 reference   HRX/Vulkan
  p33       146.55         185.55     1.27x        211.98             0.875x
  p512      251.34         476.79     1.90x       1113.21             0.428x
  p513      233.30         418.16     1.79x        940.49             0.445x
  ```

- decision:
  accept as an opt-in gfx1151 candidate and record in the tuning database.
  Do not make it an unconditional default yet: Q5_K ID support is missing,
  broader basket coverage is pending, and default selector policy must remain
  target-specific so gfx1100 legacy behavior is not perturbed.

## 2026-06-17 - q5 moe id grouped q8_1 x4 mmq16 opt-in

- source/build/model:
  HRX v1 source `0668b9ee5-dirty`, build `build/hrx-v1-catalog-gfx1151`,
  ROCm `/srv/vm-shared/rocm/rocm-head`, Qwen3 30B Q4_K_XL no-FA prefill.
- route or candidate:
  new CMake/Ninja-built HIP source `mul_mat_id_q5_k_q8_1_x4_mmq.hip.cpp`,
  export `hrx_mul_mat_id_q5_k_grouped_q8_1_x4_mmq64x16_wg64_f32`, gated by
  `GGML_HRX_ENABLE_Q5_K_ID_Q8_1_X4_MMQ16_PROMPT=1`. It follows the accepted
  Q4 MoE grouped Q8_1 x4 MMQ16 route and dense Q5_K Q8_1 x4 wave64 MMQ prior.
- baseline command:
  current opt-in stack with Q4/Q6 dense MMQ64, F16 rows2-cols16, and Q4 MoE ID
  wide-K enabled, but without Q5 ID.
- variant command:
  same command plus `GGML_HRX_ENABLE_Q5_K_ID_Q8_1_X4_MMQ16_PROMPT=1`.
  Focused gates also set `GGML_HRX_EXPECT_MUL_MAT_ID_PROVIDER` for Q4 and
  `GGML_HRX_EXPECT_MUL_MAT_ID_Q5_PROVIDER` for Q5.
- artifacts:
  - p512 focused correctness/timing:
    `cache/hrxv1/gfx1151/focused-moe-q5-id-mmq16-p512-20260617-162033/`.
  - p33/p513 focused correctness/timing:
    `cache/hrxv1/gfx1151/focused-moe-q5-id-mmq16-odd-20260617-162104/`.
  - p512 same-runner A/B:
    `cache/hrxv1/gfx1151/q5-id-mmq16-model-ab-p512-20260617-162146/`.
  - p33/p513 same-runner A/B:
    `cache/hrxv1/gfx1151/q5-id-mmq16-model-ab-odd-20260617-162218/`.
  - repeated current-best HRX:
    `cache/hrxv1/gfx1151/current-best-with-q4q5-id-r3-20260617-162308/`.
  - fresh Vulkan comparison:
    `cache/hrxv1/gfx1151/current-best-q4q5-id-vulkan-r3-20260617-162608/`.
- correctness:
  focused CPU-reference gates passed for p33, p512, and p513. Route traces
  prove the Q5 provider selected for the Q5 row and the Q4 provider selected
  for the two Q4 rows.
- compile evidence:
  built HSACO reports wave64, VGPR 119, SGPR 87, LDS 3264 bytes, no spills.
  Disassembly count: 128 `v_dot`, 2 `s_barrier`, 67 `global_load`, 90 `ds_*`,
  and 183 `s_waitcnt` lines.
- focused timing:

  ```text
  shape   Q5 MoE ID focused time
  p33       261 us
  p512     2246 us
  p513     2275 us
  ```

- model timing:

  ```text
  shape   q4-id HRX   q4+q5-id HRX   lift    fresh Vulkan r3   HRX/Vulkan
  p33       185.55        198.30      1.07x        219.06        0.905x
  p512      476.79        597.23      1.25x       1130.20        0.528x
  p513      418.16        514.55      1.23x        951.86        0.541x
  ```

- decision:
  accept as an opt-in gfx1151 candidate. Do not default broadly until the
  broader model basket has been covered and target-specific policy is in place.
- notes:
  This eliminates the previous Q5_K `MUL_MAT_ID` unsupported gap for the active
  Qwen3 MoE p33/p512/p513 shapes. Remaining parity gap is now less likely to be
  MoE ID and should be re-ranked from route/device evidence before the next
  kernel candidate.

## 2026-06-17 - gfx1151 telemetry note for unified memory

- source/build/model:
  documentation update only; applies to all gfx1151 HRX v1 measurements in
  this workspace.
- note:
  use `amd-smi`, not deprecated `rocm-smi`, for GPU process and utilization
  sanity checks. This machine is a unified-memory GPU, so memory utilization
  can mean residency, pressure, cached or mapped unified-memory state, or other
  runtime allocation state. Do not treat the memory number as a standalone
  equivalent of discrete-card VRAM consumption.
- operational rule:
  `amd-smi` process residency and fluctuating utilization are useful to catch
  stale benchmark processes, obvious leaks, or unexpected idle/active state.
  They are not promotion evidence by themselves. Kernel and route decisions
  still require route traces, focused CPU-reference gates, focused timing,
  profile/perf buckets where available, and same-runner HRX/Vulkan A/B rows.

## 2026-06-17 - current-best p512 boulder reranking after q4/q5 moe id

- source/build/model:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, dirty with the split HRX v1
  catalog and current opt-in stack. HRX build
  `build/hrx-v1-catalog-gfx1151`, Vulkan build `build/vulkan-gfx1151`,
  ROCm `/srv/vm-shared/rocm/rocm-head`, Qwen3 30B Q4_K_XL `p512/n0/fa0`.
- route or candidate:
  ranking checkpoint only. Current HRX stack:
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`,
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`,
  `GGML_HRX_ENABLE_F16_BATCHED_ROWS2_COLS16_WG32_PROMPT=1`,
  `GGML_HRX_ENABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`, and
  `GGML_HRX_ENABLE_Q5_K_ID_Q8_1_X4_MMQ16_PROMPT=1`.
- baseline/variant command:
  no new model A/B in this entry. Evidence is from the current-best repeated
  HRX/Vulkan artifact plus two focused checks:
  `llama-bench -m <Qwen3 Q4_K_XL> -p 512 -n 0 -b 1024 -ub 1024 -fa 0 -r 3 -o json --no-warmup -ngl 99 -dev HRX0|Vulkan0`;
  `test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <qk_decode.txt>`;
  `test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <attention_f16_prompt.txt>`.
- artifacts:
  - current repeated HRX/Vulkan:
    `cache/hrxv1/gfx1151/current-best-q4q5-id-vulkan-r3-20260617-162608/`.
  - invalid HRX dispatch profile:
    `cache/hrxv1/gfx1151/profile-q4q5-id-p512-20260617-163010/`.
  - current Q6/vocab cols1 focused gate:
    `cache/hrxv1/gfx1151/focused-q6-vocab-cols1-current-20260617-163546/`.
  - current F16 attention focused gate:
    `cache/hrxv1/gfx1151/focused-f16-attention-current-20260617-163655/`.
- correctness:
  Q6/vocab cols1 focused `MUL_MAT` rows passed 8/8 CPU-reference tests.
  F16 attention focused `MUL_MAT` rows passed 4/4 CPU-reference tests.
  Current-best model JSON reports `backends=HRX` for HRX and `backends=Vulkan`
  for Vulkan.
- timing/evidence:

  ```text
  model p512 current best:
    HRX avg_ns       857.386 ms, 597.233 tok/s
    Vulkan avg_ns    453.195 ms, 1130.195 tok/s
    HRX/Vulkan       0.528x

  HRX route counts over r3:
    480 q4 dense prompt MMQ64
    288 F16 attention rows2-cols16
    141 F32 MoE logits
    141 GET_ROWS
    102 Q4 MoE ID grouped MMQ16
     39 Q5 MoE ID grouped MMQ16
     30 Q6 dense prompt MMQ64
      3 Q6 vocab/result cols1

  Vulkan p512 largest labels:
    Q4 MoE ID down        178.4 ms total
    Q4 MoE ID gate         68.9 ms total
    Q5 MoE ID down         29.0 ms total
    SOFT_MAX               23.1 ms total
    dense Q4 prompt        43.8 ms total across three Q4 labels
    F16 attention          32.2 ms total across two labels
    Q6 vocab/result         1.13 ms

  focused Q6/vocab cols1:
    result_output q6_K m=151936 n=1 k=2048: 1113.25 us, passed
  ```

- decision:
  the current Q6 vocab/result cols1 route is not a boulder; it matches the
  Vulkan label closely. The current MoE ID path is also no longer the obvious
  route gap: focused Q4/Q5 ID timings are near the Vulkan labels after the
  grouped Q8_1 x4 candidates. HRX profile dispatch records are unavailable in
  this runtime (`dispatch_events=0`), so do not use that profile as evidence.
  The next implementation target should be a bounded dense Q4/Q6 prompt
  matmul schedule pivot from the existing prompt ledger, with F16 attention
  treated as a secondary target because the exported focused rows are useful
  for correctness/relative routing but not trustworthy as absolute model-time
  estimates.
- notes:
  The remaining p512 gap is about 404 ms per repetition. Known focused/model
  evidence rules out Q6 vocab and deprioritizes MoE ID. Dense prompt matmuls
  still have large route counts and measurable Vulkan gaps, and the ledger
  already contains HRX/Vulkan prior schedule facts for a focused candidate.

## 2026-06-17 - q4 dense prompt mmql128 wide-route candidate

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current Q4/Q6/F16/Q4-ID/Q5-ID opt-ins, and new Q4 MMQL128
  source/provider/catalog entries.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`, `fa=0`, `n=0`,
  `b=1024`, `ub=1024`, focused p33/p512/p513 Q4/Q5/Q6 prompt rows and
  same-binary model A/B at p33/p512/p513.
- route or candidate:
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_wg256_f32`, opt-in
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`, guarded at `cols >= 128`.
- baseline command:
  current best HRX stack with Q4 MMQ64, Q6 MMQ64, F16 rows2-cols16, Q4 ID
  wide-K, and Q5 ID opt-ins.
- variant command:
  same command plus `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-mmql128-threshold-route-sanity-20260617-165126/`;
  p33 selected Q4 MMQ64, p512 selected Q4 MMQL128.
- profile/timing:
  - p512 focused:
    `cache/hrxv1/gfx1151/focused-q4-mmql128-p512-20260617-164724/`.
  - odd focused:
    `cache/hrxv1/gfx1151/focused-q4-mmql128-odd-20260617-164835/`.
  - old-route odd comparison:
    `cache/hrxv1/gfx1151/focused-q4-mmq64-odd-compare-20260617-164950/`.
  - model A/B:
    `cache/hrxv1/gfx1151/q4-mmql128-model-ab-20260617-165235/`.
- correctness:
  focused CPU-reference gates passed for p512, p33, and p513. Model JSON
  reported `backends=HRX` for all rows; no fallback/error strings were found.
- timing:
  focused Q4 p512 rows: `Vcur-0 202.02 -> 198.59 us`, `node_32 1304.93 ->
  732.82 us`, `Qcur-0 1193.69 -> 696.78 us`. Odd focused A/B showed p33
  regresses on MMQL128, while p513 improves on the large Q4 rows. Same-binary
  model A/B: `p33 205.259 -> 201.773 tok/s`, `p512 601.779 -> 635.944 tok/s`,
  `p513 515.150 -> 539.050 tok/s`.
- decision:
  accepted as a gfx1151 opt-in wide-prefill candidate. Keep `cols >= 128`;
  p33 must remain on Q4 MMQ64.
- notes:
  This is the first Q4 dense prompt schedule that directly adopts the Vulkan
  AMD 128x128 K-quant MMQ shape. It improves production-width prefill but does
  not close the remaining Vulkan gap by itself.

## 2026-06-17 - q6 dense prompt focused schedule rejections

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current opt-ins, and new Q6 MMQL128x128/MMQ64x128 candidate routes.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  HIP kernels were built through CMake/Ninja.
- model/shape:
  `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`, focused p512 Q4/Q5/Q6 prompt
  rows from the exported model graph, especially Q6 `node_372` and
  `result_output`.
- route or candidate:
  - existing staged Q6 MMQL64x128/MMQL128x64 routes;
  - new `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x128_wg256_f32`, opt-in
    `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1`;
  - new `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x128_wg256_f32`, opt-in
    `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64X128_PROMPT=1`.
- baseline command:
  current best HRX stack with Q6 direct MMQ64 opt-in:
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1 test-backend-ops test/perf
  -b HRX0 -o MUL_MAT --test-file <qk_prompt.txt>`.
- variant command:
  same focused `test-backend-ops` commands with the staged Q6 route selected,
  the Q6 MMQL128x128 opt-in, or the Q6 MMQ64x128 opt-in.
- route trace:
  route stderr in:
  `cache/hrxv1/gfx1151/focused-q6-variant-rerank-p512-20260617-170020/`,
  `cache/hrxv1/gfx1151/focused-q6-mmql128-p512-20260617-170743/`, and
  `cache/hrxv1/gfx1151/focused-q6-mmq64x128-p512-20260617-171056/`.
- profile/timing:
  same artifacts as route trace; timing is in each `perf.csv`.
- correctness:
  all focused CPU-reference gates passed for the tested p512 rows.
- timing:
  p512 `result_output` timings:
  - current direct MMQ64: `94.75 ms`;
  - existing staged MMQL128x64: `440.73 ms`;
  - existing staged MMQL64x128: `468.63 ms`;
  - new staged MMQL128x128: `281.16 ms`;
  - new direct MMQ64x128: `194.44 ms`.
- decision:
  reject all tested Q6 replacements for promotion. Keep the current direct
  MMQ64 route as the best available Q6 p512 candidate.
- notes:
  The Vulkan large int-K prior is still useful, but the HRX Q6 staged
  implementations are resource/loop-shape limited on gfx1151. A wider direct
  tile also loses, so the next Q6 attempt should inspect emitted ISA/resource
  use and consider a new lane ownership or two-stage schedule rather than
  simply widening columns or copying the Q4 staged tile.

## 2026-06-17 - d128 flash-attention prefill direct route

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current best opt-ins, separate D128 route
  `hrx_flash_attn_ext_f32_f16_prefill_direct_d128`, and route tracing added for
  `FLASH_ATTN_EXT`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  HIP kernels were built through CMake/Ninja.
- model/shape:
  `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`, `fa=1`, `n=0`,
  `b=1024`, `ub=1024`, p33/p512/p513 prompt rows. The live model flash shape
  is `D=128`, `H=32`, `H_KV=4`; active KV was 512 for p512, 256 for p33, and
  768 for p513.
- route or candidate:
  `hrx_flash_attn_ext_f32_f16_prefill_direct_d128`, opt-in
  `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`.
- baseline command:
  current best HRX stack with `-fa 1` and without the D128 prefill-direct
  opt-in. Route tracing showed the baseline used the generic decode provider
  for prompt flash rows.
- variant command:
  same command plus `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`.
- route trace:
  `cache/hrxv1/gfx1151/fa1-direct-d128-traced-p512-20260617-173115/` and
  `cache/hrxv1/gfx1151/fa1-direct-d128-model-ab-odd-20260617-173548/`.
- profile/timing:
  - FA-on HRX/Vulkan probe:
    `cache/hrxv1/gfx1151/fa1-current-best-p512-20260617-171632/`;
  - focused p512 gate:
    `cache/hrxv1/gfx1151/focused-fa1-direct-d128-f32pv-p512-20260617-172350/`;
  - focused odd gates:
    `cache/hrxv1/gfx1151/focused-fa1-direct-d128-p33-20260617-173253/` and
    `cache/hrxv1/gfx1151/focused-fa1-direct-d128-p513-20260617-173313/`;
  - split-provider focused regate:
    `cache/hrxv1/gfx1151/fa1-direct-d128-split-regate-20260617-174250/`;
  - model p512 A/B:
    `cache/hrxv1/gfx1151/fa1-direct-d128-model-ab-p512-20260617-172749/`;
  - traced p512 rerun:
    `cache/hrxv1/gfx1151/fa1-direct-d128-traced-p512-20260617-173115/`;
  - split-provider traced p512 rerun:
    `cache/hrxv1/gfx1151/fa1-direct-d128-split-traced-p512-20260617-174708/`;
  - model odd A/B and Vulkan comparison:
    `cache/hrxv1/gfx1151/fa1-direct-d128-model-ab-odd-20260617-173548/`.
- correctness:
  Focused CPU-reference gates passed for p33, p512, and p513 after switching
  the D128 value accumulation to f32. The initial f16 PV accumulation failed
  tolerance narrowly (`ERR ~= 0.0077-0.0082`). After splitting the D128 route
  into its own HSACO/export, the p33/p512/p513 focused gates passed again.
- timing:
  - p33: HRX `203.072 -> 213.973 tok/s`; Vulkan FA-on `228.693 tok/s`;
    variant is `0.936x` Vulkan.
  - p512: traced split-provider HRX variant `888.692 tok/s`; prior HRX FA-on baseline
    `463.927 tok/s`; Vulkan FA-on `1266.645 tok/s`; variant is about `0.702x`
    Vulkan.
  - p513: HRX `448.089 -> 853.656 tok/s`; Vulkan FA-on `1193.498 tok/s`;
    variant is `0.715x` Vulkan.
- decision:
  accept as a gfx1151 opt-in candidate for D128/H32/HKV4 flash-attention
  prefill. Keep it target-gated and opt-in until broader basket coverage and
  target-specific default policy are added.
- notes:
  This is a structural fusion win, not an aggregate-only tuning result:
  route traces prove live prompt `FLASH_ATTN_EXT` dispatches moved from the
  generic decode kernel to the D128 prefill-direct kernel. The remaining
  p512/p513 gap to Vulkan is still large enough to require kernel/schedule A+B:
  compare the direct route against Vulkan flash-attention lane ownership,
  staging, mask handling, and value accumulation policy before attempting
  broad defaulting. On this UMA GPU, AMD-SMI residency/memory numbers were
  treated only as sanity signals; decisions used route traces, focused gates,
  and same-runner JSON timing.

## 2026-06-17 - downloaded basket p512 fa1 current-best ranking

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current best opt-ins, and split D128 flash route before broad GQA
  predicate generalization.
- build:
  HRX build `build/hrx-v1-catalog-gfx1151`; Vulkan build
  `build/vulkan-gfx1151`; ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  all eight downloaded GGUFs under
  `shared/models/llamacpp-hrx2-basket-v1`, `p512`, `n0`, `fa=1`,
  `b=1024`, `ub=1024`, one repetition.
- route or candidate:
  current best opt-in HRX stack:
  Q4/Q6 dense prompt, F16 rows2-cols16, Q4/Q5 MoE ID, Q4 MMQL128, and D128
  flash prefill-direct for the previously accepted H32/HKV4 shape.
- baseline command:
  sequential HRX and Vulkan `llama-bench -p 512 -n 0 -fa 1 -r 1 -o json`.
- variant command:
  none; this is boulder-ranking evidence.
- route trace:
  `cache/hrxv1/gfx1151/current-best-fa1-basket-p512-r1-20260617-175043/`.
- profile/timing:
  same artifact, especially `summary.csv` and per-model `stderr.log`.
- correctness:
  all rows completed with backend JSON reporting `HRX` or `Vulkan`; no
  fallback/error strings were found by the summary script.
- timing:
  HRX/Vulkan ratios:
  - Qwen2.5 Coder 7B Q5_K_M: `0.314x`;
  - Llama 3.2 3B Q4_K_M: `0.287x`;
  - Llama 3.1 8B Q4_K_M: `0.288x`;
  - Llama 3.1 8B Q8_0: `0.192x`;
  - DeepSeek R1 Distill Qwen 14B Q4_K_M: `0.291x`;
  - Qwen3 30B Q6_K: `0.210x`;
  - Qwen3 30B Q4_K_XL: `0.716x`;
  - Qwen3 Coder 30B Q4_K_M: `0.476x`.
- decision:
  use this as the next boulder ranking. The non-Qwen3-Q4XL rows were far from
  parity and route traces showed their prompt `FLASH_ATTN_EXT` rows still used
  the generic decode provider.
- notes:
  The active Qwen3 H32/HKV4 flash route was not enough for the broader basket.
  The next candidate should test whether the split D128 prefill-direct kernel
  can safely cover other D128 GQA shapes before spending effort on unrelated
  matmul knobs.

## 2026-06-17 - generalized d128 flash prefill route for gqa shapes

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog and D128 flash route predicate broadened from exact H32/HKV4 to D128
  valid-GQA shapes at production prompt widths.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  downloaded basket p512/p33/p513 `FLASH_ATTN_EXT` rows with D128 and GQA
  shapes H24/HKV8, H28/HKV4, H32/HKV8, H40/HKV8, plus the prior H32/HKV4.
- route or candidate:
  `hrx_flash_attn_ext_f32_f16_prefill_direct_d128`, opt-in
  `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`.
- baseline command:
  current best HRX stack without `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128`;
  newly-covered shapes route prompt FA through
  `hrx_flash_attn_ext_f32_f16_decode`.
- variant command:
  same command with `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`.
- route trace:
  - p512 focused:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-focused-p512-20260617-175339/`;
  - p512 model A/B:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-model-ab-p512-r3-20260617-175433/`;
  - p33/p513 focused:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-focused-odd-tail-20260617-175638/`;
  - p33/p513 model smoke:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-model-ab-odd-tail-r1-20260617-175720/`;
  - final policy sanity:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-policy-sanity-20260617-175858/`.
- profile/timing:
  same artifacts; timing is from `llama-bench` JSON and focused CSV rows.
- correctness:
  focused CPU-reference gates passed for p512 and p33/p513 across H24/HKV8,
  H28/HKV4, H32/HKV8, H40/HKV8, and H32/HKV4 representative rows.
- timing:
  p512 r3 HRX model A/B:
  - Qwen2.5 Coder 7B Q5_K_M: `356.272 -> 448.416 tok/s` (`1.259x`);
  - Llama 3.2 3B Q4_K_M: `737.653 -> 1199.245 tok/s` (`1.626x`);
  - Llama 3.1 8B Q4_K_M: `342.519 -> 465.632 tok/s` (`1.359x`);
  - Llama 3.1 8B Q8_0: `176.320 -> 205.280 tok/s` (`1.164x`);
  - DeepSeek R1 Qwen 14B Q4_K_M: `177.517 -> 239.097 tok/s` (`1.347x`);
  - Qwen3 30B Q4_K_XL no-regression row: `450.486 -> 865.318 tok/s`.

  Odd/tail model smoke found p513 wins (`1.221x-1.596x`) and mostly p33 wins,
  but Llama 3.2 3B p33 regressed slightly (`403.660 -> 398.019 tok/s`).
- decision:
  accept broad D128 GQA coverage only for production-width prompt shapes
  `N >= 128`, while preserving the previously accepted H32/HKV4 behavior.
  The final policy sanity confirms H24 p33 stays on decode, H24 p513 routes to
  D128 prefill-direct, and H28 p512 routes to D128 prefill-direct.
- notes:
  This is a selector/policy promotion for an already CMake-built candidate
  kernel. It materially improves the broader basket but does not reach Vulkan:
  after the win, p512 ratios remain about `0.22x-0.47x` on many non-Qwen3 rows.
  The next boulders are model-specific prompt matmul quality, especially Q8_0
  and non-MoE Q4/Q5/Q6 prompt routes, plus Qwen3 Q6 where attention is no
  longer the obvious blocker.

## 2026-06-17 - q8_0 prompt rows1024 mmq128x32 auto-policy

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, generalized D128 flash attention, and the
  Q8_0 auto-policy threshold change from 2048 rows to 1024 rows.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`;
  p33, p512, and p513 Q8_0 prompt `MUL_MAT` rows, especially K/V projection
  shape `k=4096, rows=1024, cols=33/512/513`.
- route or candidate:
  existing packed route
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`; policy candidate lowers
  `GGML_HRX_Q8_1_MMVQ_AUTO_Q8_0_ROWS_MIN` to 1024 while keeping `cols >= 32`.
- baseline command:
  current best HRX stack with
  `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`, default Q8_0 auto policy.
- variant command:
  same commands with `GGML_HRX_Q8_1_MMVQ=all` for evidence; final source
  change makes the rows1024 case automatic without requiring the env override.
- route trace:
  - focused p512:
    `cache/hrxv1/gfx1151/q8_0-kv1024-focused-20260617-180540/`;
  - focused p33/p513:
    `cache/hrxv1/gfx1151/q8_0-mmvq1024-odd-tail-focused-20260617-181012/`;
  - p512 model A/B:
    `cache/hrxv1/gfx1151/q8_0-mmvq1024-model-ab-p512-r3-20260617-180920/`;
  - p33/p513 model A/B:
    `cache/hrxv1/gfx1151/q8_0-mmvq1024-model-ab-odd-tail-r3-20260617-181206/`;
  - post-edit focused auto-regate:
    `cache/hrxv1/gfx1151/q8_0-mmvq1024-auto-regate-20260617-181433/`;
  - post-edit p512 model sanity:
    `cache/hrxv1/gfx1151/q8_0-mmvq1024-auto-model-p512-r3-20260617-181455/`.
- profile/timing:
  same artifacts; focused timing from `test-backend-ops perf --output csv`,
  model timing from `llama-bench -r 3 -o json --no-warmup`.
- correctness:
  focused CPU-reference gates passed for p33, p512, and p513 exported Q8_0
  prompt rows under the forced packed route. Route traces proved the rows1024
  K/V projections moved from `hrx_mul_mat_vec_q8_0_cols8_f32` to
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`.
- timing:
  focused rows1024 timing:
  - p33 Vcur: `164.996 -> 112.301 us`;
  - p512 Vcur: `2110.487 -> 545.849 us`;
  - p513 Vcur: `2170.293 -> 595.074 us`.

  Post-edit auto-regate, without `GGML_HRX_Q8_1_MMVQ=all`, selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32` for rows1024 and measured
  `545.687 us`.

  Model A/B:
  - p33: `112.958 -> 113.504 tok/s` (`1.005x`);
  - p512: `201.407 -> 218.959 tok/s` (`1.087x`);
  - p513: `191.983 -> 200.620 tok/s` (`1.045x`).

  Post-edit p512 model sanity, without the forcing env, measured
  `209.990 +/- 3.444 tok/s` and route traces showed K/V rows1024 on the packed
  route.
- decision:
  accept the gfx1151 Q8_0 rows1024 auto-policy update. This is a selector
  policy correction for an existing CMake-built HIP kernel, not a new schedule.
  Keep the `cols >= 32` guard because prior p31 forced-MMVQ evidence rejected
  broad narrow-column relaxation.
- notes:
  The full Q8_0 row remains far from Vulkan even after this win, so this is a
  useful local correction rather than parity. On this unified-memory GPU,
  AMD-SMI process/residency checks were used only to confirm no competing GPU
  jobs were active; they were not used as tuning evidence.

## 2026-06-17 - q6_k moe mul_mat_id grouped q8_1 x4 mmq16

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, and new Q6_K `MUL_MAT_ID` candidate wiring.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  Built through:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx llama-bench test-backend-ops export-graph-ops -j$(nproc)`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf`;
  Q6_K MoE `MUL_MAT_ID` prompt rows at p33, p512, and p513 with `fa=1`.
- route or candidate:
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`, opt-in
  `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1`.
- baseline command:
  current best HRX stack without `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT`;
  p512 command shape:
  `llama-bench -m <Q6_K model> -p 512 -n 0 -b 1024 -ub 1024 -fa 1 -r 3 -o json --no-warmup -ngl 99 -dev HRX0`.
- variant command:
  same commands with `GGML_HRX_ENABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1`.
- route trace:
  - focused p512 support/correctness/perf:
    `cache/hrxv1/gfx1151/q6-mul-mat-id-q8x4-focused-20260617-183451/`;
  - p512 model A/B:
    `cache/hrxv1/gfx1151/q6-id-model-ab-p512-fa1-r3-20260617-183602/`;
  - p33/p513 focused odd-tail:
    `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/`;
  - p33/p513 model A/B:
    `cache/hrxv1/gfx1151/q6-id-model-ab-odd-tail-r3-20260617-183942/`;
  - post-guard focused sanity:
    `cache/hrxv1/gfx1151/q6-id-postguard-focused-20260617-184103/`.
- profile/timing:
  same artifacts; focused timing from `test-backend-ops perf --output csv`,
  model timing from `llama-bench -r 3 -o json --no-warmup`.
- correctness:
  default support stayed off. With the opt-in gate, p512 focused
  CPU-reference correctness passed for the exported Q6_K `ffn_moe_gate` and
  `ffn_moe_down` rows. Focused p33 and p513 correctness also passed before
  policy tightening. After tightening, p33 support is rejected and p513 still
  selects the Q6 ID provider and passes correctness.
- timing:
  focused p512 Q6 ID rows:
  - `ffn_moe_gate`: `2450.596 us`;
  - `ffn_moe_down`: `2323.978 us`.

  Same-runner model A/B:
  - p512: `217.797 -> 509.261 tok/s` (`2.338x`);
  - p513: `220.570 -> 483.857 tok/s` (`2.194x`);
  - p33: `123.074 -> 108.935 tok/s` (`0.885x`, rejected).
- decision:
  accept as an opt-in production-width gfx1151 candidate with guard
  `q6_K MUL_MAT_ID, k % 256 == 0, rows % 64 == 0, n_ids == 8,
  n_tokens >= 128`. Do not enable for narrow p33 prompt; the runtime selector
  and catalog metadata were tightened from `n_tokens >= 32` to
  `n_tokens >= 128`.
- notes:
  The route closes a structural Qwen3 Q6_K MoE coverage gap. It follows the
  accepted Q5_K grouped Q8_1 x4 MMQ16 schedule family and reuses the same C++
  route compaction plus Q8_1 x4 quantization path. Aggregate p512 basket
  ranking identified the boulder, but the decision came from focused support,
  CPU-reference correctness, focused timing, route traces, and same-runner
  model A/B. Vulkan parity for the Qwen3 Q6_K p512 row is still not reached;
  the latest variant is about `0.52x` the previous same-machine Vulkan p512
  row, so further dense Q6 prompt and residual MoE/attention schedule work
  remains.

## 2026-06-17 - q8_0 prompt mmq128x32 wave64 probe

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, and new Q8_0 wave64 wrapper source.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  Built through:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops llama-bench -j$(nproc)`.
  Catalog validation passed with
  `validate_hrx_catalog.py --catalog build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hrx_catalog.json`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`;
  exported p512/fa1 Q8_0 prompt rows:
  `Vcur-0`, `Qcur-0`, `ffn_out-0`, `ffn_gate-0`, and `result_output`.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_wave64_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X32_WAVE64_PROMPT=1`.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file cache/hrxv1/gfx1151/q8_0-current-focused-p512-20260617-184658/focused/q8_0_prompt.txt --output csv`.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X32_WAVE64_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-wave64-focused-p512-20260617-185605/`.
- profile/timing:
  same artifact; focused timing from `test-backend-ops perf --output csv`.
  Built HSACO metadata confirms the new route is wave64 with `vgpr_count=122`,
  no spills, and `group_segment_fixed_size=1088`.
- correctness:
  focused CPU-reference correctness passed for all five rows. Route traces
  proved the baseline selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32` and the variant selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_wave64_f32`.
- timing:
  focused p512 timing, baseline -> wave64:
  - `Vcur-0`: `524.019 -> 543.355 us`;
  - `Qcur-0`: `2003.890 -> 2709.892 us`;
  - `ffn_out-0`: `12100.674 -> 18948.767 us`;
  - `ffn_gate-0`: `21138.738 -> 35776.988 us`;
  - `result_output`: `203806.524 -> 263176.429 us`.
- decision:
  reject the wave64-only compile probe. Do not run model A/B or odd/tail gates
  because focused p512 perf regressed every row, including the hot FFN rows.
  The route remains opt-in only as a reproducible rejected candidate and is
  recorded in the gfx1151 rejections database.
- notes:
The Q8_0 gap is not just a wavefront-size mismatch. The next Q8_0 candidate
should alter tile shape/output ownership and likely A staging toward the
Vulkan integer-MMQ dataflow. Schedule facts and next candidate axes are in
`docs/hrxv1/q8_0-prompt-schedule-ledger.md`.

## 2026-06-17 - q8_0 prompt bm64 bn64 tile probe

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, and new Q8_0 BM64/BN64 wrapper source.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  Built through:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops llama-bench -j$(nproc)`.
  Generated catalog validation passed with
  `validate_hrx_catalog.py --catalog build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hrx_catalog.json`.
- model/shape:
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`;
  exported Q8_0 prompt rows at p33, p512, and p513, plus same-run
  `llama-bench` p33/p512/p513 with `fa=1`.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- baseline command:
  focused:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <q8_0 prompt file> --output csv`;
  model:
  current best HRX opt-in stack without the BM64/BN64 gate, using
  `llama-bench -m <Q8_0 model> -p 33|512|513 -n 0 -b 512 -ub 512 -fa 1 -r 3 -o json --no-warmup -ngl 99 -dev HRX0`.
- variant command:
  same focused and model commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- route trace:
  - focused p512:
    `cache/hrxv1/gfx1151/q8_0-mmq64x64-focused-p512-20260617-190744/`;
  - focused p33/p513:
    `cache/hrxv1/gfx1151/q8_0-mmq64x64-focused-p512-20260617-190744/odd-tail/`;
  - model A/B and Vulkan comparison:
    `cache/hrxv1/gfx1151/q8_0-mmq64x64-model-ab-20260617-191149/`.
- profile/timing:
  same artifacts; focused timing from `test-backend-ops perf --output csv`,
  model timing from `llama-bench -r 3 -o json --no-warmup`, and Vulkan labels
  from `GGML_VK_PERF_LOGGER=1`.
- correctness:
  focused CPU-reference correctness passed for all p33, p512, and p513 rows.
  Route traces proved baseline selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32` and the variant selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32`.
- timing:
  focused totals over the five exported rows:
  - p33: `35438.8 -> 11564.9 us` (`3.06x`);
  - p512: `239194.7 -> 104160.8 us` (`2.30x`);
  - p513: `254447.9 -> 120127.6 us` (`2.12x`).

  Same-runner model A/B with same-run Vulkan:
  - p33: HRX `92.317 -> 194.470 tok/s`; Vulkan `196.622 tok/s`;
  - p512: HRX `209.715 -> 394.224 tok/s`; Vulkan `884.048 tok/s`;
  - p513: HRX `206.993 -> 377.980 tok/s`; Vulkan `837.168 tok/s`.
- decision:
  accept BM64/BN64 as a gfx1151 opt-in candidate. Do not make it an unguarded
  default yet; target-specific policy plumbing and broader basket coverage are
  still pending.
- notes:
  This validates the first schedule-led Q8_0 pivot after the wave64 rejection:
  changing tile/output ownership matters. The candidate nearly reaches Vulkan
  at narrow p33 but production-width p512/p513 remain about `0.45x` Vulkan, so
  the next Q8_0 axis should mine Vulkan's remaining schedule differences,
  especially cooperative A staging and smaller per-lane output/register tiles.

## 2026-06-17 - q8_0 prompt bm64 bn64 cooperative a+b staging probe

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, and new Q8_0 BM64/BN64 A+B-staged wrapper
  source.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  Built through:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops llama-bench -j$(nproc)`.
- model/shape:
  Llama 3.1 8B Q8_0 exported p512/fa1 prompt rows:
  `Vcur-0`, `Qcur-0`, `ffn_out-0`, `ffn_gate-0`, and `result_output`.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_ab_wg256_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_AB_PROMPT=1`.
- baseline command:
  direct-A BM64/BN64 focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_AB_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq64x64-ab-focused-p512-20260617-191938/`.
- profile/timing:
  same artifact; focused timing from `test-backend-ops perf --output csv`.
  HSACO metadata for the public route is wave32, `vgpr_count=123`,
  `sgpr_count=32`, LDS `4352` bytes, and no spills.
- correctness:
  focused CPU-reference correctness passed for all five p512 rows. Route traces
  proved baseline selected the direct-A BM64/BN64 provider and variant selected
  the A+B-staged provider.
- timing:
  focused p512 timing, direct BM64/BN64 -> A+B-staged BM64/BN64:
  - `Vcur-0`: `566.437 -> 684.076 us`;
  - `Qcur-0`: `1939.990 -> 2326.206 us`;
  - `ffn_out-0`: `9800.640 -> 11385.032 us`;
  - `ffn_gate-0`: `8739.343 -> 9634.424 us`;
  - `result_output`: `83885.357 -> 91222.690 us`;
  - total: `104931.8 -> 115252.4 us` (`0.91x`).
- decision:
  reject the naive A+B LDS staging probe. Do not run model A/B or odd/tail
  gates because focused p512 perf regressed every row.
- notes:
  The result narrows the next Q8_0 search: copying Vulkan's A-cache reuse
  literally into the current one-row-per-thread and sixteen-columns-per-thread
  HRX schedule is not enough. Any A-reuse retry should change the register tile
  and per-lane output ownership at the same time.

## 2026-06-17 - q8_0 prompt bm32 bn64 smaller output tile probe

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, and new Q8_0 BM32/BN64 wrapper source.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  Built through:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops llama-bench -j$(nproc)`.
- model/shape:
  Llama 3.1 8B Q8_0 exported p512/fa1 prompt rows:
  `Vcur-0`, `Qcur-0`, `ffn_out-0`, `ffn_gate-0`, and `result_output`.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq32x64_wg256_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ32X64_PROMPT=1`.
- baseline command:
  accepted BM64/BN64 focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ32X64_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq32x64-focused-p512-20260617-192604/`.
- profile/timing:
  same artifact; focused timing from `test-backend-ops perf --output csv`.
  HSACO metadata for the public route is wave32, `vgpr_count=78`,
  `sgpr_count=22`, LDS `2176` bytes, and no spills.
- correctness:
  focused CPU-reference correctness passed for all five p512 rows. Route traces
  proved baseline selected BM64/BN64 and variant selected BM32/BN64.
- timing:
  focused p512 timing, BM64/BN64 -> BM32/BN64:
  - `Vcur-0`: `568.529 -> 660.519 us`;
  - `Qcur-0`: `2817.721 -> 2383.890 us`;
  - `ffn_out-0`: `9915.608 -> 12187.366 us`;
  - `ffn_gate-0`: `9046.741 -> 10254.701 us`;
  - `result_output`: `83383.405 -> 91489.524 us`;
  - total: `105732.0 -> 116976.0 us` (`0.90x`).
- decision:
  reject BM32/BN64. Do not run model A/B or odd/tail gates because focused
  p512 perf regressed total and hot rows.
- notes:
  Lower VGPR count alone did not win. The row-tile amortization loss from
  doubling row workgroups dominates this direct-A/staged-B spelling.

## 2026-06-17 - q8_0 prompt bm128 bn64 compile-resource probe

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, and new Q8_0 BM128/BN64 wrapper source.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  Built through:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops llama-bench -j$(nproc)`.
- model/shape:
  Llama 3.1 8B Q8_0 p512 prompt rows were the intended focused target, but
  the candidate was rejected before runtime.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x64_wg256_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X64_PROMPT=1`.
- baseline command:
  not run; compile metadata was sufficient to reject.
- variant command:
  not run; compile metadata was sufficient to reject.
- route trace:
  not captured.
- profile/timing:
  compile metadata from
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq128x64.hsaco`.
- correctness:
  not run.
- timing:
  not run. HSACO metadata for the public route: wave32, `vgpr_count=192`,
  `vgpr_spill_count=47`, private segment 192 bytes, LDS `2176` bytes.
- decision:
  reject before focused runtime testing. Do not run model A/B or odd/tail
  gates for a spilling candidate.
- notes:
  Simple column widening is too register-heavy in the direct-A/staged-B
  spelling. Any future widening needs a different register tile or loop shape.

## 2026-06-17 - current-best basket after q8_0 bm64 bn64

- source:
  `sources/llama.cpp` `0668b9ee54e6` on `hrx-kernel-lib-v1`, dirty with split
  catalog, current gfx1151 opt-ins, accepted Q8_0 BM64/BN64 candidate, and
  rejected Q8_0 follow-up probes.
- build:
  - HRX: `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
    `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
  - Vulkan: `build/vulkan-gfx1151`, same-source Vulkan baseline.
- model/shape:
  eight downloaded basket rows, `p512/n0/fa1/b1024/ub1024/r1`.
- route or candidate:
  current-best gfx1151 opt-in stack, including
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- baseline command:
  per-model Vulkan `llama-bench -p 512 -n 0 -b 1024 -ub 1024 -fa 1 -r 1
  -o json --no-warmup -ngl 99 -dev Vulkan0`.
- variant command:
  per-model HRX `llama-bench -p 512 -n 0 -b 1024 -ub 1024 -fa 1 -r 1
  -o json --no-warmup -ngl 99 -dev HRX0` with the current-best opt-ins.
- route trace:
  `cache/hrxv1/gfx1151/current-best-q8bm64-basket-p512-fa1-r1-20260617-193412/`.
- profile/timing:
  `cache/hrxv1/gfx1151/current-best-q8bm64-basket-p512-fa1-r1-20260617-193412/summary.json`
  and `summary.csv`.
- correctness:
  benchmark rows completed and JSON `backends` fields were checked as
  `HRX`/`Vulkan`. No focused CPU-reference gate was run for this basket-level
  comparison.
- timing:
  geomean HRX/Vulkan ratio across the eight rows was `0.489x`.
  - `llama3_1_8b_q4km`: HRX `468.195`, Vulkan `1185.611`, ratio `0.395`;
  - `qwen2_5_coder_7b_q5km`: HRX `458.712`, Vulkan `1159.724`, ratio `0.396`;
  - `deepseek_r1_qwen_14b_q4km`: HRX `251.811`, Vulkan `619.149`, ratio
    `0.407`;
  - `qwen3_30b_q6k`: HRX `434.913`, Vulkan `1019.239`, ratio `0.427`;
  - `llama3_1_8b_q8_0`: HRX `401.278`, Vulkan `892.507`, ratio `0.450`;
  - `llama3_2_3b_q4km`: HRX `1218.525`, Vulkan `2543.653`, ratio `0.479`;
  - `qwen3_30b_q4xl`: HRX `882.495`, Vulkan `1202.812`, ratio `0.734`;
  - `qwen3_coder_30b_q4km`: HRX `869.161`, Vulkan `1138.765`, ratio
    `0.763`.
- decision:
  use this as the current basket-ranking checkpoint. It is not sufficient for a
  new broad promotion because it is one repetition and no focused op gate was
  attached to the basket as a whole.
- notes:
  The accepted Q8_0 BM64/BN64 route is selected in the Q8_0 row, but the
  production-width Q8_0 model remains only `0.45x` Vulkan. The worst rows are
  now dense Q4_K/Q5_K/Q6_K prompt matmul families and related FFN output
  buckets, not an unsupported MoE-only blocker. Next work should mine Vulkan
  and HRX1 priors for those dense quantized prompt schedules before adding more
  candidates.

## 2026-06-17 - q8_0 direct wmma16 f16acc diagnostic

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, dirty with the opt-in
  `hrx_mul_mat_vec_q8_0_wmma16x16_f16acc_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  focused Llama 3.1 8B Q8_0 prompt rows: p512 plus synthetic p33 and p513
  odd/tail rows, `k=4096`, `rows=1024`.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_wmma16x16_f16acc_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_WMMA16_F16ACC_PROMPT=1`.
- baseline command:
  `test-backend-ops perf --test-file
  cache/hrxv1/gfx1151/q8_0-kv1024-focused-20260617-180540/q8_0_prompt_rows1024.txt
  --output csv` with default HRX routing.
- variant command:
  same command with `GGML_HRX_ENABLE_Q8_0_WMMA16_F16ACC_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-wmma16-f16acc-focused-20260617-204054/`.
  Correctness and perf traces selected
  `hrx_mul_mat_vec_q8_0_wmma16x16_f16acc_f32`; default perf selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`.
- profile/timing:
  HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16.hsaco`.
  The f16acc export emits `v_wmma_f16_16x16x16_f16`, wave32, `SGPR=20`,
  `VGPR=34`, `LDS=0`, no spills.
- correctness:
  CPU-reference focused p512, p33, and p513 rows passed.
- timing:
  p512 same-row default packed Q8_1 route: `521.935217 us`.
  p512 f16acc WMMA16 route: `729.020000 us`.
- decision:
  reject for production promotion. Matching Vulkan's accumulator instruction
  alone is not enough; the direct 16x16 dequant-to-f16 path lacks Vulkan's
  128x128 cooperative-matrix dataflow and packed/staged reuse.
- notes:
  The route remains useful as an opt-in lane-layout and compiler diagnostic.
  Next Q8_0 parity work should scale toward the Vulkan large aligned schedule
  family instead of spending more time on standalone 16x16 WMMA tiles.

## 2026-06-17 - q8_0 f16acc wmma16 wg256 aggregation diagnostic

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, dirty with the opt-in
  `hrx_mul_mat_vec_q8_0_wmma16x16_f16acc_wg256_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  focused Llama 3.1 8B Q8_0 prompt rows: p512 plus synthetic p33 and p513
  odd/tail rows, `k=4096`, `rows=1024`.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_wmma16x16_f16acc_wg256_f32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_WMMA16_F16ACC_WG256_PROMPT=1`.
- baseline command:
  `test-backend-ops perf --test-file
  cache/hrxv1/gfx1151/q8_0-kv1024-focused-20260617-180540/q8_0_prompt_rows1024.txt
  --output csv` with default HRX routing.
- variant command:
  same command with `GGML_HRX_ENABLE_Q8_0_WMMA16_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-wmma16-f16acc-wg256-focused-20260617-204624/`.
  Correctness and perf traces selected
  `hrx_mul_mat_vec_q8_0_wmma16x16_f16acc_wg256_f32`; default perf selected
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`.
- profile/timing:
  HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16.hsaco`.
  The WG256 export emits `v_wmma_f16_16x16x16_f16`, wave32,
  `max_flat_workgroup_size=256`, `SGPR=18`, `VGPR=37`, `LDS=0`, no spills.
- correctness:
  CPU-reference focused p512, p33, and p513 rows passed.
- timing:
  p512 same-row default packed Q8_1 route: `520.350435 us`.
  p512 f16acc WG256 WMMA16 route: `825.078261 us`.
- decision:
  reject for production promotion. Grouping eight validated 16x16 f16acc WMMA
  subtiles into one 256-thread workgroup did not recover Vulkan-like behavior.
- notes:
  This narrows the Q8_0 interpretation: the useful Vulkan delta is not merely
  workgroup size or f16 accumulator type. The missing piece remains data reuse
  and packed/staged ownership across the 128x128 large aligned schedule family.

## 2026-06-17 - q4_k mmql128 BK2 staging-depth diagnostic

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, dirty with the opt-in
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bk2_wg256_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  focused Qwen3 30B Q4_XL exported Q4_K prompt rows at p512, p513, and p33.
- route or candidate:
  Q4_K x Q8_1 x4 MMQL128 route with `BK_STEP=2`, opt-in
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BK2_PROMPT=1`.
- prior:
  Vulkan oracle large aligned dense route uses BM128/BN128/WG256 with
  `LDS=22528`, `VGPR=192`, no spills. Existing HRX BK1 uses the same broad
  tile family with `LDS=8192`, `VGPR=139`, no spills.
- compile evidence:
  BK2 compiled through CMake/Ninja to
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q4_k_q8_1_x4_mmql128_bk2.hsaco`
  with `LDS=16384`, `SGPR=55`, `VGPR=145`, no spills.
- correctness:
  CPU-reference focused p512 and p513 rows passed with BK2 selected. Focused
  p33 rows passed and correctly stayed off BK2 because of the `cols >= 128`
  guard.
- timing:
  p512 BK1 -> BK2: Vcur `280.78 -> 201.36 us`, node `1189.61 -> 823.01 us`,
  Qcur p512 `738.53 -> 742.52 us`.
  p513 BK1 -> BK2: Vcur `207.89 -> 216.53 us`, node `886.36 -> 851.36 us`,
  Qcur narrow `42.36 -> 54.21 us`, Qcur p513 `877.63 -> 1003.39 us`.
- decision:
  reject for production promotion. BK2 is a useful bounded staging-depth probe,
  but the gains are not robust across odd/tail prompt rows. Keep it opt-in
  only and continue mining the Vulkan oracle for dataflow/ownership deltas
  beyond simple staged-K depth.

## 2026-06-17 - q8_0 BM64/BN128 compile-resource diagnostic

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, dirty with the opt-in
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_wg256_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  intended for focused Llama 3.1 8B Q8_0 prompt rows, but rejected before
  runtime testing by the compile-resource gate.
- route or candidate:
  Q8_0 x Q8_1 x4 direct-A/staged-B MMQ route with `BM=64`, `BN=128`,
  `COLS_PER_THREAD=32`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_PROMPT=1`.
- prior:
  accepted Q8_0 BM64/BN64 avoids spills and greatly improves p512/p513 but
  remains around `0.45x` Vulkan. Vulkan's production prompt family uses
  128-column workgroup-denominator tiles, so this probe widens columns while
  keeping the accepted BM64 row ownership instead of repeating the rejected
  BM128/BN64 row widening.
- compile evidence:
  HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x128.hsaco`
  compiled through CMake/Ninja, but the public route reports
  `wavefront_size=32`, `LDS=4352`, `SGPR=27`, `VGPR=192`,
  `VGPR spills=55`, and `private_segment_fixed_size=224`.
- correctness:
  skipped. The route failed the pre-runtime compile-resource gate.
- timing:
  skipped. A spilling candidate is not a valid production A/B row.
- decision:
  reject before focused runtime testing. Simple Vulkan-width column widening
  on the current scalar-dot direct-A/staged-B schedule hits the same live-state
  wall as the previous wide-column probe. The next Q8_0 attempt needs a
  different register tile or a more fundamental WMMA/cooperative dataflow, not
  another `COLS_PER_THREAD=32` extension.

## 2026-06-17 - q8_0 BM64/BN96 focused diagnostic

- source:
  `sources/llama.cpp` on `hrx-kernel-lib-v1`, dirty with the opt-in
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x96_wg256_f32` route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  focused Llama 3.1 8B Q8_0 prompt rows at p512, p33, and p513.
- route or candidate:
  Q8_0 x Q8_1 x4 direct-A/staged-B MMQ route with `BM=64`, `BN=96`,
  `COLS_PER_THREAD=24`, opt-in
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1`.
- prior:
  accepted BM64/BN64 is spill-free and improves the current Q8_0 prompt path,
  while BM64/BN128 spills. This probe brackets the column-width axis without
  crossing into the spilling `COLS_PER_THREAD=32` live-state regime.
- compile evidence:
  HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x96.hsaco`
  compiled through CMake/Ninja with `wavefront_size=32`, `LDS=3264`,
  `SGPR=26`, `VGPR=181`, no spills, and no private segment.
- correctness:
  CPU-reference focused p512 and p513 rows passed with BN96 selected. Focused
  p33 rows passed and correctly stayed on the default narrow route because of
  the `cols >= 64` guard.
- timing:
  p512 BN64 -> BN96: Vcur `554.67 -> 652.96 us`, Qcur
  `2062.37 -> 2066.61 us`, ffn_out `10031.44 -> 8405.61 us`, ffn_gate
  `9196.35 -> 7728.52 us`, result_output `78999.95 -> 67033.31 us`.
  The attempted p513 BN96 perf run is invalid as BN96 evidence: its route log
  selected `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`, not BN96.
- decision:
  reject for production promotion. BN96 shows useful large-row p512 wins, but
  it regresses the smaller p512 rows, does not have valid odd-tail timing, and
  increases VGPR pressure close to the BN128 spill wall. Keep the result as a
  bounded schedule-axis rejection and continue with a different Q8_0 dataflow
  or register tile rather than wider direct-A/staged-B scalar-dot columns.

## 2026-06-17 - q8_0 BM64/BN96 with BN64 fallback model A/B

- source:
  `sources/llama.cpp` at `0a836dc5b hrx: add q8 mmq64x96 diagnostic`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, `GGML_HRX=ON`,
  `GGML_HRX_AMDGPU_TARGETS=gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Llama 3.1 8B Q8_0, HRX0, `fa=1`, three no-warmup repetitions at p33,
  p512, p513 with `b=512/ub=512`, plus p513 with `b=1024/ub=1024` to test the
  single-graph tail regime.
- route or candidate:
  Opt-in paired policy:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1` plus
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`. BN96 is first in selector
  order for `cols >= 64`; BN64 remains the narrow fallback.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-mmq64x96-model-ab-20260617-213256/`.
- route evidence:
  p33 selected BN64 for 477 Q8_0 x Q8_1 prompt routes and never selected BN96.
  p512 selected BN96 for 477 prompt routes. p513 `ub512` selected BN96 for the
  p512 graph and used scalar Q8_0 for residual-token routes. p513 `ub1024`
  selected BN96 for all 477 prompt routes.
- timing:
  BN64-only -> BN96+BN64 fallback:
  p33 `203.733 -> 207.523 tok/s`;
  p512 `402.465 -> 424.832 tok/s`;
  p513 `ub512` `386.261 -> 408.731 tok/s`;
  p513 `ub1024` `372.236 -> 412.379 tok/s`.
- decision:
  accept the paired policy as a gfx1151 opt-in candidate. Do not enable BN96
  alone: p33 then falls through to the older BM128/BN32 route and regresses to
  `113.638 tok/s`. This is still not Vulkan parity, so the next Q8_0 work
  should move beyond direct-A/staged-B scalar-dot column widening.

## 2026-06-17 - current best with q8 bn96 downloaded basket p512 fa1

- source:
  `sources/llama.cpp` at
  `534276fc8 hrx: promote q8 mmq64x96 policy evidence`, clean before
  recording this evidence.
- build:
  `build/hrx-v1-catalog-gfx1151` for HRX and `build/vulkan-gfx1151` for
  Vulkan; Release; ROCm `/srv/vm-shared/rocm/rocm-head`; `spirv-dis` from
  SPIRV-Tools v2026.1 is installed for Vulkan oracle follow-up.
- model/shape:
  eight downloaded basket models, `p512/n0`, `fa=1`, `b=1024`, `ub=1024`,
  `r=1`.
- route or candidate:
  current best opt-in policy plus Q8_0 BN96/BN64 fallback:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1` and
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- baseline command:
  same-source Vulkan `llama-bench` rows with `-p 512 -n 0 -b 1024 -ub 1024
  -fa 1 -r 1 -o json --no-warmup -ngl 99 -dev Vulkan0`.
- variant command:
  same-source HRX `llama-bench` rows with `-p 512 -n 0 -b 1024 -ub 1024
  -fa 1 -r 1 -o json --no-warmup -ngl 99 -dev HRX0` and the current-best HRX
  policy environment.
- route trace:
  per-model HRX stderr route traces under
  `cache/hrxv1/gfx1151/current-best-q8bn96-basket-p512-fa1-r1-20260617-213941/`.
- profile/timing:
  `cache/hrxv1/gfx1151/current-best-q8bn96-basket-p512-fa1-r1-20260617-213941/summary.csv`
  and `summary.json`.
- correctness:
  all rows completed and JSON reported the intended backend identity
  (`backends=HRX` for HRX rows, `backends=Vulkan` for Vulkan rows). This is a
  model-level throughput comparison, not a CPU-reference kernel promotion
  gate.
- timing:
  geomean HRX/Vulkan ratio is `0.5025x`. Worst rows:
  Llama 3.1 8B Q4_K_M `457.005 / 1164.657 = 0.392x`,
  DeepSeek R1 Qwen 14B Q4_K_M `246.487 / 613.449 = 0.402x`,
  Qwen2.5 Coder 7B Q5_K_M `453.777 / 1094.265 = 0.415x`,
  Llama 3.1 8B Q8_0 `424.000 / 903.967 = 0.469x`,
  Qwen3 30B Q6_K `476.577 / 983.289 = 0.485x`.
- decision:
  accept as the current boulder-ranking snapshot. It is not a route promotion
  by itself; aggregate basket numbers decide what to inspect next, while route
  promotion still requires focused correctness, route evidence, compile
  resource checks, odd/tail coverage, and same-runner A/B.
- notes:
  The updated worst row is no longer a pure Q8_0 problem. Llama 3.1 8B
  Q4_K_M selects the Q4 MMQL prompt route heavily, but its route samples also
  include Q6_K output rows. The next focused probe should isolate those exact
  Q6_K output shapes before spending more time on broad model-level runs.

## 2026-06-17 - llama31 q4km exact q6 output rerank

- source:
  `sources/llama.cpp` at
  `534276fc8 hrx: promote q8 mmq64x96 policy evidence`, clean before
  recording the source-side tuning metadata.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  exact exported Q6_K output rows from Llama 3.1 8B Q4_K_M, `p512/n0`,
  `fa=1`, `b=1024`, `ub=1024`: `Vcur-0` `[1024,1024]` from
  `q6_K[4096,1024]`, `ffn_out-0` `[4096,1024]` from
  `q6_K[14336,4096]`, and `result_output` `[128256,1024]` from
  `q6_K[4096,128256]`.
- route or candidate:
  current `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x64_wg256_f32` versus existing
  opt-in `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x128_wg256_f32` and
  `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x128_wg256_f32`.
- baseline command:
  `test-backend-ops perf --test-file <artifact>/q6_prompt.txt --output csv`
  with `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- variant command:
  same focused perf command with
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64X128_PROMPT=1` or
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/llama31-q4km-q6-output-rerank-20260617-214220/*routes.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/llama31-q4km-q6-output-rerank-20260617-214220/*perf.csv`.
- correctness:
  focused CPU-reference test passed for the current MMQ64 route on all three
  exported rows. The perf reranks also reported `passed=1`.
- timing:
  current MMQ64: `2258.90 us` Vcur, `66862.64 us` ffn_out,
  `375999.17 us` result_output. MMQ64x128: `2384.88 us`, `113829.66 us`,
  `724312.69 us`. MMQL128: `2383.38 us`, `121840.03 us`,
  `1039822.69 us`.
- decision:
  reject the existing wider/staged Q6 variants for this exact worst-row
  follow-up. The current direct MMQ64 route remains fastest but is still far
  from the Vulkan oracle family, so the Q6 blocker needs a new schedule/dataflow
  rather than enabling one of the existing variants.
- notes:
  The legacy-auto perf pass was stopped after several minutes with no CSV
  output, after the intended candidates had completed. Route traces for the
  completed rows selected the intended providers plus the small fallback
  `hrx_mul_mat_vec_q6_k_rows2_cols8_wg32_f32` for non-packed rows.

## 2026-06-17 - q6_k direct wmma16 f16acc diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_f32`, not yet promoted to an
  automatic route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  exact exported Q6_K output rows from Llama 3.1 8B Q4_K_M, `p512/n0`,
  `fa=1`, plus synthesized p33 and p513 odd-tail focused rows using the same
  weight shapes.
- route or candidate:
  direct Q6_K dequant-to-f16 plus F32 RHS cast-to-f16 WMMA route,
  `hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_f32`, gated by
  `GGML_HRX_ENABLE_Q6_K_WMMA16_F16ACC_PROMPT=1`.
- prior-art schedule source:
  Vulkan oracle Q6 dense prompt pipeline
  `matmul_q6_k_f32_f16acc_aligned_l`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`, workgroup 256x1x1,
  LDS `22528`, VGPR `192`, no spills, and `v_wmma_f16_16x16x16_f16`.
  See `docs/hrxv1/q6k-vulkan-oracle-schedule-ledger.md`.
- compile report:
  `mul_mat_vec_q6_k_wmma16.hsaco` emits `v_wmma_f16_16x16x16_f16` with
  wave32, SGPR `24`, VGPR `35`, LDS `0`, and no spills.
- baseline command:
  `test-backend-ops perf --test-file <artifact>/q6_prompt.txt --output csv`
  with `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- variant command:
  same focused test/perf files with
  `GGML_HRX_ENABLE_Q6_K_WMMA16_F16ACC_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q6-wmma16-f16acc-focused-20260617-215803/*routes.log`.
- profile/timing:
  baseline `current_mmq64-perf.csv`; variant `wmma16_f16acc-perf.csv` stayed
  empty after the run was stopped for timeout.
- correctness:
  CPU-reference focused gates passed for exact p512 rows and synthesized p33
  and p513 odd-tail rows. Route traces selected
  `hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_f32`.
- timing:
  current MMQ64 completed the exact rows at `2254.51 us`, `56728.72 us`, and
  `379448.50 us`. The direct WMMA run was stopped after several minutes with
  1688 selected dispatches and no CSV result, so it is noncompetitive.
- decision:
  reject the direct one-wave WMMA route for promotion. Keep it as evidence that
  gfx1151 HIP C++ can compile and run the same f16 WMMA opcode family as
  Vulkan, but that throughput needs the Vulkan-shaped 256-thread,
  128x128-tile, staged dataflow rather than isolated 16x16 direct global
  loads.
- notes:
  Next Q6 work should preserve the Vulkan oracle structure more closely:
  staged A/B reuse, 256-thread CTA, 128x128 output tile, and bounded live state.
  Do not repeat direct-WMMA variants without a concrete data-reuse change.

## 2026-06-17 - q6_k staged wmma16 wg256 row-bound candidate

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_wg256_f32`, guarded by
  `GGML_HRX_ENABLE_Q6_K_WMMA16_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  exact exported Q6_K rows from Llama 3.1 8B Q4_K_M `p512/n0/fa1`, synthesized
  p33/p513 odd-tail rows using those same weight shapes, and Qwen3 30B Q6_K
  `p512/n0/fa1/b1024/ub1024`.
- route or candidate:
  256-thread staged Q6_K WMMA f16acc candidate. It stages a 64x32 by 16 f16
  A/B panel in LDS and runs eight wave32 `v_wmma_f16_16x16x16_f16` subtiles per
  workgroup. Selector policy is row-bound to `rows <= 4096`; larger rows fall
  back to `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x64_wg256_f32` when
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1` is also set.
- prior-art schedule source:
  Vulkan oracle Q6 dense prompt pipeline
  `matmul_q6_k_f32_f16acc_aligned_l`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`, workgroup 256x1x1,
  LDS `22528`, VGPR `192`, no spills, and `v_wmma_f16_16x16x16_f16`.
- compile report:
  `mul_mat_vec_q6_k_wmma16_wg256.hsaco` emits
  `v_wmma_f16_16x16x16_f16` with wave32, SGPR `36`, VGPR `45`, LDS `3072`,
  and no spills.
- baseline command:
  `test-backend-ops test|perf -b HRX0 -o MUL_MAT --test-file <artifact>/q6_prompt.txt --output csv`
  with `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- variant command:
  same focused files with
  `GGML_HRX_ENABLE_Q6_K_WMMA16_F16ACC_WG256_PROMPT=1`; row-bound mixed pass
  also used `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q6-wmma16-f16acc-wg256-focused-20260617-221229/*routes.log`
  and
  `cache/hrxv1/gfx1151/q6-wmma16-wg256-qwen3q6-model-ab-20260617-221937/variant.routes.log`.
- profile/timing:
  focused CSVs under
  `cache/hrxv1/gfx1151/q6-wmma16-f16acc-wg256-focused-20260617-221229/`;
  model JSON under
  `cache/hrxv1/gfx1151/q6-wmma16-wg256-qwen3q6-model-ab-20260617-221937/`.
- correctness:
  CPU-reference focused gates passed for exact p512 rows and synthesized p33
  and p513 odd-tail rows. The row-bound mixed p512 pass selected staged WMMA
  for `Vcur-0` and `ffn_out-0`, and MMQ64 for `result_output`.
- timing:
  focused exact rows:
  current MMQ64 `2288.11 us`, `70762.23 us`, `373409.03 us`;
  forced staged WMMA `2034.99 us`, `52681.14 us`, `486642.92 us`;
  row-bound mixed `2148.40 us`, `58137.49 us`, `385767.64 us`.
  Qwen3 30B Q6_K no-trace r3 model A/B improved from `492.92 tok/s` to
  `520.82 tok/s` with samples `[451.107, 515.439, 512.213]` versus
  `[488.037, 536.758, 537.675]`. The earlier traced r1 A/B improved from
  `397.55 tok/s` to `483.08 tok/s` and selected the staged Q6 route 192 times;
  the traced run is route evidence, not the preferred timing row.
- decision:
  accept as a gfx1151 opt-in candidate with the `rows <= 4096` guard. Do not
  force it on large output rows. This is a meaningful Vulkan-oracle-driven
  step, but it is still a 64x32 staged probe rather than the full Vulkan
  128x128/22528-byte LDS schedule.
- notes:
  Next Q6 work should expand toward the Vulkan 128x128 staged cooperative
  matrix family or add a separate wide-output schedule. The row-bound candidate
  should be included in the next downloaded-basket rerank, but broad default
  promotion still needs more model coverage.

## 2026-06-17 - current-best basket after row-bound q6 staged wmma

- source:
  `sources/llama.cpp` at
  `f1cb33b10 hrx: add row-bound q6 staged wmma`, clean.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  eight downloaded basket rows, `p512/n0/fa1/b1024/ub1024/r1`.
- route or candidate:
  current-best opt-in stack plus row-bound Q6 staged WMMA:
  `GGML_HRX_ENABLE_Q6_K_WMMA16_F16ACC_WG256_PROMPT=1` composed with
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64_PROMPT=1` for wide-row fallback.
- baseline command:
  no new Vulkan run in this entry. Ratios reuse the same-machine Vulkan rows
  from
  `cache/hrxv1/gfx1151/current-best-q8bn96-basket-p512-fa1-r1-20260617-213941/`.
- variant command:
  per-model HRX `llama-bench -p 512 -n 0 -b 1024 -ub 1024 -fa 1 -r 1
  -o json --no-warmup -ngl 99 -dev HRX0` with the current-best plus Q6 staged
  WMMA environment.
- route trace:
  this is a no-trace timing rerank. Route evidence for Q6 staged WMMA is in
  `cache/hrxv1/gfx1151/q6-wmma16-wg256-qwen3q6-model-ab-20260617-221937/variant.routes.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/current-best-q6wmma-basket-p512-fa1-r1-20260617-222645/summary.csv`
  and `summary.json`.
- correctness:
  all HRX rows completed with JSON `backends=HRX`. Focused CPU-reference gates
  for the new Q6 route are recorded in
  `cache/hrxv1/gfx1151/q6-wmma16-f16acc-wg256-focused-20260617-221229/`.
- timing:
  geomean HRX/Vulkan ratio improved from `0.5025x` to `0.5354x`.
  - Qwen2.5 Coder 7B Q5_K_M: `453.777 -> 527.661 tok/s`, ratio `0.482x`;
  - Llama 3.2 3B Q4_K_M: `1210.734 -> 1329.159 tok/s`, ratio `0.534x`;
  - Llama 3.1 8B Q4_K_M: `457.005 -> 536.096 tok/s`, ratio `0.460x`;
  - Llama 3.1 8B Q8_0: `424.000 -> 422.153 tok/s`, ratio `0.467x`;
  - DeepSeek R1 Qwen 14B Q4_K_M: `246.487 -> 248.663 tok/s`, ratio `0.405x`;
  - Qwen3 30B Q6_K: `476.577 -> 522.811 tok/s`, ratio `0.532x`;
  - Qwen3 30B Q4_K_XL: `880.368 -> 884.455 tok/s`, ratio `0.744x`;
  - Qwen3 Coder 30B Q4_K_M: `865.809 -> 867.735 tok/s`, ratio `0.761x`.
- decision:
  accept as the current boulder-ranking snapshot. The Q6 staged route is
  moving the basket, but the worst rows are again Q4/Q5/Q8 dense prompt
  matmul. Next focused work should trace DeepSeek R1 Qwen 14B Q4_K_M and mine
  the Q4_K Vulkan oracle schedule before adding another HIP candidate.

## 2026-06-17 - deepseek q4_k direct staged wmma16 diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_wmma16x16_f16acc_wg256_f32`, guarded by
  `GGML_HRX_ENABLE_Q4_K_WMMA16_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513` with `b=ub=prompt`.
- route or candidate:
  256-thread staged Q4_K WMMA diagnostic. It directly dequants Q4_K to f16,
  casts the F32 RHS tile to f16, stages a 64x32 by 16 A/B panel in LDS, and
  uses eight wave32 `v_wmma_f16_16x16x16_f16` subtiles per workgroup.
- prior-art schedule source:
  Vulkan Q4_K `matmul_q4_k_f32_f16acc_aligned_l`; RADV emits
  `v_wmma_f16_16x16x16_f16` with `LDS=22528`, `VGPR=192`, no spills. The
  current accepted HRX Q4_K MMQL128 route emits integer `v_dot4_i32_iu8`.
- compile report:
  `mul_mat_vec_q4_k_wmma16_wg256.hsaco` emits
  `v_wmma_f16_16x16x16_f16` with wave32, SGPR `27`, VGPR `53`, LDS `3072`,
  and no spills.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <artifact>/q4_prompt_p512.txt --output csv`
  with the current-best env, including
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q4_K_WMMA16_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-wmma16-wg256-focused-20260617-223548/test-p512-wmma.routes.log`,
  `test-p33-exact-wmma.routes.log`, `test-p513-exact-wmma.routes.log`, and
  `q4-wmma16-perf.routes.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/q4-wmma16-wg256-focused-20260617-223548/`.
- correctness:
  CPU-reference focused gates passed for p512 and exact p33/p513 rows while
  selecting `hrx_mul_mat_vec_q4_k_wmma16x16_f16acc_wg256_f32`.
- timing:
  focused p512 rows regressed versus current MMQL128:
  `Kcur 986.46 -> 2402.45 us`, `Qcur 4123.76 -> 14151.28 us`,
  `ffn_out 13126.86 -> 55425.92 us`, and
  `ffn_gate 13244.67 -> 58826.01 us`.
- decision:
  reject for production promotion. Keep as opt-in diagnostic evidence that
  merely matching Vulkan's WMMA opcode is not enough on gfx1151.
- notes:
  The next Q4_K candidate should preserve more of Vulkan's packed/staged
  dataflow: 128x128 ownership, deeper LDS/reuse, and the actual A/B load and
  barrier schedule. Do not spend another cycle on direct dequant-to-WMMA unless
  the pivot changes data reuse, not just tile grouping.

## 2026-06-17 - deepseek q4_k vk128 direct wmma diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_f16acc_wg256_f32`, guarded by
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513`.
- route or candidate:
  Vulkan-large direct-F32 Q4_K WMMA diagnostic. It uses `BM=128`, `BN=128`,
  `BK=32`, `WG=256`, Q4_K dequant to f16 LDS, direct F32 RHS cast to f16 LDS,
  and multiple 16x16 f16acc WMMA accumulator tiles per workgroup.
- prior-art schedule source:
  Vulkan Q4_K `matmul_q4_k_f32_f16acc_aligned_l`, resolved from
  `mul_mm.comp` / `mul_mm_funcs.glsl` with `LOAD_VEC_A=4`, `LOAD_VEC_B=8`,
  `BK=32`, `BM=128`, `BN=128`, padded f16vec2 LDS tiles, and RADV
  `v_wmma_f16_16x16x16_f16`.
- compile report:
  `mul_mat_vec_q4_k_wmma16_vk128_wg256.hsaco` emits 16 static
  `v_wmma_f16_16x16x16_f16` sites with wave32, SGPR `38`, VGPR `115`,
  LDS `16384`, and no spills.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <artifact>/q4_prompt_p512.txt --output csv`
  with the current-best env, including
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-wmma16-vk128-focused-20260617-225628/test-p512-vk128.routes.log`,
  `test-p33-exact-vk128.routes.log`, `test-p513-exact-vk128.routes.log`, and
  `q4-wmma16-vk128-perf.routes.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/q4-wmma16-vk128-focused-20260617-225628/`.
- correctness:
  CPU-reference focused gates passed for p512 and exact p513 rows while
  selecting `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_f16acc_wg256_f32`; exact
  p33 passed while staying on the existing narrow Q4_K route.
- timing:
  focused p512 rows regressed versus current MMQL128:
  `Kcur 979.20 -> 1565.46 us`, `Qcur 3948.47 -> 6674.58 us`,
  `ffn_out 14573.74 -> 19495.02 us`, and
  `ffn_gate 13196.74 -> 23377.37 us`.
- decision:
  reject for production promotion. Keep as opt-in diagnostic evidence that a
  high-level Vulkan-large direct-F32 HIP WMMA clone still misses the winning
  schedule.
- notes:
  The next Q4_K direction should not be another direct-F32 RHS WMMA tile clone.
  Either reproduce RADV's lower-level cooperative-matrix load/LDS/wait
  schedule more exactly, or pivot back to the packed Q8_1/x4 dataflow and use
  the Vulkan oracle as a resource and scheduling target rather than an RHS
  dataflow target.

## 2026-06-17 - deepseek q4_k vk128 padded direct wmma diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_f16acc_wg256_f32`, guarded by
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513`.
- route or candidate:
  Single-axis follow-up to VK128 direct WMMA. It keeps `BM=128`, `BN=128`,
  `BK=32`, `WG=256`, direct F32 RHS, and f16acc WMMA, but changes shared A/B
  row stride from 32 halfs to Vulkan's padded 40-half f16vec2-equivalent
  stride.
- prior-art schedule source:
  Vulkan Q4_K `matmul_q4_k_f32_f16acc_aligned_l`, where `mul_mm.comp` uses
  `SHMEM_STRIDE = BK / 2 + 4 = 20` f16vec2 values, equivalent to a 40-half
  row stride for the staged A/B tiles.
- compile report:
  `mul_mat_vec_q4_k_wmma16_vk128_padded_wg256.hsaco` emits 16 static
  `v_wmma_f16_16x16x16_f16` sites with wave32, SGPR `25`, VGPR `121`,
  LDS `20480`, and no spills.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <artifact>/q4_prompt_p512.txt --output csv`
  with the current-best env, including
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-wmma16-vk128-padded-focused-20260617-230429/test-p512-padded.routes.log`,
  `test-p33-exact-padded.routes.log`, `test-p513-exact-padded.routes.log`,
  and `q4-wmma16-vk128-padded-perf.routes.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/q4-wmma16-vk128-padded-focused-20260617-230429/`.
- correctness:
  CPU-reference focused gates passed for p512 and exact p513 rows while
  selecting `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_f16acc_wg256_f32`;
  exact p33 passed while staying on the existing narrow Q4_K route.
- timing:
  focused p512 rows regressed versus current MMQL128:
  `Kcur 987.01 -> 1543.83 us`, `Qcur 4107.51 -> 7244.42 us`,
  `ffn_out 15515.40 -> 24740.20 us`, and
  `ffn_gate 12186.69 -> 25907.86 us`.
- decision:
  reject for production promotion. Padding the direct-F32 HIP WMMA LDS layout
  toward Vulkan does not recover the RADV schedule.
- notes:
  Direct-F32 HIP WMMA now has opcode-only, VK128, and VK128-padded rejected
  points. The next Q4_K pass should either reproduce RADV lowerings much more
  exactly or return to packed Q8_1/x4 and tune that dataflow.

## 2026-06-17 - deepseek q4_k mmql128x64 packed diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x64_wg256_f32`, guarded by
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128X64_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513`.
- route or candidate:
  Packed-Q8_1/x4 output-ownership probe. It keeps Q4_K MMQL staged packed
  dataflow, `BM=128`, wave64, and `BK_STEP=1`, but narrows the output tile from
  `BN=128/WN=64` to `BN=64/WN=32` to halve live accumulators.
- prior-art schedule source:
  Current accepted Q4_K MMQL128 route plus the direct-F32 WMMA rejections,
  which showed the next useful Q4 work should return to packed Q8_1/x4 and
  test specific dataflow axes.
- compile report:
  `mul_mat_vec_q4_k_q8_1_x4_mmql128x64.hsaco` has wave64, SGPR `53`, VGPR
  `109`, LDS `5632`, no spills, 256 static integer-dot sites, and 119
  `s_waitcnt` sites.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <artifact>/q4_prompt_p512.txt --output csv`
  with current-best env including
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- variant command:
  same focused file with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128X64_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-mmql128x64-focused-20260617-231147/test-p512-mmql128x64.routes.log`,
  `test-p33-exact-mmql128x64.routes.log`,
  `test-p513-exact-mmql128x64.routes.log`, and
  `q4-mmql128x64-perf.routes.log`.
- profile/timing:
  `cache/hrxv1/gfx1151/q4-mmql128x64-focused-20260617-231147/`.
- correctness:
  CPU-reference focused gates passed for p512 and exact p513 rows while
  selecting `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x64_wg256_f32`; exact p33
  passed while staying on the existing MMQ64 route.
- timing:
  focused p512 rows regressed versus current MMQL128:
  `Kcur 989.40 -> 1100.29 us`, `Qcur 4018.51 -> 4817.41 us`,
  `ffn_out 14563.47 -> 22394.71 us`, and
  `ffn_gate 14017.86 -> 20459.41 us`.
- decision:
  reject for production promotion. Lower output ownership reduced VGPR/LDS but
  lost B reuse and doubled column workgroups.
- notes:
  Keep the 128-column production tile for Q4 packed prompt paths. The next
  packed-path axis should target load ordering, wait placement, or B-cache
  ownership without shrinking the output tile.

## 2026-06-18 - deepseek q4_k mmql128 bsplit4 packed diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bsplit4_wg256_f32`, guarded by
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BSPLIT4_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513`.
- route or candidate:
  Packed-Q8_1/x4 B-cache ownership probe. It preserves the current Q4_K
  MMQL128 production dataflow, `BM=128`, `BN=128`, wave64, `BK_STEP=1`, and
  output ownership, but changes staged B payload ownership from two lanes per
  row loading four packed words each to four lanes per row loading two packed
  words each.
- prior-art schedule source:
  Vulkan/RADV Q4_K oracle plus the current accepted packed-Q8_1/x4 MMQL128
  route. This is a bounded probe on the load/wait axis after direct-F32 WMMA,
  narrower column ownership, and cache-row padding all failed to close the
  gap.
- compile report:
  `mul_mat_vec_q4_k_q8_1_x4_mmql128_bsplit4.hsaco` has wave64, SGPR `48`,
  VGPR `140`, LDS `8192`, no spills, 512 static integer-dot sites, 46
  `ds_loads`, 12 `ds_stores`, 18 global loads, 64 global stores, 188 waits,
  and 2 barriers.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <artifact>/q4_prompt_p512.txt --output csv`
  with current-best env including
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- variant command:
  same focused rows with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BSPLIT4_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-mmql128-bsplit4-focused-20260618-004510/`.
- correctness:
  CPU-reference focused gates passed for p512 and exact p513 rows while
  selecting the bsplit4 provider; exact p33 passed while staying on the
  existing MMQ64 route.
- timing:
  p512 was mixed versus current MMQL128:
  `Kcur 977.70 -> 1039.18 us`, `Qcur 3941.94 -> 4511.51 us`,
  `ffn_out 14287.02 -> 14024.76 us`, and
  `ffn_gate 14007.78 -> 13693.10 us`.
  p513 regressed on every row:
  `Kcur 603.45 -> 644.47 us`, `Qcur 2420.84 -> 2479.59 us`,
  `ffn_out 8053.23 -> 8359.72 us`, and
  `ffn_gate 7682.05 -> 9098.94 us`.
- decision:
  reject for production promotion. Splitting B payload loads across more lanes
  does not recover the Vulkan parity axis and harms odd/tail behavior.
- notes:
  Keep this route opt-in only as evidence. The next packed-path probe should
  be closer to the actual RADV load/issue schedule, not just a different B
  payload ownership split.

## 2026-06-18 - deepseek q4_k vk128 padded w64 half4 LDS diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_h4load_f16acc_wg256_f32`,
  guarded by
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_H4LOAD_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M p512 Vulkan oracle schedule versus the
  current HRX wave64 padded direct-WMMA Q4_K route. This was an emitted-ISA
  gate before focused runtime benchmarking.
- route or candidate:
  Direct-F32 RHS Q4_K WMMA source spelling probe. It preserves the existing
  Vulkan-shaped BM128/BN128/BK32/WG256, wave64 WMMA builtin, and 40-half padded
  LDS stride, but spells A/B LDS fragment reads as half4 chunks in an attempt
  to recover RADV's `ds_load_b64` feed instead of the HIP route's coalesced
  `ds_load_b128` feed.
- prior-art schedule source:
  DeepSeek Q4_K Vulkan oracle RADV ISA:
  `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p512-fa1-20260617-200426/radv/isa/matmul_q4_k_f32_f16acc_aligned_l__main__5666175250529efb.amdgcn.txt`.
- compile report:
  `cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/`.
  RADV has 64 `ds_load_b64`, 0 `ds_load_b128`, 32 static WMMA sites, 50
  event-window `s_waitcnt` sites, SGPR `108`, VGPR `192`, LDS `22528`.
  The existing HRX padded W64 route has 32 `ds_load_b128`, 0 `ds_load_b64`, 32
  WMMA sites, SGPR `29`, VGPR `165`, LDS `20480`.
- baseline command:
  HSACO/ISA inspection of the existing
  `mul_mat_vec_q4_k_wmma16_vk128_padded_w64_wg256.hsaco`.
- variant command:
  HSACO/ISA inspection of
  `mul_mat_vec_q4_k_wmma16_vk128_padded_w64_h4load_wg256.hsaco`.
- route trace:
  not run. The candidate failed the compile/ISA schedule gate before focused
  correctness/perf.
- correctness:
  not run. This was rejected before runtime because it did not produce the
  intended emitted schedule.
- timing:
  not run.
- decision:
  reject for production promotion. Non-volatile half4 source coalesced back to
  32 `ds_load_b128`, 0 `ds_load_b64`, 32 WMMA sites, 155 waits, SGPR `29`,
  VGPR `165`, LDS `20480`, which duplicates the existing HIP issue shape.
  The attempted volatile spelling produced `flat_load_b64` instead of LDS
  reads, 28 WMMA sites, 231 waits, SGPR `32`, VGPR `233`, LDS `20480`.
- notes:
  C++ half4 spelling is not a viable knob for reaching the RADV `ds_load_b64`
  schedule. The next direct-WMMA attempt needs a different lowering mechanism
  or must move back to packed-Q8_1/x4 with RADV issue order as the oracle.

## 2026-06-18 - deepseek q4_k vk128 padded w64 b64asm LDS diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_b64asm_f16acc_wg256_f32`,
  guarded by
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_B64ASM_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513`.
- route or candidate:
  Direct-F32 RHS Q4_K WMMA explicit-LDS-read probe. It preserves the existing
  Vulkan-shaped BM128/BN128/BK32/WG256, wave64 WMMA builtin, and 40-half padded
  LDS stride, but forces A/B fragment reads through inline-assembly
  `ds_read_b64`.
- prior-art schedule source:
  DeepSeek Q4_K Vulkan oracle RADV ISA and the rejected half4 source spelling
  probe. The concrete target was RADV's `ds_load_b64` fragment feed around
  `v_wmma_f16_16x16x16_f16`.
- compile report:
  `cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/`.
  The no-memory-clobber B64ASM kernel emitted 256 `ds_load_b64`, 0
  `ds_load_b128`, 32 static WMMA sites, 140 waits, SGPR `30`, VGPR `169`, LDS
  `20480`, but produced NaN/inf outputs. The final half4-output,
  memory-clobbered, conservative-wait variant still emitted 256 `ds_load_b64`
  and 32 WMMA sites, but raised waits to `396`.
- baseline command:
  not run for timing; this probe first had to pass CPU-reference correctness.
- variant command:
  `test-backend-ops test -b HRX0 -o MUL_MAT --test-file <artifact>/q4_prompt_p512.txt --output csv`
  and exact p33/p513 variants with
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_B64ASM_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-w64-b64asm-focused-20260618-010917/`,
  `cache/hrxv1/gfx1151/q4-w64-b64asm-focused-20260618-010917-wait/`, and
  final artifact
  `cache/hrxv1/gfx1151/q4-w64-b64asm-h4out-memclobber-20260618-011846/`.
  Follow-up explicit LDS-base artifact:
  `cache/hrxv1/gfx1151/q4-w64-b64asm-ldsbase-20260618-012420/`.
- correctness:
  p512 and p513 selected the B64ASM provider for all four focused rows. The
  original asm failed CPU-reference comparison with NaN/inf outputs. Adding a
  `memory` clobber to the inline read and returning the half4 asm output
  directly changed this to finite ERR but still failed tolerance: p512 roughly
  `0.0060-0.0115`, p513 roughly `0.0016-0.0117`, tolerance `0.0005`. Exact
  p33 stayed on the existing narrow Q4 route and passed. Recasting the shared
  arrays to explicit `address_space(3)` base pointers before fragment pointer
  arithmetic did not materially improve the result: p512 still failed around
  `0.0058-0.0115`, and p513 around `0.0017-0.0115`.
- timing:
  not run because correctness failed.
- decision:
  reject for production promotion. The forced instruction width is achievable,
  and optimizer-visible inline asm avoids the catastrophic NaN/inf failure,
  but the current inline-asm fragment spelling is not semantically equivalent
  to RADV's cooperative-matrix load path. Full `lgkmcnt(0)` waits after each
  inline-asm read, direct half4 output, and explicit LDS base-pointer
  arithmetic did not fix correctness.
- notes:
  Do not treat this as evidence against the RADV schedule itself. It is
  evidence that a correct HIP clone needs a correct fragment map and reuse
  schedule before load-width and wait scheduling can be optimized.

## 2026-06-18 - deepseek q4_k vk128 padded w64 b64group reuse diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`,
  guarded by
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, HRX target `gfx1151`, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  DeepSeek R1 Distill Qwen 14B Q4_K_M focused Q4_K prompt rows exported from
  `p512/n0/fa1/b1024/ub1024`, plus exact odd/tail exports for `p33` and
  `p513`.
- route or candidate:
  Direct-F32 RHS Q4_K WMMA grouped explicit-LDS-read probe. It keeps the
  VK128 padded wave64 shape, but loads four A fragments and four B fragments
  per k-half before issuing the 4x4 WMMA block.
- prior-art schedule source:
  DeepSeek Q4_K Vulkan oracle RADV ISA and the rejected B64ASM diagnostic. The
  concrete target was RADV's `64 ds_load_b64` / `32 v_wmma` load-count family
  and B-fragment reuse across row tiles.
- compile report:
  `cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/`.
  The no-wait B64GROUP route emitted 64 `ds_load_b64`, 0 `ds_load_b128`, 32
  WMMA sites, 142 waits, SGPR `29`, VGPR `199`, LDS `20480`, and no spills.
  The conservative waited version kept 64 `ds_load_b64`, 0 `ds_load_b128`, and
  32 WMMA sites, but raised waits to `206`.
- baseline command:
  same-row Kcur p512 perf with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- variant command:
  same-row Kcur p512 perf with the MMQL128 baseline gate plus
  `GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.
- route trace:
  no-wait correctness artifact:
  `cache/hrxv1/gfx1151/q4-w64-b64group-focused-20260618-013423/`;
  waited correctness artifact:
  `cache/hrxv1/gfx1151/q4-w64-b64group-waited-focused-20260618-013533/`;
  waited Kcur perf artifact:
  `cache/hrxv1/gfx1151/q4-w64-b64group-waited-kcur-perf-20260618-014159/`.
- correctness:
  No-wait grouping selected p512/p513 but failed with NaNs. The waited grouping
  selected p512 and p513, kept exact p33 on the existing narrow Q4 route, and
  passed p512, exact p33, and exact p513 focused CPU-reference gates.
- timing:
  Kcur p512 same-runner focused perf regressed from `1038.69 us` for current
  MMQL128 to `3103.19 us` for B64GROUP.
- decision:
  reject for production promotion. This is an accepted schedule milestone
  because HIP now matches RADV's LDS load-count family with correctness, but
  it is still too slow. The next direct-WMMA work should target RADV's
  wait/issue structure or a lower-level cooperative-matrix spelling; another
  load-width-only probe is not enough.
- notes:
  The killed artifact
  `cache/hrxv1/gfx1151/q4-w64-b64group-waited-perf-20260618-013605/` is
  invalid as performance evidence because the baseline was not pinned to
  MMQL128 and routed through the slow scalar Q4 path.

## 2026-06-18 - qwen2.5 q5_k vk128 padded w64 direct-f32 wmma diagnostic

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`,
  guarded by
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.
- build:
  `sources/llama.cpp/build/hrx-v1-catalog-gfx1151`, Release, HRX target
  `gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`, compiled through
  CMake/Ninja.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M focused Q5_K dense prompt rows at p512, with
  synthetic p33 and p513 odd/tail rows generated from the same exported shapes.
- prior-art schedule source:
  Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p512-fa1-20260617-200349/`.
  Dominant pipeline is `matmul_q5_k_f32_f16acc_aligned_l`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`, `wg_denoms=[128,128,1]`,
  `LDS=22528`, `VGPR=192`, no spills, and 32 f16 WMMA sites.
- route or candidate:
  Direct-F32 RHS Q5_K WMMA probe cloned from the Q4/Q8 VK128 padded wave64
  skeleton, with Q5 high-bit-plane decode and a direct MUL_MAT selector path.
- compile report:
  `cache/hrxv1/gfx1151/q5-radv-vs-hrx-wmma-vk128-padded-w64-isa-20260618/`.
  The probe emitted 32 `v_wmma_f16_16x16x16_f16`, no spills, SGPR `32`, VGPR
  `164`, LDS `20480`, 32 `ds_load_b128`, and 64 stores. It is closer than the
  integer-dot Q5 route in opcode class, but still misses RADV's `64 ds_load_b64
  + 128 ds_load_u16_d16`, `22528` LDS, and 192 stores.
- correctness:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-focused-20260618-025116/`.
  p512 and p513 selected the WMMA route and passed focused CPU-reference gates.
  p33 correctly stayed on the existing `rows2_cols8_wg64` narrow route and
  passed.
- timing:
  same-runner focused p512 timing regressed versus current Q5 MMQL128 on all
  Q5 rows: Kcur `866.67 -> 971.27 us`, Qcur `1215.98 -> 2476.18 us`,
  ffn_out `7191.96 -> 17643.78 us`, ffn_gate `6332.38 -> 14489.87 us`.
- decision:
  reject for production promotion. This confirms that merely moving Q5 to the
  direct-F32 VK128 wave64 WMMA skeleton is not enough; the next Q5 work should
  target the remaining RADV LDS/load/store schedule deltas or a packed-dataflow
  variant rather than broadening this route.

## 2026-06-18 - qwen3 q6_k vk128 padded w64 direct-f32 wmma promotion

- source:
  `sources/llama.cpp` after adding
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`.
  It is default on `gfx1151`, can be rolled back with
  `GGML_HRX_DISABLE_Q6_K_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`, and
  remains opt-in elsewhere with
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.
- build:
  `sources/llama.cpp/build/hrx-v1-catalog-gfx1151`, Release, HRX target
  `gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`, compiled through
  CMake/Ninja.
- prior-art schedule source:
  Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/`.
  Dominant dense pipeline is `matmul_q6_k_f32_f16acc_aligned_l`, hash
  `0x6eebdfb4c3043b23`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`, `wg_denoms=[128,128,1]`,
  `LDS=22528`, `VGPR=192`, no spills, and 32 f16 WMMA sites.
- compile report:
  `cache/hrxv1/gfx1151/q6-radv-vs-hrx-wmma-vk128-padded-w64-isa-20260618/`.
  The HIP route emits 32 `v_wmma_f16_16x16x16_f16`, wave64, no spills, SGPR
  `36`, VGPR `156`, LDS `20480`, 32 `ds_load_b128`, and 64 stores. This is
  materially closer than the prior integer-dot default but still not exact
  RADV parity: RADV uses `22528` LDS, `64 ds_load_b64 + 128 ds_load_u16_d16`,
  and 192 stores.
- correctness:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-focused-20260618-030507/`.
  p512, p33, and p513 focused CPU-reference gates passed while selecting the
  VK128 route on all Q6 rows. A no-env p512 gate also passed and confirmed
  default gfx1151 selection.
- timing:
  same-runner focused p512 versus accepted current-best Q6 policy:
  Vcur `2024.14 -> 1261.13 us`, ffn_out `45578.26 -> 22575.58 us`,
  result_output `382338.19 -> 185386.36 us`.
- model A/B:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-model-ab-20260618-030914/`.
  Qwen3 30B Q6_K p512/fa1/r3 improved from `325.44` to `345.70 tok/s`.
  The traced r1 variant run confirms dense Q6 rows select the VK128 route while
  Q6 `MUL_MAT_ID` keeps the accepted grouped MMQ16 route.
- decision:
  accept as the current gfx1151 default Q6_K dense prompt route. Continue
  mechanical RADV schedule matching from the remaining LDS/load/store deltas;
  do not treat this as final Vulkan parity.

## 2026-06-18 - llama 3.1 8b q8_0 vk128 b64group rejection

- source:
  `sources/llama.cpp` after adding opt-in route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/`.
  Dominant dense pipeline is `matmul_q8_0_f32_f16acc_aligned_l`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`, `LDS=22528`, `VGPR=192`, no spills,
  32 f16 WMMA sites, 64 `ds_load_b64`, 128 `ds_load_u16_d16`, 128
  `ds_store_b16`, and 192 `buffer_store_b32`.
- schedule delta tested:
  preserve the direct-F32 VK128 wave64 WMMA skeleton, but change fragment reads
  from per-tile `ds_load_b128` to grouped `ds_read_b64` loads: four A
  fragments and four B fragments are loaded for each k-half before the 4x4
  WMMA block.
- compile report:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-b64group-isa-20260618/`.
  The probe matched RADV on 32 `v_wmma_f16_16x16x16_f16`, 64 `ds_load_b64`,
  zero `ds_load_b128`, and two barriers, with wave64, SGPR `28`, VGPR `196`,
  LDS `20480`, and no spills. It still misses RADV's `ds_load_u16_d16`, LDS
  store, and output store shape.
- correctness:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-b64group-focused-20260618-032500/`.
  p512 and p513 candidate-selected rows passed CPU-reference gates; p33 stayed
  on existing narrow Q8 routes and passed.
- timing:
  same-runner focused timing regressed badly versus accepted BN96/BN64
  packed-Q8_1 policy. p512: ffn_out `8443.89 -> 15560.60 us`, ffn_gate
  `7574.49 -> 12631.81 us`, result_output `65982.12 -> 105784.41 us`. p513:
  ffn_out `8263.92 -> 15854.83 us`, ffn_gate `7459.87 -> 14484.62 us`,
  result_output `67232.12 -> 122643.50 us`.
- decision:
  reject for production promotion. Matching the RADV `ds_load_b64` count alone
  is not sufficient; the next Q8_0 work needs to target the remaining
  cooperative-matrix LDS/writeback shape or a different way to recover RADV's
  output ownership without losing to the packed-Q8_1 route.

## 2026-06-18 - llama 3.1 8b q8_0 vk128 store-stage rejection

- source:
  `sources/llama.cpp` after adding opt-in route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_store_stage_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_STORE_STAGE_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-continued/`.
  The large route uses `matmul_q8_0_f32_f16acc_aligned_l`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`, `LDS=22528`, `VGPR=192`, no spills,
  32 f16 WMMA sites, 64 `ds_load_b64`, 128 `ds_load_u16_d16`, 128
  `ds_store_b16`, 192 `buffer_store_b32`, and only 2 barriers.
- schedule delta tested:
  preserve the direct-F32 VK128 wave64 WMMA skeleton but add a per-wave shared
  half output tile before global writeback, testing whether source-visible
  output staging can reproduce the RADV LDS footprint/store-side lowering.
- compile report:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-store-stage-isa-20260618/`.
  The probe compiled through CMake/Ninja with wave64, SGPR `22`, VGPR `145`,
  LDS `22528`, no spills, 32 `v_wmma_f16_16x16x16_f16`, 66 `ds_store_b16`,
  64 `ds_load_u16_d16`, and 64 `global_store_b32`. It matched RADV's LDS byte
  count but not the cooperative-matrix store schedule: the HIP route has 34
  barriers and only 64 scalar global stores versus RADV's 2 barriers and 192
  `buffer_store_b32`.
- correctness:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-store-stage-focused-20260618/`.
  Forced p512 and p513 large rows passed CPU-reference gates and selected the
  store-stage route. p33 stayed on existing narrow Q8 routes and passed.
- timing:
  same-runner focused timing regressed on every p512/p513 row versus accepted
  BN96/BN64 packed-Q8_1. p512: ffn_out `7930.25 -> 10013.10 us`, ffn_gate
  `7207.99 -> 10160.74 us`, result_output `64968.64 -> 98961.64 us`. p513:
  ffn_out `8020.62 -> 10732.69 us`, ffn_gate `7563.19 -> 12591.31 us`,
  result_output `66538.71 -> 109607.60 us`.
- decision:
  reject for production promotion. Explicit shared output staging is useful as
  a falsifier because it matches the RADV LDS allocation, but it is not the
  winning Vulkan cooperative-matrix store/lane-ownership schedule. The next Q8
  probe should target global-store lane ownership directly or drop lower than
  HIP C++ if that schedule cannot be expressed.

## 2026-06-18 - llama 3.1 8b q8_0 vk128 fullstore rejection

- source:
  `sources/llama.cpp` after adding opt-in route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_fullstore_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-continued/`.
  The shader source uses direct `coopMatStore` for full in-bounds aligned
  tiles and a staged scalar fallback only for unaligned or partial tiles.
- schedule delta tested:
  preserve the direct-F32 VK128 wave64 WMMA skeleton but split writeback into
  an unguarded full-tile store path and a guarded edge path, matching the
  Vulkan full-tile control structure more closely than the base HIP helper.
- compile report:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-fullstore-isa-20260618/`.
  The probe compiled through CMake/Ninja with wave64, SGPR `28`, VGPR `129`,
  LDS `20480`, no spills, 32 `v_wmma_f16_16x16x16_f16`, 32 `ds_load_b128`,
  2 barriers, and 128 `global_store_b32`. That is a real move from the base
  direct route's 64 stores, but still below RADV's 192 `buffer_store_b32` and
  missing RADV's `64 ds_load_b64`, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  and `22528` byte LDS shape.
- correctness:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-fullstore-focused-20260618/`.
  p512 and p513 large rows selected the fullstore route and passed
  CPU-reference gates. p33 stayed on existing narrow Q8 routes and passed.
- timing:
  same-runner focused timing regressed on every p512/p513 row versus accepted
  BN96/BN64 packed-Q8_1. p512: ffn_out `8209.91 -> 10046.26 us`, ffn_gate
  `7348.06 -> 10622.95 us`, result_output `66045.31 -> 100014.33 us`. p513:
  ffn_out `8252.34 -> 10091.96 us`, ffn_gate `7590.76 -> 14325.28 us`,
  result_output `67126.55 -> 113273.14 us`.
- decision:
  reject for production promotion. Full-tile/edge splitting is not sufficient
  to clone Vulkan's cooperative-matrix writeback. Either the next Q8 exact
  probe needs a lower-level store/lane spelling that reaches RADV's 192-store
  shape without the store-stage barriers, or effort should pivot back to the
  accepted packed-Q8_1 path where HRX is already closer.

## 2026-06-18 - qwen2.5 coder 7b q5_k mmql128 bquad rejection

- source:
  `sources/llama.cpp` after adding opt-in route
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_wg256_f32` behind
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`.
- prior-art schedule source:
  the positive Q4_K packed-path B-quad result in
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-focused-20260618-020428/`,
  plus the Q5_K Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p512-fa1-20260617-200349/`.
- schedule delta tested:
  preserve Q5_K MMQL128 packed-Q8_1/x4 dataflow, BM128, BN128, wave64, and
  BK_STEP1, but preload four B-cache rows across two adjacent WNITER positions
  before dot consumption. This directly tests whether the Q4_K B-cache
  LDS-read clustering win transfers to Q5_K.
- compile report:
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-compile-20260618/`. The probe built
  through CMake/Ninja with wave64, SGPR `50`, VGPR `169`, LDS `10240`, and no
  spills.
- correctness and route policy:
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-focused-20260618/`. Focused
  CPU-reference gates passed for p512, p33, and p513. Route traces show p512
  selected B-quad on the three packed Q5_K rows, p33 stayed on MMQ64, and exact
  p513 tails fell back to current MMQL128.
- timing:
  same-runner p512 focused timing regressed on every row versus current
  routing: Kcur `863.88 -> 877.07 us`, Qcur `1229.71 -> 1264.62 us`,
  ffn_out `7272.30 -> 7526.55 us`, and ffn_gate `6761.44 -> 6774.53 us`.
- decision:
  reject for production promotion. The first productive Q4 packed-path
  read-clustering axis is quant-family-specific and does not transfer to Q5_K
  as-is. Next Q5 work should inspect Q5-specific issue order or a different
  packed-RHS ownership axis rather than cloning Q4 B-quad again.

## 2026-06-18 - qwen3 30b q6_k vk128 b64group rejection

- source:
  `sources/llama.cpp` after adding opt-in route
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  Q6_K Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/`.
  The active RADV schedule is `matmul_q6_k_f32_f16acc_aligned_l` with
  BM128/BN128/BK32/WG256 wave64, 32 f16 WMMA sites, LDS `22528`,
  VGPR `192`, 64 `ds_load_b64`, 128 `ds_load_u16_d16`, 128 LDS b16 stores,
  and 192 global stores.
- schedule delta tested:
  preserve the accepted Q6 VK128 padded W64 direct-F32 WMMA route but group
  four A fragments and four B fragments before each 4x4 WMMA block, targeting
  RADV's `64 ds_load_b64` load-count axis.
- compile report:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-focused-20260618-050234/`.
  The probe built through CMake/Ninja with wave64, SGPR `40`, VGPR `197`,
  LDS `20480`, no spills, 32 `v_wmma_f16_16x16x16_f16`, 64 `ds_load_b64`,
  0 `ds_load_b128`, 2 barriers, and 64 `global_store_b32`.
- correctness and route policy:
  focused CPU-reference gates passed for p33, p512, and p513. Route traces
  show the B64GROUP provider selected on all Q6 rows for all three sizes.
- timing:
  same-runner focused timing regressed every row versus the accepted default:
  p33 Vcur `678.80 -> 798.20 us`, ffn_out `3360.94 -> 3581.94 us`,
  result_output `17916.01 -> 22896.35 us`; p512 Vcur
  `1231.62 -> 2428.77 us`, ffn_out `22822.28 -> 29021.04 us`,
  result_output `184755.28 -> 227402.03 us`; p513 Vcur
  `1032.27 -> 1157.09 us`, ffn_out `11579.62 -> 17035.95 us`,
  result_output `111122.29 -> 134430.43 us`.
- decision:
  reject for production promotion. This proves the work is still schedule
  convergence, not aggregate guessing: one exact RADV axis was matched, but it
  worsened HIP because the remaining RADV LDS-store/writeback/lane-ownership
  shape is still absent and VGPR pressure rose past the oracle. Do not repeat
  a standalone B64 load-count probe for Q6; combine it with the missing
  `ds_load_u16_d16`/LDS-store/writeback shape or pivot to a packed-path
  schedule family.

## 2026-06-18 - q4_k mmql128 bquad gfx1151 default regate

- source:
  `sources/llama.cpp` after changing
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` from opt-in to
  gfx1151 default for full-tile Q4_K packed-Q8_1/x4 prompt rows. Rollback is
  `GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`; opt-in outside
  gfx1151 remains
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`.
- prior-art schedule source:
  Q4_K Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/`
  and B-quad focused/model evidence in
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-focused-20260618-020428/`.
- acceptance artifact:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-default-regate-20260618-051326/`.
- correctness and route policy:
  focused CPU-reference gates passed for p33, p512, and exact p513 without
  setting the B-quad enable env. Route traces show p512 selected B-quad, p33
  stayed on MMQ64, and exact p513 fell back to current MMQL128. The rollback
  perf trace selected current MMQL128.
- timing:
  p512 same-runner default B-quad vs rollback MMQL128 was Kcur
  `1004.51 vs 1003.02 us`, Qcur `3878.57 vs 4091.33 us`, ffn_out
  `13806.01 vs 14990.07 us`, and ffn_gate `11124.93 vs 13488.00 us`.
- decision:
  promote to gfx1151 default only for full column tiles. This is a
  schedule-level promotion: the route follows the positive RADV LDS-read-window
  axis and has direct focused, odd-size, route, rollback, and model A/B
  evidence. Do not widen it to p513 tails without a tail-specific schedule win.

## 2026-06-18 - q4_k mmql128 bquad large-tail promotion

- source:
  `sources/llama.cpp` after extending
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` selection from
  full tiles only to gfx1151 large prompt tails with `cols >= 512`. Rollback
  for all B-quad remains
  `GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`; rollback for only
  large tails is
  `GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_TAIL_PROMPT=1`.
- prior-art schedule source:
  Q4_K p513 Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p513-fa1-20260617-200751/`.
  Vulkan uses the same large aligned route as p512 and covers p513 with a fifth
  column workgroup. The HRX B-quad kernel already has a full-tile/edge-tile
  split, so this probe tests that policy axis without introducing a new kernel
  source.
- focused probe:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-tail-probe-20260618-052036/`.
  With `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_TAIL_PROMPT=1`, p33, p512,
  and exact p513 CPU-reference gates passed. Routes proved p33 stayed on
  MMQ64, p512 selected B-quad, and p513 selected B-quad.
- focused timing:
  p513 default MMQL128 vs B-quad tail was Kcur `642.97 -> 605.69 us`,
  Qcur `2321.87 -> 2493.24 us`, ffn_out `8453.90 -> 7847.65 us`, and
  ffn_gate `7390.19 -> 7682.43 us`. Row effects are mixed but summed focused
  time improved slightly.
- model A/B:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-tail-llama31-model-ab-20260618-052222/`.
  Same-binary Llama 3.1 8B Q4_K_M p513/fa1 improved `564.62 -> 585.20 tok/s`.
  Route logs show baseline selected current MMQL128 for `570` Q4 dispatches,
  while tail selected B-quad for the same `570` dispatches.
- default-regate:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-tail-default-regate-20260618-052352/`.
  Default p513 selects B-quad, tail rollback selects current MMQL128, p512
  remains B-quad, and p33 remains MMQ64. Focused default vs tail rollback was
  Kcur `608.39 vs 614.95 us`, Qcur `2477.16 vs 2347.09 us`, ffn_out
  `8248.31 vs 8013.47 us`, and ffn_gate `7617.92 vs 8274.78 us`.
- decision:
  accept for gfx1151 production-width Q4_K prompt tails with `cols >= 512`.
  Do not broaden B-quad to narrow odd sizes; p33 remains explicitly on the
  narrow route. Future Q4 work should move to the remaining Vulkan/RADV
  dataflow deltas rather than re-testing this tail policy.

## 2026-06-18 - q8_0 exact Vulkan schedule checkpoint and gfx1151 default policy

- source:
  `sources/llama.cpp` after changing the Q8_0 packed-Q8_1 selector so
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x96_wg256_f32` is default on gfx1151 for
  `cols >= 64`, with
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32` as the narrow/odd fallback.
  Rollbacks are
  `GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1` and
  `GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.
- prior-art schedule source:
  Q8_0 Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/`
  and cooperative schedule extraction
  `cache/hrxv1/gfx1151/q8_0-coopmat-schedule-extract-20260618/`.
  Vulkan's large route is `matmul_q8_0_f32_f16acc_aligned_l` with
  `BM128/BN128/BK32/WG256`, 32 f16 WMMA sites, `LDS=22528`, and cooperative
  matrix load/store lowering.
- direct-WMMA status:
  B64GROUP, STORE_STAGE, FULLSTORE, B64GROUP_FULLSTORE, and HI probes all
  passed focused correctness but regressed versus the packed BN96/BN64 policy.
  The closest HIP C++ direct route still misses RADV's cooperative-matrix
  writeback/lane-ownership shape.
- default-regate:
  `cache/hrxv1/gfx1151/q8_0-bn96-default-regate-20260618-053457/`.
  Focused CPU-reference gates passed for p33, p512, and exact p513 with no Q8
  enable envs set. Routes show p33 selected BN64, while p512 and p513 selected
  BN96. Rollback with both BN96 and BN64 disabled selected the older
  MMQ128x32 route.
- focused timing:
  p512 rollback MMQ128x32 vs default BN96 was Vcur
  `576.34 -> 636.59 us`, Qcur `2316.92 -> 2030.16 us`, ffn_out
  `11496.55 -> 8133.81 us`, ffn_gate `32026.66 -> 7282.08 us`, and
  result_output `241314.48 -> 66272.21 us`. p513 was Vcur
  `616.80 -> 660.66 us`, Qcur `2178.30 -> 2063.83 us`, ffn_out
  `13131.21 -> 8243.04 us`, ffn_gate `26280.68 -> 7362.14 us`, and
  result_output `255766.02 -> 67142.86 us`.
- environment blocker:
  the next source-visible primitive to test would be a rocWMMA or equivalent
  matrix-fragment store path, but `/srv/vm-shared/rocm/rocm-head` does not
  contain `rocm/include/rocwmma/rocwmma.hpp`. Exact-clone work on this axis
  should therefore move to an installed matrix-store-capable path or lower-level
  codegen rather than another scalarized HIP store variant.
- decision:
  accept BN96 plus BN64 fallback as the current gfx1151 production policy while
  keeping exact Vulkan parity work focused on the cooperative-store route. This
  is not the parity endpoint; it prevents the selector from defaulting to a
  known-worse Q8_0 packed route while schedule-level work continues.

## 2026-06-18 - Q5_K VK128 B64GROUP oracle-axis rejection

- source:
  `sources/llama.cpp` adds the opt-in route
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  RADV `matmul_q5_k_f32_f16acc_aligned_l` from
  `cache/hrxv1/gfx1151/q5-radv-vs-hrx-wmma-vk128-padded-w64-isa-20260618/`.
  The probe mechanically targets the `64 ds_load_b64` / `32 WMMA` axis while
  preserving the rejected direct-F32 VK128 wave64 route.
- compile evidence:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-focused-20260618-054536/`.
  Built through CMake/Ninja, wave64, SGPR `32`, VGPR `199`, LDS `20480`, no
  spills, `32` WMMA sites, `64 ds_load_b64`, `0 ds_load_b128`, `2` barriers,
  and `64 global_store_b32`.
- focused gates:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on rows2/cols8,
  while p512 and p513 selected the candidate.
- focused timing:
  p512 current vs B64GROUP was Kcur `887.78 -> 1046.75 us`, Qcur
  `1232.95 -> 2471.40 us`, ffn_out `7308.67 -> 17817.09 us`, and ffn_gate
  `6662.22 -> 16905.31 us`. p513 current vs B64GROUP was Kcur
  `891.04 -> 964.61 us`, Qcur `1594.35 -> 3375.88 us`, ffn_out
  `9362.91 -> 23583.29 us`, and ffn_gate `8558.62 -> 20055.20 us`.
- decision:
  reject for production. Matching the RADV `ds_load_b64` count alone is not
  enough; Q5 parity work needs the remaining LDS-store/writeback/lane-ownership
  deltas or a return to packed-Q8_1 schedule axes.

## 2026-06-18 - Q6_K VK128 fullstore oracle-axis rejection

- source:
  `sources/llama.cpp` adds the opt-in route
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_fullstore_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  RADV `matmul_q6_k_f32_f16acc_aligned_l` from
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/`.
  The probe mechanically targets the full-tile writeback axis while preserving
  the accepted VK128 padded wave64 route.
- compile evidence:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-fullstore-focused-20260618-055935/`.
  Built through CMake/Ninja, wave64, SGPR `36`, VGPR `156`, LDS `20480`, no
  spills, `32` WMMA sites, `32 ds_load_b128`, `128 global_store_b32`, and `2`
  barriers.
- focused gates:
  p33, p512, and p513 CPU-reference gates passed. Route traces selected the
  candidate on all three Q6 rows for each size.
- focused timing:
  p33 current vs fullstore was Vcur `679.31 -> 679.32 us`, ffn_out
  `3204.62 -> 3223.13 us`, and result_output `18061.42 -> 17961.53 us`.
  p512 was Vcur `1246.93 -> 1201.46 us`, ffn_out `22209.12 -> 22331.66 us`,
  and result_output `185590.17 -> 183399.28 us`. p513 was Vcur
  `1032.30 -> 1020.43 us`, ffn_out `11798.46 -> 11725.39 us`, and
  result_output `113830.84 -> 113984.53 us`.
- decision:
  reject for production. The store split moved one ISA axis toward RADV without
  spills, but the runtime signal is mixed and small. Keep mechanically chasing
  the exact winning Vulkan schedule; the next useful direct-WMMA Q6 attempt
  needs to combine the remaining LDS footprint, b64/u16 LDS load/store, and
  cooperative writeback/lane-ownership deltas rather than promoting this
  isolated store policy.

## 2026-06-18 - Q6_K VK128 B64GROUP+fullstore oracle-axis rejection

- source:
  `sources/llama.cpp` adds the opt-in route
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_fullstore_f16acc_wg256_f32`
  behind
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- prior-art schedule source:
  RADV `matmul_q6_k_f32_f16acc_aligned_l` from
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/`.
  The probe mechanically combines the previously isolated grouped-fragment-load
  and full-tile writeback axes.
- compile evidence:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-fullstore-focused-20260618-060839/`.
  Built through CMake/Ninja, wave64, SGPR `40`, VGPR `196`, LDS `20480`, no
  spills, `32` WMMA sites, `64 ds_load_b64`, `128 global_store_b32`, and `2`
  barriers.
- focused gates:
  p33, p512, and p513 CPU-reference gates passed. Route traces selected the
  candidate on all three Q6 rows for each size.
- focused timing:
  p33 current vs candidate was Vcur `678.53 -> 798.70 us`, ffn_out
  `3266.17 -> 3836.85 us`, and result_output `17918.60 -> 23491.26 us`.
  p512 was Vcur `1229.96 -> 2383.51 us`, ffn_out `22244.78 -> 28722.42 us`,
  and result_output `180519.25 -> 232795.11 us`. p513 was Vcur
  `1031.38 -> 1138.76 us`, ffn_out `12337.31 -> 16923.53 us`, and
  result_output `112612.90 -> 135186.16 us`.
- decision:
  reject for production. Combining the b64 fragment-load axis with the
  full-tile store axis still regresses every focused row. The remaining direct
  Q6 parity problem is not recoverable by these source-visible pivots alone; it
  likely needs the lower-level cooperative-matrix LDS/load/store/lane-ownership
  lowering that RADV emits, or a different packed-path schedule.

## 2026-06-18 - qwen3 30b q6_k vulkan oracle odd tail rows

- source:
  Vulkan prior capture only; no source change.
- build:
  `build/vulkan-gfx1151`, Release, Vulkan backend, RADV STRIX_HALO.
- model/op/shape:
  Qwen3 30B Q6_K dense prompt matmul, `p33/n0/fa1` and `p513/n0/fa1`,
  `b=1024`, `ub=1024`, one no-warmup repetition on `Vulkan0`.
- route or candidate:
  same-machine Vulkan oracle prior for Q6_K odd and tail prompt regimes.
- baseline command:
  `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench build/vulkan-gfx1151/bin/llama-bench --model shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf --prompt <33|513> --gen 0 --batch 1024 --ubatch 1024 --flash-attn 1 --repetitions 1 --device Vulkan0`.
- variant command:
  not applicable; this is Vulkan prior evidence.
- artifacts:
  - `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p33-fa1-20260618-061613/`
  - `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p513-fa1-20260618-061619/`
- correctness:
  both benchmark rows completed and reported `backends=Vulkan`; these are
  schedule-oracle captures, not HRX correctness gates.
- result:
  p33 captured `27` pipeline identities, `1292` dispatch signatures, `37`
  normalized shape signatures, and `27` SPIR-V asm files. p513 captured `28`
  pipeline identities, `1435` dispatch signatures, `39` normalized shape
  signatures, and `28` SPIR-V asm files.
- decision:
  accept as Q6_K odd/tail oracle evidence. p33 uses the medium aligned
  `matmul_q6_k_f32_f16acc_aligned_m` route with
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `LDS=11264`, `VGPR=144`, no
  spills, `16` WMMA, `48 ds_load_b64`, and `96 buffer_store_b32`. p513 uses
  the large aligned `matmul_q6_k_f32_f16acc_aligned_l` route with
  `spec=[256,128,128,32,64,64,2,16,16,16,64]`, `LDS=22528`, `VGPR=192`, no
  spills, `32` WMMA, `64 ds_load_b64`, and `192 buffer_store_b32`, plus
  `split_k_reduce` tail reductions. This reinforces that aggregate model
  throughput is only a boulder selector; Q6 parity work must mechanically clone
  or explain the exact Vulkan schedule at p33, p512, and p513 before promotion.

## 2026-06-18 - Q6_K VK64 padded44 medium-route narrow prompt promotion

- source:
  `sources/llama.cpp` adds
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`,
  parameterizes the existing Q6 VK128 direct-WMMA source for BM/BN, and builds
  the new HSACO through CMake/Ninja.
- prior-art schedule source:
  Qwen3 30B Q6_K p33 Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p33-fa1-20260618-061613/`.
  Vulkan uses medium aligned `matmul_q6_k_f32_f16acc_aligned_m` with
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `LDS=11264`, `VGPR=144`, no
  spills, `16` WMMA, `48 ds_load_b64`, `64 ds_load_u16_d16`, `64 ds_store_b16`,
  and `96 buffer_store_b32`.
- route or candidate:
  BM64/BN64/BK32, 256-thread wave64 direct-F32 WMMA route with a 44-half LDS
  stride, default on gfx1151 for `16 <= cols <= 64`. Rollback:
  `GGML_HRX_DISABLE_Q6_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_PROMPT=1`.
- compile evidence:
  `cache/hrxv1/gfx1151/q6-wmma-vk64-padded44-medium-focused-20260618-062606/`.
  HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_wg256.hsaco`.
  Static LDS footprint matches RADV at `11264` bytes. Visible ISA still differs:
  `8` WMMA sites, `16 global_store_b32`, and `2` barriers.
- focused gates:
  p33 CPU-reference gate passed and selected VK64 for Vcur, ffn_out, and
  result_output. Rollback gate passed and selected the previous VK128 route.
  Post-promotion p512 focused gate passed and still selected VK128.
- focused timing:
  p33 default VK128 vs VK64 was Vcur `678.27 -> 397.77 us`, ffn_out
  `3175.03 -> 2239.18 us`, and result_output `18046.03 -> 11744.51 us`.
- model timing:
  Qwen3 30B Q6_K p33/fa1/no-warmup/r1 HRX model A/B improved
  `96.68 -> 109.61 tok/s`.
- decision:
  accept as the gfx1151 Q6_K narrow/odd prompt default. This is a targeted
  schedule-led improvement from the Vulkan medium route, not a claim of exact
  parity: visible HIP ISA still differs from RADV's cooperative-matrix
  load/store shape, and p512/p513 remain on the large VK128 policy.

## 2026-06-18 - Q5_K VK128 fullstore probe rejection

- source:
  `sources/llama.cpp` adds
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_fullstore_f16acc_wg256_f32`
  as an opt-in route, guarded by
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; compiled through CMake/Ninja.
- prior-art schedule source:
  Q5_K p512/p513 Vulkan oracle
  `matmul_q5_k_f32_f16acc_aligned_l`:
  `spec=[256,128,128,32,64,64,2,16,16,16,64]`, `LDS=22528`, `VGPR=192`,
  no spills, `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, and `192 buffer_store_b32`.
- route or candidate:
  direct-F32 Q5_K VK128 padded wave64 route with full in-bounds 16x16 tile
  stores and guarded edge fallback.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file q5_prompt_<p512|p513>.txt --output csv`
  with current default Q5 routing.
- variant command:
  same focused files with
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-fullstore-focused-20260618-065217/`.
- compile evidence:
  HSACO reports wave64, SGPR `32`, VGPR `164`, LDS `20480`, no spills,
  `32` WMMA sites, `128 global_store_b32`, and `2` barriers.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on rows2; p512
  and p513 selected the fullstore candidate.
- timing:
  p512 regressed on Qcur `1237.64 -> 2344.02 us`, ffn_out
  `7204.77 -> 16963.43 us`, and ffn_gate `6912.03 -> 14368.33 us`.
  p513 regressed on Qcur `1595.89 -> 2510.60 us`, ffn_out
  `9548.74 -> 18271.78 us`, and ffn_gate `8779.65 -> 16906.24 us`.
- decision:
  reject for production. Moving from `64` to `128` HIP global stores is not
  sufficient and generally makes the direct route slower. The next Q5 direct
  attempt should mechanically target the Vulkan cooperative-matrix
  store/lane-ownership schedule or use a lower-level source form that can emit
  it, rather than testing another wrapper-only tile variant.

## 2026-06-18 - Q8_0 BN104 packed compile rejection

- source:
  `sources/llama.cpp` adds
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x104_wg256_f32` as an opt-in route,
  guarded by `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X104_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; compiled through CMake/Ninja.
- prior-art schedule source:
  accepted BN96 packed-Q8_1 route and rejected BN112/BN128 pressure brackets,
  against the Q8_0 p512/p513 Vulkan oracle. Direct Vulkan coopmat cloning is
  currently blocked by missing HIP/rocWMMA matrix-store primitives, so this was
  a bounded packed-path pressure probe rather than another scalarized WMMA
  wrapper clone.
- route or candidate:
  BM64/BN104, `COLS_PER_THREAD=26`, direct-A/staged-B packed-Q8_1 dataflow.
- compile evidence:
  `cache/hrxv1/gfx1151/q8_0-mmq64x104-compile-20260618/`. The target HSACO
  reports wave32, SGPR `28`, VGPR `192`, LDS `3536`, `vgpr_spill_count=9`,
  and `private_segment_fixed_size=40`.
- comparison:
  accepted BN96 is VGPR `181` with no spills; rejected BN112 is VGPR `192`
  with `24` spills; rejected BN128 is VGPR `192` with `55` spills.
- decision:
  reject at the compile-resource gate before runtime timing. BN104 proves the
  simple column-widening axis hits the same 192-VGPR cliff before it can reach
  a Vulkan-like 128-column production tile. The next Q8_0 packed candidate
  should change register tile, split accumulation, or output ownership rather
  than raising `COLS_PER_THREAD` above `24`.

## 2026-06-18 - Q6_K VK128 store-stage probe rejection with tail signal

- source:
  `sources/llama.cpp` adds
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_store_stage_f16acc_wg256_f32`
  as an opt-in route, guarded by
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_STORE_STAGE_F16ACC_WG256_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; compiled through CMake/Ninja.
- prior-art schedule source:
  Qwen3 30B Q6_K p512/p513 Vulkan oracle
  `matmul_q6_k_f32_f16acc_aligned_l`:
  `spec=[256,128,128,32,64,64,2,16,16,16,64]`, `LDS=22528`, no spills,
  `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, and `192 buffer_store_b32`.
- route or candidate:
  direct-F32 Q6_K VK128 padded wave64 route that stages f16 accumulator values
  through LDS before scalar global writeback. This isolates the LDS
  footprint/output-stage axis after B64GROUP, FULLSTORE, and
  B64GROUP_FULLSTORE rejected.
- compile evidence:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-store-stage-compile-20260618/`.
  The HSACO reports wave64, SGPR `36`, VGPR `148`, LDS `22528`, no spills,
  `32` WMMA, `66 ds_store_b16`, `64 ds_load_u16`, `64 global_store_b32`,
  and `34` barriers.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on the accepted
  VK64 narrow route; p512 and p513 selected the store-stage candidate.
- timing:
  p512 was flat-to-regressive versus current routing: Vcur
  `1223.75 -> 1233.23 us`, ffn_out `22550.84 -> 23574.10 us`, and
  result_output `187339.61 -> 187393.19 us`. p513 showed a useful tail signal:
  Vcur `1032.48 -> 1027.26 us`, ffn_out `11686.69 -> 11651.95 us`, and
  result_output `113641.02 -> 108827.76 us`.
- decision:
  reject for broad production promotion. Matching the RADV LDS footprint alone
  is not enough because the HIP source spelling introduces many barriers and
  still misses RADV's low-barrier cooperative-matrix writeback/lane ownership.
  Keep the route opt-in as evidence for a future p513/tail-only policy test.

## 2026-06-18 - Q8_0 BN112 split-qsum packed-path promotion

- source:
  `sources/llama.cpp` adds
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` as a
  CMake-built HIP C++ route. It is default on gfx1151 for Q8_0 prompt rows
  with `cols >= 128`; rollback:
  `GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X112_SPLITQSUM_PROMPT=1`.
- prior-art schedule source:
  accepted BN96 packed-Q8_1 route plus the rejected BN104/BN112 pressure
  brackets, against the Q8_0 p512/p513 Vulkan oracle. Direct Vulkan
  cooperative-store cloning remains blocked in current HIP C++/ROCm, so this
  is a bounded packed-path register-tile probe.
- route or candidate:
  BM64/BN112, direct-A/staged-B packed-Q8_1 dataflow, split `qsum` into two
  14-column chunks to reduce live register pressure.
- compile evidence:
  `cache/hrxv1/gfx1151/q8_0-mmq64x112-splitqsum-compile-20260618/`.
  HSACO reports wave32, SGPR `28`, VGPR `134`, LDS `3808`, no spills. This
  fixes the simple BN112 route's `VGPR=192`, `24` spills, and private segment
  `100`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on BN64; p512 and
  p513 selected split-qsum.
- default/rollback regate:
  `cache/hrxv1/gfx1151/q8_0-mmq64x112-splitqsum-default-regate-20260618/`.
  Default p33 stayed on BN64, default p512/p513 selected split-qsum, and
  rollback returned p512 to BN96.
- focused timing:
  p512 improved every row versus BN96/BN64: Vcur `664.28 -> 525.19 us`,
  Qcur `2038.53 -> 1990.77 us`, ffn_out `8273.41 -> 7583.48 us`, ffn_gate
  `7640.07 -> 7040.21 us`, result_output `65759.52 -> 64159.07 us`.
  p513 also improved every row, including result_output
  `66845.24 -> 64617.43 us`.
- model timing:
  Llama 3.1 8B Q8_0 p512/fa1/r3 improved `430.303 -> 435.902 tok/s`; p513
  single-graph/fa1/r3 improved `409.652 -> 424.678 tok/s`.
- decision:
  accept as gfx1151 production-width Q8_0 default. This is still not Vulkan
  parity and not an exact cooperative-matrix store clone; it is a demonstrated
  packed-path improvement that proves the prior simple column-widening spill
  wall was mostly a `qsum` live-range problem.

## 2026-06-18 - Q6_K store-stage tail-only recheck rejection

- source:
  no selector or kernel source change. This rechecked whether the existing
  opt-in `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_store_stage_f16acc_wg256_f32`
  should be defaulted only for p513/large-tail prompt rows.
- prior-art schedule source:
  Qwen3 30B Q6_K p513 Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p513-fa1-20260618-061619/`.
  Vulkan keeps the large aligned route for p513 and adds split-K reductions.
- route or candidate:
  direct-F32 Q6_K VK128 padded wave64 store-stage route. It matches the RADV
  `22528` byte LDS footprint but still has explicit LDS staging, `34`
  barriers, scalar global writeback, and no cooperative-matrix store lowering.
- focused evidence:
  `cache/hrxv1/gfx1151/q6-store-stage-tail-regate-probe-20260618-current/`.
  CPU-reference gates passed for p33, p512, and p513 under default and
  store-stage env. Focused p513 showed a small repeated signal on two rows:
  ffn_out `11876.06 -> 11609.42 us` and result_output
  `112353.98 -> 109342.20 us`; p512 result_output regressed
  `184830.25 -> 187056.97 us`.
- model evidence:
  `cache/hrxv1/gfx1151/q6-store-stage-tail-model-ab-20260618-current/`.
  Qwen3 30B Q6_K p513/fa1/r3 regressed `187.730 -> 181.227 tok/s`.
  Route traces prove baseline selected the accepted VK128 provider for `576`
  dense Q6 prompt routes, while the variant selected store-stage for `576`.
- decision:
  reject tail-only promotion. The focused p513 signal does not survive
  model-level A/B, so do not add a default selector. The useful conclusion is
  narrower: matching RADV LDS allocation by explicit store staging is still not
  the Vulkan cooperative-matrix writeback schedule.

## 2026-06-18 - Q4_K B-quad-CR QK-only policy rejection

- source:
  `sources/llama.cpp` dirty after adding the opt-in selector
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_CR_QK_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; selector built and linked through
  CMake/Ninja.
- model/shape:
  DeepSeek R1 Qwen 14B Q4_K_M p512/fa1 and Llama 3.1 8B Q4_K_M p512/fa1,
  plus focused p33, p512, and p513 Q4_K CPU-reference gates.
- route or candidate:
  row-family split for the B-quad-CR packed-Q8_1/x4 MMQL128 probe. Select
  B-quad-CR only for full-tile K/Q-style rows with `k <= 5120`,
  `rows <= 5120`, and `cols % 128 == 0`; keep accepted B-quad for FFN and
  tail rows, and keep p33 on the narrow routes.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file q4_prompt_p512.txt --output csv`
  with current default B-quad routing, then same-binary `llama-bench` model
  A/B with `GGML_HRX_TRACE_ROUTES=1`.
- variant command:
  same focused and model commands with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_CR_QK_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-qk-policy-20260618/`.
- profile/timing:
  focused artifact above and model summary
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-qk-policy-model-ab-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p512
  selected B-quad-CR only for the two K/Q-style rows, selected accepted B-quad
  for FFN rows, p513 selected accepted B-quad for all Q4_K rows, and p33 stayed
  narrow.
- timing:
  DeepSeek p512 improved only `246.404409 -> 247.043538 tok/s` (`1.0026x`);
  Llama 3.1 p512 regressed `444.387307 -> 443.123607 tok/s` (`0.9972x`).
- decision:
  reject default promotion. Keep the opt-in selector only as a reproducer for
  future finer-grained row scheduling.
- notes:
  This was a mechanical follow-up to the B-quad-CR focused signal, not an
  aggregate-first route change. The result says the K/Q consume-order signal is
  too small and model-dependent to default; further Q4 work should return to
  exact Vulkan/RADV schedule deltas and focused kernel/schedule A/B.

## 2026-06-18 - Q8_0 BN128 split-qsum full-column promotion

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` as a CMake-built
  HIP C++ route and defaulting it only for `cols % 128 == 0` on gfx1151.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  Llama 3.1 8B Q8_0 p512/fa1, with focused p33, p512, and p513 Q8_0 prompt
  rows.
- route or candidate:
  BM64/BN128 direct-A/staged-B packed-Q8_1 route with split `qsum` live ranges.
  This is a bounded packed-path move toward Vulkan's 128-column large-route
  denominator after direct cooperative-matrix store cloning remained blocked.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <q8 rows> --output csv`
  with current BN112 split-qsum default, plus same-binary `llama-bench` p512
  with rollback env.
- variant command:
  broad focused probe with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_PROMPT=1`; final default
  regate used no env and rollback
  `GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-focused-20260618/` and
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-default-regate-20260618/`.
- profile/timing:
  focused artifact above and model artifact
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-default-model-ab-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Default route traces prove
  p33 stays BN64, p512 selects BN128 split-qsum, p513 stays BN112 split-qsum,
  and p512 rollback returns to BN112.
- timing:
  compile gate passed with wave32, SGPR `27`, VGPR `152`, LDS `4352`, no
  spills. Focused p512 improved the large rows despite a Vcur regression:
  Qcur `1952.62 -> 1903.90 us`, ffn_gate `6844.92 -> 6394.20 us`, and
  result_output `62651.74 -> 56614.31 us`. Broad p513 selected BN128 but
  regressed every row, so the default guard is `cols % 128 == 0`.
- model evidence:
  Llama 3.1 8B Q8_0 p512/fa1/r3 default BN128 improved over rollback BN112:
  `440.297327 -> 456.488968 tok/s` (`1.0368x`). Route traces show `477`
  dense Q8_0 BN128 routes under default versus `477` BN112 routes under
  rollback.
- decision:
  accept as gfx1151 full-column Q8_0 default only. Do not broaden to p513/tail
  rows.
- notes:
  This is still not Vulkan parity and not an exact RADV cooperative-matrix
  clone. The useful schedule conclusion is that split-qsum fixes the old BN128
  spill wall, and exact full columns can use the 128-column denominator, while
  odd/tail columns need the narrower BN112 policy.

## 2026-06-18 - Q8_0 BN128 split-qsum8 live-range rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum8_wg256_f32` as a CMake-built
  opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Llama 3.1 8B Q8_0 p33, p512, and p513 prompt rows.
- route or candidate:
  BM64/BN128 direct-A/staged-B packed-Q8_1 route with qsum chunk size reduced
  from 16 to 8 columns.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <q8 rows> --output csv`
  with current default Q8_0 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM8_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum8-focused-20260618/`.
- profile/timing:
  same focused artifact, plus compile evidence in
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum8-compile-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p33
  stayed on BN64 while p512 and p513 selected split-qsum8 under the opt-in env.
- timing:
  compile evidence improved register pressure from accepted BN128 split-qsum
  VGPR `152` to VGPR `120`, with no spills and unchanged LDS `4352`. Focused
  timing rejected the route: p512 improved Vcur, Qcur, and ffn_out, but
  regressed ffn_gate `6303.30 -> 6843.30 us` and result_output
  `56355.07 -> 61269.62 us`; p513 regressed every row.
- decision:
  reject production promotion. Keep accepted BN128 split-qsum16 for exact
  full-column rows.
- notes:
  This was a bounded schedule-axis test, not blind tuning. It proves that
  reducing qsum live range alone is not the active limiter for dominant Q8_0
  prompt rows; the extra chunk-loop overhead loses too much throughput.

## 2026-06-18 - Q8_0 BM32/BN128 output-ownership rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq32x128_wg256_f32` as a CMake-built opt-in
  HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Llama 3.1 8B Q8_0 p33, p512, and p513 prompt rows.
- route or candidate:
  BM32/BN128 direct-A/staged-B packed-Q8_1 route. It preserves BN128
  full-column coverage but changes output ownership from BM64/COLS32
  split-qsum to BM32/COLS16 with a simple qsum tile.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <q8 rows> --output csv`
  with current default Q8_0 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ32X128_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq32x128-focused-20260618/`.
- profile/timing:
  same focused artifact, plus compile evidence in
  `cache/hrxv1/gfx1151/q8_0-mmq32x128-compile-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. p512 and p513 selected
  BM32/BN128 under the opt-in env; p33 stayed on the narrow/default path.
- timing:
  compile evidence passed with wave32, SGPR `27`, VGPR `135`, LDS `4352`, no
  spills. Focused timing rejected the route: every p512 and p513 row regressed
  versus current default routing. p512 ffn_out regressed
  `7043.92 -> 8482.13 us`, ffn_gate `6416.43 -> 7242.67 us`, and
  result_output `55913.71 -> 64160.60 us`; p513 result_output regressed
  `63218.36 -> 76252.10 us`.
- decision:
  reject production promotion. Keep accepted BN128 split-qsum16 for exact
  full-column rows.
- notes:
  This was the remaining packed-path output-ownership bracket after the direct
  cooperative-store Q8 clone was blocked by missing rocWMMA/cooperative matrix
  store support in the active ROCm tree. Lower register pressure did not
  translate to throughput; the lost row-tile amortization dominates.

## 2026-06-18 - Q8_0 VK128 B64GROUP plus store-stage rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_store_stage_f16acc_wg256_f32`
  as a CMake-built opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Llama 3.1 8B Q8_0 p33, p512, and p513 prompt rows.
- route or candidate:
  direct-F32 VK128 wave64 WMMA route combining grouped `ds_read_b64` A/B
  fragment loads with LDS output staging. This was the most direct
  source-visible combination of RADV large-route axes after isolated
  B64GROUP, FULLSTORE, and STORE_STAGE probes failed.
- baseline command:
  focused `test-backend-ops perf -b HRX0 --test-file <q8 rows> --output csv`
  with current default Q8_0 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_STAGE_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-b64group-store-stage-focused-20260618/`.
- profile/timing:
  same focused artifact; compile/ISA notes in
  `cache/hrxv1/gfx1151/catalog-validate-b64group-store-stage-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p512 and
  p513 selected the candidate on the large direct rows while p33 stayed on the
  existing narrow routes.
- timing:
  compile evidence: wave64, SGPR `28`, VGPR `195`, LDS `22528`, no spills,
  `32` f16 WMMA, `64` `ds_load_b64`, `64` `ds_load_u16_d16`, `66`
  `ds_store_b16`, `64` global stores, and `34` barriers. Focused timing
  rejected the route: p512 ffn_out `7098.06 -> 15934.15 us`, ffn_gate
  `6312.28 -> 12520.22 us`, result_output `56219.21 -> 105243.50 us`; p513
  ffn_out `7843.30 -> 16267.99 us`, ffn_gate `7513.31 -> 14443.90 us`,
  result_output `63883.26 -> 122219.81 us`.
- decision:
  reject production promotion. Keep the route opt-in as evidence only.
- notes:
  This closes the source-visible HIP C++ combination of RADV's grouped LDS
  fragment-load and LDS-footprint/output-stage axes. The remaining direct-Q8
  Vulkan gap is still the true cooperative-matrix store/lane-ownership
  lowering: RADV has only two barriers, `128` `ds_load_u16_d16`, `128`
  `ds_store_b16`, and `192` buffer stores. The HIP C++ spelling reaches the
  high-level tile and LDS budget but pays too many barriers and still emits a
  scalarized writeback schedule.

## 2026-06-18 - Q6_K VK128 B64GROUP plus store-stage rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_b64group_store_stage_f16acc_wg256_f32`
  as a CMake-built opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Qwen3/Q6-derived p33, p512, and p513 prompt rows.
- route or candidate:
  direct-F32 VK128 wave64 WMMA route combining grouped `ds_read_b64` A/B
  fragment loads with LDS output staging.
- baseline command:
  focused `test-backend-ops perf -b HRX0 --test-file <q6 rows> --output csv`
  with current default Q6_K routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_STAGE_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-store-stage-focused-20260618/`.
- profile/timing:
  same focused artifact; compile/ISA notes in
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-store-stage-compile-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p33
  stayed on the accepted VK64 narrow route, while p512 and p513 selected the
  combined candidate.
- timing:
  compile evidence: wave64, SGPR `40`, VGPR `196`, LDS `22528`, no spills,
  `32` f16 WMMA, `64` `ds_load_b64`, `64` `ds_load_u16`, `66`
  `ds_store_b16`, `64` global stores, `34` barriers, and `367` `s_waitcnt`.
  Focused timing rejected the route: p512 Vcur `1253.70 -> 2442.08 us`,
  ffn_out `22306.27 -> 28932.14 us`, result_output
  `182744.78 -> 230756.64 us`; p513 Vcur `1041.04 -> 1151.76 us`,
  ffn_out `11754.13 -> 16996.90 us`, result_output
  `110050.98 -> 136412.67 us`.
- decision:
  reject production promotion. Keep the route opt-in as evidence only.
- notes:
  This closes the source-visible Q6_K HIP C++ combination of RADV's grouped LDS
  fragment-load and LDS-footprint/output-stage axes. The remaining direct-Q6
  Vulkan gap is the same structural cooperative-matrix load/store/lane
  ownership lowering seen on Q8: RADV reaches two barriers, `128`
  `ds_load_u16_d16`, `128` LDS b16 stores, and `192` buffer stores, while the
  HIP C++ store-stage spelling pays many barriers and scalarized writeback.

## 2026-06-18 - Q5_K MMQL128 B-pair packed-path rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bpair_wg256_f32` as a CMake-built
  opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Qwen2.5 Coder 7B Q5_K_M-derived p33, p512, and p513 prompt rows.
- route or candidate:
  packed-Q8_1/x4 MMQL128 B-cache read-clustering probe. It preserves BM128,
  BN128, wave64, BK_STEP1, TM4, TN2, and WNITER8, but preloads only the two
  B-cache rows for each TN=2 micro-iteration before dot consumption.
- baseline command:
  focused `test-backend-ops perf -b HRX0 --test-file q5_prompt_p512.txt
  --output csv` with current Q5 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BPAIR_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-mmql128-bpair-focused-20260618/`.
- profile/timing:
  same focused artifact; compile HSACO at
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q5_k_q8_1_x4_mmql128_bpair.hsaco`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p512
  selected B-pair for the three packed Q5 rows, p33 stayed on existing narrow
  routes, and p513 fell back to current MMQL128.
- timing:
  compile evidence: wave64, SGPR `50`, VGPR `149`, LDS `10240`, no spills.
  Focused p512 timing rejected the route: Kcur `870.47 -> 876.16 us`, Qcur
  `1220.50 -> 1268.61 us`, ffn_out `7342.03 -> 7576.46 us`; ffn_gate was the
  only win at `6940.84 -> 6729.86 us`.
- decision:
  reject production promotion. Keep the route opt-in as evidence only.
- notes:
  This lower-live-state B-pair probe plus the earlier B-quad rejection shows
  the Q4 B-cache clustering win does not transfer directly to Q5_K. The next
  Q5 packed-path work should be Q5-specific rather than another local
  B-cluster-size clone.

## 2026-06-18 - Q8_0 MMQ64x104 split-qsum tail rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x104_splitqsum_wg256_f32` as a
  CMake-built opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Llama 3.1 8B Q8_0 p33, p512, and p513 prompt rows, plus p513/fa1/r3
  same-binary model A/B.
- route or candidate:
  packed-Q8_1/x4 BM64 BN104 split-qsum probe. The simple BN104 route had
  spilled; BN112/BN128 split-qsum showed the qsum live range was the spill
  wall, so this retested BN104 as a p513 tail bracket with smaller covered
  column overrun than BN112 or broad BN128.
- baseline command:
  focused `test-backend-ops perf -b HRX0 --test-file <q8 rows> --output csv`
  with current Q8_0 routing, plus `llama-bench -p 513 -n 0 -fa 1 -r 3 -o json
  --no-warmup -ngl 99 -dev HRX0`.
- variant command:
  same focused and model commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X104_SPLITQSUM_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq64x104-splitqsum-focused-20260618/` and
  `cache/hrxv1/gfx1151/q8_0-mmq64x104-splitqsum-p513-model-ab-20260618/`.
- profile/timing:
  same artifacts; HSACO at
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x104_splitqsum.hsaco`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p33
  stayed on BN64 while p512 and p513 selected BN104 split-qsum.
- timing:
  compile evidence: wave32, SGPR `28`, VGPR `127`, LDS `3536`, no spills.
  Focused p512 rejected broad promotion: result_output
  `55810.40 -> 61559.98 us`. Focused p513 was mixed and mildly positive:
  result_output `63491.00 -> 62140.38 us`, while ffn_out regressed
  `7613.79 -> 7782.58 us`. Model p513/fa1/r3 improved only
  `427.339702 -> 428.405933 tok/s` (`+0.25%`).
- decision:
  reject production promotion. Keep the route opt-in as evidence only.
- notes:
  Split-qsum fixed the BN104 register-spill failure, but the same-runner tail
  gain was too small to add selector complexity, and p512 is a clear regression
  if selected broadly. Keep current p512 BN128 full-column and p513 BN112
  policies while moving the Q8 work back toward the remaining Vulkan schedule
  deltas instead of narrower aggregate nudges.

## 2026-06-18 - Vulkan-oracle versus HIP direct-WMMA delta refresh

- source:
  `sources/llama.cpp` at `bf5f005fc`, after teaching
  `tools/vulkan-oracle/compare_amdgcn_isa.py` to emit normalized resource and
  opcode deltas.
- build:
  reused `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  no model run; this was a static schedule comparison for the p512 large
  aligned Vulkan oracle family.
- route or candidate:
  closest existing direct-WMMA probes for Q5_K, Q6_K, and Q8_0 versus their
  matching RADV large aligned pipelines.
- baseline command:
  `tools/vulkan-oracle/compare_amdgcn_isa.py --radv-isa <oracle isa>
  --radv-stats <oracle stats> --hsaco <candidate hsaco> --hsaco-symbol
  <candidate symbol>`.
- variant command:
  same command after the tool change, writing JSON and Markdown reports with
  `delta.resources` and `delta.interesting_opcodes`.
- route trace:
  not applicable.
- profile/timing:
  `cache/hrxv1/gfx1151/oracle-hip-comparison-refresh-20260618/`.
- correctness:
  not applicable; this consumed already built HSACO and already captured RADV
  oracle ISA.
- timing:
  static delta only. Q6/Q8 closest B64GROUP+STORE_STAGE probes match the
  oracle on `32` WMMA sites and `22528` LDS bytes, but still have `34`
  barriers versus RADV `2`, `64` versus `128` `ds_load_u16_d16`, `66` versus
  `128` `ds_store_b16`, `64` scalar global stores versus RADV `192`
  `buffer_store_b32`, and VGPR `196/195` versus RADV `192`. Q5 fullstore
  still misses the `22528` LDS footprint, all `ds_load_u16_d16`, almost all
  LDS b16 stores, and all RADV buffer stores.
- decision:
  do not spend the next work item on another isolated tile or store-count
  perturbation. The next direct-WMMA route must attack cooperative-matrix
  load/store lane ownership and the 2-barrier loop together, or switch to a
  lower-level source form capable of producing that RADV-like lowering.
- notes:
  This aligns with the user's direction: until HRX is much closer to Vulkan
  parity, route work should zero in on exact winning schedules and fail at the
  kernel/schedule comparison layer before model-level A/B.

## 2026-06-18 - large-coopmat ISA contract gate

- source:
  `sources/llama.cpp` dirty after adding
  `tools/vulkan-oracle/check_isa_contract.py`.
- build:
  reused `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  no model run; this was an ISA/schedule gate for the p512 large aligned
  Vulkan oracle family.
- route or candidate:
  closest existing direct-WMMA probes for Q5_K, Q6_K, and Q8_0 checked against
  an explicit large-coopmat contract.
- baseline command:
  existing `tools/vulkan-oracle/compare_amdgcn_isa.py` JSON reports under
  `cache/hrxv1/gfx1151/oracle-hip-comparison-refresh-20260618/`.
- variant command:
  `tools/vulkan-oracle/check_isa_contract.py --compare-json <compare.json>
  --match-resource lds_bytes --match-opcode v_wmma_f16_16x16x16_f16
  --match-opcode buffer_store_b32 --match-opcode ds_load_u16_d16
  --match-opcode ds_store_b16 --rhs-opcode-max s_barrier=2
  --rhs-resource-max vgpr=192 --require-zero-spills`.
- route trace:
  not applicable.
- profile/timing:
  contract outputs:
  `cache/hrxv1/gfx1151/oracle-hip-comparison-refresh-20260618/q5_k-large-coopmat-contract.json`,
  `cache/hrxv1/gfx1151/oracle-hip-comparison-refresh-20260618/q6_k-large-coopmat-contract.json`,
  and
  `cache/hrxv1/gfx1151/oracle-hip-comparison-refresh-20260618/q8_0-large-coopmat-contract.json`.
- correctness:
  not applicable; this is a static schedule gate over already built and
  previously correctness-tested candidate HSACOs.
- timing:
  not run. The gate intentionally fails the current closest candidates:
  Q6/Q8 match `32` WMMA and `22528` LDS bytes, but fail buffer-store,
  halfword-LDS traffic, two-barrier, and VGPR contract checks. Q5 fullstore
  passes the two-barrier and VGPR checks but fails LDS bytes, buffer stores,
  and halfword-LDS checks.
- decision:
  use this contract before spending model time on any future large-aligned
  direct-WMMA claim. A candidate that cannot pass these checks is not the exact
  Vulkan schedule and should be treated as a diagnostic, not parity work.
- notes:
  The check is intentionally stricter than a performance heuristic. It encodes
  the current user direction: zero in mechanically on the winning RADV schedule
  until HRX is much closer to Vulkan.

## 2026-06-18 - Q8/Q6 direct-WMMA contract matrices

- source:
  `sources/llama.cpp` at `c2b06c150`, using the new ISA contract checker.
- build:
  reused `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  no model run; this was a static schedule sweep over existing CMake-built
  direct-WMMA HSACOs for p512 large aligned Q8_0 and Q6_K oracle rows.
- route or candidate:
  all existing `mul_mat_vec_q8_0_wmma16_vk128*.hsaco` and
  `mul_mat_vec_q6_k_wmma16_vk128*.hsaco` variants.
- baseline command:
  Vulkan oracle RADV ISA/stats for
  `matmul_q8_0_f32_f16acc_aligned_l` and
  `matmul_q6_k_f32_f16acc_aligned_l`.
- variant command:
  for each HSACO, run `compare_amdgcn_isa.py`, then
  `check_isa_contract.py` with the large-coopmat contract.
- route trace:
  not applicable.
- profile/timing:
  Q8_0 matrix:
  `cache/hrxv1/gfx1151/q8_0-large-coopmat-contract-matrix-20260618/summary.md`.
  Q6_K matrix:
  `cache/hrxv1/gfx1151/q6_k-large-coopmat-contract-matrix-20260618/summary.md`.
- correctness:
  not applicable for this static sweep; it ranks already built diagnostic
  candidates by emitted schedule facts.
- timing:
  not run. Q8_0 closest groups:
  `store_stage` gets `22528` LDS and `64/66` halfword LDS load/store counts
  but has `34` barriers and only `64` global stores; low-barrier/fullstore
  variants have `2` barriers and up to `128` global stores but miss the
  `22528` LDS footprint and all RADV halfword-LDS traffic. Q6_K shows the same
  split.
- decision:
  the next source candidate should not be another isolated wrapper around the
  existing direct-WMMA helpers. It must either remove the per-tile
  `STORE_STAGE` barriers while preserving the 22528-byte LDS and halfword LDS
  traffic, or use a different/lower-level source form that can emit RADV-like
  cooperative matrix stores (`192` `buffer_store_b32`) directly.
- notes:
  The matrix gives a concrete start point for the next kernel edit:
  fullstore/low-barrier is the better base if attacking writeback directly;
  store-stage is the better base only if attacking barrier hoisting/removal.

## 2026-06-18 - Q8_0 VK128 stagealloc fullstore rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_stagealloc_fullstore_f16acc_wg256_f32`
  as a CMake-built opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  focused Llama 3.1 8B Q8_0 p33, p512, and p513 prompt rows.
- route or candidate:
  direct-F32 VK128 wave64 WMMA route preserving the low-barrier fullstore path
  while allocating/touching the missing 2048-byte coopmat-stage LDS region.
  This isolated the LDS-footprint axis from the earlier store-stage probes
  that matched `22528` LDS only by adding `34` barriers.
- baseline command:
  focused `test-backend-ops perf -b HRX0 --test-file <q8 rows> --output csv`
  with current Q8_0 routing.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_STAGEALLOC_FULLSTORE_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-stagealloc-fullstore-focused-20260618/`.
- profile/timing:
  same focused artifact; compile and contract artifact
  `cache/hrxv1/gfx1151/q8_0-stagealloc-fullstore-compile-20260618/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces prove p33
  stayed on BN64 while p512 and p513 selected the candidate for the three
  large rows and kept the current packed-Q8_1 route for the two narrow rows.
- timing:
  compile evidence: wave64, SGPR `28`, VGPR `145`, LDS `22528`, no spills,
  `32` WMMA sites, `128` global stores, `2` barriers. The large-coopmat
  contract improved versus fullstore by matching RADV LDS bytes while
  preserving the two-barrier and VGPR gates, but still failed `buffer_store_b32`,
  `ds_load_u16_d16`, and `ds_store_b16`. Focused p512/p513 timing rejected the
  route: p512 ffn_out `7062.83 -> 10018.10 us`, ffn_gate
  `6303.72 -> 10697.66 us`, result_output `55853.17 -> 95195.40 us`; p513
  ffn_out `7749.84 -> 10655.57 us`, ffn_gate `6970.77 -> 11635.07 us`,
  result_output `63169.17 -> 107084.50 us`.
- decision:
  reject production promotion. Keep the route opt-in as evidence only.
- notes:
  Matching RADV's LDS footprint without RADV's halfword LDS traffic and
  cooperative writeback/lane ownership is actively worse. The next direct-WMMA
  attempt should not spend another loop on LDS footprint or store count alone.

## 2026-06-18 - Q8_0 VK128 buffer-store compile-contract rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_bufferstore_f16acc_wg256_f32`
  as a CMake-built opt-in HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; kernel compiled through CMake/Ninja.
- model/shape:
  no model run. This was intentionally stopped at the static Vulkan schedule
  contract before spending GPU time on a non-matching source form.
- route or candidate:
  direct-F32 VK128 wave64 WMMA route using the stage-allocation LDS footprint
  and raw AMDGPU buffer stores for both full-tile and guarded edge/tail
  writeback. The raw-store spelling was validated first in scratch with
  `__builtin_amdgcn_raw_buffer_store_b32` and an opaque
  `__amdgpu_buffer_rsrc_t` descriptor.
- baseline command:
  Vulkan oracle RADV ISA/stats for
  `matmul_q8_0_f32_f16acc_aligned_l` from
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/`.
- variant command:
  `tools/vulkan-oracle/compare_amdgcn_isa.py` and
  `tools/vulkan-oracle/check_isa_contract.py` against
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_bufferstore_wg256.hsaco`.
- route trace:
  not applicable.
- profile/timing:
  compile artifact
  `cache/hrxv1/gfx1151/q8_0-bufferstore-compile-20260618/`.
- correctness:
  not run because the candidate failed the agreed compile-contract screen.
- timing:
  not run. The candidate matched RADV on `32` f16 WMMA, `22528` LDS bytes,
  `2` barriers, and zero spills, and proved all static stores can be emitted
  as `buffer_store_b32` from HIP C++. It still failed the large-coopmat
  contract: RADV has `192` `buffer_store_b32`, `128` `ds_load_u16_d16`, and
  `128` `ds_store_b16`; the candidate has `128`, `0`, and `2` respectively.
- decision:
  reject before runtime. Keep the opt-in route as diagnostic evidence only.
- notes:
  This is the current strongest signal that the remaining direct-WMMA gap is
  not "use buffer stores" in isolation. The next serious attempt should
  mechanically reproduce RADV's cooperative halfword LDS load/store topology
  and 192-store lane ownership, likely with a lower-level source spelling or a
  hand-controlled codegen path, before any model A/B.

## 2026-06-18 - Q8_0 VK128 buffer-store runtime-correctness rejection

- source:
  `sources/llama.cpp` updated the buffer-store route to create its destination
  descriptor with `__builtin_amdgcn_make_buffer_rsrc` instead of an ad hoc
  four-word reinterpret cast.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-ops -j$(nproc)`
  with ROCm `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  focused Llama 3.1 8B Q8_0 p512 exported rows.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_bufferstore_f16acc_wg256_f32`
  under
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-bufferstore-make-rsrc-focused-20260618/`.
- route trace:
  `Vcur-0` and `Qcur-0` stayed on
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32`.
  The buffer-store route selected for `ffn_out-0`
  (`k=14336, rows=4096`), `ffn_gate-0` (`k=4096, rows=14336`), and
  `result_output` (`k=4096, rows=128256`).
- correctness:
  `Vcur-0` and `Qcur-0` passed because they did not select the candidate.
  Every large selected row failed CPU reference with `ERR=inf`.
- compile evidence:
  the proper builtin descriptor preserved the exact raw-store axis and improved
  pressure versus the ad hoc descriptor: wave64, SGPR `28`, VGPR `145`, LDS
  `22528`, no spills, `32` f16 WMMA sites, and no `global_store_b32`.
- decision:
  reject production promotion and do not run model timing. Static RADV whole
  shader counts overstate the p512 full-aligned writeback path because they
  include fallback paths, so direct buffer-store writeback remains a valid
  exact-schedule axis. This particular HIP C++ route is not correct yet; next
  work should prove raw descriptor/store semantics in a controlled fixture or
  switch to a lower-level spelling that directly reproduces RADV's cooperative
  halfword LDS load/store topology.

## 2026-06-18 - Q8_0 VK128 buffer-store gfx11 descriptor fix and perf rejection

- source:
  `sources/llama.cpp` adds a CMake/Ninja-built
  `hrx-hip-bench-raw-buffer-store` fixture and patches the Q8_0 VK128
  buffer-store route to use the gfx11 raw buffer resource word `0x31004000`
  instead of the inherited `0x27000`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-ops hrx-hip-bench-raw-buffer-store -j$(nproc)`
  with ROCm `/srv/vm-shared/rocm/rocm-head`.
- descriptor evidence:
  `cache/hrxv1/gfx1151/raw-buffer-store-fixture-20260618/`.
  The old `0x27000` descriptor failed the fixture for every tested
  make/manual and max/byte extent variant. The Tensile gfx11 value
  `0x31004000` passed `make-max`, `make-bytes`, `manual-max`, and
  `manual-bytes` on `4096 x 64`, then passed production-shaped `make-max`
  stores for `4096 x 512`, `14336 x 512`, and `128256 x 512`.
- focused artifact:
  `cache/hrxv1/gfx1151/q8_0-bufferstore-gfx11-rsrc-focused-20260618/`.
- route/correctness:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_BUFFERSTORE_F16ACC_WG256_PROMPT=1`
  selected the buffer-store route for p512 and p513 large rows
  (`ffn_out`, `ffn_gate`, `result_output`) and all CPU-reference gates passed.
  p33 stayed on the existing narrow packed route and passed.
- timing:
  same-runner focused `test-backend-ops perf` rejects the route despite fixed
  correctness. p512 regressed `ffn_out 7048.33 -> 10357.40 us`,
  `ffn_gate 6358.83 -> 10973.75 us`, and
  `result_output 56008.26 -> 98015.71 us`. p513 regressed
  `ffn_out 7611.67 -> 10779.03 us`,
  `ffn_gate 6875.18 -> 13367.04 us`, and
  `result_output 62863.76 -> 111411.43 us`.
- decision:
  keep the gfx11 descriptor fix and fixture, but keep the route opt-in and
  rejected for production. This proves raw buffer stores are a necessary
  primitive, not the Vulkan-winning schedule. The next Q8_0 parity attempt
  should mechanically target RADV's cooperative-matrix load/store lane
  ownership and halfword LDS topology rather than another high-level timing
  sweep.

## 2026-06-18 - Q8_0 VK128 b64group-bufferstore exact-schedule probe rejection

- source:
  `sources/llama.cpp` adds
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_bufferstore_f16acc_wg256_f32`
  as an opt-in CMake/Ninja-built route.
- gate:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- rationale:
  this is the mechanical combination left after the gfx11 raw-buffer descriptor
  fix: grouped `ds_read_b64` fragment loads, 22528-byte stage allocation,
  full-tile direct writeback, and raw `buffer_store_b32` output.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-ops -j$(nproc)`.
- ISA artifact:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-b64group-bufferstore-isa-20260618/`.
  The route matches RADV on `32` f16 WMMA, `64` `ds_load_b64`, `22528` LDS
  bytes, wave64, two barriers, and zero spills. It still misses
  `ds_load_u16_d16` (`128 -> 0`), `ds_store_b16` (`128 -> 2`), and the
  full static buffer-store shape (`192 -> 128`).
- focused artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-b64group-bufferstore-focused-20260618/`.
- correctness:
  p512 and p513 selected the candidate on `ffn_out`, `ffn_gate`, and
  `result_output`; p33 stayed on existing narrow packed routes. All focused
  CPU-reference gates passed.
- timing:
  same-runner focused perf rejects the candidate. p512 regressed
  `ffn_out 7093.01 -> 15624.94 us`, `ffn_gate 6323.39 -> 12512.46 us`, and
  `result_output 56201.69 -> 107212.45 us`. p513 regressed
  `ffn_out 7750.36 -> 15292.70 us`, `ffn_gate 6951.29 -> 14250.05 us`, and
  `result_output 63109.00 -> 124419.50 us`.
- decision:
  reject for production. This closes the source-visible combination of the
  b64-load and raw-buffer-store axes. The remaining Q8_0 large-route work needs
  a lower-level way to express RADV's cooperative halfword LDS load/store and
  lane ownership, or a different prior family, before another route should be
  expected to move toward Vulkan parity.

## 2026-06-18 - Q8_0 LDS halfword-stage primitive fixture

- source:
  `sources/llama.cpp` adds the CMake/Ninja-built
  `hrx-hip-bench-lds-halfword-stage` wave64 fixture.
- purpose:
  isolate the remaining RADV fallback-store primitive before putting another
  candidate in the Q8_0 catalog: explicit `ds_store_b16` plus
  `ds_load_u16_d16` for cooperative halfword LDS staging.
- artifact:
  `cache/hrxv1/gfx1151/lds-halfword-stage-fixture-20260618/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target hrx-hip-bench-lds-halfword-stage -j$(nproc)`.
- correctness:
  typed LDS mode passed `tiles=1`; explicit asm mode passed `tiles=1`,
  `tiles=16`, and `tiles=17`.
- ISA evidence:
  final fixture disassembly contains `4 ds_store_b16` and
  `4 ds_load_u16_d16` in the asm kernel. The typed C++ path instead emits flat
  halfword LDS operations and barriers, so it is not a substitute for the RADV
  opcode contract.
- implementation note:
  `ds_load_u16_d16` requires an explicit `s_waitcnt lgkmcnt(0)` before using
  the loaded value from HIP inline asm. Without that wait, the fixture stores
  stale low-half values, which explains why this primitive should stay isolated
  until the production writeback helper has its lane/value contract proven.
- decision:
  accept the fixture as a primitive proof, not a production route. The next
  Q8_0 candidate may use an explicit halfword-stage helper for odd/tail
  writeback, but it should first be compared against RADV's full store-path
  control flow and then gated on p33, p512, and p513 focused rows.

## 2026-06-18 - Q5_K rocWMMA cooperative accumulator store probe rejection

- artifact:
  `cache/hrxv1/gfx1151/rocwmma-coopstore-probe-20260618/`.
- purpose:
  determine whether HIP C++/rocWMMA can express the RADV
  `OpCooperativeMatrixStoreKHR` accumulator writeback used by the Vulkan
  direct-F32 Q5_K oracle path.
- compile matrix:
  default `fragment<accumulator, 16,16,16,float,row_major>` store compiled;
  `coop_row_major_2d<2,2>` and `coop_col_major_2d<2,2>` accumulator stores
  failed template instantiation; `single<2,2,0>` compiled but only gives
  one-wave ownership.
- header evidence:
  rocWMMA public stores are implemented through `OpaqueStore` vector/scalar
  memory stores from fragment access, not a cooperative-matrix store intrinsic.
  The cooperative accumulator forms hit invalid `SplitK=0` diagnostics in the
  current headers.
- ISA evidence:
  the default store compiled wave64 with SGPR `4`, VGPR `4`, LDS `0`, no
  spills, and `8 global_store_b32`. The `single` variant compiled wave64 with
  SGPR `4`, VGPR `5`, LDS `0`, no spills, and another scalar
  `global_store_b32` store sequence. Neither emitted `buffer_store_b32`,
  `ds_store_b16`, `ds_load_u16_d16`, or a RADV-like cooperative store path.
- decision:
  reject rocWMMA public accumulator stores as the next Q5 direct-F32 route
  source. The direct-F32 clone needs a lower-level compiler/IR path for
  cooperative-matrix store semantics, or we should pivot back to Q5-specific
  packed-Q8_1 schedule work with p33/p512/p513 gates.

## 2026-06-18 - Q5_K MMQL128 CR issue-order rejection

- route:
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_cr_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_CR_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q5-mmql128-cr-focused-20260618/`.
- purpose:
  test a Q5-specific packed-Q8_1/x4 issue-order axis after B-quad and B-pair
  showed that the Q4_K B-cache clustering win does not transfer directly.
  The probe preserved BM128/BN128/WG256/wave64/BK_STEP1 and changed the inner
  dot loop to consume one Q5 row cache in `cr`-major order across columns.
- compile evidence:
  built through CMake/Ninja, but with wave64, SGPR `50`, VGPR `192`,
  private segment `452`, and VGPR spills `135`. It preserved
  `512 v_dot4_i32_iu8`, `64 global_store_b32`, and `2` barriers, but raised
  wait count pressure to `230 s_waitcnt`.
- correctness and route evidence:
  p33, p512, and p513 CPU-reference gates passed. p512 selected CR on Qcur,
  ffn_out, and ffn_gate; p33 stayed on existing narrow routes; p513 tails
  fell back to current MMQL128.
- timing:
  same-runner p512 focused timing rejects the candidate. Qcur regressed
  `1225.57 -> 3832.51 us`, ffn_out regressed
  `7453.69 -> 21017.01 us`, and ffn_gate regressed
  `7161.45 -> 19333.78 us`.
- decision:
  reject for production. CR-major issue order increases live ranges and spills
  instead of closing on the Vulkan schedule. Do not continue this axis without
  a different B ownership or prefetch strategy that removes the spill wall.

## 2026-06-18 - Q5_K VK128 B64GROUP store-batch8 rejection

- route:
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_store_batch8_f16acc_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_STORE_BATCH8_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-store-batch8-20260618/`.
- purpose:
  test the closest source-visible direct-F32 Q5_K clone after B64GROUP
  store-stage: keep the grouped LDS fragment loads and f16 WMMA math, but
  batch eight accumulator tiles per wave through LDS before the writeback
  barrier. This specifically targeted the `34` barrier count in the prior
  store-stage probe versus RADV's `2` barriers.
- compile evidence:
  built through CMake/Ninja. The route emitted wave64, SGPR `32`, VGPR `214`,
  LDS `36864`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `64 ds_load_u16_d16`, `66 ds_store_b16`, `64 global_store_b32`, and
  `6` barriers.
- correctness and route evidence:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on rows2/cols8,
  while p512 and p513 selected store-batch8 for all focused rows.
- timing:
  same-runner focused timing rejected the route. p512 regressed
  `Kcur 864.14 -> 1043.49 us`, `Qcur 1223.22 -> 2499.40 us`,
  `ffn_out 7229.01 -> 17649.52 us`, and
  `ffn_gate 6903.14 -> 16915.11 us`. p513 regressed
  `Kcur 886.35 -> 960.72 us`, `Qcur 1585.01 -> 3389.07 us`,
  `ffn_out 9601.19 -> 23409.53 us`, and
  `ffn_gate 8720.37 -> 20101.28 us`.
- decision:
  reject for production. Lowering explicit barriers from `34` to `6` does not
  recover the Vulkan schedule while the route still lacks RADV's
  cooperative-matrix store/lane ownership, `192 buffer_store_b32` writeback,
  and full `128/128` halfword LDS load/store topology.

## 2026-06-18 - Q5_K VK128 B64GROUP bufferstore rejection

- route:
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_bufferstore_f16acc_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_PADDED_W64_B64GROUP_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q5-wmma-vk128-b64group-bufferstore-20260618/`.
- purpose:
  port the fixed gfx11 raw-buffer-store primitive from Q8_0 to Q5_K and pair
  it with grouped LDS fragment loads. This directly tested the RADV
  `buffer_store_b32` writeback opcode class without adding manual LDS
  store-stage barriers.
- compile evidence:
  built through CMake/Ninja. The route emitted wave64, SGPR `32`, VGPR `198`,
  LDS `20480`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `0 ds_load_u16_d16`, `2 ds_store_b16`, `128 buffer_store_b32`,
  `0 global_store_b32`, and `2` barriers.
- correctness and route evidence:
  p33, p512, and p513 CPU-reference gates passed. p33 stayed on rows2/cols8,
  while p512 and p513 selected bufferstore for all focused rows.
- timing:
  same-runner focused timing rejected the route. p512 regressed
  `Kcur 871.11 -> 1038.82 us`, `Qcur 1232.52 -> 2388.60 us`,
  `ffn_out 7322.86 -> 17957.57 us`, and
  `ffn_gate 7126.12 -> 16776.44 us`. p513 regressed
  `Kcur 897.54 -> 959.48 us`, `Qcur 1590.42 -> 3315.05 us`,
  `ffn_out 9404.53 -> 23052.67 us`, and
  `ffn_gate 8409.39 -> 20001.36 us`.
- decision:
  reject for production. The result matches the Q8_0 bufferstore conclusion:
  raw buffer stores are a working primitive, but not the Vulkan-winning
  cooperative-matrix store/lane-ownership schedule.

## 2026-06-18 - Q8_0 Vulkan0 CPU-reference contract check

- source:
  `sources/llama.cpp` dirty after tightening `test-backend-ops -b` handling so
  a mistyped backend filter fails instead of silently skipping every device.
- purpose:
  decide whether the finite errors from the Q8_0 stream-row/stream-column HIP
  probes could be accepted as Vulkan-equivalent numerical behavior.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-vulkan0-focused-cpuref-20260618-140129/`.
- invalidated artifact:
  `cache/hrxv1/gfx1151/q8_0-vulkan-focused-cpuref-20260618-135708/` used
  `-b Vulkan`, which skipped `Vulkan0` and produced zero op rows. The rebuilt
  runner now exits `1` with
  `backend filter 'Vulkan' did not match any backend device`.
- command:
  `build/vulkan-gfx1151/bin/test-backend-ops test -b Vulkan0 -o MUL_MAT --test-file <q8_0_prompt_all.txt> --output csv`.
- correctness:
  Vulkan0 passed all focused Q8_0 rows for p33, p512, and p513: five rows per
  prompt size, zero failures, backend name `Vulkan0` in every CSV row.
- decision:
  the stream-row/stream-column HIP probes are not acceptable under a relaxed
  Vulkan-equivalent tolerance. Vulkan itself passes the strict CPU-reference
  gate on these exported rows, so Q8_0 parity work must preserve correctness
  while mechanically matching the Vulkan/RADV schedule.

## 2026-06-18 - Q8_0 VK128 B64GROUP dualstage bufferstore compile-contract rejection

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_dualstage_bufferstore_f16acc_wg256_f32`.
- gate:
  none; intentionally not wired into runtime selection.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-dualstage-bufferstore-compile-20260618/`.
- purpose:
  continue the exact Vulkan-oracle contract path after B64GROUP+bufferstore
  still missed RADV's halfword LDS topology. This candidate preserves grouped
  `ds_load_b64` fragment reads, raw `buffer_store_b32` writeback, and
  22528-byte LDS allocation, then uses inline `ds_write_b16` and
  `ds_read_u16_d16` to force a duplicated halfword LDS stage without changing
  selected output semantics.
- compile evidence:
  built through CMake/Ninja. Final HSACO emitted wave64, SGPR `28`, VGPR
  `195`, LDS `22528`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `130 ds_store_b16`, `64 buffer_store_b32`, and
  `34` barriers.
- correctness and route evidence:
  not run. The candidate failed the compile-contract screen and is not wired
  into provider selection.
- decision:
  reject before runtime. Inline DS proves HIP C++ can emit the halfword LDS
  read/store opcode class, but the exact RADV large-coopmat contract still
  fails on `buffer_store_b32` (`64` vs `192`), barriers (`34` vs `2`), and
  VGPR (`195` vs `192`). The next direct-WMMA attempt must solve low-barrier
  cooperative store/lane ownership, not add more manual LDS staging.

## 2026-06-18 - Q8_0 coopmat store contract fixture

- fixture:
  `hrx-hip-bench-coopmat-store-contract`.
- artifact:
  `cache/hrxv1/gfx1151/coopmat-store-contract-fixture-20260618-122510/`.
- purpose:
  isolate the RADV large-aligned Q8_0 store-side contract without the full
  WMMA/LDS body. The prior direct-WMMA routes kept missing RADV's
  `192 buffer_store_b32` writeback surface, so this tested whether the raw
  store surface itself is expressible from CMake-built HIP C++.
- correctness:
  `linear64`, `linear128`, `linear192`, and `branch192` all passed exact value
  checks with the gfx11 raw buffer descriptor `0x31004000`.
- compile evidence:
  built through CMake/Ninja as a wave64 HIP bench. Disassembly emitted:
  `linear64 = 64 buffer_store_b32`, `linear128 = 128`,
  `linear192 = 192`, and `branch192 = 192`. None of the fixture kernels used
  barriers. `branch192` emitted `49 s_waitcnt` and `48 s_cbranch_execz`, while
  `linear192` emitted only `2 s_waitcnt`.
- decision:
  accept as a contract-narrowing primitive. Source-visible HIP can emit the
  raw 192-store surface, so the remaining direct-WMMA gap is not "HIP cannot
  produce enough buffer stores." The missing piece is how WMMA accumulators are
  owned and mapped into those store groups while preserving RADV's two-barrier
  halfword LDS topology. Do not spend another route merely increasing raw
  store count.

## 2026-06-18 - Q8_0 VK128 B64GROUP fullpair bufferstore compile-contract rejection

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_fullpair_bufferstore_f16acc_wg256_f32`.
- gate:
  none; intentionally not wired into runtime selection.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-fullpair-bufferstore-compile-20260618-122909/`.
- purpose:
  test whether the real direct-WMMA Q8_0 kernel can reach the RADV
  `192 buffer_store_b32` writeback surface after the standalone store fixture
  proved the raw store count is expressible from HIP C++.
- compile evidence:
  built through CMake/Ninja. Final HSACO emitted wave64, SGPR `28`, VGPR
  `195`, LDS `22528`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `192 buffer_store_b32`, and `2` barriers.
- strict contract failures:
  `ds_load_u16_d16` remains `0` versus RADV `128`, `ds_store_b16` remains `2`
  versus RADV `128`, and VGPR is `195` versus the RADV target `192`.
- correctness and route evidence:
  not run. The widened full-tile store writes both accumulator halves and is a
  static schedule probe, not a correctness candidate.
- decision:
  reject before runtime. This is useful progress: the real Q8 direct-WMMA
  kernel can now match RADV's `32` WMMA, `64 ds_load_b64`,
  `192 buffer_store_b32`, 22528-byte LDS, two-barrier surface. The remaining
  exact-schedule miss is isolated to RADV's halfword LDS cooperative
  store/load topology and precise accumulator-lane mapping, not raw store
  count.

## 2026-06-18 - LDS halfword bulk128 WG256 primitive fixture

- fixture:
  `hrx-hip-bench-lds-halfword-stage --mode=bulk128-wg256`.
- artifact:
  `cache/hrxv1/gfx1151/lds-halfword-stage-bulk128-wg256-20260618-123403/`.
- purpose:
  isolate the remaining Q8_0 Vulkan large-route halfword LDS topology after
  the fullpair-bufferstore direct-WMMA probe matched the `192 buffer_store_b32`
  writeback surface.
- correctness:
  `bulk128` and `bulk128-wg256` both passed exact deterministic value checks.
- compile evidence:
  the 256-thread `bulk128-wg256` fixture emitted `128 ds_store_b16`,
  `128 ds_load_u16_d16`, and `2 s_barrier`. The strict halfword contract check
  against the RADV Q8_0 large oracle passed for these opcodes/barriers.
- decision:
  accept as a primitive proof. Source-visible HIP can emit the RADV-scale
  halfword LDS read/store topology with a 256-thread workgroup and two
  barriers. The next production-facing Q8_0 compile-contract probe should
  combine this low-barrier halfword topology with the already proven real-kernel
  low-barrier `192 buffer_store_b32` writeback, without returning to the old
  `34`-barrier dualstage path.

## 2026-06-18 - Q8_0 VK128 fast-half fullpair bufferstore compile-contract near miss

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_fast_half_fullpair_bufferstore_f16acc_wg256_f32`.
- gate:
  none; intentionally not wired into runtime selection.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-fast-half-fullpair-bufferstore-compile-20260618-123820/`.
- purpose:
  combine the two isolated source-visible primitives that matched the RADV
  Q8_0 large oracle: real-kernel `192 buffer_store_b32` writeback from the
  fullpair probe, plus low-barrier `128 ds_store_b16` and
  `128 ds_load_u16_d16` halfword LDS topology from the bulk128 WG256 fixture.
- compile evidence:
  built through CMake/Ninja. The HSACO emitted wave64, SGPR `28`, VGPR `195`,
  LDS `22528`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `130 ds_store_b16`, `192 buffer_store_b32`, and
  `2 s_barrier`.
- correctness and route evidence:
  not run. This candidate is a static schedule probe only; the fullpair
  accumulator-half mapping is diagnostic and is not provider-selected.
- decision:
  reject before runtime but keep as the current closest direct-WMMA Vulkan
  oracle convergence point. It matches RADV on WMMA count, grouped LDS loads,
  halfword LDS loads, raw buffer-store count, LDS bytes, barrier count, and
  spill-free compilation. The strict contract still fails by two extra
  `ds_store_b16` sites and `VGPR=195` versus RADV's `192`. The extra stores
  appear in the pre-output staging/control region, not the final fast-half
  writeback loop, so the next attempt should reduce the live prewrite staging
  footprint or split the mapping without changing the now-matched final-store
  surface.

## 2026-06-18 - Q8_0 VK128 pack-stage fast-half bufferstore VGPR rejection

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_fullpair_bufferstore_f16acc_wg256_f32`.
- gate:
  none; intentionally not wired into runtime selection.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-packstage-fast-half-fullpair-bufferstore-compile-20260618-124840/`.
- purpose:
  remove the two extra pre-output `ds_store_b16` sites in the prior fast-half
  probe by changing only A/B LDS staging from scalar half stores to packed
  `ds_write_b32` pair stores.
- compile evidence:
  built through CMake/Ninja. The HSACO emitted wave64, SGPR `28`, VGPR `196`,
  LDS `22528`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`, two
  A/B-stage `ds_store_b32`, and `2 s_barrier`.
- correctness and route evidence:
  not run. This is still a static schedule probe, not a correctness candidate.
- decision:
  reject before runtime due VGPR `196 > 192`, but keep as the closest opcode
  match so far. It proves source-visible HIP can make the Q8_0 direct-WMMA
  kernel match RADV's key opcode/LDS/barrier/spill contract exactly, with the
  remaining strict miss isolated to register pressure.

## 2026-06-18 - Q8_0 VK128 pack-stage pressure brackets

- routes:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_dummyhalf_fullpair_bufferstore_f16acc_wg256_f32`
  and
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_fullpair_bufferstore_lb2_f16acc_wg256_f32`.
- gate:
  none; intentionally not wired into runtime selection.
- artifacts:
  `cache/hrxv1/gfx1151/q8_0-packstage-dummyhalf-fullpair-bufferstore-compile-20260618-124840/`
  and
  `cache/hrxv1/gfx1151/q8_0-packstage-fast-half-fullpair-bufferstore-lb2-compile-20260618-124958/`.
- purpose:
  determine whether the remaining `196` VGPR count comes from consuming
  halfword reload values in writeback, or from compiler occupancy pressure
  that `__launch_bounds__(256, 2)` can influence.
- compile evidence:
  both variants preserved the same exact opcode contract as pack-stage
  fast-half: `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `192 buffer_store_b32`, `22528` LDS, two barriers, and
  no spills. Both still allocated `196` VGPR.
- decision:
  reject before runtime. The pressure is not caused by consuming the halfword
  reload values, and the launch-bounds hint does not reduce it. The next
  mechanical step needs a real live-range or accumulator ownership change while
  preserving the now-exact opcode surface.

## 2026-06-18 - Q8_0 VK128 stream-column pack-stage static contract pass

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamcol_packstage_fast_half_fullpair_bufferstore_f16acc_wg256_f32`.
- static artifact:
  `cache/hrxv1/gfx1151/q8_0-streamcol-packstage-fast-half-fullpair-bufferstore-compile-20260618-125457/`.
- runtime artifact:
  `cache/hrxv1/gfx1151/q8_0-streamcol-packstage-runtime-diagnostic-20260618-130143/`.
- purpose:
  reduce the remaining `196` VGPR pressure in the exact-opcode pack-stage
  probe by preserving four A fragments per `k_tile` but streaming one B column
  fragment at a time. This keeps the same `64 ds_load_b64` count while reducing
  live B-fragment state.
- compile evidence:
  built through CMake/Ninja. The HSACO emitted wave64, SGPR `28`, VGPR `188`,
  LDS `22528`, no spills, `32` f16 WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`, two
  A/B-stage `ds_store_b32`, and `2 s_barrier`.
- correctness and route evidence:
  after adding the missing route catalog entry and opt-in selector, p512 and
  p513 focused CPU-reference gates selected the provider for the intended large
  rows. p512 selected it for `ffn_out`, `ffn_gate`, and `result_output`; p513
  selected it for the same large/tail rows. Those selected rows failed with
  `HRX0=inf` versus finite CPU values and p512 sentinel mismatches, while
  non-selected q8_1/default rows still passed.
- decision:
  reject for runtime promotion. This proves CMake-built HIP C++ can hit the
  selected RADV Q8_0 large-oracle opcode, LDS, barrier, VGPR, and spill
  contract, but the fullpair accumulator-half writeback corrupts the selected
  output rows. The next mechanical step is not tile-shape tuning; it is to keep
  this stream-column pack-stage schedule and replace the fullpair writeback
  with a correctness-proven accumulator lane/output coordinate map.

## 2026-06-18 - Q8_0 VK128 pack-stage selected-writeback correctness control

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_bufferstore_f16acc_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- correctness artifact:
  `cache/hrxv1/gfx1151/q8_0-packstage-bufferstore-route-20260618-132009/`.
- ISA artifact:
  `cache/hrxv1/gfx1151/q8_0-packstage-bufferstore-isa-20260618-132038/`.
- purpose:
  wire the correctness-clean isolation route into the catalog as an opt-in
  diagnostic. It preserves B64GROUP direct-WMMA math, packed A/B LDS staging,
  and raw buffer stores, but deliberately uses the known-correct selected-half
  writeback rather than the stream-column issue order or full-pair halfword
  output stage.
- correctness and route evidence:
  focused p512 and odd p513 `MUL_MAT` CPU-reference gates passed. Route traces
  selected this provider on the intended large Q8_0 rows: p512 selected
  `ffn_out`, `ffn_gate`, and `result_output`; p513 selected those same large
  tail rows. Smaller/non-target rows remained on existing q8_1/default routes.
- ISA evidence:
  built through CMake/Ninja. The HSACO emits wave64, `32` f16 WMMA,
  `64 ds_load_b64`, `2 ds_store_b32`, `128 buffer_store_b32`, `2 s_barrier`,
  no spills, LDS `20480`, and VGPR `196`. Compared with RADV it is not
  Vulkan-exact: it lacks the halfword output stage
  (`128 ds_store_b16`, `128 ds_load_u16_d16`) and has `128` rather than
  `192 buffer_store_b32`.
- decision:
  keep as an opt-in diagnostic correctness control, not as a promotion route.
  This proves packed A/B LDS staging is not the correctness bug. The isolated
  failures are now mechanical: stream-column issue/order changes strict
  numerical output, while full-pair/halfword writeback corrupts or NaNs the
  selected rows. The next Vulkan-parity step is to preserve the static
  stream-column contract and repair accumulator lane/output ownership, not to
  return to aggregate throughput tuning.

## 2026-06-18 - Q8_0 VK128 stream-column OPSEL bracket

- temporary source:
  rebuilt the committed pack-stage selected-writeback wrapper as
  `STREAM_COL=1` and `W64_OPSEL=1`, then restored the source afterward.
- gate:
  reused
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`
  during the temporary rebuild.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-streamcol-opsel1-selected-buffer-20260618-132354/`.
- purpose:
  determine whether the stream-column selected-writeback mismatch was caused
  by using the low OPSEL half or by the stream-column issue/order itself.
- result:
  focused p512 and odd p513 selected the temporary provider on the same three
  large rows. The failures stayed finite and small, around `ERR=0.0028-0.0037`,
  matching the low-OPSEL stream-column selected-writeback failure class rather
  than the full-pair corruption/NaN/inf class.
- decision:
  reject and keep out of the catalog. Both OPSEL halves are arithmetically
  valid in the non-stream route, so this bracket points at stream-column
  accumulation/order as the CPU-reference mismatch. The exact Vulkan-clone path
  needs either Vulkan-equivalent numerical acceptance evidence or a
  correctness-preserving way to lower pressure without reordering the WMMA
  accumulation sequence. The `row+16` full-pair store remains invalid because
  the RDNA3 wave64 WMMA mapping selects only one accumulator half by OPSEL.

## 2026-06-18 - Q8_0 VK128 stream-row static-contract bracket

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamrow_packstage_fast_half_fullpair_bufferstore_f16acc_wg256_f32`.
- runtime selection:
  not wired into runtime selection; compile/ISA contract probe only.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-streamrow-packstage-fast-half-fullpair-bufferstore-compile-20260618-133139/`.
- purpose:
  bracket the stream-column static-contract win without changing WMMA
  accumulation order the same way. This variant keeps the four B fragments
  resident and streams one A row fragment at a time.
- ISA evidence:
  built through CMake/Ninja and compared with the saved RADV Q8_0 large
  oracle. The HSACO emits wave64, VGPR `189`, no spills, LDS `22528`,
  `32 v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `2 ds_store_b32`, `192 buffer_store_b32`, and
  `2 s_barrier`.
- decision:
  accept as the current best static-contract schedule, but reject for runtime
  promotion. It inherits the diagnostic full-pair writeback, and the current
  `row+16` mapping is invalid for gfx11 wave64 WMMA because only the
  OPSEL-selected accumulator half owns the 16x16 tile output. The next
  candidate should combine this row-stream issue order with a correctness-proven
  selected-half store, then separately solve the full Vulkan halfword output
  map.

## 2026-06-18 - Q8_0 VK128 stream-row selected-writeback gate

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamrow_packstage_bufferstore_f16acc_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_STREAMROW_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- correctness artifact:
  `cache/hrxv1/gfx1151/q8_0-streamrow-packstage-bufferstore-q8route-20260618-133828/`.
- ISA artifact:
  `cache/hrxv1/gfx1151/q8_0-streamrow-packstage-bufferstore-isa-20260618-133905/`.
- result:
  focused p512 and p513 route traces selected this provider on the intended
  large Q8_0 rows: `ffn_out`, `ffn_gate`, and `result_output`. Those rows
  failed strict CPU-reference with finite `ERR ~= 0.0036-0.0039`. Narrow p33
  stayed on existing Q8_1 routes and passed.
- ISA evidence:
  wave64, VGPR `189`, no spills, LDS `20480`, `32` f16 WMMA,
  `64 ds_load_b64`, `2 ds_store_b32`, `128 buffer_store_b32`, and
  `2 s_barrier`.
- decision:
  reject for promotion. Row-stream reduces live pressure, but it changes
  accumulation order enough to fail the same strict correctness class as
  stream-column. The next pressure-reduction attempt must preserve the
  non-stream WMMA issue order, or we need an explicit Vulkan-equivalent
  numerical acceptance contract before using stream-order variants.

## 2026-06-18 - Q8_0 VK128 full-tile selected-writeback gate

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fulltile_bufferstore_f16acc_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_FULLTILE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-packstage-fulltile-bufferstore-q8route-20260618-134640/`.
- purpose:
  test whether dynamic edge/tail handling accounts for the remaining VGPR
  pressure in the correctness-clean non-stream pack-stage selected route.
  The selector requires exact 128-multiple rows and columns; p513 and p33 are
  intentionally guarded out.
- compile evidence:
  CMake/Ninja HSACO emits wave64, VGPR `193`, SGPR `24`, LDS `20480`, no
  spills, `32` f16 WMMA, `64 ds_load_b64`, `2 ds_store_b32`,
  `64 buffer_store_b32`, and `2 s_barrier`. This improves pressure versus the
  selected control (`196 -> 193`) but still misses the RADV `192` ceiling and
  moves the store surface farther from RADV.
- correctness and route evidence:
  p512 selected the provider on `ffn_out`, `ffn_gate`, and `result_output` and
  passed strict CPU-reference. p33 and p513 stayed on existing Q8_1 routes and
  passed.
- same-runner focused perf:
  `ffn_out` regressed `7031.762 -> 8422.038 us`, `ffn_gate` regressed
  `6433.416 -> 7332.741 us`, and `result_output` regressed
  `56130.024 -> 64645.333 us`.
- decision:
  reject for promotion. Removing full-tile guards is a valid pressure bracket,
  but it is slower and still not the exact RADV resource/store contract.

## 2026-06-18 - Q8_0 VK128 named-fragment pack-stage compile bracket

- route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_namedfrag_packstage_bufferstore_f16acc_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_NAMEDFRAG_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-namedfrag-packstage-bufferstore-compile-20260618-135058/`.
- purpose:
  preserve the strict-correct non-stream issue order but spell the four A and
  four B fragments as named variables instead of indexed arrays, testing
  whether LLVM was keeping array/index live range.
- compile evidence:
  CMake/Ninja HSACO emits the same shape as the indexed pack-stage selected
  control: wave64, VGPR `196`, SGPR `28`, LDS `20480`, no spills, `32` f16
  WMMA, `64 ds_load_b64`, `2 ds_store_b32`, `128 buffer_store_b32`, and
  `2 s_barrier`.
- decision:
  reject at compile-evidence gate. This spelling does not move toward the
  RADV `192` VGPR target and does not change the emitted schedule.

## 2026-06-18 - WMMA f16 wave64 lane-map fixture

- source:
  `sources/llama.cpp` dirty after adding the CMake/Ninja-built
  `hrx-hip-bench-wmma-f16-lane-map` fixture.
- purpose:
  prove the gfx1151 `v_wmma_f16_16x16x16_f16_w64` accumulator slot ownership
  before attempting another Q8_0 fullpair writeback route. This directly tests
  whether the rejected diagnostic `row+16` mapping has a plausible lane-map
  interpretation.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-lane-map-20260618-140647/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target hrx-hip-bench-wmma-f16-lane-map -j$(nproc)`.
- runtime result:
  the fixture initializes every accumulator slot with a unique sentinel, runs
  one all-ones WMMA, and checks which slots changed. OPSEL 0 changed exactly
  `256` even accumulator slots and `0` odd slots. OPSEL 1 changed exactly `0`
  even slots and `256` odd slots. All unchanged slots retained their sentinel
  values, and all changed slots advanced by the expected dot contribution
  `16`.
- ISA evidence:
  extracted HSACO metadata reports wave64, SGPR `12`, VGPR `13`, LDS `0`, and
  no spills for both kernels. Each OPSEL kernel emits one
  `v_wmma_f16_16x16x16_f16`; the OPSEL 1 kernel disassembles with
  `op_sel:[0,0,1]`.
- decision:
  accept as a primitive proof. The current Q8_0 fullpair `row+16` writeback is
  invalid for the HIP gfx1151 wave64 builtin: OPSEL selects even versus odd
  accumulator slots on the same 16x16 output ownership surface, not a second
  independent 16-row tile. The next exact-schedule route should preserve
  strict correctness with selected-half ownership, or use a lower-level
  cooperative-store path whose lane map is proven separately.

## 2026-06-18 - Q8_0 RADV event-window extraction

- source:
  `sources/llama.cpp/tools/vulkan-oracle/extract_coopmat_schedule.py` updated
  to extract RADV ISA event windows in addition to opcode/resource totals.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-vulkan-coopmat-schedule-extract-20260618-followup/`.
- purpose:
  make the Vulkan oracle workflow stricter. The working unit should be the
  actual schedule window and store/load lane contract, not aggregate
  throughput and not only coarse opcode counts.
- command:
  ran the extractor against
  `vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453` with the current
  pack-stage selected HIP comparison JSON.
- result:
  the generated `schedule.md` records pre-WMMA `ds_load_b64` offset families,
  store basic blocks, and the first WMMA window. The RADV large route contains
  separate direct full-aligned store blocks and staged fallback blocks; this is
  the missing lane/writeback contract that the HIP selected-half probes avoid.
- decision:
  accept as process/tooling evidence. Until HRX v1 is much closer to Vulkan
  parity, aggregate numbers remain boulder selection and promotion guardrails
  only. Candidate work should mechanically close named RADV event-window
  deltas, then pass focused correctness and route evidence before model A/B.

## 2026-06-18 - Q8_0 selected pack-stage RADV-vs-HIP event compare

- source:
  `sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py` updated to
  emit event summaries for both sides of a RADV-vs-HSACO comparison.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-packstage-bufferstore-eventcompare-20260618-followup/`.
- compared route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_bufferstore_f16acc_wg256_f32`.
- purpose:
  make the current selected-half Q8_0 direct-WMMA probe fail or advance on the
  exact Vulkan schedule contract, not on aggregate token rate or coarse opcode
  totals.
- result:
  the HIP route still matches only part of the RADV surface: both have
  `32` WMMA, `64 ds_load_b64`, and two barriers, but RADV has `22528` LDS
  bytes, `192` VGPR, `128 ds_store_b16`, `128 ds_load_u16_d16`, and
  `192 buffer_store_b32`; HIP has `20480` LDS bytes, `196` VGPR,
  no halfword writeback stage, and `128 buffer_store_b32`.
- decision:
  reject further selected-half pack-stage variants unless they close a named
  event-window delta. The next exact-schedule candidate should either prove a
  lower-level cooperative-store lane map and reproduce the RADV direct/staged
  writeback split, or import a specific RADV issue-window fact into the current
  packed-Q8_1 production path without reintroducing known spill/pressure
  failures.

## 2026-06-18 - Q8_0 mixed direct/staged coopstore fixture

- source:
  `sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/coopmat_store_contract_bench.hip.cpp`
  adds `--mode=radv-mixed192`.
- purpose:
  test whether CMake/Ninja-built HIP C++ can spell the RADV Q8_0 store-side
  surface in one WG256 kernel: direct full-tile raw buffer stores plus
  halfword LDS staged reload/writeback blocks.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target hrx-hip-bench-coopmat-store-contract -j$(nproc)`.
- runtime evidence:
  after one non-reproduced post-rebuild miss in group 35/slot 1, the fixture
  passed five consecutive runs and then a saved clean pass:
  `cache/hrxv1/gfx1151/coopmat-store-radv-mixed192-pass-20260618-142609/`.
- ISA evidence:
  `cache/hrxv1/gfx1151/coopmat-store-radv-mixed192-isa-20260618-142532/`.
  The `coopstore_probe_radv_mixed192` kernel emits `192 buffer_store_b32`,
  `128 ds_store_b16`, `128 ds_load_u16_d16`, `2 s_barrier`, and
  `135 s_waitcnt`.
- decision:
  accept as a diagnostic primitive, not a production route. HIP source can
  emit the RADV-like mixed store-side opcode surface, so the remaining Q8_0
  parity work should focus on WMMA accumulator ownership, pre-WMMA issue
  windows, and cooperative writeback lane mapping rather than simply adding
  more raw stores.

## 2026-06-18 - Q8_0 first-WMMA schedule score

- source:
  `sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py` now scores
  the first-WMMA issue window in addition to printing raw event windows.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-window-score-20260618-143001/`.
- purpose:
  make the pre-WMMA LDS-load cadence an explicit compile-evidence gate. The
  current HIP candidates match `64 ds_load_b64` in aggregate, but RADV keeps a
  much deeper outstanding LDS-load window before issuing early WMMA ops.
- result:
  RADV scores `59` pre-WMMA `ds_load_b64` events in the first window, with
  `59` loads immediately before the final pre-WMMA wait and
  `final_pre_wmma_lgkmcnt=51`. Both the selected pack-stage route and the
  dualstage halfword route score only `1` load immediately before a final
  `lgkmcnt(0)`, with many interleaved waits after early WMMA instructions.
- decision:
  reject the current pack-stage and dualstage routes as exact Vulkan schedule
  clones at the compile-evidence gate. The next useful Q8_0 candidate must
  preserve correctness while widening the outstanding `ds_load_b64` issue
  window toward RADV; adding halfword stores or raw buffer stores without this
  issue-window convergence is not enough.

## 2026-06-18 - Q8_0 nowait B64 issue-window probe

- source:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp`
  now supports `HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT`, and the catalog adds
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_packstage_bufferstore_wg256.hip.cpp`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-nowait-packstage-score-20260618-143649/`.
- purpose:
  test the first-WMMA schedule-score axis directly by removing the per-helper
  `lgkmcnt(0)` waits around `ds_read_b64` while keeping the selected-half
  pack-stage buffer-store route otherwise unchanged.
- result:
  the route compiled through CMake/Ninja and changed codegen: total
  `s_waitcnt` dropped to `8`, `ds_load_b64` stayed `64`, `buffer_store_b32`
  stayed `128`, and the first scored window increased to `16` WMMA. It did
  not reproduce RADV's outstanding-load cadence: RADV scores `59` pre-WMMA
  `ds_load_b64` and final `lgkmcnt(51)`, while the HIP nowait route scores
  `24` pre-WMMA `ds_load_b64` and final `lgkmcnt(0)`.
- decision:
  reject at compile-evidence gate without runtime. Removing helper waits alone
  is not enough; the compiler schedules early WMMA directly after the load
  burst rather than creating RADV's high-latency outstanding LDS-load window.

## 2026-06-18 - Q8_0 nowait named-fragment issue-window probe

- source:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp`
  combines `HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT` with
  `HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NAMED_FRAGS`; the catalog adds
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_packstage_bufferstore_wg256.hip.cpp`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-packstage-score-20260618-144436/`.
- purpose:
  test whether named A/B fragment variables plus no-wait B64 LDS helpers cause
  LLVM to keep a wider fragment load set live before the first WMMA block,
  instead of interleaving early WMMA immediately after the initial load burst.
- result:
  the route compiled through CMake/Ninja with `64 ds_load_b64`,
  `128 buffer_store_b32`, `2` barriers, `VGPR=196`, `LDS=20480`, `8`
  `s_waitcnt`, and `44 s_waitcnt_depctr`. The first-WMMA score stayed far
  from RADV: RADV scores `59` pre-WMMA `ds_load_b64`, `59` loads immediately
  before the final wait, `final_pre_wmma_lgkmcnt=51`, and `22` WMMA in the
  window; this HIP route scores `24`, `0`, `0`, and `16`.
- decision:
  reject at compile-evidence gate without runtime. Named fragment spelling
  plus helper wait removal does not recover RADV's outstanding LDS-load
  cadence and still lacks the halfword LDS store/load topology and `192`
  `buffer_store_b32` writeback surface.

## 2026-06-18 - Q5_K packed-path hot-op issue-window score

- source:
  `sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py` now reports a
  generic first-hot-op score in addition to the WMMA-specific score.
- artifact:
  `cache/hrxv1/gfx1151/q5-packed-hotop-score-20260618-145056/`.
- purpose:
  quantify the Q5 packed-path `v_dot4_i32_iu8` issue window after B-pair,
  B-quad, and CR-major probes were rejected by focused timing. This keeps the
  next Q5 route evidence-driven even though packed-Q8_1 does not directly
  clone Vulkan's cooperative-matrix WMMA path.
- result:
  default Q5 MMQL128 scores `13` pre-dot LDS loads, final `lgkmcnt=2`,
  `VGPR=141`, no spills. B-pair moves to `15`/`4` with `VGPR=149`, B-quad to
  `20`/`9` with `VGPR=169`, and CR-major to `28`/`17` but spills heavily
  (`VGPR=192`, `135` spills). The rejected variants did widen the intended
  local read window, but timing still regressed for B-pair/B-quad and CR-major
  hit the register cliff.
- decision:
  record as a negative schedule-family result. The next Q5 packed candidate
  should not be another B-cache clustering or CR-major variant; widening the
  pre-dot LDS window alone is not sufficient for Q5_K parity.

## 2026-06-18 - Q5_K BHALF packed RHS scale-cache probe

- source:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc`
  now supports an opt-in B-cache representation that stores Q8_1 RHS `d/s` as
  half payloads in LDS and converts after shared load. The catalog adds
  `mul_mat_vec_q5_k_q8_1_x4_mmql128_bhalf.hip.cpp` and route
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bhalf_wg256_f32`.
- gate:
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BHALF_PROMPT=1`.
- artifacts:
  static score
  `cache/hrxv1/gfx1151/q5-bhalf-hotop-score-20260618-145959/`, focused gate
  `cache/hrxv1/gfx1151/q5-mmql128-bhalf-focused-20260618-150104/`, model A/B
  `cache/hrxv1/gfx1151/q5-mmql128-bhalf-model-ab-20260618-150334/`.
- result:
  after fixing the shared template's missing no-prefetch/default accumulation
  branch, BHALF compiled through CMake/Ninja with wave64, SGPR `49`,
  VGPR `140`, LDS `9728`, no spills, and `512 v_dot4_i32_iu8`.
  It passed focused p33/p512/p513 CPU-reference tests. Route traces show p33
  stayed on the narrow rows2 route, p512 selected BHALF, and p513 stayed on
  the existing MMQL tail path.
- timing:
  focused p512 was mixed but slightly positive in sum: Kcur `+1.59%`, Qcur
  `-1.90%`, ffn_out `-2.68%`, ffn_gate `+0.42%`, total `-1.10%` time.
  Same-binary Qwen2.5 Coder 7B Q5_K_M p512/fa1 model A/B regressed
  `455.164 -> 454.457 tok/s` with BHALF selected and no fallback.
- decision:
  reject for gfx1151 promotion. RHS half-scale cache footprint is not the
  missing Q5 parity axis by itself; future packed-Q5 work should target
  tail/split-K policy, packed RHS ownership, or a lower-level RADV-like store
  path rather than another small cache-layout tweak.

## 2026-06-18 - Q5_K B-quad large-tail policy

- source:
  `sources/llama.cpp/ggml/src/ggml-hrx/ggml-hrx.cpp` now defaults Q5_K large
  non-128-multiple prompt tails (`cols >= 512 && cols % 128 != 0`) to the
  existing B-quad MMQL128 provider on gfx1151.
- rollback:
  `GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL128_BQUAD_TAIL_PROMPT=1`.
- artifacts:
  opt-in focused probe
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-probe-20260618-150846/`,
  opt-in model A/B
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-model-ab-20260618-150954/`,
  default regate
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-default-regate-20260618-151118/`,
  default model A/B
  `cache/hrxv1/gfx1151/q5-mmql128-bquad-tail-default-model-ab-20260618-151218/`.
- result:
  focused p33/p512/p513 CPU-reference gates passed. Route traces prove p33
  stayed on rows2, p512 stayed on current MMQL128, and p513 selected B-quad;
  rollback returned p513 to current MMQL128. Focused p513 default improved the
  four-row sum by `5.63%` versus rollback.
- model:
  Qwen2.5 Coder 7B Q5_K_M p513/fa1/r3 improved
  `393.579 -> 405.465 tok/s` (`1.030x`) default versus rollback, with no
  fallback strings.
- decision:
  accept as a gfx1151 tail-only default. Do not promote Q5 B-quad for p512
  full tiles; that path remains rejected by focused p512 timing. This is a
  tail-policy lift informed by the Vulkan p513 large-tail oracle, while exact
  RADV `split_k_reduce` parity remains future work.

## 2026-06-18 - Q8_0 BM128/BN64 split-qsum rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x64_splitqsum_wg256_f32` as an opt-in
  CMake-built HIP C++ route.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; built through
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops -j$(nproc)`.
- model/shape:
  focused Llama 3.1 8B Q8_0 p33, p512, and p513 prompt rows.
- route or candidate:
  BM128/BN64 packed-Q8_1 route with 16-column split qsum chunks, testing
  whether the accepted split-qsum live-range fix rescues BM128 row ownership.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file <q8 rows>
  --output csv` with current defaults: BN128 split-qsum for p512 and BN112
  split-qsum for p513.
- variant command:
  same focused commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X64_SPLITQSUM_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-mmq128x64-splitqsum-focused-20260618-152805/`.
- profile/timing:
  same artifact, plus focused Vulkan timing
  `cache/hrxv1/gfx1151/q8_0-vulkan0-focused-perf-20260618-152147/`.
- correctness:
  p33, p512, and p513 CPU-reference gates passed; route traces selected the
  opt-in provider for all focused rows.
- timing:
  compile gate improved from old BM128/BN64 VGPR `192` with `47` spills to
  VGPR `144`, no spills. Runtime still regressed: p512 focused total
  `72.631 ms -> 132.665 ms`, p513 focused total
  `80.896 ms -> 139.952 ms`. The largest regressions were p512
  `ffn_gate 6.394 ms -> 15.636 ms` and `result_output 56.614 ms ->
  107.062 ms`.
- decision:
  reject production promotion. Split-qsum fixes the BM128/BN64 compiler cliff
  but does not make BM128 row ownership competitive with the accepted BM64
  split-qsum defaults.
- notes:
  The same focused pass measured current HRX Q8_0 p512 rows at `1.51x`
  Vulkan total and p513 rows at `1.42x` Vulkan total, so Q8_0 remains a real
  parity gap. The next candidate should not be another BM128/BN64 ownership
  probe; it needs either a different packed-path schedule axis or a lower-level
  way to reproduce RADV's cooperative-matrix load/store contract.

## 2026-06-18 - WMMA F16 wave64 output ownership map

- source:
  `sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/wmma_f16_lane_map_bench.hip.cpp`
  now prints and validates the gfx1151 wave64 D-coordinate map for
  `v_wmma_f16_16x16x16_f16`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-f16-lane-map -j$(nproc)`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-lane-map-coords-20260618-154317/`.
- evidence:
  AMD Matrix Instruction Calculator outputs for D matrix layout/register
  layout are saved for OPSEL `0` and OPSEL `4`, and the device fixture passed
  with `opsel=0 changed_even=256 changed_odd=0`,
  `opsel=1 changed_even=0 changed_odd=256`, and
  `check: elements=1024 bad=0 coord_bad=0`.
- conclusion:
  OPSEL `0` and OPSEL `4` select low/high halves for the same 16x16 D
  coordinates. The row/column map is
  `row = 4 * (slot >> 1) + floor(lane / 16)`, `col = lane % 16`.
  This invalidates the earlier fullpair mental model where odd slots were
  treated as another output row band. The next Q8_0 direct-WMMA route should
  use this table as a correctness constraint before attempting to match RADV's
  cooperative-matrix store surface.

## 2026-06-18 - Q8_0 selected fast-half compile-contract rejection

- source:
  `sources/llama.cpp` adds a compile-contract-only wrapper
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_selected_bufferstore_f16acc_wg256_f32`
  plus the shared selected-half fast-half writeback helper.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j$(nproc)` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-packstage-fast-half-selected-bufferstore-compile-20260618-155052/`.
- evidence:
  the CMake-built HSACO is wave64 with SGPR `28`, VGPR `196`, LDS `22528`,
  no spills, `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `2 ds_store_b32`, `128 buffer_store_b32`, and two
  barriers. The opcode surface preserves much of the RADV large Q8_0 oracle,
  but the first-WMMA schedule remains wrong: HIP emits a serialized
  `ds_load_b64` / `s_waitcnt lgkmcnt(0)` pattern with one load immediately
  before the final wait, while RADV keeps `59` LDS loads outstanding with final
  pre-WMMA `lgkmcnt(51)`. The candidate also misses RADV's `VGPR=192` target
  and `192 buffer_store_b32` store surface.
- decision:
  reject before focused runtime. The selected-half ownership map is valid, but
  dummy staging of the unselected half does not make HIP C++ lower to the RADV
  cooperative-matrix schedule. Future Q8_0 direct-WMMA work should target the
  outstanding LDS issue window and full store topology without reusing the
  invalid row+16 fullpair assumption.

## 2026-06-18 - Q8_0 nowait named-fragment staged-wait rejection

- source:
  `sources/llama.cpp` adds a compile-contract-only wrapper
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_stagedwait_packstage_bufferstore_f16acc_wg256_f32`
  plus a gated staged-wait path in the shared Q8_0 VK128 source.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j$(nproc)` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-stagedwait-packstage-score-20260618-continued/`.
- evidence:
  the CMake-built HSACO is wave64 with SGPR `28`, VGPR `196`, LDS `20480`,
  no spills, `32` WMMA, `64 ds_load_b64`, `2 ds_store_b32`,
  `128 buffer_store_b32`, two barriers, `14 s_waitcnt`, and
  `44 s_waitcnt_depctr`. The intended source waits were
  `lgkmcnt(12/8/4/0)`, but the emitted first-WMMA score still shows
  `24` pre-WMMA LDS loads, `0` loads immediately before the final pre-WMMA
  wait, and final pre-WMMA `lgkmcnt(0)`, versus RADV's `59`, `59`, and
  `lgkmcnt(51)`. The partial waits appear after the first WMMA group.
- decision:
  reject at compile-contract gate. Source-visible partial waits in this HIP
  C++ spelling do not force the RADV high-outstanding-LDS issue window. The
  remaining Q8_0 direct-WMMA work likely needs a lower-level load/cooperative
  matrix spelling or a different packed-Q8_1 schedule axis rather than more
  local wait-count decoration.

## 2026-06-18 - Q8_0 nowait named-fragment preuse rejection

- source:
  `sources/llama.cpp` adds a compile-contract-only wrapper
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_preuse_packstage_bufferstore_f16acc_wg256_f32`
  plus an opt-in empty inline-asm use of all eight A/B fragment vectors before
  the first wait.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j$(nproc)` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-preuse-packstage-score-20260618-continued/`.
- evidence:
  the CMake-built HSACO is wave64 with SGPR `28`, VGPR `212`, LDS `20480`,
  no spills, `32` WMMA, `64 ds_load_b64`, `2 ds_store_b32`,
  `128 buffer_store_b32`, two barriers, `8 s_waitcnt`, and
  `44 s_waitcnt_depctr`. The preuse dependency moved the first-WMMA score in
  the intended direction versus the prior nowait named-frag route:
  `24/0/lgkmcnt(0)` became `32/32/lgkmcnt(0)` for pre-WMMA loads, immediate
  loads before final wait, and final wait. RADV remains `59/59/lgkmcnt(51)`.
- decision:
  reject before runtime. The preuse trick is useful evidence but not a viable
  schedule: it still drains LDS to zero, reaches only half the RADV pre-WMMA
  load window, raises VGPR to `212`, and lacks the RADV halfword LDS
  store/load plus `192 buffer_store_b32` topology.

## 2026-06-18 - Q8_0 preuse staged-wait rejection

- source:
  `sources/llama.cpp` adds a compile-contract-only wrapper
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_preuse_stagedwait_packstage_bufferstore_f16acc_wg256_f32`
  and makes the staged-wait ladder constants configurable in the shared Q8_0
  VK128 source.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j$(nproc)` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-preuse-stagedwait-packstage-score-20260618-continued/`.
- evidence:
  the CMake-built HSACO is wave64 with SGPR `28`, VGPR `212`, LDS `20480`,
  no spills, `32` WMMA, `64 ds_load_b64`, `2 ds_store_b32`,
  `128 buffer_store_b32`, two barriers, `14 s_waitcnt`, and
  `44 s_waitcnt_depctr`. The requested `lgkmcnt(24)` and `lgkmcnt(16)` waits
  appear after the first WMMA, but the final pre-WMMA score is still
  `32/0/lgkmcnt(0)` versus RADV's `59/59/lgkmcnt(51)`.
- decision:
  reject at compile-contract gate. Combining all-fragment preuse with nonzero
  wait decoration still does not reproduce RADV's high-outstanding LDS window
  and keeps the bad `VGPR=212` pressure. This closes the cheap HIP C++
  preuse/wait-ladder axis for Q8_0 direct-WMMA.

## 2026-06-18 - Q8_0 and Q6_K HSACO family static triage

- source:
  `sources/llama.cpp` adds
  `tools/vulkan-oracle/summarize_hsaco_family.py`.
- build:
  no rebuild required; the script inspects CMake/Ninja-built HSACOs under
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151`
  using ROCm `/srv/vm-shared/rocm/rocm-head` `llvm-objdump` and
  `llvm-readelf`.
- model/shape:
  static schedule triage for active dense prompt boulders, especially Llama
  3.1 8B Q8_0 p512/p513 and Qwen3 30B Q6_K p512/p513.
- route or candidate:
  batch comparison of already-built Q8_0 packed-Q8_1 `mul_mat_vec_q8_0_mmq*`
  HSACOs and Q6_K `mul_mat_vec_q6_k*` HSACOs.
- baseline command:
  `python3 sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py
  --hsaco-dir build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151
  --glob 'mul_mat_vec_q8_0_mmq*.hsaco' --glob
  'mul_mat_vec_q8_0_q8_1_x4_mmql128.hsaco' ...`.
- variant command:
  same tool with `--glob 'mul_mat_vec_q6_k*.hsaco'`.
- route trace:
  not applicable; this is static compile evidence only.
- profile/timing:
  Q8_0 artifact
  `cache/hrxv1/gfx1151/q8_0-packed-hsaco-family-summary-20260618-161943/`;
  Q6_K artifact
  `cache/hrxv1/gfx1151/q6_k-hsaco-family-summary-20260618-162005/`.
- correctness:
  not run in this entry; all promotion/rejection decisions still defer to the
  focused gates already recorded for each route.
- timing:
  not measured in this entry. The value is static schedule comparison:
  accepted Q8_0 BN112 split-qsum is wave32, VGPR `134`, LDS `3808`, no spills;
  accepted Q8_0 BN128 split-qsum is wave32, VGPR `152`, LDS `4352`, no spills.
  Rejected non-split BN104/112/128 variants hit VGPR `192` and spill, while
  split-qsum8 lowers VGPR to `120` but had already regressed focused runtime.
  All packed Q8_0 variants share the same parsed first-dot issue-window score:
  `28` pre-hot loads, final pre-hot `lgkmcnt(14)`, and `60` hot ops in the
  window. Q6_K large packed/wave64 variants continue to show either heavy
  spilling or the already-rejected direct-WMMA store/load mismatch.
- decision:
  accept the tool and artifacts as triage evidence. Do not start another Q8_0
  simple BN/BM or split-qsum variant without a new dataflow hypothesis: the
  static screen shows split-qsum is a live-range fix, not an issue-window or
  RADV-schedule fix. The next parity candidate should either use a lower-level
  primitive that can express RADV's cooperative load/store schedule or pivot to
  another quant family with a materially different oracle delta.
- notes:
  The first version of the tool accidentally selected renamed `_unused`
  include kernels from multi-kernel HSACOs; it now selects the public
  non-`unused` kernel export by default. Promotion remains governed by focused
  CPU-reference gates, route traces, odd/tail coverage, and same-runner
  timing.

## 2026-06-18 - Vulkan split-K reduce oracle extraction

- source:
  `sources/llama.cpp` adds
  `tools/vulkan-oracle/extract_split_k_reduce.py`.
- artifact:
  `cache/hrxv1/gfx1151/split-k-reduce-oracle-summary-20260618-162944/`.
- inputs:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p513-fa1-20260618-063522/inventory/dispatches_full.jsonl`
  and
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p513-fa1-20260618-061619/inventory/dispatches_full.jsonl`.
- command:
  `python3 sources/llama.cpp/tools/vulkan-oracle/extract_split_k_reduce.py
  --capture qwen25coder-7b-q5km-p513=.../dispatches_full.jsonl
  --capture qwen3-30b-q6k-p513=.../dispatches_full.jsonl
  --out-json .../split_k_reduce_summary.json
  --out-md .../split_k_reduce_summary.md`.
- evidence:
  Qwen2.5 Coder 7B Q5_K_M p513 has `56` `split_k_reduce` dispatches and
  all pair with the immediately preceding producer. Normalized families are
  `42` Q5_K K/V-style rows and `14` Q6_K rows, all with producer workgroups
  `[8,5,1]`, reduce workgroups `[257,1,1]`, output elements `262656`,
  reduction factor `2`, and scratch bytes `2101248`.
  Qwen3 30B Q6_K p513 has `143` `split_k_reduce` dispatches and all pair
  with their producers. Normalized families are `96` Q6_K K/V-style rows with
  the same `[8,5,1] -> [257,1,1]`, `262656`, factor-`2`, `2101248` contract,
  plus `47` F32 MoE-logit rows with output elements `65664`, reduction factor
  `8`, and the same scratch byte budget.
- decision:
  accept the extractor and summary as the p513 split-K acceptance reference.
  This does not promote a HIP route. It upgrades the next Q5/Q6 p513 parity
  task from "notice split_k_reduce in Vulkan" to a concrete runtime/kernel
  contract: matmul producer writes `output_elements * reduce_factor` F32
  partials to scratch, then a separate reduce dispatch writes the final output.
  Do not expect another local Q5/Q6 matmul route knob to close this tail delta
  unless HRX also reproduces or intentionally replaces that scratch/reduce
  behavior.

## 2026-06-18 - HRX split-K runtime feasibility check

- source inspected:
  `sources/llama.cpp/ggml/src/ggml-hrx/ggml-hrx.cpp`,
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q5_k_q8_1.hip.cpp`,
  and `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q6_k.hip.cpp`.
- evidence:
  HRX already has transient scratch allocation through
  `ggml_backend_hrx_request_scratch_buffer`, and the K-quant x Q8_1 prompt
  dispatcher already uses scratch for the quantized RHS. However, the active
  prompt matmul ABI is only `k, rows, cols`, with bindings `[src0, q8_1, dst]`.
  The kernels write directly to `dst[col * rows + row]`.
- blocker:
  existing Q5/Q6 prompt providers cannot be safely reused as Vulkan-style
  split-K partial producers by launching two half-K passes. Their row addressing
  derives `blocks_per_row = k / 256` from the reduced `k`, so a half-K launch
  would use the wrong row stride for the original K-quant tensor. They also
  lack K-offset, source full-row stride, split index/count, partial-output
  stride, and an explicit partial-output layout.
- candidate gate:
  - Production target:
    Qwen2.5 Coder 7B Q5_K_M p513/fa1 K/V rows and Qwen3 30B Q6_K p513/fa1
    K/V rows from
    `cache/hrxv1/gfx1151/split-k-reduce-oracle-summary-20260618-162944/`.
  - Baseline command:
    existing same-runner HRX p513 default-vs-rollback model A/B for Q5 and
    Q6 p513 focused rows, with route traces enabled.
  - Variant command:
    opt-in HRX split-K partial-producer plus reduce path, disabled by default
    until it passes focused gates.
  - Same-runner comparison method:
    focused backend-op timing for the exact K/V p513 rows, then same-binary
    llama-bench p513/fa1 HRX default vs rollback, and Vulkan comparison only
    after the route is selected and correct.
  - Route trace path:
    must show a matmul partial producer followed by an HRX split-K reduce for
    K/V p513 rows; p33 must remain on the medium/narrow path.
  - Scheduler/profile trace path:
    use the same HRX route trace plus per-dispatch profile buckets once the
    prototype is correct.
  - Focused CPU-reference command:
    `test-backend-ops -b HRX0 -o MUL_MAT` narrowed to the exported Q5/Q6 p513
    rows before any model run.
  - Compile report path:
    CMake/Ninja-built HSACOs under
    `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151`.
  - Target listing path:
    `tools/vulkan-oracle/extract_split_k_reduce.py` summary plus route trace
    histogram for the HRX candidate.
  - Prior-art schedule source:
    Vulkan `split_k_reduce` oracle rows: Q5/Q6 K/V use producer WG `[8,5,1]`,
    reduce WG `[257,1,1]`, output elements `262656`, factor `2`, scratch bytes
    `2101248`; Qwen3 F32 MoE logits show the generalized factor-`8` case.
  - Promotion rule:
    only promote if the split-K path passes focused CPU-reference gates,
    selects only on intended p513 tail rows, improves same-runner p513 timing
    versus the current accepted route, and does not regress p33/p512 policy
    rows.
- decision:
  do not implement a runtime hook that simply reuses current matmul kernels.
  The next source change needs a new partial-producer kernel ABI, likely with
  full row stride, K offset, split count, and partial-output stride, plus a
  small reduce kernel or a generalized reduce provider. This is a real
  schedule/dataflow port, not a dispatch wrapper.

## 2026-06-18 - HRX split-K reduce catalog route

- source:
  `sources/llama.cpp` adds
  `ggml/src/ggml-hrx/kernels/split_k_reduce_f32.hip.cpp`, CMake catalog
  build wiring, split catalog entries, and a loadable backend provider handle
  `hrx_split_k_reduce_f32`.
- artifact:
  `cache/hrxv1/gfx1151/split-k-reduce-hrx-catalog-20260618-163707/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- catalog evidence:
  generated catalog
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hrx_catalog.json`
  contains route `hrx_split_k_reduce_f32`; generated embedded catalog
  `hrx_kernel_catalog.cpp` contains the route; generated HSACO exists at
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/split_k_reduce_f32.hsaco`.
- static schedule evidence:
  the CMake-built HSACO is wave32 with SGPR `14`, VGPR `6`, no LDS, no spills,
  and no private segment. The ABI is two bindings and 16 bytes of constants:
  `n` output elements and `split` F32 partials per output.
- validation:
  split catalog JSON parsed cleanly; assembled catalog validation passed
  without `--require-artifacts`. Full `--require-artifacts` is not usable for
  this build because the pre-existing ROCWMMA-dependent
  `flash_attn_ext_f32_f16_prefill_wmma16_hsaco` artifact is skipped when
  ROCWMMA headers are absent.
- decision:
  accept as build-only plumbing for the Vulkan p513 split-K contract. This is
  not a production route and has no selector yet. It prepares the reduce half
  of the schedule; the next required source work is a Q5/Q6 partial-producer
  matmul ABI that writes `src[split][n]` scratch in the exact layout this
  kernel reduces.

## 2026-06-18 - Q5 B-quad split-K partial producer catalog route

- source:
  `sources/llama.cpp` adds
  `mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad_splitk_part.hip.cpp` as a wrapper
  around the accepted Q5 B-quad MMQL128 source. The wrapper preserves the
  existing full-output route when `HRX_Q5_K_Q8_1_X4_MMQL128_SPLIT_K_PARTIAL=0`
  and adds a split-K ABI when the macro is enabled:
  `src0`, `src1`, `dst`, `k`, `rows`, `cols`, `kb_start`, `kb_count`,
  `partial_base`.
- artifact:
  `cache/hrxv1/gfx1151/q5-bquad-splitk-part-catalog-20260618-164535/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- catalog evidence:
  route
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_splitk_part_wg256_f32`
  is present in the generated catalog and embedded catalog. The route is
  build-only, has no production selector, and documents the scratch contract as
  `scratch[split][cols*rows]` with `partial_base` in F32 elements.
- static schedule evidence:
  the CMake-built HSACO is wave64 with SGPR `54`, VGPR `169`, LDS `10240`,
  no private segment, and no SGPR/VGPR spills. It retains the expected integer
  dot shape with `512` `v_dot4_i32_iu8` instructions, `2` barriers, `62` LDS
  instructions, and `64` global stores. The expanded ABI reports
  `kernarg_segment_size=72`, matching three global-buffer arguments plus six
  64-bit scalar values.
- next dispatcher gate:
  allocate scratch for two Q5 p513 partials, launch split 0 with
  `kb_start=0`, `kb_count=q8_blocks/2`, `partial_base=0`, launch split 1 with
  `kb_start=q8_blocks/2`, `kb_count=q8_blocks-q8_blocks/2`,
  `partial_base=rows*cols`, then launch `hrx_split_k_reduce_f32` over
  `rows*cols` outputs with `split=2`.
- decision:
  accept as build-only partial-producer plumbing for the Vulkan Q5 p513
  split-K contract. Do not promote or select it until focused p513
  CPU-reference, route trace, and same-runner timing prove the producer plus
  reduce path correct and useful. p33 must remain on the medium/narrow policy
  row, and p512 must not regress the accepted non-split route.

## 2026-06-18 - Q5 B-quad split-K opt-in dispatcher rejection

- source:
  `sources/llama.cpp` adds an opt-in dispatcher path guarded by
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BQUAD_SPLITK_PROMPT=1`. It selects the
  Q5 B-quad split-K partial producer only for large odd-tail Q5_K prompt rows
  with `rows % 128 == 0`, `cols >= 512`, `cols % 128 != 0`, and both the
  partial producer and `hrx_split_k_reduce_f32` providers available. Default
  behavior is unchanged.
- artifact:
  `cache/hrxv1/gfx1151/q5-bquad-splitk-optin-focused-20260618-165026/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"` passed.
- correctness:
  focused CPU-reference `test-backend-ops test -b HRX0 --output csv
  --test-file q5_prompt_p513.txt` passed all four Q5 p513 rows with the split-K
  env enabled.
- route evidence:
  Kcur stayed on `hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32`; Qcur, ffn_out,
  and ffn_gate each selected two
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_splitk_part_wg256_f32`
  partial producer dispatches followed by `hrx_split_k_reduce_f32`.
- focused perf:
  default vs split-K opt-in on the four Q5 p513 rows:
  `Kcur 880.014 -> 895.594 us`, `Qcur 1504.718 -> 4104.109 us`,
  `ffn_out 8978.841 -> 10011.246 us`, and
  `ffn_gate 8076.927 -> 39143.203 us`. Summed row time regressed
  `19440.500 -> 54154.152 us` (`2.785x` slower).
- decision:
  reject for promotion. The mechanical Vulkan-contract clone is correct, but
  the two full partial producer passes plus a separate global-memory F32 reduce
  are not the performant Vulkan schedule. Keep this path only as an opt-in
  diagnostic hook. The next split-K attempt needs to mine the producer-side
  Vulkan/RADV schedule more deeply, especially how partial production and
  reduction locality avoid the extra traffic exposed by this wrapper.

## 2026-06-18 - Q5 B-quad split-K reduce wg1024 rerun

- source:
  `sources/llama.cpp` changes only `hrx_split_k_reduce_f32` catalog dispatch
  geometry from 256 to 1024 threads per workgroup, matching the Q5 p513 RADV
  `split_k_reduce` oracle `wg_denoms=[1024,1,1]`.
- artifact:
  `cache/hrxv1/gfx1151/q5-bquad-splitk-reduce-wg1024-focused-20260618-172949/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"` passed.
- correctness:
  focused CPU-reference p513 passed all four Q5 rows with split-K enabled.
- route evidence:
  Qcur, ffn_out, and ffn_gate selected two
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_splitk_part_wg256_f32`
  partial producers followed by `hrx_split_k_reduce_f32`. The reduce grids now
  match the 1024-thread contract: `n=1838592 -> [1796,1,1]` and
  `n=9718272 -> [9491,1,1]`, instead of the older 256-thread
  `[7182,1,1]` and `[37962,1,1]`.
- focused perf:
  default vs split-K opt-in on the four Q5 p513 rows:
  `Kcur 896.241 -> 891.726 us`, `Qcur 1507.874 -> 1643.134 us`,
  `ffn_out 8943.630 -> 8887.670 us`, and
  `ffn_gate 8384.141 -> 11830.322 us`. Summed row time regressed
  `19731.885 -> 23252.851 us` (`1.178x` slower).
- decision:
  keep rejected/diagnostic. Matching the reduce launch geometry fixed a real
  oracle mismatch and cut the old rejected split-K diagnostic from `54.15 ms`
  to `23.25 ms`, but the mechanical two-pass global F32 reduce still loses to
  the default `19.73 ms` focused row sum. The next split-K route must move
  producer/reduction locality closer to RADV instead of only adjusting reduce
  workgroup size.

## 2026-06-18 - Q8_0 CMake WMMA issue-window fixture

- source:
  `sources/llama.cpp` adds
  `ggml/src/ggml-hrx/tools/hip-bench/wmma_issue_window_bench.hip.cpp` and a
  CMake/Ninja target `hrx-hip-bench-wmma-issue-window`.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-issue-window-bench-20260618-170000/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-issue-window -j "$(nproc)"` passed using ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- runtime smoke:
  both `--mode=lgkm51` and `--mode=wait0` completed with
  `checksum=65536.000000` and zero NaNs.
- static evidence:
  extracted `device.hsaco` from `.hip_fatbin`, disassembled with
  `llvm-objdump -d --arch-name=amdgcn --mcpu=gfx1151`, and compared against
  the RADV Q8_0 large pipeline with
  `tools/vulkan-oracle/compare_amdgcn_isa.py`.
- schedule score:
  RADV has `59` pre-WMMA `ds_load_b64`, `59` loads immediately before the
  final pre-WMMA wait, final `lgkmcnt(51)`, and `22` WMMA instructions in the
  first window. The HIP `lgkm51` fixture has only `25` pre-WMMA `ds_load_b64`,
  zero loads immediately before the final pre-WMMA wait, final
  `lgkmcnt(0)`, and `7` WMMA instructions in the first window; its explicit
  `lgkmcnt(51)` appears after the first WMMA window because the finite WMMA
  operands are independent of the LDS-loaded fragments.
- decision:
  reject as a route source and keep as a diagnostic fixture. This shows that
  inline `ds_read_b64` alone is not enough; the next low-level Q8_0 probe must
  pin the finite WMMA operands to the issued LDS loads through an inline-asm
  dependency block, or move to a still-lower-level schedule spelling.

## 2026-06-18 - Q8_0 dependency-pinned WMMA issue-window fixture

- source:
  `sources/llama.cpp` updates
  `ggml/src/ggml-hrx/tools/hip-bench/wmma_issue_window_bench.hip.cpp` so the
  finite WMMA operand is produced by an inline-asm block that takes all staged
  LDS-loaded fragments as inputs and emits the selected `s_waitcnt` before the
  operand-defining moves.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-issue-window-depconst-20260618-171500/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-issue-window -j "$(nproc)"` passed.
- runtime smoke:
  both `--mode=lgkm51` and `--mode=wait0` completed with
  `checksum=32768.000000` and zero NaNs.
- static evidence:
  extracted `device.hsaco` from `.hip_fatbin`, disassembled with
  `llvm-objdump`, and compared against the RADV Q8_0 large pipeline with
  `tools/vulkan-oracle/compare_amdgcn_isa.py`.
- schedule score:
  the `lgkm51` fixture now has `64` pre-WMMA `ds_load_b64`, `64` loads
  immediately before the final pre-WMMA wait, final `lgkmcnt(51)`, and zero
  load-like ops after the first WMMA. RADV has `59`, `59`, `lgkmcnt(51)`, and
  `7` load-like ops after the first WMMA. The fixture uses only `8` WMMAs
  versus RADV's `32`, but it preserves the core pre-WMMA issue-window
  signature that previous HIP C++ Q8 probes failed to express.
- decision:
  accept as a diagnostic primitive and use it as the next Q8_0 route-candidate
  axis. The production attempt should port this dependency-pinned
  fragment-production/wait pattern into the real Q8_0 direct-WMMA path, then
  gate it with focused correctness, p512/p513 timing, route trace, and HSACO
  schedule scoring before any selector promotion.

## 2026-06-18 - Q8_0 real-fragment depwait compile-contract rejection

- source:
  `sources/llama.cpp` adds the compile-only route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_packstage_bufferstore_f16acc_wg256_f32`
  plus catalog source/artifact/family/route metadata and a gfx1151 rejection
  row. The route is not wired into runtime selection.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-depwait-realfrag-compile-20260618-174000/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"` passed and regenerated the embedded catalog.
- catalog evidence:
  generated `hrx_catalog.json` contains the depwait route with
  `selector_gate: none; compile-contract probe only`, and
  `hrx_kernel_catalog.cpp` embeds the same route.
- static evidence:
  the HSACO is wave64 with SGPR `28`, VGPR `240`, LDS `20480`, no spills,
  `32` `v_wmma_f16_16x16x16_f16`, `64` `ds_load_b64`, `128`
  `buffer_store_b32`, `2` barriers, `38` `s_waitcnt`, and `44`
  `s_waitcnt_depctr`.
- schedule score:
  the route preserves final pre-WMMA `lgkmcnt(51)` and emits the RADV-like
  wait ladder, but only scores `32` pre-WMMA `ds_load_b64` and `32` loads
  immediately before the final wait. RADV scores `59`, `59`, and
  `lgkmcnt(51)` with VGPR `192`. The route's first window contains `15`
  WMMAs versus RADV's `22`.
- decision:
  reject before focused correctness or timing. The dependency-pinned primitive
  can be transplanted into real payloads, but this spelling exposes only one
  K tile before first WMMA and raises VGPR pressure to `240`. The next Q8_0
  attempt should change the live-range/dataflow shape so both K tiles can feed
  the first issue window without retaining all copied fragments at once.

## 2026-06-18 - Q8_0 two-K-tile depwait compile-contract rejection

- source:
  `sources/llama.cpp` adds the compile-only route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_packstage_bufferstore_f16acc_wg256_f32`.
  It extends the real-fragment depwait probe by forcing both VK128 K tiles'
  A/B fragments live before the first WMMA.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-depwait-k2-realfrag-compile-20260618-172158/`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"` passed and generated the K2 HSACO through CMake/Ninja.
- static evidence:
  the HSACO is wave64 with SGPR `29`, VGPR `256`, VGPR spills `30`, private
  segment `124`, LDS `20480`, `32` `v_wmma_f16_16x16x16_f16`, `64`
  `ds_load_b64`, `128` `buffer_store_b32`, `2` barriers, `67` `s_waitcnt`,
  and `44` `s_waitcnt_depctr`.
- schedule score:
  K2 materially improves the first-WMMA issue window versus the one-tile
  depwait probe: `57` pre-WMMA `ds_load_b64`, `31` loads immediately before
  the final wait, final `lgkmcnt(51)`, and `20` WMMAs in the first window.
  RADV remains better at `59`, `59`, `lgkmcnt(51)`, and `22` WMMAs with VGPR
  `192` and no spills.
- decision:
  reject before runtime. The experiment proves that source-visible HIP C++ can
  be coerced near the RADV first-issue load window, but the cost is a hard
  register cliff. The next useful Q8_0 axis should not retain both K tiles'
  full copied fragments at once; it needs either a lower-level fragment
  spelling/lane-ownership path or a schedule that preserves the two-K issue
  window with shorter live ranges.

## 2026-06-18 - Q8_0 BN128 B-stripe compile-contract rejection

- source:
  `sources/llama.cpp` at `a5c52c84e` plus dirty CMake/source changes adding
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bstripe_wg256_f32` as a
  CMake-built compile probe. No runtime selector or production route was
  added.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; built through CMake/Ninja.
- model/shape:
  static Q8_0 packed prompt route screen for the current p512 full-column
  bottleneck. Focused correctness and timing were intentionally skipped after
  the compile-contract screen failed to move.
- route or candidate:
  BM64/BN128 wave32 packed-Q8_1 split-qsum with explicit local preload of each
  16-column B-cache `iqs` stripe before dot consumption.
- baseline command:
  `python3 sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py
  --hsaco-dir build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151
  --glob 'mul_mat_vec_q8_0_mmq64x128_splitqsum*.hsaco'`.
- variant command:
  same command after compiling
  `mul_mat_vec_q8_0_mmq64x128_splitqsum_bstripe.hip.cpp`.
- route trace:
  not applicable; this candidate was not runtime-selected.
- profile/timing:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-bstripe-compile-20260618/`.
- correctness:
  not run. The compile-contract gate rejected the candidate before a focused
  CPU-reference screen.
- timing:
  not run. Static schedule comparison showed the accepted BN128 split-qsum and
  B-stripe probe both compile as wave32, SGPR `27`, VGPR `152`, LDS `4352`, no
  spills, `512` dot sites, `28` pre-hot loads, final pre-hot `lgkmcnt(14)`,
  and `60` hot ops in the first window.
- decision:
  reject before runtime. Source-visible B-cache stripe preloading does not
  change the emitted packed-Q8_1 first-dot issue/window or resource contract.
- notes:
  This reinforces the current Q8 ledger conclusion: the accepted BN112/BN128
  split-qsum routes solved a live-range/spill problem, not the remaining Vulkan
  schedule gap. Further Q8_0 packed work needs a materially different
  issue/window primitive or a lower-level path, not another local staging
  spelling.

## 2026-06-18 - Q8_0 two-K depwait no-preuse compile-contract rejection

- source:
  `sources/llama.cpp` after commit `5824356d1`, dirty with a compile-only
  wrapper
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_nopreuse_packstage_bufferstore_f16acc_wg256_f32`
  and a shared-source macro that disables the explicit K2 fragment preuse.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; `cmake --build ... --target
  test-backend-ops -j"$(nproc)"` passed and generated the HSACO through
  CMake/Ninja.
- model/shape:
  static Q8_0 direct-WMMA p512 large-route compile screen against the Llama
  3.1 8B Q8_0 Vulkan oracle.
- route or candidate:
  two-K-tile dependency-wait direct-WMMA route with the RADV-like wait ladder,
  but without the explicit empty inline-asm use of all 16 loaded A/B fragments.
- baseline command:
  `python3 sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py
  --glob 'mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait*k2*packstage_bufferstore_wg256.hsaco'`.
- variant command:
  same plus
  `python3 sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py`
  against
  `vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/radv/isa/matmul_q8_0_f32_f16acc_aligned_l__main__72d309e22f889977.amdgcn.txt`.
- route trace:
  not applicable; this candidate was not runtime-selected.
- profile/timing:
  `cache/hrxv1/gfx1151/q8_0-depwait-k2-nopreuse-compile-20260618-continued/`.
- correctness:
  not run. The compile-contract gate rejected the candidate before focused
  CPU-reference.
- timing:
  not run. Static schedule comparison showed no improvement over the original
  K2 probe: wave64, SGPR `29`, VGPR `256`, `30` VGPR spills, private segment
  `124`, LDS `20480`, `32` WMMA sites, `64` `ds_load_b64`, `128`
  `buffer_store_b32`, `2` barriers, first-WMMA score `57` pre-WMMA loads,
  `31` loads immediately before final wait, final `lgkmcnt(51)`, and `20`
  WMMAs in the first window.
- decision:
  reject before runtime. Removing the explicit K2 preuse dependency does not
  reduce pressure or improve the window; the register cliff is inherent to
  exposing both K tiles' real fragments in this HIP C++ source shape.
- notes:
  This narrows the next Q8_0 direct-WMMA path: do not keep both full K tiles'
  copied fragments live in C++. A useful next probe needs a lower-level
  fragment/load primitive, different lane ownership, or a staged schedule that
  preserves RADV's issue window while shortening live ranges.

## 2026-06-18 - Q8_0 two-K direct-wait compile-contract rejection

- source:
  `sources/llama.cpp` after commit `4f64c7803`, dirty with compile-only wrapper
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_packstage_bufferstore_f16acc_wg256_f32`
  and a shared-source direct-wait macro path.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; `cmake --build ... --target
  test-backend-ops -j"$(nproc)"` passed and generated the HSACO through
  CMake/Ninja.
- model/shape:
  static Q8_0 direct-WMMA p512 large-route compile screen against the Llama
  3.1 8B Q8_0 Vulkan oracle.
- route or candidate:
  two-K-tile preload with direct WMMA calls and explicit RADV-like wait counts,
  avoiding the dependency-copy `v_mov` path used by the prior K2 depwait
  probes.
- baseline command:
  `python3 sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py
  --glob 'mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_*k2*packstage_bufferstore_wg256.hsaco'`.
- variant command:
  same plus
  `python3 sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py`
  against the Q8_0 large RADV oracle ISA.
- route trace:
  not applicable; this candidate was not runtime-selected.
- profile/timing:
  `cache/hrxv1/gfx1151/q8_0-k2-directwait-compile-20260618-185220/`.
- correctness:
  not run. The compile-contract gate rejected the candidate before focused
  CPU-reference.
- timing:
  not run. Static screen: wave64, SGPR `29`, VGPR `196`, no spills, private
  segment `0`, LDS `20480`, `32` WMMA sites, `64` `ds_load_b64`, `128`
  `buffer_store_b32`, and `2` barriers. The resource result is the useful
  signal: it removes the K2 depwait cliff (`VGPR=256`, `30` spills). The
  schedule result fails: first-WMMA score is only `24` pre-WMMA loads, `0`
  loads immediately before final wait, final `lgkmcnt(0)`, and `16` WMMAs in
  the first window, versus RADV `59/59/lgkmcnt(51)` and `22` WMMAs.
- decision:
  reject before runtime. The dependency-copy path is what forces the useful
  wait window but also causes the spill cliff; direct waits keep pressure
  reasonable but are rescheduled away from the RADV issue contract.
- notes:
  This is an important negative result. A source-level C++ route now has two
  separated halves of the target schedule, but not both at once. The next Q8_0
  path likely needs lower-level load/WMMA issue control or a different
  lane-ownership/writeback strategy, not another direct wait-count decoration.

## 2026-06-18 - DeepSeek Q6 rollback policy rejection

- source:
  `sources/llama.cpp` at `02111f1a6`, clean before recording the rejection.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  DeepSeek R1 Qwen 14B Q4_K_M exported focused Q6_K prompt rows from p512,
  plus a non-clean p513/single-graph export used only as a sanity signal.
- route or candidate:
  selector rollback for the accepted Q6 large WMMA default
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`.
- baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1
  test-backend-ops perf -b HRX0 -o MUL_MAT --test-file
  focus-p512/q6_prompt.txt --output csv`.
- variant command:
  same with
  `GGML_HRX_DISABLE_Q6_K_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/deepseek-q6-row-policy-probe-20260618-191114/`.
- profile/timing:
  `perf-default-p512.csv`, `perf-rollback-p512.csv`, and route traces in the
  same artifact.
- correctness:
  p512 CPU-reference passed for default and rollback on Vcur-0, ffn_out-0,
  and result_output.
- timing:
  rollback regressed all p512 rows: Vcur `1450.68 -> 2939.99 us`, ffn_out
  `24145.94 -> 198921.26 us`, result_output
  `292046.56 -> 2199564.33 us`.
- decision:
  reject rollback/policy change. Current Q6 VK128 WMMA remains the correct
  default for these DeepSeek rows.
- notes:
  DeepSeek's remaining Vulkan gap is not a simple embedded-Q6 selector mistake.
  The next useful Q6 work needs a materially different schedule primitive or a
  different measured boulder, not reverting to the old rows2/MMQL64 paths.

## 2026-06-18 - DeepSeek p33/p513 Vulkan oracle coverage

- source:
  `sources/llama.cpp` clean at `344f74416`.
- build:
  `build/vulkan-gfx1151/bin/llama-bench`, Vulkan backend, `spirv-dis`
  available from system `spirv-tools`.
- model:
  `shared/models/llamacpp-hrx2-basket-v1/unsloth__DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`.
- p513 artifact:
  `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p513-fa1-20260618-192702/`.
- p33 artifact:
  `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p33-fa1-20260618-192733/`.
- command shape:
  p513 used `-p 513 -n 0 -b 1024 -ub 1024 -fa 1 -r 1 -dev Vulkan0`;
  p33 used `-p 33 -n 0 -b 33 -ub 33 -fa 1 -r 1 -dev Vulkan0`.
- capture outputs:
  both rows produced `17` pipeline blocks, `17` SPIR-V files, SPIR-V asm,
  split RADV ISA/stats, `917` dispatch signatures, and `29` normalized shape
  signatures.
- p513 timing:
  Vulkan `246.389660 tok/s`. Against the existing same-build HRX scoreboard
  row `212.575388 tok/s`, the measured HRX/Vulkan ratio is `0.863x`.
- p33 timing:
  Vulkan `33.009663 tok/s`; no matching current HRX scoreboard row was in the
  reduced p512/p513 artifact.
- schedule facts:
  p33 selects the medium aligned Q4/Q6 pipelines,
  `matmul_q4_k_f32_f16acc_aligned_m` and
  `matmul_q6_k_f32_f16acc_aligned_m`, with
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`. p513 selects the large aligned
  Q4/Q6 pipelines, `matmul_q4_k_f32_f16acc_aligned_l` and
  `matmul_q6_k_f32_f16acc_aligned_l`, with
  `spec=[256,128,128,32,64,64,2,16,16,16,64]` and a fifth workgroup column.
- decision:
  accept as oracle coverage. This closes the missing DeepSeek p513 Vulkan
  comparison and adds the narrow p33 oracle row. It does not promote a HIP
  route. Combined with the rejected DeepSeek Q6 rollback, the next DeepSeek
  route work should target the large aligned Q4/Q6 schedule itself rather than
  reverting embedded Q6 rows to old rows2/MMQL64 paths.

## 2026-06-18 - Q4_K MMQ64 narrow prompt default for DeepSeek p33

- source:
  `sources/llama.cpp` after `344f74416`, selector patched to default MMQ64 for
  narrow gfx1151 Q4_K prompt rows.
- route:
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmq64x64_wg256_f32`.
- policy:
  default on gfx1151 for Q4_K prompt rows with `32 <= cols < 128`,
  `rows % 64 == 0`, `k % 256 == 0`, contiguous tensors, and packed Q8_1 x4
  available.
- rollback:
  `GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`.
- opt-in elsewhere:
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`.
- prior-art schedule source:
  DeepSeek p33 Vulkan oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p33-fa1-20260618-192733/`
  selected medium aligned Q4/Q6 pipelines. HRX did not have an exact Q4 medium
  WMMA clone that had survived large-route evidence, but the existing packed
  MMQ64 route matches the narrow tile regime much better than scalar Q4 routes.
- focused probe:
  `cache/hrxv1/gfx1151/deepseek-q4-p33-mmq64-policy-probe-20260618-193047/`.
  CPU-reference passed for Kcur-0, Qcur-0, ffn_out-6, and ffn_gate-0.
  Focused total improved `30.428 ms -> 3.468 ms`.
- model A/B:
  `cache/hrxv1/gfx1151/deepseek-q4-p33-mmq64-model-ab-20260618-193209/`.
  DeepSeek p33/fa1 improved `15.989970 -> 107.104324 tok/s`, which is
  `3.245x` the captured Vulkan p33 oracle row (`33.009663 tok/s`).
- default/rollback regate:
  `cache/hrxv1/gfx1151/deepseek-q4-p33-mmq64-default-regate-20260618-193411/`.
  Default selected MMQ64 for `858` model-route rows, passed focused
  CPU-reference, and ran at `104.827265 tok/s`. Rollback selected the prior
  `q4_k_q8_1_f32`/`q4_k_f32` routes and ran at `15.688643 tok/s`.
- broader p33 guardrail:
  `cache/hrxv1/gfx1151/llama31-q4-p33-mmq64-default-smoke-20260618-193620/`.
  Llama 3.1 8B Q4_K_M p33/fa1 default selected MMQ64 for `570` model-route
  rows and ran at `177.760794 tok/s`; rollback selected the old
  `q4_k_q8_1_f32`/`q4_k_f32` routes and ran at `35.147854 tok/s`. The captured
  Llama p33 Vulkan oracle row is `36.504047 tok/s`, so the new default is
  `4.870x` Vulkan on this guardrail.
- decision:
  accept as gfx1151 default for narrow Q4_K prompt rows. This fixes a concrete
  odd-size p33 policy miss and does not perturb accepted p512/p513 B-quad
  policy because the selector is limited to `cols < 128`.

## 2026-06-18 - Q8_0 clean odd/tail parity recheck at source HEAD

- source:
  `sources/llama.cpp` clean at `4e2d724d1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; rebuilt with CMake/Ninja after the source
  HEAD check. CMake reported `ggml commit: 4e2d724d1`.
- model/shape:
  Llama 3.1 8B Q8_0, `p33/n0/fa1 b33 ub33`,
  `p512/n0/fa1 b512 ub512`, and `p513/n0/fa1 b1024 ub1024`.
- route or candidate:
  current accepted packed-Q8_1 Q8_0 prompt routes:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32` for p33,
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` for p512, and
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` for p513.
- baseline command:
  clean Vulkan oracle captures with
  `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench build/vulkan-gfx1151/bin/llama-bench --device Vulkan0`
  at the same prompt/batch shapes.
- variant command:
  serial HRX `llama-bench` runs with `GGML_HRX_TRACE_ROUTES=1
  GGML_HRX_TRACE_PROVIDERS=1`, `-dev HRX0`, `--no-warmup`, `-ngl 99`, and
  repetitions `3` for p33/p513 and `5` for p512.
- route trace:
  - p33:
    `cache/hrxv1/gfx1151/llama31-q8_0-p33-hrx-score-head-serial-20260618-194803/`
  - p512:
    `cache/hrxv1/gfx1151/llama31-q8_0-p512-hrx-score-head-serial-20260618-194812/`
  - p513:
    `cache/hrxv1/gfx1151/llama31-q8_0-p513-hrx-score-head-serial-20260618-194828/`
- profile/timing:
  Vulkan oracle artifacts:
  - p33:
    `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-194300/`
  - p512:
    `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260618-194436/`
  - p513:
    `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-193949/`
- correctness:
  model runs completed with `backends=HRX` for HRX rows and `backends=Vulkan`
  for Vulkan rows. Route traces showed no fallback or CPU lines. This is a
  parity recheck, not a new kernel correctness promotion.
- timing:
  p33 HRX `203.789607 tok/s` versus Vulkan `44.605816 tok/s`, ratio `4.569x`;
  p512 HRX `458.351637 tok/s` versus Vulkan `394.089035 tok/s`, ratio
  `1.163x`; p513 HRX `420.915286 tok/s` versus Vulkan `399.650136 tok/s`,
  ratio `1.053x`.
- decision:
  Q8_0 is not the next active production-width boulder. Keep the accepted
  packed-Q8_1 Q8_0 routes guarded by odd/tail coverage, but move the next
  schedule work back to the remaining measured Q6_K and DeepSeek Q4_K_M gaps
  unless a fresh full-basket rerun contradicts this recheck.
- notes:
  The earlier p33 Vulkan oracle artifact
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-continued/`
  reported `backends=Vulkan,HRX` and is superseded by the clean Vulkan-only
  p33 capture above. The artifacts
  `llama31-q8_0-p{33,512,513}-hrx-score-head-20260618-1947*` were run
  concurrently and must not be used for timing; the serial artifacts listed
  above are the valid HEAD measurements.

## 2026-06-18 - captured Vulkan-oracle matrix parity checkpoint

- source:
  `sources/llama.cpp` clean at `4e2d724d1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; rebuilt with CMake/Ninja before current
  HEAD measurements.
- model/shape:
  captured Vulkan-oracle rows for Llama 3.1 8B Q4_K_M, Qwen2.5 Coder 7B
  Q5_K_M, DeepSeek R1 Qwen 14B Q4_K_M, Qwen3 30B Q6_K, and Llama 3.1 8B
  Q8_0. Production rows use p512/p513 with `b=1024 ub=1024` except Q8 p512
  recheck uses exact `b=512 ub=512`; narrow rows use exact `p33 b33 ub33`.
- route or candidate:
  current accepted HRX v1 gfx1151 HIP C++ catalog routes. No new source route
  was promoted in this entry.
- baseline command:
  saved clean Vulkan oracle captures from the `vulkan-oracle-*` artifacts in
  `cache/hrxv1/gfx1151/`, with exact p33 recaptures for Q5/Q6 and clean p33
  recapture for Q8_0.
- variant command:
  serial HRX `llama-bench` runs with `GGML_HRX_TRACE_ROUTES=1
  GGML_HRX_TRACE_PROVIDERS=1`, `-dev HRX0`, `--no-warmup`, `-ngl 99`, and
  repetitions `3` unless otherwise noted.
- route trace:
  HRX p512/p513 artifacts:
  - `cache/hrxv1/gfx1151/llama31-8b-q4km-p512-hrx-score-head-serial-20260618-195151/`
  - `cache/hrxv1/gfx1151/llama31-8b-q4km-p513-hrx-score-head-serial-20260618-195155/`
  - `cache/hrxv1/gfx1151/qwen25coder-7b-q5km-p512-hrx-score-head-serial-20260618-195158/`
  - `cache/hrxv1/gfx1151/qwen25coder-7b-q5km-p513-hrx-score-head-serial-20260618-195202/`
  - `cache/hrxv1/gfx1151/deepseek-r1-qwen14b-q4km-p512-hrx-score-head-serial-20260618-195012/`
  - `cache/hrxv1/gfx1151/deepseek-r1-qwen14b-q4km-p513-hrx-score-head-serial-20260618-195018/`
  - `cache/hrxv1/gfx1151/qwen3-30b-q6k-p512-hrx-score-head-serial-20260618-195024/`
  - `cache/hrxv1/gfx1151/qwen3-30b-q6k-p513-hrx-score-head-serial-20260618-195033/`
  - `cache/hrxv1/gfx1151/llama31-q8_0-p512-hrx-score-head-serial-20260618-194812/`
  - `cache/hrxv1/gfx1151/llama31-q8_0-p513-hrx-score-head-serial-20260618-194828/`
  HRX p33 artifacts:
  - `cache/hrxv1/gfx1151/llama31-8b-q4km-p33-hrx-score-head-serial-20260618-195403/`
  - `cache/hrxv1/gfx1151/deepseek-r1-qwen14b-q4km-p33-hrx-score-head-serial-20260618-195405/`
  - `cache/hrxv1/gfx1151/qwen3-30b-q6k-p33-hrx-score-head-serial-20260618-195301/`
  - `cache/hrxv1/gfx1151/qwen25coder-7b-q5km-p33-hrx-score-head-serial-20260618-195303/`
  - `cache/hrxv1/gfx1151/llama31-q8_0-p33-hrx-score-head-serial-20260618-194803/`
- profile/timing:
  saved Vulkan oracle artifacts:
  `vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212`,
  `vulkan-oracle-llama31-8b-q4km-p513-fa1-20260617-200751`,
  `vulkan-oracle-qwen25coder-7b-q5km-p512-fa1-20260617-200349`,
  `vulkan-oracle-qwen25coder-7b-q5km-p513-fa1-20260618-063522`,
  `vulkan-oracle-deepseek-r1-qwen14b-q4km-p512-fa1-20260617-200426`,
  `vulkan-oracle-deepseek-r1-qwen14b-q4km-p513-fa1-20260618-192702`,
  `vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437`,
  `vulkan-oracle-qwen3-30b-q6k-p513-fa1-20260618-061619`,
  `vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260618-194436`,
  `vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-193949`,
  `vulkan-oracle-qwen3-30b-q6k-p33-fa1-exact-20260618-195321`,
  `vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-exact-20260618-195327`,
  and `vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-194300`.
- correctness:
  all HRX route traces for this checkpoint reported no fallback or CPU lines.
  This is a model-level parity checkpoint, not a replacement for focused
  CPU-reference gates when changing routes.
- timing:
  p512/p513 captured matrix ratios: Llama Q4 `1.778x/1.667x`, Qwen2.5 Q5
  `1.660x/1.517x`, DeepSeek Q4 `1.295x/1.232x`, Qwen3 Q6
  `1.076x/1.051x`, and Llama Q8 `1.163x/1.053x`. The ten-row geomean is
  `1.323x` Vulkan. Exact p33 ratios: Llama Q4 `4.867x`, DeepSeek Q4 `3.300x`,
  Qwen3 Q6 `7.508x`, Qwen2.5 Q5 `3.711x`, and Llama Q8 `4.569x`; p33 geomean
  is `4.593x` Vulkan.
- decision:
  accept as a current parity checkpoint for the captured Vulkan-oracle rows.
  Do not mark the broader goal complete yet: continue with full-basket reruns
  as more GGUFs finish downloading, preserve focused CPU-reference gates for
  any source change, and expand odd/narrow/tail coverage before broad selector
  changes.
- notes:
  the older `current-scoreboard-after-q5tail-20260618-151637` artifact is now
  stale for boulder ranking. The current evidence says the next work should be
  validation breadth and route robustness, not blind Q8/Q6/DeepSeek schedule
  tuning.

## 2026-06-18 - direct full-basket KPI supersedes captured-oracle parity checkpoint

- source:
  `sources/llama.cpp` clean at `4e2d724d1` before the Q6 ID default edit.
- build:
  `build/hrx-v1-catalog-gfx1151` and `build/vulkan-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  full downloaded basket under
  `shared/models/llamacpp-hrx2-basket-v1`, cases `p33`, `p512`, and `p513`.
- route or candidate:
  current accepted HRX v1 gfx1151 HIP C++ catalog routes at `4e2d724d1`.
- baseline command:
  `python3 tools/hrxv1_basket_benchmark.py --backends hrx,vulkan --repetitions 3 --timeout 1200 --tag basket-head-full-commitaligned-20260618-200300`.
- variant command:
  none; this is the direct KPI checkpoint for subsequent promotions.
- route trace:
  `cache/hrxv1/gfx1151/basket-head-full-commitaligned-20260618-200300/`.
- profile/timing:
  same artifact; HRX and Vulkan rows both report build commit `4e2d724d1`.
- correctness:
  model runs completed with JSON-confirmed `backends=HRX` for HRX rows and
  `backends=Vulkan` for Vulkan rows. Route traces reported zero HRX fallback
  lines.
- timing:
  average geomean HRX/Vulkan `0.433x`; steady-state geomean `0.422x`; all
  `24/24` rows below parity. Worst steady rows were Qwen3 30B Q6_K p512
  `206.594 / 984.058 = 0.210x`, Qwen3 30B Q6_K p513
  `202.137 / 942.842 = 0.214x`, Qwen3 30B Q4_K_XL p512
  `281.452 / 1190.790 = 0.236x`, and Qwen3 30B Q4_K_XL p513
  `279.168 / 1134.165 = 0.246x`.
- decision:
  this supersedes the captured Vulkan-oracle parity checkpoint for KPI
  decisions. Vulkan-oracle artifacts remain schedule evidence, but direct
  same-binary basket rows are the throughput baseline. The immediate boulder
  is Qwen3 MoE `MUL_MAT_ID`, starting with Q6_K.

## 2026-06-18 - Q6_K grouped MUL_MAT_ID threshold-32 default promotion

- source:
  `sources/llama.cpp` edited from `4e2d724d1` to default the Q6_K grouped
  `MUL_MAT_ID` selector on gfx1151 unless
  `GGML_HRX_DISABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1`, and to align the support
  predicate with `n_tokens >= 32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-ops -j$(nproc)` passed.
- model/shape:
  Qwen3 30B Q6_K MoE `MUL_MAT_ID` prompt rows p33, p512, and p513 with
  `fa=1`; focused exported rows cover `ffn_moe_gate` and `ffn_moe_down`.
- route or candidate:
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`, default on gfx1151
  for Q6_K `MUL_MAT_ID` with `k % 256 == 0`, `rows % 64 == 0`, `n_ids == 8`,
  and `n_tokens >= 32`.
- baseline command:
  rollback model sweep with
  `GGML_HRX_DISABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1 python3 tools/hrxv1_basket_benchmark.py --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k --cases p33,p512,p513 --backends hrx --repetitions 3 --timeout 1200 --tag qwen3-q6-id-threshold32-rollback-20260618-201820`.
- variant command:
  default model sweep:
  `python3 tools/hrxv1_basket_benchmark.py --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k --cases p33,p512,p513 --backends hrx --repetitions 3 --timeout 1200 --tag qwen3-q6-id-threshold32-default-20260618-201756`.
- same-runner comparison method:
  same build, same HRX runner, three repetitions, compared against rollback
  and direct same-commit Vulkan rows from
  `cache/hrxv1/gfx1151/basket-head-full-commitaligned-20260618-200300/`.
- route trace path:
  default:
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold32-default-20260618-201756/`;
  rollback:
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold32-rollback-20260618-201820/`.
- scheduler/profile trace path:
  route traces in the model artifacts; no separate profiler trace was captured
  for this selector-policy promotion.
- focused CPU-reference command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_EXPECT_MUL_MAT_ID_Q6_PROVIDER=hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test -b HRX0 -o MUL_MAT_ID --test-file <p33|p512|p513 moe_qk_prompt.txt> --output csv`.
- focused CPU-reference result:
  `cache/hrxv1/gfx1151/q6-id-threshold32-default-regate-20260618-201739/`;
  p33, p512, and p513 each had two supported rows, zero failures, and route
  traces selecting the expected grouped Q6 ID provider.
- compile report path:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_id_q6_k_q8_1_x4_mmq.hsaco`.
- target listing path:
  `ninja -C build/hrx-v1-catalog-gfx1151 -t targets` lists
  `mul_mat_id_q6_k_q8_1_x4_mmq.hsaco`; the generated catalog structure
  validates. Strict `--require-artifacts` catalog validation currently fails
  on unrelated pre-existing metadata for excluded
  `flash_attn_ext_f32_f16_prefill_wmma16.hsaco`.
- prior-art schedule source:
  Vulkan Q6_K `MUL_MAT_ID` oracle capture
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-commitaligned-20260618-200645/`
  reports `matmul_id_subgroup_q6_k_f32_f16acc_aligned_m`,
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `SGPR=108`, `VGPR=144`,
  `LDS=12288`, and no spills.
- timing:
  p33 default `97.017 tok/s`, rollback `114.288`, Vulkan `170.298`;
  p512 default `588.052`, rollback `238.257`, Vulkan `984.058`;
  p513 default `573.081`, rollback `215.994`, Vulkan `942.842`.
  Three-row steady geomean improved from `0.334x` Vulkan with rollback to
  `0.591x` Vulkan with the default route; default over rollback geomean was
  `1.771x`.
- rejected adjacent policy:
  threshold-64 default artifact
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold64-default-20260618-202012/`
  preserved p512/p513 but reproduced the p33 `~15 tok/s` placement cliff while
  selecting no visible Q6 ID route. It is recorded in
  `ggml/src/ggml-hrx/catalog/tuning/gfx1151/rejections.json`.
- promotion rule:
  accept as a gfx1151 default because it removes the largest Qwen3 Q6_K MoE
  structural miss and materially improves the direct KPI rows, with rollback
  available. Do not call this parity: p33 needs a true Vulkan-medium Q6 ID
  route or expert-weight placement fix, and p512/p513 still sit near `0.60x`
  Vulkan.

## 2026-06-18 - post-Q6-ID direct full-basket KPI checkpoint

- source:
  `sources/llama.cpp` clean at `07167d398`.
- build:
  `build/hrx-v1-catalog-gfx1151` and `build/vulkan-gfx1151`, rebuilt after
  the Q6 ID commit. Both HRX and Vulkan JSON rows report build commit
  `07167d398`.
- model/shape:
  full downloaded basket under `shared/models/llamacpp-hrx2-basket-v1`, cases
  `p33`, `p512`, and `p513`.
- route or candidate:
  committed Q6_K grouped `MUL_MAT_ID` default plus the current HRX v1 gfx1151
  catalog.
- baseline command:
  prior direct basket
  `cache/hrxv1/gfx1151/basket-head-full-commitaligned-20260618-200300/`,
  source/build commit `4e2d724d1`.
- variant command:
  `python3 tools/hrxv1_basket_benchmark.py --models all --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --timeout 1200 --tag basket-after-q6-id-default-commitaligned-20260618-203035`.
- route trace:
  `cache/hrxv1/gfx1151/basket-after-q6-id-default-commitaligned-20260618-203035/`.
- profile/timing:
  same artifact.
- correctness:
  model runs completed with JSON-confirmed `backends=HRX` for HRX rows and
  `backends=Vulkan` for Vulkan rows. HRX fallback lines were zero for all rows.
- timing:
  steady-state geomean improved from `0.422x` Vulkan before the Q6 ID default
  to `0.478x` Vulkan after it. Average geomean improved from `0.433x` to
  `0.485x`. All `24/24` rows remain below parity.
- worst rows:
  Qwen3 30B Q4_K_XL p512 `281.264 / 1202.565 = 0.234x`, Qwen3 30B Q4_K_XL
  p513 `274.572 / 1129.760 = 0.243x`, Qwen3-Coder 30B Q4_K_M p512
  `390.842 / 1126.665 = 0.347x`, Qwen3-Coder 30B Q4_K_M p513
  `381.238 / 1059.255 = 0.360x`, and Qwen2.5 Coder 7B Q5_K_M p33
  `136.317 / 359.810 = 0.379x`.
- decision:
  the Q6 default is a real KPI lift but not parity. The next boulder is the
  Qwen3/Qwen3-Coder Q4_K MoE prompt path. Use the Vulkan oracle schedules for
  `matmul_id_subgroup_q4_k_f32_f16acc_aligned_m` and dense Q4_K prompt rows as
  the mechanical reference before adding more HIP C++ candidates.

## 2026-06-18 - Q5_K narrow FFN gate VK64 default promotion

- source:
  `sources/llama.cpp` edited from `0db6dede3` to default the existing
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`
  provider only for the gfx1151 Qwen2.5 Coder Q5_K_M narrow FFN gate shape.
- route or candidate:
  default on gfx1151 when `k == 3584`, `rows == 18944`, and
  `32 <= cols <= 64`; rollback with
  `GGML_HRX_DISABLE_Q5_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_NARROW_PROMPT=1`
  or the broader `GGML_HRX_DISABLE_FAST_APPROX_PROMPT=1`.
- prior-art schedule source:
  Vulkan p33 oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-exact-20260618-195327/`
  selects `matmul_q5_k_f32_f16acc_aligned_m`, the medium
  BM64/BN64/BK32/WG256 wave64 schedule with `LDS=11264`, `VGPR=144`, and no
  spills.
- focused CPU-reference result:
  `cache/hrxv1/gfx1151/q5-vk64-narrow-default-regate-20260618-210315/`.
  p33 Q5 rows passed against CPU reference. p512 and p513 focused rows also
  passed and route traces showed the new policy does not steal production-width
  routes: p512 stays on rows2 plus MMQL128, and p513 stays on rows2 plus
  MMQL128 B-quad.
- focused timing:
  p33 `ffn_gate-0` improved from rollback `2438.595 us` to default
  `1819.745 us` (`1.34x`). Kcur, Qcur, and ffn_out remained on the existing
  routes.
- model A/B:
  `cache/hrxv1/gfx1151/q5-vk64-narrow-model-ab-p33-code-ff1f839c8-20260618-210913/`,
  Qwen2.5 Coder 7B Q5_K_M p33/fa1/r3. Default reached `153.594 tok/s`;
  rollback reached `137.120 tok/s`; the current basket Vulkan row is
  `360.453 tok/s`. This moves the row from `0.380x` to `0.426x` Vulkan.
- decision:
  accept only this narrow gfx1151 slice. The broad VK64/padded44 Q5 route
  remains rejected because it regresses Kcur, Qcur, and ffn_out. This is not a
  parity claim; further Q5 work should mechanically target the remaining RADV
  medium-schedule load/store/lane-ownership delta instead of widening the
  selector.

## 2026-06-19 - Q4_K MMQL64 BK2 B-pair rejection

- source:
  `sources/llama.cpp` committed as `24fc9766e` with the opt-in diagnostic
  route retained in the CMake/Ninja-built catalog.
- route or candidate:
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bpair_wg256_f32`, enabled only
  with `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL64_BK2_BPAIR_PROMPT=1`.
- prior-art schedule source:
  accepted Q4_K MMQL64 BK2 narrow route plus the accepted Q4_K B-quad narrow
  policy. The B-pair probe changed only B-cache consume order to preload each
  TN=2 B-cache pair before issuing dots.
- focused CPU-reference result:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bpair-focused-20260619-001650/`.
  Llama 3.1 p33, p512, p513, and Llama 3.2 p33 rows passed. p512/p513 stayed
  on the accepted MMQL128 B-quad route.
- compile report:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2_bpair.hsaco`,
  wave64, SGPR `50`, VGPR `103`, LDS `8192`, no spills.
- timing:
  Llama 3.2 focused p33 improved from default `10503.04 us` to B-pair
  `10311.44 us`, and beat forced B-quad `10397.44 us`.
- model A/B:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bpair-llama32-model-ab-20260619-001844/`.
  Llama 3.2 3B Q4_K_M p33/fa1 rejected the route: default `460.129 tok/s`,
  forced B-pair `441.843 tok/s`, with `830` Q4 rows selecting B-pair.
- decision:
  reject for default promotion. Keep as an opt-in diagnostic only. This is a
  useful guardrail: a focused backend-op lift did not survive integrated model
  scheduling, so future rows=3072 work needs a different axis than B-cache
  pair ordering.

## 2026-06-19 - Qwen3 Q6_K current-head subset check

- source:
  `sources/llama.cpp` clean at `24fc9766e`; the benchmark binary reported
  HRX build commit `c888948cf` because generated build metadata was not
  refreshed after the opt-in B-pair commit. Since B-pair is disabled by
  default, this is default-equivalent evidence, not a strict commit-aligned
  artifact.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k --cases p33,p512,p513 --backends hrx,vulkan --repetitions 1 --timeout 1200 --tag qwen3-q6-current-head-r1-20260619-continued`.
- artifact:
  `cache/hrxv1/gfx1151/qwen3-q6-current-head-r1-20260619-continued/`.
- correctness/route evidence:
  HRX fallback lines were zero. p33 selected
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`
  for dense Q6 and
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32` for MoE Q6 ID.
  p512/p513 selected the VK128 dense Q6 route plus the same grouped Q6 ID
  provider.
- timing:
  p33 `84.683 / 150.088 = 0.564x` Vulkan, p512
  `547.507 / 994.327 = 0.551x`, p513 `545.891 / 979.095 = 0.558x`;
  three-row geomean `0.557x` Vulkan.
- decision:
  Q6 remains a first-tier parity gap. The current dense VK64/VK128 and grouped
  ID routes are correct and selected, but still far from Vulkan. The next Q6
  work should not be another selector threshold; it needs either a true
  Vulkan-medium Q6 ID schedule for p33 or a lower-level cooperative
  load/WMMA/store primitive that closes the documented RADV issue-window and
  writeback delta.

## 2026-06-19 - Q6_K MUL_MAT_ID direct-F32 WMMA diagnostic rejection

- source:
  `sources/llama.cpp` dirty after `24fc9766e`, with the opt-in diagnostic route
  added to the CMake/Ninja-built catalog.
- route or candidate:
  `hrx_mul_mat_id_q6_k_wmma16x16_direct_f16acc_wg32_f32`, enabled only with
  `GGML_HRX_ENABLE_Q6_K_ID_WMMA16_DIRECT_PROMPT=1`.
- prior-art schedule source:
  Vulkan Qwen3 Q6_K p33 oracle
  `matmul_q6_k_f32_f16acc_aligned_m` plus the existing grouped Q6 ID route.
  The probe intentionally skipped Q8_1 x4 packing to test raw F32 RHS semantics
  and the proven dense Q6 16x16 direct-WMMA lane mapping before attempting a
  fuller RADV-like grouped schedule.
- focused CPU-reference result:
  `cache/hrxv1/gfx1151/q6-id-wmma16-direct-focused-20260619-003510/`.
  p33, p512, and p513 Qwen3 Q6_K `MUL_MAT_ID` rows passed. Route traces proved
  the opt-in direct provider selected with `direct_f32=1`; default traces
  selected the current grouped Q8_1/x4 route.
- compile report:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_id_q6_k_wmma16_direct.hsaco`,
  wave32, SGPR `54`, VGPR `52`, LDS `0`, no private segment, no spills, one
  visible `v_wmma_f16_16x16x16_f16`.
- focused timing:
  `cache/hrxv1/gfx1151/q6-id-wmma16-direct-perf-20260619-003759/`.
  Direct-F32 regressed every row versus the grouped Q8_1/x4 default:
  p33 gate `300.02 -> 2347.06 us`, p33 down `262.13 -> 2153.97 us`;
  p512 gate `2399.79 -> 15890.97 us`, p512 down
  `2319.38 -> 19716.80 us`; p513 gate `2495.16 -> 16175.99 us`, p513 down
  `2363.23 -> 20285.69 us`.
- decision:
  reject before model A/B. The raw F32 grouped path is correct, but it is
  `6.5x-8.6x` slower than the current packed grouped route. This rules out
  direct-F32 wrapper variants as the Q6 ID path to Vulkan parity; the next Q6
  ID attempt needs a true Vulkan-medium grouped schedule or lower-level
  cooperative load/WMMA/store primitive.

## 2026-06-19 - Q8_0 current-head commit-aligned parity check

- source:
  `sources/llama.cpp` clean at `ebb85c542`. HRX and Vulkan build metadata both
  report `ebb85c542` after rebuilding `llama-bench` and `test-backend-ops`
  through CMake/Ninja in `build/hrx-v1-catalog-gfx1151` and
  `build/vulkan-gfx1151`.
- model/shape:
  Llama 3.1 8B Q8_0, p33, p512, and p513, `--flash-attn 1`, three
  repetitions, no warmup.
- route or candidate:
  current default Q8_0 gfx1151 policy. p33 selects
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32`, p512 selects
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32`, and p513
  selects `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32`.
- baseline command:
  `tools/hrxv1_basket_benchmark.py --models <llama3.1-8b-q8_0> --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --flash-attn 1`.
- variant command:
  none; this is a current-head parity/ranking check.
- route trace:
  `cache/hrxv1/gfx1151/q8_0-current-head-commitaligned-r3-20260619-004524/`.
- profile/timing:
  same model artifact, plus focused backend-op artifact
  `cache/hrxv1/gfx1151/q8_0-current-focused-eb85c542-20260619-004731/`.
- correctness:
  HRX model runs report zero fallback lines. Focused p512 and p513
  CPU-reference gates passed for the exported Q8_0 prompt rows.
- timing:
  model steady ratios were p33 `204.287 / 233.279 = 0.876x` Vulkan, p512
  `457.849 / 921.731 = 0.497x`, and p513
  `419.430 / 812.311 = 0.516x`. Focused p512 current-route timings were
  `Vcur 558.792 us`, `Qcur 1906.867 us`, `ffn_out 7137.794 us`,
  `ffn_gate 6449.436 us`, and exported batched `result_output 56659.810 us`.
  Focused p513 timings were `Vcur 547.193 us`, `Qcur 2148.921 us`,
  `ffn_out 7735.595 us`, `ffn_gate 7055.443 us`, and exported batched
  `result_output 63606.143 us`.
- decision:
  keep Q8_0 p512/p513 as an active parity gap. p33 is close enough to protect,
  but p512/p513 remain about half of Vulkan. The next Q8_0 candidate must
  target production batched rows, not the exported `[128256,512]`
  `result_output` stress row.
- notes:
  The real p512 model graph dispatches `result_output` as `cols=1` in both HRX
  and Vulkan. Vulkan uses `mul_mat_vec_q8_0_f32_f32` for that final row, while
  HRX uses `hrx_mul_mat_vec_q8_0_f32`; the focused exported batched
  `result_output` row is not the production p512 prompt route. Vulkan perf
  logger samples for the K/V projection family include a suspicious
  sub-microsecond steady value after the first sample, so that row should be
  confirmed with profiler/common-runner timing before being used as an exact
  kernel target.

## 2026-06-19 - Q8_0 BN128 split-qsum wave64 default rejection

- source:
  `sources/llama.cpp` dirty after `ebb85c542`, with a new CMake/Ninja-built
  opt-in HIP C++ route added to the HRX v1 catalog.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wave64_wg256_f32`, enabled
  only with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_WAVE64_PROMPT=1`.
- prior-art schedule source:
  current accepted BN128 split-qsum packed-Q8_1 route plus the Vulkan Q8_0
  p512/p513 large oracle, which is wave64. This was a single-axis wavefront
  mode probe, not a new RADV cooperative-matrix clone.
- focused CPU-reference result:
  `cache/hrxv1/gfx1151/q8_0-wave64-default-postedit-focused-20260619-011031/`.
  p512 and exact p513 rows passed. Route traces selected wave64 BN128 for p512,
  kept p513 on BN112, and rollback restored wave32 BN128.
- odd/tail model smoke:
  `cache/hrxv1/gfx1151/q8_0-wave64-fullcols-default-p33p513-r3-20260619-010642/`
  and
  `cache/hrxv1/gfx1151/q8_0-wave64-fullcols-optin-p33p513-r3-20260619-010656/`.
  p33 stayed on BN64, p513 stayed on BN112, and fallback lines were zero.
- compile report:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x128_splitqsum_wave64.hsaco`,
  wave64, SGPR `40`, VGPR `152`, LDS `4352`, no private segment, no spills.
- focused timing:
  post-edit p512 focused timing improved versus rollback on all five rows:
  `Vcur 547.266 -> 528.474 us`, `Qcur 1925.167 -> 1888.803 us`,
  `ffn_out 7167.517 -> 7029.020 us`,
  `ffn_gate 6498.759 -> 6133.599 us`, and exported `result_output`
  `57170.548 -> 53821.143 us`.
- model A/B:
  initial opt-in pair showed a small p512 win
  `452.464 -> 460.691 tok/s`, but the post-edit same-build default/rollback
  pair rejected promotion:
  `cache/hrxv1/gfx1151/q8_0-wave64-postedit-default-p512-r3-20260619-011242/`
  reached `446.647` steady tok/s, while
  `cache/hrxv1/gfx1151/q8_0-wave64-postedit-rollback-p512-r3-20260619-011254/`
  reached `447.814` steady tok/s. Both had zero fallback lines.
- decision:
  reject default promotion and keep the route opt-in only. The final route
  check
  `cache/hrxv1/gfx1151/q8_0-wave64-final-optin-routecheck-20260619-011522/`
  proves p512 defaults back to the original wave32 BN128 split-qsum provider.
  This closes the isolated wave-mode pivot; the remaining Q8_0 work should
  target RADV's cooperative-matrix store/lane-ownership and common-runner
  Vulkan gap directly.

## 2026-06-19 - Q8_0 current Vulkan oracle refresh and ISA split

- source:
  `sources/llama.cpp` clean at `5200f0b01`; no source edits in this entry.
- model/shape:
  Llama 3.1 8B Q8_0 p512/fa1, current `build/vulkan-gfx1151/bin/llama-bench`.
- command:
  `python3 sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py --bench build/vulkan-gfx1151/bin/llama-bench --model shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --prompt 512 --gen 0 --batch 1024 --ubatch 1024 --flash-attn 1 --repetitions 1 --device Vulkan0`.
- artifact:
  `cache/hrxv1/gfx1151/vulkan-oracle-current-llama31-8b-q8_0-p512-fa1-20260619-012416/`.
- capture evidence:
  JSON reports build commit `5200f0b01`, `backends=Vulkan`, and
  `419.430 tok/s`. The capture produced `13` pipeline identities, `516`
  dispatch signatures, `13` SPIR-V files, SPIR-V asm, split RADV ISA/stats,
  and inventories.
- Vulkan schedule:
  dominant dense route remains `matmul_q8_0_f32_f16acc_aligned_l`, hash
  `0x72d309e22f889977`, with `221` dispatches. The current large-route
  contract is unchanged: spec `[256,128,128,32,64,64,2,16,16,16,64]`,
  `wg_denoms=[128,128,1]`, `LDS=22528`, `VGPR=192`, no spills, `32` f16
  WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  `192 buffer_store_b32`, and two barriers.
- comparison artifact:
  `cache/hrxv1/gfx1151/q8_0-current-radv-vs-hrx-isa-20260619-012526/`.
- compared HSACOs:
  accepted packed BN128 split-qsum, K2 directwait direct-WMMA, and K2 depwait
  direct-WMMA.
- ISA conclusion:
  accepted BN128 split-qsum is structurally not the Vulkan schedule: it is a
  wave32 packed integer-dot route with VGPR `152`, `LDS=4352`, no WMMA, no
  halfword LDS topology, and no `buffer_store_b32`. K2 directwait keeps
  pressure acceptable at VGPR `196` with no spills but collapses the first
  WMMA issue window to `24/0/lgkmcnt(0)`. K2 depwait recovers the high
  outstanding-load shape, scoring `64` pre-hot LDS loads and final
  `lgkmcnt(51)`, but hits VGPR `256` with `30` spills. Both direct-WMMA routes
  still emit only `128 buffer_store_b32` and miss RADV's halfword
  LDS/load/store topology.
- decision:
  do not start another Q8_0 BN/BM/split-qsum/wait-decoration probe. The
  current evidence says source-visible HIP C++ has split the target into two
  losing halves: good pressure with the wrong issue window, or a closer issue
  window with a register cliff. The next Q8 source change needs a lower-level
  load/WMMA/writeback primitive or a materially different dataflow; otherwise
  pivot to another current basket boulder with an unexhausted axis.

## 2026-06-19 - Current basket perf-rank target selection

- source/tooling:
  added source-side analyzer
  `sources/llama.cpp/tools/vulkan-oracle/analyze_basket_perf.py`.
  It parses a basket artifact's `summary.json`, `records.json`, Vulkan
  perf-logger stderr rows, and HRX route traces, then writes `perf-rank.json`
  and `perf-rank.md`.
- input artifact:
  `cache/hrxv1/gfx1151/current-head-p512-r3-20260619-011833/`.
- output artifact:
  `cache/hrxv1/gfx1151/current-head-p512-r3-20260619-011833/perf-rank.md`
  and `perf-rank.json`.
- current p512 ranking:
  worst row remains Llama 3.1 8B Q8_0 at `0.490x` Vulkan; Vulkan measured
  `dense/q8_0` at `1593.75 ms` summed over the repeated perf rows, and HRX
  selected `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32`.
  This is still the largest dense non-MoE bucket, but the Q8 schedule ledger
  now parks local source-visible HIP pivots until a lower-level
  load/WMMA/writeback primitive is available.
- Qwen3 MoE ranking:
  Qwen3 30B Q4_K_XL p512 is `0.511x` Vulkan. Vulkan's largest measured
  family is not dense Q4; it is `moe_id/q4_K` at `770.02 ms`, followed by
  dense Q4 at `144.39 ms` and `moe_id/q5_K` at `88.18 ms`. Qwen3 30B Q6_K
  p512 is `0.554x` Vulkan, with `moe_id/q6_K` at `1070.35 ms` and dense Q6
  at `185.31 ms`.
- non-MoE dense ranking:
  Qwen2.5 Coder 7B Q5_K_M p512 is `0.518x` Vulkan with Vulkan
  `dense/q5_K` at `970.08 ms`; Llama 3.1 8B Q4_K_M p512 is `0.518x` with
  Vulkan `dense/q4_K` at `988.39 ms`; DeepSeek 14B Q4_K_M p512 is `0.597x`
  but has the largest absolute dense Q4 time at `1867.64 ms`.
- decision:
  keep Q8/Q5/Q4 dense prompt kernels in the parity ledger, but do not repeat
  the already rejected HIP C++ visible axes. The next implementation target
  should be a Qwen3 `MUL_MAT_ID` schedule, starting from the Vulkan
  `matmul_id_subgroup_q*_k_f32_f16acc_aligned_m/l` oracle and the existing
  HRX grouped Q8_1 x4 ID routes. That target is now evidence-backed by
  measured Vulkan bucket time rather than route count alone, and it can still
  be gated with exported backend-op correctness, route traces, odd/tail rows,
  and same-runner model A/B.

## 2026-06-19 - Q4_K MoE SWIGLU route trace coverage

- source:
  `sources/llama.cpp` dirty after `e9b3a53a9`, with a trace-only change in
  `ggml_backend_hrx_dispatch_mul_mat_id_q4_k_swiglu`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; rebuilt `llama-bench` and
  `test-backend-ops` through CMake/Ninja. No HIP kernel source was changed.
- model/shape:
  Qwen3 30B Q4_K_XL p33, p512, and p513, `--flash-attn 1`, one repetition,
  `-b 1024 -ub 1024`.
- route or candidate:
  observability hook for existing `MUL_MAT_ID_SWIGLU` providers. This is not a
  promoted schedule change.
- baseline command:
  current HRX route trace with `GGML_HRX_TRACE_ROUTES=1` and
  `GGML_HRX_TRACE_PROVIDERS=1`.
- variant command:
  same command after adding `HRX route MUL_MAT_ID_SWIGLU ...` trace rows for
  grouped and non-grouped SWIGLU dispatches.
- route trace:
  `cache/hrxv1/gfx1151/q4xl-swiglu-route-trace-20260619-013755/` for p512 and
  `cache/hrxv1/gfx1151/q4xl-swiglu-route-trace-odd-tail-20260619-013827/` for
  p33/p513.
- profile/timing:
  route-only `llama-bench` JSON files in the same artifacts; p512 reports
  `657.644 tok/s`, p33 `163.744 tok/s`, and p513 `644.183 tok/s` for this
  one-repetition trace run.
- correctness:
  no compute code changed. The HRX model runs reported `backends=HRX`, build
  commit `e9b3a53a9`, and zero fallback lines.
- timing:
  not used for promotion. The useful result is route accounting:
  p512 and p513 both show `47` fused
  `hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_bn16_wg64_f32` dispatches. The
  p512 SWIGLU shape is `k=2048`, `rows=768`, `n_ids=8`, `n_tokens=512`,
  `route_capacity=4096`, `wg_count=[48,32,128]`; p513 is the same with
  `n_tokens=513`, `route_capacity=4104`, `wg_count=[48,33,128]`.
- decision:
  keep as source instrumentation and commit it. The trace closes a missing
  route-evidence gap for Qwen3/Qwen3-Coder Q4 MoE schedule work.
- notes:
  p33 did not emit `MUL_MAT_ID_SWIGLU`; it emitted separate grouped Q4
  `MUL_MAT_ID` gate/up/down routes with `n_tokens=33`, including gate/up
  `wg_count=[12,3,128]` and down `wg_count=[32,3,128]`. The selector supports
  grouped SWIGLU at this prompt size, so the narrow-row issue appears to be a
  graph/fusion-shape or node-ordering problem before it is a kernel schedule
  problem. Future p33 MoE work should first explain why the graph fails the
  adjacent `MUL_MAT_ID, MUL_MAT_ID, GLU` fusion pattern.

## 2026-06-19 - Q4_K MoE narrow SWIGLU scheduler fusion promotion

- source:
  `sources/llama.cpp` after `361bf8cd1`, dirty with a scheduler support-policy
  change for narrow Q4_K `MUL_MAT_ID_SWIGLU`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; rebuilt `llama-bench` and
  `test-backend-ops` through CMake/Ninja. No HIP kernel source was changed.
- model/shape:
  Qwen3 30B Q4_K_XL and Qwen3-Coder 30B Q4_K_M p33, plus Qwen3 Q4_K_XL
  p512/p513 route smoke.
- route or candidate:
  default scheduler support for the existing fused provider
  `hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_bn16_wg64_f32` when the GLU node
  is a narrow MoE SWIGLU produced by supported Q4_K `MUL_MAT_ID` gate/up nodes.
  Rollback:
  `GGML_HRX_DISABLE_Q4_K_ID_SWIGLU_NARROW_PROMPT_FUSION=1`.
- baseline command:
  `llama-bench -p 33 -n 0 -b 1024 -ub 1024 -fa 1 -r 3 -o json --no-warmup
  -ngl 99 -dev HRX0` with the rollback env.
- variant command:
  same command with the new default policy, and earlier opt-in validation with
  `GGML_HRX_ENABLE_Q4_K_ID_SWIGLU_NARROW_PROMPT_FUSION=1`.
- route trace:
  scheduler and graph evidence:
  `cache/hrxv1/gfx1151/q4xl-p33-sched-trace-20260619-014237/`,
  `cache/hrxv1/gfx1151/q4xl-p512-sched-trace-20260619-014310/`, and
  `cache/hrxv1/gfx1151/q4xl-p33-swiglu-narrow-optin-route-20260619-014538/`.
  Post-edit default/rollback:
  `cache/hrxv1/gfx1151/q4-moe-p33-swiglu-narrow-default-postedit-20260619-014747/`.
  Wide/tail smoke:
  `cache/hrxv1/gfx1151/q4xl-swiglu-narrow-default-wide-tail-smoke-20260619-014825/`.
- profile/timing:
  model A/B artifacts above plus opt-in artifacts
  `cache/hrxv1/gfx1151/q4xl-p33-swiglu-narrow-optin-model-ab-20260619-014617/`
  and
  `cache/hrxv1/gfx1151/qwen3coder-q4-p33-swiglu-narrow-optin-model-ab-20260619-014656/`.
- correctness:
  No compute kernel changed; the existing fused SWIGLU provider was already
  production-selected for p512/p513. Post-edit runs report `backends=HRX` and
  zero fallback/unsupported lines. Route traces prove the rollback path emits
  zero `MUL_MAT_ID_SWIGLU` routes and default emits the fused route.
- timing:
  Qwen3 Q4_K_XL p33 post-edit default/rollback:
  `198.246 / 160.342 = 1.236x`. Qwen3-Coder Q4_K_M p33:
  `194.831 / 152.243 = 1.280x`. Earlier opt-in A/B measured
  `201.294 / 163.707 = 1.230x` and `195.199 / 149.601 = 1.305x`.
  Default route counts were `144` fused SWIGLU dispatches across three reps.
- decision:
  promote as gfx1151 default with rollback env. This removes a p33 graph
  placement miss: the scheduler previously assigned `ffn_moe_swiglu` to CPU
  for `n_tokens=33`, forcing HRX->CPU copies of gate/up and CPU->HRX copy of
  SWIGLU before down-projection. The new support exception keeps the exact
  supported gate/up/GLU pattern on HRX so the existing fused dispatcher can
  fire.
- notes:
  p512/p513 route smoke remains unchanged: Qwen3 Q4_K_XL selects
  `hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_bn16_wg64_f32` with
  `wg_count=[48,32,128]` at p512 and `[48,33,128]` at p513. This is a
  scheduler/fusion fix, not a new kernel schedule. The remaining MoE parity
  gap still requires schedule/resource work against the Vulkan
  `matmul_id_subgroup_q4_k_f32_f16acc_aligned_m` oracle.

## 2026-06-19 - Q5_K VK64 narrow fullstore diagnostic rejection

- source:
  `sources/llama.cpp` after `6cc94290c`, dirty with a new opt-in
  CMake-built Q5_K VK64 fullstore wrapper plus Vulkan basket analyzer updates.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; rebuilt `llama-bench` and
  `test-backend-ops` through CMake/Ninja.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused rows, plus p512/p513 non-steal gates.
- route or candidate:
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_fullstore_f16acc_wg256_f32`,
  gated by
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_FULLSTORE_F16ACC_WG256_NARROW_PROMPT=1`.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file
  q5_prompt_p33.txt --output csv` with the current default VK64 narrow FFN
  route.
- variant command:
  same focused command with the fullstore opt-in env above.
- route trace:
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-narrow-focused-20260619-020742/`,
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-default-postedit-20260619-021200/`,
  and final opt-in smoke
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-final-optin-smoke-20260619-021442/`.
- profile/timing:
  focused artifacts above plus model A/B artifacts
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-model-default-r7-20260619-020954/`
  and
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-model-optin-r7-20260619-020957/`.
- correctness:
  CPU-reference passed for p33, p512, and p513. Route traces show default
  remains on `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`,
  while opt-in selects the fullstore provider only for the exact
  `k=3584`, `rows=18944`, `cols=33` FFN gate/up slice. p512/p513 stay on
  rows2 plus MMQL128 routes.
- timing:
  initial focused p33 improved `ffn_gate-0` from `1797.31 -> 1758.83 us`, and
  r7 model A/B improved Qwen2.5 Coder Q5_K_M p33 steady tok/s
  `169.717 -> 171.064`. Post-edit focused timing was mixed: one pass regressed
  `ffn_gate-0` `1827.65 -> 1864.51 us`, while a rerun improved it
  `1821.96 -> 1810.51 us`.
- decision:
  reject for default promotion; keep as opt-in diagnostic. The model-level
  signal is positive but the focused kernel signal is not stable enough for a
  default route.
- notes:
  Static HSACO facts: wave64, SGPR `32`, VGPR `75`, LDS `11264`, no private
  segment/spills. This is a writeback-axis probe around the accepted VK64
  schedule. The remaining Q5 p33 gap still needs a more mechanical RADV
  cooperative store/lane-ownership clone rather than another local default
  policy tweak.

## 2026-06-19 - Q5_K VK64 GROUPK2 diagnostic rejection

- source:
  `sources/llama.cpp` dirty after adding opt-in CMake-built Q5_K VK64
  GROUPK2 and GROUPK2_WAIT wrappers.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`; rebuilt `llama-bench` and
  `test-backend-ops` through CMake/Ninja.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused rows.
- route or candidate:
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_groupk2_f16acc_wg256_f32`
  and
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_groupk2_wait_f16acc_wg256_f32`.
- baseline command:
  focused `test-backend-ops test -b HRX0 -o MUL_MAT --test-file
  q5_prompt_p33.txt --output csv` with current default routing.
- variant command:
  same focused command with either
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_GROUPK2_F16ACC_WG256_PROMPT=1`
  or
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_GROUPK2_WAIT_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-vk64-groupk2-focused-20260619-023220/`.
- profile/timing:
  no timing run; rejected before perf on static schedule and CPU-reference
  correctness. ISA artifact:
  `cache/hrxv1/gfx1151/q5-vk64-groupk2-isa-20260619-023058/`.
- correctness:
  default p33 passed all rows. GROUPK2 selected on all four rows and failed
  all four, including a NaN on ffn_out. GROUPK2_WAIT selected on all four rows
  and passed Kcur/Qcur, but failed ffn_out and ffn_gate with finite errors.
- timing:
  not run.
- decision:
  reject before timing/model tests; keep both as opt-in diagnostics only.
- notes:
  Static ISA forced `40 ds_load_b64`, but still missed the RADV medium p33
  schedule: `8` visible WMMA sites vs RADV `16`, pre-WMMA b64 loads `24` vs
  `48`, final pre-WMMA `lgkmcnt(0)` vs `40`, no `ds_load_u16_d16`, no
  `buffer_store_b32`, and only `16` global stores. This closes the Q6-style
  GROUPK2 transplant for Q5; the next direct-WMMA attempt needs a lower-level
  cooperative load/store/lane-ownership fixture or a source form that can
  actually emit the RADV issue and writeback topology.

## 2026-06-19 - Q5_K p33 medium lower-level fixture split

- source:
  `sources/llama.cpp` dirty after adding CMake-built diagnostic modes to
  `hrx-hip-bench-coopmat-store-contract` and
  `hrx-hip-bench-wmma-issue-window`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract hrx-hip-bench-wmma-issue-window
  -j"$(nproc)"`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 RADV medium oracle
  `matmul_q5_k_f32_f16acc_aligned_m`.
- route or candidate:
  no production route; diagnostic fixture modes
  `--mode=radv-mixed96` and `--mode=mediumfrag12`.
- baseline command:
  RADV oracle artifact
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-exact-20260618-195327/`.
- variant command:
  run both HIP bench modes, extract embedded gfx1151 objects with
  `llvm-objdump --offloading`, then compare via
  `sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py`.
- route trace:
  n/a; these are standalone HIP bench fixtures, not catalog providers.
- profile/timing:
  no timing; this is a compile-contract probe. Artifact:
  `cache/hrxv1/gfx1151/q5-p33-fixture-medium-contract-20260619-024558/`.
- correctness:
  `radv-mixed96` output check passed with `bad=0`; `mediumfrag12` finite-output
  smoke passed with `nan=0`.
- timing:
  not run.
- decision:
  accept as diagnostic infrastructure only. The store fixture matches the p33
  RADV store surface (`64 ds_load_u16_d16`, `64 ds_store_b16`, `96
  buffer_store_b32`, `2` barriers). The issue-window fixture matches the p33
  RADV load/WMMA surface (`16` WMMA, `48 ds_load_b64`, `48` first-window
  loads, final pre-WMMA `lgkmcnt(40)`). They are not production routes and do
  not change parity.
- notes:
  This narrows the direct-WMMA gap: HIP can emit both halves separately, but
  the production Q5 route still needs a combined arithmetic/lane-mapping
  reproducer that carries the RADV cooperative-matrix load/store topology in
  one kernel before another catalog candidate is justified.

## 2026-06-19 - Q5_K p33 medium combined fixture

- source:
  `sources/llama.cpp` dirty after adding
  `hrx-hip-bench-wmma-issue-window --mode=mediumfrag12-combined96`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-issue-window -j"$(nproc)"`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 RADV medium oracle
  `matmul_q5_k_f32_f16acc_aligned_m`.
- route or candidate:
  no production route; standalone HIP bench fixture combining the p33
  issue-window and writeback contracts.
- baseline command:
  RADV oracle artifact
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-exact-20260618-195327/`.
- variant command:
  run `hrx-hip-bench-wmma-issue-window --mode=mediumfrag12-combined96`,
  extract embedded gfx1151 object with `llvm-objdump --offloading`, then
  compare via `sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py`.
- route trace:
  n/a; standalone fixture.
- profile/timing:
  no timing; compile-contract probe. Artifact:
  `cache/hrxv1/gfx1151/q5-p33-combined96-padded-fixture-20260619-025352/`.
- correctness:
  finite-output smoke passed with `nan=0`.
- timing:
  not run.
- decision:
  accept as diagnostic infrastructure and the current p33 direct-WMMA
  reproducer. It matches RADV p33 on LDS bytes, WMMA count, `ds_load_b64`,
  `ds_load_u16_d16`, `ds_store_b16`, `buffer_store_b32`, first-window load
  count, final pre-WMMA `lgkmcnt`, and hot-op window count. It still has
  `3` barriers versus RADV `2` and does not consume real Q5/Q8 data.
- notes:
  This is the first HIP C++/inline-asm artifact in this spike that carries the
  p33 RADV medium schedule facts together in one kernel. The next route work
  should transplant this skeleton into a tightly gated Q5 p33 provider and let
  focused CPU-reference determine whether accumulator lane ownership is now
  correct.

## 2026-06-19 - Q5_K p33 combined96 catalog probe

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_combined96_f16acc_wg256_f32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j"$(nproc)"`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused backend-op rows.
- route or candidate:
  p33-only opt-in Q5_K direct-F32 WMMA provider,
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_COMBINED96_F16ACC_WG256_PROMPT=1`.
- baseline command:
  RADV oracle artifact
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-exact-20260618-195327/`.
- variant command:
  `test-backend-ops test -b HRX0 -o MUL_MAT --output csv --test-file
  cache/hrxv1/gfx1151/q5-vk64-narrow-default-regate-20260618-210315/q5_prompt_p33.txt`.
- route trace:
  `cache/hrxv1/gfx1151/q5-p33-combined96-catalog-20260619-030619/correct_p33_combined96.stderr.log`.
- profile/timing:
  not run; correctness failed.
- correctness:
  failed all four p33 rows with NaNs after route traces confirmed combined96
  selected.
- timing:
  skipped.
- decision:
  reject before timing/model tests. The built HSACO matched RADV on LDS size,
  WMMA count, `ds_load_b64`, `ds_load_u16_d16`, and `buffer_store_b32`, but
  it emitted `66 ds_store_b16`, `4` barriers, only `20` pre-WMMA `ds_load_b64`
  in the scored first window, and final pre-WMMA `lgkmcnt(0)` instead of
  RADV's `48` and `40`.
- notes:
  This is useful negative evidence: the fixture schedule does not survive a
  straightforward transplant into the real Q5 ABI. Continue mechanically from
  RADV, but the next step should isolate dependency/lane ownership rather than
  benchmark this provider.

## 2026-06-19 - Q5_K p33 combined96 store-contract diagnostic

- source:
  `sources/llama.cpp` dirty after adding
  `hrx-hip-bench-wmma-f16-lane-map --mode=combined96-store-contract`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-f16-lane-map -j"$(nproc)"`.
- model/shape:
  synthetic `64 x {32,33,64}` medium-tile writeback cases for the Q5_K p33
  combined96 store pattern.
- route or candidate:
  no production route; standalone CMake-built HIP bench fixture that mimics
  combined96's `24` logical writeback groups from eight accumulator vectors,
  including the raw groups `0..7` and staged groups `8..23`.
- baseline command:
  n/a; this isolates the candidate's own store contract after the catalog
  p33 CPU-reference gate failed with NaNs.
- variant command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-wmma-f16-lane-map
  --mode=combined96-store-contract`.
- route trace:
  n/a; standalone fixture.
- profile/timing:
  no timing. Artifact:
  `cache/hrxv1/gfx1151/q5-combined96-store-contract-20260619-031500/`.
- correctness:
  diagnostic completed without NaNs, but reported `contract_valid=0` for all
  tested column widths. At `cols=33`, it wrote `2112` coordinates with `2050`
  duplicate-coordinate writes and `64` writes into the second-half column
  region.
- timing:
  not run.
- decision:
  reject the combined96 catalog direction as structurally invalid, not merely
  mistuned. Do not continue tuning this route's issue window until output
  ownership is redesigned.
- notes:
  The failure comes from using `group & 7` to source `24` writeback groups from
  eight accumulator vectors. Groups `16..23` alias earlier coordinates, and the
  p33 stage path writes column `32` from accumulators that only represent the
  first two 16-column output tiles. The next p33 candidate must either compute
  the full medium `64x64` ownership set or narrow the declared output tile
  contract before schedule matching is meaningful.

## 2026-06-19 - Q5_K p33 full64 ownership catalog probe

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32`
  as an opt-in CMake-built HIP C++ route.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j"$(nproc)"`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused backend-op rows.
- route or candidate:
  p33-only direct-F32 WMMA provider,
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_FULL64_F16ACC_WG256_PROMPT=1`,
  designed to compute all `16` medium output tile groups instead of aliasing
  eight accumulators through `group & 7`.
- baseline command:
  RADV oracle artifact
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-exact-20260618-195327/`.
- variant command:
  `test-backend-ops test -b HRX0 -o MUL_MAT --output csv --test-file
  cache/hrxv1/gfx1151/q5-vk64-narrow-default-regate-20260618-210315/q5_prompt_p33.txt`.
- route trace:
  `cache/hrxv1/gfx1151/q5-p33-full64-catalog-20260619-032340/correct_p33_full64.stderr.log`.
- profile/timing:
  not run; correctness failed. Static artifact:
  `cache/hrxv1/gfx1151/q5-p33-full64-catalog-20260619-032340/static/`.
- correctness:
  failed all four p33 rows with NaNs after route traces confirmed full64
  selected.
- timing:
  skipped.
- decision:
  reject before timing/model tests. This repairs the combined96 coordinate
  aliasing but is not a valid RADV clone or production candidate.
- notes:
  Static compare against RADV p33 medium showed VGPR `198` versus `144`,
  `32` WMMA versus `16`, `64 ds_load_b64` versus `48`, `64 buffer_store_b32`
  versus `96`, no `ds_load_u16_d16` store stage, `3` barriers versus `2`, and
  final pre-WMMA `lgkmcnt(0)` versus `40`. Full ownership is necessary but not
  sufficient; scaling live f16 WMMA accumulators in one HIP wave is outside the
  useful schedule family.

## 2026-06-19 - Q4_K MoE ID direct-F32 WMMA diagnostic rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_id_q4_k_wmma16x16_direct_f16acc_wg32_f32` as an opt-in
  CMake-built HIP C++ route.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j"$(nproc)"`.
- model/shape:
  Qwen3/Qwen3-Coder-style Q4_K MoE `MUL_MAT_ID` focused rows for p512 and the
  current p33/p513 exported odd-tail files. The p33/p513 export files contain
  `n_tokens=1024`, so they are valid route/tail coverage but not exact p33 or
  p513 single-graph kernel shapes.
- route or candidate:
  direct-F32 Q4_K MoE ID diagnostic,
  `GGML_HRX_ENABLE_Q4_K_ID_WMMA16_DIRECT_PROMPT=1`, using compacted expert
  routes but binding raw F32 activations instead of Q8_1 x4 packed RHS.
- baseline command:
  focused `test-backend-ops perf -b HRX0 -o MUL_MAT_ID --output csv` with
  expected provider
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- variant command:
  same focused files with
  `GGML_HRX_ENABLE_Q4_K_ID_WMMA16_DIRECT_PROMPT=1` and expected provider
  `hrx_mul_mat_id_q4_k_wmma16x16_direct_f16acc_wg32_f32`.
- route trace:
  `cache/hrxv1/gfx1151/q4-id-wmma16-direct-focused-20260619-033650/`.
- profile/timing:
  same focused artifact. Static RADV-vs-HRX comparison:
  `cache/hrxv1/gfx1151/q4-id-wmma16-direct-focused-20260619-033650/radv-vs-hrx-compare.md`.
- correctness:
  focused CPU-reference passed for p512 and for both odd-tail exported files.
  Route traces selected `direct_f32=1` with p512 grids
  `[48,32,128]` and `[128,32,128]`, and odd-tail grids `[48,64,128]` and
  `[128,64,128]`.
- timing:
  rejected versus the accepted grouped Q8_1 x4 route. p512 regressed gate
  `2007.26 -> 11632.45 us` and down `2020.09 -> 16746.50 us`. The p513-export
  file regressed gate `4028.31 -> 27770.79 us` and down
  `3883.70 -> 35241.63 us`. The p33-export file regressed gate
  `4098.52 -> 26938.39 us` and down `3918.56 -> 36080.42 us`.
- decision:
  reject for production promotion. Keep the provider opt-in only as a
  correctness/ABI bridge.
- notes:
  The diagnostic proves the MoE ID ABI, route compaction, and direct-F32 WMMA
  semantics are sound for Q4_K, but it does not reproduce RADV's winning
  schedule. RADV's Q4 ID medium pipeline uses `12288` bytes LDS, `144` VGPR,
  `16` WMMA sites, staged half writeback, and `32 buffer_store_b32`; the HIP
  route has `0` LDS, `72` VGPR, one WMMA site, and scalar global stores. The
  next Q4 MoE parity attempt should either clone RADV's cooperative
  staging/writeback more literally or improve the current packed-Q8 path, not
  replace the accepted grouped route with raw-F32 direct WMMA.

## 2026-06-19 - Q5_K p33 fullstore current-head rerun

- source:
  `sources/llama.cpp` clean at commit `69f2b1ce8` before the evidence-log
  update.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33/fa1, plus focused Q5_K p33 backend-op rows.
- route or candidate:
  opt-in `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_fullstore_f16acc_wg256_f32`
  through
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_FULLSTORE_F16ACC_WG256_NARROW_PROMPT=1`.
- artifacts:
  default model r9:
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-current-default-r9-20260619-continue/`;
  opt-in model r9:
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-current-optin-r9-20260619-continue/`;
  focused rerun:
  `cache/hrxv1/gfx1151/q5-vk64-fullstore-current-focused-rerun-20260619-035210/`.
- route trace:
  model top routes confirmed the intended switch: default used
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`
  for `486` dispatches, while opt-in used the corresponding fullstore provider
  for the same `486` dispatches. The accepted packed B-quad route remained the
  top route at `630` dispatches.
- timing:
  model r9 stayed slightly positive: `170.072 -> 171.744` steady tok/s.
  Focused backend-op timing was not durable: Kcur `71.865 -> 74.072 us`, Qcur
  `259.392 -> 259.291 us`, ffn_out `1537.784 -> 1532.410 us`, and the target
  ffn_gate row `1716.238 -> 1717.690 us`.
- decision:
  keep rejected for default promotion. The small model lift is not enough to
  override flat focused evidence on the exact target row.
- notes:
  This closes the simple full-tile scalar writeback axis for the accepted VK64
  narrow route. The remaining Q5 p33 gap still points at RADV's cooperative
  staging/writeback and lane ownership, not another selector broadening of this
  fullstore variant.

## 2026-06-19 - Q5_K p33 full64 store-contract fixture

- source:
  `sources/llama.cpp` dirty after adding
  `hrx-hip-bench-wmma-f16-lane-map --mode=full64-store-contract`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-f16-lane-map -j"$(nproc)"`.
- model/shape:
  synthetic `64 x {32,33,64}` medium-tile writeback cases for the Q5_K p33
  full64 store pattern.
- route or candidate:
  diagnostic fixture for
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32`
  output-coordinate ownership.
- artifact:
  `cache/hrxv1/gfx1151/q5-full64-store-contract-20260619-035618/`.
- correctness:
  all tested widths passed the store contract:
  `cols=32` wrote `2048/2048`, `cols=33` wrote `2112/2112`, and `cols=64`
  wrote `4096/4096`, with zero duplicate coordinates and zero NaNs.
- decision:
  keep the full64 catalog route rejected. This fixture narrows the reason: its
  focused NaNs are not caused by missing or duplicate output coordinates.
- notes:
  The next Q5 p33 direct-WMMA step should isolate the real fragment/lane-value
  dependency path while preserving full medium-tile ownership, instead of
  adding more store-count or selector variants.

## 2026-06-19 - Q5_K p33 LDS WMMA fixture narrowing

- source:
  `sources/llama.cpp` dirty after extending
  `hrx-hip-bench-wmma-f16-lane-map` with LDS-fed fulltile, two-K, repeated-K,
  and production stride44 diagnostics.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-f16-lane-map -j"$(nproc)"`.
- model/shape:
  synthetic WMMA F16 accumulator fixtures that isolate the Q5_K p33 full64
  failure without invoking full llama.cpp graph execution.
- route or candidate:
  no production route; standalone CMake-built HIP bench modes
  `fulltile-lds-wait`, `fulltile-lds-k2`, `fulltile-lds-k2-wait`,
  `fulltile-lds-repeat`, `fulltile-lds-repeat-wait`,
  `fulltile-lds-repeat-one`, `fulltile-prodstride-k2`, and
  `fulltile-prodstride-k2-small`.
- artifacts:
  `cache/hrxv1/gfx1151/wmma-lds-wait-contract-20260619-035949/`,
  `cache/hrxv1/gfx1151/wmma-lds-k2-contract-20260619-040138/`,
  `cache/hrxv1/gfx1151/wmma-lds-repeat-scale-20260619-040346/`, and
  `cache/hrxv1/gfx1151/wmma-prodstride-contract-20260619-040530/`;
  current validation sweep:
  `cache/hrxv1/gfx1151/wmma-lds-diagnostics-current-20260619-041040/`.
- correctness:
  one-K, two-K, and production stride44 LDS-fed fixtures produced NaNs only in
  inactive odd accumulator slots. Per-load `lgkmcnt(0)` waits did not clear
  those inactive-slot NaNs. The selected even output slots stayed finite in
  the generic, long-repeat, and production-stride fixtures.
- timing:
  not run; this is a correctness/localization fixture.
- decision:
  keep the Q5 full64 catalog route rejected. The real route's selected-output
  NaNs are not explained by output-coordinate ownership, simple LDS wait
  placement, or stride44 fragment loads alone.
- notes:
  The next useful diagnostic should include the exact Q5 dequant/shared-layout
  source context and register-pressure shape, or move to a lower-level
  cooperative-matrix spelling closer to RADV. Another store-count or selector
  variant is unlikely to move the mission toward Vulkan parity.

## 2026-06-19 - Q5_K p33 exact full64 kernel reproducer

- source:
  `sources/llama.cpp` dirty after adding the CMake-built wave64 bench
  `hrx-hip-bench-q5-wmma-full64-repro`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- model/shape:
  synthetic Q5_K blocks and F32 RHS launched through the actual
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_full64_f16acc_wg256_f32`
  catalog kernel.
- route or candidate:
  standalone exact-kernel reproducer for the rejected full64 route; no
  production dispatch changes.
- artifact:
  `cache/hrxv1/gfx1151/q5-full64-exact-repro-profiles-20260619-041650/`.
- correctness:
  reproduced selected-output NaNs. The small-scale profile reported p33 NaNs
  with no Infs for `rows=64, cols=33, k=256`, `rows=64, cols=33, k=512`, and
  `rows=64, cols=33, k=3584`; the stress profile also reported NaNs/Infs for
  the same p33 cases plus p64 and rows128 coverage.
- timing:
  not run; this is a correctness/localization fixture.
- decision:
  keep the full64 catalog route rejected. The exact Q5 dequant/shared-layout
  source context plus the full64 live-accumulator topology is sufficient to
  reproduce selected-output NaNs without requiring stress-scale overflow.
- notes:
  This closes the gap left by the synthetic LDS fixtures. The next diagnostic
  should use this exact bench to bracket the live-accumulator/dequant
  interaction, for example by reducing the full64 accumulator set, changing
  fragment lifetime, or comparing a lower-level cooperative-matrix spelling
  against RADV's medium route.

## 2026-06-19 - Q5_K p33 full64 active-group bracket

- source:
  `sources/llama.cpp` dirty after extending
  `hrx-hip-bench-q5-wmma-full64-repro` with reduced-active-group diagnostic
  kernels.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- model/shape:
  synthetic Q5_K blocks and F32 RHS on the exact p33 medium shape
  `rows=64, cols=33, k=3584`, small-scale profile.
- route or candidate:
  standalone diagnostic only. The variants keep the same Q5 dequant, stride44
  LDS staging, b64 fragment loads, and full64 store contract while varying
  active output groups.
- artifact:
  `cache/hrxv1/gfx1151/q5-full64-active-groups-repro-20260619-042113/`.
- correctness:
  `active1` and `active4` were finite. `active8` stayed finite but had a large
  mismatch. `active12` and `active16` reproduced selected-output NaNs.
- timing:
  not run; this is a correctness/localization fixture.
- decision:
  keep the full64 catalog route rejected. The failure is tied to the exact Q5
  source context plus high live-accumulator topology, not store-coordinate
  ownership.
- notes:
  The next Q5 p33 candidate should stop scaling one HIP wave to 12+ live f16
  WMMA accumulators and should instead move toward RADV's medium route: fewer
  WMMAs/live outputs, staged halfword writeback, and cooperative-matrix lane
  ownership.

## 2026-06-19 - Q5_K p33 batched4 full-output diagnostic

- source:
  `sources/llama.cpp` dirty after adding a `batched4` variant to
  `hrx-hip-bench-q5-wmma-full64-repro`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- model/shape:
  synthetic Q5_K blocks and F32 RHS on p33/p64 medium-tile shapes,
  small-scale profile.
- route or candidate:
  standalone diagnostic only. It computes all sixteen output groups four at a
  time, keeping live f16 WMMA accumulators under the active-group failure
  threshold while preserving Q5 dequant and stride44 LDS staging.
- artifact:
  `cache/hrxv1/gfx1151/q5-full64-batched4-repro-20260619-042415/`.
- correctness:
  p33 `rows=64, cols=33, k=3584` produced no NaNs/Infs and matched the finite
  error scale of `active4`. The p64 case was also finite but had a larger
  mismatch, so this is not a production route.
- timing:
  not run; the diagnostic reloads/stages too much to be a promotion candidate
  as written.
- decision:
  use batched4 as positive direction evidence. Full p33 output ownership can
  be finite when the live accumulator set stays small.
- notes:
  The next production candidate should preserve this low live-accumulator
  property without the obvious reload waste, or should follow RADV's staged
  cooperative writeback more literally.

## 2026-06-19 - Q5_K p33 batched4 catalog route

- source:
  `sources/llama.cpp` adds the opt-in catalog route
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_batched4_f16acc_wg256_f32`
  and builds its HSACO through CMake/Ninja.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j"$(nproc)"`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused `MUL_MAT` rows from
  `cache/hrxv1/gfx1151/q5-vk64-narrow-default-regate-20260618-210315/q5_prompt_p33.txt`.
- route or candidate:
  p33-only, gfx1151-only, opt-in low-live-accumulator route gated by
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_BATCHED4_F16ACC_WG256_PROMPT=1`.
- artifacts:
  correctness:
  `cache/hrxv1/gfx1151/q5-p33-batched4-catalog-20260619-043426/`;
  focused perf:
  `cache/hrxv1/gfx1151/q5-p33-batched4-focused-perf-20260619-043508/`.
- correctness:
  passed all four focused p33 rows. Route traces selected the batched4 provider
  for Kcur, Qcur, ffn_out, and ffn_gate.
- timing:
  rejected. Same-runner focused timing versus default rows2/cols8:
  Kcur `72.754 -> 1386.552 us`, Qcur `259.498 -> 1771.570 us`,
  ffn_out `1535.251 -> 12762.069 us`, and ffn_gate
  `1715.737 -> 8246.900 us`.
- decision:
  reject before model tests. Low live accumulator count fixes the full64 NaN
  failure, but the naive four-pass implementation is reload/barrier bound.
- notes:
  The next candidate needs to keep the low-live property while avoiding four
  complete A/B staging passes, or move more directly to the RADV cooperative
  staging/writeback schedule.

## 2026-06-19 - Q5_K p33 wave4row catalog route

- source:
  `sources/llama.cpp` dirty after adding the opt-in catalog route
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_wave4row_f16acc_wg256_f32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j"$(nproc)"`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused `MUL_MAT` rows from
  `cache/hrxv1/gfx1151/q5-vk64-narrow-default-regate-20260618-210315/q5_prompt_p33.txt`.
- route or candidate:
  p33-only, gfx1151-only, opt-in low-live all-wave route gated by
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_WAVE4ROW_F16ACC_WG256_PROMPT=1`.
  Each wave owns one 16-row stripe and four column groups.
- artifacts:
  nowait failure:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-catalog-20260619-044305/`;
  waited correctness:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-wait-catalog-20260619-044416/`;
  focused perf:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-focused-perf-20260619-044505/`.
- correctness:
  nowait B64 LDS fragment loads failed. Waited B64 fragment loads passed all
  four focused p33 rows and route traces selected wave4row for every row.
- timing:
  rejected. Same-runner focused timing versus default rows2/cols8:
  Kcur `72.210 -> 398.540 us`, Qcur `259.574 -> 474.473 us`,
  ffn_out `1538.760 -> 3406.255 us`, and ffn_gate
  `1721.514 -> 2380.085 us`.
- decision:
  reject before model tests. This is a useful structural probe but not a
  production route.
- notes:
  This is the closest direct-F32 Q5 p33 WMMA schedule so far: all four waves
  participate, live accumulator pressure is low, A/B are staged once per K
  tile, and the selected export is wave64 with VGPR 102, LDS 11264, no spills,
  40 `ds_load_b64`, 8 visible WMMA, and 2 barriers. It still loses to the
  scalar rows2/cols8 path, so the next p33 candidate should target RADV's
  cooperative store/fragment contract or a packed-Q8_1 medium route instead
  of another direct-F32 WMMA ownership variant.

## 2026-06-19 - Q5_K p33 FFN gate packed-route policy promotion

- source:
  `sources/llama.cpp` commit
  `8d1283334 hrx: route q5 p33 ffn gate to bquad`.
- change:
  the exact Qwen2.5 Coder Q5_K_M p33 narrow FFN gate direct-WMMA route is no
  longer a gfx1151 default. It now requires
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_F16ACC_WG256_NARROW_PROMPT=1`.
  Default routing falls through to the existing packed Q8_1/x4
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32` route.
- artifacts:
  selector A/B:
  `cache/hrxv1/gfx1151/q5-p33-disable-vk64-narrow-probe-20260619-045124/`;
  model A/B:
  `cache/hrxv1/gfx1151/q5-p33-packed-ffngate-policy-ab-20260619-045226/`;
  post-patch verification:
  `cache/hrxv1/gfx1151/q5-p33-bquad-ffngate-default-verify-20260619-045515/`.
- correctness:
  focused p33, p512, and p513 `MUL_MAT` CPU-reference gates passed after the
  policy change. Route traces show p33 ffn_gate now selects BQUAD, p512 stays
  on MMQL128, p513 stays on MMQL128 BQUAD tail, and the old VK64 narrow route
  still selects under its new opt-in.
- timing:
  same-runner focused p33 ffn_gate improved from `1711.823 us` on VK64 narrow
  to `1156.896 us` on BQUAD. Kcur/Qcur/ffn_out were flat. Same-binary
  Qwen2.5 Coder 7B Q5_K_M p33/fa1/r5 improved from `170.239 tok/s` to
  `209.917 tok/s`; post-patch no-env verification was `209.879 tok/s`.
- remaining gap:
  focused post-patch HRX versus Vulkan is now Kcur `0.729x` Vulkan time
  (HRX faster), Qcur `1.508x`, ffn_out `1.433x`, and ffn_gate `1.682x`.
  The four-row focused sum is `1.487x` Vulkan. Next Q5 p33 work should not
  return to direct-F32 WMMA ownership reshuffles unless it can reproduce RADV's
  cooperative lane/store contract; the stronger immediate branch is a packed
  Q8_1 medium-schedule axis that attacks Qcur, ffn_out, and ffn_gate together.

## 2026-06-19 - Q5_K p33 MMQL64 BK2 B-pair rejection

- source:
  `sources/llama.cpp` dirty after adding the opt-in catalog route
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_bpair_wg256_f32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  llama-bench -j"$(nproc)"`, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused `MUL_MAT` rows, plus p512/p513
  non-steal guards from
  `cache/hrxv1/gfx1151/q5-vk64-narrow-default-regate-20260618-210315/`.
- route or candidate:
  packed Q8_1/x4 MMQL64 BK2 B-pair route. It preserves the accepted
  BM64/BN64/BK_STEP=2/WG256/wave64 dataflow and changes only the B-cache live
  window from the accepted full BQUAD window to one TN-wide pair per WNITER.
- baseline command:
  focused p33 `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file
  q5_prompt_p33.txt --output csv` with current default BQUAD routing.
- variant command:
  same command with
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL64_BK2_BPAIR_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-bpair-focused-20260619-050557/`.
- profile/timing:
  same artifact; includes two focused perf A/B rounds plus B-pair and BQUAD
  HSACO notes/ISA dumps.
- correctness:
  p33, p512, and p513 CPU-reference gates passed. Route traces show p33
  selects B-pair for Qcur, ffn_out, and ffn_gate, Kcur stays on rows2, p512
  stays on MMQL128, and p513 stays on MMQL128 BQUAD tail.
- timing:
  rejected. Repeat focused A/B showed p33 four-row sum regressions:
  `3071.563 -> 3144.005 us` and `3067.537 -> 3145.376 us`. B-pair lowered
  VGPR from `124` to `105` versus BQUAD with the same LDS, dot, LDS-load/store,
  global-load/store, and barrier counts, but ffn_gate regressed about `7.5%`.
- decision:
  reject before model tests; keep opt-in only as evidence.
- notes:
  Lower B-cache live state is not the right packed-medium axis for Q5 p33.
  Keep attacking the remaining Vulkan gap from Qcur/ffn_out/ffn_gate with
  packed-RHS ownership/issue-order or a closer RADV cooperative store/fragment
  contract, not more direct-F32 WMMA ownership reshuffles.

## 2026-06-19 - Q5_K p33 MMQL64 BK2 TN4/BQUAD compile rejection

- source:
  `sources/llama.cpp` dirty after adding the opt-in catalog route
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_tn4_bquad_wg256_f32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  llama-bench -j"$(nproc)"`, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused `MUL_MAT` rows, specifically the
  packed-Q8_1/x4 medium route that currently owns Qcur, ffn_out, and ffn_gate.
- route or candidate:
  packed Q8_1/x4 MMQL64 BK2 TN4/BQUAD route. It preserves the accepted
  BM64/BN64/BK_STEP=2/WG256/wave64 dataflow and BQUAD issue window, but changes
  per-lane output ownership from TM4/TN2 to TM2/TN4.
- baseline:
  accepted BQUAD HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_bquad.hsaco`.
- variant:
  TN4/BQUAD HSACO
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_tn4_bquad.hsaco`,
  gated by
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL64_BK2_TN4_BQUAD_PROMPT=1`.
- compile/profile artifact:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-tn4-bquad-focused-20260619-051644/`.
- static evidence:
  rejected before runtime. TN4/BQUAD emits wave64, SGPR `62`, VGPR `192`,
  VGPR spills `183`, private segment `736`, LDS `10240`, `256` v_dot sites,
  `50` LDS loads, `24` LDS stores, `39` global loads, `16` global stores,
  `249` waitcnt sites, and `2` barriers. The accepted BQUAD route is wave64,
  SGPR `62`, VGPR `124`, no spills, private segment `0`, LDS `10240`, `40`
  LDS loads, and `119` waitcnt sites.
- decision:
  reject at compile-resource gate. Do not run focused correctness/perf; the
  output-ownership pivot hits the gfx1151 register cliff and is not a viable
  packed-medium schedule.
- notes:
  This closes another local packed-output ownership axis for the p33 Q5 gap.
  Remaining work should either mechanically reproduce RADV's lower-level
  cooperative store/lane contract or change the packed-route contract more
  substantially than B-cache window or TM/TN ownership pivots.

## 2026-06-19 - Q5_K p33 current HRX/Vulkan focused checkpoint

- source:
  `sources/llama.cpp` commit
  `ae015fa5b hrx: add q5 mmql64 tn4 probe`.
- artifact:
  `cache/hrxv1/gfx1151/q5-p33-current-hrx-vulkan-focused-20260619-0600/`.
- shape:
  Qwen2.5 Coder 7B Q5_K_M p33 focused `MUL_MAT` rows from
  `q5_prompt_p33.txt`.
- route evidence:
  correctness-mode HRX trace shows Kcur on
  `hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32`; Qcur, ffn_out, and ffn_gate on
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32`.
- timing:

  | Row | HRX us | Vulkan us | HRX/Vulkan |
  | --- | ---: | ---: | ---: |
  | Kcur | 71.984 | 106.961 | 0.673 |
  | Qcur | 259.554 | 171.240 | 1.516 |
  | ffn_out | 1571.059 | 1113.970 | 1.410 |
  | ffn_gate | 1173.281 | 921.746 | 1.273 |
  | sum | 3075.879 | 2313.917 | 1.329 |

- decision:
  current p33 focused gap is no longer an aggregate mystery. Kcur is faster
  than Vulkan; the remaining shortfall is the packed-Q8_1/x4
  MMQL64/BK2/BQUAD route on Qcur, ffn_out, and ffn_gate.
- next schedule target:
  do not repeat closed local axes: BK1, B-pair, TN4, CR-major issue order,
  broad B-cache window, BN48 shrink, or Qcur-only selector split. The next
  useful Q5 p33 attempt should either change the packed-RHS/layout contract
  more substantially, or return to the lower-level cooperative-matrix
  load/store lane-ownership problem that RADV solves and HIP C++ has not yet
  exposed cleanly.

## 2026-06-19 - Q5_K p33 combined96 real-Q5 reproducer

- source:
  `sources/llama.cpp` dirty after extending
  `hrx-hip-bench-q5-wmma-full64-repro` with a `combined96` synthetic Q5/F32
  variant.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/q5-combined96-repro-20260619-052854/`.
- purpose:
  reproduce the rejected p33 combined96 catalog behavior outside
  `test-backend-ops` using synthetic Q5 blocks, F32 RHS, the real Q5 dequant
  helper, stride44 LDS staging, b64 fragment loads, halfword staged writeback,
  and CPU reference comparison.
- static evidence:
  `q5_combined96_repro_kernel` emits wave64, SGPR `55`, VGPR `142`, LDS
  `11264`, no spills, `16` WMMA, `48` `ds_load_b64`, `64`
  `ds_load_u16_d16`, `66` `ds_store_b16`, `96` `buffer_store_b32`, and `3`
  barriers.
- runtime evidence:
  combined96 reproduces the failure on small synthetic rows: p33/k256
  `nan=42`, p33/k512 `nan=28`, p33/k3584 `nan=26`, and p64/k3584
  `nan=28 inf=1`. Stress p33 also reproduces `nan=32 inf=30` at k256 and
  `nan=24 inf=30` at k3584. The low-live controls remain finite on the same
  small p33/k3584 case: `active4 nan=0 inf=0 max_abs=0.0996094` and
  `batched4 nan=0 inf=0 max_abs=0.0996094`.
- decision:
  keep combined96 rejected. This diagnostic proves the real-Q5 context plus
  the RADV-like combined writeback topology is enough to reproduce the
  numerical failure, while low-live controls stay finite. The next route-facing
  work needs a lane/value dependency fix, not another tile-size, selector, or
  B-cache-window pivot.

## 2026-06-19 - Q5_K p33 combined96 raw8/wait0 dependency probe

- source:
  `sources/llama.cpp` dirty after extending
  `hrx-hip-bench-q5-wmma-full64-repro` with raw-only and wait0 combined96
  variants.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/q5-combined96-wait0-repro-20260619-054018/`.
- purpose:
  split first-eight accumulator corruption from staged second-half writeback,
  and test whether the relaxed RADV-like `lgkmcnt(40)` wait is invalid in the
  HIP spelling.
- runtime evidence:
  on small p33/k3584, `combined96` produced `nan=26`, all in the first eight
  groups. `combined96-raw8` produced `nan=104 inf=1`, also entirely in the
  first eight groups. `combined96-raw8-wait0` still failed with
  `nan=144 inf=4`, and `combined96-wait0` still failed with `nan=30`. The
  controls remain finite: `active8 nan=0 inf=0` and `batched4 nan=0 inf=0`.
- static evidence:
  `combined96-raw8` and `combined96-raw8-wait0` both emit `16` WMMA,
  `48` `ds_load_b64`, `32` `buffer_store_b32`, `2` barriers, and no staged
  halfword loads. `combined96` and `combined96-wait0` both emit `16` WMMA,
  `48` `ds_load_b64`, `64` `ds_load_u16_d16`, `66` `ds_store_b16`, `96`
  `buffer_store_b32`, and `3` barriers. The finite `active8` control emits the
  same `16` WMMA and `32` raw stores but `64` `ds_load_b64`.
- decision:
  reject wait placement as the root-cause fix. The failing value appears
  before staged writeback and survives `lgkmcnt(0)`. The next diagnostic should
  target the explicit combined96 fragment-load/dependency contract itself,
  especially the difference between the failing 48-load path and finite
  64-load active8 path.

## 2026-06-19 - Q5_K p33 combined96 B-fragment padding probe

- source:
  `sources/llama.cpp` dirty after adding `combined96-raw8-bpad` and
  `combined96-bpad` diagnostic variants to
  `hrx-hip-bench-q5-wmma-full64-repro`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/q5-combined96-bpad-repro-20260619-054349/`.
- purpose:
  test whether the finite `active8` control is finite because it materializes
  all four B column fragments per K tile (`64` B64 LDS loads) while failing
  combined96 materializes only the two used B column fragments (`48` B64 LDS
  loads).
- static evidence:
  the padded variants did emit the intended `64` `ds_load_b64` instructions.
  `combined96-raw8-bpad` reports SGPR `65`, VGPR `178`, LDS `11264`, no
  private memory, `16` WMMA, `64` `ds_load_b64`, `32` `buffer_store_b32`, and
  `2` barriers. `combined96-bpad` reports the same SGPR/VGPR/LDS and `16`
  WMMA, plus `64` `ds_load_u16_d16`, `66` `ds_store_b16`, `96`
  `buffer_store_b32`, and `3` barriers.
- runtime evidence:
  on small p33/k3584, `combined96-raw8-bpad` still failed with `nan=136`, all
  in first-eight groups, and `combined96-bpad` still failed with `nan=38`, all
  in first-eight groups. The finite controls remain `active8 nan=0 inf=0` and
  `batched4 nan=0 inf=0`.
- decision:
  reject extra B-fragment materialization as the fix. The next useful
  diagnostic should compare explicit combined96 operand issue order against
  the finite `active8` array-loop body order, not vary wait count, B-load
  count, tile shape, or staged writeback.

## 2026-06-19 - Q5_K p33 array-loop reduced-B probe

- source:
  `sources/llama.cpp` dirty after adding an `array8-b2` diagnostic variant to
  `hrx-hip-bench-q5-wmma-full64-repro`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/q5-array8-b2-repro-20260619-054730/`.
- purpose:
  preserve active8's array-loop WMMA topology while reducing B fragment loads
  to the two column tiles used by combined96.
- static evidence:
  `array8-b2` emitted wave64, SGPR `52`, VGPR `142`, LDS `11264`, no private
  memory, `16` WMMA, `48` `ds_load_b64`, `32` `buffer_store_b32`, `2`
  barriers, and `24` wait instructions.
- runtime evidence:
  `array8-b2` failed on the small profile: p33/k256 `nan=160 inf=2`,
  p33/k3584 `nan=136 inf=1`, and p64/k3584 `nan=112`; all NaNs were in the
  first-eight groups. `active8` and `batched4` remained finite on the same
  p33/k3584 case.
- decision:
  reject reduced-B array-loop as a route direction. The first-eight failure
  appears whenever the path uses only the two live B column fragments, even if
  the WMMA body is array-looped and fully waited. The next probe should keep
  all four B fragments live in the active8 loop topology and vary only the
  compute-loop structure.

## 2026-06-19 - Q5_K p33 array-loop full-B no-if probe

- source:
  `sources/llama.cpp` dirty after adding `array8-fullb-noif` to
  `hrx-hip-bench-q5-wmma-full64-repro`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/q5-array8-fullb-noif-repro-20260619-055054/`.
- purpose:
  keep all four B fragments live as in finite `active8`, but remove the
  `tile < active_groups` branch and compute only the first eight groups with a
  branch-free `col_sub < 2` loop.
- static evidence:
  `array8-fullb-noif` emitted wave64, SGPR `64`, VGPR `170`, LDS `11264`, no
  private memory, `16` WMMA, `64` `ds_load_b64`, `32` `buffer_store_b32`, `2`
  barriers, and `24` waits.
- runtime evidence:
  the variant is finite on the small profile: p33/k256 `nan=0 inf=0
  max_abs=0.0107422`, p33/k3584 `nan=0 inf=0 max_abs=0.0996094`, and
  p64/k3584 `nan=0 inf=0 max_abs=0.0996094`. It is also more accurate than
  the old `active8` control on the p64 case, while `array8-b2` still fails.
- decision:
  accept as a positive first-eight dependency contract, not as a route. The
  stable ingredients are full four-column B fragment materialization plus the
  array-loop fragment topology; the `active8` branch is not required. A p33
  route still needs `col=32` coverage, so the next diagnostic should compute
  groups `8..11` in a separate low-live phase without batched4's full
  restaging cost.

## 2026-06-19 - Q5_K p33 array-loop tail phase probes

- source:
  `sources/llama.cpp` dirty after adding `array8-tail4-2dispatch` and
  `array8-tail8-2dispatch` diagnostics to
  `hrx-hip-bench-q5-wmma-full64-repro`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifacts:
  `cache/hrxv1/gfx1151/q5-array8-tail4-2dispatch-repro-20260619-055405/`
  and
  `cache/hrxv1/gfx1151/q5-array8-tail8-2dispatch-repro-20260619-055502/`.
- purpose:
  extend the positive first-eight contract to full p33 by computing groups
  `8..11` in a separate low-live phase, then test whether a paired groups
  `8..15` phase is needed.
- static evidence:
  first-eight and tail8 phases both emit SGPR `64`, VGPR `170`, `16` WMMA,
  `64` `ds_load_b64`, `32` `buffer_store_b32`, `2` barriers, and no private
  memory. The tail4 phase emits SGPR `55`, VGPR `146`, `8` WMMA, `64`
  `ds_load_b64`, `16` stores, and no private memory. These are low-live,
  spill-free shapes.
- runtime evidence:
  both two-dispatch variants remove p33 NaNs but fail value correctness:
  `array8-tail4-2dispatch` p33/k3584 `max_abs=5649.78`, and
  `array8-tail8-2dispatch` p33/k3584 `max_abs=3642.56`. Tail8 also produces
  p64 NaNs outside the first-eight groups. The full batched4 control remains
  correct on p33/k3584 with `max_abs=0.0996094`.
- decision:
  reject the simple two-dispatch tail repair. The p33 tail-column value
  contract is still not captured unless the full batched4 source/codegen
  context is used. Next route-facing work should start from the exact batched4
  loop shape and only then reduce phases/restaging if correctness survives.

## 2026-06-19 - Q5_K p33 pruned batched4 probe

- source:
  `sources/llama.cpp` dirty after templating
  `hrx-hip-bench-q5-wmma-full64-repro`'s batched4 diagnostic on
  `group_base_end` and adding `batched4-p33`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q5-wmma-full64-repro -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/q5-batched4-p33-repro-20260619-055914/`.
- purpose:
  preserve the exact correct batched4 source/codegen contract while skipping
  only the unused group_base `12` phase for p33.
- static evidence:
  `batched4-p33` emits SGPR `51`, VGPR `182`, no private memory, `24` WMMA,
  `192` `ds_load_b64`, `48` `buffer_store_b32`, `9` barriers, and `80` waits.
  Full batched4 emits `32` WMMA, `256` `ds_load_b64`, `64` stores, `12`
  barriers, and `108` waits.
- runtime evidence:
  `batched4-p33` is correctness-clean on the small profile: p33/k256
  `nan=0 inf=0 max_abs=0.0107422`, and p33/k3584 `nan=0 inf=0
  max_abs=0.0996094`, matching full batched4's p33/k3584 error scale.
- decision:
  accept as the p33 direct-WMMA correctness oracle, not as a production route.
  It removes one of four batched4 phases but still does three full K-restaging
  passes, so the next production candidate should start from this contract and
  reduce restaging only if correctness survives.

## 2026-06-19 - Q5_K p33 pruned batched4 catalog route

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_batched4_p33_f16acc_wg256_f32`
  as an opt-in catalog route guarded by
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_BATCHED4_P33_F16ACC_WG256_PROMPT=1`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench
  test-backend-ops -j"$(nproc)"`; CMake generated
  `mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_batched4_p33_wg256.hsaco`.
- artifact:
  `cache/hrxv1/gfx1151/q5-p33-batched4-p33-catalog-20260619-060808/`.
- purpose:
  move the standalone p33-pruned batched4 result into the real catalog ABI and
  focused Qwen2.5 Coder Q5_K_M p33 rows.
- route evidence:
  selected
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_batched4_p33_f16acc_wg256_f32`
  for all four p33 rows.
- static evidence:
  selected export emits wave64, SGPR `51`, VGPR `182`, LDS `11264`, no
  private memory, no spills, `24` WMMA, `192` `ds_load_b64`, `48`
  `buffer_store_b32`, `9` barriers, and `80` waits.
- correctness:
  Kcur, Qcur, and ffn_gate passed. `ffn_out` failed CPU-reference at
  `k=18944 rows=3584 cols=33` with finite `ERR=0.043113960`.
- timing:
  skipped; a route that fails the focused CPU-reference gate is not a
  production timing candidate.
- decision:
  reject as a catalog route. The standalone p33-pruned result only proved
  `k=256` and `k=3584`; it does not generalize to the wide-K p33 row. Full
  batched4 remains the direct-WMMA correctness oracle, and the next useful
  route should either keep the full source contract for wide-K or move back to
  the packed-Q8_1 medium path.

## 2026-06-19 - Q5_K p33 MMQL64 BK2 BQUAD BHALF default

- source:
  `sources/llama.cpp` dirty after adding
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_bquad_bhalf_wg256_f32`
  and defaulting it on `gfx1151` for Q5_K packed-Q8_1 prompt rows with
  `32 <= cols <= 64`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench
  test-backend-ops -j"$(nproc)"`; the new HIP C++ source is built through
  CMake/Ninja as
  `mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_bquad_bhalf.hsaco`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M p33, plus p512 and p513 focused non-steal gates.
- route or candidate:
  packed-Q8_1/x4 MMQL64 BM64/BN64/WG256/wave64/BK_STEP=2/BQUAD route that
  stores Q8_1 RHS `d/s` payloads in LDS as half values and converts after
  shared load. Rollback:
  `GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL64_BK2_BQUAD_BHALF_PROMPT=1`.
- baseline command:
  `test-backend-ops perf -b HRX0 -o MUL_MAT --test-file q5_prompt_p33.txt
  --output csv` with the rollback env above.
- variant command:
  same focused command without rollback env, using the new default selector.
- route trace:
  `cache/hrxv1/gfx1151/q5-bquad-bhalf-default-postpromotion-20260619-062244/`.
- profile/timing:
  focused default/rollback CSVs and Qwen2.5 Coder p33 `llama-bench -r 5`
  JSONs in the artifact above.
- correctness:
  post-promotion default CPU-reference gates passed for p33, p512, and p513.
  Route traces show p33 selects BHALF for Qcur, ffn_out, and ffn_gate while
  Kcur stays on rows2/cols8; p512 remains on MMQL128 and p513 remains on
  MMQL128 BQUAD tail routing.
- timing:
  focused p33 default-vs-rollback summed four-row time improved
  `3027.406807 -> 3076.388160 us` in round 1 and
  `3035.154640 -> 3086.045951 us` in round 2, about a 1.6% local lift.
  Same-binary model smoke improved `209.982618 -> 211.406107 tok/s`
  (`1.006779x`) with `backends=HRX`.
- decision:
  promote as a small gfx1151 p33 default with rollback env. This is not Vulkan
  parity; it only closes a narrow packed-route cache/layout axis while the
  remaining Q5 p33 gap still needs a true RADV cooperative-matrix clone or a
  stronger packed medium schedule.
- notes:
  static selected-symbol evidence from
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-bquad-bhalf-20260619-061615/` shows
  wave64, SGPR `61`, VGPR `122`, LDS `9728`, no private memory, `256`
  `v_dot4_i32_iu8`, `39` global loads, `16` global stores, `2` barriers, and
  `117` waits. The previous BQUAD route emitted SGPR `62`, VGPR `124`, LDS
  `10240`, and `119` waits with the same dot/store/barrier counts.

## 2026-06-19 - Q5_K p33 VK64 wave4row batch-wait rejection

- source:
  `sources/llama.cpp` dirty after adding the opt-in
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk64_padded44_w64_wave4row_batchwait_f16acc_wg256_f32`
  route.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench
  test-backend-ops -j"$(nproc)"`; CMake generated
  `mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_wave4row_batchwait_wg256.hsaco`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M focused p33 rows, with p512/p513 non-steal gates.
- route or candidate:
  direct-F32 WMMA VK64 wave4row route with all four wave64s active, four live
  accumulators per wave, one A/B staging pass per K tile, and batch-waited
  four-B64 fragment loads. Gate:
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK64_PADDED44_W64_WAVE4ROW_BATCHWAIT_F16ACC_WG256_PROMPT=1`.
- route trace:
  `cache/hrxv1/gfx1151/q5-p33-wave4row-batchwait-20260619-064007/`.
- correctness:
  p33 CPU-reference passed all four focused rows and traces selected the
  batch-wait route for all four. p512 and p513 CPU-reference passed with no
  route stealing; p512 stayed on MMQL128 and p513 stayed on MMQL128 BQUAD.
- timing:
  focused p33 default BHALF versus batch-wait sums were
  `3038.316277 us -> 6225.337728 us` (`2.048943x` slower). Row ratios were
  Kcur `5.025046x`, Qcur `1.791643x`, ffn_out `2.022950x`, and ffn_gate
  `1.957118x` slower.
- decision:
  reject before model tests. Batch-waiting fixes the invalid nowait contract
  and reduces selected-symbol wait count versus the older waited wave4row
  route, but direct-F32 WMMA remains far slower than the accepted packed route.
- notes:
  selected-symbol static facts are wave64, SGPR `32`, VGPR `102`, LDS `11264`,
  no private memory, no spills, `8` visible WMMA, `40` `ds_load_b64`, `16`
  `buffer_store_b32`, `2` barriers, and `36` waits. This closes the simple
  wave4row load-window pivot; the next Q5 p33 attempt should target the RADV
  halfword/writeback store contract or return to packed-Q8_1 medium schedule
  work.

## 2026-06-19 - Q8_0 RADV-like coopstore primitive checkpoint

- source:
  `sources/llama.cpp` commit
  `f0395d8f3 hrx: record gfx1151 coopstore primitive`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract hrx-hip-bench-lds-halfword-stage
  -j"$(nproc)"`; both targets were up to date and built through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-radv-mixed192-primitive-20260619-070221/`.
- purpose:
  retire the broad "missing matrix-store primitive" blocker after confirming
  whether HIP inline asm can emit the RADV-like halfword LDS plus raw
  buffer-store surface without rocWMMA.
- correctness:
  `hrx-hip-bench-coopmat-store-contract --mode=radv-mixed192` passed with
  `elements=12288 bad=0 max_abs=0`. `hrx-hip-bench-lds-halfword-stage
  --mode=bulk128-wg256` passed with `elements=8192 bad=0`.
- static evidence:
  selected `coopstore_probe_radv_mixed192` is wave64, SGPR `10`, VGPR `32`,
  private segment `0`, `192` `buffer_store_b32`, `0` `global_store_b32`,
  `128` `ds_load_u16_d16` or `ds_read_u16_d16`, `128` `ds_store_b16`, `2`
  barriers, `135` waits, and no WMMA instructions.
- decision:
  record as a primitive contract pass, not a production route. rocWMMA remains
  absent from `/srv/vm-shared/rocm/rocm-head`, but the required store-side
  opcode surface is expressible with HIP inline asm. The remaining Q8/Q6
  blocker is integrating this low-barrier halfword LDS plus `192`
  raw-buffer-store topology with live WMMA fragments, the gfx1151 OPSEL lane
  map, RADV-like outstanding LDS-load issue windows, and acceptable VGPR
  pressure.
- notes:
  the Q8 ledger already covers the nearby wrapper variants: fullpair reaches
  `192` buffer stores but is not a correctness map, packstage fast-half
  fullpair matches the static opcode surface but misses pressure and full-pair
  semantics, stream-row fullpair matches the tracked static contract but uses
  invalid row+16 output ownership, and stream-row selected-half failed focused
  CPU-reference with finite error. Another catalog wrapper is not justified
  until a fixture proves a new lane/output contract or a lower-level source
  form exposes RADV's cooperative-store semantics directly.

## 2026-06-19 - Q8_0 dual-OPSEL two-tile WMMA fixture

- source:
  `sources/llama.cpp` dirty after adding
  `--mode=dual-opsel-two-tile-store-contract` to
  `hrx-hip-bench-wmma-f16-lane-map`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-f16-lane-map -j"$(nproc)"`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-dual-opsel-two-tile-20260619-070903/`.
- purpose:
  answer whether the high OPSEL half can be used as a second independent
  output tile if it is produced deliberately, rather than read accidentally
  from a single selected-OPSEL WMMA call.
- correctness:
  runtime passed with `rows=32 cols=16 written=512 duplicate_coords=0
  missing_target=0 nan=0 bad_value=0 contract_valid=1`. The low-half tile
  writes expected value `16`; the high-half tile writes expected value `32`.
- static evidence:
  extracted the embedded gfx1151 code object from `.hip_fatbin` with
  `clang-offload-bundler`. The selected symbol
  `wmma_f16_dual_opsel_two_tile_store_contract_probe` emits wave64, SGPR `12`,
  VGPR `22`, no private segment, no spills, `2`
  `v_wmma_f16_16x16x16_f16` instructions, `8` global stores, `8` global
  atomics for fixture counts, and one wait.
- decision:
  dual-OPSEL packing is semantically valid only when the source issues a
  separate high-OPSEL WMMA. This does not rescue the rejected Q8_0 fullpair
  route or explain RADV's 32-WMMA large route: reading both halves after one
  selected-OPSEL WMMA is not a valid second tile, while making it valid adds
  WMMA work. Keep future Q8 direct-WMMA candidates on selected-half ownership
  unless a lower-level cooperative-matrix primitive exposes RADV's store
  semantics directly.

## 2026-06-19 - Q8_0 live-WMMA RADV-like mixed store contract

- source:
  `sources/llama.cpp` dirty after adding `wmma-radv-mixed96` and
  `wmma-radv-mixed192` to
  `hrx-hip-bench-coopmat-store-contract`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-live-wmma-mixed-probe-20260619-071800/`.
- model/shape:
  standalone gfx1151 schedule fixture for the Q8/Q5 direct-WMMA store-side
  blocker; no model route was changed.
- route or candidate:
  live f16 WMMA accumulator values feeding the RADV-like raw
  `buffer_store_b32` plus halfword LDS stage topology.
- baseline command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-coopmat-store-contract
  --mode=radv-mixed192`.
- variant command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-coopmat-store-contract
  --mode=wmma-radv-mixed96` and `--mode=wmma-radv-mixed192`.
- route trace:
  not applicable; standalone bench.
- profile/timing:
  selected-symbol ISA and metadata in the artifact:
  `wmma-radv-mixed96.amdgcn.txt`, `wmma-radv-mixed192.amdgcn.txt`, and
  `llvm-readobj-notes.txt`.
- correctness:
  both modes passed with `elements=12288 bad=0 max_abs=0`.
- timing:
  not a timing candidate.
- static evidence:
  `wmma-radv-mixed96` emits wave64, SGPR `26`, VGPR `59`, LDS `8192`, no
  private segment, no spills, `8` WMMA, `96` `buffer_store_b32`, `64`
  halfword LDS loads/stores, and `2` barriers. `wmma-radv-mixed192` emits
  wave64, SGPR `26`, VGPR `92`, LDS `16384`, no private segment, no spills,
  `16` WMMA, `192` `buffer_store_b32`, `128` halfword LDS loads/stores, and
  `2` barriers.
- decision:
  record as a primitive contract pass, not a production route. This proves the
  mixed store topology can consume live WMMA accumulator values without
  immediate selected-lane corruption or a register cliff in a standalone
  fixture.
- notes:
  the fixture still uses synthetic A/B fragments and does not reproduce the
  real Q8/Q5 dequant, LDS fragment load window, or catalog ABI. The next
  direct-WMMA route work should combine this selected-half store contract with
  the real fragment load path and compare the first-WMMA issue window against
  RADV before model-level promotion.

## 2026-06-19 - Q8_0 LDS-fragment live-WMMA mixed store contract

- source:
  `sources/llama.cpp` dirty after adding `wmma-lds-radv-mixed96` and
  `wmma-lds-radv-mixed192` to
  `hrx-hip-bench-coopmat-store-contract`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-coopmat-store-contract -j"$(nproc)"`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-lds-live-wmma-mixed-probe-20260619-072609/`.
- model/shape:
  standalone gfx1151 schedule fixture for the Q8/Q5 direct-WMMA load plus
  store blocker; no model route was changed.
- route or candidate:
  LDS fragment setup loaded through `ds_read_b64`, f16 WMMA, selected-half raw
  buffer stores, and the halfword LDS store/load path.
- baseline command:
  prior passing `--mode=wmma-radv-mixed96` and `--mode=wmma-radv-mixed192`.
- variant command:
  `--mode=wmma-lds-radv-mixed96` and `--mode=wmma-lds-radv-mixed192`.
- route trace:
  not applicable; standalone bench.
- profile/timing:
  selected-symbol ISA and metadata in the artifact:
  `wmma-lds-radv-mixed96.amdgcn.txt`,
  `wmma-lds-radv-mixed192.amdgcn.txt`, and `llvm-readobj-notes.txt`.
- correctness:
  large direct-store control passed with `elements=12288 bad=0 max_abs=0`.
  Medium failed exactly when live accumulator groups `8..15` were staged
  through halfword LDS: `bad=1984 max_abs=256 first_bad=2048 actual=2.005
  expected=48`.
- timing:
  not a timing candidate.
- static evidence:
  medium emits wave64, SGPR `14`, VGPR `95`, LDS `24576`, no spills,
  `32` `ds_load_b64`, `16` WMMA, `96` `buffer_store_b32`, `64` halfword LDS
  stores/loads, and `3` barriers. Large emits wave64, SGPR `14`, VGPR `103`,
  LDS `32768`, no spills, `32` `ds_load_b64`, `16` WMMA, `192`
  `buffer_store_b32`, `128` halfword LDS stores/loads, and `3` barriers.
- decision:
  reject the live-accumulator halfword-stage contract for production route
  work. Keep the large direct-store control as positive evidence that LDS
  fragment loads plus live WMMA plus direct raw stores can coexist.
- notes:
  this explains why selected-half packstage-style routes remain dangerous:
  the invalid part is not raw `buffer_store_b32` or LDS fragment loading in
  isolation, but the lane map when live WMMA accumulator values are staged
  through halfword LDS. The next route should keep selected live accumulator
  values on the direct raw-store path or use a true lower-level cooperative
  matrix store primitive.

## 2026-06-19 - Q8_0 packed split-qsum qpack-hoist rejection

- source:
  temporary edit in `sources/llama.cpp` hoisted `hrx_q8_0_pack4(block, iqs)`
  out of the two-column-chunk loops in the accepted BN128 and BN112
  split-qsum packed-Q8_1 kernels. Runtime source was reverted after the
  focused regression.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  llama-bench -j"$(nproc)"`, Release, ROCm `/srv/vm-shared/rocm/rocm-head`,
  CMake/Ninja generated both touched HSACOs.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-qpack-hoist-focused-20260619-073318/`.
- model/shape:
  Llama 3.1 8B Q8_0 exported prompt rows, p512 and exact p513 odd-tail.
- route or candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` for p512 and
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` for p513.
- baseline command:
  prior current-head focused artifact
  `cache/hrxv1/gfx1151/q8_0-current-focused-eb85c542-20260619-004731/`.
- variant command:
  focused `test-backend-ops test/perf -b HRX0 -o MUL_MAT` against
  `q8_0_prompt.txt` and `q8_0_prompt_all.txt` under
  `GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_TRACE_ROUTES=1`.
- route trace:
  p512 selected BN128 split-qsum for all five rows; p513 selected BN112
  split-qsum for all five rows.
- profile/timing:
  p512 focused total regressed `72712.698 -> 75663.326 us`; p513 focused total
  regressed `81093.295 -> 83117.543 us`. The large `result_output` row
  regressed `56659.810 -> 58707.143 us` at p512 and `63606.143 -> 64912.571 us`
  at p513.
- correctness:
  CPU-reference focused gates passed for all five p512 rows and all five p513
  rows.
- static evidence:
  BN128 remained wave32, SGPR `27`, VGPR `152`, LDS `4352`, no spills. BN112
  moved from the prior VGPR `134` to `135`, with SGPR `28`, LDS `3808`, no
  spills. No catastrophic resource cliff appeared; the loss is likely emitted
  schedule/live-range quality rather than an obvious spill or occupancy failure.
- decision:
  reject and revert. Hoisting the A-pack values is a source-level cleanup that
  loses to the current compiler schedule on gfx1151. Do not repeat this axis
  unless it is paired with a materially different packed dataflow or lower-level
  generated schedule evidence.

## 2026-06-19 - Q8_0 focused HRX/Vulkan backend-op gap checkpoint

- source:
  `sources/llama.cpp` commit
  `ab41b8701 hrx: test q6 vk128 bufferstore path`; no llama.cpp source edit for
  this checkpoint.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-current-focused-hrx-vulkan-compare-ab41b8701-20260619-075951/`.
- tool:
  added `tools/hrxv1_compare_backend_op_perf.py` to compare
  `test-backend-ops --output csv` files by exported op row name and emit JSON
  plus Markdown summaries.
- purpose:
  pin the current Q8_0 p512/p513 gap at the backend-op/schedule level before
  adding another direct-WMMA route. Inputs were the existing current HRX
  focused artifact
  `cache/hrxv1/gfx1151/q8_0-current-focused-eb85c542-20260619-004731/` and
  focused Vulkan artifact
  `cache/hrxv1/gfx1151/q8_0-current-focused-vulkan-eb85c542-20260619-005337/`.
- evidence:
  p512 focused total is HRX `72712.698 us` versus Vulkan `48100.936 us`,
  `1.512x` slower. p513 focused total is HRX `81093.295 us` versus Vulkan
  `55428.701 us`, `1.463x` slower.
- row ratios:
  p512 Vcur `2.039x`, Qcur `1.834x`, ffn_out `1.772x`, ffn_gate `1.595x`,
  result_output `1.464x`; p513 Vcur `1.990x`, Qcur `1.934x`, ffn_out
  `1.838x`, ffn_gate `1.505x`, result_output `1.409x`.
- decision:
  keep Q8_0 as an active schedule boulder. This confirms the current
  model-level Q8_0 gap has a real focused kernel component, not only graph or
  runtime overhead. The next source candidate should not be another packed
  split-qsum cleanup or scalar direct-WMMA wrapper; it needs to target the
  RADV structural delta already identified in the ledger: cooperative-matrix
  load/store lowering, selected live accumulator ownership, and raw
  buffer-store writeback without the failing halfword-stage lane map.

## 2026-06-19 - Q8_0 K2 direct192 raw-store control

- source:
  `sources/llama.cpp` added
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-direct192-raw`.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-direct192-raw-probe-20260619-083318/`.
- purpose:
  separate full RADV-sized raw `buffer_store_b32` count from the failing
  scalarized halfword LDS stage in the K2 live-WMMA mixed fixture.
- correctness:
  repeated runs passed `wmma-lds-k2-direct192-raw` and
  `wmma-lds-k2-mixed128-padded32` with `bad=0`; the same repeated run failed
  `wmma-lds-k2-radv-mixed192` consistently at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`.
- static evidence:
  direct192 raw emitted wave64, SGPR `14`, VGPR `153`, LDS `16384`, no private
  segment, no spills, `32` WMMA, `64 ds_load_b64`,
  `192 buffer_store_b32`, no `ds_store_b16`, no `ds_load_u16_d16`, two
  barriers, and `7 s_waitcnt`.
- decision:
  close the raw-store-count hypothesis. The remaining Q8_0 direct-WMMA
  blocker is the scalarized halfword LDS stage expansion, not the full
  `192 buffer_store_b32` raw writeback surface. Do not spend the next pass on
  more HIP C++ halfword-stage expansion; move to a lower-level
  cooperative-store primitive or a compact compiler reproducer.

## 2026-06-19 - Q8_0 K2 stage96 accsink control

- source:
  `sources/llama.cpp` added
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-stage96-accsink`.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-stage96-accsink-probe-20260619-084215/`.
- purpose:
  test whether K2 WMMA plus `96` scalarized halfword LDS stage pairs fails
  without the full checked direct accumulator raw-store surface.
- correctness:
  `stage96-accsink` passed initial repeated sweeps, then failed once in a
  20-rep validation at
  `first_bad=8640 group=33 slot=3 lane=0 actual=1024 expected=16643`; in the
  same binary `mixed160-lo-tight` failed repeatedly at
  `first_bad=4800 group=18 slot=3 lane=0`.
- static evidence:
  accsink emitted wave64, SGPR `14`, VGPR `57`, LDS `29184`, no private
  segment, no spills, `2` WMMA, `64 ds_load_b64`, `100 ds_store_b16`,
  `96 ds_load_u16_d16`, `96 buffer_store_b32`, three barriers, and
  `104 s_waitcnt`. The failing tight mixed160 emitted the same `96`
  halfword-stage load/store count but `32` WMMA, VGPR `162`, and
  `160 buffer_store_b32`.
- decision:
  treat accsink as a flaky minimized reproducer, not a clean pass. The
  deterministic failure requires the expanded halfword stage together with the
  high-live direct accumulator raw-store surface. The next useful production
  direction is lower-level cooperative store or low-live batching, not more
  scalarized full-tile HIP C++ staging.

## 2026-06-19 - Q8_0 K2 mixed160 linear-stage topology control

- source:
  `sources/llama.cpp` at `604ff0903-dirty` added
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-mixed160-linearstage`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- route or candidate:
  standalone cooperative-store fixture only; no production selector.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-linearstage-probe-20260619-085145/`.
- purpose:
  keep the failing K2 live-WMMA plus `96` halfword-stage pair surface, but
  change the synthetic stage mapping from col-major 16x16 to lane-linear.
- correctness:
  `wmma-lds-k2-mixed160-linearstage` passed the first five repeated runs, then
  failed in a longer validation at rep 6 with
  `first_bad=7808 group=30 slot=2 lane=0 actual=1024 expected=15874`. In the
  same binary, `wmma-lds-k2-mixed160-lo-tight` failed immediately at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`, and
  `wmma-lds-k2-direct192-raw` passed five repeated runs.
- static evidence:
  linear-stage and failing col-major tight modes both emitted wave64, SGPR
  `14`, VGPR `162`, LDS `28672`, no private segment, no spills, `32` WMMA,
  `64 ds_load_b64`, `96 ds_store_b16`, `96 ds_load_u16_d16`,
  `160 buffer_store_b32`, three barriers, and `103 s_waitcnt`.
- decision:
  reject lane-linear staging as a production workaround. Address topology
  changes the failure from deterministic early col-major corruption to later
  flaky corruption without changing opcode/resource counts, so the blocker is
  still the combined high-live K2 WMMA plus expanded scalarized halfword-stage
  surface. The next route work should reconstruct the real cooperative store
  lane ownership or use a lower-level primitive instead of adding more
  scalarized HIP C++ stage groups.

## 2026-06-19 - Q8_0 K2 mixed160 split-stage topology control

- source:
  `sources/llama.cpp` at `69b38afdd-dirty` added
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-mixed160-splitstage`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- route or candidate:
  standalone cooperative-store fixture only; no production selector.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-splitstage-probe-20260619-085731/`.
- purpose:
  keep the K2 live-WMMA and total mixed160 halfword-stage count, but split the
  synthetic halfword stage into a 16-group store/load chunk and an 8-group
  store/load chunk instead of bulk-storing all `96` halfword pairs before
  loading them.
- correctness:
  `wmma-lds-k2-mixed160-splitstage` passed 20 repeated runs with `bad=0`. In
  the same binary, `wmma-lds-k2-mixed160-lo-tight` failed immediately at
  `first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`, and
  `wmma-lds-k2-direct192-raw` passed five repeated runs.
- static evidence:
  split-stage emitted wave64, SGPR `14`, VGPR `162`, LDS `24576`, no private
  segment, no spills, `32` WMMA, `64 ds_load_b64`, `96 ds_store_b16`,
  `96 ds_load_u16_d16`, `160 buffer_store_b32`, five barriers, and
  `105 s_waitcnt`. The failing bulk-stage tight mode has the same key opcode
  counts and VGPR but only three barriers and LDS `28672`.
- decision:
  accept as positive diagnostic evidence, not as a production route. Bulk
  halfword-stage residency/order is a real part of the failure: chunking the
  stage into store/load phases fixes the fixture without reducing WMMA count or
  VGPR. The next Q8_0 route experiment should test chunked cooperative
  writeback or low-residency halfword staging in the real catalog ABI, then
  compare barrier/order cost against the current packed-Q8_1 default.

## 2026-06-19 - Q8_0 packstage fast-half split-selected catalog route

- source:
  `sources/llama.cpp` at `6bb9d9c24-dirty` added
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_split_selected_bufferstore_f16acc_wg256_f32`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- route or candidate:
  opt-in catalog route under
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_FAST_HALF_SPLIT_SELECTED_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifacts:
  static comparison:
  `cache/hrxv1/gfx1151/q8_0-packstage-fast-half-split-selected-compile-20260619-090824/`;
  focused CPU-reference:
  `cache/hrxv1/gfx1151/q8_0-packstage-fast-half-split-selected-focused-20260619-091236/`.
- purpose:
  port the passing split-stage cooperative-store fixture into the real Q8_0
  direct-WMMA ABI while keeping the selected-half D-coordinate map. This tests
  whether lower-residency halfword output staging fixes the real route without
  reintroducing the invalid full-pair output mapping.
- static evidence:
  emitted wave64, SGPR `28`, VGPR `196`, LDS `22528`, no private segment, no
  spills, `32` WMMA, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `128 buffer_store_b32`, two barriers, and `199`
  waitcnt-class instructions. This matches RADV on broad halfword-LDS topology
  but misses the Q8_0 large oracle's VGPR `192`, `192 buffer_store_b32`, and
  first-WMMA issue window. RADV keeps `59 ds_load_b64` before the final wait
  with final pre-WMMA `lgkmcnt=51`; this HIP route has `24` pre-WMMA
  `ds_load_b64`, one load immediately before the final wait, and final
  `lgkmcnt=0`.
- correctness:
  focused p512 CPU-reference with `GGML_HRX_TRACE_ROUTES=1` showed `Vcur-0`
  and `Qcur-0` stayed on the packed Q8_1 provider and passed. The opt-in route
  selected for the wide rows and failed: `ffn_out-0 ERR=2.934616473`,
  `ffn_gate-0 ERR=2.956634549`, and `result_output` produced
  `NaN at index 6188064`.
- decision:
  reject before timing/model A/B. The split-stage fixture proved that halfword
  stage residency/order matters, but the real HIP C++ catalog route still does
  not reproduce RADV's cooperative-store lane ownership or load scheduling.
  Keep the route opt-in only as diagnostic evidence.

## 2026-06-19 - Q8_0 WMMA issue-window recheck

- source:
  existing `hrx-hip-bench-wmma-issue-window` bench from `sources/llama.cpp` at
  `77b8bd6f4`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/wmma-issue-window-recheck-20260619-091718/`.
- purpose:
  determine whether the next Q8_0 route should pursue the full RADV
  16-fragment load window or a smaller correctness-clean ladder rung.
- runtime:
  `lgkm51`, `mediumfrag12`, and `mediumfrag12-combined96` passed with no NaNs.
  `realfrag16`, `realfrag16-direct`, and `realfrag8` produced NaNs.
- static evidence:
  `lgkm51` proves HIP C++ plus inline wait syntax can emit a RADV-like
  outstanding LDS window in a synthetic control: `64 ds_load_b64`, final
  pre-WMMA `lgkmcnt=51`, and `64` B64 loads immediately before the final wait.
  `realfrag16` is not usable despite the explicit wait cadence: it emits only
  `32` B64 loads before first WMMA and fails numerically. The best clean ladder
  rung is `mediumfrag12-combined96`, which emits `48 ds_load_b64`, final
  `lgkmcnt=40`, `16` WMMA, `64 ds_store_b16`, `64 ds_load_u16_d16`,
  `96 buffer_store_b32`, and three barriers with no NaNs.
- decision:
  do not port the full `realfrag16` source shape into the catalog. The next
  Q8_0 parity-directed probe should port `mediumfrag12-combined96` into an
  opt-in catalog route or focused catalog ABI shim, then check whether the
  compiler preserves the `48`-load window and whether the partial store
  topology can be made semantically correct on model-derived rows.

## 2026-06-19 - Q8_0 group12 selected-stage remap split

- source:
  `sources/llama.cpp` at `c7c41157f-dirty` extended
  `hrx-hip-bench-q8-wmma-repro` with selected-stage remap modes and bad-sample
  output.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/q8-wmma-remap-stage-20260619-112352/`.
- purpose:
  determine whether the deterministic group12 selected-stage failure follows
  group12 WMMA compute, group12 staged-store coordinates, or their combination.
- correctness:
  `remap-c0-s12` and `remap-c0-s12-stage-selected` passed p64 with `bad=0`.
  `remap-c12-s0-abcopy-stage-selected` also passed p64 with `bad=0`.
  The uncopied `remap-c12-s0` failed completely and the B-copy-only staged
  remap still failed, so the clean group12 compute remap requires explicit
  A+B fragment materialization. Earlier same-binary bad-sample output showed
  `single-group12-*-stage-selected` failures concentrated at col `50` odd rows
  and col `57` even rows, with no NaNs/infinities.
- decision:
  reject the isolated compute-bug and isolated store-coordinate-bug
  hypotheses. The bad case is the combined `col_sub=3` accumulator plus
  group12 selected staged writeback lane contract. Keep the probes as
  diagnostics; the next production-directed Q8_0 work should use a lower-level
  cooperative store/lane primitive or compact compiler reproducer rather than
  another scalarized selected-stage HIP C++ route.

## 2026-06-19 - Q8_0 group12 accumulator-copy stage probe

- source:
  `sources/llama.cpp` at `701aed6f2-dirty` extended
  `hrx-hip-bench-q8-wmma-repro` with one-time accumulator-copy and per-slot
  regcopy variants for the group12 selected-stage writeback.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/q8-wmma-group12-regcopy-20260619-113027/`.
- purpose:
  test whether explicit accumulator moves can repair the combined group12 WMMA
  accumulator plus selected staged writeback failure.
- correctness:
  baseline group12 selected-stage modes reproduced `bad=16`. One-time
  accumulator copy reduced both B-copy and A+B-copy variants to `bad=12`,
  removing slot `0` from the bad-sample pattern but leaving slots `1..3`
  wrong. Per-slot regcopy worsened sharply: B-copy regcopy had `bad=46`,
  max_abs `514.864`; A+B-copy regcopy had `bad=43`, max_abs `514.867`.
- static evidence:
  extracted object, notes, disassembly, and symbol summary are under
  `cache/hrxv1/gfx1151/q8-wmma-group12-regcopy-20260619-113027/static/`.
  The selected-stage group12 variants compiled as separate template
  instantiations; copy variants add `v_mov_b32` pressure as intended.
- decision:
  reject accumulator-copy as a production workaround. The slot-0 movement
  confirms register-layout/source-spelling sensitivity, but no scalarized copy
  variant is correctness-clean. Move this axis toward a lower-level cooperative
  store/lane primitive or compact compiler reproducer.

## 2026-06-19 - Q8_0 group12 synthetic selected-stage minimization

- source:
  `sources/llama.cpp` at `84c947b71-dirty` extended
  `hrx-hip-bench-wmma-f16-lane-map` with
  `group12-selected-stage-contract` and
  `group12-selected-stage-contract-hi`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-group12-selected-stage-20260619-113828/`.
- purpose:
  test whether the real-Q8 group12 selected-stage failure can be minimized to
  WMMA plus the selected halfword LDS writeback contract without Q8
  dequant/model data.
- correctness:
  the synthetic low/opsel0 and high/opsel1 modes both passed with
  `active=256`, no duplicate/missing/unexpected stores, no NaNs, and
  `mismatch=0`. The same artifact reran the real
  `single-group12-abcopy-stage-selected` control, which still failed p64 with
  `active=256`, `bad=16`, max_abs `0.348336`, and the known col `50`/`57`
  bad-lane pattern.
- static evidence:
  extracted lane-map and Q8-repro gfx1151 objects are under
  `cache/hrxv1/gfx1151/wmma-f16-group12-selected-stage-20260619-113828/static/`.
  The synthetic selected-stage symbols are wave64/no-spill with LDS `11776`,
  SGPR `22`, VGPR `40`, and two WMMA. The real Q8 selected-stage group12
  symbols are wave64/no-spill with LDS `10752`, SGPR `41`, VGPR `117`, and two
  WMMA, preserving the Q8 load/dequant dependency surface that the synthetic
  minimization lacks.
- decision:
  reject the too-small minimization. WMMA plus selected halfword staging alone
  is not enough to reproduce the group12 failure. The next compact reproducer
  must preserve more of the real Q8 fragment/dequant/register dependency
  surface, or use a lower-level cooperative store/lane primitive that avoids
  the scalarized HIP C++ selected-stage writeback spelling.

## 2026-06-19 - Q8_0 group12 dual raw/stage order probe

- source:
  `sources/llama.cpp` at `09cf43a14-dirty` extended
  `hrx-hip-bench-q8-wmma-repro` with
  `single-group12-abcopy-dual-stage-raw-first` and
  `single-group12-abcopy-dual-stage-stage-first`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/q8-wmma-dual-stage-20260619-114444/`.
- purpose:
  preserve the real Q8 A+B-copy dependency surface and compare raw vs
  selected-stage writeback from the same group12 accumulator in one dispatch.
- correctness:
  raw control `single-group12-abcopy` passed p64 with `bad=0`, max_abs
  `0.00239494`. Standalone selected-stage control reproduced the known finite
  p64 failure with `bad=16`, max_abs `0.348336`. In the dual raw-first mode,
  raw stayed clean, while staged no longer crossed the `0.25` CPU threshold
  but mismatched raw on `60/256` active values, with staged max_abs `0.154939`
  and mismatch max_abs `0.156356`. In the dual stage-first mode, the source
  order became more intrusive: `raw_sentinel=128`, `staged_sentinel=64`, and
  `mismatch=44`.
- static evidence:
  extracted object, notes, disassembly, and per-symbol summaries are under
  `cache/hrxv1/gfx1151/q8-wmma-dual-stage-20260619-114444/static/`. The
  dual-stage symbols are wave64/no-spill with LDS `10752`, SGPR `34`, VGPR
  `117`, and two WMMA, preserving the real Q8 load/dequant dependency surface.
- decision:
  reject dual raw/stage consumption as a production workaround. This is strong
  evidence that the group12 selected-stage failure is register-lifetime and
  source-order sensitive. The next parity-directed step should stop varying
  scalarized HIP C++ staging around this helper and move to a lower-level
  cooperative store/lane primitive or compiler-facing reproducer with the real
  Q8 dependency surface.

## 2026-06-19 - Q8_0 Vulkan cooperative-store extract

- source:
  `sources/llama.cpp` at `87985ff41-dirty` extended
  `tools/vulkan-oracle/extract_coopmat_schedule.py` to emit compact
  store-window summaries.
- artifact:
  `cache/hrxv1/gfx1151/q8_0-coopmat-store-extract-20260619-115145/`.
- purpose:
  turn the current Vulkan p512 Q8_0 oracle into a reusable writeback contract
  before implementing another HIP route.
- evidence:
  the Vulkan source window confirms that full in-bounds aligned tiles cast the
  accumulator to a `D_TYPE` cooperative matrix and `coopMatStore` directly to
  `data_d`; `coopmat_stage` is used for unaligned stride or partial edge
  tiles. The RADV p512 kernel emits `32` WMMA sites, `192 buffer_store_b32`,
  `128 ds_store_b16`, `128 ds_load_u16_d16`, `64 ds_load_b64`, LDS `22528`,
  VGPR `192`, and no spills. The generated store windows expose the repeated
  output-store pattern after the initial LDS setup stores.
- decision:
  treat this as the Q8_0 p512 writeback contract. Future direct-WMMA work must
  mechanically explain how it reproduces the cooperative-matrix global-store
  ownership; otherwise it should be framed as a separate packed-Q8/dot4 pivot,
  not as a Vulkan coopmat clone.

## 2026-06-19 - Q8_0 coopstore contract sweep

- source:
  existing CMake-built `hrx-hip-bench-coopmat-store-contract` in
  `sources/llama.cpp` at `87985ff41-dirty`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- artifacts:
  `cache/hrxv1/gfx1151/coopmat-store-contract-recheck-20260619-115310/`,
  `cache/hrxv1/gfx1151/coopmat-store-contract-sweep-20260619-115359/`, and
  `cache/hrxv1/gfx1151/coopmat-store-contract-repeat-20260619-115655/`.
- purpose:
  bracket the next Q8_0 writeback primitive after the Vulkan extract showed
  that the aligned oracle path is cooperative-matrix direct global store, not
  scalar halfword LDS staging.
- correctness:
  direct raw paths passed: `linear192`, `branch192`,
  `wmma-lds-k2-direct64`, `wmma-lds-k2-direct160-raw`, and
  `wmma-lds-k2-direct192-raw` all reported `bad=0`. `wmma-lds-k2-mixed96`
  and `wmma-lds-k2-mixed128` also passed. The larger halfword-staged paths
  failed: `wmma-lds-k2-mixed160-lo` repeated `bad=1600`, and
  `wmma-lds-k2-radv-mixed192` repeated `bad=2112`. Non-WMMA `radv-mixed192`
  was flaky, failing 4/5 repeated runs at group 34 slot 2.
- static evidence:
  the sweep unbundled `.hip_fatbin` and wrote
  `static/fatbin.gfx1151.o` plus `static/symbol-summary.md`.
  `wmma-lds-k2-direct192-raw` is wave64/no-spill with SGPR `14`, VGPR `153`,
  LDS `16384`, `32` WMMA, `64 ds_load_b64`, and `192 buffer_store_b32`.
  The failing `wmma-lds-k2-radv-mixed192` is wave64/no-spill with SGPR `14`,
  VGPR `162`, LDS `32768`, `32` WMMA, `64 ds_load_b64`,
  `128 ds_store_b16`, `128 ds_load_u16_d16`, and `192 buffer_store_b32`.
- decision:
  the HIP compiler can express a correctness-clean direct 192-store WMMA
  surface. The large halfword-LDS output staging path is the hazard. The next
  production-directed Q8_0 probe should port direct192-style output ownership
  into the real Q8 WMMA dependency surface, then run focused CPU-reference rows
  before considering catalog routing.

## 2026-06-19 - Q8_0 real-Q8 full direct-store copy pivot

- source:
  `sources/llama.cpp` at `7a76f2cc4-dirty` added explicit
  `hrx-hip-bench-q8-wmma-repro` modes for a full 64x64 real-Q8 WMMA tile with
  direct raw `buffer_store_b32` output ownership: `array16-direct-raw`,
  `array16-direct-raw-bcopy`, and `array16-direct-raw-abcopy`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-q8-wmma-repro -j$(nproc)` using ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- artifacts:
  initial raw/control probe
  `cache/hrxv1/gfx1151/q8-wmma-array16-direct-raw-20260619-120319/`;
  copy-control confirmation
  `cache/hrxv1/gfx1151/q8-wmma-copy-controls-after-array16-20260619-120512/`;
  final pivot and static extraction
  `cache/hrxv1/gfx1151/q8-wmma-array16-copy-pivot-20260619-120609/`.
- purpose:
  port the synthetic `direct192_raw` lesson into the real Q8 load/dequant/WMMA
  dependency surface and test both aligned p64 and odd/tail p33 output
  ownership before any catalog route work.
- correctness:
  `array16-direct-raw` failed p64 and p33 in group 0 with NaNs and large finite
  error. `array16-direct-raw-bcopy` passed p33 but failed p64 in groups 12-15
  with `bad=912` and `nan=288`. `array16-direct-raw-abcopy` passed p64 and p33
  with `bad=0`, max_abs `0.00268994` and `0.00262882`, matching the existing
  `array8-fullb-2phase-abcopy` control. Copy controls confirmed
  `single-group8-abcopy`, `single-group12-abcopy`, and
  `array8-fullb-2phase-abcopy` remain clean in the same rebuilt binary.
- static evidence:
  `array16-direct-raw` emits wave64, SGPR `64`, VGPR `195`, no private memory,
  `32` WMMA, `64 ds_load_b64`, and `64 buffer_store_b32`.
  `array16-direct-raw-abcopy` emits the same WMMA/store count but reaches VGPR
  `256` and spills (`private_segment_fixed_size=84`, `20 scratch_load` and
  `20 scratch_store`). The no-spill `array8-fullb-2phase-abcopy` control emits
  two phase symbols with SGPR `64`, VGPR `247`, `16` WMMA and
  `32 buffer_store_b32` per phase.
- decision:
  accept the A+B-copy lifetime boundary as correctness evidence, but do not
  promote the single-dispatch `array16-direct-raw-abcopy` shape because it
  spills. The next production candidate should preserve A+B-copy semantics in a
  no-spill two-phase or lower-level store/lifetime spelling, then compare
  device time against the current HRX v1 Q8_0 catalog route and the Vulkan
  oracle.

## 2026-06-19 - Current-head full basket KPI after Q8 evidence commits

- source:
  `sources/llama.cpp` at commit `77e389e06`.
- build:
  rebuilt `build/hrx-v1-catalog-gfx1151` and `build/vulkan-gfx1151` through
  CMake/Ninja; both `llama-bench` binaries report build commit `77e389e06`.
- artifact:
  `cache/hrxv1/gfx1151/basket-current-head-77e389e06-20260619-121528/`.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --tag basket-current-head-77e389e06-20260619-121528 --models all --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --timeout 1200`.
- validation:
  all HRX rows report `backends=HRX`, all Vulkan rows report
  `backends=Vulkan`, and HRX fallback lines are zero for all rows.
- result:
  average geomean HRX/Vulkan `0.610x`; steady geomean HRX/Vulkan `0.609x`;
  `22/24` rows remain below parity.
- current worst steady rows:
  Llama 3.1 8B Q8_0 p512 `457.292 / 913.835 = 0.500x`;
  Qwen3 30B Q4_K_XL p512 `640.505 / 1238.080 = 0.517x`;
  Qwen2.5 Coder 7B Q5_K_M p512 `610.729 / 1175.300 = 0.520x`;
  Qwen3 30B Q6_K p33 `94.192 / 179.372 = 0.525x`;
  Qwen3 30B Q6_K p512 `551.976 / 1049.760 = 0.526x`.
- decision:
  keep Q8_0 p512 as the top single-row boulder, with Q4_K_XL/Q5/Q6 large
  prompt rows close enough that any Q8 work should be schedule-directed and
  focused. The current KPI does not change the rule: aggregate rows rank work;
  they do not promote routes.

## 2026-06-19 - Q8_0 streamrow A+B-copy catalog probe

- source:
  `sources/llama.cpp` at `77e389e06-dirty` added
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamrow_packstage_abcopy_bufferstore_f16acc_wg256_f32`
  as an opt-in route and taught the shared streamrow VK128 B64GROUP branch to
  honor the existing `COPY_A_FRAG`/`COPY_B_FRAG` macros.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx
  test-backend-ops -j$(nproc)` using ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifacts:
  static check `cache/hrxv1/gfx1151/q8-streamrow-abcopy-static-20260619-122823/`;
  focused gate `cache/hrxv1/gfx1151/q8-streamrow-abcopy-focused-20260619-122913/`.
- purpose:
  test the untried intersection between the RADV-static streamrow selected-half
  route and the A+B fragment materialization boundary that repaired the
  standalone real-Q8 direct-store repro.
- static evidence:
  the repaired candidate is wave64, SGPR `28`, VGPR `188`, LDS `20480`, no
  private memory, no spills, `32` WMMA, `64 ds_load_b64`, `2 ds_store_b32`,
  `128 buffer_store_b32`, and two barriers. It is statically viable and close
  to the prior streamrow selected route, but it remains a selected-half
  128-store route rather than the RADV 192-store cooperative writeback contract.
- correctness:
  p512 and p513 route traces selected the new provider only for the intended
  large Q8_0 rows while Vcur/Qcur stayed on packed Q8_1. p33 was guarded out
  and passed on existing narrow Q8_1 routes. The selected large rows failed
  strict CPU reference badly: p512 ffn_out `ERR=0.799129460`, ffn_gate
  `ERR=0.800699661`, result_output `ERR=0.800311620`; p513 ffn_out
  `ERR=0.798920265`, ffn_gate `ERR=0.800372515`, result_output
  `ERR=0.798270479`.
- decision:
  reject before perf/model tests. A+B materialization is not the missing repair
  for the streamrow selected-half production lane contract; it worsens the
  prior streamrow finite error. The next Q8 parity-directed path should target
  the true RADV cooperative writeback/store ownership or a lower-level
  lane/store primitive, not another streamrow selected-half fragment-copy
  spelling.

## 2026-06-19 - Q8_0 real-data contract192 store-surface probe

- source:
  `sources/llama.cpp` at `094e1d1b3-dirty` after adding
  `contract-direct192-abcopy` and `contract-phase96-abcopy` modes to
  `hrx-hip-bench-q8-wmma-repro`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja.
- model/shape:
  standalone real-Q8 repro rows `rows=64`, `k=4096`, `cols=64` and
  odd/narrow `cols=33`.
- route or kernel candidate:
  CMake-built HIP diagnostics for the Vulkan Q8_0 direct192 store contract:
  single-dispatch `contract-direct192-abcopy` and two-dispatch lifetime split
  `contract-phase96-abcopy`.
- baseline command:
  prior `array16-direct-raw-abcopy` and `array8-fullb-2phase-abcopy` evidence
  from `cache/hrxv1/gfx1151/q8-wmma-array16-copy-pivot-20260619-120609/`.
- variant command:
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q8-wmma-repro --mode contract-direct192-abcopy`
  and
  `build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q8-wmma-repro --mode contract-phase96-abcopy`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-contract192-repro-20260619-123924/`.
- correctness result:
  both modes passed CPU-reference validation for p64 and odd p33. The
  single-dispatch direct192 mode reported `bad=0`, no NaNs, infinities,
  sentinels, or unexpected inactive writes, with max absolute error
  `0.00268994` on cols=64 and `0.00262882` on cols=33. The phase96 split
  reported the same correctness facts.
- timing result:
  not run; this is a correctness/static resource boundary probe.
- static evidence:
  the executable fatbin was extracted directly because the sidecar device
  object was stale. `contract-direct192-abcopy` emits the target one-kernel
  surface with `32` WMMA, `64 ds_load_b64`, and `192 buffer_store_b32`, but
  reaches VGPR `256`, `private_segment_fixed_size=60`, and `14` VGPR spills.
  The two phase96 symbols each emit `16` WMMA, `64 ds_load_b64`, and
  `96 buffer_store_b32`, stay at VGPR `247`, and have no private segment or
  spills.
- decision:
  reject the single-dispatch direct192 shape as a production route because it
  spills, and treat the phase96 split as positive evidence rather than a
  promotion. The next Q8 candidate should either port the phase/lifetime split
  into the production ABI for measured A/B, or use a lower-level cooperative
  writeback primitive that keeps the true 192-store Vulkan surface without the
  HIP C++ VGPR cliff.

## 2026-06-19 - Q8_0 phase96 catalog-transfer rejection

- source:
  `sources/llama.cpp` at `cd7712cdc-dirty` with two opt-in catalog providers:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_phase96_abcopy_bufferstore_phase0_f16acc_wg256_f32`
  and `phase1`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops -j$(nproc)`
  using ROCm `/srv/vm-shared/rocm/rocm-head`; providers are compiled through
  CMake/Ninja and embedded through the JSON catalog.
- model/shape:
  Llama 3.1 8B Q8_0 focused rows, p512, odd/narrow p33, and tail p513.
- route or kernel candidate:
  two-dispatch production ABI transfer of the passing
  `contract-phase96-abcopy` standalone diagnostic. The runtime opt-in gate is
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PHASE96_ABCOPY_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q8-phase96-abcopy-focused-20260619-125841/`.
- route trace:
  p512 and p513 selected `phase0+phase1` only for the intended large rows;
  Vcur/Qcur stayed on packed Q8_1. p33 was guarded out and passed on the
  packed Q8_1 narrow route.
- correctness:
  reject before timing/model tests. p512 failed selected large rows with
  ffn_out `ERR=0.249924303`, ffn_gate `ERR=0.249939249`, and result_output
  `ERR=0.249768278`. p513 failed selected large rows with ffn_out
  `ERR=0.249700372`, ffn_gate `ERR=0.249900558`, and result_output
  `ERR=0.249852420`.
- static evidence:
  each production phase emits `16` WMMA and `64 ds_load_b64`, but only
  `32 buffer_store_b32`, with wave64, SGPR `50`, VGPR `179`, LDS `20480`, no
  private segment, and two barriers. A production-safe duplicate same-address
  store attempt was not accepted because the compiler collapsed it back to the
  same `32` static stores.
- decision:
  reject the production-safe phase96 catalog transfer. The key lesson is that
  the standalone `96 buffer_store_b32` phase success used synthetic side-effect
  stores to a separate contract buffer; a real-output-only catalog ABI transfer
  does not reproduce that surface and repeats the finite `~0.25` lane/accumulator
  contract failure. The next Q8 parity path should move to a lower-level
  cooperative writeback/lane-store primitive or an ABI that can safely carry a
  scratch side effect, not another HIP C++ real-output-only phase clone.

## 2026-06-19 - Q8_0 phase96 BM128 synthetic-vs-backend data probe

- source:
  `sources/llama.cpp` at `0207511b9-dirty`, adding
  `hrx-hip-bench-q8-wmma-repro --mode=phase96-bm128-abcopy`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target hrx-hip-bench-q8-wmma-repro -j$(nproc)`.
- purpose:
  isolate whether the rejected catalog transfer failed because the passing
  standalone phase96 diagnostic used one-wave `BM64/BN64`, while the production
  route uses four-wave `BM128/BN128` ownership.
- artifact:
  synthetic repro `cache/hrxv1/gfx1151/q8-phase96-bm128-repro-20260619-131659/`;
  backend-op forced-selector check
  `cache/hrxv1/gfx1151/q8-phase96-forceall-focused-20260619-131506/`.
- synthetic correctness:
  the new standalone BM128 phase pair passed rows `128` cols `128/129` k
  `4096`, rows `128` cols `33` k `4096`, rows `128` cols `128/129` k
  `14336`, and rows `512/1024` cols `128` k `4096`. Max absolute error was
  `0.00262882` to `0.00796381`, with no NaNs, infinities, or sentinels.
- backend-op diagnostic:
  a temporary uncommitted force-all selector routed the same catalog phase
  providers onto all p512 focused Q8_0 rows, including Vcur/Qcur. Route traces
  confirmed phase0+phase1 selection for every row. All five rows failed strict
  CPU reference with finite `~0.25` error: Vcur `0.247994866`, Qcur
  `0.250663819`, ffn_out `0.250380090`, ffn_gate `0.249577588`,
  result_output `0.250078394`.
- decision:
  reject the hypothesis that the phase96 failure is caused by BM128 four-wave
  ownership, large K, row tile index up to 1024, or p129/p33 edge handling in
  isolation. The remaining difference is backend-op/model-derived data
  distribution, exact CPU contract, or a layout assumption not exercised by the
  synthetic fill. Next Q8 work should either feed exported backend-op tensor
  data into the repro harness or move to a lower-level lane/writeback primitive
  that can be checked directly against backend-op data.

## 2026-06-19 - Q8_0 BM128 one-dispatch direct192 contract probe

- source:
  `sources/llama.cpp` at `0411af555-dirty`, adding standalone bench modes
  `contract-bm128-direct192-raw`, `contract-bm128-direct192-abcopy`, and
  `contract-bm128-direct192-abcopy-bhoist`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target hrx-hip-bench-q8-wmma-repro -j$(nproc)`.
- purpose:
  test the missing production-width one-dispatch Q8_0 WMMA surface directly:
  `BM128/BN128`, four wave64 quadrants, real Q8_0 dequant, f16 WMMA
  accumulation, and a full `192` static `buffer_store_b32` contract.
- artifact:
  `cache/hrxv1/gfx1151/q8-bm128-direct192-contract-20260619-141120/`.
- correctness:
  raw fails p128 and p33 with finite bad values (`bad=494/247`,
  `max_abs=7.57994`, first bad at group 16). A+B-copy passes p128 and p33
  with `max_abs=0.00268994/0.00262882`. A+B-copy with B-copy hoisted also
  passes p128 and p33 with the same max errors.
- static evidence:
  raw is the desired no-spill resource shape: wave64, SGPR `50`, VGPR `195`,
  LDS `20480`, private segment `0`, `32` WMMA, `64 ds_load_b64`, and
  `192 buffer_store_b32`. A+B-copy reaches VGPR `256`, private segment `112`,
  and `27` VGPR spills. A+B-copy/B-hoist improves pressure but still spills:
  VGPR `256`, private segment `80`, and `19` VGPR spills.
- decision:
  reject these as production route templates. This proves the production-width
  one-dispatch 192-store contract can be semantically correct in HIP C++ only
  after explicit fragment materialization, but the correct spelling still hits
  the HIP compiler pressure cliff. The raw no-spill shape remains the target,
  but its B-fragment lane contract is invalid. The next Q8 path should be a
  lower-level B-fragment/cooperative-store primitive or compact compiler-facing
  reduction that restores the raw no-spill contract without the invalid
  B-fragment mapping.

## 2026-06-19 - Q8_0 packstage inline-WMMA catalog transfer

- source:
  `sources/llama.cpp` at `9b4538af5-dirty`, adding the opt-in catalog route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_bufferstore_f16acc_wg256_f32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-ops hrx-hip-bench-q8-wmma-repro -j$(nproc)`.
- purpose:
  transfer the standalone BM128 inline-WMMA discovery into the real HRX v1
  catalog ABI. The hypothesis was that explicitly emitting
  `v_wmma_f16_16x16x16_f16` through inline asm would preserve the corrected B
  operand contract without the HIP builtin lane cliff.
- artifact:
  `cache/hrxv1/gfx1151/q8-asmwmma-packstage-wired-focused-20260619-143542/`.
- correctness and routing:
  focused p33, p512, and p513 CPU-reference gates passed. p33 stayed on the
  existing narrow packed route. p512 and p513 selected the asm-WMMA provider
  only for `ffn_out`, `ffn_gate`, and `result_output`; `Vcur` and `Qcur`
  stayed on packed Q8_1.
- static evidence:
  production HSACO is wave64, SGPR `28`, VGPR `212`, LDS `20480`, private
  segment `0`, no spills, `32` WMMA, `64 ds_load_b64`, `128 buffer_store_b32`,
  and two barriers. Compared with RADV p512 Q8_0 large route, it still misses
  `64` buffer stores, `128 ds_store_b16`, `128 ds_load_u16_d16`, and the
  `22528` byte LDS footprint.
- perf:
  same-runner focused timing rejected promotion: p512 total regressed
  `72948.716 -> 94465.939 us` (`1.295x`) and p513 total regressed
  `81603.554 -> 109759.971 us` (`1.345x`). Selected large rows were
  `1.29x-1.41x` slower on p512 and `1.33x-1.38x` slower on p513.
- decision:
  keep the provider opt-in only. Inline WMMA is positive ABI evidence because
  it fixes the production B operand contract without spills, but it is not a
  performance route. The remaining gap is the RADV cooperative halfword
  store/load/writeback topology, not merely HIP builtin WMMA operand lowering.

## 2026-06-19 - Q8_0 inline-WMMA plus fast-half split-selected catalog rejection

- source:
  `sources/llama.cpp` at `e7b542058-dirty`, adding the opt-in catalog route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_fast_half_split_selected_bufferstore_f16acc_wg256_f32`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml-hrx test-backend-ops -j$(nproc)`.
- purpose:
  cross the correctness-clean inline-WMMA B operand contract with the
  RADV-like fast-half split-selected halfword LDS output topology.
- static artifact:
  `cache/hrxv1/gfx1151/q8-asmwmma-fast-half-split-selected-static-20260619-145526/`.
- focused artifact:
  `cache/hrxv1/gfx1151/q8-asmwmma-fast-half-split-selected-focused-20260619-145643/`.
- static evidence:
  the route is wave64, SGPR `28`, VGPR `212`, LDS `22528`, no private segment
  or spills, `32` WMMA, `64 ds_load_b64`, `128 ds_store_b16`,
  `128 ds_load_u16_d16`, `128 buffer_store_b32`, `2 ds_store_b32`, and two
  barriers. It matches several RADV topology facts but still misses RADV's
  `192 buffer_store_b32` surface and first-WMMA issue window
  (`lgkmcnt=0` versus RADV `lgkmcnt=51`).
- correctness and routing:
  p33 was correctly guarded out and all five p33 rows passed on the existing
  narrow packed Q8_1 route. p512 and p513 selected this provider only for
  `ffn_out`, `ffn_gate`, and `result_output`; `Vcur` and `Qcur` stayed on
  packed Q8_1.
- result:
  reject before timing/model tests. Selected p512 rows failed with
  `ffn_out ERR=2.629715292`, `ffn_gate ERR=2.122694135`, and `result_output`
  NaN at index `6188064`. Selected p513 rows failed with
  `ffn_out ERR=2.519378912`, `ffn_gate ERR=2.249089220`, and `result_output`
  NaN at the same index.
- decision:
  keep opt-in only as negative evidence. Inline WMMA fixes the plain packstage
  B operand contract, but combining it with selected halfword LDS output
  staging does not reproduce Vulkan's cooperative-store lane ownership on real
  model-derived large rows.

## 2026-06-19 - Q8_0 RADV store ownership motif extract

- source:
  `sources/llama.cpp` after enhancing
  `tools/vulkan-oracle/extract_coopmat_schedule.py`.
- purpose:
  stop treating RADV's `192 buffer_store_b32` writeback as a single fullpair
  helper. Extract per-basic-block store motifs from the actual RADV ISA so the
  next HIP primitive has an exact lane/store target.
- large-route artifact:
  `cache/hrxv1/gfx1151/q8_0-coopmat-store-ownership-20260619-150308/`.
- p33 medium-route artifact:
  `cache/hrxv1/gfx1151/q8_0-coopmat-store-ownership-p33-20260619-150331/`.
- large p512/p513 motif result:
  RADV's large route decomposes its `192` global stores into three equal
  store families: `64` direct stores from 16 direct-four blocks, `64` stores
  from 16 staged-four blocks that also perform `64 ds_store_b16` and
  `64 ds_load_u16_d16`, and `64` scalar-reload stores from 64 one-store
  blocks. A separate 16 guarded-LDS-store blocks supply another
  `64 ds_store_b16`.
- p33 result:
  the medium route uses the same motif family at half scale:
  `96 buffer_store_b32`, `64 ds_store_b16`, `64 ds_load_u16_d16`, `16` WMMA,
  and `48 ds_load_b64`.
- decision:
  use this as the next Q8 store contract. A new cooperative-matrix clone should
  implement motif-level direct-four/staged-four/scalar-reload ownership, not a
  monolithic selected-half or fullpair writeback helper.

## 2026-06-19 - Q8_0 RADV motif192 HIP fixture

- source:
  `sources/llama.cpp` adding the CMake/Ninja-built diagnostic mode
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-radv-motif192`.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-radv-motif192-20260619-150932/`.
- purpose:
  test the motif-level store family extracted from RADV before spending another
  production route attempt: 16 direct accumulator groups plus two staged
  16-group halfword LDS reload/writeback motifs, with the second staged half
  reusing the same 16-group LDS window.
- controls:
  `wmma-lds-k2-direct192-raw` passed with `bad=0`, proving the direct 192-store
  output surface is not inherently broken. `wmma-lds-k2-mixed160-splitstage`
  also passed with `bad=0`, proving the halfword stage path is correct when it
  has the existing phase boundary.
- motif result:
  the first oversized 32-group LDS-stage allocation failed reproducibly with
  `bad=1792`, first at `group=32 slot=2 lane=0`, `actual=512`,
  `expected=16386`. Tightening the stage allocation to the actual reused
  16-group window made the same motif pass with `bad=0` and `max_abs=0`.
- static evidence:
  extracted HSACO confirms the fixture emitted the intended headline contract:
  wave64, SGPR `14`, VGPR `137`, group segment `24576`, private segment `0`,
  `32 v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`, `128 ds_store_b16`,
  `128 ds_load_u16_d16`, `192 buffer_store_b32`, `168 s_waitcnt`, and two
  barriers.
- decision:
  accept this as a standalone schedule prior, not yet as a production route.
  The critical lesson is that the RADV motif is sensitive to the exact LDS
  footprint/window as well as store counts: the oversized same-source shape
  corrupts, while the tight reused window passes. The next Q8_0 route should
  port this tight motif into the real catalog ABI and keep the p33 half-scale
  variant separate.

## 2026-06-19 - Q8_0 motif192 catalog transfer rejection

- source:
  `sources/llama.cpp` opt-in route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_bufferstore_f16acc_wg256_f32`.
- artifact:
  `cache/hrxv1/gfx1151/q8-asmwmma-motif192-focused-20260619-153414/`.
- purpose:
  port the tight standalone RADV motif192 fixture into the real Q8_0 catalog
  ABI with CMake/Ninja HSACO compilation and explicit HRX v1 runtime provider
  wiring.
- static evidence:
  the production HSACO preserves the headline motif: wave64, LDS `22528`,
  no private segment, no spills, `32` WMMA, `64 ds_load_b64`,
  `128 ds_store_b16`, `128 ds_load_u16_d16`, and `192 buffer_store_b32`.
  It is not the same compiler surface as the fixture: SGPR/VGPR rise to
  `56/212`, waitcnts drop to `60`, and barriers rise to `3`.
- focused evidence:
  p33 stayed guarded out and passed all five rows on the existing narrow
  Q8_1 x4 route. p512 selected the motif provider after `Vcur`/`Qcur` stayed
  on Q8_1 x4 split-qsum; the first selected wide row, `ffn_out`, faulted the
  GPU with a page-not-present/supervisor-privilege memory access fault before
  CSV output.
- decision:
  reject before timing/model tests. The tight standalone motif remains a
  useful schedule prior, but the full catalog ABI transfer has an unsafe
  output/LDS/addressing contract. Do not use this route for performance work
  until the memory fault is reduced to a smaller repro.

## 2026-06-19 - Q8_0 motif192 address repro

- source:
  `sources/llama.cpp` CMake/Ninja-built diagnostic modes added to
  `hrx-hip-bench-q8-wmma-repro`.
- artifacts:
  `cache/hrxv1/gfx1151/q8-motif192-synth-address-20260619-154442/` and
  `cache/hrxv1/gfx1151/q8-motif192-wmma-address-linebuf-20260619-154833/`.
- purpose:
  reduce the motif192 production fault outside HRX runtime dispatch and model
  data while preserving the real BM128/BN128 row/column output address formula.
- result:
  the synthetic arbitrary-accumulator payload corrupts aligned and odd shapes
  but does not fault. The WMMA-payload variant executes the small aligned and
  odd shapes, then reproduces a GPU memory fault on the 4096x512 p512-style
  dispatch with `grid=[8192,4,1]`, `group_seg_size=22528`, and
  `private_seg_size=0`.
- decision:
  the motif192 failure is now reduced to a standalone CMake-built HIP kernel:
  WMMA payload plus the real four-wave row/column motif address shape is enough
  to fault. The next step should be a tighter store primitive/lane ownership
  reduction, not another catalog route timing attempt.

## 2026-06-19 - Q8_0 motif192 staged-window isolation

- source:
  `sources/llama.cpp` CMake/Ninja-built diagnostic submodes added to
  `hrx-hip-bench-q8-wmma-repro`.
- artifacts:
  `cache/hrxv1/gfx1151/q8-motif192-wmma-direct-address-20260619-155416/`,
  `cache/hrxv1/gfx1151/q8-motif192-wmma-stage16-address-20260619-155431/`,
  and
  `cache/hrxv1/gfx1151/q8-motif192-wmma-stage32-address-20260619-155447/`.
- purpose:
  split the full motif into raw direct stores, the first halfword LDS
  stage/reload/writeback window, and the second reused halfword LDS
  stage/reload/writeback window while preserving the same WMMA payload and
  BM128/BN128 row/column address shape.
- result:
  direct-only runs through 4096x513 without a GPU fault. Both staged windows
  independently fault at the 4096x512 dispatch with the same `grid=[8192,4,1]`,
  `group_seg_size=22528`, and `private_seg_size=0` surface.
- decision:
  the p512 fault is now isolated to the staged halfword LDS reload/writeback
  contract under real row/column addressing. The next candidate should change
  that primitive or lane ownership directly; direct raw stores alone are not the
  fault trigger.

## 2026-06-19 - Q8_0 motif192 exact-shape threshold sweep

- source:
  `sources/llama.cpp` added `--rows` and `--cols` controls for the motif
  address repro modes.
- artifacts:
  `cache/hrxv1/gfx1151/q8-motif192-stage-threshold-20260619-155915/` and
  `cache/hrxv1/gfx1151/q8-motif192-stage-threshold-repeat-20260619-155951/`.
- purpose:
  determine whether the staged halfword writeback fault has a simple row-count
  threshold or whether it is sensitive to allocation/scheduler state.
- result:
  the first sweep showed stage16 faults at 1024 and 4096 rows but not
  intermediate row counts; stage32 faulted at 4096. The repeat sweep confirmed
  instability rather than a clean threshold: direct-only had no faults across
  repeated 1024/1536/4096 controls, while staged modes faulted intermittently at
  1024 and/or 4096.
- decision:
  treat the staged halfword writeback path as unsafe even when a given shape
  happens not to fault. The next Q8 large-route work should alter the staged
  primitive or lane ownership itself instead of relying on a shape guard.

## 2026-06-19 - Q8_0 motif192 wait-after-load fix

- source:
  `sources/llama.cpp` added wait-after-load motif repro modes and revised the
  opt-in production motif route to wait after `ds_read_u16_d16` before using
  the staged halfword value.
- artifacts:
  `cache/hrxv1/gfx1151/q8-motif192-stage-waitload-20260619-160737/`,
  `cache/hrxv1/gfx1151/q8-motif192-full-waitload-20260619-161019/`,
  `cache/hrxv1/gfx1151/q8-asmwmma-motif192-waitload-focused-20260619-161223/`,
  and
  `cache/hrxv1/gfx1151/q8-asmwmma-motif192-waitload-perf-20260619-161300/`.
- result:
  the wait-after-load spelling fixes the staged-window repros and the full
  motif repro across repeated aligned, odd-column, and odd-row shapes. The
  production route then passes p33, p512, and p513 focused CPU-reference gates.
  Route traces show p33 remains on existing narrow routes, while p512 and p513
  select the motif provider for `ffn_out`, `ffn_gate`, and `result_output`.
- timing:
  same-runner focused perf rejects promotion. p512 regresses `1.278x` total
  versus the current default; selected wide rows regress `1.431x` on
  `ffn_out`, `1.313x` on `ffn_gate`, and `1.266x` on `result_output`. p513
  regresses `1.309x` total; selected wide rows regress `1.279x`, `1.373x`,
  and `1.318x` respectively.
- decision:
  keep the route opt-in as a correctness-clean RADV-motif transfer and as proof
  that the prior GPU fault was a missing wait-after-LDS-load contract. Do not
  default it. The next Q8 large-route work needs to preserve this contract while
  recovering the throughput lost to the explicit waits/high HIP register
  surface.

## 2026-06-19 - Q8_0 motif192 K2 waitload issue-window repro

- source:
  `sources/llama.cpp` added CMake/Ninja-built `hrx-hip-bench-q8-wmma-repro`
  modes `motif192-wmma-k2-directwait-waitload-address` and
  `motif192-wmma-k2-depwait-waitload-address`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-waitload-20260619-162840/`.
- purpose:
  mechanically test the remaining RADV/HIP delta after the wait-after-load
  motif fix: load both VK128 K tiles before WMMA and issue the RADV-like
  `lgkmcnt` ladder, while preserving the corrected motif192 halfword
  wait-after-load writeback.
- correctness:
  both directwait and depwait modes passed `bad=0`, `nan=0`, `inf=0`, and
  `sentinel=0` for `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`,
  and `4096x513`.
- static evidence:
  both template instantiations were extracted from the executable fatbin and
  compared against the Llama 3.1 8B Q8_0 p512 RADV large oracle. Both emit
  wave64, SGPR `62`, VGPR `190`, LDS `22528`, private segment `0`, no spills,
  `64 ds_load_b64`, `32 v_wmma_f16_16x16x16_f16`, `132 ds_store_b16`,
  `128 ds_load_u16_d16`, `192 buffer_store_b32`, and two barriers. The
  first-window score is `64/64/lgkmcnt(51)` with the RADV-style wait ladder
  and all 32 WMMAs in the hot-op window.
- interpretation:
  this is the first corrected motif192 artifact that matches or exceeds the
  RADV large-route issue-window signature without the earlier K2 production
  spill cliff. The result is still synthetic A/B data in a standalone bench,
  not a production route. The production kernel currently makes the motif192
  and K2 branches mutually exclusive, so the next promotion step must be a new
  opt-in catalog branch/wrapper or a specialized real-Q8 repro that preserves
  this exact issue-window and writeback contract.

## 2026-06-19 - Q8_0 motif192 K2 real-data catalog transfer

- source:
  `sources/llama.cpp` added the opt-in route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_directwait_bufferstore_f16acc_wg256_f32`
  and built it through CMake/Ninja.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-20260619-163905/`.
- static result:
  the real-data catalog HSACO keeps the headline motif counts versus the RADV
  Q8_0 p512 large oracle: LDS `22528`, `64 ds_load_b64`, `32 v_wmma`, `128
  ds_store_b16`, `128 ds_load_u16_d16`, and `192 buffer_store_b32`. It does not
  preserve the standalone no-spill K2 contract: the catalog ABI compiles at
  VGPR `256`, private segment `68`, and `16` VGPR spills.
- correctness and routes:
  focused p33, p512, and p513 CPU-reference gates passed. p33 stayed on the
  existing narrow/default routes. p512 and p513 selected the K2 motif provider
  only for `ffn_out`, `ffn_gate`, and `result_output`.
- timing:
  focused same-runner perf rejects promotion. p512 is `1.845x` default time and
  p513 is `1.924x` default time. Selected wide rows are `1.811x-2.327x`
  slower.
- decision:
  keep the route opt-in as correctness-clean real-data spill-cliff evidence,
  but do not default it. The next Q8 candidate needs to preserve the standalone
  no-spill K2 motif shape or reduce the real-data ABI pressure before timing.

## 2026-06-19 - Q8_0 motif192 K2 real-data K32 fixture

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` mode
  `motif192-wmma-k2-realdata-k32-directwait-waitload-address`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-k32-20260619-165112/`.
- purpose:
  isolate whether real Q8_0 dequant and real F32 RHS values are enough to
  trigger the K2 spill cliff, while preserving the tight standalone motif
  source shape and limiting the fixture to one `BK=32` tile.
- correctness:
  passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513` with `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`. Worst max
  error was `0.0135875`; NMSE stayed around `4.3e-7`.
- static evidence:
  the extracted gfx1151 object reports wave64, SGPR `60`, VGPR `193`, LDS
  `22528`, private segment `0`, no spills, `64 ds_load_b64`, `32 v_wmma`,
  `130 ds_store_b16`, `128 ds_load_u16_d16`, `192 buffer_store_b32`, two
  barriers, and final pre-WMMA `lgkmcnt(51)`.
- interpretation:
  real Q8/RHS payloads are not the source of the production K2 pressure cliff.
  The no-spill shape survives in a single-BK real-data fixture. The catalog
  transfer's VGPR `256` and spills come from the full production ABI/loop/store
  surface. Next probes should add that surface back one axis at a time before
  another catalog route is attempted.

## 2026-06-19 - Q8_0 motif192 K2 real-data full-K fixture

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` mode
  `motif192-wmma-k2-realdata-fullk-directwait-waitload-address`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-20260619-165821/`.
- purpose:
  add back the full `k=4096` loop while preserving the tight standalone K2
  motif and corrected wait-after-load writeback, to test whether the loop
  surface alone causes the production spill cliff.
- correctness:
  passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513` with sampled checking on large rows. All rows had `bad=0`,
  `nan=0`, `inf=0`, and `sentinel=0`; worst sampled max error was `1.0958`.
- static evidence:
  the extracted gfx1151 object reports wave64, SGPR `60`, VGPR `256`, LDS
  `22528`, private segment `64`, `15` VGPR spills, `64 ds_load_b64`, `32
  v_wmma`, `130 ds_store_b16`, `128 ds_load_u16_d16`, `192 buffer_store_b32`,
  and first-loop pre-WMMA `lgkmcnt(51)`.
- interpretation:
  the full-K loop alone recreates the catalog pressure cliff. The next Q8 probe
  should target loop-carried accumulator and fragment lifetime, not catalog
  argument ABI. A viable path needs to split/lower the loop lifetime while
  retaining the RADV-like first-window load/wait/store facts.

## 2026-06-19 - Q8_0 motif192 K2 real-data full-K phase8 fixture

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` mode
  `motif192-wmma-k2-realdata-fullk-phase8-directwait-waitload-address`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-phase8-20260619-170420/`.
- purpose:
  test the active accumulator-lifetime hypothesis by splitting the full
  16-group output tile into two full-K launches with eight accumulators each.
- correctness:
  passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513` with sampled checking on large rows. All rows had `bad=0`,
  `nan=0`, `inf=0`, and `sentinel=0`; worst sampled max error was `1.0958`.
- static evidence:
  both phase kernels compile wave64, SGPR `58`, VGPR `211`, LDS `22528`,
  private segment `0`, and no spills. Each phase emits `64 ds_load_b64`, `16
  v_wmma`, `66 ds_store_b16`, `64 ds_load_u16_d16`, and `96 buffer_store_b32`.
  The lower phase preserves first pre-WMMA `lgkmcnt(51)`; the upper phase uses
  `lgkmcnt(20)`.
- interpretation:
  accumulator phasing removes the full-K spill cliff while keeping the
  lower-half RADV-like load/wait window. This is now the most plausible Q8
  production candidate family, but it still needs timing and/or catalog
  transfer evidence because the fixture pays two launches and duplicates some
  fragment loading.

## 2026-06-19 - Q8_0 motif192 K2 full-K phase8 timing

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` mode
  `motif192-wmma-k2-realdata-fullk-timing`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-fullk-phase8-timing-20260619-170826/`.
- purpose:
  time the spilling full-K 16-group fixture against the no-spill two-launch
  phase8 fixture in the same process with the same real-data inputs.
- measurement note:
  HIP events returned zero on this ROCm build. The accepted timing uses
  synchronized host wall-clock timing, which includes launch overhead. That is
  relevant here because the phase8 fixture currently pays two launches.
- timing:
  repeated wall-clock runs were stable. The accepted rerun reported
  `128x128x4096` at `0.999 ms` full-K versus `1.798 ms` phase8 (`1.80x`),
  `1024x512x4096` at `1.090 ms` versus `1.842 ms` (`1.69x`),
  `4096x512x4096` at `5.024 ms` versus `6.007 ms` (`1.20x`), and
  `4096x513x4096` at `5.101 ms` versus `6.121 ms` (`1.20x`).
- interpretation:
  no-spill is not sufficient. Two-launch phase8 removes private memory and
  preserves the lower-phase RADV-like wait window, but duplicated K-loop work
  plus launch overhead make it slower than the spilling full-K fixture in the
  same runner.
- decision:
  reject two-launch phase8 as a production route shape. Keep the accumulator
  lifetime result: reducing the live accumulator set fixes the compiler cliff.
  The next useful Q8 probe is a single-launch sequential phase shape or a
  lower-level spelling that keeps at most eight accumulators live without
  doubling provider submissions. It must preserve the no-spill resource row and
  beat the full-K fixture in the same timing harness before catalog work.

## 2026-06-19 - Q8_0 motif192 K2 full-K phase8seq fixture

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` modes
  `motif192-wmma-k2-realdata-fullk-phase8seq-directwait-waitload-address` and
  `motif192-wmma-k2-realdata-fullk-phase8seq-timing`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-phase8seq-20260619-171619/`.
- purpose:
  test the direct follow-up to two-launch phase8: keep the live accumulator set
  scoped to eight outputs, but run lower and upper phases sequentially inside
  one kernel launch.
- correctness:
  passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513` with sampled checking on large rows. All rows had `bad=0`,
  `nan=0`, `inf=0`, and `sentinel=0`; worst sampled max error was `1.0958`.
- static evidence:
  the extracted gfx1151 device object reports wave64, SGPR `76`, VGPR `231`,
  LDS `22528`, private segment `0`, no spills, `128 ds_load_b64`, `32
  v_wmma`, `132 ds_store_b16`, `128 ds_load_u16_d16`, `192 buffer_store_b32`,
  six barriers, and first pre-WMMA `lgkmcnt(51)`.
- timing:
  repeated same-runner timing rejected the shape. The accepted rerun reported
  `128x128x4096` at `1.000 ms` full-K versus `1.806 ms` phase8seq (`1.81x`),
  `1024x512x4096` at `1.090 ms` versus `1.839 ms` (`1.69x`),
  `4096x512x4096` at `4.932 ms` versus `6.183 ms` (`1.25x`), and
  `4096x513x4096` at `5.068 ms` versus `6.416 ms` (`1.27x`).
- interpretation:
  the HIP compiler honors the scoped eight-accumulator lifetime inside one
  kernel and removes spills, but replaying the whole K loop for each half of
  the output tile still costs too much.
- decision:
  reject phase8seq as a production route shape. The next Q8 probe should keep
  one K traversal and target only compiler-visible accumulator, fragment, or
  writeback lifetime, rather than splitting the whole output tile into full-K
  phases.

## 2026-06-19 - Q8_0 motif192 K2 full-K accpark fixture

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` modes
  `motif192-wmma-k2-realdata-fullk-accpark-directwait-waitload-address` and
  `motif192-wmma-k2-realdata-fullk-accpark-timing`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-accpark-20260619-172434/`.
- purpose:
  keep one K traversal and one launch, but park selected accumulator lanes in
  explicit LDS storage between BK chunks to shorten compiler-visible
  accumulator lifetime without replaying the K loop.
- correctness:
  failed every exact/odd/tail row. `128x128` reported `bad=5010/8192`,
  `max_abs=2059.88`, and `nmse=46.4751`; `4096x512` reported
  `bad=3687/8192`, `max_abs=580.928`, and `nmse=20.0988`. There were no
  NaNs, Infs, or sentinels, so this is a wrong accumulator-state contract.
- static evidence:
  the extracted gfx1151 device object reports wave64, SGPR `94`, VGPR `256`,
  LDS `55296`, private segment `72`, `17` VGPR spills, `64 ds_load_b64`,
  `384 ds_load_u16_d16`, `194 ds_store_b16`, `32 v_wmma`, and `192
  buffer_store_b32`.
- timing:
  not run because the correctness gate failed.
- interpretation:
  selected OPSEL lanes are not sufficient to preserve the WMMA accumulator
  across K chunks. The non-selected half of the `_Float16x8` accumulator vector
  matters to subsequent WMMA updates, even though final output stores only the
  selected OPSEL lane. The LDS parking shape also exceeds the RADV-like LDS
  budget and still spills.
- decision:
  reject selected-lane accpark. Do not continue this path unless the candidate
  preserves the full accumulator vector and has a credible LDS/occupancy plan.
  The next Q8 probe should move toward lower-level one-traversal full-acc
  spelling or a narrower output-ownership shape, not partial accumulator
  parking.

## 2026-06-19 - Q8_0 motif192 K2 full-K StreamFrag fixture

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` modes
  `motif192-wmma-k2-realdata-fullk-streamfrag-directwait-waitload-address` and
  `motif192-wmma-k2-realdata-fullk-streamfrag-timing`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-streamfrag-20260619-173258/`.
- purpose:
  keep one K traversal, one launch, and all sixteen output accumulators, but
  shorten compiler-visible fragment lifetime by loading each A/B WMMA fragment
  pair immediately before its `v_wmma` instead of materializing
  `a_frag[2][4]` and `b_frag[2][4]` arrays for the whole ladder.
- correctness:
  passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`; all rows reported `bad=0`, `nan=0`, `inf=0`, and
  `sentinel=0`. Worst sampled max absolute error was `1.0958`, matching the
  prior full-K tolerance regime.
- static evidence:
  the extracted gfx1151 device object reports wave64, SGPR `60`, VGPR `168`,
  LDS `22528`, private segment `0`, and no spills. It emits `256
  ds_load_b64`, `32 v_wmma`, `130 ds_store_b16`, `128 ds_load_u16_d16`, and
  `192 buffer_store_b32`. The first hot window is no longer RADV-like because
  each fragment pair is serialized with `lgkmcnt(0)`.
- timing:
  repeated same-runner timing showed a split result. The accepted rerun
  reported `128x128x4096` at `1.000 ms` full-K versus `1.160 ms` StreamFrag
  (`1.16x`), `1024x512x4096` at `1.094 ms` versus `1.263 ms` (`1.15x`),
  `4096x512x4096` at `4.980 ms` versus `4.060 ms` (`0.82x`), and
  `4096x513x4096` at `5.166 ms` versus `4.018 ms` (`0.78x`).
- interpretation:
  fragment lifetime is a real part of the full-K spill cliff. StreamFrag gives
  up the RADV-like pre-WMMA load window, but removing fragment pressure beats
  the spilling full-K fixture on wide production-width rows. It regresses
  narrow rows and must not become a universal route.
- decision:
  accept StreamFrag as the next production-facing Q8 direct-WMMA candidate
  family for wide p512/p513-style rows only. Transfer it to the catalog behind
  an opt-in selector, then compare against the current packed-Q8_1 default with
  focused p512/p513 correctness, route traces, static HSACO evidence, and
  same-runner timing before any default promotion.

## 2026-06-19 - Q8_0 motif192 K2 StreamFrag catalog transfer

- source:
  `sources/llama.cpp` added opt-in catalog route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_streamfrag_bufferstore_f16acc_wg256_f32`.
- env:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_ASMWMMA_MOTIF192_K2_STREAMFRAG_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-streamfrag-focused-20260619-174433/`.
- correctness:
  passed focused CPU-reference gates for p33, p512, and p513. Route traces
  proved p33 stayed on existing narrow packed-Q8_1 routes, while p512 and p513
  selected StreamFrag only for `ffn_out`, `ffn_gate`, and `result_output`;
  `Vcur` and `Qcur` stayed on the packed-Q8_1 split-qsum provider.
- static evidence:
  built HSACO reports wave64, SGPR `56`, VGPR `168`, LDS `22528`, private
  segment `0`, no spills, `32 v_wmma`, `256 ds_load_b64`, `128
  ds_store_b16`, `128 ds_load_u16_d16`, and `192 buffer_store_b32`.
- timing:
  same-runner focused p512 total regressed `2.058x` versus default. Selected
  rows regressed: `ffn_gate 2.217x`, `ffn_out 2.003x`, and `result_output
  2.090x`. p513 total regressed `2.151x`; selected rows regressed
  `ffn_gate 2.142x`, `ffn_out 1.809x`, and `result_output 2.241x`.
- decision:
  reject for promotion. Keep the route opt-in as no-spill fragment-lifetime
  evidence only. The catalog transfer proves spill removal is not sufficient
  when every fragment pair is serialized behind `lgkmcnt(0)`; the next Q8 path
  needs to recover a RADV-like pre-WMMA issue window or use a lower-level
  lane/writeback primitive.

## 2026-06-19 - Q8_0 motif192 K2 KTileFrag fixture and catalog transfer

- source:
  `sources/llama.cpp` added CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` modes
  `motif192-wmma-k2-realdata-fullk-ktilefrag-directwait-waitload-address` and
  `motif192-wmma-k2-realdata-fullk-ktilefrag-timing`, then transferred the
  same schedule into opt-in catalog route
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_ktilefrag_bufferstore_f16acc_wg256_f32`.
- artifacts:
  bench fixture:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-ktilefrag-20260619-175606/`;
  focused catalog gate:
  `cache/hrxv1/gfx1151/q8-motif192-k2-ktilefrag-focused-20260619-180304/`.
- purpose:
  bracket the middle schedule between spilling full-K fragment retention and
  serialized StreamFrag. KTileFrag loads one K tile's four A plus four B
  fragments, issues that tile's 16 WMMA updates with delayed waits
  `12/8/4/0`, then repeats for the second K tile.
- fixture correctness:
  passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`; all rows had `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`.
- fixture static evidence:
  extracted gfx1151 object reports wave64, SGPR `60`, VGPR `211`, LDS
  `22528`, private segment `0`, no spills, `64 ds_load_b64`, `32 v_wmma`,
  `130 ds_store_b16`, `128 ds_load_u16_d16`, and `192 buffer_store_b32`.
  The intended `12/8/4/0` wait window survived in the hot ladder.
- fixture timing:
  repeated same-runner timing beat the spilling full-K fixture on every row:
  `128x128` about `0.923x`, `1024x512` about `0.882x`, `4096x512` about
  `0.655x`, and `4096x513` about `0.685x` of full-K time.
- catalog static evidence:
  built HSACO reports wave64, SGPR `56`, VGPR `212`, LDS `22528`, private
  segment `0`, no spills, `64 ds_load_b64`, `32 v_wmma`, `128 ds_store_b16`,
  `128 ds_load_u16_d16`, and `192 buffer_store_b32`.
- catalog correctness/routes:
  focused CPU-reference gates passed for p33, p512, and p513. Route traces
  proved p33 stayed on existing narrow packed-Q8_1 routes, while p512 and p513
  selected KTileFrag only for `ffn_out`, `ffn_gate`, and `result_output`;
  `Vcur` and `Qcur` stayed on packed-Q8_1 split-qsum.
- catalog timing:
  rejected versus the current packed-Q8_1 default. p512 total regressed
  `1.272x`; selected rows regressed `ffn_gate 1.289x`, `ffn_out 1.414x`,
  and `result_output 1.263x`. p513 total regressed `1.335x`; selected rows
  regressed `ffn_gate 1.374x`, `ffn_out 1.370x`, and `result_output 1.339x`.
- decision:
  reject for promotion and keep opt-in. This is still positive schedule
  evidence: HIP C++ can preserve a no-spill delayed-wait direct-WMMA schedule
  when fragment lifetime is scoped per K tile. It is not enough to beat the
  packed-Q8_1 policy, so the next Q8 parity move should either improve the
  packed-Q8_1 dataflow toward the Vulkan large family or drop below this HIP
  C++ direct-WMMA store/load abstraction.

## 2026-06-19 - Q8_0 phase96 direct-wait inline-WMMA catalog probe

- source:
  `sources/llama.cpp` added CMake/Ninja-built opt-in phase providers
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_phase96_directwait_asmwmma_bufferstore_phase0_f16acc_wg256_f32`
  and `phase1`.
- env:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PHASE96_DIRECTWAIT_ASMWMMA_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- artifacts:
  static compile:
  `cache/hrxv1/gfx1151/q8_0-phase96-directwait-asmwmma-compile-20260619-183436/`;
  focused route/correctness:
  `cache/hrxv1/gfx1151/q8-phase96-directwait-asmwmma-focused-20260619-183828/`.
- purpose:
  bracket the K2 inline-WMMA result by reducing live accumulators/fragments
  with a two-phase output split while preserving direct nowait B64 LDS loads
  and explicit inline-WMMA wait scheduling.
- static evidence:
  each phase is wave64, SGPR `50`, VGPR `163`, LDS `20480`, private segment
  `0`, no scratch references, `16 v_wmma`, `64 ds_load_b64`, and `32
  buffer_store_b32`. The hot ladder preserved a smaller phase-local
  `32`-load/`8`-WMMA window with waits `31/27/23/19/20/16/12/8`, but not the
  full RADV `64`-load/`32`-WMMA/`192`-store surface.
- correctness/routes:
  p33 passed and stayed on existing narrow routes. p512 and p513 selected the
  phase0+phase1 route only for the intended large rows.
- focused failures:
  p512 failed `ffn_out ERR=0.924906869`, `ffn_gate ERR=0.738921369`, and
  `result_output ERR=0.295680974`; p513 failed `ffn_out ERR=0.940868841`,
  `ffn_gate ERR=0.588860476`, and `result_output ERR=0.218003867`.
- decision:
  reject before timing/model tests. This is a useful negative bracket:
  spill-free inline-WMMA phase splitting is not enough to repair the
  production lane/writeback contract. The next Q8 parity move should target
  lower-level RADV cooperative store/lane ownership or a packed-Q8_1 dataflow
  improvement, not another scalarized phase96 clone.

## 2026-06-19 - Q8_0 BN128 split-qsum BK2 focused win, model A/B rejection

- source:
  `sources/llama.cpp` dirty after adding
  `mul_mat_vec_q8_0_mmq64x128_splitqsum_bk2.hip.cpp`, CMake/catalog entries,
  and opt-in selector
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_BK2_PROMPT=1`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, candidate built through CMake/Ninja.
- model/shape:
  Llama 3.1 8B Q8_0 p512/fa1 production-width row, with p33 and p513
  odd/tail guards.
- route or kernel candidate:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bk2_wg256_f32`.
- baseline command:
  current default BN128 split-qsum route via `test-backend-ops perf` and
  `llama-bench -p 512 -n 0 -b 512 -ub 512 -fa 1 -r 5`.
- variant command:
  same commands with
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_BK2_PROMPT=1`.
- route trace path:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-bk2-focused-20260619-184940/`
  and
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-bk2-model-ab-20260619-185116/`.
- profile or timing artifact path:
  focused perf summary:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-bk2-focused-20260619-184940/perf-summary.md`;
  model summary:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-bk2-model-ab-20260619-185116/summary.md`.
- correctness result:
  p33, p512, and p513 CPU-reference gates passed. p512 selected BK2 for all
  five Q8_0 rows; p33 stayed on BN64; p513 stayed on BN112.
- timing result:
  focused p512 improved `72626.813 -> 71171.571 us` (`1.020x`), but model
  A/B rejected: baseline `458.964864 tok/s`, BK2 `457.465526 tok/s`
  (`0.9967x`).
- static evidence:
  target symbol wave32, SGPR `34`, VGPR `173`, LDS `8704`, private segment
  `0`, no spills.
- decision:
  reject for default promotion; keep opt-in as a useful packed-path staging
  depth bracket. The focused/model mismatch argues that this is too small a
  local schedule win to matter at production level, and the next Q8_0 parity
  move should return to the larger RADV cooperative store/lane ownership gap or
  a more structural packed-route change.

## 2026-06-19 - Q8_0 motif192 K2 full-vector half-tile accpark rejection

- source:
  `sources/llama.cpp` dirty after adding CMake/Ninja-built
  `hrx-hip-bench-q8-wmma-repro` modes
  `motif192-wmma-k2-realdata-fullk-accparkfull8-directwait-waitload-address`
  and `motif192-wmma-k2-realdata-fullk-accparkfull8-timing`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, built through CMake/Ninja target
  `hrx-hip-bench-q8-wmma-repro`.
- model/shape:
  standalone real-Q8 BM128/BN128 Q8_0 rows `128x128`, `128x129`,
  `129x128`, `1024x512`, `4096x512`, and `4096x513`, all at `k=4096`.
- route or kernel candidate:
  `q8_motif192_wmma_k2_realdata_fullk_accparkfull8_store_kernel`, a
  one-launch/one-K-traversal diagnostic that parks all eight accumulator
  elements for the lower eight output groups in LDS while keeping the upper
  eight groups resident.
- baseline command:
  prior selected-lane accpark mode
  `hrx-hip-bench-q8-wmma-repro --mode motif192-wmma-k2-realdata-fullk-accpark-directwait-waitload-address`.
- variant command:
  `hrx-hip-bench-q8-wmma-repro --mode motif192-wmma-k2-realdata-fullk-accparkfull8-directwait-waitload-address`.
- route trace path:
  not applicable; standalone HIP diagnostic.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-accparkfull8-20260619-190341/`.
- correctness result:
  failed all exact, odd, wide, and tail rows with finite bad values and no
  NaNs, infinities, or sentinels. Representative rows: `128x128`
  `bad=2525/8192`, `max_abs=13086.7`, `nmse=2329.39`; `4096x512`
  `bad=808/8192`, `max_abs=583.445`, `nmse=8.7801`; `4096x513`
  `bad=878/8208`, `max_abs=570.87`, `nmse=12.798`.
- timing result:
  not run because correctness and static gates failed.
- static evidence:
  extracted from the rebuilt executable's embedded `.hip_fatbin`, not the stale
  sidecar device object. The accparkfull8 symbol reports wave64, SGPR `90`,
  VGPR `256`, LDS `55296`, private segment `388`, and `152` VGPR spills. It
  preserves `32 v_wmma`, `64 ds_load_b64`, `194 ds_store_b16`, `288
  ds_load_u16_d16`, and `192 buffer_store_b32`.
- decision:
  reject before timing or catalog transfer. This closes the accumulator parking
  axis for the current HIP C++ source shape: selected-lane parking is
  semantically incomplete, while full-vector half-tile parking is still wrong
  and far beyond the spill budget.

## 2026-06-19 - Q5_K p33 MMQL64 BK2 B-pair+BHALF rejection

- source:
  `sources/llama.cpp` dirty after adding opt-in catalog route
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql64x64_bk2_bpair_bhalf_wg256_f32`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, built through CMake/Ninja target
  `llama-bench test-backend-ops`.
- model/shape:
  Qwen2.5 Coder 7B Q5_K_M focused p33 rows, with p512 and p513 non-steal
  guards.
- route or kernel candidate:
  packed Q8_1/x4 MMQL64 BK2 route that combines the accepted BHALF RHS
  scale-cache payload with the lower-live B-pair issue window.
- baseline command:
  current default focused p33 `test-backend-ops perf` with the accepted
  BHALF+BQUAD route.
- variant command:
  same command with
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL64_BK2_BPAIR_BHALF_PROMPT=1`.
- route trace path:
  `cache/hrxv1/gfx1151/q5-mmql64-bk2-bpair-bhalf-20260619-191522/`.
- profile or timing artifact path:
  `perf-compare-r1.md` and `perf-compare-r2.md` under the artifact directory.
- correctness result:
  p33, p512, and p513 CPU-reference gates passed. p33 selected the new route
  for Qcur, ffn_out, and ffn_gate; Kcur stayed on rows2. p512 stayed on
  MMQL128 and p513 stayed on MMQL128 BQUAD tail routing.
- timing result:
  rejected versus default in both focused p33 rounds: total `3029.411 ->
  3140.802 us` and `3018.821 -> 3100.522 us`. Qcur regressed about `3.2%`
  and ffn_gate regressed `5.5-10.0%`; ffn_out was flat/mixed.
- static evidence:
  wave64, SGPR `61`, VGPR `105`, LDS `9728`, no private segment, no spills,
  `256 v_dot4_i32_iu8`, `48` LDS reads, `30` LDS writes, `39` VMEM loads,
  `16` VMEM stores, and `2` barriers.
- decision:
  reject for promotion and keep opt-in only. Combining the positive half-scale
  payload with lower-live B-pair does not recover the B-pair timing loss; the
  accepted BHALF+BQUAD route remains the Q5 p33 default.

## 2026-06-19 - Vulkan oracle schedule-contract checker extension

- source:
  `sources/llama.cpp` updated `tools/vulkan-oracle/check_isa_contract.py`.
- purpose:
  move the HRX v1 gfx1151 loop further away from aggregate-token or
  headline-opcode decisions by making the pass/fail contract check emitted
  first-window schedule facts from RADV-vs-HIP compare JSON.
- new checks:
  `--match-wmma-score`, `--match-hot-score`, `--rhs-wmma-score-max`, and
  `--rhs-hot-score-max` over `event_summary.wmma_score` and
  `event_summary.hot_op_score`.
- validation artifact:
  `cache/hrxv1/gfx1151/oracle-contract-checker-20260619-continued/`.
- validation result:
  pycompile passed for the oracle scripts. The exact Q8 motif192 K2
  schedule-contract check correctly failed despite matching `32 v_wmma` and
  LDS `22528`: RHS had `pre_wmma_ds_load_b64=64` versus RADV `59`,
  `hot_op_in_window=31` versus RADV `32`, and `16` VGPR spills. An old compare
  artifact without event summaries failed closed. A zero-spill max-contract
  check on `q8-motif192-waitload-static` passed.
- Q8 matrix result:
  `q8-event-contract-matrix.md` in the artifact directory scanned 14 existing
  event-summary Q8 comparisons. No candidate matched the exact event contract.
  The closest zero-spill rows only matched WMMA count, LDS bytes, and no-spill
  metadata; they still missed RADV's pre-WMMA `ds_load_b64=59`,
  `final_pre_wmma_lgkmcnt=51`, `hot_op_in_window=32`, and store-block shape.
- decision:
  use these event-score gates before promoting further HIP C++ candidates.
  Matching the Vulkan schedule now means matching RADV-visible issue-window
  facts, not only math opcode counts, LDS bytes, and no-spill metadata.

## 2026-06-20 - Vulkan oracle store-cluster contract extension

- source:
  `sources/llama.cpp` updated `tools/vulkan-oracle/compare_amdgcn_isa.py`
  and `tools/vulkan-oracle/check_isa_contract.py`.
- purpose:
  make cooperative store/lane-ownership evidence usable without depending on
  basic-block labels. The previous Q8 matrix showed `store_blocks=0` for some
  HIP-side text because HSACO objdump did not expose the same BB labels as RADV
  shader dumps, even though store windows existed.
- new evidence:
  `compare_amdgcn_isa.py` now emits label-independent `store_clusters`,
  `store_cluster_motifs`, and `store_cluster_score`. `check_isa_contract.py`
  accepts `--match-store-score` and `--rhs-store-score-max`.
- validation artifact:
  `cache/hrxv1/gfx1151/oracle-store-cluster-checker-20260620-continued/`.
- validation result:
  regenerated a raw Q8 RADV-vs-HIP ISA comparison for the b64group candidate.
  The store delta remains structural after removing the BB-label bias: RADV has
  `18` store clusters, `324` store ops, `192` buffer stores, and `132` LDS
  stores; the HIP b64group row has `1` cluster, `66` store ops, `0` buffer
  stores, `64` global stores, and `2` LDS stores. The exact store contract
  fails while a loose max contract passes, proving both checker paths work.
- decision:
  keep Q8 direct-WMMA work focused on cooperative store/lane ownership and
  output distribution. The current HIP candidates are not merely missing BB
  labels; they are emitting a materially different store schedule from RADV.

## 2026-06-20 - Store-cluster VMEM normalization

- source:
  `sources/llama.cpp` updated `tools/vulkan-oracle/compare_amdgcn_isa.py`
  and `tools/vulkan-oracle/summarize_isa_compare_matrix.py`.
- purpose:
  separate backend-specific VMEM store opcode spelling from the store topology
  that matters for schedule matching. RADV reports Vulkan buffer writes as
  `buffer_store_*`, while HIP pointer stores often lower as `global_store_*`;
  exact opcode split is useful, but it should not hide the normalized VMEM
  store count.
- new evidence:
  `store_cluster_score` and store motifs now include `vmem_store_ops`.
  `summarize_isa_compare_matrix.py` reports store clusters, normalized VMEM
  stores, and store-side LDS stores for both candidate and RADV rows.
- validation artifact:
  `cache/hrxv1/gfx1151/oracle-store-vmem-normalized-20260620-continued/`.
- validation result:
  regenerated the Q8 b64group raw RADV-vs-HIP comparison and one-row matrix.
  The normalized store contract still fails: RADV has `192` VMEM stores,
  `132` store-side LDS stores, and `18` store clusters; HIP has `64` VMEM
  stores, `2` store-side LDS stores, and `1` store cluster, while both have
  `32 v_wmma`.
- decision:
  use normalized VMEM-store fields for backend-neutral topology checks, and
  exact `buffer_store`/`global_store` fields only when investigating lowering
  choices. The active Q8 gap remains output redistribution/cooperative store
  topology, not only opcode spelling.

## 2026-06-20 - Q8 HSACO family store-contract ranking

- source:
  `sources/llama.cpp` dirty after extending
  `tools/vulkan-oracle/summarize_hsaco_family.py`.
- purpose:
  rank the already-built CMake/Ninja Q8 direct-WMMA HSACO catalog against the
  RADV Q8 b64group oracle contract before adding another HIP C++ route. This
  turns the latest store-cluster and WMMA issue-window metrics into a family
  triage tool.
- new evidence:
  the summarizer now accepts `--reference-compare-json`, imports the RADV-side
  `store_cluster_score` and `wmma_score`, emits per-HSACO store/WMMA deltas,
  and can sort by store or total contract gap.
- validation artifact:
  `cache/hrxv1/gfx1151/q8-hsaco-store-rank-20260619-194009/`.
- validation result:
  scanned `47` current Q8 VK128 padded/w64/b64group HSACOs from
  `build/hrx-v1-catalog-gfx1151`. The closest total contract match is
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_directwait_bufferstore_wg256.hsaco`,
  which reaches `store_gap=19` and `wmma_gap=7` but is already statically
  rejected because it has `VGPR=256`, scratch `68`, and `16` VGPR spills. The
  best no-scratch/no-spill route is
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_ktilefrag_bufferstore_wg256.hsaco`
  with `store_gap=20`, `wmma_gap=72`, `VGPR=212`, and no spills. The common
  no-spill full-pair candidates match RADV's `192` VMEM stores and nearly
  match store-side LDS stores (`130` vs `132`) but collapse RADV's `18` store
  clusters into `1-3` clusters and miss the WMMA issue window.
- decision:
  do not add another B-copy, selected-half, or full-pair route from source
  similarity alone. The next Q8 production candidate should target the missing
  low-level issue-window/store-cluster topology while preserving the no-spill
  surface, or else start from the spilling inline-WMMA K2 directwait route and
  reduce live range/ABI pressure before any focused correctness gate.

## 2026-06-20 - Q8 K2 directwait no-memory-clobber compile rejection

- source:
  `sources/llama.cpp` dirty after adding
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_nomem_motif192_k2_directwait_bufferstore_wg256.hip.cpp`,
  registering it in CMake, and adding catalog source/artifact/family metadata.
- purpose:
  test whether the closest RADV-contract Q8 row spills because the inline
  `v_wmma_f16_16x16x16_f16` wrapper carries an artificial `memory` clobber.
  This preserves the K2 directwait dataflow and changes only the inline-WMMA
  clobber surface.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_nomem_motif192_k2_directwait_bufferstore_wg256.hsaco -j$(nproc)`.
- baseline:
  existing
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_directwait_bufferstore_wg256.hsaco`.
- validation artifact:
  `cache/hrxv1/gfx1151/q8-nomem-asmwmma-k2-directwait-static-20260619-194459/`.
- validation result:
  the new HSACO built through CMake/Ninja and preserved the same broad static
  surface as the baseline: wave64, `SGPR=56`, `VGPR=256`, `LDS=22528`,
  scratch `68`, `16` VGPR spills, `32 v_wmma`, `192` LDS reads, `130` LDS
  writes, `192` VMEM stores, `235` waitcnt-class instructions, and three
  barriers. The oracle score moved only from `wmma_gap=7` to `wmma_gap=6`
  because `wmma_in_window` reached `32`; the pressure cliff did not improve.
- decision:
  reject before route selection, CPU-reference, or timing. The K2 directwait
  spill is not caused by the inline-WMMA `memory` clobber. The next Q8 attempt
  must reduce the live two-K-tile fragment/accumulator surface or use a lower
  level cooperative writeback primitive rather than tuning this clobber axis.

## 2026-06-20 - Q4_K MoE ID production-width route-tile promotion

- source:
  `sources/llama.cpp` changed `ggml/src/ggml-hrx/ggml-hrx.cpp` selector policy
  and route trace text, plus
  `ggml/src/ggml-hrx/catalog/tuning/gfx1151/promotions.json`.
- purpose:
  test the existing wider grouped Q8_1 x4 Q4_K `MUL_MAT_ID` export as a
  route-tile axis against the accepted BN16 route. This is not a new math
  implementation and not a direct Vulkan cooperative-matrix clone; it is a
  bounded packed-path probe for the current Qwen3/Qwen3-Coder MoE gap.
- route or candidate:
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32`.
  Default on gfx1151 for production-width Q4_K `MUL_MAT_ID` rows with
  `n_tokens >= 128`. Rollback:
  `GGML_HRX_DISABLE_Q4_K_ID_Q8_1_X4_MMQ64X64_WIDE_K_PROMPT=1`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-ops -j$(nproc)`.
- static evidence:
  CMake/Ninja-built `mul_mat_id_q4_k_q8_1_x4_mmq.hsaco` reports the wider
  export as wave64, SGPR `87`, VGPR `136`, LDS `3968`, no private segment,
  and no spills. The accepted BN16 export is wave64, SGPR `87`, VGPR `107`,
  LDS `3264`, no spills.
- focused correctness:
  `cache/hrxv1/gfx1151/q4-id-mmq64x64-wide-focused-20260619-195332/`.
  CPU-reference passed for p33, p512, and p513 when forced to the wider route.
- focused timing:
  `cache/hrxv1/gfx1151/q4-id-mmq64x64-wide-perf-20260619-195409/`.
  Forced p33 was rejected: supported total `499.393 -> 529.953 us`
  (`1.061x` slower). p512 improved `4058.687 -> 3261.186 us`
  (`0.804x` time), and p513 improved `4120.652 -> 3439.192 us`
  (`0.835x` time).
- final route guard:
  `cache/hrxv1/gfx1151/q4-id-mmq64x64-wide-default-final-20260619-195832/`.
  Default p512 selects `mmq64x64` with `mmq16=0`; p33 stays on BN16; rollback
  returns p512 to BN16.
- model A/B:
  default artifact:
  `cache/hrxv1/gfx1151/q4-id-mmq64x64-wide-default-model-20260619-195624/`.
  opt-in artifact:
  `cache/hrxv1/gfx1151/q4-id-mmq64x64-wide-optin-model-20260619-195653/`.
  Qwen3 30B Q4_K_XL steady p512 improved `593.0285 -> 652.8210 tok/s` and
  p513 improved `592.1030 -> 633.3515 tok/s`. Qwen3-Coder 30B Q4_K_M steady
  p512 improved `855.7555 -> 903.0075 tok/s` and p513 improved
  `825.4795 -> 847.0980 tok/s`. Route traces confirm Q4 ID dispatches switched
  from BN16 to the wider export while SWIGLU stayed on the existing BN16 route.
- decision:
  promote as a gfx1151 production-width default with rollback. Keep p33 guarded
  to BN16. This is progress toward the MoE parity boulder but not Vulkan
  parity; next MoE work should attack the fused SWIGLU route or a lower-level
  Vulkan-like subgroup ID schedule.
## 2026-06-19 - Q8_0 current contract rank after fa5235f3e basket

- Current basket: `cache/hrxv1/gfx1151/basket-current-fa5235f3e-20260619-200143/`; steady geomean HRX/Vulkan `0.610x`, with Q8_0 p512/p513 at `0.501x`/`0.513x` and zero HRX fallbacks.
- Focused prior: `cache/hrxv1/gfx1151/q8_0-current-focused-hrx-vulkan-compare-ab41b8701-20260619-075951/`; HRX/Vulkan backend-op time ratio `1.512x` at p512 and `1.463x` at p513. The largest absolute miss remains `result_output`.
- New static rank artifact: `cache/hrxv1/gfx1151/q8-current-contract-rank-20260619-201034/`, generated with `summarize_hsaco_family.py --sort contract`. It confirms no current Q8 HSACO satisfies the RADV store/WMMA contract. The closest no-spill row is `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_ktilefrag_bufferstore_wg256.hsaco`, but it still misses store clustering and the WMMA issue window (`total_gap=92`) and was already runtime-rejected.
- Tooling change in `sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py` adds explicit contract failure labels and `--sort contract` so future Q8 work can avoid repeating closed variants.
- Conclusion: do not promote or re-run existing Q8 packed or direct-WMMA variants. Next Q8 experiment must change the lower-level cooperative store/lane ownership contract or introduce a genuinely new packed-Q8_1 dataflow, then pass p33/p512/p513 focused correctness, route, static, and timing gates.

## 2026-06-20 - Q4_K MoE SWIGLU production-width BN32 promotion

- source:
  `sources/llama.cpp` changed `ggml/src/ggml-hrx/ggml-hrx.cpp`,
  `tests/test-backend-hrx.cpp`, and
  `ggml/src/ggml-hrx/catalog/tuning/gfx1151/promotions.json`.
- purpose:
  continue the Qwen MoE parity work after the Q4_K ID route-tile promotion by
  testing the existing wider fused SWIGLU grouped Q8_1 x4 export as a bounded
  production-width candidate. This is a route-width bracket around the packed
  HRX schedule, not a direct Vulkan subgroup/cooperative-matrix clone.
- route or candidate:
  `hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32`.
  Default on gfx1151 only for `n_tokens >= 128`; rollback:
  `GGML_HRX_DISABLE_Q4_K_SWIGLU_Q8_1_X4_MMQ_PROMPT=1`.
- build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target llama-bench test-backend-hrx -j$(nproc)`.
- static evidence:
  `cache/hrxv1/gfx1151/q4-swiglu-current-static-20260619-201635/`.
  The CMake/Ninja-built `mul_mat_id_q4_k_q8_1_x4_mmq.hsaco` is wave64,
  SGPR `87`, VGPR `136`, LDS `3968`, no scratch, and no spills. The artifact
  is a coarse HSACO-level pass because the object contains multiple Q4 MoE
  exports.
- focused correctness and route evidence:
  `cache/hrxv1/gfx1151/q4-swiglu-bn32-default-focused-20260619-202435/`.
  p33 defaults to BN16 with `wg_count=[1,3,4]`; p512 and p513 default to BN32
  with `wg_count=[1,16,4]` and `[1,17,4]`. All three focused
  CPU-reference cases passed.
- model A/B:
  `cache/hrxv1/gfx1151/q4-swiglu-bn32-default-rollback-model-ab-20260619-202537/`
  and the noisy Coder p512 rerun
  `cache/hrxv1/gfx1151/q4-swiglu-bn32-coder-p512-repeat-20260619-202658/`.
  Qwen3 30B Q4_K_XL p512 improved `649.672260 -> 675.584051 tok/s`, p513
  improved `638.696683 -> 658.203124 tok/s`. Qwen3-Coder 30B Q4_K_M p513
  improved `839.875777 -> 877.717409 tok/s`; the repeated p512 r7 check
  improved `897.755701 -> 910.888614 tok/s`.
- rejection:
  `cache/hrxv1/gfx1151/q4xl-swiglu-bn32-model-ab-20260619-201944/`.
  Forcing BN32 on Qwen3 Q4_K_XL p33 selected the expected provider for all
  SWIGLU prompt rows, then `llama-bench` failed prompt decode with `res=-3`.
  Keep p33/narrow prompts guarded to BN16.
- decision:
  promote BN32 as a gfx1151 production-width SWIGLU default with rollback and
  keep p33 on BN16. This is incremental MoE progress, not a Vulkan parity
  claim; the next MoE step still needs a lower-level Vulkan-like subgroup ID
  schedule or a genuinely new packed SWIGLU dataflow.

## 2026-06-20 - Post-SWIGLU promotion basket checkpoint

- source:
  `sources/llama.cpp` commit `2ef0583fb`
  (`hrx: widen q4 swiglu route on gfx1151`).
- artifact:
  `cache/hrxv1/gfx1151/basket-current-2ef0583fb-postbuild-20260619-203256/basket-head-20260619-203256/`.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --models all --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --flash-attn 1 --timeout 240`.
- validation:
  `48` rows, zero failures. HRX rows report `backends=HRX`, `devices=HRX0`;
  Vulkan rows report `backends=Vulkan`, `devices=Vulkan0`; HRX build commit
  is `2ef0583fb`.
- result:
  avg geomean HRX/Vulkan `0.617`; steady geomean `0.612`. Zero HRX fallback
  lines.
- worst steady rows:
  Llama 3.1 8B Q8_0 p512 `0.494x`; Qwen2.5 Coder 7B Q5_K_M p512 `0.522x`;
  Llama 3.1 8B Q8_0 p513 `0.526x`; Llama 3.1 8B Q4_K_M p33 `0.527x`;
  Qwen3 30B Q6_K p33 `0.529x`.
- MoE status:
  Qwen3 Q4_K_XL p33 is `0.898x` steady, but production-width p512/p513 remain
  `0.553x`/`0.565x`; Qwen3-Coder Q4_K_M is stronger at p33/p512/p513
  `0.990x`/`0.781x`/`0.791x`. The SWIGLU promotion moved the MoE rows, but the
  remaining parity gap is still dense prompt-matmul schedule quality.
- decision:
  keep the goal active. The next boulders are still Q8_0 p512/p513 store/lane
  ownership, Q5_K production-width p512, Q4_K narrow p33, and Q6_K p33. Do not
  revisit exhausted Q8 packed/direct-WMMA variants unless the experiment changes
  the lower-level cooperative store/lane-ownership contract or introduces a new
  dataflow.

## 2026-06-20 - Qwen2.5 Q5 row triage exposes Q6 large-row miss

- source:
  `sources/llama.cpp` commit `2ef0583fb`
  (`hrx: widen q4 swiglu route on gfx1151`).
- trigger:
  the post-SWIGLU basket ranked Qwen2.5 Coder 7B Q5_K_M p512 as the second
  worst steady aggregate row at `0.522x` Vulkan. A focused backend-op export
  was run to determine whether the aggregate miss was actually the Q5_K route
  or another quant family inside the same model graph.
- artifact:
  `cache/hrxv1/gfx1151/q5-current-focused-2ef0583fb-20260619-203813/`.
  The p512 and p513 op exports were produced from the Qwen2.5 Coder 7B Q5_K_M
  model and focused to dense quantized prompt rows.
- measurement hygiene:
  the first Vulkan `test-backend-ops` attempt was contaminated by an HRX
  `LD_LIBRARY_PATH`. The accepted comparison reran Vulkan with an isolated
  Vulkan build/runtime path and verified the backend side by command context.
  Do not use the contaminated intermediate rows.
- focused timing:
  p512 total backend-op time is HRX `126710.530 us` versus Vulkan
  `54520.525 us` (`2.324x`). The Q5_K MMQL128 rows are only moderately behind:
  `Qcur-0` is `1.530x`, `ffn_gate-0` is `1.578x`, and `ffn_out-3` is
  `1.558x`. The largest absolute misses are Q6_K direct-WMMA rows:
  `ffn_out-0` is `2.856x` and `result_output` is `2.420x`.
- odd/tail timing:
  p513 repeats the same pattern with total HRX/Vulkan `2.368x`.
  Q5_K rows are `1.621x` to `1.942x`, while Q6_K `ffn_out-0` and
  `result_output` are `2.473x` and `2.474x`.
- current static evidence:
  `cache/hrxv1/gfx1151/q6-current-vk128-contract-matrix-2ef0583fb-20260619-204917/`.
  This refresh compares the RADV Q6 large oracle
  `matmul_q6_k_f32_f16acc_aligned_l` against every current CMake/Ninja-built
  Q6 VK128 HIP HSACO, including the latest B64GROUP+bufferstore route. No
  current variant satisfies the RADV static contract. The closest static row is
  `mul_mat_vec_q6_k_wmma16_vk128_padded_w64_b64group_bufferstore_wg256.hsaco`,
  but it still has VGPR `196` versus RADV `192`, LDS `20480` versus `22528`,
  no `ds_load_u16_d16`, only two LDS stores, two store clusters, and a shorter
  WMMA issue window. That same route was already rejected by focused timing in
  `cache/hrxv1/gfx1151/q6-wmma-vk128-b64group-bufferstore-focused-20260619-074930/`.
- decision:
  do not treat the Qwen2.5 Q5 aggregate row as a pure Q5_K tuning target.
  Q5_K production-width MMQL remains a real `~1.5x-1.9x` gap, but the largest
  absolute miss in this row is Q6_K large prompt direct-WMMA. The visible Q6
  HIP C++ axes already tested are closed: B64GROUP, FULLSTORE, STORE_STAGE,
  B64GROUP+FULLSTORE, B64GROUP+STORE_STAGE, and B64GROUP+BUFFERSTORE all fail
  either timing or model A/B despite focused correctness. The next Q6 parity
  step must change the implementation surface: a lower-level cooperative
  store/lane-ownership primitive, a different source form that preserves
  RADV-like halfword LDS load/store topology without 34 barriers, or a genuinely
  different packed-Q8_1 dataflow. Another source-visible permutation of the
  current VK128 direct-WMMA family is not promotion-aligned.

## 2026-06-20 - Llama 3.1 Q4_K_M p33 focused triage points at Q6 medium rows

- source:
  `sources/llama.cpp` commit `2ef0583fb`
  (`hrx: widen q4 swiglu route on gfx1151`).
- trigger:
  the post-SWIGLU basket ranked Llama 3.1 8B Q4_K_M p33 as a current worst
  row at HRX/Vulkan steady `0.527x`. The accepted HRX route for this row is
  already `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32`, so a
  fresh focused backend-op comparison was needed before assuming this was still
  a Q4 packed-route problem.
- artifact:
  `cache/hrxv1/gfx1151/q4-p33-current-focused-2ef0583fb-20260619-205126/`.
  Exported from
  `shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
  with `p33/n0/b33/ub33/fa1`, then focused to `qk_prompt`.
- correctness:
  HRX and Vulkan `test-backend-ops test -o MUL_MAT` both passed all seven
  focused rows. HRX route traces show Q4 rows select the intended BK2/BQUAD
  packed providers, and Q6 rows select
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32`.
- focused timing:
  total HRX/Vulkan backend-op time is `16564.207 / 10778.828 us = 1.537x`.
  The Q4_K rows are no longer the dominant miss: `Kcur-0` is `1.210x`,
  `Qcur-0` is `1.164x`, `ffn_gate-0` is `1.189x`, and `ffn_out-4` is
  `1.246x`. The Q6_K p33 rows are the larger misses: `Vcur-0` is `2.946x`,
  `ffn_out-0` is `2.964x`, and `result_output` is `1.446x`.
- static evidence:
  `cache/hrxv1/gfx1151/q6-current-vk64-contract-matrix-2ef0583fb-20260619-205308/`.
  This compares the RADV Q6 p33 medium oracle
  `matmul_q6_k_f32_f16acc_aligned_m` against current CMake/Ninja-built Q6
  VK64 HIP HSACOs. No current VK64 variant satisfies the RADV static contract.
  The accepted default and H4LOAD compile to the same visible surface:
  wave64, VGPR `59`, LDS `11264`, no spills, `8` WMMA sites, `20`
  `ds_load_2addr_b64`, `2` LDS stores, and `16` global stores. RADV medium has
  VGPR `144`, LDS `11264`, no spills, `96` buffer stores, `70` LDS stores,
  `48` pre-WMMA `ds_load_b64`, and a much deeper pre-WMMA wait window. The
  GROUPK2 variants move to `40` `ds_load_b64` and VGPR `101`, but still have
  only `16` global stores, `2` LDS stores, and were already rejected on
  correctness/timing in earlier artifacts.
- decision:
  do not chase the Llama 3.1 Q4_K_M p33 aggregate row by changing Q4 BK2/BQUAD
  policy first. The focused row says the current p33 boulder is Q6 medium
  direct-WMMA. The next Q6 p33 candidate must target RADV's medium
  cooperative-store/writeback contract or a new packed-dataflow alternative;
  another local Q4 narrow packed route is not the highest-leverage move for
  this row.

## 2026-06-20 - Q6 medium cooperative-store diagnostic reaches core RADV counts

- source:
  `sources/llama.cpp` commit `2ef0583fb` plus local diagnostic source change
  in `ggml/src/ggml-hrx/tools/hip-bench/coopmat_store_contract_bench.hip.cpp`.
- trigger:
  the Q6 VK64 static matrix showed the accepted p33 route has the right LDS
  footprint but the wrong schedule surface: only `8` WMMA sites, `20`
  `ds_load_2addr_b64`, `16` output stores, and `2` LDS stores versus RADV
  medium's `16` WMMA, `48` pre-WMMA `ds_load_b64`, `96` output stores,
  `64` `ds_load_u16_d16`, and `70` LDS stores. Existing CMake-built
  cooperative-store probes were close but either failed the fixture or missed
  key counts.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-vk64-radv96-2ef0583fb-20260619-210448/`.
- mode:
  added CMake/Ninja-built bench mode `wmma-lds-vk64-radv96`, kernel symbol
  `coopstore_probe_wmma_lds_vk64_radv96`.
- correctness:
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-vk64-radv96` passes
  with `bad=0 max_abs=0`.
- static comparison:
  the new diagnostic matches the core RADV Q6 medium counts that current
  production Q6 VK64 misses: wave64, LDS `11264`, scratch `0`, `16` WMMA,
  `48` pre-WMMA `ds_load_b64`, `64` `ds_load_u16_d16`, and `96`
  `buffer_store_b32`.
- remaining delta:
  HIP diagnostic resources are SGPR `14`, VGPR `104` versus RADV SGPR `108`,
  VGPR `144`. Store organization is still wrong: the summary reports only `2`
  store clusters and `88` store-side LDS stores versus RADV's `11` clusters
  and `70` store-side LDS stores. Barriers are also higher (`8` versus `2`).
- decision:
  this is an accepted diagnostic prior, not a production route. It proves HIP
  C++ can be spelled to hit the main Q6 medium RADV math/load/output-store
  counts under the proper LDS footprint. The next Q6 p33 production candidate
  should port this compact-fragment/ring-load source shape into the VK64 Q6
  kernel, then fix writeback clustering and barrier placement before any
  route promotion.

## 2026-06-20 - Q6 VK64 RING96 catalog transfer fails focused correctness

- source:
  `sources/llama.cpp` local work on top of commit `9bee589d7`
  (`hrx: add q6 vk64 radv store diagnostic`).
- trigger:
  the `wmma-lds-vk64-radv96` coopstore diagnostic matched the Q6 p33 RADV
  medium route's core 16-WMMA and 48-LDS-b64 surface. The next question was
  whether the same compact ring fragment-load spelling could transfer into
  the real Q6 VK64 catalog ABI.
- route:
  opt-in provider
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_f16acc_wg256_f32`,
  selector gate
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_RING96_F16ACC_WG256_PROMPT=1`.
- build:
  built through normal CMake/Ninja in
  `build/hrx-v1-catalog-gfx1151`, not through an assembler helper.
- static artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-static-9bee589d7-20260619-211725/`.
- focused artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-focused-9bee589d7-20260619-211810/`.
- static comparison:
  the route reached wave64, LDS `11264`, scratch `0`, `16` WMMA, and `48`
  `ds_load_b64`, matching the main intended source-shape pivot. It still
  misses the RADV medium contract: VGPR `163` versus `144`, `0`
  `ds_load_u16_d16` versus `64`, `2` LDS stores versus `64`, `64`
  buffer stores versus `96`, only `3` store clusters versus `11`, and final
  pre-WMMA `lgkmcnt=0` versus RADV `40`.
- correctness:
  focused p33 CPU-reference selected the new provider for `Vcur-0-p33`,
  `ffn_out-0-p33`, and `result_output-p33` and failed all three with finite
  ERR around `33-34`.
- decision:
  reject before timing or model-level tests. The useful evidence is narrow but
  real: the catalog source can be pushed to the 16-WMMA/48-b64 surface, but
  that alone is not the Vulkan schedule. The next Q6 medium candidate should
  preserve that surface while adding RADV's halfword LDS staging and writeback
  lane-ownership contract; another WMMA issue-window-only probe is not enough.

## 2026-06-20 - Q6 VK64 RING96 fast-half buffer-store rejection

- source:
  `sources/llama.cpp` commit `e3f638511`
  (`hrx: reject q6 ring96 fast-half probe`).
- trigger:
  the prior RING96 catalog transfer reached the useful Q6 p33 RADV medium
  `16` WMMA and `48` `ds_load_b64` surface but failed focused correctness.
  The WN32 fast-half buffer-store probe proved the selected/dummy halfword LDS
  writeback stage can be made semantically safe. This probe combined those two
  axes to test whether missing halfword writeback was the reason RING96 failed.
- route:
  opt-in provider
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_fast_half_bufferstore_f16acc_wg256_f32`,
  selector gate
  `GGML_HRX_ENABLE_Q6_K_WMMA16_VK64_PADDED44_W64_RING96_FAST_HALF_BUFFERSTORE_F16ACC_WG256_PROMPT=1`.
- build:
  built through normal CMake/Ninja in
  `build/hrx-v1-catalog-gfx1151`; no assembler helper was used.
- static artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-fast-half-bufferstore-static-6e893b66e-20260619-222137/`.
- focused artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-fast-half-bufferstore-focused-6e893b66e-20260619-222222/`.
- static comparison:
  the route preserved wave64, `16` WMMA, and `48` total `ds_load_b64`, and
  replaced global stores with halfword LDS writeback plus raw buffer stores:
  `128` `ds_load_u16_d16`, `130` `ds_store_b16`, `64` `buffer_store_b32`, and
  no global stores. It still diverges from the Qwen3 30B Q6_K p33 RADV medium
  oracle: LDS `15360` versus `11264`, VGPR `163` versus `144`, buffer stores
  `64` versus `96`, halfword LDS load/store counts overshoot RADV's `64/64`,
  pre-WMMA `ds_load_b64` is `28` versus `48`, and final pre-WMMA `lgkmcnt`
  remains `0` versus RADV `40`.
- correctness:
  focused p33 and generated p64 CPU-reference rows selected this provider for
  all Q6 rows. p33 failed with finite ERR around `33.1-33.9`; p64 failed with
  finite ERR around `2.99-3.00`.
- decision:
  reject before timing or model tests. Collision-safe selected-half writeback
  does not repair RING96's correctness failure. The remaining fault is in the
  ring fragment/lane ownership or outstanding-load schedule itself, so the
  next Q6 medium probe should not layer more post-accumulator staging on the
  same RING96 source.

## 2026-06-20 - Q6 VK64 RADV96 accumulator-direct fixture rejection

- source:
  `sources/llama.cpp` dirty after adding
  `hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-vk64-radv96-accdirect`.
- trigger:
  the earlier `wmma-lds-vk64-radv96` diagnostic was accepted because it hit
  the Q6 p33 RADV medium headline surface, but it wrote synthetic direct-output
  values for groups `0..7` while only keeping the computed ring WMMA
  accumulators live through a sink. That left the RING96 accumulator/lane
  contract unproven.
- build:
  rebuilt the bench through normal CMake/Ninja in
  `build/hrx-v1-catalog-gfx1151`; no assembler helper was used.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-vk64-radv96-accdirect-20260619-223343/`.
- control:
  `wmma-lds-vk64-radv96` still passes with `bad=0 max_abs=0`.
- variant:
  `wmma-lds-vk64-radv96-accdirect` keeps the same fixture shape but writes
  computed ring WMMA accumulator groups `0..7` through the direct raw-store
  path, with staged synthetic groups `8..23`.
- correctness:
  fails immediately with `bad=256`; first failure is
  `group=5 slot=0 lane=0`, with actual values such as `-28848` where the
  expected ring WMMA value is `128`.
- static note:
  the artifact contains the extracted executable `.hip_fatbin` object and raw
  objdump. The generic HSACO family summarizer was not used for the conclusion
  because symbol extraction fell back to whole-object disassembly on this
  embedded bench object.
- decision:
  reject RING96 as a correctness prior, not just as a production route. This
  explains why both Q6 RING96 catalog transfers failed with finite ERR even
  after fixing selected-half writeback. The next Q6 p33 probe must first repair
  or replace the ring WMMA lane ownership in a fixture before another real Q6
  catalog route is justified.

## 2026-06-20 - Q6 VK64 RADV96 fragment-copy fixture repair

- source:
  `sources/llama.cpp` dirty after extending
  `hrx-hip-bench-coopmat-store-contract` with failure group reporting and
  `wmma-lds-vk64-radv96-accdirect-copy{a,b,ab}` modes.
- trigger:
  `wmma-lds-vk64-radv96-accdirect` proved the prior RING96 fixture only failed
  when writing real ring WMMA accumulators, and the enhanced comparator showed
  the failure was isolated to output group `5`.
- build:
  rebuilt the bench through normal CMake/Ninja in
  `build/hrx-v1-catalog-gfx1151`; no assembler helper was used.
- artifact:
  `cache/hrxv1/gfx1151/coopstore-vk64-radv96-accdirect-copy-20260619-224152/`.
- control:
  `wmma-lds-vk64-radv96` passes with `bad=0 max_abs=0`.
- failing baseline:
  `wmma-lds-vk64-radv96-accdirect` fails with
  `bad=256 max_abs=inf first_bad=1280 group=5 slot=0 lane=0 actual=-11128 expected=128 nan=80 inf=48 bad_groups=5:256`.
- copy variants:
  `wmma-lds-vk64-radv96-accdirect-copya`,
  `wmma-lds-vk64-radv96-accdirect-copyb`, and
  `wmma-lds-vk64-radv96-accdirect-copyab` all pass with `bad=0 max_abs=0`.
- decision:
  accept as standalone diagnostic evidence. Explicit `v_mov_b32` fragment
  materialization repairs the isolated group-5 RING96 accumulator failure in
  the fixture, so the next production-directed Q6 probe can test opt-in
  fragment materialization in the real VK64 RING96 catalog route. That probe
  still needs focused p33 and p64 correctness before timing, plus static
  pressure review because Q8 copy repairs have often increased VGPR cost.

## 2026-06-20 - Q6 VK64 RING96 copy-A catalog rejection

- source:
  `sources/llama.cpp` dirty after adding opt-in route
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_copya_f16acc_wg256_f32`.
- trigger:
  the standalone `coopstore` fixture showed explicit `v_mov_b32`
  materialization of A, B, or A+B fragments repaired the isolated RING96
  group-5 accumulator failure. This probe transferred the smallest repair,
  A-fragment materialization, into the real Q6 catalog route.
- build:
  built through normal CMake/Ninja in `build/hrx-v1-catalog-gfx1151`; no
  assembler helper was used.
- static artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-copya-static-824c8bcc5-20260619-225048/`.
- focused artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-copya-focused-824c8bcc5-20260619-225129/`.
- static comparison:
  the route preserved wave64, LDS `11264`, no spills, `16` WMMA, and `48`
  total `ds_load_b64`, with all `48` visible before WMMA. It raised VGPR from
  base RING96 `163` to `191`, versus RADV medium `144`, and still missed the
  RADV halfword/store contract: only `64` buffer stores, `2` `ds_store_b16`,
  no `ds_load_u16_d16`, and final pre-WMMA `lgkmcnt=0`.
- correctness:
  focused p33 route traces selected the copy-A provider for `Vcur`, `ffn_out`,
  and `result_output`. All three failed strict CPU-reference with finite ERR
  around `33.1-34.3`.
- decision:
  reject before timing or model tests. A-fragment materialization alone does
  not transfer the standalone fixture repair into the production Q6 route. The
  next route-facing probe should test B or A+B materialization, or change the
  lane/writeback contract instead of timing this route.

## 2026-06-20 - Q6 VK64 RING96 copy-B/copy-A+B catalog rejection

- source:
  `sources/llama.cpp` dirty after adding opt-in routes
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_copyb_f16acc_wg256_f32`
  and
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_copyab_f16acc_wg256_f32`.
- trigger:
  copy-A failed focused p33 CPU-reference, while the standalone
  `coopstore` fixture had also shown B-fragment and A+B fragment
  materialization could repair the isolated RING96 group-5 accumulator
  failure. This completed the route-facing whole-fragment materialization
  axis.
- build:
  built both HSACOs through normal CMake/Ninja in
  `build/hrx-v1-catalog-gfx1151`; no assembler helper was used.
- static artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-copyb-copyab-static-05ac2f85a-20260619-225807/`.
- focused artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-copyb-copyab-focused-05ac2f85a-20260619-225847/`.
- static comparison:
  copy-B compiled wave64 with VGPR `175`, LDS `11264`, no spills, `16`
  WMMA, `48` total `ds_load_b64`, and `64` buffer stores. Copy-A+B compiled
  wave64 with VGPR `183` and the same LDS, spill, WMMA, LDS-read, and
  buffer-store counts. Both are lower-pressure than copy-A VGPR `191`, but
  both remain above base RING96 VGPR `163` and the Qwen3 30B Q6_K p33 RADV
  medium oracle VGPR `144`. Both still miss RADV's halfword/store contract:
  only `2` `ds_store_b16`, no `ds_load_u16_d16`, `64` not `96` buffer
  stores, and final pre-WMMA `lgkmcnt=0`.
- correctness:
  focused p33 route traces selected the intended copy-B provider for `Vcur`,
  `ffn_out`, and `result_output`; strict CPU-reference failed with finite ERR
  `33.059`, `34.182`, and `33.989`. Copy-A+B selected for the same three
  rows and failed with finite ERR `32.207`, `34.983`, and `33.559`.
- decision:
  reject both before timing or model tests. The standalone whole-fragment
  materialization repair does not transfer through the real Q6 RING96 catalog
  ABI. Future Q6 medium work should change the lane/writeback contract or
  move to a different schedule family instead of adding more whole-fragment
  copies to this route.

## 2026-06-20 - Current-head basket after Q6 RING96 rejection

- source:
  `sources/llama.cpp` clean at `ab18e9465`
  (`hrx: reject q6 ring96 copyb probes`).
- build:
  rebuilt both `build/hrx-v1-catalog-gfx1151/bin/llama-bench` and
  `build/vulkan-gfx1151/bin/llama-bench` through CMake/Ninja. Both reported
  build commit `ab18e9465` in the benchmark JSON.
- artifact:
  `cache/hrxv1/gfx1151/basket-current-ab18e9465-postbuild-20260619-230808/`.
- analyzer:
  `cache/hrxv1/gfx1151/basket-current-ab18e9465-postbuild-20260619-230808/perf-rank.md`
  and
  `cache/hrxv1/gfx1151/basket-current-ab18e9465-postbuild-20260619-230808/perf-rank.json`,
  generated with `sources/llama.cpp/tools/vulkan-oracle/analyze_basket_perf.py`.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --tag basket-current-ab18e9465-postbuild-20260619-230808 --models all --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --flash-attn 1 --timeout 1200`.
- identity:
  all `24` HRX rows exited `0` with `backends=HRX`; all `24` Vulkan rows
  exited `0` with `backends=Vulkan`; total HRX fallback lines were `0`.
- result:
  average geomean HRX/Vulkan `0.606x`; steady-state geomean HRX/Vulkan
  `0.595x`. This is not parity.
- worst steady rows:
  | Row | HRX tok/s | Vulkan tok/s | Ratio | HRX top route |
  | --- | ---: | ---: | ---: | --- |
  | Llama 3.2 3B Q4_K_M p512 | `1040.788` | `2534.535` | `0.411x` | `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` |
  | Llama 3.1 8B Q4_K_M p33 | `171.762` | `403.164` | `0.426x` | `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32` |
  | Llama 3.1 8B Q8_0 p512 | `450.500` | `887.284` | `0.508x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` |
  | Qwen3 30B Q6_K p33 | `92.001` | `175.637` | `0.524x` | `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32` |
  | Llama 3.2 3B Q4_K_M p513 | `1134.200` | `2157.255` | `0.526x` | `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` |
- decision:
  after closing the Q6 RING96 whole-fragment copy axis, the next schedule
  target should pivot back to Q4_K packed prompt matmul, not another Q6
  RING96 route. Q4_K owns the two worst rows and uses the current B-quad
  packed route family, so the next work should run focused backend-op
  Q4_K p33/p512/p513 rows with route traces and compare the current HSACO
  against the Vulkan oracle before adding another candidate. Q8_0 p512 remains
  the next non-Q4 boulder.
- perf-rank note:
  the worst row, Llama 3.2 3B Q4_K_M p512, spends steady Vulkan time mostly in
  dense Q4_K (`292.62 ms`, `332` dispatches). The largest labels are
  `MUL_MAT q4_K m=8192 n=512 k=3072` (`148.57 ms`),
  `MUL_MAT q4_K m=3072 n=512 k=3072` (`83.97 ms`), and
  `MUL_MAT q4_K m=3072 n=512 k=8192` (`59.63 ms`), while HRX selects
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` `498` times.
  The second-worst row, Llama 3.1 8B Q4_K_M p33, similarly spends steady
  Vulkan time mostly in dense Q4_K (`122.73 ms`, `380` dispatches), while HRX
  selects `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32` `426`
  times and the non-B-quad BK2 route `144` times. This supports a Q4 packed
  route A/B before returning to Q6 RING96.

## 2026-06-20 - Current-head Q4 focused gate and oracle comparison

- source:
  `sources/llama.cpp` clean at `ab18e9465`
  (`hrx: reject q6 ring96 copyb probes`).
- focused artifact:
  `cache/hrxv1/gfx1151/q4-current-focused-ab18e9465-20260619-231517/`.
- static/oracle artifact:
  `cache/hrxv1/gfx1151/q4-current-hsaco-radv-compare-ab18e9465-20260619-232251/`.
- purpose:
  follow the basket result by checking whether current Q4_K prompt rows are
  still a focused kernel problem and whether the selected HRX routes resemble
  the RADV winning schedules closely enough to justify another direct
  schedule-transfer attempt.
- correctness:
  focused CPU-reference gates passed for all tested HRX and Vulkan rows:
  Llama 3.2 3B Q4_K_M `p512`, Llama 3.2 3B Q4_K_M odd `p513`, and
  Llama 3.1 8B Q4_K_M narrow `p33`.
- focused timing:
  | Shape | HRX total us | Vulkan total us | HRX/Vulkan time |
  | --- | ---: | ---: | ---: |
  | Llama 3.2 Q4_K_M `p512` | `5456.868` | `4257.676` | `1.282x` |
  | Llama 3.2 Q4_K_M `p513` | `10071.576` | `7919.635` | `1.272x` |
  | Llama 3.1 Q4_K_M `p33` | `1942.661` | `1461.704` | `1.329x` |
- selected HRX routes:
  `p512` and `p513` selected
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32`;
  `p33` selected `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_wg256_f32`
  for Kcur rows and
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32` for
  Q/FFN rows.
- static comparison:
  the selected HRX large route remains a packed Q8_1 integer-dot route:
  wave64, SGPR `53`, VGPR `169`, LDS `8192`, no spills, `512`
  `v_dot4_i32_iu8`, no WMMA, `46` LDS reads, `12` LDS writes, and
  `64` global stores. The RADV large Q4 route is a different schedule family:
  SGPR `108`, VGPR `192`, LDS `22528`, no spills, `32`
  `v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `192 buffer_store_b32`, and two barriers.
  Medium `p33` shows the same structural split: RADV uses a WMMA/f16acc
  schedule while selected HRX uses the BK2/B-quad packed integer-dot family.
- decision:
  keep Q4_K as a real boulder, but do not start another broad direct-WMMA Q4
  clone from scratch. The Q4 ledger already contains many rejected direct-WMMA
  attempts, including variants that matched RADV-like WMMA/LDS counts but
  failed to approach timing. The current evidence supports either a named
  packed-route schedule axis, or a fresh lower-level cooperative-store/lane-map
  primitive borrowed from the Q8 oracle work, not blind C++ route additions.

## 2026-06-20 - Q8 ledger pass after current Q4 checkpoint

- source:
  `sources/llama.cpp` remains clean at `ab18e9465`.
- checked ledger:
  `docs/hrxv1/q8_0-prompt-schedule-ledger.md`.
- status:
  the Q8 BK2 packed-path candidate is already rejected in the gfx1151 catalog:
  `q8_0_dense_prompt_mmq64x128_splitqsum_bk2_model_ab`, with focused p512
  timing up about `2%` but same-binary Llama 3.1 8B Q8_0 p512 model A/B flat
  to regressive (`458.964864 -> 457.465526 tok/s`, ratio `0.9967`).
- ledger conclusion:
  Q8_0 has been narrowed further than a simple tile-size problem. The current
  default packed-Q8_1 BN128/BN112 split-qsum routes are clean and remain the
  best production routes. Direct-WMMA work has proven that HIP C++ can express
  individual RADV-like pieces through CMake/Ninja-built sources, including
  high-outstanding LDS/WMMA issue windows, raw `192 buffer_store_b32`
  surfaces, and mixed direct/staged store fixtures. The failure is combining
  those pieces in the real production ABI without either correctness failures
  or hard VGPR/spill cliffs.
- important compiler lesson:
  RADV's winning Q8 schedule is not captured by aggregate opcode counts alone.
  The event-window facts matter: RADV carries a deep pre-WMMA LDS load window
  with high `lgkmcnt`, while normal HIP C++ source tends to drain or interleave
  the window unless dependency-pinned; when dependency-pinned in real routes,
  pressure rises sharply. This is the concrete reason the Vulkan oracle is
  useful: it gives schedule facts that should be mechanically copied and
  rejected one axis at a time, rather than inferred from top-level throughput.
- decision:
  no source edit from this checkpoint. The next code move should target a named
  schedule primitive rather than another aggregate benchmark probe. Best
  candidates are a compact lower-level cooperative-store/lane-map primitive
  for Q8/Q4 direct-WMMA transfer, or a documented packed-route dataflow axis
  that changes the actual first-dot issue window. Aggregate basket runs remain
  useful only as boulder selection and final regression/promotion guardrails.

## 2026-06-20 - Q4 MMQL128 B-half compile-gate rejection

- source:
  `sources/llama.cpp` dirty after adding the opt-in route
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bhalf_wg256_f32`.
- trigger:
  the current Q4 packed-family summary showed accepted B-quad is the best
  no-spill wide route, while B-oct moves the pre-dot LDS load window further
  but spills badly. B-half tests the midpoint: preload four WNITER B-cache
  positions before issuing dots, preserving MMQL128/B-quad dataflow,
  `BM=128`, `BN=128`, wave64, `BK_STEP=1`, and output ownership.
- build:
  built through normal CMake/Ninja in `build/hrx-v1-catalog-gfx1151`; no
  assembler helper was used. The build generated
  `mul_mat_vec_q4_k_q8_1_x4_mmql128_bhalf.hsaco` and refreshed the embedded
  catalog.
- static artifact:
  `cache/hrxv1/gfx1151/q4-mmql128-bhalf-static-ab18e9465-20260619-233321/`.
- static comparison:
  B-half moved in the intended schedule direction but crossed the compile
  resource gate. Compared with accepted B-quad's no-spill VGPR `169`, B-half
  compiled as wave64, SGPR `53`, VGPR `192`, LDS `8192`,
  `private_segment_fixed_size=76`, and `vgpr_spill_count=18`. The parsed
  first hot-op score moved from B-quad pre-hot loads `16`/final wait `9` to
  B-half pre-hot loads `26`/final wait `19`, still short of B-oct pre-hot
  loads `46` but already spilling.
- decision:
  reject before focused runtime. This brackets the current packed B-cache live
  window axis: B-quad is the largest no-spill accepted point, B-half spills,
  and B-oct spills harder. The next Q4 packed-path attempt should not preload
  more B-cache state in the same source shape; it needs either a lower-pressure
  issue-order primitive or a different dataflow/lane ownership idea.

## 2026-06-20 - Q4 MMQL128 B-half-CR compile-gate rejection

- source:
  `sources/llama.cpp` dirty after adding the opt-in route
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bhalf_cr_wg256_f32`.
- trigger:
  B-half crossed the gfx1151 spill cliff while moving the packed B-cache
  pre-dot load window in the intended direction. B-half-CR keeps the same
  four-WNITER B-cache preload window but drains each loaded B cluster by
  A micro-row first, testing whether CR-major consumption can shorten live
  ranges enough to recover a no-spill schedule.
- build:
  built through normal CMake/Ninja in `build/hrx-v1-catalog-gfx1151`; no
  assembler helper was used. The build generated
  `mul_mat_vec_q4_k_q8_1_x4_mmql128_bhalf_cr.hsaco` and refreshed the embedded
  catalog.
- static artifact:
  `cache/hrxv1/gfx1151/q4-mmql128-bhalf-cr-static-e7e35977e-20260619-234116/`.
- static comparison:
  B-half-CR slightly reduced the spill footprint versus B-half, but not enough
  to pass the compile/resource gate. It compiled as wave64, SGPR `53`, VGPR
  `192`, LDS `8192`, `scratch_bytes=68`, and `vgpr_spills=16`. Its parsed
  first hot-op score stayed in the same broad class as B-half: pre-hot loads
  `26`, final pre-hot `lgkmcnt=19`.
- decision:
  reject before focused runtime. CR-major ordering is useful at B-quad live
  state because it compiles no-spill, but it does not rescue B-half's four
  WNITER live B-cache window. The current packed B-cache preload axis remains
  bracketed: B-quad/B-quad-CR are no-spill, B-half-CR and B-half spill, and
  B-oct spills harder.

## 2026-06-20 - Q4_K Llama 3.2 rows=3072 MMQL128 policy promotion

- source:
  `sources/llama.cpp` dirty after adding a selector-only policy for
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_wg256_f32`.
- trigger:
  the current basket's worst row is Llama 3.2 3B Q4_K_M p512, which uses
  rows `3072` heavily. The accepted B-quad route improved earlier Q4 rows, but
  the focused p512 sweep showed base MMQL128 is better on this row family.
- candidate:
  default base MMQL128 only on gfx1151 Q4_K packed-Q8_1 prompt rows with
  `rows == 3072`, `cols == 512`, `k % 256 == 0`, and
  contiguous tensors by default. The force env can still be used for broader
  full-column experiments. Rollback:
  `GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_ROWS3072_PROMPT=1`.
- evidence:
  existing-variant sweep:
  `cache/hrxv1/gfx1151/q4-llama32-p512-existing-variant-sweep-2d7392555-20260619-234839/`;
  routecheck:
  `cache/hrxv1/gfx1151/q4-llama32-base-mmql128-routecheck-2d7392555-20260619-235056/`;
  post-edit focused regate:
  `cache/hrxv1/gfx1151/q4-llama32-rows3072-cols512-default-regate-2d7392555-20260620-000208/`;
  post-edit model A/B:
  `cache/hrxv1/gfx1151/q4-llama32-rows3072-cols512-default-model-ab-2d7392555-20260620-000317/`;
  p33 non-steal:
  `cache/hrxv1/gfx1151/q4-llama32-rows3072-p33-nonsteal-2d7392555-20260619-235629/`.
- focused result:
  the first sweep found base MMQL128 best among existing built variants on the
  actual Llama 3.2 p512 Q4 rows: `4964.757 us` versus default B-quad
  `5413.983 us`. Final cols==512 default selected base only for rows=3072
  Qcur and ffn_out, kept B-quad for Kcur and ffn_gate, and improved focused
  p512 `5464.462 -> 5073.574 us` versus rollback. The p33 non-steal gate
  passed and stayed on the existing MMQL64 BK2/B-quad narrow routes; p513
  selected only B-quad under both default and rollback, so it is unchanged by
  the default policy.
- model result:
  post-edit Llama 3.2 3B Q4_K_M p512 improved `1475.154347 ->
  1534.832691 tok/s` (`1.040x`) versus rollback. Route traces show default
  selected base MMQL128 for `350` Q4 dispatches and kept B-quad for `480`;
  rollback selected B-quad for all `830` Q4 dispatches. p513 is not included
  in the default policy and is not the primary promotion row.
- decision:
  accept the narrow rows=3072 policy. This is evidence-based route tuning, not
  a new Vulkan-clone kernel and not a broad B-quad rollback. Other Q4_K rows
  that previously benefited from B-quad remain on B-quad; p33 remains narrow.

## 2026-06-20 - Current basket after Q4 rows=3072 policy

- source:
  `sources/llama.cpp` clean at `af39e5ecf hrx: route q4 rows3072 to base mmql128`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --tag current-af39e5ecf --out cache/hrxv1/gfx1151 --models all --cases p33,p512,p513 --backends hrx,vulkan --repetitions 3 --flash-attn 1 --timeout 900`.
- artifact:
  `cache/hrxv1/gfx1151/current-af39e5ecf/`.
- result:
  full downloaded basket geomean improved to `0.6219x` average and `0.6163x`
  steady HRX/Vulkan. This is still not close to parity.
- worst steady rows:
  | Ratio | Model/case | Top HRX route |
  | ---: | --- | --- |
  | `0.499x` | Llama 3.1 8B Q8_0 `p512` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` |
  | `0.515x` | Qwen2.5 Coder 7B Q5_K_M `p512` | `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32` |
  | `0.527x` | Llama 3.1 8B Q4_K_M `p33` | `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32` |
  | `0.529x` | Qwen3 30B Q6_K `p33` | `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_f16acc_wg256_f32` |
  | `0.536x` | Llama 3.1 8B Q8_0 `p513` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` |
- decision:
  continue schedule-led work. Q8_0 remains the largest gap, but the direct
  WMMA route family is already heavily bracketed and still loses to the packed
  route. Q5 p512 is the next cheap packed-route policy target because existing
  MMQL128 variants can be swept without adding new source.

## 2026-06-20 - Q5_K Qwen2.5 p512 MMQL128 B-half policy promotion

- source:
  `sources/llama.cpp` dirty after adding a selector-only policy for
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bhalf_wg256_f32`.
- trigger:
  fresh basket artifact `cache/hrxv1/gfx1151/current-af39e5ecf/` showed
  Qwen2.5 Coder 7B Q5_K_M `p512` as the second-worst steady row at `0.515x`
  Vulkan.
- candidate:
  default B-half MMQL128 only on gfx1151 Q5_K packed-Q8_1 prompt rows with
  `rows % 128 == 0`, `cols == 512`, and full-column MMQL128 conditions.
  Rollback:
  `GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL128_BHALF_P512_PROMPT=1`.
  Broad force remains:
  `GGML_HRX_ENABLE_Q5_K_Q8_1_X4_MMQL128_BHALF_PROMPT=1`.
- evidence:
  existing-variant sweep:
  `cache/hrxv1/gfx1151/q5-p512-existing-mmql128-sweep-af39e5ecf-20260620-001049/`;
  post-edit focused regate:
  `cache/hrxv1/gfx1151/q5-p512-bhalf-default-regate-af39e5ecf-20260620-001436/`;
  force model A/B:
  `cache/hrxv1/gfx1151/q5-p512-mmql128-bhalf-model-ab-af39e5ecf-20260620-001239/`;
  post-edit model A/B:
  `cache/hrxv1/gfx1151/q5-p512-bhalf-default-model-ab-af39e5ecf-20260620-001615/`.
- focused result:
  sweep found B-half best among default, B-half, B-pair, B-quad, and CR:
  `16914.507 -> 16389.827 us` (`0.969x`). Post-edit default versus rollback
  improved `17012.774 -> 16270.776 us` (`0.956x`). p33 and p513 selected the
  same providers under default and rollback.
- model result:
  post-edit Qwen2.5 Coder 7B Q5_K_M p512 improved `575.319795 ->
  591.028398 tok/s` (`1.027x`) versus rollback. Route traces switched `620`
  Q5 dispatches from base MMQL128 to B-half over five repetitions; rows2,
  wg32, Q6, and flash-attention routes were unchanged.
- decision:
  accept the narrow p512 B-half policy. This is a small packed-route lift
  while the direct-WMMA VK128 family remains rejected for Q5 p512.

## 2026-06-20 - Current p512 rank after Q8 dual-stage rejection

- source:
  `sources/llama.cpp` at `feed7b008 hrx: reject q8 group12 dualstage repair`.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --tag current-p512-r1-after-feed7b008-20260620-054240 --cases p512 --repetitions 1 --timeout 900`.
- artifact:
  `cache/hrxv1/gfx1151/current-p512-r1-after-feed7b008-20260620-054240/`.
- result:
  eight downloaded GGUFs, p512/fa1/r1, same-machine HRX/Vulkan. Steady
  geomean is `0.582000x` Vulkan with zero HRX fallback lines. This is a
  boulder-rank checkpoint, not a promotion gate.
- worst steady p512 rows:
  | Ratio | Model | Top HRX route |
  | ---: | --- | --- |
  | `0.535x` | Llama 3.1 8B Q8_0 | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bk2_wave64_wg256_f32` |
  | `0.562x` | Llama 3.1 8B Q4_K_M | `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` |
  | `0.576x` | Qwen2.5 Coder 7B Q5_K_M | `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bhalf_wg256_f32` |
  | `0.577x` | Qwen3 30B Q6_K | `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32` |
  | `0.583x` | Qwen3 30B Q4_K_XL | `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` |
- interpretation:
  the newly downloaded Qwen3-Coder 30B Q4_K_M and Qwen3 30B Q4_K_XL rows no
  longer dominate p512 after the accepted MoE work. Q8_0 is again the lead
  p512 boulder, but the recent route-facing direct192/motif/KTileFrag and
  packed issue-order probes mean the next Q8 move should be a lower-level
  lane/store primitive or materially different packed dataflow, not another
  local stage-order or B-cache reshuffle.

## 2026-06-20 - Post Q8 chunk8 basket and Q6 p33 recheck

- source:
  `sources/llama.cpp` at `08451f1de hrx: default q8 chunk8 full-column route`.
- basket command:
  `python3 tools/hrxv1_basket_benchmark.py --models all --cases p33,p512,p513 --backends hrx,vulkan --repetitions 1 --flash-attn 1 --tag basket-after-q8-chunk8-08451f1de`.
- basket artifact:
  `cache/hrxv1/gfx1151/basket-after-q8-chunk8-08451f1de/`.
- basket result:
  eight downloaded GGUFs, p33/p512/p513/fa1/r1. Steady geomean is `0.626x`
  Vulkan with zero HRX fallback lines. The HRX binary was rebuilt immediately
  afterward so later focused rows report build commit `08451f1de`; the basket
  still selected the new Q8 chunk8 provider for Llama 3.1 Q8_0 p512.
- worst steady rows from the one-rep basket:
  | Ratio | Model/case | Top HRX route |
  | ---: | --- | --- |
  | `0.389x` | Qwen3 30B Q6_K `p33` | `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32` |
  | `0.537x` | Llama 3.1 8B Q8_0 `p512` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bk2_wave64_chunk8_wg256_f32` |
  | `0.537x` | Llama 3.1 8B Q8_0 `p513` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` |
  | `0.545x` | Qwen2.5 Coder 7B Q5_K_M `p512` | `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bhalf_wg256_f32` |
  | `0.547x` | Qwen2.5 Coder 7B Q5_K_M `p513` | `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_bquad_wg256_f32` |
- Q6 p33 recheck command:
  `python3 tools/hrxv1_basket_benchmark.py --models unsloth-qwen3-30b-a3b-instruct-2507-gguf-qwen3-30b-a3b-instruct-2507-q6-k --cases p33 --backends hrx,vulkan --repetitions 3 --flash-attn 1 --timeout 900 --tag q6-p33-recheck-after-q8chunk8-08451f1de`.
- Q6 p33 recheck artifact:
  `cache/hrxv1/gfx1151/q6-p33-recheck-after-q8chunk8-08451f1de/`.
- Q6 p33 recheck result:
  steady HRX/Vulkan is `0.515x` (`93.443` vs `181.474 tok/s`), with build
  commits `08451f1de/08451f1de`, `backends=HRX/Vulkan`, zero HRX fallback
  lines, and `576` dense Q6 p33 dispatches on the H4LOAD VK64 provider.
- interpretation:
  treat the basket's `0.389x` Q6 p33 row as a noisy one-rep outlier. The
  current mission is still far from Vulkan parity, but the stable worst cluster
  remains dense quantized prompt matmul: Q8_0 p512/p513, Q5_K p512/p513, Q6_K
  p33, and broad Q4_K production-width rows. The next route work should stay
  mechanically schedule-led from RADV evidence; for Q6 p33 specifically, the
  ledger says existing RING96/copy/staging variants have closed that source
  axis without promotion, so do not re-run those unless the lane/writeback
  primitive materially changes.

## 2026-06-20 - Q5_K motif192 small-projection opt-in probe

- source:
  `sources/llama.cpp` dirty from adding the opt-in selector
  `GGML_HRX_ENABLE_Q5_K_WMMA16_VK128_MOTIF192_SMALLPROJ_PROMPT=1` on top of
  commit `08451f1de hrx: default q8 chunk8 full-column route`.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, rebuilt through CMake/Ninja target
  `llama-bench test-backend-ops`.
- trigger:
  Q5_K p512/p513 remains a stable prompt gap in the post-Q8-chunk8 basket, but
  broad motif192 was already rejected because it regressed large Q/FFN rows.
  The prior focused table showed motif192 winning only smaller projection rows
  (`Kcur`/`Vcur`-like shapes).
- static artifact:
  `cache/hrxv1/gfx1151/q5-current-contract-refresh-08451f1de-20260620-165448/`.
  The current packed p512 default
  `mul_mat_vec_q5_k_q8_1_x4_mmql128_bhalf.hsaco` is a different `v_dot` family
  from RADV (`512 v_dot`, LDS `9728`, `64` VMEM stores, no WMMA). The closest
  current RADV-like HSACO is
  `mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_packstage_fast_half_motif192_bufferstore_wg256.hsaco`
  (`32` WMMA, LDS `22528`, `192` VMEM stores, no spills), but it still misses
  RADV's first-WMMA/load window and collapses store clusters (`2` vs RADV
  `19`).
- route or kernel candidate:
  existing motif192 provider
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_motif192_bufferstore_f16acc_wg256_f32`,
  selected only when the new opt-in is set, on gfx1151, with `k % 256 == 0`,
  `k <= 3584`, `rows <= 3584`, and `cols >= 512`.
- focused artifact:
  `cache/hrxv1/gfx1151/q5-motif192-smallproj-probe-08451f1de-dirty-20260620-165733/`.
- focused correctness:
  p33, p512, and p513 all passed CPU-reference rows. p33 selected the same
  default providers under default and opt-in. p512/p513 selected motif192 only
  for the `Kcur`-class row and left Q/FFN rows on the packed Q8_1 routes.
- focused timing:
  p33 was unchanged within noise (`0.994x` opt-in/default). p512 improved
  `16439.519 -> 16174.565 us` (`0.984x`) with `Kcur` improving
  `886.380 -> 779.397 us`. p513 focused total was slightly worse
  (`19724.834 -> 19862.544 us`, `1.007x`) despite `Kcur` improving
  `900.448 -> 726.757 us`, because unrelated packed rows moved against the
  candidate in that run.
- model artifact:
  `cache/hrxv1/gfx1151/q5-motif192-smallproj-model-ab-08451f1de-dirty-20260620-165955/`.
- model timing:
  same-binary Qwen2.5 Coder 7B Q5_K_M HRX-only p512/fa1/r3 improved
  `610.217 -> 616.180 tok/s` (`1.0098x`), and p513/fa1/r3 improved
  `519.849 -> 523.236 tok/s` (`1.0065x`). Opt-in runs had `126` motif192
  dispatches and default runs had zero.
- decision:
  keep as an opt-in diagnostic/probe and commit the selector only if retained
  as explicit evidence infrastructure. Do not default yet. The win is small and
  shape-based rather than model-basket-proven, and the direct-WMMA family still
  has the known RADV static contract miss. Promotion would require broader Q5
  model coverage plus p33/p512/p513 route traces proving that only the intended
  small-projection rows move.

## 2026-06-20 - Q5_K motif192 small-projection default promotion

- source:
  `sources/llama.cpp`, post-probe selector promoted from opt-in to default-on
  with rollback
  `GGML_HRX_DISABLE_Q5_K_WMMA16_VK128_MOTIF192_SMALLPROJ_PROMPT=1`.
  Committed as `85edc5327 hrx: default q5 motif192 small projection route`.
- route:
  `hrx_mul_mat_vec_q5_k_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_motif192_bufferstore_f16acc_wg256_f32`.
- policy:
  gfx1151 Q5_K prompt rows with `k % 256 == 0`, `k <= 3584`,
  `128 <= rows <= 3584`, and `cols >= 512`. This is intended to catch
  Kcur/Vcur-style small projections only, not Qcur or FFN rows.
- committed-source opt-in artifact:
  `cache/hrxv1/gfx1151/q5-motif192-smallproj-model-ab-committed-20260620-170557/`.
  Qwen2.5 Coder 7B Q5_K_M p512 steady improved `1.0198x`, p513 improved
  `1.0114x`, and p33 was unchanged with zero motif192 dispatches.
- focused default-vs-rollback artifact:
  `cache/hrxv1/gfx1151/q5-motif192-smallproj-default-regate-20260620-171010/`.
  All p33/p512/p513 CPU-reference rows passed. Route traces show p33 unchanged;
  p512/p513 move only `k=3584, rows=512, cols>=512` from `rows2_cols8` to
  motif192, while Qcur and FFN stay on MMQL128 packed routes. Focused sums:
  p33 `0.995x`, p512 `0.981x`, p513 `0.968x` default/rollback.
- model guardrail artifacts:
  `cache/hrxv1/gfx1151/q5-motif192-smallproj-default-model-ab-20260620-171203/`
  and repeat
  `cache/hrxv1/gfx1151/q5-motif192-smallproj-default-model-ab-repeat-20260620-171312/`.
  First default A/B was flat/noisy on p512 (`0.999x` steady) and positive on
  p513 (`1.018x`); rollback-first r7 repeat was positive on both p512
  (`1.007x`) and p513 (`1.009x`).
- decision:
  accept the narrow default with rollback. This is a small production lift and
  not a full RADV clone. Continue treating Q5 large-route parity as open unless
  later basket rows show it is no longer a boulder.

## 2026-06-20 - Post Q5 motif192 commit-aligned KPI and Q6 repeat

- source:
  `sources/llama.cpp` at
  `85edc5327 hrx: default q5 motif192 small projection route`.
- full basket artifact:
  `cache/hrxv1/gfx1151/basket-after-q5-motif192-default-85edc5327-commitaligned-r1/`.
- Q6 repeat artifact:
  `cache/hrxv1/gfx1151/q6-repeat-after-q5-motif192-85edc5327-r3/`.
- validation:
  both HRX and Vulkan rows report build commit `85edc5327`; HRX rows report
  `backends=HRX`, Vulkan rows report `backends=Vulkan`, and the summarized
  records show zero HRX fallback lines.
- basket result:
  steady HRX/Vulkan geomean is `0.642x`. Worst rows remain dense quantized
  prompt matmul, led by Qwen3 30B Q6_K p33/p512/p513, Llama 3.1 8B Q8_0
  p512/p513, and Qwen2.5 Coder 7B Q5_K_M p512/p513.
- Q6 repeated result:
  Qwen3 30B Q6_K steady geomean is `0.523x` Vulkan. Rows are p33 `0.476x`,
  p512 `0.544x`, and p513 `0.554x`. p33 selects
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32`;
  p512/p513 select
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`.
- decision:
  Q6 is still a stable boulder, but the existing RING96, WN32, wave4row,
  padladder faststage, and mixedstage probes have already been rejected by
  correctness or same-runner timing. The next Q6 code should change the
  selected-output/cooperative-store ownership primitive or another named RADV
  event-window delta, not just re-route one of the rejected wrappers.

## 2026-06-20 - Q8_0 cooperative-store contract refresh

- source:
  `sources/llama.cpp` at
  `f38c84b5e hrx: record gfx1151 post-q5 parity checkpoint`.
- artifact:
  `cache/hrxv1/gfx1151/q8-coopstore-contract-refresh-f38c84b5e-20260620-173114/`.
- tool:
  CMake/Ninja-built `hrx-hip-bench-coopmat-store-contract`.
- result:
  `wmma-lds-k2-direct192-raw` passes with `elements=12288`, `bad=0`, and
  `max_abs=0`. `wmma-lds-k2-radv-mixed192` fails with `bad=2112`,
  `max_abs=50625`, and first bad lane `group=18 slot=3 lane=0`.
- decision:
  this refresh confirms the Q8 useful axis is not another halfword-staging or
  wrapper-only route. Direct 192 raw selected-output ownership remains the
  correctness-clean lower-level surface; mixed192 still exposes a HIP C++
  register/lane contract failure. The next Q8 code should transfer the direct
  raw ownership primitive into a production-ABI diagnostic or reduce the
  mixed192 failure to a compact compiler reproducer.

## 2026-06-20 - Q8_0 accumulator-slot reuse rejection

- source:
  `sources/llama.cpp` at
  `1689ea832 hrx: record q8 coopstore refresh`.
- artifact:
  `cache/hrxv1/gfx1151/q8-coopstore-accslots-refresh-1689ea832-20260620-173516/`.
- tool:
  CMake/Ninja-built `hrx-hip-bench-coopmat-store-contract`.
- hypothesis:
  reuse eight accumulator slots under the direct raw 192-store ownership
  pattern to reduce the live surface that has been causing Q8 route-local
  resource and timing cliffs.
- result:
  `wmma-lds-k2-direct192-raw` reconfirms clean with `bad=0`.
  `wmma-lds-k2-accslots-raw192` fails with `bad=3072`, `max_abs=65472`,
  `nan=75`, and first bad lane `group=0 slot=0 lane=0`.
- decision:
  reject simple accumulator-slot reuse before any catalog transfer. This is an
  ownership-contract failure, not a timing candidate. Future Q8 lower-live-state
  work must pass the same store-contract gate before touching production route
  selection.

## 2026-06-20 - Q6_K VK64 RADV96 fragment-copy refresh

- source:
  `sources/llama.cpp` at
  `d02d00f8b hrx: record q8 accslot coopstore rejection`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-coopstore-contract-refresh-d02d00f8b-20260620-173747/`.
- tool:
  CMake/Ninja-built `hrx-hip-bench-coopmat-store-contract`.
- result:
  the synthetic `wmma-lds-vk64-radv96` control passes with `bad=0`. Plain
  `accdirect` fails with `bad=256` and NaNs/Infs. `accslots` fails with
  `bad=2816` and NaNs/Infs. `accdirect-copya`, `accdirect-copyb`, and
  `accdirect-copyab` all pass with `bad=0`.
- decision:
  Q6 p33 should not continue plain RING96, accdirect, or accslot production
  transfers. The next implementation candidate should start from the
  operand-copy-repaired accumulator ownership primitive and then solve the
  production-ABI B-tile ownership and pressure problems that caused the earlier
  catalog transfers to fail.

## 2026-06-20 - Q6_K VK64 K-loop builtin typed-stage rejection

- source:
  `sources/llama.cpp` at
  `5a60ea699 hrx: record q6 vk64 coopstore copy prior` plus the local probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-builtin-typedstage-repro-5a60ea699-20260620-174620/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-builtin-typedstage-repro`.
- hypothesis:
  keep the K-loop, copy-A+B, typed-linear store, and buffer-store contract from
  the route-facing Q6 typed-stage probe, but remove inline WMMA so the compiler
  lowers the WMMA primitive and perhaps reduces the wait/live-range shape.
- result:
  normal Q6 fixture rows pass with `bad_gt_0p25=0`, but the stress row fails
  with `bad_gt_0p25=2044/2112`. Static comparison is also unfavorable:
  builtin typed-stage is `VGPR=178`, `LDS=19456`, `scratch=0`, `16` WMMA,
  `64` `buffer_store_b32`, `96` LDS loads, `10` LDS stores, and `92`
  `s_waitcnt`; the existing inline typed-stage comparator has the same visible
  instruction counts but only `VGPR=154`.
- decision:
  reject before any selector or model-level A/B. This bracket does not move Q6
  toward the Vulkan p33 medium target (`VGPR=144`, `LDS=11264`, `96`
  `buffer_store_b32`, no spills). The next Q6 attempt still needs a lower-level
  output-ownership/store primitive that preserves the copy-repaired correctness
  contract while reaching the RADV 96-store surface.
- next axis:
  the passing low-level fixture is a 24-group/96-store pattern:
  direct `GROUPS_0_7`, then staged bands `8..13`, `14..19`, and `20..23`.
  The route-facing typed-stage path instead drains 16 groups through a 64-store
  output surface. The next useful probe should first build a production-ABI
  mini fixture that transfers this exact 24-group staged ownership pattern with
  real Q6/RHS coordinates, before touching a selector or model-level A/B.

## 2026-06-20 - Q6_K VK64 route-facing RADV96 sidecar rejection

- source:
  `sources/llama.cpp` at
  `8f0d67bef hrx: record q6 builtin typedstage rejection` plus the local
  sidecar probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-radv96-sidecar-repro-8f0d67bef-20260620-175437/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-radv96-sidecar-repro`.
- hypothesis:
  transfer the passing low-level VK64/RADV96 ownership motif into the
  route-facing Q6 ABI: direct real result stores for groups `0..7`, staged real
  result stores for groups `8..15`, and staged sidecar stores for groups
  `16..23`. The sidecar lets the diagnostic request the 24-group store surface
  without corrupting the logical result tensor.
- result:
  normal Q6 fixture rows pass with `bad_gt_0p25=0`, and every row writes all
  `2048` sidecar floats without NaNs/Infs. The stress row still fails with
  `bad_gt_0p25=2044/2112`. Static lowering does not preserve the intended
  RADV96 surface: the built HSACO still has `64` `buffer_store_b32`
  instructions rather than `96`. It does improve some local static facts versus
  inline typed-stage (`LDS 19456 -> 14336`, `ds_load 96 -> 68`,
  `s_waitcnt 92 -> 76`, `VGPR 154 -> 153`) but adds barriers (`4 -> 8`) and
  remains a diagnostic sidecar path.
- decision:
  reject as a RADV96 transfer before selector or model-level A/B. A separate
  future probe may test the banded direct/staged 16-group drain as a real route,
  because that is the useful static hint from this experiment, but this did not
  clone Vulkan's medium 96-store ownership contract.

## 2026-06-20 - Q6_K VK64 real 16-group banded-stage rejection

- source:
  `sources/llama.cpp` at
  `e10ed3617 hrx: record q6 radv96 sidecar rejection` plus the local banded
  stage probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-banded-stage-repro-e10ed3617-20260620-175952/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-banded-stage-repro`.
- hypothesis:
  keep the route-facing K-loop/copy-A+B compute path, but make the useful
  sidecar finding real: direct stores for logical groups `0..7`, staged stores
  for `8..13`, and staged stores for `14..15`, with no sidecar stores.
- result:
  normal Q6 fixture rows pass with `bad_gt_0p25=0`, but the stress row still
  fails with `bad_gt_0p25=2044/2112`. A follow-up calibration against the
  accepted VK64 repro shows the same stress failure, so this is not
  route-specific:
  `cache/hrxv1/gfx1151/q6-vk64-accepted-repro-stress-calibration-f79532fb8-20260620-180205/`.
  Static shape is the best of this K-loop staged-store family so far:
  `LDS=14336`, `VGPR=153`, no
  scratch/spills, `16` WMMA, `64` `buffer_store_b32`, `64` LDS loads, `6` LDS
  stores, `68` `s_waitcnt`, and `6` barriers. Versus inline typed-stage it
  reduces `LDS 19456 -> 14336`, `ds_load 96 -> 64`, `ds_store 10 -> 6`,
  `s_waitcnt 92 -> 68`, and `VGPR 154 -> 153`, while barriers increase
  `4 -> 6`. Standalone host timing is materially slower than accepted VK64 at
  comparable k3584 rows: `407.211 us` versus `309.269 us` on `64x33`, and
  `451.665 us` versus `352.721 us` on `64x64`.
- decision:
  reject as production route material because standalone timing regresses
  despite the improved static shape. Keep it as a static prior. The next useful
  Q6 work is either a lower-overhead spelling of the banded drain that removes
  the extra barrier cost, or a transfer of this drain idea to a Q6 compute
  schedule that can beat the accepted default in a focused backend-op gate.

## 2026-06-20 - Q6_K VK64 upper-half staged-drain rejection

- source:
  `sources/llama.cpp` at
  `8c096ba3d hrx: record q6 stress calibration` plus the local upper-stage
  probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-upper-stage-repro-8c096ba3d-20260620-180931/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-upper-stage-repro`.
- hypothesis:
  keep the same K-loop/copy-A+B compute path and direct stores for groups
  `0..7`, but stage groups `8..15` in one upper-half pass. This brackets the
  prior banded-stage result by removing the second staged-drain barrier pair.
- result:
  normal Q6 fixture rows pass with `bad_gt_0p25=0`, including odd/narrow
  `64x33` rows at `k=256`, `k=512`, and `k=3584`; `64x64 k3584`; and
  `128x33 k3584`. The synthetic stress row still fails with
  `bad_gt_0p25=2044/2112`, which matches the accepted VK64 stress calibration
  and is not route-specific enough for promotion/rejection by itself. Static
  lowering improves the banded-stage barrier surface (`s_barrier 6 -> 4`,
  `s_waitcnt 68 -> 64`) while keeping `VGPR=153`, no spills, `16` WMMA, `64`
  `buffer_store_b32`, `64` LDS loads, and `6` LDS stores. The cost is higher
  LDS (`14336 -> 15360`). Standalone timing remains slower than accepted VK64:
  `409.154 us` on `64x33 k3584` and `449.998 us` on `64x64 k3584`, versus the
  accepted VK64 calibration at `309.269 us` and `352.721 us`.
- decision:
  reject as production route material and skip model A/B. The useful conclusion
  is that simply reducing the staged-drain barrier count does not rescue this
  route-facing HIP C++ family. The next Q6 attempt needs a materially different
  output-ownership/store primitive closer to RADV's medium `96`
  `buffer_store_b32` surface, not another small staged-drain spelling.

## 2026-06-20 - Q6_K VK64 RADV96 no-merge sidecar rejection

- source:
  `sources/llama.cpp` at
  `d040c15cb hrx: record q6 upper stage probe` plus the local no-merge
  sidecar probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-radv96-sidecar-nomerge-repro-d040c15cb-20260620-181457/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-radv96-sidecar-nomerge-repro`.
- hypothesis:
  the earlier route-facing RADV96 sidecar asked for scalar raw buffer stores,
  but LLVM coalesced part of the sidecar drain into `buffer_store_b128`,
  leaving only `64` visible `buffer_store_b32` instructions. Add a compiler
  memory barrier after each raw buffer store to test whether the same real
  Q6/RHS K-loop can preserve the exact RADV-like `96` scalar-store surface.
- result:
  normal Q6 fixture rows pass with `bad_gt_0p25=0`, including odd/narrow
  `64x33` rows, and every row writes all `2048` sidecar floats with no
  NaNs/Infs. The synthetic stress row still fails with `bad_gt_0p25=2044/2112`,
  matching accepted VK64 stress behavior. Static lowering proves the immediate
  opcode-surface point: no-merge emits `96` `buffer_store_b32` and zero
  `buffer_store_b128`, while the coalesced sidecar emitted `64`
  `buffer_store_b32` plus `8` `buffer_store_b128`. The cost is severe wait
  pressure: `ds_load 68 -> 96` and `s_waitcnt 76 -> 104`, with the same
  `VGPR=153`, `LDS=14336`, and `8` barriers. Standalone timing remains slow:
  `408.219 us` on `64x33 k3584` and `449.753 us` on `64x64 k3584`, versus the
  accepted VK64 calibration at `309.269 us` and `352.721 us`.
- decision:
  reject for production and skip model A/B. Keep this as evidence that exact
  `96` scalar `buffer_store_b32` is possible from the route-facing HIP C++
  surface, but the memory-clobber no-merge spelling is not the answer. The next
  useful Q6 work needs a lower-wait ownership/store spelling, likely closer to
  explicit inline buffer-store sequencing or a different accumulator ownership
  plan, not broad compiler barriers after every store.

## 2026-06-20 - Q6_K VK64 RADV96 value-barrier sidecar rejection

- source:
  `sources/llama.cpp` at
  `1b54d92c8 hrx: record q6 no-merge sidecar probe` plus the local
  value-barrier sidecar probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-radv96-sidecar-valuebarrier-repro-1b54d92c8-20260620-181853/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-radv96-sidecar-valuebarrier-repro`.
- hypothesis:
  the memory-clobber no-merge probe proved exact `96` scalar
  `buffer_store_b32` is possible, but it raised wait pressure badly. Replace
  the full memory clobber with an empty inline asm that only consumes the stored
  value bits and byte offset, to see whether this blocks store coalescing
  without wrecking load scheduling.
- result:
  normal Q6 fixture rows pass with `bad_gt_0p25=0`, including odd/narrow
  `64x33` rows, and every row writes all `2048` sidecar floats with no
  NaNs/Infs. The synthetic stress row still fails with `bad_gt_0p25=2044/2112`,
  matching accepted VK64 stress behavior. Static lowering exactly matches the
  memory-clobber no-merge shape: `96` `buffer_store_b32`, zero
  `buffer_store_b128`, `ds_load=96`, `ds_store=10`, `s_waitcnt=104`,
  `s_barrier=8`, `VGPR=153`, and `LDS=14336`. Standalone timing is also in
  the same rejected band: `407.541 us` on `64x33 k3584` and `450.8 us` on
  `64x64 k3584`, versus accepted VK64 calibration at `309.269 us` and
  `352.721 us`.
- decision:
  reject for production and skip model A/B. Empty value/offset barriers are
  enough to force scalar stores, but not enough to preserve the lower-wait
  schedule. The next useful Q6 work should avoid compiler barriers as the
  mechanism and instead try explicit store sequencing or a different
  accumulator ownership plan that does not inflate LDS/wait pressure.

## 2026-06-20 - Q8_0 mixed192 staged-drain ordering bracket

- source:
  `sources/llama.cpp` at
  `daab8df81 hrx: bracket q8 mixed stage repro` plus the local splitstage
  fixture probes.
- artifact:
  `cache/hrxv1/gfx1151/q8-coopstore-mixed192-splitstage8-daab8df81-20260620-184051/`.
  An intermediate codegen-sensitive failed build with only the 16-group
  splitstage probe is preserved at
  `cache/hrxv1/gfx1151/q8-coopstore-mixed192-splitstage-daab8df81-20260620-183842/`.
- tool:
  CMake/Ninja-built `hrx-hip-bench-coopmat-store-contract` plus embedded
  gfx1151 object extraction from the final binary.
- hypothesis:
  the full RADV-shaped Q8 mixed192 halfword-stage fixture fails because HIP C++
  codegen cannot safely keep the whole staged writeback surface live and drain
  it as one batch. Split the staged drains into explicit batches while keeping
  the same 32 WMMA and 192 scalar-store output surface.
- result:
  the direct192 raw control passes with `bad=0`. The unsplit
  `wmma-lds-k2-radv-mixed192` control still fails with `bad=2112` and first
  bad `group=18 slot=3 lane=0`. The existing `mixed160-splitstage` control
  passes. The new `mixed192-splitstage` and `mixed192-splitstage8` modes both
  pass with `bad=0`, and both repeated five times cleanly in the final binary.
  Static extraction shows both passing mixed192 variants keep `32` WMMA,
  `64` `ds_load_b64`, `128` halfword LDS stores, `128` halfword LDS loads,
  `192` scalar `buffer_store_b32`, `VGPR=162`, `LDS=24576`, and no spills.
  The repair is not free: the unsplit failing control has `137` `s_waitcnt`
  and `3` barriers, while `mixed192-splitstage` has `140` `s_waitcnt` and
  `5` barriers, and `mixed192-splitstage8` has `143` `s_waitcnt` and
  `7` barriers.
- decision:
  accept as a low-level ordering repair, not as route material. This proves
  full 192-output staged writeback can be made semantically safe by batching
  staged drains, but the production candidate still needs a route-facing
  transfer with focused odd/tail backend-op correctness and timing evidence
  against the accepted packed-Q8_1 default before model-level A/B.

## 2026-06-20 - Q8_0 route-local full192 splitstage rejection

- source:
  `sources/llama.cpp` at
  `6d97c30a7 hrx: bracket q8 mixed192 split stage` plus the local
  route-facing full192 splitstage compile-contract probe.
- artifact:
  `cache/hrxv1/gfx1151/q8-motif192-k2-splitstage-full192-compile-6d97c30a7-dirty-20260620-184843/`.
- tool:
  CMake/Ninja-built HRX v1 catalog HSACO plus
  `tools/vulkan-oracle/summarize_hsaco_family.py`.
- hypothesis:
  the low-level mixed192 staged-drain fixture proved that splitting the staged
  accumulator writeback can repair full 192-output correctness. Transfer that
  ordering fact into the real Q8_0 motif192 K2 catalog ABI by extending the
  splitstage helper from groups `16..39` to groups `16..47`.
- result:
  the route-local wrapper compiles and preserves the intended full output
  surface: `32` WMMA, `64` `ds_load_b64`, `128` halfword LDS stores,
  `128` halfword LDS loads, and `192` scalar `buffer_store_b32`. Static
  resources fail badly: wave64, `SGPR=106`, `VGPR=256`, `LDS=22528`,
  `scratch=76`, `sgpr_spills=38`, and `vgpr_spills=18`, with `80` wait-class
  ops and `5` barriers.
- decision:
  reject before backend-op or model timing. The route-facing full192 staged
  drain lands in the same HIP C++ register/spill cliff as the prior 160-store
  splitstage wrappers. The low-level ordering repair remains useful evidence,
  but a promotable Q8 large route now needs a lower-pressure cooperative
  writeback or lane ownership primitive rather than another larger staged
  splitstage wrapper.

## 2026-06-20 - Q6_K VK128 current static delta matrix

- source:
  `sources/llama.cpp` at
  `f706b1831 hrx: record q8 splitstage full192 reject`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk128-current-static-matrix-f706b1831-20260620-185341/`.
- tool:
  `tools/vulkan-oracle/compare_amdgcn_isa.py` and
  `tools/vulkan-oracle/summarize_hsaco_family.py` against the Qwen3 30B Q6_K
  p512 Vulkan oracle.
- hypothesis:
  before adding another Q6 route, make the remaining RADV-vs-HIP VK128 gap
  explicit across the accepted default and nearby B64/fullstore/store-stage
  probes.
- result:
  RADV large is `SGPR=108`, `VGPR=192`, `LDS=22528`, `32` WMMA,
  `64 ds_load_b64`, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  `192 buffer_store_b32`, `2` barriers, and no spills. The accepted VK128
  default is clean and low-barrier, but has `LDS=20480`, no `ds_load_b64`,
  no halfword loadback, and only `64` stores. The closest structural current
  HIP probe, `b64group_store_stage`, reaches `LDS=22528` and `64 ds_load_b64`,
  but has only `64 ds_load_u16_d16`, `66 ds_store_b16`, `64` stores,
  `VGPR=196`, and `34` barriers.
- decision:
  accept as a static constraint. Do not add another isolated
  B64/fullstore/store-stage wrapper. The next Q6 dense prompt candidate needs
  a lower-level cooperative store/load ownership path or a staged writeback
  spelling that approaches RADV's two-barrier topology before it deserves
  focused runtime gates.

## 2026-06-20 - Q6_K VK128 B64GROUP typed-stage compile-contract rejection

- source:
  `sources/llama.cpp` at
  `f3cf206e5 hrx: record q6 vk128 static deltas` plus the local
  compile-only B64GROUP typed-stage wrapper.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk128-b64group-typedstage-compile-f3cf206e5-dirty-20260620-185939/`.
- tool:
  CMake/Ninja-built HRX v1 catalog HSACO plus
  `tools/vulkan-oracle/summarize_hsaco_family.py` and
  `tools/vulkan-oracle/compare_amdgcn_isa.py` against the Qwen3 30B Q6_K p512
  Vulkan oracle.
- hypothesis:
  combine the existing VK128 B64GROUP fragment-load spelling with the typed
  linear accumulator staging path to see whether HIP C++ can emit a lower
  barrier staged writeback surface closer to RADV without the 34-barrier
  `STORE_STAGE` shape.
- result:
  the wrapper builds through normal CMake/Ninja HSACO generation and has no
  spills. It preserves the B64GROUP side of the contract with `32` WMMA and
  `64 ds_load_b64`, and it keeps `2` barriers. It does not preserve the RADV
  staged writeback surface: the candidate is `SGPR=40`, `VGPR=197`,
  `LDS=20480`, `2 ds_store_b16`, zero `ds_load_u16_d16`, and only
  `64 buffer_store_b32`. The RADV large Q6 route remains `SGPR=108`,
  `VGPR=192`, `LDS=22528`, `128 ds_store_b16`, `128 ds_load_u16_d16`,
  `192 buffer_store_b32`, and `2` barriers.
- decision:
  superseded by the corrected non-RING output-branch probe below. This artifact
  is still useful as a reminder to verify that compile-only macros are actually
  reachable in the intended route family, but it should not be used for Q6
  route decisions.

## 2026-06-20 - Q6_K VK128 corrected B64GROUP stage probes

- source:
  `sources/llama.cpp` at
  `3ed63ee6f hrx: bracket q6 typed stage compile contract` plus the local
  corrected normal VK128 output-branch and dual-stage compile probe.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk128-b64group-stage-corrected-static-3ed63ee6f-dirty-20260620-190630/`.
- tool:
  CMake/Ninja-built HRX v1 catalog HSACOs plus
  `tools/vulkan-oracle/summarize_hsaco_family.py` and
  `tools/vulkan-oracle/compare_amdgcn_isa.py` against the Qwen3 30B Q6_K p512
  Vulkan oracle.
- hypothesis:
  the previous B64GROUP typed-stage wrapper did not actually reach the normal
  non-RING VK128 output branch. Correct that branch and test two compile-only
  variants: typed linear staging and a Q8-style inline-DS dual halfword stage.
- result:
  the corrected typed-stage path is not useful: `SGPR=40`, `VGPR=196`,
  `LDS=28672`, `64 buffer_store_b32`, `34` barriers, and only a partial
  halfword loadback surface. The dual-stage path proves the forced inline-DS
  halfword surface is possible in normal VK128: `LDS=22528`, `32` WMMA,
  `64 ds_load_b64`, `128 ds_load_u16_d16`, `130 ds_store_b16`, no spills, and
  `64 buffer_store_b32`. It still fails the RADV contract: Vulkan has
  `192 buffer_store_b32`, `2` barriers, and `VGPR=192`, while dual-stage has
  `34` barriers and `VGPR=196`.
- decision:
  reject both before focused CPU-reference or model timing. This closes the
  "force halfword LDS traffic from HIP C++" bracket for Q6 large VK128: it can
  be forced, but only as the wrong per-tile staged-drain topology. The next Q6
  large-route attempt needs a lower-level cooperative writeback/lane ownership
  primitive or a packed dataflow pivot that changes the barrier/store family,
  not another typed or per-tile inline-DS stage.

## 2026-06-20 - Q6_K VK64 RADV96 sidecar inline-only rejection

- source:
  `sources/llama.cpp` at
  `f3aa1d211 hrx: add radv branch64 coopstore probe` plus the local
  sidecar-inline-only diagnostic.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-radv96-sidecar-inlineonly-repro-f3aa1d211-dirty-20260620-1941/`.
- tool:
  CMake/Ninja-built catalog HSACO and
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-radv96-sidecar-inlineonly-repro`.
- hypothesis:
  the prior no-merge/value-barrier/inlineasm sidecar probes proved exact
  `96 buffer_store_b32` emission is possible, but paid for it with high wait
  pressure. Force inline `buffer_store_b32` only on the diagnostic sidecar
  stores while leaving real result stores on the normal raw-buffer helper, to
  test whether the generic helper or all-store inline asm was the cause of the
  high-wait schedule.
- result:
  normal focused rows pass with `bad_gt_0p25=0`, including `64x33` at
  `k=256`, `k=512`, and `k=3584`, `64x64 k3584`, and `128x33 k3584`.
  All sidecar rows write `2048` floats with no NaNs/Infs. The stress row
  remains in the accepted VK64 calibration failure band. Static lowering keeps
  the exact scalar store surface (`96 buffer_store_b32`, no global stores,
  no spills), but remains in the rejected high-wait family: `VGPR=153`,
  `LDS=14336`, `96` LDS reads, `10` LDS stores, `103` waitcnt-class ops, and
  `8` barriers. Host timing is also unchanged in practice: `405.131 us` on
  `64x33 k3584` and `449.793 us` on `64x64 k3584`, versus accepted VK64
  calibration around `309.269 us` and `352.721 us`.
- RADV comparison:
  the Qwen3 30B Q6_K p33 Vulkan medium oracle still has the materially better
  topology: `VGPR=144`, `LDS=11264`, `48 ds_load_b64`,
  `64 ds_load_u16_d16`, `64 ds_store_b16`, `96 buffer_store_b32`, `2`
  barriers, and final pre-WMMA `lgkmcnt(40)`. The issue-window comparison
  shows HIP at `43` pre-hot loads, `32` immediate loads, and final
  `lgkmcnt(0)`, while RADV has `59`, `48`, and `lgkmcnt(40)`.
- decision:
  reject before selector/model A/B. Store syntax alone is not the missing Q6
  p33 parity axis. The next useful Q6 p33 probe must change the pre-WMMA
  load/window, accumulator ownership, or barrier topology while preserving the
  operand-copy-repaired correctness contract; do not spend another loop on
  sidecar store-only spellings.

## 2026-06-20 - Q6_K VK64 depwait RADV-window probe

- source:
  `sources/llama.cpp` at
  `b88fbe0f4 hrx: reject q6 sidecar inline-only probe` plus the local
  depwait diagnostic.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-radv96-sidecar-b88fbe0f4-dirty-20260620-195046/`.
- tool:
  CMake/Ninja-built
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-radv96-sidecar-repro`,
  the matching CMake-generated HSACO, and
  `tools/vulkan-oracle/{summarize_hsaco_family.py,compare_amdgcn_isa.py,analyze_amdgcn_issue_window.py,extract_store_clusters.py}`.
- hypothesis:
  the prior sidecar-inline-only route missed RADV's Q6 p33 pre-WMMA issue
  window (`59` pre-hot loads, `48` immediate loads, final `lgkmcnt(40)`).
  Reuse the successful Q8 fixture technique: make the first WMMA's operands
  depend on the LDS-loaded fragments through an inline dep-copy block, paired
  with the existing pad-load ladder so `lgkmcnt(40)` is a real outstanding LDS
  window.
- result:
  after forcing a rebuild of the generated bench executable, normal focused
  rows pass with `bad_gt_0p25=0` for `64x33 k256`, `64x33 k512`,
  `64x33 k3584`, `64x64 k3584`, and `128x33 k3584`. The stress row remains
  in the known VK64 calibration failure band. Static analysis shows the
  issue-window target is exactly reproduced: both RADV and HIP_DEPWAIT report
  `16` WMMA, `59` pre-hot loads, `48` immediate LDS loads before the final
  wait, and final pre-WMMA `lgkmcnt(40)`.
- remaining mismatch:
  the candidate is not faster and is structurally heavier than RADV. Focused
  timing is `404.94 us` on `64x33 k3584`, `455.297 us` on `64x64 k3584`, and
  `408.837 us` on `128x33 k3584`, still worse than the accepted VK64
  calibration family. Static deltas also remain material: HIP_DEPWAIT has
  `VGPR=170`, `LDS=14336`, `8` barriers, `32 ds_load_u16_d16`,
  `2 ds_store_b16`, `64 buffer_store_b32`, and `8 buffer_store_b128`; RADV
  medium has `VGPR=144`, `LDS=11264`, `2` barriers,
  `64 ds_load_u16_d16`, `64 ds_store_b16`, and `96 buffer_store_b32`.
- decision:
  reject before route selector or model A/B, but accept the static finding.
  The final pre-WMMA RADV wait/load window can be forced from HIP C++ and
  should no longer be treated as the unknown Q6 p33 blocker. The next Q6 p33
  probe should preserve this issue-window contract while changing accumulator
  ownership, lower-barrier writeback topology, or the compact branch/store
  primitive.

## 2026-06-20 - Q6_K VK64 direct depwait rejection

- source:
  `sources/llama.cpp` at
  `1d2fabbb1 hrx: reject q6 depwait window probe` plus the local direct
  depwait diagnostic.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-direct-1d2fabbb1-dirty-20260620-195910/`.
- tool:
  CMake/Ninja-built
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-repro`, the matching
  CMake-generated HSACO, and
  `tools/vulkan-oracle/{summarize_hsaco_family.py,compare_amdgcn_isa.py,analyze_amdgcn_issue_window.py,extract_store_clusters.py}`.
- hypothesis:
  the prior depwait sidecar proved the RADV first-WMMA issue window can be
  forced, but it still carried extra staged sidecar barriers and writes. Remove
  the sidecar and keep direct output only to test whether the preserved
  `lgkmcnt(40)` window helps once the staged drain is gone.
- result:
  normal focused rows pass with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, `64x64 k3584`, and `128x33 k3584`. The stress
  row remains in the known VK64 calibration failure band. Static analysis shows
  a clean low-barrier window: `LDS=11264`, `2` barriers, `48 ds_load_b64`,
  `59` pre-hot loads, `48` immediate loads, and final `lgkmcnt(40)`.
- remaining mismatch:
  the route is still slower than the accepted VK64 family: `410.367 us` on
  `64x33 k3584`, `457.691 us` on `64x64 k3584`, and `413.699 us` on
  `128x33 k3584`. The pressure/writeback surface remains wrong for RADV p33:
  HIP_DIRECT is `VGPR=169`, `64 buffer_store_b32`, `2 ds_store_b16`, and no
  `ds_load_u16_d16`, while RADV medium is `VGPR=144`,
  `96 buffer_store_b32`, `64 ds_store_b16`, and `64 ds_load_u16_d16`.
- decision:
  reject before route selector or model A/B. This closes the direct-output
  depwait branch: HIP C++ can express the RADV issue window, `LDS=11264`, and
  two-barrier shape together, but that combination is not sufficient. The next
  Q6 p33 probe must attack the RADV output/lane-ownership family that emits
  `96 buffer_store_b32` plus halfword LDS writeback without increasing barrier
  count or VGPR pressure.

## 2026-06-20 - Q6_K VK64 full-tile depwait rejection

- source:
  `sources/llama.cpp` at
  `f1ab14e15 hrx: reject q6 direct depwait probe` plus the local full-tile
  depwait diagnostic.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, compiled through CMake/Ninja target
  `hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-fulltile-repro` and the
  matching generated HSACO target.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium schedule probe for Qwen3 30B Q6_K
  narrow prompt rows.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_fulltile_f16acc_wg256_f32`.
- baseline command:
  prior direct depwait artifact
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-direct-1d2fabbb1-dirty-20260620-195910/`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-fulltile-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-fulltile-f1ab14e15-dirty-20260620-200848/`.
- correctness result:
  normal focused rows passed with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, `64x64 k3584`, and `128x33 k3584`.
  The stress row remains in the known VK64 calibration failure band.
- timing result:
  `64x33 k3584` was `410.691 us`, `64x64 k3584` was `455.024 us`, and
  `128x33 k3584` was `412.987 us`, flat versus the prior direct depwait
  probe and still materially slower than accepted VK64 calibration.
- static evidence:
  the RADV first-WMMA issue window is preserved exactly: both RADV and the
  full-tile HIP probe have `59` pre-hot loads, `48` immediate LDS loads, final
  `lgkmcnt(40)`, `16` WMMA, and `2` barriers. The writeback family remains
  wrong: HIP full-tile emits `VGPR=169`, `64 buffer_store_b32`,
  `2 ds_store_b16`, no `ds_load_u16_d16`, and singleton guarded store
  clusters; RADV p33 medium emits `VGPR=144`, `96 buffer_store_b32`,
  `64 ds_store_b16`, `64 ds_load_u16_d16`, and compact direct/staged branch
  blocks.
- decision:
  reject before route selector or model A/B. The existing full-tile split is
  not enough to recover RADV's compact writeback topology under this HIP C++
  spelling. The next Q6 p33 probe should target the RADV halfword
  stage/load-store ownership or a lower-level compact direct-store primitive,
  not another full-tile/guard split around the same accumulator map.

## 2026-06-20 - Q6_K VK64 clause4 full-tile depwait rejection

- source:
  `sources/llama.cpp` at
  `ec1656f3a hrx: reject q6 full-tile depwait probe` plus the local wired
  clause4 diagnostic.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`. The generated HSACO was rebuilt through
  CMake/Ninja, and the standalone bench binary was removed and rebuilt because
  its wrapper include dependency is not tracked by Ninja.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium schedule probe for Qwen3 30B Q6_K
  narrow prompt rows.
- route or kernel candidate:
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_ring96_k2_kloop_asm_copyab_depwait_clause4_fulltile_f16acc_wg256_f32`.
- baseline command:
  prior direct and full-tile depwait artifacts
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-direct-1d2fabbb1-dirty-20260620-195910/`
  and
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-fulltile-f1ab14e15-dirty-20260620-200848/`.
- variant command:
  `HRX_Q6_REPRO_TIMING_ITERS=200 build/hrx-v1-catalog-gfx1151/bin/hrx-hip-bench-q6-wmma-ring96-kloop-asm-depwait-clause4-fulltile-repro`.
- route trace path:
  not applicable; standalone probe and catalog HSACO only.
- profile or timing artifact path:
  final wired artifact:
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-clause4-fulltile-wired-ec1656f3a-dirty-20260620-201418/`.
  The earlier stale-binary artifact
  `cache/hrxv1/gfx1151/q6-vk64-ring96-kloop-depwait-clause4-fulltile-ec1656f3a-dirty-20260620-201215/`
  is superseded.
- correctness result:
  normal focused rows passed with `bad_gt_0p25=0` for `64x33 k256`,
  `64x33 k512`, `64x33 k3584`, `64x64 k3584`, and `128x33 k3584`.
  The stress row remains in the known VK64 calibration failure band.
- timing result:
  `64x33 k3584` was `410.856 us`, `64x64 k3584` was `455.854 us`, and
  `128x33 k3584` was `414.732 us`, flat or slightly worse than direct
  depwait and still slower than accepted VK64 calibration.
- static evidence:
  the RADV first-WMMA issue window is still preserved exactly: `59` pre-hot
  loads, `48` immediate LDS loads, final `lgkmcnt(40)`, `16` WMMA, and
  `2` barriers. The inline-asm primitive does emit literal `s_clause 0x3`
  groups with four `buffer_store_b32` ops at offsets `0,16,32,48`, but the
  emitted branch ladder contains both full-tile clause blocks and guarded
  fallback blocks, raising the static store surface to `128 buffer_store_b32`
  with `VGPR=169`, `LDS=11264`, and no halfword staged writeback. RADV p33
  medium remains `VGPR=144`, `96 buffer_store_b32`, `64 ds_store_b16`,
  `64 ds_load_u16_d16`, and a mixed direct/staged branch family.
- decision:
  reject before route selector or model A/B. Literal four-store clauses are
  expressible from HIP C++, but they do not solve Q6 p33 while the source still
  emits a duplicated full/tail branch surface and omits RADV's halfword
  stage/load-store ownership. The next Q6 p33 work should target RADV's staged
  ownership map directly, not more direct-store clause forcing.

## 2026-06-20 - Q6_K VK64 H4LOAD-family repro/static refresh

- source:
  `sources/llama.cpp` at
  `81f7a86e0 hrx: reject q6 p33 staged-tail probe` plus local H4LOAD-family
  standalone repro wrappers.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`. The repros are CMake/Ninja targets:
  `hrx-hip-bench-q6-wmma-vk64-h4load-repro`,
  `hrx-hip-bench-q6-wmma-vk64-h4load-bufferstore-repro`, and
  `hrx-hip-bench-q6-wmma-vk64-h4load-prefetch2-repro`.
- model/shape:
  standalone Q6_K VK64/RADV p33-medium schedule probe for Qwen3 30B Q6_K
  narrow prompt rows, including odd/narrow `64x33`, aligned `64x64`, and
  wider odd `128x33` fixture rows.
- profile or timing artifact path:
  `cache/hrxv1/gfx1151/q6-vk64-h4load-family-repro-81f7a86e0-dirty-20260620-204835/`.
- correctness result:
  all normal focused rows passed with `bad_gt_0p25=0`, including
  `64x33 k256`, `64x33 k512`, `64x33 k3584`, `64x64 k3584`, and
  `128x33 k3584`. The stress row fails for all three variants with
  `bad_gt_0p25=2044`, matching the known VK64 calibration failure band rather
  than a new variant-specific failure.
- timing result:
  H4LOAD measured `311.401 us` on `64x33 k3584`, `352.139 us` on
  `64x64 k3584`, and `309.350 us` on `128x33 k3584`. H4LOAD bufferstore
  measured `307.143 us`, `354.416 us`, and `312.210 us`. H4LOAD prefetch2
  measured `305.589 us`, `352.576 us`, and `312.322 us`. The p33-only gains
  are small and do not generalize across the aligned or wider odd rows.
- static evidence:
  the current H4LOAD family is not a RADV schedule clone. RADV p33 medium has
  `16` WMMA, `48 ds_load_b64` immediately before the first WMMA window, final
  `lgkmcnt(40)`, `96 buffer_store_b32`, `64 ds_store_b16`,
  `64 ds_load_u16_d16`, `VGPR=144`, and `LDS=11264`. H4LOAD and prefetch2
  emit only `8` WMMA, `20 ds_load_2addr_b64`, `16 global_store_b32`,
  `2 ds_store_b16`, `VGPR=59`, and final `lgkmcnt(0)`. Bufferstore improves
  the store opcode shape to `32 buffer_store_b32` and lowers depctr waits, but
  still has only `8` WMMA, `20 ds_load_2addr_b64`, `2 ds_store_b16`, and
  `VGPR=59`.
- decision:
  do not promote from this harness. Bufferstore and prefetch2 may remain
  focused repro candidates, but they are not meaningful convergence toward the
  Vulkan/RADV p33 medium schedule. The larger parity path still needs a
  production-ABI mini fixture or kernel candidate that implements the RADV
  output ownership/writeback family directly: `16` WMMA, `96` buffer stores,
  halfword LDS stage/load-store ownership, and the `lgkmcnt(40)` first-WMMA
  window without the high-VGPR branch surfaces seen in the rejected depwait
  probes.

## 2026-06-20 - Q6_K VK64 RADV96 Q6-address fixture

- source:
  `sources/llama.cpp` at
  `0d05975c9 hrx: add q6 h4load repro benches` plus the local
  `hrx-hip-bench-coopmat-store-contract` Q6-address diagnostic mode.
- build:
  `build/hrx-v1-catalog-gfx1151`, Release, ROCm
  `/srv/vm-shared/rocm/rocm-head`, rebuilt through CMake/Ninja target
  `hrx-hip-bench-coopmat-store-contract`.
- mode:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr`.
- purpose:
  transfer the passing VK64/RADV96 copy-A+B accumulator ownership motif from
  the synthetic linear fixture into Q6-style column-major output addressing:
  `row = row_tile * 16 + lane/16 + slot*4`,
  `col = col_tile * 16 + lane%16`, and output index `col * rows + row`.
  This isolates whether real Q6 row/column masking is the blocker before
  another route-facing catalog transfer.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-fixture-0d05975c9-dirty-20260620-205652/`.
- correctness result:
  the new mode passed with `bad=0` and `max_abs=0` for `64x33`, `64x64`,
  `64x96`, `128x33`, and `128x96`. The `64x96` and `128x96` rows exercise
  the full 24-group/96-store address surface, while the `x33` rows cover the
  p33-style narrow masked surface.
- static evidence:
  the current fatbin was extracted from the rebuilt executable's
  `.hip_fatbin` section and the symbol
  `_Z58coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addrILb1ELb1EEvPfyjjj`
  was analyzed. The diagnostic emits `16 v_wmma_f16_16x16x16_f16`,
  `48 ds_load_b64`, `96 buffer_store_b32`, `80 ds_store_b16`,
  `8 ds_store_b128`, `64 ds_load_u16_d16`, and no `global_store_b32`.
  This matches RADV's headline p33 medium ownership counts for WMMA,
  pre-WMMA LDS loads, buffer stores, and halfword LDS reloads, but not its
  schedule shape: the diagnostic has `8` barriers and final first-WMMA
  `lgkmcnt(0)`, while RADV p33 medium has `2` barriers and final first-WMMA
  `lgkmcnt(40)`.
- decision:
  accept as a standalone diagnostic prior, not as a route candidate. Real
  Q6-style column-major addressing is not inherently incompatible with the
  24-group/96-store ownership motif. The next route-facing Q6 p33 attempt
  should preserve this Q6-address ownership contract while reducing the staged
  writeback phase boundaries and restoring the RADV first-WMMA outstanding LDS
  window; otherwise it will repeat the earlier high-barrier sidecar failures.

## 2026-06-20 - Q6_K VK64 Q6-address no-sink fixture rejection

- source:
  `sources/llama.cpp` at
  `297277b48 hrx: add q6 radv96 address fixture` plus the local no-sink
  `hrx-hip-bench-coopmat-store-contract` mode.
- build:
  CMake/Ninja target `hrx-hip-bench-coopmat-store-contract` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-nosink-fixture-297277b48-dirty-20260620-210212/`.
- modes:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr` and
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-nosink`.
- correctness result:
  both modes passed `64x33`, `64x64`, `64x96`, `128x33`, and `128x96` with
  `bad=0` and `max_abs=0`.
- static evidence:
  the sink variant preserved the prior headline contract:
  `16` WMMA, `48 ds_load_b64`, `96 buffer_store_b32`, `80 ds_store_b16`,
  `64 ds_load_u16_d16`, `8` barriers, and `74` `s_waitcnt`. Removing the
  artificial accumulator sink reduced visible halfword stores to the RADV count
  (`64 ds_store_b16`) but also let HIP collapse the live accumulator surface to
  only `8` WMMA. It still emitted `96 buffer_store_b32`, `48 ds_load_b64`,
  `64 ds_load_u16_d16`, and `8` barriers.
- decision:
  reject the no-sink fixture as a RADV p33 clone despite correctness. The sink
  is currently what forces the full `16` WMMA ownership surface to stay visible
  to HIP C++; removing it makes the static kernel less like RADV. The next
  useful Q6 fixture should preserve `16` WMMA without the artificial sink,
  likely by making the upper accumulator groups real consumers or changing
  accumulator ownership, then separately reduce the eight staged barriers
  toward RADV's two-barrier schedule.

## 2026-06-20 - Q6_K VK64 Q6-address upper-real fixture

- source:
  `sources/llama.cpp` at
  `950b32714 hrx: add q6 radv96 nosink fixture` plus the local upper-real
  `hrx-hip-bench-coopmat-store-contract` mode.
- build:
  CMake/Ninja target `hrx-hip-bench-coopmat-store-contract` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-upperreal-fixture-950b32714-dirty-20260620-210916/`.
- mode:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperreal`.
- purpose:
  replace the artificial accumulator sink used by the Q6-address fixture with
  real checked consumers for upper accumulator groups. Groups `0..7` store
  directly from accumulators through Q6-style addressing, groups `8..15` stage
  accumulator payload through LDS and then store through Q6-style addressing,
  and groups `16..23` preserve the synthetic staged payload surface. This tests
  whether HIP can keep the full RADV96 accumulator ownership surface without a
  dummy sink.
- correctness result:
  the mode passed `64x33`, `64x64`, `64x96`, `128x33`, and `128x96` with
  `bad=0` and `max_abs=0`.
- static evidence:
  the current executable `.hip_fatbin` was unbundled and the symbol
  `_Z68coopstore_probe_wmma_lds_vk64_radv96_accdirect_copy_q6addr_upperrealILb1ELb1EEvPfyjjj`
  was analyzed. The upper-real fixture emits `16 v_wmma_f16_16x16x16_f16`,
  `96 buffer_store_b32`, `64 ds_store_b16`, `64 ds_load_u16_d16`,
  `48 ds_load_b64`, `8 ds_store_b128`, `7` barriers, `VGPR=155`,
  `SGPR=50`, `LDS=11264`, and no spills. In the same rebuilt binary, the
  sink variant is `16` WMMA, `80 ds_store_b16`, `8` barriers, `VGPR=150`,
  while the no-sink variant still collapses to `8` WMMA.
- RADV comparison:
  against the Qwen3 30B Q6_K p33 medium oracle
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p33-fa1-20260618-061613/`,
  upper-real now matches RADV's headline output contract:
  `16` WMMA, `96 buffer_store_b32`, `64 ds_store_b16`,
  `64 ds_load_u16_d16`, `48` immediate LDS loads, `LDS=11264`, and no spills.
  The remaining deltas are schedule-topology deltas, not store-count deltas:
  RADV has `VGPR=144`, `2` barriers, only `2 ds_store_b128`, and first-WMMA
  final wait `lgkmcnt(40)` with `59` pre-hot loads, while HIP upper-real has
  `VGPR=155`, `7` barriers, `8 ds_store_b128`, and final wait `lgkmcnt(0)`
  with `48` pre-hot loads.
- decision:
  accept as the strongest Q6 p33 fixture prior so far, still not as a
  production route. It proves the artificial sink is no longer required to
  preserve the full `16` WMMA / `96` store ownership surface under Q6-style
  addressing. The next useful probe should collapse the staged writeback phases
  toward RADV's two-barrier topology and preserve outstanding LDS work into the
  first WMMA window; another store-surface spelling without that topology
  change is unlikely to close the remaining gap.

## 2026-06-20 - Q6_K VK64 Q6-address upperwide barrier-axis probe

- source:
  `sources/llama.cpp` at
  `edf980f39 hrx: add q6 radv96 upper-real fixture` plus the local upperwide
  `hrx-hip-bench-coopmat-store-contract` mode.
- build:
  CMake/Ninja target `hrx-hip-bench-coopmat-store-contract` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-upperwide-fixture-edf980f39-dirty-20260620-211355/`.
- mode:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperwide`.
- purpose:
  isolate the staged-drain barrier axis after the upper-real fixture proved the
  output ownership contract. Upperwide keeps the same direct groups `0..7`,
  real staged accumulator groups `8..15`, and staged synthetic groups `16..23`,
  but gives the fixture a 16-group staging window so all staged groups can be
  drained in one phase instead of three.
- correctness result:
  the mode passed `64x33`, `64x64`, `64x96`, `128x33`, and `128x96` with
  `bad=0` and `max_abs=0`.
- static evidence:
  upperwide preserves the useful upper-real/RADV output surface:
  `16 v_wmma_f16_16x16x16_f16`, `96 buffer_store_b32`, `64 ds_store_b16`,
  `64 ds_load_u16_d16`, `48 ds_load_b64`, `8 ds_store_b128`, and no spills.
  It reduces the staged-drain topology from upper-real's `7` barriers,
  `81 s_waitcnt`, `4 s_waitcnt_depctr`, and `4 s_waitcnt_vscnt` to
  `3` barriers, `73 s_waitcnt`, `2 s_waitcnt_depctr`, and
  `2 s_waitcnt_vscnt`.
- RADV comparison:
  upperwide moves the barrier count closer to the Qwen3 30B Q6_K p33 medium
  RADV oracle (`3` versus `2`) while keeping the same store and reload counts.
  The tradeoff is explicit: upperwide raises LDS from RADV/upper-real
  `11264` bytes to `16384` bytes and raises VGPR from upper-real `155` to
  `157` (`RADV=144`). It still does not recover RADV's first-WMMA wait shape:
  HIP remains `lgkmcnt(0)` with `48` pre-hot loads, while RADV is
  `lgkmcnt(40)` with `59` pre-hot loads.
- decision:
  accept as a diagnostic barrier-axis probe, not as a production route. It
  shows HIP can keep the correct Q6-address ownership and collapse the drain
  barriers if scratch capacity is widened, so the remaining parity problem is
  not simply "HIP cannot express the store surface." The next route-facing
  probe should search for RADV's low-barrier staging topology at the original
  `11264` byte LDS footprint, or separately attack the first-WMMA outstanding
  LDS wait window.

## 2026-06-20 - Q6_K VK64 Q6-address upperwait low-barrier probe

- source:
  `sources/llama.cpp` at
  `2a34186c7 hrx: add q6 radv96 upperwide fixture` plus the local upperwait
  `hrx-hip-bench-coopmat-store-contract` mode.
- build:
  CMake/Ninja target `hrx-hip-bench-coopmat-store-contract` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-upperwait-fixture-2a34186c7-dirty-20260620-211812/`.
- mode:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperwait`.
- purpose:
  test whether upper-real's extra staged-drain barriers are required at the
  original RADV-sized `11264` byte LDS footprint. Upperwait keeps the six-group
  staging window and Q6-address ownership from upper-real, but removes the
  inter-phase workgroup barriers and relies on same-wave LDS ordering plus
  `s_waitcnt` between staged stores and reloads.
- correctness result:
  the mode passed `64x33`, `64x64`, `64x96`, `128x33`, and `128x96` with
  `bad=0` and `max_abs=0`.
- static evidence:
  upperwait preserves the output/reload contract shared by RADV, upper-real,
  and upperwide: `16 v_wmma_f16_16x16x16_f16`, `96 buffer_store_b32`,
  `64 ds_store_b16`, `64 ds_load_u16_d16`, `48 ds_load_b64`, no global stores,
  no spills, and `LDS=11264`. It reduces barriers from upper-real's `7` to
  `1` without upperwide's larger scratch window. It also lowers wait forms to
  `73 s_waitcnt`, `1 s_waitcnt_depctr`, and no `s_waitcnt_vscnt`, versus
  upper-real's `81`, `4`, and `4`.
- RADV comparison:
  upperwait is now closer than upper-real/upperwide on LDS footprint and
  barrier topology: `LDS=11264` like RADV and `1` barrier versus RADV's `2`.
  The remaining deltas are still material: RADV uses `VGPR=144`,
  `2 ds_store_b128`, first-WMMA final wait `lgkmcnt(40)`, and `59` pre-hot
  loads; upperwait uses `VGPR=154`, `8 ds_store_b128`, first-WMMA final wait
  `lgkmcnt(0)`, and `48` pre-hot loads.
- decision:
  accept as the best current Q6-address fixture prior, not as a production
  route. It proves the low-barrier staged-drain topology is expressible at the
  original LDS footprint and that correctness does not require the extra
  workgroup barriers. The next schedule probe should stop modifying the
  writeback surface and instead target the pre-WMMA load/wait window:
  reproduce RADV's `lgkmcnt(40)` first-WMMA issue shape or explain why HIP C++
  forces all 48 LDS loads to complete before the first WMMA.

## 2026-06-20 - Q6_K VK64 Q6-address depwait first-WMMA rejection

- source:
  `sources/llama.cpp` at
  `8cc447285 hrx: add q6 radv96 upperwait fixture` plus the local depwait and
  depnomem `hrx-hip-bench-coopmat-store-contract` modes.
- build:
  CMake/Ninja target `hrx-hip-bench-coopmat-store-contract` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifacts:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-upperdepwait-fixture-8cc447285-dirty-20260620-212133/`
  and
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-upperdepnomem-fixture-8cc447285-dirty-20260620-212425/`.
- modes:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperdepwait` and
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperdepnomem`.
- purpose:
  test whether upperwait's first-WMMA `lgkmcnt(0)` was caused by the explicit
  full wait in `coopstore_probe_accumulate_wmma_from_lds_ring12_copy`, or by
  the `memory` clobber in the copy-fragment `v_mov` asm. Depwait removes the
  explicit full wait before the WMMA loop. Depnomem also uses a copy primitive
  without a memory clobber.
- correctness result:
  both modes passed `64x33`, `64x64`, `64x96`, `128x33`, and `128x96` with
  `bad=0` and `max_abs=0`.
- static evidence:
  both modes preserve the upperwait output/reload contract: `16` WMMA,
  `96 buffer_store_b32`, `64 ds_store_b16`, `64 ds_load_u16_d16`,
  `48 ds_load_b64`, `1` barrier, `LDS=11264`, `VGPR=154`, and no spills.
  Removing the explicit wait reduces total `s_waitcnt` count from upperwait's
  `73` to `72`, but it does not recover RADV's first-WMMA issue window.
  Depwait and depnomem both still emit first-WMMA final wait `lgkmcnt(0)`,
  with the same `48` pre-hot LDS loads and no immediate loads before that
  final wait. Depnomem is static-equivalent to depwait for this contract, so
  the copy asm memory clobber is not the blocker.
- RADV comparison:
  the Qwen3 30B Q6_K p33 medium oracle remains materially different:
  `lgkmcnt(40)` before the first WMMA, `59` pre-hot loads including `11` VMEM
  loads, `2` barriers, `VGPR=144`, and only `2 ds_store_b128`. The depwait
  probes match the store surface and LDS footprint, but not RADV's load/wait
  overlap.
- decision:
  reject both as first-WMMA-window solutions. The explicit source wait and the
  copy-fragment memory clobber are not sufficient explanations for the HIP
  `lgkmcnt(0)` cliff. The next probe should change the load grouping/dataflow
  itself, for example by loading only the operand fragments needed by the first
  one or two WMMAs before issuing them, then loading later fragments, instead
  of materializing all six A/B ring fragments up front.

## 2026-06-20 - Q6_K VK64 Q6-address explicit first-WMMA wait probe

- source:
  `sources/llama.cpp` at
  `6366a6eb8 hrx: add q6 radv96 depwait fixtures` plus the local expwait
  `hrx-hip-bench-coopmat-store-contract` mode.
- build:
  CMake/Ninja target `hrx-hip-bench-coopmat-store-contract` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q6-vk64-radv96-q6addr-upperexpwait-fixture-6366a6eb8-dirty-20260620-212919/`.
- mode:
  `wmma-lds-vk64-radv96-accdirect-copyab-q6addr-upperexpwait`.
- purpose:
  test whether an explicit decreasing LDS wait schedule can reproduce RADV's
  first-WMMA issue shape after the depwait and depnomem probes showed that
  simply removing the full source wait is not enough. Expwait keeps the
  upperwait low-barrier Q6-address writeback topology, loads the same six A/B
  ring fragments, and then places explicit waits before the WMMA sequence:
  `lgkmcnt(40)`, `36`, `32`, `28`, `24`, and so on.
- correctness result:
  the mode passed `64x33`, `64x64`, `64x96`, `128x33`, and `128x96` with
  `bad=0` and `max_abs=0`.
- static evidence:
  expwait preserves the current best fixture contract: `16` WMMA,
  `96 buffer_store_b32`, `64 ds_store_b16`, `64 ds_load_u16_d16`,
  `48 ds_load_b64`, `1` barrier, `LDS=11264`, `VGPR=154`, and no spills.
  It increases total `s_waitcnt` from upperwait's `73` to `88`, as expected
  from the inserted wait ladder, but it is the first HIP fixture in this series
  to emit first-WMMA final wait `lgkmcnt(40)` instead of `lgkmcnt(0)`.
- RADV comparison:
  expwait now matches RADV's headline first-WMMA dependency shape:
  `48` immediate LDS loads followed by `lgkmcnt(40)` before the first
  `v_wmma_f16_16x16x16_f16`. The remaining deltas are resource and local
  schedule details: RADV has `VGPR=144`, `2` barriers, `2 ds_store_b128`,
  `59` pre-hot loads including `11` VMEM loads, and `117 s_waitcnt`, while
  expwait has `VGPR=154`, `1` barrier, `8 ds_store_b128`, `48` pre-hot loads,
  and `88 s_waitcnt`.
- decision:
  accept as the strongest Q6 p33 schedule fixture prior so far, still not a
  production route. The next useful step is to transfer this explicit
  first-WMMA wait ladder into the route-facing Q6 VK64 p33 diagnostic or the
  relevant generated HIP catalog candidate and run focused correctness/timing
  there. If route-facing performance does not move, compare the remaining
  `VGPR`, `ds_store_b128`, and scalar/VMEM pre-hot deltas before adding more
  fixture variants.

## 2026-06-21 - Q8_0 large baseoff split59 issue-window probe

- source:
  `sources/llama.cpp` commits
  `fd70da2e0 hrx: add q8 split issue-window probe` and
  `67045a7c9 hrx: record q8 split issue-window result`.
- build:
  CMake/Ninja target `hrx-hip-bench-q8-wmma-repro` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/q8-baseoff-split59-probe-fd70da2e0-20260621-001116/`.
- mode:
  `bm128-direct192-baseoff-split59-raw-asm-output`.
- purpose:
  test the concrete Q8 large-route RADV/HIP issue-window delta exposed by the
  new static screen. RADV issues `59` LDS loads before the first
  `s_waitcnt lgkmcnt(51)`, performs four WMMAs, then issues five more A-side
  LDS loads. The prior HIP baseoff fixture issued all `64` LDS loads before
  the first WMMA. The split59 fixture delays the same five A-side loads.
- correctness result:
  passed `128x128`, narrow `128x33`, and sampled `4096x513` tail coverage with
  `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`.
- static evidence:
  `static/hip-split59-issue-window.json` passes the executable issue-window
  contract: one hot region, `32` f16 WMMAs, final first-WMMA wait
  `lgkmcnt(51)`, and exactly `59` immediate LDS loads. The control baseoff
  symbol still fails the same contract with `64` immediate LDS loads.
- timing evidence:
  same-run fixture timing shows a small positive signal. On `4096x512`,
  baseoff was `3204.14 us` and split59 was `3173.51 us`; no-store baseoff was
  `3026.60 us` and no-store split59 was `3009.54 us`. On `4096x513`, baseoff
  was `3320.93 us` and split59 was `3306.26 us`; no-store baseoff was
  `3143.98 us` and no-store split59 was `3107.06 us`.
- decision:
  accept as a diagnostic schedule prior, not as a production route. The
  split-load window is now worth transferring into a production-ABI Q8_0 large
  candidate, but promotion still requires focused p512/p513 route correctness,
  route trace selection, static issue-window pass, and same-binary model A/B
  wins over the current route.

## 2026-06-21 - F16 WMMA inline compact accumulator primitive probe

- source:
  `sources/llama.cpp` at
  `7172c1b12 hrx: record q8 split59 ownership blocker` plus the local
  `hrx-hip-bench-wmma-f16-lane-map --mode=inline-compact-acc` probe.
- build:
  CMake/Ninja target `hrx-hip-bench-wmma-f16-lane-map` in
  `build/hrx-v1-catalog-gfx1151`, ROCm `/srv/vm-shared/rocm/rocm-head`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-inline-compact-acc-7172c1b12-dirty-20260621-004515/`.
- purpose:
  test the most direct remaining HIP C++ inline-asm axis for the RADV Q6/Q8
  compact accumulator mismatch. The Q8 split59 production route matched the
  visible issue window but still lost badly, and the ownership extractor showed
  the structural mismatch: RADV emits width-4 f16 WMMA `dst`/`C` operands while
  HIP emitted width-8. This probe tries three operand-constraint spellings:
  tied read/write `+v(out)`, separate output with `v(acc)` C, and output with
  matching `0(acc)` C.
- runtime result:
  completed with nonzero output:
  `wmma-f16-inline-compact-acc variants=3 lanes=64 words=768 zero_words=0`.
- static evidence:
  the executable `.hip_fatbin` was extracted with `llvm-objcopy` and
  `clang-offload-bundler`, then disassembled from `device.hsaco`. The selected
  symbols were:
  `_Z33wmma_f16_inline_compact_acc_probeILi0EEvPj`,
  `_Z33wmma_f16_inline_compact_acc_probeILi1EEvPj`, and
  `_Z33wmma_f16_inline_compact_acc_probeILi2EEvPj`.
  `extract_wmma_ownership.py --require-compact-f16-accumulators` failed all
  three variants. Each emitted one `v_wmma_f16_16x16x16_f16` with `dst`
  width8 and `C` width8, and zero width4 accumulator operands.
- decision:
  reject these inline-asm constraint spellings as a RADV compact-accumulator
  primitive. This closes the obvious HIP inline-asm operand-constraint axis for
  f16 WMMA compact dst/C ownership on gfx1151. Future Q6/Q8 f16-WMMA
  oracle-clone work should use a different lowering/primitive path, or be
  explicitly documented as a width-8 deviation with focused p33/p512/p513
  timing evidence before route promotion.

## 2026-06-21 - LLVM MC compact WMMA form screen

- source:
  no source change; local ROCm LLVM assembler screen after
  `3ab4a1c89 hrx: add compact wmma inline asm probe`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-llvm-mc-compact-form-3ab4a1c89-20260621-004934/`.
- purpose:
  determine whether the compact width-4 f16 WMMA operand form is blocked only
  by HIP C++ inline-asm constraints or also by the local LLVM gfx1151 assembler
  target description.
- static evidence:
  `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1151 --show-encoding` accepts
  the width8 form:
  `v_wmma_f16_16x16x16_f16 v[0:7], v[0:7], v[0:7], v[0:7]`. It rejects the
  width4 form:
  `v_wmma_f16_16x16x16_f16 v[0:3], v[0:3], v[0:3], v[0:3]` with
  `operands are not valid for this GPU or mode`. Adding
  `-mattr=+wmma-128b-insts,+wavefrontsize64` still rejects the width4 form.
- decision:
  treat RADV's compact f16 WMMA accumulator ownership as unavailable through
  the straightforward ROCm LLVM assembler path for gfx1151, not merely through
  HIP C++ constraints. The next exact-clone options are a target-description or
  compiler/runtime change, a non-LLVM lower-level code-object path, or a
  different measured schedule family. Do not spend another route-facing loop on
  C++ inline-asm operand spelling alone.

## 2026-06-21 - Current Q4_K packed issue-screen checkpoint

- source:
  `sources/llama.cpp` at `425e16cf2 hrx: add wmma mc compact probe` plus the
  new reusable static wrapper
  `tools/vulkan-oracle/run_q4_packed_issue_screen.py`.
- artifact:
  `cache/hrxv1/gfx1151/q4-packed-issue-screen-current-425e16cf2-20260621-014443/`.
- purpose:
  make the next Q4 dense prompt step evidence based by comparing all current
  CMake/Ninja-built packed Q4_K HSACOs against the separate RADV large
  p512/p513 and narrow p33 oracle contracts. This is a static screen only; it
  does not promote or select a route.
- large p512/p513 result:
  the RADV `aligned_l` contract has a cooperative-matrix schedule with
  first-window `v_wmma`, final pre-hot `lgkmcnt(24)`, and a much larger
  buffer/LDS store surface. The nearest current packed row by the screen is
  `mul_mat_vec_q4_k_q8_1_x4_mmql128_bk2.hsaco`, but it is still an
  integer-dot route with `1024` `v_dot`, zero `v_wmma`, final `lgkmcnt(2)`,
  and the prior focused BK2 p512/p513 runtime rejection remains authoritative.
  Accepted B-quad remains resource-clean but has the same structural gap:
  zero `v_wmma`, only `16` pre-hot loads, final `lgkmcnt(9)`, and global-store
  writeback instead of the RADV cooperative surface.
- narrow p33 result:
  the RADV `aligned_m` contract has `48` immediate LDS loads, final
  `lgkmcnt(40)`, and `16` first-window WMMA hot ops. The accepted
  `mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2_bquad.hsaco` remains the closest
  packed narrow row, but it is still an integer-dot route with zero `v_wmma`,
  final `lgkmcnt(9)`, and a large store/WMMA gap.
- decision:
  accept this as a static checkpoint and keep the Q4 packed route evidence
  loop active. Do not rerun BK2/B-pair/B-quad/B-half/B-oct as if they were new
  axes. The next Q4 route attempt needs a materially different lower-level
  primitive or packed dataflow that changes the store surface, first-hot load
  window, or WMMA-equivalent ownership while preserving the accepted
  p33/p512/p513 route-policy split.

## 2026-06-21 - Q4/Q5 direct-WMMA compact accumulator screen

- source:
  `sources/llama.cpp` at `364fe690b hrx: add q4 packed issue screen`; no new
  kernel source change, only static evidence recorded in the gfx1151 tuning
  catalog.
- artifacts:
  `cache/hrxv1/gfx1151/q4-w64-b64group-compact-screen-364fe690b-20260621-014858/`,
  `cache/hrxv1/gfx1151/q5-large-motif192-compact-screen-364fe690b-20260621-014858/`,
  and
  `cache/hrxv1/gfx1151/q5-p33-wave4row-compact-screen-364fe690b-20260621-014858/`.
- purpose:
  test whether the current Q4_K and Q5_K direct-WMMA HIP C++ candidates pass
  the RADV compact f16 accumulator ownership prerequisite before spending more
  focused timing or selector work.
- static evidence:
  RADV Q4/Q5 large `aligned_l` uses `32` WMMA with width8 A/B operands and
  compact width4 dst/C operands. HIP Q4 b64group and Q5 motif192 both emit
  width8 dst/C operands for all `32` WMMA. Q5 motif192 is the important near
  miss: it already matches the large RADV store-surface headline with `192`
  `buffer_store_b32`, `128` `ds_store_b16`, `128` `ds_load_u16_d16`, and `64`
  `ds_load_b64`, but it still fails compact accumulator ownership and has
  heavier wait/barrier topology. For Q5 p33, RADV has `16` WMMA, compact
  width4 dst/C, `96` buffer stores, and `64` halfword store/loadback ops;
  HIP wave4row has only `8` WMMA, width8 dst/C, `16` buffer stores, and no
  `ds_load_u16_d16`.
- decision:
  reject more source-level Q4/Q5 direct-WMMA clone variants in this family
  unless they change the lower-level accumulator ownership primitive or
  explicitly pivot to a measured packed-Q8_1 dataflow. This extends the Q8
  compact-accumulator blocker to Q4/Q5 and explains why matching tile geometry
  or store count alone has not reached Vulkan parity.

## 2026-06-21 - Q6 accepted dense route compact accumulator screen

- source:
  `sources/llama.cpp` at `56e8e5cca hrx: record q4 q5 compact wmma blocker`;
  no kernel source change, only a static screen against existing accepted
  CMake/Ninja-built Q6 dense prompt HSACOs.
- artifacts:
  `cache/hrxv1/gfx1151/q6-large-accepted-compact-screen-56e8e5cca-20260621-continue/`
  and
  `cache/hrxv1/gfx1151/q6-p33-h4load-compact-screen-56e8e5cca-20260621-continue/`.
- purpose:
  check whether the current accepted dense Q6_K routes explain the remaining
  repeated Qwen3 30B Q6_K gap (`~0.523x` Vulkan steady geomean after accepted
  Q5 motif192 work) through the same RADV compact-accumulator/writeback delta.
- static evidence:
  Q6 large p512/p513 accepts the same headline `32` WMMA count as RADV, but
  HIP emits width8 dst/C for all `32` WMMA, zero `ds_load_b64` in the extracted
  symbol, `64` global stores, `2` `ds_store_b16`, no `ds_load_u16_d16`, and
  `128` depctrs. RADV large uses compact width4 dst/C, `64` `ds_load_b64`,
  `192` buffer stores, `128` `ds_store_b16`, `128` `ds_load_u16_d16`, and two
  barriers. Q6 p33 H4LOAD is farther from the oracle: HIP has `8` WMMA,
  width8 dst/C, `16` global stores, `2` `ds_store_b16`, and no
  `ds_load_u16_d16`; RADV p33 has `16` WMMA, compact width4 dst/C, `48`
  `ds_load_b64`, `96` buffer stores, and `64` halfword store/loadback ops.
- decision:
  this extends the compact-accumulator/writeback blocker across Q4_K, Q5_K,
  Q6_K, and Q8_0 direct-WMMA clone attempts on gfx1151. Stop adding local
  VK128/VK64 HIP C++ WMMA wrappers as the primary Q6 path unless they change
  the lower-level ownership primitive. The next Q6 route-facing axis should be
  either a lower-level cooperative ownership/codegen path or a measured
  packed-Q8_1/x4 dataflow pivot with focused p33/p512/p513 evidence.

## 2026-06-21 - RADV mixed compact WMMA assembler correction

- source:
  updated `sources/llama.cpp/tools/vulkan-oracle/run_wmma_mc_compact_probe.py`
  after `ee02cd25e hrx: record q6 compact wmma blocker`.
- artifacts:
  default target mode:
  `cache/hrxv1/gfx1151/wmma-mc-mixed-compact-default-ee02cd25e-20260621-continue/`;
  explicit feature mode:
  `cache/hrxv1/gfx1151/wmma-mc-mixed-compact-mattr-ee02cd25e-rerun-20260621-continue/`.
- purpose:
  correct the earlier LLVM-MC primitive screen. The prior probe tested an
  all-width4 operand form, but the RADV dense prompt ISA uses mixed operands:
  width4 dst/C accumulators with width8 A/B matrix operands.
- static evidence:
  default `llvm-mc -mcpu=gfx1151` accepts the old width8 form and rejects the
  RADV mixed form. With
  `-mattr=+wmma-128b-insts,+wavefrontsize64`, the exact RADV mixed form such
  as
  `v_wmma_f16_16x16x16_f16 v[72:75], v[96:103], v[104:111], v[72:75]`
  assembles successfully; the old width8 dst/C form is rejected in that mode.
- decision:
  compact accumulator ownership is not an absolute assembler impossibility.
  It is a target-feature/codegen plumbing problem for the HIP C++ path. The
  next useful primitive is a CMake/Ninja-built HIP inline-asm HSACO that emits
  the mixed width4 dst/C + width8 A/B form and passes
  `extract_wmma_ownership.py --require-compact-f16-accumulators`. Only after
  that should a route-facing Q6/Q8 candidate be attempted.

## 2026-06-21 - HIP C++ mixed compact WMMA primitive verified

- source:
  `sources/llama.cpp` working tree after
  `ee02cd25e hrx: record q6 compact wmma blocker`.
- artifact:
  `cache/hrxv1/gfx1151/wmma-f16-mixed-compact-hip-bench-ee02cd25e-dirty-20260621-continue/`.
- purpose:
  check whether a CMake/Ninja-built HIP C++ inline-asm fixture can emit the
  RADV mixed compact f16 WMMA form before attempting a route-facing Q6/Q8
  clone.
- static evidence:
  the bench builds and runs with
  `wmma-f16-mixed-compact lanes=64 words=256 zero_words=0 first=0x3c004c00`.
  Its extracted device object contains the compact encoding. Ordinary
  `llvm-objdump -d --mcpu=gfx1151` prints that instruction as width8
  dst/C (`v[9:16]`), but the same bytes disassembled with
  `--mattr=+wmma-128b-insts,+wavefrontsize64` print as compact mixed
  dst/C (`v[9:12]`) with width8 A/B (`v[1:8]`). The ownership extractor then
  passes `--require-compact-f16-accumulators`.
- decision:
  the immediate blocker is no longer "HIP C++ cannot emit the primitive".
  The concrete next route work is to put this `uint32x4` accumulator /
  `half16` A/B inline-asm primitive into a real dense Q6/Q8 candidate, then
  require focused p33/p512/p513 correctness, route trace, static compact
  ownership with the WMMA-128b objdump mattr, and timing before promotion.
  Compact-screen tooling now disassembles HIP HSACOs with that mattr by
  default to avoid false width8 rejections.

## 2026-06-21 - Corrected Q4/Q5/Q6 compact screen results

- source:
  same `sources/llama.cpp` working tree after the WMMA-128b objdump correction.
- artifacts:
  `cache/hrxv1/gfx1151/q4-w64-b64group-compact-screen-fixed-disasm-ee02cd25e-20260621-continue/`,
  `cache/hrxv1/gfx1151/q5-large-motif192-compact-screen-fixed-disasm-ee02cd25e-20260621-continue/`,
  `cache/hrxv1/gfx1151/q5-p33-wave4row-compact-screen-fixed-disasm-ee02cd25e-20260621-continue/`,
  `cache/hrxv1/gfx1151/q6-large-accepted-compact-screen-fixed-disasm-ee02cd25e-20260621-continue/`,
  and
  `cache/hrxv1/gfx1151/q6-p33-h4load-compact-screen-fixed-disasm-ee02cd25e-20260621-continue/`.
- purpose:
  re-run the prior direct-WMMA compact accumulator screens with
  `llvm-objdump --mattr=+wmma-128b-insts,+wavefrontsize64`.
- corrected static evidence:
  all five current Q4/Q5/Q6 direct-WMMA HSACOs pass the compact f16
  accumulator screen. The previous width8 dst/C rejection was a disassembly
  artifact. Q5 large motif192 is now the closest static clone: `32` WMMA,
  `64` `ds_load_b64`, compact dst/C, `192` `buffer_store_b32`, and `128`
  `ds_store_b16`, matching RADV headline counts. Q4 large and Q6 large still
  have much smaller output surfaces (`64` global stores and only `2`
  `ds_store_b16`). Q6 p33 still has half RADV's WMMA count (`8` vs `16`) and
  no parsed `ds_load_b64`; Q5 p33 is a different over-issued shape (`40` WMMA
  and `104` `ds_load_b64`) with fewer stores than RADV.
- decision:
  do not pursue "fix width8 accumulators" as the next boulder. The active
  direct-WMMA work should focus on store/writeback surface, LDS loadback,
  wait/barrier ordering, p33-specific route policy, lane-map correctness, and
  measured p33/p512/p513 timing. The Q5 large motif192 route deserves the next
  focused correctness/timing and wait/store-address investigation before
  another broad variant sweep.

## 2026-06-21 - Q5 motif192 exact focused regate after compact-screen correction

- source:
  `sources/llama.cpp` at
  `793ba1865 hrx: fix gfx1151 compact wmma screening`.
- artifact:
  `cache/hrxv1/gfx1151/q5-motif192-current-regate-exact-793ba1865-20260621-022123/`.
- purpose:
  refresh the current default-vs-rollback evidence for the Q5 large motif192
  route using the exact focused Qwen2.5 Coder 7B Q5_K_M p33/p512/p513 rows
  from `cache/hrxv1/gfx1151/q5-mmql128-bquad-focused-20260618/`. This replaces
  the non-authoritative diagnostic refresh that accidentally used an older
  p512 file from `q5-motif192-opgate-20260620-111343`.
- gate:
  default and rollback both passed CPU-reference `test-backend-ops` correctness
  for all 4 rows at p33, p512, and p513. Rollback used
  `GGML_HRX_DISABLE_Q5_K_WMMA16_VK128_MOTIF192_SMALLPROJ_PROMPT=1`.
- route evidence:
  p33 is unchanged by design and stays on `rows2_cols8` for `Kcur` plus the
  narrow MMQL64x64 x4 routes for Qcur/FFN rows. p512 and p513 select motif192
  only for `k=3584, rows=512, cols=512/513`; Qcur and FFN rows stay on
  existing MMQL128x128 routes. Rollback moves only that small projection row
  back to `rows2_cols8`.
- focused timing:
  selected-row `Kcur` improves from `894.233 us` to `780.326 us` at p512
  (`1.146x`) and from `894.608 us` to `721.101 us` at p513 (`1.241x`).
  p33 is effectively unchanged. The full four-row focused set remains noisy
  because the larger Qcur/FFN rows are not using motif192, including one p512
  FFN row that moved against the selected-row win.
- decision:
  keep the current narrow motif192 default policy, but do not broaden it based
  on this result. Treat it as a validated selected-row improvement and continue
  Vulkan-oracle parity work on the larger Q5/Q4/Q6 prompt rows and their
  wait/barrier, LDS loadback, writeback surface, and lane-map deltas.

## 2026-06-21 - Q5 motif192 current issue-window refresh

- source:
  `sources/llama.cpp` at
  `672a5c6ed hrx: record q5 motif192 exact regate`.
- artifact:
  `cache/hrxv1/gfx1151/q5-motif192-current-issue-window-672a5c6ed-20260621-022817/`.
- purpose:
  tie the Q5 motif192 static schedule gap to the current committed source and
  corrected gfx1151 WMMA disassembly, after proving the route is only a narrow
  small-projection win.
- static evidence:
  RADV and HIP motif192 both have the headline direct-WMMA surface: `32` WMMA,
  `64` `ds_load_b64`, `128` `ds_load_u16_d16`, `128` `ds_store_b16`, and
  `192` `buffer_store_b32`, with compact dst/C ownership confirmed by the
  fixed-disasm ownership screen. HIP remains no-spill wave64 with LDS `22528`,
  SGPR `56`, and VGPR `199`, while RADV reports SGPR `108`, VGPR `192`, LDS
  `22528`, and no spills.
- issue-window evidence:
  RADV splits the `32` WMMA into `3` hot regions. The first hot region has
  `64` pre-loads (`32` VMEM plus `32` LDS), `32` immediate LDS loads before
  the final wait, and enters WMMA at `lgkmcnt(24)`. HIP motif192 has `2` hot
  regions. Its first hot region has only `24` LDS pre-loads, `1` immediate LDS
  load before the final wait, and enters WMMA at `lgkmcnt(0)`. HIP also emits
  many more wait/dependency counters: `245` `s_waitcnt` plus `21`
  `s_waitcnt_depctr` versus RADV `185` plus `1`.
- store-cadence evidence:
  RADV exposes `19` label-independent store clusters and `5` store motifs,
  while HIP motif192 collapses writeback into `2` broad clusters and `2`
  motifs, with one extra barrier (`3` vs `2`). The counts match, but the
  cadence and dependency shape do not.
- decision:
  the next Q5 direct-WMMA candidate should be a bracketed load/window and
  store-cadence probe that tries to reproduce RADV's first-region pre-load
  window and delayed wait ladder. Do not spend the next pass broadening
  motif192 routing or cloning only the headline WMMA/store counts.

## 2026-06-21 - Q5 motif192 RADVLadder static rejection

- source:
  `sources/llama.cpp` at `9a77735f1-dirty`.
- artifact:
  `cache/hrxv1/gfx1151/q5-motif192-radvladder-static-9a77735f1-dirty-20260621-024106/`.
- purpose:
  test a narrow, RADV-derived issue-ladder sibling of the current Q5 motif192
  route. The probe preserves motif192 writeback, CMake/Ninja HSACO generation,
  and the headline `32` WMMA / `64` `ds_load_b64` / `192` buffer-store surface,
  but changes operand loading to `A0,B0,B1,B2,B3,A1,A2,A3` and row-major WMMA
  issue with explicit `lgkmcnt 12/8/4/0` waits.
- static evidence:
  the HSACO remains resource-clean: wave64, SGPR `56`, VGPR `199`, LDS
  `22528`, no spills. The first emitted WMMAs do reuse `A0` across `B0..B3`,
  matching one visible RADV motif.
- rejection evidence:
  the compiler did not preserve the intended wait ladder. RADV's first hot
  region has `12` WMMA, `64` pre-loads, `32` immediate LDS loads, and enters at
  `lgkmcnt(24)`. The HIP RADVLadder first region has `8` WMMA, only `24`
  pre-loads, `0` immediate loads, and enters at `lgkmcnt(0)`. The explicit
  `lgkmcnt 12/8/4/0` waits appear after the first eight WMMAs, not before the
  intended uses.
- decision:
  reject before focused correctness or timing. This source-level C++ ladder is
  not a valid mechanical port of the RADV schedule. The next Q5 direct-WMMA
  attempt needs a lower-level scheduling spelling, such as inline WMMA/wait
  blocks that keep waits attached to the intended uses, or a pivot toward the
  packed-Q8_1 dataflow if direct-WMMA continues to resist schedule control.

## 2026-06-21 - Q5 motif192 ASMWAIT small-projection default

- source:
  `sources/llama.cpp` at `463473bdd-dirty`.
- static artifact:
  `cache/hrxv1/gfx1151/q5-motif192-asmwait-static-463473bdd-dirty-20260621-024909/`.
- focused artifact:
  `cache/hrxv1/gfx1151/q5-motif192-asmwait-default-gate-20260621-030041/`.
- purpose:
  follow the rejected RADVLadder probe with a lower-level spelling that keeps
  the intended `lgkmcnt` waits attached to the matching `v_wmma` instructions
  through inline asm, while preserving normal CMake/Ninja HSACO generation.
- static evidence:
  ASMWAIT preserves the direct-WMMA surface and fixes the source-level wait
  floating failure: emitted WMMAs remain after their intended waits and the
  first hot region now has `32` immediate LDS loads before the final wait.
  The route is still not a full RADV clone: HIP ASMWAIT has `2` hot regions,
  final pre-WMMA `lgkmcnt(12)`, `233` waits, `3` barriers, and VGPR `215`,
  versus RADV's `3` regions, `lgkmcnt(24)`, `186` waits, `2` barriers, and
  VGPR `192`.
- focused correctness:
  robust CSV parsing confirms all p33, p512, and p513 CPU-reference rows pass
  for both default and rollback (`4/4` each, `24/24` total).
- route evidence:
  p33 is unchanged. At p512 and p513, default switches only the Kcur
  `k=3584, rows=512` row to ASMWAIT; Qcur/FFN stay on the existing packed-Q8
  routes. Rollback env
  `GGML_HRX_DISABLE_Q5_K_WMMA16_VK128_MOTIF192_ASMWAIT_SMALLPROJ_PROMPT=1`
  returns Kcur to the prior motif192 route.
- timing:
  selected-row Kcur improves from `780.981 us` to `742.777 us` at p512
  (`1.051x`) and from `728.029 us` to `693.861 us` at p513 (`1.049x`).
  Non-selected Qcur/FFN row movement is same-route noise and is not used as
  promotion evidence.
- decision:
  accept ASMWAIT as a narrow gfx1151 default for the Qwen2.5 Q5
  small-projection row only: `k=3584`, `rows=512`, `cols>=512`. Do not broaden
  Q5 routing from this result. Continue Vulkan-oracle work on the larger
  packed-Q8 prompt rows or a lower-level direct-WMMA schedule that closes the
  remaining RADV/HIP wait, region, barrier, VGPR, and writeback-cadence deltas.

## 2026-06-21 - Current basket refresh after Q5 ASMWAIT

- source:
  `sources/llama.cpp` at
  `4f30f87cd hrx: promote q5 motif192 asmwait smallproj`.
- artifact:
  `cache/hrxv1/gfx1151/basket-current-4f30f87cd-r1/`.
- command:
  `python3 tools/hrxv1_basket_benchmark.py --tag basket-current-4f30f87cd-r1 --cases p33,p512,p513 --backends hrx,vulkan --repetitions 1 --flash-attn 1 --timeout 1200`.
- audit:
  all rows exited `0`; backend labels were clean (`HRX` for HRX rows,
  `Vulkan` for Vulkan rows); HRX fallback lines were `0`.
- KPI:
  average and steady geomean were both `0.603x` Vulkan over `24` downloaded
  rows, with `23/24` rows below parity. This supersedes the older
  parity-looking checkpoint for current KPI decisions.
- worst rows:
  Qwen3 30B Q6_K p33 is the clear next boulder at `49.189` tok/s HRX versus
  `169.685` tok/s Vulkan (`0.290x`), selecting
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32`.
  The next rows are Q8_0 p512 (`0.494x`), Q8_0 p513 (`0.518x`), Q5_K_M p513
  (`0.551x`), and Q6_K p512 (`0.552x`).
- Q6 p33 constraints:
  H4LOAD is still only a small accepted floor, with `8` WMMA, `20` LDS reads,
  and `16` global stores, far from the RADV medium oracle's `16` WMMA, `48`
  `ds_load_b64`, `96` `buffer_store_b32`, `64` `ds_store_b16`, and `64`
  `ds_load_u16_d16`. The obvious route-facing pivots have already been
  rejected: padladder-expwait and RADV96 duplicate-output passed focused
  correctness but regressed same-runner timing, and padladder faststage /
  mixedstage bench timing lost to accepted VK64 by roughly `1.30x-1.34x`.
- decision:
  make Q6 p33 the next production boulder, but do not replay selector-only,
  prefetch, bufferstore, expwait, duplicate-output, faststage, or mixedstage
  axes. The next source change needs a new lower-cost selected-output
  ownership/store primitive or a different dataflow with focused p33
  correctness, route traces, static evidence, and timing against H4LOAD.

## 2026-06-21 - Current Q6 p33 focused H4LOAD baseline

- source:
  `sources/llama.cpp` at `b94b8052d hrx: record gfx1151 current basket gap`.
- artifact:
  `cache/hrxv1/gfx1151/q6-p33-current-b94b8052d-baseline-20260621-0340/`.
- purpose:
  refresh the focused backend-op floor for the current worst basket row before
  attempting another Q6 p33 route. The older e5f41 baseline is useful history,
  but future source candidates need a same-commit comparison point.
- correctness:
  `test-backend-ops test -b HRX0 -o MUL_MAT` passed `10/10` rows from
  `cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/ops.txt`.
- route evidence:
  Q6 `cols=33` rows select
  `hrx_mul_mat_vec_q6_k_wmma16x16_vk64_padded44_w64_h4load_f16acc_wg256_f32`.
  Q6 `cols=1` rows stay on
  `hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32`.
- timing:
  total focused perf is `7938.780 us`. Selected Q6 p33 prompt rows are:
  `Vcur-0 182.914 us`, `node_28 399.617 us`, `Qcur-0 243.114 us`, and
  `result_output 5935.772 us`.
- decision:
  this is the current promotion floor. A replacement Q6 p33 route must pass
  the same rows, keep the decode-shaped `cols=1` split, beat H4LOAD on
  `result_output` and total focused time, and include static RADV/HIP schedule
  evidence before model-level A/B.
