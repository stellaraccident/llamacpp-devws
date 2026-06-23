# HRX2 Q4_K Prompt Schedule Ledger

Date: 2026-06-16.

Purpose: ground the next Q4_K Phase 2a optimization in schedule evidence
instead of local schedule guessing. Current reduced baseline after Q5/Q6 HIP
bridges:
`cache/hrx2/phase2a/q5q6-hip-bridge-reduced-20260616-091135/`.

## Current Accepted HRX2 Route

- Route:
  `mul_mat_q4_k_q8_1_x4_hip_vkm64x64_pack2_gfx1100_k256_32768_r64_32768_c64_512_wg128`.
- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q4_k_q8_1_wave64.hip.cpp`,
  export `hrx2_mul_mat_vec_q4_k_q8_1_x4_vkm64x64_pack2_wg128_u32`.
- Shape regime: prompt Q4_K x packed Q8_1/x4, `k % 256 == 0`,
  `rows % 64 == 0`, `cols % 64 == 0`, p64 and p512 prompt rows.
- Tile/workgroup: BM64 x BN64, WG128, logical wave64, two waves per workgroup.
  This follows the Vulkan medium K-quant integer-MMQ tuple
  `BLOCK_SIZE=128/BM64/BN64/WM64/WN32/WMITER1/TM2/TN2/WARP64`.
- Lane ownership: each lane owns two rows by two columns within each WNITER
  slice; eight WNITER slices cover each wave's WN32 tile, and the two waves
  cover BN64.
- K loop: one Q8 block (`32` values) per barrier; `BK_STEP=1`.
- A staging: Q4_K A payload, scale, and min are staged in LDS. The accepted
  pack2 cache stores four `i32` words per row/group, each combining two Q4
  pack4 payloads as `q0 | (q1 << 4)`, matching the Vulkan Q4_K MMQ prior.
- B staging: packed Q8_1/x4 RHS stores 64 columns x eight `i32` payload words
  plus f32 d/s metadata.
- Dot form: `__builtin_amdgcn_sudot4(false, qpack, true, rpack, 0, false)`.
- Rollback: `GGML_HRX2_DISABLE_Q4_HIP_VKM64X64_PROMPT=1`.
- Focused acceptance evidence:
  `cache/hrx2/phase2a/q4-vkm64x64-default-opgate-20260616-124657/`.
  The route passed p64 and p512 Q4_K backend-op correctness, selected by
  default without provider failures, and improved focused prompt Q4 rows by
  roughly 1.02x-1.25x versus the previous pack2 bridge.
- Model acceptance evidence:
  `cache/hrx2/phase2a/q4-vkm64x64-hrx2-smoke-20260616-124520/` and
  `cache/hrx2/phase2a/q4-vkm64x64-default-reduced-20260616-124844/`.
  The reduced Q4 slice reached about 0.53x-0.61x Vulkan on p512 and
  0.45x-0.50x Vulkan on p64 with zero CPU compute fallback.

## Previous HRX2 Loom Route

- Route:
  `mul_mat_q4_k_q8_1_x4_mmq64x32_k256_32768_r1_32768_c32_512_wg256`.
- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`,
  export `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static`.
- Shape regime: prompt Q4_K x packed Q8_1/x4, `k % 256 == 0`,
  `cols % 32 == 0`, current hot rows at p512 include rows 3072, 8192,
  16384 and cols 512.
- Tile/workgroup: BM64 x BN32, WG256, wave32, four waves per workgroup.
- Lane ownership: `row_lane = lane % 64`, `col_lane = lane / 64`; each lane
  owns one output row and eight columns.
- K loop: one Q8 block (`32` values) per barrier; `BK_STEP=1`.
- A staging: Q4_K A payload, scale, and min are staged in workgroup memory.
  Current A payload stores eight `i32` words per staged row (`512xi32` for
  64 rows), one word per Q4 4-value group.
- B staging: packed Q8_1/x4 RHS stores 32 columns x 8 `i32` payload words
  plus f16 d/s metadata.
- Dot form: `vector.dot4i<u8s8>`, lowered to `v_dot4_i32_iu8`.
- Writeback: each lane writes eight `f32` output elements.
- Resource facts from focused provider evidence:
  `cache/hrx2/phase2a/q5q6-hip-bridge-opgate-20260616-091045/.../mul_mat_q4...`.
  Compile report: no spills, `local_memory_bytes=3712`,
  `register_pressure_peak_live_units=64`, `instruction_count=620`,
  `dot_count=64`. HSACO metadata: `vgpr_count=137`, `sgpr_count=21`,
  `group_segment_fixed_size=3712`, `wavefront_size=32`.
- Focused perf after Q5/Q6 bridges:
  `cache/hrx2/phase2a/q5q6-hip-bridge-opgate-20260616-091045/`.
  Representative Q4 rows are still about `1.14 ms`, `3.39 ms`,
  `3.23 ms`, and `6.30 ms`.

## Vulkan Prior

- Route family: Vulkan `mul_mmq.comp` generated Q4_K integer MMQ.
- Source:
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`,
  `mul_mmq_funcs.glsl`, `mul_mmq_shmem_types.glsl`,
  `dequant_funcs_cm2.glsl`, `types.glsl`.
- Dispatch/tile constants:
  `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`.
  For non-coopmat paths, K-quant MMQ priors include large and medium
  warptiles such as `{256,128,256,64,1}` and `{256,128,128,64,1}` with
  `BK_STEP=4` for normal `MUL_MAT`. Integer K-quant coopmat-era priors use
  `WMITER=1` to manage register pressure.
- Q4_K A cache:
  `mul_mmq_shmem_types.glsl` defines `QUANT_R_MMQ=2` and
  `block_a_cache { uint32_t qs[4]; FLOAT_TYPE_VEC2 dm; }`.
- Q4_K A staging:
  `mul_mmq_funcs.glsl:block_a_to_shmem` loads two 32-bit Q4 payload words and
  packs them into one `uint32_t` with low and high nibbles:
  `vals0 | (vals1 << 4)`. This stores four A payload words per row, not eight.
- Dot form:
  `mmq_dot_product` extracts low/high nibbles from each packed A word and calls
  `dotPacked4x8EXT(qs_a, cache_b.qs[iqs])` for eight Q4/Q8 groups.
- B staging:
  `block_b_cache { int32_t qs[8]; FLOAT_TYPE_VEC2 ds; }`, matching the Q8_1
  packed payload structure HRX2 already uses.
- Performance evidence from same-machine Vulkan perf logs in
  `q5q6-hip-bridge-reduced-20260616-091135/`: Q4_K p512 buckets are roughly
  `264-286 us` for rows3072/k3072, `601 us` for rows3072/k8192, `616 us` for
  rows8192/k3072, and `1232 us` for rows16384/k3072.

## HRX1 Prior

- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp`.
- Shape regime: grouped MoE Q4_K x packed Q8_1/x4, not a direct dense
  `MUL_MAT` drop-in.
- Tile/workgroup:
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq_wg64_impl<BN,TN>` uses BM64,
  BN16/32, WG64, wave64, `TM=4`, `TN=1/2`, `BK_STEP=1`.
- Lane ownership:
  one wave owns the tile; each lane owns four rows x one or two route columns.
- A staging:
  helper `hrx_q4_k_moe_mmq_fetch_a` and `commit_a` stage one Q4 payload word
  per `iqs` plus f32 d/min metadata into LDS.
- B staging:
  stages packed Q8_1/x4 payloads and f32 d/s metadata, with route indirection.
- Dot form:
  `__builtin_amdgcn_sudot4(false, qpack, true, rpack, 0, false)`.
- Relevance:
  proves a wave64 single-wave Q4_K route-tiled schedule can work, and provides
  a HIP bridge template, but it is less directly applicable than Vulkan's
  standard dense `MUL_MAT` path because it bakes in counts/routes and MoE
  output indexing.

## Analytical Alternatives

| ID | Prior Basis | Change | Expected Signal | Risk |
| --- | --- | --- | --- | --- |
| `q4-a-pack2` | Vulkan Q4_K MMQ | Keep the accepted HRX1-derived BM64/BN32 bridge, but stage A payload as four packed words per row using the Vulkan `vals0 | vals1 << 4` layout. Compute extracts low/high nibbles per `iqs`. | Accepted. Lowered LDS/A payload traffic and improved focused Q4 rows plus model smoke. | Keep rollback and revisit only if another schedule supersedes it. |
| `q4-bkstep4-pack2` | Vulkan normal `MUL_MAT` MMQ | After `q4-a-pack2` passed, stage four K blocks per barrier with matching A/B scratch layout. | Rejected. Correctness passed and route selected, but p64 rows regressed about 5.8x-11.8x versus accepted pack2. Artifact: `cache/hrx2/phase2a/q4-pack2-bkstep4-opgate-20260616-122640/`. | Do not retry `BK_STEP=4` inside the same BM64/BN32 single-wave pack2 topology. |
| `q4-vkm64x64-pack2` | Vulkan medium K-quant integer MMQ | Preserve pack2 Q4 A-cache and packed Q8_1/x4 RHS, but move from HRX1-derived `WG64/BM64/BN32/TM4/TN2` to Vulkan-medium `BLOCK_SIZE=128/BM64/BN64/WM64/WN32/WMITER1/TM2/TN2/WARP64`. | Accepted. Focused Q4 prompt rows improved 1.02x-1.25x, all six repeated model rows improved 1.10x-1.23x, and p512 now exceeds the Phase 2a target. Artifacts: `q4-vkm64x64-default-opgate-20260616-124657`, `q4-vkm64x64-default-reduced-20260616-124844`. | p64 is still only 0.45x-0.50x Vulkan, so do not keep pushing this exact axis; next work needs Q8_1 reuse/quantize or a larger schedule/fusion change. |
| `q4-vulkan-large-tile` | Vulkan large K-quant MMQ | Move toward BM128/BN128 or BM128/BN64-style ownership with WMITER=1 and explicit per-lane multi-row/multi-col outputs. | Needed if BM64/BN32 topology is the main ceiling. | Large Loom rewrite; requires full prior matrix and compile-report/ISA comparison. |
| `q4-hrx1-wave64-bridge` | HRX1 grouped MoE Q4_K | Build a llama.cpp-local dense `MUL_MAT` HIP bridge from the grouped core, replacing route indirection with direct columns. | Refutes whether the HRX1 single-wave schedule family beats current Loom. | Not directly comparable to Vulkan dense MMQ; should be diagnostic unless it clearly wins focused rows. |

Current decision: `q4-vkm64x64-pack2` is accepted as the production Q4_K prompt
bridge, with rollback `GGML_HRX2_DISABLE_Q4_HIP_VKM64X64_PROMPT=1`. The
`q4-bkstep4-pack2` adjacent pivot was rejected after a focused backend-op
sweep. The next Q4 work should treat VKM64x64 as the baseline and bracket
around documented axes only: graph-level Q8_1 RHS reuse, quantize reuse, or a
larger Vulkan-style tile. p64 remains the weaker regime, so any p64-specific
pivot should first prove itself in backend-op sweeps before full model
integration.
