# HRX v1 gfx1151 Q4_K Vulkan Oracle Schedule Ledger

Date: 2026-06-17

## Scope

This ledger compares the first same-machine Vulkan oracle capture for dense
Q4_K prompt matmul against the current HRX v1 HIP C++ Q4_K x Q8_1 x4 prompt
candidate. It is the schedule anchor for the next HIP C++ dense prompt kernel
work.

This is not route-promotion evidence by itself. It is a prior-art schedule
comparison that must be referenced before adding or promoting another Q4_K
dense prompt candidate on `gfx1151`.

## Artifacts

- Vulkan oracle:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/`
- Vulkan inventory:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/inventory/kernel_inventory.md`
- Vulkan SPIR-V:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/spv/matmul_q4_k_f32_f16acc_aligned_l__main__0x5666175250529efb.spv`
- Vulkan SPIR-V asm:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/spvasm/matmul_q4_k_f32_f16acc_aligned_l__main__0x5666175250529efb.spvasm`
- Vulkan RADV ISA/stats:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/radv/isa/matmul_q4_k_f32_f16acc_aligned_l__main__5666175250529efb.amdgcn.txt`
  and
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/radv/stats/matmul_q4_k_f32_f16acc_aligned_l__main__5666175250529efb.stats.txt`
- Current HRX route source:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp`
- Current HRX HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q4_k_q8_1_x4_mmql128.hsaco`
- Current HRX focused gate:
  `cache/hrxv1/gfx1151/focused-q4-mmql128-p512-20260617-164724/`
- Current HRX model A/B:
  `cache/hrxv1/gfx1151/q4-mmql128-model-ab-20260617-165235/`

## Vulkan Pipeline Facts

- Pipeline: `matmul_q4_k_f32_f16acc_aligned_l`
- Hash: `0x5666175250529efb`
- Model row: Llama 3.1 8B Q4_K_M, `p512/n0/fa1`, Vulkan0
- Runtime: `371.806637 tok/s`, one no-warmup oracle run
- Dispatch count in graph: `190`
- Specialization tuple:
  `[256, 128, 128, 32, 64, 64, 2, 16, 16, 16, 64]`
- Workgroup denominators: `[128, 128, 1]`
- Workgroup size from RADV: `256 x 1 x 1`
- Full-subgroup requirement: true
- RADV resource facts: `SGPR=108`, `VGPR=192`, `LDS=22528`,
  `spills=0`, `Subgroups per SIMD=8`, `Instructions=3867`,
  `VALU=1757`, `SALU=702`, `VMEM=220`, `SMEM=102`
- Shader source family: `ggml-vulkan/vulkan-shaders/mul_mm.comp` with
  `DATA_A_Q4_K=1`, `LOAD_VEC_A=4`, `LOAD_VEC_B=8`,
  `B_TYPE=mat2x4`, `D_TYPE=float`, `ALIGNED=1`, `COOPMAT=1`,
  and `f16acc=1`.
- Specialization resolves `BK=32`, `BM=128`, `BN=128`, `WM=64`,
  `WN=64`, `WMITER=2`, `TM=16`, `TN=16`, `TK=16`, and `WARP=64`.
- LDS decomposition from the shader source and RADV stats:
  `SHMEM_STRIDE = BK / 2 + 4 = 20`; Q4 A tile is
  `BM * SHMEM_STRIDE * sizeof(f16vec2) = 10240` bytes; f32 RHS B tile is
  `BN * SHMEM_STRIDE * sizeof(f16vec2) = 10240` bytes after conversion to
  f16; `coopmat_stage = TM * TN * (BLOCK_SIZE / WARP) * sizeof(f16) = 2048`
  bytes; total `22528` bytes.
- SPIR-V uses `OpCapability CooperativeMatrixKHR` and the RADV ISA emits
  `v_wmma_f16_16x16x16_f16`. The hot loop has two shared-memory barriers and
  32 static `v_wmma` instructions interleaved with LDS loads.
- A specialized SPIR-V disassembly was materialized under
  `.tmp/hrxv1/q4-oracle/matmul_q4_k_f32_f16acc_aligned_l.specialized.spvasm`
  with the captured spec tuple. It resolves `TM=16`, `TN=16`, `TK=16`,
  `WARP=64`, `cms_per_row=4`, `cms_per_col=4`, `loadstride_a=32`,
  `loadstride_b=64`, and `storestride=4`. The source-level coopmat loop is
  therefore `BK/TK=2` K slices times `4x4` accumulator matrices per subgroup,
  explaining the RADV ISA's 32 static WMMAs for each subgroup-64 owner.
- The captured coopmat1 SPIR-V does not contain a separate shared scale cache
  such as `shAscales`; that pattern belongs to the coopmat2 source family and
  should not be used as evidence for this Q4_K pipeline without a matching
  capture.

Normalized p512 shape buckets:

| Dispatches | Src0 | Src1 | Dst | Workgroups |
| ---: | --- | --- | --- | --- |
| 62 | q4_K `[4096,14336]` | f32 `[4096,512]` | f32 `[14336,512]` | `[112,4,1]` |
| 48 | q4_K `[4096,1024]` | f32 `[4096,512]` | f32 `[1024,512]` | `[8,4,1]` |
| 64 | q4_K `[4096,4096]` or q4_K `[14336,4096]` | f32 `[4096,512]` or f32 `[14336,512]` | f32 `[4096,512]` | `[32,4,1]` |

## Odd And Tail Captures

Additional Vulkan oracle captures for the same Llama 3.1 8B Q4_K_M family:

- p33:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p33-fa1-20260617-200738/`
- p513:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p513-fa1-20260617-200751/`

The p33 row does not use the large 128x128 aligned pipeline. Vulkan switches to
the medium aligned route:

- Pipeline: `matmul_q4_k_f32_f16acc_aligned_m`
- Spec: `[128,64,64,32,64,32,2,16,16,16,64]`
- Workgroup denominators: `[64,64,1]`
- Representative normalized bucket: q4_K `[4096,14336]` x f32 `[4096,33]`
  -> f32 `[14336,33]`, workgroups `[224,1,1]`
- RADV resource facts: `SGPR=108`, `VGPR=144`, `LDS=11264`, no spills,
  `Subgroups per SIMD=10`, `Instructions=2224`

The p513 row returns to the large aligned route and handles the tail in the
fifth workgroup column:

- Pipeline: `matmul_q4_k_f32_f16acc_aligned_l`
- Spec: `[256,128,128,32,64,64,2,16,16,16,64]`
- Representative normalized bucket: q4_K `[4096,14336]` x f32 `[4096,513]`
  -> f32 `[14336,513]`, workgroups `[112,5,1]`
- RADV resource facts match p512: `SGPR=108`, `VGPR=192`, `LDS=22528`, no
  spills

This confirms the HRX route policy must keep narrow prompt and production-width
tail behavior separate. A p512/p513 large-tile win should not automatically
replace the p33 route, and p33 evidence should not reject the large route for
production-width tails.

## Current HRX Route Facts

- Route: `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_wg256_f32`
- Gate: `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`
- Shape guard: Q4_K `MUL_MAT` prompt, `k % 256 == 0`,
  `rows % 128 == 0`, `cols >= 128`, contiguous tensors, packed Q8_1 x4 RHS
- Workgroup size: `256 x 1 x 1`
- Source constants: `BM=128`, `BN=128`, `BK_STEP=1`, `WARP=64`,
  `WM=64`, `WN=64`, `WMITER=1`, `TM=4`, `TN=2`, `WNITER=8`
- Staging: shared Q4 A cache plus Q8_1 x4 B cache for one `BK_STEP`
- HIP dot primitive: `__builtin_amdgcn_sudot4`
- HSACO metadata: `SGPR=53`, `VGPR=139`, `LDS=8192`,
  `spills=0`, `wavefront_size=64`, `private_segment_fixed_size=0`
- Focused p512 gate passed CPU-reference rows and selected the route.
- Focused p512 timings in
  `focused-q4-mmql128-p512-20260617-164724/perf.csv` include
  `Qcur` at `696.8 us` for q4_K `[2048,4096] x f32 [2048,512]` and
  a q4_K `[4096,2048] x f32 [4096,512]` row at `732.8 us`.
- Same-run HRX model A/B on Qwen3 30B Q4_K_XL improved production-width
  p512/p513 but p33 stayed better on the narrower Q4 route.

## Schedule Comparison

| Axis | Vulkan oracle | Current HRX Q4_K x Q8_1 x4 |
| --- | --- | --- |
| Tile shape | 128 rows x 128 cols | 128 rows x 128 cols |
| Workgroup | 256 threads | 256 threads |
| Wave size | Full subgroup, RADV API subgroup 64 | `wavefront_size=64` |
| Per-subgroup output tile | 64 x 64 using 16 cooperative-matrix accumulators (`TM=16`, `TN=16`, `cms_per_row=4`, `cms_per_col=4`) | 64 x 64 logical wave tile with scalar integer-dot micro-tiles (`TM=4`, `TN=2`, `WNITER=8`) |
| K step/staging | `BK=32`, padded f16vec2 A/B LDS tiles, 22528 bytes including `coopmat_stage` | `BK_STEP=1`, 8192 bytes group segment |
| A dataflow | Q4_K is dequantized directly into f16vec2 LDS with `LOAD_VEC_A=4`; scale/min reconstruction happens in the shader load path | Shared Q4 A cache, one K-quant block step at a time |
| B dataflow | Consumes graph f32 RHS directly, vector-loads `LOAD_VEC_B=8`, converts to f16vec2 LDS for cooperative matrix load | Separate HRX Q8_1 x4 quantize route then shared packed B cache |
| Dot form | RADV ISA emits `v_wmma_f16_16x16x16_f16` in the dominant Q4_K pipeline | `v_dot4_i32_iu8` from `__builtin_amdgcn_sudot4` |
| Resource pressure | 108 SGPR, 192 VGPR, 22528 LDS, no spills | 53 SGPR, 139 VGPR, 8192 LDS, no spills |
| Current evidence | Vulkan p512/fa1 oracle row is a same-machine schedule prior | HRX route is accepted only as an opt-in production-width candidate |

The main delta is not the headline 128x128 workgroup tile. The current HRX
MMQL128 route already matches the large-tile prompt regime but stays in the
integer-dot, explicit Q8_1 x4 RHS family. The Vulkan route instead uses the
generic large aligned coopmat1 family: direct f32 RHS, Q4_K dequant into f16
shared memory, `BK=32`, `LOAD_VEC_A=4`, `LOAD_VEC_B=8`, padded A/B LDS
strides, and cooperative-matrix f16 accumulation. The next useful HIP work is
therefore a Vulkan-large coopmat clone or a bounded dataflow probe against that
clone, not another local reshuffle of the current `sudot4` schedule.

## Direct WMMA Diagnostic

A direct staged-WG256 Q4_K WMMA diagnostic was added as
`hrx_mul_mat_vec_q4_k_wmma16x16_f16acc_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-wmma16-wg256-focused-20260617-223548/`

Facts:

- compiled through CMake/Ninja;
- emitted `v_wmma_f16_16x16x16_f16`;
- HSACO metadata: wave32, SGPR `27`, VGPR `53`, LDS `3072`, no spills;
- passed CPU-reference focused DeepSeek Q4_K rows for p512 plus exact p33 and
  p513 odd/tail rows;
- selected the intended provider in route traces.

The performance result rejects this direct-dequant WMMA spelling as a
production route. Same-runner p512 focused rows regressed versus the accepted
packed Q8_1 MMQL128 path:

| Row | Current MMQL128 | Direct Q4 WMMA |
| --- | ---: | ---: |
| Kcur | `986.46 us` | `2402.45 us` |
| Qcur | `4123.76 us` | `14151.28 us` |
| ffn_out | `13126.86 us` | `55425.92 us` |
| ffn_gate | `13244.67 us` | `58826.01 us` |

Conclusion: the opcode mismatch was real, but opcode parity alone is not the
missing performance axis. The next Q4_K candidate must move toward Vulkan's
packed/staged data reuse, LDS depth, load vectorization, and barrier schedule,
not another direct global-load dequant-to-WMMA route.

Static ISA comparison against the captured Vulkan prior:

| Kernel | WMMA | integer dot | barriers | LDS ops | global loads | global stores | waits | LDS bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vulkan large coopmat Q4_K | 32 | 0 | 2 | 324 | not normalized | not normalized | 182 | 22528 |
| HRX MMQL128 | 0 | 512 | 2 | 58 | 18 | 64 | 185 | 8192 |
| HRX MMQL128 BK2 diagnostic | 0 | 1024 | 2 | 116 | 36 | 64 | 247 | 16384 |
| HRX direct WMMA diagnostic | 1 | 0 | 2 | 12 | 8 | 8 | 12 | 3072 |
| HRX VK128 direct WMMA diagnostic | 16 | 0 | 2 | 98 | 26 | 64 | 59 | 16384 |
| HRX VK128 padded direct WMMA diagnostic | 16 | 0 | 2 | 92 | 8 | 64 | 35 | 20480 |
| HRX VK128 padded prefetch WMMA diagnostic | 16 | 0 | 2 | 20 | 8 | 64 | 38 | 20480 |
| HRX VK128 padded pair64 WMMA diagnostic | 16 | 0 | 2 | 14 | 8 | 64 | 28 | 20480 |
| HRX VK128 padded wave64 WMMA diagnostic | 32 | 0 | 2 | 34 | 8 | 64 | 155 | 20480 |
| HRX VK128 padded wave64 half4 diagnostic | 32 | 0 | 2 | 34 | 8 | 64 | 155 | 20480 |
| HRX VK128 padded wave64 B64ASM diagnostic | 32 | 0 | 2 | 258 | 8 | 64 | 396 | 20480 |
| HRX MMQL128x64 packed diagnostic | 0 | 256 | 2 | 38 | 18 | 32 | 119 | 5632 |

The BK2 diagnostic moved only the K-depth axis within the integer-dot family
and regressed. The direct WMMA diagnostic moved only the opcode axis without
the Vulkan tile/data-reuse structure and regressed. The VK128 direct WMMA
diagnostic moved closer to the large Vulkan tile/dataflow but still regressed.
The remaining untested class is not the high-level BM128/BN128/BK32/direct-RHS
shape alone; it is the lower-level RADV cooperative-matrix lowering details,
including exact LDS padding/stage layout, vectorized global/LDS scheduling,
wait placement, and whether explicit Q8_1 packing reuse is still required in
HIP despite Vulkan consuming graph f32 RHS.

## VK128 Direct WMMA Diagnostic

A wider direct staged-WG256 Q4_K WMMA diagnostic was added as
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_f16acc_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-wmma16-vk128-focused-20260617-225628/`

Facts:

- compiled through CMake/Ninja;
- cloned the resolved Vulkan-large class at the high level:
  `BM=128`, `BN=128`, `BK=32`, `WG=256`, direct f32 RHS, Q4_K dequant to f16
  LDS, f16acc WMMA;
- emitted 16 static `v_wmma_f16_16x16x16_f16` sites;
- HSACO metadata: wave32, SGPR `38`, VGPR `115`, LDS `16384`, no spills;
- passed CPU-reference focused DeepSeek Q4_K p512 rows and exact p513 tail
  rows while selecting the intended provider;
- exact p33 rows passed while staying on the existing narrow Q4_K route.

The performance result rejects this VK128 direct-f32-RHS WMMA spelling as a
production route:

| Row | Current MMQL128 | VK128 Direct WMMA |
| --- | ---: | ---: |
| Kcur | `979.20 us` | `1565.46 us` |
| Qcur | `3948.47 us` | `6674.58 us` |
| ffn_out | `14573.74 us` | `19495.02 us` |
| ffn_gate | `13196.74 us` | `23377.37 us` |

Conclusion: a high-level Vulkan-large HIP WMMA clone is not enough. The next
Q4_K work should either reproduce RADV's lower-level cooperative-matrix
load/store/LDS schedule more exactly or pivot back to the current packed
Q8_1/x4 dataflow and use the Vulkan evidence only for load scheduling and
resource-shape targets.

## VK128 Padded Direct WMMA Diagnostic

A single-axis LDS-padding follow-up was added as
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_f16acc_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-wmma16-vk128-padded-focused-20260617-230429/`

Facts:

- compiled through CMake/Ninja;
- preserved the VK128 direct-F32 WMMA shape but changed the A/B shared row
  stride from 32 halfs to the Vulkan f16vec2-equivalent 40-half padded stride;
- emitted 16 static `v_wmma_f16_16x16x16_f16` sites;
- HSACO metadata: wave32, SGPR `25`, VGPR `121`, LDS `20480`, no spills;
- passed CPU-reference focused DeepSeek Q4_K p512 rows and exact p513 tail
  rows while selecting the intended provider;
- exact p33 rows passed while staying on the existing narrow Q4_K route.

The performance result rejects padding alone as a production route:

| Row | Current MMQL128 | VK128 Padded Direct WMMA |
| --- | ---: | ---: |
| Kcur | `987.01 us` | `1543.83 us` |
| Qcur | `4107.51 us` | `7244.42 us` |
| ffn_out | `15515.40 us` | `24740.20 us` |
| ffn_gate | `12186.69 us` | `25907.86 us` |

Conclusion: Vulkan's padded LDS stride is not the isolated missing axis. The
direct-F32 HIP WMMA family now has three rejected points: opcode-only 64x32,
VK128 unpadded, and VK128 padded. Future Q4_K work should either reproduce
RADV's exact lower-level cooperative-matrix instruction/data movement schedule
with substantially stronger evidence, or move back to the current packed
Q8_1/x4 dataflow and tune load ordering, reuse, or output ownership there.

## VK128 Padded Prefetch Direct WMMA Diagnostic

A RADV-schedule prefetch follow-up was added as
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_prefetch_f16acc_wg256_f32`,
guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_PREFETCH_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-wmma16-vk128-padded-prefetch-focused-20260617-232901/`

Facts:

- compiled through CMake/Ninja;
- preserved the rejected VK128 padded direct-F32 WMMA shape while explicitly
  loading both k-half A/B fragments before issuing WMMA, to move toward the
  Vulkan large coopmat LDS-load-before-WMMA cadence;
- emitted 16 static `v_wmma_f16_16x16x16_f16` sites, 18 `ds_loads`, 2
  `ds_store`s, 8 global loads, 64 global stores, and 38 `s_waitcnt`s;
- HSACO metadata: wave32, SGPR `25`, VGPR `133`, LDS `20480`, no spills;
- a direct wave64 variant could not compile because
  `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` requires
  `wavefrontsize32`;
- passed CPU-reference focused DeepSeek Q4_K p512 rows and exact p513 tail
  rows while selecting the intended provider;
- exact p33 rows passed while staying on the existing narrow Q4_K route.

The performance result rejects source-level prefetching as a production route:

| Row | Current MMQL128 | VK128 Padded Prefetch WMMA |
| --- | ---: | ---: |
| Kcur | `990.28 us` | `1508.75 us` |
| Qcur | `4038.39 us` | `7253.08 us` |
| ffn_out | `15180.79 us` | `25845.81 us` |
| ffn_gate | `13737.11 us` | `25565.23 us` |

Conclusion: simple HIP C++ fragment prefetching changed issue order and raised
VGPR pressure, but it still did not reproduce RADV's 32-static-WMMA,
192-ds-load, 132-ds-store cooperative-matrix lowering. The next exact-schedule
step is to mechanically inspect RADV's SPIR-V cooperative-matrix lane/subgroup
ownership and decide whether HIP needs a lower-level inline-assembly path or a
return to the packed Q8_1/x4 route with schedule evidence. Do not add more
broad direct-WMMA probes until this lane ownership gap is explained.

## VK128 Padded Pair64 Direct WMMA Diagnostic

The specialized SPIR-V evidence shows Vulkan maps each subgroup-64 owner to a
64x64 output tile: `cms_per_row=4`, `cms_per_col=4`, and `BK/TK=2`, for 32
WMMA operations per subgroup. The earlier HIP VK128 routes instead used eight
wave32 owners per workgroup, each owning a 16x128 strip. A pair64 ownership
probe was added as
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_pair64_f16acc_wg256_f32`,
guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_PAIR64_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-wmma16-vk128-padded-pair64-focused-20260617-234150/`

Facts:

- compiled through CMake/Ninja;
- preserved the rejected VK128 padded direct-F32 WMMA shape and Vulkan's
  40-half padded LDS stride;
- remapped pairs of wave32 waves to one Vulkan-style 64x64 output owner while
  staying within the HIP WMMA builtin's wave32 compile requirement;
- emitted 16 static `v_wmma_f16_16x16x16_f16` sites, 12 `ds_loads`, 2
  `ds_store`s, 8 global loads, 64 global stores, and 28 `s_waitcnt`s;
- HSACO metadata: wave32, SGPR `25`, VGPR `134`, LDS `20480`, no spills;
- passed CPU-reference focused DeepSeek Q4_K p512 rows and exact p513 tail
  rows while selecting the intended provider;
- exact p33 rows passed while staying on the existing narrow Q4_K route.

The performance result rejects wave-pair output ownership as a production
route:

| Row | Current MMQL128 | VK128 Padded Pair64 WMMA |
| --- | ---: | ---: |
| Kcur | `979.99 us` | `1449.28 us` |
| Qcur | `3955.07 us` | `7255.33 us` |
| ffn_out | `15772.01 us` | `24397.54 us` |
| ffn_gate | `12102.39 us` | `25685.38 us` |

Conclusion: output ownership alone is not enough. The HIP route still lowers
as independent wave32 WMMA owners with far shallower LDS traffic than RADV's
subgroup-64 cooperative-matrix lowering. The useful direct-WMMA path now needs
a lower-level way to express or emulate subgroup-64 cooperative-matrix
fragments; otherwise Q4_K work should return to the packed Q8_1/x4 path and
use the Vulkan oracle only to drive load/wait/resource-shape changes.

## VK128 Padded Wave64 Direct WMMA Diagnostic

The earlier "wave64" compile failure was caused by compiling the wave32 WMMA
builtin in wave64 mode. ROCm exposes the actual RDNA3 wave64 builtin:
`__builtin_amdgcn_wmma_f16_16x16x16_f16_w64`, with A/B fragment type `V16x`
and accumulator type `V8x`. The AMD matrix-instruction-calculator confirms the
RDNA3 wave64 mapping:

- A/B lanes use `lane % 16`, replicated across lanes `+16`, `+32`, and `+48`;
- A/B register number maps to `k = 2 * reg + half`;
- D stores use `i = 4 * reg + floor(lane / 16)` and `j = lane % 16`;
- only one accumulator half is selected by OPSEL.

Two wave64 routes were added:

- `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`;
- `hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_hi_f16acc_wg256_f32`.

Artifact:
`cache/hrxv1/gfx1151/q4-wmma16-vk128-padded-w64-lane64-full-20260618-000016/`

Facts:

- compiled through CMake/Ninja with `WAVEFRONT_SIZE=64`;
- used the real ROCm wave64 WMMA builtin, not the wave32 builtin in wave64
  mode;
- preserved the VK128 padded direct-F32 shape and Vulkan's 40-half padded LDS
  stride;
- emitted 32 static `v_wmma_f16_16x16x16_f16` sites, matching the RADV static
  WMMA count;
- HSACO metadata: wave64, SGPR `29`, VGPR `165`, LDS `20480`, no spills;
- static ISA count: 32 `v_wmma`, 32 `ds_loads`, 2 `ds_store`s, 8 global
  loads, 64 global stores, and 155 `s_waitcnt`s;
- low-half and high-half OPSEL routes both selected correctly.

The first implementation accidentally used a wave32 lane id (`tid & 31`) in
wave64 mode. That failed all p512 rows with finite ERR around `3.0`. After
fixing the lane id to `tid & (WAVE - 1)`, both OPSEL routes passed p512
CPU-reference rows, the low route passed exact p513, and exact p33 stayed on
the existing narrow Q4_K route.

The performance result rejects the wave64 direct-F32 route as a production
route:

| Row | Current MMQL128 | w64 low OPSEL | w64 high OPSEL |
| --- | ---: | ---: | ---: |
| Kcur | `972.56 us` | `1535.23 us` | `1503.17 us` |
| Qcur | `3943.78 us` | `8354.39 us` | `8357.47 us` |
| ffn_out | `14110.24 us` | `26927.58 us` | `27073.09 us` |
| ffn_gate | `12483.27 us` | `27026.65 us` | `26952.06 us` |

Conclusion: the direct-WMMA path now proves wave64 WMMA correctness and the
32-static-WMMA structural target, but the direct-F32 dataflow is still much
slower than the packed Q8_1 MMQL128 route. The remaining delta is not the
headline WMMA/wave64 structure; it is RADV's deeper cooperative-matrix
load/store schedule and/or the cost of consuming graph F32 RHS directly instead
of reusing HRX's packed Q8_1 prepath.

## MMQL128x64 Packed Diagnostic

A packed-Q8_1/x4 output-ownership follow-up was added as
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x64_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128X64_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-mmql128x64-focused-20260617-231147/`

Facts:

- compiled through CMake/Ninja;
- preserved the Q4_K MMQL staged packed dataflow, `BM=128`, wave64, and
  `BK_STEP=1`;
- narrowed output ownership from `BN=128/WN=64/WNITER=8` to
  `BN=64/WN=32/WNITER=4`, halving the per-lane accumulator footprint;
- HSACO metadata: wave64, SGPR `53`, VGPR `109`, LDS `5632`, no spills;
- static ISA count: 256 integer-dot sites, 38 LDS ops, 32 stores, 119 waits;
- passed CPU-reference focused DeepSeek Q4_K p512 rows and exact p513 tail
  rows while selecting the intended provider;
- exact p33 rows passed while staying on the existing MMQ64 route.

The performance result rejects lower output ownership alone as a production
route:

| Row | Current MMQL128 | MMQL128x64 Packed |
| --- | ---: | ---: |
| Kcur | `989.40 us` | `1100.29 us` |
| Qcur | `4018.51 us` | `4817.41 us` |
| ffn_out | `14563.47 us` | `22394.71 us` |
| ffn_gate | `14017.86 us` | `20459.41 us` |

Conclusion: reducing accumulator pressure loses because it doubles column
workgroups and reduces packed-B reuse. The current Q4 packed path should keep
the 128-column production tile. The next packed-path probe should target load
ordering/wait behavior or B-cache ownership without reducing the output tile.

## MMQL128 Padcache Packed Diagnostic

A packed-Q8_1/x4 LDS-resource follow-up was added as
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_padcache_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PADCACHE_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-mmql128-padcache-focused-20260618-001039/`

Facts:

- compiled through CMake/Ninja;
- preserved the current Q4_K MMQL128 staged packed dataflow, `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, and output ownership;
- padded shared A-cache rows from 24 to 32 bytes and packed-B cache rows from
  40 to 64 bytes, raising LDS from `8192` to `12288` bytes;
- HSACO metadata: wave64, SGPR `52`, VGPR `145`, LDS `12288`, no spills;
- static ISA count stayed close to current MMQL128: 512 integer-dot sites, 46
  `ds_loads`, 12 `ds_stores`, 18 global loads, 64 global stores, 184 waits, and
  2 barriers;
- passed CPU-reference focused p512 and exact p513 rows while selecting the
  padcache provider;
- exact p33 rows passed while staying on the existing MMQ64 route.

The performance result rejects packed-cache row padding alone as a production
route:

| Row | Current MMQL128 | MMQL128 Padcache |
| --- | ---: | ---: |
| Kcur | `979.58 us` | `1326.86 us` |
| Qcur | `4108.71 us` | `5317.20 us` |
| ffn_out | `13816.55 us` | `17334.05 us` |
| ffn_gate | `12309.63 us` | `15366.25 us` |

Conclusion: simply spending more LDS/VGPR budget on padded packed-cache rows
does not recover the Vulkan/RADV schedule. It leaves the instruction mix and
LDS-read cadence essentially unchanged while lowering residency/resource
quality. The next packed-path probe should target actual LDS-read issue order,
B-cache ownership, or instruction scheduling; the next direct-WMMA probe should
target RADV-like cooperative-matrix load/store scheduling rather than another
padding variant.

## MMQL128 Bsplit4 Packed Diagnostic

A packed-Q8_1/x4 B-cache ownership follow-up was added as
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bsplit4_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BSPLIT4_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bsplit4-focused-20260618-004510/`

Facts:

- compiled through CMake/Ninja;
- preserved the current Q4_K MMQL128 staged packed dataflow, `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, and output ownership;
- changed staged B payload ownership from two lanes per B row loading four
  packed words each to four lanes per B row loading two packed words each;
- HSACO metadata: wave64, SGPR `48`, VGPR `140`, LDS `8192`, no spills;
- static ISA count stayed close to current MMQL128: 512 integer-dot sites, 46
  `ds_loads`, 12 `ds_stores`, 18 global loads, 64 global stores, 188 waits,
  and 2 barriers;
- passed CPU-reference focused p512 and exact p513 rows while selecting the
  bsplit4 provider;
- exact p33 rows passed while staying on the existing MMQ64 route.

The performance result rejects B payload split alone as a production route:

| Row | Current p512 | Bsplit4 p512 | Current p513 | Bsplit4 p513 |
| --- | ---: | ---: | ---: | ---: |
| Kcur | `977.70 us` | `1039.18 us` | `603.45 us` | `644.47 us` |
| Qcur | `3941.94 us` | `4511.51 us` | `2420.84 us` | `2479.59 us` |
| ffn_out | `14287.02 us` | `14024.76 us` | `8053.23 us` | `8359.72 us` |
| ffn_gate | `14007.78 us` | `13693.10 us` | `7682.05 us` | `9098.94 us` |

Conclusion: splitting the packed-B payload load across more lanes is not the
missing RADV parity axis. It improves two p512 FFN rows slightly but loses K/Q
and regresses every p513 tail row. The next packed-path probe should compare
actual RADV and HRX issue order around global loads, LDS stores, waitcnts, and
the first dot-consume sequence before changing another local ownership rule.

## Half4 LDS Fragment-Load Diagnostic

After comparing the DeepSeek Q4_K RADV oracle ISA against the current HRX
wave64 padded direct-WMMA route, the next exact-schedule hypothesis was that
the HIP route's LDS fragment feed was too coarse. RADV issues 64 static
`ds_load_b64` sites around 32 `v_wmma_f16_16x16x16_f16` sites, while the
existing HIP wave64 route issues 32 `ds_load_b128` sites around the same 32
static WMMAs.

Diagnostic source:
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_h4load_f16acc_wg256_f32`,
guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_H4LOAD_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/`

Compile/ISA comparison:

| Kernel | LDS fragment loads | WMMA | waits | SGPR | VGPR | LDS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RADV large coopmat Q4_K | 64 x `ds_load_b64` | 32 | 50 event-window waits | 108 | 192 | 22528 |
| HRX padded W64 baseline | 32 x `ds_load_b128` | 32 | 133-155 | 29 | 165 | 20480 |
| HRX half4 non-volatile source | 32 x `ds_load_b128` | 32 | 155 | 29 | 165 | 20480 |
| HRX half4 volatile source | `flat_load_b64`, not LDS | 28 | 231 | 32 | 233 | 20480 |

Decision: reject at the compile/ISA gate. Non-volatile half4 source coalesces
back to the existing `ds_load_b128` schedule, so it does not test the RADV
schedule. Volatile half4 produces a worse lowering: flat memory reads with
immediate `vmcnt` waits, higher VGPR pressure, and fewer static WMMA sites.

Next direct-WMMA Q4_K work should not repeat C++ vector spelling as the control
knob. To move closer to RADV, the useful options are a source layout that
naturally lowers to LDS `ds_load_b64`, an explicit compiler/inline-asm route
for the fragment reads if maintainable, or a return to the packed-Q8_1 path
with the RADV issue order used only as an oracle for load/wait scheduling.

## B64ASM LDS Fragment-Load Diagnostic

The explicit inline-assembly follow-up used `ds_read_b64` for the A/B LDS
fragment reads in the same wave64 padded direct-WMMA route:
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_b64asm_f16acc_wg256_f32`,
guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_B64ASM_F16ACC_WG256_PROMPT=1`.

Artifact:
`cache/hrxv1/gfx1151/q4-w64-b64asm-focused-20260618-010917/` and
`cache/hrxv1/gfx1151/q4-w64-b64asm-focused-20260618-010917-wait/`.
Final LDS-base artifact:
`cache/hrxv1/gfx1151/q4-w64-b64asm-ldsbase-20260618-012420/`.
ISA dumps are in
`cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/`.

Compile/ISA result:

| Kernel | LDS fragment loads | WMMA | waits | SGPR | VGPR | LDS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RADV large coopmat Q4_K | 64 x `ds_load_b64` | 32 | 50 event-window waits | 108 | 192 | 22528 |
| HRX B64ASM no explicit wait | 256 x `ds_load_b64` | 32 | 140 | 30 | 169 | 20480 |
| HRX B64ASM half4 output + full explicit wait | 256 x `ds_load_b64` | 32 | 396 | 30 | 169 | 20480 |
| HRX B64ASM explicit LDS base | 256 x `ds_load_b64` | 32 | 396 | 30 | 169 | 20480 |

Correctness result:

- p512 selected the B64ASM provider for all four rows. Without a memory
  clobber on the inline asm, CPU-reference comparison failed with NaN/inf
  outputs. Adding the clobber changed the failure to finite ERR roughly
  `0.0060-0.0115`, still above the `0.0005` tolerance.
- p33 did not select B64ASM and passed on the existing narrow route.
- p513 selected B64ASM for all four rows. The final half4-output,
  memory-clobbered spelling failed with finite ERR roughly `0.0016-0.0117`.
- Recasting the shared-memory base arrays to explicit `address_space(3)`
  pointers before fragment pointer arithmetic did not materially change the
  error band: p512 failed with ERR roughly `0.0058-0.0115`, and p513 failed
  with ERR roughly `0.0017-0.0115`.
- Direct half4 asm output, `s_waitcnt lgkmcnt(0)` after each forced LDS read,
  and explicit LDS address-space pointer arithmetic did not restore
  correctness.

Decision: reject for production promotion. This is valuable evidence because
it proves HIP C++ can force LDS `ds_load_b64`, but the current generic-pointer
inline-asm fragment spelling is not semantically equivalent to the passing W64
route or the RADV cooperative-matrix load sequence. Future direct-WMMA work
needs a correct fragment map and reuse schedule, not just forced instruction
width or a different LDS pointer spelling.

## B64GROUP Fragment-Reuse Diagnostic

The grouped follow-up kept the same wave64 VK128 padded direct-WMMA route and
explicit `ds_read_b64` LDS reads, but changed the compute window to load four
A fragments and four B fragments per k-half before issuing the 4x4 WMMA block:
`hrx_mul_mat_vec_q4_k_wmma16x16_vk128_padded_w64_b64group_f16acc_wg256_f32`,
guarded by
`GGML_HRX_ENABLE_Q4_K_WMMA16_VK128_PADDED_W64_B64GROUP_F16ACC_WG256_PROMPT=1`.

Artifacts:
`cache/hrxv1/gfx1151/q4-w64-b64group-focused-20260618-013423/`,
`cache/hrxv1/gfx1151/q4-w64-b64group-waited-focused-20260618-013533/`, and
`cache/hrxv1/gfx1151/q4-w64-b64group-waited-kcur-perf-20260618-014159/`.
ISA dumps:
`cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/hrx-q4-vk128-padded-w64-b64group.isa.txt`
and
`cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/hrx-q4-vk128-padded-w64-b64group-waited.isa.txt`.

Compile/ISA result:

| Kernel | LDS fragment loads | WMMA | waits | SGPR | VGPR | LDS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RADV large coopmat Q4_K | 64 x `ds_load_b64` | 32 | 50 event-window waits | 108 | 192 | 22528 |
| HRX B64GROUP no-wait | 64 x `ds_load_b64` | 32 | 142 | 29 | 199 | 20480 |
| HRX B64GROUP waited | 64 x `ds_load_b64` | 32 | 206 | 29 | 199 | 20480 |

Correctness and timing result:

- The no-wait grouped spelling selected p512/p513 but failed with NaNs.
- The waited grouped spelling selected p512/p513, kept p33 on the narrow Q4
  route, and passed p512, exact p33, and exact p513 focused CPU-reference
  gates.
- Same-row Kcur p512 perf with the accepted MMQL128 baseline gate pinned
  regressed from `1038.69 us` to `3103.19 us`.

Decision: reject for production promotion. This is useful because it proves the
HIP source can match RADV's `64 ds_load_b64` / `32 WMMA` load-count family with
correct results and no spills, but that alone is still about 3x slower on the
small Kcur row. The remaining schedule gap is now the wait/issue structure and
the broader direct-F32 WMMA dataflow, not merely LDS load width or fragment
reuse count.

## MMQL128 B-Pair Packed Diagnostic

The next packed-Q8_1/x4 probe stayed in the current winning MMQL128 schedule
family instead of adding another direct-F32 WMMA variant. It added
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bpair_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BPAIR_PROMPT=1`.

Artifacts:

- Focused correctness/timing:
  `cache/hrxv1/gfx1151/q4-mmql128-bpair-focused-20260618-015209/`
- DeepSeek p512 model A/B:
  `cache/hrxv1/gfx1151/q4-mmql128-bpair-deepseek-model-ab-20260618-015712/`
- ISA:
  `cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/hrx-q4-mmql128-bpair.isa.txt`

Schedule axis:

- Preserve current packed dataflow: Q4_K A cache, Q8_1 x4 B cache, `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, `TM=4`, `TN=2`, `WNITER=8`.
- Change only B-cache consume order: preload the two B-cache rows for each
  `TN=2` micro-iteration before issuing the dot loop.
- This brackets the RADV oracle's larger post-barrier LDS-read window and
  gradual wait drain while avoiding another direct-F32 WMMA dataflow change.

Compile/ISA facts:

| Kernel | v_dot | LDS ops | global loads | waits | SGPR | VGPR | LDS | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current MMQL128 | 512 | 58 | 18 | 185 | 53 | 139 | 8192 | 0 |
| B-pair MMQL128 | 512 | 58 | 18 | 183 | 53 | 149 | 8192 | 0 |

Focused result before tail guard:

| Row | Current p512 | B-pair p512 | Current p513 | B-pair p513 |
| --- | ---: | ---: | ---: | ---: |
| Kcur | `997.49 us` | `973.23 us` | `616.21 us` | `606.88 us` |
| Qcur | `3980.86 us` | `3985.36 us` | `2482.39 us` | `2672.07 us` |
| ffn_out | `16693.81 us` | `13301.63 us` | `9188.83 us` | `9436.65 us` |
| ffn_gate | `12954.04 us` | `12313.35 us` | `7481.62 us` | `7007.56 us` |

The p512 full-tile signal is positive, especially on FFN out. The exact p513
tail signal is mixed: Kcur and ffn_gate improve, but Qcur and ffn_out regress.
The selector was therefore tightened to `cols % 128 == 0`; with the guard,
p512 selects B-pair, exact p513 falls back to current MMQL128, and p33 stays on
the narrow Q4 routes. Guarded p512/p33/p513 focused CPU-reference gates passed.

DeepSeek R1 Qwen 14B Q4_K_M same-binary p512/fa1 model A/B with current-best
env plus the B-pair gate improved `246.34 -> 258.77 tok/s` (`+5.0%`) with no
fallback strings and the same Q4 route count. This is not a broad production
promotion yet because it is p512/full-tile evidence only, but it is the first
post-oracle packed-path schedule axis with a positive model-level signal.

Interpretation: B-cache LDS issue order is a productive packed-path axis, but
tail policy must stay separate. The next exact-schedule step should inspect
why the full-tile FFN rows benefit while Qcur is flat and p513 tails split,
then either add a tail-specific variant or test a nearby B/A read-clustering
axis under the same full-tile/tail split discipline.

## MMQL128 B-Quad Packed Diagnostic

The B-pair result made B-cache LDS issue order the first productive
post-oracle packed-path axis. The adjacent follow-up widened that read cluster:
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`.

Artifacts:

- Focused correctness/timing:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-focused-20260618-020428/`
- DeepSeek p512 model A/B:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-deepseek-model-ab-20260618-020603/`
- Llama 3.1 8B p512 model A/B:
  `cache/hrxv1/gfx1151/q4-mmql128-bquad-llama31-model-ab-20260618-020644/`
- ISA:
  `cache/hrxv1/gfx1151/q4-radv-hrx-isa-compare-20260618/hrx-q4-mmql128-bquad.isa.txt`

Schedule axis:

- Preserve current packed dataflow: Q4_K A cache, Q8_1 x4 B cache, `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, `TM=4`, `TN=2`, `WNITER=8`.
- Change only B-cache consume order: preload four B-cache rows across two
  adjacent `WNITER` positions before issuing dots.
- Keep the same full-tile/tail split as B-pair: `cols % 128 == 0` selects
  B-quad, exact p513 tails fall back to current MMQL128, and p33 stays narrow.

Compile/ISA facts:

| Kernel | v_dot | LDS ops | global loads | waits | SGPR | VGPR | LDS | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current MMQL128 | 512 | 58 | 18 | 185 | 53 | 139 | 8192 | 0 |
| B-pair MMQL128 | 512 | 58 | 18 | 183 | 53 | 149 | 8192 | 0 |
| B-quad MMQL128 | 512 | 58 | 18 | 183 | 53 | 169 | 8192 | 0 |

The emitted event window starts with a broader post-barrier LDS read cluster
than B-pair. It spends more VGPR but does not spill.

Focused p512 timing:

| Row | Current | B-pair | B-quad |
| --- | ---: | ---: | ---: |
| Kcur | `984.94 us` | `966.22 us` | `979.48 us` |
| Qcur | `3953.79 us` | `3902.92 us` | `3944.81 us` |
| ffn_out | `15371.16 us` | `12725.16 us` | `11617.99 us` |
| ffn_gate | `13138.23 us` | `12014.82 us` | `11865.43 us` |

B-quad is not uniformly better on K/Q, but it is the best summed p512 focused
candidate because it materially improves the hot FFN rows. Guarded focused
correctness passed for p512, p33, and p513; p512 selected B-quad, p513 fell
back to current MMQL128, and p33 stayed on narrow routes.

Same-binary p512/fa1 model A/B with current-best env:

| Model | Current | B-pair | B-quad |
| --- | ---: | ---: | ---: |
| DeepSeek R1 Qwen 14B Q4_K_M | `249.47 tok/s` | `256.59 tok/s` | `261.03 tok/s` |
| Llama 3.1 8B Q4_K_M | `499.38 tok/s` | not rerun | `527.82 tok/s` |

Decision: accept B-quad as the current full-tile Q4_K MMQL128 gfx1151 default
candidate. It is still not a tail route. The selector defaults it only on
gfx1151 and only for full column tiles; rollback is
`GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`, and opt-in elsewhere
remains `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_PROMPT=1`.

Default-regate artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bquad-default-regate-20260618-051326/`.
Focused CPU-reference tests passed for p33, p512, and exact p513 without the
B-quad enable env. Route traces prove p512 selected B-quad, rollback selected
current MMQL128, p33 stayed on MMQ64, and exact p513 fell back to current
MMQL128. Same-runner p512 perf default vs rollback was:

| Row | Default B-quad | Rollback MMQL128 |
| --- | ---: | ---: |
| Kcur | `1004.51 us` | `1003.02 us` |
| Qcur | `3878.57 us` | `4091.33 us` |
| ffn_out | `13806.01 us` | `14990.07 us` |
| ffn_gate | `11124.93 us` | `13488.00 us` |

The remaining work for parity is to carry this exact schedule discipline into
the next rows: add a p513-specific Q4_K tail schedule or mine the next
Vulkan/RADV delta for Q5_K/Q6_K/Q8_0 prompt rows.

## B-quad Large-Tail Policy

The p513 tail follow-up tested whether the accepted B-quad full-tile provider
could cover Vulkan's production-width tail regime. This is not a new kernel;
it exposes the existing `full_tile`/edge-tile split in
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_wg256_f32` for large tails.
The first four p513 column workgroups use the unguarded full-tile path and the
fifth workgroup uses the guarded edge path, matching the Vulkan oracle's large
pipeline plus fifth workgroup shape.

Probe artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bquad-tail-probe-20260618-052036/`.
With `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_TAIL_PROMPT=1`, focused
CPU-reference tests passed for p33, p512, and exact p513. Route traces proved
p33 stayed on MMQ64, p512 stayed on B-quad, and p513 selected B-quad.

p513 same-runner focused default MMQL128 vs B-quad tail:

| Row | Default MMQL128 | B-quad Tail |
| --- | ---: | ---: |
| Kcur | `642.97 us` | `605.69 us` |
| Qcur | `2321.87 us` | `2493.24 us` |
| ffn_out | `8453.90 us` | `7847.65 us` |
| ffn_gate | `7390.19 us` | `7682.43 us` |

Same-binary Llama 3.1 8B Q4_K_M p513/fa1 model A/B:
`cache/hrxv1/gfx1151/q4-mmql128-bquad-tail-llama31-model-ab-20260618-052222/`.
Baseline selected current MMQL128 for `570` Q4 dispatches and reached
`564.62 tok/s`; tail selected B-quad for the same `570` Q4 dispatches and
reached `585.20 tok/s`.

Default-regate artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bquad-tail-default-regate-20260618-052352/`.
Default p513 now selects B-quad, tail rollback
`GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_TAIL_PROMPT=1` selects current
MMQL128, p33 stays on MMQ64, and p512 stays on B-quad. p513 default vs
tail-rollback focused timing was:

| Row | Default B-quad Tail | Tail Rollback MMQL128 |
| --- | ---: | ---: |
| Kcur | `608.39 us` | `614.95 us` |
| Qcur | `2477.16 us` | `2347.09 us` |
| ffn_out | `8248.31 us` | `8013.47 us` |
| ffn_gate | `7617.92 us` | `8274.78 us` |

Decision: accept B-quad for gfx1151 large Q4_K prompt tails with `cols >= 512`.
Keep p33 on the narrow route. The next Q4_K work should not broaden this below
production-width tails without new odd-size evidence; it should mine the
remaining direct Vulkan/RADV schedule deltas.

## MMQL128 B-Oct Packed Compile Gate

The adjacent packed-path probe after B-pair and B-quad was
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_boct_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BOCT_PROMPT=1`.

Schedule axis:

- Preserve the accepted packed-Q8_1/x4 MMQL128 dataflow: `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, `TM=4`, `TN=2`, `WNITER=8`.
- Broaden only the B-cache consume window beyond B-quad by preloading all
  eight WNITER B-cache positions before issuing dots.
- Keep the selector opt-in and full-tile only, because the accepted B-quad tail
  policy is already guarded separately.

Compile artifact:
`cache/hrxv1/gfx1151/q4-mmql128-boct-compile-20260618/`.

Compile result:

| Kernel | SGPR | VGPR | VGPR spills | Private segment | LDS | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B-quad MMQL128 | 53 | 169 | 0 | 0 | 8192 | accepted default |
| B-oct MMQL128 | 53 | 192 | 94 | 376 | 8192 | reject compile gate |

Decision: reject before focused runtime. B-oct brackets the positive B-cache
read-clustering axis but exceeds the gfx1151 no-spill envelope. The useful
packed-path cluster remains B-quad; further work should look for a lower-cost
way to change wait/issue order rather than keeping the entire B window live.

## MMQL128 B-Half Packed Diagnostic

The midpoint between accepted B-quad and rejected B-oct was added as:
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bhalf_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BHALF_PROMPT=1`.

Schedule axis:

- Preserve the accepted B-quad dataflow: packed Q8_1/x4 RHS, `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, `TM=4`, `TN=2`, and current output
  ownership.
- Preload four WNITER B-cache positions before issuing dots. This carries half
  of the full B-oct live B window, so it directly tests whether B-oct's
  pre-dot load-window movement can be approximated without B-oct's full spill
  cost.
- Keep the selector opt-in and full-column-tile only.

Artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bhalf-static-ab18e9465-20260619-233321/`.

Compile result:

| Kernel | SGPR | VGPR | VGPR spills | Private segment | LDS | Pre-hot loads | Final wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-quad MMQL128 | 53 | 169 | 0 | 0 | 8192 | 16 | 9 |
| B-half MMQL128 | 53 | 192 | 18 | 76 | 8192 | 26 | 19 |
| B-oct MMQL128 | 53 | 192 | 94 | 376 | 8192 | 46 | 0 |

Decision: reject before focused runtime. B-half moves the parsed issue window
in the intended direction, but it already hits scratch traffic and VGPR spills.
This closes simple larger B-cache live-window preloading in the current HIP
C++ source shape: B-quad is the largest no-spill accepted point, B-half spills,
and B-oct spills harder.

## MMQL128 B-Quad-CR Packed Diagnostic

The next adjacent packed-path probe kept the B-quad cluster size but changed
consume order:
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_cr_wg256_f32`, guarded by
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_CR_PROMPT=1`.

Schedule axis:

- Preserve the accepted B-quad dataflow: packed Q8_1/x4 RHS, `BM=128`,
  `BN=128`, wave64, `BK_STEP=1`, `TM=4`, `TN=2`, and two-WNITER B-cache
  clustering.
- Change only dot consume order: each A micro-row drains the loaded B cluster
  before moving to the next A row.
- Keep the selector opt-in and full-tile only.

Artifacts:

- Compile: `cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-compile-20260618/`
- Focused: `cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-focused-20260618/`

Compile result: wave64, SGPR `53`, VGPR `167`, LDS `8192`, no spills. This
passes the compile gate and uses slightly fewer VGPR than accepted B-quad.

Focused correctness passed for p33, p512, and exact p513. Route traces prove
p512 selected B-quad-CR, p33 stayed on the narrow route, and p513 stayed on the
accepted B-quad tail route.

Focused p512 timing:

| Row | Default B-quad | B-quad-CR | Ratio |
| --- | ---: | ---: | ---: |
| Kcur | `982.78 us` | `962.51 us` | `0.979x` |
| Qcur | `3904.55 us` | `3893.61 us` | `0.997x` |
| ffn_out | `13024.67 us` | `14032.85 us` | `1.077x` |
| ffn_gate | `11422.31 us` | `12004.63 us` | `1.051x` |

Decision: reject for production. The consume-order change lowers VGPR pressure
and slightly improves K/Q rows, but regresses the FFN rows that dominate model
throughput. Keep accepted B-quad as the current point on this axis. The next
packed-path work should be row-family specific if it targets K/Q, or otherwise
preserve the B-quad FFN consume order.

## MMQL128 B-Quad-CR QK-Only Policy

The row-family split implied by the B-quad-CR focused result was tested as an
opt-in selector policy:
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BQUAD_CR_QK_PROMPT=1`.

Policy:

- select `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bquad_cr_wg256_f32`
  only for full-tile Q4_K rows with `k <= 5120`, `rows <= 5120`,
  `cols >= 128`, and `cols % 128 == 0`;
- keep accepted B-quad for FFN rows, p513 tails, and other production-width
  full-tile rows;
- keep p33/narrow prompt on the existing narrow Q4_K routes.

Artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-qk-policy-20260618/`.
Model A/B artifact:
`cache/hrxv1/gfx1151/q4-mmql128-bquad-cr-qk-policy-model-ab-20260618/`.

Focused p33, p512, and p513 CPU-reference gates passed. Route traces prove
p512 selected B-quad-CR only for the K/Q-style rows, selected accepted B-quad
for FFN rows, p513 selected accepted B-quad for all Q4_K rows, and p33 stayed
on the narrow routes.

Same-binary model A/B did not justify a default policy:

| Model | Baseline | QK-only policy | Ratio |
| --- | ---: | ---: | ---: |
| DeepSeek R1 Qwen 14B Q4_K_M p512 | `246.404409 tok/s` | `247.043538 tok/s` | `1.0026x` |
| Llama 3.1 8B Q4_K_M p512 | `444.387307 tok/s` | `443.123607 tok/s` | `0.9972x` |

Decision: reject default promotion. This is a useful schedule-local result, not
a production win. The next Q4_K packed-path attempt should keep zeroing in on
the exact Vulkan/RADV winning schedule and current HRX HSACO deltas at the
kernel/schedule level before another model-level promotion test.

## MMQL64 BK2 B-Quad Narrow Policy

The successful Q5 narrow B-quad pivot was ported to the accepted Q4_K p33
MMQL64 BK2 route as:
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bquad_wg256_f32`.

Schedule axis:

- Preserve the accepted Q4_K MMQL64 BK2 narrow dataflow: `BM=64`, `BN=64`,
  wave64, `BK_STEP=2`, `TM=4`, `TN=2`, and packed Q8_1/x4 RHS.
- Change only B-cache consume order: preload the two WNITER B-cache positions
  before issuing dots.
- Keep p512/p513 on accepted MMQL128 B-quad; do not let this narrow route
  steal production-width rows.

Artifacts:

- Initial focused and force-policy probe:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bquad-focused-20260619-000155/`
- Llama 3.1 8B model A/B:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bquad-llama31-model-ab-20260619-000310/`
- Llama 3.2 3B guardrail A/B:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bquad-llama32-model-ab-20260619-000347/`
- Final narrowed default regate:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bquad-rows4096-regate-20260619-000813/`

Compile result: wave64, SGPR `50`, VGPR `115`, LDS `8192`, no spills.

The initial broad force policy showed the axis is useful but shape-sensitive:

| Row | Result |
| --- | --- |
| Focused p33 exact rows | `2877.13 us -> 2390.95 us` |
| Llama 3.1 8B Q4_K_M p33 force A/B | `208.68 -> 222.70 tok/s` |
| Llama 3.2 3B Q4_K_M p33 force A/B | `473.99 -> 424.65 tok/s` |

The Llama 3.2 regression came from selecting the new route for
`k=8192, rows=3072` `ffn_out` rows. The accepted default therefore requires
`k >= 4096`, `rows >= 4096`, and `32 <= cols <= 64`, with rollback
`GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL64_BK2_BQUAD_PROMPT=1` and force
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL64_BK2_BQUAD_PROMPT=1`.

Final regate:

| Gate | Result |
| --- | --- |
| p33 CPU-reference | passed; three rows selected B-quad and Kcur stayed BK2 |
| p512/p513 CPU-reference | passed; both stayed on MMQL128 B-quad |
| Focused p33 timing | `2931.30 us -> 2430.14 us` vs rollback |
| Llama 3.1 8B Q4_K_M p33 | `200.02 -> 220.22 tok/s` vs rollback |
| Llama 3.2 3B Q4_K_M p33 | selected no B-quad rows by default |

Decision: accept as a gfx1151 narrow-row default under the narrowed shape
guard. This is a schedule-level lift toward the Vulkan medium p33 route, not a
parity claim; future p33 work should continue to separate `rows=3072`/small
hidden rows from `rows>=4096` rows instead of using one narrow policy for all
models.

## MMQL64 BK2 B-Pair Narrow Probe

The B-pair variant tested the adjacent B-cache consume order after B-quad won
for Llama 3.1 rows `>=4096` but regressed Llama 3.2 rows `3072` under force.
It is retained as an opt-in diagnostic:
`hrx_mul_mat_vec_q4_k_q8_1_x4_mmql64x64_bk2_bpair_wg256_f32`.

Schedule axis:

- Preserve the accepted Q4_K MMQL64 BK2 narrow dataflow: `BM=64`, `BN=64`,
  wave64, `BK_STEP=2`, `TM=4`, `TN=2`, and packed Q8_1/x4 RHS.
- Change only B-cache consume order: preload each TN=2 B-cache pair before
  issuing dots.
- Force only with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL64_BK2_BPAIR_PROMPT=1`.

Artifacts:

- Focused/backend-op gate:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bpair-focused-20260619-001650/`
- Llama 3.2 3B model A/B:
  `cache/hrxv1/gfx1151/q4-mmql64-bk2-bpair-llama32-model-ab-20260619-001844/`

Compile result: wave64, SGPR `50`, VGPR `103`, LDS `8192`, no spills.

Focused gates passed: Llama 3.1 p33, p512, p513, and Llama 3.2 p33
CPU-reference rows were correct. p512/p513 stayed on the accepted MMQL128
B-quad route even when B-pair was enabled.

Focused timing was locally positive for Llama 3.2 p33:

| Gate | Result |
| --- | --- |
| Llama 3.2 focused p33 default | `10503.04 us` |
| Llama 3.2 focused p33 forced B-pair | `10311.44 us` |
| Llama 3.2 focused p33 forced B-quad | `10397.44 us` |

Model-level A/B rejected the selector:

| Row | Result |
| --- | --- |
| Llama 3.2 3B Q4_K_M p33 default | `460.129 tok/s` |
| Llama 3.2 3B Q4_K_M p33 forced B-pair | `441.843 tok/s` |
| Forced B-pair route count | `830` Q4 rows |
| Llama 3.1 focused p33 forced globally | `2480.52 -> 2636.33 us` regression |

Decision: reject for default promotion. This is a useful example of why
focused backend-op timing is necessary but not sufficient; the exact same
route had a small per-op signal that did not survive integrated model
scheduling. Future rows `3072` work needs another axis, likely row/column
ownership or a closer RADV medium clone, not another B-cache pair-order retry.

## Candidate Gate - Vulkan-Large Coopmat Clone Probe

- Production target: DeepSeek R1 Qwen 14B Q4_K_M and Llama 3.1 8B Q4_K_M
  dense prefill at p512/p513, with p33 kept on the existing narrow route unless
  separately proven.
- Baseline command: focused `test-backend-ops` Q4_K p512/p513 rows with
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1` and current-best gfx1151
  gates.
- Variant command: same focused rows with an opt-in direct-f32-RHS Q4_K
  coopmat/WMMA clone gate, no default promotion.
- Same-runner comparison method: one binary, one build tree, sequential
  baseline/variant focused rows, route trace required for each row; model A/B
  only after focused correctness and timing pass.
- Route trace path:
  `cache/hrxv1/gfx1151/<candidate>/routes.log` and exact p33/p513 route logs.
- Scheduler/per-op trace path: same focused artifact `perf.csv`; add HSACO ISA
  dumps under `.tmp/hrxv1/q4-isa/` before interpreting timing.
- Focused CPU-reference command: exported Q4_K rows from the DeepSeek trace
  plus exact p33 and p513 rows from
  `cache/hrxv1/gfx1151/q4-wmma16-wg256-focused-20260617-223548/`.
- Compile report path: built HSACO metadata in
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/`
  plus `llvm-objdump` ISA/resource notes.
- Target listing path: generated HRX catalog and build target list from CMake;
  HIP C++ compilation must remain under Ninja.
- Prior-art schedule source:
  `mul_mm.comp` and `mul_mm_funcs.glsl` Vulkan `DATA_A_Q4_K` coopmat1 path,
  RADV artifact
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/`.
- Promotion rule: may only become a production-width gfx1151 route if it
  passes CPU-reference p512/p513 rows, does not select for p33 unless p33 also
  wins, beats current MMQL128 focused timings on representative DeepSeek Q4_K
  rows, and improves same-binary model A/B without introducing CPU fallback or
  route contamination.

## B-half-CR Compile Gate

The post-B-half adjacent probe tested whether the CR-major consume order that
reduced B-quad pressure could rescue the B-half live B-cache window:

- route:
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_bhalf_cr_wg256_f32`;
- env:
  `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_BHALF_CR_PROMPT=1`;
- artifact:
  `cache/hrxv1/gfx1151/q4-mmql128-bhalf-cr-static-e7e35977e-20260619-234116/`;
- build:
  normal CMake/Ninja HSACO generation, no assembler helper;
- axis:
  preserve B-half's four-WNITER B-cache preload, but drain the loaded cluster
  by A micro-row first.

Static resource summary:

| Route | SGPR | VGPR | VGPR Spills | Scratch/Private Bytes | LDS Bytes | Pre-Hot Loads | Final Pre-Hot LGKMCNT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-quad MMQL128 | 53 | 169 | 0 | 0 | 8192 | 16 | 9 |
| B-quad-CR MMQL128 | 53 | 167 | 0 | 0 | 8192 | 16 | 9 |
| B-half-CR MMQL128 | 53 | 192 | 16 | 68 | 8192 | 26 | 19 |
| B-half MMQL128 | 53 | 192 | 18 | 76 | 8192 | 26 | 19 |
| B-oct MMQL128 | 53 | 192 | 94 | 376 | 8192 | 46 | 0 |

Decision: reject before focused runtime. CR-major consume order helps only
inside the B-quad live-state envelope; it does not make four WNITER B-cache
positions viable on gfx1151. Do not add more probes that simply preload a
larger B-cache cluster in this source shape. The next packed Q4 path needs a
lower-pressure issue-order primitive, different B ownership, or a different
dataflow/lane ownership idea.

## Rows=3072 MMQL128 Policy

The current worst basket row, Llama 3.2 3B Q4_K_M p512, exposed a row-family
split inside the packed MMQL128 family after the B-cache preload axis was
bracketed:

- artifact:
  `cache/hrxv1/gfx1151/q4-llama32-rows3072-cols512-default-regate-2d7392555-20260620-000208/`;
- model artifact:
  `cache/hrxv1/gfx1151/q4-llama32-rows3072-cols512-default-model-ab-2d7392555-20260620-000317/`;
- selector:
  default base `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_wg256_f32` on
  gfx1151 only when `rows == 3072`, `cols == 512`, `k % 256 == 0`, and packed Q8_1 x4
  is available by default;
- rollback:
  `GGML_HRX_DISABLE_Q4_K_Q8_1_X4_MMQL128_ROWS3072_PROMPT=1`.

Focused p512 result:

| Variant | Total us | Route selection |
| --- | ---: | --- |
| Rollback B-quad | 5464.462 | B-quad for all four focused rows |
| Rows=3072 cols512 default | 5073.574 | base MMQL128 for rows=3072 rows, B-quad elsewhere |

Model A/B on Llama 3.2 3B Q4_K_M p512 improved `1475.154347 ->
1534.832691 tok/s` (`1.040x`) versus rollback. Route traces showed the new
default selected base MMQL128 for `350` Q4 dispatches and kept B-quad for
`480`; rollback selected B-quad for all `830` Q4 dispatches. The p33
non-steal gate passed and stayed on MMQL64 BK2/B-quad narrow routes. The final
focused p513 regate selected only B-quad under both default and rollback, so
p513 remains outside the default policy.

Decision: accept as a narrow policy split. This does not reopen the simple
larger-B-cache preload axis and does not invalidate B-quad as the general Q4
full-tile route. It records that rows=3072 prefers the base MMQL128 consume
order on gfx1151, while other production-width Q4 rows still need separate
evidence before moving away from B-quad.

## Next Candidate Rules

Before adding a new Q4_K dense prompt candidate:

- compare the RADV ISA around `v_wmma`, loads, LDS traffic, and barriers
  against the current HRX HSACO disassembly;
- write down the exact axis being changed, bounded by this ledger;
- keep `p33` on the narrower route unless focused evidence proves otherwise;
- test focused CPU-reference rows for p33, p512, and p513 before model A/B;
- require route evidence proving the intended provider selected;
- keep any new HIP C++ source in CMake/Ninja, not a helper compiler path.

First candidate axes to consider:

- prove RADV lane/subgroup ownership from SPIR-V and ISA before adding another
  direct-WMMA HIP route;
- treat wave64 direct-WMMA Q4_K as structurally proven but not promotable:
  correctness passes after the lane-id fix, yet focused perf regresses; future
  direct-WMMA work must target RADV-like load/store scheduling or avoid direct
  F32 RHS costs rather than re-testing the same layout;
- rework packed Q8_1 x4 load ordering, B ownership, or wait placement only
  against a written schedule delta from the Vulkan oracle and current HSACO;
- preserve the current wide-route guard and only widen it after odd/tail rows
  show a clear win.
