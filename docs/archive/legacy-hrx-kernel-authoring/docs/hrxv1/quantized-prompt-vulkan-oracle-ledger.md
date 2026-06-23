# HRX v1 gfx1151 Quantized Prompt Vulkan Oracle Ledger

Date: 2026-06-17

## Scope

This ledger consolidates the first p512/fa1 Vulkan oracle matrix for the
current worst HRX v1 gfx1151 dense quantized prompt rows. It is the prior matrix
for the next HIP C++ prompt-matmul candidates.

The important result is that Vulkan is not using unrelated one-off schedules
for Q4_K, Q5_K, Q6_K, and Q8_0 production-width prefill. The hot dense
quantized prompt pipelines all use the same large aligned schedule family:

```text
spec=[256,128,128,32,64,64,2,16,16,16,64]
wg_denoms=[128,128,1]
workgroup=256x1x1
full subgroup required
LDS=22528
VGPR=192
spills=0
```

Any new HRX dense prompt candidate should be treated as a test against this
family, not as another blind tile sweep.

## Capture Matrix

| Model row | Artifact | Tok/s | Top dense pipeline | Dispatches | Resource facts |
| --- | --- | ---: | --- | ---: | --- |
| Llama 3.1 8B Q4_K_M p512/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q4km-p512-fa1-20260617-195212/` | 371.807 | `matmul_q4_k_f32_f16acc_aligned_l` `0x5666175250529efb` | 190 | SGPR 108, VGPR 192, LDS 22528, no spills |
| Qwen2.5 Coder 7B Q5_K_M p512/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p512-fa1-20260617-200349/` | 365.256 | `matmul_q5_k_f32_f16acc_aligned_l` `0x0ee599afb33ff07b` | 166 | SGPR 108, VGPR 192, LDS 22528, no spills |
| DeepSeek R1 Qwen 14B Q4_K_M p512/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-deepseek-r1-qwen14b-q4km-p512-fa1-20260617-200426/` | 290.539 | `matmul_q4_k_f32_f16acc_aligned_l` `0x5666175250529efb` | 286 | SGPR 108, VGPR 192, LDS 22528, no spills |
| Qwen3 30B Q6_K p512/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p512-fa1-20260617-200437/` | 206.049 | `matmul_q6_k_f32_f16acc_aligned_l` `0x6eebdfb4c3043b23` | 192 | SGPR 108, VGPR 192, LDS 22528, no spills |
| Llama 3.1 8B Q8_0 p512/fa1 | `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p512-fa1-20260617-200453/` | 423.308 | `matmul_q8_0_f32_f16acc_aligned_l` `0x72d309e22f889977` | 221 | SGPR 108, VGPR 192, LDS 22528, no spills |

All rows reported `backends=Vulkan`, emitted SPIR-V, SPIR-V assembly, split
RADV ISA/stats, and normalized shape inventories.

## Normalized Schedule Shape

The large aligned pipelines cover production-width prompt matmul as 128x128
workgroup-denominator tiles. Representative normalized shape buckets:

- Q4_K Llama 8B: q4_K `[4096,14336]` x f32 `[4096,512]` -> f32
  `[14336,512]`, workgroups `[112,4,1]`.
- Q4_K DeepSeek 14B: q4_K production rows dispatch through the same Q4_K
  pipeline with 286 total dense Q4_K dispatches.
- Q5_K Qwen2.5 Coder: q5_K `[3584,18944]` x f32 `[3584,512]` -> f32
  `[18944,512]`, workgroups `[148,4,1]`.
- Q6_K Qwen3 30B: dense Q6_K routes use the same aligned family and pair with
  `split_k_reduce` for split-K rows.
- Q8_0 Llama 8B: q8_0 production prompt rows use
  `matmul_q8_0_f32_f16acc_aligned_l`, not the vector route, for 221
  dispatches.

The residual final-token/tail regime is separate. The captures also show
`mul_mat_vec_q*_...` and `quantize_q8_1_x4` one-off residual dispatches; those
should not be used to tune the dense p512 prompt schedule.

## Consequences For HRX v1

The current HRX v1 gfx1151 prompt work already tried several tile-local
pivots. The Vulkan matrix narrows the next useful axes:

- `BM128/BN128/WG256/WAVE64` is the baseline family, not the whole solution.
  Current HRX Q4 already matches that family and still has a gap.
- The shared schedule target spends more register and LDS budget than current
  HRX Q4 MMQL: RADV is at `VGPR=192` and `LDS=22528` without spilling, while
  the current HRX Q4 MMQL route is `VGPR=139` and `LDS=8192`.
- The Q8_0 Vulkan row argues against stopping at the accepted HRX BM64/BN64
  route. Vulkan's Q8_0 production prompt path is the same 128x128 large aligned
  family as K-quants and still has no spills.
- The Q8_0 large aligned pipeline is not a scalar integer-dot clone. Its
  SPIR-V declares `OpCapability CooperativeMatrixKHR` and uses
  `OpCooperativeMatrixLoadKHR`, `OpCooperativeMatrixMulAddKHR`, and
  `OpCooperativeMatrixStoreKHR`; RADV lowers the hot loop to
  `v_wmma_f16_16x16x16_f16`. The first HRX scalar-dot 128x128 probe
  (`hrx_mul_mat_vec_q8_0_q8_1_x4_mmql128x128_wg256_f32`) compiled through
  CMake/Ninja but spilled badly (`vgpr_count=192`,
  `vgpr_spill_count=472`, `private_segment_fixed_size=1892`) and is recorded
  as rejected. Treat further scalar-dot Q8 reshuffles as diagnostics, not the
  primary parity path.
- The workspace ROCm `rocm-head` compiler accepts gfx1151 HIP C++ WMMA
  builtins directly: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` and
  `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` both emit `v_wmma_*` in a
  scratch gfx1151 offload probe with no spills. This makes a HIP C++ WMMA
  candidate viable without adding ROCWMMA as a dependency.
- Q5_K and Q6_K need schedule-family treatment, not selector-only promotion.
  The Q5/Q6 Vulkan rows reuse the same large aligned tuple and should share as
  much HIP C++ schedule structure as the quant layout allows.
- Odd and tail policy must stay explicit. For Llama 3.1 8B Q4_K_M, p33 uses
  Vulkan's medium aligned family
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`,
  `LDS=11264`, `VGPR=144`, no spills, while p513 returns to the large aligned
  family and covers the extra token with a fifth workgroup column. This matches
  the prior HRX observation that p33 and production-width prompt can prefer
  different routes.
- Llama 3.1 8B Q8_0 now has the same odd/tail oracle coverage. p33 uses
  `matmul_q8_0_f32_f16acc_aligned_m` with
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`,
  `LDS=11264`, `VGPR=144`, no spills. p513 uses
  `matmul_q8_0_f32_f16acc_aligned_l` with the large
  `spec=[256,128,128,32,64,64,2,16,16,16,64]`, `wg_denoms=[128,128,1]`,
  fifth column workgroup, `LDS=22528`, `VGPR=192`, no spills. Artifacts:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-continued/`
  and
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-continued/`.
- Qwen2.5 Coder 7B Q5_K_M now has the same odd/tail oracle coverage. p33 uses
  `matmul_q5_k_f32_f16acc_aligned_m` with
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`, `wg_denoms=[64,64,1]`,
  `LDS=11264`, `VGPR=144`, no spills, and the same cooperative-matrix lowering
  pattern as the large Q5 route. p513 uses
  `matmul_q5_k_f32_f16acc_aligned_l` with the large
  `spec=[256,128,128,32,64,64,2,16,16,16,64]`, `wg_denoms=[128,128,1]`,
  fifth-column workgroups, `LDS=22528`, `VGPR=192`, no spills, and
  `split_k_reduce` dispatches for tail reductions. Artifacts:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p33-fa1-20260618-063510/`
  and
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen25coder-7b-q5km-p513-fa1-20260618-063522/`.

## Next HIP Candidate Gate

The next dense prompt candidate should define one of these axes before source
changes:

- deeper staged K-step, aiming toward the RADV 22528-byte LDS footprint while
  preserving no-spill codegen;
- packed RHS ownership and vector-load layout, especially where HRX currently
  depends on an explicit Q8_1 x4 quantize route;
- direct f32 RHS consumption versus explicit packed-Q8_1 consumption for the
  p512 prompt regime;
- wait/barrier and LDS load issue order around the integer dot loop;
- Q8_0 WMMA spelling that first proves lane/output mapping on a focused
  fixture, then scales toward the Vulkan 128x128 large aligned family.

Required gate before promotion:

- focused CPU-reference rows for p33, p512, and p513;
- route trace proving the intended provider selected;
- HSACO metadata and disassembly showing VGPR/SGPR/LDS/spills;
- same-runner model A/B versus current HRX and same-source Vulkan;
- rejection or promotion entry in `ggml/src/ggml-hrx/catalog/tuning/gfx1151/`.

Do not promote from a p512-only win; the prior HRX work already showed odd and
narrow prefill routes can prefer different schedules.
