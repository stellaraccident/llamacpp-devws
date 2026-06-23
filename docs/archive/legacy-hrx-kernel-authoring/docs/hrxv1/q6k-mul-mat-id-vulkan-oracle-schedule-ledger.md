# Q6_K MUL_MAT_ID Vulkan Oracle Schedule Ledger

Date: 2026-06-20

This ledger isolates the Qwen3 Q6_K MoE `MUL_MAT_ID` route from dense Q6
prompt matmul. Dense Q6 and grouped-ID Q6 both matter for Vulkan parity, but
they are different schedule families and should not share candidate gates.

## Evidence Sources

- Vulkan p33 oracle:
  `cache/hrxv1/gfx1151/vulkan-oracle-qwen3-30b-q6k-p33-fa1-20260618-061613/`.
- RADV/HIP ISA comparison artifact:
  `cache/hrxv1/gfx1151/q6-id-radv-hip-schedule-ledger-c012bdbf0-20260620-221909/`.
- Current-head focused p33 HRX gate:
  `cache/hrxv1/gfx1151/q6-id-current-head-p33-focused-c012bdbf0-20260620-222329/`.
- Promoted grouped-ID route evidence:
  `cache/hrxv1/gfx1151/q6-id-threshold32-default-regate-20260618-201739/`,
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold32-default-20260618-201756/`,
  and
  `cache/hrxv1/gfx1151/qwen3-q6-id-threshold32-rollback-20260618-201820/`.
- Rejected direct-F32 WMMA diagnostic:
  `cache/hrxv1/gfx1151/q6-id-wmma16-direct-focused-20260619-003510/`
  and
  `cache/hrxv1/gfx1151/q6-id-wmma16-direct-perf-20260619-003759/`.

## Vulkan Oracle Shape

The p33 Qwen3 Q6_K Vulkan capture has `141` dispatches of
`matmul_id_subgroup_q6_k_f32_f16acc_aligned_m` and `141` matching
`count_experts` dispatches.

Normalized dispatch groups:

| Count | Node family | Dst | Src0 | Src1 | Workgroups |
| ---: | --- | --- | --- | --- | --- |
| 94 | gate/up | `[768,8,33,1]` | q6_K `[2048,768,128,1]` | f32 `[2048,1,33,1]` | `[12,1,128]` |
| 47 | down | `[2048,8,33,1]` | q6_K `[768,2048,128,1]` | f32 `[768,8,33,1]` | `[32,1,128]` |

Pipeline identity:

- pipeline: `matmul_id_subgroup_q6_k_f32_f16acc_aligned_m`;
- SPIR-V hash: `0x1f7c55892b74932b`;
- spec: `[128,64,64,32,64,32,2,16,16,16,64]`;
- workgroup denominators: `[64,64,1]`;
- push constants: `56` bytes;
- parameters: `5`;
- full subgroups required;
- robustness disabled: `false`.

RADV stats for the p33 ID route:

| Fact | RADV value |
| --- | ---: |
| SGPR | 108 |
| VGPR | 144 |
| LDS | 12288 bytes |
| SGPR spills | 0 |
| VGPR spills | 0 |
| Scratch | 0 |
| Subgroups per SIMD | 10 |
| Instructions | 2972 |
| VALU | 1503 |
| VMEM | 100 |
| SMEM | 67 |

Opcode surface from the split RADV ISA:

- `16` visible `v_wmma_f16_16x16x16_f16`;
- `52` `ds_load_b64`;
- `32` `ds_load_u16_d16`;
- `32` `ds_store_b16`;
- `2` `ds_store_b128`;
- `69` buffer loads;
- `32` `buffer_store_b32`;
- `24` barriers;
- `244` `s_waitcnt`;
- `40` `s_waitcnt_depctr`.

This is a staged wave64 f16-WMMA subgroup route with per-expert workgroups on
Z and one token-group workgroup in Y for p33.

## Current HRX Route

Current default provider:

```text
hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32
```

Source:

```text
sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_id_q6_k_q8_1_x4_mmq.hip.cpp
```

Selector policy:

- default on gfx1151 unless
  `GGML_HRX_DISABLE_Q6_K_ID_Q8_1_X4_MMQ16_PROMPT=1`;
- `q6_K`, `k % 256 == 0`, `rows % 64 == 0`, `n_ids == 8`,
  `n_tokens >= 32`;
- launches `[ceil(rows/64), ceil(n_tokens/16), n_experts]`.

Current p33 focused gate at `sources/llama.cpp` commit `c012bdbf0`:

| Row | Correct | Time |
| --- | --- | ---: |
| `ffn_moe_gate-0` | yes | `306.466 us` |
| `ffn_moe_down-0` | yes | `277.896 us` |

Route trace:

- gate/up: `wg_count=[12,3,128]`;
- down: `wg_count=[32,3,128]`;
- both rows selected
  `hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32`;
- no fallback or unsupported rows.

HSACO static facts:

| Fact | HRX grouped Q8_1/x4 |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 68 |
| VGPR | 119 |
| LDS | 3200 bytes |
| Private segment | 0 |
| VGPR spills | 0 |
| WMMA | 0 |
| `v_dot4_i32_iu8` | 128 |
| `ds_load_b128` | 4 |
| global loads | 35 |
| global stores | 16 |
| barriers | 2 |
| `s_waitcnt` | 163 |
| `s_waitcnt_depctr` | 92 |

This route is a packed Q8_1/x4 integer-dot schedule. It is the right current
default because it massively beats the direct-F32 bridge, but it is not a
mechanical clone of RADV's f16-WMMA subgroup schedule.

## Rejected Direct-F32 WMMA Bridge

Rejected provider:

```text
hrx_mul_mat_id_q6_k_wmma16x16_direct_f16acc_wg32_f32
```

Source:

```text
sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_id_q6_k_wmma16_direct.hip.cpp
```

Static facts:

| Fact | Direct-F32 bridge |
| --- | ---: |
| Wavefront | 32 |
| SGPR | 54 |
| VGPR | 52 |
| LDS | 0 bytes |
| Private segment | 0 |
| VGPR spills | 0 |
| WMMA | 1 |
| global loads | 70 |
| global stores | 8 |
| barriers | 0 |

Focused timing versus the grouped Q8_1/x4 default:

| Size | Row | Default grouped | Direct-F32 WMMA | Ratio |
| --- | --- | ---: | ---: | ---: |
| p33 | `ffn_moe_gate-0` | `300.02 us` | `2347.06 us` | `7.82x` slower |
| p33 | `ffn_moe_down-0` | `262.13 us` | `2153.97 us` | `8.22x` slower |
| p512 | `ffn_moe_gate-0` | `2399.79 us` | `15890.97 us` | `6.62x` slower |
| p512 | `ffn_moe_down-0` | `2319.38 us` | `19716.80 us` | `8.50x` slower |
| p513 | `ffn_moe_gate-0` | `2495.16 us` | `16175.99 us` | `6.48x` slower |
| p513 | `ffn_moe_down-0` | `2363.23 us` | `20285.69 us` | `8.58x` slower |

Conclusion: raw F32 RHS semantics and grouped direct-WMMA indexing are correct,
but direct-F32 wrapper variants are not a plausible parity path.

## Rejected MMQ64x64 Token-Grouping Probe

Rejected provider:

```text
hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x64_wg64_f32
```

Artifact:

```text
cache/hrxv1/gfx1151/q6-id-mmq64x64-focused-c012bdbf0-dirty-20260620-223040/
```

This probe kept the current packed Q8_1/x4 integer-dot schedule but widened
the route tile from `16` to `64` tokens. It directly tested whether matching
Vulkan's p33 one-Y-group launch was enough:

- default p33 route: `route_tile_n=16`, workgroups `[12,3,128]` and
  `[32,3,128]`;
- opt-in route: `route_tile_n=64`, workgroups `[12,1,128]` and `[32,1,128]`;
- focused CPU-reference passed for `ffn_moe_gate-0` and `ffn_moe_down-0`;
- static metadata: wave64, SGPR `72`, VGPR `163`, LDS `5120`, no spills;
- symbol-sliced ISA: `512` `v_dot4`, `114` LDS-load-class ops,
  `28` LDS-store-class ops, `64` global-store-class ops, and `2` barriers.

Focused timing rejected promotion:

| Row | Default MMQ64x16 | Opt-in MMQ64x64 | Result |
| --- | ---: | ---: | --- |
| `ffn_moe_gate-0` | `311.808 us` | `377.120 us` | slower |
| `ffn_moe_down-0` | `282.727 us` | `224.608 us` | faster |
| two-row sum | `594.534 us` | `601.728 us` | slightly slower |

The model has two gate/up-shaped rows for each down row, so the likely p33
model mix is worse than the two-row sum. This closes the simple token-grouping
axis: matching Vulkan's Y-grid inside the current packed-dot schedule is not
enough.

## Rejected Staged VK64 Direct-F32 WMMA Probe

Rejected provider:

```text
hrx_mul_mat_id_q6_k_wmma16x16_staged_vk64_f16acc_wg256_f32
```

Artifact:

```text
cache/hrxv1/gfx1151/q6-id-wmma16-staged-vk64-focused-c95718b4e-dirty-20260620-224319/
```

This probe tested the next cleaner RADV-style bridge after the direct WG32
F32-WMMA route and the packed MMQ64x64 token-grouping route:

- direct F32 RHS, no Q8_1/x4 packing;
- wave64, WG256, BM64/BN64/BK32;
- `route_tile_n=64`, giving p33 workgroups `[12,1,128]` and `[32,1,128]`;
- 48-half shared stride and `12288` bytes LDS, matching the Vulkan ID route's
  LDS footprint;
- p33 and p513 focused CPU-reference passed for `ffn_moe_gate-0` and
  `ffn_moe_down-0`.

Static facts:

| Fact | Staged VK64 |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 67 |
| VGPR | 113 |
| LDS | 12288 bytes |
| Private segment | 0 |
| VGPR spills | 0 |
| Static WMMA | 8 |
| LDS read-class ops | 16 |
| LDS write-class ops | 2 |
| Global/buffer loads | 16 |
| Global/buffer stores | 16 |
| Barriers | 2 |
| `s_waitcnt` | 66 |

Focused timing rejected promotion:

| Size | Row | Default grouped | Staged VK64 | Result |
| --- | --- | ---: | ---: | --- |
| p33 | `ffn_moe_gate-0` | `290.697 us` | `342.517 us` | slower |
| p33 | `ffn_moe_down-0` | `261.210 us` | `359.243 us` | slower |
| p513 | `ffn_moe_gate-0` | `2474.333 us` | `2901.407 us` | slower |
| p513 | `ffn_moe_down-0` | `2331.777 us` | `3232.209 us` | slower |

Conclusion: matching the Vulkan p33 Y-grid and LDS footprint with a staged
direct-F32 WMMA route is still insufficient. The useful remaining target is the
actual RADV subgroup-ID dataflow: 16 static WMMA, 52 `ds_load_b64`, 32
`ds_load_u16_d16`, 32 `ds_store_b16`, 2 `ds_store_b128`, 32 global stores, and
the associated lane ownership/wait contract. A packed-route deviation is still
allowed only if it improves gate/up and down together.

## Schedule Delta

| Axis | Vulkan RADV ID | Current HRX grouped ID | Direct-F32 bridge |
| --- | --- | --- | --- |
| Math primitive | f16 WMMA | packed integer dot | f16 WMMA |
| Wave | subgroup/wave64 route | wave64 | wave32 |
| Workgroups p33 gate/up | `[12,1,128]` | `[12,3,128]` | `[48,3,128]` |
| Workgroups p33 down | `[32,1,128]` | `[32,3,128]` | not production |
| LDS | 12288 bytes | 3200 bytes | 0 |
| WMMA/dot surface | 16 WMMA | 128 dot4 | 1 WMMA |
| LDS loads | 52 b64 + 32 u16_d16 | 4 b128 plus scalar/b32 | none |
| LDS stores | 32 b16 + 2 b128 | 24 b32 + 1 b64 + 1 2addr_b64 | none |
| Global stores | 32 b32 | 16 b32 | 8 |
| Barriers | 24 | 2 | 0 |
| Token grouping | one Y group for p33 | three Y groups of 16 tokens | three Y groups |
| Status | Vulkan oracle | accepted default | rejected diagnostic |

The important mismatch is not simply "WMMA versus dot." RADV is doing a
subgroup-ID schedule with a different token grouping and a much larger staged
LDS surface. Current HRX wins over direct-F32 by preserving packed Q8_1/x4 RHS
reuse, but it gives up the RADV-style f16-WMMA staged path.

## Candidate Gate For The Next Q6 ID Route

Any new route in this family is exploratory until every line here is filled and
the evidence is captured.

- Production target:
  Qwen3 30B Q6_K `MUL_MAT_ID` prompt, p33 first, then p512 and p513; focused
  rows `ffn_moe_gate-0` and `ffn_moe_down-0`.
- Baseline command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 build/hrx-v1-catalog-gfx1151/bin/test-backend-ops perf -b HRX0 -o MUL_MAT_ID --test-file cache/hrxv1/gfx1151/q6-id-odd-tail-focused-20260617-183706/p33/focused/moe_qk_prompt.txt --output csv`.
- Variant command:
  same focused command with the new provider opt-in env var set, plus
  `GGML_HRX_EXPECT_MUL_MAT_ID_Q6_PROVIDER=<new-provider>`.
- Same-runner comparison method:
  one backend-op perf run at a time, same binary, same focused rows, default
  route and opt-in route compared from the same artifact; repeat before
  promotion if the delta is small.
- Route trace path:
  artifact must contain test and perf `stderr.log` proving provider selection
  and no fallback.
- Scheduler/profile trace path:
  static HSACO disassembly and metadata must be captured next to the focused
  timing artifact. If model A/B is reached, capture model route traces too.
- Focused CPU-reference command:
  `GGML_HRX_TRACE_ROUTES=1 GGML_HRX_TRACE_PROVIDERS=1 GGML_HRX_EXPECT_MUL_MAT_ID_Q6_PROVIDER=<new-provider> build/hrx-v1-catalog-gfx1151/bin/test-backend-ops test -b HRX0 -o MUL_MAT_ID --test-file <p33|p512|p513 moe_qk_prompt.txt> --output csv`.
- Compile report path:
  generated HSACO under
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/`;
  disassemble with the same ROCm install used for build.
- Target listing path:
  `ninja -C build/hrx-v1-catalog-gfx1151 -t targets` must show the HSACO
  target; the route must be built by CMake/Ninja, not by an assembler helper.
- Prior-art schedule source:
  `matmul_id_subgroup_q6_k_f32_f16acc_aligned_m` from the p33 Vulkan oracle,
  plus the current grouped Q8_1/x4 HRX route as the throughput floor.
- Promotion rule:
  p33 focused CPU-reference passes, p512 and p513 focused CPU-reference pass
  without stealing unrelated routes, route trace proves the new provider
  selected only under its intended gate, static facts move toward the RADV ID
  family rather than the rejected direct-F32 bridge, and focused timing beats
  current grouped Q8_1/x4 by a material margin before any model-level A/B.

## Next Implementation Hypothesis

Do not implement another direct-F32 wrapper. The next useful route needs to
choose one of two explicit strategies:

1. A true RADV-style subgroup ID clone:
   wave64, p33 workgroups `[12 or 32,1,128]`, staged Q6/F32 data through
   roughly 12 KiB LDS, 16 visible WMMA sites, and RADV-like b16/b128 LDS
   store/load ownership. This must preserve the expert count/route ABI.
2. A packed-Q8_1/x4 route that intentionally deviates from RADV but attacks the
   current HRX-specific gap:
   keep Q8_1/x4 reuse, reduce the p33 Y grid from 3 token groups toward the
   Vulkan one-group shape, and increase useful per-workgroup output ownership
   without reintroducing the direct-F32 global-load cost.

Strategy 1 is the cleaner oracle clone. Strategy 2 is acceptable only if the
candidate ledger names the precise deviation from RADV and explains why it
should beat the current accepted packed route.

## Low-Level Subgroup-Contract Fixture

The CMake/Ninja-built fixture
`hrx-hip-bench-q6-id-subgroup-contract` now isolates the source-visible
subgroup-ID store/load surface without changing production routes.

Artifact:
`cache/hrxv1/gfx1151/q6-id-subgroup-contract-bench-a93bb23ff-dirty-20260620-225935/`.

The staged fixture row validates and emits:

| Fact | Fixture staged motif |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 14 |
| VGPR | 50 |
| LDS | 12288 bytes |
| Private segment | 0 |
| Static WMMA | 16 |
| `ds_load_b64` | 16 |
| `ds_load_u16_d16` | 32 |
| `ds_store_b16` | 32 |
| `ds_store_b128` | 8 |
| `buffer_store_b32` | 32 |
| Barriers | 2 |
| `s_waitcnt` | 39 |

Same-executable timing over 2000 reps was:

| Row | Valid | Time |
| --- | ---: | ---: |
| direct writeback | 1 | 2.228195 us |
| staged halfword motif | 1 | 3.366400 us |

Interpretation:
the fixture proves HIP C++ can preserve the 12 KiB LDS footprint, 16-WMMA
surface, 32 halfword stores, 32 halfword loadbacks, and 32 global stores in a
small valid wave64 motif. It also shows this source-visible halfword writeback
surface is not enough: the motif is slower than direct writeback in isolation
and still lacks RADV ID's operand-load depth (`52 ds_load_b64`) and `2`
`ds_store_b128` surface. Use this fixture as the next route's low-level
contract gate, but do not promote a route from this evidence alone.

Follow-up artifact:
`cache/hrxv1/gfx1151/q6-id-subgroup-contract-loaddeep-2f1322537-dirty-20260620-230734/`.

The added `loaddeep` fixture row preserves the same output contract but sinks
thirteen 4x64-bit LDS fragment loads before the WMMA sequence. It validates and
emits:

| Fact | Fixture loaddeep motif |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 14 |
| VGPR | 61 |
| LDS | 12288 bytes |
| Private segment | 0 |
| Static WMMA | 16 |
| `ds_load_b64` | 52 |
| `ds_load_u16_d16` | 32 |
| `ds_store_b16` | 32 |
| `ds_store_b128` | 8 |
| `buffer_store_b32` | 32 |
| Barriers | 2 |
| `s_waitcnt` | 39 |

Same-executable timing over 2000 reps:

| Row | Valid | Time |
| --- | ---: | ---: |
| direct writeback | 1 | 2.241520 us |
| staged halfword motif | 1 | 3.378919 us |
| loaddeep motif | 1 | 3.566414 us |

Interpretation:
matching RADV ID's `52 ds_load_b64` count is expressible in HIP C++ without
spills or extra LDS, but it is not sufficient and is slower in isolation. An
earlier variant that fed all thirteen loaded fragments into the WMMA path
produced NaNs even with full waits, so the true missing route property is the
lane/operand ownership and wait-overlap contract, not the load count by itself.

Second follow-up artifact:
`cache/hrxv1/gfx1151/q6-id-subgroup-contract-minstorepad-dcba2f11e-dirty-20260620-231235/`.

The `minstore` fixture row reuses one operand fragment bank for all thirteen
4x64-bit LDS loads and pads LDS allocation back to 12 KiB. It validates and
emits:

| Fact | Fixture minstore motif |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 12 |
| VGPR | 35 |
| LDS | 12288 bytes |
| Private segment | 0 |
| Static WMMA | 16 |
| `ds_load_b64` | 52 |
| `ds_load_u16_d16` | 32 |
| `ds_store_b16` | 32 |
| `ds_store_b128` | 2 |
| `buffer_store_b32` | 32 |
| Barriers | 2 |
| `s_waitcnt` | 39 |

Same-executable timing over 2000 reps:

| Row | Valid | Time |
| --- | ---: | ---: |
| direct writeback | 1 | 2.221468 us |
| staged halfword motif | 1 | 3.375103 us |
| loaddeep motif | 1 | 3.533869 us |
| minstore motif | 1 | 3.448519 us |

Interpretation:
HIP C++ can now express the RADV ID headline static surface in a valid fixture:
12 KiB LDS, 16 WMMA, 52 b64 LDS operand loads, 32 halfword stores/loadbacks,
2 vector LDS stores, and 32 global stores. This still is not a production
route because the one-bank ownership is synthetic and slower than the simpler
staged row. The next Q6 ID route must transfer the actual useful RADV
lane/operand ownership, not just the opcode counts.

## WMMA Operand Ownership Extract

Source tool:

```text
sources/llama.cpp/tools/vulkan-oracle/extract_wmma_ownership.py
```

Artifact:

```text
cache/hrxv1/gfx1151/q6-id-wmma-ownership-68bd4a40e-dirty-20260620-232047/
```

The extractor compares the RADV p33 Q6 ID AMDGCN against the CMake/Ninja-built
HIP minstore fixture symbol. It confirms the static-surface convergence and
exposes the remaining ownership mismatch.

Both paths emit:

- `16` `v_wmma_f16_16x16x16_f16`;
- `52` `ds_load_b64`.

RADV's first main WMMA tile uses eight A fragment banks, each twice:

```text
v[32:39], v[72:79], v[80:87], v[88:95],
v[96:103], v[120:127], v[128:135], v[136:143]
```

It uses four B fragment banks, each four times:

```text
v[40:47], v[64:71], v[104:111], v[112:119]
```

It accumulates into eight four-register destination banks, each twice:

```text
v[52:55], v[56:59], v[60:63], v[8:11],
v[24:27], v[12:15], v[16:19], v[28:31]
```

RADV's visible wait ladder before issue is:

```text
lgkmcnt(40), 36, 32, 28, 24, 16, 12, 8, 4, 0
```

The HIP minstore fixture instead feeds all 16 WMMAs from a single A/B bank
`v[4:11]`, uses four overlapping destination ranges, and only has two explicit
`lgkmcnt(0)` anchors. Therefore minstore is a useful proof that the opcode
surface can be expressed in HIP C++, but it is not a useful proof of the RADV
subgroup-ID dataflow.

Next route-facing hypothesis: build a banked Q6 ID clone that preserves RADV's
8xA/4xB/8xacc operand ownership and decreasing wait ladder before attempting
another selector or token-grouping change. A candidate that still collapses
the A and B operands into one reusable fragment bank should be rejected at the
fixture/static stage even if its opcode counts match RADV.

Follow-up artifact:

```text
cache/hrxv1/gfx1151/q6-id-subgroup-contract-banked-deps-4bbd6ce54-dirty-20260620-232846/
```

The added `banked` row tests the 8xA/4xB ownership axis directly in the
CMake/Ninja-built subgroup-contract fixture. It validates and emits:

| Fact | Fixture banked motif |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 14 |
| VGPR | 110 |
| LDS | 12288 bytes |
| Private segment | 0 |
| Static WMMA | 16 |
| `ds_load_b64` | 52 |
| A operand banks | 8 |
| B operand banks | 4 |
| Dst/C operand banks | 7 |
| `ds_load_u16_d16` | 32 |
| `ds_store_b16` | 32 |
| `ds_store_b128` | 24 |
| Global stores | 32 |
| Barriers | 2 |
| `s_waitcnt` | 46 |

Same-executable timing over 2000 reps:

| Row | Valid | Time |
| --- | ---: | ---: |
| direct writeback | 1 | `2.266737 us` |
| staged halfword motif | 1 | `3.361962 us` |
| loaddeep motif | 1 | `3.539264 us` |
| minstore motif | 1 | `3.454530 us` |
| banked motif | 1 | `3.766118 us` |

Interpretation:
the banked fixture closes one important uncertainty: HIP C++ can preserve the
RADV Q6 ID A/B operand-bank cardinality (`8` A banks, `4` B banks) in a valid
wave64 fixture with no spills. It also exposes the next blockers before a
route-facing clone is worthwhile:

- HIP still uses wide/overlapping half-vector accumulator ranges instead of
  RADV's eight four-register dst/C banks;
- the explicit dependency waits do not preserve the exact RADV visible ladder,
  emitting `[0,36,32,28,24,16,8,4]` for the WMMA sequence instead of
  `[40,36,32,28,24,16,12,8,4,0]`;
- the store surface inflates to `24 ds_store_b128` versus RADV's `2`.

Therefore the next production candidate should not merely port this fixture
into `MUL_MAT_ID`. It should preserve the banked A/B ownership while first
solving accumulator packing/wait ordering/store inflation, or intentionally
deviate back to the packed Q8_1/x4 family with focused wins on both gate/up and
down rows.

Follow-up artifact:

```text
cache/hrxv1/gfx1151/q6-id-subgroup-contract-bankedcompact-0a468b9fd-dirty-20260620-233607/
```

The added `bankedcompact` row keeps the same 12-fragment A/B ownership pattern
as `banked` but replaces the source-level half-vector accumulator with a
packed `uint32_t x4` inline-asm accumulator operand. It validates and emits:

| Fact | Fixture bankedcompact motif |
| --- | ---: |
| Wavefront | 64 |
| SGPR | 14 |
| VGPR | 110 |
| LDS | 28672 bytes |
| Private segment | 0 |
| Static WMMA | 16 |
| `ds_load_b64` | 52 |
| A operand banks | 8 |
| B operand banks | 4 |
| Dst/C operand banks | 5 |
| `ds_load_u16_d16` | 32 |
| `ds_store_b16` | 32 |
| `ds_store_b128` | 24 |
| Global stores | 32 |
| Barriers | 2 |
| `s_waitcnt` | 46 |
| `s_waitcnt_depctr` | 2 |

Same-executable timing over 2000 reps:

| Row | Valid | Time |
| --- | ---: | ---: |
| direct writeback | 1 | `2.263697 us` |
| staged halfword motif | 1 | `3.363540 us` |
| loaddeep motif | 1 | `3.539424 us` |
| minstore motif | 1 | `3.449291 us` |
| banked motif | 1 | `3.764099 us` |
| bankedcompact motif | 1 | `3.774900 us` |

Interpretation:
the packed accumulator spelling closes the wait-placement uncertainty: the
extracted WMMA sequence now has the full RADV visible ladder
`[40,36,32,28,24,16,12,8,4,0]` while preserving the `8` A banks and `4` B
banks. It does not solve accumulator ownership. HIP still lowers the WMMA
destination/C operand as wide eight-register ranges and overlaps them into
only five extracted banks rather than RADV's eight compact four-register
banks. It also keeps the inflated `24 ds_store_b128` initialization surface.

Therefore the next useful Q6 ID route-facing attempt should not reuse the
`uint32_t x4` accumulator spelling as-is. The remaining mechanical target is
an accumulator/store primitive that preserves banked A/B ownership and the
full wait ladder while producing RADV-like compact dst/C ownership and the
two-`ds_store_b128` setup surface, or a documented pivot back to packed
Q8_1/x4 if focused p33/p512/p513 gate/up and down rows show a better route.

Primitive follow-up artifact:

```text
cache/hrxv1/gfx1151/wmma-f16-tied-lane-map-ownership-0eba9b41c-dirty-20260620-234412/
```

The existing `wmma_f16_lane_map` fixture was extended with `--mode=tied-basic`
to test Clang's tied f16 WMMA builtin:

```text
__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64
```

The tied form validates the same logical lane map as the ordinary builtin:
op_sel 0 updates the even/low slots and op_sel 1 updates the odd/high slots.
It does not change the emitted operand form in the simple probe:

```text
v_wmma_f16_16x16x16_f16 v[9:16], v[1:8], v[1:8], v[9:16]
```

Decision:
do not expect the tied builtin alone to recover RADV's compact dst/C examples
such as `v[52:55]`. It may still matter semantically when preserving the
non-selected accumulator half, but it is not the missing Q6 ID ownership
primitive. The next useful low-level test needs either a different LLVM/MLIR
intrinsic type path or a lower-level assembly/codegen route that can actually
emit compact dst/C ownership in a CMake-built fixture.

The generated Clang builtin table also lists a compact-looking gfx12 form:

```text
__builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12
```

with `V4xV4xV4xV4x` operand typing, but a transient CMake/Ninja build probe
on this `gfx1151` target rejected it with:

```text
needs target feature wmma-128b-insts,wavefrontsize64
```

So the obvious compact builtin is not currently available to this HRX v1
gfx1151 HIP C++ build. Treat that as a primitive availability fact, not a route
performance result.

Extractor follow-up artifact:

```text
cache/hrxv1/gfx1151/q6-id-bankedcompact-width-analysis-67da9c87c-dirty-20260620-234925/
```

The WMMA ownership extractor now reports operand-width histograms. This makes
the remaining primitive mismatch explicit:

| Operand | RADV width4 | RADV width8 | HIP bankedcompact width4 | HIP bankedcompact width8 |
| --- | ---: | ---: | ---: | ---: |
| dst | 16 | 0 | 0 | 16 |
| A | 0 | 16 | 0 | 16 |
| B | 0 | 16 | 0 | 16 |
| C | 16 | 0 | 0 | 16 |

Interpretation:
HIP `bankedcompact` already matches RADV on A/B operand width, A/B bank
cardinality, WMMA count, `ds_load_b64` count, and wait ladder. The hard
remaining RADV-style f16-WMMA primitive mismatch is dst/C width: RADV emits
compact width-4 accumulator operands for all 16 WMMAs, while the tested HIP
C++ builtin and inline-asm paths emit width-8 dst/C operands for all 16 WMMAs.

Future RADV-style Q6 ID candidates should use dst/C width4 as a static screen.
If a candidate still emits width8 dst/C, it needs a focused performance reason
to proceed as an intentional deviation, not as an oracle clone.

## Compact Accumulator Static Gate

Source commit:

```text
bfa902fec hrx: gate compact wmma accumulators
```

Artifact:

```text
cache/hrxv1/gfx1151/q6-id-compact-accumulator-screen-577ddc0f5-dirty-20260620-235650/
```

The WMMA ownership extractor now has a hard screen:

```bash
python3 tools/vulkan-oracle/extract_wmma_ownership.py \
  --isa <amdgcn-or-objdump.txt> \
  --symbol <optional-symbol> \
  --require-compact-f16-accumulators
```

This exits non-zero unless every f16 WMMA in the summarized ISA uses width-4
`dst` and `C` operands. It is not a throughput gate; it is a primitive
availability gate for RADV-style Q6 ID clone attempts.

Control results:

| ISA | WMMA | `ds_load_b64` | Wait ladder | `dst` width | `C` width | Gate |
| --- | ---: | ---: | --- | --- | --- | --- |
| RADV Q6 ID oracle | 16 | 52 | `[40,36,32,28,24,16,12,8,4,0]` | 16x width4 | 16x width4 | pass |
| HIP `bankedcompact` fixture | 16 | 52 | `[40,36,32,28,24,16,12,8,4,0]` | 16x width8 | 16x width8 | fail |

Decision:
the current HIP `bankedcompact` fixture remains a useful load/wait/A-B
ownership control, but it is no longer a plausible RADV clone candidate by
itself. Future f16-WMMA Q6 ID work should first pass this static gate, or be
explicitly labeled as a packed-route deviation that will be judged by focused
p33/p512/p513 backend-op timing against the grouped Q8_1/x4 floor.

## Inline ASM Compact Accumulator Probe

Artifact:

```text
cache/hrxv1/gfx1151/wmma-f16-inline-compact-acc-7172c1b12-dirty-20260621-004515/
```

Follow-up tested the obvious HIP C++ inline-asm operand-constraint axis in the
shared `hrx-hip-bench-wmma-f16-lane-map` CMake target. Three
`v_wmma_f16_16x16x16_f16` spellings were checked:

| Variant | Constraint shape | `dst` width | `C` width | Gate |
| --- | --- | --- | --- | --- |
| 0 | `+v(out)` tied read/write | 1x width8 | 1x width8 | fail |
| 1 | `=v(out)` plus separate `v(acc)` C | 1x width8 | 1x width8 | fail |
| 2 | `=v(out)` plus matching `0(acc)` C | 1x width8 | 1x width8 | fail |

Runtime smoke produced nonzero output, so this is not a dead-code artifact.
The emitted ISA still uses width-8 dst/C accumulator operands in every variant.
This closes the simple inline-asm constraint spelling route for recovering the
RADV compact accumulator primitive from HIP C++ on gfx1151. The next compact
attempt needs a different lowering/API path, not another route-facing kernel
using the same width-8 primitive.

A direct LLVM MC screen strengthens that conclusion:

```text
cache/hrxv1/gfx1151/wmma-f16-llvm-mc-compact-form-3ab4a1c89-20260621-004934/
```

`llvm-mc` accepts the normal gfx1151 width8 form:

```text
v_wmma_f16_16x16x16_f16 v[0:7], v[0:7], v[0:7], v[0:7]
```

It rejects the width4 form:

```text
v_wmma_f16_16x16x16_f16 v[0:3], v[0:3], v[0:3], v[0:3]
```

with `operands are not valid for this GPU or mode`, and still rejects it with
`-mattr=+wmma-128b-insts,+wavefrontsize64`. Treat compact f16 WMMA ownership as
unavailable through the straightforward ROCm LLVM gfx1151 assembler path unless
the target description/compiler is changed or a non-LLVM code-object path is
introduced.

## Decision

Q6 `MUL_MAT_ID` remains open for parity work. The current grouped Q8_1/x4 route
is the accepted floor and should stay default. The direct-F32 WMMA bridge is
rejected. The next code change should be a route-gated, CMake/Ninja-built
candidate that mechanically targets the RADV ID schedule delta above, not a
selector threshold change or a dense-Q6 waitcnt/store experiment.
