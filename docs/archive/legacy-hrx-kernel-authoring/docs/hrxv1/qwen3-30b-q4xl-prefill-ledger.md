# Qwen3 30B Q4_K_XL gfx1151 Prefill Ledger

Date: 2026-06-17

This ledger tracks the first HRX v1 HIP C++ prefill boulders for
`Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf` on `gfx1151`.

It is not a promotion record. It is the staging area for prior-driven schedule
work before adding or defaulting new HIP C++ routes.

## Current Evidence

Artifact roots:

```text
cache/hrxv1/gfx1151/prefill-matrix-qwen3-30b-q4xl-20260617-135156/
cache/hrxv1/gfx1151/prefill-tail-matrix-qwen3-30b-q4xl-20260617-135329/
cache/hrxv1/gfx1151/hrx-profile-dispatch-qwen3-30b-q4xl-p512n0-20260617-135530/
cache/hrxv1/gfx1151/p31-q8-1-mmvq-forced-r3-20260617-140229/
```

Model and mode:

```text
model: shared/models/llamacpp-hrx2-basket-v1/unsloth__Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf
mode: prefill, n_gen=0, flash_attn=0
```

Single-repetition same-source HRX/Vulkan rows:

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

Interpretation:

- The original single-sample `p31` row looked like an odd-size route cliff, but
  a repeated control supersedes that interpretation for the Q8_1 guard:
  unforced HRX `p31` was stable at `94.48 tok/s` average, while forcing packed
  Q8_1 with `GGML_HRX_Q8_1_MMVQ=all` selected packed routes and regressed to
  `63.51 tok/s`. Vulkan `p31` averaged `260.58 tok/s`, so the narrow-row gap
  remains, but blindly lowering the packed-Q8_1 auto gate is rejected for this
  row.
- `p512/p513` are production-regime gaps; HRX remains about `0.12x-0.15x`
  Vulkan there.
- `p513 ub512` doubles the HRX route count due to residual graph behavior, but
  `p513 ub1024` is still far below Vulkan, so the production gap is not only
  residual-graph overhead.

## Current HRX Route Facts

For `p512 ub512`, HRX route histogram:

```text
ops:
  337 MUL_MAT
   49 GET_ROWS

providers:
  118 hrx_mul_mat_vec_q4_k_f32
   96 hrx_mul_mat_vec_f16_batched_cols16_f32
   47 hrx_mul_mat_vec_f32_batched_rows2_cols8_f32
   47 hrx_get_rows_f32
   42 hrx_mul_mat_vec_q4_k_q8_1_f32
   12 hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32
   10 hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32
   10 hrx_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_f32
    2 hrx_get_rows_f32_nr1_x4
    1 hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32
    1 hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32
```

For the original `p31 ub512` HRX route histogram, auto policy selected no
Q4/Q5/Q6 packed Q8_1 prompt routes and instead selected:

```text
160 hrx_mul_mat_vec_q4_k_f32
 22 hrx_mul_mat_vec_q5_k_rows2_cols8_wg64_f32
 10 hrx_mul_mat_vec_q6_k_rows2_cols8_wg32_f32
```

For the repeated forced-control artifact, `GGML_HRX_Q8_1_MMVQ=all` selected:

```text
160 hrx_mul_mat_vec_q4_k_q8_1_f32
 22 hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32
 11 hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32
```

That forced route set was slower than the default route set at `p31`. The next
narrow-prefill investigation should look for a different narrow schedule or
runtime/device-time evidence, not a broad relaxation of
`GGML_HRX_Q8_1_MMVQ_AUTO_COLS_MIN`.

## Focused Schedule Evidence

Model-derived p512 op shapes were exported with `export-graph-ops` and filtered
with `tools/hrxv1_focus_exported_ops.py`.

Artifacts:

```text
cache/hrxv1/gfx1151/model-op-shapes-qwen3-30b-q4xl-p512-20260617-140526/
cache/hrxv1/gfx1151/focused-qk-prompt-opgate-20260617-140712/
cache/hrxv1/gfx1151/focused-qk-prompt-perf-timing-20260617-141110/
cache/hrxv1/gfx1151/focused-qk-prompt-q8all-20260617-141500/
cache/hrxv1/gfx1151/focused-qk-prompt-q4x4mmq64-opgate-20260617-142923/
cache/hrxv1/gfx1151/focused-qk-prompt-q4x4mmq64-perf-20260617-142946/
cache/hrxv1/gfx1151/q4x4mmq64-model-ab-p512-20260617-143156/
cache/hrxv1/gfx1151/q4x4mmq64-model-ab-odd-20260617-143237/
cache/hrxv1/gfx1151/focused-q6-prompt-mmql-vs-mmq64-20260617-145055/
cache/hrxv1/gfx1151/focused-qk-prompt-narrow-threshold-q4q6mmq64-20260617-145835/
cache/hrxv1/gfx1151/focused-qk-prompt-q4q6mmq64-threshold-final-opgate-20260617-150129/
cache/hrxv1/gfx1151/q4q6-mmq64-model-ab-20260617-145232/
```

The focused Q4/Q5/Q6 prompt set contains 8 exact p512 `MUL_MAT` rows:

```text
Vcur-0        q4_K k=2048 rows=512    cols=512
Vcur-1        q5_K k=2048 rows=512    cols=512
node_32       q4_K k=4096 rows=2048   cols=512
node_100      q5_K k=4096 rows=2048   cols=512
node_372      q6_K k=4096 rows=2048   cols=512
Qcur-0        q4_K k=2048 rows=4096   cols=512
Qcur-1        q5_K k=2048 rows=4096   cols=512
result_output q6_K k=2048 rows=151936 cols=512
```

CPU-reference focused correctness passed for the default route set and for
`GGML_HRX_Q8_1_MMVQ=all`.

Focused timing with current default route policy versus forced packed Q8_1:

```text
shape          default us    q8all us    q8all/default    decision
Vcur-0           2168.82      2511.85        1.16x        reject forced Q4 Q8_1
Vcur-1            540.18       249.14        0.46x        candidate: Q5 small rows
node_32         16165.07     19907.09        1.23x        reject forced Q4 Q8_1
node_100          955.90      1130.07        1.18x        keep default Q5 x4
node_372        10753.86     23340.47        2.17x        reject forced Q6 x4
Qcur-0          22548.38     30107.66        1.34x        reject forced Q4 Q8_1
Qcur-1            995.79      1210.79        1.22x        keep default Q5 x4
result_output 1117752.89   1131429.98        1.01x        reject forced Q6 x4
```

Focused timing for opt-in Q4_K x4 MMQ64 candidate
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`:

```text
shape          default us    q4x4mmq64 us   variant/default   decision
Vcur-0           2168.82          202.02          0.09x        strong focused win
node_32         16165.07         1304.93          0.08x        strong focused win
Qcur-0          22548.38         1193.69          0.05x        strong focused win
```

Same-binary HRX model smokes with the candidate still opt-in:

```text
shape          default tok/s   q4x4mmq64 tok/s   variant/default
p33 ub512          91.567            120.005          1.31x
p512 ub512        119.602            182.491          1.53x
p513 ub1024       108.977            158.754          1.46x
```

Route traces show the candidate selected exactly the dense Q4_K prompt family:
160 `hrx_mul_mat_vec_q4_k_q8_1_x4_mmq64x64_wg256_f32` dispatches in the
model runs, with Q5/Q6/F16 route counts unchanged. The focused CPU-reference
gate passed all 8 exported p512 Q4/Q5/Q6 rows. No fallback or CPU strings were
present in the model-smoke traces.

Interpretation:

- Aggregate model timing is now demoted to acceptance evidence for this family.
  The active schedule loop should use the focused p512 op rows above.
- The existing Q4_K Q8_1 scalar route is not a production p512 schedule. It is
  slower than the legacy row/column route on the exact Q4 rows tested here.
- The opt-in Q4_K x4 MMQ64 candidate is a valid schedule-family pivot on
  `gfx1151`: it passes focused correctness, emits integer-dot wave64 code, and
  materially improves focused p512 rows plus odd/tail model smokes. It is still
  not default-promoted until repeated model rows, full available basket rows,
  and broader odd/tail focused gates are recorded.
- The existing Q5_K x4 MMQ family is strong for large p512 rows, and forced
  Q8_1 only opens a candidate for the small `Vcur-1` shape.
- The existing Q6_K x4 MMQL policy is not enough for `node_372` and especially
  `result_output`; this family needs a new schedule or a major retune, not a
  route-policy flip.
- Next Q4/Q6 work should add candidate HIP schedules to the CMake/Ninja catalog
  and compare them against these focused rows before any full-model A/B.

Additional focused evidence for the Q6_K x4 MMQ64 candidate:

```text
row             default us   q6 mmq64 us   speedup
node_372          10475.60        2682.39    3.91x
result_output    437618.52       98076.84    4.46x
```

Narrow prompt threshold evidence for the combined Q4/Q6 MMQ64 candidates:

```text
p2:  reject MMQ64 route selection; Q4 rows regress badly.
p8:  mixed Q4 result; not enough for default.
p16: mixed Q4 result; not enough for default.
p32: clean focused win on selected Q4/Q6 rows.
p33: exact odd CPU-reference gate passes with Q4/Q6 MMQ64 selected.
p513: exact odd CPU-reference gate passes with Q4/Q6 MMQ64 selected.
```

Final selector guard for the current candidates is `cols >= 32`, `rows % 64 == 0`,
`k % 256 == 0`, contiguous tensors, packed Q8_1 x4 RHS, and explicit opt-in
environment variables. Decode `cols=1` and p2 very-narrow prompt stay on the
existing specialty routes.

Same-binary HRX model A/B with Q4+Q6 MMQ64 enabled:

```text
shape   default tok/s   q4+q6 mmq64 tok/s   speedup
p33          90.409             128.015       1.42x
p512        118.151             185.620       1.57x
p513        107.658             161.665       1.50x
```

Repeated same-source HRX/Vulkan comparison with `-b 1024 -ub 1024 -r 3`:

```text
shape   HRX default   HRX q4+q6   Vulkan     q4+q6/default   q4+q6/Vulkan
p33        92.230      127.378    218.193        1.38x          0.58x
p512      117.057      181.212   1095.640        1.55x          0.17x
p513      104.277      156.052    922.711        1.50x          0.17x
```

Wide Q4 dense prompt MMQL128 follow-up on top of the current
Q4/Q6/F16/Q4-ID/Q5-ID stack:

```text
focused p512 Q4 rows:
shape      q4 mmq64 us   q4 mmql128 us   decision
Vcur-0        202.02          198.59      neutral/slight win
node_32      1304.93          732.82      wide route wins
Qcur-0       1193.69          696.78      wide route wins

odd focused A/B:
shape        q4 mmq64 us   q4 mmql128 us   decision
p33 Vcur-0       99.07          192.29      keep MMQ64
p33 node_32     231.72          338.22      keep MMQ64
p33 Qcur-0      166.76          218.90      keep MMQ64
p513 Vcur-0     210.06          202.98      wide route slight win
p513 node_32   1389.71          782.28      wide route wins
p513 Qcur-0    1348.02          815.95      wide route wins

same-binary model A/B:
shape   current stack   + Q4 MMQL128   variant/current
p33       205.259          201.773         0.98x
p512      601.779          635.944         1.06x
p513      515.150          539.050         1.05x
```

Route sanity with both Q4 env flags enabled selected Q4 MMQ64 for p33 and Q4
MMQL128 for p512. The accepted selector guard for the Q4 MMQL128 candidate is
therefore `cols >= 128`, `rows % 128 == 0`, `k % 256 == 0`, contiguous
tensors, packed Q8_1 x4 RHS, and explicit opt-in
`GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.

Current p512 interpretation after Q4/Q6 cleanup:

- Q4/Q6 MMQ64 materially improves HRX, but the remaining Vulkan gap is still
  large enough that default promotion should wait for broader basket evidence.
- Existing F16 attention selector alternatives do not provide a clean win:
  cols8 helps KQV and hurts KQ by about the same amount; cols4 and generic are
  rejected.
- Existing F32 MoE-logits route policy is already best on the focused exported
  row; rows2-cols8 beats cols8, cols16, and generic.
- Q5 small-row MMQL128 is useful in focused timing but fails model-level
  acceptance, so it remains opt-in diagnostic only.
- Next schedule work should prioritize a real F16 attention-chain schedule or
  a MoE-path fusion/schedule, using Vulkan `MUL_MAT_ID` and attention labels
  as priors rather than selector-only flips.

## Vulkan Reference Facts

For `p512 ub512`, the largest Vulkan perf-label families are:

```text
MUL_MAT_ID q4_K m=768 n=8 k=2048 n_expert=128 batch=512:
  94 x 1935.13 us = 181903 us
MUL_MAT f16 m=512 n=512 k=128 batch=32:
  48 x 3023.17 us = 145112 us
MUL_MAT_ID q4_K m=2048 n=8 k=768 n_expert=128 batch=512:
  34 x 2058.28 us = 69981.6 us
SOFT_MAX:
  48 x 474.009 us = 22752.4 us
MUL_MAT q4_K m=4096 n=512 k=2048:
  42 x 558.785 us = 23469 us
MUL_MAT q4_K m=2048 n=512 k=4096:
  34 x 497.483 us = 16914.4 us
MUL_MAT q6_K m=2048 n=512 k=4096:
  10 x 556.621 us = 5566.21 us
MUL_MAT q4_K m=512 n=512 k=2048:
  84 x 64.255 us = 5397.46 us
```

These labels make three first-class schedule families:

- MoE `MUL_MAT_ID` Q4_K/Q5_K.
- F16 attention-chain matmuls.
- Q4/Q5/Q6 prompt matmul.

## Prior Rows To Fill Before Coding

### Prior Row: Vulkan Q4/Q5/Q6 Prompt Matmul

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

### Prior Row: HRX v1 Q4/Q5/Q6 Prompt Matmul

- source/symbol/backend:
  - `ggml/src/ggml-hrx/kernels/mul_mat_vec_q4_k_q8_1.hip.cpp`,
    `hrx_mul_mat_vec_q4_k_q8_1_f32`.
  - `ggml/src/ggml-hrx/kernels/mul_mat_vec_q4_k_q8_1_x4_wave64.hip.cpp`,
    `hrx_mul_mat_vec_q4_k_q8_1_x4_mmq64x64_wg256_f32`.
  - `ggml/src/ggml-hrx/kernels/mul_mat_vec_q5_k_q8_1.hip.cpp`,
    `hrx_mul_mat_vec_q5_k_q8_1_f32`,
    `hrx_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32`,
    `hrx_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32`.
  - `ggml/src/ggml-hrx/kernels/mul_mat_vec_q5_k_q8_1_wave64.hip.cpp`,
    `hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32`,
    `hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32`.
  - `ggml/src/ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1.hip.cpp`,
    `hrx_mul_mat_vec_q6_k_q8_1_f32`,
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32`.
  - `ggml/src/ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp`,
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32`,
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_f32`.
- shape regime and evidence artifact:
  focused p512 Qwen3 Q4_K_XL prompt rows in
  `cache/hrxv1/gfx1151/focused-qk-prompt-perf-timing-20260617-141110/`
  and forced-Q8 control in
  `cache/hrxv1/gfx1151/focused-qk-prompt-q8all-20260617-141500/`.
- tile/workgroup/subgroup:
  Q4 scalar Q8_1 is one row by one column per workgroup with 256 threads. Q4
  x4 MMQ64 uses `BM=64`, `BN=64`, 256 threads, wave64, 64 row lanes by four
  column lanes, and 16 output columns per thread. Q5/Q6 scalar Q8_1 routes use
  the same reduction shape. Q5 x4 MMQL uses `BM=128`,
  `BN=128`, `BK_STEP=1`, 256 threads, wave64, `WM=64`, `WN=64`, `TM=4`,
  `TN=2`. Q6 x4 MMQL uses `BM=64/128`, `BN=64/128`, `BK_STEP=4`, 256 threads,
  wave64, `WM=64`, `WN=32`, `TM=4`, `TN=2`.
- lane ownership and per-lane outputs:
  Q4/Q5/Q6 scalar Q8_1 routes cooperatively reduce one output per workgroup.
  Q4 x4 MMQ64 owns 1024 outputs per workgroup, 16 columns for each of 64 rows.
  Q5 x4 MMQL owns `WNITER * TM * TN = 64` outputs per workgroup. Q6 x4 MMQL
  owns `WNITER * TM * TN = 32` outputs per workgroup.
- vector/packed load widths:
  scalar routes consume ordinary Q8_1 RHS blocks. x4 routes consume
  `hrx_block_q8_1_x4` packed RHS. Q4 x4 MMQ64 stages each 64-column RHS tile
  in LDS and streams Q4 A directly per owned row. Q5/Q6 MMQL stage A with
  `LOAD_VEC_A=4` and B with `LOAD_VEC_B=16`.
- quant or element layout:
  Q4/Q5/Q6 K-quant blocks on A, F32 activations quantized to Q8_1 or x4-packed
  Q8_1 on B before matmul.
- dot/WMMA/ALU primitive and signedness:
  Q4/Q5 use `__builtin_amdgcn_sudot4(false, qpack, true, rpack, ...)` for
  unsigned A codes times signed Q8_1 RHS. Q6 uses signed dot forms
  `__builtin_amdgcn_sudot4(true, qpack, true, rpack, ...)`. Q4 scalar Q8_1
  uses scalar integer multiply/accumulate inside the workgroup.
- A/B staging, barriers, unroll, reduction, writeback:
  scalar routes reduce through wave shuffles and shared scratch, then write one
  output. Q4 x4 MMQ64 stages only the RHS tile into LDS, synchronizes once per
  Q8 block, unrolls eight dot lanes, and writes 16 column outputs per row lane
  with explicit column tails. Q5/Q6 MMQL stage A and B into LDS, synchronize
  for every BK step, unroll over staged qpack lanes, and write output tiles
  with explicit tail checks.
- emitted resource facts:
  HSACOs are built by CMake/Ninja under
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/`.
  Q4 x4 MMQ64 metadata: wavefront 64, VGPR 192, SGPR 40, no spills, LDS 2304
  bytes. Disassembly contains 128 `v_dot4_i32_iu8` instructions, 2
  `s_barrier`, 68 `s_waitcnt`, 19 `global_load`, and 84 `ds_*` operations.
- known win/regression/constraint:
  Q4 x4 MMQ64 wins strongly on focused p512 rows and improves p33/p512/p513
  model smokes under an opt-in gate. Q5 x4 MMQL wins strongly for large p512
  rows. Forced packed Q8_1 improves only `Vcur-1` among the focused p512 rows.
  Current Q6 x4 MMQL is not sufficient for the p512 production gap.

### Prior Row: Vulkan MoE MUL_MAT_ID Q4_K

- source/symbol/backend:
  `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`,
  `mul_mm_id_funcs.glsl`, `mul_mm_funcs.glsl`, and `ggml-vulkan.cpp`
  `pipeline_dequant_mul_mat_mat_id_q8_1[Q4_K/Q5_K]`; Vulkan backend.
- shape regime and evidence artifact:
  Qwen3 30B Q4_K_XL `--flash-attn 0` p512 MoE labels in
  `cache/hrxv1/gfx1151/current-best-hrx-vulkan-r3-20260617-154104/vulkan-p512/stderr.log`.
  Steady labels are about 1.88-2.04 ms for Q4_K `MUL_MAT_ID` and about
  2.20-2.24 ms for Q5_K `MUL_MAT_ID` at `n=8`, `batch=512`,
  `n_expert=128`.
- tile/workgroup/subgroup:
  Vulkan uses the quantized `MUL_MAT_ID` MMQ path with `warptile_mmqid_int_k`
  for K-quants. On the RADV/AMD path the relevant tuple is the integer-dot
  ID MMQ family from `ggml-vulkan.cpp`: large/medium/small variants around
  `128x128`, `64x64`, or `32x32`, `BK=32`, subgroup 16 for K-quant ID MMQ,
  and integer `TM/TN/TK` specialization.
- lane ownership and per-lane outputs:
  a workgroup owns a tile of output rows by selected token/id columns rather
  than one scalar output. Expert IDs are used inside the tiled shader so the
  schedule keeps MMQ-style data reuse while respecting routed experts.
- vector/packed load widths:
  RHS is Q8_1-packed through the Vulkan MMQ path. A-side K-quants are consumed
  through the generated Q4_K/Q5_K dequant and packed integer dot helpers.
- quant or element layout:
  Q4_K/Q5_K expert weights on A, F32 activations converted to Q8_1 for the
  MMQ path, F32 output, expert ids carried as the `MUL_MAT_ID` source.
- dot/WMMA/ALU primitive and signedness:
  integer dot product path, not coopmat. Q4 codes are unsigned against signed
  Q8_1 activations; Q5 uses the matching K-quant integer dot/dequant helper.
- A/B staging, barriers, unroll, reduction, writeback:
  Vulkan keeps the `MUL_MAT_ID` operation fused into the MMQ shader family.
  The useful prior is the tiled packed-Q8 schedule and ID-aware writeback, not
  a separate route compaction plus scalar per-row matvec.
- emitted resource facts:
  SPIR-V was not disassembled in this pass. The evidence to preserve is the
  same-machine Vulkan perf logger label, pipeline family, and specialization
  constants above.
- known win/regression/constraint:
  Vulkan's p512 MoE labels are fast relative to the existing HRX focused
  `MUL_MAT_ID` route. Use this as the schedule target, but verify real route
  densities because old HRX spike MoE wins could depend on dense expert usage.

### Prior Row: HRX v1 MoE MUL_MAT_ID Q4_K

- source/symbol/backend:
  `ggml/src/ggml-hrx/kernels/mul_mat_id_q4_k.hip.cpp` and
  `mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp`; HRX v1 HIP C++ backend. Relevant
  exports include `hrx_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32`,
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32`, and
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- shape regime and evidence artifact:
  Focused Qwen3 p512 exported MoE rows in
  `cache/hrxv1/gfx1151/focused-moe-mul-mat-id-opgate-20260617-154557/`.
  Route-traced provider-pinned rerun:
  `cache/hrxv1/gfx1151/focused-moe-mul-mat-id-trace-20260617-155927/`.
  Q4_K `ffn_moe_gate-0` and `ffn_moe_down-0` pass CPU-reference support/test;
  Q5_K `ffn_moe_down-1` is unsupported.
- tile/workgroup/subgroup:
  selector prefers grouped Q8_1 x4 MMQ64x16 for `k==512`, `rows%64==0`,
  `n_ids==8`, and prompt tokens >=32 when providers are available. Built
  grouped Q8_1 x4 exports are wave64 with 64-thread workgroups; older row
  grouped routes are wave32. The Qwen3 p512 expert rows here are `k=2048`
  and `k=768`, so route tracing shows the selector choosing
  `hrx_mul_mat_id_q4_k_wg64_f32`, grouped=0, q8_1_x4=0, with workgroup grids
  `[768,4096,1]` and `[2048,4096,1]`.
- lane ownership and per-lane outputs:
  grouped Q8_1 routes compact token/id assignments by expert, then dispatch a
  row tile by token tile by expert grid. Existing non-Q8 grouped routes process
  route groups with row2/route8 ownership.
- vector/packed load widths:
  grouped Q8_1 routes require `hrx_quantize_q8_1_x4` scratch and consume
  `hrx_block_q8_1_x4`. Non-Q8 routes consume F32 RHS directly.
- quant or element layout:
  Q4_K expert weights with F32 or packed Q8_1 activations, F32 output, route
  scratch split into counts and per-expert route lists.
- dot/WMMA/ALU primitive and signedness:
  Q4 grouped Q8_1 x4 MMQ uses integer dot in wave64 HSACO. Metadata for
  `mul_mat_id_q4_k_q8_1_x4_mmq.hsaco` reports wavefront 64, VGPR 107-136,
  SGPR 87, no spills.
- A/B staging, barriers, unroll, reduction, writeback:
  dispatch path clears counts, compacts routes, optionally quantizes RHS to
  x4 Q8_1 scratch, then launches grouped MMQ. This is functionally fused at
  the op boundary but costs multiple HRX dispatches before the matmul.
- emitted resource facts:
  CMake/Ninja built HSACOs under
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/`.
  `mul_mat_id_q4_k_q8_1_x4_mmq.hsaco` contains the grouped Q8_1 exports with
  wavefront 64 and no private spills.
- known win/regression/constraint:
  Focused p512 timing is not competitive: Q4 gate is about 20.84 ms and Q4
  down is about 28.47 ms, while Vulkan's corresponding p512 labels are around
  1.9-2.1 ms. Full HRX `llama-bench` currently decomposes MoE into ordinary
  packed `MUL_MAT`, F32 MoE logits, and `GET_ROWS`, and does not dispatch
  `MUL_MAT_ID` for the model path. Provider pinning now proves that blindly
  expecting the existing grouped Q8_1 x4 route fails for these shapes. Do not
  promote this route without a new Q4/Q5 expert schedule for the `k=2048/768`
  Qwen3 regime.

### Prior Row: Vulkan F16 Attention Chain

- source/symbol/backend:
  `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp`,
  `mul_mat_vec_base.glsl`, and `ggml-vulkan.cpp` DMMV pipeline creation for
  `pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_F16]`; Vulkan backend.
- shape regime and evidence artifact:
  Qwen3 30B Q4_K_XL `--flash-attn 0` p512 attention-chain labels in
  `cache/hrxv1/gfx1151/q4q6-mmq64-hrx-vulkan-r3-20260617-150414/vulkan-p512/stderr.log`.
  Steady Vulkan labels:
  `MUL_MAT f16 m=128 n=512 k=512 batch=32` at about 409-411 us and
  `MUL_MAT f16 m=512 n=512 k=128 batch=32` at about 248 us.
- tile/workgroup/subgroup:
  DMMV specialization for AMD non-GCN uses subgroup workgroup mode:
  `BLOCK_SIZE=subgroup_size`, `NUM_ROWS=2` for F16 A, and
  `NUM_COLS=i+1` up to 16 for the active prompt width.
- lane ownership and per-lane outputs:
  one subgroup cooperatively reduces two rows by up to sixteen columns. Each
  lane accumulates partial products for its K stripe and reduction is local to
  the subgroup or hybrid subgroup/shared path depending on selected pipeline.
- vector/packed load widths:
  F16 A uses `K_PER_ITER=2`, fetching/dequantizing a pair of F16 elements per
  lane iteration. F32 RHS uses scalar F32 loads per active output column.
- quant or element layout:
  A is F16, RHS is F32, output is F32. This is exact relative to the current
  HRX conservative F16 route family; it is not a BF16/WMMA approximate route.
- dot/WMMA/ALU primitive and signedness:
  scalar FMA over F16-expanded-to-F32 A values and F32 RHS. Vulkan is not using
  flash attention or coopmat for these `--flash-attn 0` labels.
- A/B staging, barriers, unroll, reduction, writeback:
  no packed RHS conversion. Reduction is subgroup-shaped, avoiding HRX's
  selected 256-thread cross-wave reduction for the same rows.
- emitted resource facts:
  SPIR-V was not disassembled in this pass. The useful prior facts are the
  specialization constants and same-machine Vulkan label timings above.
- known win/regression/constraint:
  This is the strongest local prior for F16 attention-chain prompt matmuls in
  the no-flash-attention graph. It does not answer whether an FA route should
  replace the chain when `--flash-attn 1` is the target.

### Prior Row: HRX v1 F16 Attention Chain

- source/symbol/backend:
  `ggml/src/ggml-hrx/kernels/mul_mat_vec_f16_batched.hip.cpp`,
  `hrx_mul_mat_vec_f16_batched_cols16_f32`; HRX v1 HIP backend.
- shape regime and evidence artifact:
  selected for Qwen3 30B Q4_K_XL p512 attention-chain rows in
  `cache/hrxv1/gfx1151/q4q6-mmq64-hrx-vulkan-r3-20260617-150414/q4q6-p512/stderr.log`.
  Focused policy sweep:
  `cache/hrxv1/gfx1151/focused-f16-attention-policy-sweep-20260617-151010/`.
- tile/workgroup/subgroup:
  one row by sixteen columns per 256-thread workgroup. On gfx1151 the HSACO is
  wave32, so this uses eight waves and then reduces across wave partials.
- lane ownership and per-lane outputs:
  each workgroup owns one A row and up to sixteen RHS columns. Each lane walks
  a K stripe and accumulates sixteen scalar sums.
- vector/packed load widths:
  scalar F16 loads for A, scalar F32 loads for each RHS column. No RHS packing
  or matrix-tile reuse beyond the sixteen columns owned by the workgroup.
- quant or element layout:
  A is F16, RHS is F32, output is F32.
- dot/WMMA/ALU primitive and signedness:
  scalar FMA after `__half2float`; no `v_wmma`/`v_dot` family.
- A/B staging, barriers, unroll, reduction, writeback:
  current cols16 route uses shared scratch for cross-wave reductions, then lane
  zero writes up to sixteen output columns. Focused selector-only sweeps showed
  cols8 helped KQV but hurt KQ, cols4 and generic were rejected.
- emitted resource facts:
  built HSACO metadata for `hrx_mul_mat_vec_f16_batched_cols16_f32` reports
  wavefront 32, VGPR 48, SGPR 104, no spills, and 512 bytes LDS.
- known win/regression/constraint:
  Correct but far slower than Vulkan for the prompt attention-chain rows.
  Policy-only route changes were insufficient; a work-ownership schedule pivot
  was required.

## Candidate Rows

### Candidate: Q4_K x Q8_1 x4 MMQ64 Dense Prompt

- follows prior: Vulkan K-quant MMQ and HRX v1 Q5 x4 packed-Q8_1 schedule
  family.
- source/symbol: `mul_mat_vec_q4_k_q8_1_x4_wave64.hip.cpp`,
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmq64x64_wg256_f32`.
- gate: `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQ64_PROMPT=1`.
- pivot axis: replace scalar one-output Q4 Q8_1 reduction with a dense
  64-row by 64-column packed-Q8_1 tile that stages RHS and owns 16 output
  columns per lane group.
- correctness: focused p512 Q4/Q5/Q6 `test-backend-ops test` passed all 8
  rows at
  `cache/hrxv1/gfx1151/focused-qk-prompt-q4x4mmq64-opgate-20260617-142923/`.
- timing: focused p512 Q4 rows improved by roughly 10.7x, 12.4x, and 18.9x.
- model smoke: HRX-only same-binary one-rep p33, p512, and p513 improved by
  1.31x, 1.53x, and 1.46x respectively.
- decision: keep as opt-in candidate; next gates are repeated HRX/Vulkan rows,
  focused odd exported-op rows, and full available basket coverage before
  default promotion.

### Candidate: Q4_K x Q8_1 x4 MMQL128 Dense Prompt

- follows prior: Vulkan AMD K-quant MMQ tuple `BLOCK_SIZE=256`, `BM=128`,
  `BN=128`, `WM=64`, `WN=64`, `WMITER=1`, `TM=4`, `TN=2`, `WARP=64`, with
  staged Q4 A cache and packed Q8_1 x4 B cache.
- source/symbol:
  `mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp`,
  `hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_wg256_f32`.
- gate: `GGML_HRX_ENABLE_Q4_K_Q8_1_X4_MMQL128_PROMPT=1`.
- selector: `cols >= 128`, `rows % 128 == 0`, `k % 256 == 0`; p33 must stay
  on Q4 MMQ64.
- correctness: p512, p33, and p513 focused CPU-reference gates passed.
- timing: focused p512/p513 Q4 large rows improve by about 1.6x to 1.8x over
  Q4 MMQ64; p33 regresses, which defines the `cols >= 128` guard.
- model evidence: same-binary HRX A/B on top of the current opt-in stack
  improves p512 by about 5.7% and p513 by about 4.6%; p33 route selection is
  unchanged and the measured delta is treated as run noise.
- decision: accept as a gfx1151 opt-in wide-prefill candidate, not a default.

### Candidate: F16 Batched Attention Rows2 Cols16 WG32

- follows prior: Vulkan F16 DMMV `NUM_ROWS=2`, `NUM_COLS=16`, subgroup-sized
  workgroup schedule.
- source/symbol: `mul_mat_vec_f16_batched.hip.cpp`,
  `hrx_mul_mat_vec_f16_batched_rows2_cols16_wg32_f32`.
- gate: `GGML_HRX_ENABLE_F16_BATCHED_ROWS2_COLS16_WG32_PROMPT=1`.
- pivot axis: replace HRX's one-row by sixteen-column 256-thread cross-wave
  reduction with a two-row by sixteen-column one-wave32 workgroup. Preserve
  exact F16-to-F32 expansion, F32 RHS, and F32 accumulation.
- correctness:
  p512 focused gate passed 4/4 rows at
  `cache/hrxv1/gfx1151/focused-f16-rows2-cols16-wg32-opgate-20260617-153054/`.
  Odd p33+p513 focused gate passed 8/8 rows at
  `cache/hrxv1/gfx1151/focused-f16-rows2-cols16-wg32-odd-opgate-20260617-153411/`.
- timing:
  focused p512 prompt-width F16 rows improved from about 3.18 s to 1.57 s
  for KQV and from about 7.47 s to 1.67 s for KQ in the backend-op replay.
  Same-binary HRX model A/B on top of Q4/Q6 MMQ64 improved p33 by 1.14x,
  p512 by 1.31x, and p513 by 1.41x.
- compile evidence:
  CMake/Ninja-built `mul_mat_vec_f16_batched.hsaco`; new symbol reports
  wavefront 32, VGPR 67, SGPR 83, no spills, and zero LDS.
- decision:
  keep as opt-in gfx1151 candidate pending broader basket coverage and
  target-specific selector-policy migration. It is the leading current F16
  attention-chain route for the Qwen3 Q4_K_XL no-FA prompt regime.

### Candidate: Q4_K MoE ID Wide-K Grouped Q8_1 x4 MMQ16

- follows prior: Vulkan MoE `MUL_MAT_ID` integer-MMQ path and the existing HRX
  grouped Q4_K ID Q8_1 x4 MMQ16 kernel.
- source/symbol: `mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp`,
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- gate: `GGML_HRX_ENABLE_Q4_K_ID_Q8_1_X4_MMQ16_WIDE_K_PROMPT=1`.
- pivot axis: remove the selector-only `k==512` restriction under opt-in for
  Qwen3-style expert rows while preserving conservative shape guards:
  `k % 256 == 0`, `rows % 64 == 0`, `n_ids == 8`, `n_tokens >= 32`.
- correctness:
  p512 focused gate passed at
  `cache/hrxv1/gfx1151/focused-moe-q4-id-widek-mmq16-20260617-160209/`.
  Odd p33 and p513 focused gates passed at
  `cache/hrxv1/gfx1151/focused-moe-q4-id-widek-mmq16-odd-20260617-160622/`.
- timing:
  focused p512 Q4 MoE ID rows improved from about 20.85 ms and 28.24 ms to
  2.09 ms and 1.99 ms. Focused p33 rows were about 0.26 ms and 0.25 ms;
  focused p513 rows were about 2.14 ms and 2.10 ms.
- model smoke:
  repeated HRX p33, p512, and p513 improved by 1.27x, 1.90x, and 1.79x over
  the previous opt-in stack in
  `cache/hrxv1/gfx1151/current-best-with-q4-id-widek-r3-20260617-160734/`.
  Compared to the existing validated Vulkan r3 artifact, HRX is now about
  0.88x Vulkan at p33, 0.43x at p512, and 0.45x at p513.
- route evidence:
  p512 uses grouped=1/q8_1_x4=1 with route capacity 4096 and grids
  `[12,32,128]` for `k=2048 rows=768` and `[32,32,128]` for
  `k=768 rows=2048`. p33 and p513 route grids also passed provider-pinned
  focused gates.
- decision:
  accept as an opt-in gfx1151 candidate. Do not make it a broad default until
  Q5_K `MUL_MAT_ID` support, broader model coverage, and target-specific
  policy migration are complete.

### Candidate Gate: Q5_K MoE ID Grouped Q8_1 x4 MMQ16 Probe

- Production target:
  Qwen3 30B Q4_K_XL MoE Q5_K expert down row,
  `MUL_MAT_ID q5_K k=768 rows=2048 n_ids=8 n_tokens=33/512/513`.
- Baseline command:
  focused `test-backend-ops` currently reports the Q5_K `MUL_MAT_ID` row as
  unsupported; model baseline is the current opt-in stack with Q4 ID wide-K.
- Variant command:
  focused `test-backend-ops test/perf -b HRX0 -o MUL_MAT_ID --test-file
  <moe_qk_prompt.txt>` with the Q5 candidate env and
  `GGML_HRX_EXPECT_MUL_MAT_ID_PROVIDER` pinned.
- Same-runner comparison method:
  same-binary HRX A/B at p33, p512, and p513 only after focused Q5 correctness
  passes.
- Route trace path:
  to be filled by the focused Q5 artifacts.
- Scheduler/per-op trace path:
  HRX route histogram from the model A/B artifact; no default promotion from
  focused-only timing.
- Focused CPU-reference command:
  `test-backend-ops test -b HRX0 -o MUL_MAT_ID --test-file <moe_qk_prompt.txt>
  --output csv`.
- Compile report path:
  built HSACO metadata under
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/`.
- Target listing path:
  CMake/Ninja build output and catalog validation artifact for the new route.
- Prior-art schedule source:
  Q4_K MoE grouped Q8_1 x4 MMQ16 and dense Q5_K Q8_1 x4 wave64 MMQ.
- Odd-size and tail gate:
  focused p33, p512, and p513 exported MoE rows, plus same-runner model smoke
  for p33 and p513 if focused gates pass.
- Promotion rule:
  accepted only as opt-in unless Q5 correctness/timing passes, model A/B helps,
  route traces prove selection, and broader basket coverage does not regress.

### Candidate: Q5_K MoE ID Grouped Q8_1 x4 MMQ16

- follows prior:
  accepted Q4_K MoE ID grouped Q8_1 x4 MMQ16 route plus dense Q5_K Q8_1 x4
  wave64 MMQ packing/high-bit schedule.
- source/symbol:
  `mul_mat_id_q5_k_q8_1_x4_mmq.hip.cpp`,
  `hrx_mul_mat_id_q5_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- gate:
  `GGML_HRX_ENABLE_Q5_K_ID_Q8_1_X4_MMQ16_PROMPT=1`.
- pivot axis:
  add Q5_K A-side high-bit packing to the grouped MoE ID Q8_1 x4 schedule.
  Preserve Q4 route compaction, route tile `BN=16`, wave64, unsigned K-quant
  codes times signed Q8_1 activations, and the conservative prompt guard:
  `k % 256 == 0`, `rows % 64 == 0`, `n_ids == 8`, `n_tokens >= 32`.
- correctness:
  p512 focused gate passed at
  `cache/hrxv1/gfx1151/focused-moe-q5-id-mmq16-p512-20260617-162033/`.
  Odd p33 and p513 focused gates passed at
  `cache/hrxv1/gfx1151/focused-moe-q5-id-mmq16-odd-20260617-162104/`.
- timing:
  focused Q5 MoE ID rows were about 261 us at p33, 2246 us at p512, and
  2275 us at p513. Repeated model current-best improved p33, p512, and p513
  by 1.07x, 1.25x, and 1.23x over the Q4-ID-only stack.
- compile evidence:
  CMake/Ninja-built `mul_mat_id_q5_k_q8_1_x4_mmq.hsaco`; wavefront 64,
  VGPR 119, SGPR 87, LDS 3264 bytes, no spills, 128 `v_dot` instructions.
- model smoke:
  `cache/hrxv1/gfx1151/current-best-with-q4q5-id-r3-20260617-162308/`.
  Fresh Vulkan comparison:
  `cache/hrxv1/gfx1151/current-best-q4q5-id-vulkan-r3-20260617-162608/`.
  HRX is about 0.91x Vulkan at p33, 0.53x at p512, and 0.54x at p513.
- route evidence:
  focused p512 Q5 row selects grouped=1/q8_1_x4=1 with route capacity 4096 and
  grid `[32,32,128]` for `k=768 rows=2048`. Odd p33 and p513 select the same
  provider with route capacities 264 and 4104.
- decision:
  accept as an opt-in gfx1151 candidate. Do not default broadly until broader
  basket coverage and target-specific policy migration are complete.

### Rejected Candidate Set: Q6_K Dense Prompt p512 Schedule Pivots

- follows prior:
  Vulkan AMD large int-K MMQ tuple (`BM128`, `BN128`, `BK32`, `WM64`,
  `WN64`, `WMITER1`, `TM4`, `TN2`, `TK1`, wave64) and the current fastest HRX
  Q6 direct MMQ64 route.
- source/symbols:
  - existing staged:
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_f32` and
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32`;
  - new staged:
    `mul_mat_vec_q6_k_q8_1_x4_mmql128.hip.cpp`,
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x128_wg256_f32`;
  - new direct:
    `mul_mat_vec_q6_k_q8_1_x4_wave64_direct.hip.cpp`,
    `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x128_wg256_f32`.
- gates:
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQL128_PROMPT=1` and
  `GGML_HRX_ENABLE_Q6_K_Q8_1_X4_MMQ64X128_PROMPT=1`.
- pivot axes:
  - staged 128x128 candidate uses the Vulkan/Q4-style `BK_STEP=1`, `WN=64`,
    `WNITER=8` shape with Q6 A and Q8_1 x4 B staged through LDS.
  - direct 64x128 candidate keeps the current fastest direct Q6 lane ownership
    and doubles the column tile from 64 to 128 to test A reuse.
- correctness:
  all p512 focused CPU-reference gates passed.
- timing:
  p512 `result_output q6_K[2048,151936] x f32[2048,512]`:
  `MMQ64 94.75 ms`, `MMQL128x64 440.73 ms`, `MMQL64x128 468.63 ms`,
  `MMQL128x128 281.16 ms`, `MMQ64x128 194.44 ms`.
  p512 `node_372 q6_K[4096,2048] x f32[4096,512]`:
  `MMQ64 2.57 ms`, `MMQL128x64 9.93 ms`, `MMQL64x128 10.46 ms`,
  `MMQL128x128 7.83 ms`, `MMQ64x128 5.75 ms`.
- artifacts:
  - rerank:
    `cache/hrxv1/gfx1151/focused-q6-variant-rerank-p512-20260617-170020/`;
  - staged 128x128:
    `cache/hrxv1/gfx1151/focused-q6-mmql128-p512-20260617-170743/`;
  - direct 64x128:
    `cache/hrxv1/gfx1151/focused-q6-mmq64x128-p512-20260617-171056/`.
- decision:
  reject these Q6 replacements. Keep direct MMQ64 as the current best Q6
  prompt route, and do not run model-level A/B for slower focused candidates.
- next evidence needed:
  inspect HSACO resource/ISA for the staged Q6 routes and compare against
  Vulkan shader behavior before another Q6 schedule. The next Q6 candidate
  should change lane ownership, staging granularity, or accumulation strategy;
  simple wider columns and direct Q4 schedule transplant have been falsified.

Valid first candidate directions after prior mining:

- Fix the `p31` packed Q8_1 prompt route guard or add a narrow odd-size route
  if correctness and timing support it.
- Add or retune a `p512/p513` Q4/Q5/Q6 prompt route only after comparing HRX
  and Vulkan packed schedule facts.
- Add or retune a MoE `MUL_MAT_ID` route only after decomposing the Vulkan and
  HRX route schedules.
- Investigate F16 attention-chain matmul only after confirming whether a
  flash-attention route should replace the chain for this model/mode.

### Candidate: D128 F16 Flash Attention Prefill Direct

- follows prior:
  existing HRX1 `hrx_flash_attn_ext_f32_f16_prefill_direct` route and the
  Vulkan same-model FA-on result. This is a fusion/schedule candidate, not an
  aggregate-only optimization.
- source/symbol:
  `flash_attn_ext_f32_f16_prefill_direct.hip.cpp` with gfx11 include
  `flash_attn_ext_f32_f16_prefill_direct_gfx11.inc`,
  split into the gfx1151 candidate export
  `hrx_flash_attn_ext_f32_f16_prefill_direct_d128` so the legacy D256 route
  keeps its original codegen.
- gate:
  `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`.
- pivot axis:
  extend the existing D256 prefill-direct flash route to the Qwen3 MoE shape
  `D=128`, `H=32`, `H_KV=4`, keep live-variable sequence support, skip the
  second D128 output half, and use f32 PV accumulation for D128 correctness.
- correctness:
  p512 focused gate passed at
  `cache/hrxv1/gfx1151/focused-fa1-direct-d128-f32pv-p512-20260617-172350/`.
  Odd p33 and p513 focused gates passed at
  `cache/hrxv1/gfx1151/focused-fa1-direct-d128-p33-20260617-173253/` and
  `cache/hrxv1/gfx1151/focused-fa1-direct-d128-p513-20260617-173313/`.
  The split-provider route regate passed p33/p512/p513 at
  `cache/hrxv1/gfx1151/fa1-direct-d128-split-regate-20260617-174250/`.
- timing:
  model FA-on p33 improved `203.072 -> 213.973 tok/s` (`0.936x` Vulkan);
  p512 improved from about `463.927` to `888.692 tok/s` (`0.702x` Vulkan);
  p513 improved `448.089 -> 853.656 tok/s` (`0.715x` Vulkan).
- artifacts:
  - p512 FA-on HRX/Vulkan probe:
    `cache/hrxv1/gfx1151/fa1-current-best-p512-20260617-171632/`;
  - traced p512 route proof:
    `cache/hrxv1/gfx1151/fa1-direct-d128-traced-p512-20260617-173115/`;
  - split-provider traced p512 route proof:
    `cache/hrxv1/gfx1151/fa1-direct-d128-split-traced-p512-20260617-174708/`;
  - odd model A/B and Vulkan comparison:
    `cache/hrxv1/gfx1151/fa1-direct-d128-model-ab-odd-20260617-173548/`.
- route evidence:
  split-provider p512 selected provider
  `hrx_flash_attn_ext_f32_f16_prefill_direct_d128`,
  `mode=prefill_direct_d128`, `D=128`, `KV=512`, `N=512`, `H=32`, `H_KV=4`,
  grid `[32,32,1]` for 192 dispatches. Odd runs selected prefill-direct with
  active `KV=256` at p33 and `KV=768` at p513.
- decision:
  accept as an opt-in gfx1151 candidate. Do not default broadly until it is
  represented in target-specific selector policy and checked against the
  broader model basket.
- next evidence needed:
  close the remaining p512/p513 gap with kernel/schedule A+B against Vulkan
  flash attention. Compare lane ownership, K/V staging, mask handling,
  accumulation precision, workgroup shape, and emitted resource use before
  trying blind local knobs.

### Candidate Update: D128 Flash Attention Generalized GQA Policy

- follows prior:
  the split D128 prefill-direct route above. Basket traces showed the same D128
  flash shape class with valid GQA ratios outside Qwen3 H32/HKV4:
  H24/HKV8, H28/HKV4, H32/HKV8, and H40/HKV8.
- gate:
  `GGML_HRX_ENABLE_F16_PREFILL_FA_DIRECT_D128=1`.
- shape policy:
  keep the original H32/HKV4 behavior, and allow other valid D128 GQA shapes
  only for production-width prompt rows (`N >= 128`). This avoids the observed
  Llama 3.2 3B p33 regression while preserving p512/p513 wins.
- correctness:
  p512 focused gates passed at
  `cache/hrxv1/gfx1151/fa1-d128-gqa-focused-p512-20260617-175339/`.
  p33/p513 focused gates passed at
  `cache/hrxv1/gfx1151/fa1-d128-gqa-focused-odd-tail-20260617-175638/`.
- timing:
  p512 r3 model A/B improved newly-covered rows by `1.164x-1.626x`; p513
  model smoke improved representative rows by `1.221x-1.596x`. Llama 3.2 3B
  p33 regressed `0.986x`, so the general policy now leaves new non-H32/HKV4
  p33 rows on the safe decode provider.
- artifacts:
  - basket ranking:
    `cache/hrxv1/gfx1151/current-best-fa1-basket-p512-r1-20260617-175043/`;
  - p512 model A/B:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-model-ab-p512-r3-20260617-175433/`;
  - odd/tail model smoke:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-model-ab-odd-tail-r1-20260617-175720/`;
  - final policy sanity:
    `cache/hrxv1/gfx1151/fa1-d128-gqa-policy-sanity-20260617-175858/`.
- decision:
  accept as an opt-in gfx1151 policy broadening, not as a default route.
  Broader basket parity still requires prompt-matmul work.
