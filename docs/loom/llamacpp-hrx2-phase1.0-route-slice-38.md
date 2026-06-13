# HRX2 Phase 1.0 Route Slice 38: Q4_K MUL_MAT_ID

Date: 2026-06-13

## Scope

Add a target-neutral Loom coverage route for Qwen3 MoE `MUL_MAT_ID`
with Q4_K expert weights and F32 activations:

- gate/up expert weights: `src0=[2048,768,128,1]`, `dst=[768,8,T,1]`
- down expert weights: `src0=[768,2048,128,1]`, `dst=[2048,8,T,1]`
- `src2=[8,T,1,1]` expert ids
- tested token buckets: `T=1,16,64`

The route is coverage-quality. It deliberately reuses the simple direct
Q4_K dequant/dot pattern so phase 1 can remove CPU fallback islands before
later fusion and performance refutation.

## Implementation

New files and metadata:

- `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_id_q4_k_f32.loom`
- `mul_mat_id_q4_k_f32` catalog source/family/artifact/route
- HRX2 C++ support, shape extraction, JIT config binding, route dispatch, and
  graph execution for `GGML_OP_MUL_MAT_ID`

The route specializes on:

- `k`, `rows`, `nexperts`, `nselected`, `ntokens`
- RHS selected stride, RHS token stride
- expert-id token stride
- destination token stride

`src1_selected_stride=0` represents the gate/up broadcast RHS layout
`[k,1,T,1]`; down uses a selected-expert stride for `[k,8,T,1]`.

## Loader Integration Lesson

The route initially passed focused `test-backend-ops` rows but did not move
model graphs: Qwen3 still placed Q4_K `MUL_MAT` and `MUL_MAT_ID` on CPU.

Root cause: llama.cpp `select_weight_buft` probes weight compatibility with
representative 512-token synthetic ops. The existing Q4_K `MUL_MAT` route and
the new Q4_K `MUL_MAT_ID` route only accepted runtime token/column domains up
to 64, so the loader rejected HRX2 buffers for the weights and fell back to
CPU mapped/repack buffers.

Fix: route domains for load-bearing quantized matmuls must include the
loader's 512-token compatibility probe unless HRX2 implements a different
loader/offload contract. Slice 38 widened:

- `mul_mat_q4_k_f32` `cols_max` from 64 to 512
- `mul_mat_id_q4_k_f32` `nrows_max` from 64 to 512

This does not force 512-token dispatch during normal decode/prefill; it makes
the kernel eligible for HRX2 model weight placement.

## Evidence

Focused exported-op validation:

- `cache/hrx2/phase1_0/route-slice-38-mul-mat-id-q4-focused-current/rerun2`
- Passed four real-trace rows:
  - gate `T=1`
  - gate `T=64`
  - down `T=1`
  - down `T=64`

Qwen3 UD-Q4 model smoke after domain widening:

- `cache/hrx2/phase1_0/route-slice-38-mul-mat-id-q4-smoke-current/p1n1-verbose-after-domain`
- `cache/hrx2/phase1_0/route-slice-38-mul-mat-id-q4-smoke-current/after-domain-p16-p64`

Observed placement after the fix:

- HRX20 model buffer: `14430.05 MiB`
- q4 `MUL_MAT` HRX20 scheduler placements at p1: `1920`
- q4 `MUL_MAT_ID` HRX20 scheduler placements at p1: `1560`
- q4 `MUL_MAT_ID` HRX2 dispatches at p1/p16/p64: `260`

Remaining Qwen3 fallbacks moved to the next true gaps:

- small F32/F32 `MUL_MAT`
- Q5_K `MUL_MAT` and `MUL_MAT_ID`
- Q6_K `MUL_MAT`
- small MoE support ops such as `GET_ROWS`

## Loom Notes

The first store guard used a scalar boolean `andi` across two comparisons and
hit an AMDGPU register-class/SCC lowering failure. Rewriting the guard as
nested `scf.if` blocks compiled cleanly.

Practical rule for current Loom AMDGPU work: prefer structured control flow
over boolean-valued `andi` when combining predicate results around stores or
packed integer boundary checks.

