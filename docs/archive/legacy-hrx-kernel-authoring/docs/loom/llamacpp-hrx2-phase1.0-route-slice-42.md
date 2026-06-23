# HRX2 Phase 1.0 Route Slice 42: F32/F32 MoE Logits Matmul

Date: 2026-06-13

## Scope

This checkpoint adds target-neutral F32/F32 `MUL_MAT` coverage for Qwen3 MoE
logits:

```text
dst[expert, token] =
  dot(src0=ffn_gate_inp.weight[hidden, expert],
      src1=ffn_norm[hidden, token])
```

Accepted family:

```text
mul_mat_f32_f32
```

Accepted route domain:

```text
k    = 2048
rows = 128
cols = 1..512
```

The basket uses `cols={1,16,64}`. The wider `cols_max=512` is intentional for
llama.cpp load-time weight-placement probes, matching the loader-domain lesson
from the quantized direct and indexed matmul slices.

## Algorithm

- one workgroup owns one output element `(row, col)`;
- lanes accumulate strided F32 dot-product partials over `k`;
- `kernel.workgroup.reduce<addf>` reduces the partials;
- lane 0 writes the output element;
- all shape facts are JIT-configured through catalog metadata.

This is a coverage-grade baseline, not a final performance claim. Old HRX prior
art has specialized F32 batched cols1/cols16 variants that should become tuning
axes once Phase 1 coverage is closed.

## Evidence

Focused graph-op test file:

```text
cache/hrx2/phase1_0/route-slice-42-mul-mat-f32-f32-ops.txt
```

Focused CPU-reference validation:

```text
cache/hrx2/phase1_0/route-slice-42-mul-mat-f32-f32-focused-20260613-013821
```

Result:

- `3/3` focused rows passed against ggml CPU reference;
- HRX2 selected `mul_mat_f32_f32_moe_logits_k2048_r128_c1_512_wg256`;
- JIT compiled specializations for `cols=1`, `cols=16`, and `cols=64`;
- no provider failures.

Compile-report guardrails:

| Metric | Range |
| --- | ---: |
| HSACO bytes | `9208` |
| Schedule nodes | `89-94` |
| Code bytes | `476-500` |
| Spills | `0` |
| Private bytes | `0` |
| Local bytes | `32` |
| Peak live units | `14-16` |

Focused Qwen3 model smoke:

```text
cache/hrx2/phase1_0/route-slice-42-mul-mat-f32-f32-qwen3-smoke-20260613-013851
```

Result:

- decode, narrow, and prefill64 regimes passed;
- `288` dispatches of the new route across the three runs;
- no F32/F32 `MUL_MAT` fallback remained in the Qwen3 trace;
- no provider failures.

Full 11-model basket smoke:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-42-20260613-014041
```

Result: `33/33` passed across decode `p=1`, narrow `p=16`, and prefill64
`p=64`.

Aggregate:

- nodes: `506424`
- accelerated: `435628`
- host orchestration: `25140`
- infrastructure blocker: `32112`
- compute fallback: `13544`
- HRX20 compute: `467740`
- CPU compute: `13544`
- provider unavailable: `0`
- F32/F32 `MUL_MAT` route dispatches: `864`

Compared with slice 41, CPU compute fallbacks dropped from `18728` to `13544`.
F32/F32 `MUL_MAT` no longer appears in the aggregate fallback list.

## Remaining Backlog

Top remaining compute fallback families after this slice:

- ROPE with frequency source operand:
  `ROPE f32,i32,f32` at `128x32x{1,16,64}x1`;
- residual no-frequency ROPE variants at `nheads=8` and `nheads=32`;
- GLU/SWIGLU split coverage for widths
  `13824`, `14336`, `18944`, `21504`, and `32768`;
- GET_ROWS coverage for F32 and quantized state/embedding rows.

## Process Notes

- Generic `test-backend-ops -o MUL_MAT -p type_a=f32,...` produced no rows for
  this model-shaped case. The accepted validation used exact graph-op rows
  exported from the model basket.
- Keep the source target-neutral. This family uses no target-specific ISA
  primitive or ABI.
