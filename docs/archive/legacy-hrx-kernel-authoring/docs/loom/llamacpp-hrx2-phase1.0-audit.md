# HRX2 Phase 1.0 Coverage Audit

Date: 2026-06-13

## Latest Checkpoint: Route Slice 48

Slice 48 closes the final unexplained compute fallback from the Phase 1 basket:
the Llama 3.1 NORMAL frequency-source h32/p64 ROPE row.

```text
rope_normal_f32_freq_n128_d128_h32_t1_64_wg256
```

The original h32 frequency-source implementation recomputed each pair's theta
scale through an independent `exp(log(base) * exponent)` expression. That
compiled and passed decode/narrow rows but failed the strict ggml CPU-reference
gate at `ntokens=64`. Slice 48 changes the NORMAL frequency-source Loom root to
match the CPU recurrence: compute one `theta_scale = powf(freq_base, -2/n_dims)`
equivalent, then multiply `theta` forward once per pair before dividing by the
frequency-factor buffer.

Evidence:

```text
cache/hrx2/phase1_0/basket-smoke-phase1-current-20260613-043737
cache/hrx2/phase1_0/route-slice-48-rope-normal-h32-p64/focused-final-20260613-044740
cache/hrx2/phase1_0/route-slice-48-rope-normal-h32-p64/llama31-p64-smoke-20260613-044755
cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836
```

Focused CPU-reference replay covered `ntokens=1`, `16`, and `64` for the exact
h32 frequency-source row. All three rows selected
`rope_normal_f32_freq_n128_d128_h32_t1_64_wg256`, compiled successfully, and
passed.

Full 11-model coverage basket result: 33/33 passed.

Aggregate after slice 48, using the corrected graph-node-only reducer:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| HRX20 compute nodes | 481284 |
| CPU compute fallbacks | 0 |
| infrastructure blockers | 32112 |

Delta versus slice 47: `compute_fallback -192`, `HRX20 compute +192`.

`top_compute_fallbacks` is empty and `cpu_assigned_but_hrx_supported` is empty.
The remaining `infrastructure_blocker` rows are the deferred `SET_ROWS` host
orchestration path already classified outside Phase 1 unfused compute coverage.
Slice doc:

```text
docs/loom/llamacpp-hrx2-phase1.0-route-slice-48.md
```

## Previous Checkpoint: Route Slice 47

Slice 47 added quantized embedding `GET_ROWS` coverage and the scheduler
placement hook needed for CPU-seeded embedding gathers:

```text
get_rows_q4_k_f32_n2048_r1_512_wg256
get_rows_q4_k_f32_n4096_r1_512_wg256
get_rows_q4_k_f32_n5120_r1_512_wg256
get_rows_q5_k_f32_n3584_r1_512_wg256
get_rows_q6_k_f32_n2048_r1_512_wg256
get_rows_q6_k_f32_n3072_r1_512_wg256
get_rows_q6_k_f32_n5376_r1_512_wg256
get_rows_q8_0_f32_n4096_r1_512_wg256
```

Evidence:

```text
cache/hrx2/phase1_0/route-slice-47-get-rows-quant/focused-final-20260613-042127
cache/hrx2/phase1_0/route-slice-47-get-rows-quant/offload-hook-qwen-q4-p1-20260613-042034
cache/hrx2/phase1_0/basket-smoke-route-slice-47-offload-hook-20260613-042340
```

Full 11-model coverage basket result: 33/33 passed.

Aggregate after slice 47, using the corrected graph-node-only reducer:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| HRX20 compute nodes | 481092 |
| CPU compute fallbacks | 192 |
| infrastructure blockers | 32112 |

Delta versus slice 46: `compute_fallback -396`, `HRX20 compute +396`.

Remaining compute fallback is the deliberately unclaimed Llama 3.1 NORMAL
frequency-source h32/p64 ROPE row:

```text
ROPE f32 <- f32,i32,f32, shape 128x32x64x1, count 192
```

`cpu_assigned_but_hrx_supported` is empty after the offload hook. Slice doc:

```text
docs/loom/llamacpp-hrx2-phase1.0-route-slice-47.md
```

## Previous Checkpoint: Route Slice 46

Slice 46 added compact dense F32 `GET_ROWS` coverage:

```text
get_rows_f32_n2048_r1_64_wg256
get_rows_f32_n3072_r1_64_wg256
get_rows_f32_n3584_r1_64_wg256
get_rows_f32_n4096_r1_64_wg256
get_rows_f32_n5120_r1_64_wg256
get_rows_f32_n5376_r1_64_wg256
```

Evidence:

```text
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/focused-existing-exports-20260613-032516
cache/hrx2/phase1_0/route-slice-46-get-rows-f32/focused-phi4-3072-20260613-032546
cache/hrx2/phase1_0/basket-smoke-route-slice-46-20260613-032627
```

Full 11-model coverage basket result: 33/33 passed.

Aggregate after slice 46, using the corrected graph-node-only reducer:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| HRX20 compute nodes | 480696 |
| CPU compute fallbacks | 588 |
| infrastructure blockers | 32112 |

Delta versus slice 45: `compute_fallback -792`, `HRX20 compute +792`.

Remaining compute fallbacks are now the deliberately unclaimed Llama 3.1
NORMAL frequency-source h32/p64 ROPE row and quantized embedding `GET_ROWS`
for `q4_K`, `q5_K`, `q6_K`, and `q8_0` sources. Slice doc:

```text
docs/loom/llamacpp-hrx2-phase1.0-route-slice-46.md
```

## Previous Checkpoint: Route Slice 45

Slice 45 added Mistral NORMAL-mode no-frequency F32 ROPE coverage:

```text
rope_normal_f32_n128_d128_h8_t1_64_wg256
rope_normal_f32_n128_d128_h32_t1_64_wg256
```

Evidence:

```text
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/focused-20260613-025336
cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-smoke-20260613-025422
cache/hrx2/phase1_0/basket-smoke-route-slice-45-20260613-025611
```

Full 11-model coverage basket result: 33/33 passed.

Aggregate after slice 45, using the corrected graph-node-only reducer:

| Metric | Count |
| --- | ---: |
| graph nodes | 481284 |
| HRX20 compute nodes | 479904 |
| CPU compute fallbacks | 1380 |
| infrastructure blockers | 32112 |

Delta versus slice 44 re-reduced with the same graph-node-only reducer:
`compute_fallback -2880`, `HRX20 compute +2880`.

Remaining compute fallbacks are now the deliberately unclaimed Llama 3.1
NORMAL frequency-source h32/p64 ROPE row and GET_ROWS embedding lookups. Slice
doc:

```text
docs/loom/llamacpp-hrx2-phase1.0-route-slice-45.md
```

## Previous Checkpoint: Route Slice 43

Slice 43 added the Llama 3.1 normal-mode frequency-factor ROPE h32 decode and
narrow bucket:

```text
rope_normal_f32_freq_n128_d128_h32_t1_16_wg256
```

Evidence:

```text
cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/focused-split-t1-t16-20260613-015955
cache/hrx2/phase1_0/route-slice-43-rope-normal-h32/llama31-smoke-split-20260613-020010
cache/hrx2/phase1_0/basket-smoke-route-slice-43-20260613-020246
```

Full 11-model coverage basket result: 33/33 passed.

Aggregate after slice 43:

| Metric | Count |
| --- | ---: |
| graph nodes | 503494 |
| HRX20 compute nodes | 469852 |
| CPU compute fallbacks | 11432 |
| infrastructure blockers | 32112 |
| host orchestration | 22210 |

Remaining top compute fallbacks are now dominated by split `GLU`, no-frequency
ROPE, GET_ROWS, and the rejected h32/p64 normal-frequency ROPE row. The h32/p64
frequency-source route compiled but failed strict ggml CPU-reference tolerance,
so it remains deliberately unclaimed until numeric parity is fixed.

This is the checkpoint after the first phase 1.0 miniature sweep. The goal was
not to finish the catalog; it was to prove the route-admission loop on a small
set of unfused kernels before scaling to 10-15 more.

## Basket State

`tools/download_hrx2_model_basket.py --dry-run` reports the default
`coverage` basket as complete:

```text
Profile: coverage
Destination: shared/models/llamacpp-hrx2-basket-v1
Files: 11
Expected size: 116.07 GiB
```

All 11 coverage-profile GGUFs are present locally. The directory also contains
extra quant variants that can be used for follow-up sweeps.

## Accepted Route Surface

The current Phi-4 smoke selects these HRX2 routes:

| Family | Routes selected in smoke | Notes |
| --- | ---: | --- |
| `rms_norm_f32` | 2 | `n3072/r1` and `n3072/r16` exact-shape routes. |
| `add_f32` | 2 | `n3072/r1` and `n3072/r16` pointwise routes. |
| `mul_f32` | 2 | `n3072/r1` and `n3072/r16` pointwise routes. |
| `scale_f32` | 2 | `n128/r24` and `n128/r384` attention-scale routes. |
| `cont_f32` | 4 | `n128/r8`, `n128/r24`, `n128/r128`, and `n128/r384` row-contiguous copy routes. |

Focused exact-shape tests also accept these routes, even though the full
model smoke keeps them in a CPU island until neighboring quantized matmuls are
offloaded:

| Family | Focused routes selected | Notes |
| --- | ---: | --- |
| `swiglu_f32` | 2 | `n8192/r1` and `n8192/r16` packed, non-split, non-swapped SWIGLU routes from the Phi-4 fallback audit. |

`SET_ROWS` is still an infrastructure fallback, not accepted optimized kernel
coverage. GET_ROWS remains on CPU.

## Prior Evidence

Route slice 44 large split GLU coverage:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-44-20260613-current
```

All 33 basket runs passed. Current aggregate:

| Class | Count |
| --- | ---: |
| accelerated | 444912 |
| infrastructure blocker | 32112 |
| host orchestration | 7726 |
| compute fallback | 4260 |

Current compute backend counts:

| Backend | Count |
| --- | ---: |
| HRX20 | 477024 |
| CPU | 4260 |

Delta versus route slice 43: `compute_fallback -7172`, `HRX20 compute +7172`.
No GLU row remains in `top_compute_fallbacks`; the remaining top compute
fallbacks are normal ROPE and GET_ROWS shapes. Slice doc:

```text
docs/loom/llamacpp-hrx2-phase1.0-route-slice-44.md
```

Focused CONT correctness:

```text
cache/hrx2/phase1_0/cont-test-20260612-184111
```

Command:

```bash
LD_LIBRARY_PATH="$PWD/build/llama-hrx2/bin:$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}" \
  GGML_HRX2_TRACE_JSONL="$OUT/hrx2.jsonl" \
  GGML_HRX2_EVIDENCE_DIR="$OUT/evidence" \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o CONT -p 'type=f32' --output csv
```

Result: supported f32 CONT cases pass against ggml CPU reference. Unsupported
transpose/permuted layouts are deliberately rejected by the HRX2 support
predicate.

Latest model smoke:

```text
cache/hrx2/phase1_0/smoke-20260612-184121-phi4-p16n1-cont-clean
```

Command:

```bash
LD_LIBRARY_PATH="$PWD/build/llama-hrx2/bin:$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}" \
  GGML_HRX2_TRACE_JSONL="$RUN/hrx2.jsonl" \
  GGML_HRX2_EVIDENCE_DIR="$RUN/evidence" \
  GGML_SCHED_TRACE_JSONL="$RUN/sched.jsonl" \
  build/llama-hrx2/bin/llama-bench \
  -m shared/models/llamacpp-hrx2-basket-v1/bartowski__microsoft_Phi-4-mini-instruct-GGUF/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf \
  -p 16 -n 1 -b 64 -ub 64 -r 1 -o json --no-warmup -ngl 99 -dev HRX20
```

Dispatch counts:

| Count | Op | Route |
| ---: | --- | --- |
| 128 | `SET_ROWS` | `host_fallback_set_rows_f32_f16` |
| 67 | `RMS_NORM` | `rms_norm_f32_n3072_r1_vector_vw4_wg512` |
| 67 | `MUL` | `mul_f32_n3072_r1_wg256` |
| 66 | `ADD` | `add_f32_n3072_r1_wg256` |
| 63 | `RMS_NORM` | `rms_norm_f32_n3072_r16_vector_vw4_wg512` |
| 63 | `MUL` | `mul_f32_n3072_r16_wg256` |
| 62 | `ADD` | `add_f32_n3072_r16_wg256` |
| 32 | `SCALE` | `scale_f32_n128_r384_wg256` |
| 32 | `CONT` | `cont_f32_n128_r128_wg256` |
| 32 | `CONT` | `cont_f32_n128_r384_wg256` |
| 32 | `SCALE` | `scale_f32_n128_r24_wg256` |
| 32 | `CONT` | `cont_f32_n128_r8_wg256` |
| 32 | `CONT` | `cont_f32_n128_r24_wg256` |

There were no `provider_unavailable` events.

Scheduler reduction:

| Metric | Count |
| --- | ---: |
| graph nodes | 8136 |
| HRX20 compute nodes | 4248 |
| CPU compute fallbacks | 3888 |
| infrastructure blockers | 768 |

Compared with the pointwise-only smoke, `CONT` removed 768 CPU compute
fallbacks and raised HRX20 compute-node coverage by 768. The prompt sample got
slower, which is expected for tiny unfused copy dispatches before attention
fusions remove the surrounding traffic and launch overhead.

Coverage-basket smoke after removing the overbroad RMS_NORM route:

```text
cache/hrx2/phase1_0/basket-smoke-fixed-20260612-190821
```

Scope: 11 coverage GGUFs across decode (`p=1,n=1`), narrow (`p=16,n=1`),
and prefill64 (`p=64,n=1`) regimes.

Result:

| Status | Count |
| --- | ---: |
| Passed | 30 |
| Failed | 3 |

The only failures were all three regimes for
`Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`. The runs abort prompt decode with
`res = -3`. There are no `provider_unavailable` events; the decode trace shows
successful RMS_NORM and MUL dispatch, then the scheduler-assigned q8_0
`MUL_MAT` path is the next HRX2 compute node. This is now tracked in
`docs/loom/loom-bugs-limitations.md` as a q8_0 runtime/route-admission issue.

Follow-up Q8_0 isolation found the root cause in the HRX2 runtime support
predicate, not in Loom q8 codegen. The q8_0 `MUL_MAT` route still had
allocated-pointer non-overlap guards, which were true during pre-allocation
scheduler probing and false during allocated split execution. Removing those
guards fixed the model-level abort while preserving focused q8 route
correctness.

Clean production Q8_0 decode, narrow, and prefill64 smokes passed after the
fix:

```text
cache/hrx2/phase1_0/q8-three-regimes-after-overlap-fix-20260612-192957
```

Result:

| Regime | Prompt | Decode | Status |
| --- | ---: | ---: | ---: |
| decode | 1 | 1 | 0 |
| narrow | 16 | 1 | 0 |
| prefill64 | 64 | 1 | 0 |

Q8_0 scheduler reduction after the fix:

| Metric | Count |
| --- | ---: |
| graph nodes | 34666 |
| HRX20 compute nodes | 14272 |
| CPU compute fallbacks | 11288 |
| infrastructure blockers | 2304 |

Top remaining Q8_0 fallbacks are expected phase-1 work: larger q8_0 matmul
row buckets, GLU, ROPE, SOFT_MAX, f16 attention matmuls, and GET_ROWS.

Full coverage-basket smoke after the Q8_0 fix:

```text
cache/hrx2/phase1_0/basket-smoke-after-q8-overlap-fix-20260612-193159
```

Scope: 11 coverage GGUFs across decode (`p=1,n=1`), narrow (`p=16,n=1`),
and prefill64 (`p=64,n=1`) regimes.

Result:

| Status | Count |
| --- | ---: |
| Passed | 33 |
| Failed | 0 |

Coverage-basket scheduler reduction after the fix:

| Metric | Count |
| --- | ---: |
| graph nodes | 694186 |
| HRX20 compute nodes | 180864 |
| CPU compute fallbacks | 300420 |
| infrastructure blockers | 32112 |

Top remaining fallbacks in the basket are now dominated by `ADD`/`RMS_NORM`
for unrepresented hidden sizes, quantized `MUL_MAT`, attention `ROPE`,
`SOFT_MAX`, `GET_ROWS`, `MUL_MAT_ID`, and split-form `GLU`. The reducer also
reports supported-but-CPU-assigned `GLU` shapes for Llama 3.1 Q8_0; these are
kept in a CPU graph island by adjacent matmul fallbacks and should not be
counted as a route correctness failure.

## CONT Compile Report Guardrails

| Route | HSACO bytes | Code bytes | Inst | Spills | Private bytes | Peak live | Moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cont_f32_n128_r128_wg256` | 9176 | 132 | 29 | 0 | 0 | 6 | 1 |
| `cont_f32_n128_r24_wg256` | 9176 | 128 | 27 | 0 | 0 | 6 | 1 |
| `cont_f32_n128_r384_wg256` | 9176 | 148 | 31 | 0 | 0 | 6 | 1 |
| `cont_f32_n128_r8_wg256` | 9176 | 96 | 23 | 0 | 0 | 6 | 1 |

These are acceptable for coverage. Future copy kernels should still tune vector
width and row mapping before being called done-done for performance.

## SWIGLU Focused Evidence

The full Phi-4 smoke exposes `SWIGLU` f32 as a top fallback, but the scheduler
keeps it on CPU because the adjacent quantized matmuls are still CPU-owned.
The focused test-file path proves the exact model shapes independently.

Evidence:

```text
cache/hrx2/phase1_0/swiglu-exact-test-20260612-185715
```

Test-file records:

```text
95 0 8192 1 1 1 2 2 0 1 0 16384 1 1 1 0 0 0 0 swiglu_n8192_r1
95 0 8192 16 1 1 2 2 0 1 0 16384 16 1 1 0 0 0 0 swiglu_n8192_r16
```

Command:

```bash
LD_LIBRARY_PATH="$PWD/build/llama-hrx2/bin:$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}" \
  GGML_HRX2_TRACE_JSONL="$RUN/hrx2.jsonl" \
  GGML_HRX2_EVIDENCE_DIR="$RUN/evidence" \
  GGML_HRX2_DUMP_COMPILE_REPORT=1 \
  GGML_HRX2_DUMP_MANIFEST=1 \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 --test-file "$RUN/swiglu_exact_ops.txt" --output csv
```

Result: both rows passed focused ggml CPU-reference validation and selected
the intended exact routes:

| Shape | Route | Workgroups | WG size |
| --- | --- | ---: | ---: |
| `ncols=8192,nrows=1` | `swiglu_f32_n8192_r1_wg256` | 32 | 256 |
| `ncols=8192,nrows=16` | `swiglu_f32_n8192_r16_wg256` | 512 | 256 |

`test-backend-ops perf` with the same test file reported:

| Shape | Runs | Time | Traffic | Harness GB/s |
| --- | ---: | ---: | ---: | ---: |
| `n8192/r1` | 131072 | 7.94 us/run | 96 kB/run | 11.54 |
| `n8192/r16` | 114688 | 8.72 us/run | 1536 kB/run | 167.94 |

Compile-report guardrails:

| Route | HSACO bytes | Code bytes | Inst | Spills | Private bytes | Local bytes | Peak live | Global mem inst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `swiglu_f32_n8192_r1_wg256` | 9184 | 128 | 30 | 0 | 0 | 0 | 6 | 3 |
| `swiglu_f32_n8192_r16_wg256` | 9184 | 156 | 34 | 0 | 0 | 0 | 6 | 3 |

This is accepted as coverage, not as a final performance claim. The current
source uses one element per lane with explicit x/gate loads and one `siluf`.
That matches the one-pass activation prior and has no spills, but a later
done-done pass should evaluate vectorized load/store groupings and fusion with
the two producer matmuls.

Post-build validation also passed with the rebuilt binary:

```text
cache/hrx2/phase1_0/swiglu-exact-test-postbuild-20260612-190010
```

## Rejected Candidate

GET_ROWS f32 was attempted and rejected. The source was preserved at:

```text
cache/hrx2/phase1_0/rejected-get-rows/get_rows_f32.loom
```

Failure mode:

```text
TARGET/003: target 'amdgpu-rdna3' export 'hrx2_get_rows_f32'
config 'amdgpu.rdna3.core' rejected 'index.shli' address-width 'u32'
constraint 'amdgpu.address.u32' is not satisfied
```

Some `ncols=256` cases also produced numeric mismatches when they reached
execution. GET_ROWS routes were removed from the catalog and runtime route
discovery. See `docs/loom/loom-bugs-limitations.md`.

## Remaining Top Fallbacks

From the latest Phi-4 smoke:

| Count | Op | Source types | Shape |
| ---: | --- | --- | --- |
| 336 | `MUL_MAT` | `q4_K,f32` | `3072x1x1x1` |
| 240 | `MUL_MAT` | `q4_K,f32` | `3072x16x1x1` |
| 225 | `MUL_MAT` | `q4_K,f32` | `16384x1x1x1` |
| 224 | `MUL_MAT` | `q5_K,f32` | `5120x1x1x1` |
| 224 | `ROPE` | `f32,i32,f32` | `128x24x1x1` |
| 224 | `ROPE` | `f32,i32,f32` | `128x8x1x1` |
| 224 | `MUL_MAT` | `f16,f32` | `256x1x24x1` |
| 224 | `SOFT_MAX` | `f32,f32` | `256x1x24x1` |
| 224 | `MUL_MAT` | `f16,f32` | `128x1x24x1` |

The next phase 1 slice should either take the high-impact quantized matmuls
head-on or deliberately harvest the non-matmul attention path first:
`ROPE`, `SOFT_MAX`, and then GET_ROWS after the address/indexing issue is
resolved. `GLU/SWIGLU` is now represented as standalone focused coverage, but
it will remain visible as a model-level CPU fallback until neighboring matmuls
are offloaded or fused.

## Checkpoint Decision

Current catalog route count is 85, including the deliberately non-optimized
host-mediated SET_ROWS fallback rows:

```text
6 add_f32
3 argsort_f32_i32
3 clamp_f32
5 cont_f32
3 div_f32
3 get_rows_moe_weights_f32
5 mul_f32
10 mul_mat_q8_0_f32
32 rms_norm_f32
3 scale_f32
2 set_rows_f32
3 sum_rows_f32
7 swiglu_f32
```

The route-admission backplane is viable for continued phase 1 work:
target-neutral Loom source, embedded bytecode, JSON route metadata, runtime
JIT config specialization, focused ggml CPU-reference validation, compile
report capture, manifest capture, and HRX2 route traces have all been exercised
on multiple operation classes.

The Q8_0 abort is fixed and documented, and the 11-model decode/narrow/prefill64
basket now passes 33/33. The route-admission backplane is ready for the next
10-15-route phase-1 slice. The lesson remains part of the acceptance gate:
model-level failures are owned by the HRX2 implementation even when they look
environmental.

The first 15-route slice after this checkpoint is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-15.md`. It added RMS_NORM,
row-strided ADD, and split-SWIGLU routes; the full basket still passed 33/33
and CPU compute fallbacks dropped from 300420 to 264240.

The next accepted route slice is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-26.md`. It added eight
`ncols=128` RMS_NORM row buckets and three MoE pointwise layout routes:
RHS column-broadcast MUL for `2048x8` decode/narrow rows and row-strided ADD
for `2048x16`. The full basket still passed 33/33, CPU compute fallbacks
dropped from 264240 to 242502, and HRX20 compute-node ownership rose from
217044 to 238782. Infrastructure blockers remained unchanged at 32112.

The next accepted route slice is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-27.md`. It added 15 routes:
three MoE `DIV` RHS-column-broadcast rows, three MoE `CLAMP` rows, three
`SUM_ROWS` row-reduction rows, and six residual RMS_NORM rows for
`3072x64`, `3584x{1,16,64}`, and `4096x{16,64}`. Focused validation passed
15/15 with no provider failures, zero spills, and zero private memory. The full
basket still passed 33/33, CPU compute fallbacks dropped from 242502 to
238536, and HRX20 compute-node ownership rose from 238782 to 242748.
Infrastructure blockers remained unchanged at 32112.

The next accepted route slice is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-28.md`. It added six MoE
support routes: DESC `ARGSORT` for `128x{1,16,64}` and narrow MoE weight
`GET_ROWS` for `1x8x{1,16,64}`. Focused validation passed 6/6 with six
successful JIT compiles, six dispatches, no provider failures, zero spills,
zero private memory, and zero local memory. The full basket still passed 33/33.
CPU compute fallbacks stayed at 238536 and HRX20 compute-node ownership stayed
at 242748, because the full-model scheduler now reports `ARGSORT` and
`GET_ROWS` as `supported_by=HRX20,CPU` but still assigns the whole MoE island
to CPU behind CPU-only `MUL_MAT_ID` gate/up/down paths.

Important placement finding from route slice 28: the traced top-k/gather
support prefix is no longer unsupported, but it still cannot reduce model-level
fallback counts until `MUL_MAT_ID` coverage is added or the MoE graph island is
otherwise split. The accepted `ARGSORT` implementation is a rank-count no-LDS
fallback for small `ncols=128`; bitonic/LDS candidates remain blocked by a
dispatch-time GPU fault documented in `docs/loom/loom-bugs-limitations.md`.

The next accepted route slice is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-29.md`. It added 18 no-`src2`
NEOX F32 `ROPE` routes for `ncols=128`,
`nheads={4,8,16,28,32,40}`, and `ntokens={1,16,64}`. Focused validation
passed 18/18 with 18 successful JIT compiles, 18 dispatches, no provider
failures, zero spills, zero private memory, and zero local memory. The full
basket still passed 33/33, CPU compute fallbacks dropped from 238536 to
218232, and HRX20 compute-node ownership rose from 242748 to 263052.
Infrastructure blockers remained unchanged at 32112.

Important implementation finding from route slice 29: `scalar.powf<afn>` does
not currently lower to AMDGPU target-low. The accepted ROPE source spells
`pow(freq_base, exponent)` as `exp(log(freq_base) * exponent)` using
`scalar.logf<afn>` and `scalar.expf<afn>`, which passed focused ggml
CPU-reference validation. The limitation is recorded in
`docs/loom/loom-bugs-limitations.md`.

The next accepted route slice is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-30.md`. It added 15 F32
`SOFT_MAX` routes: 12 masked attention routes for `ncols=256` and
`nrows={24,28,32,40,384,448,512,640,1536,1792,2048,2560}`, plus three
unmasked MoE probability routes for `ncols=128` and `nrows={1,16,64}`.
Focused validation passed 6/6 masked/unmasked representative rows against
ggml CPU reference, with six successful JIT compiles, six dispatches, no
provider failures, zero spills, zero private memory, 32-64 bytes local memory,
and peak live units at 7-11. The full basket still passed 33/33, CPU compute
fallbacks dropped from 218232 to 202176, and HRX20 compute-node ownership rose
from 263052 to 279108. Infrastructure blockers remained unchanged at 32112.

Important placement finding from route slice 30: unmasked MoE `SOFT_MAX`
routes are now supported and pass focused validation, but the full basket still
assigns them to CPU because the upstream `ffn_moe_logits` producer remains in
a CPU-only `MUL_MAT_ID` island. This is not a softmax provider failure; it is a
route-placement dependency on MoE matmul coverage.

The next accepted route slice is documented in
`docs/loom/llamacpp-hrx2-phase1.0-route-slice-31.md`. It added a
target-neutral F16/F32 batched attention `MUL_MAT` family for the observed KQ
and KQV layouts: `k={128,256}`, `rows={128,256}`, `cols={1,16,64}`, attention
heads `{24,28,32,40}`, and grouped F16 source heads `{4,8,16}`. Focused
validation passed 12/12 exact graph-op rows against ggml CPU reference, with
12 successful JIT compiles, 12 dispatches, no provider failures, zero spills,
zero private memory, and peak live units at 11-20. The full basket still
passed 33/33, selected `mul_mat_f16_f32_batched_attention_wg256` 2676 times,
and left zero F16/F32 `MUL_MAT` compute fallbacks.

Important trace interpretation finding from route slice 31: moving attention
matmuls from CPU to HRX2 changes graph partitioning and removes a large amount
of CPU island/copy structure, so total scheduler-node counts are not a clean
apples-to-apples delta against route slice 30. The direct acceptance signals
are the absence of residual F16/F32 `MUL_MAT` fallbacks, the route dispatch
count, no provider failures, and 33/33 basket pass evidence.

Current remaining top fallback priorities after route slice 31:

| Priority | Families | Why |
| ---: | --- | --- |
| 1 | Q4_K/Q5_K/Q6_K `MUL_MAT` and `MUL_MAT_ID` | Dominates dense and MoE model fallback counts; `MUL_MAT_ID` keeps supported MoE top-k/gather/normalization/GLU support routes in CPU islands. |
| 2 | `ROPE` with `src2` frequency factors | Gemma/Phi-style ROPE ABI remains a separate CPU fallback family after no-`src2` NEOX coverage. |
| 3 | F32/F32 MoE logits and residual dense matmuls | Keeps narrow MoE support chains on CPU for some models. |
| 4 | Broader GLU/SWIGLU route coverage | Some focused GLU routes are accepted, but many dense model GLU shapes still report CPU-only support and should be covered after producer matmul route plans are clear. |
| 5 | Residual pointwise/layout gaps found by future traces | Current RMS residuals from slice 26 are covered; new rows should be admitted only from fresh fallback evidence. |

Next slice rules:

1. Implement the next 10-15 routes with the same acceptance gates:
   focused ggml CPU-reference tests, route traces, compile reports, and
   reduced basket evidence.
2. Run a short model smoke after each route cluster, then a full 11-model
   decode/narrow/prefill64 basket smoke before accepting the checkpoint.
3. Every rejected candidate or runtime blocker gets a concrete entry in
   `docs/loom/loom-bugs-limitations.md` before moving on.

Rejected candidates and runtime blockers must go into
`docs/loom/loom-bugs-limitations.md` with exact diagnostics. General feature
requests stay in `docs/loom/loom-author-feedback.md`.

Verification commands used for this checkpoint:

```bash
python3 sources/llama.cpp/ggml/src/ggml-hrx2/tools/validate_hrx2_catalog.py \
  --catalog build/llama-hrx2/ggml/src/ggml-hrx2/generated/catalog.json \
  --source-root sources/llama.cpp/ggml/src/ggml-hrx2
git -C sources/llama.cpp diff --check
git diff --check
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench -j$(nproc)
```

The coherent llama.cpp checkpoints are:

```text
be0c4b2bb hrx2: add softmax f32 routes
e2931a1bc hrx2: add rope neox f32 routes
7ae0bce4b hrx2: add moe argsort and weight gather routes
6a95a9ec5 hrx2: add moe support and residual rms routes
22432eccf hrx2: add phase one pointwise and rms routes
20db711cc hrx2: add phase one route slice
3d1f29c9f hrx2: fix q8 matmul graph execution
```
## 2026-06-13: Quantized Expert Matmul Loader And Indexed Coverage

Route slices 38 and 40 closed the Q4_K/Q5_K/Q6_K indexed expert matmul
coverage gap and fixed the loader-domain issue that left quantized weights in
CPU buffers despite runtime route support.

Latest full basket smoke:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-40-20260613-010859
```

Result: `33/33` passed across the 11-model decode/narrow/prefill64 basket.

Aggregate scheduler reduction:

- nodes: `516270`
- accelerated: `426182`
- host orchestration: `34986`
- infrastructure blocker: `32112`
- compute fallback: `22990`
- HRX20 compute: `458294`
- CPU compute: `22990`

Important shift: quantized `MUL_MAT_ID` is now accelerated in the aggregate
instead of dominating the fallback list. Remaining top compute fallbacks are
F32/F32 attention matmul, Q8_0 wider matmul, ROPE frequency-source variants,
GLU width coverage, and GET_ROWS rows.

Route slice 41 widened the existing target-neutral Q8_0 direct `MUL_MAT`
source and route domains for large dense rows and `k=14336`/larger reductions.
The full basket still passed `33/33` at:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-41-20260613-012314
```

Aggregate scheduler reduction:

- nodes: `515064`
- accelerated: `430444`
- host orchestration: `33780`
- infrastructure blocker: `32112`
- compute fallback: `18728`
- HRX20 compute: `462556`
- CPU compute: `18728`

Q8_0 direct `MUL_MAT` no longer appears in the aggregate fallback list. The
remaining top fallback priorities are F32/F32 matmul, ROPE frequency-source
variants, GLU/SWIGLU width coverage, and GET_ROWS rows.

Route slice 42 added target-neutral F32/F32 `MUL_MAT` coverage for Qwen3 MoE
logits, `k=2048`, `rows=128`, and `cols=1..512`. Focused ggml CPU-reference
validation passed exact graph-op rows for `cols={1,16,64}` with zero spills and
zero private memory. The full basket still passed `33/33` at:

```text
cache/hrx2/phase1_0/basket-smoke-route-slice-42-20260613-014041
```

Aggregate scheduler reduction:

- nodes: `506424`
- accelerated: `435628`
- host orchestration: `25140`
- infrastructure blocker: `32112`
- compute fallback: `13544`
- HRX20 compute: `467740`
- CPU compute: `13544`

The new route dispatched `864` times in the full basket, and F32/F32 `MUL_MAT`
no longer appears in the aggregate fallback list. Remaining top fallback
priorities are ROPE frequency-source variants, GLU/SWIGLU width coverage,
residual no-frequency ROPE variants, and GET_ROWS rows.
