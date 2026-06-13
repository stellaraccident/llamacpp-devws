# HRX2 Phase 1.0 Coverage Audit

Date: 2026-06-12

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

## Latest Evidence

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

Current accepted route count is 14:

```text
2 rms_norm_f32
2 add_f32
2 mul_f32
2 scale_f32
4 cont_f32
2 swiglu_f32
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
  --catalog sources/llama.cpp/ggml/src/ggml-hrx2/catalog.json \
  --source-root sources/llama.cpp/ggml/src/ggml-hrx2
git -C sources/llama.cpp diff --check
git diff --check
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench -j$(nproc)
```

The coherent llama.cpp checkpoints are:

```text
c501479c4 hrx2: add swiglu f32 route coverage
6f524f373 hrx2: remove unvalidated rms norm fallback route
3d1f29c9f hrx2: fix q8 matmul graph execution
```
