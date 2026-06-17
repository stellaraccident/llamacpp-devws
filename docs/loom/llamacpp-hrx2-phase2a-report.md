# HRX2 Phase 2a Throughput Report

Date: 2026-06-15

Phase 2a is using Vulkan as the same-machine performance target and `llama-bench`
plus HRX2 scheduler/route traces as the acceptance evidence. Current work is in
`sources/llama.cpp`; `hrx-system` remains on `main` with only local packaging
patches.

## Baseline

Initial slice:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag baseline-slice-20260615-102035 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64,prefill-p64n0,prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 600
```

Artifacts:

- `cache/hrx2/phase2a/baseline-slice-20260615-102035/summary.md`
- `cache/hrx2/phase2a/baseline-slice-20260615-102035/p512-sched-summary.md`

Baseline result: HRX2 was roughly 22x to 130x behind Vulkan on the selected
decode/prefill slice. p512 prefill had 2953 CPU compute fallback nodes across
the three-model basket. The biggest p512 route holes were RMS_NORM, ROPE,
masked SOFT_MAX, split SWIGLU, F32 GET_ROWS, and F16 attention MUL_MAT rows.

## Accepted Batch

Accepted llama.cpp route coverage:

- `RMS_NORM` p512 rows for hidden sizes 3072 and 4096.
- F32 `GET_ROWS` p512 rows for hidden sizes 3072 and 4096.
- NEOX+frequency ROPE p512 rows for Phi-4-mini h8/h24, d96.
- normal+frequency ROPE p512 rows for Llama h8/h24/h32, d128.
- split-source SWIGLU p512 rows for n8192 and the large n13824..32768 bucket.

Focused backend-op gate:

```bash
env ROCM_PATH=$PWD/rocm GGML_HRX_ROCM_PATH=$PWD/rocm \
  LD_LIBRARY_PATH=$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-} \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase2a/p512-c512-op-export-20260615-103625/basket-p512-c512-touched-route-ops.txt \
  --output csv
```

Result:

- `cache/hrx2/phase2a/p512-c512-op-export-20260615-103625/hrx2-touched-route-test.csv`
- 26 rows tested, 26 supported, 0 unsupported.

Post-fix p512 comparison:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag route-coverage-p512-comparison-20260615-103954 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 600
```

| Model | HRX2 Before | HRX2 After | Vulkan After | HRX2/Vulkan After |
| --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | 37.718 | 41.902 | 4357.440 | 0.0096 |
| `llama32-3b-q4` | 37.731 | 45.878 | 4894.633 | 0.0094 |
| `llama31-8b-q4` | 18.463 | 21.228 | 2243.944 | 0.0095 |

p512 scheduler reduction after the accepted batch:

- `cache/hrx2/phase2a/route-coverage-p512-comparison-20260615-103954/p512-sched-summary.md`
- CPU compute fallback: 2953 -> 1656.
- Remaining CPU fallback is attention-shaped F16 `MUL_MAT` plus masked
  `SOFT_MAX`.

## Accepted SET_ROWS Default

The refreshed Phase 2a slice showed that decode and p64 had zero CPU compute
fallback but still spent most visible time under stream synchronization. The
largest non-kernel issue was `SET_ROWS`: support probing admitted the op for
HRX2, but dispatch used the synchronized host fallback unless
`GGML_HRX2_ENABLE_SET_ROWS_LOOM=1` was set.

Current `hrx-system` main fixes the previous Loom f32->f16 SET_ROWS lowering
failure. Focused replay of exact exported Phi p512 SET_ROWS rows passed with
the Loom route selected:

```bash
build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase2a/set-rows-default-20260615-110043/set_rows_phi4_p512_ops.txt \
  --output csv
```

Result:

- `cache/hrx2/phase2a/set-rows-default-20260615-110043/test.csv`
- 4 rows tested, 4 supported, 0 failures.
- HRX2 route trace selected `set_rows_f32_f16_generic` for all rows.

llama.cpp now defaults to the Loom SET_ROWS route when a provider is available
and keeps `GGML_HRX2_DISABLE_SET_ROWS_LOOM=1` as the old host-fallback escape
hatch.

Measured effect on the three-model slice before changing the default, using
`GGML_HRX2_ENABLE_SET_ROWS_LOOM=1`:

| Model | Case | HRX2 base | HRX2 SET_ROWS Loom | Speedup | Vulkan | New HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 1.447 | 1.815 | 1.254 | 124.045 | 0.0146 |
| `phi4-mini-q4` | `prefill-p64n0` | 33.054 | 35.330 | 1.069 | 1518.318 | 0.0233 |
| `phi4-mini-q4` | `prefill-p512n0` | 42.987 | 46.345 | 1.078 | 4316.991 | 0.0107 |
| `llama32-3b-q4` | `decode-p1n64` | 2.030 | 2.679 | 1.319 | 140.856 | 0.0190 |
| `llama32-3b-q4` | `prefill-p64n0` | 37.419 | 41.720 | 1.115 | 1702.306 | 0.0245 |
| `llama32-3b-q4` | `prefill-p512n0` | 46.360 | 49.962 | 1.078 | 4822.783 | 0.0104 |
| `llama31-8b-q4` | `decode-p1n64` | 1.946 | 2.610 | 1.341 | 91.342 | 0.0286 |
| `llama31-8b-q4` | `prefill-p64n0` | 20.024 | 20.872 | 1.042 | 1151.927 | 0.0181 |
| `llama31-8b-q4` | `prefill-p512n0` | 21.500 | 22.268 | 1.036 | 2257.110 | 0.0099 |

No-env smoke after the default flip:

- `cache/hrx2/phase2a/setrows-default-smoke-20260615/`
- Phi decode/p64/p512 selected `set_rows_f32_f16_generic`; the host fallback
  route no longer appears.

Full no-env HRX2 rerun after the default flip:

- `cache/hrx2/phase2a/setrows-default-full-20260615/`
- `host_fallback_set_rows_f32_f16`: 0 in decode, p64, and p512.
- `set_rows_f32_f16_generic`: 11960 decode dispatches and 184 dispatches in
  each prefill regime across the three-model slice.

## Accepted Q4_K FFN SWIGLU Fusions

The first Phase 2a hero-fusion pass landed two Q4_K/F32 FFN epilogue routes in
llama.cpp:

- Split gate/up path for Llama-style graphs:
  `MUL_MAT(Q4_K,F32) + MUL_MAT(Q4_K,F32) + split SWIGLU`.
- Packed path for Phi-style graphs:
  `MUL_MAT(Q4_K,F32, rows=2*out) + packed SWIGLU`.

Both routes are authored in one Loom source:

- `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_swiglu_f32.loom`

The C++ graph walker fuses only single-consumer local subgraphs, keeps the
unfused fallback, and exposes `GGML_HRX2_DISABLE_Q4K_SWIGLU_FUSION=1` for A/B
and rollback. The routes are target-generic and specialized by JIT config for
`k`, output rows, cols, and workgroup size.

Focused gates:

- Build: `cmake --build build/llama-hrx2 --target ggml-hrx2 llama-bench llama-cli test-backend-ops`
- Backend op gate:
  `cache/hrx2/phase2a/test-backend-ops-q4-swiglu-final-20260615-113527/`
  - `MUL_MAT`: 49 supported cases passed, 1277 expected unsupported variants.
  - `SWIGLU`: 1 supported case passed, 23 expected unsupported variants.
- Standalone configured Loom compile for the split route:
  `cache/hrx2/phase2a/q4k-swiglu-standalone-manifest-20260615-112352/`
  - Manifest ABI: 4 bindings, 4 parameters, 0 constants.
  - Compile report: 0 spills, peak live units 45, 9224-byte HSACO.

Manual same-binary A/B evidence:

| Model/path | Case | Fused | Unfused | Speedup | Dispatch delta |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama 3.2 split | p64 | 43.765 tok/s | 42.979 tok/s | 1.018x | -56 |
| Llama 3.2 split | p512 | 53.163 tok/s | 51.292 tok/s | 1.036x | -56 |
| Llama 3.2 split | p1/n64 | 2.733 tok/s | 2.692 tok/s | 1.015x | -3640 |
| Phi packed | p64 | 36.668 tok/s | 35.928 tok/s | 1.021x | -32 |
| Phi packed | p512 | 47.470 tok/s | 45.199 tok/s | 1.050x | -32 |

Standard three-model HRX2 slice after both fusions:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag q4k-swiglu-fusions-20260615 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64,prefill-p64n0,prefill-p512n0 \
  --backends hrx2 --repetitions 1 --timeout 900
```

Artifact: `cache/hrx2/phase2a/q4k-swiglu-fusions-20260615/`

Compared to `setrows-default-full-20260615`:

| Model | Case | Before | After | Speedup |
| --- | --- | ---: | ---: | ---: |
| `llama31-8b-q4` | `decode-p1n64` | 2.609 | 2.674 | 1.025x |
| `llama31-8b-q4` | `prefill-p64n0` | 20.856 | 21.548 | 1.033x |
| `llama31-8b-q4` | `prefill-p512n0` | 22.239 | 22.786 | 1.025x |
| `llama32-3b-q4` | `decode-p1n64` | 2.684 | 2.727 | 1.016x |
| `llama32-3b-q4` | `prefill-p64n0` | 41.484 | 42.328 | 1.020x |
| `llama32-3b-q4` | `prefill-p512n0` | 49.815 | 50.727 | 1.018x |
| `phi4-mini-q4` | `decode-p1n64` | 1.811 | 1.832 | 1.011x |
| `phi4-mini-q4` | `prefill-p64n0` | 35.389 | 35.919 | 1.015x |
| `phi4-mini-q4` | `prefill-p512n0` | 44.976 | 47.799 | 1.063x |

Interpretation: this is a valid, accepted fusion pattern and useful dispatch
elimination. It is not the Phase 2a bulk lift by itself. HRX2 remains orders of
magnitude behind Vulkan because the dominant boulders are still quantized
matmul throughput and attention, not just standalone epilogue dispatch count.

## Accepted Runtime Interop Fix: Disable GET_ROWS Op-Offload

HRX2 had a scheduler interop bug that made decode look artificially
CPU-bound. The backend advertised `offload_op` for every supported `GET_ROWS`.
llama.cpp intentionally keeps `token_embd.weight` on CPU, so the scheduler
offloaded the embedding lookup to HRX2 and copied the full CPU-resident
embedding table into the HRX split input every decode graph.

Evidence on Llama 3.2 3B Q4_K, `p0 n16`, `-ngl 99`, default `op_offload`:

| Variant | tok/s | CPU split-input bytes | Large split inputs |
| --- | ---: | ---: | ---: |
| Before | 3.018 | ~5.49 GB | 17 copies of `token_embd.weight` at 323 MB each |
| `-nopo 1` A/B | 30.002 | not traced | n/a |
| Patched default | 29.590 | 348 KB | 0 |

Artifacts:

- Before: `cache/hrx2/phase2a/cpu-underfeed-diagnostic-20260615-114026/`
- `-nopo 1` A/B: `cache/hrx2/phase2a/no-op-offload-ab-20260615-114713/`
- Patched default: `cache/hrx2/phase2a/offload-policy-fix-20260615-114809/`
- Three-model slice: `cache/hrx2/phase2a/offload-policy-fix-20260615/`

Compared to the prior Q4_K SWIGLU fusion baseline:

| Model | Case | Before | After | Speedup |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 1.832 | 14.016 | 7.65x |
| `phi4-mini-q4` | `prefill-p64n0` | 35.919 | 50.200 | 1.40x |
| `phi4-mini-q4` | `prefill-p512n0` | 47.799 | 50.975 | 1.07x |
| `llama32-3b-q4` | `decode-p1n64` | 2.727 | 15.971 | 5.86x |
| `llama32-3b-q4` | `prefill-p64n0` | 42.328 | 53.651 | 1.27x |
| `llama32-3b-q4` | `prefill-p512n0` | 50.727 | 52.799 | 1.04x |
| `llama31-8b-q4` | `decode-p1n64` | 2.674 | 10.299 | 3.85x |
| `llama31-8b-q4` | `prefill-p64n0` | 21.548 | 23.685 | 1.10x |
| `llama31-8b-q4` | `prefill-p512n0` | 22.786 | 23.059 | 1.01x |

The fix is to match HRX1's conservative runtime contract and stop providing
`offload_op` in HRX2. CUDA and Vulkan also avoid this trap by returning zero
batch size for `GET_ROWS` in their op-offload policy; CANN explicitly excludes
`GET_ROWS`. Future HRX2 host-weight offload should require a gather path that
does not materialize the full CPU source tensor as a recurring split input.

## Accepted Runtime Cleanup: HRX2 Full-Offload Embedding Placement

After the `offload_op` fix, Phase 2a basket traces still showed 6-12 CPU
compute nodes per run. These were supported quantized embedding `GET_ROWS`
nodes assigned to CPU because llama.cpp normally keeps `token_embd.weight` on
CPU.

The accepted cleanup is not to re-enable `offload_op`. HRX2 now places the
input embedding table on the first HRX2 device when the run is already full
offload (`-ngl` covers all layers). This loads the embedding weight onto HRX2
once with the rest of the model instead of copying the full CPU table as a
split input every graph.

Evidence on Llama 3.2 3B Q4_K:

| Case | Before tok/s | After tok/s | CPU compute before | CPU compute after |
| --- | ---: | ---: | ---: | ---: |
| `decode-p1n64` | 40.713 | 41.360 | 12 | 0 |
| `prefill-p64n0` | 76.879 | 78.093 | 6 | 0 |

Artifact:

- `cache/hrx2/phase2a/hrx2-input-embd-smoke-20260615/`

This is a correctness/interop cleanup, not a bulk throughput lift. The next
Phase 2a boulders remain decode dispatch/runtime backpressure and prompt
matmul/attention quality.

## Runtime Backpressure Finding: Decode Split Inputs

After CPU compute fallback reached zero on the Llama 3.2 3B Q4_K smoke,
decode still ran at only about `41.4 tok/s` versus Vulkan at about
`140.9 tok/s`. The scheduler trace shows why small models remain sensitive to
runtime overhead:

- `260` HRX2 scheduler splits for `65` decode graphs, or four splits per token.
- `455` split-input copies from CPU leaf tensors into HRX2 copies.
- `725` traced `hrx_stream_synchronize` calls totaling about `1.01 s` inside
  a `1.55 s` benchmark row.

The split inputs are tiny but numerous:

- `inp_tokens` for embedding `GET_ROWS`.
- position leaves for `ROPE`.
- KV row index leaves for `SET_ROWS`.
- `attn_inp_kq_mask` for attention `SOFT_MAX`.

An opt-in scheduler experiment that assigned all graph inputs to the first
backend was rejected. It removed some split copies in the assignment trace, but
llama.cpp's KV input setters assert host buffers and fill them directly, so the
run aborted in `llama_kv_cache::set_input_k_idxs`.

Conclusion: do not solve this with a blind global scheduler placement change.
The near-term HRX2-scoped path is to remove hot CPU leaf dependencies through
specialized routes/fusions: e.g. ROPE/SET_ROWS decode variants that take
position and KV row as config/kernarg, and attention routes that generate the
common causal mask internally. A broader llama.cpp fix would require changing
graph input setters to use backend tensor set APIs for selected device-resident
inputs.

## Accepted Runtime Interop Fix: HRX1-Style Submit Batching

HRX1 had a measured stream-submission policy that flushed queued dispatches by
real dispatch count and matmul-byte progress. HRX2 only flushed at graph end.
That left decode looking CPU-heavy even after the embedding offload bug was
fixed, because thousands of tiny decode dispatches were not being submitted to
the runtime with the same cadence as HRX1.

llama.cpp HRX2 now ports the HRX1 policy under HRX2-specific knobs:

- `GGML_HRX2_DISPATCHES_PER_SUBMIT`, default `12`.
- `GGML_HRX2_MAX_MUL_MAT_BYTES_PER_SUBMIT`, default `100000000`.
- `GGML_HRX2_DISABLE_SUBMIT_BATCHING=1` for A/B and rollback.

Focused backend-op gate after the runtime patch:

- `cache/hrx2/phase2a/submit-batching-op-tests-20260615-120924/`
- `MUL_MAT`: 49/49 passed.
- `RMS_NORM`: 10/10 passed.
- `SET_ROWS`: 33/33 passed.

Same-binary decode A/B on Llama 3.2 3B Q4_K, `p1 n64`, with order reversed to
rule out run ordering:

| Variant | tok/s | Dispatches | Submit flushes | Stream sync time |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 16.067 | 36725 | 0 | 1.489 s |
| Enabled | 22.690 | 36725 | 6045 | 0.124 s |

Artifacts:

- `cache/hrx2/phase2a/submit-batching-ab-20260615-120956/`
- `cache/hrx2/phase2a/submit-batching-ab-reverse-20260615-121032/`

Three-model decode slice after submit batching:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag submit-batching-three-model-decode-20260615-121056 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64 \
  --backends hrx2 --repetitions 1 --timeout 1200
```

Compared to `phase2a-vulkan-hrx2-baseline-20260615-115355`:

| Model | Before | After | Speedup | Dispatches | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | 14.043 | 19.374 | 1.380x | 41925 | 12 |
| `llama32-3b-q4` | 15.961 | 22.270 | 1.395x | 36725 | 12 |
| `llama31-8b-q4` | 10.276 | 17.868 | 1.739x | 41925 | 12 |

Interpretation: submit batching is a real decode runtime fix, not a kernel
quality fix. It reduces synchronization cost and raises decode throughput, but
HRX2 is still far behind Vulkan because decode remains dominated by many tiny
dispatches and weak quantized matmul/attention/fusion coverage.

## Rejected Batch

Masked p512 SOFT_MAX route coverage was tested and rejected:

- The exact c512 focused gate passed after adding an ncols=512 masked softmax
  route.
- Phi p512 smoke regressed from 42.733 tok/s with softmax on CPU to
  36.870 tok/s with softmax offloaded.
- The route was removed. Current evidence says the standalone p512 softmax
  kernel is not a bulk-lift fix; it needs a faster kernel or an attention
  fusion path.

Artifacts:

- `cache/hrx2/phase2a/route-coverage-smoke-20260615-103535/`
- `cache/hrx2/phase2a/route-coverage-softmax512-smoke-20260615-103731/`

The p512 F16 attention route was later corrected and accepted after using the
real graph trace shapes instead of guessed nominal shapes:

- `kq` needed `rows=512` and `cols={1,16,512}`.
- masked softmax needed `ncols=512` and `nrows=24..16384`, with `wg512`
  because the Loom kernel maps one workitem to one softmax column.
- `kqv` needed `k=512`, `rows=128`, and `cols={1,16,512}`.
- The route metadata now widens the generic F16/F32 batched attention route to
  `k<=512`, `rows<=512`, `cols<=512`, and adds a target-generic masked
  `soft_max_f32_mask_n512_r24_16384_wg512` route.

Evidence:

| Model | p512 HRX2 before | p512 HRX2 after | Speedup | Vulkan after | HRX2/Vulkan | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | 50.566 | 54.572 | 1.079x | 4326.749 | 0.0126 | 582 -> 6 |
| `llama32-3b-q4` | 52.674 | 59.352 | 1.127x | 4855.719 | 0.0122 | 510 -> 6 |
| `llama31-8b-q4` | 23.066 | 24.256 | 1.052x | 2228.412 | 0.0109 | 582 -> 6 |

Artifacts:

- `cache/hrx2/phase2a/p512-attn-route-smoke-20260615-122138/` shows the
  intermediate state where `kq` and softmax moved to HRX2 but `kqv` remained on
  CPU because `k=512` was still outside the route domain.
- `cache/hrx2/phase2a/p512-attn-k512-smoke-20260615-122235/` shows the
  corrected Llama 3.2 3B route with attention CPU fallback removed.
- `cache/hrx2/phase2a/p512-attn-k512-three-model-20260615-122321/` shows the
  three-model p512 basket slice.
- `cache/hrx2/phase2a/attn-route-regression-slice-20260615-122440/` shows no
  decode regression and no p64 prefill regression on Llama 3.2 3B.

Interpretation: this is an important coverage fix because p512 attention no
longer falls back to CPU, but it is not enough for the Phase 2a bulk lift. The
p512 basket remains about 1.1% to 1.3% of Vulkan. The next boulders are
standalone kernel quality and/or fusion for Q4_K prompt matmul and attention,
not more small route holes.

## Trace Measurement Fix

The Phase 2a comparison harness records `GGML_HRX2_TRACE_JSONL` and
`GGML_SCHED_TRACE_JSONL` so that every timed run has route and scheduler
evidence. The old trace writers reopened the JSONL file for each event, which
made decode measurements artificially slow because a single 64-token decode
run can produce more than 160k HRX2 trace events plus scheduler events.

Llama 3.2 3B Q4_K_M A/B before the fix:

| Case | No trace | Traced before | Traced after |
| --- | ---: | ---: | ---: |
| `decode-p1n64` | 40.606 | 22.683 | 39.471 |
| `prefill-p512n0` | 60.934 | 59.699 | 59.452 |

Artifacts:

- `cache/hrx2/phase2a/no-trace-ab-20260615-123304/`
- `cache/hrx2/phase2a/trace-ab-20260615-123335/`
- `cache/hrx2/phase2a/trace-writer-patched-ab-20260615-123542/`

Interpretation: prior traced decode numbers were not valid for runtime
throughput comparisons. After keeping trace files open for the process
lifetime, traced decode is within about 2% of no-trace. Prefill was not
materially affected because it has far fewer dispatch/trace events and is
dominated by device kernel time.

Corrected three-model HRX2/Vulkan baseline:

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | Dispatches | CPU compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 34.275 | 126.461 | 0.2710 | 41925 | 12 |
| `llama32-3b-q4` | `decode-p1n64` | 38.749 | 141.285 | 0.2743 | 36725 | 12 |
| `llama31-8b-q4` | `decode-p1n64` | 18.210 | 90.466 | 0.2013 | 41925 | 12 |
| `phi4-mini-q4` | `prefill-p64n0` | 53.139 | 1548.226 | 0.0343 | 645 | 6 |
| `llama32-3b-q4` | `prefill-p64n0` | 55.616 | 1679.582 | 0.0331 | 565 | 6 |
| `llama31-8b-q4` | `prefill-p64n0` | 24.226 | 1205.813 | 0.0201 | 645 | 6 |
| `phi4-mini-q4` | `prefill-p512n0` | 54.092 | 4276.348 | 0.0126 | 645 | 6 |
| `llama32-3b-q4` | `prefill-p512n0` | 58.753 | 4813.919 | 0.0122 | 565 | 6 |
| `llama31-8b-q4` | `prefill-p512n0` | 24.278 | 2243.780 | 0.0108 | 645 | 6 |

Artifact: `cache/hrx2/phase2a/trace-writer-patched-full-20260615-123632/`.

## Deferred Fusion: RMS_NORM -> MUL

An opt-in `rms_norm_mul_f32` Loom route was added and tested for the common
norm-weight row-broadcast pattern. It fuses the RMS reduction with the
following pointwise weight multiply and is gated by:

```bash
GGML_HRX2_ENABLE_RMS_NORM_MUL_FUSION=1
```

Focused gates passed:

- `RMS_NORM`: 10/10
- `MUL`: 20/20
- Artifact: `cache/hrx2/phase2a/trace-fix-rmsmul-gates-20260615-124739/`

Llama 3.2 3B Q4_K_M p1/n64 A/B:

| Mode | Tok/s | Dispatches | Fused dispatches |
| --- | ---: | ---: | ---: |
| disabled | 40.251 | 36725 | 0 |
| enabled | 39.892 | 33020 | 3705 |

Llama 3.2 3B Q4_K_M p512/n0 A/B:

| Mode | Tok/s | Dispatches | Fused dispatches |
| --- | ---: | ---: | ---: |
| disabled | 60.997 | 565 | 0 |
| enabled | 60.025 | 508 | 57 |

Artifact: `cache/hrx2/phase2a/rms-norm-mul-fusion-ab-20260615-124513/`.

Interpretation: the graph/fusion control plane works and reduces dispatch
count, but this is not a Phase 2a boulder because it did not improve t/s. Keep
it disabled by default. The corrected baseline points at larger issues:
decode still has many graphs/synchronizations and hero kernels, while prefill
is dominated by Q4_K prompt matmul and attention/fusion quality.

## Accepted Q4_K Prompt Cols4 Route

The first Q4_K prompt-matmul bulk lift keeps the existing F32 RHS path but
computes four prompt columns per workgroup. This reuses each Q4_K block
dequantization across four RHS columns instead of repeating it once per column.
The old one-column route remains the decode fallback and handles non-multiple
column counts.

Accepted route:

- `mul_mat_q4_k_f32_cols4_k256_32768_c4_512_wg256`
- Loom export: `hrx2_mul_mat_q4_k_f32_cols4_static`
- Guard: `k_multiple_of=256`, `cols_multiple_of=4`, `cols>=4`

Focused validation:

- Odd-column fallback probe, `cols=5`: passed and selected the old direct
  route without attempting the cols4 JIT.
- Model-derived Q4_K test file:
  `cache/hrx2/phase1_0/route-slice-32-q4-focused-current/mul_mat_q4_k_f32_ops.txt`
  passed. Provider trace selected direct for `cols=1` and cols4 for
  `cols=16/64`.

Full compact HRX2/Vulkan comparison:

| Model | Case | Previous HRX2 tok/s | New HRX2 tok/s | New Vulkan tok/s | HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `prefill-p64n0` | 53.139 | 64.722 | 1512.107 | 0.0428 |
| `phi4-mini-q4` | `prefill-p512n0` | 54.092 | 66.609 | 4331.100 | 0.0154 |
| `llama32-3b-q4` | `prefill-p64n0` | 55.616 | 75.264 | 1645.145 | 0.0457 |
| `llama32-3b-q4` | `prefill-p512n0` | 58.753 | 80.709 | 4882.947 | 0.0165 |
| `llama31-8b-q4` | `prefill-p64n0` | 24.226 | 32.052 | 1211.771 | 0.0265 |
| `llama31-8b-q4` | `prefill-p512n0` | 24.278 | 32.222 | 2249.823 | 0.0143 |

Decode stayed on the direct Q4_K route and remained in the same broad band:
`llama32-3b-q4` measured 39.504 tok/s HRX2 versus 140.567 tok/s Vulkan, and
`llama31-8b-q4` measured 18.287 tok/s HRX2 versus 90.769 tok/s Vulkan.

Artifact: `cache/hrx2/phase2a/q4k-cols4-guarded-full/`.

Interpretation: this is a real structural prompt improvement, especially on
small and mid-size Q4_K models, but it is still not the final Q4_K solution.
The route proves that avoiding repeated Q4_K dequantization matters. Closing
the remaining order-of-magnitude prefill gap requires the larger Vulkan/HRX1
prior-art path: quantize/pack RHS to Q8_1 scratch and run an MMQ-class tiled
Q4_K x Q8_1 prompt matmul instead of direct Q4_K x F32 RHS.

## Q8_1 RHS Backplane Smoke

The next Q4_K prompt experiment added the runtime backplane needed for packed
RHS work:

- a target-generic Loom `quantize_q8_1` provider;
- per-backend device scratch allocation for the packed RHS;
- route metadata and JIT config plumbing for `shape.q8_1.blocks`,
  `shape.q8_1.ne1`, and `shape.q8_1.z_count`;
- an opt-in Q4_K x Q8_1 direct-dot route guarded by
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`.

Focused op validation passed with the opt-in path selected:

- `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/serial2/`
- `test-backend-ops -b HRX20 -o MUL_MAT --output csv`
- trace contains both `quantize_q8_1_f32_generic_wg32` and
  `mul_mat_q4_k_q8_1_f32_direct_k256_32768_c1_512_wg256` dispatches.

Same-binary Llama 3.2 3B Q4_K_M p64 smoke:

| Variant | tok/s | Dispatches | Q8_1 dispatches | Q4_K cols4 dispatches |
| --- | ---: | ---: | ---: | ---: |
| Default F32-RHS cols4 | 77.000 | 566 | 0 | 112 |
| Opt-in direct Q8_1 RHS | 69.187 | 678 | 224 | 0 |

Artifacts:

- Default:
  `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/llama32-3b-q4-p64-default/`
- Opt-in:
  `cache/hrx2/phase2a/q8-backplane-20260615/fixed-quantizer/llama32-3b-q4-p64-optin/`

Interpretation: this proves the production control plane for scratch-backed
RHS conversion and multi-dispatch route selection, but it is not a performance
route. A direct Q4_K x Q8_1 dot kernel adds one quantize dispatch per prompt
matmul and does not reuse the packed RHS enough to beat the accepted F32-RHS
cols4 route. Leave it opt-in. The next Q4_K bulk-lift target must be the real
packed/MMQ schedule: quantize/pack RHS once per prompt tile, reuse it across
multiple rows/columns, and reduce Q4_K decode/dequant traffic substantially.

## Experimental Q4_K Q8_1 Prompt Cols4 Route

The direct Q8_1 backplane was extended with an opt-in cols4 prompt route:

- `mul_mat_q4_k_q8_1_f32_cols4_k256_32768_c4_512_wg256`
- Loom export: `hrx2_mul_mat_q4_k_q8_1_f32_cols4_static`
- Guard: `k_multiple_of=256`, `cols_multiple_of=4`, and the existing
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1` opt-in gate.

Focused backend-op gate:

- Artifact: `cache/hrx2/phase2a/q4-q8-cols4-20260615-160225/op-gate/`
- `test-backend-ops -b HRX20 -o MUL_MAT --output csv`
- Result: 1326 rows tested, no failures.
- Trace selected the new Q8_1 cols4 route on 12 prompt-shaped cases and the
  Q8_1 quantizer on 30 cases.

Same-binary three-model prefill A/B:

| Model | Case | Default F32 cols4 | Q8_1 cols4 opt-in | Speedup | Dispatch delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `prefill-p64n0` | 62.115 | 59.141 | 0.952x | +48 |
| `phi4-mini-q4` | `prefill-p512n0` | 65.356 | 65.125 | 0.996x | +48 |
| `llama32-3b-q4` | `prefill-p64n0` | 73.664 | 71.503 | 0.971x | +112 |
| `llama32-3b-q4` | `prefill-p512n0` | 79.973 | 82.557 | 1.032x | +112 |
| `llama31-8b-q4` | `prefill-p64n0` | 31.103 | 31.855 | 1.024x | +128 |
| `llama31-8b-q4` | `prefill-p512n0` | 31.936 | 33.512 | 1.049x | +128 |

Artifacts:

- Default: `cache/hrx2/phase2a/q4-q8-cols4-full-default-20260615-1605/`
- Opt-in: `cache/hrx2/phase2a/q4-q8-cols4-full-optin-20260615-1605/`

Interpretation: this is a better Q8_1 backplane probe than the one-column
variant and can win larger Llama prompt shapes, but it is still not a
production default. The extra quantize dispatch per Q4_K prompt matmul erases
or reverses the dot4/RHS-bandwidth gain on smaller prompts and Phi. Preserve it
as opt-in evidence only. The next Q4_K bulk-lift candidate must be a true MMQ
tile that reuses the Q8_1 RHS and Q4_K dequant across a larger row/column
tile instead of merely replacing F32 cols4 with Q8_1 cols4.

## Rejected Q4_K Decode Rows2 Variant

A Q4_K decode experiment tried to reduce direct-route workgroup count by
computing two output rows per workgroup for `cols=1` and even row counts. This
was motivated by decode dispatch volume, but it was not accepted.

Focused gate:

- `test-backend-ops -b HRX20 -o MUL_MAT --output csv`
- Artifact:
  `cache/hrx2/phase2a/q4k-rows2-decode-20260615/op-gate/`
- Result: passed correctness and selected
  `mul_mat_q4_k_f32_rows2_k256_32768_c1_wg256` for matching Q4_K rows.

Model smoke on Llama 3.2 3B Q4_K_M, `p1 n64`:

| Variant | tok/s | Dispatches |
| --- | ---: | ---: |
| Baseline direct Q4_K decode | 41.370 | 36790 |
| Rows2 Q4_K decode candidate | 9.124 | 36790 |

Artifact:
`cache/hrx2/phase2a/q4k-rows2-decode-20260615/llama32-3b-decode/default/`.

After removing the candidate, the same small decode smoke returned to the
expected direct-route band:

- `cache/hrx2/phase2a/post-q4-rows2-revert-smoke-20260615/`
- `41.935 tok/s`, `36790` dispatches, `0` CPU compute fallback.
- Route trace selected `mul_mat_q4_k_f32_direct_k256_32768_c1_512_wg256`
  `7280` times and had no rows2 route.

Interpretation: reducing per-route workgroups did not reduce graph dispatches
and made the kernel much slower. The likely loss is lower occupancy/scheduling
quality from doing two independent row reductions in one workgroup without a
matching data-reuse win. Do not reintroduce rows-per-workgroup decode routes
based only on workgroup-count arithmetic; require same-binary model throughput
and route trace evidence. For decode, useful work should target real dispatch
elimination/fusion, better direct-dot schedules, and tiny dynamic-input
backpressure. For prefill, the larger boulder remains Q4_K packed/MMQ prompt
matmul and attention quality.

## Accepted Q6_K Decode Rows2 WG32 Route

Q6_K decode was a smaller bucket than Q4_K, but HRX1 had a concrete winning
prior-art schedule: two output rows per workgroup, 32 workitems, dot16-style
packed Q6 unpacking, and vector-width RHS loads. Unlike the rejected Q4_K
rows2 probe, this route changes the lane-level schedule instead of merely
duplicating two row reductions inside a WG256 kernel.

Accepted route:

- `mul_mat_q6_k_f32_rows2_k256_32768_r1_262144_c1_wg32`
- Loom export: `hrx2_mul_mat_q6_k_f32_rows2_wg32_static`
- Domain: Q6_K x F32 `MUL_MAT`, `k_multiple_of=256`, `cols=1`,
  `rows=1..262144`, `wg32`, `rows_per_workgroup=2`.
- Prompt shapes remain on the existing direct WG256 route.

Focused backend-op gate:

- Build:
  `cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench`
- Gate:
  `test-backend-ops test -b HRX20 -o MUL_MAT --output csv`
- Artifact:
  `cache/hrx2/phase2a/q6-rows2-wg32-20260615/final-op-gate/`
- Result: 49 supported MUL_MAT rows passed, 1277 unsupported rows rejected.
  The new Q6 route compiled successfully and selected for the `cols=1` Q6_K
  rows in the generic op suite.

Three-model HRX2/Vulkan comparison:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag q6-rows2-wg32-full-20260615 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64,prefill-p64n0,prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 1200
```

| Model | Case | HRX2 before | HRX2 after | Speedup | Vulkan after | HRX2/Vulkan | CPU | Dispatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 36.116 | 40.784 | 1.129x | 125.481 | 0.3250 | 0 | 41990 |
| `llama32-3b-q4` | `decode-p1n64` | 41.370 | 46.006 | 1.112x | 140.812 | 0.3267 | 0 | 36790 |
| `llama31-8b-q4` | `decode-p1n64` | 18.722 | 20.476 | 1.094x | 90.748 | 0.2256 | 0 | 41990 |
| `phi4-mini-q4` | `prefill-p64n0` | 64.678 | 64.723 | 1.001x | 1488.602 | 0.0435 | 0 | 646 |
| `phi4-mini-q4` | `prefill-p512n0` | 67.113 | 66.929 | 0.997x | 4378.741 | 0.0153 | 0 | 646 |
| `llama32-3b-q4` | `prefill-p64n0` | 76.521 | 76.310 | 0.997x | 1644.011 | 0.0464 | 0 | 566 |
| `llama32-3b-q4` | `prefill-p512n0` | 81.116 | 80.923 | 0.998x | 4868.353 | 0.0166 | 0 | 566 |
| `llama31-8b-q4` | `prefill-p64n0` | 31.909 | 31.868 | 0.999x | 1180.909 | 0.0270 | 0 | 646 |
| `llama31-8b-q4` | `prefill-p512n0` | 32.181 | 32.127 | 0.998x | 2224.838 | 0.0144 | 0 | 646 |

Decode route selection:

| Model | Q6 rows2 dispatches | Q6 direct dispatches |
| --- | ---: | ---: |
| `phi4-mini-q4` | 1105 | 0 |
| `llama32-3b-q4` | 1885 | 0 |
| `llama31-8b-q4` | 2145 | 0 |

Interpretation: accept and default-enable this route. It is a real decode lift
with neutral prefill behavior because the route only admits `cols=1`. It does
not change the top blockers: Q4_K direct decode remains the largest quantized
matmul bucket, and p64/p512 prefill still need packed/MMQ Q4_K and attention
work.

## Accepted Decode-Only ADD -> RMS_NORM -> MUL Fusion

Runtime backpressure remains a first-class decode blocker, especially on small
models where thousands of tiny dispatches compete with relatively little device
work. A submit-policy sweep on Llama 3.2 3B Q4_K_M decode showed that changing
`GGML_HRX2_DISPATCHES_PER_SUBMIT` from the default to `0`, `64`, or `256` did
not materially move throughput:

| Submit policy | tok/s | Dispatches | Batch flushes |
| --- | ---: | ---: | ---: |
| default | 46.864 | 36790 | 3055 |
| `GGML_HRX2_DISPATCHES_PER_SUBMIT=0` | 46.489 | 36790 | 378 |
| `GGML_HRX2_DISPATCHES_PER_SUBMIT=64` | 46.713 | 36790 | 646 |
| `GGML_HRX2_DISPATCHES_PER_SUBMIT=256` | 46.476 | 36790 | 382 |

Interpretation: the current HRX1-style submit batching is not the main decode
boulder. The useful runtime work is dispatch elimination and graph fusion, not
threshold tuning.

The next high-frequency HRX1-backed candidate was `ADD -> RMS_NORM -> MUL`.
HRX2 now has a Loom provider and graph-walker fusion for the common contiguous
same-shape residual add path:

- Loom source: `ggml/src/ggml-hrx2/kernels/add_rms_norm_mul_f32.loom`
- Routes:
  - `add_rms_norm_mul_f32_n3072_r1_vector_vw4_wg512`
  - `add_rms_norm_mul_f32_n4096_r1_vector_vw4_wg512`
- The fusion preserves both the `ADD` output and final `MUL` output and uses
  `ggml_can_fuse_subgraph(..., { ADD, RMS_NORM, MUL }, { ADD, MUL })`.
- It is enabled by default for decode-shaped `nrows=1` routes and can be
  disabled with `GGML_HRX2_DISABLE_ADD_RMS_NORM_MUL_FUSION=1`.

Focused validation:

- Build:
  `cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench`
- Backend op gate:
  `test-backend-ops test -b HRX20 -o ADD/RMS_NORM/MUL --output csv`
- Artifact:
  `cache/hrx2/phase2a/add-rms-norm-mul-decode-only-20260615/`
- Result: focused op commands exited successfully; unsupported shape rows
  remained unsupported as expected.

Same-binary A/B on Llama 3.2 3B Q4_K_M decode, `p1 n64`, before narrowing to
decode-only:

| Variant | Decode token row | Dispatches | `ADD_RMS_NORM_MUL` dispatches |
| --- | ---: | ---: | ---: |
| Fusion enabled | 47.435 tok/s | 29510 | 3640 |
| Fusion disabled | 46.536 tok/s | 36790 | 0 |

Full three-model comparison after narrowing routes to decode-shaped `nrows=1`:

| Model | Case | HRX2 before | HRX2 after | Speedup | Dispatch before | Dispatch after | Fusion dispatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 40.784 | 41.097 | 1.008x | 41990 | 33670 | 4160 |
| `llama32-3b-q4` | `decode-p1n64` | 46.006 | 46.038 | 1.001x | 36790 | 29510 | 3640 |
| `llama31-8b-q4` | `decode-p1n64` | 20.476 | 20.504 | 1.001x | 41990 | 33670 | 4160 |
| `phi4-mini-q4` | `prefill-p64n0` | 64.723 | 64.177 | 0.992x | 646 | 642 | 2 |
| `phi4-mini-q4` | `prefill-p512n0` | 66.929 | 66.248 | 0.990x | 646 | 642 | 2 |
| `llama32-3b-q4` | `prefill-p64n0` | 76.310 | 75.633 | 0.991x | 566 | 562 | 2 |
| `llama32-3b-q4` | `prefill-p512n0` | 80.923 | 80.079 | 0.990x | 566 | 562 | 2 |
| `llama31-8b-q4` | `prefill-p64n0` | 31.868 | 31.591 | 0.991x | 646 | 642 | 2 |
| `llama31-8b-q4` | `prefill-p512n0` | 32.127 | 32.002 | 0.996x | 646 | 642 | 2 |

The intermediate prefill-enabled route set was rejected. It selected the fusion
for all `nrows=64/512` RMS blocks, reduced prefill dispatches from 646 to 518
or 566 to 454, but regressed p64/p512 throughput by about 0.7-1.0%. That is the
expected failure mode for a memory/kernel-quality-bound regime: fewer dispatches
do not compensate for a less optimal fused memory schedule. Keep this fusion
decode-only unless a tuned prefill variant beats the separate chain.

Interpretation: accept the decode-only fusion as useful runtime plumbing and
evidence. Do not mistake it for a bulk throughput lift. The remaining two
orders-of-magnitude prefill gap and the three-to-four-times decode gap still
need the boulders: Q4_K/Q5_K packed matvec/MMQ quality, attention/cache-write
fusions such as `ROPE -> SET_ROWS`, and eventually true prompt-side MMQ using
Q8_1/x4 or equivalent packed RHS formats.

## Remaining Blockers

- p512 attention no longer falls back to CPU, but the accepted standalone
  F16/F32 attention and softmax kernels are still slow enough that p512 prefill
  remains around 1.4% to 1.7% of Vulkan after the Q4_K cols4 lift. The next
  bulk-lift work should target packed quantized prompt matmul throughput and
  attention/fusion, not tiny route holes.

## Accepted Runtime Transfer Backplane

HRX2 now has the first non-kernel piece of the HRX1 runtime interop flow:
per-stream host-visible staging arenas, stream-ordered tensor upload/readback,
backend `cpy_tensor`, graph-entry stream synchronization, active-stream
tracking, and a transfer-stream fallback.

Focused validation:

- `ADD`, `MUL`, `MUL_MAT`, `RMS_NORM`, and `SET_ROWS` backend op tests all
  passed for supported cases.
- Artifact:
  `cache/hrx2/phase2a/transfer-stream-trimmed-gates-20260615-1355/`.

Llama 3.2 3B Q4_K_M smoke:

| Case | Previous HRX2 tok/s | New HRX2 tok/s | New Vulkan tok/s | HRX2/Vulkan |
| --- | ---: | ---: | ---: | ---: |
| `decode-p1n64` | 39.504 | 41.323 | 141.484 | 0.2921 |
| `prefill-p64n0` | 75.264 | 78.143 | 1711.681 | 0.0457 |

Artifact: `cache/hrx2/phase2a/transfer-stream-trimmed-smoke-20260615-1357/`.

Interpretation: this is a small runtime lift and a necessary correctness
backplane, but it is not the Phase 2a bulk lift. Small-model decode remains
runtime-backpressure sensitive because it still executes about 36.7k dispatches
for a 64-token generation. Larger models can tolerate more runtime slop, but
the same issues remain in scope. The next boulders are dispatch elimination via
hero fusions and true high-throughput quantized prompt/decode kernels.

Runtime parity follow-up:

- HRX1's timeline-mediated `hrx_queue_copy` / `hrx_queue_fill` helper shape is
  now present in HRX2. Earlier measurements showed that this surface is not a
  decode throughput boulder by itself, but it is still part of the HRX1 runtime
  contract and should remain available for synchronous buffer operations,
  graph-level CPY, staging, and future interop hazards.

## Accepted ROPE Factor Placement Fix

Decode traces after full-offload input embedding placement still showed
`rope_freqs.weight` copied into every ROPE split. This was not a model-layer
placement policy problem. The llama.cpp weight-buffer support probe for ROPE
created a synthetic `ggml_rope_ext` op with `n_dims=0`, so HRX2 correctly
rejected the representative op and the shared ROPE factor tensor fell back to
CPU even though the real decode ROPE routes were supported.

Fix:

- Derive probe `n_dims` from the factor tensor length (`w->ne[0] * 2`).
- Use a one-token representative ROPE op with `n_embd_head >= n_dims`.
- Do not broaden HRX2 `supports_op`; real graph scheduling still requires an
  actual matching route.

Focused validation:

- Build:
  `cmake --build build/llama-hrx2 --target llama-bench test-backend-ops`.
- Backend op gate:
  `build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o ROPE --output csv`.
- Manual override A/B proved the placement hypothesis first:
  `-ot rope_freqs.weight=HRX20` removed 672 split-input copies in the 3B decode
  trace and moved decode from about 41.36 to 42.13 tok/s.

Accepted no-override comparison:

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | CPU compute | Remaining split inputs |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `llama32-3b-q4` | `decode-p1n64` | 41.939 | 140.084 | 0.2994 | 0 | `leaf_5`, `leaf_9`, `leaf_11`, `attn_inp_kq_mask`, `inp_tokens` |
| `llama32-3b-q4` | `prefill-p64n0` | 77.804 | 1701.151 | 0.0457 | 0 | prompt path still Q4_K/attention bound |

Artifact:
`cache/hrx2/phase2a/rope-probe-fix-compare-20260615/`.

Interpretation: this is accepted because it removes avoidable recurrent CPU
split-input traffic with no route broadening. It is still a pebble relative to
the remaining boulders. Small-model decode is dominated by true dynamic input
backpressure plus about 36.8k dispatches for 64 generated tokens. Prefill is
still dominated by quantized matmul and attention quality.

## Rejected Runtime Variant: Scheduler Event Copy Rotation

HRX2 backend events were prototyped against the existing HRX event API and then
tested as a decode backpressure fix. Two scheduler experiments matter:

- Advertising backend events alone is a no-op for the normal single-GPU
  llama.cpp path, because `ggml_backend_sched_new(..., parallel=false, ...)`
  creates only one scheduler copy and no scheduler events.
- Reusing `cparams.pipeline_parallel` to force events is also wrong for
  single-GPU HRX2: llama.cpp synchronizes before setting graph inputs whenever
  `pipeline_parallel` is true, so it cancels the intended overlap.

A more precise local experiment decoupled scheduler copy rotation from
`pipeline_parallel`. It created events and rotated scheduler input-copy buffers
without triggering llama.cpp's graph-reuse pre-sync. The trace changed as
expected on Llama 3.2 3B Q4_K, `decode-p1n64`:

| Variant | tok/s | Full stream syncs | Event synchronizes | Stream sync elapsed |
| --- | ---: | ---: | ---: | ---: |
| Baseline after ROPE fix | 41.939 | 595 | 0 | 1.008 s |
| Event copy rotation | 41.947 | 205 | 390 | 1.013 s |

Artifacts:

- Baseline: `cache/hrx2/phase2a/rope-probe-fix-compare-20260615/`.
- Event-only no-op:
  `cache/hrx2/phase2a/hrx2-events-compare-20260615-rerun/`.
- Forced pipeline-parallel event path:
  `cache/hrx2/phase2a/hrx2-sched-events-compare-20260615/`.
- Decoupled scheduler copy rotation:
  `cache/hrx2/phase2a/hrx2-sched-copy-rotation-compare-20260615/`.

Conclusion: do not spend more Phase 2a time on scheduler-copy rotation as a
standalone fix. The remaining dynamic split inputs are only about 9 KiB per
decode graph. Event copy rotation changes wait accounting but does not move
throughput because the normal decode loop still must observe graph completion
for logits/sampling, and HRX2 still issues about 36.8k dispatches for the
64-token smoke. For small models, the runtime boulders are dispatch count and
hero fusion structure; for larger models, kernel quality hides more runtime
slop but the same dispatch/fusion work remains in scope.

## Accepted Q5_K Decode Dot16 Route

The next decode kernel-quality probe targeted Q5_K matvec. HRX1's Q5_K decode
path uses a 16-lane block schedule with strided packed loads:

- `itid = tid & 15`
- `block_slot = tid >> 4`
- `block_stride = WG_SIZE >> 4`
- each active lane computes one `hrx_q5_k_dot16` over packed `qs`, `qh`, scale,
  and min fields.

HRX2's previous Q5_K Loom route was a broad direct route using a scalar-ish
four-values-per-lane spelling and `wg256`. The accepted route adds a separate
decode-only export in the same Loom source:

- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q5_k_f32.loom`
- Route:
  `mul_mat_q5_k_f32_dot16_k256_32768_r1_262144_c1_wg32`
- Export:
  `hrx2_mul_mat_q5_k_f32_dot16_static`
- Shape:
  `cols == 1`, `k % 256 == 0`, rows `1..262144`
- Fallback:
  existing `mul_mat_q5_k_f32_direct_k256_32768_r1_262144_c1_512_wg256`
  remains active for prompt and other multi-column cases.

The Loom source deliberately spells the algorithm rather than depending on
compiler recovery:

- explicit 16-lane decomposition from the workitem id;
- explicit strided byte loads for `qs0`, `qs4`, and `qh`;
- explicit `vector<2xf32>` RHS loads matching the HIP `float2` offsets
  `0/16/32/48/128/144/160/176`;
- explicit packed scale/min reconstruction;
- explicit vector product/reduce of the four partial dot groups.

Focused validation:

```bash
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench -j"$(nproc)"

GGML_HRX2_TRACE_JSONL="$out/hrx2.jsonl" \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o MUL_MAT --output csv \
  > "$out/mul_mat.csv" 2> "$out/stderr.txt"
```

Artifact:
`cache/hrx2/phase2a/q5-dot16-20260615-154659/op-gate/`.

Result:

- Supported `MUL_MAT` cases passed.
- The new Q5_K dot16 route selected on 5 decode-shaped synthetic cases.
- No HRX2 trace failure events.

Basket decode comparison:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag q5-dot16-decode-20260615-154740 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64 \
  --backends hrx2,vulkan \
  --repetitions 1 \
  --timeout 1200
```

Artifact:
`cache/hrx2/phase2a/q5-dot16-decode-20260615-154740/`.

| Model | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | Q5 dot16 dispatches | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `phi4-mini-q4` | 45.292 | 125.210 | 0.3617 | 2080 | Real win; Phi has Q5_K decode matmuls. |
| `llama32-3b-q4` | 47.354 | 140.376 | 0.3373 | 0 | No Q5_K MUL_MAT dispatches in this slice. |
| `llama31-8b-q4` | 21.092 | 91.282 | 0.2311 | 0 | No Q5_K MUL_MAT dispatches in this slice. |

Compared to the prior decode slice after `ADD -> RMS_NORM -> MUL`, Phi moved
from about 41.1 tok/s to 45.3 tok/s. The other models did not exercise the new
route, so their movement should be treated as run-to-run noise unless repeated.

This is accepted as a kernel-quality improvement and a useful packed-K-quant
authoring example, but it is still not the Phase 2a bulk lift. Remaining
decode boulders in the same slice:

- Llama 3.x: Q4_K direct matvec and attention/SET_ROWS/ROPE dispatch volume.
- Phi: SET_ROWS and attention remain top dispatch families after Q5_K improves.
- Small-model decode remains runtime-backpressure sensitive; at about 30B more
  runtime slop can be hidden by heavier kernels, but dispatch/fusion cleanup is
  still in scope for all sizes.

## Rejected Probe: F16 Attention Cols4 Prompt Route

Hypothesis: the batched F16 attention matvec used in prompt/prefill could
reuse each F16 source row load across four RHS columns. A temporary
`hrx2_mul_mat_f16_f32_batched_cols4` route was added with `cols_per_workgroup=4`
and then guarded to `cols >= 128` after the first p64 slice showed that narrow
prompt shapes were not a good fit.

Focused validation passed:

- Build target: `ggml-hrx2 test-backend-ops llama-bench`.
- Backend op gate:
  `cache/hrx2/phase2a/f16-attn-cols4-wide-20260615-161853/op-gate/`.
- `MUL_MAT` suite: 1326 rows, 0 fail-like rows.

Model-shaped A/B after narrowing the guard:

- Full p64/p512 slice:
  `cache/hrx2/phase2a/f16-attn-cols4-wide-full-20260615-161923/`.
- HRX-only repeat:
  `cache/hrx2/phase2a/f16-attn-cols4-wide-hrx-repeat-20260615-162104/`.

The route split behaved correctly:

- p64 stayed on `mul_mat_f16_f32_batched_attention_wg256`.
- p512 selected `mul_mat_f16_f32_batched_attention_cols4_wg256`.

The performance signal was not durable. The first full slice showed small p512
wins of about `1.7%` to `2.6%`, but the repeat moved Phi to neutral, Llama 3.2
3B to negative, and Llama 3.1 8B to only `+0.7%`. p64 also moved by a few
percent even though it did not select the new route, which indicates benchmark
noise or JIT/cache effects at this scale.

Decision: reject and remove the route. This is a useful negative result for
future agents: do not spend Phase 2a effort on standalone F16 attention
multi-column scalar matvec tweaks unless there is device-time evidence or a
larger fusion such as ROPE/SET_ROWS/attention-cache update changes the problem.
The boulders remain quantized prompt MMQ quality, KV/cache-update fusion, and
decode runtime backpressure.

## Runtime Parity: HRX1 Copy/Fill and CPY Surfaces

HRX1's strongest runtime pieces were semantic, not just individual kernels:
timeline-mediated queue copy/fill for synchronous buffer operations, device
staging/copy behavior that does not fight the graph stream, and graph-level
`GGML_OP_CPY` support including `F32 -> F16` conversion. HRX2 now carries those
surfaces forward.

Landed in llama.cpp HRX2:

- Synchronous buffer `memset_tensor`, `cpy_tensor`, and `clear` now use
  timeline-mediated `hrx_queue_fill` / `hrx_queue_copy` helpers tied to the
  active HRX2 stream, mirroring the HRX1 runtime contract.
- Graph `GGML_OP_CPY` supports contiguous same-type stream copies.
- Strided same-type CPY supports row-contiguous sources; F32 strided copies
  reuse the existing `cont_f32` Loom route where possible and otherwise fall
  back to row-by-row stream copies.
- Contiguous `F32 -> F16` graph CPY now has a Loom provider,
  `copy_f32_f16_generic_wg256`, matching the HRX1 HIP provider role.

Focused validation:

```bash
GGML_HRX2_TRACE_PROVIDERS=1 GGML_HRX2_EVIDENCE_DIR="$OUT/evidence" \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o CPY,CONT --output csv
```

Artifact:
`cache/hrx2/phase2a/runtime-parity-copy-f32-f16-20260615-164621/`.

Result:

- `CPY`/`CONT` rows: 462.
- Fail-like rows: 0.
- Supported rows: 119.
- The new F32->F16 provider compiled and was exercised:
  `copy_f32_f16_generic_wg256|target=gfx1100|n=16384`.
- Compile report for `hrx2_copy_f32_f16`: 19 instructions, 68 code bytes, no
  spills, no private/local memory, one conversion, and two global-memory ops.

Basket decode measurements before this final F32->F16 patch already showed that
the queue helper and same-type CPY parity changes were neutral-to-small-positive
but not structural:

- Phi decode: ~1.00x versus the previous HRX2 slice.
- Llama 3.2 3B decode: ~1.01x.
- Llama 3.1 8B decode: ~1.02x.
- Dispatch counts and scheduler split-input counts were unchanged.
- Current basket decode traces had zero graph `CPY` nodes, so the model-level
  result is expected to be neutral even though the runtime surface is now
  available.

Final HRX2-only decode smoke after adding the F32->F16 provider:
`cache/hrx2/phase2a/runtime-parity-copy-final-20260615-164740/`.

| Model | HRX2 tok/s | Dispatches | CPU compute |
| --- | ---: | ---: | ---: |
| `phi4-mini-q4` | 45.183 | 33670 | 0 |
| `llama32-3b-q4` | 47.362 | 29510 | 0 |
| `llama31-8b-q4` | 21.080 | 33670 | 0 |

Decision: accept as HRX1 runtime parity, not as a Phase 2a performance boulder.
The next boulders are still split-input/runtime backpressure, direct
ROPE/VIEW/SET_ROWS KV-cache writes, attention-cache fusion, and quantized prompt
MMQ quality.

## Runtime Parity Audit: Scratch Lifetime and Remaining HRX1 Gaps

After the HRX1 runtime comparison, HRX2 now carries the core HRX1 runtime
backplane:

- registered graph and transfer streams with an active-stream handoff,
- persistent mapped staging arenas,
- timeline-mediated `hrx_queue_fill` / `hrx_queue_copy` for synchronous buffer
  operations,
- graph-entry stream synchronization,
- submit batching by dispatch count and matmul-byte progress,
- conservative scheduler contract with no `offload_op`,
- graph `CPY` coverage, including same-type copies and contiguous `F32 -> F16`.

One remaining parity detail was fixed in HRX2 after the audit: Q8_1 prompt
scratch growth no longer synchronizes the compute stream and immediately
releases the old buffer. It now retires the old scratch buffer and releases
retired scratch after `ggml_backend_hrx2_synchronize`, matching HRX1's
persistent scratch lifetime model.

Validation:

- Build target: `test-backend-ops`, `llama-bench`, `llama-cli`.
- Focused op gate:
  `cache/hrx2/runtime-parity-audit-20260615-165658/`.
- Corrected reduction: 1788 `MUL_MAT`/`CPY`/`CONT` rows, 168 supported rows,
  and no backend error rows.
- Normal Phi decode smoke:
  `cache/hrx2/phase2a/runtime-parity-scratch-smoke-20260615-165731/`;
  45.106 tok/s, 33670 dispatches, zero CPU compute nodes.
- Scratch-exercising prompt smoke with
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`:
  `cache/hrx2/phase2a/runtime-parity-q8scratch-smoke-20260615-165752/`;
  79.143 tok/s, 674 dispatches, zero CPU compute nodes. The trace selected
  `quantize_q8_1_f32_generic_wg32` and
  `mul_mat_q4_k_q8_1_f32_cols4_k256_32768_c4_512_wg256` 112 times each.

Remaining HRX1-derived gaps are not basic copy/scratch/runtime backplane gaps,
but they matter for Phase 2a throughput:

- `ROPE -> VIEW -> SET_ROWS` KV-cache write fusion exists in HRX1 and is still
  missing in HRX2. HRX2 currently emits separate ROPE and SET_ROWS dispatches.
- Dynamic split inputs remain for token ids, positions, row indices, and masks.
  These are still a decode backpressure source.
- HRX1 recurrent/state scheduling paths for SSM/GDN-style models are not in
  HRX2. They are basket-dependent, but should be ported before claiming broad
  HRX1 runtime feature parity.
- HRX1 has quantized SET_ROWS providers (`Q8_0`, `Q4_0`); HRX2 SET_ROWS
  currently covers F32/F16 only.

Decision: accept the HRX2 runtime backplane as complete enough to continue
Phase 2a. Treat the remaining HRX1 items as explicit Phase 2a/Phase 2b work
items, with `ROPE/VIEW/SET_ROWS` and split-input pressure as the first decode
boulders.

## Final HRX1 Runtime Parity Pass

The follow-up audit found three remaining HRX1 runtime behavior gaps in HRX2
and fixed them in llama.cpp:

- HRX2 now has HRX1-style transient scratch buffers, recycled on backend
  synchronize and released on backend free. This is separate from the persistent
  Q8_1 scratch buffer and is needed before porting HRX1 routes that use
  short-lived work buffers.
- HRX2 now has a persistent route scratch slot matching HRX1's route scratch
  lifetime model. Current Phase 2a routes may not consume it yet, but the
  runtime substrate is available before MoE/attention-style route compaction is
  ported.
- Stream registration no longer marks a new stream active immediately. This
  restores HRX1's initial behavior where pre-graph tensor uploads fall back to
  the transfer stream until graph compute explicitly hands active execution to
  the graph stream.
- HRX2 graph compute now synchronizes before returning by default, matching
  HRX1's scheduler contract. The previous flush-only behavior is still
  available for experiments with `GGML_HRX2_ASYNC_GRAPH_COMPUTE=1`.
- HRX2 gained `GGML_HRX2_DISABLE_FUSION=1` and `GGML_HRX2_TRACE_GRAPH=1`,
  matching HRX1's broad diagnostic controls.

Validation:

- Build: `cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench`.
- Full HRX2 backend op gate:
  `cache/hrx2/phase2a/runtime-parity-opgate-20260615-171403/`.
  It produced 11534 rows, 320 supported rows, and 0 supported failures.
- Integration smoke:
  `cache/hrx2/phase2a/runtime-parity-smoke-20260615-171512/`.
  Phi-4-mini Q4_K_M decode p1/n64 completed at 44.953 tok/s, with 33670 HRX2
  dispatches and zero CPU compute nodes.

Decision: HRX2 now has the HRX1 runtime backplane needed for Phase 2a. The
remaining throughput work is not missing copy/scratch plumbing; it is the
larger boulder work already identified: KV-cache write fusion, split-input
pressure, attention/cache fusion, and quantized hero-kernel quality.

## ROPE/VIEW/SET_ROWS KV-Cache Write Fusion

HRX2 now has a target-generic Loom fusion for the K-cache write path:
`ROPE -> VIEW -> SET_ROWS`, dispatched as `ROPE_SET_ROWS`. This ports the
important HRX1/Vulkan/CUDA runtime feature where K ROPE output is written
directly into the F16 KV cache instead of materializing ROPE and then launching
SET_ROWS.

Implementation details:

- Loom source: `ggml/src/ggml-hrx2/kernels/rope_set_rows_f32.loom`.
- Catalog routes: `ggml/src/ggml-hrx2/catalog/routes/rope_set_rows_f32.json`.
- Runtime route family: `rope_set_rows_f32`, loaded as a `SET_ROWS` family.
- Disable knob: `GGML_HRX2_DISABLE_ROPE_SET_ROWS_FUSION=1`.
- The source is target-generic. Route `target_key` is empty; target-specific
  variants should be separate source files only when they use target-specific
  layouts or instructions.

Validation:

- Focused backend op gate:
  `cache/hrx2/phase2a/rope-set-rows-fusion-opgate-20260615-173503/`.
  ROPE produced 288 rows, 16 supported rows, 0 supported failures. SET_ROWS
  produced 315 rows, 33 supported rows, 0 supported failures.
- Direct Loom compile repro for the failing fused h8 config:
  `cache/hrx2/phase2a/rope-set-rows-fusion-compile-repro-20260615-173439/`;
  `loom-compile --backend=amdgpu-hal --target=gfx1100` completed successfully.
- Phi-4-mini Q4 decode p1/n64 with fusion enabled:
  `cache/hrx2/phase2a/rope-set-rows-fusion-on4-20260615-173516/`.
  It completed at 45.242 tok/s, 31590 HRX2 dispatches, zero CPU compute nodes.
  Route counts showed `rope_neox_f32_freq_set_rows_f16_n128_d96_h8_t1_64_wg256`
  selected 2080 times, separate `set_rows_f32_f16_generic` reduced from 4160
  to 2080, and separate h8 ROPE disappeared.
- Phi-4-mini Q4 decode p1/n64 with fusion disabled:
  `cache/hrx2/phase2a/rope-set-rows-fusion-off-20260615-173539/`.
  It completed at 44.927 tok/s, 33670 HRX2 dispatches, zero CPU compute nodes.
- Llama 3.2 3B Q4 decode p1/n64:
  `cache/hrx2/phase2a/rope-set-rows-fusion-llama32-20260615-173608/`.
  It completed at 47.369 tok/s, 27690 HRX2 dispatches, zero CPU compute nodes.
  Route counts showed
  `rope_normal_f32_freq_set_rows_f16_n128_d128_h8_24_t1_64_wg256` selected
  1820 times with no provider failures.

One implementation lesson matters for future Loom kernels: this fusion initially
matched graph shapes but failed JIT compilation because loaded row indices were
not bounded with `index.assume` before destination address arithmetic. Loom is
WYSIWYG here. If a kernel computes an address from dynamic data, spell the
unsigned range facts on the loaded/index-cast value and on intermediate address
sums, following the standalone SET_ROWS pattern.

Impact: this is a correct dispatch-count boulder, not the full Phase 2a lift by
itself. Phi decode dispatches dropped by 2080 and single-run throughput moved
from 44.927 to 45.242 tok/s. The next dominant blockers are attention/cache
dispatches, split-input backpressure, and quantized hero-kernel quality.

## HRX1 Runtime Parity Audit

After the ROPE/SET_ROWS work, we rechecked HRX2 against HRX1 for runtime
features before attributing the remaining decode CPU/GPU behavior to missing
interop plumbing.

Present in HRX2:

- Dedicated transfer stream registered with the device context.
- Active graph stream handoff at graph entry, with transfer/active stream
  synchronization before graph compute.
- Active stream unregister semantics match HRX1: unregistering the active
  stream clears `active_stream` instead of silently falling through to another
  registered stream.
- Per-stream staging arenas using persistent mapped host-local/device-visible
  buffers.
- Timeline semaphore based queue fill/copy helpers for host-visible operations.
- Backend buffer `memset_tensor`, `set_tensor`, `get_tensor`, `cpy_tensor`, and
  `clear` use the staging/queue-copy path rather than raw synchronous transfer
  calls on the normal buffer API path.
- Graph-level `CPY` covers same-type contiguous copies, row-strided copies, the
  `cont_f32` route path, and contiguous `F32 -> F16` conversion.
- Device-local transient scratch pool with retire/recycle on synchronize.
- Persistent q8_1 and route scratch buffers with retired-buffer release after
  stream synchronization.
- HRX1-style submit batching using dispatch count and matmul-byte thresholds.
- HRX1-compatible defaults for staging arena size, alignment, dispatches per
  submit, and max matmul bytes per submit.
- Graph trace and broad fusion-disable diagnostics.
- Synchronous graph-compute contract by default, with HRX2-only
  `GGML_HRX2_ASYNC_GRAPH_COMPUTE=1` retained as an experiment knob.

Validation evidence:

- Current three-model baseline:
  `cache/hrx2/phase2a/baseline-after-rope-set-rows-20260615-174031/`.
  All HRX2 cases reported zero scheduler CPU compute nodes. Decode was still
  behind Vulkan by 2.78x to 4.40x; prefill remained much farther behind, which
  points at hero kernels/fusions rather than CPU fallback.
- Async graph-compute A/B:
  `cache/hrx2/phase2a/async-graph-ab-20260615-174334/` and
  `cache/hrx2/phase2a/async-graph-on-20260615-174342/`. Phi and Llama 3.2 3B
  decode throughput was effectively unchanged. Sync counts shifted into flush
  counts, so graph-end synchronization is not the current decode boulder.
- Submit-batching A/B:
  `cache/hrx2/phase2a/submit-batching-disabled-20260615-174422/` and
  `cache/hrx2/phase2a/submit-batching-256-20260615-174431/`. Disabling submit
  batching regressed Phi decode from 45.186 to 34.640 tok/s and Llama 3.2 3B
  decode from 46.726 to 36.672 tok/s. Relaxing to 256 dispatches per submit
  also regressed both cases. Keep the HRX1-style batching policy for now.

Conclusion: the HRX1 runtime backplane is represented in HRX2 for the features
that matter to llama.cpp integration. The remaining Phase 2a gap should be
treated as missing/weak hero fusions and kernel quality unless a new trace shows
actual CPU fallback, provider failures, or transfer synchronization on the hot
path.

Follow-up parity fix:

- HRX1 always clears `active_stream` when unregistering that stream. HRX2 had
  been reassigning it to another live stream. That looked benign in the current
  traces, but it was not HRX1-compatible, so HRX2 now clears `active_stream`
  on unregister as HRX1 does.
- Build validation:
  `cmake --build build/llama-hrx2 --target test-backend-ops llama-bench llama-cli`.
- Focused backend op validation:
  `GGML_HRX2_TRACE_JSONL=cache/hrx2/phase2a/optest-runtime-parity-*.jsonl build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o RMS_NORM,MUL_MAT,SET_ROWS,CPY,CONT`.
  The runner reported 211/211 supported HRX20 cases passing.
- Integration smoke:
  `cache/hrx2/phase2a/runtime-parity-smoke-20260615-175417/`.
  Phi-4-mini and Llama 3.2 3B Q4_K_M decode p1/n64 plus prefill p64/n0 all
  completed with zero scheduler CPU compute nodes.
- Refresh after the final source audit:
  `cache/hrx2/phase2a/runtime-parity-refresh-20260615-180355/`.
  The focused `RMS_NORM,MUL_MAT,SET_ROWS,CPY,CONT` backend-op gate produced
  2124 rows, 211 supported rows, and 0 supported failures. This revalidated the
  HRX1-derived stream/staging/copy/scratch surfaces after correcting the stale
  queue-copy/fill note above.
- Recheck after the runtime parity reminder:
  `cache/hrx2/phase2a/runtime-parity-recheck-20260615-181232/`.
  Build target `test-backend-ops llama-bench` was current. The focused
  `RMS_NORM,MUL_MAT,SET_ROWS,CPY,CONT` gate again produced 2124 rows, 211
  supported rows, and 0 supported errors. Route trace coverage included
  `cpy_contiguous_stream`, `cpy_strided_rows_stream`,
  `cpy_strided_f32_cont_route`, `copy_f32_f16_generic_wg256`,
  `set_rows_f32_f32_generic`, `set_rows_f32_f16_generic`, scratch-using
  quantized matmul routes, and the current RMS_NORM routes.

Operational rule: Phase 2 work must preserve this HRX1 runtime parity checklist.
If a future patch touches buffer transfer, graph entry/exit, stream ownership,
submit flushing, scratch growth, or graph `CPY`/`CONT` behavior, rerun the
focused backend-op gate before any model-level benchmark. Performance work
should move on to boulders only when this runtime backplane remains clean.

## FLASH_ATTN_EXT Gap

A separate `-fa 1` diagnostic showed a real HRX1/Vulkan op-coverage gap:
`GGML_OP_FLASH_ATTN_EXT` is not implemented in HRX2. This is not missing stream,
staging, scratch, or submit-batching plumbing; it is missing fused-attention
provider coverage on top of that runtime backplane.

Evidence:

- Diagnostic run:
  `cache/hrx2/phase2a/flash-attn-on-20260615-175102/`.
- Vulkan used `FLASH_ATTN_EXT` directly. Its stderr includes the fused attention
  dispatches for both decode and prompt cases, for example Phi-4-mini p512
  `FLASH_ATTN_EXT dst(128,24,512,1) ...`.
- HRX2 ran the same model shapes but scheduled every `FLASH_ATTN_EXT` node to
  CPU:
  - Phi-4-mini decode: 384 CPU compute nodes, all `FLASH_ATTN_EXT`.
  - Phi-4-mini p64/p512 prefill: 192 CPU compute nodes, all `FLASH_ATTN_EXT`.
  - Llama 3.2 3B decode: 336 CPU compute nodes, all `FLASH_ATTN_EXT`.
  - Llama 3.2 3B p64/p512 prefill: 168 CPU compute nodes, all `FLASH_ATTN_EXT`.
- HRX2 `ggml-hrx2.cpp` has no `GGML_OP_FLASH_ATTN_EXT` support entry or dispatch
  case. HRX1 has a full provider set for this op, including F16 decode split,
  decode reduce, and prefill direct/tile/WMMA variants.

Decision:

- Keep Phase 2a `-fa 0` comparisons as the current no-CPU-fallback production
  baseline until HRX2 has a real Loom `FLASH_ATTN_EXT` implementation.
- Treat `FLASH_ATTN_EXT` as a Phase 2a boulder, not a measurement detail. Any
  future `-fa 1` run that shows CPU fallback is expected until this provider
  family is implemented.
- Do not port HRX1 HIP flash-attention providers into HRX2 as a casual fix.
  That would undermine the clean Loom-backed HRX2 architecture unless explicitly
  chosen as a temporary bridge. The preferred path is a Loom fused-attention
  family with separate decode and prompt routes, informed by HRX1, Vulkan, and
  CUDA prior art.

## Accepted Incremental Route: Q4_K F32 RHS cols8 Prompt

The next Q4_K prompt probe widened the accepted F32-RHS column-reuse route from
four prompt columns per workgroup to eight. This keeps the same direct
Q4_K-dequant/F32-RHS algorithm and only changes the wide prompt tile; decode
continues to use the existing `cols=1` direct route.

Landed in llama.cpp HRX2:

- `hrx2_mul_mat_q4_k_f32_cols8_static` in
  `ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`.
- Catalog route
  `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256` with
  `cols_per_workgroup=8`, `cols_multiple_of=8`, and priority above the direct
  route.

Validation:

- HRX and HRX2 rebuild:
  `cmake --build build/hrx-system --target install` and
  `cmake --build build/llama-hrx2 --target test-backend-ops llama-bench llama-cli`.
- Catalog assembly/validation:
  `assemble_hrx2_catalog.py --catalog-dir ggml/src/ggml-hrx2/catalog` and
  `validate_hrx2_catalog.py`.
- Focused op gate:
  `cache/hrx2/phase2a/q4k-cols8-opgate-20260615-181743/`.
  `MUL_MAT` produced 1326 rows, 49 supported rows, and 0 supported errors. The
  new route selected for 9 focused cases and JIT-compiled successfully.
- Current HRX2/Vulkan prefill comparison:
  `cache/hrx2/phase2a/q4k-cols8-prefill-hvx-20260615-181938/`.
- HRX2 repeat:
  `cache/hrx2/phase2a/q4k-cols8-repeat-20260615-182111/`.
- Decode smoke:
  `cache/hrx2/phase2a/q4k-cols8-decode-smoke-20260615-182244/`.
  Phi-4-mini and Llama 3.2 3B decode completed with zero CPU compute nodes and
  stayed on the existing decode routes.

Result versus the prior HRX2 baseline:

| Model | Case | Old HRX2 | cols8 repeat | Change |
| --- | --- | ---: | ---: | ---: |
| `llama31-8b-q4` | `prefill-p512n0` | 32.095 | 33.000 | +2.8% |
| `llama31-8b-q4` | `prefill-p64n0` | 31.818 | 32.083 | +0.8% |
| `llama32-3b-q4` | `prefill-p512n0` | 80.835 | 83.885 | +3.8% |
| `llama32-3b-q4` | `prefill-p64n0` | 76.306 | 75.144 | -1.5% |
| `phi4-mini-q4` | `prefill-p512n0` | 66.872 | 66.906 | +0.1% |
| `phi4-mini-q4` | `prefill-p64n0` | 64.784 | 63.310 | -2.3% |

Current same-machine Vulkan ratios with cols8 remain far from target:

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: |
| `llama31-8b-q4` | `prefill-p512n0` | 33.039 | 2206.600 | 0.0150 |
| `llama31-8b-q4` | `prefill-p64n0` | 32.540 | 1167.828 | 0.0279 |
| `llama32-3b-q4` | `prefill-p512n0` | 83.583 | 4753.700 | 0.0176 |
| `llama32-3b-q4` | `prefill-p64n0` | 77.372 | 1540.510 | 0.0502 |
| `phi4-mini-q4` | `prefill-p512n0` | 67.457 | 4233.126 | 0.0159 |
| `phi4-mini-q4` | `prefill-p64n0` | 62.981 | 1451.023 | 0.0434 |

Decision: accept as a small wide-prompt kernel-quality improvement and a useful
shape-specific route, but do not spend more Phase 2a time on scalar F32-RHS
column widening. The remaining prefill gap is still a boulder: Q4_K/Q5_K/Q6_K
packed/MMQ prompt schedules, fused attention coverage, or higher-level fusions
that remove memory traffic and dispatches. Direct cols8 does not change dispatch
count and is not a substitute for the packed RHS path.

## Prefill Root Cause Update: Warm Steady State Still Points At Prompt MMQ

After the cols8 route landed, the question was whether the large p512 gap was
still a measurement/JIT artifact or a true GPU workload problem. A serial warmup
check on Llama 3.2 3B Q4_K_M p512 shows it is the latter:

- Artifact: `cache/hrx2/phase2a/warmup-serial-20260615-183030/`.
- HRX2 cold/no-warmup p512: `83.430 tok/s`, 562 dispatches, 23 provider
  compiles.
- HRX2 warm/repeated p512: `82.749 tok/s`, 2248 dispatches across warmup plus
  repetitions, still only 23 provider compiles total.
- Vulkan warm/repeated p512: `5730.808 tok/s`.
- Vulkan per-op logger for the same run reports Q4_K prompt matmul in the
  tens-to-hundreds of microseconds per op:
  - `m=1024 n=512 k=3072`: about `68-78 us`.
  - `m=3072 n=512 k=3072`: about `274-291 us`.
  - `m=3072 n=512 k=8192`: about `616-668 us`.
  - `m=8192 n=512 k=3072`: about `599-647 us`.

An opt-in HRX2 Q8_1 RHS probe was also tested:

```bash
GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1 python3 tools/hrx2_phase2a_benchmark.py \
  --tag q4k-q8-prompt-ab-20260615-182802 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases prefill-p64n0,prefill-p512n0 \
  --backends hrx2 --repetitions 1 --timeout 1200 --flash-attn 0
```

That route selected `quantize_q8_1_f32_generic_wg32` plus
`mul_mat_q4_k_q8_1_f32_cols4...`, but performance moved only at noise scale.
This is expected: the route changes the RHS dot form, but it is still a direct
one-output-row-per-workgroup matmul. It does not implement the row/column-tiled
MMQ schedule that Vulkan and CUDA use for prompt. HRX1 has the same lesson in
its runtime/route table: Q8_1 prompt packing is paired with x4/tiled providers
where they exist, and route selection records prompt-token thresholds for those
variants.

Conclusion:

- JIT compile cost is not the p512 explanation; warm HRX2 is no faster than
  cold HRX2 for this case.
- Pointwise fusion and scalar column widening are not the missing boulder.
- The primary p64/p512 Q4_K boulder is a proper Loom Q8_1 x4/MMQ prompt matmul
  family with row tiles such as 32/64/128 and column tiles such as 32/64/128,
  selected by shape and target metadata.
- `FLASH_ATTN_EXT` remains the other major boulder, especially for `-fa 1`, but
  the no-FA prefill baseline is already dominated by quantized prompt matmul
  quality.

Implementation target for the next pass:

1. Add Loom x4 Q8_1 quantization output compatible with a tiled prompt matmul
   route, not just the current scalar `block_q8_1` layout.
2. Add a Q4_K x Q8_1 prompt MMQ Loom kernel that computes multiple output rows
   and multiple prompt columns per workgroup.
3. Tune row/column tiles against the basket p64/p512 shapes and preserve the
   winning schedule in catalog JSON.
4. Keep the current cols8 direct F32 route as fallback for odd/small shapes.

## Reverify: 100% GPU Utilization Is Device Work, Not CPU Fallback

After syncing `hrx-system` main and rebuilding HRX2/Vulkan, the current
no-flash-attention Phase 2a slice was rerun:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag phase2a-reverify-20260615-190855 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64,prefill-p64n0,prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 1200 --flash-attn 0
```

Artifact:

- `cache/hrx2/phase2a/phase2a-reverify-20260615-190855/`

Result:

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | HRX2 dispatches | CPU compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 45.216 | 119.177 | 0.379 | 31590 | 0 |
| `llama32-3b-q4` | `decode-p1n64` | 46.807 | 139.396 | 0.336 | 27690 | 0 |
| `llama31-8b-q4` | `decode-p1n64` | 20.639 | 88.577 | 0.233 | 31590 | 0 |
| `phi4-mini-q4` | `prefill-p64n0` | 65.730 | 1447.550 | 0.045 | 610 | 0 |
| `llama32-3b-q4` | `prefill-p64n0` | 78.031 | 1649.595 | 0.047 | 534 | 0 |
| `llama31-8b-q4` | `prefill-p64n0` | 32.543 | 1170.924 | 0.028 | 610 | 0 |
| `phi4-mini-q4` | `prefill-p512n0` | 68.301 | 4251.122 | 0.016 | 642 | 0 |
| `llama32-3b-q4` | `prefill-p512n0` | 83.992 | 4742.411 | 0.018 | 562 | 0 |
| `llama31-8b-q4` | `prefill-p512n0` | 32.980 | 2184.104 | 0.015 | 642 | 0 |

The important runtime signal is that prefill is not stuck on host fallback.
For Llama 3.2 3B p512, HRX2 records about `5.94 s` of
`hrx_stream_synchronize` wait, about `2.0 ms` of flush time, and about
`1.2 ms` of dispatch-recording time. In other words, the observed 100% GPU
utilization is queued device work, not an integration loop spinning on CPU.

Vulkan's same-machine p512 perf labels show the target class:

- Llama 3.2 3B p512 total logged matmul time: about `83.7 ms`.
- Q4_K prompt matmul share: about `62.4 ms`.
- The major Q4_K prompt shapes run in tens to hundreds of microseconds per op,
  at roughly `33-42 TFLOP/s` equivalent in Vulkan's logger.

HRX2 p512 is therefore still in the bulk-lift regime. The selected HRX2 route
families are structurally too weak:

- `mul_mat_q4_k_f32_cols8...` is only a scalar F32-RHS column-widening route,
  not a packed/MMQ prompt kernel.
- `mul_mat_q4_k_swiglu_f32_direct...` fuses an epilogue but still uses the
  direct one-output-row-per-workgroup Q4_K algorithm.
- Generic wide pointwise routes (`mul_f32_generic_wg256`,
  `add_f32_generic_wg256`) still do per-element generic row/column arithmetic
  for regular contiguous or row-broadcast p512 shapes.
- Standalone attention pieces (`mul_mat_f16_f32_batched_attention_wg256`,
  `soft_max_f32_mask...`, `cont_f32`, `set_rows_f32_f16`) remain dispatch and
  memory-traffic surfaces rather than a fused attention path.

`HRX_PROFILE_MODE=all` currently fails without an executable trace capture
filter, so the safe profiling path is `HRX_PROFILE_MODE=dispatch`/queue-level
evidence. A diagnostic run with `GGML_HRX2_DISPATCHES_PER_SUBMIT=1` produced
one queue device event per HRX2 graph dispatch after model-load events:

- Artifact:
  `cache/hrx2/phase2a/profile-dispatch1-llama32-p512-20260615-191308/`.
- The suffix queue-event count exactly matches the `562` p512 dispatches.
- The attribution is coarse but points at wide pointwise/broadcast routes,
  direct Q4_K SWIGLU, attention pieces, and Q4_K prompt matmul as the next
  boulder classes.

The existing opt-in `RMS_NORM -> MUL` fusion was retested under the corrected
measurement:

- Disabled artifact:
  `cache/hrx2/phase2a/rmsmul-retest-disabled-20260615-191553/`.
- Enabled artifact:
  `cache/hrx2/phase2a/rmsmul-retest-enabled-20260615-191609/`.
- Llama 3.2 3B p512: disabled `85.340 tok/s`, enabled `83.910 tok/s`.
- Dispatches drop `562 -> 507`, but device wait rises and throughput regresses.

Decision: do not default-enable the current RMS/MUL fusion. Dispatch-count
reduction alone is not sufficient; wide pointwise/fusion kernels need to be
rewritten and tuned as WYSIWYG kernels before they can carry Phase 2a.

## Pointwise 2D Route Probe

The ADD/MUL generic pointwise routes were extended with opt-in 2D contiguous
and RHS-row-broadcast variants:

- `hrx2_add_f32_contiguous_2d`
- `hrx2_mul_f32_contiguous_2d`
- `hrx2_add_f32_rhs_row_broadcast_2d`
- `hrx2_mul_f32_rhs_row_broadcast_2d`

The route guards prove only the simple cases: contiguous full rows or
zero-stride row broadcast. Dispatch now supports 2D pointwise workgroup geometry
when a route advertises `cols_per_workgroup > 1`.

Validation artifact:

- `cache/hrx2/phase2a/pointwise2d-opgate-20260615-192741/`

Same-binary Llama 3.2 3B Q4_K_M p512 A/B:

- Disabled/default routes:
  `cache/hrx2/phase2a/pointwise2d-llama32-p512-20260615-192819/`,
  `84.940 tok/s`.
- Enabled with `GGML_HRX2_ENABLE_POINTWISE_2D=1`: same artifact,
  `82.661 tok/s`.

Decision: keep these behind `GGML_HRX2_ENABLE_POINTWISE_2D=1`. They are useful
control-plane coverage for row-aware route guards and 2D dispatch, but they are
not a Phase 2a boulder and should not distract from prompt MMQ/attention.

## Blocked Boulder Probe: Q4_K x Q8_1 x4 MMQ

A true packed prompt route was wired as a production-control-plane experiment:

- Loom export: `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`.
- Route id:
  `mul_mat_q4_k_q8_1_x4_mmq32x32_k256_32768_r1_32768_c32_512_wg128`.
- Intended shape class: Q4_K weights, prompt RHS quantized to packed Q8_1 x4,
  32 output rows by 32 prompt columns per workgroup.

The point is structural, not incremental. The accepted F32-RHS cols8 route and
the older Q8_1 direct/cols4 probe are still one-output-row-style kernels. This
MMQ candidate is the class that can plausibly close the prefill cliff by
reusing both Q4_K dequant work and packed RHS data across a larger tile.

Current state:

- The route is not default. It requires both
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1` and
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`.
- With only the broader Q8 prompt opt-in enabled, the focused `MUL_MAT` backend
  op gate remains clean:
  `cache/hrx2/phase2a/q4k-q8-prompt-gated-opgate-20260615-194009/`.
  It produced 49 supported rows, 0 supported errors, 59 HRX2 dispatches, and
  0 `q8_1_x4` route events.
- With the x4-MMQ route enabled on real model p512 shapes, provider compilation
  fails before execution. Observed failing shapes include
  `k=3072, rows=3072, cols=512`, `k=3072, rows=1024, cols=512`, and
  `k=8192, rows=3072, cols=512`.
- Standalone compiler repros:
  `cache/hrx2/phase2a/q4k-q8x4-mmq32-compile-repro-20260615-193610/` and
  `cache/hrx2/phase2a/q4k-q8x4-mmq32-compile-repro-ir-20260615-193621/`.
  The failure is `source-to-low`:
  `INTERNAL; AMDGPU branch argument materializer selected for an unsupported type`.

The same-binary model A/B without the x4 route did not improve:

- Default Llama 3.2 3B p512: `85.076 tok/s`.
- `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`: `84.071 tok/s`.
- Artifact:
  `cache/hrx2/phase2a/q4k-q8x4-mmq32-llama32-p512-20260615-193423/`.

Decision: this confirms the boulder hypothesis but does not land a performance
win yet. The next useful work is to reduce/fix the Loom lowering failure for
the x4-MMQ candidate, then run model-derived backend op rows before another
full model benchmark. More route holes or small pointwise rewrites will not
move p512 while this prompt MMQ class is absent.

## ROPE_SET_ROWS Prompt Route-Domain Cleanup

The Llama 3.2 3B Q4_K_M p512 trace showed an HRX1-derived fusion gap: HRX2
already had `ROPE -> VIEW -> SET_ROWS` fusion code and Loom kernels, but the
route domains were capped at `t1_64`. The p512 K-cache update has
`nheads=8`, `ntokens=512`, and `nrows=4096`, so it fell back to separate ROPE
and SET_ROWS dispatches.

Change made in llama.cpp:

- Widened `rope_set_rows_f32` route domains from token count 64 to 512.
- Renamed route ids from `t1_64` to `t1_512` so traces reflect the actual
  accepted shape range.
- Left the Loom source unchanged; its config declarations already allow
  `@hrx2.shape.rope.ntokens` up to 4096.

Focused backend-op gate after the change:

```bash
build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o ROPE --output csv
build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o SET_ROWS --output csv
```

Artifact:
`cache/hrx2/phase2a/rope-setrows-route512-opgate-20260615-195919/`

- ROPE: 288 rows, 16 supported, 0 supported errors.
- SET_ROWS: 315 rows, 33 supported, 0 supported errors.

Same-binary Llama 3.2 3B Q4_K_M p512 A/B:

| Variant | tok/s | Dispatches | Route effect |
| --- | ---: | ---: | --- |
| Fusion disabled | 85.432 | 562 | 56 standalone ROPE, 56 standalone SET_ROWS |
| `t1_512` fusion enabled | 85.492 | 534 | 28 `ROPE_SET_ROWS`, 28 standalone ROPE, 28 standalone SET_ROWS |

Artifacts:

- Enabled: `cache/hrx2/phase2a/rope-setrows-route512-llama32-p512-20260615-195931/`
- Disabled: `cache/hrx2/phase2a/rope-setrows-route512-disabled-llama32-p512-20260615-200005/`

Interpretation: this is a valid dispatch/copy cleanup and should stay, but it
is performance-neutral on the p512 cliff. It removes 28 dispatches and two
batch flushes, but the remaining top routes are still:

- `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256` x112.
- `mul_mat_f16_f32_batched_attention_wg256` x56.
- Generic RMS/MUL/ADD and attention support kernels.

Vulkan reference on the same model and p512 case:

- Artifact:
  `cache/hrx2/phase2a/rope-setrows-route512-vulkan-llama32-p512-20260615-200137/`
- Vulkan p512: `4793.871 tok/s`.
- Current HRX2 p512: about `85.4 tok/s`, roughly `0.018x` Vulkan.

Queue-profile/trace artifact:
`cache/hrx2/phase2a/rope-setrows-route512-profile-queue-20260615-200237/`

Decision: the user-observed 100% GPU utilization is consistent with real
device work in suboptimal kernels, not CPU fallback or this ROPE_SET_ROWS
copy surface. Continue Phase 2a on the packed Q4_K prompt MMQ boulder and
attention fusions/kernels; small route-domain cleanups are worthwhile but will
not close a 50x prefill gap.

## Q4_K x Packed Q8_1 x4 MMQ Follow-Up

The x4 MMQ route now compiles and routes, but it is not a win yet.

Changes proved in llama.cpp:

- Rewrote the Q4_K group loop in
  `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static` so the eight Q4_K groups are an
  inner `scf.for ... unroll(%eight)` loop. This lets the `group < 4`
  scale/min branch fold before AMDGPU lowering.
- Standalone compile now succeeds for the representative shape
  `k=3072, rows=3072, cols=512`:
  `cache/hrx2/phase2a/q4k-x4mmq-unrolled-group-repro-20260615-201222/`.
  Compile report: peak live units 85, 0 spills, 36.5 KB code.
- The x4 RHS quantizer originally failed because
  `kernel.subgroup.shuffle` used computed lane ids and AMDGPU lowering requires
  exact lanes. Replacing the shuffle packing with per-lane byte stores preserves
  the same packed byte layout and compiles:
  `cache/hrx2/phase2a/q8-1-x4-quantize-byte-store-compile-20260615-201538/`.
  Compile report: peak live units 18, 0 spills, 788 bytes code.
- Added graph-local last-entry Q8_1 RHS caching in HRX2, modeled after Vulkan's
  `prealloc_y_last_tensor_used` behavior. This avoids some repeated Q8_1
  quantize dispatches for adjacent matmuls that share the same F32 RHS tensor.

Validation:

- Focused op gate after the quantizer and runtime-cache changes:
  `cache/hrx2/phase2a/q8-cache-opgate-20260615-201926/`.
  Result: `MUL_MAT`, 1326 rows, 49 supported, 0 supported errors.
- Default-flag guard:
  `cache/hrx2/phase2a/q8-cache-default-guard-llama32-prefill-20260615-202030/`.
  It stayed on `mul_mat_q4_k_f32_cols8...`, 534 dispatches, and remained within
  noise of the previous default (`p64=77.548 tok/s`, `p512=85.066 tok/s`).

Opt-in x4 result on Llama 3.2 3B Q4_K_M:

| Variant | p64 tok/s | p512 tok/s | Dispatches | Top route |
| --- | ---: | ---: | ---: | --- |
| Default cols8 | 78.187 | 85.502 | 534 | `mul_mat_q4_k_f32_cols8... x112` |
| x4 MMQ, no Q8 cache | 35.993 | 82.500 | 646 | `quantize_q8_1_x4... x112` |
| x4 MMQ, Q8 cache | 36.162 | 82.575 | 604 | `mul_mat_q4_k_q8_1_x4_mmq32x32... x112` |

Artifacts:

- Default cols8 reference:
  `cache/hrx2/phase2a/q4k-cols8-straightline-llama32-prefill-20260615-200636/`
- Functional x4 route:
  `cache/hrx2/phase2a/q4k-x4mmq-byte-quant-llama32-prefill-20260615-201603/`
- Functional x4 route plus Q8 cache:
  `cache/hrx2/phase2a/q8-cache-x4-llama32-prefill-20260615-201938/`

Interpretation:

- The user-observed 100% GPU utilization is real device work, but not useful
  work yet. The functional x4 path shifts the top route to the intended MMQ
  kernel and removes the compile fallback, but it is still slower than the
  simpler F32-RHS cols8 route.
- Q8 RHS caching mechanically works, reducing model dispatches from 646 to 604
  and producing 42 `quantize_cache_hit` events, but the throughput barely moves.
  The primary issue is therefore not just redundant quantize dispatch overhead.
- The current x4 MMQ kernel is the next boulder only if its schedule improves
  materially. It uses the right broad algorithm class, but the Loom spelling is
  still not competitive with the existing cols8 route on W7900.

Decision: keep the x4 path opt-in. It is now a valid development artifact and
evidence generator, not a route to promote. The next Phase 2a pass should focus
on MMQ schedule quality versus the HRX1/Vulkan prior art, or move to another
hero fusion if compile-report/ISA inspection shows this x4 schedule is too far
from the intended packed-tile algorithm.

## Fresh Basket Baseline After Phase 2a Prework

Fresh run after commit `dde564fb6`:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag fresh-basket-dde564fb6-20260615-202351 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64,prefill-p64n0,prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 1200 --flash-attn 0
```

Artifact:
`cache/hrx2/phase2a/fresh-basket-dde564fb6-20260615-202351/`

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | HRX2 dispatches | CPU compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 45.077 | 117.953 | 0.382 | 31590 | 0 |
| `phi4-mini-q4` | `prefill-p64n0` | 65.404 | 1480.190 | 0.044 | 610 | 0 |
| `phi4-mini-q4` | `prefill-p512n0` | 67.874 | 4190.465 | 0.016 | 610 | 0 |
| `llama32-3b-q4` | `decode-p1n64` | 46.487 | 139.223 | 0.334 | 27690 | 0 |
| `llama32-3b-q4` | `prefill-p64n0` | 77.481 | 1564.361 | 0.050 | 534 | 0 |
| `llama32-3b-q4` | `prefill-p512n0` | 83.530 | 4773.171 | 0.018 | 534 | 0 |
| `llama31-8b-q4` | `decode-p1n64` | 20.573 | 88.923 | 0.231 | 31590 | 0 |
| `llama31-8b-q4` | `prefill-p64n0` | 32.350 | 1158.721 | 0.028 | 610 | 0 |
| `llama31-8b-q4` | `prefill-p512n0` | 32.892 | 2164.602 | 0.015 | 610 | 0 |

Interpretation:

- The basket has zero CPU compute fallback. The very low p512 ratio is not a
  CPU fallback problem.
- HRX2 can keep the W7900 busy for a long time because it is doing real device
  work, but the work is inefficient. This matches the user-observed 100% GPU
  utilization.
- The main p512 top route for Llama-family models is
  `mul_mat_q4_k_f32_cols8...`; Vulkan's per-op logger shows Q4_K prompt matmul
  buckets running at tens of TFLOP/s-equivalent on the same shapes. HRX2's
  simple cols8 route is the largest prefill boulder.
- Phi p512 also shows `cont_f32_generic_wg256` and F16 attention matmul in the
  top route mix, but quantized prompt matmul is still a hero class across the
  basket.

Submit-batching A/B on decode:

```bash
GGML_HRX2_DISPATCHES_PER_SUBMIT=64 \
python3 tools/hrx2_phase2a_benchmark.py \
  --tag decode-submit64-dde564fb6-20260615-202649 \
  --models phi4-mini-q4,llama32-3b-q4,llama31-8b-q4 \
  --cases decode-p1n64 --backends hrx2 --repetitions 1 --timeout 1200 \
  --flash-attn 0
```

Artifact:
`cache/hrx2/phase2a/decode-submit64-dde564fb6-20260615-202649/`

| Model | Default tok/s | Submit64 tok/s | Speedup |
| --- | ---: | ---: | ---: |
| `phi4-mini-q4` | 45.077 | 44.851 | 0.995x |
| `llama32-3b-q4` | 46.487 | 47.154 | 1.014x |
| `llama31-8b-q4` | 20.573 | 20.950 | 1.018x |

Increasing the submit threshold reduces some flush counts, but tok/s barely
moves. Do not spend the next tranche on submit batching as the primary lever.
Decode still needs dispatch-eliminating fusions and hero-kernel quality, but
the giant p512 gap is kernel/fusion quality first.

## Q4_K x Packed Q8_1 x4 Gate Status Correction

The x4 MMQ route should be treated as a development artifact only. A fresh
model-derived backend op gate over Llama 3.2 p512 prompt rows shows:

- Default no-env route: `mul_mat_q4_k_f32_cols8...` passes all four exported
  rows.
- `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`
  `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`: selects
  `mul_mat_q4_k_q8_1_x4_mmq32x32...` and fails the Q4_K rows with NaNs at
  index 0.

This supersedes the earlier note that the x4 route was "functional" for
production tuning. Keep it opt-in and require a fresh zero-error op gate before
using it in any performance comparison. The next Q4_K prompt-matmul attempt
should be a clean HRX1/Vulkan-inspired tiled kernel or a HIP reference route
that can be compared apples-to-apples.

## Rejected Q4_K/F32 cols16 Direct Route

Probe artifact:
`cache/hrx2/phase2a/q4k-cols16-hrx2-20260615-204347/`

Change attempted and reverted:

- Added `hrx2_mul_mat_q4_k_f32_cols16_static`, a mechanical widening of the
  accepted cols8 route to compute sixteen prompt columns per workgroup.
- Added a priority-165 catalog route above cols8 for `cols_multiple_of=16`.
- The old cols8 route remained as fallback during the probe.

Focused backend-op gate:

```bash
build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase2a/q4k-op-perf-testfile-20260615-183317/llama32-q4k-p512-ops.txt \
  --output csv
```

Result: passed all four exported rows. The Q4_K rows selected
`mul_mat_q4_k_f32_cols16...`; the Q6_K row stayed on the existing Q6 route.

Model result versus the fresh cols8 baseline:

| Model | Case | cols8 tok/s | cols16 tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | `prefill-p64n0` | 65.404 | 62.306 | 0.953x |
| `phi4-mini-q4` | `prefill-p512n0` | 67.874 | 66.501 | 0.980x |
| `llama32-3b-q4` | `prefill-p64n0` | 77.481 | 71.253 | 0.920x |
| `llama32-3b-q4` | `prefill-p512n0` | 83.530 | 82.723 | 0.990x |
| `llama31-8b-q4` | `prefill-p64n0` | 32.350 | 31.200 | 0.964x |
| `llama31-8b-q4` | `prefill-p512n0` | 32.892 | 32.412 | 0.985x |

Decision: rejected and reverted. This confirms that the F32-RHS direct route is
not fixed by simply increasing the number of output columns per workgroup. The
next Q4_K prompt route should change schedule class: packed RHS, real A/B tile
reuse, or a HIP reference modeled after HRX1/Vulkan. Do not repeat cols16 as a
default-route candidate without a materially different schedule.

## Q4_K Packed Q8_1/MMQ Correctness Isolation

Current accepted fallback is still the F32-RHS cols8 route. Two correctness-clean
packed-RHS probes show why they are not enough:

- Non-x4 Q8_1 cols4 prompt route passes the focused Q4_K op gate, but is only
  flat to +3% on p512 and regresses some p64 cases. Artifact:
  `cache/hrx2/phase2a/q4k-q8-cols4-hrx2-20260615-205729/`.
- A temporary x4-direct diagnostic route consumed the packed x4 RHS layout
  without MMQ/LDS tiling and passed the same focused Q4_K op gate. Artifact:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-diagnostic-20260615-210115/`.

The high-priority x4/MMQ route still fails the same op gate with NaNs:

- Current unmodified MMQ failure:
  `cache/hrx2/phase2a/q4k-x4-current-op-test-20260615-205701/`.
- Diagnostic with Q8 scale/sum loaded directly from global instead of the f16
  LDS side buffer still failed:
  `cache/hrx2/phase2a/q4k-x4-mmq-global-ds-diagnostic-20260615-210230/`.
- Diagnostic with Q8 payload and scale/sum both loaded directly from global
  instead of LDS still failed:
  `cache/hrx2/phase2a/q4k-x4-mmq-global-payload-diagnostic-20260615-210331/`.

Interpretation: the x4 quantizer and packed x4 memory layout are likely sound.
The NaN is in the MMQ kernel's row/column lane mapping, accumulator/control
spelling, or Q4/Q8 arithmetic structure, not simply an LDS staging hazard. The
next serious prefill push should replace or simplify
`hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static` around a correctness-first tiled
reference, then reintroduce MMQ schedule optimizations under backend-op gates.

## Rejected F16 Batched Attention cols4 Direct Route

Probe artifact:
`cache/hrx2/phase2a/f16-batched-cols4-hrx2-20260615-205222/`

Change attempted and reverted:

- Added `hrx2_mul_mat_f16_f32_batched_cols4`, a direct scalar F16/F32 batched
  attention matmul route that computes four adjacent prompt columns per
  workgroup.
- Added a priority-120 catalog route guarded by `cols_multiple_of=4`, leaving
  the existing `cols=1` decode path on `mul_mat_f16_f32_batched_attention_wg256`.

Infrastructure fix kept:

- `mul_mat_f16_f32_routes` was collected but not sorted by priority. This meant
  the lower-priority scalar fallback won even when a narrower high-priority F16
  route matched. The route-family priority sort was added and retained.

Focused backend-op gate:

```bash
build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase2a/p512-c512-op-export-20260615-103625/basket-p512-c512-f16-attention-ops.txt \
  --output csv
```

Result: passed all eight exported rows. Four `cols=512` rows selected the
cols4 provider; four `cols=1` rows stayed on the scalar provider. Artifact:
`cache/hrx2/phase2a/f16-cols4-op-test-20260615-205158/`.

Model result versus the fresh phase2a baseline:

| Model | Case | baseline tok/s | cols4 tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | `decode-p1n64` | 45.077 | 45.188 | 1.002x |
| `phi4-mini-q4` | `prefill-p64n0` | 65.404 | 63.438 | 0.970x |
| `phi4-mini-q4` | `prefill-p512n0` | 67.874 | 68.948 | 1.016x |
| `llama32-3b-q4` | `decode-p1n64` | 46.487 | 46.756 | 1.006x |
| `llama32-3b-q4` | `prefill-p64n0` | 77.481 | 74.770 | 0.965x |
| `llama32-3b-q4` | `prefill-p512n0` | 83.530 | 86.401 | 1.034x |
| `llama31-8b-q4` | `decode-p1n64` | 20.573 | 20.640 | 1.003x |
| `llama31-8b-q4` | `prefill-p64n0` | 32.350 | 32.198 | 0.995x |
| `llama31-8b-q4` | `prefill-p512n0` | 32.892 | 33.503 | 1.019x |

Decision: rejected and reverted. The route is correctness-clean but not a
Phase 2a bulk lift. It regresses the p64 basket and only gives small p512
movement. The saturated-GPU prefill gap remains dominated by Q4_K prompt
matmul schedule quality and, for Phi, copy/contiguous and attention-route
fusion quality. Future F16 attention work should change schedule class or
fusion shape, not repeat adjacent-column scalar widening as a default route.

## Accepted Phi CONT n128 Vec4 Route

Phi prefill traces repeatedly surfaced `cont_f32_generic_wg256` as a visible
route family. The accepted route specializes the common `ncols=128` F32
contiguous-materialization shape and copies `vector<4xf32>` per workitem
instead of one scalar F32 per workitem.

Accepted route:

- `cont_f32_n128_vec4_wg256`
- Loom export: `hrx2_cont_f32_n128_vec4`
- Domain: F32 `CONT`, `ncols=128`, `ncols_multiple_of=4`,
  target-generic source.
- Rollback: `GGML_HRX2_DISABLE_CONT_VEC4=1`.

Focused validation:

- Build:
  `cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops llama-bench`.
- Catalog assembly and validation:
  `cache/hrx2/phase2a/cont-vec4-regate-20260615-214912/`.
- Phi p512 exported op gate:
  `cache/hrx2/phase2a/cont-vec4-regate-20260615-214912/phi4-p512-ops.csv`.
  The command exited successfully; the six unsupported rows were unrelated
  attention `MUL_MAT` and `SOFT_MAX` rows. The route trace compiled and selected
  `cont_f32_n128_vec4_wg256` for four model-derived `CONT` shapes.
- Generic `CONT` suite:
  `cache/hrx2/phase2a/cont-vec4-regate-20260615-214912/cont-suite.csv`.
  Existing unsupported rows remained unsupported; supported F32 rows passed.

Same-binary three-model prefill A/B, three repetitions per case:

| Model | Case | Enabled tok/s | Disabled tok/s | Speedup | Dispatches | CPU compute | Top enabled | Top disabled |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `llama31-8b-q4` | `prefill-p512n0` | 32.731 | 32.514 | 1.0067x | 1830 | 0 | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x384` | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x384` |
| `llama31-8b-q4` | `prefill-p64n0` | 32.620 | 32.520 | 1.0031x | 1830 | 0 | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x384` | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x384` |
| `llama32-3b-q4` | `prefill-p512n0` | 82.313 | 81.560 | 1.0092x | 1602 | 0 | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x336` | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x336` |
| `llama32-3b-q4` | `prefill-p64n0` | 80.851 | 79.391 | 1.0184x | 1602 | 0 | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x336` | `mul_mat_q4_k_f32_cols8_k256_32768_c8_512_wg256 x336` |
| `phi4-mini-q4` | `prefill-p512n0` | 67.068 | 65.133 | 1.0297x | 1830 | 0 | `cont_f32_n128_vec4_wg256 x192` | `cont_f32_generic_wg256 x192` |
| `phi4-mini-q4` | `prefill-p64n0` | 68.274 | 64.151 | 1.0643x | 1830 | 0 | `cont_f32_n128_vec4_wg256 x192` | `cont_f32_generic_wg256 x192` |

Artifacts:

- Enabled:
  `cache/hrx2/phase2a/cont-vec4-ab-current-20260615-215003-enabled/`.
- Disabled:
  `cache/hrx2/phase2a/cont-vec4-ab-current-20260615-215003-disabled/`.

Interpretation: accept this as a small but real copy/materialization cleanup.
It is not a Phase 2a bulk lift. The same run still shows Q4_K prompt matmul as
the top Llama prefill blocker, and Phi remains dominated by attention,
pointwise/fusion, and quantized matmul after the `CONT` route improves.

## Current Q4_K Prompt Matmul Prior Audit Checkpoint

Artifact directory:
`cache/hrx2/phase2a/prior-asm-20260615-211124/`

This checkpoint was created after repeated local HRX2/Loom route tweaks failed
to move the prefill basket. The conclusion is structural: the current HRX2
Q4_K prompt route is not in the same schedule family as the known-good
Vulkan/HRX1 MMQ-style implementations. Do not resume by widening the current
direct route or tuning small local constants.

Current HRX2/Loom x4 MMQ evidence:

- Linked root:
  `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`.
- Compiled artifact:
  `q4k_x4_mmq.native.hsaco`.
- ISA contains 512 `v_dot4_i32_iu8`, no spills in the compile report, and
  peak live units around 85. Loom can emit the dot primitive.
- The same kernel has very high scalarized traffic relative to priors:
  about 292 `global_load_b32`, 512 `ds_load_b32`, 128 `ds_load_u16`, 16
  barriers, and many f16/f32 conversions.
- Focused model-derived op gate still fails the Q4_K rows with NaNs:
  `cache/hrx2/phase2a/q4k-x4-current-op-test-20260615-205701/`.
- Follow-up scale-path diagnostics after the rebuild confirmed this is not a
  simple dot-payload failure:
  - fresh repro:
    `cache/hrx2/phase2a/q4k-x4-repro-20260615-211932/`;
  - forcing Q4 scale/min finite moved but did not remove NaNs:
    `cache/hrx2/phase2a/q4k-x4-diag-q4scale1-20260615-212014/`;
  - forcing both Q4 and Q8 scale/sum finite removed NaNs and left only expected
    numerical mismatch:
    `cache/hrx2/phase2a/q4k-x4-diag-allscale1-20260615-212051/`;
  - direct global Q8 `d/s` loads and explicit i8 Q4 scale-byte loads did not
    fix the NaN:
    `cache/hrx2/phase2a/q4k-x4-direct-ds-20260615-212302/`,
    `cache/hrx2/phase2a/q4k-x4-byte-scale-direct-ds-20260615-212434/`.
  The temporary diagnostic source diff was reverted and saved at
  `cache/hrx2/phase2a/q4k-x4-diagnostic-edits.patch`.

Vulkan prior facts:

- Shader: `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`.
- For the relevant non-coopmat integer MMQ family, the shader uses explicit
  tile constants such as `BM=64`, `BN=64`, `WM=32`, `WN=32`, `TM=4`,
  `TN=2`, `WARP=32`, and staged A/B workgroup memory.
- The generated SPIR-V for `matmul_q4_k_q8_1.spv` includes `OpSDot` in the
  inner loop and workgroup memory barriers around staged tiles.

HRX1 HIP prior facts:

- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp`.
- Closest symbols:
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32` and
  `hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32`.
- Schedule family: `BM=64`, `BK_STEP=1`, `BLOCK_SIZE=64`, `WARP=64`,
  `TM=4`, `TN=1 or 2`, cooperative staging of Q4_K A rows and Q8_1 B columns,
  then each lane computes a small `TM x TN` output tile using eight dot4
  operations per output.
- Extracted symbol ISA shows much lower pressure and fewer barriers than the
  current Loom MMQ probe: roughly 107-134 max VGPR, 128-256 dot4 instructions,
  18 `global_load_b32`, 2 barriers, wide LDS loads such as `ds_load_b128`, and
  16-32 output stores depending on the tile.

Next implementation direction:

- Build a clean plain `MUL_MAT` Loom root modeled on the HRX1/Vulkan family,
  not the HRX1 MoE routing ABI. Start with a correctness-first tile such as
  BM64/BN16/TM4/TN1/wg64 or BM64/BN32/TM4/TN2/wg64.
- Keep the source target-generic unless target-specific WMMA or ISA layout is
  introduced. Route selection and metadata should carry target and shape
  applicability.
- Reuse the existing Q8_1/x4 quantizer cautiously. The dot payload path can
  produce finite values when scale metadata is forced finite, but the current
  route still has a scale/metadata correctness failure. A clean rewrite should
  make metadata ownership and layout explicit rather than patching the existing
  32x32 source in place.
- Validate in this order: standalone Loom compile report and ISA, focused
  `test-backend-ops` exported Q4_K rows, then p64/p512 basket A/B against the
  default F32-RHS cols8 route and Vulkan.

## Q4_K x4 MMQ Correctness Follow-Up: Metadata Patches Rejected

Follow-up artifact:
`cache/hrx2/phase2a/q4k-x4-failed-experiment-20260615-221511/`

The current x4 MMQ source was tested with a focused model-derived Q4_K op gate
after several local metadata/layout patches:

- Q8_1 `d/s` metadata in LDS was made single-writer.
- Q8_1 `d/s` metadata was stored as f32 in LDS.
- The x4 quantizer requested subgroup size 32 at dispatch.
- Q8 payload packing was spelled as one explicit i32 store per four q bytes.
- Q4 scale/min byte loads in the x4 consumer were changed from unaligned i32
  views to explicit i8 loads.

Focused gate:
`cache/hrx2/phase2a/q4k-x4-byte-scale-20260615-221046/`

Result: rejected. The NaNs were removed by the f32/single-writer metadata
change, but the two c64 Q4_K rows still failed CPU-reference comparison:

```text
q4_k2048_r4096_c64: ERR = 1.007537122 > 0.000500000
q4_k4096_r4096_c64: ERR = 1.003691523 > 0.000500000
```

The same op file with only the non-x4 Q8_1 cols4 route enabled had already
passed, so this is not a generic Q4_K/Q8_1 backplane failure. The local patches
were saved and reverted from `sources/llama.cpp`; they are diagnostic evidence,
not a candidate to resume from.

Standalone Loom compile artifacts:
`cache/hrx2/phase2a/q4k-x4-standalone-compile-20260615-221313/`

Facts from the emitted HSACOs:

- `hrx2_quantize_q8_1_x4_f32`: wavefront size 32, workgroup size 128, 12 VGPR,
  14 SGPR, no LDS, no spills.
- `hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`: wavefront size 32, workgroup
  size 128, 1280 bytes LDS, 153 VGPR, 14 SGPR, no spills.
- The x4 matmul emits `v_dot4_i32_iu8`; this is not blocked on emitting the dot
  primitive.
- The x4 quantizer uses cross-lane operations consistent with subgroup
  reduction/packing; the HSACO metadata confirms wave32, so the earlier
  wide-subgroup contamination theory is not supported by the current artifact.

Decision: stop patching the current 32x32/wg128 MMQ route. It is both
correctness-failing and structurally far from the HRX1/Vulkan schedule family.
The next serious implementation should be a clean BM64/BN16-or-BN32/TM4/TN1-or
TN2 wg64 Loom route modeled on HRX1/Vulkan, or a narrow diagnostic consumer
that proves the exact x4 layout/lane-mapping bug before any more tuning.

## Q4_K x4 Layout Diagnostic: Direct Consumer Passed

Follow-up artifacts:

- Clean non-x4 control:
  `cache/hrx2/phase2a/q4k-direct-diag-control-20260615-222156/`.
- Corrected x4 direct-cols4 diagnostic:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-wg256-diag-20260615-222302/`.
- Saved, reverted source patch:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-passing-diagnostic-20260615-222414/passing-diagnostic.patch`.

The temporary diagnostic route added
`hrx2_mul_mat_q4_k_q8_1_x4_direct_cols4_static`: a direct four-column Q4_K x
packed-Q8_1-x4 consumer with no MMQ/LDS tiling. It used the same opt-in x4
gate as the MMQ route only for the probe, then was reverted so it would not
shadow the failing MMQ route in production testing.

Focused gate:

```bash
GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1 \
GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1 \
GGML_HRX2_TRACE_JSONL="$OUT/hrx2.jsonl" \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  --test-file cache/hrx2/phase1_0/route-slice-32-q4-focused-current/mul_mat_q4_k_f32_ops.txt \
  --output csv
```

Result: passed all eight model-derived Q4_K focused rows. The trace selected:

- `mul_mat_q4_k_f32_direct_k256_32768_c1_512_wg256`: 12 events.
- `mul_mat_q4_k_q8_1_x4_direct_cols4_k256_32768_c4_512_wg256`: 12 events.
- `quantize_q8_1_x4_f32_generic_wg128`: 12 events.

The corrected diagnostic compiled the direct route with
`@hrx2.tuning.workgroup_size=256`. This is stronger evidence than the earlier
scale-forcing probes: the packed x4 Q8_1 quantizer/layout can feed a Q4_K
consumer that matches the CPU reference on the same rows where the 32x32 MMQ
route fails.

Decision: the x4 layout is no longer the main suspect. Treat the current MMQ
failure as a consumer schedule/lane-mapping/metadata-use bug in
`hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static`. Do not restore the diagnostic
route as a candidate because it is direct and slow by construction. The next
throughput attempt remains a clean HRX1/Vulkan-style tiled MMQ rewrite, not
more patching of the 32x32/wg128 source.

## Q4_K Dot Signedness Correction

Loom's HIP authoring corpus and the quantization math both say Q4_K x Q8_1
dot products should be spelled as unsigned-Q4 times signed-Q8:
`vector.dot4i<u8s8>`. The previous HRX2 Q4_K Loom source used `s8s8` in the
direct Q8_1 prompt routes and the x4 MMQ route.

Patch under test:
`sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`

Focused control gate:
`cache/hrx2/phase2a/q4k-u8s8-control-20260615-222811/`

- `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`
- Passed all eight model-derived Q4_K backend-op rows.
- Selected the accepted direct c1 route, non-x4 Q8_1 cols4 route, and
  `quantize_q8_1_f32_generic_wg32`.

Focused x4 MMQ gate:
`cache/hrx2/phase2a/q4k-u8s8-x4mmq-20260615-222824/`

- `GGML_HRX2_ENABLE_Q4_K_Q8_1_PROMPT=1`
- `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`
- Still failed the two c64 rows with NaNs at index 0.
- Selected `mul_mat_q4_k_q8_1_x4_mmq32x32...` for the c64 rows.

Decision: keep `u8s8` as the correct WYSIWYG spelling for Q4_K x Q8_1
dot products, but do not treat it as the root-cause fix for the x4 MMQ route.
The remaining boulder is still the 32x32 MMQ consumer/schedule, and the next
candidate should be the clean BM64/BN16-or-BN32 wg64 schedule.

## Q4_K BM64/BN8 x4 MMQ Rewrite Diagnostic

New diagnostic attempt:
`hrx2_mul_mat_q4_k_q8_1_x4_mmq64x8_static`, BM64/BN8/TM4/wg64, packed Q8_1 x4
RHS, `vector.dot4i<u8s8>`.

The initial flattened `%kb` loop spelling failed Loom `source-to-low`:

```text
AMDGPU branch argument materializer selected for an unsupported type
```

BN8 did not fix that by itself. Rewriting to the old compile-friendly topology
did: outer `%q4_block_iter` loop plus inner unrolled `%group` loop. Compile
artifact:
`cache/hrx2/phase2a/q4k-mmq64x8-nested-group-compile-repro-20260615-225606/`.

Focused backend-op findings:

- Nested BM64/BN8 fully staged A+B route JIT-compiled and dispatched, but
  failed with finite `ERR ~= 1.0`:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-nested-opgate-20260615-225642/`.
- Removing the Q4_K min correction barely changed the failure:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-no-min-diag-20260615-225824/`.
- B direct from global while A was staged still failed:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-b-global-diag-20260615-225944/`.
- A and B direct from global passed all eight focused Q4_K rows:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-a-b-global-diag-20260615-230050/`.
- A scalar-i32 LDS plus B global still failed:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-a-scalar-lds-b-global-diag-20260615-230303/`.
- A global plus B LDS still failed:
  `cache/hrx2/phase2a/q4k-x4-mmq64x8-a-global-b-lds-diag-20260615-230410/`.

Small model smoke:
`cache/hrx2/phase2a/q4k-x4-mmq64x8-smoke-20260615-230153/`.

| Variant | Llama 3.2 3B Q4_K_M p64/n0 | Top Q4 route |
| --- | ---: | --- |
| Accepted non-x4 Q8_1 fallback | 79.58 tok/s | `mul_mat_q4_k_q8_1_f32_cols4...` |
| Correct A+B-global x4 diagnostic | 22.39 tok/s | `mul_mat_q4_k_q8_1_x4_mmq64x8...` |

Decision: reject this route as a performance candidate. It establishes useful
boundaries but is not a bulk lift. The route/lane math can be correct when the
integer payloads are loaded from global memory, but either A or B integer
payload reuse through Loom workgroup memory breaks correctness in this shape.
The next Q4_K path should be either a small standalone integer-LDS reproducer
for the Loom author or a different staging/low-level spelling before more MMQ
performance tuning. Current diagnostic patches were saved at
`cache/hrx2/phase2a/q4k-mmq64x8-diagnostic-patches-20260615-230451/`.

## Accepted F16 Batched Attention c512 Cols8 Route

After the Q4_K staging boundary, the next prior-driven prompt candidate was the
batched F16 x F32 attention matmul. HRX2's previous Loom route computed one
output column per workgroup:
`hrx2_mul_mat_f16_f32_batched`. HRX1 has a proven family of batched F16
attention kernels that compute multiple output columns per workgroup; the
conservative HRX2 port added an eight-column variant:
`hrx2_mul_mat_f16_f32_batched_cols8`.

Implementation:

- Source: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_f16_f32_batched.loom`
- Route: `mul_mat_f16_f32_batched_attention_cols8_wg256`
- Shape domain: k=128..512, rows=128..512, cols=512 only.
- Reason for tight domain: p512/cols512 improved modestly, but p64/narrow
  prompt smoke regressed when the route was allowed for all cols>=8.

Focused backend-op gates:

- Initial focused attention rows:
  `cache/hrx2/phase2a/f16-cols8-opgate-20260615-231845/`
- Exact c512 rows:
  `cache/hrx2/phase2a/f16-cols8-c512-opgate-20260615-231919/`
- Tightened c512 route:
  `cache/hrx2/phase2a/f16-cols8-c512-tight-opgate-20260615-232119/`

All focused c512 rows passed and the trace selected the new cols8 route for the
c512 attention rows while leaving c1 rows on the scalar fallback route.

Smoke evidence against the fresh reduced baseline
`cache/hrx2/phase2a/current-reduced-20260615-231115/`:

| Model | Case | Baseline HRX2 | Cols8 HRX2 | Change |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | `prefill-p512n0` | 68.399 | 69.018 | +0.90% |
| `llama32-3b-q4` | `prefill-p512n0` | 84.297 | 86.077 | +2.11% |

Negative/narrow-shape evidence:

- `cache/hrx2/phase2a/f16-cols8-p64-smoke-20260615-232032/` selected the
  cols8 route and regressed p64 smoke.
- The route was then tightened to `cols=512`.
- `cache/hrx2/phase2a/f16-cols8-tight-p64-smoke-20260615-232132/` confirmed
  p64 returned to the scalar fallback route.

Decision: accept the p512-only cols8 route as a small, prior-driven lift. It
reduces workgroups for a real attention bucket, but it is not the Phase 2a bulk
prefill boulder. The next large prompt work remains Q4_K/Q5_K/Q6_K prompt
matmul quality and attention-chain/fusion candidates.

## Accepted CONT -> SET_ROWS V-Cache Fusion

Date: 2026-06-16.

Trace evidence showed a repeated Phi V-cache update pattern:

```text
CONT Vcur-N -> RESHAPE -> RESHAPE cache_v_lN -> SET_ROWS cache_v_lN
```

The unfused path materialized the contiguous `Vcur-N (cont)` tensor and then
immediately read it back to write the V cache. The accepted route fuses the
CONT linearization into the SET_ROWS write for the observed f32-to-f16,
`ncols=128` V-cache shape:

- Source: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/cont_set_rows_f32.loom`
- Route: `cont_set_rows_f32_f16_n128_wg256`
- Dispatch trace op: `CONT_SET_ROWS`
- Rollback: `GGML_HRX2_DISABLE_CONT_SET_ROWS_FUSION=1`
- Applicability guard: f32 CONT source, f16 SET_ROWS destination,
  zero-offset view/reshape chain, `cont.ncols=128`, `set_rows.nc=1`, and
  `set_rows.ne1 <= 1048576`.

Validation and A/B artifacts:

- Focused primitive gate before model testing:
  `cache/hrx2/phase2a/cont-setrows-opgate-20260615-234407/`
- Final focused primitive gate after the large-cache guard:
  `cache/hrx2/phase2a/cont-setrows-opgate-final-20260615-235916/`
- Initial fusion smoke:
  `cache/hrx2/phase2a/cont-setrows-fusion-smoke-20260615-234541/`
- Same-binary opt-out p64/p512 A/B:
  `cache/hrx2/phase2a/cont-setrows-fusion-p64-p512-ab-20260615-234817/`
- Bounded cache-use smoke:
  `cache/hrx2/phase2a/cont-setrows-fusion-p64n1-smoke-20260615-235743/`

Same-binary A/B on Phi-4-mini Q4_K_M:

| Case | Fusion on | Fusion off | Change | Dispatch change |
| --- | ---: | ---: | ---: | ---: |
| `prefill-p64n0` | 67.965 tok/s | 65.972 tok/s | +3.02% | 1734 vs 1830 over 3 reps |
| `prefill-p512n0` | 69.028 tok/s | 67.834 tok/s | +1.76% | 1734 vs 1830 over 3 reps |

Current reduced HRX2/Vulkan comparison after this route:
`cache/hrx2/phase2a/current-reduced-after-cont-setrows-20260615-235800/`.

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan | HRX2 dispatches | CPU compute | Top blocker |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `llama32-3b-q4` | `decode-p1n64` | 46.858 | 139.587 | 0.3357 | 27690 | 0 | `mul_mat_q4_k_f32_direct... x7280` |
| `llama32-3b-q4` | `prefill-p512n0` | 86.498 | 4728.105 | 0.0183 | 534 | 0 | `mul_mat_q4_k_f32_cols8... x112` |
| `llama32-3b-q4` | `prefill-p64n0` | 78.285 | 1600.341 | 0.0489 | 534 | 0 | `mul_mat_q4_k_f32_cols8... x112` |
| `phi4-mini-q4` | `decode-p1n64` | 45.481 | 116.645 | 0.3899 | 29510 | 0 | `mul_mat_f16_f32_batched_attention_wg256 x4160` |
| `phi4-mini-q4` | `prefill-p512n0` | 70.071 | 4265.776 | 0.0164 | 578 | 0 | `mul_mat_f16_f32_batched_attention_cols8_wg256 x64` |
| `phi4-mini-q4` | `prefill-p64n0` | 65.542 | 1504.848 | 0.0436 | 578 | 0 | `mul_mat_f16_f32_batched_attention_wg256 x64` |

Decision: accept. This is not the bulk prefill breakthrough, but it is a
measured structural fusion that removes one V-cache copy/writeback dispatch per
Phi layer where the configured KV-cache shape is within the route domain. It
also shifted Phi prefill top blockers away from `CONT` and back to attention
and quantized prompt matmul, which is the intended Phase 2a narrowing.

## Accepted Q5_K Packed-Q8_1 x4 Direct Prompt Route

Date: 2026-06-16.

The Phi-4-mini Q4_K_M p512 exported op set includes a prompt Q5_K matmul:

```text
MUL_MAT wqkv-0: q5_K[3072,5120] x f32[3072,512] -> f32[5120,512]
```

The previous HRX2 route used the generic Q5_K x F32 direct path, one output
column per workgroup. HRX1 and Vulkan priors both indicate that prompt K-quants
should first quantize RHS activations to Q8_1 and consume packed dot4 payloads.

Accepted implementation:

- Source: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q5_k_f32.loom`
- Route: `mul_mat_q5_k_q8_1_x4_direct_cols4_k256_32768_r1_262144_c4_512_wg256`
- Runtime gates:
  - `GGML_HRX2_ENABLE_Q5_K_Q8_1_PROMPT=1`
  - `GGML_HRX2_ENABLE_Q5_K_Q8_1_X4_PROMPT=1`
- Algorithm: direct Q5_K x packed-Q8_1-x4, four prompt columns per workgroup,
  explicit Q5 high-bit packing, `vector.dot4i<u8s8>`, and existing x4 Q8_1 RHS
  quantizer. A post-commit audit caught an earlier `s8s8` spelling; the
  committed route uses the intended unsigned-Q5 by signed-Q8 dot form.

Validation artifacts:

- Failing MMQ probe, selected and JIT compiled but produced NaN:
  `cache/hrx2/phase2a/q5k-x4-mmq32x32-opgate-20260616-005220/`.
- Direct x4 diagnostic passed the full Phi p512 exported op file:
  `cache/hrx2/phase2a/q5k-x4-direct-cols4-opgate-20260616-005533/`.
- Clean final op gate after removing the failing MMQ route and renaming the
  env gate:
  `cache/hrx2/phase2a/q5k-x4-direct-clean-opgate-20260616-005818/`.
- Focused two-op gate after correcting the direct route to `u8s8`:
  `cache/hrx2/phase2a/q5k-x4-direct-u8s8-focused-opgate-20260616-010403/`.

Reduced Phi prefill comparison against the latest baseline:

| Case | Baseline HRX2 | Q5 x4 direct HRX2 | Change | Vulkan | HRX2/Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill-p64n0` | 65.521 tok/s | 69.299 tok/s | +5.77% | 1451.435 tok/s | 0.0477 |
| `prefill-p512n0` | 70.075 tok/s | 78.575 tok/s | +12.13% | 4266.070 tok/s | 0.0184 |

Final reduced artifact:
`cache/hrx2/phase2a/q5k-x4-direct-u8s8-reduced-20260616-010904/`.

Decision: accept as a correctness-clean structural improvement and as a
validated Q5_K/Q8_1-x4 layout substrate. This is not the final prompt matmul
answer: dispatch count increases from 578 to 610 because the route adds 32 RHS
quantize dispatches, and the top blockers remain attention and quantized prompt
matmul. The next Q5_K boulder is a correct A/B-staged MMQ schedule equivalent
to the HRX1/Vulkan prior, not more tuning of this direct route.

Follow-up policy change: after the Q6_K route below also passed, Q5_K/Q8_1-x4
was changed from opt-in to default-on for prompt shapes. Rollback variables:

- `GGML_HRX2_DISABLE_Q5_K_Q8_1_PROMPT=1`
- `GGML_HRX2_DISABLE_Q5_K_Q8_1_X4_PROMPT=1`

Default-on focused gate:
`cache/hrx2/phase2a/q5k-packed-default-on-opgate-20260616-012519/`.
Opt-out focused gate:
`cache/hrx2/phase2a/q5k-packed-optout-opgate-20260616-012703/`.

## Accepted Q6_K Packed-Q8_1 x4 Direct Prompt Route

Date: 2026-06-16.

The Phi-4-mini Q4_K_M p512 exported op set includes two prompt Q6_K matmuls,
including the large output projection:

```text
MUL_MAT ffn_out-0: q6_K[8192,3072] x f32[8192,512] -> f32[3072,512]
MUL_MAT result_output: q6_K[3072,200064] x f32[3072,512] -> f32[200064,512]
```

Accepted implementation:

- Source: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q6_k_f32.loom`
- Route: `mul_mat_q6_k_q8_1_x4_direct_cols4_k256_32768_r1_262144_c4_512_wg256`
- Runtime gates:
  - `GGML_HRX2_ENABLE_Q6_K_Q8_1_PROMPT=1`
  - `GGML_HRX2_ENABLE_Q6_K_Q8_1_X4_PROMPT=1`
- Algorithm: direct Q6_K x packed-Q8_1-x4, four prompt columns per workgroup,
  explicit signed Q6 unpacking to `i8`, `vector.dot4i<s8s8>`, and the existing
  x4 Q8_1 RHS quantizer. Q6_K has no min-correction term, so the packed route
  only applies the Q6 block scale and Q8_1 block scale.

Validation artifacts:

- Focused four-op Q6 gate with route and quantizer selected:
  `cache/hrx2/phase2a/q6k-x4-direct-opgate-20260616-011732/`.
- Default-off focused gate proving the new route does not shadow the existing
  Q6 routes without env gates:
  `cache/hrx2/phase2a/q6k-x4-direct-default-off-opgate-20260616-011921/`.

Focused trace summary:

- `mul_mat_q6_k_q8_1_x4_direct_cols4...`: selected for the two prompt rows.
- `mul_mat_q6_k_f32_rows2...`: selected for the two decode rows.
- `quantize_q8_1_x4_f32_generic_wg128`: selected for the two prompt rows.
- `provider_unavailable` / `dispatch_failed`: zero.

Reduced Phi prefill comparison:

| Case | Baseline HRX2 | Q5 x4 direct HRX2 | Q5+Q6 x4 direct HRX2 | Change vs Q5 | Vulkan | HRX2/Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill-p64n0` | 65.521 tok/s | 69.299 tok/s | 74.908 tok/s | +8.09% | 1859.586 tok/s | 0.0403 |
| `prefill-p512n0` | 70.075 tok/s | 78.575 tok/s | 79.770 tok/s | +1.52% | 4710.067 tok/s | 0.0169 |

Final reduced artifact:
`cache/hrx2/phase2a/q5-q6-x4-direct-reduced-20260616-011820/`.

Decision: accept as an opt-in, correctness-clean Q6_K/Q8_1-x4 substrate and a
small real prefill lift. This is still not the final Q6_K prompt matmul answer:
the route adds 31 more RHS quantize dispatches on Phi prefill, taking total
dispatches from 610 to 625 in the Q5+Q6 run. The next Q6_K boulder is a staged
MMQ/tiled schedule that reuses RHS work across rows and columns instead of
direct global loads per output row.

Follow-up policy change: Q6_K/Q8_1-x4 was changed from opt-in to default-on for
prompt shapes after a two-model run showed consistent gains and the opt-out
gate proved rollback. Rollback variables:

- `GGML_HRX2_DISABLE_Q6_K_Q8_1_PROMPT=1`
- `GGML_HRX2_DISABLE_Q6_K_Q8_1_X4_PROMPT=1`

Default-on focused gate:
`cache/hrx2/phase2a/q6k-packed-default-on-opgate-20260616-012519/`.
Opt-out focused gate:
`cache/hrx2/phase2a/q6k-packed-optout-opgate-20260616-012703/`.

Default-on reduced two-model comparison:
`cache/hrx2/phase2a/q5-q6-packed-default-on-two-model-20260616-012557/`.

| Model | Case | Old default HRX2 | New default HRX2 | Change | Vulkan | HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | `prefill-p64n0` | 78.001 tok/s | 82.705 tok/s | +6.03% | 1561.106 tok/s | 0.0530 |
| `llama32-3b-q4` | `prefill-p512n0` | 86.501 tok/s | 94.325 tok/s | +9.04% | 4705.305 tok/s | 0.0200 |
| `phi4-mini-q4` | `prefill-p64n0` | 65.521 tok/s | 74.009 tok/s | +12.96% | 1458.591 tok/s | 0.0507 |
| `phi4-mini-q4` | `prefill-p512n0` | 70.075 tok/s | 84.651 tok/s | +20.80% | 4247.906 tok/s | 0.0199 |

No provider failures or CPU compute fallback were present in the default-on
reduced run. Remaining top blockers are unchanged in kind: Llama is dominated
by Q4_K prompt matmul, while Phi is dominated by F16 attention matmul plus the
unfused attention/elementwise chain.

## Accepted Q4_K Direct-Q8_1 Prompt Default

Date: 2026-06-16.

After the Q5_K and Q6_K packed prompt routes were default-enabled, the existing
Q4_K direct Q8_1 route was re-tested against the current two-model p64/p512
prefill slice. This is not the final Q4_K MMQ schedule. It is the conservative
direct route that quantizes the F32 RHS to contiguous Q8_1 and computes four
prompt columns per workgroup:

- Source: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`
- Route: `mul_mat_q4_k_q8_1_f32_cols4_k256_32768_c4_512_wg256`
- Quantizer: `quantize_q8_1_f32_generic_wg32`
- Rollback: `GGML_HRX2_DISABLE_Q4_K_Q8_1_PROMPT=1`
- Still-disabled MMQ probe: `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`

Focused backend gates:

- Default-on Q4 gate:
  `cache/hrx2/phase2a/q4-q8-direct-default-on-opgate-20260616-013704/`.
- Opt-out rollback gate:
  `cache/hrx2/phase2a/q4-q8-direct-optout-opgate-20260616-013720/`.

No-env production-flow comparison against
`cache/hrx2/phase2a/current-prefill-trace-20260616-013218/`:

| Model | Case | Previous HRX2 | New HRX2 | Change | Vulkan | New HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | `prefill-p64n0` | 74.447 tok/s | 75.762 tok/s | +1.77% | 1470.342 tok/s | 0.0515 |
| `phi4-mini-q4` | `prefill-p512n0` | 84.973 tok/s | 85.917 tok/s | +1.11% | 4193.248 tok/s | 0.0205 |
| `llama32-3b-q4` | `prefill-p64n0` | 83.225 tok/s | 83.656 tok/s | +0.52% | 1607.125 tok/s | 0.0521 |
| `llama32-3b-q4` | `prefill-p512n0` | 94.881 tok/s | 95.739 tok/s | +0.90% | 4737.141 tok/s | 0.0202 |

Decision: accept as a small, correctness-clean default with rollback. The
route increases dispatch count because every Q4_K prompt matmul now has a Q8_1
quantize dispatch, so it should not be mistaken for the Phase 2a bulk lift. It
does, however, consistently beat the F32-RHS cols8 route on the current
two-model slice and keeps the Q4 direct packed path exercised while the real
MMQ/fusion work proceeds.

The current post-change blockers remain structural:

- Llama p64/p512: Q4_K prompt matmul still dominates, now via the direct Q8_1
  route and 84 extra Q8_1 quantize dispatches per run.
- Phi p64/p512: F16 attention matmul plus the unfused
  `MUL_MAT -> SOFT_MAX -> MUL_MAT -> CONT` chain dominates.
- The Q4_K x4 MMQ64x8 probe is now correctness-clean after the latest
  hrx-system fixes, but it is slower than the direct route and should stay
  disabled until a different schedule is authored.

## Rejected F16 Attention Cols8 p64 Widening

Date: 2026-06-16.

The existing F16 batched-attention cols8 route was temporarily widened from
`cols == 512` to `cols >= 64` to re-test the exact p64 buckets after the Q4
default policy changed. The source change was reverted after the experiment.

Artifact: `cache/hrx2/phase2a/f16-attn-cols8-p64-candidate-20260616-014058/`.

| Model | Case | Current default | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | `prefill-p64n0` | 75.762 tok/s | 75.751 tok/s | -0.015% |
| `llama32-3b-q4` | `prefill-p64n0` | 83.656 tok/s | 83.341 tok/s | -0.38% |

Decision: reject. The route selected cleanly but did not improve the p64
bucket. Do not spend more time widening this standalone dot route; the
attention bulk lift needs either a real tiled/fused attention kernel or a graph
fusion over `MUL_MAT -> SOFT_MAX -> MUL_MAT -> CONT`.

## Accepted F16 KQV + CONT p512 Fusion

Date: 2026-06-16.

The current `--flash-attn 0` p512 traces showed the attention chain as one of
the remaining structural boulders:

```text
MUL_MAT KQ -> SOFT_MAX -> MUL_MAT KQV -> PERMUTE -> CONT -> MUL_MAT attn_out
```

This pass implemented the first conservative attention-chain fusion: keep the
existing F16/F32 cols8 KQV dot schedule, but write the post-`PERMUTE` contiguous
layout directly into the `CONT` destination. This removes one full F32
read/write and one dispatch per layer without changing the dot algorithm.

- Source: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_f16_f32_batched.loom`
- Route: `mul_mat_f16_f32_batched_attention_cols8_contiguous_wg256`
- Graph fusion: `MUL_MAT(F16,F32) -> PERMUTE -> CONT`
- Rollback: `GGML_HRX2_DISABLE_F16_KQV_CONT_FUSION=1`
- Shape domain: p512 attention KQV buckets, `rows=128`, `cols=512`,
  `dst_ne2=24`, `dst_ne3=1`.

Initial JIT failed because the route metadata passed unused
`dst_stride_{col,ne2,ne3}` config keys while HRX2 JIT uses
`REJECT_UNKNOWN`. Standalone `loom-compile` ignored those unused bindings and
therefore hid the issue. Removing the unused bindings made the route compile,
load, and select cleanly.

Evidence:

- Standalone compile before the config fix:
  `cache/hrx2/phase2a/f16-kqv-cont-standalone-20260616-020155/`
  - contiguous export: 743 instructions, peak live units 37, 0 spills,
    256 bytes local memory.
  - baseline cols8 export in the same source: 694 instructions, peak live
    units 30, 0 spills, 256 bytes local memory.
- Focused backend-op gate after the fix:
  `cache/hrx2/phase2a/f16-kqv-cont-opgate-20260616-021027/`
  - `MUL_MAT,CONT`: 1362 rows, 57 supported, status 0.
- Llama p512 A/B:
  `cache/hrx2/phase2a/f16-kqv-cont-ab-fixed-20260616-021052/`
- Phi p512 A/B:
  `cache/hrx2/phase2a/f16-kqv-cont-phi-ab-fixed-20260616-021132/`
- Small integration smoke:
  `cache/hrx2/phase2a/f16-kqv-cont-small-bench-smoke-20260616-021602/`
- Fresh reduced HRX2/Vulkan comparison:
  `cache/hrx2/phase2a/current-reduced-after-f16-kqv-cont-20260616-021629/`

| Model | Disabled | Enabled | Change | Dispatch Delta | Fusion Dispatches | Removed `cont_f32_n128_vec4` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` p512 | 96.916 tok/s | 97.958 tok/s | +1.08% | -28 | 28 | 28 |
| `phi4-mini-q4` p512 | 83.775 tok/s | 85.011 tok/s | +1.48% | -32 | 32 | 32 |

Fresh reduced slice after acceptance:

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Dispatches | CPU compute | Fusion dispatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | `prefill-p512n0` | 96.483 tok/s | 4769.300 tok/s | 0.0202 | 617 | 0 | 28 |
| `llama32-3b-q4` | `prefill-p64n0` | 84.277 tok/s | 1581.316 tok/s | 0.0533 | 645 | 0 | 0 |
| `phi4-mini-q4` | `prefill-p512n0` | 86.720 tok/s | 4270.811 tok/s | 0.0203 | 641 | 0 | 32 |
| `phi4-mini-q4` | `prefill-p64n0` | 76.256 tok/s | 1456.857 tok/s | 0.0523 | 673 | 0 | 0 |

Decision: accept as a small, correctness-clean p512 attention traffic lift with
rollback. This is not the Phase 2a bulk prefill answer. It proves the graph
fusion/control-plane path for KQV layout folding and removes measurable memory
traffic, but the remaining gap is still dominated by quantized prompt matmul
schedule quality and the unfused attention algorithm itself.

Next attention boulder: use HRX1/Vulkan/flash-attention priors to replace the
separate KQ, softmax, and KQV chain with a real tiled or streaming attention
kernel/fusion. This KQV+CONT route should remain as a conservative fallback or
as one component of a larger fusion if full attention fusion is not applicable.

## Q4_K Block-Staged x4 MMQ Diagnostic Rejected

Date: 2026-06-16.

After the accepted Q4_K Q8_1 direct route, the next prior-driven attempt was to
move the opt-in `q8_1_x4_mmq32x32` diagnostic closer to the HRX1/Vulkan MMQ
schedule by staging all eight Q8_1 sub-blocks for one Q4_K block before the
dot loop. This reduced the obvious WYSIWYG barrier/global-load problem:

- Original current report:
  `cache/hrx2/phase2a/q4k-current-loom-report-20260616-022639/report.json`
  - `global_memory=348`, `barrier=16`, local memory `1152`.
- Block-staged report:
  `cache/hrx2/phase2a/q4k-mmq32x32-blockstaged-compile-20260616-023247/report.json`
  - `global_memory=156`, `barrier=2`, local memory `9216`.

However, the model-derived backend-op gate selected the intended provider and
failed correctness with NaNs on three Q4_K rows:

- `cache/hrx2/phase2a/q4k-x4-mmq32x32-blockstaged-model-opgate-20260616-023618/`
- stage-loop-unrolled retry:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-blockstaged-stageunroll-opgate-20260616-023736/`
- saved rejected patch:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-blockstaged-rejected-20260616-023755/`

Decision: reject and keep the production source clean. The route metadata and
packed Q8_1 x4 quantizer selected correctly; the failure is in the staged MMQ
spelling/lowering. Do not continue local knob sweeps on this source without a
smaller reducer or a different low-level staging path.

## Focused Backend-Op Perf Harness Fix

Date: 2026-06-16.

`test-backend-ops perf --output csv` already computed `time_us`, `n_runs`,
FLOP, and bandwidth fields, but CSV output omitted them. It also always ran
each perf case for at least one second and duplicated large model-derived ops
many times inside the graph, which made focused HRX2 route A/Bs too slow.

The llama.cpp test tool now:

- includes `time_us`, `flops`, `bandwidth_gb_s`, `memory_kb`, and `n_runs` in
  CSV output;
- accepts `GGML_TEST_BACKEND_OPS_PERF_MIN_US` to lower or raise the timing
  window;
- accepts `GGML_TEST_BACKEND_OPS_PERF_MAX_RUNS` to cap graph duplication.

Validation artifact:
`cache/hrx2/phase2a/test-backend-ops-perf-csv-capped-20260616-024525/`.

Example inner-loop command:

```bash
GGML_TEST_BACKEND_OPS_PERF_MIN_US=10000 \
GGML_TEST_BACKEND_OPS_PERF_MAX_RUNS=16 \
GGML_HRX2_TRACE_JSONL="$OUT/hrx2.jsonl" \
  build/llama-hrx2/bin/test-backend-ops perf -b HRX20 \
  --test-file cache/hrx2/phase2a/q4k-op-perf-testfile-20260615-183317/llama32-q4k-p512-ops.txt \
  --output csv > "$OUT/q4-perf.csv"
```

First capped A/B artifact:
`cache/hrx2/phase2a/quant-prompt-capped-perf-ab-20260616-024606/`.

| Family/row | Current | Opt-out fallback | Finding |
| --- | ---: | ---: | --- |
| Q4 `Qcur-0` | 5335.56 us | 5467.19 us | Q8_1 direct is slightly faster than F32 fallback. |
| Q4 `ffn_out-2` | 13828.69 us | 17484.31 us | Q8_1 direct is materially faster on the large row. |
| Q4 `ffn_gate-0` | 15790.19 us | 16400.81 us | Q8_1 direct is modestly faster. |
| Q5 prompt row | 12332.06 us | 37417.50 us | Packed x4 direct is a large win over F32 fallback. |
| Q6 prompt row | 11225.81 us | 47423.81 us | Packed x4 direct is a large win over F32 fallback. |
| Q6 large `result_output` | 363312.94 us | 1508040.13 us | Packed x4 direct avoids a severe F32 fallback cliff. |

Conclusion: the currently accepted packed/direct prompt routes are real wins,
but they are still one-row/four-column direct schedules. The remaining prefill
bulk gap is the missing tiled/MMQ prompt schedule and attention fusion, not a
route-disabled accident.

## Q4_K x4 MMQ Narrowed To Metadata-LDS Hazard

Date: 2026-06-16.

The Q4_K `q8_1_x4_mmq32x32` route was re-tested after the block-staged NaN
failure by staging only the packed Q8 payload in workgroup memory and reading
Q8_1 `d/s` metadata directly from the packed x4 global layout. This isolates
integer payload staging from f16 metadata staging.

Focused model-derived backend-op gates:

- First diagnostic:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-global-ds-opgate-20260616-025517/`
  - 4 rows, 0 bad, route selected, 0 provider-unavailable events.
- Cleaned candidate with unused f16 LDS staging removed:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-payloadlds-globalds-clean-20260616-025707/`
  - 4 rows, 0 bad, route selected, 0 provider-unavailable events.

Capped backend-op perf on the Llama 3.2 p512 Q4 rows:

| Row | Current Q8_1 cols4 | x4 MMQ payload-LDS/global-ds | Change |
| --- | ---: | ---: | ---: |
| `Qcur-0` | 5336.5 us | 1650.1 us | 3.23x faster |
| `ffn_out-2` | 13815.4 us | 4618.0 us | 2.99x faster |
| `ffn_gate-0` | 15763.2 us | 4588.9 us | 3.43x faster |

Bucket-matched full-op perf using freshly exported p64/p512 op files:
`cache/hrx2/phase2a/q4k-p64-p512-fullop-perf-ab-20260616-030132/`.
The prompt Q4 rows also improved in the p64 bucket:

| Row | p64 current | p64 x4 MMQ | p512 current | p512 x4 MMQ |
| --- | ---: | ---: | ---: | ---: |
| `Qcur-0` | 617.1 us | 234.1 us | 5160.1 us | 1571.4 us |
| `ffn_out-2` | 1615.9 us | 597.1 us | 13292.6 us | 4489.3 us |
| `ffn_gate-0` | 1853.4 us | 631.6 us | 15286.5 us | 4386.8 us |

Model-level A/B is mixed and therefore not promotable as a default route:

- One-repetition two-model reduced run:
  `cache/hrx2/phase2a/q4k-x4-mmq32x32-globalds-prefill-ab-20260616-025741/`
  regressed p64 and was approximately flat/slightly down on p512.
- Same-binary default comparison:
  `cache/hrx2/phase2a/q4k-current-default-prefill-ab-20260616-025845/`.
- Three-repetition Llama-only run:
  `cache/hrx2/phase2a/q4k-x4-mmq-repeat3-20260616-030241-default/` and
  `cache/hrx2/phase2a/q4k-x4-mmq-repeat3-20260616-030241-x4/`.

Three-repetition Llama result:

| Case | Default | x4 MMQ | Change |
| --- | ---: | ---: | ---: |
| p64 | 92.641 tok/s | 79.602 tok/s | 0.86x |
| p512 | 96.444 tok/s | 97.941 tok/s | 1.02x |

Decision: keep the route opt-in. This is a useful source repair and diagnostic
because it proves packed integer LDS payload staging can be correctness-clean
and much faster at the backend-op level. It is not the Phase 2a bulk lift yet.
The next boulder is the packed Q8_1 x4 quantization/backplane and amortization
strategy: the model trace moves the top route family to
`quantize_q8_1_x4_f32_generic_wg128` on Phi and still shows p64 sensitivity on
Llama. Do not default-enable x4 MMQ without either a faster/fused/reused x4
quantizer or a shape-specific route table proving net model-level wins.

### Q8_1 x4 Quantizer/Backplane Follow-Up

Evidence bundle:
`cache/hrx2/phase2a/q4k-x4-evidence-reports-20260616-030624/`.

The JIT compile reports for `quantize_q8_1_x4_f32_generic_wg128` are clean:

- 168 instructions, 788 code bytes.
- 0 spills, 0 private memory, 0 local memory.
- Peak live units 18.
- Static mix: 4 global memory ops, 2 local ops, 4 conversions, 69 vector ALU.

The emitted ISA shows the payload writer is still byte-oriented:

- one `global_load_b32` input load per lane;
- `ds_bpermute_b32` subgroup reductions for max/sum;
- one `global_store_b8` payload store per lane;
- two `global_store_b16` metadata stores from lane 0 of each subgroup.

That is not an obvious compiler/resource-pressure failure. It means the x4
quantizer is primarily a layout producer, not a 4x-cheaper producer. The model
trace count also shows the current cache is only a one-entry last-use cache:

| Run | Quantize dispatches | Cache hits | Notes |
| --- | ---: | ---: | --- |
| Default Llama p512 | 111 | 28 | Mixed non-x4/x4 layouts. |
| x4 Llama p512 | 83 | 56 | Better reuse, still many dispatches. |
| Default Phi p512 | 95 | 0 | No adjacent reuse. |
| x4 Phi p512 | 95 | 0 | x4-only but no cache benefit. |

Conclusion: the next structural backplane target is not a local Loom compiler
fix for the quantizer. It is either multi-entry/per-graph quantized RHS reuse,
producer fusion that avoids materializing Q8_1 for every consumer, or a route
planner that chooses one packed layout for an activation cluster instead of
letting individual matmul routes request incompatible layouts. Implementing a
multi-entry cache is non-trivial because the current cache owns one reusable
scratch buffer; multiple live cached RHS values require explicit arena offsets
or retained buffers with graph-lifetime management.

## Q4_K x4 MMQ Fresh Re-Gate Invalidates Prior Pass Claim

Date: 2026-06-16.

The committed `q8_1_x4_mmq32x32` source was re-gated after the prior
payload-LDS/global-metadata artifacts appeared contradictory. Current source
state is authoritative: with `GGML_HRX2_ENABLE_Q4_K_Q8_1_X4_MMQ=1`, the route
selects and the focused model-derived Q4_K rows fail with NaNs.

- Fresh current-source gate:
  `cache/hrx2/phase2a/q4k-x4-current-regate-20260616-032128/`.
- Route selected: `q8_1_x4_mmq32x32` appears in trace; do not infer fallback
  from the CSV.
- The older artifacts
  `q4k-x4-mmq32x32-global-ds-opgate-20260616-025517/` and
  `q4k-x4-mmq32x32-payloadlds-globalds-clean-20260616-025707/` should no
  longer be used as acceptance evidence for the committed source.

Narrow diagnostics were run and saved in
`cache/hrx2/phase2a/q4k-x4-current-diagnostics-20260616-032727/` plus the
individual gate artifacts:

| Variant | Artifact | Result |
| --- | --- | --- |
| Scalar i32 workgroup payload load/store spelling | `q4k-x4-scalar-lds-current-20260616-032241/` | NaNs remain |
| Direct global Q8 payload, staging side effects still present | `q4k-x4-direct-payload-current-20260616-032408/` | NaNs remain |
| Direct global Q8 payload, no LDS stores/barriers | `q4k-x4-direct-payload-nolds-current-20260616-032515/` | NaNs remain |
| Direct global payload, Q8 sums forced to 0 | `q4k-x4-zero-bs-nolds-current-20260616-032553/` | NaNs remain |
| Direct global payload, Q8 scales forced to 1 and sums to 0 | `q4k-x4-one-bd-zero-bs-nolds-current-20260616-032633/` | NaNs disappear; finite `ERR ~= 0.996` |
| Load packed Q8 `d/s` as one i32 word and bitcast to two f16 lanes | `q4k-x4-packed-ds-i32-opgate-20260616-033311/` | NaNs remain |
| Add explicit bounded `index.assume` facts to all Q8 metadata f16 indices | `q4k-x4-ds-index-assume-opgate-20260616-033629/` | NaNs remain |
| Re-apply direct x4 cols4 consumer against current quantizer/runtime | `q4k-x4-direct-cols4-current-quantizer-opgate-20260616-033411/` | Passes; direct route selected |

Current conclusion: the NaN-producing part of the committed MMQ32x32 route is
the Q8 scale (`d`) load/use path in this larger unrolled/MMQ spelling. It is
not isolated to Q8 sum correction, not isolated to vector-vs-scalar LDS access,
not isolated to LDS staging, not fixed by packed i32 metadata loading, and not
fixed by explicit f16 metadata index bounds. The current direct x4 consumer uses
the same quantizer/runtime and passes, so the basic x4 quantizer/buffer layout
is not the culprit. Treat this as a route-spelling or lowering issue specific
to the MMQ32x32 source shape until a standalone reducer proves otherwise.

Decision: keep Q4_K x4 MMQ disabled by default and do not use its backend-op or
model-level performance numbers for Phase 2a decisions. The prefill boulder
remains a true tiled packed prompt matmul, but the current MMQ32x32 spelling is
not a valid candidate. The next productive paths are either:

- build a minimal Loom reducer around the MMQ32x32 Q8 scale-load pattern for
  the author; or
- start a cleaner HRX1/Vulkan-shaped Q4_K/Q8_1 prompt tile that uses the
  known-good direct x4 metadata formula and validates correctness before
  adding LDS reuse.

## Q6_K x Q8_1 x4 MMQ32x32 Route Accepted

Date: 2026-06-16.

Implemented a Loom port of the HRX1 Q6_K packed-prompt prior:
`mul_mat_q6_k_q8_1_x4_mmq32x32_k256_32768_r1_262144_c32_512_wg128`.
The route consumes the existing packed Q8_1 x4 RHS layout, uses a 32 row x 32
column tile, stages the 32x8 packed RHS tile in workgroup memory, and assigns
each lane one output row and eight output columns.

Focused gates:

- Catalog validation: `cache/hrx2/phase2a/catalog-q6-mmq32x32-20260616-040411/`.
- Initial MMQ gate exposed a port bug in the Q6 block index:
  `cache/hrx2/phase2a/q6-mmq32x32-opgate-20260616-040534/`.
- Fixed gate passed with the MMQ route selected and no provider-unavailable
  events:
  `cache/hrx2/phase2a/q6-mmq32x32-fix-opgate-20260616-040708/`.
- Capped backend-op perf:
  `cache/hrx2/phase2a/q6-mmq32x32-perf-20260616-040759/`.

Backend-op prompt rows improved materially versus the previous x4 direct route:

| Row | Previous x4 direct | Q6 MMQ32x32 | Speedup |
| --- | ---: | ---: | ---: |
| Phi `ffn_out`, k=8192 rows=3072 cols=512 | 11225.812 us | 3721.812 us | 3.02x |
| Phi `result_output`, k=3072 rows=200064 cols=512 | 363312.938 us | 92144.375 us | 3.94x |

The one-column rows are unchanged and continue to use the existing rows2 WG32
decode/skinnier route. A one-column generic-Q8 WG32 diagnostic was also tested
and was correctness-clean but slower than the existing x4 direct route for
prompt rows; it remains below the x4 routes and must not be treated as the
default prefill solution.

Reduced HRX2/Vulkan prefill rerun:
`cache/hrx2/phase2a/q6-mmq32x32-reduced-20260616-040909/`.

| Model | Case | HRX2 tok/s | Vulkan tok/s | HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 76.084 | 1434.772 | 0.0530 |
| `phi4-mini-q4` | p512/n0 | 88.388 | 4266.714 | 0.0207 |
| `llama32-3b-q4` | p64/n0 | 80.948 | 1599.930 | 0.0506 |
| `llama32-3b-q4` | p512/n0 | 97.897 | 4805.973 | 0.0204 |

Decision: accept the Q6_K MMQ32x32 route. It proves the Loom spelling can carry
the HRX1 packed-Q8_1 RHS tiling pattern for Q6_K and gives a real 3-4x
backend-op improvement. It does not move the full model much because Q6 prompt
matmuls are no longer the top visible boulder in this reduced basket. The next
bulk prefill work should prioritize Q4_K prompt MMQ correctness/perf, Q5_K
MMQ32x32, packed-layout reuse/quantizer amortization, and attention-chain
fusion rather than further local Q6 tuning.

## Q4_K x Q8_1 x4 MMQ32x32 Route Accepted

Date: 2026-06-16.

The previous Q4_K x4 MMQ attempts were invalid acceptance evidence: some older
artifacts passed only through fallback, and the committed source later failed
focused Q4 rows with NaNs. The accepted implementation is a fresh rewrite of
`hrx2_mul_mat_q4_k_q8_1_x4_mmq32x32_static` using the proven Q6 MMQ topology:
one loop over Q8_1 sub-blocks, 32 rows x 32 prompt columns per workgroup, 128
threads, staged packed Q8_1 payload plus staged `d/s` metadata, and explicit
`vector.dot4i<u8s8>` dot form for unsigned Q4 codes times signed Q8 values.

The key WYSIWYG change is removing the old outer Q4-block plus eight-way group
clone. The old source emitted 6891 static instructions, 512 dot ops, 16
barriers, 444 global-memory ops, 528 local-memory ops, and a 46 KiB HSACO. The
accepted source emits 964 static instructions, 64 dot ops, 2 barriers, 50
global-memory ops, 83 local-memory ops, 1152 bytes LDS, and zero spills:

```text
cache/hrx2/phase2a/q4-x4-mmq-kb-loop-report-20260616-045914/report.json
```

Two Loom lowering limitations shaped the final spelling:

- `source-to-low` rejected integer CFG branch arguments in the looped route
  with `AMDGPU branch argument materializer selected for an unsupported type`.
  The accepted source yields already-scaled `f32` scale/min values from the
  Q4 scale decode branch instead of yielding `i32`.
- `scf.select` on `i32` did not have a target-low contract in this route. The
  accepted source uses shift/mask nibble selection instead.

Focused gates and performance:

- Current-source focused test gate:
  `cache/hrx2/phase2a/q4-x4-mmq-kb-loop-regate-20260616-045825/`.
- Capped backend-op perf:
  `cache/hrx2/phase2a/q4-x4-mmq-kb-loop-perf-20260616-045852/`.
- Default-on and opt-out route gates:
  `cache/hrx2/phase2a/q4-x4-mmq-default-on-gates-20260616-050208/`.

Backend-op p512 Llama rows improved versus the previous Q8_1 cols4 fallback:

| Row | Q8_1 cols4 fallback | Q4 x4 MMQ32x32 | Speedup |
| --- | ---: | ---: | ---: |
| `Qcur-0` | 5300 us | 1258 us | 4.21x |
| `ffn_out-2` | 13984 us | 3819 us | 3.66x |
| `ffn_gate-0` | 15911 us | 3549 us | 4.48x |

Model-level Llama A/B before defaulting:

| Case | Default | Q4 x4 MMQ | Change |
| --- | ---: | ---: | ---: |
| p64/n0 | 94.526 tok/s | 97.132 tok/s | +2.76% |
| p512/n0 | 100.385 tok/s | 105.784 tok/s | +5.38% |

The route is now default-on for prompt shapes with rollback:

```text
GGML_HRX2_DISABLE_Q4_K_Q8_1_X4_MMQ=1
```

Reduced HRX2/Vulkan prefill rerun after acceptance:
`cache/hrx2/phase2a/current-reduced-after-q4-x4-mmq-20260616-050233/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Top HRX2 route |
| --- | --- | ---: | ---: | ---: | --- |
| `phi4-mini-q4` | p64/n0 | 77.191 | 1452.964 | 0.0531 | `quantize_q8_1_x4_f32_generic_wg128 x95` |
| `phi4-mini-q4` | p512/n0 | 91.916 | 4251.767 | 0.0216 | `quantize_q8_1_x4_f32_generic_wg128 x95` |
| `llama32-3b-q4` | p64/n0 | 82.030 | 1593.727 | 0.0515 | `mul_mat_q4_k_q8_1_x4_mmq32x32... x112` |
| `llama32-3b-q4` | p512/n0 | 104.863 | 4726.343 | 0.0222 | `mul_mat_q4_k_q8_1_x4_mmq32x32... x112` |

Decision: accept and default-enable. This is a real Q4 prompt matmul schedule
repair and removes one known blocker, but it is not the bulk Phase 2a answer.
The remaining prefill gap is still structural. The next high-leverage targets
are Q5_K MMQ32x32, packed-Q8_1 layout reuse or quantizer amortization across
activation clusters, and a true attention-chain fusion replacing the separate
KQ, softmax, KQV, and layout-copy sequence.

## Q5_K x Q8_1 x4 MMQ32x32 Route Accepted

Date: 2026-06-16.

Implemented and default-enabled
`mul_mat_q5_k_q8_1_x4_mmq32x32_k256_32768_r1_262144_c32_512_wg128`.
The accepted route uses the same 32 row x 32 prompt-column MMQ topology as the
accepted Q4/Q6 routes, with staged packed Q8_1 payload, staged Q8_1 `d/s`
metadata, 128 threads, and eight output columns per lane group.

The decisive correctness fix was to decode Q5_K low nibbles and high bits as
packed aligned i32 words, matching the Q6-style schedule:

- QL: load four consecutive Q5 low-nibble bytes as one aligned i32, shift the
  whole word by `0` or `4`, mask with `0x0f0f0f0f`, then unpack the four byte
  lanes.
- QH: load four consecutive high-bit bytes as one aligned i32, shift the whole
  word by the Q5 group index, mask with `0x01010101`, shift into bit 4, then
  unpack the four byte lanes.
- Q8_1 `d/s`: store metadata in LDS as f32. The same source shape with f16
  metadata in LDS compiled but failed strict CPU-reference correctness with a
  finite error; f32 LDS passed.

Rejected/diagnostic spellings:

- Per-byte dynamic low-nibble shifts compiled but failed strict correctness
  with finite `ERR ~= 1.1`.
- `scf.if` yielding i32 low nibbles hit
  `AMDGPU branch argument materializer selected for an unsupported type` and
  fell back to the existing direct route; passing CSV output from that run was
  not acceptance evidence.
- `scf.select`/branchless arithmetic low-nibble spelling compiled but produced
  NaNs in this larger MMQ loop shape.
- Direct global Q8 metadata passed after the packed QL/QH fix, but the final
  route keeps metadata staged as f32 to preserve the intended MMQ backplane.

Focused validation and performance:

- Final focused test/perf gate:
  `cache/hrx2/phase2a/q5-mmq32x32-final-gates-20260616-053952/`.
- Evidence dump with compile report and manifest:
  `cache/hrx2/phase2a/q5-mmq32x32-evidence-20260616-054240/`.
- Compile report for the MMQ provider: 1164 instructions, 64 dot ops, 2
  barriers, 34 global-memory ops, 83 local-memory ops, 1280 bytes LDS, peak
  live 85, zero spills, zero private memory, 13.0 KiB HSACO.
- Capped backend-op p512 row `wqkv-0`, k=3072 rows=5120 cols=512:
  2141.75 us. The previous direct x4 route class for this row was roughly
  12.3 ms, so this is a ~5.8x focused route-level improvement.

Reduced HRX2/Vulkan prefill rerun after acceptance:
`cache/hrx2/phase2a/current-reduced-after-q5-mmq32x32-20260616-054510/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `phi4-mini-q4` | p64/n0 | 80.088 | 1445.866 | 0.0554 | Q5 MMQ selected x32 |
| `phi4-mini-q4` | p512/n0 | 97.446 | 4252.687 | 0.0229 | Q5 MMQ selected x32 |
| `llama32-3b-q4` | p64/n0 | 82.852 | 1566.054 | 0.0529 | Q5 not a top route |
| `llama32-3b-q4` | p512/n0 | 105.773 | 4786.709 | 0.0221 | Q4 remains dominant |

Decision: accept. This removes the Q5 prompt-matmul boulder and gives a large
focused win, but model-level throughput remains far from the Vulkan target.
The current top structural issues are repeated x4 Q8_1 quantization/layout
materialization, attention-chain traffic, and missing pragmatic fusions rather
than local Q5 schedule quality.

## Q4_K SWIGLU Direct Fusion Restricted To Single-Column Decode

Date: 2026-06-16.

After Q4/Q5/Q6 packed prompt MMQ routes were accepted, the earlier Q4_K SWIGLU
fusion became a severe prompt regression. The fused route eliminated launches,
but it still used a direct Q4_K x F32 RHS algorithm with one output row and one
prompt column per workgroup. The unfused prompt path now uses the packed
Q4_K x Q8_1 x4 MMQ route for the two FFN projections plus a standalone SWIGLU
kernel, which is much faster despite the extra dispatches and intermediate.

Same-binary HRX2 A/B:

- Default fused prompt:
  `cache/hrx2/phase2a/q4k-swiglu-current-default-20260616-061015/`.
- Forced unfused prompt with `GGML_HRX2_DISABLE_Q4K_SWIGLU_FUSION=1`:
  `cache/hrx2/phase2a/q4k-swiglu-current-disabled-20260616-061049/`.

| Model | Case | Direct fused | Unfused packed-MMQ path | Speedup |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 79.959 | 219.648 | 2.75x |
| `phi4-mini-q4` | p512/n0 | 95.700 | 617.662 | 6.45x |
| `llama32-3b-q4` | p64/n0 | 81.083 | 199.001 | 2.45x |
| `llama32-3b-q4` | p512/n0 | 105.475 | 632.209 | 5.99x |

Catalog change: the two direct Q4_K SWIGLU fusion routes are now single-column
only:

- `mul_mat_q4_k_swiglu_f32_direct_k256_32768_c1_wg256`
- `mul_mat_q4_k_packed_swiglu_f32_direct_k256_32768_c1_wg256`

Focused component gate:
`cache/hrx2/phase2a/q4k-swiglu-c1-guard-opgate-20260616-061401/`.
It passed 26/26 model-derived rows. The gate covers the component routes; the
graph-fusion decision is proven by the provider traces in the reduced
benchmarks below.

Reduced HRX2/Vulkan prefill rerun after the catalog guard:
`cache/hrx2/phase2a/q4k-swiglu-c1-guard-reduced-20260616-061439/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Top HRX2 route |
| --- | --- | ---: | ---: | ---: | --- |
| `phi4-mini-q4` | p64/n0 | 219.032 | 1471.304 | 0.1489 | `quantize_q8_1_x4_f32_generic_wg128 x126` |
| `phi4-mini-q4` | p512/n0 | 624.923 | 4284.708 | 0.1458 | `quantize_q8_1_x4_f32_generic_wg128 x126` |
| `llama32-3b-q4` | p64/n0 | 199.965 | 1563.697 | 0.1279 | `mul_mat_q4_k_q8_1_x4_mmq32x32... x166` |
| `llama32-3b-q4` | p512/n0 | 642.889 | 4842.621 | 0.1328 | `mul_mat_q4_k_q8_1_x4_mmq32x32... x166` |

Decode smoke:
`cache/hrx2/phase2a/q4k-swiglu-c1-guard-decode-smoke-20260616-061521/`.
Decode remains in the existing band: Phi `45.455` tok/s vs Vulkan `116.996`
(`0.3885x`) and Llama 3.2 `47.290` tok/s vs Vulkan `139.627` (`0.3387x`).
The renamed single-column fused routes still select for decode:

- Phi:
  `mul_mat_q4_k_packed_swiglu_f32_direct_k256_32768_c1_wg256 x2080`.
- Llama 3.2:
  `mul_mat_q4_k_swiglu_f32_direct_k256_32768_c1_wg256 x1820`.

Decision: accept the catalog guard. This is a major prefill boulder fix and
also a process correction: a fusion is not accepted per operator family; it is
accepted per shape regime and only if it beats the best available unfused route
composition. For this family, prompt needs a future true packed-MMQ SWIGLU
fusion before it should replace the separate MMQ + standalone SWIGLU path.

## F16 Attention KQ Row-Tiled Probe Accepted

Date: 2026-06-16.

The p512 hot-route evidence after the Q4_K SWIGLU guard showed the unfused
attention KQ matmul as a disproportionate remaining boulder. HRX2 was still
using `mul_mat_f16_f32_batched_attention_cols8_wg256`, a dot-per-output-family
schedule with one output row and eight prompt columns per workgroup. For the
KQ prompt shape (`k=128`, `rows=512`, `cols=512`, batched heads), this creates
hundreds of thousands of tiny reductions. The Vulkan prior is a tiled matmul
shader (`mul_mm.comp`) that computes many rows and columns per workgroup with
explicit staging, so the HRX2 route was in the wrong schedule family.

Implemented an intermediate Loom route:

- `mul_mat_f16_f32_batched_attention_rows2_cols8_wg128`
- export `hrx2_mul_mat_f16_f32_batched_rows2_cols8`
- domain: KQ prompt bucket only, `k=128`, `rows=512`, `cols=512`
- schedule: two output rows and eight output columns per workgroup; RHS loads
  are reused across both rows; WG128 matches the static `k=128` shape

Rejected probe:

- `rows2_cols8_wg256` passed correctness but was mixed at op level: it slightly
  improved the 24-head KQ row and regressed the 32-head row. The mismatch was
  expected in hindsight because KQ has `k=128`, so WG256 leaves half the
  workgroup idle while doubling the number of reductions per workgroup.

Focused validation:

- Correctness and route-selection gate:
  `cache/hrx2/phase2a/f16-rows2-cols8-wg128-opgate-20260616-063143/`.
- Perf gate:
  `cache/hrx2/phase2a/f16-rows2-cols8-wg128-perf-20260616-063202/`.
- The new route selected for the two p512 KQ rows, with zero
  `provider_unavailable` events.

Focused KQ timings:

| Row | Previous hot-list/cols8 class | Rows2 WG128 |
| --- | ---: | ---: |
| 24-head KQ, `k=128 rows=512 cols=512` | ~4.8-4.9 ms | 2.036 ms |
| 32-head KQ, `k=128 rows=512 cols=512` | ~4.8-4.9 ms | 2.746 ms |

Reduced HRX2/Vulkan prefill rerun:
`cache/hrx2/phase2a/f16-rows2-cols8-wg128-reduced-20260616-063239/`.

| Model | Case | Previous HRX2 | New HRX2 | Vulkan | New HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 219.032 | 219.305 | 1453.528 | 0.1509 |
| `phi4-mini-q4` | p512/n0 | 624.923 | 674.326 | 4266.147 | 0.1581 |
| `llama32-3b-q4` | p64/n0 | 199.965 | 200.310 | 1566.679 | 0.1279 |
| `llama32-3b-q4` | p512/n0 | 642.889 | 690.675 | 4832.416 | 0.1429 |

Decision: accept the WG128 row-tiled KQ route. It is a real p512 lift and it
confirms that attention KQ route quality is a live prefill boulder. It is not
the final attention solution: the route is still dot-per-output-family, not a
Vulkan-style 32x32/64x64 tiled matmul. The remaining attention work should move
toward a true tiled F16/F32 matmul or an attention-chain fusion rather than
continuing local row/column-count tweaks indefinitely.

## F16 Attention KQV Row-Tiled Probe Rejected

Date: 2026-06-16.

After the accepted KQ rows2 route, an analogous KQV fused-contiguous route was
tested:

- export `hrx2_mul_mat_f16_f32_batched_rows2_cols8_contiguous`
- route `mul_mat_f16_f32_batched_attention_rows2_cols8_contiguous_wg256`
- domain: KQV prompt bucket only, `k=512`, `rows=128`, `cols=512`
- schedule: two output rows and eight output columns per workgroup; RHS loads
  reused across both rows; contiguous post-permute attention layout written
  directly

Focused raw attention backend-op correctness still passed, but the exported
op file exercises the raw `MUL_MAT` rows rather than the fused `CONT` provider:

- `cache/hrx2/phase2a/f16-kqv-rows2-cols8-wg256-opgate-trace-20260616-064358/`
- 8/8 rows supported, zero errors
- selected routes were only `mul_mat_f16_f32_batched_attention_wg256`,
  `mul_mat_f16_f32_batched_attention_cols8_wg256`, and the accepted KQ
  `mul_mat_f16_f32_batched_attention_rows2_cols8_wg128`

Real-graph route proof used a Phi p512 HRX2 smoke:

- `cache/hrx2/phase2a/f16-kqv-rows2-cols8-wg256-phi-p512-smoke-20260616-064429/`
- selected `mul_mat_f16_f32_batched_attention_rows2_cols8_contiguous_wg256`
  32 times
- completed at `680.038` tok/s, only slightly above the prior one-run
  `674.326` tok/s baseline

Reduced HRX2/Vulkan comparison:
`cache/hrx2/phase2a/f16-kqv-rows2-cols8-wg256-reduced-20260616-064458/`.

| Model | Case | Previous HRX2 | Probe HRX2 | Probe Vulkan | Probe HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 219.305 | 220.546 | 1446.704 | 0.1524 |
| `phi4-mini-q4` | p512/n0 | 674.326 | 677.450 | 4311.215 | 0.1571 |
| `llama32-3b-q4` | p64/n0 | 200.310 | 199.262 | 1568.502 | 0.1270 |
| `llama32-3b-q4` | p512/n0 | 690.675 | 688.392 | 4866.678 | 0.1415 |

Decision: reject and remove the KQV rows2 contiguous route. It selected in the
real graph and did not create CPU fallback, but model throughput stayed within
noise and did not improve the Vulkan ratio. This reinforces the KQ conclusion:
local row-count tweaks are not the attention solution. The profitable next
attention step needs to match the Vulkan schedule family more directly:
proper tiled F16/F32 matmul or an attention-chain fusion that removes the
KQ/SOFT_MAX/KQV/layout traffic, not another dot-per-output variant.

## Rejected Runtime Probe: Graph-Local Q8_1 x4 Cache

Date: 2026-06-16.

The current reduced traces show `quantize_q8_1_x4_f32_generic_wg128` as the
top route family for Phi prefill and a visible route family for Llama prefill.
A runtime probe extended the existing one-entry Q8_1 RHS cache into a
graph-local offset-backed cache so non-adjacent packed matmuls could reuse the
same quantized activation tensor.

Focused K-quant backend-op gate passed with the intended Q4/Q5/Q6 MMQ routes:

- `cache/hrx2/phase2a/q8-graph-cache-kquant-opgate-20260616-065650/`
- 9/9 hot model-derived rows supported, 0 errors.

Same-binary HRX2 prefill A/B rejected the change:

- Enabled:
  `cache/hrx2/phase2a/q8-graph-cache-ab-20260616-065734-enabled/`
- Disabled:
  `cache/hrx2/phase2a/q8-graph-cache-ab-20260616-065734-disabled/`

| Model | Case | Graph cache | Existing cache | Change |
| --- | --- | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 206.921 | 217.540 | 0.951x |
| `phi4-mini-q4` | p512/n0 | 670.067 | 680.894 | 0.984x |
| `llama32-3b-q4` | p64/n0 | 190.701 | 198.339 | 0.961x |
| `llama32-3b-q4` | p512/n0 | 681.785 | 695.433 | 0.980x |

Trace interpretation:

- Phi had 126 Q8_1 x4 quantize dispatches and zero cache hits in both modes.
  The graph-local cache inserted entries but did not find reusable tensor
  identity matches.
- Llama already had 83 cache hits with the accepted one-entry last-use cache.
  The graph-local cache did not reduce quantize dispatches beyond that.
- Dispatch counts and top route families were unchanged.

Decision: reject and revert. The remaining packed-RHS backplane issue is not a
simple non-adjacent tensor-pointer cache miss. Future work should focus on
planner-level layout reuse or producer/consumer fusion that avoids requesting
separate packed RHS materializations, rather than adding a larger runtime cache
for the current graph order.

## Rejected Q4_K MMQ Metadata-Staging Probe

Date: 2026-06-16.

The accepted Q4_K x Q8_1 x4 MMQ route stages Q8_1 `d/s` metadata as f16 in
workgroup memory, then converts the metadata to f32 during the dot loop. A
small WYSIWYG probe changed this to f32 LDS metadata, mirroring the Q5_K route
where f32 metadata staging was required for strict correctness.

Focused gate:

- `cache/hrx2/phase2a/q4-mmq-f32-meta-opgate-20260616-070112/`
- 9/9 hot model-derived K-quant rows supported, 0 errors.
- Intended Q4/Q5/Q6 MMQ routes selected; no fallback evidence.

Capped backend-op perf versus the previous accepted Q4 MMQ timing from
`cache/hrx2/phase2a/kquant-mmq-evidence-20260616-064737/`:

| Q4 row | Accepted route | f32 metadata probe | Change |
| --- | ---: | ---: | ---: |
| `Qcur-0`, k=3072 rows=3072 cols=512 | 1284.280 us | 1264.750 us | 1.015x |
| `ffn_out-2`, k=8192 rows=3072 cols=512 | 3666.736 us | 3877.438 us | 0.946x |
| `ffn_gate-0`, k=3072 rows=8192 cols=512 | 3538.509 us | 3605.313 us | 0.981x |
| `ffn_up-0`, k=3072 rows=16384 cols=512 | 7232.774 us | 7132.125 us | 1.014x |

Decision: reject and revert. Q4 f16 metadata staging is not the current broad
bottleneck; changing it gives mixed row-level movement and hurts important FFN
rows. Continue with larger schedule changes such as K-slice batching, tile
shape, attention fusion, or packed-layout planning rather than local metadata
precision changes.

## Rejected F16 Attention KQ Rows4/Cols4 Probe

Date: 2026-06-16.

The accepted KQ route computes two output rows and eight output columns per
workgroup. A controlled probe changed that schedule to four output rows and
four output columns per workgroup, keeping 16 accumulators per workgroup while
trading fewer f32 RHS column loads for more f16 row loads.

Focused gate:

- `cache/hrx2/phase2a/f16-kq-rows4-cols4-probe-20260616-070717/`
- 8/8 focused F16 attention rows supported, 0 errors.
- The probe route selected for the two p512 KQ rows; no provider fallback was
  hiding the result.

Backend-op timings versus the accepted rows2/cols8 checkpoint:

| Row | Accepted rows2/cols8 | Probe rows4/cols4 | Change |
| --- | ---: | ---: | ---: |
| 24-head KQ, `k=128 rows=512 cols=512` | ~2036 us | 2096 us | 0.97x |
| 32-head KQ, `k=128 rows=512 cols=512` | ~2746 us | 2886 us | 0.95x |

Decision: reject and revert. This confirms that more local row/column tiling
around the same dot-per-output reduction family is unlikely to be the bulk
attention answer. The next attention step should be a true tiled matmul or
attention-chain fusion, using Vulkan/HRX1 as schedule priors.

## Rejected Q4_K MMQ A-Tile Staging Probe

Date: 2026-06-16.

Prior comparison against Vulkan showed a structural schedule gap in the Q4_K
packed prompt MMQ route: Vulkan stages both A and B tiles, while the current
Loom route stages the packed Q8_1 RHS tile but re-decodes each Q4_K A row once
per column-lane group. A probe staged each decoded Q4 vector and its scale/min
metadata in workgroup memory once per row and `kb` group, then had all four
column-lane groups consume the staged A payload.

Artifacts:

- Scalar LDS payload attempt:
  `cache/hrx2/phase2a/q4-a-staged-opgate-20260616-071646/`.
- Vector LDS payload attempt:
  `cache/hrx2/phase2a/q4-a-staged-vector-opgate-20260616-071840/`.
- Rejected patch:
  `cache/hrx2/phase2a/q4-a-staged-rejected-20260616-071952/q4-a-staged-rejected.patch`.

Both attempts compiled and selected
`mul_mat_q4_k_q8_1_x4_mmq32x32_k256_32768_r1_32768_c32_512_wg128` with no
provider fallback, but Q4_K rows failed strict CPU-reference checks with small
finite errors. Q5_K and Q6_K rows in the same focused gate remained correct.

Decision: reject and revert. A-side staging remains the right prior-derived
schedule direction for Q4_K, but the current high-level Loom spelling is not
correctness-clean. Future work should either produce a standalone reducer for
Q4_K A-side integer/f32 workgroup staging or use a lower-level spelling that
can preserve exact payload/metadata movement.

## Q6_K MMQ64x32 Route Accepted

Date: 2026-06-16.

The focused K-quant evidence showed the Q6_K output projection as the largest
single backend-op row, with `result_output` taking about `62-99 ms` for the
large vocab shapes. The accepted Q6_K route used a 32-row by 32-column
workgroup tile. A controlled probe doubled the row tile to 64 and the workgroup
to 256 threads, reusing each staged packed Q8_1 RHS tile across twice as many
Q6_K rows.

Accepted route:

- `mul_mat_q6_k_q8_1_x4_mmq64x32_k256_32768_r1_262144_c32_512_wg256`
- export `hrx2_mul_mat_q6_k_q8_1_x4_mmq64x32_static`
- schedule: 64 output rows by 32 columns per workgroup; four column-lane
  groups, eight output columns per lane group; Q8_1 RHS tile staged in LDS.

Focused gate:

- `cache/hrx2/phase2a/q6-mmq64x32-wg256-opgate-20260616-072056/`
- 9/9 hot K-quant rows supported, 0 errors.
- Renamed route proof:
  `cache/hrx2/phase2a/q6-mmq64x32-renamed-opgate-20260616-072323/`.

Focused backend-op perf versus the prior Q6_K MMQ32x32 checkpoint:

| Row | Prior | MMQ64x32 | Change |
| --- | ---: | ---: | ---: |
| `result_output`, rows=200064 | 98820.605 us | 95093.750 us | 1.039x |
| `result_output`, rows=128256 | 62223.339 us | 60164.375 us | 1.034x |
| `ffn_out`, k=8192 rows=3072 | 3746-3801 us | 3791-3913 us | mixed |

Reduced HRX2/Vulkan comparison after the final renamed route and corrected
workitem range assumption:
`cache/hrx2/phase2a/q6-mmq64x32-final-reduced-20260616-072605/`.

| Model | Case | Prior HRX2 | New HRX2 | New Vulkan | New HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 219.305 | 219.586 | 1461.002 | 0.1503 |
| `phi4-mini-q4` | p512/n0 | 674.326 | 683.609 | 4248.812 | 0.1609 |
| `llama32-3b-q4` | p64/n0 | 200.310 | 201.368 | 1564.646 | 0.1287 |
| `llama32-3b-q4` | p512/n0 | 690.675 | 692.810 | 4720.838 | 0.1468 |

Decision: accept as a small p512/output-projection lift. This does not change
the Phase 2a diagnosis: Q4_K packed prompt MMQ and attention/fusion remain the
major prefill boulders. The Q6 result also confirms that coarse tile-shape
changes should be gated by focused op rows plus reduced model runs, since the
largest single row improved but small-row Q6 movement was mixed.

## Q4_K MMQ64x32 Route Accepted

Date: 2026-06-16.

After Q6_K showed a small benefit from reusing each staged packed Q8_1 RHS tile
across 64 rows instead of 32, the same coarse row-tile probe was applied to the
default Q4_K x packed Q8_1 x4 prompt MMQ route. This is not the final
Vulkan/HRX1-style A/B-staged Q4_K schedule: it keeps the existing Q4_K decode
path and only widens the output-row tile plus workgroup size.

Accepted route:

- `mul_mat_q4_k_q8_1_x4_mmq64x32_k256_32768_r1_32768_c32_512_wg256`
- export `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static`
- schedule: 64 output rows by 32 columns per workgroup; WG256; four
  column-lane groups, eight output columns per lane group; Q8_1 RHS tile
  staged in LDS.

Focused gate:

- Probe gate:
  `cache/hrx2/phase2a/q4-mmq64x32-wg256-opgate-20260616-073018/`.
- Final renamed route proof:
  `cache/hrx2/phase2a/q4-mmq64x32-final-opgate-20260616-073350/`.
- 9/9 hot K-quant rows supported, 0 errors.
- Final route trace selected
  `mul_mat_q4_k_q8_1_x4_mmq64x32_k256_32768_r1_32768_c32_512_wg256`
  68 times in the focused perf run, with no unavailable providers.

Focused backend-op perf versus the prior Q4 MMQ32x32 checkpoint was mixed but
plausibly favorable on the largest Q4 rows:

| Row | Prior | MMQ64x32 | Change |
| --- | ---: | ---: | ---: |
| `Qcur`, k=3072 rows=3072 | 1284.280 us | 1212.000 us | 1.060x |
| `ffn_up`, k=3072 rows=16384 | 7232.774 us | 7107.438 us | 1.018x |
| `ffn_gate`, k=3072 rows=8192 | 3538.509 us | 3582.875 us | 0.988x |
| `ffn_out`, k=8192 rows=3072 | 3666.736 us | 3760.188 us | 0.975x |

Reduced HRX2/Vulkan comparison after the final renamed route:
`cache/hrx2/phase2a/q4-mmq64x32-final-reduced-20260616-073443/`.

| Model | Case | Prior HRX2 | New HRX2 | New Vulkan | New HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 219.586 | 223.061 | 1477.116 | 0.1510 |
| `phi4-mini-q4` | p512/n0 | 683.609 | 690.144 | 4217.251 | 0.1636 |
| `llama32-3b-q4` | p64/n0 | 201.368 | 207.970 | 1583.374 | 0.1313 |
| `llama32-3b-q4` | p512/n0 | 692.810 | 707.996 | 4771.949 | 0.1484 |

Decision: accept as a small prompt-matmul lift. The remaining prefill gap is
still structural: the final summary has zero CPU compute fallback but HRX2 is
only about 0.13x-0.16x Vulkan on the reduced p64/p512 basket. Llama remains
dominated by this Q4_K MMQ route, while Phi still shows the Q8_1 quantizer,
Q4_K/Q5_K/Q6_K prompt matmuls, and attention-chain routes near the top.

## Rejected Q5_K MMQ64x32 Row-Tile Probe

Date: 2026-06-16.

Because Q6_K and Q4_K both saw small wins from widening the packed Q8_1 x4 MMQ
row tile from 32 rows/WG128 to 64 rows/WG256, the same narrow probe was tested
against the accepted Q5_K MMQ32x32 route.

Artifacts:

- Focused gate:
  `cache/hrx2/phase2a/q5-mmq64x32-wg256-opgate-20260616-073750/`.
- Rejected patch:
  `cache/hrx2/phase2a/q5-mmq64x32-rejected-20260616-073834/q5-mmq64x32-rejected.patch`.

Result:

- 9/9 hot K-quant rows supported, 0 errors.
- The Q5_K hot row `wqkv`, `k=3072 rows=5120 cols=512`, regressed from about
  `2172.375 us` on the accepted Q5 MMQ32x32 route to `2246.875 us` on the
  WG256/MMQ64x32 probe.

Decision: reject without a model-level rerun. The 64-row tile is a useful
candidate knob, but it is not generically profitable across K-quant formats.
Q5_K should stay on MMQ32x32/WG128 until a prior-driven schedule change
addresses its own bottleneck.

## Q4_K MMQ64x32 A-Side Staging Accepted

Date: 2026-06-16.

The next prior-driven Q4_K prompt-matmul pass repaired the previously rejected
A-side workgroup-staging path. The old rejection was based on a stale
MMQ32x32/WG128 route and a 32-row LDS payload. Reapplying the idea to the
accepted MMQ64x32/WG256 route required resizing the A-side scratch tile to
64 rows:

- decoded Q4 payload: `64 rows * 8 i32 = 512xi32`, 2048 bytes;
- Q4 scale/min metadata: `64xf32` each.

The route now stages both sides used by the 64-row tile: the packed Q8_1 x4 RHS
tile and the decoded Q4_K payload plus scale/min metadata for each active row.
An explicit workgroup barrier before consuming the staged A payload and another
barrier before the next `kb` iteration preserve LDS lifetime.

Accepted route:

- `mul_mat_q4_k_q8_1_x4_mmq64x32_k256_32768_r1_32768_c32_512_wg256`
- export `hrx2_mul_mat_q4_k_q8_1_x4_mmq64x32_static`
- schedule: 64 output rows by 32 columns per workgroup; WG256; staged Q8_1
  RHS tile; staged Q4_K A payload, scale, and min for reuse across the four
  column-lane groups.

Focused gates:

- Correctness-only probe:
  `cache/hrx2/phase2a/q4-a-staged64-optest-20260616-074238/`.
- Perf probe:
  `cache/hrx2/phase2a/q4-a-staged64-perf-20260616-074312/`.
- Final gate after cleanup:
  `cache/hrx2/phase2a/q4-a-staged64-final-opgate-20260616-074455/`.
- 9/9 hot K-quant rows supported, 0 errors, no unavailable providers.

Focused backend-op perf versus the prior accepted Q4 MMQ64x32 route:

| Row | Prior MMQ64x32 | A-staged MMQ64x32 | Change |
| --- | ---: | ---: | ---: |
| `Qcur`, k=3072 rows=3072 | 1212.000 us | 1146.750 us | 1.057x |
| `ffn_gate`, k=3072 rows=8192 | 3582.875 us | 3300.812 us | 1.085x |
| `ffn_out`, k=8192 rows=3072 | 3760.188 us | 3417.688 us | 1.100x |
| `ffn_up`, k=3072 rows=16384 | 7107.438 us | 6432.188 us | 1.105x |

Reduced HRX2/Vulkan comparison after commit `dcb8958c5`:
`cache/hrx2/phase2a/q4-a-staged64-final-reduced-20260616-074651/`.

| Model | Case | Prior HRX2 | New HRX2 | New Vulkan | New HRX2/Vulkan |
| --- | --- | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | 223.061 | 245.608 | 1453.578 | 0.1690 |
| `phi4-mini-q4` | p512/n0 | 690.144 | 736.907 | 4268.960 | 0.1726 |
| `llama32-3b-q4` | p64/n0 | 207.970 | 232.920 | 1575.610 | 0.1478 |
| `llama32-3b-q4` | p512/n0 | 707.996 | 761.677 | 4856.921 | 0.1568 |

Decision: accept. This is the first Phase 2a Q4_K change that moves the
reduced prefill basket by high single digits to low double digits while passing
strict backend-op correctness. The remaining gap is still large; Vulkan p512
per-op logs put Q4_K prompt matmuls in the `265-626 us` range for the common
rows, while HRX2 is still `1147-6432 us` depending on shape. The next Q4_K work
should continue matching the known-good schedule family rather than returning
to generic tile-size sweeps.

## Rejected Q6_K A-Side Staging Probes

Date: 2026-06-16.

The next Q6_K output-projection probe tried to copy the successful Q4_K
A-side reuse pattern into the accepted Q6_K MMQ64x32 route. The prior looked
plausible because the current Q6 route stages the packed Q8_1 RHS tile but
recomputes Q6 unpack work once per column-lane group.

Two variants were tested:

- Payload plus f32 scale staging:
  `cache/hrx2/phase2a/q6-a-staged64-opgate-20260616-080255/`.
- Payload-only staging with original per-lane scale multiply:
  `cache/hrx2/phase2a/q6-a-payload-staged64-opgate-20260616-080424/`.

Results:

| Variant | Correctness | Large `result_output` rows | Decision |
| --- | --- | --- | --- |
| Payload + scale LDS | Failed one large Q6 row, `ERR=0.000646920 > 0.0005` | Not benchmarked | reject |
| Payload-only LDS | Passed 9/9 hot rows | Regressed to `63603.563 us` and `99853.313 us` | reject |
| Guarded baseline after revert | Passed 9/9 hot rows | `60312.563 us` and `95344.500 us` | keep baseline |

Saved rejected patches:

- `cache/hrx2/phase2a/q6-a-staged64-opgate-20260616-080255/saved/q6-a-payload-and-scale-staged-failing.patch`
- `cache/hrx2/phase2a/q6-a-payload-staged64-opgate-20260616-080424/saved/q6-a-payload-only-staged-rejected.patch`

Decision: reject both Q6_K A-staging variants. Unlike Q4_K, the current high
level A-payload LDS spelling does not improve the Q6_K MMQ64x32 schedule. The
scale-staged version also shows a small correctness drift on the largest row,
so f32 metadata staging should not be promoted without a standalone reducer or
a lower-level spelling.

The only retained code from this pass is generic `rows_multiple_of` route-guard
support plus a `rows_multiple_of: 64` guard on the Q6_K MMQ64x32 route. This
keeps the route away from partial final row tiles, which would otherwise put
workgroup barriers behind per-lane row bounds. The current basket rows are
multiples of 64, so this does not change selected prefill routes for the
measured p64/p512 cases.

## Rejected Q6_K MMQL64x64 Probe

Date: 2026-06-16.

The next Q6_K output-projection probe tried a larger MMQL-style tile:
64 rows by 64 columns, WG256, BK step 4, staged Q6_K A payload/scales and
packed Q8_1 x4 RHS in workgroup memory. It was intended to test whether a
more HRX1-like 4-row by 4-column per-thread schedule could improve the large
vocabulary Q6_K output projection.

Artifacts:

- Initial fully unrolled compute-loop gate:
  `cache/hrx2/phase2a/q6-mmql64x64-opgate-20260616-081713/`.
- No-k-step-unroll gate:
  `cache/hrx2/phase2a/q6-mmql64x64-no-kunroll-opgate-20260616-082059/`.
- Saved rejected patch:
  `cache/hrx2/phase2a/q6-mmql64x64-no-kunroll-opgate-20260616-082059/saved/q6-mmql64x64-rejected.patch`.

Both variants selected the intended route with no provider-unavailable events.
The initial fully unrolled spelling was correctness-clean but regressed every
Q6_K row:

| Row | Guarded MMQ64x32 baseline | MMQL64x64 unrolled |
| --- | ---: | ---: |
| `ffn_out-0` k8192 rows3072 cols512 | `3874.313 us` | `5139.625 us` |
| `ffn_out-0` k8192 rows3072 cols512 | `3806.438 us` | `5202.688 us` |
| `result_output` k3072 rows128256 cols512 | `60312.563 us` | `75693.063 us` |
| `result_output` k3072 rows200064 cols512 | `95344.500 us` | `118753.125 us` |

Removing the k-step unroll reduced HSACO size and recovered most of the large
row regression, but still did not beat the accepted route:

| Row | Guarded MMQ64x32 baseline | MMQL64x64 no-k-unroll |
| --- | ---: | ---: |
| `ffn_out-0` k8192 rows3072 cols512 | `3874.313 us` | `4290.875 us` |
| `ffn_out-0` k8192 rows3072 cols512 | `3806.438 us` | `4386.000 us` |
| `result_output` k3072 rows128256 cols512 | `60312.563 us` | `60610.438 us` |
| `result_output` k3072 rows200064 cols512 | `95344.500 us` | `95024.313 us` |

Decision: reject and remove from the catalog. This spelling proved that Loom
can compile a larger Q6_K MMQL-style route and that k-step unroll policy is a
material schedule knob, but the high-level 64x64 variant is not the right
production schedule. The next Q6_K attempt should start from a closer HRX1
128x64/B-lifetime prior or a lower-level spelling, not from this 16-accumulator
high-level variant.

Post-revert focused gate:
`cache/hrx2/phase2a/q6-post-reject-baseline-opgate-20260616-082316/`.
Result: 9/9 hot K-quant rows passed, the accepted
`mul_mat_q6_k_q8_1_x4_mmq64x32_k256_32768_r1_262144_c32_512_wg256` route
selected for Q6_K rows, and there were no provider-unavailable events.

## Current Reduced Prefill Baseline

Date: 2026-06-16.

Fresh reduced same-machine prefill slice after the accepted Q4/Q5/Q6 prompt
routes, F16 attention routes, runtime placement fixes, and the Q6 row guard:

```bash
python3 tools/hrx2_phase2a_benchmark.py \
  --tag current-prefill-reduced-20260616-082418 \
  --models phi4-mini-q4,llama32-3b-q4 \
  --cases prefill-p64n0,prefill-p512n0 \
  --backends hrx2,vulkan --repetitions 1 --timeout 1200 --flash-attn 0
```

Artifact: `cache/hrx2/phase2a/current-prefill-reduced-20260616-082418/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | HRX2 dispatches | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | `244.438` | `1472.768` | `0.1660` | 735 | 0 |
| `phi4-mini-q4` | p512/n0 | `732.751` | `4247.314` | `0.1725` | 703 | 0 |
| `llama32-3b-q4` | p64/n0 | `233.290` | `1559.373` | `0.1496` | 698 | 0 |
| `llama32-3b-q4` | p512/n0 | `760.460` | `4830.718` | `0.1574` | 670 | 0 |

Route traces show zero CPU compute fallback. The top dispatch families are still
packed K-quant prompt matmuls plus Q8_1 quantization:

- Llama 3.2 p512: Q4_K MMQ64x32 x166, Q8_1 x4 quantize x110,
  Q6_K MMQ64x32 x27, attention F16 KQ/KQV/softmax x28 each.
- Phi p512: Q8_1 x4 quantize x126, Q4_K MMQ64x32 x79,
  Q5_K MMQ32x32 x32, attention F16 KQ/KQV/softmax x32 each,
  Q6_K MMQ64x32 x15.

Focused backend-op and Vulkan timing evidence still put the largest per-route
gap in prompt K-quant matmul quality. Example p512 shape comparison:

| Shape family | HRX2 focused row | Vulkan timing bucket | Gap |
| --- | ---: | ---: | ---: |
| Q4_K k3072 rows3072 cols512 | `1146 us` | `263-288 us` | about 4x |
| Q4_K k3072 rows8192 cols512 | `3253-3423 us` | `610 us` | about 5x |
| Q4_K k3072 rows16384 cols512 | `6455 us` | `1239 us` | about 5x |
| Q6_K k8192 rows3072 cols512 | `3896-3914 us` | `739-758 us` | about 5x |

ISA check for the accepted Q4_K MMQ64x32 route:
`cache/hrx2/phase2a/q4-accepted-route-isa-20260616-082836/`.
The emitted ISA uses `v_dot4_i32_iu8`, has no reported spills, and the
compile report allocation count is about 461-465 assignments. This makes dot
lowering an unlikely primary explanation for the remaining gap. The stronger
hypothesis is schedule shape: Vulkan's MMQ path stages multiple K blocks
(`BK_STEP=4`) and uses a different tile/work ownership model, while the current
HRX2 Loom route computes one output row by eight columns per thread and
barriers every 32-wide K block.

## Rejected Q4_K BK_STEP4 Probe

Date: 2026-06-16.

To isolate the barrier/staging cadence delta from a full tile rewrite, the
accepted Q4_K MMQ64x32 Loom route was temporarily changed to stage four
32-wide K blocks before the barrier while preserving the existing 64x32 output
tile and one-row/eight-column thread ownership.

Artifacts:

- Initial indexing-bug gate:
  `cache/hrx2/phase2a/q4-bkstep4-opgate-20260616-083215/`.
- Corrected correctness/perf gate:
  `cache/hrx2/phase2a/q4-bkstep4-fixed-opgate-20260616-083353/`.
- Saved rejected patch:
  `cache/hrx2/phase2a/q4-bkstep4-fixed-opgate-20260616-083353/saved/q4-bkstep4-correct-but-regressed.patch`.
- Post-revert baseline gate:
  `cache/hrx2/phase2a/q4-post-bkstep-revert-opgate-20260616-083505/`.

The first attempt compiled and selected the intended route but failed all Q4_K
rows with finite errors around `1.04`; the bug was an incorrect B-scratch
K-slice index for columns 1-7. After fixing that index, the route was
correctness-clean and selected with no provider-unavailable events, but it
regressed every Q4_K perf row:

| Row | Baseline after revert | BK_STEP4 corrected |
| --- | ---: | ---: |
| Q4_K k3072 rows3072 cols512 | `1146.188 us` | `1343.000 us` |
| Q4_K k8192 rows3072 cols512 | `3422.688 us` | `3853.750 us` |
| Q4_K k3072 rows8192 cols512 | `3253.188 us` | `3677.563 us` |
| Q4_K k3072 rows16384 cols512 | `6454.625 us` | `7194.688 us` |

Decision: reject. The experiment proves that a looped high-level BK_STEP4
spelling is legal and correctness-clean after fixing the indexing, but simply
bolting four-step staging onto the current 64x32 one-row-per-thread route is
not the missing performance class. The next Q4_K attempt should move closer to
the Vulkan/CUDA MMQ work ownership and tile shape at the same time, instead of
only reducing barrier count.

## Accepted Q5_K/Q6_K HRX1 HIP Bridge Routes

Date: 2026-06-16.

Motivation: the focused K-quant evidence showed that wide prompt Q5_K and
Q6_K rows were several multiples slower than Vulkan and HRX1 priors. Instead
of continuing local Loom knob sweeps on a schedule-family mismatch, HRX2 now
supports embedded target-specific HSACO artifacts and uses llama.cpp-local HIP
bridge wrappers for the HRX1 wide packed-Q8_1/x4 schedules.

Implementation:

- Q6_K bridge:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp`,
  route
  `mul_mat_q6_k_q8_1_x4_hip_mmql64x128_gfx1100_k256_32768_r64_262144_c128_512_wg256`.
- Q5_K bridge:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q5_k_q8_1_wave64.hip.cpp`,
  route
  `mul_mat_q5_k_q8_1_x4_hip_mmql128x128_gfx1100_k256_32768_r128_262144_c128_512_wg256`.
- Both exports use the HRX2 u32 shape ABI (`k`, `rows`, `cols`) and direct
  embedded `amdgpu-hsaco` artifact loading. The original raw HRX1 Q6 export
  with 64-bit by-value shape args loaded but produced NaNs/Infs, so bridge
  kernels should use explicit HRX2 ABI wrappers rather than raw legacy HIP
  exports.

Focused gate artifacts:

- Q6 u32 bridge correctness/perf:
  `cache/hrx2/phase2a/q6-hip-bridge-u32-opgate-20260616-090550/`.
- Q5+Q6 bridge correctness/perf:
  `cache/hrx2/phase2a/q5q6-hip-bridge-opgate-20260616-091045/`.

Focused perf deltas versus the latest pre-bridge hot-row baseline:

| Row | Before bridge | After bridge |
| --- | ---: | ---: |
| Q6_K k8192 rows3072 cols512 | `~3850-3870 us` | `~483-520 us` |
| Q6_K k3072 rows128256 cols512 | `~60132 us` | `~7970 us` |
| Q6_K k3072 rows200064 cols512 | `~94999 us` | `~12287 us` |
| Q5_K k3072 rows5120 cols512 | `~2156 us` | `~240 us` |

Reduced same-machine HRX2/Vulkan prefill artifact:
`cache/hrx2/phase2a/q5q6-hip-bridge-reduced-20260616-091135/`.

| Model | Case | HRX2 before | HRX2 after | Vulkan after | HRX2/Vulkan after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `233.177` | `231.709` | `1580.106` | `0.1466` |
| `llama32-3b-q4` | p512/n0 | `763.768` | `931.816` | `4818.236` | `0.1934` |
| `phi4-mini-q4` | p64/n0 | `247.054` | `246.705` | `1446.141` | `0.1706` |
| `phi4-mini-q4` | p512/n0 | `734.423` | `967.426` | `4286.982` | `0.2257` |

Superseded decision: the first focused gate and reduced run looked promising,
but the bridge was later re-run standalone and failed repeatedly with
NaNs/Infs. Do not treat this as an accepted default route. The direct HSACO
loading path remains useful infrastructure, but the Q5/Q6 bridge routes are now
opt-in diagnostics only via `GGML_HRX2_ENABLE_Q5_Q6_HIP_BRIDGE_PROMPT`.

## Q5_K/Q6_K HIP Bridge Correction And Q4_K Packed-A Probe

Date: 2026-06-16.

While validating the next Q4_K schedule probe, the mixed K-quant focused gate
failed on Q5_K/Q6_K rows even though the Q4_K rows passed. Isolating the rows
showed the failure was not Q4 scratch corruption:

- Q4-only `q4-a-pack2` gate passed:
  `cache/hrx2/phase2a/q4-a-pack2-isolation-20260616-092634/q4-only/`.
- Q5/Q6-only bridge gate failed twice with NaN/Inf mismatches:
  `cache/hrx2/phase2a/q4-a-pack2-isolation-20260616-092634/q5q6-only/`
  and `.../q5q6-only-second/`.
- Disabling the x4 prompt bridge path made the same Q5/Q6 rows pass on the
  safe Loom routes:
  `cache/hrx2/phase2a/q5q6-fallback-check-20260616-092804/`.

Implementation change: the target-specific HIP bridge routes are skipped by
default in both support checks and dispatch. They can be re-enabled only for
diagnostics with:

```bash
GGML_HRX2_ENABLE_Q5_Q6_HIP_BRIDGE_PROMPT=1
```

The Q4_K `q4-a-pack2` candidate follows the documented Vulkan schedule delta
from `docs/loom/llamacpp-hrx2-q4k-schedule-ledger.md`: stage Q4_K A payload
as four packed words per 64-row tile instead of eight expanded words, using the
`low | high << 4` layout and extracting low/high nibbles in the dot loop.

Focused mixed gate after disabling the bad bridge:
`cache/hrx2/phase2a/q4-a-pack2-safe-routes-opgate-20260616-093021/`.
Result: 9/9 K-quant hot rows passed, with Q4_K selecting
`mul_mat_q4_k_q8_1_x4_mmq64x32_k256_32768_r1_32768_c32_512_wg256`, Q5_K
selecting `mul_mat_q5_k_q8_1_x4_mmq32x32...`, and Q6_K selecting
`mul_mat_q6_k_q8_1_x4_mmq64x32...`. No provider-unavailable events.
Final rebuilt-source focused gate:
`cache/hrx2/phase2a/q4-a-pack2-final-opgate-20260616-093421/`.

Focused perf after the Q4 packed-A change:
`cache/hrx2/phase2a/q4-a-pack2-safe-routes-perf-20260616-093114/`.

| Row | Before packed-A | After packed-A |
| --- | ---: | ---: |
| Q4_K k3072 rows3072 cols512 | `~1143 us` | `969.5 us` |
| Q4_K k8192 rows3072 cols512 | `~3395 us` | `2897.0 us` |
| Q4_K k3072 rows8192 cols512 | `~3230 us` | `2771.8 us` |
| Q4_K k3072 rows16384 cols512 | `~6302 us` | `5491.3 us` |

Reduced same-machine HRX2/Vulkan prefill artifact after the Q4 change and
safe bridge routing:
`cache/hrx2/phase2a/q4-a-pack2-safe-routes-reduced-20260616-093151/`.
Post-rebuild HRX2-only small-model smoke:
`cache/hrx2/phase2a/q4-a-pack2-final-smoke-20260616-093457/`, Llama 3.2 3B
Q4_K_M p64/n0, `234.048 tok/s`, zero CPU fallback.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | HRX2 dispatches | CPU compute | Top route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `llama32-3b-q4` | p64/n0 | `232.102` | `1615.319` | `0.1437` | 698 | 0 | Q4_K MMQ x166 |
| `llama32-3b-q4` | p512/n0 | `811.440` | `4802.689` | `0.1690` | 670 | 0 | Q4_K MMQ x166 |
| `phi4-mini-q4` | p64/n0 | `245.384` | `1451.914` | `0.1690` | 735 | 0 | Q8_1 x4 quantize x126 |
| `phi4-mini-q4` | p512/n0 | `779.777` | `4277.983` | `0.1823` | 703 | 0 | Q8_1 x4 quantize x126 |

Interpretation: Q4_K packed-A is an accepted focused-kernel improvement, but
the net p512 model score is lower than the invalid bridge run because Q5/Q6
bridges are no longer used by default. Current valid prefill remains about
0.14x-0.18x Vulkan on the reduced slice, with zero CPU compute fallback. The
next boulders are still packed K-quant schedule quality, especially a correct
Q5/Q6 wide schedule, and attention-chain/fusion traffic.

## Accepted Q5_K/Q6_K HIP Bridge Workgroup-Size Fix

Date: 2026-06-16.

Root cause: the HRX2 direct-HSACO dispatch path used
`provider->export_info.workgroup_size[0]` before route metadata. The embedded
HIP bridge HSACOs reported a nonzero local size of `1`, while the catalog route
correctly specified `256`. That launched the HRX1-derived Q5/Q6 tiled kernels
with one workitem per workgroup, leaving almost all shared A/B tile entries
uninitialized and producing the repeated NaN/Inf focused failures. The earlier
very fast bridge timings were therefore invalid; they measured an incorrect
one-workitem launch.

Implementation in `sources/llama.cpp`: `_hip_` bridge routes now use
`route.workgroup_size[0]` for dispatch local size. The Q5/Q6 HIP bridge routes
are default-enabled again, with rollback:

```bash
GGML_HRX2_DISABLE_Q5_Q6_HIP_BRIDGE_PROMPT=1
```

Focused evidence:

- Old failing rows fixed:
  `cache/hrx2/phase2a/q5q6-bridge-wgfix2-opgate-20260616-094441/`.
  Result: all five Q5/Q6 rows passed; Q5 and Q6 bridge routes selected with
  `workgroup_size_x=256`; no provider-unavailable events.
- Mixed K-quant gate with Q4 plus Q5/Q6:
  `cache/hrx2/phase2a/q5q6-bridge-final-opgate-20260616-094957/`.
  Result: 9/9 rows passed; Q4 stayed on
  `mul_mat_q4_k_q8_1_x4_mmq64x32...`; Q5/Q6 selected the HIP bridge routes at
  `workgroup_size_x=256`; no provider-unavailable events.
- Rollback gate:
  `cache/hrx2/phase2a/q5q6-bridge-final-disable-opgate-20260616-095030/`.
  Result: same Q5/Q6 rows passed with the bridge disabled, selecting the safe
  Loom Q5/Q6 routes.
- Focused perf:
  `cache/hrx2/phase2a/q5q6-bridge-wgfix-perf-20260616-094552/`.

Focused perf after the real 256-thread launch:

| Row | Safe Loom route | Fixed bridge |
| --- | ---: | ---: |
| Q6_K k8192 rows3072 cols512 | `~3794-3876 us` | `2591-2611 us` |
| Q6_K k3072 rows128256 cols512 | `~60199 us` | `48958 us` |
| Q6_K k3072 rows200064 cols512 | `~94860 us` | `79179 us` |
| Q5_K k3072 rows5120 cols512 | `~2177 us` | `676 us` |

Reduced same-machine default HRX2/Vulkan prefill artifact:
`cache/hrx2/phase2a/q5q6-bridge-default-reduced-20260616-094832/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | HRX2 dispatches | CPU compute | Bridge routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `llama32-3b-q4` | p64/n0 | `230.431` | `1629.174` | `0.1414` | 698 | 0 | none |
| `llama32-3b-q4` | p512/n0 | `936.591` | `4807.533` | `0.1948` | 670 | 0 | Q6 x27 |
| `phi4-mini-q4` | p64/n0 | `247.352` | `1439.925` | `0.1718` | 735 | 0 | none |
| `phi4-mini-q4` | p512/n0 | `963.852` | `4295.064` | `0.2244` | 703 | 0 | Q5 x32, Q6 x15 |

Interpretation: this is a valid boulder fix, but it does not close Phase 2a.
It mainly improves p512, where the wide Q5/Q6 prompt rows are active. p64
does not select the bridge routes and remains dominated by Q4_K prompt matmul,
Q8_1 quantization, and attention/fusion traffic. The next prefill targets are
therefore still Q4_K schedule quality and attention-chain/fusion routes.

## Accepted Q4_K HRX1 HIP Bridge Route

Date: 2026-06-16.

After the Q5/Q6 bridge workgroup-size bug was fixed, the next Q4_K boulder was
tested as a bracketed pivot around a documented HRX1 schedule family rather than
another blind Loom tile mutation. The bridge is a dense prompt version of the
HRX1 Q4_K packed-Q8_1/x4 wide schedule:

- Route:
  `mul_mat_q4_k_q8_1_x4_hip_mmql64x32_gfx1100_k256_32768_r64_32768_c32_512_wg64`.
- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/mul_mat_vec_q4_k_q8_1_wave64.hip.cpp`.
- Algorithm: BM64/BN32, one wave64 workgroup, `TM=4`, `TN=2`, each lane owns
  four rows by eight columns across the tile, staged Q4_K A and packed Q8_1 B,
  and `__builtin_amdgcn_sudot4` for unsigned Q4 codes by signed Q8 activations.
- Default policy: enabled for the gfx1100 HSACO route, with rollback:

```bash
GGML_HRX2_DISABLE_Q4_HIP_BRIDGE_PROMPT=1
```

Focused evidence:

- Opt-in correctness gate:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-opgate-20260616-100100/`.
  Result: 9/9 mixed K-quant hot rows passed; the Q4 bridge selected for all
  four Q4 rows at `workgroup_size_x=64`; no provider-unavailable events.
- Focused perf:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-perf-20260616-100151/`.
- Default-on gate:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-default-opgate-20260616-100354/`.
  Result: Q4 bridge selected by default for all Q4 rows.
- Rollback gate:
  `cache/hrx2/phase2a/q4-hip-mmql64x32-disable-opgate-20260616-100430/`.
  Result: disabling the bridge returns Q4 rows to the Loom
  `mul_mat_q4_k_q8_1_x4_mmq64x32...` route while Q5/Q6 bridges remain active.

Focused Q4 row timings versus the previous accepted Loom Q4 route:

| Row | Previous Loom route | HRX1 HIP bridge | Change |
| --- | ---: | ---: | ---: |
| Q4_K k3072 rows3072 cols512 | `969.5 us` | `449.9 us` | `2.15x` faster |
| Q4_K k8192 rows3072 cols512 | `2897.0 us` | `1158.1 us` | `2.50x` faster |
| Q4_K k3072 rows8192 cols512 | `2771.8 us` | `1211.3 us` | `2.29x` faster |
| Q4_K k3072 rows16384 cols512 | `5491.3 us` | `2922.8 us` | `1.88x` faster |

Reduced same-machine HRX2/Vulkan prefill artifacts:
`cache/hrx2/phase2a/q4-hip-mmql64x32-optin-reduced/` and production-default
confirmation `cache/hrx2/phase2a/q4-hip-mmql64x32-default-reduced/`.

| Model | Case | HRX2 before | HRX2 default after | Vulkan after | HRX2/Vulkan after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `230.431` | `324.121` | `1564.494` | `0.2072` |
| `llama32-3b-q4` | p512/n0 | `936.591` | `1501.680` | `4775.068` | `0.3145` |
| `phi4-mini-q4` | p64/n0 | `247.352` | `326.531` | `1495.554` | `0.2183` |
| `phi4-mini-q4` | p512/n0 | `963.852` | `1436.373` | `4285.831` | `0.3351` |

Decision: accept. This is a large, real Phase 2a boulder movement and validates
the process change: start from a strong schedule prior, document the schedule,
then use adjacent probes only as bounded pivots screened by backend-op gates.
The route is target-specific because it ships a gfx1100 HIP HSACO. The longer
term Loom goal is still to spell this schedule in Loom or lower-level Loom once
the corresponding vector/staging/control forms are equally WYSIWYG.

Remaining prefill gap after this lift is no longer a Q4-only cliff. The reduced
slice is around `0.20x-0.33x` Vulkan. The next boulders should be selected from
fresh traces, with special attention to Q8_1 quantization/backplane reuse and
the unfused attention chain.

## Rejected Q5_K/Q6_K HIP Bridge cols64 Edge-Tile Widening

Date: 2026-06-16.

After the Q4 bridge lift, p64 traces still showed Q5_K/Q6_K prompt rows on the
Loom MMQ routes because the HRX1 HIP bridges were guarded to `cols>=128` and
`cols_multiple_of=128`. The bridge kernels have explicit column edge-tile
checks, so this was tested as a bounded domain-widening probe around the
accepted HRX1 schedule, not as a new schedule guess.

Artifacts:

- p64 op export:
  `cache/hrx2/phase2a/q5q6-p64-op-export-20260616-101144/`.
- Current-route baseline perf:
  `cache/hrx2/phase2a/q5q6-p64-current-perf-20260616-101220/`.
- Widened-domain correctness gate:
  `cache/hrx2/phase2a/q5q6-hip-cols64-opgate-20260616-101324/`.
  Result: all three rows passed; Q5/Q6 HIP bridge routes selected with
  `workgroup_size_x=256`; no provider-unavailable events.
- Widened-domain perf:
  `cache/hrx2/phase2a/q5q6-hip-cols64-perf-20260616-101351/`.

| Row | Current Loom route | HIP bridge cols64 | Change |
| --- | ---: | ---: | ---: |
| Q6_K k8192 rows3072 cols64 | `511.1 us` | `888.5 us` | `1.74x` slower |
| Q5_K k3072 rows5120 cols64 | `304.1 us` | `320.3 us` | `1.05x` slower |
| Q6_K k3072 rows200064 cols64 | `10962.4 us` | `15067.9 us` | `1.37x` slower |

Decision: reject and revert. The HRX1 bridge schedule is useful for p512, but
its 128-column tile wastes enough work on p64 edge tiles that the existing Loom
MMQ routes are better. Keep Q5/Q6 bridge domains at `cols>=128`. Future p64
Q5/Q6 work should use a narrower prior-matched schedule, not the 128-column
bridge as an edge-tile route.

## Rejected F16 Attention Standalone Tiled HIP Bridge Probe

Date: 2026-06-16.

After the Q4 bridge lift, the current p512 attention rows still showed a large
standalone matmul gap:

- KQ p512 focused HRX2 timing was about `2.1 ms` for the
  `k=128 rows=512 cols=512` row, versus Vulkan's matching bucket around
  `0.2 ms`.
- KQV p512 focused HRX2 timing was about `1.7 ms` for the
  `k=512 rows=128 cols=512` row, versus Vulkan's matching bucket under
  `0.1 ms`.

A bounded bridge probe tried to bracket the Vulkan tiled-matmul prior without
committing to the full flash-attention fusion. The temporary HIP bridge used
16x16 output tiles, 256 threads, shared F16 A and F32 B tiles, and exact p512
shape assumptions for the basket attention rows.

Artifacts:

- KQ failing gate:
  `cache/hrx2/phase2a/f16-hip-tile-opgate-20260616-102340/`.
- KQ writeback-layout fix attempt, still failing:
  `cache/hrx2/phase2a/f16-hip-tile-opgate-20260616-102446/`.
- KQV non-fused math gate, failing:
  `cache/hrx2/phase2a/f16-hip-tile-kqv-opgate-20260616-102730/`.

Findings:

- The route selected and loaded the temporary HSACO, so this was not a route
  priority or provider-availability failure.
- The first KQ attempt wrote a contiguous-by-head layout, while the live KQ
  tensor uses nonstandard destination strides. Correcting the obvious writeback
  assumption did not fix correctness.
- The KQV non-fused route also failed CPU-reference despite matching the
  apparent source and destination strides. This makes the simple standalone
  tiled GEMM bridge unsafe as a shortcut.

Decision: reject and remove from the production tree. No llama.cpp code from
this probe was retained. The useful conclusion is that attention should not be
advanced by more standalone rows/cols variants around the current dot-per-output
matmul. The next attention boulder should be a fused streaming attention route
over `KQ -> SOFT_MAX -> KQV -> PERMUTE/CONT`, using Vulkan `flash_attn.comp`
online-softmax dataflow as the portable prior and HRX1 gfx11 direct
flash-attention as the AMD lane/output-ownership prior. Target D=128 and p512
first, with p64 as a separate domain.
## Rejected Q6_K HRX1 BM128/BN64 Bridge Pivot

Date: 2026-06-16.

After the accepted Q4/Q5/Q6 HIP bridge work, a fresh reduced run from commit
`e909aef98` was captured at
`cache/hrx2/phase2a/current-fresh-20260616-103513/`:

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | CPU compute | Dispatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi4-mini-q4` | p64/n0 | `328.750` | `1447.030` | `0.2272` | 0 | 735 |
| `phi4-mini-q4` | p512/n0 | `1447.725` | `4300.615` | `0.3366` | 0 | 703 |
| `llama32-3b-q4` | p64/n0 | `324.308` | `1547.918` | `0.2095` | 0 | 698 |
| `llama32-3b-q4` | p512/n0 | `1507.337` | `4815.075` | `0.3130` | 0 | 670 |

A focused hot-op rerun showed the largest Q6_K output rows remain expensive:
`cache/hrx2/phase2a/current-hot-op-perf-20260616-103627/`.

A bounded schedule pivot exposed the sibling HRX1 Q6_K packed-Q8_1 schedule
`BM128/BN64` (`hrx_mul_mat_vec_q6_k_q8_1_x4_mmql_wg256_impl<128,64>`) in HRX2
alongside the current accepted `BM64/BN128` route. This was a bracket around an
existing HRX1 prior, not a blind schedule guess.

Focused correctness passed all p512 K-quant rows plus p64 Q5/Q6 rows:
`cache/hrx2/phase2a/q6-mmql128x64-opgate-20260616-103911/`. The new route
selected for six Q6 rows and had no provider-unavailable events.

Focused perf rejected it as a production default:
`cache/hrx2/phase2a/q6-mmql128x64-perf-20260616-103957/`.

| Row | Current BM64/BN128 | Candidate BM128/BN64 | Change |
| --- | ---: | ---: | ---: |
| Q6_K result_output rows200064 cols512 | `80147.9 us` | `89848.9 us` | `0.89x` |
| Q6_K result_output rows128256 cols512 | `48445.3 us` | `55133.6 us` | `0.88x` |
| Q6_K ffn_out rows3072 cols512 | `2689.8 us` | `2858.9 us` | `0.94x` |
| Q6_K result_output rows200064 cols64 | `10962.4 us` | `8899.1 us` | `1.23x` |
| Q6_K ffn_out rows3072 cols64 | `511.1 us` | `740.9 us` | `0.69x` |

Decision: reject and remove the route. The p64 huge-output win is too narrow to
justify a production-catalog entry while p512 regresses the dominant output
projection rows. Keep the current `BM64/BN128` Q6 bridge. Future Q6 work needs a
new prior or deeper schedule change, not this adjacent pivot.

## Accepted FA0 F16 Attention-Chain Fusion

Date: 2026-06-16.

The fresh reduced run at `e909aef98` still showed the `--flash-attn 0`
attention chain as a p512 boulder: separate F16/F32 KQ matmul, masked
`SOFT_MAX`, F16/F32 KQV matmul, `PERMUTE`, and final `CONT`. A graph trace in
`cache/hrx2/phase2a/fa0-graph-trace-20260616-105915/` confirmed the exact
Llama/Phi D=128 chain and the live K/V cache layouts:

- K: F16 `[D, KV, H_KV, 1]`, D-contiguous.
- Q: F32 `[D, N, H, 1]`, D-contiguous.
- V: F16 `[KV, D, H_KV, 1]`, KV-contiguous and D-strided.
- mask: F32 `[KV, N, 1, 1]`.
- output: final contiguous F32 `[D * H, N, 1, 1]`.

The accepted route is a gfx11 HIP bridge, not a generic Loom source:

- Source:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/hip/flash_attn_fa0_f32_f16_prefill_direct_d128_gfx11.hip.cpp`.
- Route:
  `flash_attn_fa0_f32_f16_direct_d128_gfx1100_n1_512_kv1_512_h1_64_hkv1_16_wg256`.
- Fusion:
  `MUL_MAT(KQ) -> SOFT_MAX -> MUL_MAT(KQV) -> PERMUTE -> CONT`.
- Rollback:
  `GGML_HRX2_DISABLE_F16_FA0_ATTENTION_FUSION=1`.

This was a prior-driven boulder, not a blind schedule guess. The kernel follows
the HRX1 gfx11 direct attention schedule, adjusted from D256 to D128 and from
`FLASH_ATTN_EXT` to the actual `--flash-attn 0` K/V cache views. The important
implementation detail was the V-cache layout: the first attempt assumed
D-contiguous V, but the live fa0 graph is KV-contiguous, so the accepted bridge
uses strided V WMMA loads.

ABI note: HRX HSACO metadata reports visible buffer and by-value entries in
`parameter_count`. The route ABI is therefore six bindings, seven reflected
parameters, and a 208-byte constant block. Setting `parameter_count=0` made the
provider load fail even though the HSACO and exported symbol were valid.

Focused evidence:

- Provider-load fix and p64 A/B:
  `cache/hrx2/phase2a/fa0-fusion-smoke-20260616-111125/`.
  - p64 enabled: `386.836 tok/s`, 28 `FLASH_ATTN_FA0` dispatches.
  - p64 disabled: `325.248 tok/s`, 56 F16 attention matmuls plus 28
    `SOFT_MAX` and 28 `CONT` dispatches.
- Bounded gates and p64/p512 A/B:
  `cache/hrx2/phase2a/fa0-fusion-gates-20260616-111517/`.
  - Constituent `CONT`, `SOFT_MAX`, and `MUL_MAT` backend-op invocations
    completed with supported rows still passing and unsupported rows skipped.
  - Llama p64: `321.796 -> 384.733 tok/s`.
  - Llama p512: `1516.688 -> 2457.537 tok/s`.
  - No provider failures in the p64/p512 fa0 traces.
- Post-rename smoke:
  `cache/hrx2/phase2a/fa0-fusion-rename-smoke-20260616-111735/`.
  - Llama p64: `384.772 tok/s`.
  - Llama p512: `2455.392 tok/s`.
  - Provider selected the renamed
    `hrx2_flash_attn_fa0_f32_f16_prefill_direct_d128` export.
- Clean final build log:
  `cache/hrx2/phase2a/fa0-fusion-final-build.log`.

Reduced same-machine HRX2/Vulkan evidence:
`cache/hrx2/phase2a/fa0-fusion-reduced/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Dispatches | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `389.348` | `1573.880` | `0.2474` | 614 | 0 |
| `llama32-3b-q4` | p512/n0 | `2406.347` | `4814.025` | `0.4999` | 614 | 0 |
| `phi4-mini-q4` | p64/n0 | `400.148` | `1486.268` | `0.2692` | 639 | 0 |
| `phi4-mini-q4` | p512/n0 | `2253.661` | `4284.555` | `0.5260` | 639 | 0 |

Decision: accept. This is the first attention boulder that moves p512 to the
Phase 2a target band on the small Q4 models. p64 remains farther behind, which
is expected because launch count, quantization, and edge-tile overheads matter
more at that width. The remaining p512 blockers are now dominated by quantized
prompt matmul and Q8_1 backplane reuse rather than the unfused attention chain.

## Rejected Q4_K HIP Bridge BN64 p64 Pivot

Date: 2026-06-16.

After the accepted FA0 fusion, fresh current p64 backend-op timing showed Q4_K
prompt matmul was still a material p64 boulder: current HRX2 rows were roughly
`250 us` for Q4_K `k3072 rows3072 cols64`, `594 us` for
`k8192 rows3072 cols64`, and `303 us` for `k3072 rows8192 cols64`. The
same-machine Vulkan p64 buckets were roughly `134 us`, `228 us`, and `158 us`
for corresponding Q4_K shape families.

A bounded schedule pivot tested a sibling of the accepted HRX1-derived
`BM64/BN32` wave64 HIP bridge using `BM64/BN64` for exact p64 `cols=64` rows.
This bracketed the accepted HRX1 bridge along the column-tile axis and moved
toward the Vulkan dense-MMQ `BN64` prior while keeping the same Q4/Q8 layout,
A/B staging, wave64 ownership, and `sudot4` dot form.

Artifacts:

- Current p64 op timing baseline:
  `cache/hrx2/phase2a/current-p64-op-perf-20260616-112623/`.
- Candidate broad op gate with non-Q4 NaN-sensitive rows:
  `cache/hrx2/phase2a/q4-hip-mmql64x64-opgate-20260616-112941/`.
- Candidate focused Q4 matmul gate and perf:
  `cache/hrx2/phase2a/q4-hip-mmql64x64-matmul-20260616-113126/`.

Focused Q4 matmul correctness passed and the candidate route selected for all
four p64 Q4 `cols=64` rows. Focused perf rejected the pivot:

| Row | Current BN32 bridge | Candidate BN64 bridge | Change |
| --- | ---: | ---: | ---: |
| Q4_K `k3072 rows1024 cols64` | `266.4 us` | `392.2 us` | `0.68x` |
| Q4_K `k3072 rows3072 cols64` | `250.0 us` | `370.1 us` | `0.68x` |
| Q4_K `k8192 rows3072 cols64` | `593.6 us` | `921.8 us` | `0.64x` |
| Q4_K `k3072 rows8192 cols64` | `303.4 us` | `444.6 us` | `0.68x` |

Decision: reject and remove before model integration. The accepted `BN32`
bridge remains the better Q4_K p64 route. Future Q4 p64 work should not simply
widen this bridge to `BN64`; it needs a different schedule/resource explanation
such as Vulkan's multi-warp dense-MMQ ownership, better K-step/barrier
amortization without the prior `BK_STEP4` regression, or a fusion/reuse path
that reduces quantize and launch count.

## Rejected Q4_K SWIGLU HIP Bridge p64 Pivot

Date: 2026-06-16.

The next bounded p64 probe used the HRX1 grouped Q4_K SWIGLU prior:
`hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq_wg64_impl` in
`ggml-hrx/kernels/mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp`. The schedule fused
the gate/up Q4_K matmuls with the SWIGLU epilogue and reused one packed Q8_1
x4 RHS tile across both Q4 operands. It preserved the wave64 `sudot4`
dataflow, Q4/Q8 layout, and cooperative A/B staging pattern, while pivoting to
a prompt-specific `BM16/BN32/BK_STEP2` dense adaptation.

Artifact:
`cache/hrx2/phase2a/q4-swiglu-hip-p64-probe-20260616-114505/`.

The candidate route selected successfully and had no provider failures:

- Enabled dispatches: `560`, including 27
  `mul_mat_q4_k_swiglu_q8_1_x4_hip_mmql16x32...` routes.
- Disabled dispatches: `614`, including 54 standalone Q4_K matmuls plus 27
  standalone `SWIGLU` routes.
- Quantize cache hits dropped from `83` to `56`, consistent with replacing
  adjacent matmul consumers rather than missing the route.

Focused model smoke rejected the pivot despite the dispatch reduction:

| Case | Candidate enabled | Candidate disabled | Change |
| --- | ---: | ---: | ---: |
| Llama 3.2 3B Q4_K_M p64/n0 | `397.055 ms` | `383.876 ms` | `0.97x` |

Decision: reject and remove from the live catalog/runtime changes before
broader integration. The useful lesson is that fusing gate/up/SWIGLU around the
existing HRX1-style dense bridge is not enough by itself; the per-dispatch work
regression erases the launch-count savings at p64. Future Q4_K SWIGLU work
should start from a stronger Q4_K packed-MMQ schedule or a schedule that proves
RHS tile reuse without doubling the row-tile overhead. Treat this as an
example of an allowed adjacent prior pivot that was screened in the focused
kernel/model-smoke loop and rejected before production promotion.

## Accepted Q4_K HIP Bridge Pack2 A-Cache Pivot

Date: 2026-06-16.

The accepted Q4_K prompt bridge still spent too much p64/p512 time on the
HRX1-derived `BM64/BN32` HIP schedule. A bounded schedule pivot kept that
schedule family, route domain, packed Q8_1 x4 RHS, wave64 ownership, LDS
staging, and `sudot4` dot form intact, but changed the Q4 A-cache spelling to a
Vulkan-style packed pair representation:

- Old A cache: eight Q4 pack4 words per row/group.
- Pack2 A cache: four words per row/group, each holding two adjacent Q4 pack4
  payloads in low/high nibbles.

This was not a blind tile guess. It bracketed the accepted HRX1 bridge on one
axis visible in the Vulkan prior: reduce LDS/register traffic for Q4 A payloads
while preserving the proven bridge's launch geometry.

Focused backend-op artifact:
`cache/hrx2/phase2a/q4-hip-pack2-ab-opgate-20260616-115924/`.

The candidate passed p64 and p512 Q4_K backend-op correctness, selected the
pack2 route, and had zero provider-unavailable events. Same-binary perf against
the old bridge:

| Shape row | Old bridge | Pack2 bridge | Change |
| --- | ---: | ---: | ---: |
| p64 `k3072 rows1024 cols64` | `255.419 us` | `205.017 us` | `1.25x` |
| p64 `k3072 rows3072 cols64` | `234.661 us` | `174.571 us` | `1.34x` |
| p64 `k8192 rows3072 cols64` | `584.215 us` | `433.550 us` | `1.35x` |
| p64 `k3072 rows8192 cols64` | `299.581 us` | `226.010 us` | `1.33x` |
| p512 `k3072 rows1024 cols512` | `263.760 us` | `209.235 us` | `1.26x` |
| p512 `k3072 rows3072 cols512` | `452.669 us` | `380.140 us` | `1.19x` |
| p512 `k8192 rows3072 cols512` | `1262.472 us` | `1029.052 us` | `1.23x` |
| p512 `k3072 rows8192 cols512` | `1266.134 us` | `1172.242 us` | `1.08x` |

HRX2-only model smoke:
`cache/hrx2/phase2a/q4-hip-pack2-hrx2-smoke-20260616-120026/`.

| Model | Case | Pack2 enabled | Pack2 disabled | Change |
| --- | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `412.309` | `378.412` | `1.09x` |
| `llama32-3b-q4` | p512/n0 | `2614.351` | `2379.594` | `1.10x` |
| `phi4-mini-q4` | p64/n0 | `414.711` | `400.389` | `1.04x` |
| `phi4-mini-q4` | p512/n0 | `2348.753` | `2193.305` | `1.07x` |

Both models kept identical dispatch counts with zero CPU compute fallback and
zero provider-unavailable events. Enabled traces selected the pack2 route for
all Q4_K bridge dispatches; disabled traces selected the old bridge.

Reduced same-machine HRX2/Vulkan evidence:
`cache/hrx2/phase2a/q4-pack2-reduced-20260616-120046/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Dispatches | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `421.511` | `1577.378` | `0.2672` | 614 | 0 |
| `llama32-3b-q4` | p512/n0 | `2629.953` | `4754.954` | `0.5531` | 614 | 0 |
| `phi4-mini-q4` | p64/n0 | `418.188` | `1454.354` | `0.2875` | 639 | 0 |
| `phi4-mini-q4` | p512/n0 | `2350.494` | `4243.802` | `0.5539` | 639 | 0 |

Decision: accept. The p512 Q4 slice is now above the Phase 2a 0.5x Vulkan
target for the two small Q4 models. p64 remains roughly 0.27x-0.29x Vulkan, so
the next narrow-prompt work should focus on launch/quantize reuse and
small-width fusion/backplane effects rather than more blind tile widening.
The route has a dedicated rollback knob:
`GGML_HRX2_DISABLE_Q4_HIP_PACK2_PROMPT=1`.

## Rejected Q8_1 Graph-Local Cache Probe

Date: 2026-06-16.

After pack2, p64 traces still showed many Q8_1 quantize dispatches:

- Llama 3.2 3B p64: 110 quantize dispatches and 83 one-entry cache hits.
- Phi-4-mini p64: 126 quantize dispatches and zero one-entry cache hits.

The tempting hypothesis was that HRX2's current last-entry Q8_1 cache was too
small and that a graph-local associative cache would recover reuse when
different matmul sources were interleaved. A bounded runtime probe kept the
same quantizer and matmul routes but replaced the Q8_1 scratch path with
multiple graph-local scratch buffers keyed by source tensor, layout, shape, and
strides. It had rollback `GGML_HRX2_DISABLE_Q8_1_GRAPH_CACHE=1` during the
probe and was never committed.

Artifacts:

- Focused Q4 op gate:
  `cache/hrx2/phase2a/q8-graph-cache-q4-opgate-20260616-120718/`.
- p64 model smoke:
  `cache/hrx2/phase2a/q8-graph-cache-p64-smoke-20260616-120753/`.

Focused Q4 op replay passed, but it did not prove a useful model effect
because extracted-op replay already has adjacent cache locality:

- Enabled and disabled both had 12 quantize dispatches and 28,257 quantize
  cache hits in perf replay.
- Focused row timings were neutral to slightly positive but not representative
  of model graph order.

Model smoke rejected the probe:

| Model | Graph cache enabled | Graph cache disabled | Change |
| --- | ---: | ---: | ---: |
| `llama32-3b-q4` p64/n0 | `404.033` | `416.516` | `0.97x` |
| `phi4-mini-q4` p64/n0 | `403.503` | `416.930` | `0.97x` |

The traces showed no dispatch reduction and no new cache hits. Llama kept the
same 110 quantize dispatches and 83 cache hits; Phi kept the same 126 quantize
dispatches and zero hits. The added scratch-buffer churn only made the run
slower.

Decision: reject and remove before commit. The p64 quantize/backplane issue is
not solved by a larger cache over the current tensor identities. Future work
should instead inspect graph dataflow for producer/consumer fusion or explicit
Q8_1 backplane materialization shared by planned consumers, then prove it with
a route-level or graph-fusion sweep before model integration.

## Accepted RMS_NORM_MUL Default Fusion

Date: 2026-06-16.

After pack2, p64 traces still had a launch-heavy norm pattern:

- Llama 3.2 3B p64: 55 `RMS_NORM` plus 55 `MUL` dispatches.
- Phi-4-mini p64: 63 `RMS_NORM` plus 63 `MUL` dispatches.

The RMS_NORM_MUL route already existed and had prior unit coverage, but HRX2
kept it behind opt-in `GGML_HRX2_ENABLE_RMS_NORM_MUL_FUSION=1`. A same-binary
A/B showed that enabling the existing route is a small but consistent p64 and
p512 win:

Artifact:
`cache/hrx2/phase2a/rms-norm-mul-env-ab-20260616-121005/`.

| Model | Case | Fusion enabled | Fusion disabled | Change |
| --- | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `422.678` | `415.419` | `1.017x` |
| `llama32-3b-q4` | p512/n0 | `2680.126` | `2668.927` | `1.004x` |
| `phi4-mini-q4` | p64/n0 | `417.417` | `412.024` | `1.013x` |
| `phi4-mini-q4` | p512/n0 | `2356.411` | `2337.138` | `1.008x` |

The default-flip smoke used the final rollback name
`GGML_HRX2_DISABLE_RMS_NORM_MUL_FUSION=1`:
`cache/hrx2/phase2a/rms-norm-mul-default-smoke-20260616-121209/`.

| Model | Default | Disabled | Dispatch delta |
| --- | ---: | ---: | ---: |
| `llama32-3b-q4` p64/n0 | `421.065` | `415.145` | `614 -> 559` |
| `phi4-mini-q4` p64/n0 | `424.252` | `417.873` | `639 -> 576` |

Focused primitive RMS_NORM backend-op coverage passed:
`cache/hrx2/phase2a/rms-norm-mul-default-opgate-20260616-121149/`
with 10 supported rows passing and no provider-unavailable events. The route
selection evidence in the model smoke confirms `rms_norm_mul_f32...` replaced
the separate RMS_NORM and MUL dispatches.

Reduced same-machine HRX2/Vulkan evidence:
`cache/hrx2/phase2a/rms-norm-mul-default-reduced-20260616-121223/`.

| Model | Case | HRX2 | Vulkan | HRX2/Vulkan | Dispatches | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `427.605` | `1594.633` | `0.2682` | 559 | 0 |
| `llama32-3b-q4` | p512/n0 | `2708.038` | `4767.205` | `0.5681` | 559 | 0 |
| `phi4-mini-q4` | p64/n0 | `422.684` | `1454.181` | `0.2907` | 576 | 0 |
| `phi4-mini-q4` | p512/n0 | `2377.027` | `4328.190` | `0.5492` | 576 | 0 |

Decision: accept. This is not the remaining p64 boulder, but it removes
55-63 dispatches with consistent positive model evidence and no kernel changes.
RMS_NORM_MUL is now default-on under the global fusion gate, with rollback
`GGML_HRX2_DISABLE_RMS_NORM_MUL_FUSION=1`.

## Rejected ADD_RMS_NORM_MUL Prefill Route Widening

Date: 2026-06-16.

After `RMS_NORM_MUL` became default, the p64 traces still had 54-62 standalone
`ADD` dispatches. The existing `ADD_RMS_NORM_MUL` route was decode-only
(`nrows=1`), while the live residual paths use `nrows=64` and `nrows=512`.
A bounded metadata-only probe added `n3072/r64`, `n3072/r512`,
`n4096/r64`, and `n4096/r512` route entries over the existing Loom source.
No kernel source was changed.

Artifact:
`cache/hrx2/phase2a/add-rms-norm-mul-prefill-smoke-20260616-121706/`.

The route selected correctly, removed another 56-64 dispatches, and had zero
provider-unavailable or CPU fallback events:

| Model | Case | Enabled | Disabled | Dispatch delta |
| --- | ---: | ---: | ---: | ---: |
| `llama32-3b-q4` | p64/n0 | `412.566` | `417.649` | `561 -> 505` |
| `llama32-3b-q4` | p512/n0 | `2666.708` | `2705.366` | `561 -> 505` |
| `phi4-mini-q4` | p64/n0 | `422.139` | `419.978` | `578 -> 514` |
| `phi4-mini-q4` | p512/n0 | `2359.347` | `2319.406` | `578 -> 514` |

Decision: reject and remove before commit. The same n3072 route domain improves
Phi but regresses Llama, so route metadata cannot safely distinguish the useful
case. Future residual fusion should either explain the model-topology
difference and add a stronger guard, or produce a different schedule that does
not increase per-dispatch work enough to erase the launch-count savings on
Llama.

## Rejected Q4_K Pack2 BK_STEP4 Pivot

Date: 2026-06-16.

After the accepted Q4_K pack2 HIP bridge, the schedule ledger still listed one
reasonable adjacent probe: keep the accepted BM64/BN32 wave64 pack2 route but
stage four Q8 blocks per barrier. This bracketed the Vulkan `BK_STEP=4` prior
around the current production schedule instead of guessing a new tile family.

Temporary implementation:

- Route:
  `mul_mat_q4_k_q8_1_x4_hip_mmql64x32_pack2_bkstep4_gfx1100_k256_32768_r64_32768_c32_512_wg64`.
- Export:
  `hrx2_mul_mat_vec_q4_k_q8_1_x4_mmql64x32_pack2_bkstep4_wg64_u32`.
- Gate:
  `GGML_HRX2_ENABLE_Q4_HIP_PACK2_BKSTEP4_PROMPT=1`.
- Saved rejected patch:
  `cache/hrx2/phase2a/q4-pack2-bkstep4-opgate-20260616-122640/rejected-q4-pack2-bkstep4.patch`.

Focused p64 correctness passed and the route selected with zero
provider-unavailable events:
`cache/hrx2/phase2a/q4-pack2-bkstep4-opgate-20260616-122640/`.

Focused p64 timing was decisively worse than the accepted pack2 route:

| Row | Accepted pack2 | BK_STEP4 pack2 | Change |
| --- | ---: | ---: | ---: |
| Q4_K k3072 rows1024 cols64 | `205.1 us` | `1192.7 us` | `5.8x` slower |
| Q4_K k3072 rows3072 cols64 | `174.7 us` | `1671.9 us` | `9.6x` slower |
| Q4_K k8192 rows3072 cols64 | `434.3 us` | `4296.4 us` | `9.9x` slower |
| Q4_K k3072 rows8192 cols64 | `225.2 us` | `2651.9 us` | `11.8x` slower |

The p512 broad perf run was stopped after the p64 rejection because it was
using a mixed op file and generated a very large trace. The p64 result is
already enough to reject this pivot for the current p64 bottleneck.

Decision: reject and remove before commit. `BK_STEP=4` is not profitable when
bolted onto the accepted single-wave BM64/BN32 pack2 bridge. Future Q4 work
should not retry this axis inside the same tile/dataflow. The next Q4 schedule
probe needs a genuinely different Vulkan-style ownership/tile family, such as
BM128/BN64 or BM128/BN128 with WMITER-style pressure control, and should again
start as a backend-op sweep rather than model integration.

## Repeated Prefill Measurement Control

Date: 2026-06-16.

After `RMS_NORM_MUL` became default, the reduced same-machine one-shot run
still showed p64 behind Vulkan:
`cache/hrx2/phase2a/rms-norm-mul-default-reduced-20260616-121223/`.

The trace counters did not support a pure device-kernel explanation for that
p64 wall-time gap: HRX2 provider traces summed only a few milliseconds of
dispatch/flush/sync events while `llama-bench` reported roughly 150 ms for the
first p64 sample. A repeated same-process control was run for HRX2 and Vulkan:

Artifact:
`cache/hrx2/phase2a/repeated-prefill-hrx2-vulkan-20260616-123059/`.

Command shape:
`llama-bench -p {64,512} -n 0 -b 512 -ub 512 -fa 0 -r 3 --no-warmup`.

| Backend | Model | p | Samples tok/s | Steady tok/s |
| --- | --- | ---: | --- | ---: |
| HRX2 | Llama 3.2 3B Q4_K_M | 64 | `429.9, 1225.7, 1233.8` | `1229.8` |
| Vulkan | Llama 3.2 3B Q4_K_M | 64 | `421.6, 1228.4, 1233.9` | `1231.2` |
| HRX2 | Llama 3.2 3B Q4_K_M | 512 | `2695.8, 3333.7, 3051.0` | `3192.3` |
| Vulkan | Llama 3.2 3B Q4_K_M | 512 | `2694.5, 3355.9, 3112.5` | `3234.2` |
| HRX2 | Phi-4 mini Q4_K_M | 64 | `430.8, 1239.7, 1268.1` | `1253.9` |
| Vulkan | Phi-4 mini Q4_K_M | 64 | `429.9, 1242.5, 1270.8` | `1256.7` |
| HRX2 | Phi-4 mini Q4_K_M | 512 | `2404.7, 2964.5, 2594.7` | `2779.6` |
| Vulkan | Phi-4 mini Q4_K_M | 512 | `2409.5, 2896.5, 2611.3` | `2753.9` |

Steady-state HRX2/Vulkan ratios for this reduced prefill smoke are:

| Model | p64 | p512 |
| --- | ---: | ---: |
| Llama 3.2 3B Q4_K_M | `0.999x` | `0.987x` |
| Phi-4 mini Q4_K_M | `0.998x` | `1.009x` |

Decision update: reject this artifact for HRX2/Vulkan KPI comparisons. The
"Vulkan" `bench.json` rows in
`cache/hrx2/phase2a/repeated-prefill-hrx2-vulkan-20260616-123059/` report
`backends=HRX2`, so the apparent parity was caused by backend/library
contamination in the one-off runner. The reusable Phase 2a harness was updated
to report cold and steady-state samples, and later runs use backend-specific
library paths. Future Phase 2a dashboards must verify `backends` in
`llama-bench` JSON before drawing conclusions.

This does not invalidate prior-driven kernel work. It changes how to screen new
schedule ideas: bracket adjacent tile/vector/unroll/staging pivots in focused
backend-op sweeps first, then run repeated same-run HRX2/Vulkan model baskets
only after the sweep shows a material bucket-level win.

## Accepted Q4_K Vulkan-Medium HIP Bridge Pivot

Date: 2026-06-16.

The corrected repeated default-three run showed that p64 remained below the
Phase 2a target after the accepted Q4 pack2 route:
`cache/hrx2/phase2a/repeated-prefill-default3-20260616-123611/`.

Steady-state HRX2/Vulkan before this pivot:

| Model | p64 | p512 |
| --- | ---: | ---: |
| Llama 3.2 3B Q4_K_M | `0.3959x` | `0.5182x` |
| Phi-4 mini Q4_K_M | `0.4547x` | `0.5082x` |
| Llama 3.1 8B Q4_K_M | `0.3928x` | `0.4311x` |

Provider traces had zero CPU compute fallback. The top HRX2 route family for
Llama/Q4 rows was still the accepted
`mul_mat_q4_k_q8_1_x4_hip_mmql64x32_pack2...`; Phi also showed many
`quantize_q8_1_x4_f32_generic_wg128` dispatches.

The next bracketed schedule pivot followed the Vulkan medium K-quant integer
MMQ tuple visible in `ggml-vulkan.cpp` for AMD/RADV:

- `BLOCK_SIZE=128`
- `BM=64`
- `BN=64`
- `WM=64`
- `WN=32`
- `WMITER=1`
- `TM=2`
- `TN=2`
- `WARP=64`

The implementation keeps the accepted pack2 Q4 A-cache, packed Q8_1/x4 RHS
layout, LDS staging, and `sudot4` dot form, but uses two logical wave64 tiles
per workgroup to cover a 64-column output tile. Route:
`mul_mat_q4_k_q8_1_x4_hip_vkm64x64_pack2_gfx1100_k256_32768_r64_32768_c64_512_wg128`.
Export:
`hrx2_mul_mat_vec_q4_k_q8_1_x4_vkm64x64_pack2_wg128_u32`.
Rollback:
`GGML_HRX2_DISABLE_Q4_HIP_VKM64X64_PROMPT=1`.

Focused opt-in backend-op gate:
`cache/hrx2/phase2a/q4-vkm64x64-opgate-20260616-124259/`.

The candidate passed p64 and p512 CPU-reference rows, selected for the Q4
prompt rows, and had zero provider-unavailable events. Prompt Q4 rows versus
the accepted pack2 route:

| Shape row | Pack2 | VKM64x64 | Change |
| --- | ---: | ---: | ---: |
| p64 `k3072 rows1024 cols64` | `205.017 us` | `173.800 us` | `1.18x` |
| p64 `k3072 rows3072 cols64` | `174.571 us` | `140.197 us` | `1.25x` |
| p64 `k8192 rows3072 cols64` | `433.550 us` | `362.168 us` | `1.20x` |
| p64 `k3072 rows8192 cols64` | `226.010 us` | `201.065 us` | `1.12x` |
| p512 `k3072 rows1024 cols512` | `209.235 us` | `194.114 us` | `1.08x` |
| p512 `k3072 rows3072 cols512` | `380.140 us` | `372.289 us` | `1.02x` |
| p512 `k8192 rows3072 cols512` | `1029.052 us` | `999.300 us` | `1.03x` |
| p512 `k3072 rows8192 cols512` | `1172.242 us` | `1016.750 us` | `1.15x` |

HRX2-only model smoke with the route enabled:
`cache/hrx2/phase2a/q4-vkm64x64-hrx2-smoke-20260616-124520/`.

| Model | Case | Pack2 steady | VKM64x64 steady | Change |
| --- | ---: | ---: | ---: | ---: |
| Llama 3.2 3B Q4_K_M | p64 | `1251.080` | `1396.695` | `1.116x` |
| Llama 3.2 3B Q4_K_M | p512 | `3190.215` | `3606.395` | `1.131x` |
| Phi-4 mini Q4_K_M | p64 | `1272.090` | `1395.295` | `1.097x` |
| Phi-4 mini Q4_K_M | p512 | `2754.630` | `3210.550` | `1.166x` |
| Llama 3.1 8B Q4_K_M | p64 | `679.352` | `767.805` | `1.130x` |
| Llama 3.1 8B Q4_K_M | p512 | `1149.180` | `1416.060` | `1.232x` |

Default route-selection gate after changing the route from opt-in to
default-on:
`cache/hrx2/phase2a/q4-vkm64x64-default-opgate-20260616-124657/`.
All p64/p512 focused rows passed, the new route selected by default, and
provider-unavailable remained zero.

Reduced same-machine HRX2/Vulkan rerun:
`cache/hrx2/phase2a/q4-vkm64x64-default-reduced-20260616-124844/`.

| Model | Case | HRX2 steady | Vulkan steady | HRX2/Vulkan | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 3B Q4_K_M | p64 | `1396.955` | `3135.370` | `0.4455` | 0 |
| Llama 3.2 3B Q4_K_M | p512 | `3565.295` | `5858.440` | `0.6086` | 0 |
| Phi-4 mini Q4_K_M | p64 | `1356.755` | `2718.435` | `0.4991` | 0 |
| Phi-4 mini Q4_K_M | p512 | `3078.145` | `5170.735` | `0.5953` | 0 |
| Llama 3.1 8B Q4_K_M | p64 | `761.497` | `1540.455` | `0.4943` | 0 |
| Llama 3.1 8B Q4_K_M | p512 | `1391.015` | `2608.720` | `0.5332` | 0 |

Decision: accept and default-enable. This is a prior-led schedule pivot, not a
blind tile guess, and it improves all focused Q4 rows plus all six repeated
model rows. It brings p512 above the Phase 2a target and puts Phi/8B p64 at the
target edge, but Llama 3.2 3B p64 remains below target. The next p64 boulder is
not this exact axis again; route traces now put the remaining work at Q4 prompt
matmul quality plus Q8_1 x4 quantize/reuse and residual launch-heavy ADD/fusion
traffic.

### Accepted Q4_K Vulkan-Medium Wave32 Cols64 Narrow Pivot

Date: 2026-06-16.

The previous default VKM64x64 bridge still left p64 below or at the Phase 2a
target. A prior-led bracket tested the same Vulkan-medium K-quant tile as a
true wave32 HIP image instead of the earlier wave64 adaptation:

- `BLOCK_SIZE=128`
- `BM=64`
- `BN=64`
- `WM=32`
- `WN=32`
- `WMITER=1`
- `TM=2`
- `TN=2`
- `WARP=32`

The HIP compiler flag for wave32 on this ROCm clang is
`-mno-wavefrontsize64`; `-mwavefrontsize32` is not accepted. HSACO metadata for
`hrx2_mul_mat_vec_q4_k_q8_1_x4_vkm64x64_pack2_wg128_w32_u32` confirms
`.wavefront_size: 32`, `vgpr_count: 95`, `sgpr_count: 40`, zero private
segment, and 4096 bytes LDS.

Initial broad opt-in route:
`cache/hrx2/phase2a/q4-vkm64x64-w32-opgate-20260616-131818/`.
It passed focused p64/p512 MUL_MAT correctness and selected for Q4 rows with no
provider-unavailable events. Backend-op perf showed the route is p64-specific:

| Shape row | Existing VKM64x64 | Wave32 broad | Change |
| --- | ---: | ---: | ---: |
| p64 `k3072 rows1024 cols64` | `173.998 us` | `158.521 us` | `1.098x` |
| p64 `k3072 rows3072 cols64` | `142.558 us` | `127.424 us` | `1.119x` |
| p64 `k8192 rows3072 cols64` | `367.911 us` | `330.732 us` | `1.112x` |
| p64 `k3072 rows8192 cols64` | `207.450 us` | `199.565 us` | `1.040x` |
| p512 `k3072 rows1024 cols512` | `197.872 us` | `186.637 us` | `1.060x` |
| p512 `k3072 rows3072 cols512` | `377.005 us` | `422.495 us` | `0.892x` |
| p512 `k8192 rows3072 cols512` | `1010.764 us` | `1172.171 us` | `0.862x` |
| p512 `k3072 rows8192 cols512` | `1035.105 us` | `1112.181 us` | `0.931x` |

The route was narrowed to two data-driven cols64 domains to avoid the p512
regression and the model-specific p64 regressions:

- `k=3072`, `rows=1024..8192`, `cols=64`
- `k=8192`, `rows=3072`, `cols=64`

Default focused gate after narrowing:
`cache/hrx2/phase2a/q4-vkm64x64-w32-default-opgate-20260616-132707/`.
Both p64 and p512 MUL_MAT correctness passed with zero provider-unavailable
events. The p64 gate selected the two narrow wave32 routes; the p512 gate stayed
on the existing wave64 VKM64x64 route. Rollback:
`GGML_HRX2_DISABLE_Q4_HIP_VKM64X64_W32_PROMPT=1`.

Same-binary p64 HRX2-only smoke:

| Model | Default before | Narrow wave32 default | Change |
| --- | ---: | ---: | ---: |
| Phi-4 mini Q4_K_M | `1389.415` | `1438.915` | `1.036x` |
| Llama 3.2 3B Q4_K_M | `1405.980` | `1490.805` | `1.060x` |
| Llama 3.1 8B Q4_K_M | `772.745` | `781.945` | `1.012x` |

Reduced same-machine HRX2/Vulkan rerun:
`cache/hrx2/phase2a/q4-w32-narrow-default-reduced-20260616-132805/`.

| Model | Case | HRX2 steady | Vulkan steady | HRX2/Vulkan | CPU compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 3B Q4_K_M | p64 | `1486.525` | `3149.130` | `0.4720` | 0 |
| Llama 3.2 3B Q4_K_M | p512 | `3576.075` | `6095.020` | `0.5867` | 0 |
| Phi-4 mini Q4_K_M | p64 | `1431.490` | `2781.430` | `0.5147` | 0 |
| Phi-4 mini Q4_K_M | p512 | `3160.940` | `5335.150` | `0.5925` | 0 |
| Llama 3.1 8B Q4_K_M | p64 | `782.226` | `1542.550` | `0.5071` | 0 |
| Llama 3.1 8B Q4_K_M | p512 | `1414.535` | `2655.665` | `0.5326` | 0 |

Decision: accept and default-enable only for the two narrow cols64 domains.
This is a useful p64 boulder fix without p512 exposure. Phi and 8B p64 now
clear the `0.5x` Vulkan target, and Llama 3.2 p64 moved from `0.4455x` to
`0.4720x`. The remaining Llama 3.2 p64 gap is not solved by broad wave32 alone;
the next p64 work should target route-level Q8_1 x4 quantization/reuse,
attention/fusion traffic, or a new Q4 ownership/staging family seeded from
Vulkan/HRX1 disassembly.

### Rejected Q4_K BM128/BN64 Bracket

Date: 2026-06-16.

A bounded adjacent probe tried to reuse the accepted Vulkan-medium Q4_K HIP
bridge dataflow but double the row tile from `BM64` to `BM128` while keeping
`BN64`, pack2 Q4 A-cache, packed Q8_1/x4 RHS, `sudot4`, and the same
per-wave `WM64/WN32/TM2/TN2` ownership. This brackets the hypothesis that p64
could benefit from staging a 64-column B tile once for two row groups.

The candidate was opt-in only via
`GGML_HRX2_ENABLE_Q4_HIP_VKL128X64_PROMPT`. Focused correctness traces:
`cache/hrx2/phase2a/q4-vkl128x64-opgate-20260616-125932/`.
No provider-unavailable events occurred, and the new route selected for the
four Q4 prompt rows in both p64 and p512 test files. Full perf tracing was
aborted because it produced hundreds of MB of JSONL and distorted the gate;
clean no-trace perf was rerun at
`cache/hrx2/phase2a/q4-vkl128x64-perf-notrace-20260616-130057/`.

Prompt Q4 rows versus the accepted VKM64x64 default:

| Shape row | VKM64x64 | BM128/BN64 | Change |
| --- | ---: | ---: | ---: |
| p64 `k3072 rows1024 cols64` | `173.998 us` | `177.832 us` | `0.978x` |
| p64 `k3072 rows3072 cols64` | `142.558 us` | `151.921 us` | `0.938x` |
| p64 `k8192 rows3072 cols64` | `367.911 us` | `375.675 us` | `0.979x` |
| p64 `k3072 rows8192 cols64` | `207.450 us` | `208.542 us` | `0.995x` |
| p512 `k3072 rows1024 cols512` | `197.872 us` | `199.215 us` | `0.993x` |
| p512 `k3072 rows3072 cols512` | `377.005 us` | `368.106 us` | `1.024x` |
| p512 `k8192 rows3072 cols512` | `1010.764 us` | `992.308 us` | `1.019x` |
| p512 `k3072 rows8192 cols512` | `1035.105 us` | `1026.489 us` | `1.008x` |

Decision: reject and remove before model integration. The probe is useful as a
bracketed negative result: doubling the row tile does not address the p64 miss
and slightly hurts every p64 Q4 prompt row. The next Q4 schedule move should
change a different structural axis, such as B/K staging depth, lane/output
ownership, or an HRX1/Vulkan disassembly-matched implementation, and it should
again start as a backend-op sweep.

### Stable Llama 3.2 p64 Gap After Wave32 Q4

Date: 2026-06-16.

The previous reduced run left only one selected prefill case below the strict
`0.5x` Vulkan line: Llama 3.2 3B Q4_K_M p64. A longer isolated rerun confirms
that this is stable rather than a one-run artifact:

Artifact:
`cache/hrx2/phase2a/llama32-p64-rerun-current-20260616-134601/`.

| Backend | Samples tok/s | Steady tok/s |
| --- | --- | ---: |
| HRX2 | `436.6, 1499.6, 1478.1, 1498.7, 1520.8, 1538.8, 1561.2` | `1516.182` |
| Vulkan | `1630.4, 3191.1, 3132.8, 3096.3, 3127.1, 3119.1, 3133.8` | `3133.367` |

Steady ratio: `0.4839x`. CPU fallback remains zero. The HRX2 route trace over
seven repetitions has `3913` dispatches, `19` provider compiles, `74` stream
synchronizes, `239` stream flushes, and `322` submit-batch flushes.

Top active route families:

| Route | Dispatches |
| --- | ---: |
| `mul_mat_q4_k_q8_1_x4_hip_vkm64x64_pack2_w32_gfx1100_k3072_r1024_8192_c64_wg128` | `1064` |
| `quantize_q8_1_x4_f32_generic_wg128` | `770` |
| `rms_norm_mul_f32_n3072_r64_vector_vw4_wg512` | `385` |
| `add_f32_generic_wg256` | `378` |
| `flash_attn_fa0_f32_f16_direct_d128_gfx1100_n1_512_kv1_512_h1_64_hkv1_16_wg256` | `196` |
| `mul_mat_q6_k_q8_1_x4_mmq64x32_k256_32768_r1_262144_c32_512_wg256` | `189` |
| `swiglu_f32_split_n8192_r1_64_wg256` | `189` |
| `mul_mat_q4_k_q8_1_x4_hip_vkm64x64_pack2_w32_gfx1100_k8192_r3072_c64_wg128` | `98` |

Same-run Vulkan perf logger shows the remaining large per-shape gaps are still
quantized prompt matmuls. In the steady samples Vulkan reports approximately:

| Vulkan bucket | Avg time |
| --- | ---: |
| Q4_K `m=1024 n=64 k=3072` | `25 us` |
| Q4_K `m=3072 n=64 k=3072` | `56-58 us` |
| Q4_K `m=3072 n=64 k=8192` | `149-152 us` |
| Q4_K `m=8192 n=64 k=3072` | `102-106 us` |
| Q6_K `m=1024 n=64 k=3072` | `28-29 us` |
| Q6_K `m=3072 n=64 k=8192` | `164-169 us` |

The matching HRX backend-op rows after the accepted wave32 route remain roughly
`158.7 us`, `128.9 us`, `335.5 us`, `201.0 us`, `180.6 us`, and `565.7 us`.
The p64 miss should therefore stay focused on quantized matmul schedule quality
or Q8_1 materialization/reuse. Smaller runtime fusions are useful only if they
prove positive with model A/B.

### Rejected Q6_K Small-Row HIP Cols64 Bridge

Date: 2026-06-16.

A narrow metadata probe tested whether the existing HRX1 Q6 HIP bridge should
cover the active Llama 3.2 p64 attention value projection:
`k=3072`, `rows=1024`, `cols=64`. This was intentionally narrower than the
earlier rejected broad cols64 bridge so it could not disturb the known-regressed
Q6 ffn/output rows.

Artifact:
`cache/hrx2/phase2a/q6-smallrow-hip-cols64-probe-20260616-134234/`.

The touched row passed backend-op correctness and selected the new route:
`mul_mat_q6_k_q8_1_x4_hip_mmql64x128_gfx1100_k3072_r1024_c64_wg256`.
The full exported p64 op file still exits nonzero because of pre-existing
NaN-equality failures in unrelated one-column CONT/GET_ROWS/SOFT_MAX/ROPE/GLU
rows; those failures are not caused by the Q6 route and the Q6 rows themselves
passed.

Perf result:

| Row | Current Loom route | Narrow HIP bridge | Change |
| --- | ---: | ---: | ---: |
| Q6_K `k3072 rows1024 cols64` | `180.575 us` | `246.680 us` | `0.73x` |

Decision: reject and remove. The HRX1 bridge remains a p512/wide-Q6 tool, not
the p64 small-row answer. The next Q6 small-row attempt should be a different
schedule family, likely closer to Vulkan's fast `m=1024,n=64,k=3072` path, not
another route-domain widening of the existing `mmql64x128` bridge.

### Rejected Q4_K Wave32 Fast-Math Compile Flag

Date: 2026-06-16.

A low-cost compiler-codegen probe rebuilt only the Q4 wave32 HIP HSACO with
`-ffast-math`, leaving the accepted wave32 schedule and route domains unchanged.
This tested whether the remaining p64 gap was partly from conservative float
scale/min arithmetic lowering rather than schedule shape.

Artifact:
`cache/hrx2/phase2a/q4-w32-fastmath-opperf-20260616-134756/`.

Focused p64 backend-op perf was uniformly worse:

| Q4 row | Current wave32 | `-ffast-math` wave32 | Change |
| --- | ---: | ---: | ---: |
| `k3072 rows1024 cols64` | `158.720 us` | `177.547 us` | `0.89x` |
| `k3072 rows3072 cols64` | `128.886 us` | `145.496 us` | `0.89x` |
| `k3072 rows8192 cols64` | `201.048 us` | `214.209 us` | `0.94x` |
| `k8192 rows3072 cols64` | `335.481 us` | `379.830 us` | `0.88x` |

Decision: reject and remove. Keep the explicit `-O3 -mno-wavefrontsize64`
wave32 build. Future Q4 p64 work should change the schedule or emitted ISA
shape, not global fast-math flags.

### Rejected Q4_K Wave32 TM4/TN1 Ownership Bracket

Date: 2026-06-16.

A bounded opt-in ownership probe kept the accepted Q4 Vulkan-medium wave32
dataflow fixed:

- `BM=64`, `BN=64`, `BK_STEP=1`, `BLOCK_SIZE=128`
- packed-Q8_1-x4 RHS and pack2 Q4 A cache
- `WARP=32`, `WM=32`, `WN=32`, `WMITER=1`
- same narrow p64 route domains as the accepted wave32 route

The only intended schedule-axis change was per-lane output ownership:
`TM=2,TN=2` became `TM=4,TN=1`. This tested whether fewer per-lane column
accumulators and more row outputs would help the p64 small-row path.

Artifact:
`cache/hrx2/phase2a/q4-w32-tm4tn1-opperf-20260616-135444/`.

The candidate selected correctly for the p64 Q4 rows with zero
provider-unavailable events, but focused backend-op perf was uniformly worse:

| Q4 row | Current wave32 `TM2/TN2` | Candidate `TM4/TN1` | Change |
| --- | ---: | ---: | ---: |
| `k3072 rows1024 cols64` | `158.720 us` | `175.287 us` | `0.91x` |
| `k3072 rows3072 cols64` | `128.886 us` | `143.654 us` | `0.90x` |
| `k3072 rows8192 cols64` | `201.048 us` | `216.585 us` | `0.93x` |
| `k8192 rows3072 cols64` | `335.481 us` | `376.658 us` | `0.89x` |

Decision: reject and remove before model integration. The accepted wave32
ownership map should remain `TM2/TN2`. Future Q4 p64 work should pivot a
different structural axis, such as a Vulkan-disassembly-matched row/column
grouping, Q8_1 materialization/reuse, or a true fusion that removes repeated
Q8 quantize plus standalone GLU traffic.

### Rejected Q4_K Wave32 BM32/WG64 Parallelism Bracket

Date: 2026-06-16.

A temporary op-level replacement tested whether the accepted Q4 wave32
Vulkan-medium route was under-parallelized on the smallest p64 rows. The probe
kept the current export name and dataflow but changed the launch shape from
`BM=64`, `BLOCK_SIZE=128`, `rows_per_workgroup=64` to `BM=32`,
`BLOCK_SIZE=64`, `rows_per_workgroup=32`. This doubled row workgroups for
`rows=1024` and tested the broad hypothesis that small-row p64 was limited by
too few workgroups rather than per-workgroup schedule quality.

Artifact:
`cache/hrx2/phase2a/q4-w32-bm32-opperf-20260616-140144/`.

Focused backend-op perf was uniformly worse:

| Q4 row | Current wave32 BM64/WG128 | Candidate BM32/WG64 | Change |
| --- | ---: | ---: | ---: |
| `k3072 rows1024 cols64` | `158.720 us` | `168.879 us` | `0.94x` |
| `k3072 rows3072 cols64` | `128.886 us` | `140.038 us` | `0.92x` |
| `k3072 rows8192 cols64` | `201.048 us` | `216.749 us` | `0.93x` |
| `k8192 rows3072 cols64` | `335.481 us` | `373.390 us` | `0.90x` |

Decision: reject and remove before integration. The p64 miss is not fixed by
simply increasing row workgroup count in this schedule family. Future Q4 p64
work should compare against Vulkan/HRX1 prior schedule facts and target a
different dataflow axis, especially Q8_1 materialization/reuse or a
Vulkan-matched packed matmul schedule.

### Accepted Q6_K Wave32 BK_STEP4 Cols64 Prompt Bridge

Date: 2026-06-16.

The next prior-led Q6 probe targeted the remaining Llama 3.2 p64 cols64 Q6
rows. The current Loom `mmq64x32` route was far from Vulkan on the hot p64
rows, while the existing HRX1 HIP bridge was only available for cols128+ and a
metadata-only cols64 widening had previously regressed the small-row case.

The accepted route keeps the Vulkan/HRX1 packed-Q8_1-x4 MMQ dataflow but uses
a cols64-specific wave32 tile:

- `BM=64`, `BN=64`, `BLOCK_SIZE=128`
- four wave32 tiles per workgroup, `WM=32`, `WN=32`
- `TM=2`, `TN=2`, `WNITER=8`
- packed Q6_K A and packed Q8_1-x4 B staged through LDS
- `BK_STEP=4`; the initial `BK_STEP=1` variant was rejected

Artifact:
`cache/hrx2/phase2a/q6-w32-vkm64x64-opgate-20260616-140920/`.

Focused correctness passed for the touched p64 Q6 rows. Same-binary focused
perf, using `GGML_HRX2_DISABLE_Q5_Q6_HIP_BRIDGE_PROMPT=1` as the baseline for
the current Loom path:

| Q6 row | Current Loom route | Q6 wave32 `BK_STEP=1` | Q6 wave32 `BK_STEP=4` | Accepted change |
| --- | ---: | ---: | ---: | ---: |
| `k3072 rows1024 cols64` | `178.834 us` | `206.558 us` | `162.701 us` | `1.10x` |
| `k8192 rows3072 cols64` | `561.460 us` | not selected | `336.034 us` | `1.67x` |

The `BK_STEP=1` result is an important negative: simply moving the Q4 wave32
ownership pattern to Q6 was not enough. The winning variant needed the HRX1
Q6/Vulkan staging fact, reducing barrier frequency by staging four K blocks per
LDS fill.

Reduced three-model p64/p512 HRX2/Vulkan rerun:
`cache/hrx2/phase2a/q6-w32-vkm64x64-reduced-20260616-141320/`.

| Model | Case | Previous HRX2 steady | New HRX2 steady | HRX2 speedup | New HRX2/Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 3B Q4_K_M | p64 | `1486.525` | `1547.035` | `1.041x` | `0.4936` |
| Llama 3.2 3B Q4_K_M | p512 | `3576.075` | `3595.320` | `1.005x` | `0.5946` |
| Phi-4 mini Q4_K_M | p64 | `1431.490` | `1528.635` | `1.068x` | `0.5630` |
| Phi-4 mini Q4_K_M | p512 | `3160.940` | `3189.665` | `1.009x` | `0.6023` |
| Llama 3.1 8B Q4_K_M | p64 | `782.226` | `777.189` | `0.994x` | `0.4541` |
| Llama 3.1 8B Q4_K_M | p512 | `1414.535` | `1424.220` | `1.007x` | `0.5372` |

Decision: accept the two narrow Q6 cols64 routes. The Llama 3.2 p64 KPI moves
close to the `0.5x` line but remains slightly short; the top blocker is still
Q4_K prompt matmul. Future Q6 work should start from this `BK_STEP=4` wave32
cols64 family and bracket around real prior axes rather than using the current
Loom `mmq64x32` schedule as the only reference.

### Rejected Q4_K Wave32 BK_STEP4 Staging Bracket

Date: 2026-06-16.

After the Q6_K wave32 `BK_STEP=4` win, the same prior axis was tested on the
accepted Q4_K wave32 p64 bridge. This was a controlled bracket: keep the
accepted `BM64/BN64/WG128/wave32/TM2/TN2/pack2` Q4 schedule and change only
K staging from `BK_STEP=1` to `BK_STEP=4`, matching Vulkan's default MMQ
staging knob.

Artifact:
`cache/hrx2/phase2a/q4-w32-bkstep4-opperf-20260616-141721/`.

Focused backend-op correctness for the Q4 MUL_MAT rows passed, but perf
regressed catastrophically:

| Q4 row | Current wave32 `BK_STEP=1` | Candidate `BK_STEP=4` | Change |
| --- | ---: | ---: | ---: |
| `k3072 rows1024 cols64` | `158.720 us` | `1283.848 us` | `0.12x` |
| `k3072 rows3072 cols64` | `128.886 us` | `1660.229 us` | `0.08x` |
| `k3072 rows8192 cols64` | `201.048 us` | `3093.751 us` | `0.07x` |
| `k8192 rows3072 cols64` | `335.481 us` | `4306.994 us` | `0.08x` |

Decision: reject and remove. Do not generalize the Q6 staging result to Q4.
The Q4 p64 gap needs a different structural change, likely Q8_1
materialization/reuse, a different row/column ownership family, or a closer
Vulkan ISA/schedule match rather than only increasing staged K blocks.

### Accepted Q5_K Wave32 Cols64 Prompt Bridge

Date: 2026-06-16.

The latest reduced traces showed Phi-4 mini p64 still spending 96 dispatches on
the generic Loom Q5_K `mmq32x32` path:

```text
mul_mat_q5_k_q8_1_x4_mmq32x32... k3072 rows5120 cols64 x96
```

This was the same broad family as the accepted Q6 cols64 wave32 bridge: a
cols64 prompt row that was too narrow for the existing HRX1-derived wide Q5
HIP bridge (`cols>=128`) but hot enough to matter. The accepted Q5 route keeps
the HRX1/Vulkan Q5 packed-Q8_1-x4 dataflow and `sudot4` arithmetic, but uses a
cols64 wave32 tile:

- `BM=64`, `BN=64`, `BLOCK_SIZE=128`
- four wave32 tiles per workgroup, `WM=32`, `WN=32`
- `TM=2`, `TN=2`, `WNITER=8`
- packed Q5_K A and packed Q8_1-x4 B staged through LDS
- `BK_STEP=1`, matching the proven wide Q5 bridge rather than the Q6-specific
  `BK_STEP=4` result

Artifacts:

- correctness: `cache/hrx2/phase2a/q5-w32-vkm64x64-opgate-20260616-142728/`
- focused perf: `cache/hrx2/phase2a/q5-w32-vkm64x64-perf-20260616-142752/`
- Phi p64 smoke: `cache/hrx2/phase2a/q5-w32-vkm64x64-phi-smoke-20260616-142837/`
- reduced basket: `cache/hrx2/phase2a/q5-w32-vkm64x64-reduced-20260616-142906/`

Focused backend-op correctness passed for the Phi p64 Q5/Q6 hot rows, and the
Q5 row selected:

```text
mul_mat_q5_k_q8_1_x4_hip_vkm64x64_w32_gfx1100_k3072_r5120_c64_wg128
```

Same-binary focused perf, using `GGML_HRX2_DISABLE_Q5_Q6_HIP_BRIDGE_PROMPT=1`
as the old-route baseline:

| Row | Old route | New route | Change |
| --- | ---: | ---: | ---: |
| Q5_K `k3072 rows5120 cols64` | `310.158 us` | `194.608 us` | `1.59x` |
| Q6_K `k8192 rows3072 cols64` | `525.669 us` | `333.406 us` | unchanged accepted Q6 route |
| Q6_K `k3072 rows200064 cols64` | `11979.508 us` | `11850.718 us` | noise |

Reduced three-model p64/p512 HRX2/Vulkan rerun:

| Model | Case | Previous HRX2 steady | New HRX2 steady | HRX2 speedup | New HRX2/Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phi-4 mini Q4_K_M | p64 | `1528.635` | `1656.325` | `1.084x` | `0.5957` |
| Phi-4 mini Q4_K_M | p512 | `3189.665` | `3198.405` | `1.003x` | `0.5942` |
| Llama 3.2 3B Q4_K_M | p64 | `1547.035` | `1558.710` | `1.008x` | `0.4950` |
| Llama 3.2 3B Q4_K_M | p512 | `3595.320` | `3634.705` | `1.011x` | `0.6087` |
| Llama 3.1 8B Q4_K_M | p64 | `777.189` | `780.342` | `1.004x` | `0.4699` |
| Llama 3.1 8B Q4_K_M | p512 | `1424.220` | `1414.365` | `0.993x` | `0.5349` |

Decision: accept the narrow Phi p64 Q5 route. The top blocker for Phi p64
remains Q8_1 quantization and residual Q4/Q6 prompt traffic, but the Q5 row
is no longer on the generic Loom schedule. Future Q5 cols64 work should start
from this wave32 `BM64/BN64` route and widen only after focused op gates prove
adjacent shapes.
