# Q8_0/F32 MUL_MAT Pre-Flight Report: gfx1100

Date: 2026-06-12

## Scope

This pass used Q8_0/F32 `MUL_MAT` to prove the HRX2 op-by-op workflow after
the RMS_NORM exemplar. The goal was to validate the pipeline on a nontrivial
quantized matmul family while the user watched:

- harvest prior art;
- generate target-neutral standalone Loom candidates with real Q8_0 fixtures;
- tune exact shapes with compile reports;
- materialize measured `gfx1100` route rows;
- validate selected runtime routes against ggml CPU reference.

This is not the final Q8_0 algorithm sweep. It proves the authoring and tuning
backplane and identifies the next Q8 axes.

## Prior-Art Harvest

Reusable entries were added to
`docs/loom/backend-prior-art-algorithms.md`.

Important priors:

- OpenCL/Metal and old HRX use multiple rows per workgroup/subgroup for Q8_0
  matvec-style kernels.
- CUDA and old HRX/Pyre indicate prompt-like Q8_0/F32 often wants packed or
  quantized RHS forms such as Q8_1/x4, not only scalar F32 RHS reads.
- Old HRX/Pyre found Q8_0 `MUL_MAT + ADD` fusion important. HRX2 should tune it
  only after accepted standalone Q8_0/F32 and ADD routes exist.
- The current scalar Q8_0/F32 source does not use target-specific WMMA layouts,
  so it should stay target-neutral. `gfx1100` belongs in route evidence, not in
  the Loom source.

## Tooling Added

`tools/hrx2_q8_0_f32_tune.py` generates exact-shape standalone Loom candidates
with:

- deterministic nonzero packed Q8_0 LHS fixtures;
- deterministic F32 RHS fixtures;
- CPU expected F32 outputs;
- rows-per-workgroup and workgroup-size candidate axes;
- `iree-benchmark-loom --measure=dispatch_complete` timing;
- compile report capture: instructions, code bytes, spills,
  local/private memory, and peak live register pressure.

This fixed the phase0.4 issue where the checked-in Q8 benchmark declarations
planned zero dispatch samples.

## Commands

Primary bounded pre-flight sweep:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id gfx1100-q8-0-f32-preflight-20260612 \
  --shapes 512x64x1,512x64x8,4096x128x1,4096x128x8 \
  --workgroup-sizes 64,128,256 \
  --rows-per-workgroup 1,2,4 \
  --iterations 8 \
  --warmup-iterations 2 \
  --repetitions 2 \
  --timeout 20
```

ggml fixture-shape sweep for runtime validation:

```bash
./tools/hrx2_q8_0_f32_tune.py \
  --run-id gfx1100-q8-0-f32-test-shape-20260612 \
  --shapes 256x16x8 \
  --workgroup-sizes 64,128,256 \
  --rows-per-workgroup 1,2,4 \
  --iterations 8 \
  --warmup-iterations 2 \
  --repetitions 2 \
  --timeout 20
```

Focused ggml CPU-reference validation:

```bash
GGML_HRX2_TRACE_JSONL=cache/hrx2/q8_preflight_route_trace.jsonl \
  LD_LIBRARY_PATH="$PWD/build/hrx-install/lib:$PWD/rocm/lib:${LD_LIBRARY_PATH:-}" \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32,m=16,n=8,k=256' --output csv
```

## Results

All 45 standalone candidates compiled, passed Loom correctness, and
benchmarked successfully. All accepted candidates had zero spills.

| Shape | Winner | p50 ns | p90 ns | Inst | Code bytes | Peak live |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `k512_r64_c1` | rows/WG 1, WG256 | 21311 | 21900 | 112 | 540 | 15 |
| `k512_r64_c8` | rows/WG 1, WG64 | 21711 | 22741 | 107 | 508 | 17 |
| `k4096_r128_c1` | rows/WG 1, WG256 | 22961 | 24041 | 112 | 540 | 15 |
| `k4096_r128_c8` | rows/WG 1, WG128 | 37251 | 39311 | 112 | 540 | 17 |
| `k256_r16_c8` | rows/WG 1, WG256 | 20401 | 21560 | 110 | 492 | 11 |

Rows-per-workgroup 2 and 4 were valid but did not win this scalar baseline.
They increased instruction count, code bytes, and peak live values. Keep the
axis for future packed/RHS-quantized variants, but do not promote it for the
current scalar baseline.

## Runtime Materialization

Materialized measured exact routes:

- `mul_mat_q8_0_f32_k256_r16_c8_wg256`
- `mul_mat_q8_0_f32_k512_r64_c8_wg64`

The broad fallback/static Q8 routes remain in place. The checked-in
`mul_mat_q8_0_f32.loom` source is now target-neutral. Existing and new tuned
rows remain `target_key = "gfx1100"` because their evidence is from this card.
The generic contiguous fallback leaves `target_key` empty so other gfx targets
can compile the scalar baseline before target-specific tuning evidence exists.

Runtime validation selected the new exact route:

```text
route_id=mul_mat_q8_0_f32_k256_r16_c8_wg256
cache_key=mul_mat_q8_0_f32_k256_r16_c8_wg256|target=gfx1100|k=256|rows=16|cols=8|...
workgroups_x=16
workgroups_y=8
workgroup_size_x=256
```

`test-backend-ops` passed the supported contiguous Q8_0/F32 case and correctly
reported unsupported non-contiguous/batched variants.

## Interpretation

The workflow is now proven for a quantized matmul family:

- standalone Loom tuning with realistic fixtures works;
- compile report signals are captured and usable;
- route materialization can specialize exact Q8 shapes;
- ggml CPU-reference validation confirms runtime integration;
- target-neutral source plus target-keyed measured routes is the correct
  layering for non-WMMA scalar/vector baselines.

Q8_0/F32 is not algorithmically finished. The scalar F32 RHS baseline is useful
as a control and fallback, but prior art says the larger wins likely require:

- Q8_1 or otherwise packed RHS prompt variants;
- dot-like grouped loads when Loom can express the target-independent form;
- `MUL_MAT + ADD` fusion measured against selected standalone parts;
- model-basket shape evidence before broad route promotion.

## Future Op Regime Update

For matmul-like families, do not stop at scalar baselines. Use this exact
workflow, but add the algorithm families discovered from prior art as explicit
tuning axes. Promote only measured target rows. Introduce target-specific Loom
source only when using target-fiddly primitives such as chip-specific WMMA
layouts.
