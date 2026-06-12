# HRX2 Phase 0.3 Results

Phase 0.3 implements the scaled kernel-catalog backplane on the existing pilot
families only:

- `RMS_NORM` / `rms_norm_f32_contiguous`
- `MUL_MAT` / `mul_mat_q8_0_f32` Q8_0 x F32 routes

The goal was not broader op coverage. The goal was proving the authoring and
tuning substrate before a phase 1 sweep.

## Implemented

- HRX2 backend JSONL tracing:
  - `GGML_HRX2_TRACE_JSONL=/path/events.jsonl`
  - `GGML_HRX2_TRACE_ROUTES=1`
- HRX2 JIT evidence dump:
  - `GGML_HRX2_EVIDENCE_DIR=/path/dir`
  - writes per-provider `provider.json`, `compile_report.json`, and
    `manifest.json` keyed by provider cache key.
- Root pipeline tools:
  - `tools/hrx2_collect_shapes.py`
  - `tools/hrx2_generate_candidates.py`
  - `tools/hrx2_run_loom_sweep.py`
  - `tools/hrx2_reduce_tuning.py`
  - `tools/hrx2_emit_catalog.py`

The sweep runner compiles every candidate with standalone `loom-compile
--backend=amdgpu-hal`, captures compile reports/manifests, and then runs
`test-backend-ops` CPU-reference validation for the priority-selected route that
the backend can actually exercise.

## Smoke Run

Commands:

```bash
python3 tools/hrx2_collect_shapes.py \
  --fixtures-only \
  --out cache/hrx2/shapes/phase0.3.jsonl

python3 tools/hrx2_generate_candidates.py \
  --shapes cache/hrx2/shapes/phase0.3.jsonl \
  --out cache/hrx2/candidates/phase0.3.jsonl

python3 tools/hrx2_run_loom_sweep.py \
  --candidates cache/hrx2/candidates/phase0.3.jsonl \
  --run-id phase0.3-smoke \
  --timeout 90

python3 tools/hrx2_reduce_tuning.py \
  --run cache/hrx2/runs/phase0.3-smoke \
  --out cache/hrx2/reduced/gfx1100/phase0.3.json

python3 tools/hrx2_emit_catalog.py \
  --reduced cache/hrx2/reduced/gfx1100/phase0.3.json \
  --out cache/hrx2/catalog/phase0.3
```

Results:

- shapes: 6
- candidates: 9
- accepted selected route validations: 6
- compile-only fallback alternates: 3
- rejected candidates: 0

Selected route validations:

| Route | Shape | HSACO bytes | ggml CPU reference |
| --- | --- | ---: | --- |
| `rms_norm_f32_contiguous` | `ncols=64,nrows=60` | 9184 | pass |
| `rms_norm_f32_contiguous` | `ncols=1025,nrows=60` | 9184 | pass |
| `mul_mat_q8_0_f32_exact_skinny_pot` | `k=256,rows=16,cols=1` | 9304 | pass |
| `mul_mat_q8_0_f32_exact_narrow_pot` | `k=256,rows=16,cols=16` | 9304 | pass |
| `mul_mat_q8_0_f32_exact_wide_pot` | `k=256,rows=1,cols=64` | 9304 | pass |
| `mul_mat_q8_0_f32_exact_wide_irregular` | `k=5120,rows=6,cols=4096` | 9304 | pass |

The generic contiguous Q8_0 route compiled for three matmul shapes but was not
backend-selected because the exact-shape routes have higher catalog priority.

## Validation

- `cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops -j$(nproc)`
- `validate_hrx2_catalog.py --require-artifacts` on
  `cache/hrx2/catalog/phase0.3/catalog.json`
- Focused `test-backend-ops` with
  `GGML_HRX2_CATALOG_DIR=$PWD/cache/hrx2/catalog/phase0.3`
- Provider cache trace on the RMS_NORM suite:
  - one provider miss
  - one compile success
  - nine cache hits
  - ten dispatches

## Notes

An initial fixture included `m=1,n=1,k=256` for Q8_0 MUL_MAT. The ggml
operation harness does not currently instantiate that exact case, so the tool
now reports empty harness matches as `no_matching_ggml_case` and the fixture
set uses only shapes with available ggml CPU-reference coverage.
