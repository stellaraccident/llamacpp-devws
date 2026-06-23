# HRX2 Phase 2 Tranche 1: Fusion Substrate Bringup

Date: 2026-06-15

This tranche prepared HRX2 for evidence-driven fusion work without admitting any
new fused runtime routes yet. The intent is to make Phase 2 churn manageable:
small catalog files, reproducible dataflow mining, prior-art notes, and explicit
candidate records that can later be promoted only with measured evidence.

## Implemented

- Split the HRX2 catalog source of truth into
  `sources/llama.cpp/ggml/src/ggml-hrx2/catalog/`.
- Added `assemble_hrx2_catalog.py` so CMake and dev tools assemble the split
  catalog into the monolithic v1 JSON that the existing C++ embedding/linking
  tools consume.
- Updated the HRX2 CMake path to generate
  `build/llama-hrx2/ggml/src/ggml-hrx2/generated/catalog.json` from the split
  source files before validation, linking, and embedding.
- Updated workspace HRX2 Python helpers so directory catalogs are loaded by
  assembling them in memory.
- Extended catalog validation for `fusions[]` candidate records.
- Added `tools/hrx2_mine_fusion_candidates.py` to mine producer-consumer chains
  from basket `sched.jsonl` traces and attach route/cache-key evidence from
  matching `hrx2.jsonl` dispatch traces.
- Added `docs/loom/llamacpp-hrx2-phase2-prior-art.md` with local prior-art
  findings from Vulkan, CUDA, and HRX/Pyre notes.

## Catalog Layout

The checked-in catalog layout is:

```text
sources/llama.cpp/ggml/src/ggml-hrx2/catalog/
  metadata.json
  sources.json
  artifacts.json
  families.json
  routes/
    index.json
    *.json
  fusions/
    candidates.json
```

`routes/index.json` is an ordered list of route ids. Route detail files are
grouped by family, and the assembler restores the stable monolithic route order
from the index.

The production binary still embeds JSON. The split layout is for authoring and
review only.

## Fusion Mining

Command used:

```bash
python3 tools/hrx2_mine_fusion_candidates.py \
  cache/hrx2/phase1_0/basket-smoke-route-slice-48-20260613-044836 \
  --json-out cache/hrx2/phase2_0/fusion-candidates-20260615.json \
  --md-out cache/hrx2/phase2_0/fusion-candidates-20260615.md \
  --top 80
```

Result:

- Input scheduler traces: 33
- Candidate chains: 948
- Top motifs by score:
  - `ADD -> ADD -> ADD -> ADD`
  - `ADD -> ADD -> ADD`
  - `MUL_MAT -> SOFT_MAX -> MUL_MAT -> CONT`
  - `MUL_MAT -> SOFT_MAX -> MUL_MAT`
  - `MUL_MAT_ID -> GLU -> MUL_MAT_ID -> MUL`
  - `MUL_MAT -> GLU -> MUL_MAT -> RMS_NORM`

The current score is intentionally simple: estimated intermediate bytes saved,
estimated dispatches saved, and model coverage. It is a ranking input, not an
acceptance metric. Acceptance still requires same-target fused-vs-unfused timing
and backend op correctness against ggml CPU reference.

## Seeded Catalog Candidates

`catalog/fusions/candidates.json` now records these candidate classes:

- `candidate_multi_add`
- `candidate_rms_norm_mul`
- `candidate_mul_mat_add`
- `candidate_attention_matmul_softmax_matmul`
- `candidate_ffn_matvec_glu_epilogue`
- `candidate_moe_matvec_glu_epilogue`
- `candidate_mul_mat_id_add_id_mul`
- `candidate_rms_norm_mul_rope` (deferred)
- `candidate_rope_view_set_rows` (deferred)
- `candidate_topk_moe`

These records do not change route selection. They are planning and evidence
anchors for Phase 2.

## Verification

Catalog assembly and validation:

```bash
python3 sources/llama.cpp/ggml/src/ggml-hrx2/tools/assemble_hrx2_catalog.py \
  --catalog-dir sources/llama.cpp/ggml/src/ggml-hrx2/catalog \
  --out /tmp/hrx2-catalog-assembled.json

python3 sources/llama.cpp/ggml/src/ggml-hrx2/tools/validate_hrx2_catalog.py \
  --catalog /tmp/hrx2-catalog-assembled.json \
  --source-root sources/llama.cpp/ggml/src/ggml-hrx2 \
  --artifact-root build/llama-hrx2/ggml/src/ggml-hrx2/generated/catalog
```

Build:

```bash
cmake --build build/llama-hrx2 --target ggml-hrx2 test-backend-ops -j"$(nproc)"
```

Focused backend op gate:

```bash
env ROCM_PATH="$PWD/rocm" GGML_HRX_ROCM_PATH="$PWD/rocm" \
  LD_LIBRARY_PATH="$PWD/build/hrx-install/lib:$PWD/build/hrx-install/lib64:$PWD/rocm/lib:$PWD/rocm/lib/rocm_sysdeps/lib:$PWD/build/llama-hrx2/bin:${LD_LIBRARY_PATH:-}" \
  timeout 60s build/llama-hrx2/bin/test-backend-ops test -b HRX20 \
  -o RMS_NORM -p 'type=f32' --output csv
```

Supported RMS_NORM rows passed. The unsupported in-place/vector rows remained
unsupported as expected.

## Remaining Risks

- The current miner reconstructs dataflow from exported scheduler names. It is
  useful for ranking but not authoritative enough to admit routes without exact
  route-specific validation.
- Attention-adjacent candidates should wait for the current Loom/direct-dispatch
  compiler fix path before route admission.
- The Phase 2 candidate list should be refined with broader basket traces before
  committing to an implementation order.
- Promotion from candidate to accepted route needs a per-candidate fused benchmark
  that compares against the best measured unfused chain on the same target and
  shape bucket.
