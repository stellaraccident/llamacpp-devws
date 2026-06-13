---
name: hrx2-coverage-audit
description: Use this when auditing llama.cpp HRX2 model-basket execution to find CPU compute fallbacks, missing unfused kernels, provider ownership gaps, graph and tensor shapes, fusion candidates, and data-driven kernel-family backlog items.
metadata:
  short-description: Audit HRX2 model-basket kernel coverage
---

# HRX2 Coverage Audit

Use this skill to turn model-basket runs into a prioritized HRX2 kernel and
fusion backlog with shape evidence suitable for Loom authoring and tuning.

## Required Context

Read first:

- `docs/loom/llamacpp-integration-v1.md`
- `docs/loom/llamacpp-hrx-authoring.md` if a missing kernel is ready to author
- `docs/loom/backend-prior-art-algorithms.md` before proposing family names or fusions
- `docs/spike/kernel-skill/SKILL.md` only for old-HRX profiling commands or reference behavior

## Workflow

1. Verify workspace state with `tools/status.py`.
2. Use `tools/download_hrx2_model_basket.py --dry-run` and the local
   `basket_manifest.json` to identify basket GGUFs.
3. Run decode, narrow multi-token, and prefill workloads for each model with
   HRX2 provider tracing, graph tracing, and profiling enabled.
4. Record every ggml node or subgraph that falls through HRX2, including tensor
   dtype, quant/layout, contiguity, broadcast/aliasing facts, and concrete shape.
5. Classify every CPU execution:
   - compute-kernel fallback: backlog item;
   - missing fusion: backlog item only after standalone parts are represented;
   - host orchestration, sampling, I/O, tokenization: out of scope;
   - unsupported due to model load or memory: environment issue.
6. Record shape evidence, including model/file, graph op or fusion, quant,
   layout, dtype, regime, rows, columns/tokens, K, experts/routes, dispatch
   count, latency contribution, and backend owner.
7. Group fallbacks by family, quant/layout, regime, and shape bucket. Preserve
   exact shapes in JSONL; buckets are a reduction artifact, not the raw evidence.
8. Rank by basket frequency, device/CPU time, dispatch count, overhead
   sensitivity, and whether the op blocks a larger fusion.
9. For likely fusions, record the selected standalone parts and the unfused
   route costs needed to later prove the fusion beats the sum of its parts.

## Output

Write a concise audit report under `docs/loom/` or `cache/` as requested. Each
backlog row should include:

```text
family, ggml op or subgraph, model/file, quant, regime, shape facts,
current owner, fallback reason, priority, suggested Loom family name,
prior-art search roots, standalone prerequisite families
```

## Guardrails

- Do not optimize from wall tok/s alone.
- Do not count normal sampler/host work as a kernel coverage failure.
- Do not propose fusion materialization before the unfused kernels have
  standalone evidence and route ownership.
- Keep models, profiles, and scratch data under `shared/`, `cache/`, or `build/`.
- Do not port old HIP route decisions directly into HRX2; use them only as
  reference evidence.
- Keep evidence in JSON or JSONL so it can feed the HRX2 catalog and tuning
  reducers.
