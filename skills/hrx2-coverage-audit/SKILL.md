---
name: hrx2-coverage-audit
description: Use this when auditing llama.cpp HRX2 model-basket execution to find CPU compute fallbacks, missing unfused kernels, provider ownership gaps, graph shapes, and the next kernel-family backlog items.
metadata:
  short-description: Audit HRX2 model-basket kernel coverage
---

# HRX2 Coverage Audit

Use this skill to turn model-basket runs into a prioritized HRX2 kernel backlog.

## Required Context

Read first:

- `docs/loom/llamacpp-integration-v1.md`
- `docs/loom/llamacpp-hrx-authoring.md` if a missing kernel is ready to author
- `docs/spike/kernel-skill/SKILL.md` only for old-HRX profiling commands or reference behavior

## Workflow

1. Verify workspace state with `tools/status.py`.
2. Use `tools/download_hrx2_model_basket.py --dry-run` and the local
   `basket_manifest.json` to identify basket GGUFs.
3. Run decode and prefill workloads for each model with HRX2 provider tracing,
   graph tracing, and profiling enabled once HRX2 exists.
4. Classify every CPU execution:
   - compute-kernel fallback: backlog item;
   - host orchestration, sampling, I/O, tokenization: out of scope;
   - unsupported due to model load or memory: environment issue.
5. Record shape evidence, including model/file, graph op or fusion, quant,
   layout, dtype, regime, rows, columns/tokens, K, experts/routes, dispatch
   count, latency contribution, and backend owner.
6. Group fallbacks by family, quant/layout, regime, and shape key.
7. Rank by basket frequency, device/CPU time, dispatch count, and whether the
   op blocks a larger fusion.

## Output

Write a concise audit report under `docs/loom/` or `cache/` as requested. Each
backlog row should include:

```text
family, ggml op or subgraph, model/file, quant, regime, shape facts,
current owner, fallback reason, priority, suggested Loom family name
```

## Guardrails

- Do not optimize from wall tok/s alone.
- Do not count normal sampler/host work as a kernel coverage failure.
- Keep models, profiles, and scratch data under `shared/`, `cache/`, or `build/`.
- Do not port old HIP route decisions directly into HRX2; use them only as
  reference evidence.
- Keep evidence in JSON or JSONL so it can feed the HRX2 catalog and tuning
  reducers.
