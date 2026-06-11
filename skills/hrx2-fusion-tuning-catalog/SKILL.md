---
name: hrx2-fusion-tuning-catalog
description: Use this when tuning HRX2 Loom kernels or fusions offline, comparing fused routes against the sum of unfused parts, selecting measured winners, and emitting generated catalog inputs for llama.cpp.
metadata:
  short-description: Tune HRX2 fusions and emit catalogs
---

# HRX2 Fusion Tuning Catalog

Use this skill to run the offline tuning loop that produces data-driven HRX2
catalog records.

## Required Context

Read first:

- `docs/loom/llamacpp-integration-v1.md`
- `docs/loom/llamacpp-hrx-authoring.md`
- the family notes for every candidate being tuned

## Workflow

1. Build a candidate matrix from declared family/provider/config axes. Keep the
   matrix bounded and reproducible.
2. Tune one root/candidate per artifact by default, keyed by source hash, root,
   config, target key, pass program, ABI, and Loom version.
3. Run correctness before benchmarking. Reject failed candidates immediately.
4. Benchmark with `iree-benchmark-loom` or the current Loom benchmark flow.
   Use stable repetitions and keep JSON/JSONL evidence.
5. For each fusion, benchmark the fused candidate and the best measured
   unfused parts under the same target, shape, and method.
6. Accept a fusion only when it is at least 3% faster at p50 and stable enough
   that p90/p50 spread does not hide the win.
7. Reduce broad sweeps with tooling: candidate enumeration, benchmark
   ingestion, compile-report capture, Pareto pruning, and shape-bucket winner
   selection should be automated rather than coordinated one run at a time by
   an LLM.
8. Emit catalog input JSON rows for accepted winners and retain rejected
   evidence outside the runtime catalog.

## Output

Produce:

```text
tuning run manifest
candidate result table
fusion-vs-unfused comparison table
accepted runtime catalog input
rejected-candidate evidence index
compile-report and manifest artifact paths
```

## Guardrails

- Do not encode tuning decisions directly in llama.cpp C++.
- Do not compare fused and unfused measurements from different target keys,
  shape keys, or benchmark methods.
- Do not run performance comparisons in parallel on the same GPU.
- Do not accept Loom-only correctness as the source of truth; require focused
  ggml CPU-reference validation before runtime catalog inclusion.
- Keep generated catalogs reproducible from source, configs, and evidence.
