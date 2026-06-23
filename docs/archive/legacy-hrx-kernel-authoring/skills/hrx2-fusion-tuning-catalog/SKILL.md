---
name: hrx2-fusion-tuning-catalog
description: Use this when tuning HRX2 Loom kernels or fusions offline, running broad tool-driven sweeps, reducing compile-report and latency evidence, comparing fused routes against the sum of unfused parts, selecting measured winners, and emitting generated catalog inputs for llama.cpp.
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
- `docs/loom/rms-norm-standalone-done-gfx1100.md` for the completed standalone-op regime
- the family notes for every candidate being tuned

## Workflow

1. Build a candidate matrix from prior-ledger rows and declared
   family/provider/config axes. Include algorithm families that spell different
   machine schedules, not just numeric constants: vector width, load width,
   tile ownership, scale reuse, packed dot form, LDS staging, reduction
   strategy, and tail policy. Also include natural shape regimes: decode,
   narrow multi-token, prompt buckets, odd/tail, power-of-two, and common model
   hidden sizes.
   Each row must say which prior or analytical schedule it follows. For
   adjacent probes without strong direct evidence, name the pivot axis, sweep
   bounds, and expected signal. Useful examples are BN/BK brackets around one
   packed-MMQ dataflow, vector-width brackets around one packed layout, or
   unroll-depth brackets around one static loop structure.
2. Tune one root/candidate per artifact by default, keyed by source hash, root,
   config, target key, target variant if present, pass program, ABI, and Loom version.
   Keep source portability separate from route applicability: portable Loom
   sources should stay target-neutral, while measured winners are selected by
   catalog `target_key` and target-specific evidence.
3. Run fast Loom correctness before benchmarking. Reject failed candidates immediately.
4. Benchmark with `iree-benchmark-loom --measure=dispatch_complete` or the
   current Loom benchmark flow. Use stable repetitions and keep raw JSON/JSONL evidence.
5. Reduce broad sweeps with tooling: candidate enumeration, benchmark
   ingestion, compile-report capture, Pareto pruning, and shape-bucket winner
   selection should be automated rather than coordinated one run at a time by
   an LLM.
6. Gate winners by latency plus compile-report facts: spills, unexpected
   private/local memory, code size, register pressure, and p90 stability.
   Inspect target listings for WYSIWYG agreement with the algorithm family:
   expected load widths, dot/vector forms, scale reuse, and LDS traffic must be
   present before calling a candidate representative of that family.
7. Reduce winners into route buckets, not one-off exact shapes, unless the
   exact shape is itself the production bucket. Preserve raw exact-shape rows in
   evidence JSON/JSONL; emit catalog JSON with shape domains, guards, config
   sources, target key, and evidence phase.
8. Reject speculative pivots before integration unless the focused sweep shows
   a material bucket-level win or a useful refutation. Full model runs are the
   acceptance path for measured winners, not the first filter for blind
   schedule exploration.
9. Materialize accepted standalone routes into the HRX2 catalog, then run
   focused ggml CPU-reference tests. Loom-only correctness is not sufficient.
10. For each fusion, benchmark the fused candidate and the selected measured
   unfused parts under the same target, shape, benchmark method, and data fixtures.
11. Accept a fusion only when it is at least 3% faster at p50 and stable enough
   that p90/p50 spread does not hide the win.
12. Emit catalog input JSON rows for accepted winners and retain rejected
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
ggml CPU-reference validation trace
```

## Guardrails

- Do not encode tuning decisions directly in llama.cpp C++.
- Do not add kernel-specific tuner scripts to llama.cpp. Production code gets
  generic catalog/runtime/embed/validation tools; family-specific exploration
  scripts belong in the workspace or cache.
- Do not populate the candidate matrix with blind guesses. Every candidate must
  either follow a documented prior schedule or bracket one axis around such a
  schedule with a stated expected signal.
- Do not promote a bracketed schedule pivot into the production catalog until a
  focused kernel/backend-op sweep has selected it.
- Do not make broad sweeps over source that does not encode the intended
  schedule. Loom generally emits what the source says; a scalar loop is not a
  request for packed vector loads.
- Do not compare fused and unfused measurements from different target keys,
  shape keys, or benchmark methods.
- Do not run performance comparisons in parallel on the same GPU.
- Do not accept a candidate with fixed-pattern benchmark data unless that data
  is representative for the op.
- Do not accept Loom-only correctness as the source of truth; require focused
  ggml CPU-reference validation before runtime catalog inclusion.
- Do not copy a target's winner rows to another target. Add target-specific
  rows only after target-specific tuning evidence exists.
- Do not encode a source-level target attribute merely because a route was
  measured on that target. Use target-specific source files only for
  target-specific primitives such as chip-specific WMMA layouts or low/rocasm
  snippets.
- Keep generated catalogs reproducible from source, configs, and evidence.
