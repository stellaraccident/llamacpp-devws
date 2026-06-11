---
name: hrx2-loom-family-authoring
description: Use this when authoring a Loom kernel family for the llama.cpp HRX2 backend, including ABI definition, provider contracts, config axes, correctness cases, benchmarks, compile reports, and HRX manifest validation.
metadata:
  short-description: Author HRX2 Loom kernel families
---

# HRX2 Loom Family Authoring

Use this skill when creating or revising one Loom family for HRX2.

## Required Context

Read first:

- `docs/loom/llamacpp-integration-v1.md`
- `docs/loom/llamacpp-hrx-authoring.md`
- `docs/loom/llamacpp-hrx-stella-questions-2026-06-11.md` for current Loom target/provider/report details

## Workflow

1. Define the family contract: ggml op or fusion, ABI, bindings, constants,
   shape domain, numeric policy, and target key.
2. Write a semantic baseline first. Add provider contracts with
   `func.apply`/`func.template` only where schedule choices need tuning.
3. Express schedule intent explicitly: vector load width, packed dot form,
   subgroup/wave policy, tile ownership, LDS staging, and layout assumptions.
4. Add `check.case` coverage for representative decode, prefill, and boundary
   shapes. Add `check.benchmark` rows tied to those cases. Use deterministic
   random or representative synthetic inputs; do not rely on all-zero/all-one
   benchmark patterns.
5. Compile one root/candidate at a time. Request compile reports and artifact
   manifests.
6. Validate the manifest fields needed by HRX2: binding count, parameter count,
   constant bytes, workgroup size, subgroup/wavefront size, and SGPR/VGPR facts.
7. Run focused ggml CPU-reference correctness before accepting a route into the
   HRX2 runtime catalog.
8. Record accepted and rejected provider/config variants with JSON/JSONL
   evidence.

## Output

For each family, produce:

```text
family source or bytecode package
ABI note
candidate axes
correctness results
benchmark results
compile report
artifact manifest
HRX2 catalog record draft
```

## Guardrails

- Do not hand-pick a winner without benchmark evidence.
- Do not rely on the compiler to infer wide packed loads that source can spell.
- Do not compile multi-root artifacts for initial tuning unless attribution is
  explicitly required and documented.
- Treat the production catalog as embedded JSON parsed by llama.cpp, not as
  hand-authored C++ route policy.
- Keep feedback for the Loom author in the relevant `docs/loom/` feedback note.
