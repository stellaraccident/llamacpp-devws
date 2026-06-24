# GGML HRX Kernel Workbench Design

Date: 2026-06-24

This document proposes a standalone GGML HRX kernel workbench for Loom kernel
development, verification, tuning, and catalog production. The workbench is
separate from llama.cpp integration on purpose. Kernel optimization needs a
small number of controlled variables, one metric boundary, and a reproducible
evidence chain. llama.cpp graph routing, backend fallback, model execution, and
runtime cache behavior are useful integration gates, but they are poor inner
loops for kernel development.

The initial project scaffold lives at:

```text
ggml-hrx-kernel-bench/
```

It is structured as a standalone Python project so it can later be promoted to
its own repository without depending on this workspace's `build/`, `cache/`, or
`sources/` layout.

## Goals

1. Provide a route-free kernel development loop for GGML operations and graph
   fragments implemented in Loom.
2. Keep Loom source, benchmark harnesses, fixture generation, NumPy goldens,
   compiler reports, timing evidence, and produced artifacts in one coherent
   evidence system.
3. Generate structured JSON ledger rows and compact catalog-candidate metadata
   that can feed a llama.cpp backend catalog assembly tool.
4. Make every result machine-specific and reproducible. A row for `gfx1151`
   should say it is `gfx1151`, carry tool/source hashes, and keep links to the
   artifacts that proved it.
5. Keep the llama.cpp backend deterministic. llama.cpp should consume curated
   catalog outputs; it should not tune, infer routes, generate kernels, or
   perform research loops at runtime.

## Non-Goals

- Do not reimplement llama.cpp graph routing inside the workbench.
- Do not require the workbench to live inside this workspace or use this
  workspace's build/cache directories.
- Do not make whole-model timing the first proof of a kernel. Model timing is
  a later integration gate.
- Do not promote synthetic-only correctness as production correctness when a
  captured or exact reference fixture is needed.
- Do not treat one GPU's timing as transferable to adjacent GPUs without
  evidence rows from those GPUs.

## Project Shape

The proposed standalone layout is:

```text
ggml-hrx-kernel-bench/
  pyproject.toml
  README.md
  src/ggml_hrx_kernel_bench/
    cli.py
    config.py
    fixtures.py
    ledger.py
    specs.py
    tools.py
  specs/
    mul_mat_q4_k_f32.json
  kernels/
    README.md
  schemas/
    kernel-spec.schema.json
```

The `kernels/` directory is where Loom sources will be copied or authored when
the workbench owns them. The current scaffold includes a spec example but does
not yet vendor the HRX2 corpus. Importing the corpus should be an explicit
follow-up step with source provenance recorded per kernel.

## Kernel Specs

The central unit is a kernel spec, not a llama.cpp route. A spec describes what
the workbench needs to construct a concrete experiment:

```json
{
  "id": "mul_mat_q4_k_f32",
  "op": "MUL_MAT",
  "source": "kernels/mul_mat_q4_k_f32.loom",
  "root_symbol": "@hrx2_mul_mat_q4_k_f32_static",
  "export_name": "hrx2_mul_mat_q4_k_f32_static",
  "types": {
    "src0": "Q4_K",
    "src1": "F32",
    "dst": "F32"
  },
  "parameters": {
    "k": {"config_key": "@hrx2.shape.k"},
    "rows": {"config_key": "@hrx2.shape.rows"},
    "cols": {"config_key": "@hrx2.shape.cols"},
    "workgroup_size": {"config_key": "@hrx2.tuning.workgroup_size"}
  }
}
```

Specs should grow conservatively. They should encode contracts, shape domains,
fixture families, tolerance policy, tuning dimensions, and catalog export
fields only when the harness has a real use for them.

## Evidence Ledger

Every action emits ledger rows. The full ledger is append-only JSONL and should
be easy to slice with `jq`, Python, or a future dashboard.

Required row groups:

- `identity`: kernel id, source path, root symbol, spec hash, source hash
- `machine`: target key, device name, ROCm/Loom versions when available
- `shape`: concrete parameters and estimated transfer/compute stats
- `config_bindings`: exact `--config` values used for JIT specialization
- `link`: selective materialization status and linked module path
- `compile`: compiler status, compile report, artifact manifest, target artifact
- `launch`: static workgroup count and workgroup size from Loom metadata
- `fixture`: generated input/output file paths and fixture strength
- `verification`: NumPy golden status, tolerance, and error summary
- `benchmark`: `iree-benchmark-loom` timing and profile evidence
- `sanitizers`: ASAN/TSAN or other guardrail rows
- `catalog_candidate`: compact row suitable for backend catalog assembly
- `acceptance`: explicit accepted/rejected state with reasons

The catalog assembler should consume accepted ledger rows and produce a smaller
backend-facing artifact. That step is intentionally lossy: it selects proven
rows and removes development-only data while retaining pointers to evidence.

## Correctness Fixtures

The workbench should support several fixture strengths:

| Strength | Use | Acceptance |
| --- | --- | --- |
| `pattern_no_reference` | Early timing smoke with non-zero data | Not enough for production correctness |
| `numpy_reference` | Synthetic data checked against NumPy | Minimum for most kernel rows |
| `captured_reference` | Real GGML tensors and captured expected output | Preferred for tricky quantized or fused routes |
| `backend_reference` | Output produced by a trusted llama.cpp/backend run | Useful when NumPy would duplicate too much backend-specific behavior |

For performance rows, all-zero inputs should be avoided unless the point of the
experiment is specifically zero behavior. Several GPUs execute zero-heavy data
with fewer transistor flips, and that can distort timing.

## Benchmark Slicing

The prototype learned an important lesson: whole-source benchmark compilation
pollutes the metric. The Q4_K hero kernel compiled and exposed launch metadata
correctly, but `iree-benchmark-loom` run mode initially failed because the
generated workbench included the entire HRX2 kernel source. The benchmark tool
compiled unrelated exported kernels before timing the selected check case, and
one unrelated Q8/Q4 kernel hit a topology-range diagnostic on `gfx1151`.

The immediate fix was to run `loom-link --mode=link --root=<candidate>` first
and then append the generated `check.case` and `check.benchmark` to that linked
module. This gave a focused workbench containing only the selected kernel and
its reachable dependencies.

That fix is operationally useful, but it should not be described as "benchmark
the compiler by hiding the rest of the source." The more intent-true framing is:

```text
kernel library + benchmark wrapper + concrete config
  -> benchmark module materialization
  -> correctness-gated timing
```

In other words, the artifact under test is a benchmark module, not a source
catalog. Selective linking is the materialization step that constructs that
module from a kernel library and a benchmark wrapper. `loom-link` explicitly
documents selective linking, roots, config binding, and dependency walking as a
normal flow. Used this way, linking is not a workaround; it is the boundary
between a reusable kernel library and a concrete benchmark executable.

The current tool split is still awkward because `iree-benchmark-loom` accepts a
single already-materialized `.loom` file. Two cleaner long-term designs are:

1. Add link inputs to `iree-benchmark-loom`: `--library`, `--root`,
   `--config`, and `--benchmark-wrapper` would let the benchmark tool perform
   the same materialization internally and record the link plan in its output.
2. Make benchmark wrapper files first-class sources. The wrapper imports or
   references a kernel library, declares only the check cases and benchmark
   rows being measured, and the harness materializes one benchmark module per
   candidate before invoking `iree-benchmark-loom`.

Option 2 is implementable today and aligns with the current tools. Option 1 is
the better ergonomics target if the Loom author agrees that benchmark-time
selective materialization is a common enough workflow.

What the workbench should avoid is passing an entire multi-kernel catalog to a
single benchmark row and then relying on benchmark selection to keep unrelated
compile failures out of the metric. Benchmark selection chooses which
`check.benchmark` rows run; it does not by itself define the source artifact
that should be compiled.

## Command Model

The Python project should expose a small set of commands:

```text
plan      expand specs and shape/tuning grids without compiling
link      materialize one candidate benchmark module
compile   compile one or more candidates and collect compiler evidence
verify    generate fixtures, run NumPy goldens, and check tolerances
run       run correctness-gated iree-benchmark-loom timing
tune      search a configured parameter space
catalog   emit compact accepted catalog candidate rows
```

The first scaffold implements only the shared config/spec/ledger pieces and
thin command skeletons. The next useful vertical slice is to promote the Q4_K
prototype into this command model.

## Catalog Output

The catalog assembly input should be compact and deterministic:

```json
{
  "schema": "ggml_hrx.catalog_candidate.v1",
  "kernel_id": "mul_mat_q4_k_f32",
  "op": "MUL_MAT",
  "types": {"src0": "Q4_K", "src1": "F32", "dst": "F32"},
  "target_key": "gfx1151",
  "shape_domain": {"k": [256, 32768], "rows": [1, 32768], "cols": [1, 1024]},
  "shape_guards": {"k_multiple_of": 256},
  "root_symbol": "@hrx2_mul_mat_q4_k_f32_static",
  "export_name": "hrx2_mul_mat_q4_k_f32_static",
  "workgroup_size": [256, 1, 1],
  "workgroup_count_expr": ["rows", "cols", 1],
  "config_bindings": {
    "@hrx2.shape.k": "shape.k",
    "@hrx2.shape.rows": "shape.rows",
    "@hrx2.shape.cols": "shape.cols"
  },
  "artifact": {"kind": "loom-bc", "path": "..."},
  "evidence": {"ledger": "...", "row_id": "..."}
}
```

llama.cpp should consume this data and apply only structural eligibility checks
plus deterministic catalog selection. If a row is missing, rejected, or fails
to load, the backend fallback path should say so explicitly.

## Initial Milestones

1. Promote the Q4_K ledger prototype into `ggml-hrx-kernel-bench` without
   workspace-specific defaults.
2. Add a real NumPy reference for Q4_K x F32 and mark rows as
   `numpy_reference` only after full-output checks pass.
3. Import the HRX2 Loom kernel corpus with provenance metadata.
4. Run the first accepted `gfx1151` compile+verify+run ledger row from the new
   project.
5. Emit one catalog candidate row from that ledger.
6. Add the next kernel family only after the first row is fully reproducible.

## Open Questions

- Should `iree-benchmark-loom` grow native selective-link inputs so benchmark
  module materialization is visible as part of the benchmark run?
- Should benchmark wrappers be authored as `.loom` files in the workbench, or
  generated from Python specs?
- Which fixture families require captured GGML tensors instead of NumPy
  references?
- What is the minimal catalog row needed for HRX3 backend ingestion without
  leaking research-only tuning metadata into runtime?
