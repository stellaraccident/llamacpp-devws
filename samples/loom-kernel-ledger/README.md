# Loom Q4_K Kernel Ledger Prototype

This sample prototypes the offline tuning ledger flow for the HRX2
`MUL_MAT` Q4_K x F32 Loom kernels without going through llama.cpp backend
routing. It writes generated artifacts to `cache/loom-kernel-ledger/`.

## Inputs

Defaults assume the standard HRX workspace layout:

- Loom tools: `build/hrx-install/bin/{loom-link,loom-compile,iree-benchmark-loom}`
- Loom C package: `build/hrx-install/lib64/cmake/loomc`
- ROCm: `rocm`
- Kernel source: `sources/llama.cpp-ref/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`
- Route metadata: `sources/llama.cpp-ref/ggml/src/ggml-hrx2/catalog/routes/mul_mat_q4_k_f32.json`

Only Q4_K x F32 `MUL_MAT` routes are included by default. The Q4_K x
Q8_1_X4 route in the same catalog is deliberately excluded.

## Modes

Plan-only ledger, with no Loom compilation or device execution:

```bash
python3 samples/loom-kernel-ledger/q4k_ledger.py --mode plan --limit 16
```

Compile a small candidate set and collect compile reports, manifests, HSACO
paths, and static launch metadata:

```bash
python3 samples/loom-kernel-ledger/q4k_ledger.py \
  --mode compile \
  --routes direct,wmma64 \
  --limit 2
```

The ledger defaults to this machine's `gfx1151` target. The HRX2 handoff source
still authors explicit `amdgpu.target<gfx1100>` records, so the default
`--source-target-policy rewrite` rewrites those target records to the selected
`--compile-target`. This keeps the ledger rows machine-specific while recording
the rewrite under `source_rewrite` and the target keys under `machine`.

Run one dispatch benchmark on the local AMDGPU using patterned fixture data:

```bash
python3 samples/loom-kernel-ledger/q4k_ledger.py \
  --mode run \
  --limit 1 \
  --iterations 1 \
  --warmup-iterations 0
```

Add sanitizer guardrails:

```bash
python3 samples/loom-kernel-ledger/q4k_ledger.py \
  --mode compile \
  --limit 1 \
  --sanitizers asan,tsan
```

`loom-compile` and `iree-benchmark-loom` sanitizer support may differ while
Loom is under active development. Unsupported sanitizer names are retained as
failed sanitizer rows instead of aborting the whole ledger.

## Fixtures

The default `--fixture pattern` mode writes per-candidate `.npy` files with:

- finite, non-zero Q4_K block headers,
- patterned Q4 quant bytes,
- patterned F32 RHS values,
- patterned destination initialization.

This avoids all-zero performance artifacts from reduced transistor switching.
It is labeled in the ledger as `correctness_strength=pattern_no_reference`.

`--fixture zero-smoke` remains available for quick launch debugging. It uses
zero inputs and an exact zero output expectation, and is labeled
`correctness_strength=zero_reference`.

## Outputs

Each run writes:

- `ledger.jsonl`: full machine-readable ledger, one row per candidate.
- `ledger_summary.csv`: compact summary for sorting and spreadsheet review.
- `candidates/<candidate-id>/...`: linked bytecode, compile reports,
  manifests, artifacts, generated workbenches, fixtures, and benchmark bundles.

Important ledger groups:

- `shape`: static problem size, data transfer bytes, and estimated math ops.
- `config_bindings`: exact Loom config bindings used for JIT specialization.
- `launch`: workgroup count and workgroup size, preferably from `loomc`
  static metadata and otherwise from route-derived heuristics.
- `compile`: compiler status, report paths, artifact paths, and summary stats.
- `benchmark`: fixture, timing, and runtime evidence when `--mode run` is used.
- `sanitizers`: ASAN/TSAN guardrail rows when requested.
- `acceptance`: accepted tier and explicit rejection reasons.
