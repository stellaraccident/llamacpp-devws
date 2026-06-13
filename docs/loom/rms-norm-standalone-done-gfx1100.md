# RMS_NORM Standalone Tuning Report: gfx1100

Date: 2026-06-11

## Scope

This pass tuned standalone F32 contiguous `RMS_NORM` kernels for the W7900
`gfx1100` target using Loom's standalone benchmark flow. It is the exemplar
process for future HRX2 op work: harvest prior art, encode algorithmic variants
as Loom tuning axes, benchmark exact shapes with randomized fixtures, reduce by
latency plus compile report facts, and validate the current llama.cpp route
against ggml CPU reference tests.

This does not yet materialize the winners into the HRX2 runtime selector. The
current in-tree HRX2 `RMS_NORM` route is still the generic dynamic route keyed
by target only. The standalone evidence below is the input for the next
runtime-materialization step.

## Prior-Art Harvest

Permanent reusable entries from this pass were added to
`docs/loom/backend-prior-art-algorithms.md`.

Checked implementations:

- CUDA/HIPified backend: `sources/llama.cpp/ggml/src/ggml-cuda/norm.cu`
- Vulkan backend: `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp`
- Vulkan partial/fusion path:
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_partials.comp`
- OpenCL backend: `sources/llama.cpp/ggml/src/ggml-opencl/kernels/rms_norm.cl`

What mattered:

- CUDA uses a scalar-load one-workgroup-per-row reducer, with WG256 below 1024
  columns and WG1024 otherwise.
- Vulkan uses WG512 and specializes by unrolled iteration-count buckets. It also
  carries the important fusion priors: `RMS_NORM+MUL`,
  `RMS_NORM+MUL+ROPE`, and `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS`.
- OpenCL uses `float4` vector loads for the main body plus scalar cleanup for
  tails. This directly changed the search: odd hidden sizes must not be treated
  as scalar-only.
- The Vulkan partial path is a fusion/support mechanism, not the right
  standalone single-row baseline for this pass.

## Tooling Added

`tools/hrx2_rms_norm_tune.py` now generates standalone Loom candidates and
benchmarks them with:

- exact static shape per candidate;
- deterministic non-constant F32 NPY fixtures and CPU expected outputs;
- scalar, vector, and vector-tail RMS_NORM families;
- configurable workgroup size and vector width;
- optional copy-floor kernels for dispatch plus read/write traffic reference;
- JSONL results plus reduced `summary.json` and `summary.md`;
- compile report capture: instruction count, code bytes, spills, local/private
  memory, and peak live register pressure.

The vector-tail family was added after reading OpenCL. It vectorizes the full
`ncols / vector_width` body and uses scalar cleanup for the remainder, so
`1025`-wide shapes participate in vector search.

## Commands

Primary default-cache sweep:

```bash
./tools/hrx2_rms_norm_tune.py \
  --run-id gfx1100-rms-norm-done-20260611-default-cache \
  --shapes 4096x1,4096x32,512x32,1024x1,3584x1,8192x1,8192x32 \
  --workgroup-sizes 64,128,256,512 \
  --vector-widths 1,2,4 \
  --cache-policies default \
  --iterations 10 \
  --warmup-iterations 3 \
  --repetitions 3 \
  --timeout 15
```

Prior-art sweep with vector tails, WG1024, and copy floor:

```bash
./tools/hrx2_rms_norm_tune.py \
  --run-id gfx1100-rms-norm-prior-art-20260611 \
  --shapes 64x60,1025x60,4096x1,4096x32,512x32,8192x1,8192x32 \
  --workgroup-sizes 64,128,256,512,1024 \
  --vector-widths 1,2,4 \
  --cache-policies default \
  --iterations 8 \
  --warmup-iterations 3 \
  --repetitions 2 \
  --timeout 15 \
  --include-vector-tail \
  --include-copy-floor
```

ggml CPU-reference validation for the current HRX2 runtime route:

```bash
GGML_HRX2_TRACE_JSONL=cache/hrx2/rms_norm_tune/gfx1100-rms-norm-done-20260611-default-cache/ggml_rms_trace.jsonl \
  build/llama-hrx2/bin/test-backend-ops test -b HRX20 -o RMS_NORM --output csv
```

## Results

All 200 prior-art candidates compiled, passed Loom check expectations, and
benchmarked successfully. All accepted candidates had zero spills.

| Shape | Winner | p50 ns | p90 ns | Inst | Code bytes | Peak live | Best copy floor | RMS/floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `64x60` | vector WG1024 VW4 | 21245.5 | 26045.5 | 138 | 624 | 13 | vector WG64 VW2 at 20826.0 | 1.020x |
| `1025x60` | vector-tail WG64 VW2 | 21031.0 | 39976.0 | 160 | 772 | 24 | scalar WG128 at 21330.5 | 0.986x |
| `4096x1` | vector WG512 VW4 | 21406.0 | 22145.5 | 132 | 600 | 13 | vector WG1024 VW2 at 21301.0 | 1.005x |
| `4096x32` | scalar WG1024 | 22105.5 | 30001.0 | 120 | 588 | 12 | vector WG256 VW2 at 20306.0 | 1.089x |
| `512x32` | vector WG512 VW2 | 20895.5 | 22186.0 | 125 | 592 | 12 | vector WG64 VW4 at 21226.0 | 0.984x |
| `8192x1` | vector WG1024 VW4 | 21855.0 | 27981.0 | 137 | 624 | 13 | vector WG1024 VW4 at 21535.5 | 1.015x |
| `8192x32` | vector WG512 VW4 | 23225.0 | 39501.0 | 135 | 616 | 13 | scalar WG1024 at 22231.0 | 1.045x |

The copy-floor ratio is a dispatch/traffic sanity check, not a physical
roofline proof. Ratios slightly below `1.0x` are benchmark noise: the RMS_NORM
and copy kernels are separate candidates, each with two repetitions. The useful
takeaway is that the best standalone RMS_NORM variants are very close to the
dispatch plus read/write floor for these small-to-medium shapes.

## Interpretation

The standalone algorithm is effectively done for the shapes tested:

- One workgroup per row is still the right standalone structure.
- Explicit vector width matters. VW4 wins many power-of-two hidden sizes, but
  VW2 wins important small/odd cases.
- WG1024 must be in the search. It wins `64x60`, `4096x32`, and `8192x1`, and
  is competitive elsewhere. The earlier WG512 cap was an artificial search
  limitation, not a target limitation.
- Vector-tail is required. `1025x60` picked vector-tail WG64/VW2 over scalar,
  validating the OpenCL prior.
- Compile reports did not show spills or local/private memory pressure as the
  limiting factor. The relevant secondary signals are code size and peak live
  values for vector-tail, especially the peak-live jump to 24.
- The p90 distribution is noisy for several multi-row cases. For runtime
  promotion, winner selection should use more repetitions and a p90 guard, not
  p50 alone.

## Rejected / Deferred Axes

- `cache_temporal = non_temporal` on global vector loads is invalid for the
  current AMDGPU target lowering. Loom reports `ERR_AMDGPU_024`:
  device/non-temporal global memory cache policy is not encodable by
  `amdgpu.rdna3.core`.
- Manual shared-memory reduction was not implemented in this pass. Loom's
  `kernel.workgroup.reduce<addf>` produced zero-spill code and was close enough
  to copy-floor that LDS hand-authoring is not justified before runtime
  materialization evidence.
- Vulkan-style unrolled iteration-count buckets were tested indirectly through
  exact-shape static loop bounds, not as a separate manual `num_iters` dispatch
  ladder. Exact-shape Loom specialization is the cleaner HRX2 equivalent.
- Fusions are intentionally out of scope for this op pass. The harvested fusion
  priors should seed separate `RMS_NORM+MUL` and `RMS_NORM+MUL+ROPE` fusion
  candidates after standalone materialization.

## Runtime Validation

`test-backend-ops` passed for the current HRX2 runtime route. The trace shows a
single generic provider:

```text
route_id=rms_norm_f32_contiguous
cache_key=rms_norm_f32_contiguous|target=gfx1100
workgroup_size_x=512
```

It dispatched `ncols=64,nrows=60` and `ncols=1025,nrows=60` cases and matched
ggml CPU reference behavior for supported non-inplace contiguous F32 RMS_NORM.

This validates the existing runtime route, not the standalone exact-shape
winners. The next implementation step is to materialize the winner table into
HRX2 route metadata or JIT config selection and rerun the same ggml unit tests.

## Future Op Regime

For each standalone op or fusion candidate:

1. Harvest priors from CUDA/HIP, Vulkan, Metal/OpenCL, and any existing HRX
   catalog implementation. Record both algorithmic ideas and activation
   boundaries. Update `docs/loom/backend-prior-art-algorithms.md` for reusable
   patterns.
2. Convert priors into Loom axes before tuning: workgroup size, vector width,
   tail policy, unroll/static shape bucket, memory policy, and fusion-specific
   dataflow choices.
3. Generate randomized synthetic fixtures and CPU expected data for Loom checks.
   Avoid fixed all-zero/all-one benchmark data.
4. Benchmark exact static shapes first, including decode width `1`, narrow
   multi-token/prompt buckets, odd hidden sizes, and popular model hidden
   widths.
5. Include a copy or simple traffic floor when the op is bandwidth/dispatch
   dominated.
6. Reduce by p50 with p90 and compile-report guards: reject spills, unexpected
   local/private memory, pathological code size, or unstable p90 unless the
   speedup is large enough to justify deeper profiling.
7. Only after standalone winners are stable, materialize them into HRX2
   runtime/catalog selection.
8. Accept with ggml unit tests against CPU reference. Then evaluate fusion
   candidates separately against the sum of their accepted standalone parts.
