# Backend Prior-Art Algorithm Ledger

Date: 2026-06-11

This is the permanent ledger for backend algorithms discovered while building
the HRX2/Loom catalog. Its purpose is to seed future agent searches with known
winning ideas from llama.cpp backends and the old HRX catalog, before writing
new Loom variants.

Treat entries here as priors, not truth. A prior is a shape of implementation
that won somewhere, on some device, with some constraints. HRX2 work should
translate these into Loom axes, benchmark them on the target, and either accept
or reject them with evidence.

## How To Use This Ledger

For every new standalone op or fusion:

1. Search the relevant backend files listed here and any op-specific entries.
2. Extract algorithmic ideas, not syntax: dataflow, work partitioning,
   vectorization, staging, reduction structure, launch shape, fusion boundary,
   and activation constraints.
3. Convert those ideas into Loom tuning axes or explicit rejected candidates.
4. Record the result back in this ledger when the idea is reusable across ops,
   and in the per-op report when it is local to one op.

## Backend Search Roots

| Backend | Path | Why it matters |
| --- | --- | --- |
| CUDA/HIPified | `sources/llama.cpp/ggml/src/ggml-cuda/` | Mature NVIDIA-first kernels that often encode launch-size and fusion policies worth testing on AMD. |
| Vulkan C++ dispatch | `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Backend-level fusion detection, route activation constraints, pipeline selection, and scheduling policy. |
| Vulkan shaders | `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/` | Handwritten GLSL kernels with explicit workgroup sizes, subgroup use, vectorization, and specialization constants. |
| OpenCL kernels | `sources/llama.cpp/ggml/src/ggml-opencl/kernels/` | Portable GPU kernels that often expose simple vector/tail and subgroup patterns. |
| Metal kernels | `sources/llama.cpp/ggml/src/ggml-metal/` | Apple-focused kernels; useful for threadgroup memory, SIMD-group reductions, and fusion/data-layout priors. |
| Old HRX catalog | `sources/llama.cpp/ggml/src/ggml-hrx/` and `docs/spike/` | Empirical AMD/W7900 results, including rejected routes and graph-level performance notes. Use cautiously because much of it came from sparse search. |
| HRX2 Loom seeds | `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/` | Current Loom syntax and HRX2 ABI examples. |

## Reusable Algorithm Priors

### One Workgroup Per Row For Row Reductions

Seen in:

- CUDA RMS_NORM: `sources/llama.cpp/ggml/src/ggml-cuda/norm.cu`
- Vulkan RMS_NORM: `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp`
- HRX2 RMS_NORM seed: `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rms_norm_f32.loom`

Pattern:

- Map one logical row to one workgroup.
- Each lane accumulates a strided partial over columns.
- Reduce partials within the workgroup.
- Broadcast the row scale/result and do a second strided pass.

Search axes:

- workgroup size: at least 64, 128, 256, 512, 1024 when target permits;
- vector width: scalar, 2-wide, 4-wide;
- tail policy: scalar-only, vector body plus scalar tail;
- exact static hidden width vs generic dynamic width;
- p90 stability for many-row prompt buckets.

Notes:

- On gfx1100 RMS_NORM, WG1024 was valid and won several shapes. Do not cap at
  WG512 unless a target-specific compile or occupancy reason proves it.
- Loom `kernel.workgroup.reduce<addf>` produced zero-spill code for RMS_NORM
  and should be the first implementation before hand-authoring LDS reductions.

### Vector Body Plus Scalar Tail

Seen in:

- OpenCL RMS_NORM: `sources/llama.cpp/ggml/src/ggml-opencl/kernels/rms_norm.cl`

Pattern:

- Process the main body as `float4` or another explicit vector width.
- Handle `ncols % vector_width` with a scalar cleanup path.
- Avoid treating odd or non-multiple hidden sizes as scalar-only.

Search axes:

- vector width 2 and 4 on RDNA3;
- cleanup ownership: lane 0 only vs distributed lane cleanup;
- vector-tail versus scalar-only for odd hidden sizes such as `1025`;
- code size and peak live registers, because tail paths can raise pressure.

Evidence:

- In the gfx1100 RMS_NORM sweep, `1025x60` selected vector-tail WG64/VW2 over
  scalar variants.

### Static Iteration Buckets / Exact Shape Specialization

Seen in:

- Vulkan RMS_NORM: `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp`

Pattern:

- Bucket hidden widths by the number of workgroup-strided loop iterations.
- Use static or specialization-time constants to encourage unrolling and remove
  dynamic loop/control overhead.

HRX2 translation:

- Prefer exact-shape or shape-bucket Loom specialization rather than hand-coded
  dispatch ladders.
- Keep bucket boundaries in metadata so runtime selection is data-driven.

Search axes:

- exact hidden width;
- hidden-width bucket;
- decode `nrows=1` versus prompt/multi-token rows;
- unroll factor when Loom exposes it directly.

### Simple Copy / Traffic Floor

Seen in this HRX2 tuning process rather than a backend file.

Pattern:

- For bandwidth/dispatch dominated ops, include a one-pass copy-like candidate
  with the same shape, launch dimensions, vector width, and fixture sizes.
- Use it as a sanity floor, not as the winner.

Search axes:

- scalar and vector copy floors;
- same workgroup sizes as the real op;
- same sample-count and repetition policy as the real op.

Notes:

- Ratios near `1.0x` indicate the standalone op is close to dispatch/traffic
  limits and fusion may be the only meaningful next step.
- Ratios below `1.0x` are possible with short runs and should be treated as
  measurement noise unless repeated.

### Fuse Row Reduction With Immediate Consumers

Seen in:

- CUDA RMS_NORM fusions: `sources/llama.cpp/ggml/src/ggml-cuda/norm.cu`
- Vulkan RMS_NORM fusions:
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp`
- Vulkan fusion selection:
  `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Pattern:

- If the reduction output is immediately consumed by elementwise multiply,
  rope, view, or set-rows, fuse after proving the standalone op is near its
  traffic floor.
- Keep the row-scale value live and avoid writing/reading the standalone
  normalized tensor when the graph allows it.

Known fusion candidates:

- `RMS_NORM + MUL`
- `RMS_NORM + MUL + ROPE`
- `RMS_NORM + MUL + ROPE + VIEW + SET_ROWS`

Activation constraints from Vulkan/CUDA priors:

- usually F32 RMS input/output;
- contiguous row assumptions;
- broadcast constraints when RMS_NORM is the multiply RHS;
- ROPE mode restrictions;
- shared-memory or per-row staging limits, often `ncols <= 1024` for fused
  ROPE paths.

HRX2 rule:

- Tune standalone and fusion separately. Accept a fusion only when it is
  measurably faster than the sum of the selected standalone parts on the same
  target and shape bucket.

### Split Versus Packed GLU/SWIGLU

Seen in:

- llama.cpp ggml graph exports for Qwen3 MoE and Llama 3.1 Q8:
  `cache/hrx2/phase1_0/route-slice-export-20260612-194632/*-ops.txt`
- HRX2 split SWIGLU route slice:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-15.md`

Pattern:

- llama.cpp can represent SWIGLU either as one packed source where the gate and
  activation halves are adjacent in `src0`, or as split `src0`/`src1` gate/up
  tensors with the same output shape.
- These are different ABI families even when the arithmetic is identical:
  packed uses two bindings (`src0`, `dst`), split uses three (`src0`, `src1`,
  `dst`).
- Split SWIGLU has a simple one-pass standalone kernel: linear element mapping,
  one x load, one gate load, `siluf`, multiply, and one store.

Search axes:

- packed versus split ABI;
- same-shape contiguous split tensors versus row/view variants;
- decode rows, narrow token rows, and MoE route rows;
- standalone split SWIGLU versus matmul+SWIGLU fusion once producer matmuls are
  represented.

Notes:

- Full model traces can keep a supported split SWIGLU node on CPU when adjacent
  matmuls are still CPU-owned. Focused ggml CPU-reference validation is required
  before rejecting or accepting the route based on model-level dispatch alone.

### Backend Fusion Selection Is Prior Art Too

Seen in:

- Vulkan graph fusion selection:
  `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- CUDA graph fusion selection:
  `sources/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`

Pattern:

- Dispatch code encodes activation boundaries that are as important as shader
  code: shape equality, stride requirements, broadcast legality, type
  restrictions, aliasing/overlap policy, and graph-neighbor constraints.

Search axes:

- graph pattern;
- edge ownership;
- source/destination overlap;
- contiguous rows versus fully contiguous;
- broadcast shape;
- output type;
- whether a fusion remains valid for decode, prefill, or both.

HRX2 rule:

- Record activation constraints in route metadata and test them with ggml unit
  coverage. Do not infer them solely from the kernel body.

### Q8_0/F32 Matvec Rows-Per-Workgroup

Seen in:

- OpenCL Q8_0 matvec/id kernels:
  `sources/llama.cpp/ggml/src/ggml-opencl/kernels/mul_mv_id_q8_0_f32.cl`
- Metal Q8_0 matvec constants and kernels:
  `sources/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h`
  and `sources/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- Old HRX Q8_0 probe kernels:
  `sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_add_bench.hip.cpp`

Pattern:

- Decode/narrow matvec paths do not have to map one output row to one
  workgroup. Prior backends group multiple rows per workgroup or subgroup.
- Each lane handles a strided slice of K for one or more rows, then reduces per
  row and writes the output for a column.
- Good row grouping depends on K, row count, column count, register pressure,
  and launch overhead.

Search axes:

- rows per workgroup: 1, 2, 4 first;
- workgroup size: 32, 64, 128, 256;
- decode `cols=1` versus narrow/prompt `cols>1`;
- exact shape specialization for K/rows/cols;
- register pressure and local/private memory as hard guards.

### Q8_0 With Packed Or Quantized RHS

Seen in:

- CUDA vector dot helpers:
  `sources/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh`
- Old HRX/Pyre Q8_0 prompt notes:
  `docs/spike/analysis/llamacpp_pyre_kernel_optimization_spike.md`
- Old HRX Q8_0 prompt routes:
  `sources/llama.cpp/ggml/src/ggml-hrx/`

Pattern:

- For prompt-like Q8_0/F32 matmul, scalar F32 RHS reads are often not the final
  form. Prior work used Q8_1 RHS packing, x4 dot-like grouping, and tile shapes
  such as `128x32`.
- RHS quantization can be approximate. Treat it as a separate algorithm family
  with explicit correctness policy and rollback semantics, not as a silent
  replacement for exact F32 RHS.

Search axes:

- exact F32 RHS scalar baseline;
- Q8_1 packed RHS variants;
- tile shapes such as `64x64` and `128x32`;
- prompt columns versus decode columns;
- standalone route versus `MUL_MAT + ADD` fusion.

### Q8_0 MUL_MAT + ADD Fusion

Seen in:

- Old HRX/Pyre spike:
  `docs/spike/analysis/llamacpp_pyre_kernel_optimization_spike.md`
- HRX integration notes:
  `docs/spike/reports/hrx-llamacpp-integration.md`

Pattern:

- Q8_0 matvec output is commonly consumed by an immediate F32 bias/add path.
- Fusing can avoid writing and rereading the standalone output and was a major
  prompt-path win in the old backend.

HRX2 rule:

- Finish and route standalone Q8_0/F32 first.
- Then tune fused ADD against the sum of selected standalone Q8_0/F32 and ADD
  routes for the same target, shape, benchmark method, and fixtures.

### Q8_0/F32 Exact HIP Refutation Baseline

Seen in:

- `tools/hrx2_q8_0_f32_refute.py`
- `docs/loom/q8-0-f32-performance-refutation-gfx1100.md`

Pattern:

- For exact F32 RHS standalone matvec/mul_mat, a native HIP row/block schedule
  that loads `float4` RHS chunks and four Q8 quants per lane step is a strong
  baseline.
- On gfx1100 it beat the initial Loom scalar route by roughly 4-5x on the
  phase0 Q8 shapes while preserving exact F32 RHS semantics.
- One row per workgroup won this exact scalar family in the tested shapes;
  rows-per-workgroup 2 and 4 remain search axes but did not win here.

Search axes:

- workgroup size: 32, 64, 128, 256;
- rows per workgroup: 1, 2, 4;
- Q8 block ownership and `float4` RHS load pattern;
- scale-load amortization per Q8 block;
- exact F32 RHS first, then separate packed Q8_1/fusion families.

### CONT / Strided Copy To Contiguous

Seen in:

- ggml CPU reference behavior through `test-backend-ops` CONT cases.
- HRX2 phase 1 source:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/cont_f32.loom`
- Vulkan and other GPU backends route tensor copies through copy-like kernels
  once layout constraints are known.

Pattern:

- Treat CONT as a real data-movement kernel only when the source layout is
  row-contiguous enough to prove simple source addressing and the destination
  is fully contiguous.
- Preserve source strides as specialization facts. In model traces, the same
  logical `128`-column CONT appears with different `ne1/ne2` and stride
  patterns depending on attention head layout.
- Keep copy kernels narrow and correct first. They may not improve standalone
  wall time at tiny shapes, but they remove CPU fallback and become necessary
  prerequisites for attention fusions.

Search axes:

- scalar versus vectorized copy width;
- linear-element mapping versus row/workgroup mapping;
- workgroup size 128, 256, 512;
- exact stride-shape specialization;
- fusion with adjacent ROPE, SCALE, softmax, or attention layout transforms.

### Standalone GLU / SWIGLU Activation

Seen in:

- CPU GLU implementation:
  `sources/llama.cpp/ggml/src/ggml-cpu/ops.cpp`
- Vulkan GLU dispatch and shaders:
  `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`
- OpenCL GLU path:
  `sources/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp`
- HRX1 fused SWIGLU matmul families:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_bf16_swiglu.hip.cpp`,
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_id_q4_k_swiglu.hip.cpp`

Pattern:

- Plain `GGML_OP_GLU` is an elementwise activation over packed or split gate
  tensors. The common non-split SWIGLU layout is one source row shaped
  `[x..., gate...]` and one destination row shaped `[out...]`.
- Standalone backends generally use a one-pass elementwise kernel with one or
  more x/gate loads, activation, multiply, and store.
- The larger wins in old HRX are from fusing the activation with the two
  producer matmuls, especially MoE/grouped prompt routes. Treat those fused
  routes as separate candidates after the standalone operation has coverage.

Search axes:

- packed contiguous, split, swapped, and OAI/limit variants as separate route
  support domains;
- scalar element-per-lane baseline versus explicit vectorized load/store
  groupings;
- row mapping for decode (`nrows=1`), narrow multi-token, and prefill buckets;
- fusion with `gate` and `up` matmuls, measured against the selected unfused
  route sum rather than assumed from the standalone activation cost.

HRX2 evidence:

- `swiglu_f32_n8192_r1_wg256` and `swiglu_f32_n8192_r16_wg256` are accepted
  focused coverage routes for the Phi-4 packed, non-split, non-swapped fallback
  shapes. The source is target-neutral and has clean compile reports: zero
  spills, zero private/local memory, 3 global-memory instructions, and 128-156
  emitted code bytes.

### GET_ROWS / Indexed Row Gather

Seen in:

- ggml CPU reference and `test-backend-ops` GET_ROWS cases.
- Old HRX and Vulkan-style gather/copy paths that make row index, source
  stride, and destination stride legality explicit.
- Rejected HRX2 phase 1 candidate:
  `cache/hrx2/phase1_0/rejected-get-rows/get_rows_f32.loom`

Pattern:

- Map each destination element to `(column, selected-row, outer dims)`, load a
  row index, then copy from the indexed source row to the destination layout.
- Correctness depends on the exact ggml index type and stride semantics; do not
  silently narrow or assume dense destination layout unless route metadata says
  so.
- Dynamic indexed addressing is a likely stress case for Loom address proofs,
  so keep a small standalone validation source before admitting production
  routes.

Search axes:

- index type path: i32 direct, i64 direct, or documented low-lane temporary;
- row width buckets and vector copy width;
- one-dimensional flat copy versus row-major workgroup mapping;
- destination contiguous versus strided output;
- future quantized source row gathers such as `q6_K` embedding rows.

## Case Study: RMS_NORM F32 Contiguous

Per-op report:

- `docs/loom/rms-norm-standalone-done-gfx1100.md`

Backend priors harvested:

- CUDA scalar row reducer with WG256 below 1024 columns and WG1024 otherwise.
- Vulkan WG512 row reducer with static iteration buckets.
- OpenCL `float4` vector body with scalar tail cleanup.
- CUDA/Vulkan fusions around RMS_NORM immediate consumers.

HRX2 axes implemented:

- scalar, vector, vector-tail families;
- workgroup sizes 64, 128, 256, 512, 1024;
- vector widths 1, 2, 4;
- exact static shapes;
- copy-floor traffic references.

Accepted gfx1100 standalone winners:

| Shape | Winner |
| --- | --- |
| `64x60` | vector WG1024 VW4 |
| `1025x60` | vector-tail WG64 VW2 |
| `4096x1` | vector WG512 VW4 |
| `4096x32` | scalar WG1024 |
| `512x32` | vector WG512 VW2 |
| `8192x1` | vector WG1024 VW4 |
| `8192x32` | vector WG512 VW4 |

Reusable conclusions:

- Include WG1024 in reduction searches when target facts allow it.
- Include vector-tail for odd hidden sizes.
- Use copy-floor measurements before deciding a standalone op needs more
  invention; if it is already close to the floor, prioritize fusion trials.
- Use p90 stability and compile report facts as promotion guards, not p50 alone.

## Open Items For Future Entries

- Add matvec/mul_mat priors from Vulkan MMQ/cooperative matrix, CUDA MMV/MMQ,
  and old HRX Q4/Q5/Q6 route notes.
- Add softmax/attention priors around subgroup reductions, logits staging,
  vectorized KV loads, and decode/prefill split.
- Add MoE routing/top-k priors from the old HRX spike and Vulkan fusion code.
- Add rope/set-rows priors, especially shared staging limits and destination
  type constraints.
