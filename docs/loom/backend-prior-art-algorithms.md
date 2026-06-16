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
3. When the performance gap is broad, verify the prior mechanically where
   possible. Build or locate the generated artifact, disassemble SPIR-V/HSACO
   or inspect compile reports/resource summaries, and compare the emitted
   schedule against the current Loom candidate before writing a replacement.
4. Convert those ideas into Loom tuning axes or explicit rejected candidates.
5. Record the result back in this ledger when the idea is reusable across ops,
   and in the per-op report when it is local to one op.

For packed matmul and attention work, a useful prior extraction table must name
the tile dimensions, workgroup/wave size, lane-to-output mapping, vector or
packed load widths, LDS/shared-memory layout, dot primitive, reduction/writeback
policy, route activation constraints, and any ISA/resource facts. If HRX2 is
far behind Vulkan/CUDA/HRX1, assume the first failure mode is that the Loom
kernel is spelling the wrong schedule class until this table proves otherwise.

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

### NORMAL Versus NEOX ROPE Pair Layout

Seen in:

- llama.cpp graph exports for Mistral Small 3.2:
  `cache/hrx2/phase1_0/route-slice-45-rope-mistral-export/mistral-rope-focused-ops.txt`
- HRX2 NORMAL no-frequency route slice:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-45.md`
- Existing HRX2 NEOX and NORMAL frequency-source ROPE sources:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rope_neox_f32.loom`,
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/rope_neox_f32_freq.loom`

Pattern:

- NEOX pairs the first and second halves of the head dimension:
  `(pair, pair + ncols/2)`.
- NORMAL pairs adjacent elements inside the rotated prefix:
  `(2*pair, 2*pair + 1)`.
- Frequency-source and no-frequency variants have different ABIs. A no-freq
  route uses three bindings (`src0`, positions, `dst`); a frequency-source
  route uses four bindings and loads the frequency factor buffer.

HRX2 translation:

- Keep separate Loom roots for layout/ABI differences instead of hiding pair
  layout behind route heuristics.
- Keep no-frequency NORMAL source target-neutral unless a future route uses
  target-specific transcendental or table primitives.
- For no-frequency ROPE, bind `n_dims` in config even when the current C++
  predicate requires `n_dims == ncols`; this keeps the source contract explicit
  if partial-rotation no-frequency routes are admitted later.

Notes:

- On route slice 45, Mistral h8/h32 `ncols=128`, `n_dims=128`, `ntokens=1..64`
  passed focused ggml CPU-reference validation with zero spills/private/local
  memory and moved 2880 full-basket graph nodes from CPU to HRX2.
- The Llama 3.1 NORMAL frequency-source h32/p64 row was fixed in route slice
  48. For strict CPU parity, use the CPU-like theta recurrence: compute a
  single `theta_scale = powf(freq_base, -2/n_dims)` equivalent, then multiply
  `theta` forward per pair before applying the frequency-factor buffer. The
  independent per-pair `exp(log(base) * exponent)` spelling was close but
  failed the p64 ggml CPU-reference tolerance.

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

### Row-Stride And Column-Broadcast Pointwise

Seen in:

- Qwen3 MoE graph exports:
  `cache/hrx2/phase1_0/route-slice-26-20260612-201843/phase1_route_slice_26_ops.txt`
- HRX2 pointwise route slice:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-26.md`

Pattern:

- MoE weighting uses a dense output such as `dst=[2048,8,tokens]` with
  `src0=[2048,8,tokens]` and RHS weights `src1=[1,8,tokens]`.
- The RHS is a per-row scalar, not a same-width row. The kernel must index it as
  `row * src1_row_stride + (col % src1_ncols)`, with `src1_ncols=1`.
- MoE accumulation can use padded row-strided F32 views as inputs while writing
  a contiguous destination. Source row stride therefore belongs in the kernel
  ABI/config, not as an implicit contiguous-linear assumption.

Search axes:

- same-shape contiguous;
- same-shape padded row stride;
- RHS row broadcast;
- RHS column/scalar broadcast;
- vectorized contiguous source loads where the row stride equals `ncols`;
- exact source/destination alias and view constraints from scheduler traces.

HRX2 rule:

- Keep the support predicate narrow: only flattenable dense F32 rows with a
  constant row stride should use this simple pointwise family. Broader ggml
  broadcasting and permuted layouts need their own validated families.

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
- HRX2 compact dense F32 route slice:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-46.md`

Pattern:

- Map each destination element to `(column, selected-row, outer dims)`, load a
  row index, then copy from the indexed source row to the destination layout.
- Correctness depends on the exact ggml index type and stride semantics; do not
  silently narrow or assume dense destination layout unless route metadata says
  so.
- Dynamic indexed addressing is a likely stress case for Loom address proofs,
  so keep a small standalone validation source before admitting production
  routes.
- For compact dense F32 rows, prefer a 2D dense-view spelling
  `src0[row, col] -> dst[row, col]` over manually linearizing
  `row * stride + col`. On the current Stella branch, the 2D view spelling
  avoided AMDGPU address-width target-low failures hit by the flat form.

Search axes:

- index type path: i32 direct, i64 direct, or documented low-lane temporary;
- row width buckets and vector copy width;
- one-dimensional flat copy versus row-major workgroup mapping;
- destination contiguous versus strided output;
- future quantized source row gathers such as `q6_K` embedding rows.

HRX2 evidence:

- Route slice 28 accepted a deliberately narrow MoE weight gather,
  `get_rows_moe_weights_f32_ne128_k8_t{1,16,64}_wg256`, for
  `src0=[1,128,tokens]`, `idx=[8,tokens]`, and `dst=[1,8,tokens]`.
- The kernel keeps `src0`, `idx`, and `dst` token strides as JIT config facts
  because traced `idx` views have padded token stride (`128` elements) while
  `dst` is compact over eight selected experts.
- This narrow route does not rehabilitate the rejected generic GET_ROWS path.
  Generic row gather still needs separate address-proof and layout work before
  it can be admitted broadly.
- Route slice 46 accepted compact dense F32 routes for hidden widths
  `2048`, `3072`, `3584`, `4096`, `5120`, and `5376` with row buckets
  `1..64`. Focused graph-op validation passed, full basket validation passed
  33/33, and CPU compute fallbacks dropped by `792`.
- Route slice 47 accepted quantized embedding F32 gathers for `q4_K`, `q5_K`,
  `q6_K`, and `q8_0` hidden-width buckets used by the basket. These are
  separate from dense F32 gather because they decode the source block during the
  gather. Full basket validation passed 33/33, CPU compute fallbacks dropped by
  `396`, and `cpu_assigned_but_hrx_supported` became empty.
- Embedding gather placement may require backend scheduler cooperation in
  addition to route support. In slice 47, quantized gathers were
  `supported_by=[HRX20,CPU]` but CPU-assigned until HRX2 provided a conservative
  `offload_op` hook for `GET_ROWS`.

### MoE ARGSORT / Top-K Support

Seen in:

- Qwen3 and Qwen3-Coder basket traces around `ffn_moe_argsort`.
- Vulkan/Metal/OpenCL style small-row sort/top-k kernels that keep one row in
  one workgroup and use either local memory or rank/count comparisons.

Pattern:

- Current MoE graphs sort `128` expert probabilities descending and consume the
  first eight indices.
- A full bitonic sort is the natural GPU prior: one workgroup owns one token
  row, stages values/indices in workgroup memory, and emits sorted indices.
- A rank-count implementation is simpler and avoids workgroup scratch: each
  lane owns one expert, counts how many values precede it, and stores its lane
  index at that rank. For `ncols=128` it is acceptable as phase-one coverage,
  but it is `O(n^2)` and should not be treated as the final general sort path.

Search axes:

- full sort versus top-k-only selection;
- bitonic/LDS versus rank-count no-LDS fallback;
- tie policy matching ggml CPU reference;
- row counts for decode, narrow, and prefill64;
- later fusion with softmax/top-k/weight normalization when standalone parts
  are represented.

HRX2 evidence:

- Route slice 28 accepted `argsort_f32_i32_n128_r{1,16,64}_desc_wg128` using
  the rank-count no-LDS fallback. Focused ggml CPU-reference validation passed
  and compile reports showed zero spills, zero private memory, and zero local
  memory.
- Bitonic/LDS candidates compiled but faulted under HRX2 raw dispatch, and the
  dynamic bitonic source also exposed an `index.div`/`index.shrui`
  address-width lowering diagnostic. Keep those as Loom/tooling follow-up
  rather than silently widening the rank-count route.

### MoE Weight Normalization Micro-Chain

Seen in:

- Qwen3 and Qwen3-Coder basket traces around
  `ffn_moe_weights_sum`, `ffn_moe_weights_sum_clamped`, and
  `ffn_moe_weights_norm`.
- Metal/OpenCL style small-row reductions and pointwise binary/unary kernels.

Pattern:

- `SUM_ROWS` reduces eight selected expert weights into one scalar per token
  row. A simple one-workgroup-per-row WG32 reduction is enough for the current
  basket shapes.
- `CLAMP` applies scalar min/max constants to the one-column sum tensor.
- `DIV` broadcasts the clamped one-column sum back across the eight selected
  weights.

HRX2 evidence:

- Route slice 27 accepted `sum_rows_f32_n8_r{1,16,64}_wg32`,
  `clamp_f32_n1_r{1,16,64}_contiguous_wg256`, and
  `div_f32_n8_r{1,16,64}_rhscolbroadcast_wg256`.
- Focused ggml CPU-reference validation passed 15/15 rows including these
  shapes. Compile reports showed zero spills and zero private memory.
- Full-basket traces still assign the chain to CPU because upstream `ARGSORT`
  and `GET_ROWS` are CPU-only. Treat this as a placement dependency: solve
  top-k/gather before spending more effort on the downstream normalization
  kernels.

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

### ROPE NEOX Pair Rotation

Seen in:

- Old HRX HIP ROPE:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/rope_f32.hip.cpp`
- Metal/OpenCL/CUDA-style llama.cpp ROPE implementations.
- HRX2 route slice 29:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-29.md`

Pattern:

- One workitem owns one NEOX pair `(i, i + n_dims / 2)` for one
  `(token, head)` row.
- For no-`src2` NEOX F32, the rotation angle is:

  ```text
  theta = pos[token] * freq_base^(-i / n_dims) * freq_scale
  ```

- Load the two F32 inputs, compute `cos(theta)` and `sin(theta)`, and write
  the rotated pair.
- Keep `src2`/frequency-factor ROPE as a separate family. It has a different
  ABI and remains a visible fallback after no-`src2` NEOX is covered.

HRX2 evidence:

- Route slice 29 accepted 18 exact no-`src2` NEOX F32 routes for
  `ncols=128`, `nheads={4,8,16,28,32,40}`, and `ntokens={1,16,64}`.
- Focused ggml CPU-reference validation passed all 18 rows.
- Full basket validation passed 33/33 and moved 20,304 compute nodes from CPU
  to HRX2.
- Compile reports showed 57-68 instructions, zero spills, zero private memory,
  zero local memory, and peak live units at 9-10.

Reusable conclusions:

- Treat ROPE ABI variants separately: NEOX/no-`src2`, NEOX/`src2`
  frequency-factor, non-NEOX, YaRN/ext-factor, and destination type/layout
  variants should not be bundled under one overbroad route.
- `scalar.powf<afn>` currently lacks an AMDGPU target-low contract. Spell
  `pow(freq_base, exponent)` as `exp(log(freq_base) * exponent)` when an
  approximate F32 path is acceptable and verify against ggml CPU reference.
- This standalone kernel is small and likely transcendental/dispatch dominated;
  fusion with RMS_NORM/MUL/attention setup should be evaluated separately in
  Phase 2 after all unfused attention pieces are represented.

### Row Softmax With Optional Broadcast Mask

Seen in:

- Old HRX HIP softmax:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/soft_max_f32.hip.cpp`
- CUDA softmax path:
  `sources/llama.cpp/ggml/src/ggml-cuda/softmax.cu`
- Vulkan softmax shader:
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp`
- HRX2 route slice 30:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-30.md`

Pattern:

- One workgroup owns one logical softmax row.
- Use a row max reduction, then exponentiate and reduce row sum, then
  normalize each element.
- For the common attention slice in the Phase 1 basket, `ncols=256` and
  `workgroup_size=256` are a natural exact-row mapping: one lane owns one
  column and no per-lane column loop is needed.
- For MoE probabilities in the basket, `ncols=128` and `workgroup_size=128`
  give the same direct mapping.
- Optional F32 attention masks are broadcast across head/token dimensions. Use
  specialized element strides and mask extents in the route key so the masked
  address arithmetic is explicit and debuggable.

HRX2 evidence:

- Route slice 30 accepted 12 masked attention rows for `ncols=256` and three
  unmasked MoE rows for `ncols=128`.
- Focused ggml CPU-reference validation passed representative masked and
  unmasked rows.
- Full basket validation passed 33/33 and moved 16,056 compute nodes from CPU
  to HRX2.
- Compile reports showed zero spills, zero private memory, 32-64 bytes local
  memory, and peak live units at 7-11.

Reusable conclusions:

- Treat attention softmax and MoE probability softmax as the same algorithm
  family only when the route metadata preserves column count and mask
  presence. Do not hide these behind source heuristics.
- `test-backend-ops -o SOFT_MAX` does not currently generate the exact
  256-column attention route domain, so exact graph-op rows or model-exported
  rows are required for focused validation.
- Standalone softmax removes a major Phase 1 fallback bucket, but MoE
  softmax remains CPU-assigned until upstream MoE matmuls are offloaded.

### F16/F32 Batched Attention Matvec With Grouped-Head Broadcast

Seen in:

- Old HRX HIP batched F16 matvec:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_f16_batched.hip.cpp`
- Old HRX integration notes:
  `docs/spike/reports/hrx-llamacpp-integration.md`
- HRX2 route slice 31:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-31.md`

Pattern:

- Attention KQ/KQV matvecs use F16 `src0`, F32 `src1`, and F32 output with
  p021-style F16 source strides.
- `src0` often has fewer grouped heads than the destination/RHS heads. The
  head mapping is `src0_head = dst_head / (dst_heads / src0_grouped_heads)`.
- A conservative baseline maps one workgroup to one output element and reduces
  lane-local F32 dot products over `k`.
- Old HRX also contains rows2/cols1 and cols4/8/16 variants. These are the
  natural follow-up tuning axes once coverage is closed: rows per workgroup,
  columns per workgroup, vector load width, and workgroup size.

HRX2 evidence:

- Route slice 31 accepted one target-neutral baseline route for `k={128,256}`,
  `rows={128,256}`, `cols={1,16,64}`, heads `{24,28,32,40}`, and grouped
  heads `{4,8,16}`.
- Focused ggml CPU-reference validation passed 12 exact graph-op rows covering
  KQ/KQV, cols 1/16, and multiple head counts.
- Full basket validation passed 33/33, selected the route 2676 times, and left
  zero F16/F32 `MUL_MAT` compute fallbacks.
- Compile reports showed 94-121 schedule nodes, 472-612 code bytes, zero
  spills, zero private memory, and peak live units at 11-20.

Reusable conclusions:

- Do not require contiguous F16 `src0` for attention matvec. The observed
  source layout uses byte strides such as `nb=[2,1024,256,...]` or
  `nb=[2,512,65536,...]`; the route must specialize byte strides directly.
- Keep grouped-head broadcast as explicit route metadata/config, not a source
  heuristic.
- Generic `test-backend-ops -o MUL_MAT` coverage is not enough for this route
  domain. Use exact graph-op rows from the basket or construct graph-op rows
  from scheduler metadata.

### F32/F32 MoE Logits Matvec

Seen in:

- Qwen3 basket traces:
  `cache/hrx2/phase1_0/basket-smoke-route-slice-41-20260613-012314`
- Old HRX F32 batched matvec notes:
  `docs/spike/analysis/llamacpp_pyre_kernel_optimization_spike.md`
- HRX2 route slice 42:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-42.md`

Pattern:

- MoE router logits use an F32 weight matrix shaped `[hidden, n_experts]`,
  commonly `[2048,128]`, multiplied by an F32 normalized activation shaped
  `[hidden, tokens]`.
- The output is small, `[128,tokens]`, but it is a graph-placement prerequisite
  for keeping softmax, top-k, weight gather, and expert dispatch on device.
- A conservative coverage baseline maps one workgroup to one output element
  and reduces lane-local F32 dot products over `k`.
- Old HRX has specialized cols1/cols16 F32 batched variants. Preserve these as
  future tuning axes: cols per workgroup, rows per workgroup, vector load width,
  and workgroup size.

HRX2 evidence:

- Route slice 42 accepted a target-neutral baseline for `k=2048`, `rows=128`,
  and `cols=1..512`.
- Focused graph-op validation passed `cols=1/16/64` against ggml CPU reference.
- Full basket validation passed `33/33`, selected the new route `864` times,
  and removed all F32/F32 `MUL_MAT` compute fallbacks.
- Compile reports showed 89-94 schedule nodes, 476-500 code bytes, zero
  spills, zero private memory, 32 bytes local memory, and peak live units at
  14-16.

Reusable conclusions:

- Include load-time placement domains, not just runtime token counts. The route
  admits `cols` up to `512` so llama.cpp can place F32 MoE-logits weights in
  HRX2 buffers.
- Generic `test-backend-ops` parameter filters can synthesize zero F32/F32
  matmul rows. Use exact exported graph-op rows for focused validation.

### ROPE Normal Versus NeoX Pairing With Frequency Factors

Seen in:

- CPU reference:
  `sources/llama.cpp/ggml/src/ggml-cpu/ops.cpp`
- CUDA:
  `sources/llama.cpp/ggml/src/ggml-cuda/rope.cu`
- Metal:
  `sources/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- Vulkan:
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/rope_funcs.glsl`
- HRX2 route slice 33:
  `docs/loom/llamacpp-hrx2-phase1.0-route-slice-33.md`

Pattern:

- Do not infer ROPE pairing layout from tensor shape. Read the op mode.
- `GGML_ROPE_TYPE_NORMAL` rotates adjacent pairs:
  `x[i0 + 0]` with `x[i0 + 1]`.
- `GGML_ROPE_TYPE_NEOX` rotates split-half pairs:
  `x[i0 / 2]` with `x[i0 / 2 + n_dims / 2]`.
- Frequency factors are indexed by pair: `freq_factors[i0 / 2]`.
- Non-rotated tail channels copy through unchanged when `n_dims < ne0`.
- With `ext_factor == 0`, the Yarn helper degenerates to
  `cos(pos * theta / freq_factor * freq_scale) * attn_factor` and matching
  sine, so a narrower non-Yarn route is valid for the observed Llama 3.2 rows.

HRX2 evidence:

- The first frequency-factor ROPE source implemented the NeoX split-half
  pairing and passed synthetic focused tests, but it did not cover Llama 3.2
  because that graph exports `mode=0` normal ROPE.
- Exact exported rows from
  `cache/hrx2/phase1_0/route-slice-33-rope-freq-export-current/llama32-q4k-rope-ops.txt`
  revealed the mismatch.
- Route slice 33 added `@hrx2_rope_normal_f32_freq`; focused replay passed
  4/4 exact rows with zero spills, zero private/local memory, and 65-70
  emitted instructions.
- Model smoke moved K-head ROPE to HRX2. Q-head ROPE became HRX-supported but
  remains CPU-assigned with the Q4_K matmul island.
- Route slice 43 added the Llama 3.1 normal-mode frequency-factor h32 decode
  and narrow bucket as a separate catalog route. The first h32/p64 attempt
  compiled and selected but failed strict ggml CPU-reference validation by
  about `7.3e-6`. Route slice 48 fixed the row by replacing independent
  per-pair theta exponentiation with the CPU-like recurrence, so h32/t1-64 is
  now admitted.

Reusable conclusions:

- Focused synthetic rows must include raw op params from representative graph
  exports for mode-sensitive ops. Shape-only synthetic tests can validate the
  wrong algorithm.
- Catalog metadata should preserve `supports.mode` and route matching should
  filter by it. Hard-coded C++ assumptions that a family name implies a mode
  are too brittle.
- Keep normal and NeoX as explicit algorithm variants. A shared source file is
  fine, but selection should be data-driven by mode and shape, not by
  in-source heuristics.
- Route domains for transcendental kernels need per-shape CPU-reference gates,
  not just compile/report gates. Larger token buckets can expose small
  accumulated or argument-dependent differences that decode rows do not.
- When matching ggml CPU ROPE, preserve the recurrence structure where possible
  instead of simplifying directly to an algebraically equivalent closed form.

### Q6_K Direct F32 Matmul Baseline

Seen in:

- GGML block layout: `sources/llama.cpp/ggml/src/ggml-common.h`
- CUDA dequantization reference:
  `sources/llama.cpp/ggml/src/ggml-cuda/convert.cu`
- CPU quantized dot reference:
  `sources/llama.cpp/ggml/src/ggml-cpu/quants.c`
- Loom Q6/Q8 authoring corpus:
  `sources/hrx-system/loom/src/loom/test/corpus/authoring/ffn_gate_up_swiglu_q6q8.loom`
- HRX2 route slice 35 focused evidence:
  `cache/hrx2/phase1_0/route-slice-35-q6-focused-current`

Pattern:

- `block_q6_K` stores 256 values as 128 low-nibble bytes, 64 upper-two-bit
  bytes, 16 signed i8 scales, and one f16 `d` scale.
- A simple coverage baseline maps one workgroup to one output `(row, col)`.
  Workgroup lanes cover Q6 blocks; each active lane consumes four q values in
  one Q6 block and accumulates F32 against an F32 RHS column.
- For lane-local unpacking, use branchless low-nibble selection:
  `((ql_byte >> ((part / 2) * 4)) & 0xf)`. This naturally degenerates for
  parts 0/1 versus 2/3 and avoids target control-lowering gaps.
- The upper bits use `((qh_byte >> (part * 2)) & 3) << 4`, and the quantized
  value is `(low | high) - 32`.
- Scale indexing for this four-part lane mapping is
  `scale = scales[8 * ip + il / 16 + 2 * part]`.

HRX2 evidence:

- Route slice 35 accepted a target-neutral direct route over `k=256..32768`,
  rows up to `262144`, and columns `1..64`, with `k` multiple of 256.
- Focused CPU-reference validation passed real-trace Q6 rows for attention
  K/V shapes, FFN down shapes, and a 200064-row output matmul.
- Compile reports for the focused shapes showed 9208-byte HSACO, 154-159
  emitted instructions, zero spills, zero private memory, 32 bytes local, and
  peak live units of 19-20.
- Phi-4 decode/narrow/prefill64 model smoke passed after the route was added.
  The scheduler reports Q6 matmuls as HRX-supported but still CPU-assigned
  while adjacent Q4/Q5/GLU islands remain CPU-owned.

Reusable conclusions:

- Q6_K source should avoid `scf.if` or `scf.select` for tiny integer unpack
  choices on the current AMDGPU path. Spell the packed-bit arithmetic directly.
- Do not cap quantized matmul rows at 131072: real vocabulary/output rows can
  exceed that, for example Phi-4 `200064x{1,16,64}` outputs.
- This baseline is coverage-quality, not final performance refutation. Future
  Q6 work should add packed/vectorized RHS and scale amortization axes, and
  compare against CUDA/HIPified and old HRX quantized matmul references.

### Q5_K Direct F32 Matmul Baseline

Seen in:

- GGML block layout: `sources/llama.cpp/ggml/src/ggml-common.h`
- CUDA dequantization reference:
  `sources/llama.cpp/ggml/src/ggml-cuda/convert.cu`
- CPU quantized dot reference:
  `sources/llama.cpp/ggml/src/ggml-cpu/quants.c`
- Metal Q5 matvec/MM templates:
  `sources/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- Vulkan Q5 packed block types:
  `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`
- HRX2 route slice 36 focused evidence:
  `cache/hrx2/phase1_0/route-slice-36-q5-focused-current`

Pattern:

- `block_q5_K` shares Q4_K scale/min packing and adds 32 `qh` bytes, one high
  bit per quant.
- The simple coverage baseline maps one workgroup to one output `(row, col)`
  and reuses the Q4_K row/group ownership: eight 32-element groups per 256
  element block, with each active lane handling four values in one group.
- Low nibbles are selected branchlessly from the Q4 payload with
  `((q_byte >> ((group % 2) * 4)) & 0xf)`.
- The high bit is `((qh_byte >> group) & 1) << 4`, where `qh_byte` is indexed
  by the element's position within the 32-element group.
- Scale/min decoding follows Q4_K `get_scale_min_k4`.

HRX2 evidence:

- Route slice 36 accepted a target-neutral direct route over `k=256..32768`,
  rows up to `262144`, and columns `1..64`, with `k` multiple of 256.
- Focused CPU-reference validation passed the observed Phi-4 Q5_K `wqkv`
  rows: `k=3072`, `rows=5120`, `cols=1/16/64`.
- Compile reports showed 9208-byte HSACO, 204-209 emitted instructions, zero
  spills, zero private memory, 32 bytes local, and peak live units of 25.
- Phi-4 decode/narrow/prefill64 model smoke passed. The scheduler reports
  Q5_K matmuls as HRX-supported but still CPU-assigned while adjacent Q4_K,
  Q6_K, and GLU islands remain CPU-owned.

Reusable conclusions:

- Q5_K is naturally close to Q4_K. Keep the source portable and let catalog
  route metadata select measured target/shape winners.
- Preserve Q5 high-bit extraction as explicit packed-bit work in Loom source;
  do not expect target lowering to recover packed quant dataflow from scalar
  logical loops.
- This baseline is coverage-quality. Later performance work should add packed
  RHS/vectorized load and scale reuse axes, and compare against CUDA/HIPified,
  Metal, Vulkan, and old HRX Q5 matmul references.

### MoE `MUL_MAT_ID` Weight Placement And Direct Baseline

Seen in:

- llama.cpp load-time weight selector:
  `sources/llama.cpp/src/llama-model-loader.cpp`
- Old HRX MoE kernels and fusions:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_id_q4_k*.hip.cpp`
- Vulkan MoE dispatch/fusion routing:
  `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- CUDA indexed matmul reference:
  `sources/llama.cpp/ggml/src/ggml-cuda/mmid.cu`
- HRX2 route slice 38 focused and model evidence:
  `cache/hrx2/phase1_0/route-slice-38-mul-mat-id-q4-focused-current`
  and
  `cache/hrx2/phase1_0/route-slice-38-mul-mat-id-q4-smoke-current`

Pattern:

- `MUL_MAT_ID` expert weights are shaped `[k, rows, nexperts, 1]`.
- Expert ids are shaped `[nselected, ntokens, 1, 1]`.
- Gate/up RHS often broadcasts across selected experts as `[k,1,ntokens,1]`;
  down RHS is selected-strided as `[k,nselected,ntokens,1]`.
- A direct coverage route can map one workgroup to one
  `(row, selected_expert, token)` and specialize the RHS selected/token
  strides and destination token stride.
- Keep the algorithmic choice explicit in route metadata. Direct per-output
  dequant/dot is only a coverage baseline; fused gate/up/swiglu/down variants
  and packed/vectorized RHS variants are separate tuning candidates.

Reusable conclusions:

- Route domains for load-bearing quantized matmuls must include llama.cpp's
  synthetic 512-token `select_weight_buft` probe. Otherwise the model loader
  leaves expert weights in CPU buffers even if runtime `supports_op` says HRX2
  supports the real shape.
- Always run a model-load placement smoke after focused op validation. The
  acceptance signal is a large HRX model buffer and HRX scheduler placement for
  the intended weight ops, not just `test-backend-ops` success.
- Preserve separate stride parameters for broadcast RHS and selected RHS.
  Hard-coding either layout misses half of common MoE `MUL_MAT_ID` use.
- Q5_K and Q6_K indexed baselines can reuse the same expert-plane addressing
  as Q4_K, but their block byte sizes and unpack paths must come from their
  direct matmul sources. Keep the C++ route/dispatch path shared while keeping
  Loom source and catalog families quant-specific.
- If a real basket export does not include a quantized indexed row, synthesize
  a focused CPU-reference row by preserving expert topology and correcting the
  source type plus block strides. Mark it as synthetic in the evidence; do not
  treat it as model frequency evidence.

### Split-Source GLU Routes And Tanh-GELU Spelling

Seen in:

- ggml CPU GLU reference:
  `sources/llama.cpp/ggml/src/ggml-cpu/vec.h`
- HRX2 route slice 44:
  `cache/hrx2/phase1_0/route-slice-44-glu-large`
- CUDA/HIP/Vulkan FFN prior art generally treats GLU as a fusion candidate
  after gate/up matmuls rather than as a terminal optimization target.

Pattern:

- Split-source GLU has stable ABI: `src0`, `src1`, `dst`, all contiguous F32
  for the accepted HRX2 Phase 1 routes.
- The activation variant is semantically important. Route metadata must key on
  `supports.glu_op`; SWIGLU and GEGLU cannot share a route even when shape and
  bindings match.
- SWIGLU uses `silu(x) * gate`.
- GEGLU uses ggml tanh-GELU semantics. In Loom on current AMDGPU, prefer:

  ```text
  gelu_tanh(x) = x * logistic(2 * sqrt(2/pi) * x * (1 + 0.044715*x*x))
  ```

  over direct `scalar.geluf<tanh>` until the target-low contract exists for
  `scalar.tanhf`.

Reusable conclusions:

- Treat standalone GLU as Phase 1 unfused coverage. Most performance work
  should evaluate GLU fused with FFN matmul/dataflow, but the standalone route
  is still useful for proving graph placement and eliminating CPU islands.
- Keep large dense FFN widths as shape-domain metadata, not hard-coded Loom
  source. Slice 44 admitted a broad SWIGLU domain `ncols=13824..32768`,
  `nrows=1..64`, while preserving exact activation selection in metadata.
- Focused `test-backend-ops` must include exact exported rows for each
  activation variant. A SWIGLU pass says nothing about GEGLU correctness.

### Q4_K Prompt Matmul Column Reuse

Seen in:

- HRX2 Phase 2a Q4_K cols4 route:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q4_k_f32.loom`
- Vulkan prior art: `mul_mmq.comp` and Q8_1 RHS quantization route.
- Old HRX1 prior art: Q4_K direct Q8_1 RHS route plus x4/MMQ-style routes for
  other K-quants.

Pattern:

- A direct one-output-per-workgroup Q4_K x F32 RHS route repeats the same
  Q4_K scale/min unpack and dequantization once for every prompt column.
- Computing four RHS columns in one workgroup reuses that dequantized Q4_K
  value across four accumulators. On the W7900 Phase 2a basket this improved
  Q4_K prefill from roughly 24-59 tok/s to roughly 32-81 tok/s, depending on
  model and prompt bucket.
- This is still an intermediate route. It keeps F32 RHS traffic and does not
  implement the Vulkan/HRX1 packed RHS strategy.

Reusable conclusions:

- When a K-quant prompt route still uses direct F32 RHS and one output per
  workgroup, first ask whether the quantized LHS work is being repeated across
  prompt columns. A cols4/cols8 direct route can be a useful proof and a
  moderate lift.
- For production-level prefill throughput, treat packed RHS as the target:
  quantize/pack RHS to Q8_1 scratch, then run an MMQ-class tiled Q4_K x Q8_1
  kernel. Direct cols4 is not enough to close an order-of-magnitude gap to
  Vulkan.
- Q4_K x Q8_1 dot signedness is unsigned-by-signed: Q4 codes are unsigned
  4-bit values and Q8_1 activations are signed i8 values. In Loom, spell this
  as `vector.dot4i<u8s8>`, not `s8s8`, even when a scalar-looking expansion
  happens to be numerically equivalent for Q4 values in 0..15.
- If a route schedule assumes a column tile, encode that in metadata. HRX2 now
  supports `shape_guards.cols_multiple_of`; use it instead of relying on
  speculative JIT fallback or in-kernel tail checks.
- Separate packed-layout validation from MMQ schedule validation. A temporary
  HRX2 direct Q4_K x packed-Q8_1-x4 cols4 consumer passed the same focused
  backend-op rows that the 32x32 MMQ route failed:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-wg256-diag-20260615-222302/`.
  This proves the x4 packer/layout can feed a correct Q4_K consumer, but it is
  not a throughput candidate. Use this pattern to isolate layout bugs, then
  delete or keep the probe out of production selection so it cannot shadow the
  real tiled route.

### K-Quant x Q8_1 Prompt MMQ32x32

Seen in:

- HRX1 HIP Q6_K prior:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1.hip.cpp`,
  export `hrx_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32`.
- HRX1 HIP Q5_K prior:
  `sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q5_k_q8_1.hip.cpp`,
  exports `hrx_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32` and
  `hrx_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32`.
- HRX2 Loom Q6_K accepted port:
  `sources/llama.cpp/ggml/src/ggml-hrx2/kernels/mul_mat_q6_k_f32.loom`,
  export `hrx2_mul_mat_q6_k_q8_1_x4_mmq32x32_static`.

Pattern:

- Use a 32 row x 32 column prompt tile with 128 lanes.
- `tid & 31` owns one output row within the tile.
- `tid >> 5` selects one of four column groups; each lane accumulates eight
  output columns.
- For every Q8_1 block along K, cooperatively load the 32 column x 8 packed
  Q8_1 RHS words into workgroup memory. Q6_K only needs Q8 `d`; Q5_K also
  needs Q8 `s` for min correction.
- Each lane loads the current row's K-quant block, unpacks eight groups of four
  quantized values, uses explicit packed dot4 operations against the staged RHS
  words, applies per-group K-quant scale and Q8 scale, and writes eight F32
  outputs.

Reusable conclusions:

- This schedule class is the right first refutation target when a prompt
  K-quant route is one-row/one-column or one-row/four-column direct. Direct
  routes repeatedly decode the same K-quant row for each prompt column; the
  32x32 MMQ tile amortizes that decode across eight columns per lane and shares
  the RHS tile across 32 rows.
- Do not collapse this to a generic one-column WG32 diagnostic. In Phase 2a,
  a correctness-clean Q6_K generic-Q8 WG32 route was slower than the existing
  x4 direct route for prompt rows. The useful prior is the tiled x4 MMQ shape.
- In Loom, spell the schedule WYSIWYG: `cols_multiple_of: 32`, workgroup size
  128, RHS LDS tile shape 256 i32 words, workgroup barriers around the tile,
  explicit packed load width, explicit `vector.dot4i<s8s8>` for Q6_K x Q8_1,
  and full unroll of the eight Q8 groups.
- Validate block-index math carefully. The Q6_K source block is
  `row * blocks_per_row + (kb / 8)`, where `kb` iterates over 32-element Q8_1
  blocks. Using `(row * blocks_per_row + kb) / 8` silently selects the wrong
  source block and produces finite but incorrect results.
- Backend-op wins do not guarantee full-model wins. The accepted Q6_K Loom
  port improved model-derived p512 Q6 rows by 3-4x, but the reduced model
  basket barely moved because Q4_K, Q5_K, pointwise chains, and attention were
  still larger boulders. Always follow focused gates with a reduced model
  HRX2/Vulkan run and route histogram.

## Open Items For Future Entries

- Add matvec/mul_mat priors from Vulkan MMQ/cooperative matrix, CUDA MMV/MMQ,
  and old HRX Q4/Q5/Q6 route notes.
- Add attention priors around logits staging, vectorized KV loads, and
  decode/prefill split.
- Add MoE routing/top-k priors from the old HRX spike and Vulkan fusion code.
- Add set-rows priors, especially shared staging limits and destination type
  constraints.
