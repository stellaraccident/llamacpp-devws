# llama.cpp HRX Loom Kernel Catalog Design

Date: 2026-06-11

## Purpose

Replace the current `ggml-hrx` static HIP kernel catalog with a structured,
automated, Loom-authored kernel catalog that can generalize across model shapes
and AMDGPU targets.

The goal is not to preserve the current catalog shape. The current catalog is a
useful source of workload evidence and accepted/rejected ideas, but many entries
are the residue of local search on one model shape and one architecture. The new
system should make only two kinds of human or agent judgment important:

- choosing worthwhile fusion families from graph/profile evidence;
- wiring a winning family into llama.cpp once the automated loop proves it.

Everything else should be represented as explicit source facts, tunable
configuration, compiler evidence, correctness evidence, benchmark evidence, and
cache policy.

## Author-Call Anchor

The concrete proposal:

1. Treat a llama.cpp backend catalog entry as a Loom family, not a hand-picked
   kernel.
2. Put graph-level fusion judgment in agents, but put schedule selection in
   declared provider/config axes plus automated evidence.
3. Use `func.apply`/`func.template`, `check.case`, `check.benchmark`, compile
   reports, and the `loomc` embedding surface as the catalog substrate.
4. Prefer shipping Loom source/bytecode families and compiling exact-shape
   executables on demand. Use an executable cache as an optimization, not as
   the primary catalog format.
5. Prove the flow first on Q4 MoE SWIGLU, because it already exposed the right
   hard problems: explicit wide loads, dot forms, route grouping, wave policy,
   standalone timing, and compile-report quality.

The main questions for the Loom author are:

- how to express config-driven provider selection cleanly;
- how to get reliable per-candidate static summaries from compile reports;
- how to request and verify wave32/wave64;
- how to spell/lower mixed `u8s8` integer dot;
- how to make AMDGPU executable artifacts and ABI metadata available through
  the embedding path llama.cpp would use.

## Current State

The current HRX backend is organized around:

- generated static HSACO entries in `ggml-hrx/kernels/generate_hrx_kernels.py`;
- `ggml_hrx_kernel_entry` metadata containing name, target, ABI counts,
  constants size, and workgroup size;
- `ggml_backend_hrx_op_provider` objects that hold a loaded executable, export
  ordinal, export metadata, and provider name;
- hardcoded C++ routing predicates in `ggml-hrx.cpp`;
- environment variables for disabling, forcing, or expecting specific routes.

This works, but it couples several things that should be separate:

- source implementation;
- shape support predicate;
- target support predicate;
- candidate schedule choice;
- benchmark winner;
- deployment policy.

The Q4 MoE SWIGLU route is a representative example. The current selector has
multiple hardcoded provider names, historical prompt thresholds, Qwen-specific
shape checks, grouped route handling, Q8_1 packing decisions, and fallback
paths in one C++ function. That is not a scalable catalog.

## Design Thesis

The new catalog should be a Loom-native kernel-family system:

```text
ggml graph pattern + tensor/layout facts + target facts
  -> operation contract
  -> Loom source family
  -> provider/config candidates
  -> compile report filters
  -> correctness cases
  -> standalone benchmark rows
  -> optional full-graph validation
  -> executable cache entry
  -> llama.cpp provider dispatch
```

The catalog unit is not "one kernel file." The catalog unit is a family:

```text
family = fusion contract + ABI contract + shape domain + provider library +
         tunable config axes + correctness/benchmark policy + evidence records
```

Loom is useful here because it can keep the relevant facts in one system:

- `func.apply<contract>` in the model-shaped kernel body;
- `func.template<contract> ... priority(...)` providers for implementation
  variants;
- `check.case` correctness policy next to the source;
- `check.benchmark<@case>` workload rows next to the source;
- `check.param.choice` and config bindings for per-sample specialization;
- compile reports and target artifacts as machine-readable evidence;
- `loomc` as an embedding surface for AOT, JIT, executable caches, and tuning.

## Core Concepts

### Fusion Family

A fusion family is a graph-level operation contract. Examples:

- `mul_mat_id_q4_k_swiglu`
- `mul_mat_id_q4_k_mul`
- `mul_mat_vec_q6_k_silu_mul`
- `bf16_gate_up_swiglu`
- `gated_delta_net_state_update`
- `topk_moe`

Each family owns:

- a graph pattern matcher in llama.cpp;
- a tensor/layout fact extractor;
- a Loom ABI definition;
- a Loom source root kernel;
- provider contracts for internal schedule choices;
- correctness cases over representative shapes;
- benchmark rows over shape samples;
- a result database of candidate evidence.

### Provider Contract

Provider contracts are the replacement for hand-maintained sibling kernels.

For example, a Q4 MoE SWIGLU family should have a stable root kernel that asks
for logical operations:

```text
hrx.q4moe.route_prepare
hrx.q4moe.q8_pack
hrx.q4moe.accumulate_gate_up
hrx.q4moe.activate_store
```

Variants then live behind `func.template` providers:

```text
accumulate scalar packed loads
accumulate vector<4xi32> packed loads
accumulate vector<8xi32> packed loads
accumulate LDS-staged Q8
accumulate wave32 dual-lane
accumulate route-grouped
accumulate BN16 token tile
```

The key difference from the current catalog is that these are provider choices
inside one family, not unrelated C++ functions with independent routing code.

### Shape Class

Do not tune only exact shapes. Tune shape classes.

A shape class should include at least:

- op/fusion family;
- quantization type and packing layout;
- `k`, `rows`, `cols` or `n_tokens`;
- MoE expert count, `n_ids`, top-k, and route density;
- batch/prompt/decode regime;
- tensor contiguity and stride facts;
- dtype and numeric policy;
- target architecture and target feature facts;
- wavefront/subgroup support;
- known target limits such as LDS, VGPR budget, and dot/WMMA availability.

The first catalog can still seed from exact Qwen rows, but accepted routes
should be recorded against a shape class with tested boundaries.

### Candidate Config

Candidate config is the explicit search space for a family. It should be
represented as Loom config and/or provider selection, not as ad hoc C++ names.

Useful axes include:

- tile rows, columns, and route grouping;
- workgroup size and subgroup/wave size;
- lane ownership model;
- vector load width;
- whether packed Q8 is precomputed, inlined, or skipped;
- LDS staging depth and layout;
- reduction strategy and broadcast strategy;
- numeric approximation mode;
- store vectorization;
- prefetch/hoist policy when Loom supports expressing it cleanly.

The Q4 pilot showed why this matters. Loom did not infer wide packed-weight
loads from scalar source. The winning source had to spell `vector<4xi32>` loads
explicitly. That is the desired WYSIWYG property: if wide memory operations are
part of the schedule, they should be explicit in the provider.

## Autotune Loop

### 1. Discover Work

Use provider trace and HRX/IREE profile data to rank fusion families and shape
classes.

Inputs:

- `GGML_HRX_TRACE_PROVIDERS=1` route logs;
- `HRX_PROFILE_FILE` dispatch/profile summaries;
- optional Vulkan perf labels as a reference;
- graph export or graph walk data for candidate fusion patterns.

Output:

```json
{
  "family": "mul_mat_id_q4_k_swiglu",
  "shape": {
    "k": 2048,
    "rows": 512,
    "n_ids": 8,
    "n_tokens": 64,
    "quant": "q4_k",
    "rhs": "f32_or_q8_1",
    "regime": "prefill"
  },
  "baseline_provider": "static catalog provider name, if any",
  "device_time_bucket": "...",
  "dispatch_count": "..."
}
```

### 2. Generate Or Select Family Source

If the family already exists, instantiate it with the shape class. If not, an
agent writes a Loom family skeleton:

- ABI matching the llama.cpp dispatcher;
- `check.case` correctness oracle;
- `check.benchmark` rows for decode, prompt, and boundary cases;
- provider contracts for the obvious inner schedule boundaries.

This is where agent judgment belongs. The agent proposes the fusion and the
source-level schedule vocabulary. It does not hand-pick the final winner.

### 3. Enumerate Candidates

For each family, produce a bounded candidate set from declared axes. Examples:

```text
load_width = 1, 4, 8
lane_model = lane64_logical, wave32_dual_lane
q8_policy = f32_direct, q8_1_x4_prepack
route_policy = per_route, grouped_route4, grouped_route8
lds_policy = none, q8_stage
wave = 32, 64
```

The enumeration should be reproducible. A candidate ID should be a hash over:

- family source hash;
- provider library hash;
- config bindings;
- target profile;
- ABI version;
- shape class.

### 4. Compile And Triage

Compile candidates with Loom and collect:

- diagnostics;
- compile report JSON;
- emitted artifact;
- target listing;
- ELF notes;
- ABI metadata.

Compile-report filters should reject candidates before timing when possible:

- unresolved `func.apply`;
- failed target lowering;
- ABI mismatch with the family dispatcher;
- missing required primitive, such as `v_dot` or WMMA;
- wrong final wavefront size;
- spills or private memory above budget;
- LDS above target budget;
- obvious scalarization of intended vector memory ops;
- global load/store count outside expected range;
- impossible occupancy due to VGPR/LDS pressure.

The compile report should not be the final performance oracle. It is a
candidate reducer and debugging aid.

### 5. Correctness

Correctness must be source-owned where possible:

- `check.case` for synthetic exact oracles;
- fixture-backed cases for model-derived tensors;
- boundary cases for tails and strides;
- numeric-policy-specific tolerances.

For llama.cpp promotion, keep the existing focused and full-model gates:

- focused backend op/fusion tests;
- `test-backend-hrx`;
- `test-backend-ops` for affected ops;
- full Qwen gate for risky MoE, attention, recurrent, or approximate paths;
- chat/loop guards when hidden state or approximate prompt math is involved.

### 6. Benchmark

Use Loom standalone timing to rank kernel-code candidates before llama.cpp
integration:

- batch tiny kernels enough to avoid single-dispatch overhead traps;
- record p50, p90, mean, min/max, and runner/profile uncertainty;
- separate compile time from dispatch time;
- use interleaved comparisons for close candidates;
- rerun before promotion.

Then validate winners in llama.cpp:

- route must be live;
- dispatch count must be understood;
- bucket time must improve;
- adjacent hot buckets must not regress;
- wall tok/s must be neutral or positive.

### 7. Persist Evidence

Every candidate should leave a machine-readable evidence record:

```json
{
  "candidate_id": "...",
  "family": "...",
  "shape_class": "...",
  "target": "gfx1100",
  "config": {},
  "source_hash": "...",
  "compile": {
    "succeeded": true,
    "sgpr": 22,
    "vgpr": 140,
    "spills": 0,
    "lds_bytes": 16,
    "wavefront_size": 32,
    "instruction_mix": {},
    "artifact_hash": "..."
  },
  "correctness": {
    "cases": [],
    "passed": true
  },
  "benchmark": {
    "standalone": {},
    "llamacpp_bucket": {},
    "llamacpp_wall": {}
  },
  "decision": "accepted | rejected | pending",
  "reason": "short human-readable rationale"
}
```

Rejected candidates are first-class data. They prevent the next agent from
rediscovering the same bad point.

## llama.cpp Integration Architecture

### Source-First JIT Catalog

If the Loom compiler can sustain sub-microsecond compile times over realistic
kernel families, the default deployment model should change. We should not
think of Loom primarily as an offline HSACO generator. We should ship
meta-programmed Loom source or bytecode libraries and specialize them at runtime
for the exact graph shape, target profile, and policy.

In that model:

```text
packaged catalog = Loom family bytecode + provider libraries + evidence seeds
runtime catalog  = exact-shape compiled executable cache
```

The shipped artifact would contain:

- family source/bytecode;
- provider contracts and template implementations;
- ABI schemas for llama.cpp dispatchers;
- candidate-axis declarations;
- correctness/benchmark metadata where useful;
- seed evidence from known devices and shapes;
- conservative fallback policy.

The runtime would:

1. extract graph and tensor facts from llama.cpp;
2. map them to a Loom family and shape class;
3. specialize the family with exact dimensions, strides, target facts, and
   numeric policy;
4. compile or look up the exact executable;
5. optionally autotune nearby candidates when in tuning mode;
6. dispatch through the existing HRX provider path.

The executable cache remains important, but as a latency and persistence layer:

- avoid repeating target emission and executable load;
- persist winners across process runs;
- keep approved candidates separate from exploratory ones;
- allow no-network, no-tuning production runs;
- support rollback to known-good static or cached providers.

This also changes the role of a packaged static HSACO catalog. Static HSACOs
become optional emergency fallbacks or release-mode warm seeds, not the main
thing we maintain by hand.

There are still three costs to separate:

- Loom front-end/specialization compile time;
- native AMDGPU executable emission/link/load time;
- correctness and benchmark time.

Sub-microsecond compilation makes million-candidate structural exploration
plausible, but production dispatch still needs an approved executable cache
unless emission and `hrx_executable_load_data` are also cheap enough for the
hot path. The design should assume compile/search can be very aggressive while
normal inference only uses approved cached winners.

### Provider Loader

Extend the provider abstraction from "static HSACO catalog only" to:

```text
provider kind:
  static_hsaco
  loom_cached_hsaco
  loom_jit_pending
  unavailable
```

The current `ggml_backend_hrx_op_provider` already has the right general shape:

- loaded executable;
- export ordinal;
- export info;
- name.

It needs enough extra metadata to support:

- family ID;
- candidate ID;
- shape class;
- target profile;
- source/config hash;
- evidence status;
- fallback provider.

### Executable Cache

Cache key:

```text
hash(
  loom bytecode/source hash,
  provider library hash,
  family ABI version,
  compile root symbol,
  config bindings,
  shape class,
  target architecture/profile,
  ROCm/HRX/Loom codegen version
)
```

Cache value:

- executable bytes;
- artifact format;
- export name;
- ABI metadata;
- compile report;
- benchmark/evidence summary;
- last validated timestamp;
- fallback relation.

Cold path:

1. Try approved exact-shape executable cache entry.
2. Compile the packaged Loom family for the exact shape if policy allows.
3. Fall back to packaged static winner or conservative cached provider.
4. Optionally enqueue background tuning.

Tuning path:

1. Compile candidates out of band.
2. Run correctness and standalone benchmarks.
3. If enabled, run llama.cpp validation.
4. Mark the winner as approved for the shape class.

Runtime decode should not block on exploratory compilation unless explicitly
running in a tuning mode.

### Routing Policy

Replace hardcoded provider-name selection with:

```text
shape facts -> family -> candidate policy -> provider handle
```

The C++ dispatcher still owns:

- graph pattern matching;
- buffer binding construction;
- constants struct construction;
- scratch allocation;
- HRX dispatch submission.

But the schedule choice should come from the candidate policy database, not
from hardcoded C++ thresholds.

This gives us:

- static deployment with known winners;
- opt-in live tuning;
- per-device cache adaptation;
- fallback to conservative providers;
- reproducible evidence for why a provider is selected.

## First Target Family: Q4 MoE SWIGLU

This is the best first family because it is complex enough to exercise the new
workflow and already has a Loom pilot.

Existing static route characteristics:

- `MUL_MAT_ID + MUL_MAT_ID + GLU` fusion;
- Q4_K gate/up weights;
- F32 or Q8_1 RHS path;
- optional Q8_1 x4 prepack;
- optional route compaction;
- grouped and non-grouped providers;
- prompt/decode thresholds hardcoded in C++.

Proposed Loom family:

```text
family: hrx.mul_mat_id_q4_k_swiglu
root ABI:
  gate_weight, up_weight, rhs, ids, dst
constants:
  k, rows, n_ids, n_tokens, n_experts,
  strides, route_capacity or route policy facts
internal contracts:
  route_prepare
  rhs_prepare_q8_1_x4
  q4_unpack_dot_accumulate
  gate_up_reduce
  swiglu_store
```

Initial candidate axes:

- direct F32 RHS vs Q8_1 x4 prepack;
- no route compaction vs grouped route4/route8;
- lane64 logical vs wave32 dual-lane;
- scalar vs `vector<4xi32>` vs `vector<8xi32>` packed loads;
- Q8 register reuse vs LDS staging;
- rows per workgroup;
- tokens per workgroup;
- wave32 vs wave64, once Loom exposes reliable control.

Known evidence from the Loom pilot:

- explicit `vector<4xi32>` packed-weight loads beat scalar-looking loads;
- `vector<8xi32>` did not improve over vec4;
- LDS staging did not win the interleaved comparison;
- forced wave64 was not supported as a principle for this card;
- mixed `u8s8` dot lowering still needs Loom author input;
- compile report/report attribution needs improvement for multi-candidate
  workflows.

This family should become the first end-to-end proof:

```text
Loom standalone winner
  -> HRX executable cache entry
  -> llama.cpp provider route
  -> Qwen focused correctness
  -> full-model bucket/wall validation
```

## Promotion Policy

A candidate may become a default winner for a shape class only if:

- it has a passing source-owned correctness case;
- it has passing focused llama.cpp correctness;
- it has no unreviewed compile diagnostics;
- static metadata is understood;
- standalone benchmark wins beyond noise;
- full-model bucket time is neutral or positive;
- wall tok/s is neutral or positive;
- fallback policy is available.

Approximate numeric routes need stricter gates:

- explicit numeric policy in the family;
- rollback knob;
- long-generation guard;
- evidence that model behavior is stable.

## Implementation Plan

### Phase 1: Evidence Schema And Trace Import

Build a small workspace tool that converts provider traces and HRX profile
dispatch rows into candidate work items:

```text
family, op graph pattern, shape facts, current provider, bucket time
```

Also define the evidence JSON schema and store results under `cache/` during
experiments.

### Phase 2: Loom Family Package

Refactor the Q4 MoE SWIGLU pilot from sibling kernels into a real Loom family:

- one root kernel;
- `func.apply` internal contracts;
- template providers for scalar, vec4, vec8, LDS, dual-lane;
- `check.param.choice` rows for token counts and route density;
- named benchmarks for decode, small prefill, and larger prefill.

### Phase 3: Standalone Autotuner

Create a standalone runner that:

- enumerates provider/config candidates;
- compiles each candidate;
- records compile report, ELF notes, and assembly summary;
- runs correctness;
- runs batched/interleaved benchmarks;
- writes accepted/rejected evidence records.

This should remain independent of llama.cpp at first.

### Phase 4: HRX Backend Cache Loader

Teach `ggml-hrx` to load or create a Loom-generated executable for one family
behind an opt-in flag, with cache lookup first and static catalog fallback.

The dispatch ABI should be identical to the current provider where possible so
the first integration does not require a graph rewrite.

### Phase 5: llama.cpp Validation Harness

Add a focused harness for:

- forcing a family candidate ID;
- proving the route is live;
- running focused correctness;
- collecting HRX profile bucket data;
- comparing against the current static provider.

### Phase 6: Live Tuning Policy

Only after the cache-backed path works:

- add background/on-demand tuning mode;
- limit tuning to idle time or explicit benchmark commands;
- persist approved winners;
- keep production inference on approved cache entries plus static fallback.

## Questions For The Loom Author

1. What is the intended syntax for config-driven provider selection beyond
   `priority(...)`?

2. Should a tuning workflow generate one candidate module per artifact, or
   should multi-export modules have per-entry static summaries and clean report
   attribution?

3. How should wave32 vs wave64 be requested, and can the final wavefront size
   be reported per exported kernel?

4. What is the intended spelling for RDNA3 mixed signedness
   `v_dot4_i32_iu8` forms such as Q4 unsigned bytes times signed Q8 bytes?

5. Should Loom infer/coalesce adjacent scalar packed-weight loads, or should
   authoring guidance require explicit `vector<4xi32>` loads for this pattern?

6. What controls whether compile reports include register pressure, spills,
   instruction mix, memory summaries, and scheduling data?

7. Can benchmark output make tiny-kernel dispatch overhead and profile clock
   uncertainty more visible?

8. Is there already a direct `loomc` AMDGPU executable emission path with the
   artifact format HRX expects, or does llama.cpp need a small adapter layer?

9. Can Loom expose a stable target/ABI metadata record for binding count,
   parameter count, constants size, workgroup size, and export name so llama.cpp
   does not need to rediscover or duplicate it?

10. Is there an intended way to express "this value must remain lane-varying /
    VGPR" when a dot operand would otherwise become scalar?

11. Does the reported sub-microsecond compile benchmark include only Loom
    source/IR compilation, or also AMDGPU executable emission and loader-ready
    artifact production?

12. What parts of `loomc` are safe and cheap enough to call on the llama.cpp
    inference path, and which should be isolated to background tuning or cache
    warmup?

## Non-Goals

- Do not preserve every current HIP catalog entry as a Loom entry.
- Do not rely on wall tok/s alone for kernel selection.
- Do not run unconstrained benchmarking or exploratory tuning during normal
  inference, even if compile itself is cheap.
- Do not encode Qwen-only thresholds as permanent C++ route policy.
- Do not treat wave64, LDS, or Vulkan-shaped schedules as principles without
  current target evidence.

## Definition Of Success

The design succeeds when a new fusion family can be added by:

1. writing a Loom family source with correctness and benchmark rows;
2. declaring candidate axes;
3. running the standalone autotuner;
4. promoting a winning evidence record;
5. wiring one family dispatcher in llama.cpp;
6. letting the cache/policy system select the provider for future matching
   shapes.

The steady-state catalog should be explainable as data:

```text
For this family, shape class, and target, this provider is selected because
these candidates were compiled, these were rejected, this one passed, and these
benchmarks won.
```

That is the part the current catalog cannot provide.
