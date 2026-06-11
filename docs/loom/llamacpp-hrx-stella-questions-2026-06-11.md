# Stella HRX Loom questions - current answers

Date: 2026-06-11

Audience: Stella, Ben, and agents helping turn llama.cpp HRX kernels into Loom
kernels.

This note is deliberately a current-truth snapshot. It separates what exists in
the tree today from the contract we want before this becomes public guidance.
The checked-in guide should absorb the stable answers after the implementation
gaps below are closed.

Status words used below:

- Current: implemented and verified in source.
- Intended: design direction we should implement before presenting the flow as
  stable.
- Gap: product/API/reporting behavior that is not yet good enough.
- Action: the concrete thing to implement or document.

## 1. Config-driven provider selection beyond priority

Question: What is the intended syntax for config-driven provider selection
beyond `priority(...)`?

Current:

- `func.def` and `func.decl` already have `target(@...)`, `abi(...)`,
  `export(...)`, and export attributes through `_CONTRACT_ATTRS` in
  `loom/py/loom/dialect/func/defs.py`.
- `kernel.def` already has `target(@...)`, `artifact(@...)`, and export
  metadata. The example syntax is in `loom/py/loom/dialect/kernel/defs.py`.
- `func.template` and `func.ukernel` currently do not expose `target(@...)`.
  Their current provider controls are the implemented contract key, shared
  function modifiers, optional `priority(...)`, and signature `where [...]`
  predicates.
- Template selection already evaluates `where [...]` predicates and priority.
  See `loom/src/loom/transforms/symbol/template_selection.c` and
  `loom/src/loom/transforms/test/template_selection.loom-test`.
- Useful current predicate forms include `eq`, `ne`, `lt`, `le`, `gt`, `ge`,
  `mul`, `min`, `max`, `pow2`, and `range`.
- There is target-predicate machinery elsewhere, but target predicates are not
  currently the provider applicability spelling for `func.template`.

Current source shape:

```loom
func.template<q4.dot> priority(20) @aligned(%n: index, %x: tensor<[%n]xi8>)
    -> (tensor<[%n]xi32>) where [mul(%n, 16)] {
  ...
}

func.template<q4.dot> priority(1) @fallback(%n: index, %x: tensor<[%n]xi8>)
    -> (tensor<[%n]xi32>) {
  ...
}
```

Intended:

`target(@...)` should become an applicability constraint on provider ops:

```loom
amdgpu.target<gfx1100> @gfx11
amdgpu.target<gfx1200> @gfx12

func.template<q4.dot> target(@gfx11) priority(20) @gfx11_dot(...) -> (...) {
  ...
}

func.template<q4.dot> target(@gfx12) priority(20) @gfx12_dot(...) -> (...) {
  ...
}

func.template<q4.dot> priority(1) @generic_dot(...) -> (...) {
  ...
}
```

The key semantic point is that `target(@...)` on a provider is not export/ABI
metadata. It is an applicability filter for template selection. A provider is
applicable when it is untargeted or when its target matches the selected target
of the root/apply context. A targeted provider in a targetless compile should
fail closed unless there is an untargeted fallback.

Action:

Filed `loom-pdqzn`, "Select and prune target-specific func providers early".
That bead covers parser/printer/bytecode/facts support for `func.template` and
`func.ukernel`, template selection behavior, fail-closed diagnostics, liveness,
DCE/link-root flow, and tests proving off-target provider bodies are not walked
by later JIT passes.

This matters for llama.cpp because Stella may reasonably want one `.loom` or
one linked library containing gfx11/gfx12/gfx950/generic providers. A gfx11
shape miss must not pay to legalize, lower, report, or emit the bodies that can
never apply.

## 2. Candidate modules and tuning attribution

Question: Should a tuning workflow generate one candidate module per artifact,
or should multi-export modules have per-entry static summaries and clean report
attribution?

Current:

Reports carry function/export-ish identity where the backend has it, and
`loomc_compile_options_t` has `compile_root_symbol` so a caller can materialize
one root from a larger source/library. The AMDGPU examples use this shape:

- `loom/binding/c/include/loomc/compile.h`
- `loom/binding/c/example/emit_amdgpu_offline.c`
- `loom/binding/c/example/emit_amdgpu_hsa.c`

Position:

For Stella's first tuning workflow, use one linked root/candidate per compile
result/artifact. Multi-export modules should not be forbidden, but they are the
wrong default for early tuning because they conflate attribution, cache keys,
compile time, report size, and generated artifact provenance.

Practical flow:

1. Keep the authored library broad.
2. Link or compile with a single `compile_root_symbol`.
3. Emit one artifact for that root/candidate.
4. Benchmark that artifact.
5. Cache and compare using the root, config, target, pass program, source hash,
   and artifact format as part of the key.

Action:

The guide should document the one-root flow as the clean baseline. Multi-export
report attribution can become a later feature, but it should not block Stella
from getting reliable benchmark data.

## 3. Wave32 and wave64

Question: How should wave32 vs wave64 be requested, and can the final wavefront
size be reported per exported kernel?

Current:

Wavefront size is selected through the AMDGPU target record using
`subgroup_size`:

```loom
amdgpu.target<gfx1100> @gfx11
amdgpu.target<gfx1100> @gfx11_wave64 {subgroup_size = 64}
amdgpu.target<gfx942> @gfx942 {max_flat_workgroup_size = 1024, subgroup_size = 64}
```

Evidence:

- `loom/src/loom/target/arch/amdgpu/test/source_low/target_records.loom-test`
- `loom/src/loom/target/arch/amdgpu/ops/target.c`
- `loom/src/loom/target/arch/amdgpu/lower/preamble.c`

The target record verifier rejects unsupported wavefront sizes for the selected
processor. Current diagnostics cover examples such as invalid wave32 on gfx942
and invalid wave64 on gfx1250 in
`loom/src/loom/target/arch/amdgpu/test/source_low/target_record_diagnostics.loom-test`.

Gap:

There is not yet a stable per-export manifest field saying "final wavefront
size = 32/64". The value exists in the selected target bundle/snapshot and is
used by lowering, but HRX/llama.cpp should not have to infer it by re-reading
the target record if the compiler has already produced a loadable artifact.

Action:

The public guide can document `subgroup_size` as the current request spelling.
The export/artifact manifest work should include final subgroup/wavefront size
per exported kernel.

## 4. Mixed signedness dot spelling

Question: What is the intended spelling for RDNA3 mixed signedness
`v_dot4_i32_iu8` forms such as Q4 unsigned bytes times signed Q8 bytes?

Current:

Source-level spelling is semantic, not the final ISA suffix:

```loom
%dot = vector.dot4i<u8s8> %lhs, %rhs, %acc
    : vector<4xi8>, vector<4xi8>, vector<1xi32>

%dot = vector.dot4i<s8u8> %lhs, %rhs, %acc
    : vector<4xi8>, vector<4xi8>, vector<1xi32>
```

The AMDGPU lowering tests show these map to the mixed signedness RDNA3 op:

```loom
low.op<amdgpu.v_dot4_i32_iu8.u8s8>(...)
low.op<amdgpu.v_dot4_i32_iu8.s8u8>(...)
```

Evidence:

- `loom/py/loom/dialect/vector/defs.py`
- `loom/src/loom/test/corpus/source_low/packed_dot_integer.loom-test`
- `loom/src/loom/target/arch/amdgpu/test/source_low/source_low_packed_dot.loom-test`

For nibble-packed 4-bit operands, use the packed i4 dot form:

```loom
%dot = vector.dot8i4<u4s4> %lhs, %rhs, %acc : vector<1xi32>
%dot = vector.dot8i4<s4u4> %lhs, %rhs, %acc : vector<1xi32>
```

Position:

For Q4 unsigned packed weights times signed Q8 activations, the authored source
depends on where unpacking happens. If the Q4 values are unpacked to i8 lanes
before the dot, use `vector.dot4i<u8s8>`. If the operation is still on packed
4-bit lanes, use `vector.dot8i4<u4s4>`.

## 5. Scalar packed-weight load coalescing

Question: Should Loom infer/coalesce adjacent scalar packed-weight loads, or
should authoring guidance require explicit `vector<4xi32>` loads for this
pattern?

Current:

Authoring should be explicit. Loom does not currently promise to infer a
vectorized packed-load shape from adjacent scalar loads. Source-low and lower
tests are written with the memory/register shape the author intends.

Position:

For kernels Stella is writing now, the `.loom` should load and carry packed
weights in the intended shape. If the kernel wants a `vector<4xi32>` or a
single packed `i32` register for a dot path, the source should say that
directly. This keeps compiler behavior predictable and keeps benchmark sweeps
from accidentally measuring a missing load-coalescing optimization.

Future direction:

Higher-level tiling/distribution passes may eventually own this transformation.
That is a separate optimization contract from the low-level kernel authoring
flow we are trying to make reliable now.

## 6. Compile report controls and contents

Question: What controls whether compile reports include register pressure,
spills, instruction mix, memory summaries, and scheduling data?

Current controls:

The compile-report request parser accepts:

- `none`
- `summary`
- `details`
- `json`
- `json-summary`
- `json-details`
- `text`
- `text-summary`
- `text-details`

Evidence:

- `loom/src/loom/tooling/execution/compile_report_capture.c`

Tools that expose the capture use flags such as:

```text
--compile-report=summary|details|json-details|text-details
--compile-report-output=<path|-|stderr>
--compile-report-row-limit=<N>
```

Evidence:

- `loom/src/loom/tools/loom-compile/loom-compile.c`
- `loom/src/loom/tools/iree-run-loom/main.c`
- `loom/src/loom/tools/iree-benchmark-loom/*`

Current report detail categories:

- Artifact size
- Schedule summary
- Allocation summary
- Memory summary
- Emission summary
- Register-pressure rows
- Spill rows
- Source-to-target-low selection rows
- Residual move causes
- Static instruction mix
- Target-legalization rows

Evidence:

- `loom/src/loom/target/compile_report.h`
- `loom/src/loom/target/compile_report_format.*`
- `loom/src/loom/tooling/target/amdgpu/artifact_provider.c`

Gap:

The schema is real, but the "always stable and populated for every target path"
contract is not something we should overstate yet. A details report can only
include categories the selected backend records. The guide should phrase this
as "request detailed reports and inspect populated categories", not "every
AMDGPU compile always has a complete schedule/spill/instruction breakdown".

Related debug flow:

The shared pass tracing flags are already the right shape for humans and
agents:

```text
--dump-ir-before=<pass-or-stage>
--dump-ir-after=<pass-or-stage>
--dump-ir-before-all
--dump-ir-after-all
--dump-ir-format=jsonl
--dump-ir-output=<path|dir|-|stderr>
```

`--dump-ir-output=dir/` writes a trace index plus `ir/*.loom` artifacts. JSONL
is the agent-friendly format for filtering pass events; text is the
human-friendly path with pass boundaries.

Evidence:

- `loom/src/loom/tooling/pass/trace_cli.h`
- `loom/src/loom/tooling/pass/trace_cli.c`
- `loom/src/loom/tools/loom-opt/loom-opt.test.json`
- `loom/src/loom/tools/loom-compile/loom-compile.test.json`

Action:

Before this is public, `loom-opt` and `loom-compile` should have the same
spelling and behavior for dump/tracing flags. Tool-specific compile-report
flags are fine when the tool actually compiles.

## 7. Tiny-kernel benchmark overhead and clock uncertainty

Question: Can benchmark output make tiny-kernel dispatch overhead and profile
clock uncertainty more visible?

Current:

Benchmark results already include useful raw fields:

- `batch_size`
- `warmup_batch_count`
- `warmup_duration_ns`
- `measured_batch_count`
- `measured_operation_count`
- `measured_physical_dispatch_count`
- `measured_dispatch_count`
- `measured_duration_ns`
- `stop_reason`
- `batch_timing_ns`
- `operation_timing_ns`
- timing stats with `count`, `total`, `min`, `max`, `mean`, `p50`, `p90`, and
  `p90_to_p50_delta_ppm`

Evidence:

- `loom/src/loom/tooling/execution/benchmark.h`
- `loom/src/loom/tools/iree-benchmark-loom/report.c`

Gap:

There is no explicit verdict field like `dispatch_overhead_dominates=true` or
`profile_clock_uncertainty_ns=...`. Agents can infer some of this from
`batch_size`, physical dispatch count, operation timing, and p90/p50 spread,
but the report does not currently annotate tiny-kernel measurements with a
plain warning.

One concrete buglet to fix:

`iree-benchmark-loom` writes a JSON `dispatch_timing_ns` key, but the timing
result struct has only `batch_timing` and normalized `operation_timing`.
Currently the JSON writer serializes `operation_timing` for both
`dispatch_timing_ns` and `operation_timing_ns`. That name is misleading unless
we add a real dispatch timing field. The quickest cleanup is probably to remove
or rename the misleading alias, or to populate a real dispatch-normalized stat
when physical dispatch count differs from logical operation count.

Action:

The guide should explain how to read the current fields. Product cleanup should
add an explicit tiny-kernel warning/interpretation and fix the misleading
`dispatch_timing_ns` field shape before we ask Stella to use it for decisions.

## 8. Direct AMDGPU executable emission for HRX

Question: Is there already a direct `loomc` AMDGPU executable emission path
with the artifact format HRX expects, or does llama.cpp need a small adapter
layer?

Current:

Yes, the direct HSACO path exists on this branch.

Relevant files:

- `loom/binding/c/include/loomc/target/amdgpu/base.h`
- `loom/binding/c/include/loomc/target/amdgpu/emit.h`
- `loom/binding/c/example/emit_amdgpu_offline.c`
- `loom/binding/c/example/emit_amdgpu_hsa.c`

The artifact format string is:

```c
#define LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO "amdgpu-hsaco"
```

`emit_amdgpu_offline.c` proves source/module to HSACO bytes. `emit_amdgpu_hsa.c`
mirrors the Vulkan/SPIR-V sample shape without pulling in the IREE HAL: it
dynamically loads HSA, discovers the current GPU target, asks Loom for HSACO,
loads the code object, queries the kernel symbol, and launches the simplest
kernel.

Local proof:

We have a local `.notes` proof that a Loom-produced HSACO loaded through HRX and
dispatched a `targetless_store_i32` kernel that wrote `42`. That proof should
stay local until we decide what checked test shape is appropriate.

Position:

llama.cpp/HRX needs a thin adapter for policy and metadata: selecting roots,
building cache keys, supplying specialization config, holding executable
handles, and translating Loom/HRX export metadata into the caller's dispatch
shape. It should not need a code-object conversion layer for the HSACO bytes.

## 9. Stable target/ABI metadata for llama.cpp

Question: Can Loom expose a stable target/ABI metadata record for binding
count, parameter count, constants size, workgroup size, and export name so
llama.cpp does not need to rediscover or duplicate it?

Current:

HRX has runtime export metadata for loaded executables:

```c
typedef struct hrx_executable_export_info_t {
  const char* name;
  uint32_t flags;
  uint32_t constant_byte_length;
  uint32_t binding_count;
  uint32_t parameter_count;
  uint32_t workgroup_size[3];
} hrx_executable_export_info_t;
```

Evidence:

- `libhrx/include/hrx_runtime.h`
- `libhrx/src/libhrx/executable.c`

`loomc` also has module-level query APIs:

- `loomc_module_function_export_info_t` for export symbol/ordinal
- `loomc_module_kernel_function_info_t` for static dispatch workgroup count and
  static workgroup size

Evidence:

- `loom/binding/c/include/loomc/module.h`

Gap:

There is not yet one stable compiler-produced artifact/export manifest blob
that says, for each emitted export: name, binding count, parameter count,
constants size, static workgroup size, selected target, final subgroup size,
artifact format, ABI kind, and artifact hash/provenance. HRX can answer part of
this after load, and Loom can answer part of it from module metadata before
emission, but llama.cpp should not have to rediscover or join those records
itself.

Action:

Add an artifact/export manifest product surface. It should be available through
`loomc` and probably as an optional artifact emitted beside HSACO from
`loom-compile`. This is the right place to include final wavefront size and the
dispatch ABI details HRX needs.

## 10. Keeping a value lane-varying / VGPR

Question: Is there an intended way to express "this value must remain
lane-varying / VGPR" when a dot operand would otherwise become scalar?

Current:

At low/source-low levels, register class is explicit:

```loom
reg<amdgpu.vgpr>
reg<amdgpu.sgpr>
```

AMDGPU tests use explicit low register classes heavily. Source-level code can
also create genuinely varying values from workitem/subgroup data such as
`kernel.workitem.id` and `kernel.subgroup.lane.id`.

Evidence:

- `loom/src/loom/target/arch/amdgpu/test/source_low/*`
- `loom/py/loom/dialect/kernel/builders/__init__.pyi`

Gap:

I do not see a stable source-level knob whose contract is "this value must
remain lane-varying" or "materialize this operand in a VGPR" while staying in
the higher-level dialect. Today the reliable options are:

- make the value semantically lane-varying in source, or
- author/drop to source-low/low where the register class is explicit.

Action:

If Stella hits a scalarized dot operand that must be VGPR for ISA selection or
performance, that is a real feature request. The right fix is probably a
source-level assertion/fence or operand-placement contract that fails loudly if
the requested variability/register class cannot be preserved.

## 11. Compile benchmark scope

Question: Does the reported sub-microsecond compile benchmark include only Loom
source/IR compilation, or also AMDGPU executable emission and loader-ready
artifact production?

Correction:

The intended claim is milliseconds, not microseconds.

Current:

The checked targetless compile throughput benchmark in
`loom/binding/c/benchmark/targetless_compile_throughput_benchmark.cc` requests
`LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE` and validates
`LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE`. That means it exercises Loom
source/module compilation, linking/config specialization/pass execution, and
module-bytecode artifact production. It does not measure AMDGPU HSACO emission
or HSA/HRX loading.

The AMDGPU examples exercise source/module to HSACO emission:

- `loom/binding/c/example/emit_amdgpu_offline.c`
- `loom/binding/c/example/emit_amdgpu_hsa.c`

Position:

When quoting compile-time numbers to Stella, name the artifact boundary:

- text/source to transformed Loom module/bytecode
- text/source to HSACO bytes
- text/source to HSACO bytes loaded into HSA/HRX
- cache hit dispatch path

Those are different products from a latency perspective. The guide should not
blur them.

## 12. Safe and cheap `loomc` use on the llama.cpp inference path

Question: What parts of `loomc` are safe and cheap enough to call on the
llama.cpp inference path, and which should be isolated to background tuning or
cache warmup?

Current API contract:

`loomc_compile_module` returns result-owned artifacts that do not borrow from
the invocation workspace. Prepared compilers and pass programs are immutable.
Concurrent compile calls are supported when each call has distinct workspace and
module storage, or shared access is synchronized externally.

Evidence:

- `loom/binding/c/include/loomc/compile.h`

Position:

The hot inference path should dispatch cached executables. Shape/cache miss
handling can call Loom, but it should be staged and memoized:

1. Create process/device/model-lifetime objects once: `loomc_context_t`,
   target environment, compiler, pass program, linked library/index as
   appropriate.
2. On a new shape/config/root, compile in warmup or on a cache-miss path.
3. Use a cache key that includes the authored source/bytecode hash, linked
   root, config values, target processor, subgroup size, artifact format, ABI,
   Loom version/tool options, and relevant pass program identity.
4. If compile output is deterministic, hash the result before loading. If the
   HSACO or manifest matches an already loaded executable, discard the duplicate
   artifact and reuse the existing handle.
5. Load HSACO into HSA/HRX only on misses. The token-by-token dispatch path
   should see an already loaded executable and precomputed export metadata.

The practical answer is that all of Loom can be safe on an inference miss path
at the right granularity. None of it should run every inference step for a
shape/config that has already been seen.

## Immediate product gaps for the guide sprint

High impact before Stella's first serious pass:

- Implement `target(@...)` on `func.template` and `func.ukernel`, with early
  off-target pruning. Tracked as `loom-pdqzn`.
- Add or expose an artifact/export manifest that joins compiler-side and
  loader-side metadata for HRX/llama.cpp.
- Clean up benchmark JSON around `dispatch_timing_ns` and add tiny-kernel
  interpretive warnings.
- Decide whether a source-level lane-varying/VGPR assertion is needed now or
  whether source-low authoring guidance is sufficient for Stella's current
  kernels.
- Keep the HSA examples free of IREE HAL dependencies. The direct HSA example
  should remain the proof that `loomc` does not mandate IREE HAL.

Guide material to merge after implementation:

- Provider selection: document `priority`, `where`, and `target(@...)` together.
- Targeting: document `amdgpu.target<...> @name {subgroup_size = ...}` and how
  a JIT target selection maps to providers.
- Tuning: lead with one-root/one-candidate artifacts.
- Debugging: put `--dump-ir-*`, JSONL traces, compile reports, and benchmark
  report interpretation near every authoring flow, not as an appendix.
- HRX integration: describe the adapter layer as metadata/cache/dispatch policy,
  not code-object conversion.
