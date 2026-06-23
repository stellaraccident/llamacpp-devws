# AMD RDNA3 ISA and Wavefront Gotchas

Scope: lessons from optimizing HRX pure-HIP llama.cpp kernels on Radeon Pro
W7900, `gfx1100`, RDNA3. This is not a general AMD GPU manual. It records the
failure modes that repeatedly mattered for LLM inference kernels.

## Index

- [A. First Principles](#a-first-principles)
- [B. Wave32 vs Wave64](#b-wave32-vs-wave64)
- [C. Workgroup Size Is Not Wavefront Size](#c-workgroup-size-is-not-wavefront-size)
- [D. WMMA, MFMA, and Integer Dot](#d-wmma-mfma-and-integer-dot)
- [E. Reductions and Lane Broadcasts](#e-reductions-and-lane-broadcasts)
- [F. Register Pressure, LDS, and Occupancy](#f-register-pressure-lds-and-occupancy)
- [G. Load Scheduling and Waitcnt](#g-load-scheduling-and-waitcnt)
- [H. Compile Flags](#h-compile-flags)
- [I. Numerics](#i-numerics)
- [J. Inspection Commands](#j-inspection-commands)
- [K. Current Decisions](#k-current-decisions)

## A. First Principles

RDNA3 performance problems are often schedule/dataflow problems, not missing
one instruction. The same source-level idea can win in one kernel and lose in a
neighboring kernel because it changes:

- actual wavefront size;
- VGPR lifetime;
- LDS allocation and resident workgroups;
- barrier cadence;
- scalar setup dependencies;
- LLVM's ability to schedule global loads before `s_waitcnt`;
- whether a reduction result is present in one lane or all lanes.

Use profiler evidence. Do not rely on source similarity to Vulkan alone.

## B. Wave32 vs Wave64

RDNA supports both. HIP may compile a kernel as wave32 unless explicitly told
otherwise. Vulkan may request subgroup size per pipeline. A kernel written with
a logical `WARP=64` can silently run as two wave32s unless compiled for wave64.

### Wave64 wins observed

- **Q5 prompt large MMQL**: uses the Vulkan-style large K-quant tile and should
  stay wave64. Removing `-mwavefrontsize64` regressed the Q5 bucket.
- **Q6 prompt large MMQL**: accepted large prompt path is wave64. Q6 also gained
  from source schedule changes, but wave64 remains part of the intended shape.
- **Q4 prompt MoE route-tiled MMQ**: the kernel constants were written around a
  64-lane schedule. Compiling the object as wave64 improved both ID and SWIGLU
  route-tiled buckets. This is the clearest "the source assumed wave64" case.

### Wave32 wins observed

- **Gated Delta Net**: both prompt and decode schedules should stay wave32 on
  this card. Wave64 regressed GDN and introduced spill/pressure problems in the
  generic object.
- **Decode Q6 DMMV**: the accepted `rows2_cols1_wg32` path follows Vulkan's
  one-subgroup DMMV shape for the active direct F32-RHS decode route.
- **Decode dense F32/BF16 exact providers**: the accepted skinny decode kernels
  use wave32-style row shards and subgroup reductions.
- **TopK decode**: current provider is wave32. A rows1 variant accidentally
  compiled wave64 first, then correctly wave32; neither fixed the gap, but
  wave64 was not the answer.

### Rule

Choose wavefront size per kernel family:

- large prompt integer-dot MMQ: usually wave64;
- skinny decode DMMV and small reductions: usually wave32;
- GDN on this HIP schedule: wave32;
- FA gfx11 direct: inspect the emitted `v_wmma` and resource metadata; do not
  infer from workgroup size.

## C. Workgroup Size Is Not Wavefront Size

`wg64` can mean:

- one wave64, if compiled wave64;
- two wave32 waves, if compiled wave32.

The distinction matters for shared reductions and LDS partial sizing. A kernel
can be named `wg64` because it launches 64 threads while still executing as two
wave32s. Always confirm `.wavefront_size` in metadata.

Specific failure mode:

- Q4 prompt MoE route-tiled kernels had `WARP=64` schedule assumptions but the
  default gfx11 object reported `.wavefront_size: 32`. Compiling wave64 fixed
  the mismatch and improved device time without changing VGPR counts.

## D. WMMA, MFMA, and Integer Dot

### RDNA3 uses WMMA, not CDNA MFMA

On `gfx1100`, the matrix primitive to look for is usually:

- `v_wmma_f32_16x16x16_f16`
- `v_wmma_f16_16x16x16_f16`
- BF16 WMMA forms where applicable

Do not expect CDNA-style `v_mfma*` in RDNA3 kernels.

### Q5/Q6 Q8_1 prompt MMQ is integer dot, not coopmat

Vulkan's Q8_1 K-quant MMQ path for Q5/Q6 uses SPIR-V integer dot:

- `SPV_KHR_integer_dot_product`
- `OpSDot ... PackedVectorFormat4x8Bit`

The matching HRX HIP primitive is `v_dot*` from builtins such as
`__builtin_amdgcn_sudot4`. It is not missing WMMA/MFMA.

The coopmat Vulkan shaders exist for a separate F16/BF16 matmul family. Do not
conflate those with the Q8_1 MMQ route.

### BF16 prompt uses WMMA

The accepted BF16 prompt routes deliberately round RHS to BF16 and use WMMA.
They are approximate relative to CPU/F32 accumulation and must remain under the
fast-approx rollback policy.

### Flash attention uses WMMA, but the layout is dangerous

The gfx11 direct FA path taught the main WMMA lesson:

- rocWMMA fragment APIs were useful for bringup but hid layout costs.
- The hand-coded gfx11 path won by keeping output accumulators resident and
  avoiding LDS redistribution.
- F32 WMMA accumulator lane layout is even/odd interleaved. The p512 repair used
  a row mapping like `row = (lane >> 4) + 2 * i`, not a simple contiguous row
  split.
- Any change to QK/PV accumulator type, row mapping, or store path requires
  exact `FLASH_ATTN_EXT` tests at the model shape and loop guards.

Do not guess WMMA fragment layout. Treat it as an ISA contract that must be
resolved exactly from a controlled fixture, disassembly, and output-coordinate
diffs. The bug cluster here came from assuming row-major accumulator storage
where gfx11 used an interleaved lane layout. A plausible-looking store mapping
can produce coherent short text and still corrupt long generations. Before
optimizing or reusing a WMMA fragment path, prove the lane-to-matrix coordinate
mapping for the exact instruction, accumulator type, wave mode, and store path.

## E. Reductions and Lane Broadcasts

Down-shuffle reductions usually leave the final value in one lane. Many kernels
need the reduced value in every lane:

- GDN needs the column sum in every lane to update that lane's state shard.
- Softmax/top-k often needs max/sum values broadcast back to all participating
  lanes.

The bug pattern:

1. Replace LDS reduction with shuffle reduction.
2. Exact small fixture passes or partly passes because only lane 0 writes.
3. Real model-shape fails because other lanes consume stale partial sums.

Always ask: does every lane need the result, or only the store lane?

For GDN, the winning shape used an explicit cluster reduction plus broadcast.
A DPP row-shift reduction was useful; a DPP row-broadcast replacement regressed.

## F. Register Pressure, LDS, and Occupancy

### VGPR is a tradeoff, not a scalar score

Examples:

- Packed Q4 SWIGLU decode jumped to high VGPR but won because it preserved RHS
  reuse and reduced device time.
- Low-VGPR Q4 SWIGLU split the gate/up passes, duplicated RHS reads, raised
  instruction/load count, and lost.
- Q5 large MMQL won with high VGPR and wave64.
- Q6 fill-order won even with a few VGPR spills. The profiler decided.

Do not accept or reject based only on VGPR count. Use:

- rocprof bucket delta;
- spill/private segment;
- LDS size;
- ATT wait/barrier changes;
- wall guardrails.

### LDS can be the occupancy limiter

Q5 prompt MMQL improved by reducing `BK_STEP` from 4 to 1. This cut LDS from
about 40 KiB to about 10 KiB and improved residency while keeping the output
tile. The same `BK_STEP=1` idea regressed Q6, which benefits from deeper K
staging.

Rule: LDS-depth tuning is dtype/kernel-specific. Do not port it blindly.

### No spill does not mean no pressure

Several rejected variants had no private memory but still regressed because
VGPR pressure reduced occupancy or constrained scheduling.

## G. Load Scheduling and Waitcnt

ATT often attributed most stalls to `s_waitcnt`, not to arithmetic:

- Q4 route-tiled MMQ: `s_waitcnt vmcnt` immediately after scalar Q4/Q8 loads
  into LDS.
- Q6 direct pack: `s_waitcnt vmcnt` around packed Q6 global loads.
- Tiny GET_ROWS: `s_waitcnt lgkmcnt` on scalar kernarg setup and one global
  load; not worth local kernel tuning.

Useful responses:

- batch load/fetch and commit to LDS instead of immediate load/wait/store;
- hoist independent address arithmetic before waits;
- reduce scalar kernarg and dynamic stride/modulo work with exact shape gates;
- vectorize/coalesce only if the resulting extraction/VGPR pressure does not
  erase the win.

Unhelpful responses seen:

- global `float4` RHS loads across K-quant kernels;
- aligned packed loads in kernels where extraction pressure dominated;
- "prefetch first" Q4 SWIGLU schedule that worsened ATT global-load stalls.

## H. Compile Flags

### Per-kernel flags are valid

Broad policy changes were weak. Per-source compile flags were useful:

- Q5 prompt large MMQL: device `-O3` accepted.
- Q6 prompt large MMQL: later `-O3` accepted after other source changes.
- Global/speculative `-O3` was not a general answer and was removed earlier.

### Flags that often lost

- `-ffast-math` for integer MMQ kernels: Q5 regressed.
- broad `__restrict__`: no stable win.
- `__launch_bounds__(256, 2)`: raised pressure or slowed Q5.
- forcing wave64 for GDN: regressed.
- removing wave64 for Q5 prompt MMQL: regressed.

### Verify final CMake behavior

The standalone ISA helper can lie by omission if it does not reproduce CMake
per-source flags. Inspect built HSACOs for final wavefront/resource facts.

## I. Numerics

### Exact vs approximate

CPU exactness is required for conservative routes. It is not always a useful
contract for fast prompt routes that intentionally match GPU-style
approximations:

- Q6 prompt Q8_1 x4 approximate route can fail CPU exact matmul by large local
  error but still be model-stable under chat/loop guard.
- BF16 WMMA prompt intentionally rounds RHS to BF16.
- FA exactness must remain strict enough to catch accumulator layout bugs.

### Mandatory rollback

Approximate prompt paths must be covered by:

- global rollback: `GGML_HRX_DISABLE_FAST_APPROX_PROMPT=1`;
- targeted rollback for the provider family;
- exact conservative tests under the global rollback;
- default chat and loop guards without the rollback.

### Long-generation instability

Prompt MoE routing bugs can poison hidden state and only appear after O(100+)
decode tokens. The ARGSORT prompt issue was found this way. Do not rely only on
single-op exact checks.

## J. Inspection Commands

Wavefront/resources:

```bash
"$ROCM_PATH/lib/llvm/bin/llvm-readobj" --notes kernel.hsaco \
  | rg 'name:|wavefront_size|vgpr_count|sgpr_count|spill|group_segment|private'
```

Instructions:

```bash
"$ROCM_PATH/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx1100 kernel.hsaco \
  | rg 'v_wmma|v_mfma|v_dot|s_waitcnt|s_barrier|global_load|ds_'
```

HRX helper:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-kernel-isa-summary.py \
  --kernel 'q5_k_q8_1|q6_k_q8_1|q4_k.*mmq|gated_delta|flash_attn' \
  --out-dir build/isa-check \
  --json build/isa-check/summary.json
```

ATT summary:

```bash
sources/llama.cpp/tools/hrx-epic2/hrx-att-summary.py \
  build/rocprof-att-*/stats_ui_output_agent_*_dispatch_*.csv \
  --top 40
```

## K. Current Decisions

Current RDNA3 decisions from the spike:

| Kernel family | Current wave/ISA decision |
| --- | --- |
| Q5 prompt MMQL | wave64, Q8_1 x4, `v_dot`, high tile, Q5-specific `-O3` |
| Q6 prompt MMQL | wave64, Q8_1 x4, `v_dot`, `128x64`, deeper K staging |
| Q4 prompt MoE MMQ | wave64 compile because schedule assumes 64 lanes |
| BF16 prompt dense/SWIGLU | wave32 WMMA16, approximate BF16 RHS |
| FA prompt gfx11 direct | hand-coded `v_wmma`, exact lane mapping required |
| GDN prompt/decode | wave32, clustered reductions/broadcasts |
| Q6 decode DMMV | wave32 one-subgroup/two-row direct F32-RHS |
| Q5 decode DMMV | current wg128/wg64 split, not Q6 rows2 transplant |
| TopK decode | wave32, still behind Vulkan; needs ISA/ATT, not launch-only tweaks |
| GET_ROWS nr1 | do not body-tune; fuse/eliminate dispatches |

Search terms: `wavefront_size`, `mwavefrontsize64`, `v_wmma`, `v_dot`,
`s_waitcnt`, `VGPR`, `group_segment_fixed_size`, `GGML_HRX_DISABLE_FAST_APPROX_PROMPT`.
