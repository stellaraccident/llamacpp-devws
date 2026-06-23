# HRX v1 Q8_0 Prompt Schedule Ledger

Date: 2026-06-17

## Target

Active gap: Llama 3.1 8B Q8_0 p512 prefill on gfx1151.

Focused exported rows from:
`cache/hrxv1/gfx1151/q8_0-current-focused-p512-20260617-184658/focused/q8_0_prompt.txt`

Hot rows:

- `ffn_gate-0`: Q8_0 `[4096,14336]` x F32 `[4096,512]` -> F32 `[14336,512]`
- `ffn_out-0`: Q8_0 `[14336,4096]` x F32 `[14336,512]` -> F32 `[4096,512]`
- `Qcur-0`: Q8_0 `[4096,4096]` x F32 `[4096,512]` -> F32 `[4096,512]`

The historical p512 basket row was far behind same-machine Vulkan, with Q8_0 at
about `0.236x` Vulkan in
`cache/hrxv1/gfx1151/current-best-q6id-basket-p512-fa1-r1-20260617-184515/`.
The direct-F32 VK128 hybrid checkpoint improved the old/default Q8_0 path, but
the later current-best comparison rejected it against the stronger BN96/BN64
packed-Q8_1 policy. Current-best Q8_0 p512/fa1 remains the BN96/BN64 route,
with the latest basket row at `422.153 tok/s` versus Vulkan `903.967 tok/s`,
or about `0.467x` Vulkan.

## Rejected VK128 Direct-F32 Hybrid Checkpoint

Route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_F16ACC_WG256_PROMPT=1`

Schedule facts:

- Anchored to Vulkan oracle `matmul_q8_0_f32_f16acc_aligned_l`.
- Workgroup/tile: WG256, BM128, BN128, BK32.
- Wave mode: wave64.
- A path: Q8_0 dequantized into f16 LDS.
- B path: F32 RHS cast into f16 LDS.
- LDS: padded 40-half stride for both A and B, `20480` bytes total.
- Opcode: `v_wmma_f16_16x16x16_f16`, 32 static sites.
- Metadata: SGPR 22, VGPR 129, no spills.

Selector:

- The route is not globally better. It regresses Vcur/Qcur.
- Final opt-in selector therefore requires
  `cols >= 128 && (rows >= 8192 || k >= 8192)`.
- Smaller and odd Vcur rows stay on packed Q8_1 x4.

Evidence:

- Focused artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma16-vk128-padded-w64-focused-20260618-002300/`.
- Model A/B artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma16-vk128-hybrid-model-ab-20260618-002821/`.
- Focused p512 summed row time:
  `267564.84 -> 119227.42 us`.
- Same-binary model p512/fa1:
  `182.704 -> 280.436 tok/s`.
- Same-machine Vulkan p512/fa1:
  `903.038 tok/s`.

Current-best rejection:

- Focused BN96/BN64 comparison:
  `cache/hrxv1/gfx1151/q8_0-vk128-vs-bn96-focused-20260618-003553/`.
- Basket comparison:
  `cache/hrxv1/gfx1151/current-best-q8vk128hybrid-basket-p512-fa1-r1-20260618-003443/`.
- Focused p512 rows all regressed versus BN96/BN64:
  `Vcur 651.63 -> 682.40 us`, `Qcur 2026.25 -> 2069.85 us`,
  `ffn_out 8269.05 -> 9696.54 us`, `ffn_gate 7371.01 -> 10308.36 us`,
  and `result_output 66339.24 -> 92056.48 us`.
- Basket geomean regressed `0.5354x -> 0.5212x` Vulkan, and the Q8_0 row
  regressed `422.15 -> 356.70 tok/s`.

Decision:
reject for current-best promotion and keep as an opt-in diagnostic only. The
remaining Q8_0 gap is not solved by a direct-F32 WMMA clone of the Vulkan
128x128 family; future Q8 work needs to preserve packed-Q8_1 benefits or prove
a lower-level RADV-like cooperative-matrix schedule that beats BN96/BN64.

## Actual Vulkan p512 Oracle Correction

The Llama 3.1 8B Q8_0 p512 oracle capture shows that the dominant Vulkan
prefill route is not the integer `matmul_q8_0_q8_1` MMQ path. It is:

- Pipeline: `matmul_q8_0_f32_f16acc_aligned_l`
- Hash: `0x72d309e22f889977`
- Spec: `[256,128,128,32,64,64,2,16,16,16,64]`
- Workgroup denominators: `[128,128,1]`
- Top workgroups: `[112,4,1]` for FFN gate/up, `[32,4,1]` for Q/attn/out,
  `[8,4,1]` for K/V.
- RADV stats: `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills, `3740`
  instructions, `204` VMEM, `102` SMEM.

This does not invalidate the accepted BN96/BN64 packed-Q8_1 policy, because
that policy is still the best HRX route measured so far. It does mean Q8_0
parity work should mechanically compare against the f16acc aligned-large
Vulkan shader first, and only use the integer MMQ Vulkan path as a separate
prior family.

## Q8_0 Odd/Tail Vulkan Oracle

Odd and one-past-production prompt captures are now available for the same
Llama 3.1 8B Q8_0 model:

- p33 artifact:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p33-fa1-20260618-194300/`
- p513 artifact:
  `cache/hrxv1/gfx1151/vulkan-oracle-llama31-8b-q8_0-p513-fa1-20260618-193949/`

The p33 route is the medium aligned pipeline:

- Pipeline: `matmul_q8_0_f32_f16acc_aligned_m`
- Hash: `0x72d309e22f889977`
- Spec: `[128,64,64,32,64,32,2,16,16,16,64]`
- Workgroup denominators: `[64,64,1]`
- Dense Q8_0 dispatches: `221`
- Representative FFN workgroups: `[224,1,1]` for
  q8_0 `[4096,14336]` x f32 `[4096,33]` -> f32 `[14336,33]`
- RADV stats: `SGPR=108`, `VGPR=144`, `LDS=11264`, no spills,
  `2077` instructions, `108` VMEM, `54` SMEM.

The p513 route returns to the large aligned pipeline:

- Pipeline: `matmul_q8_0_f32_f16acc_aligned_l`
- Hash: `0x72d309e22f889977`
- Spec: `[256,128,128,32,64,64,2,16,16,16,64]`
- Workgroup denominators: `[128,128,1]`
- Dense Q8_0 dispatches: `221`
- Representative FFN workgroups: `[112,5,1]`; the fifth column workgroup
  covers the 513-token tail.
- RADV stats: `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills,
  `3740` instructions, `204` VMEM, `102` SMEM.

This matches the Q4_K odd/tail policy: p33 is a separate medium/narrow regime,
while p513 is production-width large aligned with an edge workgroup. Do not
promote a Q8_0 large route from p512 evidence alone; it must preserve p33 on a
medium/narrow path and pass p513 with the large route active.

## RADV vs HIP VK128 ISA Delta

Artifacts:

- Base comparison:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-isa-20260618/`
- H4LOAD comparison:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-h4load-isa-20260618/`

Static comparison against the existing HIP
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_f16acc_wg256_f32`:

- Both routes emit `32` `v_wmma_f16_16x16x16_f16` sites and `2` barriers.
- RADV uses `LDS=22528`, `VGPR=192`, `SGPR=108`, no spills.
- HIP uses `LDS=20480`, `VGPR=129`, `SGPR=22`, no spills.
- RADV emits `64 ds_load_b64`, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  `192 buffer_store_b32`, `169 s_waitcnt`, and `102` scalar memory loads.
- HIP emits `32 ds_load_b128`, `2 ds_store_b16`, `64 global_store_b32`,
  `19 s_waitcnt`, `126 s_waitcnt_depctr`, and only `2` scalar memory loads.

Interpretation:
the existing HIP VK128 source matches the headline tile and WMMA count but not
the emitted schedule. RADV is carrying substantially more accumulator and LDS
state and stages/stores through a different cooperative-matrix path. A HIP
candidate that only restates BM128/BN128/BK32/WG256/W64 is not an exact Vulkan
schedule clone.

The Vulkan shader source narrows the store-side delta. In
`ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`, full in-bounds aligned
tiles use `coopMatStore` directly to `data_d`; the LDS `coopmat_stage` path is
used for unaligned stride or partial edge tiles. For production p512 and most
p513 tiles, the schedule to clone is therefore not "stage accumulator to LDS
and then scalar-store." It is cooperative-matrix global store/lane ownership.
The HIP direct-F32 probes still scalarize this store path into far fewer global
stores (`64 global_store_b32` in the current HIP direct route versus RADV's
`192 buffer_store_b32`) and therefore remain schedule diagnostics, not exact
clones.

## Rejected H4LOAD Compile-Equivalent Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_h4load_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_H4LOAD_F16ACC_WG256_PROMPT=1`

What changed:

- Kept the rejected VK128 wave64 direct-F32 WMMA dataflow.
- Changed A/B fragment reads from scalar half indexing to half4 vector reads,
  attempting to move the emitted LDS access pattern toward the RADV
  `ds_load_b64`/cooperative-matrix-load shape.

Compile evidence:

- Built through CMake/Ninja:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_h4load_wg256.hsaco`
- Metadata: `wavefront_size=64`, `SGPR=22`, `VGPR=129`, `LDS=20480`,
  no spills.
- Static ISA was identical to the base rejected VK128 route for the key
  schedule facts: `32 ds_load_b128`, `32` WMMA sites, `64` stores, `2`
  barriers.

Decision:
reject at the compile-evidence gate without focused timing. The source change
does not change the emitted schedule, so it cannot close the RADV gap.

## First-WMMA Issue-Window Probes

The dependency-pinned fixture proved the target scheduling primitive is
expressible in isolation: `64` pre-WMMA `ds_load_b64`, `64` loads immediately
before final `lgkmcnt(51)`, and zero load-like ops after the first WMMA.

Real Q8_0 route transplants now bracket the source-visible HIP C++ limit:

- one-K-tile depwait:
  `cache/hrxv1/gfx1151/q8_0-depwait-realfrag-compile-20260618-174000/`
  preserved final `lgkmcnt(51)` but scored only `32/32/lgkmcnt(51)` with
  VGPR `240`.
- two-K-tile depwait:
  `cache/hrxv1/gfx1151/q8_0-depwait-k2-realfrag-compile-20260618-172158/`
  improved to `57` pre-WMMA `ds_load_b64`, `31` loads immediately before the
  final wait, final `lgkmcnt(51)`, and `20` WMMAs in the first window, but
  compiled at VGPR `256` with `30` VGPR spills and private segment `124`.
- two-K-tile depwait without explicit preuse:
  `cache/hrxv1/gfx1151/q8_0-depwait-k2-nopreuse-compile-20260618-continued/`
  emitted the same tracked contract as K2: wave64, VGPR `256`, `30` VGPR
  spills, private segment `124`, `57/31/lgkmcnt(51)`, and `20` WMMAs in the
  first window.
- two-K-tile direct wait without dependency copies:
  `cache/hrxv1/gfx1151/q8_0-k2-directwait-compile-20260618-185220/`
  removed the spill cliff, compiling at VGPR `196`, no spills, and private
  segment `0`, but the first-window schedule collapsed to
  `24/0/lgkmcnt(0)` with only `16` WMMAs in the window.

Decision:
reject both route transplants at the compile-contract gate. K2 gets close to
RADV's `59/59/lgkmcnt(51)` first-window signature, but the source-visible HIP
C++ fragment-retention strategy creates an unacceptable register cliff and
still misses the RADV halfword LDS/cooperative writeback topology. The next
Q8_0 route should not be another "keep more real fragments live" variant. The
no-preuse bracket shows the cliff is not caused by the explicit empty preuse
dependency; it is caused by exposing both K tiles' real fragments in this
source-level shape. The direct-wait bracket shows the dependency-copy machinery
is what preserves the wait window and causes the spill cliff; ordinary explicit
waits remove the pressure but do not survive scheduling as a RADV-like issue
window.

## Real-Fragment Issue-Window Microbench Probe

Artifact:
`cache/hrxv1/gfx1151/q8_0-wmma-issue-window-realfrag-probe-20260618-231525/`

Follow-up artifact:
`cache/hrxv1/gfx1151/q8_0-wmma-issue-window-fullacc-isolate-20260618-232442/`

Selected-lane artifact:
`cache/hrxv1/gfx1151/q8_0-wmma-selected-lane-validity-20260618-232944/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/wmma_issue_window_bench.hip.cpp`

What changed:

- Added CMake/Ninja-built `realfrag16` and `realfrag8` modes to the existing
  WMMA issue-window fixture.
- `realfrag16` loads four A fragments and four B fragments from LDS with
  `ds_read_b64`, then issues the same 4x4, 16-WMMA dependency-pinned pattern
  as the one-K-tile route transplant.
- `realfrag8` keeps the same dependency-pinned idea but halves the accumulator
  and column footprint to two B fragments and eight WMMAs.

Runtime result:

- Both modes execute but produce NaNs (`realfrag16` and `realfrag8` each
  reported 12 NaNs in the sampled output). They are compile/schedule probes,
  not correctness candidates.
- A follow-up added direct, non-dependency-copy real-fragment modes and forced
  all accumulator lanes to distinct output slots. With full accumulator
  materialization, all real-fragment variants exposed NaNs:
  `realfrag16` 113, `realfrag8` 91, `realfrag16-direct` 68, and
  `realfrag8-direct` 118 NaNs out of 8192 sampled values. The original
  constant-fragment `lgkm51` and `wait0` modes remained finite.
- A selected-lane follow-up separated valid production-selected even slots from
  nonselected odd accumulator halves. `wmma-f16-lane-map --mode=fulltile-ones`
  passed with zero NaNs and zero expected-value mismatches. The LDS-fragment
  full-tile lane-map had 66 NaNs, all in odd slots. The 16-WMMA issue-window
  variants also had only odd-slot NaNs (`realfrag16`: 111 odd, 0 even;
  `realfrag16-direct`: 72 odd, 0 even). The reduced 8-WMMA variants were not
  clean on selected slots (`realfrag8`: 16 even NaNs; `realfrag8-direct`:
  80 even NaNs).

Static result:

- Dependency-pinned `realfrag16`: `32` total `ds_load_b64`, `16` WMMA, final
  pre-WMMA `lgkmcnt(51)`, `32` loads immediately before the final wait,
  `VGPR=134`, `SGPR=8`, no spills, private segment `0`.
- Dependency-pinned `realfrag8`: `24` total `ds_load_b64`, `8` WMMA, final
  pre-WMMA `lgkmcnt(51)`, `24` loads immediately before the final wait,
  `VGPR=82`, `SGPR=8`, no spills, private segment `0`.
- Direct `realfrag16-direct`: `32` total `ds_load_b64`, `16` WMMA, final
  pre-WMMA `lgkmcnt(0)`, no loads immediately before the final wait,
  `VGPR=101`, `SGPR=8`, no spills, private segment `0`.
- Direct `realfrag8-direct`: `24` total `ds_load_b64`, `8` WMMA, final
  pre-WMMA `lgkmcnt(0)`, no loads immediately before the final wait,
  `VGPR=69`, `SGPR=8`, no spills, private segment `0`.

Interpretation:
the dependency-pinned trick is useful as a compiler scheduler lever, but the
full accumulator dump does not prove a valid writeback contract because odd
accumulator halves are not production-selected when `op_sel=false`. The
selected-lane fixture changes the conclusion: the full 4x4, 16-WMMA real
fragment shape has finite selected even slots, while the reduced 8-WMMA shape
does not. Direct real fragments collapse to `lgkmcnt(0)`, while
dependency-pinned real fragments keep `lgkmcnt(51)` at higher VGPR. The next
useful Q8_0 probe is a selected-lane production-store route or fixture for the
full 4x4 tile only; do not use the reduced 8-WMMA path, and do not judge
correctness by storing nonselected accumulator halves.

## Buffer-Store Descriptor Fix

The raw buffer-store axis had a real gfx1151 descriptor bug. The inherited
descriptor word `0x27000` is not valid for this path on gfx1151: the
CMake/Ninja-built `hrx-hip-bench-raw-buffer-store` fixture showed it effectively
does not write. Tensile's gfx11 `BUFFER_RESOURCE_3RD_DWORD` value,
`0x31004000`, passed both compiler-built and manual raw buffer descriptors.

Artifacts:

- Raw-store fixture:
  `cache/hrxv1/gfx1151/raw-buffer-store-fixture-20260618/`
- Focused route A/B:
  `cache/hrxv1/gfx1151/q8_0-bufferstore-gfx11-rsrc-focused-20260618/`

Production-shaped fixture checks passed for `4096 x 512`, `14336 x 512`, and
`128256 x 512` outputs. After patching the Q8_0 VK128 buffer-store route to use
`0x31004000`, focused CPU-reference gates passed for p512 and p513 selected
large rows; p33 stayed on the existing narrow packed route and passed.

Timing still rejects the route versus the current packed-Q8_1 default:

- p512: `ffn_out 7048.33 -> 10357.40 us`,
  `ffn_gate 6358.83 -> 10973.75 us`,
  `result_output 56008.26 -> 98015.71 us`.
- p513: `ffn_out 7611.67 -> 10779.03 us`,
  `ffn_gate 6875.18 -> 13367.04 us`,
  `result_output 62863.76 -> 111411.43 us`.

Decision:
keep the descriptor fix and fixture as a useful exact-schedule primitive, but
do not promote the buffer-store route. Correct `buffer_store_b32` emission is
not sufficient; the remaining work is to reproduce RADV's cooperative-matrix
load/store lane ownership and halfword LDS topology much more exactly.

## Live-WMMA Mixed Store Contract Probe

Artifact:
`cache/hrxv1/gfx1151/coopstore-live-wmma-mixed-probe-20260619-071800/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/coopmat_store_contract_bench.hip.cpp`

What changed:

- extended the CMake/Ninja-built `hrx-hip-bench-coopmat-store-contract`
  fixture with `wmma-radv-mixed96` and `wmma-radv-mixed192` modes;
- direct-store groups now consume real f16 WMMA accumulator values instead of
  synthetic scalar values;
- remaining groups use the existing halfword LDS `ds_write_b16` /
  `ds_read_u16_d16` stage followed by raw `buffer_store_b32`.

Runtime result:

- `wmma-radv-mixed96`: `elements=12288 bad=0 max_abs=0`;
- `wmma-radv-mixed192`: `elements=12288 bad=0 max_abs=0`.

Static result:

- `wmma-radv-mixed96`: wave64, SGPR `26`, VGPR `59`, LDS `8192`, no private
  segment, no spills, `8` WMMA, `96` `buffer_store_b32`,
  `64` `ds_load_u16_d16`/`ds_read_u16_d16`, `64` `ds_store_b16`, `2`
  barriers, and `71` waits.
- `wmma-radv-mixed192`: wave64, SGPR `26`, VGPR `92`, LDS `16384`, no private
  segment, no spills, `16` WMMA, `192` `buffer_store_b32`,
  `128` `ds_load_u16_d16`/`ds_read_u16_d16`, `128` `ds_store_b16`, `2`
  barriers, and `135` waits.

Interpretation:
this retires one narrower blocker: the RADV-like mixed raw-buffer/halfword
store surface can be connected to live WMMA accumulator values without
immediate lane-map corruption or a register cliff in a standalone fixture. It
is still not a production route, because the fixture uses synthetic A/B
fragments and does not reproduce the real Q8/Q5 dequant, LDS load window, or
catalog ABI. The next direct-WMMA production attempt must combine this
selected-half live-accumulator store contract with the real fragment load path
and compare the emitted first-WMMA issue window against RADV before model-level
promotion.

## LDS-Fragment Live-WMMA Mixed Store Probe

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-live-wmma-mixed-probe-20260619-072609/`

What changed:

- extended the same CMake/Ninja-built fixture with `wmma-lds-radv-mixed96`
  and `wmma-lds-radv-mixed192` modes;
- A/B fragments are initialized in LDS and consumed through `ds_read_b64`
  before WMMA;
- the medium mode computes 16 WMMA groups, direct-stores groups `0..7`, stages
  live accumulator groups `8..15` through halfword LDS, and stages synthetic
  groups `16..23` as a control;
- the large mode direct-stores live WMMA groups `0..15` and keeps staged
  groups synthetic.

Runtime result:

- `wmma-lds-radv-mixed192` passed: `elements=12288 bad=0 max_abs=0`;
- `wmma-lds-radv-mixed96` failed at the live-accumulator halfword stage:
  `elements=12288 bad=1984 max_abs=256 first_bad=2048 actual=2.005
  expected=48`.

Static result:

- `wmma-lds-radv-mixed96`: wave64, SGPR `14`, VGPR `95`, LDS `24576`, no
  spills, `32` `ds_load_b64`, `16` WMMA, `96` `buffer_store_b32`, `64`
  halfword LDS stores/loads, `3` barriers, and `72` waits.
- `wmma-lds-radv-mixed192`: wave64, SGPR `14`, VGPR `103`, LDS `32768`, no
  spills, `32` `ds_load_b64`, `16` WMMA, `192` `buffer_store_b32`, `128`
  halfword LDS stores/loads, `3` barriers, and `136` waits.

Interpretation:
this narrows the direct-WMMA blocker again. LDS fragment loads plus live WMMA
plus direct raw buffer stores can coexist, because the large direct-store
control is correct. The failure is specifically the path that writes live WMMA
accumulator values to halfword LDS and reads them back for raw F32 stores. Do
not build another production route around the selected-half packstage /
halfword-stage contract until the accumulator lane ownership is solved. The
next viable route should either keep live accumulator values on the direct raw
store path or use a lower-level cooperative-matrix store primitive that avoids
this invalid halfword staging map.

## Rejected B64GROUP Buffer-Store Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifacts:

- ISA comparison:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-b64group-bufferstore-isa-20260618/`
- Focused correctness/timing:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-b64group-bufferstore-focused-20260618/`

What changed:

- Combined grouped `ds_read_b64` fragment loads with the fixed gfx11 raw
  buffer-store writeback path.
- Kept the 22528-byte stage allocation and full-tile direct writeback path.
- Left p33 to the existing narrow packed route.

Compile evidence:

- Wave64, SGPR `28`, VGPR `195`, LDS `22528`, no spills.
- Matched RADV on `32` f16 WMMA, `64` `ds_load_b64`, and two barriers.
- Still missed RADV on `ds_load_u16_d16` (`128 -> 0`),
  `ds_store_b16` (`128 -> 2`), and `buffer_store_b32` (`192 -> 128`).

Focused gates:

- p512 and p513 selected the candidate for `ffn_out`, `ffn_gate`, and
  `result_output` and passed CPU-reference checks.
- p33 stayed on the narrow packed route and passed.

Timing:

- p512 regressed: `ffn_out 7093.01 -> 15624.94 us`,
  `ffn_gate 6323.39 -> 12512.46 us`,
  `result_output 56201.69 -> 107212.45 us`.
- p513 regressed: `ffn_out 7750.36 -> 15292.70 us`,
  `ffn_gate 6951.29 -> 14250.05 us`,
  `result_output 63109.00 -> 124419.50 us`.

Decision:
reject for production. Matching the `ds_load_b64` axis and the raw
`buffer_store_b32` primitive together is still not the Vulkan-winning schedule.
The remaining direct-WMMA path needs RADV's cooperative halfword LDS load/store
and lane ownership, likely below what these HIP C++ source-visible variants can
express.

## Upper B-Fragment Diagnostic

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-bfrag-bmirror-20260619-100138/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added a CMake/Ninja-built `bfrag-dump` bench mode that stages RHS through
  the same padded LDS layout and loads all four B column fragments with the
  same `ds_read_b64` helper used by the failing Q8 WMMA repro paths;
- added `single-group8-bmirror0` and `single-group12-bmirror0`, which mirror
  lower-column RHS values into the upper B column tiles, compute through the
  upper `col_sub` operand paths, and compare the stored upper outputs against
  lower-column CPU reference values.

Runtime result:

- `bfrag-dump` passed exactly for `cols=64` and `cols=33`:
  `active=8192`, `bad=0`, `nan=0`, `sentinel=0`, `max_abs=0`.
- `single-group8-bmirror0` still failed:
  - p64: `active=256`, `bad=256`, `nan=96`, `max_abs=16976`;
  - p33: `active=16`, `bad=16`, `nan=8`, `max_abs=5484`.
- `single-group12-bmirror0` still failed on p64:
  `active=256`, `bad=256`, `nan=0`, `max_abs=12.9951`.

Static result:

- `bfrag-dump`: wave64, SGPR `12`, VGPR `75`, LDS `5120`, no spill metadata,
  `32` `ds_load_b64`, no WMMA.
- `single-group8-bmirror0`: wave64, SGPR `40`, VGPR `99`, LDS `10240`, no
  spill metadata, `2` WMMA, `64` `ds_load_b64`, `4` `buffer_store_b32`, and
  `2` barriers.
- `single-group12-bmirror0`: wave64, SGPR `32`, VGPR `110`, LDS `10240`, no
  spill metadata, `2` WMMA, `64` `ds_load_b64`, `4` `buffer_store_b32`, and
  `2` barriers.

Interpretation:

This retires the raw B-value hypothesis. The upper B fragments are loaded with
the expected half-rounded values, including the p33 edge case, and the upper
WMMA paths still fail even when their B values are lower-column mirrors. The
remaining failure is therefore in the upper B operand slot/register/WMMA
consumption contract of this HIP C++ spelling, not in RHS staging, output
addressing, OPSEL, or distinct upper-column data. The next useful probe should
change operand materialization or move to a lower-level WMMA operand spelling;
do not spend another loop retesting B-value staging.

## Fragment-Copy Correctness Repair

Artifacts:

- `cache/hrxv1/gfx1151/q8-wmma-frag-copy-20260619-100703/`
- `cache/hrxv1/gfx1151/q8-wmma-phase-copy-20260619-100926/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added an explicit `v_mov_b32` fragment materialization helper;
- added single-group modes for upper groups 8 and 12 with B-only copy and
  A+B copy;
- added full two-phase modes with B-only copy and A+B copy.

Runtime result:

- `single-group8-bcopy`: passed p64 and p33, with max error `0.109412` p64
  and `0.0129853` p33.
- `single-group8-abcopy`: passed p64 and p33, with max error `0.00288854`
  p64 and `0.00212485` p33.
- `single-group12-bcopy`: passed p64, with max error `0.132035`.
- `single-group12-abcopy`: passed p64, with max error `0.00239494`.
- `array8-fullb-2phase-bcopy`: passed p64 and p33 full active outputs, with
  max error about `0.109`.
- `array8-fullb-2phase-abcopy`: passed p64 and p33 full active outputs, with
  max error about `0.0027`.

Static result:

- Single-group copy modes remain small: wave64, LDS `10240`, private segment
  `0`, no spill metadata, `2` WMMA, `64` `ds_load_b64`,
  `4` `buffer_store_b32`, and VGPR `117`.
- Full two-phase copy modes are correct but expensive: each phase is wave64,
  LDS `10240`, private segment `0`, no spill metadata, `16` WMMA,
  `64` `ds_load_b64`, `32` `buffer_store_b32`, and `2` barriers.
  B-copy uses VGPR `235` with `219` `v_mov_b32`; A+B-copy uses VGPR `247`
  with `347` `v_mov_b32`.

Interpretation:

This is the first direct repair of the upper-column Q8 WMMA correctness
failure. Explicit operand materialization fixes both isolated upper groups and
the full two-phase topology that previously failed. B-copy appears sufficient
for the repro tolerance, while A+B-copy restores the low error scale of the
passing controls.

Do not promote the current two-phase copy spelling as production. It is a
correctness primitive with a resource warning: VGPR is near the gfx1151 ceiling
and the move count is high. The next production-facing candidate should apply
the narrowest materialization needed, preferably B-only or upper-column-only,
inside a real Q8/Q5 route and then run focused CPU-reference, route trace,
static ISA, and same-runner timing gates before model-level A/B.

## Rejected Production B-Copy Transfer

Route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_split_selected_bcopy_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_FAST_HALF_SPLIT_SELECTED_BCOPY_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifacts:

- upper-only B-copy focused p512:
  `cache/hrxv1/gfx1151/q8_0-bcopy-focused-20260619-102234/`
- full B-copy static:
  `cache/hrxv1/gfx1151/q8-bcopy-full-static-20260619-102350/`
- full B-copy focused p512:
  `cache/hrxv1/gfx1151/q8_0-bcopy-full-focused-20260619-102350/`

What changed:

- Added a production-catalog sibling of the rejected split-selected route.
- Preserved B64GROUP fragment loads, packed A/B LDS staging, selected-half
  split writeback, raw buffer stores, WG256, and wave64.
- First tried the narrowest transfer: copy only upper-column B operands before
  WMMA.
- Then pivoted to full B-copy to match the passing
  `array8-fullb-2phase-bcopy` repro more closely.

Static result for full B-copy:

- wave64, SGPR `28`, VGPR `212`, LDS `22528`, no spills, private segment `0`;
- `32` `v_wmma_f16_16x16x16_f16`;
- `64` `ds_load_b64`, `128` `ds_load_u16_d16`, `128` `ds_store_b16`;
- `128` `buffer_store_b32`, `2` barriers;
- first-WMMA window still collapsed: final pre-WMMA `lgkmcnt(0)`, not RADV's
  broad outstanding LDS window.

Focused p512 result:

- Upper-only B-copy selected for `ffn_out`, `ffn_gate`, and `result_output`,
  but failed with `ERR=3.007032931`, `ERR=1.773888636`, and a
  `result_output` NaN at index `6188064`.
- Full B-copy also selected for the same rows and still failed with
  `ERR=2.699209606`, `ERR=2.588006472`, and the same `result_output` NaN
  index.
- `Vcur` and `Qcur` stayed on the packed Q8_1 route and passed in both runs.

Decision:
reject before timing/model tests. The fragment-copy primitive repairs the
standalone small/full-two-phase repro, but it does not repair the real
split-selected catalog route. The remaining invalid surface is likely selected
accumulator writeback or broader WMMA lane ownership, not just B operand
materialization. Future production candidates should avoid another
split-selected halfword-stage transfer unless they first prove the
live-accumulator selected-store contract at the exact catalog shape.

## Single-Group Split-Selected Stage Contract

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-single-group-stage-20260619-110613/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built standalone modes that combine the previously passing
  explicit fragment-copy repair with the production split-selected halfword
  stage helper;
- tested one active output group at a time:
  `single-group0-bcopy-stage`, `single-group8-bcopy-stage`,
  `single-group8-abcopy-stage`, `single-group12-bcopy-stage`, and
  `single-group12-abcopy-stage`.

Runtime result:

- group 0 bcopy-stage passed p64 and p33 with `bad=0`, `nan=0`, and
  `max_abs=0.204907`;
- group 8 bcopy-stage passed p64 and p33 with `bad=0`, `nan=0`,
  `max_abs=0.204907` on p64 and `0.00314344` on p33;
- group 8 abcopy-stage also passed p64 and p33;
- group 12 bcopy-stage failed finite p64 correctness:
  `active=256`, `bad=10`, `nan=0`, `max_abs=0.346383`;
- group 12 abcopy-stage also failed finite p64 correctness:
  `active=256`, `bad=10`, `nan=0`, `max_abs=0.348336`;
- group 12 has no active p33 outputs because p33 does not reach `col_sub=3`.

Static result:

- selected staged bcopy symbols are wave64, no spills, LDS `10752`, `8`
  WMMA, `64` B64 LDS reads, `10` halfword LDS stores, `8` halfword reloads,
  `4` buffer stores, and `81` `v_mov_b32`;
- group 8 bcopy-stage metadata: SGPR `41`, VGPR `171`;
- group 12 bcopy-stage metadata: SGPR `31`, VGPR `170`, one extra wait.

Interpretation:

This narrows the rejected production transfer. The split-selected halfword
stage helper is not universally invalid: it works for lower group 0 and upper
`col_sub=2` group 8 when the B-copy materialization repair is present. The
isolated `col_sub=3` group 12 path still fails even with B-copy or A+B-copy, so
the remaining direct-WMMA semantic bug is specifically in the group12/upper
column lane contract. Do not build another full catalog route on this helper
until the group12 contract is repaired or replaced by a lower-level cooperative
writeback primitive.

### Selected-Only Stage Axis

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-selected-only-stage-20260619-111309/`

What changed:

- added selected-only stage modes that write and reload only
  `HRX_Q8_0_WMMA_VK128_W64_OPSEL`, removing the dummy other-half writes from
  the split-selected helper;
- tested `single-group8-bcopy-stage-selected`,
  `single-group12-bcopy-stage-selected`, and
  `single-group12-abcopy-stage-selected`.

Runtime result:

- group 8 selected-only passed p64 and p33;
- group 12 selected-only still failed finite p64 correctness:
  bcopy `bad=16/256`, `max_abs=0.346383`; abcopy `bad=16/256`,
  `max_abs=0.348336`; no NaNs or infinities.

Static result:

- group8/group12 bcopy selected-only symbols are wave64, no spills, LDS
  `10752`, `8` WMMA, `64` B64 LDS reads, `6` halfword LDS stores, `4`
  halfword reloads, `4` buffer stores, and `81` `v_mov_b32`.

Interpretation:

This rejects the dummy-other-write hypothesis. Group12 does not fail because
the split-selected helper writes the non-selected accumulator half into nearby
LDS slots. The remaining failure follows the `col_sub=3` B operand,
accumulator, or store lane contract even when only selected halfwords are
staged.

## Current HRX Route

Route:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/kernels/mul_mat_vec_q8_0.hip.cpp`

Schedule facts:

- CMake wavefront: wave32.
- HSACO metadata: `wavefront_size=32`, `vgpr_count=122`, no spills,
  `group_segment_fixed_size=1088`.
- Workgroup: 256 threads.
- Tile: `BM=128`, `BN=32`.
- Ownership: one output row per thread lane over `row_lane = tid & 127`; two
  column lanes per workgroup, each accumulating `COLS_PER_THREAD=16`.
- A path: each active thread loads its own Q8_0 row block directly from global.
- B path: each K block stages `BN * 8` packed Q8_1 x4 integers plus scales in
  shared memory.
- Dot primitive: `__builtin_amdgcn_sudot4`.
- K loop: one 32-wide quant block per iteration, with a barrier before and
  after each block.

Focused p512 timing before the wave64 probe:

- `Vcur-0`: `524.019 us`
- `Qcur-0`: `2003.890 us`
- `ffn_out-0`: `12100.674 us`
- `ffn_gate-0`: `21138.738 us`
- `result_output`: `203806.524 us`

## Vulkan Prior

Relevant source:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_shmem_types.glsl`

Route:
`matmul_q8_0_q8_1` through `pipeline_dequant_mul_mat_mat_q8_1`.

Schedule facts:

- Integer-dot MMQ route, not F16/BF16 coopmat.
- Dot primitive: `dotPacked4x8EXT`.
- Q8_0 A-cache stores eight packed int32 payloads plus scale per quant block.
- Q8_1 B-cache stores eight packed int32 payloads plus `ds`.
- Both A and B are staged into shared memory.
- Shader inner loop loads small register tiles from shared memory and each lane
  accumulates a row/column tile: `WMITER * TM * WNITER * TN` outputs per lane.
- For the AMD/non-proprietary medium integer MMQ tuning branch, the selected
  medium tuple is `BLOCK_SIZE=256, BM=64, BN=64, BK=32, WM=16, WN=16,
  WMITER=2, TM=2, TN=2, TK=1, WARP=16`.
- For the generic coopmat-capable large tuple, the integer MMQ shape is
  `BLOCK_SIZE=256, BM=128, BN=128, BK=32, WM=<subgroup_size_8>, WN=64,
  WMITER=2, TM=2, TN=2, TK=1, WARP=<subgroup_size_8>`.

Vulkan p512 labels from
`cache/hrxv1/gfx1151/current-best-q6id-basket-p512-fa1-r1-20260617-184515/llama3_1_8b_q8_0/vulkan/stderr.log`
show the biggest model-relevant gaps on the FFN rows:

- `m=14336,n=512,k=4096`: Vulkan about `5045.57 us`, HRX about `21138.74 us`
- `m=4096,n=512,k=14336`: Vulkan about `4041.61 us`, HRX about `12100.67 us`

## Rejected Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_wave64_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-wave64-focused-p512-20260617-185605/`

What changed:

- Same source-level dataflow and tile as current HRX route.
- Separate CMake-built HIP object compiled with wave64.
- Opt-in selector:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X32_WAVE64_PROMPT=1`.

Result:

- Focused CPU-reference correctness passed for all five p512 rows.
- Route trace selected the wave64 provider for all five rows.
- Focused timing regressed every row:
  - `Vcur-0`: `524.019 -> 543.355 us`
  - `Qcur-0`: `2003.890 -> 2709.892 us`
  - `ffn_out-0`: `12100.674 -> 18948.767 us`
  - `ffn_gate-0`: `21138.738 -> 35776.988 us`
  - `result_output`: `203806.524 -> 263176.429 us`

Decision:
reject wave64-only for this source-level schedule. The missing performance is
not just the wavefront mode; the next candidate should change output ownership
and/or tile shape toward the Vulkan MMQ dataflow.

## Accepted Opt-In Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x64-focused-p512-20260617-190744/`

Model A/B artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x64-model-ab-20260617-191149/`

What changed:

- Tile changed from `BM=128, BN=32` to `BM=64, BN=64`.
- CMake-built HIP object remains wave32.
- Workgroup remains 256 threads.
- Row ownership changed to `row_lane = tid & 63`.
- Four column lanes per workgroup each accumulate `COLS_PER_THREAD=16`.
- The direct-A and staged-B dataflow is otherwise intentionally close to the
  existing Q8_0 route, so this isolates the tile/output-ownership pivot.
- Opt-in selector:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.

Compile evidence:

- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x64.hsaco`
- Metadata: `wavefront_size=32`, `vgpr_count=127`, `sgpr_count=22`,
  `group_segment_fixed_size=2176`, no spills.

Focused result:

- CPU-reference correctness passed for all p33, p512, and p513 rows.
- Route traces selected the BM64/BN64 provider for all tested Q8_0 prompt rows.
- Focused p512 total over the five exported rows improved
  `239194.7 -> 104160.8 us` (`2.30x`):
  - `Vcur-0`: `570.749 -> 630.539 us` (`0.91x`)
  - `Qcur-0`: `2064.730 -> 1926.867 us` (`1.07x`)
  - `ffn_out-0`: `14411.267 -> 9310.203 us` (`1.55x`)
  - `ffn_gate-0`: `18470.177 -> 8073.023 us` (`2.29x`)
  - `result_output`: `203677.762 -> 84220.167 us` (`2.42x`)
- Odd/tail focused totals also improved:
  - p33: `35438.8 -> 11564.9 us` (`3.06x`)
  - p513: `254447.9 -> 120127.6 us` (`2.12x`)

Same-runner model A/B on Llama 3.1 8B Q8_0 with flash attention enabled:

- p33: HRX baseline `92.317 tok/s`, BM64/BN64 `194.470 tok/s`,
  Vulkan `196.622 tok/s`; variant is `2.11x` baseline and `0.989x` Vulkan.
- p512: HRX baseline `209.715 tok/s`, BM64/BN64 `394.224 tok/s`,
  Vulkan `884.048 tok/s`; variant is `1.88x` baseline and `0.446x` Vulkan.
- p513: HRX baseline `206.993 tok/s`, BM64/BN64 `377.980 tok/s`,
  Vulkan `837.168 tok/s`; variant is `1.83x` baseline and `0.451x` Vulkan.

Decision:
accept BM64/BN64 as a gfx1151 opt-in candidate. This is a large production
move and nearly closes the narrow p33 Q8_0 row versus Vulkan, but it is not a
final Q8_0 solution because p512/p513 remain around `0.45x` same-run Vulkan.
The next Q8_0 candidate should mine the remaining Vulkan schedule delta,
especially cooperative A staging and per-lane output ownership/register tile
differences.

## Rejected A+B Staging Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_ab_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x64-ab-focused-p512-20260617-191938/`

What changed:

- Kept `BM=64`, `BN=64`, wave32, and 256-thread workgroups from the accepted
  direct-A BM64/BN64 route.
- Added cooperative LDS staging for Q8_0 A blocks in addition to the existing
  Q8_1 B staging.
- Opt-in selector:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_AB_PROMPT=1`.

Compile evidence:

- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x64_ab.hsaco`
- Metadata: `wavefront_size=32`, `vgpr_count=123`, `sgpr_count=32`,
  `group_segment_fixed_size=4352`, no spills.

Result:

- Focused p512 CPU-reference correctness passed.
- Route trace selected the A+B-staged provider for all five rows.
- Focused p512 timing regressed every row versus direct-A BM64/BN64:
  - `Vcur-0`: `566.437 -> 684.076 us`
  - `Qcur-0`: `1939.990 -> 2326.206 us`
  - `ffn_out-0`: `9800.640 -> 11385.032 us`
  - `ffn_gate-0`: `8739.343 -> 9634.424 us`
  - `result_output`: `83885.357 -> 91222.690 us`
  - total: `104931.8 -> 115252.4 us` (`0.91x`)

Decision:
reject the naive cooperative A+B LDS staging pivot. This result does not mean
A reuse is irrelevant, but this spelling adds LDS traffic/barriers without
recovering enough redundant A-load cost. Avoid retesting A staging unless it is
paired with a different per-lane output tile, fewer column-lane duplicate
loads, or a more Vulkan-like register tile.

## Rejected Smaller Output-Tile Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq32x64_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq32x64-focused-p512-20260617-192604/`

What changed:

- Changed tile from accepted `BM=64, BN=64` to `BM=32, BN=64`.
- Reduced per-thread output columns from 16 to 8.
- Kept direct-A loads and staged-B LDS dataflow.
- Opt-in selector:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ32X64_PROMPT=1`.

Compile evidence:

- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq32x64.hsaco`
- Metadata: `wavefront_size=32`, `vgpr_count=78`, `sgpr_count=22`,
  `group_segment_fixed_size=2176`, no spills.

Result:

- Focused p512 CPU-reference correctness passed.
- Route trace selected the BM32/BN64 provider for all five rows.
- Focused p512 timing regressed total and hot rows versus accepted BM64/BN64:
  - `Vcur-0`: `568.529 -> 660.519 us`
  - `Qcur-0`: `2817.721 -> 2383.890 us`
  - `ffn_out-0`: `9915.608 -> 12187.366 us`
  - `ffn_gate-0`: `9046.741 -> 10254.701 us`
  - `result_output`: `83383.405 -> 91489.524 us`
  - total: `105732.0 -> 116976.0 us` (`0.90x`)

Decision:
reject BM32/BN64 despite the lower VGPR count. Halving per-thread column work
does not pay for doubling row workgroups on the production-width p512 rows.

## Rejected BM128/BN64 Compile Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x64_wg256_f32`

Artifact:
`build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq128x64.hsaco`

What changed:

- Changed tile from accepted `BM=64, BN=64` to `BM=128, BN=64`.
- Increased per-thread output columns from 16 to 32.
- Kept direct-A loads and staged-B LDS dataflow.
- Opt-in selector:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X64_PROMPT=1`.

Compile evidence:

- Metadata: `wavefront_size=32`, `vgpr_count=192`, `vgpr_spill_count=47`,
  `private_segment_fixed_size=192`, `group_segment_fixed_size=2176`.

Decision:
reject before focused timing. The route spills heavily and is not a viable
production candidate. The column-widening axis needs a different register
tile, not a simple `COLS_PER_THREAD=32` direct extension.

## Accepted Opt-In BN96 Plus BN64 Fallback Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x96_wg256_f32` paired with the accepted
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32` fallback.

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x96-focused-20260617-214000/`

Model A/B artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x96-model-ab-20260617-213256/`

What changed:

- Tile changed from accepted `BM=64, BN=64` to `BM=64, BN=96` for
  production-width prompt rows.
- Per-thread output columns increased from 16 to 24, staying below the
  spilling `COLS_PER_THREAD=32` BN128 probe.
- Workgroup remains 256 threads, wave32.
- Direct-A plus staged-B dataflow is unchanged, so this isolates the column
  widening axis.
- Production policy must enable both gates:
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1` and
  `GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.

Compile evidence:

- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x96.hsaco`
- Metadata: `wavefront_size=32`, `vgpr_count=181`, `sgpr_count=26`,
  `group_segment_fixed_size=3264`, no spills.

Focused result:

- CPU-reference correctness passed for p512 and p513 with BN96 selected.
- p33 correctness passed and stayed off BN96 because of the `cols >= 64`
  guard.
- p512 focused row timing was mixed versus BN64: small rows regressed or were
  flat, while hot FFN/output rows improved enough to justify model A/B.

Same-runner model A/B on Llama 3.1 8B Q8_0 with flash attention enabled:

- BN96 alone is not a valid policy: p33 falls back to the older BM128/BN32
  route and regresses to `113.638 tok/s`.
- With BN64 fallback enabled, p33 stays on BN64 and improves
  `203.733 -> 207.523 tok/s`.
- p512 selects BN96 and improves `402.465 -> 424.832 tok/s`.
- p513 with `ub=512` selects BN96 for the p512 graph, uses scalar Q8_0 for
  the residual-token graph, and improves `386.261 -> 408.731 tok/s`.
- p513 with `ub=1024` selects BN96 for the single p513 graph and improves
  `372.236 -> 412.379 tok/s`.

Decision:
accept the paired BN96+BN64 policy as a gfx1151 opt-in candidate. This is a
small but coherent production-width Q8_0 lift with explicit odd/tail evidence.
It does not close the Vulkan gap; p512/p513 remain roughly half of Vulkan, so
the next Q8_0 work should move beyond direct-A/staged-B scalar-dot column
widening toward a more Vulkan-like cooperative or packed dataflow.

## Current-Head Commit-Aligned Q8_0 Check

Artifact:
`cache/hrxv1/gfx1151/q8_0-current-head-commitaligned-r3-20260619-004524/`

Source and build:

- `sources/llama.cpp` clean at `ebb85c542`.
- HRX and Vulkan binaries were rebuilt through CMake/Ninja, and both
  `llama-bench` JSON rows report build commit `ebb85c542`.
- Model: Llama 3.1 8B Q8_0.
- Cases: p33, p512, p513, `--flash-attn 1`, `-r 3`, no warmup.

Same-machine result:

| Case | HRX steady tok/s | Vulkan steady tok/s | Ratio | HRX top route |
| --- | ---: | ---: | ---: | --- |
| p33 | `204.287` | `233.279` | `0.876x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32` |
| p512 | `457.849` | `921.731` | `0.497x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` |
| p513 | `419.430` | `812.311` | `0.516x` | `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32` |

Focused current-head artifact:
`cache/hrxv1/gfx1151/q8_0-current-focused-eb85c542-20260619-004731/`

Focused p512 CPU-reference passed and selected the current production route
for all five exported rows. Focused p513 CPU-reference also passed and selected
the BN112 split-qsum route for all five exported rows. Focused timing:

| Row | p512 HRX us | p513 HRX us |
| --- | ---: | ---: |
| `Vcur-0` | `558.792` | `547.193` |
| `Qcur-0` | `1906.867` | `2148.921` |
| `ffn_out-0` | `7137.794` | `7735.595` |
| `ffn_gate-0` | `6449.436` | `7055.443` |
| `result_output` | `56659.810` | `63606.143` |

Production route correction:

- In the real p512 model graph, HRX dispatches `result_output` as
  `rows=128256, cols=1` through `hrx_mul_mat_vec_q8_0_f32`, not as the
  exported focused `[128256,512]` batched row.
- Vulkan does the same decode-shaped final output row with
  `mul_mat_vec_q8_0_f32_f32`, not the large aligned batched pipeline.
- Therefore the focused `[128256,512]` `result_output` row is useful stress
  evidence for the kernel family, but it is not a production p512/p513 prompt
  bottleneck. Do not use it as the primary acceptance row for Q8_0 parity work.

Relevant production p512 Vulkan timing from the commit-aligned perf logger:

- K/V projection family, `m=1024,n=512,k=4096`: 64 dispatches. The logger shows
  a warm first sample around `5.389 us` and later samples near `0.31 us`, so
  treat this row as suspect until verified with a common timing domain or
  profiler capture.
- FFN gate/up, `m=14336,n=512,k=4096`: 62 dispatches, steady samples around
  `4934-4995 us`.
- FFN out, `m=4096,n=512,k=14336`: 31 dispatches, steady samples around
  `4024 us`.
- Q/attention/out, `m=4096,n=512,k=4096`: 64 dispatches, steady samples around
  `1342-1343 us`.
- Final output, `m=128256,n=1,k=4096`: one decode-shaped dispatch around
  `2419-2423 us`.

Decision:
keep Q8_0 as an active parity gap. The p33 policy is close enough that it
should be protected while p512/p513 are attacked. The next Q8_0 acceptance row
must target the production batched rows (`ffn_gate`, `ffn_up`, `ffn_out`,
`Qcur`/`attn_out`, and the K/V projection family), not the focused batched
`result_output` stress row. The Vulkan K/V timing anomaly needs confirmation
before treating it as a concrete sub-microsecond kernel target.

## Next Candidate Axes

Prior-led axes to test next:

- The next exact Q8_0 large candidate must target the cooperative-matrix
  global store/lane ownership gap, not another aggregate tile rename. Candidate
  acceptance starts at compile evidence: p512/p513 large route, 32
  `v_wmma_f16_16x16x16_f16` sites, no spills, LDS near `22528`, and a store
  path materially closer to RADV's `buffer_store_b32`/cooperative matrix
  lowering than the current HIP scalar lane stores. If HIP C++ cannot express
  that store path, record that blocker and switch to a lower-level/ISA probe
  instead of broadening the current direct-WMMA route.
- Do not re-run a direct Vulkan AMD medium integer-MMQ clone without a new
  pressure/scheduling idea. The exact `BK_STEP=4` HIP spelling for
  `BLOCK_SIZE=256, BM64, BN64, BK32, WM16, WN16, WMITER=2, TM2, TN2, TK1,
  WARP16` compiled but spilled badly on gfx1151 (`VGPR=192`,
  `vgpr_spill_count=298`, private segment `1188`, LDS `18432`). The bounded
  `BK_STEP=1` pressure pivot compiled cleanly (`wave32`, `SGPR=30`,
  `VGPR=86`, LDS `4608`, no spills) and passed p512 CPU-reference rows plus
  p33/p513 odd-tail smoke, but regressed focused p512 production-width rows
  versus BN96/BN64: `ffn_out 8405.6 -> 9064.9 us`,
  `ffn_gate 7728.5 -> 12024.7 us`, and
  `result_output 67033.3 -> 110588.7 us`. Artifact:
  `cache/hrxv1/gfx1151/q8_0-mmq64x64-medium-focused-20260618-022051/`.
  Interpretation: simply adding cooperative A+B staging and Vulkan medium
  output ownership is not the missing Q8_0 parity axis in HIP C++; it improves
  smaller Vcur/Qcur rows but loses too much on FFN/output.
- Column widening only with a different register tile: simple BM128/BN64
  spilled heavily, while BN96 is now accepted only as a bounded opt-in paired
  with BN64 fallback.
- Per-lane smaller output tile only with unchanged or improved row coverage:
  the simple BM32/BN64 split reduced VGPRs but lost too much row-tile
  amortization.
- A-reuse only with changed output ownership: avoid the rejected naive A+B
  staging spelling; pair any A-cache retry with smaller per-lane column work or
  a Vulkan-like register tile.

Promotion gate remains focused first: exported Q8_0 rows, route trace,
CPU-reference correctness, focused perf, then p33/p513 and model A/B only if
focused p512 wins on the hot FFN rows.

## Rejected VK128 Store-Stage Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_store_stage_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_STORE_STAGE_F16ACC_WG256_PROMPT=1`

What changed:

- Preserved the rejected direct-F32 VK128 wave64 WMMA math path.
- Added a per-wave shared half output tile before global writeback, targeting
  the RADV cooperative-matrix store/LDS-footprint delta instead of another
  tile rename.
- Built through CMake/Ninja as
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_store_stage_wg256.hsaco`.

Compile evidence:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-store-stage-isa-20260618/`
- Metadata: wave64, SGPR `22`, VGPR `145`, LDS `22528`, no spills.
- The route now matches RADV large-route LDS allocation exactly and emits
  `66 ds_store_b16` plus `64 ds_load_u16_d16`.
- It still misses the actual RADV store shape: HIP has `64 global_store_b32`
  and `34` barriers, while RADV has `192 buffer_store_b32` and `2` barriers.
  RADV also has `64 ds_load_b64` and `128 ds_load_u16_d16`.

Focused result:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-store-stage-focused-20260618/`
- Forced p512 and p513 large rows passed CPU-reference correctness and selected
  the store-stage provider. p33 stayed on the narrow Q8 routes and passed.
- Same-runner focused timing regressed versus accepted BN96/BN64 packed-Q8_1 on
  every p512/p513 row:
  p512 `ffn_out 7930.25 -> 10013.10 us`,
  `ffn_gate 7207.99 -> 10160.74 us`,
  `result_output 64968.64 -> 98961.64 us`;
  p513 `ffn_out 8020.62 -> 10732.69 us`,
  `ffn_gate 7563.19 -> 12591.31 us`,
  `result_output 66538.71 -> 109607.60 us`.

Decision:
reject for production promotion. Explicit shared output staging is not the
Vulkan cooperative-matrix store schedule. Matching the `22528` byte LDS
footprint alone adds barriers and still leaves scalarized global writeback, so
the next exact-schedule attempt should target lane/global-store ownership or a
lower-level cooperative-store spelling directly.

## Rejected VK128 Full-Tile Store Split Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_fullstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_FULLSTORE_F16ACC_WG256_PROMPT=1`

What changed:

- Preserved the direct-F32 VK128 wave64 WMMA math path.
- Split writeback into an unguarded full-tile path and a guarded edge path,
  mirroring Vulkan `mul_mm.comp`: full in-bounds aligned tiles use direct
  `coopMatStore`, while partial or unaligned tiles use scalarized fallback.
- Built through CMake/Ninja as
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_fullstore_wg256.hsaco`.

Compile evidence:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-fullstore-isa-20260618/`
- Metadata: wave64, SGPR `28`, VGPR `129`, LDS `20480`, no spills.
- The route still emits `32` f16 WMMA sites and `2` barriers.
- Store-side shape moved from the base direct route's `64 global_store_b32` to
  `128 global_store_b32`, with fewer branches than the store-stage probe.
- It still misses RADV's `192 buffer_store_b32`, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, and `22528` byte LDS footprint.

Focused result:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-fullstore-focused-20260618/`
- p512 and p513 candidate-selected large rows passed CPU-reference
  correctness. p33 stayed on the narrow Q8 routes and passed.
- Same-runner focused timing regressed versus accepted BN96/BN64 packed-Q8_1 on
  every p512/p513 row:
  p512 `ffn_out 8209.91 -> 10046.26 us`,
  `ffn_gate 7348.06 -> 10622.95 us`,
  `result_output 66045.31 -> 100014.33 us`;
  p513 `ffn_out 8252.34 -> 10091.96 us`,
  `ffn_gate 7590.76 -> 14325.28 us`,
  `result_output 67126.55 -> 113273.14 us`.

Decision:
reject for production promotion. A full-tile/edge split is a real store-side
codegen pivot and should be retained as evidence, but it still does not express
the Vulkan cooperative-matrix store/lane ownership. The next exact attempt
needs either a lower-level writeback spelling that reaches RADV's 192-store
shape without the store-stage barrier cost, or a return to packed-Q8_1 work
where the accepted production path already wins.

## Rejected VK128 B64GROUP Full-Tile Store Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_fullstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_FULLSTORE_F16ACC_WG256_PROMPT=1`

What changed:

- Preserved the direct-F32 VK128 wave64 WMMA math path.
- Combined the grouped `ds_read_b64` A/B fragment-load window from the B64GROUP
  probe with the full-tile/edge writeback split from the FULLSTORE probe.
- Built all HIP C++ through CMake/Ninja as
  `mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_fullstore_wg256.hsaco`.

Compile evidence:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-b64group-fullstore-isa-20260618/`
- Metadata: wave64, SGPR `28`, VGPR `195`, LDS `20480`, no spills.
- The route emits `32` f16 WMMA sites, `64 ds_load_b64`,
  `128 global_store_b32`, and `2` barriers.
- This composes two source-visible RADV deltas, but still misses RADV's
  `22528` byte LDS footprint, `128 ds_load_u16_d16`, `128 ds_store_b16`,
  `192 buffer_store_b32`, and larger scalar-memory schedule.

Focused result:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-b64group-fullstore-focused-20260618/`
- p512 and p513 candidate-selected large rows passed CPU-reference
  correctness. p33 stayed on the narrow packed Q8 route and passed.
- Same-runner focused timing regressed versus accepted BN96/BN64 packed-Q8_1 on
  the large p512/p513 rows:
  p512 `ffn_out 8257.40 -> 15061.62 us`,
  `ffn_gate 7198.37 -> 12412.83 us`,
  `result_output 65863.29 -> 106269.36 us`;
  p513 `ffn_out 8309.78 -> 15999.88 us`,
  `ffn_gate 7522.81 -> 14544.52 us`,
  `result_output 67215.98 -> 123510.48 us`.

Decision:
reject for production promotion. This is a useful negative result because it
proves the p512/p513 gap is not closed by independently matching RADV's
fragment-load count and moving the scalar store count from 64 to 128. The
remaining exact-schedule work should focus on the cooperative-matrix
lane/writeback contract itself, or on a packed-Q8_1 route that keeps the current
production advantage while mechanically importing the Vulkan/RADV winning
schedule facts.

## Cooperative-Matrix Store Extraction

Artifact:
`cache/hrxv1/gfx1151/q8_0-coopmat-schedule-extract-20260618/`

Files:

- `q8_0-large-coopmat-schedule.md`
- `q8_0-p33-medium-coopmat-schedule.md`
- `q8_0-p513-large-coopmat-schedule.md`

Tool:
`sources/llama.cpp/tools/vulkan-oracle/extract_coopmat_schedule.py`

Extracted contract:

- Large p512/p513 route remains
  `matmul_q8_0_f32_f16acc_aligned_l`,
  `spec=[256,128,128,32,64,64,2,16,16,16,64]`,
  `wg_denoms=[128,128,1]`.
- p33 route remains `matmul_q8_0_f32_f16acc_aligned_m`,
  `spec=[128,64,64,32,64,32,2,16,16,16,64]`,
  `wg_denoms=[64,64,1]`.
- The SPIR-V source-level cooperative matrix schedule is small and generic:
  `TM=4`, `TN=2`, `TK=1`, subgroup-scoped A/B/accumulator coopmats, two static
  `OpCooperativeMatrixLoadKHR`, one static `OpCooperativeMatrixMulAddKHR`, and
  three static `OpCooperativeMatrixStoreKHR` paths.
- The full aligned store path is
  `coopMatStore(cm_dtype, data_d, offsets + ..., p.stride_d,
  gl_CooperativeMatrixLayoutColumnMajor)`. The other two stores are staged
  fallback paths for unaligned or partial tiles.
- RADV's specialization/unrolling turns that compact SPIR-V into the actual
  target schedule: `32` `v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`,
  `22528` LDS bytes, `192` VGPR, no spills, and `2` barriers.

Decision:
the next direct-WMMA HIP probe must explain how it is reproducing the aligned
cooperative-matrix store/lane mapping, not just increasing scalar store count.
If that mapping cannot be expressed cleanly in HIP C++ builtins, the next
production-oriented path should move back to the packed-Q8_1 route and import
one RADV schedule fact at a time without reintroducing the known spilling
128x128 packed-Q8_1 shape.

## High-Half OPSEL Direct-WMMA Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_hi_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_HI_F16ACC_WG256_PROMPT=1`

Artifacts:

- Focused p33/p512/p513:
  `cache/hrxv1/gfx1151/q8_0-wmma-vk128-hi-focused-20260618/`
- ISA comparison:
  `cache/hrxv1/gfx1151/q8_0-radv-vs-hrx-wmma16-vk128-hi-isa-20260618/`

Purpose:
test the narrow accumulator-half/lane-contract hypothesis directly. The route
keeps the rejected VK128 direct-F32 wave64 schedule and flips only
`HRX_Q8_0_WMMA_VK128_W64_OPSEL=1`.

Compile evidence:

- The HSACO is wave64 with SGPR `22`, VGPR `129`, LDS `20480`, no spills.
- It still emits `32 ds_load_b128`, `64 global_store_b32`, `32` f16 WMMA
  sites, and `2` barriers.
- This does not move the static schedule toward the RADV large route's
  `22528` LDS bytes, `192` VGPR, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, and
  `192 buffer_store_b32`.

Focused result:

- p512 and p513 candidate-selected large rows passed CPU-reference correctness.
  p33 stayed on existing narrow routes and passed.
- Same-runner focused timing regressed versus accepted BN96/BN64 packed-Q8_1:
  p512 `ffn_out 8424.22 -> 9845.73 us`,
  `ffn_gate 7237.72 -> 11531.29 us`,
  `result_output 65649.00 -> 91637.90 us`;
  p513 `ffn_out 8163.48 -> 10149.92 us`,
  `ffn_gate 7480.91 -> 12667.93 us`,
  `result_output 66834.12 -> 104780.57 us`.

Decision:
reject for production promotion. High-half OPSEL is arithmetically valid for
the current HIP spelling but slower, so the large remaining gap is not a simple
wrong-half accumulator bug. Continue mechanically against the Vulkan schedule
on the LDS/load/writeback contract, or pivot direct production work back to the
packed-Q8_1 path and import exact RADV facts without reintroducing the known
packed 128x128 spill shape.

## Rejected Packed-Q8_1 BN112 Compile Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X112_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x112-compile-20260618/`

Purpose:
test the remaining simple column-widening space in the accepted packed-Q8_1
family. The source preserves the BN96 direct-A/staged-B dataflow and BM64 row
ownership, changing only `BN=112` and `COLS_PER_THREAD=28`. This brackets the
axis between accepted BN96 and known-spilling BN128, and reduces p512 column
workgroups from six to five if it can compile cleanly.

Compile evidence:

- BN96 accepted route: wave32, SGPR `26`, VGPR `181`, LDS `3264`, no spills.
- BN112 probe: wave32, SGPR `28`, VGPR `192`, LDS `3808`,
  `vgpr_spill_count=24`, `private_segment_fixed_size=100`.
- BN128 rejected route: wave32, SGPR `27`, VGPR `192`, LDS `4352`,
  `vgpr_spill_count=55`, `private_segment_fixed_size=224`.

Decision:
reject at the compile-resource gate before focused runtime tests. BN112 proves
that the simple packed-Q8_1 column-widening spelling hits the register-pressure
wall before it reaches the Vulkan-like 128-column production tile. The next
packed-Q8_1 candidate needs a different register tile, split accumulation, or
output ownership, not just `COLS_PER_THREAD > 24`.

## Rejected Packed-Q8_1 BN104 Compile Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x104_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X104_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x104-compile-20260618/`

Purpose:
test whether a smaller column-width bracket between accepted BN96 and rejected
BN112 could avoid the register cliff while reducing p512 column workgroups. It
preserves the accepted direct-A/staged-B packed-Q8_1 dataflow and changes only
`BN=104` and `COLS_PER_THREAD=26`.

Compile evidence:

- BN96 accepted route: wave32, SGPR `26`, VGPR `181`, LDS `3264`, no spills.
- BN104 probe: wave32, SGPR `28`, VGPR `192`, LDS `3536`,
  `vgpr_spill_count=9`, `private_segment_fixed_size=40`.
- BN112 rejected route: wave32, SGPR `28`, VGPR `192`, LDS `3808`,
  `vgpr_spill_count=24`, `private_segment_fixed_size=100`.
- BN128 rejected route: wave32, SGPR `27`, VGPR `192`, LDS `4352`,
  `vgpr_spill_count=55`, `private_segment_fixed_size=224`.

Decision:
reject at the compile-resource gate before focused runtime tests. BN104 reduces
spill severity versus BN112, but still reaches the same 192-VGPR cliff. This
closes simple column widening past BN96 as the next Q8_0 packed axis; the next
candidate needs a changed register tile, split accumulation, or output
ownership.

## gfx1151 Default Policy And Cooperative-Store Blocker

Current source policy:

- `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x96_wg256_f32` is the gfx1151 default for
  Q8_0 prompt rows with `cols >= 64`.
- `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32` is the gfx1151 narrow/odd
  fallback.
- Rollbacks:
  `GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X96_PROMPT=1` and
  `GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X64_PROMPT=1`.

Reason:
the accepted BN96 plus BN64 policy remains the fastest measured HRX Q8_0
production path on this machine. It is still only about half of same-run Vulkan,
so it is a current-best production policy, not a parity claim.

Default-regate artifact:
`cache/hrxv1/gfx1151/q8_0-bn96-default-regate-20260618-053457/`.
Focused p33, p512, and p513 CPU-reference gates passed with default selection.
Routes prove p33 selects BN64, while p512 and p513 select BN96. Disabling both
BN96 and BN64 rolls back to `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32`.
Focused p512 rollback vs default improved the largest rows from
`ffn_out 11496.55 -> 8133.81 us`, `ffn_gate 32026.66 -> 7282.08 us`, and
`result_output 241314.48 -> 66272.21 us`; p513 improved
`ffn_out 13131.21 -> 8243.04 us`, `ffn_gate 26280.68 -> 7362.14 us`, and
`result_output 255766.02 -> 67142.86 us`.

Exact Vulkan status:
the direct-F32 VK128 HIP C++ probes now cover the source-visible axes that can
be expressed with current builtins: grouped `ds_read_b64` fragment loads, exact
`22528` byte LDS output staging, full-tile/edge store split, grouped-load plus
fullstore, and WMMA OPSEL high. Every probe passed focused p512/p513
CPU-reference when selected, but every probe regressed versus BN96/BN64. The
closest combined probe still differs from RADV on the cooperative-matrix
writeback/lane-ownership path: RADV has `128 ds_load_u16_d16`,
`128 ds_store_b16`, and `192 buffer_store_b32`, while the HIP C++ route remains
scalarized.

Environment blocker:
the active ROCm tree `/srv/vm-shared/rocm/rocm-head` does not provide
`rocm/include/rocwmma/rocwmma.hpp`. Do not spend another candidate on scalarized
per-lane stores for this family. The next exact-clone step is either installing
or otherwise providing a matrix-fragment store primitive that can build through
CMake/Ninja, or moving to lower-level codegen that can express the RADV
cooperative-store schedule directly.

## Accepted Packed-Q8_1 BN112 Split-Qsum Pivot

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32`

Policy:
default on gfx1151 for Q8_0 prompt rows with `cols >= 128`, with BN64 kept for
p33/narrow prompts. Rollback:
`GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X112_SPLITQSUM_PROMPT=1`.

Purpose:
after BN104 and BN112 simple column widening spilled, this probe preserved the
accepted BM64 direct-A/staged-B packed-Q8_1 dataflow but split the temporary
`qsum` register tile into two 14-column chunks. This tests whether the spill
wall was caused by `qsum[COLS_PER_THREAD]` live range rather than BN112 column
coverage itself.

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q8_0-mmq64x112-splitqsum-compile-20260618/`;
- focused:
  `cache/hrxv1/gfx1151/q8_0-mmq64x112-splitqsum-focused-20260618/`;
- default/rollback regate:
  `cache/hrxv1/gfx1151/q8_0-mmq64x112-splitqsum-default-regate-20260618/`;
- model A/B:
  `cache/hrxv1/gfx1151/q8_0-mmq64x112-splitqsum-model-ab-20260618/`.

Compile evidence:

- wave32, SGPR `28`, VGPR `134`, LDS `3808`, no spills;
- comparison: BN96 was VGPR `181` no-spill, BN104 spilled `9`, and BN112
  spilled `24`.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on BN64;
- p512 and p513 selected BN112 split-qsum.
- default regate passed: p33 stayed on BN64, p512/p513 selected BN112
  split-qsum, and rollback returned p512 to BN96.

Focused timing versus current BN96/BN64:

| Size | Row | Default | Split-qsum | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `664.28 us` | `525.19 us` | `0.791x` |
| p512 | `Qcur-0` | `2038.53 us` | `1990.77 us` | `0.977x` |
| p512 | `ffn_out-0` | `8273.41 us` | `7583.48 us` | `0.917x` |
| p512 | `ffn_gate-0` | `7640.07 us` | `7040.21 us` | `0.921x` |
| p512 | `result_output` | `65759.52 us` | `64159.07 us` | `0.976x` |
| p513 | `Vcur-0` | `655.01 us` | `543.96 us` | `0.830x` |
| p513 | `Qcur-0` | `2069.32 us` | `2014.67 us` | `0.974x` |
| p513 | `ffn_out-0` | `8311.01 us` | `7669.28 us` | `0.923x` |
| p513 | `ffn_gate-0` | `7610.64 us` | `7259.28 us` | `0.954x` |
| p513 | `result_output` | `66845.24 us` | `64617.43 us` | `0.967x` |

Model A/B on Llama 3.1 8B Q8_0:

- p512/fa1/r3: `430.303 -> 435.902 tok/s`;
- p513/fa1/r3 single graph: `409.652 -> 424.678 tok/s`.

Decision:
accept as the gfx1151 production-width Q8_0 default. This is not an exact
Vulkan cooperative-matrix clone and it does not solve the parity gap; it is a
packed-path register-tile improvement that keeps progress evidence based while
the cooperative-store path remains unavailable in HIP C++.

## Accepted Packed-Q8_1 BN128 Split-Qsum Full-Column Policy

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32`

Policy:
default on gfx1151 only for Q8_0 prompt rows with `cols % 128 == 0`.
Rollback:
`GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_PROMPT=1`.
Broad diagnostic opt-in:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_PROMPT=1`.

Purpose:
after BN112 split-qsum proved the simple column-widening spill wall was a
`qsum` live-range problem, this candidate applies the same split-qsum spelling
to BN128. It does not clone Vulkan's cooperative-matrix store path, but it does
move the packed-Q8_1 fallback route to Vulkan's 128-column workgroup
denominator for exact full-column p512-style rows.

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-compile-20260618/`;
- focused broad probe:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-focused-20260618/`;
- default/rollback regate:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-default-regate-20260618/`;
- model A/B:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-default-model-ab-20260618/`.

Compile evidence:

- wave32, SGPR `27`, VGPR `152`, LDS `4352`, no spills;
- comparison: simple BN128 was VGPR `192` with `55` spills, so split-qsum
  removes the old BN128 compile-resource blocker.

Correctness and route evidence:

- broad opt-in p33, p512, and p513 CPU-reference gates passed;
- broad opt-in route traces: p33 stayed on BN64, p512/p513 selected BN128;
- default regate: p33 stayed on BN64, p512 selected BN128, p513 stayed on
  BN112, and rollback returned p512 to BN112.

Focused timing versus current BN112 split-qsum:

| Size | Row | BN112 default | BN128 split-qsum | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `511.42 us` | `555.20 us` | `1.086x` |
| p512 | `Qcur-0` | `1952.62 us` | `1903.90 us` | `0.975x` |
| p512 | `ffn_out-0` | `7294.67 us` | `7162.99 us` | `0.982x` |
| p512 | `ffn_gate-0` | `6844.92 us` | `6394.20 us` | `0.934x` |
| p512 | `result_output` | `62651.74 us` | `56614.31 us` | `0.904x` |
| p513 | `Vcur-0` | `535.05 us` | `583.75 us` | `1.091x` |
| p513 | `Qcur-0` | `2005.80 us` | `2145.12 us` | `1.070x` |
| p513 | `ffn_out-0` | `7555.99 us` | `8102.52 us` | `1.072x` |
| p513 | `ffn_gate-0` | `6989.24 us` | `7713.72 us` | `1.104x` |
| p513 | `result_output` | `63810.17 us` | `68314.07 us` | `1.071x` |

Model A/B on Llama 3.1 8B Q8_0 p512/fa1/r3:

- default BN128 full-column policy: `456.488968 tok/s`;
- rollback to BN112: `440.297327 tok/s`;
- ratio: `1.0368x`.

Decision:
accept only as a full-column gfx1151 default. This is a good example of why
odd/tail gates are mandatory: the same BN128 schedule wins p512 because it
reduces column workgroups from five to four, but broad BN128 regresses p513
because the fifth tail group remains and the wider tile does more wasted work.
The next Q8_0 parity step still needs to target the RADV cooperative-matrix
load/store/lane-ownership delta or a lower-level equivalent.

## Rejected Packed-Q8_1 BN104 Split-Qsum Tail Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x104_splitqsum_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X104_SPLITQSUM_PROMPT=1`

Purpose:
simple BN104 was rejected at the compile gate because `qsum[26]` hit the
192-VGPR spill cliff. After BN112 and BN128 split-qsum proved that splitting
the temporary qsum tile removes that spill wall, this probe retested BN104 as
a p513/tail bracket. p513 uses five column groups at BN104 with 520 covered
columns, versus BN112 covering 560 and broad BN128 covering 640.

Artifacts:

- focused:
  `cache/hrxv1/gfx1151/q8_0-mmq64x104-splitqsum-focused-20260618/`;
- p513 model A/B:
  `cache/hrxv1/gfx1151/q8_0-mmq64x104-splitqsum-p513-model-ab-20260618/`;
- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x104_splitqsum.hsaco`.

Compile evidence:

- wave32, SGPR `28`, VGPR `127`, no spills, private segment `0`;
- this confirms the simple BN104 failure was the qsum live-range spelling, not
  BN104 coverage itself.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on BN64;
- p512 and p513 selected BN104 split-qsum under the opt-in env.

Focused timing versus current default routing:

| Size | Row | Default | BN104 split-qsum | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `546.57 us` | `512.41 us` | `0.937x` |
| p512 | `Qcur-0` | `1883.45 us` | `1936.25 us` | `1.028x` |
| p512 | `ffn_out-0` | `6963.63 us` | `7451.56 us` | `1.070x` |
| p512 | `ffn_gate-0` | `6348.36 us` | `6739.71 us` | `1.062x` |
| p512 | `result_output` | `55810.40 us` | `61559.98 us` | `1.103x` |
| p513 | `Vcur-0` | `530.79 us` | `520.74 us` | `0.981x` |
| p513 | `Qcur-0` | `2021.82 us` | `1972.03 us` | `0.975x` |
| p513 | `ffn_out-0` | `7613.79 us` | `7782.58 us` | `1.022x` |
| p513 | `ffn_gate-0` | `7257.82 us` | `6742.63 us` | `0.929x` |
| p513 | `result_output` | `63491.00 us` | `62140.38 us` | `0.979x` |

Model A/B on Llama 3.1 8B Q8_0 p513/fa1/r3:

- current default BN112: `427.339702 tok/s`;
- BN104 split-qsum: `428.405933 tok/s`;
- ratio: `1.0025x`.

Decision:
reject production promotion. BN104 split-qsum is a valid compile/runtime data
point and has a mild p513 focused signal, but p512 regresses sharply and the
p513 model lift is only `0.25%`, too small to justify a new tail default. Keep
the current p512 BN128 full-column and p513 BN112 policies.

## Rejected Packed-Q8_1 BN128 Split-Qsum8 Live-Range Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum8_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM8_PROMPT=1`

Purpose:
preserve the accepted BN128 full-column packed-Q8_1 route but reduce qsum
chunk size from 16 columns to 8 columns. This tests whether lower live range
and VGPR pressure can improve occupancy enough to beat the extra chunk-loop
overhead.

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum8-compile-20260618/`;
- focused:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum8-focused-20260618/`.

Compile evidence:

- split-qsum8: wave32, SGPR `27`, VGPR `120`, LDS `4352`, no spills;
- accepted split-qsum16: wave32, SGPR `27`, VGPR `152`, LDS `4352`, no
  spills.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on BN64;
- p512 and p513 selected BN128 split-qsum8 under the opt-in env.

Focused timing versus current default routing:

| Size | Row | Default | Split-qsum8 | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `541.25 us` | `538.38 us` | `0.995x` |
| p512 | `Qcur-0` | `1874.22 us` | `1778.22 us` | `0.949x` |
| p512 | `ffn_out-0` | `7048.63 us` | `6558.90 us` | `0.931x` |
| p512 | `ffn_gate-0` | `6303.30 us` | `6843.30 us` | `1.086x` |
| p512 | `result_output` | `56355.07 us` | `61269.62 us` | `1.087x` |
| p513 | `Vcur-0` | `537.46 us` | `557.28 us` | `1.037x` |
| p513 | `Qcur-0` | `2040.09 us` | `2183.83 us` | `1.071x` |
| p513 | `ffn_out-0` | `7958.16 us` | `8170.67 us` | `1.027x` |
| p513 | `ffn_gate-0` | `7449.29 us` | `7793.47 us` | `1.046x` |
| p513 | `result_output` | `64570.31 us` | `71864.74 us` | `1.113x` |

Decision:
reject production promotion. The live-range reduction is real and useful as a
compile fact, but lower VGPR alone is not the active bottleneck for the
dominant rows. The extra chunk-loop/scheduling tradeoff regresses
`result_output` on p512 and every p513 row, so keep the accepted split-qsum16
full-column policy.

## Rejected Packed-Q8_1 BM32/BN128 Output-Ownership Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq32x128_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ32X128_PROMPT=1`

Purpose:
preserve BN128 full-column coverage but change ownership from the accepted
BM64/COLS32 split-qsum route to BM32/COLS16 with a simple qsum tile. This is
the remaining packed-path axis called out by the ledger: different register
tile/output ownership without requiring a missing cooperative-matrix store
primitive.

Artifacts:

- compile:
  `cache/hrxv1/gfx1151/q8_0-mmq32x128-compile-20260618/`;
- focused:
  `cache/hrxv1/gfx1151/q8_0-mmq32x128-focused-20260618/`.

Compile evidence:

- wave32, SGPR `27`, VGPR `135`, LDS `4352`, no spills;
- this is lower pressure than accepted BN128 split-qsum16 at VGPR `152`, so it
  passed the compile-resource gate.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- p33 stayed on the narrow/default path;
- p512 and p513 selected BM32/BN128 under the opt-in env.

Focused timing versus current default routing:

| Size | Row | Default | BM32/BN128 | Ratio |
| --- | --- | ---: | ---: | ---: |
| p512 | `Vcur-0` | `550.04 us` | `625.05 us` | `1.136x` |
| p512 | `Qcur-0` | `1882.69 us` | `2087.72 us` | `1.109x` |
| p512 | `ffn_out-0` | `7043.92 us` | `8482.13 us` | `1.204x` |
| p512 | `ffn_gate-0` | `6416.43 us` | `7242.67 us` | `1.129x` |
| p512 | `result_output` | `55913.71 us` | `64160.60 us` | `1.147x` |
| p513 | `Vcur-0` | `531.02 us` | `717.80 us` | `1.352x` |
| p513 | `Qcur-0` | `2018.92 us` | `2432.01 us` | `1.205x` |
| p513 | `ffn_out-0` | `7762.03 us` | `9816.67 us` | `1.265x` |
| p513 | `ffn_gate-0` | `7209.13 us` | `8530.12 us` | `1.183x` |
| p513 | `result_output` | `63218.36 us` | `76252.10 us` | `1.206x` |

Decision:
reject production promotion. The compile-pressure hypothesis was plausible,
but reducing per-thread output columns loses row-tile amortization and
regresses every production-width row. Keep the current BN128 split-qsum16
full-column default and do not continue local Q8 packed-path tuning unless the
next probe introduces a materially new schedule fact.

## Rejected Packed-Q8_1 BN128 B-Stripe Issue-Window Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_bstripe_wg256_f32`

Gate:
not runtime-selected.

Purpose:
preserve the accepted BM64/BN128 split-qsum route but explicitly preload each
16-column B-cache `iqs` stripe into a local array before dot consumption. This
tests whether source-visible B issue ordering can move the packed path off the
shared first-dot schedule without changing row/column ownership.

Artifact:
`cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-bstripe-compile-20260618/`.

Compile evidence:

| Route | Wave | VGPR | LDS | Spills | Dot sites | First-dot score |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| accepted BN128 split-qsum | 32 | 152 | 4352 | 0 | 512 | `28 loads, lgkmcnt(14), 60 hot ops` |
| B-stripe split-qsum | 32 | 152 | 4352 | 0 | 512 | `28 loads, lgkmcnt(14), 60 hot ops` |

Decision:
reject before runtime. The source spelling compiles cleanly through
CMake/Ninja, but the emitted schedule/resource contract is identical to the
accepted BN128 split-qsum route on the tracked axes. This closes local B-cache
stripe preloading as a Q8_0 parity axis; the next packed-path candidate needs
a materially different issue/window primitive or a lower-level path.

## VK128 Buffer-Store Runtime-Correctness Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-bufferstore-compile-20260618/`

Focused artifact:
`cache/hrxv1/gfx1151/q8_0-bufferstore-make-rsrc-focused-20260618/`

What changed:

- preserved direct-F32 VK128 wave64 WMMA math;
- included the stage-allocation LDS footprint so LDS bytes match RADV;
- replaced both full-tile and guarded edge/tail writeback with raw
  `buffer_store_b32` via `__builtin_amdgcn_raw_buffer_store_b32`.
- replaced the ad hoc resource descriptor with
  `__builtin_amdgcn_make_buffer_rsrc`.

Compile-contract result:

| Fact | RADV large Q8_0 | HIP bufferstore |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| `s_barrier` | `2` | `2` |
| spills | `0` | `0` |
| `buffer_store_b32` | `192` | `128` |
| `ds_load_u16_d16` | `128` | `0` |
| `ds_store_b16` | `128` | `2` |

Decision:
reject for runtime correctness. The original static `192`-store Vulkan count is
too coarse because the RADV shader contains full-aligned and fallback writeback
paths; the p512 full-aligned production path is closer to the direct HIP
buffer-store axis than the whole-shader count suggests. However, the CMake-built
HIP route still failed focused p512 CPU-reference rows with `ERR=inf` on every
large row that selected it: `ffn_out-0`, `ffn_gate-0`, and `result_output`.
`Vcur-0` and `Qcur-0` passed only because they stayed on the existing packed
route. Do not run model timing for this candidate until the raw buffer-store
descriptor/store semantics are proven correct or the source is replaced with a
lower-level spelling that reproduces RADV's cooperative halfword LDS topology.

## LDS Halfword-Stage Primitive Fixture

Fixture:
`hrx-hip-bench-lds-halfword-stage`

Artifact:
`cache/hrxv1/gfx1151/lds-halfword-stage-fixture-20260618/`

Purpose:
prove whether HIP C++ can emit and correctly use the RADV fallback-store
halfword LDS primitive before putting that topology into another Q8_0 route.

Result:

- typed C++ shared-halfword staging passed values but did not emit the same DS
  opcode shape;
- explicit inline asm emitted `ds_store_b16` and `ds_load_u16_d16`;
- asm mode passed deterministic value checks for `tiles=1`, `tiles=16`, and
  `tiles=17`;
- the d16 load helper needs its own `s_waitcnt lgkmcnt(0)` before the loaded
  value is consumed.

Opcode count from the fixture disassembly:

| Opcode | Count |
| --- | ---: |
| `ds_store_b16` | 4 |
| `ds_load_u16_d16` | 4 |
| `global_store_b16` | 8 |
| `s_barrier` | 2 |
| `s_waitcnt` | 11 |

Decision:
accept as a primitive proof only. The next route should not be another
high-level aggregate experiment; it should mechanically insert this controlled
halfword-stage helper into the RADV fallback/partial writeback analogue, then
compare p33/p512/p513 correctness, route traces, and ISA against the Vulkan
oracle before any production timing claim.

## Rejected B64GROUP Dualstage Bufferstore Compile-Contract Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_dualstage_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-dualstage-bufferstore-compile-20260618/`

What changed:

- preserved the B64GROUP direct-F32 VK128 wave64 WMMA math/load path;
- preserved raw gfx11 `buffer_store_b32` output stores;
- used the controlled inline `ds_write_b16` and `ds_read_u16_d16` primitive to
  force a duplicated halfword LDS stage while keeping selected output values
  unchanged;
- kept the route out of runtime selection pending the static contract screen.

Compile-contract result:

| Fact | RADV large Q8_0 | HIP dualstage bufferstore |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `130` |
| `buffer_store_b32` | `192` | `64` |
| `s_barrier` | `2` | `34` |
| VGPR | `192` | `195` |

Decision:
reject before runtime. This is an important narrowing result: inline DS can
produce the RADV halfword LDS opcode class from HIP C++, but manual staging
still recreates the old barrier wall and does not reproduce the cooperative
matrix store/lane-ownership writeback. Do not spend another route on manual
per-tile LDS staging unless it also removes the per-tile barriers and raises
the writeback ownership toward the RADV `192 buffer_store_b32` shape.

## Coopmat Store Contract Fixture

Fixture:
`hrx-hip-bench-coopmat-store-contract`

Artifact:
`cache/hrxv1/gfx1151/coopmat-store-contract-fixture-20260618-122510/`

Purpose:
separate the raw writeback surface from the full Q8_0 direct-WMMA route. The
dualstage bufferstore probe proved HIP can emit RADV-like halfword LDS opcodes
but still missed `192 buffer_store_b32` and reintroduced `34` barriers. This
fixture asks whether source-visible HIP can emit the raw store surface at all.

Result:

| Kernel | `buffer_store_b32` | `s_barrier` | `s_waitcnt` | Notes |
| --- | ---: | ---: | ---: | --- |
| `coopstore_probe_linear64` | `64` | `0` | `2` | old HIP direct-route store count |
| `coopstore_probe_linear128` | `128` | `0` | `2` | widened raw writeback |
| `coopstore_probe_linear192` | `192` | `0` | `2` | matches RADV static store count |
| `coopstore_probe_branch192` | `192` | `0` | `49` | 48 guarded 4-store groups |

All modes passed exact value checks using descriptor flags `0x31004000`.

Decision:
accept as evidence only. The store opcode/count is source-expressible in HIP
C++; the production miss is the accumulator ownership and mapping into those
store groups, plus the cooperative low-barrier LDS topology. The next
direct-WMMA candidate should start from lane-to-output ownership and WMMA
accumulator layout, not from another raw-store-count tweak.

## Rejected Fullpair Bufferstore Compile-Contract Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_fullpair_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-fullpair-bufferstore-compile-20260618-122909/`

What changed:

- kept the B64GROUP direct-F32 VK128 wave64 WMMA math/load path;
- kept the 22528-byte LDS allocation and raw buffer-store output path;
- widened only the full-tile buffer-store path to write both accumulator halves
  while leaving the guarded edge/fallback path scalar;
- kept the route out of runtime selection because this is an accumulator
  ownership/static-schedule probe, not a correctness candidate.

Compile-contract result:

| Fact | RADV large Q8_0 | HIP fullpair bufferstore |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `buffer_store_b32` | `192` | `192` |
| `s_barrier` | `2` | `2` |
| `ds_load_u16_d16` | `128` | `0` |
| `ds_store_b16` | `128` | `2` |
| VGPR | `192` | `195` |

Decision:
reject before runtime. This is the first real direct-WMMA Q8_0 kernel variant
to match RADV's raw store count while preserving the matching WMMA count,
grouped LDS fragment loads, LDS allocation, zero spills, and two barriers.
The remaining direct-WMMA gap is now sharply isolated: the route still lacks
the RADV halfword LDS cooperative store/load topology and is three VGPRs over
the target. The next source-visible attempt must combine the successful
low-barrier 192-store ownership with a low-barrier halfword LDS topology; the
older dualstage path matched halfword opcodes only by regressing to `34`
barriers.

## LDS Halfword Bulk128 WG256 Primitive Fixture

Fixture:
`hrx-hip-bench-lds-halfword-stage --mode=bulk128-wg256`

Artifact:
`cache/hrxv1/gfx1151/lds-halfword-stage-bulk128-wg256-20260618-123403/`

Purpose:
prove whether the remaining RADV halfword LDS topology is source-expressible
without the old per-tile barrier wall.

Result:

| Kernel | `ds_store_b16` | `ds_load_u16_d16` | `s_barrier` | Correct |
| --- | ---: | ---: | ---: | --- |
| `lds_halfword_stage_probe_bulk128` | `128` | `128` | `0` | yes |
| `lds_halfword_stage_probe_bulk128_wg256` | `128` | `128` | `2` | yes |

The `bulk128-wg256` strict halfword contract check passed against the RADV
large Q8_0 oracle for `ds_load_u16_d16`, `ds_store_b16`, and `s_barrier <= 2`.

Decision:
accept as a primitive proof. The two previously isolated pieces of the RADV
large Q8_0 schedule are now both expressible in CMake-built HIP C++:

- the real direct-WMMA kernel can emit `192 buffer_store_b32` with `2` barriers;
- a 256-thread fixture can emit `128 ds_store_b16` plus
  `128 ds_load_u16_d16` with `2` barriers.

The next direct-WMMA compile-contract probe should combine these two mechanics
inside the Q8_0 route and target the remaining VGPR/lane-mapping mismatch.

## Rejected Fast-Half Fullpair Bufferstore Compile-Contract Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_fast_half_fullpair_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-fast-half-fullpair-bufferstore-compile-20260618-123820/`

What changed:

- kept the B64GROUP direct-F32 VK128 wave64 WMMA math/load path;
- kept the raw gfx11 fullpair `buffer_store_b32` writeback that reached RADV's
  192-store surface;
- replaced the old per-tile dualstage output staging with a fast halfword LDS
  stage that performs the output-half DS stores/loads without inserting
  per-output-tile `__syncthreads`;
- kept the route out of runtime selection because the fullpair accumulator
  mapping is still a static schedule probe, not a correctness candidate.

Compile-contract result:

| Fact | RADV large Q8_0 | HIP fast-half fullpair |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `130` |
| `buffer_store_b32` | `192` | `192` |
| `s_barrier` | `2` | `2` |
| VGPR | `192` | `195` |

Decision:
reject before runtime, but treat this as the closest current direct-WMMA
Vulkan schedule convergence. It combines the previously isolated wins in the
real Q8_0 kernel: RADV's WMMA count, grouped LDS loads, halfword LDS loads,
raw buffer-store count, LDS footprint, two barriers, and zero spills all now
coexist. The remaining strict failures are narrow: two extra `ds_store_b16`
sites and three extra VGPRs. Inspection places the two extra stores before the
final output-stage loop in the pre-output staging/control region, not in the
new final fast-half writeback. The next mechanical step should preserve the
matched final-store surface and reduce prewrite live range/staging pressure
until the static contract is exact, then move to focused correctness and timing.

## Rejected Pack-Stage Fast-Half Fullpair Bufferstore VGPR Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_fullpair_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-packstage-fast-half-fullpair-bufferstore-compile-20260618-124840/`

What changed:

- preserved the matched B64GROUP direct-F32 VK128 wave64 WMMA math/load path;
- preserved the fast-half output-stage and fullpair raw buffer writeback;
- changed A/B shared-memory staging from scalar half stores to packed
  `ds_write_b32` pair stores so the two pre-output scalar `ds_store_b16` sites
  would no longer pollute the strict RADV halfword-store count.

Compile-contract result:

| Fact | RADV large Q8_0 | HIP pack-stage fast-half |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `128` |
| `buffer_store_b32` | `192` | `192` |
| `s_barrier` | `2` | `2` |
| VGPR | `192` | `196` |

Decision:
reject before runtime due VGPR pressure, but this is the first Q8_0
direct-WMMA HIP C++ object to match the strict RADV opcode, LDS, barrier, and
spill contract on the selected axes. It adds two A/B-stage `ds_store_b32`
sites, which are the packed replacements for the prior scalar A/B staging
stores. The remaining strict failure is now register pressure only:
`VGPR=196` versus RADV's `192`.

## Rejected Pack-Stage Pressure Brackets

Candidates:

- `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_dummyhalf_fullpair_bufferstore_f16acc_wg256_f32`
- `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_fullpair_bufferstore_lb2_f16acc_wg256_f32`

Artifacts:

- `cache/hrxv1/gfx1151/q8_0-packstage-dummyhalf-fullpair-bufferstore-compile-20260618-124840/`
- `cache/hrxv1/gfx1151/q8_0-packstage-fast-half-fullpair-bufferstore-lb2-compile-20260618-124958/`

What changed:

- dummy-half bracket: still issued `128 ds_load_u16_d16`, but stored the
  accumulator values directly instead of consuming the halfword reload values;
- launch-bounds bracket: kept the real fast-half reload/writeback path and
  changed the kernel annotation to `__launch_bounds__(256, 2)`.

Result:
both variants preserved the exact opcode/LDS/barrier/spill surface from the
pack-stage fast-half probe, and both remained at `VGPR=196`.

Decision:
reject before runtime. The remaining pressure is not caused by consuming the
halfword reload values, and a launch-bounds hint does not lower allocation.
The next exact-schedule attempt needs a real live-range or ownership change
while preserving the pack-stage opcode contract.

## Stream-Column Pack-Stage Static Contract Pass

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamcol_packstage_fast_half_fullpair_bufferstore_f16acc_wg256_f32`

Static artifact:
`cache/hrxv1/gfx1151/q8_0-streamcol-packstage-fast-half-fullpair-bufferstore-compile-20260618-125457/`

Runtime artifact:
`cache/hrxv1/gfx1151/q8_0-streamcol-packstage-runtime-diagnostic-20260618-130143/`

What changed:

- preserved the pack-stage fast-half opcode surface;
- kept the four A fragments live for each `k_tile`;
- streamed one B column fragment at a time instead of keeping four B fragments
  live simultaneously;
- preserved the same total `64 ds_load_b64` count by loading each B column
  fragment once per `k_tile`.

Strict contract result:

| Fact | RADV large Q8_0 | HIP stream-column pack-stage |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `128` |
| `buffer_store_b32` | `192` | `192` |
| `s_barrier` | `2` | `2` |
| VGPR | `192` | `188` |

Decision:
accept as a static contract pass, but reject for runtime promotion. After
adding the missing catalog route and opt-in selector, focused p512/p513 gates
selected this provider on the intended large rows. The selected rows failed
CPU-reference with `HRX0=inf` against finite CPU values and p512 sentinel
mismatches. This is nevertheless an important milestone: CMake-built HIP C++
can now reproduce the selected RADV large Q8_0 schedule surface on the checked
opcode/resource axes. The next route must preserve this stream-column
pack-stage schedule while replacing the diagnostic fullpair writeback with a
correct accumulator lane/output coordinate map.

## Pack-Stage Selected-Writeback Correctness Control

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifacts:

- correctness:
  `cache/hrxv1/gfx1151/q8_0-packstage-bufferstore-route-20260618-132009/`
- ISA comparison:
  `cache/hrxv1/gfx1151/q8_0-packstage-bufferstore-isa-20260618-132038/`

What changed:

- kept the B64GROUP direct-F32 VK128 wave64 WMMA math/load path;
- kept packed A/B LDS staging through `ds_store_b32`;
- removed stream-column issue order;
- removed the full-pair halfword output stage;
- used the known-correct selected-half raw `buffer_store_b32` writeback.

Correctness result:
focused p512 and odd p513 `MUL_MAT` gates passed. Route traces selected this
provider on the large rows covered by the diagnostic domain: p512 selected
`ffn_out`, `ffn_gate`, and `result_output`; p513 selected those same large/tail
rows. Smaller rows stayed on q8_1/default routes as intended.

Static result:

| Fact | RADV large Q8_0 | HIP pack-stage selected |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `20480` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `0` |
| `ds_store_b16` | `128` | `0` |
| `ds_store_b32` | `0` | `2` |
| `buffer_store_b32` | `192` | `128` |
| `s_barrier` | `2` | `2` |
| VGPR | `192` | `196` |

Decision:
accept as an opt-in correctness control only. It is not a Vulkan-exact route,
but it proves the packed A/B LDS staging and non-stream B64GROUP issue order
are CPU-reference clean at p512 and odd p513. The remaining target is now
narrower: keep the stream-column static contract while replacing the full-pair
halfword writeback with a proven gfx1151 accumulator lane/output map. Aggregate
model throughput should remain a later acceptance gate, not the search method
for this step.

## Stream-Column OPSEL Bracket

Temporary candidate:
the committed pack-stage selected-writeback wrapper rebuilt with
`HRX_Q8_0_WMMA_VK128_W64_B64GROUP_STREAM_COL=1` and
`HRX_Q8_0_WMMA_VK128_W64_OPSEL=1`, then restored.

Artifact:
`cache/hrxv1/gfx1151/q8_0-streamcol-opsel1-selected-buffer-20260618-132354/`

Question:
is the stream-column selected-writeback CPU-reference mismatch caused by the
low accumulator half, or by the stream-column WMMA issue/order?

Result:
the temporary high-OPSEL route selected on the same large p512/p513 rows and
failed with the same finite error band as low OPSEL:

| Row class | Error range |
| --- | ---: |
| p512 large rows | `0.00317-0.00367` |
| p513 large rows | `0.00278-0.00365` |

Decision:
reject and do not add a catalog route. The stream-column mismatch is not an
OPSEL-half mapping issue. Combined with the AMD wave64 mapping rule that only
one accumulator half is selected by OPSEL, this also rules out the current
`row+16` full-pair writeback as a valid correctness path. The next schedule
probe should target pressure reduction in the non-stream accumulation order,
or establish a Vulkan-equivalent numerical gate for stream-column before using
it as the production schedule.

## Stream-Row Pack-Stage Static-Contract Bracket

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamrow_packstage_fast_half_fullpair_bufferstore_f16acc_wg256_f32`

Runtime selection:
not wired into runtime selection; compile/ISA contract probe only.

Artifact:
`cache/hrxv1/gfx1151/q8_0-streamrow-packstage-fast-half-fullpair-bufferstore-compile-20260618-133139/`

What changed:

- preserved the pack-stage fast-half opcode surface;
- kept four B fragments resident for each `k_tile`;
- streamed one A row fragment at a time to reduce pressure without the
  stream-column B-lifetime/order change;
- preserved the same total `64 ds_load_b64` count and the same full-pair
  halfword output-stage opcode surface.

Strict contract result:

| Fact | RADV large Q8_0 | HIP stream-row pack-stage |
| --- | ---: | ---: |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| LDS bytes | `22528` | `22528` |
| spills | `0` | `0` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `128` |
| `buffer_store_b32` | `192` | `192` |
| `s_barrier` | `2` | `2` |
| VGPR | `192` | `189` |

Decision:
accept as the current best static-contract schedule, but reject for runtime
promotion. It does not use the stream-column issue order that produced finite
CPU-reference mismatches, and it clears the RADV VGPR ceiling, but it still
inherits the diagnostic full-pair writeback. The current full-pair mapping
stores `acc[2*reg+0]` and `acc[2*reg+1]` as if they represented rows separated
by `+16`; gfx11 wave64 WMMA instead updates only the OPSEL-selected accumulator
half for the 16x16 output tile. The next mechanical step is a row-stream
selected-writeback correctness gate, followed by a controlled full-output
lane-map fixture if exact Vulkan halfword writeback remains necessary.

## Stream-Row Selected-Writeback Runtime Gate

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_streamrow_packstage_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_STREAMROW_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifacts:

- correctness:
  `cache/hrxv1/gfx1151/q8_0-streamrow-packstage-bufferstore-q8route-20260618-133828/`
- ISA comparison:
  `cache/hrxv1/gfx1151/q8_0-streamrow-packstage-bufferstore-isa-20260618-133905/`

Question:
can the row-stream pressure reduction keep strict correctness when paired with
the known-correct selected-half writeback?

Result:
no. p512 and p513 selected the route on `ffn_out`, `ffn_gate`, and
`result_output`, and those rows failed with finite strict errors:

| Shape | Error range |
| --- | ---: |
| p512 large rows | `0.00381-0.00394` |
| p513 large rows | `0.00357-0.00389` |

p33 stayed on existing Q8_1 routes and passed, proving the shape gate did not
poison narrow odd prefill.

Decision:
reject for promotion. The selected-half result shows the row-stream issue
order itself is not CPU-reference clean, independent of the invalid full-pair
writeback. Since both stream-column and stream-row fail strict correctness, the
next mechanical path is to lower pressure while preserving the original
non-stream WMMA accumulation order, or define and prove a Vulkan-equivalent
numerical acceptance rule for stream-order variants before timing them.

## Full-Tile Selected-Writeback Runtime Gate

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fulltile_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_FULLTILE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q8_0-packstage-fulltile-bufferstore-q8route-20260618-134640/`

Question:
can an exact full-tile specialization reduce pressure while preserving the
strict-correct non-stream WMMA issue order?

Compile result:

| Fact | Pack-stage selected control | Full-tile probe |
| --- | ---: | ---: |
| VGPR | `196` | `193` |
| SGPR | `28` | `24` |
| LDS bytes | `20480` | `20480` |
| spills | `0` | `0` |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| `ds_load_b64` | `64` | `64` |
| `ds_store_b32` | `2` | `2` |
| `buffer_store_b32` | `128` | `64` |
| `s_barrier` | `2` | `2` |

Correctness and routing:
p512 selected the probe on `ffn_out`, `ffn_gate`, and `result_output`, and all
five Q8_0 focused rows passed. p33 and p513 were guarded out by the
full-tile-only selector and passed on existing Q8_1 routes.

Focused timing:

| Row | Default Q8_1 us | Full-tile probe us | Ratio |
| --- | ---: | ---: | ---: |
| `ffn_out` | `7031.762` | `8422.038` | `1.198` |
| `ffn_gate` | `6433.416` | `7332.741` | `1.140` |
| `result_output` | `56130.024` | `64645.333` | `1.152` |

Decision:
reject for promotion. The pressure direction is useful (`196 -> 193`) and it
preserves strict correctness, but it still misses the RADV `192` VGPR ceiling,
cuts the store surface to `64 buffer_store_b32`, and regresses the focused
production rows versus the current default.

## Named-Fragment Pack-Stage Compile Bracket

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_namedfrag_packstage_bufferstore_f16acc_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_NAMEDFRAG_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q8_0-namedfrag-packstage-bufferstore-compile-20260618-135058/`

Question:
does removing fragment array indexing lower pressure while preserving the same
strict-correct non-stream WMMA issue order?

Compile result:
no. The CMake-built HSACO is resource-equivalent to the indexed selected
control:

| Fact | Indexed selected control | Named-fragment probe |
| --- | ---: | ---: |
| VGPR | `196` | `196` |
| SGPR | `28` | `28` |
| LDS bytes | `20480` | `20480` |
| spills | `0` | `0` |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| `ds_load_b64` | `64` | `64` |
| `ds_store_b32` | `2` | `2` |
| `buffer_store_b32` | `128` | `128` |
| `s_barrier` | `2` | `2` |

Decision:
reject at compile-evidence gate. Fragment array indexing is not the source of
the four-register gap versus RADV.

## Vulkan0 CPU-Reference Contract

Artifact:
`cache/hrxv1/gfx1151/q8_0-vulkan0-focused-cpuref-20260618-140129/`

Invalidated artifact:
`cache/hrxv1/gfx1151/q8_0-vulkan-focused-cpuref-20260618-135708/`

Result:
`Vulkan0` passes the same strict `test-backend-ops test` CPU-reference gate on
the exported Q8_0 p33, p512, and p513 rows: five rows per prompt size, zero
failures. The earlier `-b Vulkan` run produced zero rows because the device is
named `Vulkan0`; `test-backend-ops` now fails if a backend filter matches no
device.

Decision:
do not accept the stream-column or stream-row finite-error probes as
Vulkan-equivalent. Vulkan itself is strict-clean on these rows, so any Q8_0
route promotion must either preserve strict correctness or first establish a
separate model-stability policy with stronger evidence than "the error is
small." Until then, the exact-schedule path remains:

- preserve the non-stream WMMA accumulation order for correctness;
- keep the static RADV contract target in view: `32` WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`,
  `22528` LDS bytes, two barriers, no spills, and VGPR at or below `192`;
- solve pressure and lane ownership without using the currently invalid
  full-pair `row+16` output mapping.

## WMMA F16 Wave64 Lane-Map Fixture

Fixture:
`hrx-hip-bench-wmma-f16-lane-map`

Artifact:
`cache/hrxv1/gfx1151/wmma-f16-lane-map-20260618-140647/`

Question:
does `v_wmma_f16_16x16x16_f16_w64` expose the odd accumulator slots as a
second output row band, or does OPSEL select one half of the same 16x16 output
ownership surface?

Method:
the fixture initializes all eight accumulator slots per lane with unique
sentinels, runs one all-ones WMMA, and records which slots changed for OPSEL 0
and OPSEL 1.

Runtime result:

| OPSEL | Changed even slots | Changed odd slots | Bad checks |
| --- | ---: | ---: | ---: |
| `0` | `256` | `0` | `0` |
| `1` | `0` | `256` | `0` |

All changed slots advanced by the expected all-ones dot contribution `16`; all
unchanged slots retained their sentinel values.

ISA/resource evidence:

- wavefront size: `64`;
- SGPR: `12`;
- VGPR: `13`;
- LDS bytes: `0`;
- spills: `0`;
- each OPSEL kernel emits one `v_wmma_f16_16x16x16_f16`;
- the OPSEL 1 kernel disassembles with `op_sel:[0,0,1]`.

Decision:
the diagnostic fullpair `row+16` mapping is not a valid gfx1151 wave64 WMMA
output map. OPSEL chooses even versus odd accumulator slots for the same
16x16 output surface; it does not produce a second independent 16-row tile.
Future Q8_0 exact-schedule work should not try to fix fullpair correctness by
moving odd slots to `row + 16`. It must either keep selected-half ownership
and reduce pressure another way, or use a lower-level cooperative-store path
with a separately proven lane map.

## RADV Event-Window Extraction

Artifact:
`cache/hrxv1/gfx1151/q8_0-vulkan-coopmat-schedule-extract-20260618-followup/`

Tool update:
`sources/llama.cpp/tools/vulkan-oracle/extract_coopmat_schedule.py` now emits
basic-block event windows for RADV ISA: pre-WMMA LDS load offsets, store
basic-block opcode mixes, and the first WMMA issue window.

Result:
the Q8_0 Vulkan large route is not adequately described by aggregate opcode
counts. The full-aligned path and staged fallback path are distinct in the
RADV ISA:

- full aligned blocks use direct `buffer_store_b32` groups;
- staged blocks use `ds_store_b16`, `ds_load_u16_d16`, and then
  `buffer_store_b32`;
- the pre-WMMA load window groups `ds_load_b64` by two LDS base registers with
  offset families matching the cooperative matrix A/B tile layout;
- the schedule still totals `32` WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`,
  `22528` LDS bytes, `192` VGPR, two barriers, and no spills.

Decision:
until HRX v1 is much closer to Vulkan parity, use aggregate model throughput
only to choose the next boulder and as a final promotion guardrail. Q8_0 work
must continue at the kernel/schedule A+B level: each candidate needs a stated
RADV event-window delta it is trying to close, plus emitted ISA evidence before
focused timing. A candidate that merely matches headline
`BM128/BN128/BK32/WG256/wave64` or coarse opcode totals is not a Vulkan clone
unless the direct and staged writeback lane ownership also match.

## RADV-vs-HIP Event Comparison for Selected Pack-Stage

Artifact:
`cache/hrxv1/gfx1151/q8_0-packstage-bufferstore-eventcompare-20260618-followup/`

Compared HIP route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_bufferstore_f16acc_wg256_f32`

Tool update:
`sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py` now emits the
same event summary for both the RADV ISA and the CMake/Ninja-built HSACO.

Result:
the selected pack-stage route is correctness-clean, but it is not a
RADV-equivalent schedule.

| Fact | RADV large route | HIP selected pack-stage |
| --- | ---: | ---: |
| LDS bytes | `22528` | `20480` |
| VGPR | `192` | `196` |
| `buffer_store_b32` | `192` | `128` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `0` |
| `ds_store_b16` | `128` | `0` |
| `ds_store_b32` | `0` | `2` |
| barriers | `2` | `2` |

The event windows make the mismatch sharper than the opcode table alone:

- RADV has direct full-aligned store blocks and staged fallback blocks, where
  staged blocks combine `ds_store_b16`, `ds_load_u16_d16`, and
  `buffer_store_b32`.
- HIP selected pack-stage has no `ds_store_b16`/`ds_load_u16_d16` writeback
  windows. Its first store windows are descriptor/setup stores followed by
  straight `buffer_store_b32` groups.
- RADV's first WMMA window issues a broad preloaded LDS-read window with
  offsets under two LDS bases. HIP issues repeated `ds_load_b64` plus
  `s_waitcnt lgkmcnt(0)` pairs before early WMMA groups.

Decision:
reject further selected-half pack-stage variants unless the candidate closes a
named event-window delta. The next exact-schedule candidate must either:

- prove a lower-level cooperative-store lane map and reproduce the RADV
  direct/staged writeback split; or
- import a specific RADV issue-window fact into the current packed-Q8_1
  production path without reintroducing the known spill/pressure failures.

Aggregate p512 token rate is not sufficient evidence for this Q8_0 path until
the event-window comparison is close enough to justify model-level promotion.

## Mixed Direct/Staged Coopstore Fixture

Artifact:
`cache/hrxv1/gfx1151/coopmat-store-radv-mixed192-pass-20260618-142609/`

ISA artifact:
`cache/hrxv1/gfx1151/coopmat-store-radv-mixed192-isa-20260618-142532/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/coopmat_store_contract_bench.hip.cpp`

The `radv-mixed192` fixture adds a single WG256 diagnostic kernel that writes
the same store-side opcode surface seen in the Vulkan Q8_0 large route:

- groups 0-15 write directly through raw `buffer_store_b32`;
- groups 16-47 first write halfword values to LDS with `ds_store_b16`;
- the staged groups then reload with `ds_load_u16_d16` and write out through
  raw `buffer_store_b32`.

Runtime result:
the saved clean run passed `12288` checked float outputs with `bad=0`. One
earlier post-rebuild run missed group 35/slot 1, but five immediate reruns and
the saved pass did not reproduce it.

Emitted facts for `coopstore_probe_radv_mixed192`:

| Fact | Count |
| --- | ---: |
| `buffer_store_b32` | `192` |
| `ds_store_b16` | `128` |
| `ds_load_u16_d16` | `128` |
| `s_barrier` | `2` |
| `s_waitcnt` | `135` |

Decision:
this closes one narrow uncertainty: HIP C++ can emit the RADV-like mixed
direct/staged store surface from a normal CMake/Ninja-built source. It does not
make the current Q8_0 route Vulkan-equivalent, because the production gap also
depends on WMMA accumulator ownership, pre-WMMA LDS load issue ordering, and
the exact cooperative store lane map. The next route candidate should reuse
this fixture only as a primitive proof and should advance a named event-window
delta against the RADV oracle.

## First-WMMA Schedule Score

Artifact:
`cache/hrxv1/gfx1151/q8_0-wmma-window-score-20260618-143001/`

Tool update:
`sources/llama.cpp/tools/vulkan-oracle/compare_amdgcn_isa.py` now computes a
first-WMMA schedule score from the event window.

Compared HIP routes:

- `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_bufferstore_f16acc_wg256_f32`
- `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_dualstage_bufferstore_f16acc_wg256_f32`

Score:

| Metric | RADV large route | HIP pack-stage | HIP dualstage |
| --- | ---: | ---: | ---: |
| LDS bytes | `22528` | `20480` | `22528` |
| VGPR | `192` | `196` | `195` |
| `ds_load_b64` total | `64` | `64` | `64` |
| first-window pre-WMMA `ds_load_b64` | `59` | `24` | `24` |
| loads immediately before final pre-WMMA wait | `59` | `1` | `1` |
| final pre-WMMA `lgkmcnt` | `51` | `0` | `0` |
| first-window WMMA count | `22` | `12` | `12` |
| barriers | `2` | `2` | `34` |

Interpretation:
the pack-stage and dualstage routes match RADV's aggregate `64 ds_load_b64`
count, but they still issue a conservative load/wait cadence. RADV keeps many
LDS loads outstanding across two LDS base registers and enters the first WMMA
block with a high `lgkmcnt`, while HIP waits down to zero before early WMMA
groups. The dualstage route additionally pays the known 34-barrier penalty.

Decision:
future Q8_0 exact-schedule candidates need a first-WMMA score improvement
before model-level promotion. A useful candidate should increase
`loads immediately before final pre-WMMA wait` toward RADV while preserving
strict p512/p513 correctness and keeping p33 on the medium/narrow route. If
the score remains `1` with `lgkmcnt(0)`, the candidate is still the same
over-waited HIP schedule even if opcode totals or LDS allocation look closer.

## Nowait B64 Pack-Stage Probe

Artifact:
`cache/hrxv1/gfx1151/q8_0-nowait-packstage-score-20260618-143649/`

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_packstage_bufferstore_f16acc_wg256_f32`

Catalog/source changes:

- `HRX_Q8_0_WMMA_VK128_W64_B64GROUP_NOWAIT` adds no-wait `ds_read_b64`
  helpers to the Q8_0 VK128 source.
- The route is cataloged as a compile-contract probe only; it has no selector
  gate and should not be promoted from aggregate timing.

Score:

| Metric | RADV large route | HIP nowait pack-stage |
| --- | ---: | ---: |
| LDS bytes | `22528` | `20480` |
| VGPR | `192` | `196` |
| `ds_load_b64` total | `64` | `64` |
| first-window pre-WMMA `ds_load_b64` | `59` | `24` |
| loads immediately before final pre-WMMA wait | `59` | `0` |
| final pre-WMMA `lgkmcnt` | `51` | `0` |
| first-window WMMA count | `22` | `16` |
| `s_waitcnt` total | `169` | `8` |
| barriers | `2` | `2` |

Interpretation:
the no-wait helper changes codegen and removes most scalar waits, but it still
does not create the RADV issue window. The first WMMA instructions are emitted
immediately after a 16-load burst, with the remaining loads interleaved after
early WMMA ops. RADV instead carries a much deeper outstanding LDS-load window
and uses high `lgkmcnt` waits through the early WMMA block.

Decision:
reject at compile-evidence gate. The next Q8_0 direct-WMMA attempt needs a
different fragment-lifetime/scheduling spelling or lower-level cooperative
matrix source; simply deleting helper waits does not mechanically converge on
the Vulkan schedule.

## Nowait Named-Fragment Pack-Stage Probe

Artifact:
`cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-packstage-score-20260618-144436/`

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_packstage_bufferstore_f16acc_wg256_f32`

Catalog/source changes:

- The Q8_0 VK128 source now lets the named-fragment branch use no-wait B64 LDS
  helper loads.
- The route is cataloged as a compile-contract probe only; it has no selector
  gate and should not run in model benchmarks.

Score:

| Metric | RADV large route | HIP nowait named-frag |
| --- | ---: | ---: |
| LDS bytes | `22528` | `20480` |
| VGPR | `192` | `196` |
| `ds_load_b64` total | `64` | `64` |
| first-window pre-WMMA `ds_load_b64` | `59` | `24` |
| loads immediately before final pre-WMMA wait | `59` | `0` |
| final pre-WMMA `lgkmcnt` | `51` | `0` |
| first-window WMMA count | `22` | `16` |
| `s_waitcnt` total | `169` | `8` |
| barriers | `2` | `2` |
| `buffer_store_b32` | `192` | `128` |
| `ds_store_b16` / `ds_load_u16_d16` | `128` / `128` | `0` / `0` |

Interpretation:
the source-lifetime spelling does not move the key issue-window metric. It
still emits a 16-load burst followed by early WMMA, then continues interleaving
more loads and WMMA, rather than RADV's deeper outstanding LDS-load window with
high `lgkmcnt` waits.

Decision:
reject at compile-evidence gate. The remaining direct-Q8_0 path should combine
the proven mixed direct/staged store primitive with a genuinely different
load-window or cooperative-matrix spelling; this branch ruled out one cheap
HIP C++ source spelling axis.

## Current Focused Q8_0 Gap After Q5 Tail

Artifacts:

- current HRX/Vulkan scoreboard:
  `cache/hrxv1/gfx1151/current-scoreboard-after-q5tail-20260618-151637/`;
- focused Vulkan0 timing:
  `cache/hrxv1/gfx1151/q8_0-vulkan0-focused-perf-20260618-152147/`.

The post-Q5-tail same-machine model scoreboard moves the Q8_0 p512 row from
the older `0.45x`-class status to `333.327 tok/s` HRX versus
`423.308 tok/s` Vulkan, or `0.787x`. The remaining Q8_0 gap is therefore
narrower but still the weakest clean basket row.

Focused backend-op timing against the current HRX default shows the row-level
gap directly:

| Size | Row | HRX default | Vulkan0 | HRX/Vulkan |
| --- | ---: | ---: | ---: | ---: |
| p512 | `Vcur-0` | `555.202 us` | `280.404 us` | `1.980x` |
| p512 | `Qcur-0` | `1903.903 us` | `1076.645 us` | `1.768x` |
| p512 | `ffn_out-0` | `7162.991 us` | `3964.439 us` | `1.807x` |
| p512 | `ffn_gate-0` | `6394.198 us` | `4287.212 us` | `1.491x` |
| p512 | `result_output` | `56614.310 us` | `38476.810 us` | `1.471x` |
| p513 | `Vcur-0` | `535.054 us` | `271.694 us` | `1.969x` |
| p513 | `Qcur-0` | `2005.796 us` | `1114.676 us` | `1.799x` |
| p513 | `ffn_out-0` | `7555.994 us` | `4007.446 us` | `1.885x` |
| p513 | `ffn_gate-0` | `6989.245 us` | `4718.184 us` | `1.481x` |
| p513 | `result_output` | `63810.167 us` | `46833.738 us` | `1.362x` |

Interpretation:
the accepted BN128/BN112 split-qsum policy is cleanly selected and has no
fallback, but the focused rows are still `1.36x-1.98x` slower than Vulkan.
Aggregate model numbers remain a final guardrail; Q8_0 work should still be
driven by focused row timing and schedule/ISA evidence.

## Rejected Packed-Q8_1 BM128/BN64 Split-Qsum Probe

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x64_splitqsum_wg256_f32`

Gate:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ128X64_SPLITQSUM_PROMPT=1`

Artifacts:

- focused gate:
  `cache/hrxv1/gfx1151/q8_0-mmq128x64-splitqsum-focused-20260618-152805/`;
- baseline simple BM128/BN64 reprobe:
  `cache/hrxv1/gfx1151/q8_0-mmq128x64-reprobe-20260618-152330/`;
- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq128x64_splitqsum.hsaco`.

Purpose:
test whether the split-qsum live-range fix that made BN112 and BN128 viable
also rescues BM128/BN64 row ownership. This bracket was motivated by Vulkan's
BM128 large route and by the focused gap on Q8_0 p512/p513 rows.

Compile evidence:

- old simple BM128/BN64: wave32, VGPR `192`, `47` VGPR spills, private
  segment `192`;
- new BM128/BN64 split-qsum: wave32, SGPR `22`, VGPR `144`, LDS `2176`,
  no spills, private segment `0`.

Correctness and route evidence:

- p33, p512, and p513 CPU-reference gates passed;
- route traces selected BM128/BN64 split-qsum under the opt-in env for all
  focused rows.

Focused timing versus current default routing:

| Size | Row | Default | BM128/BN64 split | Ratio |
| --- | ---: | ---: | ---: | ---: |
| p512 | `Vcur-0` | `555.202 us` | `524.853 us` | `0.945x` |
| p512 | `Qcur-0` | `1903.903 us` | `1812.905 us` | `0.952x` |
| p512 | `ffn_out-0` | `7162.991 us` | `7628.520 us` | `1.065x` |
| p512 | `ffn_gate-0` | `6394.198 us` | `15635.817 us` | `2.445x` |
| p512 | `result_output` | `56614.310 us` | `107062.429 us` | `1.891x` |
| p513 | `Vcur-0` | `535.054 us` | `540.624 us` | `1.010x` |
| p513 | `Qcur-0` | `2005.796 us` | `1979.625 us` | `0.987x` |
| p513 | `ffn_out-0` | `7555.994 us` | `8021.778 us` | `1.062x` |
| p513 | `ffn_gate-0` | `6989.245 us` | `11948.665 us` | `1.710x` |
| p513 | `result_output` | `63810.167 us` | `117461.738 us` | `1.841x` |

Decision:
reject for production promotion. The route fixes the compile-resource cliff,
but BM128/BN64 row ownership is not the missing packed-path parity axis:
small attention rows are mixed/slightly positive, while FFN/output rows regress
badly enough to make p512 total `1.83x` and p513 total `1.73x` slower than
current default.

## WMMA F16 Wave64 Output Ownership Constraint

Artifact:
`cache/hrxv1/gfx1151/wmma-f16-lane-map-coords-20260618-154317/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/wmma_f16_lane_map_bench.hip.cpp`

Purpose:
turn the failed Q8_0 direct-WMMA `fullpair` and stream writeback probes into a
concrete ownership constraint before adding another route. The local AMD
Matrix Instruction Calculator was run for:

```text
gfx1151, v_wmma_f16_16x16x16_f16, wavefront=64, D matrix
```

The CMake/Ninja-built HIP fixture then validated OPSEL behavior on the actual
gfx1151 device.

Runtime result:

- `opsel=0 changed_even=256 changed_odd=0`;
- `opsel=1` / OPSEL field `4` changed `changed_even=0 changed_odd=256`;
- `check: elements=1024 bad=0 coord_bad=0`.

Ownership rule:

- wave64 D has four logical VGPRs for the low-half matrix view;
- `slot >> 1` is the D VGPR index;
- `row = 4 * (slot >> 1) + floor(lane / 16)`;
- `col = lane % 16`;
- OPSEL `0` updates the low half / even slots for those D coordinates;
- OPSEL `4` updates the high half / odd slots for the same D coordinates.

Decision:
do not treat odd accumulator slots or OPSEL `4` as a second row band. That was
the wrong model for the rejected fullpair/stream writeback probes and explains
why variants that matched the RADV static store surface could still fail strict
CPU-reference correctness. The next direct-WMMA Q8 route must either use only
one selected half for a 16x16 output tile or deliberately pack a second
independent tile into the high half with an explicit, proven output mapping.
Any candidate claiming RADV cooperative-matrix store parity must be checked
against this lane/slot-to-D-coordinate table before focused timing.

## Rejected Selected Fast-Half Compile Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_fast_half_selected_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-packstage-fast-half-selected-bufferstore-compile-20260618-155052/`

Purpose:
combine the validated gfx1151 wave64 D-coordinate map with the closest
low-barrier direct-WMMA output-stage primitive. The candidate writes only the
OPSEL-selected half to the real output tile, while also staging and reloading
the unselected half as dummy data to preserve the halfword LDS opcode surface.

Compile evidence:

- wave64, SGPR `28`, VGPR `196`, LDS `22528`, no spills;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `128 ds_load_u16_d16`;
- `128 ds_store_b16`;
- `2 ds_store_b32`;
- `128 buffer_store_b32`;
- `2 s_barrier`.

RADV comparison against the Q8_0 p512 large oracle:

| Axis | RADV | Selected fast-half probe |
| --- | ---: | ---: |
| VGPR | `192` | `196` |
| LDS bytes | `22528` | `22528` |
| Spills | `0` | `0` |
| WMMA | `32` | `32` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `128` |
| `buffer_store_b32` | `192` | `128` |
| Barriers | `2` | `2` |
| Final pre-WMMA `lgkmcnt` | `51` | `0` |
| Loads before final pre-WMMA wait | `59` | `1` |

Decision:
reject before focused runtime. The ownership map is now correct, but this HIP
C++ spelling does not reproduce RADV's cooperative-matrix schedule: it still
serializes LDS loads before the first WMMA, misses the four-register VGPR
target, and emits only the selected tile stores rather than RADV's wider
store surface. The next direct-WMMA attempt should preserve the selected
D-coordinate rule while targeting the outstanding LDS issue window and store
topology directly, not this dummy-half selected-output form.

## Rejected Nowait Named-Fragment Staged-Wait Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_stagedwait_packstage_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-stagedwait-packstage-score-20260618-continued/`

Purpose:
test the narrowest remaining first-WMMA issue-window hypothesis in HIP C++:
keep the no-wait named A/B fragment loads, but replace the single
`s_waitcnt lgkmcnt(0)` before the WMMA block with staged waits
`lgkmcnt(12)`, `lgkmcnt(8)`, `lgkmcnt(4)`, and `lgkmcnt(0)` before the four
B-column WMMA groups. If source-visible partial waits were enough, the score
should move toward RADV's high outstanding-LDS window.

Compile evidence:

- wave64, SGPR `28`, VGPR `196`, LDS `20480`, no spills;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `0 ds_load_u16_d16`;
- `0 ds_store_b16`;
- `2 ds_store_b32`;
- `128 buffer_store_b32`;
- `2 s_barrier`;
- `14 s_waitcnt`;
- `44 s_waitcnt_depctr`.

First-WMMA score:

| Metric | RADV large route | Staged-wait probe |
| --- | ---: | ---: |
| first-window pre-WMMA `ds_load_b64` | `59` | `24` |
| loads immediately before final pre-WMMA wait | `59` | `0` |
| final pre-WMMA `lgkmcnt` | `51` | `0` |
| first-window WMMA count | `22` | `16` |
| waits after first WMMA | `[47, 43, 39, 40, 36, 32, 24, 20, 16, 12, 8]` | `[12, 8, 4, 0]` |

Decision:
reject before runtime. The emitted HSACO did not preserve the partial waits as
a pre-WMMA high-`lgkmcnt` gate; they appear after the first WMMA group. This
rules out a cheap source-visible wait-schedule fix for the large Q8_0
direct-WMMA path. The remaining direct route needs either a lower-level load
issue spelling/cooperative-matrix primitive or a return to the packed-Q8_1 path
with an oracle-derived schedule axis.

## Rejected Nowait Named-Fragment Preuse Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_preuse_packstage_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-preuse-packstage-score-20260618-continued/`

Purpose:
test whether an artificial source-level data dependency can force all eight
named A/B fragment vectors to materialize before the first WMMA. This is the
direct follow-up to the staged-wait probe: if LLVM moved WMMA early because
later fragments had no immediate uses, an empty inline asm consuming all
fragments should widen the pre-WMMA LDS window.

Compile evidence:

- wave64, SGPR `28`, VGPR `212`, LDS `20480`, no spills;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `0 ds_load_u16_d16`;
- `0 ds_store_b16`;
- `2 ds_store_b32`;
- `128 buffer_store_b32`;
- `2 s_barrier`;
- `8 s_waitcnt`;
- `44 s_waitcnt_depctr`.

First-WMMA score:

| Metric | RADV large route | Prior nowait named-frag | Preuse probe |
| --- | ---: | ---: | ---: |
| first-window pre-WMMA `ds_load_b64` | `59` | `24` | `32` |
| loads immediately before final pre-WMMA wait | `59` | `0` | `32` |
| final pre-WMMA `lgkmcnt` | `51` | `0` | `0` |
| first-window WMMA count | `22` | `16` | `16` |
| VGPR | `192` | `196` | `212` |

Decision:
reject before runtime. This proves the source dependency can move the load
window in the intended direction, but it is not enough: the wait still drains
to `lgkmcnt(0)`, the load window reaches only `32` of RADV's `59`, and VGPR
pressure jumps to `212`. A future direct-WMMA attempt should not use this
preuse trick as a production path; the evidence points toward a lower-level
load issue form or a different packed-Q8_1 schedule axis.

## Rejected Nowait Named-Fragment Preuse Staged-Wait Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_preuse_stagedwait_packstage_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-nowait-namedfrag-preuse-stagedwait-packstage-score-20260618-continued/`

Purpose:
combine the only useful source-level movement from the preuse probe with a
nonzero wait ladder. The wrapper consumes all eight A/B fragments before the
first wait, then requests `lgkmcnt(24)`, `lgkmcnt(16)`, `lgkmcnt(8)`, and
`lgkmcnt(0)` before the four B-column WMMA groups. This tests whether HIP C++
can preserve a partially outstanding pre-WMMA LDS window once the 32-load
source dependency exists.

Compile evidence:

- wave64, SGPR `28`, VGPR `212`, LDS `20480`, no spills;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `0 ds_load_u16_d16`;
- `0 ds_store_b16`;
- `2 ds_store_b32`;
- `128 buffer_store_b32`;
- `2 s_barrier`;
- `14 s_waitcnt`;
- `44 s_waitcnt_depctr`.

First-WMMA score:

| Metric | RADV large route | Preuse probe | Preuse staged-wait |
| --- | ---: | ---: | ---: |
| first-window pre-WMMA `ds_load_b64` | `59` | `32` | `32` |
| loads immediately before final pre-WMMA wait | `59` | `32` | `0` |
| final pre-WMMA `lgkmcnt` | `51` | `0` | `0` |
| first-window WMMA count | `22` | `16` | `16` |
| waits after first WMMA | `[47, 43, 39, 40, 36, 32, 24, 20, 16, 12, 8]` | `[]` | `[24, 16]` |
| VGPR | `192` | `212` | `212` |

Decision:
reject before runtime. The requested nonzero waits appear in the HSACO, but
they do not create the RADV pre-WMMA contract; the comparator still sees a
final pre-WMMA drain to zero and no loads immediately before the final wait.
This closes the cheap source-visible preuse/wait ladder axis. Future direct
work needs a lower-level load issue primitive or a different dataflow, not
more local wait-count decoration.

## Packed-Q8_1 HSACO Family Triage

Artifact:
`cache/hrxv1/gfx1151/q8_0-packed-hsaco-family-summary-20260618-161943/`.

Tool:
`sources/llama.cpp/tools/vulkan-oracle/summarize_hsaco_family.py`.

Purpose:
compare the CMake/Ninja-built packed-Q8_1 HSACO family in one static table
after the accepted BN112 and BN128 split-qsum routes, instead of continuing
local tile probes from memory.

Key static facts:

| Route HSACO | Wave | VGPR | LDS | VGPR spills | Dot sites | First-dot score |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mul_mat_vec_q8_0_mmq64x112_splitqsum.hsaco` | 32 | 134 | 3808 | 0 | 480 | `28 loads, lgkmcnt(14), 60 hot ops` |
| `mul_mat_vec_q8_0_mmq64x128_splitqsum.hsaco` | 32 | 152 | 4352 | 0 | 512 | `28 loads, lgkmcnt(14), 60 hot ops` |
| `mul_mat_vec_q8_0_mmq64x128_splitqsum8.hsaco` | 32 | 120 | 4352 | 0 | 512 | `28 loads, lgkmcnt(14), 60 hot ops` |
| `mul_mat_vec_q8_0_mmq64x128_splitqsum_bstripe.hsaco` | 32 | 152 | 4352 | 0 | 512 | `28 loads, lgkmcnt(14), 60 hot ops` |
| `mul_mat_vec_q8_0_mmq128x64_splitqsum.hsaco` | 32 | 144 | 2176 | 0 | 512 | `28 loads, lgkmcnt(14), 60 hot ops` |
| `mul_mat_vec_q8_0_q8_1_x4_mmql128.hsaco` | 64 | 192 | 9216 | 472 | 768 | `28 loads, lgkmcnt(16), 60 hot ops` |

Interpretation:

- split-qsum is a real live-range fix: it turns otherwise spilling widened
  column routes into no-spill candidates;
- split-qsum is not an issue-window fix: all parsed packed-Q8_1 variants share
  the same first-dot schedule score, including rejected and accepted routes;
- explicit B-stripe preloading also does not perturb the first-dot issue
  window or resource contract;
- lower VGPR alone is not enough: the `splitqsum8` route reaches VGPR `120`
  but was rejected by focused timing;
- BM128 row ownership is not rescued by split-qsum: the no-spill
  BM128/BN64 split-qsum route still regressed focused p512 and p513 timing;
- the naive wave64 MMQL128 packed route is not a viable escape hatch on
  gfx1151 because it spills heavily.

Decision:
do not continue Q8_0 with another simple BN/BM or split-qsum-only route. The
current production policy remains BN128 split-qsum for full 128-column rows,
BN112 split-qsum for production-width tails, and BN64/medium policy for narrow
rows. A new Q8_0 candidate needs a materially different schedule axis, such as
lower-level RADV-like cooperative load/store control or a packed dataflow that
changes the actual first-dot issue/window behavior, not just resource shape.

## Rejected CMake WMMA Issue-Window Fixture

Candidate:
`hrx-hip-bench-wmma-issue-window`, diagnostic kernels
`wmma_issue_window_probe<51>` and `wmma_issue_window_probe<0>`.

Artifact:
`cache/hrxv1/gfx1151/q8_0-wmma-issue-window-bench-20260618-170000/`

Purpose:
test whether a CMake/Ninja-built HIP fixture using inline-asm
`ds_read_b64` can reproduce the RADV Q8_0 pre-WMMA LDS issue window before a
production route is changed. The `lgkm51` mode issues 64 explicit LDS loads,
requests `s_waitcnt lgkmcnt(51)`, then runs a finite constant-fragment WMMA
sequence so runtime data validity is separated from schedule shape.

Build and runtime evidence:

- CMake target:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-issue-window -j "$(nproc)"`;
- runtime smoke:
  `--mode=lgkm51` and `--mode=wait0` both completed with
  `checksum=65536.000000` and `nan=0`;
- extracted HSACO:
  `device.hsaco`, unbundled from the executable `.hip_fatbin` using
  `clang-offload-bundler`;
- comparison:
  `compare-lgkm51.json`, `compare-lgkm51.md`,
  `compare-wait0.json`, and `compare-wait0.md`.

Static schedule comparison:

| Metric | RADV Q8_0 large | HIP fixture `lgkm51` |
| --- | ---: | ---: |
| wavefront | `64` | `64` |
| VGPR | `192` | `145` |
| LDS bytes | `22528` | `32768` |
| spills | `0` | `0` |
| total `ds_load_b64` | `64` | `64` |
| total WMMA | `32` | `8` |
| pre-WMMA `ds_load_b64` | `59` | `25` |
| loads immediately before final pre-WMMA wait | `59` | `0` |
| final pre-WMMA `lgkmcnt` | `51` | `0` |
| first-window WMMA count | `22` | `7` |
| load-like ops after first WMMA | `7` | `39` |

Decision:
reject this spelling as a production route source. Although the source contains
64 inline-asm `ds_read_b64` operations and an explicit `lgkmcnt(51)` mode, the
emitted HIP HSACO does not preserve the RADV contract. The compiler schedules
the first WMMA after only 25 LDS loads, drains the pre-WMMA window to
`lgkmcnt(0)`, and places the explicit `lgkmcnt(51)` after the WMMA sequence in
the hot window. This is stronger evidence that source-visible HIP C++ and
basic volatile inline loads are still insufficient for the RADV schedule unless
the WMMA consumes a dependency that pins the load/wait window.

Next useful axis:
keep the CMake fixture as a diagnostic target and tighten it rather than
promoting a route. The next probe should make the finite WMMA operands depend
on all issued LDS loads through a low-level inline-asm output block, so the
compiler cannot hoist independent WMMA work before the intended wait. If that
still drains to zero or interleaves loads after the first WMMA, the remaining
path is a lower-level codegen/assembly strategy or a non-WMMA packed-Q8_1
dataflow with a different first-dot issue window.

## Accepted Diagnostic: Dependency-Pinned WMMA Issue Window

Candidate:
updated `hrx-hip-bench-wmma-issue-window`, diagnostic kernels
`wmma_issue_window_probe<51>` and `wmma_issue_window_probe<0>`.

Artifact:
`cache/hrxv1/gfx1151/q8_0-wmma-issue-window-depconst-20260618-171500/`

Purpose:
test the direct follow-up to the rejected independent-constant fixture. This
version produces the finite WMMA operand from an inline-asm block whose inputs
are all 16 LDS-loaded A/B fragments and whose first instruction is the chosen
`s_waitcnt`. This pins the WMMA operand to the LDS load issue window without
requiring the diagnostic to feed arbitrary loaded half payloads into WMMA.

Build and runtime evidence:

- CMake target:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target
  hrx-hip-bench-wmma-issue-window -j "$(nproc)"`;
- runtime smoke:
  `--mode=lgkm51` and `--mode=wait0` both completed with
  `checksum=32768.000000` and `nan=0`;
- extracted HSACO:
  `device.hsaco`, unbundled from the executable `.hip_fatbin`;
- comparison:
  `compare-lgkm51.json`, `compare-lgkm51.md`,
  `compare-wait0.json`, and `compare-wait0.md`.

Static schedule comparison:

| Metric | RADV Q8_0 large | HIP dep-pinned `lgkm51` |
| --- | ---: | ---: |
| wavefront | `64` | `64` |
| VGPR | `192` | `133` |
| LDS bytes | `22528` | `32768` |
| spills | `0` | `0` |
| total `ds_load_b64` | `64` | `64` |
| total WMMA | `32` | `8` |
| pre-WMMA `ds_load_b64` | `59` | `64` |
| loads immediately before final pre-WMMA wait | `59` | `64` |
| final pre-WMMA `lgkmcnt` | `51` | `51` |
| first-window WMMA count | `22` | `8` |
| load-like ops after first WMMA | `7` | `0` |

Decision:
accept as a diagnostic schedule primitive, not as a production route. This
proves HIP C++ plus inline asm can be made to preserve the core RADV-style
pre-WMMA LDS window when the WMMA operand is dependency-pinned to the issued
loads. It does not yet prove Q8_0 parity because the fixture uses finite
constant fragments, only eight WMMAs, no real Q8_0 packing, no production
writeback shape, and a larger artificial LDS allocation.

Next route-candidate axis:
port the dependency-pinned operand-production pattern into a Q8_0 direct-WMMA
candidate. The candidate should keep the real Q8_0 pack/load dataflow, but
replace the current independent/preuse wait tricks with an inline-asm
fragment-production block that both references the loaded fragment registers
and emits the requested `lgkmcnt(51)` before the first WMMA group. Promotion
still requires focused CPU-reference, route trace, p512 and p513 timing, and
ISA confirmation that the production HSACO keeps the same pre-WMMA score.

## Rejected Real-Fragment Dependent-Wait Q8_0 Candidate

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_packstage_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-depwait-realfrag-compile-20260618-174000/`

Purpose:
port the accepted CMake issue-window diagnostic into the real Q8_0
direct-WMMA source. The route keeps the real A/B fragments and copies them
through dependency-pinned inline asm, emits `lgkmcnt(51)` before the first
WMMA, and uses the RADV-like wait ladder before subsequent WMMAs.

Build evidence:

- CMake/Ninja target:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops
  -j "$(nproc)"`;
- generated HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_packstage_bufferstore_wg256.hsaco`;
- route metadata:
  generated catalog contains the route with `selector_gate:
  none; compile-contract probe only`;
- comparison:
  `compare.json` and `compare.md` against the Llama 3.1 8B Q8_0 p512 RADV
  large oracle.

Static schedule comparison:

| Metric | RADV Q8_0 large | HIP real-frag depwait |
| --- | ---: | ---: |
| wavefront | `64` | `64` |
| VGPR | `192` | `240` |
| LDS bytes | `22528` | `20480` |
| spills | `0` | `0` |
| total `ds_load_b64` | `64` | `64` |
| total WMMA | `32` | `32` |
| pre-WMMA `ds_load_b64` | `59` | `32` |
| loads immediately before final pre-WMMA wait | `59` | `32` |
| final pre-WMMA `lgkmcnt` | `51` | `51` |
| first-window WMMA count | `22` | `15` |
| wait ladder after first WMMA | `[47,43,39,40,36,32,24,20,16,12,8]` | `[47,43,39,40,36,32,24,20,16,12,8,4,0,0,0]` |

Decision:
reject at compile-contract gate and do not run focused correctness or timing.
The dependency-pinned real-fragment source did preserve the important
`lgkmcnt(51)` gate and wait ladder, proving that the diagnostic primitive can
be transplanted into real Q8_0 payloads. It still does not match the Vulkan
oracle: the current HIP source presents one 16-wide K tile at a time, so only
32 `ds_load_b64` operations can be scored before the first WMMA, while RADV
has 59. It also raises VGPR pressure to 240 versus RADV's 192. The next Q8_0
attempt needs a different live-range/dataflow spelling that exposes both K
tiles to the first issue window without retaining the full copied fragment set
at VGPR 240.

Runtime follow-up after selected-lane WMMA diagnostics:

- Artifact:
  `cache/hrxv1/gfx1151/q8_0-depwait-selectedlane-focused-20260618-233509/`
- New opt-in selector:
  `GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_NOWAIT_NAMEDFRAG_DEPWAIT_PACKSTAGE_BUFFERSTORE_F16ACC_WG256_PROMPT=1`
- Build:
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops llama-bench -j"$(nproc)"`

Why it was rerun:
the selected-lane fixtures showed the full 4x4 real-fragment WMMA shape has
finite production-selected even lanes, so the old full-accumulator odd-lane
NaNs were not enough to reject a production store route.

Focused result:

- p512 CPU-reference passed. Route trace selected the depwait route for
  `ffn_out`, `ffn_gate`, and `result_output`; `Vcur` and `Qcur` stayed on the
  packed-Q8_1 split-qsum default.
- p33 CPU-reference passed and stayed entirely on the BN64 packed route.
- p513 CPU-reference passed. Route trace selected the depwait route for the
  same three large rows and kept `Vcur`/`Qcur` on BN112 split-qsum.
- Same-runner focused timing rejected the route:
  p512 total `71977.24 -> 101026.76 us`;
  p513 total `81449.83 -> 116681.09 us`.
- Dominant p512 rows regressed:
  `ffn_out 7042.32 -> 10156.29 us`,
  `ffn_gate 6414.64 -> 8459.76 us`,
  `result_output 56094.07 -> 79907.17 us`.
- Dominant p513 rows regressed:
  `ffn_out 7677.91 -> 10700.95 us`,
  `ffn_gate 7221.21 -> 10301.59 us`,
  `result_output 64019.19 -> 93105.74 us`.

Updated decision:
selected-lane correctness is not the blocker for the one-K depwait route, but
the source-visible dependency-copy schedule is still much slower than the
current packed-Q8_1 default. Keep the opt-in selector only as a reproducible
diagnostic and do not promote this direct-WMMA path.

## Rejected K2 Raw-Fragment Depwait Probe

Candidate:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_raw_packstage_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8_0-depwait-k2-raw-compile-20260618-190254/`

Purpose:
test whether the K2 depwait live-range cliff comes from carrying the two
K-tile LDS fragments as `half16` vectors. This variant keeps the same RADV-like
wait ladder and two-K issue shape, but loads fragments as raw `u64x4` LDS
payloads and casts to `half16` only at the WMMA boundary.

Compile evidence:

- wave64, SGPR `29`, VGPR `256`, VGPR spills `30`, private segment `124`;
- LDS `20480`;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `128 buffer_store_b32`;
- `2 s_barrier`;
- `67 s_waitcnt`, `44 s_waitcnt_depctr`.

First-WMMA score:

| Metric | RADV large route | K2 raw |
| --- | ---: | ---: |
| first-window pre-WMMA `ds_load_b64` | `59` | `57` |
| loads immediately before final pre-WMMA wait | `59` | `31` |
| final pre-WMMA `lgkmcnt` | `51` | `51` |
| first-window WMMA count | `22` | `20` |
| VGPR | `192` | `256` |
| VGPR spills | `0` | `30` |

Decision:
reject before runtime. Raw fragment representation does not escape the same
source-level K2 pressure cliff as the half16 depwait route, and it still misses
RADV's no-spill `192` VGPR contract, `59/59/lgkmcnt(51)` issue window, and
halfword LDS/writeback topology. The remaining exact-schedule path needs lower
level control over the load/WMMA sequence or a different packed-Q8_1 dataflow,
not another representation-only K2 source variant.

## Rejected BN128 Split-Qsum Wave64 Default

Candidate:
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wave64_wg256_f32`

Route control:
`GGML_HRX_ENABLE_Q8_0_Q8_1_X4_MMQ64X128_SPLITQSUM_WAVE64_PROMPT=1`

Purpose:
isolate the wavefront-mode axis for the accepted packed-Q8_1 BN128 split-qsum
route. This keeps the same source-level dataflow as
`hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32` and compiles the
new source as wave64, testing whether the current wave32 BN128 object is one of
the remaining gfx1151 compiler cliffs.

Compile evidence:

- HSACO:
  `build/hrx-v1-catalog-gfx1151/ggml/src/ggml-hrx/generated/hsaco/gfx1151/mul_mat_vec_q8_0_mmq64x128_splitqsum_wave64.hsaco`;
- wave64, SGPR `40`, VGPR `152`, LDS `4352`, private segment `0`, spills `0`;
- built through CMake/Ninja as part of
  `cmake --build build/hrx-v1-catalog-gfx1151 --target test-backend-ops llama-bench -j"$(nproc)"`.

Focused evidence:

- Opt-in artifact:
  `cache/hrxv1/gfx1151/q8_0-mmq64x128-splitqsum-wave64-fullcols-focused-20260619-010242/`.
- Post-edit route/default artifact:
  `cache/hrxv1/gfx1151/q8_0-wave64-default-postedit-focused-20260619-011031/`.
- Final opt-in route check:
  `cache/hrxv1/gfx1151/q8_0-wave64-final-optin-routecheck-20260619-011522/`.
- p512 CPU-reference passed for `Vcur`, `Qcur`, `ffn_out`, `ffn_gate`, and
  exported `result_output` stress rows. Under opt-in/default test policy, route
  traces selected the wave64 BN128 provider for all five p512 rows.
- Exact p513 CPU-reference passed and stayed on
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x112_splitqsum_wg256_f32`.
- p33 model smoke under opt-in stayed on
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x64_wg256_f32`.
- Final post-rejection default p512 route check selected the original
  wave32 BN128 split-qsum provider, proving the candidate is opt-in only.

Focused post-edit p512 timing favored wave64:

| Row | rollback wave32 us | wave64 us |
| --- | ---: | ---: |
| Vcur | `547.266` | `528.474` |
| Qcur | `1925.167` | `1888.803` |
| ffn_out | `7167.517` | `7029.020` |
| ffn_gate | `6498.759` | `6133.599` |
| result_output stress | `57170.548` | `53821.143` |

Model A/B:

- initial opt-in pair:
  `cache/hrxv1/gfx1151/q8_0-wave64-fullcols-default-p512-r3-20260619-010350/`
  and
  `cache/hrxv1/gfx1151/q8_0-wave64-fullcols-optin-p512-r3-20260619-010355/`
  showed a small steady win, `452.464 -> 460.691 tok/s`;
- post-edit same-build default/rollback pair:
  `cache/hrxv1/gfx1151/q8_0-wave64-postedit-default-p512-r3-20260619-011242/`
  and
  `cache/hrxv1/gfx1151/q8_0-wave64-postedit-rollback-p512-r3-20260619-011254/`
  rejected default promotion, `446.647` steady tok/s for wave64 versus
  `447.814` steady tok/s for rollback wave32, with zero fallback lines on both
  runs.

Odd/tail smoke:

- default artifact:
  `cache/hrxv1/gfx1151/q8_0-wave64-fullcols-default-p33p513-r3-20260619-010642/`;
- opt-in artifact:
  `cache/hrxv1/gfx1151/q8_0-wave64-fullcols-optin-p33p513-r3-20260619-010656/`;
- p33 stayed on BN64 in both runs;
- exact p513 stayed on BN112 in both runs;
- fallback lines were zero.

Decision:
keep the route as an opt-in diagnostic and reject default promotion. The
focused kernel screen says wave64 can improve this packed-Q8_1 BN128 dataflow,
but the model-level acceptance gate did not confirm it. The remaining Q8_0
p512/p513 parity work should not spend more time on this isolated wave-mode
pivot; it should target the documented RADV/HIP structural delta, especially
cooperative-matrix store/lane ownership and common-runner focused Vulkan gaps.

## Current-Head Vulkan Oracle Refresh

Artifact:
`cache/hrxv1/gfx1151/vulkan-oracle-current-llama31-8b-q8_0-p512-fa1-20260619-012416/`

Purpose:
refresh the Llama 3.1 8B Q8_0 p512 Vulkan oracle after the direct basket
showed much higher current Vulkan throughput than the older 2026-06-18 oracle
rate. This capture is the current reference for Q8_0 p512 schedule facts.

Command shape:
`build/vulkan-gfx1151/bin/llama-bench -p 512 -n 0 -b 1024 -ub 1024 -fa 1 -r 1 -dev Vulkan0`
through `sources/llama.cpp/tools/vulkan-oracle/run_vulkan_oracle_capture.py`.

Evidence:

- JSON reports build commit `5200f0b01`, `backends=Vulkan`, and
  `419.430 tok/s` for the single capture run.
- The capture produced `13` pipeline identities, `516` dispatch signatures,
  `13` SPIR-V files, SPIR-V asm, split RADV ISA/stats, and inventory files.
- Dominant dense prompt pipeline is still
  `matmul_q8_0_f32_f16acc_aligned_l`, hash `0x72d309e22f889977`.
- Dispatch count is `221`, with normalized families `[112,4,1]` FFN gate/up,
  `[32,4,1]` Q/attention/out, and `[8,4,1]` K/V.
- The large-route contract is unchanged: spec
  `[256,128,128,32,64,64,2,16,16,16,64]`,
  `wg_denoms=[128,128,1]`, `LDS=22528`, `VGPR=192`, no spills, `32`
  `v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`, `128 ds_load_u16_d16`,
  `128 ds_store_b16`, `192 buffer_store_b32`, and two barriers.

Current RADV/HIP comparison artifact:
`cache/hrxv1/gfx1151/q8_0-current-radv-vs-hrx-isa-20260619-012526/`

Compared routes:

- accepted packed route:
  `hrx_mul_mat_vec_q8_0_q8_1_x4_mmq64x128_splitqsum_wg256_f32`;
- direct-WMMA reasonable-pressure route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_packstage_bufferstore_f16acc_wg256_f32`;
- direct-WMMA dependency-pinned route:
  `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_packstage_bufferstore_f16acc_wg256_f32`.

Key deltas:

| Route | VGPR/spills | First hot-op score | Store/LDS topology | Decision |
| --- | --- | --- | --- | --- |
| accepted BN128 split-qsum | VGPR `152`, no spills, wave32 | integer-dot path, no WMMA; first hot op is `v_dot4_i32_iu8` with `18` pre-hot LDS loads and final `lgkmcnt(16)` | `LDS=4352`, no `buffer_store_b32`, no `ds_load_u16_d16`; structurally not Vulkan | keep default as current best packed route, but not an exact schedule clone |
| K2 directwait | VGPR `196`, no spills, wave64 | `24/0/lgkmcnt(0)` before first WMMA, `24` WMMAs in window | `128 buffer_store_b32`, no halfword LDS load/store topology | reject: reasonable pressure but compiler collapses the RADV wait window |
| K2 depwait | VGPR `256`, `30` spills, wave64 | `64` pre-hot LDS loads, final `lgkmcnt(51)`, `31` WMMAs in window | `128 buffer_store_b32`, no halfword LDS load/store topology | reject: close issue-window shape but hard register/spill cliff |

Decision:
this refresh confirms the old schedule conclusion under current-head Vulkan:
Q8_0 p512/p513 parity is not blocked by a missing simple selector or wave-mode
knob. HIP C++ can currently express either the high-outstanding LDS/WMMA wait
window or acceptable register pressure, but not both in the same source-level
route, and the direct-WMMA diagnostics still miss RADV's cooperative
halfword-LDS/writeback topology. The next Q8_0 source change should require a
new low-level primitive or a materially different dataflow, not another
adjacent BN/BM/split-qsum/wait-decoration probe.

## Rejected Packed Split-Qsum Qpack Hoist

Artifact:
`cache/hrxv1/gfx1151/q8_0-qpack-hoist-focused-20260619-073318/`

Candidate:
temporary source edit to the accepted BN128/BN112 split-qsum packed-Q8_1
kernels, computing the eight `hrx_q8_0_pack4` values once per K block and
reusing them across both column chunks.

Evidence:

- focused CPU-reference passed p512 and exact p513 exported rows;
- route traces selected BN128 for all five p512 rows and BN112 for all five
  p513 rows;
- p512 focused total regressed `72712.698 -> 75663.326 us`;
- p513 focused total regressed `81093.295 -> 83117.543 us`;
- static HSACO resources did not reveal a spill cliff: BN128 stayed VGPR `152`,
  no spills; BN112 only moved VGPR `134 -> 135`, no spills.

Decision:
revert and reject. The current HIP compiler schedule prefers recomputing the
Q8_0 A-pack inside each split-qsum chunk over keeping all eight packs live
across both chunks. Treat this as closed for the present packed route; the
remaining Q8_0 parity gap needs a different schedule family or lower-level
codegen control, not this local source cleanup.

## Current Focused HRX/Vulkan Backend-Op Gap

Artifact:
`cache/hrxv1/gfx1151/q8_0-current-focused-hrx-vulkan-compare-ab41b8701-20260619-075951/`

Purpose:
convert the current Q8_0 p512/p513 model-level parity gap into a same-runner
focused backend-op table. This uses existing current HRX and Vulkan
`test-backend-ops perf` CSVs over the exported Llama 3.1 8B Q8_0 prompt rows,
then summarizes them with `tools/hrxv1_compare_backend_op_perf.py`.

Evidence:

| Size | HRX total us | Vulkan total us | HRX/Vulkan |
| --- | ---: | ---: | ---: |
| p512 | `72712.698` | `48100.936` | `1.512x` |
| p513 | `81093.295` | `55428.701` | `1.463x` |

Per-row ratios:

| Row | p512 HRX/Vulkan | p513 HRX/Vulkan |
| --- | ---: | ---: |
| `Vcur-0` | `2.039x` | `1.990x` |
| `Qcur-0` | `1.834x` | `1.934x` |
| `ffn_out-0` | `1.772x` | `1.838x` |
| `ffn_gate-0` | `1.595x` | `1.505x` |
| `result_output` | `1.464x` | `1.409x` |

Decision:
keep Q8_0 as an active kernel/schedule boulder. The focused gap is smaller
than the direct basket tok/s gap but still large enough that local cleanup is
not the right tool. Vcur/Qcur are nearly 2x behind Vulkan and the large
`result_output` row is still roughly 1.4-1.5x behind. The next production route
candidate should use these rows as the focused acceptance screen and target the
known RADV structural delta: cooperative-matrix load/store lowering, selected
live accumulator ownership, and raw buffer-store writeback without the
halfword-stage lane-map failure.

## Rejected LDS K2 Live-WMMA Mixed-Store Contract

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-live-wmma-mixed-probe-20260619-080339/`

Mode:
`hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-radv-mixed192`

Purpose:
test whether a standalone CMake-built HIP C++ fixture can hit the current RADV
large Q8_0 opcode-count contract while keeping two LDS-loaded WMMA phases live.
The fixture accumulates two identical LDS fragment phases into live accumulator
groups `0..15`, raw buffer-stores those groups, and uses the existing synthetic
halfword LDS stage/load/store path for groups `16..47`.

Controls:

- `wmma-radv-mixed192` passed twice with `bad=0`;
- `wmma-lds-radv-mixed192` passed twice with `bad=0`;
- `wmma-lds-k2-radv-mixed192` failed twice with the same signature:
  `bad=2112 max_abs=50625 first_bad=4800 actual=2048 expected=12803`.

Static comparison against the current Q8_0 p512 RADV oracle:
`cache/hrxv1/gfx1151/coopstore-lds-k2-live-wmma-mixed-probe-20260619-080339/compare-k2.md`

| Metric | RADV Q8_0 large | HIP K2 mixed-store fixture |
| --- | ---: | ---: |
| wave | Vulkan subgroup/W64 contract | `64` |
| VGPR | `192` | `162` |
| SGPR | `108` | `14` |
| LDS bytes | `22528` | `32768` |
| spills | `0` | `0` |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `128` |
| `ds_store_b16` | `128` | `128` |
| `buffer_store_b32` | `192` | `192` |
| barriers | `2` | `3` |
| `s_waitcnt` | `169` | `135` |

Decision:
reject this contract for production route work. It reaches the target opcode
counts but fails correctness, so the missing Q8_0 Vulkan schedule is not just
the count tuple `32 WMMA / 64 b64 LDS loads / 128 halfword LDS loads+stores /
192 buffer stores`. The unresolved axis is still cooperative-matrix lane
ownership/topology. Do not port this K2 mixed-store fixture into a real Q8_0
route unless the lane mapping is replaced by a proven lower-level primitive or
a source form that passes this contract first.

## Positive K2 Direct-Only Split

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-direct64-probe-20260619-081232/`

Mode:
`hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-direct64`

Purpose:
split the failing mixed-store contract to test only the two-phase live WMMA
accumulator plus direct raw buffer-store path. This removes the synthetic
halfword LDS stage/load/store groups.

Evidence:

- `wmma-lds-k2-direct64` passed with `elements=12288 bad=0 max_abs=0`;
- the same binary still failed `wmma-lds-k2-radv-mixed192` at
  `group=18 slot=3 lane=0`, confirming the split did not mask the known
  failure;
- static facts from `compare-direct64.md`: wave64, SGPR `14`, VGPR `153`,
  LDS `16384`, no spills, `32` WMMA, `64 ds_load_b64`,
  `64 buffer_store_b32`, two barriers, and no halfword LDS load/store path.

Interpretation:
the two-K-phase LDS-loaded live-accumulator direct-store primitive is correct in
isolation. The correctness break appears when that primitive is combined with
the adjacent halfword stage/load/store topology needed to match RADV's `192`
store surface. The next fixture should not retest K2 direct stores alone; it
should isolate why the synthetic halfword groups corrupt starting at group 18,
or replace the halfword-stage spelling with a lower-level cooperative-matrix
store primitive that preserves the passing direct path.

## Rejected K2 Stage-First Mixed-Store Ordering

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-stagefirst-mixed-probe-20260619-081501/`

Mode:
`hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-stagefirst-mixed192`

Purpose:
test whether the K2 mixed-store failure is just source ordering between the
synthetic halfword stage stores and the two LDS-loaded WMMA/direct-store phase.
This variant writes the synthetic stage groups first, then runs K2 WMMA and
direct raw stores, then loads/stores the synthetic stage groups after a barrier.

Evidence:

- `wmma-lds-k2-direct64` still passed in the same binary;
- `wmma-lds-k2-stagefirst-mixed192` failed with the same signature as the
  original mixed mode:
  `bad=2112 max_abs=50625 first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`;
- static facts from `compare-stagefirst.md`: wave64, SGPR `14`, VGPR `162`,
  LDS `32768`, no spills, `32` WMMA, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`,
  three barriers, and `135 s_waitcnt`.

Decision:
reject the simple source-ordering axis. Issuing the synthetic halfword stage
before the K2 WMMA/direct-store work does not fix the lane/topology break. The
remaining useful probes should either reduce the mixed halfword surface to find
the exact corruption threshold, or move below this HIP C++ spelling.

## K2 Mixed Halfword-Stage Threshold

Artifacts:

- first threshold sweep:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-mixed-threshold-probe-20260619-081846/`
- second threshold sweep and ISA:
  `cache/hrxv1/gfx1151/coopstore-lds-k2-mixed-threshold2-probe-20260619-082030/`

Modes:

- `wmma-lds-k2-mixed96`: direct groups `0..15`, staged groups `16..23`;
- `wmma-lds-k2-mixed128`: direct groups `0..15`, staged groups `16..31`;
- `wmma-lds-k2-mixed160-lo`: direct groups `0..15`, staged groups `16..39`;
- `wmma-lds-k2-mixed160-hi`: direct groups `0..15`, staged groups `16..31`
  and `40..47`;
- `wmma-lds-k2-radv-mixed192`: direct groups `0..15`, staged groups `16..47`.

Correctness:

| Mode | Result |
| --- | --- |
| `direct64` | pass |
| `mixed96` | pass |
| `mixed128` | pass |
| `mixed160-lo` | fail at `group=18 slot=3 lane=0` |
| `mixed160-hi` | fail at `group=18 slot=3 lane=0` |
| `mixed192` | fail at `group=18 slot=3 lane=0` |

Static boundary:

| Mode | Status | `buffer_store_b32` | `ds_store_b16` | `ds_load_u16_d16` |
| --- | --- | ---: | ---: | ---: |
| `mixed128` | pass | `128` | `64` | `64` |
| `mixed160-lo` | fail | `160` | `96` | `96` |
| RADV large | pass in Vulkan | `192` | `128` | `128` |

Interpretation:
the last correctness-clean HIP C++ mixed surface is `mixed128`. Adding any
extra staged group block beyond `16..31` breaks earlier staged output in
`group=18`, whether the extra block is `32..39` or `40..47`. This argues
against continuing production-route work with this scalarized HIP C++ halfword
stage expansion. The next useful direction is either a much smaller lane-map
reproducer for the compiler/runtime owner, or a lower-level cooperative-store
primitive that does not express the RADV store topology as synthetic halfword
stage groups.

## K2 Mixed160 Tight And Raw-Store Controls

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-mixed160-tight-raw-control-20260619-082600/`

Purpose:
separate the `mixed160` failure into three axes: raw buffer-store count, total
LDS allocation, and halfword LDS stage expansion.

Evidence:

| Mode | Result | Key static facts |
| --- | --- | --- |
| `wmma-lds-k2-direct160-raw` | pass | `160 buffer_store_b32`, no halfword LDS stage, VGPR `153`, no spills |
| `wmma-lds-k2-mixed160-lo-tight` | fail | `160 buffer_store_b32`, `96 ds_store_b16`, `96 ds_load_u16_d16`, VGPR `162`, no spills, LDS `28672` |
| `wmma-lds-k2-mixed160-hi-tight` | fail | same failure signature as `lo-tight`, LDS `28672` |
| `wmma-lds-k2-mixed128` | pass on repeat | `128 buffer_store_b32`, `64 ds_store_b16`, `64 ds_load_u16_d16` |

The tight variants reduce `group_segment_fixed_size` from the previous
non-tight `32768` bytes to `28672` bytes and still fail at
`group=18 slot=3 lane=0`. The raw direct-store variant proves that `160`
buffer stores with live K2 WMMA are not the problem by themselves.

Decision:
close the LDS-allocation-size and raw-store-count hypotheses. The failing axis
is the scalarized halfword LDS stage expansion from `64` to `96` store/load
pairs in the K2 live-WMMA mixed context. A production route should not attempt
to reach RADV's `128` halfword stage pairs through this HIP C++ path.

## K2 Mixed128 Padded32 Control

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-mixed128-padded32-probe-20260619-083017/`

Purpose:
separate total LDS footprint from halfword-stage count. This mode performs the
same operations as the passing `mixed128` surface but allocates a 32-group
stage buffer, matching the full mixed192 footprint.

Evidence:

- `wmma-lds-k2-mixed128`: pass;
- `wmma-lds-k2-mixed128-padded32`: pass;
- `wmma-lds-k2-mixed160-lo-tight`: fail;
- `wmma-lds-k2-direct160-raw`: pass.

Static facts for `mixed128-padded32`:

- `group_segment_fixed_size=32768`;
- SGPR `14`, VGPR `162`, no spills;
- `32` WMMA, `64 ds_load_b64`;
- `64 ds_store_b16`, `64 ds_load_u16_d16`;
- `128 buffer_store_b32`.

Decision:
total LDS footprint is not the trigger. The first failing boundary remains the
halfword-stage expansion from `64` to `96` stage store/load pairs in a K2
mixed context. This is the current strongest evidence that the route needs a
lower-level cooperative-store primitive or a small compiler reproducer rather
than more scalarized HIP C++ stage groups.

## K2 Direct192 Raw-Store Control

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-direct192-raw-probe-20260619-083318/`

Mode:
`hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-direct192-raw`

Purpose:
finish separating the K2 mixed192 failure into raw output-store count versus
the scalarized halfword LDS stage. This mode keeps the two LDS-loaded WMMA
phases and emits the full `192 buffer_store_b32` surface, but removes all
synthetic halfword LDS stage/load groups.

Repeated correctness:

| Mode | Result |
| --- | --- |
| `wmma-lds-k2-direct192-raw` | pass in three repeated runs, `bad=0` |
| `wmma-lds-k2-mixed128-padded32` | pass in three repeated runs, `bad=0` |
| `wmma-lds-k2-radv-mixed192` | fail in three repeated runs at `group=18 slot=3 lane=0` |

Static comparison against the current Q8_0 p512 RADV oracle:

| Metric | RADV Q8_0 large | HIP K2 direct192 raw |
| --- | ---: | ---: |
| wave | Vulkan subgroup/W64 contract | `64` |
| VGPR | `192` | `153` |
| SGPR | `108` | `14` |
| LDS bytes | `22528` | `16384` |
| spills | `0` | `0` |
| `v_wmma_f16_16x16x16_f16` | `32` | `32` |
| `ds_load_b64` | `64` | `64` |
| `ds_load_u16_d16` | `128` | `0` |
| `ds_store_b16` | `128` | `0` |
| `buffer_store_b32` | `192` | `192` |
| barriers | `2` | `2` |
| `s_waitcnt` | `169` | `7` |

Decision:
close the raw-store-count hypothesis. Full `192 buffer_store_b32` writeback
with two live LDS-loaded WMMA phases is correctness-clean when the halfword LDS
stage is absent. Combined with the `direct160-raw`, `mixed160-tight`, and
`mixed128-padded32` controls, the break is now localized to scalarized
halfword stage expansion beyond the passing `64` store/load-pair surface. The
next production-directed move should not be another HIP C++ scalarized
halfword-stage variant; it should be a lower-level cooperative-store primitive
or a compact compiler/runtime reproducer for the K2 plus `96+` halfword-stage
corruption.

## K2 Stage96 Accsink Control

Artifact:
`cache/hrxv1/gfx1151/coopstore-lds-k2-stage96-accsink-probe-20260619-084215/`

Mode:
`hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-stage96-accsink`

Purpose:
test whether K2 live WMMA plus `96` scalarized halfword LDS store/load pairs is
enough to reproduce the mixed160 corruption without the full direct
accumulator raw-store surface. The mode computes two LDS-loaded WMMA phases,
consumes one accumulator group through an unchecked LDS sink so the WMMA path
cannot be removed, and checks only the synthetic halfword-stage output for
groups `16..39`.

Repeated correctness:

| Mode | Result |
| --- | --- |
| `wmma-lds-k2-stage96-accsink` | first sweeps passed, then 1 failure in 20-rep validation |
| `wmma-lds-k2-mixed160-lo-tight` | fail in the same binary at `group=18 slot=3 lane=0` |

The accsink failure was later and weaker than the tight mixed160 failure:
`bad=64 first_bad=8640 group=33 slot=3 lane=0 actual=1024 expected=16643`.
The tight mixed160 control failed consistently at
`bad=1600 first_bad=4800 group=18 slot=3 lane=0 actual=2048 expected=12803`.

Static facts:

| Metric | `stage96-accsink` | `mixed160-lo-tight` |
| --- | ---: | ---: |
| wave | `64` | `64` |
| SGPR | `14` | `14` |
| VGPR | `57` | `162` |
| LDS bytes | `29184` | `28672` |
| spills | `0` | `0` |
| `v_wmma_f16_16x16x16_f16` | `2` | `32` |
| `ds_load_b64` | `64` | `64` |
| `ds_store_b16` | `100` | `96` |
| `ds_load_u16_d16` | `96` | `96` |
| `buffer_store_b32` | `96` | `160` |
| barriers | `3` | `3` |
| `s_waitcnt` | `104` | `103` |

Decision:
weaken the "K2 plus 96 halfword-stage pairs alone" hypothesis, but do not call
the minimized surface fully clean. The accsink shape is a flaky, later-failing
reproducer, while the mixed160 shape is deterministic and fails earlier. The
remaining production blocker is still the combined high-live accumulator raw
store plus expanded halfword stage surface. A compact compiler/runtime
reproducer should preserve that combined surface; a production route should
avoid it by using a lower-level cooperative-store primitive or a low-live
batching strategy that does not restage the whole tile through this scalarized
path.

## Real Q8 B-Copy Split-Selected Stage Probe

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-bcopy-stage-repro-20260619-103117/`

Mode:
`hrx-hip-bench-q8-wmma-repro --mode=array8-fullb-2phase-bcopy-stage`

Purpose:
separate the now-cleared real-Q8 compute/B-fragment materialization path from
the production split-selected halfword stage writeback primitive. The new mode
keeps the passing two-phase real-Q8 B-copy topology and changes only the final
writeback from raw selected stores to the shared catalog helper
`hrx_q8_0_wmma_vk128_store_acc_f16_row_major_w64_fast_half_buffer_split_selected`.

Controls:

| Mode | Shape | Result |
| --- | --- | --- |
| `array8-fullb-2phase-bcopy` | raw selected buffer stores | pass at `cols=64` and `cols=33`, `bad=0`, `nan=0`, max_abs about `0.11` |
| `array8-fullb-2phase-bcopy-stage` | split-selected halfword LDS stage writeback | fail at aligned `cols=64`, `bad=192`, `nan=0`, max_abs `1.94406` |

The stage failure is finite and localized to upper-column groups:
`group=12 bad=48`, `group=13 bad=48`, and `group=14 bad=96`. The aligned-only
stage run is intentional: the shared padded helper does not bounds-check narrow
tails, so odd/tail coverage remains on the raw-store control and production
backend-op rows rather than this unsafe fixture mode.

Static extraction:
`device.hsaco` was extracted from the CMake-built HIP bench with
`llvm-objdump --offloading`; `device-amdgcn-objdump.txt` and
`device-readelf-notes.txt` are saved in the artifact. The staged instantiations
show the expected additional `512` byte halfword stage surface
(`group_segment_fixed_size=10752` versus `10240` for raw B-copy) and no private
segment.

Decision:
reject split-selected halfword stage writeback as a production Q8_0 direct-WMMA
repair. The raw B-copy control clears the real Q8 compute and B-fragment
materialization contract; the remaining direct-WMMA blocker is the selected
staged writeback/lane-ownership contract for upper output coordinates. The next
production-directed path should be a lower-level cooperative store/writeback
primitive or a different lane map, not another clone of this HIP C++ helper.

## Raw Copy Catalog Transfer

Artifacts:

- `cache/hrxv1/gfx1151/q8_0-raw-bcopy-focused-20260619-103916/`
- `cache/hrxv1/gfx1151/q8_0-raw-abcopy-focused-20260619-104246/`

Routes:

- `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_bcopy_bufferstore_f16acc_wg256_f32`
- `hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_abcopy_bufferstore_f16acc_wg256_f32`

Purpose:
test whether the passing real-Q8 fixture copy controls survive the real catalog
ABI after removing the rejected split-selected halfword stage helper. The first
route uses raw selected buffer stores plus explicit B-fragment materialization.
The second route adds explicit A-fragment materialization, matching the
cleanest `array8-fullb-2phase-abcopy` fixture variant.

Focused p512 result:

| Route | Selected Rows | Result |
| --- | --- | --- |
| raw B-copy | `ffn_out`, `ffn_gate`, `result_output` | fail finite: ERR `0.250038837`, `0.248936879`, `0.250292661` |
| raw A+B-copy | `ffn_out`, `ffn_gate`, `result_output` | fail finite: ERR `0.249956095`, `0.250318662`, `0.250028868` |

In both runs, Vcur/Qcur stayed on the packed Q8_1 split-qsum provider and the
new direct-WMMA route selected only for the intended large dense rows.

Static facts:
both routes emit wave64 code with LDS `20480`, no private segment, `32`
`v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`, `128 buffer_store_b32`, two
barriers, and no halfword LDS stage.

Decision:
reject the raw-copy catalog transfer. Explicit fragment materialization moves
the route from NaN/ERR~2.7 to finite ERR~0.25, but it still fails the
CPU-reference gate by a wide margin. The remaining blocker is no longer just
B-fragment materialization or the split-selected halfword stage helper; it is
the full catalog lane/accumulator/rounding contract for this direct-WMMA
source shape. The next production-directed path should either use a different
cooperative-store/lane primitive or pivot back to the packed-Q8_1 family with
RADV-derived schedule changes.

## Group12 Remap Stage Contract

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-remap-stage-20260619-112352/`

Modes:

- `remap-c0-s12`
- `remap-c0-s12-stage-selected`
- `remap-c12-s0`
- `remap-c12-s0-bcopy-stage-selected`
- `remap-c12-s0-abcopy-stage-selected`

Purpose:
split the group12 selected-stage failure into compute-group and store-group
axes. The previous `single-group12-*-stage-selected` probes showed a
deterministic finite failure in the group12 output tile, but did not prove
whether the bad values came from the `col_sub=3` WMMA accumulator, the
group12 output coordinate map, or their combination.

Focused results:

| Mode | p64 Result | Notes |
| --- | --- | --- |
| `remap-c0-s12` | pass, `bad=0`, max_abs `0.00242398` | raw store control |
| `remap-c0-s12-stage-selected` | pass, `bad=0`, max_abs `0.0660001` | compute group0, selected-stage store into group12 |
| `remap-c12-s0` | fail, `bad=256`, max_abs `13.0649` | uncopied group12 B-fragment remap is invalid |
| `remap-c12-s0-bcopy-stage-selected` | fail, `bad=31`, max_abs `514.867` | B-copy alone is insufficient for this remap |
| `remap-c12-s0-abcopy-stage-selected` | pass, `bad=0`, max_abs `0.114` | compute group12 with A+B-copy, selected-stage store into group0 |

The prior bad-sample diagnostic for `single-group12-bcopy-stage-selected` and
`single-group12-abcopy-stage-selected` showed no NaNs/infinities and a narrow
bad-lane pattern: col `50` on odd rows and col `57` on even rows, with lanes
`18/50` and `9/41` across slots `0..3`. The selected-only helper removed
dummy OTHER_OPSEL writes, but the same group12 output still failed with
`bad=16` and max_abs about `0.346-0.348`.

Decision:
reject the isolated compute-bug and isolated store-coordinate-bug hypotheses.
Group12 compute can pass when A+B fragments are explicitly materialized and
stored through the selected-stage path into group0 coordinates; selected-stage
store into group12 coordinates can pass when fed by group0 compute. The failure
appears only when the `col_sub=3`/group12 accumulator is stored back through
the group12 selected-stage lane map. Treat this as a combined WMMA accumulator
plus selected staged writeback compiler/register-layout contract. Do not
promote another scalarized HIP C++ selected-stage route until a lower-level
cooperative store/lane primitive or compact compiler reproducer explains this
combined case.

## Group12 Accumulator-Copy Stage Probe

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-group12-regcopy-20260619-113027/`

Purpose:
test whether the combined group12 selected-stage failure is repairable by
explicitly moving the WMMA accumulator vector before scalarized halfword LDS
writeback.

Focused results:

| Mode | p64 Result | Notes |
| --- | --- | --- |
| `single-group12-bcopy-stage-selected` | fail, `bad=16`, max_abs `0.346383` | baseline |
| `single-group12-bcopy-stage-selected-acccopy` | fail, `bad=12`, max_abs `0.346383` | slot 0 failures disappear |
| `single-group12-bcopy-stage-selected-regcopy` | fail, `bad=46`, max_abs `514.864` | per-slot copy worsens |
| `single-group12-abcopy-stage-selected` | fail, `bad=16`, max_abs `0.348336` | baseline with A+B-copy |
| `single-group12-abcopy-stage-selected-acccopy` | fail, `bad=12`, max_abs `0.348337` | slot 0 failures disappear |
| `single-group12-abcopy-stage-selected-regcopy` | fail, `bad=43`, max_abs `514.867` | per-slot copy worsens |

Static extraction:
`cache/hrxv1/gfx1151/q8-wmma-group12-regcopy-20260619-113027/static/`
contains the extracted gfx1151 object, notes, disassembly, and
`symbol-summary.txt`. The group12 selected-stage template instantiations were
emitted separately. One-time accumulator-copy variants add a small `v_mov_b32`
surface; per-slot regcopy variants add more move pressure and produce large
wrong values.

Decision:
reject accumulator-copy as a route workaround. The fact that one-time copying
removes slot `0` failures confirms the combined bug is sensitive to register
layout and source spelling, but it does not produce a correctness-clean path.
Per-slot copying is actively harmful. The next production-directed Q8_0 move
should be a lower-level cooperative store/lane primitive or compact compiler
reproducer, not more scalarized selected-stage HIP C++ copies.

## Group12 Synthetic Selected-Stage Contract Probe

Artifact:
`cache/hrxv1/gfx1151/wmma-f16-group12-selected-stage-20260619-113828/`

Purpose:
test whether the current real-Q8 group12 selected-stage failure can be reduced
to WMMA plus the selected halfword LDS writeback contract without carrying Q8
dequant/model data. The new `hrx-hip-bench-wmma-f16-lane-map` modes use the
same 256-thread block / wave0 ownership shape, group12 output coordinates, a
prod-stride LDS fragment load pattern, two WMMA instructions, and raw-vs-staged
comparison for both OPSEL halves.

Focused results:

| Mode | Result |
| --- | --- |
| `group12-selected-stage-contract` | pass, `active=256`, `mismatch=0`, `contract_valid=1` |
| `group12-selected-stage-contract-hi` | pass, `active=256`, `mismatch=0`, `contract_valid=1` |
| `single-group12-abcopy-stage-selected` real-Q8 control | fail, `active=256`, `bad=16`, max_abs `0.348336` |

Static extraction:
`cache/hrxv1/gfx1151/wmma-f16-group12-selected-stage-20260619-113828/static/`
contains lane-map and Q8-repro gfx1151 objects, notes, disassembly, per-symbol
disassembly, and summaries. The synthetic selected-stage symbols are wave64,
no-spill, LDS `11776`, SGPR `22`, VGPR `40`, and two WMMA. The real Q8
selected-stage group12 symbols remain wave64/no-spill but carry the Q8
load/dequant dependency surface at LDS `10752`, SGPR `41`, VGPR `117`, and two
WMMA.

Decision:
reject the too-small minimization. WMMA plus selected halfword staging alone is
not enough to reproduce the group12 failure. The real failure still requires
more of the Q8 fragment/dequant/register dependency surface, or a lower-level
cooperative store/lane primitive that bypasses the brittle scalarized HIP C++
writeback spelling.

## Group12 Dual Raw/Stage Order Probe

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-dual-stage-20260619-114444/`

Purpose:
preserve the real Q8_0 group12 A+B-copy dependency surface, compute the
group12 accumulator once, and write it to separate raw and selected-stage
outputs in the same dispatch. This directly compares raw vs staged values
before treating the difference as a CPU-reference failure. Two source orders
were tested: raw writeback before selected-stage writeback, and selected-stage
writeback before raw writeback.

Focused results:

| Mode | Result |
| --- | --- |
| `single-group12-abcopy` | raw control passed, `bad=0`, max_abs `0.00239494` |
| `single-group12-abcopy-stage-selected` | selected-stage control failed, `bad=16`, max_abs `0.348336` |
| `single-group12-abcopy-dual-stage-raw-first` | raw passed, staged finite but shifted: `mismatch=60`, staged max_abs `0.154939`, mismatch max_abs `0.156356` |
| `single-group12-abcopy-dual-stage-stage-first` | intrusive failure: `raw_sentinel=128`, `staged_sentinel=64`, `mismatch=44` |

Static extraction:
`cache/hrxv1/gfx1151/q8-wmma-dual-stage-20260619-114444/static/`
contains the extracted gfx1151 object, notes, full disassembly, per-symbol
disassembly, and summaries. The dual-stage symbols are wave64, no-spill, LDS
`10752`, SGPR `34`, VGPR `117`, and two WMMA. That matches the real selected
Q8 dependency surface much more closely than the prior synthetic minimization.

Decision:
reject dual raw/stage consumption as a production workaround. The raw-first
consumer partially changes the staged error, and stage-first disrupts later
stores, proving the remaining group12 selected-stage issue is strongly
register-lifetime/source-order sensitive. The next parity-directed step should
stop varying scalarized HIP C++ staging around this helper and instead use a
lower-level cooperative store/lane primitive or a compiler-facing reproducer
that preserves the real Q8 dependency surface.

## Vulkan Cooperative-Store Extract

Artifact:
`cache/hrxv1/gfx1151/q8_0-coopmat-store-extract-20260619-115145/`

Tooling change:
`sources/llama.cpp/tools/vulkan-oracle/extract_coopmat_schedule.py` now emits
compact store windows in the cooperative-matrix extract, alongside the SPIR-V
cooperative-matrix ops, Vulkan source window, RADV resources, first-WMMA
window, and optional HIP/RADV compare summary.

Source and SPIR-V facts:

- `mul_mm.comp` lines 401-404 are the full aligned path: cast accumulator to
  `D_TYPE` cooperative matrix and `coopMatStore` directly to `data_d`.
- Lines 405-423 are the unaligned/partial fallback: store accumulator to
  `coopmat_stage`, then scalar-copy guarded elements to `data_d`.
- The p512 production Q8_0 oracle uses
  `matmul_q8_0_f32_f16acc_aligned_l`, spec
  `[256,128,128,32,64,64,2,16,16,16,64]`.
- SPIR-V contains two cooperative-matrix loads, one cooperative-matrix
  multiply-add, and three cooperative-matrix stores. The store sites correspond
  to the direct aligned path and the LDS fallback paths.

RADV p512 output/writeback facts:

- Resources: `SGPR=108`, `VGPR=192`, `LDS=22528`, no spills, `3740`
  instructions.
- Key opcodes: `32 v_wmma_f16_16x16x16_f16`, `64 ds_load_b64`,
  `128 ds_load_u16_d16`, `128 ds_store_b16`, `192 buffer_store_b32`.
- The compact store windows separate early LDS setup stores from repeated
  output windows. The output windows begin at lines `797`, `852`, `895`,
  `939`, `993`, `1048`, `1091`, and `1135` in the RADV ISA extract, each
  carrying `buffer_store_b32` plus either direct `ds_store_b16` staging or
  `ds_load_u16_d16` reloads.
- The first WMMA window still matches the prior issue-window evidence:
  `ds_load_b64` from bases `v112`/`v113`, final pre-WMMA
  `s_waitcnt lgkmcnt(51)`, and `32` static WMMA sites across the full kernel.

Current production HIP delta:
the compare embedded in this extract is against the current accepted BN128
packed-Q8 route, not the rejected direct-WMMA route. That contrast is useful:
the active HIP route is wave32, `v_dot4_i32_iu8`, LDS `4352`, VGPR `152`, and
only `32 global_store_b32` output stores. It is not in the same writeback or
compute family as RADV's large aligned cooperative-matrix route. Therefore,
parity work has two valid directions:

- clone RADV's cooperative-matrix family with a lower-level store/lane
  primitive that can plausibly reproduce the `192 buffer_store_b32` output
  surface; or
- stay in the packed-Q8 dot4 family and import only schedule facts that are
  compatible with that family, without claiming to be a Vulkan coopmat clone.

Decision:
use this extract as the next Q8_0 p512 writeback contract. Stop adding
scalarized selected-stage HIP C++ variants as production candidates. The next
probe must either expose the cooperative-matrix global-store ownership
mechanically, or be documented as a packed-Q8/dot4 schedule pivot with its own
prior row and focused correctness/timing gate.

## HIP Coopstore Contract Sweep

Artifacts:

- first aborted run:
  `cache/hrxv1/gfx1151/coopmat-store-contract-recheck-20260619-115310/`
- full mode sweep:
  `cache/hrxv1/gfx1151/coopmat-store-contract-sweep-20260619-115359/`
- repeated key modes:
  `cache/hrxv1/gfx1151/coopmat-store-contract-repeat-20260619-115655/`

Purpose:
use the existing CMake-built `hrx-hip-bench-coopmat-store-contract` fixture to
bracket whether the RADV-like `192 buffer_store_b32` surface is expressible in
HIP when WMMA accumulator dependencies are present, and whether the
halfword-LDS staged output surface is safe enough to keep pursuing.

Focused results:

| Mode | Result |
| --- | --- |
| `linear192` | pass, `bad=0` |
| `branch192` | pass, `bad=0` |
| `radv-mixed96` | pass, `bad=0` |
| `radv-mixed192` | flaky: one sweep pass, but repeated run failed 4/5 with `bad=32..64` at group 34 slot 2 |
| `wmma-lds-k2-mixed96` | pass, `bad=0` |
| `wmma-lds-k2-mixed128` | pass, `bad=0` across repeated run |
| `wmma-lds-k2-mixed160-lo` | fail, repeated `bad=1600` |
| `wmma-lds-k2-mixed160-hi` | fail, `bad=1600` in sweep |
| `wmma-lds-k2-radv-mixed192` | fail, repeated `bad=2112` |
| `wmma-lds-k2-stagefirst-mixed192` | fail, `bad=2112` |
| `wmma-lds-k2-direct192-raw` | pass, `bad=0` across repeated run |
| `wmma-lds-k2-direct160-raw` | pass, `bad=0` |
| `wmma-lds-k2-direct64` | pass, `bad=0` |

Static extraction:
the sweep unbundled the executable `.hip_fatbin` with `clang-offload-bundler`
and wrote the full device object and symbol summary under
`cache/hrxv1/gfx1151/coopmat-store-contract-sweep-20260619-115359/static/`.
Key symbols:

- `coopstore_probe_wmma_lds_k2_radv_mixed192`: wave64, SGPR `14`,
  VGPR `162`, LDS `32768`, no spills, `32` WMMA, `64 ds_load_b64`,
  `128 ds_store_b16`, `128 ds_load_u16_d16`, and
  `192 buffer_store_b32`; correctness fails.
- `coopstore_probe_wmma_lds_k2_mixed128`: wave64, SGPR `14`, VGPR `162`,
  LDS `32768`, no spills, `32` WMMA, `64 ds_load_b64`,
  `64 ds_store_b16`, `64 ds_load_u16_d16`, and
  `128 buffer_store_b32`; correctness passes.
- `coopstore_probe_wmma_lds_k2_direct192_raw`: wave64, SGPR `14`,
  VGPR `153`, LDS `16384`, no spills, `32` WMMA, `64 ds_load_b64`,
  `192 buffer_store_b32`, and no halfword output-stage load/store pair;
  correctness passes.

Decision:
the HIP compiler can emit a correctness-clean WMMA+LDS+`192 buffer_store_b32`
direct global-store surface. The failure follows the large
`ds_store_b16 -> ds_load_u16_d16 -> buffer_store_b32` halfword output staging
surface, not the 192 direct global stores by themselves. The next Q8_0
production-directed probe should port `direct192_raw`-style output ownership
into the real Q8 WMMA dependency surface and compare against CPU-reference
rows before any catalog promotion. Do not spend another route attempt on
large staged-halfword output unless the staging hazard is independently
explained.

## Real-Q8 Full Direct-Store Copy Pivot

Artifact:
`cache/hrxv1/gfx1151/q8-wmma-array16-copy-pivot-20260619-120609/`

Purpose:
move the direct raw-store contract from the synthetic coopstore fixture into
the real Q8_0 load/dequant/WMMA dependency surface. The new
`hrx-hip-bench-q8-wmma-repro` modes compute the full 64x64 tile in one
dispatch, write all 16 16x16 groups with raw `buffer_store_b32`, and compare
against CPU reference for aligned `cols=64` and odd/tail `cols=33`.

Focused results:

| Mode | Result |
| --- | --- |
| `array16-direct-raw` | rejected: p64 and p33 fail in group 0 with NaNs/large finite error |
| `array16-direct-raw-bcopy` | rejected: p33 passes, but p64 fails in groups 12-15 with `bad=912`, `nan=288` |
| `array16-direct-raw-abcopy` | passes p64 and p33, `bad=0`, max_abs `0.00268994` / `0.00262882` |
| `array8-fullb-2phase-abcopy` | existing no-spill control also passes p64 and p33 at the same max_abs scale |

Static extraction:
the artifact unbundles the built executable `.hip_fatbin` into
`static/fatbin.gfx1151.o` and records `static/summary.md`.

Key symbols:

- `array16-direct-raw`: wave64, SGPR `64`, VGPR `195`, LDS `10240`, no
  private segment, `32` WMMA, `64 ds_load_b64`, and
  `64 buffer_store_b32`; correctness fails.
- `array16-direct-raw-bcopy`: wave64, SGPR `64`, VGPR `256`, private segment
  `304`, `65 scratch_load`/`65 scratch_store`, `32` WMMA, and
  `64 buffer_store_b32`; p64 correctness fails.
- `array16-direct-raw-abcopy`: wave64, SGPR `64`, VGPR `256`, private segment
  `84`, `20 scratch_load`/`20 scratch_store`, `32` WMMA, and
  `64 buffer_store_b32`; correctness passes p64 and p33.
- `array8-fullb-2phase-abcopy`: each phase is wave64, SGPR `64`, VGPR `247`,
  no private segment, `16` WMMA, `64 ds_load_b64`, and
  `32 buffer_store_b32`; combined two-phase correctness matches the passing
  array16 A+B-copy mode without scratch traffic.

Decision:
the real-Q8 direct raw-store path can be made correctness-clean when both A
and B fragments are copied before WMMA, but the single-dispatch full-16
variant spills. This is not a production route yet. The production-facing
direction is to preserve the A+B-copy lifetime boundary while avoiding private
memory, most likely by keeping two no-spill output phases or by replacing the
copy barrier with a lower-level/lane-owned spelling that keeps the emitted
schedule close to the no-spill phase controls.

## Inline-WMMA plus Fast-Half Split-Selected Catalog Rejection

Artifact:
`cache/hrxv1/gfx1151/q8-asmwmma-fast-half-split-selected-focused-20260619-145643/`

Static compare:
`cache/hrxv1/gfx1151/q8-asmwmma-fast-half-split-selected-static-20260619-145526/compare.md`

Purpose:
test the direct cross-product of two prior facts:

- inline WMMA repaired the production B operand contract for the plain
  packstage route;
- fast-half split-selected staging was the closest HIP C++ selected-half route
  to RADV's halfword LDS output topology.

Static result:
the new CMake/Ninja-built catalog object reaches wave64, SGPR `28`, VGPR
`212`, LDS `22528`, no spills, `32` WMMA, `64 ds_load_b64`,
`128 ds_store_b16`, `128 ds_load_u16_d16`, `128 buffer_store_b32`,
`2 ds_store_b32`, and two barriers. It therefore matches RADV on LDS bytes,
WMMA count, B64 LDS reads, halfword LDS output-stage counts, barriers, and
spill policy. It still misses the two strongest RADV facts: `192
buffer_store_b32` output stores and the high-latency first-WMMA issue window
(`final_pre_wmma_lgkmcnt=0` versus RADV `51`).

Focused route/correctness:
p33 is correctly guarded out and all five p33 rows pass on the existing narrow
packed Q8_1 route. p512 and p513 select the new provider only for the three
large rows. Those rows fail strict CPU reference:

| Row | p512 | p513 |
| --- | ---: | ---: |
| `ffn_out` | `ERR=2.629715292` | `ERR=2.519378912` |
| `ffn_gate` | `ERR=2.122694135` | `ERR=2.249089220` |
| `result_output` | NaN at index `6188064` | NaN at index `6188064` |

Decision:
reject before focused timing or model A/B. This is stronger negative evidence
than the plain fast-half split-selected route: fixing the WMMA B operand with
inline asm does not make the selected halfword LDS output-stage topology
correct on real model-derived large Q8 rows. Stop spending production-route
attempts on this selected-half halfword staging axis unless the cooperative
store lane ownership is independently explained. The next parity-directed Q8
candidate should either use a lower-level lane/store primitive or deliberately
pivot back to the packed-Q8_1 family with its own prior row, rather than claim
to be a RADV cooperative-matrix clone.

## RADV Store Ownership Motif Extract

Large-route artifact:
`cache/hrxv1/gfx1151/q8_0-coopmat-store-ownership-20260619-150308/`

Medium p33 artifact:
`cache/hrxv1/gfx1151/q8_0-coopmat-store-ownership-p33-20260619-150331/`

Tooling:
`sources/llama.cpp/tools/vulkan-oracle/extract_coopmat_schedule.py` now emits
per-basic-block store details and a compact store-motif summary from RADV ISA.

Large p512/p513 route store motifs:

| Motif | Blocks | Buffer stores | LDS stores | LDS loads | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| staged-four | 16 | 64 | 64 | 64 | four `ds_store_b16` at output-stage offsets `20480,20488,20496,20504`, four `ds_load_u16_d16`, four `buffer_store_b32` |
| scalar-reload | 64 | 64 | 0 | 64 | one `ds_load_u16_d16` plus one `buffer_store_b32` per guarded block |
| direct-four | 16 | 64 | 0 | 0 | four direct `buffer_store_b32` at offsets `0,16,32,48` from the same base address |
| guarded-LDS-store | 16 | 0 | 64 | 0 | four guarded `ds_store_b16` blocks feeding later scalar-reload paths |
| setup | 1 | 0 | 4 | 64 | initial LDS setup stores and the pre-WMMA `ds_load_b64` window |

This decomposes the RADV `192 buffer_store_b32` contract into `64 + 64 + 64`
rather than a single uniform fullpair writeback. The selected-half HIP route
only models a 128-store subset; the old fullpair routes tried to make all
192 stores from an invalid full-output accumulator map. The next lower-level
primitive should instead target the motif contract explicitly:

- emit a direct-four path for the four values RADV stores directly;
- emit a staged-four path that stores four halfwords, reloads four halfwords,
  converts, and stores four globals;
- emit scalar-reload/guarded-LDS-store paths only where the RADV motif does;
- preserve the p33 medium route at half scale rather than forcing the large
  motif onto narrow prompt rows.

Medium p33 route contrast:
the same motif family appears at half scale: `96 buffer_store_b32`,
`64 ds_store_b16`, `64 ds_load_u16_d16`, `16 WMMA`, `48 ds_load_b64`,
and LDS output offsets starting at `10240` instead of `20480`.

Decision:
the next production-directed Q8 cooperative-matrix clone should be a
motif-level lane/store primitive, not another selected-half or fullpair helper.
If this cannot be expressed safely in HIP C++, pivot the Q8 work to the
packed-Q8_1 route family and document a separate prior row instead of claiming
to clone RADV's cooperative-store lowering.

## RADV Motif192 HIP Fixture

Artifact:
`cache/hrxv1/gfx1151/coopstore-radv-motif192-20260619-150932/`

Fixture:
`hrx-hip-bench-coopmat-store-contract --mode=wmma-lds-k2-radv-motif192`

The fixture attempted the motif-level RADV clone directly in HIP C++:

- `64` direct accumulator `buffer_store_b32` writes;
- `64` staged writes through halfword LDS for groups 16-31;
- `64` staged writes through a reused halfword LDS window for groups 32-47;
- explicit `lgkmcnt` waits between each stage-store and stage-load pair.

Controls:

- `wmma-lds-k2-direct192-raw` passed, so the 192 raw store surface is viable.
- `wmma-lds-k2-mixed160-splitstage` passed, so the staged halfword path is
  viable when separated by the existing phase boundary.

Result:
the first oversized 32-group LDS-stage allocation failed correctness even after
explicit waits and logical LDS window reuse: `bad=1792`, first failure at
`group=32 slot=2 lane=0`, `actual=512`, `expected=16386`. Tightening the
allocated stage window to the 16 groups actually reused by the motif made the
same mode pass exactly: `bad=0`, `max_abs=0`.

Emitted HIP ISA:

- wave64, SGPR `14`, VGPR `137`, group segment `24576`, private segment `0`;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `128 ds_store_b16`;
- `128 ds_load_u16_d16`;
- `192 buffer_store_b32`;
- `168 s_waitcnt`;
- two barriers.

Decision:
this is positive standalone prior evidence, not a production route yet. HIP C++
can express a correctness-clean motif-level clone when the LDS stage window is
tight. Do not promote from aggregate counts alone: the failed oversized variant
shows the same logical writeback can corrupt when the resource/window contract
changes. The next Q8 route attempt should port the tight motif into the real
catalog ABI and keep a separate p33 half-scale route instead of forcing the
large motif onto narrow prompt rows.

## Motif192 Catalog Transfer

Route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_bufferstore_f16acc_wg256_f32`

Artifact:
`cache/hrxv1/gfx1151/q8-asmwmma-motif192-focused-20260619-153414/`

The route was wired as a real HRX v1 catalog provider, not only a generated
JSON row. Required runtime edits were explicit provider storage/reset/load,
early Q8_0 direct-provider selection, the opt-in selector, and 128-column tile
accounting. This exposed a useful HRX v1 productionization issue: generated
catalog metadata and CMake-built HSACOs are necessary but not sufficient while
the runtime still has hard-coded provider handles.

Static emitted facts:

- wave64, SGPR `56`, VGPR `212`, group segment `22528`, private segment `0`;
- `32 v_wmma_f16_16x16x16_f16`;
- `64 ds_load_b64`;
- `128 ds_store_b16`;
- `128 ds_load_u16_d16`;
- `192 buffer_store_b32`;
- `60 s_waitcnt`;
- three barriers.

Focused result:
p33 stayed on the narrow Q8_1 x4 route and passed. p512 selected motif192 for
the first wide row after `Vcur`/`Qcur`, then faulted on `ffn_out` with a GPU
page-not-present/supervisor-privilege memory access fault.

Decision:
reject the catalog transfer before timing. The standalone motif is still the
right prior, but the real ABI clone is unsafe. The next step should be a small
fault reproducer around the real output/LDS addressing contract or a lower-level
store primitive, not another aggregate-count clone.

## Motif192 Address Repro

Diagnostic modes:
`hrx-hip-bench-q8-wmma-repro --mode motif192-synth-address` and
`hrx-hip-bench-q8-wmma-repro --mode motif192-wmma-address`

Artifacts:
`cache/hrxv1/gfx1151/q8-motif192-synth-address-20260619-154442/` and
`cache/hrxv1/gfx1151/q8-motif192-wmma-address-linebuf-20260619-154833/`

Result:
the synthetic arbitrary-accumulator row/column motif corrupts without fault.
The WMMA-payload variant reproduces the production-class p512 memory fault in
the standalone CMake-built repro kernel after executing smaller aligned and
odd shapes. The failing dispatch reports `grid=[8192,4,1]`,
`group_seg_size=22528`, and `private_seg_size=0`.

Decision:
the unsafe condition is narrower than HRX runtime/model dispatch and broader
than output offset arithmetic alone: it requires the WMMA payload plus the real
four-wave motif writeback/address shape. Do not resume motif192 as a catalog
route until the store primitive or lane ownership is reduced further.

## Motif192 Staged-Window Isolation

Diagnostic modes:
`motif192-wmma-direct-address`, `motif192-wmma-stage16-address`, and
`motif192-wmma-stage32-address`.

Artifacts:
`cache/hrxv1/gfx1151/q8-motif192-wmma-direct-address-20260619-155416/`,
`cache/hrxv1/gfx1151/q8-motif192-wmma-stage16-address-20260619-155431/`, and
`cache/hrxv1/gfx1151/q8-motif192-wmma-stage32-address-20260619-155447/`

Result:
direct-only WMMA payload plus row/column raw stores does not fault through
4096x513, though it remains incorrect for half the columns. The first staged
halfword LDS reload/writeback window faults independently at 4096x512. The
second reused staged window also faults independently at 4096x512. Both staged
faults report the same p512-style dispatch surface: `grid=[8192,4,1]`,
`group_seg_size=22528`, and `private_seg_size=0`.

Decision:
the production fault is specifically tied to the staged halfword LDS
reload/writeback primitive under real BM128/BN128 output addressing. The next
schedule attempt should not clone the same staged path into another catalog
route. Either replace the staged writeback with a safer primitive that still
matches the RADV ownership, or pivot to a different Q8 large-route dataflow
with explicit RADV/HSACO evidence.

## Motif192 Exact-Shape Threshold Sweep

Diagnostic update:
the motif address repro modes now accept `--rows N --cols N`, allowing one
shape per process and preserving evidence around GPU faults.

Artifacts:
`cache/hrxv1/gfx1151/q8-motif192-stage-threshold-20260619-155915/` and
`cache/hrxv1/gfx1151/q8-motif192-stage-threshold-repeat-20260619-155951/`

Result:
the staged fault is not a simple monotonic row threshold. Stage16 faulted at
1024 and 4096 rows in one sweep while passing intermediate rows. Repeated
1024/1536/4096 controls showed direct-only never faulted, while stage16 and
stage32 faults were intermittent and appeared at 1024 and/or 4096 depending on
the repetition.

Decision:
shape gating is not a viable rescue for motif192 staged writeback. Treat the
staged halfword primitive as allocation/scheduling sensitive under the real
row/column ABI. A production candidate needs a different staged store/load
contract, a lower-level RADV-like primitive with proven stability, or a
different Q8 large-route dataflow.

## Motif192 Waitload Revision

Diagnostic modes:
`motif192-wmma-stage16-waitload-address`,
`motif192-wmma-stage32-waitload-address`, and
`motif192-wmma-waitload-address`.

Artifacts:
`cache/hrxv1/gfx1151/q8-motif192-stage-waitload-20260619-160737/`,
`cache/hrxv1/gfx1151/q8-motif192-full-waitload-20260619-161019/`,
`cache/hrxv1/gfx1151/q8-asmwmma-motif192-waitload-focused-20260619-161223/`,
and
`cache/hrxv1/gfx1151/q8-asmwmma-motif192-waitload-perf-20260619-161300/`.

Result:
the standalone passing RADV-motif fixture waited immediately after
`ds_read_u16_d16`; the catalog/repro motif did not. Adding the same
`s_waitcnt lgkmcnt(0)` after the staged halfword LDS load fixes both isolated
staged windows and the full motif across repeated standard and odd shapes:
`1024x512`, `1536x512`, `4096x512`, `4096x513`, `513x512`, and `1025x513`.
The production catalog route also passes focused p33, p512, and p513
CPU-reference gates after being rebuilt by CMake/Ninja. Route traces prove
p512/p513 select the motif provider for `ffn_out`, `ffn_gate`, and
`result_output`.

Decision:
reject for default production on performance. Same-runner focused timing shows
p512 at `1.278x` baseline time and p513 at `1.309x`; selected wide rows lose by
`1.266x-1.431x`. The waitload route is valuable because it converts the RADV
motif clone from a faulting transfer into a correctness-clean schedule rung.
The next candidate should keep the explicit wait/load-use contract but recover
the schedule quality, especially VGPR/wait pressure and the selected wide-row
throughput.

## Motif192 K2 Waitload Issue-Window Repro

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-waitload-20260619-162840/`

Diagnostic modes:
`motif192-wmma-k2-directwait-waitload-address` and
`motif192-wmma-k2-depwait-waitload-address`.

What changed:

- loaded both VK128 K tiles' four A fragments and four B fragments before the
  first WMMA;
- issued the RADV-like `lgkmcnt` ladder with inline-asm WMMA;
- preserved the corrected motif192 wait-after-load halfword writeback.

Runtime result:
both directwait and depwait modes passed `128x128`, `128x129`, `129x128`,
`1024x512`, `4096x512`, and `4096x513` with no bad values, NaNs, infinities,
or sentinels.

Static result against the RADV Q8_0 p512 large oracle:

| Metric | RADV | K2 motif repro |
| --- | ---: | ---: |
| SGPR | 108 | 62 |
| VGPR | 192 | 190 |
| LDS | 22528 | 22528 |
| spills/private | 0 / 0 | 0 / 0 |
| `ds_load_b64` | 64 | 64 |
| `v_wmma_f16_16x16x16_f16` | 32 | 32 |
| `ds_store_b16` | 128 | 132 |
| `ds_load_u16_d16` | 128 | 128 |
| `buffer_store_b32` | 192 | 192 |
| first-window score | `59/59/lgkmcnt(51)` | `64/64/lgkmcnt(51)` |

Decision:
accept as a positive standalone schedule prior, not a production route. This
proves the corrected motif192 writeback can coexist with a two-K RADV-like
issue window without the earlier production K2 spill cliff, at least with
synthetic A/B payloads. The directwait and depwait instantiations compile to
the same tracked contract, so the useful source shape appears to be the tight
motif plus inline-asm WMMA ladder rather than dependency-copy machinery.

The next Q8_0 promotion step should be a new real-data catalog branch/wrapper
or a specialized real-Q8 repro preserving this exact `64/64/lgkmcnt(51)`,
`192 buffer_store_b32`, and wait-after-load contract. The existing production
kernel cannot reach this by just setting both old route macros, because the
motif192 and K2 branches are currently mutually exclusive.

## Motif192 K2 Real-Data Catalog Transfer

Route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_directwait_bufferstore_f16acc_wg256_f32`

Env:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_ASMWMMA_MOTIF192_K2_DIRECTWAIT_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-20260619-163905/`

Static result:

| Metric | RADV | Real-data K2 motif |
| --- | ---: | ---: |
| SGPR | 108 | 56 |
| VGPR | 192 | 256 |
| LDS | 22528 | 22528 |
| private segment | 0 | 68 |
| VGPR spills | 0 | 16 |
| `ds_load_b64` | 64 | 64 |
| `v_wmma_f16_16x16x16_f16` | 32 | 32 |
| `ds_store_b16` | 128 | 128 |
| `ds_load_u16_d16` | 128 | 128 |
| `buffer_store_b32` | 192 | 192 |

Focused correctness:
p33, p512, and p513 pass CPU-reference gates. Route traces are policy-clean:
p33 remains on existing narrow/default routes, and p512/p513 select this route
only for `ffn_out`, `ffn_gate`, and `result_output`.

Focused timing:

| Row | p512 variant/default | p513 variant/default |
| --- | ---: | ---: |
| total | 1.845 | 1.924 |
| `ffn_out` | 2.327 | 2.151 |
| `ffn_gate` | 1.912 | 1.938 |
| `result_output` | 1.811 | 1.932 |

Decision:
reject for promotion. The route proves the standalone K2 motif prior can be
made correctness-clean on the real catalog ABI, but the real-data compile
reintroduces the HIP pressure cliff: VGPR `256`, spills, and nearly 2x focused
runtime loss. The next Q8 large-route candidate should not add another wrapper
around this branch. It needs to preserve the standalone no-spill source shape,
reduce the ABI/live-fragment pressure, or move the motif primitive lower level
before model-level timing.

## Motif192 K2 Real-Data K32 Fixture

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-k32-20260619-165112/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built mode
  `motif192-wmma-k2-realdata-k32-directwait-waitload-address`;
- kept the tight standalone K2 motif issue/writeback surface;
- replaced synthetic A/B LDS values with backend-like Q8_0 dequant values and
  backend-like F32 RHS values;
- intentionally fixed `k=32`, so this is a single-BK pressure/correctness
  fixture rather than a production-route benchmark.

Runtime result:

- passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`;
- all rows reported `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`;
- worst max absolute error was `0.0135875`, with NMSE about `4.3e-7`.

Static result:

| Metric | RADV | Real-data K32 fixture | Real-data catalog K2 |
| --- | ---: | ---: | ---: |
| SGPR | 108 | 60 | 56 |
| VGPR | 192 | 193 | 256 |
| LDS | 22528 | 22528 | 22528 |
| private segment | 0 | 0 | 68 |
| VGPR spills | 0 | 0 | 16 |
| `ds_load_b64` | 64 | 64 | 64 |
| `v_wmma_f16_16x16x16_f16` | 32 | 32 | 32 |
| `ds_store_b16` | 128 | 130 | 128 |
| `ds_load_u16_d16` | 128 | 128 | 128 |
| `buffer_store_b32` | 192 | 192 | 192 |
| first pre-WMMA wait | `lgkmcnt(51)` | `lgkmcnt(51)` | not clean standalone |

Interpretation:
real Q8_0 dequant and real RHS values are not sufficient to cause the K2
spill cliff. The tight single-BK fixture keeps the RADV-like resource shape:
wave64, VGPR `193`, no private segment, no spills, `64` B64 LDS reads before
the first WMMA, final pre-WMMA `lgkmcnt(51)`, and the `192 buffer_store_b32`
motif. The production catalog transfer only falls off the cliff when the full
catalog ABI, broader loop surface, and route writeback machinery are present.

Decision:
accept as a diagnostic prior and keep it out of route selection. The next Q8
large-route step should isolate the production-only pressure axis by adding
one feature at a time to this fixture: full-K loop, catalog argument shape,
production writeback, then selector integration. Do not add another full
catalog route until a fixture-level variant preserves the no-spill K32
resource shape after the next production surface is introduced.

## Motif192 K2 Real-Data Full-K Fixture

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-20260619-165821/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built mode
  `motif192-wmma-k2-realdata-fullk-directwait-waitload-address`;
- kept the same standalone K2 motif issue/writeback surface as the passing
  K32 fixture;
- added the full `k=4096` loop over 32-wide K blocks;
- used exact checking for small shapes and deterministic sampled checking for
  large p512/p513-style shapes to keep the fixture practical.

Runtime result:

- passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`;
- all rows reported `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`;
- worst sampled max absolute error was `1.0958`, with NMSE about `4.3e-5`.

Static result:

| Metric | RADV | Real-data K32 fixture | Real-data full-K fixture | Real-data catalog K2 |
| --- | ---: | ---: | ---: | ---: |
| SGPR | 108 | 60 | 60 | 56 |
| VGPR | 192 | 193 | 256 | 256 |
| LDS | 22528 | 22528 | 22528 | 22528 |
| private segment | 0 | 0 | 64 | 68 |
| VGPR spills | 0 | 0 | 15 | 16 |
| `ds_load_b64` static sites | 64 | 64 | 64 | 64 |
| `v_wmma_f16_16x16x16_f16` static sites | 32 | 32 | 32 | 32 |
| `ds_store_b16` | 128 | 130 | 130 | 128 |
| `ds_load_u16_d16` | 128 | 128 | 128 | 128 |
| `buffer_store_b32` | 192 | 192 | 192 | 192 |
| first pre-WMMA wait | `lgkmcnt(51)` | `lgkmcnt(51)` | `lgkmcnt(51)` | not clean standalone |

Interpretation:
the full-K loop alone is sufficient to recreate the production K2 pressure
cliff. The fixture still keeps the RADV-like first-loop issue surface and
passes correctness, but its resource facts collapse to the catalog route's
shape: VGPR `256`, private segment, and VGPR spills. This moves the active
hypothesis away from catalog argument ABI and toward loop-carried accumulator
and fragment lifetime. The compiler can hold the one-BK motif at RADV-like
VGPR, but not the full-K loop with all 16 accumulator groups and the current
source-level fragment topology.

Decision:
reject full-K K2 motif as a production route shape until the loop lifetime is
split or lowered. The next Q8 probe should bracket accumulator lifetime: for
example phase the 16 output groups across multiple loop bodies, reduce the
live accumulator set while preserving RADV-like first-window loads, or move the
K-loop/WMMA ladder to a lower-level spelling that avoids exposing all fragments
and accumulators to HIP register allocation at once.

## Motif192 K2 Real-Data Full-K Phase8 Fixture

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-phase8-20260619-170420/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built mode
  `motif192-wmma-k2-realdata-fullk-phase8-directwait-waitload-address`;
- split the full 16-group output tile into two launches:
  lower groups `0..7` and upper groups `8..15`;
- each phase keeps the BM128/BN128 full-K real-data motif surface but only
  carries eight accumulators through the K loop;
- each phase still loads all four B column fragments, so the lower phase keeps
  the same `64 ds_load_b64` / `lgkmcnt(51)` first-window evidence.

Runtime result:

- passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`;
- all rows reported `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`;
- sampled errors matched the non-phased full-K fixture: worst max absolute
  error `1.0958`, NMSE about `4.3e-5`.

Static result:

| Metric | Full-K 16 groups | Phase8 lower | Phase8 upper |
| --- | ---: | ---: | ---: |
| SGPR | 60 | 58 | 58 |
| VGPR | 256 | 211 | 211 |
| LDS | 22528 | 22528 | 22528 |
| private segment | 64 | 0 | 0 |
| VGPR spills | 15 | 0 | 0 |
| `ds_load_b64` static sites | 64 | 64 | 64 |
| `v_wmma_f16_16x16x16_f16` static sites | 32 | 16 | 16 |
| `ds_store_b16` | 130 | 66 | 66 |
| `ds_load_u16_d16` | 128 | 64 | 64 |
| `buffer_store_b32` | 192 | 96 | 96 |
| first pre-WMMA wait | `lgkmcnt(51)` | `lgkmcnt(51)` | `lgkmcnt(20)` |

Interpretation:
phasing the accumulator set is sufficient to remove the full-K spill cliff in
this source spelling. The lower phase preserves the RADV-like first-window
load/wait shape while reducing live accumulators from 16 to 8. The upper phase
has a smaller first wait because it uses the later B column fragments, but it
also compiles without spills. The open tradeoff is launch/work duplication:
this fixture uses two kernel launches and keeps all B fragments live in each
phase. It is therefore a schedule direction, not yet a production route.

Decision:
accept phase8 as the next production-facing Q8 candidate family. The next step
should either time this phase8 fixture against the full-K 16-group fixture in a
same-runner microbench, or transfer phase8 into an opt-in catalog route with a
single provider submission strategy and focused p512/p513 gates. A direct
catalog transfer should preserve the no-spill resource row first; if it returns
to VGPR `256`/spills, reject before model timing.

## Motif192 K2 Phase8 Timing

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-fullk-phase8-timing-20260619-170826/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built mode `motif192-wmma-k2-realdata-fullk-timing`;
- times the spilling full-K 16-group fixture against the no-spill two-launch
  phase8 fixture in the same process with the same real-data inputs;
- HIP events returned zero on this ROCm build, so the accepted timing output
  uses synchronized host wall-clock timing. This includes launch overhead,
  which is relevant because phase8 currently pays two launches.

Timing result:

| Shape | Full-K 16 groups | Phase8 two-launch | Phase8/full-K |
| --- | ---: | ---: | ---: |
| `128x128x4096` | `0.999 ms` | `1.798 ms` | `1.80x` |
| `1024x512x4096` | `1.090 ms` | `1.842 ms` | `1.69x` |
| `4096x512x4096` | `5.024 ms` | `6.007 ms` | `1.20x` |
| `4096x513x4096` | `5.101 ms` | `6.121 ms` | `1.20x` |

Interpretation:
no-spill is not sufficient. The two-launch phase8 fixture removes private
memory and preserves the lower-phase RADV-like wait window, but duplicated
K-loop work plus launch overhead make it slower than the spilling full-K
fixture in the same runner. Since the spilling full-K/catalog family was
already rejected against the current packed-Q8_1 default, this two-launch
phase8 shape should not be promoted directly.

Decision:
reject two-launch phase8 as a production route shape. Keep the accumulator
lifetime result: reducing the live accumulator set fixes the compiler cliff.
The next useful Q8 probe is a single-launch sequential phase shape or a
lower-level spelling that keeps at most eight accumulators live without
doubling provider submissions; it must preserve the no-spill resource row and
then beat the full-K fixture in the same timing harness before catalog work.

## Motif192 K2 Phase8Seq Single-Launch Probe

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-phase8seq-20260619-171619/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built correctness mode
  `motif192-wmma-k2-realdata-fullk-phase8seq-directwait-waitload-address`;
- added timing mode `motif192-wmma-k2-realdata-fullk-phase8seq-timing`;
- runs lower and upper eight-accumulator phases sequentially inside one kernel
  launch, with accumulator arrays scoped inside each phase;
- preserves the same real Q8_0/F32 inputs, K-loop, motif192 writeback, odd
  shape, and p513 tail coverage as the full-K and two-launch phase8 fixtures.

Correctness:

- passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`;
- all rows reported `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`;
- worst sampled max absolute error was `1.0958`, matching the prior full-K
  fixture tolerance regime.

Static result:

| Metric | Full-K 16 groups | Phase8 two-launch half | Phase8Seq single-launch |
| --- | ---: | ---: | ---: |
| SGPR | 60 | 58 | 76 |
| VGPR | 256 | 211 | 231 |
| LDS | 22528 | 22528 | 22528 |
| private segment | 64 | 0 | 0 |
| VGPR spills | 15 | 0 | 0 |
| `ds_load_b64` static sites | 64 | 128 | 128 |
| `v_wmma_f16_16x16x16_f16` static sites | 32 | 32 | 32 |
| `ds_store_b16` | 130 | 132 | 132 |
| `ds_load_u16_d16` | 128 | 128 | 128 |
| `buffer_store_b32` | 192 | 192 | 192 |
| barriers | 3 | 6 | 6 |
| first pre-WMMA wait | `lgkmcnt(51)` | `lgkmcnt(51)` | `lgkmcnt(51)` |

Timing rerun:

| Shape | Full-K 16 groups | Phase8Seq | Phase8Seq/full-K |
| --- | ---: | ---: | ---: |
| `128x128x4096` | `1.000 ms` | `1.806 ms` | `1.81x` |
| `1024x512x4096` | `1.090 ms` | `1.839 ms` | `1.69x` |
| `4096x512x4096` | `4.932 ms` | `6.183 ms` | `1.25x` |
| `4096x513x4096` | `5.068 ms` | `6.416 ms` | `1.27x` |

Interpretation:
single-launch scoped phasing also removes the VGPR spill cliff, proving the
compiler will honor a scoped eight-accumulator lifetime inside one kernel.
However, it still duplicates the full K loop and doubles the LDS load/barrier
shape, so it remains slower than the spilling full-K fixture. The no-spill
resource row is necessary but not sufficient; the useful next probe needs a
way to reduce live accumulator pressure without replaying the whole K loop for
each half of the output tile.

Decision:
reject phase8seq as a production route shape and keep it as a negative
fixture. The next Q8 direction should not be another whole-K phase split.
Target a lower-level store/accumulator spelling that keeps all 16 output
groups in one K traversal while shortening only the compiler-visible fragment
or writeback lifetimes.

## Motif192 K2 AccPark One-Traversal Probe

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-accpark-20260619-172434/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built mode
  `motif192-wmma-k2-realdata-fullk-accpark-directwait-waitload-address`;
- added timing mode `motif192-wmma-k2-realdata-fullk-accpark-timing`, but did
  not run timing because correctness failed;
- kept one K traversal and one launch;
- after each BK chunk, parked the selected accumulator lanes for all 16 output
  groups in explicit LDS storage, then reloaded them before the next BK chunk;
- goal was to trade explicit LDS accumulator traffic for shorter compiler
  visible accumulator lifetimes without replaying the K loop.

Correctness result:

- failed all exact/odd/tail rows;
- representative failures:
  `128x128` had `bad=5010/8192`, `max_abs=2059.88`, `nmse=46.4751`;
  `4096x512` had `bad=3687/8192`, `max_abs=580.928`, `nmse=20.0988`;
- no NaNs/Infs/sentinels appeared, so this is a wrong accumulator contract,
  not a launch failure or uninitialized-output failure.

Static result:

| Metric | AccPark |
| --- | ---: |
| SGPR | 94 |
| VGPR | 256 |
| LDS | 55296 |
| private segment | 72 |
| VGPR spills | 17 |
| `ds_load_b64` | 64 |
| `ds_load_u16_d16` | 384 |
| `ds_store_b16` | 194 |
| `v_wmma_f16_16x16x16_f16` | 32 |
| `buffer_store_b32` | 192 |
| first-window score | not RADV-like; final pre-hot `lgkmcnt(20)` |

Interpretation:
parking only the final selected OPSEL lanes is not a valid representation of
the WMMA accumulator state. The non-stored half of the `_Float16x8`
accumulator vector is still required for subsequent WMMA updates, even though
final output reads only the selected OPSEL lanes. The attempted explicit LDS
parking also raises LDS to `55 KiB` and still compiles with VGPR spills, so it
does not solve either correctness or pressure.

Decision:
reject selected-lane AccPark before timing. A viable manual parking direction
would need full accumulator-vector preservation, but doing that in LDS would
exceed the useful RADV-like LDS budget and likely reduce occupancy. The next
Q8 probe should instead target source/ASM spelling of the one-traversal
full-accumulator loop, or a narrower output ownership shape that reduces the
number of true accumulators without parking partial accumulator state.

## Motif192 K2 StreamFrag One-Traversal Probe

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-streamfrag-20260619-173258/`

Source:
`sources/llama.cpp/ggml/src/ggml-hrx/tools/hip-bench/q8_wmma_repro_bench.hip.cpp`

What changed:

- added CMake/Ninja-built correctness mode
  `motif192-wmma-k2-realdata-fullk-streamfrag-directwait-waitload-address`;
- added timing mode `motif192-wmma-k2-realdata-fullk-streamfrag-timing`;
- kept one launch, one K traversal, all 16 output accumulators, the motif192
  writeback surface, and the same odd/tail coverage;
- changed only fragment lifetime: instead of materializing
  `a_frag[2][4]` and `b_frag[2][4]` before the WMMA ladder, it loads the A/B
  fragment pair for each WMMA and waits with `lgkmcnt(0)`.

Correctness:

- passed `128x128`, `128x129`, `129x128`, `1024x512`, `4096x512`, and
  `4096x513`;
- all rows reported `bad=0`, `nan=0`, `inf=0`, and `sentinel=0`;
- worst sampled max absolute error was `1.0958`, matching the prior full-K
  fixture tolerance regime.

Static result:

| Metric | Full-K 16 groups | StreamFrag |
| --- | ---: | ---: |
| SGPR | 60 | 60 |
| VGPR | 256 | 168 |
| LDS | 22528 | 22528 |
| private segment | 64 | 0 |
| VGPR spills | 15 | 0 |
| `ds_load_b64` | 64 | 256 |
| `ds_load_u16_d16` | 128 | 128 |
| `ds_store_b16` | 130 | 130 |
| `v_wmma_f16_16x16x16_f16` | 32 | 32 |
| `buffer_store_b32` | 192 | 192 |
| first-window score | `lgkmcnt(51)`, 32 hot ops | `lgkmcnt(0)`, 7 hot ops |

Timing rerun:

| Shape | Full-K 16 groups | StreamFrag | StreamFrag/full-K |
| --- | ---: | ---: | ---: |
| `128x128x4096` | `1.000 ms` | `1.160 ms` | `1.16x` |
| `1024x512x4096` | `1.094 ms` | `1.263 ms` | `1.15x` |
| `4096x512x4096` | `4.980 ms` | `4.060 ms` | `0.82x` |
| `4096x513x4096` | `5.166 ms` | `4.018 ms` | `0.78x` |

Interpretation:
fragment lifetime was a major part of the full-K pressure cliff. StreamFrag
destroys the RADV-like pre-WMMA issue window by serializing fragment loads,
but it removes spills and improves the wide production-width rows by roughly
18-22% versus the spilling full-K fixture. It regresses narrow rows, so it is
not a universal route shape.

Decision:
accept StreamFrag as the next production-facing Q8 direct-WMMA candidate
family for wide p512/p513-style rows only. The next step is an opt-in catalog
transfer with a strict wide-shape selector and focused p512/p513 gates against
the current packed-Q8_1 default. It should remain out of narrow/odd p33 policy
unless separate evidence proves a win there.

## Motif192 K2 StreamFrag Catalog Transfer

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-streamfrag-focused-20260619-174433/`

Route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_streamfrag_bufferstore_f16acc_wg256_f32`

Env:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_ASMWMMA_MOTIF192_K2_STREAMFRAG_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

What changed:

- transferred the StreamFrag one-traversal fixture into the real Q8_0 catalog
  ABI as an opt-in route;
- restricted selection to wide prompt rows with `cols >= 512` and
  `rows >= 8192 || k >= 8192`;
- kept p33/narrow rows on existing packed-Q8_1 routes;
- added the route to the JSON catalog so the HSACO is built by CMake/Ninja.

Focused gate:

- p33, p512, and p513 CPU-reference rows passed;
- p33 route traces stayed on the existing narrow packed-Q8_1 route;
- p512/p513 route traces selected StreamFrag only for `ffn_out`, `ffn_gate`,
  and `result_output`;
- `Vcur` and `Qcur` stayed on the current packed-Q8_1 split-qsum route.

Static catalog HSACO:

| Metric | StreamFrag catalog |
| --- | ---: |
| SGPR | 56 |
| VGPR | 168 |
| LDS | 22528 |
| private segment | 0 |
| VGPR spills | 0 |
| `ds_load_b64` | 256 |
| `ds_store_b16` | 128 |
| `ds_load_u16_d16` | 128 |
| `v_wmma_f16_16x16x16_f16` | 32 |
| `buffer_store_b32` | 192 |

Focused timing versus current default:

| Shape | Total ratio | Selected-row result |
| --- | ---: | --- |
| p512 | `2.058x` slower | `ffn_gate 2.217x`, `ffn_out 2.003x`, `result_output 2.090x` slower |
| p513 | `2.151x` slower | `ffn_gate 2.142x`, `ffn_out 1.809x`, `result_output 2.241x` slower |

Interpretation:
the production ABI preserves the no-spill StreamFrag resource shape, but the
schedule is still far worse than the current packed-Q8_1 split-qsum default.
Removing spills alone did not matter enough because StreamFrag serialized each
fragment pair with `lgkmcnt(0)` and lost the RADV-like issue window. This is a
clear example where static no-spill evidence is necessary but not sufficient.

Decision:
reject StreamFrag catalog promotion. Keep it opt-in as fragment-lifetime
evidence. The next Q8 parity candidate should recover the RADV pre-WMMA
load/issue window while preserving low pressure, or move below HIP C++
fragment/source shaping into a lower-level lane/writeback primitive.

## Motif192 K2 KTileFrag Probe And Catalog Transfer

Fixture artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-ktilefrag-20260619-175606/`

Catalog artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-ktilefrag-focused-20260619-180304/`

Route:
`hrx_mul_mat_vec_q8_0_wmma16x16_vk128_padded_w64_b64group_packstage_asmwmma_motif192_k2_ktilefrag_bufferstore_f16acc_wg256_f32`

Env:
`GGML_HRX_ENABLE_Q8_0_WMMA16_VK128_PADDED_W64_B64GROUP_PACKSTAGE_ASMWMMA_MOTIF192_K2_KTILEFRAG_BUFFERSTORE_F16ACC_WG256_PROMPT=1`

What changed:

- bracketed the middle point between full-K fragment retention and StreamFrag;
- loads one K tile's four A and four B fragments at a time;
- issues that tile's 16 WMMA updates with delayed waits `12/8/4/0`;
- repeats for the second K tile, keeping one launch, one K traversal, all 16
  accumulators, and motif192 writeback.

Fixture result:

- correctness passed `128x128`, `128x129`, `129x128`, `1024x512`,
  `4096x512`, and `4096x513`;
- static row: wave64, SGPR `60`, VGPR `211`, LDS `22528`, private segment
  `0`, no spills, `64 ds_load_b64`, `32 v_wmma`, `130 ds_store_b16`,
  `128 ds_load_u16_d16`, and `192 buffer_store_b32`;
- repeated same-runner timing beat the spilling full-K fixture:
  `128x128 0.923x`, `1024x512 0.882x`, `4096x512 0.655x`, and
  `4096x513 0.685x` of full-K time.

Catalog result:

- built through CMake/Ninja as a JSON catalog route;
- static row: wave64, SGPR `56`, VGPR `212`, LDS `22528`, private segment
  `0`, no spills, `64 ds_load_b64`, `32 v_wmma`, `128 ds_store_b16`,
  `128 ds_load_u16_d16`, and `192 buffer_store_b32`;
- focused p33/p512/p513 CPU-reference gates passed;
- p33 stayed on existing narrow packed-Q8_1;
- p512/p513 selected KTileFrag only for `ffn_out`, `ffn_gate`, and
  `result_output`; `Vcur` and `Qcur` stayed on packed-Q8_1 split-qsum.

Focused timing versus current default:

| Shape | Total ratio | Selected-row result |
| --- | ---: | --- |
| p512 | `1.272x` slower | `ffn_gate 1.289x`, `ffn_out 1.414x`, `result_output 1.263x` slower |
| p513 | `1.335x` slower | `ffn_gate 1.374x`, `ffn_out 1.370x`, `result_output 1.339x` slower |

Interpretation:
KTileFrag is a real compiler/schedule win inside the direct-WMMA family. It
keeps the delayed LDS wait window, preserves the motif192 store surface,
removes the full-K spill cliff, and is much faster than both spilling full-K
and serialized StreamFrag fixture shapes. It still loses to the current
packed-Q8_1 split-qsum production route.

Decision:
reject KTileFrag catalog promotion and keep it opt-in. The next Q8 parity path
should not be another direct-WMMA F32-RHS wrapper unless it changes the
algorithmic class; the evidence now points toward improving packed-Q8_1 toward
the Vulkan large-family schedule, or implementing a lower-level cooperative
matrix/lane-ownership primitive that HIP C++ source shaping has not exposed.

## AccParkFull8 Custom-K Recheck

Artifact:
`cache/hrxv1/gfx1151/q8-motif192-k2-realdata-fullk-accparkfull8-customk-20260620-114615/`

What changed:

- fixed `hrx-hip-bench-q8-wmma-repro` so custom `--k` is honored by the
  real-data full-K fixture family;
- reran the full-vector accumulator parking probe at `k=32`, `k=4096`, exact,
  odd, p512, and p513 shapes;
- extracted the rebuilt benchmark executable's embedded gfx1151 object for
  static comparison against the Q8 Vulkan large oracle.

Runtime result:

| Shape | Result |
| --- | --- |
| `128x128x32` | failed, `bad=1264`, first bad `row=10 col=0 actual=0.664062 expected=-3.19224` |
| `128x128x4096` | failed, `bad=2392` in the single-shape run |
| suite `128x128` | failed, `bad=2640` |
| suite `128x129` | failed, `bad=3217` |
| suite `129x128` | failed, `bad=2877` |
| suite `1024x512` | failed, `bad=305` |
| suite `4096x512` | failed, `bad=798` |
| suite `4096x513` | failed, `bad=901` |

Static result:

| Metric | AccParkFull8 |
| --- | ---: |
| SGPR | 90 |
| VGPR | 256 |
| LDS | 55296 |
| private segment | 388 |
| VGPR spills | 152 |
| `ds_load_b64` | 64 |
| `v_wmma_f16_16x16x16_f16` | 32 |
| `ds_store_b16` | 194 |
| `ds_load_u16_d16` | 288 |
| `buffer_store_b32` | 192 |

Issue-window result:

- RADV large has one 32-WMMA region with `59` immediate LDS loads before
  `lgkmcnt(51)`;
- AccParkFull8 has two 16-WMMA regions and `0` immediate loads before the
  first final wait despite `lgkmcnt(51)`.

Interpretation:
the `k=32` failure proves the full-vector LDS parking/loadback spelling is not
a valid representation of the real Q8 WMMA accumulator state. This is not only
a full-K lifetime or launch policy problem. Static codegen also makes the
direction non-viable: 55 KiB LDS, `VGPR=256`, private memory, and heavy VGPR
spills.

Decision:
reject AccParkFull8 before timing or catalog transfer. Do not continue the
accumulator-parking family for Q8 large-route parity. The useful next step is
a lower-level accumulator/store-ownership primitive, or a materially different
dataflow that avoids exposing the same full-tile accumulator surface to HIP C++
register allocation.
