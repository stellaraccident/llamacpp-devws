# llama.cpp HRX Workspace

This workspace is for shared development of llama.cpp with HRX support.

## Repository Rules

- The workspace root repository tracks metadata only: docs, tools, skills, and
  agent instructions.
- Do not commit, push, or change branches in the root repository unless the
  human explicitly asks.
- Code changes belong in the independent source repositories under `sources/`:
  - `sources/llama.cpp/` for llama.cpp work
  - `sources/hrx-system/` for HRX runtime work
- Keep source checkout branches explicit. Default branches are:
  - `sources/llama.cpp`: `hrx-v2` for the active HRX2 work
  - `sources/hrx-system`: `main`
- Do not vendor build outputs, models, caches, profiles, or ROCm installs into
  the root repository.

## Environment

- The root `rocm` symlink should point to:
  `/srv/vm-shared/shared/rocm-7.14.0a20260527`
- Assume `rocm/` is a full ROCm installation from the official nightly tarball
  page: `https://rocm.nightlies.amd.com/tarball/`.
- Prefer workspace-local paths:
  - `ROCM_PATH=$WORKSPACE/rocm`
  - `GGML_HRX_ROCM_PATH=$WORKSPACE/rocm`
  - build trees under `build/`
  - scratch data under `cache/` or `.tmp/`
- The workspace uses a direnv-managed `.venv`. Agents may install Python
  tooling dependencies into this venv with `python3 -m pip install ...` when
  needed; do not vendor those packages into the repository. `PyYAML` is
  installed there for skill validation.

## Agent Workflow

- Start by reading this file and `README.md`.
- For kernel optimization work, read `docs/spike/kernel-skill/SKILL.md` and
  then the specific reference it points to.
- In goal loops, treat build/runtime/environmental problems as part of the
  task. No one else is coming to clean up broken env, stale build state,
  missing package assumptions, cache poisoning, or runtime interop regressions.
  Use HRX1 as the known-good runtime interop prior before inventing a new HRX2
  flow.
- When isolating backend behavior, run focused backend op unit tests before
  moving to full integration/model tests. They exist to catch route, kernel, and
  compiler failures at the smallest useful boundary.
- For kernel performance gaps larger than a small local regression, do not
  guess new schedules from the current HRX2/Loom source alone. First mine and
  mechanically compare known priors from HRX1, Vulkan, CUDA/HIP, and other
  llama.cpp backends. Run or disassemble those priors when possible, inspect
  the emitted ISA/SPIR-V/HSACO and Loom compile reports, and write down the
  concrete schedule deltas before implementing a new kernel.
- Adjacent schedule probes are still allowed even without strong direct
  evidence. Treat them as bracketed pivots around a documented schedule family:
  name the axis, bounds, and expected signal; run them in a kernel/backend-op
  sweep with correctness and route evidence; keep speculative variants out of
  the production catalog and only move a proven variant to full llama.cpp
  integration. This kind of probing is encouraged when it brackets the active
  hypothesis; the failure mode is blind one-off integration, not exploration.
- Use `tools/status.py` to inspect source checkout state.
- Use `tools/sandbox.py` or `tools/launch_agent.py` when running long-lived
  agent sessions that need isolated filesystem access with GPU passthrough.
- There is no beads/topic workflow in this workspace initially. Work directly
  with the human request and keep changes scoped to the relevant source repo.

## Compaction Survival Rules

- After a compact or fresh agent handoff, assume the most recent goal is still
  active unless the human says otherwise. Re-read this file, run
  `tools/status.py`, inspect the relevant source diffs, and continue from the
  latest documented checkpoint instead of restarting from first principles.
- For throughput gaps, demand device-side evidence and apples-to-apples
  comparisons. Prefer backend-op timing, provider traces, compile reports, ISA,
  and same-run Vulkan/HRX2 baskets. Avoid cross-tool timing conclusions unless
  the measurement methodology has been pinned down.
- When a kernel is far from target, look for structural mistakes before local
  tuning: missing route, CPU fallback, bad pack/layout contract, wrong dot
  signedness, wrong lane ownership, insufficient staging/vectorization, missing
  fusion, or runtime backpressure. Use known-good HRX1/Vulkan/CUDA/HIP priors
  and disassembly to find the broad schedule that should work.
- Loom is WYSIWYG. Do not expect the compiler to recover unstated vectorization,
  load width, tiling, staging, dot form, unroll policy, bounds strategy, or
  target constraints. Spell those decisions in the source or route metadata and
  preserve tunable choices in JSON/catalog data rather than hiding them in
  one-off heuristics.
- Treat temporary diagnostics as disposable evidence. Save useful patches or
  artifacts under `cache/`/`.tmp/`, document the conclusion, and revert any
  diagnostic route that would shadow the production route or bias benchmarks.

## Current HRX2 Phase 2a Guidance

- Active goal-loop objective: lift HRX2 throughput in `sources/llama.cpp`
  against the same-machine Vulkan baseline, committing coherent llama.cpp
  changes as needed, until HRX2 is within about 2x Vulkan or the remaining gap
  is reduced to a short, evidence-backed blocker list needing Loom/runtime
  author changes.
- Phase 2a work is in `sources/llama.cpp`. Commit coherent llama.cpp changes as
  needed. Keep `sources/hrx-system` on `main` with only the existing packaging
  patches unless the human explicitly approves a non-trivial Loom/runtime API or
  compiler change.
- Vulkan is the same-machine throughput baseline. Use
  `tools/hrx2_phase2a_benchmark.py` against
  `shared/models/llamacpp-hrx2-basket-v1` for basket comparisons, and keep
  `docs/loom/llamacpp-hrx2-phase2a-report.md` current with accepted and
  rejected evidence.
- Always run focused `test-backend-ops` rows for the touched route before
  model-level `llama-bench` or `llama-cli` tests. A failing op gate is a kernel,
  route, or compiler problem; do not hide it inside integration tests.
- For large prefill gaps, look for boulders: quantized prompt matmul schedule,
  attention, copy/contiguous traffic, and hero fusions. Avoid spending time on
  tiny runtime knobs until provider traces prove they dominate.
- When HRX2 is still many multiples behind Vulkan, assume a structural issue
  until proven otherwise: unsupported route, CPU fallback, bad graph
  materialization, missing HRX1 runtime interop behavior, weak packed-matmul
  schedule, missing attention/fusion path, or measurement error. Do not spend a
  goal loop on pebble-sized knob tuning while prefill is still orders of
  magnitude off.
- For prefill boulders, use prior-driven implementation, not local guessing.
  Start from known-good HRX1, Vulkan, CUDA/HIP, or other backend kernels; run or
  disassemble them when feasible; extract tile shape, lane ownership, vector
  load width, packed layout, dot primitive, staging, barriers, reduction, and
  writeback policy; then spell that schedule explicitly in Loom. If the Loom
  candidate still loses badly, compare emitted ISA/resource reports before
  attributing the gap to compiler quality.
- The current Q4_K prompt-matmul checkpoint is in
  `docs/loom/llamacpp-hrx2-phase2a-report.md`. The accepted prompt path is the
  non-x4 Q8_1 cols4 route. Q4_K x Q8_1 dot spelling has been corrected to
  `vector.dot4i<u8s8>`, but packed Q8_1/x4 MMQ is still diagnostic-only and
  must not be enabled by default.
- Current fresh reduced baseline artifact after the accepted F16 attention
  route and the accepted Phi V-cache `CONT_SET_ROWS` fusion:
  `cache/hrx2/phase2a/current-reduced-after-cont-setrows-20260615-235800/`.
  Decode is roughly 0.33x-0.39x Vulkan with zero CPU compute fallback. Prefill
  remains the severe gap: roughly 0.016x-0.05x Vulkan. The large final
  `hrx_stream_synchronize_end` spans real device work, so do not treat runtime
  batching as the primary prefill explanation unless a new trace proves it.
  Provider traces and Vulkan timing buckets put the main prefill boulders at
  Q4_K prompt matmul, Q5/Q6 prompt matmuls, and attention-chain/fusion traffic.
  Treat these as the next evidence-driven targets; only return to copy/`CONT`
  work if fresh route traces show a remaining unfused structural cliff.
- Latest fresh reduced rerun artifact:
  `cache/hrx2/phase2a/current-reduced-20260616-000512/`. It reconfirmed zero
  CPU compute fallback, decode around one third of Vulkan, and prefill around
  0.016x-0.05x Vulkan. Top prefill routes remain Q4_K prompt matmul, F16
  attention matmul/softmax chain, and Q5/Q6 prompt matmuls. Use this artifact
  rather than older smoke-only runs when resuming Phase 2a triage.
- Q4_K Q8_1/x4 MMQ remains blocked as of the current checkpoint. A direct
  global-load diagnostic passed focused rows but was slow; staged integer A or
  B payloads through Loom workgroup memory failed correctness. Treat this as a
  Loom/low-level staging limitation or schedule-spelling problem, not a generic
  Q4_K/Q8_1 route or dot-form issue.

## Current Goal-Loop Resume Checkpoint

- Active source repo: `sources/llama.cpp` on `hrx-v2`.
- Latest accepted llama.cpp commit at the current checkpoint:
  `bf1c90d8d hrx2: fuse cont into v-cache set rows`.
- `sources/llama.cpp` was clean at this checkpoint. If a future agent sees
  dirty files, inspect them as new work, not as a required continuation of the
  Q4_K BM64/BN8 diagnostic or the accepted F16 route.
- Do not assume a passing CSV means the candidate provider ran: the HRX2
  dispatcher can fall back after a Loom provider compile failure. Always
  inspect `GGML_HRX2_TRACE_JSONL` route histograms and `provider_unavailable`
  events before accepting an op gate.
- `sources/hrx-system` should remain on `main` with only the known local
  packaging patches unless the human approves a non-trivial compiler/runtime
  change. Current Q4_K work should stay in llama.cpp and local scratch files
  unless the requested output is a standalone Loom reproducer for the author.
- The accepted `CONT` route is `cont_f32_n128_vec4_wg256`, Loom export
  `hrx2_cont_f32_n128_vec4`, guarded by `GGML_HRX2_DISABLE_CONT_VEC4=1`.
  Focused `CONT` op gates passed and same-binary A/B improved Phi prompt
  shapes by roughly 3-6%. Treat it as committed and do not redo that work
  unless a regression points there.
- Accepted WYSIWYG cleanup: Q4_K x Q8_1 Loom dot products now use
  `vector.dot4i<u8s8>` instead of `s8s8`, matching unsigned Q4 codes times
  signed Q8_1 activations. Focused non-x4 control gate:
  `cache/hrx2/phase2a/q4k-u8s8-control-20260615-222811/`. Current post-revert
  focused Q4_K control gate:
  `cache/hrx2/phase2a/q4k-control-after-mmq64x8-revert-20260615-230629/`.
- Completed Q4_K x4 direct diagnostic: a temporary direct-cols4 route consumed
  packed Q8_1 x4 RHS without MMQ/LDS tiling and passed all eight model-derived
  Q4_K rows. Artifact:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-wg256-diag-20260615-222302/`.
  Saved patch:
  `cache/hrx2/phase2a/q4k-x4-direct-cols4-passing-diagnostic-20260615-222414/passing-diagnostic.patch`.
  Interpretation: packed x4 Q8_1 quantization/layout is likely sound; do not
  restore this direct diagnostic as a production route because it is slow and
  shadows the MMQ route under the same x4 gate.
- Completed Q4_K BM64/BN8 diagnostic: flattened `%kb` loop spelling caused a
  Loom `source-to-low` internal, while the old-style outer `%q4_block_iter`
  plus inner unrolled `%group` topology compiled. The compiled BN8 route
  selected in focused traces, but any integer payload staged through Loom
  workgroup memory failed correctness with finite `ERR ~= 1.0`. A+B direct
  global payload loads passed focused rows but was much slower than the
  accepted non-x4 Q8_1 fallback on Llama 3.2 3B Q4_K_M p64/n0
  (`22.39 tok/s` vs `79.58 tok/s`). Diagnostic patches:
  `cache/hrx2/phase2a/q4k-mmq64x8-diagnostic-patches-20260615-230451/`.
- Current Q4_K conclusion: do not continue local knob sweeps on the same
  BM64/BN8 Loom source. The useful next Q4_K actions are either a minimal
  standalone integer-LDS reproducer for the Loom author, or a different
  low-level/staging spelling that can preserve correctness while matching the
  HRX1/Vulkan tiled packed-MMQ schedule.
- Current attention-chain investigation: the benchmark basket is currently run
  with `--flash-attn 0`, so traces expose the unfused KQ matmul, SOFT_MAX, and
  KQV matmul chain. If resuming after a compact, first check whether a
  `fa1-prefill-probe-*` artifact exists under `cache/hrx2/phase2a/` and inspect
  its HRX2/Vulkan routes before starting another long benchmark. If Vulkan gets
  a large `--flash-attn 1` lift while HRX2 falls back or stays on the unfused
  chain, treat HRX2 `GGML_OP_FLASH_ATTN_EXT` support or an equivalent attention
  fusion as the next Phase 2a boulder, using HRX1 flash-attention kernels and
  Vulkan flash-attention shaders as priors.
- Current F16 batched-attention checkpoint: the accepted llama.cpp Loom route
  `hrx2_mul_mat_f16_f32_batched_attention_cols8_wg256` is committed and based
  on the HRX1 cols8 prior. Focused backend-op gates passed for c512 attention
  rows and the route is deliberately tightened to `cols=512` after p64/narrow
  prompt smoke regressed. P512 HRX2-only smoke improved roughly +0.9% on Phi
  and +2.1% on Llama 3.2 3B, so it is a valid small lift but not the bulk
  prefill boulder. Useful artifacts:
  `cache/hrx2/phase2a/f16-cols8-c512-tight-opgate-20260615-232119/`,
  `cache/hrx2/phase2a/f16-cols8-tight-p64-smoke-20260615-232132/`, and
  `cache/hrx2/phase2a/f16-cols8-tight-p512-smoke-20260615-232157/`.
- Post-commit F16 reduced comparison artifact:
  `cache/hrx2/phase2a/current-reduced-after-f16-cols8-20260615-232708/`.
  Llama 3.2 3B p512 was 86.342 tok/s versus Vulkan 4756.767; Phi p512 was
  69.944 tok/s versus Vulkan 4269.034. Decode remains around one third of
  Vulkan. This confirms the F16 route did not break dispatch or fallback, but
  the prefill gap is still a structural kernel/fusion problem.
- Current accepted `CONT -> SET_ROWS` checkpoint: llama.cpp commit
  `bf1c90d8d hrx2: fuse cont into v-cache set rows` adds
  `cont_set_rows_f32_f16_n128_wg256`, dispatch trace op `CONT_SET_ROWS`, and
  rollback `GGML_HRX2_DISABLE_CONT_SET_ROWS_FUSION=1`. Same-binary A/B on
  Phi-4-mini Q4_K_M showed +3.02% on `prefill-p64n0` and +1.76% on
  `prefill-p512n0`, reducing dispatches from 1830 to 1734 over three reps.
  Final op gate:
  `cache/hrx2/phase2a/cont-setrows-opgate-final-20260615-235916/`.
  Reduced comparison:
  `cache/hrx2/phase2a/current-reduced-after-cont-setrows-20260615-235800/`.
  The route intentionally declines `set_rows.ne1 > 1048576` after a large
  default-KV-cache CLI smoke exposed a Loom config range failure; do not remove
  that guard without widening/testing the Loom source config ranges.
- Current Q4_K recheck after hrx-system updates:
  `cache/hrx2/phase2a/q4k-x4-current-recheck-20260615-232919/`.
  The opt-in x4 MMQ route still selects but fails focused c64 rows with NaNs.
  The non-x4 Q8_1 prompt A/B artifact
  `cache/hrx2/phase2a/q4k-q8-nonx4-ab-20260615-232954/` was flat or
  regressive except for a tiny Llama p512 lift, so it is not a defaultable
  boulder. Do not enable either route broadly without new focused evidence.
- Next Phase 2a boulders: Q4_K/Q5_K/Q6_K prompt matmul quality and
  attention-chain/fusion candidates. For Q4_K, use either a standalone
  Loom/low-level integer-LDS reproducer or a different staging spelling that
  can match HRX1/Vulkan tiled packed-MMQ correctness. For attention-chain work,
  start from route traces after `CONT_SET_ROWS`, because Phi top blockers have
  shifted back to F16 attention and quantized prompt matmul.

## Documentation Seed

The `docs/spike/` tree is a curated carryover from the Pyre workspace:
HRX/llama.cpp handoff notes, integration logs, profiling and runtime overhead
notes, Vulkan to HIP kernel strategy, the preserved Pyre kernel optimization
spike log, and the HRX kernel optimization skill.
