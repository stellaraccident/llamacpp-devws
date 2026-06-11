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
  - `sources/llama.cpp`: `hrx-integration`
  - `sources/hrx-system`: `main`
- Do not vendor build outputs, models, caches, profiles, or ROCm installs into
  the root repository.

## Environment

- The root `rocm` symlink should point to:
  `/home/stella/rocm/rocm-7.14.0a20260610`
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
- Use `tools/status.py` to inspect source checkout state.
- Use `tools/sandbox.py` or `tools/launch_agent.py` when running long-lived
  agent sessions that need isolated filesystem access with GPU passthrough.
- There is no beads/topic workflow in this workspace initially. Work directly
  with the human request and keep changes scoped to the relevant source repo.

## Documentation Seed

The `docs/spike/` tree is a curated carryover from the Pyre workspace:
HRX/llama.cpp handoff notes, integration logs, profiling and runtime overhead
notes, Vulkan to HIP kernel strategy, the preserved Pyre kernel optimization
spike log, and the HRX kernel optimization skill.
