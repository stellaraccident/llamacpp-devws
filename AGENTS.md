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
  - `sources/llama.cpp`: use the branch requested for the current development
    push.
  - `sources/hrx-system`: `main` unless the human explicitly approves runtime
    or compiler changes.
- Do not vendor build outputs, models, caches, profiles, ROCm installs, or
  temporary benchmark artifacts into the root repository.
- Install workspace-local developer programs under the untracked `programs/`
  directory. Keep each program's binaries and caches below its own directory.

## Environment

- The root `rocm` symlink should point to the ROCm install selected for the
  current push.
- Prefer workspace-local paths:
  - `ROCM_PATH=$WORKSPACE/rocm`
  - `GGML_HRX_ROCM_PATH=$WORKSPACE/rocm`
  - build trees under `build/`
  - scratch data under `cache/` or `.tmp/`
- Bazel is provided through Bazelisk under `programs/bazel/`. To install the
  pinned Linux x86-64 binary used by this workspace:

  ```bash
  mkdir -p programs/bazel/bin programs/bazel/cache
  curl -fL https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64 \
    -o programs/bazel/bin/bazel
  echo '5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992  programs/bazel/bin/bazel' \
    | sha256sum --check
  chmod +x programs/bazel/bin/bazel
  USE_BAZEL_VERSION=9.1.0 BAZELISK_HOME="$PWD/programs/bazel/cache" \
    programs/bazel/bin/bazel --version
  ```

  The workspace `.envrc` exports the Bazel version and cache location and adds
  `programs/bazel/bin` to `PATH`.
- The workspace uses a direnv-managed `.venv`. Agents may install Python
  tooling dependencies into this venv with `python3 -m pip install ...` when
  needed; do not vendor those packages into the repository.

## Agent Workflow

- Start by reading this file and `README.md`.
- Treat `docs/v2land/` as the active documentation area for this development
  push. Read the relevant files there before making HRX or llama.cpp changes.
- Use `tools/status.py` to inspect source checkout state before editing source
  repositories.
- Keep changes scoped to the relevant source repo or root metadata requested by
  the human.
- When running experiments, keep generated outputs in `cache/` or `.tmp/` and
  preserve only concise conclusions or intentionally retained artifacts.
- Use `tools/sandbox.py` or `tools/launch_agent.py` when running long-lived
  agent sessions that need isolated filesystem access with GPU passthrough.
- There is no beads/topic workflow in this workspace initially. Work directly
  with the human request.
