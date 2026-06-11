---
name: bootstrap-hrx-llama-builds
description: Bootstrap, verify, or reproduce CMake builds for HRX System, Loom, and the ROCm/HRX llama.cpp fork in /home/stella/llamacpp-devws. Use when configuring workspace-level build/ trees, building hrx-system without Bazel, checking Loom compiler binaries, or building llama.cpp CPU, Vulkan, and GGML_HRX baselines against the workspace rocm/ install.
---

# Bootstrap HRX llama.cpp Builds

Use this skill after `init-llamacpp-hrx-workspace` has created the source
checkouts. Keep builds under the workspace `build/` directory and use
`$WORKSPACE/rocm` for ROCm compilers and package discovery.

## Workflow

1. Start with read-only checks:

   ```bash
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action check
   ```

2. If host tools are missing, stop and ask the human to install the reported
   `dnf` packages. Do not switch to Makefiles, disable Vulkan, or skip HRX to
   work around missing tools.

3. Build HRX System with CMake, not Bazel:

   ```bash
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action hrx
   ```

4. Verify Loom tool binaries are present:

   ```bash
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action loom
   ```

5. Check ROCm runtime health before GPU smoke tests or HRX llama.cpp runs:

   ```bash
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action rocm-health
   ```

   If visible GPU ROCr initialization crashes but `ROCR_VISIBLE_DEVICES=` works,
   treat it as a host ROCm/kernel/firmware issue below HRX. The temporary
   CPU-only workaround is:

   ```bash
   export ROCR_VISIBLE_DEVICES=
   ```

6. Build llama.cpp baselines only after confirming the llama.cpp checkout is on
   `hrx-integration`:

   ```bash
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action llama-cpu
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action llama-vulkan
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action llama-hrx
   ```

7. For a full run after prerequisites and branch state are settled:

   ```bash
   python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py --action all
   ```

## Build Trees

The script uses these workspace-local directories:

- `build/hrx-system`: HRX System CMake build tree.
- `build/hrx-install`: installed HRX public distribution for llama.cpp.
- `build/hrx-tests`: optional installed HRX test tree when
  `--install-hrx-tests` is passed.
- `build/llama-cpu`: CPU baseline build.
- `build/llama-vulkan`: Vulkan baseline build.
- `build/llama-hrx`: llama.cpp `GGML_HRX=ON` build.

## Defaults

- Required llama.cpp branch: `hrx-integration`.
- Required hrx-system branch: `main`.
- ROCm path: `$WORKSPACE/rocm`.
- Default GPU target: `gfx1151`.
- CMake generator: `Ninja`.
- Build type: `RelWithDebInfo`.
- ROCm health checks compare visible GPU execution against
  `ROCR_VISIBLE_DEVICES=` CPU-only execution.

Pass `--gfx-targets` to override the target list, using comma or semicolon
separators:

```bash
python skills/bootstrap-hrx-llama-builds/scripts/bootstrap_builds.py \
  --action all --gfx-targets gfx1151
```

## Guardrails

- Ask the human for missing `dnf` packages. Python packages may be installed in
  the active venv if a Python import failure identifies a missing PyPI package.
- Do not change branches from the script.
- Do not run llama.cpp builds from `amd-integration`; that branch lacks
  `GGML_HRX`.
- Do not vendor build products, models, caches, or ROCm files into the root
  repository.
