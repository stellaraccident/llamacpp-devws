---
name: bootstrap-therock-rocm-source
description: Use when bootstrapping ROCm from TheRock in this workspace, including checkout into sources/TheRock, TheRock's default fetch_sources.py source sync, and a debug-symbol core-runtime build under build/.
---

# Bootstrap TheRock ROCm Source

## Overview

Use this skill when switching from the workspace ROCm tarball to a source-built TheRock tree for debugging ROCr/core-runtime behavior. Keep TheRock source under `sources/TheRock`, build trees under `build/`, and ask the human to install missing host packages with `dnf` instead of trying to work around them.

## Defaults

- Source checkout: `sources/TheRock`
- Repository: `https://github.com/ROCm/TheRock.git`
- Branch: `main`
- Fetch command: `python build_tools/fetch_sources.py` with no stage, source-set, or depth arguments unless the human explicitly asks otherwise
- Core-runtime build tree: `build/therock-core-runtime`
- GPU target: `gfx1151`
- Dist GPU target: `gfx1151`
- Build type: `RelWithDebInfo`
- Core-runtime CMake feature selection: `THEROCK_ENABLE_ALL=OFF`, `THEROCK_ENABLE_CORE_RUNTIME=ON`
- Debug symbols: keep split debug and minimal debug disabled for local debugging unless the human asks for package-style split debug artifacts

## Host Tools

Before configuring or building, check host tools:

```bash
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py check-tools
```

If tools are missing, ask the human to run the reported `sudo dnf install -y ...` command. Do not install `dnf` packages from Codex. Python requirements may be installed into the active venv:

The source build needs the normal CMake/Ninja compiler stack plus TheRock's native build helpers such as `patch`, autotools, `bison`, `flex`, `texinfo`/`makeinfo`, `gfortran`, and `xxd`.

```bash
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py install-requirements
```

## Checkout And Fetch

Create or inspect the top-level checkout:

```bash
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py checkout
```

Fetch TheRock sources using its default source workflow:

```bash
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py fetch-sources
```

Do not slice the fetch by stage or source set for this workflow. The default fetch pulls the expected submodules, DVC payloads, and applies TheRock's patches. It can reset submodule state, so inspect local submodule work before rerunning it.

## Configure And Build Core Runtime

Configure a debug-symbol core-runtime build:

```bash
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py configure-core-runtime
```

Build the `core-runtime` artifact:

```bash
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py build-core-runtime
```

The outer build target is `artifact-core-runtime`. After the initial build exists, use TheRock's subproject targets such as `ROCR-Runtime+dist` for tighter runtime rebuild/debug loops.

Do not pass `THEROCK_RESET_FEATURES=ON` in the same configure invocation that enables a specific feature. The reset path forces feature cache entries back to their defaults, so `THEROCK_ENABLE_ALL=OFF` plus reset will leave `THEROCK_ENABLE_CORE_RUNTIME` disabled unless it is set again in a later configure.

## Verify

Verify source and build outputs:

```bash
git -C sources/TheRock submodule status
python skills/bootstrap-therock-rocm-source/scripts/bootstrap_therock.py verify-core-runtime
```

The verification looks for `libhsa-runtime64.so` and `rocminfo` under the TheRock build tree, and reports whether debug sections are visible when `readelf` is available.
