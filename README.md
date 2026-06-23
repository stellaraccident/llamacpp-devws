# llama.cpp HRX Development Workspace

This workspace is for shared llama.cpp development with HRX support on ROCm.
The root repository tracks workspace metadata: docs, tools, skills, and agent
instructions. Upstream source projects live under `sources/` as independent git
repositories.

## Layout

```text
sources/
  llama.cpp/    ROCm llama.cpp checkout for active development
  hrx-system/   HRX runtime checkout, branch main
build/          local build trees and install trees
cache/          local caches and profiling scratch
docs/           workspace documentation
  v2land/       active notes for the current development push
  archive/      preserved legacy notes and retired skills
skills/         workspace-specific Codex skills
tools/          sandbox, agent launcher, status, and gh proxy helpers
rocm            symlink to a local ROCm nightly install
```

## ROCm Nightly

Install ROCm from the official nightly tarball page:
`https://rocm.nightlies.amd.com/tarball/`.

Pick a local install location outside this repository. For example:

```bash
export ROCM_INSTALL_ROOT="$HOME/opt/rocm"
export ROCM_INSTALL="/srv/vm-shared/shared/rocm-7.14.0a20260527"
mkdir -p "$ROCM_INSTALL"
```

Download the Linux tarball that matches your GPU target, such as
`therock-dist-linux-gfx1151-7.14.0a20260527.tar.gz` for `gfx1151`, then unpack
it into the chosen install directory:

```bash
cd "$ROCM_INSTALL_ROOT"
curl -LO "https://rocm.nightlies.amd.com/tarball/therock-dist-linux-<gfx-target>-7.14.0a20260527.tar.gz"
tar -xzf "therock-dist-linux-<gfx-target>-7.14.0a20260527.tar.gz" -C "$ROCM_INSTALL"
test -x "$ROCM_INSTALL/bin/amdclang++"
```

Use the directory containing `bin/amdclang++` as the ROCm path for bootstrap.

## Bootstrap

From the workspace root:

```bash
python skills/init-llamacpp-hrx-workspace/scripts/bootstrap_workspace.py --rocm "$ROCM_INSTALL"
```

The bootstrap creates required directories, creates or verifies the `rocm`
symlink, and clones:

- `https://github.com/ROCm/hrx-system.git` on `main`
- `https://github.com/ROCm/llama.cpp.git` on the active development branch

It refuses to overwrite dirty or mismatched existing checkouts.

## Environment

This workspace is set up for `direnv`. `direnv allow` is a local shell trust
action; run it from your normal external shell after bootstrap, not from inside
agent or sandbox sessions:

```bash
direnv allow
```

The root `.envrc` exports:

```bash
LLAMACPP_DEVWS="$PWD"
LLAMACPP_DEVWS_CACHE_DIR="$PWD/cache"
ROCM_PATH="$PWD/rocm"
GGML_HRX_ROCM_PATH="$PWD/rocm"
HRX_SYSTEM_SOURCE="$PWD/sources/hrx-system"
LLAMA_CPP_SOURCE="$PWD/sources/llama.cpp"
```

It also prepends workspace tools and ROCm compiler/runtime paths to `PATH` and
`LD_LIBRARY_PATH`. On first entry, direnv creates `.venv` with `python3 -m venv`
and then sources it.

Agents should leave `direnv allow` to the external shell and verify setup with
`tools/status.py`. The sandbox tool sets the same core variables automatically.
