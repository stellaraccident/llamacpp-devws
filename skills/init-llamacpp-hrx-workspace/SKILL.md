---
name: init-llamacpp-hrx-workspace
description: Bootstrap, verify, or repair a llama.cpp HRX development workspace. Use when setting up /home/stella/llamacpp-devws, creating the ROCm nightly symlink, cloning ROCm/hrx-system and ROCm/llama.cpp into sources/, or checking that the workspace source layout is ready for agents.
---

# Init llama.cpp HRX Workspace

Use this skill to bootstrap or verify the shared llama.cpp HRX workspace.
Prefer the bundled script over hand-running clone commands.

## Workflow

1. Run from the workspace root unless the user provides another path:

   ```bash
   python skills/init-llamacpp-hrx-workspace/scripts/bootstrap_workspace.py
   ```

2. For a preview:

   ```bash
   python skills/init-llamacpp-hrx-workspace/scripts/bootstrap_workspace.py --dry-run
   ```

3. If the skill is being used from outside the workspace, pass the workspace:

   ```bash
   python /path/to/skill/scripts/bootstrap_workspace.py --workspace /home/stella/llamacpp-devws
   ```

4. Do not run `direnv allow` from inside agent or sandbox sessions. After
   bootstrap, tell the human to run it from their normal external shell if they
   want automatic shell activation:

   ```bash
   cd /home/stella/llamacpp-devws
   direnv allow
   ```

5. Verify setup with:

   ```bash
   tools/status.py
   ```

## What The Script Does

- Create `sources/`, `build/`, `cache/`, `docs/`, `skills/`, `tools/`, and
  `.tmp/`.
- Create or repair `rocm -> /home/stella/rocm/rocm-7.14.0a20260610`.
- Clone full-history repositories:
  - `https://github.com/ROCm/hrx-system.git` to `sources/hrx-system`, branch
    `main`.
  - `https://github.com/ROCm/llama.cpp.git` to `sources/llama.cpp`, branch
    `amd-integration`.
- Refuse dirty or mismatched existing source checkouts and print remediation.

## Defaults

- ROCm nightly install:
  `/home/stella/rocm/rocm-7.14.0a20260610`
- ROCm nightly source page:
  `https://rocm.nightlies.amd.com/tarball/`
- Source checkout root:
  `sources/`
- Build outputs:
  `build/`

## Guardrails

- Do not overwrite source checkouts that have local changes.
- Do not replace a non-symlink `rocm` path.
- Do not use shallow clones unless the user explicitly asks to change the
  bootstrap policy.
