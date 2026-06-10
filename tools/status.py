#!/usr/bin/env python3
"""Quick llama.cpp HRX workspace status overview."""

import os
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent


def run_capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def show_source_checkouts() -> None:
    sources_dir = WORKSPACE / "sources"
    if not sources_dir.exists():
        print("  (no sources/ directory)")
        return

    for entry in sorted(sources_dir.iterdir()):
        git_dir = entry / ".git"
        if not (entry.is_dir() and (git_dir.exists() or git_dir.is_file())):
            continue
        name = entry.name
        branch = run_capture(["git", "branch", "--show-current"], cwd=entry)
        if not branch:
            branch = "detached"
        remote = run_capture(["git", "config", "--get", "remote.origin.url"], cwd=entry)
        porcelain = run_capture(["git", "status", "--porcelain"], cwd=entry)
        dirty = len(porcelain.splitlines()) if porcelain else 0
        print(f"  {name}: branch={branch} dirty={dirty}")
        if remote:
            print(f"    origin={remote}")


def show_rocm() -> None:
    rocm = WORKSPACE / "rocm"
    if not rocm.exists() and not rocm.is_symlink():
        print("  rocm: missing")
        return
    if rocm.is_symlink():
        print(f"  rocm -> {os.readlink(rocm)}")
    else:
        print(f"  rocm: {rocm}")
    print(f"  ROCM_PATH={os.environ.get('ROCM_PATH', '(not set)')}")
    print(f"  GGML_HRX_ROCM_PATH={os.environ.get('GGML_HRX_ROCM_PATH', '(not set)')}")


def show_dir_size(label: str, path: Path) -> None:
    if not path.exists():
        print(f"  ({label} empty)")
        return
    du = run_capture(["du", "-sh", str(path)])
    print(f"  {du}" if du else f"  ({label} empty)")


def main() -> None:
    print("=== llama.cpp HRX Workspace Status ===")
    print(f"Location: {WORKSPACE}")
    print()

    print("--- Source Checkouts ---")
    show_source_checkouts()
    print()

    print("--- ROCm ---")
    show_rocm()
    print()

    print("--- Build ---")
    show_dir_size("build", WORKSPACE / "build")
    print()

    print("--- Cache ---")
    show_dir_size("cache", WORKSPACE / "cache")
    print()

    print("--- Environment ---")
    print(f"  LLAMACPP_DEVWS={os.environ.get('LLAMACPP_DEVWS', '(not set)')}")
    print(f"  VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '(not active)')}")


if __name__ == "__main__":
    main()
