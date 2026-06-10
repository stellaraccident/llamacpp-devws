#!/usr/bin/env python3
"""Bootstrap a llama.cpp HRX development workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROCM = Path("/home/stella/rocm/rocm-7.14.0a20260610")
ROCM_NIGHTLY_URL = "https://rocm.nightlies.amd.com/tarball/"


@dataclass(frozen=True)
class SourceRepo:
    name: str
    url: str
    branch: str
    path: str


SOURCE_REPOS = (
    SourceRepo(
        name="hrx-system",
        url="https://github.com/ROCm/hrx-system.git",
        branch="main",
        path="sources/hrx-system",
    ),
    SourceRepo(
        name="llama.cpp",
        url="https://github.com/ROCm/llama.cpp.git",
        branch="amd-integration",
        path="sources/llama.cpp",
    ),
)

REQUIRED_DIRS = (
    "sources",
    "build",
    "cache",
    "docs",
    "skills",
    "tools",
    ".tmp",
)


class BootstrapError(RuntimeError):
    pass


def infer_workspace() -> Path:
    env_workspace = os.environ.get("LLAMACPP_DEVWS")
    if env_workspace:
        return Path(env_workspace).expanduser().resolve()
    # scripts/bootstrap_workspace.py -> skill -> skills -> workspace
    return Path(__file__).resolve().parents[3]


def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    cwd_text = f" (cwd={cwd})" if cwd else ""
    print(f"  $ {' '.join(cmd)}{cwd_text}")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise BootstrapError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def capture(cmd: list[str], *, cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BootstrapError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def command_ok(cmd: list[str], *, cwd: Path) -> bool:
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def canonical_remote(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:") :]
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url[len("ssh://git@github.com/") :]
    if url.endswith(".git"):
        url = url[:-4]
    return url


def ensure_directories(workspace: Path, *, dry_run: bool) -> None:
    print("Creating directories...")
    for rel in REQUIRED_DIRS:
        path = workspace / rel
        print(f"  dir: {path.relative_to(workspace)}/")
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)


def ensure_rocm_link(workspace: Path, rocm_target: Path, *, dry_run: bool) -> None:
    print("Checking ROCm symlink...")
    rocm_target = rocm_target.expanduser().resolve()
    rocm_link = workspace / "rocm"

    if not rocm_target.is_dir():
        raise BootstrapError(
            f"ROCm target does not exist: {rocm_target}\n"
            f"Install or unpack ROCm from {ROCM_NIGHTLY_URL}, or pass --rocm <path>."
        )

    if rocm_link.exists() or rocm_link.is_symlink():
        if rocm_link.is_symlink():
            current = rocm_link.resolve()
            if current == rocm_target:
                print(f"  rocm -> {os.readlink(rocm_link)}")
                return
            print(f"  repair: rocm currently points to {current}")
            if not dry_run:
                rocm_link.unlink()
        else:
            raise BootstrapError(
                f"{rocm_link} exists and is not a symlink. Move it aside before bootstrap."
            )

    print(f"  symlink: rocm -> {rocm_target}")
    if not dry_run:
        rocm_link.symlink_to(rocm_target, target_is_directory=True)


def check_clean_checkout(path: Path) -> None:
    dirty = capture(["git", "status", "--short"], cwd=path)
    if dirty:
        raise BootstrapError(
            f"{path} has local changes. Commit, stash, or move them before bootstrap."
        )


def ensure_existing_repo(repo: SourceRepo, dest: Path, *, dry_run: bool) -> None:
    if not command_ok(["git", "rev-parse", "--is-inside-work-tree"], cwd=dest):
        raise BootstrapError(f"{dest} exists but is not a git checkout.")

    remote = capture(["git", "config", "--get", "remote.origin.url"], cwd=dest)
    if canonical_remote(remote) != canonical_remote(repo.url):
        raise BootstrapError(
            f"{dest} origin mismatch.\n"
            f"  expected: {repo.url}\n"
            f"  actual:   {remote}\n"
            f"Move the directory aside or fix origin before bootstrap."
        )

    check_clean_checkout(dest)
    run(["git", "fetch", "origin", repo.branch], cwd=dest, dry_run=dry_run)

    local_branch = command_ok(
        ["git", "rev-parse", "--verify", f"refs/heads/{repo.branch}"], cwd=dest
    )
    if local_branch:
        run(["git", "checkout", repo.branch], cwd=dest, dry_run=dry_run)
        run(["git", "pull", "--ff-only", "origin", repo.branch], cwd=dest, dry_run=dry_run)
    else:
        run(
            ["git", "checkout", "-B", repo.branch, f"origin/{repo.branch}"],
            cwd=dest,
            dry_run=dry_run,
        )


def ensure_source_repo(workspace: Path, repo: SourceRepo, *, dry_run: bool) -> None:
    dest = workspace / repo.path
    print(f"Checking {repo.name}...")
    if dest.exists():
        ensure_existing_repo(repo, dest, dry_run=dry_run)
        return

    run(
        ["git", "clone", "--branch", repo.branch, repo.url, str(dest)],
        cwd=workspace,
        dry_run=dry_run,
    )


def print_next_steps(workspace: Path) -> None:
    print()
    print("Done. Next steps:")
    print("  tools/status.py")
    print()
    print("For interactive shell activation, run this from your normal external shell:")
    print(f"  cd {workspace}")
    print("  direnv allow")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=infer_workspace(),
        help="Workspace root to bootstrap",
    )
    parser.add_argument(
        "--rocm",
        type=Path,
        default=DEFAULT_ROCM,
        help="ROCm install path for the workspace rocm symlink",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without changing the filesystem or git checkouts",
    )
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    print(f"Bootstrapping llama.cpp HRX workspace at {workspace}")
    print(f"ROCm nightly source page: {ROCM_NIGHTLY_URL}")
    print()

    try:
        ensure_directories(workspace, dry_run=args.dry_run)
        print()
        ensure_rocm_link(workspace, args.rocm, dry_run=args.dry_run)
        print()
        for repo in SOURCE_REPOS:
            ensure_source_repo(workspace, repo, dry_run=args.dry_run)
            print()
        print_next_steps(workspace)
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
