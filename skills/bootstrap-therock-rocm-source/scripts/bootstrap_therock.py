#!/usr/bin/env python3
"""Bootstrap TheRock checkout, source fetch, and core-runtime debug builds."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/ROCm/TheRock.git"
DEFAULT_BRANCH = "main"
DEFAULT_SOURCE_DIR = "sources/TheRock"
DEFAULT_BUILD_DIR = "build/therock-core-runtime"
DEFAULT_AMDGPU_TARGETS = "gfx1151"

REQUIRED_HOST_TOOLS = {
    "git": "git",
    "cmake": "cmake",
    "ninja": "ninja-build",
    "gcc": "gcc",
    "g++": "gcc-c++",
    "gfortran": "gcc-gfortran",
    "pkg-config": "pkgconf-pkg-config",
    "patchelf": "patchelf",
    "patch": "patch",
    "make": "make",
    "autoconf": "autoconf",
    "automake": "automake",
    "libtool": "libtool",
    "libtoolize": "libtool",
    "m4": "m4",
    "bison": "bison",
    "flex": "flex",
    "makeinfo": "texinfo",
    "xxd": "xxd",
}

OPTIONAL_HOST_TOOLS = {
    "ccache": "ccache",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    location = f" [{cwd}]" if cwd else ""
    print(f"++{location}$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def resolve_workspace(path: str) -> Path:
    return Path(path).expanduser().resolve()


def resolve_under_workspace(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (workspace / path).resolve()


def check_tools(strict: bool = True) -> bool:
    missing: list[str] = []
    for tool in REQUIRED_HOST_TOOLS:
        if shutil.which(tool) is None:
            missing.append(tool)

    print("Required host tools:")
    for tool, package in REQUIRED_HOST_TOOLS.items():
        status = shutil.which(tool) or "MISSING"
        print(f"  {tool}: {status} (dnf: {package})")

    print("Optional host tools:")
    for tool, package in OPTIONAL_HOST_TOOLS.items():
        status = shutil.which(tool) or "MISSING"
        print(f"  {tool}: {status} (dnf: {package})")

    if missing:
        packages = " ".join(dict.fromkeys(REQUIRED_HOST_TOOLS[tool] for tool in missing))
        print()
        print(f"Missing required host tools: {', '.join(missing)}")
        print(f"Ask the human to run: sudo dnf install -y {packages}")
        if strict:
            return False
    return True


def checkout(source_dir: Path, repo_url: str, branch: str) -> None:
    if source_dir.exists():
        if not (source_dir / ".git").exists():
            raise SystemExit(f"{source_dir} exists but is not a git checkout")
        current_branch = capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=source_dir)
        head = capture(["git", "rev-parse", "HEAD"], cwd=source_dir)
        print(f"TheRock checkout exists: {source_dir}")
        print(f"  branch: {current_branch}")
        print(f"  HEAD:   {head}")
        if current_branch != branch:
            raise SystemExit(
                f"Existing checkout is on {current_branch}, expected {branch}. "
                "Change branches explicitly before continuing."
            )
        return

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--branch", branch, "--single-branch", repo_url, str(source_dir)])


def install_requirements(source_dir: Path) -> None:
    requirements = source_dir / "requirements.txt"
    if not requirements.exists():
        raise SystemExit(f"Missing requirements file: {requirements}")
    run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def fetch_sources(source_dir: Path) -> None:
    fetch_script = source_dir / "build_tools" / "fetch_sources.py"
    if not fetch_script.exists():
        raise SystemExit(f"Missing fetch script: {fetch_script}")
    run([sys.executable, str(fetch_script)], cwd=source_dir)


def configure_core_runtime(args: argparse.Namespace, source_dir: Path, build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    dist_amdgpu_targets = args.dist_amdgpu_targets or args.amdgpu_targets
    cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DROCR-Runtime_BUILD_TYPE={args.build_type}",
        f"-Drocminfo_BUILD_TYPE={args.build_type}",
        f"-Drocprofiler-register_BUILD_TYPE={args.build_type}",
        f"-DTHEROCK_AMDGPU_TARGETS={args.amdgpu_targets}",
        f"-DTHEROCK_DIST_AMDGPU_TARGETS={dist_amdgpu_targets}",
        "-DTHEROCK_ENABLE_ALL=OFF",
        "-DTHEROCK_ENABLE_CORE_RUNTIME=ON",
        "-DTHEROCK_SPLIT_DEBUG_INFO=OFF",
        "-DTHEROCK_MINIMAL_DEBUG_INFO=OFF",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if not args.no_ccache and shutil.which("ccache") is not None:
        cmd.extend(
            [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]
        )
    run(cmd)


def build_core_runtime(args: argparse.Namespace, build_dir: Path) -> None:
    target = args.build_target
    cmd = ["cmake", "--build", str(build_dir), "--target", target]
    if args.jobs:
        cmd.extend(["--parallel", str(args.jobs)])
    run(cmd)


def has_debug_info(path: Path) -> str:
    readelf = shutil.which("readelf")
    if readelf is None:
        return "unknown: readelf not found"
    result = subprocess.run(
        [readelf, "-S", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if ".debug_info" in result.stdout:
        return "yes"
    return "no"


def verify_core_runtime(build_dir: Path) -> None:
    if not build_dir.exists():
        raise SystemExit(f"Build directory does not exist: {build_dir}")

    libs = sorted(build_dir.rglob("libhsa-runtime64.so*"))
    infos = sorted(build_dir.rglob("rocminfo"))

    print("ROCR runtime libraries:")
    if not libs:
        print("  MISSING: libhsa-runtime64.so*")
    for path in libs:
        if path.is_file() or path.is_symlink():
            print(f"  {path} (debug_info: {has_debug_info(path)})")

    print("rocminfo binaries:")
    if not infos:
        print("  MISSING: rocminfo")
    for path in infos:
        if path.is_file() and os.access(path, os.X_OK):
            print(f"  {path} (debug_info: {has_debug_info(path)})")

    if not libs or not infos:
        raise SystemExit("Core-runtime outputs are incomplete")


def expand_actions(actions: list[str]) -> list[str]:
    if not actions:
        return ["check-tools"]
    expanded: list[str] = []
    for action in actions:
        if action == "all":
            expanded.extend(
                [
                    "check-tools",
                    "checkout",
                    "install-requirements",
                    "fetch-sources",
                    "configure-core-runtime",
                    "build-core-runtime",
                    "verify-core-runtime",
                ]
            )
        else:
            expanded.append(action)
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "actions",
        nargs="*",
        choices=[
            "check-tools",
            "checkout",
            "install-requirements",
            "fetch-sources",
            "configure-core-runtime",
            "build-core-runtime",
            "verify-core-runtime",
            "all",
        ],
        help="Actions to run. Defaults to check-tools.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR)
    parser.add_argument("--amdgpu-targets", default=DEFAULT_AMDGPU_TARGETS)
    parser.add_argument("--dist-amdgpu-targets")
    parser.add_argument("--build-type", default="RelWithDebInfo")
    parser.add_argument("--build-target", default="artifact-core-runtime")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--no-ccache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = resolve_workspace(args.workspace)
    source_dir = resolve_under_workspace(workspace, args.source_dir)
    build_dir = resolve_under_workspace(workspace, args.build_dir)

    for action in expand_actions(args.actions):
        if action == "check-tools":
            if not check_tools(strict=True):
                return 2
        elif action == "checkout":
            checkout(source_dir, args.repo_url, args.branch)
        elif action == "install-requirements":
            install_requirements(source_dir)
        elif action == "fetch-sources":
            fetch_sources(source_dir)
        elif action == "configure-core-runtime":
            configure_core_runtime(args, source_dir, build_dir)
        elif action == "build-core-runtime":
            build_core_runtime(args, build_dir)
        elif action == "verify-core-runtime":
            verify_core_runtime(build_dir)
        else:
            raise AssertionError(action)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
