#!/usr/bin/env python3
"""Bootstrap HRX System, Loom, and llama.cpp builds for this workspace."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_GFX_TARGETS = "auto"
FALLBACK_GFX_TARGETS = "gfx1151"
EXPECTED_LLAMA_BRANCH = "hrx-v2"
EXPECTED_HRX_BRANCH = "main"

LOOM_BINARIES = [
    "loom-compile",
    "loom-link",
    "loom-check",
    "iree-test-loom",
    "iree-benchmark-loom",
    "source_info",
    "compile_text",
    "link_modules",
]

LOOM_BUILD_TARGETS = [
    "loom/src/loom/tools/loom-compile/loom-compile",
    "loom/src/loom/tools/loom-link/loom-link",
    "loom/src/loom/tools/loom-check/loom-check",
    "loom/src/loom/tools/iree-test-loom/iree-test-loom",
    "loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom",
    "loom/binding/c/example/source_info",
    "loom/binding/c/example/compile_text",
    "loom/binding/c/example/link_modules",
]


def quote_cmd(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run(args: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool = False) -> None:
    print(f"+ cd {cwd}")
    print(f"+ {quote_cmd(args)}")
    if dry_run:
        return
    subprocess.run(args, cwd=str(cwd), env=env, check=True)


def run_probe(args: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(args, 124, exc.stdout or "", exc.stderr or "timed out")


def capture(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def git_branch(repo: Path) -> str:
    return capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)


def split_targets(raw: str) -> list[str]:
    targets = []
    for part in raw.replace(",", ";").split(";"):
        part = part.strip()
        if part:
            targets.append(part)
    return targets


def cmake_target_string(targets: list[str]) -> str:
    return ";".join(targets)


def detect_gfx_targets(workspace: Path, env: dict[str, str]) -> list[str]:
    rocminfo = workspace / "rocm" / "bin" / "rocminfo"
    if not rocminfo.exists():
        raise SystemExit(
            f"cannot auto-detect --gfx-targets because rocminfo is missing: {rocminfo}\n"
            f"Pass --gfx-targets {FALLBACK_GFX_TARGETS} or another explicit target."
        )

    result = run_probe([str(rocminfo)], cwd=workspace, env=env, timeout=30)
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        detail = f"\nrocminfo output:\n{output}" if output else ""
        raise SystemExit(
            "cannot auto-detect --gfx-targets because rocminfo failed. "
            f"Pass --gfx-targets {FALLBACK_GFX_TARGETS} or another explicit target.{detail}"
        )

    targets = []
    for target in re.findall(r"\bName:\s+(gfx[0-9a-zA-Z_]+)\b", result.stdout):
        if target not in targets:
            targets.append(target)
    if not targets:
        raise SystemExit(
            "cannot auto-detect --gfx-targets because rocminfo reported no GPU target. "
            f"Pass --gfx-targets {FALLBACK_GFX_TARGETS} or another explicit target."
        )

    print(f"Auto-detected AMDGPU targets: {cmake_target_string(targets)}")
    return targets


def resolve_gfx_targets(workspace: Path, raw: str, env: dict[str, str]) -> list[str]:
    if raw.strip().lower() == "auto":
        return detect_gfx_targets(workspace, env)
    return split_targets(raw)


def find_under(root: Path, name: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob(name) if path.is_file())


def find_executable(root: Path, name: str) -> Path | None:
    for path in find_under(root, name):
        if os.access(path, os.X_OK):
            return path
    return None


def check_file(path: Path, missing: list[str], label: str | None = None) -> None:
    if not path.exists():
        missing.append(label or str(path))


def check_command(name: str, missing: list[str]) -> None:
    if shutil.which(name) is None:
        missing.append(name)


def print_dnf_hint(missing: list[str]) -> None:
    packages: set[str] = set()
    for item in missing:
        if item == "gcc":
            packages.add("gcc")
        elif item == "g++":
            packages.add("gcc-c++")
        elif item in {"Scrt1.o", "crti.o", "crtn.o", "libc.so"}:
            packages.add("glibc-devel")
            packages.add("glibc-headers")
        elif item == "libstdc++ headers":
            packages.add("libstdc++-devel")
        elif item == "kernel headers":
            packages.add("kernel-headers")
        elif item == "libgcc_s.so":
            packages.add("libgcc")
        elif item == "ninja":
            packages.add("ninja-build")
        elif item == "glslc":
            packages.add("glslc")
        elif item == "pkg-config":
            packages.add("pkgconf-pkg-config")
        elif item == "vulkan/vulkan.h":
            packages.add("vulkan-headers")
        elif item == "libvulkan.so":
            packages.add("vulkan-loader-devel")
    if packages:
        print("\nAsk the human to run:")
        print(f"  sudo dnf install -y {' '.join(sorted(packages))}")


def require_prereqs(workspace: Path) -> None:
    rocm = workspace / "rocm"
    hrx_src = workspace / "sources" / "hrx-system"
    llama_src = workspace / "sources" / "llama.cpp"
    missing_host: list[str] = []
    missing_rocm: list[str] = []
    errors: list[str] = []

    for cmd in ["git", "cmake", "ninja", "python3", "pkg-config", "gcc", "g++"]:
        check_command(cmd, missing_host)
    check_command("glslc", missing_host)

    check_file(Path("/usr/include/vulkan/vulkan.h"), missing_host, "vulkan/vulkan.h")
    if not (Path("/usr/lib64/libvulkan.so").exists() or Path("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists()):
        missing_host.append("libvulkan.so")
    for crt in ["Scrt1.o", "crti.o", "crtn.o"]:
        if not list(Path("/usr").glob(f"lib*/**/{crt}")):
            missing_host.append(crt)
    if not (Path("/usr/lib64/libc.so").exists() or Path("/usr/lib/x86_64-linux-gnu/libc.so").exists()):
        missing_host.append("libc.so")
    if not Path("/usr/include/c++").exists():
        missing_host.append("libstdc++ headers")
    if not Path("/usr/include/linux").exists():
        missing_host.append("kernel headers")

    for rel in [
        "lib/llvm/bin/clang",
        "lib/llvm/bin/clang++",
        "lib/llvm/bin/llvm-ar",
        "lib/llvm/bin/llvm-ranlib",
        "lib/llvm/bin/lld",
        "bin/amdclang++",
        "bin/hipcc",
        "lib/cmake",
    ]:
        check_file(rocm / rel, missing_rocm)

    if not hrx_src.exists():
        errors.append(f"missing HRX source checkout: {hrx_src}")
    if not llama_src.exists():
        errors.append(f"missing llama.cpp source checkout: {llama_src}")

    if hrx_src.exists():
        branch = git_branch(hrx_src)
        if branch != EXPECTED_HRX_BRANCH:
            errors.append(f"hrx-system branch is {branch!r}; expected {EXPECTED_HRX_BRANCH!r}")

    if llama_src.exists():
        branch = git_branch(llama_src)
        if branch != EXPECTED_LLAMA_BRANCH:
            errors.append(f"llama.cpp branch is {branch!r}; expected {EXPECTED_LLAMA_BRANCH!r}")
        if not (llama_src / "ggml" / "src" / "ggml-hrx" / "CMakeLists.txt").exists():
            errors.append("llama.cpp checkout does not contain ggml/src/ggml-hrx/CMakeLists.txt")
        if not (llama_src / "ggml" / "src" / "ggml-hrx2" / "CMakeLists.txt").exists():
            errors.append("llama.cpp checkout does not contain ggml/src/ggml-hrx2/CMakeLists.txt")

    if missing_host:
        print("Missing host tools/files:")
        for item in missing_host:
            print(f"  - {item}")
        print_dnf_hint(missing_host)
        raise SystemExit(2)

    if missing_rocm:
        print("Missing ROCm workspace files:")
        for item in missing_rocm:
            print(f"  - {item}")
        raise SystemExit(2)

    if errors:
        print("Configuration errors:")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(2)

    print("Prerequisite check passed.")


def base_env(workspace: Path) -> dict[str, str]:
    rocm = workspace / "rocm"
    env = os.environ.copy()
    env["LLAMACPP_DEVWS"] = str(workspace)
    env["ROCM_PATH"] = str(rocm)
    env["GGML_HRX_ROCM_PATH"] = str(rocm)
    env["HRX_SYSTEM_SOURCE"] = str(workspace / "sources" / "hrx-system")
    env["LLAMA_CPP_SOURCE"] = str(workspace / "sources" / "llama.cpp")
    env["PATH"] = (
        f"{rocm / 'bin'}:{rocm / 'lib' / 'llvm' / 'bin'}:"
        f"{env.get('PATH', '')}"
    )
    rocm_libs = [rocm / "lib", rocm / "lib64", rocm / "lib" / "rocm_sysdeps" / "lib"]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in rocm_libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def configure_hrx(workspace: Path, gfx_targets: list[str], env: dict[str, str], dry_run: bool) -> None:
    rocm = workspace / "rocm"
    build = workspace / "build" / "hrx-system"
    src = workspace / "sources" / "hrx-system"
    clang = rocm / "lib" / "llvm" / "bin" / "clang"
    clangxx = rocm / "lib" / "llvm" / "bin" / "clang++"
    args = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DIREE_ROCM_PATH={rocm}",
        "-DIREE_ROCM_DEPENDENCY_MODE=package",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        f"-DCMAKE_C_COMPILER={clang}",
        f"-DCMAKE_CXX_COMPILER={clangxx}",
        f"-DCMAKE_ASM_COMPILER={clang}",
        f"-DCMAKE_AR={rocm / 'lib' / 'llvm' / 'bin' / 'llvm-ar'}",
        f"-DCMAKE_RANLIB={rocm / 'lib' / 'llvm' / 'bin' / 'llvm-ranlib'}",
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DIREE_HAL_DRIVER_AMDGPU=ON",
        "-DIREE_HAL_DRIVER_VULKAN=ON",
        "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
        "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
        "-DIREE_HAL_DRIVER_NULL=ON",
        f"-DIREE_HAL_AMDGPU_TARGETS={cmake_target_string(gfx_targets)}",
        "-DLOOM_BUILD=ON",
    ]
    run(args, cwd=workspace, env=env, dry_run=dry_run)


def build_hrx(
    workspace: Path,
    env: dict[str, str],
    jobs: int,
    dry_run: bool,
    configure_only: bool,
    install_tests: bool,
) -> None:
    build = workspace / "build" / "hrx-system"
    if configure_only:
        return
    run(["cmake", "--build", str(build), "--parallel", str(jobs)], cwd=workspace, env=env, dry_run=dry_run)
    run(
        ["cmake", "--install", str(build), "--prefix", str(workspace / "build" / "hrx-install"), "--component", "HrxPublicDist"],
        cwd=workspace,
        env=env,
        dry_run=dry_run,
    )
    if install_tests:
        run(
            ["cmake", "--install", str(build), "--prefix", str(workspace / "build" / "hrx-tests"), "--component", "HrxTestsDist"],
            cwd=workspace,
            env=env,
            dry_run=dry_run,
        )


def build_loom_targets(workspace: Path, env: dict[str, str], jobs: int, dry_run: bool, configure_only: bool) -> None:
    build = workspace / "build" / "hrx-system"
    if configure_only:
        return
    run(
        ["cmake", "--build", str(build), "--parallel", str(jobs), "--target", *LOOM_BUILD_TARGETS],
        cwd=workspace,
        env=env,
        dry_run=dry_run,
    )


def verify_loom(workspace: Path) -> None:
    build = workspace / "build" / "hrx-system"
    missing = []
    found = []
    for name in LOOM_BINARIES:
        path = find_executable(build, name)
        if path is None:
            missing.append(name)
        else:
            found.append(path)
    if missing:
        print("Missing Loom binaries:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(3)
    print("Loom binaries:")
    for path in found:
        print(f"  - {path}")


def describe_returncode(returncode: int) -> str:
    if returncode == 0:
        return "OK"
    if returncode == -11:
        return "segmentation fault"
    if returncode < 0:
        return f"failed by signal {-returncode}"
    if returncode == 124:
        return "timed out"
    if returncode == 139:
        return "segmentation fault"
    return f"failed with exit code {returncode}"


def print_probe(label: str, result: subprocess.CompletedProcess[str], *, max_lines: int = 12) -> None:
    print(f"{label}: {describe_returncode(result.returncode)}")
    output = (result.stdout or "") + (result.stderr or "")
    lines = [line for line in output.splitlines() if line.strip()]
    for line in lines[:max_lines]:
        print(f"  {line}")
    if len(lines) > max_lines:
        print(f"  ... {len(lines) - max_lines} more lines")


def rocm_health(workspace: Path, env: dict[str, str]) -> None:
    rocm = workspace / "rocm"
    rocminfo = rocm / "bin" / "rocminfo"
    hrx_info = workspace / "build" / "hrx-install" / "bin" / "hrx-info"

    if not rocminfo.exists():
        raise SystemExit(f"missing ROCm rocminfo: {rocminfo}")

    visible = run_probe([str(rocminfo)], cwd=workspace, env=env)
    hidden_env = env.copy()
    hidden_env["ROCR_VISIBLE_DEVICES"] = ""
    hidden = run_probe([str(rocminfo)], cwd=workspace, env=hidden_env)

    print_probe("rocminfo with visible GPU", visible)
    print_probe("rocminfo with ROCR_VISIBLE_DEVICES=", hidden)

    hrx_visible = None
    hrx_hidden = None
    if hrx_info.exists():
        hrx_env = env.copy()
        hrx_env["LD_LIBRARY_PATH"] = (
            f"{workspace / 'build' / 'hrx-install' / 'lib'}:"
            f"{hrx_env.get('LD_LIBRARY_PATH', '')}"
        )
        hrx_visible = run_probe([str(hrx_info)], cwd=workspace, env=hrx_env)
        hrx_hidden_env = hrx_env.copy()
        hrx_hidden_env["ROCR_VISIBLE_DEVICES"] = ""
        hrx_hidden = run_probe([str(hrx_info)], cwd=workspace, env=hrx_hidden_env)
        print_probe("hrx-info with visible GPU", hrx_visible)
        print_probe("hrx-info with ROCR_VISIBLE_DEVICES=", hrx_hidden)
    else:
        print(f"hrx-info not built yet: {hrx_info}")

    visible_failed = visible.returncode != 0 or (hrx_visible is not None and hrx_visible.returncode != 0)
    hidden_ok = hidden.returncode == 0 and (hrx_hidden is None or hrx_hidden.returncode == 0)
    if visible_failed and hidden_ok:
        print("\nDiagnosis:")
        print("  Visible GPU ROCr initialization is failing below HRX.")
        print("  CPU-only ROCr/HRX paths are healthy when the GPU is masked.")
        print("\nTemporary CPU-only workaround:")
        print("  export ROCR_VISIBLE_DEVICES=")
        print("\nAsk the human to update and reboot the host kernel/firmware stack:")
        print("  sudo dnf upgrade -y linux-firmware amd-gpu-firmware kernel kernel-core kernel-modules kernel-modules-core kernel-modules-extra")


def configure_llama(
    workspace: Path,
    name: str,
    extra_args: list[str],
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    rocm = workspace / "rocm"
    src = workspace / "sources" / "llama.cpp"
    build = workspace / "build" / f"llama-{name}"
    base_args = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_WEBUI=OFF",
        "-DGGML_CPU=ON",
        "-DGGML_NATIVE=ON",
        f"-DCMAKE_PREFIX_PATH={workspace / 'build' / 'hrx-install'};{rocm};{rocm / 'lib' / 'cmake'}",
    ]
    run(base_args + extra_args, cwd=workspace, env=env, dry_run=dry_run)
    return build


def build_llama(workspace: Path, build: Path, env: dict[str, str], jobs: int, dry_run: bool, configure_only: bool) -> None:
    if configure_only:
        return
    run(["cmake", "--build", str(build), "--parallel", str(jobs)], cwd=workspace, env=env, dry_run=dry_run)


def verify_llama_build(build: Path, backend_hint: str) -> None:
    cache = build / "CMakeCache.txt"
    if not cache.exists():
        raise SystemExit(f"missing CMake cache: {cache}")
    cache_text = cache.read_text(errors="replace")
    if backend_hint not in cache_text:
        raise SystemExit(f"{backend_hint} not present in {cache}")
    print(f"Verified {backend_hint} in {cache}")


def llama_cpu_args() -> list[str]:
    return ["-DGGML_HRX=OFF", "-DGGML_HRX2=OFF", "-DGGML_VULKAN=OFF", "-DGGML_HIP=OFF"]


def llama_vulkan_args() -> list[str]:
    return ["-DGGML_HRX=OFF", "-DGGML_HRX2=OFF", "-DGGML_VULKAN=ON", "-DGGML_HIP=OFF"]


def llama_hrx_args(workspace: Path, gfx_targets: list[str]) -> list[str]:
    rocm = workspace / "rocm"
    return [
        "-DGGML_HRX=ON",
        "-DGGML_HRX2=OFF",
        "-DGGML_VULKAN=OFF",
        "-DGGML_HIP=OFF",
        f"-DGGML_HRX_ROCM_PATH={rocm}",
        f"-DGGML_HRX_CLANGXX={rocm / 'lib' / 'llvm' / 'bin' / 'clang++'}",
        f"-DGGML_HRX_AMDGPU_TARGETS={cmake_target_string(gfx_targets)}",
        f"-DCMAKE_C_COMPILER={rocm / 'lib' / 'llvm' / 'bin' / 'clang'}",
        f"-DCMAKE_CXX_COMPILER={rocm / 'lib' / 'llvm' / 'bin' / 'clang++'}",
    ]


def llama_hrx2_args(workspace: Path, dry_run: bool) -> list[str]:
    loom_link = workspace / "build" / "hrx-system" / "loom" / "src" / "loom" / "tools" / "loom-link" / "loom-link"
    if not dry_run and not (loom_link.exists() and os.access(loom_link, os.X_OK)):
        raise SystemExit(f"missing built loom-link; run --action loom before llama-hrx2: {loom_link}")
    return [
        "-DGGML_HRX=OFF",
        "-DGGML_HRX2=ON",
        "-DGGML_VULKAN=OFF",
        "-DGGML_HIP=OFF",
        f"-DGGML_HRX2_LOOM_LINK_EXECUTABLE={loom_link}",
    ]


def expand_actions(actions: list[str]) -> list[str]:
    expanded: list[str] = []
    for action in actions:
        if action == "all":
            expanded.extend(["check", "hrx", "loom", "llama-cpu", "llama-vulkan", "llama-hrx", "llama-hrx2"])
        else:
            expanded.append(action)
    result: list[str] = []
    for action in expanded:
        if action not in result:
            result.append(action)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace root")
    parser.add_argument(
        "--action",
        action="append",
        choices=["check", "rocm-health", "hrx", "loom", "llama-cpu", "llama-vulkan", "llama-hrx", "llama-hrx2", "all"],
        default=None,
        help="Action to run. May be passed more than once. Defaults to check.",
    )
    parser.add_argument(
        "--gfx-targets",
        default=DEFAULT_GFX_TARGETS,
        help="Comma or semicolon separated AMDGPU targets, or 'auto' to detect from rocminfo",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--install-hrx-tests", action="store_true", help="Also install the HRX HrxTestsDist test tree")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    actions = expand_actions(args.action or ["check"])
    env = base_env(workspace)

    needs_gfx_targets = any(action in actions for action in ("hrx", "loom", "llama-hrx"))
    gfx_targets = resolve_gfx_targets(workspace, args.gfx_targets, env) if needs_gfx_targets else []
    if needs_gfx_targets and not gfx_targets:
        raise SystemExit("--gfx-targets must contain at least one target")

    if "check" in actions or len(actions) > 1 or actions[0] != "check":
        require_prereqs(workspace)

    if "hrx" in actions:
        configure_hrx(workspace, gfx_targets, env, args.dry_run)
        build_hrx(workspace, env, args.jobs, args.dry_run, args.configure_only, args.install_hrx_tests)

    if "loom" in actions:
        configure_hrx(workspace, gfx_targets, env, args.dry_run)
        build_loom_targets(workspace, env, args.jobs, args.dry_run, args.configure_only)
        if not args.configure_only and not args.dry_run:
            verify_loom(workspace)

    if "rocm-health" in actions:
        if args.dry_run:
            print("Would run ROCm runtime health checks.")
        else:
            rocm_health(workspace, env)

    if "llama-cpu" in actions:
        build = configure_llama(workspace, "cpu", llama_cpu_args(), env, args.dry_run)
        build_llama(workspace, build, env, args.jobs, args.dry_run, args.configure_only)
        if not args.configure_only and not args.dry_run:
            verify_llama_build(build, "GGML_CPU:BOOL=ON")

    if "llama-vulkan" in actions:
        build = configure_llama(workspace, "vulkan", llama_vulkan_args(), env, args.dry_run)
        build_llama(workspace, build, env, args.jobs, args.dry_run, args.configure_only)
        if not args.configure_only and not args.dry_run:
            verify_llama_build(build, "GGML_VULKAN:BOOL=ON")

    if "llama-hrx" in actions:
        build = configure_llama(workspace, "hrx", llama_hrx_args(workspace, gfx_targets), env, args.dry_run)
        build_llama(workspace, build, env, args.jobs, args.dry_run, args.configure_only)
        if not args.configure_only and not args.dry_run:
            verify_llama_build(build, "GGML_HRX:BOOL=ON")

    if "llama-hrx2" in actions:
        build = configure_llama(workspace, "hrx2", llama_hrx2_args(workspace, args.dry_run), env, args.dry_run)
        build_llama(workspace, build, env, args.jobs, args.dry_run, args.configure_only)
        if not args.configure_only and not args.dry_run:
            verify_llama_build(build, "GGML_HRX2:BOOL=ON")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
