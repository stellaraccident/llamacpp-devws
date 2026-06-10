#!/usr/bin/env python3
"""Bubblewrap sandbox launcher for llama.cpp HRX agent sessions.

Creates a contained environment where claude code can run with
--dangerously-skip-permissions safely. Linux only (requires bwrap).

Usage:
    sandbox.py [command...]
    sandbox.py                        # interactive bash
    sandbox.py claude --resume        # resume a claude session

Environment variables:
    LLAMACPP_DEVWS        Override workspace root
    LLAMACPP_DEVWS_NET    Set to 0 to block network (default: 1)
    LLAMACPP_DEVWS_GPU    Set to 0 to disable GPU passthrough (default: 1)
    LLAMACPP_DEVWS_PODMAN Set to 0 to disable host podman socket passthrough (default: 1)
    LLAMACPP_DEVWS_EXTRA_RO
                          Colon-separated extra read-only bind mounts
    LLAMACPP_DEVWS_EXTRA_RW
                          Colon-separated extra read-write bind mounts
"""

import os
import stat
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("LLAMACPP_DEVWS", str(SCRIPT_DIR.parent)))
HOME_DIR = Path.home()
SYSTEM_BIN_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def find_nvm_node_bin() -> str | None:
    nvm_versions = HOME_DIR / ".nvm" / "versions" / "node"
    if not nvm_versions.is_dir():
        return None
    versions = sorted(nvm_versions.iterdir(), key=lambda p: p.name)
    if not versions:
        return None
    return str(versions[-1] / "bin")


def build_bwrap_args() -> list[str]:
    allow_net = os.environ.get("LLAMACPP_DEVWS_NET", "1")
    gpu = os.environ.get("LLAMACPP_DEVWS_GPU", "1")

    args: list[str] = ["--die-with-parent"]

    # Read-only system.
    for d in ["/usr", "/lib", "/bin", "/sbin", "/etc"]:
        if Path(d).exists():
            args.extend(["--ro-bind", d, d])
    if Path("/lib64").is_dir():
        args.extend(["--ro-bind", "/lib64", "/lib64"])

    # Read-only shared projects, when present on the host.
    if Path("/srv/vm-shared").is_dir():
        args.extend(["--ro-bind", "/srv/vm-shared", "/srv/vm-shared"])

    # DNS (systemd-resolved).
    resolve_dir = Path("/run/systemd/resolve")
    if resolve_dir.is_dir():
        args.extend(["--ro-bind", str(resolve_dir), str(resolve_dir)])

    # Proc and dev.
    args.extend(["--proc", "/proc", "--dev", "/dev"])
    args.extend(["--dev-bind", "/dev/pts", "/dev/pts"])
    args.extend(["--dev-bind", "/dev/ptmx", "/dev/ptmx"])

    # Tmp: workspace-local.
    tmp_dir = WORKSPACE / ".tmp"
    if tmp_dir.is_dir():
        args.extend(["--bind", str(tmp_dir), "/tmp"])
    else:
        args.extend(["--tmpfs", "/tmp"])

    # Home directory: minimal tmpfs + selective binds.
    args.extend(["--tmpfs", str(HOME_DIR)])

    # Claude Code config, codex, and cache (persisted).
    for name in [".claude", ".codex", ".cache"]:
        p = HOME_DIR / name
        if p.is_dir():
            args.extend(["--bind", str(p), str(p)])
    claude_json = HOME_DIR / ".claude.json"
    if claude_json.is_file():
        args.extend(["--bind", str(claude_json), str(claude_json)])

    # Local binaries (claude CLI, pip tools).
    local_dir = HOME_DIR / ".local"
    if local_dir.is_dir():
        args.extend(["--ro-bind", str(local_dir), str(local_dir)])

    # Node.js / nvm.
    for name in [".nvm", ".npm"]:
        p = HOME_DIR / name
        if p.is_dir():
            rw = name == ".npm"
            args.extend(["--bind" if rw else "--ro-bind", str(p), str(p)])

    # Git config (read-only).
    gitconfig = HOME_DIR / ".gitconfig"
    if gitconfig.is_file():
        args.extend(["--ro-bind", str(gitconfig), str(gitconfig)])

    # Block credentials.
    for cred_dir in [".ssh", ".gnupg", ".aws"]:
        args.extend(["--tmpfs", str(HOME_DIR / cred_dir)])

    # The workspace: full read-write. This must come after the HOME tmpfs
    # because the workspace normally lives under /home/stella.
    args.extend(["--bind", str(WORKSPACE), str(WORKSPACE)])

    # Working directory.
    args.extend(["--chdir", str(WORKSPACE)])

    # Clean environment.
    args.append("--clearenv")
    env = {
        "HOME": str(HOME_DIR),
        "USER": os.environ.get("USER", "stella"),
        "TERM": os.environ.get("TERM", "xterm-256color"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "SHELL": "/bin/bash",
        "COLUMNS": os.environ.get("COLUMNS", "120"),
        "LINES": os.environ.get("LINES", "40"),
        "XDG_CACHE_HOME": str(HOME_DIR / ".cache"),
        "XDG_CONFIG_HOME": str(HOME_DIR / ".config"),
        "VIRTUAL_ENV": f"{WORKSPACE}/.venv",
        "TMPDIR": f"{WORKSPACE}/.tmp",
        "TEMP": f"{WORKSPACE}/.tmp",
        "TMP": f"{WORKSPACE}/.tmp",
        "LLAMACPP_DEVWS": str(WORKSPACE),
        "LLAMACPP_DEVWS_CACHE_DIR": f"{WORKSPACE}/cache",
        "LLAMACPP_DEVWS_SANDBOX": "1",
        "ROCM_PATH": f"{WORKSPACE}/rocm",
        "GGML_HRX_ROCM_PATH": f"{WORKSPACE}/rocm",
        "LD_LIBRARY_PATH": (
            f"{WORKSPACE}/rocm/lib:"
            f"{WORKSPACE}/rocm/lib64:"
            f"{WORKSPACE}/rocm/lib/rocm_sysdeps/lib"
        ),
    }
    for k, v in env.items():
        args.extend(["--setenv", k, v])

    # PATH.
    nvm_node = find_nvm_node_bin()
    path_parts = [
        f"{WORKSPACE}/tools",
        f"{WORKSPACE}/.venv/bin",
        f"{WORKSPACE}/rocm/bin",
        f"{WORKSPACE}/rocm/lib/llvm/bin",
        f"{HOME_DIR}/.local/bin",
    ]
    if nvm_node:
        path_parts.append(nvm_node)
    path_parts.extend(["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    args.extend(["--setenv", "PATH", ":".join(path_parts)])

    # GPU passthrough.
    if gpu == "1":
        if Path("/dev/kfd").exists():
            args.extend(["--dev-bind", "/dev/kfd", "/dev/kfd"])
        if Path("/dev/dri").is_dir():
            args.extend(["--dev-bind", "/dev/dri", "/dev/dri"])
        args.extend(["--ro-bind", "/sys", "/sys"])
        rocm_link = WORKSPACE / "rocm"
        if rocm_link.exists():
            rocm_target = rocm_link.resolve()
            if rocm_target.is_dir() and not str(rocm_target).startswith(str(WORKSPACE)):
                args.extend(["--dir", str(rocm_target.parent)])
                args.extend(["--ro-bind", str(rocm_target), str(rocm_target)])

    # Network.
    if allow_net == "0":
        args.append("--unshare-net")

    # Extra mounts.
    for envvar, flag in [
        ("LLAMACPP_DEVWS_EXTRA_RO", "--ro-bind"),
        ("LLAMACPP_DEVWS_EXTRA_RW", "--bind"),
    ]:
        extra = os.environ.get(envvar, "")
        for mount in extra.split(":"):
            if mount:
                args.extend([flag, mount, mount])

    return args


def start_gh_proxy() -> tuple[subprocess.Popen | None, Path | None]:
    """Start the gh proxy server if gh CLI is available on the host.

    Returns (process, socket_path) or (None, None) if gh is unavailable.
    """
    if not shutil.which("gh"):
        return None, None

    import tempfile
    sock_dir = Path(tempfile.mkdtemp(prefix="gh-proxy-"))
    sock_path = sock_dir / "gh.sock"
    server_script = SCRIPT_DIR / "gh_proxy_server.py"

    proc = subprocess.Popen(
        [sys.executable, str(server_script), str(sock_path)],
        stdout=subprocess.DEVNULL,
    )

    # Wait for socket to appear.
    import time
    for _ in range(20):
        if sock_path.exists():
            return proc, sock_path
        time.sleep(0.1)

    # Server didn't start.
    proc.kill()
    return None, None


def _is_unix_socket(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except OSError:
        return False


def find_existing_podman_socket() -> Path | None:
    """Find a host podman service socket, if one is already running."""
    candidates: list[Path] = []

    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        candidates.append(Path(xdg_runtime_dir) / "podman" / "podman.sock")

    candidates.append(Path("/run") / "user" / str(os.getuid()) / "podman" / "podman.sock")

    for candidate in candidates:
        if _is_unix_socket(candidate):
            return candidate
    return None


def start_podman_proxy() -> tuple[subprocess.Popen | None, Path | None, bool]:
    """Expose host podman to the sandbox via its remote API socket.

    Rootless podman cannot reliably start inside the bubblewrap sandbox because
    the nested user namespace only maps the current uid. Use the host podman
    service instead and make tools/podman connect to it in remote mode.

    Returns (process, socket_path, owns_socket_dir). If process is None and
    socket_path is set, an existing host socket is being used.
    """
    if os.environ.get("LLAMACPP_DEVWS_PODMAN", "1") == "0":
        return None, None, False
    podman_bin = shutil.which("podman", path=SYSTEM_BIN_PATH)
    if not podman_bin:
        return None, None, False

    existing_socket = find_existing_podman_socket()
    if existing_socket is not None:
        return None, existing_socket, False

    import tempfile
    import time

    sock_dir = Path(tempfile.mkdtemp(prefix="podman-proxy-"))
    sock_path = sock_dir / "podman.sock"
    proc = subprocess.Popen(
        [podman_bin, "system", "service", "--time=0", f"unix://{sock_path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(50):
        if _is_unix_socket(sock_path):
            return proc, sock_path, True
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    proc.kill()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    sock_path.unlink(missing_ok=True)
    sock_dir.rmdir()
    return None, None, False


def main() -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        print(
            "Error: bwrap (bubblewrap) not found. Install with:\n"
            "  sudo dnf install bubblewrap",
            file=sys.stderr,
        )
        sys.exit(1)

    # Start gh proxy server (runs on host, sandbox connects via socket).
    gh_proc, gh_socket = start_gh_proxy()
    podman_proc, podman_socket, podman_owns_socket_dir = start_podman_proxy()

    bwrap_args = build_bwrap_args()

    # If gh proxy is running, pass socket into sandbox and bind the tools/gh wrapper.
    if gh_socket is not None:
        # Bind the socket directory into the sandbox.
        sock_dir = str(gh_socket.parent)
        bwrap_args.extend(["--bind", sock_dir, sock_dir])
        bwrap_args.extend(["--setenv", "GH_PROXY_SOCKET", str(gh_socket)])

    if podman_socket is not None:
        sock_dir = str(podman_socket.parent)
        podman_url = f"unix://{podman_socket}"
        bwrap_args.extend(["--bind", sock_dir, sock_dir])
        bwrap_args.extend(["--setenv", "PODMAN_PROXY_SOCKET", str(podman_socket)])
        bwrap_args.extend(["--setenv", "CONTAINER_HOST", podman_url])
        bwrap_args.extend(["--setenv", "PODMAN_HOST", podman_url])

    if len(sys.argv) > 1:
        cmd = sys.argv[1:]
    else:
        cmd = ["bash", "--rcfile", str(SCRIPT_DIR / "sandbox-bashrc.sh")]
        print("Starting interactive shell...")

    full_cmd = [bwrap] + bwrap_args + ["--"] + cmd

    try:
        result = subprocess.run(full_cmd)
        sys.exit(result.returncode)
    finally:
        # Clean up gh proxy.
        if gh_proc is not None:
            gh_proc.terminate()
            try:
                gh_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                gh_proc.kill()
            if gh_socket and gh_socket.parent.exists():
                gh_socket.unlink(missing_ok=True)
                gh_socket.parent.rmdir()
        if podman_proc is not None:
            podman_proc.terminate()
            try:
                podman_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                podman_proc.kill()
        if podman_socket and podman_owns_socket_dir and podman_socket.parent.exists():
            podman_socket.unlink(missing_ok=True)
            podman_socket.parent.rmdir()


if __name__ == "__main__":
    main()
