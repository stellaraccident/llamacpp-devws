#!/usr/bin/env python3
"""gh_proxy_server.py — Host-side proxy that executes whitelisted gh commands.

Runs OUTSIDE the sandbox. Listens on a Unix socket and executes `gh` commands
that match an allowlist. Returns stdout/stderr/exitcode to the caller.

The socket path is passed to the sandbox, which has a thin `gh` wrapper that
connects to it.

Usage:
    # Normally launched by sandbox.py automatically
    python tools/gh_proxy_server.py /tmp/gh-proxy-XXXX.sock

Protocol (newline-delimited JSON over Unix socket):
    Request:  {"args": ["pr", "list", "--repo", "ROCm/llama.cpp"]}
    Response: {"returncode": 0, "stdout": "...", "stderr": "..."}
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

# Allowed gh subcommand prefixes. Only these pass through.
# Each entry is a tuple of (subcommand, sub-subcommand) that is allowed.
ALLOWED_COMMANDS: set[tuple[str, ...]] = {
    # Read-only PR operations
    ("pr", "list"),
    ("pr", "view"),
    ("pr", "diff"),
    ("pr", "checks"),
    ("pr", "status"),
    # Read-only issue operations
    ("issue", "list"),
    ("issue", "view"),
    ("issue", "status"),
    # Read-only repo operations
    ("repo", "view"),
    ("repo", "list"),
    # Search (read-only)
    ("search", "issues"),
    ("search", "prs"),
    ("search", "repos"),
    ("search", "code"),
    # API (read-only — we block mutating methods below)
    ("api",),
    # Release listing
    ("release", "list"),
    ("release", "view"),
    # Run viewing
    ("run", "list"),
    ("run", "view"),
    ("run", "download"),
    # General
    ("auth", "status"),
}

# For `gh api`, block mutating HTTP methods
API_MUTATING_FLAGS = {"--method PUT", "--method POST", "--method DELETE", "--method PATCH",
                      "-X PUT", "-X POST", "-X DELETE", "-X PATCH"}


def is_command_allowed(args: list[str]) -> tuple[bool, str]:
    """Check if a gh command is in the allowlist.

    Returns (allowed, reason).
    """
    if not args:
        return False, "empty command"

    # Match against allowed command prefixes
    for prefix in ALLOWED_COMMANDS:
        if len(args) >= len(prefix) and tuple(args[:len(prefix)]) == prefix:
            # Special case: block mutating API calls
            if prefix == ("api",):
                args_str = " ".join(args)
                for flag in API_MUTATING_FLAGS:
                    if flag in args_str:
                        return False, f"mutating API method not allowed: {flag}"
            return True, "ok"

    return False, f"command not in allowlist: gh {' '.join(args[:2])}"


def handle_client(conn: socket.socket) -> None:
    """Handle a single client connection."""
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        if not data:
            return

        request = json.loads(data.decode().strip())
        args = request.get("args", [])

        allowed, reason = is_command_allowed(args)
        if not allowed:
            response = {
                "returncode": 1,
                "stdout": "",
                "stderr": f"gh-proxy: blocked — {reason}\n"
                          f"Allowed: pr list/view/diff, issue list/view, "
                          f"repo view, search, api (GET only), run list/view\n",
            }
        else:
            try:
                result = subprocess.run(
                    ["gh"] + args,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                response = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired:
                response = {
                    "returncode": 124,
                    "stdout": "",
                    "stderr": "gh-proxy: command timed out (30s)\n",
                }
            except FileNotFoundError:
                response = {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "gh-proxy: gh CLI not found on host\n",
                }

        conn.sendall((json.dumps(response) + "\n").encode())
    except Exception as e:
        try:
            error_resp = json.dumps({
                "returncode": 1,
                "stdout": "",
                "stderr": f"gh-proxy: internal error: {e}\n",
            })
            conn.sendall((error_resp + "\n").encode())
        except Exception:
            pass
    finally:
        conn.close()


def serve(socket_path: str) -> None:
    """Run the proxy server."""
    # Clean up stale socket
    sock_path = Path(socket_path)
    if sock_path.exists():
        sock_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o600)
    server.listen(5)
    server.settimeout(1.0)  # Allow periodic interrupt check

    def shutdown(signum, frame):
        server.close()
        sock_path.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"gh-proxy: listening on {socket_path}", file=sys.stderr)

    try:
        while True:
            try:
                conn, _ = server.accept()
                thread = threading.Thread(target=handle_client, args=(conn,), daemon=True)
                thread.start()
            except socket.timeout:
                continue
    except Exception:
        pass
    finally:
        server.close()
        sock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <socket-path>", file=sys.stderr)
        sys.exit(1)
    serve(sys.argv[1])
