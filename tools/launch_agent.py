#!/usr/bin/env python3
"""Launch a sandboxed agent session for the llama.cpp HRX workspace.

Usage:
    launch_agent.py generalist
    launch_agent.py coder
    launch_agent.py kernel
    launch_agent.py build
    launch_agent.py reviewer
    launch_agent.py shell
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ROLE_PROMPTS: dict[str, str] = {
    "generalist": (
        "You are co-working with stella in the llama.cpp HRX development "
        "workspace. Read AGENTS.md and README.md first. Work directly from the "
        "human request; there is no beads or topic workflow in this workspace. "
        "Keep root-repo edits limited to docs, tools, and skills, and put code "
        "changes in sources/llama.cpp or sources/hrx-system."
    ),
    "coder": (
        "You are a coding agent for llama.cpp with HRX support. Read AGENTS.md "
        "and README.md first. Inspect sources/llama.cpp and sources/hrx-system "
        "before changing code. Keep commits and branch operations out of the "
        "workspace root unless the human explicitly asks."
    ),
    "kernel": (
        "You are focused on HRX pure-HIP kernel optimization for llama.cpp. "
        "Read AGENTS.md, README.md, and docs/kernel-skill/SKILL.md first, then "
        "load only the referenced kernel skill material needed for the task. "
        "Use profiling and correctness gates before promoting kernel changes."
    ),
    "build": (
        "You are focused on build and runtime integration for llama.cpp and "
        "hrx-system. Read AGENTS.md and README.md first. Prefer workspace-local "
        "ROCm via $ROCM_PATH, build under build/, and keep source-repo changes "
        "scoped to the relevant checkout."
    ),
    "reviewer": (
        "You are reviewing llama.cpp/HRX changes. Read AGENTS.md and README.md "
        "first. Prioritize correctness, profiling evidence, build reproducibility, "
        "and whether changes are scoped to the relevant source checkout."
    ),
}

ROLE_INITIAL: dict[str, str] = {
    "generalist": "Read AGENTS.md and README.md, then ask what we are looking at.",
    "coder": "Read AGENTS.md and README.md, then inspect source checkout status.",
    "kernel": "Read AGENTS.md, README.md, and docs/kernel-skill/SKILL.md.",
    "build": "Read AGENTS.md and README.md, then inspect the build environment.",
    "reviewer": "Read AGENTS.md and README.md, then ask which change to review.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a sandboxed agent session")
    parser.add_argument(
        "role",
        choices=list(ROLE_PROMPTS.keys()) + ["shell"],
        help="Agent role to launch",
    )
    parser.add_argument("--model", default=None, help="Override model")
    parser.add_argument("--no-sandbox", action="store_true", help="Run without bwrap")
    args = parser.parse_args()

    sandbox = SCRIPT_DIR / "sandbox.py"

    if args.role == "shell":
        print("Launching sandboxed shell...")
        os.execvp(sys.executable, [sys.executable, str(sandbox)])
        return

    model = args.model or ("opus" if args.role == "generalist" else "sonnet")

    claude_args = [
        "claude",
        "--dangerously-skip-permissions",
        "--append-system-prompt",
        ROLE_PROMPTS[args.role],
        "--model",
        model,
        ROLE_INITIAL[args.role],
    ]

    print(f"Launching {args.role} agent (model={model}) in sandbox...")

    if args.no_sandbox:
        os.execvp("claude", claude_args)
    else:
        os.execvp(sys.executable, [sys.executable, str(sandbox)] + claude_args)


if __name__ == "__main__":
    main()
