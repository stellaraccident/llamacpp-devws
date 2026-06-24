from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    elapsed_ms: float
    stdout: str
    stderr: str

    @property
    def status(self) -> str:
        return "ok" if self.returncode == 0 else "failed"

    def to_ledger(self, *, stdout_limit: int = 6000, stderr_limit: int = 6000) -> dict:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "stdout_tail": self.stdout[-stdout_limit:] if self.stdout else None,
            "stderr_tail": self.stderr[-stderr_limit:] if self.stderr else None,
        }


def run_command(
    argv: Sequence[str | Path],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    argv_str = [str(item) for item in argv]
    start = time.monotonic()
    proc = subprocess.run(
        argv_str,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(
        argv=argv_str,
        returncode=proc.returncode,
        elapsed_ms=(time.monotonic() - start) * 1000.0,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
