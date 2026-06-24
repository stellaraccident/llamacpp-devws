from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolPaths:
    loom_link: Path | None = None
    loom_compile: Path | None = None
    iree_benchmark_loom: Path | None = None

    def require_loom_link(self) -> Path:
        return _require_path(self.loom_link, "loom-link")

    def require_loom_compile(self) -> Path:
        return _require_path(self.loom_compile, "loom-compile")

    def require_iree_benchmark_loom(self) -> Path:
        return _require_path(self.iree_benchmark_loom, "iree-benchmark-loom")


@dataclass(frozen=True)
class BenchConfig:
    output_dir: Path
    target: str
    tools: ToolPaths
    rocm_path: Path | None = None

    def command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.rocm_path is not None:
            env.setdefault("ROCM_PATH", str(self.rocm_path))
            env.setdefault("GGML_HRX_ROCM_PATH", str(self.rocm_path))
            lib_paths = [self.rocm_path / "lib", self.rocm_path / "lib64"]
            existing = env.get("LD_LIBRARY_PATH")
            additions = [str(path) for path in lib_paths if path.exists()]
            if additions:
                env["LD_LIBRARY_PATH"] = ":".join(additions + ([existing] if existing else []))
        return env


def _require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} path is required for this command")
    if not path.exists():
        raise FileNotFoundError(path)
    return path
