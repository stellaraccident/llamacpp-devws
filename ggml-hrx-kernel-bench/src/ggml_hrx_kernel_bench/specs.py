from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelSpec:
    id: str
    op: str
    source: Path
    root_symbol: str
    export_name: str | None = None
    types: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    shape_domain: dict[str, Any] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_json(path: Path, *, kernel_source_override: Path | None = None) -> "KernelSpec":
        data = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.parent
        source = kernel_source_override or Path(data["source"])
        if not source.is_absolute():
            source = (base_dir / source).resolve()
        return KernelSpec(
            id=data["id"],
            op=data["op"],
            source=source,
            root_symbol=data["root_symbol"],
            export_name=data.get("export_name"),
            types=dict(data.get("types", {})),
            parameters=dict(data.get("parameters", {})),
            shape_domain=dict(data.get("shape_domain", {})),
            tuning=dict(data.get("tuning", {})),
        )

    def config_bindings(self, values: dict[str, Any]) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for name, meta in self.parameters.items():
            key = meta.get("config_key")
            if not key or name not in values:
                continue
            bindings[str(key)] = str(values[name])
        return bindings


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_args(bindings: dict[str, str]) -> list[str]:
    return [f"--config={key}={bindings[key]}" for key in sorted(bindings)]
