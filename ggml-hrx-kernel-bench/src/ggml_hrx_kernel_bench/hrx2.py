from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .family_specs import concrete_shapes_for_route, resolve_binding_value
from .observed_shapes import ObservedShapeCatalog
from .specs import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HRX2_KERNEL_DIR = PROJECT_ROOT / "kernels" / "hrx2"
DEFAULT_HRX2_CATALOG_DIR = PROJECT_ROOT / "catalog" / "hrx2"

KERNEL_DEF_RE = re.compile(
    r"kernel\.def(?:\s+target\([^)]*\))?(?:\s+export\(\"(?P<export>[^\"]+)\"\))?\s+(?P<root>@[A-Za-z0-9_.$-]+)"
)
CONFIG_DECL_RE = re.compile(r"config\.decl\s+(?P<key>@[A-Za-z0-9_.$-]+)")

@dataclass(frozen=True)
class KernelSource:
    source_id: str
    path: Path
    imported_sha256: str | None
    root_symbols: tuple[str, ...]
    export_names: tuple[str | None, ...]
    config_keys: tuple[str, ...]
    route_count: int
    coverage: str
    original_path: str | None = None


@dataclass(frozen=True)
class Candidate:
    id: str
    family: str
    op: str
    source_id: str
    source_path: Path
    root_symbol: str
    export_name: str | None
    route_id: str | None
    route: dict[str, Any] | None
    shape: dict[str, int]
    values: dict[str, int | str]
    config: dict[str, str]
    dispatch: dict[str, Any]
    supports: dict[str, Any]
    coverage: str
    status: str = "planned"
    message: str | None = None

    def to_ledger(self) -> dict[str, Any]:
        return {
            "candidate_id": self.id,
            "family": self.family,
            "op": self.op,
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "source_sha256": file_sha256(self.source_path),
            "root_symbol": self.root_symbol,
            "export_name": self.export_name,
            "route_id": self.route_id,
            "shape": self.shape,
            "values": self.values,
            "config_bindings": self.config,
            "dispatch": self.dispatch,
            "supports": self.supports,
            "coverage": self.coverage,
            "status": self.status,
            "message": self.message,
        }


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_id(*parts: Any, length: int = 10) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def read_sources(catalog_dir: Path) -> dict[str, dict[str, Any]]:
    path = catalog_dir / "sources.json"
    if not path.exists():
        return {}
    return dict(load_json(path))


def iter_routes(catalog_dir: Path) -> Iterable[dict[str, Any]]:
    routes_dir = catalog_dir / "routes"
    if not routes_dir.exists():
        return
    for path in sorted(routes_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                route = dict(row)
                route["_route_file"] = path.name
                yield route


def parse_roots(path: Path) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    text = path.read_text(encoding="utf-8")
    roots: list[str] = []
    exports: list[str | None] = []
    for match in KERNEL_DEF_RE.finditer(text):
        roots.append(match.group("root"))
        exports.append(match.group("export"))
    return tuple(roots), tuple(exports)


def parse_config_keys(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(sorted(set(match.group("key") for match in CONFIG_DECL_RE.finditer(text))))


def build_manifest(kernel_dir: Path, catalog_dir: Path, *, original_root: Path | None = None) -> dict[str, Any]:
    sources = read_sources(catalog_dir)
    routes = list(iter_routes(catalog_dir))
    route_counts: dict[str, int] = {}
    for route in routes:
        source_id = str(route.get("source_id") or "")
        if source_id:
            route_counts[source_id] = route_counts.get(source_id, 0) + 1

    source_by_path: dict[str, list[str]] = {}
    for source_id, row in sources.items():
        rel = str(row.get("path", ""))
        if rel:
            source_by_path.setdefault(Path(rel).name, []).append(source_id)

    entries: list[dict[str, Any]] = []
    for path in sorted(kernel_dir.glob("*.loom")):
        roots, exports = parse_roots(path)
        source_ids = source_by_path.get(path.name, [])
        if not source_ids:
            source_ids = [path.stem]
        coverage = "route_backed" if any(route_counts.get(source_id, 0) for source_id in source_ids) else "source_only"
        original_path = None
        original_sha256 = None
        if original_root is not None:
            candidate = original_root / "kernels" / path.name
            if candidate.exists():
                original_path = str(candidate)
                original_sha256 = file_sha256(candidate)
        entries.append(
            {
                "source_ids": source_ids,
                "path": project_relative(path),
                "imported_sha256": file_sha256(path),
                "original_path": original_path,
                "original_sha256": original_sha256,
                "mechanical_rewrites": ["content differs from original HRX2 source"] if original_sha256 and original_sha256 != file_sha256(path) else [],
                "root_symbols": list(roots),
                "export_names": list(exports),
                "config_keys": list(parse_config_keys(path)),
                "route_count": sum(route_counts.get(source_id, 0) for source_id in source_ids),
                "coverage": coverage,
            }
        )

    kernel_file_names = {Path(entry["path"]).name for entry in entries}
    catalog_file_names = {Path(row.get("path", "")).name for row in sources.values()}
    route_source_ids = {str(route.get("source_id")) for route in routes if route.get("source_id")}
    return {
        "schema": "ggml_hrx_kernel_bench.hrx2_manifest.v1",
        "kernel_count": len(entries),
        "catalog_source_count": len(sources),
        "route_count": len(routes),
        "entries": entries,
        "source_ids_without_routes": sorted(set(sources) - route_source_ids),
        "route_source_ids_without_source_entry": sorted(route_source_ids - set(sources)),
        "kernel_files_without_source_entry": sorted(kernel_file_names - catalog_file_names),
        "source_entries_without_kernel_file": sorted(catalog_file_names - kernel_file_names),
    }


def load_sources_by_id(kernel_dir: Path, catalog_dir: Path) -> dict[str, KernelSource]:
    sources = read_sources(catalog_dir)
    routes = list(iter_routes(catalog_dir))
    route_counts: dict[str, int] = {}
    for route in routes:
        source_id = str(route.get("source_id") or "")
        if source_id:
            route_counts[source_id] = route_counts.get(source_id, 0) + 1

    out: dict[str, KernelSource] = {}
    for source_id, row in sources.items():
        rel = Path(str(row.get("path", ""))).name
        path = kernel_dir / rel
        if not path.exists():
            continue
        roots, exports = parse_roots(path)
        route_count = route_counts.get(source_id, 0)
        out[source_id] = KernelSource(
            source_id=source_id,
            path=path,
            imported_sha256=file_sha256(path),
            root_symbols=roots,
            export_names=exports,
            config_keys=parse_config_keys(path),
            route_count=route_count,
            coverage="route_backed" if route_count else "source_only",
        )

    known_paths = {source.path.name for source in out.values()}
    for path in sorted(kernel_dir.glob("*.loom")):
        if path.name in known_paths:
            continue
        roots, exports = parse_roots(path)
        out[path.stem] = KernelSource(
            source_id=path.stem,
            path=path,
            imported_sha256=file_sha256(path),
            root_symbols=roots,
            export_names=exports,
            config_keys=parse_config_keys(path),
            route_count=0,
            coverage="source_only",
        )
    return out


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def route_short_name(route: dict[str, Any]) -> str:
    root = str(route.get("root_symbol", ""))
    if "splitk2" in root:
        return "splitk2"
    if "wmma64x64" in root:
        return "wmma64"
    if "wmma128x128" in root:
        return "wmma128"
    if root.endswith("_static"):
        return "direct"
    route_id = re.sub(r"[^a-z0-9]+", "_", str(route.get("id", "route")).lower()).strip("_")
    return route_id[:48] or "route"


def concrete_shape(route: dict[str, Any], *, sweep: str, observed_shapes: ObservedShapeCatalog | None = None) -> dict[str, int]:
    shapes = concrete_shapes(route, sweep=sweep, observed_shapes=observed_shapes)
    return shapes[0] if shapes else {}


def concrete_shapes(route: dict[str, Any], *, sweep: str, observed_shapes: ObservedShapeCatalog | None = None) -> list[dict[str, int]]:
    observed = observed_shapes.shapes_for_route(route) if observed_shapes else ()
    return concrete_shapes_for_route(route, sweep=sweep, observed_shapes=observed)


def build_config(route: dict[str, Any], shape: dict[str, int]) -> tuple[dict[str, str], dict[str, int | str], list[str]]:
    config: dict[str, str] = {}
    values: dict[str, int | str] = dict(shape)
    missing: list[str] = []
    family = str(route.get("family") or route.get("source_id") or "unknown")
    for binding in (route.get("specialization") or {}).get("bindings", []):
        key = str(binding["key"])
        if "source" in binding:
            source = str(binding["source"])
            value = resolve_binding_value(family, source, shape)
            if value is None:
                missing.append(source)
                continue
            config[key] = str(value)
            values[source] = value
        else:
            config[key] = str(binding["value"])
    return config, values, missing


def route_launch(route: dict[str, Any], shape: dict[str, int]) -> dict[str, Any]:
    dispatch = dict(route.get("dispatch") or {})
    rows_per_workgroup = int(dispatch.get("rows_per_workgroup", 1) or 1)
    cols_per_workgroup = int(dispatch.get("cols_per_workgroup", 1) or 1)
    rows = int(shape.get("rows", shape.get("nrows", 1)))
    cols = int(shape.get("cols", shape.get("ncols", 1)))
    return {
        "workgroup_count": [ceil_div(rows, rows_per_workgroup), ceil_div(cols, cols_per_workgroup), 1],
        "workgroup_size": dispatch.get("workgroup_size", [None, None, None]),
        "rows_per_workgroup": rows_per_workgroup,
        "cols_per_workgroup": cols_per_workgroup,
        "metadata_source": "route_heuristic",
        "has_static_dispatch_workgroup_count": False,
        "has_static_workgroup_size": bool(dispatch.get("workgroup_size")),
    }


def route_candidates(
    kernel_dir: Path,
    catalog_dir: Path,
    *,
    families: set[str] | None = None,
    limit: int | None = None,
    sweep: str = "minimal",
    observed_shapes: ObservedShapeCatalog | None = None,
) -> list[Candidate]:
    sources = load_sources_by_id(kernel_dir, catalog_dir)
    candidates: list[Candidate] = []
    for route in iter_routes(catalog_dir):
        family = str(route.get("family") or route.get("source_id") or "unknown")
        if families and family not in families and str(route.get("source_id")) not in families and str(route.get("id")) not in families:
            continue
        source_id = str(route.get("source_id") or "")
        source = sources.get(source_id)
        for shape in concrete_shapes(route, sweep=sweep, observed_shapes=observed_shapes):
            config, values, missing = build_config(route, shape)
            status = "planned" if source and not missing else "missing_config"
            message = None
            if not source:
                status = "missing_source"
                message = f"source_id {source_id!r} is not present in imported kernel corpus"
            elif missing:
                message = "missing shape/config values: " + ", ".join(missing)
            candidate_id = "_".join(
                [
                    route_short_name(route),
                    str(route.get("source_id") or "source"),
                    stable_id(route.get("id"), shape, config, length=8),
                ]
            )
            candidates.append(
                Candidate(
                    id=candidate_id,
                    family=family,
                    op=str(route.get("op") or ""),
                    source_id=source_id,
                    source_path=source.path if source else kernel_dir / "__missing__.loom",
                    root_symbol=str(route.get("root_symbol") or ""),
                    export_name=route.get("export_name"),
                    route_id=str(route.get("id") or ""),
                    route=route,
                    shape=shape,
                    values=values,
                    config=config,
                    dispatch=route_launch(route, shape),
                    supports=dict(route.get("supports") or {}),
                    coverage="route_backed",
                    status=status,
                    message=message,
                )
            )
            if limit and len(candidates) >= limit:
                return candidates
    return candidates


def source_only_candidates(
    kernel_dir: Path,
    catalog_dir: Path,
    *,
    families: set[str] | None = None,
    limit: int | None = None,
) -> list[Candidate]:
    routed_source_ids = {str(route.get("source_id")) for route in iter_routes(catalog_dir) if route.get("source_id")}
    sources = load_sources_by_id(kernel_dir, catalog_dir)
    candidates: list[Candidate] = []
    seen_source_hashes: set[str] = set()
    for source in sorted(sources.values(), key=lambda item: item.source_id):
        if source.source_id in routed_source_ids:
            continue
        if families and source.source_id not in families and source.path.stem not in families:
            continue
        source_hash = file_sha256(source.path)
        if source_hash in seen_source_hashes:
            continue
        seen_source_hashes.add(source_hash)
        for index, root in enumerate(source.root_symbols):
            candidates.append(
                Candidate(
                    id=f"source_only_{source.source_id}_{stable_id(root, length=8)}",
                    family=source.source_id,
                    op="SOURCE_ONLY",
                    source_id=source.source_id,
                    source_path=source.path,
                    root_symbol=root,
                    export_name=source.export_names[index] if index < len(source.export_names) else None,
                    route_id=None,
                    route=None,
                    shape={},
                    values={},
                    config={},
                    dispatch={},
                    supports={},
                    coverage="source_only",
                    status="missing_shape" if source.config_keys else "planned",
                    message="source-only kernel has no route-derived shape/config" if source.config_keys else None,
                )
            )
            if limit and len(candidates) >= limit:
                return candidates
    return candidates


def all_candidates(
    kernel_dir: Path,
    catalog_dir: Path,
    *,
    families: set[str] | None = None,
    limit: int | None = None,
    sweep: str = "minimal",
    observed_shapes: ObservedShapeCatalog | None = None,
    include_source_only: bool = False,
) -> list[Candidate]:
    candidates = route_candidates(kernel_dir, catalog_dir, families=families, limit=limit, sweep=sweep, observed_shapes=observed_shapes)
    if include_source_only and (limit is None or len(candidates) < limit):
        remaining = None if limit is None else limit - len(candidates)
        candidates.extend(source_only_candidates(kernel_dir, catalog_dir, families=families, limit=remaining))
    return candidates
