from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "ggml_hrx_kernel_bench.observed_shapes.v1"


@dataclass(frozen=True)
class ObservedShape:
    family: str
    source_id: str | None
    route_id: str | None
    root_symbol: str | None
    shape: dict[str, int]
    count: int = 1
    tags: tuple[str, ...] = ()
    sources: tuple[dict[str, Any], ...] = ()

    @property
    def key(self) -> tuple[str, str | None, str | None, str | None, tuple[tuple[str, int], ...]]:
        return (
            self.family,
            self.source_id,
            self.route_id,
            self.root_symbol,
            tuple(sorted(self.shape.items())),
        )

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "family": self.family,
            "shape": dict(sorted(self.shape.items())),
            "count": self.count,
        }
        if self.source_id:
            row["source_id"] = self.source_id
        if self.route_id:
            row["route_id"] = self.route_id
        if self.root_symbol:
            row["root_symbol"] = self.root_symbol
        if self.tags:
            row["tags"] = list(self.tags)
        if self.sources:
            row["sources"] = list(self.sources)
        return row


@dataclass
class ObservedShapeCatalog:
    rows: list[ObservedShape] = field(default_factory=list)

    def shapes_for_route(self, route: Mapping[str, Any]) -> list[dict[str, int]]:
        family = str(route.get("family") or route.get("source_id") or "")
        source_id = str(route.get("source_id") or "")
        route_id = str(route.get("id") or "")
        root_symbol = str(route.get("root_symbol") or "")
        matches: list[ObservedShape] = []
        for row in self.rows:
            if row.family and row.family != family:
                continue
            if row.route_id and row.route_id != route_id:
                continue
            if row.root_symbol and row.root_symbol != root_symbol:
                continue
            if row.source_id and row.source_id != source_id:
                continue
            matches.append(row)
        matches.sort(key=lambda row: (-row.count, row.route_id or "", row.root_symbol or "", sorted(row.shape.items())))
        return [dict(row.shape) for row in matches]

    def merge(self, observations: Iterable[ObservedShape]) -> None:
        merged: dict[tuple[str, str | None, str | None, str | None, tuple[tuple[str, int], ...]], ObservedShape] = {
            row.key: row for row in self.rows
        }
        for row in observations:
            existing = merged.get(row.key)
            if existing is None:
                merged[row.key] = row
                continue
            tags = tuple(sorted(set(existing.tags) | set(row.tags)))
            sources = tuple(_dedupe_sources((*existing.sources, *row.sources)))
            merged[row.key] = ObservedShape(
                family=existing.family,
                source_id=existing.source_id,
                route_id=existing.route_id,
                root_symbol=existing.root_symbol,
                shape=existing.shape,
                count=existing.count + row.count,
                tags=tags,
                sources=sources,
            )
        self.rows = sorted(merged.values(), key=lambda row: (row.family, row.route_id or "", row.root_symbol or "", sorted(row.shape.items())))

    def to_json(self) -> dict[str, Any]:
        by_family: dict[str, int] = defaultdict(int)
        for row in self.rows:
            by_family[row.family] += 1
        return {
            "schema": SCHEMA,
            "summary": {
                "row_count": len(self.rows),
                "families": dict(sorted(by_family.items())),
            },
            "rows": [row.to_json() for row in self.rows],
        }


def load_observed_shapes(path: Path | None) -> ObservedShapeCatalog:
    if path is None or not path.exists():
        return ObservedShapeCatalog()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [_row_from_json(row) for row in data.get("rows", [])]
    return ObservedShapeCatalog(rows)


def save_observed_shapes(path: Path, catalog: ObservedShapeCatalog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_observations_jsonl(paths: Iterable[Path]) -> list[ObservedShape]:
    rows: list[ObservedShape] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                observed = observation_from_event(event)
                if observed is None:
                    raise ValueError(f"{path}:{line_no}: missing family/shape fields")
                rows.append(observed)
    return rows


def observation_from_event(event: Mapping[str, Any]) -> ObservedShape | None:
    candidate = event.get("candidate") if isinstance(event.get("candidate"), Mapping) else {}
    shape = event.get("shape") or candidate.get("shape")
    if not isinstance(shape, Mapping):
        return None
    family = event.get("family") or candidate.get("family")
    if not family:
        return None
    count = int(event.get("count") or event.get("hit_count") or 1)
    tags = event.get("tags") if isinstance(event.get("tags"), list) else []
    source = event.get("source") if isinstance(event.get("source"), Mapping) else {}
    return ObservedShape(
        family=str(family),
        source_id=_optional_str(event.get("source_id") or candidate.get("source_id")),
        route_id=_optional_str(event.get("route_id") or candidate.get("route_id")),
        root_symbol=_optional_str(event.get("root_symbol") or candidate.get("root_symbol")),
        shape={str(key): int(value) for key, value in shape.items()},
        count=count,
        tags=tuple(str(tag) for tag in tags),
        sources=(dict(source),) if source else (),
    )


def _row_from_json(row: Mapping[str, Any]) -> ObservedShape:
    shape = row.get("shape")
    if not isinstance(shape, Mapping):
        raise ValueError("observed shape row is missing shape object")
    return ObservedShape(
        family=str(row["family"]),
        source_id=_optional_str(row.get("source_id")),
        route_id=_optional_str(row.get("route_id")),
        root_symbol=_optional_str(row.get("root_symbol")),
        shape={str(key): int(value) for key, value in shape.items()},
        count=int(row.get("count") or 1),
        tags=tuple(str(tag) for tag in row.get("tags", [])),
        sources=tuple(dict(source) for source in row.get("sources", [])),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dedupe_sources(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        key = json.dumps(source, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result
