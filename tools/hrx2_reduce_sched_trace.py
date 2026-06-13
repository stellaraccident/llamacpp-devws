#!/usr/bin/env python3
"""Reduce GGML_SCHED_TRACE_JSONL files into HRX2 fallback summaries."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


HOST_ORCHESTRATION_OPS = {
    "NONE",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
}

INFRASTRUCTURE_BLOCKER_OPS = {
    "SET_ROWS",
}


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
                row["_trace_path"] = str(path)
                rows.append(row)
    return rows


def src_signature(row: dict[str, Any]) -> str:
    return ",".join(src.get("type", "") for src in row.get("src", []))


def shape_key(row: dict[str, Any]) -> str:
    ne = row.get("ne", [])
    return "x".join(str(v) for v in ne)


def classify(row: dict[str, Any]) -> str:
    if not row.get("compute", False):
        return "host_orchestration"
    op = row.get("op", "")
    if op in HOST_ORCHESTRATION_OPS:
        return "host_orchestration"
    if op in INFRASTRUCTURE_BLOCKER_OPS:
        return "infrastructure_blocker"
    if row.get("is_cpu", False):
        return "compute_fallback"
    return "accelerated"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row.get("event") == "sched_node"]
    by_class: collections.Counter[str] = collections.Counter()
    by_backend: collections.Counter[str] = collections.Counter()
    fallback: collections.Counter[tuple[str, str, str, str]] = collections.Counter()
    accelerated: collections.Counter[tuple[str, str, str, str]] = collections.Counter()
    supported_cpu: collections.Counter[tuple[str, str, str, str, str]] = collections.Counter()

    for row in rows:
        cls = classify(row)
        by_class[cls] += 1
        if row.get("compute", False):
            by_backend[row.get("backend", "")] += 1
        key = (row.get("op", ""), row.get("type", ""), src_signature(row), shape_key(row))
        if cls == "compute_fallback":
            fallback[key] += 1
            supported = ",".join(row.get("supported_by", []))
            if "HRX" in supported:
                supported_cpu[(key[0], key[1], key[2], key[3], supported)] += 1
        elif cls == "accelerated":
            accelerated[key] += 1

    return {
        "node_count": len(rows),
        "class_counts": dict(by_class.most_common()),
        "compute_backend_counts": dict(by_backend.most_common()),
        "top_compute_fallbacks": [
            {
                "count": count,
                "op": key[0],
                "type": key[1],
                "src_types": key[2],
                "shape": key[3],
            }
            for key, count in fallback.most_common(100)
        ],
        "top_accelerated": [
            {
                "count": count,
                "op": key[0],
                "type": key[1],
                "src_types": key[2],
                "shape": key[3],
            }
            for key, count in accelerated.most_common(50)
        ],
        "cpu_assigned_but_hrx_supported": [
            {
                "count": count,
                "op": key[0],
                "type": key[1],
                "src_types": key[2],
                "shape": key[3],
                "supported_by": key[4],
            }
            for key, count in supported_cpu.most_common(50)
        ],
    }


def write_markdown(summary: dict[str, Any], trace_paths: list[Path], out: Path) -> None:
    lines: list[str] = []
    lines.append("# HRX2 Scheduler Trace Reduction")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for path in trace_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- nodes: `{summary['node_count']}`")
    for key, value in summary["class_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Compute Backends")
    lines.append("")
    for key, value in summary["compute_backend_counts"].items():
        lines.append(f"- {key or '(none)'}: `{value}`")
    lines.append("")
    lines.append("## Top Compute Fallbacks")
    lines.append("")
    lines.append("| Count | Op | Dst | Src | Shape |")
    lines.append("| ---: | --- | --- | --- | --- |")
    for row in summary["top_compute_fallbacks"][:40]:
        lines.append(
            f"| {row['count']} | `{row['op']}` | `{row['type']}` | "
            f"`{row['src_types']}` | `{row['shape']}` |"
        )
    lines.append("")
    lines.append("## CPU Assigned But HRX Supported")
    lines.append("")
    lines.append("| Count | Op | Dst | Src | Shape | Supported By |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for row in summary["cpu_assigned_but_hrx_supported"][:30]:
        lines.append(
            f"| {row['count']} | `{row['op']}` | `{row['type']}` | "
            f"`{row['src_types']}` | `{row['shape']}` | `{row['supported_by']}` |"
        )
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="Scheduler JSONL trace files.")
    parser.add_argument("--json-out", type=Path, help="Write reduced JSON summary.")
    parser.add_argument("--md-out", type=Path, help="Write markdown report.")
    args = parser.parse_args()

    rows = load_rows(args.traces)
    summary = summarize(rows)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        write_markdown(summary, args.traces, args.md_out)
    if not args.json_out and not args.md_out:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
