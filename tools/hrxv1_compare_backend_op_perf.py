#!/usr/bin/env python3
"""Compare focused test-backend-ops perf CSVs by op row name."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs", required=True, type=Path, help="left/baseline CSV")
    parser.add_argument("--rhs", required=True, type=Path, help="right/reference CSV")
    parser.add_argument("--lhs-label", default="lhs")
    parser.add_argument("--rhs-label", default="rhs")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args()


def row_key(row: dict[str, str]) -> str:
    params = row.get("op_params", "")
    match = re.search(r"name=([^,]+)", params)
    if match:
        return match.group(1)
    return params or row.get("op_name", "")


def read_perf(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            key = row_key(row)
            rows[key] = {
                "backend_name": row.get("backend_name", ""),
                "op_name": row.get("op_name", ""),
                "op_params": row.get("op_params", ""),
                "supported": row.get("supported") == "1",
                "passed": row.get("passed") == "1",
                "time_us": float(row.get("time_us") or 0.0),
                "n_runs": int(float(row.get("n_runs") or 0)),
            }
    return rows


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    lhs = read_perf(args.lhs)
    rhs = read_perf(args.rhs)
    keys = sorted(set(lhs) | set(rhs))

    rows: list[dict[str, Any]] = []
    lhs_total = 0.0
    rhs_total = 0.0
    for key in keys:
        lrow = lhs.get(key)
        rrow = rhs.get(key)
        ltime = lrow["time_us"] if lrow else None
        rtime = rrow["time_us"] if rrow else None
        ratio = (ltime / rtime) if ltime is not None and rtime and rtime > 0 else None
        if ltime is not None:
            lhs_total += ltime
        if rtime is not None:
            rhs_total += rtime
        rows.append({
            "name": key,
            "lhs_time_us": ltime,
            "rhs_time_us": rtime,
            "lhs_over_rhs": ratio,
            "lhs_n_runs": lrow["n_runs"] if lrow else None,
            "rhs_n_runs": rrow["n_runs"] if rrow else None,
            "lhs_passed": lrow["passed"] if lrow else None,
            "rhs_passed": rrow["passed"] if rrow else None,
        })

    summary = {
        "lhs": str(args.lhs),
        "rhs": str(args.rhs),
        "lhs_label": args.lhs_label,
        "rhs_label": args.rhs_label,
        "lhs_total_us": lhs_total,
        "rhs_total_us": rhs_total,
        "lhs_over_rhs_total": lhs_total / rhs_total if rhs_total > 0 else None,
        "rows": rows,
    }

    md_lines = [
        f"# Backend Op Perf Compare: {args.lhs_label} vs {args.rhs_label}",
        "",
        f"- `{args.lhs_label}` total us: `{fmt(lhs_total)}`",
        f"- `{args.rhs_label}` total us: `{fmt(rhs_total)}`",
        f"- total ratio `{args.lhs_label}/{args.rhs_label}`: `{fmt(summary['lhs_over_rhs_total'])}`",
        "",
        "| Row | " + args.lhs_label + " us | " + args.rhs_label + " us | Ratio | " +
        args.lhs_label + " runs | " + args.rhs_label + " runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| `{row['name']}` | {fmt(row['lhs_time_us'])} | {fmt(row['rhs_time_us'])} | "
            f"{fmt(row['lhs_over_rhs'])} | {row['lhs_n_runs'] or ''} | {row['rhs_n_runs'] or ''} |"
        )
    md = "\n".join(md_lines) + "\n"

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(md, end="")


if __name__ == "__main__":
    main()
