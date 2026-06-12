#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from hrx2_pipeline_lib import read_jsonl, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Reduce HRX2 phase0.3 sweep evidence into route decisions.")
    parser.add_argument("--run", required=True, help="Run directory containing results.jsonl.")
    parser.add_argument("--out", required=True, help="Reduced JSON output path.")
    parser.add_argument("--summary-md", help="Optional Markdown summary output.")
    return parser.parse_args()


def classify(result):
    compile_status = result.get("loom_compile", {}).get("status")
    ggml_status = result.get("ggml_validation", {}).get("status")
    if compile_status != "pass":
        return "rejected", "compile_failed"
    if result.get("selected_by_priority"):
        if ggml_status != "pass":
            return "rejected", "ggml_failed"
        return "accepted", "validated_selected_route"
    return "compile_only", "not_backend_selected"


def load_json_if_exists(path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def main():
    args = parse_args()
    run_dir = Path(args.run)
    results = read_jsonl(run_dir / "results.jsonl")
    if not results:
        raise SystemExit(f"no results found under {run_dir}")

    decisions = []
    for result in results:
        decision, reason = classify(result)
        compile_report = load_json_if_exists(result.get("loom_compile", {}).get("compile_report_path", ""))
        manifest = load_json_if_exists(result.get("loom_compile", {}).get("manifest_path", ""))
        decisions.append({
            "route_id": result["route_id"],
            "shape_id": result["shape_id"],
            "target_key": result["target_key"],
            "cache_key": result["cache_key"],
            "decision": decision,
            "reason": reason,
            "selected_by_priority": result.get("selected_by_priority", False),
            "hsaco_size": result.get("loom_compile", {}).get("hsaco_size", 0),
            "compile_report_path": result.get("loom_compile", {}).get("compile_report_path"),
            "manifest_path": result.get("loom_compile", {}).get("manifest_path"),
            "compile_report_available": compile_report is not None,
            "manifest_available": manifest is not None,
            "ggml_validation": result.get("ggml_validation", {}),
            "candidate_dir": result.get("candidate_dir"),
        })

    reduced = {
        "schema": "hrx2-reduced-v1",
        "run_dir": str(run_dir),
        "accepted": [item for item in decisions if item["decision"] == "accepted"],
        "compile_only": [item for item in decisions if item["decision"] == "compile_only"],
        "rejected": [item for item in decisions if item["decision"] == "rejected"],
        "decisions": decisions,
    }
    write_json(args.out, reduced)

    summary_path = Path(args.summary_md) if args.summary_md else Path(args.out).with_suffix(".md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HRX2 Phase 0.3 Reduction",
        "",
        f"- Run: `{run_dir}`",
        f"- Accepted selected routes: {len(reduced['accepted'])}",
        f"- Compile-only alternates: {len(reduced['compile_only'])}",
        f"- Rejected: {len(reduced['rejected'])}",
        "",
        "| Decision | Route | Shape | Reason | HSACO bytes | GGML |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in decisions:
        lines.append(
            f"| {item['decision']} | `{item['route_id']}` | `{item['shape_id']}` | "
            f"{item['reason']} | {item['hsaco_size']} | {item['ggml_validation'].get('status', '')} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote reduced decisions to {args.out}", file=sys.stderr)
    print(f"wrote summary to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
