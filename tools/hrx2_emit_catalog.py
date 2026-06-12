#!/usr/bin/env python3
import argparse
import copy
import shutil
import sys
from pathlib import Path

from hrx2_pipeline_lib import DEFAULT_ARTIFACT_ROOT, DEFAULT_CATALOG, DEFAULT_SOURCE_ROOT, load_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Emit exploded HRX2 catalog directory from reduced evidence.")
    parser.add_argument("--reduced", required=True)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--catalog-id", default="hrx2-phase0.3")
    return parser.parse_args()


def copy_tree_contents(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    args = parse_args()
    catalog = load_json(args.catalog)
    reduced = load_json(args.reduced)
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    emitted = copy.deepcopy(catalog)
    emitted["catalog_id"] = args.catalog_id
    emitted["generated_at"] = "phase0.3"

    accepted_by_route = {}
    compile_only_by_route = {}
    rejected_by_route = {}
    for item in reduced.get("accepted", []):
        accepted_by_route.setdefault(item["route_id"], []).append(item)
    for item in reduced.get("compile_only", []):
        compile_only_by_route.setdefault(item["route_id"], []).append(item)
    for item in reduced.get("rejected", []):
        rejected_by_route.setdefault(item["route_id"], []).append(item)

    for route in emitted.get("routes", []):
        evidence = route.setdefault("evidence_summary", {})
        phase = evidence.setdefault("phase0.3", {})
        route_id = route["id"]
        phase["accepted_shapes"] = len(accepted_by_route.get(route_id, []))
        phase["compile_only_shapes"] = len(compile_only_by_route.get(route_id, []))
        phase["rejected_shapes"] = len(rejected_by_route.get(route_id, []))
        if accepted_by_route.get(route_id):
            phase["ggml_cpu_reference"] = "pass"

    write_json(out_dir / "catalog.json", emitted)
    shutil.copy2(args.reduced, out_dir / "reduced.json")
    copy_tree_contents(Path(args.source_root) / "kernels", out_dir / "kernels")
    copy_tree_contents(Path(args.artifact_root) / "artifacts", out_dir / "artifacts")
    print(f"wrote exploded catalog to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
