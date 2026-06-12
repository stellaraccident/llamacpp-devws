#!/usr/bin/env python3
import argparse
import sys

from hrx2_pipeline_lib import (
    DEFAULT_CATALOG,
    load_json,
    provider_cache_key,
    read_jsonl,
    resolve_config_bindings,
    route_matches_shape,
    shape_identity,
    test_backend_filter,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HRX2 route candidates from shape evidence and catalog JSON.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-key", default=None, help="Override target key for candidate matching.")
    return parser.parse_args()


def main():
    args = parse_args()
    catalog = load_json(args.catalog)
    routes = catalog.get("routes", [])
    sources = catalog.get("sources", {})
    artifacts = catalog.get("artifacts", {})
    shapes = read_jsonl(args.shapes)
    candidates = []

    for shape in shapes:
        target_key = args.target_key or shape.get("target_key") or "gfx1100"
        matching = [
            route for route in routes
            if route_matches_shape(route, shape, target_key=target_key)
        ]
        matching.sort(key=lambda route: int(route.get("priority", 0)), reverse=True)
        for rank, route in enumerate(matching):
            bindings = resolve_config_bindings(route, shape)
            candidates.append({
                "schema": "hrx2-candidate-v1",
                "catalog_id": catalog.get("catalog_id", ""),
                "target_key": target_key,
                "shape_id": shape_identity(shape),
                "route_rank": rank,
                "selected_by_priority": rank == 0,
                "shape": shape,
                "route": route,
                "source": sources.get(route.get("source_id"), {}),
                "artifact": artifacts.get(route.get("artifact_id"), {}),
                "config_bindings": bindings,
                "cache_key": provider_cache_key(route, shape, target_key, bindings),
                "test_backend_filter": test_backend_filter(shape),
            })

    write_jsonl(args.out, candidates)
    print(f"wrote {len(candidates)} candidates to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
