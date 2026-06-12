#!/usr/bin/env python3
import argparse
import sys

from hrx2_pipeline_lib import read_jsonl, shape_identity, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Collect HRX2 phase0.3 shape evidence.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--fixtures-only", action="store_true", help="Emit deterministic phase0.3 fixture shapes.")
    parser.add_argument("--trace-jsonl", action="append", default=[], help="Ingest GGML_HRX2_TRACE_JSONL output.")
    parser.add_argument("--target-key", default="gfx1100")
    return parser.parse_args()


def fixture_shapes(target_key):
    rows = []
    for ne, eps in [([64, 5, 4, 3], 0.000001), ([1025, 5, 4, 3], 0.000001)]:
        nrows = ne[1] * ne[2] * ne[3]
        rows.append({
            "schema": "hrx2-shape-v1",
            "source": "phase0.3-fixture",
            "target_key": target_key,
            "op": "RMS_NORM",
            "family": "rms_norm_f32",
            "src0_type": "F32",
            "dst_type": "F32",
            "layout": "contiguous",
            "ncols": ne[0],
            "nrows": nrows,
            "ne": ne,
            "eps": eps,
        })

    for k, rows_, cols in [
        (256, 16, 1),
        (256, 16, 16),
        (256, 1, 64),
        (5120, 6, 4096),
    ]:
        rows.append({
            "schema": "hrx2-shape-v1",
            "source": "phase0.3-fixture",
            "target_key": target_key,
            "op": "MUL_MAT",
            "family": "mul_mat_q8_0_f32",
            "src0_type": "Q8_0",
            "src1_type": "F32",
            "dst_type": "F32",
            "layout": "contiguous",
            "k": k,
            "rows": rows_,
            "cols": cols,
        })
    return rows


def shape_from_dispatch(event):
    if event.get("event") != "dispatch":
        return None
    op = event.get("op")
    if op == "RMS_NORM":
        return {
            "schema": "hrx2-shape-v1",
            "source": "hrx2-trace",
            "target_key": event.get("target_key", ""),
            "op": "RMS_NORM",
            "family": "rms_norm_f32",
            "src0_type": "F32",
            "dst_type": "F32",
            "layout": "contiguous",
            "ncols": int(event["ncols"]),
            "nrows": int(event["nrows"]),
        }
    if op == "MUL_MAT":
        return {
            "schema": "hrx2-shape-v1",
            "source": "hrx2-trace",
            "target_key": event.get("target_key", ""),
            "op": "MUL_MAT",
            "family": "mul_mat_q8_0_f32",
            "src0_type": "Q8_0",
            "src1_type": "F32",
            "dst_type": "F32",
            "layout": "contiguous",
            "k": int(event["k"]),
            "rows": int(event["rows"]),
            "cols": int(event["cols"]),
        }
    return None


def main():
    args = parse_args()
    if not args.fixtures_only and not args.trace_jsonl:
        raise SystemExit("provide --fixtures-only, --trace-jsonl, or both")

    shapes = []
    if args.fixtures_only:
        shapes.extend(fixture_shapes(args.target_key))
    for trace_path in args.trace_jsonl:
        for event in read_jsonl(trace_path):
            shape = shape_from_dispatch(event)
            if shape:
                shapes.append(shape)

    deduped = {}
    for shape in shapes:
        key = (shape.get("target_key"), shape_identity(shape))
        deduped.setdefault(key, shape)
    write_jsonl(args.out, deduped.values())
    print(f"wrote {len(deduped)} shapes to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
