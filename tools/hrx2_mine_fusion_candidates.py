#!/usr/bin/env python3
"""Mine HRX2 fusion candidates from basket scheduler and HRX2 traces."""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIOR_ART_SEEDS = {
    ("RMS_NORM", "MUL"): {
        "class": "norm_weight",
        "source": "common transformer norm-weight fusion; reduces one F32 vector materialization",
    },
    ("MUL_MAT", "ADD"): {
        "class": "matmul_epilogue",
        "source": "HRX/Pyre notes and common GPU matmul epilogue pattern",
    },
    ("MUL_MAT", "ROPE"): {
        "class": "attention_projection_rope",
        "source": "attention backend precedent; defer until direct-dispatch/compiler stability is settled",
    },
    ("MUL_MAT", "SWIGLU"): {
        "class": "ffn_epilogue",
        "source": "FFN activation adjacency; practical only when graph layout and source fanout align",
    },
    ("MUL_MAT", "GEGLU"): {
        "class": "ffn_epilogue",
        "source": "FFN activation adjacency; practical only when graph layout and source fanout align",
    },
    ("MUL_MAT", "GLU"): {
        "class": "ffn_epilogue",
        "source": "CUDA quant matvec has GLU/gate/bias fusion support; trace op uses generic GLU",
    },
    ("MUL_MAT", "SOFT_MAX", "MUL_MAT"): {
        "class": "attention_matmul_softmax_matmul",
        "source": "common attention fusion/search target; accept only when fused memory traffic and launch savings beat unfused attention routes",
    },
}


HOST_OR_INFRA_OPS = {"NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE", "SET_ROWS"}


@dataclass
class Node:
    run: str
    regime: str
    index: int
    op: str
    name: str
    typ: str
    ne: tuple[int, ...]
    nb: tuple[int, ...]
    nbytes: int
    backend: str
    route_id: str = ""
    cache_key: str = ""
    src_types: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Basket run directory, run subdirectory, or sched.jsonl file.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--max-chain-length", type=int, default=4)
    parser.add_argument("--top", type=int, default=80)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def discover_sched_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_file():
            paths.append(item)
        elif item.is_dir():
            direct = item / "sched.jsonl"
            if direct.exists():
                paths.append(direct)
            else:
                paths.extend(sorted(item.glob("*/sched.jsonl")))
        else:
            raise SystemExit(f"input does not exist: {item}")
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise SystemExit("no sched.jsonl files found")
    return unique


def normalize_tensor_name(name: str) -> str:
    name = re.sub(r" \((reshaped|view|permuted|transposed)\)$", "", name)
    return name


def regime_from_path(path: Path) -> str:
    text = path.parent.name
    match = re.search(r"__p(\d+)n(\d+)", text)
    if not match:
        return "unknown"
    prompt = int(match.group(1))
    if prompt <= 1:
        return "decode"
    if prompt <= 16:
        return "narrow_prompt"
    return "prefill"


def shape_key(row: dict[str, Any]) -> str:
    return "x".join(str(v) for v in row.get("ne", []))


def dispatches_for_run(sched_path: Path) -> list[dict[str, Any]]:
    hrx_path = sched_path.with_name("hrx2.jsonl")
    if not hrx_path.exists():
        return []
    return [row for row in load_jsonl(hrx_path) if row.get("event") == "dispatch"]


def attach_dispatches(nodes: list[Node], dispatches: list[dict[str, Any]]) -> None:
    cursor = 0
    for node in nodes:
        while cursor < len(dispatches) and dispatches[cursor].get("op") != node.op:
            cursor += 1
        if cursor >= len(dispatches):
            return
        dispatch = dispatches[cursor]
        node.route_id = dispatch.get("route_id", "")
        node.cache_key = dispatch.get("cache_key", "")
        cursor += 1


def nodes_and_edges(sched_path: Path) -> tuple[list[Node], dict[int, set[int]], dict[int, int]]:
    rows = [
        row for row in load_jsonl(sched_path)
        if row.get("event") == "sched_node"
        and row.get("compute")
        and not row.get("is_cpu")
        and row.get("op") not in HOST_OR_INFRA_OPS
    ]
    run = sched_path.parent.name
    regime = regime_from_path(sched_path)
    nodes: list[Node] = []
    by_name: dict[str, int] = {}
    for ordinal, row in enumerate(rows):
        name = row.get("name", "")
        node = Node(
            run=run,
            regime=regime,
            index=ordinal,
            op=row.get("op", ""),
            name=name,
            typ=row.get("type", ""),
            ne=tuple(int(v) for v in row.get("ne", [])),
            nb=tuple(int(v) for v in row.get("nb", [])),
            nbytes=int(row.get("nbytes", 0) or 0),
            backend=row.get("backend", ""),
            src_types=tuple(src.get("type", "") for src in row.get("src", [])),
        )
        nodes.append(node)
        by_name[name] = ordinal
        by_name[normalize_tensor_name(name)] = ordinal
    attach_dispatches(nodes, dispatches_for_run(sched_path))

    edges: dict[int, set[int]] = collections.defaultdict(set)
    consumer_count: collections.Counter[int] = collections.Counter()
    for consumer_idx, row in enumerate(rows):
        for src in row.get("src", []):
            producer_idx = by_name.get(normalize_tensor_name(src.get("name", "")))
            if producer_idx is None or producer_idx >= consumer_idx:
                continue
            edges[producer_idx].add(consumer_idx)
            consumer_count[producer_idx] += 1
    return nodes, edges, dict(consumer_count)


def route_signature(nodes: list[Node]) -> tuple[str, ...]:
    return tuple(node.route_id for node in nodes if node.route_id)


def source_signature(nodes: list[Node]) -> str:
    return " -> ".join(
        f"{node.op}[{node.typ};{shape_key({'ne': node.ne})};src={','.join(node.src_types)}]"
        for node in nodes
    )


def prior_art_for_ops(ops: tuple[str, ...]) -> dict[str, str] | None:
    if ops in PRIOR_ART_SEEDS:
        return PRIOR_ART_SEEDS[ops]
    if len(ops) >= 2 and all(op == "ADD" for op in ops):
        return {
            "class": "multi_add",
            "source": "CUDA and Vulkan both encode multi-ADD fusion paths",
        }
    if len(ops) >= 2 and ops[0] == "MUL_MAT_ID" and "GLU" in ops:
        return {
            "class": "moe_matvec_glu_epilogue",
            "source": "CUDA mmvq fuses quant matvec gate/bias/GLU epilogues",
        }
    if len(ops) >= 2 and ops[0] == "MUL_MAT" and "GLU" in ops:
        return {
            "class": "ffn_matvec_glu_epilogue",
            "source": "CUDA mmvq fuses quant matvec gate/bias/GLU epilogues",
        }
    if len(ops) >= 3 and ops[:3] == ("MUL_MAT", "SOFT_MAX", "MUL_MAT"):
        return PRIOR_ART_SEEDS[("MUL_MAT", "SOFT_MAX", "MUL_MAT")]
    if len(ops) >= 2 and ops[0] == "MUL_MAT" and ops[1] == "ADD":
        return PRIOR_ART_SEEDS[("MUL_MAT", "ADD")]
    return None


def collect_paths(
        nodes: list[Node],
        edges: dict[int, set[int]],
        consumer_count: dict[int, int],
        max_chain_length: int) -> list[list[Node]]:
    chains: list[list[Node]] = []

    def visit(path: list[int]) -> None:
        if len(path) >= 2:
            chains.append([nodes[i] for i in path])
        if len(path) >= max_chain_length:
            return
        last = path[-1]
        for nxt in sorted(edges.get(last, [])):
            if len(path) >= 1 and consumer_count.get(last, 0) != 1:
                continue
            visit(path + [nxt])

    for index in range(len(nodes)):
        visit([index])
    return chains


def summarize(paths: list[Path], max_chain_length: int) -> dict[str, Any]:
    aggregate: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
    run_count = len(paths)
    for sched_path in paths:
        nodes, edges, consumer_count = nodes_and_edges(sched_path)
        for chain in collect_paths(nodes, edges, consumer_count, max_chain_length):
            ops = tuple(node.op for node in chain)
            regime = chain[0].regime
            key = (regime, ops, source_signature(chain))
            row = aggregate.setdefault(key, {
                "regime": regime,
                "ops": list(ops),
                "signature": source_signature(chain),
                "count": 0,
                "runs": set(),
                "models": set(),
                "bytes_saved_estimate": 0,
                "dispatches_saved_estimate": 0,
                "route_signatures": collections.Counter(),
                "examples": [],
                "prior_art": prior_art_for_ops(ops),
            })
            row["count"] += 1
            row["runs"].add(chain[0].run)
            row["models"].add(chain[0].run.split("__", 1)[0])
            row["bytes_saved_estimate"] += sum(node.nbytes for node in chain[:-1])
            row["dispatches_saved_estimate"] += len(chain) - 1
            routes = route_signature(chain)
            if routes:
                row["route_signatures"][routes] += 1
            if len(row["examples"]) < 5:
                row["examples"].append({
                    "run": chain[0].run,
                    "node_indices": [node.index for node in chain],
                    "names": [node.name for node in chain],
                    "routes": [node.route_id for node in chain],
                    "cache_keys": [node.cache_key for node in chain],
                })

    candidates = []
    for row in aggregate.values():
        route_counter = row.pop("route_signatures")
        row["runs"] = sorted(row["runs"])
        row["models"] = sorted(row["models"])
        row["model_count"] = len(row["models"])
        row["run_count"] = len(row["runs"])
        row["route_signatures"] = [
            {"count": count, "routes": list(routes)}
            for routes, count in route_counter.most_common(8)
        ]
        bytes_saved = int(row["bytes_saved_estimate"])
        dispatches_saved = int(row["dispatches_saved_estimate"])
        row["score"] = bytes_saved + dispatches_saved * 16384 + row["model_count"] * 1024
        row["status"] = "candidate"
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            row["score"],
            row["bytes_saved_estimate"],
            row["count"],
            row["model_count"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank
    return {
        "schema": "hrx2-fusion-candidates-v1",
        "input_count": run_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any], top: int) -> None:
    lines = [
        "# HRX2 Phase 2 Fusion Candidates",
        "",
        f"- Inputs: `{summary['input_count']}` scheduler traces",
        f"- Candidates: `{summary['candidate_count']}`",
        "",
        "| Rank | Regime | Ops | Count | Models | Est. bytes saved | Est. dispatches saved | Prior art |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(summary["candidates"][:top], 1):
        prior = row.get("prior_art") or {}
        lines.append(
            f"| {rank} | `{row['regime']}` | `{' -> '.join(row['ops'])}` | "
            f"{row['count']} | {row['model_count']} | {row['bytes_saved_estimate']} | "
            f"{row['dispatches_saved_estimate']} | {prior.get('class', '')} |"
        )
    lines.extend(["", "## Top Candidate Details", ""])
    for rank, row in enumerate(summary["candidates"][: min(top, 25)], 1):
        prior = row.get("prior_art") or {}
        lines.extend([
            f"### {rank}. {row['regime']} `{' -> '.join(row['ops'])}`",
            "",
            f"- Count: `{row['count']}` across `{row['model_count']}` models and `{row['run_count']}` runs.",
            f"- Estimated intermediate bytes saved: `{row['bytes_saved_estimate']}`.",
            f"- Estimated dispatches saved: `{row['dispatches_saved_estimate']}`.",
        ])
        if prior:
            lines.append(f"- Prior-art seed: `{prior['class']}` - {prior['source']}.")
        if row["route_signatures"]:
            routes = " -> ".join(row["route_signatures"][0]["routes"])
            lines.append(f"- Most common unfused route signature: `{routes}`.")
        example = row["examples"][0] if row["examples"] else {}
        if example:
            lines.append(f"- Example run: `{example['run']}` nodes `{example['node_indices']}`.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = discover_sched_paths(args.inputs)
    summary = summarize(paths, args.max_chain_length)
    write_json(args.json_out, summary)
    write_markdown(args.md_out, summary, args.top)


if __name__ == "__main__":
    main()
