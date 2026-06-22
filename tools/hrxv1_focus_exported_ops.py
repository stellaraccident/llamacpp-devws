#!/usr/bin/env python3
"""Filter llama.cpp export-graph-ops output into focused HRX v1 test files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


GGML_OP = {
    29: "MUL_MAT",
    30: "MUL_MAT_ID",
}

GGML_TYPE = {
    0: "f32",
    1: "f16",
    12: "q4_K",
    13: "q5_K",
    14: "q6_K",
    26: "i32",
}


@dataclass
class Tensor:
    typ: int
    ne: list[int]
    nb: list[int]

    @property
    def type_name(self) -> str:
        return GGML_TYPE.get(self.typ, str(self.typ))


@dataclass
class ExportedOp:
    line: str
    op: int
    typ: int
    ne: list[int]
    params: list[int]
    sources: list[Tensor]
    name: str

    @property
    def op_name(self) -> str:
        return GGML_OP.get(self.op, str(self.op))

    @property
    def type_name(self) -> str:
        return GGML_TYPE.get(self.typ, str(self.typ))

    def summary(self) -> dict[str, object]:
        return {
            "op": self.op_name,
            "type": self.type_name,
            "ne": self.ne,
            "name": self.name,
            "sources": [
                {
                    "type": src.type_name,
                    "ne": src.ne,
                    "nb": src.nb,
                }
                for src in self.sources
            ],
        }


def parse_line(line: str) -> ExportedOp:
    parts = line.rstrip("\n").split()
    i = 0
    op = int(parts[i]); i += 1
    typ = int(parts[i]); i += 1
    ne = [int(parts[i + j]) for j in range(4)]; i += 4
    n_params = int(parts[i]); i += 1
    params = [int(parts[i + j]) for j in range(n_params)]; i += n_params
    n_sources = int(parts[i]); i += 1
    sources: list[Tensor] = []
    for _ in range(n_sources):
        src_type = int(parts[i]); i += 1
        src_ne = [int(parts[i + j]) for j in range(4)]; i += 4
        src_nb = [int(parts[i + j]) for j in range(4)]; i += 4
        sources.append(Tensor(src_type, src_ne, src_nb))
    name = " ".join(parts[i:]) if i < len(parts) else ""
    if name == "-":
        name = ""
    return ExportedOp(line=line, op=op, typ=typ, ne=ne, params=params, sources=sources, name=name)


def match_family(op: ExportedOp, family: str) -> bool:
    if family == "qk_prompt":
        return (
            op.op_name == "MUL_MAT"
            and len(op.sources) >= 2
            and op.sources[0].type_name in {"q4_K", "q5_K", "q6_K"}
            and op.sources[1].type_name == "f32"
            and op.ne[1] > 1
        )
    if family == "qk_decode":
        return (
            op.op_name == "MUL_MAT"
            and len(op.sources) >= 2
            and op.sources[0].type_name in {"q4_K", "q5_K", "q6_K"}
            and op.sources[1].type_name == "f32"
            and op.ne[1] == 1
        )
    if family == "moe_qk_prompt":
        return (
            op.op_name == "MUL_MAT_ID"
            and len(op.sources) >= 3
            and op.sources[0].type_name in {"q4_K", "q5_K", "q6_K"}
            and op.sources[1].type_name == "f32"
            and op.ne[2] > 1
        )
    if family == "attention_f16_prompt":
        return (
            op.op_name == "MUL_MAT"
            and len(op.sources) >= 2
            and op.sources[0].type_name == "f16"
            and op.sources[1].type_name == "f32"
            and (op.ne[1] > 1 or op.ne[2] > 1)
        )
    raise ValueError(f"unknown family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--family",
        action="append",
        choices=["qk_prompt", "qk_decode", "moe_qk_prompt", "attention_f16_prompt"],
        help="Family to emit. May be repeated. Defaults to all known families.",
    )
    args = parser.parse_args()

    families = args.family or ["qk_prompt", "qk_decode", "moe_qk_prompt", "attention_f16_prompt"]
    ops = [parse_line(line) for line in args.input.read_text().splitlines() if line.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "input": str(args.input),
        "total_ops": len(ops),
        "families": {},
    }

    for family in families:
        selected = [op for op in ops if match_family(op, family)]
        text = "\n".join(op.line.rstrip("\n") for op in selected)
        if text:
            text += "\n"
        (args.out_dir / f"{family}.txt").write_text(text)
        (args.out_dir / f"{family}.json").write_text(
            json.dumps([op.summary() for op in selected], indent=2) + "\n"
        )
        manifest["families"][family] = {
            "count": len(selected),
            "test_file": str(args.out_dir / f"{family}.txt"),
            "summary_file": str(args.out_dir / f"{family}.json"),
        }

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
