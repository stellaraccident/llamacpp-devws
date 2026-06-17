#!/usr/bin/env python3
"""Run a small apples-to-apples HRX2 vs Vulkan Phase 2a baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = WORKSPACE / "cache" / "hrx2" / "phase2a"
DEFAULT_MODELS = [
    (
        "phi4-mini-q4",
        "shared/models/llamacpp-hrx2-basket-v1/bartowski__microsoft_Phi-4-mini-instruct-GGUF/"
        "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    ),
    (
        "llama32-3b-q4",
        "shared/models/llamacpp-hrx2-basket-v1/bartowski__Llama-3.2-3B-Instruct-GGUF/"
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    (
        "llama31-8b-q4",
        "shared/models/llamacpp-hrx2-basket-v1/bartowski__Meta-Llama-3.1-8B-Instruct-GGUF/"
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
]
DEFAULT_CASES = [
    ("decode-p1n64", 1, 64, 512, 512),
    ("prefill-p64n0", 64, 0, 512, 512),
    ("prefill-p512n0", 512, 0, 512, 512),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="")
    parser.add_argument("--models", default=",".join(name for name, _ in DEFAULT_MODELS))
    parser.add_argument("--cases", default=",".join(name for name, *_ in DEFAULT_CASES))
    parser.add_argument("--backends", default="hrx2,vulkan")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--flash-attn", choices=("0", "1"), default="0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def env_for_backend(backend: str, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    ld_parts: list[str] = []
    if backend == "hrx2":
        ld_parts.extend([
            str(WORKSPACE / "build" / "hrx-install" / "lib"),
            str(WORKSPACE / "build" / "hrx-install" / "lib64"),
            str(WORKSPACE / "rocm" / "lib"),
            str(WORKSPACE / "rocm" / "lib" / "rocm_sysdeps" / "lib"),
            str(WORKSPACE / "build" / "llama-hrx2" / "bin"),
        ])
        env["ROCM_PATH"] = str(WORKSPACE / "rocm")
        env["GGML_HRX_ROCM_PATH"] = str(WORKSPACE / "rocm")
        env["GGML_HRX2_TRACE_JSONL"] = str(run_dir / "hrx2.jsonl")
        env["GGML_SCHED_TRACE_JSONL"] = str(run_dir / "sched.jsonl")
    elif backend == "vulkan":
        ld_parts.append(str(WORKSPACE / "build" / "llama-vulkan" / "bin"))
        env["GGML_VK_PERF_LOGGER"] = "1"
        env["GGML_SCHED_TRACE_JSONL"] = str(run_dir / "sched.jsonl")
    else:
        raise ValueError(f"unknown backend {backend}")
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def bench_bin(backend: str) -> Path:
    if backend == "hrx2":
        return WORKSPACE / "build" / "llama-hrx2" / "bin" / "llama-bench"
    if backend == "vulkan":
        return WORKSPACE / "build" / "llama-vulkan" / "bin" / "llama-bench"
    raise ValueError(f"unknown backend {backend}")


def selected_models(names: set[str]) -> list[tuple[str, Path]]:
    models = []
    known = {name: rel for name, rel in DEFAULT_MODELS}
    for name in names:
        if name not in known:
            raise SystemExit(f"unknown model {name}; known: {', '.join(known)}")
    for name, rel in DEFAULT_MODELS:
        if name in names:
            path = WORKSPACE / rel
            if not path.exists():
                raise SystemExit(f"model is missing: {path}")
            models.append((name, path))
    return models


def selected_cases(names: set[str]) -> list[tuple[str, int, int, int, int]]:
    cases = []
    known = {name: rest for name, *rest in DEFAULT_CASES}
    for name in names:
        if name not in known:
            raise SystemExit(f"unknown case {name}; known: {', '.join(known)}")
    for row in DEFAULT_CASES:
        if row[0] in names:
            cases.append(row)
    return cases


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reduce_hrx2(run_dir: Path) -> dict[str, Any]:
    sched = load_jsonl(run_dir / "sched.jsonl")
    hrx = load_jsonl(run_dir / "hrx2.jsonl")
    compute = [row for row in sched if row.get("event") == "sched_node" and row.get("compute")]
    cpu = [row for row in compute if row.get("is_cpu")]
    route_counts = Counter(row.get("route_id", "") for row in hrx if row.get("event") == "dispatch")
    event_counts = Counter(row.get("event", "") for row in hrx)
    return {
        "sched_compute_nodes": len(compute),
        "sched_cpu_compute_nodes": len(cpu),
        "hrx2_dispatches": sum(route_counts.values()),
        "hrx2_unique_routes": len(route_counts),
        "hrx2_event_counts": dict(event_counts),
        "hrx2_top_routes": [{"route_id": route, "count": count} for route, count in route_counts.most_common(20)],
    }


def bench_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def row_for_regime(rows: list[dict[str, Any]], prompt: int, gen: int) -> dict[str, Any] | None:
    # llama-bench decomposes p+n runs into prefill and generation rows.
    want_prompt = prompt if gen == 0 else 0
    want_gen = gen if gen != 0 else 0
    if gen == 0:
        for row in rows:
            if row.get("n_prompt") == prompt and row.get("n_gen") == 0:
                return row
    for row in rows:
        if row.get("n_prompt") == want_prompt and row.get("n_gen") == want_gen:
            return row
    return rows[-1] if rows else None


def sample_summary(bench: dict[str, Any]) -> dict[str, Any]:
    samples_ts = [float(v) for v in bench.get("samples_ts") or []]
    samples_ns = [int(v) for v in bench.get("samples_ns") or []]
    steady_ts = sum(samples_ts[1:]) / len(samples_ts[1:]) if len(samples_ts) > 1 else (samples_ts[0] if samples_ts else None)
    steady_ns = int(sum(samples_ns[1:]) / len(samples_ns[1:])) if len(samples_ns) > 1 else (samples_ns[0] if samples_ns else None)
    return {
        "samples_ts": samples_ts,
        "samples_ns": samples_ns,
        "cold_ts": samples_ts[0] if samples_ts else None,
        "cold_ns": samples_ns[0] if samples_ns else None,
        "steady_ts": steady_ts,
        "steady_ns": steady_ns,
    }


def run_one(args: argparse.Namespace, backend: str, model_name: str, model_path: Path, case: tuple[str, int, int, int, int], root: Path) -> dict[str, Any]:
    case_name, prompt, gen, batch, ubatch = case
    run_dir = root / backend / model_name / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout = run_dir / "llama-bench.json"
    stderr = run_dir / "stderr.txt"
    command = [
        str(bench_bin(backend)),
        "-m", str(model_path),
        "-p", str(prompt),
        "-n", str(gen),
        "-b", str(batch),
        "-ub", str(ubatch),
        "-fa", args.flash_attn,
        "-r", str(args.repetitions),
        "-o", "json",
        "--no-warmup",
        "-ngl", "99",
    ]
    if backend == "hrx2":
        command.extend(["-dev", "HRX20"])
    record = {
        "backend": backend,
        "model": model_name,
        "case": case_name,
        "command": command,
        "run_dir": str(run_dir),
    }
    if args.dry_run:
        print(" ".join(command))
        record["status"] = "dry-run"
        return record
    if args.skip_existing and stdout.exists() and stdout.stat().st_size > 0:
        status = 0
    else:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                command,
                cwd=WORKSPACE,
                env=env_for_backend(backend, run_dir),
                stdout=out,
                stderr=err,
                timeout=args.timeout,
                check=False,
            )
        status = proc.returncode
    record["status"] = status
    rows = bench_rows(stdout)
    bench = row_for_regime(rows, prompt, gen)
    if bench:
        record.update({
            "build_commit": bench.get("build_commit"),
            "avg_ts": bench.get("avg_ts"),
            "avg_ns": bench.get("avg_ns"),
            "bench_rows": len(rows),
        })
        record.update(sample_summary(bench))
    if backend == "hrx2":
        record.update(reduce_hrx2(run_dir))
    return record


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def fmt_samples(values: Any) -> str:
    if not values:
        return ""
    return ", ".join(fmt_float(value, 1) for value in values)


def write_report(root: Path, records: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault((record["model"], record["case"]), {})[record["backend"]] = record
    lines = [
        "# HRX2 Phase 2a Baseline",
        "",
        "| Model | Case | HRX2 avg tok/s | Vulkan avg tok/s | Avg ratio | HRX2 steady tok/s | Vulkan steady tok/s | Steady ratio | HRX2 cold tok/s | Vulkan cold tok/s | HRX2 samples tok/s | Vulkan samples tok/s | HRX2 dispatches | CPU compute | Top blocker |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for (model, case), backends in sorted(by_key.items()):
        hrx = backends.get("hrx2", {})
        vk = backends.get("vulkan", {})
        hrx_ts = float(hrx.get("avg_ts") or 0.0)
        vk_ts = float(vk.get("avg_ts") or 0.0)
        ratio = hrx_ts / vk_ts if vk_ts else 0.0
        hrx_steady = float(hrx.get("steady_ts") or 0.0)
        vk_steady = float(vk.get("steady_ts") or 0.0)
        steady_ratio = hrx_steady / vk_steady if vk_steady else 0.0
        top = ""
        routes = hrx.get("hrx2_top_routes") or []
        if routes:
            top = f"{routes[0]['route_id']} x{routes[0]['count']}"
        lines.append(
            f"| `{model}` | `{case}` | {hrx_ts:.3f} | {vk_ts:.3f} | {ratio:.4f} | "
            f"{fmt_float(hrx.get('steady_ts'))} | {fmt_float(vk.get('steady_ts'))} | {steady_ratio:.4f} | "
            f"{fmt_float(hrx.get('cold_ts'))} | {fmt_float(vk.get('cold_ts'))} | "
            f"`{fmt_samples(hrx.get('samples_ts'))}` | `{fmt_samples(vk.get('samples_ts'))}` | "
            f"{int(hrx.get('hrx2_dispatches') or 0)} | {int(hrx.get('sched_cpu_compute_nodes') or 0)} | `{top}` |"
        )
    root.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    root.joinpath("summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tag = args.tag or subprocess.check_output(["date", "+%Y%m%d-%H%M%S"], text=True).strip()
    root = args.out / tag
    root.mkdir(parents=True, exist_ok=True)
    models = selected_models(set(filter(None, args.models.split(","))))
    cases = selected_cases(set(filter(None, args.cases.split(","))))
    backends = list(filter(None, args.backends.split(",")))
    records: list[dict[str, Any]] = []
    for model_name, model_path in models:
        for case in cases:
            for backend in backends:
                print(f"[{backend}] {model_name} {case[0]}", flush=True)
                records.append(run_one(args, backend, model_name, model_path, case, root))
                write_report(root, records)
    print(root)


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as exc:
        print(f"timed out: {exc}", file=sys.stderr)
        raise SystemExit(124)
