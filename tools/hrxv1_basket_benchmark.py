#!/usr/bin/env python3
"""Run serial HRX v1 vs Vulkan llama-bench rows for the local GGUF basket."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = WORKSPACE / "shared/models/llamacpp-hrx2-basket-v1"
DEFAULT_OUT_ROOT = WORKSPACE / "cache/hrxv1/gfx1151"
DEFAULT_CASES = {
    "p33": (33, 0, 33, 33),
    "p512": (512, 0, 512, 512),
    "p513": (513, 0, 1024, 1024),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--tag", default=time.strftime("basket-head-%Y%m%d-%H%M%S"))
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--models", default="all", help="comma-separated slugs or all")
    parser.add_argument("--cases", default="p33,p512,p513", help="comma-separated case names")
    parser.add_argument("--backends", default="hrx,vulkan", help="hrx,vulkan or subset")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--flash-attn", choices=("0", "1"), default="1")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug_for_model(path: Path) -> str:
    rel = path.relative_to(DEFAULT_MODELS_ROOT) if path.is_relative_to(DEFAULT_MODELS_ROOT) else path
    parent = re.sub(r"[^A-Za-z0-9]+", "-", rel.parent.name).strip("-").lower()
    stem = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").lower()
    if parent and parent not in stem:
        return f"{parent}-{stem}"
    return stem


def discover_models(root: Path) -> list[tuple[str, Path]]:
    models = [(slug_for_model(path), path) for path in sorted(root.rglob("*.gguf"))]
    seen: Counter[str] = Counter(slug for slug, _ in models)
    out: list[tuple[str, Path]] = []
    for slug, path in models:
        if seen[slug] > 1:
            digest = str(abs(hash(str(path))) % 100000)
            slug = f"{slug}-{digest}"
        out.append((slug, path))
    return out


def selected_models(args: argparse.Namespace) -> list[tuple[str, Path]]:
    models = discover_models(args.models_root)
    if args.models == "all":
        return models
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    by_slug = {slug: path for slug, path in models}
    missing = sorted(requested - set(by_slug))
    if missing:
        known = "\n  ".join(sorted(by_slug))
        raise SystemExit(f"unknown model slug(s): {', '.join(missing)}\nknown:\n  {known}")
    return [(slug, by_slug[slug]) for slug in sorted(requested)]


def selected_cases(args: argparse.Namespace) -> list[tuple[str, int, int, int, int]]:
    names = [item.strip() for item in args.cases.split(",") if item.strip()]
    missing = sorted(set(names) - set(DEFAULT_CASES))
    if missing:
        raise SystemExit(f"unknown case(s): {', '.join(missing)}; known: {', '.join(DEFAULT_CASES)}")
    return [(name, *DEFAULT_CASES[name]) for name in names]


def bench_bin(backend: str) -> Path:
    if backend == "hrx":
        return WORKSPACE / "build/hrx-v1-catalog-gfx1151/bin/llama-bench"
    if backend == "vulkan":
        return WORKSPACE / "build/vulkan-gfx1151/bin/llama-bench"
    raise ValueError(backend)


def env_for_backend(backend: str) -> dict[str, str]:
    env = os.environ.copy()
    ld_parts: list[str] = []
    if backend == "hrx":
        env["ROCM_PATH"] = str(WORKSPACE / "rocm")
        env["GGML_HRX_ROCM_PATH"] = str(WORKSPACE / "rocm")
        env["GGML_HRX_TRACE_ROUTES"] = "1"
        env["GGML_HRX_TRACE_PROVIDERS"] = "1"
        ld_parts.append(str(WORKSPACE / "build/hrx-v1-catalog-gfx1151/bin"))
    elif backend == "vulkan":
        env["GGML_VK_PERF_LOGGER"] = "1"
        ld_parts.append(str(WORKSPACE / "build/vulkan-gfx1151/bin"))
    ld_parts.extend([
        str(WORKSPACE / "rocm/lib"),
        str(WORKSPACE / "rocm/lib/rocm_sysdeps/lib"),
    ])
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def row_for_case(rows: list[dict[str, Any]], prompt: int, gen: int) -> dict[str, Any] | None:
    for row in rows:
        if row.get("n_prompt") == prompt and row.get("n_gen") == gen:
            return row
    return rows[-1] if rows else None


def route_counts(stderr_path: Path) -> tuple[list[dict[str, Any]], int]:
    providers: Counter[str] = Counter()
    fallback_lines = 0
    if not stderr_path.exists():
        return [], 0
    for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "provider=" in line:
            match = re.search(r"provider=([^\s]+)", line)
            if match:
                providers[match.group(1)] += 1
        low = line.lower()
        if "fallback" in low or "cpu" in low:
            fallback_lines += 1
    return [{"provider": key, "count": value} for key, value in providers.most_common(20)], fallback_lines


def summarize_samples(row: dict[str, Any]) -> dict[str, Any]:
    samples = [float(value) for value in row.get("samples_ts") or []]
    steady = sum(samples[1:]) / len(samples[1:]) if len(samples) > 1 else (samples[0] if samples else None)
    return {
        "avg_ts": row.get("avg_ts"),
        "stddev_ts": row.get("stddev_ts"),
        "samples_ts": samples,
        "steady_ts": steady,
        "cold_ts": samples[0] if samples else None,
        "backends": row.get("backends"),
        "devices": row.get("devices"),
        "build_commit": row.get("build_commit"),
    }


def run_one(
    args: argparse.Namespace,
    root: Path,
    backend: str,
    model_slug: str,
    model_path: Path,
    case: tuple[str, int, int, int, int],
) -> dict[str, Any]:
    case_name, prompt, gen, batch, ubatch = case
    run_dir = root / backend / model_slug / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout = run_dir / "llama-bench.json"
    stderr = run_dir / "stderr.log"
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
        "-dev", "HRX0" if backend == "hrx" else "Vulkan0",
    ]
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    record: dict[str, Any] = {
        "backend": backend,
        "model": model_slug,
        "model_path": str(model_path),
        "case": case_name,
        "prompt": prompt,
        "gen": gen,
        "batch": batch,
        "ubatch": ubatch,
        "run_dir": str(run_dir),
        "command": command,
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
                env=env_for_backend(backend),
                stdout=out,
                stderr=err,
                timeout=args.timeout,
                check=False,
            )
        status = proc.returncode
    record["status"] = status
    rows = read_rows(stdout)
    row = row_for_case(rows, prompt, gen)
    record["bench_rows"] = len(rows)
    if row:
        record.update(summarize_samples(row))
    if backend == "hrx":
        top_routes, fallback_lines = route_counts(stderr)
        record["top_routes"] = top_routes
        record["fallback_lines"] = fallback_lines
    return record


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_summary(root: Path, records: list[dict[str, Any]]) -> None:
    (root / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault((record["model"], record["case"]), {})[record["backend"]] = record

    ratios = []
    steady_ratios = []
    lines = [
        "# HRX v1 Basket Benchmark",
        "",
        "| Model | Case | HRX avg tok/s | Vulkan avg tok/s | Avg ratio | HRX steady tok/s | Vulkan steady tok/s | Steady ratio | HRX fallback lines | HRX top route |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    summary_rows = []
    for (model, case), backends in sorted(by_key.items()):
        hrx = backends.get("hrx", {})
        vk = backends.get("vulkan", {})
        ratio = None
        steady_ratio = None
        if hrx.get("avg_ts") and vk.get("avg_ts"):
            ratio = float(hrx["avg_ts"]) / float(vk["avg_ts"])
            ratios.append(ratio)
        if hrx.get("steady_ts") and vk.get("steady_ts"):
            steady_ratio = float(hrx["steady_ts"]) / float(vk["steady_ts"])
            steady_ratios.append(steady_ratio)
        top_route = ""
        if hrx.get("top_routes"):
            route = hrx["top_routes"][0]
            top_route = f"{route['provider']} ({route['count']})"
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    case,
                    fmt(hrx.get("avg_ts")),
                    fmt(vk.get("avg_ts")),
                    fmt(ratio),
                    fmt(hrx.get("steady_ts")),
                    fmt(vk.get("steady_ts")),
                    fmt(steady_ratio),
                    str(hrx.get("fallback_lines", "")),
                    top_route,
                ]
            )
            + " |"
        )
        summary_rows.append({
            "model": model,
            "case": case,
            "hrx_avg_ts": hrx.get("avg_ts"),
            "vulkan_avg_ts": vk.get("avg_ts"),
            "avg_ratio": ratio,
            "hrx_steady_ts": hrx.get("steady_ts"),
            "vulkan_steady_ts": vk.get("steady_ts"),
            "steady_ratio": steady_ratio,
            "hrx_fallback_lines": hrx.get("fallback_lines"),
            "hrx_top_routes": hrx.get("top_routes"),
        })

    geomean = math.prod(ratios) ** (1 / len(ratios)) if ratios else None
    steady_geomean = math.prod(steady_ratios) ** (1 / len(steady_ratios)) if steady_ratios else None
    summary = {
        "geomean_avg_ratio": geomean,
        "geomean_steady_ratio": steady_geomean,
        "rows": summary_rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines.extend([
        "",
        f"- avg geomean HRX/Vulkan: `{fmt(geomean)}`",
        f"- steady geomean HRX/Vulkan: `{fmt(steady_geomean)}`",
        "",
    ])
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.out / args.tag
    root.mkdir(parents=True, exist_ok=True)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    for backend in backends:
        if backend not in {"hrx", "vulkan"}:
            raise SystemExit(f"unknown backend {backend}")
        if not bench_bin(backend).exists():
            raise SystemExit(f"missing bench binary: {bench_bin(backend)}")

    records: list[dict[str, Any]] = []
    for model_slug, model_path in selected_models(args):
        for case in selected_cases(args):
            for backend in backends:
                print(f"running {backend} {model_slug} {case[0]}", flush=True)
                records.append(run_one(args, root, backend, model_slug, model_path, case))
                write_summary(root, records)
    write_summary(root, records)
    print(root)


if __name__ == "__main__":
    main()
