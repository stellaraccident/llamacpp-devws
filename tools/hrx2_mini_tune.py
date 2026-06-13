#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from hrx2_pipeline_lib import DEFAULT_HRX_INSTALL, WORKSPACE, env_for_tools, read_jsonl, write_json


DEFAULT_SOURCE = WORKSPACE / "sources" / "llama.cpp" / "ggml" / "src" / "ggml-hrx2" / "kernels" / "rms_norm_f32.loom"
DEFAULT_OUT_ROOT = WORKSPACE / "cache" / "hrx2" / "mini_tune"
DEFAULT_BENCHMARKS = [
    "hrx2_rms_norm_f32_decode",
    "hrx2_rms_norm_f32_small_prefill",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny HRX2 Loom tuning loop for RMS_NORM.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--run-id", default="phase0.4-mini")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--iree-benchmark-loom", default=str(DEFAULT_HRX_INSTALL / "bin" / "iree-benchmark-loom"))
    parser.add_argument("--workgroup-sizes", default="128,256,512")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--input-ring-count", type=int, default=1)
    return parser.parse_args()


def split_csv_ints(text):
    return [int(part) for part in text.split(",") if part.strip()]


def split_csv_symbols(text):
    return [part.strip().lstrip("@") for part in text.split(",") if part.strip()]


def generate_variant(source_path, variant_path, workgroup_size):
    text = source_path.read_text(encoding="utf-8")
    needle = "%workgroup_size = index.constant 512 : index"
    count = text.count(needle)
    if count != 2:
        raise RuntimeError(f"expected two RMS_NORM workgroup-size constants in {source_path}, found {count}")
    text = text.replace(needle, f"%workgroup_size = index.constant {workgroup_size} : index")
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variant_path.write_text(text, encoding="utf-8")


def run_command(cmd, timeout):
    try:
        completed = subprocess.run(
            cmd,
            cwd=WORKSPACE,
            env=env_for_tools(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def extract_rows(jsonl_path):
    rows = read_jsonl(jsonl_path)
    plan = next((row for row in rows if row.get("row") == "plan"), None)
    compile_row = next((row for row in rows if row.get("row") == "compile"), None)
    benchmark_row = next((row for row in rows if row.get("row") == "benchmark"), None)
    summary = next((row for row in rows if row.get("row") == "summary"), None)
    failures = [row for row in rows if row.get("row") == "failure"]
    return plan, compile_row, benchmark_row, summary, failures


def summarize_compile(compile_row):
    if not compile_row:
        return {}
    static = compile_row.get("static_summary") or {}
    return {
        "status": compile_row.get("status"),
        "sample_constant_argument_count": compile_row.get("sample_constant_argument_count"),
        "artifact_size": static.get("artifact_size"),
        "instruction_count": static.get("instruction_count"),
        "code_byte_count": static.get("code_byte_count"),
        "private_memory_bytes": static.get("private_memory_bytes"),
        "local_memory_bytes": static.get("local_memory_bytes"),
        "allocation_spill_count": static.get("allocation_spill_count"),
        "register_pressure_peak_live_units": static.get("register_pressure_peak_live_units"),
        "vector_alu_count": static.get("vector_alu_count"),
        "global_memory_count": static.get("global_memory_count"),
    }


def summarize_benchmark(benchmark_row):
    if not benchmark_row:
        return {}
    result = benchmark_row.get("benchmark_result") or {}
    timing = result.get("operation_timing_ns") or {}
    return {
        "status": result.get("status"),
        "p50_ns": timing.get("p50"),
        "p90_ns": timing.get("p90"),
        "mean_ns": timing.get("mean"),
        "count": timing.get("count"),
        "measured_dispatch_count": result.get("measured_dispatch_count"),
        "stop_reason": result.get("stop_reason"),
        "sample": benchmark_row.get("sample"),
        "data_cache": result.get("data_cache"),
    }


def write_summary(run_dir, results):
    winners = {}
    for result in results:
        bench = result["benchmark"]
        timing = result.get("benchmark_result") or {}
        p50 = timing.get("p50_ns")
        if result["status"] != "ok" or p50 is None:
            continue
        current = winners.get(bench)
        if current is None or p50 < current["benchmark_result"]["p50_ns"]:
            winners[bench] = result

    summary = {
        "schema": "hrx2-mini-tune-summary-v1",
        "run_dir": str(run_dir),
        "result_count": len(results),
        "ok_count": sum(1 for item in results if item["status"] == "ok"),
        "winners": {
            bench: {
                "workgroup_size": item["workgroup_size"],
                "p50_ns": item["benchmark_result"]["p50_ns"],
                "p90_ns": item["benchmark_result"]["p90_ns"],
                "compile": item["compile_result"],
            }
            for bench, item in winners.items()
        },
    }
    write_json(run_dir / "summary.json", summary)

    lines = [
        "# HRX2 Mini Tune",
        "",
        f"- Run: `{run_dir}`",
        f"- Results: {summary['ok_count']}/{summary['result_count']} ok",
        "",
        "| Benchmark | WG | Status | p50 ns | p90 ns | Inst | Code bytes | Spills | Peak live |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(results, key=lambda row: (row["benchmark"], row["workgroup_size"])):
        bench = item["benchmark"]
        timing = item.get("benchmark_result") or {}
        compile_result = item.get("compile_result") or {}
        lines.append(
            f"| `{bench}` | {item['workgroup_size']} | {item['status']} | "
            f"{timing.get('p50_ns', '')} | {timing.get('p90_ns', '')} | "
            f"{compile_result.get('instruction_count', '')} | {compile_result.get('code_byte_count', '')} | "
            f"{compile_result.get('allocation_spill_count', '')} | "
            f"{compile_result.get('register_pressure_peak_live_units', '')} |"
        )
    lines.extend(["", "## Winners", ""])
    for bench, item in sorted(winners.items()):
        lines.append(
            f"- `{bench}`: WG {item['workgroup_size']} at "
            f"{item['benchmark_result']['p50_ns']} ns p50"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    args = parse_args()
    source_path = Path(args.source)
    workgroup_sizes = split_csv_ints(args.workgroup_sizes)
    benchmarks = split_csv_symbols(args.benchmarks)
    run_dir = Path(args.out_root) / args.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    variants_dir = run_dir / "variants"
    results_dir = run_dir / "benchmark_jsonl"
    variants_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    results = []
    for workgroup_size in workgroup_sizes:
        variant_path = variants_dir / f"rms_norm_f32_wg{workgroup_size}.loom"
        generate_variant(source_path, variant_path, workgroup_size)
        for benchmark in benchmarks:
            output_path = results_dir / f"{benchmark}_wg{workgroup_size}.jsonl"
            cmd = [
                str(args.iree_benchmark_loom),
                str(variant_path),
                "--device=amdgpu",
                f"--benchmark=@{benchmark}",
                "--measure=dispatch_complete",
                "--sample=0",
                "--sample-compilation=per_sample",
                f"--iterations={args.iterations}",
                f"--warmup-iterations={args.warmup_iterations}",
                "--min-time-ms=0",
                f"--max-batches={args.iterations}",
                "--stable-p90-to-p50-ppm=0",
                f"--input-ring-count={args.input_ring_count}",
                "--compile-report=summary",
                "--output-format=jsonl",
                f"--output={output_path}",
            ]
            run = run_command(cmd, args.timeout)
            plan, compile_row, benchmark_row, summary_row, failures = extract_rows(output_path)
            compile_result = summarize_compile(compile_row)
            benchmark_result = summarize_benchmark(benchmark_row)
            status = "ok" if run["returncode"] == 0 and benchmark_result.get("status") == "ok" else "failed"
            if run["timed_out"]:
                status = "timeout"
            result = {
                "schema": "hrx2-mini-tune-result-v1",
                "run_id": args.run_id,
                "source": str(source_path),
                "variant_source": str(variant_path),
                "workgroup_size": workgroup_size,
                "benchmark": benchmark,
                "status": status,
                "command": cmd,
                "returncode": run["returncode"],
                "timed_out": run["timed_out"],
                "stdout": run["stdout"],
                "stderr": run["stderr"],
                "output_path": str(output_path),
                "plan": plan,
                "compile_result": compile_result,
                "benchmark_result": benchmark_result,
                "summary": summary_row.get("summary") if summary_row else None,
                "failures": failures,
            }
            results.append(result)
            print(
                f"{benchmark} wg={workgroup_size} status={status} "
                f"p50={benchmark_result.get('p50_ns', '')}",
                file=sys.stderr,
            )

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    summary = write_summary(run_dir, results)
    print(f"wrote {results_path}", file=sys.stderr)
    print(f"wrote {run_dir / 'summary.md'}", file=sys.stderr)
    if summary["ok_count"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
