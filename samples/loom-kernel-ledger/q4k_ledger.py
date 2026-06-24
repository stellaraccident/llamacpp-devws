#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
"""Generate a Loom tuning ledger for the HRX2 Q4_K x F32 MUL_MAT kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


QK_K = 256
Q4_K_BLOCK_BYTES = 144
F32_BYTES = 4

DEFAULT_K_VALUES = [256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 11008, 14336, 16384, 28672, 32768]
DEFAULT_ROW_VALUES = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 8192, 11008, 14336, 16384, 28672, 32768]
DEFAULT_COL_VALUES = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    route: dict[str, Any]
    k: int
    rows: int
    cols: int
    config: dict[str, str]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_short(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def route_short_name(route: dict[str, Any]) -> str:
    root = route.get("root_symbol", "")
    if "splitk2" in root:
        return "splitk2"
    if "wmma64x64" in root:
        return "wmma64"
    if "wmma128x128" in root:
        return "wmma128"
    if root.endswith("_static"):
        return "direct"
    return re.sub(r"[^a-z0-9]+", "_", route.get("id", "route").lower()).strip("_")


def parse_routes(route_path: Path, route_filter: set[str] | None) -> list[dict[str, Any]]:
    with route_path.open("r", encoding="utf-8") as f:
        routes = json.load(f)
    selected = []
    for route in routes:
        if route.get("op") != "MUL_MAT":
            continue
        supports = route.get("supports", {})
        if supports.get("src0_type") != "Q4_K":
            continue
        if supports.get("src1_type") != "F32" or supports.get("dst_type") != "F32":
            continue
        if supports.get("rhs_type"):
            continue
        if route_filter and route_short_name(route) not in route_filter and route.get("id") not in route_filter:
            continue
        selected.append(route)
    selected.sort(key=lambda item: item.get("priority", 0), reverse=True)
    return selected


def is_shape_valid(route: dict[str, Any], k: int, rows: int, cols: int) -> bool:
    domain = route.get("shape_domain", {})
    guards = route.get("shape_guards", {})
    if not (domain.get("k_min", k) <= k <= domain.get("k_max", k)):
        return False
    if not (domain.get("rows_min", rows) <= rows <= domain.get("rows_max", rows)):
        return False
    if not (domain.get("cols_min", cols) <= cols <= domain.get("cols_max", cols)):
        return False
    if guards.get("k_multiple_of") and k % int(guards["k_multiple_of"]) != 0:
        return False
    if guards.get("rows_multiple_of") and rows % int(guards["rows_multiple_of"]) != 0:
        return False
    if guards.get("cols_multiple_of") and cols % int(guards["cols_multiple_of"]) != 0:
        return False
    return True


def values_for_route(route: dict[str, Any], user_values: list[int] | None, defaults: list[int], key: str) -> list[int]:
    domain = route.get("shape_domain", {})
    values = user_values or defaults
    min_value = domain.get(f"{key}_min", min(values))
    max_value = domain.get(f"{key}_max", max(values))
    guard = route.get("shape_guards", {}).get(f"{key}_multiple_of")
    out = []
    for value in values:
        if value < min_value or value > max_value:
            continue
        if guard and value % int(guard) != 0:
            continue
        out.append(value)
    if min_value == max_value and min_value not in out:
        out.append(int(min_value))
    return sorted(set(out))


def build_config(route: dict[str, Any], k: int, rows: int, cols: int) -> dict[str, str]:
    values = {"shape.k": str(k), "shape.rows": str(rows), "shape.cols": str(cols)}
    config: dict[str, str] = {}
    for binding in route.get("specialization", {}).get("bindings", []):
        key = binding["key"]
        if "source" in binding:
            config[key] = values[binding["source"]]
        else:
            config[key] = str(binding["value"])
    return config


def generate_candidates(
    routes: list[dict[str, Any]],
    k_values: list[int] | None,
    row_values: list[int] | None,
    col_values: list[int] | None,
    limit: int | None,
) -> list[Candidate]:
    candidates = []
    for route in routes:
        for k in values_for_route(route, k_values, DEFAULT_K_VALUES, "k"):
            for row in values_for_route(route, row_values, DEFAULT_ROW_VALUES, "rows"):
                for col in values_for_route(route, col_values, DEFAULT_COL_VALUES, "cols"):
                    if not is_shape_valid(route, k, row, col):
                        continue
                    route_name = route_short_name(route)
                    raw_id = f"{route_name}_k{k}_r{row}_c{col}_{route.get('root_symbol')}"
                    candidates.append(
                        Candidate(
                            candidate_id=f"{route_name}_k{k}_r{row}_c{col}_{sha256_short(raw_id, 8)}",
                            route=route,
                            k=k,
                            rows=row,
                            cols=col,
                            config=build_config(route, k, row, col),
                        )
                    )
                    if limit and len(candidates) >= limit:
                        return candidates
    return candidates


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def q4_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q4_K_BLOCK_BYTES


def shape_stats(candidate: Candidate) -> dict[str, Any]:
    src0_bytes = q4_bytes(candidate.k, candidate.rows)
    src1_bytes = candidate.k * candidate.cols * F32_BYTES
    dst_bytes = candidate.rows * candidate.cols * F32_BYTES
    transfer = src0_bytes + src1_bytes + dst_bytes
    ops = 2 * candidate.k * candidate.rows * candidate.cols
    return {
        "k": candidate.k,
        "rows": candidate.rows,
        "cols": candidate.cols,
        "q4_blocks": candidate.rows * (candidate.k // QK_K),
        "src0_bytes": src0_bytes,
        "src1_bytes": src1_bytes,
        "dst_bytes": dst_bytes,
        "total_transfer_bytes": transfer,
        "estimated_fma_ops": ops,
        "estimated_arithmetic_intensity_ops_per_byte": ops / transfer if transfer else None,
    }


def route_launch(candidate: Candidate) -> dict[str, Any]:
    dispatch = candidate.route.get("dispatch", {})
    rows_per_workgroup = int(dispatch.get("rows_per_workgroup", 1))
    cols_per_workgroup = int(dispatch.get("cols_per_workgroup", 1))
    return {
        "workgroup_count": [ceil_div(candidate.rows, rows_per_workgroup), ceil_div(candidate.cols, cols_per_workgroup), 1],
        "workgroup_size": dispatch.get("workgroup_size", [None, None, None]),
        "rows_per_workgroup": rows_per_workgroup,
        "cols_per_workgroup": cols_per_workgroup,
        "metadata_source": "route_heuristic",
        "has_static_dispatch_workgroup_count": False,
        "has_static_workgroup_size": False,
    }


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(args.hrx_install / "lib64"), str(args.hrx_install / "lib"), str(args.rocm / "lib")]
    old_ld = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join(paths + ([old_ld] if old_ld else []))
    env.setdefault("ROCM_PATH", str(args.rocm))
    env.setdefault("GGML_HRX_ROCM_PATH", str(args.rocm))
    return env


def prepare_kernel_source(args: argparse.Namespace, output_dir: Path) -> Path:
    original = args.source_root / "kernels" / "mul_mat_q4_k_f32.loom"
    if args.source_target_policy == "original":
        args.source_rewrite = None
        return original
    text = original.read_text(encoding="utf-8")
    if args.source_target_policy == "neutral":
        rewritten = text.replace("kernel.def target(@hrx2_oracle_gfx1100_wave64) export(", "kernel.def export(")
        rewrite = "removed kernel.def target(@hrx2_oracle_gfx1100_wave64)"
    elif args.source_target_policy == "rewrite":
        rewritten = text.replace("amdgpu.target<gfx1100>", f"amdgpu.target<{args.compile_target}>")
        rewrite = f"amdgpu.target<gfx1100> -> amdgpu.target<{args.compile_target}>"
    else:
        raise ValueError(f"unsupported source target policy {args.source_target_policy}")
    if rewritten == text:
        args.source_rewrite = None
        return original
    source_dir = output_dir / "source"
    ensure_dir(source_dir)
    path = source_dir / f"mul_mat_q4_k_f32.{args.source_target_policy}.{args.compile_target}.loom"
    path.write_text(rewritten, encoding="utf-8")
    args.source_rewrite = {
        "original_source_path": str(original),
        "rewritten_source_path": str(path),
        "rewrite": rewrite,
    }
    return path


def run_command(argv: list[str], env: dict[str, str], cwd: Path | None = None) -> dict[str, Any]:
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "elapsed_ms": (time.monotonic() - start) * 1000.0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "ok" if proc.returncode == 0 else "failed",
    }


def build_helper(args: argparse.Namespace, env: dict[str, str]) -> Path | None:
    if args.helper and args.helper.exists():
        return args.helper
    ensure_dir(args.sample_build_dir)
    configure = run_command(
        ["cmake", "-S", str(args.sample_dir), "-B", str(args.sample_build_dir), f"-DCMAKE_PREFIX_PATH={args.hrx_install}"],
        env,
    )
    if configure["returncode"] != 0:
        print(configure["stderr"], file=sys.stderr)
        return None
    build = run_command(["cmake", "--build", str(args.sample_build_dir), "--target", "query_launch_info"], env)
    if build["returncode"] != 0:
        print(build["stderr"], file=sys.stderr)
        return None
    helper = args.sample_build_dir / "query_launch_info"
    return helper if helper.exists() else None


def config_args(config: dict[str, str]) -> list[str]:
    return [f"--config={key}={config[key]}" for key in sorted(config)]


def write_npy_header(f: BinaryIO, descr: str, shape: tuple[int, ...]) -> None:
    shape_text = "(" + ", ".join(str(x) for x in shape) + ("," if len(shape) == 1 else "") + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    header_bytes = header.encode("latin1")
    pad = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * pad + b"\n"
    f.write(b"\x93NUMPY\x01\x00")
    f.write(struct.pack("<H", len(header_bytes)))
    f.write(header_bytes)


def write_q4k_pattern_npy(path: Path, k: int, rows: int) -> None:
    ensure_dir(path.parent)
    blocks = rows * (k // QK_K)
    with path.open("wb") as f:
        write_npy_header(f, "|i1", (blocks * Q4_K_BLOCK_BYTES,))
        for block_index in range(blocks):
            block = bytearray(Q4_K_BLOCK_BYTES)
            block[0:2] = b"\x00\x38"  # f16 0.5
            block[2:4] = b"\x00\x30"  # f16 0.125
            for i in range(12):
                block[4 + i] = (17 + 13 * i + 7 * block_index) & 0x3F
            state = (block_index * 1103515245 + 12345) & 0xFFFFFFFF
            for i in range(128):
                state = (1664525 * state + 1013904223) & 0xFFFFFFFF
                lo = (state >> 16) & 0x0F
                hi = (state >> 24) & 0x0F
                block[16 + i] = lo | (hi << 4)
            f.write(block)


def write_f32_pattern_npy(path: Path, element_count: int, phase: int) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        write_npy_header(f, "<f4", (element_count,))
        chunk = bytearray()
        for i in range(element_count):
            value = (((i * 17 + phase * 29) % 257) - 128) / 64.0
            chunk += struct.pack("<f", value)
            if len(chunk) >= 1 << 20:
                f.write(chunk)
                chunk.clear()
        if chunk:
            f.write(chunk)


def write_zero_i8_npy(path: Path, element_count: int) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        write_npy_header(f, "|i1", (element_count,))
        f.write(b"\x00" * element_count)


def write_zero_f32_npy(path: Path, element_count: int) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        write_npy_header(f, "<f4", (element_count,))
        zero_chunk = b"\x00\x00\x00\x00" * 262144
        remaining = element_count
        while remaining:
            take = min(remaining, 262144)
            f.write(zero_chunk[: take * 4])
            remaining -= take


def write_workbench(candidate: Candidate, kernel_source: Path, path: Path, fixture: str) -> str:
    src0_elems = q4_bytes(candidate.k, candidate.rows)
    src1_elems = candidate.k * candidate.cols
    dst_elems = candidate.rows * candidate.cols
    root = candidate.route["root_symbol"]
    case_name = f"@case_{candidate.candidate_id}"
    bench_name = f"@bench_{candidate.candidate_id}"
    fixtures = path.parent / "fixtures"
    if fixture == "zero-smoke":
        write_zero_i8_npy(fixtures / "src0.npy", src0_elems)
        write_zero_f32_npy(fixtures / "src1.npy", src1_elems)
        write_f32_pattern_npy(fixtures / "dst_init.npy", dst_elems, phase=2)
        write_zero_f32_npy(fixtures / "expected.npy", dst_elems)
        expectation = f"""
  %expected = check.file.read.npy path("fixtures/expected.npy") : tensor<{dst_elems}xf32>
  check.expect.close actual(%dst) expected(%expected) atol(0.0) rtol(0.0) : tensor<{dst_elems}xf32>"""
    elif fixture == "pattern":
        write_q4k_pattern_npy(fixtures / "src0.npy", candidate.k, candidate.rows)
        write_f32_pattern_npy(fixtures / "src1.npy", src1_elems, phase=1)
        write_f32_pattern_npy(fixtures / "dst_init.npy", dst_elems, phase=2)
        expectation = ""
    else:
        raise ValueError(f"unsupported fixture {fixture}")

    with kernel_source.open("r", encoding="utf-8") as f:
        kernel_text = f.read()
    suffix = f"""

check.case public {case_name} {{
  %src0 = check.file.read.npy path("fixtures/src0.npy") : tensor<{src0_elems}xi8>
  %src1 = check.file.read.npy path("fixtures/src1.npy") : tensor<{src1_elems}xf32>
  %dst = check.file.read.npy path("fixtures/dst_init.npy") : tensor<{dst_elems}xf32>
  func.call {root}(%src0, %src1, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>){expectation}
  check.return
}}

check.benchmark<{case_name}> {bench_name}
"""
    path.write_text(kernel_text + suffix, encoding="utf-8")
    return bench_name


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc)}


def dig(value: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def compile_report_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "emission_code_byte_count": dig(report, "emission", "code_byte_count"),
        "allocation_spill_count": dig(report, "allocation", "spill_count"),
        "memory_local_bytes": dig(report, "memory", "local_bytes"),
        "static_instruction_mix": report.get("static_instruction_mix"),
        "entries_row_count": len(dig(report, "entries", "rows") or []),
    }


def tail_text(text: str, limit: int = 6000) -> str | None:
    return text[-limit:] if text else None


def compile_candidate(
    args: argparse.Namespace,
    env: dict[str, str],
    candidate: Candidate,
    run_dir: Path,
    helper: Path | None,
    sanitizer: str = "none",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidate_dir = run_dir / "candidates" / candidate.candidate_id / sanitizer
    ensure_dir(candidate_dir)
    report_path = candidate_dir / "compile_report.json"
    manifest_path = candidate_dir / "artifact_manifest.json"
    artifact_path = candidate_dir / "artifact.bin"
    target_artifact_path = candidate_dir / "target.hsaco"
    linked_path = candidate_dir / "linked.loom"

    link_result = run_command(
        [
            str(args.loom_link),
            str(args.effective_kernel_source),
            "--mode=link",
            "--to=text",
            "--strip-check",
            "--require-resolved-config",
            f"--root={candidate.route['root_symbol']}",
            f"--output={linked_path}",
            *config_args(candidate.config),
        ],
        env,
    )

    launch_from_helper = None
    helper_result = None
    if link_result["returncode"] == 0 and helper:
        helper_result = run_command([str(helper), str(linked_path), candidate.route["root_symbol"]], env)
        if helper_result["returncode"] == 0:
            try:
                functions = json.loads(helper_result["stdout"]).get("functions", [])
                if functions:
                    info = functions[0]
                    launch_from_helper = {
                        "workgroup_count": info.get("static_dispatch_workgroup_count"),
                        "workgroup_size": info.get("static_workgroup_size"),
                        "rows_per_workgroup": candidate.route.get("dispatch", {}).get("rows_per_workgroup"),
                        "cols_per_workgroup": candidate.route.get("dispatch", {}).get("cols_per_workgroup"),
                        "metadata_source": "loomc_module",
                        "has_static_dispatch_workgroup_count": info.get("has_static_dispatch_workgroup_count", False),
                        "has_static_workgroup_size": info.get("has_static_workgroup_size", False),
                    }
            except json.JSONDecodeError:
                pass

    compile_cmd = [
        str(args.loom_compile),
        str(args.effective_kernel_source),
        "--backend=amdgpu-hal",
        f"--target={args.compile_target}",
        f"--root={candidate.route['root_symbol']}",
        f"--output={artifact_path}",
        f"--emit-target-artifact={target_artifact_path}",
        "--compile-report=details",
        f"--compile-report-output={report_path}",
        "--artifact-manifest=analysis",
        f"--emit-artifact-manifest={manifest_path}",
        *config_args(candidate.config),
    ]
    if sanitizer != "none":
        compile_cmd.append(f"--sanitizer={sanitizer}")
    compile_result = run_command(compile_cmd, env)
    report = load_json(report_path)
    return (
        {
            "status": compile_result["status"],
            "returncode": compile_result["returncode"],
            "elapsed_ms": compile_result["elapsed_ms"],
            "sanitizer": sanitizer,
            "report_path": str(report_path) if report_path.exists() else None,
            "manifest_path": str(manifest_path) if manifest_path.exists() else None,
            "artifact_path": str(artifact_path) if artifact_path.exists() else None,
            "target_artifact_path": str(target_artifact_path) if target_artifact_path.exists() else None,
            "target_artifact_bytes": target_artifact_path.stat().st_size if target_artifact_path.exists() else None,
            "link_status": link_result["status"],
            "link_returncode": link_result["returncode"],
            "linked_module_path": str(linked_path) if linked_path.exists() else None,
            "helper_status": helper_result["status"] if helper_result else ("skipped" if not helper else "not_run"),
            "summary": compile_report_summary(report),
            "stderr_tail": tail_text(compile_result["stderr"]),
            "stdout_tail": tail_text(compile_result["stdout"]),
        },
        launch_from_helper,
    )


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def benchmark_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for event in events:
        if event.get("row") == "benchmark":
            result = event.get("benchmark_result", {})
            measurement = result.get("measurement", {})
            summary["state"] = result.get("state")
            summary["correctness"] = result.get("correctness")
            summary["operation_timing_ns"] = measurement.get("operation_timing_ns")
            summary["physical_dispatches_per_logical_operation"] = measurement.get("physical_dispatches_per_logical_operation")
            summary["mean_physical_dispatch_duration_ns"] = measurement.get("mean_physical_dispatch_duration_ns")
            summary["timing_warnings"] = dig(measurement, "timing_interpretation", "warnings")
            if result.get("compile_report"):
                summary["compile_report"] = compile_report_summary(result.get("compile_report"))
            if result.get("failure"):
                summary["failure"] = result.get("failure")
        elif event.get("row") == "summary":
            summary["tool_summary"] = event.get("summary")
        elif event.get("row") == "failure":
            summary.setdefault("failures", []).append(event)
    return summary


def run_benchmark(args: argparse.Namespace, env: dict[str, str], candidate: Candidate, run_dir: Path, sanitizer: str = "none") -> dict[str, Any]:
    candidate_dir = run_dir / "candidates" / candidate.candidate_id / f"run_{sanitizer}_{args.fixture}"
    ensure_dir(candidate_dir)
    linked_kernel_source = candidate_dir / "kernel.linked.loom"
    link_result = run_command(
        [
            str(args.loom_link),
            str(args.effective_kernel_source),
            "--mode=link",
            "--to=text",
            "--strip-check",
            "--require-resolved-config",
            f"--root={candidate.route['root_symbol']}",
            f"--output={linked_kernel_source}",
            *config_args(candidate.config),
        ],
        env,
    )
    if link_result["returncode"] != 0:
        return {
            "status": "failed",
            "returncode": link_result["returncode"],
            "elapsed_ms": link_result["elapsed_ms"],
            "sanitizer": sanitizer,
            "fixture": args.fixture,
            "correctness_strength": "zero_reference" if args.fixture == "zero-smoke" else "pattern_no_reference",
            "linked_kernel_source_path": str(linked_kernel_source) if linked_kernel_source.exists() else None,
            "link_status": link_result["status"],
            "stderr_tail": tail_text(link_result["stderr"]),
            "stdout_tail": tail_text(link_result["stdout"]),
            "summary": {"state": "link_failed"},
        }

    workbench_path = candidate_dir / "workbench.loom"
    bench_name = write_workbench(candidate, linked_kernel_source, workbench_path, args.fixture)
    results_path = candidate_dir / "results.jsonl"
    bundle_dir = candidate_dir / "bundle"
    cmd = [
        str(args.iree_benchmark_loom),
        str(workbench_path),
        "--device=amdgpu",
        f"--benchmark={bench_name}",
        "--measure=dispatch_complete",
        "--batch-size=1",
        "--input-ring-count=1",
        f"--iterations={args.iterations}",
        f"--warmup-iterations={args.warmup_iterations}",
        f"--max-batches={args.max_batches}",
        "--profile-final-batch=true",
        "--sample-compilation=per_sample",
        "--compile-report=details",
        "--artifact-manifest=analysis",
        f"--artifact-bundle-dir={bundle_dir}",
        "--artifact-bundle-policy=debug",
        f"--output={results_path}",
        "--output-format=jsonl",
    ]
    if sanitizer != "none":
        cmd.append(f"--sanitizer={sanitizer}")
    result = run_command(cmd, env)
    return {
        "status": result["status"],
        "returncode": result["returncode"],
        "elapsed_ms": result["elapsed_ms"],
        "sanitizer": sanitizer,
        "fixture": args.fixture,
        "correctness_strength": "zero_reference" if args.fixture == "zero-smoke" else "pattern_no_reference",
        "link_status": link_result["status"],
        "linked_kernel_source_path": str(linked_kernel_source) if linked_kernel_source.exists() else None,
        "workbench_path": str(workbench_path),
        "results_path": str(results_path) if results_path.exists() else None,
        "artifact_bundle_dir": str(bundle_dir) if bundle_dir.exists() else None,
        "summary": benchmark_summary(parse_jsonl(results_path)),
        "stderr_tail": tail_text(result["stderr"]),
        "stdout_tail": tail_text(result["stdout"]),
    }


def acceptance(row: dict[str, Any], mode: str, sanitizer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    tier = "planned"
    if mode in {"compile", "run"}:
        tier = "compile_only"
        if dig(row, "compile", "status") != "ok":
            reasons.append("compile_failed")
        launch = row.get("launch", {})
        if not launch.get("workgroup_count") or not launch.get("workgroup_size"):
            reasons.append("missing_launch_metadata")
    if mode == "run":
        tier = "timed"
        if dig(row, "benchmark", "status") != "ok":
            reasons.append("benchmark_failed")
        if not dig(row, "benchmark", "summary", "operation_timing_ns"):
            reasons.append("missing_timing")
    for sanitizer_row in sanitizer_rows:
        if dig(sanitizer_row, "compile", "status") not in (None, "ok"):
            reasons.append(f"sanitizer_{sanitizer_row.get('sanitizer')}_compile_failed")
        if dig(sanitizer_row, "benchmark", "status") not in (None, "ok"):
            reasons.append(f"sanitizer_{sanitizer_row.get('sanitizer')}_run_failed")
    return {"tier": tier, "accepted": not reasons, "reasons": reasons}


def base_row(args: argparse.Namespace, candidate: Candidate) -> dict[str, Any]:
    return {
        "schema": "loom_kernel_ledger.q4k.v1",
        "run_id": args.run_id,
        "machine": {
            "target_key": args.target,
            "compile_target_key": args.compile_target,
            "source_target_policy": args.source_target_policy,
            "rocm_path": str(args.rocm),
            "hrx_install": str(args.hrx_install),
        },
        "candidate_id": candidate.candidate_id,
        "route_id": candidate.route.get("id"),
        "route_short_name": route_short_name(candidate.route),
        "root_symbol": candidate.route.get("root_symbol"),
        "export_name": candidate.route.get("export_name"),
        "algorithm": dig(candidate.route, "evidence_summary", "algorithm"),
        "source_path": str(args.effective_kernel_source),
        "source_rewrite": getattr(args, "source_rewrite", None),
        "shape": shape_stats(candidate),
        "config_bindings": candidate.config,
        "launch": route_launch(candidate),
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    with (output_dir / "ledger.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    fields = ["candidate_id", "route_short_name", "k", "rows", "cols", "accepted", "acceptance_reasons", "workgroup_count", "workgroup_size", "compile_status", "timing_p50_ns"]
    with (output_dir / "ledger_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            timing = dig(row, "benchmark", "summary", "operation_timing_ns") or {}
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id"),
                    "route_short_name": row.get("route_short_name"),
                    "k": dig(row, "shape", "k"),
                    "rows": dig(row, "shape", "rows"),
                    "cols": dig(row, "shape", "cols"),
                    "accepted": dig(row, "acceptance", "accepted"),
                    "acceptance_reasons": ";".join(dig(row, "acceptance", "reasons") or []),
                    "workgroup_count": json.dumps(dig(row, "launch", "workgroup_count")),
                    "workgroup_size": json.dumps(dig(row, "launch", "workgroup_size")),
                    "compile_status": dig(row, "compile", "status"),
                    "timing_p50_ns": timing.get("p50"),
                }
            )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root_from_script()
    parser.add_argument("--mode", choices=["plan", "compile", "run"], default="plan")
    parser.add_argument("--target", default="gfx1151")
    parser.add_argument("--compile-target", default="gfx1151", help="Target key passed to loom-compile.")
    parser.add_argument("--source-target-policy", choices=["original", "neutral", "rewrite"], default="rewrite", help="How to adapt authored target records before compiling.")
    parser.add_argument("--routes", help="Comma-separated route short names or route IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k-values")
    parser.add_argument("--row-values")
    parser.add_argument("--col-values")
    parser.add_argument("--sanitizers", default="", help="Comma-separated sanitizer passes, e.g. asan,tsan")
    parser.add_argument("--fixture", choices=["pattern", "zero-smoke"], default="pattern")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument("--run-id", default=now_run_id())
    parser.add_argument("--workspace", type=Path, default=root)
    parser.add_argument("--hrx-install", type=Path, default=root / "build" / "hrx-install")
    parser.add_argument("--rocm", type=Path, default=root / "rocm")
    parser.add_argument("--source-root", type=Path, default=root / "sources" / "llama.cpp-ref" / "ggml" / "src" / "ggml-hrx2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-build-dir", type=Path, default=root / "build" / "loom-kernel-ledger")
    parser.add_argument("--helper", type=Path)
    args = parser.parse_args(argv)

    args.sample_dir = Path(__file__).resolve().parent
    args.kernel_source = args.source_root / "kernels" / "mul_mat_q4_k_f32.loom"
    args.route_json = args.source_root / "catalog" / "routes" / "mul_mat_q4_k_f32.json"
    args.loom_link = args.hrx_install / "bin" / "loom-link"
    args.loom_compile = args.hrx_install / "bin" / "loom-compile"
    args.iree_benchmark_loom = args.hrx_install / "bin" / "iree-benchmark-loom"
    output_dir = args.output_dir or (args.workspace / "cache" / "loom-kernel-ledger" / args.run_id)
    ensure_dir(output_dir)

    for required in [args.kernel_source, args.route_json]:
        if not required.exists():
            raise FileNotFoundError(required)
    args.effective_kernel_source = prepare_kernel_source(args, output_dir)
    if args.mode in {"compile", "run"}:
        for required in [args.loom_link, args.loom_compile]:
            if not required.exists():
                raise FileNotFoundError(required)
    if args.mode == "run" and not args.iree_benchmark_loom.exists():
        raise FileNotFoundError(args.iree_benchmark_loom)

    route_filter = set(part.strip() for part in args.routes.split(",") if part.strip()) if args.routes else None
    routes = parse_routes(args.route_json, route_filter)
    candidates = generate_candidates(
        routes,
        parse_csv_ints(args.k_values),
        parse_csv_ints(args.row_values),
        parse_csv_ints(args.col_values),
        args.limit,
    )
    env = command_env(args)
    helper = build_helper(args, env) if args.mode in {"compile", "run"} else None
    sanitizers = [part.strip() for part in args.sanitizers.split(",") if part.strip()]

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.candidate_id}", file=sys.stderr)
        row = base_row(args, candidate)
        sanitizer_rows = []
        if args.mode in {"compile", "run"}:
            compile_payload, launch_info = compile_candidate(args, env, candidate, output_dir, helper)
            row["compile"] = compile_payload
            if launch_info and launch_info.get("has_static_dispatch_workgroup_count") and launch_info.get("has_static_workgroup_size"):
                row["launch"] = launch_info
            if args.mode == "run":
                row["benchmark"] = run_benchmark(args, env, candidate, output_dir)
            for sanitizer in sanitizers:
                srow = {"sanitizer": sanitizer}
                scompile, _ = compile_candidate(args, env, candidate, output_dir, helper, sanitizer=sanitizer)
                srow["compile"] = scompile
                if args.mode == "run":
                    srow["benchmark"] = run_benchmark(args, env, candidate, output_dir, sanitizer=sanitizer)
                sanitizer_rows.append(srow)
            if sanitizer_rows:
                row["sanitizers"] = sanitizer_rows
        row["acceptance"] = acceptance(row, args.mode, sanitizer_rows)
        rows.append(row)
        write_outputs(rows, output_dir)

    write_outputs(rows, output_dir)
    print(f"wrote {len(rows)} rows to {output_dir / 'ledger.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
