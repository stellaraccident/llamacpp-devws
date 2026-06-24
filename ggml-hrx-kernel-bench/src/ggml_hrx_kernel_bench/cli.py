from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import BenchConfig, ToolPaths
from .hrx2 import (
    DEFAULT_HRX2_CATALOG_DIR,
    DEFAULT_HRX2_KERNEL_DIR,
    Candidate,
    all_candidates,
    build_manifest,
)
from .ledger import LedgerWriter, utc_run_id
from .observed_shapes import load_observed_shapes, read_observations_jsonl, save_observed_shapes
from .oracles import generate_oracle, write_workbench
from .specs import KernelSpec, config_args, file_sha256, spec_sha256
from .tools import CommandResult, run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ggml-hrx-kernel-bench")
    parser.add_argument("--spec", type=Path, help="single legacy kernel spec")
    parser.add_argument("--kernel-source", type=Path, help="override source for --spec mode")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="gfx1100")
    parser.add_argument("--rocm-path", type=Path)
    parser.add_argument("--loom-link", type=Path)
    parser.add_argument("--loom-compile", type=Path)
    parser.add_argument("--iree-benchmark-loom", type=Path)
    parser.add_argument("--values", type=Path, help="JSON object containing concrete parameter values for --spec mode")
    parser.add_argument("--hrx2-kernel-dir", type=Path, default=DEFAULT_HRX2_KERNEL_DIR)
    parser.add_argument("--hrx2-catalog-dir", type=Path, default=DEFAULT_HRX2_CATALOG_DIR)
    parser.add_argument("--observed-shapes", type=Path, help="observed shape metadata JSON; defaults to <catalog>/observed_shapes.json")
    parser.add_argument("--shape-trace", type=Path, action="append", default=[], help="JSONL observed-shape trace to merge with accumulate-shapes")
    parser.add_argument("--original-hrx2-root", type=Path, help="optional original HRX2 root for import hash comparison")
    parser.add_argument("--family", action="append", default=[], help="family/source/route filter; may be repeated or comma separated")
    parser.add_argument("--limit", type=int, help="limit corpus candidates")
    parser.add_argument("--sweep", choices=["minimal", "edge", "observed"], default="minimal")
    parser.add_argument("--include-source-only", action="store_true", help="include source-only/probe kernel rows in addition to route-backed catalog rows")
    parser.add_argument("--sanitizers", default="none", help="comma list, for example none,asan,tsan")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("import-hrx2")
    subparsers.add_parser("accumulate-shapes")
    subparsers.add_parser("plan")
    subparsers.add_parser("fixtures")
    subparsers.add_parser("link")
    subparsers.add_parser("compile")
    subparsers.add_parser("run")
    subparsers.add_parser("verify")
    subparsers.add_parser("tune")
    subparsers.add_parser("catalog")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = LedgerWriter(args.output_dir / "ledger.jsonl")
    config = BenchConfig(
        output_dir=args.output_dir,
        target=args.target,
        rocm_path=args.rocm_path,
        tools=ToolPaths(
            loom_link=args.loom_link,
            loom_compile=args.loom_compile,
            iree_benchmark_loom=args.iree_benchmark_loom,
        ),
    )

    try:
        if args.command == "import-hrx2":
            return command_import(args, ledger)
        if args.command == "accumulate-shapes":
            return command_accumulate_shapes(args, ledger)
        if args.spec:
            return command_legacy_spec(args, config, ledger)
        return command_corpus(args, config, ledger)
    except Exception as exc:
        ledger.append(
            {
                "schema": "ggml_hrx_kernel_bench.ledger.v1",
                "run_id": utc_run_id(),
                "action": args.command,
                "status": "tool_error",
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )
        raise


def command_import(args: argparse.Namespace, ledger: LedgerWriter) -> int:
    manifest = build_manifest(args.hrx2_kernel_dir, args.hrx2_catalog_dir, original_root=args.original_hrx2_root)
    manifest_path = args.output_dir / "hrx2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row = {
        "schema": "ggml_hrx_kernel_bench.ledger.v1",
        "run_id": utc_run_id(),
        "action": "import-hrx2",
        "status": "ok",
        "manifest_path": str(manifest_path),
        "kernel_dir": str(args.hrx2_kernel_dir),
        "catalog_dir": str(args.hrx2_catalog_dir),
        "summary": {
            "kernel_count": manifest["kernel_count"],
            "catalog_source_count": manifest["catalog_source_count"],
            "route_count": manifest["route_count"],
            "source_ids_without_routes": manifest["source_ids_without_routes"],
            "route_source_ids_without_source_entry": manifest["route_source_ids_without_source_entry"],
            "kernel_files_without_source_entry": manifest["kernel_files_without_source_entry"],
            "source_entries_without_kernel_file": manifest["source_entries_without_kernel_file"],
        },
    }
    ledger.append(row)
    return 0


def command_accumulate_shapes(args: argparse.Namespace, ledger: LedgerWriter) -> int:
    if not args.shape_trace:
        raise ValueError("accumulate-shapes requires at least one --shape-trace JSONL file")
    path = observed_shapes_path(args)
    catalog = load_observed_shapes(path)
    observations = read_observations_jsonl(args.shape_trace)
    before_count = len(catalog.rows)
    catalog.merge(observations)
    save_observed_shapes(path, catalog)
    ledger.append(
        {
            "schema": "ggml_hrx_kernel_bench.ledger.v1",
            "run_id": utc_run_id(),
            "action": "accumulate-shapes",
            "status": "ok",
            "observed_shapes_path": str(path),
            "trace_paths": [str(path) for path in args.shape_trace],
            "input_observation_count": len(observations),
            "row_count_before": before_count,
            "row_count_after": len(catalog.rows),
        }
    )
    return 0


def command_corpus(args: argparse.Namespace, config: BenchConfig, ledger: LedgerWriter) -> int:
    candidates = selected_candidates(args)
    if args.command == "plan":
        ledger.write_all(corpus_row(args, candidate, action="plan") for candidate in candidates)
        write_summary(args.output_dir, candidates)
        return 0
    if args.command == "fixtures":
        rows = [fixtures_row(args, candidate) for candidate in candidates]
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "link":
        rows = [link_candidate_row(args, config, candidate) for candidate in candidates]
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "compile":
        rows: list[dict[str, Any]] = []
        for sanitizer in sanitizer_list(args):
            for candidate in candidates:
                rows.append(compile_candidate_row(args, config, candidate, sanitizer=sanitizer))
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "run":
        rows: list[dict[str, Any]] = []
        for sanitizer in sanitizer_list(args):
            for candidate in candidates:
                rows.append(run_candidate_row(args, config, candidate, sanitizer=sanitizer))
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "verify":
        rows = [fixtures_row(args, candidate, action="verify") for candidate in candidates]
        ledger.write_all(rows)
        return status_code_from_rows(rows)
    if args.command == "tune":
        rows = tune_rows(args, config, candidates)
        ledger.write_all(rows)
        write_tune_summary(args.output_dir, rows)
        return status_code_from_rows(rows)
    if args.command == "catalog":
        return command_catalog(args, candidates, ledger)
    raise ValueError(f"unsupported command {args.command}")


def selected_candidates(args: argparse.Namespace) -> list[Candidate]:
    families = filter_set(args.family)
    observed = load_observed_shapes(observed_shapes_path(args)) if args.sweep == "observed" else None
    return all_candidates(
        args.hrx2_kernel_dir,
        args.hrx2_catalog_dir,
        families=families,
        limit=args.limit,
        sweep=args.sweep,
        observed_shapes=observed,
        include_source_only=args.include_source_only,
    )


def observed_shapes_path(args: argparse.Namespace) -> Path:
    return args.observed_shapes or (args.hrx2_catalog_dir / "observed_shapes.json")


def filter_set(values: list[str]) -> set[str] | None:
    out: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out or None


def sanitizer_list(args: argparse.Namespace) -> list[str]:
    values = [part.strip() for part in args.sanitizers.split(",") if part.strip()]
    return values or ["none"]


def corpus_row(args: argparse.Namespace, candidate: Candidate, *, action: str) -> dict[str, Any]:
    row = {
        "schema": "ggml_hrx_kernel_bench.ledger.v1",
        "run_id": utc_run_id(),
        "action": action,
        "machine": {"target": args.target, "rocm_path": str(args.rocm_path) if args.rocm_path else None},
        "candidate": candidate.to_ledger(),
        "status": candidate.status,
    }
    if candidate.message:
        row["message"] = candidate.message
    return row


def candidate_dir(args: argparse.Namespace, candidate: Candidate, *parts: str) -> Path:
    return args.output_dir / "candidates" / candidate.id / Path(*parts)


def fixtures_row(args: argparse.Namespace, candidate: Candidate, *, action: str = "fixtures") -> dict[str, Any]:
    row = corpus_row(args, candidate, action=action)
    if candidate.status != "planned":
        row["status"] = candidate.status
        row["message"] = candidate.message
        return row
    result = generate_oracle(candidate, candidate_dir(args, candidate, "fixtures"))
    row["oracle"] = result.to_ledger()
    row["status"] = result.status
    return row


def link_candidate_row(args: argparse.Namespace, config: BenchConfig, candidate: Candidate) -> dict[str, Any]:
    row = corpus_row(args, candidate, action="link")
    if candidate.status != "planned":
        return row
    out_dir = candidate_dir(args, candidate, "link")
    out_dir.mkdir(parents=True, exist_ok=True)
    linked = out_dir / "linked.loom"
    result = run_command(
        [
            config.tools.require_loom_link(),
            candidate.source_path,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={candidate.root_symbol}",
            f"--output={linked}",
            *config_args(candidate.config),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(result, out_dir)
    row["link"]["output"] = str(linked) if linked.exists() else None
    row["status"] = "linked" if result.returncode == 0 else "link_failed"
    return row


def compile_candidate_row(args: argparse.Namespace, config: BenchConfig, candidate: Candidate, *, sanitizer: str) -> dict[str, Any]:
    row = corpus_row(args, candidate, action="compile")
    row["sanitizer"] = sanitizer
    if candidate.status != "planned":
        return row
    out_dir = candidate_dir(args, candidate, "compile", sanitizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    linked = out_dir / "linked.loom"
    link_result = run_command(
        [
            config.tools.require_loom_link(),
            candidate.source_path,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={candidate.root_symbol}",
            f"--output={linked}",
            *config_args(candidate.config),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(link_result, out_dir, prefix="link")
    row["link"]["output"] = str(linked) if linked.exists() else None
    if link_result.returncode != 0:
        row["status"] = "link_failed"
        return row

    report = out_dir / "compile_report.json"
    manifest = out_dir / "artifact_manifest.json"
    artifact = out_dir / "artifact.bin"
    target_artifact = out_dir / "target.hsaco"
    compile_cmd: list[str | Path] = [
        config.tools.require_loom_compile(),
        linked,
        "--backend=amdgpu-hal",
        f"--target={config.target}",
        f"--root={candidate.root_symbol}",
        f"--output={artifact}",
        f"--emit-target-artifact={target_artifact}",
        "--compile-report=details",
        f"--compile-report-output={report}",
        "--artifact-manifest=analysis",
        f"--emit-artifact-manifest={manifest}",
    ]
    if sanitizer != "none":
        compile_cmd.append(f"--sanitizer={sanitizer}")
    compile_result = run_command(compile_cmd, env=config.command_env())
    row["compile"] = command_evidence(compile_result, out_dir, prefix="compile")
    row["compile"].update(
        {
            "report": str(report) if report.exists() else None,
            "manifest": str(manifest) if manifest.exists() else None,
            "artifact": str(artifact) if artifact.exists() else None,
            "target_artifact": str(target_artifact) if target_artifact.exists() else None,
            "target_artifact_bytes": target_artifact.stat().st_size if target_artifact.exists() else None,
            "report_summary": compile_report_summary(report),
        }
    )
    row["status"] = "compiled" if compile_result.returncode == 0 else "compile_failed"
    return row


def run_candidate_row(args: argparse.Namespace, config: BenchConfig, candidate: Candidate, *, sanitizer: str) -> dict[str, Any]:
    row = corpus_row(args, candidate, action="run")
    row["sanitizer"] = sanitizer
    if candidate.status != "planned":
        return row
    out_dir = candidate_dir(args, candidate, "run", sanitizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = generate_oracle(candidate, out_dir / "fixtures")
    row["oracle"] = fixture.to_ledger()
    if fixture.status != "fixtures_ready" or fixture.fixture_dir is None:
        row["status"] = fixture.status
        return row

    linked = out_dir / "linked.loom"
    link_result = run_command(
        [
            config.tools.require_loom_link(),
            candidate.source_path,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={candidate.root_symbol}",
            f"--output={linked}",
            *config_args(candidate.config),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(link_result, out_dir, prefix="link")
    if link_result.returncode != 0:
        row["status"] = "link_failed"
        return row

    workbench = out_dir / "workbench.loom"
    bench_name, workbench_meta = write_workbench(candidate, linked, workbench, fixture.fixture_dir)
    row["workbench"] = workbench_meta
    if bench_name is None:
        row["status"] = workbench_meta.get("status", "unsupported_golden")
        return row

    results = out_dir / "results.jsonl"
    bundle_dir = out_dir / "bundle"
    benchmark_tool = config.tools.require_iree_benchmark_loom().resolve()
    max_batches = max(args.max_batches, args.iterations)
    cmd: list[str | Path] = [
        benchmark_tool,
        workbench.resolve(),
        "--device=amdgpu",
        f"--benchmark={bench_name}",
        "--measure=dispatch_complete",
        "--batch-size=1",
        "--input-ring-count=1",
        f"--iterations={args.iterations}",
        f"--warmup-iterations={args.warmup_iterations}",
        f"--max-batches={max_batches}",
        "--profile-final-batch=true",
        "--sample-compilation=per_sample",
        "--compile-report=details",
        "--artifact-manifest=analysis",
        f"--artifact-bundle-dir={bundle_dir.resolve()}",
        "--artifact-bundle-policy=debug",
        f"--output={results.resolve()}",
        "--output-format=jsonl",
    ]
    if sanitizer != "none":
        cmd.append(f"--sanitizer={sanitizer}")
    result = run_command(cmd, env=config.command_env(), cwd=out_dir)
    row["benchmark"] = command_evidence(result, out_dir, prefix="benchmark")
    row["benchmark"].update(
        {
            "results_path": str(results) if results.exists() else None,
            "artifact_bundle_dir": str(bundle_dir) if bundle_dir.exists() else None,
            "summary": benchmark_summary(results),
        }
    )
    row["status"] = "ran" if result.returncode == 0 else "run_failed"
    return row


def tune_rows(args: argparse.Namespace, config: BenchConfig, candidates: list[Candidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sanitizer in sanitizer_list(args):
        for candidate in candidates:
            if config.tools.iree_benchmark_loom is not None:
                row = run_candidate_row(args, config, candidate, sanitizer=sanitizer)
                row["action"] = "tune"
                row["tune_stage"] = "run"
            elif config.tools.loom_compile is not None:
                row = compile_candidate_row(args, config, candidate, sanitizer=sanitizer)
                row["action"] = "tune"
                row["tune_stage"] = "compile"
            elif config.tools.loom_link is not None:
                row = link_candidate_row(args, config, candidate)
                row["action"] = "tune"
                row["tune_stage"] = "link"
            else:
                row = corpus_row(args, candidate, action="tune")
                row["status"] = "planned"
                row["tune_stage"] = "plan"
                row["message"] = "provide --iree-benchmark-loom for timing, --loom-compile for compile sweep, or --loom-link for link sweep"
            rows.append(row)
    return rows


def command_catalog(args: argparse.Namespace, candidates: list[Candidate], ledger: LedgerWriter) -> int:
    ledger_rows = read_jsonl(args.output_dir / "ledger.jsonl")
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        candidate_id = (((row.get("candidate") or {}).get("candidate_id")) or "")
        if not candidate_id:
            continue
        by_candidate.setdefault(candidate_id, {})[row.get("action", "unknown")] = row
    catalog_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence = by_candidate.get(candidate.id, {})
        compile_row = evidence.get("compile")
        run_row = evidence.get("run")
        catalog_ready = bool(compile_row and compile_row.get("status") == "compiled")
        if run_row and run_row.get("status") != "ran":
            catalog_ready = False
        catalog_rows.append(
            {
                "candidate_id": candidate.id,
                "catalog_ready": catalog_ready,
                "candidate": candidate.to_ledger(),
                "compile": (compile_row or {}).get("compile"),
                "oracle": (run_row or {}).get("oracle"),
                "benchmark": (run_row or {}).get("benchmark"),
                "rejection_reasons": rejection_reasons(candidate, compile_row, run_row),
            }
        )
    catalog_dir = args.output_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "candidates.json"
    catalog_path.write_text(json.dumps(catalog_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger.append(
        {
            "schema": "ggml_hrx_kernel_bench.ledger.v1",
            "run_id": utc_run_id(),
            "action": "catalog",
            "status": "ok",
            "catalog_path": str(catalog_path),
            "candidate_count": len(catalog_rows),
            "catalog_ready_count": sum(1 for row in catalog_rows if row["catalog_ready"]),
        }
    )
    return 0


def rejection_reasons(candidate: Candidate, compile_row: dict[str, Any] | None, run_row: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if candidate.status != "planned":
        reasons.append(candidate.status)
    if not compile_row:
        reasons.append("not_compiled")
    elif compile_row.get("status") != "compiled":
        reasons.append(str(compile_row.get("status")))
    if run_row and run_row.get("status") != "ran":
        reasons.append(str(run_row.get("status")))
    return reasons


def command_evidence(result: CommandResult, out_dir: Path, *, prefix: str = "command") -> dict[str, Any]:
    stdout_path = out_dir / f"{prefix}.stdout.txt"
    stderr_path = out_dir / f"{prefix}.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    row = result.to_ledger()
    row["stdout_path"] = str(stdout_path)
    row["stderr_path"] = str(stderr_path)
    return row


def compile_report_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc)}
    return {
        "emission_code_byte_count": dig(report, "emission", "code_byte_count"),
        "allocation_spill_count": dig(report, "allocation", "spill_count"),
        "memory_local_bytes": dig(report, "memory", "local_bytes"),
        "static_instruction_mix": report.get("static_instruction_mix"),
        "entries_row_count": len(dig(report, "entries", "rows") or []),
    }


def benchmark_summary(path: Path) -> dict[str, Any]:
    events = read_jsonl(path)
    summary: dict[str, Any] = {"event_count": len(events)}
    for event in events:
        if event.get("row") == "benchmark":
            result = event.get("benchmark_result", {})
            measurement = result.get("measurement", {})
            summary.update(
                {
                    "state": result.get("state"),
                    "correctness": result.get("correctness"),
                    "operation_timing_ns": measurement.get("operation_timing_ns"),
                    "mean_physical_dispatch_duration_ns": measurement.get("mean_physical_dispatch_duration_ns"),
                    "physical_dispatches_per_logical_operation": measurement.get("physical_dispatches_per_logical_operation"),
                    "failure": result.get("failure"),
                }
            )
        elif event.get("row") == "failure":
            summary.setdefault("failures", []).append(event)
    return summary


def dig(value: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary(output_dir: Path, candidates: list[Candidate]) -> None:
    summary = {
        "schema": "ggml_hrx_kernel_bench.plan_summary.v1",
        "candidate_count": len(candidates),
        "planned_count": sum(1 for candidate in candidates if candidate.status == "planned"),
        "by_status": {},
        "by_family": {},
    }
    for candidate in candidates:
        summary["by_status"][candidate.status] = summary["by_status"].get(candidate.status, 0) + 1
        summary["by_family"][candidate.family] = summary["by_family"].get(candidate.family, 0) + 1
    (output_dir / "plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tune_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ranked = []
    for row in rows:
        candidate = row.get("candidate") or {}
        benchmark = row.get("benchmark") or {}
        summary = benchmark.get("summary") or {}
        timing = summary.get("operation_timing_ns") or {}
        mean = timing.get("mean")
        ranked.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "family": candidate.get("family"),
                "status": row.get("status"),
                "tune_stage": row.get("tune_stage"),
                "shape": candidate.get("shape"),
                "config_bindings": candidate.get("config_bindings"),
                "mean_operation_timing_ns": mean,
                "correctness": summary.get("correctness"),
                "rejection": None if row.get("status") == "ran" else row.get("status"),
            }
        )
    ranked.sort(key=lambda item: (item["mean_operation_timing_ns"] is None, item["mean_operation_timing_ns"] or 0))
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema": "ggml_hrx_kernel_bench.tune_summary.v1",
        "candidate_count": len(rows),
        "status_counts": status_counts,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    (output_dir / "tune_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_code_from_rows(rows: list[dict[str, Any]]) -> int:
    hard_failures = {"tool_error"}
    if any(row.get("status") in hard_failures for row in rows):
        return 2
    return 0


def command_legacy_spec(args: argparse.Namespace, config: BenchConfig, ledger: LedgerWriter) -> int:
    spec = KernelSpec.from_json(args.spec, kernel_source_override=args.kernel_source)
    values = load_values(args.values)
    if args.command == "plan":
        ledger.append(base_spec_row(args, spec, values, config, action="plan"))
        return 0
    if args.command == "link":
        ledger.append(link_spec_row(args, spec, values, config))
        return 0
    if args.command == "compile":
        ledger.append(compile_spec_row(args, spec, values, config))
        return 0
    row = base_spec_row(args, spec, values, config, action=args.command)
    row["status"] = "unsupported_for_spec_mode"
    row["message"] = f"{args.command} is implemented for HRX2 corpus mode; omit --spec"
    ledger.append(row)
    return 2


def load_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def base_spec_row(
    args: argparse.Namespace,
    spec: KernelSpec,
    values: dict[str, Any],
    config: BenchConfig,
    *,
    action: str,
) -> dict[str, Any]:
    return {
        "schema": "ggml_hrx_kernel_bench.ledger.v1",
        "run_id": utc_run_id(),
        "action": action,
        "status": "planned",
        "spec": {
            "path": str(args.spec),
            "sha256": spec_sha256(args.spec),
            "id": spec.id,
            "op": spec.op,
            "root_symbol": spec.root_symbol,
            "export_name": spec.export_name,
            "types": spec.types,
        },
        "source": {
            "path": str(spec.source),
            "sha256": file_sha256(spec.source),
        },
        "machine": {
            "target": config.target,
            "rocm_path": str(config.rocm_path) if config.rocm_path else None,
        },
        "values": values,
        "config_bindings": spec.config_bindings(values),
    }


def link_spec_row(args: argparse.Namespace, spec: KernelSpec, values: dict[str, Any], config: BenchConfig) -> dict[str, Any]:
    row = base_spec_row(args, spec, values, config, action="link")
    output = args.output_dir / f"{spec.id}.linked.loom"
    result = run_command(
        [
            config.tools.require_loom_link(),
            spec.source,
            "--mode=link",
            "--to=text",
            "--require-resolved-config",
            f"--root={spec.root_symbol}",
            f"--output={output}",
            *config_args(spec.config_bindings(values)),
        ],
        env=config.command_env(),
    )
    row["link"] = command_evidence(result, args.output_dir, prefix="link")
    row["link"]["output"] = str(output) if output.exists() else None
    row["status"] = "linked" if result.returncode == 0 else "link_failed"
    return row


def compile_spec_row(args: argparse.Namespace, spec: KernelSpec, values: dict[str, Any], config: BenchConfig) -> dict[str, Any]:
    row = base_spec_row(args, spec, values, config, action="compile")
    report = args.output_dir / f"{spec.id}.compile_report.json"
    manifest = args.output_dir / f"{spec.id}.artifact_manifest.json"
    artifact = args.output_dir / f"{spec.id}.artifact.bin"
    target_artifact = args.output_dir / f"{spec.id}.target.bin"
    result = run_command(
        [
            config.tools.require_loom_compile(),
            spec.source,
            "--backend=amdgpu-hal",
            f"--target={config.target}",
            f"--root={spec.root_symbol}",
            f"--output={artifact}",
            f"--emit-target-artifact={target_artifact}",
            "--compile-report=details",
            f"--compile-report-output={report}",
            "--artifact-manifest=analysis",
            f"--emit-artifact-manifest={manifest}",
            *config_args(spec.config_bindings(values)),
        ],
        env=config.command_env(),
    )
    row["compile"] = command_evidence(result, args.output_dir, prefix="compile")
    row["compile"].update(
        {
            "report": str(report) if report.exists() else None,
            "manifest": str(manifest) if manifest.exists() else None,
            "artifact": str(artifact) if artifact.exists() else None,
            "target_artifact": str(target_artifact) if target_artifact.exists() else None,
        }
    )
    row["status"] = "compiled" if result.returncode == 0 else "compile_failed"
    return row
