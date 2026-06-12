#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from hrx2_pipeline_lib import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CATALOG,
    DEFAULT_HRX_INSTALL,
    DEFAULT_LLAMA_BUILD,
    DEFAULT_SOURCE_ROOT,
    WORKSPACE,
    append_jsonl,
    env_for_tools,
    read_jsonl,
    shape_identity,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HRX2 phase0.3 Loom compile and ggml validation sweep.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-root", default=str(WORKSPACE / "cache" / "hrx2" / "runs"))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--loom-compile", default=str(DEFAULT_HRX_INSTALL / "bin" / "loom-compile"))
    parser.add_argument("--test-backend-ops", default=str(DEFAULT_LLAMA_BUILD / "bin" / "test-backend-ops"))
    parser.add_argument("--backend", default="HRX20")
    parser.add_argument("--skip-ggml", action="store_true")
    parser.add_argument("--only-selected", action="store_true", help="Compile only candidates selected by catalog priority.")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def safe_name(text):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:180] or "candidate"


def run_command(cmd, env, cwd, timeout):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def source_for_candidate(candidate, source_root, artifact_root):
    artifact = candidate.get("artifact") or {}
    if artifact.get("path"):
        artifact_path = artifact_root / artifact["path"]
        if artifact_path.exists():
            return artifact_path, "loom-bytecode"
    source = candidate.get("source") or {}
    source_path = source_root / source.get("path", "")
    return source_path, "loom-text"


def parse_csv_rows(text):
    lines = [line for line in text.splitlines() if line.startswith('"')]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def ggml_validation(candidate, args, candidate_dir, env):
    trace_path = candidate_dir / "ggml_trace.jsonl"
    evidence_dir = candidate_dir / "ggml_evidence"
    cmd = [
        str(args.test_backend_ops),
        "test",
        "-b",
        args.backend,
        "-o",
        candidate["shape"]["op"],
        "-p",
        candidate["test_backend_filter"],
        "--output",
        "csv",
    ]
    run = run_command(
        cmd,
        env_for_tools({
            **env,
            "GGML_HRX2_TRACE_JSONL": trace_path,
            "GGML_HRX2_EVIDENCE_DIR": evidence_dir,
        }),
        WORKSPACE,
        args.timeout,
    )
    rows = parse_csv_rows(run["stdout"])
    supported_rows = [row for row in rows if row.get("supported") == "1"]
    trace_events = read_jsonl(trace_path)
    dispatches = [
        event for event in trace_events
        if event.get("event") == "dispatch" and event.get("cache_key") == candidate["cache_key"]
    ]
    if run["returncode"] == 0 and not rows:
        status = "no_matching_ggml_case"
    elif run["returncode"] == 0 and supported_rows and dispatches:
        status = "pass"
    else:
        status = "failed"
    return {
        "status": status,
        "command": cmd,
        "returncode": run["returncode"],
        "timed_out": run["timed_out"],
        "stdout_path": str(candidate_dir / "ggml_stdout.csv"),
        "stderr_path": str(candidate_dir / "ggml_stderr.log"),
        "trace_path": str(trace_path),
        "evidence_dir": str(evidence_dir),
        "supported_rows": len(supported_rows),
        "dispatches": len(dispatches),
    }, run


def main():
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    source_root = Path(args.source_root)
    artifact_root = Path(args.artifact_root)
    run_dir = Path(args.out_root) / args.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copy2(args.candidates, run_dir / "candidates.jsonl")

    results_path = run_dir / "results.jsonl"
    for index, candidate in enumerate(candidates):
        if args.only_selected and not candidate.get("selected_by_priority"):
            continue
        name = f"{index:04d}_{candidate['route']['id']}_{shape_identity(candidate['shape'])}"
        candidate_dir = run_dir / safe_name(name)
        candidate_dir.mkdir(parents=True)
        write_json(candidate_dir / "candidate.json", candidate)

        source_path, source_format = source_for_candidate(candidate, source_root, artifact_root)
        compile_dir = candidate_dir / "loom"
        compile_dir.mkdir()
        compile_cmd = [
            str(args.loom_compile),
            str(source_path),
            "--backend=amdgpu-hal",
            f"--target={candidate['target_key']}",
            f"--compile-root={candidate['route']['root_symbol']}",
            "--module-name=ggml_hrx2",
            f"--output={compile_dir / 'out.hsaco'}",
            f"--emit-artifact-manifest={compile_dir / 'manifest.json'}",
            "--compile-report=details",
            f"--compile-report-output={compile_dir / 'compile_report.json'}",
            "--compile-report-row-limit=64",
        ]
        for binding in candidate.get("config_bindings", []):
            compile_cmd.append(f"--config={binding['key']}={binding['value']}")

        compile_run = run_command(compile_cmd, env_for_tools(), WORKSPACE, args.timeout)
        (compile_dir / "stdout.log").write_text(compile_run["stdout"], encoding="utf-8")
        (compile_dir / "stderr.log").write_text(compile_run["stderr"], encoding="utf-8")

        hsaco_path = compile_dir / "out.hsaco"
        compile_status = {
            "status": "pass" if compile_run["returncode"] == 0 and hsaco_path.exists() and hsaco_path.stat().st_size > 0 else "failed",
            "command": compile_cmd,
            "returncode": compile_run["returncode"],
            "timed_out": compile_run["timed_out"],
            "source_path": str(source_path),
            "source_format": source_format,
            "hsaco_path": str(hsaco_path),
            "hsaco_size": hsaco_path.stat().st_size if hsaco_path.exists() else 0,
            "compile_report_path": str(compile_dir / "compile_report.json"),
            "manifest_path": str(compile_dir / "manifest.json"),
        }

        if args.skip_ggml:
            ggml_status = {"status": "skipped"}
            ggml_run = None
        elif candidate.get("selected_by_priority"):
            ggml_status, ggml_run = ggml_validation(candidate, args, candidate_dir, {})
            Path(ggml_status["stdout_path"]).write_text(ggml_run["stdout"], encoding="utf-8")
            Path(ggml_status["stderr_path"]).write_text(ggml_run["stderr"], encoding="utf-8")
        else:
            ggml_status = {"status": "skipped_non_selected_route"}
            ggml_run = None

        result = {
            "schema": "hrx2-sweep-result-v1",
            "run_id": args.run_id,
            "candidate_index": index,
            "candidate_dir": str(candidate_dir),
            "target_key": candidate["target_key"],
            "shape_id": candidate["shape_id"],
            "route_id": candidate["route"]["id"],
            "cache_key": candidate["cache_key"],
            "selected_by_priority": bool(candidate.get("selected_by_priority")),
            "loom_compile": compile_status,
            "ggml_validation": ggml_status,
        }
        append_jsonl(results_path, result)
        print(
            f"{index + 1}/{len(candidates)} {candidate['route']['id']} "
            f"compile={compile_status['status']} ggml={ggml_status['status']}",
            file=sys.stderr,
        )

    print(f"wrote sweep results to {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
