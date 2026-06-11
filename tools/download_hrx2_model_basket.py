#!/usr/bin/env python3
"""Download the HRX2 llama.cpp GGUF model basket.

The script intentionally uses curl instead of huggingface_hub so it works in a
fresh workspace with only standard Python and system curl available.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote


GIB = 1024**3
DEFAULT_DEST = pathlib.Path("shared/models/llamacpp-hrx2-basket-v1")


@dataclass(frozen=True)
class BasketFile:
    repo: str
    filename: str
    quant: str
    size: int
    tags: tuple[str, ...]
    reason: str

    @property
    def url(self) -> str:
        return (
            "https://huggingface.co/"
            + self.repo
            + "/resolve/main/"
            + quote(self.filename)
        )

    @property
    def model_dir(self) -> str:
        return self.repo.replace("/", "__")


BASKET: tuple[BasketFile, ...] = (
    BasketFile(
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename="Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf",
        quant="UD-Q4_K_XL",
        size=17690497440,
        tags=("lean", "coverage", "full"),
        reason="Popular current MoE default-style quant; route and MoE fusion coverage.",
    ),
    BasketFile(
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename="Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf",
        quant="Q6_K",
        size=25092532640,
        tags=("coverage", "full"),
        reason="Higher-quality K-quant for quant sensitivity.",
    ),
    BasketFile(
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename="Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf",
        quant="IQ4_XS",
        size=16378073504,
        tags=("full",),
        reason="IQ quant comparison for the main MoE shape.",
    ),
    BasketFile(
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename="Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf",
        quant="Q5_K_M",
        size=21725581728,
        tags=("full",),
        reason="Q5 K-quant comparison for the main MoE shape.",
    ),
    BasketFile(
        repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        filename="Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf",
        quant="Q8_0",
        size=32483932576,
        tags=("full",),
        reason="Q8 comparison for the main MoE shape.",
    ),
    BasketFile(
        repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=18556689568,
        tags=("coverage", "full"),
        reason="Popular coding MoE shape with different graph pressure.",
    ),
    BasketFile(
        repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
        quant="UD-Q4_K_XL",
        size=17665334432,
        tags=("full",),
        reason="UD quant comparison for coding MoE.",
    ),
    BasketFile(
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=4920739232,
        tags=("lean", "coverage", "full"),
        reason="Extremely common dense baseline.",
    ),
    BasketFile(
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        quant="Q8_0",
        size=8540775840,
        tags=("coverage", "full"),
        reason="Dense high-precision quant path.",
    ),
    BasketFile(
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf",
        quant="IQ4_XS",
        size=4447667616,
        tags=("full",),
        reason="Dense IQ quant comparison.",
    ),
    BasketFile(
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        quant="Q5_K_M",
        size=5732992416,
        tags=("full",),
        reason="Dense Q5 quant comparison.",
    ),
    BasketFile(
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=2019377696,
        tags=("lean", "coverage", "full"),
        reason="Small model overhead sensitivity.",
    ),
    BasketFile(
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q8_0.gguf",
        quant="Q8_0",
        size=3421899296,
        tags=("full",),
        reason="Small dense Q8 overhead sensitivity.",
    ),
    BasketFile(
        repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q5_k_m.gguf",
        quant="Q5_K_M",
        size=5444831232,
        tags=("lean", "coverage", "full"),
        reason="Official Qwen coding dense model and Q5_K coverage.",
    ),
    BasketFile(
        repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant="Q4_K_M",
        size=4683073536,
        tags=("full",),
        reason="Official Qwen dense Q4 comparison.",
    ),
    BasketFile(
        repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q6_k.gguf",
        quant="Q6_K",
        size=6254198784,
        tags=("full",),
        reason="Official Qwen dense Q6 comparison.",
    ),
    BasketFile(
        repo="unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=8988109984,
        tags=("lean", "coverage", "full"),
        reason="Reasoning graph and Qwen-derived dense shape.",
    ),
    BasketFile(
        repo="unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf",
        quant="Q6_K",
        size=12124683424,
        tags=("full",),
        reason="Reasoning model Q6 comparison.",
    ),
    BasketFile(
        repo="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        filename="Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=14333922848,
        tags=("lean", "coverage", "full"),
        reason="Large dense architecture, attention and long-context pressure.",
    ),
    BasketFile(
        repo="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        filename="Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf",
        quant="Q6_K",
        size=19345952288,
        tags=("full",),
        reason="Large dense Q6 comparison.",
    ),
    BasketFile(
        repo="bartowski/google_gemma-3-27b-it-GGUF",
        filename="google_gemma-3-27b-it-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=16546404992,
        tags=("lean", "coverage", "full"),
        reason="Gemma architecture and graph corner coverage.",
    ),
    BasketFile(
        repo="bartowski/google_gemma-3-27b-it-GGUF",
        filename="google_gemma-3-27b-it-Q6_K.gguf",
        quant="Q6_K",
        size=22166690432,
        tags=("full",),
        reason="Gemma Q6 comparison.",
    ),
    BasketFile(
        repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        filename="microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size=2491874688,
        tags=("lean", "coverage", "full"),
        reason="Small dense corner case and overhead sensitivity.",
    ),
    BasketFile(
        repo="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        filename="microsoft_Phi-4-mini-instruct-Q8_0.gguf",
        quant="Q8_0",
        size=4084611456,
        tags=("full",),
        reason="Small dense Q8 comparison.",
    ),
)


def selected_basket(profile: str) -> list[BasketFile]:
    return [entry for entry in BASKET if profile in entry.tags]


def format_size(size: int) -> str:
    return f"{size / GIB:.2f} GiB"


def local_path(dest: pathlib.Path, entry: BasketFile) -> pathlib.Path:
    return dest / entry.model_dir / entry.filename


def manifest_rows(dest: pathlib.Path, entries: Iterable[BasketFile], profile: str) -> list[dict[str, object]]:
    rows = []
    for entry in entries:
        path = local_path(dest, entry)
        rows.append(
            {
                "profile": profile,
                "repo": entry.repo,
                "filename": entry.filename,
                "quant": entry.quant,
                "expected_size": entry.size,
                "expected_size_gib": round(entry.size / GIB, 3),
                "url": entry.url,
                "local_path": str(path),
                "reason": entry.reason,
            }
        )
    return rows


def write_manifest(dest: pathlib.Path, rows: list[dict[str, object]], profile: str) -> None:
    manifest = {
        "schema_version": 1,
        "name": "llamacpp-hrx2-basket-v1",
        "profile": profile,
        "total_expected_size": sum(int(row["expected_size"]) for row in rows),
        "total_expected_size_gib": round(
            sum(int(row["expected_size"]) for row in rows) / GIB, 3
        ),
        "files": rows,
    }
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "basket_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def check_existing(path: pathlib.Path, expected_size: int) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size == expected_size:
        return "complete"
    if size < expected_size:
        return f"partial {format_size(size)} / {format_size(expected_size)}"
    return f"oversized {format_size(size)} / {format_size(expected_size)}"


def run_curl(entry: BasketFile, path: pathlib.Path, token: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--output",
        str(path),
        entry.url,
    ]
    if token:
        cmd[1:1] = ["--header", f"Authorization: Bearer {token}"]
    print(f"Downloading {entry.repo}:{entry.filename}")
    subprocess.run(cmd, check=True)
    actual = path.stat().st_size
    if actual != entry.size:
        raise RuntimeError(
            f"size mismatch for {path}: got {actual}, expected {entry.size}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("lean", "coverage", "full"),
        default="coverage",
        help="model basket profile to download",
    )
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=DEFAULT_DEST,
        help=f"destination directory, default {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected files without downloading",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write basket_manifest.json even during --dry-run",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="environment variable containing an optional Hugging Face token",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload files even when the expected local size is present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = selected_basket(args.profile)
    dest = args.dest
    total_size = sum(entry.size for entry in entries)

    print(f"Profile: {args.profile}")
    print(f"Destination: {dest}")
    print(f"Files: {len(entries)}")
    print(f"Expected size: {format_size(total_size)}")
    print()

    rows = manifest_rows(dest, entries, args.profile)
    for entry, row in zip(entries, rows):
        path = pathlib.Path(str(row["local_path"]))
        status = check_existing(path, entry.size)
        print(
            f"{entry.repo}:{entry.filename}\n"
            f"  quant={entry.quant} size={format_size(entry.size)} status={status}\n"
            f"  path={path}\n"
            f"  url={entry.url}"
        )

    if args.dry_run:
        if args.write_manifest:
            write_manifest(dest, rows, args.profile)
        return 0

    if not shutil.which("curl"):
        print("error: curl is required", file=sys.stderr)
        return 2

    token = os.environ.get(args.hf_token_env)
    for entry in entries:
        path = local_path(dest, entry)
        if not args.force and check_existing(path, entry.size) == "complete":
            print(f"Skipping complete file: {path}")
            continue
        run_curl(entry, path, token)

    write_manifest(dest, rows, args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
