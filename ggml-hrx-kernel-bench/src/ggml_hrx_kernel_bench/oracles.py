from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hrx2 import Candidate


QK_K = 256
Q4_K_BLOCK_BYTES = 144
F32_BYTES = 4


@dataclass(frozen=True)
class OracleResult:
    status: str
    oracle: str | None
    fixture_dir: Path | None
    metadata_path: Path | None
    expected_path: Path | None
    tolerance: dict[str, float] | None
    message: str | None = None

    def to_ledger(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "oracle": self.oracle,
            "fixture_dir": str(self.fixture_dir) if self.fixture_dir else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "expected_path": str(self.expected_path) if self.expected_path else None,
            "tolerance": self.tolerance,
            "message": self.message,
        }


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for fixture and golden generation; install the numpy extra") from exc
    return np


def q4_k_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q4_K_BLOCK_BYTES


def f32_pattern(np: Any, shape: tuple[int, ...], *, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    pattern = np.arange(values.size, dtype=np.float32).reshape(shape)
    values += (((pattern * 17 + seed * 29) % 257) - 128).astype(np.float32) / 251.0
    return (values * np.float32(scale)).astype(np.float32)


def q4_k_pattern(np: Any, k: int, rows: int, *, seed: int):
    blocks = rows * (k // QK_K)
    data = np.zeros((blocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    d = np.float16(0.5).view(np.uint16)
    dmin = np.float16(0.125).view(np.uint16)
    data[:, 0] = int(d) & 0xFF
    data[:, 1] = (int(d) >> 8) & 0xFF
    data[:, 2] = int(dmin) & 0xFF
    data[:, 3] = (int(dmin) >> 8) & 0xFF
    data[:, 4:16] = rng.integers(1, 64, size=(blocks, 12), dtype=np.uint8)
    data[:, 16:144] = rng.integers(0, 256, size=(blocks, 128), dtype=np.uint8)
    return data.reshape(-1)


def dequant_q4_k(np: Any, packed: Any, k: int, rows: int):
    blocks_per_row = k // QK_K
    blocks = packed.reshape(rows * blocks_per_row, Q4_K_BLOCK_BYTES)
    out = np.empty((rows, k), dtype=np.float32)
    for row in range(rows):
        for block_in_row in range(blocks_per_row):
            block = blocks[row * blocks_per_row + block_in_row]
            d = block[0:2].copy().view(np.float16).astype(np.float32)[0]
            dmin = block[2:4].copy().view(np.float16).astype(np.float32)[0]
            scales = block[4:16].astype(np.uint32)
            qs = block[16:144].astype(np.uint32)
            for group in range(8):
                if group < 4:
                    scale_i = scales[group] & 0x3F
                    min_i = scales[group + 4] & 0x3F
                else:
                    low = scales[group - 4]
                    mid = scales[group]
                    high = scales[group + 4]
                    scale_i = (high & 0x0F) | ((low >> 6) << 4)
                    min_i = (high >> 4) | ((mid >> 6) << 4)
                scale = np.float32(d * np.float32(scale_i))
                minimum = np.float32(dmin * np.float32(min_i))
                byte_base = (group // 2) * 32
                group_values = np.empty((32,), dtype=np.float32)
                for j in range(32):
                    q_byte = qs[byte_base + j]
                    q = (q_byte >> 4) if group % 2 else (q_byte & 0x0F)
                    group_values[j] = np.float32(scale * np.float32(q) - minimum)
                offset = block_in_row * QK_K + group * 32
                out[row, offset : offset + 32] = group_values
    return out


def candidate_seed(candidate: Candidate) -> int:
    text = candidate.id.encode("utf-8")
    value = 0
    for byte in text:
        value = ((value * 131) + byte) & 0xFFFFFFFF
    return value or 1


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_oracle(candidate: Candidate, fixture_dir: Path, *, force: bool = False) -> OracleResult:
    np = require_numpy()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    seed = candidate_seed(candidate)
    family = candidate.family
    try:
        if family == "mul_mat_q4_k_f32":
            return _mul_mat_q4_k_f32(np, candidate, fixture_dir, seed)
        if family == "rms_norm_f32":
            return _rms_norm_f32(np, candidate, fixture_dir, seed)
        if family == "copy_f32_f16":
            return _copy_f32_f16(np, candidate, fixture_dir, seed)
        if family == "cont_f32":
            return _cont_f32(np, candidate, fixture_dir, seed)
    except Exception as exc:
        if force:
            raise
        return OracleResult("oracle_failed", family, fixture_dir, None, None, None, str(exc))
    return OracleResult(
        "unsupported_golden",
        None,
        fixture_dir,
        None,
        None,
        None,
        f"no NumPy oracle implemented for family {family}",
    )


def _mul_mat_q4_k_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    k = int(candidate.shape.get("k", 256))
    rows = int(candidate.shape.get("rows", 1))
    cols = int(candidate.shape.get("cols", 1))
    src0 = q4_k_pattern(np, k, rows, seed=seed)
    src1 = f32_pattern(np, (cols, k), seed=seed + 1)
    weights = dequant_q4_k(np, src0, k, rows)
    expected = np.matmul(weights.astype(np.float32), src1.T.astype(np.float32)).T.reshape(cols * rows)
    dst_init = f32_pattern(np, (cols * rows,), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src0.npy", src0.view(np.int8), allow_pickle=False)
    np.save(fixture_dir / "src1.npy", src1.reshape(cols * k), allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.astype(np.float32), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.astype(np.float32), allow_pickle=False)
    meta = _metadata(candidate, seed, "mul_mat_q4_k_f32_numpy_dequant_matmul", {"atol": 0.08, "rtol": 0.02})
    meta["bytes"] = {
        "src0": q4_k_bytes(k, rows),
        "src1": k * cols * F32_BYTES,
        "dst": rows * cols * F32_BYTES,
    }
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _rms_norm_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    ncols = int(candidate.shape.get("ncols", candidate.shape.get("cols", 1)))
    nrows = int(candidate.shape.get("nrows", candidate.shape.get("rows", 1)))
    eps = np.float32(0.0)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    scale = np.reciprocal(np.sqrt(np.mean(src * src, axis=1, keepdims=True) + eps)).astype(np.float32)
    expected = (src * scale).astype(np.float32)
    dst_init = f32_pattern(np, (nrows, ncols), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src.npy", src.reshape(nrows * ncols), allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.reshape(nrows * ncols), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.reshape(nrows * ncols), allow_pickle=False)
    meta = _metadata(candidate, seed, "rms_norm_f32_numpy", {"atol": 1e-4, "rtol": 1e-4})
    meta["eps"] = float(eps)
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _copy_f32_f16(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    n = int(candidate.values.get("shape.copy.n") or candidate.shape.get("ncols", 1) * candidate.shape.get("nrows", 1))
    src = f32_pattern(np, (n,), seed=seed)
    expected = src.astype(np.float16).view(np.uint16)
    dst_init = np.zeros((n,), dtype=np.uint16)
    np.save(fixture_dir / "src0.npy", src, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init.view(np.int16), allow_pickle=False)
    np.save(fixture_dir / "expected.npy", expected.view(np.int16), allow_pickle=False)
    meta = _metadata(candidate, seed, "copy_f32_f16_numpy_cast_bits", {"atol": 0.0, "rtol": 0.0})
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _cont_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    ncols = int(candidate.shape.get("ncols", candidate.shape.get("cols", 1)))
    nrows = int(candidate.shape.get("nrows", candidate.shape.get("rows", 1)))
    element_count = ncols * nrows
    src = f32_pattern(np, (element_count,), seed=seed)
    dst_init = f32_pattern(np, (element_count,), seed=seed + 2, scale=0.25)
    np.save(fixture_dir / "src0.npy", src, allow_pickle=False)
    np.save(fixture_dir / "dst_init.npy", dst_init, allow_pickle=False)
    np.save(fixture_dir / "expected.npy", src.copy(), allow_pickle=False)
    meta = _metadata(candidate, seed, "cont_f32_numpy_copy", {"atol": 0.0, "rtol": 0.0})
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", meta["oracle"], fixture_dir, meta_path, fixture_dir / "expected.npy", meta["tolerance"])


def _metadata(candidate: Candidate, seed: int, oracle: str, tolerance: dict[str, float]) -> dict[str, Any]:
    return {
        "schema": "ggml_hrx_kernel_bench.oracle.v1",
        "candidate_id": candidate.id,
        "family": candidate.family,
        "op": candidate.op,
        "route_id": candidate.route_id,
        "root_symbol": candidate.root_symbol,
        "shape": candidate.shape,
        "values": candidate.values,
        "seed": seed,
        "oracle": oracle,
        "tolerance": tolerance,
    }


def write_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str | None, dict[str, Any]]:
    family = candidate.family
    if family == "mul_mat_q4_k_f32":
        return _write_mul_mat_q4_workbench(candidate, linked_source, workbench_path, fixture_dir)
    if family == "rms_norm_f32":
        return _write_rms_norm_workbench(candidate, linked_source, workbench_path, fixture_dir)
    if family == "copy_f32_f16":
        return _write_copy_workbench(candidate, linked_source, workbench_path, fixture_dir)
    if family == "cont_f32":
        return _write_cont_workbench(candidate, linked_source, workbench_path, fixture_dir)
    return None, {"status": "unsupported_golden", "message": f"no generated check.case for family {family}"}


def _source_plus_case(linked_source: Path, workbench_path: Path, suffix: str) -> None:
    text = linked_source.read_text(encoding="utf-8")
    workbench_path.write_text(text.rstrip() + "\n\n" + suffix.lstrip(), encoding="utf-8")


def _rel_fixture(workbench_path: Path, fixture_dir: Path, name: str) -> str:
    return str((fixture_dir / name).relative_to(workbench_path.parent))


def _write_mul_mat_q4_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    k = int(candidate.shape.get("k", 256))
    rows = int(candidate.shape.get("rows", 1))
    cols = int(candidate.shape.get("cols", 1))
    src0_elems = q4_k_bytes(k, rows)
    src1_elems = k * cols
    dst_elems = rows * cols
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{src0_elems}xi8>
  %src1 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src1.npy")}") : tensor<{src1_elems}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{dst_elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{dst_elems}xf32>
  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>
  check.return
}}

check.benchmark<{case_name}> {bench_name}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_rms_norm_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols = int(candidate.shape.get("ncols", candidate.shape.get("cols", 1)))
    nrows = int(candidate.shape.get("nrows", candidate.shape.get("rows", 1)))
    elems = ncols * nrows
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %eps = check.literal value(0.0) : f32
  %src = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src.npy")}") : tensor<{elems}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{elems}xf32>
  func.call {candidate.root_symbol}(%eps, %src, %dst) : (f32, tensor<{elems}xf32>, tensor<{elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>
  check.return
}}

check.benchmark<{case_name}> {bench_name}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_copy_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    n = int(candidate.values.get("shape.copy.n") or candidate.shape.get("ncols", 1) * candidate.shape.get("nrows", 1))
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{n}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{n}xi16>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{n}xi16>
  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{n}xf32>, tensor<{n}xi16>)
  check.expect.equal actual(%dst) expected(%expected) : tensor<{n}xi16>
  check.return
}}

check.benchmark<{case_name}> {bench_name}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _write_cont_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols = int(candidate.shape.get("ncols", candidate.shape.get("cols", 1)))
    nrows = int(candidate.shape.get("nrows", candidate.shape.get("rows", 1)))
    elems = ncols * nrows
    case_name = f"@case_{candidate.id}"
    bench_name = f"@bench_{candidate.id}"
    suffix = f"""
check.case public {case_name} {{
  %src0 = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}") : tensor<{elems}xf32>
  %dst = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "dst_init.npy")}") : tensor<{elems}xf32>
  %expected = check.file.read.npy path("{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}") : tensor<{elems}xf32>
  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{elems}xf32>, tensor<{elems}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0) rtol(0.0) nan(same) : tensor<{elems}xf32>
  check.return
}}

check.benchmark<{case_name}> {bench_name}
"""
    _source_plus_case(linked_source, workbench_path, suffix)
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}

