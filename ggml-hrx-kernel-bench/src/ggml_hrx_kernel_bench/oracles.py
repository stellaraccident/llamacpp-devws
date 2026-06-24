from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .hrx2 import Candidate


QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q8_0_BLOCK_BYTES = 34
F32_BYTES = 4
Q8_1_BLOCK_BYTES = 36


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


@dataclass(frozen=True)
class OracleSpec:
    family_ids: tuple[str, ...]
    generate: Callable[[Any, Candidate, Path, int], OracleResult]
    write_workbench: Callable[[Candidate, Path, Path, Path], tuple[str | None, dict[str, Any]]]


@dataclass(frozen=True)
class LogicalOracleSpec:
    family_ids: tuple[str, ...]
    oracle: str
    tolerance: dict[str, float]
    build: Callable[[Any, Candidate, int], dict[str, Any]]
    exact_kernel_abi: bool = False


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


def q8_1_bytes(ncols: int, nrows: int) -> int:
    return nrows * ((ncols + 31) // 32) * Q8_1_BLOCK_BYTES


def q8_0_bytes(k: int, rows: int) -> int:
    if k % 32 != 0:
        raise ValueError(f"k must be a multiple of 32: {k}")
    return rows * (k // 32) * Q8_0_BLOCK_BYTES


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
    data[:, 4:16] = rng.integers(1, 16, size=(blocks, 12), dtype=np.uint8)
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


def quantize_q8_0(np: Any, values: Any) -> Any:
    rows, k = values.shape
    if k % 32 != 0:
        raise ValueError(f"k must be a multiple of 32: {k}")
    blocks_per_row = k // 32
    packed = np.zeros((rows * blocks_per_row, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    for row in range(rows):
        for block in range(blocks_per_row):
            chunk = values[row, block * 32 : (block + 1) * 32].astype(np.float32)
            amax = np.max(np.abs(chunk))
            d = np.float32(amax / 127.0) if amax != 0 else np.float32(0.0)
            qs = np.rint(chunk / d).astype(np.int32) if d != 0 else np.zeros((32,), dtype=np.int32)
            qs = np.clip(qs, -128, 127).astype(np.int8)
            linear = row * blocks_per_row + block
            packed[linear, 0:2] = np.array([d], dtype=np.float16).view(np.uint8)
            packed[linear, 2:34] = qs.view(np.uint8)
    return packed.reshape(-1).view(np.int8)


def dequant_q8_0(np: Any, packed: Any, k: int, rows: int) -> Any:
    blocks_per_row = k // 32
    blocks = packed.view(np.uint8).reshape(rows * blocks_per_row, Q8_0_BLOCK_BYTES)
    out = np.empty((rows, k), dtype=np.float32)
    for row in range(rows):
        for block in range(blocks_per_row):
            raw = blocks[row * blocks_per_row + block]
            d = raw[0:2].copy().view(np.float16).astype(np.float32)[0]
            qs = raw[2:34].copy().view(np.int8).astype(np.float32)
            out[row, block * 32 : (block + 1) * 32] = qs * d
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
    spec = ORACLE_SPECS_BY_FAMILY.get(family)
    if spec is None:
        return OracleResult(
            "unsupported_golden",
            None,
            fixture_dir,
            None,
            None,
            None,
            f"no NumPy oracle implemented for family {family}",
        )
    try:
        return spec.generate(np, candidate, fixture_dir, seed)
    except Exception as exc:
        if force:
            raise
        return OracleResult("oracle_failed", family, fixture_dir, None, None, None, str(exc))


def _mul_mat_q4_k_f32(np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    if "split_k_reduce2" in candidate.root_symbol:
        spec = LogicalOracleSpec(
            ("mul_mat_q4_k_f32",),
            "split_k_reduce2_f32_numpy_logical",
            {"atol": 1e-5, "rtol": 1e-5},
            _split_k_reduce2_arrays,
        )
        return _logical_oracle(spec, np, candidate, fixture_dir, seed)
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


def _dims(candidate: Candidate) -> tuple[int, int, int]:
    ncols = int(candidate.shape.get("ncols", candidate.shape.get("cols", candidate.shape.get("k", 1))))
    nrows = int(candidate.shape.get("nrows", candidate.shape.get("rows", 1)))
    return ncols, nrows, ncols * nrows


def _matmul_dims(candidate: Candidate) -> tuple[int, int, int]:
    k = int(candidate.shape.get("k", candidate.shape.get("ncols", candidate.shape.get("cols", 256))))
    rows = int(candidate.shape.get("rows", candidate.shape.get("nrows", 1)))
    cols = int(candidate.shape.get("cols", candidate.shape.get("ncols", 1)))
    return k, rows, cols


def _write_arrays(np: Any, fixture_dir: Path, arrays: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = fixture_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = str(path)
    return paths


def _logical_oracle(spec: LogicalOracleSpec, np: Any, candidate: Candidate, fixture_dir: Path, seed: int) -> OracleResult:
    data = spec.build(np, candidate, seed)
    expected = data["arrays"].get("expected")
    if expected is None:
        raise ValueError(f"logical oracle {spec.oracle} did not produce an expected array")
    array_paths = _write_arrays(np, fixture_dir, data["arrays"])
    meta = _metadata(candidate, seed, spec.oracle, spec.tolerance)
    meta["exact_kernel_abi"] = spec.exact_kernel_abi
    meta["oracle_scope"] = "kernel_abi" if spec.exact_kernel_abi else "logical_numpy"
    meta["arrays"] = array_paths
    meta.update(data.get("metadata") or {})
    meta_path = fixture_dir / "oracle.json"
    write_json(meta_path, meta)
    return OracleResult("fixtures_ready", spec.oracle, fixture_dir, meta_path, fixture_dir / "expected.npy", spec.tolerance)


def _logical_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[None, dict[str, Any]]:
    return None, {
        "status": "unsupported_workbench",
        "message": f"{candidate.family} has a NumPy logical oracle but no generated check.case ABI yet",
        "fixture_dir": str(fixture_dir),
    }


def _case_names(candidate: Candidate) -> tuple[str, str]:
    return f"@case_{candidate.id}", f"@bench_{candidate.id}"


def _emit_case(linked_source: Path, workbench_path: Path, case_name: str, bench_name: str, body: str) -> tuple[str, dict[str, Any]]:
    _source_plus_case(
        linked_source,
        workbench_path,
        f"""
check.case public {case_name} {{
{body.rstrip()}
  check.return
}}

check.benchmark<{case_name}> {bench_name}
""",
    )
    return bench_name, {"status": "ok", "workbench_path": str(workbench_path)}


def _read_f32(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xf32>"""


def _read_i32(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xi32>"""


def _read_i16(workbench_path: Path, fixture_dir: Path, name: str, elems: int) -> str:
    return f"""  %{name} = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, name + ".npy")}\") : tensor<{elems}xi16>"""


def _binary_arrays(op: Callable[[Any, Any], Any]) -> Callable[[Any, Candidate, int], dict[str, Any]]:
    def build(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
        ncols, nrows, elems = _dims(candidate)
        src0 = f32_pattern(np, (nrows, ncols), seed=seed)
        src1 = f32_pattern(np, (nrows, ncols), seed=seed + 1)
        expected = op(src0, src1).astype(np.float32)
        return {
            "arrays": {
                "src0": src0.reshape(elems),
                "src1": src1.reshape(elems),
                "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
                "expected": expected.reshape(elems),
            }
        }

    return build


def _div_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src0 = f32_pattern(np, (nrows, ncols), seed=seed)
    src1 = f32_pattern(np, (nrows, ncols), seed=seed + 1) + np.float32(2.0)
    expected = (src0 / src1).astype(np.float32)
    return {
        "arrays": {
            "src0": src0.reshape(elems),
            "src1": src1.reshape(elems),
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(elems),
        }
    }


def _scale_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    scale = np.float32(0.625)
    bias = np.float32(-0.125)
    src0 = f32_pattern(np, (nrows, ncols), seed=seed)
    return {
        "arrays": {
            "src0": src0.reshape(elems),
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": (src0 * scale + bias).astype(np.float32).reshape(elems),
        },
        "metadata": {"scale": float(scale), "bias": float(bias)},
    }


def _clamp_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src0 = f32_pattern(np, (nrows, ncols), seed=seed)
    lo = np.float32(-0.45)
    hi = np.float32(0.55)
    return {
        "arrays": {
            "src0": src0.reshape(elems),
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": np.clip(src0, lo, hi).astype(np.float32).reshape(elems),
        },
        "metadata": {"min": float(lo), "max": float(hi)},
    }


def _rms_norm_mul_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    weight = f32_pattern(np, (ncols,), seed=seed + 1, scale=0.5) + np.float32(1.0)
    scale = np.reciprocal(np.sqrt(np.mean(src * src, axis=1, keepdims=True))).astype(np.float32)
    expected = (src * scale * weight.reshape(1, ncols)).astype(np.float32)
    return {
        "arrays": {
            "src": src.reshape(elems),
            "weight": weight,
            "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(elems),
        }
    }


def _add_rms_norm_mul_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    base = f32_pattern(np, (nrows, ncols), seed=seed)
    residual = f32_pattern(np, (nrows, ncols), seed=seed + 1, scale=0.25)
    weight = f32_pattern(np, (ncols,), seed=seed + 2, scale=0.5) + np.float32(1.0)
    added = (base + residual).astype(np.float32)
    scale = np.reciprocal(np.sqrt(np.mean(added * added, axis=1, keepdims=True))).astype(np.float32)
    expected = (added * scale * weight.reshape(1, ncols)).astype(np.float32)
    return {
        "arrays": {
            "src0": base.reshape(elems),
            "src1": residual.reshape(elems),
            "weight": weight,
            "dst_init": f32_pattern(np, (elems,), seed=seed + 3, scale=0.25),
            "expected": expected.reshape(elems),
        }
    }


def _swiglu_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    x = f32_pattern(np, (nrows, ncols), seed=seed)
    gate = f32_pattern(np, (nrows, ncols), seed=seed + 1)
    if "geglu" in candidate.root_symbol:
        inner = x * (np.float32(1.0) + np.float32(0.044715) * x * x)
        gelu = x / (np.float32(1.0) + np.exp(-np.float32(1.5957691216057308) * inner, dtype=np.float32))
        activated = gelu.astype(np.float32)
    else:
        activated = (x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))).astype(np.float32)
    expected = (activated * gate).astype(np.float32)
    arrays: dict[str, Any] = {
        "dst_init": f32_pattern(np, (elems,), seed=seed + 2, scale=0.25),
        "expected": expected.reshape(elems),
    }
    if candidate.root_symbol.endswith("_split"):
        arrays["src0"] = x.reshape(elems)
        arrays["src1"] = gate.reshape(elems)
    else:
        arrays["src0"] = np.concatenate([x, gate], axis=1).reshape(elems * 2)
    return {
        "arrays": arrays,
        "metadata": {"activation": "gelu" if "geglu" in candidate.root_symbol else "silu"},
    }


def _sum_rows_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, _ = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    expected = np.sum(src, axis=1).astype(np.float32)
    return {"arrays": {"src0": src.reshape(nrows * ncols), "expected": expected}}


def _softmax_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    scale = np.float32(0.75)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    arrays: dict[str, Any] = {
        "src0": src.reshape(elems),
        "dst_init": f32_pattern(np, (elems,), seed=seed + 1, scale=0.25),
    }
    logits = src * scale
    if "mask" in candidate.root_symbol:
        mask = f32_pattern(np, (nrows, ncols), seed=seed + 3, scale=0.125)
        arrays["mask"] = mask.reshape(elems)
        logits = logits + mask
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    expected = (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)
    arrays["expected"] = expected.reshape(elems)
    return {"arrays": arrays, "metadata": {"scale": float(scale), "has_mask": "mask" in candidate.root_symbol}}


def _argsort_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    expected = np.argsort(-src, axis=1).astype(np.int32)
    return {
        "arrays": {
            "src0": src.reshape(elems),
            "dst_init": np.full((elems,), -1, dtype=np.int32),
            "expected": expected.reshape(elems),
        }
    }


def _get_rows_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src0_nrows = int(candidate.values.get("shape.get_rows.src0_nrows") or candidate.values.get("get_rows.src0_nrows") or nrows)
    src = f32_pattern(np, (src0_nrows, ncols), seed=seed)
    indices = ((np.arange(nrows, dtype=np.int64) * 3 + seed) % src0_nrows).astype(np.int32)
    expected = src[indices].astype(np.float32)
    return {
        "arrays": {
            "src0": src.reshape(src0_nrows * ncols),
            "indices": indices,
            "dst_init": f32_pattern(np, (elems,), seed=seed + 1, scale=0.25),
            "expected": expected.reshape(elems),
        },
        "metadata": {"src0_nrows": src0_nrows},
    }


def _set_rows_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    dst = f32_pattern(np, (nrows, ncols), seed=seed, scale=0.25)
    src = f32_pattern(np, (nrows, ncols), seed=seed + 1)
    indices = np.zeros((nrows * 2,), dtype=np.int32)
    indices[0::2] = np.arange(nrows, dtype=np.int32)
    expected = dst.copy()
    expected[np.arange(nrows, dtype=np.int32)] = src
    if "f16" in candidate.root_symbol:
        dst_bits = dst.astype(np.float16).view(np.uint16).reshape(elems).view(np.int16)
        expected_bits = expected.astype(np.float16).view(np.uint16).reshape(elems).view(np.int16)
    else:
        dst_bits = dst.reshape(elems)
        expected_bits = expected.reshape(elems)
    return {
        "arrays": {
            "src0": src.reshape(elems),
            "indices": indices,
            "dst_init": dst_bits,
            "expected": expected_bits,
        },
        "metadata": {"dst_type": "i16" if "f16" in candidate.root_symbol else "f32", "idx_i32_count": int(indices.size)},
    }


def _matmul_f32_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    k, rows, cols = _matmul_dims(candidate)
    lhs = f32_pattern(np, (rows, k), seed=seed)
    rhs = f32_pattern(np, (cols, k), seed=seed + 1)
    expected = np.matmul(lhs.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0_logical_f32": lhs.reshape(rows * k),
            "src1": rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {"logical_packed_weight_fixture": False},
    }


def _mul_mat_q8_0_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    if "q8_1" in candidate.root_symbol:
        data = _matmul_f32_arrays(np, candidate, seed)
        data["metadata"]["message"] = "q8_0 matmul with q8_1 RHS requires q8_1 RHS ABI packer"
        data["metadata"]["logical_packed_weight_fixture"] = True
        return data
    k, rows, cols = _matmul_dims(candidate)
    lhs_f32 = f32_pattern(np, (rows, k), seed=seed)
    src0 = quantize_q8_0(np, lhs_f32)
    rhs = f32_pattern(np, (cols, k), seed=seed + 1)
    weights = dequant_q8_0(np, src0, k, rows)
    expected = np.matmul(weights.astype(np.float32), rhs.T.astype(np.float32)).T.astype(np.float32)
    return {
        "arrays": {
            "src0": src0,
            "src1": rhs.reshape(cols * k),
            "dst_init": f32_pattern(np, (rows * cols,), seed=seed + 2, scale=0.25),
            "expected": expected.reshape(rows * cols),
        },
        "metadata": {
            "q8_0_block_bytes": Q8_0_BLOCK_BYTES,
            "bytes": {"src0": q8_0_bytes(k, rows), "src1": k * cols * F32_BYTES, "dst": rows * cols * F32_BYTES},
        },
    }


def _quantized_matmul_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    data = _matmul_f32_arrays(np, candidate, seed)
    data["metadata"]["logical_packed_weight_fixture"] = True
    data["metadata"]["message"] = "logical f32 oracle for packed quantized family; ABI-specific packed fixtures are pending"
    return data


def _quantize_q8_1_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols, nrows, elems = _dims(candidate)
    src = f32_pattern(np, (nrows, ncols), seed=seed)
    block_count = (ncols + 31) // 32
    if "x4" in candidate.root_symbol:
        outer_count = (block_count + 3) // 4
        expected = np.zeros((nrows, outer_count, 144), dtype=np.uint8)
        for row in range(nrows):
            for block in range(block_count):
                outer = block // 4
                inner = block % 4
                start = block * 32
                values = np.zeros((32,), dtype=np.float32)
                chunk = src[row, start : min(start + 32, ncols)]
                values[: chunk.size] = chunk
                absmax = np.max(np.abs(values))
                d = np.float32(absmax / 127.0) if absmax != 0 else np.float32(0.0)
                if d != 0:
                    qs = np.rint(values / d).astype(np.int32)
                else:
                    qs = np.zeros((32,), dtype=np.int32)
                qs = np.clip(qs, -128, 127).astype(np.int8)
                s = np.float32(np.sum(qs.astype(np.float32)) * d)
                expected[row, outer, inner * 4 : inner * 4 + 2] = np.array([d], dtype=np.float16).view(np.uint8)
                expected[row, outer, inner * 4 + 2 : inner * 4 + 4] = np.array([s], dtype=np.float16).view(np.uint8)
                expected[row, outer, 16 + inner * 32 : 16 + inner * 32 + 32] = qs.view(np.uint8)
        return {
            "arrays": {
                "src0": src.reshape(elems),
                "expected": expected.reshape(nrows * outer_count * 144).view(np.int8),
            },
            "metadata": {"q8_1_x4_block_bytes": 144, "block_count": block_count, "outer_count": outer_count},
        }
    expected = np.zeros((nrows, block_count, Q8_1_BLOCK_BYTES), dtype=np.uint8)
    for row in range(nrows):
        for block in range(block_count):
            start = block * 32
            values = np.zeros((32,), dtype=np.float32)
            chunk = src[row, start : min(start + 32, ncols)]
            values[: chunk.size] = chunk
            absmax = np.max(np.abs(values))
            d = np.float32(absmax / 127.0) if absmax != 0 else np.float32(1.0)
            scaled = values / d if d != 0 else np.zeros((32,), dtype=np.float32)
            qs = np.where(scaled < 0, np.ceil(scaled - np.float32(0.5)), np.floor(scaled + np.float32(0.5)))
            qs = np.clip(qs, -128, 127).astype(np.int8)
            s = np.float32(np.sum(qs.astype(np.float32) * d))
            expected[row, block, 0:2] = np.array([d], dtype=np.float16).view(np.uint8)
            expected[row, block, 2:4] = np.array([s], dtype=np.float16).view(np.uint8)
            expected[row, block, 4:36] = qs.view(np.uint8)
    return {
        "arrays": {
            "src0": src.reshape(elems),
            "expected": expected.reshape(nrows * block_count * Q8_1_BLOCK_BYTES).view(np.int8),
        },
        "metadata": {"q8_1_block_bytes": Q8_1_BLOCK_BYTES, "block_count": block_count},
    }


def _rms_norm_mul_quantize_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    base = _rms_norm_mul_arrays(np, candidate, seed)
    normalized = base["arrays"]["expected"].reshape(-1)
    ncols, nrows, _ = _dims(candidate)
    pseudo = Candidate(
        id=candidate.id,
        family=candidate.family,
        op=candidate.op,
        source_id=candidate.source_id,
        source_path=candidate.source_path,
        root_symbol=candidate.root_symbol,
        export_name=candidate.export_name,
        route_id=candidate.route_id,
        route=candidate.route,
        shape=candidate.shape,
        values=candidate.values,
        config=candidate.config,
        dispatch=candidate.dispatch,
        supports=candidate.supports,
        coverage=candidate.coverage,
        status=candidate.status,
        message=candidate.message,
    )
    quant = _quantize_q8_1_arrays(np, pseudo, seed + 7)
    quant["arrays"]["src0"] = normalized.astype(np.float32)
    quant["metadata"]["source_oracle"] = "rms_norm_mul_f32_numpy"
    quant["metadata"]["bytes"] = {"expected": q8_1_bytes(ncols, nrows)}
    return quant


def _rope_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    ncols = int(candidate.values.get("shape.rope.ncols") or candidate.shape.get("ncols", 1))
    n_dims = int(candidate.values.get("shape.rope.n_dims") or candidate.shape.get("n_dims", ncols))
    nheads = int(candidate.values.get("shape.rope.nheads") or candidate.shape.get("rows", 1))
    ntokens = int(candidate.values.get("shape.rope.ntokens") or candidate.shape.get("cols", 1))
    src_head_stride = int(candidate.values.get("shape.rope.src0_head_stride") or ncols)
    src_token_stride = int(candidate.values.get("shape.rope.src0_token_stride") or (ncols * nheads))
    dst_head_stride = int(candidate.values.get("shape.rope.dst_head_stride") or ncols)
    dst_token_stride = int(candidate.values.get("shape.rope.dst_token_stride") or (ncols * nheads))
    pos_token_stride = int(candidate.values.get("shape.rope.pos_token_stride") or 1)
    src_elems = src_token_stride * ntokens
    dst_elems = dst_token_stride * ntokens
    pos_elems = pos_token_stride * ntokens
    src = f32_pattern(np, (src_elems,), seed=seed)
    expected = f32_pattern(np, (dst_elems,), seed=seed + 1, scale=0.25)
    positions = np.zeros((pos_elems,), dtype=np.int32)
    for token in range(ntokens):
        positions[token * pos_token_stride] = token + 1
    half_cols = ncols // 2
    half_dims = n_dims // 2
    theta_scale = np.float32(0.75)
    freq_scale = np.float32(1.1)
    attn_factor = np.float32(0.9)
    output_scale = np.float32(0.5)
    has_freq = "freq" in candidate.root_symbol
    freq = (np.arange(max(half_dims, 1), dtype=np.float32) * np.float32(0.125) + np.float32(1.0)).astype(np.float32)
    neox = "neox" in candidate.root_symbol
    scale_output = "scale" in candidate.root_symbol
    if half_cols:
        itemsize = np.dtype(np.float32).itemsize
        src_view = np.lib.stride_tricks.as_strided(
            src,
            shape=(ntokens, nheads, ncols),
            strides=(src_token_stride * itemsize, src_head_stride * itemsize, itemsize),
            writeable=False,
        )
        dst_view = np.lib.stride_tricks.as_strided(
            expected,
            shape=(ntokens, nheads, ncols),
            strides=(dst_token_stride * itemsize, dst_head_stride * itemsize, itemsize),
            writeable=True,
        )
        pairs = np.arange(half_cols, dtype=np.int32)
        if neox:
            idx0 = pairs
            idx1 = pairs + half_cols
        else:
            idx0 = pairs * 2
            idx1 = idx0 + 1

        x0 = src_view[:, :, idx0]
        x1 = src_view[:, :, idx1]
        out0 = x0.copy()
        out1 = x1.copy()
        active = pairs < half_dims
        if np.any(active):
            active_pairs = pairs[active]
            pos = positions[np.arange(ntokens, dtype=np.int64) * pos_token_stride].astype(np.float32)
            theta = (
                pos[:, None]
                * np.power(theta_scale, active_pairs.astype(np.float32)).astype(np.float32)[None, :]
            ).astype(np.float32)
            if has_freq:
                theta = (theta / freq[active_pairs][None, :]).astype(np.float32)
            theta = (theta * freq_scale).astype(np.float32)
            c = (np.cos(theta).astype(np.float32) * attn_factor).astype(np.float32)
            s = (np.sin(theta).astype(np.float32) * attn_factor).astype(np.float32)
            xa0 = x0[:, :, active]
            xa1 = x1[:, :, active]
            rot0 = (xa0 * c[:, None, :] - xa1 * s[:, None, :]).astype(np.float32)
            rot1 = (xa0 * s[:, None, :] + xa1 * c[:, None, :]).astype(np.float32)
            if scale_output:
                rot0 = (rot0 * output_scale).astype(np.float32)
                rot1 = (rot1 * output_scale).astype(np.float32)
            out0[:, :, active] = rot0
            out1[:, :, active] = rot1
        dst_view[:, :, idx0] = out0.astype(np.float32)
        dst_view[:, :, idx1] = out1.astype(np.float32)
    arrays: dict[str, Any] = {
        "src0": src.astype(np.float32),
        "positions": positions,
        "dst_init": f32_pattern(np, (dst_elems,), seed=seed + 2, scale=0.25),
        "expected": expected.astype(np.float32),
    }
    if has_freq:
        arrays["freq"] = freq.astype(np.float32)
    return {
        "arrays": arrays,
        "metadata": {
            "theta_scale": float(theta_scale),
            "freq_scale": float(freq_scale),
            "attn_factor": float(attn_factor),
            "output_scale": float(output_scale),
            "has_freq": has_freq,
            "neox": neox,
            "src_elems": src_elems,
            "dst_elems": dst_elems,
            "pos_elems": pos_elems,
            "freq_elems": int(freq.size) if has_freq else 0,
        },
    }


def _softmax_kqv_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    k, rows, cols = _matmul_dims(candidate)
    logits = f32_pattern(np, (cols, k), seed=seed)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted, dtype=np.float32)
    weights = (weights / np.sum(weights, axis=1, keepdims=True)).astype(np.float32)
    values = f32_pattern(np, (k, rows), seed=seed + 1)
    expected = np.matmul(weights, values).astype(np.float32)
    return {
        "arrays": {
            "logits": logits.reshape(cols * k),
            "values": values.reshape(k * rows),
            "expected": expected.reshape(cols * rows),
        }
    }


def _split_k_reduce2_arrays(np: Any, candidate: Candidate, seed: int) -> dict[str, Any]:
    rows = int(candidate.shape.get("rows", candidate.shape.get("nrows", 1)))
    plane0 = f32_pattern(np, (rows,), seed=seed)
    plane1 = f32_pattern(np, (rows,), seed=seed + 1)
    src = np.concatenate([plane0, plane1]).astype(np.float32)
    return {
        "arrays": {
            "src0": src,
            "dst_init": f32_pattern(np, (rows,), seed=seed + 2, scale=0.25),
            "expected": (plane0 + plane1).astype(np.float32),
        },
        "metadata": {"layout": "two_plane_f32_reduce2", "rows": rows},
    }


def _logical_generate(spec: LogicalOracleSpec) -> Callable[[Any, Candidate, Path, int], OracleResult]:
    return lambda np, candidate, fixture_dir, seed: _logical_oracle(spec, np, candidate, fixture_dir, seed)


LOGICAL_ORACLE_SPECS: tuple[LogicalOracleSpec, ...] = (
    LogicalOracleSpec(("add_f32",), "add_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _binary_arrays(lambda lhs, rhs: lhs + rhs)),
    LogicalOracleSpec(("mul_f32",), "mul_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _binary_arrays(lambda lhs, rhs: lhs * rhs)),
    LogicalOracleSpec(("div_f32",), "div_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _div_arrays),
    LogicalOracleSpec(("scale_f32",), "scale_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _scale_arrays),
    LogicalOracleSpec(("clamp_f32",), "clamp_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _clamp_arrays),
    LogicalOracleSpec(("argsort_f32_i32",), "argsort_f32_i32_numpy_desc", {"atol": 0.0, "rtol": 0.0}, _argsort_arrays),
    LogicalOracleSpec(("get_rows_f32", "get_rows_q4_k_f32", "get_rows_q5_k_f32", "get_rows_q6_k_f32", "get_rows_q8_0_f32"), "get_rows_numpy_logical", {"atol": 1e-5, "rtol": 1e-5}, _get_rows_arrays),
    LogicalOracleSpec(("get_rows_moe_weights_f32",), "get_rows_moe_weights_numpy_logical", {"atol": 1e-5, "rtol": 1e-5}, _get_rows_arrays),
    LogicalOracleSpec(("mul_mat_f32_f32", "mul_mat_f16_f32_batched", "mul_mat_f16_f32_batched_cont"), "mul_mat_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _matmul_f32_arrays),
    LogicalOracleSpec(("mul_mat_q5_k_f32", "mul_mat_q6_k_f32", "mul_mat_q8_0_f32", "mul_mat_q4_k_swiglu_f32", "mul_mat_id_q4_k_f32", "mul_mat_id_q5_k_f32", "mul_mat_id_q6_k_f32"), "quantized_mul_mat_numpy_logical", {"atol": 0.12, "rtol": 0.04}, _quantized_matmul_arrays),
    LogicalOracleSpec(("quantize_q8_1_f32",), "quantize_q8_1_numpy", {"atol": 0.0, "rtol": 0.0}, _quantize_q8_1_arrays),
    LogicalOracleSpec(("rms_norm_mul_f32",), "rms_norm_mul_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _rms_norm_mul_arrays),
    LogicalOracleSpec(("add_rms_norm_mul_f32",), "add_rms_norm_mul_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _add_rms_norm_mul_arrays),
    LogicalOracleSpec(("rms_norm_mul_quantize_q8_1_f32",), "rms_norm_mul_quantize_q8_1_numpy", {"atol": 0.0, "rtol": 0.0}, _rms_norm_mul_quantize_arrays),
    LogicalOracleSpec(("rope_f32", "rope_neox_f32", "rope_scale_f32", "rope_set_rows_f32"), "rope_numpy_structural_placeholder", {"atol": 1e-5, "rtol": 1e-5}, _rope_arrays),
    LogicalOracleSpec(("set_rows_f32", "cont_set_rows_f32"), "set_rows_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _set_rows_arrays),
    LogicalOracleSpec(("soft_max_f32",), "soft_max_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _softmax_arrays),
    LogicalOracleSpec(("softmax_kqv_f32_f16",), "softmax_kqv_f32_f16_numpy_logical", {"atol": 0.08, "rtol": 0.02}, _softmax_kqv_arrays),
    LogicalOracleSpec(("sum_rows_f32",), "sum_rows_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _sum_rows_arrays),
    LogicalOracleSpec(("swiglu_f32",), "swiglu_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _swiglu_arrays),
)


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
    spec = ORACLE_SPECS_BY_FAMILY.get(candidate.family)
    if spec is None:
        return None, {"status": "unsupported_golden", "message": f"no generated check.case for family {candidate.family}"}
    return spec.write_workbench(candidate, linked_source, workbench_path, fixture_dir)


def _source_plus_case(linked_source: Path, workbench_path: Path, suffix: str) -> None:
    text = linked_source.read_text(encoding="utf-8")
    workbench_path.write_text(text.rstrip() + "\n\n" + suffix.lstrip(), encoding="utf-8")


def _rel_fixture(workbench_path: Path, fixture_dir: Path, name: str) -> str:
    return str((fixture_dir / name).relative_to(workbench_path.parent))


def _write_mul_mat_q4_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    if "split_k_reduce2" in candidate.root_symbol:
        return _logical_workbench(candidate, linked_source, workbench_path, fixture_dir)
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


def _write_pointwise_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    lines = [
        _read_f32(workbench_path, fixture_dir, "src0", elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", elems),
    ]
    if candidate.family in {"add_f32", "mul_f32", "div_f32"}:
        lines.insert(1, _read_f32(workbench_path, fixture_dir, "src1", elems))
        call_args = "%src0, %src1, %dst"
        call_types = f"tensor<{elems}xf32>, tensor<{elems}xf32>, tensor<{elems}xf32>"
    elif candidate.family == "scale_f32":
        lines.insert(0, "  %scale = check.literal value(0.625) : f32")
        lines.insert(1, "  %bias = check.literal value(-0.125) : f32")
        call_args = "%scale, %bias, %src0, %dst"
        call_types = f"f32, f32, tensor<{elems}xf32>, tensor<{elems}xf32>"
    elif candidate.family == "clamp_f32":
        lines.insert(0, "  %min = check.literal value(-0.45) : f32")
        lines.insert(1, "  %max = check.literal value(0.55) : f32")
        call_args = "%min, %max, %src0, %dst"
        call_types = f"f32, f32, tensor<{elems}xf32>, tensor<{elems}xf32>"
    else:
        return _logical_workbench(candidate, linked_source, workbench_path, fixture_dir)
    lines.extend(
        [
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_swiglu_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    src0_elems = elems if candidate.root_symbol.endswith("_split") else elems * 2
    case_name, bench_name = _case_names(candidate)
    lines = [
        _read_f32(workbench_path, fixture_dir, "src0", src0_elems),
    ]
    if candidate.root_symbol.endswith("_split"):
        lines.append(_read_f32(workbench_path, fixture_dir, "src1", elems))
        call_args = "%src0, %src1, %dst"
        call_types = f"tensor<{src0_elems}xf32>, tensor<{elems}xf32>, tensor<{elems}xf32>"
    else:
        call_args = "%src0, %dst"
        call_types = f"tensor<{src0_elems}xf32>, tensor<{elems}xf32>"
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_sum_rows_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", elems),
            f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}\") : tensor<{nrows}xf32>",
            _read_f32(workbench_path, fixture_dir, "expected", nrows),
            f"  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{elems}xf32>, tensor<{nrows}xf32>)",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{nrows}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_argsort_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", elems),
            _read_i32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_i32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}(%src0, %dst) : (tensor<{elems}xf32>, tensor<{elems}xi32>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{elems}xi32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_get_rows_f32_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    if candidate.family != "get_rows_f32":
        return _logical_workbench(candidate, linked_source, workbench_path, fixture_dir)
    ncols, nrows, elems = _dims(candidate)
    src0_nrows = int(candidate.values.get("shape.get_rows.src0_nrows") or candidate.values.get("get_rows.src0_nrows") or nrows)
    case_name, bench_name = _case_names(candidate)
    body = "\n".join(
        [
            _read_f32(workbench_path, fixture_dir, "src0", src0_nrows * ncols),
            _read_i32(workbench_path, fixture_dir, "indices", nrows).replace("%indices", "%idx"),
            _read_f32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}(%src0, %idx, %dst) : (tensor<{src0_nrows * ncols}xf32>, tensor<{nrows}xi32>, tensor<{elems}xf32>)",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, body)


def _write_softmax_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    case_name, bench_name = _case_names(candidate)
    lines = [
        "  %scale = check.literal value(0.75) : f32",
        _read_f32(workbench_path, fixture_dir, "src0", elems),
    ]
    if "mask" in candidate.root_symbol:
        lines.append(_read_f32(workbench_path, fixture_dir, "mask", elems))
        call_args = "%scale, %src0, %mask, %dst"
        call_types = f"f32, tensor<{elems}xf32>, tensor<{elems}xf32>, tensor<{elems}xf32>"
    else:
        call_args = "%scale, %src0, %dst"
        call_types = f"f32, tensor<{elems}xf32>, tensor<{elems}xf32>"
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", elems),
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_set_rows_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    idx_elems = nrows * 2
    f16_dst = "f16" in candidate.root_symbol
    dst_type = "xi16" if f16_dst else "xf32"
    reader = _read_i16 if f16_dst else _read_f32
    expect = "equal" if f16_dst else "close"
    case_name, bench_name = _case_names(candidate)
    lines = [
        _read_f32(workbench_path, fixture_dir, "src0", elems),
        _read_i32(workbench_path, fixture_dir, "indices", idx_elems).replace("%indices", "%idx"),
        reader(workbench_path, fixture_dir, "dst_init", elems).replace("%dst_init", "%dst"),
        reader(workbench_path, fixture_dir, "expected", elems),
        f"  func.call {candidate.root_symbol}(%src0, %idx, %dst) : (tensor<{elems}xf32>, tensor<{idx_elems}xi32>, tensor<{elems}{dst_type}>)",
    ]
    if expect == "equal":
        lines.append(f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{elems}{dst_type}>")
    else:
        lines.append(f"  check.expect.close actual(%dst) expected(%expected) atol(0.00001) rtol(0.00001) nan(same) : tensor<{elems}{dst_type}>")
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_quantize_q8_1_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str, dict[str, Any]]:
    ncols, nrows, elems = _dims(candidate)
    blocks = (ncols + 31) // 32
    x4 = "x4" in candidate.root_symbol
    expected_elems = nrows * ((blocks + 3) // 4) * 144 if x4 else q8_1_bytes(ncols, nrows)
    case_name, bench_name = _case_names(candidate)
    lines = [_read_f32(workbench_path, fixture_dir, "src0", elems).replace("%src0", "%src")]
    if x4 and "vk_clone" in candidate.root_symbol:
        num_blocks = ((ncols + 127) // 128) * 4
        lines = [
            f"  %ne = check.literal value({elems}) : index",
            f"  %num_blocks = check.literal value({num_blocks}) : index",
            *lines,
            f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}\") : tensor<{expected_elems}xi8>",
            f"  %expected = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}\") : tensor<{expected_elems}xi8>",
            f"  func.call {candidate.root_symbol}(%ne, %num_blocks, %src, %dst) : (index, index, tensor<{elems}xf32>, tensor<{expected_elems}xi8>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{expected_elems}xi8>",
        ]
    else:
        lines = [
            f"  %ne00 = check.literal value({ncols}) : index",
            f"  %s01 = check.literal value({ncols}) : index",
            f"  %s02 = check.literal value({ncols * nrows}) : index",
            f"  %s03 = check.literal value({ncols * nrows}) : index",
            f"  %ne0 = check.literal value({ncols}) : index",
            f"  %ne1 = check.literal value({nrows}) : index",
            "  %ne2 = check.literal value(1) : index",
            *lines,
            f"  %dst = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}\") : tensor<{expected_elems}xi8>",
            f"  %expected = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, "expected.npy")}\") : tensor<{expected_elems}xi8>",
            f"  func.call {candidate.root_symbol}(%ne00, %s01, %s02, %s03, %ne0, %ne1, %ne2, %src, %dst) : (index, index, index, index, index, index, index, tensor<{elems}xf32>, tensor<{expected_elems}xi8>)",
            f"  check.expect.equal actual(%dst) expected(%expected) : tensor<{expected_elems}xi8>",
        ]
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_mul_mat_q8_0_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str | None, dict[str, Any]]:
    if "q8_1" in candidate.root_symbol:
        return None, {"status": "unsupported_workbench", "message": "q8_0 matmul q8_1 RHS ABI packer is not implemented yet"}
    k, rows, cols = _matmul_dims(candidate)
    src0_elems = q8_0_bytes(k, rows)
    src1_elems = k * cols
    dst_elems = rows * cols
    case_name, bench_name = _case_names(candidate)
    lines = [
        f"  %src0 = check.file.read.npy path(\"{_rel_fixture(workbench_path, fixture_dir, "src0.npy")}\") : tensor<{src0_elems}xi8>",
        _read_f32(workbench_path, fixture_dir, "src1", src1_elems),
        _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
        _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
    ]
    if candidate.root_symbol == "@hrx2_mul_mat_q8_0_f32":
        lines = [
            f"  %k = check.literal value({k}) : index",
            f"  %rows = check.literal value({rows}) : index",
            f"  %cols = check.literal value({cols}) : index",
            *lines,
            f"  func.call {candidate.root_symbol}(%k, %rows, %cols, %src0, %src1, %dst) : (index, index, index, tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)",
        ]
    else:
        lines.append(
            f"  func.call {candidate.root_symbol}(%src0, %src1, %dst) : (tensor<{src0_elems}xi8>, tensor<{src1_elems}xf32>, tensor<{dst_elems}xf32>)"
        )
    lines.append(f"  check.expect.close actual(%dst) expected(%expected) atol(0.08) rtol(0.02) nan(same) : tensor<{dst_elems}xf32>")
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


def _write_rope_workbench(candidate: Candidate, linked_source: Path, workbench_path: Path, fixture_dir: Path) -> tuple[str | None, dict[str, Any]]:
    if candidate.family == "rope_set_rows_f32":
        return None, {"status": "unsupported_workbench", "message": "ROPE set_rows f16 ABI writer is not implemented yet"}
    ncols = int(candidate.values.get("shape.rope.ncols") or candidate.shape.get("ncols", 1))
    n_dims = int(candidate.values.get("shape.rope.n_dims") or candidate.shape.get("n_dims", ncols))
    nheads = int(candidate.values.get("shape.rope.nheads") or candidate.shape.get("rows", 1))
    ntokens = int(candidate.values.get("shape.rope.ntokens") or candidate.shape.get("cols", 1))
    src_token_stride = int(candidate.values.get("shape.rope.src0_token_stride") or (ncols * nheads))
    dst_token_stride = int(candidate.values.get("shape.rope.dst_token_stride") or (ncols * nheads))
    pos_token_stride = int(candidate.values.get("shape.rope.pos_token_stride") or 1)
    src_elems = src_token_stride * ntokens
    dst_elems = dst_token_stride * ntokens
    pos_elems = pos_token_stride * ntokens
    freq_elems = max(n_dims // 2, 1)
    has_freq = "freq" in candidate.root_symbol
    scale_output = "scale" in candidate.root_symbol
    case_name, bench_name = _case_names(candidate)
    lines = [
        "  %theta_scale = check.literal value(0.75) : f32",
        "  %freq_scale = check.literal value(1.1) : f32",
        "  %attn_factor = check.literal value(0.9) : f32",
    ]
    if scale_output:
        lines.append("  %output_scale = check.literal value(0.5) : f32")
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "src0", src_elems),
            _read_i32(workbench_path, fixture_dir, "positions", pos_elems).replace("%positions", "%pos"),
        ]
    )
    if has_freq:
        lines.append(_read_f32(workbench_path, fixture_dir, "freq", freq_elems))
    lines.extend(
        [
            _read_f32(workbench_path, fixture_dir, "dst_init", dst_elems).replace("%dst_init", "%dst"),
            _read_f32(workbench_path, fixture_dir, "expected", dst_elems),
        ]
    )
    if scale_output:
        call_args = "%theta_scale, %freq_scale, %attn_factor, %output_scale, %src0, %pos, %freq, %dst"
        call_types = f"f32, f32, f32, f32, tensor<{src_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{freq_elems}xf32>, tensor<{dst_elems}xf32>"
    elif has_freq:
        call_args = "%theta_scale, %freq_scale, %attn_factor, %src0, %pos, %freq, %dst"
        call_types = f"f32, f32, f32, tensor<{src_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{freq_elems}xf32>, tensor<{dst_elems}xf32>"
    else:
        call_args = "%theta_scale, %freq_scale, %attn_factor, %src0, %pos, %dst"
        call_types = f"f32, f32, f32, tensor<{src_elems}xf32>, tensor<{pos_elems}xi32>, tensor<{dst_elems}xf32>"
    lines.extend(
        [
            f"  func.call {candidate.root_symbol}({call_args}) : ({call_types})",
            f"  check.expect.close actual(%dst) expected(%expected) atol(0.0005) rtol(0.0005) nan(same) : tensor<{dst_elems}xf32>",
        ]
    )
    return _emit_case(linked_source, workbench_path, case_name, bench_name, "\n".join(lines))


ORACLE_SPECS: tuple[OracleSpec, ...] = (
    OracleSpec(
        family_ids=("mul_mat_q4_k_f32",),
        generate=_mul_mat_q4_k_f32,
        write_workbench=_write_mul_mat_q4_workbench,
    ),
    OracleSpec(
        family_ids=("rms_norm_f32",),
        generate=_rms_norm_f32,
        write_workbench=_write_rms_norm_workbench,
    ),
    OracleSpec(
        family_ids=("copy_f32_f16",),
        generate=_copy_f32_f16,
        write_workbench=_write_copy_workbench,
    ),
    OracleSpec(
        family_ids=("cont_f32",),
        generate=_cont_f32,
        write_workbench=_write_cont_workbench,
    ),
    *(
        OracleSpec(
            family_ids=spec.family_ids,
            generate=_logical_generate(spec),
            write_workbench=_logical_workbench,
        )
        for spec in LOGICAL_ORACLE_SPECS
    ),
    OracleSpec(
        family_ids=("add_f32", "mul_f32", "div_f32"),
        generate=_logical_generate(LogicalOracleSpec(("add_f32", "mul_f32", "div_f32"), "pointwise_binary_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, lambda np, candidate, seed: {"add_f32": _binary_arrays(lambda lhs, rhs: lhs + rhs), "mul_f32": _binary_arrays(lambda lhs, rhs: lhs * rhs), "div_f32": _div_arrays}[candidate.family](np, candidate, seed), exact_kernel_abi=True)),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("scale_f32",),
        generate=_logical_generate(LogicalOracleSpec(("scale_f32",), "scale_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _scale_arrays, exact_kernel_abi=True)),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("clamp_f32",),
        generate=_logical_generate(LogicalOracleSpec(("clamp_f32",), "clamp_f32_numpy", {"atol": 1e-6, "rtol": 1e-6}, _clamp_arrays, exact_kernel_abi=True)),
        write_workbench=_write_pointwise_workbench,
    ),
    OracleSpec(
        family_ids=("swiglu_f32",),
        generate=_logical_generate(LogicalOracleSpec(("swiglu_f32",), "swiglu_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _swiglu_arrays, exact_kernel_abi=True)),
        write_workbench=_write_swiglu_workbench,
    ),
    OracleSpec(
        family_ids=("sum_rows_f32",),
        generate=_logical_generate(LogicalOracleSpec(("sum_rows_f32",), "sum_rows_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _sum_rows_arrays, exact_kernel_abi=True)),
        write_workbench=_write_sum_rows_workbench,
    ),
    OracleSpec(
        family_ids=("argsort_f32_i32",),
        generate=_logical_generate(LogicalOracleSpec(("argsort_f32_i32",), "argsort_f32_i32_numpy_desc", {"atol": 0.0, "rtol": 0.0}, _argsort_arrays, exact_kernel_abi=True)),
        write_workbench=_write_argsort_workbench,
    ),
    OracleSpec(
        family_ids=("get_rows_f32",),
        generate=_logical_generate(LogicalOracleSpec(("get_rows_f32",), "get_rows_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _get_rows_arrays, exact_kernel_abi=True)),
        write_workbench=_write_get_rows_f32_workbench,
    ),
    OracleSpec(
        family_ids=("soft_max_f32",),
        generate=_logical_generate(LogicalOracleSpec(("soft_max_f32",), "soft_max_f32_numpy", {"atol": 1e-4, "rtol": 1e-4}, _softmax_arrays, exact_kernel_abi=True)),
        write_workbench=_write_softmax_workbench,
    ),
    OracleSpec(
        family_ids=("set_rows_f32", "cont_set_rows_f32"),
        generate=_logical_generate(LogicalOracleSpec(("set_rows_f32", "cont_set_rows_f32"), "set_rows_f32_numpy", {"atol": 1e-5, "rtol": 1e-5}, _set_rows_arrays, exact_kernel_abi=True)),
        write_workbench=_write_set_rows_workbench,
    ),
    OracleSpec(
        family_ids=("quantize_q8_1_f32",),
        generate=_logical_generate(LogicalOracleSpec(("quantize_q8_1_f32",), "quantize_q8_1_numpy", {"atol": 0.0, "rtol": 0.0}, _quantize_q8_1_arrays, exact_kernel_abi=True)),
        write_workbench=_write_quantize_q8_1_workbench,
    ),
    OracleSpec(
        family_ids=("mul_mat_q8_0_f32",),
        generate=_logical_generate(LogicalOracleSpec(("mul_mat_q8_0_f32",), "mul_mat_q8_0_f32_numpy_dequant_matmul", {"atol": 0.08, "rtol": 0.02}, _mul_mat_q8_0_arrays, exact_kernel_abi=True)),
        write_workbench=_write_mul_mat_q8_0_workbench,
    ),
    OracleSpec(
        family_ids=("rope_f32", "rope_neox_f32", "rope_scale_f32"),
        generate=_logical_generate(LogicalOracleSpec(("rope_f32", "rope_neox_f32", "rope_scale_f32"), "rope_f32_numpy", {"atol": 5e-4, "rtol": 5e-4}, _rope_arrays, exact_kernel_abi=True)),
        write_workbench=_write_rope_workbench,
    ),
)


ORACLE_SPECS_BY_FAMILY: dict[str, OracleSpec] = {
    family_id: spec
    for spec in ORACLE_SPECS
    for family_id in spec.family_ids
}
