#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
LLAMA_ROOT = WORKSPACE / "sources" / "llama.cpp"
HRX2_ROOT = LLAMA_ROOT / "ggml" / "src" / "ggml-hrx2"
DEFAULT_CATALOG = HRX2_ROOT / "catalog.json"
DEFAULT_SOURCE_ROOT = HRX2_ROOT
DEFAULT_ARTIFACT_ROOT = WORKSPACE / "build" / "llama-hrx2" / "ggml" / "src" / "ggml-hrx2" / "generated" / "catalog"
DEFAULT_LLAMA_BUILD = WORKSPACE / "build" / "llama-hrx2"
DEFAULT_HRX_INSTALL = WORKSPACE / "build" / "hrx-install"
DEFAULT_ROCM = WORKSPACE / "rocm"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def is_pow2(value):
    return isinstance(value, int) and value > 0 and (value & (value - 1)) == 0


def route_matches_shape(route, shape, target_key=None):
    if target_key and route.get("target_key") and route["target_key"] != target_key:
        return False
    if route.get("op") != shape.get("op"):
        return False

    domain = route.get("shape_domain", {})
    if shape.get("op") == "RMS_NORM":
        ncols = int(shape["ncols"])
        nrows = int(shape["nrows"])
        return (
            ncols >= int(domain.get("ncols_min", 0))
            and ncols <= int(domain.get("ncols_max", 2**32 - 1))
            and nrows >= int(domain.get("nrows_min", 0))
            and nrows <= int(domain.get("nrows_max", 2**32 - 1))
        )

    if shape.get("op") == "MUL_MAT":
        k = int(shape["k"])
        rows = int(shape["rows"])
        cols = int(shape["cols"])
        if not (
            k >= int(domain.get("k_min", 0))
            and k <= int(domain.get("k_max", 2**32 - 1))
            and rows >= int(domain.get("rows_min", 0))
            and rows <= int(domain.get("rows_max", 2**32 - 1))
            and cols >= int(domain.get("cols_min", 0))
            and cols <= int(domain.get("cols_max", 2**32 - 1))
        ):
            return False
        guards = route.get("shape_guards", {})
        if "k_pow2" in guards and bool(guards["k_pow2"]) != is_pow2(k):
            return False
        if "all_pot" in guards and bool(guards["all_pot"]) != (is_pow2(k) and is_pow2(rows) and is_pow2(cols)):
            return False
        return True

    return False


def resolve_config_bindings(route, shape):
    bindings = []
    specialization = route.get("specialization") or {}
    mode = specialization.get("mode", "")
    if mode and mode != "jit_config":
        raise ValueError(f"unsupported specialization mode {mode!r} on {route.get('id')}")
    for spec in specialization.get("bindings", []):
        key = spec["key"]
        if "value" in spec:
            value = str(spec["value"])
        else:
            source = spec.get("source")
            if source == "shape.k":
                value = str(shape["k"])
            elif source == "shape.rows":
                value = str(shape["rows"])
            elif source == "shape.cols":
                value = str(shape["cols"])
            else:
                raise ValueError(f"unsupported config source {source!r} on {route.get('id')}")
        bindings.append({"key": key, "value": value})
    return bindings


def provider_cache_key(route, shape, target_key, bindings):
    cache_key = f"{route['id']}|target={target_key or ''}"
    if (route.get("specialization") or {}).get("mode") == "jit_config":
        cache_key += f"|k={shape.get('k')}|rows={shape.get('rows')}|cols={shape.get('cols')}"
        for binding in bindings:
            cache_key += f"|{binding['key']}={binding['value']}"
    return cache_key


def test_backend_filter(shape):
    if shape["op"] == "MUL_MAT":
        return f"type_a=q8_0,type_b=f32,m={shape['rows']},n={shape['cols']},k={shape['k']}"
    if shape["op"] == "RMS_NORM":
        ne = shape.get("ne", [shape["ncols"], shape.get("ne1", shape["nrows"]), shape.get("ne2", 1), shape.get("ne3", 1)])
        escaped_ne = r"\[" + ",".join(str(int(v)) for v in ne) + r"\]"
        eps = float(shape.get("eps", 0.000001))
        return f"type=f32,ne={escaped_ne},v=0,eps={eps:.6f},inplace=0"
    raise ValueError(f"unsupported op {shape.get('op')}")


def shape_identity(shape):
    if shape["op"] == "MUL_MAT":
        return f"mul_mat_q8_0_f32_k{shape['k']}_r{shape['rows']}_c{shape['cols']}"
    if shape["op"] == "RMS_NORM":
        return f"rms_norm_f32_ncols{shape['ncols']}_nrows{shape['nrows']}"
    return re.sub(r"[^0-9A-Za-z_]+", "_", shape.get("op", "shape")).lower()


def env_for_tools(extra=None):
    env = os.environ.copy()
    ld_parts = [
        str(DEFAULT_HRX_INSTALL / "lib"),
        str(DEFAULT_HRX_INSTALL / "lib64"),
        str(DEFAULT_ROCM / "lib"),
        str(DEFAULT_ROCM / "lib" / "rocm_sysdeps" / "lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        ld_parts.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env.setdefault("ROCM_PATH", str(DEFAULT_ROCM))
    env.setdefault("GGML_HRX_ROCM_PATH", str(DEFAULT_ROCM))
    if extra:
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env
