#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import random
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hrx2_pipeline_lib import DEFAULT_HRX_INSTALL, WORKSPACE, env_for_tools, read_jsonl, write_json


DEFAULT_OUT_ROOT = WORKSPACE / "cache" / "hrx2" / "rms_norm_tune"
DEFAULT_SHAPES = "4096x1,4096x32,512x32,1024x1,3584x1,8192x1,8192x32"


@dataclass(frozen=True)
class Shape:
    ncols: int
    nrows: int

    @property
    def id(self):
        return f"n{self.ncols}_r{self.nrows}"

    @property
    def element_count(self):
        return self.ncols * self.nrows


@dataclass(frozen=True)
class Candidate:
    shape: Shape
    operation: str
    family: str
    workgroup_size: int
    vector_width: int
    cache_policy: str

    @property
    def id(self):
        return (
            f"{self.operation}_{self.shape.id}_{self.family}_wg{self.workgroup_size}_"
            f"vw{self.vector_width}_{self.cache_policy}"
        )

    @property
    def benchmark(self):
        return f"bench_{self.id}"

    @property
    def source_name(self):
        return f"{self.id}.loom"


def parse_args():
    parser = argparse.ArgumentParser(description="Tune exact-shape HRX2 RMS_NORM Loom variants.")
    parser.add_argument("--run-id", default="gfx1100-rms-norm-done")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--iree-benchmark-loom", default=str(DEFAULT_HRX_INSTALL / "bin" / "iree-benchmark-loom"))
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help="Comma-separated ncolsxnrows list.")
    parser.add_argument("--workgroup-sizes", default="64,128,256,512")
    parser.add_argument("--vector-widths", default="1,2,4")
    parser.add_argument("--cache-policies", default="default,non_temporal")
    parser.add_argument(
        "--include-vector-tail",
        action="store_true",
        help="Include vectorized kernels with scalar cleanup for ncols not divisible by vector width.",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--eps", type=float, default=0.000001)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument(
        "--include-copy-floor",
        action="store_true",
        help="Also benchmark one-pass read/write copy kernels as a dispatch+traffic floor.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_shapes(text):
    shapes = []
    for part in split_csv(text):
        if "x" not in part:
            raise ValueError(f"shape {part!r} must be ncolsxnrows")
        ncols_text, nrows_text = part.lower().split("x", 1)
        shapes.append(Shape(int(ncols_text), int(nrows_text)))
    return shapes


def npy_header(shape):
    shape_text = "(" + ", ".join(str(dim) for dim in shape)
    if len(shape) == 1:
        shape_text += ","
    shape_text += ")"
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}"
    prefix_len = 10
    pad_len = 16 - ((prefix_len + len(header) + 1) % 16)
    header = header + (" " * pad_len) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("ascii")


def write_npy_f32(path, shape, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(npy_header(shape))
        f.write(struct.pack("<" + "f" * len(values), *values))


def fixture_values(shape, seed):
    rng = random.Random(seed + shape.ncols * 131 + shape.nrows * 17)
    values = []
    for i in range(shape.element_count):
        # Keep values representative but bounded; avoid all-constant benchmark data.
        wave = math.sin((i % shape.ncols) * 0.017) * 0.75
        row_bias = ((i // shape.ncols) % 13 - 6) * 0.03125
        jitter = rng.uniform(-0.25, 0.25)
        values.append(wave + row_bias + jitter)
    return values


def rms_norm_expected(shape, src, eps):
    expected = [0.0] * len(src)
    for row in range(shape.nrows):
        base = row * shape.ncols
        row_values = src[base:base + shape.ncols]
        mean = sum(value * value for value in row_values) / shape.ncols
        scale = 1.0 / math.sqrt(mean + eps)
        for col, value in enumerate(row_values):
            expected[base + col] = value * scale
    return expected


def ensure_fixtures(run_dir, shapes, eps, seed):
    fixture_dir = run_dir / "fixtures"
    manifest = []
    for shape in shapes:
        src = fixture_values(shape, seed)
        expected = rms_norm_expected(shape, src, eps)
        src_path = fixture_dir / f"rms_src_{shape.id}.npy"
        expected_path = fixture_dir / f"rms_expected_{shape.id}.npy"
        write_npy_f32(src_path, (shape.nrows, shape.ncols), src)
        write_npy_f32(expected_path, (shape.nrows, shape.ncols), expected)
        manifest.append({
            "shape": {"ncols": shape.ncols, "nrows": shape.nrows},
            "src": str(src_path),
            "expected": str(expected_path),
            "eps": eps,
        })
    write_json(run_dir / "fixtures_manifest.json", manifest)


def cache_attrs(policy):
    if policy == "default":
        return ""
    if policy == "non_temporal":
        return " {cache_scope = device, cache_temporal = non_temporal}"
    raise ValueError(f"unsupported cache policy {policy!r}")


def generate_scalar_kernel(candidate, eps):
    shape = candidate.shape
    wg = candidate.workgroup_size
    return f"""kernel.def @rms_norm_candidate() {{
  %unit = index.constant 1 : index
  %nrows = index.constant {shape.nrows} : index
  %workgroup_size = index.constant {wg} : index
  kernel.launch.config workgroups(%nrows, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%eps: f32, %src: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %ncols = index.constant {shape.ncols} : index
  %workgroup_size = index.constant {wg} : index
  %row0 = kernel.workgroup.id<x> : index
  %row = index.assume %row0 [range(%row0, 0, {shape.nrows - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {wg - 1})] : index
  %zero_f32 = scalar.constant 0.0 : f32

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %row_base = index.mul %row, %ncols : index

  %sum = scf.for %col = [%lane to %ncols step %workgroup_size](%acc = %zero_f32 : f32) -> (f32) {{
    %linear0 = index.add %row_base, %col : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - 1})] : index
    %value = view.load %src_view[%linear] : view<{shape.element_count}xf32, #dense> -> f32
    %square = scalar.mulf<reassoc|nnan|ninf|nsz> %value, %value : f32
    %next = scalar.addf<reassoc|nnan|ninf|nsz> %acc, %square : f32
    scf.yield %next : f32
  }}

  %row_sum = kernel.workgroup.reduce<addf> %sum : f32
  %ncols_i32 = index.cast %ncols : index to i32
  %ncols_f32 = scalar.sitofp %ncols_i32 : i32 to f32
  %mean = scalar.divf<nnan|ninf|nsz|arcp> %row_sum, %ncols_f32 : f32
  %mean_eps = scalar.addf<nnan|ninf|nsz> %mean, %eps : f32
  %scale = scalar.rsqrtf<nnan|ninf|nsz|afn> %mean_eps : f32

  scf.for %col = [%lane to %ncols step %workgroup_size] {{
    %linear0 = index.add %row_base, %col : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - 1})] : index
    %value = view.load %src_view[%linear] : view<{shape.element_count}xf32, #dense> -> f32
    %result = scalar.mulf<nnan|ninf|nsz> %value, %scale : f32
    view.store %result, %dst_view[%linear] : f32, view<{shape.element_count}xf32, #dense>
  }}
  kernel.return
}}
"""


def generate_vector_kernel(candidate, eps):
    shape = candidate.shape
    wg = candidate.workgroup_size
    vw = candidate.vector_width
    logical_cols = shape.ncols // vw
    attrs = cache_attrs(candidate.cache_policy)
    return f"""kernel.def @rms_norm_candidate() {{
  %unit = index.constant 1 : index
  %nrows = index.constant {shape.nrows} : index
  %workgroup_size = index.constant {wg} : index
  kernel.launch.config workgroups(%nrows, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%eps: f32, %src: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %ncols = index.constant {shape.ncols} : index
  %logical_cols = index.constant {logical_cols} : index
  %vector_width = index.constant {vw} : index
  %workgroup_size = index.constant {wg} : index
  %row0 = kernel.workgroup.id<x> : index
  %row = index.assume %row0 [range(%row0, 0, {shape.nrows - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {wg - 1})] : index
  %zero_f32 = scalar.constant 0.0 : f32

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %row_base = index.mul %row, %ncols : index

  %sum = scf.for %vcol = [%lane to %logical_cols step %workgroup_size](%acc = %zero_f32 : f32) -> (f32) {{
    %col0 = index.mul %vcol, %vector_width : index
    %linear0 = index.add %row_base, %col0 : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - vw}), mul(%linear0, {vw})] : index
    %values = vector.load %src_view[%linear]{attrs} : view<{shape.element_count}xf32, #dense> -> vector<{vw}xf32>
    %squares = vector.mulf<reassoc|nnan|ninf|nsz> %values, %values : vector<{vw}xf32>
    %next = vector.reduce<addf, reassoc|nnan|ninf|nsz> %squares, %acc : vector<{vw}xf32>, f32
    scf.yield %next : f32
  }}

  %row_sum = kernel.workgroup.reduce<addf> %sum : f32
  %ncols_i32 = index.cast %ncols : index to i32
  %ncols_f32 = scalar.sitofp %ncols_i32 : i32 to f32
  %mean = scalar.divf<nnan|ninf|nsz|arcp> %row_sum, %ncols_f32 : f32
  %mean_eps = scalar.addf<nnan|ninf|nsz> %mean, %eps : f32
  %scale = scalar.rsqrtf<nnan|ninf|nsz|afn> %mean_eps : f32
  %scale_vector = vector.splat %scale : vector<{vw}xf32>

  scf.for %vcol = [%lane to %logical_cols step %workgroup_size] {{
    %col0 = index.mul %vcol, %vector_width : index
    %linear0 = index.add %row_base, %col0 : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - vw}), mul(%linear0, {vw})] : index
    %values = vector.load %src_view[%linear]{attrs} : view<{shape.element_count}xf32, #dense> -> vector<{vw}xf32>
    %result = vector.mulf<nnan|ninf|nsz> %values, %scale_vector : vector<{vw}xf32>
    vector.store %result, %dst_view[%linear] : vector<{vw}xf32>, view<{shape.element_count}xf32, #dense>
  }}
  kernel.return
}}
"""


def generate_vector_tail_kernel(candidate, eps):
    shape = candidate.shape
    wg = candidate.workgroup_size
    vw = candidate.vector_width
    logical_cols = shape.ncols // vw
    tail_start = logical_cols * vw
    attrs = cache_attrs(candidate.cache_policy)
    return f"""kernel.def @rms_norm_candidate() {{
  %unit = index.constant 1 : index
  %nrows = index.constant {shape.nrows} : index
  %workgroup_size = index.constant {wg} : index
  kernel.launch.config workgroups(%nrows, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%eps: f32, %src: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %ncols = index.constant {shape.ncols} : index
  %logical_cols = index.constant {logical_cols} : index
  %tail_start = index.constant {tail_start} : index
  %vector_width = index.constant {vw} : index
  %workgroup_size = index.constant {wg} : index
  %row0 = kernel.workgroup.id<x> : index
  %row = index.assume %row0 [range(%row0, 0, {shape.nrows - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {wg - 1})] : index
  %zero_f32 = scalar.constant 0.0 : f32

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %row_base = index.mul %row, %ncols : index

  %sum_vec = scf.for %vcol = [%lane to %logical_cols step %workgroup_size](%acc = %zero_f32 : f32) -> (f32) {{
    %col0 = index.mul %vcol, %vector_width : index
    %linear0 = index.add %row_base, %col0 : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - vw}), mul(%linear0, {vw})] : index
    %values = vector.load %src_view[%linear]{attrs} : view<{shape.element_count}xf32, #dense> -> vector<{vw}xf32>
    %squares = vector.mulf<reassoc|nnan|ninf|nsz> %values, %values : vector<{vw}xf32>
    %next = vector.reduce<addf, reassoc|nnan|ninf|nsz> %squares, %acc : vector<{vw}xf32>, f32
    scf.yield %next : f32
  }}

  %tail_lane0 = index.add %tail_start, %lane : index
  %sum = scf.for %col = [%tail_lane0 to %ncols step %workgroup_size](%acc = %sum_vec : f32) -> (f32) {{
    %linear0 = index.add %row_base, %col : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - 1})] : index
    %value = view.load %src_view[%linear] : view<{shape.element_count}xf32, #dense> -> f32
    %square = scalar.mulf<reassoc|nnan|ninf|nsz> %value, %value : f32
    %next = scalar.addf<reassoc|nnan|ninf|nsz> %acc, %square : f32
    scf.yield %next : f32
  }}

  %row_sum = kernel.workgroup.reduce<addf> %sum : f32
  %ncols_i32 = index.cast %ncols : index to i32
  %ncols_f32 = scalar.sitofp %ncols_i32 : i32 to f32
  %mean = scalar.divf<nnan|ninf|nsz|arcp> %row_sum, %ncols_f32 : f32
  %mean_eps = scalar.addf<nnan|ninf|nsz> %mean, %eps : f32
  %scale = scalar.rsqrtf<nnan|ninf|nsz|afn> %mean_eps : f32
  %scale_vector = vector.splat %scale : vector<{vw}xf32>

  scf.for %vcol = [%lane to %logical_cols step %workgroup_size] {{
    %col0 = index.mul %vcol, %vector_width : index
    %linear0 = index.add %row_base, %col0 : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - vw}), mul(%linear0, {vw})] : index
    %values = vector.load %src_view[%linear]{attrs} : view<{shape.element_count}xf32, #dense> -> vector<{vw}xf32>
    %result = vector.mulf<nnan|ninf|nsz> %values, %scale_vector : vector<{vw}xf32>
    vector.store %result, %dst_view[%linear] : vector<{vw}xf32>, view<{shape.element_count}xf32, #dense>
  }}

  scf.for %col = [%tail_lane0 to %ncols step %workgroup_size] {{
    %linear0 = index.add %row_base, %col : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - 1})] : index
    %value = view.load %src_view[%linear] : view<{shape.element_count}xf32, #dense> -> f32
    %result = scalar.mulf<nnan|ninf|nsz> %value, %scale : f32
    view.store %result, %dst_view[%linear] : f32, view<{shape.element_count}xf32, #dense>
  }}
  kernel.return
}}
"""


def generate_copy_scalar_kernel(candidate, eps):
    shape = candidate.shape
    wg = candidate.workgroup_size
    return f"""kernel.def @rms_norm_candidate() {{
  %unit = index.constant 1 : index
  %nrows = index.constant {shape.nrows} : index
  %workgroup_size = index.constant {wg} : index
  kernel.launch.config workgroups(%nrows, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%eps: f32, %src: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %ncols = index.constant {shape.ncols} : index
  %workgroup_size = index.constant {wg} : index
  %row0 = kernel.workgroup.id<x> : index
  %row = index.assume %row0 [range(%row0, 0, {shape.nrows - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {wg - 1})] : index

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %row_base = index.mul %row, %ncols : index

  scf.for %col = [%lane to %ncols step %workgroup_size] {{
    %linear0 = index.add %row_base, %col : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - 1})] : index
    %value = view.load %src_view[%linear] : view<{shape.element_count}xf32, #dense> -> f32
    view.store %value, %dst_view[%linear] : f32, view<{shape.element_count}xf32, #dense>
  }}
  kernel.return
}}
"""


def generate_copy_vector_kernel(candidate, eps):
    shape = candidate.shape
    wg = candidate.workgroup_size
    vw = candidate.vector_width
    logical_cols = shape.ncols // vw
    attrs = cache_attrs(candidate.cache_policy)
    return f"""kernel.def @rms_norm_candidate() {{
  %unit = index.constant 1 : index
  %nrows = index.constant {shape.nrows} : index
  %workgroup_size = index.constant {wg} : index
  kernel.launch.config workgroups(%nrows, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%eps: f32, %src: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %ncols = index.constant {shape.ncols} : index
  %logical_cols = index.constant {logical_cols} : index
  %vector_width = index.constant {vw} : index
  %workgroup_size = index.constant {wg} : index
  %row0 = kernel.workgroup.id<x> : index
  %row = index.assume %row0 [range(%row0, 0, {shape.nrows - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {wg - 1})] : index

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{shape.element_count}xf32, #dense>
  %row_base = index.mul %row, %ncols : index

  scf.for %vcol = [%lane to %logical_cols step %workgroup_size] {{
    %col0 = index.mul %vcol, %vector_width : index
    %linear0 = index.add %row_base, %col0 : index
    %linear = index.assume %linear0 [range(%linear0, 0, {shape.element_count - vw}), mul(%linear0, {vw})] : index
    %values = vector.load %src_view[%linear]{attrs} : view<{shape.element_count}xf32, #dense> -> vector<{vw}xf32>
    vector.store %values, %dst_view[%linear] : vector<{vw}xf32>, view<{shape.element_count}xf32, #dense>
  }}
  kernel.return
}}
"""


def generate_source(candidate, eps):
    shape = candidate.shape
    if candidate.operation == "copy_floor" and candidate.vector_width == 1:
        kernel = generate_copy_scalar_kernel(candidate, eps)
        expected_path = f"../fixtures/rms_src_{shape.id}.npy"
    elif candidate.operation == "copy_floor":
        kernel = generate_copy_vector_kernel(candidate, eps)
        expected_path = f"../fixtures/rms_src_{shape.id}.npy"
    elif candidate.family == "vector_tail":
        kernel = generate_vector_tail_kernel(candidate, eps)
        expected_path = f"../fixtures/rms_expected_{shape.id}.npy"
    elif candidate.vector_width == 1:
        kernel = generate_scalar_kernel(candidate, eps)
        expected_path = f"../fixtures/rms_expected_{shape.id}.npy"
    else:
        kernel = generate_vector_kernel(candidate, eps)
        expected_path = f"../fixtures/rms_expected_{shape.id}.npy"
    return kernel + f"""
check.case @case_{candidate.id} {{
  %eps = check.literal value({eps:.9g}) : f32
  %src = check.file.read.npy path("../fixtures/rms_src_{shape.id}.npy") : tensor<{shape.nrows}x{shape.ncols}xf32>
  %dst = check.generate.fill value(0.0) : tensor<{shape.nrows}x{shape.ncols}xf32>
  %expected = check.file.read.npy path("{expected_path}") : tensor<{shape.nrows}x{shape.ncols}xf32>
  func.call @rms_norm_candidate(%eps, %src, %dst) : (f32, tensor<{shape.nrows}x{shape.ncols}xf32>, tensor<{shape.nrows}x{shape.ncols}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<{shape.nrows}x{shape.ncols}xf32>
  check.return
}}

check.benchmark<@case_{candidate.id}> @{candidate.benchmark}
"""


def enumerate_candidates(
    shapes,
    workgroup_sizes,
    vector_widths,
    cache_policies,
    include_copy_floor=False,
    include_vector_tail=False,
):
    candidates = []
    for shape, wg, vw, policy in itertools.product(shapes, workgroup_sizes, vector_widths, cache_policies):
        if vw == 1 and policy != "default":
            continue
        has_tail = vw > 1 and shape.ncols % vw != 0
        if has_tail and not include_vector_tail:
            continue
        if vw > 1 and policy not in ("default", "non_temporal"):
            continue
        if wg > 1024 or wg < 32:
            continue
        family = "scalar" if vw == 1 else ("vector_tail" if has_tail else "vector")
        candidates.append(Candidate(shape, "rms_norm", family, wg, vw, policy))
        if include_copy_floor and not has_tail:
            candidates.append(Candidate(shape, "copy_floor", family, wg, vw, policy))
    return candidates


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


def extract_first(rows, name):
    return next((row for row in rows if row.get("row") == name), None)


def summarize_compile(row):
    static = (row or {}).get("static_summary") or {}
    return {
        "status": (row or {}).get("status"),
        "artifact_size": static.get("artifact_size"),
        "instruction_count": static.get("instruction_count"),
        "code_byte_count": static.get("code_byte_count"),
        "private_memory_bytes": static.get("private_memory_bytes"),
        "local_memory_bytes": static.get("local_memory_bytes"),
        "allocation_spill_count": static.get("allocation_spill_count"),
        "register_pressure_peak_live_units": static.get("register_pressure_peak_live_units"),
        "vector_alu_count": static.get("vector_alu_count"),
        "global_memory_count": static.get("global_memory_count"),
        "local_memory_count": static.get("local_memory_count"),
    }


def summarize_benchmark(row):
    result = (row or {}).get("benchmark_result") or {}
    timing = result.get("operation_timing_ns") or {}
    return {
        "status": result.get("status"),
        "p50_ns": timing.get("p50"),
        "p90_ns": timing.get("p90"),
        "mean_ns": timing.get("mean"),
        "count": timing.get("count"),
        "measured_dispatch_count": result.get("measured_dispatch_count"),
        "measured_duration_ns": result.get("measured_duration_ns"),
        "stop_reason": result.get("stop_reason"),
        "data_cache": result.get("data_cache"),
    }


def median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2


def reduce_results(run_dir, results):
    grouped = {}
    for result in results:
        if result["status"] != "ok":
            continue
        if result.get("operation", "rms_norm") != "rms_norm":
            continue
        key = result["shape_id"]
        grouped.setdefault(key, []).append(result)

    winners = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda item: (
            item["benchmark_result"].get("p50_ns") if item["benchmark_result"].get("p50_ns") is not None else 10**30,
            item["benchmark_result"].get("p90_ns") if item["benchmark_result"].get("p90_ns") is not None else 10**30,
        ))
        winners[key] = rows[0]

    summary = {
        "schema": "hrx2-rms-norm-tune-summary-v1",
        "run_dir": str(run_dir),
        "result_count": len(results),
        "ok_count": sum(1 for item in results if item["status"] == "ok"),
        "failed_count": sum(1 for item in results if item["status"] != "ok"),
        "winners": {
            key: {
                "candidate_id": item["candidate_id"],
                "shape": item["shape"],
                "family": item["family"],
                "workgroup_size": item["workgroup_size"],
                "vector_width": item["vector_width"],
                "cache_policy": item["cache_policy"],
                "benchmark": item["benchmark_result"],
                "compile": item["compile_result"],
            }
            for key, item in winners.items()
        },
    }
    write_json(run_dir / "summary.json", summary)

    lines = [
        "# HRX2 RMS_NORM Tuning Summary",
        "",
        f"- Run: `{run_dir}`",
        f"- Results: {summary['ok_count']}/{summary['result_count']} ok",
        "",
        "| Shape | Family | WG | VW | Cache | Status | p50 ns | p90 ns | Inst | Code bytes | Spills | Peak live |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(results, key=lambda row: (row["shape_id"], row["family"], row["workgroup_size"], row["vector_width"], row["cache_policy"])):
        bench = item.get("benchmark_result") or {}
        comp = item.get("compile_result") or {}
        lines.append(
            f"| `{item['shape_id']}` | {item.get('operation', 'rms_norm')}:{item['family']} | {item['workgroup_size']} | "
            f"{item['vector_width']} | {item['cache_policy']} | {item['status']} | "
            f"{bench.get('p50_ns', '')} | {bench.get('p90_ns', '')} | "
            f"{comp.get('instruction_count', '')} | {comp.get('code_byte_count', '')} | "
            f"{comp.get('allocation_spill_count', '')} | {comp.get('register_pressure_peak_live_units', '')} |"
        )
    lines.extend(["", "## Winners", ""])
    for key, item in sorted(winners.items()):
        lines.append(
            f"- `{key}`: {item['family']} WG {item['workgroup_size']} VW "
            f"{item['vector_width']} cache {item['cache_policy']} at "
            f"{item['benchmark_result'].get('p50_ns')} ns p50"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def benchmark_candidate(args, run_dir, candidate):
    source_dir = run_dir / "variants"
    output_dir = run_dir / "benchmark_jsonl"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / candidate.source_name
    source_path.write_text(generate_source(candidate, args.eps), encoding="utf-8")
    repetitions = []
    plan = None
    compile_result = {}
    failure_rows = []
    command = None
    for rep in range(args.repetitions):
        output_path = output_dir / f"{candidate.id}_rep{rep}.jsonl"
        cmd = [
            str(args.iree_benchmark_loom),
            str(source_path),
            "--device=amdgpu",
            f"--benchmark=@{candidate.benchmark}",
            "--measure=dispatch_complete",
            "--sample=0",
            "--sample-compilation=per_sample",
            f"--iterations={args.iterations}",
            f"--warmup-iterations={args.warmup_iterations}",
            "--min-time-ms=0",
            f"--max-batches={args.iterations}",
            "--stable-p90-to-p50-ppm=0",
            "--input-ring-count=1",
            "--compile-report=summary",
            "--output-format=jsonl",
            f"--output={output_path}",
        ]
        command = cmd
        run = run_command(cmd, args.timeout)
        rows = read_jsonl(output_path)
        rep_plan = extract_first(rows, "plan")
        compile_row = extract_first(rows, "compile")
        benchmark_row = extract_first(rows, "benchmark")
        rep_compile = summarize_compile(compile_row)
        rep_benchmark = summarize_benchmark(benchmark_row)
        rep_status = "ok" if run["returncode"] == 0 and rep_benchmark.get("status") == "ok" else "failed"
        if run["timed_out"]:
            rep_status = "timeout"
        repetitions.append({
            "repetition": rep,
            "status": rep_status,
            "returncode": run["returncode"],
            "timed_out": run["timed_out"],
            "stdout": run["stdout"],
            "stderr": run["stderr"],
            "output_path": str(output_path),
            "compile_result": rep_compile,
            "benchmark_result": rep_benchmark,
        })
        if plan is None:
            plan = rep_plan
        if not compile_result and rep_compile:
            compile_result = rep_compile
        failure_rows.extend(row for row in rows if row.get("row") == "failure")

    ok_reps = [rep for rep in repetitions if rep["status"] == "ok"]
    p50_values = [rep["benchmark_result"].get("p50_ns") for rep in ok_reps]
    p90_values = [rep["benchmark_result"].get("p90_ns") for rep in ok_reps]
    mean_values = [rep["benchmark_result"].get("mean_ns") for rep in ok_reps]
    benchmark_result = {
        "status": "ok" if len(ok_reps) == args.repetitions else "failed",
        "p50_ns": median(p50_values),
        "p90_ns": median(p90_values),
        "mean_ns": median(mean_values),
        "repetition_count": args.repetitions,
        "ok_repetition_count": len(ok_reps),
        "p50_ns_repetitions": p50_values,
        "p90_ns_repetitions": p90_values,
    }
    status = "ok" if benchmark_result["status"] == "ok" else "failed"
    return {
        "schema": "hrx2-rms-norm-tune-result-v1",
        "candidate_id": candidate.id,
        "operation": candidate.operation,
        "shape_id": candidate.shape.id,
        "shape": {"ncols": candidate.shape.ncols, "nrows": candidate.shape.nrows},
        "family": candidate.family,
        "workgroup_size": candidate.workgroup_size,
        "vector_width": candidate.vector_width,
        "cache_policy": candidate.cache_policy,
        "status": status,
        "command": command,
        "returncode": 0 if status == "ok" else 1,
        "timed_out": any(rep["timed_out"] for rep in repetitions),
        "source_path": str(source_path),
        "plan": plan,
        "compile_result": compile_result,
        "benchmark_result": benchmark_result,
        "repetitions": repetitions,
        "failures": failure_rows,
    }


def main():
    args = parse_args()
    run_dir = Path(args.out_root) / args.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    shapes = parse_shapes(args.shapes)
    workgroup_sizes = split_csv_ints(args.workgroup_sizes)
    vector_widths = split_csv_ints(args.vector_widths)
    cache_policies = split_csv(args.cache_policies)
    candidates = enumerate_candidates(
        shapes,
        workgroup_sizes,
        vector_widths,
        cache_policies,
        include_copy_floor=args.include_copy_floor,
        include_vector_tail=args.include_vector_tail,
    )
    if args.max_candidates > 0:
        candidates = candidates[:args.max_candidates]
    ensure_fixtures(run_dir, shapes, args.eps, args.seed)
    write_json(run_dir / "candidate_manifest.json", [
        {
            "candidate_id": candidate.id,
            "operation": candidate.operation,
            "shape": {"ncols": candidate.shape.ncols, "nrows": candidate.shape.nrows},
            "family": candidate.family,
            "workgroup_size": candidate.workgroup_size,
            "vector_width": candidate.vector_width,
            "cache_policy": candidate.cache_policy,
        }
        for candidate in candidates
    ])

    if args.dry_run:
        print(f"planned {len(candidates)} candidates under {run_dir}", file=sys.stderr)
        return

    results = []
    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as f:
        for ordinal, candidate in enumerate(candidates, start=1):
            result = benchmark_candidate(args, run_dir, candidate)
            results.append(result)
            f.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            p50 = result.get("benchmark_result", {}).get("p50_ns", "")
            print(
                f"{ordinal}/{len(candidates)} {candidate.id} "
                f"status={result['status']} p50={p50}",
                file=sys.stderr,
            )

    summary = reduce_results(run_dir, results)
    print(f"wrote {results_path}", file=sys.stderr)
    print(f"wrote {run_dir / 'summary.md'}", file=sys.stderr)
    if summary["ok_count"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
