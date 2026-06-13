#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import random
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hrx2_pipeline_lib import DEFAULT_HRX_INSTALL, WORKSPACE, env_for_tools, read_jsonl, write_json


DEFAULT_OUT_ROOT = WORKSPACE / "cache" / "hrx2" / "q8_0_f32_tune"
DEFAULT_SHAPES = "512x64x8,4096x128x1,4096x128x8,4096x512x1,4096x512x8,8192x128x8"
LISTING_PATTERNS = {
    "global_load_b32": r"\bglobal_load_b32\b",
    "global_load_d16_b16": r"\bglobal_load_d16_b16\b",
    "global_load_b128": r"\bglobal_load_b128\b",
    "v_bfe_i32": r"\bv_bfe_i32\b",
    "v_fma_mix_f32": r"\bv_fma_mix_f32\b",
    "v_fmac_f32": r"\bv_fmac_f32\b",
}
ALGORITHM_REQUIRED_SIGNATURES = {
    "block4_rhsvec": ["global_load_b128"],
    "block4_rhsvec_dotf": ["global_load_b128", "v_fmac_f32"],
    "word4_rhsvec": ["global_load_b32", "global_load_b128"],
    "word4_rhsvec_dotf": ["global_load_b32", "global_load_b128", "v_fmac_f32"],
    "word4_bitunpack_rhsvec_dotf": [
        "global_load_b32",
        "global_load_d16_b16",
        "global_load_b128",
        "v_bfe_i32",
        "v_fma_mix_f32",
        "v_fmac_f32",
    ],
    "word4_bitunpack_scalefirst_rhsvec_dotf": [
        "global_load_b32",
        "global_load_d16_b16",
        "global_load_b128",
        "v_bfe_i32",
        "v_fma_mix_f32",
        "v_fmac_f32",
    ],
    "word4_bitunpack_unrolled_rhsvec_dotf": [
        "global_load_b32",
        "global_load_d16_b16",
        "global_load_b128",
        "v_bfe_i32",
        "v_fma_mix_f32",
        "v_fmac_f32",
    ],
    "word4_bitunpack_scfunroll_rhsvec_dotf": [
        "global_load_b32",
        "global_load_d16_b16",
        "global_load_b128",
        "v_bfe_i32",
        "v_fma_mix_f32",
        "v_fmac_f32",
    ],
    "word4_bitunpack_rhsvec_dotf_subgroup": [
        "global_load_b32",
        "global_load_d16_b16",
        "global_load_b128",
        "v_bfe_i32",
        "v_fma_mix_f32",
        "v_fmac_f32",
    ],
}


@dataclass(frozen=True)
class Shape:
    k: int
    rows: int
    cols: int

    @property
    def id(self):
        return f"k{self.k}_r{self.rows}_c{self.cols}"

    @property
    def blocks_per_row(self):
        return self.k // 32

    @property
    def src0_bytes(self):
        return self.rows * self.blocks_per_row * 34

    @property
    def src1_count(self):
        return self.cols * self.k

    @property
    def dst_count(self):
        return self.cols * self.rows


@dataclass(frozen=True)
class Candidate:
    shape: Shape
    workgroup_size: int
    rows_per_workgroup: int
    cols_per_workgroup: int
    algorithm: str

    @property
    def id(self):
        return (
            f"q8_0_f32_{self.algorithm}_{self.shape.id}"
            f"_rpg{self.rows_per_workgroup}_cpg{self.cols_per_workgroup}_wg{self.workgroup_size}"
        )

    @property
    def benchmark(self):
        return f"bench_{self.id}"


def parse_args():
    parser = argparse.ArgumentParser(description="Tune exact-shape HRX2 Q8_0/F32 MUL_MAT Loom variants.")
    parser.add_argument("--run-id", default="gfx1100-q8-0-f32-preflight")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--iree-benchmark-loom", default=str(DEFAULT_HRX_INSTALL / "bin" / "iree-benchmark-loom"))
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help="Comma-separated kxrowsxcols list.")
    parser.add_argument("--workgroup-sizes", default="64,128,256")
    parser.add_argument("--rows-per-workgroup", default="1,2,4")
    parser.add_argument(
        "--cols-per-workgroup",
        default="1",
        help="Comma-separated output columns per workgroup. Values >1 are supported for word4_bitunpack_rhsvec_dotf.",
    )
    parser.add_argument(
        "--algorithms",
        default="scalar",
        help=(
            "Comma-separated algorithm families: scalar,block4,block4_rhsvec,"
            "block4_rhsvec_dotf,word4_rhsvec,word4_rhsvec_dotf,"
            "word4_bitunpack_rhsvec_dotf,word4_bitunpack_scalefirst_rhsvec_dotf,"
            "word4_bitunpack_unrolled_rhsvec_dotf,word4_bitunpack_scfunroll_rhsvec_dotf,"
            "word4_bitunpack_rhsvec_dotf_subgroup,"
            "word4_rhsvec_mixasm_hi,chunk4. "
            "Default keeps historical scalar search."
        ),
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def split_csv_strings(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_shapes(text):
    shapes = []
    for part in [item.strip() for item in text.split(",") if item.strip()]:
        pieces = part.lower().split("x")
        if len(pieces) != 3:
            raise ValueError(f"shape {part!r} must be kxrowsxcols")
        shape = Shape(*(int(piece) for piece in pieces))
        if shape.k % 32 != 0:
            raise ValueError(f"shape {part!r} k must be divisible by 32 for Q8_0")
        shapes.append(shape)
    return shapes


def npy_header(shape, descr):
    shape_text = "(" + ", ".join(str(dim) for dim in shape)
    if len(shape) == 1:
        shape_text += ","
    shape_text += ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    prefix_len = 10
    pad_len = 16 - ((prefix_len + len(header) + 1) % 16)
    header = header + (" " * pad_len) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("ascii")


def write_npy(path, shape, descr, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(npy_header(shape, descr))
        f.write(payload)


def q8_fixtures(shape, seed):
    rng = random.Random(seed + shape.k * 3 + shape.rows * 17 + shape.cols * 101)
    src0_bytes = bytearray()
    dequantized = []
    for row in range(shape.rows):
        row_values = []
        for block in range(shape.blocks_per_row):
            d = 0.00390625 * (1 + ((row + block) % 7))
            d_half = struct.unpack("<e", struct.pack("<e", d))[0]
            src0_bytes.extend(struct.pack("<e", d_half))
            for qi in range(32):
                q = rng.randint(-63, 63)
                src0_bytes.extend(struct.pack("b", q))
                row_values.append(float(q) * d_half)
        dequantized.append(row_values)

    src1 = []
    for col in range(shape.cols):
        for k in range(shape.k):
            wave = math.sin((k + 1) * 0.013 + col * 0.11) * 0.5
            bias = ((k % 17) - 8) * 0.0078125
            src1.append(wave + bias + rng.uniform(-0.125, 0.125))

    expected = []
    for col in range(shape.cols):
        rhs = src1[col * shape.k:(col + 1) * shape.k]
        for row in range(shape.rows):
            expected.append(sum(a * b for a, b in zip(dequantized[row], rhs)))
    return bytes(src0_bytes), src1, expected


def ensure_fixtures(run_dir, shapes, seed):
    fixture_dir = run_dir / "fixtures"
    manifest = []
    for shape in shapes:
        src0_bytes, src1, expected = q8_fixtures(shape, seed)
        src0_path = fixture_dir / f"q8_src0_{shape.id}.npy"
        src1_path = fixture_dir / f"q8_src1_{shape.id}.npy"
        expected_path = fixture_dir / f"q8_expected_{shape.id}.npy"
        write_npy(src0_path, (shape.src0_bytes,), "|i1", src0_bytes)
        write_npy(src1_path, (shape.src1_count,), "<f4", struct.pack("<" + "f" * len(src1), *src1))
        write_npy(expected_path, (shape.dst_count,), "<f4", struct.pack("<" + "f" * len(expected), *expected))
        manifest.append({
            "shape": {"k": shape.k, "rows": shape.rows, "cols": shape.cols},
            "src0": str(src0_path),
            "src1": str(src1_path),
            "expected": str(expected_path),
        })
    write_json(run_dir / "fixtures_manifest.json", manifest)


def generate_scalar_sum_block(shape, wg, row_symbol, suffix):
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %i_{suffix} = [%lane to %k step %workgroup_size](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %block_{suffix} = index.div %i_{suffix}, %qk : index
    %in_block_{suffix} = index.rem %i_{suffix}, %qk : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_base_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_base_{suffix}, %in_block_{suffix} : index
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %src1_index_{suffix} = index.add %src1_col_base, %i_{suffix} : index
    %q_{suffix} = view.load %src0_i8_view[%q_byte_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %rhs_{suffix} = view.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> f32
    %q_f32_{suffix} = scalar.sitofp %q_{suffix} : i8 to f32
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %scaled_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q_f32_{suffix}, %d_{suffix} : f32
    %product_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %scaled_{suffix}, %rhs_{suffix} : f32
    %next_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %acc_{suffix}, %product_{suffix} : f32
    scf.yield %next_{suffix} : f32
  }}
  %dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_chunk4_sum_block(shape, wg, row_symbol, suffix):
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %q_i8_{suffix} = vector.load %src0_i8_view[%q_byte_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> vector<4xi8>
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %q0_{suffix} = vector.extract %q_i8_{suffix}[0] : vector<4xi8> -> i8
    %q1_{suffix} = vector.extract %q_i8_{suffix}[1] : vector<4xi8> -> i8
    %q2_{suffix} = vector.extract %q_i8_{suffix}[2] : vector<4xi8> -> i8
    %q3_{suffix} = vector.extract %q_i8_{suffix}[3] : vector<4xi8> -> i8
    %rhs0_{suffix} = vector.extract %rhs_{suffix}[0] : vector<4xf32> -> f32
    %rhs1_{suffix} = vector.extract %rhs_{suffix}[1] : vector<4xf32> -> f32
    %rhs2_{suffix} = vector.extract %rhs_{suffix}[2] : vector<4xf32> -> f32
    %rhs3_{suffix} = vector.extract %rhs_{suffix}[3] : vector<4xf32> -> f32
    %q0_f32_{suffix} = scalar.sitofp %q0_{suffix} : i8 to f32
    %q1_f32_{suffix} = scalar.sitofp %q1_{suffix} : i8 to f32
    %q2_f32_{suffix} = scalar.sitofp %q2_{suffix} : i8 to f32
    %q3_f32_{suffix} = scalar.sitofp %q3_{suffix} : i8 to f32
    %s0_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q0_f32_{suffix}, %d_{suffix} : f32
    %s1_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q1_f32_{suffix}, %d_{suffix} : f32
    %s2_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q2_f32_{suffix}, %d_{suffix} : f32
    %s3_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q3_f32_{suffix}, %d_{suffix} : f32
    %p0_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s0_{suffix}, %rhs0_{suffix} : f32
    %p1_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s1_{suffix}, %rhs1_{suffix} : f32
    %p2_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s2_{suffix}, %rhs2_{suffix} : f32
    %p3_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s3_{suffix}, %rhs3_{suffix} : f32
    %sum01_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %p0_{suffix}, %p1_{suffix} : f32
    %sum23_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %p2_{suffix}, %p3_{suffix} : f32
    %sum4_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %sum01_{suffix}, %sum23_{suffix} : f32
    %next_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %acc_{suffix}, %sum4_{suffix} : f32
    scf.yield %next_{suffix} : f32
  }}
  %dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_block4_sum_block(shape, wg, row_symbol, suffix):
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte0_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte1_{suffix} = index.add %q_byte0_{suffix}, %one : index
    %q_byte2_{suffix} = index.add %q_byte0_{suffix}, %two : index
    %q_byte3_{suffix} = index.add %q_byte0_{suffix}, %three : index
    %rhs_index0_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %rhs_index1_{suffix} = index.add %rhs_index0_{suffix}, %one : index
    %rhs_index2_{suffix} = index.add %rhs_index0_{suffix}, %two : index
    %rhs_index3_{suffix} = index.add %rhs_index0_{suffix}, %three : index
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %q0_{suffix} = view.load %src0_i8_view[%q_byte0_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %q1_{suffix} = view.load %src0_i8_view[%q_byte1_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %q2_{suffix} = view.load %src0_i8_view[%q_byte2_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %q3_{suffix} = view.load %src0_i8_view[%q_byte3_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %rhs0_{suffix} = view.load %src1_view[%rhs_index0_{suffix}] : view<{shape.src1_count}xf32, #dense> -> f32
    %rhs1_{suffix} = view.load %src1_view[%rhs_index1_{suffix}] : view<{shape.src1_count}xf32, #dense> -> f32
    %rhs2_{suffix} = view.load %src1_view[%rhs_index2_{suffix}] : view<{shape.src1_count}xf32, #dense> -> f32
    %rhs3_{suffix} = view.load %src1_view[%rhs_index3_{suffix}] : view<{shape.src1_count}xf32, #dense> -> f32
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %q0_f32_{suffix} = scalar.sitofp %q0_{suffix} : i8 to f32
    %q1_f32_{suffix} = scalar.sitofp %q1_{suffix} : i8 to f32
    %q2_f32_{suffix} = scalar.sitofp %q2_{suffix} : i8 to f32
    %q3_f32_{suffix} = scalar.sitofp %q3_{suffix} : i8 to f32
    %s0_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q0_f32_{suffix}, %d_{suffix} : f32
    %s1_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q1_f32_{suffix}, %d_{suffix} : f32
    %s2_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q2_f32_{suffix}, %d_{suffix} : f32
    %s3_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q3_f32_{suffix}, %d_{suffix} : f32
    %p0_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s0_{suffix}, %rhs0_{suffix} : f32
    %p1_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s1_{suffix}, %rhs1_{suffix} : f32
    %p2_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s2_{suffix}, %rhs2_{suffix} : f32
    %p3_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %s3_{suffix}, %rhs3_{suffix} : f32
    %sum01_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %p0_{suffix}, %p1_{suffix} : f32
    %sum23_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %p2_{suffix}, %p3_{suffix} : f32
    %sum4_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %sum01_{suffix}, %sum23_{suffix} : f32
    %next_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %acc_{suffix}, %sum4_{suffix} : f32
    scf.yield %next_{suffix} : f32
  }}
  %dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_block4_rhsvec_sum_block(shape, wg, row_symbol, suffix, use_dotf=False):
    accumulate = (
        f"    %next_{suffix} = vector.dotf %qvec_{suffix}, %rhs_{suffix}, %acc_{suffix} : vector<4xf32>, vector<4xf32>, f32"
        if use_dotf else
        f"""    %products_{suffix} = vector.mulf<reassoc|nnan|ninf|nsz> %qvec_{suffix}, %rhs_{suffix} : vector<4xf32>
    %next_{suffix} = vector.reduce<addf, reassoc|nnan|ninf|nsz> %products_{suffix}, %acc_{suffix} : vector<4xf32>, f32"""
    )
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte0_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte1_{suffix} = index.add %q_byte0_{suffix}, %one : index
    %q_byte2_{suffix} = index.add %q_byte0_{suffix}, %two : index
    %q_byte3_{suffix} = index.add %q_byte0_{suffix}, %three : index
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %q0_{suffix} = view.load %src0_i8_view[%q_byte0_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %q1_{suffix} = view.load %src0_i8_view[%q_byte1_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %q2_{suffix} = view.load %src0_i8_view[%q_byte2_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %q3_{suffix} = view.load %src0_i8_view[%q_byte3_{suffix}] : view<{shape.src0_bytes}xi8, #dense> -> i8
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %q0_f32_{suffix} = scalar.sitofp %q0_{suffix} : i8 to f32
    %q1_f32_{suffix} = scalar.sitofp %q1_{suffix} : i8 to f32
    %q2_f32_{suffix} = scalar.sitofp %q2_{suffix} : i8 to f32
    %q3_f32_{suffix} = scalar.sitofp %q3_{suffix} : i8 to f32
    %s0_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q0_f32_{suffix}, %d_{suffix} : f32
    %s1_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q1_f32_{suffix}, %d_{suffix} : f32
    %s2_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q2_f32_{suffix}, %d_{suffix} : f32
    %s3_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q3_f32_{suffix}, %d_{suffix} : f32
    %qvec_{suffix} = vector.from_elements %s0_{suffix}, %s1_{suffix}, %s2_{suffix}, %s3_{suffix} : vector<4xf32>
{accumulate}
    scf.yield %next_{suffix} : f32
  }}
  %dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_word4_rhsvec_sum_block(shape, wg, row_symbol, suffix, use_dotf=False):
    accumulate = (
        f"    %next_{suffix} = vector.dotf %qvec_{suffix}, %rhs_{suffix}, %acc_{suffix} : vector<4xf32>, vector<4xf32>, f32"
        if use_dotf else
        f"""    %products_{suffix} = vector.mulf<reassoc|nnan|ninf|nsz> %qvec_{suffix}, %rhs_{suffix} : vector<4xf32>
    %next_{suffix} = vector.reduce<addf, reassoc|nnan|ninf|nsz> %products_{suffix}, %acc_{suffix} : vector<4xf32>, f32"""
    )
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte_offset_{suffix} = index.cast %q_byte_{suffix} : index to offset
    %q_word_view_{suffix} = buffer.view %src0_noalias[%q_byte_offset_{suffix}] : buffer -> view<1xi32, #dense>
    %q_word_{suffix} = view.load %q_word_view_{suffix}[%zero_index] : view<1xi32, #dense> -> i32
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %q0_shifted_{suffix} = scalar.shli %q_word_{suffix}, %shift24 : i32
    %q1_left_{suffix} = scalar.shli %q_word_{suffix}, %shift16 : i32
    %q2_left_{suffix} = scalar.shli %q_word_{suffix}, %shift8 : i32
    %q0_i32_{suffix} = scalar.shrsi %q0_shifted_{suffix}, %shift24 : i32
    %q1_i32_{suffix} = scalar.shrsi %q1_left_{suffix}, %shift24 : i32
    %q2_i32_{suffix} = scalar.shrsi %q2_left_{suffix}, %shift24 : i32
    %q3_i32_{suffix} = scalar.shrsi %q_word_{suffix}, %shift24 : i32
    %q0_f32_{suffix} = scalar.sitofp %q0_i32_{suffix} : i32 to f32
    %q1_f32_{suffix} = scalar.sitofp %q1_i32_{suffix} : i32 to f32
    %q2_f32_{suffix} = scalar.sitofp %q2_i32_{suffix} : i32 to f32
    %q3_f32_{suffix} = scalar.sitofp %q3_i32_{suffix} : i32 to f32
    %s0_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q0_f32_{suffix}, %d_{suffix} : f32
    %s1_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q1_f32_{suffix}, %d_{suffix} : f32
    %s2_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q2_f32_{suffix}, %d_{suffix} : f32
    %s3_{suffix} = scalar.mulf<reassoc|nnan|ninf|nsz> %q3_f32_{suffix}, %d_{suffix} : f32
    %qvec_{suffix} = vector.from_elements %s0_{suffix}, %s1_{suffix}, %s2_{suffix}, %s3_{suffix} : vector<4xf32>
{accumulate}
    scf.yield %next_{suffix} : f32
  }}
  %dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_word4_bitunpack_rhsvec_sum_block(shape, wg, row_symbol, suffix, reduce_op="workgroup"):
    reduction = (
        f"%dot_{suffix} = kernel.subgroup.reduce<addf> %sum_{suffix} : f32"
        if reduce_op == "subgroup" else
        f"%dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32"
    )
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte_offset_{suffix} = index.cast %q_byte_{suffix} : index to offset
    %q_word_view_{suffix} = buffer.view %src0_noalias[%q_byte_offset_{suffix}] : buffer -> view<1xi32, #dense>
    %q_packed_{suffix} = vector.load %q_word_view_{suffix}[%zero_index] : view<1xi32, #dense> -> vector<1xi32>
    %q_i32_{suffix} = vector.bitunpacks<8> %q_packed_{suffix} : vector<1xi32> -> vector<4xi32>
    %q_f32_{suffix} = vector.sitofp %q_i32_{suffix} : vector<4xi32> to vector<4xf32>
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %d_vec_{suffix} = vector.splat %d_{suffix} : vector<4xf32>
    %q_scaled_{suffix} = vector.mulf<reassoc|nnan|ninf|nsz|contract> %q_f32_{suffix}, %d_vec_{suffix} : vector<4xf32>
    %next_{suffix} = vector.dotf %q_scaled_{suffix}, %rhs_{suffix}, %acc_{suffix} : vector<4xf32>, vector<4xf32>, f32
    scf.yield %next_{suffix} : f32
  }}
  {reduction}
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_word4_bitunpack_scalefirst_rhsvec_sum_block(shape, wg, row_symbol, suffix, reduce_op="workgroup"):
    reduction = (
        f"%dot_{suffix} = kernel.subgroup.reduce<addf> %sum_{suffix} : f32"
        if reduce_op == "subgroup" else
        f"%dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32"
    )
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte_offset_{suffix} = index.cast %q_byte_{suffix} : index to offset
    %q_word_view_{suffix} = buffer.view %src0_noalias[%q_byte_offset_{suffix}] : buffer -> view<1xi32, #dense>
    %q_packed_{suffix} = vector.load %q_word_view_{suffix}[%zero_index] : view<1xi32, #dense> -> vector<1xi32>
    %q_i32_{suffix} = vector.bitunpacks<8> %q_packed_{suffix} : vector<1xi32> -> vector<4xi32>
    %q_f32_{suffix} = vector.sitofp %q_i32_{suffix} : vector<4xi32> to vector<4xf32>
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %d_vec_{suffix} = vector.splat %d_{suffix} : vector<4xf32>
    %q_scaled_{suffix} = vector.mulf<reassoc|nnan|ninf|nsz|contract> %q_f32_{suffix}, %d_vec_{suffix} : vector<4xf32>
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %next_{suffix} = vector.dotf %q_scaled_{suffix}, %rhs_{suffix}, %acc_{suffix} : vector<4xf32>, vector<4xf32>, f32
    scf.yield %next_{suffix} : f32
  }}
  {reduction}
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_word4_bitunpack_unrolled_rhsvec_sum_block(shape, wg, row_symbol, suffix, reduce_op="workgroup"):
    block_step = wg // 8
    if shape.blocks_per_row % block_step != 0:
        raise ValueError("unrolled Q8 path requires blocks_per_row divisible by WG/8")
    reduction = (
        f"%dot_{suffix} = kernel.subgroup.reduce<addf> %sum_{suffix} : f32"
        if reduce_op == "subgroup" else
        f"%dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32"
    )
    lines = [
        f"  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index",
        f"  %acc_init_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> %zero_f32_{suffix}, %zero_f32_{suffix} : f32",
    ]
    acc_name = f"%acc_init_{suffix}"
    for iter_idx in range(shape.blocks_per_row // block_step):
        step = iter_idx * block_step
        if step == 0:
            lines.append(f"  %block_idx_{suffix}_{iter_idx} = index.assume %block_slot [range(%block_slot, 0, {shape.blocks_per_row - 1})] : index")
        else:
            lines.extend([
                f"  %block_unroll_step_{suffix}_{iter_idx} = index.constant {step} : index",
                f"  %block_idx_raw_{suffix}_{iter_idx} = index.add %block_slot, %block_unroll_step_{suffix}_{iter_idx} : index",
                f"  %block_idx_{suffix}_{iter_idx} = index.assume %block_idx_raw_{suffix}_{iter_idx} [range(%block_idx_raw_{suffix}_{iter_idx}, 0, {shape.blocks_per_row - 1})] : index",
            ])
        lines.extend([
            f"  %src_base_block_{suffix}_{iter_idx} = index.mul %block_idx_{suffix}_{iter_idx}, %qk : index",
            f"  %src_base_{suffix}_{iter_idx} = index.add %src_base_block_{suffix}_{iter_idx}, %in_block_base : index",
            f"  %linear_block_{suffix}_{iter_idx} = index.add %row_block_base_{suffix}, %block_idx_{suffix}_{iter_idx} : index",
            f"  %block_byte_base_{suffix}_{iter_idx} = index.mul %linear_block_{suffix}_{iter_idx}, %block_bytes : index",
            f"  %q_byte_payload_{suffix}_{iter_idx} = index.add %block_byte_base_{suffix}_{iter_idx}, %qs_offset : index",
            f"  %q_byte_{suffix}_{iter_idx} = index.add %q_byte_payload_{suffix}_{iter_idx}, %in_block_base : index",
            f"  %q_byte_offset_{suffix}_{iter_idx} = index.cast %q_byte_{suffix}_{iter_idx} : index to offset",
            f"  %q_word_view_{suffix}_{iter_idx} = buffer.view %src0_noalias[%q_byte_offset_{suffix}_{iter_idx}] : buffer -> view<1xi32, #dense>",
            f"  %q_packed_{suffix}_{iter_idx} = vector.load %q_word_view_{suffix}_{iter_idx}[%zero_index] : view<1xi32, #dense> -> vector<1xi32>",
            f"  %q_i32_{suffix}_{iter_idx} = vector.bitunpacks<8> %q_packed_{suffix}_{iter_idx} : vector<1xi32> -> vector<4xi32>",
            f"  %q_f32_{suffix}_{iter_idx} = vector.sitofp %q_i32_{suffix}_{iter_idx} : vector<4xi32> to vector<4xf32>",
            f"  %src1_index_{suffix}_{iter_idx} = index.add %src1_col_base, %src_base_{suffix}_{iter_idx} : index",
            f"  %rhs_{suffix}_{iter_idx} = vector.load %src1_view[%src1_index_{suffix}_{iter_idx}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>",
            f"  %block_f16_base_{suffix}_{iter_idx} = index.mul %linear_block_{suffix}_{iter_idx}, %block_f16s : index",
            f"  %d_f16_{suffix}_{iter_idx} = view.load %src0_f16_view[%block_f16_base_{suffix}_{iter_idx}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16",
            f"  %d_{suffix}_{iter_idx} = scalar.extf %d_f16_{suffix}_{iter_idx} : f16 to f32",
            f"  %d_vec_{suffix}_{iter_idx} = vector.splat %d_{suffix}_{iter_idx} : vector<4xf32>",
            f"  %q_scaled_{suffix}_{iter_idx} = vector.mulf<reassoc|nnan|ninf|nsz|contract> %q_f32_{suffix}_{iter_idx}, %d_vec_{suffix}_{iter_idx} : vector<4xf32>",
            f"  %next_{suffix}_{iter_idx} = vector.dotf %q_scaled_{suffix}_{iter_idx}, %rhs_{suffix}_{iter_idx}, {acc_name} : vector<4xf32>, vector<4xf32>, f32",
        ])
        acc_name = f"%next_{suffix}_{iter_idx}"
    lines.extend([
        f"  %sum_{suffix} = scalar.addf<reassoc|nnan|ninf|nsz> {acc_name}, %zero_f32_{suffix} : f32",
        f"  {reduction}",
        f"  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index",
    ])
    return "\n".join(lines)


def generate_word4_bitunpack_scfunroll_rhsvec_sum_block(shape, wg, row_symbol, suffix, reduce_op="workgroup"):
    block_step = wg // 8
    if shape.blocks_per_row % block_step != 0:
        raise ValueError("scf-unrolled Q8 path requires blocks_per_row divisible by WG/8")
    trip_count = shape.blocks_per_row // block_step
    reduction = (
        f"%dot_{suffix} = kernel.subgroup.reduce<addf> %sum_{suffix} : f32"
        if reduce_op == "subgroup" else
        f"%dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32"
    )
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %trip_count_{suffix} = index.constant {trip_count} : index
  %sum_{suffix} = scf.for %unroll_iter_{suffix} = [%zero_index to %trip_count_{suffix} step %one](%acc_{suffix} = %zero_f32_{suffix} : f32) -> (f32) unroll(%q8_unroll_factor) {{
    %block_iter_step_{suffix} = index.mul %unroll_iter_{suffix}, %block_step : index
    %block_idx_raw_{suffix} = index.add %block_slot, %block_iter_step_{suffix} : index
    %block_idx_{suffix} = index.assume %block_idx_raw_{suffix} [range(%block_idx_raw_{suffix}, 0, {shape.blocks_per_row - 1})] : index
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte_offset_{suffix} = index.cast %q_byte_{suffix} : index to offset
    %q_word_view_{suffix} = buffer.view %src0_noalias[%q_byte_offset_{suffix}] : buffer -> view<1xi32, #dense>
    %q_packed_{suffix} = vector.load %q_word_view_{suffix}[%zero_index] : view<1xi32, #dense> -> vector<1xi32>
    %q_i32_{suffix} = vector.bitunpacks<8> %q_packed_{suffix} : vector<1xi32> -> vector<4xi32>
    %q_f32_{suffix} = vector.sitofp %q_i32_{suffix} : vector<4xi32> to vector<4xf32>
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %d_vec_{suffix} = vector.splat %d_{suffix} : vector<4xf32>
    %q_scaled_{suffix} = vector.mulf<reassoc|nnan|ninf|nsz|contract> %q_f32_{suffix}, %d_vec_{suffix} : vector<4xf32>
    %next_{suffix} = vector.dotf %q_scaled_{suffix}, %rhs_{suffix}, %acc_{suffix} : vector<4xf32>, vector<4xf32>, f32
    scf.yield %next_{suffix} : f32
  }}
  {reduction}
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_word4_bitunpack_rhsvec_multi_col_sum_block(shape, wg, row_symbol, suffix, cpg, reduce_op="workgroup"):
    if cpg <= 1:
        return generate_word4_bitunpack_rhsvec_sum_block(shape, wg, row_symbol, suffix, reduce_op=reduce_op)
    init_args = ", ".join(f"%acc{col}_{suffix} = %zero_f32_{suffix}_{col} : f32" for col in range(cpg))
    result_types = ", ".join("f32" for _ in range(cpg))
    src1_indices = []
    rhs_loads = []
    next_values = []
    for col in range(cpg):
        src1_indices.append(
            f"""    %col_offset_{col}_{suffix} = index.constant {col} : index
    %col_{col}_{suffix} = index.add %col_group_base, %col_offset_{col}_{suffix} : index
    %src1_col_base_{col}_{suffix} = index.mul %col_{col}_{suffix}, %k : index
    %src1_index_{col}_{suffix} = index.add %src1_col_base_{col}_{suffix}, %src_base_{suffix} : index"""
        )
        rhs_loads.append(
            f"    %rhs{col}_{suffix} = vector.load %src1_view[%src1_index_{col}_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>"
        )
        next_values.append(
            f"    %next{col}_{suffix} = vector.dotf %q_scaled_{suffix}, %rhs{col}_{suffix}, %acc{col}_{suffix} : vector<4xf32>, vector<4xf32>, f32"
        )
    yield_values = ", ".join(f"%next{col}_{suffix}" for col in range(cpg))
    acc_results = ", ".join(f"%sum{col}_{suffix}" for col in range(cpg))
    reduce_kind = "subgroup" if reduce_op == "subgroup" else "workgroup"
    reduce_lines = "\n".join(
        f"  %dot{col}_{suffix} = kernel.{reduce_kind}.reduce<addf> %sum{col}_{suffix} : f32"
        for col in range(cpg)
    )
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  {acc_results} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step]({init_args}) -> ({result_types}) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte_offset_{suffix} = index.cast %q_byte_{suffix} : index to offset
    %q_word_view_{suffix} = buffer.view %src0_noalias[%q_byte_offset_{suffix}] : buffer -> view<1xi32, #dense>
    %q_packed_{suffix} = vector.load %q_word_view_{suffix}[%zero_index] : view<1xi32, #dense> -> vector<1xi32>
    %q_i32_{suffix} = vector.bitunpacks<8> %q_packed_{suffix} : vector<1xi32> -> vector<4xi32>
    %q_f32_{suffix} = vector.sitofp %q_i32_{suffix} : vector<4xi32> to vector<4xf32>
{chr(10).join(src1_indices)}
{chr(10).join(rhs_loads)}
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %d_{suffix} = scalar.extf %d_f16_{suffix} : f16 to f32
    %d_vec_{suffix} = vector.splat %d_{suffix} : vector<4xf32>
    %q_scaled_{suffix} = vector.mulf<reassoc|nnan|ninf|nsz|contract> %q_f32_{suffix}, %d_vec_{suffix} : vector<4xf32>
{chr(10).join(next_values)}
    scf.yield {yield_values} : {result_types}
  }}
{reduce_lines}"""


def generate_word4_rhsvec_mixasm_sum_block(shape, wg, row_symbol, suffix):
    return f"""
  %row_block_base_{suffix} = index.mul %{row_symbol}, %blocks_per_row : index
  %sum_{suffix} = scf.for %block_idx_{suffix} = [%block_slot to %blocks_per_row step %block_step](%acc_{suffix} = %zero_f32 : f32) -> (f32) {{
    %src_base_block_{suffix} = index.mul %block_idx_{suffix}, %qk : index
    %src_base_{suffix} = index.add %src_base_block_{suffix}, %in_block_base : index
    %linear_block_{suffix} = index.add %row_block_base_{suffix}, %block_idx_{suffix} : index
    %block_byte_base_{suffix} = index.mul %linear_block_{suffix}, %block_bytes : index
    %q_byte_payload_{suffix} = index.add %block_byte_base_{suffix}, %qs_offset : index
    %q_byte_{suffix} = index.add %q_byte_payload_{suffix}, %in_block_base : index
    %q_byte_offset_{suffix} = index.cast %q_byte_{suffix} : index to offset
    %q_word_view_{suffix} = buffer.view %src0_noalias[%q_byte_offset_{suffix}] : buffer -> view<1xi32, #dense>
    %q_word_{suffix} = view.load %q_word_view_{suffix}[%zero_index] : view<1xi32, #dense> -> i32
    %src1_index_{suffix} = index.add %src1_col_base, %src_base_{suffix} : index
    %rhs_{suffix} = vector.load %src1_view[%src1_index_{suffix}] : view<{shape.src1_count}xf32, #dense> -> vector<4xf32>
    %block_f16_base_{suffix} = index.mul %linear_block_{suffix}, %block_f16s : index
    %d_f16_{suffix} = view.load %src0_f16_view[%block_f16_base_{suffix}] : view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense> -> f16
    %q0_shifted_{suffix} = scalar.shli %q_word_{suffix}, %shift24 : i32
    %q1_left_{suffix} = scalar.shli %q_word_{suffix}, %shift16 : i32
    %q2_left_{suffix} = scalar.shli %q_word_{suffix}, %shift8 : i32
    %q0_i32_{suffix} = scalar.shrsi %q0_shifted_{suffix}, %shift24 : i32
    %q1_i32_{suffix} = scalar.shrsi %q1_left_{suffix}, %shift24 : i32
    %q2_i32_{suffix} = scalar.shrsi %q2_left_{suffix}, %shift24 : i32
    %q3_i32_{suffix} = scalar.shrsi %q_word_{suffix}, %shift24 : i32
    %q0_f32_{suffix} = scalar.sitofp %q0_i32_{suffix} : i32 to f32
    %q1_f32_{suffix} = scalar.sitofp %q1_i32_{suffix} : i32 to f32
    %q2_f32_{suffix} = scalar.sitofp %q2_i32_{suffix} : i32 to f32
    %q3_f32_{suffix} = scalar.sitofp %q3_i32_{suffix} : i32 to f32
    %s0_{suffix} = llvmir.inline_asm "v_fma_mix_f32 $0, $1, $2, 0 op_sel_hi:[1,0,0]", "=v,v,v"(%d_f16_{suffix}, %q0_f32_{suffix}) : (f16, f32) -> f32
    %s1_{suffix} = llvmir.inline_asm "v_fma_mix_f32 $0, $1, $2, 0 op_sel_hi:[1,0,0]", "=v,v,v"(%d_f16_{suffix}, %q1_f32_{suffix}) : (f16, f32) -> f32
    %s2_{suffix} = llvmir.inline_asm "v_fma_mix_f32 $0, $1, $2, 0 op_sel_hi:[1,0,0]", "=v,v,v"(%d_f16_{suffix}, %q2_f32_{suffix}) : (f16, f32) -> f32
    %s3_{suffix} = llvmir.inline_asm "v_fma_mix_f32 $0, $1, $2, 0 op_sel_hi:[1,0,0]", "=v,v,v"(%d_f16_{suffix}, %q3_f32_{suffix}) : (f16, f32) -> f32
    %qvec_{suffix} = vector.from_elements %s0_{suffix}, %s1_{suffix}, %s2_{suffix}, %s3_{suffix} : vector<4xf32>
    %next_{suffix} = vector.dotf %qvec_{suffix}, %rhs_{suffix}, %acc_{suffix} : vector<4xf32>, vector<4xf32>, f32
    scf.yield %next_{suffix} : f32
  }}
  %dot_{suffix} = kernel.workgroup.reduce<addf> %sum_{suffix} : f32
  %dst_index_{suffix} = index.add %dst_col_base, %{row_symbol} : index"""


def generate_source(candidate):
    shape = candidate.shape
    wg = candidate.workgroup_size
    rpg = candidate.rows_per_workgroup
    cpg = candidate.cols_per_workgroup
    row_groups = shape.rows // rpg
    col_groups = shape.cols // cpg
    if candidate.algorithm not in {"scalar", "block4", "block4_rhsvec", "block4_rhsvec_dotf", "word4_rhsvec", "word4_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf", "word4_bitunpack_scalefirst_rhsvec_dotf", "word4_bitunpack_unrolled_rhsvec_dotf", "word4_bitunpack_scfunroll_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf_subgroup", "word4_rhsvec_mixasm_hi", "chunk4"}:
        raise ValueError(f"unknown algorithm: {candidate.algorithm}")
    row_defs = []
    zero_acc_consts = []
    sum_blocks = []
    stores = []
    for row_offset in range(rpg):
        suffix = str(row_offset)
        if cpg == 1:
            zero_acc_consts.append(f"  %zero_f32_{suffix} = scalar.constant 0.0 : f32")
        else:
            for col_offset in range(cpg):
                zero_acc_consts.append(f"  %zero_f32_{suffix}_{col_offset} = scalar.constant 0.0 : f32")
        if row_offset == 0:
            row_defs.append(f"  %row_0 = index.assume %row_base [range(%row_base, 0, {shape.rows - 1})] : index")
        else:
            row_defs.append(f"  %row_{suffix}_raw = index.add %row_base, %row_offset_{suffix} : index")
            row_defs.append(f"  %row_{suffix} = index.assume %row_{suffix}_raw [range(%row_{suffix}_raw, 0, {shape.rows - 1})] : index")
        if cpg != 1 and candidate.algorithm not in {"word4_bitunpack_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf_subgroup"}:
            raise ValueError(f"cols_per_workgroup={cpg} is only supported for word4_bitunpack_rhsvec_dotf variants")
        if candidate.algorithm == "scalar":
            sum_block = generate_scalar_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "block4":
            sum_block = generate_block4_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "block4_rhsvec":
            sum_block = generate_block4_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "block4_rhsvec_dotf":
            sum_block = generate_block4_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix, use_dotf=True)
        elif candidate.algorithm == "word4_rhsvec":
            sum_block = generate_word4_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "word4_rhsvec_dotf":
            sum_block = generate_word4_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix, use_dotf=True)
        elif candidate.algorithm == "word4_bitunpack_rhsvec_dotf":
            sum_block = generate_word4_bitunpack_rhsvec_multi_col_sum_block(shape, wg, f"row_{suffix}", suffix, cpg)
        elif candidate.algorithm == "word4_bitunpack_scalefirst_rhsvec_dotf":
            sum_block = generate_word4_bitunpack_scalefirst_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "word4_bitunpack_unrolled_rhsvec_dotf":
            sum_block = generate_word4_bitunpack_unrolled_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "word4_bitunpack_scfunroll_rhsvec_dotf":
            sum_block = generate_word4_bitunpack_scfunroll_rhsvec_sum_block(shape, wg, f"row_{suffix}", suffix)
        elif candidate.algorithm == "word4_bitunpack_rhsvec_dotf_subgroup":
            sum_block = generate_word4_bitunpack_rhsvec_multi_col_sum_block(
                shape, wg, f"row_{suffix}", suffix, cpg, reduce_op="subgroup")
        elif candidate.algorithm == "word4_rhsvec_mixasm_hi":
            sum_block = generate_word4_rhsvec_mixasm_sum_block(shape, wg, f"row_{suffix}", suffix)
        else:
            sum_block = generate_chunk4_sum_block(shape, wg, f"row_{suffix}", suffix)
        if cpg == 1:
            sum_block = sum_block.replace(f"%acc_{suffix} = %zero_f32 : f32", f"%acc_{suffix} = %zero_f32_{suffix} : f32")
        sum_blocks.append(sum_block)
        if cpg == 1:
            stores.append(f"    view.store %dot_{suffix}, %dst_view[%dst_index_{suffix}] : f32, view<{shape.dst_count}xf32, #dense>")
        else:
            for col_offset in range(cpg):
                stores.append(
                    f"    %dst_col_{col_offset}_{suffix} = index.add %col_group_base, %col_offset_{col_offset} : index\n"
                    f"    %dst_col_base_{col_offset}_{suffix} = index.mul %dst_col_{col_offset}_{suffix}, %rows : index\n"
                    f"    %dst_index_{col_offset}_{suffix} = index.add %dst_col_base_{col_offset}_{suffix}, %row_{suffix} : index\n"
                    f"    view.store %dot{col_offset}_{suffix}, %dst_view[%dst_index_{col_offset}_{suffix}] : f32, view<{shape.dst_count}xf32, #dense>"
                )

    row_offset_consts = "\n".join(
        f"  %row_offset_{i} = index.constant {i} : index"
        for i in range(1, rpg)
    )
    col_offset_consts = "\n".join(
        f"  %col_offset_{i} = index.constant {i} : index"
        for i in range(cpg)
    )
    config_defs = ""
    config_gets = ""
    if candidate.algorithm == "word4_bitunpack_scfunroll_rhsvec_dotf":
        block_step = wg // 8
        if shape.blocks_per_row % block_step != 0:
            raise ValueError("scf-unrolled Q8 path requires blocks_per_row divisible by WG/8")
        config_defs = f"config.def @hrx2.tuning.q8_0_f32.unroll_factor = {shape.blocks_per_row // block_step} : index\n\n"
        config_gets = "  %q8_unroll_factor = config.get @hrx2.tuning.q8_0_f32.unroll_factor : index\n"
    return f"""{config_defs}kernel.def @q8_0_f32_candidate() {{
  %unit = index.constant 1 : index
  %row_groups = index.constant {row_groups} : index
  %col_groups = index.constant {col_groups} : index
  %workgroup_size = index.constant {wg} : index
  kernel.launch.config workgroups(%row_groups, %col_groups, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
}} launch(%src0: buffer, %src1: buffer, %dst: buffer) {{
  %base = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %qk = index.constant 32 : index
  %block_bytes = index.constant 34 : index
  %block_f16s = index.constant 17 : index
  %qs_offset = index.constant 2 : index
  %k = index.constant {shape.k} : index
  %rows = index.constant {shape.rows} : index
  %row_group_size = index.constant {rpg} : index
  %blocks_per_row = index.constant {shape.blocks_per_row} : index
  %workgroup_size = index.constant {wg} : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %four = index.constant 4 : index
  %eight = index.constant 8 : index
{config_gets.rstrip()}
{row_offset_consts}
{col_offset_consts}
  %row_group0 = kernel.workgroup.id<x> : index
  %row_group = index.assume %row_group0 [range(%row_group0, 0, {row_groups - 1})] : index
  %col_group0 = kernel.workgroup.id<y> : index
  %col_group = index.assume %col_group0 [range(%col_group0, 0, {col_groups - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {wg - 1})] : index
  %zero_f32 = scalar.constant 0.0 : f32
{chr(10).join(zero_acc_consts)}
  %shift8 = scalar.constant 8 : i32
  %shift16 = scalar.constant 16 : i32
  %shift24 = scalar.constant 24 : i32
  %row_base_raw = index.mul %row_group, %row_group_size : index
  %row_base = index.assume %row_base_raw [range(%row_base_raw, 0, {shape.rows - 1})] : index
  %cols_per_group = index.constant {cpg} : index
  %col_group_base = index.mul %col_group, %cols_per_group : index
  %col = index.assume %col_group_base [range(%col_group_base, 0, {shape.cols - 1})] : index
{chr(10).join(row_defs)}
  %block_lane = index.rem %lane, %eight : index
  %block_slot = index.div %lane, %eight : index
  %in_block_base = index.mul %block_lane, %four : index
  %block_step = index.div %workgroup_size, %eight : index

  %src0_global = buffer.assume.memory_space<global> %src0 : buffer
  %src1_global = buffer.assume.memory_space<global> %src1 : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src0_noalias, %src1_noalias, %dst_noalias = buffer.assume.noalias %src0_global, %src1_global, %dst_global : buffer, buffer, buffer
  %src0_i8_view = buffer.view %src0_noalias[%base] : buffer -> view<{shape.src0_bytes}xi8, #dense>
  %src0_f16_view = buffer.view %src0_noalias[%base] : buffer -> view<{shape.rows * shape.blocks_per_row * 17}xf16, #dense>
  %src1_view = buffer.view %src1_noalias[%base] : buffer -> view<{shape.src1_count}xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<{shape.dst_count}xf32, #dense>
  %src1_col_base = index.mul %col, %k : index
  %dst_col_base = index.mul %col, %rows : index
{chr(10).join(sum_blocks)}

  %lane_i32 = index.cast %lane : index to i32
  %zero_i32 = scalar.constant 0 : i32
  %is_lane0 = scalar.cmpi eq, %lane_i32, %zero_i32 : i32
  scf.if %is_lane0 {{
{chr(10).join(stores)}
  }}
  kernel.return
}}

check.case @case_{candidate.id} {{
  %src0 = check.file.read.npy path("../fixtures/q8_src0_{shape.id}.npy") : tensor<{shape.src0_bytes}xi8>
  %src1 = check.file.read.npy path("../fixtures/q8_src1_{shape.id}.npy") : tensor<{shape.src1_count}xf32>
  %dst = check.generate.fill value(0.0) : tensor<{shape.dst_count}xf32>
  %expected = check.file.read.npy path("../fixtures/q8_expected_{shape.id}.npy") : tensor<{shape.dst_count}xf32>
  func.call @q8_0_f32_candidate(%src0, %src1, %dst) : (tensor<{shape.src0_bytes}xi8>, tensor<{shape.src1_count}xf32>, tensor<{shape.dst_count}xf32>)
  check.expect.close actual(%dst) expected(%expected) atol(0.001) rtol(0.001) nan(same) : tensor<{shape.dst_count}xf32>
  check.return
}}

check.benchmark<@case_{candidate.id}> @{candidate.benchmark}
"""


def enumerate_candidates(shapes, workgroup_sizes, rows_per_workgroup, cols_per_workgroup, algorithms):
    valid_algorithms = {"scalar", "block4", "block4_rhsvec", "block4_rhsvec_dotf", "word4_rhsvec", "word4_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf", "word4_bitunpack_scalefirst_rhsvec_dotf", "word4_bitunpack_unrolled_rhsvec_dotf", "word4_bitunpack_scfunroll_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf_subgroup", "word4_rhsvec_mixasm_hi", "chunk4"}
    unknown = sorted(set(algorithms) - valid_algorithms)
    if unknown:
        raise ValueError(f"unknown algorithm(s): {', '.join(unknown)}")
    candidates = []
    for shape, wg, rpg, cpg, algorithm in itertools.product(shapes, workgroup_sizes, rows_per_workgroup, cols_per_workgroup, algorithms):
        if shape.rows % rpg != 0:
            continue
        if shape.cols % cpg != 0:
            continue
        if wg < 32 or wg > 1024:
            continue
        if cpg != 1 and algorithm not in {"word4_bitunpack_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf_subgroup"}:
            continue
        if algorithm in {"block4", "block4_rhsvec", "block4_rhsvec_dotf", "word4_rhsvec", "word4_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf", "word4_bitunpack_scalefirst_rhsvec_dotf", "word4_bitunpack_unrolled_rhsvec_dotf", "word4_bitunpack_scfunroll_rhsvec_dotf", "word4_bitunpack_rhsvec_dotf_subgroup", "word4_rhsvec_mixasm_hi", "chunk4"} and wg % 8 != 0:
            continue
        if algorithm == "word4_bitunpack_rhsvec_dotf_subgroup" and wg != 32:
            continue
        candidates.append(Candidate(shape, wg, rpg, cpg, algorithm))
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


def extract_rows(jsonl_path):
    rows = read_jsonl(jsonl_path)
    plan = next((row for row in rows if row.get("row") == "plan"), None)
    compile_row = next((row for row in rows if row.get("row") == "compile"), None)
    benchmark_row = next((row for row in rows if row.get("row") == "benchmark"), None)
    profile_rows = [row for row in rows if row.get("row") == "profile"]
    profile_summary_rows = [row for row in rows if row.get("row") == "profile_summary"]
    summary = next((row for row in rows if row.get("row") == "summary"), None)
    failures = [row for row in rows if row.get("row") == "failure"]
    return plan, compile_row, benchmark_row, profile_rows, profile_summary_rows, summary, failures


def summarize_compile(compile_row):
    if not compile_row:
        return {}
    static = compile_row.get("static_summary") or (compile_row.get("compile_report") or {})
    report = compile_row.get("compile_report") or {}
    allocation = report.get("allocation") or {}
    memory = report.get("memory") or {}
    schedule = report.get("schedule") or {}
    mix = report.get("static_instruction_mix") or {}
    emission = report.get("emission") or {}
    return {
        "status": compile_row.get("status"),
        "artifact_size": static.get("artifact_size"),
        "target_key": static.get("target_key") or report.get("target_key"),
        "target_bundle": static.get("target_bundle") or report.get("target_bundle"),
        "instruction_count": static.get("instruction_count") or emission.get("instruction_count"),
        "code_byte_count": static.get("code_byte_count") or emission.get("code_byte_count"),
        "private_memory_bytes": static.get("private_memory_bytes") if "private_memory_bytes" in static else memory.get("private_bytes"),
        "local_memory_bytes": static.get("local_memory_bytes") if "local_memory_bytes" in static else memory.get("local_bytes"),
        "allocation_spill_count": static.get("allocation_spill_count") if "allocation_spill_count" in static else allocation.get("spill_count"),
        "allocation_spill_plan_count": static.get("allocation_spill_plan_count") if "allocation_spill_plan_count" in static else allocation.get("spill_plan_count"),
        "register_pressure_peak_live_units": static.get("register_pressure_peak_live_units") or schedule.get("register_pressure_peak_live_units"),
        "vector_alu_count": static.get("vector_alu_count") if "vector_alu_count" in static else mix.get("vector_alu_count"),
        "scalar_alu_count": static.get("scalar_alu_count") if "scalar_alu_count" in static else mix.get("scalar_alu_count"),
        "global_memory_count": static.get("global_memory_count") if "global_memory_count" in static else mix.get("global_memory_count"),
        "unknown_descriptor_count": static.get("unknown_descriptor_count") if "unknown_descriptor_count" in static else mix.get("unknown_count"),
        "compile_report_path": compile_row.get("compile_report_path"),
        "target_listing_path": compile_row.get("target_listing_path"),
        "target_artifact_path": compile_row.get("target_artifact_path"),
    }


def _extract_device_profile_ns(benchmark_row, profile_rows):
    profile_payloads = []
    if benchmark_row:
        result = benchmark_row.get("benchmark_result") or {}
        if result.get("profile"):
            profile_payloads.append(result["profile"])
    for row in profile_rows:
        if row.get("profile"):
            profile_payloads.append(row["profile"])
    for profile in profile_payloads:
        for prow in profile.get("rows") or []:
            if prow.get("type") in {"dispatch_function", "dispatch_command_operation"}:
                timing = prow.get("timing") or {}
                if timing.get("available") and timing.get("mean_ns") is not None:
                    return {
                        "mean_ns": timing.get("mean_ns"),
                        "min_ns": timing.get("min_ns"),
                        "max_ns": timing.get("max_ns"),
                        "total_ns": timing.get("total_ns"),
                        "function_name": prow.get("function_name"),
                        "row_type": prow.get("type"),
                    }
    return {}


def summarize_benchmark(benchmark_row, profile_rows):
    if not benchmark_row:
        return {}
    result = benchmark_row.get("benchmark_result") or {}
    timing = result.get("operation_timing_ns") or {}
    device = _extract_device_profile_ns(benchmark_row, profile_rows)
    return {
        "status": result.get("status"),
        "dispatch_p50_ns": timing.get("p50"),
        "dispatch_p90_ns": timing.get("p90"),
        "dispatch_mean_ns": timing.get("mean"),
        "device_mean_ns": device.get("mean_ns"),
        "device_min_ns": device.get("min_ns"),
        "device_max_ns": device.get("max_ns"),
        "device_profile_row": device,
        "count": timing.get("count"),
        "measured_dispatch_count": result.get("measured_dispatch_count"),
        "stop_reason": result.get("stop_reason"),
        "sample": benchmark_row.get("sample"),
        "data_cache": result.get("data_cache"),
        "timing_warnings": (result.get("timing_interpretation") or {}).get("warnings") or [],
    }


def summarize_listing(path, algorithm):
    required = ALGORITHM_REQUIRED_SIGNATURES.get(algorithm, [])
    summary = {
        "path": str(path) if path else None,
        "required": required,
        "counts": {key: 0 for key in LISTING_PATTERNS},
        "missing_required": list(required),
    }
    if not path:
        return summary
    path = Path(path)
    if not path.exists():
        return summary
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = {}
    for key, pattern in LISTING_PATTERNS.items():
        counts[key] = len(re.findall(pattern, text))
    summary["counts"] = counts
    summary["missing_required"] = [key for key in required if counts.get(key, 0) == 0]
    return summary


def classify_result(run, compile_result, benchmark_result, listing_result):
    reasons = []
    if run["timed_out"]:
        reasons.append("timeout")
    if run["returncode"] != 0:
        reasons.append("tool_failed")
    if compile_result.get("status") != "ok":
        reasons.append("compile_failed")
    if benchmark_result.get("status") != "ok":
        reasons.append("benchmark_failed")
    if benchmark_result.get("device_mean_ns") is None:
        reasons.append("missing_device_profile")
    if compile_result.get("allocation_spill_count") not in (None, 0):
        reasons.append("spills")
    if compile_result.get("allocation_spill_plan_count") not in (None, 0):
        reasons.append("spill_plan")
    if compile_result.get("private_memory_bytes") not in (None, 0):
        reasons.append("private_memory")
    if compile_result.get("unknown_descriptor_count") not in (None, 0):
        reasons.append("unknown_descriptors")
    for missing in listing_result.get("missing_required") or []:
        reasons.append(f"missing_listing:{missing}")
    if reasons:
        if reasons == ["missing_device_profile"]:
            return "no_device_profile", reasons
        if all(reason.startswith("missing_listing:") for reason in reasons):
            return "rejected_static", reasons
        return "failed", reasons
    return "ok", []


def write_summary(run_dir, results):
    winners = {}
    for result in results:
        shape_id = result["shape_id"]
        timing = result.get("benchmark_result") or {}
        device_ns = timing.get("device_mean_ns")
        if result["status"] != "ok" or device_ns is None:
            continue
        current = winners.get(shape_id)
        if current is None or device_ns < current["benchmark_result"]["device_mean_ns"]:
            winners[shape_id] = result

    summary = {
        "schema": "hrx2-q8-0-f32-tune-summary-v1",
        "run_dir": str(run_dir),
        "result_count": len(results),
        "ok_count": sum(1 for item in results if item["status"] == "ok"),
        "winners": {
            shape_id: {
                "workgroup_size": item["workgroup_size"],
                "rows_per_workgroup": item["rows_per_workgroup"],
                "cols_per_workgroup": item["cols_per_workgroup"],
                "algorithm": item["algorithm"],
                "device_mean_ns": item["benchmark_result"]["device_mean_ns"],
                "dispatch_p50_ns": item["benchmark_result"]["dispatch_p50_ns"],
                "dispatch_p90_ns": item["benchmark_result"]["dispatch_p90_ns"],
                "compile": item["compile_result"],
                "listing": item["listing_result"],
            }
            for shape_id, item in winners.items()
        },
    }
    write_json(run_dir / "summary.json", summary)

    lines = [
        "# HRX2 Q8_0/F32 MUL_MAT Tune",
        "",
        f"- Run: `{run_dir}`",
        f"- Results: {summary['ok_count']}/{summary['result_count']} ok",
        "",
        "| Shape | Algorithm | Rows/WG | Cols/WG | WG | Status | Device ns | Dispatch p50 ns | Inst | Code bytes | Spills | Peak live | Rejection |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in sorted(results, key=lambda row: (row["shape_id"], row["algorithm"], row["rows_per_workgroup"], row["cols_per_workgroup"], row["workgroup_size"])):
        timing = item.get("benchmark_result") or {}
        compile_result = item.get("compile_result") or {}
        lines.append(
            f"| `{item['shape_id']}` | `{item['algorithm']}` | {item['rows_per_workgroup']} | {item['cols_per_workgroup']} | {item['workgroup_size']} | {item['status']} | "
            f"{timing.get('device_mean_ns', '')} | {timing.get('dispatch_p50_ns', '')} | "
            f"{compile_result.get('instruction_count', '')} | {compile_result.get('code_byte_count', '')} | "
            f"{compile_result.get('allocation_spill_count', '')} | "
            f"{compile_result.get('register_pressure_peak_live_units', '')} | "
            f"{', '.join(item.get('rejection_reasons') or [])} |"
        )
    lines.extend(["", "## Winners", ""])
    for shape_id, item in sorted(winners.items()):
        lines.append(
            f"- `{shape_id}`: `{item['algorithm']}` rows/WG {item['rows_per_workgroup']} cols/WG {item['cols_per_workgroup']} "
            f"WG {item['workgroup_size']} at {item['benchmark_result']['device_mean_ns']} ns device mean "
            f"({item['benchmark_result']['dispatch_p50_ns']} ns dispatch p50)"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    args = parse_args()
    shapes = parse_shapes(args.shapes)
    workgroup_sizes = split_csv_ints(args.workgroup_sizes)
    rows_per_workgroup = split_csv_ints(args.rows_per_workgroup)
    cols_per_workgroup = split_csv_ints(args.cols_per_workgroup)
    algorithms = split_csv_strings(args.algorithms)
    candidates = enumerate_candidates(shapes, workgroup_sizes, rows_per_workgroup, cols_per_workgroup, algorithms)
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]

    run_dir = Path(args.out_root) / args.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    variants_dir = run_dir / "variants"
    results_dir = run_dir / "benchmark_jsonl"
    bundles_dir = run_dir / "bundles"
    variants_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    bundles_dir.mkdir(parents=True)
    ensure_fixtures(run_dir, shapes, args.seed)

    manifest = {
        "schema": "hrx2-q8-0-f32-tune-run-v1",
        "run_id": args.run_id,
        "shapes": [{"k": s.k, "rows": s.rows, "cols": s.cols} for s in shapes],
        "workgroup_sizes": workgroup_sizes,
        "rows_per_workgroup": rows_per_workgroup,
        "cols_per_workgroup": cols_per_workgroup,
        "algorithms": algorithms,
        "candidate_count": len(candidates),
        "scoring": "profile dispatch-event device mean ns; dispatch_complete retained as integration overhead evidence",
    }
    write_json(run_dir / "run_manifest.json", manifest)

    if args.dry_run:
        for candidate in candidates:
            print(candidate.id)
        return 0

    results = []
    for candidate in candidates:
        variant_path = variants_dir / f"{candidate.id}.loom"
        variant_path.write_text(generate_source(candidate), encoding="utf-8")
        best_result = None
        for repetition in range(args.repetitions):
            output_path = results_dir / f"{candidate.id}_rep{repetition}.jsonl"
            bundle_dir = bundles_dir / f"{candidate.id}_rep{repetition}"
            cmd = [
                str(args.iree_benchmark_loom),
                str(variant_path),
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
                "--compile-report=details",
                "--profile-final-batch=true",
                "--profile-data=dispatch-events,executable-metadata",
                f"--artifact-bundle-dir={bundle_dir}",
                "--artifact-bundle-policy=full",
                "--output-format=jsonl",
                f"--output={output_path}",
            ]
            run = run_command(cmd, args.timeout)
            plan, compile_row, benchmark_row, profile_rows, profile_summary_rows, summary_row, failures = extract_rows(output_path)
            compile_result = summarize_compile(compile_row)
            benchmark_result = summarize_benchmark(benchmark_row, profile_rows)
            listing_result = summarize_listing(compile_result.get("target_listing_path"), candidate.algorithm)
            status, rejection_reasons = classify_result(run, compile_result, benchmark_result, listing_result)
            result = {
                "schema": "hrx2-q8-0-f32-tune-result-v2",
                "run_id": args.run_id,
                "shape": {"k": candidate.shape.k, "rows": candidate.shape.rows, "cols": candidate.shape.cols},
                "shape_id": candidate.shape.id,
                "candidate_id": candidate.id,
                "variant_source": str(variant_path),
                "workgroup_size": candidate.workgroup_size,
                "rows_per_workgroup": candidate.rows_per_workgroup,
                "cols_per_workgroup": candidate.cols_per_workgroup,
                "algorithm": candidate.algorithm,
                "repetition": repetition,
                "status": status,
                "rejection_reasons": rejection_reasons,
                "command": cmd,
                "returncode": run["returncode"],
                "timed_out": run["timed_out"],
                "stdout": run["stdout"],
                "stderr": run["stderr"],
                "output_path": str(output_path),
                "artifact_bundle_dir": str(bundle_dir),
                "plan": plan,
                "compile_result": compile_result,
                "benchmark_result": benchmark_result,
                "listing_result": listing_result,
                "profile_summary_status": [
                    row.get("profile_summary") for row in profile_summary_rows
                    if (row.get("profile_summary") or {}).get("type") == "profile_summary_status"
                ],
                "summary": summary_row.get("summary") if summary_row else None,
                "failures": failures,
            }
            if best_result is None:
                best_result = result
            elif (
                result["status"] == "ok" and
                (
                    best_result["status"] != "ok"
                    or result["benchmark_result"].get("device_mean_ns", float("inf"))
                    < best_result["benchmark_result"].get("device_mean_ns", float("inf"))
                )
            ):
                best_result = result
        results.append(best_result)
        print(
            f"{candidate.id} status={best_result['status']} "
            f"device={best_result.get('benchmark_result', {}).get('device_mean_ns', '')} "
            f"dispatch_p50={best_result.get('benchmark_result', {}).get('dispatch_p50_ns', '')} "
            f"reasons={','.join(best_result.get('rejection_reasons') or [])}",
            file=sys.stderr,
        )

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    summary = write_summary(run_dir, results)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
