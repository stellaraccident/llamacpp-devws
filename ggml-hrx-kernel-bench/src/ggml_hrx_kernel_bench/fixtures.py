from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO


QK_K = 256
Q4_K_BLOCK_BYTES = 144
F32_BYTES = 4


def q4_k_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q4_K_BLOCK_BYTES


def write_npy_header(f: BinaryIO, descr: str, shape: tuple[int, ...]) -> None:
    shape_text = "(" + ", ".join(str(x) for x in shape) + ("," if len(shape) == 1 else "") + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    header_bytes = header.encode("latin1")
    pad = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * pad + b"\n"
    f.write(b"\x93NUMPY\x01\x00")
    f.write(struct.pack("<H", len(header_bytes)))
    f.write(header_bytes)


def write_q4_k_pattern_npy(path: Path, k: int, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
