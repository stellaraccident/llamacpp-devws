from __future__ import annotations

from pathlib import Path
from typing import Any


QK_K = 256
Q4_K_BLOCK_BYTES = 144
F32_BYTES = 4


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for fixture generation; install numpy in the venv") from exc
    return np


def q4_k_bytes(k: int, rows: int) -> int:
    if k % QK_K != 0:
        raise ValueError(f"k must be a multiple of {QK_K}: {k}")
    return rows * (k // QK_K) * Q4_K_BLOCK_BYTES


def write_q4_k_pattern_npy(path: Path, k: int, rows: int) -> None:
    np = require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = rows * (k // QK_K)
    block_index = np.arange(blocks, dtype=np.uint32).reshape(blocks, 1)
    data = np.empty((blocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
    data[:, 0:2] = np.array([0.5], dtype=np.float16).view(np.uint8)
    data[:, 2:4] = np.array([0.125], dtype=np.float16).view(np.uint8)
    scale_index = np.arange(12, dtype=np.uint32).reshape(1, 12)
    data[:, 4:16] = ((17 + 13 * scale_index + 7 * block_index) & 0x3F).astype(np.uint8)
    state = (block_index * np.uint32(1103515245) + np.uint32(12345)) & np.uint32(0xFFFFFFFF)
    bytes_out = np.empty((blocks, 128), dtype=np.uint8)
    for i in range(128):
        state = (np.uint32(1664525) * state + np.uint32(1013904223)) & np.uint32(0xFFFFFFFF)
        lo = (state >> np.uint32(16)) & np.uint32(0x0F)
        hi = (state >> np.uint32(24)) & np.uint32(0x0F)
        bytes_out[:, i : i + 1] = (lo | (hi << np.uint32(4))).astype(np.uint8)
    data[:, 16:144] = bytes_out
    np.save(path, data.reshape(-1).view(np.int8), allow_pickle=False)


def write_f32_pattern_npy(path: Path, element_count: int, phase: int) -> None:
    np = require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.arange(element_count, dtype=np.uint64)
    values = (((indices * 17 + phase * 29) % 257).astype(np.float32) - np.float32(128.0)) / np.float32(64.0)
    np.save(path, values.astype(np.float32), allow_pickle=False)
