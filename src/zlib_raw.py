"""Lossless zlib baseline for one block of float32 vibration samples."""

from __future__ import annotations

import struct
import zlib

import numpy as np

MAGIC = b"ZR01"
HEADER = struct.Struct("<4sI")


def encode(samples: np.ndarray, level: int = 9) -> bytes:
    """Store the sample count followed by zlib-compressed float32 bytes."""
    x = np.asarray(samples, dtype="<f4")
    if x.ndim != 1:
        raise ValueError("samples must be one-dimensional")

    raw = np.ascontiguousarray(x).tobytes()
    return HEADER.pack(MAGIC, x.size) + zlib.compress(raw, level)


def decode(payload: bytes) -> np.ndarray:
    """Recover the exact float32 samples from one encoded block."""
    if len(payload) < HEADER.size:
        raise ValueError("payload is too short")

    magic, n_samples = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("not a zlib raw payload")

    raw = zlib.decompress(payload[HEADER.size:])
    expected = n_samples * np.dtype("<f4").itemsize
    if len(raw) != expected:
        raise ValueError("decoded size does not match the header")

    return np.frombuffer(raw, dtype="<f4").copy()
