"""Lossless zlib baselines for raw and delta-coded int16 samples."""

from __future__ import annotations

import struct
import zlib

import numpy as np

RAW_MAGIC = b"ZI01"
DELTA_MAGIC = b"ZD01"
RAW_HEADER = struct.Struct("<4sI")
DELTA_HEADER = struct.Struct("<4sIBh")


def _check(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype="<i2")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a nonempty one-dimensional array")
    return x


def encode_raw(samples: np.ndarray, level: int = 9) -> bytes:
    """Compress the original int16 bytes."""
    x = _check(samples)
    return RAW_HEADER.pack(RAW_MAGIC, x.size) + zlib.compress(x.tobytes(), level)


def decode_raw(payload: bytes) -> np.ndarray:
    if len(payload) < RAW_HEADER.size:
        raise ValueError("payload is too short")
    magic, n_samples = RAW_HEADER.unpack_from(payload)
    if magic != RAW_MAGIC:
        raise ValueError("not a zlib int16 payload")
    raw = zlib.decompress(payload[RAW_HEADER.size:])
    if len(raw) != n_samples * 2:
        raise ValueError("decoded size does not match the header")
    return np.frombuffer(raw, dtype="<i2").copy()


def encode_delta(samples: np.ndarray, level: int = 9) -> bytes:
    """Store the first sample, then zlib-compress first differences."""
    x = _check(samples)
    diff = np.diff(x.astype(np.int32))
    width = 2 if np.all((diff >= -32768) & (diff <= 32767)) else 4
    dtype = "<i2" if width == 2 else "<i4"
    body = np.asarray(diff, dtype=dtype).tobytes()
    header = DELTA_HEADER.pack(DELTA_MAGIC, x.size, width, int(x[0]))
    return header + zlib.compress(body, level)


def decode_delta(payload: bytes) -> np.ndarray:
    if len(payload) < DELTA_HEADER.size:
        raise ValueError("payload is too short")
    magic, n_samples, width, first = DELTA_HEADER.unpack_from(payload)
    if magic != DELTA_MAGIC or width not in (2, 4) or n_samples == 0:
        raise ValueError("not a zlib delta payload")

    raw = zlib.decompress(payload[DELTA_HEADER.size:])
    if len(raw) != (n_samples - 1) * width:
        raise ValueError("decoded size does not match the header")
    dtype = "<i2" if width == 2 else "<i4"
    diff = np.frombuffer(raw, dtype=dtype).astype(np.int64)
    out = np.empty(n_samples, dtype=np.int64)
    out[0] = first
    out[1:] = first + np.cumsum(diff)
    if np.any((out < -32768) | (out > 32767)):
        raise OverflowError("decoded samples do not fit in int16")
    return out.astype("<i2")
