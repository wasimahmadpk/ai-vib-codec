"""Lossless AR(4) prediction residuals followed by zlib."""

from __future__ import annotations

import struct
import zlib

import numpy as np

from codec_utils import pack_signed, unpack_signed

MAGIC = b"AR04"
ORDER = 4
HEADER = struct.Struct("<4sIBB")


def fit(samples: np.ndarray, order: int = ORDER) -> np.ndarray:
    """Fit one linear predictor to a block."""
    x = np.asarray(samples, dtype=np.float64)
    rows = np.column_stack([
        x[order - lag - 1:len(x) - lag - 1]
        for lag in range(order)
    ])
    target = x[order:]
    coef, *_ = np.linalg.lstsq(rows, target, rcond=None)
    return coef.astype("<f4")


def _predict(past: np.ndarray, coef: np.ndarray) -> int:
    value = np.dot(coef.astype(np.float64), past[::-1].astype(np.float64))
    return int(np.rint(value))


def encode(samples: np.ndarray, level: int = 9) -> bytes:
    """Fit AR(4), then compress its exact integer residual."""
    x = np.asarray(samples, dtype="<i2")
    if x.ndim != 1 or x.size <= ORDER:
        raise ValueError("samples must be a one-dimensional block longer than four")

    coef = fit(x)
    residual = np.empty(x.size - ORDER, dtype=np.int64)
    work = x.astype(np.int64)
    for t in range(ORDER, x.size):
        residual[t - ORDER] = work[t] - _predict(work[t - ORDER:t], coef)

    width, body = pack_signed(residual)
    header = HEADER.pack(MAGIC, x.size, ORDER, width)
    return header + coef.tobytes() + x[:ORDER].tobytes() + zlib.compress(body, level)


def decode(payload: bytes) -> np.ndarray:
    if len(payload) < HEADER.size:
        raise ValueError("payload is too short")
    magic, n_samples, order, width = HEADER.unpack_from(payload)
    if magic != MAGIC or order != ORDER or n_samples <= order:
        raise ValueError("not an AR4 payload")

    pos = HEADER.size
    coef_bytes = order * 4
    warm_bytes = order * 2
    coef = np.frombuffer(payload[pos:pos + coef_bytes], dtype="<f4").copy()
    pos += coef_bytes
    warm = np.frombuffer(payload[pos:pos + warm_bytes], dtype="<i2").copy()
    pos += warm_bytes
    residual = unpack_signed(
        zlib.decompress(payload[pos:]),
        n_samples - order,
        width,
    )

    out = np.empty(n_samples, dtype=np.int64)
    out[:order] = warm
    for t in range(order, n_samples):
        out[t] = _predict(out[t - order:t], coef) + residual[t - order]
    if np.any((out < -32768) | (out > 32767)):
        raise OverflowError("decoded samples do not fit in int16")
    return out.astype("<i2")
