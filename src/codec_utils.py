"""Small shared helpers for the integer baselines."""

from __future__ import annotations

import numpy as np


def pack_signed(values: np.ndarray) -> tuple[int, bytes]:
    """Use int16 when possible, otherwise int32."""
    x = np.asarray(values, dtype=np.int64)
    if np.all((x >= -32768) & (x <= 32767)):
        return 2, x.astype("<i2").tobytes()
    if np.any((x < np.iinfo(np.int32).min) | (x > np.iinfo(np.int32).max)):
        raise OverflowError("residual does not fit in int32")
    return 4, x.astype("<i4").tobytes()


def unpack_signed(raw: bytes, count: int, width: int) -> np.ndarray:
    if width not in (2, 4) or len(raw) != count * width:
        raise ValueError("residual size does not match the header")
    dtype = "<i2" if width == 2 else "<i4"
    return np.frombuffer(raw, dtype=dtype).astype(np.int64)
