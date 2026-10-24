"""Convert acceleration in g to a shared int16 representation."""

from __future__ import annotations

import numpy as np

G_PER_COUNT = 0.001


def to_int16(samples_g: np.ndarray) -> np.ndarray:
    """Round acceleration to millig counts.

    One count is 0.001 g, so rounding adds at most 0.0005 g error.
    The scale is fixed and does not need to be sent in each packet.
    """
    counts = np.rint(np.asarray(samples_g) / G_PER_COUNT)
    info = np.iinfo(np.int16)
    if np.any(counts < info.min) or np.any(counts > info.max):
        raise OverflowError("acceleration does not fit in millig int16")
    return counts.astype("<i2")


def to_g(counts: np.ndarray) -> np.ndarray:
    """Convert millig counts back to float32 acceleration in g."""
    return np.asarray(counts, dtype=np.float32) * G_PER_COUNT
