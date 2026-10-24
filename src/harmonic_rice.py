"""Lossless harmonic prediction followed by Golomb-Rice residual coding."""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"HR01"
N_TONES = 8
RICE_BLOCK = 256
QMAX = 8
HEADER = struct.Struct("<4sIBf")
TONE = struct.Struct("<fff")


class BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.value = 0
        self.used = 0

    def write(self, value: int, width: int) -> None:
        for bit in range(width - 1, -1, -1):
            self.value = (self.value << 1) | ((value >> bit) & 1)
            self.used += 1
            if self.used == 8:
                self.data.append(self.value)
                self.value = 0
                self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.data.append(self.value << (8 - self.used))
        return bytes(self.data)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.value = 0
        self.left = 0

    def read(self, width: int) -> int:
        value = 0
        for _ in range(width):
            if self.left == 0:
                if self.pos == len(self.data):
                    raise ValueError("truncated Rice payload")
                self.value = self.data[self.pos]
                self.pos += 1
                self.left = 8
            value = (value << 1) | ((self.value >> 7) & 1)
            self.value = (self.value << 1) & 0xFF
            self.left -= 1
        return value


def _zigzag(value: int) -> int:
    return value << 1 if value >= 0 else ((-value) << 1) - 1


def _unzigzag(value: int) -> int:
    return -(value + 1) // 2 if value & 1 else value // 2


def _signed_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def _read_signed(reader: BitReader, width: int) -> int:
    value = reader.read(width)
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def _write_rice(writer: BitWriter, value: int, k: int) -> None:
    unsigned = _zigzag(value)
    quotient = unsigned >> k
    if quotient >= QMAX:
        writer.write((1 << QMAX) - 1, QMAX)
        writer.write(0, 1)
        fits16 = -32768 <= value <= 32767
        writer.write(0 if fits16 else 1, 1)
        width = 16 if fits16 else 32
        writer.write(_signed_bits(value, width), width)
        return

    if quotient:
        writer.write((1 << quotient) - 1, quotient)
    writer.write(0, 1)
    writer.write(unsigned & ((1 << k) - 1), k)


def _read_rice(reader: BitReader, k: int) -> int:
    quotient = 0
    while quotient < QMAX and reader.read(1):
        quotient += 1
    if quotient == QMAX:
        if reader.read(1):
            raise ValueError("bad Rice escape")
        width = 32 if reader.read(1) else 16
        return _read_signed(reader, width)
    remainder = reader.read(k) if k else 0
    return _unzigzag((quotient << k) | remainder)


def _rice_cost(values: np.ndarray, k: int) -> int:
    x = values.astype(np.int64)
    unsigned = np.where(x >= 0, x << 1, ((-x) << 1) - 1)
    quotient = unsigned >> k
    normal = quotient + 1 + k
    escape = np.where(
        (x >= -32768) & (x <= 32767),
        QMAX + 2 + 16,
        QMAX + 2 + 32,
    )
    return int(np.where(quotient >= QMAX, escape, normal).sum())


def _best_k(values: np.ndarray) -> int:
    costs = [_rice_cost(values, k) for k in range(15)]
    return int(np.argmin(costs))


def _fit_tones(samples: np.ndarray) -> tuple[np.float32, np.ndarray]:
    x = samples.astype(np.float64)
    n = len(x)
    magnitude = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    magnitude[0] = 0.0

    chosen = []
    work = magnitude.copy()
    for _ in range(N_TONES):
        i = int(np.argmax(work))
        if i == 0 or work[i] == 0:
            break
        chosen.append(i)
        work[max(0, i - 3):min(len(work), i + 4)] = 0.0

    t = np.arange(n, dtype=np.float64)
    freqs = np.asarray(chosen, dtype=np.float64) / n
    columns = [np.ones(n)]
    for freq in freqs:
        columns.extend([
            np.cos(2.0 * np.pi * freq * t),
            np.sin(2.0 * np.pi * freq * t),
        ])
    coef, *_ = np.linalg.lstsq(np.column_stack(columns), x, rcond=None)
    dc = np.float32(coef[0])
    tones = np.array([
        (np.float32(freq), np.float32(coef[1 + 2 * i]), np.float32(coef[2 + 2 * i]))
        for i, freq in enumerate(freqs)
    ], dtype=np.float32)
    return dc, tones


def _synthesize(n_samples: int, dc: float, tones: np.ndarray) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64)
    out = np.full(n_samples, np.float64(np.float32(dc)))
    for freq, cosine, sine in tones:
        angle = 2.0 * np.pi * np.float64(freq) * t
        out += np.float64(cosine) * np.cos(angle)
        out += np.float64(sine) * np.sin(angle)
    return out


def encode(samples: np.ndarray) -> bytes:
    """Store eight tones and Rice-code the exact remaining integer residual."""
    x = np.asarray(samples, dtype="<i2")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a nonempty one-dimensional array")

    dc, tones = _fit_tones(x)
    prediction = np.rint(_synthesize(x.size, dc, tones)).astype(np.int64)
    residual = x.astype(np.int64) - prediction

    writer = BitWriter()
    for start in range(0, x.size, RICE_BLOCK):
        block = residual[start:start + RICE_BLOCK]
        k = _best_k(block)
        writer.write(k, 4)
        for value in block:
            _write_rice(writer, int(value), k)

    header = HEADER.pack(MAGIC, x.size, len(tones), dc)
    tone_bytes = b"".join(TONE.pack(*tone) for tone in tones)
    return header + tone_bytes + writer.finish()


def decode(payload: bytes) -> np.ndarray:
    if len(payload) < HEADER.size:
        raise ValueError("payload is too short")
    magic, n_samples, n_tones, dc = HEADER.unpack_from(payload)
    if magic != MAGIC or n_samples == 0:
        raise ValueError("not a harmonic Rice payload")

    pos = HEADER.size
    tones = []
    for _ in range(n_tones):
        if pos + TONE.size > len(payload):
            raise ValueError("truncated tone parameters")
        tones.append(TONE.unpack_from(payload, pos))
        pos += TONE.size
    tone_array = np.asarray(tones, dtype=np.float32).reshape(-1, 3)

    reader = BitReader(payload[pos:])
    residual = np.empty(n_samples, dtype=np.int64)
    for start in range(0, n_samples, RICE_BLOCK):
        stop = min(n_samples, start + RICE_BLOCK)
        k = reader.read(4)
        for i in range(start, stop):
            residual[i] = _read_rice(reader, k)

    prediction = np.rint(_synthesize(n_samples, dc, tone_array)).astype(np.int64)
    out = prediction + residual
    if np.any((out < -32768) | (out > 32767)):
        raise OverflowError("decoded samples do not fit in int16")
    return out.astype("<i2")
