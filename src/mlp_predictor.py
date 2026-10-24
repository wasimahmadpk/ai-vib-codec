"""Shared tiny MLP predictor with exact zlib-coded residuals."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from codec_utils import pack_signed, unpack_signed

MAGIC = b"MLP1"
LAG = 8
HIDDEN = 16
HEADER = struct.Struct("<4sIBB")


@dataclass
class TinyMLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.float32
    scale: np.float32

    def predict(self, history: np.ndarray) -> int:
        x = history.astype(np.float32) / self.scale
        hidden = np.maximum(x @ self.w1 + self.b1, 0.0)
        value = (hidden @ self.w2 + self.b2) * self.scale
        return int(np.rint(value))

    @property
    def nbytes(self) -> int:
        return int(
            self.w1.nbytes
            + self.b1.nbytes
            + self.w2.nbytes
            + self.b2.nbytes
            + self.scale.nbytes
        )


def _windows(recordings: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for recording in recordings:
        view = np.lib.stride_tricks.sliding_window_view(recording, lag + 1)
        xs.append(view[:, :lag])
        ys.append(view[:, lag])
    return np.concatenate(xs), np.concatenate(ys)


def _loss(model: TinyMLP, x: np.ndarray, y: np.ndarray) -> float:
    xn = x.astype(np.float32) / model.scale
    yn = y.astype(np.float32) / model.scale
    hidden = np.maximum(xn @ model.w1 + model.b1, 0.0)
    pred = hidden @ model.w2 + model.b2
    return float(np.mean((pred - yn) ** 2))


def train(
    recordings: np.ndarray,
    validation: np.ndarray,
    epochs: int = 30,
    batch_size: int = 2048,
    seed: int = 0,
) -> TinyMLP:
    """Train 8 samples -> 16 ReLU -> 1 on complete training recordings."""
    rng = np.random.default_rng(seed)
    train_x, train_y = _windows(recordings, LAG)
    val_x, val_y = _windows(validation, LAG)

    max_train = 100_000
    if len(train_x) > max_train:
        take = rng.choice(len(train_x), max_train, replace=False)
        train_x, train_y = train_x[take], train_y[take]
    max_val = 50_000
    if len(val_x) > max_val:
        take = rng.choice(len(val_x), max_val, replace=False)
        val_x, val_y = val_x[take], val_y[take]

    scale = np.float32(max(float(np.std(train_y)), 1.0))
    limit = np.sqrt(6.0 / (LAG + HIDDEN))
    model = TinyMLP(
        w1=rng.uniform(-limit, limit, (LAG, HIDDEN)).astype(np.float32),
        b1=np.zeros(HIDDEN, dtype=np.float32),
        w2=rng.uniform(-limit, limit, HIDDEN).astype(np.float32),
        b2=np.float32(0.0),
        scale=scale,
    )

    params = [model.w1, model.b1, model.w2]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    b2_first = np.float32(0.0)
    b2_second = np.float32(0.0)
    step = 0
    best_loss = np.inf
    best = None
    stale = 0

    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = train_x[idx].astype(np.float32) / scale
            y = train_y[idx].astype(np.float32) / scale

            z = x @ model.w1 + model.b1
            hidden = np.maximum(z, 0.0)
            pred = hidden @ model.w2 + model.b2
            grad_pred = 2.0 * (pred - y) / len(idx)

            grad_w2 = hidden.T @ grad_pred
            grad_b2 = np.sum(grad_pred, dtype=np.float32)
            grad_hidden = grad_pred[:, None] * model.w2[None, :]
            grad_z = grad_hidden * (z > 0.0)
            grad_w1 = x.T @ grad_z
            grad_b1 = grad_z.sum(axis=0)
            grads = [grad_w1, grad_b1, grad_w2]

            step += 1
            for p, grad, m, v in zip(params, grads, first, second):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad * grad
                m_hat = m / (1.0 - 0.9 ** step)
                v_hat = v / (1.0 - 0.999 ** step)
                p -= 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)

            b2_first = np.float32(0.9 * b2_first + 0.1 * grad_b2)
            b2_second = np.float32(0.999 * b2_second + 0.001 * grad_b2 * grad_b2)
            m_hat = b2_first / (1.0 - 0.9 ** step)
            v_hat = b2_second / (1.0 - 0.999 ** step)
            model.b2 -= np.float32(0.001 * m_hat / (np.sqrt(v_hat) + 1e-8))

        val_loss = _loss(model, val_x, val_y)
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best = TinyMLP(
                model.w1.copy(),
                model.b1.copy(),
                model.w2.copy(),
                np.float32(model.b2),
                np.float32(model.scale),
            )
            stale = 0
        else:
            stale += 1
            if stale == 5:
                break

    if best is None:
        raise RuntimeError("MLP training did not produce a model")
    return best


def encode(samples: np.ndarray, model: TinyMLP, level: int = 9) -> bytes:
    """Predict each sample from eight previous samples and store the residual."""
    x = np.asarray(samples, dtype="<i2")
    if x.ndim != 1 or x.size <= LAG:
        raise ValueError("samples must be a one-dimensional block longer than eight")

    work = x.astype(np.int64)
    residual = np.empty(x.size - LAG, dtype=np.int64)
    for t in range(LAG, x.size):
        residual[t - LAG] = work[t] - model.predict(work[t - LAG:t])

    width, body = pack_signed(residual)
    header = HEADER.pack(MAGIC, x.size, LAG, width)
    return header + x[:LAG].tobytes() + zlib.compress(body, level)


def decode(payload: bytes, model: TinyMLP) -> np.ndarray:
    if len(payload) < HEADER.size:
        raise ValueError("payload is too short")
    magic, n_samples, lag, width = HEADER.unpack_from(payload)
    if magic != MAGIC or lag != LAG or n_samples <= lag:
        raise ValueError("not an MLP residual payload")

    pos = HEADER.size
    warm_bytes = lag * 2
    warm = np.frombuffer(payload[pos:pos + warm_bytes], dtype="<i2").copy()
    pos += warm_bytes
    residual = unpack_signed(
        zlib.decompress(payload[pos:]),
        n_samples - lag,
        width,
    )

    out = np.empty(n_samples, dtype=np.int64)
    out[:lag] = warm
    for t in range(lag, n_samples):
        out[t] = model.predict(out[t - lag:t]) + residual[t - lag]
    if np.any((out < -32768) | (out > 32767)):
        raise OverflowError("decoded samples do not fit in int16")
    return out.astype("<i2")
