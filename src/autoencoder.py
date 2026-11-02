"""Lossy RPM-aware convolutional autoencoder with an int8 code."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAGIC = b"AE01"
MAGIC_SIDE = b"AE02"
MAGIC_PACK = b"AE03"
BLOCK = 2048
CODE_TIME = 8
GEAR_ORDER = 35
HEADER = struct.Struct("<4sHH")
SIDE = struct.Struct("<Heee")
FS = 12_800


def code_steps(code_time: int) -> int:
    """How many stride-4 layers turn one block into ``code_time`` steps."""
    if code_time <= 0 or BLOCK % code_time:
        raise ValueError(f"code time must divide {BLOCK}")
    factor = BLOCK // code_time
    stages = 0
    while factor > 1:
        if factor % 4:
            raise ValueError(f"code time {code_time} is not reachable by stride 4")
        factor //= 4
        stages += 1
    if stages < 1:
        raise ValueError("code time must be shorter than one block")
    return stages


def _widths(code_channels: int, stages: int) -> list[int]:
    hidden = {4: [16, 32, 32], 3: [16, 32], 2: [32], 1: []}
    if stages not in hidden:
        raise ValueError(f"unsupported code depth {stages}")
    return hidden[stages] + [code_channels]


class Encoder(nn.Module):
    """Map one block and its shaft speed to a short code. This is the edge side."""

    def __init__(self, code_channels: int, code_time: int = CODE_TIME):
        super().__init__()
        widths = _widths(code_channels, code_steps(code_time))
        layers: list[nn.Module] = []
        incoming = 2
        for index, outgoing in enumerate(widths):
            layers.append(nn.Conv1d(incoming, outgoing, 8, stride=4, padding=2))
            if index < len(widths) - 1:
                layers.append(nn.ReLU())
            incoming = outgoing
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        speed = rpm.reshape(-1, 1, 1).expand(-1, 1, x.shape[-1])
        return self.net(torch.cat([x.unsqueeze(1), speed], dim=1))


class Decoder(nn.Module):
    """Rebuild one block from the code and the same shaft speed."""

    def __init__(self, code_channels: int, code_time: int = CODE_TIME):
        super().__init__()
        encoded = _widths(code_channels, code_steps(code_time))
        outputs = list(reversed(encoded[:-1])) + [1]
        layers: list[nn.Module] = []
        incoming = code_channels + 1
        for index, outgoing in enumerate(outputs):
            layers.append(nn.ConvTranspose1d(incoming, outgoing, 8, stride=4, padding=2))
            if index < len(outputs) - 1:
                layers.append(nn.ReLU())
            incoming = outgoing
        self.net = nn.Sequential(*layers)

    def forward(self, code: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        speed = rpm.reshape(-1, 1, 1).expand(-1, 1, code.shape[-1])
        return self.net(torch.cat([code, speed], dim=1)).squeeze(1)


class Autoencoder(nn.Module):
    """Encoder, int8 rounding, and decoder. Weights stay off the packet."""

    def __init__(self, code_channels: int = 8, code_time: int = CODE_TIME):
        super().__init__()
        self.code_channels = code_channels
        self.code_time = code_time
        self.encoder = Encoder(code_channels, code_time)
        self.decoder = Decoder(code_channels, code_time)
        self.register_buffer("scale", torch.tensor(1.0 / 127.0))

    def quantize(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Squash the code to int8. The backward pass skips the rounding."""
        bounded = torch.tanh(code)
        levels = torch.round(bounded / self.scale).clamp(-127, 127)
        rounded = levels * self.scale
        return levels, bounded + (rounded - bounded).detach()

    def forward(self, x: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        _levels, code = self.quantize(self.encoder(x, rpm))
        return self.decoder(code, rpm)


def _tone_basis(rpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine of order 35. Its frequency follows the shaft speed."""
    freq = GEAR_ORDER * float(rpm) / 60.0
    phase = 2.0 * np.pi * freq * np.arange(BLOCK, dtype=np.float64) / FS
    return np.cos(phase), np.sin(phase)


def _fit_order(block: np.ndarray, rpm: float) -> tuple[float, float]:
    cos, sin = _tone_basis(rpm)
    cc = float(np.dot(cos, cos))
    ss = float(np.dot(sin, sin))
    cs = float(np.dot(cos, sin))
    cx = float(np.dot(cos, block))
    sx = float(np.dot(sin, block))
    det = cc * ss - cs * cs
    if abs(det) < 1e-8:
        return 0.0, 0.0
    return (ss * cx - cs * sx) / det, (cc * sx - cs * cx) / det


def _quantized_tone(rpm: float, coeff_cos: float, coeff_sin: float) -> np.ndarray:
    cos, sin = _tone_basis(rpm)
    tone = coeff_cos * cos + coeff_sin * sin
    return tone.astype(np.float32)


def _window_start(center: int, window: int) -> int:
    window = min(int(window), BLOCK)
    start = int(center) - window // 2
    return max(0, min(start, BLOCK - window))


def strip_block(
    block: np.ndarray,
    rpm: float,
    window: int = 1,
) -> tuple[np.ndarray, tuple[int, np.ndarray, float, float]]:
    """Take out the order-35 tone and a short run of samples around the shock.

    The returned numbers are what the packet stores. Adding them back restores
    that tone and those samples. The residual is what the network compresses.
    """
    samples = np.asarray(block, dtype=np.float64)
    coeff_cos, coeff_sin = _fit_order(samples, rpm)
    coeff_cos = float(np.float16(coeff_cos))
    coeff_sin = float(np.float16(coeff_sin))
    residual = samples - _quantized_tone(rpm, coeff_cos, coeff_sin)
    center = int(np.argmax(np.abs(residual)))
    start = _window_start(center, window)
    kept = np.asarray(samples[start:start + window], dtype=np.float16).astype(np.float32)
    residual = residual.astype(np.float32)
    residual[start:start + window] = np.float32(0)
    return residual, (start, kept, coeff_cos, coeff_sin)


def restore_block(
    residual: np.ndarray,
    rpm: float,
    start: int,
    kept: np.ndarray,
    coeff_cos: float,
    coeff_sin: float,
) -> np.ndarray:
    """Put the stored tone and the stored samples back onto a decoded residual."""
    restored = np.asarray(residual, dtype=np.float32) + _quantized_tone(rpm, coeff_cos, coeff_sin)
    begin = int(start)
    restored[begin:begin + len(kept)] = np.asarray(kept, dtype=np.float32)
    return restored


def pack_side(start: int, kept: np.ndarray, coeff_cos: float, coeff_sin: float) -> bytes:
    body = struct.pack("<H", int(start))
    body += b"".join(struct.pack("<e", float(value)) for value in kept)
    body += struct.pack("<ee", float(coeff_cos), float(coeff_sin))
    return body


def unpack_side(blob: bytes, window: int) -> tuple[int, np.ndarray, float, float]:
    start = struct.unpack_from("<H", blob)[0]
    kept = np.array(
        [struct.unpack_from("<e", blob, 2 + 2 * index)[0] for index in range(window)],
        dtype=np.float32,
    )
    coeff_cos, coeff_sin = struct.unpack_from("<ee", blob, 2 + 2 * window)
    return start, kept, float(coeff_cos), float(coeff_sin)


def strip_recording(recording: np.ndarray, rpm: float, window: int = 1) -> np.ndarray:
    """Strip every block of one recording. Training sees this residual."""
    out = np.empty(recording.shape, dtype=np.float32)
    for start in range(0, recording.size, BLOCK):
        residual, _side = strip_block(recording[start:start + BLOCK], rpm, window)
        out[start:start + BLOCK] = residual
    return out


def loss_terms(reconstructed: torch.Tensor, original: torch.Tensor):
    """Waveform error plus spectrum error. The spectrum term is training-only."""
    waveform = nn.functional.mse_loss(reconstructed, original)
    spectrum = nn.functional.mse_loss(
        torch.fft.rfft(reconstructed, dim=-1).abs(),
        torch.fft.rfft(original, dim=-1).abs(),
    )
    return waveform, spectrum


class Codec:
    """Shared model. Each packet contains the int8 code and a short header.

    With side information, the packet also holds the order-35 tone and the
    largest sample. Those are removed before encoding and added back after decoding.
    """

    def __init__(
        self,
        model: Autoencoder,
        mean: float,
        std: float,
        rpm_mean: float,
        rpm_std: float,
        side_info: bool = False,
        window: int = 1,
        compress_code: bool = False,
    ):
        self.model = model
        self.mean = float(mean)
        self.std = float(std)
        self.rpm_mean = float(rpm_mean)
        self.rpm_std = float(rpm_std)
        self.side_info = bool(side_info)
        self.window = int(window)
        self.compress_code = bool(compress_code)
        self.model.eval()

    def encode(self, block: np.ndarray, rpm: float) -> bytes:
        samples = np.asarray(block, dtype=np.float32)
        if samples.shape != (BLOCK,):
            raise ValueError(f"block must be {BLOCK} samples")

        side = b""
        if self.side_info:
            samples, (start, kept, coeff_cos, coeff_sin) = strip_block(samples, rpm, self.window)
            side = pack_side(start, kept, coeff_cos, coeff_sin)

        device = self.model.scale.device
        normalized = ((samples - np.float32(self.mean)) / np.float32(self.std)).astype(np.float32)
        x = torch.from_numpy(normalized).view(1, -1).to(device)
        speed = torch.tensor([(rpm - self.rpm_mean) / self.rpm_std], dtype=torch.float32, device=device)
        with torch.inference_mode():
            levels, _code = self.model.quantize(self.model.encoder(x, speed))
        body = levels.detach().cpu().numpy().astype(np.int8).reshape(-1).tobytes()
        if self.compress_code:
            body = zlib.compress(body, 9)
            magic = MAGIC_PACK
        else:
            magic = MAGIC_SIDE if self.side_info else MAGIC
        header = HEADER.pack(magic, self.model.code_channels, self.model.code_time)
        return header + body + side

    def decode(self, payload: bytes, rpm: float) -> np.ndarray:
        if len(payload) < HEADER.size:
            raise ValueError("not an autoencoder payload")
        magic, channels, length = HEADER.unpack(payload[:HEADER.size])
        if self.compress_code:
            expected = MAGIC_PACK
        else:
            expected = MAGIC_SIDE if self.side_info else MAGIC
        if magic != expected:
            raise ValueError("not an autoencoder payload")
        if channels != self.model.code_channels or length != self.model.code_time:
            raise ValueError("payload shape does not match the model")

        code_bytes = channels * length
        side_bytes = (2 + 2 * self.window + 4) if self.side_info else 0
        if len(payload) < HEADER.size + side_bytes:
            raise ValueError("truncated autoencoder payload")
        middle = payload[HEADER.size:len(payload) - side_bytes if side_bytes else None]
        if self.compress_code:
            try:
                middle = zlib.decompress(middle)
            except zlib.error as exc:
                raise ValueError("truncated autoencoder payload") from exc
        if len(middle) != code_bytes:
            raise ValueError("truncated autoencoder payload")

        body = np.frombuffer(middle, dtype=np.int8)
        device = self.model.scale.device
        code = torch.from_numpy(body.astype(np.float32).reshape(1, channels, length)).to(device)
        speed = torch.tensor([(rpm - self.rpm_mean) / self.rpm_std], dtype=torch.float32, device=device)
        with torch.inference_mode():
            restored = self.model.decoder(code * self.model.scale, speed)
        values = restored.detach().cpu().numpy().reshape(-1)
        values = (values * self.std + self.mean).astype(np.float32)
        if self.side_info:
            start, kept, coeff_cos, coeff_sin = unpack_side(payload[-side_bytes:], self.window)
            values = restore_block(values, rpm, start, kept, coeff_cos, coeff_sin)
        return values


def save_codec(
    path: Path,
    model: Autoencoder,
    mean: float,
    std: float,
    rpm_mean: float,
    rpm_std: float,
    side_info: bool = False,
    window: int = 1,
    compress_code: bool = False,
) -> None:
    """Store the shared weights and the train-set normalization."""
    # Ensure the target directory exists before writing the serialized model.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Saved model parameters needed to rebuild the same encoder/decoder.
            "state": model.state_dict(),
            "code_channels": model.code_channels,
            "code_time": model.code_time,
            # Normalization values used during training/encoding/decoding.
            "mean": float(mean),
            "std": float(std),
            "rpm_mean": float(rpm_mean),
            "rpm_std": float(rpm_std),
            # Codec settings that affect packet format and reconstruction behavior.
            "side_info": bool(side_info),
            "window": int(window),
            "compress_code": bool(compress_code),
        },
        path,
    )


def open_codec(path: Path, device: str | torch.device = "cpu") -> Codec:
    """Load a trained codec onto one device."""
    # Load the saved checkpoint and map tensors to the requested device.
    stored = torch.load(path, map_location=device, weights_only=False)
    # Recreate the architecture with the same code dimensionality as the saved model.
    model = Autoencoder(
        int(stored["code_channels"]),
        int(stored.get("code_time", CODE_TIME)),
    )
    # Restore the trained weights into the new model instance.
    model.load_state_dict(stored["state"])
    model.to(device)
    model.eval()
    # Build the runtime Codec wrapper with saved normalization and packet settings.
    return Codec(
        model,
        stored["mean"],
        stored["std"],
        stored["rpm_mean"],
        stored["rpm_std"],
        side_info=bool(stored.get("side_info", False)),
        window=int(stored.get("window", 1)),
        compress_code=bool(stored.get("compress_code", False)),
    )
