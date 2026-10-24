"""Lossy RPM-aware convolutional autoencoder with an int8 code."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAGIC = b"AE01"
BLOCK = 2048
CODE_TIME = 8
HEADER = struct.Struct("<4sHH")


class Encoder(nn.Module):
    """Map one block and its shaft speed to a short code. This is the edge side."""

    def __init__(self, code_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, code_channels, 8, stride=4, padding=2),
        )

    def forward(self, x: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        speed = rpm.reshape(-1, 1, 1).expand(-1, 1, x.shape[-1])
        return self.net(torch.cat([x.unsqueeze(1), speed], dim=1))


class Decoder(nn.Module):
    """Rebuild one block from the code and the same shaft speed."""

    def __init__(self, code_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(code_channels + 1, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 8, stride=4, padding=2),
        )

    def forward(self, code: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        speed = rpm.reshape(-1, 1, 1).expand(-1, 1, code.shape[-1])
        return self.net(torch.cat([code, speed], dim=1)).squeeze(1)


class Autoencoder(nn.Module):
    """Encoder, int8 rounding, and decoder. Weights stay off the packet."""

    def __init__(self, code_channels: int = 8):
        super().__init__()
        self.code_channels = code_channels
        self.encoder = Encoder(code_channels)
        self.decoder = Decoder(code_channels)
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


def loss_terms(reconstructed: torch.Tensor, original: torch.Tensor):
    """Waveform error plus spectrum error. The spectrum term is training-only."""
    waveform = nn.functional.mse_loss(reconstructed, original)
    spectrum = nn.functional.mse_loss(
        torch.fft.rfft(reconstructed, dim=-1).abs(),
        torch.fft.rfft(original, dim=-1).abs(),
    )
    return waveform, spectrum


class Codec:
    """Shared model. Each packet contains the int8 code and a short header."""

    def __init__(
        self,
        model: Autoencoder,
        mean: float,
        std: float,
        rpm_mean: float,
        rpm_std: float,
    ):
        self.model = model
        self.mean = float(mean)
        self.std = float(std)
        self.rpm_mean = float(rpm_mean)
        self.rpm_std = float(rpm_std)
        self.model.eval()

    def encode(self, block: np.ndarray, rpm: float) -> bytes:
        samples = np.asarray(block, dtype=np.float32)
        if samples.shape != (BLOCK,):
            raise ValueError(f"block must be {BLOCK} samples")

        device = self.model.scale.device
        normalized = ((samples - np.float32(self.mean)) / np.float32(self.std)).astype(np.float32)
        x = torch.from_numpy(normalized).view(1, -1).to(device)
        speed = torch.tensor([(rpm - self.rpm_mean) / self.rpm_std], dtype=torch.float32, device=device)
        with torch.inference_mode():
            levels, _code = self.model.quantize(self.model.encoder(x, speed))
        body = levels.detach().cpu().numpy().astype(np.int8).reshape(-1)
        header = HEADER.pack(MAGIC, self.model.code_channels, CODE_TIME)
        return header + body.tobytes()

    def decode(self, payload: bytes, rpm: float) -> np.ndarray:
        if len(payload) < HEADER.size:
            raise ValueError("not an autoencoder payload")
        magic, channels, length = HEADER.unpack(payload[:HEADER.size])
        if magic != MAGIC:
            raise ValueError("not an autoencoder payload")
        if channels != self.model.code_channels or length != CODE_TIME:
            raise ValueError("payload shape does not match the model")

        body = np.frombuffer(payload[HEADER.size:], dtype=np.int8)
        if body.size != channels * length:
            raise ValueError("truncated autoencoder payload")

        device = self.model.scale.device
        code = torch.from_numpy(body.astype(np.float32).reshape(1, channels, length)).to(device)
        speed = torch.tensor([(rpm - self.rpm_mean) / self.rpm_std], dtype=torch.float32, device=device)
        with torch.inference_mode():
            restored = self.model.decoder(code * self.model.scale, speed)
        values = restored.detach().cpu().numpy().reshape(-1)
        return (values * self.std + self.mean).astype(np.float32)


def save_codec(path: Path, model: Autoencoder, mean: float, std: float, rpm_mean: float, rpm_std: float) -> None:
    """Store the shared weights and the train-set normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "code_channels": model.code_channels,
            "mean": float(mean),
            "std": float(std),
            "rpm_mean": float(rpm_mean),
            "rpm_std": float(rpm_std),
        },
        path,
    )


def open_codec(path: Path, device: str | torch.device = "cpu") -> Codec:
    """Load a trained codec onto one device."""
    stored = torch.load(path, map_location=device, weights_only=False)
    model = Autoencoder(int(stored["code_channels"]))
    model.load_state_dict(stored["state"])
    model.to(device)
    model.eval()
    return Codec(
        model,
        stored["mean"],
        stored["std"],
        stored["rpm_mean"],
        stored["rpm_std"],
    )
