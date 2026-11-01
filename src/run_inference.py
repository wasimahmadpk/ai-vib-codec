"""Compress one recording with the saved model and report the result.

A turbine recording is a finished file, so the whole file is compressed at
once. The model is loaded, not trained. This is separate from the benchmark,
which scores the test set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from autoencoder import BLOCK, open_codec
from signals import main_peak, split_by_time
from terminal_table import print_table

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("data/dataset02_f32.npz")
CODEC = Path("data/ae_2048_zlib.pt")
REPORT = Path("results/inference.md")


def prepare(samples: np.ndarray, rpm: float) -> tuple[np.ndarray, float]:
    """One recording in g, and a shaft speed, ready to compress."""
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0 or values.size % BLOCK:
        raise ValueError(f"recording must be a multiple of {BLOCK} samples, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("recording has a non-finite sample")
    speed = float(rpm)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("shaft speed must be a positive number")
    return values, speed


def compress(codec, samples: np.ndarray, rpm: float) -> tuple[bytes, np.ndarray, float, float]:
    """Encode every block, then decode it. Return the packet and the rebuild."""
    restored = np.empty_like(samples)
    parts = []
    encode_seconds = 0.0
    decode_seconds = 0.0
    for start in range(0, samples.size, BLOCK):
        block = samples[start:start + BLOCK]
        before = perf_counter()
        payload = codec.encode(block, rpm)
        encode_seconds += perf_counter() - before
        before = perf_counter()
        restored[start:start + BLOCK] = codec.decode(payload, rpm)
        decode_seconds += perf_counter() - before
        parts.append(payload)
    return b"".join(parts), restored, encode_seconds, decode_seconds


def measure(original: np.ndarray, restored: np.ndarray, packet: bytes, rpm: float) -> dict[str, float]:
    """Packet size against the int16 file, and the error of the rebuild."""
    error = restored - original
    power = float(np.mean(original ** 2))
    noise = float(np.mean(error ** 2))
    raw_bytes = original.size * 2
    packet_bytes = len(packet)
    original_hz, _order = main_peak(original, rpm)
    restored_hz, _order = main_peak(restored, rpm)
    return {
        "samples": float(original.size),
        "blocks": float(original.size // BLOCK),
        "raw_bytes": float(raw_bytes),
        "packet_bytes": float(packet_bytes),
        "removed": 100.0 * (1.0 - packet_bytes / raw_bytes),
        "energy": 100.0 * (1.0 - noise / power) if power > 0 else 100.0,
        "rmse": float(np.sqrt(noise)),
        "peak": float(np.max(np.abs(error))),
        "tone_hz": abs(restored_hz - original_hz),
    }


def render(source: str, rpm: float, codec_path: Path, stats: dict[str, float], encode_s: float, decode_s: float) -> str:
    """The printed result for one recording."""
    lines = [
        "Inference, one recording",
        "The file is already finished, so the whole recording is compressed at once.",
        "",
        f"Source: {source}",
        f"Shaft speed: {rpm:.2f} rpm",
        f"Codec: {codec_path}",
        f"Samples: {int(stats['samples'])} in {int(stats['blocks'])} blocks",
        "",
        f"Packet: {int(stats['packet_bytes'])} bytes",
        f"int16 file: {int(stats['raw_bytes'])} bytes",
        f"Removed: {stats['removed']:.1f}%",
        f"Energy back: {stats['energy']:.1f}%",
        f"RMSE: {stats['rmse']:.3f} g",
        f"Peak error: {stats['peak']:.3f} g",
        f"Loudest tone: {stats['tone_hz']:.1f} Hz off",
        f"Encode: {encode_s * 1000:.2f} ms",
        f"Decode: {decode_s * 1000:.2f} ms",
        "",
    ]
    return "\n".join(lines)


def replay(data: Path, index: int | None) -> tuple[np.ndarray, float, str]:
    """One row of the dataset. The default row is the first held-out test recording."""
    stored = np.load(data)
    years = np.asarray(stored["years"])
    rpm = np.asarray(stored["rpm"])
    signal = np.asarray(stored["signal"])
    _train, _validation, test_idx = split_by_time(years)
    row = int(test_idx[0] if index is None else index)
    if row < 0 or row >= len(signal):
        raise ValueError(f"recording index {row} is outside 0..{len(signal) - 1}")
    label = "first held-out test recording" if index is None else "dataset row"
    source = f"{label} {row} in {data}"
    return signal[row], float(rpm[row]), source


def incoming(path: Path, rpm: float) -> tuple[np.ndarray, float, str]:
    """A new recording stored as a float .npy file, plus its shaft speed."""
    if rpm is None:
        raise ValueError("a new recording needs --rpm")
    samples = np.load(path)
    return samples, rpm, str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress one recording with the saved model.")
    parser.add_argument("--codec", type=Path, default=CODEC)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--index", type=int, default=None, help="dataset row to replay")
    parser.add_argument("--signal", type=Path, default=None, help="new recording as a float .npy file")
    parser.add_argument("--rpm", type=float, default=None, help="shaft speed for --signal")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    os.chdir(ROOT)

    if args.signal is not None and args.index is not None:
        raise SystemExit("pass a dataset row or a new file, not both")
    if not args.codec.is_file():
        raise SystemExit(f"missing codec {args.codec}")

    if args.signal is not None:
        samples, rpm, source = incoming(args.signal, args.rpm)
    else:
        if not args.data.is_file():
            raise SystemExit(f"missing dataset {args.data}")
        samples, rpm, source = replay(args.data, args.index)

    samples, rpm = prepare(samples, rpm)
    codec = open_codec(args.codec)
    packet, restored, encode_s, decode_s = compress(codec, samples, rpm)
    stats = measure(samples, restored, packet, rpm)
    text = render(source, rpm, args.codec, stats, encode_s, decode_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print_table(
        "Inference  ·  one recording",
        ["", ""],
        [
            ["Source", source],
            ["Shaft speed", f"{rpm:.2f} rpm"],
            ["Codec", str(args.codec)],
            ["Samples", f"{int(stats['samples'])} in {int(stats['blocks'])} blocks"],
            ["Packet", f"{int(stats['packet_bytes'])} bytes"],
            ["int16 file", f"{int(stats['raw_bytes'])} bytes"],
            ["Removed", f"{stats['removed']:.1f}%"],
            ["Energy back", f"{stats['energy']:.1f}%"],
            ["RMSE", f"{stats['rmse']:.3f} g"],
            ["Peak error", f"{stats['peak']:.3f} g"],
            ["Loudest tone", f"{stats['tone_hz']:.1f} Hz off"],
            ["Encode", f"{encode_s * 1000:.2f} ms"],
            ["Decode", f"{decode_s * 1000:.2f} ms"],
        ],
        highlight={"Removed", "Energy back"},
        numeric=False,
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
