"""Run the classical baselines on one chronological test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

import ar4
import harmonic_rice
import mlp_predictor
import zlib_int
import zlib_raw
from fixed_point import to_g, to_int16
from signals import CACHE, load_arrays, split_by_time

BLOCK = 2048


def measure(
    recording: np.ndarray,
    block_size: int,
    encode,
    decode,
) -> tuple[int, float, float]:
    """Compress one recording in blocks and verify exact recovery."""
    if recording.size % block_size:
        raise ValueError("recording length must be divisible by block size")

    compressed_bytes = 0
    encode_seconds = 0.0
    decode_seconds = 0.0

    for start in range(0, recording.size, block_size):
        block = recording[start:start + block_size]

        before = perf_counter()
        payload = encode(block)
        encode_seconds += perf_counter() - before

        before = perf_counter()
        recovered = decode(payload)
        decode_seconds += perf_counter() - before

        if not np.array_equal(block, recovered):
            raise AssertionError("codec round trip changed the samples")

        compressed_bytes += len(payload)

    return compressed_bytes, encode_seconds, decode_seconds


def run_method(
    recordings: np.ndarray,
    block_size: int,
    encode,
    decode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = []
    encode_times = []
    decode_times = []
    for recording in recordings:
        size, encode_time, decode_time = measure(
            recording,
            block_size,
            encode,
            decode,
        )
        sizes.append(size)
        encode_times.append(encode_time)
        decode_times.append(decode_time)
    return (
        np.asarray(sizes),
        np.asarray(encode_times),
        np.asarray(decode_times),
    )


def print_result(
    name: str,
    sizes: np.ndarray,
    encode_times: np.ndarray,
    decode_times: np.ndarray,
    raw_bytes: int,
    n_samples: int,
) -> None:
    removed = 100.0 * (raw_bytes - sizes) / raw_bytes
    ratio = raw_bytes / sizes
    print(
        f"{name:20s}"
        f"{removed.mean():6.1f} ± {removed.std(ddof=1):4.1f}%"
        f"{ratio.mean():9.2f}x"
        f"{8.0 * sizes.mean() / n_samples:9.2f}"
        f"{1e3 * encode_times.mean():10.2f}"
        f"{1e3 * decode_times.mean():10.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=CACHE)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--level", type=int, default=9)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    args = parser.parse_args()

    years, rpm, samples = load_arrays(args.data)
    train_idx, validation_idx, test_idx = split_by_time(years)
    float32 = samples.astype("<f4")
    counts = to_int16(samples)
    test_float = float32[test_idx]
    test_counts = counts[test_idx]
    n_samples = samples.shape[1]

    quantized = to_g(counts)
    quant_error = quantized - samples
    max_error = float(np.max(np.abs(quant_error)))
    rmse = float(np.sqrt(np.mean(quant_error ** 2)))

    before = perf_counter()
    model = mlp_predictor.train(
        counts[train_idx],
        counts[validation_idx],
        epochs=args.mlp_epochs,
    )
    train_seconds = perf_counter() - before

    float_methods = [
        (
            "zlib(raw float32)",
            lambda x: zlib_raw.encode(x, level=args.level),
            zlib_raw.decode,
        ),
    ]
    int_methods = [
        (
            "zlib(raw int16)",
            lambda x: zlib_int.encode_raw(x, level=args.level),
            zlib_int.decode_raw,
        ),
        (
            "zlib(delta)",
            lambda x: zlib_int.encode_delta(x, level=args.level),
            zlib_int.decode_delta,
        ),
        (
            "zlib(AR4)",
            lambda x: ar4.encode(x, level=args.level),
            ar4.decode,
        ),
        (
            "zlib(MLP)",
            lambda x: mlp_predictor.encode(x, model, level=args.level),
            lambda p: mlp_predictor.decode(p, model),
        ),
        (
            "Rice(harmonic)",
            harmonic_rice.encode,
            harmonic_rice.decode,
        ),
    ]

    print("preliminary chronological split")
    print(
        f"recordings: {len(train_idx)} train, {len(validation_idx)} validation, "
        f"{len(test_idx)} test"
    )
    print(
        f"test period: year {years[test_idx].min():.3f}-{years[test_idx].max():.3f}, "
        f"{rpm[test_idx].min():.0f}-{rpm[test_idx].max():.0f} RPM"
    )
    print(
        f"block: {args.block} samples, {n_samples // args.block} blocks/recording"
    )
    print(
        f"int16 scale: 0.001 g/count, quantization RMSE {rmse:.6f} g, "
        f"max {max_error:.6f} g"
    )
    print(
        f"shared MLP: 8 -> 16 ReLU -> 1, {model.nbytes} bytes, "
        f"trained in {train_seconds:.2f} s"
    )
    print()
    print(
        f"{'method':20s}{'bytes removed':>14s}{'ratio':>9s}"
        f"{'bits/sample':>12s}{'enc ms':>10s}{'dec ms':>10s}"
    )
    print("-" * 75)

    float_raw_bytes = n_samples * 4
    print("float32 reference:")
    for name, encode, decode in float_methods:
        print(f"running {name}...", flush=True)
        result = run_method(
            test_float,
            args.block,
            encode,
            decode,
        )
        print_result(name, *result, float_raw_bytes, n_samples)

    int_raw_bytes = n_samples * 2
    print("millig int16 reference:")
    for name, encode, decode in int_methods:
        print(f"running {name}...", flush=True)
        result = run_method(
            test_counts,
            args.block,
            encode,
            decode,
        )
        print_result(name, *result, int_raw_bytes, n_samples)

    print()
    print("All rows include packet headers and recover their input bytes exactly.")
    print("The int16 rows are exact after the stated millig quantization.")


if __name__ == "__main__":
    main()
