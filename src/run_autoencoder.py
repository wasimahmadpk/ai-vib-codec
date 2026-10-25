"""Train the lossy autoencoder and measure it on the held-out recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autoencoder import BLOCK, GEAR_ORDER, SIDE, Autoencoder, loss_terms, open_codec, save_codec, strip_recording
from signals import load_full, main_peak, split_by_time


class Blocks(Dataset):
    """One normalized 2048-sample block and the shaft speed of its recording."""

    def __init__(self, signal, rpm, indices, mean, std, rpm_mean, rpm_std):
        self.signal = signal
        self.rpm = rpm
        self.indices = np.asarray(indices)
        self.mean = np.float32(mean)
        self.std = np.float32(std)
        self.rpm_mean = np.float32(rpm_mean)
        self.rpm_std = np.float32(rpm_std)
        self.n_blocks = signal.shape[1] // BLOCK

    def __len__(self) -> int:
        return len(self.indices) * self.n_blocks

    def __getitem__(self, item: int):
        recording = int(self.indices[item // self.n_blocks])
        start = (item % self.n_blocks) * BLOCK
        block = self.signal[recording, start:start + BLOCK]
        samples = ((block - self.mean) / self.std).astype(np.float32)
        speed = (self.rpm[recording] - self.rpm_mean) / self.rpm_std
        return (
            torch.from_numpy(np.ascontiguousarray(samples)),
            torch.tensor(speed, dtype=torch.float32),
        )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rmse_g(model: Autoencoder, loader: DataLoader, std: float, device: torch.device) -> float:
    """Root mean square error in g on normalized loader batches."""
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for samples, speed in loader:
            restored = model(samples.to(device), speed.to(device))
            error = (restored - samples.to(device)) * std
            total += float(torch.sum(error * error))
            count += error.numel()
    return float(np.sqrt(total / count))


def train(model, train_loader, val_loader, std, device, epochs, patience):
    """Minimize waveform and spectrum error. Keep the best validation RMSE."""
    samples, speed = next(iter(train_loader))
    samples = samples.to(device)
    speed = speed.to(device)
    model.eval()
    with torch.no_grad():
        waveform, spectrum = loss_terms(model(samples, speed), samples)
    spec_weight = float((waveform / spectrum.clamp(min=1e-8)).detach())
    print(f"spectrum weight {spec_weight:.6f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_rmse = np.inf
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch, speed in train_loader:
            batch = batch.to(device)
            speed = speed.to(device)
            optimizer.zero_grad(set_to_none=True)
            waveform, spectrum = loss_terms(model(batch, speed), batch)
            loss = waveform + spec_weight * spectrum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(waveform.detach()) * len(batch)
            seen += len(batch)

        val_rmse = rmse_g(model, val_loader, std, device)
        print(
            f"epoch {epoch:02d}  train mse {running / seen:.5f}  "
            f"val rmse {val_rmse:.4f} g",
            flush=True,
        )
        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale == patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a model")
    model.load_state_dict(best_state)
    model.to(device)
    return best_rmse, spec_weight


def measure_test(codec, signal, rpm, test_idx):
    """Packet size, error, and CPU time on each held-out recording."""
    sizes = []
    rmses = []
    snrs = []
    peaks = []
    encode_times = []
    decode_times = []
    recovered = []
    rms_error = []
    peak_hz = []
    for index in test_idx:
        recording = np.asarray(signal[index], dtype=np.float32)
        speed = float(rpm[index])
        restored = np.empty_like(recording)
        size = 0
        encode_seconds = 0.0
        decode_seconds = 0.0
        for start in range(0, recording.size, BLOCK):
            block = recording[start:start + BLOCK]
            before = perf_counter()
            payload = codec.encode(block, speed)
            encode_seconds += perf_counter() - before
            before = perf_counter()
            restored[start:start + BLOCK] = codec.decode(payload, speed)
            decode_seconds += perf_counter() - before
            size += len(payload)
        error = restored - recording
        power = float(np.mean(recording ** 2))
        noise = float(np.mean(error ** 2))
        sizes.append(size)
        rmses.append(float(np.sqrt(noise)))
        snrs.append(float(10.0 * np.log10(power / noise)))
        peaks.append(float(np.max(np.abs(error))))
        encode_times.append(encode_seconds)
        decode_times.append(decode_seconds)
        recovered.append(100.0 * (1.0 - noise / power))
        original_rms = float(np.sqrt(power))
        restored_rms = float(np.sqrt(np.mean(restored ** 2)))
        rms_error.append(abs(restored_rms - original_rms) / original_rms)
        original_hz, _order = main_peak(recording, speed)
        restored_hz, _order = main_peak(restored, speed)
        peak_hz.append(abs(restored_hz - original_hz))
    return {
        "sizes": np.asarray(sizes),
        "rmses": np.asarray(rmses),
        "snrs": np.asarray(snrs),
        "peaks": np.asarray(peaks),
        "encode_times": np.asarray(encode_times),
        "decode_times": np.asarray(decode_times),
        "recovered": np.asarray(recovered),
        "rms_error": np.asarray(rms_error),
        "peak_hz": np.asarray(peak_hz),
    }


def _recording_scores(original: np.ndarray, restored: np.ndarray, speed: float) -> dict[str, float]:
    error = restored - original
    power = float(np.mean(original ** 2))
    noise = float(np.mean(error ** 2))
    original_hz, _order = main_peak(original, speed)
    restored_hz, _order = main_peak(restored, speed)
    original_rms = float(np.sqrt(power))
    restored_rms = float(np.sqrt(np.mean(restored ** 2)))
    return {
        "rmse": float(np.sqrt(noise)),
        "snr": float(10.0 * np.log10(power / noise)),
        "recovered": 100.0 * (1.0 - noise / power),
        "rms_error": abs(restored_rms - original_rms) / original_rms,
        "peak_hz": abs(restored_hz - original_hz),
    }


def compare_rounding(path: Path, device: torch.device) -> None:
    """Compare the trained code before and after int8 rounding."""
    codec = open_codec(path, device)
    model = codec.model.to(device)
    model.eval()
    years, rpm, signal = load_full()
    signal = np.asarray(signal, dtype=np.float32)
    _train, _validation, test_idx = split_by_time(years)
    n_blocks = signal.shape[1] // BLOCK
    names = ("float code", "int8 code")
    totals = {name: [] for name in names}

    with torch.inference_mode():
        for index in test_idx:
            row = np.asarray(signal[index], dtype=np.float32)
            speed = float(rpm[index])
            normalized = ((row - np.float32(codec.mean)) / np.float32(codec.std)).astype(np.float32)
            samples = torch.from_numpy(normalized.reshape(n_blocks, BLOCK)).to(device)
            rpm_code = np.float32((speed - codec.rpm_mean) / codec.rpm_std)
            shaft = torch.full((n_blocks,), rpm_code, dtype=torch.float32, device=device)
            raw = model.encoder(samples, shaft)
            continuous = torch.tanh(raw)
            _levels, rounded = model.quantize(raw)
            versions = {
                "float code": model.decoder(continuous, shaft),
                "int8 code": model.decoder(rounded, shaft),
            }
            for name, restored in versions.items():
                values = restored.detach().cpu().numpy().reshape(-1)
                values = (values * codec.std + codec.mean).astype(np.float32)
                totals[name].append(_recording_scores(row, values, speed))

    print(f"rounding check on {path}")
    print(f"{len(test_idx)} test recordings, {model.code_channels * model.code_time} code values")
    for name in names:
        rows = totals[name]
        rmse = np.array([item["rmse"] for item in rows])
        snr = np.array([item["snr"] for item in rows])
        recovered = np.array([item["recovered"] for item in rows])
        rms_error = np.array([item["rms_error"] for item in rows])
        peak_hz = np.array([item["peak_hz"] for item in rows])
        print(
            f"{name:12s}  RMSE {rmse.mean():.4f} g  "
            f"SNR {snr.mean():.2f} dB  "
            f"energy {recovered.mean():.1f}%  "
            f"RMS error {100.0 * rms_error.mean():.1f}%  "
            f"main peak {peak_hz.mean():.1f} Hz"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-channels", type=int, default=32)
    parser.add_argument("--code-time", type=int, default=8)
    parser.add_argument("--quant-check", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--side-info",
        action="store_true",
        help="store the order-35 tone and a shock window in each packet",
    )
    parser.add_argument("--shock-window", type=int, default=1)
    parser.add_argument(
        "--pack-code",
        action="store_true",
        help="zlib the int8 code; the samples that come back stay the same",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    if args.quant_check is not None:
        compare_rounding(args.quant_check, device)
        return
    years, rpm, signal = load_full()
    if signal.shape[1] % BLOCK:
        raise ValueError("recording length must be divisible by the block size")
    signal = np.asarray(signal, dtype=np.float32)
    train_idx, val_idx, test_idx = split_by_time(years)
    raw = signal
    if args.side_info:
        signal = raw.copy()
        for index in np.concatenate([train_idx, val_idx]):
            signal[index] = strip_recording(raw[index], float(rpm[index]), args.shock_window)
        print(
            f"side info on: order {GEAR_ORDER} tone and {args.shock_window} shock samples, "
            f"packed code {args.pack_code}"
        )

    mean = float(signal[train_idx].mean())
    std = float(signal[train_idx].std())
    rpm_mean = float(rpm[train_idx].mean())
    rpm_std = float(rpm[train_idx].std())
    if std < 1e-8 or rpm_std < 1e-8:
        raise RuntimeError("training scale is degenerate")

    print(
        f"device {device}  recordings {len(train_idx)} train, "
        f"{len(val_idx)} validation, {len(test_idx)} test"
    )
    print(
        f"test period year {years[test_idx].min():.3f}-{years[test_idx].max():.3f}, "
        f"{rpm[test_idx].min():.0f}-{rpm[test_idx].max():.0f} RPM"
    )
    print(f"train signal mean {mean:.5f} g, std {std:.5f} g")

    train_blocks = Blocks(signal, rpm, train_idx, mean, std, rpm_mean, rpm_std)
    val_blocks = Blocks(signal, rpm, val_idx, mean, std, rpm_mean, rpm_std)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_blocks,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_blocks, batch_size=args.batch_size)

    model = Autoencoder(args.code_channels, args.code_time).to(device)
    n_weights = sum(item.numel() for item in model.parameters())
    encoder_bytes = sum(item.numel() * item.element_size() for item in model.encoder.parameters())
    code_values = args.code_channels * args.code_time
    print(
        f"encoder+decoder {n_weights:,} weights, "
        f"encoder {encoder_bytes / 1024:.1f} KiB, "
        f"code {code_values} int8 values over {args.code_time} steps"
    )

    before = perf_counter()
    best_rmse, _spec_weight = train(
        model,
        train_loader,
        val_loader,
        std,
        device,
        args.epochs,
        args.patience,
    )
    train_seconds = perf_counter() - before
    print(f"best validation RMSE {best_rmse:.4f} g after {train_seconds:.1f} s")

    model.to("cpu")
    suffix = ""
    if args.pack_code:
        suffix = "_pack"
    elif args.side_info:
        suffix = "_side"
    path = Path(f"data/ae_{code_values}{suffix}.pt")
    save_codec(
        path,
        model,
        mean,
        std,
        rpm_mean,
        rpm_std,
        side_info=args.side_info,
        window=args.shock_window,
        compress_code=args.pack_code,
    )
    codec = open_codec(path)
    print(f"wrote {path}")

    measured = measure_test(codec, raw, rpm, test_idx)
    sizes = measured["sizes"]
    rmses = measured["rmses"]
    snrs = measured["snrs"]
    peaks = measured["peaks"]
    encode_times = measured["encode_times"]
    decode_times = measured["decode_times"]
    recovered = measured["recovered"]
    rms_error = measured["rms_error"]
    peak_hz = measured["peak_hz"]
    n_samples = signal.shape[1]
    raw_int16 = n_samples * 2
    raw_float32 = n_samples * 4
    print()
    print(
        f"code {code_values} int8 values, {sizes[0] // (n_samples // BLOCK)} bytes/block, "
        f"{int(sizes[0])} bytes/recording"
    )
    print(f"removed vs int16   {100.0 * (raw_int16 - sizes.mean()) / raw_int16:6.2f}%")
    print(f"removed vs float32 {100.0 * (raw_float32 - sizes.mean()) / raw_float32:6.2f}%")
    print(f"bits/sample        {8.0 * sizes.mean() / n_samples:6.3f}")
    print(f"RMSE               {rmses.mean():6.4f} ± {rmses.std(ddof=1):.4f} g")
    print(f"SNR                {snrs.mean():6.2f} ± {snrs.std(ddof=1):.2f} dB")
    print(f"peak error         {peaks.mean():6.3f} ± {peaks.std(ddof=1):.3f} g")
    print(f"energy recovered   {recovered.mean():6.1f} ± {recovered.std(ddof=1):.1f}%")
    print(f"RMS error          {100.0 * rms_error.mean():6.1f} ± {100.0 * rms_error.std(ddof=1):.1f}%")
    print(f"main-peak error    {peak_hz.mean():6.1f} ± {peak_hz.std(ddof=1):.1f} Hz")
    print(f"encode             {1e3 * encode_times.mean():6.2f} ms")
    print(f"decode             {1e3 * decode_times.mean():6.2f} ms")


if __name__ == "__main__":
    main()
