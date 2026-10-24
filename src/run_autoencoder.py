"""Train the lossy autoencoder and measure it on the held-out recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autoencoder import BLOCK, CODE_TIME, Autoencoder, loss_terms, open_codec, save_codec
from signals import load_full, split_by_time


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
    return (
        np.asarray(sizes),
        np.asarray(rmses),
        np.asarray(snrs),
        np.asarray(peaks),
        np.asarray(encode_times),
        np.asarray(decode_times),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    years, rpm, signal = load_full()
    if signal.shape[1] % BLOCK:
        raise ValueError("recording length must be divisible by the block size")
    signal = np.asarray(signal, dtype=np.float32)
    train_idx, val_idx, test_idx = split_by_time(years)

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

    model = Autoencoder(args.code_channels).to(device)
    n_weights = sum(item.numel() for item in model.parameters())
    code_values = args.code_channels * CODE_TIME
    print(f"encoder+decoder {n_weights:,} weights, code {code_values} int8 values")

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
    path = Path(f"data/ae_{code_values}.pt")
    save_codec(path, model, mean, std, rpm_mean, rpm_std)
    codec = open_codec(path)
    print(f"wrote {path}")

    sizes, rmses, snrs, peaks, encode_times, decode_times = measure_test(
        codec, signal, rpm, test_idx,
    )
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
    print(f"encode             {1e3 * encode_times.mean():6.2f} ms")
    print(f"decode             {1e3 * decode_times.mean():6.2f} ms")


if __name__ == "__main__":
    main()
