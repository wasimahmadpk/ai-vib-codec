"""Compress the CWRU slice and test whether the recovered trace still shows the fault.

Each file stays in one split. Healthy 48 kHz traces are resampled to 12 kHz.
The autoencoder is the same network as on the turbine, without the order-35 side
information. A light classifier is trained on the original traces and then scored
on each reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.signal import hilbert, resample
from torch.utils.data import DataLoader, Dataset

import ar4
import harmonic_rice
import mlp_predictor
import zlib_int
import zlib_raw
from autoencoder import BLOCK, Autoencoder, open_codec, save_codec
from cwru import FS_FAULT, RECORDS, fault_orders, load_recording
from fixed_point import to_g, to_int16
from run_autoencoder import pick_device, train
from run_baseline import measure

FS = FS_FAULT
WINDOW = FS
LEVEL = 9
SEED = 0
OUT = Path("results/cwru_numbers.json")

# Whole files only. The test files are unseen visits, with both loads and every fault place.
TRAIN_NAMES = (
    "Normal_0",
    "Normal_1",
    "IR007_0",
    "IR021_3",
    "B007_0",
    "B021_3",
    "OR007@6_0",
    "OR007@3_0",
    "OR007@12_0",
    "OR021@6_3",
)
VAL_NAMES = ("Normal_2", "IR007_3", "B021_0")
TEST_NAMES = ("Normal_3", "IR021_0", "B007_3", "OR007@6_3", "OR021@6_0")

AE_LAYOUTS = (
    (2, 8),
    (8, 8),
    (32, 8),
    (16, 32),
    (16, 128),
)
PLACES = ("normal", "inner", "ball", "outer")


class FlatBlocks(Dataset):
    """Normalized blocks drawn from several recordings of different lengths."""

    def __init__(self, blocks, rpm, mean, std, rpm_mean, rpm_std):
        self.blocks = np.asarray(blocks, dtype=np.float32)
        self.rpm = np.asarray(rpm, dtype=np.float32)
        self.mean = np.float32(mean)
        self.std = np.float32(std)
        self.rpm_mean = np.float32(rpm_mean)
        self.rpm_std = np.float32(rpm_std)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, item: int):
        samples = ((self.blocks[item] - self.mean) / self.std).astype(np.float32)
        speed = (self.rpm[item] - self.rpm_mean) / self.rpm_std
        return torch.from_numpy(np.ascontiguousarray(samples)), torch.tensor(float(speed))


def place_of(fault: str) -> str:
    if fault in PLACES:
        return fault
    return "outer"


def prepare() -> list[dict]:
    """Drive-end trace at 12 kHz, trimmed to whole blocks of 2048."""
    rows = []
    for record in RECORDS:
        loaded = load_recording(record)
        samples = np.asarray(loaded["drive"], dtype=np.float64)
        if loaded["fs"] != FS:
            target = int(np.round(samples.size * FS / loaded["fs"]))
            samples = resample(samples, target)
        usable = (samples.size // BLOCK) * BLOCK
        samples = np.asarray(samples[:usable], dtype=np.float32)
        rows.append(
            {
                "name": loaded["name"],
                "fault": loaded["fault"],
                "place": place_of(loaded["fault"]),
                "load_hp": int(loaded["load_hp"]),
                "rpm": float(loaded["rpm"]),
                "signal": samples,
            }
        )
    names = [row["name"] for row in rows]
    expected = set(TRAIN_NAMES) | set(VAL_NAMES) | set(TEST_NAMES)
    if set(names) != expected or len(names) != len(expected):
        raise RuntimeError("the file split does not cover each recording once")
    return rows


def select(rows: list[dict], names: tuple[str, ...]) -> list[dict]:
    by_name = {row["name"]: row for row in rows}
    return [by_name[name] for name in names]


def stack_blocks(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    speeds = []
    for row in rows:
        count = row["signal"].size // BLOCK
        blocks.append(row["signal"].reshape(count, BLOCK))
        speeds.append(np.full(count, row["rpm"], dtype=np.float32))
    return np.concatenate(blocks), np.concatenate(speeds)


def quality(original: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    error = restored.astype(np.float64) - original.astype(np.float64)
    power = float(np.mean(original.astype(np.float64) ** 2))
    noise = float(np.mean(error ** 2))
    return {
        "rmse": float(np.sqrt(noise)),
        "energy": 100.0 * (1.0 - noise / power),
        "peak": float(np.max(np.abs(error))),
    }


def reconstruct(codec, signal: np.ndarray, rpm: float) -> tuple[np.ndarray, int, float, float]:
    restored = np.empty_like(signal)
    size = 0
    encode_seconds = 0.0
    decode_seconds = 0.0
    for start in range(0, signal.size, BLOCK):
        block = signal[start:start + BLOCK]
        before = perf_counter()
        payload = codec.encode(block, rpm)
        encode_seconds += perf_counter() - before
        before = perf_counter()
        restored[start:start + BLOCK] = codec.decode(payload, rpm)
        decode_seconds += perf_counter() - before
        size += len(payload)
    return restored, size, encode_seconds, decode_seconds


def file_row(row: dict, size: int, encode_seconds: float, decode_seconds: float, restored: np.ndarray) -> dict:
    scores = quality(row["signal"], restored)
    raw_int16 = row["signal"].size * 2
    return {
        "name": row["name"],
        "place": row["place"],
        "samples": int(row["signal"].size),
        "removed": 100.0 * (raw_int16 - size) / raw_int16,
        "bits": 8.0 * size / row["signal"].size,
        "encode_ms": 1e3 * encode_seconds,
        "decode_ms": 1e3 * decode_seconds,
        **scores,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def pack_method(name: str, files: list[dict], extra: dict | None = None) -> dict:
    removed, removed_std = mean_std([item["removed"] for item in files])
    energy, energy_std = mean_std([item["energy"] for item in files])
    rmse, rmse_std = mean_std([item["rmse"] for item in files])
    peak, peak_std = mean_std([item["peak"] for item in files])
    bits, _bits_std = mean_std([item["bits"] for item in files])
    method = {
        "name": name,
        "removed": removed,
        "removed_std": removed_std,
        "bits": bits,
        "energy": energy,
        "energy_std": energy_std,
        "rmse": rmse,
        "rmse_std": rmse_std,
        "peak": peak,
        "peak_std": peak_std,
        "encode_ms": float(np.mean([item["encode_ms"] for item in files])),
        "decode_ms": float(np.mean([item["decode_ms"] for item in files])),
        "files": files,
    }
    if extra:
        method.update(extra)
    return method


def print_method(method: dict) -> None:
    print(
        f"{method['name']:22s}"
        f"{method['removed']:6.1f} ± {method['removed_std']:4.1f}%"
        f"{method['bits']:8.2f}"
        f"{method['energy']:8.1f} ± {method['energy_std']:4.1f}%"
        f"{method['rmse']:8.3f}"
        f"{method['peak']:7.2f}"
        f"{method['encode_ms']:8.1f}"
        f"{method['decode_ms']:8.1f}",
        flush=True,
    )


def run_exact(rows: list[dict], encode, decode) -> tuple[list[dict], list[np.ndarray]]:
    files = []
    restored_rows = []
    for row in rows:
        counts = to_int16(row["signal"])
        size, encode_seconds, decode_seconds = measure(counts, BLOCK, encode, decode)
        restored = to_g(counts)
        files.append(file_row(row, size, encode_seconds, decode_seconds, restored))
        restored_rows.append(restored)
    return files, restored_rows


def run_float_zlib(rows: list[dict]) -> list[dict]:
    files = []
    for row in rows:
        signal = np.asarray(row["signal"], dtype="<f4")
        size, encode_seconds, decode_seconds = measure(
            signal,
            BLOCK,
            lambda block: zlib_raw.encode(block, level=LEVEL),
            zlib_raw.decode,
        )
        raw = signal.size * 4
        scores = quality(signal, signal)
        files.append(
            {
                "name": row["name"],
                "place": row["place"],
                "samples": int(signal.size),
                "removed": 100.0 * (raw - size) / raw,
                "bits": 8.0 * size / signal.size,
                "encode_ms": 1e3 * encode_seconds,
                "decode_ms": 1e3 * decode_seconds,
                **scores,
            }
        )
    return files


def relative_terms(reconstructed: torch.Tensor, original: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Error of each block divided by that block's own strength.

    Absolute error lets the loud outer-race files own the update. A quiet
    healthy block then keeps whatever leftover shape the decoder paints on it.
    """
    power = original.square().mean(dim=-1).clamp(min=1e-4)
    waveform = (reconstructed - original).square().mean(dim=-1) / power
    original_spec = torch.fft.rfft(original, dim=-1).abs()
    restored_spec = torch.fft.rfft(reconstructed, dim=-1).abs()
    spec_power = original_spec.square().mean(dim=-1).clamp(min=1e-4)
    spectrum = (restored_spec - original_spec).square().mean(dim=-1) / spec_power
    return waveform.mean(), spectrum.mean()


def relative_rmse(model, loader: DataLoader, device: torch.device) -> float:
    """Root mean square of per-block error over per-block strength."""
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for samples, speed in loader:
            samples = samples.to(device)
            restored = model(samples, speed.to(device))
            power = samples.square().mean(dim=-1).clamp(min=1e-4)
            mse = (restored - samples).square().mean(dim=-1)
            total += float((mse / power).sum())
            count += int(mse.shape[0])
    return float(np.sqrt(total / count))


def train_relative(model, train_loader, val_loader, device, epochs, patience):
    """Same optimizer as the turbine run, with each block weighted equally."""
    samples, speed = next(iter(train_loader))
    samples = samples.to(device)
    speed = speed.to(device)
    model.eval()
    with torch.no_grad():
        waveform, spectrum = relative_terms(model(samples, speed), samples)
    spec_weight = float((waveform / spectrum.clamp(min=1e-8)).detach())
    print(f"spectrum weight {spec_weight:.6f}", flush=True)

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
            waveform, spectrum = relative_terms(model(batch, speed), batch)
            loss = waveform + spec_weight * spectrum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(waveform.detach()) * len(batch)
            seen += len(batch)
        val_score = relative_rmse(model, val_loader, device)
        print(
            f"epoch {epoch:02d}  train relative {running / seen:.5f}  "
            f"val relative {val_score:.4f}",
            flush=True,
        )
        if val_score < best_rmse - 1e-3:
            best_rmse = val_score
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
    return best_rmse


def train_autoencoder(train_rows, val_rows, channels: int, code_time: int, device, relative: bool = False) -> tuple[Path, float, float]:
    train_blocks, train_rpm = stack_blocks(train_rows)
    val_blocks, val_rpm = stack_blocks(val_rows)
    mean = float(train_blocks.mean())
    std = float(train_blocks.std())
    rpm_mean = float(train_rpm.mean())
    rpm_std = float(train_rpm.std())
    if std < 1e-8 or rpm_std < 1e-8:
        raise RuntimeError("training scale is degenerate")

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        FlatBlocks(train_blocks, train_rpm, mean, std, rpm_mean, rpm_std),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        FlatBlocks(val_blocks, val_rpm, mean, std, rpm_mean, rpm_std),
        batch_size=128,
    )
    model = Autoencoder(channels, code_time).to(device)
    before = perf_counter()
    if relative:
        best_rmse = train_relative(model, train_loader, val_loader, device, epochs=120, patience=12)
    else:
        best_rmse, _weight = train(model, train_loader, val_loader, std, device, epochs=100, patience=10)
    seconds = perf_counter() - before
    code_values = channels * code_time
    path = Path(f"data/cwru_ae_{code_values}.pt")
    save_codec(path, model.cpu(), mean, std, rpm_mean, rpm_std, side_info=False, compress_code=False)
    return path, best_rmse, seconds


def band_peak(freqs: np.ndarray, amplitude: np.ndarray, center: float) -> float:
    band = (freqs >= center - 3.0) & (freqs <= center + 3.0)
    if not np.any(band):
        return 0.0
    return float(amplitude[band].max())


def features(window: np.ndarray, rpm: float) -> np.ndarray:
    """Loudness, sharpness, and the three bearing-hit lines."""
    centred = window.astype(np.float64) - float(np.mean(window))
    rms = float(np.sqrt(np.mean(centred ** 2)))
    scale = max(rms, 1e-8)
    kurtosis = float(np.mean((centred / scale) ** 4))
    crest = float(np.max(np.abs(centred)) / scale)
    envelope = np.abs(hilbert(centred))
    envelope = envelope - float(np.mean(envelope))
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / FS)
    shaft = rpm / 60.0
    orders = fault_orders()
    lines = [
        band_peak(freqs, spectrum, orders[name] * shaft) / scale
        for name in ("inner", "outer", "ball")
    ]
    return np.asarray([rms, kurtosis, crest, *lines], dtype=np.float64)


def recording_windows(signal: np.ndarray, rpm: float) -> np.ndarray:
    rows = []
    for start in range(0, signal.size - WINDOW + 1, WINDOW):
        rows.append(features(signal[start:start + WINDOW], rpm))
    if not rows:
        raise RuntimeError("recording is shorter than one classifier window")
    return np.vstack(rows)


def fit_softmax(samples: np.ndarray, labels: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Light linear classifier. Feature scale comes from these samples only."""
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std[std < 1e-8] = 1.0
    scaled = (samples - mean) / std
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    class_weight = counts.sum() / np.maximum(counts, 1.0)
    class_weight = class_weight / class_weight.mean()

    torch.manual_seed(SEED)
    weight = torch.zeros(samples.shape[1], n_classes, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=torch.float32, requires_grad=True)
    inputs = torch.tensor(scaled, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    weights = torch.tensor(class_weight, dtype=torch.float32)
    optimizer = torch.optim.Adam((weight, bias), lr=0.05, weight_decay=1e-3)
    for _ in range(800):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(inputs @ weight + bias, targets, weight=weights)
        loss.backward()
        optimizer.step()
    return weight.detach().numpy(), bias.detach().numpy(), mean, std


def predict_recording(windows: np.ndarray, weight, bias, mean, std) -> int:
    scaled = (windows - mean) / std
    scores = scaled @ weight + bias
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    return int(np.argmax(probs.mean(axis=0)))


def class_result(
    name: str,
    rows: list[dict],
    restored: dict[str, np.ndarray],
    models: dict,
    columns: slice | None = None,
) -> dict:
    files = []
    for row in rows:
        windows = recording_windows(restored[row["name"]], row["rpm"])
        if columns is not None:
            windows = windows[:, columns]
        binary = predict_recording(windows, *models["binary"])
        place = predict_recording(windows, *models["place"])
        files.append(
            {
                "name": row["name"],
                "true_place": row["place"],
                "true_binary": 0 if row["place"] == "normal" else 1,
                "pred_binary": binary,
                "pred_place": PLACES[place],
            }
        )
    binary_correct = sum(item["pred_binary"] == item["true_binary"] for item in files)
    place_correct = sum(item["pred_place"] == item["true_place"] for item in files)
    return {
        "name": name,
        "binary_correct": binary_correct,
        "binary_total": len(files),
        "place_correct": place_correct,
        "place_total": len(files),
        "files": files,
    }


def print_classes(result: dict) -> None:
    misses = [
        f"{item['name']} {item['true_place']}->{item['pred_place']}"
        for item in result["files"]
        if item["pred_place"] != item["true_place"] or item["pred_binary"] != item["true_binary"]
    ]
    note = ", ".join(misses) if misses else "none"
    print(
        f"{result['name']:22s}"
        f"{result['binary_correct']}/{result['binary_total']:d} healthy-fault"
        f"    {result['place_correct']}/{result['place_total']} fault place"
        f"    misses: {note}",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = prepare()
    train_rows = select(rows, TRAIN_NAMES)
    val_rows = select(rows, VAL_NAMES)
    test_rows = select(rows, TEST_NAMES)
    print(
        f"{len(rows)} recordings at {FS} Hz, block {BLOCK}. "
        f"{len(train_rows)} train, {len(val_rows)} validation, {len(test_rows)} test.",
        flush=True,
    )
    print("test:", ", ".join(row["name"] for row in test_rows), flush=True)

    print(
        f"{'method':22s}{'removed':>12s}{'bits':>8s}{'energy':>16s}{'rmse':>8s}{'peak':>7s}{'enc ms':>8s}{'dec ms':>8s}",
        flush=True,
    )
    methods = []

    print("running zlib float32...", flush=True)
    float_method = pack_method("zlib float32 *", run_float_zlib(test_rows), {"raw": "float32"})
    methods.append(float_method)
    print_method(float_method)

    count_rows = {"train": train_rows, "val": val_rows}
    before = perf_counter()
    mlp = mlp_predictor.train(
        [to_int16(row["signal"]) for row in count_rows["train"]],
        [to_int16(row["signal"]) for row in count_rows["val"]],
        epochs=30,
        seed=SEED,
    )
    mlp_seconds = perf_counter() - before
    exact_codecs = [
        ("zlib int16", lambda block: zlib_int.encode_raw(block, level=LEVEL), zlib_int.decode_raw),
        ("zlib delta", lambda block: zlib_int.encode_delta(block, level=LEVEL), zlib_int.decode_delta),
        ("zlib AR4", lambda block: ar4.encode(block, level=LEVEL), ar4.decode),
        ("zlib MLP", lambda block: mlp_predictor.encode(block, mlp, level=LEVEL), lambda payload: mlp_predictor.decode(payload, mlp)),
        ("Rice", harmonic_rice.encode, harmonic_rice.decode),
    ]
    exact_restored = None
    for name, encode, decode in exact_codecs:
        print(f"running {name}...", flush=True)
        files, restored_rows = run_exact(test_rows, encode, decode)
        method = pack_method(name, files, {"raw": "int16", "exact": True})
        if name == "zlib MLP":
            method["train_seconds"] = mlp_seconds
            method["model_bytes"] = mlp.nbytes
        methods.append(method)
        print_method(method)
        if exact_restored is None:
            exact_restored = {row["name"]: values for row, values in zip(test_rows, restored_rows)}

    device = pick_device()
    print(f"autoencoder device {device}", flush=True)
    reconstructions = {"original": {row["name"]: row["signal"] for row in test_rows}}
    reconstructions["int16"] = exact_restored
    for channels, code_time in AE_LAYOUTS:
        code_values = channels * code_time
        print(f"training AE {code_values}...", flush=True)
        path, best_rmse, seconds = train_autoencoder(train_rows, val_rows, channels, code_time, device)
        codec = open_codec(path)
        files = []
        restored = {}
        for row in test_rows:
            values, size, encode_seconds, decode_seconds = reconstruct(codec, row["signal"], row["rpm"])
            files.append(file_row(row, size, encode_seconds, decode_seconds, values))
            restored[row["name"]] = values
        method = pack_method(
            f"AE {code_values}",
            files,
            {
                "raw": "int16",
                "checkpoint": str(path),
                "val_rmse": best_rmse,
                "train_seconds": seconds,
                "code_values": code_values,
            },
        )
        methods.append(method)
        print_method(method)
        reconstructions[f"AE {code_values}"] = restored
        if code_values == 2048:
            packed_path = Path("data/cwru_ae_2048_zlib.pt")
            save_codec(
                packed_path,
                codec.model,
                codec.mean,
                codec.std,
                codec.rpm_mean,
                codec.rpm_std,
                side_info=False,
                compress_code=True,
            )
            packed = open_codec(packed_path)
            packed_files = []
            for row in test_rows:
                values, size, encode_seconds, decode_seconds = reconstruct(packed, row["signal"], row["rpm"])
                packed_files.append(file_row(row, size, encode_seconds, decode_seconds, values))
                if np.max(np.abs(values - restored[row["name"]])) != 0:
                    raise RuntimeError("zlib changed the reconstruction")
            packed_method = pack_method(
                "AE 2048 packed",
                packed_files,
                {"raw": "int16", "checkpoint": str(packed_path), "code_values": 2048},
            )
            methods.append(packed_method)
            print_method(packed_method)

    turbine_path = Path("data/ae_2048_zlib.pt")
    if turbine_path.exists():
        print("running turbine codec without retraining...", flush=True)
        turbine = open_codec(turbine_path)
        files = []
        restored = {}
        for row in test_rows:
            values, size, encode_seconds, decode_seconds = reconstruct(turbine, row["signal"], row["rpm"])
            files.append(file_row(row, size, encode_seconds, decode_seconds, values))
            restored[row["name"]] = values
        method = pack_method(
            "turbine AE, no retrain",
            files,
            {"raw": "int16", "checkpoint": str(turbine_path)},
        )
        methods.append(method)
        print_method(method)
        reconstructions["turbine AE, no retrain"] = restored

    classifier_rows = train_rows + val_rows
    binary_x = []
    binary_y = []
    place_x = []
    place_y = []
    for row in classifier_rows:
        windows = recording_windows(row["signal"], row["rpm"])
        binary_x.append(windows)
        binary_y.append(np.full(len(windows), 0 if row["place"] == "normal" else 1))
        place_x.append(windows)
        place_y.append(np.full(len(windows), PLACES.index(row["place"])))
    binary_model = fit_softmax(np.vstack(binary_x), np.concatenate(binary_y), 2)
    place_model = fit_softmax(np.vstack(place_x), np.concatenate(place_y), len(PLACES))
    rms_model = fit_softmax(np.vstack(binary_x)[:, :1], np.concatenate(binary_y), 2)
    models = {"binary": binary_model, "place": place_model}

    print()
    print("classifier on the 5 test recordings. Trained on the other original recordings.", flush=True)
    class_rows = []
    rms_place = fit_softmax(np.vstack(place_x)[:, :1], np.concatenate(place_y), len(PLACES))
    rms_only = class_result(
        "rms only, original",
        test_rows,
        reconstructions["original"],
        {"binary": rms_model, "place": rms_place},
        columns=slice(0, 1),
    )
    class_rows.append(rms_only)
    print_classes(rms_only)
    for name in ("original", "int16", "AE 16", "AE 64", "AE 256", "AE 512", "AE 2048", "turbine AE, no retrain"):
        if name not in reconstructions:
            continue
        result = class_result(name, test_rows, reconstructions[name], models)
        class_rows.append(result)
        print_classes(result)

    payload = {
        "fs": FS,
        "block": BLOCK,
        "window": WINDOW,
        "train": list(TRAIN_NAMES),
        "validation": list(VAL_NAMES),
        "test": list(TEST_NAMES),
        "methods": methods,
        "classifier": class_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


def fit_classifiers(train_rows, val_rows):
    """The same linear classifier as the full benchmark, fit on original files."""
    binary_x = []
    binary_y = []
    place_x = []
    place_y = []
    for row in train_rows + val_rows:
        windows = recording_windows(row["signal"], row["rpm"])
        binary_x.append(windows)
        binary_y.append(np.full(len(windows), 0 if row["place"] == "normal" else 1))
        place_x.append(windows)
        place_y.append(np.full(len(windows), PLACES.index(row["place"])))
    return {
        "binary": fit_softmax(np.vstack(binary_x), np.concatenate(binary_y), 2),
        "place": fit_softmax(np.vstack(place_x), np.concatenate(place_y), len(PLACES)),
    }


def improve_packed() -> None:
    """Retrain only the 2048-value codec so a quiet block is not ignored."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = prepare()
    train_rows = select(rows, TRAIN_NAMES)
    val_rows = select(rows, VAL_NAMES)
    test_rows = select(rows, TEST_NAMES)
    previous = Path("data/cwru_ae_2048.pt")
    kept = Path("data/cwru_ae_2048_abs.pt")
    if previous.exists() and not kept.exists():
        kept.write_bytes(previous.read_bytes())
        print(f"kept the previous weights at {kept}", flush=True)

    device = pick_device()
    print(f"relative-loss 2048 on {device}", flush=True)
    path, best_score, seconds = train_autoencoder(train_rows, val_rows, 16, 128, device, relative=True)
    codec = open_codec(path)
    files = []
    restored = {}
    for row in test_rows:
        values, size, encode_seconds, decode_seconds = reconstruct(codec, row["signal"], row["rpm"])
        files.append(file_row(row, size, encode_seconds, decode_seconds, values))
        restored[row["name"]] = values
        print(
            f"  {row['name']:<12} energy {files[-1]['energy']:6.1f}%  rmse {files[-1]['rmse']:.3f}",
            flush=True,
        )
    method = pack_method(
        "AE 2048",
        files,
        {
            "raw": "int16",
            "checkpoint": str(path),
            "val_relative": best_score,
            "train_seconds": seconds,
            "code_values": 2048,
            "loss": "relative",
        },
    )
    print_method(method)

    packed_path = Path("data/cwru_ae_2048_zlib.pt")
    save_codec(
        packed_path,
        codec.model,
        codec.mean,
        codec.std,
        codec.rpm_mean,
        codec.rpm_std,
        side_info=False,
        compress_code=True,
    )
    packed = open_codec(packed_path)
    packed_files = []
    for row in test_rows:
        values, size, encode_seconds, decode_seconds = reconstruct(packed, row["signal"], row["rpm"])
        packed_files.append(file_row(row, size, encode_seconds, decode_seconds, values))
        if np.max(np.abs(values - restored[row["name"]])) != 0:
            raise RuntimeError("zlib changed the reconstruction")
    packed_method = pack_method(
        "AE 2048 packed",
        packed_files,
        {"raw": "int16", "checkpoint": str(packed_path), "code_values": 2048, "loss": "relative"},
    )
    print_method(packed_method)

    models = fit_classifiers(train_rows, val_rows)
    result = class_result("AE 2048", test_rows, restored, models)
    print_classes(result)

    stored = json.loads(OUT.read_text())
    stored["methods"] = [
        packed_method if item["name"] == "AE 2048 packed" else method if item["name"] == "AE 2048" else item
        for item in stored["methods"]
    ]
    stored["classifier"] = [
        result if item["name"] == "AE 2048" else item
        for item in stored["classifier"]
    ]
    OUT.write_text(json.dumps(stored, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    import sys

    if "--relative-2048" in sys.argv:
        improve_packed()
    else:
        main()
