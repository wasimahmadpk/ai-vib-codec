"""Load turbine vibration recordings and measure a segment."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

URL = "https://api.researchdata.se/dataset/2024-248/1/file/data/DATASET02.csv"
FS = 12_800
N_SAMPLES = 16_384
CHUNK = 2_000_000
CACHE = Path("data/dataset02.npz")
FULL_CACHE = Path("data/dataset02_f32.npz")
CSV_PATH = Path("data/DATASET02.csv")


def csv_size(url: str) -> int:
    """Byte length, from Content-Range on a 1-byte read."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        # bytes 0-0/<size>
        return int(resp.headers["Content-Range"].split("/")[-1])


def read_span(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def full_lines(blob: bytes, skip_first: bool) -> list[str]:
    """Drop a line cut off at either end of the byte slice."""
    text = blob.decode("ascii")
    parts = text.split("\n")
    if skip_first:
        parts = parts[1:]
    if not text.endswith("\n"):
        parts = parts[:-1]
    return [p.strip("\r").strip() for p in parts if p.strip("\r").strip()]


def parse_line(line: str) -> tuple[float, float, np.ndarray, int]:
    t_txt, rpm_txt, rest = line.split(",", 2)
    x = np.fromstring(rest, sep=",")
    return float(t_txt), float(rpm_txt), x, len(line.encode("ascii"))


def load_subset(cache: Path = CACHE):
    """Return years, rpm, samples, text_bytes, file_size.

    Uses the local cache when it exists. Otherwise reads five 2 MB windows
    of DATASET02.csv: start, quarter, half, three quarters, and the end.
    """
    if cache.exists():
        stored = np.load(cache)
        years = stored["years"]
        rpm = stored["rpm"]
        samples = stored["signal"]
        text_bytes = stored["text_bytes"]
        file_size = int(stored["file_size"])
        print(f"loaded {cache} ({samples.shape[0]} segments)")
        return years, rpm, samples, text_bytes, file_size

    file_size = csv_size(URL)
    starts = sorted({
        0,
        file_size // 4,
        file_size // 2,
        (3 * file_size) // 4,
        max(0, file_size - CHUNK),
    })
    rows = []
    for start in starts:
        end = min(file_size - 1, start + CHUNK - 1)
        blob = read_span(URL, start, end)
        got = full_lines(blob, skip_first=start > 0)
        print(f"bytes {start:,}-{end:,}: {len(got)} complete rows")
        rows.extend(got)

    parsed = [parse_line(row) for row in rows]
    years = np.array([item[0] for item in parsed])
    rpm = np.array([item[1] for item in parsed])
    text_bytes = np.array([item[3] for item in parsed])
    lengths = {len(item[2]) for item in parsed}
    if len(lengths) != 1:
        raise RuntimeError(f"segment lengths differ: {sorted(lengths)}")
    samples = np.vstack([item[2] for item in parsed])

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        years=years,
        rpm=rpm,
        signal=samples,
        text_bytes=text_bytes,
        file_size=np.int64(file_size),
    )
    print(f"wrote {cache}")
    return years, rpm, samples, text_bytes, file_size


def load_arrays(path: Path = CACHE):
    """Return years, rpm, and the vibration matrix from a local cache."""
    stored = np.load(path)
    years = np.asarray(stored["years"])
    rpm = np.asarray(stored["rpm"])
    signal = np.asarray(stored["signal"])
    print(f"loaded {path} ({signal.shape[0]} segments)")
    return years, rpm, signal


def load_full(cache: Path = FULL_CACHE, csv_path: Path = CSV_PATH):
    """Return every DATASET02 recording as years, rpm, and float32 samples."""
    if cache.exists():
        return load_arrays(cache)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("rb") as handle:
        n = sum(1 for line in handle if line.strip())
    years = np.empty(n, dtype=np.float64)
    rpm = np.empty(n, dtype=np.float64)
    signal = np.empty((n, N_SAMPLES), dtype=np.float32)
    i = 0
    with csv_path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            t_txt, rpm_txt, rest = line.decode("ascii").split(",", 2)
            samples = np.fromstring(rest, sep=",", dtype=np.float32)
            if samples.size != N_SAMPLES:
                raise ValueError(f"row {i} has {samples.size} samples")
            years[i] = float(t_txt)
            rpm[i] = float(rpm_txt)
            signal[i] = samples
            if i % 500 == 0:
                print(f"parsed {i:,} / {n:,}", flush=True)
            i += 1
    if i != n:
        raise RuntimeError(f"parsed {i} rows, expected {n}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, years=years, rpm=rpm, signal=signal)
    print(f"wrote {cache} ({n} segments)")
    return years, rpm, signal


def split_by_time(years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 70/15/15 recording indices in chronological order."""
    order = np.argsort(years)
    n_train = int(0.70 * len(order))
    n_validation = max(1, int(0.15 * len(order)))
    train = order[:n_train]
    validation = order[n_train:n_train + n_validation]
    test = order[n_train + n_validation:]
    return train, validation, test


def row_stats(x: np.ndarray) -> dict[str, float]:
    centred = x - x.mean()
    scale = centred.std()
    diff = np.diff(x)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    return {
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "peak": float(np.max(np.abs(x))),
        "kurt": float(np.mean((centred / scale) ** 4)),
        "lag1": float(lag1),
        "dstd": float(diff.std() / x.std()),
    }


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(len(x))
    fft = np.fft.rfft(x * w)
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    # Divide by the window sum so a unit sine stays near amplitude 1.
    amp = np.abs(fft) * 2.0 / w.sum()
    return freq, amp


def shake_parts(
    x: np.ndarray,
    fs: float = FS,
    n_parts: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest separated shakes as sine waves."""
    w = np.hanning(len(x))
    fft = np.fft.rfft(x * w)
    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
    amp = np.abs(fft) * 2.0 / w.sum()
    phase = np.angle(fft)

    band = np.flatnonzero((freq >= 40) & (freq <= 2500))
    peaks = [
        i for i in band[2:-2]
        if amp[i] > amp[i - 1]
        and amp[i] > amp[i + 1]
        and amp[i] > amp[i - 2]
        and amp[i] > amp[i + 2]
    ]
    peaks.sort(key=lambda i: amp[i], reverse=True)

    chosen = []
    for i in peaks:
        if all(abs(freq[i] - freq[j]) > 30 for j in chosen):
            chosen.append(i)
        if len(chosen) == n_parts:
            break

    t = np.arange(len(x)) / fs
    parts = np.array([
        amp[i] * np.cos(2 * np.pi * freq[i] * t + phase[i])
        for i in chosen
    ])
    return freq[chosen], parts


def main_peak(x: np.ndarray, speed: float) -> tuple[float, float]:
    """Loudest line above 40 Hz, as Hz and as a shaft order."""
    freq, amp = spectrum(x, FS)
    band = (freq >= 40) & (freq <= 3000)
    i = int(np.argmax(amp[band]))
    hz = float(freq[band][i])
    return hz, hz / (speed / 60.0)


def energy_bins(x: np.ndarray, frac: float) -> int:
    """How many frequency bins hold ``frac`` of the power."""
    _, amp = spectrum(x, FS)
    power = amp[1:] ** 2
    ranked = np.sort(power)[::-1]
    cume = np.cumsum(ranked) / power.sum()
    return int(np.searchsorted(cume, frac) + 1)
