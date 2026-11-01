"""Rerun the benchmark on the saved models and write the results.

The autoencoders are loaded, not trained. Replace a checkpoint and run this
again; results/benchmark.md is rewritten from the measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import ar4
import harmonic_rice
import mlp_predictor
import run_cwru_benchmark as cwru
import zlib_int
import zlib_raw
from autoencoder import open_codec
from run_autoencoder import measure_test
from run_baseline import run_method
from signals import load_arrays, split_by_time
from terminal_table import print_table

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 2048
LEVEL = 9
SEED = 0
TURBINE_DATA = Path("data/dataset02_f32.npz")
TURBINE_PACKED = Path("data/ae_2048_zlib.pt")
CWRU_PACKED = Path("data/cwru_ae_2048_zlib.pt")
REPORT = Path("results/benchmark.md")

TURBINE_AE = (
    ("AE 16", Path("data/ae_16.pt")),
    ("AE 64", Path("data/ae_64.pt")),
    ("AE 256", Path("data/ae_256.pt")),
    ("AE 512", Path("data/ae_512.pt")),
    ("AE 1024", Path("data/ae_1024.pt")),
    ("AE 2048", Path("data/ae_2048.pt")),
)
CWRU_AE = (
    ("AE 16", Path("data/cwru_ae_16.pt")),
    ("AE 64", Path("data/cwru_ae_64.pt")),
    ("AE 256", Path("data/cwru_ae_256.pt")),
    ("AE 512", Path("data/cwru_ae_512.pt")),
    ("AE 1024", Path("data/cwru_ae_1024.pt")),
    ("AE 2048", Path("data/cwru_ae_2048.pt")),
)


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return path


def summary(method: dict) -> dict:
    keep = (
        "name",
        "removed",
        "energy",
        "encode_ms",
        "decode_ms",
        "rmse",
        "peak",
        "reference",
        "checkpoint",
        "fault_place",
    )
    return {key: method[key] for key in keep if key in method}


def turbine_exact(signal, counts, test_idx, mlp) -> list[dict]:
    """Size and time for the codecs that return every int16 count."""
    rows = []
    test_float = np.asarray(signal[test_idx], dtype="<f4")
    test_counts = counts[test_idx]
    n_samples = signal.shape[1]
    int_rows = [
        {"name": str(int(index)), "place": "turbine", "signal": row}
        for index, row in zip(test_idx, test_float)
    ]
    codecs = [
        ("zlib float32", test_float, n_samples * 4, "float32", lambda block: zlib_raw.encode(block, level=LEVEL), zlib_raw.decode),
        ("zlib int16", test_counts, n_samples * 2, "int16", lambda block: zlib_int.encode_raw(block, level=LEVEL), zlib_int.decode_raw),
        ("delta", test_counts, n_samples * 2, "int16", lambda block: zlib_int.encode_delta(block, level=LEVEL), zlib_int.decode_delta),
        ("AR4", test_counts, n_samples * 2, "int16", lambda block: ar4.encode(block, level=LEVEL), ar4.decode),
        ("MLP", test_counts, n_samples * 2, "int16", lambda block: mlp_predictor.encode(block, mlp, level=LEVEL), lambda payload: mlp_predictor.decode(payload, mlp)),
        ("Rice", test_counts, n_samples * 2, "int16", lambda block: harmonic_rice.encode(block), harmonic_rice.decode),
    ]
    float_energy = cwru.pack_method("unused", cwru.run_float_zlib(int_rows))["energy"]
    int_energy = cwru.pack_method("unused", cwru.run_exact(int_rows, zlib_int.encode_raw, zlib_int.decode_raw)[0])["energy"]
    for name, recordings, raw_bytes, reference, encode, decode in codecs:
        print(f"turbine {name}...", flush=True)
        sizes, encode_times, decode_times = run_method(recordings, BLOCK, encode, decode)
        removed = 100.0 * (raw_bytes - sizes) / raw_bytes
        rows.append(
            {
                "name": name,
                "removed": float(removed.mean()),
                "energy": float_energy if reference == "float32" else int_energy,
                "encode_ms": float(1e3 * encode_times.mean()),
                "decode_ms": float(1e3 * decode_times.mean()),
                "reference": reference,
            }
        )
    return rows


def turbine_autoencoders(signal, rpm, test_idx, packed_path: Path) -> list[dict]:
    rows = []
    n_samples = signal.shape[1]
    raw_bytes = n_samples * 2
    codecs = list(TURBINE_AE) + [("AE 2048 packed", packed_path)]
    for name, path in codecs:
        print(f"turbine {name}...", flush=True)
        require(path)
        measured = measure_test(open_codec(path), signal, rpm, test_idx)
        removed = 100.0 * (raw_bytes - measured["sizes"]) / raw_bytes
        rows.append(
            {
                "name": name,
                "removed": float(removed.mean()),
                "energy": float(measured["recovered"].mean()),
                "encode_ms": float(1e3 * measured["encode_times"].mean()),
                "decode_ms": float(1e3 * measured["decode_times"].mean()),
                "rmse": float(measured["rmses"].mean()),
                "peak": float(measured["peaks"].mean()),
                "reference": "int16",
                "checkpoint": str(path),
            }
        )
    return rows


def cwru_methods(packed_path: Path) -> tuple[list[dict], list[str]]:
    rows_all = cwru.prepare()
    train_rows = cwru.select(rows_all, cwru.TRAIN_NAMES)
    val_rows = cwru.select(rows_all, cwru.VAL_NAMES)
    test_rows = cwru.select(rows_all, cwru.TEST_NAMES)
    print("CWRU test:", ", ".join(row["name"] for row in test_rows), flush=True)

    methods = []
    print("CWRU zlib float32...", flush=True)
    methods.append(cwru.pack_method("zlib float32", cwru.run_float_zlib(test_rows), {"reference": "float32"}))

    mlp = mlp_predictor.train(
        [cwru.to_int16(row["signal"]) for row in train_rows],
        [cwru.to_int16(row["signal"]) for row in val_rows],
        epochs=30,
        seed=SEED,
    )
    exact = [
        ("zlib int16", lambda block: zlib_int.encode_raw(block, level=LEVEL), zlib_int.decode_raw),
        ("delta", lambda block: zlib_int.encode_delta(block, level=LEVEL), zlib_int.decode_delta),
        ("AR4", lambda block: ar4.encode(block, level=LEVEL), ar4.decode),
        ("MLP", lambda block: mlp_predictor.encode(block, mlp, level=LEVEL), lambda payload: mlp_predictor.decode(payload, mlp)),
        ("Rice", harmonic_rice.encode, harmonic_rice.decode),
    ]
    restored = {"original": {row["name"]: row["signal"] for row in test_rows}}
    int_restored = None
    for name, encode, decode in exact:
        print(f"CWRU {name}...", flush=True)
        files, restored_rows = cwru.run_exact(test_rows, encode, decode)
        methods.append(cwru.pack_method(name, files, {"reference": "int16"}))
        if int_restored is None:
            int_restored = {row["name"]: values for row, values in zip(test_rows, restored_rows)}
    restored["int16"] = int_restored

    codecs = list(CWRU_AE) + [("AE 2048 packed", packed_path)]
    for name, path in codecs:
        print(f"CWRU {name}...", flush=True)
        require(path)
        codec = open_codec(path)
        files = []
        back = {}
        for row in test_rows:
            values, size, encode_seconds, decode_seconds = cwru.reconstruct(codec, row["signal"], row["rpm"])
            files.append(cwru.file_row(row, size, encode_seconds, decode_seconds, values))
            back[row["name"]] = values
        methods.append(cwru.pack_method(name, files, {"reference": "int16", "checkpoint": str(path)}))
        restored[name] = back

    print("CWRU classifier...", flush=True)
    models = cwru.fit_classifiers(train_rows, val_rows)
    places = {}
    for name in ("int16",) + tuple(item[0] for item in codecs):
        result = cwru.class_result(name, test_rows, restored[name], models)
        places[name] = f"{result['place_correct']}/{result['place_total']}"
        misses = [
            f"{item['name']} {item['true_place']}->{item['pred_place']}"
            for item in result["files"]
            if item["pred_place"] != item["true_place"]
        ]
        if misses:
            print(f"  {name}: {', '.join(misses)}", flush=True)
    for method in methods:
        if method["reference"] == "int16" and method["name"] in ("zlib int16", "delta", "AR4", "MLP", "Rice"):
            method["fault_place"] = places["int16"]
        elif method["name"] in places:
            method["fault_place"] = places[method["name"]]
    return methods, [row["name"] for row in test_rows]


def cell(value, digits: int) -> str:
    return f"{value:.{digits}f}"


CHOSEN = "AE 2048 packed"


def show_table(title: str, methods: list[dict], fault: bool, note: str = "") -> None:
    headers = ["Method", "Removed", "Energy back", "Encode ms", "Decode ms"]
    if fault:
        headers.append("Fault place")
    rows = []
    for method in methods:
        row = [
            method["name"],
            f"{method['removed']:.1f}%",
            f"{method['energy']:.1f}%",
            f"{method['encode_ms']:.2f}",
            f"{method['decode_ms']:.2f}",
        ]
        if fault:
            row.append(str(method.get("fault_place", "—")))
        rows.append(row)
    print_table(title, headers, rows, highlight={CHOSEN}, note=note)


def markdown_table(methods: list[dict], fault: bool) -> str:
    headers = ["Method", "Removed", "Energy back", "Encode (ms)", "Decode (ms)"]
    if fault:
        headers.append("Fault place")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(headers))) + "|",
    ]
    for method in methods:
        row = [
            method["name"],
            cell(method["removed"], 1) + "%",
            cell(method["energy"], 1) + "%",
            cell(method["encode_ms"], 2),
            cell(method["decode_ms"], 2),
        ]
        if fault:
            row.append(method.get("fault_place", "—"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(path: Path, turbine: dict, bearing: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Benchmark",
            "",
            f"Written {turbine['written']} by `src/run_benchmark.py`.",
            "The autoencoder weights were loaded. This run did not train them.",
            "",
            "## Wind turbine",
            "",
            f"{turbine['test_recordings']} later recordings, 1.28 s each. "
            "Removed is versus the int16 file, except zlib float32, which is versus the float32 file. "
            "Encode and decode are milliseconds for one recording.",
            "",
            markdown_table(turbine["methods"], fault=False),
            "",
            f"Packed checkpoint: `{turbine['packed_checkpoint']}`.",
            "",
            "## CWRU",
            "",
            "Five held-out files, about 10 s each. "
            + ", ".join(bearing["test"])
            + ". Fault place is how many of those files are still named. "
            "The exact codecs share one reconstruction, so they share that score.",
            "",
            markdown_table(bearing["methods"], fault=True),
            "",
            f"Packed checkpoint: `{bearing['packed_checkpoint']}`.",
            "",
        ]
    )
    path.write_text(text)
    path.with_suffix(".json").write_text(json.dumps({"turbine": turbine, "cwru": bearing}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the saved codecs and write results/benchmark.md.")
    parser.add_argument("--data", type=Path, default=TURBINE_DATA)
    parser.add_argument("--turbine-codec", type=Path, default=TURBINE_PACKED)
    parser.add_argument("--cwru-codec", type=Path, default=CWRU_PACKED)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    os.chdir(ROOT)

    require(args.data)
    require(args.turbine_codec)
    require(args.cwru_codec)
    years, rpm, signal = load_arrays(args.data)
    train_idx, validation_idx, test_idx = split_by_time(years)
    counts = cwru.to_int16(signal)
    print(
        f"turbine split: {len(train_idx)} train, {len(validation_idx)} validation, {len(test_idx)} test",
        flush=True,
    )
    mlp = mlp_predictor.train(counts[train_idx], counts[validation_idx], epochs=30, seed=SEED)
    turbine_methods = turbine_exact(signal, counts, test_idx, mlp)
    turbine_methods.extend(turbine_autoencoders(signal, rpm, test_idx, args.turbine_codec))
    bearing_methods, test_names = cwru_methods(args.cwru_codec)

    stamp = datetime.now().isoformat(timespec="seconds")
    turbine = {
        "written": stamp,
        "test_recordings": int(len(test_idx)),
        "packed_checkpoint": str(args.turbine_codec),
        "methods": [summary(item) for item in turbine_methods],
    }
    bearing = {
        "written": stamp,
        "test": test_names,
        "packed_checkpoint": str(args.cwru_codec),
        "methods": [summary(item) for item in bearing_methods],
    }
    write_report(args.output, turbine, bearing)
    show_table(
        f"Wind turbine  ·  {len(test_idx)} test recordings",
        turbine_methods,
        fault=False,
        note="zlib float32 is versus the float32 file. Every other row is versus int16.",
    )
    show_table(
        f"CWRU  ·  {len(test_names)} held-out files",
        bearing_methods,
        fault=True,
        note="The exact codecs share one reconstruction, so they share the fault-place score.",
    )
    print(f"\nwrote {args.output}")
    print(f"wrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
