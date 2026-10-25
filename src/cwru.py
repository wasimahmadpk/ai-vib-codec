"""A slice of the Case Western Reserve University bearing recordings.

The fault files are the 12 kHz drive-end experiment. The four healthy files are
the normal baseline, which was sampled at 48 kHz. Each file is one recording.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DIR = Path("data/cwru")
URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"
FS_FAULT = 12_000
FS_NORMAL = 48_000

# Drive-end bearing, SKF 6205-2RS JEM. Orders of shaft speed.
BALLS = 9
BALL_DIAMETER_IN = 0.3126
PITCH_DIAMETER_IN = 1.537

RECORDS = (
    {"name": "Normal_0", "file_id": 97, "fault": "normal", "diameter_in": 0.0, "load_hp": 0, "rpm_table": 1797},
    {"name": "Normal_1", "file_id": 98, "fault": "normal", "diameter_in": 0.0, "load_hp": 1, "rpm_table": 1772},
    {"name": "Normal_2", "file_id": 99, "fault": "normal", "diameter_in": 0.0, "load_hp": 2, "rpm_table": 1750},
    {"name": "Normal_3", "file_id": 100, "fault": "normal", "diameter_in": 0.0, "load_hp": 3, "rpm_table": 1730},
    {"name": "IR007_0", "file_id": 105, "fault": "inner", "diameter_in": 0.007, "load_hp": 0, "rpm_table": 1797},
    {"name": "IR007_3", "file_id": 108, "fault": "inner", "diameter_in": 0.007, "load_hp": 3, "rpm_table": 1730},
    {"name": "B007_0", "file_id": 118, "fault": "ball", "diameter_in": 0.007, "load_hp": 0, "rpm_table": 1797},
    {"name": "B007_3", "file_id": 121, "fault": "ball", "diameter_in": 0.007, "load_hp": 3, "rpm_table": 1730},
    {"name": "OR007@6_0", "file_id": 130, "fault": "outer_6", "diameter_in": 0.007, "load_hp": 0, "rpm_table": 1797},
    {"name": "OR007@6_3", "file_id": 133, "fault": "outer_6", "diameter_in": 0.007, "load_hp": 3, "rpm_table": 1730},
    {"name": "OR007@3_0", "file_id": 144, "fault": "outer_3", "diameter_in": 0.007, "load_hp": 0, "rpm_table": 1797},
    {"name": "OR007@12_0", "file_id": 156, "fault": "outer_12", "diameter_in": 0.007, "load_hp": 0, "rpm_table": 1797},
    {"name": "IR021_0", "file_id": 209, "fault": "inner", "diameter_in": 0.021, "load_hp": 0, "rpm_table": 1797},
    {"name": "IR021_3", "file_id": 212, "fault": "inner", "diameter_in": 0.021, "load_hp": 3, "rpm_table": 1730},
    {"name": "B021_0", "file_id": 222, "fault": "ball", "diameter_in": 0.021, "load_hp": 0, "rpm_table": 1797},
    {"name": "B021_3", "file_id": 225, "fault": "ball", "diameter_in": 0.021, "load_hp": 3, "rpm_table": 1730},
    {"name": "OR021@6_0", "file_id": 234, "fault": "outer_6", "diameter_in": 0.021, "load_hp": 0, "rpm_table": 1797},
    {"name": "OR021@6_3", "file_id": 237, "fault": "outer_6", "diameter_in": 0.021, "load_hp": 3, "rpm_table": 1730},
)


def fault_orders() -> dict[str, float]:
    """Repeating-hit orders for the drive-end bearing, in shaft revolutions."""
    ratio = BALL_DIAMETER_IN / PITCH_DIAMETER_IN
    return {
        "inner": (BALLS / 2) * (1 + ratio),
        "outer": (BALLS / 2) * (1 - ratio),
        "ball": (PITCH_DIAMETER_IN / (2 * BALL_DIAMETER_IN)) * (1 - ratio ** 2),
    }


def ensure(record: dict) -> Path:
    """Download one official MATLAB file if it is not already cached."""
    path = DIR / f"{record['file_id']}.mat"
    if path.exists() and path.stat().st_size > 1000:
        return path
    DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL.format(file_id=record["file_id"]), path)
    return path


def load_recording(record: dict) -> dict:
    """Drive-end trace for one file. Fan-end and base traces are kept when present."""
    mat = loadmat(ensure(record))
    file_id = int(record["file_id"])
    prefix = f"X{file_id:03d}"
    drive = np.asarray(mat[f"{prefix}_DE_time"], dtype=np.float64).reshape(-1)
    fan = np.asarray(mat[f"{prefix}_FE_time"], dtype=np.float64).reshape(-1)
    base_key = f"{prefix}_BA_time"
    base = np.asarray(mat[base_key], dtype=np.float64).reshape(-1) if base_key in mat else None
    rpm_key = f"{prefix}RPM"
    if rpm_key in mat:
        rpm = float(np.asarray(mat[rpm_key]).reshape(-1)[0])
        rpm_source = "file"
    else:
        rpm = float(record["rpm_table"])
        rpm_source = "table"
    fs = FS_NORMAL if record["fault"] == "normal" else FS_FAULT
    return {
        **record,
        "drive": drive,
        "fan": fan,
        "base": base,
        "rpm": rpm,
        "rpm_source": rpm_source,
        "fs": fs,
    }
