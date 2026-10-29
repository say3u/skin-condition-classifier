"""
Post-hoc confidence calibration via temperature scaling.

Softmax outputs from image classifiers are famously over-confident. Dividing the
logits by a single learned scalar T before softmax recovers calibrated
probabilities with no retraining and no accuracy loss. See:

    Guo et al. (2017), "On Calibration of Modern Neural Networks" (ICML).

Usage:
    1. Run scripts/calibrate_temperature.py on the HAM10000 validation set.
       It writes `calibration.json` at the project root with the optimal T.
    2. This module loads that T at startup. If the file is missing or malformed
       we fall back to T=1.0, which is a no-op (identical to the uncalibrated
       behavior). The deployment never breaks because calibration is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_T = 1.0
_CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.json"


def load_temperature() -> float:
    """Return the saved temperature, or 1.0 if nothing's been calibrated yet."""
    try:
        if not _CALIBRATION_FILE.exists():
            return _DEFAULT_T
        data = json.loads(_CALIBRATION_FILE.read_text(encoding="utf-8"))
        t = float(data.get("temperature", _DEFAULT_T))
        if not (0.1 <= t <= 10.0):
            print(f"[calibration] temperature {t} out of sane range, using 1.0")
            return _DEFAULT_T
        return t
    except Exception as exc:
        print(f"[calibration] couldn't load {_CALIBRATION_FILE}: {exc!r}, using 1.0")
        return _DEFAULT_T
