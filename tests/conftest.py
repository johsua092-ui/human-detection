"""Shared test helpers: config factory + synthetic feature-row builders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from wifisense.config import Config, load_config
from wifisense.processing.features import extract_rssi_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg() -> Config:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    cfg.set_path("dashboard.port", 8765)
    cfg.set_path("sampling.window_s", 6.0)
    cfg.set_path("sampling.resample_hz", 20.0)
    cfg.set_path("detection.hold_s", 2.0)
    cfg.set_path("detection.hysteresis_up", 2)
    cfg.set_path("detection.hysteresis_down", 3)
    cfg.set_path("calibration.duration_s", 6.0)
    return cfg


def rssi_window(std: float, n: int = 240, fs: float = 20.0, level: float = -50.0,
                seed: int = 0, motion_phase: float = 0.0) -> Dict[str, float]:
    """One feature window from a Gaussian RSSI process with the given std.

    ``motion_phase`` > 0 overlays a low-frequency burst so the window is
    classified as motion-rich by the detectors.
    """
    rng = np.random.default_rng(seed)
    x = level + rng.normal(0.0, std, n)
    if motion_phase:
        tt = np.arange(n) / fs
        x += 2.0 * np.sin(2 * np.pi * 0.8 * tt + motion_phase)
    t = np.arange(n) / fs + 1700000000.0
    return extract_rssi_features(t, x, fs)


def empty_session(n_windows: int = 40, std: float = 0.35, seed: int = 11) -> List[Dict[str, float]]:
    return [rssi_window(std, seed=seed + i) for i in range(n_windows)]


def human_session(n_windows: int = 40, std: float = 2.2, seed: int = 22) -> List[Dict[str, float]]:
    return [rssi_window(std, seed=seed + i, motion_phase=0.7 + i) for i in range(n_windows)]
