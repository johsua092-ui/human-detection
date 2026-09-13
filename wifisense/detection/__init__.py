"""Presence / motion detection: calibration, thresholds, engine, optional ML."""

from . import baseline, engine, ml
from .baseline import (
    Baseline, DriftAdapter, FeatureStats, baseline_path, compute_baseline, load_baseline,
)
from .engine import DetectionEngine, DetectionState

__all__ = [
    "baseline", "engine", "ml",
    "Baseline", "FeatureStats", "DriftAdapter",
    "compute_baseline", "load_baseline", "baseline_path",
    "DetectionEngine", "DetectionState",
]
