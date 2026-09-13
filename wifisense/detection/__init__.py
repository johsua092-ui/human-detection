"""Presence / motion detection: calibration, thresholds, engine, optional ML."""

from . import baseline, engine, localize, ml
from .baseline import (
    Baseline, DriftAdapter, FeatureStats, baseline_path, compute_baseline, load_baseline,
)
from .engine import DetectionEngine, DetectionState
from .localize import (
    LocalizationResult, Zone, ZoneLocalizer, default_zones, load_localizer,
    room_geometry, zone_model_path, zones_from_config,
)

__all__ = [
    "baseline", "engine", "localize", "ml",
    "Baseline", "FeatureStats", "DriftAdapter",
    "compute_baseline", "load_baseline", "baseline_path",
    "DetectionEngine", "DetectionState",
    "Zone", "ZoneLocalizer", "LocalizationResult", "zones_from_config", "default_zones",
    "room_geometry", "zone_model_path", "load_localizer",
]
