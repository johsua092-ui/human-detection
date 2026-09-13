"""Baseline / calibration / detection-engine tests.

These run the exact same code path as live sensing but on controlled synthetic
windows, so the numbers are deterministic and the expectations are physical:
a room with variance 0.1 dBm^2 must be called empty, a room with variance 4
dBm^2 and derivative energy must be called occupied, and the boundary between
them must sit where the calibration put it.
"""

import math

import numpy as np
import pytest

from wifisense.detection.baseline import (
    Baseline, DriftAdapter, FeatureStats, compute_baseline,
)
from wifisense.detection.engine import DetectionEngine, DetectionState

from conftest import empty_session, human_session, rssi_window


def test_compute_baseline_from_empty_room():
    rows = empty_session(n_windows=40, std=0.35)
    bl = compute_baseline(rows, presence_k=3.0, motion_k=2.2)
    assert bl.n_windows == 40
    assert "var" in bl.features
    assert "diff_rms" in bl.features
    var_stats = bl.features["var"]
    assert 0.0 < var_stats.median < 0.2      # std 0.35 -> var ~0.12
    assert var_stats.sigma > 0.0
    assert bl.presence_thresholds["var"] > var_stats.median


def test_baseline_roundtrip(tmp_path):
    rows = empty_session(n_windows=20)
    bl = compute_baseline(rows, mode="rssi", source="test")
    path = bl.save(tmp_path / "baseline.json")
    loaded = Baseline.load(path)
    assert loaded.n_windows == bl.n_windows
    assert loaded.features["var"].median == pytest.approx(bl.features["var"].median)


def test_baseline_rejects_schema_mismatch(tmp_path):
    bl = compute_baseline(empty_session(10))
    path = bl.save(tmp_path / "old.json")
    path.write_text(path.read_text().replace('"schema": 2', '"schema": 1'))
    with pytest.raises(ValueError, match="schema"):
        Baseline.load(path)


def test_engine_calls_empty_and_human_correctly():
    bl = compute_baseline(empty_session(40), presence_k=3.0, motion_k=2.2)
    engine = DetectionEngine(baseline=bl, method="adaptive", config=None)
    engine.hysteresis_up, engine.hysteresis_down = 2, 3

    # the empty room must stay NO HUMAN
    empty_ok = 0
    for i in range(30):
        st = engine.update(rssi_window(0.35, seed=100 + i))
        if not st.presence:
            empty_ok += 1
    assert empty_ok >= 25

    engine.reset()
    # a person walking (variance + motion) must be flagged
    hits = 0
    for i in range(30):
        st = engine.update(rssi_window(2.5, seed=200 + i, motion_phase=0.3 + i))
        if st.presence:
            hits += 1
    assert hits >= 25
    assert any(s.motion for s in engine.history[:10])


def test_engine_reports_confidence_shape():
    bl = compute_baseline(empty_session(30), presence_k=3.0)
    engine = DetectionEngine(baseline=bl)
    st_empty = engine.update(rssi_window(0.3, seed=5))
    engine.reset()
    st_human = engine.update(rssi_window(3.0, seed=6, motion_phase=0.5))
    assert 0.0 <= st_empty.confidence <= 99.9
    assert 0.0 <= st_human.confidence <= 99.9
    assert st_human.confidence > st_empty.confidence
    assert st_human.confidence > 50.0
    assert st_empty.presence_label == "NO HUMAN"


def test_engine_hysteresis_suppresses_single_blip():
    bl = compute_baseline(empty_session(30), presence_k=3.0, motion_k=2.2)
    engine = DetectionEngine(baseline=bl)
    engine.hysteresis_up = 3
    engine.hysteresis_down = 3
    engine.motion_k = 1e6   # keep motion out of the picture: this is a PRESENCE test
    states = []
    for i in range(14):
        if i == 7:
            states.append(engine.update(rssi_window(4.0, seed=400)))
        else:
            states.append(engine.update(rssi_window(0.3, seed=401 + i)))
    # one loud window must not be enough to assert presence (hysteresis 3)
    assert states[7].presence is False


def test_engine_without_baseline_fallback_runs():
    engine = DetectionEngine(baseline=None, method="adaptive", config=None)
    engine.hysteresis_up = 1
    states = []
    for i in range(40):
        std = 0.3 if i < 20 else 3.0
        states.append(engine.update(rssi_window(std, seed=1000 + i, motion_phase=0.4 if i >= 20 else 0)))
    assert states[-1].presence is True
    assert states[-1].calibrated is False  # honest about missing calibration


def test_engine_hold_keeps_presence_alive():
    bl = compute_baseline(empty_session(20), presence_k=3.0)
    engine = DetectionEngine(baseline=bl, config=None)
    engine.hysteresis_up = 1
    engine.hysteresis_down = 1
    engine.hold_s = 5.0
    t0 = 1700000000.0
    engine.update(rssi_window(3.5, seed=300, motion_phase=0.2), t=t0)
    st = engine.update(rssi_window(0.3, seed=301), t=t0 + 3.0)
    assert st.presence is True
    assert st.holding is True


def test_drift_adapter_shifts_baseline_slowly():
    bl = compute_baseline(empty_session(20), presence_k=3.0)
    adapter = DriftAdapter(bl, rate=0.05, max_shift=0.5)
    before = bl.features["mean"].median
    feats = {"mean": before + 4.0}
    adapter.update(feats, is_empty=True)
    assert 0 < (bl.features["mean"].median - before) < 0.5  # capped per window


def test_baseline_notes_on_noisy_calibration():
    # a burst of genuinely noisy windows at the end => the calibration is flagged
    rows = empty_session(n_windows=35, std=0.4, seed=500)
    rows += [rssi_window(12.0, seed=600 + i, motion_phase=0.3) for i in range(5)]
    bl = compute_baseline(rows)
    notes = " ".join(bl.notes).lower()
    assert "not empty" in notes or "activity" in notes
