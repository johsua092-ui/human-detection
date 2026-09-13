"""Collector factory, capability detection and data-logger tests.

These must pass on a machine with no wireless hardware at all (CI, containers):
capability detection is expected to report "nothing available" *cleanly*, the
factory is expected to raise :class:`CollectorError` with actionable hints
instead of inventing a backend, and the synthetic/replay sources are expected to
be explicitly tagged as non-measurements.
"""

import csv
import json
from pathlib import Path

import pytest

from wifisense.collectors import CollectorError, build_collector
from wifisense.collectors import capabilities as caps
from wifisense.collectors.synthetic import ReplayCollector, SyntheticRSSICollector
from wifisense.utils.data_logger import DataLogger, label_windows


def test_detect_os_and_tools():
    os_info = caps.detect_os()
    assert os_info["system"] in ("linux", "darwin", "windows")
    assert isinstance(os_info["is_root"], bool)
    tools = caps.detect_tools()
    assert "iw" in tools  # key exists even when the binary is missing


def test_capability_report_is_serialisable(cfg):
    rep = caps.capability_report()
    json.dumps(rep, default=str)  # must never blow up the API
    assert "decision" in rep
    assert rep["decision"]["mode"] in ("rssi", "csi", None)


def test_format_report_renders():
    text = caps.format_report(caps.capability_report(), color=False)
    assert "capability report" in text
    assert "Decision" in text


def test_choose_backend_synthetic_marks_demo():
    decision = caps.choose_backend(caps.detect_os(), [], {}, caps.detect_csi(),
                                   mode="auto", source="synthetic")
    assert decision["source"] == "synthetic"
    assert "DEMO" in " ".join(decision["limitations"]).upper() or \
           "SYNTHETIC" in " ".join(decision["limitations"]).upper()


def test_choose_backend_no_hardware_has_hint():
    decision = caps.choose_backend({"is_linux": False, "is_windows": False, "is_macos": False},
                                   [], {}, {"available": False, "backends": [], "live_csi": False,
                                            "offline_csi": False},
                                   mode="auto", source="auto")
    assert decision["mode"] is None
    assert any("--source synthetic" in line for line in decision["limitations"])


def test_factory_rejects_auto(cfg):
    with pytest.raises(CollectorError) as exc:
        build_collector(cfg, source="auto")
    assert "auto" in str(exc.value)


def test_factory_rejects_unknown_source(cfg):
    with pytest.raises(CollectorError) as exc:
        build_collector(cfg, source="telepathy")
    assert "unknown source" in str(exc.value)


def test_factory_rejects_replay_without_file(cfg):
    with pytest.raises(CollectorError):
        build_collector(cfg, source="replay")


def test_factory_builds_synthetic_and_replay(cfg, tmp_path):
    synth = build_collector(cfg, source="synthetic")
    assert isinstance(synth, SyntheticRSSICollector)

    log = tmp_path / "windows_test.csv"
    with log.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["t", "rssi"])
        writer.writeheader()
        for i in range(10):
            writer.writerow({"t": 1700000000 + i * 0.1, "rssi": -50 - i * 0.1})
    replay = build_collector(cfg, source="replay", replay=str(log))
    assert isinstance(replay, ReplayCollector)


def test_replay_and_synthetic_streams(cfg, tmp_path):
    synth = SyntheticRSSICollector(cfg, scenario="empty", seed=3, rate_hz=100.0)
    samples = []
    for sample in synth.stream():
        samples.append(sample)
        if len(samples) >= 25:
            break
    assert all(s.rssi is not None for s in samples)
    assert all(s.meta.get("synthetic") for s in samples)  # never mistakable for real

    path = tmp_path / "samples.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["t", "rssi"])
        writer.writeheader()
        for i in range(30):
            writer.writerow({"t": 1700000000 + i * 0.05, "rssi": -49.5})
    replay = ReplayCollector(cfg, path=str(path), speed=0.0)
    replayed = []
    for sample in replay.stream():
        replayed.append(sample)
        if len(replayed) >= 30:
            break
    assert len(replayed) == 30
    assert replayed[0].rssi == pytest.approx(-49.5)


def test_csi_without_hardware_raises_with_hints(cfg):
    from wifisense.collectors.csi import CSICollector
    collector = CSICollector(cfg, tool="iwl5300")
    with pytest.raises(CollectorError) as exc:
        collector.preflight()
    assert exc.value.hints  # must tell the user what to do instead
    assert any("csi" in h.lower() for h in exc.value.hints)


def test_data_logger_writes_all_three_files(cfg, tmp_path):
    from wifisense.collectors.base import Sample
    from wifisense.detection.baseline import compute_baseline
    from wifisense.detection.engine import DetectionEngine

    logger = DataLogger(directory=tmp_path, enabled=True)
    baseline = compute_baseline([{"var": 0.1, "diff_rms": 0.01}] * 20)
    engine = DetectionEngine(cfg, baseline=baseline)

    for i in range(5):
        logger.log_sample(Sample(t=1700000000.0 + i, rssi=-50.0 + i, source="test"))
    feats = {"var": 0.12, "diff_rms": 0.02, "n": 240.0}
    state = engine.update(feats, t=1700000010.0)
    logger.log_window(feats, state)
    logger.log_event({"t": 1700000020.0, "kind": "triggered", "severity": "alarm",
                      "message": "test event"})
    meta_path = logger.write_meta({"note": "unit test"})
    summary = logger.close()

    assert Path(summary["files"]["samples"]).exists()
    assert Path(summary["files"]["windows"]).exists()
    assert Path(summary["files"]["events"]).exists()
    assert meta_path.exists()
    assert summary["counts"]["samples"] == 5

    rows = list(csv.DictReader(Path(summary["files"]["windows"]).open()))
    assert rows[0]["presence"] in ("0", "1")
    assert float(rows[0]["var"]) == pytest.approx(0.12)
    assert "label" in rows[0]  # empty column ready for labelling


def test_label_windows_writes_label_column(tmp_path):
    path = tmp_path / "windows_x.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["t", "var", "label"])
        writer.writeheader()
        writer.writerow({"t": 1, "var": 0.1, "label": ""})
        writer.writerow({"t": 2, "var": 0.2, "label": ""})
    assert label_windows(path, "human") == 2
    rows = list(csv.DictReader(path.open()))
    assert all(row["label"] == "human" for row in rows)


def test_state_hub_incremental_feed():
    from wifisense.utils.state import StateHub
    hub = StateHub(history_points=50)
    for i in range(10):
        hub.publish_sample(1700000000 + i, -50 + i)
    snap = hub.snapshot(points=5)
    assert len(snap["history"]) == 5
    since = hub.since(4)
    assert len(since["points"]) == 6
    assert since["last_index"] == 10
    hub.request_calibration()
    control = hub.consume_control()
    assert control["calibration_requested"] is True
    assert hub.consume_control()["calibration_requested"] is False
