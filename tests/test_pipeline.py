"""End-to-end pipeline test: synthetic stream -> features -> engine -> state.

Uses the project's own labeled synthetic source (explicitly requested, tagged
``meta.synthetic=True``) so the entire chain — collector loop, filter, window,
engine, alarm, hub — runs in-process without any radio. The scenario cycles
empty -> presence -> motion, so both detections get exercised.
"""

import pytest

from wifisense.alarm.manager import AlarmManager
from wifisense.collectors.synthetic import SyntheticRSSICollector
from wifisense.detection.baseline import compute_baseline
from wifisense.detection.engine import DetectionEngine, DetectionState
from wifisense.processing.features import RSSIFeatureExtractor
from wifisense.processing.filters import StreamingFilter
from wifisense.utils.state import StateHub

from conftest import empty_session


def _run(cfg, scenario, seconds_limit, samples_limit, window_s=3.0, rate_hz=50.0, seed=7):
    """Drive the synthetic collector through the real processing chain."""
    collector = SyntheticRSSICollector(cfg, scenario=scenario, seed=seed, rate_hz=rate_hz)
    baseline = compute_baseline(empty_session(40), presence_k=3.0, motion_k=2.2)
    engine = DetectionEngine(cfg, baseline=baseline, method="adaptive")
    engine.hysteresis_up, engine.hysteresis_down = 2, 3
    hub = StateHub(history_points=200)
    sf = StreamingFilter(outlier_sigma=4.0, median_window=5)
    ex = RSSIFeatureExtractor(window_s=window_s, fs=rate_hz, min_samples=int(window_s * rate_hz * 0.5))

    counts = {"samples": 0, "synthetic_tagged": 0, "present": 0, "empty": 0, "motion": 0,
              "windows": 0}
    since_analysis = 0.0
    for sample in collector.stream():
        counts["samples"] += 1
        if sample.meta.get("synthetic"):
            counts["synthetic_tagged"] += 1
        result = sf.update(sample.rssi)
        ex.push(sample.t, result["filtered"])
        hub.publish_sample(sample.t, result["ema"])
        since_analysis += 1.0 / rate_hz
        if ex.ready() and since_analysis >= 0.5:
            since_analysis = 0.0
            feats = ex.features()
            if feats:
                counts["windows"] += 1
                state = engine.update(feats)
                hub.publish_state(state, feats)
                counts["present"] += int(state.presence)
                counts["motion"] += int(state.motion)
                counts["empty"] += int(not state.presence)
        if counts["samples"] >= samples_limit:
            break
        if (sample.t - collector.started_at) > seconds_limit:
            break
    return counts, hub, engine


def test_pipeline_synthetic_cycle(cfg):
    counts, hub, engine = _run(cfg, "cycle", seconds_limit=35, samples_limit=3000)
    assert counts["synthetic_tagged"] >= counts["samples"]  # everything is tagged demo
    assert counts["windows"] > 20
    # the cycle spends time both empty and occupied
    assert counts["present"] > 0
    assert counts["empty"] > 0
    assert counts["motion"] > 0
    snap = hub.snapshot(points=10)
    assert len(snap["history"]) > 0
    assert snap["latest"]["presence"] in (True, False)


def test_pipeline_empty_scenario_never_fires(cfg):
    counts, _hub, _engine = _run(cfg, "empty", seconds_limit=15, samples_limit=1500, seed=8)
    assert counts["windows"] > 10
    assert counts["present"] == 0


def test_pipeline_presence_scenario_always_fires(cfg):
    counts, _hub, _engine = _run(cfg, "presence", seconds_limit=15, samples_limit=1500, seed=9)
    assert counts["windows"] > 5
    # a person breathing/drifting in the room must be seen as present
    assert counts["present"] > 0


def _fake_state(presence, confidence, motion, t):
    return DetectionState(t=t, presence=presence, motion=motion, confidence=confidence,
                          presence_score=10.0, motion_score=5.0)


class _NullNotifier:
    def __init__(self):
        self.sent = 0
        self.notifiers = []
        self.min_severity = "info"

    def send(self, *args, **kwargs):
        self.sent += 1
        return True

    def health(self):
        return {"notifiers": [], "sent": self.sent}

    def add(self, notifier):
        pass


def test_alarm_triggers_only_when_armed_sustained_and_confident(cfg):
    notifier = _NullNotifier()
    manager = AlarmManager(cfg, notifier=notifier)
    manager.enabled = True
    manager.arm_delay_s = 0.0
    manager.sustain_s = 0.0
    manager.min_confidence = 50.0

    manager.arm("away", delay=0.0)
    assert manager.armed

    # a weak detection must not trigger
    events = manager.update(_fake_state(True, 20.0, False, 100.0), now=101.0)
    assert not any(e.kind == "triggered" for e in events)

    # strong + sustained: triggers immediately (sustain_s=0) and notifies
    events = manager.update(_fake_state(True, 88.0, True, 102.0), now=102.0)
    assert any(e.kind == "triggered" for e in events)
    assert manager.trigger_count == 1
    assert notifier.sent >= 1

    # inside the cooldown: suppressed and logged, not re-alerted
    events = manager.update(_fake_state(True, 90.0, True, 104.0), now=104.0)
    assert not any(e.kind == "triggered" for e in events)
    assert any(e.kind == "suppressed" for e in events)

    # disarmed: silent
    manager.disarm(now=110.0)
    events = manager.update(_fake_state(True, 95.0, True, 111.0), now=112.0)
    assert not events


def test_alarm_home_mode_requires_motion(cfg):
    manager = AlarmManager(cfg, notifier=_NullNotifier())
    manager.enabled = True
    manager.arm_delay_s = 0.0
    manager.sustain_s = 0.0
    manager.min_confidence = 0.0
    manager.trigger_on = "motion"
    manager.arm("home", delay=0.0)

    events = manager.update(_fake_state(True, 95.0, False, 1.0), now=2.0)
    assert not any(e.kind == "triggered" for e in events)

    events = manager.update(_fake_state(True, 95.0, True, 3.0), now=4.0)
    assert any(e.kind == "triggered" for e in events)


def test_alarm_exit_delay_blocks_immediate_trigger(cfg):
    manager = AlarmManager(cfg, notifier=_NullNotifier())
    manager.enabled = True
    manager.arm_delay_s = 30.0
    manager.sustain_s = 0.0
    manager.min_confidence = 0.0
    manager.arm("away", delay=30.0, now=0.0)
    events = manager.update(_fake_state(True, 99.0, True, 1.0), now=2.0)
    assert not events  # still in ARMING
    assert manager.state == "ARMING"
    events = manager.update(_fake_state(True, 99.0, True, 40.0), now=40.0)
    assert any(e.kind == "armed" for e in events)
