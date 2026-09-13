"""Synthetic and replay sources.

**These are not sensors.** They exist for exactly three legitimate jobs:

1. unit/integration tests that must run on machines with no radio;
2. working on the dashboard, feature pipeline and calibration logic offline;
3. replaying a real recording you captured earlier (``ReplayCollector``) —
   which *is* real data, just not live.

Every sample they produce is tagged ``meta["synthetic"]=True`` so it can never
be mistaken for a measurement, the runner prints a loud warning, and the mode is
only reachable when explicitly requested on the command line.
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import Collector, CollectorError, Sample

DEMO_BANNER = (
    "SYNTHETIC SOURCE ACTIVE — the numbers below are generated demo data, not radio "
    "measurements. Presence/motion results are meaningless. Use a real backend "
    "(--source radiotap / iw / proc) for actual sensing."
)


class SyntheticRSSICollector(Collector):
    """Physically-shaped fake RSSI: OU drift + optional breathing/motion modulation.

    Scenario cycle (configurable): ``empty`` → ``presence`` → ``motion`` → repeat,
    so the dashboard and detection engine can be exercised end to end.
    """

    name = "rssi_synthetic"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None,
                 scenario: Optional[str] = None, seed: Optional[int] = None,
                 rate_hz: Optional[float] = None):
        super().__init__(config, interface)
        # Explicit constructor arguments win over the config file: a caller that
        # asks for 50 Hz must not be silently downgraded to the config default.
        self.scenario = scenario or "cycle"
        self.seed = seed
        self.rate_hz = rate_hz
        self._explicit = {"scenario": scenario is not None, "seed": seed is not None,
                          "rate": rate_hz is not None}
        self._phase = 0.0

    def preflight(self) -> None:
        if self.config:
            if not self._explicit["scenario"]:
                self.scenario = self.config.get_path("synthetic.scenario", self.scenario)
            if not self._explicit["seed"]:
                self.seed = int(self.config.get_path("synthetic.seed", 1337))
            if not self._explicit["rate"]:
                self.rate_hz = float(self.config.get_path("synthetic.rate_hz", 20.0))
        self.seed = int(self.seed if self.seed is not None else 1337)
        self.rate_hz = float(max(1.0, min(self.rate_hz or 20.0, 200.0)))
        self.status["demo"] = True

    def scenario_at(self, t: float) -> str:
        if self.scenario != "cycle":
            return self.scenario
        cycle = t % 30.0
        if cycle < 12.0:
            return "empty"
        if cycle < 21.0:
            return "presence"
        return "motion"

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        self.started_at = time.time()
        rng = random.Random(self.seed)
        interval = 1.0 / self.rate_hz
        level = -52.0
        t0 = time.time()
        next_t = time.time()
        while not self.stopped:
            now = time.time()
            # absolute-time pacing: emit at exactly rate_hz regardless of loop
            # overhead, so downstream window sizes match the configured rate
            if now < next_t:
                self._sleep(next_t - now)
                continue
            next_t += interval
            elapsed = now - t0
            scenario = self.scenario_at(elapsed)

            # Ornstein-Uhlenbeck drift: the room's slow multipath baseline.
            level += 0.08 * (-52.0 - level) * interval + rng.gauss(0.0, 0.25 * math.sqrt(interval))
            rssi = level + rng.gauss(0.0, 0.28)

            if scenario == "presence":
                rssi += 0.9 * math.sin(2 * math.pi * 0.28 * elapsed)      # breathing-scale
                rssi += 0.4 * math.sin(2 * math.pi * 0.05 * elapsed + 1.1)
            elif scenario == "motion":
                burst = math.sin(2 * math.pi * 0.7 * elapsed) ** 2
                rssi += rng.gauss(0.0, 1.5) * (0.4 + burst)
                rssi += 2.2 * math.sin(2 * math.pi * 1.4 * elapsed) * burst

            yield self.emit(Sample(
                t=now,
                rssi=rssi,
                source=self.name,
                meta={"synthetic": True, "scenario": scenario, "demo": True},
            ))

    def stop(self) -> None:
        self._stop.set()


class ReplayCollector(Collector):
    """Replay a CSV written by the data logger (or any file with a t/rssi column).

    This is real recorded data, re-timed onto the wall clock (``speed`` = 1.0 is
    real time, ``speed`` = 0 or None replays as fast as possible) — the standard
    way to re-run detection over a finished session or to compare thresholds.
    """

    name = "replay"
    mode = "rssi"

    def __init__(self, config: Any = None, path: str = "", speed: Optional[float] = 1.0,
                 loop: bool = False, interface: Optional[str] = None):
        super().__init__(config, interface)
        self.path = path
        self.speed = speed
        self.loop = loop

    def preflight(self) -> None:
        p = Path(self.path).expanduser()
        if not p.exists():
            raise CollectorError(f"replay file not found: {p}")
        rows = self._read(p)
        if not rows:
            raise CollectorError(
                f"{p} has no usable rows (need columns: t,rssi)",
                ["point --replay at a file written by this tool (data/session_*.csv)"],
            )
        self.status["rows"] = len(rows)

    def _read(self, path: Path) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                try:
                    t = float(raw.get("t") or raw.get("timestamp") or 0.0)
                    rssi = float(raw["rssi"])
                except (TypeError, ValueError, KeyError):
                    continue
                rows.append({"t": t, "rssi": rssi})
        return rows

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        path = Path(self.path).expanduser()
        rows = self._read(path)
        self.started_at = time.time()
        while not self.stopped:
            prev_t = None
            wall0 = time.time()
            base_t = rows[0]["t"]
            for i, row in enumerate(rows):
                if self.stopped:
                    return
                if self.speed and prev_t is not None:
                    target = (row["t"] - base_t) / self.speed
                    drift = (time.time() - wall0) - target
                    if drift < 0:
                        self._sleep(min(-drift, 1.0))
                prev_t = row["t"]
                yield self.emit(Sample(
                    t=time.time(),
                    rssi=row["rssi"],
                    source=self.name,
                    meta={"replay": True, "source_t": row["t"], "index": i,
                          "file": path.name},
                ))
            if not self.loop:
                return

    def stop(self) -> None:
        self._stop.set()
