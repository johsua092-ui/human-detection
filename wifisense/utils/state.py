"""Shared live state for the dashboard, console and API.

One lock, one version counter, one snapshot dict. The sensing loop is the only
writer; the web server, the console renderer and the API read snapshots. Keeping
this tiny is deliberate — a dashboard that can stall acquisition is worse than no
dashboard.

The hub also carries the *control* channel back into the loop: the dashboard's
"calibrate" button sets a flag that the acquisition loop picks up on its next
tick, so calibration always runs in the same thread/timing context as sensing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional


class StateHub:
    def __init__(self, history_points: int = 900, event_limit: int = 200):
        self._lock = threading.RLock()
        self._version = 0
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history_points)
        self.latest: Dict[str, Any] = {}
        self.windows: Deque[Dict[str, Any]] = deque(maxlen=600)
        self.events: Deque[Dict[str, Any]] = deque(maxlen=event_limit)
        self.meta: Dict[str, Any] = {}
        self.alarm: Dict[str, Any] = {}
        self.calibration: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.control = {
            "calibration_requested": False,
            "recalibrate_requested": False,
            "dismiss_warnings": False,
            "arm_request": None,
            "disarm_request": False,
        }
        self.started_at = time.time()
        # Monotonic sample index so many phone clients can each pull only the
        # points they have not seen yet, instead of the server re-sending the
        # whole history to every client on every tick (phone battery + LAN).
        self.sample_index = 0

    # ------------------------------------------------------------------ #
    def publish_sample(self, t: float, rssi: Optional[float], extra: Optional[Dict[str, Any]] = None
                       ) -> None:
        point: Dict[str, Any] = {"t": round(float(t), 4), "rssi": None if rssi is None else round(rssi, 3)}
        if extra:
            point.update(extra)
        with self._lock:
            self.sample_index += 1
            point["i"] = self.sample_index
            self.history.append(point)
            self._version += 1

    def since(self, index: int, limit: int = 2000) -> Dict[str, Any]:
        """Points with ``i > index`` — the incremental feed for a WS client."""
        with self._lock:
            fresh = [p for p in list(self.history) if p.get("i", 0) > index]
            if len(fresh) > limit:
                fresh = fresh[-limit:]
            return {"points": fresh, "last_index": self.sample_index}

    def publish_state(self, state: Any, features: Optional[Dict[str, float]] = None,
                      extra: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(state.to_dict() if hasattr(state, "to_dict") else (state or {}))
        if features:
            payload["features"] = {k: round(float(v), 6) for k, v in features.items()
                                   if isinstance(v, (int, float))}
        if extra:
            payload.update(extra)
        with self._lock:
            self.latest = payload
            self.windows.append({"t": payload.get("t"), "presence": payload.get("presence"),
                                 "motion": payload.get("motion"),
                                 "confidence": payload.get("confidence"),
                                 "motion_score": payload.get("motion_score"),
                                 "presence_score": payload.get("presence_score")})
            self._version += 1

    def publish_meta(self, **meta: Any) -> None:
        with self._lock:
            self.meta.update(meta)
            self._version += 1

    def publish_alarm(self, alarm: Dict[str, Any]) -> None:
        with self._lock:
            self.alarm = dict(alarm)
            self._version += 1

    def publish_stats(self, **stats: Any) -> None:
        with self._lock:
            self.stats.update(stats)
            self._version += 1

    def publish_calibration(self, **fields: Any) -> None:
        with self._lock:
            self.calibration.update(fields)
            self._version += 1

    def add_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            self._version += 1

    def add_warning(self, message: str) -> None:
        with self._lock:
            if message not in self.warnings:
                self.warnings.append(message)
                if len(self.warnings) > 8:
                    del self.warnings[:-8]
                self._version += 1

    # ------------------------------------------------------------------ #
    def snapshot(self, points: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            history = list(self.history)
            if points:
                history = history[-points:]
            return {
                "version": self._version,
                "now": time.time(),
                "uptime_s": time.time() - self.started_at,
                "last_index": self.sample_index,
                "latest": dict(self.latest),
                "history": history,
                "windows": list(self.windows)[-120:],
                "events": list(self.events)[-60:],
                "meta": dict(self.meta),
                "alarm": dict(self.alarm),
                "calibration": dict(self.calibration),
                "stats": dict(self.stats),
                "warnings": list(self.warnings),
            }

    # ---- control channel ---------------------------------------------- #
    def request_calibration(self, recalibrate: bool = False) -> None:
        with self._lock:
            if recalibrate:
                self.control["recalibrate_requested"] = True
            else:
                self.control["calibration_requested"] = True

    def request_arm(self, mode: str = "away") -> None:
        with self._lock:
            self.control["arm_request"] = mode

    def request_disarm(self) -> None:
        with self._lock:
            self.control["disarm_request"] = True

    def consume_control(self) -> Dict[str, Any]:
        """Read-and-clear the control flags (called by the acquisition loop)."""
        with self._lock:
            out = dict(self.control)
            self.control = {k: (None if k == "arm_request" else False) for k in self.control}
            if out["dismiss_warnings"]:
                self.warnings.clear()
            return out
