"""Intrusion alarm layer — the "someone is in my room while I'm out" use case.

The sensing stack answers *is there a human in this radio environment*. This
module turns that answer into an alarm, with the state machine a real alarm
needs or it becomes useless white noise:

* **ARMING** — an exit delay, so arming the system and walking out does not
  immediately trigger it.  ``arm_delay_s``.
* **ARMED_AWAY** — any presence at all triggers.  This is the thief case: a
  burglar who freezes in the dark still perturbs the multipath, so presence
  (not motion) is what must fire.  Motion-only arming is how you miss people.
* **ARMED_HOME** — only motion triggers.  Someone is home, so mere presence is
  expected; what matters is movement, e.g. at night while you sleep.
* **Sustain + confidence gates** — a single window above threshold never
  triggers: the condition must hold for ``sustain_s`` and clear
  ``min_confidence``.  That kills the false alarm from a neighbour's microwave.
* **Cooldown** — after an alert, re-alerts are suppressed for ``cooldown_s`` but
  still recorded, so the event log keeps the full timeline.
* **CLEAR** — an optional "room clear again" notice, off by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..detection.engine import DetectionState
from .notifier import MultiNotifier

DISARMED = "DISARMED"
ARMING = "ARMING"
ARMED_AWAY = "ARMED_AWAY"
ARMED_HOME = "ARMED_HOME"


@dataclass
class AlarmEvent:
    t: float
    kind: str            # armed | arming | disarmed | triggered | cleared | suppressed
    severity: str        # info | notice | alarm
    message: str
    state: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = {"t": round(self.t, 6), "kind": self.kind, "severity": self.severity,
               "message": self.message}
        for key in ("presence", "motion", "confidence", "presence_ratio", "motion_ratio",
                    "reason", "source", "interface"):
            if key in self.state:
                row[key] = self.state[key]
        return row

    def to_dict(self) -> Dict[str, Any]:
        return {"t": self.t, "kind": self.kind, "severity": self.severity,
                "message": self.message, "state": self.state}


class AlarmManager:
    """Arming state + trigger logic + notification dispatch."""

    def __init__(self, config: Any = None, notifier: Optional[MultiNotifier] = None,
                 on_event: Optional[Callable[[AlarmEvent], None]] = None):
        cfg = (lambda path, default: config.get_path(path, default) if config else default)
        self.config = config
        self.notifier = notifier or MultiNotifier()
        self.on_event = on_event

        self.enabled = bool(cfg("alarm.enabled", False))
        self.trigger_on = str(cfg("alarm.trigger_on", "presence"))       # presence | motion
        self.min_confidence = float(cfg("alarm.min_confidence", 70))
        self.sustain_s = float(cfg("alarm.sustain_s", 1.5))
        self.cooldown_s = float(cfg("alarm.cooldown_s", 120))
        self.clear_s = float(cfg("alarm.clear_s", 25))
        self.arm_delay_s = float(cfg("alarm.arm_delay_s", 30))
        self.notify_clear = bool(cfg("alarm.notify_clear", False))

        self.state = ARMED_AWAY if bool(cfg("alarm.start_armed", False)) else DISARMED
        self.armed_at: Optional[float] = None
        self.last_trigger: Optional[float] = None
        self.last_clear: Optional[float] = None
        self.trigger_count = 0
        self.suppressed_count = 0
        self._sustain_start: Optional[float] = None
        self._clear_start: Optional[float] = None
        self._triggered = False
        self.events: List[AlarmEvent] = []
        self._event_limit = 1000

    # ------------------------------------------------------------------ #
    @property
    def armed(self) -> bool:
        return self.state in (ARMED_AWAY, ARMED_HOME)

    @property
    def exit_delay_remaining(self) -> float:
        if self.state != ARMING or self.armed_at is None:
            return 0.0
        return max(0.0, self.arm_delay_s - (time.time() - self.armed_at))

    def arm(self, mode: str = "away", delay: Optional[float] = None,
            now: Optional[float] = None) -> AlarmEvent:
        now = float(now if now is not None else time.time())
        delay = self.arm_delay_s if delay is None else float(delay)
        self.armed_at = now
        self._sustain_start = None
        self._triggered = False
        if delay > 0:
            self.state = ARMING
            event = AlarmEvent(now, "arming", "info",
                               f"arming in {delay:.0f}s ({mode}) — exit the room now")
        else:
            self.state = ARMED_AWAY if mode == "away" else ARMED_HOME
            event = AlarmEvent(now, "armed", "info", f"armed ({self._mode_label()})")
        return self._emit(event, notify=False)

    def disarm(self, now: Optional[float] = None) -> AlarmEvent:
        now = float(now if now is not None else time.time())
        self.state = DISARMED
        self.armed_at = None
        self._sustain_start = None
        self._clear_start = None
        self._triggered = False
        event = AlarmEvent(now, "disarmed", "info", "alarm disarmed")
        return self._emit(event, notify=False)

    def _mode_label(self) -> str:
        return "away — any presence triggers" if self.state == ARMED_AWAY else "home — motion triggers"

    # ------------------------------------------------------------------ #
    def update(self, det: DetectionState, now: Optional[float] = None,
               context: Optional[Dict[str, Any]] = None) -> List[AlarmEvent]:
        """Feed one detection state; get back the events it caused (possibly none)."""
        now = float(now if now is not None else det.t)
        out: List[AlarmEvent] = []

        if not self.enabled:
            return out

        if self.state == ARMING:
            if self.armed_at is not None and (now - self.armed_at) >= self.arm_delay_s:
                self.state = ARMED_AWAY if self.trigger_on == "presence" else ARMED_HOME
                out.append(self._emit(AlarmEvent(now, "armed", "info",
                                                 f"armed ({self._mode_label()})"), notify=False))
            return out

        if not self.armed:
            return out

        condition = det.presence if self.trigger_on == "presence" else det.motion
        strong = det.confidence >= self.min_confidence

        if condition and strong:
            if self._sustain_start is None:
                self._sustain_start = now
            self._clear_start = None
            sustained = (now - self._sustain_start) >= self.sustain_s
            context = dict(context or {})
            if sustained and not self._triggered:
                self._triggered = True
                self.trigger_count += 1
                self.last_trigger = now
                state = det.to_dict()
                state.update(context)
                message = (f"{det.presence_label} · motion {det.motion_label} · "
                           f"confidence {det.confidence:.0f}%")
                event = AlarmEvent(now, "triggered", "alarm", message, state)
                out.append(self._emit(event, notify=True, det=det, context=context))
            elif sustained and self._triggered:
                if self.last_trigger is None or (now - self.last_trigger) >= self.cooldown_s:
                    self.last_trigger = now
                    state = det.to_dict()
                    state.update(context)
                    out.append(self._emit(
                        AlarmEvent(now, "triggered", "alarm",
                                   f"still active · confidence {det.confidence:.0f}%", state),
                        notify=True, det=det, context=context))
                else:
                    self.suppressed_count += 1
                    out.append(self._emit(
                        AlarmEvent(now, "suppressed", "info",
                                   f"within cooldown ({self.cooldown_s:.0f}s)", det.to_dict()),
                        notify=False))
        else:
            self._sustain_start = None
            if self._triggered:
                if self._clear_start is None:
                    self._clear_start = now
                if (now - self._clear_start) >= self.clear_s:
                    self._triggered = False
                    self.last_clear = now
                    self._clear_start = None
                    out.append(self._emit(
                        AlarmEvent(now, "cleared", "notice",
                                   f"room clear again (confidence {det.confidence:.0f}% empty)",
                                   det.to_dict()),
                        notify=self.notify_clear, det=det))
        return out

    # ------------------------------------------------------------------ #
    def _emit(self, event: AlarmEvent, notify: bool,
              det: Optional[DetectionState] = None,
              context: Optional[Dict[str, Any]] = None) -> AlarmEvent:
        self.events.append(event)
        if len(self.events) > self._event_limit:
            del self.events[:self._event_limit // 2]
        if notify:
            title = {"triggered": "INTRUSION ALARM", "cleared": "Room clear",
                     "armed": "Alarm armed", "disarmed": "Alarm disarmed",
                     "arming": "Arming"}.get(event.kind, event.kind)
            extra: Dict[str, Any] = {}
            if det is not None:
                extra = {
                    "presence": det.presence_label,
                    "motion": det.motion_label,
                    "confidence": f"{det.confidence:.0f}%",
                    "time": time.strftime("%H:%M:%S", time.localtime(event.t)),
                }
            if context:
                extra.update({k: v for k, v in context.items() if k != "features"})
            try:
                self.notifier.send(title, event.message, event.severity, extra)
            except Exception:
                pass
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass
        return event

    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "armed": self.armed,
            "trigger_on": self.trigger_on,
            "min_confidence": self.min_confidence,
            "sustain_s": self.sustain_s,
            "cooldown_s": self.cooldown_s,
            "exit_delay_remaining": round(self.exit_delay_remaining, 1),
            "trigger_count": self.trigger_count,
            "suppressed_count": self.suppressed_count,
            "last_trigger": self.last_trigger,
            "last_clear": self.last_clear,
            "notifiers": self.notifier.health(),
        }

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events[-limit:]]


ATTR_PRESENCE = "presence"
ATTR_MOTION = "motion"
