"""Detection engine: features + baseline -> presence / motion / confidence.

Design rules that keep this honest:

* **Nothing is absolute.** Every score is a robust z-score against *this* room's
  calibrated empty conditions. No "variance > 0.1 means human" magic constants.
* **Presence and motion are different questions.** Presence = "is the signal
  environment perturbed at all" (variance/level-shape deviation). Motion =
  "is it changing fast right now" (derivative energy, motion band power). A
  person sitting still reads PRESENT / NO MOTION. That distinction matters for a
  thief detector: a burglar standing still in the dark is presence without
  motion, and that is exactly the case you must not miss.
* **Hysteresis + hold** stop the output from strobing: 1 window of noise must not
  raise the alarm, and a decision must survive the gaps between bursts.
* **Confidence is a calibrated probability-shaped score**, reported for whichever
  state is being shown, so "NO HUMAN — 92%" and "HUMAN DETECTED — 87%" are
  comparable statements rather than decoration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .baseline import MOTION_FEATURES, PRESENCE_FEATURES, Baseline, DriftAdapter


@dataclass
class DetectionState:
    t: float
    presence: bool = False
    motion: bool = False
    confidence: float = 0.0           # % confidence in the reported state
    presence_score: float = 0.0       # robust z, 0 = at baseline
    motion_score: float = 0.0
    presence_ratio: float = 0.0       # score / threshold
    motion_ratio: float = 0.0
    holding: bool = False             # presence is being held by the hold timer
    reason: str = ""
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    calibrated: bool = True
    method: str = "adaptive"

    @property
    def presence_label(self) -> str:
        return "HUMAN DETECTED" if self.presence else "NO HUMAN"

    @property
    def motion_label(self) -> str:
        return "DETECTED" if self.motion else "NONE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "presence": self.presence,
            "motion": self.motion,
            "presence_label": self.presence_label,
            "motion_label": self.motion_label,
            "confidence": round(self.confidence, 1),
            "presence_score": round(self.presence_score, 3),
            "motion_score": round(self.motion_score, 3),
            "presence_ratio": round(self.presence_ratio, 3),
            "motion_ratio": round(self.motion_ratio, 3),
            "holding": self.holding,
            "reason": self.reason,
            "top_features": [{"name": n, "z": round(z, 2)} for n, z in self.top_features],
            "calibrated": self.calibrated,
            "method": self.method,
        }


def _combine(zs: Dict[str, float], weights: Optional[Dict[str, float]] = None,
             top_n: int = 4) -> float:
    """Combine per-feature z-scores.

    Mean of the top-N is used instead of max (one noisy feature could otherwise
    trigger alone) and instead of the mean of all (weak features would dilute a
    strong one). Ties are broken by a small weight table so physics-informed
    features (variance, derivative energy) count more than proxies.
    """
    if not zs:
        return 0.0
    weights = weights or {}
    ranked = sorted(zs.items(), key=lambda kv: kv[1] * weights.get(kv[0], 1.0), reverse=True)
    chosen = ranked[:top_n]
    total_w = 0.0
    acc = 0.0
    for name, z in chosen:
        w = weights.get(name, 1.0)
        acc += z * w
        total_w += w
    return acc / total_w if total_w else 0.0


WEIGHTS: Dict[str, float] = {
    "var": 1.4, "std": 1.1, "diff_rms": 1.5, "motion_energy": 1.5,
    "band_motion": 1.3, "band_ratio": 1.0, "spectral_flatness": 0.7,
    "zcr": 0.9, "amp_var_mean": 1.4, "amp_var_max": 1.0, "csi_motion_index": 1.5,
    "phase_diff_var": 1.2,
}


class DetectionEngine:
    """Turns a stream of feature dicts into decided states."""

    def __init__(self, config: Any = None, baseline: Optional[Baseline] = None,
                 method: str = "adaptive", model: Any = None,
                 adaptive_drift: float = 0.02):
        self.config = config
        self.baseline = baseline
        self.method = method
        self.model = model
        cfg = (lambda path, default: config.get_path(path, default) if config else default)
        self.presence_k = float(cfg("detection.presence_k", baseline.presence_k if baseline else 3.0))
        self.motion_k = float(cfg("detection.motion_k", baseline.motion_k if baseline else 2.2))
        self.hyst_up = int(cfg("detection.hysteresis_up", 2))
        self.hyst_down = int(cfg("detection.hysteresis_down", 5))
        self.hold_s = float(cfg("detection.hold_s", 6.0))
        self.min_confidence = float(cfg("detection.min_confidence", 35))
        self.adapter = DriftAdapter(baseline, rate=adaptive_drift) if baseline else None

        self._up = 0
        self._down = 0
        self._presence = False
        self._motion = False
        self._last_hit: Optional[float] = None
        self._motion_until = 0.0
        self.history: List[DetectionState] = []
        self.history_limit = int(cfg("dashboard.history_points", 900))
        self._fallback_stats = self._accumulate_fallback()

    # Public hysteresis controls (tests and callers use these names).
    @property
    def hysteresis_up(self) -> int:
        return self.hyst_up

    @hysteresis_up.setter
    def hysteresis_up(self, value: int) -> None:
        self.hyst_up = int(value)

    @property
    def hysteresis_down(self) -> int:
        return self.hyst_down

    @hysteresis_down.setter
    def hysteresis_down(self, value: int) -> None:
        self.hyst_down = int(value)

    # ------------------------------------------------------------------ #
    def _accumulate_fallback(self) -> Dict[str, float]:
        """Without a baseline we still must produce something useful.

        A short running estimate of the empty-room scale is accumulated from the
        first windows. It is weaker than a real calibration and the engine says
        so (``calibrated=False``) instead of pretending otherwise.
        """
        return {"var": 0.0, "diff_rms": 0.0, "n": 0.0}

    # ------------------------------------------------------------------ #
    def features_ready(self) -> bool:
        return self.baseline is not None or self._fallback_stats["n"] >= 30

    def _fallback_stats_update(self, feats: Dict[str, float]) -> None:
        n = self._fallback_stats["n"]
        for key in ("var", "diff_rms"):
            value = float(feats.get(key, 0.0))
            # exponential running mean with a floor, i.e. "what the quiet parts
            # of the last minute looked like"
            alpha = 1.0 / max(n + 1, 1)
            if n < 30 or value < self._fallback_stats[key] * 1.5:
                self._fallback_stats[key] = (1 - alpha) * self._fallback_stats[key] + alpha * value
        self._fallback_stats["n"] = n + 1

    def _z_scores(self, feats: Dict[str, float], names: Sequence[str]) -> Dict[str, float]:
        zs: Dict[str, float] = {}
        if self.baseline is not None:
            for name in names:
                if name in feats and name in self.baseline.features:
                    zs[name] = self.baseline.features[name].z(feats[name])
            return zs
        # fallback: crude normalisation against the adaptive running estimate
        n = max(self._fallback_stats["n"], 1.0)
        for name in names:
            if name not in feats:
                continue
            if name == "var":
                scale = max(self._fallback_stats["var"], 1e-3)
                zs[name] = max(0.0, float(feats[name]) / scale - 1.0) * 2.0
            elif name == "diff_rms":
                scale = max(self._fallback_stats["diff_rms"], 1e-3)
                zs[name] = max(0.0, float(feats[name]) / scale - 1.0) * 2.0
            elif name in ("diff_mean_abs", "std", "motion_energy"):
                # derived from var: reuse the var score, slightly discounted
                if "var" in zs:
                    zs[name] = zs["var"] * 0.8
        return zs

    # ------------------------------------------------------------------ #
    def update(self, feats: Dict[str, float], t: Optional[float] = None) -> DetectionState:
        now = float(t if t is not None else time.time())
        if self.method == "ml" and self.model is not None:
            state = self._update_ml(feats, now)
            self._record(state)
            return state

        presence_zs = self._z_scores(feats, PRESENCE_FEATURES)
        motion_zs = self._z_scores(feats, MOTION_FEATURES)

        raw_presence = _combine(presence_zs, WEIGHTS)
        raw_motion = _combine(motion_zs, WEIGHTS)

        pr = raw_presence / self.presence_k if self.presence_k > 0 else 0.0
        mr = raw_motion / self.motion_k if self.motion_k > 0 else 0.0

        # --- hysteresis on the presence decision ---
        was_present = self._presence
        above = pr >= 1.0
        if above:
            self._up += 1
            self._down = 0
        else:
            self._down += 1
            self._up = 0

        if not self._presence and self._up >= self.hyst_up:
            self._presence = True
        elif self._presence and self._down >= self.hyst_down:
            self._presence = False

        if self._presence:
            self._last_hit = now

        # --- hold: keep an ALREADY ASSERTED presence alive for hold_s ---
        # Deliberately not applied on the first window that merely crosses the
        # threshold: hold extends a detection, it must never fabricate one.
        holding = False
        if (not self._presence and was_present and self._last_hit is not None
                and (now - self._last_hit) <= self.hold_s):
            self._presence = True
            holding = True

        # --- motion decision ---
        motion_hit = mr >= 1.0
        if motion_hit:
            self._motion_until = now + 1.0
        self._motion = motion_hit or now < self._motion_until
        if self._motion:
            self._presence = True
            if self._last_hit is None or now > (self._last_hit or 0):
                self._last_hit = now

        # --- confidence ---
        score = max(pr, mr if self._motion else 0.0)
        if self._presence:
            confidence = 100.0 * (1.0 - math.exp(-0.7 * max(score, 0.0)))
            confidence = max(confidence, self.min_confidence if score >= 1.0 else confidence)
        else:
            # confidence in "empty": how far below threshold we are
            quiet = max(0.0, 1.0 - score)
            confidence = 100.0 * (0.5 + 0.5 * (1.0 - math.exp(-3.0 * quiet)))
        confidence = float(min(99.9, max(0.0, confidence)))

        top = sorted(presence_zs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        reason = self._reason(pr, mr, holding)

        state = DetectionState(
            t=now,
            presence=self._presence,
            motion=self._motion,
            confidence=confidence,
            presence_score=raw_presence,
            motion_score=raw_motion,
            presence_ratio=pr,
            motion_ratio=mr,
            holding=holding,
            reason=reason,
            top_features=[(n, z) for n, z in top],
            calibrated=self.baseline is not None,
            method=self.method,
        )

        if self.adapter is not None and not self._presence and not holding:
            self.adapter.update(feats, is_empty=True)
        self._record(state)
        return state

    def _reason(self, pr: float, mr: float, holding: bool) -> str:
        if self._motion:
            return f"motion energy {mr:.1f}× the empty-room threshold"
        if self._presence and holding:
            return f"holding presence ({self.hold_s:.0f}s decay after last event)"
        if self._presence:
            return f"signal environment perturbed at {pr:.1f}× the empty-room threshold"
        if pr > 0.7:
            return "near threshold — watching"
        return "signal consistent with the calibrated empty room"

    def _record(self, state: DetectionState) -> None:
        self.history.append(state)
        if len(self.history) > self.history_limit:
            del self.history[:len(self.history) - self.history_limit]

    # ------------------------------------------------------------------ #
    def _update_ml(self, feats: Dict[str, float], now: float) -> DetectionState:
        from .ml import predict_one

        proba, motion_prob = predict_one(self.model, feats)
        presence = proba >= 0.5
        motion = motion_prob >= 0.5 and presence
        self._presence = presence
        self._motion = motion
        score = proba
        confidence = 100.0 * (proba if presence else (1.0 - proba))
        return DetectionState(
            t=now, presence=presence, motion=motion,
            confidence=float(min(99.9, max(0.0, confidence))),
            presence_score=score * self.presence_k,
            motion_score=motion_prob * self.motion_k,
            presence_ratio=proba * 2.0,
            motion_ratio=motion_prob * 2.0,
            reason=f"classifier probability {proba:.2f}",
            top_features=[(n, float(feats.get(n, 0.0))) for n in list(feats)[:3]],
            calibrated=True, method="ml",
        )

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._up = self._down = 0
        self._presence = self._motion = False
        self._last_hit = None
        self._motion_until = 0.0
        self.history.clear()

    def summary(self) -> Dict[str, Any]:
        total = len(self.history)
        present = sum(1 for s in self.history if s.presence)
        return {
            "windows": total,
            "presence_windows": present,
            "presence_ratio": (present / total) if total else 0.0,
            "calibrated": self.baseline is not None,
            "presence_k": self.presence_k,
            "motion_k": self.motion_k,
            "hold_s": self.hold_s,
        }
