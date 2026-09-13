"""Zone-level localization: "which part of the room is the disturbance in?"

Honest scope, stated up front: RSSI/CSI presence sensing can tell you that a body
perturbs the radio environment. Turning that into a *position* needs geometry and
a signature per place, and the accuracy you get is **zone-level, not centimetres**:

* One link (laptop/phone ↔ router) → a disturbance anywhere near the line between
  the two changes the signal; the best you can honestly say is "near the
  transmitter path" vs "far from it".
* Multiple links (a phone per family member, or an AP-mode NIC with several
  clients) → each link gives an independent projection, so zone classification
  becomes genuinely useful: "person is in the kitchen-side zone".
* CSI → per-subcarrier phase makes distance/angle estimation possible in
  research setups, still with controlled geometry and calibration.

The method here is honest and simple: during a zone walk-through you record a
feature signature while a person stands in each declared zone; at runtime the
current feature vector is compared to every signature (standardised Euclidean /
cosine mixture) and turned into softmax probabilities, and the reported position
is the probability-weighted centroid of the zone centres. When nothing matches
well the answer is "unknown" — which is the correct answer, not a guess.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

# Features that actually move when a body occupies a zone. Presence-style
# aggregate features carry the "how much" and the spectral split carries the
# "where" (how close to the link the body sits changes which bands are excited).
DEFAULT_LOCALIZATION_FEATURES = (
    "var", "std", "diff_rms", "motion_energy", "band_slow", "band_motion",
    "band_fast", "band_ratio", "dom_freq", "spectral_centroid",
    "spectral_entropy", "perm_entropy", "mean", "median", "range", "kurtosis",
    "amp_var_mean", "amp_var_max", "active_ratio", "subcarrier_corr",
    "phase_var", "phase_diff_var", "csi_motion_index",
)


@dataclass
class Zone:
    name: str
    x: float
    y: float
    z: float = 0.0
    radius: float = 0.8
    label: str = ""

    def center(self) -> Tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


@dataclass
class ZoneSignature:
    name: str
    features: Dict[str, float]
    scale: Dict[str, float]        # robust per-feature scale from the empty baseline
    n_windows: int
    center: Tuple[float, float, float]


@dataclass
class LocalizationResult:
    zone: Optional[str]
    confidence: float               # 0-100, probability of the winning zone
    position: Optional[Tuple[float, float, float]]
    probabilities: Dict[str, float]
    distances: Dict[str, float]
    calibrated: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone": self.zone,
            "confidence": round(self.confidence, 1),
            "position": None if self.position is None else [round(v, 3) for v in self.position],
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "distances": {k: round(v, 4) for k, v in self.distances.items()},
            "calibrated": self.calibrated,
            "reason": self.reason,
        }


class ZoneLocalizer:
    """Signature-matching localizer over declared room zones."""

    def __init__(self, zones: Sequence[Zone], features: Sequence[str] = DEFAULT_LOCALIZATION_FEATURES,
                 min_confidence: float = 35.0, temperature: float = 1.0,
                 unknown_margin: float = 0.08):
        self.zones: List[Zone] = list(zones)
        self.feature_names: List[str] = [f for f in features]
        self.signatures: Dict[str, ZoneSignature] = {}
        self.empty_scale: Dict[str, float] = {}
        self.empty_mean: Dict[str, float] = {}
        self.min_confidence = float(min_confidence)
        self.temperature = float(temperature)
        self.unknown_margin = float(unknown_margin)
        self.created_at = time.time()
        self.host_notes: List[str] = []
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # calibration
    # ------------------------------------------------------------------ #
    def fit(self, zone_rows: Dict[str, List[Dict[str, float]]],
            empty_rows: Optional[List[Dict[str, float]]] = None) -> "ZoneLocalizer":
        """Learn one signature per zone from windows recorded while a person stood there."""
        if not zone_rows:
            raise ValueError("no zone windows given")
        names = [f for f in self.feature_names
                 if any(f in row for rows in zone_rows.values() for row in rows)]
        if len(names) < 3:
            raise ValueError("need at least 3 usable features to localise")
        self.feature_names = names

        if empty_rows:
            self.empty_mean = {f: _median([r.get(f, 0.0) for r in empty_rows]) for f in names}
            self.empty_scale = {
                f: max(_robust_sigma([r.get(f, 0.0) for r in empty_rows]), 1e-3)
                for f in names
            }
        else:
            all_rows = [row for rows in zone_rows.values() for row in rows]
            self.empty_mean = {f: 0.0 for f in names}
            self.empty_scale = {f: max(_robust_sigma([r.get(f, 0.0) for r in all_rows]), 1e-3)
                                for f in names}
            self.host_notes.append("no empty-room windows supplied: zone contrast is weaker")

        centres = {z.name: z.center() for z in self.zones}
        for zone_name, rows in zone_rows.items():
            if not rows:
                continue
            feat = {f: _median([r.get(f, 0.0) for r in rows]) for f in names}
            self.signatures[zone_name] = ZoneSignature(
                name=zone_name, features=feat, scale=dict(self.empty_scale),
                n_windows=len(rows), center=centres.get(zone_name, (0.0, 0.0, 0.0)),
            )

        if len(self.signatures) < 2:
            self.host_notes.append("only one zone had usable data — localization will be trivial")
        self._warn_on_collapsed_zones()
        return self

    def _warn_on_collapsed_zones(self) -> None:
        """Two zones that look identical in feature space make the output a coin flip."""
        if len(self.signatures) < 2:
            return
        names = list(self.signatures)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = self._pair_distance(self.signatures[a].features, self.signatures[b].features)
                if d < 0.35:
                    self.host_notes.append(
                        f"zones '{a}' and '{b}' are nearly indistinguishable (distance {d:.2f}); "
                        f"move them apart, add a second link/sensor, or merge them"
                    )

    # ------------------------------------------------------------------ #
    # runtime
    # ------------------------------------------------------------------ #
    def _standardise(self, feats: Dict[str, float]) -> np.ndarray:
        return np.asarray([
            (float(feats.get(f, 0.0)) - self.empty_mean.get(f, 0.0)) / self.empty_scale.get(f, 1.0)
            for f in self.feature_names
        ], dtype=float)

    def _pair_distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        va = np.asarray([(a.get(f, 0.0) - self.empty_mean.get(f, 0.0)) /
                         self.empty_scale.get(f, 1.0) for f in self.feature_names])
        vb = np.asarray([(b.get(f, 0.0) - self.empty_mean.get(f, 0.0)) /
                         self.empty_scale.get(f, 1.0) for f in self.feature_names])
        return float(np.linalg.norm(va - vb) / math.sqrt(max(len(va), 1)))

    def predict(self, feats: Dict[str, float]) -> LocalizationResult:
        if not self.signatures:
            return LocalizationResult(None, 0.0, None, {}, {}, calibrated=False,
                                      reason="no zone signatures — calibrate zones first")
        v = self._standardise(feats)
        distances: Dict[str, float] = {}
        for name, sig in self.signatures.items():
            u = self._standardise(sig.features)
            euclid = float(np.linalg.norm(v - u) / math.sqrt(len(v)))
            # cosine term catches "same shape, different magnitude" (person
            # standing vs leaning) which pure Euclidean over-penalises
            denom = (np.linalg.norm(v) * np.linalg.norm(u)) or 1.0
            cosine = 1.0 - float(np.dot(v, u) / denom)
            distances[name] = 0.7 * euclid + 0.3 * cosine

        ranked = sorted(distances.items(), key=lambda kv: kv[1])
        best_name, best_d = ranked[0]
        second_d = ranked[1][1] if len(ranked) > 1 else best_d + 1.0
        # softmax over negative distances
        logits = np.asarray([-d / max(self.temperature, 1e-3) for _n, d in ranked], dtype=float)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        probabilities = {name: float(p) for (name, _d), p in zip(ranked, probs)}
        confidence = float(probabilities[best_name] * 100.0)

        # Unknown when the winner is not clearly better than the runner-up, or when
        # nothing looks like any zone at all (person far away / signal artefact).
        margin = (second_d - best_d) / max(second_d, 1e-6)
        if best_d > 3.5:
            return LocalizationResult(None, confidence, None, probabilities, distances,
                                      reason=f"no zone matches the current signature "
                                             f"(best distance {best_d:.2f})")
        if confidence < self.min_confidence or margin < self.unknown_margin:
            return LocalizationResult(None, confidence, None, probabilities, distances,
                                      reason=f"ambiguous between zones "
                                             f"({best_name} vs {ranked[1][0] if len(ranked) > 1 else '-'})")

        position = (0.0, 0.0, 0.0)
        weight_sum = 0.0
        for name, prob in probabilities.items():
            cx, cy, cz = self.signatures[name].center
            position = (position[0] + cx * prob, position[1] + cy * prob, position[2] + cz * prob)
            weight_sum += prob
        if weight_sum > 0:
            position = tuple(c / weight_sum for c in position)

        zone = next((z for z in self.zones if z.name == best_name), None)
        radius = zone.radius if zone else 0.8
        result = LocalizationResult(
            zone=best_name, confidence=confidence, position=position,
            probabilities=probabilities, distances=distances,
            reason=(f"signature matches zone '{best_name}' "
                    f"({confidence:.0f}% vs {probabilities.get(ranked[1][0], 0.0) * 100:.0f}% "
                    f"for the runner-up)" if len(ranked) > 1 else
                    f"matches zone '{best_name}'"),
        )
        result_dict = result.to_dict()
        result_dict["radius"] = radius
        self.history.append(result_dict)
        if len(self.history) > 500:
            del self.history[:250]
        return result

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "created_at": self.created_at,
            "feature_names": self.feature_names,
            "empty_mean": self.empty_mean,
            "empty_scale": self.empty_scale,
            "zones": [{"name": z.name, "x": z.x, "y": z.y, "z": z.z,
                       "radius": z.radius, "label": z.label or z.name} for z in self.zones],
            "signatures": {n: {"features": s.features, "n_windows": s.n_windows,
                               "center": list(s.center)}
                           for n, s in self.signatures.items()},
            "params": {"min_confidence": self.min_confidence, "temperature": self.temperature,
                       "unknown_margin": self.unknown_margin},
            "notes": self.host_notes,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ZoneLocalizer":
        raw = json.loads(Path(path).expanduser().read_text())
        if raw.get("schema") != SCHEMA_VERSION:
            raise ValueError("zone model schema mismatch — re-run --calibrate-zones")
        zones = [Zone(z["name"], z["x"], z["y"], z.get("z", 0.0), z.get("radius", 0.8),
                      z.get("label", "")) for z in raw.get("zones", [])]
        loc = cls(zones)
        loc.feature_names = raw.get("feature_names", loc.feature_names)
        loc.empty_mean = raw.get("empty_mean", {})
        loc.empty_scale = raw.get("empty_scale", {})
        loc.created_at = raw.get("created_at", time.time())
        loc.host_notes = raw.get("notes", [])
        params = raw.get("params", {})
        loc.min_confidence = params.get("min_confidence", loc.min_confidence)
        loc.temperature = params.get("temperature", loc.temperature)
        loc.unknown_margin = params.get("unknown_margin", loc.unknown_margin)
        for name, sig in raw.get("signatures", {}).items():
            loc.signatures[name] = ZoneSignature(
                name=name, features=sig.get("features", {}), scale=loc.empty_scale,
                n_windows=sig.get("n_windows", 0), center=tuple(sig.get("center", (0, 0, 0))),
            )
        return loc

    def describe(self) -> str:
        lines = [f"zones: {len(self.signatures)} signatures "
                 f"({', '.join(sorted(self.signatures)) or 'none'})",
                 f"features: {len(self.feature_names)} · min confidence {self.min_confidence:.0f}%"]
        for note in self.host_notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def _median(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else 0.0


def _robust_sigma(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size < 2:
        return 0.0
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


def zones_from_config(config: Any) -> List[Zone]:
    """Read the ``zones:`` block; fall back to a sensible 4-zone demo layout."""
    raw = (config.get_path("zones", None) if config else None)
    if not raw:
        return default_zones(config)
    zones: List[Zone] = []
    for entry in raw:
        if isinstance(entry, str):
            zones.append(Zone(entry, 0.0, 0.0))
            continue
        zones.append(Zone(
            name=str(entry.get("name")),
            x=float(entry.get("x", 0.0)),
            y=float(entry.get("y", 0.0)),
            z=float(entry.get("z", 0.0)),
            radius=float(entry.get("radius", 0.8)),
            label=str(entry.get("label", entry.get("name"))),
        ))
    return zones


def default_zones(config: Any = None) -> List[Zone]:
    """A 4-quadrant split of the room box — works without any configuration."""
    w = float(config.get_path("room.width_m", 5.0) if config else 5.0)
    d = float(config.get_path("room.depth_m", 4.0) if config else 4.0)
    qx, qy = w / 4.0, d / 4.0
    radius = min(w, d) / 3.2
    return [
        Zone("near_ap", qx, qy, 0.0, radius, "dekat router"),
        Zone("far_ap", w - qx, d - qy, 0.0, radius, "jauh dari router"),
        Zone("left", qx, d - qy, 0.0, radius, "sisi kiri"),
        Zone("right", w - qx, qy, 0.0, radius, "sisi kanan"),
    ]


def room_geometry(config: Any) -> Dict[str, Any]:
    """Room box + device positions, used by the 3D view and the physics sim."""
    cfg = (lambda path, default: config.get_path(path, default) if config else default)
    width = float(cfg("room.width_m", 5.0))
    depth = float(cfg("room.depth_m", 4.0))
    height = float(cfg("room.height_m", 2.7))
    ap = cfg("room.ap_position", None) or [width - 0.3, depth - 0.3, height - 0.6]
    sensor = cfg("room.sensor_position", None) or [0.3, 0.3, 1.0]
    return {
        "width_m": width, "depth_m": depth, "height_m": height,
        "ap": [float(v) for v in ap],
        "sensor": [float(v) for v in sensor],
        "assumed": cfg("room.ap_position", None) is None,
    }


def zone_model_path(config: Any, interface: Optional[str] = None) -> Path:
    root = Path(config.get_path("localization.dir", "models") if config else "models")
    return root / f"zones_{interface or 'default'}.json"


def load_localizer(config: Any, interface: Optional[str] = None,
                   explicit: Optional[str] = None) -> Optional[ZoneLocalizer]:
    path = Path(explicit).expanduser() if explicit else zone_model_path(config, interface)
    if not path.exists():
        return None
    try:
        return ZoneLocalizer.load(path)
    except Exception:
        return None
