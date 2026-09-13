"""Calibration / baseline model.

Detection here is *relative to a measured empty room*, never absolute. That is
the whole reason calibration exists: RSSI depends on the AP's transmit power,
the wall between you and it, the channel, the NIC's AGC curve and the time of
day. A fixed threshold like "variance > 0.1 dBm² means a human" is meaningless
across rooms and fails silently when someone moves the router.

So: record the room while it is empty, estimate the noise scale of every feature
robustly (median + MAD, not mean + std — one truck passing outside must not
raise the ceiling for the rest of the night), and derive the thresholds from
that.

The baseline is also *drift-aware*: while the detector says the room is empty it
nudges the baseline toward the observed level, so a slow thermal drift over
hours does not eventually read as an intruder.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2

# Which features carry the presence signal, and which carry motion. Names match
# processing.features.FEATURE_ORDER / CSI_FEATURE_ORDER.
PRESENCE_FEATURES: Sequence[str] = (
    "var", "std", "diff_rms", "motion_energy", "band_motion", "band_ratio",
    "spectral_flatness", "amp_var_mean", "amp_var_max", "csi_motion_index",
)
MOTION_FEATURES: Sequence[str] = (
    "diff_rms", "motion_energy", "band_motion", "band_ratio", "zcr",
    "csi_motion_index", "phase_diff_var",
)


@dataclass
class FeatureStats:
    """Robust scale estimates for one feature during the empty-room capture."""

    median: float = 0.0
    sigma: float = 0.0          # 1.4826 * MAD
    mean: float = 0.0
    std: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    n: int = 0

    def z(self, value: float, floor: float = 1e-3, cap: float = 25.0) -> float:
        """Robust z-score, with a sigma floor so a perfectly quiet feature does
        not produce infinite scores (a dead-flat feature is evidence of a broken
        backend, not of a ghost)."""
        sigma = max(float(self.sigma), floor)
        z = (float(value) - self.median) / sigma
        if z < 0:
            z = 0.0  # presence can only push these features up
        return min(z, cap)


@dataclass
class Baseline:
    created_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    n_windows: int = 0
    mode: str = "rssi"
    source: str = "unknown"
    interface: Optional[str] = None
    sample_rate: float = 0.0
    features: Dict[str, FeatureStats] = field(default_factory=dict)
    presence_k: float = 3.0
    motion_k: float = 2.2
    noise_floor: float = 0.0
    schema: int = SCHEMA_VERSION
    host: str = field(default_factory=platform.node)
    notes: List[str] = field(default_factory=list)

    # ---- derived thresholds ------------------------------------------- #
    @property
    def presence_thresholds(self) -> Dict[str, float]:
        return {name: self.features[name].median + self.presence_k * max(self.features[name].sigma, 1e-3)
                for name in PRESENCE_FEATURES if name in self.features}

    @property
    def motion_thresholds(self) -> Dict[str, float]:
        return {name: self.features[name].median + self.motion_k * max(self.features[name].sigma, 1e-3)
                for name in MOTION_FEATURES if name in self.features}

    # ---- persistence ---------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["presence_thresholds"] = self.presence_thresholds
        data["motion_thresholds"] = self.motion_thresholds
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        path = Path(path).expanduser()
        raw = json.loads(path.read_text())
        if raw.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"baseline schema mismatch (file={raw.get('schema')}, expected={SCHEMA_VERSION}); "
                f"re-calibrate with --calibrate"
            )
        feats = {name: FeatureStats(**stats) for name, stats in raw.get("features", {}).items()}
        return cls(
            created_at=raw.get("created_at", time.time()),
            duration_s=raw.get("duration_s", 0.0),
            n_windows=raw.get("n_windows", 0),
            mode=raw.get("mode", "rssi"),
            source=raw.get("source", "unknown"),
            interface=raw.get("interface"),
            sample_rate=raw.get("sample_rate", 0.0),
            features=feats,
            presence_k=raw.get("presence_k", 3.0),
            motion_k=raw.get("motion_k", 2.2),
            noise_floor=raw.get("noise_floor", 0.0),
            schema=raw.get("schema", SCHEMA_VERSION),
            host=raw.get("host", "?"),
            notes=raw.get("notes", []),
        )

    def describe(self) -> str:
        age = time.time() - self.created_at
        lines = [
            f"baseline: {self.n_windows} windows over {self.duration_s:.1f}s "
            f"({self.mode}/{self.source}, iface={self.interface}) on {self.host}",
            f"captured {age/60:.1f} min ago · noise floor {self.noise_floor:.4g}",
            f"thresholds: presence_k={self.presence_k} motion_k={self.motion_k}",
        ]
        for name in ("var", "diff_rms", "motion_energy", "band_motion"):
            if name in self.features:
                st = self.features[name]
                lines.append(f"  {name:<14} med={st.median:9.4g}  sigma={st.sigma:8.4g}  "
                             f"thr={st.median + self.presence_k * max(st.sigma, 1e-3):9.4g}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #

def _stats(values: Sequence[float]) -> FeatureStats:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return FeatureStats()
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return FeatureStats(
        median=med,
        sigma=1.4826 * mad,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(arr.min()),
        max=float(arr.max()),
        n=int(arr.size),
    )


def compute_baseline(feature_rows: List[Dict[str, float]], presence_k: float = 3.0,
                     motion_k: float = 2.2, mode: str = "rssi", source: str = "unknown",
                     interface: Optional[str] = None, sample_rate: float = 0.0,
                     duration_s: float = 0.0, notes: Optional[List[str]] = None) -> Baseline:
    """Build a :class:`Baseline` from a list of feature dicts.

    ``notes`` is surfaced in the dashboard and README so a calibration taken
    with the cat still in the room can be labelled as such instead of quietly
    producing a detector that ignores cats.
    """
    if not feature_rows:
        raise ValueError("cannot build a baseline from zero windows")

    names: set[str] = set()
    for row in feature_rows:
        names.update(row.keys())
    names -= {"n", "fs", "duration_s", "presence_stat"}  # bookkeeping, not signal

    features = {name: _stats([row.get(name, 0.0) for row in feature_rows]) for name in sorted(names)}

    baseline = Baseline(
        duration_s=duration_s,
        n_windows=len(feature_rows),
        mode=mode,
        source=source,
        interface=interface,
        sample_rate=sample_rate,
        features=features,
        presence_k=presence_k,
        motion_k=motion_k,
        notes=list(notes or []) + _quality_notes(feature_rows),
    )
    baseline.noise_floor = float(features.get("band_fast", FeatureStats()).median)
    return baseline


def _quality_notes(rows: List[Dict[str, float]]) -> List[str]:
    """Warn about a calibration that will produce a useless detector."""
    notes: List[str] = []
    if len(rows) < 20:
        notes.append(f"only {len(rows)} windows captured — a short calibration makes "
                     f"wide thresholds and a deaf detector")
    key = [row.get("diff_rms", 0.0) for row in rows]
    if key and max(key) > 4 * (statistics.median(key) + 1e-9):
        notes.append("large activity spike during calibration — the room was probably not "
                     "empty (baseline inflated, sensitivity reduced)")
    if len(rows) > 5:
        drift = abs(rows[-1].get("mean", 0.0) - rows[0].get("mean", 0.0))
        if drift > 6.0:
            notes.append(f"RSSI drifted {drift:.1f} dBm during calibration — re-calibrate "
                         f"or expect a higher threshold")
    return notes


# --------------------------------------------------------------------------- #
# live drift adaptation
# --------------------------------------------------------------------------- #

class DriftAdapter:
    """Slow EMA re-centring of the baseline while the room is believed empty.

    Keeps the detector honest across temperature swings and AP retunes without
    ever adapting *toward* a person: adaptation only runs on windows the engine
    classified as empty, and is capped at ``max_shift`` dBm per window so a
    sustained presence cannot walk the baseline up to itself.
    """

    def __init__(self, baseline: Baseline, rate: float = 0.02, max_shift: float = 0.4,
                 enabled: bool = True):
        self.baseline = baseline
        self.rate = float(rate)
        self.max_shift = float(max_shift)
        self.enabled = bool(enabled)
        self.shift = 0.0
        self.updates = 0

    def update(self, features: Dict[str, float], is_empty: bool) -> bool:
        if not self.enabled or not is_empty:
            return False
        changed = False
        for name in ("mean", "median"):
            if name not in features or name not in self.baseline.features:
                continue
            current = self.baseline.features[name].median
            delta = (features[name] - current) * self.rate
            delta = max(-self.max_shift, min(self.max_shift, delta))
            if abs(delta) < 1e-4:
                continue
            self.baseline.features[name].median = current + delta
            self.shift += delta
            changed = True
        if changed:
            self.updates += 1
        return changed


def baseline_path(config: Any, interface: Optional[str] = None,
                  mode: str = "rssi") -> Path:
    """Where this machine's calibration for this interface lives."""
    root = Path(config.get_path("calibration.dir", "models") if config else "models")
    tag = f"{mode}_{interface or 'default'}"
    return root / f"baseline_{tag}.json"


def load_baseline(config: Any, interface: Optional[str] = None,
                  mode: str = "rssi", explicit: Optional[str] = None) -> Optional[Baseline]:
    path = Path(explicit).expanduser() if explicit else baseline_path(config, interface, mode)
    if not path.exists():
        return None
    try:
        return Baseline.load(path)
    except Exception:
        return None
