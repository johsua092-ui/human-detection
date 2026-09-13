"""Feature extraction from RSSI / CSI windows.

Two extractors, one output shape:

* :class:`RSSIFeatureExtractor` — statistics, dynamics and spectral content of a
  single RSSI time series.
* :class:`CSIFeatureExtractor` — per-subcarrier statistics plus phase dynamics.

The feature dict is the contract between acquisition and detection, and the
column set for the ML classifier, so the names are stable and documented in
``FEATURE_ORDER``.

What these features can and cannot see (the honest version): RSSI variance and
its rate-of-change track *bulk motion and presence* in the propagation path.
Spectral bands let a stationary-but-breathing person show up as low-frequency
power, but on a single RSSI stream that signal is near the noise floor — treat
the breathing band as a hint, not a measurement. Real respiration is a CSI
feature; see ``wifisense/experimental/breathing.py``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import filters

# Order matters: it is the model input order and the CSV column order.
FEATURE_ORDER: Tuple[str, ...] = (
    "n", "duration_s", "fs",
    "mean", "median", "std", "var", "range", "iqr", "mad", "skew", "kurtosis",
    "slope", "diff_mean_abs", "diff_rms", "jerk_rms", "zcr", "perm_entropy",
    "band_slow", "band_motion", "band_fast", "band_ratio", "dom_freq",
    "spectral_centroid", "spectral_entropy", "spectral_flatness",
    "motion_energy", "presence_stat",
)

CSI_FEATURE_ORDER: Tuple[str, ...] = (
    "amp_mean", "amp_var_mean", "amp_var_max", "amp_var_std", "amp_iqr",
    "active_subcarriers", "active_ratio", "subcarrier_corr",
    "phase_var", "phase_diff_var", "phase_spread", "csi_motion_index",
    "rssi", "n_subcarriers",
)


def _safe(value: float) -> float:
    """Keep JSON/JSONL output finite: inf/nan would break the API contract."""
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Bandt-Pompe permutation entropy, normalised to [0, 1].

    Cheap, robust to monotone distortion, and a good "is this signal structured
    or is it noise" indicator: near 1.0 means the window looks like noise,
    a drop means a repeated pattern appeared.
    """
    x = np.asarray(x, dtype=float)
    n = x.size - (order - 1) * delay
    if n < 2:
        return 0.0
    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(n):
        window = x[i:i + (order - 1) * delay + 1:delay]
        key = tuple(np.argsort(window, kind="stable"))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.fromiter(patterns.values(), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    n_patterns = math.factorial(order)
    return _safe(entropy / math.log(n_patterns)) if n_patterns > 1 else 0.0


def signal_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    total = psd.sum()
    if total <= 1e-16:
        return 0.0
    p = psd / total
    p = p[p > 0]
    return _safe(-np.sum(p * np.log2(p)) / math.log2(len(psd))) if len(psd) > 1 else 0.0


def spectral_flatness(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    psd = psd[psd > 0]
    if psd.size < 2:
        return 0.0
    return _safe(np.exp(np.log(psd).mean()) / max(psd.mean(), 1e-16))


# --------------------------------------------------------------------------- #
# RSSI features
# --------------------------------------------------------------------------- #

def extract_rssi_features(t: Sequence[float], rssi: Sequence[float], fs: float,
                          bands: Optional[Dict[str, Sequence[float]]] = None,
                          spectral: bool = True,
                          nperseg: int = 128) -> Dict[str, float]:
    """Full feature set for one window.

    ``bands`` maps a name to ``[low_hz, high_hz]``; defaults to the slow/motion/
    fast split from ``config/default.yaml``.
    """
    bands = bands or {"slow": (0.05, 0.30), "motion": (0.30, 2.00), "fast": (2.00, 8.00)}
    t = np.asarray(t, dtype=float)
    x = np.asarray(rssi, dtype=float)
    mask = np.isfinite(x)
    t, x = t[mask], x[mask]
    feats: Dict[str, float] = {name: 0.0 for name in FEATURE_ORDER}
    feats["n"] = float(x.size)
    if x.size < 4:
        return feats

    # Uniform grid: required for every spectral feature.
    grid, xu = filters.resample_uniform(t, x, fs)
    if xu.size < 4:
        xu = x
        grid = np.arange(x.size) / max(fs, 1.0)
    feats["duration_s"] = _safe(grid[-1] - grid[0]) if grid.size > 1 else 0.0
    feats["fs"] = _safe(fs)

    # Level statistics on the raw window.
    feats["mean"] = _safe(xu.mean())
    feats["median"] = _safe(np.median(xu))
    feats["std"] = _safe(xu.std(ddof=1) if xu.size > 1 else 0.0)
    feats["var"] = _safe(np.var(xu, ddof=1) if xu.size > 1 else 0.0)
    feats["range"] = _safe(np.ptp(xu))
    q75, q25 = np.percentile(xu, [75, 25])
    feats["iqr"] = _safe(q75 - q25)
    feats["mad"] = _safe(np.median(np.abs(xu - np.median(xu))))
    if xu.size > 2 and xu.std() > 1e-12:
        z = (xu - xu.mean()) / xu.std()
        feats["skew"] = _safe(np.mean(z ** 3))
        feats["kurtosis"] = _safe(np.mean(z ** 4) - 3.0)

    # Dynamics on the detrended signal: this is what "motion" actually is.
    detrended = filt = filters.detrend_linear(xu, grid) if xu.size > 3 else xu - xu.mean()
    if filt.size > 4:
        filt = filters.hampel_filter(filt, window=9, n_sigma=3.0)
    diff = np.diff(filt) if filt.size > 1 else np.zeros(1)
    feats["diff_mean_abs"] = _safe(np.mean(np.abs(diff)))
    feats["diff_rms"] = _safe(np.sqrt(np.mean(diff ** 2)))
    if diff.size > 1:
        feats["jerk_rms"] = _safe(np.sqrt(np.mean(np.diff(diff) ** 2)))
    if filt.size > 3 and np.ptp(filt) > 1e-12:
        centred = filt - filt.mean()
        crossings = np.sum(np.diff(np.signbit(centred)) != 0)
        feats["zcr"] = _safe(crossings / max(feats["duration_s"], 1e-6))
        feats["perm_entropy"] = permutation_entropy(filt, order=3)

    # Slope (slow drift) from the original window.
    if grid.size > 2 and np.ptp(grid) > 0:
        feats["slope"] = _safe(np.polyfit(grid, xu, 1)[0])

    # Spectral features.
    if spectral and xu.size >= 16:
        freqs, psd = filters.welch_psd(filt, fs=fs, nperseg=nperseg)
        for name, (low, high) in bands.items():
            key = f"band_{name}"
            if key in feats:
                feats[key] = _safe(filters.band_power(freqs, psd, float(low), float(high)))
        motion_power = feats.get("band_motion", 0.0)
        total_power = feats.get("band_slow", 0.0) + motion_power + feats.get("band_fast", 0.0)
        feats["band_ratio"] = _safe(motion_power / total_power) if total_power > 1e-12 else 0.0
        if psd.size and psd.max() > 1e-18:
            feats["dom_freq"] = _safe(freqs[int(np.argmax(psd))])
        if freqs.size > 1:
            band_mask = (freqs >= 0.05) & (freqs <= 10.0)
            p = psd[band_mask]
            f = freqs[band_mask]
            if p.sum() > 1e-18:
                feats["spectral_centroid"] = _safe(float(np.sum(f * p) / np.sum(p)))
        feats["spectral_entropy"] = signal_entropy(psd)
        feats["spectral_flatness"] = spectral_flatness(psd)

    # Composite motion energy: RMS of the derivative, normalised by window length.
    feats["motion_energy"] = _safe(feats["diff_rms"] * math.sqrt(max(feats["duration_s"], 1e-3)))
    # `presence_stat` gets overwritten by the detector with the calibrated score;
    # keeping a raw fallback here means an uncalibrated run still produces output.
    feats["presence_stat"] = _safe(feats["var"] ** 0.5 + feats["diff_rms"])
    return feats


# --------------------------------------------------------------------------- #
# CSI features
# --------------------------------------------------------------------------- #

def extract_csi_features(amp: np.ndarray, phase: Optional[np.ndarray] = None,
                         rssi: Optional[float] = None) -> Dict[str, float]:
    """Per-subcarrier statistics over a window of CSI frames.

    ``amp`` shape: ``(n_frames, n_subcarriers)``. Motion shows up as *correlated*
    variance across subcarriers, while a stationary body keeps most subcarriers
    steady except a few; that contrast is what ``active_ratio`` and
    ``subcarrier_corr`` measure.
    """
    amp = np.asarray(amp, dtype=float)
    feats: Dict[str, float] = {name: 0.0 for name in CSI_FEATURE_ORDER}
    if amp.ndim == 1:
        amp = amp.reshape(1, -1)
    if amp.size == 0:
        return feats
    n_frames, n_sub = amp.shape
    feats["n_subcarriers"] = float(n_sub)
    feats["rssi"] = _safe(rssi) if rssi is not None else 0.0
    feats["amp_mean"] = _safe(np.nanmean(amp))

    if n_frames > 1:
        per_sub_var = np.nanvar(amp, axis=0, ddof=1)
    else:
        per_sub_var = np.zeros(n_sub)
    feats["amp_var_mean"] = _safe(np.nanmean(per_sub_var)) if n_sub else 0.0
    feats["amp_var_max"] = _safe(np.nanmax(per_sub_var)) if n_sub else 0.0
    feats["amp_var_std"] = _safe(np.nanstd(per_sub_var)) if n_sub else 0.0
    if n_sub > 1:
        q75, q25 = np.nanpercentile(per_sub_var, [75, 25])
        feats["amp_iqr"] = _safe(q75 - q25)

    if n_sub:
        thresh = feats["amp_var_mean"] + feats["amp_var_std"]
        active = int(np.sum(per_sub_var > thresh))
        feats["active_subcarriers"] = float(active)
        feats["active_ratio"] = _safe(active / n_sub)

    # Mean absolute correlation between neighbouring subcarriers: a body moving
    # through the path couples subcarriers together, raising this.
    if n_frames > 3 and n_sub > 2:
        centred = amp - np.nanmean(amp, axis=0, keepdims=True)
        std = np.nanstd(centred, axis=0)
        valid = std > 1e-9
        if np.sum(valid) > 2:
            c = centred[:, valid] / std[valid]
            corr = np.corrcoef(c, rowvar=False)
            off = corr[~np.eye(corr.shape[0], dtype=bool)]
            off = off[np.isfinite(off)]
            if off.size:
                feats["subcarrier_corr"] = _safe(np.mean(np.abs(off)))

    phase_feats = _phase_features(phase)
    feats.update(phase_feats)

    # Single composite motion index (0..): scaled so typical room motion is O(1).
    feats["csi_motion_index"] = _safe(
        math.sqrt(max(feats["amp_var_mean"], 0.0)) * (0.5 + feats["active_ratio"])
        + feats["phase_diff_var"]
    )
    return feats


def _phase_features(phase: Optional[np.ndarray]) -> Dict[str, float]:
    out = {"phase_var": 0.0, "phase_diff_var": 0.0, "phase_spread": 0.0}
    if phase is None:
        return out
    p = np.asarray(phase, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.size == 0:
        return out
    # Unwrap along time per subcarrier so a 2π crossing is not read as a jump.
    if p.shape[0] > 1:
        p = np.unwrap(p, axis=0)
    out["phase_var"] = _safe(np.nanvar(p))
    if p.shape[0] > 1:
        d = np.diff(p, axis=0)
        out["phase_diff_var"] = _safe(np.nanvar(d))
    if p.shape[1] > 1:
        # spread across subcarriers in the last frame (SFO/CFO makes absolute
        # phase meaningless; its *variation* is the useful part)
        out["phase_spread"] = _safe(np.nanstd(p[-1]))
    return out


# --------------------------------------------------------------------------- #
# windowed extractors used by the live loop
# --------------------------------------------------------------------------- #

class RSSIFeatureExtractor:
    """Rolling window → feature dict, called once per analysis tick."""

    def __init__(self, window_s: float = 12.0, fs: float = 20.0, min_samples: int = 24,
                 spectral: bool = True, nperseg: int = 128,
                 bands: Optional[Dict[str, Sequence[float]]] = None,
                 maxlen: int = 20000):
        self.window_s = float(window_s)
        self.fs = float(fs)
        self.min_samples = int(min_samples)
        self.spectral = bool(spectral)
        self.nperseg = int(nperseg)
        self.bands = bands
        self.t: Deque[float] = deque(maxlen=maxlen)
        self.x: Deque[float] = deque(maxlen=maxlen)

    def push(self, t: float, rssi: float) -> None:
        if rssi is None or not np.isfinite(rssi):
            return
        self.t.append(float(t))
        self.x.append(float(rssi))
        self._trim()

    def _trim(self) -> None:
        if not self.t:
            return
        cutoff = self.t[-1] - self.window_s
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.x.popleft()

    def ready(self) -> bool:
        return len(self.x) >= self.min_samples and (self.t[-1] - self.t[0]) > 1.0

    def features(self) -> Optional[Dict[str, float]]:
        if not self.ready():
            return None
        return extract_rssi_features(list(self.t), list(self.x), self.fs,
                                     bands=self.bands, spectral=self.spectral,
                                     nperseg=self.nperseg)

    def window(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.fromiter(self.t, dtype=float, count=len(self.t)), \
               np.fromiter(self.x, dtype=float, count=len(self.x))

    def reset(self) -> None:
        self.t.clear()
        self.x.clear()


class CSIFeatureExtractor:
    """Rolling window of CSI frames → feature dict.

    CSI frames arrive at the packet rate; the window holds the last
    ``window_s`` seconds so amplitude/phase statistics stay comparable with the
    RSSI path's cadence.
    """

    def __init__(self, window_s: float = 4.0, max_frames: int = 800, min_frames: int = 12):
        self.window_s = float(window_s)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.t: Deque[float] = deque(maxlen=max_frames)
        self.amp: Deque[np.ndarray] = deque(maxlen=max_frames)
        self.phase: Deque[np.ndarray] = deque(maxlen=max_frames)
        self.rssi: Deque[float] = deque(maxlen=max_frames)

    def push(self, t: float, amplitude: Sequence[float], phase: Optional[Sequence[float]] = None,
             rssi: Optional[float] = None) -> None:
        if amplitude is None or len(amplitude) == 0:
            return
        self.t.append(float(t))
        self.amp.append(np.asarray(amplitude, dtype=float))
        self.phase.append(np.asarray(phase, dtype=float) if phase is not None else None)
        self.rssi.append(float(rssi) if rssi is not None else float("nan"))
        self._trim()

    def _trim(self) -> None:
        if not self.t:
            return
        cutoff = self.t[-1] - self.window_s
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.amp.popleft()
            self.phase.popleft()
            self.rssi.popleft()

    def ready(self) -> bool:
        return len(self.t) >= self.min_frames

    def features(self) -> Optional[Dict[str, float]]:
        if not self.ready():
            return None
        n_sub = min(int(a.size) for a in self.amp)
        amp = np.vstack([a[:n_sub] for a in self.amp])
        phases = [p[:n_sub] for p in self.phase if p is not None]
        phase = np.vstack(phases) if phases else None
        rssis = [v for v in self.rssi if np.isfinite(v)]
        rssi = float(np.mean(rssis)) if rssis else None
        return extract_csi_features(amp, phase, rssi=rssi)

    def reset(self) -> None:
        for buf in (self.t, self.amp, self.phase, self.rssi):
            buf.clear()
