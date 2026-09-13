"""Signal conditioning.

The RSSI stream coming off a real NIC is ugly: dropouts to -100 dBm when the
NIC retunes, quantisation steps of 0.5-1 dBm, AGC jumps of several dBm, slow
thermal drift, and (in monitor mode) samples that arrive at irregular intervals
because they are event-driven. Presence detection lives or dies on this stage —
a raw variance threshold on unfiltered RSSI mostly measures the NIC, not you.

Everything here is plain NumPy/SciPy and stateless except :class:`StreamingFilter`,
which is the per-sample front end used by the runner.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import numpy as np

try:  # SciPy is a hard requirement, but keep imports explicit and greppable
    from scipy import signal as sp_signal
    from scipy import ndimage as sp_ndimage
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# scalar / window filters
# --------------------------------------------------------------------------- #

def median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Median filter with nearest-edge padding (kills single-sample dropouts)."""
    x = np.asarray(x, dtype=float)
    if window < 3 or x.size < 3:
        return x.copy()
    window = min(window if window % 2 else window + 1, x.size if x.size % 2 else x.size - 1)
    if window < 3:
        return x.copy()
    if HAVE_SCIPY:
        return sp_ndimage.median_filter(x, size=window, mode="nearest")
    out = np.empty_like(x)
    half = window // 2
    for i in range(x.size):
        lo, hi = max(0, i - half), min(x.size, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x.copy()
    if HAVE_SCIPY:
        return sp_ndimage.uniform_filter1d(x, size=window, mode="nearest")
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def exponential_smoothing(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    alpha = float(min(max(alpha, 1e-3), 1.0))
    out = np.empty_like(x)
    acc = x[0]
    for i, value in enumerate(x):
        acc = alpha * value + (1.0 - alpha) * acc
        out[i] = acc
    return out


def hampel_filter(x: np.ndarray, window: int = 9, n_sigma: float = 3.0,
                  return_mask: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Hampel identifier: replace samples that deviate more than ``n_sigma``
    robust sigmas from a local median.

    Robust because it uses median + MAD, so a few big outliers cannot inflate
    the scale estimate the way a mean/std filter would.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return (x.copy(), np.zeros(n, dtype=bool)) if return_mask else x.copy()
    half = max(1, window // 2)
    filtered = x.copy()
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sigma = 1.4826 * mad
        # Perfectly flat segments (dropout mid-window, quantised RSSI) give a
        # zero MAD; fall back to a small absolute floor so a genuine spike of
        # several dBm is still caught instead of being silently kept.
        if sigma <= 1e-12:
            sigma = 0.25
        if abs(x[i] - med) > n_sigma * sigma:
            filtered[i] = med
            mask[i] = True
    return (filtered, mask) if return_mask else filtered


def detrend_linear(x: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove a straight-line trend (thermal drift, RSSI decay as the AP backs off)."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x - (np.mean(x) if x.size else 0.0)
    idx = np.arange(x.size, dtype=float) if t is None else np.asarray(t, dtype=float)
    if idx.size != x.size or np.ptp(idx) == 0:
        idx = np.arange(x.size, dtype=float)
    coef = np.polyfit(idx, x, 1)
    return x - np.polyval(coef, idx)


def robust_scale(x: np.ndarray, floor: float = 0.1) -> float:
    """MAD-based sigma (1.4826 * median absolute deviation), floored at
    ``floor`` so a perfectly quiet RSSI stream cannot produce a zero scale that
    makes every later sample look like an outlier (or, worse, none)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return floor
    med = np.median(x)
    return float(max(1.4826 * np.median(np.abs(x - med)), floor))


def robust_z(value: float, med: float, sigma: float) -> float:
    if sigma is None or sigma <= 1e-12:
        return 0.0
    return float((value - med) / sigma)


# --------------------------------------------------------------------------- #
# resampling + spectral
# --------------------------------------------------------------------------- #

def resample_uniform(t: Iterable[float], x: Iterable[float], fs: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Put irregularly timed samples on a uniform grid so FFT/Welch are valid.

    Monitor-mode RSSI arrives per frame, so the spacing is uneven. Linear
    interpolation onto a fixed grid is the standard fix; the grid starts at the
    first timestamp and ends at the last, so no data is invented beyond the
    measured span.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < 2:
        return t, x
    order = np.argsort(t)
    t, x = t[order], x[order]
    mask = np.isfinite(x)
    t, x = t[mask], x[mask]
    if t.size < 2 or t[-1] <= t[0]:
        return t, x
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    grid = t[0] + np.arange(n) / fs
    return grid, np.interp(grid, t, x)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass; used to isolate motion bands."""
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    low = max(low / nyq, 1e-4)
    high = min(high / nyq, 0.99)
    if not HAVE_SCIPY or x.size < 16 or low >= high:
        return x.copy()
    sos = sp_signal.butter(order, [low, high], btype="bandpass", output="sos")
    try:
        return sp_signal.sosfiltfilt(sos, x)
    except ValueError:  # window shorter than the pad length
        return sp_signal.sosfilt(sos, x)


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.size < 8 or fs <= 0:
        return np.array([0.0]), np.array([0.0])
    nperseg = int(min(max(8, nperseg), x.size))
    if HAVE_SCIPY:
        freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
        return freqs, psd
    # crude periodogram fallback
    x = x - x.mean()
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2 / (fs * x.size)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    if freqs.size == 0:
        return 0.0
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


# --------------------------------------------------------------------------- #
# streaming front end
# --------------------------------------------------------------------------- #

class StreamingFilter:
    """Per-sample conditioning for the live loop.

    Applies, in order: outlier gate (robust z against a running median) →
    short median filter → optional EMA for display. Keeps the raw value too so
    the dashboard can show both and calibration can see dropouts.
    """

    def __init__(self, outlier_sigma: float = 4.0, median_window: int = 5,
                 smooth_alpha: float = 0.35, history: int = 600):
        self.outlier_sigma = float(outlier_sigma)
        self.median_window = int(median_window)
        self.smooth_alpha = float(smooth_alpha)
        self.raw: Deque[float] = deque(maxlen=history)
        self.clean: Deque[float] = deque(maxlen=history)
        self.ema: Optional[float] = None
        self.rejected = 0
        self.accepted = 0
        self.last_reject_reason: Optional[str] = None

    def update(self, value: float) -> Dict[str, Any]:
        """Feed one raw sample, get ``{raw, filtered, ema, rejected}``."""
        if value is None or not np.isfinite(value):
            self.rejected += 1
            self.last_reject_reason = "non-finite"
            return {"raw": value, "filtered": self.clean[-1] if self.clean else None,
                    "ema": self.ema, "rejected": True}

        self.raw.append(float(value))
        filtered = float(value)
        rejected = False

        if len(self.raw) >= max(9, self.median_window * 2):
            tail = np.fromiter(list(self.raw)[-max(9, self.median_window * 2):],
                               dtype=float, count=min(len(self.raw), max(9, self.median_window * 2)))
            med = float(np.median(tail))
            sigma = robust_scale(tail)
            if sigma > 1e-9 and abs(value - med) > self.outlier_sigma * sigma:
                self.rejected += 1
                self.last_reject_reason = f"outlier gate ({(value - med) / sigma:.1f} sigma)"
                filtered = med
                rejected = True
            else:
                self.accepted += 1
                self.last_reject_reason = None
        else:
            self.accepted += 1

        if self.median_window >= 3 and len(self.raw) >= self.median_window:
            window = np.fromiter(list(self.raw)[-self.median_window:], dtype=float,
                                 count=self.median_window)
            filtered = float(np.median(window))

        self.clean.append(filtered)
        if self.ema is None:
            self.ema = filtered
        else:
            self.ema = self.smooth_alpha * filtered + (1.0 - self.smooth_alpha) * self.ema

        return {"raw": float(value), "filtered": filtered, "ema": float(self.ema),
                "rejected": rejected}

    def reset(self) -> None:
        self.raw.clear()
        self.clean.clear()
        self.ema = None
        self.rejected = 0
        self.accepted = 0

    def stats(self) -> Dict[str, Any]:
        total = self.rejected + self.accepted
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "reject_ratio": (self.rejected / total) if total else 0.0,
            "last_reject_reason": self.last_reject_reason,
        }
