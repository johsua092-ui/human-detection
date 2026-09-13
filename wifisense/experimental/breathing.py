"""EXPERIMENTAL: breathing / heartbeat estimation from CSI amplitude.

Status: **experimental and intentionally separate.** This module is only ever
active when a real CSI backend is running, and even then its output must be
treated as a rough research signal, not a medical measurement.

Why it deserves all those caveats: a resting human's chest moves the multipath
by millimetres, which modulates CSI amplitude at roughly 0.15-0.5 Hz. But the
same band contains the NIC's own AGC ripples, the AP's power-saving cycles, and
any slow environmental drift, and RSSI (even CSI) is blind to where in the room
the breathing body is. Anything the model reports below ~0.4 Hz is a *candidate
oscillation*, consistent with breathing but not proof of it.

Deliberate limitations (also printed at runtime):

* Requires CSI (per-subcarrier amplitude over time); RSSI alone is explicitly
  NOT used — single-stream RSSI at 1 Hz cannot resolve this band reliably.
* Requires a still subject. Motion pollutes the same subcarriers.
* Output is band power + dominant frequency in the breathing band, with a
  "consistency" score from autocorrelation; no identity, no diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

BREATHING_BAND = (0.15, 0.55)   # Hz — adults at rest sit roughly here
HEARTBEAT_BAND = (0.8, 2.5)     # Hz — gated behind CSI quality, very unreliable


def estimate(amplitude: np.ndarray, t: np.ndarray | None = None,
             fs: float = 20.0, min_frames: int = 240) -> Dict[str, Any]:
    """Estimate breathing-like modulation in a CSI amplitude matrix.

    ``amplitude`` shape: ``(n_frames, n_subcarriers)`` — a window of at least
    ``min_frames`` frames. Returns band power, dominant frequency, consistency
    (normalised autocorrelation at the dominant lag) and a plain-language
    ``read`` string. Raises ``ValueError`` when the window is too short or too
    still to say anything.
    """
    amp = np.asarray(amplitude, dtype=float)
    if amp.ndim == 1:
        amp = amp.reshape(-1, 1)
    if amp.shape[0] < min_frames:
        raise ValueError(
            f"need >= {min_frames} CSI frames for a breathing estimate, got {amp.shape[0]}"
        )
    if amp.shape[1] < 8:
        raise ValueError("fewer than 8 subcarriers — breathing estimation needs CSI resolution")

    t = np.arange(amp.shape[0]) / fs if t is None else np.asarray(t, dtype=float)
    if t.size != amp.shape[0] or t[-1] <= t[0]:
        t = np.arange(amp.shape[0]) / fs

    # Normalise each subcarrier to its own scale, then average across them so a
    # single loud subcarrier cannot dominate.
    centred = amp - amp.mean(axis=0, keepdims=True)
    scale = amp.std(axis=0)
    scale[scale < 1e-9] = 1.0
    series = (centred / scale).mean(axis=1)

    series = series - series.mean()
    n = series.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(series * win)) ** 2

    lo, hi = BREATHING_BAND
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        raise ValueError("sampling rate too low to resolve the breathing band")
    band_psd = spectrum[band]
    band_freqs = freqs[band]
    total = spectrum.sum()
    band_power = float(band_psd.sum() / max(total, 1e-18))

    dom_idx = int(np.argmax(band_psd))
    dom_freq = float(band_freqs[dom_idx])
    # consistency: normalised autocorrelation at the dominant period (skip lag 0)
    period = max(2, int(round(fs / dom_freq))) if dom_freq > 0 else 2
    if period < n // 3:
        autocorr = np.corrcoef(series[:-period], series[period:])[0, 1]
    else:
        autocorr = 0.0
    consistency = float(max(0.0, autocorr))

    result: Dict[str, Any] = {
        "band": list(BREATHING_BAND),
        "band_power_ratio": round(band_power, 5),
        "dominant_freq_hz": round(dom_freq, 3),
        "period_s": round(1.0 / dom_freq, 2) if dom_freq > 0 else None,
        "consistency": round(consistency, 3),
        "n_frames": int(n),
        "fs": fs,
        "experimental": True,
    }

    if consistency > 0.35 and 0.12 < dom_freq < 0.7:
        result["read"] = (f"periodic modulation at {result['period_s']} s — "
                          f"consistent with a resting body breathing, but not proof")
    elif consistency > 0.2:
        result["read"] = "weak periodic structure in the breathing band"
    else:
        result["read"] = "no consistent breathing-band oscillation in this window"
    return result
