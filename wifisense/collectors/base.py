"""Collector base types: the :class:`Sample` contract every backend produces.

A collector is a *stream of physical measurements*. It owns its process/socket,
reports a human-readable description of what it is doing, and raises
:class:`CollectorError` with an actionable message when the machine cannot do
what was asked — never silently substituting invented data.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


class CollectorError(RuntimeError):
    """Raised when a backend cannot run. Message must be actionable."""

    def __init__(self, message: str, hints: Optional[List[str]] = None):
        super().__init__(message)
        self.hints = hints or []

    def pretty(self) -> str:
        out = [f"ERROR: {self}"]
        for hint in self.hints:
            out.append(f"  · {hint}")
        return "\n".join(out)


@dataclass
class Sample:
    """One physical measurement.

    ``rssi`` is dBm (negative). CSI samples additionally carry per-subcarrier
    ``amplitude`` / ``phase`` lists and ``n_subcarriers``. ``t`` is epoch
    seconds taken from the radio stack where possible (tshark frame timestamps,
    nexmon packet arrival) and from ``time.time()`` otherwise.
    """

    t: float
    rssi: Optional[float]
    source: str
    bssid: Optional[str] = None
    freq_mhz: Optional[float] = None
    channel: Optional[float] = None
    noise_dbm: Optional[float] = None
    amplitude: Optional[List[float]] = None
    phase: Optional[List[float]] = None
    n_subcarriers: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_csi(self) -> bool:
        return bool(self.amplitude)

    def to_row(self) -> Dict[str, Any]:
        """Flat row for CSV logging (CSI vectors are summarised, full CSI goes to npz)."""
        row: Dict[str, Any] = {
            "t": round(self.t, 6),
            "rssi": None if self.rssi is None else round(self.rssi, 3),
            "source": self.source,
            "bssid": self.bssid,
            "freq_mhz": self.freq_mhz,
            "channel": self.channel,
            "noise_dbm": self.noise_dbm,
            "n_subcarriers": self.n_subcarriers,
        }
        if self.amplitude:
            amps = self.amplitude
            row["csi_amp_mean"] = sum(amps) / len(amps)
            row["csi_amp_var"] = _variance(amps)
        return row


def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


class Collector:
    """Base class. Subclasses implement :meth:`preflight`, :meth:`stream`, :meth:`stop`."""

    name = "abstract"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None):
        self.config = config
        self.interface = interface
        self._stop = threading.Event()
        self.status: Dict[str, Any] = {}
        self.started_at: Optional[float] = None
        self.sample_count = 0

    # ---- lifecycle ------------------------------------------------------ #
    def preflight(self) -> None:
        """Validate the environment. Raise :class:`CollectorError` if unusable."""

    def stream(self) -> Iterator[Sample]:  # pragma: no cover - abstract
        raise NotImplementedError

    def stop(self) -> None:
        self._stop.set()

    # ---- helpers -------------------------------------------------------- #
    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep."""
        self._stop.wait(seconds)

    def describe(self) -> Dict[str, Any]:
        return {
            "collector": self.name,
            "mode": self.mode,
            "interface": self.interface,
            "samples": self.sample_count,
            "uptime_s": None if self.started_at is None else round(time.time() - self.started_at, 1),
            **self.status,
        }

    def emit(self, sample: Sample) -> Sample:
        self.sample_count += 1
        return sample
