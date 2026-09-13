"""Collector registry / factory.

``build_collector`` is the single place that maps a config + CLI request onto a
concrete backend, and it refuses to guess when the request cannot be honoured:
an unsupported combination raises :class:`CollectorError` carrying the reasons
and the alternatives, instead of quietly degrading.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Collector, CollectorError, Sample
from .csi import CSICollector
from .platform_rssi import AirportRSSICollector, NetshRSSICollector, TermuxRSSICollector
from .router import RouterRSSICollector
from .rssi_iw import IwRSSICollector
from .rssi_proc import ProcRSSICollector
from .rssi_radiotap import RadiotapRSSICollector
from .synthetic import ReplayCollector, SyntheticRSSICollector

SUPPORTED_SOURCES = (
    "auto", "radiotap", "iw", "proc", "netsh", "airport", "termux", "router",
    "csi", "csi_pcap", "synthetic", "replay",
)

__all__ = [
    "Collector", "CollectorError", "Sample",
    "RadiotapRSSICollector", "IwRSSICollector", "ProcRSSICollector",
    "NetshRSSICollector", "AirportRSSICollector", "TermuxRSSICollector",
    "RouterRSSICollector",
    "CSICollector", "SyntheticRSSICollector", "ReplayCollector",
    "build_collector", "SUPPORTED_SOURCES",
]


def build_collector(config: Any = None, source: str = "auto",
                    interface: Optional[str] = None, mode: str = "auto",
                    replay: Optional[str] = None, csi_capture: Optional[str] = None,
                    csi_tool: str = "nexmon") -> Collector:
    """Instantiate the requested backend. Raises CollectorError if impossible."""
    source = (source or "auto").lower()

    if source == "replay" or replay:
        if not replay:
            raise CollectorError("--source replay needs --replay <file.csv>")
        return ReplayCollector(config, path=replay, speed=_speed(config), interface=interface)

    if source == "synthetic":
        return SyntheticRSSICollector(config, interface=interface)

    if source == "csi_pcap":
        return CSICollector(config, interface=interface, tool=csi_tool, capture=csi_capture)

    if source == "csi":
        return CSICollector(config, interface=interface, tool=csi_tool, capture=csi_capture)

    if source == "radiotap":
        return RadiotapRSSICollector(config, interface=interface)

    if source == "iw":
        return IwRSSICollector(config, interface=interface)

    if source == "proc":
        return ProcRSSICollector(config, interface=interface)

    if source == "netsh":
        return NetshRSSICollector(config, interface=interface)

    if source == "airport":
        return AirportRSSICollector(config, interface=interface)

    if source == "termux":
        return TermuxRSSICollector(config, interface=interface)

    if source == "router":
        return RouterRSSICollector(config, interface=interface)

    if source == "auto":
        raise CollectorError(
            "build_collector(source='auto') must be resolved first",
            ["call capabilities.choose_backend(), then pass the chosen source here"],
        )

    raise CollectorError(
        f"unknown source '{source}'",
        [f"supported: {', '.join(SUPPORTED_SOURCES)}"],
    )


def _speed(config: Any) -> Optional[float]:
    if config is None:
        return 1.0
    return config.get_path("replay.speed", 1.0)
