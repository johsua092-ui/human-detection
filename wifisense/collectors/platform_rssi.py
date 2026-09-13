"""Cross-platform RSSI backends: Windows (``netsh``), macOS (``airport``) and
Android/Termux (``termux-wifi-connectioninfo``).

All three are best-effort: they only see the NIC's own associated link, they
update about once per second, and Windows reports a quantised 0-100 % bar that is
converted to dBm with the standard ``dBm = pct/2 - 100`` mapping. That is enough
for gross presence/motion detection and nothing finer. This module never
pretends otherwise.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from . import parsers
from .base import Collector, CollectorError, Sample

AIRPORT_PATHS = (
    "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
    "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/A/Resources/airport",
)


class NetshRSSICollector(Collector):
    """Windows: ``netsh wlan show interfaces``."""

    name = "rssi_netsh"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None):
        super().__init__(config, interface)
        self.interval = 1.0

    def preflight(self) -> None:
        if not sys.platform.startswith("win"):
            raise CollectorError(
                "netsh backend is Windows-only",
                ["on Linux use --source iw or --source radiotap"],
            )
        if not shutil.which("netsh"):
            raise CollectorError("netsh not found on PATH")
        if self.config:
            self.interval = max(float(self.config.get_path("sampling.interval_s", 1.0)), 0.2)
        self.status["interval_s"] = self.interval
        if not self._poll():
            raise CollectorError(
                "netsh reported no wireless interface",
                ["check `netsh wlan show interfaces` in a terminal",
                 "the WLAN AutoConfig service must be running"],
            )

    def _poll(self):
        from .capabilities import run
        res = run(["netsh", "wlan", "show", "interfaces"], timeout=8)
        return parsers.parse_netsh_interfaces(res["out"]) if res["ok"] else []

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        self.started_at = time.time()
        while not self.stopped:
            for row in self._poll():
                rssi = row.get("signal_dbm")
                if rssi is None:
                    continue
                yield self.emit(Sample(
                    t=time.time(), rssi=rssi, source=self.name,
                    bssid=row.get("bssid"), channel=row.get("channel"),
                    meta={"ssid": row.get("ssid"), "signal_pct": row.get("signal_pct"),
                          "quantised": True},
                ))
            self._sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()


class AirportRSSICollector(Collector):
    """macOS: the legacy ``airport -I`` utility (RSSI + noise for the current link)."""

    name = "rssi_airport"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None):
        super().__init__(config, interface)
        self.interval = 1.0

    def _airport_path(self) -> Optional[str]:
        for path in AIRPORT_PATHS:
            if shutil.which(path) or Path(path).exists():
                return path
        return shutil.which("airport")

    def preflight(self) -> None:
        if sys.platform != "darwin":
            raise CollectorError("airport backend is macOS-only",
                                 ["on Linux use --source iw or --source radiotap"])
        if not self._airport_path():
            raise CollectorError(
                "the `airport` utility was removed in recent macOS releases",
                ["run the sensor on a Linux host instead (best supported platform)",
                 "or use an offline capture: --replay <csv log>"],
            )
        if self.config:
            self.interval = max(float(self.config.get_path("sampling.interval_s", 1.0)), 0.2)
        self.status["interval_s"] = self.interval

    def stream(self) -> Iterator[Sample]:
        from .capabilities import run
        self.preflight()
        self.started_at = time.time()
        path = self._airport_path()
        failures = 0
        while not self.stopped:
            res = run([path, "-I"], timeout=8)
            info = parsers.parse_airport(res["out"]) if res["ok"] else {}
            if info.get("signal_dbm") is not None:
                failures = 0
                yield self.emit(Sample(
                    t=time.time(), rssi=float(info["signal_dbm"]), source=self.name,
                    noise_dbm=info.get("noise_dbm"), channel=info.get("channel"),
                    meta={"ssid": info.get("ssid")},
                ))
            else:
                failures += 1
                if failures * self.interval > 12:
                    raise CollectorError(
                        "airport -I returned no RSSI for 12 s (no associated link?)",
                        ["connect the Mac to a WiFi network and re-run",
                         "or the utility is missing on modern macOS — run on Linux"],
                    )
            self._sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()


class TermuxRSSICollector(Collector):
    """Android via Termux: ``termux-wifi-connectioninfo`` RSSI polling.

    This is how the project runs **on a phone** without root and without any
    external sensor: the termux-api helper asks Android's own WiFi service for
    the current connection, which contains a live RSSI in dBm. One sample per
    call (~1 Hz), so it is as sensitive as the ``iw`` link backend — good for
    "did somebody just walk in", not for anything subtle.

    Requirements: Termux + ``pkg install termux-api`` + the Termux:API app, and
    ``termux-wake-lock`` held so Android does not freeze the loop in the
    background. Rooted devices with Broadcom radios can instead use live CSI via
    nexmon_csi (``--mode csi``) — see the CSI section of the README.
    """

    name = "rssi_termux"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None):
        super().__init__(config, interface)
        self.interval = 1.0
        self.binary = "termux-wifi-connectioninfo"

    def preflight(self) -> None:
        path = shutil.which(self.binary)
        if not path:
            raise CollectorError(
                "termux-wifi-connectioninfo not found — this backend is for Android/Termux",
                ["in Termux: pkg install termux-api   (plus install the Termux:API app)",
                 "on a Linux laptop use --source iw or --source radiotap instead",
                 "check it by hand: termux-wifi-connectioninfo"],
            )
        self.binary = path
        if self.config:
            self.interval = max(float(self.config.get_path("sampling.interval_s", 1.0)), 0.2)
        self.status.update({"interval_s": self.interval, "helper": path})
        if not self._poll():
            raise CollectorError(
                f"{Path(path).name} returned nothing usable — is WiFi connected and the "
                f"Termux:API app granted the location permission?",
                ["Android hides WiFi details without location permission (API >= 30)",
                 "run by hand: termux-wifi-connectioninfo"],
            )

    def _poll(self) -> Dict[str, Any]:
        from .capabilities import run
        res = run([self.binary], timeout=10)
        return parsers.parse_termux_wifi(res["out"]) if res["ok"] else {}

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        self.started_at = time.time()
        misses = 0
        while not self.stopped:
            info = self._poll()
            rssi = info.get("signal_dbm")
            if rssi is not None:
                misses = 0
                yield self.emit(Sample(
                    t=time.time(), rssi=float(rssi), source=self.name,
                    bssid=info.get("bssid"), freq_mhz=info.get("freq_mhz"),
                    channel=info.get("channel"),
                    meta={"ssid": info.get("ssid"),
                          "link_speed_mbps": info.get("link_speed_mbps"),
                          "state": info.get("supplicant_state")},
                ))
            else:
                misses += 1
                if misses * self.interval > 20:
                    raise CollectorError(
                        "no usable WiFi RSSI from termux-wifi-connectioninfo for 20 s",
                        ["is WiFi actually connected?",
                         "grant the location permission to the Termux:API app",
                         "run by hand: termux-wifi-connectioninfo"],
                    )
            self._sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
