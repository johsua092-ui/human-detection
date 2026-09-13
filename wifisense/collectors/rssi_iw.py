"""RSSI from the NIC's own link: ``iw`` polling (no root, no monitor mode).

Two sub-sources:

* ``link``    — ``iw dev <iface> link`` → the RSSI of our association with the AP.
                One sample per poll. Enough to see a person walk in front of the
                laptop ↔ AP path; weak for anything subtle.
* ``station`` — ``iw dev <iface> station dump`` → one RSSI per *associated client*
                (only when this NIC is acting as an AP). Every client is then an
                independent RSSI probe of the room, which is much better. This is
                the trick that makes a plain laptop-as-hotspot a usable sensor.

No wireless hardware? This raises a clear error instead of inventing numbers.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Optional

from . import parsers
from .base import Collector, CollectorError, Sample


class IwRSSICollector(Collector):
    name = "rssi_iw"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None,
                 sub_source: str = "auto"):
        super().__init__(config, interface)
        self.sub_source = sub_source
        self.interval = 0.2
        self._grace_s = 12.0
        self._mode_used = None

    # ------------------------------------------------------------------ #
    def preflight(self) -> None:
        from .capabilities import detect_tools, list_interfaces, run

        if not detect_tools().get("iw"):
            raise CollectorError(
                "`iw` is not installed — it is required for this backend",
                ["sudo apt install iw    # Debian/Ubuntu",
                 "sudo dnf install iw    # Fedora",
                 "or force another backend: --source proc (fallback, less reliable)"],
            )

        if not self.interface:
            wireless = [i for i in list_interfaces() if i["wireless"]]
            if not wireless:
                raise CollectorError(
                    "no wireless interface found",
                    ["check `iw dev` — a wired-only host cannot do WiFi sensing",
                     "for UI/pipeline demo only: python main.py --source synthetic"],
                )
            self.interface = wireless[0]["name"]

        if self.config:
            self.interval = float(self.config.get_path("sampling.interval_s", 0.2))
        self.interval = max(self.interval, 0.1)

        # Decide between link-polling and station-dump.
        if self.sub_source == "auto":
            res = run(["iw", "dev", self.interface, "station", "dump"])
            stations = parsers.parse_iw_station_dump(res["out"]) if res["ok"] else []
            if len(stations) >= 1:
                self.sub_source = "station"
            else:
                self.sub_source = "link"
        self._mode_used = self.sub_source

    # ------------------------------------------------------------------ #
    def stream(self) -> Iterator[Sample]:
        from .capabilities import run

        self.preflight()
        self.status.update({"sub_source": self.sub_source, "interval_s": self.interval})
        self.started_at = time.time()

        if self.sub_source == "station":
            yield from self._stream_station(run)
            return

        deadline = time.time() + self._grace_s
        gave_up = False
        while not self.stopped:
            res = run(["iw", "dev", self.interface, "link"], timeout=4)
            link = parsers.parse_iw_link(res["out"]) if res["ok"] else {"connected": False}
            if link.get("connected") and link.get("signal_dbm") is not None:
                deadline = time.time() + self._grace_s
                yield self.emit(Sample(
                    t=time.time(),
                    rssi=link["signal_dbm"],
                    source=self.name,
                    bssid=link.get("bssid"),
                    freq_mhz=link.get("freq_mhz"),
                    meta={"ssid": link.get("ssid"), "tx_bitrate": link.get("tx_bitrate")},
                ))
            else:
                if time.time() > deadline and not gave_up:
                    gave_up = True
                    raise CollectorError(
                        f"{self.interface} is not associated with any AP, so there is no link "
                        f"RSSI to sample (12 s of empty polls)",
                        ["connect to a WiFi network, then re-run",
                         "or use monitor mode: sudo python main.py --source radiotap",
                         "or put this NIC in AP/hotspot mode and use station-dump",
                         "check `iw dev %s link` by hand to confirm" % self.interface],
                    )
            self._sleep(self.interval)
        return

    def _stream_station(self, run) -> Iterator[Sample]:
        empty_polls = 0
        while not self.stopped:
            res = run(["iw", "dev", self.interface, "station", "dump"], timeout=5)
            now = time.time()
            if res["ok"]:
                stations = parsers.parse_iw_station_dump(res["out"])
                for st in stations:
                    rssi = st.get("signal_dbm")
                    if rssi is None:
                        continue
                    yield self.emit(Sample(
                        t=now,
                        rssi=rssi,
                        source=self.name,
                        bssid=st.get("mac"),
                        meta={"sub_source": "station",
                              "avg": st.get("signal_avg_dbm"),
                              "rx_bitrate": st.get("rx_bitrate"),
                              "inactive_ms": st.get("inactive_ms")},
                    ))
                empty_polls = 0 if stations else empty_polls + 1
            else:
                empty_polls += 1

            self.status["clients"] = empty_polls and 0 or len(
                parsers.parse_iw_station_dump(res.get("out", ""))
            )
            if empty_polls * self.interval > self._grace_s:
                raise CollectorError(
                    f"no station is associated with {self.interface} — station-dump has nothing "
                    f"to measure",
                    [f"start a hotspot on {self.interface} (e.g. `nmcli dev wifi hotspot`) and "
                     "connect a phone/laptop to it; each client becomes an RSSI probe",
                     "or fall back to link polling: --source iw with --mode rssi on an "
                     "interface that is associated to an external AP"],
                )
            self._sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
