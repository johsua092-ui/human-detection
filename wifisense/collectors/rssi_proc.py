"""RSSI from ``/proc/net/wireless`` — last-resort Linux backend.

No root, no extra tools, one syscall per poll. Caveats that are handled
honestly instead of papered over:

* several modern drivers (``iwlwifi`` especially) put a constant bogus value in
  the ``level`` column — that is detected and reported as unreliable;
* the update rate is whatever the driver refreshes at, typically 1 Hz;
* only interfaces that are up and associated appear at all.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional

from . import parsers
from .base import Collector, CollectorError, Sample

PROC_PATH = "/proc/net/wireless"


class ProcRSSICollector(Collector):
    name = "rssi_proc"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None):
        super().__init__(config, interface)
        self.interval = 1.0
        self._unreliable_polls = 0

    def preflight(self) -> None:
        if not Path(PROC_PATH).exists():
            raise CollectorError(
                f"{PROC_PATH} does not exist — this backend is Linux-only and needs a "
                f"wireless subsystem",
                ["use --source iw on Linux", "or --source netsh (Windows) / --source airport (macOS)"],
            )
        rows = parsers.parse_proc_wireless(Path(PROC_PATH).read_text(errors="replace"))
        if not rows:
            raise CollectorError(
                "no wireless interface is listed in /proc/net/wireless",
                ["the NIC may be down: sudo ip link set wlan0 up",
                 "or it may be in monitor mode already (which hides it from /proc)",
                 "or there is no wireless hardware on this machine"],
            )
        if self.config:
            self.interval = max(float(self.config.get_path("sampling.interval_s", 1.0)), 0.2)

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        self.started_at = time.time()
        self.status["interval_s"] = self.interval
        warned = False
        while not self.stopped:
            try:
                rows = parsers.parse_proc_wireless(Path(PROC_PATH).read_text(errors="replace"))
            except Exception as exc:  # pragma: no cover - defensive
                raise CollectorError(f"could not read {PROC_PATH}: {exc}") from exc

            picked = None
            for row in rows:
                if self.interface is None or row["iface"] == self.interface:
                    if row.get("level_reliable"):
                        picked = row
                        break
                    if picked is None:
                        picked = row

            now = time.time()
            if picked and picked.get("level_reliable"):
                self.status["level_dbm"] = picked["level_dbm"]
                yield self.emit(Sample(
                    t=now, rssi=float(picked["level_dbm"]), source=self.name,
                    noise_dbm=picked.get("noise_dbm"),
                    meta={"iface": picked["iface"], "link_quality": picked["link_quality"]},
                ))
                self._unreliable_polls = 0
            else:
                self._unreliable_polls += 1
                if picked and not warned:
                    warned = True
                    self.status["warning"] = (
                        f"driver reports an unreliable level value for {picked['iface']} — "
                        "use --source iw for a trustworthy RSSI"
                    )
                if self._unreliable_polls * self.interval > 15:
                    raise CollectorError(
                        "/proc/net/wireless did not yield a single trustworthy RSSI value in 15 s",
                        [f"check by hand:  cat {PROC_PATH}",
                         "this driver reports a constant bogus level; use --source iw instead",
                         "or use monitor mode:  sudo python main.py --source radiotap"],
                    )
            self._sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
