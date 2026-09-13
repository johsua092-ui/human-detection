"""Passive monitor-mode RSSI sensing (the best RSSI backend).

Idea: put the NIC into monitor mode and stop transmitting. Every frame that the
radio hears — AP beacons (≈10 Hz each), neighbour APs, client data traffic —
carries a radiotap signal-strength field. That gives a *multi-transmitter* RSSI
time series at tens-to-hundreds of samples per second, which is a far richer
input to presence/motion detection than polling our own link at 1 Hz.

The human body is mostly water: moving through the propagation path changes the
multipath sum, which shows up as correlated fluctuation across all transmitters
in the room. Stationary breathing produces a subtler, low-frequency modulation.

Requires: root, a NIC that supports monitor mode, and tshark (preferred) or
tcpdump. The NIC's previous mode is saved and restored on stop.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional

from . import parsers
from .base import Collector, CollectorError, Sample

DEFAULT_UDP_AP_TRAFFIC = True


class RadiotapRSSICollector(Collector):
    name = "rssi_radiotap"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None):
        super().__init__(config, interface)
        self.tool: Optional[str] = None
        self.signal_field: str = "wlan_radio.signal_dbm"
        self._proc: Optional[subprocess.Popen] = None
        self._original_type: Optional[str] = None
        self._monitor_enabled = False
        self._bssid_filter: Optional[str] = None
        self._per_bssid: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    def preflight(self) -> None:
        from .capabilities import detect_tools, list_interfaces, monitor_mode_supported, run

        if not self.interface:
            wireless = [i for i in list_interfaces() if i["wireless"]]
            if not wireless:
                raise CollectorError(
                    "no wireless interface available for monitor-mode capture",
                    ["plug in / enable a WiFi NIC, or run with a different --source"],
                )
            self.interface = wireless[0]["name"]

        tools = detect_tools()
        if tools.get("tshark"):
            self.tool = "tshark"
        elif tools.get("tcpdump"):
            self.tool = "tcpdump"
        else:
            raise CollectorError(
                "neither tshark nor tcpdump found — monitor-mode capture needs one of them",
                ["sudo apt install tshark    # recommended (per-frame RSSI, MAC, freq)",
                 "sudo apt install tcpdump   # fallback (fewer fields)"],
            )

        if hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() != 0:
            raise CollectorError(
                "monitor mode requires root",
                [f"sudo $(which python3) main.py --source radiotap --interface {self.interface}"],
            )

        monitor_ok = monitor_mode_supported(self.interface)
        if monitor_ok is False:
            raise CollectorError(
                f"{self.interface} does not advertise monitor mode support",
                ["use --source iw (poll this NIC's own link RSSI) instead",
                 "or use a NIC with monitor support (most USB Atheros/Realtek adapters)"],
            )

        res = run(["iw", "dev", self.interface])
        for rec in parsers.parse_iw_dev(res["out"]):
            if rec.get("iface") == self.interface:
                self._original_type = rec.get("type")
        if self.tool == "tshark":
            self.signal_field = self._pick_tshark_field()

    def _pick_tshark_field(self) -> str:
        """tshark >= 3.0 exposes ``wlan_radio.signal_dbm``; older builds used radiotap."""
        try:
            out = subprocess.run(["tshark", "--version"], capture_output=True, text=True,
                                 timeout=10).stdout
            m = re.search(r"TShark[^\d]*(\d+)\.(\d+)", out)
            if m:
                major = int(m.group(1))
                return "wlan_radio.signal_dbm" if major >= 3 else "radiotap.dbm_antsignal"
        except Exception:
            pass
        return self.signal_field

    # ------------------------------------------------------------------ #
    def _set_monitor(self) -> None:
        from .capabilities import run

        cmds = [
            ["ip", "link", "set", self.interface, "down"],
            ["iw", "dev", self.interface, "set", "type", "monitor"],
        ]
        channel = self.config.get_path("channel") if self.config else None
        if channel:
            cmds.append(["iw", "dev", self.interface, "set", "channel", str(int(channel))])
        cmds.append(["ip", "link", "set", self.interface, "up"])

        for cmd in cmds:
            res = run(cmd)
            if not res["ok"]:
                # `ip link set down` on an nmcli-managed iface can complain about
                # state; anything else is fatal.
                if cmd[1] == "link" and "down" in cmd:
                    continue
                raise CollectorError(
                    f"failed to enable monitor mode: {' '.join(cmd)}\n  {res['err'].strip()}",
                    ["close NetworkManager's grip on the interface: "
                     "sudo nmcli dev set %s managed no" % self.interface,
                     "make sure no other process (wpa_supplicant) holds the interface"],
                )
        self._monitor_enabled = True

    def _restore(self) -> None:
        from .capabilities import run

        if not self._monitor_enabled:
            return
        target = self._original_type if self._original_type in ("managed", "station") else "managed"
        for cmd in (["ip", "link", "set", self.interface, "down"],
                    ["iw", "dev", self.interface, "set", "type", "managed"],
                    ["ip", "link", "set", self.interface, "up"]):
            run(cmd)
        # Hand it back to NetworkManager so the user's normal WiFi keeps working.
        if shutil.which("nmcli"):
            run(["nmcli", "dev", "set", self.interface, "managed", "yes"], timeout=10)
            run(["nmcli", "dev", "connect", self.interface], timeout=20)
        self._monitor_enabled = False

    # ------------------------------------------------------------------ #
    def _tshark_cmd(self) -> List[str]:
        return [
            "tshark",
            "-i", self.interface,
            "-l",                     # line buffered
            "-n",                     # no name resolution (speed)
            "-q",                     # suppress packet counter on stderr
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", self.signal_field,
            "-e", "wlan.bssid",
            "-e", "wlan_radio.frequency",
            "-e", "frame.len",
            "-Y", self.signal_field,  # only frames that actually carry an RSSI
        ]

    def _tcpdump_cmd(self) -> List[str]:
        return ["tcpdump", "-i", self.interface, "-l", "-n", "-e",
                "-y", "IEEE802_11_RADIO", "-v"]

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        self._set_monitor()
        cmd = self._tshark_cmd() if self.tool == "tshark" else self._tcpdump_cmd()
        self.status.update({"tool": self.tool, "cmd": " ".join(cmd),
                            "signal_field": self.signal_field})
        self.started_at = time.time()
        started = False
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL,
                                          text=True, bufsize=1)
            started = True
            for line in self._proc.stdout:  # type: ignore[union-attr]
                if self.stopped:
                    break
                sample = self._parse(line)
                if sample is not None:
                    yield self.emit(sample)
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            if started:
                self.stop()
            self._restore()

    def _parse(self, line: str) -> Optional[Sample]:
        if self.tool == "tshark":
            rec = parsers.parse_tshark_line(line)
            if not rec or rec.get("rssi") is None:
                return None
            bssid = rec.get("bssid")
            self._count_bssid(bssid)
            return Sample(
                t=rec["t"],
                rssi=rec["rssi"],
                source=self.name,
                bssid=bssid,
                freq_mhz=rec.get("freq_mhz"),
                meta={"len": rec.get("length")},
            )
        rec = parsers.parse_tcpdump_line(line)
        if not rec or rec.get("rssi") is None:
            return None
        bssid = rec.get("bssid")
        self._count_bssid(bssid)
        return Sample(t=time.time(), rssi=rec["rssi"], source=self.name, bssid=bssid)

    def _count_bssid(self, bssid: Optional[str]) -> None:
        key = bssid or "unknown"
        entry = self._per_bssid.setdefault(key, {"count": 0.0, "last": 0.0})
        entry["count"] += 1
        entry["last"] = time.time()
        if len(self._per_bssid) <= 12:
            self.status["transmitters"] = len(self._per_bssid)

    def transmitter_summary(self, window_s: float = 30.0) -> Dict[str, int]:
        """How many frames per BSSID in the last *window_s* — sanity check that
        we are actually hearing a room and not a single saturated beacon."""
        now = time.time()
        return {b: int(v["count"]) for b, v in self._per_bssid.items()
                if now - v["last"] <= window_s}

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
        self._restore()
