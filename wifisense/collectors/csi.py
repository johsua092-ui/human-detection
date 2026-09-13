"""CSI (Channel State Information) collectors.

CSI is per-subcarrier amplitude **and** phase, and it is the only way to see
anything finer than gross motion. It is also the part of WiFi sensing that
*actually* requires special hardware: stock drivers throw the CSI away.

Supported acquisition paths, all without ESP32/ESP8266/Raspberry Pi:

======================  =========================================  ============
toolchain               hardware                                   status here
======================  =========================================  ============
``nexmon_csi``          Broadcom BCM43xx with patched firmware     live UDP
Intel 5300 CSI Tool     iwlwifi 5300 + patched kernel module      file/pcap
``ath9k`` CSI tool      Atheros AR9xxx + patched driver           file/pcap
any of the above        captured ``.pcap`` / CSI log file         via ``csiread``
======================  =========================================  ============

The live path needs root (patched firmware is loaded at boot, ``nexutil``
configures the chip and the firmware then streams CSI over UDP). The file paths
need the optional ``csiread`` package. When none of it is present this module
raises :class:`CollectorError` listing exactly what was checked — the runner then
falls back to RSSI with that explanation, and never fabricates CSI.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import parsers
from .base import Collector, CollectorError, Sample

DEFAULT_UDP_PORT = 5500


class CSICollector(Collector):
    """Common surface for CSI sources; subclasses implement :meth:`stream`."""

    name = "csi"
    mode = "csi"

    def __init__(self, config: Any = None, interface: Optional[str] = None,
                 tool: str = "nexmon", udp_port: int = DEFAULT_UDP_PORT,
                 capture: Optional[str] = None, chip: Optional[str] = None):
        super().__init__(config, interface)
        self.tool = tool
        self.udp_port = udp_port
        self.capture = capture
        self.chip = chip
        self._probe_msg = ""

    # ------------------------------------------------------------------ #
    def _unsupported(self) -> CollectorError:
        from .capabilities import detect_csi
        rep = detect_csi()
        hints: List[str] = []
        for chk in rep["checks"]:
            mark = "OK  " if chk["available"] else "NO  "
            hints.append(f"{mark}{chk['backend']:<14} {chk['detail']}")
        hints.append("offline alternative: capture a CSI pcap with any CSI tool in a VM/other "
                     "machine and run:  python main.py --mode csi --source csi_pcap "
                     "--csi-pcap capture.pcap --csi-tool nexmon")
        hints.append("or run RSSI mode, which needs no special hardware:  python main.py --mode rssi")
        return CollectorError(
            f"no live CSI source available (requested tool: {self.tool})", hints
        )

    # ------------------------------------------------------------------ #
    def preflight(self) -> None:
        if self.tool in ("csiread", "pcap", "file"):
            return self._preflight_file()
        if self.tool != "nexmon":
            raise CollectorError(
                f"live CSI with '{self.tool}' is not supported: the Intel 5300 and ath9k "
                f"toolchains write CSI to a log/pcap file, they do not stream live",
                ["use --csi-tool iwl5300 --csi-pcap capture.pcap (offline replay)",
                 "or --csi-tool nexmon for live capture on Broadcom hardware"],
            )
        return self._preflight_nexmon()

    # -- nexmon live ---------------------------------------------------- #
    def _preflight_nexmon(self) -> None:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise CollectorError(
                "live nexmon_csi capture requires root (monitor mode + nexutil)",
                ["sudo python main.py --mode csi --source csi"],
            )
        nexutil = shutil.which("nexutil")
        if not nexutil:
            raise self._unsupported()
        if not self.interface:
            from .capabilities import list_interfaces
            wireless = [i for i in list_interfaces() if i["wireless"]]
            if not wireless:
                raise self._unsupported()
            self.interface = wireless[0]["name"]
        if not shutil.which("makecsiparams"):
            raise CollectorError(
                "`makecsiparams` (from nexmon_csi utils) not found",
                ["build it: https://github.com/seemoo-lab/nexmon_csi  (utils/makecsiparams)",
                 "it builds the chanspec argument nexutil needs"],
            )

    def _configure_nexmon(self) -> None:
        from .capabilities import run

        cfg = self.config
        chanspec = cfg.get_path("csi.chanspec", "36/80") if cfg else "36/80"
        core = cfg.get_path("csi.core", 1) if cfg else 1
        size = cfg.get_path("csi.nexutil_size", 500) if cfg else 500
        port = cfg.get_path("csi.udp_port", self.udp_port) if cfg else self.udp_port
        self.udp_port = int(port)

        channel = cfg.get_path("channel") if cfg else None
        for cmd in (["ip", "link", "set", self.interface, "down"],
                    ["iw", "dev", self.interface, "set", "type", "monitor"],
                    ["ip", "link", "set", self.interface, "up"]):
            res = run(cmd)
            if not res["ok"] and not (cmd[1:3] == ["link", "set"] and "down" in cmd):
                raise CollectorError(f"failed to prepare {self.interface} for CSI: {' '.join(cmd)}\n"
                                     f"  {res['err'].strip()}")
        if channel:
            run(["iw", "dev", self.interface, "set", "channel", str(int(channel))])

        cmd = (f"nexutil -I{self.interface} -s{int(size)} -b -l34 "
               f"-v$(makecsiparams -c {chanspec} -C {int(core)})")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            raise CollectorError(
                f"nexutil failed to configure CSI extraction:\n  {cmd}\n  "
                f"{proc.stderr.strip() or proc.stdout.strip()}",
                ["is the patched nexmon firmware loaded on this chip?",
                 "does the chanspec match your chip's capabilities (chip must support the bw)?"],
            )
        self.status.update({"nexutil_cmd": cmd, "udp_port": self.udp_port})

    def _stream_nexmon(self) -> Iterator[Sample]:
        self._preflight_nexmon()
        self._configure_nexmon()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.udp_port))
        except OSError as exc:
            raise CollectorError(
                f"could not bind UDP port {self.udp_port}: {exc}",
                ["another csireader may already be running; kill it or change csi.udp_port"],
            ) from exc
        sock.settimeout(1.0)
        self.started_at = time.time()
        bad = 0
        got = 0
        deadline = time.time() + 20
        try:
            while not self.stopped:
                try:
                    data, _addr = sock.recvfrom(65535)
                except socket.timeout:
                    if got == 0 and time.time() > deadline:
                        raise CollectorError(
                            f"no CSI packets on UDP {self.udp_port} after 20 s",
                            ["did nexutil succeed? re-run the printed nexutil command by hand",
                             "check the firmware actually streams to this host/IP",
                             "the patched firmware reloads on reboot — re-install it if so"],
                        )
                    continue
                pkt = parsers.parse_nexmon_packet(data)
                if pkt is None:
                    bad += 1
                    if bad > 200 and got == 0:
                        self.status["warning"] = (
                            "packets arrived but none matched the known nexmon_csi layout — "
                            "your firmware's header may differ (see parsers.parse_nexmon_packet)"
                        )
                    continue
                got += 1
                amp_phase = parsers.nexmon_csi_to_amplitude_phase(pkt)
                yield self.emit(Sample(
                    t=time.time(),
                    rssi=pkt.get("rssi"),
                    source="csi_nexmon",
                    bssid=pkt.get("src_mac"),
                    amplitude=amp_phase["amplitude"],
                    phase=amp_phase["phase"],
                    n_subcarriers=pkt.get("n_subcarriers"),
                    meta={"core": pkt.get("core"), "chanspec": pkt.get("chanspec"),
                          "chip_version": pkt.get("chip_version"), "seq": pkt.get("seq")},
                ))
        finally:
            sock.close()

    # -- offline files via csiread -------------------------------------- #
    def _preflight_file(self) -> None:
        if not self.capture:
            raise CollectorError(
                "no CSI capture file given",
                ["pass --csi-pcap path/to/capture.pcap (or the Intel 5300 .dat log)"],
            )
        path = Path(self.capture).expanduser()
        if not path.exists():
            raise CollectorError(f"CSI capture not found: {path}")
        try:
            import csiread  # noqa: F401
        except Exception as exc:
            raise CollectorError(
                f"python package `csiread` is required to parse {path.name}: {exc}",
                ["pip install csiread",
                 "csiread parses nexmon (pcap), Intel 5300 (.dat) and ath9k (pcap) captures"],
            ) from exc
        self.capture = str(path)

    def _stream_file(self) -> Iterator[Sample]:
        self._preflight_file()
        rows = _load_csi_file(self.capture, self.tool, self.chip, self.config)
        self.started_at = time.time()
        self.status.update({"file": self.capture, "packets": len(rows)})
        for row in rows:
            if self.stopped:
                return
            yield self.emit(Sample(
                t=row.get("t", time.time()),
                rssi=row.get("rssi"),
                source=f"csi_{self.tool}",
                amplitude=row.get("amplitude"),
                phase=row.get("phase"),
                n_subcarriers=len(row.get("amplitude") or []) or None,
                meta={"file": Path(self.capture).name, "index": row.get("index")},
            ))

    # ------------------------------------------------------------------ #
    def stream(self) -> Iterator[Sample]:
        if self.tool == "nexmon":
            yield from self._stream_nexmon()
        else:
            yield from self._stream_file()

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# csiread wrapper
# --------------------------------------------------------------------------- #

def _load_csi_file(path: str, tool: str, chip: Optional[str], config: Any) -> List[Dict[str, Any]]:
    """Parse a CSI capture with ``csiread`` into Sample-ready dicts.

    ``csiread`` exposes a different class per toolchain; all of them give a
    complex ``csi`` array plus timestamps. Amplitude/phase are derived here so
    everything downstream sees the same shape.
    """
    import csiread  # guaranteed by preflight

    tool = tool.lower()
    tool = {"file": "nexmon", "pcap": "nexmon"}.get(tool, tool)

    if tool == "nexmon":
        chip = chip or "4358"
        cs = csiread.Nexmon(path, chip=chip, bw=80)
        cs.read()
        csi = cs.csi
        ts = getattr(cs, "sec", None)
        rssi = getattr(cs, "rssi", None)
    elif tool in ("iwl5300", "intel", "5300"):
        cs = csiread.Iwl5300(path)
        cs.read()
        csi = cs.csi
        ts = getattr(cs, "timestamp", None)
        rssi = getattr(cs, "rssi1", None)
    elif tool in ("atheros", "ath9k"):
        cs = csiread.Atheros(path, nrxnum=3 if chip in (None, "3") else 1)
        cs.read()
        csi = cs.csi
        ts = getattr(cs, "timestamp", None)
        rssi = getattr(cs, "rssi", None)
    else:
        raise CollectorError(f"unsupported --csi-tool '{tool}'",
                             ["supported: nexmon | iwl5300 | atheros"])

    import numpy as np

    rows: List[Dict[str, Any]] = []
    n = int(getattr(csi, "shape", [0])[0])
    for i in range(n):
        frame = np.asarray(csi[i])
        flat = frame.reshape(-1)
        finite = np.isfinite(flat)
        if not finite.any():
            continue
        amps = np.abs(flat[finite])
        phases = np.unwrap(np.angle(flat[finite]))
        t = float(ts[i]) if ts is not None and i < len(ts) else time.time()
        r = None
        if rssi is not None and i < len(rssi):
            try:
                r = float(np.asarray(rssi[i]).reshape(-1)[0])
            except Exception:
                r = None
        rows.append({
            "index": i,
            "t": t,
            "rssi": r,
            "amplitude": amps.tolist(),
            "phase": phases.tolist(),
        })
    return rows
