"""Router-as-sensor backend (best free option for a home with an ISP router).

The home router is an Access Point that runs Linux and is on 24/7 — it is
already listening to every client's signal strength. If you can get a shell on
it (telnet/SSH root — common on IndiHome ZTE routers such as the F670L via the
known factory-mode unlock, or superadmin credentials), then this backend polls
``iw dev <iface> station dump`` and turns **every connected device into an
independent RSSI probe of the room**. No extra hardware, no laptop, no cost.

The web admin page shows the same number manually (Device Info → Wireless →
Associated Client List → RSSI); this backend just reads it automatically.

Two transports, zero new Python dependencies:

* telnet — a minimal RFC 854 client over a raw socket (stdlib only). Works on
  the ZTE/Fiberhome/Huawei ONTs that expose telnetd.
* ssh — the system ``ssh`` client via ``sshpass`` (for password auth) or SSH
  keys (``BatchMode=yes``). Use when the router only exposes sshd.

Each poll yields ONE sample whose RSSI is the mean over all connected clients,
plus the per-client breakdown in ``meta`` (the multi-link view that the zone
localization feature wants). When no client is connected the sample is skipped
and the failure is surfaced as a clear error after a grace period.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional

from . import parsers
from .base import Collector, CollectorError, Sample


# --------------------------------------------------------------------------- #
# minimal telnet (stdlib-only, no deprecated telnetlib)
# --------------------------------------------------------------------------- #

class _Telnet:
    """Tiny RFC 854 client: login + command() + close()."""

    def __init__(self, host: str, port: int = 23, timeout: float = 8.0,
                 login_user: str = "", login_password: str = "",
                 prompt: str = r"[#$>]"):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.prompt = prompt
        self.sock: Optional[socket.socket] = None
        self.buffer = b""
        self._login(login_user, login_password)

    # -- io -------------------------------------------------------------- #
    def _read_until(self, marker: bytes, deadline: float) -> bytes:
        out = b""
        while time.time() < deadline:
            if self.buffer:
                chunk = self.buffer
                self.buffer = b""
            else:
                try:
                    chunk = self.sock.recv(4096) if self.sock else b""
                except socket.timeout:
                    break
                except OSError:
                    break
                if not chunk:
                    break
            chunk = self._strip_iac(chunk)
            out += chunk
            if marker in out:
                break
        return out

    @staticmethod
    def _strip_iac(data: bytes) -> bytes:
        """Remove telnet IAC negotiation sequences entirely (read path)."""
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            b = data[i]
            if b == 0xFF:                     # IAC: skip the whole sequence
                if i + 1 < n:
                    cmd = data[i + 1]
                    if cmd == 0xFA:           # SB ... SE: skip until IAC SE
                        j = i + 2
                        while j + 1 < n and not (data[j] == 0xFF and data[j + 1] == 0xF0):
                            j += 1
                        i = min(j + 2, n)
                        continue
                    i += 2                    # IAC <cmd> [<opt>]
                    continue
                break
            out.append(b)
            i += 1
        return bytes(out)

    def _send(self, data: bytes) -> None:
        if self.sock:
            try:
                self.sock.sendall(data)
            except OSError as exc:
                raise CollectorError(f"telnet write to {self.host}:{self.port} failed: {exc}") from exc

    # -- lifecycle -------------------------------------------------------- #
    def _login(self, user: str, password: str) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)
        except OSError as exc:
            raise CollectorError(
                f"cannot reach {self.host}:{self.port} (telnet)",
                ["is the router's telnet enabled? (ZTE: enable via factory mode or superadmin)",
                 "check with: nc -zv {self.host} {self.port}"] if False else
                [f"is the router's telnet enabled? check: nc -zv {self.host} {self.port}",
                 "the IndiHome ZTE unlock path: see README 'router as sensor' section"],
            ) from exc

        deadline = time.time() + self.timeout
        banner = self._read_until(b"login:", deadline) or self._read_until(b"Login:", deadline)
        if b"login:" not in banner.lower() and b"username:" not in banner.lower():
            # some routers print a menu without login; try to continue anyway
            pass
        if user:
            self._send(user.encode() + b"\r\n")
            time.sleep(0.3)
            self._read_until(b"assword:", deadline)
        if password:
            self._send(password.encode() + b"\r\n")
            time.sleep(0.5)
        # consume the prompt (best-effort)
        self._read_until(b"#", deadline)

    def command(self, cmd: str, wait_s: float = 1.2) -> str:
        if not self.sock:
            raise CollectorError("telnet session is closed")
        self._send(cmd.encode() + b"\r\n")
        time.sleep(wait_s)
        deadline = time.time() + self.timeout
        data = self._read_until(b"#", deadline)
        try:
            return data.decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def close(self) -> None:
        if self.sock:
            try:
                self._send(b"exit\r\n")
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# --------------------------------------------------------------------------- #
# collector
# --------------------------------------------------------------------------- #

DEFAULT_COMMANDS = (
    "iw dev {iface} station dump",
    "cat /proc/net/wireless",
)


class RouterRSSICollector(Collector):
    """Poll per-client RSSI from a Linux-based home router over telnet/SSH."""

    name = "router"
    mode = "rssi"

    def __init__(self, config: Any = None, interface: Optional[str] = None,
                 host: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, protocol: Optional[str] = None,
                 port: Optional[int] = None):
        super().__init__(config, interface)
        cfg = (lambda path, default: config.get_path(path, default) if config else default)
        self.host = host or cfg("router.host", None)
        self.user = user or cfg("router.user", None)
        self.password = password or cfg("router.password", None)
        self.protocol = (protocol or cfg("router.protocol", "telnet")).lower()
        self.port = port or int(cfg("router.port", 23 if self.protocol == "telnet" else 22))
        self.iface = interface or cfg("router.iface", None)
        self.command = str(cfg("router.command", "")).strip() or None
        self.interval = max(float(cfg("router.interval_s", 1.0)), 0.5)
        self.grace_s = float(cfg("router.grace_s", 20.0))
        self._transport = None
        self._last_clients: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def preflight(self) -> None:
        if not self.host:
            raise CollectorError(
                "no router host configured",
                ["set router.host (e.g. 192.168.1.1) in config/default.yaml",
                 "or pass --source router --router-host 192.168.1.1 ... "
                 "(CLI flags for router are: --router-host/--router-user/--router-pass/--router-protocol)"],
            )
        if self.protocol == "ssh" and not shutil.which("sshpass"):
            # password auth without sshpass: we can still try key-based auth
            pass

    def _connect(self) -> None:
        if self.protocol == "telnet":
            self._transport = _Telnet(self.host, self.port, login_user=self.user or "",
                                      login_password=self.password or "")
        else:
            self._transport = _SSHWrapper(self.host, self.port, self.user or "root",
                                          self.password or "")

    def _run_command(self) -> str:
        if self.protocol == "telnet":
            cmd = self.command or DEFAULT_COMMANDS[0].format(iface=self.iface or "wlan0")
            return self._transport.command(cmd) if self._transport else ""
        return self._transport.command(self.command or DEFAULT_COMMANDS[0].format(iface=self.iface or "wlan0")) if self._transport else ""

    def _parse_clients(self, output: str) -> List[Dict[str, Any]]:
        stations = parsers.parse_iw_station_dump(output)
        return [s for s in stations if s.get("signal_dbm") is not None]

    def stream(self) -> Iterator[Sample]:
        self.preflight()
        self._connect()
        self.started_at = time.time()
        self.status.update({"host": self.host, "protocol": self.protocol, "iface": self.iface,
                            "interval_s": self.interval})
        misses = 0
        try:
            while not self.stopped:
                try:
                    output = self._run_command()
                    clients = self._parse_clients(output)
                    self._last_clients = {c["mac"]: float(c["signal_dbm"]) for c in clients}
                except CollectorError:
                    misses += 1
                    if misses * self.interval > self.grace_s:
                        raise
                    self._sleep(self.interval)
                    continue

                if clients:
                    misses = 0
                    mean = sum(c["signal_dbm"] for c in clients) / len(clients)
                    self.status["clients"] = len(clients)
                    yield self.emit(Sample(
                        t=time.time(),
                        rssi=round(mean, 2),
                        source=self.name,
                        meta={
                            "clients": len(clients),
                            "per_client": {c["mac"]: float(c["signal_dbm"]) for c in clients},
                            "router": self.host,
                        },
                    ))
                else:
                    misses += 1
                    if misses * self.interval > self.grace_s:
                        raise CollectorError(
                            f"router {self.host} reports no connected WiFi clients for "
                            f"{self.grace_s:.0f}s",
                            ["are family devices actually connected to the router's WiFi?",
                             "check the router's web UI: Device Info -> Wireless -> Associated "
                             "Client List",
                             "the router may need a different interface: set router.iface "
                             "(try wlan0, ath0, ra0, wl0)"],
                        )
                self._sleep(self.interval)
        finally:
            if self._transport:
                try:
                    self._transport.close()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()


class _SSHWrapper:
    """Password/key SSH via the system ssh client; sshpass used when available."""

    def __init__(self, host: str, port: int, user: str, password: str):
        self.host = host
        self.port = int(port)
        self.user = user or "root"
        self.password = password
        self.base = [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=8", "-p", str(self.port),
        ]
        if shutil.which("sshpass") and self.password:
            self.base = ["sshpass", "-p", self.password] + self.base

    def command(self, cmd: str, timeout: float = 15.0) -> str:
        argv = self.base + [f"{self.user}@{self.host}", cmd]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise CollectorError("ssh not found on PATH",
                                 ["install openssh-client (Debian/Ubuntu: sudo apt install openssh-client)"]) from None
        except subprocess.TimeoutExpired:
            raise CollectorError(f"ssh to {self.host} timed out") from None
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().split("\n")[-1]
            raise CollectorError(f"ssh command failed on {self.host}: {err}",
                                 ["the router may not accept the password (root shell required)",
                                  "or sshpass is missing: sudo apt install sshpass",
                                  "or use telnet instead: --router-protocol telnet"])
        return proc.stdout

    def close(self) -> None:
        return
