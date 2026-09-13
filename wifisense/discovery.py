"""Automatic discovery of every way this machine can sense WiFi.

`python main.py --discover` runs all of these probes, prints one ranked table,
and (with `--auto-setup`) writes the winning configuration to
``config/local.yaml`` so the next run "just works".

Probes, in priority order:

1. **Local NIC with monitor mode + tshark** — the richest RSSI source, needs root.
2. **Local NIC link/station RSSI** (`iw`, `/proc/net/wireless`) — free, no root.
3. **Home router as a sensor** — telnet/ssh reachability on the *default gateway
   only* (never other hosts) plus a short list of documented vendor defaults, so
   the router's per-client RSSI table can be polled without any new hardware.
4. **Android/Termux** — the phone-as-sensor path.
5. **CSI toolchains** (nexmon / Intel 5300 / ath9k / offline pcap).

The credential probe is deliberately narrow: it only ever targets the LAN's own
default gateway (your own router, the device you already administer) and reports
what it found — it is an unlock helper for your own equipment, not a scanner.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .collectors import capabilities as caps

# Documented vendor/ISP defaults (IndiHome ZTE/Fiberhome/Huawei, generic Linux
# routers). ZTE F670L factory-mode unlock prints the real telnet root password;
# these are the widely published defaults to try first.
TELNET_CREDENTIALS: Sequence[tuple[str, str]] = (
    ("root", "Zte521"),
    ("root", "Zte521!"),
    ("root", "root"),
    ("root", "admin"),
    ("admin", "admin"),
    ("telecomadmin", "admintelecom"),
    ("telecomadmin", "ntelecomadmin"),
    ("user", "user"),
    ("admin", "Telkomdso123"),
    ("root", "zte"),
)

HTTP_CREDENTIALS: Sequence[tuple[str, str]] = (
    ("admin", "admin"),
    ("user", "user"),
    ("telecomadmin", "admintelecom"),
    ("admin", "Telkomdso123"),
    ("support", "theworldisyours"),
)


@dataclass
class Candidate:
    kind: str
    name: str
    usable: bool
    detail: str
    config: Dict[str, Any] = field(default_factory=dict)
    command: str = ""
    rank: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "usable": self.usable,
                "detail": self.detail, "config": self.config,
                "command": self.command, "rank": self.rank}


# --------------------------------------------------------------------------- #
# network helpers (pure enough to unit-test)
# --------------------------------------------------------------------------- #

def parse_default_gateway(proc_route: str, ip_route: str = "") -> Optional[str]:
    """Default gateway from ``/proc/net/route`` (preferred) or ``ip route``."""
    for line in proc_route.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "00000000":
            hexip = parts[2]
            if len(hexip) == 8:
                octets = [int(hexip[i:i + 2], 16) for i in (6, 4, 2, 0)]
                return ".".join(str(o) for o in octets)
    m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", ip_route)
    return m.group(1) if m else None


def default_gateway() -> Optional[str]:
    try:
        proc = Path("/proc/net/route").read_text()
    except Exception:
        proc = ""
    ip_route = ""
    if shutil.which("ip"):
        res = caps.run(["ip", "route"])
        ip_route = res["out"]
    return parse_default_gateway(proc, ip_route)


def probe_port(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def http_banner(host: str, port: int = 80, timeout: float = 3.0) -> str:
    """Fetch the router's web title/server header (used to identify the model)."""
    import urllib.error
    import urllib.request

    for scheme in ("http", "https"):
        url = f"{scheme}://{host}:{port}/"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wifisense-discovery/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read(4096).decode("utf-8", "replace")
            title = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
            server = resp.headers.get("Server", "")
            return f"{server} | {title.group(1).strip()[:60]}" if title else server
        except Exception:
            continue
    return ""


def try_telnet_login(host: str, port: int = 23, credentials: Sequence[tuple[str, str]] = TELNET_CREDENTIALS,
                     timeout: float = 4.0) -> Optional[Dict[str, Any]]:
    """Try the documented defaults against the visitor's own gateway. Returns the
    credential that produced a shell prompt, with a probe command's output."""
    from .collectors.router import _Telnet

    if not probe_port(host, port, timeout=1.5):
        return None
    for user, password in credentials:
        session = None
        try:
            session = _Telnet(host, port, timeout=timeout, login_user=user,
                              login_password=password)
            out = session.command("echo WIFISENSE_OK; iw dev 2>/dev/null | head -5")
            if "WIFISENSE_OK" in out or "Interface" in out:
                return {"user": user, "password": password, "output": out.strip()[:400]}
        except Exception:
            continue
        finally:
            if session:
                try:
                    session.close()
                except Exception:
                    pass
    return None


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def discover_local_nics() -> List[Candidate]:
    report = caps.capability_report()
    found: List[Candidate] = []
    tools = caps.detect_tools()
    wireless = [i for i in report["interfaces"] if i["wireless"]]

    monitor = [i for i in wireless if i.get("monitor_capable")]
    if monitor and (tools.get("tshark") or tools.get("tcpdump")):
        iface = monitor[0]["name"]
        tool = "tshark" if tools.get("tshark") else "tcpdump"
        found.append(Candidate(
            kind="local_monitor", name=f"monitor mode on {iface}", usable=True, rank=1,
            detail=f"passive radiotap capture via {tool} (richest RSSI stream, "
                   f"{'root available' if caps.detect_os()['is_root'] else 'RUN WITH SUDO'})",
            config={"interface": iface, "mode": "rssi", "source": "radiotap"},
            command=f"{'sudo ' if not caps.detect_os()['is_root'] else ''}python main.py "
                    f"--source radiotap --interface {iface}",
        ))
    elif wireless and not (tools.get("tshark") or tools.get("tcpdump")):
        found.append(Candidate(
            kind="local_monitor", name="monitor mode blocked", usable=False, rank=60,
            detail="NIC supports monitor mode but neither tshark nor tcpdump is installed",
            command="sudo apt install tshark",
        ))

    if wireless and tools.get("iw"):
        linked = None
        for iface in wireless:
            res = caps.run(["iw", "dev", iface["name"], "link"])
            from .collectors import parsers
            if res["ok"] and parsers.parse_iw_link(res["out"]).get("connected"):
                linked = iface["name"]
                break
        iface = linked or wireless[0]["name"]
        found.append(Candidate(
            kind="local_iw", name=f"link RSSI on {iface}", usable=True, rank=5,
            detail=("NIC is associated to an AP" if linked else
                    "NIC present but not associated — connect to WiFi first"),
            config={"interface": iface, "mode": "rssi", "source": "iw"},
            command=f"python main.py --source iw --interface {iface}",
        ))
    elif wireless:
        found.append(Candidate(
            kind="local_proc", name="proc/net/wireless", usable=True, rank=30,
            detail="no `iw` installed; /proc fallback (less reliable on some drivers)",
            config={"mode": "rssi", "source": "proc"},
            command="sudo apt install iw   # recommended\npython main.py --source proc",
        ))

    if not wireless:
        found.append(Candidate(
            kind="none", name="no local wireless NIC", usable=False, rank=90,
            detail="this machine has no WiFi radio — sense from another device "
                   "(phone via Termux, or your router)",
            command="python main.py --source termux      # on an Android phone\n"
                    "python main.py --source router      # polling your router",
        ))
    return found


def discover_router(host: Optional[str] = None, try_credentials: bool = True) -> List[Candidate]:
    """Probe the default gateway (your own router) for a usable shell."""
    found: List[Candidate] = []
    gateway = host or default_gateway()
    if not gateway:
        return [Candidate(kind="router", name="router", usable=False, rank=80,
                          detail="could not determine the default gateway",
                          command="set router.host manually in config/default.yaml")]

    ports = {"telnet": 23, "ssh": 22}
    open_ports = [name for name, port in ports.items() if probe_port(gateway, port)]
    banner = http_banner(gateway, 80) or http_banner(gateway, 8080)
    if open_ports:
        detail = f"gateway {gateway}: open {', '.join(open_ports)}" + (f" | web: {banner}" if banner else "")
        creds: Dict[str, Any] = {}
        if try_credentials:
            for name in open_ports:
                if name == "telnet":
                    hit = try_telnet_login(gateway, 23)
                    if hit:
                        creds = hit
                        break
        protocol = "telnet" if "telnet" in open_ports else "ssh"
        usable = bool(creds) or ("telnet" in open_ports and not try_credentials)
        found.append(Candidate(
            kind="router", name=f"router at {gateway}", usable=usable, rank=2,
            detail=(f"{detail} — shell OK with {creds.get('user')}/{creds.get('password')}"
                    if creds else
                    f"{detail} — shell not confirmed "
                    f"({'defaults rejected; run the ZTE factory-mode unlock' if try_credentials else 'not attempted'})"),
            config={
                "router.host": gateway,
                "router.protocol": protocol,
                "router.port": 23 if protocol == "telnet" else 22,
                "router.user": creds.get("user") if creds else None,
                "router.password": creds.get("password") if creds else None,
            },
            command=(f"python main.py --source router --router-host {gateway} "
                     f"--router-protocol {protocol}"),
        ))
    else:
        found.append(Candidate(
            kind="router", name=f"router at {gateway}", usable=False, rank=40,
            detail=f"no telnet/ssh on the gateway{f' | web: {banner}' if banner else ''} — "
                   f"web-UI only (RSSI visible manually under Device Info -> Wireless)",
            config={"router.host": gateway},
            command="python main.py --discover --probe-host " + gateway,
        ))
    return found


def discover_other_paths() -> List[Candidate]:
    found: List[Candidate] = []
    tools = caps.detect_tools()
    if tools.get("termux-wifi-connectioninfo"):
        found.append(Candidate(
            kind="termux", name="this Android phone (Termux)", usable=True, rank=3,
            detail="termux-api reports this phone's own WiFi RSSI (~1 Hz). "
                   "Hold termux-wake-lock so Android does not freeze the loop.",
            config={"mode": "rssi", "source": "termux"},
            command="termux-wake-lock\npython main.py --source termux",
        ))
    csi = caps.detect_csi()
    if csi.get("live_csi"):
        found.append(Candidate(
            kind="csi", name=f"CSI: {', '.join(csi['backends'])}", usable=True, rank=0,
            detail="per-subcarrier amplitude+phase available — the only mode that can "
                   "see fine motion and (experimentally) breathing",
            config={"mode": "csi", "source": "csi"},
            command="sudo python main.py --mode csi",
        ))
    elif csi.get("offline_csi"):
        found.append(Candidate(
            kind="csi_offline", name="CSI pcap replay (csiread)", usable=True, rank=20,
            detail="csiread installed: replay a CSI capture recorded on CSI-capable hardware",
            config={"mode": "csi", "source": "csi_pcap"},
            command="python main.py --mode csi --source csi_pcap --csi-pcap capture.pcap",
        ))
    return found


def discover(host: Optional[str] = None, try_credentials: bool = True) -> Dict[str, Any]:
    candidates: List[Candidate] = []
    candidates += discover_local_nics()
    candidates += discover_other_paths()
    candidates += discover_router(host, try_credentials=try_credentials)
    candidates.sort(key=lambda c: (not c.usable, c.rank))
    best = next((c for c in candidates if c.usable), None)
    return {
        "gateway": host or default_gateway(),
        "candidates": [c.to_dict() for c in candidates],
        "best": best.to_dict() if best else None,
    }


def format_discovery(result: Dict[str, Any], color: bool = True) -> str:
    c = {"b": "\033[1m" if color else "", "g": "\033[32m" if color else "",
         "r": "\033[31m" if color else "", "d": "\033[2m" if color else "",
         "0": "\033[0m" if color else ""}
    lines = [f"{c['b']}[WiFi Sense] discovery{c['0']}   gateway: {result.get('gateway') or '-'}",
             ""]
    for cand in result["candidates"]:
        mark = f"{c['g']}[ok]{c['0']}" if cand["usable"] else f"{c['r']}[--]{c['0']}"
        lines.append(f"  {mark} {cand['name']}")
        lines.append(f"       {cand['detail']}")
        if cand.get("command"):
            first = cand["command"].splitlines()[0]
            lines.append(f"       {c['d']}→ {first}{c['0']}")
    best = result.get("best")
    lines.append("")
    if best:
        lines.append(f"  {c['b']}BEST: {best['name']}{c['0']}")
        lines.append(f"  run: {' '.join(best['command'].split())}")
    else:
        lines.append(f"  {c['r']}No usable sensing source found on this machine.{c['0']}")
        lines.append("  Options: run this on an Android phone (Termux), unlock the router's "
                     "shell, or attach a USB WiFi adapter.")
    return "\n".join(lines)


def auto_setup(config: Any, result: Dict[str, Any], path: str | Path = "config/local.yaml") -> Path:
    """Write the discovered settings into a YAML overlay (never touches default.yaml)."""
    import yaml

    best = result.get("best")
    overlay: Dict[str, Any] = {"_auto_discovered_at": __import__("time").time()}
    if best:
        overlay.update({k: v for k, v in (best.get("config") or {}).items() if v is not None})
        if best.get("config", {}).get("source"):
            overlay["source"] = best["config"]["source"]
        if best.get("config", {}).get("mode"):
            overlay["mode"] = best["config"]["mode"]

    # nested keys (router.*) go into their own mapping
    nested: Dict[str, Any] = {}
    for key in list(overlay):
        if "." in key:
            head, _, tail = key.partition(".")
            nested.setdefault(head, {})[tail] = overlay.pop(key)
    overlay.update(nested)

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if out.exists():
        try:
            existing = yaml.safe_load(out.read_text()) or {}
        except Exception:
            existing = {}
    existing.update(overlay)
    out.write_text(yaml.safe_dump(existing, sort_keys=False, default_flow_style=False))
    return out


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(format_discovery(discover()))
