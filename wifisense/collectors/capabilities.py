"""Hardware / OS capability detection.

Answers, before any sensing starts:

1. which OS are we on, with which tools available (``iw``, ``tshark``, ...)?
2. which network interfaces exist and which of them are wireless?
3. does the wireless NIC support monitor mode (required for passive radiotap
   RSSI sensing, which is by far the best RSSI backend)?
4. does anything on this machine expose **CSI** (Channel State Information)?
5. given all that, which mode/backend should the runner pick, and what are the
   honest limitations of that choice?

Never raises for missing hardware: every probe is guarded and reports *why* it
failed. Detection is read-only — it never changes interface state.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import parsers

TIMEOUT = 6.0

# --------------------------------------------------------------------------- #
# low level shell helpers
# --------------------------------------------------------------------------- #


def run(cmd: List[str], timeout: float = TIMEOUT) -> Dict[str, Any]:
    """Run a command, never raise, always return ``{ok, out, err, code}``."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"ok": proc.returncode == 0, "out": proc.stdout,
                "err": proc.stderr, "code": proc.returncode, "cmd": cmd}
    except FileNotFoundError:
        return {"ok": False, "out": "", "err": f"{cmd[0]} not found", "code": 127, "cmd": cmd}
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "", "err": "timeout", "code": 124, "cmd": cmd}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "out": "", "err": str(exc), "code": -1, "cmd": cmd}


def _read_text(path: str | Path, limit: int = 65536) -> Optional[str]:
    try:
        p = Path(path)
        if p.is_file():
            return p.read_text(errors="replace")[:limit]
        # debugfs / procfs pseudo files report size 0 but are still readable
        with open(p, "r", errors="replace") as fh:
            return fh.read(limit)
    except Exception:
        return None


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


# --------------------------------------------------------------------------- #
# OS
# --------------------------------------------------------------------------- #

def detect_os() -> Dict[str, Any]:
    system = platform.system().lower()
    info: Dict[str, Any] = {
        "system": system,
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "is_root": _is_root(),
        "is_linux": system == "linux",
        "is_macos": system == "darwin",
        "is_windows": system == "windows",
        "distro": None,
    }
    if system == "linux":
        try:
            osrel = Path("/etc/os-release").read_text(errors="replace")
            for line in osrel.splitlines():
                if line.startswith("PRETTY_NAME="):
                    info["distro"] = line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass
        # container / VM sniffing: monitor mode is usually impossible in a container
        info["containerized"] = any(
            Path(p).exists() for p in ("/.dockerenv", "/run/.containerenv")
        )
    return info


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #

TOOLS = ("iw", "ip", "nmcli", "iwconfig", "iwlist", "tshark", "tcpdump",
         "rfkill", "netsh", "airport", "nexutil", "ifconfig", "ip",
         "termux-wifi-connectioninfo", "termux-wake-lock", "termux-battery-status")


def detect_tools() -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {}
    for tool in TOOLS:
        found[tool] = shutil.which(tool)
    return found


# --------------------------------------------------------------------------- #
# interfaces
# --------------------------------------------------------------------------- #

def _sys_net_interfaces() -> List[str]:
    net = Path("/sys/class/net")
    if not net.is_dir():
        return []
    return sorted(p.name for p in net.iterdir())


def _driver_of(iface: str) -> Optional[str]:
    link = Path(f"/sys/class/net/{iface}/device/driver")
    try:
        return Path(os.path.realpath(link)).name
    except Exception:
        return None


def _iface_is_wireless(iface: str, tools: Dict[str, Optional[str]]) -> bool:
    if Path(f"/sys/class/net/{iface}/wireless").exists():
        return True
    if Path(f"/sys/class/net/{iface}/phy80211").exists():
        return True
    if tools.get("iwconfig"):
        res = run(["iwconfig", iface])
        if res["ok"] and "no wireless extensions" not in res["out"]:
            return True
    if platform.system().lower() == "darwin":
        return iface.startswith("en") or iface.startswith("awdl")
    return False


def list_interfaces() -> List[Dict[str, Any]]:
    tools = detect_tools()
    ifaces: List[Dict[str, Any]] = []
    names = _sys_net_interfaces()
    if not names:
        # macOS / Windows without /sys
        if tools.get("airport") or platform.system().lower() == "darwin":
            res = run(["networksetup", "-listallhardwareports"])
            for line in res["out"].splitlines():
                if line.startswith("Device:"):
                    names.append(line.split(":", 1)[1].strip())
    for name in names:
        if name == "lo":
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "wireless": _iface_is_wireless(name, tools),
            "driver": _driver_of(name),
            "operstate": None,
            "type": None,
            "monitor_capable": None,
        }
        state = _read_text(f"/sys/class/net/{name}/operstate")
        if state:
            entry["operstate"] = state.strip()
        if entry["wireless"]:
            entry["monitor_capable"] = monitor_mode_supported(name)
            phy = _phy_of(name)
            entry["phy"] = phy
            if phy and tools.get("iw"):
                res = run(["iw", "phy", phy, "info"])
                if res["ok"]:
                    entry["bands"] = parsers.parse_iw_phy_bands(res["out"])["bands"]
                    entry["n_antennas"] = parsers.parse_iw_phy_bands(res["out"])["n_antennas"]
        ifaces.append(entry)
    return ifaces


def _phy_of(iface: str) -> Optional[str]:
    try:
        for entry in sorted(Path(f"/sys/class/net/{iface}/phy80211").iterdir()):
            if entry.name.startswith("phy"):
                return entry.name
    except Exception:
        pass
    link = Path(f"/sys/class/net/{iface}/phy80211")
    try:
        return Path(os.path.realpath(link)).name
    except Exception:
        return None


def monitor_mode_supported(iface: str) -> Optional[bool]:
    """Read-only probe: does the phy advertise ``monitor`` as an interface mode?"""
    tools = detect_tools()
    if not tools.get("iw"):
        return None
    res = run(["iw", "dev", iface])
    modes: List[str] = []
    phy = None
    if res["ok"]:
        for rec in parsers.parse_iw_dev(res["out"]):
            if rec.get("iface") == iface:
                phy = rec.get("phy")
    if phy is None:
        phy = _phy_of(iface)
    if phy:
        res = run(["iw", "phy", phy, "info"])
        if res["ok"]:
            modes = parsers.parse_iw_phy_modes(res["out"])
    if modes:
        return "monitor" in modes
    # Fallback: nmcli reports what it can do
    if tools.get("nmcli"):
        res = run(["nmcli", "-f", "WIFI-PROPERTIES.MONITOR-MODE", "dev", "show", iface])
        if res["ok"]:
            if "yes" in res["out"].lower():
                return True
            if "no" in res["out"].lower():
                return False
    return None


# --------------------------------------------------------------------------- #
# CSI capability
# --------------------------------------------------------------------------- #

def detect_csi() -> Dict[str, Any]:
    """Probe every known Linux CSI toolchain. Read-only, no firmware loading."""
    report: Dict[str, Any] = {"available": False, "backends": [], "checks": []}
    tools = detect_tools()

    def note(name: str, ok: bool, detail: str, **extra: Any) -> None:
        entry = {"backend": name, "available": ok, "detail": detail}
        entry.update(extra)
        report["checks"].append(entry)
        if ok:
            report["backends"].append(name)
            report["available"] = True

    # 1. nexmon_csi (Broadcom BCM43xx: RPi 3B+/4/5, Nexus 5, some routers).
    #    Needs patched firmware + the `nexutil` binary; CSI is then streamed
    #    over UDP by the firmware's csi module.
    nexutil = tools.get("nexutil") or shutil.which("makecsiparams")
    nexmon_mod = False
    modules = _read_text("/proc/modules", limit=200000) or ""
    if "nexmon" in modules:
        nexmon_mod = True
    nexmon_debugfs = any(Path(glob_root).exists() for glob_root in (
        "/sys/kernel/debug/ieee80211",
    )) and bool(_read_text("/sys/kernel/debug/ieee80211", limit=4096) and "nexmon" in
                (_read_text("/sys/kernel/debug/ieee80211", limit=4096) or ""))
    if nexutil:
        note("nexmon_csi", True,
             f"nexutil found at {nexutil}; CSI streamed over UDP (default port 5500)",
             binary=nexutil, module_loaded=nexmon_mod)
    else:
        note("nexmon_csi", False,
             "nexutil not found — nexmon_csi firmware/tools are not installed "
             "(see README: CSI section)")

    # 2. Intel 5300 / Linux 802.11n CSI Tool (modified iwlwifi + iwl_connector).
    conn_log = _read_text("/proc/net/iwl_conn")
    conn_param = _read_text("/sys/module/iwlwifi/parameters/connector_log")
    connector_mod = "iwl_connector" in modules or "iwlwifi" in modules and conn_param is not None
    if conn_log is not None:
        note("iwl5300_csi", True,
             "/proc/net/iwl_conn present — Intel 5300 Linux 802.11n CSI Tool active",
             module_loaded=connector_mod)
    elif conn_param is not None:
        note("iwl5300_csi", False,
             "iwlwifi connector_log parameter exists but /proc/net/iwl_conn is missing — "
             "kernel module loaded without the CSI patch")
    else:
        note("iwl5300_csi", False,
             "no Intel 5300 CSI Tool kernel module (custom iwlwifi + iwl_connector not loaded)")

    # 3. Atheros ath9k CSI tool (modified ath9k + recv_csi userspace).
    recv_csi = shutil.which("recv_csi")
    ath_csi_active = False
    try:
        for phy in Path("/sys/kernel/debug/ieee80211").iterdir():
            if (phy / "ath9k").exists() and any((phy / "ath9k").iterdir()):
                ath_csi_active = True
    except Exception:
        pass
    if recv_csi:
        note("ath9k_csi", True, f"recv_csi found at {recv_csi}", binary=recv_csi)
    else:
        note("ath9k_csi", False,
             "recv_csi / ath9k CSI driver not installed "
             "(debugfs ath9k CSI nodes %s)"
             % ("present" if ath_csi_active else "absent"))

    # 4. Offline CSI captures via python `csiread` — always usable when a
    #    capture file produced by ANY of the tools above is available.
    try:
        import csiread  # noqa: F401
        note("csiread_pcap", True,
             "python `csiread` installed — offline CSI pcaps (nexmon/iwl5300/atheros) can be parsed")
    except Exception:
        note("csiread_pcap", False,
             "python `csiread` not installed (pip install csiread) — offline CSI pcap parsing unavailable")

    report["live_csi"] = any(b in report["backends"] for b in ("nexmon_csi", "iwl5300_csi", "ath9k_csi"))
    report["offline_csi"] = "csiread_pcap" in report["backends"]
    return report


# --------------------------------------------------------------------------- #
# full report + mode decision
# --------------------------------------------------------------------------- #

LIMITATIONS = {
    "csi": [
        "CSI carries per-subcarrier amplitude AND phase — the only mode where",
        "fine-grained motion, and in ideal conditions breathing rate, is measurable.",
        "Needs a patched NIC/driver (nexmon_csi / Intel 5300 / ath9k); stock NICs cannot do this.",
        "Absolute phase is contaminated by carrier/SFO offsets — only relative phase is usable.",
    ],
    "rssi_radiotap": [
        "Passive monitor-mode capture: every nearby frame gives one RSSI sample,",
        "so the sample rate is high (tens..hundreds per second) and the time series is rich.",
        "RSSI is a single scalar per frame — it cannot resolve subcarriers, so fine",
        "physiological detail (heartbeat) is out of reach; presence/motion is well within reach.",
        "Monitor mode needs root and a NIC that supports it; the link to the AP is not used.",
    ],
    "rssi_iw": [
        "Polls the RSSI of the NIC's OWN link to the AP: ~1-2 samples/second.",
        "Coarse but zero-setup. Works on any stock Linux NIC in station mode.",
        "Because only one link is observed, sensitivity to breathing is poor;",
        "walking into the room is reliably visible as a variance/link-quality shift.",
    ],
    "rssi_proc": [
        "Reads /proc/net/wireless: last-resort backend, no root, no dependencies.",
        "Several modern drivers report a bogus constant level — flagged at runtime.",
    ],
    "rssi_netsh": [
        "Windows: netsh reports Signal as a 0-100 % bar, converted to dBm.",
        "Quantisation is coarse (0.5 dBm steps) and update rate is ~1 Hz.",
    ],
    "rssi_termux": [
        "Android/Termux: the phone's own link RSSI, read from the Android WiFi service.",
        "~1 Hz, quantised to whole dBm, and only one link is observed — the same",
        "sensitivity as the `iw` link backend. Hold termux-wake-lock so Android does",
        "not suspend the loop in the background.",
    ],
    "rssi_airport": [
        "macOS: the legacy `airport -I` utility (removed in recent releases).",
        "Reports RSSI + noise for the current link at ~1 Hz.",
    ],
    "synthetic": [
        "SYNTHETIC SOURCE — this is generated demo data, NOT radio measurements.",
        "Only for tests, UI work and pipeline demos. It never detects a real person.",
    ],
}


def choose_backend(os_info: Dict[str, Any], ifaces: List[Dict[str, Any]],
                   tools: Dict[str, Optional[str]], csi: Dict[str, Any],
                   mode: str = "auto", source: str = "auto") -> Dict[str, Any]:
    """Decide mode + backend, and explain the decision and its limitations."""
    wireless = [i for i in ifaces if i["wireless"]]
    monitor_ok = [i for i in wireless if i.get("monitor_capable")]

    decision: Dict[str, Any] = {
        "mode": None, "source": None, "interface": None,
        "reasons": [], "limitations": [], "fallbacks": [],
    }
    reasons = decision["reasons"]

    # ---- forced synthetic (explicitly requested only) ----
    if source == "synthetic":
        decision.update(mode="csi" if mode == "csi" else "rssi", source="synthetic",
                        interface=None)
        reasons.append("synthetic source explicitly requested — DEMO DATA ONLY")
        decision["limitations"] = LIMITATIONS["synthetic"]
        return decision

    # ---- CSI ----
    if mode in ("auto", "csi") and csi.get("live_csi") and source in ("auto", "csi"):
        decision.update(mode="csi", source="csi",
                        interface=(wireless[0]["name"] if wireless else None))
        reasons.append(f"live CSI backend available: {', '.join(csi['backends'])}")
        decision["limitations"] = LIMITATIONS["csi"]
        return decision

    if mode == "csi" and not csi.get("live_csi"):
        reasons.append("CSI requested but no live CSI backend detected on this machine")
        decision["fallbacks"].append("rssi")
        if source not in ("auto", "csi"):
            decision.update(mode="rssi", source="auto")
            return decision
        # fall through to RSSI with a loud explanation

    # ---- offline CSI capture ----
    if source == "csi_pcap":
        decision.update(mode="csi", source="csi_pcap", interface=None)
        reasons.append("offline CSI pcap replay requested")
        if not csi.get("offline_csi"):
            decision["fallbacks"].append("install csiread: pip install csiread")
        decision["limitations"] = LIMITATIONS["csi"]
        return decision

    # ---- RSSI ----
    if source == "radiotap" or (source == "auto" and monitor_ok and tools.get("tshark")):
        iface = (monitor_ok[0]["name"] if monitor_ok else (wireless[0]["name"] if wireless else None))
        decision.update(mode="rssi", source="radiotap", interface=iface)
        reasons.append("monitor-mode NIC + tshark: passive radiotap RSSI capture (best RSSI backend)")
        decision["limitations"] = LIMITATIONS["rssi_radiotap"]
        if not _is_root():
            decision["reasons"].append("root is required to enable monitor mode — run with sudo")
        return decision

    if source == "auto" and monitor_ok and not tools.get("tshark") and tools.get("tcpdump"):
        iface = monitor_ok[0]["name"]
        decision.update(mode="rssi", source="radiotap", interface=iface)
        reasons.append("no tshark, falling back to tcpdump radiotap parsing (fewer fields, "
                       "no per-frame MAC/freq filtering)")
        decision["limitations"] = LIMITATIONS["rssi_radiotap"]
        return decision

    if source in ("auto", "iw") and tools.get("iw") and wireless:
        # prefer an interface that is actually associated — that is what gives RSSI
        linked = None
        for iface in wireless:
            res = run(["iw", "dev", iface["name"], "link"])
            if res["ok"] and parsers.parse_iw_link(res["out"]).get("connected"):
                linked = iface["name"]
                break
        decision.update(mode="rssi", source="iw", interface=linked or wireless[0]["name"])
        if linked:
            reasons.append(f"{linked} is associated with an AP — polling its link RSSI with `iw`")
        else:
            reasons.append("no associated wireless link found — `iw` backend started anyway, "
                           "it will error clearly until the NIC connects (or use monitor mode)")
        decision["limitations"] = LIMITATIONS["rssi_iw"]
        return decision

    if source == "termux" or (source == "auto" and tools.get("termux-wifi-connectioninfo")):
        if tools.get("termux-wifi-connectioninfo"):
            decision.update(mode="rssi", source="termux", interface=None)
            reasons.append("running on Android/Termux: reading this phone's own link RSSI "
                           "through the Android WiFi service (termux-api)")
            decision["limitations"] = LIMITATIONS["rssi_termux"]
            if not tools.get("termux-wake-lock"):
                decision["reasons"].append("termux-wake-lock is missing — Android will freeze the "
                                           "loop in the background")
            return decision

    if source in ("auto", "proc") and os_info["is_linux"]:
        decision.update(mode="rssi", source="proc",
                        interface=(wireless[0]["name"] if wireless else None))
        reasons.append("/proc/net/wireless fallback (no root, no extra tools)")
        decision["limitations"] = LIMITATIONS["rssi_proc"]
        return decision

    if source in ("auto", "netsh") and os_info["is_windows"]:
        decision.update(mode="rssi", source="netsh", interface=None)
        reasons.append("Windows: netsh wlan signal polling (best effort, ~1 Hz)")
        decision["limitations"] = LIMITATIONS["rssi_netsh"]
        return decision

    if source in ("auto", "airport") and os_info["is_macos"]:
        decision.update(mode="rssi", source="airport", interface=None)
        reasons.append("macOS: airport -I link RSSI polling (best effort)")
        decision["limitations"] = LIMITATIONS["rssi_airport"]
        return decision

    decision["reasons"].append("no usable radio backend found on this machine")
    decision["limitations"] = [
        "No wireless interface was detected at all.",
        "If this is a container/VM, pass the NIC through to the guest, or run on the host.",
        "For UI / pipeline testing only, use:  python main.py --source synthetic",
    ]
    return decision


def capability_report(mode: str = "auto", source: str = "auto") -> Dict[str, Any]:
    os_info = detect_os()
    tools = detect_tools()
    ifaces = list_interfaces()
    csi = detect_csi()
    decision = choose_backend(os_info, ifaces, tools, csi, mode=mode, source=source)
    return {
        "os": os_info,
        "tools": {k: v for k, v in tools.items() if v},
        "tools_missing": [k for k, v in tools.items() if not v],
        "interfaces": ifaces,
        "csi": csi,
        "decision": decision,
    }


def format_report(rep: Dict[str, Any], color: bool = True) -> str:
    """Human-readable capability report (printed at startup and by --capabilities)."""
    c = {
        "b": "\033[1m" if color else "",
        "d": "\033[2m" if color else "",
        "g": "\033[32m" if color else "",
        "y": "\033[33m" if color else "",
        "r": "\033[31m" if color else "",
        "0": "\033[0m" if color else "",
    }
    os_info = rep["os"]
    lines = [
        f"{c['b']}[WiFi Sense] capability report{c['0']}",
        f"  OS         : {os_info['system']} {os_info['release']} ({os_info['machine']})"
        + (f" — {os_info['distro']}" if os_info.get("distro") else ""),
        f"  Python     : {os_info['python']}   root: {'yes' if os_info['is_root'] else 'no'}",
    ]
    if os_info.get("containerized"):
        lines.append(f"  {c['y']}! running inside a container: monitor mode + raw radio access are "
                     f"usually impossible here{c['0']}")

    wireless = [i for i in rep["interfaces"] if i["wireless"]]
    lines.append(f"  Interfaces : {len(rep['interfaces'])} total, {len(wireless)} wireless")
    for iface in rep["interfaces"]:
        if not iface["wireless"]:
            continue
        mm = iface.get("monitor_capable")
        tag = {True: f"{c['g']}monitor OK{c['0']}", False: f"{c['r']}no monitor{c['0']}",
               None: "monitor unknown"}[mm]
        driver = iface.get("driver") or "?"
        lines.append(f"    - {iface['name']:<10} driver={driver:<12} {tag}"
                     f"  operstate={iface.get('operstate')}")
    if not wireless:
        lines.append(f"    {c['r']}(none) — no wireless NIC visible{c['0']}")

    lines.append(f"  Tools      : {', '.join(sorted(rep['tools'].keys())) or 'none'}")
    if rep["tools_missing"]:
        lines.append(f"  {c['d']}Missing    : {', '.join(sorted(rep['tools_missing']))}{c['0']}")

    csi = rep["csi"]
    lines.append(f"  CSI        : {'AVAILABLE' if csi['available'] else 'not available'}"
                 f"{' (live)' if csi['live_csi'] else ''}"
                 f"{' (offline pcap only)' if (csi['offline_csi'] and not csi['live_csi']) else ''}")
    for chk in csi["checks"]:
        mark = f"{c['g']}ok{c['0']}" if chk["available"] else f"{c['d']}--{c['0']}"
        lines.append(f"    [{mark}] {chk['backend']:<14} {c['d']}{chk['detail']}{c['0']}")

    dec = rep["decision"]
    lines.append(f"  {c['b']}Decision   : mode={dec['mode']}  source={dec['source']}  "
                 f"interface={dec['interface']}{c['0']}")
    for reason in dec["reasons"]:
        lines.append(f"    → {reason}")
    if dec["limitations"]:
        lines.append(f"  {c['d']}Limitations:{c['0']}")
        for lim in dec["limitations"]:
            lines.append(f"    {c['d']}· {lim}{c['0']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import json
    print(format_report(capability_report()))
    print(json.dumps(capability_report(), indent=2, default=str))
