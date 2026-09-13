"""Pure parsing helpers.

Every function here takes **text or bytes** and returns plain Python data. That
split exists so the parsing logic is unit-testable on machines with no radio at
all (CI, containers, cloud VMs) and so a malformed OS tool output can never
crash the sensing loop — parsers return ``None`` / empty lists instead.

Nothing in this module shells out; see ``capabilities.py`` and the individual
collector modules for that.
"""

from __future__ import annotations

import re
import struct
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #

_FLOAT_RE = r"[+-]?\d+(?:\.\d+)?"
_IW_SIGNAL_RE = re.compile(rf"signal:\s*({_FLOAT_RE})\s*dBm")
_IW_NOISE_RE = re.compile(rf"noise:\s*({_FLOAT_RE})\s*dBm")


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct_to_dbm(percent: float) -> float:
    """Windows / nmcli style 0-100 % -> dBm approximation.

    Windows reports ``Signal : 78%``. The de-facto mapping used by every
    Windows WiFi tool is ``dBm = (percent / 2) - 100``.
    """
    return (float(percent) / 2.0) - 100.0


def dbm_to_pct(dbm: float) -> float:
    pct = (float(dbm) + 100.0) * 2.0
    return max(0.0, min(100.0, pct))


# --------------------------------------------------------------------------- #
# iw
# --------------------------------------------------------------------------- #

def parse_iw_dev(text: str) -> List[Dict[str, Any]]:
    """``iw dev`` output -> list of ``{"phy", "iface", "type", "addr"}``."""
    out: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("phy#"):
            if current.get("iface"):
                out.append(current)
            current = {"phy": line[3:].strip()}
        elif line.startswith("Interface "):
            if current.get("iface"):
                out.append(current)
            current = dict(current, iface=line.split(" ", 1)[1].strip())
        elif line.startswith("type "):
            current["type"] = line.split(" ", 1)[1].strip()
        elif line.startswith("addr "):
            current["addr"] = line.split(" ", 1)[1].strip()
        elif line.startswith("channel "):
            parts = line.split()
            if len(parts) > 1:
                current["channel"] = _to_float(parts[1])
            if "(" in line and "MHz" in line:
                m = re.search(r"\((\d+)\s*MHz\)", line)
                if m:
                    current["freq_mhz"] = int(m.group(1))
    if current.get("iface"):
        out.append(current)
    return out


def parse_iw_phy_modes(text: str) -> List[str]:
    """``iw phy <phy> info`` -> list of supported interface modes (lowercase)."""
    modes: List[str] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Supported interface modes:"):
            in_block = True
            continue
        if in_block:
            if line.startswith("* "):
                modes.append(line[2:].strip().lower())
            elif line and not line.startswith("*"):
                break
    return modes


def parse_iw_phy_bands(text: str) -> Dict[str, Any]:
    """Extract a coarse capability summary from ``iw phy info``."""
    info: Dict[str, Any] = {
        "bands": [],
        "ht": "HT20" in text or "HT Capabilities" in text,
        "vht": "VHT Capabilities" in text,
        "he": "HE Iftypes" in text or "HE MAC Capabilities" in text,
        "channels": len(re.findall(r"^\s*\*\s*\d{4}\s*MHz", text, re.MULTILINE)),
        "monitor": "monitor" in text.lower(),
        "n_antennas": None,
    }
    for band in ("2412 MHz", "5180 MHz", "5955 MHz"):
        if band in text:
            info["bands"].append(band.split()[0])
    m = re.search(r"RX chains?:\s*(\d+)", text)
    if m:
        info["n_antennas"] = int(m.group(1))
    return info


def parse_iw_link(text: str) -> Dict[str, Any]:
    """``iw dev <iface> link`` -> link status dict.

    ``connected`` False when the NIC is not associated (monitor mode, AP mode,
    or simply unassociated) — in that case there is no per-link RSSI at all,
    which is exactly the situation where the radiotap sniffer must be used.
    """
    result: Dict[str, Any] = {"connected": False, "signal_dbm": None}
    stripped = text.strip()
    if not stripped or "Not connected" in stripped:
        return result
    result["connected"] = True
    for raw in stripped.splitlines():
        line = raw.strip()
        if line.startswith("Connected to "):
            result["bssid"] = line.split(" ", 3)[2]
        elif line.startswith("SSID:"):
            result["ssid"] = line.split(":", 1)[1].strip()
        elif line.startswith("freq:"):
            result["freq_mhz"] = _to_float(line.split(":", 1)[1])
        elif line.startswith("signal:"):
            result["signal_dbm"] = _to_float(line.split(":", 1)[1].replace("dBm", "").strip())
        elif line.startswith("tx bitrate:"):
            result["tx_bitrate"] = line.split(":", 1)[1].strip()
        elif line.startswith("rx bitrate:"):
            result["rx_bitrate"] = line.split(":", 1)[1].strip()
        elif "DTIM period" in line:
            result["dtim"] = _to_float(line.split(":", 1)[1])
    return result


def parse_iw_station_dump(text: str) -> List[Dict[str, Any]]:
    """``iw dev <iface> station dump`` -> per-station RSSI list.

    Only meaningful when the interface is acting as an AP (or a mesh peer), but
    it is a rich source: every associated phone/laptop is an independent RSSI
    probe of the room.
    """
    stations: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Station "):
            if current:
                stations.append(current)
            current = {"mac": line.split(" ", 1)[1].split(" ")[0]}
        elif current is not None:
            if line.startswith("signal:"):
                m = _IW_SIGNAL_RE.search(line) or re.search(rf"signal:\s*({_FLOAT_RE})", line)
                if m:
                    current["signal_dbm"] = _to_float(m.group(1))
            elif line.startswith("signal avg:"):
                m = re.search(rf"({_FLOAT_RE})\s*dBm", line)
                if m:
                    current["signal_avg_dbm"] = _to_float(m.group(1))
            elif line.startswith("rx bitrate:"):
                current["rx_bitrate"] = line.split(":", 1)[1].strip()
            elif line.startswith("tx bitrate:"):
                current["tx_bitrate"] = line.split(":", 1)[1].strip()
            elif line.startswith("inactive time:"):
                current["inactive_ms"] = _to_float(line.split(":", 1)[1].replace("ms", "").strip())
            elif line.startswith("connected time:"):
                current["connected_s"] = _to_float(line.split(":", 1)[1])
    if current:
        stations.append(current)
    return stations


# --------------------------------------------------------------------------- #
# /proc/net/wireless
# --------------------------------------------------------------------------- #

def parse_proc_wireless(text: str) -> List[Dict[str, Any]]:
    """``/proc/net/wireless`` -> per-interface level/noise/link.

    Formula (documented in the kernel): ``dBm = level - 256`` when level > 127
    (the kernel hands back an unsigned byte for legacy drivers), otherwise the
    value is already dBm. Some modern drivers report only a "quality" value and
    put ``0`` (or a bogus constant) in ``level``; that is detected and flagged.
    """
    rows: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        iface, _, rest = raw.partition(":")
        iface = iface.strip()
        if iface in ("Inter-| sta", "Face") or not iface:
            continue
        values: List[Optional[float]] = []
        for token in rest.split():
            token = token.rstrip(".")
            values.append(_to_float(token))
        if len(values) < 4:
            continue
        status, link, level, noise = values[0], values[1], values[2], values[3]
        level_dbm = None
        level_reliable = False
        if level is not None and level != 0:
            level_dbm = level - 256.0 if level > 127 else level
            # Sanity: real RSSI never sits outside this window. Drivers that
            # report a fake constant land here and get flagged.
            level_reliable = -110.0 < level_dbm < -5.0
        noise_dbm = None
        if noise is not None and noise != 0:
            noise_dbm = noise - 256.0 if noise > 127 else noise
        rows.append({
            "iface": iface,
            "status": status,
            "link_quality": link,
            "level_dbm": level_dbm,
            "noise_dbm": noise_dbm,
            "level_reliable": level_reliable,
        })
    return rows


# --------------------------------------------------------------------------- #
# tshark / tcpdump (monitor-mode radiotap)
# --------------------------------------------------------------------------- #

def parse_tshark_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one ``-T fields -e frame.time_epoch -e wlan_radio.signal_dbm ...`` row.

    tshark separates fields with TAB. Missing fields arrive as empty strings.
    Multi-antenna NICs emit comma separated values (``-52,-54``); the strongest
    antenna is kept.
    """
    parts = [p.strip() for p in line.rstrip("\n").split("\t")]
    if not parts or not parts[0]:
        return None
    ts = _to_float(parts[0])
    if ts is None:
        return None
    rssi = None
    if len(parts) > 1 and parts[1]:
        candidates = [_to_float(v) for v in parts[1].replace(" ", "").split(",")]
        candidates = [c for c in candidates if c is not None]
        if candidates:
            rssi = max(candidates)
    rec: Dict[str, Any] = {"t": ts, "rssi": rssi}
    if len(parts) > 2 and parts[2]:
        rec["bssid"] = parts[2].lower()
    if len(parts) > 3 and parts[3]:
        rec["freq_mhz"] = _to_float(parts[3])
    if len(parts) > 4 and parts[4]:
        rec["length"] = _to_float(parts[4])
    return rec


_TCPDUMP_SIGNAL_RE = re.compile(rf"({_FLOAT_RE})\s*dBm signal")


def parse_tcpdump_line(line: str) -> Optional[Dict[str, Any]]:
    """tcpdump is only a fallback: it prints radiotap RSSI in human prose.

    Only call it with ``-y IEEE802_11_RADIO -v``; the DBm value otherwise never
    appears in the output at all.
    """
    m = _TCPDUMP_SIGNAL_RE.search(line)
    if not m:
        return None
    rec: Dict[str, Any] = {"rssi": _to_float(m.group(1))}
    ts = re.search(r"^(\d{2}:\d{2}:\d{2}\.\d+)", line)
    if ts:
        rec["time_str"] = ts.group(1)
    bssid = re.search(r"BSSID:([0-9a-fA-F:]{17})", line)
    if bssid:
        rec["bssid"] = bssid.group(1).lower()
    return rec


# --------------------------------------------------------------------------- #
# Windows (netsh) / macOS (airport) — best effort backends
# --------------------------------------------------------------------------- #

def parse_termux_wifi(text: str) -> Dict[str, Any]:
    """``termux-wifi-connectioninfo`` JSON -> link fields.

    Termux (Android) is the one phone platform where this project can actually
    *sense*: the termux-api helper asks the Android framework for the current
    WiFi connection, which includes a live RSSI in dBm. No root needed, no
    monitor mode, one sample per call (~1 Hz).
    """
    import json

    try:
        data = json.loads(text)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    if data.get("rssi") is not None:
        out["signal_dbm"] = _to_float(data.get("rssi"))
    if data.get("ssid"):
        out["ssid"] = str(data.get("ssid"))
    if data.get("bssid"):
        out["bssid"] = str(data.get("bssid")).lower()
    freq = _to_float(data.get("frequency"))
    if freq:
        out["freq_mhz"] = freq
        out["channel"] = freq_to_channel(freq)
    if data.get("link_speed_mbps") is not None:
        out["link_speed_mbps"] = _to_float(data.get("link_speed_mbps"))
    out["supplicant_state"] = str(data.get("supplicant_state") or "")
    out["connected"] = bool(out.get("signal_dbm") is not None
                            and out["supplicant_state"].upper().startswith("COMPLETED"))
    return out


def freq_to_channel(freq_mhz: float) -> Optional[float]:
    """Map a centre frequency to its 802.11 channel number."""
    f = float(freq_mhz)
    if f == 2484:
        return 14.0
    if 2412 <= f <= 2472:
        return float((f - 2407) // 5)
    if 5000 <= f <= 5895:
        return float((f - 5000) // 5)
    if 5955 <= f <= 7115:
        return float((f - 5950) // 5)
    return None

def parse_netsh_interfaces(text: str) -> List[Dict[str, Any]]:
    """``netsh wlan show interfaces`` -> SSID / BSSID / signal% / channel."""
    out: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    tr = str.maketrans("", "", ":\u200b\u00a0")
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip().translate(tr).lower()
        value = value.strip()
        if not value:
            continue
        if key == "name":
            if current:
                out.append(current)
            current = {"iface": value}
            continue
        if current is None:
            continue
        if key == "ssid":
            current["ssid"] = value
        elif key == "bssid":
            current["bssid"] = value.lower()
        elif key == "signal":
            pct = _to_float(value.replace("%", ""))
            if pct is not None:
                current["signal_pct"] = pct
                current["signal_dbm"] = pct_to_dbm(pct)
        elif key == "channel":
            current["channel"] = _to_float(value)
        elif key == "radio type":
            current["radio_type"] = value
        elif key == "state":
            current["state"] = value
    if current:
        out.append(current)
    return out


_AIRPORT_RSSI_RE = re.compile(rf"agrCtlRSSI:\s*({_FLOAT_RE})")
_AIRPORT_NOISE_RE = re.compile(rf"agrCtlNoise:\s*({_FLOAT_RE})")


def parse_airport(text: str) -> Dict[str, Any]:
    """macOS ``airport -I`` -> RSSI/noise/channel/state (deprecated but real)."""
    info: Dict[str, Any] = {"connected": False}
    for raw in text.splitlines():
        key, _, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        if key == "agrCtlRSSI":
            m = _AIRPORT_RSSI_RE.search(raw)
            if m:
                info["signal_dbm"] = _to_float(m.group(1))
        elif key == "agrCtlNoise":
            m = _AIRPORT_NOISE_RE.search(raw)
            if m:
                info["noise_dbm"] = _to_float(m.group(1))
        elif key == "state":
            info["state"] = value
            info["connected"] = value.lower().startswith("running")
        elif key == "channel":
            info["channel"] = _to_float(value.split(",")[0])
        elif key == "lastTxRate":
            info["tx_rate"] = value
        elif key == "SSID":
            info["ssid"] = value
    if info.get("signal_dbm") is not None:
        info["connected"] = True
    return info


# --------------------------------------------------------------------------- #
# nexmon_csi UDP packet
# --------------------------------------------------------------------------- #

NEXMON_MAGIC = 0x1111


def parse_nexmon_packet(data: bytes) -> Optional[Dict[str, Any]]:
    """Decode a nexmon_csi UDP payload.

    Layout as emitted by the nexmon_csi firmware (little endian)::

        uint16 magic (0x1111) | uint8 rssi | uint8 fc | uint8 src_mac[6]
        uint16 seq | uint16 core | uint16 chanspec | uint16 chip_version
        uint16 csi_len (bytes) | int16 csi[csi_len/2]  (imag, real pairs)

    The 20-byte header is the canonical one, but builds exist (and `csiread`
    documents an 18-byte variant) that drop ``src_mac``. Both are tried, and the
    variant that yields a self-consistent ``csi_len`` wins. If the layout of
    your firmware differs, use ``--csi-tool`` with a pcap + ``csiread`` instead,
    or point ``CSI_NEXMON_HEADER`` at the right size.
    """
    if len(data) < 18 or data[:2] != b"\x11\x11":
        return None

    for fmt, header_len in (( "<HBB6sHHHHH", 20), ("<HBBHHHHH", 18)):
        try:
            fields = struct.unpack(fmt, data[:header_len])
        except struct.error:
            continue
        magic, rssi_u, fc, mac, seq, core, chanspec, chip, csi_len = fields
        if magic != NEXMON_MAGIC:
            continue
        if csi_len <= 0 or csi_len % 4 != 0 or csi_len > 4096:
            continue
        payload = data[header_len:header_len + csi_len]
        if len(payload) < csi_len:
            continue
        pairs = struct.unpack(f"<{csi_len // 2}h", payload)
        real = pairs[1::2]
        imag = pairs[0::2]
        rssi = rssi_u - 256 if rssi_u > 127 else rssi_u
        src_mac = ":".join(f"{b:02x}" for b in mac) if isinstance(mac, bytes) else None
        return {
            "rssi": float(rssi),
            "fc": fc,
            "src_mac": src_mac,
            "seq": seq,
            "core": core,
            "chanspec": chanspec,
            "chip_version": chip,
            "csi_len": csi_len,
            "n_subcarriers": csi_len // 4,
            "imag": list(imag),
            "real": list(real),
            "header_len": header_len,
        }
    return None


def nexmon_csi_to_amplitude_phase(pkt: Dict[str, Any]) -> Dict[str, List[float]]:
    """int16 (imag, real) pairs -> per-subcarrier amplitude and unwrapped phase."""
    import math

    real = pkt.get("real") or []
    imag = pkt.get("imag") or []
    amp: List[float] = []
    phase: List[float] = []
    for r, i in zip(real, imag):
        amp.append(math.hypot(r, i))
        phase.append(math.atan2(i, r))
    return {"amplitude": amp, "phase": phase}


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #

def parse_scan_results(text: str) -> List[Dict[str, Any]]:
    """``iw dev <iface> scan`` (or ``iwlist scan``) -> BSS list with RSSI.

    An active scan is heavy (~1 s of airtime, drops the link briefly), so the
    collector only uses it once at startup to pick a transmitter BSSID.
    """
    bss: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("BSS "):
            if current:
                bss.append(current)
            mac = re.search(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})", line)
            current = {"bssid": mac.group(1).lower() if mac else None}
        elif current is not None:
            if line.startswith("signal:"):
                m = re.search(rf"({_FLOAT_RE})\s*dBm", line)
                if m:
                    current["signal_dbm"] = _to_float(m.group(1))
            elif line.startswith("freq:"):
                m = re.search(r"([\d.]+)", line)
                if m:
                    current["freq_mhz"] = _to_float(m.group(1))
            elif line.startswith("SSID:"):
                current["ssid"] = line.split(":", 1)[1].strip()
            elif "DS Parameter set: channel" in line:
                m = re.search(r"channel\s+(\d+)", line)
                if m:
                    current["channel"] = _to_float(m.group(1))
    if current:
        bss.append(current)
    return bss
