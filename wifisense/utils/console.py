"""Terminal rendering.

The block the user asked for, verbatim::

    [WiFi Sense]
    Signal: -48.2 dBm
    Variance: 0.183
    Motion: DETECTED
    Presence: HUMAN DETECTED
    Confidence: 87%

On a TTY the block is redrawn in place (cursor up, no full-screen clears, no
ncurses); piped to a file or a log it degrades to plain append-only lines. Icons
are text glyphs, not emoji, so the output stays readable in any terminal font and
inside `less`/`journalctl`.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

from ..detection.engine import DetectionState

RESET = "\033[0m"


class Palette:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and sys.stdout.isatty())

    def wrap(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.enabled else text

    def bold(self, text: str) -> str:
        return self.wrap("\033[1m", text)

    def dim(self, text: str) -> str:
        return self.wrap("\033[2m", text)

    def green(self, text: str) -> str:
        return self.wrap("\033[32m", text)

    def yellow(self, text: str) -> str:
        return self.wrap("\033[33m", text)

    def red(self, text: str) -> str:
        return self.wrap("\033[31m", text)

    def on_red(self, text: str) -> str:
        return self.wrap("\033[41;97;1m", text)

    def cyan(self, text: str) -> str:
        return self.wrap("\033[36m", text)


class ConsoleRenderer:
    """Prints the live sensing block at a fixed refresh rate."""

    def __init__(self, refresh_hz: float = 2.0, color: bool = True,
                 show_features: bool = True, stream=None):
        self.refresh_hz = max(0.2, float(refresh_hz))
        self.interval = 1.0 / self.refresh_hz
        self.stream = stream or sys.stdout
        self.palette = Palette(color and self.stream.isatty())
        self.show_features = show_features
        self._last_print = 0.0
        self._block_lines = 0
        self.header_info: Dict[str, Any] = {}
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ #
    def set_header(self, **info: Any) -> None:
        self.header_info.update(info)

    def warn(self, message: str) -> None:
        """Persistent warning shown under the block (repeat calls are deduped)."""
        if message and message not in self.warnings:
            self.warnings.append(message)
            if len(self.warnings) > 4:
                del self.warnings[:-4]

    def clear_warnings(self) -> None:
        self.warnings.clear()

    # ------------------------------------------------------------------ #
    def render(self, det: Optional[DetectionState], features: Optional[Dict[str, float]],
               rssi: Optional[float], stats: Optional[Dict[str, Any]] = None,
               alarm: Optional[Dict[str, Any]] = None, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_print) < self.interval:
            return
        self._last_print = now
        lines = self._compose(det, features, rssi, stats or {}, alarm or {})
        self._write_block(lines)

    def _compose(self, det, features, rssi, stats, alarm) -> List[str]:
        p = self.palette
        head = self.header_info
        stamp = time.strftime("%H:%M:%S")
        mode = head.get("mode", "?")
        source = head.get("source", "?")
        iface = head.get("interface") or "-"

        lines = [p.bold("[WiFi Sense]")
                 + p.dim(f"  {stamp}  mode={mode}  source={source}  iface={iface}")]
        if head.get("synthetic"):
            lines.append(p.on_red(" DEMO DATA — synthetic source, not radio measurements "))

        if head.get("warming_up"):
            lines.append(p.yellow(f"warming up: {head.get('warming_detail', '')}"))

        signal = "--" if rssi is None else f"{rssi:.1f} dBm"
        baseline = head.get("baseline_median")
        if baseline is not None and rssi is not None:
            delta = rssi - baseline
            lines.append(f"Signal: {signal}   {p.dim(f'(baseline {baseline:.1f}, {delta:+.2f} dBm)')}")
        else:
            lines.append(f"Signal: {signal}")

        var = (features or {}).get("var")
        lines.append(f"Variance: {var:.3f}" if var is not None else "Variance: --")

        if det is None:
            lines.append("Motion: --")
            lines.append("Presence: --")
            lines.append("Confidence: --")
        else:
            motion = p.red("DETECTED") if det.motion else p.dim("NONE")
            presence = p.on_red("HUMAN DETECTED") if det.presence else p.green("NO HUMAN")
            conf_color = p.red if det.confidence >= 70 else (p.yellow if det.confidence >= 40 else p.dim)
            lines.append(f"Motion: {motion}")
            lines.append(f"Presence: {presence}")
            lines.append(f"Confidence: {conf_color(f'{det.confidence:.0f}%')}"
                         + p.dim(f"   {det.presence_label if not det.presence else ''}"))
            if det.holding:
                lines.append(p.dim("  (presence held — decaying after last event)"))
            lines.append(p.dim(f"  reason: {det.reason}"))

        if self.show_features and features:
            keys = ("diff_rms", "motion_energy", "band_motion", "band_ratio",
                    "spectral_entropy", "perm_entropy", "amp_var_mean", "csi_motion_index")
            parts = [f"{k}={features[k]:.3g}" for k in keys if k in features]
            if parts:
                lines.append(p.dim("  features: " + "  ".join(parts)))

        if alarm:
            state = alarm.get("state", "DISARMED")
            color = p.red if state.startswith("ARMED") and state != "ARMED_HOME" else p.yellow
            if state == "ARMED_HOME":
                color = p.yellow
            armed_txt = color(state)
            extra = f"triggers={alarm.get('trigger_count', 0)}"
            if state == "ARMING":
                extra += f"  exit in {alarm.get('exit_delay_remaining', 0):.0f}s"
            if alarm.get("last_trigger"):
                extra += f"  last={time.strftime('%H:%M:%S', time.localtime(alarm['last_trigger']))}"
            lines.append(f"Alarm: {armed_txt}   {p.dim(extra)}")

        if stats:
            samples = stats.get("samples", 0)
            rate = stats.get("rate_hz", 0.0)
            uptime = _fmt_duration(stats.get("uptime_s", 0.0))
            events = stats.get("events", 0)
            lines.append(p.dim(f"samples={samples}  {rate:.1f} Hz  windows={stats.get('windows', 0)}"
                               f"  events={events}  uptime={uptime}"))
            if stats.get("calibration"):
                lines.append(p.cyan(f"calibration: {stats['calibration']}"))

        for warning in self.warnings:
            lines.append(p.yellow(f"[!] {warning}"))
        return lines

    def _write_block(self, lines: List[str]) -> None:
        stream = self.stream
        if stream.isatty() and self._block_lines:
            stream.write(f"\033[{self._block_lines}A")
        for line in lines:
            stream.write("\033[K" + line + "\n")
        self._block_lines = len(lines)
        stream.flush()


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def banner(text: str, color: bool = True) -> str:
    p = Palette(color)
    return p.bold(text)
