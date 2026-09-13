"""The sensing runner: wiring + main loop.

Everything that keeps the process alive lives here, because the difference
between a demo and something you can leave running for a week is entirely in
this file:

* **Collector restart on failure.** A NIC that gets pulled, an AP that reboots,
  NetworkManager reclaiming the interface — each raises :class:`CollectorError`,
  each is logged, and the backend is rebuilt with exponential backoff instead of
  killing the process.
* **Stall watchdog.** Some backends can hang while still "alive" (tshark running
  but silent after the radio is put back to managed mode). A watchdog thread
  notices that no sample has arrived for ``watchdog.stall_s`` and stops the
  collector so the loop can restart it.
* **Calibration as a first-class state**, triggerable at startup or from the
  dashboard at any time, executed inside the sensing loop so baseline windows
  have exactly the same cadence as live windows.
* **Multi-client friendly telemetry** through :class:`StateHub` — the dashboard,
  console and alarm all read snapshots; the run loop is the only writer.
"""

from __future__ import annotations

import math
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alarm.manager import AlarmEvent, AlarmManager
from .alarm.notifier import build_notifiers
from .collectors import Collector, CollectorError, Sample, build_collector
from .collectors import capabilities as caps
from .config import Config
from .detection.baseline import baseline_path, compute_baseline
from .detection.engine import DetectionEngine, DetectionState
from .detection.localize import (
    Zone, load_localizer, room_geometry, zone_model_path, zones_from_config, ZoneLocalizer,
)
from .processing.filters import StreamingFilter
from .processing.features import CSIFeatureExtractor, RSSIFeatureExtractor
from .utils.console import ConsoleRenderer
from .utils.data_logger import DataLogger
from .utils.state import StateHub


class SensingRunner:
    def __init__(self, config: Config, source: str = "auto", mode: str = "auto",
                 interface: Optional[str] = None, replay: Optional[str] = None,
                 csi_capture: Optional[str] = None, csi_tool: str = "nexmon",
                 method: Optional[str] = None, model_path: Optional[str] = None,
                 dashboard: Optional[bool] = None, log: Optional[bool] = None,
                 console: bool = True, retry_forever: bool = True,
                 calibrate: bool = False, calibrate_only: bool = False,
                 calibrate_zones: bool = False, zone_duration: Optional[float] = None,
                 zone_names: Optional[List[str]] = None,
                 arm: Optional[str] = None, duration: Optional[float] = None,
                 points: int = 900, quiet: bool = False):
        self.config = config
        self.requested_source = source
        self.requested_mode = mode
        self.interface = interface
        self.replay = replay
        self.csi_capture = csi_capture
        self.csi_tool = csi_tool
        self.method = method or config.get_path("detection.method", "adaptive")
        self.model_path = model_path or config.get_path("detection.ml_model")
        self.retry_forever = retry_forever
        self.calibrate_on_start = calibrate
        self.calibrate_only = calibrate_only
        self.calibrate_zones_on_start = calibrate_zones
        self.zone_duration = zone_duration
        self.zone_names = list(zone_names or [])
        self.quiet = quiet
        self.arm_on_start = arm
        self.duration = duration
        self.console_enabled = console

        self.hub = StateHub(history_points=points)
        self.stop_event = threading.Event()
        self._watchdog: Optional[threading.Thread] = None
        self._dashboard_handle = None
        self._collector: Optional[Collector] = None
        self._logger: Optional[DataLogger] = None
        self._engine: Optional[DetectionEngine] = None
        self._alarm: Optional[AlarmManager] = None
        self._renderer: Optional[ConsoleRenderer] = None
        self._extractor: Any = None
        self._filter = StreamingFilter(
            outlier_sigma=config.get_path("processing.outlier_gate_sigma", 4.0),
            median_window=config.get_path("processing.median_window", 5),
            smooth_alpha=config.get_path("processing.smooth_alpha", 0.35),
        )
        self._last_sample_t = 0.0
        self._sample_times: deque = deque(maxlen=400)
        self._last_analysis_t = 0.0
        self._last_render_t = 0.0
        self._windows = 0
        self._events = 0
        self._calibration: Optional[Dict[str, Any]] = None
        self._calib_windows: List[Dict[str, float]] = []
        self._calib_started: Optional[float] = None
        self._localizer: Any = None
        self._localization: Optional[Dict[str, Any]] = None
        self._stall_s = float(config.get_path("watchdog.stall_s", 45.0))
        self._restarts = 0
        self._decision: Dict[str, Any] = {}
        self._report: Dict[str, Any] = {}
        self.dashboard_enabled = (config.get_path("dashboard.enabled", True)
                                  if dashboard is None else dashboard)
        self.log_enabled = (config.get_path("logging.enabled", True) if log is None else log)
        self._analysis_interval = max(0.5, float(config.get_path("sampling.window_s", 12.0)) / 12.0)

    # ================================================================== #
    # setup
    # ================================================================== #
    def setup(self) -> None:
        self._report = caps.capability_report(mode=self.requested_mode, source=self.requested_source)
        decision = self._report["decision"]
        self._decision = decision

        if not self.interface:
            self.interface = decision.get("interface")
        source = decision.get("source") or self.requested_source
        self.config.set_path("mode", decision.get("mode"))

        if self.console_enabled and not self.quiet:
            print(caps.format_report(self._report,
                                     color=self.config.get_path("console.color", True)))
            print()

        if source is None:
            raise CollectorError("no usable sensing backend on this machine",
                                 self._report["decision"]["limitations"])

        self._extractor = (CSIFeatureExtractor(
                               window_s=float(self.config.get_path("csi.window_s", 4.0)),
                               min_frames=int(self.config.get_path("csi.min_frames", 12)))
                           if decision.get("mode") == "csi" else
                           RSSIFeatureExtractor(
                               window_s=float(self.config.get_path("sampling.window_s", 12.0)),
                               fs=float(self.config.get_path("sampling.resample_hz", 20.0)),
                               min_samples=int(self.config.get_path("sampling.min_samples", 24)),
                               spectral=bool(self.config.get_path("features.spectral", True)),
                               nperseg=int(self.config.get_path("features.nperseg", 128))))

        self._collector = build_collector(
            self.config, source=source, interface=self.interface, mode=decision.get("mode", "rssi"),
            replay=self.replay, csi_capture=self.csi_capture, csi_tool=self.csi_tool,
        )

        # ---- engine (baseline may be loaded later) -------------------- #
        baseline = None
        from .detection.baseline import load_baseline
        baseline_mode = decision.get("mode", "rssi")
        baseline = load_baseline(self.config, self.interface, baseline_mode)
        model = None
        if self.method == "ml" and self.model_path:
            try:
                from .detection.ml import load_model
                model = load_model(self.model_path)
            except Exception as exc:
                print(f"[WiFi Sense] could not load ML model ({exc}) — using the adaptive engine")
                self.method = "adaptive"
        self._engine = DetectionEngine(
            self.config, baseline=baseline, method=self.method, model=model,
            adaptive_drift=float(self.config.get_path("calibration.adaptive_drift", 0.02)),
        )
        if baseline and self.console_enabled and not self.quiet:
            print("[WiFi Sense] loaded calibration:")
            print("  " + baseline.describe().replace("\n", "\n  "))
            print()

        self._load_localizer()

        # ---- logging -------------------------------------------------- #
        self._logger = DataLogger(
            directory=self.config.get_path("logging.dir", "data"),
            enabled=self.log_enabled,
            fmt=self.config.get_path("logging.format", "csv"),
            flush_every=int(self.config.get_path("logging.flush_every", 1)),
            log_events=bool(self.config.get_path("logging.events", True)),
        )
        if self.log_enabled:
            self._logger.write_meta({
                "capabilities": self._report,
                "decision": decision,
                "config": dict(self.config),
                "baseline": baseline.to_dict() if baseline else None,
                "platform": platform.platform(),
                "python": sys.version,
            })

        # ---- alarm ---------------------------------------------------- #
        notifier = build_notifiers(self.config)
        self._alarm = AlarmManager(self.config, notifier=notifier, on_event=self._on_alarm_event)
        if self.arm_on_start:
            event = self._alarm.arm(self.arm_on_start)
            self._log_alarm(event)
            if self.console_enabled:
                print(f"[WiFi Sense] {event.message}")

        # ---- console -------------------------------------------------- #
        self._renderer = ConsoleRenderer(
            refresh_hz=float(self.config.get_path("console.refresh_hz", 2.0)),
            color=bool(self.config.get_path("console.color", True)),
            show_features=bool(self.config.get_path("console.show_features", True)),
        )
        if self._renderer.palette.enabled:
            pass
        self._publish_meta(warming_up=True)
        if self._renderer:
            self._renderer.set_header(mode=decision.get("mode"), source=source,
                                       interface=self.interface,
                                       synthetic=(source in ("synthetic",)),
                                       warming_up=True, warming_detail="filling the analysis window")

        # ---- dashboard ------------------------------------------------- #
        if self.dashboard_enabled:
            try:
                from .dashboard.server import serve_in_thread
                host = self.config.get_path("dashboard.host", "0.0.0.0")
                port = int(self.config.get_path("dashboard.port", 8000))
                self._dashboard_handle = serve_in_thread(self.hub, self.config, self._logger,
                                                         host=host, port=port)
                self._announce_dashboard(host, port)
            except Exception as exc:
                print(f"[WiFi Sense] dashboard failed to start: {exc}")

        # ---- watchdog -------------------------------------------------- #
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="wifisense-watchdog",
                                          daemon=True)
        self._watchdog.start()

    # ------------------------------------------------------------------ #
    def _load_localizer(self) -> None:
        """Load a trained zone model if present, and always publish the room
        geometry + declared zones so the 3D dashboard can render them."""
        interface = self.interface or "default"
        self._localizer = load_localizer(self.config, interface)
        geometry = room_geometry(self.config)
        zones = zones_from_config(self.config)
        self.hub.publish_meta(room=geometry, zones=[z.__dict__ for z in zones])
        if self._localizer is not None:
            self.hub.publish_meta(localization_notes=self._localizer.host_notes,
                                  localization_zones=sorted(self._localizer.signatures))
            if self.console_enabled and not self.quiet:
                print("[WiFi Sense] loaded zone localization:")
                print("  " + self._localizer.describe().replace("\n", "\n  "))
                print()

    # ------------------------------------------------------------------ #
    def _run_zone_calibration(self) -> None:
        """Walk through every declared zone so the localizer learns its signature.

        For each zone the operator stands still in that part of the room for a
        few seconds; the feature windows collected become that zone's template.
        Requires a working collector (--source ...)."""
        zones = zones_from_config(self.config)
        if self.zone_names:
            zones = [z for z in zones if z.name in self.zone_names]
        if not zones:
            print("[WiFi Sense] zone calibration: no zones declared in config")
            return
        if not self._collector:
            print("[WiFi Sense] zone calibration: collector not ready")
            return

        duration = self.zone_duration or float(
            self.config.get_path("localization.duration_per_zone", 10.0))
        print(f"\n[WiFi Sense] ZONE CALIBRATION: stand still in each zone for {duration:.0f}s.\n")

        zone_rows: Dict[str, List[Dict[str, float]]] = {}
        for zone in zones:
            label = zone.label or zone.name
            print(f"  >>> Move to zone '{zone.name}' ({label}) and STAND STILL — {duration:.0f}s")
            windows = self._collect_zone_windows(duration)
            if len(windows) < 3:
                print(f"      (zone {zone.name}: only {len(windows)} windows — weak signature)")
            zone_rows[zone.name] = windows
            print(f"      collected {len(windows)} windows for '{zone.name}'")

        localizer = ZoneLocalizer(
            zones, min_confidence=float(self.config.get_path("localization.min_confidence", 35)))
        localizer.fit(zone_rows, empty_rows=self._calib_windows or None)
        path = zone_model_path(self.config, self.interface)
        localizer.save(path)
        self._localizer = localizer
        self.hub.publish_meta(localization_notes=localizer.host_notes,
                              localization_zones=sorted(localizer.signatures))
        print(f"\n[WiFi Sense] zone localization saved: {path}")
        print("  " + localizer.describe().replace("\n", "\n  "))
        for note in localizer.host_notes:
            print(f"  note: {note}")
        print()

    def _collect_zone_windows(self, seconds: float) -> List[Dict[str, float]]:
        """Collect feature windows from the live collector for ``seconds``."""
        extractor = RSSIFeatureExtractor(
            window_s=float(self.config.get_path("sampling.window_s", 12.0)),
            fs=float(self.config.get_path("sampling.resample_hz", 20.0)),
            min_samples=int(self.config.get_path("sampling.min_samples", 24)),
            spectral=bool(self.config.get_path("features.spectral", True)),
        )
        filter_ = StreamingFilter(
            outlier_sigma=float(self.config.get_path("processing.outlier_gate_sigma", 4.0)),
            median_window=int(self.config.get_path("processing.median_window", 5)),
        )
        windows: List[Dict[str, float]] = []
        deadline = time.time() + float(seconds)
        last_analysis = 0.0
        for sample in self._collector.stream():
            if self.stop_event.is_set():
                break
            if sample.rssi is not None:
                result = filter_.update(sample.rssi)
                extractor.push(sample.t, result["filtered"])
            if extractor.ready() and (time.time() - last_analysis) >= self._analysis_interval:
                feats = extractor.features()
                if feats:
                    windows.append(feats)
                    last_analysis = time.time()
            if time.time() >= deadline:
                break
        return windows

    # ------------------------------------------------------------------ #
    def _announce_dashboard(self, host: str, port: int) -> None:
        lan = self._lan_ip()
        url_local = f"http://127.0.0.1:{port}/"
        url_lan = f"http://{lan}:{port}/" if lan else None
        lines = [f"[WiFi Sense] dashboard ready:"]
        lines.append(f"  this machine : {url_local}")
        if url_lan:
            lines.append(f"  phones / LAN : {url_lan}")
            if self.config.get_path("dashboard.auth.enabled", False):
                lines.append("  PIN required : open the URL, then enter the view or admin PIN")
        if self.config.get_path("dashboard.show_qr", True) and url_lan:
            qr = _qr_terminal(url_lan)
            if qr:
                lines.append(qr)
            else:
                lines.append("  (install `qrcode` to print a scannable QR: pip install qrcode)")
        print("\n".join(lines) + "\n")

    @staticmethod
    def _lan_ip() -> Optional[str]:
        """Best-effort primary LAN address (no traffic is actually sent)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return None

    # ================================================================== #
    # calibration
    # ================================================================== #
    def start_calibration(self, recalibrate: bool = False) -> None:
        duration = self.duration or float(self.config.get_path("calibration.duration_s", 30.0))
        self._calibration = {"state": "running", "recalibrate": recalibrate,
                             "started_at": time.time(), "duration_s": duration, "progress": 0.0}
        self._calib_windows = []
        self._calib_started = time.time()
        self.hub.publish_calibration(state="running", progress=0.0, duration_s=duration)
        if self.console_enabled:
            print(f"\n[WiFi Sense] CALIBRATION: keep the measuring area EMPTY for {duration:.0f}s\n")
        if self._renderer:
            self._renderer.warn(f"CALIBRATING — room must stay empty for {duration:.0f}s")

    def _calibration_tick(self, features: Dict[str, float]) -> None:
        if self._calibration is None or self._calibration.get("state") != "running":
            return
        duration = float(self._calibration["duration_s"])
        elapsed = time.time() - self._calib_started
        settle = float(self.config.get_path("calibration.settle_s", 3.0))
        if elapsed >= settle:
            self._calib_windows.append(features)
        progress = min(1.0, elapsed / duration)
        self._calibration["progress"] = progress
        self.hub.publish_calibration(state="running", progress=progress,
                                     windows=len(self._calib_windows), duration_s=duration)
        if elapsed < duration:
            return

        # ---- finish ---------------------------------------------------- #
        mode = self._decision.get("mode", "rssi")
        try:
            baseline = compute_baseline(
                self._calib_windows,
                presence_k=float(self.config.get_path("detection.presence_k", 3.0)),
                motion_k=float(self.config.get_path("detection.motion_k", 2.2)),
                mode=mode, source=self._collector.name if self._collector else "unknown",
                interface=self.interface, sample_rate=self._rate_hz(),
                duration_s=duration,
                notes=["recalibrated from the dashboard" if self._calibration.get("recalibrate")
                       else "initial calibration"],
            )
        except ValueError as exc:
            self._calibration = {"state": "failed", "error": str(exc)}
            self.hub.publish_calibration(state="failed", error=str(exc))
            if self._renderer:
                self._renderer.warn(f"calibration failed: {exc}")
            return

        path = baseline_path(self.config, self.interface, mode)
        baseline.save(path)
        self._engine.baseline = baseline
        self._engine.adapter = None
        from .detection.baseline import DriftAdapter
        self._engine.adapter = DriftAdapter(
            baseline, rate=float(self.config.get_path("calibration.adaptive_drift", 0.02)))
        self._engine.reset()
        self._calibration = {"state": "done", "path": str(path),
                             "windows": len(self._calib_windows),
                             "finished_at": time.time()}
        self.hub.publish_calibration(state="done", path=str(path),
                                     windows=len(self._calib_windows), progress=1.0)
        self.hub.publish_meta(baseline_path=str(path),
                              baseline_median=baseline.features.get("mean").median
                              if "mean" in baseline.features else None,
                              notes=baseline.notes)
        if self._renderer:
            self._renderer.clear_warnings()
        if self.console_enabled:
            print(f"[WiFi Sense] calibration saved: {path}")
            print("  " + baseline.describe().replace("\n", "\n  "))
            for note in baseline.notes:
                print(f"  note: {note}")
            print()
        if self.calibrate_only:
            self.stop_event.set()

    # ================================================================== #
    # main loop
    # ================================================================== #
    def run(self) -> int:
        self.setup()
        install_signals(self)
        if self.calibrate_on_start:
            self.start_calibration(recalibrate=False)
        if self.calibrate_zones_on_start:
            self._run_zone_calibration()
        if self.calibrate_only:
            self.shutdown()
            return 0

        backoff = 2.0
        try:
            while not self.stop_event.is_set():
                collector = self._collector
                try:
                    if self.console_enabled and collector is not None:
                        print(f"[WiFi Sense] acquiring from {collector.name}"
                              f"{' on ' + collector.interface if collector.interface else ''} ...\n")
                    for sample in collector.stream():
                        if self.stop_event.is_set():
                            break
                        self._handle_sample(sample)
                    if not self.stop_event.is_set():
                        self._note_collector_end(collector)
                    backoff = 2.0
                except CollectorError as exc:
                    if not self._handle_collector_error(exc, backoff):
                        return 2
                    backoff = min(backoff * 2, 60.0)
                    self._sleep(backoff)
                except Exception as exc:  # unexpected: never lose the process silently
                    if not self.retry_forever:
                        raise
                    self._hub_warning(f"unexpected collector error: {type(exc).__name__}: {exc}")
                    backoff = min(backoff * 2, 60.0)
                    self._sleep(backoff)
                if self.stop_event.is_set():
                    break
                self._rebuild_collector()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    # ------------------------------------------------------------------ #
    def _handle_sample(self, sample: Sample) -> None:
        now = sample.t or time.time()
        self._last_sample_t = time.time()
        self._sample_times.append(self._last_sample_t)

        if sample.meta.get("synthetic"):
            self.hub.publish_meta(synthetic=True)
        result = self._filter.update(sample.rssi) if sample.rssi is not None else None
        shown = (result or {}).get("ema") if result else None
        self.hub.publish_sample(now, shown if shown is not None else sample.rssi,
                                extra={"motion": bool(getattr(self._engine, "_motion", False))} )
        if self._logger:
            self._logger.log_sample(sample)

        if sample.is_csi:
            self._extractor.push(now, sample.amplitude, sample.phase, sample.rssi)
        else:
            if result and result.get("filtered") is not None:
                self._extractor.push(now, result["filtered"])

        if self._extractor.ready() and (time.time() - self._last_analysis_t) >= self._analysis_interval:
            self._analyse()

    def _analyse(self) -> None:
        features = self._extractor.features()
        if not features:
            return
        self._last_analysis_t = time.time()
        self._windows += 1

        if self._calibration and self._calibration.get("state") == "running":
            self._calibration_tick(features)
            return

        state: DetectionState = self._engine.update(features)
        localization = None
        if self._localizer is not None and state.presence:
            localization = self._localizer.predict(features).to_dict()
        self._localization = localization
        baseline_median = None
        if self._engine.baseline and "mean" in self._engine.baseline.features:
            baseline_median = self._engine.baseline.features["mean"].median
        self.hub.publish_state(state, features, extra={
            "baseline_median": baseline_median,
            "localization": localization,
            "room": self.hub.meta.get("room"),
            "zones": self.hub.meta.get("zones"),
        })
        if self._logger:
            self._logger.log_window(features, state)
        self._publish_meta(warming_up=False)

        events = self._alarm.update(state, now=state.t, context={
            "source": self._collector.name if self._collector else "?",
            "interface": self.interface,
        }) if self._alarm else []
        for event in events:
            self._log_alarm(event)

        self.hub.publish_stats(samples=self._filter.accepted + self._filter.rejected,
                               windows=self._windows, rate_hz=self._rate_hz(),
                               window_s=float(self.config.get_path("sampling.window_s", 12.0)),
                               events=self._events, uptime_s=time.time() - (self._start_ts()),
                               filter=self._filter.stats())
        self._render()

    # ------------------------------------------------------------------ #
    def _render(self) -> None:
        if not self._renderer:
            return
        snap = self.hub.snapshot(points=1)
        latest = snap["latest"]
        det = None
        if latest:
            det = DetectionState(
                t=latest.get("t", 0.0), presence=bool(latest.get("presence")),
                motion=bool(latest.get("motion")), confidence=float(latest.get("confidence") or 0.0),
                presence_score=float(latest.get("presence_score") or 0.0),
                motion_score=float(latest.get("motion_score") or 0.0),
                presence_ratio=float(latest.get("presence_ratio") or 0.0),
                motion_ratio=float(latest.get("motion_ratio") or 0.0),
                holding=bool(latest.get("holding")), reason=latest.get("reason", ""),
                calibrated=bool(latest.get("calibrated")), method=latest.get("method", "adaptive"),
            )
        features = latest.get("features") or {}
        last_rssi = None
        if snap["history"]:
            last_rssi = snap["history"][-1].get("rssi")
        stats = dict(snap["stats"])
        if self._calibration and self._calibration.get("state") == "running":
            stats["calibration"] = (f"{self._calibration['progress']*100:.0f}% "
                                    f"({len(self._calib_windows)} windows, room must stay empty)")
        self._renderer.render(det, features, last_rssi, stats=stats,
                              alarm=self._alarm.status() if self._alarm else None,
                              zone=latest.get("localization"))

    # ------------------------------------------------------------------ #
    def _handle_collector_error(self, exc: CollectorError, backoff: float) -> bool:
        self._hub_warning(str(exc))
        if self.console_enabled:
            lines = ["", caps.format_report({}, color=False) if False else "", exc.pretty(), ""]
            print("\n".join(lines))
            if self.retry_forever:
                print(f"[WiFi Sense] retrying in {backoff:.0f}s (backoff). "
                      f"The process stays alive; Ctrl-C to stop.\n")
            else:
                print("[WiFi Sense] not retrying (--no-retry). Exiting.\n")
        if self._renderer:
            self._renderer.warn(str(exc).split("\n")[0])
        if not self.retry_forever:
            return False
        return True

    def _note_collector_end(self, collector: Collector) -> None:
        self._hub_warning(f"collector {collector.name} stopped delivering samples — restarting")
        if self.console_enabled:
            print(f"[WiFi Sense] {collector.name} stream ended; restarting it.\n")
        self._sleep(1.0)

    def _rebuild_collector(self) -> None:
        self._restarts += 1
        self.hub.publish_stats(collector_restarts=self._restarts)
        try:
            self._collector = build_collector(
                self.config, source=self._decision.get("source", self.requested_source),
                interface=self.interface, mode=self._decision.get("mode", "rssi"),
                replay=self.replay, csi_capture=self.csi_capture, csi_tool=self.csi_tool)
        except CollectorError as exc:
            self._hub_warning(f"rebuild failed: {exc}")

    def _watchdog_loop(self) -> None:
        while not self.stop_event.is_set():
            self.stop_event.wait(2.0)
            if self.stop_event.is_set():
                break
            if self._calibration and self._calibration.get("state") == "running":
                continue
            last = self._last_sample_t
            if last and (time.time() - last) > self._stall_s:
                self._hub_warning(f"no samples for {time.time() - last:.0f}s — restarting the backend")
                if self.console_enabled:
                    print(f"\n[WiFi Sense] watchdog: no samples for "
                          f"{time.time() - last:.0f}s — restarting {self._collector.name}\n")
                self._last_sample_t = time.time()
                if self._collector:
                    self._collector.stop()

    # ------------------------------------------------------------------ #
    def _publish_meta(self, **extra: Any) -> None:
        meta = {
            "mode": self._decision.get("mode"),
            "source": self._decision.get("source"),
            "interface": self.interface,
            "reasons": self._decision.get("reasons"),
            "limitations": self._decision.get("limitations"),
            "csi_available": self._report.get("csi", {}).get("available"),
            "os": (self._report.get("os") or {}).get("system"),
            "method": self.method,
            "collector_restarts": self._restarts,
        }
        meta.update(extra)
        self.hub.publish_meta(**meta)

    def _hub_warning(self, message: str) -> None:
        self.hub.add_warning(message)
        if self._logger:
            self._logger.log_event({"t": time.time(), "kind": "warning", "severity": "notice",
                                    "message": message})

    def _rate_hz(self) -> float:
        if len(self._sample_times) < 3:
            return 0.0
        span = self._sample_times[-1] - self._sample_times[0]
        if span <= 0:
            return 0.0
        return (len(self._sample_times) - 1) / span

    def _start_ts(self) -> float:
        return self.hub.started_at

    def _sleep(self, seconds: float) -> None:
        """Leftover-control-aware sleep: calibration can be requested while waiting."""
        end = time.time() + seconds
        while time.time() < end and not self.stop_event.is_set():
            self._drain_control()
            self.stop_event.wait(0.5)

    def _drain_control(self) -> None:
        control = self.hub.consume_control()
        if control.get("calibration_requested"):
            self.start_calibration(recalibrate=False)
        if control.get("recalibrate_requested"):
            self.start_calibration(recalibrate=True)
        if control.get("arm_request") and self._alarm:
            self._log_alarm(self._alarm.arm(control["arm_request"]))
        if control.get("disarm_request") and self._alarm:
            self._log_alarm(self._alarm.disarm())

    def _on_alarm_event(self, event: AlarmEvent) -> None:
        if event.kind in ("triggered", "cleared"):
            self.hub.add_event(event.to_dict())

    def _log_alarm(self, event: AlarmEvent) -> None:
        self._events += 1
        if self._logger:
            self._logger.log_event(event.to_row())
        self.hub.add_event(event.to_dict())
        if self._alarm:
            self.hub.publish_alarm(self._alarm.status())

    # ================================================================== #
    def shutdown(self) -> None:
        self.stop_event.set()
        try:
            if self._collector:
                self._collector.stop()
        except Exception:
            pass
        if self._dashboard_handle:
            try:
                self._dashboard_handle.stop()
            except Exception:
                pass
        summary: Dict[str, Any] = {}
        if self._logger:
            summary = self._logger.close()
            if self.log_enabled:
                path = self._logger.write_meta({
                    "summary": summary,
                    "engine": self._engine.summary() if self._engine else None,
                    "alarm": self._alarm.status() if self._alarm else None,
                    "collector": self._collector.describe() if self._collector else None,
                    "stopped_at": time.time(),
                })
                summary["meta_file"] = str(path)
        if self.console_enabled:
            print("\n[WiFi Sense] stopped.")
            if summary.get("files"):
                for kind, path in summary["files"].items():
                    print(f"  {kind:<8} {path}")
            print(f"  windows analysed: {self._windows}   events: {self._events}   "
                  f"collector restarts: {self._restarts}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def install_signals(runner: SensingRunner) -> None:
    def handler(signum, frame):  # noqa: ARG001
        runner.stop_event.set()
        try:
            if runner._collector:
                runner._collector.stop()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # non-main thread
            pass


def _qr_terminal(text: str) -> Optional[str]:
    try:
        import qrcode  # type: ignore
    except Exception:
        return None
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(text)
        qr.make()
        matrix = qr.get_matrix()
        lines = []
        for row in matrix:
            lines.append("".join("██" if cell else "  " for cell in row))
        return "\n".join("  " + line for line in lines)
    except Exception:
        return None
