#!/usr/bin/env python3
"""WiFi Human Presence Sensing — command-line entry point.

Commands (see README for the full guide)::

    python main.py                 # auto: best backend for this machine
    python main.py --mode rssi     # force RSSI
    python main.py --mode csi      # force CSI (needs CSI hardware, see README)
    python main.py --calibrate     # capture the empty-room baseline, then keep sensing
    python main.py --calibrate-only
    python main.py --dashboard     # dashboard explicitly (it is on by default)
    python main.py --capabilities  # print the hardware report and exit
    python main.py --source synthetic   # DEMO data only — not a real measurement
    python main.py --replay data/windows_xxx.csv
    python main.py --train --empty e1.csv e2.csv --human h1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wifisense import __version__
from wifisense.collectors import SUPPORTED_SOURCES
from wifisense.config import apply_cli_overrides, dump_config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="WiFi Human Presence Detection — RSSI/CSI, no ESP32/ESP8266 required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    mode = parser.add_argument_group("mode")
    mode.add_argument("--mode", choices=("auto", "rssi", "csi"), default="auto",
                      help="sensing mode (default: auto = use CSI if the hardware has it)")
    mode.add_argument("--source", choices=SUPPORTED_SOURCES, default="auto",
                      help="specific backend instead of auto-selection")
    mode.add_argument("--interface", default=None,
                      help="wireless interface name (e.g. wlan0). null = auto-detect")

    mode.add_argument("--replay", default=None,
                      help="replay a recorded windows/samples CSV instead of live sensing")
    mode.add_argument("--replay-speed", type=float, default=None,
                      help="replay speed (1.0 = real time, 0 = as fast as possible)")
    mode.add_argument("--csi-pcap", default=None,
                      help="offline CSI capture file (.pcap / Intel 5300 .dat) — needs csiread")
    mode.add_argument("--csi-tool", choices=("nexmon", "iwl5300", "atheros", "pcap", "file"),
                      default="nexmon", help="CSI toolchain for --csi-pcap")
    mode.add_argument("--router-host", default=None,
                      help="router IP for --source router (overrides config)")
    mode.add_argument("--router-user", default=None, help="router shell user")
    mode.add_argument("--router-pass", default=None, help="router shell password")
    mode.add_argument("--router-protocol", choices=("telnet", "ssh"), default=None,
                      help="how to reach the router shell")
    mode.add_argument("--router-iface", default=None,
                      help="router wireless interface to poll (wlan0 / wlan1 / ath0 ...)")

    actions = parser.add_argument_group("actions")
    actions.add_argument("--calibrate", action="store_true",
                         help="capture the empty-room baseline, then keep sensing")
    actions.add_argument("--calibrate-only", action="store_true",
                         help="capture the baseline and exit")
    actions.add_argument("--discover", action="store_true",
                         help="scan this machine (NICs, tools, CSI) AND the LAN gateway "
                              "(router shell) and report the best way to sense")
    actions.add_argument("--auto-setup", action="store_true",
                         help="with --discover: write the winner to config/local.yaml")
    actions.add_argument("--probe-host", default=None,
                         help="router to probe during --discover (default: the gateway)")
    actions.add_argument("--no-cred-probe", action="store_true",
                         help="--discover: do not try documented default router logins")
    actions.add_argument("--calibrate-zones", action="store_true",
                         help="walk through each declared zone so the 3D view can place people")
    actions.add_argument("--zone", action="append", default=[], dest="zone_names",
                         help="limit --calibrate-zones to this zone (repeatable)")
    actions.add_argument("--zone-duration", type=float, default=None,
                         help="seconds to stand still per zone (default 10)")
    actions.add_argument("--capabilities", action="store_true",
                         help="print the hardware capability report and exit")
    actions.add_argument("--list-sources", action="store_true",
                         help="list the supported backends and exit")
    actions.add_argument("--dashboard", action="store_true",
                         help="start the web dashboard (on by default; this flag is explicit)")
    actions.add_argument("--arm", choices=("away", "home"), default=None,
                         help="arm the intrusion alarm on startup (away = any presence, "
                              "home = motion only)")
    actions.add_argument("--train", action="store_true",
                         help="train the optional ML classifier (see --empty/--human)")
    actions.add_argument("--empty", nargs="+", default=[],
                         help="windows CSV(s) recorded with the room EMPTY (for --train)")
    actions.add_argument("--human", nargs="+", default=[],
                         help="windows CSV(s) recorded with a person present (for --train)")
    actions.add_argument("--motion", nargs="*", default=[],
                         help="windows CSV(s) recorded with a person MOVING (optional, for --train)")
    actions.add_argument("--method", choices=("adaptive", "ml"), default=None,
                         help="detection method (default: adaptive; ml needs --model)")

    tune = parser.add_argument_group("tuning")
    tune.add_argument("--duration", type=float, default=None,
                      help="calibration duration in seconds (default 30)")
    tune.add_argument("--window", type=float, default=None, dest="window_s",
                      help="analysis window in seconds (default 12)")
    tune.add_argument("--interval", type=float, default=None, dest="interval_s",
                      help="poll interval for polling backends")
    tune.add_argument("--presence-k", type=float, default=None,
                      help="presence threshold in robust sigmas (default 3.0)")
    tune.add_argument("--motion-k", type=float, default=None,
                      help="motion threshold in robust sigmas (default 2.2)")
    tune.add_argument("--model", default=None, dest="model_path",
                      help="path to a joblib model for --method ml")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--config", default=None, help="path to a YAML config overlay")
    runtime.add_argument("--no-dashboard", action="store_true")
    runtime.add_argument("--no-log", action="store_true")
    runtime.add_argument("--log-dir", default=None, help="where session logs are written")
    runtime.add_argument("--host", default=None, help="dashboard bind host (default 0.0.0.0)")
    runtime.add_argument("--port", type=int, default=None, help="dashboard port (default 8000)")
    runtime.add_argument("--points", type=int, default=None,
                         help="history points kept for the live graph")
    runtime.add_argument("--no-color", action="store_true")
    runtime.add_argument("--no-console", action="store_true")
    runtime.add_argument("--no-retry", action="store_true",
                         help="exit on collector errors instead of restarting with backoff")
    runtime.add_argument("--strict", action="store_true",
                         help="exit instead of falling back when the requested mode is missing")
    runtime.add_argument("--quiet", action="store_true", help="suppress the capability report")
    runtime.add_argument("--print-config", action="store_true",
                         help="print the effective configuration and exit")
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    if not args.empty or not args.human:
        print("--train needs --empty <csv...> and --human <csv...> (windows logs)")
        return 2
    from wifisense.detection.ml import main as ml_main
    argv = ["--empty", *args.empty, "--human", *args.human]
    if args.motion:
        argv += ["--motion", *args.motion]
    if args.model_path:
        argv += ["--out", args.model_path]
    return ml_main(argv)


def cmd_capabilities(args: argparse.Namespace, cfg) -> int:
    from wifisense.collectors import capabilities as caps
    rep = caps.capability_report(mode=args.mode, source=args.source)
    print(caps.format_report(rep, color=not args.no_color))
    if args.print_config:
        print("\n" + dump_config(cfg))
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    from wifisense import discovery

    result = discovery.discover(host=args.probe_host,
                                try_credentials=not args.no_cred_probe)
    print(discovery.format_discovery(result, color=not args.no_color))
    if args.auto_setup:
        path = discovery.auto_setup(cfg_for_discover(args), result)
        print(f"\n[WiFi Sense] auto-setup written: {path}")
        print("  re-run:  python main.py")
    return 0


def cfg_for_discover(args: argparse.Namespace):
    from wifisense.config import load_config
    return load_config(args.config)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_sources:
        print("supported --source backends:")
        for name in SUPPORTED_SOURCES:
            print(f"  {name}")
        print("\n--mode choices: auto | rssi | csi")
        return 0

    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    if args.replay_speed is not None:
        cfg.set_path("replay.speed", args.replay_speed)
    if args.router_host:
        cfg.set_path("router.host", args.router_host)
    if args.router_user:
        cfg.set_path("router.user", args.router_user)
    if args.router_pass:
        cfg.set_path("router.password", args.router_pass)
    if args.router_protocol:
        cfg.set_path("router.protocol", args.router_protocol)
        cfg.set_path("router.port", 23 if args.router_protocol == "telnet" else 22)
    if args.router_iface:
        cfg.set_path("router.iface", args.router_iface)
    if args.print_config and not args.capabilities:
        print(dump_config(cfg))
        return 0

    if args.train:
        return cmd_train(args)

    if args.capabilities:
        return cmd_capabilities(args, cfg)

    if args.discover:
        return cmd_discover(args)

    from wifisense.runner import SensingRunner

    runner = SensingRunner(
        config=cfg,
        source=args.source,
        mode=args.mode,
        interface=args.interface,
        replay=args.replay,
        csi_capture=args.csi_pcap,
        csi_tool=args.csi_tool,
        method=args.method,
        model_path=args.model_path,
        dashboard=None if args.no_dashboard else True,
        log=None if args.no_log else True,
        console=not args.no_console,
        retry_forever=not args.no_retry,
        calibrate=args.calibrate or args.calibrate_only,
        calibrate_only=args.calibrate_only,
        calibrate_zones=args.calibrate_zones,
        zone_duration=args.zone_duration,
        zone_names=args.zone_names,
        arm=args.arm,
        duration=args.duration,
        points=args.points or 900,
        quiet=args.quiet,
    )

    try:
        return runner.run()
    except Exception as exc:  # final safety net — clear error, never a traceback dump to a phone
        print(f"\n[WiFi Sense] FATAL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
