"""Configuration loading / merging.

YAML file + CLI overrides -> a plain nested dict wrapped in :class:`Config`.
No dependency beyond PyYAML, which is a hard requirement anyway.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config(dict):
    """dict with dotted-path access: ``cfg.get_path("detection.presence_k")``."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def clone(self) -> "Config":
        return Config(copy.deepcopy(dict(self)))


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike | None = None,
                overrides: Mapping[str, Any] | None = None) -> Config:
    """Load ``config/default.yaml``, overlay *path*, then overlay *overrides*."""
    cfg: Dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return Config(cfg)


def apply_cli_overrides(cfg: Config, args: Any) -> Config:
    """Map argparse namespace fields onto config paths (only if not None/False)."""
    mapping = {
        "interface": "interface",
        "channel": "channel",
        "source": "source",
        "log_dir": "logging.dir",
        "port": "dashboard.port",
        "host": "dashboard.host",
        "duration": "calibration.duration_s",
        "window_s": "sampling.window_s",
        "interval_s": "sampling.interval_s",
        "presence_k": "detection.presence_k",
        "motion_k": "detection.motion_k",
    }
    for attr, dotted in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            cfg.set_path(dotted, value)

    if getattr(args, "no_dashboard", False):
        cfg.set_path("dashboard.enabled", False)
    if getattr(args, "no_log", False):
        cfg.set_path("logging.enabled", False)
    if getattr(args, "no_color", False):
        cfg.set_path("console.color", False)
    if getattr(args, "method", None):
        cfg.set_path("detection.method", args.method)
    return cfg


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(dict(cfg), sort_keys=False, default_flow_style=False)


def iter_leaf_paths(cfg: Mapping[str, Any], prefix: str = "") -> Iterable[str]:
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from iter_leaf_paths(value, f"{path}.")
        else:
            yield path
