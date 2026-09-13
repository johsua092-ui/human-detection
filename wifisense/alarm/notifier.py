"""Notifications: console, generic webhook, Telegram, and a multi-fanout wrapper.

Used by the intrusion-alarm layer ("deteksi maling") to tell you that somebody
walked into the room while you were out. Everything is stdlib ``urllib`` — no
requests dependency — and every send is wrapped so a dead network can never take
the sensing loop down with it.

Telegram credentials, in priority order:

1. explicit config (``alarm.notify.telegram.bot_token`` / ``.chat_id``),
2. environment (``WIFISENSE_TELEGRAM_TOKEN`` / ``WIFISENSE_TELEGRAM_CHAT``),
3. this machine's Synapse install if present — token from ``~/.synapse/.env``
   (``TELEGRAM_BOT_TOKEN``) and chat id from ``~/.synapse/config.yaml``
   (``platforms.telegram.home_channel.chat_id``).

Nothing is ever printed back, so a token cannot leak into a screenshot or log.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

TIMEOUT = 10.0
SEVERITY_ORDER = {"info": 0, "notice": 1, "alarm": 2}

# --------------------------------------------------------------------------- #
# credential discovery
# --------------------------------------------------------------------------- #

def _read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def discover_telegram(configured_token: Optional[str] = None,
                      configured_chat: Optional[str] = None
                      ) -> Dict[str, Optional[str]]:
    token = configured_token or os.environ.get("WIFISENSE_TELEGRAM_TOKEN")
    chat = configured_chat or os.environ.get("WIFISENSE_TELEGRAM_CHAT")
    source = "config" if (configured_token or configured_chat) else "env"

    if not token or not chat:
        env_file = Path(os.environ.get("SYNAPSE_HOME", "~/.synapse")).expanduser() / ".env"
        env = _read_env_file(env_file)
        if not token:
            token = env.get("TELEGRAM_BOT_TOKEN") or env.get("SYNAPSE_TELEGRAM_BOT_TOKEN")
            if token:
                source = f"synapse:{env_file.name}"
        if not chat:
            token2, chat2 = _from_synapse_config()
            chat = chat2
            if token2 and not token:
                token = token2
    return {"token": token, "chat_id": chat, "source": source}


def _from_synapse_config() -> tuple[Optional[str], Optional[str]]:
    path = Path(os.environ.get("SYNAPSE_HOME", "~/.synapse")).expanduser() / "config.yaml"
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None, None
    try:
        home = cfg["platforms"]["telegram"]["home_channel"]
    except Exception:
        return None, None
    chat = home.get("chat_id")
    token = home.get("token") or home.get("bot_token")
    return (str(token) if token else None), (str(chat) if chat else None)


# --------------------------------------------------------------------------- #
# notifiers
# --------------------------------------------------------------------------- #

class Notifier:
    name = "base"

    def send(self, title: str, message: str, severity: str = "info",
             extra: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        return {"notifier": self.name, "configured": True}


class ConsoleNotifier(Notifier):
    name = "console"

    def __init__(self, stream=None, color: bool = True):
        import sys
        self.stream = stream or sys.stdout
        self.color = color

    def send(self, title: str, message: str, severity: str = "info",
             extra: Optional[Dict[str, Any]] = None) -> bool:
        color = ""
        reset = ""
        if self.color:
            color = {"alarm": "\033[41;97;1m", "notice": "\033[33;1m",
                     "info": "\033[36m"}.get(severity, "")
            reset = "\033[0m" if color else ""
        stamp = time.strftime("%H:%M:%S")
        self.stream.write(f"{color}[{stamp}] {title}{reset} {message}\n")
        self.stream.flush()
        return True


class WebhookNotifier(Notifier):
    """POST a small JSON body to any URL (ntfy, Discord, Slack, Home Assistant...)."""

    name = "webhook"

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = TIMEOUT):
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def send(self, title: str, message: str, severity: str = "info",
             extra: Optional[Dict[str, Any]] = None) -> bool:
        body = json.dumps({"title": title, "message": message, "severity": severity,
                           "ts": time.time(), **(extra or {})}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None,
                 timeout: float = TIMEOUT, parse_mode: str = "HTML"):
        creds = discover_telegram(token, chat_id)
        self.token = creds["token"]
        self.chat_id = creds["chat_id"]
        self.credential_source = creds["source"]
        self.timeout = timeout
        self.parse_mode = parse_mode
        self.sent = 0
        self.failed = 0
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def health(self) -> Dict[str, Any]:
        return {"notifier": self.name, "configured": self.configured,
                "credential_source": self.credential_source,
                "sent": self.sent, "failed": self.failed, "last_error": self.last_error}

    def send(self, title: str, message: str, severity: str = "info",
             extra: Optional[Dict[str, Any]] = None) -> bool:
        if not self.configured:
            self.last_error = "no bot token / chat id available"
            return False
        label = {"alarm": "[ALARM]", "notice": "[NOTICE]", "info": "[INFO]"}.get(severity, "[INFO]")
        text = f"<b>{label} {_escape(title)}</b>\n{_escape(message)}"
        if extra:
            tail = "\n".join(f"{_escape(str(k))}: <code>{_escape(str(v))}</code>"
                             for k, v in extra.items())
            text = f"{text}\n{tail}"
        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text[:4000],
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": "true",
        }).encode()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=payload),
                                        timeout=self.timeout) as resp:
                ok = 200 <= resp.status < 300
                self.sent += 1 if ok else 0
                self.failed += 0 if ok else 1
                return ok
        except urllib.error.HTTPError as exc:
            self.failed += 1
            self.last_error = f"HTTP {exc.code}"
            return False
        except (urllib.error.URLError, OSError) as exc:
            self.failed += 1
            self.last_error = str(exc)
            return False


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class MultiNotifier(Notifier):
    """Fan out to several notifiers; report per-target success."""

    name = "multi"

    def __init__(self, notifiers: Optional[List[Notifier]] = None,
                 min_severity: str = "info"):
        self.notifiers: List[Notifier] = list(notifiers or [])
        self.min_severity = min_severity
        self.log: List[Dict[str, Any]] = []

    def add(self, notifier: Notifier) -> None:
        self.notifiers.append(notifier)

    def send(self, title: str, message: str, severity: str = "info",
             extra: Optional[Dict[str, Any]] = None) -> bool:
        if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get(self.min_severity, 0):
            return False
        any_ok = False
        for notifier in self.notifiers:
            try:
                ok = notifier.send(title, message, severity, extra)
            except Exception as exc:  # a broken notifier must not kill the loop
                ok = False
                notifier.__dict__.setdefault("last_error", str(exc))
            any_ok = any_ok or ok
            self.log.append({"t": time.time(), "notifier": notifier.name,
                             "severity": severity, "ok": ok})
            if len(self.log) > 500:
                del self.log[:250]
        return any_ok

    def health(self) -> Dict[str, Any]:
        return {"notifiers": [n.health() for n in self.notifiers],
                "sent": len(self.log)}


def build_notifiers(config: Any) -> MultiNotifier:
    """Read ``alarm.notify.*`` from config and build the fan-out."""
    notify_cfg = (config.get_path("alarm.notify", {}) if config else {}) or {}
    multi = MultiNotifier(min_severity=notify_cfg.get("min_severity", "info"))
    if notify_cfg.get("console", True):
        multi.add(ConsoleNotifier(color=(config.get_path("console.color", True) if config else True)))
    webhook = notify_cfg.get("webhook_url")
    if webhook:
        multi.add(WebhookNotifier(webhook, headers=notify_cfg.get("webhook_headers") or {}))
    tg = notify_cfg.get("telegram") or {}
    if tg.get("enabled"):
        multi.add(TelegramNotifier(tg.get("bot_token"), tg.get("chat_id")))
    return multi
