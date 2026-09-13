"""FastAPI dashboard backend.

Serves a phone-first web UI plus a small JSON API over the same
:class:`~wifisense.utils.state.StateHub` the sensing loop writes into.

Design notes that matter on a home LAN with several phones watching at once:

* **Incremental websocket feed.** Each client is sent a snapshot once on
  connect, then only the new samples (``i`` > last seen), so ten family phones
  do not each pull the full history 5×/second.
* **Polling fallback.** If the websocket cannot be established (some Android
  browsers on captive-portal-flagged LANs), the same data is available from
  ``/api/snapshot`` and ``/api/history?since=``. The UI degrades to 1 Hz polling
  instead of showing a dead page.
* **Optional PIN roles.** ``dashboard.auth`` can require a PIN: the *view* PIN
  lets family members watch, the *admin* PIN unlocks calibrate/arm/disarm. With
  auth off (default) the LAN is trusted and everyone is admin — which is a
  deliberate, documented choice, not an oversight.
* The loop is the only writer; the API never blocks acquisition.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import pathlib
import time
from typing import Any, Dict, Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
VIEW = "view"
ADMIN = "admin"


def _pin_ok(supplied: Optional[str], expected: Optional[str]) -> bool:
    if not expected:
        return False
    if not supplied:
        return False
    return hmac.compare_digest(str(supplied), str(expected))


class AccessControl:
    """Tiny PIN gate. Two roles: family (view) and owner (admin)."""

    def __init__(self, config: Any = None):
        cfg = (lambda path, default: config.get_path(path, default) if config else default)
        self.enabled = bool(cfg("dashboard.auth.enabled", False))
        self.view_pin = cfg("dashboard.auth.view_pin", None)
        self.admin_pin = cfg("dashboard.auth.admin_pin", None)
        self._failures: Dict[str, int] = {}
        if self.enabled and not (self.view_pin or self.admin_pin):
            self.enabled = False  # misconfigured: fail open on the LAN, but say so
        self.warning = ("dashboard.auth is enabled but no PIN is set — auth disabled"
                        if (cfg("dashboard.auth.enabled", False) and not self.enabled) else None)

    def role_for(self, pin: Optional[str]) -> Optional[str]:
        if not self.enabled:
            return ADMIN
        if pin and self.admin_pin and _pin_ok(pin, self.admin_pin):
            return ADMIN
        if pin and self.view_pin and _pin_ok(pin, self.view_pin):
            return VIEW
        return None

    def note_failure(self, client: str) -> int:
        self._failures[client] = self._failures.get(client, 0) + 1
        return self._failures[client]

    def clear_failures(self, client: str) -> None:
        self._failures.pop(client, None)

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "roles": [VIEW, ADMIN] if self.enabled else [ADMIN],
                "warning": self.warning}


def create_app(hub: Any, config: Any = None, logger: Any = None) -> FastAPI:
    app = FastAPI(title="WiFi Human Presence Sensing", version="1.0.0",
                  docs_url="/api/docs", redoc_url=None)
    access = AccessControl(config)
    push_hz = float(config.get_path("dashboard.push_hz", 5.0)) if config else 5.0
    push_interval = 1.0 / max(0.2, push_hz)
    started = time.time()

    app.state.hub = hub
    app.state.access = access
    app.state.started = started

    # ------------------------------------------------------------------ #
    # auth helper
    # ------------------------------------------------------------------ #
    def current_role(request: Request, pin: Optional[str] = None) -> str:
        supplied = pin or request.cookies.get("wifisense_pin") or request.query_params.get("pin")
        role = access.role_for(supplied)
        if role is None:
            client = request.client.host if request.client else "?"
            count = access.note_failure(client)
            if count >= 5:
                raise HTTPException(status_code=429, detail="too many attempts")
            raise HTTPException(status_code=401, detail="PIN required")
        access.clear_failures(request.client.host if request.client else "?")
        return role

    def require_admin(request: Request) -> str:
        role = current_role(request)
        if role != ADMIN:
            raise HTTPException(status_code=403, detail="admin PIN required")
        return role

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    @app.get("/")
    async def index(pin: Optional[str] = None, request: Request = None):
        role = "admin"
        if access.enabled:
            role = access.role_for(pin or (request.cookies.get("wifisense_pin") if request else None)) or "none"
        response = FileResponse(STATIC_DIR / "index.html")
        if pin and access.role_for(pin):
            response.set_cookie("wifisense_pin", pin, max_age=60 * 60 * 24 * 30,
                                httponly=False, samesite="lax")
        response.headers["X-WiFiSense-Role"] = role
        return response

    @app.get("/manifest.webmanifest")
    async def manifest():
        return JSONResponse({
            "name": "WiFi Sense", "short_name": "WiFi Sense", "start_url": "/",
            "display": "standalone", "background_color": "#0b0f14", "theme_color": "#0b0f14",
            "icons": [],
        })

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    @app.get("/api/health")
    async def health():
        snap = hub.snapshot(points=1)
        return {
            "ok": True,
            "uptime_s": round(time.time() - started, 1),
            "sensing_uptime_s": round(snap.get("uptime_s", 0.0), 1),
            "clients": getattr(app.state, "ws_clients", 0),
            "mode": snap.get("meta", {}).get("mode"),
            "source": snap.get("meta", {}).get("source"),
            "auth": access.status(),
            "version": "1.0.0",
        }

    @app.get("/api/capabilities")
    async def capabilities():
        return hub.snapshot(points=1).get("meta", {})

    @app.get("/api/state")
    async def state():
        snap = hub.snapshot(points=1)
        return {
            "now": snap["now"], "latest": snap["latest"], "alarm": snap["alarm"],
            "calibration": snap["calibration"], "stats": snap["stats"],
            "warnings": snap["warnings"], "meta": snap["meta"],
            "last_index": snap["last_index"],
        }

    @app.get("/api/snapshot")
    async def snapshot(points: int = Query(300, ge=0, le=5000)):
        return hub.snapshot(points=points)

    @app.get("/api/history")
    async def history(since: int = Query(0, ge=0), limit: int = Query(2000, ge=1, le=5000)):
        return hub.since(since, limit=limit)

    @app.get("/api/events")
    async def events(n: int = Query(60, ge=1, le=500)):
        snap = hub.snapshot(points=1)
        return {"events": snap["events"][-n:], "alarm": snap["alarm"]}

    @app.get("/api/windows.csv")
    async def windows_csv():
        """Download the windows log — the file you label to train --method ml."""
        if logger is None:
            raise HTTPException(status_code=404, detail="logging disabled for this session")
        path = (logger.paths or {}).get("windows")
        if not path or not pathlib.Path(path).exists():
            raise HTTPException(status_code=404, detail="no windows log written yet")
        return FileResponse(path, media_type="text/csv", filename=pathlib.Path(path).name)

    @app.post("/api/calibrate")
    async def calibrate(request: Request, recalibrate: bool = False, role: str = Depends(require_admin)):
        hub.request_calibration(recalibrate=recalibrate)
        hub.publish_calibration(state="requested", started_at=time.time())
        return {"ok": True, "requested": "recalibrate" if recalibrate else "calibrate"}

    @app.post("/api/arm")
    async def arm(mode: str = Query("away", pattern="^(away|home)$"), role: str = Depends(require_admin)):
        hub.request_arm(mode)
        return {"ok": True, "mode": mode}

    @app.post("/api/disarm")
    async def disarm(role: str = Depends(require_admin)):
        hub.request_disarm()
        return {"ok": True}

    @app.get("/api/whoami")
    async def whoami(request: Request, pin: Optional[str] = None):
        supplied = pin or request.cookies.get("wifisense_pin") or request.query_params.get("pin")
        role = access.role_for(supplied)
        return {"role": role or "none", "auth_enabled": access.enabled}

    # ------------------------------------------------------------------ #
    # websocket
    # ------------------------------------------------------------------ #
    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        role = "none"
        try:
            pin = sock.query_params.get("pin") or sock.cookies.get("wifisense_pin")
            role = access.role_for(pin) or ("none" if access.enabled else ADMIN)
            if access.enabled and role == "none":
                await sock.send_json({"type": "error", "message": "pin required"})
                await sock.close(code=4401)
                return

            app.state.ws_clients = getattr(app.state, "ws_clients", 0) + 1
            try:
                snap = hub.snapshot(points=int(sock.query_params.get("points", 300)))
                snap["type"] = "snapshot"
                snap["role"] = role
                snap["auth_enabled"] = access.enabled
                await sock.send_json(snap)
                last_index = int(snap.get("last_index", 0))

                async def reader():
                    nonlocal last_index
                    while True:
                        message = await sock.receive_text()
                        try:
                            payload = json.loads(message)
                        except Exception:
                            continue
                        kind = payload.get("type")
                        if kind == "snapshot":
                            fresh = hub.snapshot(points=int(payload.get("points", 300)))
                            fresh["type"] = "snapshot"
                            fresh["role"] = role
                            await sock.send_json(fresh)
                            last_index = int(fresh.get("last_index", last_index))
                        elif kind == "ping":
                            await sock.send_json({"type": "pong", "t": time.time()})
                        elif kind == "calibrate" and role == ADMIN:
                            hub.request_calibration(recalibrate=bool(payload.get("recalibrate")))
                        elif kind == "arm" and role == ADMIN:
                            hub.request_arm(str(payload.get("mode", "away")))
                        elif kind == "disarm" and role == ADMIN:
                            hub.request_disarm()

                read_task = asyncio.create_task(reader())
                try:
                    while True:
                        await asyncio.sleep(push_interval)
                        if read_task.done():
                            break
                        delta = hub.since(last_index)
                        last_index = int(delta.get("last_index", last_index))
                        snap = hub.snapshot(points=0)
                        await sock.send_json({
                            "type": "tick",
                            "t": time.time(),
                            "points": delta["points"],
                            "last_index": last_index,
                            "latest": snap["latest"],
                            "alarm": snap["alarm"],
                            "calibration": snap["calibration"],
                            "stats": snap["stats"],
                            "warnings": snap["warnings"],
                            "meta": snap["meta"],
                        })
                finally:
                    read_task.cancel()
            finally:
                app.state.ws_clients = max(0, getattr(app.state, "ws_clients", 1) - 1)
        except WebSocketDisconnect:
            return
        except Exception:
            try:
                await sock.close()
            except Exception:
                pass

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz():
        return "ok"

    return app


def serve(hub: Any, config: Any = None, logger: Any = None, host: str = "0.0.0.0",
          port: int = 8000, log_level: str = "warning") -> None:
    """Blocking uvicorn run (used when the dashboard runs in its own process)."""
    import uvicorn

    app = create_app(hub, config, logger)
    uvicorn.run(app, host=host, port=int(port), log_level=log_level, access_log=False)


def serve_in_thread(hub: Any, config: Any = None, logger: Any = None,
                    host: str = "0.0.0.0", port: int = 8000) -> Any:
    """Start uvicorn on a background thread; returns a handle with .stop()."""
    import threading
    import uvicorn

    app = create_app(hub, config, logger)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=int(port),
                                           log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="wifisense-dashboard", daemon=True)
    thread.start()

    class Handle:
        def __init__(self):
            self.server = server
            self.thread = thread

        def stop(self) -> None:
            server.should_exit = True
            thread.join(timeout=5)

    return Handle()
