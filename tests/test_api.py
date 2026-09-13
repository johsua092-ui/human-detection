"""Dashboard API + websocket tests.

The important contract here is that the API never blocks or crashes the sensing
loop, and that PIN roles are enforced when auth is enabled (family = view only).
"""

import json

import pytest

from wifisense.dashboard.server import create_app
from wifisense.utils.state import StateHub


def _make(cfg, auth=False, view_pin=None, admin_pin=None):
    cfg = cfg.clone()
    cfg.set_path("dashboard.auth.enabled", auth)
    cfg.set_path("dashboard.auth.view_pin", view_pin)
    cfg.set_path("dashboard.auth.admin_pin", admin_pin)
    hub = StateHub(history_points=100)
    hub.publish_sample(1700000000.0, -48.2)
    hub.publish_sample(1700000000.05, -48.9)
    hub.publish_meta(mode="rssi", source="rssi_iw", interface="wlan0")
    hub.publish_state({"t": 1700000000.05, "presence": True, "motion": True,
                       "confidence": 87.0, "presence_label": "HUMAN DETECTED",
                       "motion_label": "DETECTED", "reason": "test",
                       "features": {"var": 0.183}})
    hub.publish_alarm({"state": "ARMED_AWAY", "trigger_count": 1, "notifiers": {"notifiers": []}})
    app = create_app(hub, cfg, None)
    return app, hub


def test_health_and_state(cfg):
    from fastapi.testclient import TestClient
    app, _hub = _make(cfg)
    client = TestClient(app)
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["mode"] == "rssi"
    state = client.get("/api/state").json()
    assert state["latest"]["presence"] is True
    assert state["alarm"]["state"] == "ARMED_AWAY"


def test_snapshot_and_incremental_history(cfg):
    from fastapi.testclient import TestClient
    app, _hub = _make(cfg)
    client = TestClient(app)
    snap = client.get("/api/snapshot?points=100").json()
    assert len(snap["history"]) == 2
    assert snap["last_index"] == 2
    delta = client.get("/api/history?since=1").json()
    assert len(delta["points"]) == 1


def test_index_and_manifest(cfg):
    from fastapi.testclient import TestClient
    app, _hub = _make(cfg)
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "WiFi Sense" in page.text
    assert page.headers["X-WiFiSense-Role"] == "admin"   # auth disabled -> LAN trusted
    manifest = client.get("/manifest.webmanifest").json()
    assert manifest["name"] == "WiFi Sense"


def test_actions_require_admin_when_auth_off(cfg):
    from fastapi.testclient import TestClient
    app, hub = _make(cfg)
    client = TestClient(app)
    assert client.post("/api/calibrate").json()["ok"] is True
    assert hub.consume_control()["calibration_requested"] is True
    assert client.post("/api/arm?mode=away").json()["mode"] == "away"
    assert client.post("/api/disarm").json()["ok"] is True


def test_pin_roles_enforced(cfg):
    from fastapi.testclient import TestClient
    app, hub = _make(cfg, auth=True, view_pin="1111", admin_pin="9999")
    client = TestClient(app)

    # no pin -> reads allowed (the PIN overlay itself must be served),
    # control endpoints blocked
    assert client.get("/api/state").status_code == 200
    assert client.get("/api/whoami").json()["role"] == "none"
    assert client.post("/api/calibrate").status_code == 401

    # view pin: read yes, control no
    assert client.get("/api/state?pin=1111").status_code == 200
    assert client.get("/api/whoami?pin=1111").json()["role"] == "view"
    assert client.post("/api/calibrate?pin=1111").status_code == 403

    # admin pin: control yes
    assert client.post("/api/calibrate?pin=9999").status_code == 200
    assert hub.consume_control()["calibration_requested"] is True
    assert client.get("/api/whoami?pin=9999").json()["role"] == "admin"

    # wrong pin
    assert client.get("/api/whoami?pin=0000").json()["role"] == "none"


def test_websocket_snapshot_then_tick(cfg):
    from fastapi.testclient import TestClient
    app, hub = _make(cfg)
    client = TestClient(app)
    with client.websocket_connect("/ws?points=50") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        assert first["role"] == "admin"
        assert len(first["history"]) == 2
        hub.publish_sample(1700000001.0, -47.0)
        tick = ws.receive_json()
        assert tick["type"] == "tick"
        assert tick["points"]
        assert tick["points"][-1]["rssi"] == pytest.approx(-47.0)
        ws.send_json({"type": "calibrate"})
        # the control flag is consumed by the sensing loop, not by the socket;
        # the reader task runs concurrently, so poll briefly for it
        for _ in range(50):
            if hub.consume_control()["calibration_requested"]:
                break
            import time
            time.sleep(0.02)
        else:
            raise AssertionError("calibration request never reached the hub")
