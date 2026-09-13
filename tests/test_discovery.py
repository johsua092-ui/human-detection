"""Discovery tests: gateway parsing, candidate generation, auto-setup write.

Everything is pure/mocked so it runs on a machine with no WiFi and no router.
"""

import json
from pathlib import Path

import pytest

from wifisense import discovery
from wifisense.discovery import Candidate, discover, parse_default_gateway


PROC_ROUTE = """Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
eth0	00000000	0101A8C0	0003	0	0	100	00000000	0	0	0
eth0	0000A8C0	00000000	0001	0	0	100	00FFFFFF	0	0	0
"""


def test_parse_default_gateway():
    assert parse_default_gateway(PROC_ROUTE) == "192.168.1.1"
    assert parse_default_gateway("no routes here") is None


def test_probe_port_returns_bool(monkeypatch):
    import socket

    def fake(addr, timeout=None):
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", fake)
    assert discovery.probe_port("127.0.0.1", 23) is False


def test_candidate_serialisable():
    cand = Candidate("router", "gateway", True, "detail",
                     config={"router.host": "192.168.1.1"}, command="x")
    d = cand.to_dict()
    assert d["usable"] is True
    json.dumps(d)


def test_discover_returns_structured_result(monkeypatch):
    monkeypatch.setattr(discovery, "default_gateway", lambda: "192.168.1.1")
    monkeypatch.setattr(discovery, "probe_port", lambda host, port, timeout=1.0: False)
    monkeypatch.setattr(discovery, "http_banner", lambda *a, **k: "")
    result = discovery.discover(try_credentials=False)
    assert result["gateway"] == "192.168.1.1"
    assert isinstance(result["candidates"], list)
    assert "best" in result


def test_discover_with_open_telnet_picks_router(monkeypatch, cfg):
    monkeypatch.setattr(discovery, "default_gateway", lambda: "192.168.1.1")

    def fake_probe(host, port, timeout=1.0):
        return port == 23   # telnet open, ssh closed

    def fake_telnet_login(host, port=23, credentials=(), timeout=4.0):
        return {"user": "root", "password": "Zte521", "output": "BusyBox #"}

    monkeypatch.setattr(discovery, "probe_port", fake_probe)
    monkeypatch.setattr(discovery, "try_telnet_login", fake_telnet_login)
    monkeypatch.setattr(discovery, "http_banner", lambda *a, **k: "")

    result = discovery.discover(try_credentials=True)
    router = next(c for c in result["candidates"] if c["kind"] == "router")
    assert router["usable"] is True
    assert "192.168.1.1" in router["config"]["router.host"]


def test_auto_setup_writes_local_config(tmp_path, cfg):
    result = {
        "gateway": "192.168.1.1",
        "candidates": [],
        "best": Candidate(
            "router", "gateway", True, "telnet shell ok",
            config={"router.host": "192.168.1.1", "router.user": "root",
                    "router.password": "Zte521", "router.protocol": "telnet"},
            command="python main.py --source router").to_dict(),
    }
    out = discovery.auto_setup(cfg, result, path=tmp_path / "local.yaml")
    text = out.read_text()
    assert "router" in text and "192.168.1.1" in text
    # the base config must be loadable together with the overlay
    from wifisense.config import load_config
    merged = load_config(out)
    assert merged.get_path("router.host") == "192.168.1.1"
