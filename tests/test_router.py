"""Router-as-sensor tests: telnet negotiation, command parsing, factory wiring."""

import pytest

from wifisense.collectors import CollectorError, build_collector
from wifisense.collectors.router import RouterRSSICollector, _SSHWrapper, _Telnet


IW_STATION_TEXT = """Station 6a:3c:09:12:dd:55 (on wlan0)
	inactive time:  120 ms
	signal: 	-42 dBm
	signal avg:	-44 dBm
	rx bitrate:	144.4 MBit/s
	connected time:  33 seconds
Station 00:11:22:33:44:55 (on wlan0)
	signal: 	-61 dBm
"""


def test_parse_clients():
    collector = RouterRSSICollector(config=None, host="192.168.1.1")
    clients = collector._parse_clients(IW_STATION_TEXT)
    assert len(clients) == 2
    assert clients[0]["signal_dbm"] == -42.0
    assert clients[1]["signal_dbm"] == -61.0


def test_telnet_strip_iac():
    data = b"\xff\xfb\x01login:\xff\xfc\x00"
    out = _Telnet._strip_iac(data)
    # IAC negotiation removed; login text preserved
    assert b"\xff" not in out
    assert b"login:" in out


def test_factory_router_without_host_raises_clear_error(cfg):
    collector = build_collector(cfg, source="router")
    with pytest.raises(CollectorError) as exc:
        collector.preflight()
    assert "no router host configured" in str(exc.value)
    assert exc.value.hints


def test_factory_router_wiring(cfg):
    local = cfg.clone()   # never mutate the session fixture: it leaks across tests
    local.set_path("router.host", "192.168.1.1")
    local.set_path("router.user", "root")
    local.set_path("router.password", "Zte521")
    local.set_path("router.protocol", "ssh")
    collector = build_collector(local, source="router")
    assert isinstance(collector, RouterRSSICollector)
    assert collector.host == "192.168.1.1"
    assert collector.protocol == "ssh"


def test_ssh_wrapper_mocked_command(monkeypatch):
    import subprocess

    class FakeProc:
        returncode = 0
        stdout = IW_STATION_TEXT
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    wrapper = _SSHWrapper("192.168.1.1", 22, "root", "")
    out = wrapper.command("iw dev wlan0 station dump")
    assert "signal:" in out
    assert "-42 dBm" in out


def test_ssh_wrapper_error_surfaces(monkeypatch):
    import subprocess

    class FailProc:
        returncode = 255
        stdout = ""
        stderr = "Permission denied (publickey)."

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FailProc())
    wrapper = _SSHWrapper("192.168.1.1", 22, "root", "wrong")
    with pytest.raises(CollectorError) as exc:
        wrapper.command("iw dev wlan0 station dump")
    assert "ssh command failed" in str(exc.value)


def test_router_preflight_requires_host(cfg):
    local = cfg.clone()
    local.set_path("router.host", None)
    collector = RouterRSSICollector(config=local, host=None)
    with pytest.raises(CollectorError):
        collector.preflight()
