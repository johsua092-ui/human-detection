"""Parser unit tests — the text<->data boundary of every backend."""

from wifisense.collectors import parsers as p

IWLINK = """
Connected to 6a:3c:09:12:dd:55 (on wlan0)
	SSID: RumahKita
	freq: 2437
	RX: 17325 bytes (147 packets)
	TX: 6120 bytes (45 packets)
	signal: -48 dBm
	rx bitrate: 144.4 MBit/s MCS 7 short GI
	tx bitrate: 72.2 MBit/s MCS 3 short GI
	dtim period: 2
	beacon int: 100
"""

NOT_LINKED = "Not connected.\n"


def test_parse_iw_link_connected():
    link = p.parse_iw_link(IWLINK)
    assert link["connected"] is True
    assert link["bssid"] == "6a:3c:09:12:dd:55"
    assert link["ssid"] == "RumahKita"
    assert link["signal_dbm"] == -48.0
    assert link["freq_mhz"] == 2437.0


def test_parse_iw_link_not_connected():
    link = p.parse_iw_link(NOT_LINKED)
    assert link["connected"] is False
    assert link["signal_dbm"] is None


def test_parse_iw_dev():
    out = p.parse_iw_dev("""phy#0
	Interface wlan0
		ifindex 3
		wdev 0x1
		addr 12:34:56:78:9a:bc
		type managed
		channel 6 (2437 MHz), width: 20 MHz
""")
    assert out[0]["iface"] == "wlan0"
    assert out[0]["type"] == "managed"
    assert out[0]["addr"] == "12:34:56:78:9a:bc"


def test_parse_iw_phy_modes():
    text = """Supported interface modes:
		 * IBSS
		 * managed
		 * AP
		 * monitor
		 * P2P-client
"""
    modes = p.parse_iw_phy_modes(text)
    assert "monitor" in modes
    assert "managed" in modes


def test_parse_station_dump():
    text = """Station 6a:3c:09:12:dd:55 (on wlan0)
	inactive time:  120 ms
	rx bytes: 1324
	signal: 	-42 dBm
	signal avg:	-44 dBm
	rx bitrate:	144.4 MBit/s
	tx bitrate:	72.2 MBit/s
	connected time:  33 seconds
Station 00:11:22:33:44:55 (on wlan0)
	signal: 	-61 dBm
"""
    stations = p.parse_iw_station_dump(text)
    assert len(stations) == 2
    assert stations[0]["mac"] == "6a:3c:09:12:dd:55"
    assert stations[0]["signal_dbm"] == -42.0
    assert stations[0]["signal_avg_dbm"] == -44.0
    assert stations[1]["signal_dbm"] == -61.0


PROC_WIRELESS = """Inter-| sta-|   Quality        |   Signal  |  Noise  |  Tx
        in |     |   /100       |    dBm    |   dBm   | rate
wlan0   : 100  100.  -44        -256      1000
eth0    : 0    0.    0          0         0
"""


def test_parse_proc_wireless():
    rows = p.parse_proc_wireless(PROC_WIRELESS)
    wlan = [r for r in rows if r["iface"] == "wlan0"]
    eth = [r for r in rows if r["iface"] == "eth0"]
    assert len(wlan) == 1 and len(eth) == 1  # parser keeps every row (fidelity)
    row = wlan[0]
    assert row["level_dbm"] == -44.0
    assert row["level_reliable"] is True
    assert row["noise_dbm"] == -256.0  # the sentinel for "no real noise value"
    assert eth[0]["level_dbm"] is None  # level 0 -> unknown, not a real RSSI


def test_parse_proc_wireless_unreliable_level():
    text = PROC_WIRELESS.replace("-44", "0")
    rows = p.parse_proc_wireless(text)
    assert rows[0]["level_reliable"] is False


def test_parse_tshark_line():
    rec = p.parse_tshark_line("1712345678.123456\t-52\t6a:3c:09:12:dd:55\t2437\t146")
    assert rec["rssi"] == -52.0
    assert rec["bssid"] == "6a:3c:09:12:dd:55"
    assert rec["freq_mhz"] == 2437.0


def test_parse_tshark_line_multi_antenna():
    rec = p.parse_tshark_line("1712345678.5\t-55,-49\t6a:3c:09:12:dd:55\t5180\t200")
    assert rec["rssi"] == -49.0  # strongest antenna wins


def test_parse_tshark_line_empty():
    assert p.parse_tshark_line("") is None
    assert p.parse_tshark_line("\t\t\t\t") is None


def test_parse_tcpdump_line():
    rec = p.parse_tcpdump_line(
        "08:12:33.100111 6a:3c:09:12:dd:55 Beacon (RumahKita) [1.0* 6.0* 11.0* 18.0] "
        "BSSID:6a:3c:09:12:dd:55 -48dBm signal")
    assert rec["rssi"] == -48.0
    assert rec["bssid"] == "6a:3c:09:12:dd:55"


def test_pct_to_dbm():
    assert p.pct_to_dbm(100) == -50.0
    assert p.pct_to_dbm(78) == -61.0
    assert p.pct_to_dbm(0) == -100.0


def test_parse_netsh():
    text = """Name                   : Wi-Fi
SSID                   : RumahKita
BSSID                  : 6a:3c:09:12:dd:55
Signal                 : 78%
Radio type             : 802.11ac
Channel                : 36
"""
    rows = p.parse_netsh_interfaces(text)
    assert rows[0]["ssid"] == "RumahKita"
    assert rows[0]["signal_dbm"] == -61.0
    assert rows[0]["channel"] == 36.0


def test_parse_airport():
    text = """
     agrCtlRSSI: -47
     agrCtlNoise: -92
        state: running
      op mode: station
       lastTxRate: 867
          SSID: RumahKita
      channel: 36,1
"""
    info = p.parse_airport(text)
    assert info["connected"] is True
    assert info["signal_dbm"] == -47.0
    assert info["noise_dbm"] == -92.0
    assert info["channel"] == 36.0


def test_parse_termux_wifi():
    text = '{"supplicant_state":"COMPLETED","rssi":-56,"ssid":"RumahKita",' \
           '"bssid":"6a:3c:09:12:dd:55","frequency":2412,"link_speed_mbps":72}'
    info = p.parse_termux_wifi(text)
    assert info["signal_dbm"] == -56.0
    assert info["connected"] is True
    assert info["channel"] == 1.0


def test_freq_to_channel():
    assert p.freq_to_channel(2412) == 1.0
    assert p.freq_to_channel(2437) == 6.0
    assert p.freq_to_channel(5180) == 36.0


def test_parse_scan_results():
    text = """BSS 6a:3c:09:12:dd:55(on wlan0)
	freq: 2437
	DS Parameter set: channel 6
	SSID: RumahKita
	signal: -52.00 dBm
"""
    bss = p.parse_scan_results(text)
    assert bss[0]["bssid"] == "6a:3c:09:12:dd:55"
    assert bss[0]["channel"] == 6.0
    assert bss[0]["signal_dbm"] == -52.0
