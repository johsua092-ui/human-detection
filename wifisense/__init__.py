"""WiFi Human Presence Sensing.

Detects human presence / motion from ordinary WiFi radio signals using only a
Linux (or macOS / Windows, best-effort) host NIC and any nearby Access Point as
the transmitter. No ESP32 / ESP8266 / Arduino / Raspberry Pi required.

Two physical-layer modes:

* ``csi``  - Channel State Information. Per-subcarrier amplitude + phase.
             Requires a NIC/driver that exposes CSI (nexmon_csi, Intel 5300
             Linux 802.11n CSI Tool, ath9k CSI tool) or an offline CSI capture.
* ``rssi`` - Received Signal Strength Indicator. Universal fallback, works on
             any stock NIC. Coarser, but genuinely detects presence/motion
             because a human body moving through the room changes the multipath
             sum that the RSSI value is a scalar projection of.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
