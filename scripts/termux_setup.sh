#!/usr/bin/env bash
# Termux (Android): setup sensor di HP kamu sendiri.
# 1) install Termux + Termux:API dari F-Droid
# 2) pkg install python python-pip termux-api
# 3) termux-setup-storage
# 4) bash scripts/termux_setup.sh
set -e
echo "[termux] install python deps..."
pkg install -y python python-pip termux-api qr-code-terminal 2>/dev/null || true
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[termux] wake lock biar HP gak tidurin prosesnya"
termux-wake-lock || true

echo "[termux] cek WiFi API (butuh izin lokasi untuk Termux:API)"
termux-wifi-connectioninfo | head -c 300
echo

echo
echo "[termux] jalankan:"
echo "    python main.py --source termux --calibrate --duration 45"
echo "    python main.py --source termux"
echo
echo "HP kamu jadi sensor. Dashboard: buka http://127.0.0.1:8000/ di browser HP"
echo "atau akses dari HP lain di WiFi yang sama."
