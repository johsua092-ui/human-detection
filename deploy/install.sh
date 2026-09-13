#!/usr/bin/env bash
# install.sh — pasang wifi-human-sensing sebagai service systemd (nyala 24/7)
# Usage: sudo bash deploy/install.sh /path/ke/wifi-human-sensing
set -euo pipefail

PROJECT_DIR="${1:-/opt/wifi-human-sensing}"
PROJECT_DIR="$(realpath "$PROJECT_DIR")"

echo "[install] project dir : $PROJECT_DIR"
test -f "$PROJECT_DIR/main.py" || { echo "gak ketemu main.py di $PROJECT_DIR"; exit 1; }
test -x "$PROJECT_DIR/.venv/bin/python" || { echo "buat venv dulu: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

# kalau belum ada venv deps, install
"$PROJECT_DIR/.venv/bin/python" -c "import numpy, scipy, fastapi" 2>/dev/null || \
    "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# sesuaikan path di unit file
UNIT="$PROJECT_DIR/deploy/wifisense.service"
sed "s|/opt/wifi-human-sensing|$PROJECT_DIR|g" "$UNIT" > /etc/systemd/system/wifisense.service

systemctl daemon-reload
systemctl enable wifisense
systemctl restart wifisense

echo
echo "[install] selesai. Cek dengan:"
echo "    systemctl status wifisense"
echo "    journalctl -u wifisense -f"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "    dashboard: http://${IP:-<IP-laptop>}:8000/"
