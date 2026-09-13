#!/usr/bin/env bash
# run_forever.sh — alternatif tanpa systemd: setsid + watchdog sederhana.
# Bisa dipakai di Termux juga (dengan termux-wake-lock).
# Usage: bash deploy/run_forever.sh [args...]
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG="data/run_forever.log"
mkdir -p data

while true; do
    echo "$(date '+%F %T') starting: python main.py $*" >> "$LOG"
    "$PY" main.py --no-console "$@" >> "$LOG" 2>&1
    code=$?
    echo "$(date '+%F %T') exited with code $code — restart in 5s" >> "$LOG"
    sleep 5
done
