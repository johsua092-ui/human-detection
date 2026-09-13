# WiFi Human Presence Sensing

Deteksi keberadaan & pergerakan **manusia** dari sinyal WiFi biasa — tanpa
ESP32, ESP8266, Arduino, atau Raspberry Pi. Cukup laptop/PC Linux + WiFi NIC
yang sudah ada + router/AP WiFi di sekitarnya sebagai transmitter.

```
[WiFi Sense]  17:24:30  mode=rssi  source=synthetic  iface=-
Signal: -52.2 dBm
Variance: 0.523
Motion: DETECTED
Presence: HUMAN DETECTED
Confidence: 100%
```

---

## Isi project

```
wifi-human-sensing/
├── main.py                  # entry point (semua command ada di sini)
├── requirements.txt
├── config/default.yaml      # semua tuning ada di sini
├── wifisense/
│   ├── runner.py            # loop utama: collector → filter → features → engine
│   ├── collectors/          # backend pengambilan sinyal
│   │   ├── capabilities.py  # deteksi OS/NIC/tools/CSI → pilih backend otomatis
│   │   ├── rssi_radiotap.py # monitor mode + tshark/tcpdump (backend terbaik)
│   │   ├── rssi_iw.py       # `iw dev link` / `station dump` (tanpa root)
│   │   ├── rssi_proc.py     # /proc/net/wireless (last resort)
│   │   ├── platform_rssi.py # Windows (netsh), macOS (airport), Android/Termux
│   │   ├── csi.py           # CSI live (nexmon) + offline pcap (csiread)
│   │   └── synthetic.py     # DEMO ONLY + replay data rekaman
│   ├── processing/          # filter (median/hampel/EMA) + feature extraction
│   ├── detection/           # kalibrasi baseline, adaptive engine, optional ML
│   ├── alarm/               # mode "deteksi maling": arm/disarm + notif Telegram
│   ├── dashboard/           # FastAPI + websocket + UI mobile-first
│   ├── experimental/        # breathing/heartbeat (CSI only, eksperimental)
│   └── utils/               # console, data logger (CSV/JSONL), state hub
├── tests/                   # 66 unit/integration test
├── deploy/                  # systemd unit + install script + run_forever
└── data/, models/           # hasil logging & baseline (dibuat otomatis)
```

---

## Cara kerja (fisikanya singkat)

Manusia itu sebagian besar air. Tubuh yang bergerak di ruangan mengubah jalur
pantulan multipath sinyal WiFi (pantulan dari dinding, lantai, perabot). Perubahan
itu terlihat di RSSI sebagai fluktuasi kecil — dan di CSI sebagai perubahan
amplitude/phase per subcarrier. Sistem mengukur fluktuasi itu, membandingkannya
dengan *baseline ruangan kosong* hasil kalibrasi, lalu memutuskan:

- **PRESENCE** = lingkungan sinyal terganggu (ada yang masuk path, diam sekalipun)
- **MOTION** = sinyal berubah cepat sekarang (ada yang bergerak)

Pipeline: `WiFi → RSSI/CSI → filter noise → fitur (variance, FFT/STFT, band power,
rate of change) → z-score vs baseline → confidence → dashboard real-time + alarm`.

**Yang TIDAK bisa & tidak diklaim:** identitas orang, wajah, lokasi GPS, detak
jantung dari RSSI. Breathing sensing hanya modul eksperimental terpisah dan hanya
aktif dengan CSI berkualitas.

---

## Instalasi (Linux, Debian/Ubuntu)

```bash
# 1. tool dasar untuk WiFi sensing
sudo apt update
sudo apt install -y python3 python3-venv iw wireless-tools net-tools iproute2
sudo apt install -y tshark        # wajib untuk backend radiotap (monitor mode)
#    saat ditanya "non-superuser capture?" pilih <No> biar aman, kita jalan via sudo

# 2. project
cd ~
git clone <ini-repo-nya> wifi-human-sensing   # atau ekstrak zip-nya
cd wifi-human-sensing

# 3. virtualenv + dependency
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 4. cek hardware
.venv/bin/python main.py --capabilities
```

> Windows/macOS/Android: bisa jalan dengan backend terbatas (netsh / airport /
> Termux), lihat bagian "Platform lain". Linux tetap yang paling lengkap.

---

## Cara menjalankan di laptop (langkah per langkah)

### 1. Pastikan laptop connect ke WiFi (sensor butuh sinyal yang ada isinya)

```bash
# lihat interface wireless:
iw dev

# connect via NetworkManager (paling gampang):
nmcli device wifi list
nmcli device wifi connect "NamaWifi" password "password123"

# atau via iwd:
iwctl station wlan0 scan
iwctl station wlan0 connect "NamaWifi"

# cek status link + RSSI:
iw dev wlan0 link          # harus ada "signal: -xx dBm"
```

### 2. Kalibrasi ruangan kosong (WAJIB sekali sebelum dipakai)

Pastikan **tidak ada orang/hewan di area pengukuran**, lalu:

```bash
.venv/bin/python main.py --calibrate --duration 45
```

Selesai → tersimpan `models/baseline_rssi_wlan0.json`. Kalibrasi ulang setiap
kali posisi router/laptop dipindah, atau setelah beberapa hari (ada tombol
"re-calibrate" di dashboard juga).

### 3. Jalanin sensing + dashboard

```bash
.venv/bin/python main.py
```

Output terminal seperti di atas, dan dashboard kebuka otomatis:
- di laptop: `http://127.0.0.1:8000/`
- dari **HP keluarga**: buka `http://<IP-laptop>:8000/` (QR code dicetak di
  terminal kalau `pip install qrcode`). HP harus satu WiFi yang sama.

### 4. Pakai sebagai deteksi maling

```bash
.venv/bin/python main.py --arm away        # langsung armed
# atau arm dari dashboard (tombol "arm away" / "arm home")
```

Notifikasi Telegram: set `alarm.notify.telegram.enabled: true` + isi
`bot_token` & `chat_id` di `config/default.yaml` (atau env
`WIFISENSE_TELEGRAM_TOKEN` / `WIFISENSE_TELEGRAM_CHAT`). Kalau di mesin ini ada
Synapse, token bot Telegram otomatis ketarik dari `~/.synapse/.env`.

---

## Command lengkap

```bash
python main.py                     # auto: pilih backend terbaik
python main.py --mode rssi         # paksa RSSI
python main.py --mode csi          # paksa CSI (butuh hardware CSI)
python main.py --calibrate         # kalibrasi baseline, lalu lanjut sensing
python main.py --calibrate-only    # kalibrasi lalu exit
python main.py --dashboard         # dashboard eksplisit (default-nya sudah nyala)
python main.py --capabilities      # laporan hardware + mode yang dipakai
python main.py --list-sources
python main.py --source iw --interface wlan0
python main.py --source radiotap --interface wlan0   # butuh sudo
python main.py --source proc                          # fallback tanpa root
python main.py --replay data/windows_xxx.csv          # replay rekaman
python main.py --train --empty e1.csv e2.csv --human h1.csv   # training ML opsional
python main.py --method ml --model models/clf.joblib
python main.py --arm away --no-dashboard --no-log    # sentry mode murni
```

Semua bisa dikombinasi dengan tuning: `--window 10 --presence-k 3.5 --port 9000
--no-color --quiet --strict --no-retry`.

---

## Biar nyala 24/7 (systemd — otomatis start saat boot + auto-restart)

```bash
# sekali saja:
sudo bash deploy/install.sh /root/wifi-human-sensing
#   (membuat /etc/systemd/system/wifisense.service + enable + start)

sudo systemctl status wifisense   # cek
sudo systemctl restart wifisense  # restart
sudo journalctl -u wifisense -f   # lihat log terminal live
```

Isi unit-nya (lihat `deploy/wifisense.service`): `Restart=always`, `--arm away`
kalau mau langsung armed, `Environment=WIFISENSE_TELEGRAM_TOKEN=...` untuk alarm.

Tanpa systemd (termux/sandbox): `nohup`/`setsid` via `deploy/run_forever.sh`.

Proses sudah punya **retry-forever + watchdog**: kalau NIC lepas, AP reboot,
tshark mati — collector di-restart otomatis dengan backoff, proses gak mati.

---

## Buka dari HP keluarga (banyak HP sekaligus)

1. Laptop & HP satu WiFi.
2. Jalanin `python main.py` → terminal print URL `http://192.168.1.x:8000/`.
3. HP buka URL itu (atau scan QR). Bisa dibuka **banyak HP sekaligus** — tiap
   client dapat feed websocket sendiri (incremental, hemat baterai).
4. Mau ada PIN? `dashboard.auth.enabled: true` + `view_pin` (keluarga, cuma
   nonton) + `admin_pin` (kamu, bisa arm/disarm/calibrate) di config.
5. "Add to Home Screen" di Safari/Chrome HP → jadi kayak app.

---

## Platform lain

| Platform | Backend | Cara |
|---|---|---|
| **Linux** (terbaik) | `radiotap` (monitor, root), `iw`, `proc` | di atas |
| **Windows** | `netsh` (signal %, ~1 Hz) | `python main.py --source netsh` |
| **macOS** | `airport -I` (kalau utility masih ada) | `python main.py --source airport` |
| **Android (Termux)** | `termux-wifi-connectioninfo` | `pkg install termux-api` + app Termux:API + `termux-wake-lock`; `python main.py --source termux` |

Catatan jujur: backend selain radiotap cuma lihat 1 link (link HP/laptop ke AP),
~1 Hz, dan sensitivitasnya "orang lewat" — bukan yang halus-halus. Monitor mode
radiotap di Linux yang paling kaya.

---

## CSI (opsional, butuh hardware khusus)

CSI = per-subcarrier amplitude + phase. Ini satu-satunya mode yang bisa lihat
halus (breathing). Tapi butuh NIC/driver yang di-patch:

| Toolchain | Hardware | Cara aktif |
|---|---|---|
| nexmon_csi | Broadcom BCM43xx (beberapa router/Android rooted) | `sudo python main.py --mode csi` |
| Intel 5300 CSI Tool | kartu Intel 5300 + kernel custom | rekam .dat → `--source csi_pcap --csi-tool iwl5300 --csi-pcap file.dat` |
| ath9k CSI | Atheros AR9xxx + driver patch | rekam → `--csi-tool atheros --csi-pcap file.pcap` |

Kalau hardware gak ada → program print error jelas + fallback RSSI otomatis
(atau exit kalau `--strict`). Gak pernah ada data CSI palsu. Install `csiread`
untuk parse pcap offline: `pip install csiread`.

---

## Tuning & FAQ

**RSSI naik-turun sendiri padahal ruangan kosong?** Kalibrasi ulang, atau naikin
`detection.presence_k` (default 3.0 → 3.5-4). Jangan pindah router.

**Sensitif banget (sering false positive)?** `presence_k` naikin, `motion_k`
naikin, `min_confidence` alarm naikin.

**Kurang sensitif (orang lewat gak ke-deteksi)?** Turunin `presence_k`/`motion_k`
ke 2.5-2.8. Mode `radiotap` jauh lebih sensitif daripada `iw`.

**Gak ada interface wireless?** VM/container biasanya gak bisa lihat NIC wireless
— jalankan di host. Cek `main.py --capabilities`.

**Dashboard gak kebuka dari HP?** Firewall: `sudo ufw allow 8000/tcp`. Pastikan
satu WiFi (atau set `dashboard.host: 0.0.0.0` sudah default). Hotspot HP sendiri
juga bisa asal laptop connect ke situ.

**RAM?** ~190 MB untuk seluruh stack (numpy+scipy+pandas+fastapi) yang kami ukur.
Tanpa dashboard `--no-dashboard` lebih kecil.

**Gak mau data ke-log?** `--no-log`. File ada di `data/` (samples/windows/events
CSV + session JSON) untuk analisis/training.

---

## Troubleshooting

| Gejala | Penyebab → Solusi |
|---|---|
| `iw: command not found` | `sudo apt install iw` |
| "requires root" untuk radiotap | `sudo .venv/bin/python main.py --source radiotap` |
| `tshark not found` | `sudo apt install tshark` |
| "not associated with any AP" | laptop belum connect WiFi → `nmcli device wifi connect ...` |
| Monitor mode gagal | NetworkManager pegang interface: `sudo nmcli dev set wlan0 managed no`, lalu coba lagi (di-restore otomatis saat exit) |
| level `/proc` tidak reliable | driver nulis nilai konstan → pakai `--source iw` |
| Dashboard tidak muncul | cek `sudo ufw status`, `ss -tlnp \| grep 8000` |
| CSI "not supported" | memang butuh hardware khusus; pakai `--mode rssi` |
| Alarm tidak notif | cek `alarm.notify.telegram.enabled`, token/chat_id, internet |
| Loop restart terus | `journalctl -u wifisense -f` untuk alasan; cek AP/NIC |

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q     # 66 tests
```

Test mencakup: parser tiap backend, filter (median/hampel/gate), fitur
(variance/tone recovery/entropy), baseline+threshold adaptif, engine
(hysteresis/hold/fallback), alarm state machine, API dashboard + websocket,
dan pipeline end-to-end dengan sumber synthetic yang **eksplisit ditandai** DEMO.

## Lisensi & etika

Sensing ini hanya memberi tahu "ada manusia / ada gerakan" — tidak pernah
identitas, wajah, isi percakapan, atau data pribadi. Data mentah (RSSI/CSI)
adalah milikmu, tersimpan lokal di `data/`. Jangan pakai untuk mengawasi orang
tanpa izin mereka; aturan ini sejalan dengan privasi rumah tangga sendiri.
