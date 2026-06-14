# Smart Eye Firmware

Assistive vision system for visually impaired users, built on the Radxa CM5 Lite (RK3582)
with a custom carrier board. Features real-time object detection (currency, potholes,
stairs), OCR with text-to-speech, ultrasonic obstacle sensing, and audio/haptic
feedback — all running fully offline.

## Hardware Platform

| Component             | Part / Module                          | Interface                          |
|-----------------------|----------------------------------------|------------------------------------|
| SoM                   | Radxa CM5 Lite (RK3582)               | —                                  |
| Camera                | IMX219 (RPi Camera v2)                | CSI (MIPI), /dev/video11           |
| Ultrasonic (front)    | AJ-SR04M (Mode 4)                     | UART3-M2 (/dev/ttyS3)             |
| Ultrasonic (down)     | AJ-SR04M (Mode 4)                     | UART6-M2 (/dev/ttyS6)             |
| Speaker amplifier     | MAX98357A (U10)                       | I2S1-M0, GPIO4_A3 SD_MODE (131)   |
| Battery charger       | BQ25895RTWR (U2)                      | I2C3 @ 0x6A                       |
| Fuel gauge            | BQ27220YZFR (U20)                     | I2C3 @ 0x55                       |
| Battery protection    | BQ29700DSER (U15)                     | Hardware only                      |
| Battery               | 1S 10000 mAh Li-ion                   | J5                                 |
| Vibration motor       | ERM via N-FET                         | PWM4-M0 (GPIO0_C5)                |
| Buttons               | LANG_BTN (GPIO0_D0), OCR_BTN (GPIO0_C7) | gpio-keys                        |
| Headphone output      | ES8316 codec (I2C8 @ 0x11)            | I2C8-M2, I2S0 (card 2)            |

> **RK3582 / RKNN note:** The RK3582 is a binned RK3588S with the same NPU architecture.
> RKNN models must be built with `target_platform='rk3588'`; on the board they are loaded
> via RKNNLite which auto-detects the hardware. The app normalises `rk3582` → `rk3588`
> automatically — no manual flag is required.

## Audio System

Smart Eye has **two independent audio outputs:**

### Speaker Output (MAX98357A Amplifier) — primary
- **Interface:** I2S1-M0 (GPIO4_A1/A2/B2 for SCLK/LRCK/SDO1)
- **ALSA Card:** `card 3: SmartEyeAudio` (device name `smarteye_loud`)
- **AMP_SD control:** GPIO4_A3 = gpio131 via R24 (300 Ω) — **HIGH = amp ON, LOW = amp OFF**
  (direct logic, no BSS138 inverter)
- **Volume:** softvol limited to 90 % at startup to prevent overdriving the 15 dB HW gain
- **Test:**
  ```bash
  sudo bash -c 'echo out > /sys/class/gpio/gpio131/direction; echo 1 > /sys/class/gpio/gpio131/value'
  aplay -D plughw:SmartEyeAudio,0 ~/Smart_eye_firmware/wav/English/battery_shutdown.wav
  sudo bash -c 'echo 0 > /sys/class/gpio/gpio131/value'
  ```

### Headphone Output (ES8316 Codec) — secondary
- **Interface:** I2C8-M2 (GPIO1_D6/D7 @ 0x11), I2S0 (fe470000)
- **ALSA Card:** `card 2: rockchipes8316` (use `plughw:` to auto-convert mono→stereo)
- **Jack detection:** SARADC channel 3 (`in_voltage3_raw`) — threshold 2800 raw counts;
  switched automatically by `HeadphoneDetector` in the app
- **AMP_SD:** kept LOW during headphone playback — the app never enables the MAX98357A
  while playing through the ES8316 (prevents DC on the speaker output)
- **Test:**
  ```bash
  aplay -D plughw:rockchipes8316,0 ~/Smart_eye_firmware/wav/English/battery_shutdown.wav
  # Check raw SARADC (unplugged ≈ 3137, plugged ≈ 2460)
  cat /sys/bus/iio/devices/iio:device0/in_voltage3_raw
  ```

**Headphone detection — SARADC VIN3 mic-line divider:**

The carrier uses the same design as the ROCK 5C — no mechanical detect switch, no ES8316
GPIO path. Detection uses a voltage divider on the mic contact:

```
MICBIAS1 → R35 (2.2 kΩ) → MIC_IN2P (jack pin 1)
         → R36 (10 kΩ) → SARADC_VIN3 → R37 (10 kΩ) → GND
```

| State     | Raw ADC | Voltage (1.8 V ref, 12-bit) |
|-----------|---------|-----------------------------|
| Unplugged | ~3137   | ~1.38 V (MICBIAS float)      |
| Plugged   | ~2460   | ~1.08 V (plug loads mic contact) |
| Threshold | 2800    | midpoint                     |

`HeadphoneDetector` auto-discovers the IIO device (scans `/sys/bus/iio/devices/` for
`name = saradc`), polls every 250 ms with a 2-read debounce, and fires `audio.set_output()`
on state change. MICBIAS must be powered — ES8316 keeps it on whenever the codec is active.

> **Why not GPIO / ES8316 reg 0x4D?** The jack detect switch on the PJ-342 is shorted to
> HPOL on the carrier PCB, defeating mechanical detection. R33/NC isolates the jack switch
> from ES8316 GPIO1 on all Radxa designs — reg 0x4D has nothing to detect.

**Critical hardware notes:**
- ES8316 requires MCLK from I2S0 at probe time — the base DTB provides this via `es8316-sound`.
- I2S1 uses `i2s-tx-route = <1 0 2 3>` to route audio to SDO1 (GPIO4_B2 → MAX98357A DIN).
  `i2s-lrck-gpio` must be null so LRCK is generated as an I2S signal.
- `sdi0` (GPIO4_A5) and `sdo0` (GPIO4_B0) are excluded from I2S1 pinctrl to avoid conflicts
  with UART3-M2 and I2C6-M3 respectively.

## Features

- **Object Detection**: YOLOv8 PathPal model on RKNN NPU — detects Indian currency
  denominations (₹1–₹2000), potholes, and stairs
- **OCR**: RapidOCR (onnxruntime) with text-to-speech via Piper (English & Hindi)
- **Translation**: Argostranslate English ↔ Hindi (offline)
- **Ultrasonic Sensors**: Front sensor (60 cm threshold) and downward sensor (130 cm threshold)
  with 200 ms inter-sensor delay to prevent acoustic crosstalk
- **Audio Feedback**: Pre-recorded WAV announcements + Piper TTS for OCR results;
  auto-switches between headphone (ES8316) and speaker (MAX98357A) on jack insert/remove
- **Haptic Feedback**: PWM vibration motor (GPIO0_C5) with GPIO on/off fallback
- **Language Switch**: Hardware slide switch (LANG_BTN) — state read at startup via
  EVIOCGKEY ioctl, live toggle during operation
- **Battery Monitoring**: SOC, voltage, current, temperature, time-to-empty (BQ27220)
- **Power Management**: CPU governor tuning, GPU/NPU idle, unused service shutdown
- **Camera Rotation**: 90° CCW rotation applied in background capture thread
- **Fully Offline**: No internet required for any functionality
- **Auto-Start**: systemd service launches on boot

## Project Structure

```
Smart_eye_firmware/
├── README.md
├── CLAUDE.md                              # Claude Code guidance (architecture notes)
├── requirements.txt
├── power_config.py                        # Power system setup (charger + gauge + daemon)
├── setup_tts.sh                           # TTS engine & voice model installer
├── smart-eye-start.sh                     # Service control: start
├── smart-eye-stop.sh                      # Service control: stop
├── smart-eye-restart.sh                   # Service control: restart
│
├── Overlays/                              # Device tree overlay (build here)
│   ├── smart-eye-carrier.dts             # Source
│   ├── smart-eye-carrier.dtbo            # Compiled overlay
│   ├── Makefile                          # make / sudo make install / sudo make enable
│   └── BASE_DTB_PATCH.md                 # Patch notes for the base DTB
│
├── pathpal_project/                       # Main application
│   ├── main.py                           # Entry point (SmartEyeApp)
│   ├── yolov8.py                         # YOLOv8 inference + post-processing
│   ├── convert.py                        # ONNX → RKNN converter (simple)
│   ├── convert_yolov8.py                 # ONNX/PT → RKNN converter (YOLOv8 + rk3582 support)
│   └── py_utils/
│       ├── rknn_executor.py              # RKNNLite model loader (on-device)
│       ├── coco_utils.py                 # Letterbox + COCO helpers
│       └── onnx_executor.py              # ONNX fallback loader
│
├── models/
│   ├── pathpal/                          # Active inference model
│   │   ├── model_v2_large.rknn          # PathPal YOLOv8 (currency + pothole + stairs)
│   │   └── labels.txt                   # 12 class labels (overrides COCO list at runtime)
│   └── yolov8/                          # Generic COCO model (not used in production)
│
├── piper/                                # Piper TTS engine
│   ├── piper/piper                      # Binary
│   ├── en_US-amy-medium.onnx            # English voice model
│   └── hi_IN-pratham-medium.onnx        # Hindi voice model
│
├── tests/
│   ├── test_all.py                      # Run all hardware tests
│   ├── test_camera.py                   # CSI camera pipeline test
│   ├── test_camera_stream.py            # Live MJPEG stream over HTTP
│   ├── test_audio.py                    # Speaker test
│   ├── test_pwm.py                      # Vibration motor test
│   ├── test_buttons.py                  # GPIO button test
│   ├── test_uart6.py                    # UART serial test
│   ├── battery_test.py                  # Charger + fuel gauge status
│   ├── config_fuel_gauge.py             # BQ27220 data memory programmer
│   └── test_overlay.sh                  # DT overlay verification
│
└── wav/                                  # Audio feedback files
    ├── English/                         # English prompts
    └── Hindi/                           # Hindi prompts
```

## Quick Start

### 1. Install the Device Tree Overlay

```bash
cd ~/Smart_eye_firmware/Overlays
make
sudo make install
sudo make enable
sudo make status      # verify pin map
sudo reboot
```

After reboot:
```bash
dmesg | grep -i 'smart-eye\|i2s\|es8316'   # check for overlay load errors
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y cmake python3.11-venv python3-dev
sudo apt install -y pkg-config libcairo2-dev v4l-utils i2c-tools
```

### 3. Configure the Power System

Run once after assembly:
```bash
sudo python3 ~/Smart_eye_firmware/power_config.py
```

### 4. Install Python Dependencies

```bash
cd ~/Smart_eye_firmware
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Install TTS Models

```bash
bash ~/Smart_eye_firmware/setup_tts.sh
```

### 6. Install Translation Packages (one-time, needs internet)

```bash
source venv/bin/activate
argospm install translate-en_hi
argospm install translate-hi_en
```

### 7. Run the Application

**Manual (must be root to access GPIO, I2C, /dev/input):**
```bash
cd ~/Smart_eye_firmware
sudo -E venv/bin/python3 pathpal_project/main.py
```

**Via systemd service (auto-starts on boot):**
```bash
./smart-eye-start.sh       # Start
./smart-eye-stop.sh        # Stop
./smart-eye-restart.sh     # Restart

sudo systemctl status smart-eye
sudo journalctl -u smart-eye -f    # Live logs
```

## Detection Classes

The PathPal model (`models/pathpal/model_v2_large.rknn`) detects 12 classes:

| # | Class               | Audio Alert      |
|---|---------------------|------------------|
| 0 | fifty rupees        | ✓                |
| 1 | five hundred rupees | ✓                |
| 2 | five rupees         | ✓                |
| 3 | hundred rupees      | ✓                |
| 4 | one rupees          | ✓                |
| 5 | ten rupees          | ✓                |
| 6 | twenty rupees       | ✓                |
| 7 | two hundred rupees  | ✓                |
| 8 | two rupees          | ✓                |
| 9 | two thousand rupees | ✓                |
| 10| pothole             | ✓ + vibration    |
| 11| stairs              | ✓ + vibration    |

Class labels are loaded at runtime from `models/pathpal/labels.txt` and override the
built-in 80-class COCO list in `yolov8.py`.

## Ultrasonic Sensors

| Sensor   | Port        | Direction | Alert threshold | Alert             |
|----------|-------------|-----------|-----------------|-------------------|
| US-front | /dev/ttyS3  | Forward   | 60 cm           | Vibration buzz    |
| US-down  | /dev/ttyS6  | Downward  | 130 cm          | Vibration pulse   |

Protocol: AJ-SR04M Mode 4 — send `0x01`, receive `0xFF H L SUM`, distance = `(H<<8|L)` mm.  
200 ms inter-sensor delay between readings to prevent acoustic crosstalk.  
Both sensors reconnect automatically if powered on after boot (no-response threshold: 20 reads).

## Camera

- Device: `/dev/video11` (rkisp_mainpath, IMX219)
- Capture: 640×480 @ 10 FPS (background thread, 1-frame buffer)
- Rotation: 90° counter-clockwise applied in the capture thread
- All consumers (detection, OCR) receive pre-rotated frames

### Live Camera Stream (for testing)

```bash
sudo -E venv/bin/python3 tests/test_camera_stream.py
```

Open `http://<board-ip>:8080` in a browser to view the MJPEG stream.
Endpoints: `/` (web UI), `/stream` (raw MJPEG), `/snapshot` (single JPEG).

## Language Switching

- **Hardware**: LANG_BTN slide switch (GPIO0_D0, active-low via gpio-keys, keycode 0x100)
- **Startup**: Physical switch position read via EVIOCGKEY ioctl, then GPIO sysfs fallback
  - Switch active (GPIO low, value=1) → English
  - Switch released (GPIO high, value=0) → Hindi
- **Runtime**: Toggle detected via input event polling thread
- **OCR**: Recognized text spoken via Piper TTS in current language
- **Translation**: English OCR text auto-translated to Hindi when in Hindi mode

## Power Management

```
┌──────────────────────────────────────────────┐
│  Layer 4: BQ29700 Hardware Protection (U15)  │  Cell UV/OV disconnect
├──────────────────────────────────────────────┤
│  Layer 3: BQ25895 Ship Mode (BATFET)         │  Zero-drain when off
├──────────────────────────────────────────────┤
│  Layer 2: battery_daemon.py (systemd)        │  SOC/voltage/temp monitor
├──────────────────────────────────────────────┤
│  Layer 1: BQ25895 Charger IC Limits          │  HW charge safety
└──────────────────────────────────────────────┘
```

Runtime power optimizations applied at startup by `_apply_power_profile()`:
- Little cores (cpu0–3): `conservative` governor
- Big cores (cpu4–7): `powersave` governor
- GPU: minimum frequency (`userspace` governor)
- NPU: `rknpu_ondemand` governor
- Unused services stopped: cups, avahi-daemon, wpa_supplicant, bluetooth, gdm, ModemManager
- HDMI/DP outputs disabled

### Battery Status

```bash
sudo python3 ~/Smart_eye_firmware/power_config.py --status
systemctl status battery-mgr.service
tail -f /var/log/battery-mgr.log
```

BQ27220 alert thresholds (in `SmartEyeApp`):
- SOC ≤ 20 %: spoken battery warning (`battery_10` audio)
- SOC ≤ 10 %: shutdown warning (`battery_shutdown` audio)

## Overlay Installation

The custom carrier board requires a device tree overlay to configure all peripherals:

```bash
cd ~/Smart_eye_firmware/Overlays

# Compile and install
make
sudo make install
sudo make enable

# Verify pin assignments
sudo make status
sudo reboot
```

After reboot:
```bash
ls /boot/dtbo/smart-eye-carrier.dtbo         # Should exist
dmesg | grep -i 'smart-eye\|i2s\|es8316'    # Check for load errors
```

## Model Conversion (RKNN)

Models must be converted on a host with `rknn-toolkit2` installed (not on the board).
Both converters accept `rk3582` on the command line and normalise it to `rk3588`
(the RK3582 shares the same NPU ISA).

```bash
# Simple ONNX → RKNN (float32)
python3 pathpal_project/convert.py model.onnx rk3582 fp model.rknn

# YOLOv8 with int8 quantization (requires dataset.txt with calibration image paths)
python3 pathpal_project/convert_yolov8.py model.onnx rk3582 i8 model_int8.rknn

# From a PyTorch .pt file (requires ultralytics installed on the host)
python3 pathpal_project/convert_yolov8.py yolov8n.pt rk3582 fp yolov8n.rknn
```

`dataset.txt` for quantization must list one representative image path per line
(camera frames from the device work best, ~20+ images). It is **not** the class-label
file (`labels.txt`).

## Hardware Tests

```bash
cd ~/Smart_eye_firmware

# Audio — verify both outputs
aplay -l                                                    # list sound cards
aplay -D plughw:rockchipes8316,0 wav/English/battery_shutdown.wav   # headphone
sudo bash -c 'echo out > /sys/class/gpio/gpio131/direction; echo 1 > /sys/class/gpio/gpio131/value'
aplay -D plughw:SmartEyeAudio,0 wav/English/battery_shutdown.wav    # speaker
sudo bash -c 'echo 0 > /sys/class/gpio/gpio131/value'               # amp off
cat /sys/bus/iio/devices/iio:device0/in_voltage3_raw        # HP detect ADC (unplugged≈3137, plugged≈2460)

# Full hardware test suite
sudo python3 tests/test_all.py
sudo -E venv/bin/python3 tests/test_camera_stream.py   # MJPEG at http://<ip>:8080
sudo python3 tests/test_audio.py
sudo python3 tests/test_pwm.py
sudo python3 tests/battery_test.py
sudo python3 tests/test_buttons.py
sudo python3 tests/test_uart6.py
sudo bash tests/test_overlay.sh
```

## CLI Options

```
usage: main.py [-h] [--model MODEL] [--platform PLATFORM] [--camera CAMERA]
               [--threshold THRESHOLD] [--labels LABELS] [--fps FPS]
               [--us-front-device DEV] [--us-down-device DEV]

options:
  --model MODEL           Path to RKNN model file
                          (default: models/pathpal/model_v2_large.rknn)
  --platform PLATFORM     Target SoC: rk3582, rk3588, rk3588s, rk3566, rk3568, rk3562
                          rk3582/rk3588s are normalised to rk3588 for RKNNLite
                          (default: rk3582)
  --camera CAMERA         V4L2 camera device (default: /dev/video11)
  --threshold THRESHOLD   Detection confidence threshold (default: 0.65)
  --labels LABELS         Class labels file (default: models/pathpal/labels.txt)
  --fps FPS               Target detection FPS — lower = less power (default: 5)
  --us-front-device DEV   Front ultrasonic serial port (default: /dev/ttyS3)
                          Pass 'none' to disable
  --us-down-device DEV    Bottom ultrasonic serial port (default: /dev/ttyS6)
                          Pass 'none' to disable
```

## Dependencies

**System packages:** `v4l-utils`, `i2c-tools`, `espeak-ng` (TTS fallback), `alsa-utils`

**Python packages:** see `requirements.txt` — OpenCV, NumPy, pyserial,
rknn-toolkit-lite2, rapidocr-onnxruntime, argostranslate, etc.

**TTS:** Piper (local binary + ONNX voice models installed by `setup_tts.sh`),
espeak-ng (fallback if Piper models are not present)
