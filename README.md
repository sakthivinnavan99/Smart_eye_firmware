# Smart Eye Firmware

Assistive vision system for visually impaired users, built on the Radxa CM5 (RK3588S)
with a custom carrier board. Features real-time object detection (currency, potholes,
stairs), OCR with text-to-speech, ultrasonic obstacle sensing, and audio/haptic
feedback — all running fully offline.

## Hardware Platform

| Component             | Part / Module                     | Interface              |
|-----------------------|-----------------------------------|------------------------|
| SoM                   | Radxa CM5 (RK3588S)              | —                      |
| Camera                | IMX219 (RPi Camera v2)           | CSI (MIPI), /dev/video11 |
| Ultrasonic (front)    | AJ-SR04M (Mode 4)               | UART2-M0 (/dev/ttyS2) |
| Ultrasonic (down)     | AJ-SR04M (Mode 4)               | UART6-M2 (/dev/ttyS6) |
| Speaker amplifier     | MAX98357A (U10)                  | I2S1-M0, GPIO4_A3 SD_MODE |
| Battery charger       | BQ25895RTWR (U2)                 | I2C3 @ 0x6A           |
| Fuel gauge            | BQ27220YZFR (U20)                | I2C3 @ 0x55           |
| Battery protection    | BQ29700DSER (U15)                | Hardware only          |
| Battery               | 1S 10000 mAh Li-ion              | J5                     |
| Vibration motor       | ERM via N-FET                    | PWM7 (GPIO4_B3)       |
| Buttons               | LANG_BTN (GPIO0_D0), OCR_BTN (GPIO0_C7) | gpio-keys      |
| Headphone output      | ES8316 codec (I2C8 @ 0x11)       | I2C8-M2, I2S0 (card 2)  |

## Audio System

Smart Eye has **two independent audio outputs** for different use cases:

### Headphone Output (ES8316 Codec)
- **Interface:** I2C8-M2 (GPIO1_D6/D7 @ 0x11), I2S0 (fe470000)
- **ALSA Card:** `card 2: rockchipes8316`
- **Features:** Jack detection via GPIO, audio routing controls
- **Usage:** OCR results, voice feedback in noisy environments
- **Test:** `aplay -D plughw:rockchipes8316,0 ~/Smart_eye_firmware/wav/English/battery_shutdown.wav`

### Speaker Output (MAX98357A Amplifier)
- **Interface:** I2S1-M0 (GPIO4_A1/A2/B2 for SCLK/LRCK/SDO1), GPIO4_A3 (SD_MODE, inverted)
- **ALSA Card:** `card 3: SmartEyeAudio`
- **Power Control:** GPIO 131 (LOW=amp ON, HIGH=amp OFF) — controlled by AudioPlayer
- **Features:** Class-D amplifier, DC protection via synchronized BCLK/LRCK shutdown
- **Usage:** Primary audio output (louder, more robust in outdoor use)
- **Test:** `aplay -D plughw:SmartEyeAudio,0 ~/Smart_eye_firmware/wav/English/battery_shutdown.wav`

**Note:** Mono WAV files auto-convert to stereo via `plughw:` prefix. For manual testing without the app:
```bash
sudo bash -c 'echo out > /sys/class/gpio/gpio131/direction; echo 0 > /sys/class/gpio/gpio131/value'
aplay -D plughw:SmartEyeAudio,0 file.wav
```

## Features

- **Object Detection**: YOLOv8n PathPal model on RKNN NPU — detects Indian currency
  denominations (₹1–₹2000), potholes, and stairs
- **OCR**: RapidOCR (onnxruntime) with text-to-speech via Piper (English & Hindi)
- **Translation**: Argostranslate English ↔ Hindi (offline)
- **Ultrasonic Sensors**: Forward (180 cm threshold) and downward (135 cm threshold)
  obstacle detection with 50 ms inter-sensor delay to prevent crosstalk
- **Audio Feedback**: Pre-recorded WAV announcements + Piper TTS for OCR results
- **Haptic Feedback**: PWM vibration motor with GPIO fallback
- **Language Switch**: Hardware slide switch (LANG_BTN) — state read at startup via
  EVIOCGKEY ioctl, live toggle during operation
- **Battery Monitoring**: SOC, voltage, current, temperature, time-to-empty
- **Power Management**: CPU governor tuning, GPU/NPU idle, unused service shutdown
- **Camera Rotation**: 90° CCW rotation applied in background capture thread
- **Fully Offline**: No internet required for any functionality
- **Auto-Start**: systemd service launches on boot

## Project Structure

```
Smart_eye_firmware/
├── README.md
├── requirements.txt
├── power_config.py                    # Power system setup (charger + gauge + daemon)
├── setup_tts.sh                       # TTS engine & voice model installer
├── smart-eye-start.sh                 # Service control: start
├── smart-eye-stop.sh                  # Service control: stop
├── smart-eye-restart.sh               # Service control: restart
│
├── overlays/                          # Device tree & kernel config
│   ├── smart-eye-carrier.dts
│   ├── smart-eye-carrier.dtbo
│   └── ...
│
├── Overlays/                          # Compiled overlay backups
│   ├── smart-eye-carrier.dts
│   └── rk3588-smart-eye-carrier.dtbo
│
├── pathpal_project/                   # Main application
│   ├── main.py                        # Entry point (Smart Eye app)
│   ├── yolov8.py                      # YOLOv8 inference + post-processing
│   └── py_utils/
│       ├── rknn_executor.py           # RKNNLite model loader
│       ├── coco_utils.py              # Letterbox + COCO helpers
│       └── onnx_executor.py           # ONNX fallback loader
│
├── models/
│   ├── pathpal/                       # Active model
│   │   ├── yolov8n_2912.rknn         # PathPal YOLOv8n (currency+pothole+stairs)
│   │   └── labels.txt                 # 12 class labels
│   └── yolov8/                        # Generic COCO model (not used)
│
├── piper/                             # Piper TTS engine
│   ├── piper/piper                    # Binary
│   ├── en_US-amy-medium.onnx          # English voice model
│   └── hi_IN-pratham-medium.onnx      # Hindi voice model
│
├── tests/
│   ├── test_all.py                    # Run all hardware tests
│   ├── test_camera.py                 # CSI camera pipeline test
│   ├── test_camera_stream.py          # Live MJPEG stream over HTTP
│   ├── test_audio.py                  # Speaker test
│   ├── test_pwm.py                    # Vibration motor test
│   ├── test_buttons.py                # GPIO button test
│   ├── test_uart6.py                  # UART serial test
│   ├── battery_test.py                # Charger + fuel gauge status
│   ├── config_fuel_gauge.py           # BQ27220 data memory programmer
│   └── test_overlay.sh               # DT overlay verification
│
└── wav/                               # Audio feedback files
    ├── English/                       # English prompts
    └── Hindi/                         # Hindi prompts
```

## Quick Start

### 1. Install the Device Tree Overlay

```bash
cd ~/Smart_eye_firmware/overlays
make
sudo make install
sudo make enable
sudo reboot
```

### 2. Configure the Power System

Run once after assembly:

```bash
sudo python3 ~/Smart_eye_firmware/power_config.py
```

### 3. Install Python Dependencies

```bash
cd ~/Smart_eye_firmware
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Install TTS Models

```bash
bash ~/Smart_eye_firmware/setup_tts.sh
```

### 5. Install Translation Packages (one-time, needs internet)

```bash
source venv/bin/activate
argospm install translate-en_hi
argospm install translate-hi_en
```

### 6. Run the Application

**Manual:**

```bash
cd ~/Smart_eye_firmware
sudo -E venv/bin/python3 pathpal_project/main.py
```

**Via systemd service (auto-starts on boot):**

```bash
./smart-eye-start.sh       # Start
./smart-eye-stop.sh        # Stop
./smart-eye-restart.sh     # Restart

# Or directly:
sudo systemctl start smart-eye
sudo systemctl stop smart-eye
sudo systemctl restart smart-eye
sudo systemctl status smart-eye
```

**View live logs:**

```bash
sudo journalctl -u smart-eye -f
```

## Detection Classes

The PathPal model (`models/pathpal/yolov8n_2912.rknn`) detects 12 classes:

| # | Class               | Audio Alert |
|---|---------------------|-------------|
| 0 | fifty rupees        | ✓           |
| 1 | five hundred rupees | ✓           |
| 2 | five rupees         | ✓           |
| 3 | hundred rupees      | ✓           |
| 4 | one rupees          | ✓           |
| 5 | ten rupees          | ✓           |
| 6 | twenty rupees       | ✓           |
| 7 | two hundred rupees  | ✓           |
| 8 | two rupees          | ✓           |
| 9 | two thousand rupees | ✓           |
| 10| pothole             | ✓ + vibration |
| 11| stairs              | ✓ + vibration |

## Ultrasonic Sensors

| Sensor   | Port       | Direction | Threshold | Alert             |
|----------|------------|-----------|-----------|-------------------|
| US-front | /dev/ttyS2 | Forward   | 180 cm    | Vibration buzz    |
| US-down  | /dev/ttyS6 | Downward  | 135 cm    | Vibration pulse   |

Protocol: AJ-SR04M Mode 4 — send `0x01`, receive `0xFF H L SUM`, distance = `(H<<8|L)` mm.
50 ms delay between sensor readings to prevent crosstalk.

## Camera

- Device: `/dev/video11` (rkisp_mainpath, IMX219)
- Capture: 640×480 @ 10 FPS
- Rotation: 90° counter-clockwise (applied in background capture thread)
- All consumers (detection, OCR) receive pre-rotated frames

### Live Camera Stream (for testing)

```bash
sudo -E venv/bin/python3 tests/test_camera_stream.py
```

Open `http://<board-ip>:8080` in a browser to view the MJPEG stream.
Endpoints: `/` (web UI), `/stream` (raw MJPEG), `/snapshot` (single JPEG).

## Language Switching

- **Hardware**: LANG_BTN slide switch (GPIO0_D0, active-low via gpio-keys)
- **Startup**: Physical switch position read via EVIOCGKEY ioctl
  - Switch active (GPIO low) → English
  - Switch released (GPIO high) → Hindi
- **Runtime**: Toggle detected via input event polling
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

Runtime power optimizations applied at startup:
- Little cores (cpu0-3): conservative governor
- Big cores (cpu4-7): powersave governor
- GPU: minimum frequency (userspace governor)
- NPU: rknpu_ondemand governor
- Unused services stopped (cups, avahi, bluetooth, gdm, etc.)
- HDMI/DP outputs disabled

### Battery Status

```bash
sudo python3 ~/Smart_eye_firmware/power_config.py --status
systemctl status battery-mgr.service
tail -f /var/log/battery-mgr.log
```

## Overlay Installation

The custom carrier board requires a device tree overlay to configure all peripherals:

```bash
cd ~/Smart_eye_firmware/Overlays

# Compile (on host with dtc) or use pre-built smart-eye-carrier.dtbo
make

# Install to boot partition
sudo make install
sudo make enable

# Verify and reboot
sudo make status
sudo reboot
```

After reboot, verify with:
```bash
ls /boot/dtbo/smart-eye-carrier.dtbo         # Should exist
dmesg | grep -i 'smart-eye\|i2s\|es8316'     # Check for load errors
```

## Hardware Tests

```bash
cd ~/Smart_eye_firmware
source venv/bin/activate

# Audio tests (verify both outputs)
aplay -l                                       # List all sound cards
aplay -D plughw:rockchipes8316,0 wav/English/battery_shutdown.wav   # Headphone test
sudo bash -c 'echo out > /sys/class/gpio/gpio131/direction; echo 0 > /sys/class/gpio/gpio131/value'
aplay -D plughw:SmartEyeAudio,0 wav/English/battery_shutdown.wav    # Speaker test

# Full hardware test suite
sudo python3 tests/test_all.py            # All tests
sudo python3 tests/test_camera.py          # Camera capture & recording
sudo -E venv/bin/python3 tests/test_camera_stream.py  # Live MJPEG stream
sudo python3 tests/test_audio.py           # Speaker output test
sudo python3 tests/test_pwm.py            # Vibration motor
sudo python3 tests/battery_test.py         # Charger + fuel gauge
sudo python3 tests/test_buttons.py         # GPIO buttons (LANG_BTN, OCR_BTN)
sudo python3 tests/test_uart6.py           # UART serial (ultrasonic)
sudo bash tests/test_overlay.sh            # DT overlay verification
```

## CLI Options

```
usage: main.py [-h] [--model MODEL] [--camera CAMERA] [--threshold THRESHOLD]
               [--labels LABELS] [--fps FPS]

options:
  --model MODEL         Path to RKNN model (default: models/pathpal/yolov8n_2912.rknn)
  --camera CAMERA       V4L2 camera device (default: /dev/video11)
  --threshold THRESHOLD Detection confidence threshold (default: 0.55)
  --labels LABELS       Class labels file (default: models/pathpal/labels.txt)
  --fps FPS             Target detection FPS (default: 5)
```

## Dependencies

**System packages:** `v4l-utils`, `libmraa2`, `i2c-tools`

**Python packages:** see `requirements.txt` — OpenCV, NumPy, pyserial,
rknn-toolkit-lite2, rapidocr-onnxruntime, argostranslate, etc.

**TTS:** Piper (local binary + ONNX voice models), espeak-ng (fallback)
