# Smart Eye Firmware

Assistive vision system for visually impaired users, built on the Radxa CM5 (RK3588)
with a custom carrier board. Features real-time object detection, OCR, audio feedback,
and a complete battery-powered portable design with power management.

## Hardware Platform

| Component             | Part / Module                     | Interface        |
|-----------------------|-----------------------------------|------------------|
| SoM                   | Radxa CM5 (RK3588S)              | --               |
| Camera                | IMX219 (RPi Camera v2)           | CSI (MIPI)       |
| Ultrasonic sensor     | JSN-SR04T                        | UART6-M2         |
| Speaker amplifier     | MAX98357A                        | I2S1-M0          |
| Battery charger       | BQ25895RTWR (U2)                 | I2C3 @ 0x6A      |
| Fuel gauge            | BQ27220YZFR (U20)                | I2C3 @ 0x55      |
| Battery protection    | BQ29700DSER (U15)                | Hardware only     |
| Battery               | 1S 10000 mAh Li-ion              | J5               |
| Vibration motor       | ERM via N-FET                    | PWM7 (GPIO4_B3)  |
| Buttons               | LANG_BTN, OCR_BTN, PWR (PMIC)   | GPIO / RK806     |

## Project Structure

```
Smart_eye_firmware/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── power_config.py                    # Power system setup (charger + gauge + daemon)
│
├── overlays/                          # Device tree & kernel config
│   ├── smart-eye-carrier.dts          # Current DT overlay source
│   ├── smart-eye-carrier.dtbo         # Compiled overlay
│   ├── rk3588-smart-eye-carrier.dts   # Legacy overlay variant
│   ├── setup_uboot_gpio_fix.sh        # U-Boot vibration motor fix
│   ├── Makefile                       # Build/install automation
│   └── USAGE.md                       # Overlay & power management docs
│
├── pathpal_project/                   # Main application
│   ├── main.py                        # Entry point
│   ├── CameraTest.py                  # Camera streaming & recording
│   ├── LiveVideoDetection.py          # Real-time YOLO detection
│   ├── ObjectDetection.py             # Detection pipeline
│   ├── UltrasonicSensor.py            # Distance measurement (UART)
│   ├── convert_yolov8.py              # ONNX → RKNN model converter
│   └── yolov8.py                      # YOLOv8 inference wrapper
│
├── models/                            # ML models
│   ├── pathpal/                       # PathPal YOLOv8 models
│   └── yolov8/                        # Generic YOLOv8 assets
│
├── tests/                             # Hardware & integration tests
│   ├── test_all.py                    # Run all tests
│   ├── test_camera.py                 # CSI camera pipeline test
│   ├── test_audio.py                  # I2S / MAX98357A speaker test
│   ├── test_pwm.py                    # Vibration motor PWM test
│   ├── battery_test.py                # BQ25895 + BQ27220 status reader
│   ├── config_fuel_gauge.py           # BQ27220 data memory programmer
│   ├── test_buttons.py                # GPIO button input test
│   ├── test_i2c_bq27220.py           # Fuel gauge I2C test
│   ├── test_uart6.py                  # UART6 loopback test
│   └── test_overlay.sh               # Overlay load verification
│
└── wav/                               # Audio feedback files
    ├── English/                       # English voice prompts
    └── Hindi/                         # Hindi voice prompts
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

Run once after assembly to program the charger, fuel gauge, and install
the battery management daemon:

```bash
sudo python3 ~/Smart_eye_firmware/power_config.py
```

This configures:
- **BQ25895**: 2A input limit, 2A charge current, 4.208V regulation, watchdog off
- **BQ27220**: 10000 mAh design capacity, CEDV parameters
- **Daemon**: installs `battery-mgr.service` for continuous monitoring

### 3. Install Python Dependencies

```bash
cd ~/Smart_eye_firmware
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Application

```bash
source venv/bin/activate
python pathpal_project/main.py
```

## Power Management

The system includes a multi-layer battery protection stack:

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

### Power Button

Short press on the RK806 PMIC power key triggers a clean shutdown with
vibration feedback. Configuration is in `/etc/systemd/logind.conf.d/`.

### Shutdown Sequence

1. Vibration motor: 3 short buzzes (tactile confirmation)
2. System gracefully shuts down all services
3. If no USB charger connected: BQ25895 enters ship mode (BATFET off)
4. System fully powers down with near-zero battery drain

### Battery Status

```bash
# Daemon log
tail -f /var/log/battery-mgr.log

# Full status report
sudo python3 ~/Smart_eye_firmware/power_config.py --status

# Daemon health
systemctl status battery-mgr.service
```

See [`overlays/USAGE.md`](overlays/USAGE.md) for detailed power management
documentation, register maps, and troubleshooting.

## Hardware Tests

```bash
cd ~/Smart_eye_firmware

# Run all tests
sudo python3 tests/test_all.py

# Individual tests
sudo python3 tests/test_camera.py       # CSI camera
sudo python3 tests/test_audio.py        # Speaker output
sudo python3 tests/test_pwm.py          # Vibration motor
sudo python3 tests/battery_test.py      # Charger + fuel gauge
sudo python3 tests/test_buttons.py      # GPIO buttons
sudo python3 tests/test_uart6.py        # UART serial
sudo bash tests/test_overlay.sh         # DT overlay verification
```

## Camera

```bash
# Live camera feed (requires display)
python pathpal_project/CameraTest.py

# Headless mode (saves frames to disk)
python pathpal_project/CameraTest.py --headless

# Record video
python pathpal_project/CameraTest.py --record --duration 30
```

## Dependencies

System packages:
- `v4l-utils`, `libmraa2`, `libmraa-dev`, `mraa-tools`
- `i2c-tools` (for power management)

Python packages: see `requirements.txt` (OpenCV, PyTorch, YOLOv8, NumPy, etc.)

## License

[Add your license here]
