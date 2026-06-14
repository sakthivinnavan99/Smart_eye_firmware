# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Firmware for the **Smart Eye** assistive-vision device: a Radxa CM5 Lite (RK3582) on a
custom carrier board that helps visually-impaired users via real-time object detection
(Indian currency, potholes, stairs), OCR + text-to-speech, ultrasonic obstacle sensing,
and audio/haptic feedback. Everything runs **fully offline** on the device.

**This code targets ARM64 Linux hardware and cannot run on the development machine
(macOS).** There is no local build/run loop here — changes are deployed to the board and
exercised there. The board-side workflow (venv, systemd, hardware tests) lives in
`README.md`; reproduce it on the device, not locally. On macOS you can only edit, read,
lint by eye, and commit.

## Architecture

The entire application is one file: `pathpal_project/main.py`. It is structured as a set
of **hardware-abstraction classes**, each wrapping one peripheral through raw Linux
sysfs/ioctl/subprocess calls (no heavy device libraries), plus one orchestrator:

- `VibrationMotor` — PWM via sysfs (`/sys/class/pwm`) with a GPIO on/off fallback. It
  deliberately *unexports* GPIO 21 (GPIO0_C5, shared with PWM4-M0) first so the PWM
  pinctrl can claim the pin.
- `ButtonListener` — reads gpio-keys events from `/dev/input/eventN`; uses the
  `EVIOCGKEY` ioctl to read the **current** switch position at startup (not just edges).
  Keycodes: `0x100` = LANG_BTN (slide switch, en/hi), `0x101` = OCR_BTN.
- `AudioPlayer` — queues WAV playback through `aplay`; routes output to either
  headphone (`plughw:rockchipes8316,0`) or speaker (`smarteye_loud` softvol device).
  AMP_SD (GPIO4_A3=131) is only raised HIGH when playing through the speaker — it stays
  LOW for headphone playback to prevent DC on the MAX98357A during BCLK-without-LRCK
  states. `set_output(headphone: bool)` switches the active output; the change takes
  effect on the next queued file, never mid-playback.
  **Audio hardware:** Headphone output via ES8316 codec on I2C8-M2 (GPIO1_D6/D7, I2S0);
  speaker output via MAX98357A on I2S1-M0 (GPIO4_A1/A2/B2 for SCLK/LRCK/SDO1).
- `HeadphoneDetector` — detects headphone insertion via the SARADC VIN3 mic-line
  voltage divider (MICBIAS1 → R35 2.2K → MIC_IN2P → R36 10K → SARADC_VIN3 → R37 10K →
  GND). Polls IIO sysfs (`in_voltage3_raw`) every 250 ms with a 2-read debounce.
  Threshold = 2800 raw counts (12-bit ADC, 1.8 V ref): below = inserted (~2460 measured);
  above = removed (~3137 measured). Fires a callback (`audio.set_output`) on state change.
  MICBIAS must be powered (ES8316 always keeps it on while the codec is active).
  **Not** based on GPIO or ES8316 register 0x4D — those paths are dead on this carrier
  (R33 NC isolates the jack switch from ES8316 GPIO1; HP_DET_L dead-ends on the PCB).
- `BatteryGauge` — BQ27220 fuel gauge over `/dev/i2c-3` raw read/write. SOC is at register
  `0x2C` (not the usual `0x28`); capacity is estimated from a hardcoded `DESIGN_CAPACITY_MAH`.
- `UltrasonicSensor` — AJ-SR04M over UART (Mode 4: write `0x01`, read `0xFF H L SUM`).
  Self-heals: reconnects if the port appears late and closes the port after
  `FAIL_RESET_THRESHOLD` no-responses. Pass `"none"`/`"off"` as the device to disable a sensor.
- `CameraDetector` — V4L2 capture on a background thread that rotates every frame **90° CCW**
  before storing it, so detection and OCR both get correctly-oriented frames. Runs YOLOv8
  RKNN inference via the `yolov8`/`py_utils` modules. The `target` parameter is passed from
  `args.platform` via `_normalize_platform()` — for the RK3582 this resolves to `'rk3588'`
  before reaching `setup_model`.
- `OCREngine` — RapidOCR (onnxruntime backend; no PaddlePaddle). One instance for all langs.
- `Translator` — argostranslate en↔hi, offline; language packs must be pre-installed.
- `SmartEyeApp` — wires everything together and runs the main loop: grab frame → detect →
  announce, plus periodic ultrasonic checks, a battery-monitor thread, and OCR triggered by
  the button. Tunable interval/threshold/cooldown constants are class attributes at the top.

**Threading model:** camera capture, audio playback, button polling, battery monitoring,
headphone detection, and each OCR run are separate daemon threads. The main thread only
does the detect loop.

**Inference pipeline:** `pathpal_project/yolov8.py` (`setup_model`, `post_process`, letterbox
preprocessing) backed by `py_utils/rknn_executor.py` (RKNNLite on the NPU) with
`onnx_executor.py` / `pytorch_executor.py` as alternate backends. `yolov8.py`'s `CLASSES`
constant is the 80-class COCO list — it is **overridden at runtime** by `--labels`
(`models/pathpal/labels.txt`, 12 classes). Don't rely on the COCO list for the real classes.

**RK3582 / platform normalization:** `_normalize_platform()` at module level maps
`'rk3582'` and `'rk3588s'` → `'rk3588'`. The RK3582 is a binned RK3588S with the same
NPU ISA; RKNNLite loads `rk3588` models on it transparently. The `--platform` CLI arg
defaults to `'rk3582'` and is normalised before being passed to `CameraDetector`.
`convert.py` and `convert_yolov8.py` apply the same alias so the converters also accept
`rk3582` on the command line.

## Audio Configuration (ES8316 Headphone + MAX98357A Speaker)

The Smart Eye has **two independent audio outputs:**
- **Headphone (ES8316 codec):** I2C8-M2 (GPIO1_D6/D7 SCL/SDA), I2S0 (fe470000), card 2
- **Speaker (MAX98357A amp):** I2S1-M0 (GPIO4_A1/A2/B2 for SCLK/LRCK/SDO1), card 3

Both cards coexist and can be tested with:
```bash
# Headphone: mono files auto-convert to stereo via plughw
aplay -D plughw:rockchipes8316,0 ~/Smart_eye_firmware/wav/English/battery_shutdown.wav

# Speaker: enable GPIO4_A3 (AMP_SD HIGH=ON) then play
sudo bash -c 'echo out > /sys/class/gpio/gpio131/direction; echo 1 > /sys/class/gpio/gpio131/value'
aplay -D plughw:SmartEyeAudio,0 ~/Smart_eye_firmware/wav/English/battery_shutdown.wav
sudo bash -c 'echo 0 > /sys/class/gpio/gpio131/value'
```

**Critical hardware details:**
- ES8316 requires MCLK from I2S0 at probe time — without it the chip NAKs all I2C transactions.
  Base DTB enables this via `es8316-sound` (rockchip,multicodecs-card) linking I2S0→ES8316.
- I2S1 speaker output uses `i2s-tx-route = <1 0 2 3>` to route stereo to SDO1 (GPIO4_B2 → MAX98357A DIN).
  The `i2s-lrck-gpio` property must be **disabled** (set to null) so LRCK is generated as an I2S signal,
  not monitored as GPIO — this is handled in the overlay as `i2s-lrck-gpio = <0x00>`.
- I2S1 pinctrl uses only 3 groups (sclk/lrck/sdo1) — sdi0 (GPIO4_A5) and sdo0 (GPIO4_B0) are
  excluded: sdi0 conflicts with UART3-M2 TX/RX, sdo0 conflicts with I2C6-M3. MAX98357A only
  needs output pins.
- AMP_SD (GPIO4_A3=131): **HIGH = amp ON, LOW = amp OFF** (direct via R24 300K, no BSS138 inverter).
  The app controls this via sysfs to prevent DC on speaker outputs during BCLK-without-LRCK states.

## Things that will trip you up

- **The app must run as root** (`sudo -E venv/bin/python3 ...`) — it writes governors,
  exports GPIOs, opens `/dev/i2c-*`, stops systemd services. Use `-E` to keep the venv.
- **`main.py` is the source of truth for all defaults.** `parse_args()` defaults:
  model = `model_v2_large.rknn`, threshold = `0.65`, platform = `rk3582`,
  us-front-device = `/dev/ttyS3` (UART3-M2), us-down-device = `/dev/ttyS6` (UART6-M2).
  `SmartEyeApp` class-level thresholds: `US_FRONT_THRESHOLD_CM = 60`,
  `US_DOWN_THRESHOLD_CM = 130`, `US_INTER_SENSOR_DELAY = 0.2 s`.
- **Two overlay directories exist:** `Overlays/` (capital, with the `Makefile` and the
  `smart-eye-carrier.*` files that the build uses) and `overlays/` (lowercase).
  On case-insensitive macOS these may collide — check `git status` before assuming which
  file you edited. Build/install the device tree from the `Overlays/` Makefile:
  `make`, `sudo make install`, `sudo make enable`, then reboot. `make status` summarizes
  the pin map.
- **Audio assets are keyed maps:** `AUDIO_FILES` in `main.py` maps logical keys → WAV paths
  under `wav/English` and `wav/Hindi`. A detection label must exactly match a `labels.txt`
  entry and an `AUDIO_FILES` key to get a pre-recorded announcement; otherwise it falls
  back to Piper TTS.
- **TTS** shells out to the bundled `piper/piper/piper` binary with per-language ONNX voice
  models, falling back to `espeak-ng`. Models are installed by `setup_tts.sh`.
- **RKNN model conversion:** use `convert.py` or `convert_yolov8.py` on a host with
  `rknn-toolkit2`. Both scripts accept `rk3582` on the CLI. The quantization `dataset.txt`
  must list **image paths**, not class labels — confusing it with `labels.txt` produces a
  broken or poorly-calibrated model.

## Commands (run on the device)

```bash
# Run the app manually (root + venv)
sudo -E venv/bin/python3 pathpal_project/main.py

# Service control
./smart-eye-start.sh / ./smart-eye-stop.sh / ./smart-eye-restart.sh
sudo journalctl -u smart-eye -f          # live logs

# Hardware bring-up tests (each targets one peripheral)
sudo python3 tests/test_all.py           # everything
sudo -E venv/bin/python3 tests/test_camera_stream.py   # MJPEG at http://<ip>:8080
sudo python3 tests/battery_test.py

# Device tree overlay (from Overlays/)
make && sudo make install && sudo make enable && sudo reboot

# Convert ONNX -> RKNN (on a host with rknn-toolkit2; rk3582 is accepted)
python3 pathpal_project/convert.py model.onnx rk3582 i8 out.rknn
python3 pathpal_project/convert_yolov8.py model.onnx rk3582 fp out.rknn
```

There is no test runner or linter configured; `tests/` are standalone hardware scripts,
not a unit suite.
