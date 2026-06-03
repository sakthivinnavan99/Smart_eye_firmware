#!/usr/bin/env python3
"""
Smart Eye Firmware - Main Application
======================================
Assistive vision system for visually impaired users.

Hardware platform: Radxa CM5 (RK3588S) on Smart Eye carrier board
  - Camera:      IMX219 via V4L2 (/dev/video11)
  - Inference:   YOLOv8n PathPal on RKNN NPU (models/pathpal/yolov8n_2912.rknn)
  - Audio:       MAX98357A on ALSA Smart-Eye-Audio (card 2)
  - Buttons:     LANG_BTN (GPIO0_D0), OCR_BTN (GPIO0_C7) via gpio-keys
  - Vibration:   PWM7 on GPIO4_B3 via sysfs
  - Ultrasonic1: AJ-SR04M on UART3 (/dev/ttyS3) - front facing
  - Ultrasonic2: AJ-SR04M on UART6 (/dev/ttyS6) - bottom facing
  - Battery:     BQ27220 fuel gauge on I2C3 (0x55)
  - OCR:         RapidOCR (onnxruntime)
  - Translation: argostranslate (English <-> Hindi, offline only)
"""

import argparse
import sys
import os
import time
import threading
import queue
import struct
import fcntl
import glob
import subprocess
import traceback
import gc
import logging
import wave

import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("ORT_GLOBAL_THREAD_POOL_SIZE", "4")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("smart-eye")


# ---------------------------------------------------------------------------
#  System-level power optimizations (applied once at startup)
# ---------------------------------------------------------------------------
def _apply_power_profile():
    """Reduce SoC power consumption by tuning governors, disabling unused
    peripherals, and stopping unnecessary services.  Safe to call as root;
    silently skips anything that doesn't exist or fails."""

    def _write(path, val):
        try:
            with open(path, "w") as f:
                f.write(val)
        except OSError:
            pass

    def _read(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return ""

    # -- CPU: switch big cores (cpu4-7) to powersave, keep little cores
    #    on conservative.  All cores stay online because onnxruntime crashes
    #    with a vector OOB if it tries to set thread affinity on offline cores.
    for cpu in range(8):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        if cpu < 4:
            _write(f"{base}/scaling_governor", "conservative")
        else:
            _write(f"{base}/scaling_governor", "powersave")

    # -- GPU: drop to lowest frequency since we don't use the GPU at all
    for gpu_path in glob.glob("/sys/class/devfreq/*gpu*"):
        avail = _read(f"{gpu_path}/available_frequencies")
        if avail:
            lowest = avail.split()[0]
            _write(f"{gpu_path}/governor", "userspace")
            _write(f"{gpu_path}/userspace/set_freq", lowest)

    # -- NPU: use on-demand (already default), but cap idle frequency
    for npu_path in glob.glob("/sys/class/devfreq/*npu*"):
        _write(f"{npu_path}/governor", "rknpu_ondemand")

    # -- Disable unused services that consume CPU/RAM
    _stopped = []
    for svc in [
        "cups.service",
        "packagekit.service",
        "avahi-daemon.service",
        "wpa_supplicant.service",
        "gdm.service",
        "ModemManager.service",
        "bluetooth.service",
    ]:
        ret = subprocess.run(
            ["systemctl", "stop", svc],
            capture_output=True, timeout=5,
        )
        if ret.returncode == 0:
            _stopped.append(svc)

    if _stopped:
        log.info("Stopped unused services: %s", ", ".join(_stopped))

    # -- Disable HDMI output (both ports are disconnected)
    for card in glob.glob("/sys/class/drm/card*-HDMI*"):
        _write(os.path.join(card, "enabled"), "disabled")
    for card in glob.glob("/sys/class/drm/card*-DP*"):
        _write(os.path.join(card, "enabled"), "disabled")

    log.info("Power profile applied (big cores powersave, GPU min, unused services stopped)")


# ---------------------------------------------------------------------------
#  Hardware abstraction: Vibration motor (PWM sysfs with GPIO fallback)
# ---------------------------------------------------------------------------
_VIB_PWM_REG = "febd0030"
_VIB_GPIO = 139                # GPIO4_B3 — shared with PWM7-M1


class VibrationMotor:
    """Control vibration motor on GPIO4_B3.

    Primary mode: PWM via sysfs (variable intensity).
    Fallback mode: GPIO on/off (if PWM chip unavailable).

    The init service or a previous run may hold GPIO 139 via sysfs,
    which prevents the PWM pinctrl from claiming the pin.  We release
    it first so the PWM controller can take over.
    """

    def __init__(self):
        self._mode = None       # "pwm" | "gpio"
        self._pwm_dir = None
        self._gpio_val = None

        self._release_gpio()
        chip = self._find_pwm_chip()
        if chip:
            self._init_pwm(chip)
        if self._mode is None:
            self._init_gpio_fallback()

    # -- GPIO release (so PWM pinctrl can claim the pin) --------------------

    @staticmethod
    def _release_gpio():
        gpath = f"/sys/class/gpio/gpio{_VIB_GPIO}"
        if os.path.exists(gpath):
            try:
                with open("/sys/class/gpio/unexport", "w") as f:
                    f.write(str(_VIB_GPIO))
                time.sleep(0.1)
                log.info("Released GPIO %d for PWM", _VIB_GPIO)
            except OSError:
                pass

    # -- PWM chip discovery -------------------------------------------------

    @staticmethod
    def _find_pwm_chip():
        for p in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
            try:
                link = os.readlink(os.path.join(p, "device"))
                if _VIB_PWM_REG in link:
                    return p
            except OSError:
                pass
            try:
                with open(os.path.join(p, "device", "uevent")) as f:
                    if _VIB_PWM_REG in f.read():
                        return p
            except (OSError, FileNotFoundError):
                pass
        return None

    # -- PWM init -----------------------------------------------------------

    def _init_pwm(self, chip):
        self._pwm_dir = os.path.join(chip, "pwm0")
        if not os.path.isdir(self._pwm_dir):
            try:
                with open(os.path.join(chip, "export"), "w") as f:
                    f.write("0")
                time.sleep(0.1)
            except OSError:
                log.warning("Cannot export PWM channel, falling back to GPIO")
                self._pwm_dir = None
                return
        self._pwm_write("period", "2000000")    # 500 Hz default
        self._pwm_write("duty_cycle", "0")
        self._mode = "pwm"
        log.info("Vibration motor: PWM mode (%s)", chip)

    def _pwm_write(self, attr, val):
        if not self._pwm_dir:
            return
        try:
            with open(os.path.join(self._pwm_dir, attr), "w") as f:
                f.write(val)
        except OSError:
            pass

    def _pwm_read(self, attr):
        if not self._pwm_dir:
            return None
        try:
            with open(os.path.join(self._pwm_dir, attr)) as f:
                return f.read().strip()
        except OSError:
            return None

    # -- GPIO fallback init -------------------------------------------------

    def _init_gpio_fallback(self):
        gpath = f"/sys/class/gpio/gpio{_VIB_GPIO}"
        if not os.path.exists(gpath):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(_VIB_GPIO))
                time.sleep(0.05)
            except OSError:
                log.warning("Vibration motor: neither PWM nor GPIO available")
                return
        try:
            with open(f"{gpath}/direction", "w") as f:
                f.write("out")
        except OSError:
            log.warning("Vibration motor: cannot set GPIO direction")
            return
        self._gpio_val = f"{gpath}/value"
        self._gpio_set(0)
        self._mode = "gpio"
        log.info("Vibration motor: GPIO fallback mode (on/off only)")

    def _gpio_set(self, val):
        if self._gpio_val is None:
            return
        try:
            with open(self._gpio_val, "w") as f:
                f.write(str(val))
        except OSError:
            pass

    # -- Public API (works in both modes) -----------------------------------

    def set_frequency(self, freq_hz):
        if self._mode == "pwm":
            period_ns = int(1e9 / freq_hz)
            self._pwm_write("period", str(period_ns))

    def buzz(self, duration_ms, duty_pct=50):
        if self._mode == "pwm":
            period = int(self._pwm_read("period") or 2000000)
            duty = int(period * max(0, min(100, duty_pct)) / 100)
            self._pwm_write("duty_cycle", str(duty))
            self._pwm_write("enable", "1")
            time.sleep(duration_ms / 1000.0)
            self._pwm_write("duty_cycle", "0")
            self._pwm_write("enable", "0")
        elif self._mode == "gpio":
            self._gpio_set(1)
            time.sleep(duration_ms / 1000.0)
            self._gpio_set(0)

    def pulse(self, count=3, on_ms=150, off_ms=100, duty_pct=50):
        for _ in range(count):
            self.buzz(on_ms, duty_pct)
            time.sleep(off_ms / 1000.0)

    def cleanup(self):
        if self._mode == "pwm":
            self._pwm_write("duty_cycle", "0")
            self._pwm_write("enable", "0")
        elif self._mode == "gpio":
            self._gpio_set(0)


# ---------------------------------------------------------------------------
#  Hardware abstraction: Buttons via gpio-keys input events
# ---------------------------------------------------------------------------
class ButtonListener:
    """Read button presses from gpio-keys /dev/input/eventN."""

    EV_KEY = 0x01
    EVENT_SIZE = struct.calcsize("llHHI")

    def __init__(self, callback_map):
        """callback_map: {linux_keycode: callable(value)}
        value: 1=pressed/switch-on, 0=released/switch-off"""
        self.callback_map = callback_map
        self._fds = {}
        self._thread = None
        self._running = False
        self._open_devices()

    def _open_devices(self):
        try:
            with open("/proc/bus/input/devices") as f:
                content = f.read()
            for block in content.split("\n\n"):
                if "smart-eye" not in block.lower() and "gpio-keys" not in block.lower():
                    continue
                handler = None
                name = ""
                for line in block.split("\n"):
                    if line.startswith("N: Name="):
                        name = line.split("=", 1)[1].strip().strip('"')
                    if line.startswith("H: Handlers="):
                        for h in line.split("=")[1].split():
                            if h.startswith("event"):
                                handler = h
                if handler:
                    path = f"/dev/input/{handler}"
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                        self._fds[fd] = path
                        log.info("Input device: %s -> %s (fd=%d)", name, path, fd)
                    except OSError:
                        log.warning("Cannot open %s (need root?)", path)
            if not self._fds:
                log.warning("No smart-eye / gpio-keys input devices found")
        except Exception as e:
            log.warning("Cannot enumerate input devices: %s", e)

    @staticmethod
    def _eviocgkey(length):
        """Build EVIOCGKEY(length) ioctl number: _IOC(_IOC_READ, 'E', 0x18, length)."""
        return 0x80004518 | (length << 16)

    def read_key_state(self, keycode):
        """Read current pressed/released state of a key via EVIOCGKEY ioctl.
        Returns 1 if pressed (switch active), 0 if released, or None on error."""
        import array
        if not self._fds:
            log.warning("read_key_state: no input fds open")
            return None
        buf_size = max((keycode // 8) + 1, 64)
        for fd in self._fds:
            try:
                buf = array.array("B", bytes(buf_size))
                fcntl.ioctl(fd, self._eviocgkey(buf_size), buf)
                byte_idx = keycode // 8
                bit_idx = keycode % 8
                if byte_idx < len(buf):
                    val = (buf[byte_idx] >> bit_idx) & 1
                    log.info("EVIOCGKEY fd=%d keycode=0x%x byte[%d]=0x%02x bit%d -> %d",
                             fd, keycode, byte_idx, buf[byte_idx], bit_idx, val)
                    return val
            except Exception as e:
                log.warning("EVIOCGKEY ioctl failed on fd=%d: %s", fd, e)
        return None

    def drain(self):
        """Discard any buffered input events so stale presses don't fire callbacks."""
        for fd in self._fds:
            while True:
                try:
                    data = os.read(fd, self.EVENT_SIZE * 16)
                    if not data:
                        break
                except (BlockingIOError, OSError):
                    break

    def start(self):
        if not self._fds:
            log.warning("No button input devices found")
            return
        self.drain()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        import select as sel
        while self._running:
            try:
                ready, _, _ = sel.select(list(self._fds.keys()), [], [], 0.2)
                for fd in ready:
                    data = os.read(fd, self.EVENT_SIZE)
                    if len(data) < self.EVENT_SIZE:
                        continue
                    _, _, ev_type, code, value = struct.unpack("llHHI", data)
                    if ev_type == self.EV_KEY:
                        cb = self.callback_map.get(code)
                        if cb:
                            cb(value)
            except Exception:
                pass

    def stop(self):
        self._running = False
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
#  Hardware abstraction: Audio playback via ALSA aplay
# ---------------------------------------------------------------------------
_SDMODE_GPIO = 131          # GPIO4_A3 -> BSS138 gate -> MAX98357A SD_MODE

SPEAKER_CARD = "SmartEyeAudio"
SPEAKER_DEV  = "smarteye_loud"

def _sysfs_gpio_export(gpio):
    """Export a GPIO via sysfs if not already exported."""
    path = f"/sys/class/gpio/gpio{gpio}"
    if not os.path.isdir(path):
        try:
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(gpio))
            time.sleep(0.05)
        except OSError:
            return False
    try:
        with open(f"{path}/direction", "w") as f:
            f.write("out")
    except OSError:
        return False
    return True

def _sysfs_gpio_set(gpio, value):
    """Set a GPIO value (0 or 1) via sysfs."""
    try:
        with open(f"/sys/class/gpio/gpio{gpio}/value", "w") as f:
            f.write(str(value))
    except OSError:
        pass


class AudioPlayer:
    """Audio playback via MAX98357A speaker amplifier.

    BSS138 inverted logic for SD_MODE:
      AMP_SD HIGH -> Q1 ON  -> SD_MODE LOW  -> amp OFF
      AMP_SD LOW  -> Q1 OFF -> pullup -> SD_MODE HIGH -> amp ON
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._sd_gpio_ok = self._init_sdmode()
        self._spk_card = self._find_card(SPEAKER_CARD)
        self._init_softvol()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Audio: speaker=card%s", self._spk_card)

    def _init_sdmode(self):
        """Export AMP_SD GPIO; set HIGH at init (amp OFF)."""
        ok = _sysfs_gpio_export(_SDMODE_GPIO)
        if ok:
            _sysfs_gpio_set(_SDMODE_GPIO, 1)
            log.info("AMP_SD (gpio%d) HIGH — speaker amp OFF", _SDMODE_GPIO)
        else:
            log.warning("Could not export AMP_SD gpio%d", _SDMODE_GPIO)
        return ok

    def _init_softvol(self):
        """Set the speaker softvol to 100%."""
        try:
            subprocess.run(
                ["amixer", "-c", SPEAKER_CARD, "sset", "SoftMaster", "100%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except Exception:
            pass

    @staticmethod
    def _find_card(name):
        try:
            with open("/proc/asound/cards") as f:
                for line in f:
                    if name in line:
                        return int(line.strip().split()[0])
        except Exception:
            pass
        return None

    @property
    def _active_dev(self):
        return SPEAKER_DEV if self._spk_card is not None else "default"

    def play(self, wav_path):
        if os.path.isfile(wav_path):
            self._queue.put(wav_path)

    def play_beep(self, freq=440, duration_ms=80, sample_rate=44100):
        """Play a short beep on the speaker. Non-blocking; queued like play()."""
        try:
            n = int(sample_rate * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, n, False)
            samples = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
            path = "/tmp/smart_eye_beep.wav"
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(samples.tobytes())
            self.play(path)
        except Exception as e:
            log.warning("Beep failed: %s", e)

    def _worker(self):
        while True:
            path = self._queue.get()
            if path is None:
                break
            self._play_sync(path)

    def _play_sync(self, path):
        dev = self._active_dev
        try:
            if self._sd_gpio_ok:
                _sysfs_gpio_set(_SDMODE_GPIO, 0)   # AMP_SD LOW -> amp ON
                time.sleep(0.01)
            self._proc = subprocess.Popen(
                ["aplay", "-D", dev, "-q", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._proc.wait(timeout=15)
        except Exception as e:
            log.warning("Audio playback error (%s): %s", dev, e)
        finally:
            if self._sd_gpio_ok:
                _sysfs_gpio_set(_SDMODE_GPIO, 1)   # AMP_SD HIGH -> amp OFF
            self._proc = None

    def stop(self):
        self._queue.put(None)
        if self._proc:
            self._proc.terminate()
        if self._sd_gpio_ok:
            _sysfs_gpio_set(_SDMODE_GPIO, 1)


# ---------------------------------------------------------------------------
#  Hardware abstraction: Battery gauge (BQ27220 on I2C3)
# ---------------------------------------------------------------------------
I2C_SLAVE = 0x0703

class BatteryGauge:
    """Read battery status from BQ27220 fuel gauge via /dev/i2c-3.

    BQ27220 register map (TI SLUSE14):
      0x06 Temperature       (0.1 K)
      0x08 Voltage           (mV)
      0x10 RemainingCapacity (mAh)
      0x12 FullChargeCapacity(mAh)
      0x14 AverageCurrent    (mA, signed)
      0x16 TimeToEmpty       (minutes, 0xFFFF = charging/not discharging)
      0x22 AveragePower      (cW, signed)
      0x2C StateOfCharge     (%)   -- NOTE: NOT 0x28 on BQ27220

    The gauge's internal DesignCapacity/FCC registers may not match the
    actual battery (requires sealed-mode programming).  We use the known
    DESIGN_CAPACITY_MAH for runtime estimation instead.
    """

    DESIGN_CAPACITY_MAH = 10000

    REG_TEMP     = 0x06
    REG_VOLTAGE  = 0x08
    REG_REM_CAP  = 0x10
    REG_FCC      = 0x12
    REG_AVG_CUR  = 0x14
    REG_TTE      = 0x16
    REG_AVG_PWR  = 0x22
    REG_SOC      = 0x2C

    def __init__(self, bus=3, addr=0x55):
        self.bus = bus
        self.addr = addr
        self.fd = None
        self._lock = threading.Lock()
        try:
            self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
        except OSError as e:
            log.warning("Cannot open BQ27220: %s", e)

    def _rw(self, reg):
        if self.fd is None:
            return 0
        with self._lock:
            os.write(self.fd, bytes([reg]))
            d = os.read(self.fd, 2)
        return d[0] | (d[1] << 8)

    def _rw_signed(self, reg):
        v = self._rw(reg)
        return v if v < 0x8000 else v - 0x10000

    @property
    def voltage_mv(self):
        return self._rw(self.REG_VOLTAGE)

    @property
    def soc(self):
        return self._rw(self.REG_SOC)

    @property
    def current_ma(self):
        return self._rw_signed(self.REG_AVG_CUR)

    @property
    def remaining_mah(self):
        """Estimated remaining capacity based on SOC and known design capacity."""
        return int(self.soc * self.DESIGN_CAPACITY_MAH / 100)

    @property
    def full_charge_mah(self):
        return self.DESIGN_CAPACITY_MAH

    @property
    def avg_power_mw(self):
        return self._rw_signed(self.REG_AVG_PWR) * 10

    @property
    def time_to_empty_min(self):
        """Compute TTE from remaining capacity and average current draw."""
        cur = self.current_ma
        if cur >= 0:
            return None
        rem = self.remaining_mah
        return int(rem / abs(cur) * 60) if cur != 0 else None

    @property
    def temperature_c(self):
        return self._rw(self.REG_TEMP) * 0.1 - 273.15

    def snapshot(self):
        """Return a dict of all key battery parameters in one call."""
        soc = self.soc
        vmv = self.voltage_mv
        cur = self.current_ma
        rem = int(soc * self.DESIGN_CAPACITY_MAH / 100)
        tte = int(rem / abs(cur) * 60) if cur < 0 else None
        power_mw = int(vmv * abs(cur) / 1000)
        return {
            "soc": soc,
            "voltage_mv": vmv,
            "current_ma": cur,
            "remaining_mah": rem,
            "full_charge_mah": self.DESIGN_CAPACITY_MAH,
            "avg_power_mw": power_mw,
            "tte_min": tte,
            "temp_c": self.temperature_c,
        }

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# ---------------------------------------------------------------------------
#  Hardware abstraction: Ultrasonic sensors via UART
# ---------------------------------------------------------------------------
class UltrasonicSensor:
    """AJ-SR04M ultrasonic sensor on UART (Mode 4: serial trigger/response).

    Protocol: send 0x01, receive 0xFF H_DATA L_DATA SUM.
    Distance = (H_DATA<<8 | L_DATA) in mm.

    Supports reconnection when the sensor is powered on after boot (e.g. via
    a manual power switch on UART2 to avoid boot hangs). If init fails or the
    connection is lost, try_reconnect() is called periodically until the port
    becomes available.
    """

    RECONNECT_INTERVAL = 5.0   # seconds between reconnect attempts
    FAIL_RESET_THRESHOLD = 20  # after this many consecutive no-response, close & retry (helps when sensor powered on late)

    def __init__(self, device="/dev/ttyS6", baudrate=9600, label="ultrasonic"):
        import serial
        self.label = label
        self._last_reconnect = 0
        self.ser = None
        if not device or str(device).lower() in ("none", "off", "disabled"):
            self._device = None
            self._baudrate = baudrate
            self._reconnect_enabled = False
            self._consecutive_failures = 0
            log.info("%s sensor disabled (no device)", label)
            return
        self._device = device
        self._baudrate = baudrate
        self._reconnect_enabled = True
        self._consecutive_failures = 0
        self._try_connect()

    def _try_connect(self):
        """Attempt to open the serial port. Sets self.ser on success."""
        import serial
        if not self._reconnect_enabled or self._device is None:
            return
        try:
            self.ser = serial.Serial(
                port=self._device, baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=1.0,
            )
            self.ser.reset_input_buffer()
            log.info("%s sensor on %s @ %d baud", self.label, self._device, self._baudrate)
        except Exception as e:
            log.warning("%s sensor init failed (%s): %s", self.label, self._device, e)
            if "No such file or directory" in str(e) and "ttyS" in self._device:
                log.info("Serial port %s not found. Verify UART overlay is installed and device tree is compiled.", self._device)
            self.ser = None

    def try_reconnect(self):
        """Reconnect if the port was unavailable. Throttled to avoid log spam."""
        if not self._reconnect_enabled or self._device is None or self.ser is not None:
            return
        now = time.monotonic()
        if now - self._last_reconnect < self.RECONNECT_INTERVAL:
            return
        self._last_reconnect = now
        log.info("%s: attempting reconnection on %s", self.label, self._device)
        self._try_connect()

    def _close_port(self):
        """Close serial port and clear handle (allows try_reconnect to reopen)."""
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def measure_mm(self):
        if self.ser is None:
            self.try_reconnect()
            return None
        try:
            self.ser.reset_input_buffer()
            self.ser.write(bytes([0x01]))
            self.ser.flush()
            time.sleep(0.06)

            timeout = 0
            while timeout < 20:
                if self.ser.in_waiting > 0 and self.ser.read(1)[0] == 0xFF:
                    break
                time.sleep(0.01)
                timeout += 1
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.FAIL_RESET_THRESHOLD:
                    log.info("%s: no response for %d reads, closing port to retry later (sensor powered on?)",
                             self.label, self._consecutive_failures)
                    self._close_port()
                return None

            remaining = self.ser.read(3)
            if len(remaining) < 3:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.FAIL_RESET_THRESHOLD:
                    self._close_port()
                return None
            h, l, _ = remaining
            self._consecutive_failures = 0
            return (h << 8) | l
        except Exception as e:
            log.info("%s read error (disconnected?): %s", self.label, e)
            self._close_port()
            return None

    def measure_cm(self):
        mm = self.measure_mm()
        return mm / 10.0 if mm else None

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


# ---------------------------------------------------------------------------
#  Camera + RKNN inference
# ---------------------------------------------------------------------------
class CameraDetector:
    """V4L2 camera capture with YOLOv8 RKNN inference.

    All frames are rotated 90° counter-clockwise before use, so
    both object detection and OCR always receive the correctly
    oriented image.
    """

    def __init__(self, model_path, device="/dev/video11",
                 target="rk3588", confidence=0.25, labels=None):
        self.device = device
        self.confidence = confidence
        self.cap = None
        self.model = None
        self.platform = None
        self.co_helper = None
        self.custom_labels = labels
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._cam_running = True
        self._init_camera()
        self._init_model(model_path, target)

    def _init_camera(self):
        dev_num = int(self.device.replace("/dev/video", ""))
        self.cap = cv2.VideoCapture(dev_num)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera %s opened at %dx%d (rotated 90° CCW -> %dx%d)", self.device, w, h, h, w)
        self._cam_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cam_thread.start()

    def _capture_loop(self):
        """Continuously capture and rotate frames in background."""
        while self._cam_running:
            ret, frame = self.cap.read()
            if ret:
                rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                with self._frame_lock:
                    self._latest_frame = rotated

    def _init_model(self, model_path, target):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "pathpal_project"))
        from yolov8 import setup_model, post_process, IMG_SIZE, CLASSES
        from py_utils.coco_utils import COCO_test_helper
        self._post_process = post_process
        self._IMG_SIZE = IMG_SIZE
        self._CLASSES = self.custom_labels or CLASSES
        self.model, self.platform = setup_model(model_path, target, None)
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        log.info("RKNN model loaded: %s", model_path)

    def grab_frame(self):
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.co_helper.letter_box(
            im=img, new_shape=(self._IMG_SIZE[1], self._IMG_SIZE[0]),
            pad_color=(0, 0, 0),
        )
        inp = np.expand_dims(img, 0)
        outputs = self.model.run([inp])
        boxes, classes, scores = self._post_process(outputs)
        results = []
        if boxes is not None:
            for box, cls_id, score in zip(boxes, classes, scores):
                if score >= self.confidence:
                    label = self._CLASSES[cls_id] if cls_id < len(self._CLASSES) else f"class_{cls_id}"
                    results.append({
                        "label": label.strip(),
                        "score": float(score),
                        "box": [int(x) for x in box],
                    })
        return results

    def close(self):
        self._cam_running = False
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
#  OCR
# ---------------------------------------------------------------------------
class OCREngine:
    """RapidOCR wrapper (onnxruntime backend, no paddlepaddle dependency).

    Uses a single RapidOCR instance for all languages -- the ONNX models
    handle multilingual text (Latin + Devanagari) natively.
    """

    def __init__(self):
        self._ocr = None

    def warmup(self):
        """Pre-load OCR models so first real call is fast."""
        self._get_ocr()

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            log.info("RapidOCR (onnxruntime) initialized")
        return self._ocr

    def recognize(self, frame, lang="en", scale=1.5):
        h, w = frame.shape[:2]
        upscaled = cv2.resize(frame, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_LINEAR)
        ocr = self._get_ocr()
        result, _ = ocr(upscaled)
        texts = []
        if result:
            for box, txt, conf in result:
                if conf > 0.5:
                    texts.append(txt)
        return " ".join(texts) if texts else None


# ---------------------------------------------------------------------------
#  Translation
# ---------------------------------------------------------------------------
class Translator:
    """argostranslate wrapper for English <-> Hindi (fully offline).

    Requires language packages to be pre-installed on the device.
    Install once (with internet) via:
        argospm install translate-en_hi
        argospm install translate-hi_en
    """

    def __init__(self):
        self._checked = False
        self._tr = None

    def _ensure_ready(self):
        if self._checked:
            return
        try:
            import argostranslate.translate as tr
            self._tr = tr
            langs = {(t.from_language.code, t.to_language.code)
                     for t in tr.get_installed_languages()
                     for t in (t.translations_from if hasattr(t, 'translations_from') else [])}
            self._checked = True
        except Exception as e:
            log.warning("Translation init failed (offline only): %s", e)
            self._checked = True

    def translate(self, text, from_lang="en", to_lang="hi"):
        self._ensure_ready()
        try:
            tr = self._tr
            if tr is None:
                import argostranslate.translate as tr
            return tr.translate(text, from_lang, to_lang)
        except Exception as e:
            log.warning("Translation error: %s", e)
            return text


# ---------------------------------------------------------------------------
#  Audio file map
# ---------------------------------------------------------------------------
AUDIO_FILES = {
    "en": {
        "device_on":       "wav/English/device_turned_on.wav",
        "pothole":         "wav/English/pothole.wav",
        "stairs":          "wav/English/stairs.wav",
        "no_text":         "wav/English/no_text_detected.wav",
        "battery_10":      "wav/English/battery_10.wav",
        "battery_shutdown": "wav/English/battery_shutdown.wav",
        "lang_toggle":     "wav/English/English_mode.wav",
        "fifty rupees":    "wav/English/fifty_rupees.wav",
        "five hundred rupees": "wav/English/five_hundred_rupees.wav",
        "five rupees":     "wav/English/five_rupees.wav",
        "hundred rupees":  "wav/English/hundred_rupees.wav",
        "one rupees":      "wav/English/one_rupees.wav",
        "ten rupees":      "wav/English/ten_rupees.wav",
        "twenty rupees":   "wav/English/twenty_rupees.wav",
        "two hundred rupees": "wav/English/two_hundred_rupees.wav",
        "two rupees":      "wav/English/two_rupees.wav",
        "two thousand rupees": "wav/English/two_thousand_rupees.wav",
    },
    "hi": {
        "device_on":       "wav/Hindi/turned_on.wav",
        "pothole":         "wav/Hindi/pothole_hindi.wav",
        "stairs":          "wav/Hindi/stairs_detected.wav",
        "no_text":         "wav/Hindi/no_text_found.wav",
        "battery_10":      "wav/Hindi/ten_percent_battery_alert.wav",
        "battery_shutdown": "wav/Hindi/five_percent_battery_alert.wav",
        "lang_toggle":     "wav/Hindi/Hindi_mode.wav",
        "fifty rupees":    "wav/Hindi/fifty_rupees.wav",
        "five hundred rupees": "wav/Hindi/five_hundred_rupees.wav",
        "five rupees":     "wav/Hindi/five_rupees.wav",
        "hundred rupees":  "wav/Hindi/hundred_rupees.wav",
        "one rupees":      "wav/Hindi/one_rupees.wav",
        "ten rupees":      "wav/Hindi/ten_rupees.wav",
        "twenty rupees":   "wav/Hindi/twenty_rupees.wav",
        "two hundred rupees": "wav/Hindi/two_hundred_rupees.wav",
        "two rupees":      "wav/Hindi/two_rupees.wav",
        "two thousand rupees": "wav/Hindi/two_thousand_rupees.wav",
    },
}


# ---------------------------------------------------------------------------
#  Application
# ---------------------------------------------------------------------------
class SmartEyeApp:
    US_FRONT_THRESHOLD_CM = 60
    US_DOWN_THRESHOLD_CM = 130
    ANNOUNCE_COOLDOWN = 2.0
    OCR_COOLDOWN = 1.5
    BATTERY_INTERVAL = 30.0
    DETECTION_FPS = 5             # target detection frame rate (saves ~70% CPU vs unlimited)
    ULTRASONIC_INTERVAL = 0.5     # check ultrasonic every 500ms, not every frame
    US_INTER_SENSOR_DELAY = 0.2   # delay between front and down to avoid acoustic crosstalk (seconds)
    GC_INTERVAL = 10.0            # gc.collect every 10s instead of every frame

    SOC_WARN = 20
    SOC_SHUTDOWN = 10

    def __init__(self, args):
        self.args = args
        self.lang = "en"
        self._running = True
        self._last_announce = 0
        self._last_ocr = 0
        self._last_ultrasonic = 0
        self._last_gc = 0
        self._battery_warned = False

        log.info("Initializing Smart Eye system...")

        _apply_power_profile()

        self.vibrator = VibrationMotor()
        self.audio = AudioPlayer()
        self.gauge = BatteryGauge()
        self.us_front = UltrasonicSensor(
            getattr(self.args, "us_front_device", "/dev/ttyS2"), label="US-front"
        )
        self.us_down = UltrasonicSensor(
            getattr(self.args, "us_down_device", "/dev/ttyS6"), label="US-down"
        )
        self.ocr = OCREngine()
        self.translator = Translator()

        self.detector = CameraDetector(
            model_path=args.model,
            device=args.camera,
            confidence=args.threshold,
            labels=self._load_labels(args.labels),
        )

        self.buttons = ButtonListener({
            0x100: self._on_lang_event,
            0x101: self._on_ocr_btn,
        })
        self.buttons.start()
        self._init_language()

        self._battery_thread = threading.Thread(target=self._battery_loop, daemon=True)
        self._battery_thread.start()

        threading.Thread(target=self.ocr.warmup, daemon=True).start()

    def _init_language(self):
        """Read the physical LANG_BTN switch position and set self.lang."""
        lang_state = self.buttons.read_key_state(0x100)
        if lang_state is None:
            lang_state = self._read_lang_gpio()
        if lang_state is not None:
            self.lang = "en" if lang_state == 1 else "hi"
            log.info("LANG_BTN initial state: %d -> language=%s", lang_state, self.lang)
        else:
            log.warning("Could not read LANG_BTN state, defaulting to %s", self.lang)

    @staticmethod
    def _read_lang_gpio():
        """Fallback: read LANG_BTN GPIO0_D0 (gpio 24) via any available method."""
        _LANG_GPIO = 24
        try:
            with open(f"/sys/class/gpio/gpio{_LANG_GPIO}/value") as f:
                raw = int(f.read().strip())
                val = 0 if raw == 1 else 1
                log.info("LANG_BTN gpio%d sysfs raw=%d -> state=%d", _LANG_GPIO, raw, val)
                return val
        except OSError:
            pass
        log.warning("LANG_BTN GPIO fallback also failed")
        return None

    @staticmethod
    def _load_labels(path):
        if not path or not os.path.isfile(path):
            return None
        with open(path) as f:
            return tuple(l.strip() for l in f if l.strip())

    def _audio_path(self, key):
        rel = AUDIO_FILES.get(self.lang, AUDIO_FILES["en"]).get(key)
        if rel:
            return os.path.join(PROJECT_ROOT, rel)
        return None

    def _play(self, key):
        p = self._audio_path(key)
        if p:
            self.audio.play(p)

    # --- Language slide switch (LANG_BTN GPIO0_D0, active-low) ---
    # gpio-keys sends EV_KEY with BTN_MISC (0x100):
    #   value=1 -> switch active (GPIO low)  -> English
    #   value=0 -> switch released (GPIO high) -> Hindi
    def _on_lang_event(self, value):
        new_lang = "en" if value == 1 else "hi"
        if new_lang != self.lang:
            self.lang = new_lang
            log.info("Language switched to %s (switch=%d)", self.lang, value)
            self.vibrator.buzz(100)
            self._play("lang_toggle")

    def _on_ocr_btn(self, value):
        if value != 1:
            return
        now = time.time()
        if now - self._last_ocr < self.OCR_COOLDOWN:
            return
        self._last_ocr = now
        log.info("OCR triggered")
        threading.Thread(target=lambda: self.vibrator.buzz(80), daemon=True).start()
        self.audio.play_beep()
        threading.Thread(target=self._run_ocr, daemon=True).start()

    # --- OCR ---
    def _run_ocr(self):
        frame = self.detector.grab_frame()
        if frame is None:
            return
        text = self.ocr.recognize(frame, lang=self.lang)
        if not text:
            self._play("no_text")
            return

        log.info("OCR result: %s", text[:100])

        if self.lang == "hi":
            translated = self.translator.translate(text, "en", "hi")
            log.info("Translated to Hindi: %s", translated[:100])
            self._tts_speak(translated, force_lang="hi")
        else:
            self._tts_speak(text, force_lang="en")

    _MAX_TTS_CHARS = 300

    def _tts_speak(self, text, force_lang=None):
        wav_out = "/tmp/smart_eye_tts.wav"
        lang = force_lang or self.lang
        if len(text) > self._MAX_TTS_CHARS:
            text = text[:self._MAX_TTS_CHARS]
        safe_text = text.replace('"', '\\"').replace("'", "\\'")

        try:
            piper_bin = os.path.join(PROJECT_ROOT, "piper/piper/piper")
            piper_models = {
                "en": os.path.join(PROJECT_ROOT, "piper/en_US-amy-medium.onnx"),
                "hi": os.path.join(PROJECT_ROOT, "piper/hi_IN-pratham-medium.onnx"),
            }
            model = piper_models.get(lang, piper_models["en"])

            if os.path.isfile(piper_bin) and os.path.isfile(model):
                subprocess.run(
                    f'echo "{safe_text}" | "{piper_bin}" --model "{model}" '
                    f'--length-scale 0.85 --output_file "{wav_out}"',
                    shell=True, timeout=10,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.audio.play(wav_out)
                return

            espeak = "/usr/bin/espeak-ng"
            if not os.path.isfile(espeak):
                espeak = "/usr/bin/espeak"
            if os.path.isfile(espeak):
                subprocess.run(
                    [espeak, "-v", lang, "-s", "180", "-w", wav_out, safe_text],
                    timeout=5,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.audio.play(wav_out)
                return

            log.info("No TTS engine available (install espeak-ng: sudo apt install espeak-ng)")
        except Exception as e:
            log.warning("TTS error: %s", e)

    # --- Battery monitor ---
    def _battery_loop(self):
        while self._running:
            try:
                b = self.gauge.snapshot()
                soc = b["soc"]
                tte = b["tte_min"]
                tte_str = f"{tte}min ({tte / 60:.1f}h)" if tte else "charging/N/A"
                log.info(
                    "Battery: %d%% %dmV %dmA %dmW rem=%dmAh TTE=%s T=%.0fC",
                    soc, b["voltage_mv"], b["current_ma"],
                    b["avg_power_mw"], b["remaining_mah"], tte_str, b["temp_c"],
                )

                if soc <= self.SOC_SHUTDOWN and not self._battery_warned:
                    self._play("battery_shutdown")
                    self._battery_warned = True
                elif soc <= self.SOC_WARN and not self._battery_warned:
                    self._play("battery_10")
                    self._battery_warned = True
                elif soc > self.SOC_WARN:
                    self._battery_warned = False
            except Exception as e:
                log.warning("Battery read error: %s", e)
            time.sleep(self.BATTERY_INTERVAL)

    # --- Detection announcement ---
    def _announce(self, label):
        now = time.time()
        if now - self._last_announce < self.ANNOUNCE_COOLDOWN:
            return
        self._last_announce = now

        audio = self._audio_path(label)
        if audio:
            self.audio.play(audio)
            self.vibrator.buzz(200)
        elif label in ("pothole", "stairs"):
            self._play(label)
            self.vibrator.pulse(3, 200, 100, 80)
        else:
            log.info("TTS announce: %s", label)
            self._tts_speak(f"{label} detected")
            self.vibrator.buzz(200)

    # --- Main loop ---
    def run(self):
        self._play("device_on")
        self.vibrator.pulse(2, 150, 100)

        log.info("System ready. Detection loop starting...")
        log.info("  Model:       %s", self.args.model)
        log.info("  Camera:      %s", self.args.camera)
        log.info("  Threshold:   %.2f", self.args.threshold)
        log.info("  Language:    %s", self.lang)
        log.info("  Target FPS:  %d", self.DETECTION_FPS)

        frame_interval = 1.0 / self.DETECTION_FPS

        try:
            while self._running:
                t_start = time.monotonic()

                frame = self.detector.grab_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                detections = self.detector.detect(frame)
                if detections:
                    log.info("Detections: %s",
                             ", ".join(f'{d["label"]}({d["score"]:.2f})' for d in detections))
                for det in detections:
                    self._announce(det["label"])

                now = time.monotonic()
                if now - self._last_ultrasonic >= self.ULTRASONIC_INTERVAL:
                    self._last_ultrasonic = now
                    d_front = self.us_front.measure_cm()
                    time.sleep(self.US_INTER_SENSOR_DELAY)
                    d_down  = self.us_down.measure_cm()

                    if d_front is not None and d_front < self.US_FRONT_THRESHOLD_CM:
                        log.info("US-front obstacle: %.0f cm", d_front)
                        self.vibrator.buzz(300, 80)

                    if d_down is not None and d_down < self.US_DOWN_THRESHOLD_CM:
                        log.info("US-down obstacle: %.0f cm", d_down)
                        self.vibrator.pulse(2, 150, 100, 60)

                if now - self._last_gc >= self.GC_INTERVAL:
                    self._last_gc = now
                    gc.collect()

                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        self._running = False
        self.vibrator.cleanup()
        self.buttons.stop()
        self.audio.stop()
        self.detector.close()
        self.us_front.close()
        self.us_down.close()
        self.gauge.close()
        log.info("Cleanup complete.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smart Eye assistive vision system")
    p.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models/pathpal/model_v2_large.rknn"),
                   help="Path to RKNN model file")
    p.add_argument("--camera", default="/dev/video11",
                   help="V4L2 camera device")
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Detection confidence threshold")
    p.add_argument("--labels", default=os.path.join(PROJECT_ROOT, "models/pathpal/labels.txt"),
                   help="Path to custom class labels file (one per line)")
    p.add_argument("--fps", type=int, default=5,
                   help="Target detection FPS (lower = less power, default 5)")
    p.add_argument("--us-front-device", default="/dev/ttyS3",
                   help="Serial device for front ultrasonic (default /dev/ttyS3 via UART3-M2)")
    p.add_argument("--us-down-device", default="/dev/ttyS6",
                   help="Serial device for bottom ultrasonic (default /dev/ttyS6 via UART6-M2; use 'none' to disable)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = SmartEyeApp(args)
    if hasattr(args, "fps"):
        app.DETECTION_FPS = args.fps
    app.run()