#!/usr/bin/env python3
"""
Smart Eye Firmware - Main Application
======================================
Assistive vision system for visually impaired users.

Hardware platform: Radxa CM5 (RK3588) on Smart Eye carrier board
  - Camera:      IMX219 via V4L2 (/dev/video11)
  - Inference:   YOLOv8 on RKNN NPU
  - Audio:       MAX98357A on ALSA Smart-Eye-Audio (card 2)
  - Buttons:     LANG_BTN (GPIO0_D0), OCR_BTN (GPIO0_C7) via gpio-keys
  - Vibration:   PWM7 on GPIO4_B3 via sysfs
  - Ultrasonic:  JSN-SR04T on UART6 (/dev/ttyS6)
  - Battery:     BQ27220 fuel gauge on I2C3 (0x55)
  - OCR:         PaddleOCR
  - Translation: argostranslate (English <-> Hindi)
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

import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("smart-eye")


# ---------------------------------------------------------------------------
#  Hardware abstraction: Vibration motor (PWM sysfs)
# ---------------------------------------------------------------------------
class VibrationMotor:
    """Control vibration motor via sysfs PWM (pwmchip0, GPIO4_B3)."""

    PWM_REG = "febd0"

    def __init__(self):
        self.chip = self._find_chip()
        self.pwm_dir = None
        if self.chip:
            self._export()

    def _find_chip(self):
        for p in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
            try:
                link = os.readlink(os.path.join(p, "device"))
                if self.PWM_REG in link:
                    return p
            except OSError:
                pass
        log.warning("PWM chip for vibration motor not found")
        return None

    def _export(self):
        self.pwm_dir = os.path.join(self.chip, "pwm0")
        if not os.path.isdir(self.pwm_dir):
            try:
                with open(os.path.join(self.chip, "export"), "w") as f:
                    f.write("0")
                time.sleep(0.1)
            except OSError:
                log.warning("Cannot export PWM channel")
                self.pwm_dir = None
                return
        self._write("period", "2000000")  # 500 Hz
        self._write("duty_cycle", "0")

    def _write(self, attr, val):
        if not self.pwm_dir:
            return
        try:
            with open(os.path.join(self.pwm_dir, attr), "w") as f:
                f.write(val)
        except OSError:
            pass

    def buzz(self, duration_ms, duty_pct=50):
        if not self.pwm_dir:
            return
        period = int(self._read("period") or 2000000)
        duty = int(period * duty_pct / 100)
        self._write("duty_cycle", str(duty))
        self._write("enable", "1")
        time.sleep(duration_ms / 1000.0)
        self._write("duty_cycle", "0")
        self._write("enable", "0")

    def pulse(self, count=3, on_ms=150, off_ms=100, duty_pct=50):
        for _ in range(count):
            self.buzz(on_ms, duty_pct)
            time.sleep(off_ms / 1000.0)

    def _read(self, attr):
        if not self.pwm_dir:
            return None
        try:
            with open(os.path.join(self.pwm_dir, attr)) as f:
                return f.read().strip()
        except OSError:
            return None

    def cleanup(self):
        self._write("duty_cycle", "0")
        self._write("enable", "0")


# ---------------------------------------------------------------------------
#  Hardware abstraction: Buttons via gpio-keys input events
# ---------------------------------------------------------------------------
class ButtonListener:
    """Read button presses from gpio-keys /dev/input/eventN."""

    EV_KEY = 0x01
    EVENT_SIZE = struct.calcsize("llHHI")

    def __init__(self, callback_map):
        """callback_map: {linux_keycode: callable}"""
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
                if "smart-eye" not in block.lower():
                    continue
                handler = None
                for line in block.split("\n"):
                    if line.startswith("H: Handlers="):
                        for h in line.split("=")[1].split():
                            if h.startswith("event"):
                                handler = h
                if handler:
                    path = f"/dev/input/{handler}"
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                        self._fds[fd] = path
                    except OSError:
                        log.warning("Cannot open %s (need root?)", path)
        except Exception as e:
            log.warning("Cannot enumerate input devices: %s", e)

    def start(self):
        if not self._fds:
            log.warning("No button input devices found")
            return
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
                    if ev_type == self.EV_KEY and value == 1:
                        cb = self.callback_map.get(code)
                        if cb:
                            cb()
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
class AudioPlayer:
    """Non-blocking WAV playback through ALSA Smart-Eye-Audio card."""

    def __init__(self):
        self.card = self._find_card()
        self._lock = threading.Lock()
        self._proc = None
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _find_card(self):
        try:
            with open("/proc/asound/cards") as f:
                for line in f:
                    if "SmartEyeAudio" in line:
                        return int(line.strip().split()[0])
        except Exception:
            pass
        log.warning("Smart-Eye-Audio card not found, falling back to default")
        return None

    def play(self, wav_path):
        if os.path.isfile(wav_path):
            self._queue.put(wav_path)

    def _worker(self):
        while True:
            path = self._queue.get()
            if path is None:
                break
            self._play_sync(path)

    def _play_sync(self, path):
        dev = f"plughw:{self.card},0" if self.card is not None else "default"
        try:
            self._proc = subprocess.Popen(
                ["aplay", "-D", dev, "-q", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._proc.wait(timeout=15)
        except Exception as e:
            log.warning("Audio playback error: %s", e)
        finally:
            self._proc = None

    def stop(self):
        self._queue.put(None)
        if self._proc:
            self._proc.terminate()


# ---------------------------------------------------------------------------
#  Hardware abstraction: Battery gauge (BQ27220 on I2C3)
# ---------------------------------------------------------------------------
I2C_SLAVE = 0x0703

class BatteryGauge:
    """Read battery status from BQ27220 fuel gauge via /dev/i2c-3."""

    def __init__(self, bus=3, addr=0x55):
        self.bus = bus
        self.addr = addr
        self.fd = None
        try:
            self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
        except OSError as e:
            log.warning("Cannot open BQ27220: %s", e)

    def _rw(self, reg):
        if self.fd is None:
            return 0
        os.write(self.fd, bytes([reg]))
        d = os.read(self.fd, 2)
        return d[0] | (d[1] << 8)

    def _rw_signed(self, reg):
        v = self._rw(reg)
        return v if v < 0x8000 else v - 0x10000

    @property
    def voltage_mv(self):
        return self._rw(0x08)

    @property
    def soc(self):
        return self._rw(0x2C)

    @property
    def current_ma(self):
        return self._rw_signed(0x14)

    @property
    def temperature_c(self):
        return self._rw(0x06) / 10.0 - 273.15

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# ---------------------------------------------------------------------------
#  Hardware abstraction: Ultrasonic sensor via UART6
# ---------------------------------------------------------------------------
class UltrasonicSensor:
    """JSN-SR04T ultrasonic sensor on UART6 (/dev/ttyS6).

    Protocol: send 0x01, receive 0xFF H_DATA L_DATA SUM.
    Distance = (H_DATA<<8 | L_DATA) in mm.
    """

    def __init__(self, device="/dev/ttyS6", baudrate=9600):
        import serial
        try:
            self.ser = serial.Serial(
                port=device, baudrate=baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=1.0,
            )
            self.ser.reset_input_buffer()
            log.info("Ultrasonic sensor on %s @ %d baud", device, baudrate)
        except Exception as e:
            log.warning("Ultrasonic sensor init failed: %s", e)
            self.ser = None

    def measure_mm(self):
        if self.ser is None:
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
                return None

            remaining = self.ser.read(3)
            if len(remaining) < 3:
                return None
            h, l, _ = remaining
            return (h << 8) | l
        except Exception:
            return None

    def measure_cm(self):
        mm = self.measure_mm()
        return mm / 10.0 if mm else None

    def close(self):
        if self.ser:
            self.ser.close()


# ---------------------------------------------------------------------------
#  Camera + RKNN inference
# ---------------------------------------------------------------------------
class CameraDetector:
    """V4L2 camera capture with YOLOv8 RKNN inference."""

    def __init__(self, model_path, device="/dev/video11",
                 target="rk3588", confidence=0.25, labels=None):
        self.device = device
        self.confidence = confidence
        self.cap = None
        self.model = None
        self.platform = None
        self.co_helper = None
        self.custom_labels = labels
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
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera %s opened at %dx%d", self.device, w, h)

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
        ret, frame = self.cap.read()
        return frame if ret else None

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
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
#  OCR
# ---------------------------------------------------------------------------
class OCREngine:
    """PaddleOCR wrapper."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        self.ocr_hi = None  # lazy-loaded

    def _get_hi(self):
        if self.ocr_hi is None:
            from paddleocr import PaddleOCR
            self.ocr_hi = PaddleOCR(use_angle_cls=True, lang="hi", show_log=False)
        return self.ocr_hi

    def recognize(self, frame, lang="en", scale=2.0):
        h, w = frame.shape[:2]
        upscaled = cv2.resize(frame, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
        engine = self.ocr_en if lang == "en" else self._get_hi()
        result = engine.ocr(upscaled, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                txt = line[1][0]
                conf = line[1][1]
                if conf > 0.5:
                    texts.append(txt)
        return " ".join(texts) if texts else None


# ---------------------------------------------------------------------------
#  Translation
# ---------------------------------------------------------------------------
class Translator:
    """argostranslate wrapper for English <-> Hindi."""

    def __init__(self):
        self._installed = False

    def _ensure_models(self):
        if self._installed:
            return
        try:
            import argostranslate.package as pkg
            import argostranslate.translate as tr
            pkg.update_package_index()
            available = pkg.get_available_packages()
            for p in available:
                if (p.from_code == "en" and p.to_code == "hi") or \
                   (p.from_code == "hi" and p.to_code == "en"):
                    if not any(ip.from_code == p.from_code and ip.to_code == p.to_code
                              for ip in pkg.get_installed_packages()):
                        pkg.install_from_path(p.download())
            self._installed = True
        except Exception as e:
            log.warning("Translation model install failed: %s", e)

    def translate(self, text, from_lang="en", to_lang="hi"):
        self._ensure_models()
        try:
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
    ULTRASONIC_THRESHOLD_CM = 155
    HANGING_THRESHOLD_CM = 90
    ANNOUNCE_COOLDOWN = 2.0
    OCR_COOLDOWN = 3.0
    BATTERY_INTERVAL = 30.0

    SOC_WARN = 20
    SOC_SHUTDOWN = 10

    def __init__(self, args):
        self.args = args
        self.lang = "en"
        self._running = True
        self._last_announce = 0
        self._last_ocr = 0
        self._battery_warned = False

        log.info("Initializing Smart Eye system...")

        self.vibrator = VibrationMotor()
        self.audio = AudioPlayer()
        self.gauge = BatteryGauge()
        self.ultrasonic = UltrasonicSensor()
        self.ocr = OCREngine()
        self.translator = Translator()

        self.detector = CameraDetector(
            model_path=args.model,
            device=args.camera,
            confidence=args.threshold,
            labels=self._load_labels(args.labels),
        )

        self.buttons = ButtonListener({
            0x100: self._on_lang_btn,   # BTN_MISC  -> language toggle
            0x101: self._on_ocr_btn,    # BTN_1     -> OCR trigger
        })
        self.buttons.start()

        self._battery_thread = threading.Thread(target=self._battery_loop, daemon=True)
        self._battery_thread.start()

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

    # --- Button callbacks ---
    def _on_lang_btn(self):
        self.lang = "hi" if self.lang == "en" else "en"
        log.info("Language switched to %s", self.lang)
        self.vibrator.buzz(100)
        self._play("lang_toggle")

    def _on_ocr_btn(self):
        now = time.time()
        if now - self._last_ocr < self.OCR_COOLDOWN:
            return
        self._last_ocr = now
        log.info("OCR triggered")
        self.vibrator.buzz(200)
        threading.Thread(target=self._run_ocr, daemon=True).start()

    # --- OCR ---
    def _run_ocr(self):
        frame = self.detector.grab_frame()
        if frame is None:
            return
        text = self.ocr.recognize(frame, lang=self.lang)
        if text:
            log.info("OCR result: %s", text[:100])
            self._tts_speak(text)
        else:
            self._play("no_text")

    def _tts_speak(self, text):
        try:
            wav_out = "/tmp/smart_eye_tts.wav"
            piper_bin = os.path.join(PROJECT_ROOT, "piper/piper/piper")
            model = os.path.join(PROJECT_ROOT,
                                 "piper/en_US-amy-medium.onnx" if self.lang == "en"
                                 else "piper/hi_IN-pratham-medium.onnx")
            if os.path.isfile(piper_bin) and os.path.isfile(model):
                subprocess.run(
                    f'echo "{text}" | "{piper_bin}" --model "{model}" --output_file "{wav_out}"',
                    shell=True, timeout=10,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.audio.play(wav_out)
            else:
                log.info("Piper TTS not available, skipping speech")
        except Exception as e:
            log.warning("TTS error: %s", e)

    # --- Battery monitor ---
    def _battery_loop(self):
        while self._running:
            try:
                soc = self.gauge.soc
                v = self.gauge.voltage_mv
                cur = self.gauge.current_ma
                log.info("Battery: %d%% %dmV %dmA", soc, v, cur)

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

    # --- Main loop ---
    def run(self):
        self._play("device_on")
        self.vibrator.pulse(2, 150, 100)

        log.info("System ready. Detection loop starting...")
        log.info("  Model:       %s", self.args.model)
        log.info("  Camera:      %s", self.args.camera)
        log.info("  Threshold:   %.2f", self.args.threshold)
        log.info("  Language:    %s", self.lang)

        try:
            while self._running:
                frame = self.detector.grab_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                detections = self.detector.detect(frame)
                for det in detections:
                    self._announce(det["label"])

                dist = self.ultrasonic.measure_cm()
                if dist is not None:
                    if dist < self.HANGING_THRESHOLD_CM:
                        log.info("Ultrasonic obstacle: %.0f cm", dist)
                        self.vibrator.buzz(300, 80)

                gc.collect()

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
        self.ultrasonic.close()
        self.gauge.close()
        log.info("Cleanup complete.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smart Eye assistive vision system")
    p.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models/yolov8/yolov8.rknn"),
                   help="Path to RKNN model file")
    p.add_argument("--camera", default="/dev/video11",
                   help="V4L2 camera device")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Detection confidence threshold")
    p.add_argument("--labels", default=None,
                   help="Path to custom class labels file (one per line)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = SmartEyeApp(args)
    app.run()
