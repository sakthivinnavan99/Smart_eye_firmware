"""
Smart Eye device configuration.

All tunable parameters live here.  Edit this file on the device and restart
the service — no need to touch main.py for routine threshold or timing changes.

    sudo systemctl restart smart-eye
"""

# ── Hardware paths ────────────────────────────────────────────────────────────

CAMERA_DEVICE   = "/dev/video11"
US_FRONT_DEVICE = "/dev/ttyS3"    # front-facing  AJ-SR04M via UART3-M2
US_DOWN_DEVICE  = "/dev/ttyS6"    # downward-facing AJ-SR04M via UART6-M2

# ── Model / inference ─────────────────────────────────────────────────────────
#
# MODEL_PATH and LABELS_PATH are relative to the project root (the directory
# that contains pathpal_project/, wav/, piper/, etc.).  main.py prepends
# PROJECT_ROOT automatically, so you only need to edit the filename here.

MODEL_PATH          = "models/pathpal/model_v2_large.rknn"
LABELS_PATH         = "models/pathpal/labels.txt"
PLATFORM            = "rk3582"    # rk3582 / rk3588s both normalise to rk3588 at runtime
DETECTION_THRESHOLD = 0.65        # YOLO confidence threshold (0–1)
DETECTION_FPS       = 5           # target inference frames per second

# ── Ultrasonic distance thresholds ───────────────────────────────────────────

US_FRONT_THRESHOLD_CM = 60        # vibrate if front obstacle is closer than this (cm)
US_DOWN_THRESHOLD_CM  = 130       # vibrate if downward reading is closer than this (cm)

# ── Ultrasonic background-thread timing ──────────────────────────────────────
#
# The UltrasonicManager runs both sensors on a single dedicated thread so the
# detection loop is never blocked by sensor I/O.  Timing rules:
#
#   US_POLL_INTERVAL_S    – minimum time between the *start* of successive
#                           front→down measurement pairs.  If a pair takes
#                           longer than this (worst case ~510 ms), the next
#                           pair begins immediately with no extra delay.
#
#   US_BETWEEN_SENSORS_S  – mandatory silence gap inserted between the end of
#                           the front measurement and the start of the down
#                           measurement.  At 343 m/s, 250 ms allows residual
#                           acoustic energy from the front burst to travel
#                           ~85 m — far beyond either sensor's max range —
#                           before the second trigger fires.  This is the
#                           primary anti-crosstalk guard.
#
#   US_VIBRATE_COOLDOWN_S – minimum time between vibrator alerts for the same
#                           sensor (prevents buzzing on every detection loop
#                           frame while a persistent obstacle is present).

US_POLL_INTERVAL_S    = 0.5       # min seconds between measurement pair starts
US_BETWEEN_SENSORS_S  = 0.25      # mandatory gap between front and down triggers (s)
US_VIBRATE_COOLDOWN_S = 1.0       # min seconds between vibrator alerts per sensor

# ── Audio / haptic cooldowns ──────────────────────────────────────────────────

ANNOUNCE_COOLDOWN_S = 2.0         # min seconds between YOLO detection announcements
OCR_COOLDOWN_S      = 1.5         # min seconds between OCR button presses

# ── Battery monitor ───────────────────────────────────────────────────────────

BATTERY_POLL_INTERVAL_S = 30.0    # how often to read SOC and log battery status (s)
SOC_WARN_PCT            = 20      # announce "low battery" when SOC falls to this %
SOC_SHUTDOWN_PCT        = 10      # announce "critical battery" when SOC falls to this %

# ── System ────────────────────────────────────────────────────────────────────

GC_INTERVAL_S = 10.0              # gc.collect() period (s)
