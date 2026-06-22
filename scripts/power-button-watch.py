#!/usr/bin/env python3
"""
Power button hold-to-shutdown daemon.
Monitors ALL /dev/input/event* devices for KEY_POWER (code 116).
Initiates poweroff only after HOLD_SECS continuous hold.
systemd-logind must have HandlePowerKey=ignore.
"""
import os
import struct
import glob
import time
import subprocess
import select

HOLD_SECS = 3.0
KEY_POWER  = 116
EV_KEY     = 1

# ARM64 Linux input_event: tv_sec(8) + tv_usec(8) + type(2) + code(2) + value(4) = 24 bytes
# 'q' = signed int64.  Do NOT use 'l' — Python struct '=l' is always 4 bytes.
EVENT_FMT  = "=qqHHi"
EVENT_SIZE = 24


# ── vibration motor ──────────────────────────────────────────────────────────

def _write(path, value):
    try:
        with open(path, "w") as f:
            f.write(str(value))
        return True
    except OSError:
        return False


def vibrate_shutdown():
    """2 long pulses = 'shutdown confirmed'."""
    PERIOD_NS = 10_000_000
    DUTY_NS   =  5_000_000

    chip = None
    for c in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
        target = os.path.realpath(c)
        if "febe0000" in target or "pwm4" in target:
            chip = c
            break
    chip = chip or "/sys/class/pwm/pwmchip0"

    pwm = f"{chip}/pwm0"
    if not os.path.isdir(pwm):
        _write(f"{chip}/export", 0)
        time.sleep(0.1)

    if os.path.isdir(pwm):
        _write(f"{pwm}/period",     PERIOD_NS)
        _write(f"{pwm}/duty_cycle", DUTY_NS)
        for _ in range(2):
            _write(f"{pwm}/enable", 1)
            time.sleep(0.4)
            _write(f"{pwm}/enable", 0)
            time.sleep(0.15)
        _write(f"{chip}/unexport", 0)
    else:
        GPIO = 21
        _write("/sys/class/gpio/export", GPIO)
        _write(f"/sys/class/gpio/gpio{GPIO}/direction", "out")
        for _ in range(2):
            _write(f"/sys/class/gpio/gpio{GPIO}/value", 1)
            time.sleep(0.4)
            _write(f"/sys/class/gpio/gpio{GPIO}/value", 0)
            time.sleep(0.15)
        _write("/sys/class/gpio/unexport", GPIO)


# ── main loop ────────────────────────────────────────────────────────────────

def open_all_devices():
    fds = {}
    for path in sorted(glob.glob("/dev/input/event*")):
        try:
            f = open(path, "rb")
            fds[f.fileno()] = (f, path)
            print(f"power-button-watch: opened {path}", flush=True)
        except Exception as e:
            print(f"power-button-watch: cannot open {path}: {e}", flush=True)
    return fds


def main():
    print(f"power-button-watch: EVENT_SIZE={EVENT_SIZE} HOLD={HOLD_SECS}s", flush=True)

    fds = open_all_devices()
    if not fds:
        print("power-button-watch: no input devices — retrying in 10s", flush=True)
        time.sleep(10)
        return

    press_time = None
    triggered = False   # avoid double-triggering if button held past 3s

    while fds:
        # Poll every 100 ms so we can detect the hold threshold mid-press
        timeout = 0.1 if press_time is not None else 1.0
        try:
            r, _, _ = select.select(list(fds.keys()), [], [], timeout)
        except Exception as e:
            print(f"power-button-watch: select error {e}", flush=True)
            break

        # Check hold threshold even with no new event
        if press_time is not None and not triggered:
            held = time.monotonic() - press_time
            if held >= HOLD_SECS:
                triggered = True
                print(f"power-button-watch: {held:.2f}s hold — shutdown", flush=True)
                vibrate_shutdown()
                subprocess.run(["systemctl", "poweroff"], check=False)

        for fd in r:
            f, path = fds[fd]
            try:
                data = f.read(EVENT_SIZE)
            except OSError as e:
                print(f"power-button-watch: read error on {path}: {e}", flush=True)
                f.close()
                del fds[fd]
                continue

            if len(data) < EVENT_SIZE:
                continue

            _, _, ev_type, code, value = struct.unpack(EVENT_FMT, data)

            if ev_type != EV_KEY or code != KEY_POWER:
                continue

            print(f"power-button-watch: KEY_POWER on {path} value={value}", flush=True)

            if value == 1:      # pressed
                press_time = time.monotonic()
                triggered = False

            elif value == 0:    # released
                if press_time is not None and not triggered:
                    held = time.monotonic() - press_time
                    remaining = HOLD_SECS - held
                    print(f"power-button-watch: released after {held:.2f}s — need {remaining:.1f}s more", flush=True)
                press_time = None


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"power-button-watch: unhandled error {e} — retrying in 5s", flush=True)
            time.sleep(5)
