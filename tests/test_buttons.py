#!/usr/bin/env python3
"""
Test GPIO buttons and charger interrupt.

  LANG_BTN   - GPIO0_D0 (pin 24) - Language select button
  OCR_BTN    - GPIO0_C7 (pin 23) - OCR trigger button
  CHG_INT0_L - GPIO0_D3 (pin 27) - Charger interrupt

Tests both direct GPIO reading (gpioget) and gpio-keys input events.

Run with: sudo python3 test_buttons.py
"""

import os
import sys
import time
import select
import subprocess
import struct

BUTTONS = {
    "LANG_BTN":   {"chip": "gpiochip0", "line": 24, "gpio": "GPIO0_D0", "desc": "Language select"},
    "OCR_BTN":    {"chip": "gpiochip0", "line": 23, "gpio": "GPIO0_C7", "desc": "OCR trigger"},
    "CHG_INT0_L": {"chip": "gpiochip0", "line": 27, "gpio": "GPIO0_D3", "desc": "Charger interrupt"},
}

# linux/input-event-codes.h
EV_KEY = 0x01
EV_SYN = 0x00
INPUT_EVENT_SIZE = struct.calcsize("llHHI")


def gpioget(chip, line):
    """Read GPIO value using gpioget command."""
    try:
        result = subprocess.run(
            ["gpioget", chip, str(line)],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return None


def find_input_devices():
    """Find gpio-keys input devices from /proc/bus/input/devices."""
    devices = {}
    try:
        with open("/proc/bus/input/devices") as f:
            content = f.read()

        blocks = content.split("\n\n")
        for block in blocks:
            if "smart-eye" not in block.lower():
                continue
            name = ""
            handlers = ""
            for line in block.split("\n"):
                if line.startswith("N: Name="):
                    name = line.split('"')[1]
                if line.startswith("H: Handlers="):
                    handlers = line.split("=")[1]

            for handler in handlers.split():
                if handler.startswith("event"):
                    path = "/dev/input/" + handler
                    devices[name] = path
    except Exception:
        pass
    return devices


def test_gpio_read():
    """Test 1: Direct GPIO read of all buttons."""
    print("\n--- Test 1: Direct GPIO Read ---")

    for name, info in BUTTONS.items():
        val = gpioget(info["chip"], info["line"])
        if val is not None:
            state = "RELEASED (high)" if val == "1" else "PRESSED (low)"
            print("  {} ({}) = {} - {}".format(name, info["gpio"], val, state))
            print("  [PASS] {} readable".format(name))
        else:
            print("  {} ({}) - could not read".format(name, info["gpio"]))
            print("  [WARN] Pin may be claimed by gpio-keys driver (this is OK)")


def test_button_press():
    """Test 2: Interactive button press test via GPIO polling."""
    print("\n--- Test 2: Button Press Test (GPIO polling, 15s) ---")
    print("  Press each button to test. Ctrl+C to skip.")
    print()

    detected = set()
    start = time.time()
    timeout = 15

    prev_states = {}
    for name, info in BUTTONS.items():
        val = gpioget(info["chip"], info["line"])
        prev_states[name] = val

    try:
        while time.time() - start < timeout:
            remaining = timeout - (time.time() - start)
            print("  Waiting for button press... ({:.0f}s remaining)  ".format(remaining), end="\r")

            for name, info in BUTTONS.items():
                val = gpioget(info["chip"], info["line"])
                if val is None:
                    continue

                if prev_states.get(name) == "1" and val == "0":
                    print("\n  >> {} PRESSED  ({})".format(name, info["desc"]))
                    detected.add(name)
                elif prev_states.get(name) == "0" and val == "1":
                    print("  >> {} RELEASED".format(name))

                prev_states[name] = val

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n  Skipped.")
        return

    print("                                                      ")
    if detected:
        print("  Buttons detected: {}".format(", ".join(sorted(detected))))
        for name in BUTTONS:
            if name in detected:
                print("  [PASS] {} works".format(name))
            else:
                print("  [INFO] {} not pressed during test".format(name))
    else:
        print("  [INFO] No button presses detected (buttons may be claimed by gpio-keys)")


def test_input_devices():
    """Test 3: Check gpio-keys input device registration."""
    print("\n--- Test 3: Input Device Registration ---")

    devices = find_input_devices()
    if devices:
        for name, path in devices.items():
            exists = os.path.exists(path)
            status = "exists" if exists else "MISSING"
            print("  '{}' -> {} ({})".format(name, path, status))
        print("  [PASS] {} gpio-keys input device(s) found".format(len(devices)))
    else:
        print("  [WARN] No 'smart-eye' input devices in /proc/bus/input/devices")
        print("         This may be normal if gpio-keys driver did not load")

    return devices


def test_input_events(devices, duration=15):
    """Test 4: Read actual input events from gpio-keys."""
    if not devices:
        print("\n--- Test 4: Input Events (SKIPPED - no devices) ---")
        return

    print("\n--- Test 4: Input Events ({}s, press buttons now) ---".format(duration))

    fds = {}
    for name, path in devices.items():
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            fds[fd] = name
            print("  Opened {}".format(path))
        except PermissionError:
            print("  [WARN] Cannot open {} (need root)".format(path))
        except Exception as e:
            print("  [WARN] Cannot open {}: {}".format(path, e))

    if not fds:
        print("  [FAIL] No input devices could be opened")
        return

    events_received = 0
    start = time.time()

    try:
        while time.time() - start < duration:
            remaining = duration - (time.time() - start)
            print("  Listening... ({:.0f}s remaining)  ".format(remaining), end="\r")

            readable, _, _ = select.select(list(fds.keys()), [], [], 0.5)
            for fd in readable:
                try:
                    data = os.read(fd, INPUT_EVENT_SIZE * 8)
                    offset = 0
                    while offset + INPUT_EVENT_SIZE <= len(data):
                        tv_sec, tv_usec, ev_type, ev_code, ev_value = struct.unpack_from(
                            "llHHI", data, offset
                        )
                        offset += INPUT_EVENT_SIZE

                        if ev_type == EV_KEY:
                            dev_name = fds[fd]
                            if ev_value == 1:
                                state = "PRESSED"
                            elif ev_value == 0:
                                state = "RELEASED"
                            else:
                                state = "REPEAT"
                            print("\n  >> [{}] code=0x{:03X} {}".format(dev_name, ev_code, state))
                            events_received += 1
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n  Skipped.")

    print("                                          ")
    for fd in fds:
        os.close(fd)

    if events_received > 0:
        print("  [PASS] Received {} input events".format(events_received))
    else:
        print("  [INFO] No input events received")


def test_charger_interrupt():
    """Test 5: Monitor charger interrupt pin."""
    print("\n--- Test 5: Charger Interrupt (CHG_INT0_L) ---")

    info = BUTTONS["CHG_INT0_L"]
    val = gpioget(info["chip"], info["line"])

    if val is None:
        print("  [WARN] Cannot read CHG_INT0_L directly (claimed by gpio-keys)")
        return

    if val == "1":
        print("  CHG_INT0_L = 1 (HIGH) - No charger interrupt active")
        print("  [PASS] Pin is idle (expected when not charging or no charger IC)")
    else:
        print("  CHG_INT0_L = 0 (LOW) - Charger interrupt ACTIVE")
        print("  [INFO] Charger IC is signaling an event")

    print("\n  Monitoring for 5 seconds...")
    transitions = 0
    prev = val
    start = time.time()
    try:
        while time.time() - start < 5:
            current = gpioget(info["chip"], info["line"])
            if current != prev and current is not None:
                transitions += 1
                ts = time.time() - start
                print("  [{:.2f}s] CHG_INT0_L: {} -> {}".format(ts, prev, current))
                prev = current
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    if transitions == 0:
        print("  [PASS] No spurious transitions (stable)")
    else:
        print("  [INFO] {} transitions detected".format(transitions))


def main():
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 test_buttons.py")
        sys.exit(1)

    print("========================================")
    print(" GPIO Buttons & Charger Interrupt Test")
    print("========================================")
    print(" LANG_BTN   = GPIO0_D0 (pin 24)")
    print(" OCR_BTN    = GPIO0_C7 (pin 23)")
    print(" CHG_INT0_L = GPIO0_D3 (pin 27)")
    print("========================================")

    try:
        test_gpio_read()
        test_button_press()
        devices = test_input_devices()
        test_input_events(devices)
        test_charger_interrupt()

        print("\n========================================")
        print(" All button tests complete!")
        print("========================================")
    except KeyboardInterrupt:
        print("\n\nInterrupted!")


if __name__ == "__main__":
    main()
