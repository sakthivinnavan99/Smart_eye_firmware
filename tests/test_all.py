#!/usr/bin/env python3
"""
Smart Eye Carrier Board - Run All Peripheral Tests

Runs the overlay verification and all peripheral tests sequentially.
Use --auto to skip interactive prompts (for CI/automated testing).

Run with: sudo python3 test_all.py [--auto]
"""

import os
import sys
import subprocess
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    {
        "name": "Overlay Verification",
        "cmd": ["bash", os.path.join(TESTS_DIR, "test_overlay.sh")],
        "interactive": False,
    },
    {
        "name": "PWM15 Vibration Motor",
        "cmd": [sys.executable, os.path.join(TESTS_DIR, "test_pwm.py")],
        "interactive": True,
    },
    {
        "name": "BQ27220 Fuel Gauge (I2C3)",
        "cmd": [sys.executable, os.path.join(TESTS_DIR, "test_i2c_bq27220.py")],
        "interactive": False,
    },
    {
        "name": "UART6 Serial Port",
        "cmd": [sys.executable, os.path.join(TESTS_DIR, "test_uart6.py")],
        "interactive": True,
    },
    {
        "name": "GPIO Buttons & Charger Interrupt",
        "cmd": [sys.executable, os.path.join(TESTS_DIR, "test_buttons.py")],
        "interactive": True,
    },
    {
        "name": "RPi Camera v2 (IMX219)",
        "cmd": [sys.executable, os.path.join(TESTS_DIR, "test_camera.py")],
        "interactive": False,
    },
]


def run_test(test, auto_mode=False):
    name = test["name"]
    cmd = test["cmd"]
    interactive = test["interactive"]

    print("\n")
    print("=" * 50)
    print(" TEST: {}".format(name))
    print("=" * 50)

    if auto_mode and interactive:
        print("  [SKIP] Skipped in --auto mode (requires interaction)")
        return "skip"

    try:
        result = subprocess.run(cmd, timeout=120)
        return "pass" if result.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        print("  [FAIL] Test timed out after 120 seconds")
        return "fail"
    except FileNotFoundError:
        print("  [FAIL] Test file not found: {}".format(cmd))
        return "fail"
    except KeyboardInterrupt:
        print("\n  [SKIP] Interrupted by user")
        return "skip"


def main():
    auto_mode = "--auto" in sys.argv

    if os.geteuid() != 0:
        print("This test suite requires root.")
        print("Run with: sudo python3 test_all.py [--auto]")
        sys.exit(1)

    print("=" * 50)
    print(" Smart Eye Carrier Board - Full Test Suite")
    print("=" * 50)
    print(" Platform: Radxa CM5 on custom carrier PCB")
    print(" Mode:     {}".format("Automated (--auto)" if auto_mode else "Interactive"))
    print(" Tests:    {}".format(len(TESTS)))
    print("=" * 50)

    if not auto_mode:
        print("\n  Interactive tests will prompt for input.")
        print("  Use --auto to skip interactive tests.")
        try:
            input("\n  Press Enter to begin... ")
        except KeyboardInterrupt:
            print("\n\nAborted.")
            sys.exit(0)

    results = {}
    for test in TESTS:
        results[test["name"]] = run_test(test, auto_mode)

    print("\n")
    print("=" * 50)
    print(" TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result == "pass":
            icon = "PASS"
            passed += 1
        elif result == "fail":
            icon = "FAIL"
            failed += 1
        else:
            icon = "SKIP"
            skipped += 1
        print("  [{}] {}".format(icon, name))

    print()
    print("  Total: {} passed, {} failed, {} skipped".format(passed, failed, skipped))
    print("=" * 50)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
