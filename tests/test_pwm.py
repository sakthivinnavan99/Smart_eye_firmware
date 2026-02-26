#!/usr/bin/env python3
"""
Test PWM15 vibration motor on GPIO4_B3.

Finds the correct pwmchip for PWM15 (register 0xfebf0030),
then runs through duty cycle sweeps and patterns.

Run with: sudo python3 test_pwm.py
"""

import os
import sys
import time
import glob


def find_pwm15_chip():
    """Find the pwmchip sysfs path for PWM15 (0xfebf0030)."""
    for chip_path in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
        uevent_path = os.path.join(chip_path, "device", "uevent")
        try:
            with open(uevent_path) as f:
                if "febf0030" in f.read():
                    return chip_path
        except FileNotFoundError:
            continue
    return None


class PWM15:
    def __init__(self):
        self.chip_path = find_pwm15_chip()
        if not self.chip_path:
            print("[FAIL] PWM15 (0xfebf0030) not found.")
            print("       Is the overlay installed and enabled? Reboot required.")
            sys.exit(1)

        self.pwm_path = os.path.join(self.chip_path, "pwm0")
        print(f"[INFO] Found PWM15 at {self.chip_path}")

        if not os.path.exists(self.pwm_path):
            with open(os.path.join(self.chip_path, "export"), "w") as f:
                f.write("0")
            time.sleep(0.1)
        print(f"[INFO] PWM channel at {self.pwm_path}")

    def set_frequency(self, freq_hz):
        period_ns = int(1e9 / freq_hz)
        self._write("period", str(period_ns))

    def set_duty(self, percent):
        period = int(self._read("period"))
        duty_ns = int(period * max(0, min(100, percent)) / 100)
        self._write("duty_cycle", str(duty_ns))

    def enable(self):
        self._write("enable", "1")

    def disable(self):
        self._write("duty_cycle", "0")
        self._write("enable", "0")

    def _write(self, attr, value):
        with open(os.path.join(self.pwm_path, attr), "w") as f:
            f.write(value)

    def _read(self, attr):
        with open(os.path.join(self.pwm_path, attr)) as f:
            return f.read().strip()


def test_basic_on_off(pwm):
    """Test 1: Basic on/off at 50% duty."""
    print("\n--- Test 1: Basic On/Off (500Hz, 50% duty) ---")
    pwm.set_frequency(500)
    pwm.set_duty(50)
    pwm.enable()
    print("  Motor ON at 50% ... (2 seconds)")
    time.sleep(2)
    pwm.disable()
    print("  Motor OFF")
    time.sleep(0.5)
    input("  >> Did the motor vibrate? [Enter to continue] ")


def test_duty_sweep(pwm):
    """Test 2: Sweep duty cycle 0-100-0%."""
    print("\n--- Test 2: Duty Cycle Sweep (0% -> 100% -> 0%) ---")
    pwm.set_frequency(500)
    pwm.enable()

    print("  Ramping up...")
    for duty in range(0, 101, 5):
        pwm.set_duty(duty)
        print(f"    {duty:3d}%", end="\r")
        time.sleep(0.15)

    print("  Ramping down...")
    for duty in range(100, -1, -5):
        pwm.set_duty(duty)
        print(f"    {duty:3d}%", end="\r")
        time.sleep(0.15)

    pwm.disable()
    print("  Sweep complete           ")
    input("  >> Did intensity change smoothly? [Enter to continue] ")


def test_frequency_sweep(pwm):
    """Test 3: Test different frequencies."""
    print("\n--- Test 3: Frequency Sweep ---")
    freqs = [100, 250, 500, 1000, 2000, 5000]

    for freq in freqs:
        pwm.set_frequency(freq)
        pwm.set_duty(50)
        pwm.enable()
        print(f"  {freq:5d} Hz at 50% ... (1 second)")
        time.sleep(1)
        pwm.disable()
        time.sleep(0.3)

    print("  Frequency sweep complete")
    input("  >> Did you feel different vibration patterns? [Enter to continue] ")


def test_pulse_pattern(pwm):
    """Test 4: Pulse pattern (3 short bursts)."""
    print("\n--- Test 4: Pulse Pattern (3 short bursts) ---")
    pwm.set_frequency(500)

    for i in range(3):
        pwm.set_duty(80)
        pwm.enable()
        time.sleep(0.2)
        pwm.disable()
        time.sleep(0.3)
        print(f"  Pulse {i+1}/3")

    print("  Pattern complete")
    input("  >> Did you feel 3 distinct pulses? [Enter to continue] ")


def main():
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 test_pwm.py")
        sys.exit(1)

    print("========================================")
    print(" PWM15 Vibration Motor Test")
    print(" Pin: GPIO4_B3 (PWM15-M1)")
    print("========================================")

    pwm = PWM15()

    try:
        test_basic_on_off(pwm)
        test_duty_sweep(pwm)
        test_frequency_sweep(pwm)
        test_pulse_pattern(pwm)

        print("\n========================================")
        print(" All PWM tests complete!")
        print("========================================")
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    finally:
        pwm.disable()
        print("[INFO] Motor stopped, PWM disabled.")


if __name__ == "__main__":
    main()
