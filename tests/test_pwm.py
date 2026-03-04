#!/usr/bin/env python3
"""
Test vibration motor on GPIO4_B3 (PWM7-M1 in Radxa BSP).

The Radxa BSP kernel maps the PWM controller at 0xfebd0030 as "pwm7"
(not "pwm15" -- that label points to a different controller at 0xfebf0030).

This script:
  1. Releases GPIO 139 if the init service holds it
  2. Finds the correct pwmchip for febd0030
  3. Runs PWM tests with duty cycle sweeps and patterns

Run with: sudo python3 test_pwm.py
"""

import os
import sys
import time
import glob

PWM_REG_ADDR = "febd0030"
GPIO_NUM = 139


def release_gpio():
    """Release GPIO 139 so PWM pinctrl can claim the pin."""
    gpio_path = "/sys/class/gpio/gpio{}".format(GPIO_NUM)
    if os.path.exists(gpio_path):
        print("[INFO] Releasing GPIO {} from sysfs...".format(GPIO_NUM))
        with open("/sys/class/gpio/unexport", "w") as f:
            f.write(str(GPIO_NUM))
        time.sleep(0.1)


def find_pwm_chip():
    """Find the pwmchip sysfs path for the controller at febd0030."""
    for chip_path in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
        device_link = os.path.join(chip_path, "device")
        try:
            real_path = os.readlink(device_link)
            if PWM_REG_ADDR in real_path:
                return chip_path
        except (OSError, FileNotFoundError):
            pass

        uevent_path = os.path.join(chip_path, "device", "uevent")
        try:
            with open(uevent_path) as f:
                if PWM_REG_ADDR in f.read():
                    return chip_path
        except FileNotFoundError:
            pass

    return None


def try_gpio_fallback():
    """If PWM isn't available, fall back to GPIO on/off control."""
    print("[WARN] PWM chip not available. Falling back to GPIO on/off mode.")
    print("[INFO] (No variable intensity -- just on/off)")

    gpio_path = "/sys/class/gpio/gpio{}".format(GPIO_NUM)
    if not os.path.exists(gpio_path):
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(GPIO_NUM))
        time.sleep(0.05)

    with open(os.path.join(gpio_path, "direction"), "w") as f:
        f.write("out")

    val_path = os.path.join(gpio_path, "value")

    def motor_on():
        with open(val_path, "w") as f:
            f.write("1")

    def motor_off():
        with open(val_path, "w") as f:
            f.write("0")

    print("\n--- GPIO Test: Basic On/Off ---")
    print("  Motor ON ... (2 seconds)")
    motor_on()
    time.sleep(2)
    motor_off()
    print("  Motor OFF")
    time.sleep(0.5)

    print("\n--- GPIO Test: 3 Pulse Pattern ---")
    for i in range(3):
        motor_on()
        time.sleep(0.2)
        motor_off()
        time.sleep(0.3)
        print("  Pulse {}/3".format(i + 1))

    print("\n[INFO] GPIO tests complete. Motor OFF.")
    return True


class PWMMotor:
    def __init__(self, chip_path):
        self.chip_path = chip_path
        self.pwm_path = os.path.join(chip_path, "pwm0")
        print("[INFO] Found PWM at {}".format(chip_path))

        if not os.path.exists(self.pwm_path):
            with open(os.path.join(chip_path, "export"), "w") as f:
                f.write("0")
            time.sleep(0.1)
        print("[INFO] PWM channel at {}".format(self.pwm_path))

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


def test_basic(pwm):
    print("\n--- Test 1: Basic On/Off (500Hz, 50% duty) ---")
    pwm.set_frequency(500)
    pwm.set_duty(50)
    pwm.enable()
    print("  Motor ON at 50% ... (2 seconds)")
    time.sleep(2)
    pwm.disable()
    print("  Motor OFF")
    time.sleep(0.5)


def test_duty_sweep(pwm):
    print("\n--- Test 2: Duty Cycle Sweep (0% -> 100% -> 0%) ---")
    pwm.set_frequency(500)
    pwm.enable()

    print("  Ramping up...")
    for duty in range(0, 101, 5):
        pwm.set_duty(duty)
        sys.stdout.write("    {}%   \r".format(duty))
        sys.stdout.flush()
        time.sleep(0.15)

    print("  Ramping down...")
    for duty in range(100, -1, -5):
        pwm.set_duty(duty)
        sys.stdout.write("    {}%   \r".format(duty))
        sys.stdout.flush()
        time.sleep(0.15)

    pwm.disable()
    print("  Sweep complete           ")


def test_frequency_sweep(pwm):
    print("\n--- Test 3: Frequency Sweep ---")
    for freq in [100, 250, 500, 1000, 2000, 5000]:
        pwm.set_frequency(freq)
        pwm.set_duty(50)
        pwm.enable()
        print("  {} Hz at 50% ... (1 second)".format(freq))
        time.sleep(1)
        pwm.disable()
        time.sleep(0.3)
    print("  Frequency sweep complete")


def test_pulses(pwm):
    print("\n--- Test 4: Pulse Pattern (3 short bursts) ---")
    pwm.set_frequency(500)
    for i in range(3):
        pwm.set_duty(80)
        pwm.enable()
        time.sleep(0.2)
        pwm.disable()
        time.sleep(0.3)
        print("  Pulse {}/3".format(i + 1))
    print("  Pattern complete")


def main():
    if os.geteuid() != 0:
        print("Run with: sudo python3 test_pwm.py")
        sys.exit(1)

    print("=" * 45)
    print(" Vibration Motor Test")
    print(" Pin: GPIO4_B3 (PWM7-M1, reg {})".format(PWM_REG_ADDR))
    print("=" * 45)

    release_gpio()

    chip_path = find_pwm_chip()
    if not chip_path:
        print("[WARN] PWM chip for {} not found in /sys/class/pwm/".format(PWM_REG_ADDR))
        print("[INFO] Available chips:")
        for p in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
            try:
                real = os.readlink(os.path.join(p, "device"))
                print("       {} -> {}".format(os.path.basename(p), real))
            except OSError:
                print("       {}".format(p))

        try_gpio_fallback()
        return

    pwm = PWMMotor(chip_path)

    try:
        test_basic(pwm)
        test_duty_sweep(pwm)
        test_frequency_sweep(pwm)
        test_pulses(pwm)

        print("\n" + "=" * 45)
        print(" All tests complete!")
        print("=" * 45)
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as e:
        print("\n[ERROR] {}".format(e))
        print("[INFO] Falling back to GPIO mode...")
        try_gpio_fallback()
    finally:
        try:
            pwm.disable()
        except Exception:
            pass
        print("[INFO] Motor stopped.")


if __name__ == "__main__":
    main()
