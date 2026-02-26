#!/usr/bin/env python3
"""
Test BQ27220YZFR fuel gauge on I2C3 (address 0x55).

Uses raw I2C via /dev/i2c-3 (no smbus2 dependency).
Also tests the interrupt pin GPIO4_A4.

Run with: sudo python3 test_i2c_bq27220.py
"""

import os
import sys
import time
import struct
import fcntl

I2C_BUS = 3
BQ27220_ADDR = 0x55
I2C_SLAVE = 0x0703

REGS = {
    "Control":         0x00,
    "Temperature":     0x06,
    "Voltage":         0x08,
    "Flags":           0x0A,
    "Current":         0x0C,
    "RemainingCap":    0x10,
    "FullChargeCap":   0x12,
    "AverageCurrent":  0x14,
    "StandbyCurrent":  0x18,
    "StateOfCharge":   0x1C,
    "InternalTemp":    0x1E,
    "StateOfHealth":   0x20,
}

GPIO_INT_CHIP = "gpiochip4"
GPIO_INT_LINE = 4


class I2CDev:
    """Minimal I2C access via /dev/i2c-N ioctl."""

    def __init__(self, bus, addr):
        self.path = "/dev/i2c-{}".format(bus)
        self.addr = addr
        self.fd = None

    def open(self):
        self.fd = os.open(self.path, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, self.addr)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_word(self, reg):
        os.write(self.fd, bytes([reg]))
        data = os.read(self.fd, 2)
        if len(data) != 2:
            raise IOError("Short read from reg 0x{:02X}".format(reg))
        return struct.unpack("<H", data)[0]

    def read_word_signed(self, reg):
        raw = self.read_word(reg)
        return struct.unpack("<h", struct.pack("<H", raw))[0]

    def write_word(self, reg, value):
        data = bytes([reg]) + struct.pack("<H", value)
        os.write(self.fd, data)


def test_i2c_bus():
    print("\n--- Test 1: I2C3 Bus ---")
    path = "/dev/i2c-{}".format(I2C_BUS)
    if os.path.exists(path):
        print("  [PASS] {} exists".format(path))
        return True
    else:
        print("  [FAIL] {} not found. Is I2C3 overlay enabled?".format(path))
        return False


def test_device_detect(dev):
    print("\n--- Test 2: BQ27220 Detection (0x{:02X}) ---".format(BQ27220_ADDR))
    try:
        dev.open()
        voltage = dev.read_word(REGS["Voltage"])
        print("  [PASS] BQ27220 responded at 0x{:02X}".format(BQ27220_ADDR))
        print("         Raw voltage register: 0x{:04X} ({} mV)".format(voltage, voltage))
        return True
    except Exception as e:
        print("  [FAIL] Cannot communicate with BQ27220: {}".format(e))
        print("         Check wiring: SDA=GPIO3_C0, SCL=GPIO3_B7")
        print("         Check pull-up resistors on SDA/SCL lines")
        return False


def test_read_registers(dev):
    print("\n--- Test 3: Register Dump ---")
    results = {}
    errors = 0

    for name, reg in REGS.items():
        try:
            if name in ("Current", "AverageCurrent", "StandbyCurrent"):
                raw = dev.read_word_signed(reg)
            else:
                raw = dev.read_word(reg)
            results[name] = raw
            print("  0x{:02X} {:20s} = 0x{:04X} ({})".format(reg, name, raw & 0xFFFF, raw))
        except Exception as e:
            print("  0x{:02X} {:20s} = ERROR: {}".format(reg, name, e))
            errors += 1

    if errors == 0:
        print("  [PASS] All {} registers read successfully".format(len(REGS)))
    else:
        print("  [WARN] {}/{} register reads failed".format(errors, len(REGS)))

    return results


def test_interpreted_values(results):
    print("\n--- Test 4: Interpreted Values ---")

    if "Voltage" in results:
        v = results["Voltage"]
        print("  Battery Voltage:    {} mV ({:.3f} V)".format(v, v / 1000))
        if 2500 <= v <= 4500:
            print("  [PASS] Voltage in valid Li-ion range")
        elif v == 0:
            print("  [WARN] Voltage is 0 - no battery connected?")
        else:
            print("  [WARN] Voltage {} mV outside typical Li-ion range".format(v))

    if "Temperature" in results:
        t_raw = results["Temperature"]
        t_celsius = (t_raw * 0.1) - 273.15
        print("  Temperature:        {:.1f} C (raw: {})".format(t_celsius, t_raw))
        if -10 <= t_celsius <= 60:
            print("  [PASS] Temperature in valid range")
        else:
            print("  [WARN] Temperature {:.1f} C seems unusual".format(t_celsius))

    if "StateOfCharge" in results:
        soc = results["StateOfCharge"]
        print("  State of Charge:    {}%".format(soc))
        if 0 <= soc <= 100:
            print("  [PASS] SoC in valid range")

    if "AverageCurrent" in results:
        ma = results["AverageCurrent"]
        if ma > 0:
            direction = "charging"
        elif ma < 0:
            direction = "discharging"
        else:
            direction = "idle"
        print("  Average Current:    {} mA ({})".format(ma, direction))

    if "RemainingCap" in results:
        print("  Remaining Capacity: {} mAh".format(results["RemainingCap"]))

    if "FullChargeCap" in results:
        print("  Full Charge Cap:    {} mAh".format(results["FullChargeCap"]))

    if "StateOfHealth" in results:
        print("  State of Health:    {}%".format(results["StateOfHealth"]))

    if "Flags" in results:
        flags = results["Flags"]
        print("  Flags:              0x{:04X}".format(flags))
        flag_bits = {
            0: "DSG (discharging)",
            1: "SOCF (SoC final)",
            2: "SOC1 (SoC threshold 1)",
            3: "CHG (charge allowed)",
            4: "OCVFAIL",
            8: "ITPOR (POR detected)",
            9: "CFGUPD (config update mode)",
        }
        for bit, desc in flag_bits.items():
            if flags & (1 << bit):
                print("                      bit {}: {}".format(bit, desc))


def test_device_type(dev):
    print("\n--- Test 5: Device Identification ---")
    try:
        dev.write_word(REGS["Control"], 0x0001)
        time.sleep(0.05)
        device_type = dev.read_word(REGS["Control"])
        print("  Device Type: 0x{:04X}".format(device_type))
        if device_type == 0x0220:
            print("  [PASS] Confirmed BQ27220")
        else:
            print("  [INFO] Device type 0x{:04X} (expected 0x0220 for BQ27220)".format(device_type))
        return True
    except Exception as e:
        print("  [FAIL] Control command failed: {}".format(e))
        return False


def test_interrupt_pin():
    print("\n--- Test 6: Interrupt Pin (GPIO4_A4) ---")
    try:
        stream = os.popen("gpioget {} {} 2>/dev/null".format(GPIO_INT_CHIP, GPIO_INT_LINE))
        val = stream.read().strip()
        stream.close()
        if val in ("0", "1"):
            state = "ACTIVE (low)" if val == "0" else "IDLE (high)"
            print("  GPIO4_A4 value: {} - {}".format(val, state))
            print("  [PASS] Interrupt pin readable")
        else:
            print("  [WARN] Could not read GPIO4_A4 (claimed by kernel driver?)")
    except Exception as e:
        print("  [WARN] Interrupt pin test error: {}".format(e))


def test_continuous_read(dev, duration=5):
    print("\n--- Test 7: Continuous Read ({}s stability test) ---".format(duration))
    readings = []
    errors = 0
    start = time.time()

    while time.time() - start < duration:
        try:
            v = dev.read_word(REGS["Voltage"])
            readings.append(v)
        except Exception:
            errors += 1
        time.sleep(0.5)

    if readings:
        avg = sum(readings) / len(readings)
        vmin = min(readings)
        vmax = max(readings)
        print("  Readings:  {} successful, {} errors".format(len(readings), errors))
        print("  Voltage:   avg={:.0f} mV, min={} mV, max={} mV".format(avg, vmin, vmax))
        print("  Variation: {} mV".format(vmax - vmin))
        if vmax - vmin < 100:
            print("  [PASS] Stable readings (variation < 100 mV)")
        else:
            print("  [WARN] High variation ({} mV)".format(vmax - vmin))
    else:
        print("  [FAIL] No successful readings ({} errors)".format(errors))


def main():
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 test_i2c_bq27220.py")
        sys.exit(1)

    print("========================================")
    print(" BQ27220 Fuel Gauge Test (I2C3)")
    print(" Addr: 0x55")
    print(" SCL:  GPIO3_B7 | SDA: GPIO3_C0")
    print(" INT:  GPIO4_A4")
    print("========================================")

    if not test_i2c_bus():
        sys.exit(1)

    dev = I2CDev(I2C_BUS, BQ27220_ADDR)

    try:
        if not test_device_detect(dev):
            sys.exit(1)

        results = test_read_registers(dev)
        test_interpreted_values(results)
        test_device_type(dev)
        test_interrupt_pin()
        test_continuous_read(dev)

        print("\n========================================")
        print(" All BQ27220 tests complete!")
        print("========================================")
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
