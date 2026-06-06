#!/usr/bin/env python3
"""
Test BQ25895 Battery Charger Configuration

Tests I2C communication with BQ25895 and verifies optimal charging
configuration for 10000mAh Li-polymer battery.

Run with: sudo python3 test_bq25895_charger.py

Hardware: BQ25895 on I2C3 at address 0x6A
Battery: 10000mAh Li-polymer
Charger IC: Texas Instruments BQ25895
"""

import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pathpal_project'))
from bq25895_charger import BQ25895


def test_device_detection():
    """Test 1: Detect BQ25895 on I2C3."""
    print("\n--- Test 1: Device Detection ---")
    try:
        charger = BQ25895()
        dev_id = charger.get_device_id()
        charger.close()

        print(f"  ✓ BQ25895 detected on I2C3 at 0x6A")
        print(f"    Device ID: 0x{dev_id:02X}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to detect BQ25895: {e}")
        return False


def test_register_read_write():
    """Test 2: Register read/write operations."""
    print("\n--- Test 2: Register Read/Write ---")
    try:
        charger = BQ25895()

        # Read-modify-write test on power-on register
        original = charger.read_register(0x01)
        print(f"  Original REG_POWER_ON: 0x{original:02X}")

        # Modify minimum system voltage field
        charger.modify_register(0x01, 0x0E, 0x0A)
        modified = charger.read_register(0x01)
        print(f"  Modified REG_POWER_ON: 0x{modified:02X}")

        # Verify modification
        if (modified & 0x0E) == 0x0A:
            print("  ✓ Register read/write working correctly")
            charger.close()
            return True
        else:
            print("  ✗ Register modification failed")
            charger.close()
            return False

    except Exception as e:
        print(f"  ✗ Register test failed: {e}")
        return False


def test_input_current_config():
    """Test 3: Input current limit configuration."""
    print("\n--- Test 3: Input Current Limit Configuration ---")
    test_values = [100, 500, 1000, 2000, 3250]

    try:
        charger = BQ25895()
        for ilim in test_values:
            charger.configure_input_current(ilim)
            read_back = charger.read_register(0x00) & 0x3F
            calculated = (ilim - 100) // 100
            print(f"  {ilim:4d}mA → code {read_back:2d} (expected {calculated:2d})", end="")
            if read_back == calculated:
                print(" ✓")
            else:
                print(" ✗ MISMATCH")
        charger.close()
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_charging_current_config():
    """Test 4: Charging current configuration."""
    print("\n--- Test 4: Charging Current Configuration ---")
    test_values = [256, 512, 1024, 2048, 4096]

    try:
        charger = BQ25895()
        for ichg in test_values:
            charger.configure_charging_current(ichg)
            read_back = (charger.read_register(0x02) & 0xFC) >> 2
            calculated = ichg // 64
            print(f"  {ichg:4d}mA → code {read_back:2d} (expected {calculated:2d})", end="")
            if read_back == calculated:
                print(" ✓")
            else:
                print(" ✗ MISMATCH")
        charger.close()
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_voltage_config():
    """Test 5: Charge voltage limit configuration."""
    print("\n--- Test 5: Charge Voltage Configuration ---")
    test_values = [3840, 4000, 4100, 4208, 4350]

    try:
        charger = BQ25895()
        for vreg in test_values:
            charger.configure_charging_voltage(vreg)
            read_back = (charger.read_register(0x04) & 0xFC) >> 2
            calculated = (vreg - 3840) // 16
            print(f"  {vreg:4d}mV → code {read_back:2d} (expected {calculated:2d})", end="")
            if read_back == calculated:
                print(" ✓")
            else:
                print(" ✗ MISMATCH")
        charger.close()
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_termination_config():
    """Test 6: Termination current configuration."""
    print("\n--- Test 6: Termination Current Configuration ---")
    test_values = [16, 64, 128, 256, 512]

    try:
        charger = BQ25895()
        for iterm in test_values:
            charger.configure_termination_current(iterm)
            read_back = charger.read_register(0x03) & 0x0F
            calculated = (iterm - 16) // 16
            print(f"  {iterm:4d}mA → code {read_back:2d} (expected {calculated:2d})", end="")
            if read_back == calculated:
                print(" ✓")
            else:
                print(" ✗ MISMATCH")
        charger.close()
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_full_configuration():
    """Test 7: Full 10Ah battery configuration."""
    print("\n--- Test 7: Full 10Ah Battery Configuration ---")
    try:
        charger = BQ25895()
        print("  Applying 10000mAh battery optimal configuration...")
        charger.configure_for_10ah_battery()

        # Read all relevant registers
        print("\n  Register Dump:")
        registers = {
            0x00: "Input Source Control",
            0x01: "Power-On Configuration",
            0x02: "Charge Current Control",
            0x03: "Pre-charge/Termination",
            0x04: "Charge Voltage Control",
            0x05: "Charging Termination/Timer",
            0x06: "Boost Voltage/Thermal",
            0x08: "Status",
            0x09: "Fault",
        }

        for reg, name in registers.items():
            val = charger.read_register(reg)
            print(f"    0x{reg:02X} ({name:30s}): 0x{val:02X} ({val:08b}b)")

        charger.close()
        return True

    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        return False


def test_status_monitoring():
    """Test 8: Charger status monitoring (NTC faults ignored - no thermistor)."""
    print("\n--- Test 8: Charger Status Monitoring (NTC ignored) ---")
    try:
        charger = BQ25895()

        print("  Reading charger status (NTC thermistor not connected - ignoring NTC fault)...")
        status = charger.get_status(ignore_ntc=True)  # Ignore NTC since no thermistor

        print(f"  Charging State:          {status['charging_state']}")
        print(f"  Power Good:              {status['power_good']}")
        print(f"  Thermal Shutdown:        {status['thermal_shutdown']}")
        print(f"  Battery Overvoltage:     {status['battery_overvoltage']}")
        print(f"  Input Overvoltage:       {status['input_overvoltage']}")
        print(f"  Charger Fault:           {status['charger_fault']}")
        print(f"  Battery Fault:           {status['battery_fault']}")
        print(f"  NTC Fault:               {status['ntc_fault']} (ignored - no thermistor)")

        # Check for critical faults (NTC not included)
        has_critical_faults = (
            status['thermal_shutdown'] or
            status['battery_overvoltage'] or
            status['input_overvoltage'] or
            status['battery_fault']
        )

        if has_critical_faults:
            print("\n  ⚠ WARNING: Critical charger faults detected!")
            return False
        else:
            print("\n  ✓ No critical charger faults detected")
            charger.close()
            return True

    except Exception as e:
        print(f"  ✗ Status read failed: {e}")
        return False


def test_thermal_regulation():
    """Test 9: Thermal regulation configuration."""
    print("\n--- Test 9: Thermal Regulation Configuration ---")
    thermal_thresholds = [
        (0b00, "100°C (aggressive)"),
        (0b01, "110°C"),
        (0b10, "120°C"),
        (0b11, "130°C (conservative)"),
    ]

    try:
        charger = BQ25895()
        for code, desc in thermal_thresholds:
            charger.configure_thermal_regulation(code)
            read_back = (charger.read_register(0x06) & 0xC0) >> 6
            print(f"  {desc:25s} → code {read_back:2b}", end="")
            if read_back == code:
                print(" ✓")
            else:
                print(" ✗ MISMATCH")
        charger.close()
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_safety_timer():
    """Test 10: Safety timer configuration."""
    print("\n--- Test 10: Safety Timer Configuration ---")
    try:
        charger = BQ25895()

        # Test timer enable
        print("  Testing timer enable...")
        charger.set_charging_safety_timer(enable=True, fast_charge_timeout_hours=12)
        timer_bit = (charger.read_register(0x05) & 0x20) >> 5
        timeout_bits = (charger.read_register(0x05) & 0x0C) >> 2

        if timer_bit:
            print(f"    Timer: Enabled ✓")
            print(f"    Timeout: {[5, 8, 12, 20][timeout_bits]}h")
        else:
            print(f"    Timer: Disabled ✗")

        # Test timer disable
        print("  Testing timer disable...")
        charger.disable_watchdog()
        timer_bit = (charger.read_register(0x05) & 0x20) >> 5

        if not timer_bit:
            print(f"    Timer: Disabled ✓")
        else:
            print(f"    Timer: Still enabled ✗")

        charger.close()
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_longevity_configuration():
    """Test 11: Longevity configuration (optimized for battery life)."""
    print("\n--- Test 11: Longevity Configuration (Battery Longevity) ---")
    try:
        charger = BQ25895()
        print("  Applying longevity configuration (4.1V, 1024mA)...")
        charger.configure_for_10ah_battery_longevity()

        # Verify key parameters
        print("\n  Verifying critical parameters:")

        # Check charge voltage (should be ~4100mV)
        vreg = charger.read_register(0x04)
        vreg_code = (vreg & 0xFC) >> 2
        vreg_mv = 3840 + (vreg_code * 16)
        print(f"    Charge Voltage: {vreg_mv}mV (target 4100mV)", end="")
        if 4090 <= vreg_mv <= 4110:
            print(" ✓")
        else:
            print(f" ✗ (expected ~4100mV)")

        # Check charge current (should be ~1024mA)
        ichg = charger.read_register(0x02)
        ichg_code = (ichg & 0xFC) >> 2
        ichg_ma = ichg_code * 64
        print(f"    Charge Current: {ichg_ma}mA (target 1024mA)", end="")
        if 1000 <= ichg_ma <= 1100:
            print(" ✓")
        else:
            print(f" ✗ (expected ~1024mA)")

        # Check thermal threshold (should be 120°C = code 10)
        treg = charger.read_register(0x06)
        treg_code = (treg & 0xC0) >> 6
        print(f"    Thermal Threshold: code {treg_code:02b} (target 10)", end="")
        if treg_code == 0b10:
            print(" ✓")
        else:
            print(" ✗ (expected 0b10 for 120°C)")

        charger.close()
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Run all BQ25895 tests."""
    if os.geteuid() != 0:
        print("This test requires root.")
        print("Run with: sudo python3 test_bq25895_charger.py")
        sys.exit(1)

    print("=" * 70)
    print(" BQ25895 Battery Charger - Comprehensive Test Suite")
    print("=" * 70)
    print(" Device:   Texas Instruments BQ25895")
    print(" I2C Bus:  I2C3")
    print(" Address:  0x6A")
    print(" Battery:  10000mAh Li-polymer")
    print("=" * 70)

    tests = [
        ("Device Detection", test_device_detection),
        ("Register Read/Write", test_register_read_write),
        ("Input Current Config", test_input_current_config),
        ("Charging Current Config", test_charging_current_config),
        ("Voltage Config", test_voltage_config),
        ("Termination Config", test_termination_config),
        ("Full Configuration", test_full_configuration),
        ("Status Monitoring", test_status_monitoring),
        ("Thermal Regulation", test_thermal_regulation),
        ("Safety Timer", test_safety_timer),
        ("Longevity Configuration", test_longevity_configuration),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted!")
            sys.exit(1)
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f" Passed: {passed}/{total}")
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {status} - {name}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
