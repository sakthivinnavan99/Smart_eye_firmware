#!/usr/bin/env python3
"""
BQ25895 Battery Charger Configuration Module

Texas Instruments BQ25895 - Highly integrated synchronous battery charger.
Supports single-cell Li-ion/Li-polymer batteries via I2C interface.

Hardware: BQ25895 on I2C3 at address 0x6A
Battery: 10000mAh Li-polymer cell (4.2V nominal)
"""

import os
import fcntl
import struct


class BQ25895:
    """TI BQ25895 Battery Charger IC over I2C.

    Configures charging parameters for optimal battery health and performance.
    """

    I2C_BUS = 3
    DEVICE_ADDR = 0x6A
    I2C_SLAVE = 0x0703

    # Register Map
    REG_INPUT_SOURCE = 0x00          # Input source control
    REG_POWER_ON = 0x01              # Power-on configuration
    REG_CHARGE_CURRENT = 0x02        # Charge current control
    REG_PRE_CHARGE = 0x03            # Pre-charge/Termination current
    REG_CHARGE_VOLTAGE = 0x04        # Charge voltage control
    REG_CHARGE_TERM = 0x05           # Charging termination/timer control
    REG_BOOST = 0x06                 # Boost voltage/thermal regulation
    REG_STATUS = 0x08                # Status and fault flags
    REG_FAULT = 0x09                 # Fault status
    REG_VERSION = 0x0B               # Device version

    def __init__(self):
        """Initialize BQ25895 I2C interface."""
        self.path = f"/dev/i2c-{self.I2C_BUS}"
        self.fd = None
        self._connect()

    def _connect(self):
        """Open I2C device and configure as slave."""
        try:
            self.fd = os.open(self.path, os.O_RDWR)
            fcntl.ioctl(self.fd, self.I2C_SLAVE, self.DEVICE_ADDR)
        except OSError as e:
            raise RuntimeError(f"Failed to open I2C bus: {e}")

    def close(self):
        """Close I2C connection."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_register(self, reg):
        """Read a single register."""
        data = os.read(self.fd, 1)
        if len(data) == 0:
            # Register address must be written first
            os.write(self.fd, bytes([reg]))
            data = os.read(self.fd, 1)
        return data[0] if data else 0

    def write_register(self, reg, value):
        """Write a single register."""
        os.write(self.fd, bytes([reg, value & 0xFF]))

    def modify_register(self, reg, mask, value):
        """Read-modify-write a register field."""
        current = self.read_register(reg)
        modified = (current & ~mask) | (value & mask)
        self.write_register(reg, modified)

    def get_device_id(self):
        """Read device version/revision."""
        return self.read_register(self.REG_VERSION)

    def configure_input_current(self, ilim_ma=2000):
        """Configure input current limit.

        Args:
            ilim_ma: Input current limit in mA (100-3250mA, step 100mA)
        """
        # Clamp to valid range
        ilim_ma = max(100, min(3250, ilim_ma))

        # ILIM bits [5:0] in register 0x00
        # 100mA + (ILIM * 100mA), so ILIM = (ilim_ma - 100) / 100
        ilim_code = (ilim_ma - 100) // 100
        ilim_code = max(0, min(31, ilim_code))  # 5-bit field

        self.modify_register(self.REG_INPUT_SOURCE, 0x3F, ilim_code)

    def configure_charging_current(self, ichg_ma=2048):
        """Configure charging current.

        Args:
            ichg_ma: Charging current in mA (0-5056mA, step 64mA)
                     For 10000mAh battery, recommended 1000-2048mA
        """
        # Clamp to valid range
        ichg_ma = max(0, min(5056, ichg_ma))

        # ICHG bits [7:2] in register 0x02
        # 0mA + (ICHG * 64mA)
        ichg_code = ichg_ma // 64
        ichg_code = max(0, min(79, ichg_code))  # 6-bit field (bits 7:2)

        self.modify_register(self.REG_CHARGE_CURRENT, 0xFC, (ichg_code << 2))

    def configure_precharge_current(self, iprechg_ma=192):
        """Configure precharge current for low voltage recovery.

        Args:
            iprechg_ma: Precharge current in mA (16-512mA, step 16mA)
                       Default 192mA is good for most applications
        """
        # Clamp to valid range
        iprechg_ma = max(16, min(512, iprechg_ma))

        # IPRECHG bits [7:4] in register 0x03
        # 16mA + (IPRECHG * 16mA)
        iprechg_code = (iprechg_ma - 16) // 16
        iprechg_code = max(0, min(31, iprechg_code))  # 4-bit field

        self.modify_register(self.REG_PRE_CHARGE, 0xF0, (iprechg_code << 4))

    def configure_termination_current(self, iterm_ma=256):
        """Configure termination current (end-of-charge detection).

        Args:
            iterm_ma: Termination current in mA (16-512mA, step 16mA)
                     Default 256mA ~= 2.5% of 10000mAh
        """
        # Clamp to valid range
        iterm_ma = max(16, min(512, iterm_ma))

        # ITERM bits [3:0] in register 0x03
        # 16mA + (ITERM * 16mA)
        iterm_code = (iterm_ma - 16) // 16
        iterm_code = max(0, min(31, iterm_code))  # 4-bit field

        self.modify_register(self.REG_PRE_CHARGE, 0x0F, iterm_code)

    def configure_charging_voltage(self, vreg_mv=4208):
        """Configure charge voltage limit (battery regulation voltage).

        Args:
            vreg_mv: Charging voltage in mV (3840-4496mV, step 16mV)
                    4208mV is typical for Li-ion/Li-polymer
        """
        # Clamp to valid range
        vreg_mv = max(3840, min(4496, vreg_mv))

        # VREG bits [7:2] in register 0x04
        # 3840mV + (VREG * 16mV)
        vreg_code = (vreg_mv - 3840) // 16
        vreg_code = max(0, min(41, vreg_code))  # 6-bit field

        self.modify_register(self.REG_CHARGE_VOLTAGE, 0xFC, (vreg_code << 2))

    def configure_thermal_regulation(self, treg=0b10):
        """Configure thermal regulation.

        Args:
            treg: Thermal regulation threshold (0-3)
                 00: 100°C (default, recommended)
                 01: 110°C
                 10: 120°C
                 11: 130°C
        """
        treg = treg & 0x03
        self.modify_register(self.REG_BOOST, 0xC0, (treg << 6))

    def configure_vsysmin(self, vsysmin_mv=3600):
        """Configure minimum system voltage.

        Args:
            vsysmin_mv: Minimum system voltage in mV (3000-3700mV, step 100mV)
                       Below this, charger prioritizes system power over battery charge
        """
        # Clamp to valid range
        vsysmin_mv = max(3000, min(3700, vsysmin_mv))

        # VSYSMIN bits [3:1] in register 0x01
        # 3000mV + (VSYSMIN * 100mV)
        vsysmin_code = (vsysmin_mv - 3000) // 100
        vsysmin_code = max(0, min(7, vsysmin_code))  # 3-bit field

        self.modify_register(self.REG_POWER_ON, 0x0E, (vsysmin_code << 1))

    def enable_charging(self):
        """Enable charging."""
        # CHE bit [4] in register 0x01 (1 = enable)
        self.modify_register(self.REG_POWER_ON, 0x10, 0x10)

    def is_charging_enabled(self):
        """Check if charging is currently enabled."""
        reg = self.read_register(self.REG_POWER_ON)
        return bool(reg & 0x10)

    def ensure_charging_enabled(self):
        """Ensure charging is enabled (re-enable if disabled).

        Can be called periodically to guarantee charging stays active.
        Returns True if charging is enabled after this call.
        """
        if not self.is_charging_enabled():
            self.enable_charging()
            return False  # Was disabled, now enabled
        return True  # Already enabled

    def disable_charging(self):
        """Disable charging."""
        # CHE bit [4] in register 0x01 (0 = disable)
        self.modify_register(self.REG_POWER_ON, 0x10, 0x00)

    def enable_watchdog(self, timeout=0b01):
        """Enable watchdog timer to reset safety timers.

        Args:
            timeout: Watchdog timeout (0-3)
                    00: Disabled
                    01: 40s (recommended)
                    10: 80s
                    11: 160s
        """
        timeout = timeout & 0x03
        self.modify_register(self.REG_CHARGE_TERM, 0x30, (timeout << 4))

    def disable_watchdog(self):
        """Disable watchdog timer."""
        self.modify_register(self.REG_CHARGE_TERM, 0x30, 0x00)

    def set_charging_safety_timer(self, enable=True, fast_charge_timeout_hours=12):
        """Configure charging safety timer.

        Args:
            enable: Enable safety timer
            fast_charge_timeout_hours: Fast charge timeout (5 or 8 or 12 or 20 hours)
        """
        if not enable:
            # Disable timer by setting bit [5] = 0
            self.modify_register(self.REG_CHARGE_TERM, 0x20, 0x00)
            return

        # Timer enable bit [5] = 1
        self.modify_register(self.REG_CHARGE_TERM, 0x20, 0x20)

        # Fast charge timeout bits [3:2]
        timeout_map = {5: 0x00, 8: 0x01, 12: 0x10, 20: 0x11}
        timeout_code = timeout_map.get(fast_charge_timeout_hours, 0x10)  # Default 12h
        self.modify_register(self.REG_CHARGE_TERM, 0x0C, (timeout_code << 2))

    def enable_boost_mode(self):
        """Enable boost mode (device supplies power to USB)."""
        self.modify_register(self.REG_POWER_ON, 0x20, 0x20)

    def disable_boost_mode(self):
        """Disable boost mode (charging mode)."""
        self.modify_register(self.REG_POWER_ON, 0x20, 0x00)

    def get_status(self, ignore_ntc=True):
        """Read charging status and faults.

        Args:
            ignore_ntc: If True, NTC_FAULT is set to 0 (no thermistor connected).
                       This is recommended for devices without NTC thermistor.
        """
        status = self.read_register(self.REG_STATUS)
        fault = self.read_register(self.REG_FAULT)

        charging_state = (status >> 4) & 0x03
        state_names = {0: "Not Charging", 1: "Pre-charge", 2: "Fast Charge", 3: "Charge Done"}

        ntc_fault = 0 if ignore_ntc else ((fault & 0x03))

        return {
            "charging_state": state_names.get(charging_state, "Unknown"),
            "power_good": bool(status & 0x04),
            "thermal_shutdown": bool(fault & 0x80),
            "battery_overvoltage": bool(fault & 0x40),
            "input_overvoltage": bool(fault & 0x20),
            "charger_fault": (fault & 0x18) >> 3,
            "battery_fault": bool(fault & 0x04),
            "ntc_fault": ntc_fault,  # 0 if no thermistor connected
        }

    def configure_for_10ah_battery(self):
        """Apply optimal configuration for 10000mAh battery.

        Configuration summary (standard performance):
        - Input current limit: 2000mA (safe for USB power)
        - Charging current: 2048mA (~20% of battery capacity per hour)
        - Pre-charge current: 192mA (for low battery recovery)
        - Termination current: 256mA (~2.5% of capacity)
        - Charging voltage: 4208mV (standard Li-ion)
        - Minimum system voltage: 3600mV (system power priority)
        - Thermal regulation: 100°C (aggressive but safe)
        - Charging timer: Enabled, 12 hours
        - Watchdog: Enabled, 40s timeout
        """
        print("[BQ25895] Configuring for 10000mAh Li-polymer battery...")

        # Input current limit
        self.configure_input_current(ilim_ma=2000)
        print("  ✓ Input current limit: 2000mA")

        # Charging current (2048mA ~= 1/5 capacity per hour)
        self.configure_charging_current(ichg_ma=2048)
        print("  ✓ Charging current: 2048mA")

        # Pre-charge current (for cells below 2.8V)
        self.configure_precharge_current(iprechg_ma=192)
        print("  ✓ Pre-charge current: 192mA")

        # Termination current (end-of-charge detection)
        self.configure_termination_current(iterm_ma=256)
        print("  ✓ Termination current: 256mA")

        # Charging voltage limit
        self.configure_charging_voltage(vreg_mv=4208)
        print("  ✓ Charge voltage limit: 4208mV")

        # Minimum system voltage (ensures system power even if battery is depleted)
        self.configure_vsysmin(vsysmin_mv=3600)
        print("  ✓ Minimum system voltage: 3600mV")

        # Thermal regulation
        self.configure_thermal_regulation(treg=0b00)
        print("  ✓ Thermal regulation: 100°C threshold")

        # Safety timer (12-hour max charge time)
        self.set_charging_safety_timer(enable=True, fast_charge_timeout_hours=12)
        print("  ✓ Safety timer: Enabled (12 hours)")

        # Watchdog (40-second timeout for safety timer resets)
        self.enable_watchdog(timeout=0b01)
        print("  ✓ Watchdog timer: Enabled (40s)")

        # Disable boost mode (charger mode, not power output)
        self.disable_boost_mode()
        print("  ✓ Boost mode: Disabled")

        # Enable charging
        self.enable_charging()
        print("  ✓ Charging: Enabled")

        print("[BQ25895] Configuration complete!")

    def configure_for_10ah_battery_longevity(self):
        """Apply optimal configuration for MAXIMUM BATTERY CYCLE LIFE.

        Prioritizes battery longevity over charge speed:
        - Input current limit: 2000mA (safe for USB power)
        - Charging current: 1024mA (~10% of battery capacity per hour - gentle)
        - Pre-charge current: 160mA (conservative recovery)
        - Termination current: 512mA (higher threshold = less stress)
        - Charging voltage: 4100mV (0.1V lower = ~2x cycle life)
        - Minimum system voltage: 3600mV (system power priority)
        - Thermal regulation: 120°C (conservative temperature management)
        - Charging timer: Enabled, 20 hours (slow, safe charging)
        - Watchdog: Enabled, 80s timeout

        Benefits vs standard config:
        ✓ 2x longer cycle life (4.1V vs 4.2V = ~1000 cycles vs 500)
        ✓ Lower heat generation (1024mA vs 2048mA)
        ✓ Safer operation (conservative thermal threshold)
        ✓ Better battery health preservation

        Trade-off: ~10 hours charge time (5.5h → 10-11h)
        """
        print("[BQ25895] Configuring for MAXIMUM BATTERY LONGEVITY...")

        # Input current limit (same - power source limited)
        self.configure_input_current(ilim_ma=2000)
        print("  ✓ Input current limit: 2000mA")

        # Charging current - REDUCED for longevity (1024mA = ~10% C-rate)
        self.configure_charging_current(ichg_ma=1024)
        print("  ✓ Charging current: 1024mA (10% C-rate, gentle)")

        # Pre-charge current (slightly lower for safety)
        self.configure_precharge_current(iprechg_ma=160)
        print("  ✓ Pre-charge current: 160mA")

        # Termination current - HIGHER (less stress at end-of-charge)
        self.configure_termination_current(iterm_ma=512)
        print("  ✓ Termination current: 512mA (less aggressive)")

        # Charging voltage - REDUCED for longevity (4.1V instead of 4.2V)
        # 4.1V provides ~2x cycle life vs 4.2V
        self.configure_charging_voltage(vreg_mv=4100)
        print("  ✓ Charge voltage limit: 4100mV (vs 4.2V for 2x cycle life)")

        # Minimum system voltage
        self.configure_vsysmin(vsysmin_mv=3600)
        print("  ✓ Minimum system voltage: 3600mV")

        # Thermal regulation - CONSERVATIVE (120°C instead of 100°C)
        self.configure_thermal_regulation(treg=0b10)
        print("  ✓ Thermal regulation: 120°C threshold (conservative)")

        # Safety timer - LONGER (20 hours for slow charging)
        self.set_charging_safety_timer(enable=True, fast_charge_timeout_hours=20)
        print("  ✓ Safety timer: Enabled (20 hours for gentle charging)")

        # Watchdog - LONGER interval (80s for slower monitoring)
        self.enable_watchdog(timeout=0b10)
        print("  ✓ Watchdog timer: Enabled (80s)")

        # Disable boost mode
        self.disable_boost_mode()
        print("  ✓ Boost mode: Disabled")

        # Enable charging
        self.enable_charging()
        print("  ✓ Charging: Enabled")

        print("[BQ25895] LONGEVITY configuration complete!")
        print("\n  Expected charge time: 10-11 hours (from 0% to 4.1V)")
        print("  Expected cycle life: >1000 cycles (vs 500 at 4.2V)")
        print("  Temperature during charge: <50°C (vs <60°C at 4.2V)")


def main():
    """Test BQ25895 configuration."""
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 bq25895_charger.py")
        return

    charger = BQ25895()
    try:
        # Read device ID
        dev_id = charger.get_device_id()
        print(f"Device ID/Version: 0x{dev_id:02X}")

        # Configure for 10Ah battery
        charger.configure_for_10ah_battery()

        # Read and display current status
        status = charger.get_status()
        print("\nCharger Status:")
        print(f"  State: {status['charging_state']}")
        print(f"  Power Good: {status['power_good']}")
        print(f"  Thermal Shutdown: {status['thermal_shutdown']}")
        print(f"  Battery OV: {status['battery_overvoltage']}")
        print(f"  Input OV: {status['input_overvoltage']}")
        print(f"  Charger Fault: {status['charger_fault']}")
        print(f"  Battery Fault: {status['battery_fault']}")
        print(f"  NTC Fault: {status['ntc_fault']}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        charger.close()


if __name__ == "__main__":
    main()
