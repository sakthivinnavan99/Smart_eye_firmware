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


class BQ25895:
    """TI BQ25895 Battery Charger IC over I2C."""

    I2C_BUS = 3
    DEVICE_ADDR = 0x6A
    I2C_SLAVE = 0x0703

    # Register Map (verified against datasheet REG00–REG14)
    REG_INPUT_SOURCE  = 0x00  # EN_HIZ[7], EN_ILIM[6], IINLIM[5:0]
    REG_VINDPM_OS     = 0x01  # BHOT[7:6], BCOLD[5], VINDPM_OS[4:0]
    REG_ADC_CTRL      = 0x02  # CONV_START[7], CONV_RATE[6], BOOST_FREQ[5], ICO_EN[4], HVDCP_EN[3], MAXC_EN[2], FORCE_DPDM[1], AUTO_DPDM_EN[0]
    REG_CHG_CTRL      = 0x03  # BAT_LOADEN[7], WD_RST[6], OTG_CONFIG[5], CHG_CONFIG[4], SYS_MIN[3:1], Reserved[0]
    REG_CHARGE_CURRENT= 0x04  # EN_PUMPX[7], ICHG[6:0]  (0mA + code×64mA)
    REG_PRECHARGE     = 0x05  # IPRECHG[7:4], ITERM[3:0]  (64mA offset, 64mA step each)
    REG_CHARGE_VOLTAGE= 0x06  # VREG[7:2], BATLOWV[1], VRECHG[0]  (3840mV + code×16mV)
    REG_CHARGE_TIMER  = 0x07  # EN_TERM[7], STAT_DIS[6], WATCHDOG[5:4], EN_TIMER[3], CHG_TIMER[2:1], Reserved[0]
    REG_IR_THERMAL    = 0x08  # BAT_COMP[7:5], VCLAMP[4:2], TREG[1:0]  (00=60°C … 11=120°C)
    REG_BATFET        = 0x09  # FORCE_ICO[7], TMR2X_EN[6], BATFET_DIS[5], Reserved[4], BATFET_DLY[3], BATFET_RST_EN[2], PUMPX_UP[1], PUMPX_DN[0]
    REG_BOOST_VOLT    = 0x0A  # BOOSTV[7:4]
    REG_STATUS        = 0x0B  # VBUS_STAT[7:5], CHRG_STAT[4:3], PG_STAT[2], SDP_STAT[1], VSYS_STAT[0]  Read-only
    REG_FAULT         = 0x0C  # WATCHDOG_FAULT[7], BOOST_FAULT[6], CHRG_FAULT[5:4], BAT_FAULT[3], NTC_FAULT[2:0]  Read-only
    REG_VERSION       = 0x14  # REG_RST[7], ICO_OPTIMIZED[6], PN[5:3]=111, TS_PROFILE[2], DEV_REV[1:0]=01

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
        """Read a single register. Address must be written first (I2C protocol)."""
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
        """Read device part number from REG14. PN[5:3]=111 for BQ25895."""
        return self.read_register(self.REG_VERSION)

    def configure_input_current(self, ilim_ma=2000):
        """Configure input current limit (IINLIM, REG00[5:0]).

        Range 100–3250mA, offset 100mA, step 50mA, 6-bit field (codes 0–63).
        """
        ilim_ma = max(100, min(3250, ilim_ma))
        ilim_code = (ilim_ma - 100) // 50
        ilim_code = max(0, min(63, ilim_code))
        self.modify_register(self.REG_INPUT_SOURCE, 0x3F, ilim_code)

    def configure_charging_current(self, ichg_ma=2048):
        """Configure fast-charge current (ICHG, REG04[6:0]).

        Range 0–5056mA, offset 0mA, step 64mA, 7-bit field.
        """
        ichg_ma = max(0, min(5056, ichg_ma))
        ichg_code = ichg_ma // 64
        ichg_code = max(0, min(79, ichg_code))
        self.modify_register(self.REG_CHARGE_CURRENT, 0x7F, ichg_code)

    def configure_precharge_current(self, iprechg_ma=256):
        """Configure pre-charge current (IPRECHG, REG05[7:4]).

        Range 64–1024mA, offset 64mA, step 64mA.
        """
        iprechg_ma = max(64, min(1024, iprechg_ma))
        iprechg_code = (iprechg_ma - 64) // 64
        iprechg_code = max(0, min(15, iprechg_code))
        self.modify_register(self.REG_PRECHARGE, 0xF0, (iprechg_code << 4))

    def configure_termination_current(self, iterm_ma=256):
        """Configure termination current (ITERM, REG05[3:0]).

        Range 64–1024mA, offset 64mA, step 64mA.
        """
        iterm_ma = max(64, min(1024, iterm_ma))
        iterm_code = (iterm_ma - 64) // 64
        iterm_code = max(0, min(15, iterm_code))
        self.modify_register(self.REG_PRECHARGE, 0x0F, iterm_code)

    def configure_charging_voltage(self, vreg_mv=4208):
        """Configure charge voltage limit (VREG, REG06[7:2]).

        Range 3840–4608mV, offset 3840mV, step 16mV.
        """
        vreg_mv = max(3840, min(4608, vreg_mv))
        vreg_code = (vreg_mv - 3840) // 16
        vreg_code = max(0, min(48, vreg_code))
        self.modify_register(self.REG_CHARGE_VOLTAGE, 0xFC, (vreg_code << 2))

    def configure_thermal_regulation(self, treg=0b11):
        """Configure thermal regulation threshold (TREG, REG08[1:0]).

        00=60°C, 01=80°C, 10=100°C, 11=120°C (default).
        """
        treg = treg & 0x03
        self.modify_register(self.REG_IR_THERMAL, 0x03, treg)

    def configure_vsysmin(self, vsysmin_mv=3500):
        """Configure minimum system voltage (SYS_MIN, REG03[3:1]).

        Range 3000–3700mV, offset 3000mV, step 100mV.
        """
        vsysmin_mv = max(3000, min(3700, vsysmin_mv))
        vsysmin_code = (vsysmin_mv - 3000) // 100
        vsysmin_code = max(0, min(7, vsysmin_code))
        self.modify_register(self.REG_CHG_CTRL, 0x0E, (vsysmin_code << 1))

    def enable_charging(self):
        """Enable charging (CHG_CONFIG, REG03[4]=1)."""
        self.modify_register(self.REG_CHG_CTRL, 0x10, 0x10)

    def disable_charging(self):
        """Disable charging (CHG_CONFIG, REG03[4]=0)."""
        self.modify_register(self.REG_CHG_CTRL, 0x10, 0x00)

    def is_charging_enabled(self):
        """Check if charging is currently enabled."""
        return bool(self.read_register(self.REG_CHG_CTRL) & 0x10)

    def ensure_charging_enabled(self):
        """Re-enable charging if it was disabled. Returns True if was already enabled."""
        if not self.is_charging_enabled():
            self.enable_charging()
            return False
        return True

    def enable_boost_mode(self):
        """Enable OTG boost mode (OTG_CONFIG, REG03[5]=1)."""
        self.modify_register(self.REG_CHG_CTRL, 0x20, 0x20)

    def disable_boost_mode(self):
        """Disable OTG boost mode (OTG_CONFIG, REG03[5]=0)."""
        self.modify_register(self.REG_CHG_CTRL, 0x20, 0x00)

    def enable_watchdog(self, timeout=0b01):
        """Enable watchdog timer (WATCHDOG, REG07[5:4]).

        00=disabled, 01=40s, 10=80s, 11=160s.
        """
        timeout = timeout & 0x03
        self.modify_register(self.REG_CHARGE_TIMER, 0x30, (timeout << 4))

    def disable_watchdog(self):
        """Disable watchdog timer (WATCHDOG=00, REG07[5:4])."""
        self.modify_register(self.REG_CHARGE_TIMER, 0x30, 0x00)

    def set_charging_safety_timer(self, enable=True, fast_charge_timeout_hours=12):
        """Configure charging safety timer (EN_TIMER REG07[3], CHG_TIMER REG07[2:1]).

        fast_charge_timeout_hours: 5, 8, 12, or 20 hours.
        """
        if not enable:
            self.modify_register(self.REG_CHARGE_TIMER, 0x08, 0x00)
            return

        self.modify_register(self.REG_CHARGE_TIMER, 0x08, 0x08)

        timeout_map = {5: 0b00, 8: 0b01, 12: 0b10, 20: 0b11}
        timeout_code = timeout_map.get(fast_charge_timeout_hours, 0b10)
        self.modify_register(self.REG_CHARGE_TIMER, 0x06, (timeout_code << 1))

    def get_status(self, ignore_ntc=True):
        """Read charging status (REG0B) and faults (REG0C, read twice for current state).

        Args:
            ignore_ntc: If True, NTC_FAULT forced to 0 (no thermistor connected).
        """
        status = self.read_register(self.REG_STATUS)
        # Datasheet: REG0C must be read twice; second read reflects current state
        self.read_register(self.REG_FAULT)
        fault = self.read_register(self.REG_FAULT)

        # CHRG_STAT = REG0B[4:3]
        charging_state = (status >> 3) & 0x03
        state_names = {0: "Not Charging", 1: "Pre-charge", 2: "Fast Charge", 3: "Charge Done"}

        ntc_fault = 0 if ignore_ntc else (fault & 0x07)

        return {
            "charging_state": state_names.get(charging_state, "Unknown"),
            "power_good": bool(status & 0x04),          # PG_STAT REG0B[2]
            "thermal_shutdown": bool(fault & 0x80),      # WATCHDOG_FAULT REG0C[7]
            "battery_overvoltage": bool(fault & 0x08),   # BAT_FAULT REG0C[3]
            "input_overvoltage": bool((fault >> 4) & 0x03 == 0x01),  # CHRG_FAULT[5:4]=01
            "charger_fault": (fault >> 4) & 0x03,        # CHRG_FAULT REG0C[5:4]
            "battery_fault": bool(fault & 0x08),         # BAT_FAULT REG0C[3]
            "ntc_fault": ntc_fault,                      # NTC_FAULT REG0C[2:0]
        }

    def configure_for_10ah_battery(self):
        """Apply optimal configuration for 10000mAh battery (standard performance).

        - Input current: 2000mA
        - Charge current: 2048mA (~0.2C)
        - Pre-charge current: 256mA
        - Termination current: 256mA (~2.5% capacity)
        - Charge voltage: 4208mV
        - SYS_MIN: 3500mV
        - TREG: 120°C (11)
        - Safety timer: 20h
        - Watchdog: disabled (app manages resets)
        """
        print("[BQ25895] Configuring for 10000mAh Li-polymer battery...")

        self.configure_input_current(ilim_ma=2000)
        print("  Input current limit: 2000mA")

        self.configure_charging_current(ichg_ma=2048)
        print("  Charging current: 2048mA")

        self.configure_precharge_current(iprechg_ma=256)
        print("  Pre-charge current: 256mA")

        self.configure_termination_current(iterm_ma=256)
        print("  Termination current: 256mA")

        self.configure_charging_voltage(vreg_mv=4208)
        print("  Charge voltage: 4208mV")

        self.configure_vsysmin(vsysmin_mv=3500)
        print("  SYS_MIN: 3500mV")

        # TREG=11 → 120°C
        self.configure_thermal_regulation(treg=0b11)
        print("  Thermal regulation: 120°C")

        self.set_charging_safety_timer(enable=True, fast_charge_timeout_hours=20)
        print("  Safety timer: 20h")

        self.disable_watchdog()
        print("  Watchdog: disabled")

        self.disable_boost_mode()
        # Enable continuous ADC; disable AUTO_DPDM so VBUS replug doesn't reset IINLIM to 500mA.
        # FUSB302 on this board handles USB-C negotiation — BQ25895's DPDM detection is redundant.
        # Mask 0xC1 = set bits [7:6] (CONV_START|CONV_RATE), clear bit [0] (AUTO_DPDM_EN=0).
        self.modify_register(0x02, 0xC1, 0xC0)
        self.enable_charging()
        print("[BQ25895] Configuration complete.")

    def configure_for_10ah_battery_longevity(self):
        """Apply configuration optimized for maximum battery cycle life.

        - Charge current: 1024mA (~0.1C)
        - Charge voltage: 4100mV (~2× cycle life vs 4.2V)
        - TREG: 120°C (11)
        - Safety timer: 20h
        """
        print("[BQ25895] Configuring for MAXIMUM BATTERY LONGEVITY...")

        self.configure_input_current(ilim_ma=2000)
        print("  Input current limit: 2000mA")

        self.configure_charging_current(ichg_ma=1024)
        print("  Charging current: 1024mA (0.1C)")

        self.configure_precharge_current(iprechg_ma=256)
        print("  Pre-charge current: 256mA")

        self.configure_termination_current(iterm_ma=512)
        print("  Termination current: 512mA")

        self.configure_charging_voltage(vreg_mv=4100)
        print("  Charge voltage: 4100mV (longevity)")

        self.configure_vsysmin(vsysmin_mv=3500)
        print("  SYS_MIN: 3500mV")

        # TREG=11 → 120°C
        self.configure_thermal_regulation(treg=0b11)
        print("  Thermal regulation: 120°C")

        self.set_charging_safety_timer(enable=True, fast_charge_timeout_hours=20)
        print("  Safety timer: 20h")

        self.disable_watchdog()
        print("  Watchdog: disabled")

        self.disable_boost_mode()
        self.modify_register(0x02, 0xC1, 0xC0)  # continuous ADC + AUTO_DPDM_EN=0
        self.enable_charging()
        print("[BQ25895] LONGEVITY configuration complete.")


def main():
    """Test BQ25895 configuration."""
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 bq25895_charger.py")
        return

    charger = BQ25895()
    try:
        dev_id = charger.get_device_id()
        print(f"Device ID (REG14): 0x{dev_id:02X}  PN[5:3]={(dev_id>>3)&0x07:03b} (expect 111 for BQ25895)")

        charger.configure_for_10ah_battery()

        status = charger.get_status()
        print("\nCharger Status:")
        print(f"  State:            {status['charging_state']}")
        print(f"  Power Good:       {status['power_good']}")
        print(f"  Thermal Shutdown: {status['thermal_shutdown']}")
        print(f"  Battery OV:       {status['battery_overvoltage']}")
        print(f"  Input OV:         {status['input_overvoltage']}")
        print(f"  Charger Fault:    {status['charger_fault']}")
        print(f"  Battery Fault:    {status['battery_fault']}")
        print(f"  NTC Fault:        {status['ntc_fault']}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        charger.close()


if __name__ == "__main__":
    main()
