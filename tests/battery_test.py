#!/usr/bin/env python3
"""
Battery & Fuel Gauge Test Script
- BQ27220 fuel gauge on I2C3 @ 0x55
- BQ25895 charger on I2C3 @ 0x6A
"""

import os, fcntl, time, sys

I2C_SLAVE = 0x0703
BUS = 3
FG_ADDR = 0x55
CHG_ADDR = 0x6A

class I2C:
    def __init__(self, bus, addr):
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        self.addr = addr
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def set_addr(self, addr):
        self.addr = addr
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def rb(self, reg):
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, 1)[0]

    def rw(self, reg):
        os.write(self.fd, bytes([reg]))
        d = os.read(self.fd, 2)
        return d[0] | (d[1] << 8)

    def ww(self, reg, val):
        os.write(self.fd, bytes([reg, val & 0xFF, (val >> 8) & 0xFF]))

    def rblk(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return list(os.read(fd, n))

    def close(self):
        os.close(self.fd)

fd = os.open(f"/dev/i2c-{BUS}", os.O_RDWR)

def fg_rw(reg):
    fcntl.ioctl(fd, I2C_SLAVE, FG_ADDR)
    os.write(fd, bytes([reg]))
    d = os.read(fd, 2)
    return d[0] | (d[1] << 8)

def fg_ww(reg, val):
    fcntl.ioctl(fd, I2C_SLAVE, FG_ADDR)
    os.write(fd, bytes([reg, val & 0xFF, (val >> 8) & 0xFF]))

def fg_rblk(reg, n):
    fcntl.ioctl(fd, I2C_SLAVE, FG_ADDR)
    os.write(fd, bytes([reg]))
    return list(os.read(fd, n))

def chg_rb(reg):
    fcntl.ioctl(fd, I2C_SLAVE, CHG_ADDR)
    os.write(fd, bytes([reg]))
    return os.read(fd, 1)[0]

def signed16(v):
    return v if v < 32768 else v - 65536

def bar(pct, width=30):
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_header(title):
    w = 62
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def test_fuel_gauge():
    print_header("FUEL GAUGE - BQ27220 (I2C3 @ 0x55)")

    # Standard registers
    voltage     = fg_rw(0x08)
    temp_raw    = fg_rw(0x06)
    temp_c      = temp_raw / 10.0 - 273.15
    batt_status = fg_rw(0x0A)
    op_status   = fg_rw(0x3A)
    remain_cap  = fg_rw(0x10)
    full_cap    = fg_rw(0x12)
    design_cap  = fg_rw(0x3C)
    avg_current = signed16(fg_rw(0x14))
    avg_power   = signed16(fg_rw(0x24))
    soc         = fg_rw(0x2C)
    soh         = fg_rw(0x2E)
    int_temp    = fg_rw(0x28)
    int_temp_c  = int_temp / 10.0 - 273.15
    tte         = fg_rw(0x16)
    ttf         = fg_rw(0x18)
    cycle_cnt   = fg_rw(0x2A)
    chg_volt    = fg_rw(0x30)
    chg_curr    = fg_rw(0x32)
    raw_current = signed16(fg_rw(0x7A))
    raw_voltage = fg_rw(0x7C)

    sec = (op_status >> 1) & 0x03
    sec_str = {0b11: "SEALED", 0b10: "UNSEALED", 0b01: "FULL_ACCESS"}.get(sec, f"?({sec})")
    cfgu = bool(op_status & (1 << 10))
    initcomp = bool(op_status & (1 << 5))

    # BatteryStatus decode
    dsg  = bool(batt_status & (1 << 6))
    chg  = bool(batt_status & (1 << 8))
    fc   = bool(batt_status & (1 << 9))
    fd_f = bool(batt_status & (1 << 4))
    otd  = bool(batt_status & (1 << 13))
    otc  = bool(batt_status & (1 << 14))

    print(f"\n  --- Battery Measurements ---")
    print(f"  Voltage:          {voltage:>6d} mV   ({voltage/1000:.3f} V)")
    print(f"  Temperature:      {temp_c:>6.1f} C")
    print(f"  Internal Temp:    {int_temp_c:>6.1f} C")
    print(f"  Avg Current:      {avg_current:>6d} mA   ({'charging' if avg_current > 0 else 'discharging' if avg_current < 0 else 'idle'})")
    print(f"  Avg Power:        {avg_power:>6d} mW")
    print(f"  Raw Current:      {raw_current:>6d} mA")
    print(f"  Raw Voltage:      {raw_voltage:>6d} mV")

    print(f"\n  --- Capacity ---")
    print(f"  Remaining:        {remain_cap:>6d} mAh")
    print(f"  Full Charge:      {full_cap:>6d} mAh")
    print(f"  Design Capacity:  {design_cap:>6d} mAh")
    print(f"  Cycle Count:      {cycle_cnt:>6d}")

    print(f"\n  --- State ---")
    print(f"  State of Charge:  {soc:>3d} %   {bar(min(soc, 100))}")
    print(f"  State of Health:  {soh:>3d} %   {bar(min(soh, 100))}")
    if tte != 65535:
        print(f"  Time to Empty:    {tte:>6d} min  ({tte//60}h {tte%60}m)")
    else:
        print(f"  Time to Empty:       N/A")
    if ttf != 65535:
        print(f"  Time to Full:     {ttf:>6d} min  ({ttf//60}h {ttf%60}m)")
    else:
        print(f"  Time to Full:        N/A")
    print(f"  Charge Voltage:   {chg_volt:>6d} mV")
    print(f"  Charge Current:   {chg_curr:>6d} mA")

    print(f"\n  --- Status Flags ---")
    print(f"  BatteryStatus:    0x{batt_status:04X}")
    print(f"    Discharging:    {'YES' if dsg else 'NO'}")
    print(f"    Charging:       {'YES' if chg else 'NO'}")
    print(f"    Full Charge:    {'YES' if fc else 'NO'}")
    print(f"    Full Discharge: {'YES' if fd_f else 'NO'}")
    print(f"    Over Temp Dchg: {'YES' if otd else 'NO'}")
    print(f"    Over Temp Chg:  {'YES' if otc else 'NO'}")

    print(f"\n  --- Gauge Status ---")
    print(f"  OperationStatus:  0x{op_status:04X}")
    print(f"    Security:       {sec_str}")
    print(f"    CFGUPDATE:      {'YES' if cfgu else 'NO'}")
    print(f"    INITCOMP:       {'YES' if initcomp else 'NO'}")

    # Read DM config (unseal, read, re-seal)
    print(f"\n  --- Data Memory Config ---")
    fg_ww(0x00, 0x0414)
    time.sleep(0.01)
    fg_ww(0x00, 0x3672)
    time.sleep(0.01)

    dm_params = [
        (0x929C, "Design Capacity",  "mAh"),
        (0x929E, "Design Energy",    "cWh"),
        (0x92A0, "Terminate Voltage","mV"),
        (0x92A3, "Taper Rate",       ""),
    ]
    for addr, name, unit in dm_params:
        fg_ww(0x3E, addr)
        time.sleep(0.005)
        raw = fg_rblk(0x40, 2)
        val = (raw[0] << 8) | raw[1]
        print(f"  {name:22s}: {val:>6d} {unit}")

    fg_ww(0x00, 0x0030)
    time.sleep(0.005)

    return voltage, soc, avg_current, temp_c


def test_charger():
    print_header("CHARGER - BQ25895 (I2C3 @ 0x6A)")

    try:
        regs = {}
        for r in range(0x15):
            regs[r] = chg_rb(r)

        # REG0B - System Status
        r0b = regs[0x0B]
        vbus_stat = (r0b >> 5) & 0x07
        chrg_stat = (r0b >> 3) & 0x03
        pg_stat   = (r0b >> 2) & 0x01
        vsys_stat = r0b & 0x01

        vbus_map = {0: "No input", 1: "USB SDP", 2: "USB CDP", 3: "USB DCP",
                    4: "MaxCharge DCP", 5: "Unknown", 6: "Non-standard", 7: "OTG"}
        chrg_map = {0: "Not charging", 1: "Pre-charge", 2: "Fast charge", 3: "Charge done"}

        # REG0C - Fault
        r0c = regs[0x0C]
        wdt_fault = (r0c >> 7) & 1
        boost_fault = (r0c >> 6) & 1
        chrg_fault = (r0c >> 4) & 0x03
        bat_fault = (r0c >> 3) & 1
        ntc_fault = r0c & 0x07

        chrg_fault_map = {0: "Normal", 1: "Input fault", 2: "Thermal shutdown", 3: "Safety timer"}
        ntc_fault_map = {0: "Normal", 1: "TS cold (Buck)", 2: "TS hot (Buck)",
                         5: "TS cold (Boost)", 6: "TS hot (Boost)"}

        # REG0E - VBAT (ADC)
        r0e = regs[0x0E]
        vbat = 2304 + ((r0e & 0x7F) * 20)

        # REG0F - VSYS (ADC)
        r0f = regs[0x0F]
        vsys = 2304 + ((r0f & 0x7F) * 20)

        # REG10 - TSPCT
        r10 = regs[0x10]
        tspct = 21 + ((r10 & 0x7F) * 0.465)

        # REG11 - VBUS (ADC)
        r11 = regs[0x11]
        vbus_v = 2600 + ((r11 & 0x7F) * 100)

        # REG12 - ICHG (ADC)
        r12 = regs[0x12]
        ichg = (r12 & 0x7F) * 50

        # REG00 - Input current limit
        r00 = regs[0x00]
        iinlim = 100 + ((r00 & 0x3F) * 50)
        en_hiz = (r00 >> 7) & 1
        en_ilim = (r00 >> 6) & 1

        # REG02 - ICO / Charge options
        r02 = regs[0x02]
        ico_en = (r02 >> 4) & 1
        auto_dpdm = (r02 >> 0) & 1

        # REG04 - Fast charge current
        r04 = regs[0x04]
        ichg_set = (r04 & 0x7F) * 64

        # REG06 - Charge voltage
        r06 = regs[0x06]
        vreg = 3840 + ((r06 >> 2) * 16)

        print(f"\n  --- Charger Status ---")
        print(f"  VBUS Status:      {vbus_map.get(vbus_stat, '?')}")
        print(f"  Charge Status:    {chrg_map.get(chrg_stat, '?')}")
        print(f"  Power Good:       {'YES' if pg_stat else 'NO'}")
        print(f"  VSYS Regulation:  {'YES' if vsys_stat else 'NO'}")

        print(f"\n  --- ADC Measurements ---")
        print(f"  VBUS:             {vbus_v:>6d} mV   ({vbus_v/1000:.2f} V)")
        print(f"  VBAT:             {vbat:>6d} mV   ({vbat/1000:.3f} V)")
        print(f"  VSYS:             {vsys:>6d} mV   ({vsys/1000:.3f} V)")
        print(f"  Charge Current:   {ichg:>6d} mA")
        print(f"  TS PCT:           {tspct:>6.1f} %")

        print(f"\n  --- Charge Settings ---")
        print(f"  Input Current Lim:{iinlim:>6d} mA")
        print(f"  Fast Chg Current: {ichg_set:>6d} mA")
        print(f"  Charge Voltage:   {vreg:>6d} mV   ({vreg/1000:.3f} V)")
        print(f"  HiZ Mode:         {'ON' if en_hiz else 'OFF'}")
        print(f"  ILIM Pin:         {'Enabled' if en_ilim else 'Disabled'}")
        print(f"  ICO:              {'Enabled' if ico_en else 'Disabled'}")
        print(f"  Auto DPDM:        {'Enabled' if auto_dpdm else 'Disabled'}")

        print(f"\n  --- Faults ---")
        print(f"  Fault Reg:        0x{r0c:02X}")
        print(f"    Watchdog:       {'FAULT' if wdt_fault else 'OK'}")
        print(f"    Boost:          {'FAULT' if boost_fault else 'OK'}")
        print(f"    Charge:         {chrg_fault_map.get(chrg_fault, '?')}")
        print(f"    Battery:        {'OVP' if bat_fault else 'OK'}")
        print(f"    NTC:            {ntc_fault_map.get(ntc_fault, f'?({ntc_fault})')}")

        print(f"\n  --- Raw Registers ---")
        for r in range(0x15):
            print(f"    REG{r:02X}: 0x{regs[r]:02X}  ({regs[r]:08b})", end="")
            if (r + 1) % 3 == 0:
                print()
            else:
                print("   ", end="")
        print()

        return vbat, ichg, chrg_stat, pg_stat

    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, 0, 0, 0


def test_battery_health(voltage, soc, current, temp):
    print_header("BATTERY HEALTH ASSESSMENT")

    issues = []

    # Voltage check
    if voltage > 4250:
        issues.append("WARN: Battery voltage above 4.25V - possible overcharge")
    elif voltage < 3000:
        issues.append("CRIT: Battery voltage below 3.0V - deep discharge")
    elif voltage < 3300:
        issues.append("WARN: Battery voltage low (< 3.3V)")

    # Temperature check
    if temp > 45:
        issues.append("WARN: Battery temperature high (> 45C)")
    elif temp > 60:
        issues.append("CRIT: Battery temperature dangerously high (> 60C)")
    elif temp < 0:
        issues.append("WARN: Battery temperature below 0C - charging should be disabled")

    # Current check
    if current > 3000:
        issues.append("WARN: High charge current (> 3A)")
    elif current < -3000:
        issues.append("WARN: High discharge current (> 3A)")

    print(f"\n  Voltage:     {voltage} mV  ", end="")
    if 3300 <= voltage <= 4200:
        print("  [GOOD]")
    elif 3000 <= voltage < 3300 or 4200 < voltage <= 4250:
        print("  [WARN]")
    else:
        print("  [CRITICAL]")

    print(f"  Temperature: {temp:.1f} C  ", end="")
    if 10 <= temp <= 45:
        print("  [GOOD]")
    elif 0 <= temp < 10 or 45 < temp <= 60:
        print("  [WARN]")
    else:
        print("  [CRITICAL]")

    print(f"  SOC:         {soc} %      ", end="")
    if soc > 20:
        print("  [GOOD]")
    elif soc > 5:
        print("  [LOW]")
    else:
        print("  [CRITICAL]")

    print(f"  Current:     {current} mA  ", end="")
    if abs(current) < 3000:
        print("  [GOOD]")
    else:
        print("  [WARN]")

    if issues:
        print(f"\n  --- Issues Found ---")
        for i in issues:
            print(f"    * {i}")
    else:
        print(f"\n  No issues detected.")


def main():
    print()
    print("##############################################################")
    print("#       BATTERY & FUEL GAUGE TEST REPORT                     #")
    print(f"#       {time.strftime('%Y-%m-%d %H:%M:%S'):>45s}   #")
    print("##############################################################")

    voltage, soc, current, temp = test_fuel_gauge()
    vbat, ichg, chrg_stat, pg_stat = test_charger()
    test_battery_health(voltage, soc, current, temp)

    print_header("SUMMARY")
    print(f"  Battery:  {voltage/1000:.3f}V  {soc}%  {current:+d}mA  {temp:.1f}C")
    chg_s = {0: "Not charging", 1: "Pre-charge", 2: "Fast charge", 3: "Done"}.get(chrg_stat, "?")
    print(f"  Charger:  {vbat/1000:.3f}V  {ichg}mA  {chg_s}  PG={'YES' if pg_stat else 'NO'}")
    print()

if __name__ == "__main__":
    main()
