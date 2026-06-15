#!/usr/bin/env python3
"""
BQ27220 Fuel Gauge Configuration for 10000mAh Battery

Based on Flipper Zero's proven BQ27220 driver
(flipperdevices/flipperzero-firmware, lib/drivers/bq27220*).

The BQ27220 is a CEDV gauge. The only capacity parameters that exist in its
Data Memory are FullChargeCapacity and DesignCapacity (both inside the
"Gas Gauging / CEDV Profile 1" subclass). There is NO "Design Energy",
"Terminate Voltage" or "Taper Rate" field on this part -- those belong to the
Impedance Track family. Writing to those offsets here corrupts the CEDV
profile coefficients (EMF/C0/R0/EDV...) and produces nonsense SOC after reinit.

Data Memory addresses (CEDV Profile 1, big-endian 16-bit, verified against the
Flipper bq27220_data_memory.h):
  0x929B - GaugingConfig
  0x929D - FullChargeCapacity (mAh)
  0x929F - DesignCapacity     (mAh)
  0x92A3 - EMF  (first CEDV coefficient -- do NOT overwrite)

NOTE: for an accurate gauge on a new cell the full CEDV profile (EMF, C0, R0,
EDV0/1/2, StartDOD table) should be regenerated with TI's GPCCEDV tool and
loaded via bqStudio. This script only sets the capacity so reported mAh/SOC
scale to a 10000mAh pack.

Register map (BQ27220-specific):
  0x00 - Control
  0x3A - OperationStatus (CFGUPDATE flag is here, bit 10)
  0x3C - DesignCapacity (standard command, read-only mirror of DM)
  0x3E - SelectSubclass / MAC command
  0x40 - MACData (read back after MAC select)
  0x60 - MACDataSum (checksum)
  0x61 - MACDataLen (length)

Control sub-commands (written as 16-bit LE to 0x3E):
  0x0090 - ENTER_CFG_UPDATE
  0x0091 - EXIT_CFG_UPDATE_REINIT
  0x0092 - EXIT_CFG_UPDATE

Unseal keys: 0x0414, 0x3672 (written to Control 0x00)
Full access key: 0xFFFF
"""

import os, fcntl, time, sys

I2C_SLAVE = 0x0703
I2C_BUS = 3
ADDR = 0x55

# Battery configuration
DESIGN_CAPACITY_MAH   = 10000

# Data Memory addresses (CEDV Profile 1 subclass, big-endian 16-bit)
DM_FULL_CHARGE_CAP    = 0x929D  # 2 bytes, mAh
DM_DESIGN_CAPACITY    = 0x929F  # 2 bytes, mAh

# Delays (microseconds, matching Flipper driver)
MAC_WRITE_DELAY    = 0.001     # 250us in Flipper, using 1ms for safety
SELECT_DELAY       = 0.002     # 1ms in Flipper, using 2ms
MAGIC_DELAY        = 0.01      # 5ms in Flipper, using 10ms
CONFIG_DELAY       = 0.02      # 10ms in Flipper, using 20ms
CONFIG_APPLY_DELAY = 2.0       # 2s in Flipper


class BQ27220:
    def __init__(self, bus_num=I2C_BUS, addr=ADDR):
        self.fd = os.open(f"/dev/i2c-{bus_num}", os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def close(self):
        os.close(self.fd)

    def _read(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return list(os.read(self.fd, n))

    def _write(self, reg, data):
        os.write(self.fd, bytes([reg] + list(data)))

    def read_word(self, reg):
        d = self._read(reg, 2)
        return d[0] | (d[1] << 8)  # LE

    def write_word(self, reg, val):
        self._write(reg, [val & 0xFF, (val >> 8) & 0xFF])

    # --- Control commands (written to reg 0x00) ---
    def control(self, subcmd):
        self.write_word(0x00, subcmd)

    # --- Operation Status (reg 0x3A) ---
    def get_operation_status(self):
        return self.read_word(0x3A)

    def is_cfgupdate(self):
        """CFGUPDATE is bit 10 of OperationStatus"""
        return bool(self.get_operation_status() & (1 << 10))

    def get_sec_mode(self):
        """SEC field is bits 1:2 of OperationStatus: 11=sealed, 10=unsealed, 01=full"""
        return (self.get_operation_status() >> 1) & 0x03

    # --- Unseal ---
    def unseal(self):
        self.control(0x0414)  # UnsealKey1
        time.sleep(MAGIC_DELAY)
        self.control(0x3672)  # UnsealKey2
        time.sleep(MAGIC_DELAY)

    # --- Full Access ---
    def full_access(self):
        self.control(0xFFFF)  # FullAccessKey
        time.sleep(MAGIC_DELAY)
        self.control(0xFFFF)
        time.sleep(MAGIC_DELAY)

    # --- Seal ---
    def seal(self):
        self.control(0x0030)
        time.sleep(SELECT_DELAY)

    # --- CFGUPDATE mode ---
    def enter_cfgupdate(self):
        """Enter CFG_UPDATE mode: write 0x0090 to SelectSubclass (0x3E)"""
        self.write_word(0x3E, 0x0090)
        for _ in range(2000):
            time.sleep(0.001)
            if self.is_cfgupdate():
                return True
        return False

    def exit_cfgupdate(self):
        """Exit CFG_UPDATE with reinit: write 0x0091 to SelectSubclass (0x3E)"""
        self.write_word(0x3E, 0x0091)
        time.sleep(CONFIG_APPLY_DELAY)
        for _ in range(2000):
            time.sleep(0.001)
            if not self.is_cfgupdate():
                return True
        return False

    # --- Data Memory read/write via MAC ---
    def read_dm(self, dm_addr, size):
        """
        Read `size` bytes from Data Memory address dm_addr.
        1. Write address to 0x3E (SelectSubclass)
        2. Wait for data load
        3. Read from 0x40 (MACData)
        """
        self.write_word(0x3E, dm_addr)
        time.sleep(SELECT_DELAY)
        data = self._read(0x40, size)
        time.sleep(SELECT_DELAY)
        return data

    def read_dm_u16(self, dm_addr):
        """Read big-endian 16-bit from DM"""
        d = self.read_dm(dm_addr, 2)
        return (d[0] << 8) | d[1]

    def write_dm(self, dm_addr, data_bytes):
        """
        Write data to Data Memory using corrected procedure from TI E2E / Flipper.
        1. Build payload: [addr_lo, addr_hi, data...]
        2. Write payload to 0x3E (SelectSubclass) with auto-increment
        3. Wait
        4. Calculate checksum: 0xFF - sum(payload) & 0xFF
        5. Write checksum to 0x60 and total_length to 0x61
        """
        payload = [dm_addr & 0xFF, (dm_addr >> 8) & 0xFF] + list(data_bytes)

        self._write(0x3E, payload)
        time.sleep(MAC_WRITE_DELAY)

        cksum = (0xFF - (sum(payload) & 0xFF)) & 0xFF
        total_len = 2 + len(data_bytes) + 1 + 1  # addr(2) + data + checksum(1) + length(1)
        self._write(0x60, [cksum, total_len])
        time.sleep(CONFIG_DELAY)

    def write_dm_u16(self, dm_addr, value):
        """Write big-endian 16-bit to DM"""
        self.write_dm(dm_addr, [(value >> 8) & 0xFF, value & 0xFF])

    # --- Status display ---
    def print_status(self):
        v = self.read_word(0x08)
        print(f"  Voltage:          {v} mV")
        t = self.read_word(0x06)
        print(f"  Temperature:      {t / 10 - 273.15:.1f} C")
        bs = self.read_word(0x0A)
        print(f"  BatteryStatus:    0x{bs:04X}")
        os_ = self.get_operation_status()
        print(f"  OperationStatus:  0x{os_:04X}")
        sec = (os_ >> 1) & 0x03
        sec_str = {0b11: "SEALED", 0b10: "UNSEALED", 0b01: "FULL_ACCESS"}.get(sec, f"?({sec})")
        cfgu = "YES" if os_ & (1 << 10) else "NO"
        init = "YES" if os_ & (1 << 5) else "NO"
        print(f"    SEC={sec_str}  CFGUPDATE={cfgu}  INITCOMP={init}")
        rc = self.read_word(0x10)
        print(f"  RemainingCap:     {rc} mAh")
        fc = self.read_word(0x12)
        print(f"  FullChargeCap:    {fc} mAh")
        dc = self.read_word(0x3C)
        print(f"  DesignCapacity:   {dc} mAh")
        soc = self.read_word(0x2C)
        print(f"  StateOfCharge:    {soc} %")
        cur = self.read_word(0x14)
        if cur > 32767:
            cur -= 65536
        print(f"  AvgCurrent:       {cur} mA")


def main():
    g = BQ27220()

    print("=" * 58)
    print("  BQ27220 Configuration for 10000mAh Battery")
    print("=" * 58)

    # --- Current State ---
    print("\n[1] Current State")
    g.print_status()

    # --- Unseal ---
    print("\n[2] Unseal")
    sec = g.get_sec_mode()
    if sec == 0b11:
        print("  Gauge is SEALED, unsealing...")
        g.unseal()
        sec = g.get_sec_mode()
        if sec == 0b11:
            print("  ERROR: Unseal failed!")
            g.close()
            sys.exit(1)
    print(f"  SEC mode: {sec:#04b} ({'UNSEALED' if sec == 0b10 else 'FULL' if sec == 0b01 else '?'})")

    # --- Full Access ---
    print("\n[3] Full Access")
    if sec != 0b01:
        g.full_access()
        sec = g.get_sec_mode()
    print(f"  SEC mode: {sec:#04b} ({'FULL_ACCESS' if sec == 0b01 else '?'})")

    # --- Read current DM values ---
    print("\n[4] Current Data Memory")
    for addr, name in [
        (DM_FULL_CHARGE_CAP, "Full Charge Capacity (mAh)"),
        (DM_DESIGN_CAPACITY, "Design Capacity (mAh)"),
    ]:
        v = g.read_dm_u16(addr)
        print(f"  {name:30s}: {v}")

    # --- Enter CFGUPDATE ---
    print("\n[5] Enter CFGUPDATE")
    if g.is_cfgupdate():
        print("  Already in CFGUPDATE mode")
    else:
        if g.enter_cfgupdate():
            print("  CFGUPDATE entered OK")
        else:
            print("  ERROR: Failed to enter CFGUPDATE!")
            g.close()
            sys.exit(1)

    # --- Write parameters ---
    print("\n[6] Write Parameters")
    params = [
        (DM_FULL_CHARGE_CAP, DESIGN_CAPACITY_MAH, "Full Charge Capacity"),
        (DM_DESIGN_CAPACITY, DESIGN_CAPACITY_MAH, "Design Capacity"),
    ]
    for addr, val, name in params:
        g.write_dm_u16(addr, val)
        print(f"  {name:25s} = {val}")

    # --- Verify while still in CFGUPDATE ---
    print("\n[7] Verify (in CFGUPDATE)")
    time.sleep(0.1)
    all_ok = True
    for addr, expect, name in params:
        got = g.read_dm_u16(addr)
        ok = got == expect
        if not ok:
            all_ok = False
        s = "OK" if ok else f"FAIL (got {got})"
        print(f"  {name:25s}: {expect:6d} [{s}]")

    # --- Exit CFGUPDATE ---
    print("\n[8] Exit CFGUPDATE (reinit)")
    if g.exit_cfgupdate():
        print("  Exited OK")
    else:
        print("  WARNING: Exit timeout")

    # --- Seal ---
    print("\n[9] Seal")
    g.seal()
    sec = g.get_sec_mode()
    print(f"  SEC mode: {sec:#04b} ({'SEALED' if sec == 0b11 else 'NOT sealed'})")

    # --- Final state ---
    time.sleep(1)
    print("\n[10] Final State")
    g.print_status()

    g.close()

    print("\n" + "=" * 58)
    if all_ok:
        print("  SUCCESS: 10000mAh configuration applied and verified!")
    else:
        print("  WARNING: Verification failed")
    print("=" * 58)


if __name__ == "__main__":
    main()
