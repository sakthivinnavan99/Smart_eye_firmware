#!/usr/bin/env python3
"""
BQ27220 SOC Reset Utility

The BQ27220 tracks remaining capacity via coulomb counting. If the gauge
starts from an incorrect initial point (e.g. after ship-mode exit, partial
battery disconnect, or first boot without a prior learning cycle), its
RemainingCapacity drifts far from the actual value even though FCC is correct.

Fix: issue SOFT_RESET (Control 0x0042). The gauge re-reads the open-circuit
voltage (OCV) and re-initialises SOC from the built-in OCV→SOC curve.
DesignCapacity and FCC are preserved across a SOFT_RESET.

Usage:
    sudo python3 tests/fix_bq27220_soc.py [--force]

Options:
    --force   Apply SOFT_RESET even if SOC looks consistent with voltage.

Best accuracy: run with no current flowing (charger disconnected, device idle)
so the voltage reading reflects the true OCV. Running while charging is still
safe and will give a good estimate since VBAT ≈ OCV at low charge currents.
"""

import os
import sys
import fcntl
import time
import argparse

BUS       = 3
ADDR      = 0x55
I2C_SLAVE = 0x0703

# ── I2C helpers ────────────────────────────────────────────────────────────────

def open_gauge():
    fd = os.open(f'/dev/i2c-{BUS}', os.O_RDWR)
    fcntl.ioctl(fd, I2C_SLAVE, ADDR)
    return fd

def rw(fd, reg):
    """Read 16-bit little-endian standard command register."""
    os.write(fd, bytes([reg]))
    d = os.read(fd, 2)
    return d[0] | (d[1] << 8)

def rw_signed(fd, reg):
    v = rw(fd, reg)
    return v if v < 0x8000 else v - 0x10000

def ctrl(fd, subcmd):
    """Write a 16-bit subcommand to the Control register (0x00)."""
    os.write(fd, bytes([0x00, subcmd & 0xFF, (subcmd >> 8) & 0xFF]))

# ── Status snapshot ────────────────────────────────────────────────────────────

def snapshot(fd):
    voltage = rw(fd, 0x08)           # mV
    soc     = rw(fd, 0x2C)           # %
    rem     = rw(fd, 0x10)           # mAh
    fcc     = rw(fd, 0x12)           # mAh
    des     = rw(fd, 0x3C)           # mAh (DesignCapacity)
    cur     = rw_signed(fd, 0x14)    # mA  (negative = discharging)
    op_st   = rw(fd, 0x3A)           # OpStatus flags
    return dict(voltage=voltage, soc=soc, rem=rem, fcc=fcc,
                des=des, cur=cur, op_st=op_st)

def print_snapshot(label, s):
    cur_sign = '+' if s['cur'] >= 0 else ''
    cur_label = 'charging' if s['cur'] > 50 else ('discharging' if s['cur'] < -50 else 'idle')
    print(f"  {label}")
    print(f"    Voltage        : {s['voltage']} mV")
    print(f"    SOC            : {s['soc']} %")
    print(f"    Remaining      : {s['rem']} mAh")
    print(f"    FullChargeCap  : {s['fcc']} mAh")
    print(f"    DesignCapacity : {s['des']} mAh")
    print(f"    Current        : {cur_sign}{s['cur']} mA  ({cur_label})")
    print(f"    OpStatus       : 0x{s['op_st']:04X}")

# ── OCV→SOC lookup (typical Li-ion, good enough for sanity check) ──────────────

_OCV_TABLE = [
    (4200, 100), (4150, 97), (4100, 93), (4050, 89), (4000, 84),
    (3950, 79),  (3900, 73), (3850, 67), (3800, 60), (3750, 52),
    (3700, 43),  (3650, 35), (3600, 27), (3550, 20), (3500, 13),
    (3450, 8),   (3400, 4),  (3350, 2),  (3300, 1),  (3000, 0),
]

def ocv_to_soc_estimate(mv):
    for v, s in _OCV_TABLE:
        if mv >= v:
            return s
    return 0

# ── Unseal helpers ─────────────────────────────────────────────────────────────

def unseal(fd):
    """Send default unseal key pair and full-access key pair."""
    ctrl(fd, 0x0414); time.sleep(0.01)
    ctrl(fd, 0x3672); time.sleep(0.01)
    ctrl(fd, 0xFFFF); time.sleep(0.01)
    ctrl(fd, 0xFFFF); time.sleep(0.01)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='Apply SOFT_RESET even if SOC looks consistent')
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: Run as root:  sudo python3 tests/fix_bq27220_soc.py")
        sys.exit(1)

    print("=" * 60)
    print("  BQ27220 SOC Reset Utility")
    print("=" * 60)

    try:
        fd = open_gauge()
    except OSError as e:
        print(f"ERROR: Cannot open BQ27220 on i2c-{BUS}/0x{ADDR:02X}: {e}")
        sys.exit(1)

    # ── Read current state ─────────────────────────────────────────────────────
    print()
    before = snapshot(fd)
    print_snapshot("Before reset:", before)

    # ── Sanity check: is SOC consistent with voltage? ──────────────────────────
    expected_soc = ocv_to_soc_estimate(before['voltage'])
    soc_error    = abs(before['soc'] - expected_soc)
    print()
    print(f"  Expected SOC for {before['voltage']} mV : ~{expected_soc}%")
    print(f"  Reported SOC                   : {before['soc']}%")
    print(f"  Error                          : {soc_error} percentage points")

    if soc_error <= 10 and not args.force:
        print()
        print("  SOC looks consistent with voltage (error ≤ 10%). No reset needed.")
        print("  Use --force to reset anyway.")
        os.close(fd)
        return

    # ── Warn if high current is flowing ───────────────────────────────────────
    if abs(before['cur']) > 500:
        print()
        print(f"  WARNING: Current is {before['cur']:+d} mA.")
        print("  For the most accurate SOC after reset, disconnect the charger,")
        print("  wait 30+ minutes so the battery voltage settles to true OCV,")
        print("  then run this script again.")
        print("  Proceeding anyway — estimate will still be much better than current value.")

    # ── Ensure DesignCapacity is correct before resetting ─────────────────────
    if before['des'] != 10000 or before['fcc'] != 10000:
        print()
        print(f"  DesignCapacity={before['des']} mAh / FCC={before['fcc']} mAh — expected 10000.")
        print("  Re-programming capacity before reset...")
        unseal(fd)

        # Enter CFGUPDATE
        ctrl(fd, 0x0090)
        for _ in range(100):
            time.sleep(0.05)
            if rw(fd, 0x3A) & (1 << 10):
                break
        else:
            print("  ERROR: Could not enter CFGUPDATE. Aborting.")
            os.close(fd)
            sys.exit(1)

        # Write FullChargeCapacity (DM 0x929D) and DesignCapacity (DM 0x929F)
        for dm_addr, value in [(0x929D, 10000), (0x929F, 10000)]:
            payload = [dm_addr & 0xFF, (dm_addr >> 8) & 0xFF,
                       (value >> 8) & 0xFF, value & 0xFF]
            os.write(fd, bytes([0x3E] + payload))
            time.sleep(0.005)
            cksum = (0xFF - (sum(payload) & 0xFF)) & 0xFF
            os.write(fd, bytes([0x60, cksum, len(payload) + 2]))
            time.sleep(0.02)

        # Exit CFGUPDATE
        ctrl(fd, 0x0091)
        time.sleep(2.0)
        for _ in range(100):
            time.sleep(0.05)
            if not (rw(fd, 0x3A) & (1 << 10)):
                break

        ctrl(fd, 0x0030)  # re-seal
        time.sleep(0.1)
        print("  Capacity re-programmed.")

    # ── Issue SOFT_RESET ───────────────────────────────────────────────────────
    print()
    print("  Issuing SOFT_RESET (Control 0x0042)...")
    ctrl(fd, 0x0042)

    # Wait for the gauge to reinitialise (typically < 2 s)
    for wait in range(20):
        time.sleep(0.5)
        try:
            os.close(fd)
            fd = open_gauge()
            s = snapshot(fd)
            if s['soc'] != before['soc'] or wait >= 4:
                break
        except OSError:
            pass

    print("  Waiting 3 s for OCV settling...")
    time.sleep(3.0)

    os.close(fd)
    fd = open_gauge()
    after = snapshot(fd)

    print()
    print_snapshot("After reset:", after)

    # ── Result ─────────────────────────────────────────────────────────────────
    print()
    soc_delta = after['soc'] - before['soc']
    if abs(after['soc'] - expected_soc) <= 10:
        print(f"  ✓ SOC corrected: {before['soc']}% → {after['soc']}%  "
              f"(expected ~{expected_soc}% for {after['voltage']} mV)")
    else:
        print(f"  SOC changed {before['soc']}% → {after['soc']}%  "
              f"(expected ~{expected_soc}% for {after['voltage']} mV)")
        print()
        print("  If SOC is still off, the gauge needs a full learning cycle:")
        print("    1. Charge fully until CHRG_STAT = 'Charge Done' (ICHG < 256 mA)")
        print("    2. Rest 1 hour with charger disconnected")
        print("    3. Run this script again")
        print("    4. Discharge to protection cutoff under normal use")
        print("    5. Charge fully again — gauge will learn true capacity")

    os.close(fd)


if __name__ == '__main__':
    main()
