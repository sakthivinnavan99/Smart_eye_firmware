#!/usr/bin/env python3
"""
BQ25895 real-time monitor.
Decodes all status, fault, and ADC registers every second.
Highlights changes in yellow. Faults shown in red.

Run with: sudo python3 tests/bq25895_monitor.py
"""

import fcntl
import time
import sys
import os

BUS  = 3
ADDR = 0x6A
FG_ADDR = 0x55   # BQ27220 fuel gauge

# ANSI
R  = '\033[91m'  # red
Y  = '\033[93m'  # yellow
G  = '\033[92m'  # green
C  = '\033[96m'  # cyan
W  = '\033[1m'   # bold
D  = '\033[2m'   # dim
X  = '\033[0m'   # reset


# ── I2C helpers ────────────────────────────────────────────────────────────

def read_reg(reg):
    with open(f'/dev/i2c-{BUS}', 'rb+', buffering=0) as f:
        fcntl.ioctl(f, 0x0703, ADDR)
        f.write(bytes([reg]))
        return ord(f.read(1))

def read_all():
    regs = {}
    with open(f'/dev/i2c-{BUS}', 'rb+', buffering=0) as f:
        fcntl.ioctl(f, 0x0703, ADDR)
        for reg in range(0x00, 0x15):
            try:
                f.write(bytes([reg]))
                regs[reg] = ord(f.read(1))
            except OSError:
                regs[reg] = None
    return regs

def read_fuel_gauge():
    """Read BQ27220 fuel gauge on I2C3/0x55. Returns dict or None on error.

    Register map (16-bit little-endian, TI SLUSE14):
      0x2C  StateOfCharge    (%)
      0x08  Voltage          (mV)
      0x10  RemainingCapacity(mAh)
      0x12  FullChargeCapacity(mAh)
      0x14  AverageCurrent   (mA, signed — negative = discharging)
      0x16  TimeToEmpty      (minutes, 0xFFFF = not discharging)
    """
    try:
        with open(f'/dev/i2c-{BUS}', 'rb+', buffering=0) as f:
            fcntl.ioctl(f, 0x0703, FG_ADDR)
            def rw(reg):
                f.write(bytes([reg]))
                d = f.read(2)
                return d[0] | (d[1] << 8)
            soc     = rw(0x2C)
            voltage = rw(0x08)
            rem_cap = rw(0x10)
            fcc     = rw(0x12)
            cur_raw = rw(0x14)
            avg_cur = cur_raw if cur_raw < 0x8000 else cur_raw - 0x10000
            tte_raw = rw(0x16)
            tte     = None if tte_raw == 0xFFFF else tte_raw
        return {
            'soc'    : min(soc, 100),
            'voltage': voltage,
            'rem_cap': rem_cap,
            'fcc'    : fcc,
            'avg_cur': avg_cur,
            'tte'    : tte,
        }
    except OSError:
        return None


# ── Register decoders ──────────────────────────────────────────────────────

def decode_00(v):
    """REG00 Input Source Control"""
    en_hiz  = (v >> 7) & 1
    en_ilim = (v >> 6) & 1
    iinlim  = v & 0x3F
    return {
        'EN_HIZ'    : en_hiz,      # 1 = input high impedance (disables charging)
        'EN_ILIM'   : en_ilim,
        'IINLIM_mA' : 100 + iinlim * 50,
    }

def decode_03(v):
    """REG03 Charger Control (BQ25895)
    Bit 5: OTG_CONFIG  Bit 4: CHG_CONFIG  Bits[3:1]: SYS_MIN
    """
    otg_en  = (v >> 5) & 1
    chg_en  = (v >> 4) & 1
    sysmin  = (v >> 1) & 0x07
    return {
        'OTG_CONFIG' : otg_en,   # 1 = boost (OTG) mode
        'CHG_CONFIG' : chg_en,   # 1 = charging enabled, 0 = charging DISABLED
        'SYS_MIN_mV' : 3000 + sysmin * 100,
    }

def decode_07(v):
    """REG07 Charge Timer Control"""
    wdt = (v >> 4) & 0x03
    wdt_map = ['Disabled (0)', '40 s', '80 s', '160 s']
    chg_timer_map = ['5 h', '8 h', '12 h', '20 h']
    return {
        'EN_TERM'      : (v >> 7) & 1,
        'STAT_DIS'     : (v >> 6) & 1,
        'WATCHDOG'     : wdt_map[wdt],
        'CHG_TIMER_EN' : (v >> 3) & 1,
        'CHG_TIMER'    : chg_timer_map[(v >> 1) & 0x03],
    }

def decode_09(v):
    """REG09 BATFET / Pump (BQ25895)
    Bit 7: FORCE_ICO  Bit 6: TMR2X_EN  Bit 5: BATFET_DIS  Bit 4: Reserved
    Bit 3: BATFET_DLY  Bit 2: BATFET_RST_EN  Bit 1: PUMPX_UP  Bit 0: PUMPX_DN
    """
    return {
        'FORCE_ICO'      : (v >> 7) & 1,
        'TMR2X_EN'       : (v >> 6) & 1,
        'BATFET_DIS'     : (v >> 5) & 1,  # 1 = ship mode (battery isolated)
        'BATFET_DLY'     : (v >> 3) & 1,  # 1 = 10 s delay before isolation
        'BATFET_RST_EN'  : (v >> 2) & 1,  # 1 = allow BATFET reset on plug-in
        'PUMPX_UP'       : (v >> 1) & 1,
        'PUMPX_DN'       : (v >> 0) & 1,
    }

def decode_0b(v):
    """REG0B Charger Status  ← LED is driven from CHRG_STAT / PG_STAT"""
    vbus_stat = (v >> 5) & 0x07
    chrg_stat = (v >> 3) & 0x03
    pg        = (v >> 2) & 0x01
    sdp       = (v >> 1) & 0x01
    vsys_stat = (v >> 0) & 0x01

    vbus_map = [
        'No Input',
        'USB SDP (500 mA)',
        'USB CDP (1.5 A)',
        'USB DCP (3.25 A)',
        'Adj HV DCP',
        'Unknown Adapter',
        'Non-Standard Adapter',
        'OTG',
    ]
    chrg_map = [
        'Not Charging',
        'Pre-charge',
        'Fast Charging',
        'Charge Done',
    ]
    return {
        'VBUS_STAT' : (vbus_stat, vbus_map[vbus_stat]),
        'CHRG_STAT' : (chrg_stat, chrg_map[chrg_stat]),
        'PG_STAT'   : pg,        # 1 = power good (VBUS valid)
        'SDP_STAT'  : sdp,
        'VSYS_STAT' : vsys_stat, # 1 = VSYS regulation active
    }

def decode_0c(v):
    """REG0C Fault Register  ← blink on LED means non-zero here"""
    wdog  = (v >> 7) & 1
    boost = (v >> 6) & 1
    chrg  = (v >> 4) & 0x03
    bat   = (v >> 3) & 1
    ntc   = (v >> 0) & 0x07

    chrg_map = ['Normal', 'Input OVP', 'Thermal Shutdown', 'Safety Timer']
    ntc_map  = ['Normal', 'Warm', 'Cool', 'Cold',
                'Hot (charging)', 'Cold (charging)', 'Hot (boost)', 'Cold (boost)']
    return {
        'WATCHDOG_FAULT' : wdog,  # 1 = watchdog timer expired
        'BOOST_FAULT'    : boost,
        'CHRG_FAULT'     : (chrg, chrg_map[chrg]),
        'BAT_FAULT'      : bat,   # 1 = VBAT OVP
        'NTC_FAULT'      : (ntc, ntc_map[ntc]),  # non-zero = temperature issue
        'ANY'            : v != 0,
    }

def decode_adc(regs):
    """ADC result registers REG0E-REG13"""
    def r(reg): return regs.get(reg, 0) or 0
    return {
        'VBAT_mV'  : 2304 + (r(0x0E) & 0x7F) * 20,
        'VSYS_mV'  : 2304 + (r(0x0F) & 0x7F) * 20,
        'VBUS_mV'  : 2600 + (r(0x11) & 0x7F) * 100,
        'ICHG_mA'  : (r(0x12) & 0x7F) * 50,
        'IDPM_mA'  : 100 + (r(0x13) & 0x3F) * 50,
        'THERM_pct': (r(0x10) & 0x7F) * 0.465 + 21.0,   # TS pin %
    }


# ── Display ────────────────────────────────────────────────────────────────

def colour_val(key, val, prev_val):
    changed = (prev_val is not None) and (val != prev_val)
    if changed:
        return f'{Y}{val}{X}'
    return str(val)

def print_status(regs, prev_regs, fg=None):
    os.system('clear')

    def changed(reg):
        return prev_regs and regs.get(reg) != prev_regs.get(reg)

    def cv(reg):
        v = regs.get(reg, 0)
        return f'{Y}0x{v:02X}{X}' if changed(reg) else f'0x{v:02X}'

    s0b  = decode_0b(regs.get(0x0B, 0))
    s0c  = decode_0c(regs.get(0x0C, 0))
    s00  = decode_00(regs.get(0x00, 0))
    s03  = decode_03(regs.get(0x03, 0))
    s09  = decode_09(regs.get(0x09, 0))
    s07  = decode_07(regs.get(0x07, 0))
    adc  = decode_adc(regs)

    print(f'{W}══════════════════════════════════════════════════{X}')
    print(f'{W}  BQ25895 Monitor  {D}{time.strftime("%H:%M:%S")}{X}')
    print(f'{W}══════════════════════════════════════════════════{X}')

    # ── Charger Status (REG0B) ─────────────────────────────────────────────
    vbus_n, vbus_s = s0b['VBUS_STAT']
    chrg_n, chrg_s = s0b['CHRG_STAT']
    pg             = s0b['PG_STAT']
    vsys_s         = s0b['VSYS_STAT']

    vbus_col = (R if vbus_n != 0 and not pg else G) if vbus_n != 0 else D
    pg_col   = G if pg else D
    chrg_col = G if chrg_n in (1, 2) else (Y if chrg_n == 3 else D)

    print(f'\n{C}  ── REG0B  Charger Status  {cv(0x0B)} ──{X}')
    print(f'  VBUS   : {vbus_col}{vbus_s}{X}')
    print(f'  PG     : {pg_col}{"Power Good ✓" if pg else "No Power"}{X}')
    print(f'  CHARGE : {chrg_col}{chrg_s}{X}')
    if vsys_s:
        print(f'  VSYS   : {Y}VSYS Regulation Active (battery below SYSMIN){X}')

    # LED logic explanation
    print()
    if chrg_n in (1, 2):
        print(f'  {G}STAT LED → ON  (charging){X}')
    elif s0c['ANY']:
        print(f'  {R}STAT LED → BLINKING  (fault detected — see below){X}')
    elif chrg_n == 0:
        print(f'  {D}STAT LED → OFF (not charging){X}')
    else:
        print(f'  {Y}STAT LED → OFF (charge done){X}')

    # ── Fault Register (REG0C) ─────────────────────────────────────────────
    print(f'\n{C}  ── REG0C  Fault Register  {cv(0x0C)} ──{X}')
    if not s0c['ANY']:
        print(f'  {G}No faults{X}')
    else:
        if s0c['WATCHDOG_FAULT']:
            print(f'  {R}WATCHDOG_FAULT  — watchdog expired, registers may have reset{X}')
        if s0c['BOOST_FAULT']:
            print(f'  {R}BOOST_FAULT{X}')
        chrg_fn, chrg_fs = s0c['CHRG_FAULT']
        if chrg_fn:
            print(f'  {R}CHRG_FAULT  — {chrg_fs}{X}')
        if s0c['BAT_FAULT']:
            print(f'  {R}BAT_FAULT   — VBAT OVP (battery overvoltage){X}')
        ntc_fn, ntc_fs = s0c['NTC_FAULT']
        if ntc_fn:
            print(f'  {R}NTC_FAULT   — {ntc_fs}  (TS pin voltage out of range){X}')

    # ── ADC Readings ───────────────────────────────────────────────────────
    print(f'\n{C}  ── ADC Readings ──{X}')
    vbus_adc = adc['VBUS_mV']
    vbus_col2 = Y if (vbus_adc > 3000 and not pg) else X
    print(f'  VBUS   : {vbus_col2}{vbus_adc:5d} mV{X}  {"← phantom VBUS!" if vbus_adc > 3000 and not pg else ""}')
    print(f'  VBAT   : {adc["VBAT_mV"]:5d} mV')
    print(f'  VSYS   : {adc["VSYS_mV"]:5d} mV')
    print(f'  ICHG   : {adc["ICHG_mA"]:5d} mA')
    print(f'  IDPM   : {adc["IDPM_mA"]:5d} mA')
    print(f'  TS pin : {adc["THERM_pct"]:5.1f} %  {"← NTC fault likely if outside 15-80%" if not (15 < adc["THERM_pct"] < 80) else ""}')

    # ── BQ27220 Fuel Gauge ─────────────────────────────────────────────────
    print(f'\n{C}  ── BQ27220 Fuel Gauge ──{X}')
    if fg is None:
        print(f'  {D}(not available — BQ27220 not found on i2c-{BUS}/0x{FG_ADDR:02X}){X}')
    else:
        soc = fg['soc']
        soc_bar_filled = int(soc / 5)
        soc_bar = '█' * soc_bar_filled + '░' * (20 - soc_bar_filled)
        soc_col = G if soc > 30 else (Y if soc > 10 else R)
        print(f'  SOC    : {soc_col}{soc:3d}%  [{soc_bar}]{X}')
        print(f'  Voltage: {fg["voltage"]:5d} mV  (gauge)')
        rem = fg['rem_cap']
        fcc = fg['fcc'] if fg['fcc'] > 0 else 10000
        print(f'  Remain : {rem:5d} mAh / {fcc} mAh')
        avg = fg['avg_cur']
        avg_col = G if avg > 0 else (D if avg == 0 else Y)
        avg_label = 'charging' if avg > 0 else ('discharging' if avg < 0 else 'idle')
        print(f'  Current: {avg_col}{avg:+6d} mA  ({avg_label}){X}')
        tte = fg['tte']
        if tte is not None:
            h, m = divmod(tte, 60)
            print(f'  TTE    : {h}h {m:02d}m  (to empty)')
        else:
            print(f'  TTE    : {D}—  (not discharging){X}')

    # ── Key Control Registers ──────────────────────────────────────────────
    print(f'\n{C}  ── Control Registers ──{X}')
    hiz = s00['EN_HIZ']
    print(f'  REG00  EN_HIZ    : {R if hiz else D}{"ON — input disabled!" if hiz else "off"}{X}  '
          f'IINLIM={s00["IINLIM_mA"]} mA')
    chg_en = s03['CHG_CONFIG']
    print(f'  REG03  CHG_CONFIG: {G if chg_en else R}{"enabled" if chg_en else "DISABLED — no charging!"}{X}  '
          f'SYS_MIN={s03["SYS_MIN_mV"]} mV')
    print(f'  REG07  WATCHDOG  : {cv(0x07)}  {s07["WATCHDOG"]}')
    bdis = s09['BATFET_DIS']
    print(f'  REG09  BATFET_DIS: {R if bdis else D}{bdis}{X}'
          f'{"  ← ship mode active!" if bdis else ""}')

    # ── Raw register dump ──────────────────────────────────────────────────
    print(f'\n{C}  ── Raw Registers ──{X}')
    row = ''
    for reg in range(0x00, 0x15):
        v = regs.get(reg)
        val = f'0x{v:02X}' if v is not None else '----'
        c_str = Y if (prev_regs and regs.get(reg) != prev_regs.get(reg)) else D
        row += f'  {c_str}[{reg:02X}]={val}{X}'
        if (reg + 1) % 5 == 0:
            print(row)
            row = ''
    if row:
        print(row)

    # ── Diagnosis hint ────────────────────────────────────────────────────
    print(f'\n{C}  ── Diagnosis ──{X}')
    hints = []
    if vbus_n == 0 and s0b['PG_STAT'] and adc['VBUS_mV'] > 3000:
        hints.append(f'{Y}VBUS_STAT=No Input but PG=1 and VBUS ADC={adc["VBUS_mV"]}mV — '
                     f'normal when AUTO_DPDM_EN=0 (REG02[0]=0): DPDM detection never runs '
                     f'so VBUS_STAT stays 000. Charging is working correctly.{X}')
    if s0b['PG_STAT'] == 0 and vbus_n != 0:
        hints.append(f'{R}VBUS_STAT shows input but PG=0 — false VBUS detection{X}')
    if s0b['PG_STAT'] == 0 and s0b['CHRG_STAT'][0] in (1, 2):
        hints.append(f'{R}Charging without PG — abnormal, check VBUS line{X}')
    if adc['VBUS_mV'] > 2800 and not pg:
        hints.append(f'{R}VBUS ADC reads {adc["VBUS_mV"]} mV but PG=0 — VBUS leakage/phantom{X}')
    ntc_n, _ = s0c['NTC_FAULT']
    if ntc_n:
        hints.append(f'{R}NTC fault → LED blinks. Check TS pin voltage (should be ~30-40% REGN). '
                     f'If no NTC fitted, tie TS to GND via 10kΩ or disable in REG08{X}')
    if s0c['WATCHDOG_FAULT']:
        hints.append(f'{Y}Watchdog fault → registers reset to default. '
                     f'Disable watchdog: i2cset -y 3 0x6a 0x07 0x8f b{X}')
    if not s03['CHG_CONFIG']:
        hints.append(f'{R}CHG_CONFIG=0 in REG03 — charging is DISABLED. '
                     f'Enable: sudo i2cset -y 3 0x6a 0x03 0x5a b{X}')
    if s09['BATFET_DIS']:
        hints.append(f'{R}BATFET_DIS=1 — battery isolated (ship mode). '
                     f'Clear: sudo i2cset -y 3 0x6a 0x09 0x04 b{X}')
    if s0b['VSYS_STAT'] and adc['ICHG_mA'] == 0 and s0b['PG_STAT']:
        hints.append(f'{Y}VSYS Regulation Active + ICHG=0: system load ≥ IINLIM ({s00["IINLIM_mA"]} mA). '
                     f'Use a charger with higher current rating (DCP/USB-C PD).{X}')
    if regs.get(0x02, 0) & 0x01:
        hints.append(f'{Y}AUTO_DPDM_EN=1 in REG02 — BQ25895 will reset IINLIM to 500mA on every VBUS replug '
                     f'(D+/D- detection overrides software limit). '
                     f'Fix: sudo i2cset -y 3 0x6a 0x02 0xfc b{X}')
    if not hints:
        hints.append(f'{G}No obvious issues detected{X}')
    for h in hints:
        print(f'  • {h}')

    print(f'\n{D}  Polling every 1 s — Ctrl+C to stop{X}\n')


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    if os.geteuid() != 0:
        print(f'{R}Run as root: sudo python3 tests/bq25895_monitor.py{X}')
        sys.exit(1)

    try:
        _ = read_reg(0x0B)
    except OSError as e:
        print(f'{R}Cannot reach BQ25895 on i2c-{BUS}/0x{ADDR:02X}: {e}{X}')
        sys.exit(1)

    print(f'{G}BQ25895 found on i2c-{BUS}/0x{ADDR:02X} — starting monitor...{X}')
    time.sleep(0.5)

    prev = None
    while True:
        try:
            regs = read_all()
            fg   = read_fuel_gauge()
            print_status(regs, prev, fg)
            prev = regs
            time.sleep(1.0)
        except KeyboardInterrupt:
            print('\nStopped.')
            break
        except OSError as e:
            print(f'{R}I2C error: {e}{X}')
            time.sleep(2)


if __name__ == '__main__':
    main()
