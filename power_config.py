#!/usr/bin/env python3
"""
Smart Eye Carrier Board - Power System Configuration
=====================================================
Configures BQ25895 charger and BQ27220 fuel gauge for 10000mAh battery,
installs battery management daemon with ship mode protection.

Usage:
    sudo python3 power_config.py [--status] [--ship-mode]

    (no args)     Full configuration: charger + fuel gauge + daemon install
    --status      Read and display current power system state
    --ship-mode   Enter ship mode (disconnect battery, wake by USB plug-in)

Hardware:
    U2  - BQ25895RTWR  Charger       I2C3 @ 0x6A
    U20 - BQ27220YZFR  Fuel Gauge    I2C3 @ 0x55
    U15 - BQ29700DSER  Batt Protect  (hardware only, no I2C)
    R77 - 10 mohm      Sense Resistor
    J5  - BAT_Conn     10000mAh 1S Li-ion

Battery parameters:
    Design Capacity   : 10000 mAh
    Design Energy     : 3700  cWh  (37 Wh)
    Terminate Voltage : 3000  mV
    Taper Rate        : 217
    Charge Voltage    : 4208  mV
    Charge Current    : 2048  mA   (0.2C)
    Input Current Lim : 2000  mA
"""

import os, fcntl, time, sys, subprocess, textwrap

I2C_SLAVE = 0x0703
BUS = 3
FG_ADDR = 0x55
CHG_ADDR = 0x6A

# =====================================================================
#  I2C helpers
# =====================================================================

fd = None

def i2c_open():
    global fd
    fd = os.open(f"/dev/i2c-{BUS}", os.O_RDWR)

def i2c_close():
    os.close(fd)

def _sel(addr):
    fcntl.ioctl(fd, I2C_SLAVE, addr)

def chg_rb(reg):
    _sel(CHG_ADDR)
    os.write(fd, bytes([reg]))
    return os.read(fd, 1)[0]

def chg_wb(reg, val):
    _sel(CHG_ADDR)
    os.write(fd, bytes([reg, val]))

def fg_rw(reg):
    _sel(FG_ADDR)
    os.write(fd, bytes([reg]))
    d = os.read(fd, 2)
    return d[0] | (d[1] << 8)

def fg_ww(reg, val):
    _sel(FG_ADDR)
    os.write(fd, bytes([reg, val & 0xFF, (val >> 8) & 0xFF]))

def fg_rblk(reg, n):
    _sel(FG_ADDR)
    os.write(fd, bytes([reg]))
    return list(os.read(fd, n))

def fg_write_block(reg, data):
    _sel(FG_ADDR)
    os.write(fd, bytes([reg] + list(data)))

def s16(v):
    return v if v < 32768 else v - 65536

def bar(pct, w=25):
    f = max(0, min(w, int(pct / 100 * w)))
    return "[" + "#" * f + "-" * (w - f) + "]"

H = "=" * 62

# =====================================================================
#  BQ25895 Charger
# =====================================================================

CHG_REGS = {
    0x00: 0x26,  # IINLIM=2000mA, EN_HIZ=0, EN_ILIM=0
    0x02: 0xFD,  # Continuous ADC, ICO, Auto DPDM
    0x03: 0x5A,  # CHG=ON, OTG=OFF, SYS_MIN=3500mV, WD_RST
    0x04: 0x20,  # ICHG=2048mA
    0x05: 0x33,  # IPRECHG=256mA, ITERM=256mA
    0x06: 0x5E,  # VREG=4208mV, BATLOWV=3.0V, VRECHG=100mV
    0x07: 0x8F,  # WDT=OFF, EN_TERM=ON, CHG_TIMER=20h
    0x08: 0x03,  # BAT_COMP=0, TREG=120C
    0x09: 0x4C,  # BATFET=ON, DLY=10s, RST=enabled
    0x0A: 0x93,  # BOOSTV=5126mV, BOOST_LIM=1400mA
}

CHG_REG_DESC = {
    0x00: "IINLIM=2000mA  EN_HIZ=0  EN_ILIM=0",
    0x02: "Continuous ADC  ICO  Auto DPDM",
    0x03: "CHG=ON  OTG=OFF  SYS_MIN=3500mV",
    0x04: "ICHG=2048mA (0.2C)",
    0x05: "IPRECHG=256mA  ITERM=256mA",
    0x06: "VREG=4208mV  BATLOWV=3.0V  VRECHG=100mV",
    0x07: "WDT=OFF  EN_TERM=ON  CHG_TIMER=20h",
    0x08: "BAT_COMP=0  TREG=120C",
    0x09: "BATFET=ON  DLY=10s  RST=enabled",
    0x0A: "BOOSTV=5126mV  BOOST_LIM=1400mA",
}


def configure_charger():
    print(f"\n{H}")
    print("  BQ25895 Charger Configuration")
    print(H)

    for reg, val in CHG_REGS.items():
        chg_wb(reg, val)
        desc = CHG_REG_DESC.get(reg, "")
        print(f"  REG{reg:02X} = 0x{val:02X}   {desc}")

    chg_rb(0x0C)  # clear faults
    time.sleep(0.3)

    # Verify
    ok = True
    for reg, expect in CHG_REGS.items():
        got = chg_rb(reg)
        if reg == 0x03:
            got &= 0b10111111  # mask WD_RST (self-clearing)
            expect &= 0b10111111
        if reg == 0x02:
            got |= 0x80   # mask CONV_START (self-clearing after ADC starts)
        if got != expect:
            print(f"  FAIL  REG{reg:02X}: wrote 0x{CHG_REGS[reg]:02X}, read 0x{chg_rb(reg):02X}")
            ok = False

    faults = chg_rb(0x0C)
    if faults:
        print(f"  WARN  Faults: 0x{faults:02X}")
        ok = False

    print(f"\n  Result: {'ALL REGISTERS SET' if ok else 'SOME REGISTERS FAILED'}")
    return ok

# =====================================================================
#  BQ27220 Fuel Gauge
# =====================================================================

DM_DESIGN_CAPACITY  = 0x929C
DM_DESIGN_ENERGY    = 0x929E
DM_TERMINATE_VOLT   = 0x92A0
DM_TAPER_RATE       = 0x92A3

FG_PARAMS = [
    (DM_DESIGN_CAPACITY, 10000, "Design Capacity",  "mAh"),
    (DM_DESIGN_ENERGY,   3700,  "Design Energy",    "cWh"),
    (DM_TERMINATE_VOLT,  3000,  "Terminate Voltage","mV"),
    (DM_TAPER_RATE,      217,   "Taper Rate",       ""),
]


def fg_get_op_status():
    return fg_rw(0x3A)

def fg_get_sec():
    return (fg_get_op_status() >> 1) & 0x03

def fg_is_cfgupdate():
    return bool(fg_get_op_status() & (1 << 10))

def fg_unseal():
    fg_ww(0x00, 0x0414)
    time.sleep(0.01)
    fg_ww(0x00, 0x3672)
    time.sleep(0.01)

def fg_full_access():
    fg_ww(0x00, 0xFFFF)
    time.sleep(0.01)
    fg_ww(0x00, 0xFFFF)
    time.sleep(0.01)

def fg_seal():
    fg_ww(0x00, 0x0030)
    time.sleep(0.01)

def fg_enter_cfgupdate():
    fg_ww(0x3E, 0x0090)
    for _ in range(2000):
        time.sleep(0.001)
        if fg_is_cfgupdate():
            return True
    return False

def fg_exit_cfgupdate():
    fg_ww(0x3E, 0x0091)
    time.sleep(2.0)
    for _ in range(2000):
        time.sleep(0.001)
        if not fg_is_cfgupdate():
            return True
    return False

def fg_read_dm_u16(addr):
    fg_ww(0x3E, addr)
    time.sleep(0.005)
    raw = fg_rblk(0x40, 2)
    return (raw[0] << 8) | raw[1]

def fg_write_dm_u16(addr, val):
    payload = [addr & 0xFF, (addr >> 8) & 0xFF, (val >> 8) & 0xFF, val & 0xFF]
    fg_write_block(0x3E, payload)
    time.sleep(0.001)
    cksum = (0xFF - (sum(payload) & 0xFF)) & 0xFF
    total_len = len(payload) + 2
    fg_write_block(0x60, [cksum, total_len])
    time.sleep(0.02)


def configure_fuel_gauge():
    print(f"\n{H}")
    print("  BQ27220 Fuel Gauge Configuration (10000mAh)")
    print(H)

    # Unseal + full access
    fg_unseal()
    fg_full_access()
    sec = fg_get_sec()
    sec_s = {0b11: "SEALED", 0b10: "UNSEALED", 0b01: "FULL_ACCESS"}.get(sec, "?")
    print(f"  Security: {sec_s}")

    # Read current values
    print(f"\n  Current DM values:")
    needs_update = False
    for addr, expect, name, unit in FG_PARAMS:
        val = fg_read_dm_u16(addr)
        match = "OK" if val == expect else "NEEDS UPDATE"
        if val != expect:
            needs_update = True
        print(f"    {name:22s}: {val:>6d} {unit:4s}  [{match}]")

    if not needs_update:
        print(f"\n  Fuel gauge already configured correctly.")
        fg_seal()
        return True

    # Enter CFGUPDATE
    print(f"\n  Entering CFGUPDATE...")
    if fg_enter_cfgupdate():
        print("  OK")
    else:
        print("  CFGUPDATE flag not detected, trying anyway")

    # Write parameters
    print(f"  Writing parameters:")
    for addr, val, name, unit in FG_PARAMS:
        fg_write_dm_u16(addr, val)
        print(f"    {name:22s} = {val} {unit}")

    # Verify in CFGUPDATE
    time.sleep(0.1)
    ok = True
    print(f"  Verifying:")
    for addr, expect, name, unit in FG_PARAMS:
        got = fg_read_dm_u16(addr)
        status = "OK" if got == expect else f"FAIL ({got})"
        if got != expect:
            ok = False
        print(f"    {name:22s}: {expect:>6d}  [{status}]")

    # Exit CFGUPDATE
    print(f"  Exiting CFGUPDATE...")
    if fg_exit_cfgupdate():
        print("  OK")
    else:
        print("  Timeout")

    fg_seal()
    print(f"\n  Result: {'ALL PARAMETERS SET' if ok else 'SOME PARAMETERS FAILED'}")
    return ok

# =====================================================================
#  Battery Management Daemon
# =====================================================================

DAEMON_PATH = "/opt/battery-mgr/battery_daemon.py"
SERVICE_PATH = "/etc/systemd/system/battery-mgr.service"
SHIP_MODE_PATH = "/opt/battery-mgr/ship_mode.py"

DAEMON_SOURCE = textwrap.dedent('''\
    #!/usr/bin/env python3
    """Battery Management Daemon - BQ25895 + BQ27220"""

    import os, fcntl, time, sys, subprocess, signal, logging

    I2C_SLAVE = 0x0703
    BUS, FG, CHG = 3, 0x55, 0x6A
    SOC_WARN, SOC_SHUT, SOC_CRIT = 20, 15, 10
    V_CUTOFF, T_CUTOFF = 3200, 60
    POLL, LOG_IV = 30, 300

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("/var/log/battery-mgr.log"), logging.StreamHandler()])
    log = logging.getLogger("battery-mgr")
    running = True

    def _sig(s, f):
        global running; running = False
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    class Bus:
        def __init__(self):
            self.fd = os.open(f"/dev/i2c-{BUS}", os.O_RDWR); self._a = None
        def _s(self, a):
            if self._a != a: fcntl.ioctl(self.fd, I2C_SLAVE, a); self._a = a
        def rb(self, a, r):
            self._s(a); os.write(self.fd, bytes([r])); return os.read(self.fd, 1)[0]
        def rw(self, a, r):
            self._s(a); os.write(self.fd, bytes([r])); d = os.read(self.fd, 2); return d[0]|(d[1]<<8)
        def wb(self, a, r, v):
            self._s(a); os.write(self.fd, bytes([r, v]))
        def close(self): os.close(self.fd)

    def cfg_chg(b):
        for r, v in [(0x00,0x26),(0x02,0xFD),(0x03,0x5A),(0x04,0x20),(0x05,0x33),
                      (0x06,0x5E),(0x07,0x8F),(0x08,0x03),(0x09,0x4C),(0x0A,0x93)]:
            b.wb(CHG, r, v)
        b.rb(CHG, 0x0C)
        log.info("Charger configured: ICHG=2048mA IINLIM=2000mA VREG=4208mV WDT=OFF TIMER=20h")

    def ship_mode(b):
        log.critical("ENTERING SHIP MODE - BATFET disconnect in 10s")
        b.wb(CHG, 0x09, 0x6C)

    def read_fg(b):
        v = b.rw(FG, 0x08); t = b.rw(FG, 0x06)/10-273.15; soc = b.rw(FG, 0x2C)
        c = b.rw(FG, 0x14); c = c if c < 32768 else c - 65536
        return v, t, soc, c

    def read_chg(b):
        s = b.rb(CHG, 0x0B); f = b.rb(CHG, 0x0C)
        return (s>>3)&3, bool((s>>2)&1), f, 2600+(b.rb(CHG,0x11)&0x7F)*100, (b.rb(CHG,0x12)&0x7F)*50

    def shutdown(b, reason):
        log.critical("SHUTDOWN: %s", reason)
        subprocess.run(["wall", f"BATTERY LOW ({reason}). Shutting down!"], check=False)
        ship_mode(b)
        subprocess.run(["sync"], check=False)
        subprocess.run(["systemctl", "poweroff"], check=False)

    def main():
        global running
        log.info("Battery Management Daemon starting")
        b = Bus(); cfg_chg(b)
        lt, warned = 0, False
        while running:
            try:
                v, t, soc, c = read_fg(b); now = time.time(); dsc = c <= 0
                if now - lt >= LOG_IV:
                    try:
                        cs, pg, fl, vb, ic = read_chg(b)
                        log.info("BAT: %dmV %d%% %+dmA %.1fC | CHG: %s PG=%s VBUS=%d ICHG=%d F=0x%02X",
                            v, soc, c, t, ["Off","Pre","Fast","Done"][cs], "Y" if pg else "N", vb, ic, fl)
                    except: log.info("BAT: %dmV %d%% %+dmA %.1fC", v, soc, c, t)
                    lt = now
                if soc <= SOC_CRIT and dsc: shutdown(b, f"SOC={soc}%% CRITICAL"); break
                if soc <= SOC_SHUT and dsc: shutdown(b, f"SOC={soc}%%"); break
                if soc <= SOC_WARN and dsc and not warned:
                    log.warning("SOC=%d%% LOW BATTERY", soc)
                    subprocess.run(["wall", f"LOW BATTERY: {soc}%%. Shutdown at {SOC_SHUT}%%."], check=False)
                    warned = True
                elif soc > SOC_WARN: warned = False
                if v < V_CUTOFF and dsc: shutdown(b, f"V={v}mV"); break
                if t > T_CUTOFF: shutdown(b, f"T={t:.0f}C"); break
            except Exception as e: log.error("Error: %s", e)
            time.sleep(POLL)
        b.close(); log.info("Daemon stopped")

    if __name__ == "__main__": main()
''')

SERVICE_SOURCE = textwrap.dedent('''\
    [Unit]
    Description=Battery Management Daemon (BQ25895 + BQ27220)
    After=multi-user.target

    [Service]
    Type=simple
    ExecStart=/usr/bin/python3 /opt/battery-mgr/battery_daemon.py
    Restart=on-failure
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
''')

SHIP_MODE_SOURCE = textwrap.dedent('''\
    #!/usr/bin/env python3
    """Enter BQ25895 ship mode. Wake up by plugging USB charger."""
    import os, fcntl, sys
    fd = os.open("/dev/i2c-3", os.O_RDWR)
    fcntl.ioctl(fd, 0x0703, 0x6A)
    if "--force" not in sys.argv:
        print("WARNING: This disconnects battery from system (power off in ~10s).")
        print("Wake up: plug in USB charger.\\nUse --force to confirm.")
        os.close(fd); sys.exit(0)
    print("Entering ship mode..."); os.write(fd, bytes([0x09, 0x6C]))
    print("BATFET disconnect in ~10s."); os.system("sync"); os.close(fd)
''')


def install_daemon():
    print(f"\n{H}")
    print("  Install Battery Management Daemon")
    print(H)

    os.makedirs("/opt/battery-mgr", exist_ok=True)

    with open(DAEMON_PATH, "w") as f:
        f.write(DAEMON_SOURCE)
    os.chmod(DAEMON_PATH, 0o755)
    print(f"  Wrote {DAEMON_PATH}")

    with open(SHIP_MODE_PATH, "w") as f:
        f.write(SHIP_MODE_SOURCE)
    os.chmod(SHIP_MODE_PATH, 0o755)
    print(f"  Wrote {SHIP_MODE_PATH}")

    with open(SERVICE_PATH, "w") as f:
        f.write(SERVICE_SOURCE)
    print(f"  Wrote {SERVICE_PATH}")

    subprocess.run(["systemctl", "daemon-reload"], check=False, capture_output=True)
    subprocess.run(["systemctl", "enable", "battery-mgr.service"],
                   check=False, capture_output=True)
    subprocess.run(["systemctl", "stop", "battery-mgr.service"],
                   check=False, capture_output=True, timeout=15)
    time.sleep(2)
    subprocess.run(["systemctl", "start", "battery-mgr.service"],
                   check=False, capture_output=True)
    time.sleep(3)

    r = subprocess.run(["systemctl", "is-active", "battery-mgr"],
                       capture_output=True, text=True)
    active = r.stdout.strip() == "active"
    print(f"  Service: {'RUNNING' if active else 'FAILED'}")

    print(f"\n  Protection thresholds:")
    print(f"    SOC warning:    20%")
    print(f"    SOC shutdown:   15% (enters ship mode)")
    print(f"    SOC critical:   10% (immediate ship mode)")
    print(f"    Voltage cutoff: 3200 mV")
    print(f"    Temp cutoff:    60 C")
    print(f"\n  Ship mode: sudo python3 {SHIP_MODE_PATH} --force")
    return active

# =====================================================================
#  Status display
# =====================================================================

def show_status():
    print(f"\n{H}")
    print("  Power System Status")
    print(H)

    # Fuel gauge
    v = fg_rw(0x08); t = fg_rw(0x06)/10-273.15; soc = fg_rw(0x2C)
    cur = s16(fg_rw(0x14)); rem = fg_rw(0x10); fcc = fg_rw(0x12)
    ops = fg_rw(0x3A)
    print(f"\n  --- Fuel Gauge (BQ27220) ---")
    print(f"  Voltage:        {v} mV  ({v/1000:.3f}V)")
    print(f"  Temperature:    {t:.1f} C")
    print(f"  SOC:            {soc}% {bar(min(soc,100))}")
    print(f"  Current:        {cur:+d} mA")
    print(f"  Remaining:      {rem} mAh / {fcc} mAh")
    print(f"  OpStatus:       0x{ops:04X}")

    # Charger
    r0b = chg_rb(0x0B); r0c = chg_rb(0x0C)
    vbus = 2600 + (chg_rb(0x11) & 0x7F) * 100
    vbat = 2304 + (chg_rb(0x0E) & 0x7F) * 20
    ichg = (chg_rb(0x12) & 0x7F) * 50
    chrg_s = ["Not charging","Pre-charge","Fast charge","Charge done"][(r0b>>3)&3]
    iinlim = 100 + (chg_rb(0x00) & 0x3F) * 50
    ichg_set = (chg_rb(0x04) & 0x7F) * 64
    vreg = 3840 + ((chg_rb(0x06) >> 2) & 0x3F) * 16

    print(f"\n  --- Charger (BQ25895) ---")
    print(f"  VBUS:           {vbus} mV  PG={'YES' if (r0b>>2)&1 else 'NO'}")
    print(f"  VBAT:           {vbat} mV")
    print(f"  Status:         {chrg_s}")
    print(f"  Charge Current: {ichg} mA (ADC)")
    print(f"  Settings:       IINLIM={iinlim}mA  ICHG={ichg_set}mA  VREG={vreg}mV")
    print(f"  Faults:         0x{r0c:02X} {'(clear)' if r0c==0 else '(!!)'}")

    # Daemon
    r = subprocess.run(["systemctl","is-active","battery-mgr"],
                       capture_output=True, text=True)
    print(f"\n  --- Daemon ---")
    print(f"  battery-mgr:    {r.stdout.strip()}")

    # DM config
    fg_unseal()
    print(f"\n  --- DM Config ---")
    for addr, expect, name, unit in FG_PARAMS:
        val = fg_read_dm_u16(addr)
        s = "OK" if val == expect else "MISMATCH"
        print(f"    {name:22s}: {val:>6d} {unit:4s} [{s}]")
    fg_seal()
    print()

# =====================================================================
#  Main
# =====================================================================

def main():
    if os.geteuid() != 0:
        print("Error: Must run as root (sudo).")
        sys.exit(1)

    i2c_open()

    if "--status" in sys.argv:
        show_status()
        i2c_close()
        return

    if "--ship-mode" in sys.argv:
        if "--force" not in sys.argv:
            print("Enter ship mode: disconnects battery, system powers off in ~10s.")
            print("Wake up by plugging USB charger.")
            print("Use: sudo python3 power_config.py --ship-mode --force")
            i2c_close()
            return
        print("Entering ship mode...")
        chg_wb(0x09, 0x6C)
        print("BATFET disconnect in ~10s. Syncing...")
        os.system("sync")
        i2c_close()
        return

    print()
    print("##############################################################")
    print("#       Smart Eye Power System Configuration                 #")
    print(f"#       {time.strftime('%Y-%m-%d %H:%M:%S'):>45s}   #")
    print("##############################################################")

    ok1 = configure_charger()
    ok2 = configure_fuel_gauge()
    ok3 = install_daemon()

    print(f"\n{H}")
    print("  SUMMARY")
    print(H)
    print(f"  Charger (BQ25895):  {'OK' if ok1 else 'FAILED'}")
    print(f"  Fuel Gauge (BQ27220): {'OK' if ok2 else 'FAILED'}")
    print(f"  Daemon:             {'OK' if ok3 else 'FAILED'}")
    print(f"\n  Battery: 10000mAh  Charge: 2048mA @ 4.208V  Input: 2000mA")
    print(f"  Cutoff: 15% SOC -> ship mode -> USB wake")
    print(H)
    print()

    i2c_close()


if __name__ == "__main__":
    main()
