# Base DTB Patches

> **Automated:** Patches 2 and 3 are applied automatically by
> `sudo bash scripts/setup_device.sh --phase=3`.
> Run that before the first reboot instead of applying them manually.

## Warning

All patches below modify the base DTB directly. A kernel update will
overwrite them. Re-apply after kernel upgrades:

```bash
sudo bash scripts/setup_device.sh --phase=3
```

The original (unpatched) DTB is kept at `rk3588s-radxa-cm5-io.dtb.bak.usb`.

---

# Patch 1: Disable FIQ Debugger on UART2

UART2 (`serial@feb50000`) is claimed by the FIQ debugger in the stock base
DTB. This conflicts with using UART2 for peripherals.

## What was changed

In `/usr/lib/linux-image-<KVER>/rockchip/rk3588s-radxa-cm5-io.dtb`:

```
fiq-debugger {
    rockchip,serial-id = <0xffffffff>;  // was <0x02> (UART2)
    status = "disabled";                 // was "okay"
};
```

## Why

Without this patch U-Boot's 1.5 Mbaud console output on UART2 causes a
connected sensor to respond with garbage data and can corrupt the boot
process.

## How to apply (manual)

```bash
KVER=$(uname -r)
DTB="/usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb"
sudo cp "$DTB" "${DTB}.bak"
dtc -I dtb -O dts "$DTB" -o /tmp/base.dts 2>/dev/null
# Edit /tmp/base.dts: in the fiq-debugger node set
#   rockchip,serial-id = <0xffffffff>;
#   status = "disabled";
dtc -I dts -O dtb /tmp/base.dts -o /tmp/patched.dtb 2>/dev/null
sudo cp /tmp/patched.dtb "$DTB"
sudo reboot
```

## Restore

```bash
KVER=$(uname -r)
sudo cp /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb.bak \
        /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb
```

---

# Patch 2: Remove fusb302 vbus-supply (USB Type-C Fix)

**Automated by `setup_device.sh --phase=3`.**

## What was changed

In `/usr/lib/linux-image-<KVER>/rockchip/rk3588s-radxa-cm5-io.dtb`,
inside the `fusb302@22` node (under `i2c@fec80000`):

```
/* removed: */
vbus-supply = <0x168>;
```

## Why

The base DTB's `fusb302` node references `vbus5v0_typec` (phandle 0x168) as
its `vbus-supply`. On the stock CM5-IO board the `vbus5v0_typec` fixed-regulator
uses GPIO4_A5 to control a TPS22916 load switch. On the Smart Eye carrier board:

- GPIO4_A5 is used for UART3-M2 TX (front ultrasonic sensor)
- The TPS22916 ON pin is hardwired to VCC5V0_SYS (always enabled, no GPIO needed)

Result: `vbus5v0_typec` regulator can never claim GPIO4_A5 (busy with UART3),
returns EPROBE_DEFER indefinitely → fusb302 driver never finishes probing →
DWC3 USB controller has no VBUS detection events → USB Type-C not detected.

Removing the `vbus-supply` property lets `devm_regulator_get_optional()` return
`-ENODEV` (clean skip) instead of looping forever on EPROBE_DEFER.

## How to apply (manual — prefer setup_device.sh)

```bash
KVER=$(uname -r)
DTB="/usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb"
sudo cp "$DTB" "${DTB}.bak.usb"
dtc -I dtb -O dts "$DTB" -o /tmp/base.dts 2>/dev/null
# Find and delete the vbus-supply line inside fusb302@22:
grep -n "vbus-supply" /tmp/base.dts   # note the line number
sed -i '<LINE>d' /tmp/base.dts
dtc -I dts -O dtb /tmp/base.dts -o /tmp/patched.dtb 2>/dev/null
sudo cp /tmp/patched.dtb "$DTB"
sudo reboot
```

## Restore

```bash
KVER=$(uname -r)
sudo cp /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb.bak.usb \
        /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb
```

---

# Patch 3: Remove TCPM port endpoint dependencies (USB Type-C deadlock fix)

**Automated by `setup_device.sh --phase=3`.**

## What was changed

In `/usr/lib/linux-image-<KVER>/rockchip/rk3588s-radxa-cm5-io.dtb`,
inside the `fusb302@22` node and its `connector` sub-node:

```
/* removed from fusb302@22 top-level: */
ports {
    port@0 { endpoint@0 { remote-endpoint = <0x169>; }; };  /* DWC3 role switch */
};

/* removed from fusb302@22/connector: */
altmodes { altmode@0 { svid = <0xff01>; vdo = <0xffffffff>; }; };  /* DP altmode */

ports {
    /* port@0 removed: orientation switch → fed80000.phy endpoint@0 */
    /* port@1 removed: DP alt-mode mux  → fed80000.phy endpoint@1  */
};
```

## Why

Three circular deadlocks blocked the fusb302 TCPM from ever probing:

1. **DP alt-mode mux** (`connector/port@1` → `fed80000.phy endpoint@1`): The USBDP
   PHY defers registering its DP mux until it receives CC info from the fusb302
   connector. TCPM calls `typec_mux_get()` and gets EPROBE_DEFER forever.
   The Smart Eye carrier has no DisplayPort hardware, so this is not needed.

2. **USB orientation switch** (`connector/port@0` → `fed80000.phy endpoint@0`):
   Same circular dependency with the USBDP PHY's orientation switch. The PHY
   handles USB lane orientation internally once it probes.

3. **USB role switch** (`fusb302 top-level port@0/endpoint@0` → `fc000000.usb`):
   DWC3 role switch endpoint. Not needed because `rk3588-dwc3-peripheral.dtbo`
   already forces `dr_mode = "peripheral"` statically.

## How to apply (manual — prefer setup_device.sh)

```bash
KVER=$(uname -r)
DTB="/usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb"
dtc -I dtb -O dts "$DTB" -o /tmp/base.dts 2>/dev/null

python3 - <<'EOF'
import re, sys

DTS = "/tmp/base.dts"
with open(DTS) as f:
    text = f.read()

def blk_end(t, i):
    d = 0
    while i < len(t):
        if t[i] == "{": d += 1
        elif t[i] == "}":
            d -= 1
            if d == 0:
                e = i + 1
                if e < len(t) and t[e] == ";": e += 1
                if e < len(t) and t[e] == "\n": e += 1
                return e
        i += 1
    return len(t)

def ln_start(t, i): return t.rfind("\n", 0, i) + 1

def find_blk(t, pat, lo=0, hi=None):
    m = re.search(pat, t[lo:hi])
    if not m: return None
    abs_hdr = lo + m.start()
    brace = lo + m.end() - 1
    while brace < len(t) and t[brace] != "{": brace += 1
    return (ln_start(t, abs_hdr), blk_end(t, brace))

def rm_blk(t, pat, lo=0, hi=None):
    b = find_blk(t, pat, lo, hi)
    if b is None: return t, False
    return t[:b[0]] + t[b[1]:], True

def rm_prop(t, prop, lo=0, hi=None):
    m = re.search(r"[ \t]*" + re.escape(prop) + r"\s*=\s*[^;\n]*;\s*\n", t[lo:hi])
    if not m: return t, False
    s, e = lo + m.start(), lo + m.end()
    return t[:s] + t[e:], True

fusb = find_blk(text, r"fusb302@22\s*\{")
if fusb is None: sys.exit("ERROR: fusb302@22 not found")

text, _ = rm_prop(text, "vbus-supply", *fusb)
fusb = find_blk(text, r"fusb302@22\s*\{")

conn = find_blk(text, r"\bconnector\s*\{", *fusb)
pre_conn = conn[0] if conn else fusb[1]
text, _ = rm_blk(text, r"\bports\s*\{", fusb[0], pre_conn)
fusb = find_blk(text, r"fusb302@22\s*\{")
conn = find_blk(text, r"\bconnector\s*\{", *fusb)

if conn:
    text, _ = rm_blk(text, r"\baltmodes\s*\{", *conn)
    fusb = find_blk(text, r"fusb302@22\s*\{")
    conn = find_blk(text, r"\bconnector\s*\{", *fusb)
    text, _ = rm_blk(text, r"\bports\s*\{", *conn)

with open(DTS, "w") as f: f.write(text)
print("Done")
EOF

dtc -I dts -O dtb /tmp/base.dts -o /tmp/patched.dtb 2>/dev/null
sudo cp /tmp/patched.dtb "$DTB"
sudo reboot
```

## Restore

```bash
KVER=$(uname -r)
sudo cp /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb.bak.usb \
        /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb
```
