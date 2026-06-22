# BQ25895 Battery Charger Configuration Guide

## Overview

The **BQ25895** is a highly integrated, synchronous battery charger IC from Texas Instruments. It manages charging for a single-cell Li-ion/Li-polymer battery with configurable parameters for optimal performance and battery health.

**Hardware:**
- **IC:** Texas Instruments BQ25895
- **Interface:** I2C3 (bus 3)
- **I2C Address:** 0x6A
- **Battery:** 10000mAh Li-polymer (3.7V nominal, 4.2V max)
- **Mount:** U2 on Smart Eye carrier board

---

## Configuration Parameters for 10000mAh Battery

### 1. Input Current Limit (IINLIM)
**Purpose:** Limits current drawn from USB/power source to prevent voltage collapse

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | REG00 [5:0] | 6-bit field |
| **Range** | 100–3250 mA | Offset 100mA, step **50mA** |
| **Recommended** | 2000 mA | USB-C or DCP adapter |
| **Default** | 100 mA | Very conservative |

**Formula:** `Input Current = 100 + (IINLIM_CODE × 50)` mA

---

### 2. Charging Current (ICHG)
**Purpose:** Main fast-charge current

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | REG04 [6:0] | 7-bit field |
| **Range** | 0–5056 mA | Offset 0mA, step 64mA |
| **Recommended** | 2048 mA | ~0.2C for 10Ah |
| **Default** | 2048 mA (code 32) | |

**Formula:** `Charge Current = ICHG_CODE × 64` mA

---

### 3. Pre-Charge Current (IPRECHG)
**Purpose:** Low-current recovery when battery voltage < BATLOWV (~3.0V)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | REG05 [7:4] | 4-bit field |
| **Range** | 64–1024 mA | Offset 64mA, step **64mA** |
| **Recommended** | 256 mA | |
| **Default** | 128 mA (code 1) | |

**Formula:** `Pre-charge Current = 64 + (IPRECHG_CODE × 64)` mA

---

### 4. Termination Current (ITERM)
**Purpose:** End-of-charge detection — charging stops when current drops below this threshold

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | REG05 [3:0] | 4-bit field |
| **Range** | 64–1024 mA | Offset 64mA, step **64mA** |
| **Recommended** | 256 mA | ~2.5% of 10Ah capacity |
| **Default** | 128 mA (code 1) | |

**Formula:** `Termination Current = 64 + (ITERM_CODE × 64)` mA

---

### 5. Charge Voltage Limit (VREG)
**Purpose:** Maximum battery regulation voltage

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | REG06 [7:2] | 6-bit field |
| **Range** | 3840–**4608** mV | Offset 3840mV, step 16mV |
| **Recommended** | 4208 mV | Standard Li-ion |
| **Default** | 4208 mV (code 23) | |

**Formula:** `Charge Voltage = 3840 + (VREG_CODE × 16)` mV

**Critical limits:**
- **4200 mV** = standard full charge, longest cycle life
- **4100 mV** = ~2× cycle life trade-off vs capacity
- **Do NOT exceed 4608 mV** (hardware maximum)

---

### 6. Minimum System Voltage (SYS_MIN)
**Purpose:** Minimum SYS voltage; charger prioritizes system over battery when SYS drops to this level

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | **REG03 [3:1]** | 3-bit field |
| **Range** | 3000–3700 mV | Offset 3000mV, step 100mV |
| **Recommended** | 3500 mV | |
| **Default** | 3500 mV (code 5) | |

**Formula:** `SYS_MIN = 3000 + (SYS_MIN_CODE × 100)` mV

---

### 7. Thermal Regulation (TREG)
**Purpose:** Reduce charge current when IC junction temperature reaches threshold

| Code | Threshold | Notes |
|------|-----------|-------|
| 00 | **60°C** | Most aggressive derating |
| 01 | 80°C | |
| 10 | 100°C | |
| 11 | **120°C** | Default; least aggressive derating |

**Register:** REG08 [1:0]

**Recommended:** 0b11 (120°C) — let the charge current hold; reduce ICHG in software if thermals are a concern.

---

### 8. Charging Safety Timer
**Purpose:** Abort charge if not complete within timeout (protection against stuck-charge)

| Register | Field | Function |
|----------|-------|----------|
| REG07 [3] | EN_TIMER | 1 = safety timer enabled |
| REG07 [2:1] | CHG_TIMER | Timeout selection |

**Timeout Options:** 00=5h, 01=8h, 10=12h, 11=20h

---

### 9. Watchdog Timer (WATCHDOG)
**Purpose:** Resets charger registers to default if software stops refreshing

| Code | Timeout |
|------|---------|
| 00 | **Disabled** |
| 01 | 40 s |
| 10 | 80 s |
| 11 | 160 s |

**Register:** REG07 [5:4]

**Recommended:** Disable (00) — prevents silent resets wiping configuration.

---

## Optimal Configuration for 10000mAh

```python
charger.configure_for_10ah_battery()
```

Applied settings:

```
Input Current Limit:     2000 mA  (code 38, REG00)
Charging Current:        2048 mA  (code 32, REG04)
Pre-charge Current:       256 mA  (code  3, REG05[7:4])
Termination Current:      256 mA  (code  3, REG05[3:0])
Charge Voltage:          4208 mV  (code 23, REG06)
SYS_MIN:                 3500 mV  (code  5, REG03)
Thermal Regulation:       120°C   (code 11, REG08)
Safety Timer:              20 h   (code 11, REG07)
Watchdog:             Disabled    (code 00, REG07)
```

---

## Register Map (verified vs datasheet)

| Address | Name | Key Fields |
|---------|------|------------|
| 0x00 | Input Source Control | EN_HIZ[7], EN_ILIM[6], IINLIM[5:0] |
| 0x01 | ADC / VINDPM Offset | BHOT[7:6], BCOLD[5], VINDPM_OS[4:0] |
| 0x02 | ADC Control | CONV_START[7], CONV_RATE[6], ICO_EN[4], AUTO_DPDM[0] |
| 0x03 | Charge Control | OTG_CONFIG[5], CHG_CONFIG[4], SYS_MIN[3:1] |
| 0x04 | Charge Current | EN_PUMPX[7], ICHG[6:0] |
| 0x05 | Precharge / Termination | IPRECHG[7:4], ITERM[3:0] |
| 0x06 | Charge Voltage | VREG[7:2], BATLOWV[1], VRECHG[0] |
| 0x07 | Charge Timer / Watchdog | EN_TERM[7], WATCHDOG[5:4], EN_TIMER[3], CHG_TIMER[2:1] |
| 0x08 | IR Comp / Thermal | BAT_COMP[7:5], VCLAMP[4:2], TREG[1:0] |
| 0x09 | BATFET Control | FORCE_ICO[7], TMR2X_EN[6], BATFET_DIS[5], BATFET_DLY[3], BATFET_RST_EN[2] |
| 0x0A | Boost Voltage | BOOSTV[7:4] |
| 0x0B | **Status (Read-only)** | VBUS_STAT[7:5], CHRG_STAT[4:3], PG_STAT[2], VSYS_STAT[0] |
| 0x0C | **Fault (Read-only)** | WATCHDOG_FAULT[7], BOOST_FAULT[6], CHRG_FAULT[5:4], BAT_FAULT[3], NTC_FAULT[2:0] |
| 0x0D | VINDPM / IDPM | IDPM_INT_MASK[1:0] |
| 0x0E | ADC VBAT | VBAT_ADC[6:0] — VBAT = 2304 + val×20 mV |
| 0x0F | ADC VSYS | VSYS_ADC[6:0] — VSYS = 2304 + val×20 mV |
| 0x10 | ADC TS | TSPCT[6:0] — TS% = 21 + val×0.465 % |
| 0x11 | ADC VBUS | VBUS_ADC[6:0] — VBUS = 2600 + val×100 mV |
| 0x12 | ADC ICHG | ICHG_ADC[6:0] — ICHG = val×50 mA |
| 0x13 | ADC IDPM | IDPM_LIM[5:0] — IDPM = 100 + val×50 mA |
| 0x14 | **Device ID** | PN[5:3]=111 (BQ25895), DEV_REV[1:0]=01 |

---

## Status Register REG0B (Read-only)

```
Bits [7:5]  VBUS_STAT — Input type detected
            000 = No Input
            001 = USB SDP (500mA)
            010 = USB CDP (1.5A)
            011 = USB DCP (3.25A)
            100 = Adj HV DCP
            101 = Unknown Adapter
            110 = Non-Standard Adapter
            111 = OTG

Bits [4:3]  CHRG_STAT — Charge status
            00 = Not Charging
            01 = Pre-charge
            10 = Fast Charging
            11 = Charge Done

Bit  [2]    PG_STAT — Power Good (1 = VBUS valid)
Bit  [1]    SDP_STAT
Bit  [0]    VSYS_STAT — 1 = VSYS regulation active
```

**Note:** REG0B is always current — read once.

---

## Fault Register REG0C (Read-only)

```
Bit  [7]    WATCHDOG_FAULT — 1 = watchdog expired
Bit  [6]    BOOST_FAULT
Bits [5:4]  CHRG_FAULT
            00 = Normal
            01 = Input Overvoltage (OVP)
            10 = Thermal Shutdown
            11 = Safety Timer Expired
Bit  [3]    BAT_FAULT — 1 = VBAT OVP
Bits [2:0]  NTC_FAULT — TS pin thermistor fault
            000 = Normal
            010 = TS Warm
            011 = TS Cool
            101 = TS Cold
            110 = TS Hot (boost)
            111 = TS Cold (boost)
```

**Note:** REG0C latches faults — must be **read twice** to get current state (first read clears latch).

---

## Ship Mode (BATFET_DIS)

To enter ship mode (battery isolated from SYS, ~2µA drain):

```bash
# REG09 = 0x6C = TMR2X_EN | BATFET_DIS | BATFET_DLY | BATFET_RST_EN
# BATFET_DLY=1 adds 10s delay so OS can finish writing before disconnect
sudo i2cset -y 3 0x6a 0x09 0x6c b
```

Bit map of 0x6C (0b01101100):
- Bit 6 TMR2X_EN=1
- Bit 5 BATFET_DIS=1 (disconnect)
- Bit 3 BATFET_DLY=1 (10s delay)
- Bit 2 BATFET_RST_EN=1 (allow wake on plug-in)

---

## Python Usage

### Quick Configuration
```python
from pathpal_project.bq25895_charger import BQ25895

charger = BQ25895()
charger.configure_for_10ah_battery()
charger.close()
```

### Status Monitoring
```python
charger = BQ25895()
status = charger.get_status()   # reads REG0B + REG0C (twice)
print(status['charging_state'])
if status['battery_fault']:
    print("Battery fault!")
charger.close()
```

---

## Troubleshooting

### Charger not detected
```
→ Verify I2C3 is enabled in device tree
→ i2cdetect -y 3   (expect 0x6A)
→ Check BQ25895 solder joints
```

### Charging not starting
```
→ Check CHG_CONFIG (REG03[4]) = 1
→ Check EN_HIZ (REG00[7]) = 0
→ Check PG_STAT (REG0B[2]) = 1 (power good)
→ sudo i2cget -y 3 0x6a 0x0b b
```

### Fault LED blinking
```
→ Read REG0C twice: sudo i2cget -y 3 0x6a 0x0c b
→ Check NTC_FAULT — if no thermistor: TS pin must be tied correctly
```

### Excessive heat during charging
```
→ Reduce ICHG (configure_charging_current)
→ Increase adapter current rating to reduce (VIN-VBAT) drop
→ TREG=11 (120°C) allows full current until junction hits 120°C
```

---

**Last Updated:** 2026-06-21
