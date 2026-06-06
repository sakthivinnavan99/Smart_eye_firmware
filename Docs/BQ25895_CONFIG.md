# BQ25895 Battery Charger Configuration Guide

## Overview

The **BQ25895** is a highly integrated, synchronous battery charger IC from Texas Instruments. It manages charging for a single-cell Li-ion/Li-polymer battery with configurable parameters for optimal performance and battery health.

**Hardware:**
- **IC:** Texas Instruments BQ25895
- **Interface:** I2C3
- **I2C Address:** 0x6A
- **Battery:** 10000mAh Li-polymer (3.7V nominal, 4.2V max)
- **Mount:** U2 on Smart Eye carrier board

---

## Configuration Parameters for 10000mAh Battery

### 1. Input Current Limit (ILIM)
**Purpose:** Limits current drawn from USB/power source to prevent voltage drop

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | 0x00 [5:0] | 6-bit field |
| **Range** | 100-3250 mA | Step: 100mA |
| **Recommended** | 2000 mA | Safe for USB 2.0 |
| **Default** | 100 mA | Very conservative |

**Formula:** `Input Current = 100 + (ILIM_CODE × 100)` mA

**Why 2000mA?**
- USB 2.0 provides up to 500mA
- USB 3.0/3.1 provides up to 900mA
- Type-C with PD can provide 1.5-3A
- 2000mA is safe headroom for most USB supplies

---

### 2. Charging Current (ICHG)
**Purpose:** Main charging current when battery is in fast-charge phase

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | 0x02 [7:2] | 6-bit field |
| **Range** | 0-5056 mA | Step: 64mA |
| **Recommended** | 2048 mA | 20% of capacity/hour |
| **Default** | 2048 mA | ✓ Good default |

**Formula:** `Charging Current = ICHG_CODE × 64` mA

**Why 2048mA?**
- 10000mAh ÷ 2048mA ≈ 4.88 hours (safe charge time)
- Not too aggressive (avoids heating)
- Not too slow (practical charge time)
- Roughly 20% C-rate (industry standard for Li-ion)

**Heat Dissipation:**
```
Power dissipated in charger ≈ (VIN - VBAT) × ICHG
  = (5V - 4.2V) × 2048mA
  ≈ 1.6W (acceptable with heatsinking)
```

---

### 3. Pre-Charge Current (IPRECHG)
**Purpose:** Ultra-low current when battery voltage < 3.0V (recovery from over-discharge)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | 0x03 [7:4] | 4-bit field |
| **Range** | 16-512 mA | Step: 16mA |
| **Recommended** | 192 mA | Safe for recovery |
| **Default** | 128 mA | Conservative |

**Formula:** `Pre-charge Current = 16 + (IPRECHG_CODE × 16)` mA

**Why 192mA?**
- ~2% of battery capacity (very gentle)
- Allows safe recovery from deep discharge
- Reduces stress on degraded batteries

---

### 4. Termination Current (ITERM)
**Purpose:** End-of-charge detection — charging stops when current drops below this

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | 0x03 [3:0] | 4-bit field |
| **Range** | 16-512 mA | Step: 16mA |
| **Recommended** | 256 mA | 2.5% of capacity |
| **Default** | 128 mA | Safe but slower |

**Formula:** `Termination Current = 16 + (ITERM_CODE × 16)` mA

**Why 256mA?**
- ~2.5% of 10Ah (standard Li-ion practice)
- Balances charge speed with battery longevity
- Ensures full charge without overcharging

---

### 5. Charge Voltage Limit (VREG)
**Purpose:** Maximum battery voltage regulation point

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | 0x04 [7:2] | 6-bit field |
| **Range** | 3840-4496 mV | Step: 16mV |
| **Recommended** | 4208 mV | Standard Li-ion |
| **Default** | 4208 mV | ✓ Correct for Li-ion |

**Formula:** `Charge Voltage = 3840 + (VREG_CODE × 16)` mV

**Critical:** 
- **4.2V (4200mV)** = 100% state-of-charge, longest cycle life
- **4.25V (4250mV)** = 3% more capacity, ~5% fewer cycles
- **4.3V (4300mV)** = 5% more capacity, ~20% fewer cycles
- Do NOT exceed 4.35V (battery damage risk)

---

### 6. Minimum System Voltage (VSYSMIN)
**Purpose:** Minimum voltage to maintain system operation (prioritizes system over battery charge)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Register** | 0x01 [3:1] | 3-bit field |
| **Range** | 3000-3700 mV | Step: 100mV |
| **Recommended** | 3600 mV | System power priority |
| **Default** | 3400 mV | Battery charge priority |

**Formula:** `VSYSMIN = 3000 + (VSYSMIN_CODE × 100)` mV

**Behavior:**
- If battery voltage drops below VSYSMIN, charger stops charging and prioritizes system power supply
- At 3600mV: Ensures system doesn't brownout when battery depletes

---

### 7. Thermal Regulation (TREG)
**Purpose:** Temperature-based charge current limiting

| Code | Threshold | Use Case |
|------|-----------|----------|
| 00 | 100°C | **Aggressive** - Warm climate, high performance |
| 01 | 110°C | Balanced (recommended) |
| 10 | 120°C | Conservative - Cool climate |
| 11 | 130°C | Very conservative - Cold climate |

**Recommended:** 0b00 (100°C) for normal operation

---

### 8. Charging Safety Timer
**Purpose:** Prevents charging stuck condition (max charge time limit)

| Register | Field | Function |
|----------|-------|----------|
| 0x05 | [5] | Timer enable (1=enabled) |
| 0x05 | [3:2] | Timeout selection |

**Timeout Options:**
- 5 hours  (fast charge timeout)
- 8 hours
- 12 hours (recommended)
- 20 hours

**Why Enable?**
- Prevents battery damage from firmware bugs
- Ensures stuck charger detection
- Standard safety feature

---

### 9. Watchdog Timer
**Purpose:** Resets safety timers periodically (requires software refresh)

| Code | Timeout | Typical Use |
|------|---------|-------------|
| 00 | Disabled | Only if firmware monitors timer |
| 01 | 40 seconds | **Recommended** - Standard polling |
| 10 | 80 seconds | Slower monitoring |
| 11 | 160 seconds | Very slow monitoring |

**Action if watchdog expires:**
- Charging safety timer is reset
- Must be refreshed by firmware regularly

---

## Optimal Configuration for 10000mAh

```python
charger.configure_for_10ah_battery()
```

This applies:

```
Input Current Limit:     2000 mA  (safe USB power)
Charging Current:        2048 mA  (~20% C-rate)
Pre-charge Current:      192 mA   (0.2% C-rate)
Termination Current:     256 mA   (2.5% of capacity)
Charge Voltage Limit:    4208 mV  (standard Li-ion)
Minimum System Voltage:  3600 mV  (system power priority)
Thermal Regulation:      100°C    (aggressive cooling)
Safety Timer:            12 hours (maximum charge)
Watchdog:                40s      (software refresh)
Boost Mode:              Disabled (charger mode)
Charging:                Enabled
```

---

## Register Map

| Address | Name | Purpose |
|---------|------|---------|
| 0x00 | Input Source Control | Input current limit, battery disconnect |
| 0x01 | Power-On Configuration | Charging enable, system voltage, boost |
| 0x02 | Charge Current Control | Fast charge current |
| 0x03 | Pre-charge/Termination | Pre-charge & termination currents |
| 0x04 | Charge Voltage Control | Battery voltage regulation |
| 0x05 | Charging Termination/Timer | End-of-charge config, safety timer |
| 0x06 | Boost Voltage/Thermal | Thermal regulation, boost voltage |
| 0x08 | Status | Charger state, power good flag |
| 0x09 | Fault | Fault conditions |
| 0x0B | Device Version | IC revision |

---

## Status Register (0x08) Interpretation

```
Bit 7-6: Reserved
Bit 5-4: CHRG_STAT (Charging Status)
         00 = Not Charging
         01 = Pre-charge
         10 = Fast Charge
         11 = Charge Done
Bit 3:   Reserved
Bit 2:   PG_STAT (Power Good Flag)
         0 = Not ready, input < min limit
         1 = Input ready for charging
Bit 1-0: Reserved
```

---

## Fault Register (0x09) Interpretation

```
Bit 7:   LBATLV (Low Battery Alert)
Bit 6:   BATOVP (Battery Overvoltage Protected)
Bit 5:   INPUTOVP (Input Overvoltage)
Bit 4-3: CHRG_FAULT
         00 = Normal
         01 = Input Fault
         10 = Thermal Shutdown
         11 = Charge Timer Fault
Bit 2:   BAT_FAULT (Battery Fault)
Bit 1-0: NTC_FAULT (NTC Thermistor Fault)
         00 = Normal
         01 = TS Cold
         10 = TS Cool (or buck boost thermal)
         11 = TS Warm
```

---

## Python Usage

### Quick Configuration
```python
from pathpal_project.bq25895_charger import BQ25895

charger = BQ25895()
charger.configure_for_10ah_battery()
charger.close()
```

### Custom Configuration
```python
charger = BQ25895()

# Set individual parameters
charger.configure_input_current(2000)      # 2A from USB
charger.configure_charging_current(2048)   # 2A charging
charger.configure_charging_voltage(4208)   # 4.208V max
charger.configure_termination_current(256) # 256mA end-of-charge

# Enable charging
charger.enable_charging()

# Read status
status = charger.get_status()
print(status['charging_state'])

charger.close()
```

### Monitoring
```python
charger = BQ25895()

while True:
    status = charger.get_status()
    
    # Check for faults
    if status['thermal_shutdown']:
        print("WARNING: Thermal shutdown active!")
    if status['battery_overvoltage']:
        print("ERROR: Battery overvoltage!")
    
    print(f"State: {status['charging_state']}")
    time.sleep(1)

charger.close()
```

---

## Testing

Run the comprehensive test suite:
```bash
sudo python3 tests/test_bq25895_charger.py
```

**Tests include:**
1. Device detection
2. Register read/write
3. Input current configuration
4. Charging current configuration
5. Voltage configuration
6. Termination configuration
7. Full 10Ah battery configuration
8. Status monitoring
9. Thermal regulation
10. Safety timer

---

## Safety Guidelines

✅ **DO:**
- Monitor charger status periodically
- Keep battery voltage ≤ 4.25V for longevity
- Use thermal regulation (100°C threshold)
- Enable watchdog timer for safety
- Respect termination current settings

❌ **DON'T:**
- Set VREG above 4.35V (battery damage)
- Disable all safety timers
- Ignore fault conditions
- Exceed 5C charging rate (for safety)
- Ignore thermal shutdown alerts

---

## Integration with main.py

The BQ25895 configuration should be called during system initialization:

```python
from pathpal_project.bq25895_charger import BQ25895

class SmartEyeApp:
    def __init__(self, args):
        # ... other initialization ...
        
        # Configure battery charger
        self.charger = BQ25895()
        self.charger.configure_for_10ah_battery()
        
    def monitor_charging(self):
        """Periodically check charger status."""
        status = self.charger.get_status()
        
        if status['thermal_shutdown']:
            log.warning("Charger thermal shutdown - cooling needed")
        if status['battery_fault']:
            log.error("Battery fault detected!")
```

---

## Troubleshooting

### Charger not detected
```
Error: Failed to open I2C bus
→ Check UART6-M2 overlay is installed
→ Verify I2C3 is enabled
→ Check BQ25895 U2 is soldered correctly
→ Test with: i2cdetect -y 3
```

### Charging stuck
```
Status shows "Fast Charge" but not progressing
→ Check termination current setting
→ Verify battery is not full (check voltage)
→ Confirm thermal regulation threshold
→ Enable safety timer if disabled
```

### Excessive heat
```
Charger getting hot during charging
→ Reduce charging current (ICHG)
→ Lower thermal regulation threshold (TREG)
→ Improve heatsinking/ventilation
→ Check input voltage (lower = more heat)
```

### Battery not charging
```
Status shows "Not Charging" or "Pre-charge"
→ Check power is applied (Power Good flag)
→ Verify input current limit (ILIM)
→ Check battery voltage isn't too high
→ Confirm charging is enabled (CHE bit)
```

---

## References

- **Datasheet:** [BQ25895 TI Datasheet](https://www.ti.com/product/BQ25895)
- **Battery Safety:** IEC 62619 (Li-ion battery safety standard)
- **Charging Protocols:** IEC 61960-1 (Li-ion chemistry standards)

---

**Last Updated:** 2026-06-04
**Configuration Version:** 1.0
**Status:** Production Ready
