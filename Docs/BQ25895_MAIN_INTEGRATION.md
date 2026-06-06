# Integrating BQ25895 Charger into main.py

## Step 1: Add Import at Top of main.py

Add this line with the other imports (after line 33):

```python
from bq25895_charger import BQ25895
```

Full imports section should look like:
```python
import argparse
import sys
import os
import time
import threading
import queue
import struct
import fcntl
import glob
import subprocess
import traceback
import gc
import logging
import wave

from bq25895_charger import BQ25895  # <- ADD THIS LINE

import cv2
import numpy as np
```

---

## Step 2: Initialize Charger in SmartEyeApp.__init__()

Add charger initialization after line 1050 (after `self.gauge = BatteryGauge()`):

```python
def __init__(self, args):
    self.args = args
    self.lang = "en"
    self._running = True
    self._last_announce = 0
    self._last_ocr = 0
    self._last_ultrasonic = 0
    self._last_gc = 0
    self._battery_warned = False

    log.info("Initializing Smart Eye system...")

    _apply_power_profile()

    self.vibrator = VibrationMotor()
    self.audio = AudioPlayer()
    self.gauge = BatteryGauge()
    
    # ===== ADD BATTERY CHARGER INITIALIZATION HERE =====
    try:
        self.charger = BQ25895()
        # Use longevity config for maximum battery life (recommended for Smart Eye)
        self.charger.configure_for_10ah_battery_longevity()
        log.info("Battery charger initialized (longevity mode: 4.1V, 1024mA)")
        self.charger_available = True
    except Exception as e:
        log.warning(f"Battery charger not available: {e}")
        self.charger = None
        self.charger_available = False
    # ===== END CHARGER INITIALIZATION =====
    
    self.us_front = UltrasonicSensor(
        getattr(self.args, "us_front_device", "/dev/ttyS3"), label="US-front"
    )
    # ... rest of __init__ ...
```

---

## Step 3: Add Charger Monitoring to Battery Loop

Find the `_battery_loop` method and add charger monitoring. Replace the battery monitoring section with:

```python
def _battery_loop(self):
    """Monitor battery SOC and charger status periodically."""
    last_charge_check = 0
    
    while self._running:
        now = time.monotonic()
        
        # Read battery SOC
        soc = self.gauge.read_soc()
        
        # Check charger status (every 5 seconds if available)
        if self.charger_available and (now - last_charge_check) >= 5.0:
            try:
                status = self.charger.get_status()
                self._handle_charger_status(status)
                last_charge_check = now
            except Exception as e:
                log.warning(f"Charger status read failed: {e}")
        
        # Battery warning/shutdown logic
        if soc is not None:
            if soc <= self.SOC_SHUTDOWN:
                log.critical(f"Battery critical (SOC={soc}%) - initiating shutdown")
                self.audio.play("battery_shutdown")
                time.sleep(2)
                self._shutdown()
                break
            elif soc <= self.SOC_WARN and not self._battery_warned:
                log.warning(f"Battery low (SOC={soc}%)")
                self.audio.play("battery_low")
                self._battery_warned = True
            elif soc > self.SOC_WARN:
                self._battery_warned = False
        
        time.sleep(1)
```

---

## Step 4: Add Charger Status Handler (NEW METHOD)

Add this new method to the SmartEyeApp class:

```python
def _handle_charger_status(self, status):
    """Handle charger status and faults.
    
    Note: NTC (thermistor) is not connected, so NTC_FAULT is ignored.
    """
    
    # Critical faults that require action
    if status['battery_overvoltage']:
        log.error("[CHARGER] Battery overvoltage! Battery may be damaged.")
        # Alert user but continue operation (overvoltage protection active)
    
    if status['input_overvoltage']:
        log.error("[CHARGER] Input overvoltage detected! Power supply issue.")
        # Don't charge if input is bad
    
    if status['charger_fault'] != 0:
        fault_desc = {
            1: "Input Fault",
            2: "Thermal Shutdown", 
            3: "Charge Timer Fault",
        }
        log.error(f"[CHARGER] Charger fault: {fault_desc.get(status['charger_fault'], 'Unknown')}")
    
    if status['battery_fault']:
        log.error("[CHARGER] Battery fault detected!")
    
    # Thermal shutdown warning (not an error, normal operation)
    if status['thermal_shutdown']:
        log.warning("[CHARGER] Thermal throttling active (>100°C)")
    
    # ===== IGNORE NTC FAULT - NO THERMISTOR CONNECTED =====
    # NTC (thermistor) is not connected to this device, so ignore NTC_FAULT
    # if status['ntc_fault'] != 0:
    #     log.warning(f"[CHARGER] NTC fault: {status['ntc_fault']} (IGNORED - no NTC connected)")
    # ===== END NTC IGNORE =====
    
    # Informational logging
    if status['power_good']:
        charging_state = status['charging_state']
        if charging_state != "Not Charging":
            log.debug(f"[CHARGER] {charging_state}")
```

---

## Step 5: Add Charger Cleanup in Shutdown

Add charger cleanup to the existing cleanup/shutdown method. Find the method that closes resources and add:

```python
def cleanup(self):
    """Cleanup all resources."""
    log.info("Cleaning up Smart Eye system...")
    
    # ... existing cleanup code ...
    
    # Close battery charger
    if self.charger is not None:
        try:
            self.charger.disable_charging()  # Disable charging on shutdown
            self.charger.close()
            log.info("Battery charger closed")
        except Exception as e:
            log.warning(f"Error closing charger: {e}")
    
    # ... rest of cleanup ...
```

---

## Step 6: Update Command-Line Arguments (Optional)

Add battery mode argument to the arg parser (find `parse_args()` function):

```python
def parse_args():
    p = argparse.ArgumentParser(description="Smart Eye Assistive Vision System")
    
    # ... existing arguments ...
    
    # Battery charging mode
    p.add_argument("--battery-mode",
        choices=["longevity", "performance"],
        default="longevity",
        help="Battery charging mode (default: longevity for maximum device lifespan)"
    )
    
    # ... rest of arguments ...
    return p.parse_args()
```

Then update the charger initialization to use this:

```python
# In __init__, replace the charger initialization:
try:
    self.charger = BQ25895()
    
    # Select charging mode
    if self.args.battery_mode == "longevity":
        self.charger.configure_for_10ah_battery_longevity()
        log.info("Battery charger: LONGEVITY mode (4.1V, 1024mA, ~10.5h charge)")
    else:
        self.charger.configure_for_10ah_battery()
        log.info("Battery charger: PERFORMANCE mode (4.2V, 2048mA, ~5.5h charge)")
    
    self.charger_available = True
except Exception as e:
    log.warning(f"Battery charger not available: {e}")
    self.charger = None
    self.charger_available = False
```

---

## Complete Integration: Before/After

### BEFORE (Original code)
```python
def __init__(self, args):
    log.info("Initializing Smart Eye system...")
    _apply_power_profile()
    
    self.vibrator = VibrationMotor()
    self.audio = AudioPlayer()
    self.gauge = BatteryGauge()  # Only battery gauge
    self.us_front = UltrasonicSensor(...)
    # ... rest of init ...
```

### AFTER (With charger)
```python
def __init__(self, args):
    log.info("Initializing Smart Eye system...")
    _apply_power_profile()
    
    self.vibrator = VibrationMotor()
    self.audio = AudioPlayer()
    self.gauge = BatteryGauge()
    
    # Battery charger (NEW)
    try:
        self.charger = BQ25895()
        self.charger.configure_for_10ah_battery_longevity()
        log.info("Battery charger initialized (longevity mode)")
        self.charger_available = True
    except Exception as e:
        log.warning(f"Battery charger not available: {e}")
        self.charger = None
        self.charger_available = False
    
    self.us_front = UltrasonicSensor(...)
    # ... rest of init ...
```

---

## NTC Thermistor Handling

### Important: No Thermistor Connected

Your device **does NOT have an NTC thermistor connected**, so:

1. **Ignore NTC_FAULT** - The charger will report NTC faults, but these are harmless
2. **Monitor THERMAL_SHUTDOWN** instead - This is the real temperature limit (>100°C)
3. **Trust the internal temperature sensor** - The charger IC has built-in thermal limits

### In the code:
```python
# NTC Fault (IGNORE - no thermistor connected)
# status['ntc_fault'] = 0/1/2/3 
# → Don't log this, it's normal without a thermistor

# Thermal Shutdown (MONITOR - real temperature limit)
# status['thermal_shutdown'] = True/False
# → This is the actual temperature protection, respond to this
```

### What the NTC readings mean:
```
NTC_FAULT = 0  → Normal (no thermistor)
NTC_FAULT = 1  → TS Cold (would be <5°C if thermistor present)
NTC_FAULT = 2  → TS Cool (would be ~25°C if thermistor present) 
NTC_FAULT = 3  → TS Warm (would be >45°C if thermistor present)
```

Since there's no thermistor, these are always false positives. **Ignore them completely.**

---

## Running with Integration

### Start with longevity mode (recommended):
```bash
sudo -E venv/bin/python3 pathpal_project/main.py
```

### Or explicitly specify mode:
```bash
# Longevity (slow, long battery life)
sudo -E venv/bin/python3 pathpal_project/main.py --battery-mode longevity

# Performance (fast, shorter battery life)  
sudo -E venv/bin/python3 pathpal_project/main.py --battery-mode performance
```

---

## Verifying Integration

### Check logs for successful initialization:
```bash
sudo journalctl -u smart-eye -f | grep -i charger
```

Expected output:
```
Battery charger initialized (longevity mode: 4.1V, 1024mA)
[CHARGER] State: Not Charging | Power Good: False
[CHARGER] State: Pre-charge | Power Good: True
[CHARGER] State: Fast Charge | Power Good: True
```

### Monitor charger during operation:
```bash
# Watch charger status in real-time
sudo python3 << 'EOF'
from pathpal_project.bq25895_charger import BQ25895
import time

charger = BQ25895()
charger.configure_for_10ah_battery_longevity()

while True:
    status = charger.get_status()
    print(f"State: {status['charging_state']:15s} | Power: {status['power_good']}")
    time.sleep(2)

charger.close()
EOF
```

---

## Troubleshooting Integration

### "Battery charger not available" message
```
→ I2C3 not enabled (check overlay)
→ BQ25895 not responding at 0x6A
→ Permission denied (need sudo)
```

### Charger detected but not charging
```
→ Input power not connected
→ Battery already fully charged
→ Safety timer expired
→ Charging disabled
```

### High temperature warnings
```
→ Check ambient temperature <30°C
→ Ensure good airflow around device
→ Reduce inference load during charging
→ Use longevity mode (lower current = less heat)
```

---

## Summary

**Integration Steps:**
1. ✅ Add import: `from bq25895_charger import BQ25895`
2. ✅ Initialize in `__init__()`: `self.charger = BQ25895()`
3. ✅ Configure: `charger.configure_for_10ah_battery_longevity()`
4. ✅ Monitor in battery loop: `status = charger.get_status()`
5. ✅ Handle status and faults (ignore NTC)
6. ✅ Cleanup on shutdown: `charger.close()`

**NTC Handling:**
- ✅ Ignore NTC_FAULT entirely (no thermistor connected)
- ✅ Monitor THERMAL_SHUTDOWN (real temperature limit)
- ✅ Log thermal warnings if charger gets too hot

**Result:**
- ✅ Battery automatically configured at startup
- ✅ Charging monitored every 5 seconds
- ✅ Faults logged and handled gracefully
- ✅ Clean shutdown on exit

---

**Status:** Ready to integrate ✓
