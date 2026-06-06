# Exact Code Changes for main.py Integration

## Copy-Paste Ready Code Snippets

### Change 1: Add Import (Line ~35, after imports)

**ADD THIS:**
```python
from bq25895_charger import BQ25895  # Battery charger configuration
```

**Location:** After `import wave` and before `import cv2`

---

### Change 2: Initialize Charger in __init__ (After line 1050)

**FIND THIS (lines 1048-1050):**
```python
        self.vibrator = VibrationMotor()
        self.audio = AudioPlayer()
        self.gauge = BatteryGauge()
```

**REPLACE WITH THIS:**
```python
        self.vibrator = VibrationMotor()
        self.audio = AudioPlayer()
        self.gauge = BatteryGauge()

        # Battery charger initialization (NTC thermistor not connected)
        try:
            self.charger = BQ25895()
            # Use longevity config for maximum battery life (recommended)
            self.charger.configure_for_10ah_battery_longevity()
            log.info("✓ Battery charger initialized: longevity mode (4.1V, 1024mA, ~10.5h charge)")
            self.charger_available = True
        except Exception as e:
            log.warning(f"✗ Battery charger not available: {e}")
            self.charger = None
            self.charger_available = False
```

---

### Change 3: Update Battery Loop (Replace _battery_loop method)

**FIND THIS METHOD:** `def _battery_loop(self):`

**REPLACE THE ENTIRE METHOD WITH THIS:**
```python
    def _battery_loop(self):
        """Monitor battery SOC and charger status periodically."""
        last_charger_check = 0
        
        while self._running:
            now = time.monotonic()
            
            # Read battery SOC
            soc = self.gauge.read_soc()
            
            # Check charger status periodically (every 5 seconds if available)
            if self.charger_available and (now - last_charger_check) >= 5.0:
                try:
                    # Get status with NTC faults ignored (no thermistor connected)
                    status = self.charger.get_status(ignore_ntc=True)
                    self._handle_charger_status(status)
                    last_charger_check = now
                except Exception as e:
                    log.debug(f"Charger status read failed: {e}")
            
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

### Change 4: Add Charger Status Handler (NEW METHOD - Add to class)

**ADD THIS AS A NEW METHOD to SmartEyeApp class:**
```python
    def _handle_charger_status(self, status):
        """Handle charger status and log important events.
        
        Note: NTC (thermistor) is not connected, so NTC_FAULT is always ignored.
        Monitor THERMAL_SHUTDOWN instead for temperature issues.
        """
        # Critical faults - log as errors
        if status['battery_overvoltage']:
            log.error("⚠ CRITICAL: Battery overvoltage protection active")
        
        if status['input_overvoltage']:
            log.error("⚠ CRITICAL: Input overvoltage detected - power supply issue")
        
        # Charger fault codes
        if status['charger_fault'] != 0:
            fault_names = {1: "Input Fault", 2: "Thermal Shutdown", 3: "Charge Timer Fault"}
            log.error(f"⚠ Charger fault: {fault_names.get(status['charger_fault'], 'Unknown')}")
        
        if status['battery_fault']:
            log.error("⚠ CRITICAL: Battery fault detected")
        
        # Thermal warning (not critical, just monitor)
        if status['thermal_shutdown']:
            log.warning("⚠ Charger thermal throttling (>100°C) - reduce load")
        
        # Normal operation logging (only debug level)
        if status['power_good'] and status['charging_state'] != "Not Charging":
            log.debug(f"Charging: {status['charging_state']}")
        
        # NTC is ignored (no thermistor connected) - don't log it
```

---

### Change 5: Add Cleanup (Find cleanup method and add charger close)

**FIND THE METHOD:** `def cleanup(self):` or `def __del__(self):` or similar shutdown

**ADD THIS INSIDE THE CLEANUP METHOD:**
```python
        # Close battery charger
        if self.charger is not None:
            try:
                self.charger.disable_charging()
                self.charger.close()
                log.info("Battery charger closed")
            except Exception as e:
                log.warning(f"Error closing charger: {e}")
```

---

## Quick Summary of Changes

1. **Line ~35:** Add `from bq25895_charger import BQ25895`
2. **Line ~1050:** Initialize charger in `__init__` (14 lines)
3. **Method:** Replace `_battery_loop` with updated version (35 lines)
4. **Method:** Add new `_handle_charger_status` method (24 lines)
5. **Method:** Add charger cleanup in shutdown (7 lines)

**Total additions:** ~80 lines of code

---

## Testing the Integration

### 1. Check syntax
```bash
python3 -m py_compile pathpal_project/main.py
```

### 2. Check logs during startup
```bash
sudo -E venv/bin/python3 pathpal_project/main.py 2>&1 | head -20
```

**Expected to see:**
```
[INFO] Initializing Smart Eye system...
[INFO] ✓ Battery charger initialized: longevity mode (4.1V, 1024mA, ~10.5h charge)
[DEBUG] Charging: Pre-charge
[DEBUG] Charging: Fast Charge
```

### 3. Monitor charger during operation
```bash
sudo journalctl -u smart-eye -f | grep -i charger
```

### 4. Run test suite
```bash
sudo python3 tests/test_bq25895_charger.py
```

---

## Important Notes

### NTC Thermistor Status
- ✅ **NOT connected** to your device
- ✅ `ignore_ntc=True` is the default
- ✅ NTC_FAULT will always be 0
- ✅ Safe to ignore in logs

### Charging Modes Available

**Longevity (Recommended - current setting):**
```python
charger.configure_for_10ah_battery_longevity()
# 4.1V, 1024mA, ~10.5h charge, >1000 cycles
```

**Performance (if fast charging needed):**
```python
charger.configure_for_10ah_battery()
# 4.2V, 2048mA, ~5.5h charge, ~500 cycles
```

---

## Troubleshooting

### "Cannot import BQ25895"
```
→ Make sure bq25895_charger.py is in pathpal_project/ directory
→ Check Python path is correct (from bq25895_charger import...)
```

### "Battery charger not available"
```
→ Run with sudo (needs I2C access)
→ Check I2C3 is enabled (overlay installed)
→ Device not responding at 0x6A
```

### "Charger monitoring logs spam"
```
→ Change log.info() to log.debug() in _handle_charger_status()
→ Or reduce check interval from 5.0s to 30.0s
```

---

## Verification Checklist

- [ ] Import added (line ~35)
- [ ] Charger init added to __init__ (line ~1050)
- [ ] _battery_loop method replaced
- [ ] _handle_charger_status method added
- [ ] Cleanup code added to shutdown
- [ ] Syntax check passes: `python3 -m py_compile pathpal_project/main.py`
- [ ] Startup shows charger initialized message
- [ ] No NTC fault warnings in logs
- [ ] Charger status logged every 5 seconds
- [ ] Tests pass: `sudo python3 tests/test_bq25895_charger.py`

---

## Final Deployment

Once integrated and tested:

```bash
# Run normally
sudo -E venv/bin/python3 pathpal_project/main.py

# Or with systemd service
sudo systemctl start smart-eye
sudo journalctl -u smart-eye -f
```

---

**Status:** Ready to patch ✓
