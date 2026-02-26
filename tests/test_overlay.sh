#!/bin/bash
#
# Verify that the Smart Eye carrier board overlay loaded correctly.
# Run after reboot with the overlay enabled.

PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
warn() { echo "  [WARN] $1"; ((WARN++)); }

echo "========================================"
echo " Smart Eye Overlay Verification"
echo "========================================"
echo ""

# --- 1. Overlay installed and enabled ---
echo "1. Overlay Installation"
if [ -f /boot/dtbo/rk3588-smart-eye-carrier.dtbo ]; then
    pass "Overlay DTBO is enabled in /boot/dtbo/"
elif [ -f /boot/dtbo/rk3588-smart-eye-carrier.dtbo.disabled ]; then
    fail "Overlay DTBO exists but is DISABLED (run: cd ~/Smart_eye_firmware/overlays && sudo make enable)"
else
    fail "Overlay DTBO not found (run: cd ~/Smart_eye_firmware/overlays && sudo make install && sudo make enable)"
fi
echo ""

# --- 2. Activity LED disabled ---
echo "2. Activity LED (GPIO4_B4)"
if [ -d /sys/class/leds/status-led-blue ]; then
    fail "status-led-blue LED still active (overlay not loaded or not disabling gpio-leds)"
else
    pass "status-led-blue LED disabled"
fi
echo ""

# --- 3. GPIO4_B3 hog (vibration motor off at boot) ---
echo "3. Vibration Motor GPIO Hog (GPIO4_B3)"
HOG_VAL=$(gpioget gpiochip4 11 2>/dev/null)
if [ "$HOG_VAL" = "0" ]; then
    pass "GPIO4_B3 is LOW (motor off) - gpio-hog working"
elif [ "$HOG_VAL" = "1" ]; then
    fail "GPIO4_B3 is HIGH (motor on) - gpio-hog not applied"
else
    warn "Could not read GPIO4_B3 (may be claimed by PWM driver already)"
fi

HOG_LINE=$(gpioinfo gpiochip4 2>/dev/null | grep "line  11:")
if echo "$HOG_LINE" | grep -q "vibration-motor-off\|pwm"; then
    pass "GPIO4_B3 line name: $HOG_LINE"
else
    warn "GPIO4_B3 line info: $HOG_LINE"
fi
echo ""

# --- 4. PWM15 ---
echo "4. PWM15 (Vibration Motor)"
PWM15_FOUND=0
for chip in /sys/class/pwm/pwmchip*/; do
    uevent=$(cat "${chip}device/uevent" 2>/dev/null)
    if echo "$uevent" | grep -q "febf0030"; then
        pass "PWM15 registered at $chip (0xfebf0030)"
        PWM15_FOUND=1
        echo "       npwm=$(cat ${chip}npwm 2>/dev/null)"
    fi
done
if [ $PWM15_FOUND -eq 0 ]; then
    fail "PWM15 (0xfebf0030) not found in /sys/class/pwm/"
fi
echo ""

# --- 5. I2C3 bus ---
echo "5. I2C3 Bus"
if [ -e /dev/i2c-3 ]; then
    pass "/dev/i2c-3 exists"
else
    fail "/dev/i2c-3 not found (I2C3 not enabled)"
fi

I2C_PINMUX=$(cat /sys/kernel/debug/pinctrl/pinctrl-rockchip-pinctrl/pinmux-pins 2>/dev/null | grep -i "i2c3")
if [ -n "$I2C_PINMUX" ]; then
    pass "I2C3 pinmux active"
else
    warn "Could not verify I2C3 pinmux (may need root)"
fi
echo ""

# --- 6. BQ27220 fuel gauge ---
echo "6. BQ27220 Fuel Gauge (I2C3 @ 0x55)"
if [ -e /dev/i2c-3 ]; then
    BQ_SCAN=$(i2cdetect -y 3 2>/dev/null | grep -o "55")
    if [ "$BQ_SCAN" = "55" ]; then
        pass "BQ27220 detected at address 0x55 on I2C3"
    else
        fail "BQ27220 NOT detected at 0x55 (check wiring, pull-ups)"
    fi
else
    fail "Cannot scan - /dev/i2c-3 missing"
fi

if ls /sys/class/power_supply/bq27* 2>/dev/null | grep -q .; then
    pass "BQ27220 kernel driver loaded (power_supply class)"
else
    warn "BQ27220 kernel driver not loaded (will use userspace I2C)"
fi
echo ""

# --- 7. BQ27220 interrupt (GPIO4_A4) ---
echo "7. Fuel Gauge Interrupt (GPIO4_A4)"
INT_VAL=$(gpioget gpiochip4 4 2>/dev/null)
if [ -n "$INT_VAL" ]; then
    pass "GPIO4_A4 readable, value=$INT_VAL (1=no interrupt, 0=interrupt active)"
else
    warn "Could not read GPIO4_A4 (may be claimed by kernel driver)"
fi
echo ""

# --- 8. UART6 ---
echo "8. UART6 (Serial Port)"
if [ -e /dev/ttyS6 ]; then
    pass "/dev/ttyS6 exists"
else
    fail "/dev/ttyS6 not found (UART6 not enabled)"
fi
echo ""

# --- 9. GPIO buttons ---
echo "9. GPIO Buttons"
for pin_info in "23:OCR_BTN:GPIO0_C7" "24:LANG_BTN:GPIO0_D0"; do
    pin=$(echo "$pin_info" | cut -d: -f1)
    name=$(echo "$pin_info" | cut -d: -f2)
    gpio=$(echo "$pin_info" | cut -d: -f3)
    val=$(gpioget gpiochip0 "$pin" 2>/dev/null)
    if [ -n "$val" ]; then
        pass "$name ($gpio) readable, value=$val (1=not pressed, 0=pressed)"
    else
        warn "$name ($gpio) could not read (may be claimed by gpio-keys driver)"
    fi
done
echo ""

# --- 10. Charger interrupt ---
echo "10. Charger Interrupt (CHG_INT0_L)"
CHG_VAL=$(gpioget gpiochip0 27 2>/dev/null)
if [ -n "$CHG_VAL" ]; then
    pass "CHG_INT0_L (GPIO0_D3) readable, value=$CHG_VAL"
else
    warn "CHG_INT0_L (GPIO0_D3) could not read (may be claimed by gpio-keys driver)"
fi
echo ""

# --- 11. Input devices (gpio-keys) ---
echo "11. Input Devices (gpio-keys)"
if cat /proc/bus/input/devices 2>/dev/null | grep -q "smart-eye"; then
    pass "gpio-keys input device(s) registered"
    grep -A3 "smart-eye" /proc/bus/input/devices 2>/dev/null | head -8
else
    warn "No 'smart-eye' input devices found in /proc/bus/input/devices"
fi
echo ""

# --- Summary ---
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "========================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
