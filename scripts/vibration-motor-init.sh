#!/bin/bash
#
# Vibration motor init: hand GPIO0_C5 from GPIO sysfs to the PWM4 driver.
#
# This service runs early in systemd (Before=sysinit.target) and does two things:
#   1. Force the pin LOW via GPIO sysfs in case the initramfs hook left it exported.
#   2. Export the PWM4 chip so the main app's VibrationMotor class can find it.
#
# The boot vibration PATTERN is played by the initramfs hook
# (scripts/initramfs-vibration-boot → /etc/initramfs-tools/scripts/init-bottom/)
# which fires before systemd starts.  This service only handles the PWM handoff.
#
# GPIO0_C5 = bank 0, group C, pin 5 = global GPIO 21 (0*32 + 2*8 + 5)
# PWM4-M0  = febd0000
#
# Install:
#   sudo cp scripts/vibration-motor-init.sh /usr/local/bin/
#   sudo chmod +x /usr/local/bin/vibration-motor-init.sh
#   sudo cp services/vibration-motor-init.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable vibration-motor-init.service

GPIO=21
PWM_ADDR="febd0000"
GPIO_PATH="/sys/class/gpio/gpio${GPIO}"

# ---------------------------------------------------------------------------
# Phase 1: Ensure pin is LOW (idempotent — initramfs may have already done this)
# ---------------------------------------------------------------------------
if [ ! -d "$GPIO_PATH" ]; then
    echo $GPIO > /sys/class/gpio/export 2>/dev/null
    sleep 0.01
fi
echo out > "$GPIO_PATH/direction" 2>/dev/null || true
echo 0   > "$GPIO_PATH/value"    2>/dev/null || true

# Unexport so PWM pinctrl can claim the pin
echo $GPIO > /sys/class/gpio/unexport 2>/dev/null || true
sleep 0.05

# ---------------------------------------------------------------------------
# Phase 2: Export PWM4 chip so VibrationMotor in main.py can find it
# ---------------------------------------------------------------------------
find_pwmchip() {
    for chip in /sys/class/pwm/pwmchip*; do
        if readlink -f "$chip/device" 2>/dev/null | grep -q "$PWM_ADDR"; then
            basename "$chip"
            return 0
        fi
    done
    return 1
}

CHIP=$(find_pwmchip)
if [ -z "$CHIP" ]; then
    echo "vibration-motor-init: PWM4 ($PWM_ADDR) not found — GPIO fallback will be used by main app" >&2
    # Re-export as GPIO so VibrationMotor's GPIO fallback can use it
    echo $GPIO > /sys/class/gpio/export 2>/dev/null
    echo out > "$GPIO_PATH/direction"
    echo 0   > "$GPIO_PATH/value"
    exit 0
fi

PWMPATH="/sys/class/pwm/$CHIP"
if [ ! -d "$PWMPATH/pwm0" ]; then
    echo 0 > "$PWMPATH/export" 2>/dev/null
    sleep 0.1
fi

echo 2000000 > "$PWMPATH/pwm0/period"     # 500 Hz default period
echo 0       > "$PWMPATH/pwm0/duty_cycle"
echo 1       > "$PWMPATH/pwm0/enable"
echo "vibration-motor-init: $CHIP/pwm0 ready, duty=0 (motor OFF, PWM handed to app)"
