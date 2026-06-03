#!/bin/bash
#
# Vibration motor init: immediately drive GPIO4_B3 low, then hand off to PWM.
#
# GPIO4_B3 = bank 4, pin B3 = global GPIO 139 (128 + 8 + 3)
#
# Phase 1: Use GPIO sysfs to force pin low ASAP (kills the motor).
# Phase 2: Set up PWM15 sysfs so userspace can control vibration intensity.
#
# Install:
#   sudo cp vibration-motor-init.sh /usr/local/bin/
#   sudo chmod +x /usr/local/bin/vibration-motor-init.sh
#   sudo cp vibration-motor-init.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable vibration-motor-init.service

GPIO=139
PWM_ADDR="febd0030"

# --- Phase 1: GPIO force-low (immediate) ---
if [ ! -d /sys/class/gpio/gpio${GPIO} ]; then
    echo $GPIO > /sys/class/gpio/export 2>/dev/null
    sleep 0.01
fi
echo out > /sys/class/gpio/gpio${GPIO}/direction
echo 0   > /sys/class/gpio/gpio${GPIO}/value
echo "vibration-motor-init: GPIO $GPIO forced LOW (motor OFF)"

# --- Phase 2: Hand off to PWM driver ---
# Unexport GPIO so PWM pinctrl can claim the pin
echo $GPIO > /sys/class/gpio/unexport 2>/dev/null
sleep 0.05

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
    echo "vibration-motor-init: pwm15 ($PWM_ADDR) not found, staying in GPIO mode" >&2
    echo $GPIO > /sys/class/gpio/export 2>/dev/null
    echo out > /sys/class/gpio/gpio${GPIO}/direction
    echo 0   > /sys/class/gpio/gpio${GPIO}/value
    exit 0
fi

PWMPATH="/sys/class/pwm/$CHIP"
if [ ! -d "$PWMPATH/pwm0" ]; then
    echo 0 > "$PWMPATH/export" 2>/dev/null
    sleep 0.1
fi

echo 1000000 > "$PWMPATH/pwm0/period"
echo 0       > "$PWMPATH/pwm0/duty_cycle"
echo 1       > "$PWMPATH/pwm0/enable"
echo "vibration-motor-init: $CHIP/pwm0 active, duty=0 (motor OFF via PWM)"
