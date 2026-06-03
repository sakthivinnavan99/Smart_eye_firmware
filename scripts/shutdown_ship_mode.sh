#!/bin/bash
# Shutdown hook: vibration motor feedback + BQ25895 ship mode.
# Runs as part of ship-mode-shutdown.service before poweroff.target.

LOG="/var/log/battery-mgr.log"

# ---------------------------------------------------------------
#  Vibration motor buzz pattern (3 short pulses = "shutting down")
#  PWM7 = pwmchip0 @ febd0000, GPIO4_B3
# ---------------------------------------------------------------
vibrate() {
    local CHIP=/sys/class/pwm/pwmchip0
    local CH=0
    local PWM="$CHIP/pwm${CH}"

    # Export PWM channel if not already
    if [ ! -d "$PWM" ]; then
        echo $CH > "$CHIP/export" 2>/dev/null
        sleep 0.1
    fi

    if [ -d "$PWM" ]; then
        # period=10ms (100Hz), duty=50%
        echo 10000000 > "$PWM/period"
        echo  5000000 > "$PWM/duty_cycle"

        # 3 short buzzes: on-off-on-off-on-off
        for i in 1 2 3; do
            echo 1 > "$PWM/enable"
            sleep 0.15
            echo 0 > "$PWM/enable"
            sleep 0.1
        done

        # Cleanup
        echo $CH > "$CHIP/unexport" 2>/dev/null
    else
        # Fallback: toggle GPIO directly if PWM export fails
        GPIO=139  # GPIO4_B3 = 4*32+11 = 139
        echo $GPIO > /sys/class/gpio/export 2>/dev/null
        echo out > /sys/class/gpio/gpio${GPIO}/direction 2>/dev/null
        for i in 1 2 3; do
            echo 1 > /sys/class/gpio/gpio${GPIO}/value
            sleep 0.15
            echo 0 > /sys/class/gpio/gpio${GPIO}/value
            sleep 0.1
        done
        echo $GPIO > /sys/class/gpio/unexport 2>/dev/null
    fi
}

vibrate &

# ---------------------------------------------------------------
#  Ship mode: disconnect BATFET if no charger is connected
# ---------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] Shutdown hook running" >> "$LOG"

VBUS_STAT=$(i2cget -y 3 0x6a 0x0b b 2>/dev/null)
if [ -n "$VBUS_STAT" ]; then
    PG=$(( (VBUS_STAT >> 2) & 1 ))
    if [ "$PG" = "1" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] Charger connected, skipping ship mode" >> "$LOG"
        wait
        exit 0
    fi
fi

# BATFET_DIS=1, BATFET_DLY=10s, BATFET_RST=1 -> 0x6C
i2cset -y 3 0x6a 0x09 0x6c b 2>/dev/null
echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] Ship mode set (BATFET off in 10s)" >> "$LOG"

wait
