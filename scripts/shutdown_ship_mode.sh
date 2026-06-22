#!/bin/bash
# Shutdown hook: vibration motor feedback + BQ25895 ship mode.
# Runs as part of ship-mode-shutdown.service before poweroff.target.
#
# Ship mode decision is based on PG_STAT (REG0B[2]) at shutdown time:
#   - VBUS absent  (PG=0): Enter ship mode → BATFET opens after 10s → ~2µA drain.
#   - VBUS present (PG=1): Skip ship mode → BQ25895 keeps charging the battery.
#                           SoC stays in power-off with SYS fed from VBUS; battery
#                           charges normally. On charger unplug, SYS collapses and
#                           the device remains off (no auto-boot).

LOG="/var/log/battery-mgr.log"

# ---------------------------------------------------------------
#  Vibration motor buzz pattern (3 short pulses = "shutting down")
#  PWM4-M0 = GPIO0_C5 (global #21) via pwmchip4
# ---------------------------------------------------------------
vibrate() {
    local CHIP
    for c in /sys/class/pwm/pwmchip*; do
        # Find the chip whose sysfs path contains pwm@
        if readlink -f "$c" 2>/dev/null | grep -q "febe0000\|pwm4"; then
            CHIP="$c"; break
        fi
    done
    # Fallback: try pwmchip0
    CHIP="${CHIP:-/sys/class/pwm/pwmchip0}"

    local CH=0
    local PWM="$CHIP/pwm${CH}"

    if [ ! -d "$PWM" ]; then
        echo $CH > "$CHIP/export" 2>/dev/null
        sleep 0.1
    fi

    if [ -d "$PWM" ]; then
        echo 10000000 > "$PWM/period"
        echo  5000000 > "$PWM/duty_cycle"
        for i in 1 2 3; do
            echo 1 > "$PWM/enable"
            sleep 0.15
            echo 0 > "$PWM/enable"
            sleep 0.1
        done
        echo $CH > "$CHIP/unexport" 2>/dev/null
    else
        # GPIO fallback
        GPIO=21
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
#  Ship mode decision: check PG_STAT before acting
# ---------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] Shutdown hook running" >> "$LOG"

REG0B=$(i2cget -y 3 0x6a 0x0b b 2>/dev/null || echo "")
if [ -n "$REG0B" ]; then
    PG=$(( (REG0B >> 2) & 1 ))
    VBUS_TYPE=$(( (REG0B >> 5) & 0x07 ))
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] BQ25895 REG0B=${REG0B} PG=${PG} VBUS_TYPE=${VBUS_TYPE}" >> "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] WARNING: BQ25895 not found on i2c-3/0x6A — assuming no VBUS" >> "$LOG"
    PG=0
fi

if [ "${PG}" -eq 1 ]; then
    # Charger/VBUS is present — do NOT enter ship mode.
    # Rewrite all critical charging registers as the LAST I2C writes before the
    # kernel powers down its I2C controller. This guards against bus-glitch phantom
    # writes that can reset CHG_CONFIG to 0 during Rockchip poweroff sequence.
    # Note: the STAT LED will go off because VCC_3V3_S3 (the pull-up rail) loses
    # power during shutdown — this does NOT mean charging stopped. BQ25895 operates
    # autonomously from VBUS and continues charging the battery.
    i2cset -y 3 0x6a 0x00 0x66 b 2>/dev/null || true   # IINLIM=2000mA
    i2cset -y 3 0x6a 0x03 0x5a b 2>/dev/null || true   # CHG_CONFIG=1, SYS_MIN=3500mV
    i2cset -y 3 0x6a 0x04 0x20 b 2>/dev/null || true   # ICHG=2048mA
    i2cset -y 3 0x6a 0x06 0x5e b 2>/dev/null || true   # VREG=4208mV
    i2cset -y 3 0x6a 0x07 0x8f b 2>/dev/null || true   # WDT=OFF, CHG_TIMER=20h
    i2cset -y 3 0x6a 0x09 0x4c b 2>/dev/null || true   # BATFET_DIS=0 (BATFET ON)
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] Charger present (PG=1) — charging registers reaffirmed, skipping ship mode" >> "$LOG"
else
    # No VBUS — enter ship mode so battery drains at ~2µA instead of SoC standby.
    # REG09 = 0x6C:
    #   Bit 6 TMR2X_EN=1      harmless; keeps bit pattern clean
    #   Bit 5 BATFET_DIS=1    disconnect battery from SYS
    #   Bit 3 BATFET_DLY=1    10 s delay so OS finishes writing before disconnect
    #   Bit 2 BATFET_RST_EN=1 allow BATFET reset when charger is plugged in later
    SHIP_REG=0x6C
    if i2cset -y 3 0x6a 0x09 $SHIP_REG b 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] No charger — ship mode set (REG09=${SHIP_REG}, BATFET off in 10s)" >> "$LOG"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SHUTDOWN] ERROR: i2cset failed — ship mode NOT set" >> "$LOG"
    fi
fi

wait
