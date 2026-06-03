# Smart Eye Carrier Board - Device Tree Overlay & Power Management

Custom device tree overlay and power management system for the Radxa CM5
on the Smart Eye carrier PCB. Configures PWM, I2C, UART, GPIO, audio,
and battery charging/monitoring peripherals at kernel level.

## Pin Map

| Signal              | RK3588 Pin | GPIO# | Function                         | Direction / Type              |
|---------------------|------------|-------|----------------------------------|-------------------------------|
| VIBRATION_PWM7_M1   | GPIO4_B3   | 139   | Vibration motor (hardware PWM)   | Output (PWM)                  |
| I2C3_SCL_M1_CHG     | GPIO3_B7   | 111   | BQ27220 / BQ25895 I2C clock      | I2C SCL                       |
| I2C3_SDA_M1_CHG     | GPIO3_C0   | 112   | BQ27220 / BQ25895 I2C data       | I2C SDA                       |
| BQ27220_GPOUT       | GPIO4_A4   | 132   | Fuel gauge interrupt/alert       | Input (pull-up, falling edge) |
| CHG_INT0_L          | GPIO0_D3   | 27    | BQ25895 charger interrupt        | Input (pull-up, falling edge) |
| UART6_TX_M2_S       | GPIO1_D1   | 57    | UART6 transmit                   | Output                        |
| UART6_RX_M2_S       | GPIO1_D0   | 56    | UART6 receive                    | Input                         |
| LANG_BTN            | GPIO0_D0   | 24    | Language select button            | Input (pull-up, active low)   |
| OCR_BTN             | GPIO0_C7   | 23    | OCR trigger button                | Input (pull-up, active low)   |
| AMP_SD (SD_MODE)    | GPIO4_A3   | 131   | MAX98357A speaker enable         | Output (active high)          |
| I2S1_SCLK_M0        | GPIO4_A1   | 129   | I2S1 bit clock                   | Output                        |
| I2S1_LRCK_M0        | GPIO4_A2   | 130   | I2S1 frame clock                 | Output                        |
| I2S1_SDO1_M0        | GPIO4_B2   | 138   | I2S1 data out (to MAX98357A)     | Output                        |

## Power Management ICs

### Hardware

| IC              | Part Number    | I2C Addr | Function                      |
|-----------------|----------------|----------|-------------------------------|
| U2              | BQ25895RTWR    | 0x6A     | 1S 5A buck charger            |
| U20             | BQ27220YZFR    | 0x55     | CEDV fuel gauge               |
| U15             | BQ29700DSER    | --       | Battery protection (HW only)  |
| R77             | 10 mΩ          | --       | Current sense resistor        |

### BQ25895 Charger Configuration

Applied by `power_config.py` and re-applied on every boot by the battery daemon:

| Parameter                | Value   | Register |
|--------------------------|---------|----------|
| Input Current Limit      | 2000 mA | REG00    |
| Fast Charge Current      | 2048 mA | REG04    |
| Charge Voltage           | 4208 mV | REG06    |
| Minimum System Voltage   | 3500 mV | REG03    |
| Watchdog Timer           | OFF     | REG07    |
| Safety Timer             | 20 h    | REG07    |
| BATFET Delay (ship mode) | 10 s    | REG09    |

### BQ27220 Fuel Gauge Configuration

CEDV Data Memory parameters written via MAC commands by `power_config.py`:

| Parameter         | Value     | DM Address |
|-------------------|-----------|------------|
| Design Capacity   | 10000 mAh | 0x929C     |
| Design Energy     | 3700 cWh  | 0x929E     |
| Terminate Voltage | 3000 mV   | 0x9280     |
| Taper Rate        | 217       | 0x9282     |

## Battery Management Daemon

A systemd service (`battery-mgr.service`) runs `/opt/battery-mgr/battery_daemon.py`
to continuously monitor battery health and enforce safety limits.

### Monitoring

- Polls BQ27220 every 30 seconds for SOC, voltage, current, temperature
- Logs detailed status to `/var/log/battery-mgr.log` every 5 minutes
- Re-applies BQ25895 register configuration on startup

### Protection Thresholds

| Condition                  | Action                                        |
|----------------------------|-----------------------------------------------|
| SOC ≤ 20%                  | Warning broadcast via `wall`                  |
| SOC ≤ 15%                  | Clean shutdown + ship mode                    |
| SOC ≤ 10%                  | Immediate shutdown + ship mode                |
| Cell voltage < 3.2 V       | Immediate shutdown + ship mode                |
| Temperature > 60°C         | Immediate shutdown + ship mode                |

### Ship Mode

Ship mode disconnects the battery from the system via BQ25895 BATFET to
prevent idle current drain when powered off. Re-enables automatically
when a USB charger is plugged in.

- Triggered by the daemon on low-battery shutdown
- Also triggered by `ship-mode-shutdown.service` during every poweroff
  (skipped if USB charger is connected)
- Manual entry: `sudo python3 /opt/battery-mgr/ship_mode.py --force`

### Installed Files

```
/opt/battery-mgr/
├── battery_daemon.py          # Main monitoring daemon
├── ship_mode.py               # Manual ship mode entry
└── shutdown_ship_mode.sh      # Shutdown hook (vibration + ship mode)

/etc/systemd/system/
├── battery-mgr.service        # Daemon unit (After=multi-user.target)
└── ship-mode-shutdown.service  # Shutdown hook (Before=poweroff.target)
```

## Power Button Behavior

The RK806 PMIC `pwrkey` generates `KEY_POWER` events on `/dev/input/event0`.

| Action           | Result   |
|------------------|----------|
| Short press      | Poweroff |
| Long press       | Poweroff |

Configuration files:

- `/etc/systemd/logind.conf.d/power-button.conf` -- sets `HandlePowerKey=poweroff`
  and `PowerKeyIgnoreInhibited=yes` to bypass GNOME desktop interception
- `/etc/dconf/db/gdm.d/10-power-button` -- sets GNOME `power-button-action=nothing`
- `/etc/dconf/db/local.d/10-power-button` -- same for all user sessions

## Shutdown Sequence

When poweroff is initiated (by power button, low battery, or `systemctl poweroff`):

1. Systemd begins shutdown, reaches `ship-mode-shutdown.service`
2. Vibration motor produces **3 short buzzes** (150 ms on, 100 ms off) as
   tactile feedback via PWM7 (`/sys/class/pwm/pwmchip0`)
3. Script checks BQ25895 for USB charger presence:
   - **Charger connected**: skips ship mode, system powers off normally
   - **No charger**: writes `0x6C` to BQ25895 REG09 to disconnect BATFET
     with 10-second delay, preventing battery drain while off
4. Systemd completes poweroff

## Audio: MAX98357A Speaker on I2S1

The overlay configures a MAX98357A I2S DAC amplifier on I2S1-M0:

| Signal     | RK3588 Pin | MAX98357A Pin |
|------------|------------|---------------|
| I2S1_SCLK  | GPIO4_A1   | BCLK          |
| I2S1_LRCK  | GPIO4_A2   | LRCLK         |
| I2S1_SDO1  | GPIO4_B2   | DIN           |
| SD_MODE    | GPIO4_A3   | SD_MODE       |

The `i2s-tx-route = <1 0 2 3>` setting swaps SDO0/SDO1 so audio data
goes out SDO1 (GPIO4_B2) to the MAX98357A DIN pin.

```bash
# Test speaker output
aplay -D plughw:CARD=SmartEyeAudio,DEV=0 /usr/share/sounds/alsa/Front_Center.wav

# Check sound card
aplay -l | grep Smart
```

## GPIO4_B3 Boot-High Fix (Vibration Motor)

RK3588 GPIO4_B3 floats high at power-on, briefly activating the vibration
motor before the PWM driver loads. Fix options:

**Option A (hardware, recommended):** Add a 47K–100K pull-down resistor
from the MOSFET gate to GND on the carrier PCB.

**Option B (U-Boot):** Run `overlays/setup_uboot_gpio_fix.sh` to configure
U-Boot `preboot` to drive GPIO 139 low before Linux loads.

## Quick Reference

```bash
# Full power system configuration (run once)
sudo python3 ~/Smart_eye_firmware/power_config.py

# Check power system status
sudo python3 ~/Smart_eye_firmware/power_config.py --status

# View battery daemon log
tail -f /var/log/battery-mgr.log

# Check daemon health
systemctl status battery-mgr.service

# Manual ship mode (battery disconnect)
sudo python3 /opt/battery-mgr/ship_mode.py --force

# Scan I2C bus 3
sudo i2cdetect -y 3

# Read BQ25895 charger status
sudo i2cget -y 3 0x6a 0x0b b

# Read BQ27220 SOC
sudo i2cget -y 3 0x55 0x1c w
```

## Overlay Structure

```
overlays/
├── smart-eye-carrier.dts              # Current overlay (PWM7 + I2C3 + I2S1 + BQ25895)
├── smart-eye-carrier.dtbo             # Compiled binary
├── rk3588-smart-eye-carrier.dts       # Legacy overlay (dual-codec variant)
├── rk3588-smart-eye-carrier.dtbo      # Legacy compiled binary
├── setup_uboot_gpio_fix.sh            # U-Boot GPIO fix installer
├── Makefile                           # Build/install/enable/disable
└── USAGE.md                           # This file
```

## Building & Installing the Overlay

```bash
cd ~/Smart_eye_firmware/overlays

# Compile
make

# Install to /boot/dtbo/ and enable
sudo make install
sudo make enable

# Reboot to apply
sudo reboot
```

## Troubleshooting

**Power button does nothing:**
```bash
# Check logind sees the key
journalctl -u systemd-logind | grep -i power

# Verify inhibitors are bypassed
systemd-inhibit --list
cat /etc/systemd/logind.conf.d/power-button.conf
```

**Battery daemon not running:**
```bash
systemctl status battery-mgr.service
journalctl -u battery-mgr.service -n 50
```

**Charger/fuel gauge not detected on I2C:**
```bash
sudo i2cdetect -y 3
# Expected: 0x55 (BQ27220) and 0x6a (BQ25895)
```

**Vibration motor runs at boot:**
```bash
# Check if U-Boot fix is applied
sudo fw_printenv preboot
# Should show: gpio clear 139
```

**Overlay not loading:**
```bash
ls -l /boot/dtbo/*smart-eye*
cat /boot/extlinux/extlinux.conf | grep overlay
sudo dmesg | grep -i "overlay\|dtbo\|error"
```

**PWM not appearing:**
```bash
ls /sys/class/pwm/
for chip in /sys/class/pwm/pwmchip*/; do
    echo "$chip: npwm=$(cat ${chip}npwm) device=$(readlink -f ${chip}device)"
done
```

**Audio not working:**
```bash
aplay -l
sudo dmesg | grep -i "i2s\|max98357\|sound\|audio"
cat /proc/asound/cards
```
