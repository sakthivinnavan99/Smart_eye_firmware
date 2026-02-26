# Smart Eye Carrier Board - Device Tree Overlay

Custom device tree overlay for the Radxa CM5 on the Smart Eye carrier PCB.
Configures PWM, I2C, UART, and GPIO peripherals at kernel level.

## Pin Map

| Signal Name       | RK3588 Pin | Linux GPIO# | Function                        | Direction  |
|-------------------|------------|-------------|---------------------------------|------------|
| VIBRATION_PWM15_M1| GPIO4_B3   | 139         | Vibration motor (hardware PWM)  | Output     |
| I2C3_SCL_M1_CHG   | GPIO3_B7   | 111         | BQ27220 fuel gauge clock        | I2C SCL    |
| I2C3_SDA_M1_CHG   | GPIO3_C0   | 112         | BQ27220 fuel gauge data         | I2C SDA    |
| GPIO4_A4          | GPIO4_A4   | 132         | BQ27220 interrupt/alert         | Input (pull-up, falling edge) |
| UART6_TX_M2_S     | GPIO1_D1   | 57          | UART6 transmit                  | Output     |
| UART6_RX_M2_S     | GPIO1_D0   | 56          | UART6 receive                   | Input      |
| CHG_INT0_L        | GPIO0_D3   | 27          | Charger interrupt               | Input (pull-up, active low) |
| LANG_BTN          | GPIO0_D0   | 24          | Language select button           | Input (pull-up, active low) |
| OCR_BTN           | GPIO0_C7   | 23          | OCR trigger button               | Input (pull-up, active low) |

## GPIO4_B3 Boot-High Fix (Vibration Motor)

**Problem:** GPIO4_B3 (`PI_nLED_Activity` on the CM5 connector) defaults
to a high state at power-on due to the RK3588 boot ROM / U-Boot
initialization. On the custom carrier board this pin drives the vibration
motor, causing it to run continuously during and after boot until the PWM
driver takes over.

**Solution:** The overlay applies two fixes:

1. **gpio-hog** (fragment@1) -- Adds a `gpio-hog` child node to the `gpio4`
   controller that forces GPIO4_B3 **output-low** as soon as the GPIO
   controller driver initializes (very early in the kernel boot sequence,
   before most other drivers). This keeps the motor off during boot.

2. **gpio-leds disabled** (fragment@0) -- Disables the CM5-IO `gpio-leds`
   node (heartbeat LED on adjacent GPIO4_B4) to prevent the LED driver
   from interfering with GPIO bank 4 on the custom carrier.

After applying the overlay:
- GPIO4_B3 is driven low immediately when the GPIO4 controller probes
- The vibration motor stays off until your application enables PWM
- `/sys/class/leds/status-led-blue` will no longer exist
- PWM15-M1 takes over the pin mux when the PWM driver loads later in boot

## Prerequisites

```bash
sudo apt install device-tree-compiler
```

## Build and Install

```bash
cd ~/Smart_eye_firmware/overlays

# Compile the overlay
make

# Install to system overlay directories
sudo make install

# Enable for next boot
sudo make enable

# Reboot to apply
sudo reboot
```

## Makefile Targets

| Target    | Description                                     |
|-----------|-------------------------------------------------|
| `make`    | Compile `.dts` to `.dtbo`                       |
| `make install` | Copy `.dtbo` to system overlay directories |
| `make enable`  | Enable overlay for next boot                |
| `make disable` | Disable overlay (reboot required)           |
| `make status`  | Show current overlay state and pin listing  |
| `make clean`   | Remove compiled files                       |

## Verification After Reboot

### Check overlay was loaded

```bash
sudo dmesg | grep -i "overlay\|pwm15\|i2c3\|uart6\|bq27\|gpio-keys"
```

### PWM15 (vibration motor)

After the overlay loads, PWM15 appears under sysfs. The channel index depends
on how many PWM controllers are already enabled.

```bash
# Find the pwmchip that corresponds to PWM15
# PWM15 is at register 0xfebf0030
ls /sys/class/pwm/

# Export the channel and test
echo 0 | sudo tee /sys/class/pwm/pwmchipN/export
echo 2000000 | sudo tee /sys/class/pwm/pwmchipN/pwm0/period      # 500 Hz
echo 1000000 | sudo tee /sys/class/pwm/pwmchipN/pwm0/duty_cycle   # 50%
echo 1 | sudo tee /sys/class/pwm/pwmchipN/pwm0/enable
```

Python usage (after identifying the correct pwmchip):

```python
import os

PWM_CHIP = "/sys/class/pwm/pwmchipN"  # replace N with actual chip number
PWM_CH   = PWM_CHIP + "/pwm0"

def pwm_init(freq_hz=500):
    if not os.path.exists(PWM_CH):
        with open(PWM_CHIP + "/export", "w") as f:
            f.write("0")
    period_ns = int(1e9 / freq_hz)
    with open(PWM_CH + "/period", "w") as f:
        f.write(str(period_ns))

def pwm_set_duty(percent):
    """Set duty cycle 0-100."""
    period = int(open(PWM_CH + "/period").read())
    duty = int(period * percent / 100)
    with open(PWM_CH + "/duty_cycle", "w") as f:
        f.write(str(duty))

def pwm_enable(on=True):
    with open(PWM_CH + "/enable", "w") as f:
        f.write("1" if on else "0")
```

### I2C3 (BQ27220 fuel gauge)

```bash
# Scan I2C3 bus - BQ27220 should appear at address 0x55
sudo i2cdetect -y 3

# Read voltage register (0x04) as a quick test
sudo i2cget -y 3 0x55 0x04 w
```

Python usage:

```python
import smbus2

bus = smbus2.SMBus(3)
BQ27220_ADDR = 0x55

def read_voltage():
    """Read battery voltage in mV."""
    raw = bus.read_word_data(BQ27220_ADDR, 0x04)
    return raw  # value in mV

def read_soc():
    """Read state of charge in percent."""
    raw = bus.read_word_data(BQ27220_ADDR, 0x1C)
    return raw  # value in %

def read_current():
    """Read average current in mA (signed)."""
    import struct
    raw = bus.read_word_data(BQ27220_ADDR, 0x10)
    return struct.unpack('h', struct.pack('H', raw))[0]  # signed

def read_temperature():
    """Read temperature in 0.1 K, convert to Celsius."""
    raw = bus.read_word_data(BQ27220_ADDR, 0x06)
    return (raw * 0.1) - 273.15
```

### UART6

```bash
# UART6 appears as /dev/ttyS6
ls -l /dev/ttyS6

# Quick loopback test (connect TX to RX)
stty -F /dev/ttyS6 9600 cs8 -cstopb -parenb
echo "hello" > /dev/ttyS6
cat /dev/ttyS6
```

Python usage:

```python
import serial

uart6 = serial.Serial(
    port="/dev/ttyS6",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    stopbits=serial.STOPBITS_ONE,
    parity=serial.PARITY_NONE,
    timeout=1
)

uart6.write(b"hello\n")
response = uart6.readline()
```

### GPIO Buttons and Charger Interrupt

The overlay registers LANG_BTN, OCR_BTN, and CHG_INT0_L as `gpio-keys`
input devices. They appear as `/dev/input/eventX` devices automatically.

```bash
# Find the input devices
cat /proc/bus/input/devices | grep -A4 "smart-eye"

# Test with evtest (install: sudo apt install evtest)
sudo evtest /dev/input/eventX
# Press buttons to see events
```

Python usage with evdev:

```python
# sudo apt install python3-evdev
# pip install evdev
import evdev
import select

def find_smart_eye_inputs():
    """Find input devices created by the overlay."""
    devices = {}
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if "smart-eye" in dev.name.lower() or dev.name in ("LANG_BTN", "OCR_BTN", "CHG_INT0_L"):
            devices[dev.name] = dev
    return devices

def poll_buttons():
    """Poll for button press events."""
    devices = find_smart_eye_inputs()
    if not devices:
        print("No Smart Eye input devices found")
        return

    print(f"Monitoring: {list(devices.keys())}")
    devs = list(devices.values())

    while True:
        r, _, _ = select.select(devs, [], [])
        for dev in r:
            for event in dev.read():
                if event.type == evdev.ecodes.EV_KEY:
                    key = evdev.ecodes.KEY.get(event.code) or evdev.ecodes.BTN.get(event.code) or event.code
                    state = "pressed" if event.value == 1 else "released" if event.value == 0 else "held"
                    print(f"{dev.name}: {key} {state}")
```

Alternative: direct GPIO sysfs access (without gpio-keys driver):

```python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

LANG_BTN_PIN = 24    # GPIO0_D0
OCR_BTN_PIN  = 23    # GPIO0_C7
CHG_INT_PIN  = 27    # GPIO0_D3

GPIO.setup(LANG_BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(OCR_BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(CHG_INT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def on_lang_btn(channel):
    print("Language button pressed")

def on_ocr_btn(channel):
    print("OCR button pressed")

def on_chg_int(channel):
    print("Charger interrupt triggered")

GPIO.add_event_detect(LANG_BTN_PIN, GPIO.FALLING, callback=on_lang_btn, bouncetime=200)
GPIO.add_event_detect(OCR_BTN_PIN, GPIO.FALLING, callback=on_ocr_btn, bouncetime=200)
GPIO.add_event_detect(CHG_INT_PIN, GPIO.FALLING, callback=on_chg_int, bouncetime=200)
```

### BQ27220 Fuel Gauge Interrupt (GPIO4_A4)

The interrupt is wired directly to the BQ27220 I2C device node in the
overlay. The kernel's `bq27xxx_battery` driver handles it automatically
if loaded. To check:

```bash
# Verify the interrupt is registered
cat /proc/interrupts | grep bq27

# Check power supply class device
ls /sys/class/power_supply/
cat /sys/class/power_supply/bq27220-*/uevent
```

If the kernel driver is not available, read the interrupt GPIO manually:

```python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
BQ_INT_PIN = 132  # GPIO4_A4 (sysfs number, not BCM)

# Using sysfs directly for GPIO4 bank pins
import os

def read_bq_interrupt():
    gpio_path = "/sys/class/gpio/gpio132"
    if not os.path.exists(gpio_path):
        with open("/sys/class/gpio/export", "w") as f:
            f.write("132")
        with open(gpio_path + "/direction", "w") as f:
            f.write("in")
    with open(gpio_path + "/value") as f:
        return int(f.read().strip())
```

## Overlay Structure

```
overlays/
  rk3588-smart-eye-carrier.dts   # Device tree overlay source
  rk3588-smart-eye-carrier.dtbo  # Compiled overlay binary
  Makefile                        # Build/install/enable/disable automation
  USAGE.md                        # This file
```

## Modifying the Overlay

1. Edit `rk3588-smart-eye-carrier.dts`
2. Recompile and reinstall:

```bash
make clean && make
sudo make install
sudo make enable
sudo reboot
```

## Disabling the Overlay

```bash
cd ~/Smart_eye_firmware/overlays
sudo make disable
sudo reboot
```

## Troubleshooting

**Overlay not loading:**
```bash
# Check if the overlay file exists in boot
ls -l /boot/dtbo/rk3588-smart-eye-carrier.dtbo*

# Check managed list
cat /boot/dtbo/managed.list

# Check kernel log for errors
sudo dmesg | grep -i "overlay\|dtbo\|error"
```

**I2C device not detected:**
```bash
# Verify I2C3 bus exists
ls /dev/i2c-*

# Scan with verbose output
sudo i2cdetect -y 3

# Check if pins are muxed correctly
sudo cat /sys/kernel/debug/pinctrl/pinctrl-rockchip-pinctrl/pinmux-pins | grep -i i2c3
```

**UART not available:**
```bash
# Check if UART6 device exists
ls -l /dev/ttyS6

# Verify pin mux
sudo cat /sys/kernel/debug/pinctrl/pinctrl-rockchip-pinctrl/pinmux-pins | grep -i uart6
```

**PWM not appearing:**
```bash
# List all PWM chips
ls /sys/class/pwm/

# Check which PWMs are registered
for chip in /sys/class/pwm/pwmchip*/; do
    echo "$chip: npwm=$(cat ${chip}npwm) device=$(readlink -f ${chip}device)"
done
```

**Buttons not responding:**
```bash
# List all input devices
cat /proc/bus/input/devices

# Check gpio-keys driver
sudo dmesg | grep -i gpio-keys

# Verify GPIO state directly
sudo gpioget gpiochip0 23   # OCR_BTN
sudo gpioget gpiochip0 24   # LANG_BTN
sudo gpioget gpiochip0 27   # CHG_INT0_L
```
