#!/bin/bash
#
# Configure U-Boot to drive GPIO4_B3 low at boot, turning off the
# vibration motor before Linux loads.
#
# GPIO4_B3 = bank 4, offset 11 = global GPIO number 139
# (bank4_base=128, B3=8+3=11, 128+11=139)
#
# The stock Radxa U-Boot supports the "gpio" and "preboot" commands.
# We set the 'preboot' environment variable so U-Boot drives GPIO 139
# low before loading the kernel.
#
# Run this script ON THE BOARD (not on a Windows PC).
#
# Usage: sudo bash setup_uboot_gpio_fix.sh

set -e

echo "=== Vibration Motor U-Boot GPIO Fix ==="
echo ""
echo "This will configure U-Boot to drive GPIO4_B3 (GPIO 139) low"
echo "at every boot, turning off the vibration motor before Linux loads."
echo ""

# Method 1: fw_setenv (preferred -- writes directly to U-Boot env partition)
if command -v fw_setenv &>/dev/null; then
    echo "Using fw_setenv to set preboot command..."

    # Read existing preboot value (if any) to avoid clobbering it
    EXISTING=$(fw_printenv -n preboot 2>/dev/null || true)

    if echo "$EXISTING" | grep -q "gpio clear 139"; then
        echo "  Already configured. Nothing to do."
        exit 0
    fi

    if [ -n "$EXISTING" ]; then
        NEW_PREBOOT="gpio clear 139; ${EXISTING}"
    else
        NEW_PREBOOT="gpio clear 139"
    fi

    fw_setenv preboot "$NEW_PREBOOT"
    echo "  Set preboot = \"$NEW_PREBOOT\""
    echo ""
    echo "Done. Reboot to test: sudo reboot"

# Method 2: guide user through U-Boot console
else
    echo "fw_setenv not found. Install it or set manually from U-Boot console."
    echo ""
    echo "Option A: Install u-boot-tools and re-run this script:"
    echo "  sudo apt-get install u-boot-tools"
    echo "  sudo bash setup_uboot_gpio_fix.sh"
    echo ""
    echo "Option B: Set from U-Boot serial console (interrupt boot with Ctrl+C):"
    echo "  => setenv preboot \"gpio clear 139\""
    echo "  => saveenv"
    echo "  => boot"
    echo ""
fi
