#!/bin/bash
#
# Configure U-Boot to drive GPIO0_C5 low at boot, turning off the
# vibration motor before Linux loads.
#
# GPIO0_C5 = bank 0, group C, pin 5 = global GPIO 21 (0*32 + 2*8 + 5)
# PWM4-M0 on this pin drives the vibration motor.
#
# U-Boot runs before the kernel's pinctrl/DTS takes effect, so without
# this fix the motor stays ON from power-on until Linux initializes the
# GPIO pull-down (several seconds into boot).
#
# Run this script ON THE BOARD:
#   sudo bash scripts/setup_uboot_gpio_fix.sh

set -e

GPIO_NUM=21          # global GPIO number for GPIO0_C5
GPIO_CMD="gpio clear ${GPIO_NUM}"   # U-Boot command: set pin LOW

echo "=== Vibration Motor U-Boot GPIO Fix ==="
echo "Target: GPIO0_C5 (global ${GPIO_NUM}) — drive LOW before kernel loads"
echo ""

# ── Method 1: fw_setenv (preferred) ─────────────────────────────────────────
if command -v fw_setenv &>/dev/null; then

    # Verify fw_env.config exists and points to a real partition
    if [ ! -f /etc/fw_env.config ]; then
        echo "ERROR: /etc/fw_env.config not found."
        echo "Create it for your eMMC layout (see Method 3 below) then re-run."
        exit 1
    fi

    echo "Using fw_setenv (writes to U-Boot env partition)..."

    EXISTING=$(fw_printenv -n preboot 2>/dev/null || true)

    if echo "$EXISTING" | grep -q "${GPIO_CMD}"; then
        echo "  Already configured (preboot already contains '${GPIO_CMD}'). Nothing to do."
        exit 0
    fi

    if [ -n "$EXISTING" ]; then
        NEW_PREBOOT="${GPIO_CMD}; ${EXISTING}"
    else
        NEW_PREBOOT="${GPIO_CMD}"
    fi

    fw_setenv preboot "$NEW_PREBOOT"
    echo "  preboot = \"$NEW_PREBOOT\""
    echo ""
    echo "Done. Reboot to verify: sudo reboot"

# ── Method 2: u-boot-tools not installed ────────────────────────────────────
elif ! command -v fw_setenv &>/dev/null; then
    echo "fw_setenv not found. Install u-boot-tools and re-run:"
    echo "  sudo apt-get install u-boot-tools"
    echo "  sudo bash scripts/setup_uboot_gpio_fix.sh"
    echo ""
    echo "If fw_env.config is missing, see Method 3 below."
    echo ""
    print_manual_steps
fi

# ── Method 3: Manual — U-Boot serial console ────────────────────────────────
print_manual_steps() {
    echo "─── Manual fix via U-Boot serial console ───────────────────────────"
    echo "Connect a USB-UART adapter to the debug UART pins on the carrier board."
    echo "Power on and interrupt U-Boot with any key within ~2 s."
    echo ""
    echo "Then at the => prompt, run:"
    echo ""
    echo "  => setenv preboot \"gpio clear ${GPIO_NUM}\""
    echo "  => saveenv"
    echo "  => boot"
    echo ""
    echo "The 'gpio clear' command drives GPIO0_C5 LOW before the kernel loads,"
    echo "cutting power to the vibration motor instantly on every subsequent boot."
    echo "────────────────────────────────────────────────────────────────────"
}

# ── Verify fw_env.config for Radxa CM5 ──────────────────────────────────────
# On Radxa CM5 (eMMC boot), U-Boot env is typically at:
#   /dev/mmcblk0  offset 0x3F8000  size 0x8000  (two copies: 0x3F8000 + 0x400000)
#
# If /etc/fw_env.config is missing or wrong, create it:
#   sudo tee /etc/fw_env.config <<'EOF'
#   # Radxa CM5 eMMC — U-Boot environment (verify offsets with: strings /dev/mmcblk0 | grep preboot)
#   /dev/mmcblk0    0x3F8000    0x8000
#   /dev/mmcblk0    0x3FF8000   0x8000
#   EOF
