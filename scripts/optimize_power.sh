#!/bin/bash
# Smart Eye - System power optimization script
# Run once after OS setup to disable unnecessary services and configure
# the SoC for minimum power consumption.
#
# Usage: sudo bash optimize_power.sh

set -e

echo "=== Smart Eye Power Optimization ==="

# 1. Disable desktop environment (saves ~200mA)
#    GNOME + Xorg consume significant power even with no display connected.
echo "[1/6] Disabling desktop environment..."
systemctl set-default multi-user.target
systemctl disable gdm.service 2>/dev/null || true

# 2. Disable unnecessary services
echo "[2/6] Disabling unused services..."
for svc in \
    cups.service \
    cups-browsed.service \
    packagekit.service \
    avahi-daemon.service \
    wpa_supplicant.service \
    ModemManager.service \
    bluetooth.service \
    colord.service \
    switcheroo-control.service \
    accounts-daemon.service \
    udisks2.service \
    power-profiles-daemon.service \
; do
    systemctl disable "$svc" 2>/dev/null && echo "  Disabled $svc" || true
    systemctl mask "$svc" 2>/dev/null || true
done

# 3. CPU governor configuration (persistent via sysfs at boot)
echo "[3/6] Creating CPU power governor service..."
cat > /etc/systemd/system/smart-eye-power.service << 'EOF'
[Unit]
Description=Smart Eye CPU Power Governor
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes

# Little cores (A55 cpu0-3): conservative governor
ExecStart=/bin/bash -c 'for c in 0 1 2 3; do echo conservative > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null; done'

# Big cores (A76 cpu4-7): powersave (all cores stay online for onnxruntime compatibility)
ExecStart=/bin/bash -c 'for c in 4 5 6 7; do echo powersave > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null; done'

# GPU: minimum frequency (no GPU workload)
ExecStart=/bin/bash -c 'for g in /sys/class/devfreq/*gpu*; do f=$(head -1 $g/available_frequencies | cut -d" " -f1); echo userspace > $g/governor; echo $f > $g/userspace/set_freq; done 2>/dev/null'

# HDMI: disable (no display connected)
ExecStart=/bin/bash -c 'for d in /sys/class/drm/card*-HDMI* /sys/class/drm/card*-DP*; do echo disabled > $d/enabled 2>/dev/null; done'

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable smart-eye-power.service

# 4. Reduce kernel logging to conserve I/O
echo "[4/6] Reducing kernel log verbosity..."
echo 'kernel.printk = 3 4 1 3' > /etc/sysctl.d/99-smart-eye.conf
sysctl -p /etc/sysctl.d/99-smart-eye.conf 2>/dev/null || true

# 5. Set I/O scheduler to power-efficient mode for eMMC
echo "[5/6] Configuring I/O scheduler..."
for dev in /sys/block/mmcblk*/queue/scheduler; do
    echo "mq-deadline" > "$dev" 2>/dev/null || true
done

# 6. Disable USB autosuspend for connected devices (avoid latency)
echo "[6/6] Configuring USB power management..."
echo 'ACTION=="add", SUBSYSTEM=="usb", ATTR{power/autosuspend_delay_ms}="1000"' \
    > /etc/udev/rules.d/99-usb-power.rules

echo ""
echo "=== Power optimization complete ==="
echo "Estimated power savings: ~200-400mA"
echo "Reboot to apply all changes: sudo reboot"
