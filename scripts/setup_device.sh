#!/bin/bash
# =============================================================================
#  Smart Eye Device Setup Script
#  Configures a fresh Radxa CM5 Lite (RK3582) image for Smart Eye.
#
#  Usage (run from project root on the device):
#    sudo bash scripts/setup_device.sh [OPTIONS]
#
#  Options:
#    --phase N     Run only phase N (1-14)
#    --from N      Start from phase N and continue to end
#    --skip-dl     Skip internet downloads (TTS models, argostranslate)
#    --no-reboot   Do not reboot automatically at the end
#
#  This script is idempotent — safe to re-run.
#  Logs every action to /var/log/smart-eye-setup.log
#
#  Phases:
#    1   System packages (apt)
#    2   ALSA audio config  (smarteye_loud softvol PCM)
#    3   Device tree overlay (compile + install + enable)
#    4   Power optimization  (disable unused services + CPU governor service)
#    5   U-Boot GPIO fix     (vibration motor silent during boot)
#    6   initramfs vibration boot hook
#    7   Vibration motor init service
#    8   Ship mode shutdown service
#    9   Python 3.11 venv + pip packages + RKNN Lite wheel
#   10   TTS engines (espeak-ng + Piper binary + voice models)
#   11   Argostranslate language packages  (needs internet)
#   12   Smart Eye main systemd service
#   13   First-boot power system config (BQ25895 + BQ27220) trigger
#   14   Final verification
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Helpers / globals
# ---------------------------------------------------------------------------

LOG=/var/log/smart-eye-setup.log
PHASE_ONLY=""
FROM_PHASE=1
SKIP_DL=0
NO_REBOOT=0

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

log()  { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }
info() { echo -e "${BLU}[INFO]${RST}  $*"; log "INFO  $*"; }
ok()   { echo -e "${GRN}[OK]${RST}    $*"; log "OK    $*"; }
warn() { echo -e "${YLW}[WARN]${RST}  $*"; log "WARN  $*"; }
err()  { echo -e "${RED}[ERR]${RST}   $*" >&2; log "ERR   $*"; }
hdr()  { echo -e "\n${BLD}${CYN}━━━ Phase $1: $2 ━━━${RST}"; log ""; log "===== Phase $1: $2 ====="; }

die() { err "$*"; exit 1; }

skip_phase() {
    [[ -n "$PHASE_ONLY" && "$PHASE_ONLY" != "$1" ]] && return 0
    [[ "$1" -lt "$FROM_PHASE" ]] && return 0
    return 1
}

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --phase=*)   PHASE_ONLY="${arg#--phase=}" ;;
        --from=*)    FROM_PHASE="${arg#--from=}" ;;
        --skip-dl)   SKIP_DL=1 ;;
        --no-reboot) NO_REBOOT=1 ;;
        --help|-h)
            sed -n '3,20p' "$0"
            exit 0 ;;
        *) die "Unknown option: $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
#  Preflight
# ---------------------------------------------------------------------------

[[ $EUID -ne 0 ]] && die "Must run as root.  Use: sudo bash scripts/setup_device.sh"

ARCH=$(uname -m)
[[ "$ARCH" != "aarch64" ]] && die "This script targets aarch64 (ARM64), not $ARCH."

# Resolve project root from script location regardless of cwd
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$PROJ/pathpal_project/main.py" ]] || die "Cannot find main.py — run from the project root."

PROJ_USER=$(stat -c '%U' "$PROJ")
PYTHON="python3.11"

mkdir -p "$(dirname "$LOG")"
touch "$LOG"

echo -e "\n${BLD}Smart Eye Device Setup${RST}  |  $(date)"
echo -e "Project : ${PROJ}"
echo -e "Log     : ${LOG}"
echo -e "User    : ${PROJ_USER}"
echo ""

# ---------------------------------------------------------------------------
#  Phase 1 — System packages
# ---------------------------------------------------------------------------
if ! skip_phase 1; then
    hdr 1 "System packages"

    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq

    PKGS=(
        # Build tools for DT overlay compilation
        device-tree-compiler
        # U-Boot environment tools (fw_setenv)
        u-boot-tools
        # Python 3.11 and venv
        python3.11 python3.11-venv python3.11-dev
        # Core build / compile deps
        cmake build-essential pkg-config git wget curl
        # Audio utilities
        alsa-utils espeak-ng
        # Camera / V4L2 utilities
        v4l-utils libv4l-dev
        # I2C tools (i2cget, i2cset, i2cdetect)
        i2c-tools
        # Cairo (required by some rapidocr deps)
        libcairo2-dev libglib2.0-dev
        # initramfs tools (for vibration boot hook)
        initramfs-tools
        # Misc
        libopenblas-dev liblapack-dev
    )

    MISSING=()
    for pkg in "${PKGS[@]}"; do
        dpkg -s "$pkg" &>/dev/null || MISSING+=("$pkg")
    done

    if [[ ${#MISSING[@]} -eq 0 ]]; then
        ok "All system packages already installed"
    else
        info "Installing: ${MISSING[*]}"
        apt-get install -y "${MISSING[@]}" 2>&1 | tee -a "$LOG"
        ok "System packages installed"
    fi
fi

# ---------------------------------------------------------------------------
#  Phase 2 — ALSA audio configuration
# ---------------------------------------------------------------------------
if ! skip_phase 2; then
    hdr 2 "ALSA audio configuration"

    ASOUND=/etc/asound.conf

    # The MAX98357A speaker card ("SmartEyeAudio") needs a softvol PCM alias
    # so main.py can do:  aplay -D smarteye_loud <file>
    # The softvol caps at 90% (-0.9 dB) to prevent overdriving the 15 dB
    # fixed hardware gain of the MAX98357A.
    # The headphone card (rockchipes8316) is used directly via plughw — no
    # extra config needed for that path.

    if grep -q "smarteye_loud" "$ASOUND" 2>/dev/null; then
        ok "ALSA softvol config already present"
    else
        info "Writing $ASOUND"
        cat > "$ASOUND" <<'ASOUND_EOF'
# Smart Eye ALSA system config
# Defines the softvol PCM alias used by main.py for speaker output.
# The MAX98357A amp has a fixed +15 dB gain — we cap volume at 90 % here
# to protect the speaker and prevent digital clipping.

pcm.smarteye_loud {
    type            softvol
    slave.pcm       "plughw:SmartEyeAudio,0"
    control {
        name    "SmartEye Speaker Volume"
        card    "SmartEyeAudio"
    }
    max_dB          -0.9
    min_dB          -51.0
    resolution      256
}

ctl.smarteye_loud {
    type hw
    card "SmartEyeAudio"
}
ASOUND_EOF
        ok "ALSA config written"
    fi
fi

# ---------------------------------------------------------------------------
#  Phase 3 — Device tree overlay
# ---------------------------------------------------------------------------
if ! skip_phase 3; then
    hdr 3 "Device tree overlay"

    OVERLAY_SRC="$PROJ/Overlays/smart-eye-carrier.dts"
    OVERLAY_DTBO="$PROJ/Overlays/smart-eye-carrier.dtbo"
    BOOT_DTBO_DIR="/boot/dtbo"

    [[ -f "$OVERLAY_SRC" ]] || die "Overlay source not found: $OVERLAY_SRC"

    # Compile
    info "Compiling overlay..."
    dtc -@ -I dts -O dtb -o "$OVERLAY_DTBO" "$OVERLAY_SRC" 2>&1 | tee -a "$LOG"
    ok "Compiled $OVERLAY_DTBO"

    # Determine kernel overlay directory
    KVER=$(uname -r)
    OVERLAY_KDIR="/usr/lib/linux-image-${KVER}/rockchip/overlays"
    [[ -d "$OVERLAY_KDIR" ]] || { warn "Kernel overlay dir not found: $OVERLAY_KDIR — skipping kernel copy"; OVERLAY_KDIR=""; }

    # Install into kernel overlay dir
    if [[ -n "$OVERLAY_KDIR" ]]; then
        cp "$OVERLAY_DTBO" "$OVERLAY_KDIR/"
        ok "Installed to $OVERLAY_KDIR/"
    fi

    # Install into /boot/dtbo
    mkdir -p "$BOOT_DTBO_DIR"
    DTBO_NAME="smart-eye-carrier.dtbo"
    DTBO_BOOT="$BOOT_DTBO_DIR/$DTBO_NAME"
    DTBO_DISABLED="$BOOT_DTBO_DIR/$DTBO_NAME.disabled"

    if [[ -f "$DTBO_BOOT" ]]; then
        ok "Overlay already enabled in $BOOT_DTBO_DIR"
    else
        cp "$OVERLAY_DTBO" "$DTBO_DISABLED"
        mv "$DTBO_DISABLED" "$DTBO_BOOT"
        # Add to managed.list if it exists
        if [[ -f "$BOOT_DTBO_DIR/managed.list" ]]; then
            grep -qF "$DTBO_NAME" "$BOOT_DTBO_DIR/managed.list" || \
                echo "$DTBO_NAME" >> "$BOOT_DTBO_DIR/managed.list"
        fi
        ok "Overlay enabled: $DTBO_BOOT"
    fi

    warn "Overlay will take effect after reboot (performed at end of setup)"
fi

# ---------------------------------------------------------------------------
#  Phase 4 — Power optimization
# ---------------------------------------------------------------------------
if ! skip_phase 4; then
    hdr 4 "Power optimization"

    # Disable unnecessary services (safe to mask — none are needed)
    MASK_SVCS=(
        cups.service cups-browsed.service packagekit.service
        avahi-daemon.service wpa_supplicant.service ModemManager.service
        bluetooth.service colord.service switcheroo-control.service
        accounts-daemon.service udisks2.service power-profiles-daemon.service
        gdm.service gdm3.service
    )
    for svc in "${MASK_SVCS[@]}"; do
        if systemctl is-enabled "$svc" &>/dev/null; then
            systemctl disable "$svc" 2>/dev/null || true
            systemctl mask "$svc"   2>/dev/null || true
            info "Masked $svc"
        fi
    done

    # Default to text console (saves ~200 mA vs GNOME/Xorg)
    systemctl set-default multi-user.target
    ok "Default target: multi-user"

    # CPU governor service (persistent across reboots)
    CPU_SVC=/etc/systemd/system/smart-eye-power.service
    if [[ ! -f "$CPU_SVC" ]]; then
        info "Creating $CPU_SVC"
        cat > "$CPU_SVC" <<'EOF'
[Unit]
Description=Smart Eye CPU Power Governor
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes

# Little cores A55 (cpu0-3): conservative governor
ExecStart=/bin/bash -c 'for c in 0 1 2 3; do echo conservative > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true; done'

# Big cores A76 (cpu4-7): powersave
ExecStart=/bin/bash -c 'for c in 4 5 6 7; do echo powersave > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true; done'

# GPU: minimum frequency (no display workload)
ExecStart=/bin/bash -c 'for g in /sys/class/devfreq/*gpu*; do f=$(head -1 "$g/available_frequencies" | cut -d" " -f1); echo userspace > "$g/governor"; echo "$f" > "$g/userspace/set_freq"; done 2>/dev/null || true'

# HDMI / DP outputs: disable (no display connected on Smart Eye)
ExecStart=/bin/bash -c 'for d in /sys/class/drm/card*-HDMI* /sys/class/drm/card*-DP*; do [ -f "$d/enabled" ] && echo disabled > "$d/enabled"; done 2>/dev/null || true'

# BQ25895 watchdog: disable so register settings survive across minutes
# REG07 = 0x8F: WDT=OFF, EN_TERM=ON, CHG_TIMER=20h
ExecStart=/bin/bash -c 'i2cset -y 3 0x6a 0x07 0x8f b 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
EOF
        systemctl daemon-reload
        systemctl enable smart-eye-power.service
        ok "smart-eye-power.service enabled"
    else
        ok "smart-eye-power.service already exists"
    fi

    # Kernel log verbosity (reduce eMMC writes)
    SYSCTL=/etc/sysctl.d/99-smart-eye.conf
    if [[ ! -f "$SYSCTL" ]]; then
        echo 'kernel.printk = 3 4 1 3' > "$SYSCTL"
        sysctl -p "$SYSCTL" 2>/dev/null || true
        ok "Kernel log verbosity reduced"
    fi

    # eMMC I/O scheduler
    for dev in /sys/block/mmcblk*/queue/scheduler; do
        echo "mq-deadline" > "$dev" 2>/dev/null || true
    done

    ok "Power optimization complete"
fi

# ---------------------------------------------------------------------------
#  Phase 5 — U-Boot GPIO fix (vibration motor silent during boot)
# ---------------------------------------------------------------------------
if ! skip_phase 5; then
    hdr 5 "U-Boot GPIO fix"

    # GPIO0_C5 (global #21) drives the vibration motor via PWM4-M0.
    # Without this fix, U-Boot leaves the pin floating and the motor buzzes
    # for several seconds before Linux takes control.

    FWENV=/etc/fw_env.config
    if [[ ! -f "$FWENV" ]]; then
        info "Creating $FWENV for Radxa CM5 eMMC layout"
        cat > "$FWENV" <<'EOF'
# Radxa CM5 eMMC — U-Boot environment partition offsets
# Verify with: strings /dev/mmcblk0 | grep -i preboot
/dev/mmcblk0    0x3F8000    0x8000
/dev/mmcblk0    0x3FF8000   0x8000
EOF
    fi

    if command -v fw_setenv &>/dev/null && [[ -f "$FWENV" ]]; then
        GPIO_CMD="gpio clear 21"
        EXISTING=$(fw_printenv -n preboot 2>/dev/null || true)
        if echo "$EXISTING" | grep -q "$GPIO_CMD"; then
            ok "U-Boot preboot already contains GPIO clear"
        else
            if [[ -n "$EXISTING" ]]; then
                fw_setenv preboot "${GPIO_CMD}; ${EXISTING}"
            else
                fw_setenv preboot "${GPIO_CMD}"
            fi
            ok "U-Boot preboot set: gpio clear 21 (GPIO0_C5 LOW before kernel)"
        fi
    else
        warn "fw_setenv not available — U-Boot GPIO fix skipped"
        warn "Manually run in U-Boot console:"
        warn "  => setenv preboot \"gpio clear 21\""
        warn "  => saveenv"
    fi
fi

# ---------------------------------------------------------------------------
#  Phase 6 — initramfs vibration boot hook
# ---------------------------------------------------------------------------
if ! skip_phase 6; then
    hdr 6 "initramfs vibration boot hook"

    SRC="$PROJ/scripts/initramfs-vibration-boot"
    DST=/etc/initramfs-tools/scripts/init-bottom/smart-eye-vibration

    [[ -f "$SRC" ]] || die "initramfs hook source not found: $SRC"

    if [[ -f "$DST" ]] && cmp -s "$SRC" "$DST"; then
        ok "initramfs hook already installed (unchanged)"
    else
        install -m 0755 "$SRC" "$DST"
        info "Rebuilding initramfs (this takes ~30 s)..."
        update-initramfs -u 2>&1 | tee -a "$LOG"
        ok "initramfs hook installed and initramfs rebuilt"
    fi
fi

# ---------------------------------------------------------------------------
#  Phase 7 — Vibration motor init service
# ---------------------------------------------------------------------------
if ! skip_phase 7; then
    hdr 7 "Vibration motor init service"

    # Script: hands GPIO0_C5 from sysfs to the PWM4 driver at boot
    SRC_SH="$PROJ/scripts/vibration-motor-init.sh"
    DST_SH=/usr/local/bin/vibration-motor-init.sh
    SRC_SVC="$PROJ/services/vibration-motor-init.service"
    DST_SVC=/etc/systemd/system/vibration-motor-init.service

    [[ -f "$SRC_SH"  ]] || die "Not found: $SRC_SH"
    [[ -f "$SRC_SVC" ]] || die "Not found: $SRC_SVC"

    install -m 0755 "$SRC_SH"  "$DST_SH"
    install -m 0644 "$SRC_SVC" "$DST_SVC"

    systemctl daemon-reload
    systemctl enable vibration-motor-init.service
    ok "vibration-motor-init.service enabled"
fi

# ---------------------------------------------------------------------------
#  Phase 8 — Ship mode shutdown service
# ---------------------------------------------------------------------------
if ! skip_phase 8; then
    hdr 8 "Ship mode shutdown service"

    # On poweroff: vibration feedback (3 pulses) + BQ25895 BATFET_DIS
    # → SYS collapses to ~0 V → ≈2 µA drain instead of ~100 mA
    # Wake from ship mode: press power button (QON pulled low via diode)
    # or plug in USB (VBUS presence re-enables BATFET automatically).

    OPT_DIR=/opt/battery-mgr
    SRC_SH="$PROJ/scripts/shutdown_ship_mode.sh"
    DST_SH="$OPT_DIR/shutdown_ship_mode.sh"
    SRC_SVC="$PROJ/services/ship-mode-shutdown.service"
    DST_SVC=/etc/systemd/system/ship-mode-shutdown.service

    [[ -f "$SRC_SH"  ]] || die "Not found: $SRC_SH"
    [[ -f "$SRC_SVC" ]] || die "Not found: $SRC_SVC"

    mkdir -p "$OPT_DIR"
    install -m 0755 "$SRC_SH"  "$DST_SH"
    install -m 0644 "$SRC_SVC" "$DST_SVC"

    systemctl daemon-reload
    systemctl enable ship-mode-shutdown.service
    ok "ship-mode-shutdown.service enabled"
fi

# ---------------------------------------------------------------------------
#  Phase 9 — Python 3.11 venv + pip packages + RKNN Lite wheel
# ---------------------------------------------------------------------------
if ! skip_phase 9; then
    hdr 9 "Python 3.11 venv + packages"

    VENV="$PROJ/venv"
    RKNN_WHL=$(ls "$PROJ"/rknn_toolkit_lite2-*.whl 2>/dev/null | head -1)

    # Create venv if missing
    if [[ ! -f "$VENV/bin/activate" ]]; then
        info "Creating venv at $VENV"
        $PYTHON -m venv "$VENV" --system-site-packages
    fi

    PIP="$VENV/bin/pip"
    "$PIP" install --upgrade pip --quiet 2>&1 | tee -a "$LOG"

    # Install project requirements
    info "Installing requirements.txt..."
    "$PIP" install -r "$PROJ/requirements.txt" --quiet 2>&1 | tee -a "$LOG"
    ok "requirements.txt installed"

    # Install RKNN Lite wheel (bundled in repo, aarch64 cp311 only)
    if [[ -n "$RKNN_WHL" ]]; then
        WHL_NAME=$(basename "$RKNN_WHL")
        if "$PIP" show rknn-toolkit-lite2 &>/dev/null; then
            ok "RKNN Lite already installed ($(\"$PIP\" show rknn-toolkit-lite2 | grep Version))"
        else
            info "Installing $WHL_NAME"
            "$PIP" install "$RKNN_WHL" --quiet 2>&1 | tee -a "$LOG"
            ok "RKNN Lite wheel installed"
        fi
    else
        warn "No RKNN Lite wheel found in $PROJ — install manually if needed"
    fi

    # Fix ownership so radxa user can use the venv
    chown -R "${PROJ_USER}:${PROJ_USER}" "$VENV"
    ok "venv owned by $PROJ_USER"
fi

# ---------------------------------------------------------------------------
#  Phase 10 — TTS engines
# ---------------------------------------------------------------------------
if ! skip_phase 10; then
    hdr 10 "TTS engines (espeak-ng + Piper)"

    PIPER_DIR="$PROJ/piper"
    mkdir -p "$PIPER_DIR"

    # espeak-ng is installed in phase 1 — just verify
    if command -v espeak-ng &>/dev/null; then
        ok "espeak-ng present: $(espeak-ng --version 2>&1 | head -1)"
    else
        warn "espeak-ng not found — re-run phase 1"
    fi

    # Piper binary
    PIPER_BIN="$PIPER_DIR/piper/piper"
    if [[ ! -x "$PIPER_BIN" ]]; then
        if [[ "$SKIP_DL" -eq 1 ]]; then
            warn "Piper binary not found and --skip-dl set — skipping download"
        else
            info "Downloading Piper TTS binary (aarch64)..."
            PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
            wget -q "$PIPER_URL" -O /tmp/piper.tar.gz
            tar xzf /tmp/piper.tar.gz -C "$PIPER_DIR"
            rm -f /tmp/piper.tar.gz
            chmod +x "$PIPER_BIN"
            ok "Piper binary installed: $PIPER_BIN"
        fi
    else
        ok "Piper binary already present"
    fi

    # Voice models
    VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
    declare -A VOICES=(
        ["en_US-amy-medium"]="en/en_US/amy/medium/en_US-amy-medium"
        ["hi_IN-pratham-medium"]="hi/hi_IN/pratham/medium/hi_IN-pratham-medium"
    )

    for name in "${!VOICES[@]}"; do
        ONNX="$PIPER_DIR/${name}.onnx"
        if [[ -f "$ONNX" ]]; then
            ok "Voice model already present: $name"
        elif [[ "$SKIP_DL" -eq 1 ]]; then
            warn "Voice model missing and --skip-dl set: $name"
        else
            info "Downloading voice model: $name"
            wget -q "${VOICE_BASE}/${VOICES[$name]}.onnx"      -O "$ONNX"
            wget -q "${VOICE_BASE}/${VOICES[$name]}.onnx.json" -O "${ONNX}.json"
            ok "Downloaded: $name"
        fi
    done

    # Smoke-test Piper if binary and model are available
    if [[ -x "$PIPER_BIN" && -f "$PIPER_DIR/en_US-amy-medium.onnx" ]]; then
        if echo "test" | "$PIPER_BIN" \
               --model "$PIPER_DIR/en_US-amy-medium.onnx" \
               --output_file /tmp/se_piper_test.wav 2>/dev/null; then
            ok "Piper smoke test passed"
        else
            warn "Piper smoke test failed — check $PIPER_BIN and model"
        fi
        rm -f /tmp/se_piper_test.wav
    fi

    chown -R "${PROJ_USER}:${PROJ_USER}" "$PIPER_DIR"
fi

# ---------------------------------------------------------------------------
#  Phase 11 — Argostranslate language packages
# ---------------------------------------------------------------------------
if ! skip_phase 11; then
    hdr 11 "Argostranslate (en↔hi offline translation)"

    ARGOSPM="$PROJ/venv/bin/argospm"

    if [[ ! -x "$ARGOSPM" ]]; then
        warn "argospm not found in venv — skipping (run phase 9 first)"
    elif [[ "$SKIP_DL" -eq 1 ]]; then
        warn "--skip-dl set — skipping argostranslate package download"
    else
        for pkg in translate-en_hi translate-hi_en; do
            if "$ARGOSPM" list 2>/dev/null | grep -q "$pkg"; then
                ok "Argostranslate package already installed: $pkg"
            else
                info "Installing argostranslate: $pkg"
                "$ARGOSPM" install "$pkg" 2>&1 | tee -a "$LOG" || \
                    warn "Failed to install $pkg — check internet access"
            fi
        done
        ok "Argostranslate packages done"
    fi
fi

# ---------------------------------------------------------------------------
#  Phase 12 — Smart Eye main systemd service
# ---------------------------------------------------------------------------
if ! skip_phase 12; then
    hdr 12 "Smart Eye main systemd service"

    SVC=/etc/systemd/system/smart-eye.service
    MODEL_DEFAULT="$PROJ/models/pathpal/model_v2_large.rknn"

    if [[ ! -f "$MODEL_DEFAULT" ]]; then
        warn "Default RKNN model not found: $MODEL_DEFAULT"
        warn "Ensure the model file is present before starting the service"
    fi

    # Write service only if missing or PROJ path has changed
    NEED_WRITE=1
    if [[ -f "$SVC" ]] && grep -qF "$PROJ" "$SVC"; then
        NEED_WRITE=0
    fi

    if [[ "$NEED_WRITE" -eq 1 ]]; then
        info "Writing $SVC"
        cat > "$SVC" <<EOF
[Unit]
Description=Smart Eye Assistive Vision System
Documentation=file://${PROJ}/README.md
After=multi-user.target sound.target vibration-motor-init.service
Wants=sound.target vibration-motor-init.service
# Do not start until the battery manager is configured
After=smart-eye-first-boot.service

[Service]
Type=simple
WorkingDirectory=${PROJ}

# Must run as root: writes CPU governors, exports GPIOs, opens /dev/i2c-*
ExecStart=${PROJ}/venv/bin/python3 pathpal_project/main.py

Restart=on-failure
RestartSec=5
StartLimitBurst=3
StartLimitIntervalSec=30

# Keep the venv environment (set by sudo -E when running manually)
Environment=HOME=/home/${PROJ_USER}
Environment=PYTHONUNBUFFERED=1

StandardOutput=journal
StandardError=journal
SyslogIdentifier=smart-eye

[Install]
WantedBy=multi-user.target
EOF
        systemctl daemon-reload
        ok "smart-eye.service written"
    else
        ok "smart-eye.service already present for this project path"
    fi

    systemctl enable smart-eye.service
    ok "smart-eye.service enabled (auto-starts on boot)"
fi

# ---------------------------------------------------------------------------
#  Phase 13 — First-boot power system config (BQ25895 + BQ27220)
# ---------------------------------------------------------------------------
if ! skip_phase 13; then
    hdr 13 "First-boot power system config"

    # power_config.py programs the BQ25895 charger registers and the BQ27220
    # fuel gauge data memory for the 10000 mAh battery.
    # It requires /dev/i2c-3 which only appears after the overlay is loaded
    # (i.e., after reboot).  We install a one-shot systemd service that runs
    # power_config.py on the first boot after setup and then disables itself.

    SENTINEL=/etc/smart-eye-first-boot
    FIRST_BOOT_SVC=/etc/systemd/system/smart-eye-first-boot.service
    FIRST_BOOT_SH=/opt/battery-mgr/first_boot_power_config.sh
    OPT_DIR=/opt/battery-mgr

    mkdir -p "$OPT_DIR"

    # Write the first-boot shell wrapper
    cat > "$FIRST_BOOT_SH" <<EOF
#!/bin/bash
# One-shot: configure BQ25895 + BQ27220 on first boot after overlay is active.
LOG=/var/log/smart-eye-setup.log
echo "\$(date '+%Y-%m-%d %H:%M:%S')  [FIRST-BOOT] Running power_config.py" >> "\$LOG"

if ! i2cdetect -y 3 2>/dev/null | grep -q "6a"; then
    echo "\$(date '+%Y-%m-%d %H:%M:%S')  [FIRST-BOOT] BQ25895 not found on i2c-3 — overlay may not be loaded" >> "\$LOG"
    exit 1
fi

cd ${PROJ}
${PROJ}/venv/bin/python3 power_config.py 2>&1 | tee -a "\$LOG"
echo "\$(date '+%Y-%m-%d %H:%M:%S')  [FIRST-BOOT] power_config.py complete" >> "\$LOG"
EOF
    chmod 0755 "$FIRST_BOOT_SH"

    # Write the one-shot service
    cat > "$FIRST_BOOT_SVC" <<EOF
[Unit]
Description=Smart Eye First-Boot Power System Configuration
# Run after I2C devices are settled (overlay must be active)
After=network.target i2c.target multi-user.target
# Only run once — sentinel file removed on success
ConditionPathExists=${SENTINEL}

[Service]
Type=oneshot
ExecStart=${FIRST_BOOT_SH}
# Remove sentinel on success so this service never runs again
ExecStartPost=/bin/rm -f ${SENTINEL}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    # Create sentinel so the service fires on the first boot after reboot
    touch "$SENTINEL"

    systemctl daemon-reload
    systemctl enable smart-eye-first-boot.service
    ok "smart-eye-first-boot.service enabled (runs once after first reboot)"
    ok "Sentinel: $SENTINEL"
fi

# ---------------------------------------------------------------------------
#  Phase 14 — Final verification
# ---------------------------------------------------------------------------
if ! skip_phase 14; then
    hdr 14 "Final verification"

    PASS=0; FAIL=0

    chk() {
        if eval "$2" &>/dev/null; then
            ok "  $1"
            ((PASS++)) || true
        else
            err "  $1  [MISSING]"
            ((FAIL++)) || true
        fi
    }

    echo ""
    echo -e "${BLD}System:${RST}"
    chk "dtc (device-tree-compiler)"     "command -v dtc"
    chk "i2c-tools (i2cdetect)"          "command -v i2cdetect"
    chk "espeak-ng"                      "command -v espeak-ng"
    chk "aplay (alsa-utils)"             "command -v aplay"
    chk "v4l2-ctl"                       "command -v v4l2-ctl"

    echo ""
    echo -e "${BLD}Device Tree:${RST}"
    chk "Overlay DTBO compiled"          "test -f $PROJ/Overlays/smart-eye-carrier.dtbo"
    chk "Overlay installed in /boot"     "test -f /boot/dtbo/smart-eye-carrier.dtbo"

    echo ""
    echo -e "${BLD}Services:${RST}"
    chk "vibration-motor-init.service"   "systemctl is-enabled vibration-motor-init.service"
    chk "ship-mode-shutdown.service"     "systemctl is-enabled ship-mode-shutdown.service"
    chk "smart-eye-power.service"        "systemctl is-enabled smart-eye-power.service"
    chk "smart-eye-first-boot.service"   "systemctl is-enabled smart-eye-first-boot.service"
    chk "smart-eye.service"              "systemctl is-enabled smart-eye.service"

    echo ""
    echo -e "${BLD}Files:${RST}"
    chk "initramfs vibration hook"       "test -x /etc/initramfs-tools/scripts/init-bottom/smart-eye-vibration"
    chk "vibration-motor-init.sh"        "test -x /usr/local/bin/vibration-motor-init.sh"
    chk "shutdown_ship_mode.sh"          "test -x /opt/battery-mgr/shutdown_ship_mode.sh"
    chk "first_boot_power_config.sh"     "test -x /opt/battery-mgr/first_boot_power_config.sh"
    chk "ALSA asound.conf"               "grep -q smarteye_loud /etc/asound.conf"

    echo ""
    echo -e "${BLD}Python:${RST}"
    chk "venv created"                   "test -f $PROJ/venv/bin/activate"
    chk "rapidocr-onnxruntime"           "$PROJ/venv/bin/pip show rapidocr-onnxruntime"
    chk "argostranslate"                 "$PROJ/venv/bin/pip show argostranslate"
    chk "pyserial"                       "$PROJ/venv/bin/pip show pyserial"
    chk "rknn-toolkit-lite2"             "$PROJ/venv/bin/pip show rknn-toolkit-lite2"

    echo ""
    echo -e "${BLD}TTS:${RST}"
    chk "Piper binary"                   "test -x $PROJ/piper/piper/piper"
    chk "English voice (amy-medium)"     "test -f $PROJ/piper/en_US-amy-medium.onnx"
    chk "Hindi voice (pratham-medium)"   "test -f $PROJ/piper/hi_IN-pratham-medium.onnx"

    echo ""
    echo -e "${BLD}Models:${RST}"
    chk "PathPal RKNN model"             "test -f $PROJ/models/pathpal/model_v2_large.rknn"
    chk "labels.txt"                     "test -f $PROJ/models/pathpal/labels.txt"

    echo ""
    echo -e "${BLD}Audio assets:${RST}"
    chk "English WAV files"              "ls $PROJ/wav/English/*.wav"
    chk "Hindi WAV files"                "ls $PROJ/wav/Hindi/*.wav"

    echo ""
    echo -e "─────────────────────────────────────────"
    echo -e "  ${GRN}PASS: $PASS${RST}   ${RED}FAIL: $FAIL${RST}"
    echo -e "─────────────────────────────────────────"
    log "Verification: PASS=$PASS FAIL=$FAIL"
fi

# ---------------------------------------------------------------------------
#  Post-setup summary + reboot
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLD}${CYN}━━━ Setup complete ━━━${RST}"
echo ""
echo -e "What happens on first reboot:"
echo -e "  1. Device tree overlay loads → I2C3, UARTs, PWM4, I2S1, buttons active"
echo -e "  2. smart-eye-first-boot.service runs power_config.py"
echo -e "     → BQ25895 charger registers set (4.208 V, 2 A charge, WDT off)"
echo -e "     → BQ27220 fuel gauge programmed for 10000 mAh"
echo -e "  3. smart-eye-power.service sets CPU governors + disables HDMI"
echo -e "  4. vibration-motor-init.service hands GPIO0_C5 to PWM4 driver"
echo -e "  5. smart-eye.service starts the main app (auto on every boot)"
echo ""
echo -e "After reboot, verify with:"
echo -e "  sudo journalctl -u smart-eye -f"
echo -e "  sudo journalctl -u smart-eye-first-boot"
echo -e "  sudo python3 ${PROJ}/tests/test_all.py"
echo -e "  sudo python3 ${PROJ}/tests/battery_test.py"
echo ""
echo -e "Manual power system check (after reboot):"
echo -e "  sudo python3 ${PROJ}/power_config.py --status"
echo ""

log "Setup complete. Rebooting..."

if [[ "$NO_REBOOT" -eq 1 ]]; then
    warn "--no-reboot set. Reboot manually: sudo reboot"
else
    echo -e "${YLW}Rebooting in 5 seconds — press Ctrl+C to cancel${RST}"
    sleep 5
    reboot
fi
