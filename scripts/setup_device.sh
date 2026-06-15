#!/bin/bash
# =============================================================================
#  Smart Eye Device Setup Script
#  Configures a fresh Radxa CM5 Lite (RK3582) image for Smart Eye.
#
#  Usage (run from project root on the device):
#    sudo bash scripts/setup_device.sh [OPTIONS]
#
#  Options:
#    --phase N     Run only phase N (1-15)
#    --from N      Start from phase N and continue to end
#    --skip-dl     Skip internet downloads (TTS models, argostranslate)
#    --offline     No internet: skip apt update + all downloads (auto-detected)
#    --no-reboot   Do not reboot automatically at the end
#
#  This script is idempotent — safe to re-run.
#  Logs every action to /var/log/smart-eye-setup.log
#
#  Phases:
#    1   System packages (apt)
#    2   ALSA audio config  (smarteye_loud softvol PCM)
#    3   Device tree overlay (carrier + IMX219 camera + FIQ debugger)
#    4   Power optimization  (disable unused services + CPU governor service)
#    5   U-Boot GPIO fix     (vibration motor silent during boot)
#    6   initramfs vibration boot hook
#    7   Vibration motor init service
#    8   Ship mode shutdown service
#    9   USB network (usb0 static IP 192.168.2.100 + no-sleep)
#   10   Python 3.11 venv + pip packages + RKNN Lite wheel
#   11   TTS engines (espeak-ng + Piper binary + voice models)
#   12   Argostranslate language packages  (needs internet)
#   13   Smart Eye main systemd service
#   14   First-boot power system config (BQ25895 + BQ27220) trigger
#   15   Final verification
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
OFFLINE=0

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
        --offline)   OFFLINE=1; SKIP_DL=1 ;;
        --no-reboot) NO_REBOOT=1 ;;
        --help|-h)
            sed -n '3,21p' "$0"
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

# ---------------------------------------------------------------------------
#  Connectivity auto-detection
#  Uses bash built-in TCP so no curl/wget needed at this early stage.
# ---------------------------------------------------------------------------
if [[ "$OFFLINE" -eq 0 ]]; then
    if timeout 3 bash -c 'echo > /dev/tcp/8.8.8.8/53' 2>/dev/null; then
        true  # online — leave OFFLINE=0
    else
        warn "No internet connection detected — switching to offline mode"
        warn "Pass --offline explicitly to suppress this message."
        OFFLINE=1
        SKIP_DL=1
    fi
fi

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
    if [[ "$OFFLINE" -eq 0 ]]; then
        apt-get update -qq
    else
        info "Offline: skipping apt-get update (using cached package lists)"
    fi

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
    elif [[ "$OFFLINE" -eq 1 ]]; then
        warn "Offline: cannot install missing packages: ${MISSING[*]}"
        warn "Connect to internet then re-run phase 1:  sudo bash scripts/setup_device.sh --from=1 --phase=1"
        warn "Or install manually: sudo apt-get install ${MISSING[*]}"
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

    # ── Camera overlay: IMX219 (RPi Camera Module V2) on CSI-0 ─────────────
    #
    # The Radxa BSP often ships a pre-built IMX219 overlay.  We prefer that
    # over our local DTS because it is tuned to the exact BSP kernel version.
    # If not present, we compile Overlays/imx219-cam0.dts ourselves.

    info "Setting up IMX219 camera overlay (CSI-0)..."

    CAMERA_DTBO="imx219-cam0.dtbo"
    CAMERA_BOOT="$BOOT_DTBO_DIR/$CAMERA_DTBO"

    # Search for a pre-built Radxa overlay first (several known names)
    CAMERA_SRC=""
    for candidate in \
        "${OVERLAY_KDIR}/rk3588-camera-imx219-cam0.dtbo" \
        "${OVERLAY_KDIR}/rk3588s-camera-imx219-cam0.dtbo" \
        "${OVERLAY_KDIR}/radxa-camera-imx219-cam0.dtbo" \
        "/usr/share/dtbs/$(uname -r)/rockchip/overlays/rk3588-camera-imx219-cam0.dtbo" \
        "$BOOT_DTBO_DIR/rk3588-camera-imx219-cam0.dtbo"; do
        if [[ -f "$candidate" ]]; then
            CAMERA_SRC="$candidate"
            CAMERA_DTBO=$(basename "$candidate")
            CAMERA_BOOT="$BOOT_DTBO_DIR/$CAMERA_DTBO"
            ok "  Found Radxa camera overlay: $CAMERA_SRC"
            break
        fi
    done

    # Fall back to our local DTS
    if [[ -z "$CAMERA_SRC" ]]; then
        LOCAL_DTS="$PROJ/Overlays/imx219-cam0.dts"
        LOCAL_DTBO="$PROJ/Overlays/imx219-cam0.dtbo"
        if [[ -f "$LOCAL_DTS" ]]; then
            info "  Compiling $LOCAL_DTS..."
            dtc -@ -I dts -O dtb -o "$LOCAL_DTBO" "$LOCAL_DTS" 2>&1 | tee -a "$LOG"
            CAMERA_SRC="$LOCAL_DTBO"
            ok "  Compiled local camera overlay"
        else
            warn "  Camera overlay DTS not found: $LOCAL_DTS"
            CAMERA_SRC=""
        fi
    fi

    if [[ -n "$CAMERA_SRC" ]]; then
        if [[ ! -f "$CAMERA_BOOT" ]]; then
            cp "$CAMERA_SRC" "$CAMERA_BOOT"
        fi
        if [[ -f "$BOOT_DTBO_DIR/managed.list" ]]; then
            grep -qF "$CAMERA_DTBO" "$BOOT_DTBO_DIR/managed.list" || \
                echo "$CAMERA_DTBO" >> "$BOOT_DTBO_DIR/managed.list"
        else
            echo "$CAMERA_DTBO" > "$BOOT_DTBO_DIR/managed.list"
        fi
        ok "Camera overlay enabled: $CAMERA_DTBO"
        info "After reboot, verify with: v4l2-ctl --device /dev/video11 --list-formats-ext"
    fi

    # ── FIQ debugger overlay: re-enable UART2 kernel console on J6 ─────────
    #
    # The base DTB was patched to disable the FIQ debugger (BASE_DTB_PATCH.md).
    # This overlay restores it so J6 (UART2-M0) provides a kernel-level serial
    # console at 1.5 Mbaud — accessible even during kernel panics.
    # UART2 is not used by any Smart Eye peripheral (sensors are on UART3/UART6).

    info "Setting up FIQ debugger overlay (UART2 / J6)..."

    FIQ_DTS="$PROJ/Overlays/fiq-debugger-uart2.dts"
    FIQ_DTBO_NAME="fiq-debugger-uart2.dtbo"
    FIQ_DTBO="$PROJ/Overlays/$FIQ_DTBO_NAME"
    FIQ_BOOT="$BOOT_DTBO_DIR/$FIQ_DTBO_NAME"

    if [[ -f "$FIQ_DTS" ]]; then
        dtc -@ -I dts -O dtb -o "$FIQ_DTBO" "$FIQ_DTS" 2>&1 | tee -a "$LOG"

        if [[ ! -f "$FIQ_BOOT" ]]; then
            cp "$FIQ_DTBO" "$FIQ_BOOT"
        else
            # Replace if source is newer
            [[ "$FIQ_DTS" -nt "$FIQ_BOOT" ]] && cp "$FIQ_DTBO" "$FIQ_BOOT"
        fi

        if [[ -f "$BOOT_DTBO_DIR/managed.list" ]]; then
            grep -qF "$FIQ_DTBO_NAME" "$BOOT_DTBO_DIR/managed.list" || \
                echo "$FIQ_DTBO_NAME" >> "$BOOT_DTBO_DIR/managed.list"
        else
            echo "$FIQ_DTBO_NAME" > "$BOOT_DTBO_DIR/managed.list"
        fi
        ok "FIQ debugger overlay enabled: $FIQ_DTBO_NAME"
        info "After reboot, connect USB-UART to J6 at 1500000 baud (type 'h' for commands)"
    else
        warn "FIQ debugger DTS not found: $FIQ_DTS — skipping"
    fi

    # Print final managed.list so it's visible in the log
    if [[ -f "$BOOT_DTBO_DIR/managed.list" ]]; then
        info "Active overlays in $BOOT_DTBO_DIR/managed.list:"
        while IFS= read -r line; do
            info "  $line"
        done < "$BOOT_DTBO_DIR/managed.list"
    fi
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
    SRC_CPU_SVC="$PROJ/services/smart-eye-power.service"

    [[ -f "$SRC_CPU_SVC" ]] || die "Not found: $SRC_CPU_SVC"

    if [[ ! -f "$CPU_SVC" ]] || ! cmp -s "$SRC_CPU_SVC" "$CPU_SVC"; then
        install -m 0644 "$SRC_CPU_SVC" "$CPU_SVC"
        systemctl daemon-reload
        ok "smart-eye-power.service installed from $SRC_CPU_SVC"
    else
        ok "smart-eye-power.service already up to date"
    fi
    systemctl enable smart-eye-power.service

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
    # Without this fix U-Boot leaves the pin floating and the motor buzzes for
    # several seconds before Linux takes control.
    #
    # fw_env.config must point to the ACTUAL U-Boot environment partition.
    # The Radxa CM5 uses a GPT layout — the env lives in the partition whose
    # PARTLABEL matches "uboot-env" or "env", NOT at fixed byte offsets.
    # Hardcoded offsets (e.g. 0x3F8000) cause "Configuration file corrupted".
    #
    # Detection order:
    #   1. GPT PARTLABEL containing "env" on mmcblk0 / mmcblk1
    #   2. Raw eMMC scan for known U-Boot env variable strings
    #   3. Fallback: print manual U-Boot console instructions and continue
    #
    # NOTE: The DTS pull-down on GPIO0_C5 (overlay phase 3) and the initramfs
    # hook (phase 6) together suppress the motor from kernel init onward.
    # The U-Boot fix closes the remaining gap before the kernel loads (~1-2 s).

    FWENV=/etc/fw_env.config
    GPIO_CMD="gpio clear 21"

    # ── Helper: write fw_env.config and verify with fw_printenv ─────────────
    try_fwenv() {
        local cfg_line="$1"
        local desc="$2"
        info "Trying fw_env.config: $desc"
        printf '%s\n' "# Smart Eye — $desc" "$cfg_line" > "$FWENV"
        # fw_printenv exits non-zero and prints "corrupted" if offsets are wrong
        if fw_printenv 2>&1 | grep -qi "corrupted\|wrong\|error"; then
            warn "  Offsets not valid: $desc"
            return 1
        fi
        # A valid env always contains bootcmd or some known variable
        if fw_printenv 2>/dev/null | grep -qE "^(bootcmd|preboot|bootargs)="; then
            ok "  Valid U-Boot environment found: $desc"
            return 0
        fi
        warn "  No bootcmd/preboot found: $desc"
        return 1
    }

    # ── Method 1: GPT partition label ───────────────────────────────────────
    ENV_PART=""
    for disk in /dev/mmcblk0 /dev/mmcblk1; do
        [[ -b "$disk" ]] || continue
        # lsblk -P gives KEY="VALUE" pairs, safe for parsing
        while IFS= read -r line; do
            eval "$line" 2>/dev/null || true
            # shellcheck disable=SC2154
            label_lc="${PARTLABEL,,}"   # lowercase
            if [[ "$label_lc" == *"env"* || "$label_lc" == *"uboot"* ]]; then
                ENV_PART="/dev/${NAME}"
                break 2
            fi
        done < <(lsblk -P -o NAME,PARTLABEL "$disk" 2>/dev/null | tail -n +2)
    done

    if [[ -n "$ENV_PART" ]] && [[ -b "$ENV_PART" ]]; then
        SIZE=$(blockdev --getsize64 "$ENV_PART" 2>/dev/null || echo "32768")
        info "Found env partition: $ENV_PART  (${SIZE} bytes)"
        ENV_FOUND=0
        try_fwenv "${ENV_PART}    0x0    ${SIZE}" "GPT partition $ENV_PART" && ENV_FOUND=1
    else
        info "No GPT partition with 'env' label found — trying known Rockchip offsets"
        ENV_FOUND=0
    fi

    # ── Method 2: Known Rockchip / Radxa BSP offsets ────────────────────────
    # Different Radxa image versions place the env at different raw offsets.
    # Try each until fw_printenv succeeds.
    if [[ "${ENV_FOUND:-0}" -eq 0 ]]; then
        declare -a CANDIDATES=(
            # Radxa CM5 Debian BSP (rk3588-u-boot 2017.09 / 2022.10)
            "/dev/mmcblk0    0x7F8000    0x8000 | Radxa-BSP-v1 offset"
            "/dev/mmcblk0    0x3F8000    0x8000 | Radxa-BSP-v2 offset"
            # Armbian Rockchip offsets
            "/dev/mmcblk0    0xFF8000    0x8000 | Armbian offset"
            # Two-copy env (some U-Boot builds)
            "/dev/mmcblk0    0x7F8000    0x8000
/dev/mmcblk0    0x800000    0x8000 | Radxa-BSP-v1 dual-copy"
        )
        for entry in "${CANDIDATES[@]}"; do
            cfg_line="${entry%|*}"
            desc="${entry#*|}"
            try_fwenv "$cfg_line" "$desc" && { ENV_FOUND=1; break; }
        done
    fi

    # ── Method 3: Raw eMMC string scan (last resort, limited to first 64 MB) ─
    if [[ "${ENV_FOUND:-0}" -eq 0 ]] && [[ -b /dev/mmcblk0 ]]; then
        info "Scanning first 64 MB of eMMC for U-Boot env strings (may take ~10 s)..."
        HIT=$(dd if=/dev/mmcblk0 bs=1M count=64 2>/dev/null \
              | strings -t d \
              | grep -E "^[0-9]+ (bootcmd|preboot)=" \
              | head -1 | awk '{print $1}')
        if [[ -n "$HIT" ]]; then
            # Align down to 0x8000 (32 KB) boundary — env is sector-aligned
            ALIGNED=$(( (HIT / 32768) * 32768 ))
            HEX=$(printf "0x%X" "$ALIGNED")
            info "Detected env strings near offset $HEX"
            try_fwenv "/dev/mmcblk0    ${HEX}    0x8000" "raw scan at $HEX" && ENV_FOUND=1
        fi
    fi

    # ── Apply the fix or print manual fallback ───────────────────────────────
    if [[ "${ENV_FOUND:-0}" -eq 1 ]] && command -v fw_setenv &>/dev/null; then
        EXISTING=$(fw_printenv -n preboot 2>/dev/null || true)
        if echo "$EXISTING" | grep -qF "$GPIO_CMD"; then
            ok "U-Boot preboot already contains: $GPIO_CMD"
        else
            if [[ -n "$EXISTING" ]]; then
                fw_setenv preboot "${GPIO_CMD}; ${EXISTING}"
            else
                fw_setenv preboot "${GPIO_CMD}"
            fi
            ok "U-Boot preboot → \"${GPIO_CMD}\" (GPIO0_C5 LOW before kernel loads)"
        fi
    else
        rm -f "$FWENV"   # remove invalid config so fw_* tools don't fail later
        warn "U-Boot environment auto-detection failed."
        warn "The DTS pull-down + initramfs hook (phases 3 & 6) already"
        warn "suppress the motor from kernel init onward (~1-2 s after power-on)."
        warn ""
        warn "To close the remaining U-Boot window manually:"
        warn "  1. Connect a USB-UART adapter to the debug port (J6)"
        warn "  2. Power on and press any key within 2 s to stop autoboot"
        warn "  3. At the => prompt:"
        warn "       => setenv preboot \"gpio clear 21\""
        warn "       => saveenv"
        warn "       => boot"
        warn ""
        warn "To identify your env offset:"
        warn "  sudo strings -t x /dev/mmcblk0 | grep 'bootcmd='"
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
#  Phase 9 — USB network (usb0 static IP + no-sleep)
# ---------------------------------------------------------------------------
if ! skip_phase 9; then
    hdr 9 "USB network (usb0 static IP + no-sleep)"

    # The Radxa CM5 exposes a USB NCM gadget (usb0) that provides SSH access
    # during development.  We configure a static IP so the device is always
    # reachable at 192.168.2.100 regardless of host DHCP state.
    # Sleep/suspend targets are also masked here — Smart Eye must never
    # auto-suspend (it needs instant response to the user at all times).

    USB_IF="usb0"
    STATIC_IP="192.168.2.100"
    PREFIX="24"
    GATEWAY="192.168.2.1"
    DNS="8.8.8.8"
    UDC="fc000000.usb"
    NCM_SVC="radxa-ncm@${UDC}.service"
    NET_SVC="smart-eye-usb-net.service"
    NET_SVC_PATH="/etc/systemd/system/$NET_SVC"

    # ── 9a: Enable radxa-ncm USB gadget ─────────────────────────────────────
    if systemctl list-unit-files "$NCM_SVC" 2>/dev/null | grep -q "$NCM_SVC"; then
        systemctl enable "$NCM_SVC" 2>/dev/null || true
        ok "USB NCM gadget service enabled: $NCM_SVC"
    else
        warn "radxa-ncm service not found — USB gadget may need manual activation"
        warn "Check: systemctl list-unit-files | grep ncm"
    fi

    # ── 9b: Tell NetworkManager to never touch usb0 ─────────────────────────
    mkdir -p /etc/NetworkManager/conf.d
    cat > /etc/NetworkManager/conf.d/20-usb0-unmanaged.conf <<EOF
[keyfile]
unmanaged-devices=interface-name:${USB_IF}
EOF

    # Prevent NM from overwriting /etc/resolv.conf
    cat > /etc/NetworkManager/conf.d/10-dns-none.conf <<EOF
[main]
dns=none
EOF
    nmcli connection reload 2>/dev/null || true
    ok "NetworkManager: usb0 unmanaged, DNS=none"

    # ── 9c: Create static-IP systemd service ────────────────────────────────
    if [[ -f "$NET_SVC_PATH" ]] && grep -q "STATIC_IP=$STATIC_IP" "$NET_SVC_PATH" 2>/dev/null; then
        ok "$NET_SVC already configured for $STATIC_IP"
    else
        info "Writing $NET_SVC_PATH..."
        cat > "$NET_SVC_PATH" <<EOF
[Unit]
Description=Smart Eye USB network static IP (${USB_IF} ${STATIC_IP}/${PREFIX})
After=${NCM_SVC} network-pre.target
Wants=${NCM_SVC}
Before=network.target
DefaultDependencies=no
# Variables embedded at setup time (re-run phase 9 to change IP)
# STATIC_IP=${STATIC_IP}  GATEWAY=${GATEWAY}  DNS=${DNS}

[Service]
Type=oneshot
RemainAfterExit=yes

# Wait up to 10 s for usb0 to appear after NCM gadget activates
ExecStartPre=/bin/sh -c '\
    i=0; \
    until ip link show ${USB_IF} >/dev/null 2>&1 || [ \$i -ge 20 ]; do \
        sleep 0.5; i=\$((i+1)); \
    done'

ExecStart=/bin/sh -c '\
    ip link set ${USB_IF} up && \
    ip addr flush dev ${USB_IF} && \
    ip addr add ${STATIC_IP}/${PREFIX} dev ${USB_IF} && \
    ip route replace default via ${GATEWAY} dev ${USB_IF}'

ExecStartPost=/bin/sh -c 'echo "nameserver ${DNS}" > /etc/resolv.conf'

ExecStop=/bin/sh -c '\
    ip addr flush dev ${USB_IF} 2>/dev/null || true; \
    ip route del default via ${GATEWAY} dev ${USB_IF} 2>/dev/null || true'

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
        systemctl daemon-reload
        systemctl enable "$NET_SVC"
        ok "$NET_SVC created and enabled"
    fi

    # ── 9d: Write /etc/resolv.conf (persistent DNS) ─────────────────────────
    if grep -q "^nameserver $DNS" /etc/resolv.conf 2>/dev/null; then
        ok "/etc/resolv.conf already has $DNS"
    else
        # Remove immutable flag if set by a previous run before overwriting
        chattr -i /etc/resolv.conf 2>/dev/null || true
        printf "nameserver %s\n" "$DNS" > /etc/resolv.conf
        chattr +i /etc/resolv.conf 2>/dev/null && \
            ok "/etc/resolv.conf written and locked (immutable)" || \
            ok "/etc/resolv.conf written"
    fi

    # ── 9e: Disable sleep / suspend / hibernate ─────────────────────────────
    # Smart Eye must never auto-suspend — the user depends on instant response.
    mkdir -p /etc/systemd/logind.conf.d
    cat > /etc/systemd/logind.conf.d/nosleep.conf <<'EOF'
[Login]
HandlePowerKey=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
IdleActionSec=0
EOF

    mkdir -p /etc/systemd/sleep.conf.d
    cat > /etc/systemd/sleep.conf.d/nosleep.conf <<'EOF'
[Sleep]
AllowSuspend=no
AllowHibernation=no
AllowSuspendThenHibernate=no
AllowHybridSleep=no
EOF

    for target in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
        systemctl mask "$target" 2>/dev/null || true
    done

    systemctl daemon-reload
    ok "Sleep / suspend / hibernate masked"

    # ── 9f: Apply right now (no reboot needed for network) ──────────────────
    systemctl start "$NET_SVC" 2>/dev/null && ok "$NET_SVC started immediately" || \
        info "$NET_SVC queued (usb0 not yet up — will apply on reboot)"

    # Quick connectivity check
    sleep 1
    CURRENT_IP=$(ip -4 addr show "$USB_IF" 2>/dev/null | awk '/inet /{print $2}' | head -1)
    if [[ -n "$CURRENT_IP" ]]; then
        ok "usb0 IP: $CURRENT_IP  (SSH: ssh radxa@$STATIC_IP)"
    else
        info "usb0 not yet up — after reboot: ssh radxa@$STATIC_IP"
    fi
fi

# ---------------------------------------------------------------------------
#  Phase 10 — Python 3.11 venv + pip packages + RKNN Lite wheel
# ---------------------------------------------------------------------------
if ! skip_phase 10; then
    hdr 10 "Python 3.11 venv + packages"

    VENV="$PROJ/venv"
    # Offline: keep pip cache so previously-downloaded wheels can be reused.
    # Online: disable cache to avoid stale wheel collisions.
    if [[ "$OFFLINE" -eq 1 ]]; then
        PIP_OPTS="--quiet"
    else
        PIP_OPTS="--no-cache-dir --quiet"
    fi

    # ── 9a: Verify Python 3.11 ──────────────────────────────────────────────
    info "[9a] Checking Python 3.11..."

    if ! command -v python3.11 &>/dev/null; then
        info "python3.11 not in PATH — trying apt install..."
        apt-get install -y python3.11 python3.11-venv python3.11-dev 2>&1 | tee -a "$LOG"
    fi

    command -v python3.11 &>/dev/null || die "python3.11 not available after install — check your OS image"

    PY_VER=$(python3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    PY_ABI=$(python3.11 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    ok "Python $PY_VER  ABI: $PY_ABI  arch: $(uname -m)"

    # The bundled RKNN wheel is cp311 — guard against running on a wrong Python
    RKNN_WHL=$(ls "$PROJ"/rknn_toolkit_lite2-*.whl 2>/dev/null | head -1)
    if [[ -n "$RKNN_WHL" ]]; then
        WHL_ABI=$(basename "$RKNN_WHL" | grep -oP 'cp\d+' | head -1)
        if [[ "$PY_ABI" != "$WHL_ABI" ]]; then
            die "RKNN wheel requires ${WHL_ABI} but running ${PY_ABI}. Wheel: $(basename "$RKNN_WHL")"
        fi
        ok "RKNN wheel ABI match: $WHL_ABI"
    else
        warn "No rknn_toolkit_lite2-*.whl found in $PROJ"
    fi

    # ── 9b: Create venv ─────────────────────────────────────────────────────
    info "[9b] Creating Python venv..."

    if [[ -f "$VENV/bin/activate" ]]; then
        EXISTING_VER=$("$VENV/bin/python3" --version 2>&1)
        ok "venv already exists ($EXISTING_VER)"
    else
        python3.11 -m venv "$VENV" --system-site-packages
        ok "venv created: $VENV"
    fi

    PIP="$VENV/bin/pip"
    PYBIN="$VENV/bin/python3"

    # ── 9c: Bootstrap pip, setuptools, wheel ────────────────────────────────
    info "[9c] Upgrading pip / setuptools / wheel..."
    if [[ "$OFFLINE" -eq 1 ]]; then
        info "Offline: skipping pip/setuptools/wheel upgrade"
    else
        "$PIP" install --upgrade pip setuptools wheel $PIP_OPTS 2>&1 | tee -a "$LOG"
    fi
    PIP_VER=$("$PIP" --version | awk '{print $2}')
    ok "pip $PIP_VER ready"

    # ── 9d: Install core runtime packages first ──────────────────────────────
    # Installing these before the full requirements.txt gives better error
    # messages if a critical package fails (e.g. numpy build on bare system).
    info "[9d] Installing core runtime packages..."

    CORE_PKGS=(
        "numpy==1.26.4"
        "pillow==12.1.1"
        "opencv-python-headless==4.11.0.86"
        "pyserial==3.5"
        "psutil==5.9.4"
        "onnxruntime==1.19.2"
    )

    for pkg in "${CORE_PKGS[@]}"; do
        pkg_name="${pkg%%==*}"
        if "$PIP" show "$pkg_name" &>/dev/null; then
            ok "  $pkg_name already installed"
        elif [[ "$OFFLINE" -eq 1 ]]; then
            warn "  Offline: $pkg_name not installed — needs internet"
        else
            info "  Installing $pkg..."
            "$PIP" install "$pkg" $PIP_OPTS 2>&1 | tee -a "$LOG" || \
                warn "  Failed to install $pkg — will retry via requirements.txt"
        fi
    done

    # ── 9e: Install full requirements.txt ───────────────────────────────────
    info "[9e] Installing requirements.txt (this can take several minutes)..."
    SPACE_BEFORE=$(df -BM "$PROJ" | awk 'NR==2{print $3}')

    if [[ "$OFFLINE" -eq 1 ]]; then
        info "Offline: installing from pip cache only (previously downloaded wheels)..."
        "$PIP" install \
            -r "$PROJ/requirements.txt" \
            $PIP_OPTS \
            --ignore-installed \
            2>&1 | tee -a "$LOG" || \
            warn "Some packages could not be installed offline — run with internet to complete"
    else
        # Use --ignore-installed to avoid conflicts with system-site-packages
        "$PIP" install \
            -r "$PROJ/requirements.txt" \
            $PIP_OPTS \
            --ignore-installed \
            2>&1 | tee -a "$LOG"
    fi

    SPACE_AFTER=$(df -BM "$PROJ" | awk 'NR==2{print $3}')
    ok "requirements.txt done  (disk: ${SPACE_BEFORE} → ${SPACE_AFTER})"

    # ── 9f: Install RKNN Lite2 wheel ────────────────────────────────────────
    info "[9f] Installing RKNN Lite2 wheel..."

    if [[ -n "$RKNN_WHL" ]]; then
        WHL_NAME=$(basename "$RKNN_WHL")
        INSTALLED_VER=$("$PIP" show rknn-toolkit-lite2 2>/dev/null | awk '/^Version/{print $2}')
        WHL_VER=$(echo "$WHL_NAME" | grep -oP '\d+\.\d+\.\d+' | head -1)

        if [[ "$INSTALLED_VER" == "$WHL_VER" ]]; then
            ok "RKNN Lite2 $INSTALLED_VER already installed"
        else
            [[ -n "$INSTALLED_VER" ]] && info "Replacing existing RKNN Lite2 $INSTALLED_VER with $WHL_VER"
            "$PIP" install "$RKNN_WHL" --force-reinstall $PIP_OPTS 2>&1 | tee -a "$LOG"
            ok "RKNN Lite2 $WHL_VER installed from $WHL_NAME"
        fi
    else
        warn "No RKNN wheel found — NPU inference will not work"
        warn "Copy the .whl to $PROJ/ then re-run:  sudo bash scripts/setup_device.sh --phase=9"
    fi

    # ── 9g: Install rapidocr (OCR engine for text reading) ──────────────────
    info "[9g] Verifying rapidocr-onnxruntime..."
    if ! "$PIP" show rapidocr-onnxruntime &>/dev/null; then
        if [[ "$OFFLINE" -eq 1 ]]; then
            warn "Offline: rapidocr-onnxruntime not installed — needs internet"
        else
            "$PIP" install "rapidocr-onnxruntime==1.4.4" $PIP_OPTS 2>&1 | tee -a "$LOG"
        fi
    fi
    ok "rapidocr-onnxruntime present"

    # ── 9h: Verify critical imports load correctly ───────────────────────────
    info "[9h] Verifying critical Python imports..."

    IMPORT_ERRORS=0
    verify_import() {
        local module="$1"
        local label="${2:-$1}"
        if "$PYBIN" -c "import $module" 2>/dev/null; then
            ok "  import $label"
        else
            err "  import $label  FAILED"
            "$PYBIN" -c "import $module" 2>&1 | tee -a "$LOG" || true
            ((IMPORT_ERRORS++)) || true
        fi
    }

    verify_import "cv2"                "cv2 (OpenCV)"
    verify_import "numpy"              "numpy"
    verify_import "PIL"                "PIL (Pillow)"
    verify_import "serial"             "serial (pyserial)"
    verify_import "onnxruntime"        "onnxruntime"
    verify_import "rapidocr_onnxruntime" "rapidocr_onnxruntime"
    verify_import "argostranslate"     "argostranslate"
    verify_import "psutil"             "psutil"

    # RKNN Lite2 — only present on aarch64 with the NPU driver loaded
    if "$PYBIN" -c "from rknnlite.api import RKNNLite" 2>/dev/null; then
        ok "  import rknnlite (RKNN Lite2)"
    else
        warn "  rknnlite import failed — NPU driver may not be loaded yet (expected before first reboot)"
    fi

    if [[ "$IMPORT_ERRORS" -gt 0 ]]; then
        warn "$IMPORT_ERRORS import(s) failed — check $LOG for details"
    else
        ok "All critical imports verified"
    fi

    # ── 9i: Print installed package summary ─────────────────────────────────
    info "[9i] Key package versions:"
    for pkg in numpy opencv-python-headless onnxruntime rapidocr-onnxruntime \
                argostranslate pyserial psutil rknn-toolkit-lite2; do
        ver=$("$PIP" show "$pkg" 2>/dev/null | awk '/^Version/{print $2}')
        if [[ -n "$ver" ]]; then
            printf '    %-35s %s\n' "$pkg" "$ver" | tee -a "$LOG"
        fi
    done

    # ── 9j: Create venv activation helper ───────────────────────────────────
    HELPER="$PROJ/activate_env.sh"
    cat > "$HELPER" <<EOF
#!/bin/bash
# Source this file to activate the Smart Eye Python environment:
#   source activate_env.sh
#
# Or run the app directly (must be root):
#   sudo bash activate_env.sh python3 pathpal_project/main.py

if [[ "\${BASH_SOURCE[0]}" == "\${0}" ]]; then
    # Executed directly — pass args to python3 inside the venv
    exec sudo -E ${VENV}/bin/python3 "\$@"
else
    # Sourced — activate the venv in the current shell
    source "${VENV}/bin/activate"
    echo "Smart Eye venv active  (Python \$(python3 --version))"
    echo "Run app: sudo -E python3 pathpal_project/main.py"
fi
EOF
    chmod 0755 "$HELPER"
    ok "Activation helper: $HELPER"

    # ── 9k: Fix ownership ───────────────────────────────────────────────────
    chown -R "${PROJ_USER}:${PROJ_USER}" "$VENV" "$HELPER"
    ok "venv owned by $PROJ_USER"

    echo ""
    ok "Python environment ready: $VENV"
    info "To use manually:  source $HELPER   (or: sudo -E $VENV/bin/python3 ...)"
fi

# ---------------------------------------------------------------------------
#  Phase 11 — TTS engines
# ---------------------------------------------------------------------------
if ! skip_phase 11; then
    hdr 11 "TTS engines (espeak-ng + Piper)"

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
#  Phase 12 — Argostranslate language packages
# ---------------------------------------------------------------------------
if ! skip_phase 12; then
    hdr 12 "Argostranslate (en↔hi offline translation)"

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
#  Phase 13 — Smart Eye main systemd service
# ---------------------------------------------------------------------------
if ! skip_phase 13; then
    hdr 13 "Smart Eye main systemd service"

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
#  Phase 14 — First-boot power system config (BQ25895 + BQ27220)
# ---------------------------------------------------------------------------
if ! skip_phase 14; then
    hdr 14 "First-boot power system config"

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
# One-shot: configure BQ25895 charger + BQ27220 fuel gauge on first boot.
# Runs after the carrier overlay is active so I2C bus 3 exposes both ICs.
LOG=/var/log/smart-eye-setup.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "\$(ts)  [FIRST-BOOT] === Power system first-boot configuration ===" >> "\$LOG"

# ── Verify both ICs are visible on I2C bus 3 ──────────────────────────────
if ! i2cdetect -y 3 2>/dev/null | grep -q "6a"; then
    echo "\$(ts)  [FIRST-BOOT] ERROR: BQ25895 (0x6A) not found on i2c-3 — carrier overlay may not be active" >> "\$LOG"
    exit 1
fi
echo "\$(ts)  [FIRST-BOOT] BQ25895 found at i2c-3/0x6A" >> "\$LOG"

if ! i2cdetect -y 3 2>/dev/null | grep -q "55"; then
    echo "\$(ts)  [FIRST-BOOT] ERROR: BQ27220 (0x55) not found on i2c-3 — carrier overlay may not be active" >> "\$LOG"
    exit 1
fi
echo "\$(ts)  [FIRST-BOOT] BQ27220 found at i2c-3/0x55" >> "\$LOG"

# ── Configure BQ25895 charger registers ───────────────────────────────────
# REG07 = 0x8F  WDT=OFF, EN_TERM=ON, CHG_TIMER=20h
#   Disabling watchdog is critical: without it the IC resets charge params
#   every 40 s, which can cause SYS voltage spikes under load.
echo "\$(ts)  [FIRST-BOOT] Configuring BQ25895..." >> "\$LOG"
i2cset -y 3 0x6a 0x07 0x8f b 2>&1 | tee -a "\$LOG"

# REG09 = 0x4C  BATFET_DIS=0 (ON), BATFET_DLY=10s
#   Ensures SYS rail is powered from battery; BATFET stays ON until an
#   explicit ship-mode command (BATFET_DIS=1 written at poweroff).
i2cset -y 3 0x6a 0x09 0x4c b 2>&1 | tee -a "\$LOG"

# Verify key registers
REG07=\$(i2cget -y 3 0x6a 0x07 b 2>/dev/null || echo "ERR")
REG09=\$(i2cget -y 3 0x6a 0x09 b 2>/dev/null || echo "ERR")
echo "\$(ts)  [FIRST-BOOT] BQ25895 REG07=\${REG07} (want 0x8f)  REG09=\${REG09} (want 0x4c)" >> "\$LOG"

# ── Configure BQ27220 fuel gauge for 10000mAh pack ────────────────────────
# Sets FullChargeCapacity and DesignCapacity so SOC/mAh readings scale
# correctly to the 10000mAh pack on the Smart Eye carrier board.
echo "\$(ts)  [FIRST-BOOT] Configuring BQ27220 via tests/config_fuel_gauge.py..." >> "\$LOG"

cd ${PROJ}
${PROJ}/venv/bin/python3 tests/config_fuel_gauge.py 2>&1 | tee -a "\$LOG"
FG_STATUS=\${PIPESTATUS[0]}

if [[ \$FG_STATUS -eq 0 ]]; then
    echo "\$(ts)  [FIRST-BOOT] BQ27220 configured OK" >> "\$LOG"
else
    echo "\$(ts)  [FIRST-BOOT] ERROR: config_fuel_gauge.py exited \$FG_STATUS" >> "\$LOG"
    exit \$FG_STATUS
fi

echo "\$(ts)  [FIRST-BOOT] === Power system configuration complete ===" >> "\$LOG"
EOF
    chmod 0755 "$FIRST_BOOT_SH"

    # Install the one-shot service from the repo source file
    SRC_FIRST_BOOT_SVC="$PROJ/services/smart-eye-first-boot.service"
    [[ -f "$SRC_FIRST_BOOT_SVC" ]] || die "Not found: $SRC_FIRST_BOOT_SVC"
    install -m 0644 "$SRC_FIRST_BOOT_SVC" "$FIRST_BOOT_SVC"

    # Create sentinel so the service fires on the first boot after reboot
    touch "$SENTINEL"

    systemctl daemon-reload
    systemctl enable smart-eye-first-boot.service
    ok "smart-eye-first-boot.service enabled (runs once after first reboot)"
    ok "Sentinel: $SENTINEL"
fi

# ---------------------------------------------------------------------------
#  Phase 15 — Final verification
# ---------------------------------------------------------------------------
if ! skip_phase 15; then
    hdr 15 "Final verification"

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
    chk "Carrier overlay compiled"       "test -f $PROJ/Overlays/smart-eye-carrier.dtbo"
    chk "Carrier overlay in /boot"       "test -f /boot/dtbo/smart-eye-carrier.dtbo"
    chk "Camera overlay (IMX219) DTS"    "test -f $PROJ/Overlays/imx219-cam0.dts"
    chk "Camera overlay in /boot"        "ls /boot/dtbo/*imx219*.dtbo"
    chk "FIQ debugger overlay DTS"       "test -f $PROJ/Overlays/fiq-debugger-uart2.dts"
    chk "FIQ debugger overlay in /boot"  "test -f /boot/dtbo/fiq-debugger-uart2.dtbo"

    echo ""
    echo -e "${BLD}Services:${RST}"
    chk "vibration-motor-init.service"   "systemctl is-enabled vibration-motor-init.service"
    chk "ship-mode-shutdown.service"     "systemctl is-enabled ship-mode-shutdown.service"
    chk "smart-eye-power.service"        "systemctl is-enabled smart-eye-power.service"
    chk "smart-eye-usb-net.service"      "systemctl is-enabled smart-eye-usb-net.service"
    chk "smart-eye-first-boot.service"   "systemctl is-enabled smart-eye-first-boot.service"
    chk "smart-eye.service"              "systemctl is-enabled smart-eye.service"

    echo ""
    echo -e "${BLD}Files:${RST}"
    chk "initramfs vibration hook"       "test -x /etc/initramfs-tools/scripts/init-bottom/smart-eye-vibration"
    chk "vibration-motor-init.sh"        "test -x /usr/local/bin/vibration-motor-init.sh"
    chk "shutdown_ship_mode.sh"          "test -x /opt/battery-mgr/shutdown_ship_mode.sh"
    chk "first_boot_power_config.sh"     "test -x /opt/battery-mgr/first_boot_power_config.sh"
    chk "ALSA asound.conf"               "grep -q smarteye_loud /etc/asound.conf"
    chk "NM usb0 unmanaged config"       "test -f /etc/NetworkManager/conf.d/20-usb0-unmanaged.conf"
    chk "No-sleep logind config"         "test -f /etc/systemd/logind.conf.d/nosleep.conf"

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
echo -e "  1. All overlays load → I2C3, UARTs, PWM4, I2S1, IMX219 camera, FIQ debugger"
echo -e "  2. smart-eye-usb-net.service → usb0 at 192.168.2.100 (SSH access)"
echo -e "  3. smart-eye-first-boot.service runs first_boot_power_config.sh"
echo -e "     → BQ25895: REG07=0x8F (WDT off), REG09=0x4C (BATFET ON)"
echo -e "     → BQ27220: tests/config_fuel_gauge.py sets 10000mAh capacity"
echo -e "  4. smart-eye-power.service sets CPU governors + disables HDMI"
echo -e "  5. vibration-motor-init.service hands GPIO0_C5 to PWM4 driver"
echo -e "  6. smart-eye.service starts the main app (auto on every boot)"
echo ""
echo -e "After reboot, verify with:"
echo -e "  ssh radxa@192.168.2.100                      # USB serial console"
echo -e "  sudo journalctl -u smart-eye -f              # app logs"
echo -e "  sudo journalctl -u smart-eye-first-boot      # power config log"
echo -e "  v4l2-ctl --device /dev/video11 --list-formats-ext  # camera"
echo -e "  sudo python3 ${PROJ}/tests/test_all.py"
echo -e "  sudo python3 ${PROJ}/tests/battery_test.py"
echo ""
echo -e "FIQ debugger (kernel serial console on J6):"
echo -e "  Connect USB-UART adapter at 1500000 baud, type 'h' for commands"
echo ""
echo -e "Manual fuel gauge check (after reboot):"
echo -e "  sudo python3 ${PROJ}/tests/config_fuel_gauge.py"
echo ""

log "Setup complete. Rebooting..."

if [[ "$NO_REBOOT" -eq 1 ]]; then
    warn "--no-reboot set. Reboot manually: sudo reboot"
else
    echo -e "${YLW}Rebooting in 5 seconds — press Ctrl+C to cancel${RST}"
    sleep 5
    reboot
fi
