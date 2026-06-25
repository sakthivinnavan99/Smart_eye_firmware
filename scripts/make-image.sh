#!/bin/bash
# make-image.sh — Clone the running Smart Eye device into a flashable .img.gz
#
# Runs ON THE BOARD (NVMe-only CM5 Lite, drive not removable). It images the live
# root, but the smallest-possible image comes from three steps:
#   1. zero all free space  -> empty blocks compress to nothing
#   2. copy only up to the END of the last partition (not the whole NVMe)
#   3. gzip on the fly while streaming over SSH to a host machine
#
# The result still has full partition size; finish the shrink on the HOST with
# pishrink (prints the exact command at the end), which trims the rootfs to the
# real data and adds a first-boot auto-expand so each clone grows back to its
# full NVMe.
#
# Prompts for: SSH username, host IP, and the destination path on the host.
# Must be run as root (writes nothing to the disk except a temporary zero-fill).

set -euo pipefail

# ---------------------------------------------------------------
#  Must be root — we read a block device and stop a systemd service
# ---------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must run as root.  Try: sudo $0" >&2
    exit 1
fi

# ---------------------------------------------------------------
#  Collect destination details from the operator
# ---------------------------------------------------------------
read -r -p "SSH username on the host machine: " HOST_USER
read -r -p "Host IP address: "                 HOST_IP
read -r -p "Destination path on host (e.g. /home/you/smarteye-master.img.gz): " DEST_PATH

if [ -z "$HOST_USER" ] || [ -z "$HOST_IP" ] || [ -z "$DEST_PATH" ]; then
    echo "Username, host IP and destination path are all required." >&2
    exit 1
fi

# Default the filename if the operator gave a directory
case "$DEST_PATH" in
    */) DEST_PATH="${DEST_PATH}smarteye-master-$(date +%Y%m%d).img.gz" ;;
esac

# Source disk — CM5 Lite boots NVMe only. Override with: DISK=/dev/nvmeXn1 sudo ...
DISK="${DISK:-/dev/nvme0n1}"
if [ ! -b "$DISK" ]; then
    echo "Block device $DISK not found.  Set DISK=/dev/... and retry." >&2
    lsblk -dpno NAME,SIZE,MODEL 2>/dev/null || true
    exit 1
fi

echo
echo "Source disk      : $DISK"
echo "Destination      : ${HOST_USER}@${HOST_IP}:${DEST_PATH}"
echo
read -r -p "Proceed? This stops the smart-eye service and reads $DISK. [y/N] " CONFIRM
case "$CONFIRM" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
esac

# ---------------------------------------------------------------
#  1. Quiesce the system so the live image is consistent
# ---------------------------------------------------------------
echo "==> Stopping smart-eye service"
systemctl stop smart-eye 2>/dev/null || true

echo "==> Cleaning per-unit identity (golden master)"
rm -f /etc/ssh/ssh_host_*               2>/dev/null || true
truncate -s 0 /etc/machine-id           2>/dev/null || true
rm -f /var/lib/dbus/machine-id          2>/dev/null || true
journalctl --rotate                     2>/dev/null || true
journalctl --vacuum-time=1s             2>/dev/null || true

# ---------------------------------------------------------------
#  2. Zero free space so empty blocks compress away
# ---------------------------------------------------------------
echo "==> Zeroing free space (makes the image compress) — this can take a while"
ZERO_FILL="/zero.fill"
# dd will fail with ENOSPC when the filesystem fills; that is expected and fine.
dd if=/dev/zero of="$ZERO_FILL" bs=4M status=none 2>/dev/null || true
sync
rm -f "$ZERO_FILL"
sync

# ---------------------------------------------------------------
#  3. Find the end of the last partition so we don't copy the whole NVMe
# ---------------------------------------------------------------
echo "==> Computing last-partition end sector"
# sfdisk dump lines look like:  /dev/nvme0n1p3 : start=  1085440, size=  ... ,
LAST_LINE="$(sfdisk -d "$DISK" | grep -E '^\S+\s*:.*start=' | tail -n1)"
START_SECTOR="$(echo "$LAST_LINE" | sed -n 's/.*start=\s*\([0-9]\+\).*/\1/p')"
SIZE_SECTORS="$(echo "$LAST_LINE" | sed -n 's/.*size=\s*\([0-9]\+\).*/\1/p')"

if [ -z "$START_SECTOR" ] || [ -z "$SIZE_SECTORS" ]; then
    echo "Could not parse partition layout from sfdisk; copying the whole disk instead." >&2
    END_SECTOR=""
else
    END_SECTOR=$(( START_SECTOR + SIZE_SECTORS ))
    echo "    Last partition ends at sector $END_SECTOR (512-byte sectors)"
    echo "    Copying $(( END_SECTOR / 2048 )) MiB of $(( $(blockdev --getsz "$DISK") / 2048 )) MiB total"
fi

# ---------------------------------------------------------------
#  4. Stream: dd -> gzip -> ssh -> file on host
# ---------------------------------------------------------------
echo "==> Streaming image to ${HOST_USER}@${HOST_IP}:${DEST_PATH}"
echo "    (you may be prompted for the host SSH password)"

if [ -n "$END_SECTOR" ]; then
    dd if="$DISK" bs=512 count="$END_SECTOR" conv=fsync status=progress
else
    dd if="$DISK" bs=4M conv=fsync status=progress
fi | gzip -1 | ssh "${HOST_USER}@${HOST_IP}" "cat > '${DEST_PATH}'"

echo
echo "==> Done. Image written to ${HOST_USER}@${HOST_IP}:${DEST_PATH}"
echo
echo "Next, ON THE HOST, shrink it to the real used size + add auto-expand:"
echo "    gunzip '${DEST_PATH}'"
echo "    wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh"
echo "    chmod +x pishrink.sh"
echo "    sudo ./pishrink.sh -z '${DEST_PATH%.gz}' smarteye-clone.img"
echo
echo "Flash smarteye-clone.img.gz onto new units' NVMe (Imager / Etcher / dd)."
echo
echo "Re-enable the app on THIS board with:  sudo systemctl start smart-eye"
