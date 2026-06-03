#!/bin/bash
# setup_usb_network.sh
# Configures usb0 static IP (192.168.2.100/24), DNS (8.8.8.8),
# default route (via 192.168.2.1), and disables all sleep on Radxa CM5.
#
# Strategy:
#   - NM is told to NEVER touch usb0 (unmanaged via conf.d)
#   - A dedicated systemd service runs AFTER radxa-ncm creates usb0
#     and applies the static IP + route using plain `ip` commands
#   - DNS is written directly to /etc/resolv.conf (plain file, not symlink)
#   - Sleep/suspend/hibernate are disabled via drop-ins + masked targets
#
# Run: sudo bash setup_usb_network.sh

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
info() { echo -e "${YELLOW}[--]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

[[ $EUID -ne 0 ]] && fail "Run as root: sudo bash $0"

USB_IF="usb0"
STATIC_IP="192.168.2.100"
PREFIX="24"
GATEWAY="192.168.2.1"
DNS="8.8.8.8"
UDC="fc000000.usb"
NCM_SVC="radxa-ncm@${UDC}.service"
NET_SVC="smart-eye-usb-net.service"

echo ""
echo "=================================================="
echo "  Smart Eye — USB Network + No-Sleep Setup"
echo "=================================================="
echo "  Interface : $USB_IF  ($STATIC_IP/$PREFIX)"
echo "  Gateway   : $GATEWAY"
echo "  DNS       : $DNS"
echo "  Runs after: $NCM_SVC"
echo "=================================================="
echo ""

# ------------------------------------------------------------------
# 1. Ensure radxa-ncm USB gadget service is enabled at boot
# ------------------------------------------------------------------
info "Ensuring $NCM_SVC is enabled..."
systemctl enable "$NCM_SVC"
ok "$NCM_SVC enabled"

# ------------------------------------------------------------------
# 2. Tell NetworkManager to NEVER manage usb0.
#    NM is created before usb0 exists; trying to let NM manage it
#    causes timing races at boot. We own usb0 via our own service.
# ------------------------------------------------------------------
info "Marking $USB_IF as unmanaged in NetworkManager..."

# Remove any NM keyfile we may have left from a previous attempt
rm -f /etc/NetworkManager/system-connections/usb0-static.nmconnection

# Drop-in that explicitly excludes usb0 from NM management
cat > /etc/NetworkManager/conf.d/20-usb0-unmanaged.conf << EOF
[keyfile]
unmanaged-devices=interface-name:$USB_IF
EOF
ok "NM conf.d/20-usb0-unmanaged.conf written"

# Also tell NM not to overwrite /etc/resolv.conf
cat > /etc/NetworkManager/conf.d/10-dns-none.conf << EOF
[main]
dns=none
EOF
ok "NM dns=none written"

nmcli connection reload 2>/dev/null || true

# ------------------------------------------------------------------
# 3. Create the systemd network service
#    Runs after radxa-ncm so usb0 is guaranteed to exist.
#    Uses plain `ip` commands — no race, no NM dependency.
# ------------------------------------------------------------------
info "Creating $NET_SVC..."

cat > /etc/systemd/system/"$NET_SVC" << EOF
[Unit]
Description=Smart Eye USB network static IP ($USB_IF)
Documentation=https://github.com/smart-eye
After=${NCM_SVC} network-pre.target
Wants=${NCM_SVC}
Before=network.target
DefaultDependencies=no

[Service]
Type=oneshot
RemainAfterExit=yes

# Wait up to 10 s for usb0 to appear after the gadget service exits
ExecStartPre=/bin/sh -c 'i=0; until ip link show $USB_IF >/dev/null 2>&1 || [ \$i -ge 20 ]; do sleep 0.5; i=\$((i+1)); done'

ExecStart=/bin/sh -c '\
    ip link set $USB_IF up && \
    ip addr flush dev $USB_IF && \
    ip addr add $STATIC_IP/$PREFIX dev $USB_IF && \
    ip route replace default via $GATEWAY dev $USB_IF && \
    echo "nameserver $DNS" > /etc/resolv.conf'

ExecStop=/bin/sh -c '\
    ip addr flush dev $USB_IF 2>/dev/null || true && \
    ip route del default via $GATEWAY dev $USB_IF 2>/dev/null || true'

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$NET_SVC"
ok "$NET_SVC created and enabled"

# ------------------------------------------------------------------
# 4. Persist DNS in /etc/resolv.conf
#    Written here so it's correct even before the service runs.
# ------------------------------------------------------------------
info "Writing /etc/resolv.conf..."
cat > /etc/resolv.conf << EOF
nameserver $DNS
EOF
# Lock it from being overwritten by DHCP clients or NM
chattr +i /etc/resolv.conf 2>/dev/null && ok "/etc/resolv.conf locked (immutable)" \
  || info "/etc/resolv.conf written (chattr not available)"

# ------------------------------------------------------------------
# 5. Apply the network right now (don't wait for reboot)
# ------------------------------------------------------------------
info "Applying network config now..."
systemctl start "$NET_SVC" && ok "$NET_SVC started" || info "$NET_SVC start failed (will apply on reboot)"

# ------------------------------------------------------------------
# 6. Disable sleep / suspend / hibernate — three layers
# ------------------------------------------------------------------
info "Disabling sleep, suspend, and hibernate..."

mkdir -p /etc/systemd/logind.conf.d
cat > /etc/systemd/logind.conf.d/nosleep.conf << EOF
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
ok "logind.conf.d/nosleep.conf written"

mkdir -p /etc/systemd/sleep.conf.d
cat > /etc/systemd/sleep.conf.d/nosleep.conf << EOF
[Sleep]
AllowSuspend=no
AllowHibernation=no
AllowSuspendThenHibernate=no
AllowHybridSleep=no
EOF
ok "sleep.conf.d/nosleep.conf written"

for target in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    systemctl mask "$target" 2>/dev/null && ok "Masked $target" || info "$target already masked"
done

systemctl daemon-reload
ok "systemd daemon reloaded"

# ------------------------------------------------------------------
# 7. Verification
# ------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  Verification"
echo "=================================================="

sleep 1
CURRENT_IP=$(ip -4 addr show "$USB_IF" 2>/dev/null | awk '/inet /{print $2}' | head -1)
CURRENT_GW=$(ip route show default dev "$USB_IF" 2>/dev/null | awk '{print $3}' | head -1)
CURRENT_DNS=$(grep '^nameserver' /etc/resolv.conf | awk '{print $2}')

info "usb0 IP        : ${CURRENT_IP:-not set}"
info "Default gateway: ${CURRENT_GW:-not set}"
info "DNS nameserver : ${CURRENT_DNS:-not set}"

if [[ "${CURRENT_IP%%/*}" == "$STATIC_IP" ]]; then
    ok "IP matches expected $STATIC_IP"
else
    fail "IP mismatch — expected $STATIC_IP, got ${CURRENT_IP:-none}"
fi

if curl -s --max-time 5 http://1.1.1.1 > /dev/null 2>&1; then
    ok "Internet reachable via $GATEWAY"
else
    info "Internet not reachable (ensure host $GATEWAY has NAT/forwarding enabled)"
fi

echo ""
info "Sleep target states:"
for target in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    MASKED=$(systemctl is-enabled "$target" 2>/dev/null || echo unknown)
    info "  $target -> $MASKED"
done

echo ""
info "Service status:"
systemctl is-active "$NCM_SVC"  && ok "  $NCM_SVC: active" || info "  $NCM_SVC: inactive"
systemctl is-active "$NET_SVC"  && ok "  $NET_SVC: active" || info "  $NET_SVC: inactive"

echo ""
echo "=================================================="
ok "Setup complete. All settings survive reboot."
echo ""
echo "  Reboot test:  sudo reboot"
echo "  Check after:  ip addr show usb0 && ip route"
echo "  Journal:      journalctl -u $NET_SVC"
echo "=================================================="
echo ""
