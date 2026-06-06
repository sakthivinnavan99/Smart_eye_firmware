# Serial Port Overlay Transfer Guide

## Quick Fix (Serial Connection)

Your Radxa is connected via serial → audio not working → overlay not installed.

---

## Option 1: Transfer via Python Script (Easiest)

### On macOS:

```bash
cd ~/Freelancing/Smart_Eye/Smart_eye_firmware

# Make sure the overlay is compiled
cd Overlays && make && cd ..

# Run transfer script
python3 send_dtbo_serial.py --port /dev/cu.usbserial-0001 --baud 1500000
```

**Expected output:**
```
Opening /dev/cu.usbserial-0001 at 1500000 baud...
Connected!
Sending smart-eye-carrier.dtbo (6338 bytes)...
Encoded as base64 (8450 bytes)
Sending in 76-char chunks...
[==============================] 100% Done!
Verifying...
File received: 6338 bytes ✓
```

### On Radxa (via serial):

```bash
# File should be at ~/smart-eye-carrier.dtbo
ls -la ~/smart-eye-carrier.dtbo

# Install it
sudo cp ~/smart-eye-carrier.dtbo /boot/dtbo/
echo "smart-eye-carrier.dtbo" | sudo tee -a /boot/dtbo/managed.list

# Reboot
sudo reboot
```

---

## Option 2: Manual Base64 Transfer (If Script Fails)

### On macOS:

```bash
# Encode overlay to base64
base64 Overlays/smart-eye-carrier.dtbo > /tmp/overlay.b64

# Count lines (for reference)
wc -l /tmp/overlay.b64

# Show first few lines
head -5 /tmp/overlay.b64

# Copy all to clipboard
base64 Overlays/smart-eye-carrier.dtbo | pbcopy
```

### On Radxa (via serial):

```bash
# Create file and paste base64
cat > /tmp/overlay.b64 << 'EOF'
# PASTE THE BASE64 HERE (Cmd+V on macOS)
# Use Cmd+Shift+V if regular paste doesn't work
# When done, press Enter then Ctrl+D
EOF

# Verify file was received
wc -c /tmp/overlay.b64

# Decode it
base64 -d /tmp/overlay.b64 > ~/smart-eye-carrier.dtbo

# Verify decoded file
ls -la ~/smart-eye-carrier.dtbo
# Should be about 6338 bytes
```

### Install on Radxa:

```bash
# Copy to boot
sudo cp ~/smart-eye-carrier.dtbo /boot/dtbo/

# Enable in managed list
echo "smart-eye-carrier.dtbo" | sudo tee -a /boot/dtbo/managed.list

# Verify
grep smart-eye /boot/dtbo/managed.list

# Reboot
sudo reboot
```

---

## Step 3: Verify After Reboot

### Check audio cards appear:

```bash
aplay -l
```

**Should show:**
```
**** List of PLAYBACK Hardware Devices ****
card 2: rockchipes8316   [rockchip-es8316]
card 3: SmartEyeAudio    [Smart-Eye-Audio]
```

---

## Troubleshooting Transfer

### "base64 command not found"
```
→ On macOS, base64 should be built-in
→ Try: /usr/bin/base64 instead
```

### "File size doesn't match"
```
→ Transfer was incomplete
→ Retry: Paste more slowly or use smaller chunks
→ Or use Python script instead
```

### "Permission denied" when copying to /boot/dtbo
```
→ You need sudo
→ Use: sudo cp ~/smart-eye-carrier.dtbo /boot/dtbo/
```

### "aplay -l still shows no cards after reboot"
```
→ Overlay may not have loaded
→ Run diagnostic: sudo bash audio_diagnostic.sh
→ Check boot config: grep dtbo /boot/extlinux/extlinux.conf
```

---

## Alternative: Using scp (If You Have Network)

If your Radxa has Ethernet or WiFi:

### On macOS:

```bash
scp Overlays/smart-eye-carrier.dtbo radxa@<IP_ADDRESS>:~/
```

### On Radxa:

```bash
sudo cp ~/smart-eye-carrier.dtbo /boot/dtbo/
echo "smart-eye-carrier.dtbo" | sudo tee -a /boot/dtbo/managed.list
sudo reboot
```

---

## Fastest Method: All-in-One

### On Radxa (copy-paste entire block):

```bash
# Step 1: Remove any old overlay
sudo rm -f /boot/dtbo/smart-eye-carrier.dtbo

# Step 2: Create new base64 file
cat > /tmp/overlay.b64 << 'EOF'
[PASTE BASE64 HERE]
EOF

# Step 3: Decode and install
base64 -d /tmp/overlay.b64 | sudo tee /boot/dtbo/smart-eye-carrier.dtbo > /dev/null

# Step 4: Enable
echo "smart-eye-carrier.dtbo" | sudo tee -a /boot/dtbo/managed.list

# Step 5: Verify and reboot
ls -la /boot/dtbo/smart-eye-carrier.dtbo
echo "Rebooting..."
sudo reboot
```

---

## After Successful Transfer

### Run diagnostic to confirm:

```bash
sudo bash ~/Smart_eye_firmware/audio_diagnostic.sh
```

**Should show:**
```
✓ PASS: Overlay file found
✓ PASS: Overlay enabled in managed.list
✓ PASS: ES8316 found at I2C8
✓ PASS: ES8316 headphone codec registered
✓ PASS: SmartEyeAudio speaker registered
```

---

## Test Audio

### Test Headphone:

```bash
aplay -D plughw:rockchipes8316,0 /usr/share/sounds/alsa/Noise.wav
```

**If you hear noise: ✓ Headphone working!**

### Test Speaker:

```bash
# Enable speaker amp
sudo bash -c 'echo out > /sys/class/gpio/gpio131/direction; echo 0 > /sys/class/gpio/gpio131/value'

# Play sound
aplay -D plughw:SmartEyeAudio,0 /usr/share/sounds/alsa/Noise.wav
```

**If you hear sound from speaker: ✓ Speaker working!**

---

## If Transfer Still Fails

**Get diagnostic output:**

```bash
echo "=== Checking Overlay ===" && \
ls -la /boot/dtbo/smart-eye-carrier.dtbo && \
echo "=== Checking Boot Config ===" && \
grep -i dtbo /boot/extlinux/extlinux.conf && \
echo "=== I2C Check ===" && \
i2cdetect -y 8 && \
echo "=== Sound Cards ===" && \
aplay -l && \
echo "=== Kernel Errors ===" && \
dmesg | tail -30
```

**Share this output for detailed help.**

---

## Serial Connection Troubleshooting

### "Connection timeout"
```
→ Check serial port: ls -la /dev/cu.usbserial-*
→ Try different baud rates: 115200, 921600, 1500000
→ Verify USB cable is secure
```

### "Permission denied"
```
→ macOS may need: sudo chown $USER /dev/cu.usbserial-*
→ Or use screen/minicom with sudo
```

### "Characters corrupted during paste"
```
→ Slow down pasting (copy-paste might be too fast)
→ Or use the Python script which sends in controlled chunks
```

---

## Summary

| Method | Speed | Ease | Works? |
|--------|-------|------|--------|
| Python script | ⭐⭐⭐ | ⭐⭐⭐ | ✓✓✓ |
| Base64 manual | ⭐⭐ | ⭐⭐ | ✓✓ |
| Network scp | ⭐⭐⭐ | ⭐⭐⭐ | ✓✓✓ |

**Recommended: Python script (`send_dtbo_serial.py`) — most reliable for serial.**

---

**Next: Run audio_diagnostic.sh to confirm installation**
