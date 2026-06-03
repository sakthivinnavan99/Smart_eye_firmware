#!/bin/bash
# Install TTS engines for Smart Eye
# Usage: sudo bash setup_tts.sh
set -e

PROJ="/home/radxa/Smart_eye_firmware"

echo "=== Installing espeak-ng (lightweight fallback TTS) ==="
apt-get install -y espeak-ng

echo ""
echo "=== Downloading Piper TTS (high-quality offline TTS) ==="
PIPER_DIR="$PROJ/piper"
mkdir -p "$PIPER_DIR"

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
else
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
fi

if [ ! -f "$PIPER_DIR/piper/piper" ]; then
    echo "Downloading Piper binary..."
    wget -q "$PIPER_URL" -O /tmp/piper.tar.gz
    tar xzf /tmp/piper.tar.gz -C "$PIPER_DIR"
    rm /tmp/piper.tar.gz
    chmod +x "$PIPER_DIR/piper/piper"
    echo "Piper binary installed at $PIPER_DIR/piper/piper"
else
    echo "Piper binary already exists"
fi

echo ""
echo "=== Downloading voice models ==="
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

for voice in "en/en_US/amy/medium/en_US-amy-medium" "hi/hi_IN/pratham/medium/hi_IN-pratham-medium"; do
    name=$(basename "$voice")
    if [ ! -f "$PIPER_DIR/${name}.onnx" ]; then
        echo "Downloading ${name}..."
        wget -q "${VOICE_BASE}/${voice}.onnx" -O "$PIPER_DIR/${name}.onnx"
        wget -q "${VOICE_BASE}/${voice}.onnx.json" -O "$PIPER_DIR/${name}.onnx.json"
    else
        echo "${name} already exists"
    fi
done

# Fix ownership
chown -R radxa:radxa "$PIPER_DIR"

echo ""
echo "=== Testing espeak-ng ==="
espeak-ng -v en "Test" -w /tmp/tts_test.wav 2>/dev/null && echo "espeak-ng OK" || echo "espeak-ng FAILED"

echo ""
echo "=== Testing Piper ==="
if [ -f "$PIPER_DIR/piper/piper" ] && [ -f "$PIPER_DIR/en_US-amy-medium.onnx" ]; then
    echo "Hello world" | "$PIPER_DIR/piper/piper" --model "$PIPER_DIR/en_US-amy-medium.onnx" --output_file /tmp/tts_test_piper.wav 2>/dev/null && echo "Piper OK" || echo "Piper FAILED"
else
    echo "Piper not fully installed"
fi

echo ""
echo "=== TTS setup complete ==="
