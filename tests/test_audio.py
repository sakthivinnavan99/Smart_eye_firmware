#!/usr/bin/env python3
"""
Audio test for Smart Eye board speaker (MAX98357A on I2S1).

Plays WAV files from ~/Smart_eye_firmware/wav/English through the
Smart-Eye-Audio card and verifies SD_MODE GPIO behavior.

Usage:
    python3 test_audio.py                   # play all files
    python3 test_audio.py device_turned_on  # play matching file
"""

import subprocess, sys, os, glob, time, wave

WAV_DIR = "/home/radxa/Smart_eye_firmware/wav/English"
CARD_NAME = "SmartEyeAudio"
ALSA_DEV = None  # auto-detected
GPIO_DEBUG = "/sys/kernel/debug/gpio"
SDO1_MIXER = ("Transmit SDO1 Source Select", "From PATH0")


def run(cmd, timeout=10):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def find_card():
    """Find the ALSA card number for Smart-Eye-Audio."""
    rc, out, _ = run("cat /proc/asound/cards")
    for line in out.splitlines():
        if CARD_NAME in line:
            num = line.strip().split()[0]
            return int(num)
    return None


def check_sdmode():
    """Read SD_MODE (gpio-131) state: 'hi' or 'lo'."""
    try:
        with open(GPIO_DEBUG) as f:
            for line in f:
                if "gpio-131" in line:
                    return "hi" if " hi" in line else "lo"
    except PermissionError:
        rc, out, _ = run("sudo cat /sys/kernel/debug/gpio 2>/dev/null")
        for line in out.splitlines():
            if "gpio-131" in line:
                return "hi" if " hi" in line else "lo"
    return "unknown"


def wav_info(path):
    """Return a short description of a WAV file."""
    try:
        with wave.open(path, "rb") as w:
            dur = w.getnframes() / w.getframerate()
            return f"{w.getnchannels()}ch {w.getframerate()}Hz {w.getsampwidth()*8}bit {dur:.1f}s"
    except Exception as e:
        return f"error: {e}"


def play(path, card):
    """Play a WAV file and return (success, duration_s)."""
    dev = f"plughw:{card},0"
    t0 = time.time()
    rc, out, err = run(f"aplay -D {dev} '{path}'", timeout=30)
    elapsed = time.time() - t0
    return rc == 0, elapsed


def main():
    print("=" * 50)
    print("  Smart Eye Speaker Test")
    print("=" * 50)

    # 1. Find card
    card = find_card()
    if card is None:
        print("\n[FAIL] Smart-Eye-Audio card not found.")
        print("       Check overlay and I2S1 status.")
        sys.exit(1)
    print(f"\n[OK]  Card {card}: {CARD_NAME}")

    # 2. Ensure SDO1 routing
    rc, _, _ = run(f"amixer -c {card} sset '{SDO1_MIXER[0]}' '{SDO1_MIXER[1]}'")
    if rc == 0:
        print(f"[OK]  SDO1 routed from PATH0")
    else:
        print(f"[WARN] Could not set SDO1 mixer")

    # 3. Gather WAV files
    pattern = os.path.join(WAV_DIR, "*.wav")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\n[FAIL] No WAV files in {WAV_DIR}")
        sys.exit(1)

    # Filter by keyword if provided
    if len(sys.argv) > 1:
        keyword = sys.argv[1].lower()
        files = [f for f in files if keyword in os.path.basename(f).lower()]
        if not files:
            print(f"\n[FAIL] No files matching '{keyword}'")
            sys.exit(1)

    print(f"\n  Found {len(files)} file(s) to play\n")
    print(f"  {'File':<30s} {'Format':<28s} {'Result'}")
    print(f"  {'-'*30} {'-'*28} {'-'*10}")

    passed = 0
    failed = 0

    for path in files:
        name = os.path.basename(path)
        info = wav_info(path)

        sd_before = check_sdmode()
        ok, dur = play(path, card)
        sd_after = check_sdmode()

        if ok and sd_after == "lo":
            status = "PASS"
            passed += 1
        elif ok:
            status = "PLAY OK"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"  {name:<30s} {info:<28s} {status}")

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(files)}")
    print(f"  SD_MODE after all tests: {check_sdmode()}")
    print("=" * 50)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
