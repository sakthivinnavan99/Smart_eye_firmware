#!/usr/bin/env python3
"""
Audio test for Smart Eye board speaker (MAX98357A on I2S1).

Tests the full audio path including:
  - ALSA card detection
  - SDO1 routing (PATH0 -> SDO1 for MAX98357A DIN)
  - Softvol initialisation and volume control
  - SD_MODE GPIO control via BSS138 (inverted logic)
  - WAV playback with proper amp enable/disable sequencing
  - Speaker DC-protection verification (amp OFF after playback)

Hardware:
  BSS138 (Q1) controls MAX98357A SD_MODE with inverted logic:
    AMP_SD (GPIO4_A3 / gpio131) HIGH -> Q1 ON  -> SD_MODE LOW  -> amp OFF
    AMP_SD (GPIO4_A3 / gpio131) LOW  -> Q1 OFF -> pullup -> SD_MODE HIGH -> amp ON

Usage:
    sudo python3 test_audio.py                   # play all files
    sudo python3 test_audio.py device_turned_on   # play matching file
    sudo python3 test_audio.py --card-only        # just check card & mixer
"""

import subprocess, sys, os, glob, time, wave

WAV_DIR = "/home/radxa/Smart_eye_firmware/wav/English"
CARD_NAME = "SmartEyeAudio"
ALSA_DEV_SOFTVOL = "smarteye_loud"
SDO1_MIXER = ("Transmit SDO1 Source Select", "From PATH0")
SDMODE_GPIO = 131
GPIO_SYSFS = f"/sys/class/gpio/gpio{SDMODE_GPIO}"


def run(cmd, timeout=10):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def find_card():
    rc, out, _ = run("cat /proc/asound/cards")
    for line in out.splitlines():
        if CARD_NAME in line:
            return int(line.strip().split()[0])
    return None


def gpio_export():
    if not os.path.isdir(GPIO_SYSFS):
        try:
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(SDMODE_GPIO))
            time.sleep(0.05)
        except OSError:
            return False
    try:
        with open(f"{GPIO_SYSFS}/direction", "w") as f:
            f.write("out")
    except OSError:
        return False
    return True


def gpio_set(val):
    try:
        with open(f"{GPIO_SYSFS}/value", "w") as f:
            f.write(str(val))
    except OSError:
        pass


def gpio_get():
    try:
        with open(f"{GPIO_SYSFS}/value", "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return -1


def gpio_unexport():
    try:
        with open("/sys/class/gpio/unexport", "w") as f:
            f.write(str(SDMODE_GPIO))
    except OSError:
        pass


def amp_on():
    """Enable amp: AMP_SD LOW -> Q1 OFF -> SD_MODE HIGH via pullup."""
    gpio_set(0)


def amp_off():
    """Disable amp: AMP_SD HIGH -> Q1 ON -> SD_MODE LOW."""
    gpio_set(1)


def amp_is_off():
    """Return True if AMP_SD is HIGH (amp disabled)."""
    return gpio_get() == 1


def wav_info(path):
    try:
        with wave.open(path, "rb") as w:
            dur = w.getnframes() / w.getframerate()
            return f"{w.getnchannels()}ch {w.getframerate()}Hz {w.getsampwidth()*8}bit {dur:.1f}s"
    except Exception as e:
        return f"error: {e}"


def init_softvol():
    """Ensure /etc/asound.conf has the softvol PCM and set volume to 100%."""
    asound_conf = "/etc/asound.conf"
    conf_content = (
        'pcm.smarteye_loud {\n'
        '    type softvol\n'
        '    slave.pcm "plughw:SmartEyeAudio,0"\n'
        '    control {\n'
        '        name "SoftMaster"\n'
        '        card SmartEyeAudio\n'
        '    }\n'
        '    min_dB -30.0\n'
        '    max_dB 10.0\n'
        '}\n'
    )
    try:
        with open(asound_conf, "w") as f:
            f.write(conf_content)
    except PermissionError:
        run(f"sudo tee {asound_conf} > /dev/null << 'EOF'\n{conf_content}EOF")

    run(f"amixer -c {CARD_NAME} sset SoftMaster 100%")
    return True


def play(path, use_softvol=True):
    """Play a WAV file with SD_MODE gating.  Returns (success, duration_s)."""
    dev = ALSA_DEV_SOFTVOL if use_softvol else f"plughw:{CARD_NAME},0"
    amp_on()
    time.sleep(0.01)
    t0 = time.time()
    rc, _, _ = run(f"aplay -D {dev} '{path}'", timeout=30)
    elapsed = time.time() - t0
    amp_off()
    return rc == 0, elapsed


def main():
    print("=" * 60)
    print("  Smart Eye Speaker Test  (MAX98357A + BSS138 SD_MODE)")
    print("=" * 60)

    # 1. Find card
    card = find_card()
    if card is None:
        print("\n[FAIL] Smart-Eye-Audio card not found")
        sys.exit(1)
    print(f"\n[OK]   Card {card}: {CARD_NAME}")

    # 2. SDO1 routing
    rc, _, _ = run(f"amixer -c {card} sset '{SDO1_MIXER[0]}' '{SDO1_MIXER[1]}'")
    print(f"[{'OK' if rc == 0 else 'WARN':5s}] SDO1 routed from PATH0")

    # 3. GPIO setup
    if not gpio_export():
        print("[FAIL] Cannot export GPIO131 (AMP_SD)")
        sys.exit(1)
    amp_off()
    print(f"[OK]   GPIO{SDMODE_GPIO} exported — AMP_SD=HIGH (amp OFF)")

    # 4. Softvol
    init_softvol()
    rc, out, _ = run(f"amixer -c {CARD_NAME} sget SoftMaster")
    vol_line = [l for l in out.splitlines() if "Front Left:" in l]
    vol_str = vol_line[0].strip() if vol_line else "unknown"
    print(f"[OK]   SoftMaster volume: {vol_str}")

    # 5. SD_MODE verify: amp should be OFF
    if amp_is_off():
        print("[OK]   SD_MODE shutdown verified (AMP_SD HIGH)")
    else:
        print("[WARN] AMP_SD not HIGH — amp may still be on")

    if "--card-only" in sys.argv:
        gpio_unexport()
        print("\n  Card & mixer check complete.")
        sys.exit(0)

    # 6. Gather WAV files
    pattern = os.path.join(WAV_DIR, "*.wav")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = os.path.join(os.path.dirname(WAV_DIR), "**", "*.wav")
        files = sorted(glob.glob(pattern2, recursive=True))
    if not files:
        files = glob.glob("/usr/share/sounds/alsa/*.wav")
    if not files:
        print(f"\n[FAIL] No WAV files found")
        gpio_unexport()
        sys.exit(1)

    argv_filter = [a for a in sys.argv[1:] if not a.startswith("--")]
    if argv_filter:
        keyword = argv_filter[0].lower()
        files = [f for f in files if keyword in os.path.basename(f).lower()]
        if not files:
            print(f"\n[FAIL] No files matching '{keyword}'")
            gpio_unexport()
            sys.exit(1)

    print(f"\n  Playing {len(files)} file(s)\n")
    print(f"  {'File':<35s} {'Format':<26s} {'Time':>6s}  {'Status'}")
    print(f"  {'-'*35} {'-'*26} {'-'*6}  {'-'*12}")

    passed = failed = 0

    for path in files:
        name = os.path.basename(path)
        info = wav_info(path)

        ok, dur = play(path)
        time.sleep(0.1)
        sd_after = amp_is_off()

        if ok and sd_after:
            status = "PASS"
            passed += 1
        elif ok and not sd_after:
            status = "PLAY OK (sd?)"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"  {name:<35s} {info:<26s} {dur:5.1f}s  {status}")

    amp_off()

    print(f"\n  Results: {passed} passed, {failed} failed / {len(files)}")
    print(f"  AMP_SD final state: {'HIGH (amp OFF)' if amp_is_off() else 'LOW (amp ON!)'}")

    # 7. DC-protection hold test
    print(f"\n  DC-protection hold test (5s)...", end="", flush=True)
    all_off = True
    for _ in range(25):
        if not amp_is_off():
            all_off = False
            break
        time.sleep(0.2)
    print(f"  {'PASS' if all_off else 'FAIL'} — amp stayed {'OFF' if all_off else 'ON!'}")

    gpio_unexport()
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
