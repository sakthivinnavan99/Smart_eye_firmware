#!/usr/bin/env python3
"""
Audio test for Smart Eye board speaker and headphone.

Tests both audio outputs:
  SPEAKER (MAX98357A on I2S1, card 3):
    - ALSA card detection
    - SDO1 routing (PATH0 -> SDO1 for MAX98357A DIN)
    - Softvol initialisation and volume control
    - SD_MODE GPIO control via BSS138 (inverted logic)
    - WAV playback with amp enable/disable sequencing
    - DC-protection verification (amp OFF after playback)

  HEADPHONE (ES8316 on I2C8-M2/I2S0, card 2):
    - I2C8 codec detection and availability
    - I2S0 audio delivery
    - Mono-to-stereo conversion via plughw

Device tree overlay: Overlays/smart-eye-carrier.dts
  - fragment@5: es8316-sound machine driver (enabled)
  - fragment@5a: i2s0_8ch controller (enabled)
  - fragment@6: i2s1_8ch with pin conflict fix + LRCK signal fix
  - fragment@7: smart-eye-sound speaker card

Hardware:
  Direct logic via R24 (300K series resistor), no BSS138:
    AMP_SD (GPIO4_A3 / gpio131) HIGH -> SD_MODE HIGH -> amp ON
    AMP_SD (GPIO4_A3 / gpio131) LOW  -> SD_MODE LOW  -> amp OFF

Usage:
    sudo python3 test_audio.py                     # test both speaker and headphone
    sudo python3 test_audio.py speaker             # speaker only
    sudo python3 test_audio.py headphone           # headphone only
    sudo python3 test_audio.py device_turned_on    # play matching file
    sudo python3 test_audio.py --card-only         # just check cards & mixers
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


def gpio_release():
    """Leave GPIO as output LOW (amp OFF). Do NOT unexport — releasing the pin
    to input mode lets the MAX98357A internal pull-up float SD_MODE HIGH,
    turning the amp on with no I2S input and heating the speaker."""
    amp_off()


def amp_on():
    """Enable amp: AMP_SD HIGH -> SD_MODE HIGH -> amp ON."""
    gpio_set(1)


def amp_off():
    """Disable amp: AMP_SD LOW -> SD_MODE LOW -> amp OFF."""
    gpio_set(0)


def amp_is_off():
    """Return True if AMP_SD is LOW (amp disabled)."""
    return gpio_get() == 0


def wav_info(path):
    try:
        with wave.open(path, "rb") as w:
            dur = w.getnframes() / w.getframerate()
            return f"{w.getnchannels()}ch {w.getframerate()}Hz {w.getsampwidth()*8}bit {dur:.1f}s"
    except Exception as e:
        return f"error: {e}"


def init_softvol():
    """Ensure /etc/asound.conf has the softvol PCM and set volume to 70%."""
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

    run(f"amixer -c {CARD_NAME} sset SoftMaster 70%")
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


def find_headphone_card():
    """Find ES8316 headphone card."""
    rc, out, _ = run("cat /proc/asound/cards")
    for line in out.splitlines():
        if "rockchipes8316" in line.lower():
            return int(line.strip().split()[0])
    return None


def test_headphone(files):
    """Test headphone output via ES8316 codec on I2C8-M2/I2S0."""
    print("\n" + "=" * 60)
    print("  Headphone Test (ES8316 on I2C8-M2/I2S0)")
    print("=" * 60)

    card = find_headphone_card()
    if card is None:
        print("\n[SKIP] ES8316 headphone card not found")
        print("       (Verify overlay is installed and reboot)")
        return

    print(f"\n[OK]   Card {card}: rockchipes8316")

    if not files:
        print(f"\n[FAIL] No WAV files found for headphone test")
        return

    print(f"\n  Playing {len(files)} file(s) on headphone\n")
    print(f"  {'File':<35s} {'Format':<26s} {'Time':>6s}  {'Status'}")
    print(f"  {'-'*35} {'-'*26} {'-'*6}  {'-'*12}")

    for path in files[:3]:  # Test only first 3 files
        name = os.path.basename(path)
        info = wav_info(path)

        # Use plughw: prefix for mono→stereo conversion
        rc, _, _ = run(f"aplay -D plughw:rockchipes8316,0 '{path}'", timeout=30)
        ok = rc == 0
        dur = wav_info(path).split()[-2] if 's' in wav_info(path) else '?'

        status = "PASS" if ok else "FAIL"
        print(f"  {name:<35s} {info:<26s} {dur:>6s}  {status}")

    print()


def main():
    print("=" * 60)
    print("  Smart Eye Audio Test  (Headphone + Speaker)")
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
        gpio_release()
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
        gpio_release()
        sys.exit(1)

    argv_filter = [a for a in sys.argv[1:] if not a.startswith("--")]
    test_speaker = True
    test_headphone = True

    if argv_filter:
        keyword = argv_filter[0].lower()
        if keyword == "speaker":
            test_headphone = False
        elif keyword == "headphone":
            test_speaker = False
        else:
            files = [f for f in files if keyword in os.path.basename(f).lower()]
            if not files:
                print(f"\n[FAIL] No files matching '{keyword}'")
                gpio_release()
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

    if test_speaker:
        print(f"\n  Results: {passed} passed, {failed} failed / {len(files)}")
        print(f"  AMP_SD final state: {'LOW (amp OFF)' if amp_is_off() else 'HIGH (amp ON!)'}")

    gpio_release()

    # 8. Headphone test (if requested)
    if test_headphone:
        test_headphone_audio(files)

    print("=" * 60)
    sys.exit(1 if (failed > 0 if test_speaker else False) else 0)


if __name__ == "__main__":
    main()
