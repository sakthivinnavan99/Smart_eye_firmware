#!/usr/bin/env python3
"""
Transfer Overlays/smart-eye-carrier.dtbo to the Radxa over serial console.

Strategy:
  1. Base64-encode the binary locally
  2. Send it in 76-char chunks via printf >> /tmp/_.b64
  3. sudo base64 -d /tmp/_.b64 > /boot/dtbo/smart-eye-carrier.dtbo
  4. Verify byte-count matches

Usage (from repo root):
    python3 send_dtbo_serial.py [--port /dev/cu.usbserial-0001] [--baud 1500000]
"""

import argparse
import base64
import sys
import time
import serial

LOCAL_FILE  = "Overlays/smart-eye-carrier.dtbo"
TMP_B64     = "/tmp/_dtbo.b64"
TMP_DTBO    = "/tmp/smart-eye-carrier.dtbo"
REMOTE_HOME = "~/smart-eye-carrier.dtbo"
REMOTE_DEST = "/boot/dtbo/smart-eye-carrier.dtbo"
LINE_DELAY  = 0.05   # seconds between printf lines


def read_until_prompt(ser: serial.Serial, timeout: float = 4.0) -> str:
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = ser.read(ser.in_waiting or 1)
        if data:
            buf += data
            text = buf.decode("utf-8", errors="replace")
            if text.rstrip().endswith(("$", "#", ">")):
                return text
        else:
            time.sleep(0.05)
    return buf.decode("utf-8", errors="replace")


def cmd(ser: serial.Serial, line: str, wait: float = 0.5) -> str:
    """Send a command and wait for the prompt, return the response."""
    ser.write((line + "\n").encode())
    ser.flush()
    time.sleep(wait)
    resp = read_until_prompt(ser)
    return resp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=1500000)
    args = parser.parse_args()

    # ── Load and encode the file ──────────────────────────────────────
    try:
        raw = open(LOCAL_FILE, "rb").read()
    except FileNotFoundError:
        sys.exit(f"ERROR: {LOCAL_FILE} not found — run from the repo root.")

    b64 = base64.b64encode(raw).decode()
    lines = [b64[i:i+76] for i in range(0, len(b64), 76)]

    print(f"Port     : {args.port} @ {args.baud} baud")
    print(f"Local    : {LOCAL_FILE}  ({len(raw)} bytes)")
    print(f"Remote   : {REMOTE_DEST}")
    print(f"Transfer : {len(lines)} base64 lines")
    print()

    # ── Open serial port ─────────────────────────────────────────────
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"ERROR: cannot open {args.port}: {e}")

    # ── Wake the shell ────────────────────────────────────────────────
    print("[1/5] Waking remote shell...")
    ser.write(b"\n")
    ser.flush()
    read_until_prompt(ser, timeout=3.0)
    ser.write(b"\n")
    ser.flush()
    read_until_prompt(ser, timeout=3.0)

    # ── Clear any leftover tmp file ───────────────────────────────────
    print("[2/5] Clearing temp file...")
    cmd(ser, f"rm -f {TMP_B64}", wait=0.3)

    # ── Send base64 lines ─────────────────────────────────────────────
    print(f"[3/5] Sending {len(lines)} lines...")
    for i, line in enumerate(lines):
        ser.write(f"printf '%s' '{line}' >> {TMP_B64}\n".encode())
        ser.flush()
        time.sleep(LINE_DELAY)
        # Drain output every 20 lines so the remote buffer doesn't fill
        if (i + 1) % 20 == 0:
            read_until_prompt(ser, timeout=3.0)
            print(f"      {i+1}/{len(lines)} lines ...")

    read_until_prompt(ser, timeout=3.0)
    print("      All lines sent.")

    # ── Verify tmp file size ──────────────────────────────────────────
    print("[4/5] Verifying base64 tmp file...")
    resp = cmd(ser, f"wc -c {TMP_B64}", wait=0.5)
    expected_b64_len = len(b64)
    tmp_size_line = [l for l in resp.splitlines() if TMP_B64 in l or l.strip().split() and l.strip().split()[0].isdigit()]
    got_size = None
    for l in resp.splitlines():
        parts = l.strip().split()
        if parts and parts[0].isdigit():
            got_size = int(parts[0])
            break

    if got_size is None:
        print(f"      WARNING: could not parse wc output: {resp.strip()!r}")
    elif got_size == expected_b64_len:
        print(f"      OK: {got_size} chars (matches expected {expected_b64_len})")
    else:
        print(f"      WARNING: expected {expected_b64_len} chars but got {got_size}")

    # ── Decode to home dir (no sudo required) ────────────────────────
    print(f"[5/5] Decoding to {REMOTE_HOME} ...")
    cmd(ser, f"base64 -d {TMP_B64} > {REMOTE_HOME}", wait=1.0)
    cmd(ser, f"rm -f {TMP_B64}", wait=0.3)

    # Verify
    resp = cmd(ser, f"wc -c {REMOTE_HOME}", wait=0.5)
    final_size = None
    for l in resp.splitlines():
        parts = l.strip().split()
        if parts and parts[0].isdigit():
            final_size = int(parts[0])
            break

    print()
    if final_size == len(raw):
        print(f"SUCCESS: {REMOTE_HOME} = {final_size} bytes  (matches local {len(raw)} bytes)")
    elif final_size is not None:
        print(f"WARNING: size mismatch — local {len(raw)} B, remote {final_size} B")
    else:
        print(f"Could not verify. Check on Radxa: wc -c {REMOTE_HOME}")
        print(f"  wc output: {resp.strip()!r}")

    ser.close()

    print()
    print("File is at ~/smart-eye-carrier.dtbo on the Radxa.")
    print("To install the overlay, run on the Radxa:")
    print(f"  sudo mkdir -p /boot/dtbo")
    print(f"  sudo cp ~/smart-eye-carrier.dtbo {REMOTE_DEST}")
    print(f"  sudo reboot")


if __name__ == "__main__":
    main()
