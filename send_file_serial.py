#!/usr/bin/env python3
"""
Transfer any local file to the Radxa over a serial console.

HOW IT WORKS
------------
  1. Base64-encode the binary locally (safe for any file type — binary, text, etc.)
  2. Send it in 76-char chunks via  printf '%s' '<chunk>' >> /tmp/_<name>.b64
  3. Decode on the Radxa:           base64 -d /tmp/_<name>.b64 > <remote_dest>
  4. Verify byte-count matches between local and remote

USAGE
-----
    python3 send_file_serial.py <local_file> <remote_dest> [--port PORT] [--baud BAUD]

    Positional arguments:
      local_file    Path to the file on this machine (relative or absolute)
      remote_dest   Destination path on the Radxa  (~/... or /absolute/path)

    Optional arguments:
      --port PORT   Serial device  (default: /dev/cu.usbserial-0001)
      --baud BAUD   Baud rate      (default: 1500000)

FILE-TYPE EXAMPLES
------------------

  Device-tree overlay (.dtbo)
      python3 send_file_serial.py \\
          Overlays/smart-eye-carrier.dtbo \\
          /boot/dtbo/smart-eye-carrier.dtbo

  RKNN model (default app model)
      python3 send_file_serial.py \\
          models/pathpal/model_v2_large.rknn \\
          ~/Smart_eye_firmware/models/pathpal/model_v2_large.rknn

  RKNN model (alternate variant)
      python3 send_file_serial.py \\
          models/pathpal/yolov8n_2912.rknn \\
          ~/Smart_eye_firmware/models/pathpal/yolov8n_2912.rknn

  ONNX model
      python3 send_file_serial.py \\
          models/yolov8/yolov8n.onnx \\
          ~/Smart_eye_firmware/models/yolov8/yolov8n.onnx

  Python source file
      python3 send_file_serial.py \\
          pathpal_project/main.py \\
          ~/Smart_eye_firmware/pathpal_project/main.py

  Shell script
      python3 send_file_serial.py \\
          scripts/setup_device.sh \\
          ~/Smart_eye_firmware/scripts/setup_device.sh

  WAV audio asset
      python3 send_file_serial.py \\
          wav/English/battery_low.wav \\
          ~/Smart_eye_firmware/wav/English/battery_low.wav

  TTS voice model (.onnx)
      python3 send_file_serial.py \\
          piper/en_US-amy-medium.onnx \\
          ~/Smart_eye_firmware/piper/en_US-amy-medium.onnx

  Any binary to /tmp (no sudo needed)
      python3 send_file_serial.py \\
          myfile.bin \\
          /tmp/myfile.bin

NOTES
-----
  - Destinations under /boot/ require root on the Radxa. The script first tries a
    plain write and automatically retries with  sudo bash -c '...'  if it fails.
  - Parent directories are created automatically (mkdir -p, with sudo fallback).
  - Large files (RKNN models are typically 5–30 MB) will take several minutes at
    1500000 baud over serial. SSH/SCP is faster when the network is available.
  - The 76-char line width and 0.05 s LINE_DELAY at the top of the file can be
    tuned if you see buffer overruns or dropped characters on your serial adapter.
"""

import argparse
import base64
import os
import sys
import time
import serial

LINE_DELAY = 0.05   # seconds between printf lines


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
    ser.write((line + "\n").encode())
    ser.flush()
    time.sleep(wait)
    return read_until_prompt(ser)


def parse_size(resp: str) -> int | None:
    for l in resp.splitlines():
        parts = l.strip().split()
        if parts and parts[0].isdigit():
            return int(parts[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer a file to the Radxa over serial console."
    )
    parser.add_argument("local_file", help="Path to the local file to send")
    parser.add_argument("remote_dest", help="Destination path on the Radxa (e.g. ~/foo or /boot/dtbo/foo)")
    parser.add_argument("--port", default="/dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=1500000)
    args = parser.parse_args()

    # ── Fix ~ expanded by the local shell ────────────────────────────
    local_home = os.path.expanduser("~")
    if args.remote_dest.startswith(local_home + "/") or args.remote_dest == local_home:
        args.remote_dest = "~" + args.remote_dest[len(local_home):]
        print(f"NOTE: remote path corrected to {args.remote_dest} (~ was expanded by local shell)")

    basename = os.path.basename(args.local_file)
    tmp_b64 = f"/tmp/_{basename}.b64"

    # ── Load and encode the file ──────────────────────────────────────
    try:
        raw = open(args.local_file, "rb").read()
    except FileNotFoundError:
        sys.exit(f"ERROR: {args.local_file} not found.")

    b64 = base64.b64encode(raw).decode()
    lines = [b64[i:i+76] for i in range(0, len(b64), 76)]

    print(f"Port     : {args.port} @ {args.baud} baud")
    print(f"Local    : {args.local_file}  ({len(raw)} bytes)")
    print(f"Remote   : {args.remote_dest}")
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
    cmd(ser, f"rm -f {tmp_b64}", wait=0.3)

    # ── Send base64 lines ─────────────────────────────────────────────
    print(f"[3/5] Sending {len(lines)} lines...")
    for i, line in enumerate(lines):
        ser.write(f"printf '%s' '{line}' >> {tmp_b64}\n".encode())
        ser.flush()
        time.sleep(LINE_DELAY)
        if (i + 1) % 20 == 0:
            read_until_prompt(ser, timeout=3.0)
            print(f"      {i+1}/{len(lines)} lines ...")

    read_until_prompt(ser, timeout=3.0)
    print("      All lines sent.")

    # ── Verify tmp b64 file size ──────────────────────────────────────
    print("[4/5] Verifying base64 tmp file...")
    resp = cmd(ser, f"wc -c {tmp_b64}", wait=0.5)
    got_size = parse_size(resp)
    expected_b64_len = len(b64)

    if got_size is None:
        print(f"      WARNING: could not parse wc output: {resp.strip()!r}")
    elif got_size == expected_b64_len:
        print(f"      OK: {got_size} chars (matches expected {expected_b64_len})")
    else:
        print(f"      WARNING: expected {expected_b64_len} chars but got {got_size}")

    # ── Decode to destination ─────────────────────────────────────────
    print(f"[5/5] Decoding to {args.remote_dest} ...")

    # Create parent directory if needed (best-effort with sudo)
    remote_dir = os.path.dirname(args.remote_dest)
    if remote_dir and remote_dir not in ("~", "."):
        cmd(ser, f"mkdir -p {remote_dir} 2>/dev/null || sudo mkdir -p {remote_dir}", wait=0.5)

    # Try plain write first; if it fails (e.g. /boot/ needs root) retry with sudo.
    # Use sudo -n (non-interactive) so it never hangs waiting for a password —
    # if passwordless sudo isn't configured the error is reported immediately.
    resp = cmd(ser, f"base64 -d {tmp_b64} > {args.remote_dest} 2>/tmp/_write_err", wait=1.5)
    err = cmd(ser, "cat /tmp/_write_err", wait=0.3)
    if "Permission denied" in err or "cannot" in err.lower():
        print("      Plain write failed, retrying with sudo -n ...")
        resp = cmd(ser, f"sudo -n bash -c 'base64 -d {tmp_b64} > {args.remote_dest}'", wait=1.5)
        err2 = cmd(ser, "echo exit:$?", wait=0.3)
        if "exit:0" not in err2:
            print(f"      WARNING: sudo write may have failed: {resp.strip()!r}")
            print("      Tip: run  'echo \"radxa ALL=(ALL) NOPASSWD: ALL\" | sudo tee /etc/sudoers.d/radxa'  on the Radxa first.")
    cmd(ser, f"rm -f {tmp_b64} /tmp/_write_err", wait=0.3)

    # ── Verify final file ─────────────────────────────────────────────
    resp = cmd(ser, f"wc -c {args.remote_dest}", wait=0.5)
    final_size = parse_size(resp)

    print()
    if final_size == len(raw):
        print(f"SUCCESS: {args.remote_dest} = {final_size} bytes  (matches local {len(raw)} bytes)")
    elif final_size is not None:
        print(f"WARNING: size mismatch — local {len(raw)} B, remote {final_size} B")
    else:
        print(f"Could not verify. Check on Radxa: wc -c {args.remote_dest}")
        print(f"  wc output: {resp.strip()!r}")

    ser.close()


if __name__ == "__main__":
    main()
