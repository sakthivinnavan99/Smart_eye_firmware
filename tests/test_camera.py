#!/usr/bin/env python3
"""
Test RPi Camera v2 (IMX219) on Radxa CM5 - headless (no display required).

Tests:
  1. Camera overlay enabled
  2. Video device detection (/dev/video*)
  3. V4L2 device query (formats, resolution)
  4. Single frame capture to JPEG
  5. Short video recording (GStreamer HW-accelerated H264)
  6. Multi-frame capture stability test
  7. Video file integrity check

Run with: sudo python3 test_camera.py

Before first run:
  sudo cp /boot/dtbo/radxa-cm5-io-rpi-camera-v2.dtbo.disabled \
          /boot/dtbo/radxa-cm5-io-rpi-camera-v2.dtbo
  sudo reboot
"""

import os
import sys
import time
import glob
import subprocess
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_output")

CAMERA_OVERLAY = "radxa-cm5-io-rpi-camera-v2.dtbo"
EXPECTED_SENSOR = "imx219"

PASS = 0
FAIL = 0
WARN = 0


def log_pass(msg):
    global PASS
    print("  [PASS] {}".format(msg))
    PASS += 1


def log_fail(msg):
    global FAIL
    print("  [FAIL] {}".format(msg))
    FAIL += 1


def log_warn(msg):
    global WARN
    print("  [WARN] {}".format(msg))
    WARN += 1


def log_info(msg):
    print("  [INFO] {}".format(msg))


def run(cmd, timeout=30):
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def find_video_device():
    """Find the ISP mainpath video device for the camera."""
    devices = sorted(glob.glob("/dev/video*"))
    # Filter out codec-only devices
    candidates = [d for d in devices if d not in ("/dev/video-dec0", "/dev/video-enc0")]

    if not candidates:
        return None

    for dev in candidates:
        num = dev.replace("/dev/video", "")
        if not num.isdigit():
            continue
        rc, out, _ = run("v4l2-ctl -d {} --all 2>/dev/null".format(dev))
        if rc == 0 and ("mainpath" in out.lower() or "capture" in out.lower()):
            return dev

    # Fallback: return the highest numbered video device (Radxa ISP convention)
    numbered = [d for d in candidates if d.replace("/dev/video", "").isdigit()]
    if numbered:
        return sorted(numbered, key=lambda d: int(d.replace("/dev/video", "")))[-1]

    return candidates[0] if candidates else None


def test_overlay():
    """Test 1: Check camera overlay is enabled."""
    print("\n--- Test 1: Camera Overlay ---")

    enabled_path = "/boot/dtbo/" + CAMERA_OVERLAY
    disabled_path = enabled_path + ".disabled"

    if os.path.exists(enabled_path):
        log_pass("Camera overlay enabled: {}".format(CAMERA_OVERLAY))
        return True
    elif os.path.exists(disabled_path):
        log_fail("Camera overlay exists but DISABLED")
        print("         Enable it with:")
        print("           sudo mv {} {}".format(disabled_path, enabled_path))
        print("           sudo reboot")
        return False
    else:
        log_fail("Camera overlay not found")
        return False


def test_video_devices():
    """Test 2: Detect video devices."""
    print("\n--- Test 2: Video Devices ---")

    rc, out, _ = run("ls -la /dev/video* 2>/dev/null")
    if rc != 0 or not out:
        log_fail("No /dev/video* devices found")
        print("         Camera not detected. Check:")
        print("         - Overlay is enabled and system rebooted")
        print("         - Camera ribbon cable is seated properly")
        print("         - Camera connector latch is closed")
        return None

    for line in out.split("\n"):
        log_info(line.strip())

    dev = find_video_device()
    if dev:
        log_pass("Camera device found: {}".format(dev))
    else:
        log_fail("Could not identify camera capture device")

    return dev


def test_v4l2_query(dev):
    """Test 3: Query V4L2 device capabilities."""
    print("\n--- Test 3: V4L2 Device Info ---")

    rc, out, _ = run("v4l2-ctl -d {} --all 2>/dev/null".format(dev))
    if rc != 0:
        log_warn("v4l2-ctl query failed for {}".format(dev))
        return

    for line in out.split("\n"):
        line = line.strip()
        if any(k in line.lower() for k in ["driver", "card", "bus", "width", "height",
                                             "pixel", "format", "capability"]):
            log_info(line)

    log_pass("V4L2 device query successful")

    rc, out, _ = run("v4l2-ctl -d {} --list-formats-ext 2>/dev/null".format(dev))
    if rc == 0 and out:
        print("\n  Supported formats:")
        for line in out.split("\n")[:20]:
            print("    {}".format(line.strip()))


def test_dmesg_sensor():
    """Test 3b: Check dmesg for sensor detection."""
    print("\n--- Test 3b: Sensor Detection (dmesg) ---")

    rc, out, _ = run("dmesg 2>/dev/null | grep -iE 'imx219|ov5647|camera|rkisp|sensor' | tail -10")
    if rc != 0 or not out:
        log_warn("Cannot read dmesg (need root) or no sensor messages")
        return

    for line in out.split("\n"):
        log_info(line.strip())

    if EXPECTED_SENSOR in out.lower():
        log_pass("{} sensor detected in dmesg".format(EXPECTED_SENSOR.upper()))
    else:
        log_warn("{} not found in dmesg".format(EXPECTED_SENSOR.upper()))


def test_capture_jpeg(dev):
    """Test 4: Capture a single JPEG frame."""
    print("\n--- Test 4: Single Frame Capture (JPEG) ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "test_frame.jpg")

    if os.path.exists(out_file):
        os.remove(out_file)

    # Use GStreamer to capture a single frame
    cmd = (
        "gst-launch-1.0 -e v4l2src device={dev} num-buffers=5 "
        "! video/x-raw,width=640,height=480 "
        "! videoconvert "
        "! jpegenc quality=90 "
        "! filesink location={out} "
        "2>&1"
    ).format(dev=dev, out=out_file)

    log_info("Capturing frame with GStreamer...")
    rc, out, err = run(cmd, timeout=15)

    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        size_kb = os.path.getsize(out_file) / 1024
        log_pass("Frame captured: {} ({:.1f} KB)".format(out_file, size_kb))
        return True
    else:
        log_info("GStreamer output: {}".format(out or err))

        # Fallback: try v4l2-ctl
        log_info("Trying v4l2-ctl fallback...")
        raw_file = os.path.join(OUTPUT_DIR, "test_frame.raw")
        cmd2 = "v4l2-ctl -d {} --stream-mmap --stream-count=1 --stream-to={} 2>&1".format(dev, raw_file)
        rc2, out2, _ = run(cmd2, timeout=15)

        if os.path.exists(raw_file) and os.path.getsize(raw_file) > 0:
            size_kb = os.path.getsize(raw_file) / 1024
            log_pass("Raw frame captured: {} ({:.1f} KB)".format(raw_file, size_kb))
            return True
        else:
            log_fail("Frame capture failed")
            log_info(out2)
            return False


def test_video_record(dev, duration=5):
    """Test 5: Record a short video clip."""
    print("\n--- Test 5: Video Recording ({}s) ---".format(duration))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "test_video.mp4")

    if os.path.exists(out_file):
        os.remove(out_file)

    # HW-accelerated H264 encoding with Rockchip MPP
    cmd = (
        "timeout {dur} gst-launch-1.0 -e v4l2src device={dev} "
        "! video/x-raw,width=1920,height=1080,framerate=30/1 "
        "! videoconvert "
        "! mpph264enc "
        "! h264parse "
        "! mp4mux "
        "! filesink location={out} "
        "2>&1"
    ).format(dev=dev, dur=duration + 3, out=out_file)

    log_info("Recording {}s video (1080p, HW H264)...".format(duration))
    rc, out, err = run(cmd, timeout=duration + 10)

    if os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
        size_kb = os.path.getsize(out_file) / 1024
        log_pass("Video recorded: {} ({:.1f} KB)".format(out_file, size_kb))
        return out_file

    # Fallback: 640x480 software encode
    log_info("1080p HW encode failed, trying 640x480 software encode...")
    out_file2 = os.path.join(OUTPUT_DIR, "test_video_sw.mp4")

    cmd2 = (
        "timeout {dur} gst-launch-1.0 -e v4l2src device={dev} "
        "! video/x-raw,width=640,height=480,framerate=30/1 "
        "! videoconvert "
        "! x264enc tune=zerolatency speed-preset=ultrafast "
        "! h264parse "
        "! mp4mux "
        "! filesink location={out} "
        "2>&1"
    ).format(dev=dev, dur=duration + 3, out=out_file2)

    rc2, out2, _ = run(cmd2, timeout=duration + 10)

    if os.path.exists(out_file2) and os.path.getsize(out_file2) > 5000:
        size_kb = os.path.getsize(out_file2) / 1024
        log_pass("Video recorded (SW fallback): {} ({:.1f} KB)".format(out_file2, size_kb))
        return out_file2

    log_fail("Video recording failed")
    log_info("GStreamer output: {}".format(out or out2))
    return None


def test_video_integrity(video_path):
    """Test 6: Verify recorded video file."""
    print("\n--- Test 6: Video Integrity Check ---")

    if not video_path or not os.path.exists(video_path):
        log_fail("No video file to check")
        return

    size_kb = os.path.getsize(video_path) / 1024
    log_info("File: {} ({:.1f} KB)".format(video_path, size_kb))

    # Check with GStreamer discoverer
    cmd = "gst-discoverer-1.0 {} 2>&1".format(video_path)
    rc, out, _ = run(cmd, timeout=10)

    if rc == 0 and out:
        for line in out.split("\n"):
            line = line.strip()
            if any(k in line.lower() for k in ["duration", "video", "width", "height",
                                                 "framerate", "codec", "container"]):
                log_info(line)
        log_pass("Video file is valid")
    else:
        # Fallback: check with gst-launch decode test
        log_info("gst-discoverer not available, checking with decode test...")
        cmd2 = (
            "timeout 5 gst-launch-1.0 filesrc location={} "
            "! qtdemux ! h264parse ! decodebin ! fakesink 2>&1"
        ).format(video_path)
        rc2, out2, _ = run(cmd2, timeout=10)
        if rc2 == 0:
            log_pass("Video file decodes successfully")
        else:
            log_warn("Could not verify video integrity")


def test_multi_frame(dev, count=30):
    """Test 7: Capture multiple frames to check stability."""
    print("\n--- Test 7: Multi-Frame Stability ({} frames) ---".format(count))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = (
        "v4l2-ctl -d {dev} --stream-mmap --stream-count={count} "
        "--stream-to=/dev/null 2>&1"
    ).format(dev=dev, count=count)

    log_info("Capturing {} frames to /dev/null...".format(count))
    start = time.time()
    rc, out, _ = run(cmd, timeout=30)
    elapsed = time.time() - start

    if rc == 0:
        fps = count / elapsed if elapsed > 0 else 0
        log_pass("{} frames captured in {:.1f}s ({:.1f} FPS)".format(count, elapsed, fps))
        if fps < 10:
            log_warn("FPS is low ({:.1f}), check camera connection".format(fps))
    else:
        log_fail("Multi-frame capture failed")
        if out:
            log_info(out[:200])


def test_resolution_sweep(dev):
    """Test 8: Test multiple resolutions."""
    print("\n--- Test 8: Resolution Test ---")

    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (3280, 2464),  # IMX219 full resolution
    ]

    for w, h in resolutions:
        cmd = (
            "timeout 5 gst-launch-1.0 -e v4l2src device={dev} num-buffers=3 "
            "! video/x-raw,width={w},height={h} "
            "! fakesink 2>&1"
        ).format(dev=dev, w=w, h=h)

        rc, out, _ = run(cmd, timeout=10)
        if rc == 0:
            log_pass("{}x{} - OK".format(w, h))
        else:
            log_info("{}x{} - not supported or failed".format(w, h))


def main():
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 test_camera.py")
        sys.exit(1)

    print("========================================")
    print(" RPi Camera v2 (IMX219) Test")
    print(" Headless - no display required")
    print(" Output: {}".format(OUTPUT_DIR))
    print("========================================")

    if not test_overlay():
        print("\n>> Camera overlay not enabled. Enable and reboot first.")
        print("   sudo mv /boot/dtbo/{}.disabled /boot/dtbo/{}".format(
            CAMERA_OVERLAY, CAMERA_OVERLAY))
        print("   sudo reboot")
        sys.exit(1)

    dev = test_video_devices()
    if not dev:
        print("\n>> No camera device found. Check:")
        print("   - Camera ribbon cable connected properly")
        print("   - System was rebooted after enabling overlay")
        sys.exit(1)

    try:
        test_v4l2_query(dev)
        test_dmesg_sensor()
        test_capture_jpeg(dev)
        video_path = test_video_record(dev, duration=5)
        test_video_integrity(video_path)
        test_multi_frame(dev)
        test_resolution_sweep(dev)
    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    print("\n========================================")
    print(" Results: {} passed, {} failed, {} warnings".format(PASS, FAIL, WARN))
    print("========================================")

    if os.path.isdir(OUTPUT_DIR):
        print("\n Output files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath) / 1024
            print("   {} ({:.1f} KB)".format(f, size))
        print("\n   Transfer to your PC to view:")
        print("   scp radxa@<board-ip>:{}/* .".format(OUTPUT_DIR))

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
