#!/usr/bin/env python3
"""
Live camera stream over local network (MJPEG via HTTP).

Streams the IMX219 camera feed as MJPEG over HTTP so you can view it
from any browser on the same network.

Usage:
    sudo -E venv/bin/python3 tests/test_camera_stream.py [--port 8080] [--device /dev/video11]

Then open in browser:
    http://192.168.42.100:8080
"""

import argparse
import sys
import os
import time
import threading
import cv2

from http.server import HTTPServer, BaseHTTPRequestHandler

CAMERA_DEV = "/dev/video-camera0"
WIDTH = 640
HEIGHT = 480
FPS = 30
QUALITY = 80


class CameraCapture:
    def __init__(self, device, width, height, fps):
        self._lock = threading.Lock()
        self._frame = None
        self._running = True

        real_path = os.path.realpath(device)
        dev_num = int(real_path.replace("/dev/video", ""))
        print(f"[INFO] Opening {device} -> {real_path} (index {dev_num})")
        self.cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {device} ({real_path})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera ready: {w}x{h}")
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self.cap.release()


camera = None


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._serve_page()
        elif self.path == "/stream":
            self._serve_stream()
        elif self.path == "/snapshot":
            self._serve_snapshot()
        else:
            self.send_error(404)

    def _serve_page(self):
        host = self.headers.get("Host", "localhost")
        html = f"""\
<!DOCTYPE html>
<html>
<head>
    <title>Smart Eye - Live Camera</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #1a1a2e; color: #e0e0e0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; min-height: 100vh;
            padding: 20px;
        }}
        h1 {{
            font-size: 1.5rem; font-weight: 300;
            letter-spacing: 2px; text-transform: uppercase;
            margin-bottom: 16px; color: #00d4ff;
        }}
        .stream-container {{
            border: 2px solid #00d4ff33;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
        }}
        img#stream {{
            display: block; max-width: 100%;
            width: 480px; height: auto;
        }}
        .info {{
            margin-top: 12px; font-size: 0.85rem;
            color: #888; text-align: center;
        }}
        .info a {{ color: #00d4ff; text-decoration: none; }}
        .controls {{
            margin-top: 16px; display: flex; gap: 12px;
        }}
        button {{
            background: #00d4ff22; color: #00d4ff;
            border: 1px solid #00d4ff44; border-radius: 6px;
            padding: 8px 20px; cursor: pointer;
            font-size: 0.9rem; transition: all 0.2s;
        }}
        button:hover {{ background: #00d4ff44; }}
    </style>
</head>
<body>
    <h1>Smart Eye &mdash; Live Camera</h1>
    <div class="stream-container">
        <img id="stream" src="/stream" alt="Live Stream">
    </div>
    <div class="controls">
        <button onclick="window.open('/snapshot','_blank')">Snapshot</button>
        <button onclick="document.getElementById('stream').src='/stream?'+Date.now()">Reconnect</button>
    </div>
    <div class="info">
        MJPEG stream: <a href="http://{host}/stream">http://{host}/stream</a><br>
        Snapshot: <a href="http://{host}/snapshot">http://{host}/snapshot</a>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        try:
            while True:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue
                _, jpeg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                data = jpeg.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_snapshot(self):
        frame = camera.get_frame()
        if frame is None:
            self.send_error(503, "No frame available")
            return
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        data = jpeg.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", "inline; filename=snapshot.jpg")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if "/stream" not in (args[0] if args else ""):
            print(f"[HTTP] {args[0]}" if args else "")


def main():
    global camera, FPS, QUALITY

    parser = argparse.ArgumentParser(description="Live camera stream over HTTP")
    parser.add_argument("--device", default=CAMERA_DEV, help="V4L2 camera device")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--width", type=int, default=WIDTH, help="Frame width")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Frame height")
    parser.add_argument("--fps", type=int, default=FPS, help="Target FPS")
    parser.add_argument("--quality", type=int, default=QUALITY, help="JPEG quality 1-100")
    args = parser.parse_args()

    FPS = args.fps
    QUALITY = args.quality

    print("=" * 50)
    print("  Smart Eye - Live Camera Stream")
    print("=" * 50)
    print(f"  Device:  {args.device}")
    print(f"  Size:    {args.width}x{args.height} @ {args.fps} FPS")
    print(f"  Quality: {args.quality}%")
    print()

    camera = CameraCapture(args.device, args.width, args.height, args.fps)

    server = HTTPServer(("0.0.0.0", args.port), StreamHandler)
    ip = "192.168.42.100"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print(f"  Stream:  http://{ip}:{args.port}")
    print(f"  Snap:    http://{ip}:{args.port}/snapshot")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
