# Smart Eye Firmware

Intelligent vision and sensor fusion system for Radxa devices with IMX219 camera and ultrasonic distance sensor. Features real-time object detection using YOLOv8, camera streaming, and distance measurement capabilities.

## Project Overview

This project implements a comprehensive smart vision system for Radxa RK3588 boards including:
- **Real-time camera streaming** with IMX219 Raspberry Pi Camera v2.0 (with autofocus)
- **Ultrasonic distance sensing** via UART serial communication
- **Object detection** using YOLOv8 with RKNN optimization
- **Headless mode support** for systems without display

## Installation

### Prerequisites
- Radxa RK3588 board (CM5-IO recommended)
- IMX219 Raspberry Pi Camera v2.0
- Ultrasonic sensor module (JSN-SR04T or similar)
- Python 3.11+
- System packages: `v4l-utils`, `libmraa2`, `libmraa-dev`

### Setup Instructions

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y v4l-utils libmraa2 libmraa-dev mraa-tools
   ```

2. **Create and activate virtual environment with system packages:**
   ```bash
   cd /home/radxa/Smart_eye_firmware
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Smart_eye_firmware/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── models/                            # ML models
│   ├── labels.txt                     # YOLOv8 class labels
│   ├── network.rpk                    # RKNN model
│   └── yolov8/                        # YOLOv8 model files
├── pathpal_project/                   # Main project modules
│   ├── __init__.py
│   ├── main.py                        # Main entry point
│   ├── CameraTest.py                  # Camera streaming module
│   └── UltrasonicSensor.py            # Ultrasonic sensor module
└── wav/                               # Audio files and utilities
    ├── all_metrics.py                 # Metrics calculation
    ├── English/                       # English audio samples
    └── Hindi/                         # Hindi audio samples
```

## Usage

### Camera Streaming

**Live camera feed (with display):**
```bash
python ./pathpal_project/CameraTest.py
```

**Headless mode (saves frames to disk):**
```bash
python ./pathpal_project/CameraTest.py --headless
```

**Record video (30 seconds):**
```bash
python ./pathpal_project/CameraTest.py --record --duration 30
```

**Camera options:**
```
--headless, -hl          Run without display, save frames to camera_frames/
--device, -d             Video device path (default: /dev/video11)
--fps                    Target FPS (default: 30)
--record, -r             Record video to MP4
--duration, -t           Recording duration in seconds (default: 30)
```

### Ultrasonic Distance Sensing

**Single distance measurement:**
```bash
python ./pathpal_project/UltrasonicSensor.py
```

**Continuous measurements (10 seconds):**
Edit `UltrasonicSensor.py` and uncomment the continuous measurement section in `main()`.

**Features:**
- Serial protocol: 0xFF + H_DATA + L_DATA + SUM
- Baud rate: 9600
- Data format: 16-bit distance in millimeters
- Automatic checksum validation

## Configuration

### Camera Settings
- Device: `/dev/video11` (Radxa ISP mainpath)
- Resolution: 640x480 (adjustable)
- Autofocus: Enabled by default (Radxa ISP driver)
- Auto exposure and white balance: Enabled

### Sensor Settings
- UART Port: 0 (default)
- Baud rate: 9600
- Data bits: 8
- Stop bits: 1
- Parity: None

## Dependencies

Key dependencies installed via requirements.txt:
- **OpenCV** (opencv-python, opencv-contrib-python): Computer vision
- **PyTorch** (torch): Deep learning framework
- **YOLOv8**: Object detection
- **NumPy, SciPy, scikit-image**: Scientific computing
- **Pillow**: Image processing
- **MRAA**: Hardware control (GPIO, SPI, I2C, UART)
- **v4l-utils**: Video device control

See `requirements.txt` for complete list.

## Troubleshooting

### Camera not found
```bash
# List available video devices
v4l2-ctl -d /dev/video11 --all

# Check if /dev/video11 is accessible
ls -la /dev/video11
```

### Display error (Qt platform plugin)
Run in headless mode:
```bash
python ./pathpal_project/CameraTest.py --headless
```

### UART connection issues
Check UART device:
```bash
ls -la /dev/ttyS0  # or appropriate UART device
```

## Development

To activate the virtual environment:
```bash
source venv/bin/activate
```

To deactivate:
```bash
deactivate
```

To update dependencies:
```bash
pip freeze > requirements.txt
```

## License

[Add your license here]

## Support

For issues and questions, refer to:
- Radxa documentation: https://docs.radxa.com/
- YOLOv8 documentation: https://docs.ultralytics.com/

