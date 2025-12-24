"""
Live Video Detection with GPU/NPU Usage Monitoring
Captures live video feed from camera, processes with YOLOv8 RKNN model,
and displays detected objects with FPS and system resource usage
# Higher FPS mode
python ./pathpal_project/LiveVideoDetection.py --model_path ./models/yolov8/yolov8.rknn

# Headless for maximum performance
python ./pathpal_project/LiveVideoDetection.py --model_path ./models/yolov8/yolov8.rknn --headless
"""

import cv2
import time
import os
import sys
import numpy as np
import argparse
import logging
import threading
from datetime import datetime
from collections import deque

# Configure logging
logging.basicConfig(level=logging.ERROR)
for logger_name in ['torch', 'torchvision', 'rknn', 'tensorflow']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Try to import psutil for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add path for imports
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('pathpal_project')+1]))

from yolov8 import (
    setup_model, post_process, draw, IMG_SIZE, CLASSES
)
from py_utils.coco_utils import COCO_test_helper


def load_custom_classes(classes_file):
    """
    Load custom class labels from a file.
    File format: one class name per line
    
    Args:
        classes_file: Path to classes file
    
    Returns:
        Tuple of class names, or None if file doesn't exist
    """
    if not classes_file or not os.path.exists(classes_file):
        return None
    
    try:
        with open(classes_file, 'r') as f:
            classes = tuple(line.strip() for line in f.readlines() if line.strip())
        print(f"Loaded {len(classes)} custom classes from {classes_file}")
        return classes
    except Exception as e:
        print(f"Error loading classes file: {e}")
        return None


class LiveVideoDetector:
    def __init__(self, model_path, device_path="/dev/video11", target='rk3588', 
                 device_id=None, fps=30, headless=False, confidence_threshold=0.25, custom_classes=None):
        """
        Initialize Live Video Detector with optimizations for high FPS
        
        Args:
            model_path: Path to YOLOv8 RKNN model file
            device_path: Path to video device (default /dev/video11)
            target: Target RKNPU platform (default rk3588)
            device_id: Device ID for RKNN (default None)
            fps: Desired frames per second (default 30)
            headless: Run without display window (default False)
            confidence_threshold: Confidence threshold for detections (default 0.25)
            custom_classes: Tuple of custom class names (optional)
        """
        self.model_path = model_path
        self.device_path = device_path
        self.fps = fps
        self.headless = headless
        self.confidence_threshold = confidence_threshold
        self.custom_classes = custom_classes
        self.cap = None
        self.model = None
        self.platform = None
        self.co_helper = None
        
        # Metrics tracking
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        self.inference_time = 0
        self.total_detections = 0
        
        # FPS history for smoothing
        self.fps_history = deque(maxlen=30)
        self.inference_history = deque(maxlen=30)
        
        # Thread-safe frame buffer
        self.frame_buffer = None
        self.buffer_lock = threading.Lock()
        self.stop_thread = False
        
        # System stats (updated periodically)
        self.last_cpu_check = time.time()
        self.cpu_percent = 0
        self.mem_percent = 0
        self.stats_update_interval = 0.5  # Update stats every 0.5s
        
        # Initialize model and camera
        self._initialize_model(target, device_id)
        self._initialize_camera()
        
        # Start background frame capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
    
    def _initialize_model(self, target, device_id):
        """Initialize YOLOv8 model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"Loading model from {self.model_path}...")
            print(f"Target platform: {target}")
            self.model, self.platform = setup_model(self.model_path, target, device_id)
            self.co_helper = COCO_test_helper(enable_letter_box=True)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _initialize_camera(self):
        """Initialize camera capture"""
        try:
            device_number = int(self.device_path.replace('/dev/video', ''))
            self.cap = cv2.VideoCapture(device_number)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera device: {self.device_path}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized successfully!")
            print(f"Device: {self.device_path}")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {actual_fps}")
            
            self.start_time = time.time()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for model inference"""
        self.original_shape = frame.shape[:2]
        
        img = self.co_helper.direct_resize(
            im=frame.copy(), 
            new_shape=(IMG_SIZE[1], IMG_SIZE[0])
        )
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.0
        elif self.platform in ['rknn']:
            input_data = np.expand_dims(img, 0)
        else:
            input_data = img
        
        return input_data, frame
    
    def _detect_objects(self, input_data):
        """Run object detection on preprocessed input"""
        try:
            inference_start = time.time()
            outputs = self.model.run([input_data])
            inference_elapsed = time.time() - inference_start
            self.inference_time = inference_elapsed
            self.inference_history.append(inference_elapsed)
            
            boxes, classes, scores = post_process(outputs)
            
            # Filter by confidence threshold
            if boxes is not None and len(boxes) > 0:
                mask = scores >= self.confidence_threshold
                boxes = boxes[mask]
                classes = classes[mask]
                scores = scores[mask]
                
                if len(boxes) > 0:
                    self.total_detections += len(boxes)
                    # Only print detections every 30 frames to reduce overhead
                    # if self.frame_count % 30 == 0:
                        # print(f"Frame {self.frame_count}: {len(boxes)} objects")
                    return boxes, classes, scores
            
            return None, None, None
        except Exception as e:
            error_str = str(e).lower()
            if 'unknown level' not in error_str and 'warning' not in error_str:
                print(f"Error during detection: {e}")
                import traceback
                traceback.print_exc()
            return None, None, None
    
    def _draw_detections(self, frame, boxes, classes, scores):
        """Draw detection results on frame"""
        if boxes is not None and len(boxes) > 0:
            real_boxes = self.co_helper.get_real_box(boxes)
            # Use custom classes if provided, otherwise use default CLASSES
            class_labels = self.custom_classes if self.custom_classes else CLASSES
            draw(frame, real_boxes, scores, classes, verbose=False, class_labels=class_labels)
        return frame
    
    def _get_system_stats(self):
        """Get CPU and memory usage (non-blocking, cached)"""
        try:
            now = time.time()
            # Only update stats every N seconds to avoid blocking
            if now - self.last_cpu_check >= self.stats_update_interval:
                if PSUTIL_AVAILABLE:
                    # Non-blocking, fast check
                    self.cpu_percent = psutil.cpu_percent(interval=None)
                    self.mem_percent = psutil.virtual_memory().percent
                    self.last_cpu_check = now
            return self.cpu_percent, self.mem_percent
        except Exception:
            return None, None
    
    def _draw_overlay(self, frame):
        """Draw information overlay with metrics (optimized for speed)"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 0:
            self.fps_history.append(1.0 / max(0.001, self.inference_time))
            # Smooth FPS over last 30 frames
            if self.fps_history:
                self.actual_fps = np.mean(list(self.fps_history))
        
        # Get system stats (non-blocking, cached)
        cpu_percent, mem_percent = self._get_system_stats()
        
        # Prepare overlay text
        avg_inference = np.mean(list(self.inference_history)) * 1000 if self.inference_history else 0
        
        overlay_texts = [
            f"FPS: {self.actual_fps:.1f}",
            f"Inference: {avg_inference:.1f}ms",
        ]
        
        if cpu_percent is not None:
            overlay_texts.append(f"CPU: {cpu_percent:.0f}%")
        if mem_percent is not None:
            overlay_texts.append(f"Mem: {mem_percent:.0f}%")
        
        # Only update timestamp every 10 frames to reduce overhead
        if self.frame_count % 10 == 0:
            self.last_timestamp = datetime.now().strftime("%H:%M:%S")
        
        overlay_texts.extend([
            f"Frame: {self.frame_count}",
            f"Objects: {self.total_detections}",
            self.last_timestamp if hasattr(self, 'last_timestamp') else ""
        ])
        
        # Draw background (less transparent for speed)
        overlay = frame.copy()
        box_height = 20 + len(overlay_texts) * 20
        cv2.rectangle(overlay, (5, 5), (300, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw text (smaller font for speed)
        colors = [
            (0, 255, 0),      # FPS
            (0, 255, 255),    # Inference
            (255, 165, 0),    # CPU
            (255, 0, 0),      # Memory
            (255, 255, 255),  # Frame
            (200, 200, 200),  # Objects
            (150, 150, 255)   # Timestamp
        ]
        
        for i, text in enumerate(overlay_texts):
            if text:  # Skip empty strings
                cv2.putText(frame, text, (10, 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)
        
        return frame
    
    def _capture_frames(self):
        """Background thread for continuous frame capture (non-blocking)"""
        while not self.stop_thread:
            ret, frame = self.cap.read()
            if ret:
                with self.buffer_lock:
                    self.frame_buffer = frame
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def start_detection(self, window_name="Live Video Detection"):
        """Start live video detection (optimized for high FPS)"""
        try:
            print(f"\nStarting live video detection from {self.device_path}")
            if not self.headless:
                print("Press 'q' to quit, 's' to save frame")
            print(f"Target FPS: {self.fps}, Confidence: {self.confidence_threshold}")
            print("-" * 50)
            
            while True:
                # Get latest frame from buffer (non-blocking)
                with self.buffer_lock:
                    frame = self.frame_buffer
                
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Preprocess and detect
                input_data, original_frame = self._preprocess_frame(frame)
                boxes, classes, scores = self._detect_objects(input_data)
                
                # Draw detections
                frame = self._draw_detections(original_frame, boxes, classes, scores)
                
                # Draw overlay with metrics
                frame = self._draw_overlay(frame)
                
                # Display (skip in headless mode)
                if not self.headless:
                    cv2.imshow(window_name, frame)
                
                # Handle key press (only if not headless)
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        filename = f"detection_{self.frame_count}_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.close()
            if not self.headless:
                cv2.destroyAllWindows()
            self._print_summary()
    
    def _print_summary(self):
        """Print detection session summary"""
        print(f"\n{'='*50}")
        print(f"Detection session ended")
        print(f"{'='*50}")
        print(f"Total frames processed: {self.frame_count}")
        if self.frame_count > 0:
            print(f"Average FPS: {self.actual_fps:.1f}")
        print(f"Total objects detected: {self.total_detections}")
        if self.frame_count > 0:
            print(f"Average detections per frame: {self.total_detections/self.frame_count:.2f}")
        print(f"{'='*50}\n")
    
    def close(self):
        """Release resources"""
        try:
            self.stop_thread = True
            self.capture_thread.join(timeout=1.0)
            if self.cap:
                self.cap.release()
                print("Camera released successfully")
        except Exception as e:
            print(f"Error closing camera: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Live Video Detection with System Metrics')
    parser.add_argument('--model_path', type=str, 
                       default='models/yolov8/yolov8.rknn',
                       help='Path to YOLOv8 RKNN model file')
    parser.add_argument('--device', type=str, default='/dev/video11',
                       help='Camera device path (default: /dev/video11)')
    parser.add_argument('--target', type=str, default='rk3588',
                       help='Target RKNPU platform (default: rk3588)')
    parser.add_argument('--device_id', type=str, default=None,
                       help='Device ID for RKNN (default: None)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Desired FPS (default: 30)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without GUI display')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--classes', type=str, default=None,
                       help='Path to custom classes file (one class per line). If not provided, uses COCO classes.')
    
    args = parser.parse_args()
    
    try:
        # Resolve model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        if not os.path.isabs(args.model_path):
            model_path = os.path.join(project_root, args.model_path)
        else:
            model_path = args.model_path
        
        # Load custom classes if provided
        custom_classes = load_custom_classes(args.classes) if args.classes else None
        
        # Initialize and start detection
        detector = LiveVideoDetector(
            model_path=model_path,
            device_path=args.device,
            target=args.target,
            device_id=args.device_id,
            fps=args.fps,
            headless=args.headless,
            confidence_threshold=args.confidence,
            custom_classes=custom_classes
        )
        
        detector.start_detection()
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
