"""
Live Object Detection using YOLOv8 RKNN Model
Displays live video feed with object detection and FPS
"""

import cv2
import time
import os
import sys
import numpy as np
import logging
from datetime import datetime

# Configure logging to suppress warnings from RKNN and other libraries
logging.basicConfig(level=logging.ERROR)
logging.getLogger('rknn').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress specific warning about logging levels
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

# Import YOLOv8 functions
from yolov8 import (
    setup_model, post_process, draw, IMG_SIZE, CLASSES
)
from py_utils.coco_utils import COCO_test_helper


class ObjectDetection:
    def __init__(self, model_path, device_path="/dev/video11", target='rk3588', device_id=None, fps=30, headless=False):
        """
        Initialize Object Detection with camera and model
        
        Args:
            model_path: Path to YOLOv8 RKNN model file
            device_path: Path to video device (default /dev/video11)
            target: Target RKNPU platform (default rk3566)
            device_id: Device ID for RKNN (default None)
            fps: Desired frames per second (default 30)
            headless: Run without display window (default False)
        """
        self.model_path = model_path
        self.device_path = device_path
        self.fps = fps
        self.headless = headless
        self.cap = None
        self.model = None
        self.platform = None
        self.co_helper = None
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        self.inference_time = 0
        
        # Initialize model and camera
        self._initialize_model(target, device_id)
        self._initialize_camera()
    
    def _initialize_model(self, target, device_id):
        """Initialize YOLOv8 model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"Loading model from {self.model_path}...")
            print(f"Target platform: {target} (using NPU)")
            self.model, self.platform = setup_model(self.model_path, target, device_id)
            self.co_helper = COCO_test_helper(enable_letter_box=True)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _initialize_camera(self):
        """Initialize camera capture"""
        try:
            # Open video device
            device_number = int(self.device_path.replace('/dev/video', ''))
            self.cap = cv2.VideoCapture(device_number)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera device: {self.device_path}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            # Get actual properties
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
        """
        Preprocess frame for model inference using full video scale (direct resize)
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            input_data: Preprocessed input for model
            original_frame: Original frame for drawing results
        """
        # Store original frame dimensions for coordinate conversion
        self.original_shape = frame.shape[:2]  # (height, width)
        
        # Direct resize to model input size (full video scale, no padding)
        # This stretches the image to fit 640x640, using the full frame
        img = self.co_helper.direct_resize(
            im=frame.copy(), 
            new_shape=(IMG_SIZE[1], IMG_SIZE[0])  # (height, width)
        )
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare input based on platform
        if self.platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.0
        elif self.platform in ['rknn']:
            # RKNN expects HWC format with batch dimension
            input_data = np.expand_dims(img, 0)
        else:
            input_data = img
        
        return input_data, frame
    
    def _detect_objects(self, input_data):
        """
        Run object detection on preprocessed input
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            boxes: Detected bounding boxes (or None)
            classes: Detected class indices (or None)
            scores: Detection confidence scores (or None)
        """
        try:
            # Run inference
            inference_start = time.time()
            outputs = self.model.run([input_data])
            self.inference_time = time.time() - inference_start
            
            # Post-process results
            boxes, classes, scores = post_process(outputs)
            
            # Print detected objects
            if boxes is not None and len(boxes) > 0:
                print(f"\n[Frame {self.frame_count}] Detected {len(boxes)} object(s):")
                for i, (box, cls_idx, score) in enumerate(zip(boxes, classes, scores)):
                    class_name = CLASSES[int(cls_idx)] if int(cls_idx) < len(CLASSES) else "Unknown"
                    x1, y1, x2, y2 = box
                    print(f"  [{i+1}] {class_name} - Confidence: {score:.2%} - Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            return boxes, classes, scores
        except Exception as e:
            # Only print non-warning-level errors
            error_str = str(e).lower()
            if 'unknown level' not in error_str and 'warning' not in error_str:
                print(f"Error during detection: {e}")
            return None, None, None
    
    def _draw_detections(self, frame, boxes, classes, scores):
        """
        Draw detection results on frame
        
        Args:
            frame: Original frame to draw on
            boxes: Bounding boxes in model coordinates
            classes: Class indices
            scores: Confidence scores
            
        Returns:
            frame: Frame with detections drawn
        """
        if boxes is not None and len(boxes) > 0:
            # Convert boxes from model coordinates to original frame coordinates
            real_boxes = self.co_helper.get_real_box(boxes)
            # Draw detections
            draw(frame, real_boxes, scores, classes, verbose=False)
        return frame
    
    def _draw_info_overlay(self, frame):
        """
        Draw information overlay (FPS, timestamp, etc.)
        
        Args:
            frame: Frame to draw on
            
        Returns:
            frame: Frame with overlay
        """
        # Calculate actual FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.actual_fps = self.frame_count / elapsed
        
        # Draw FPS
        fps_text = f"FPS: {self.actual_fps:.1f}"
        inference_text = f"Inference: {self.inference_time*1000:.1f}ms"
        
        # Background rectangle for better visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (280, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, inference_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_detection(self, window_name="Object Detection - Live Feed"):
        """
        Start live object detection
        
        Args:
            window_name: Name of the display window
        """
        try:
            print(f"\nStarting live object detection from {self.device_path}")
            if not self.headless:
                print("Press 'q' to quit, 's' to save current frame")
            print("-" * 50)
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Preprocess frame
                input_data, original_frame = self._preprocess_frame(frame)
                
                # Detect objects
                boxes, classes, scores = self._detect_objects(input_data)
                
                # Draw detections
                frame = self._draw_detections(original_frame, boxes, classes, scores)
                
                # Draw info overlay
                frame = self._draw_info_overlay(frame)
                
                # Display frame (skip in headless mode)
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
            print(f"\nDetection session ended")
            print(f"Total frames processed: {self.frame_count}")
            if self.frame_count > 0:
                print(f"Average FPS: {self.actual_fps:.1f}")
    
    def close(self):
        """Release camera and model resources"""
        try:
            if self.cap:
                self.cap.release()
                print("Camera released successfully")
        except Exception as e:
            print(f"Error closing camera: {e}")


def main():
    """Main function for object detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Object Detection using YOLOv8')
    parser.add_argument('--model_path', type=str, 
                       default='models/yolov8/yolov8.rknn',
                       help='Path to YOLOv8 RKNN model file')
    parser.add_argument('--device', type=str, default='/dev/video11',
                       help='Camera device path (default: /dev/video11)')
    parser.add_argument('--target', type=str, default='rk3588',
                       help='Target RKNPU platform (default: rk3588 for Radxa CM5 IO)')
    parser.add_argument('--device_id', type=str, default=None,
                       help='Device ID for RKNN (default: None)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Desired FPS (default: 30)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without GUI display')
    
    args = parser.parse_args()
    
    try:
        # Resolve model path relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        if not os.path.isabs(args.model_path):
            model_path = os.path.join(project_root, args.model_path)
        else:
            model_path = args.model_path
        
        # Initialize and start detection
        detector = ObjectDetection(
            model_path=model_path,
            device_path=args.device,
            target=args.target,
            device_id=args.device_id,
            fps=args.fps,
            headless=args.headless
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

