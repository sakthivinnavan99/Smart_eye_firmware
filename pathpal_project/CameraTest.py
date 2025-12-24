"""
Camera Test Module
Displays live video feed from /dev/video11 device
Uses OpenCV for video capture and display
"""

import cv2
import time
import sys
from datetime import datetime


class CameraTest:
    def __init__(self, device_path="/dev/video11", fps=30):
        """+
        Initialize camera from device
        
        Args:
            device_path: Path to video device (default /dev/video11)
            fps: Desired frames per second (default 30)
        """
        self.device_path = device_path
        self.fps = fps
        self.cap = None
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera capture"""
        try:
            # Open video device
            device_number = int(self.device_path.replace('/dev/video', ''))
            self.cap = cv2.VideoCapture(device_number)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera device: {self.device_path}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
    
    def _draw_info(self, frame):
        """
        Draw information overlay on frame
        
        Args:
            frame: Input frame to draw on
            
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
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw resolution
        height, width = frame.shape[:2]
        res_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, res_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def start_live_feed(self, window_name="Camera Feed - /dev/video11"):
        """
        Display live camera feed in a window
        
        Args:
            window_name: Name of the display window
            
        Returns:
            total_frames: Total frames captured before exit
        """
        try:
            print(f"\nStarting live feed from {self.device_path}")
            print("Press 'q' to quit, 's' to save frame")
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Draw information overlay
                frame = self._draw_info(frame)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    filename = f"frame_{self.frame_count}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved: {filename}")
        
        except Exception as e:
            print(f"Error during live feed: {e}")
        
        finally:
            self.close()
            cv2.destroyAllWindows()
            print(f"\nTotal frames captured: {self.frame_count}")
            print(f"Average FPS: {self.actual_fps:.1f}")
            return self.frame_count
    
    def capture_frame(self):
        """
        Capture a single frame from camera
        
        Returns:
            frame: Captured frame or None if failed
        """
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Failed to capture frame")
                return None
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def record_video(self, output_path="output.mp4", duration=30):
        """
        Record video from camera
        
        Args:
            output_path: Path to save video file
            duration: Recording duration in seconds
        """
        try:
            # Get frame properties
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                 (frame_width, frame_height))
            
            print(f"Recording to {output_path} for {duration} seconds...")
            
            start_time = time.time()
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                out.write(frame)
                frame_count += 1
                
                elapsed = time.time() - start_time
                if elapsed > duration:
                    break
                
                # Print progress
                if frame_count % 30 == 0:
                    print(f"Recorded {frame_count} frames ({elapsed:.1f}s)")
            
            out.release()
            print(f"Recording saved! Total frames: {frame_count}")
        
        except Exception as e:
            print(f"Error recording video: {e}")
    
    def close(self):
        """Release camera resources"""
        try:
            if self.cap:
                self.cap.release()
                print(f"Camera released successfully")
        except Exception as e:
            print(f"Error closing camera: {e}")


def main():
    """Main function for camera testing"""
    try:
        # Initialize camera
        camera = CameraTest(device_path="/dev/video11", fps=30)
        
        # Start live feed
        camera.start_live_feed()
        
        # Uncomment below to record video instead (30 seconds)
        # camera.record_video(output_path="camera_recording.mp4", duration=30)
        # camera.close()
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
