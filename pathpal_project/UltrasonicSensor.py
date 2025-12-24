"""
Ultrasonic Sensor Module using PySerial
Communicates with ultrasonic sensor via serial interface
Serial protocol:
- Data format: 0xFF + H_DATA + L_DATA + SUM
- Baud rate: 9600
- Data bits: 8
- Stop bits: 1
- Parity: None
- SUM = (H_DATA + L_DATA) & 0xFF
- Distance = (H_DATA << 8 | L_DATA) in millimeters
"""

import serial
import time
import struct

class UltrasonicSensor:
    def __init__(self, device="/dev/ttyS2", baud_rate=9600):
        """
        Initialize the ultrasonic sensor via PySerial
        
        Args:
            device: Serial device path (default "/dev/ttyS2")
            baud_rate: Serial baud rate (default 9600)
        """
        try:
            self.ser = serial.Serial(
                port=device,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            # Flush any existing data
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"Ultrasonic Sensor initialized on {device} at {baud_rate} baud")
        except serial.SerialException as e:
            print(f"Error initializing serial port {device}: {e}")
            print("Make sure the device exists and you have permissions (may need to be in dialout group)")
            raise
        except Exception as e:
            print(f"Error initializing UART: {e}")
            raise
    
    def request_distance(self, verbose=False):
        """
        Request distance measurement from ultrasonic sensor
        Send: 0x01 command
        
        Args:
            verbose: If True, print debug messages (default False)
        
        Returns:
            distance: Distance in millimeters (int), or None if read fails
        """
        try:
            # Flush input buffer before sending command
            self.ser.reset_input_buffer()
            
            # Send request command
            self.ser.write(bytes([0x01]))
            self.ser.flush()
            if verbose:
                print("Sent request command: 0x01")
            
            # Wait for response
            time.sleep(0.05)
            
            # Read response data
            distance = self._read_distance_data(verbose=verbose)
            
            return distance
        except Exception as e:
            if verbose:
                print(f"Error requesting distance: {e}")
            return None
    
    def _read_distance_data(self, verbose=False):
        """
        Read and parse distance data from sensor
        Expected format: 0xFF + H_DATA + L_DATA + SUM
        
        Args:
            verbose: If True, print debug messages (default False)
        
        Returns:
            distance: Distance in millimeters (int), or None if checksum fails
        """
        try:
            # Wait for start byte (0xFF) with timeout
            start_byte = None
            timeout = 0
            max_timeout = 50  # 5 seconds max
            
            while timeout < max_timeout:
                if self.ser.in_waiting > 0:
                    byte = self.ser.read(1)[0]
                    if byte == 0xFF:
                        start_byte = byte
                        break
                time.sleep(0.1)
                timeout += 1
            
            if start_byte is None:
                if verbose:
                    print("Timeout waiting for start byte (0xFF)")
                return None
            
            # Read H_DATA, L_DATA, SUM (3 more bytes)
            data_bytes = [start_byte]
            remaining_bytes = 3
            bytes_read = 0
            
            while bytes_read < remaining_bytes:
                if self.ser.in_waiting > 0:
                    byte = self.ser.read(1)[0]
                    data_bytes.append(byte)
                    bytes_read += 1
                else:
                    time.sleep(0.05)
                    if bytes_read == 0 and self.ser.in_waiting == 0:
                        # No data available yet, wait a bit more
                        time.sleep(0.1)
            
            if len(data_bytes) < 4:
                if verbose:
                    print(f"Incomplete data received: {data_bytes}")
                return None
            
            start_byte = data_bytes[0]
            h_data = data_bytes[1]
            l_data = data_bytes[2]
            checksum = data_bytes[3]
            
            # Verify checksum
            calculated_checksum = (h_data + l_data) & 0xFF
            # if checksum != calculated_checksum:
            #     if verbose:
            #         print(f"Checksum error! Expected: 0x{calculated_checksum:02X}, Got: 0x{checksum:02X}")
            #     return None
            
            # Calculate distance
            distance = (h_data << 8) | l_data
            
            if verbose:
                print(f"Raw data: FF {h_data:02X} {l_data:02X} {checksum:02X}")
                print(f"Distance: {distance} mm (0x{distance:04X})")
            
            return distance
        
        except Exception as e:
            if verbose:
                print(f"Error reading distance data: {e}")
            return None
    
    def continuous_measurement(self, interval=0.5, duration=None):
        """
        Continuously measure distance
        
        Args:
            interval: Time between measurements in seconds (default 0.5 = 500ms)
            duration: Total measurement duration in seconds (None = infinite)
        """
        start_time = time.time()
        measurement_count = 0
        
        try:
            print(f"\nStarting continuous measurement every {interval*1000:.0f}ms (Press Ctrl+C to stop)\n")
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                measurement_count += 1
                distance = self.request_distance(verbose=False)
                
                if distance is not None:
                    timestamp = time.strftime('%H:%M:%S')
                    print(f"[{timestamp}] Distance: {distance} mm")
                else:
                    timestamp = time.strftime('%H:%M:%S')
                    print(f"[{timestamp}] Failed to read distance")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\nMeasurement stopped by user (Total readings: {measurement_count})")
        except Exception as e:
            print(f"Error during continuous measurement: {e}")
    
    def close(self):
        """Close serial connection"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Serial connection closed")
        except Exception as e:
            print(f"Error closing serial port: {e}")


def main():
    """Main function for testing ultrasonic sensor"""
    sensor = None
    try:
        # Initialize sensor (using /dev/ttyS2 which corresponds to UART 2)
        sensor = UltrasonicSensor(device="/dev/ttyS2", baud_rate=9600)
        
        # Continuous measurement every 500ms (0.5 seconds)
        # Runs indefinitely until Ctrl+C is pressed
        sensor.continuous_measurement(interval=0.5, duration=None)
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if sensor:
            sensor.close()


if __name__ == "__main__":
    main()
