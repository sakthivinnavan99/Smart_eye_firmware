#!/usr/bin/env python3
"""
Test UART6 serial port on GPIO1_D1 (TX) / GPIO1_D0 (RX).

Tests:
  1. Device existence
  2. Open/close at various baud rates
  3. Loopback test (requires TX wired to RX)
  4. Transmit-only test (no loopback needed)
  5. Receive test (waits for external data)

Run with: sudo python3 test_uart6.py
"""

import os
import sys
import time
import termios
import select

UART_DEV = "/dev/ttyS6"
DEFAULT_BAUD = 9600

BAUD_MAP = {
    1200: termios.B1200,
    2400: termios.B2400,
    4800: termios.B4800,
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
}


class UARTDev:
    """Minimal UART access without pyserial dependency."""

    def __init__(self, device, baudrate=9600):
        self.device = device
        self.baudrate = baudrate
        self.fd = None

    def open(self):
        self.fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)

        baud_const = BAUD_MAP.get(self.baudrate)
        if baud_const is None:
            raise ValueError("Unsupported baud rate: {}".format(self.baudrate))

        # 8N1, no flow control
        attrs[0] = 0  # iflag
        attrs[1] = 0  # oflag
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD  # cflag
        attrs[3] = 0  # lflag
        attrs[4] = baud_const  # ispeed
        attrs[5] = baud_const  # ospeed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 10  # 1 second timeout

        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write(self, data):
        if isinstance(data, str):
            data = data.encode()
        return os.write(self.fd, data)

    def read(self, size=256, timeout=2.0):
        result = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if ready:
                chunk = os.read(self.fd, size - len(result))
                if chunk:
                    result += chunk
                if len(result) >= size:
                    break
        return result

    def flush(self):
        termios.tcflush(self.fd, termios.TCIOFLUSH)


def test_device_exists():
    print("\n--- Test 1: UART6 Device ---")
    if os.path.exists(UART_DEV):
        print("  [PASS] {} exists".format(UART_DEV))
        return True
    else:
        print("  [FAIL] {} not found. Is UART6 overlay enabled?".format(UART_DEV))
        return False


def test_open_close():
    print("\n--- Test 2: Open/Close at Various Baud Rates ---")
    success = 0
    for baud in sorted(BAUD_MAP.keys()):
        uart = UARTDev(UART_DEV, baud)
        try:
            uart.open()
            uart.close()
            print("  {:>7d} baud: OK".format(baud))
            success += 1
        except Exception as e:
            print("  {:>7d} baud: FAIL ({})".format(baud, e))

    if success == len(BAUD_MAP):
        print("  [PASS] All {} baud rates opened successfully".format(success))
    else:
        print("  [WARN] {}/{} baud rates succeeded".format(success, len(BAUD_MAP)))


def test_loopback():
    print("\n--- Test 3: Loopback Test (TX -> RX) ---")
    print("  NOTE: Requires TX (GPIO1_D1) connected to RX (GPIO1_D0)")

    uart = UARTDev(UART_DEV, DEFAULT_BAUD)
    try:
        uart.open()
        uart.flush()

        test_data = b"SMART_EYE_LOOPBACK_TEST_1234\r\n"
        written = uart.write(test_data)
        print("  Sent {} bytes: {}".format(written, test_data.strip().decode()))

        time.sleep(0.1)
        received = uart.read(len(test_data), timeout=2.0)

        if received == test_data:
            print("  Received: {}".format(received.strip().decode()))
            print("  [PASS] Loopback data matches perfectly")
        elif len(received) > 0:
            print("  Received {} bytes: {}".format(len(received), received))
            if test_data.strip() in received:
                print("  [PASS] Loopback data found in response")
            else:
                print("  [WARN] Received data differs from sent data")
        else:
            print("  Received: (nothing)")
            print("  [INFO] No loopback data - TX/RX not connected? This is normal if")
            print("         you have a device connected instead of a loopback wire.")

    except Exception as e:
        print("  [FAIL] Loopback test error: {}".format(e))
    finally:
        uart.close()


def test_transmit():
    print("\n--- Test 4: Transmit Test ---")
    uart = UARTDev(UART_DEV, DEFAULT_BAUD)
    try:
        uart.open()

        messages = [
            b"Smart Eye UART6 Test\r\n",
            b"Hello from Radxa CM5\r\n",
            bytes(range(256)),
        ]

        for i, msg in enumerate(messages):
            written = uart.write(msg)
            if i < 2:
                label = msg.strip().decode()
            else:
                label = "256 binary bytes (0x00-0xFF)"
            print("  Packet {}: sent {} bytes - {}".format(i + 1, written, label))
            time.sleep(0.05)

        print("  [PASS] All packets transmitted")
        print("  Verify with oscilloscope or USB-serial adapter on TX (GPIO1_D1)")

    except Exception as e:
        print("  [FAIL] Transmit error: {}".format(e))
    finally:
        uart.close()


def test_receive(duration=5):
    print("\n--- Test 5: Receive Test ({}s window) ---".format(duration))
    print("  Send data to RX (GPIO1_D0) now...")

    uart = UARTDev(UART_DEV, DEFAULT_BAUD)
    try:
        uart.open()
        uart.flush()

        total_bytes = 0
        start = time.time()
        while time.time() - start < duration:
            data = uart.read(256, timeout=1.0)
            if data:
                total_bytes += len(data)
                printable = data.decode("ascii", errors="replace").strip()
                print("  Received {} bytes: {}".format(len(data), repr(printable[:60])))

            remaining = duration - (time.time() - start)
            if remaining > 0:
                print("  Waiting... ({:.0f}s remaining)".format(remaining), end="\r")

        print("                                    ")
        if total_bytes > 0:
            print("  [PASS] Received {} bytes total".format(total_bytes))
        else:
            print("  [INFO] No data received (normal if nothing is connected to RX)")

    except Exception as e:
        print("  [FAIL] Receive error: {}".format(e))
    finally:
        uart.close()


def main():
    if os.geteuid() != 0:
        print("This test requires root. Run with: sudo python3 test_uart6.py")
        sys.exit(1)

    print("========================================")
    print(" UART6 Serial Port Test")
    print(" Device: {}".format(UART_DEV))
    print(" TX: GPIO1_D1 | RX: GPIO1_D0")
    print(" Default: {} 8N1".format(DEFAULT_BAUD))
    print("========================================")

    if not test_device_exists():
        sys.exit(1)

    try:
        test_open_close()
        test_loopback()
        test_transmit()
        test_receive()

        print("\n========================================")
        print(" All UART6 tests complete!")
        print("========================================")
    except KeyboardInterrupt:
        print("\n\nInterrupted!")


if __name__ == "__main__":
    main()
