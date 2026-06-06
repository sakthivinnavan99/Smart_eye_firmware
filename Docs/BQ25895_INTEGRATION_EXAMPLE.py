#!/usr/bin/env python3
"""
BQ25895 Battery Charger Integration Example

Shows how to integrate BQ25895 configuration and monitoring into the
SmartEyeApp main application loop.

This is an example/reference - copy relevant sections into main.py
"""

import logging
import time
from pathpal_project.bq25895_charger import BQ25895


log = logging.getLogger("smart-eye")


class BatteryChargerMonitor:
    """Wraps BQ25895 charger with status monitoring and fault handling."""

    # Status monitoring interval (seconds)
    STATUS_CHECK_INTERVAL = 5

    # Alert thresholds
    THERMAL_WARNING_THRESHOLD = 100  # Celsius

    def __init__(self):
        """Initialize charger and apply 10Ah battery configuration."""
        try:
            self.charger = BQ25895()
            self.charger.configure_for_10ah_battery()
            self.last_status_check = 0
            self.fault_history = {}  # Track persistent faults
            log.info("Battery charger initialized for 10000mAh battery")
        except Exception as e:
            log.error(f"Failed to initialize battery charger: {e}")
            self.charger = None

    def check_status(self):
        """Check charger status and handle faults.

        Returns:
            dict: Charger status if available, else None
        """
        if self.charger is None:
            return None

        try:
            status = self.charger.get_status()
            self._handle_faults(status)
            return status
        except Exception as e:
            log.warning(f"Failed to read charger status: {e}")
            return None

    def _handle_faults(self, status):
        """Analyze status and handle fault conditions.

        Args:
            status: Status dictionary from charger.get_status()
        """
        # Thermal shutdown
        if status['thermal_shutdown']:
            self._count_fault('thermal_shutdown')
            if self.fault_history['thermal_shutdown'] == 1:
                log.warning(
                    "[CHARGER] Thermal shutdown detected! "
                    "Reduce load or check heatsinking."
                )
            if self.fault_history['thermal_shutdown'] > 5:
                log.error(
                    "[CHARGER] Persistent thermal shutdown! "
                    "Device may overheat - consider emergency shutdown."
                )

        # Battery overvoltage
        if status['battery_overvoltage']:
            self._count_fault('battery_overvoltage')
            if self.fault_history['battery_overvoltage'] == 1:
                log.error(
                    "[CHARGER] Battery overvoltage protection triggered! "
                    "Battery may be damaged or charger misconfigured."
                )

        # Input overvoltage
        if status['input_overvoltage']:
            self._count_fault('input_overvoltage')
            if self.fault_history['input_overvoltage'] == 1:
                log.error(
                    "[CHARGER] Input overvoltage detected! "
                    "Power supply may be faulty."
                )

        # Charger fault
        if status['charger_fault'] != 0:
            self._count_fault(f'charger_fault_{status["charger_fault"]}')
            fault_desc = {
                1: "Input Fault",
                2: "Thermal Shutdown",
                3: "Charge Timer Fault",
            }
            log.error(
                f"[CHARGER] Charger fault: "
                f"{fault_desc.get(status['charger_fault'], 'Unknown')}"
            )

        # Battery fault
        if status['battery_fault']:
            self._count_fault('battery_fault')
            if self.fault_history['battery_fault'] == 1:
                log.error("[CHARGER] Battery fault detected!")

        # NTC (temperature sensor) fault
        if status['ntc_fault'] != 0:
            self._count_fault(f'ntc_fault_{status["ntc_fault"]}')
            ntc_desc = {
                1: "TS Cold",
                2: "TS Cool/Thermal",
                3: "TS Warm",
            }
            log.warning(
                f"[CHARGER] NTC fault: "
                f"{ntc_desc.get(status['ntc_fault'], 'Unknown')}"
            )

    def _count_fault(self, fault_key):
        """Track fault occurrences.

        Args:
            fault_key: Identifier for the fault type
        """
        if fault_key not in self.fault_history:
            self.fault_history[fault_key] = 0
        self.fault_history[fault_key] += 1

    def log_status(self, status):
        """Log charger status for debugging.

        Args:
            status: Status dictionary from charger.get_status()
        """
        state = status['charging_state']
        power_ok = "✓" if status['power_good'] else "✗"

        log.info(
            f"[CHARGER] State: {state:15s} | "
            f"Power: {power_ok} | "
            f"Battery: {'OK' if not status['battery_fault'] else 'FAULT'}"
        )

    def shutdown(self):
        """Cleanup charger connection."""
        if self.charger is not None:
            try:
                self.charger.disable_charging()
                self.charger.close()
                log.info("Battery charger shutdown complete")
            except Exception as e:
                log.warning(f"Error during charger shutdown: {e}")


# ============================================================================
# Integration into SmartEyeApp - Copy these sections into main.py
# ============================================================================


class SmartEyeAppWithCharger:
    """Example SmartEyeApp with battery charger integration."""

    def __init__(self, args):
        """Initialize application with charger support.

        Args:
            args: Parsed command-line arguments
        """
        # ... existing initialization ...

        # Initialize battery charger
        self.battery_monitor = BatteryChargerMonitor()
        self.charger_check_counter = 0

        # ... rest of initialization ...

    def main_loop(self):
        """Main application loop with charger monitoring.

        This shows the minimal integration pattern.
        """
        frame_count = 0

        while not self.exit_event.is_set():
            try:
                # ... existing frame capture and processing ...

                # Charger status check (every 5 seconds)
                self.charger_check_counter += 1
                if self.charger_check_counter >= 50:  # 50 * 100ms = 5s
                    status = self.battery_monitor.check_status()
                    if status:
                        self.battery_monitor.log_status(status)
                    self.charger_check_counter = 0

                frame_count += 1
                time.sleep(0.1)

            except KeyboardInterrupt:
                log.info("Interrupted by user")
                break
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                time.sleep(0.1)

    def shutdown(self):
        """Shutdown application including charger.

        This ensures charger is properly disabled before exit.
        """
        log.info("Shutting down Smart Eye...")

        # Disable charger
        self.battery_monitor.shutdown()

        # ... rest of shutdown ...


# ============================================================================
# Advanced: Charging-aware Behavior
# ============================================================================


class SmartEyeAppWithChargingAwareness(SmartEyeAppWithCharger):
    """Extended example with charging-aware power management."""

    # Power states
    POWER_STATE_NORMAL = 0
    POWER_STATE_CHARGING = 1
    POWER_STATE_THERMAL = 2
    POWER_STATE_LOW_BATTERY = 3

    def __init__(self, args):
        """Initialize with charging awareness."""
        super().__init__(args)
        self.power_state = self.POWER_STATE_NORMAL
        self.previous_state = None

    def update_power_state(self):
        """Update power state based on charger status."""
        status = self.battery_monitor.check_status()

        if status is None:
            self.power_state = self.POWER_STATE_NORMAL
            return

        # Check for critical issues
        if (status['thermal_shutdown'] or
                status['battery_overvoltage'] or
                status['input_overvoltage']):
            self.power_state = self.POWER_STATE_THERMAL
            return

        # Check charging state
        if status['charging_state'] in ["Pre-charge", "Fast Charge"]:
            self.power_state = self.POWER_STATE_CHARGING
            return

        self.power_state = self.POWER_STATE_NORMAL

        # Notify on state change
        if self.power_state != self.previous_state:
            self._on_power_state_changed()
            self.previous_state = self.power_state

    def _on_power_state_changed(self):
        """Handle power state transitions."""
        state_names = {
            self.POWER_STATE_NORMAL: "Normal",
            self.POWER_STATE_CHARGING: "Charging",
            self.POWER_STATE_THERMAL: "Thermal",
            self.POWER_STATE_LOW_BATTERY: "Low Battery",
        }
        log.info(f"Power state changed to: {state_names[self.power_state]}")

    def main_loop_with_charging_awareness(self):
        """Main loop with charging-aware behavior."""
        frame_count = 0

        while not self.exit_event.is_set():
            try:
                # Update power state
                self.update_power_state()

                # Adjust behavior based on power state
                if self.power_state == self.POWER_STATE_CHARGING:
                    # Reduce inference load when charging
                    # (lower heat generation, avoid thermal issues)
                    self._run_light_inference()
                elif self.power_state == self.POWER_STATE_THERMAL:
                    # Pause operations during thermal shutdown
                    self._pause_operations()
                else:
                    # Normal operation
                    self._run_full_inference()

                frame_count += 1
                time.sleep(0.1)

            except Exception as e:
                log.error(f"Error in main loop: {e}")
                time.sleep(0.1)

    def _run_full_inference(self):
        """Run full YOLOv8 inference."""
        # Normal operation
        pass

    def _run_light_inference(self):
        """Run reduced-load inference during charging."""
        # Skip every other frame, use lower resolution, etc.
        pass

    def _pause_operations(self):
        """Pause inference during thermal issues."""
        # Stop processing, wait for charger to cool
        log.warning("Operations paused due to thermal issues")
        time.sleep(1)


# ============================================================================
# Minimal Integration Pattern
# ============================================================================

"""
To add BQ25895 support to existing main.py, add these lines:

1. At the top of main.py, after other imports:

    from pathpal_project.bq25895_charger import BQ25895

2. In SmartEyeApp.__init__():

    # Configure battery charger for 10Ah battery
    try:
        self.charger = BQ25895()
        self.charger.configure_for_10ah_battery()
    except Exception as e:
        log.warning(f"Battery charger not available: {e}")
        self.charger = None

3. In SmartEyeApp main loop (every 5 seconds):

    if self.charger and <check interval>:
        status = self.charger.get_status()
        if status['thermal_shutdown']:
            log.warning("Thermal shutdown - reduce load")
        if status['battery_fault']:
            log.error("Battery fault!")

4. In SmartEyeApp.cleanup():

    if self.charger:
        self.charger.close()

That's it! The BQ25895 will be configured automatically at startup
and monitored passively in the main loop.
"""


def example_usage():
    """Demonstrate BQ25895 usage patterns."""
    print("=" * 70)
    print(" BQ25895 Integration Examples")
    print("=" * 70)

    # Example 1: Basic configuration
    print("\n1. Basic Configuration:")
    print("   charger = BQ25895()")
    print("   charger.configure_for_10ah_battery()")
    print("   charger.close()")

    # Example 2: Status monitoring
    print("\n2. Status Monitoring:")
    print("   status = charger.get_status()")
    print("   if status['thermal_shutdown']:")
    print("       print('Thermal issue!')")

    # Example 3: Custom configuration
    print("\n3. Custom Configuration:")
    print("   charger.configure_charging_current(1024)  # 1A instead of 2A")
    print("   charger.configure_charging_voltage(4100)  # 4.1V instead of 4.2V")

    # Example 4: Wrapper class
    print("\n4. Using BatteryChargerMonitor Wrapper:")
    print("   monitor = BatteryChargerMonitor()")
    print("   status = monitor.check_status()")
    print("   monitor.shutdown()")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_usage()
