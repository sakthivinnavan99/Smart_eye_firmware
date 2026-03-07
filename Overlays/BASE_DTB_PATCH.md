# Base DTB Patch: Disable FIQ Debugger on UART2

UART2 (serial@feb50000) is used by the FIQ debugger (debug console) in the
stock Radxa CM5 base DTB. This conflicts with using UART2 for peripherals
(e.g. ultrasonic sensor on J8 connector).

## What was changed

In `/usr/lib/linux-image-<KVER>/rockchip/rk3588s-radxa-cm5-io.dtb`:

```
fiq-debugger {
    rockchip,serial-id = <0xffffffff>;  // was <0x02> (UART2)
    status = "disabled";                 // was "okay"
};
```

## Why

- U-Boot uses the FIQ debugger UART for its console output during boot
- When an ultrasonic sensor is connected to UART2, U-Boot's console output
  at 1.5 Mbaud causes the sensor to respond with garbage data
- This can halt or corrupt the boot process

## How to restore

```bash
KVER=$(uname -r)
sudo cp /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb.bak \
        /usr/lib/linux-image-${KVER}/rockchip/rk3588s-radxa-cm5-io.dtb
```

## Warning

A kernel update will overwrite this patch. Re-apply after kernel upgrades.
