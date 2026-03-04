DTC ?= dtc
KERNEL_VER := $(shell uname -r)
OVERLAY_DIR := /usr/lib/linux-image-$(KERNEL_VER)/rockchip/overlays
BOOT_DTBO_DIR := /boot/dtbo

DTS := rk3588-smart-eye-carrier.dts
DTBO := $(DTS:.dts=.dtbo)

.PHONY: all clean install enable disable status

all: $(DTBO)

$(DTBO): $(DTS)
	$(DTC) -@ -I dts -O dtb -o $@ $<

clean:
	rm -f $(DTBO) *.dts.preprocessed

install: $(DTBO)
	sudo cp $(DTBO) $(OVERLAY_DIR)/
	sudo cp $(DTBO) $(BOOT_DTBO_DIR)/$(DTBO).disabled
	@echo "Overlay installed. Run 'make enable' to activate it."

enable:
	@if [ -f $(BOOT_DTBO_DIR)/$(DTBO).disabled ]; then \
		sudo mv $(BOOT_DTBO_DIR)/$(DTBO).disabled $(BOOT_DTBO_DIR)/$(DTBO); \
		echo "$(DTBO)" | sudo tee -a $(BOOT_DTBO_DIR)/managed.list > /dev/null; \
		echo "Overlay enabled. Reboot to apply."; \
	elif [ -f $(BOOT_DTBO_DIR)/$(DTBO) ]; then \
		echo "Overlay is already enabled."; \
	else \
		echo "Overlay not installed. Run 'make install' first."; \
	fi

disable:
	@if [ -f $(BOOT_DTBO_DIR)/$(DTBO) ]; then \
		sudo mv $(BOOT_DTBO_DIR)/$(DTBO) $(BOOT_DTBO_DIR)/$(DTBO).disabled; \
		sudo sed -i '/$(DTBO)/d' $(BOOT_DTBO_DIR)/managed.list; \
		echo "Overlay disabled. Reboot to apply."; \
	elif [ -f $(BOOT_DTBO_DIR)/$(DTBO).disabled ]; then \
		echo "Overlay is already disabled."; \
	else \
		echo "Overlay not installed."; \
	fi

status:
	@echo "=== Overlay Status ==="
	@if [ -f $(BOOT_DTBO_DIR)/$(DTBO) ]; then \
		echo "  Boot:    ENABLED"; \
	elif [ -f $(BOOT_DTBO_DIR)/$(DTBO).disabled ]; then \
		echo "  Boot:    DISABLED"; \
	else \
		echo "  Boot:    NOT INSTALLED"; \
	fi
	@if [ -f $(OVERLAY_DIR)/$(DTBO) ]; then \
		echo "  System:  INSTALLED"; \
	else \
		echo "  System:  NOT INSTALLED"; \
	fi
	@echo ""
	@echo "=== Peripherals Configured ==="
	@echo "  PWM15-M1  (GPIO4_B3)         -> Vibration motor"
	@echo "  I2C3-M1   (GPIO3_B7 SCL / GPIO3_C0 SDA) -> BQ27220 fuel gauge"
	@echo "  GPIO4_A4                      -> Fuel gauge interrupt"
	@echo "  UART6-M2  (GPIO1_D1/GPIO1_D0) -> Serial port"
	@echo "  GPIO0_D3  CHG_INT0_L          -> Charger interrupt (active low)"
	@echo "  GPIO0_D0  LANG_BTN            -> Language select button (active low)"
	@echo "  GPIO0_C7  OCR_BTN             -> OCR trigger button (active low)"
	@echo "  I2S0      ES8316 + MAX98357A  -> Headphone + Speaker (auto-switch)"
	@echo "  GPIO4_A3  AMP_SD             -> MAX98357A speaker enable"
	@echo "  GPIO1_C0  HP_DET             -> Headphone jack detect"
