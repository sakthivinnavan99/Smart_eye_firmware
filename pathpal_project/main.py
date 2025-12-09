#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import argparse

import sys

import time

import os

import threading

import queue

import traceback

from functools import lru_cache

from datetime import datetime

import cv2

import re

import numpy as np

import mraa

import RPi.GPIO as GPIO

from picamera2 import MappedArray, Picamera2

from picamera2.devices import IMX500

from picamera2.devices.imx500 import (NetworkIntrinsics,

                                      postprocess_nanodet_detection)

from paddleocr import PaddleOCR

import pygame

import subprocess

import argostranslate.package

import argostranslate.translate

import gc

import termios

import tty

import select

import spidev  # Added for MCP3002 ADC





# Suppress KMP duplicate lib warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define project root (change this to your project directory)
PROJECT_ROOT = "~/Smart_eye_firmware"

# Initialize pygame mixer

pygame.mixer.init()



last_audio_play_time = 0

AUDIO_COOLDOWN = 3.0  # seconds between audio announcements



# GPIO Setup

VIBRATION_MOTOR_PIN = 18

US1_TRIGGER = 12  # Front-facing (for hanging objects)

US1_ECHO = 16

US2_TRIGGER = 25  # Downward-facing

US2_ECHO = 26

PUSH_BUTTON_PIN = 27      # Push button to trigger OCR

TOGGLE_PIN = 17           # Toggle switch for language selection



# Initialize GPIO

GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)



# Vibration motor PWM setup

VIBRATION_PWM_FREQ = 500  # 500Hz PWM frequency

vibration_pwm = None



# Output devices

GPIO.setup(VIBRATION_MOTOR_PIN, GPIO.OUT)

vibration_pwm = GPIO.PWM(VIBRATION_MOTOR_PIN, VIBRATION_PWM_FREQ)

vibration_pwm.start(0)  # Start with 0% duty cycle



# Input devices

GPIO.setup(US1_TRIGGER, GPIO.OUT)

GPIO.setup(US1_ECHO, GPIO.IN)

GPIO.setup(US2_TRIGGER, GPIO.OUT)

GPIO.setup(US2_ECHO, GPIO.IN)

GPIO.setup(PUSH_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)



# Toggle switch for language selection

GPIO.setup(TOGGLE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

ocr_english = PaddleOCR(use_angle_cls=True, lang='en')

# SPI setup for MCP3002 ADC

spi = spidev.SpiDev()

spi.open(0, 0)  # bus 0, device 0

spi.max_speed_hz = 500000

spi.mode = 0



# Battery voltage parameters

R1 = 22000  # 10k resistor in voltage divider

R2 = 10000   # 1k resistor in voltage divider

VREF = 5.0  # MCP3002 reference voltage

DIVIDER_RATIO = (R1 + R2) / R2  # = 11

BATTERY_FULL = 8.4  # Voltage at 100%

BATTERY_EMPTY = 7.2  # Voltage at 0%

BATTERY_10_PERCENT = 7.32  # 10% threshold (7.2 + 0.12)

BATTERY_5_PERCENT = 7.26   # 5% threshold (7.2 + 0.06)




# Piper TTS Configuration
PIPER_BINARY_PATH = os.path.join(PROJECT_ROOT, "piper/piper/piper")

PIPER_MODEL_EN = os.path.join(PROJECT_ROOT, "piper/en_US-amy-medium.onnx")
PIPER_MODEL_HI = os.path.join(PROJECT_ROOT, "piper/hi_IN-pratham-medium.onnx")  # Hindi model

PIPER_OUTPUT_RATE = 22050



# Audio files for detections - separated by language
AUDIO_FILES = {
    "en": {
        "device_on": os.path.join(PROJECT_ROOT, "wav/English/device_turned_on.wav"),
        "pothole": os.path.join(PROJECT_ROOT, "wav/English/pothole.wav"),
        "stairs": os.path.join(PROJECT_ROOT, "wav/English/stairs.wav"),
        "fifty rupees": os.path.join(PROJECT_ROOT, "wav/English/fifty_rupees.wav"),
        "five hundred rupees": os.path.join(PROJECT_ROOT, "wav/English/five_hundred_rupees.wav"),
        "five rupees": os.path.join(PROJECT_ROOT, "wav/English/five_rupees.wav"),
        "hundred rupees": os.path.join(PROJECT_ROOT, "wav/English/hundred_rupees.wav"),
        "one rupees": os.path.join(PROJECT_ROOT, "wav/English/one_rupees.wav"),
        "ten rupees": os.path.join(PROJECT_ROOT, "wav/English/ten_rupees.wav"),
        "twenty rupees": os.path.join(PROJECT_ROOT, "wav/English/twenty_rupees.wav"),
        "two hundred rupees": os.path.join(PROJECT_ROOT, "wav/English/two_hundred_rupees.wav"),
        "two rupees": os.path.join(PROJECT_ROOT, "wav/English/two_rupees.wav"),
        "two thousand rupees": os.path.join(PROJECT_ROOT, "wav/English/two_thousand_rupees.wav"),
        "no_text_detected": os.path.join(PROJECT_ROOT, "wav/English/no_text_detected.wav"),
        "battery_10": os.path.join(PROJECT_ROOT, "wav/English/battery_10.wav"),
        "battery_shutdown": os.path.join(PROJECT_ROOT, "wav/English/battery_shutdown.wav"),
        "eng_toggle": os.path.join(PROJECT_ROOT, "wav/English/English_mode.wav"),
    },
    "hi": {
        "device_on": os.path.join(PROJECT_ROOT, "wav/Hindi/turned_on.wav"),
        "pothole": os.path.join(PROJECT_ROOT, "wav/Hindi/pothole_hindi.wav"),
        "stairs": os.path.join(PROJECT_ROOT, "wav/Hindi/stairs_detected.wav"),
        "fifty rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/fifty_rupees.wav"),
        "five hundred rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/five_hundred_rupees.wav"),
        "five rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/five_rupees.wav"),
        "hundred rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/hundred_rupees.wav"),
        "one rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/one_rupees.wav"),
        "ten rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/ten_rupees.wav"),
        "twenty rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/twenty_rupees.wav"),
        "two hundred rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/two_hundred_rupees.wav"),
        "two rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/two_rupees.wav"),
        "two thousand rupees": os.path.join(PROJECT_ROOT, "wav/Hindi/two_thousand_rupees.wav"),
        "no_text_detected": os.path.join(PROJECT_ROOT, "wav/Hindi/no_text_found.wav"),
        "battery_10": os.path.join(PROJECT_ROOT, "wav/Hindi/ten_percent_battery_alert.wav"),
        "battery_shutdown": os.path.join(PROJECT_ROOT, "wav/Hindi/five_percent_battery_alert.wav"),
        "hindi_toggle": os.path.join(PROJECT_ROOT, "wav/Hindi/Hindi_mode.wav")
    }
}



# Constants

ULTRASONIC_THRESHOLD = 155  # 155cm for downward US

HANGING_OBJECT_THRESHOLD = 90  # 90cm for front US (hanging objects)

SPEED_OF_SOUND = 34300    # cm/s

ULTRASONIC_TIMEOUT = 0.1  # seconds

MIN_VALID_DISTANCE = 2    # cm

MAX_VALID_DISTANCE = 400  # cm



DEBOUNCE_TIME = 0.3  # seconds

ANNOUNCE_COOLDOWN = 2.0  # seconds between announcements

OCR_COOLDOWN = 3.0  # seconds between OCR scans

TTS_COOLDOWN = 1.5  # seconds between TTS messages

BATTERY_CHECK_INTERVAL = 5.0  # seconds between battery checks

TOGGLE_CHECK_INTERVAL = 0.5   # seconds between toggle checks



# Global variables

last_detections = []

ultrasonic_detection = False

last_ultrasonic_print = 0

last_button_press = 0

last_announce_time = 0

detection_timestamp = 0

picam2 = None

intrinsics = None

args = None

last_ocr_time = 0

specific_object_vibration_active = False

tts_queue = queue.Queue()

tts_thread = None

audio_queue = queue.Queue()

audio_thread = None

battery_thread = None

battery_running = True

battery_10_warning_given = False

battery_5_warning_given = False

last_toggle_state = None

last_toggle_check = 0

beep_sound = None  # Global variable for beep sound



# NEW: Function to generate beep sound

def generate_beep_sound(volume=0.5):

    """Generate a short melodious 1-second sound using pygame and NumPy"""

    try:

        import pygame.sndarray

        sample_rate = pygame.mixer.get_init()[0] or 44100

        channels = pygame.mixer.get_init()[2] or 2



        notes = [

            (659, 300),  # E5

            (784, 300),  # G5

            (880, 400)   # A5

        ]



        full_wave = np.zeros((0, channels), dtype=np.int16)



        for freq, duration in notes:

            t = np.linspace(0, duration / 1000.0, int(sample_rate * duration / 1000.0), False)

            wave = volume * np.sin(2 * np.pi * freq * t)



            if channels == 2:

                wave_stereo = np.column_stack((wave, wave))

            else:

                wave_stereo = wave.reshape(-1, 1)



            wave_int16 = (wave_stereo * 32767).astype(np.int16)

            full_wave = np.vstack((full_wave, wave_int16))



        return pygame.sndarray.make_sound(full_wave)



    except Exception as e:

        print(f"Error generating melodious beep: {e}")

        return None



# NEW: Function to play beep sound

def play_beep():

    """Play the pre-generated beep sound if available"""

    global beep_sound

    try:

        if beep_sound:

            beep_sound.play()

    except Exception as e:

        print(f"Error playing beep sound: {e}")



def read_adc():

    """Read ADC value from MCP3002"""

    cmd = 0b01101000  # Start bit + single-ended + CH0

    result = spi.xfer2([cmd, 0])

    adc_value = ((result[0] & 0x03) << 8) | result[1]

    return adc_value



def get_battery_voltage():

    """Calculate actual battery voltage from ADC reading"""

    adc_val = read_adc()

    voltage_at_adc = (adc_val * VREF) / 1023

    actual_battery_voltage = voltage_at_adc * DIVIDER_RATIO

    return actual_battery_voltage



def get_battery_percentage(voltage):

    """Convert battery voltage to percentage (0-100)"""

    if voltage >= BATTERY_FULL:

        return 100

    elif voltage <= BATTERY_EMPTY:

        return 0

    else:

        return int(((voltage - BATTERY_EMPTY) / (BATTERY_FULL - BATTERY_EMPTY)) * 100)



def draw_simple_boxes(image, results):

    """Draws simple rectangles on the image for detected text."""

    img_display = image.copy()

    for item in results:

        if not isinstance(item, (list, tuple)) or len(item) < 3:

            continue

        bbox, text, prob = item

        color = (255, 0, 0)

        top_left_x = int(min(p[0] for p in bbox))

        top_left_y = int(min(p[1] for p in bbox))

        bottom_right_x = int(max(p[0] for p in bbox))

        bottom_right_y = int(max(p[1] for p in bbox))

        cv2.rectangle(img_display, (top_left_x, top_left_y),

                      (bottom_right_x, bottom_right_y), color, 2)

    return img_display



def vibrate(duration_ms):

    vibration_pwm.ChangeDutyCycle(100)  # Max power

    time.sleep(duration_ms / 1000.0)

    vibration_pwm.ChangeDutyCycle(0)



def pause(duration_ms):

    time.sleep(duration_ms / 1000.0)



def run_specific_vibration_pattern(pattern_function):

    global specific_object_vibration_active

    if specific_object_vibration_active:

        return

    specific_object_vibration_active = True

    try:

        pattern_function()

    finally:

        specific_object_vibration_active = False



def pattern_stairs():

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.15)

    vibration_pwm.ChangeDutyCycle(0)

    time.sleep(0.1)

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.25)

    vibration_pwm.ChangeDutyCycle(0)

    time.sleep(0.1)

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.35)

    vibration_pwm.ChangeDutyCycle(0)



def pattern_pothole():

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.5)

    vibration_pwm.ChangeDutyCycle(0)

    time.sleep(0.2)

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.25)

    vibration_pwm.ChangeDutyCycle(0)



def pattern_currency():

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.2)

    vibration_pwm.ChangeDutyCycle(0)

    time.sleep(0.1)

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.2)

    vibration_pwm.ChangeDutyCycle(0)



def pattern_front_us():  # For hanging objects (one long buzz)

    vibration_pwm.ChangeDutyCycle(100)  # Max power

    time.sleep(0.8)  # Single long 800ms vibration

    vibration_pwm.ChangeDutyCycle(0)



def pattern_down_us():  # For ground obstacles (keep existing pattern)

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.3)

    vibration_pwm.ChangeDutyCycle(0)

    time.sleep(0.1)

    vibration_pwm.ChangeDutyCycle(100)

    time.sleep(0.2)

    vibration_pwm.ChangeDutyCycle(0)



def audio_worker():

    """Thread worker for playing audio files"""

    while True:

        audio_file = audio_queue.get()

        if audio_file is None:

            audio_queue.task_done()

            break



        try:

            if pygame.mixer.get_busy():

                pygame.mixer.stop()



            sound = pygame.mixer.Sound(audio_file)

            sound.play()

            while pygame.mixer.get_busy():

                time.sleep(0.1)



        except Exception as e:

            print(f"Error playing audio file {audio_file}: {e}")

        finally:

            audio_queue.task_done()



def tts_worker():

    """Thread worker for TTS with language support"""

    while True:

        message_tuple = tts_queue.get()

        if message_tuple is None:

            tts_queue.task_done()

            break



        text_to_speak, model_path = message_tuple



        piper_process = None

        aplay_process = None

        try:

            piper_cmd = [

                PIPER_BINARY_PATH,

                "--model", model_path,

                "--output-raw",

                "--length-scale", "1.0"

            ]

            aplay_cmd = [

                "aplay",

                "-r", str(PIPER_OUTPUT_RATE),

                "-f", "S16_LE",

                "-t", "raw",

                "-"

            ]



            piper_process = subprocess.Popen(

                piper_cmd,

                stdin=subprocess.PIPE,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE

            )

            aplay_process = subprocess.Popen(

                aplay_cmd,

                stdin=piper_process.stdout,

                stderr=subprocess.PIPE

            )



            piper_process.stdin.write(text_to_speak.encode('utf-8'))

            piper_process.stdin.close()



            piper_process.wait()

            aplay_process.wait()



        except Exception as e:

            print(f"Error during TTS: {e}")

        finally:

            if piper_process and piper_process.poll() is None:

                piper_process.terminate()

            if aplay_process and aplay_process.poll() is None:

                aplay_process.terminate()

            tts_queue.task_done()



def get_current_language():

    """Determine current language based on toggle switch position"""

    if GPIO.input(TOGGLE_PIN) == GPIO.LOW:  # Left position (English)

        return "en"

    else:  # Right position (Hindi)

        return "hi"



def play_audio(audio_key):

    """Play the corresponding audio file in the current language"""

    global last_audio_play_time

    current_time = time.time()



    if current_time - last_audio_play_time < AUDIO_COOLDOWN:

        print(f"Ignoring {audio_key} detection due to cooldown")

        return



    lang = get_current_language()



    # Get audio file for the current language

    if lang in AUDIO_FILES and audio_key in AUDIO_FILES[lang]:

        audio_file = AUDIO_FILES[lang][audio_key]

        last_audio_play_time = current_time

        audio_queue.put(audio_file)

    else:

        print(f"No audio file found for key: {audio_key} in {lang} language")



def check_toggle_switch():

    """Check if toggle switch position has changed and play appropriate audio"""

    global last_toggle_state, last_toggle_check, last_audio_play_time



    current_time = time.time()

    if current_time - last_toggle_check < TOGGLE_CHECK_INTERVAL:

        return



    last_toggle_check = current_time

    current_toggle_state = GPIO.input(TOGGLE_PIN)



    # Only play audio if state has changed

    if current_toggle_state != last_toggle_state:

        # Play English toggle sound when switched to left position

        if current_toggle_state == GPIO.LOW:

            play_audio("eng_toggle")

        # Play Hindi toggle sound when switched to right position

        else:

            play_audio("hindi_toggle")



        last_toggle_state = current_toggle_state



def play_startup_alert():

    """Play device turn-on sound using WAV file"""

    play_audio("device_on")

    time.sleep(1)



def upscale_image(image, scale):

    if scale == 1.0:

        return image

    height, width = image.shape[:2]

    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)



def preprocess_image(bgr_img):

    """Enhanced image preprocessing for OCR"""

    try:

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((2, 2), np.uint8)

        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        processed = clahe.apply(morph)

        return processed

    except Exception as e:

        print(f"Image processing error: {e}")

        return bgr_img



def translate_english_to_hindi(text_to_translate):

    """Translates English text to Hindi using Argos Translate"""

    from_code = "en"  # English

    to_code = "hi"    # Hindi



    # Handle all-caps text

    if text_to_translate.isupper():

        text_to_translate = text_to_translate.lower()



    installed_languages = argostranslate.translate.get_installed_languages()



    # Find language objects

    source_lang = next((lang for lang in installed_languages if lang.code == from_code), None)

    target_lang = next((lang for lang in installed_languages if lang.code == to_code), None)



    if source_lang and target_lang:

        try:

            translation = source_lang.get_translation(target_lang)

            return translation.translate(text_to_translate)

        except Exception as e:

            return f"Translation error: {str(e)}"

    else:

        return "Translation failed: Language models not loaded"



# NEW: Function to normalize phone numbers for TTS

def normalize_numbers(text):

    """

    Convert contiguous digit sequences longer than or equal 5 digits into space-separated

    digits to ensure proper TTS pronunciation (e.g., for phone numbers).

    Shorter sequences are left unchanged.

    """

    # Regular expression to find sequences of 5 or more digits

    digit_pattern = r'\d{6,}'



    def replace_with_digits(match):

        digit_sequence = match.group(0)

        return ' '.join(digit_sequence)



    normalized_text = re.sub(digit_pattern, replace_with_digits, text)

    return normalized_text



def run_ocr_on_frame(bgr_frame, scale=2.0):

    """Run OCR with language selection based on toggle switch"""

    global last_ocr_time



    current_time = time.time()

    if current_time - last_ocr_time < OCR_COOLDOWN:

        return bgr_frame, []



    try:

        vibration_pwm.ChangeDutyCycle(0)

        zoomed = upscale_image(bgr_frame, scale)

        processed = preprocess_image(zoomed)



        # Get current language mode from toggle switch

        lang_mode = "hindi" if get_current_language() == "hi" else "english"



        # Perform OCR

        results = ocr_english.ocr(processed, cls=True)

        texts = []

        if results and results[0]:

            for box_info in results[0]:

                box, (text, score) = box_info

                if score > 0.75:

                    texts.append(text)



        # Process results based on language mode

        if texts:

            combined_text = ' '.join(texts)



            # NEW: Normalize phone numbers in the OCR result

            normalized_text = normalize_numbers(combined_text)



            if lang_mode == "hindi":

                translated = translate_english_to_hindi(normalized_text)

                print(f"Hindi Translation: {translated}")

                tts_queue.put((translated, PIPER_MODEL_HI))

            else:

                print(f"English Text: {normalized_text}")

                tts_queue.put((normalized_text, PIPER_MODEL_EN))

        else:

            play_audio("no_text_detected")



        last_ocr_time = current_time

        return zoomed, texts

    except Exception as e:

        print(f"OCR processing error: {e}")

        traceback.print_exc()

        return bgr_frame, []

    finally:

        vibration_pwm.ChangeDutyCycle(0)

        gc.collect()



def get_distance(trigger_pin, echo_pin):

    try:

        GPIO.output(trigger_pin, False)

        time.sleep(0.01)



        GPIO.output(trigger_pin, True)

        time.sleep(0.00001)

        GPIO.output(trigger_pin, False)



        pulse_start = time.time()

        pulse_end = time.time()

        timeout_start = time.time()



        while GPIO.input(echo_pin) == 0:

            pulse_start = time.time()

            if pulse_start - timeout_start > ULTRASONIC_TIMEOUT:

                return None



        while GPIO.input(echo_pin) == 1:

            pulse_end = time.time()

            if pulse_end - timeout_start > ULTRASONIC_TIMEOUT:

                return None



        distance = (pulse_end - pulse_start) * SPEED_OF_SOUND / 2



        if distance < MIN_VALID_DISTANCE or distance > MAX_VALID_DISTANCE:

            return None



        return distance

    except Exception as e:

        print(f"Error in get_distance: {e}")

        return None



def check_ultrasonic_sensors():

    global ultrasonic_detection, last_ultrasonic_print



    try:

        distance1 = get_distance(US1_TRIGGER, US1_ECHO)

        time.sleep(0.05)

        distance2 = get_distance(US2_TRIGGER, US2_ECHO)



        current_time = time.time()

        if current_time - last_ultrasonic_print > 0.5:

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            dist1_str = f"{distance1:.1f}cm" if distance1 is not None else "N/A"

            dist2_str = f"{distance2:.1f}cm" if distance2 is not None else "N/A"

            print(f"[{timestamp}] US1(Front): {dist1_str} (Threshold: {HANGING_OBJECT_THRESHOLD}cm), US2(Down): {dist2_str} (Threshold: {ULTRASONIC_THRESHOLD}cm)")

            last_ultrasonic_print = current_time



        us1_detected = distance1 is not None and distance1 < HANGING_OBJECT_THRESHOLD

        us2_detected = distance2 is not None and distance2 < ULTRASONIC_THRESHOLD



        ultrasonic_detection = us1_detected or us2_detected



        if us1_detected and not specific_object_vibration_active:

            threading.Thread(target=run_specific_vibration_pattern, args=(pattern_front_us,), daemon=True).start()

        elif us2_detected and not specific_object_vibration_active:

            threading.Thread(target=run_specific_vibration_pattern, args=(pattern_down_us,), daemon=True).start()



        return ultrasonic_detection



    except Exception as e:

        print(f"Error in check_ultrasonic_sensors: {e}")

        return False





class Detection:

    def __init__(self, coords, category, conf, metadata):

        self.category = category

        self.conf = conf

        self.box = imx500.convert_inference_coords(coords, metadata, picam2)



def parse_detections(metadata: dict):

    global last_detections, detection_timestamp

    bbox_normalization = intrinsics.bbox_normalization

    bbox_order = intrinsics.bbox_order

    threshold = args.threshold

    iou = args.iou

    max_detections = args.max_detections



    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    input_w, input_h = imx500.get_input_size()

    if np_outputs is None:

        last_detections = []

        return last_detections



    if intrinsics.postprocess == "nanodet":

        boxes, scores, classes = postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,

                                          max_out_dets=max_detections)[0]

        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)

    else:

        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        if bbox_normalization:

            boxes = boxes / input_h



        if bbox_order == "xy":

            boxes = boxes[:, [1, 0, 3, 2]]

        boxes = np.array_split(boxes, 4, axis=1)

        boxes = zip(*boxes)



    new_detections = [

        Detection(box, category, score, metadata)

        for box, score, category in zip(boxes, scores, classes)

        if score > threshold

    ]



    if new_detections:

        last_detections = new_detections

        detection_timestamp = time.time()



        labels = get_labels()



        for det in new_detections:

            label = labels[int(det.category)]

            label_lower = label.lower()



            if "pothole" in label_lower:

                threading.Thread(target=run_specific_vibration_pattern, args=(pattern_pothole,), daemon=True).start()

                print(f"Camera: {label} detected")

                play_audio("pothole")

            elif "stairs" in label_lower:

                threading.Thread(target=run_specific_vibration_pattern, args=(pattern_stairs,), daemon=True).start()

                print(f"Camera: {label} detected")

                play_audio("stairs")

            # In the parse_detections function, modify the currency detection section:

            elif "rupee" in label_lower:

                # Currency detection is always active

                threading.Thread(target=run_specific_vibration_pattern, args=(pattern_currency,), daemon=True).start()

                print(f"Camera: {label} detected")



                # Find matching audio key - exact match first, then partial

                audio_key = None

                label_lower = label.lower().strip()

                

                # First try exact match

                for key in AUDIO_FILES["en"]:

                    if key == label_lower:

                        audio_key = key

                        break

                

                # If no exact match, try partial match (longest first)

                if audio_key is None:

                    # Sort keys by length (longest first) to match most specific first

                    sorted_keys = sorted(AUDIO_FILES["en"].keys(), key=len, reverse=True)

                    for key in sorted_keys:

                        if key in label_lower:

                            audio_key = key

                            break



                if audio_key:

                    play_audio(audio_key)

                else:

                    print(f"No matching audio for currency: {label}")

            else:

                print(f"Camera: {label} detected - no haptic reaction")



    else:

        check_ultrasonic_sensors()



    return last_detections



@lru_cache

def get_labels():

    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:

        labels = [label for label in labels if label and label != "-"]

    return labels



def draw_detections(request):

    detections = last_detections

    if not detections:

        return



    if time.time() - detection_timestamp > 0.5:

        return



    labels = get_labels()

    with MappedArray(request, "main") as m:

        for detection in detections:

            x, y, w, h = detection.box

            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"



            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            text_x = x + 5

            text_y = y + 15



            overlay = m.array.copy()

            cv2.rectangle(overlay,

                          (text_x, text_y - text_height),

                          (text_x + text_width, text_y + baseline),

                          (255, 255, 255),

                          cv2.FILLED)



            alpha = 0.30

            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)



            cv2.putText(m.array, label, (text_x, text_y),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)



        if intrinsics.preserve_aspect_ratio:

            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)

            color = (255, 0, 0)

            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,

                        default="/home/abhimanyu/testing/network.rpk")

    parser.add_argument("--fps", type=int)

    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)

    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx")

    parser.add_argument("--threshold", type=float, default=0.70)

    parser.add_argument("--iou", type=float, default=0.65)

    parser.add_argument("--max-detections", type=int, default=10)

    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction)

    parser.add_argument("--postprocess", choices=["", "nanodet"], default=None)

    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction)

    parser.add_argument("--labels", type=str)

    parser.add_argument("--print-intrinsics", action="store_true")



    return parser.parse_args([

        "--model", "/home/abhimanyu/testing/network.rpk",

        "--labels", "/home/abhimanyu/testing/labels.txt",

        "--fps", "25",

        "--bbox-normalization",

        "--ignore-dash-labels",

        "--bbox-order", "xy"

    ])



def check_button():

    global last_button_press



    try:

        current_time = time.time()

        if GPIO.input(PUSH_BUTTON_PIN) == GPIO.LOW and (current_time - last_button_press) > DEBOUNCE_TIME:

            last_button_press = current_time



            print("Button pressed - Running OCR...")

            vibrate(200)  # Tactile feedback for OCR start

            play_beep()  # NEW: Play beep sound on button press



            try:

                bgr_frame = picam2.capture_array()

                run_ocr_on_frame(bgr_frame, scale=2.0)

            except Exception as e:

                print(f"OCR error: {e}")

            finally:

                vibration_pwm.ChangeDutyCycle(0)



    except Exception as e:

        print(f"Error in check_button: {e}")

        vibration_pwm.ChangeDutyCycle(0)



def enter_pressed():

    dr, _, _ = select.select([sys.stdin], [], [], 0)

    return dr and sys.stdin.read(1) == '\n'



def battery_monitor():

    """Thread function to monitor battery levels using MCP3002 ADC every 5 sec, display average every 50 sec"""

    global battery_10_warning_given, battery_5_warning_given



    readings = []



    while battery_running:

        try:

            # Take one reading

            adc_val = read_adc()

            voltage_at_adc = (adc_val * VREF) / 1023

            actual_battery_voltage = voltage_at_adc * DIVIDER_RATIO



            readings.append(actual_battery_voltage)



            if len(readings) >= 10:

                # After 10 readings (i.e., every 50 sec)

                avg_voltage = sum(readings) / len(readings)

                readings.clear()  # Reset readings for next cycle



                battery_percent = get_battery_percentage(avg_voltage)

                print(f"Battery: {avg_voltage:.2f}V ({battery_percent}%)")



                # Warnings based on average

                if avg_voltage <= BATTERY_5_PERCENT and not battery_5_warning_given:

                    print("Battery at 5 percent shutting down!")

                    play_audio("battery_shutdown")

                    battery_5_warning_given = True



                if avg_voltage <= BATTERY_10_PERCENT and not battery_10_warning_given:

                    print("Battery at 10% - please charge soon!")

                    play_audio("battery_10")

                    battery_10_warning_given = True



                # Reset warnings if charging

                if avg_voltage > BATTERY_10_PERCENT:

                    battery_10_warning_given = False

                if avg_voltage > BATTERY_5_PERCENT:

                    battery_5_warning_given = False



            time.sleep(5.0)  # Wait 5 seconds before next reading



        except Exception as e:

            print(f"Error in battery monitor: {e}")

            time.sleep(5.0)



def cleanup():

    global tts_thread, audio_thread, battery_thread, battery_running, vibration_pwm



    if vibration_pwm:

        vibration_pwm.ChangeDutyCycle(0)

        vibration_pwm.stop()



    # Signal battery thread to stop

    battery_running = False

    if battery_thread and battery_thread.is_alive():

        battery_thread.join()



    if tts_thread and tts_thread.is_alive():

        tts_queue.put(None)

        tts_thread.join()



    if audio_thread and audio_thread.is_alive():

        audio_queue.put(None)

        audio_thread.join()



    # Close SPI connection

    spi.close()

    GPIO.cleanup()



if __name__ == "__main__":

    # Set up terminal for non-blocking input

    fd = sys.stdin.fileno()

    old_settings = termios.tcgetattr(fd)

    tty.setcbreak(fd)



    # NEW: Generate beep sound at startup

    beep_sound = generate_beep_sound(volume=0.5)



    try:

        args = get_args()



        imx500 = IMX500(args.model)

        intrinsics = imx500.network_intrinsics

        if not intrinsics:

            intrinsics = NetworkIntrinsics()

            intrinsics.task = "object detection"

        elif intrinsics.task != "object detection":

            print("Network is not an object detection task", file=sys.stderr)

            exit()



        for key, value in vars(args).items():

            if key == 'labels' and value is not None:

                with open(value, 'r') as f:

                    intrinsics.labels = f.read().splitlines()

            elif hasattr(intrinsics, key) and value is not None:

                setattr(intrinsics, key, value)



        if intrinsics.labels is None:

            with open("assets/coco_labels.txt", "r") as f:

                intrinsics.labels = f.read().splitlines()

        intrinsics.update_with_defaults()



        if args.print_intrinsics:

            print(intrinsics)

            exit()



        picam2 = Picamera2(imx500.camera_num)

        config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)



        imx500.show_network_fw_progress_bar()

        picam2.start(config, show_preview=True)



        if intrinsics.preserve_aspect_ratio:

            imx500.set_auto_aspect_ratio()



        picam2.pre_callback = draw_detections



        # Start TTS thread

        tts_thread = threading.Thread(target=tts_worker, daemon=True)

        tts_thread.start()



        # Start audio thread

        audio_thread = threading.Thread(target=audio_worker, daemon=True)

        audio_thread.start()



        # Start battery monitoring thread

        battery_thread = threading.Thread(target=battery_monitor, daemon=True)

        battery_thread.start()



        # Initialize toggle state

        last_toggle_state = GPIO.input(TOGGLE_PIN)



        # Play startup alert

        play_startup_alert()



        print("Starting detection system...")

        print(f"US1 (Front) Range: {HANGING_OBJECT_THRESHOLD}cm, US2 (Down) Range: {ULTRASONIC_THRESHOLD}cm")

        print(f"Camera detection threshold: {args.threshold}")

        print("Toggle Switch: Left=English | Right=Hindi")

        print("Press button OR ENTER KEY for text recognition")

        print("System ready - waiting for detections...")



        while True:

            metadata = picam2.capture_metadata()

            parse_detections(metadata)

            check_button()

            check_toggle_switch()  # Check for toggle switch changes



            # Check for Enter key press

            if enter_pressed():

                current_time = time.time() 

                if current_time - last_button_press > DEBOUNCE_TIME:

                    last_button_press = current_time

                    print("Enter key pressed - Running OCR...")

                    vibrate(200)  # Tactile feedback

                    play_beep()    # NEW: Play beep sound on Enter key press



                    try:

                        bgr_frame = picam2.capture_array()

                        run_ocr_on_frame(bgr_frame, scale=2.0)

                    except Exception as e:

                        print(f"OCR error: {e}")

                    finally:

                        vibration_pwm.ChangeDutyCycle(0)



            gc.collect()  # Regular garbage collection



    except KeyboardInterrupt:

        print("\nExiting...")

    finally:

        # Restore terminal settings

        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        cleanup()