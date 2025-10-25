# rpi spida v0.2.0 software sim
# Date: 10/24/2025
# Author: Dalton Alwin Nisbett
# Location: Dayton. OH USA
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
leg_pins = [11,13,15,18,22,29]  # Servo pins
for pin in leg_pins: GPIO.setup(pin, GPIO.OUT)

def step_forward():
    # Simulate leg sequence
    for pin in leg_pins:
        GPIO.output(pin, 1); time.sleep(0.1); GPIO.output(pin, 0)

while True:
    step_forward(); time.sleep(1)
GPIO.cleanup()