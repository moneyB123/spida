"""
Setup Steps to Run this Script on Raspberry Pi:
1. Install camera support:
   sudo apt update && sudo apt install -y libraspberrypi-bin
2. Verify camera config:
   sudo nano /boot/firmware/config.txt
   Ensure/add: camera_auto_detect=1, start_x=1, gpu_mem=128
   Save, then reboot: sudo reboot
3. Test camera:
   raspistill -o test.jpg
   Check for test.jpg in /home/nisbetda/
4. Check video devices:
   ls /dev/video*
   Note index (e.g., 0 for /dev/video0)
5. Activate virtual environment:
   source ~/ultralytics_venv/bin/activate
6. Save this script:
   nano /tmp/xa-I031E3/HUDv0.3.0.py
   Paste this code, adjust cap = cv2.VideoCapture(0) index if needed, save
7. Run script:
   python3 /tmp/xa-I031E3/HUDv0.3.0.py
   Press 'q' to quit
Troubleshooting:
- If camera error, try index 1 or 2 in cap = cv2.VideoCapture()
- Check camera: vcgencmd get_camera (should show supported=1 detected=1)
- Add permissions: sudo usermod -a -G video $USER && sudo reboot
"""

import cv2
# from ultralytics import YOLO  # Commented out as unused

# Initialize camera (adjust index: 0, 1, or 2)
cap = cv2.VideoCapture(0)

# Set window size (width, height)
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for smaller window
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 100, 100)
    upper_red2 = (180, 255, 255)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Blue color range
    lower_blue = (100, 100, 100)
    upper_blue = (130, 255, 255)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours for red
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        if cv2.contourArea(cnt) > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 0, 255), 2)  # Red circle

    # Find contours for blue
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

    # Display frame
    cv2.imshow('HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
