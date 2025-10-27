import cv2
import numpy as np

# Initialize camera with V4L2 backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 100, 100)
    upper_red2 = (180, 255, 255)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green color range
    lower_green = (40, 100, 100)
    upper_green = (80, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # White color range
    lower_white = (0, 0, 200)
    upper_white = (180, 30, 255)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Count contours for red
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_count = 0
    for cnt in contours_red:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            red_count += 1

    # Count contours for green
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_count = 0
    for cnt in contours_green:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            green_count += 1

    # Count contours for white
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_count = 0
    for cnt in contours_white:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            white_count += 1

    # Display counters
    cv2.putText(frame, f"Green: {green_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"White: {white_count}", (WINDOW_WIDTH // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Red: {red_count}", (WINDOW_WIDTH - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('HUD', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
