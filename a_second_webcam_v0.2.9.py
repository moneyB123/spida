import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# --- Setup ---
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

WINDOW_WIDTH = 420
WINDOW_HEIGHT = 420
MAX_SAVED_PHOTOS = 10
saved_count = 0
last_save_time = 0

save_dir = Path("detected_faces")
save_dir.mkdir(exist_ok=True)

cv2.namedWindow('HUD - Regular | Nightvision', cv2.WINDOW_NORMAL)


def process_frame(frame):
    global saved_count, last_save_time
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Face Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
        cv2.putText(frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save image if under limit and cooldown has passed
        now = time.time()
        if saved_count < MAX_SAVED_PHOTOS and (now - last_save_time) > 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"face_{timestamp}_{saved_count+1}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved: {filename}")
            saved_count += 1
            last_save_time = now

    # --- Green Object Counter ---
    lower_green = (40, 100, 100)
    upper_green = (80, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_count = 0
    for cnt in contours_green:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            green_count += 1

    cv2.putText(frame, f"Green: {green_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    frame1 = cv2.resize(frame1, (WINDOW_WIDTH, WINDOW_HEIGHT))
    frame2 = cv2.resize(frame2, (WINDOW_WIDTH, WINDOW_HEIGHT))

    processed1 = process_frame(frame1)
    processed2 = process_frame(frame2)

    combined = np.hstack((processed1, processed2))
    cv2.imshow('HUD - Regular | Nightvision', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
