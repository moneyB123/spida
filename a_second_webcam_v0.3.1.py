import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import face_recognition

# --- Setup ---
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)

WINDOW_WIDTH = 420
WINDOW_HEIGHT = 420
BOX_SIZE = 200
BOX_X = (WINDOW_WIDTH - BOX_SIZE) // 2
BOX_Y = (WINDOW_HEIGHT - BOX_SIZE) // 2

save_dir = Path("captured")
save_dir.mkdir(exist_ok=True)

cv2.namedWindow('HUD', cv2.WINDOW_NORMAL)

state = "idle"  # idle, countdown, comparing
photo1 = None
photo2 = None
countdown_start = 0
box_color = (255, 255, 255)  # white
status_text = ""
status_color = (0, 0, 0)

def draw_hud(frame, face_detected):
    global box_color, status_text, status_color
    overlay = frame.copy()
    cv2.rectangle(overlay, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), box_color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), (0, 0, 0), 2)
    
    if state == "countdown":
        elapsed = int(time.time() - countdown_start)
        remaining = max(10 - elapsed, 0)
        timer_text = str(remaining)
        cv2.putText(frame, timer_text, (BOX_X + 10, BOX_Y + BOX_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    elif status_text:
        cv2.putText(frame, status_text, (frame.shape[1]//2 - 100, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
    
    box_color = (0, 255, 0) if face_detected else (255, 255, 255)
    return frame

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

def save_photo(frame, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_dir / f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    return filename

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2: break

    frame1 = cv2.resize(frame1, (WINDOW_WIDTH, WINDOW_HEIGHT))
    frame2 = cv2.resize(frame2, (WINDOW_WIDTH, WINDOW_HEIGHT))
    combined = np.hstack((frame1, frame2))

    faces1 = detect_faces(frame1)
    faces2 = detect_faces(frame2)
    face_detected = len(faces1) > 0 or len(faces2) > 0

    if state == "idle":
        if cv2.waitKey(1) & 0xFF == ord('c'):
            photo1 = combined.copy()
            save_photo(photo1, "photo1")
            countdown_start = time.time()
            state = "countdown"
    elif state == "countdown":
        elapsed = time.time() - countdown_start
        if elapsed >= 10:
            photo2 = combined.copy()
            save_photo(photo2, "photo2")
            state = "comparing"
    elif state == "comparing":
        enc1 = face_recognition.face_encodings(cv2.cvtColor(photo1, cv2.COLOR_BGR2RGB))
        enc2 = face_recognition.face_encodings(cv2.cvtColor(photo2, cv2.COLOR_BGR2RGB))
        if not enc1 or not enc2:
            status_text, status_color = "NO FACE", (0, 0, 0)
            combined[:] = (255, 255, 255)
        else:
            match = face_recognition.compare_faces(enc1, enc2[0])[0]
            if match:
                status_text, status_color = "ACCESS GRANTED", (255, 255, 255)
                combined[:] = (0, 255, 0)
            else:
                status_text, status_color = "ACCESS DENIED", (255, 255, 255)
                combined[:] = (0, 0, 255)
        time.sleep(3)
        state = "idle"
        status_text = ""

    combined = draw_hud(combined, face_detected)
    cv2.imshow('HUD', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
