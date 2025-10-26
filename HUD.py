import cv2
import numpy as np
from ultralytics import YOLO  # For future AI model integration

# Motion detection config
MIN_MOTION_AREA = 5000  # Adjusted for Pi 5 sensitivity
FRAME_DIFF_BLUR = (5, 5)
THRESHOLD_VALUE = 25
DILATE_ITERATIONS = 2

# AI model config (placeholder for YOLO)
MODEL_FILE = 'yolov8n.pt'  # Use a lightweight model like YOLOv8 nano
CONFIDENCE_THRESHOLD = 0.6

# Webcam setup
cap = cv2.VideoCapture(0)  # USB webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Load YOLO model (uncomment when model is ready)
# yolo_model = YOLO(MODEL_FILE)
# class_names = yolo_model.names

ret, frame1 = cap.read()
ret2, frame2 = cap.read()
if not ret or not ret2:
    print("Error: Unable to read frames.")
    cap.release()
    exit()

print("Motion detection running. Press 'q' to quit.")

while cap.isOpened():
    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, FRAME_DIFF_BLUR, 0)
    _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_MOTION_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            motion_detected = True

    # Placeholder for YOLO inference (uncomment when model is ready)
    # results = yolo_model(frame1, stream=True, verbose=False)
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:
    #         conf = box.conf[0]
    #         if conf > CONFIDENCE_THRESHOLD:
    #             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
    #             cls_id = int(box.cls[0])
    #             label = class_names[cls_id]
    #             cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 165, 255), 2)
    #             cv2.putText(frame1, f'{label} ({conf:.2f})', (x1, y1 - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # Display status
    status = "MOTION DETECTED" if motion_detected else "NO MOTION"
    cv2.putText(frame1, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if not motion_detected else (0, 0, 255), 2)

    cv2.imshow("Motion Detection", frame1)

    frame1 = frame2
    ok, frame2 = cap.read()
    if not ok:
        print("Error: Stream ended.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()