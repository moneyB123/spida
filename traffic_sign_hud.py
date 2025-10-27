
"""
Traffic Sign Color Detection HUD - No ML Required
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red detection (stop signs)
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(frame, "STOP SIGN", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            detected = True
            break
    
    if not detected:
        cv2.putText(frame, "NO SIGN", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("HUD", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
