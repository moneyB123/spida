import cv2
import numpy as np
from pathlib import Path
import time
import datetime
import os
import threading
import queue

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("dashcam_footage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = {
    "small":  (480, 360),
    "medium": (640, 480),
    "large":  (1280, 720)
}

ACTIVE_RES = "medium"
FRAME_W, FRAME_H = RESOLUTIONS[ACTIVE_RES]
FPS = 20
SEGMENT_MINUTES = 5
KEEP_HOURS = 2
EVENT_CLIP_SECONDS = 90  # Changed to 90 seconds

CAM_INDICES = [0, 2]
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# AI EVENT TRIGGERS
MOTION_THRESHOLD = 5000
SUDDEN_BRIGHT_THRESHOLD = 50
RED_OBJECT_THRESHOLD = 3

# ============================================================
# CAMERA SETUP
# ============================================================

def open_cam(index):
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cam

cam1 = open_cam(CAM_INDICES[0])
cam2 = open_cam(CAM_INDICES[1])

# ============================================================
# VIDEO WRITER
# ============================================================

def new_writer():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"seg_{ts}.mp4"
    writer = cv2.VideoWriter(str(filename), FOURCC, FPS, (FRAME_W * 2, FRAME_H))
    return writer, filename, time.time()

writer, current_file, segment_start = new_writer()

# ============================================================
# AUTO DELETE
# ============================================================

def clean_old_files():
    cutoff = time.time() - KEEP_HOURS * 3600
    for f in sorted(OUTPUT_DIR.glob("seg_*.mp4")):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except:
            pass

# ============================================================
# AI EVENT DETECTION
# ============================================================

prev_gray = None
prev_brightness = None

def detect_events(frame):
    global prev_gray, prev_brightness
    
    events = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Motion detection
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_pixels = np.sum(diff > 30)
        if motion_pixels > MOTION_THRESHOLD:
            events.append("MOTION")
    
    # 2. Sudden brightness change
    brightness = np.mean(gray)
    if prev_brightness is not None:
        if abs(brightness - prev_brightness) > SUDDEN_BRIGHT_THRESHOLD:
            events.append("BRIGHTNESS_SPIKE")
    
    # 3. Red object detection
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
    
    if red_count >= RED_OBJECT_THRESHOLD:
        events.append("RED_OBJECTS")
    
    prev_gray = gray
    prev_brightness = brightness
    
    return events

# ============================================================
# EVENT CLIP THREAD
# ============================================================

event_queue = queue.Queue()

def event_clip_saver(frames, trigger_reason):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"event_{ts}_{trigger_reason}.mp4"
    w = cv2.VideoWriter(str(filename), FOURCC, FPS, (FRAME_W*2, FRAME_H))
    for frame in frames:
        w.write(frame)
    w.release()

def event_worker():
    while True:
        item = event_queue.get()
        if item is None:
            break
        frames, reason = item
        event_clip_saver(frames, reason)

threading.Thread(target=event_worker, daemon=True).start()

event_buffer_frames = []
EVENT_BUFFER_MAX = EVENT_CLIP_SECONDS * FPS  # 90 seconds * 20 fps = 1800 frames
event_cooldown = 0
flash_frames_remaining = 0  # White flash counter

# ============================================================
# PREVIEW WINDOW
# ============================================================

cv2.namedWindow("Dashcam (q quit, e manual event)", cv2.WINDOW_NORMAL)

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret1, f1 = cam1.read()
    ret2, f2 = cam2.read()

    if not ret1:
        cam1.release()
        time.sleep(0.2)
        cam1 = open_cam(CAM_INDICES[0])
        continue

    if not ret2:
        cam2.release()
        time.sleep(0.2)
        cam2 = open_cam(CAM_INDICES[1])
        continue

    f1 = cv2.resize(f1, (FRAME_W, FRAME_H))
    f2 = cv2.resize(f2, (FRAME_W, FRAME_H))
    combined = np.hstack((f1, f2))

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined, ts, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # AI Event Detection
    events = detect_events(combined)
    
    if events and event_cooldown == 0:
        trigger = "+".join(events)
        event_queue.put((list(event_buffer_frames), trigger))
        event_cooldown = FPS * 5  # 5 second cooldown
        flash_frames_remaining = FPS * 3  # 3 seconds white flash
        cv2.putText(combined, f"AUTO EVENT: {trigger}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if event_cooldown > 0:
        event_cooldown -= 1

    # Event buffer (90 seconds)
    event_buffer_frames.append(combined.copy())
    if len(event_buffer_frames) > EVENT_BUFFER_MAX:
        event_buffer_frames.pop(0)

    # Segment control
    if time.time() - segment_start > SEGMENT_MINUTES * 60:
        writer.release()
        clean_old_files()
        writer, current_file, segment_start = new_writer()

    writer.write(combined)

    # White flash overlay for event indication
    display_frame = combined.copy()
    if flash_frames_remaining > 0:
        white_overlay = np.ones_like(display_frame) * 255
        alpha = 0.7  # Flash intensity
        display_frame = cv2.addWeighted(display_frame, 1-alpha, white_overlay, alpha, 0)
        flash_frames_remaining -= 1

    # Preview
    cv2.imshow("Dashcam (q quit, e manual event)", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('e'):
        event_queue.put((list(event_buffer_frames), "MANUAL"))
        flash_frames_remaining = FPS * 3  # Manual events also get flash

# ============================================================
# SHUTDOWN
# ============================================================

writer.release()
cam1.release()
cam2.release()
cv2.destroyAllWindows()
event_queue.put(None)
