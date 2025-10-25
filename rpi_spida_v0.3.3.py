# rpi spida v0.3.3 - COMBINED MOTION & CARD COMMAND LOGIC
# Date: 10/25/2025
# Author: Dalton Alwin Nisbett
# Location: Dayton, OH USA
"""
Spider Leg Controller with Card Command and Filtered Motion Detection.

Priority Logic:
1. Card Command (Card detector is the primary brain).
2. Motion Detection (Acts as an override safety stop for obstacles).

NEW COMMAND MAPPING:
- Ace of Clubs (AC): TURN LEFT (Rotation)
- All other Clubs (e.g., 2C, KC): MOVE LEFT (Strafe/Sideways)
- Ace of Spades (AS): TURN RIGHT (Rotation)
- All other Spades (e.g., 2S, KS): MOVE RIGHT (Strafe/Sideways)
- Diamonds (D): FORWARD
- Hearts (H): BACKWARD
- 9 of Diamonds (9D): CRITICAL CRASH (Exits the entire program)
"""

from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import sys
import traceback
import os
from ultralytics import YOLO  # Import YOLO for card detection
import math

# ----- Configuration -----
STREAM_URL = "http://192.168.0.101:8080/stream"

# --- MOTION DETECTION FILTER ---
# MIN_MOTION_AREA is the threshold for the PRIMARY SAFETY OVERRIDE (Large Motion)
MIN_MOTION_AREA = 99999
# SMALL_ITEM_MAX_AREA defines the upper limit for "small" objects to be visualized/counted
SMALL_ITEM_MAX_AREA = 5000
# NOISE_FLOOR_AREA filters out minor video noise and is the lower bound for "small" objects
NOISE_FLOOR_AREA = 50
FRAME_DIFF_BLUR = (5, 5)
THRESHOLD_VALUE = 20
DILATE_ITERATIONS = 3
LOG_DIR = Path("spider_logs")

# --- CARD DETECTION CONFIG ---
# IMPORTANT: This .pt file MUST be in the same folder as this script.
MODEL_FILE = 'yolov8s_playing_cards.pt'
CONFIDENCE_THRESHOLD = 0.65  # Only accept detections above this confidence level

# ----- Card to Command Mapping -----
# This is our rule set for how the spider should move based on a detected card
# Note: The logic in the main loop handles the Ace exceptions.
CARD_COMMANDS = {
    "D": "FORWARD",  # Red Diamonds -> FORWARD
    "H": "BACKWARD",  # Red Hearts -> BACKWARD
    "C": "MOVE LEFT",  # Generic Black Clubs -> STRAFE LEFT
    "S": "MOVE RIGHT",  # Generic Black Spades -> STRAFE RIGHT
    "9D": "CRITICAL CRASH"  # 9 of Diamonds (special override)
}


# ----- Spider controller (holds simulation state) -----
class SpiderLegController:
    def __init__(self, frame_width, frame_height):
        self.leg_positions = [0, 0, 0, 0, 0, 0]
        self.command_log = []

        # Simulation attributes
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sim_x = frame_width // 2
        self.sim_y = frame_height - 50
        # -90 = "up" in OpenCV (FORWARD), 0 = right, 90 = down, 180 = left
        self.sim_orientation = -90
        self.sim_color = (0, 255, 0)
        self.move_speed = 3
        self.turn_rate = 5  # degrees per step
        self.strafe_speed = 2  # Slower strafe speed

    def process_obstacle(self, x, y, w, h):
        """Processes motion detection for logging."""
        center_x = x + w // 2
        left_boundary = self.frame_width * 0.33
        right_boundary = self.frame_width * 0.67

        # Log the theoretical command (for records)
        zone = "CENTER"
        if center_x < left_boundary:
            zone = "LEFT"
        elif center_x > right_boundary:
            zone = "RIGHT"

        command = f"MOTION_STOP: {int(w * h)} area detected in {zone} zone."

        self.command_log.append({
            "timestamp": datetime.now(),
            "source": "MOTION",
            "zone": zone,
            "command": command
        })
        return zone, command

    def execute_command(self, command):
        """Translates high-level command (string) to simulation movement."""

        # Log the card command
        if command not in ["STOP_MOTION", "NO_COMMAND", "IDLE", "CRITICAL CRASH"]:
            self.command_log.append({
                "timestamp": datetime.now(),
                "source": "CARD",
                "zone": "N/A",
                "command": command
            })

        # --- COMMAND EXECUTION ---
        current_orientation_rad = np.deg2rad(self.sim_orientation)
        dx, dy = 0, 0  # Initialize movement vector components

        if command == "HALT" or command == "STOP_MOTION":
            self.sim_color = (0, 0, 255)  # Red (Halt/Error/Stop)
            # dx, dy remain 0

        elif command == "FORWARD":
            self.sim_color = (0, 255, 0)  # Green (Drive forward)
            # Movement vector is along the current orientation
            dx = int(self.move_speed * np.cos(current_orientation_rad))
            dy = int(self.move_speed * np.sin(current_orientation_rad))

        elif command == "BACKWARD":
            self.sim_color = (255, 165, 0)  # Orange (Reverse)
            # Movement vector is opposite the current orientation
            dx = -int(self.move_speed * 0.5 * np.cos(current_orientation_rad))  # slower reverse
            dy = -int(self.move_speed * 0.5 * np.sin(current_orientation_rad))

        elif command == "TURN LEFT":
            self.sim_color = (255, 255, 0)  # Yellow (Rotate left)
            self.sim_orientation -= self.turn_rate  # Counter-clockwise rotation

        elif command == "TURN RIGHT":
            self.sim_color = (0, 255, 255)  # Cyan (Rotate right)
            self.sim_orientation += self.turn_rate  # Clockwise rotation

        # --- NEW STRAFING COMMANDS (Body-Relative Movement) ---
        elif command == "MOVE LEFT":
            self.sim_color = (153, 50, 204)  # Slate Blue (Strafe left)
            # Strafe Left is 90 degrees COUNTER-CLOCKWISE from the current orientation
            strafe_angle_rad = current_orientation_rad - np.deg2rad(90)
            dx = int(self.strafe_speed * np.cos(strafe_angle_rad))
            dy = int(self.strafe_speed * np.sin(strafe_angle_rad))

        elif command == "MOVE RIGHT":
            self.sim_color = (255, 0, 255)  # Magenta (Strafe right)
            # Strafe Right is 90 degrees CLOCKWISE from the current orientation
            strafe_angle_rad = current_orientation_rad + np.deg2rad(90)
            dx = int(self.strafe_speed * np.cos(strafe_angle_rad))
            dy = int(self.strafe_speed * np.sin(strafe_angle_rad))

        else:  # Default: No Command / No Movement / Idle
            self.sim_color = (100, 100, 100)  # Gray
            # dx, dy remain 0

        # Apply movement vector
        self.sim_x += dx
        self.sim_y += dy

        # Keep spider on screen
        if self.sim_y < 20: self.sim_y = 20
        if self.sim_y > self.frame_height - 20: self.sim_y = self.frame_height - 20
        if self.sim_x < 20: self.sim_x = 20
        if self.sim_x > self.frame_width - 20: self.sim_x = self.frame_width - 20

    def draw_spider(self, frame, current_command_str="IDLE"):
        """Draws the simulated spider on the video frame."""
        sim_center = (int(self.sim_x), int(self.sim_y))

        # Draw body
        cv2.circle(frame, sim_center, 15, self.sim_color, -1)
        cv2.circle(frame, sim_center, 15, (255, 255, 255), 2)  # White outline

        # Draw orientation line (head/front)
        line_length = 20
        angle_rad = np.deg2rad(self.sim_orientation)

        end_x = int(self.sim_x + line_length * np.cos(angle_rad))
        end_y = int(self.sim_y + line_length * np.sin(angle_rad))

        cv2.line(frame, sim_center, (end_x, end_y), (255, 255, 255), 3)

        # Display current command near the spider
        cv2.putText(frame, current_command_str, (sim_center[0] + 20, sim_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def ensure_log_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        print(f"ERROR: Unable to create log directory: {path}")
        raise


def main():
    # --- Logging Setup ---
    try:
        ensure_log_dir(LOG_DIR)
    except Exception as e:
        print(e)
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    motion_log_path = LOG_DIR / f"motion_log_{ts}.txt"
    command_log_path = LOG_DIR / f"command_log_{ts}.txt"

    try:
        motion_log_file = motion_log_path.open("w")
        command_log_file = command_log_path.open("w")
    except Exception as e:
        print(f"ERROR: Failed to open log files: {e}")
        return

    motion_log_file.write("Timestamp,X,Y,Width,Height,Area,Zone\n")
    command_log_file.write("Timestamp,Source,Command\n")

    # --- YOLO Model Setup ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_FILE)

    if not os.path.exists(model_path):
        print(f"\nCRITICAL ERROR: Card Model file not found at: {model_path}")
        print("Please ensure 'yolov8s_playing_cards.pt' is in the same folder.")
        # Clean up logs and exit
        motion_log_file.close()
        command_log_file.close()
        return

    try:
        yolo_model = YOLO(model_path)
        class_names = yolo_model.names
        string_class_names = [str(name) for name in class_names.values()]
        print("\n" + "=" * 60)
        print("Model loaded successfully.")
        print(f"Card Classes: {', '.join(string_class_names)}")
        print(f"Safety Motion Thresh: >= {MIN_MOTION_AREA}")
        print(f"Small Item Range: {NOISE_FLOOR_AREA} < Area < {SMALL_ITEM_MAX_AREA}")
        print("=" * 60)
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        motion_log_file.close()
        command_log_file.close()
        return

    # --- Video Stream Setup ---
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video stream at {STREAM_URL}")
        motion_log_file.close()
        command_log_file.close()
        return

    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()
    if not ret or not ret2:
        print("ERROR: Unable to read initial frames from stream.")
        cap.release()
        motion_log_file.close()
        command_log_file.close()
        return

    frame_height, frame_width = frame1.shape[:2]
    spider = SpiderLegController(frame_width, frame_height)

    print("Starting combined control loop. Press 'q' to quit.")

    try:
        while cap.isOpened():
            # --- 1. DETERMINE CARD COMMAND (PRIMARY BRAIN) ---
            current_card_command = "IDLE"  # Default command is IDLE
            detected_cards_in_frame = []

            # Run YOLO on the frame
            results = yolo_model(frame1, stream=True, verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = math.ceil(box.conf[0] * 100) / 100
                    if conf > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cls_id = int(box.cls[0])
                        card_name = class_names[cls_id]  # e.g., '9D', 'KH', 'AC'

                        # Draw bounding box for the card
                        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange-ish box
                        cv2.putText(frame1, f'{card_name} ({conf})', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        detected_cards_in_frame.append(card_name)

            # --- Map detected cards to highest priority command (NEW LOGIC) ---
            if detected_cards_in_frame:
                print(f"Card Detected: {', '.join(detected_cards_in_frame)}")

                # 1. Check for CRITICAL CRASH override (9D)
                if "9D" in detected_cards_in_frame:
                    print("\033[91m!!! CRITICAL OVERRIDE: 9D (CRASH) DETECTED. EXITING PROGRAM !!!\033[0m")
                    # Log the command before crashing
                    command_log_file.write(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')},CARD,CRITICAL_CRASH_9D\n")
                    # Ensure logs are flushed and closed before exiting
                    motion_log_file.close()
                    command_log_file.close()
                    sys.exit(1)  # Crash the program

                # 2. Check for TURN commands (Ace of Clubs/Spades) - These take priority over moves
                elif "AC" in detected_cards_in_frame:
                    current_card_command = "TURN LEFT"
                elif "AS" in detected_cards_in_frame:
                    current_card_command = "TURN RIGHT"

                # 3. Check for generic directional commands (Suits D, H, C, S)
                else:
                    # Loop through detected cards to find the first valid suit command
                    for card_name in detected_cards_in_frame:
                        # Card names usually end in the suit letter (e.g., 'AS', '3C')
                        suit = card_name[-1]
                        # This now catches D, H, and the generic C/S (which are MOVE LEFT/RIGHT)
                        if suit in CARD_COMMANDS:
                            current_card_command = CARD_COMMANDS[suit]
                            break  # Use the first valid command found

            # --- 2. DETERMINE MOTION OVERRIDE (SAFETY STOP) AND COUNT SMALL ITEMS ---
            motion_override = False
            total_motion_area = 0
            small_item_count = 0

            # Compute motion difference
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, FRAME_DIFF_BLUR, 0)
            _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)  # Get coords once

                # --- 2a. SAFETY OVERRIDE (Motion >= MIN_MOTION_AREA) ---
                if area >= MIN_MOTION_AREA:
                    total_motion_area += area
                    motion_override = True
                    # Draw Red rectangle for filtered motion (Large/Safety)
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Log the motion event
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    zone, _ = spider.process_obstacle(x, y, w, h)
                    motion_log_file.write(f"{timestamp},{x},{y},{w},{h},{int(area)},{zone}\n")

                # --- 2b. SMALL ITEM VISUALIZATION/COUNT (NOISE_FLOOR_AREA <= area < SMALL_ITEM_MAX_AREA) ---
                elif NOISE_FLOOR_AREA <= area < SMALL_ITEM_MAX_AREA:
                    small_item_count += 1
                    center_x, center_y = x + w // 2, y + h // 2
                    # Draw blue circle for small items
                    cv2.circle(frame1, (center_x, center_y), int(w * 0.5), (255, 0, 0), 2)
                    # Small items DO NOT trigger the motion_override flag

            # --- 3. FINAL COMMAND DECISION ---
            final_command = current_card_command

            if motion_override:
                # Motion detection is the highest safety override
                final_command = "STOP_MOTION"
                print(
                    f"\033[91m!!! OVERRIDE: Large Motion Detected (Area: {int(total_motion_area)}) -> STOP !!!\033[0m")
                command_log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')},MOTION,STOP_MOTION\n")
                cv2.putText(frame1, "SAFETY STOP", (10, frame_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- Update and Draw Simulation ---
            spider.execute_command(final_command)
            spider.draw_spider(frame1, final_command)
            # ----------------------------------

            # --- Display Info ---
            cv2.putText(frame1, f"Command: {final_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame1, f"Safety Thresh: >{MIN_MOTION_AREA}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

            # Display Small Item Counter in bottom right
            counter_text = f"Small Items: {small_item_count}"
            text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = frame_width - text_size[0] - 10
            text_y = frame_height - 10
            # Draw a semi-transparent background rectangle for better readability
            cv2.rectangle(frame1, (text_x - 5, text_y - text_size[1] - 5), (frame_width, frame_height), (0, 0, 0), -1)
            cv2.putText(frame1, counter_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Combined Control (Card and Motion)", frame1)

            # advance frames
            frame1 = frame2
            ok, frame2 = cap.read()
            if not ok:
                print("Stream ended or failed to read next frame.")
                break

            # break on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        print("Unhandled exception:")
        traceback.print_exc()
    finally:
        # cleanup
        cap.release()
        cv2.destroyAllWindows()
        # Log files are closed upon sys.exit() or here.
        motion_log_file.close()
        command_log_file.close()
        print("\n" + "=" * 60)
        print("Session complete.")
        print("=" * 60)


if __name__ == "__main__":
    main()
