# rpi spida v0.2.3 - immediate stop on any motion + VISUAL SIMULATION
# Date: 10/25/2025
# Author: Dalton Alwin Nisbett
# Location: Dayton, OH USA
"""
Spider Leg Controller with Motion Detection & Visual Simulation
- If any movement is detected in the frame, print "STOP: Movement detected"
- Logs detections and commands to timestamped files
- A simple spider is simulated in the OpenCV window:
  - Moves FORWARD when no motion is detected.
  - STOPS and turns RED when motion is detected.
"""

from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import sys
import traceback

# ----- Configuration -----
STREAM_URL = "http://192.168.0.101:8080/stream"
MIN_MOTION_AREA = 1  # >=1 means ANY detected contour triggers STOP (very sensitive)
FRAME_DIFF_BLUR = (5, 5)  # smoothing kernel for diff
THRESHOLD_VALUE = 20  # binary threshold for motion
DILATE_ITERATIONS = 3  # expand detected blobs
LOG_DIR = Path("spider_logs")  # Using a specific log directory name


# ----- Spider controller (holds simulation state) -----
class SpiderLegController:
    def __init__(self, frame_width, frame_height):
        self.leg_positions = [0, 0, 0, 0, 0, 0]
        self.command_log = []

        # Simulation attributes
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sim_x = frame_width // 2  # Starting X position (center)
        self.sim_y = frame_height - 50  # Starting Y position (bottom)
        self.sim_orientation = -90  # Angle in degrees (-90 = "up" in OpenCV)
        self.sim_color = (0, 255, 0)  # Start green (moving)
        self.move_speed = 3

    def process_obstacle(self, x, y, w, h):
        """
        Classify obstacle zone and create a command; returns (zone, command).
        This is used for LOGGING purposes in this version.
        """
        center_x = x + w // 2
        left_boundary = self.frame_width * 0.33
        right_boundary = self.frame_width * 0.67

        if center_x < left_boundary:
            zone = "LEFT"
            command = "TURN RIGHT: Extend legs [4,5,6], Retract [1,2,3]"
        elif center_x > right_boundary:
            zone = "RIGHT"
            command = "TURN LEFT: Extend legs [1,2,3], Retract [4,5,6]"
        else:
            zone = "CENTER"
            command = "STOP/BACKUP: All legs retract, reverse gait"

        if w * h > 50000:
            command += " - CLOSE OBSTACLE!"

        # Log the theoretical command
        self.command_log.append({
            "timestamp": datetime.now(),
            "zone": zone,
            "command": command
        })

        return zone, command

    def update_simulation_state(self, main_command):
        """
        Updates the spider's simulated position and color based on the
        OVERALL command (FORWARD or STOP).
        """
        if main_command == "STOP":
            self.sim_color = (0, 0, 255)  # Turn RED
            # No movement

        elif main_command == "FORWARD":
            self.sim_color = (0, 255, 0)  # Turn GREEN

            # Calculate forward movement
            angle_rad = np.deg2rad(self.sim_orientation)
            self.sim_x += int(self.move_speed * np.cos(angle_rad))
            self.sim_y += int(self.move_speed * np.sin(angle_rad))

            # Keep spider on screen (simple boundary check)
            if self.sim_y < 20: self.sim_y = 20
            if self.sim_y > self.frame_height - 20: self.sim_y = self.frame_height - 20
            if self.sim_x < 20: self.sim_x = 20
            if self.sim_x > self.frame_width - 20: self.sim_x = self.frame_width - 20

    def draw_spider(self, frame):
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


def ensure_log_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        print(f"ERROR: Unable to create log directory: {path}")
        # Re-raise to stop execution if we can't log
        raise


def main():
    # prepare logs
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

    motion_log_file.write("Timestamp,X,Y,Width,Height,Area\n")
    command_log_file.write("Timestamp,Zone,Command\n")

    # open video stream
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

    # --- Initialize Spider Controller AFTER getting frame size ---
    spider = SpiderLegController(frame_width, frame_height)

    print(f"Frame size: {frame_width}x{frame_height}")
    print("Spider Controller initialized - 6 legs ready")
    print(f"Logs will be saved to: {LOG_DIR.resolve()}")
    print("-" * 60)

    detection_count = 0

    try:
        while cap.isOpened():
            # compute frame difference
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, FRAME_DIFF_BLUR, 0)
            _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)

            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detected_any_motion = False
            total_motion_area = 0
            sim_main_command = "FORWARD"  # Default to FORWARD

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_MOTION_AREA:
                    continue

                total_motion_area += area
                detected_any_motion = True

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                # Log the theoretical zone/command for patent data
                zone, command = spider.process_obstacle(x, y, w, h)

                motion_log_file.write(f"{timestamp},{x},{y},{w},{h},{int(area)}\n")
                command_log_file.write(f"{timestamp},{zone},{command}\n")
                detection_count += 1

            # If ANY motion in this frame -> immediate STOP
            if detected_any_motion:
                sim_main_command = "STOP"  # Set main command for simulation
                stop_msg = f"*** STOP: Movement detected (total_area={int(total_motion_area)}) ***"
                print(stop_msg)

                # also write a global STOP line to the command log
                command_log_file.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')},GLOBAL,STOP: Movement detected\n")

                # overlay STOP on video feed
                cv2.putText(frame1, "STOP: Movement detected", (10, frame_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # No motion, so command is FORWARD
                sim_main_command = "FORWARD"
                print("No motion detected. Executing: FORWARD")

            # --- Update and Draw Simulation ---
            spider.update_simulation_state(sim_main_command)
            spider.draw_spider(frame1)
            # ----------------------------------

            # overlay some useful info
            cv2.putText(frame1, f"Detections: {detection_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame1, f"MotionArea: {int(total_motion_area)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            left_line = int(frame_width * 0.33)
            right_line = int(frame_width * 0.67)
            cv2.line(frame1, (left_line, 0), (left_line, frame_height), (255, 0, 0), 1)
            cv2.line(frame1, (right_line, 0), (right_line, frame_height), (255, 0, 0), 1)

            cv2.imshow("Motion Detection", frame1)

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
        motion_log_file.close()
        command_log_file.close()
        print("\n" + "=" * 60)
        print("Session complete:")
        print(f"   Total individual detections logged: {detection_count}")
        print(f"   Theoretical commands logged: {len(spider.command_log)}")
        print(f"   Logs saved in: {LOG_DIR.resolve()}")
        print("=" * 60)


if __name__ == "__main__":
    main()
