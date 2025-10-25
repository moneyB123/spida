# rpi spida - Card Recognition v0.1
# Date: 10/25/2025
# Author: Dalton Alwin Nisbett
# Location: Dayton, OH USA
"""
This script uses a pre-trained YOLO model to perform
real-time object recognition on a video stream.
It is designed to detect and identify playing cards.
"""

import cv2
import os
from ultralytics import YOLO
import math

# ----- Configuration -----
STREAM_URL = "http://192.168.0.101:8080/stream"

# !!! CRITICAL: UPDATE THIS PATH !!!
# You must find the actual YOLO model file (it will end in .pt)
# that was inside your downloaded zip file.
#
# Common names are 'best.pt' or 'yolov8s_playing_cards.pt'.
#
# If the file is named 'best.pt' and is inside the folder you extracted,
# the path might look like this:
# MODEL_FILE = 'Playing-Cards-Detection-with-YoloV8-main/best.pt'
#
# If it is directly in the same directory as this script,
# the path is just the filename:
MODEL_FILE = 'yolov8s_playing_cards.pt'


# !!! ----------------------------- !!!


def main():
    # --- Resolve Model Path ---
    # This checks if the file exists relative to the script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_FILE)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at path: {model_path}")
        print("Please check the 'MODEL_FILE' variable in the script.")
        print("It should point to the .pt file inside your downloaded project folder.")
        return

    # Load the pre-trained YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model from '{model_path}'.")
        print(f"Error details: {e}")
        return

    # Get the class names from the model
    # Note: If the model loaded correctly, this will show the names (e.g., 'King of Spades', '3 of Hearts')
    class_names = model.names
    print("Model loaded successfully.")
    print(f"Model File Used: {MODEL_FILE}")
    print(f"Detected Class Names (Total {len(class_names)}):")

    # --- FIX: Convert class names to strings if they are integers ---
    # This handles older YOLO model files that store class names as integers
    # which causes a TypeError when trying to join them with the print statement.
    string_class_names = [str(name) for name in class_names.values()]
    print(", ".join(string_class_names))
    print("-" * 60)

    # Open video stream
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video stream at {STREAM_URL}")
        return

    # Try to get frame dimensions, using defaults if stream isn't ready
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 480

    print(f"Frame size: {frame_width}x{frame_height}")
    print("Starting card detection...")
    print("Press 'q' to quit.")
    print("-" * 60)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Attempt to restart stream if it fails
                cap.release()
                cap = cv2.VideoCapture(STREAM_URL)
                print("Stream failed, attempting to reconnect...")
                if not cap.isOpened():
                    # If reconnect fails, break out
                    print("Reconnection failed.")
                    break
                continue  # Skip processing this bad frame

            # Run the YOLO model on the frame
            results = model(frame, stream=True, verbose=False)

            detected_cards_in_frame = []
            detection_successful = False

            # Loop through all detected objects in the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence score
                    conf = math.ceil(box.conf[0] * 100) / 100

                    # Get class ID and name
                    cls_id = int(box.cls[0])
                    # class_names[cls_id] is used directly here and it works because
                    # it is an indexing operation (lookup), not a joining operation.
                    class_name = class_names[cls_id]

                    # Only draw if confidence is above a threshold (e.g., 50%)
                    if conf > 0.50:
                        detection_successful = True

                        # Draw the bounding box (color: Green)
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Prepare the label text
                        label = f'{class_name} ({conf})'

                        # Put the label above the box (Color: Green)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        detected_cards_in_frame.append(label)

            # Print all detected cards for this frame
            if detected_cards_in_frame:
                print(f"Detected: {', '.join(detected_cards_in_frame)}")

            # Display the frame
            cv2.imshow("Card Detection", frame)

            # break on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("Card detection session complete.")
        print("=" * 60)


if __name__ == "__main__":
    main()
