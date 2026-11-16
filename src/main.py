import time
from pathlib import Path

import cv2

from extract_features import (
    initialize_hand_detector,
    detect_hands,
    draw_landmarks
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
WINDOW_NAME = "SignScript - Hand Tracking"

def run_webcam(camera_index=0):
    """
    Capture webcam video and display detected hand landmarks.

    Args:
        camera_index (int): Index of the webcam to use (default: 0).

    Returns:
        None
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise IOError(f"Could not open webcam: {camera_index}")

    try:
        with initialize_hand_detector(MODEL_PATH) as detector:
            cv2.namedWindow(WINDOW_NAME)

            while True:
                # Read the next webcam frame
                ret, frame = cap.read()

                if not ret:
                    print("Could not read a frame from the webcam.")
                    break

                # Detect hands in the current frame
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = detect_hands(frame, detector, timestamp_ms)

                # Draw detected landmarks
                frame = draw_landmarks(frame, result)

                # Display the processed frame
                cv2.imshow(WINDOW_NAME, frame)

                # Process window events
                cv2.waitKey(1)

                # Exit when the window is closed
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

    finally:
        # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
