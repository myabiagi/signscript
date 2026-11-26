"""SignScript webcam interface."""

from pathlib import Path
import time

import cv2

from hand_tracking import (
    initialize_hand_detector,
    detect_hands,
    draw_landmarks,
)
from features import normalize_landmarks
from inference import load_static_model, predict_static_letter
from smoothing import PredictionSmoother

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
WINDOW_NAME = "SignScript"

def run_webcam(camera_index=0):
    """Display stable static ASL letter predictions."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    smoother = PredictionSmoother()

    try:
        static_model = load_static_model()

        with initialize_hand_detector(HAND_MODEL_PATH) as detector:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = detect_hands(
                    frame, detector, timestamp_ms
                )

                prediction = "No hand detected"
                handedness = "None"

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    handedness = (
                        result.handedness[0][0].category_name
                    )

                    features = normalize_landmarks(
                        landmarks, handedness
                    )

                    raw_prediction = predict_static_letter(
                        static_model, features
                    )

                    prediction = smoother.update(raw_prediction)

                else:
                    smoother.reset()

                frame = draw_landmarks(frame, result)

                cv2.putText(
                    frame,
                    f"Letter: {prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Hand: {handedness}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()