from pathlib import Path
import time

import cv2

from extract_features import (
    initialize_hand_detector,
    detect_hands,
    draw_landmarks,
    normalize_landmarks,
)
from inference import load_alphabet_model, predict_letter

MODEL_PATH = Path("models/hand_landmarker.task")
WINDOW_NAME = "SignScript"

def run_webcam(camera_index=0):
    """Capture webcam frames and display ASL letter predictions."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        alphabet_model = load_alphabet_model()

        with initialize_hand_detector(MODEL_PATH) as detector:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Detect hands in the current frame
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = detect_hands(frame, detector, timestamp_ms)

                # Predict the letter using the first detected hand
                prediction = "No hand detected"
                handedness = "None"

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    handedness = result.handedness[0][0].category_name
                    print(f"Detected hand: {handedness}", end="\r")
                    
                    features = normalize_landmarks(
                        landmarks, handedness
                    )
                    prediction = predict_letter(alphabet_model, features)

                # Draw detected hand landmarks
                frame = draw_landmarks(frame, result)
               
                # Display the predicted letter
                cv2.putText(
                    frame,
                    f"Prediction: {prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # Display the detected hand
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
                cv2.waitKey(1)

                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
