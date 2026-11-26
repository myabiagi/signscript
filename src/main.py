from collections import Counter, deque
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

SMOOTHING_WINDOW = 6
MIN_MATCHING_PREDICTIONS = 4

def smooth_prediction(history, previous_prediction):
    """Keep the last stable letter until a new letter has enough votes."""
    if len(history) < MIN_MATCHING_PREDICTIONS:
        return previous_prediction

    letter, count = Counter(history).most_common(1)[0]

    if count >= MIN_MATCHING_PREDICTIONS:
        return letter

    return previous_prediction


def run_webcam(camera_index=0):
    """Display the smoothed ASL letter prediction."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    stable_prediction = "Uncertain"

    try:
        alphabet_model = load_alphabet_model()

        with initialize_hand_detector(MODEL_PATH) as detector:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = detect_hands(frame, detector, timestamp_ms)

                smoothed_prediction = "No hand detected"
                handedness = "None"

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    handedness = result.handedness[0][0].category_name

                    features = normalize_landmarks(
                        landmarks, handedness
                    )

                    raw_prediction = str(
                        predict_letter(alphabet_model, features)
                    )

                    prediction_history.append(raw_prediction)

                    stable_prediction = smooth_prediction(
                        prediction_history,
                        stable_prediction
                    )

                    smoothed_prediction = stable_prediction

                else:
                    prediction_history.clear()
                    stable_prediction = "Uncertain"

                frame = draw_landmarks(frame, result)

                cv2.putText(
                    frame,
                    f"Letter: {smoothed_prediction}",
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
