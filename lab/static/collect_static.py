"""
Capture your own static ASL landmarks from the webcam.

Keys:
  A–Y (except J) — select the letter you are signing
  SPACE / C     — save the current hand as that letter
  [ / ]         — previous / next letter
  Q / ESC       — quit

Saves 63-D normalize_landmarks vectors under data/static/webcam/<LETTER>/.
Those match the live static feature path, so retraining closes the domain gap.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from data import LETTERS
from datasets import WEBCAM_STATIC_DIR
from features import normalize_landmarks, selfie_handedness
from hand_tracking import (
    detect_hands,
    draw_landmarks,
    initialize_hand_detector,
)

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
WINDOW_NAME = "Collect static landmarks"


def _counts():
    """Per-letter sample counts already on disk."""
    counts = Counter()
    for letter in LETTERS:
        folder = WEBCAM_STATIC_DIR / letter
        if folder.is_dir():
            counts[letter] = len(list(folder.glob("*.npy")))
    return counts


def _save_sample(letter, features):
    """Write one 63-D feature vector for letter."""
    folder = WEBCAM_STATIC_DIR / letter
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"{letter}_{stamp}.npy"
    np.save(path, np.asarray(features, dtype=np.float32))
    return path


def run_collector(camera_index=0):
    """Interactive webcam landmark collector."""
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {HAND_MODEL_PATH}"
        )

    letter_index = 0
    counts = _counts()
    status = "Show a hand. SPACE saves the selected letter."

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with initialize_hand_detector(
        HAND_MODEL_PATH,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    ) as detector:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            timestamp_ms = frame_id
            frame_id += 1

            result = detect_hands(frame, detector, timestamp_ms)
            features = None
            handedness = "—"

            if result.hand_landmarks:
                draw_landmarks(frame, result)
                landmarks = result.hand_landmarks[0]
                handedness = selfie_handedness(
                    result.handedness[0][0].category_name
                )
                features = normalize_landmarks(landmarks, handedness)

            letter = LETTERS[letter_index]
            overlay = [
                f"Letter: {letter}  ( [ / ] to change )",
                f"Saved {letter}: {counts[letter]}",
                f"Hand: {handedness}",
                status,
                "SPACE/C save | Q quit",
            ]
            y = 28
            for line in overlay:
                cv2.putText(
                    frame,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (20, 20, 20),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                y += 28

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("["), ord("-")):
                letter_index = (letter_index - 1) % len(LETTERS)
                status = f"Selected {LETTERS[letter_index]}"
            elif key in (ord("]"), ord("="), ord("+")):
                letter_index = (letter_index + 1) % len(LETTERS)
                status = f"Selected {LETTERS[letter_index]}"
            elif ord("A") <= key <= ord("Z") or ord("a") <= key <= ord("z"):
                chosen = chr(key).upper()
                if chosen in LETTERS:
                    letter_index = LETTERS.index(chosen)
                    status = f"Selected {chosen}"
                elif chosen in ("J", "Z"):
                    status = f"{chosen} needs motion — skip for static"
            elif key in (ord(" "), ord("c"), ord("C")):
                if features is None:
                    status = "No hand — nothing saved"
                else:
                    path = _save_sample(letter, features)
                    counts[letter] += 1
                    status = f"Saved {path.name} ({counts[letter]} total)"

    cap.release()
    cv2.destroyAllWindows()

    total = sum(counts.values())
    print(f"\nWebcam static samples: {total}")
    for letter in LETTERS:
        if counts[letter]:
            print(f"  {letter}: {counts[letter]}")
    print(f"Folder: {WEBCAM_STATIC_DIR}")
    print("Next: python dev.py prepare-static webcam")
    print("Then: python dev.py train-static webcam-mix --install")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_collector(args.camera)
