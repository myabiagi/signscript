from collections import Counter
from pathlib import Path
import re

import cv2
import mediapipe as mp
import numpy as np

from extract_features import normalize_landmarks

DATASET_PATH = Path("data/alphabet/asl_hg")
FEATURES_PATH = Path("data/processed/alphabet_asl_hg_features.npz")
MODEL_PATH = Path("models/hand_landmarker.task")

LETTERS = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           if letter not in "JZ"]

def get_image_info(image_path):
    """Extract the participant and letter from an ASL-HG filename."""
    match = re.fullmatch(
        r"(P\d+)_([A-Z0-9])_(\d+)",
        image_path.stem
    )

    if match is None:
        raise ValueError(f"Unexpected image filename: {image_path.name}")

    participant_id, letter, _ = match.groups()

    if letter != image_path.parent.name.upper():
        raise ValueError(f"Folder and filename disagree: {image_path}")

    return participant_id, letter

def prepare_alphabet():
    """Extract normalized hand landmarks from the ASL-HG dataset."""
    image_paths = sorted(
        path for path in DATASET_PATH.rglob("*.jpg")
        if path.parent.name.upper() in LETTERS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No alphabet images found in {DATASET_PATH}"
        )

    X = []
    y = []
    source_paths = []
    participant_ids = []

    # Use MediaPipe IMAGE mode for individual dataset images
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1
    )

    detected = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for index, image_path in enumerate(image_paths, start=1):
            participant_id, letter = get_image_info(image_path)

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                continue

            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            features = normalize_landmarks(landmarks, handedness)

            X.append(features)
            y.append(letter)
            source_paths.append(str(image_path))
            participant_ids.append(participant_id)

            detected += 1

            if index % 500 == 0:
                print(f"Processed {index}/{len(image_paths)} images")

    if not X:
        raise RuntimeError("No usable hand landmarks were extracted.")

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        FEATURES_PATH,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y),
        source_paths=np.asarray(source_paths),
        participant_ids=np.asarray(participant_ids),
        letters=np.asarray(LETTERS),
        attempted=len(image_paths),
        detected=detected
    )

    print(f"\nImages attempted: {len(image_paths)}")
    print(f"Hands detected: {detected}")
    print(f"Features shape: ({len(X)}, 63)")

    print("\nUsable examples per participant:")
    for participant, count in sorted(
        Counter(participant_ids).items(),
        key=lambda item: int(item[0][1:])
    ):
        print(f"{participant}: {count}")

    print(f"\nSaved features to {FEATURES_PATH}")

if __name__ == "__main__":
    prepare_alphabet()
