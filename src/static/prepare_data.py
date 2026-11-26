"""Prepare ASL-HG and Pugeault & Bowden static-letter datasets."""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from features import normalize_landmarks, enhance_features
from datasets import collect_asl_hg, collect_pugeault
from data import LETTERS
from hand_tracking import initialize_hand_detector, detect_hands

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
OUTPUT_DIR = Path("data/processed")

def prepare(dataset):
    """Extract normalized landmarks from the selected dataset."""
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {HAND_MODEL_PATH}"
        )

    if dataset == "asl-hg":
        samples = collect_asl_hg()
        output = OUTPUT_DIR / "static_asl_hg.npz"
        use_geometric_features = False

    elif dataset == "pugeault":
        samples = collect_pugeault()
        output = OUTPUT_DIR / "static_pugeault.npz"
        use_geometric_features = True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X = []
    y = []
    source_paths = []
    participant_ids = []
    failures = Counter()

    print(f"Dataset: {dataset}")
    print(f"Selected images: {len(samples)}")

    with initialize_hand_detector(
        HAND_MODEL_PATH,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    ) as detector:

        for index, (path, letter, participant) in enumerate(
            samples, start=1
        ):
            image = cv2.imread(str(path))

            if image is None:
                failures[(participant, letter)] += 1
                continue

            result = detect_hands(image, detector)

            if not result.hand_landmarks:
                failures[(participant, letter)] += 1
                continue

            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            features = normalize_landmarks(
                landmarks, handedness
            )

            if use_geometric_features:
                features = enhance_features(features)

            X.append(features)
            y.append(letter)
            source_paths.append(str(path))
            participant_ids.append(participant)

            if index % 500 == 0:
                print(
                    f"Processed {index}/{len(samples)}",
                    flush=True,
                )

    if not X:
        raise RuntimeError(
            "No usable landmarks were extracted."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    source_paths = np.asarray(source_paths)
    participant_ids = np.asarray(participant_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        X=X,
        y=y,
        source_paths=source_paths,
        participant_ids=participant_ids,
        attempted=len(samples),
        detected=len(X),
    )

    print("\nPreparation complete")
    print(f"Attempted: {len(samples)}")
    print(f"Detected: {len(X)}")
    print(f"Detection rate: {len(X) / len(samples):.2%}")
    print(f"Feature dimensions: {X.shape[1]}")

    print("\nExamples by letter:")
    for letter in LETTERS:
        count = int(np.sum(y == letter))
        print(f"{letter}: {count}")

    print("\nExamples by participant:")
    for participant, count in sorted(
        Counter(participant_ids).items()
    ):
        print(f"{participant}: {count}")

    print(f"\nDetection failures: {sum(failures.values())}")
    print(f"Saved: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=("asl-hg", "pugeault"),
    )

    args = parser.parse_args()
    prepare(args.dataset)