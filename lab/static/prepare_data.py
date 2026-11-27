"""Prepare ASL-HG, Pugeault, Graudt, and local webcam static-letter datasets."""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from features import normalize_landmarks, enhance_features
from datasets import (
    collect_asl_hg,
    collect_pugeault,
    collect_graudt,
    collect_webcam_static,
    collect_asl_dynamic_mid,
)
from data import LETTERS
from hand_tracking import initialize_hand_detector, detect_hands

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
OUTPUT_DIR = Path("data/processed")


def _ensure_normalized_63(points):
    """
    Convert Graudt landmarks to our 63-D wrist+scale format.

    Accepts (21, 3) or already-flattened (63,) — the Kaggle zip ships (63,).
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape == (63,):
        points = points.reshape(21, 3)
    elif points.shape != (21, 3):
        raise ValueError(
            f"Expected landmarks shape (21, 3) or (63,), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("Landmarks contain NaN or infinity.")

    centered = points - points[0]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale > 0:
        centered = centered / scale
    return centered.reshape(-1).astype(np.float32)


def prepare_images(dataset, samples, output, use_geometric_features):
    """Run MediaPipe on image datasets (ASL-HG / Pugeault)."""
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {HAND_MODEL_PATH}"
        )

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

    _save_npz(
        output, X, y, source_paths, participant_ids, len(samples)
    )
    print(f"\nDetection failures: {sum(failures.values())}")


def prepare_landmark_files(dataset, samples, output, already_63):
    """
    Load .npy landmark files into a processed npz (always 63-D).

    Graudt stores (21, 3); local webcam collect stores (63,).
    """
    X = []
    y = []
    source_paths = []
    participant_ids = []
    failures = 0

    print(f"Dataset: {dataset}")
    print(f"Selected files: {len(samples)}")

    for index, (path, letter, participant) in enumerate(
        samples, start=1
    ):
        try:
            raw = np.load(path)
            if already_63:
                features = np.asarray(raw, dtype=np.float32).reshape(-1)
                if features.shape[0] != 63:
                    raise ValueError(
                        f"Expected 63-D vector, got {features.shape}"
                    )
                if not np.isfinite(features).all():
                    raise ValueError("Non-finite features")
            else:
                features = _ensure_normalized_63(raw)
        except Exception as exc:
            failures += 1
            print(f"Skip {path}: {exc}")
            continue

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
            "No usable landmark files were loaded."
        )

    _save_npz(
        output, X, y, source_paths, participant_ids, len(samples)
    )
    print(f"\nSkipped files: {failures}")


def _save_npz(output, X, y, source_paths, participant_ids, attempted):
    """Write compressed feature cache and print a short summary."""
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
        attempted=attempted,
        detected=len(X),
    )

    print("\nPreparation complete")
    print(f"Attempted: {attempted}")
    print(f"Detected: {len(X)}")
    print(f"Detection rate: {len(X) / max(attempted, 1):.2%}")
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

    print(f"\nSaved: {output}")


def prepare(dataset):
    """Extract / convert landmarks for the selected static dataset."""
    if dataset == "asl-hg":
        prepare_images(
            dataset,
            collect_asl_hg(),
            OUTPUT_DIR / "static_asl_hg.npz",
            use_geometric_features=False,
        )
    elif dataset == "pugeault":
        prepare_images(
            dataset,
            collect_pugeault(),
            OUTPUT_DIR / "static_pugeault.npz",
            use_geometric_features=True,
        )
    elif dataset == "graudt":
        prepare_landmark_files(
            dataset,
            collect_graudt(),
            OUTPUT_DIR / "static_graudt.npz",
            already_63=False,
        )
    elif dataset == "webcam":
        prepare_landmark_files(
            dataset,
            collect_webcam_static(),
            OUTPUT_DIR / "static_webcam.npz",
            already_63=True,
        )
    elif dataset == "asl-dynamic-mid":
        prepare_images(
            dataset,
            collect_asl_dynamic_mid(),
            OUTPUT_DIR / "static_asl_dynamic_mid.npz",
            use_geometric_features=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=(
            "asl-hg",
            "pugeault",
            "graudt",
            "webcam",
            "asl-dynamic-mid",
        ),
    )

    args = parser.parse_args()
    prepare(args.dataset)
