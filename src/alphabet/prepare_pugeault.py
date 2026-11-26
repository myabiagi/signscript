"""Prepare Pugeault & Bowden Dataset A for SignScript."""

from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from extract_features import normalize_landmarks
from geometric_features import enhance_features

DATASET_PATH = Path("data/alphabet/pugeault_bowden")
OUTPUT_PATH = Path("data/processed/alphabet_pugeault_geometric_features.npz")
MODEL_PATH = Path("models/hand_landmarker.task")

PARTICIPANTS = ("A", "B", "C", "D", "E")
LETTERS = tuple(letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                if letter not in "JZ")

SAMPLES_PER_LETTER_PER_PARTICIPANT = 25

def sample_images(folder, count):
    """Sample images across the folder's full sequence."""
    images = sorted(
        folder.glob("color_*.png")
    )

    if not images:
        return []

    if len(images) <= count:
        return images

    indices = np.linspace(
        0,
        len(images) - 1,
        num=count,
        dtype=int
    )

    return [images[index] for index in np.unique(indices)]

def collect_images():
    """Collect labelled RGB paths with participant information."""
    samples = []

    for participant in PARTICIPANTS:
        participant_path = DATASET_PATH / participant

        if not participant_path.is_dir():
            raise FileNotFoundError(
                f"Missing participant folder: {participant_path}"
            )

        for letter in LETTERS:
            letter_path = participant_path / letter.lower()

            if not letter_path.is_dir():
                raise FileNotFoundError(
                    f"Missing letter folder: {letter_path}"
                )

            selected = sample_images(
                letter_path,
                SAMPLES_PER_LETTER_PER_PARTICIPANT
            )

            if not selected:
                raise ValueError(
                    f"No RGB images found in {letter_path}"
                )

            for image_path in selected:
                samples.append(
                    (image_path, letter, participant)
                )

    return samples

def prepare_pugeault():
    """Extract normalized and geometric features from sampled images."""
    samples = collect_images()

    print(f"Selected RGB images: {len(samples)}")
    print(f"Sampling up to {SAMPLES_PER_LETTER_PER_PARTICIPANT} "
          "images per letter per participant")

    X = []
    y = []
    source_paths = []
    participant_ids = []

    failures = Counter()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        for index, (image_path, letter, participant) in enumerate(
            samples, start=1
        ):
            image = cv2.imread(str(image_path))

            if image is None:
                failures[(participant, letter)] += 1
                continue

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                failures[(participant, letter)] += 1
                continue

            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            original_features = normalize_landmarks(
                landmarks,
                handedness
            )

            geometric_features = enhance_features(
                original_features
            )

            X.append(geometric_features)
            y.append(letter)
            source_paths.append(str(image_path))
            participant_ids.append(participant)

            if index % 250 == 0:
                print(
                    f"Processed {index}/{len(samples)} images",
                    flush=True
                )

    if not X:
        raise RuntimeError(
            "MediaPipe could not detect any usable hands."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    source_paths = np.asarray(source_paths)
    participant_ids = np.asarray(participant_ids)

    # Count exact duplicates without removing them.
    duplicate_count = len(X) - len(
        np.unique(X, axis=0)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        source_paths=source_paths,
        participant_ids=participant_ids,
        attempted=len(samples),
        detected=len(X)
    )

    print("\nPugeault preparation complete")
    print(f"Images attempted: {len(samples)}")
    print(f"Hands detected: {len(X)}")
    print(f"Detection rate: {len(X) / len(samples):.2%}")
    print(f"Feature shape: {X.shape}")
    print(f"Exact duplicate vectors: {duplicate_count}")

    print("\nUsable examples by letter:")
    letter_counts = Counter(y)

    for letter in LETTERS:
        print(
            f"{letter}: {letter_counts.get(letter, 0)}"
        )

    print("\nUsable examples by participant:")
    participant_counts = Counter(participant_ids)

    for participant in PARTICIPANTS:
        print(
            f"{participant}: "
            f"{participant_counts.get(participant, 0)}"
        )

    if failures:
        print("\nDetection failures by participant and letter:")

        for (participant, letter), count in sorted(
            failures.items()
        ):
            print(f"{participant}/{letter}: {count}")

    print(f"\nSaved features to: {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_pugeault()
