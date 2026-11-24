from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from extract_features import (
    initialize_hand_detector,
    detect_hands,
    normalize_landmarks
)

def load_image(image_path):
    """
    Load a dataset image using OpenCV.

    Args:
        image_path (str or Path): Path to the image.

    Returns:
        numpy.ndarray: Loaded image in BGR format.
    """
    # Read the image from disk
    image = cv2.imread(str(image_path))

    # Check whether the image was loaded successfully
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image

def prepare_alphabet_data(dataset_path, model_path, max_per_letter=200, seed=42):
    """
    Extract landmark features from randomly sampled static ASL letters.

    Args:
        dataset_path (str or Path): Directory containing A-Z folders.
        model_path (str or Path): Path to the MediaPipe model.
        max_per_letter (int or None): Images to sample per letter; None uses all.
        seed (int): Random seed for reproducible sampling.

    Returns:
        tuple: Features (X), labels (y), source paths, and detection statistics.
    """
    features = []
    labels = []
    source_paths = []
    statistics = []
    rng = np.random.default_rng(seed)
    dataset_path = Path(dataset_path)

    # Initialize MediaPipe for independent images
    with initialize_hand_detector(
        model_path,
        mp.tasks.vision.RunningMode.IMAGE
    ) as detector:

        # Process the 24 static letters (J and Z require motion)
        for letter_path in sorted(dataset_path.iterdir()):
            if not letter_path.is_dir():
                continue

            letter = letter_path.name.upper()

            if len(letter) != 1 or not letter.isalpha() or letter in ("J", "Z"):
                continue

            # Find all JPG images for the current letter
            image_paths = sorted(letter_path.glob("*.jpg"))

            # Randomly sample images without repetition
            if max_per_letter is not None and len(image_paths) > max_per_letter:
                indices = rng.choice(
                    len(image_paths),
                    size=max_per_letter,
                    replace=False
                )
                image_paths = [image_paths[i] for i in sorted(indices)]

            detected = 0

            for image_path in image_paths:
                # Load the image
                image = load_image(image_path)

                # Detect the hand
                result = detect_hands(image, detector)

                if not result.hand_landmarks:
                    continue

                # Normalize the first detected hand
                hand = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name

                normalized = normalize_landmarks(hand, handedness)

                features.append(normalized)
                labels.append(letter)
                source_paths.append(str(image_path.relative_to(dataset_path)))
                detected += 1

            # Record detection statistics for each letter
            attempted = len(image_paths)
            statistics.append((letter, attempted, detected))

            print(f"{letter}: {detected}/{attempted} hands detected")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(source_paths),
        statistics
    )

if __name__ == "__main__":
    dataset_path = Path("data/alphabet/signalphaset")
    model_path = Path("models/hand_landmarker.task")
    output_path = Path("data/processed/alphabet_features.npz")

    # Prepare alphabet features
    X, y, source_paths, statistics = prepare_alphabet_data(
        dataset_path,
        model_path,
        max_per_letter=200,
        seed=42
    )

    if len(X) == 0:
        raise ValueError("No hands detected; feature dataset was not saved.")

    # Create the output folder and save the features
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        source_paths=source_paths,
        letters=np.asarray([row[0] for row in statistics]),
        attempted=np.asarray([row[1] for row in statistics]),
        detected=np.asarray([row[2] for row in statistics]),
        seed=np.asarray(42)
    )

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    print(f"Saved features to: {output_path}")
    print(
        f"Detection total: "
        f"{sum(row[2] for row in statistics)}/"
        f"{sum(row[1] for row in statistics)}"
    )
