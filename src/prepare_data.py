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

def prepare_alphabet_data(dataset_path, model_path, max_per_letter=50):
    """
    Extract normalized hand landmarks from static ASL letters.

    Args:
        dataset_path (str or Path): Directory containing letter folders.
        model_path (str or Path): Path to the MediaPipe model.
        max_per_letter (int): Maximum images to process per letter.

    Returns:
        tuple: Feature vectors (X), letter labels (y).
    """
    features = []
    labels = []

    # Initialize MediaPipe for independent images
    with initialize_hand_detector(
        model_path,
        mp.tasks.vision.RunningMode.IMAGE
    ) as detector:

        # Process each static alphabet letter
        for letter_path in sorted(Path(dataset_path).iterdir()):
            if not letter_path.is_dir():
                continue

            letter = letter_path.name

            # J and Z require motion recognition
            if letter in ("J", "Z"):
                continue

            image_paths = sorted(letter_path.glob("*.jpg"))

            # Process a limited number of images per letter
            for image_path in image_paths[:max_per_letter]:
                image = load_image(image_path)

                # Detect the hand
                result = detect_hands(image, detector)

                if not result.hand_landmarks:
                    continue

                # Get the first hand and its handedness
                hand = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name

                # Normalize the landmarks
                normalized = normalize_landmarks(hand, handedness)

                features.append(normalized)
                labels.append(letter)

            print(f"Processed letter {letter}")

    return np.array(features), np.array(labels)