"""
Extract raw MediaPipe landmark sequences for Experiment B.

Classes:
  - J, Z: motion-dependent letters (positives)
  - I, D: static letters used as negative examples

Saves variable-length (Ti, 21, 3) sequences so raw / frame / sequence
normalization can be compared later without re-running MediaPipe.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from features import landmarks_array_from_hand
from hand_tracking import detect_hands, initialize_hand_detector

# Dynamic Experiment B vocabulary
DYNAMIC_CLASSES = ("J", "Z", "I", "D")

DATASET_PATH = Path("data/ASL_dynamic")
MODEL_PATH = Path("models/hand_landmarker.task")
FEATURES_PATH = Path("data/processed/dynamic_raw_sequences.npz")

# Require at least this many successful hand detections per clip
MIN_VALID_FRAMES = 5

def list_clip_frames(clip_path):
    """
    List frame image paths for one clip folder in sorted order.

    Args:
        clip_path (str or Path): Directory containing frame_*.jpg files.

    Returns:
        list: Sorted frame paths.
    """
    clip_path = Path(clip_path)
    frames = sorted(clip_path.glob("frame_*.jpg"))

    if not frames:
        frames = sorted(clip_path.glob("*.jpg"))

    return frames

def load_image(image_path):
    """Load a BGR image or raise if reading fails."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image

def extract_clip_sequence(clip_path, detector):
    """
    Run MediaPipe on every frame and collect raw (21, 3) landmarks.

    Args:
        clip_path (str or Path): Clip frame folder.
        detector: HandLandmarker in IMAGE mode.

    Returns:
        tuple: (sequence, handedness_label, num_valid)
            sequence has shape (T, 21, 3) with NaNs where detection failed.
            handedness_label is the majority Left/Right label or None.
    """
    frame_paths = list_clip_frames(clip_path)

    if not frame_paths:
        return None, None, 0

    frames = []
    handedness_votes = []

    for frame_path in frame_paths:
        image = load_image(frame_path)
        result = detect_hands(image, detector)

        if not result.hand_landmarks:
            frames.append(np.full((21, 3), np.nan, dtype=np.float32))
            continue

        hand = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name
        points = landmarks_array_from_hand(hand, handedness, mirror_left=True)
        frames.append(points)
        handedness_votes.append(handedness)

    sequence = np.stack(frames).astype(np.float32)
    num_valid = int(np.isfinite(sequence).all(axis=(1, 2)).sum())

    if handedness_votes:
        # Majority vote for metadata only (mirroring already applied)
        values, counts = np.unique(handedness_votes, return_counts=True)
        handedness_label = str(values[np.argmax(counts)])
    else:
        handedness_label = None

    return sequence, handedness_label, num_valid

def prepare_dynamic_data(
    dataset_path=DATASET_PATH,
    model_path=MODEL_PATH,
    class_names=DYNAMIC_CLASSES,
    min_valid_frames=MIN_VALID_FRAMES,
):
    """
    Extract raw landmark sequences for all Experiment B classes.

    Args:
        dataset_path (str or Path): ASL_dynamic root.
        model_path (str or Path): hand_landmarker.task path.
        class_names (tuple): Labels to include.
        min_valid_frames (int): Minimum detections to keep a clip.

    Returns:
        dict: sequences, labels, clip_ids, handedness, stats.
    """
    dataset_path = Path(dataset_path)
    sequences = []
    labels = []
    clip_ids = []
    handedness_list = []
    skipped = []

    with initialize_hand_detector(
        model_path,
        mp.tasks.vision.RunningMode.IMAGE,
    ) as detector:

        for letter in class_names:
            letter_path = dataset_path / letter

            if not letter_path.is_dir():
                print(f"Missing class folder: {letter_path}")
                continue

            clip_paths = sorted(
                path for path in letter_path.iterdir() if path.is_dir()
            )
            kept = 0

            for clip_path in clip_paths:
                sequence, handedness, num_valid = extract_clip_sequence(
                    clip_path, detector
                )
                relative = str(clip_path.relative_to(dataset_path))

                if sequence is None or num_valid < min_valid_frames:
                    skipped.append(
                        {
                            "clip": relative,
                            "valid_frames": 0 if sequence is None else num_valid,
                        }
                    )
                    continue

                sequences.append(sequence)
                labels.append(letter)
                clip_ids.append(relative)
                handedness_list.append(handedness)
                kept += 1

            print(
                f"Processed {letter}: kept {kept}/{len(clip_paths)} clips"
            )

    return {
        "sequences": sequences,
        "labels": np.asarray(labels),
        "clip_ids": np.asarray(clip_ids),
        "handedness": np.asarray(handedness_list, dtype=object),
        "class_names": tuple(class_names),
        "skipped": skipped,
    }

def save_dynamic_sequences(feature_data, output_path=FEATURES_PATH):
    """
    Save raw sequences to a compressed NumPy archive.

    Variable-length sequences are stored as an object array.

    Args:
        feature_data (dict): Output of prepare_dynamic_data.
        output_path (str or Path): Destination .npz path.

    Returns:
        Path: Path to the saved archive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Object array preserves each clip's native temporal length
    sequence_array = np.empty(len(feature_data["sequences"]), dtype=object)

    for index, sequence in enumerate(feature_data["sequences"]):
        sequence_array[index] = sequence

    lengths = np.asarray(
        [sequence.shape[0] for sequence in feature_data["sequences"]],
        dtype=np.int32,
    )
    valid_counts = np.asarray(
        [
            int(np.isfinite(sequence).all(axis=(1, 2)).sum())
            for sequence in feature_data["sequences"]
        ],
        dtype=np.int32,
    )

    np.savez_compressed(
        output_path,
        sequences=sequence_array,
        labels=feature_data["labels"],
        clip_ids=feature_data["clip_ids"],
        handedness=feature_data["handedness"],
        class_names=np.asarray(feature_data["class_names"]),
        lengths=lengths,
        valid_counts=valid_counts,
    )

    return output_path

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract raw dynamic-letter landmark sequences."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to ASL_dynamic dataset root.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to hand_landmarker.task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURES_PATH,
        help="Where to save dynamic_raw_sequences.npz.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {args.model}\n"
            "Download hand_landmarker.task into models/ first."
        )

    feature_data = prepare_dynamic_data(args.dataset, args.model)
    output_path = save_dynamic_sequences(feature_data, args.output)

    print(
        f"\nSaved {len(feature_data['labels'])} clips to {output_path}\n"
        f"Classes: {feature_data['class_names']}\n"
        f"Skipped clips: {len(feature_data['skipped'])}"
    )

    if feature_data["skipped"]:
        print("Skipped:")
        for item in feature_data["skipped"]:
            print(
                f"  {item['clip']} "
                f"(valid frames={item['valid_frames']})"
            )
