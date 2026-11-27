"""Locate static ASL training images and select dataset samples."""

import re
from pathlib import Path

import numpy as np

from data import LETTERS

ASL_HG_DIR = Path("data/static/asl_hg")
PUGEAULT_DIR = Path("data/static/pugeault_bowden")
GRAUDT_DIR = Path("data/static/graudt_landmarks")
WEBCAM_STATIC_DIR = Path("data/static/webcam")
ASL_DYNAMIC_DIR = Path("data/ASL_dynamic")
PUGEAULT_PARTICIPANTS = ("A", "B", "C", "D", "E")
PUGEAULT_SAMPLES = 25
# Frames kept around the clip midpoint for static-like poses
ASL_DYNAMIC_MID_HALF_WINDOW = 5

def collect_asl_hg():
    """Get ASL-HG images and parse participant IDs from filenames."""
    images = sorted(
        path for path in ASL_HG_DIR.rglob("*.jpg")
        if path.parent.name.upper() in LETTERS
    )

    if not images:
        raise FileNotFoundError(
            f"No images found in {ASL_HG_DIR}"
        )

    samples = []

    for path in images:
        match = re.fullmatch(
            r"(P\d+)_([A-Z0-9])_(\d+)",
            path.stem,
        )

        if match is None:
            raise ValueError(
                f"Unexpected ASL-HG filename: {path.name}"
            )

        participant, letter, _ = match.groups()

        if letter != path.parent.name.upper():
            raise ValueError(
                f"Folder/filename label mismatch: {path}"
            )

        samples.append((path, letter, participant))

    return samples

def sample_pugeault_images(folder):
    """Sample 25 images evenly across the recorded sequence."""
    images = sorted(folder.glob("color_*.png"))

    if len(images) <= PUGEAULT_SAMPLES:
        return images

    indices = np.linspace(
        0,
        len(images) - 1,
        num=PUGEAULT_SAMPLES,
        dtype=int,
    )

    return [
        images[index]
        for index in np.unique(indices)
    ]

def collect_pugeault():
    """Get RGB images organized by participant and letter."""
    samples = []

    for participant in PUGEAULT_PARTICIPANTS:
        for letter in LETTERS:
            folder = (
                PUGEAULT_DIR
                / participant
                / letter.lower()
            )

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Missing folder: {folder}"
                )

            selected = sample_pugeault_images(folder)

            if not selected:
                raise ValueError(
                    f"No RGB images in {folder}"
                )

            for path in selected:
                samples.append((path, letter, participant))

    return samples


def _landmark_letter_dirs(root):
    """Yield (letter_dir, letter) for A–Y excluding J under root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing landmark dataset folder: {root}"
        )

    found = False
    for letter in LETTERS:
        letter_dir = root / letter
        if letter_dir.is_dir():
            found = True
            yield letter_dir, letter

    if not found:
        raise FileNotFoundError(
            f"No letter folders under {root}. "
            "Expected A/, B/, ... with .npy files."
        )


def collect_graudt():
    """
    List Graudt webcam MediaPipe landmark .npy files.

    Accepts either layout:
      data/static/graudt_landmarks/A/*.npy
      data/static/graudt_landmarks/landmarks/A/*.npy   (Kaggle zip default)
    J/Z folders are skipped (static alphabet only).
    """
    root = GRAUDT_DIR
    nested = GRAUDT_DIR / "landmarks"
    if nested.is_dir() and any(nested.glob("[A-Y]")):
        root = nested

    samples = []
    for letter_dir, letter in _landmark_letter_dirs(root):
        files = sorted(letter_dir.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files in {letter_dir}")
        for path in files:
            samples.append((path, letter, "GRAUDT"))
    return samples


def collect_webcam_static():
    """List locally captured webcam landmark .npy files (63-D vectors)."""
    samples = []
    for letter_dir, letter in _landmark_letter_dirs(
        WEBCAM_STATIC_DIR
    ):
        files = sorted(letter_dir.glob("*.npy"))
        for path in files:
            samples.append((path, letter, "WEBCAM"))
    if not samples:
        raise FileNotFoundError(
            f"No webcam samples in {WEBCAM_STATIC_DIR}. "
            "Run: python dev.py collect-static"
        )
    return samples


def _ensure_asl_dynamic_frames(avi_path):
    """
    Return the frames folder for one ASL_Dynamic clip, extracting if needed.

    Args:
        avi_path (Path): Path to LETTER_clipN.avi.

    Returns:
        Path: Directory of frame_*.jpg files.
    """
    import cv2

    avi_path = Path(avi_path)
    frames_dir = avi_path.with_name(f"{avi_path.stem}_frames")
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing:
        return frames_dir

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(avi_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {avi_path}")

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        out = frames_dir / f"frame_{index:04d}.jpg"
        if not cv2.imwrite(str(out), frame):
            capture.release()
            raise RuntimeError(f"Failed to write {out}")
        index += 1
    capture.release()

    if index == 0:
        raise RuntimeError(f"No frames decoded from {avi_path}")
    print(f"Extracted {index} frames → {frames_dir}")
    return frames_dir


def _middle_frame_paths(frames_dir, half_window=ASL_DYNAMIC_MID_HALF_WINDOW):
    """Pick JPGs around the midpoint of a clip's frame folder."""
    frames_dir = Path(frames_dir)
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        return []
    mid = len(files) // 2
    lo = max(0, mid - half_window)
    hi = min(len(files), mid + half_window + 1)
    return files[lo:hi]


def collect_asl_dynamic_mid():
    """
    Middle frames from ASL_Dynamic letter clips (static alphabet only).

    Skips J/Z (motion letters). Extracts missing *_frames from .avi when needed.
    Uses ~mid±5 frames per clip as held-pose-like samples.
    """
    root = ASL_DYNAMIC_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing {root}. Extract ASL_dynamic.zip from the Mendeley "
            "ASL Dynamic Gestures dataset."
        )

    samples = []
    for letter in LETTERS:
        letter_dir = root / letter
        if not letter_dir.is_dir():
            print(f"Skipping missing letter folder: {letter_dir}")
            continue

        avis = sorted(letter_dir.glob(f"{letter}_clip*.avi"))
        if not avis:
            # Fall back to any existing *_frames dirs
            frame_dirs = sorted(
                path
                for path in letter_dir.iterdir()
                if path.is_dir() and path.name.endswith("_frames")
            )
            for frames_dir in frame_dirs:
                for path in _middle_frame_paths(frames_dir):
                    samples.append(
                        (path, letter, f"ASLDYN_{frames_dir.name}")
                    )
            continue

        for avi in avis:
            frames_dir = _ensure_asl_dynamic_frames(avi)
            for path in _middle_frame_paths(frames_dir):
                samples.append(
                    (path, letter, f"ASLDYN_{avi.stem}")
                )

    if not samples:
        raise FileNotFoundError(
            f"No mid-frame samples found under {root}"
        )
    return samples