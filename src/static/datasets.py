"""Locate static ASL training images and select dataset samples."""

import re
from pathlib import Path

import numpy as np

from data import LETTERS

ASL_HG_DIR = Path("data/static/asl_hg")
PUGEAULT_DIR = Path("data/static/pugeault_bowden")
PUGEAULT_PARTICIPANTS = ("A", "B", "C", "D", "E")
PUGEAULT_SAMPLES = 25

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