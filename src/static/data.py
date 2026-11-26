"""Load prepared static-letter data and create signer-based splits."""

from pathlib import Path

import numpy as np

ASL_HG_PATH = Path("data/processed/static_asl_hg.npz")
PUGEAULT_PATH = Path("data/processed/static_pugeault.npz")
GRAUDT_PATH = Path("data/processed/static_graudt.npz")
WEBCAM_PATH = Path("data/processed/static_webcam.npz")

LETTERS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")

TRAIN_IDS = tuple(f"P{i}" for i in range(1, 8))
VAL_IDS = ("P8",)
TEST_IDS = ("P9", "P10")

def load_features(path):
    """Load extracted features and their metadata."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python dev.py prepare-static"
        )

    with np.load(path) as data:
        return (
            data["X"],
            data["y"],
            data["source_paths"],
            data["participant_ids"],
        )

def split_data(X, y, paths, participant_ids):
    """Split ASL-HG by participant to prevent signer leakage."""
    groups = (TRAIN_IDS, VAL_IDS, TEST_IDS)

    masks = [
        np.isin(participant_ids, group)
        for group in groups
    ]

    membership = sum(
        mask.astype(int) for mask in masks
    )

    if not np.all(membership == 1):
        raise ValueError(
            "Participant groups overlap or omit samples."
        )

    splits = tuple(
        (
            X[mask],
            y[mask],
            paths[mask],
            participant_ids[mask],
        )
        for mask in masks
    )

    for name, split in zip(
        ("Training", "Validation", "Testing"),
        splits,
    ):
        if set(np.unique(split[1])) != set(LETTERS):
            raise ValueError(
                f"{name} is missing alphabet classes."
            )

    return splits

def combine_pugeault_training(X_train, y_train, X_val):
    """Add validated Pugeault features without using P8 for fitting."""
    X_extra, y_extra, _, _ = load_features(PUGEAULT_PATH)

    if X_extra.shape[1] != 108:
        raise ValueError("Pugeault must contain 108 features.")

    if set(np.unique(y_extra)) != set(LETTERS):
        raise ValueError("Pugeault alphabet classes don't match.")

    if not np.isfinite(X_extra).all():
        raise ValueError("Pugeault features contain invalid values.")

    validation_vectors = {
        row.tobytes()
        for row in np.asarray(X_val, dtype=np.float32)
    }

    overlap = sum(
        row.tobytes() in validation_vectors
        for row in np.asarray(X_extra, dtype=np.float32)
    )

    print(f"Exact Pugeault/P8 overlap: {overlap}")

    if overlap:
        raise ValueError(
            "External training data overlaps validation."
        )

    X_train = np.concatenate(
        (X_train, X_extra),
        axis=0,
    )

    y_train = np.concatenate(
        (y_train, y_extra),
        axis=0,
    )

    return X_train, y_train


def _load_63_or_none(path):
    """Load a prepared 63-D feature cache if it exists."""
    path = Path(path)
    if not path.exists():
        return None

    X, y, paths, participant_ids = load_features(path)
    if X.shape[1] != 63:
        raise ValueError(
            f"{path} must contain 63-D landmarks, got {X.shape[1]}"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{path} contains invalid values.")
    return X, y, paths, participant_ids


def load_webcam_domain_sources():
    """
    Load optional Graudt + local webcam landmark caches (63-D).

    Returns:
        list[tuple]: Non-empty (X, y, paths, ids) bundles.
    """
    bundles = []
    for path, label in (
        (GRAUDT_PATH, "Graudt"),
        (WEBCAM_PATH, "webcam"),
    ):
        loaded = _load_63_or_none(path)
        if loaded is None:
            print(f"Skipping missing {label} cache: {path}")
            continue
        X, y, paths, ids = loaded
        print(f"Loaded {label}: {len(y)} samples from {path}")
        bundles.append((X, y, paths, ids))
    return bundles


def repeat_rows(X, y, times):
    """Oversample rows to up-weight webcam-domain examples."""
    times = int(times)
    if times <= 1:
        return X, y
    return (
        np.repeat(X, times, axis=0),
        np.repeat(y, times, axis=0),
    )