"""Add hand-geometry measurements to the existing 63 landmark features.

Input must be the output of normalize_landmarks(): wrist-centered,
left-hand-mirrored, and scale-normalized MediaPipe landmarks.
"""

import numpy as np

# MediaPipe landmark numbers: thumb 1-4, index 5-8, middle 9-12,
# ring 13-16, pinky 17-20; 0 is the wrist.
FINGER_JOINTS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

TIPS = (4, 8, 12, 16, 20)
KNUCKLES = (5, 9, 13, 17)

def _angles(points, a, b, c):
    """Compute joint angles (in radians) for batches of 21 landmarks."""
    u = points[:, a, :] - points[:, b, :]
    v = points[:, c, :] - points[:, b, :]

    cosine = np.sum(u * v, axis=1) / (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
        + 1e-8
    )

    return np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi

def _distances(points, a, b):
    """Calculate the distance between two landmarks."""
    return np.linalg.norm(
        points[:, a, :] - points[:, b, :],
        axis=1
    )


def enhance_features(X):
    """Return original 63 features plus 45 geometric features (108 total).

    Accepts a single (63,) vector or a batch (N, 63).
    The first 63 output values remain unchanged.
    """
    values = np.asarray(X, dtype=np.float32)

    single = values.ndim == 1

    if single:
        values = values.reshape(1, -1)

    if values.ndim != 2 or values.shape[1] != 63:
        raise ValueError(
            f"Expected 63 coordinates; got {values.shape}."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Landmark features contain NaN or infinity."
        )

    points = values.reshape(-1, 21, 3)
    extra = []

    # 1. Finger joint angles (15 features)
    for base, joint1, joint2, tip in FINGER_JOINTS:
        extra.append(_angles(points, 0, base, joint1))
        extra.append(_angles(points, base, joint1, joint2))
        extra.append(_angles(points, joint1, joint2, tip))

    # 2. Distances between fingertip pairs (10 features)
    for i, first in enumerate(TIPS):
        for second in TIPS[i + 1:]:
            extra.append(_distances(points, first, second))

    # 3. Thumb-tip distances to finger joints (8 features)
    for knuckle in KNUCKLES:
        extra.append(_distances(points, 4, knuckle))
        extra.append(_distances(points, 4, knuckle + 1))

    # 4. Relative vertical joint positions (8 features)
    for index in (5, 6, 7, 8, 9, 10, 11, 12):
        extra.append(
            points[:, index, 1] - points[:, 4, 1]
        )

    # 5. Distances between finger bases (4 features)
    for first, second in (
        (5, 9),
        (9, 13),
        (13, 17),
        (5, 17)
    ):
        extra.append(_distances(points, first, second))

    # 63 original + 45 geometric = 108
    result = np.column_stack(
        [values] + extra
    ).astype(np.float32)

    assert result.shape[1] == 108

    return result[0] if single else result
