"""Static ASL hand landmark feature extraction."""

import numpy as np

FINGER_JOINTS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

TIPS = (4, 8, 12, 16, 20)
KNUCKLES = (5, 9, 13, 17)

def normalize_landmarks(hand_landmarks, handedness):
    """Normalize 21 landmarks into 63 coordinates."""
    if len(hand_landmarks) != 21:
        raise ValueError("Expected 21 hand landmarks.")

    if handedness not in ("Left", "Right"):
        raise ValueError(
            f"Unexpected handedness: {handedness}"
        )

    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32,
    )

    # Original wrist-centered normalization.
    wrist = points[0].copy()
    points = points - wrist

    # Original handedness normalization.
    if handedness == "Left":
        points[:, 0] = -points[:, 0]

    # Original hand-size normalization.
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    if max_distance > 0:
        points = points / max_distance

    return points.flatten()

def _angles(points, a, b, c):
    """Compute joint angles for batches of landmarks."""
    u = points[:, a, :] - points[:, b, :]
    v = points[:, c, :] - points[:, b, :]

    cosine = np.sum(u * v, axis=1) / (
        np.linalg.norm(u, axis=1)
        * np.linalg.norm(v, axis=1)
        + 1e-8
    )

    return np.arccos(
        np.clip(cosine, -1.0, 1.0)
    ) / np.pi

def _distances(points, a, b):
    """Calculate distances between pairs of landmarks."""
    return np.linalg.norm(
        points[:, a, :] - points[:, b, :],
        axis=1,
    )

def enhance_features(X):
    """Add 45 geometric features to 63 landmark coordinates.

    Accepts (63,) or (N, 63).
    Returns (108,) or (N, 108).

    Feature order matches the existing final classifier.
    """
    values = np.asarray(X, dtype=np.float32)
    single = values.ndim == 1

    if single:
        values = values.reshape(1, -1)

    if values.ndim != 2 or values.shape[1] != 63:
        raise ValueError(
            f"Expected 63 features, got {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Features contain NaN or infinity."
        )

    points = values.reshape(-1, 21, 3)
    extra = []

    # 15 finger-joint angles.
    for base, joint1, joint2, tip in FINGER_JOINTS:
        extra.append(_angles(points, 0, base, joint1))
        extra.append(_angles(points, base, joint1, joint2))
        extra.append(_angles(points, joint1, joint2, tip))

    # 10 fingertip-pair distances.
    for i, first in enumerate(TIPS):
        for second in TIPS[i + 1:]:
            extra.append(_distances(points, first, second))

    # 8 thumb-to-finger-joint distances.
    for knuckle in KNUCKLES:
        extra.append(_distances(points, 4, knuckle))
        extra.append(_distances(points, 4, knuckle + 1))

    # 8 relative vertical coordinates.
    for index in (5, 6, 7, 8, 9, 10, 11, 12):
        extra.append(
            points[:, index, 1] - points[:, 4, 1]
        )

    # 4 distances between finger bases.
    for first, second in (
        (5, 9),
        (9, 13),
        (13, 17),
        (5, 17),
    ):
        extra.append(_distances(points, first, second))

    result = np.column_stack(
        [values] + extra
    ).astype(np.float32)

    assert result.shape[1] == 108

    return result[0] if single else result
