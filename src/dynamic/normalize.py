"""
Experiment B landmark normalization for dynamic ASL letters.

Three comparable representations of a resampled gesture sequence
with shape (T, 21, 3). Missing frames may contain NaNs.
"""

import numpy as np

from features import sample_frame_indices

WRIST_INDEX = 0

def _valid_frame_mask(sequence):
    """
    Mark frames that contain a finite hand pose.

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3).

    Returns:
        numpy.ndarray: Boolean mask of length T.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    return np.isfinite(sequence).all(axis=(1, 2))

def resample_sequence(sequence, num_samples):
    """
    Resample a variable-length sequence to a fixed number of frames.

    Only valid (finite) frames are used. If fewer valid frames than
    num_samples exist, indices repeat via linspace on the valid set.

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3), NaN = missing.
        num_samples (int): Output temporal length.

    Returns:
        numpy.ndarray or None: Shape (num_samples, 21, 3), or None if
        no valid frames exist.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    valid = np.where(_valid_frame_mask(sequence))[0]

    if len(valid) == 0:
        return None

    indices = sample_frame_indices(len(valid), num_samples)
    sampled = sequence[valid[indices]]
    return sampled.astype(np.float32)

def normalize_raw(sequence):
    """
    Keep camera-frame coordinates (after any left/right mirroring at extract).

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3).

    Returns:
        numpy.ndarray: Same shape as input.
    """
    return np.asarray(sequence, dtype=np.float32).copy()

def normalize_frame(sequence):
    """
    Normalize each frame by that frame's wrist and hand size.

    Removes per-frame translation and scale. Whole-hand camera motion
    across time is largely removed.

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3).

    Returns:
        numpy.ndarray: Shape (T, 21, 3). Invalid frames stay NaN.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    output = np.full_like(sequence, np.nan)

    for index, frame in enumerate(sequence):
        if not np.isfinite(frame).all():
            continue

        wrist = frame[WRIST_INDEX].copy()
        centered = frame - wrist
        scale = float(np.max(np.linalg.norm(centered, axis=1)))

        if scale > 0:
            centered = centered / scale

        output[index] = centered

    return output

def normalize_sequence(sequence):
    """
    Normalize every frame by the first valid frame's wrist and hand size.

    Preserves whole-hand motion relative to the start of the gesture while
    reducing initial position and scale differences across clips.

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3).

    Returns:
        numpy.ndarray: Shape (T, 21, 3). Invalid frames stay NaN.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    output = np.full_like(sequence, np.nan)
    valid = np.where(_valid_frame_mask(sequence))[0]

    if len(valid) == 0:
        return output

    reference = sequence[valid[0]]
    wrist0 = reference[WRIST_INDEX].copy()
    centered0 = reference - wrist0
    scale = float(np.max(np.linalg.norm(centered0, axis=1)))

    if scale <= 0:
        scale = 1.0

    for index, frame in enumerate(sequence):
        if not np.isfinite(frame).all():
            continue

        output[index] = (frame - wrist0) / scale

    return output

NORMALIZERS = {
    "raw": normalize_raw,
    "frame": normalize_frame,
    "sequence": normalize_sequence,
}

# Landmark indices used for explicit stroke features
INDEX_TIP = 8
PINKY_TIP = 20


def flatten_sequence(sequence):
    """
    Flatten (T, 21, 3) into a 1-D feature vector for Random Forest.

    NaNs (missing frames after resampling should be rare) become zeros.

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3).

    Returns:
        numpy.ndarray: Shape (T * 63,).
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    filled = np.nan_to_num(sequence, nan=0.0)
    return filled.reshape(-1)


def _path_stats(points_xy):
    """
    Path length, mean speed, max speed, and net displacement for an (T, 2) trail.

    Args:
        points_xy (numpy.ndarray): Shape (T, 2).

    Returns:
        tuple: (path_length, mean_speed, max_speed, net_displacement)
    """
    points_xy = np.asarray(points_xy, dtype=np.float32)

    if len(points_xy) < 2:
        return 0.0, 0.0, 0.0, 0.0

    deltas = np.diff(points_xy, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    path_length = float(speeds.sum())
    mean_speed = float(speeds.mean()) if len(speeds) else 0.0
    max_speed = float(speeds.max()) if len(speeds) else 0.0
    net = float(np.linalg.norm(points_xy[-1] - points_xy[0]))
    return path_length, mean_speed, max_speed, net


def motion_features(sequence):
    """
    Explicit trajectory features for J/Z disambiguation.

    Uses wrist, index tip, and pinky tip xy paths. Helps separate:
      - I (pinky up, little motion) vs J (pinky stroke)
      - D (index up, little motion) vs Z (index zigzag)

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3), preferably sequence-normalized.

    Returns:
        numpy.ndarray: Shape (MOTION_FEATURE_COUNT,) float32.
    """
    sequence = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0)

    wrist = sequence[:, WRIST_INDEX, :2]
    index = sequence[:, INDEX_TIP, :2]
    pinky = sequence[:, PINKY_TIP, :2]

    w_path, w_mean, w_max, w_net = _path_stats(wrist)
    i_path, i_mean, i_max, i_net = _path_stats(index)
    p_path, p_mean, p_max, p_net = _path_stats(pinky)

    # Which finger carried more of the stroke (J vs Z cue)
    finger_total = i_path + p_path + 1e-6
    pinky_share = p_path / finger_total
    index_share = i_path / finger_total

    # Rough zigzag cue: direction reversals along x for the index tip
    index_dx = np.diff(index[:, 0])
    reversals = 0.0
    if len(index_dx) >= 2:
        signs = np.sign(index_dx)
        signs = signs[signs != 0]
        if len(signs) >= 2:
            reversals = float(np.sum(signs[1:] * signs[:-1] < 0))

    return np.asarray(
        [
            w_path,
            w_mean,
            w_max,
            w_net,
            i_path,
            i_mean,
            i_max,
            i_net,
            p_path,
            p_mean,
            p_max,
            p_net,
            pinky_share,
            index_share,
            reversals,
        ],
        dtype=np.float32,
    )


# Keep count in sync with motion_features() output length
MOTION_FEATURE_COUNT = 15


def sequence_to_features(sequence, include_motion=True):
    """
    Flatten landmarks and optionally append motion descriptors.

    Args:
        sequence (numpy.ndarray): Shape (T, 21, 3).
        include_motion (bool): Append trajectory features.

    Returns:
        numpy.ndarray: 1-D feature vector.
    """
    flat = flatten_sequence(sequence)

    if not include_motion:
        return flat

    return np.concatenate([flat, motion_features(sequence)])


def build_feature_matrix(sequences, method, num_samples, include_motion=True):
    """
    Resample, normalize, and vectorize a list of raw clip sequences.

    Args:
        sequences (list): Variable-length (Ti, 21, 3) arrays.
        method (str): One of NORMALIZERS keys.
        num_samples (int): Fixed temporal length T.
        include_motion (bool): Append explicit motion features.

    Returns:
        tuple: (X, keep_mask) where X has shape
        (n_kept, T*63 [+ MOTION_FEATURE_COUNT]) and keep_mask marks
        which input sequences produced features.
    """
    if method not in NORMALIZERS:
        raise ValueError(f"Unknown normalization method: {method}")

    normalize = NORMALIZERS[method]
    rows = []
    keep = []
    feature_dim = num_samples * 63 + (
        MOTION_FEATURE_COUNT if include_motion else 0
    )

    for sequence in sequences:
        sampled = resample_sequence(sequence, num_samples)

        if sampled is None:
            keep.append(False)
            continue

        normalized = normalize(sampled)

        if not np.isfinite(normalized).any():
            keep.append(False)
            continue

        rows.append(sequence_to_features(normalized, include_motion))
        keep.append(True)

    if not rows:
        return np.zeros((0, feature_dim), dtype=np.float32), np.asarray(keep)

    return np.stack(rows).astype(np.float32), np.asarray(keep, dtype=bool)
