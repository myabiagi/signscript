"""Hand speed and J/Z stroke geometry helpers."""

import numpy as np

from dynamic.thresholds import (
    INDEX_TIP,
    MAX_BUFFER_FRAMES,
    MIN_DIRECTION_CHANGES,
    MIN_DYNAMIC_CONFIDENCE,
    MIN_RELATIVE_TIP_PATH,
    MIN_STROKE_FRAMES,
    MIN_TIP_OVER_WRIST_RATIO,
    PINKY_TIP,
    TIP_INDICES,
    WRIST_INDEX,
)


def hand_speed(previous, current):
    """Mean fingertip + wrist displacement between two frames."""
    if previous is None or current is None:
        return 0.0

    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    indices = list(TIP_INDICES) + [WRIST_INDEX]
    prev_pts = previous[indices, :2]
    curr_pts = current[indices, :2]
    return float(np.linalg.norm(curr_pts - prev_pts, axis=1).mean())


fingertip_speed = hand_speed


def _xy_path(points):
    """Sum of segment lengths for an (T, 2) trail."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for prev, curr in zip(points[:-1], points[1:]):
        total += float(((curr - prev) ** 2).sum() ** 0.5)
    return total


def _direction_changes(points):
    """Count roughly opposite turns along an (T, 2) trail."""
    if len(points) < 3:
        return 0

    changes = 0
    prev_delta = None
    for prev, curr in zip(points[:-1], points[1:]):
        delta = curr - prev
        if float((delta ** 2).sum()) < 1e-8:
            continue
        if prev_delta is not None and float((prev_delta * delta).sum()) < 0:
            changes += 1
        prev_delta = delta
    return changes


def stroke_metrics(frames):
    """Wrist-relative tip path stats for stroke gating."""
    empty = {
        "index_rel": 0.0,
        "pinky_rel": 0.0,
        "best_rel": 0.0,
        "wrist": 0.0,
        "ratio": 0.0,
        "turns": 0,
        "n": len(frames),
    }
    if len(frames) < 2:
        return empty

    wrist = []
    index_rel = []
    pinky_rel = []
    for frame in frames:
        frame = np.asarray(frame, dtype=np.float32)
        w = frame[WRIST_INDEX, :2]
        wrist.append(w)
        index_rel.append(frame[INDEX_TIP, :2] - w)
        pinky_rel.append(frame[PINKY_TIP, :2] - w)

    wrist_path = _xy_path(wrist)
    index_path = _xy_path(index_rel)
    pinky_path = _xy_path(pinky_rel)
    best_rel = max(index_path, pinky_path)
    best_trail = index_rel if index_path >= pinky_path else pinky_rel
    return {
        "index_rel": index_path,
        "pinky_rel": pinky_path,
        "best_rel": best_rel,
        "wrist": wrist_path,
        "ratio": best_rel / (wrist_path + 1e-6),
        "turns": _direction_changes(best_trail),
        "n": len(frames),
    }


def looks_like_stroke(frames):
    """True if a fingertip drew a path relative to the wrist."""
    metrics = stroke_metrics(frames)
    return (
        metrics["n"] >= MIN_STROKE_FRAMES
        and metrics["best_rel"] >= MIN_RELATIVE_TIP_PATH
        and metrics["ratio"] >= MIN_TIP_OVER_WRIST_RATIO
        and metrics["turns"] >= MIN_DIRECTION_CHANGES
    )


def prepare_frames_for_dynamic(raw_frames, handedness):
    """Mirror left-hand x so sequences match desktop live."""
    prepared = []
    for frame in raw_frames:
        points = np.asarray(frame, dtype=np.float32)
        if points.shape != (21, 3):
            raise ValueError(f"Expected each frame (21, 3), got {points.shape}")
        if handedness == "Left":
            points = points.copy()
            points[:, 0] = -points[:, 0]
        prepared.append(points)
    return prepared


def accept_dynamic_label(label, confidence):
    """True if the dynamic model output is an accepted J or Z."""
    return (
        label in ("J", "Z")
        and confidence is not None
        and float(confidence) >= MIN_DYNAMIC_CONFIDENCE
    )


__all__ = [
    "hand_speed",
    "fingertip_speed",
    "stroke_metrics",
    "looks_like_stroke",
    "prepare_frames_for_dynamic",
    "accept_dynamic_label",
    "MAX_BUFFER_FRAMES",
]
