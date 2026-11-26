"""
Unified SignScript webcam: freeze-while-moving, commit-when-still.

Idea:
  - While the hand is moving, do NOT update the displayed letter.
  - Long stroke ending → J/Z dynamic model only (never static fallback,
    so a Z ending cannot be overwritten by static Q).
  - Small moves that settle → static alphabet model only.

For the J/Z/I/D lab demo only: python dev.py start-dynamic
"""

from pathlib import Path
import time

import cv2

from features import landmarks_array_from_hand, normalize_landmarks
from hand_tracking import (
    detect_hands,
    draw_landmarks,
    initialize_hand_detector,
)
from inference import load_static_model, predict_static_letter
from smoothing import PredictionSmoother
from dynamic.live_dynamic import (
    DEFAULT_METHOD,
    MAX_BUFFER_FRAMES,
    SPEED_EMA,
    finalize_capture,
    hand_speed,
    load_dynamic_bundle,
)

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
WINDOW_NAME = "SignScript"

LOW_CONFIDENCE = 0.45
HIGH_CONFIDENCE = 0.65

# Motion vs idle (normalized landmark units / frame, EMA-smoothed)
# MOVE must be clearly above IDLE so tiny jitter is not a "stroke".
MOVE_SPEED = 0.022
IDLE_SPEED = 0.012

# Commit as soon as speed drops into the idle band (1 frame ≈ instant)
STILL_COMMIT_FRAMES = 1

# Stroke = finger drawing relative to the wrist, not whole-hand translation.
WRIST_INDEX = 0
INDEX_TIP = 8
PINKY_TIP = 20
MIN_STROKE_FRAMES = 14
# Path of tip minus wrist (filters "I moved my arm")
MIN_RELATIVE_TIP_PATH = 0.28
# Tip must move more than the wrist (stroke vs sliding the hand)
MIN_TIP_OVER_WRIST_RATIO = 1.35
# Some turning in the relative path (hook / zigzag), not a straight shove
MIN_DIRECTION_CHANGES = 1
MIN_DYNAMIC_CONFIDENCE = 0.45

# Keep a committed J/Z on screen briefly so static cannot overwrite it
DYNAMIC_HOLD_FRAMES = 55


def confidence_color(confidence, kind="static"):
    """BGR color from confidence."""
    if confidence is None:
        return (180, 180, 180)

    if confidence >= HIGH_CONFIDENCE:
        return (0, 220, 255) if kind == "dynamic" else (0, 220, 0)
    if confidence >= LOW_CONFIDENCE:
        return (0, 165, 255)
    return (40, 40, 220)


def draw_letter_banner(frame, text, color, origin=(16, 16)):
    """Draw a large letter label inside a bordered box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.6
    thickness = 3
    pad_x, pad_y = 16, 12

    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x0, y0 = origin
    x1 = x0 + text_w + 2 * pad_x
    y1 = y0 + text_h + baseline + 2 * pad_y

    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 20), thickness=-1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=3)
    cv2.putText(
        frame,
        text,
        (x0 + pad_x, y0 + pad_y + text_h),
        font,
        scale,
        color,
        thickness,
    )
    return y1 + 8


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
        if prev_delta is not None:
            # 2D cross-ish: sign flip of consecutive steps (dot < 0)
            if float((prev_delta * delta).sum()) < 0:
                changes += 1
        prev_delta = delta
    return changes


def stroke_metrics(frames):
    """
    Measure finger drawing vs whole-hand translation.

    Returns:
        dict: relative tip paths, wrist path, ratios, direction changes.
    """
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
        w = frame[WRIST_INDEX, :2]
        wrist.append(w)
        index_rel.append(frame[INDEX_TIP, :2] - w)
        pinky_rel.append(frame[PINKY_TIP, :2] - w)

    wrist_path = _xy_path(wrist)
    index_path = _xy_path(index_rel)
    pinky_path = _xy_path(pinky_rel)
    best_rel = max(index_path, pinky_path)
    best_trail = index_rel if index_path >= pinky_path else pinky_rel
    turns = _direction_changes(best_trail)
    ratio = best_rel / (wrist_path + 1e-6)

    return {
        "index_rel": index_path,
        "pinky_rel": pinky_path,
        "best_rel": best_rel,
        "wrist": wrist_path,
        "ratio": ratio,
        "turns": turns,
        "n": len(frames),
    }


def looks_like_stroke(frames):
    """
    True if a fingertip drew a path relative to the wrist.

    Whole-hand sliding (all points translate together) has large camera-frame
    path but small wrist-relative tip path — that must NOT count as J/Z.
    """
    metrics = stroke_metrics(frames)
    return (
        metrics["n"] >= MIN_STROKE_FRAMES
        and metrics["best_rel"] >= MIN_RELATIVE_TIP_PATH
        and metrics["ratio"] >= MIN_TIP_OVER_WRIST_RATIO
        and metrics["turns"] >= MIN_DIRECTION_CHANGES
    )


def try_commit_dynamic(frames, dynamic_bundle):
    """
    Run the J/Z model on a motion buffer.

    Returns:
        tuple: (letter, confidence) or (None, None) if not accepted.
    """
    if not looks_like_stroke(frames):
        return None, None

    label, confidence, _ = finalize_capture(frames, dynamic_bundle)
    if (
        label in ("J", "Z")
        and confidence is not None
        and confidence >= MIN_DYNAMIC_CONFIDENCE
    ):
        return label, confidence
    return None, None


def run_webcam(camera_index=0, dynamic_method=DEFAULT_METHOD):
    """Freeze while moving; J/Z commits when the stroke ends (no hold)."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    smoother = PredictionSmoother()

    # committed display (only changes after stillness)
    shown_letter = "…"
    shown_confidence = 0.0
    shown_kind = "static"  # static | dynamic

    moving = False
    still_count = 0
    move_frames = []
    smooth_speed = 0.0
    previous_points = None
    dynamic_hold = 0

    try:
        static_model = load_static_model()
        dynamic_bundle = load_dynamic_bundle(
            dynamic_method, jz_only=True
        )

        print(
            f"Unified live | freeze-while-moving "
            f"(norm={dynamic_method})\n"
            "Letter updates when motion stops | q=quit"
        )

        with initialize_hand_detector(HAND_MODEL_PATH) as detector:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = detect_hands(frame, detector, timestamp_ms)

                handedness = "None"
                points = None
                instant_speed = 0.0
                status = ""

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    handedness = result.handedness[0][0].category_name

                    static_features = normalize_landmarks(
                        landmarks, handedness
                    )
                    raw_static, raw_conf = predict_static_letter(
                        static_model, static_features
                    )
                    smoother.update(raw_static, raw_conf)

                    points = landmarks_array_from_hand(
                        landmarks, handedness, mirror_left=True
                    )
                    instant_speed = hand_speed(previous_points, points)
                    previous_points = points.copy()
                else:
                    smoother.reset()
                    previous_points = None
                    move_frames = []
                    moving = False
                    still_count = 0

                smooth_speed = (
                    (1.0 - SPEED_EMA) * smooth_speed
                    + SPEED_EMA * instant_speed
                )

                if dynamic_hold > 0:
                    dynamic_hold -= 1

                # While a J/Z result is locked, ignore new commits (stops
                # the static model from immediately overwriting Z with Q).
                if dynamic_hold > 0:
                    status = "holding J/Z result"
                elif points is not None:
                    if smooth_speed >= MOVE_SPEED:
                        # Hand in motion: freeze display, buffer landmarks
                        if not moving:
                            moving = True
                            move_frames = []
                            still_count = 0
                        if len(move_frames) < MAX_BUFFER_FRAMES:
                            move_frames.append(points)
                        status = "moving — letter frozen"

                    else:
                        # Speed dropped below MOVE — stroke is ending.
                        # Commit J/Z immediately from the buffer (no hold).
                        if moving and looks_like_stroke(move_frames):
                            metrics = stroke_metrics(move_frames)
                            label, confidence = try_commit_dynamic(
                                move_frames, dynamic_bundle
                            )
                            if label is not None:
                                shown_letter = label
                                shown_confidence = confidence
                                shown_kind = "dynamic"
                                dynamic_hold = DYNAMIC_HOLD_FRAMES
                                print(
                                    f"Commit dynamic {label} "
                                    f"({confidence:.0%}, "
                                    f"rel={metrics['best_rel']:.2f}, "
                                    f"turns={metrics['turns']})"
                                )
                            else:
                                # Pipeline rule: after a stroke, never ask
                                # the static model (avoids Z ending → Q).
                                print(
                                    f"Stroke rejected by dynamic "
                                    f"(rel={metrics['best_rel']:.2f}) — "
                                    f"keeping {shown_letter}"
                                )
                            moving = False
                            move_frames = []
                            still_count = 0
                            status = ""

                        elif smooth_speed < IDLE_SPEED:
                            # Small motion / static letter change
                            if moving or still_count > 0:
                                still_count += 1
                            status = (
                                f"still {still_count}/{STILL_COMMIT_FRAMES}"
                                if still_count
                                else ""
                            )

                            if still_count >= STILL_COMMIT_FRAMES:
                                metrics = stroke_metrics(move_frames)
                                letter = smoother.stable_prediction
                                conf = smoother.stable_confidence
                                if letter != "Uncertain":
                                    shown_letter = letter
                                    shown_confidence = conf
                                    shown_kind = "static"
                                    print(
                                        f"Commit static {letter} "
                                        f"({conf:.0%}, "
                                        f"rel={metrics['best_rel']:.2f})"
                                    )
                                moving = False
                                move_frames = []
                                still_count = 0
                        else:
                            status = "settling…"
                            if moving and len(move_frames) < MAX_BUFFER_FRAMES:
                                move_frames.append(points)
                elif points is None:
                    shown_letter = "…"
                    shown_confidence = 0.0
                    shown_kind = "static"
                    status = "no hand"

                # Display: committed letter only (frozen during motion)
                if shown_letter in ("…", "No hand detected"):
                    display_text = shown_letter
                else:
                    display_text = (
                        f"{shown_letter} {shown_confidence:.0%}"
                    )

                frame = draw_landmarks(frame, result)
                color = confidence_color(shown_confidence, shown_kind)
                if moving:
                    color = (80, 80, 80)

                banner_bottom = draw_letter_banner(
                    frame,
                    f"Letter: {display_text}",
                    color,
                )
                cv2.putText(
                    frame,
                    f"Hand: {handedness}  {status}  "
                    f"spd={smooth_speed:.3f}",
                    (20, banner_bottom + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "J/Z locks when stroke ends — no hold needed | q quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
