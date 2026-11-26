"""
Live webcam demo for Experiment B dynamic letters (J / Z / I / D).

Motion gating:
  Start capturing when fingertip speed rises above a threshold.
  Finalize when speed stays near zero again — so a held J ending
  is not kept feeding I-like static frames into the classifier.

Controls:
  m — cycle normalization / model (raw → frame → sequence)
  c — clear buffer and last result
  q — quit
"""

from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from features import landmarks_array_from_hand
from hand_tracking import (
    detect_hands,
    draw_landmarks,
    initialize_hand_detector,
)
from dynamic.normalize import (
    NORMALIZERS,
    resample_sequence,
    sequence_to_features,
)

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
MODELS_DIR = Path("models/dynamic")
WINDOW_NAME = "SignScript Dynamic (J/Z)"

METHODS = ("raw", "frame", "sequence")
# sequence-norm keeps the stroke; pairs better with motion features
DEFAULT_METHOD = "sequence"
JZ_LABELS = ("J", "Z")

# Fingertip landmark indices (MediaPipe)
TIP_INDICES = (4, 8, 12, 16, 20)

MIN_BUFFER_FRAMES = 8
MAX_BUFFER_FRAMES = 90
LOW_CONFIDENCE = 0.45

# Motion gate thresholds (normalized image units per frame).
# Kept low so webcam signing at typical desk distance still triggers.
START_SPEED = 0.005
STOP_SPEED = 0.0025
START_HOLD_FRAMES = 1
STOP_HOLD_FRAMES = 8
# Ignore new gestures briefly after a prediction
COOLDOWN_FRAMES = 15
# Drop idle head/tail before classifying
TRIM_SPEED = 0.002
# EMA smoothing so landmark jitter does not block "settled"
SPEED_EMA = 0.35


def model_path_for(method, jz_only=False):
    """Path to the joblib bundle for one normalization method."""
    if jz_only:
        return MODELS_DIR / f"dynamic_rf_jz_{method}.joblib"
    return MODELS_DIR / f"dynamic_rf_{method}.joblib"


def load_dynamic_bundle(method, jz_only=False):
    """
    Load a saved Experiment B model bundle.

    Prefers a dedicated J/Z binary model when jz_only=True (used by the
    unified 26-letter webcam). Falls back to the 4-class model with
    J/Z probability renormalization if the binary file is missing.
    """
    if jz_only:
        jz_path = model_path_for(method, jz_only=True)
        if jz_path.exists():
            bundle = joblib.load(jz_path)
            if "model" not in bundle:
                raise ValueError(f"Unexpected dynamic model format in {jz_path}")
            bundle.setdefault("mode", "jz_binary")
            return bundle

    path = model_path_for(method, jz_only=False)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python dev.py train-dynamic"
        )

    bundle = joblib.load(path)

    if "model" not in bundle:
        raise ValueError(f"Unexpected dynamic model format in {path}")

    if jz_only:
        bundle = dict(bundle)
        bundle["mode"] = "jz_from_four"

    return bundle


def hand_speed(previous, current):
    """
    Motion score between two (21, 3) frames.

    Uses mean fingertip + wrist xy displacement so both path motion (J/Z)
    and smaller finger motion register.
    """
    if previous is None or current is None:
        return 0.0

    indices = list(TIP_INDICES) + [0]
    prev_pts = previous[indices, :2]
    curr_pts = current[indices, :2]
    return float(np.linalg.norm(curr_pts - prev_pts, axis=1).mean())


# Keep old name for trim helpers
fingertip_speed = hand_speed


def frame_speeds(frames):
    """Per-frame fingertip speed along a captured sequence (length N, first=0)."""
    speeds = [0.0]
    for index in range(1, len(frames)):
        speeds.append(fingertip_speed(frames[index - 1], frames[index]))
    return speeds


def trim_idle_edges(frames, trim_speed=TRIM_SPEED):
    """
    Remove leading/trailing near-static frames from a capture.

    Keeps the moving core of J/Z so a held end pose does not dominate.
    """
    if len(frames) < MIN_BUFFER_FRAMES:
        return frames

    speeds = frame_speeds(frames)
    start = 0
    while start < len(frames) - MIN_BUFFER_FRAMES and speeds[start] < trim_speed:
        start += 1

    end = len(frames) - 1
    while end > start + MIN_BUFFER_FRAMES and speeds[end] < trim_speed:
        end -= 1

    return frames[start : end + 1]


def _predict_label_confidence(model, features, mode="four_class"):
    """
    Predict a letter and confidence.

    For motion-gated unified live, mode jz_binary / jz_from_four restricts
    the decision to J vs Z so probability is not split with I/D.
    """
    if mode in ("jz_binary", "jz_from_four") and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [str(c) for c in model.classes_]
        jz_indices = [i for i, c in enumerate(classes) if c in JZ_LABELS]

        if len(jz_indices) >= 2:
            jz_probs = probabilities[jz_indices]
            total = float(jz_probs.sum())
            if total > 0:
                jz_probs = jz_probs / total
                best = int(np.argmax(jz_probs))
                return classes[jz_indices[best]], float(jz_probs[best])

    label = str(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
    return label, confidence


def predict_from_frames(frames, bundle):
    """Classify one captured gesture sequence."""
    frames = trim_idle_edges(frames)

    if len(frames) < MIN_BUFFER_FRAMES:
        return None, None, 0

    sequence = np.stack(frames).astype(np.float32)
    num_samples = int(bundle["num_samples"])
    method = bundle["method"]

    sampled = resample_sequence(sequence, num_samples)
    if sampled is None:
        return None, None, len(frames)

    include_motion = bool(bundle.get("include_motion", True))
    normalized = NORMALIZERS[method](sampled)
    features = sequence_to_features(
        normalized, include_motion=include_motion
    ).reshape(1, -1)

    label, confidence = _predict_label_confidence(
        bundle["model"],
        features,
        mode=bundle.get("mode", "four_class"),
    )

    return label, confidence, len(frames)


def finalize_capture(capture_frames, bundle):
    """Run prediction and build a user-facing result message."""
    n_raw = len(capture_frames)
    label, confidence, n_used = predict_from_frames(capture_frames, bundle)

    if label is None:
        message = f"Too short ({n_raw}/{MIN_BUFFER_FRAMES}) — try again"
        print(message)
        return None, None, message

    message = f"{label} ({confidence:.0%})  [{n_used} motion frames]"
    print(f"Captured {n_raw}→{n_used} frames → {label} ({confidence:.0%})")
    return label, confidence, message


def run_webcam(camera_index=0, method=DEFAULT_METHOD):
    """Run live dynamic-letter demo with motion gating."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    method_index = METHODS.index(method)
    # Lab demo keeps the 4-class J/Z/I/D model for Experiment B.
    bundle = load_dynamic_bundle(METHODS[method_index], jz_only=False)

    recording = False
    capture_frames = []
    previous_points = None
    motion_high_count = 0
    motion_low_count = 0
    cooldown = 0
    instant_speed = 0.0
    smooth_speed = 0.0

    last_label = None
    last_confidence = None
    last_message = "Show hand, then sign J or Z"

    print(
        f"Dynamic live demo | method={METHODS[method_index]} | "
        f"T={bundle['num_samples']} | motion-gate ON\n"
        f"Start spd>={START_SPEED} | stop spd<{STOP_SPEED} | "
        "m=norm | c=clear | q=quit"
    )

    try:
        with initialize_hand_detector(HAND_MODEL_PATH) as detector:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = detect_hands(frame, detector, timestamp_ms)

                handedness = "None"
                points = None

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    handedness = result.handedness[0][0].category_name
                    points = landmarks_array_from_hand(
                        hand, handedness, mirror_left=True
                    )
                    instant_speed = hand_speed(previous_points, points)
                    previous_points = points.copy()
                else:
                    instant_speed = 0.0
                    previous_points = None

                smooth_speed = (
                    (1.0 - SPEED_EMA) * smooth_speed
                    + SPEED_EMA * instant_speed
                )

                if cooldown > 0:
                    cooldown -= 1

                # --- motion gate / recording update ---
                if recording:
                    if points is not None and len(capture_frames) < MAX_BUFFER_FRAMES:
                        capture_frames.append(points)

                    # No hand counts as settled (gesture ended / left frame)
                    speed_for_stop = (
                        smooth_speed if points is not None else 0.0
                    )
                    if speed_for_stop < STOP_SPEED:
                        motion_low_count += 1
                    else:
                        motion_low_count = 0

                    should_stop = (
                        len(capture_frames) >= MIN_BUFFER_FRAMES
                        and motion_low_count >= STOP_HOLD_FRAMES
                    ) or len(capture_frames) >= MAX_BUFFER_FRAMES

                    if should_stop:
                        recording = False
                        last_label, last_confidence, last_message = (
                            finalize_capture(capture_frames, bundle)
                        )
                        capture_frames = []
                        motion_high_count = 0
                        motion_low_count = 0
                        cooldown = COOLDOWN_FRAMES
                        smooth_speed = 0.0

                elif points is not None and cooldown == 0:
                    if smooth_speed >= START_SPEED:
                        motion_high_count += 1
                    else:
                        motion_high_count = 0

                    if motion_high_count >= START_HOLD_FRAMES:
                        recording = True
                        capture_frames = [points]
                        motion_high_count = 0
                        motion_low_count = 0
                        last_label = None
                        last_confidence = None
                        last_message = "Recording…"
                        print(
                            f"Motion start → recording "
                            f"(spd={smooth_speed:.4f})"
                        )

                # --- HUD ---
                if recording:
                    status = (
                        f"RECORDING {len(capture_frames)}fr "
                        f"idle={motion_low_count}/{STOP_HOLD_FRAMES}"
                    )
                    status_color = (0, 0, 255)
                elif last_label is not None:
                    conf_text = (
                        f"{last_confidence:.0%}"
                        if last_confidence is not None
                        else "?"
                    )
                    low = (
                        last_confidence is not None
                        and last_confidence < LOW_CONFIDENCE
                    )
                    tag = " low conf" if low else ""
                    status = f"Result: {last_label} ({conf_text}){tag}"
                    status_color = (0, 165, 255) if low else (0, 255, 0)
                elif points is None:
                    status = "No hand — put hand in frame"
                    status_color = (0, 255, 255)
                elif cooldown > 0:
                    status = f"Cooldown {cooldown} — then sign again"
                    status_color = (200, 200, 0)
                else:
                    status = (
                        f"Waiting for motion "
                        f"(spd {smooth_speed:.3f} / need {START_SPEED:.3f})"
                    )
                    status_color = (0, 255, 0)

                frame = draw_landmarks(frame, result)
                method_name = METHODS[method_index]

                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    status_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Norm: {method_name}  Hand: {handedness}  "
                    f"spd={smooth_speed:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "sign J/Z then hold still | m norm | c clear | q quit",
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

                if key == ord("c"):
                    recording = False
                    capture_frames = []
                    motion_high_count = 0
                    motion_low_count = 0
                    cooldown = 0
                    last_label = None
                    last_confidence = None
                    last_message = "Cleared — sign when ready"
                    print("Cleared")

                if key == ord("m"):
                    method_index = (method_index + 1) % len(METHODS)
                    bundle = load_dynamic_bundle(METHODS[method_index])
                    recording = False
                    capture_frames = []
                    last_label = None
                    last_confidence = None
                    last_message = (
                        f"Switched to {METHODS[method_index]}"
                    )
                    print(f"Switched to method={METHODS[method_index]}")

                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live Experiment B dynamic-letter demo."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index.",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help="Normalization / model to start with.",
    )
    args = parser.parse_args()

    run_webcam(camera_index=args.camera, method=args.method)
