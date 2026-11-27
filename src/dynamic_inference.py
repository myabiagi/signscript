"""Dynamic-letter inference: load J/Z sequence RF and classify a stroke."""

from pathlib import Path

import joblib
import numpy as np

from dynamic.motion import fingertip_speed
from dynamic.normalize import (
    NORMALIZERS,
    resample_sequence,
    sequence_to_features,
)
from dynamic.thresholds import MIN_BUFFER_FRAMES, TRIM_SPEED

MODELS_DIR = Path("models")
DEFAULT_METHOD = "sequence"
JZ_LABELS = ("J", "Z")


def model_path_for(method, jz_only=False):
    """Return the joblib path for one normalization method."""
    if jz_only:
        return MODELS_DIR / f"dynamic_rf_jz_{method}.joblib"
    return MODELS_DIR / f"dynamic_rf_{method}.joblib"


def load_dynamic_bundle(method, jz_only=False):
    """Load a J/Z (or 4-class) dynamic model bundle."""
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


def frame_speeds(frames):
    """Per-frame fingertip speed along a capture."""
    speeds = [0.0]
    for index in range(1, len(frames)):
        speeds.append(fingertip_speed(frames[index - 1], frames[index]))
    return speeds


def trim_idle_edges(frames, trim_speed=TRIM_SPEED):
    """Drop near-static frames from the start and end of a capture."""
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
    """Predict a label and confidence, optionally restricted to J/Z."""
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
    sampled = resample_sequence(sequence, int(bundle["num_samples"]))
    if sampled is None:
        return None, None, len(frames)

    normalized = NORMALIZERS[bundle["method"]](sampled)
    features = sequence_to_features(
        normalized, include_motion=bool(bundle.get("include_motion", True))
    ).reshape(1, -1)

    label, confidence = _predict_label_confidence(
        bundle["model"],
        features,
        mode=bundle.get("mode", "four_class"),
    )
    return label, confidence, len(frames)


def finalize_capture(capture_frames, bundle):
    """Run prediction and print a short result message."""
    n_raw = len(capture_frames)
    label, confidence, n_used = predict_from_frames(capture_frames, bundle)
    if label is None:
        message = f"Too short ({n_raw}/{MIN_BUFFER_FRAMES}) — try again"
        print(message)
        return None, None, message

    message = f"{label} ({confidence:.0%})  [{n_used} motion frames]"
    print(f"Captured {n_raw}→{n_used} frames → {label} ({confidence:.0%})")
    return label, confidence, message
