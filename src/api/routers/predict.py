"""Health, config, and letter-prediction routes."""

import numpy as np
from fastapi import APIRouter, HTTPException

from api.dependencies import get_dynamic_bundle, get_static_model
from api.schemas import (
    PredictDynamicRequest,
    PredictRequest,
    PredictResponse,
)
from dynamic.motion import (
    accept_dynamic_label,
    looks_like_stroke,
    prepare_frames_for_dynamic,
    stroke_metrics,
)
from dynamic.thresholds import MIN_STROKE_FRAMES, as_web_config
from dynamic_inference import predict_from_frames
from features import enhance_features, normalize_points
from static_inference import STATIC_MODEL_PATH, _dominance_confidence

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    """Report whether model files can be loaded."""
    static_ok = STATIC_MODEL_PATH.exists()
    dynamic_ok = False
    try:
        get_dynamic_bundle()
        dynamic_ok = True
    except Exception:
        dynamic_ok = False
    return {
        "ok": static_ok,
        "static": static_ok,
        "dynamic": dynamic_ok,
        "model": str(STATIC_MODEL_PATH),
        "note": "Client sends landmarks only; no video upload.",
    }


@router.get("/config")
def config():
    """Motion thresholds shared with the browser."""
    return as_web_config()


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Classify one still handshape."""
    if body.handedness not in ("Left", "Right"):
        raise HTTPException(
            status_code=400,
            detail='handedness must be "Left" or "Right"',
        )

    points = np.asarray(body.landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise HTTPException(
            status_code=400,
            detail=f"Expected landmarks shape (21, 3), got {points.shape}",
        )

    try:
        features63 = normalize_points(points, body.handedness)
        enhanced = enhance_features(features63).reshape(1, -1)
        model = get_static_model()
        letter = str(model.predict(enhanced)[0])
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            confidence = _dominance_confidence(
                model.predict_proba(enhanced)[0]
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(
        letter=letter,
        confidence=float(confidence),
        handedness=body.handedness,
        kind="static",
    )


@router.post("/predict-dynamic", response_model=PredictResponse)
def predict_dynamic(body: PredictDynamicRequest):
    """Classify a J/Z stroke buffer."""
    if body.handedness not in ("Left", "Right"):
        raise HTTPException(
            status_code=400,
            detail='handedness must be "Left" or "Right"',
        )
    if len(body.frames) < MIN_STROKE_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_STROKE_FRAMES} frames for J/Z.",
        )

    try:
        frames = prepare_frames_for_dynamic(body.frames, body.handedness)
        if not looks_like_stroke(frames):
            metrics = stroke_metrics(frames)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Motion did not look like a J/Z stroke "
                    f"(n={metrics['n']}, rel={metrics['best_rel']:.2f}, "
                    f"ratio={metrics['ratio']:.2f}, turns={metrics['turns']})."
                ),
            )

        bundle = get_dynamic_bundle()
        label, confidence, _ = predict_from_frames(frames, bundle)
        if not accept_dynamic_label(label, confidence):
            raise HTTPException(
                status_code=400,
                detail="Dynamic model did not accept this stroke as J or Z.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(
        letter=str(label),
        confidence=float(confidence),
        handedness=body.handedness,
        kind="dynamic",
    )
