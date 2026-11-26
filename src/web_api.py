"""
SignScript web API: MediaPipe runs in the browser; only landmarks are POSTed.

Privacy: no video frames leave the client. Educational demo for COMP 4102.
"""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import enhance_features, normalize_points
from inference import (
    STATIC_MODEL_PATH,
    _dominance_confidence,
    load_static_model,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="SignScript",
    description="ASL static-letter demo (landmarks only).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_model = None


def get_static_model():
    global _static_model
    if _static_model is None:
        if not STATIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {STATIC_MODEL_PATH}. Train or install a classifier first."
            )
        _static_model = load_static_model()
    return _static_model


class PredictRequest(BaseModel):
    landmarks: list[list[float]] = Field(
        ...,
        description="21 MediaPipe landmarks as [x, y, z] in image coords.",
    )
    handedness: str = Field(
        ...,
        description='"Left" or "Right" from MediaPipe.',
    )


class PredictResponse(BaseModel):
    letter: str
    confidence: float
    handedness: str


@app.on_event("startup")
def preload_model():
    try:
        get_static_model()
        print(f"Loaded static classifier: {STATIC_MODEL_PATH}")
    except Exception as exc:
        print(f"Warning: static model not loaded yet ({exc})")


@app.get("/api/health")
def health():
    ready = STATIC_MODEL_PATH.exists()
    return {
        "ok": ready,
        "model": str(STATIC_MODEL_PATH),
        "note": "Client sends landmarks only; no video upload.",
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
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
    )


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html missing")
    return FileResponse(index_path)


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
