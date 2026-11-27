"""Request and response bodies for the predict API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    landmarks: list[list[float]] = Field(
        ...,
        description="21 MediaPipe landmarks as [x, y, z].",
    )
    handedness: str = Field(..., description='"Left" or "Right".')


class PredictDynamicRequest(BaseModel):
    frames: list[list[list[float]]] = Field(
        ...,
        description="Sequence of 21x3 landmark frames in selfie space.",
    )
    handedness: str = Field(..., description='"Left" or "Right".')


class PredictResponse(BaseModel):
    letter: str
    confidence: float
    handedness: str
    kind: str = "static"
