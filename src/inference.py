"""Load models and predict static ASL letters."""

from pathlib import Path

import joblib
import numpy as np

from features import enhance_features

STATIC_MODEL_PATH = Path(
    "models/static_classifier.joblib"
)

def load_static_model(model_path=STATIC_MODEL_PATH):
    """Load the trained static-letter classifier."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Static classifier not found: {model_path}"
        )

    model = joblib.load(model_path)

    if model.n_features_in_ != 108:
        raise ValueError(
            "Static classifier must expect 108 features."
        )

    return model

def predict_static_letter(model, features):
    """Predict a letter from 63 normalized landmarks."""
    values = np.asarray(
        features, dtype=np.float32
    ).reshape(1, -1)

    if values.shape[1] != 63:
        raise ValueError(
            f"Expected 63 features, got {values.shape[1]}"
        )

    enhanced = enhance_features(values)
    return str(model.predict(enhanced)[0])
