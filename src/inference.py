from pathlib import Path

import joblib
import numpy as np

from geometric_features import enhance_features

# Model currently being tested
ALPHABET_MODEL_PATH = Path("models/alphabet_rf.joblib")

def load_alphabet_model(model_path=ALPHABET_MODEL_PATH):
    """Load a trained alphabet classifier."""
    return joblib.load(model_path)

def predict_letter(model, features):
    """Predict an ASL letter using the features expected by the model."""

    # The webcam provides 63 normalized landmark coordinates.
    features = np.asarray(features, dtype=np.float32).reshape(1, -1)

    # Check how many features the selected model expects.
    expected_features = model.n_features_in_

    if expected_features == 108:
        # Geometric model: add 45 measurements to the 63 coordinates.
        if features.shape[1] == 63:
            features = enhance_features(features)

    elif expected_features == 63:
        # Original model: use the normalized landmarks unchanged.
        pass

    else:
        raise ValueError(
            f"Unsupported model feature count: {expected_features}"
        )

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but received {features.shape[1]}."
        )

    return model.predict(features)[0]
