from pathlib import Path

import joblib
import numpy as np

ALPHABET_MODEL_PATH = Path("models/alphabet_asl_hg_knn_tuned.joblib")

def load_alphabet_model(model_path=ALPHABET_MODEL_PATH):
    """Load the trained alphabet classifier."""
    return joblib.load(model_path)

def predict_letter(model, features):
    """Predict an ASL letter from normalized hand landmarks."""
    features = np.asarray(features, dtype=np.float32).reshape(1, -1)

    return model.predict(features)[0]
