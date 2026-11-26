from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from train_alphabet import (
    load_features,
    split_data,
    evaluate_validation
)

ASL_HG_FEATURES_PATH = Path(
    "data/processed/alphabet_asl_hg_features.npz"
)

SIGNALPHASET_FEATURES_PATH = Path(
    "data/processed/alphabet_features.npz"
)

KNN_BASELINE_PATH = Path(
    "models/alphabet_asl_hg_knn_tuned.joblib"
)

RF_BASELINE_PATH = Path(
    "models/alphabet_asl_hg_random_forest_tuned.joblib"
)

KNN_COMBINED_PATH = Path(
    "models/alphabet_combined_knn.joblib"
)

RF_COMBINED_PATH = Path(
    "models/alphabet_combined_random_forest.joblib"
)

def load_combined_data():
    """Load both datasets while keeping ASL-HG validation separate."""
    X_asl, y_asl, paths_asl, participant_ids = load_features(
        ASL_HG_FEATURES_PATH
    )

    train, val, test = split_data(
        X_asl,
        y_asl,
        paths_asl,
        participant_ids
    )

    X_train, y_train, _, _ = train
    X_val, y_val, _, _ = val

    # SignAlphaSet has no participant IDs
    with np.load(SIGNALPHASET_FEATURES_PATH) as data:
        X_signal = data["X"]
        y_signal = data["y"]

    # Both datasets must use the same feature representation
    if X_train.shape[1] != X_signal.shape[1]:
        raise ValueError(
            "The datasets have different feature dimensions."
        )

    if X_train.shape[1] != 63:
        raise ValueError("Expected 63 landmark features.")

    if set(np.unique(y_train)) != set(np.unique(y_signal)):
        raise ValueError(
            "The datasets do not contain the same alphabet classes."
        )

    if not np.isfinite(X_signal).all():
        raise ValueError("SignAlphaSet contains invalid feature values.")

    # Check for exact overlap with held-out validation features
    # Use the same dtype so identical vectors compare consistently
    validation_vectors = {
        row.tobytes()
        for row in np.asarray(X_val, dtype=np.float32)
    }

    overlap = sum(
        row.tobytes() in validation_vectors
        for row in np.asarray(X_signal, dtype=np.float32)
    )

    print(f"Exact SignAlphaSet/P8 feature overlap: {overlap}")

    if overlap > 0:
        raise ValueError(
            "SignAlphaSet overlaps validation data. "
            "Investigate before combining datasets."
        )

    X_combined = np.concatenate(
        [X_train, X_signal],
        axis=0
    )

    y_combined = np.concatenate(
        [y_train, y_signal],
        axis=0
    )

    print("\nTraining data:")
    print(f"ASL-HG training: {len(X_train)}")
    print(f"SignAlphaSet: {len(X_signal)}")
    print(f"Combined training: {len(X_combined)}")
    print(f"Validation (P8): {len(X_val)}")
    print(f"Final testing reserved: {len(test[0])}")

    return X_combined, y_combined, X_val, y_val

def train_combined_models(X_train, y_train):
    """Train KNN and Random Forest with fixed hyperparameters."""
    print("\nTraining combined KNN:")

    knn_model = KNeighborsClassifier(
        n_neighbors=31,
        weights="distance",
        metric="euclidean"
    )

    knn_model.fit(X_train, y_train)

    print(knn_model)

    print("\nTraining combined Random Forest:")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    print(rf_model)

    return knn_model, rf_model

def compare_models(models, X_val, y_val):
    """Compare saved and combined models on the same P8 data."""
    print("\nModel comparison on participant P8:")

    predictions_by_model = {}

    for name, model in models.items():
        predictions = model.predict(X_val)
        predictions_by_model[name] = predictions

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        print(
            f"{name}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro-F1={macro_f1:.4f}"
        )

    print("\nComparison for difficult letters:")

    for letter in ["E", "N", "R", "T", "O", "G", "V"]:
        mask = y_val == letter
        total = np.sum(mask)

        if total == 0:
            continue

        print(f"\nLetter: {letter}")

        for name, predictions in predictions_by_model.items():
            correct = np.sum(predictions[mask] == letter)
            print(f"{name}: {correct}/{total} correct")

if __name__ == "__main__":
    required_files = [
        ASL_HG_FEATURES_PATH,
        SIGNALPHASET_FEATURES_PATH,
        KNN_BASELINE_PATH,
        RF_BASELINE_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    # Prepare combined training data
    X_train, y_train, X_val, y_val = load_combined_data()

    # Train both classifiers using the combined dataset
    knn_model, rf_model = train_combined_models(
        X_train,
        y_train
    )

    # Save both models separately from the existing baselines
    KNN_COMBINED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(knn_model, KNN_COMBINED_PATH)
    joblib.dump(rf_model, RF_COMBINED_PATH)

    print(f"\nCombined KNN saved to {KNN_COMBINED_PATH}")
    print(f"Combined Random Forest saved to {RF_COMBINED_PATH}")

    # Full evaluation of combined models
    print("\nCombined KNN validation results:")
    evaluate_validation(knn_model, X_val, y_val)

    print("\nCombined Random Forest validation results:")
    evaluate_validation(rf_model, X_val, y_val)

    # Compare against models trained on ASL-HG alone
    models = {
        "Original tuned KNN": joblib.load(KNN_BASELINE_PATH),
        "Combined KNN": knn_model,
        "Original tuned RF": joblib.load(RF_BASELINE_PATH),
        "Combined RF": rf_model
    }

    compare_models(models, X_val, y_val)

    print("\nP9 and P10 remain reserved for final testing.")
