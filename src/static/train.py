"""Coordinate static ASL experiments and final model training."""

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from features import enhance_features
from data import (
    ASL_HG_PATH,
    load_features,
    split_data,
    combine_pugeault_training,
)
from tuning import tune_knn, tune_random_forest
from metrics import print_validation

MODEL_DIR = Path("models")
EXPERIMENT_DIR = MODEL_DIR / "experiments"

def run_training(mode):
    """Train with the existing participant-independent split."""
    X, y, paths, participant_ids = load_features(ASL_HG_PATH)

    if X.shape[1] != 63:
        raise ValueError("Expected 63 original ASL-HG features.")

    if mode in ("geometric-knn", "geometric-rf", "final"):
        X = enhance_features(X)

    train, val, test = split_data(
        X, y, paths, participant_ids
    )

    X_train, y_train, _, _ = train
    X_val, y_val, _, _ = val

    print(f"Training mode: {mode}")
    print(f"Training examples: {len(y_train)}")
    print(f"Validation examples: {len(y_val)}")
    print(f"Reserved test examples: {len(test[1])}")

    if mode == "final":
        X_train, y_train = combine_pugeault_training(
            X_train, y_train, X_val
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        print(f"Combined training images: {len(y_train)}")
        model.fit(X_train, y_train)

        output = (
            MODEL_DIR / "static_classifier_retrained.joblib"
        )

    elif mode in ("knn", "geometric-knn"):
        model = tune_knn(
            X_train, y_train, X_val, y_val
        )

        output = EXPERIMENT_DIR / f"{mode}.joblib"

    elif mode in ("rf", "geometric-rf"):
        model = tune_random_forest(
            X_train, y_train, X_val, y_val
        )

        output = EXPERIMENT_DIR / f"{mode}.joblib"

    else:
        raise ValueError(f"Unknown training mode: {mode}")

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)

    print(f"\nSaved model: {output}")
    print(f"Selected model: {model}")

    print_validation(model, X_val, y_val)

    print("\nP9 and P10 remain excluded from training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "knn",
            "rf",
            "geometric-knn",
            "geometric-rf",
            "final",
        ),
    )

    args = parser.parse_args()
    run_training(args.mode)