"""Evaluate a static classifier on held-out ASL-HG signers P9 and P10."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from features import enhance_features
from data import (
    load_features,
    split_data,
    LETTERS,
    ASL_HG_PATH,
)
from metrics import print_confusions

DEFAULT_MODEL_PATH = Path(
    "models/static_classifier.joblib"
)

def evaluate_model(model_path):
    """Evaluate the selected model without modifying training data."""
    X, y, paths, participant_ids = load_features(
        ASL_HG_PATH
    )

    if X.shape[1] != 63:
        raise ValueError(
            "Expected 63 prepared ASL-HG features."
        )

    X = enhance_features(X)

    _, _, test = split_data(
        X, y, paths, participant_ids
    )

    X_test, y_test, _, test_ids = test

    if set(np.unique(test_ids)) != {"P9", "P10"}:
        raise ValueError(
            "Final test must contain only P9 and P10."
        )

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)

    if model.n_features_in_ != 108:
        raise ValueError(
            "Expected a 108-feature static classifier."
        )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print(f"Model: {model_path}")
    print(f"Test images: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nPer-letter classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(LETTERS),
            zero_division=0,
        )
    )

    print("\nPer-participant results:")

    for participant in ("P9", "P10"):
        mask = test_ids == participant

        participant_accuracy = accuracy_score(
            y_test[mask], predictions[mask]
        )

        participant_f1 = f1_score(
            y_test[mask],
            predictions[mask],
            average="macro",
            zero_division=0,
        )

        print(
            f"{participant}: "
            f"Accuracy={participant_accuracy:.4f}, "
            f"Macro-F1={participant_f1:.4f}"
        )

    print_confusions(y_test, predictions)

    print("\nThis command does not modify the archived final-test results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    args = parser.parse_args()
    evaluate_model(args.model)