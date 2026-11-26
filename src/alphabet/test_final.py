"""Final evaluation of SignScript's selected alphabet classifier.

Run once, after model selection is complete.
P9 and P10 must not have been used for training or tuning.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from train_alphabet import load_features, split_data

FEATURES_PATH = Path(
    "data/processed/alphabet_asl_hg_geometric_features.npz"
)

MODEL_PATH = Path(
    "models/alphabet_asl_hg_pugeault_geometric_rf.joblib"
)

RESULTS_DIR = Path("results/final")

SUMMARY_PATH = RESULTS_DIR / "final_test.json"
CONFUSION_PATH = RESULTS_DIR / "confusion_matrix.csv"

EXPECTED_CLASSES = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")

def main():
    # Avoid overwriting an existing final evaluation.
    if SUMMARY_PATH.exists() or CONFUSION_PATH.exists():
        raise FileExistsError(
            "Final test results already exist in results/final/. "
            "Do not repeatedly test and tune against P9/P10."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing features: {FEATURES_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    # Load the already-prepared 108-feature ASL-HG data.
    X, y, paths, participant_ids = load_features(
        FEATURES_PATH
    )

    # Use the original participant-independent split.
    train, validation, test = split_data(
        X, y, paths, participant_ids
    )

    X_test, y_test, _, ids_test = test

    # Confirm that the test data contains only P9/P10.
    if set(np.unique(ids_test)) != {"P9", "P10"}:
        raise ValueError(
            "Expected final test participants P9 and P10."
        )

    if X_test.shape[1] != 108:
        raise ValueError(
            f"Expected 108 features, got {X_test.shape[1]}."
        )

    if set(np.unique(y_test)) != set(EXPECTED_CLASSES):
        raise ValueError(
            "Unexpected alphabet classes in the test set."
        )

    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain invalid values.")

    model = joblib.load(MODEL_PATH)

    if model.n_features_in_ != 108:
        raise ValueError(
            "The model does not expect 108 features."
        )

    print(f"Model: {MODEL_PATH}")
    print(f"Test participants: P9, P10")
    print(f"Test images: {len(y_test)}")
    print(f"Training images (ASL-HG): {len(train[1])}")
    print(f"Validation images (ASL-HG): {len(validation[1])}")

    # Predict once on the untouched final test set.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    labels = list(EXPECTED_CLASSES)

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("\nFINAL TEST RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nPer-letter classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("\nPer-participant results:")

    participant_results = {}

    for participant in ("P9", "P10"):
        mask = ids_test == participant

        participant_accuracy = accuracy_score(
            y_test[mask],
            predictions[mask],
        )

        participant_f1 = f1_score(
            y_test[mask],
            predictions[mask],
            average="macro",
            zero_division=0,
        )

        participant_results[participant] = {
            "samples": int(mask.sum()),
            "accuracy": float(participant_accuracy),
            "macro_f1": float(participant_f1),
        }

        print(
            f"{participant}: "
            f"Accuracy={participant_accuracy:.4f}, "
            f"Macro-F1={participant_f1:.4f}"
        )

    # Calculate the most frequent incorrect predictions.
    mistakes = []

    for row_index, actual_letter in enumerate(labels):
        for col_index, predicted_letter in enumerate(labels):
            if actual_letter == predicted_letter:
                continue

            count = int(matrix[row_index, col_index])

            if count:
                mistakes.append(
                    (actual_letter, predicted_letter, count)
                )

    mistakes.sort(key=lambda item: item[2], reverse=True)

    print("\nTop 15 confusions:")

    for actual, predicted, count in mistakes[:15]:
        print(f"{actual} -> {predicted}: {count}")

    # Save results for the report and README.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": str(MODEL_PATH),
        "dataset": "ASL-HG",
        "test_participants": ["P9", "P10"],
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "participant_results": participant_results,
        "classification_report": report,
        "top_confusions": [
            {
                "actual": actual,
                "predicted": predicted,
                "count": count,
            }
            for actual, predicted, count in mistakes[:15]
        ],
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with CONFUSION_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual/Predicted", *labels])

        for letter, row in zip(labels, matrix):
            writer.writerow([letter, *row.tolist()])

    print(f"\nSaved summary: {SUMMARY_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_PATH}")
    print("\nFinal evaluation complete.")


if __name__ == "__main__":
    main()
