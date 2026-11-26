"""Train geometric Random Forest with ASL-HG and Pugeault images."""

from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from train_alphabet import (
    load_features,
    split_data,
    evaluate_validation
)

ASL_HG_PATH = Path(
    "data/processed/alphabet_asl_hg_geometric_features.npz"
)

PUGEAULT_PATH = Path(
    "data/processed/alphabet_pugeault_geometric_features.npz"
)

BASELINE_MODEL_PATH = Path(
    "models/alphabet_asl_hg_random_forest_geometric_tuned.joblib"
)

COMBINED_MODEL_PATH = Path(
    "models/alphabet_asl_hg_pugeault_geometric_rf.joblib"
)

DIFFICULT_LETTERS = (
    "M", "N", "P", "R", "S", "T", "E"
)

def calculate_scores(model, X, y):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1, predictions

def main():
    required_paths = (
        ASL_HG_PATH,
        PUGEAULT_PATH,
        BASELINE_MODEL_PATH
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    # Load existing ASL-HG geometric features.
    X_asl, y_asl, paths_asl, participants_asl = load_features(
        ASL_HG_PATH
    )

    train, val, test = split_data(
        X_asl,
        y_asl,
        paths_asl,
        participants_asl
    )

    X_train, y_train, _, _ = train
    X_val, y_val, _, _ = val

    # Load Pugeault geometric features.
    with np.load(PUGEAULT_PATH) as data:
        X_pugeault = data["X"]
        y_pugeault = data["y"]
        paths_pugeault = data["source_paths"]
        participants_pugeault = data["participant_ids"]

    # Verify that the feature representations match.
    if X_train.shape[1] != 108:
        raise ValueError(
            "ASL-HG features must have 108 dimensions."
        )

    if X_pugeault.shape[1] != 108:
        raise ValueError(
            "Pugeault features must have 108 dimensions."
        )

    if set(np.unique(y_pugeault)) != set(np.unique(y_train)):
        raise ValueError(
            "Pugeault and ASL-HG have different letter classes."
        )

    if not np.isfinite(X_pugeault).all():
        raise ValueError(
            "Pugeault features contain invalid values."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "ASL-HG features contain invalid values."
        )

    if len(paths_pugeault) != len(y_pugeault):
        raise ValueError("Pugeault paths and labels are misaligned.")

    if len(participants_pugeault) != len(y_pugeault):
        raise ValueError("Pugeault participants and labels are misaligned.")

    # Ensure no exact Pugeault feature vector exists in P8.
    val_vectors = {
        row.tobytes()
        for row in np.asarray(X_val, dtype=np.float32)
    }

    exact_overlap = sum(
        row.tobytes() in val_vectors
        for row in np.asarray(X_pugeault, dtype=np.float32)
    )

    print(f"Exact Pugeault/P8 feature overlap: {exact_overlap}")

    if exact_overlap:
        raise ValueError(
            "Exact validation overlap detected. "
            "Investigate before training."
        )

    # Add external examples only to the training set.
    X_combined = np.concatenate(
        (X_train, X_pugeault),
        axis=0
    )

    y_combined = np.concatenate(
        (y_train, y_pugeault),
        axis=0
    )

    print("\nTraining data:")
    print(f"ASL-HG P1-P7: {len(y_train)}")
    print(f"Pugeault: {len(y_pugeault)}")
    print(f"Combined: {len(y_combined)}")

    print("\nValidation data:")
    print(f"ASL-HG P8: {len(y_val)}")
    print(f"P9-P10 reserved: {len(test[1])}")

    # Same settings as the winning geometric RF.
    combined_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining combined geometric Random Forest...")
    combined_model.fit(
        X_combined,
        y_combined
    )

    COMBINED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        combined_model,
        COMBINED_MODEL_PATH
    )

    print(f"Saved combined model: {COMBINED_MODEL_PATH}")

    # Detailed validation report for the new model.
    print("\nCombined geometric RF validation:")
    evaluate_validation(
        combined_model,
        X_val,
        y_val
    )

    # Compare against the previous geometric baseline.
    baseline_model = joblib.load(
        BASELINE_MODEL_PATH
    )

    baseline_accuracy, baseline_f1, baseline_predictions = (
        calculate_scores(
            baseline_model,
            X_val,
            y_val
        )
    )

    combined_accuracy, combined_f1, combined_predictions = (
        calculate_scores(
            combined_model,
            X_val,
            y_val
        )
    )

    print("\nDirect comparison on P8:")

    print(
        f"Original geometric RF: "
        f"Accuracy={baseline_accuracy:.4f}, "
        f"Macro-F1={baseline_f1:.4f}"
    )

    print(
        f"Combined geometric RF: "
        f"Accuracy={combined_accuracy:.4f}, "
        f"Macro-F1={combined_f1:.4f}"
    )

    print("\nPer-letter comparison:")

    for letter in DIFFICULT_LETTERS:
        mask = y_val == letter
        total = int(np.sum(mask))

        baseline_correct = int(np.sum(
            baseline_predictions[mask] == letter
        ))

        combined_correct = int(np.sum(
            combined_predictions[mask] == letter
        ))

        print(
            f"{letter}: "
            f"Original={baseline_correct}/{total}, "
            f"Combined={combined_correct}/{total}"
        )

    print("\nP9 and P10 remain untouched.")

if __name__ == "__main__":
    main()
