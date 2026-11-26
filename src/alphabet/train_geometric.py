"""Tune KNN and Random Forest with enhanced geometric landmark features."""

from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from geometric_features import enhance_features
from train_alphabet import (
    load_features,
    split_data,
    evaluate_validation
)
from train_knn import train_knn


SOURCE_PATH = Path(
    "data/processed/alphabet_asl_hg_features.npz"
)

ENHANCED_PATH = Path(
    "data/processed/alphabet_asl_hg_geometric_features.npz"
)

KNN_PATH = Path(
    "models/alphabet_asl_hg_knn_geometric_tuned.joblib"
)

RF_PATH = Path(
    "models/alphabet_asl_hg_random_forest_geometric_tuned.joblib"
)

BASELINE_KNN_PATH = Path(
    "models/alphabet_asl_hg_knn_tuned.joblib"
)

BASELINE_RF_PATH = Path(
    "models/alphabet_asl_hg_random_forest_tuned.joblib"
)

def score(model, X, y):
    """Return accuracy and macro-F1 for a model."""
    predictions = model.predict(X)

    return (
        accuracy_score(y, predictions),
        f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

def tune_random_forest(X_train, y_train, X_val, y_val):
    """Use the same 18-option search as the original RF experiment."""
    best_model = None
    best_accuracy = -1.0
    best_f1 = -1.0
    best_params = None

    for n_estimators in (100, 200):
        for max_depth in (None, 15, 30):
            for min_samples_leaf in (1, 3, 5):

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                accuracy, macro_f1 = score(
                    model,
                    X_val,
                    y_val
                )

                print(
                    f"trees={n_estimators}, "
                    f"depth={max_depth}, "
                    f"leaf={min_samples_leaf}: "
                    f"accuracy={accuracy:.4f}, "
                    f"macro-F1={macro_f1:.4f}",
                    flush=True
                )

                if (
                    macro_f1 > best_f1
                    or (
                        macro_f1 == best_f1
                        and accuracy > best_accuracy
                    )
                ):
                    best_model = model
                    best_accuracy = accuracy
                    best_f1 = macro_f1
                    best_params = (
                        n_estimators,
                        max_depth,
                        min_samples_leaf
                    )

    print(
        f"Best RF settings (trees, depth, leaf): "
        f"{best_params}"
    )

    return best_model

def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing prepared ASL-HG features: {SOURCE_PATH}"
        )

    # Load the existing normalized 63-feature dataset
    X, y, source_paths, participant_ids = load_features(
        SOURCE_PATH
    )

    # Generate 108 features without rerunning MediaPipe
    X_geo = enhance_features(X)

    print(
        f"Feature dimensions: "
        f"{X.shape[1]} -> {X_geo.shape[1]}"
    )

    # Save separately to preserve original features
    ENHANCED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        ENHANCED_PATH,
        X=X_geo,
        y=y,
        source_paths=source_paths,
        participant_ids=participant_ids
    )

    print(f"Saved enhanced data: {ENHANCED_PATH}")

    # Preserve the original signer-independent split
    train, val, test = split_data(
        X_geo,
        y,
        source_paths,
        participant_ids
    )

    X_train, y_train, _, _ = train
    X_val, y_val, _, _ = val

    print(
        f"P1-P7 training: {len(y_train)}; "
        f"P8 validation: {len(y_val)}"
    )

    print(
        f"P9-P10 reserved (not evaluated): "
        f"{len(test[1])}"
    )

    # Tune KNN
    print(
        "\nTuning geometric KNN (32 configurations):",
        flush=True
    )

    knn_model, _ = train_knn(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Tune Random Forest
    print(
        "\nTuning geometric Random Forest (18 configurations):",
        flush=True
    )

    rf_model = tune_random_forest(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Save both models
    KNN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(knn_model, KNN_PATH)
    joblib.dump(rf_model, RF_PATH)

    print(f"\nSaved KNN: {KNN_PATH}")
    print(f"Saved RF: {RF_PATH}")

    # Detailed geometric-model validation
    print("\nGeometric KNN validation:")
    evaluate_validation(knn_model, X_val, y_val)

    print("\nGeometric RF validation:")
    evaluate_validation(rf_model, X_val, y_val)

    # Compare geometric and original models on P8
    print("\nDirect comparison on P8:")

    models = {
        "Geometric KNN": (knn_model, X_val),
        "Geometric RF": (rf_model, X_val)
    }

    for name, path in (
        ("Original tuned KNN", BASELINE_KNN_PATH),
        ("Original tuned RF", BASELINE_RF_PATH)
    ):
        if path.exists():
            baseline = joblib.load(path)

            original_val = split_data(
                X,
                y,
                source_paths,
                participant_ids
            )[1][0]

            models[name] = (
                baseline,
                original_val
            )
        else:
            print(
                f"Baseline not found, skipping: {path}"
            )

    for name, (model, features) in models.items():
        accuracy, macro_f1 = score(
            model,
            features,
            y_val
        )

        print(
            f"{name}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro-F1={macro_f1:.4f}"
        )

    # Compare our troublesome letters
    print(
        "\nPer-letter comparison for S, M, N, T, R, P, V:"
    )

    for letter in (
        "S", "M", "N", "T", "R", "P", "V"
    ):
        mask = y_val == letter

        print(f"{letter}:", end=" ")

        parts = []

        for name, (model, features) in models.items():
            predictions = model.predict(
                features[mask]
            )

            correct = np.sum(
                predictions == letter
            )

            total = np.sum(mask)

            parts.append(
                f"{name}={correct}/{total}"
            )

        print("; ".join(parts))

    print(
        "\nP9 and P10 have not been used for evaluation."
    )

if __name__ == "__main__":
    main()
