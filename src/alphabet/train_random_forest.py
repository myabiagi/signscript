from itertools import product
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from train_alphabet import (
    FEATURES_PATH,
    load_features,
    split_data,
    evaluate_validation
)

RF_MODEL_PATH = Path("models/alphabet_asl_hg_random_forest_tuned.joblib")
RF_BASELINE_PATH = Path("models/alphabet_asl_hg_random_forest.joblib")
KNN_MODEL_PATH = Path("models/alphabet_asl_hg_knn_tuned.joblib")

N_ESTIMATORS = [100, 200]
MAX_DEPTHS = [None, 15, 30]
MIN_SAMPLES_LEAF = [1, 3, 5]

def train_random_forest(X_train, y_train, X_val, y_val):
    """Tune Random Forest and select the best validation macro-F1."""
    best_model = None
    best_f1 = -1
    best_accuracy = -1
    results = []

    configurations = list(product(
        N_ESTIMATORS,
        MAX_DEPTHS,
        MIN_SAMPLES_LEAF
    ))

    for number, (n_estimators, max_depth, min_samples_leaf) in enumerate(
        configurations, start=1
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "accuracy": accuracy,
            "macro_f1": f1
        })

        print(
            f"[{number}/{len(configurations)}] "
            f"trees={n_estimators}, "
            f"max_depth={str(max_depth):4s}, "
            f"min_leaf={min_samples_leaf} | "
            f"Accuracy={accuracy:.4f}, "
            f"Macro-F1={f1:.4f}",
            flush=True
        )

        # Prefer macro-F1, then accuracy if macro-F1 is tied
        if (
            f1 > best_f1
            or (f1 == best_f1 and accuracy > best_accuracy)
        ):
            best_f1 = f1
            best_accuracy = accuracy
            best_model = model

    results.sort(
        key=lambda result: (
            result["macro_f1"],
            result["accuracy"]
        ),
        reverse=True
    )

    print("\nTop 5 Random Forest configurations:")

    for result in results[:5]:
        print(
            f"trees={result['n_estimators']}, "
            f"max_depth={result['max_depth']}, "
            f"min_leaf={result['min_samples_leaf']} | "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Macro-F1={result['macro_f1']:.4f}"
        )

    return best_model, results

def compare_models(models, X_val, y_val):
    """Compare saved classifiers using the same validation examples."""
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

    for letter in ["C", "E", "N", "O", "R", "T", "U"]:
        mask = y_val == letter
        total = np.sum(mask)

        if total == 0:
            continue

        print(f"\nLetter: {letter}")

        for name, predictions in predictions_by_model.items():
            correct = np.sum(predictions[mask] == letter)
            print(f"{name}: {correct}/{total} correct")

if __name__ == "__main__":
    # Load the prepared ASL-HG features
    X, y, source_paths, participant_ids = load_features(FEATURES_PATH)

    print("Feature shape:", X.shape)
    print("Letters:", np.unique(y))
    print("Participants:", np.unique(participant_ids))

    # Keep the same participant-independent split
    train, val, test = split_data(
        X,
        y,
        source_paths,
        participant_ids
    )

    X_train, y_train, paths_train, ids_train = train
    X_val, y_val, paths_val, ids_val = val
    X_test, y_test, paths_test, ids_test = test

    print("\nDataset split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Testing reserved: {len(X_test)} samples")

    # Tune Random Forest
    print("\nTuning Random Forest:")

    rf_model, rf_results = train_random_forest(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\nBest Random Forest model:")
    print(rf_model)

    # Save the tuned model separately
    RF_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(rf_model, RF_MODEL_PATH)

    print(f"\nTuned Random Forest saved to {RF_MODEL_PATH}")

    # Evaluate the selected model
    print("\nTuned Random Forest evaluation:")

    evaluate_validation(rf_model, X_val, y_val)

    # Compare against previously saved models
    models = {
        "Tuned Random Forest": rf_model
    }

    if RF_BASELINE_PATH.exists():
        models["Original Random Forest"] = joblib.load(
            RF_BASELINE_PATH
        )
    else:
        print("\nOriginal Random Forest model not found.")

    if KNN_MODEL_PATH.exists():
        models["Tuned KNN"] = joblib.load(
            KNN_MODEL_PATH
        )
    else:
        print("\nTuned KNN model not found.")

    compare_models(models, X_val, y_val)

    print("\nP9 and P10 remain reserved for final testing.")
