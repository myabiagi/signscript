from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from train_knn import train_knn

FEATURES_PATH = Path("data/processed/alphabet_asl_hg_features.npz")
ALPHABET_MODEL_PATH = Path("models/alphabet_asl_hg_knn_tuned.joblib")

TRAIN_PARTICIPANTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
VAL_PARTICIPANTS = ["P8"]
TEST_PARTICIPANTS = ["P9", "P10"]

def load_features(features_path):
    """Load prepared alphabet features and their metadata."""
    with np.load(features_path) as data:
        X = data["X"]
        y = data["y"]
        source_paths = data["source_paths"]
        participant_ids = data["participant_ids"]

    return X, y, source_paths, participant_ids

def check_duplicates(X, y):
    """Check for duplicate feature vectors and conflicting labels."""
    unique_features, inverse, counts = np.unique(
        X,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    duplicate_count = len(X) - len(unique_features)
    conflicting_groups = 0

    for group in np.where(counts > 1)[0]:
        group_labels = y[inverse == group]

        if len(np.unique(group_labels)) > 1:
            conflicting_groups += 1

    print(f"Total samples: {len(X)}")
    print(f"Duplicate samples: {duplicate_count}")
    print(f"Duplicate groups with conflicting labels: {conflicting_groups}")

    return duplicate_count, conflicting_groups

def split_data(X, y, source_paths, participant_ids):
    """Split examples by participant rather than individual images."""
    train_mask = np.isin(participant_ids, TRAIN_PARTICIPANTS)
    val_mask = np.isin(participant_ids, VAL_PARTICIPANTS)
    test_mask = np.isin(participant_ids, TEST_PARTICIPANTS)

    membership = (
        train_mask.astype(int)
        + val_mask.astype(int)
        + test_mask.astype(int)
    )

    if not np.all(membership == 1):
        raise ValueError("Participant splits overlap or omit examples.")

    train = (
        X[train_mask],
        y[train_mask],
        source_paths[train_mask],
        participant_ids[train_mask]
    )

    val = (
        X[val_mask],
        y[val_mask],
        source_paths[val_mask],
        participant_ids[val_mask]
    )

    test = (
        X[test_mask],
        y[test_mask],
        source_paths[test_mask],
        participant_ids[test_mask]
    )

    all_letters = set(np.unique(y))

    for name, split in [
        ("Training", train),
        ("Validation", val),
        ("Testing", test)
    ]:
        if set(np.unique(split[1])) != all_letters:
            raise ValueError(f"{name} split is missing alphabet classes.")

    return train, val, test

def evaluate_validation(model, X_val, y_val):
    """Report overall scores and errors for individual letters."""
    predictions = model.predict(X_val)
    letters = sorted(np.unique(y_val))

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nValidation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nPer-letter classification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=letters,
            zero_division=0
        )
    )

    # Count actual -> predicted errors
    mistakes = Counter(
        (actual, predicted)
        for actual, predicted in zip(y_val, predictions)
        if actual != predicted
    )

    print("\nTop 15 most common letter confusions:")

    for (actual, predicted), count in mistakes.most_common(15):
        print(f"{actual} -> {predicted}: {count} images")

    # Show recognition accuracy separately for each true letter
    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=letters
    )

    print("\nAccuracy for each letter:")

    for index, letter in enumerate(letters):
        correct = matrix[index, index]
        total = matrix[index].sum()
        letter_accuracy = correct / total if total else 0

        print(
            f"{letter}: {letter_accuracy:.2%} "
            f"({correct}/{total})"
        )

    return predictions

def inspect_neighbors(
    model,
    X_val,
    y_val,
    paths_val,
    ids_val,
    y_train,
    paths_train,
    ids_train
):
    """Inspect the training neighbours of selected validation letters."""
    print("\nChecking nearest training examples:")

    for letter in ["G", "M", "N", "T", "X"]:
        indices = np.where(y_val == letter)[0]

        if len(indices) == 0:
            continue

        index = indices[0]
        features = X_val[index].reshape(1, -1)

        prediction = model.predict(features)[0]

        distances, neighbors = model.kneighbors(
            features,
            n_neighbors=model.n_neighbors
        )

        print(f"\nValidation letter: {letter}")
        print(f"Validation image: {paths_val[index]}")
        print(f"Validation participant: {ids_val[index]}")
        print(f"Predicted letter: {prediction}")

        neighbor_letters = y_train[neighbors[0]]
        votes = Counter(neighbor_letters)

        print("Neighbour letter counts:", dict(votes))

        print("Five closest training examples:")

        for distance, neighbor_index in zip(
            distances[0][:5],
            neighbors[0][:5]
        ):
            print(
                f"  Letter: {y_train[neighbor_index]}, "
                f"Participant: {ids_train[neighbor_index]}, "
                f"Distance: {distance:.6f}"
            )
            print(f"  Image: {paths_train[neighbor_index]}")

if __name__ == "__main__":
    # Load prepared features
    X, y, source_paths, participant_ids = load_features(FEATURES_PATH)

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Source path shape:", source_paths.shape)
    print("Participants:", np.unique(participant_ids))
    print("Letters:", np.unique(y))

    # Check duplicates
    print("\nDataset duplicate check:")
    check_duplicates(X, y)

    # Split by participant
    print("\nDataset split:")

    train, val, test = split_data(
        X,
        y,
        source_paths,
        participant_ids
    )

    X_train, y_train, paths_train, ids_train = train
    X_val, y_val, paths_val, ids_val = val
    X_test, y_test, paths_test, ids_test = test

    print(f"Training: {len(X_train)} samples ({TRAIN_PARTICIPANTS})")
    print(f"Validation: {len(X_val)} samples ({VAL_PARTICIPANTS})")
    print(f"Testing: {len(X_test)} samples ({TEST_PARTICIPANTS})")

    # Experiment with KNN hyperparameters
    print("\nTuning KNN:")

    knn_model, knn_results = train_knn(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\nBest KNN model:")
    print(knn_model)

    # Save separately from previous classifiers
    ALPHABET_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(knn_model, ALPHABET_MODEL_PATH)

    print(f"\nKNN classifier saved to {ALPHABET_MODEL_PATH}")

    # Evaluate performance on validation participant P8
    predictions = evaluate_validation(
        knn_model,
        X_val,
        y_val
    )

    # Inspect some validation examples
    print("\nChecking validation predictions:")

    for i in range(min(10, len(X_val))):
        print(f"\nImage: {paths_val[i]}")
        print(f"Participant: {ids_val[i]}")
        print(f"Actual: {y_val[i]}")
        print(f"Predicted: {predictions[i]}")

    # Inspect difficult letters
    inspect_neighbors(
        knn_model,
        X_val,
        y_val,
        paths_val,
        ids_val,
        y_train,
        paths_train,
        ids_train
    )

    print("\nP9 and P10 remain reserved for final testing.")
