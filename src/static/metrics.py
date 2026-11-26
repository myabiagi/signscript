"""Shared static-classifier scoring and confusion reporting."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data import LETTERS

def evaluate(model, X, y):
    """Calculate accuracy and macro-F1."""
    predictions = model.predict(X)

    return (
        accuracy_score(y, predictions),
        f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
    )

def print_confusions(y_true, predictions):
    """Print up to fifteen non-diagonal confusion counts."""
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(LETTERS),
    )

    mistakes = []

    for i, actual in enumerate(LETTERS):
        for j, predicted in enumerate(LETTERS):
            if actual != predicted and matrix[i, j]:
                mistakes.append(
                    (actual, predicted, int(matrix[i, j]))
                )

    mistakes.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    print("\nTop 15 confusions:")

    for actual, predicted, count in mistakes[:15]:
        print(f"{actual} -> {predicted}: {count}")

def print_validation(model, X_val, y_val):
    """Display validation metrics and common confusions."""
    predictions = model.predict(X_val)

    accuracy, macro_f1 = evaluate(
        model, X_val, y_val
    )

    print("\nValidation results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print(
        classification_report(
            y_val,
            predictions,
            labels=list(LETTERS),
            zero_division=0,
        )
    )

    print_confusions(y_val, predictions)