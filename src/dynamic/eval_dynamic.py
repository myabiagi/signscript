"""
Stratified k-fold evaluation for Experiment B dynamic letters.

Uses the same folds for raw / frame / sequence so the three
normalizations are compared fairly. Reports mean ± std of accuracy,
macro-F1, and false-dynamic rate.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dynamic.normalize import build_feature_matrix
from dynamic.train_dynamic import (
    FEATURES_PATH,
    METHODS,
    NUM_SAMPLES,
    RANDOM_STATE,
    RESULTS_DIR,
    evaluate_model,
    load_dynamic_sequences,
    train_random_forest,
)

N_SPLITS = 5


def run_kfold_eval(
    feature_path=FEATURES_PATH,
    results_dir=RESULTS_DIR,
    num_samples=NUM_SAMPLES,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Stratified k-fold CV for each normalization method.

    Args:
        feature_path (Path): dynamic_raw_sequences.npz path.
        results_dir (Path): Where to write CSV summaries.
        num_samples (int): Temporal length T.
        n_splits (int): Number of stratified folds.
        random_state (int): Fold RNG seed (shared across methods).

    Returns:
        dict: method -> list of per-fold metric dicts.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_dynamic_sequences(feature_path)
    sequences = data["sequences"]
    labels = data["labels"]
    class_names = data["class_names"]

    # Build features once per method; folds index into the same clip order
    features_by_method = {}
    for method in METHODS:
        X, keep = build_feature_matrix(sequences, method, num_samples)
        if not np.all(keep):
            raise RuntimeError(
                f"Method '{method}' dropped clips; k-fold needs all clips."
            )
        features_by_method[method] = X

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    print(
        f"Loaded {len(labels)} clips | T={num_samples} | "
        f"{n_splits}-fold stratified CV (seed={random_state})\n"
        f"Classes: {list(class_names)}\n"
    )

    fold_rows = []
    summary = {method: [] for method in METHODS}

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        print(
            f"--- Fold {fold}/{n_splits} "
            f"(train={len(train_idx)} test={len(test_idx)}) ---"
        )

        for method in METHODS:
            X = features_by_method[method]
            model = train_random_forest(X[train_idx], y_train)
            metrics = evaluate_model(
                model, X[test_idx], y_test, class_names
            )

            row = {
                "fold": fold,
                "method": method,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "false_dynamic_rate": metrics["false_dynamic_rate"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            fold_rows.append(row)
            summary[method].append(row)

            print(
                f"  {method:8s}  "
                f"acc={metrics['accuracy']:.3f}  "
                f"macroF1={metrics['macro_f1']:.3f}  "
                f"falseDyn={metrics['false_dynamic_rate']:.3f}"
            )
        print()

    folds_path = results_dir / "kfold_folds.csv"
    _write_folds_csv(fold_rows, folds_path)

    summary_path = results_dir / "kfold_summary.csv"
    summary_rows = _summarize(summary)
    _write_summary_csv(summary_rows, summary_path)

    print("Mean ± std across folds:")
    print(
        f"{'method':10s} {'acc':>14s} {'macroF1':>14s} {'falseDyn':>14s}"
    )
    for row in summary_rows:
        print(
            f"{row['method']:10s} "
            f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}    "
            f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}    "
            f"{row['false_dynamic_mean']:.3f}±{row['false_dynamic_std']:.3f}"
        )

    best = max(summary_rows, key=lambda r: r["macro_f1_mean"])
    print(
        f"\nBest mean macro-F1: {best['method']} "
        f"({best['macro_f1_mean']:.3f}±{best['macro_f1_std']:.3f})"
    )
    print(f"Saved per-fold → {folds_path}")
    print(f"Saved summary  → {summary_path}")

    return summary


def _summarize(summary):
    """Aggregate per-method fold lists into mean/std rows."""
    rows = []
    for method, folds in summary.items():
        acc = np.array([f["accuracy"] for f in folds], dtype=np.float64)
        f1 = np.array([f["macro_f1"] for f in folds], dtype=np.float64)
        fd = np.array(
            [f["false_dynamic_rate"] for f in folds], dtype=np.float64
        )
        rows.append(
            {
                "method": method,
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "false_dynamic_mean": float(fd.mean()),
                "false_dynamic_std": float(fd.std(ddof=1)) if len(fd) > 1 else 0.0,
                "n_folds": len(folds),
            }
        )
    return rows


def _write_folds_csv(rows, path):
    path = Path(path)
    headers = [
        "fold",
        "method",
        "accuracy",
        "macro_f1",
        "false_dynamic_rate",
        "n_train",
        "n_test",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    [
                        str(row["fold"]),
                        row["method"],
                        f"{row['accuracy']:.4f}",
                        f"{row['macro_f1']:.4f}",
                        f"{row['false_dynamic_rate']:.4f}",
                        str(row["n_train"]),
                        str(row["n_test"]),
                    ]
                )
                + "\n"
            )


def _write_summary_csv(rows, path):
    path = Path(path)
    headers = [
        "method",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "false_dynamic_mean",
        "false_dynamic_std",
        "n_folds",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    [
                        row["method"],
                        f"{row['accuracy_mean']:.4f}",
                        f"{row['accuracy_std']:.4f}",
                        f"{row['macro_f1_mean']:.4f}",
                        f"{row['macro_f1_std']:.4f}",
                        f"{row['false_dynamic_mean']:.4f}",
                        f"{row['false_dynamic_std']:.4f}",
                        str(row["n_folds"]),
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stratified k-fold eval for Experiment B."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_PATH,
        help="Path to dynamic_raw_sequences.npz.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Fixed temporal length T.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=N_SPLITS,
        help="Number of stratified folds.",
    )
    args = parser.parse_args()

    run_kfold_eval(
        feature_path=args.features,
        num_samples=args.num_samples,
        n_splits=args.folds,
    )
