"""
Train Experiment B Random Forests for dynamic ASL letters.

Compares three landmark normalizations (raw / frame / sequence) with the
same stratified clip split and Random Forest settings. Negatives I and D
are used to measure false-dynamic predictions (I/D classified as J/Z).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from dynamic.normalize import NORMALIZERS, build_feature_matrix, resample_sequence

FEATURES_PATH = Path("data/processed/dynamic_raw_sequences.npz")
MODELS_DIR = Path("models/dynamic")
RESULTS_DIR = Path("results/dynamic")

# Fixed temporal length after resampling (shared across all methods)
NUM_SAMPLES = 30

# Random Forest hyperparameters (fixed so methods stay comparable)
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
RANDOM_STATE = 42
TEST_SIZE = 0.25

POSITIVE_LABELS = ("J", "Z")
NEGATIVE_LABELS = ("I", "D")
METHODS = tuple(NORMALIZERS.keys())


def load_dynamic_sequences(path=FEATURES_PATH):
    """
    Load the raw sequence cache produced by prepare_dynamic.

    Args:
        path (str or Path): Path to dynamic_raw_sequences.npz.

    Returns:
        dict: sequences, labels, clip_ids, class_names, etc.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python dev.py prepare-dynamic"
        )

    data = np.load(path, allow_pickle=True)
    sequences = list(data["sequences"])
    labels = np.asarray(data["labels"]).astype(str)
    clip_ids = np.asarray(data["clip_ids"]).astype(str)
    class_names = np.asarray(data["class_names"]).astype(str)

    if len(sequences) == 0:
        raise ValueError(
            f"{path} contains 0 clips. Re-run prepare-dynamic with data present."
        )

    return {
        "sequences": sequences,
        "labels": labels,
        "clip_ids": clip_ids,
        "class_names": class_names,
        "lengths": np.asarray(data["lengths"]),
        "valid_counts": np.asarray(data["valid_counts"]),
    }


def stratified_clip_split(labels, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split clip indices with stratification by label.

    Args:
        labels (numpy.ndarray): Per-clip class labels.
        test_size (float): Fraction of clips for the test set.
        random_state (int): RNG seed.

    Returns:
        tuple: (train_indices, test_indices)
    """
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_idx, test_idx


def false_dynamic_rate(y_true, y_pred):
    """
    Fraction of true I/D clips predicted as J or Z.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Rate in [0, 1], or nan if no negatives in y_true.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    negative_mask = np.isin(y_true, NEGATIVE_LABELS)

    if not negative_mask.any():
        return float("nan")

    false_dynamic = np.isin(y_pred[negative_mask], POSITIVE_LABELS)
    return float(false_dynamic.mean())


def train_random_forest(X_train, y_train):
    """Fit one Random Forest with the shared Experiment B settings."""
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """
    Compute test metrics for one normalization method.

    Returns:
        dict: predictions and scalar / matrix metrics.
    """
    predictions = model.predict(X_test)
    labels_order = list(class_names)

    return {
        "predictions": predictions,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=labels_order,
                average="macro",
                zero_division=0,
            )
        ),
        "false_dynamic_rate": false_dynamic_rate(y_test, predictions),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels_order
        ),
        "report": classification_report(
            y_test,
            predictions,
            labels=labels_order,
            zero_division=0,
        ),
    }


def save_confusion_matrix_plot(matrix, class_names, title, output_path):
    """Save a labeled confusion-matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(int(matrix[row, col])),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trajectory_examples(sequences, labels, output_path, num_samples=NUM_SAMPLES):
    """
    Plot wrist (x, y) paths for one J and one Z under each normalization.

    Helps show that frame-norm collapses motion while sequence-norm keeps it.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = {}
    for label in POSITIVE_LABELS:
        matches = [seq for seq, lab in zip(sequences, labels) if lab == label]
        if matches:
            examples[label] = matches[0]

    if not examples:
        return

    fig, axes = plt.subplots(
        len(examples),
        len(METHODS),
        figsize=(4 * len(METHODS), 3.5 * len(examples)),
        squeeze=False,
    )

    for row, (label, sequence) in enumerate(examples.items()):
        sampled = resample_sequence(sequence, num_samples)
        if sampled is None:
            continue

        for col, method in enumerate(METHODS):
            normalized = NORMALIZERS[method](sampled)
            wrist = normalized[:, 0, :2]
            ax = axes[row][col]
            ax.plot(wrist[:, 0], -wrist[:, 1], marker="o", markersize=3)
            ax.scatter(wrist[0, 0], -wrist[0, 1], c="green", s=40, label="start")
            ax.scatter(wrist[-1, 0], -wrist[-1, 1], c="red", s=40, label="end")
            ax.set_title(f"{label} — {method}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Wrist trajectory after each normalization", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_comparison_table(rows, output_path):
    """Write a simple CSV comparison of the three methods."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "method",
        "accuracy",
        "macro_f1",
        "false_dynamic_rate",
        "n_train",
        "n_test",
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    [
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


def train_all_methods(
    feature_path=FEATURES_PATH,
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    num_samples=NUM_SAMPLES,
):
    """
    Train and evaluate one RF per normalization method.

    Returns:
        list: Per-method result dicts for the comparison table.
    """
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_dynamic_sequences(feature_path)
    sequences = data["sequences"]
    labels = data["labels"]
    class_names = data["class_names"]

    train_idx, test_idx = stratified_clip_split(labels)

    print(
        f"Loaded {len(labels)} clips from {feature_path}\n"
        f"Classes: {list(class_names)}\n"
        f"Split: {len(train_idx)} train / {len(test_idx)} test "
        f"(T={num_samples}, RF trees={N_ESTIMATORS})\n"
    )

    plot_trajectory_examples(
        sequences,
        labels,
        results_dir / "trajectory_examples.png",
        num_samples=num_samples,
    )

    comparison_rows = []

    for method in METHODS:
        X, keep = build_feature_matrix(sequences, method, num_samples)
        kept_labels = labels[keep]
        kept_indices = np.where(keep)[0]

        # Map original split indices onto kept feature rows
        index_map = {int(old): new for new, old in enumerate(kept_indices)}
        train_rows = [index_map[i] for i in train_idx if i in index_map]
        test_rows = [index_map[i] for i in test_idx if i in index_map]

        if not train_rows or not test_rows:
            raise RuntimeError(
                f"Method '{method}' left an empty train or test set after filtering."
            )

        X_train = X[train_rows]
        y_train = kept_labels[train_rows]
        X_test = X[test_rows]
        y_test = kept_labels[test_rows]

        print(f"=== {method} normalization ===")
        print(f"Features: {X_train.shape[1]} dims | "
              f"train={len(y_train)} test={len(y_test)}")

        model = train_random_forest(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, class_names)

        model_path = models_dir / f"dynamic_rf_{method}.joblib"
        joblib.dump(
            {
                "model": model,
                "method": method,
                "num_samples": num_samples,
                "class_names": list(class_names),
                "include_motion": True,
            },
            model_path,
        )

        cm_path = results_dir / f"confusion_{method}.png"
        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            class_names,
            title=f"Confusion — {method}",
            output_path=cm_path,
        )

        report_path = results_dir / f"report_{method}.txt"
        report_path.write_text(metrics["report"], encoding="utf-8")

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"False-dynamic={metrics['false_dynamic_rate']:.4f}"
        )
        print(metrics["report"])
        print(f"Saved model → {model_path}")
        print(f"Saved confusion → {cm_path}\n")

        comparison_rows.append(
            {
                "method": method,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "false_dynamic_rate": metrics["false_dynamic_rate"],
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
        )

        # Binary J-vs-Z model for live: motion gate already implies a stroke,
        # so we should not split probability mass with static I/D.
        jz_mask_train = np.isin(y_train, POSITIVE_LABELS)
        jz_mask_test = np.isin(y_test, POSITIVE_LABELS)
        if jz_mask_train.sum() >= 4 and jz_mask_test.sum() >= 1:
            jz_model = train_random_forest(
                X_train[jz_mask_train], y_train[jz_mask_train]
            )
            jz_path = models_dir / f"dynamic_rf_jz_{method}.joblib"
            joblib.dump(
                {
                    "model": jz_model,
                    "method": method,
                    "num_samples": num_samples,
                    "class_names": list(POSITIVE_LABELS),
                    "include_motion": True,
                    "mode": "jz_binary",
                },
                jz_path,
            )
            jz_pred = jz_model.predict(X_test[jz_mask_test])
            jz_acc = float(
                accuracy_score(y_test[jz_mask_test], jz_pred)
            )
            print(
                f"J/Z binary [{method}]: test acc={jz_acc:.3f} "
                f"on {jz_mask_test.sum()} clips → {jz_path}"
            )

    table_path = results_dir / "comparison.csv"
    save_comparison_table(comparison_rows, table_path)

    print("Comparison (same split, same RF):")
    print(f"{'method':10s} {'acc':>8s} {'macroF1':>8s} {'falseDyn':>9s}")
    for row in comparison_rows:
        print(
            f"{row['method']:10s} "
            f"{row['accuracy']:8.4f} "
            f"{row['macro_f1']:8.4f} "
            f"{row['false_dynamic_rate']:9.4f}"
        )
    print(f"\nSaved comparison → {table_path}")
    print(f"Saved trajectories → {results_dir / 'trajectory_examples.png'}")

    best = max(comparison_rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
    print(
        f"\nBest by macro-F1: {best['method']} "
        f"(macro-F1={best['macro_f1']:.4f}, acc={best['accuracy']:.4f})"
    )

    return comparison_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Experiment B dynamic-letter Random Forests."
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
        help="Fixed temporal length T after resampling.",
    )
    args = parser.parse_args()

    train_all_methods(
        feature_path=args.features,
        num_samples=args.num_samples,
    )
