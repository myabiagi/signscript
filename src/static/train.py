"""Coordinate static ASL experiments and final model training."""

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import enhance_features
from data import (
    ASL_HG_PATH,
    LETTERS,
    load_features,
    split_data,
    combine_pugeault_training,
    load_webcam_domain_sources,
    repeat_rows,
)
from tuning import tune_knn, tune_random_forest
from metrics import print_validation

MODEL_DIR = Path("models")
EXPERIMENT_DIR = MODEL_DIR / "experiments"
SUPPLIED_MODEL = MODEL_DIR / "static_classifier.joblib"
RETRAINED_MODEL = MODEL_DIR / "static_classifier_retrained.joblib"


def _fit_final_rf(X_train, y_train):
    """Random Forest settings matching the shipped final classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


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

        print(f"Combined training images: {len(y_train)}")
        model = _fit_final_rf(X_train, y_train)
        output = RETRAINED_MODEL

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
    return output


def run_webcam_mix(webcam_weight=5, install=False):
    """
    Retrain RF on ASL-HG train (if present) + Graudt + local webcam.

    Webcam-domain rows are repeated `webcam_weight` times so live
    landmarks influence the decision boundary more than photo data.
    """
    parts_X = []
    parts_y = []
    X_val = y_val = None
    used_asl_hg = False

    if ASL_HG_PATH.exists():
        X, y, paths, participant_ids = load_features(ASL_HG_PATH)
        if X.shape[1] != 63:
            raise ValueError("Expected 63 ASL-HG features.")
        X = enhance_features(X)
        train, val, test = split_data(
            X, y, paths, participant_ids
        )
        X_train, y_train, _, _ = train
        X_val, y_val, _, _ = val
        parts_X.append(X_train)
        parts_y.append(y_train)
        used_asl_hg = True
        print(
            f"ASL-HG train={len(y_train)} val={len(y_val)} "
            f"held-out test={len(test[1])}"
        )
        try:
            X_train_p, y_train_p = combine_pugeault_training(
                X_train, y_train, X_val
            )
            # replace base train with pugeault-combined
            parts_X[0] = X_train_p
            parts_y[0] = y_train_p
            print(f"After Pugeault: train={len(y_train_p)}")
        except FileNotFoundError as exc:
            print(f"Pugeault skipped: {exc}")
    else:
        print(
            f"ASL-HG cache missing ({ASL_HG_PATH}) — "
            "training from webcam-domain sources only."
        )

    domain = load_webcam_domain_sources()
    if not domain and not used_asl_hg:
        raise FileNotFoundError(
            "Need at least one of: prepared ASL-HG, Graudt, or webcam. "
            "Download Graudt into data/static/graudt_landmarks/ then "
            "`python dev.py prepare-static graudt`, and/or "
            "`python dev.py collect-static` then `prepare-static webcam`."
        )

    for X_d, y_d, _, ids in domain:
        X_enh = enhance_features(X_d)

        # If we have no ASL-HG val set, hold out 15% of this source first
        # (before oversampling) so repeats cannot leak into validation.
        if X_val is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_enh,
                y_d,
                test_size=0.15,
                random_state=42,
                stratify=y_d,
            )
            X_val, y_val = X_va, y_va
            X_enh, y_d = X_tr, y_tr
            print(
                f"Hold-out val from {ids[0]}: "
                f"train={len(y_d)} val={len(y_val)}"
            )

        # Local webcam gets the full weight; Graudt gets half that
        # (still webcam-ish, but not your exact camera).
        is_local = np.all(ids == "WEBCAM")
        weight = webcam_weight if is_local else max(1, webcam_weight // 2)
        X_w, y_w = repeat_rows(X_enh, y_d, weight)
        print(
            f"Adding {len(y_d)} domain samples "
            f"(x{weight} → {len(y_w)}), source={ids[0]}"
        )
        parts_X.append(X_w)
        parts_y.append(y_w)

    X_all = np.concatenate(parts_X, axis=0)
    y_all = np.concatenate(parts_y, axis=0)

    missing = set(LETTERS) - set(np.unique(y_all))
    if missing:
        print(
            f"Warning: training missing letters {sorted(missing)} "
            "— model will not predict them."
        )

    print(f"Fitting RF on {len(y_all)} rows, dim={X_all.shape[1]}")
    model = _fit_final_rf(X_all, y_all)

    RETRAINED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RETRAINED_MODEL)
    print(f"\nSaved model: {RETRAINED_MODEL}")
    print_validation(model, X_val, y_val)

    if install:
        shutil.copy2(RETRAINED_MODEL, SUPPLIED_MODEL)
        print(f"Installed for live use → {SUPPLIED_MODEL}")
    else:
        print(
            f"\nLive still uses {SUPPLIED_MODEL}. "
            "Re-run with --install to replace it, or copy the retrained file."
        )

    return RETRAINED_MODEL


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
            "webcam-mix",
        ),
    )
    parser.add_argument(
        "--webcam-weight",
        type=int,
        default=5,
        help="Repeat factor for local webcam samples (default 5).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Copy retrained model over models/static_classifier.joblib",
    )

    args = parser.parse_args()
    if args.mode == "webcam-mix":
        run_webcam_mix(
            webcam_weight=args.webcam_weight,
            install=args.install,
        )
    else:
        run_training(args.mode)
