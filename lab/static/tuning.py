"""Search KNN and Random Forest hyperparameters on P8 validation."""

from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from metrics import evaluate

def tune_knn(X_train, y_train, X_val, y_val):
    """Tune KNN using validation macro-F1."""
    best_model = None
    best_score = (-1.0, -1.0)

    for metric in ("euclidean", "manhattan"):
        for weights in ("uniform", "distance"):
            for k in (1, 3, 5, 7, 11, 15, 21, 31):
                model = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    metric=metric,
                )

                model.fit(X_train, y_train)

                accuracy, macro_f1 = evaluate(
                    model, X_val, y_val
                )

                print(
                    f"k={k}, weights={weights}, "
                    f"metric={metric}: "
                    f"accuracy={accuracy:.4f}, "
                    f"macro-F1={macro_f1:.4f}",
                    flush=True,
                )

                candidate = (macro_f1, accuracy)

                if candidate > best_score:
                    best_model = model
                    best_score = candidate

    return best_model

def tune_random_forest(X_train, y_train, X_val, y_val):
    """Tune Random Forest using validation macro-F1."""
    best_model = None
    best_score = (-1.0, -1.0)

    for trees, depth, leaf in product(
        (100, 200),
        (None, 15, 30),
        (1, 3, 5),
    ):
        model = RandomForestClassifier(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        accuracy, macro_f1 = evaluate(
            model, X_val, y_val
        )

        print(
            f"trees={trees}, depth={depth}, "
            f"leaf={leaf}: "
            f"accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}",
            flush=True,
        )

        candidate = (macro_f1, accuracy)

        if candidate > best_score:
            best_model = model
            best_score = candidate

    return best_model