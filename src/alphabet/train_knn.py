from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

K_VALUES = [1, 3, 5, 7, 11, 15, 21, 31]
WEIGHTS = ["uniform", "distance"]
METRICS = ["euclidean", "manhattan"]

def train_knn(X_train, y_train, X_val, y_val):
    """Train KNN configurations and select the best using validation macro-F1."""
    best_model = None
    best_f1 = -1
    best_accuracy = -1
    results = []

    for metric in METRICS:
        for weight in WEIGHTS:
            for k in K_VALUES:
                model = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weight,
                    metric=metric
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
                    "k": k,
                    "weights": weight,
                    "metric": metric,
                    "accuracy": accuracy,
                    "macro_f1": f1
                })

                print(
                    f"k={k:2d}, "
                    f"weights={weight:8s}, "
                    f"metric={metric:9s} | "
                    f"Accuracy={accuracy:.4f}, "
                    f"Macro-F1={f1:.4f}"
                )

                # Prefer higher macro-F1, then higher accuracy
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

    print("\nTop 5 KNN configurations:")

    for result in results[:5]:
        print(
            f"k={result['k']}, "
            f"weights={result['weights']}, "
            f"metric={result['metric']} | "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Macro-F1={result['macro_f1']:.4f}"
        )

    return best_model, results
