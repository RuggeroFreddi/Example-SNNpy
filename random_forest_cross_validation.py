from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# ----------------------------
# Config & paths
# ----------------------------

DATA_DIR = Path("dati")
INPUT_CSV = DATA_DIR / "snn_features_scaled.csv"
AGG_FEATURE_IMPORTANCE_CSV = DATA_DIR / "rf_feature_importance_cv_by_type.csv"

N_SPLITS = 10
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ----------------------------
# I/O
# ----------------------------

def load_features(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features from CSV, return X, y, and feature names."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).to_numpy()
    y = df["label"].to_numpy()
    feature_names = df.columns.drop("label").tolist()
    return X, y, feature_names


# ----------------------------
# Modeling
# ----------------------------

def cross_validate_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    rf_params: Dict = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
    """Run Stratified K-Fold CV with RF; return accuracies, confusion matrices."""
    if rf_params is None:
        rf_params = RF_PARAMS

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies: List[float] = []
    confusion_mats: List[np.ndarray] = []
    importances: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        accuracies.append(acc)
        confusion_mats.append(cm)
        importances.append(model.feature_importances_)

        print(f"Fold {fold_idx:02d} — Accuracy: {acc:.4f}")

    return accuracies, confusion_mats


# ----------------------------
# Analysis & plotting
# ----------------------------

def plot_mean_confusion_matrix(conf_mats: List[np.ndarray], title: str = "Mean Confusion Matrix (RF CV)") -> None:
    """Plot the mean confusion matrix over folds."""
    mean_cm = np.mean(conf_mats, axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, feature_names = load_features(INPUT_CSV)

    # CV with Random Forest
    accuracies, confusion_mats = cross_validate_random_forest(
        X, y, n_splits=N_SPLITS, rf_params=RF_PARAMS, random_state=RANDOM_STATE
    )

    # Summary
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    print("\n=== Cross-Validation Summary ===")
    print(f"Mean Accuracy:       {mean_acc:.4f}")
    print(f"Standard Deviation:  {std_acc:.4f}")
    print(f"Fold Accuracies:     {[round(a, 4) for a in accuracies]}")

    # Mean confusion matrix
    plot_mean_confusion_matrix(confusion_mats)


if __name__ == "__main__":
    main()
