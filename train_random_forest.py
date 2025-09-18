from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ----------------------------
# Config & paths
# ----------------------------

DATA_DIR = Path("dati")
INPUT_CSV = DATA_DIR / "snn_features_scaled.csv"
AGG_FEATURE_IMPORTANCE_CSV = DATA_DIR / "rf_feature_importance_by_type.csv"
CONFUSION_MATRIX_IMAGE = DATA_DIR / "confusion_matrix_rf.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS: Dict = {
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

def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load dataset and return X, y, feature names."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    df = pd.read_csv(csv_path).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    X = df.drop(columns=["label"]).to_numpy()
    y = df["label"].to_numpy()
    feature_names = df.columns.drop("label").tolist()
    return X, y, feature_names


# ----------------------------
# Modeling
# ----------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict = None,
) -> RandomForestClassifier:
    """Fit a Random Forest with provided params."""
    model = RandomForestClassifier(**(params or RF_PARAMS))
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Compute train/test accuracy and confusion matrix on test."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    return train_acc, test_acc, cm


# ----------------------------
# Analysis & plotting
# ----------------------------

def plot_and_save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Plot and save confusion matrix image."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Optimized RF")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.show()
    print(f"📊 Confusion matrix saved to: {out_path}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load & split
    X, y, feature_names = load_dataset(INPUT_CSV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Train
    model = train_random_forest(X_train, y_train, RF_PARAMS)

    # Evaluate
    train_acc, test_acc, cm = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"✅ Train accuracy: {train_acc:.4f}")
    print(f"✅ Test  accuracy: {test_acc:.4f}")

    # Confusion matrix
    plot_and_save_confusion_matrix(cm, CONFUSION_MATRIX_IMAGE)


if __name__ == "__main__":
    main()
