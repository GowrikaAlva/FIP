"""
src/model.py
------------
Train and evaluate a machine learning classifier for temperature categories.

Supported models
----------------
- ``"rf"``  → scikit-learn RandomForestClassifier
- ``"svm"`` → scikit-learn SVC with probability estimates

Public API
----------
build_model(model_type)        → sklearn estimator
train(X, y, model_type)        → (fitted model, X_test, y_test, y_pred)
evaluate(y_test, y_pred, labels) → dict of metrics
save_model(model, path)
load_model(path)               → sklearn estimator
"""

import json
import logging
import os
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import config

logger = logging.getLogger("thermal.model")


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type: str = config.MODEL_TYPE) -> Any:
    """
    Instantiate and return an untrained sklearn estimator.

    Parameters
    ----------
    model_type : str
        ``"rf"`` for Random Forest or ``"svm"`` for SVM.

    Raises
    ------
    ValueError
        For unknown model types.
    """
    mtype = model_type.lower()
    if mtype == "rf":
        model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        logger.info("Built RandomForestClassifier (n_estimators=%d)", config.RF_N_ESTIMATORS)
    elif mtype == "svm":
        model = SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            probability=True,
            random_state=config.RANDOM_STATE,
        )
        logger.info("Built SVC (kernel=%s, C=%s)", config.SVM_KERNEL, config.SVM_C)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'rf' or 'svm'.")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: list[str],
    model_type: str = config.MODEL_TYPE,
) -> tuple[Any, np.ndarray, list[str], list[str], LabelEncoder]:
    """
    Split data, train the model, and return predictions.

    Parameters
    ----------
    X : np.ndarray  shape (N, F)
    y : list[str]   length N  — string class labels
    model_type : str

    Returns
    -------
    model       : fitted sklearn estimator
    X_test      : np.ndarray  test features
    y_test      : list[str]   true string labels
    y_pred      : list[str]   predicted string labels
    le          : LabelEncoder  (for inverse-transforming if needed)
    """
    # Encode string labels → integers
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)

    X_train, X_test, y_train, y_test_enc = train_test_split(
        X, y_enc,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_enc if len(set(y_enc)) > 1 else None,
    )

    model = build_model(model_type)
    logger.info(
        "Training on %d samples, testing on %d samples …",
        len(X_train), len(X_test),
    )
    model.fit(X_train, y_train)

    y_pred_enc = model.predict(X_test)

    # Decode back to string labels
    y_test = le.inverse_transform(y_test_enc).tolist()
    y_pred = le.inverse_transform(y_pred_enc).tolist()

    logger.info("Training complete.")
    return model, X_test, y_test, y_pred, le


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    y_test: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_test  : list[str]  true labels
    y_pred  : list[str]  predicted labels
    labels  : list[str]  ordered class names (default: sorted unique values)

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1,
        confusion_matrix (list[list[int]]),
        classification_report (str),
        labels (list[str])
    """
    if labels is None:
        labels = sorted(set(y_test) | set(y_pred))

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    logger.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f", acc, prec, rec, f1)
    logger.info("\n%s", report)

    return {
        "accuracy":               round(acc, 4),
        "precision":              round(prec, 4),
        "recall":                 round(rec, 4),
        "f1":                     round(f1, 4),
        "confusion_matrix":       cm.tolist(),
        "classification_report":  report,
        "labels":                 labels,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: Any, path: str) -> None:
    """Persist the trained model with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved-> %s", path)


def load_model(path: str) -> Any:
    """Load a model previously saved with ``save_model``."""
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


def save_metrics(metrics: dict, out_dir: str = config.OUTPUT_MET_DIR) -> None:
    """Save evaluation metrics as a JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.json")

    # confusion_matrix is already list[list[int]], report is str
    serialisable = {
        k: v for k, v in metrics.items()
        if k != "classification_report"
    }
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)

    # Save the text report separately
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(metrics["classification_report"])

    logger.info("Metrics saved → %s", out_dir)