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
build_model(model_type)                      → sklearn estimator
train(X, y, model_type)                      → (pipeline, X_test, y_test, y_pred, le)
evaluate(y_test, y_pred, labels)             → dict of metrics
evaluate_cv(X, y, model_type, n_splits)      → (mean_f1, std_f1)
tune_hyperparams(X, y, model_type, n_iter)   → fitted Pipeline with best params
save_model(model, path)
load_model(path)                             → sklearn Pipeline
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def _build_pipeline(model_type: str = config.MODEL_TYPE) -> Pipeline:
    """
    Wrap the estimator in a StandardScaler → model Pipeline.

    StandardScaler is critical for SVM (features on different scales cause
    the kernel to ignore low-variance dimensions).  It also slightly
    stabilises Random Forest.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  build_model(model_type)),
    ])


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: list[str],
    model_type: str = config.MODEL_TYPE,
) -> tuple[Pipeline, np.ndarray, list[str], list[str], LabelEncoder]:
    """
    Split data, train a scaled pipeline, and return predictions.

    Parameters
    ----------
    X : np.ndarray  shape (N, F)
    y : list[str]   length N  — string class labels
    model_type : str

    Returns
    -------
    pipeline    : fitted sklearn Pipeline  (scaler + model)
    X_test      : np.ndarray  test features (unscaled — pipeline scales internally)
    y_test      : list[str]   true string labels
    y_pred      : list[str]   predicted string labels
    le          : LabelEncoder  (for inverse-transforming if needed)
    """
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)

    X_train, X_test, y_train, y_test_enc = train_test_split(
        X, y_enc,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_enc if len(set(y_enc)) > 1 else None,
    )

    pipeline = _build_pipeline(model_type)
    logger.info(
        "Training on %d samples, testing on %d samples …",
        len(X_train), len(X_test),
    )
    pipeline.fit(X_train, y_train)

    y_pred_enc = pipeline.predict(X_test)

    y_test = le.inverse_transform(y_test_enc).tolist()
    y_pred = le.inverse_transform(y_pred_enc).tolist()

    logger.info("Training complete.")
    return pipeline, X_test, y_test, y_pred, le


# ── Cross-validated evaluation ────────────────────────────────────────────────

def evaluate_cv(
    X: np.ndarray,
    y: list[str],
    model_type: str = config.MODEL_TYPE,
    n_splits: int = 5,
) -> tuple[float, float]:
    """
    Evaluate the model with Stratified K-Fold cross-validation.

    Much more reliable than a single train/test split, especially when
    the total dataset is small (< 200 samples).  Uses ALL data for both
    training and validation across folds.

    Parameters
    ----------
    X        : np.ndarray  shape (N, F)
    y        : list[str]   length N
    model_type : str
    n_splits : int         number of CV folds (default 5)

    Returns
    -------
    mean_f1 : float   mean weighted-F1 across folds
    std_f1  : float   standard deviation of weighted-F1
    """
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = _build_pipeline(model_type)
    cv       = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    scores = cross_val_score(
        pipeline, X, y_enc,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    logger.info(
        "CV (%d-fold) weighted-F1: %.4f ± %.4f",
        n_splits, scores.mean(), scores.std(),
    )
    return float(scores.mean()), float(scores.std())


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_hyperparams(
    X: np.ndarray,
    y: list[str],
    model_type: str = config.MODEL_TYPE,
    n_iter: int = 20,
) -> Pipeline:
    """
    Run RandomizedSearchCV to find better hyperparameters, then refit on
    the full dataset with the best params.

    Parameters
    ----------
    X          : np.ndarray
    y          : list[str]
    model_type : str
    n_iter     : int   number of random parameter combinations to try

    Returns
    -------
    Pipeline
        Fitted pipeline with best found hyperparameters.
    """
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = _build_pipeline(model_type)
    cv       = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    mtype = model_type.lower()
    if mtype == "rf":
        param_dist = {
            "model__n_estimators":     [50, 100, 200, 300, 500],
            "model__max_depth":        [None, 5, 10, 20, 30],
            "model__min_samples_split":[2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features":     ["sqrt", "log2", None],
        }
    elif mtype == "svm":
        param_dist = {
            "model__C":     [0.01, 0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "model__kernel":["rbf", "poly", "linear"],
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y_enc)

    logger.info("Best CV F1:     %.4f", search.best_score_)
    logger.info("Best params:    %s",   search.best_params_)

    return search.best_estimator_


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

    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    logger.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f", acc, prec, rec, f1)
    logger.info("\n%s", report)

    return {
        "accuracy":              round(acc,  4),
        "precision":             round(prec, 4),
        "recall":                round(rec,  4),
        "f1":                    round(f1,   4),
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
        "labels":                labels,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: Any, path: str) -> None:
    """Persist the trained pipeline/model with joblib."""
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

    serialisable = {
        k: v for k, v in metrics.items()
        if k != "classification_report"
    }
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)

    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(metrics["classification_report"])

    logger.info("Metrics saved → %s", out_dir)