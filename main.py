"""
main.py
-------
Entry point for the Pseudo-Color Mapping for Thermal Images project.

Pipeline
--------
1.  Setup       — directories, logging
2.  Load        — discover images in data/raw/
3.  Preprocess  — resize + noise reduction + CLAHE
4.  Augment     — optional dataset expansion (×8 per image)
5.  Colormap    — apply pseudo-color mapping
6.  Visualize   — save comparison & histogram plots
7.  Features    — extract colour histogram + stats + HSV + spatial grid
8.  Labels      — from folder name or auto from intensity
9.  CV Eval     — stratified k-fold cross-validation before final train
10. Tune        — optional RandomizedSearchCV hyperparameter tuning
11. Train       — Random Forest / SVM on train/test split
12. Evaluate    — accuracy, precision, recall, F1, confusion matrix
13. Save        — model, metrics, plots

Usage
-----
    python main.py                         # use defaults from config.py
    python main.py --data_dir data/raw     # custom data directory
    python main.py --colormap HOT          # different colormap
    python main.py --model svm             # use SVM instead of RF
    python main.py --augment               # enable data augmentation
    python main.py --tune                  # enable hyperparameter tuning
    python main.py --cv_folds 10           # custom number of CV folds
    python main.py --no_save               # skip saving outputs
"""

import argparse
import logging
import os
import sys

# ── Make src importable when run from project root ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils import ensure_directories, list_images, label_from_path, setup_logging
from src.preprocess import preprocess_batch, augment_batch, save_processed
from src.colormap import apply_batch, save_colored
from src.features import extract_batch, auto_label_from_intensity, feature_names
from src.model import (
    train,
    evaluate,
    evaluate_cv,
    tune_hyperparams,
    save_model,
    load_model,
    save_metrics,
)
from src.visualize import (
    plot_comparison,
    plot_colormap_grid,
    plot_histograms,
    plot_confusion_matrix,
    plot_metrics_bar,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-Color Mapping for Thermal Images")
    p.add_argument("--data_dir",  default=config.DATA_RAW_DIR,   help="Path to raw image folder")
    p.add_argument("--colormap",  default=config.COLORMAP_NAME,  help="Colormap name (JET, HOT, …)")
    p.add_argument("--model",     default=config.MODEL_TYPE,     help="Model type: rf | svm")
    p.add_argument("--augment",   action="store_true",           help="Enable data augmentation (×8)")
    p.add_argument("--tune",      action="store_true",           help="Run hyperparameter tuning before final train")
    p.add_argument("--cv_folds",  type=int, default=5,           help="Number of stratified CV folds (default: 5)")
    p.add_argument("--no_save",   action="store_true",           help="Disable saving outputs")
    return p.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(
    data_dir: str,
    colormap_name: str,
    model_type: str,
    augment: bool,
    tune: bool,
    cv_folds: int,
    save: bool,
) -> None:
    logger = logging.getLogger("thermal.main")

    # ── 1. Setup ──────────────────────────────────────────────────────────────
    ensure_directories()

    # ── 2. Discover images ────────────────────────────────────────────────────
    image_paths = list_images(data_dir)
    if not image_paths:
        logger.error(
            "No images found in '%s'.  "
            "Place FLIR thermal images (jpg/png) there and re-run.",
            data_dir,
        )
        logger.warning("Running in DEMO mode with synthetic thermal images.")
        image_paths = _generate_demo_images(data_dir)

    logger.info("Found %d image(s) in %s", len(image_paths), data_dir)

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    records = preprocess_batch(image_paths)
    if not records:
        logger.error("No images could be loaded. Aborting.")
        return

    if save:
        for rec in records:
            save_processed(rec["processed"], rec["name"])

    # ── 4. Labels (assign before augmentation so variants inherit them) ───────
    for rec in records:
        folder_label = label_from_path(rec["path"])
        rec["label"] = (
            folder_label
            if folder_label != "unknown"
            else auto_label_from_intensity(rec["processed"])
        )

    # ── 5. Augmentation (optional) ────────────────────────────────────────────
    if augment:
        logger.info("Augmentation enabled — expanding dataset …")
        records = augment_batch(records)
        logger.info("Dataset size after augmentation: %d records", len(records))
    else:
        logger.info("Augmentation disabled (pass --augment to enable).")

    # ── 6. Pseudo-color mapping ───────────────────────────────────────────────
    records = apply_batch(records, name=colormap_name)

    if save:
        for rec in records:
            save_colored(rec["colored"], rec["name"], colormap_name)

    # ── 7. Visualizations ─────────────────────────────────────────────────────
    # Show comparison for the first 3 *original* images (skip aug variants)
    originals = [r for r in records if "_aug" not in r["name"]]
    for rec in originals[:3]:
        plot_comparison(
            rec["processed"], rec["colored"],
            name=rec["name"], colormap_name=colormap_name, save=save,
        )
        plot_histograms(rec["colored"], name=rec["name"], save=save)

    plot_colormap_grid(originals[0]["processed"], name=originals[0]["name"], save=save)

    # ── 8. Feature extraction ─────────────────────────────────────────────────
    X, y = extract_batch(records)

    logger.info(
        "Feature vector length: %d  |  Feature names sample: %s …",
        X.shape[1],
        feature_names()[:5],
    )

    # Abort classification if only one class present
    unique_labels = sorted(set(y))
    if len(unique_labels) < 2:
        logger.warning(
            "Only one class found (%s). "
            "Skipping classification — add more labelled data.",
            unique_labels,
        )
        logger.info("Pipeline complete (visualisation only).")
        return

    # ── 9. Cross-validation ───────────────────────────────────────────────────
    logger.info("Running %d-fold stratified cross-validation …", cv_folds)
    mean_f1, std_f1 = evaluate_cv(X, y, model_type=model_type, n_splits=cv_folds)

    print("\n" + "=" * 60)
    print(f"  CROSS-VALIDATION  ({cv_folds}-fold, weighted F1)")
    print("=" * 60)
    print(f"  Mean F1 : {mean_f1:.4f}")
    print(f"  Std  F1 : {std_f1:.4f}")
    print("=" * 60)

    # ── 10. Hyperparameter tuning (optional) ──────────────────────────────────
    best_params: dict | None = None
    if tune:
        logger.info("Hyperparameter tuning enabled — running RandomizedSearchCV …")
        best_pipeline = tune_hyperparams(X, y, model_type=model_type, n_iter=20)
        logger.info("Tuning complete. Best pipeline: %s", best_pipeline)

        # Extract best params so train() can use them
        best_params = {
            k.replace("model__", ""): v
            for k, v in best_pipeline.get_params().items()
            if k.startswith("model__")
        }
        logger.info("Best params forwarded to final train: %s", best_params)

        if save:
            tuned_path = os.path.join(config.MODELS_DIR, f"model_{model_type}_tuned.pkl")
            save_model(best_pipeline, tuned_path)
            logger.info("Tuned model saved → %s", tuned_path)
    else:
        logger.info("Hyperparameter tuning disabled (pass --tune to enable).")

    # ── 11. Train (final train/test split, with best params if tuned) ─────────
    # Temporarily patch config so train() picks up tuned hyperparameters
    if best_params and model_type.lower() == "svm":
        _orig = {
            "SVM_C":      config.SVM_C,
            "SVM_KERNEL": config.SVM_KERNEL,
            "SVM_GAMMA":  config.SVM_GAMMA,
        }
        config.SVM_C      = best_params.get("C",      config.SVM_C)
        config.SVM_KERNEL = best_params.get("kernel", config.SVM_KERNEL)
        config.SVM_GAMMA  = best_params.get("gamma",  config.SVM_GAMMA)
    elif best_params and model_type.lower() == "rf":
        _orig = {
            "RF_N_ESTIMATORS": config.RF_N_ESTIMATORS,
            "RF_MAX_DEPTH":    config.RF_MAX_DEPTH,
        }
        config.RF_N_ESTIMATORS = best_params.get("n_estimators", config.RF_N_ESTIMATORS)
        config.RF_MAX_DEPTH    = best_params.get("max_depth",    config.RF_MAX_DEPTH)
    else:
        _orig = {}

    model, X_test, y_test, y_pred, le = train(X, y, model_type=model_type)

    # Restore original config values
    for k, v in _orig.items():
        setattr(config, k, v)

    # ── 12. Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate(y_test, y_pred, labels=unique_labels)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION RESULTS  (hold-out test set)")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print("=" * 60)
    print(metrics["classification_report"])

    # ── 13. Save outputs ──────────────────────────────────────────────────────
    if save:
        model_path = os.path.join(config.MODELS_DIR, f"model_{model_type}.pkl")
        save_model(model, model_path)
        save_metrics(metrics)

    plot_confusion_matrix(metrics["confusion_matrix"], unique_labels, save=save)
    plot_metrics_bar(metrics, save=save)

    logger.info("Pipeline complete.  Outputs in: %s", config.BASE_DIR)


# ── Demo helper ───────────────────────────────────────────────────────────────

def _generate_demo_images(out_dir: str) -> list[str]:
    """
    Create synthetic grayscale thermal images for testing when no real
    dataset is available.  Three temperature classes × 10 images each.
    """
    import numpy as np
    import cv2

    rng = np.random.default_rng(42)
    paths = []
    classes = {"low": (20, 80), "medium": (80, 170), "high": (170, 240)}

    for cls, (lo, hi) in classes.items():
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(10):
            img = rng.integers(lo, hi, (256, 256), dtype=np.uint8)
            cx, cy = rng.integers(60, 196, 2)
            for r in range(30):
                val = min(255, hi + r * 2)
                cv2.circle(img, (cx, cy), 30 - r, int(val), 1)
            path = os.path.join(cls_dir, f"{cls}_{i:03d}.png")
            cv2.imwrite(path, img)
            paths.append(path)

    return paths


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = parse_args()
    logger = setup_logging()

    run(
        data_dir=args.data_dir,
        colormap_name=args.colormap.upper(),
        model_type=args.model,
        augment=args.augment,
        tune=args.tune,
        cv_folds=args.cv_folds,
        save=not args.no_save,
    )