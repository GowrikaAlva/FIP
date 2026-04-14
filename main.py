"""
main.py
-------
Entry point for the Pseudo-Color Mapping for Thermal Images project.

Pipeline
--------
1. Setup  — directories, logging
2. Load   — discover images in data/raw/
3. Preprocess — resize + noise reduction
4. Colormap   — apply pseudo-color mapping
5. Visualize  — save comparison & histogram plots
6. Features   — extract colour histogram + stats
7. Labels     — from folder name or auto from intensity
8. Train      — Random Forest / SVM
9. Evaluate   — accuracy, precision, recall, F1, confusion matrix
10. Save      — model, metrics, plots

Usage
-----
    python main.py                         # use defaults from config.py
    python main.py --data_dir data/raw     # custom data directory
    python main.py --colormap HOT          # different colormap
    python main.py --model svm             # use SVM instead of RF
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
from src.preprocess import preprocess_batch, save_processed
from src.colormap import apply_batch, save_colored
from src.features import extract_batch, auto_label_from_intensity
from src.model import train, evaluate, save_model, save_metrics
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
    p.add_argument("--data_dir",  default=config.DATA_RAW_DIR,    help="Path to raw image folder")
    p.add_argument("--colormap",  default=config.COLORMAP_NAME,   help="Colormap name (JET, HOT, …)")
    p.add_argument("--model",     default=config.MODEL_TYPE,      help="Model type: rf | svm")
    p.add_argument("--no_save",   action="store_true",             help="Disable saving outputs")
    return p.parse_args()


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run(data_dir: str, colormap_name: str, model_type: str, save: bool) -> None:
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
        # ── Demo mode: generate synthetic images for testing ─────────────────
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

    # ── 4. Pseudo-color mapping ───────────────────────────────────────────────
    records = apply_batch(records, name=colormap_name)

    if save:
        for rec in records:
            save_colored(rec["colored"], rec["name"], colormap_name)

    # ── 5. Visualizations ─────────────────────────────────────────────────────
    # Show comparison for the first 3 images (or all if fewer)
    for rec in records[:3]:
        plot_comparison(rec["processed"], rec["colored"],
                        name=rec["name"], colormap_name=colormap_name, save=save)
        plot_histograms(rec["colored"], name=rec["name"], save=save)

    # Colormap grid for the very first image
    plot_colormap_grid(records[0]["processed"], name=records[0]["name"], save=save)

    # ── 6 & 7. Features + Labels ──────────────────────────────────────────────
    # Assign labels: prefer folder-based labels; fall back to intensity thresholding
    for rec in records:
        folder_label = label_from_path(rec["path"])
        if folder_label != "unknown":
            rec["label"] = folder_label
        else:
            rec["label"] = auto_label_from_intensity(rec["processed"])

    X, y = extract_batch(records)

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

    # ── 8. Train ──────────────────────────────────────────────────────────────
    model, X_test, y_test, y_pred, le = train(X, y, model_type=model_type)

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate(y_test, y_pred, labels=unique_labels)
    print("\n" + "=" * 60)
    print("  CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print("=" * 60)
    print(metrics["classification_report"])

    # ── 10. Save outputs ──────────────────────────────────────────────────────
    if save:
        # Model
        model_path = os.path.join(config.MODELS_DIR, f"model_{model_type}.pkl")
        save_model(model, model_path)

        # Metrics
        save_metrics(metrics)

    # Plots (always generated; save flag controls whether they're written to disk)
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
            # Simulate a thermal blob on a background
            img  = rng.integers(lo, hi, (256, 256), dtype=np.uint8)
            # Add a "hot spot"
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
        save=not args.no_save,
    )