"""
src/visualize.py
----------------
Visualization utilities for thermal image analysis.

Functions
---------
plot_comparison(gray, colored, name, save)
    Side-by-side grayscale vs. pseudo-colored image.

plot_colormap_grid(gray, save)
    One grayscale image mapped with multiple colormaps.

plot_histograms(colored, name, save)
    RGB channel histograms of a pseudo-colored image.

plot_confusion_matrix(cm, labels, save)
    Annotated confusion matrix heatmap.

plot_metrics_bar(metrics, save)
    Bar chart of accuracy / precision / recall / F1.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config

logger = logging.getLogger("thermal.visualize")

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
})


# ── Grayscale vs. Pseudo-Color ────────────────────────────────────────────────

def plot_comparison(
    gray: np.ndarray,
    colored: np.ndarray,
    name: str = "image",
    colormap_name: str = config.COLORMAP_NAME,
    save: bool = config.SAVE_PLOTS,
    out_dir: str = config.OUTPUT_PLOT_DIR,
) -> None:
    """
    Side-by-side comparison of the grayscale and pseudo-colored image.

    Parameters
    ----------
    gray    : np.ndarray  (H, W)       — grayscale, uint8
    colored : np.ndarray  (H, W, 3)    — RGB pseudo-color, uint8
    name    : str         — used for the title and filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Thermal Image: {name}", fontsize=14, fontweight="bold")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale (Original)")
    axes[0].axis("off")

    axes[1].imshow(colored)
    axes[1].set_title(f"Pseudo-Color ({colormap_name})")
    axes[1].axis("off")

    # Add a colorbar strip for the pseudo-color side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    gradient = np.linspace(0, 255, 256).reshape(256, 1).astype(np.uint8)
    import cv2
    import config as cfg
    from src.colormap import get_cv2_colormap
    cmap_id = get_cv2_colormap(colormap_name)
    bar_bgr = cv2.applyColorMap(gradient, cmap_id)
    bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)
    cbar_ax.imshow(bar_rgb, aspect="auto", origin="upper")
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([0, 255])
    cbar_ax.set_yticklabels(["High", "Low"], fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    _save_or_show(fig, f"comparison_{name}", save, out_dir)


# ── Multi-colormap Grid ───────────────────────────────────────────────────────

def plot_colormap_grid(
    gray: np.ndarray,
    name: str = "image",
    save: bool = config.SAVE_PLOTS,
    out_dir: str = config.OUTPUT_PLOT_DIR,
) -> None:
    """
    Show one grayscale image rendered with all supported colormaps.
    """
    import cv2
    from src.colormap import _COLORMAP_MAP

    n      = len(_COLORMAP_MAP)
    ncols  = 4
    nrows  = (n + ncols - 1) // ncols + 1   # +1 row for original

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    # First panel: original grayscale
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale", fontsize=10)
    axes[0].axis("off")

    for idx, (cmap_name, cmap_id) in enumerate(_COLORMAP_MAP.items(), start=1):
        if idx >= len(axes):
            break
        bgr = cv2.applyColorMap(gray, cmap_id)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(rgb)
        axes[idx].set_title(cmap_name, fontsize=10)
        axes[idx].axis("off")

    for ax in axes[idx + 1:]:
        ax.axis("off")

    fig.suptitle(f"Colormap Comparison — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, f"colormap_grid_{name}", save, out_dir)


# ── RGB Channel Histograms ────────────────────────────────────────────────────

def plot_histograms(
    colored: np.ndarray,
    name: str = "image",
    save: bool = config.SAVE_PLOTS,
    out_dir: str = config.OUTPUT_PLOT_DIR,
) -> None:
    """
    Plot RGB channel histograms for a pseudo-colored image.
    """
    colors  = ["red", "green", "blue"]
    labels  = ["Red channel", "Green channel", "Blue channel"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(f"RGB Histograms — {name}", fontsize=13, fontweight="bold")

    for ch, (ax, color, label) in enumerate(zip(axes, colors, labels)):
        hist, bins = np.histogram(colored[:, :, ch], bins=config.HIST_BINS, range=(0, 256))
        ax.bar(
            bins[:-1], hist,
            width=(bins[1] - bins[0]),
            color=color, alpha=0.75, edgecolor="none",
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 256)

    plt.tight_layout()
    _save_or_show(fig, f"histograms_{name}", save, out_dir)


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    save: bool = config.SAVE_PLOTS,
    out_dir: str = config.OUTPUT_PLOT_DIR,
) -> None:
    """
    Annotated confusion matrix heatmap using seaborn.
    """
    import pandas as pd
    cm_arr = np.array(cm)
    df_cm  = pd.DataFrame(cm_arr, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        df_cm,
        annot=True, fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    _save_or_show(fig, "confusion_matrix", save, out_dir)


# ── Metrics Bar Chart ─────────────────────────────────────────────────────────

def plot_metrics_bar(
    metrics: dict,
    save: bool = config.SAVE_PLOTS,
    out_dir: str = config.OUTPUT_PLOT_DIR,
) -> None:
    """
    Bar chart of accuracy, precision, recall, and F1-score.
    """
    keys   = ["accuracy", "precision", "recall", "f1"]
    values = [metrics.get(k, 0) for k in keys]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, values, color=colors, edgecolor="white", width=0.5)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics", fontsize=13, fontweight="bold")
    ax.set_xticklabels([k.capitalize() for k in keys], fontsize=11)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    _save_or_show(fig, "metrics_bar", save, out_dir)


# ── Internal helper ───────────────────────────────────────────────────────────

def _save_or_show(
    fig: plt.Figure,
    filename: str,
    save: bool,
    out_dir: str,
) -> None:
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{filename}.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Plot saved → %s", path)
    else:
        plt.show()
    plt.close(fig)