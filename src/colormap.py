"""
src/colormap.py
---------------
Apply pseudo-color mapping to grayscale thermal images.

OpenCV's ``applyColorMap`` maps each intensity value (0–255) to a BGR colour
using a predefined palette.  We convert the result to RGB for consistency with
matplotlib and scikit-learn pipelines.

Public API
----------
apply_colormap(gray, name)   → np.ndarray  (H×W×3, uint8, RGB)
apply_batch(records, name)   → list[dict]   (adds "colored" key to each record)
get_cv2_colormap(name)       → int          (cv2.COLORMAP_* constant)
"""

import logging
import os

import cv2
import numpy as np

import config
from src.utils import stem

logger = logging.getLogger("thermal.colormap")

# Map string name → OpenCV constant (evaluated lazily to avoid import-time cv2 dep issues)
_COLORMAP_MAP: dict[str, int] = {
    "JET":     cv2.COLORMAP_JET,
    "HOT":     cv2.COLORMAP_HOT,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "PARULA":  cv2.COLORMAP_PARULA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "MAGMA":   cv2.COLORMAP_MAGMA,
    "PLASMA":  cv2.COLORMAP_PLASMA,
    "BONE":    cv2.COLORMAP_BONE,
    "OCEAN":   cv2.COLORMAP_OCEAN,
}


# ── Core function ─────────────────────────────────────────────────────────────

def get_cv2_colormap(name: str) -> int:
    """
    Convert a colormap name string to the corresponding OpenCV constant.

    Parameters
    ----------
    name : str
        Case-insensitive colormap name (e.g. ``"jet"``, ``"HOT"``).

    Raises
    ------
    ValueError
        If *name* is not a supported colormap.
    """
    key = name.upper()
    if key not in _COLORMAP_MAP:
        raise ValueError(
            f"Unknown colormap '{name}'. "
            f"Choose from: {list(_COLORMAP_MAP.keys())}"
        )
    return _COLORMAP_MAP[key]


def apply_colormap(
    gray: np.ndarray,
    name: str = config.COLORMAP_NAME,
) -> np.ndarray:
    """
    Apply a pseudo-color mapping to a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of shape (H, W) and dtype uint8.
    name : str
        Colormap name (see ``_COLORMAP_MAP`` keys).

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3) and dtype uint8.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale array, got shape {gray.shape}")

    cv2_cmap = get_cv2_colormap(name)
    bgr      = cv2.applyColorMap(gray, cv2_cmap)
    rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    logger.debug("Applied %s colormap → shape=%s", name, rgb.shape)
    return rgb


# ── Batch helper ──────────────────────────────────────────────────────────────

def apply_batch(
    records: list[dict],
    name: str = config.COLORMAP_NAME,
) -> list[dict]:
    """
    Add pseudo-colored versions to a list of image records.

    Each record must have a ``"processed"`` key (grayscale np.ndarray).
    A ``"colored"`` key (RGB np.ndarray) is added in place.

    Parameters
    ----------
    records : list[dict]
        Output from ``preprocess.preprocess_batch()``.
    name : str
        Colormap name.

    Returns
    -------
    list[dict]
        Same list with ``"colored"`` added to every record.
    """
    for rec in records:
        rec["colored"] = apply_colormap(rec["processed"], name)
    logger.info("Colormap '%s' applied to %d images", name, len(records))
    return records


# ── Save helper ───────────────────────────────────────────────────────────────

def save_colored(
    image: np.ndarray,
    name: str,
    colormap_name: str = config.COLORMAP_NAME,
    out_dir: str = config.OUTPUT_IMG_DIR,
) -> str:
    """
    Save a pseudo-colored (RGB) image to disk.

    OpenCV expects BGR, so we convert before writing.

    Returns
    -------
    str
        Path of the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    bgr      = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(out_dir, f"{name}_{colormap_name.lower()}.png")
    cv2.imwrite(out_path, bgr)
    logger.debug("Saved colored image → %s", out_path)
    return out_path