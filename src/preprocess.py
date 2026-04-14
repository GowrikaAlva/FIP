"""
src/preprocess.py
-----------------
Load grayscale thermal images, resize them, and apply noise reduction.

Public API
----------
load_grayscale(path)   → np.ndarray  (H×W, uint8)
preprocess(image)      → np.ndarray  (H×W, uint8)
preprocess_batch(paths) → list[dict]
"""

import logging
import os

import cv2
import numpy as np

import config
from src.utils import stem

logger = logging.getLogger("thermal.preprocess")


# ── Single-image helpers ───────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    """
    Load an image from *path* and convert it to grayscale.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W) and dtype uint8.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV fails to decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV could not decode: {path}")

    logger.debug("Loaded %s  shape=%s", path, img.shape)
    return img


def resize(image: np.ndarray, size: tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """
    Resize *image* to *size* = (width, height) using bilinear interpolation.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def reduce_noise(image: np.ndarray, kernel_size: int = config.NOISE_KERNEL) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Parameters
    ----------
    kernel_size : int
        Must be an odd positive integer.  Larger values = stronger smoothing.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1          # force odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to enhance
    local contrast — useful for low-contrast thermal images.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for a single grayscale image:
    1. Resize to config.IMAGE_SIZE
    2. Gaussian noise reduction
    3. CLAHE contrast enhancement

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale image (any size).

    Returns
    -------
    np.ndarray
        Preprocessed grayscale image of shape config.IMAGE_SIZE[::-1].
    """
    image = resize(image)
    image = reduce_noise(image)
    image = equalize_histogram(image)
    return image


# ── Batch helper ──────────────────────────────────────────────────────────────

def preprocess_batch(paths: list[str]) -> list[dict]:
    """
    Load and preprocess a list of image paths.

    Parameters
    ----------
    paths : list[str]
        Paths returned by ``utils.list_images()``.

    Returns
    -------
    list[dict]
        Each element is::

            {
                "path":      str,          # original path
                "name":      str,          # filename stem
                "raw":       np.ndarray,   # loaded grayscale (before preprocess)
                "processed": np.ndarray,   # after preprocess()
            }

        Entries that fail to load are skipped (a warning is logged).
    """
    results = []
    for path in paths:
        try:
            raw  = load_grayscale(path)
            proc = preprocess(raw)
            results.append({
                "path":      path,
                "name":      stem(path),
                "raw":       raw,
                "processed": proc,
            })
        except Exception as exc:
            logger.warning("Skipping %s — %s", path, exc)

    logger.info("Preprocessed %d / %d images", len(results), len(paths))
    return results


# ── Save helper ───────────────────────────────────────────────────────────────

def save_processed(image: np.ndarray, name: str, out_dir: str = config.DATA_PROC_DIR) -> str:
    """Save a preprocessed grayscale image and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_processed.png")
    cv2.imwrite(out_path, image)
    return out_path