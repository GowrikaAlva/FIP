"""
src/preprocess.py
-----------------
Load grayscale thermal images, resize them, apply noise reduction,
and optionally augment the dataset.

Public API
----------
load_grayscale(path)           → np.ndarray  (H×W, uint8)
preprocess(image)              → np.ndarray  (H×W, uint8)
preprocess_batch(paths)        → list[dict]
augment(image)                 → list[np.ndarray]
augment_batch(records)         → list[dict]
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


# ── Data augmentation ─────────────────────────────────────────────────────────

def augment(image: np.ndarray) -> list[np.ndarray]:
    """
    Return the original grayscale image plus 7 augmented variants.

    Augmentations applied
    ---------------------
    1. Original (unchanged)
    2. Horizontal flip
    3. Vertical flip
    4. 90° clockwise rotation
    5. 90° counter-clockwise rotation
    6. Brightness decrease  (× 0.85)
    7. Brightness increase  (× 1.15)
    8. Centre crop → resize back  (90 % of the image area)

    All variants share the same spatial dimensions as the input so they
    can be fed directly into the preprocessing pipeline.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image of shape (H, W), dtype uint8.

    Returns
    -------
    list[np.ndarray]
        List of 8 grayscale images (including the original).
    """
    h, w = image.shape[:2]
    variants: list[np.ndarray] = [image]

    # Flips
    variants.append(cv2.flip(image, 1))   # horizontal
    variants.append(cv2.flip(image, 0))   # vertical

    # Rotations
    variants.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    variants.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # Brightness jitter  (clip to valid uint8 range)
    for alpha in (0.85, 1.15):
        bright = np.clip(
            image.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)
        variants.append(bright)

    # Centre crop (90 %) resized back to original dimensions
    r0, r1 = int(h * 0.05), int(h * 0.95)
    c0, c1 = int(w * 0.05), int(w * 0.95)
    cropped = image[r0:r1, c0:c1]
    variants.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR))

    return variants   # length = 8


def augment_batch(records: list[dict]) -> list[dict]:
    """
    Expand a list of preprocessed records by augmenting every image.

    Each input record produces 8 output records (original + 7 variants).
    The augmented records inherit ``"label"``, ``"path"``, and ``"name"``
    from their source record; ``"name"`` gets a ``_augN`` suffix so
    downstream code can distinguish originals from synthetics.

    .. note::
        Augmentation operates on the ``"processed"`` grayscale image.
        The ``"raw"`` field of augmented records points to the same
        original raw image (it is not re-augmented).

    Parameters
    ----------
    records : list[dict]
        Output of ``preprocess_batch()`` — each dict must have at least
        ``"processed"`` (np.ndarray) and ``"name"`` (str) keys.

    Returns
    -------
    list[dict]
        Augmented records, each containing::

            {
                "path":      str,          # source image path
                "name":      str,          # stem + _aug<i> suffix
                "raw":       np.ndarray,   # original raw image
                "processed": np.ndarray,   # augmented grayscale
                "label":     str,          # inherited from source
            }
    """
    augmented: list[dict] = []

    for rec in records:
        variants = augment(rec["processed"])
        for i, variant in enumerate(variants):
            suffix = "" if i == 0 else f"_aug{i}"
            augmented.append({
                "path":      rec["path"],
                "name":      rec["name"] + suffix,
                "raw":       rec.get("raw"),
                "processed": variant,
                "label":     rec.get("label", "unknown"),
            })

    original_count  = len(records)
    augmented_count = len(augmented)
    logger.info(
        "Augmentation: %d original → %d total records  (×%.1f)",
        original_count,
        augmented_count,
        augmented_count / max(original_count, 1),
    )
    return augmented


# ── Save helper ───────────────────────────────────────────────────────────────

def save_processed(image: np.ndarray, name: str, out_dir: str = config.DATA_PROC_DIR) -> str:
    """Save a preprocessed grayscale image and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_processed.png")
    cv2.imwrite(out_path, image)
    return out_path