"""
src/features.py
---------------
Extract meaningful numerical features from pseudo-colored thermal images.

Features extracted per image
-----------------------------
1. Color histogram  — ``HIST_BINS`` bins per R, G, B channel (normalised).
2. Channel mean     — mean pixel intensity for R, G, B.
3. Channel std      — standard deviation of pixel intensity for R, G, B.
4. HSV histogram    — ``HIST_BINS`` bins per H, S, V channel (normalised).
5. Spatial grid     — mean intensity per cell of a 4×4 grid (16 values).

The feature vector length = HIST_BINS * 3 + 6 + HIST_BINS * 3 + 16.

Public API
----------
extract(image)          → np.ndarray  (1-D feature vector)
extract_batch(records)  → np.ndarray  (N × F feature matrix), list[str] labels
"""

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger("thermal.features")

_SPATIAL_GRID = 4   # 4×4 = 16 cells


# ── Single-image extraction ───────────────────────────────────────────────────

def color_histogram(image: np.ndarray, bins: int = config.HIST_BINS) -> np.ndarray:
    """
    Compute a normalised colour histogram for each RGB channel.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    bins : int
        Number of histogram bins per channel.

    Returns
    -------
    np.ndarray
        Concatenated histogram of shape (bins * 3,), values in [0, 1].
    """
    histograms = []
    for ch in range(3):                       # R=0, G=1, B=2
        hist, _ = np.histogram(
            image[:, :, ch],
            bins=bins,
            range=(0, 256),
        )
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7            # normalise → probabilities
        histograms.append(hist)
    return np.concatenate(histograms)         # shape: (bins * 3,)


def channel_stats(image: np.ndarray) -> np.ndarray:
    """
    Compute per-channel mean and standard deviation.

    Returns
    -------
    np.ndarray
        Array of shape (6,): [mean_R, mean_G, mean_B, std_R, std_G, std_B].
        Values normalised to [0, 1] by dividing by 255.
    """
    means = image.mean(axis=(0, 1)) / 255.0    # shape (3,)
    stds  = image.std(axis=(0, 1))  / 255.0    # shape (3,)
    return np.concatenate([means, stds])        # shape (6,)


def hsv_histogram(image: np.ndarray, bins: int = config.HIST_BINS) -> np.ndarray:
    """
    Compute a normalised histogram for each HSV channel.

    In JET pseudo-colour thermal images, hue directly encodes temperature
    (blue ≈ cold, red ≈ hot), making HSV far more discriminative than RGB
    for temperature classification.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    bins : int
        Number of histogram bins per channel.

    Returns
    -------
    np.ndarray
        Concatenated histogram of shape (bins * 3,), values in [0, 1].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    histograms = []
    for ch in range(3):                        # H=0, S=1, V=2
        hist, _ = np.histogram(
            hsv[:, :, ch],
            bins=bins,
            range=(0, 256),
        )
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7
        histograms.append(hist)
    return np.concatenate(histograms)          # shape: (bins * 3,)


def spatial_grid_means(
    image: np.ndarray,
    grid: int = _SPATIAL_GRID,
) -> np.ndarray:
    """
    Divide the grayscale image into ``grid × grid`` non-overlapping cells
    and return the normalised mean intensity of each cell.

    Captures *where* hot/cold regions are located in the frame — information
    that global histograms discard entirely.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    grid : int
        Number of rows and columns to divide the image into.

    Returns
    -------
    np.ndarray
        Array of shape (grid * grid,), values in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ch, cw = h // grid, w // grid

    features = []
    for r in range(grid):
        for c in range(grid):
            cell = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            features.append(float(cell.mean()) / 255.0)

    return np.array(features, dtype=np.float32)  # shape: (grid*grid,)


def extract(image: np.ndarray) -> np.ndarray:
    """
    Extract the full feature vector from a single pseudo-colored image.

    Feature vector layout::

        [ hist_R  (HIST_BINS) | hist_G  (HIST_BINS) | hist_B  (HIST_BINS) |
          mean_R | mean_G | mean_B | std_R | std_G | std_B               |
          hist_H  (HIST_BINS) | hist_S  (HIST_BINS) | hist_V  (HIST_BINS) |
          grid_cell_00 … grid_cell_NN  (grid*grid values)                ]

    Total length = HIST_BINS * 6 + 6 + _SPATIAL_GRID^2.

    Parameters
    ----------
    image : np.ndarray
        RGB pseudo-colored image, shape (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        1-D feature vector.
    """
    rgb_hist  = color_histogram(image)
    stats     = channel_stats(image)
    hsv_hist  = hsv_histogram(image)
    grid      = spatial_grid_means(image)
    return np.concatenate([rgb_hist, stats, hsv_hist, grid])


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_batch(
    records: list[dict],
    label_key: str = "label",
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features from all records.

    Parameters
    ----------
    records : list[dict]
        Each record must contain ``"colored"`` (RGB np.ndarray) and,
        optionally, a label under *label_key*.
    label_key : str
        Key in each record that holds the class label string.

    Returns
    -------
    X : np.ndarray  shape (N, F)
        Feature matrix.
    y : list[str]   length N
        Corresponding labels ("unknown" if the key is missing).
    """
    X, y = [], []
    for rec in records:
        feat  = extract(rec["colored"])
        label = rec.get(label_key, "unknown")
        X.append(feat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    logger.info(
        "Feature matrix: shape=%s  unique_labels=%s",
        X.shape,
        sorted(set(y)),
    )
    return X, y


# ── Auto-labelling (no folder labels) ────────────────────────────────────────

def auto_label_from_intensity(gray: np.ndarray) -> str:
    """
    Assign a temperature label based on the mean grayscale intensity.

    Thresholds are defined in ``config.TEMP_BINS``.  This is used when
    images are NOT organised into per-class sub-folders.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (H, W), uint8.

    Returns
    -------
    str
        One of config.TEMP_LABELS.
    """
    mean_val = gray.mean()
    for i, upper in enumerate(config.TEMP_BINS[1:]):
        if mean_val <= upper:
            return config.TEMP_LABELS[i]
    return config.TEMP_LABELS[-1]


def feature_names() -> list[str]:
    """Return human-readable names for every feature dimension."""
    bins  = config.HIST_BINS
    names = []

    # RGB histograms
    for ch in config.FEATURE_CHANNELS:
        names += [f"hist_{ch}_bin{b:02d}" for b in range(bins)]

    # Channel stats
    names += [f"mean_{ch}" for ch in config.FEATURE_CHANNELS]
    names += [f"std_{ch}"  for ch in config.FEATURE_CHANNELS]

    # HSV histograms
    for ch in ["H", "S", "V"]:
        names += [f"hsv_{ch}_bin{b:02d}" for b in range(bins)]

    # Spatial grid
    for r in range(_SPATIAL_GRID):
        for c in range(_SPATIAL_GRID):
            names.append(f"grid_{r}{c}_mean")

    return names