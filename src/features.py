"""
src/features.py
---------------
Extract meaningful numerical features from pseudo-colored thermal images.

Features extracted per image
-----------------------------
1. Color histogram  — ``HIST_BINS`` bins per R, G, B channel (normalised).
2. Channel mean     — mean pixel intensity for R, G, B.
3. Channel std      — standard deviation of pixel intensity for R, G, B.

The feature vector length = HIST_BINS * 3  +  3  +  3.

Public API
----------
extract(image)          → np.ndarray  (1-D feature vector)
extract_batch(records)  → np.ndarray  (N × F feature matrix), list[str] labels
"""

import logging

import numpy as np

import config

logger = logging.getLogger("thermal.features")


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


def extract(image: np.ndarray) -> np.ndarray:
    """
    Extract the full feature vector from a single pseudo-colored image.

    Feature vector layout::

        [ hist_R (32 bins) | hist_G (32 bins) | hist_B (32 bins) |
          mean_R | mean_G | mean_B |
          std_R  | std_G  | std_B  ]

    Total length = HIST_BINS * 3 + 6.

    Parameters
    ----------
    image : np.ndarray
        RGB pseudo-colored image, shape (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        1-D feature vector.
    """
    hist  = color_histogram(image)
    stats = channel_stats(image)
    return np.concatenate([hist, stats])


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
    for ch in config.FEATURE_CHANNELS:
        names += [f"hist_{ch}_bin{b:02d}" for b in range(bins)]
    names += [f"mean_{ch}" for ch in config.FEATURE_CHANNELS]
    names += [f"std_{ch}"  for ch in config.FEATURE_CHANNELS]
    return names