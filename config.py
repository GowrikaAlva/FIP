"""
config.py
---------
Central configuration for the Pseudo-Color Mapping project.
Edit values here; all other modules import from this file.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "samples")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_IMG_DIR  = os.path.join(BASE_DIR, "outputs", "images")
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
OUTPUT_MET_DIR  = os.path.join(BASE_DIR, "outputs", "metrics")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# ── Image Preprocessing ────────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)          # (width, height) — resize target
NOISE_KERNEL    = 5                   # Gaussian blur kernel size (odd number)
SUPPORTED_EXTS  = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# ── Pseudo-Color Mapping ───────────────────────────────────────────────────────
# Available OpenCV colormaps:
#   AUTUMN BONE JET WINTER RAINBOW OCEAN SUMMER SPRING COOL HSV HOT PINK PARULA
COLORMAP_NAME   = "JET"               # Change to any key from COLORMAP_OPTIONS
COLORMAP_OPTIONS = {
    "JET":     "cv2.COLORMAP_JET",
    "HOT":     "cv2.COLORMAP_HOT",
    "RAINBOW":  "cv2.COLORMAP_RAINBOW",
    "PARULA":  "cv2.COLORMAP_PARULA",
    "INFERNO": "cv2.COLORMAP_INFERNO",
    "VIRIDIS": "cv2.COLORMAP_VIRIDIS",
    "MAGMA":   "cv2.COLORMAP_MAGMA",
    "PLASMA":  "cv2.COLORMAP_PLASMA",
    "BONE":    "cv2.COLORMAP_BONE",
    "OCEAN":   "cv2.COLORMAP_OCEAN",
}

# ── Temperature Labels ─────────────────────────────────────────────────────────
# Used when dataset images are NOT pre-labeled in sub-folders.
# Pixels are classified by grayscale intensity into these bins.
TEMP_BINS       = [0, 85, 170, 255]           # intensity thresholds
TEMP_LABELS     = ["low", "medium", "high"]   # must be len(TEMP_BINS) - 1

# ── Feature Extraction ─────────────────────────────────────────────────────────
HIST_BINS       = 32          # bins per channel for color histogram
FEATURE_CHANNELS = ["R", "G", "B"]

# ── Machine Learning ───────────────────────────────────────────────────────────
MODEL_TYPE      = "rf"        # "rf" = Random Forest | "svm" = SVM
TEST_SIZE       = 0.4         # 80/20 train-test split
RANDOM_STATE    = 42

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = None

# SVM hyperparameters
SVM_KERNEL      = "rbf"
SVM_C           = 1.0
SVM_GAMMA       = "scale"

# ── Output / Logging ──────────────────────────────────────────────────────────
SAVE_IMAGES     = True        # Save processed & pseudo-colored images
SAVE_PLOTS      = True        # Save comparison & metric plots
LOG_LEVEL       = "INFO"      # DEBUG | INFO | WARNING | ERROR