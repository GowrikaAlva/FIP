"""
src/utils.py
------------
Utility functions: logging setup, directory creation, file listing.
"""

import os
import logging
import sys
from datetime import datetime

import config


def setup_logging() -> logging.Logger:
    """Configure and return the project logger."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(config.LOGS_DIR, f"run_{timestamp}.log")

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("thermal")
    logger.info("Logger initialised — writing to %s", log_file)
    return logger


def ensure_directories() -> None:
    """Create all required output directories if they don't exist."""
    dirs = [
        config.DATA_PROC_DIR,
        config.OUTPUT_IMG_DIR,
        config.OUTPUT_PLOT_DIR,
        config.OUTPUT_MET_DIR,
        config.MODELS_DIR,
        config.LOGS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def list_images(directory: str) -> list[str]:
    """
    Recursively collect all image file paths under *directory*.

    Returns
    -------
    list[str]
        Absolute paths to every supported image file found.
    """
    paths = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.lower().endswith(config.SUPPORTED_EXTS):
                paths.append(os.path.join(root, fname))
    return paths


def label_from_path(image_path: str) -> str:
    """
    Infer the temperature label from the parent folder name.

    Expected layout::

        data/raw/low/img001.jpg
        data/raw/medium/img002.jpg
        data/raw/high/img003.jpg

    If the parent folder is not a known label, returns ``"unknown"``.
    """
    parent = os.path.basename(os.path.dirname(image_path)).lower()
    return parent if parent in config.TEMP_LABELS else "unknown"


def stem(path: str) -> str:
    """Return the filename without extension."""
    return os.path.splitext(os.path.basename(path))[0]