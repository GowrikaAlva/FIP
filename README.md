# Pseudo-Color Mapping for Thermal Images

An academic project implementing pseudo-color mapping on grayscale thermal images,
feature extraction, and machine learning classification of temperature categories.

## Project Structure

```
pseudo_color_thermal/
├── data/
│   ├── raw/            # Original FLIR grayscale thermal images
│   ├── processed/      # Preprocessed images
│   └── samples/        # Sample images for quick testing
├── src/
│   ├── preprocess.py   # Image loading & preprocessing
│   ├── colormap.py     # Pseudo-color mapping
│   ├── features.py     # Feature extraction
│   ├── model.py        # ML model training & evaluation
│   ├── visualize.py    # Visualization utilities
│   └── utils.py        # Helper functions
├── outputs/
│   ├── images/         # Saved processed images
│   ├── plots/          # Saved plots & charts
│   └── metrics/        # Saved evaluation metrics (CSV/JSON)
├── models/             # Saved trained models
├── notebooks/          # Jupyter notebooks (optional)
├── logs/               # Runtime logs
├── main.py             # Entry point — runs the full pipeline
├── config.py           # All configurable parameters
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline
python main.py

# Run with custom config
python main.py --data_dir data/raw --colormap JET --model rf
```

## Dataset
Place FLIR thermal images (grayscale `.jpg` / `.png`) inside `data/raw/`.
Organize into sub-folders by temperature category if you want supervised classification:
```
data/raw/
├── low/
├── medium/
└── high/
```

## Outputs
| Output | Location |
|---|---|
| Pseudo-colored images | `outputs/images/` |
| Grayscale vs color comparison plots | `outputs/plots/` |
| Accuracy / classification report | `outputs/metrics/` |
| Confusion matrix plot | `outputs/plots/` |
| Trained model | `models/` |