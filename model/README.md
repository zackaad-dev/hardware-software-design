# Face Recognition Model with MobileNetV2

This project implements a face recognition model using transfer learning with **MobileNetV2** and **TensorFlow/Keras**. It includes automated hyperparameter tuning and comprehensive evaluation metrics.

## Features
- **Transfer Learning:** Uses pre-trained ImageNet weights from MobileNetV2.
- **Data Augmentation:** Real-time image augmentation (rotation, shifts, zooms, flips, etc.) to improve generalization.
- **Hyperparameter Tuning:** Bayesian Optimization for tuning learning rate, dropout, and model architecture via `keras-tuner`.
- **Evaluation Metrics:** Automated generation of a confusion matrix and precision-recall curve.
- **Modern Python Tooling:** Uses `uv` for lightning-fast dependency management and reproducible environments.

## Prerequisites
- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** installed on your system.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Synchronize dependencies:**
    ```bash
    uv sync
    ```

## Data Preparation

The model dynamically detects classes based on the folder structure in your `DATA_DIR` (configurable in `main.py`). Organize your data as follows:

```text
DATA_DIR/
├── person_a/             # Training images for Person A
├── person_a-validation/  # Validation images for Person A
├── person_b/             # Training images for Person B
├── person_b-validation/  # Validation images for Person B
└── other/                # Negative samples or other class
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`.

## Usage

### Training and Evaluation
To train the model and generate evaluation metrics:
```bash
uv run python main.py
```
This script will:
1. Load and preprocess images.
2. Build and train the MobileNetV2-based model.
3. Save the best model to `model.keras`.
4. Generate `confusion_matrix.png` and `precision_recall_curve.png`.

### Hyperparameter Tuning
To search for the best model configuration:
```bash
uv run python tune.py
```
The tuner will explore different learning rates, dropout rates, and dense layer sizes to find the most accurate configuration.

## Configuration
Modify the `Config` dataclass in `main.py` to change:
- `DATA_DIR`: Path to your dataset.
- `IMG_SIZE`: Target image size (default 224x224).
- `BATCH_SIZE`: Training batch size.
- `EPOCHS`: Maximum number of training epochs.
- `PATIENCE`: Early stopping patience.

## Results
After a successful run, you will find:
- `model.keras`: The saved Keras model.
- `confusion_matrix.png`: Heatmap showing true vs. predicted classes.
- `precision_recall_curve.png`: PR curves for each class with Average Precision (AP) scores.
