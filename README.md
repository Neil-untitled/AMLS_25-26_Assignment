# AMLS Assignment 2025-2026: BreastMNIST Classification

## Project Overview

This project benchmarks two machine learning models on the **BreastMNIST** dataset for binary classification of breast ultrasound images (Benign vs. Malignant):

- **Model A (Classical ML):** k-Nearest Neighbors (k-NN) with multiple feature extraction methods
- **Model B (Deep Learning):** ResNet-18 with transfer learning

The project analyzes how **model capacity**, **data augmentation**, and **training budget** influence performance across both approaches.

## Project Structure

```
AMLS_25_26_SN25048282/
├── Code/
│   ├── A/                      # k-NN Model (Model A)
│   │   ├── knn_model.py        # k-NN implementation and benchmarking
│   │   └── knn_best_model.pkl  # Saved best k-NN model (after training)
│   ├── B/                      # ResNet Model (Model B)
│   │   ├── resnet_model.py     # ResNet-18 implementation and benchmarking
│   │   └── resnet_best_model.pth # Saved best ResNet model (after training)
│   ├── data_loader.py          # Data loading and feature extraction
│   └── evaluation.py           # Evaluation metrics and visualization
├── Datasets/                   # Dataset folder (leave empty for submission)
│   └── BreastMNIST/
│       └── breastmnist.npz     # Place dataset here
├── results/                    # Generated results and plots (after running)
├── main.py                     # Main entry point
└── README.md                   # This file
```

## Requirements

### Python Version
- Python 3.8 or higher

### Required Packages

```
numpy>=1.21.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
torch>=1.12.0
torchvision>=0.13.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
```

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install PyTorch (choose the appropriate command for your system)
# For CPU only:
pip install torch torchvision

# For CUDA 11.8 (if you have an NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy scikit-learn scikit-image matplotlib seaborn joblib

# (Optional) Install medmnist for automatic dataset download
pip install medmnist
```

#### Quick Install (All Dependencies)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac (or venv\Scripts\activate on Windows)

# Install all dependencies at once (CPU version)
pip install numpy scikit-learn scikit-image torch torchvision matplotlib seaborn joblib medmnist
```

#### Verifying Installation

```python
# Run this in Python to verify all packages are installed correctly
import numpy as np
import sklearn
import skimage
import torch
import torchvision
import matplotlib
import seaborn
import joblib
print("All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Dataset Setup

1. Download the BreastMNIST dataset from [https://medmnist.com/](https://medmnist.com/)
2. Place the `breastmnist.npz` file in `Datasets/BreastMNIST/`

The dataset contains:
- 28×28 grayscale ultrasound images
- Binary labels: 0 (Benign), 1 (Malignant)
- Pre-defined train/validation/test splits

## Usage

### Run Complete Benchmark

```bash
python main.py
```

This will:
1. Load the BreastMNIST dataset
2. Run comprehensive benchmarks on both models
3. Generate evaluation metrics and visualizations
4. Save trained models and results

### Command Line Options

```bash
# Run only k-NN model
python main.py --model knn

# Run only ResNet model
python main.py --model resnet

# Quick mode (fewer experiments, for testing)
python main.py --quick

# Skip plot generation
python main.py --no-plots

# Specify output directory
python main.py --output-dir my_results
```

## Model Descriptions

### Model A: k-Nearest Neighbors (k-NN)

**Feature Extraction Methods:**
- **Raw pixels:** Flattened 28×28 images (784 features)
- **HOG:** Histogram of Oriented Gradients (324 features)
- **PCA:** Principal Component Analysis for dimensionality reduction
- **HOG + PCA:** Combined approach

**Benchmarking Experiments:**
1. **Model Capacity:** Varying k values (1, 3, 5, 7, 9, 11, 15, 21, 31, 51)
2. **Feature Comparison:** Raw vs HOG vs PCA vs HOG+PCA
3. **Training Budget:** Effect of training sample size
4. **Hyperparameter Search:** Grid search over k, distance metrics, weights

### Model B: ResNet-18

**Architecture:**
- Pre-trained ResNet-18 backbone (ImageNet weights)
- Modified final layer for binary classification
- Dropout (0.5) for regularization

**Data Augmentation Levels:**
- **None:** Only resize and normalize
- **Light:** Horizontal flip
- **Standard:** Flips, rotation (±15°), brightness/contrast
- **Heavy:** All above + affine transforms, Gaussian blur

**Benchmarking Experiments:**
1. **Model Capacity:** Frozen backbone vs gradual unfreezing vs full fine-tuning
2. **Augmentation Analysis:** Effect of augmentation level
3. **Training Budget:** Effect of epochs and sample size

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Entry point script that orchestrates the entire pipeline |
| `Code/data_loader.py` | `BreastMNISTLoader` class for loading data; `FeatureExtractor` class for HOG/PCA |
| `Code/A/knn_model.py` | `KNNClassifier` and `KNNBenchmark` classes for k-NN experiments |
| `Code/B/resnet_model.py` | `ResNetClassifier` and `ResNetBenchmark` classes for deep learning |
| `Code/evaluation.py` | Plotting functions and evaluation utilities |

## Output

After running, the following outputs are generated:

### Saved Models
- `Code/A/knn_best_model.pkl` - Best k-NN model
- `Code/B/resnet_best_model.pth` - Best ResNet model

### Results
- `results/benchmark_results.json` - Complete benchmark results
- `results/*.png` - Visualization plots

### Generated Plots
- Learning curves
- Confusion matrices
- k-value analysis
- Feature method comparison
- Augmentation comparison
- Training budget analysis
- Model comparison

## Evaluation Metrics

Both models are evaluated using:
- **Accuracy:** Overall correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed breakdown of predictions

## Notes

- The ResNet model requires a CUDA-compatible GPU for faster training (automatically falls back to CPU)
- First run may take longer due to downloading pretrained weights
- All experiments use a fixed random seed (42) for reproducibility

## Author

Student Number: 25048282
Course: ELEC0134 Applied Machine Learning Systems (2025-2026)
