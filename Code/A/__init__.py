"""
Model A: k-Nearest Neighbors (k-NN) Implementation

This module contains the k-NN classifier with:
- Multiple feature extraction methods (raw, HOG, PCA)
- Comprehensive benchmarking suite
- Hyperparameter optimization
"""

from .knn_model import KNNClassifier, KNNBenchmark, run_full_benchmark
