"""
Model B: ResNet-18 Implementation

This module contains the ResNet-18 classifier with:
- Transfer learning from ImageNet
- Data augmentation for medical images
- Comprehensive benchmarking suite
"""

from .resnet_model import ResNetClassifier, ResNetBenchmark, run_full_benchmark
