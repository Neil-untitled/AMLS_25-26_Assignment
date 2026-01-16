"""
Data Loading and Preprocessing Module for BreastMNIST Dataset

This module handles:
- Loading BreastMNIST dataset from .npz files
- Data preprocessing and normalization
- Feature extraction (raw, HOG, PCA)
- Data splitting utilities
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np  # NumPy for numerical array operations
import os  # OS module for file path operations
from typing import Tuple, Dict, Optional  # Type hints for better code documentation
from sklearn.preprocessing import StandardScaler  # For feature normalization (zero mean, unit variance)
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
from skimage.feature import hog  # Histogram of Oriented Gradients feature extractor


# ============================================================================
# DATASET LOADER CLASS
# ============================================================================

class BreastMNISTLoader:
    """
    Loader class for BreastMNIST dataset.

    Handles loading from .npz files and provides train/val/test splits
    as defined in the original MedMNIST benchmark.
    """

    def __init__(self, data_path: str = "Datasets/BreastMNIST"):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the BreastMNIST directory containing the .npz file
        """
        self.data_path = data_path  # Store path to dataset directory
        self.data = None  # Will hold the loaded numpy data
        self._load_data()  # Immediately load data upon initialization

    def _load_data(self) -> None:
        """Load the dataset from .npz file."""
        # Construct full path to the .npz file
        npz_path = os.path.join(self.data_path, "breastmnist.npz")

        # Check if dataset file exists before attempting to load
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Dataset not found at {npz_path}. "
                "Please download BreastMNIST from https://medmnist.com/"
            )

        # Load the compressed numpy file containing all data splits
        self.data = np.load(npz_path)
        print(f"Dataset loaded from {npz_path}")
        # Print available keys: typically train/val/test images and labels
        print(f"Available keys: {list(self.data.keys())}")

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data."""
        # Return training images and flattened labels (from shape (N,1) to (N,))
        return self.data['train_images'], self.data['train_labels'].flatten()

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data."""
        # Return validation images and flattened labels
        return self.data['val_images'], self.data['val_labels'].flatten()

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data."""
        # Return test images and flattened labels
        return self.data['test_images'], self.data['test_labels'].flatten()

    def get_all_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get all data splits."""
        # Return dictionary with all three splits for convenience
        return {
            'train': self.get_train_data(),
            'val': self.get_val_data(),
            'test': self.get_test_data()
        }

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        # Retrieve all data splits
        train_images, train_labels = self.get_train_data()
        val_images, val_labels = self.get_val_data()
        test_images, test_labels = self.get_test_data()

        # Return comprehensive dataset statistics
        return {
            'train_size': len(train_images),  # Number of training samples
            'val_size': len(val_images),  # Number of validation samples
            'test_size': len(test_images),  # Number of test samples
            'image_shape': train_images.shape[1:],  # Image dimensions (28, 28)
            'num_classes': len(np.unique(train_labels)),  # Number of unique classes (2 for binary)
            'class_distribution': {
                # Count samples per class for each split using np.unique
                'train': dict(zip(*np.unique(train_labels, return_counts=True))),
                'val': dict(zip(*np.unique(val_labels, return_counts=True))),
                'test': dict(zip(*np.unique(test_labels, return_counts=True)))
            }
        }


# ============================================================================
# FEATURE EXTRACTOR CLASS
# ============================================================================

class FeatureExtractor:
    """
    Feature extraction class for classical ML models.

    Supports:
    - Raw pixel features (flattened)
    - HOG (Histogram of Oriented Gradients) features
    - PCA dimensionality reduction
    """

    def __init__(self, method: str = 'raw', n_components: Optional[int] = None):
        """
        Initialize feature extractor.

        Args:
            method: Feature extraction method ('raw', 'hog', 'pca', 'hog_pca')
            n_components: Number of PCA components (required for 'pca' and 'hog_pca')
        """
        self.method = method  # Store the feature extraction method
        self.n_components = n_components  # Number of PCA components (if applicable)
        self.scaler = StandardScaler()  # Scaler for normalizing features
        self.pca = None  # PCA transformer (initialized during fit if needed)
        self._fitted = False  # Flag to track if extractor has been fitted

        # Set default PCA components if not specified for PCA methods
        if method in ['pca', 'hog_pca'] and n_components is None:
            self.n_components = 50  # Default to 50 PCA components

    def _extract_hog_single(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from a single image."""
        # HOG: Histogram of Oriented Gradients
        # Captures edge orientations and their distribution
        features = hog(
            image,
            orientations=9,  # Number of gradient orientation bins
            pixels_per_cell=(4, 4),  # Size of each cell in pixels
            cells_per_block=(2, 2),  # Number of cells per block for normalization
            visualize=False,  # Don't return visualization image
            feature_vector=True  # Return features as 1D array
        )
        return features

    def _extract_hog_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract HOG features from a batch of images."""
        hog_features = []  # List to collect features from all images
        # Process each image individually
        for img in images:
            features = self._extract_hog_single(img)
            hog_features.append(features)
        # Convert list to numpy array for efficient operations
        return np.array(hog_features)

    def fit(self, images: np.ndarray) -> 'FeatureExtractor':
        """Fit the feature extractor on training data."""
        # Different fitting procedures based on method
        if self.method == 'raw':
            # Raw: flatten images and fit scaler
            flat_images = images.reshape(len(images), -1)  # Flatten to (N, 784)
            self.scaler.fit(flat_images)  # Compute mean and std for normalization

        elif self.method == 'hog':
            # HOG: extract features then fit scaler
            hog_features = self._extract_hog_batch(images)
            self.scaler.fit(hog_features)

        elif self.method == 'pca':
            # PCA: flatten, scale, then fit PCA
            flat_images = images.reshape(len(images), -1)
            self.scaler.fit(flat_images)
            scaled_images = self.scaler.transform(flat_images)
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(scaled_images)  # Learn principal components

        elif self.method == 'hog_pca':
            # HOG+PCA: extract HOG, scale, then fit PCA
            hog_features = self._extract_hog_batch(images)
            self.scaler.fit(hog_features)
            scaled_features = self.scaler.transform(hog_features)
            # Ensure n_components doesn't exceed feature dimensions
            self.pca = PCA(n_components=min(self.n_components, scaled_features.shape[1]))
            self.pca.fit(scaled_features)

        self._fitted = True  # Mark as fitted
        return self  # Return self for method chaining

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Transform images to features."""
        # Ensure fit() was called before transform()
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform")

        if self.method == 'raw':
            # Raw: flatten and scale
            flat_images = images.reshape(len(images), -1)
            return self.scaler.transform(flat_images)

        elif self.method == 'hog':
            # HOG: extract features and scale
            hog_features = self._extract_hog_batch(images)
            return self.scaler.transform(hog_features)

        elif self.method == 'pca':
            # PCA: flatten, scale, then project to principal components
            flat_images = images.reshape(len(images), -1)
            scaled_images = self.scaler.transform(flat_images)
            return self.pca.transform(scaled_images)

        elif self.method == 'hog_pca':
            # HOG+PCA: extract HOG, scale, then project
            hog_features = self._extract_hog_batch(images)
            scaled_features = self.scaler.transform(hog_features)
            return self.pca.transform(scaled_features)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(images)  # Fit on the data
        return self.transform(images)  # Then transform

    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        # Ensure extractor is fitted before querying dimensions
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted first")

        if self.method == 'raw':
            return 784  # 28 * 28 pixels flattened
        elif self.method == 'hog':
            return 324  # HOG feature dimension (depends on parameters above)
        elif self.method in ['pca', 'hog_pca']:
            return self.pca.n_components_  # Actual number of components used

    def get_explained_variance(self) -> Optional[float]:
        """Get total explained variance ratio (for PCA methods)."""
        # Only applicable for PCA-based methods
        if self.pca is not None:
            # Sum of variance explained by all retained components
            return np.sum(self.pca.explained_variance_ratio_)
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images to [0, 1] range."""
    # Convert to float32 and scale from [0, 255] to [0, 1]
    return images.astype(np.float32) / 255.0


def get_sample_subset(images: np.ndarray, labels: np.ndarray,
                      n_samples: int, stratified: bool = True,
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a subset of samples from the dataset.

    Args:
        images: Full image array
        labels: Full label array
        n_samples: Number of samples to select
        stratified: Whether to maintain class distribution
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (subset_images, subset_labels)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # If requesting more samples than available, return all
    if n_samples >= len(images):
        return images, labels

    if stratified:
        # Stratified sampling: maintain class proportions
        unique_labels = np.unique(labels)  # Get all unique class labels
        indices = []
        # Calculate samples per class (equal distribution)
        samples_per_class = n_samples // len(unique_labels)

        # Sample from each class
        for label in unique_labels:
            # Find indices belonging to this class
            label_indices = np.where(labels == label)[0]
            # Don't sample more than available
            n_select = min(samples_per_class, len(label_indices))
            # Randomly select indices without replacement
            selected = np.random.choice(label_indices, n_select, replace=False)
            indices.extend(selected)

        # Handle remainder (if n_samples not divisible by num_classes)
        remaining = n_samples - len(indices)
        if remaining > 0:
            # Get indices not yet selected
            all_indices = set(range(len(images)))
            available = list(all_indices - set(indices))
            # Randomly select remaining samples
            extra = np.random.choice(available, remaining, replace=False)
            indices.extend(extra)

        indices = np.array(indices)
    else:
        # Non-stratified: simple random sampling
        indices = np.random.choice(len(images), n_samples, replace=False)

    # Return subset of images and labels at selected indices
    return images[indices], labels[indices]
