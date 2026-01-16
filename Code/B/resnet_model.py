"""
ResNet-18 Model for BreastMNIST Classification

This module implements:
- ResNet-18 with transfer learning (pretrained on ImageNet)
- Custom data augmentation for medical images
- Training with learning rate scheduling
- Model capacity analysis (frozen vs unfrozen layers)
- Training budget analysis (epochs, samples)
"""

import numpy as np  # Numerical arrays and random ops.
import torch  # Core PyTorch.
import torch.nn as nn  # Neural network layers.
import torch.optim as optim  # Optimizers.
from torch.utils.data import Dataset, DataLoader  # Dataset and batching.
from torchvision import models, transforms  # Pretrained models and transforms.
from torchvision.transforms import v2  # Torchvision v2 transforms (unused here).
import time  # Timing for benchmarks.
import gc  # Garbage collection to free memory between runs.
from typing import Dict, List, Tuple, Optional  # Type hints.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Metrics.
from sklearn.metrics import confusion_matrix, classification_report  # Reports.
import os  # Path utilities.
import sys  # System path (unused here).

# Set device once to avoid repeating checks.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU if available.


class BreastMNISTDataset(Dataset):
    """
    PyTorch Dataset for BreastMNIST.

    Handles image loading and augmentation for training ResNet.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform=None, augment: bool = False):
        """
        Initialize dataset.

        Args:
            images: Numpy array of images (N, 28, 28)
            labels: Numpy array of labels (N,)
            transform: Torchvision transforms to apply
            augment: Whether to apply data augmentation
        """
        self.images = images  # Image array (N, H, W).
        self.labels = labels  # Label array (N,).
        self.transform = transform  # Transform pipeline.
        self.augment = augment  # Flag (kept for API compatibility).

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]  # Single image.
        label = self.labels[idx]  # Corresponding label.

        # Convert to PIL-like format (H, W) -> (H, W, C) for transforms
        # For grayscale, repeat to 3 channels for ResNet
        image = np.stack([image, image, image], axis=-1)  # Repeat channels to RGB.

        # Convert to float tensor and scale to [0, 1] for torchvision transforms.
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # CHW float.

        if self.transform:
            image = self.transform(image)  # Apply resize/augment/normalize.

        return image, int(label)  # Return tensor and label.


def get_augmentation_transforms(augment_level: str = 'standard') -> transforms.Compose:
    """
    Get data augmentation transforms for medical images.

    Args:
        augment_level: Level of augmentation ('none', 'light', 'standard', 'heavy')

    Returns:
        Composed transforms
    """
    # Base transforms - resize to 224x224 for ResNet input.
    base_transforms = [
        transforms.Resize((224, 224), antialias=True),  # Match ResNet input size.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # ImageNet normalization.
    ]

    if augment_level == 'none':
        return transforms.Compose(base_transforms)  # No augmentation.

    elif augment_level == 'light':
        aug_transforms = [
            transforms.Resize((224, 224), antialias=True),  # Resize.
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Normalize.
        ]
        return transforms.Compose(aug_transforms)  # Compose transforms.

    elif augment_level == 'standard':
        # Standard medical image augmentation.
        aug_transforms = [
            transforms.Resize((224, 224), antialias=True),  # Resize.
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip.
            transforms.RandomVerticalFlip(p=0.5),  # Vertical flip.
            transforms.RandomRotation(degrees=15),  # Small rotation.
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Contrast/brightness.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Normalize.
        ]
        return transforms.Compose(aug_transforms)  # Compose transforms.

    elif augment_level == 'heavy':
        # Heavy augmentation with more variations.
        aug_transforms = [
            transforms.Resize((224, 224), antialias=True),  # Resize.
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip.
            transforms.RandomVerticalFlip(p=0.5),  # Vertical flip.
            transforms.RandomRotation(degrees=30),  # Larger rotation.
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Affine.
            transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Stronger jitter.
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Blur.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Normalize.
        ]
        return transforms.Compose(aug_transforms)  # Compose transforms.

    return transforms.Compose(base_transforms)  # Fallback to base transforms.


def get_val_transforms() -> transforms.Compose:
    """Get transforms for validation/test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),  # Resize.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # Normalize.
    ])


class ResNetClassifier:
    """
    ResNet-18 classifier with transfer learning for BreastMNIST.

    Supports:
    - Full fine-tuning
    - Feature extraction (frozen backbone)
    - Gradual unfreezing
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Initialize ResNet classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        self.num_classes = num_classes  # Output classes.
        self.pretrained = pretrained  # Whether to use ImageNet weights.
        self.freeze_backbone = freeze_backbone  # Whether to freeze feature extractor.

        # Load pretrained ResNet-18 (ImageNet) unless training from scratch.
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load weights.
        else:
            self.model = models.resnet18(weights=None)  # Random init.

        # Replace final FC layer for binary classification.
        num_features = self.model.fc.in_features  # Input size to FC.
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Regularization.
            nn.Linear(num_features, num_classes)  # Output layer.
        )

        # Freeze backbone if requested to run feature extraction.
        if freeze_backbone:
            self._freeze_backbone()  # Freeze feature extractor.

        self.model = self.model.to(device)  # Move model to device.
        self.training_history = {'train_loss': [], 'train_acc': [],
                                  'val_loss': [], 'val_acc': []}  # Track training curves.

    def _freeze_backbone(self) -> None:
        """Freeze all layers except the final FC layer."""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False  # Freeze non-FC layers.

    def _unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True  # Unfreeze all layers.

    def _unfreeze_layers(self, num_layers: int) -> None:
        """
        Gradually unfreeze layers from the end.

        Args:
            num_layers: Number of layer groups to unfreeze
        """
        # ResNet-18 layer groups: conv1, bn1, layer1, layer2, layer3, layer4, fc.
        layer_groups = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']  # Unfreeze order.

        # First freeze all.
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze all layers first.

        # Always unfreeze FC for classification.
        for param in self.model.fc.parameters():
            param.requires_grad = True  # Always train FC.

        # Unfreeze specified number of layer groups from the end.
        for i in range(min(num_layers, len(layer_groups))):
            layer_name = layer_groups[i]  # Layer group to unfreeze.
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True  # Unfreeze matching params.

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                    epochs: int = 20, learning_rate: float = 0.001,
                    weight_decay: float = 1e-4, patience: int = 5,
                    scheduler_type: str = 'plateau') -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            patience: Early stopping patience
            scheduler_type: Learning rate scheduler type

        Returns:
            Training history dictionary
        """
        criterion = nn.CrossEntropyLoss()  # Classification loss.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=learning_rate, weight_decay=weight_decay)  # Trainable params only.

        # Learning rate scheduler options for stability.
        if scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )  # Reduce on plateau.
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )  # Cosine decay.
        else:
            scheduler = None  # No scheduler.

        best_val_loss = float('inf')  # Track best validation loss.
        best_model_state = None  # Cache best weights.
        patience_counter = 0  # Early stopping counter.

        print(f"\nTraining on {device}")  # Device info.
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")  # Params.

        for epoch in range(epochs):  # Epoch loop.
            # Training phase.
            self.model.train()  # Enable training mode.
            train_loss = 0.0  # Accumulate loss.
            train_correct = 0  # Correct predictions.
            train_total = 0  # Total samples.

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)  # Move batch to device.

                optimizer.zero_grad()  # Reset gradients.
                outputs = self.model(images)  # Forward pass.
                loss = criterion(outputs, labels)  # Compute loss.
                loss.backward()  # Backprop.
                optimizer.step()  # Update weights.

                train_loss += loss.item()  # Accumulate loss.
                _, predicted = outputs.max(1)  # Predicted classes.
                train_total += labels.size(0)  # Count samples.
                train_correct += predicted.eq(labels).sum().item()  # Count correct.

            train_loss /= len(train_loader)  # Average loss.
            train_acc = train_correct / train_total  # Accuracy.

            # Validation phase.
            val_loss, val_acc = self._evaluate_loader(val_loader, criterion)  # Validate.

            # Update history.
            self.training_history['train_loss'].append(train_loss)  # Log train loss.
            self.training_history['train_acc'].append(train_acc)  # Log train acc.
            self.training_history['val_loss'].append(val_loss)  # Log val loss.
            self.training_history['val_acc'].append(val_acc)  # Log val acc.

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")  # Progress log.

            # Learning rate scheduling.
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)  # Step on validation loss.
            elif scheduler_type == 'cosine':
                scheduler.step()  # Step each epoch.

            # Early stopping check.
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Update best loss.
                best_model_state = self.model.state_dict().copy()  # Save weights.
                patience_counter = 0  # Reset patience.
            else:
                patience_counter += 1  # Increase patience.
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")  # Log.
                    break  # Stop training.

        # Restore best model weights seen during training.
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)  # Restore best weights.

        return self.training_history  # Return logged history.

    def _evaluate_loader(self, data_loader: DataLoader,
                         criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()  # Eval mode.
        total_loss = 0.0  # Accumulate loss.
        correct = 0  # Correct predictions.
        total = 0  # Total samples.

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)  # Move batch.
                outputs = self.model(images)  # Forward pass.
                loss = criterion(outputs, labels)  # Loss.

                total_loss += loss.item()  # Accumulate loss.
                _, predicted = outputs.max(1)  # Predicted class.
                total += labels.size(0)  # Count samples.
                correct += predicted.eq(labels).sum().item()  # Count correct.

        return total_loss / len(data_loader), correct / total  # Avg loss and acc.

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for a data loader.

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()  # Eval mode.
        all_preds = []  # Collect predictions.
        all_probs = []  # Collect probabilities.

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)  # Move batch.
                outputs = self.model(images)  # Forward pass.
                probs = torch.softmax(outputs, dim=1)  # Probabilities.

                _, predicted = outputs.max(1)  # Predicted class.
                all_preds.extend(predicted.cpu().numpy())  # Store preds.
                all_probs.extend(probs.cpu().numpy())  # Store probs.

        return np.array(all_preds), np.array(all_probs)  # Convert to arrays.

    def evaluate(self, data_loader: DataLoader, labels: np.ndarray) -> Dict:
        """
        Comprehensive evaluation.

        Args:
            data_loader: Data loader for evaluation
            labels: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        predictions, probabilities = self.predict(data_loader)  # Get preds and probs.

        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions),
            'classification_report': classification_report(labels, predictions, output_dict=True)
        }

    def save_model(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'training_history': self.training_history
        }, filepath)  # Serialize model checkpoint.
        print(f"Model saved to {filepath}")  # User feedback.

    def load_model(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)  # Load checkpoint.
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Restore weights.
        self.training_history = checkpoint.get('training_history', {})  # Restore history.
        print(f"Model loaded from {filepath}")  # User feedback.


class ResNetBenchmark:
    """
    Comprehensive benchmarking suite for ResNet classifier.

    Performs experiments on:
    - Model capacity (frozen vs fine-tuned)
    - Data augmentation levels
    - Training budget (epochs, samples)
    """

    def __init__(self, random_state: int = 42):
        """Initialize benchmark suite."""
        self.random_state = random_state  # Seed for reproducibility.
        self.results = {}  # Storage for benchmark results.
        torch.manual_seed(random_state)  # Torch seed.
        np.random.seed(random_state)  # Numpy seed.

    def benchmark_model_capacity(self, train_images: np.ndarray, train_labels: np.ndarray,
                                  val_images: np.ndarray, val_labels: np.ndarray,
                                  test_images: np.ndarray, test_labels: np.ndarray,
                                  epochs: int = 15, batch_size: int = 8) -> Dict:
        """
        Benchmark different model capacity configurations.

        Tests:
        - Frozen backbone (feature extraction)
        - Full fine-tuning
        - Gradual unfreezing
        """
        results = {'config': [], 'trainable_params': [], 'train_acc': [],
                   'val_acc': [], 'test_acc': [], 'test_f1': [], 'training_time': []}  # Accumulators.

        print("\n" + "="*60)  # Section divider.
        print("MODEL CAPACITY ANALYSIS")  # Section title.
        print("="*60)  # Divider.

        configs = [
            ('frozen_backbone', True, 0),
            ('unfreeze_layer4', True, 1),
            ('unfreeze_layer3_4', True, 2),
            ('full_finetune', False, None),
            ('from_scratch', None, None)
        ]

        val_transform = get_val_transforms()  # No augmentation for val/test.
        train_transform = get_augmentation_transforms('standard')  # Standard augmentation.

        for config_name, freeze, unfreeze_layers in configs:  # Iterate configs.
            print(f"\n--- Configuration: {config_name} ---")

            # Create model configuration per experiment.
            if config_name == 'from_scratch':
                model = ResNetClassifier(num_classes=2, pretrained=False, freeze_backbone=False)  # No pretrain.
            else:
                model = ResNetClassifier(num_classes=2, pretrained=True, freeze_backbone=freeze)  # Pretrained.
                if unfreeze_layers:
                    model._unfreeze_layers(unfreeze_layers)  # Gradual unfreeze.

            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)  # Count.
            print(f"Trainable parameters: {trainable_params:,}")  # Log.

            # Create data loaders.
            train_dataset = BreastMNISTDataset(train_images, train_labels, transform=train_transform)  # Train set.
            val_dataset = BreastMNISTDataset(val_images, val_labels, transform=val_transform)  # Val set.
            test_dataset = BreastMNISTDataset(test_images, test_labels, transform=val_transform)  # Test set.

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Train.
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Val.
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test.

            # Train.
            start_time = time.time()  # Start timer.
            history = model.train_model(train_loader, val_loader, epochs=epochs, patience=5)  # Train.
            training_time = time.time() - start_time  # Elapsed time.

            # Evaluate.
            test_metrics = model.evaluate(test_loader, test_labels)  # Test metrics.

            results['config'].append(config_name)  # Store config.
            results['trainable_params'].append(trainable_params)  # Store params.
            results['train_acc'].append(history['train_acc'][-1])  # Store train acc.
            results['val_acc'].append(history['val_acc'][-1])  # Store val acc.
            results['test_acc'].append(test_metrics['accuracy'])  # Store test acc.
            results['test_f1'].append(test_metrics['f1_score'])  # Store test F1.
            results['training_time'].append(training_time)  # Store time.

            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")  # Log.

            # Release memory before next configuration.
            del model, train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.results['model_capacity'] = results  # Cache results.
        return results  # Return to caller.

    def benchmark_augmentation(self, train_images: np.ndarray, train_labels: np.ndarray,
                                val_images: np.ndarray, val_labels: np.ndarray,
                                test_images: np.ndarray, test_labels: np.ndarray,
                                epochs: int = 15, batch_size: int = 8) -> Dict:
        """
        Benchmark different augmentation levels.
        """
        results = {'augment_level': [], 'train_acc': [], 'val_acc': [],
                   'test_acc': [], 'test_f1': [], 'best_epoch': []}  # Accumulators.

        print("\n" + "="*60)  # Section divider.
        print("DATA AUGMENTATION ANALYSIS")  # Section title.
        print("="*60)  # Divider.

        augment_levels = ['none', 'light', 'standard', 'heavy']  # Levels to test.
        val_transform = get_val_transforms()  # Fixed val/test transform.

        for aug_level in augment_levels:  # Iterate augmentation levels.
            print(f"\n--- Augmentation Level: {aug_level} ---")

            train_transform = get_augmentation_transforms(aug_level)  # Choose transform.

            # Create model with full fine-tuning.
            model = ResNetClassifier(num_classes=2, pretrained=True, freeze_backbone=False)  # Full fine-tune.

            # Create data loaders.
            train_dataset = BreastMNISTDataset(train_images, train_labels, transform=train_transform)  # Train.
            val_dataset = BreastMNISTDataset(val_images, val_labels, transform=val_transform)  # Val.
            test_dataset = BreastMNISTDataset(test_images, test_labels, transform=val_transform)  # Test.

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Train.
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Val.
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test.

            # Train.
            history = model.train_model(train_loader, val_loader, epochs=epochs, patience=7)  # Train.

            # Find best epoch based on validation accuracy.
            best_epoch = np.argmax(history['val_acc']) + 1  # Best val epoch (1-based).

            # Evaluate.
            test_metrics = model.evaluate(test_loader, test_labels)  # Test metrics.

            results['augment_level'].append(aug_level)  # Store level.
            results['train_acc'].append(max(history['train_acc']))  # Store best train acc.
            results['val_acc'].append(max(history['val_acc']))  # Store best val acc.
            results['test_acc'].append(test_metrics['accuracy'])  # Store test acc.
            results['test_f1'].append(test_metrics['f1_score'])  # Store test F1.
            results['best_epoch'].append(best_epoch)  # Store best epoch.

            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")  # Log.
            print(f"Best epoch: {best_epoch}")  # Log.

            # Release memory before next augmentation level.
            del model, train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.results['augmentation'] = results  # Cache results.
        return results  # Return to caller.

    def benchmark_training_budget(self, train_images: np.ndarray, train_labels: np.ndarray,
                                   val_images: np.ndarray, val_labels: np.ndarray,
                                   test_images: np.ndarray, test_labels: np.ndarray,
                                   batch_size: int = 8) -> Dict:
        """
        Benchmark effect of training budget (epochs and sample size).
        """
        results_epochs = {'epochs': [], 'train_acc': [], 'val_acc': [],
                          'test_acc': [], 'test_f1': []}  # Epoch sweep results.
        results_samples = {'n_samples': [], 'fraction': [], 'train_acc': [],
                           'val_acc': [], 'test_acc': [], 'test_f1': []}  # Sample sweep results.

        print("\n" + "="*60)  # Section divider.
        print("TRAINING BUDGET ANALYSIS")  # Section title.
        print("="*60)  # Divider.

        val_transform = get_val_transforms()  # Val/test transform.
        train_transform = get_augmentation_transforms('standard')  # Train transform.

        # Part 1: Varying epochs.
        print("\n--- Varying Number of Epochs ---")
        epoch_values = [5, 10, 15, 20, 30]  # Epoch sweep.

        for max_epochs in epoch_values:  # Iterate epochs.
            print(f"\nTraining for {max_epochs} epochs...")

            model = ResNetClassifier(num_classes=2, pretrained=True, freeze_backbone=False)  # Full fine-tune.

            train_dataset = BreastMNISTDataset(train_images, train_labels, transform=train_transform)  # Train.
            val_dataset = BreastMNISTDataset(val_images, val_labels, transform=val_transform)  # Val.
            test_dataset = BreastMNISTDataset(test_images, test_labels, transform=val_transform)  # Test.

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Train.
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Val.
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test.

            history = model.train_model(train_loader, val_loader, epochs=max_epochs, patience=max_epochs+1)  # Train.
            test_metrics = model.evaluate(test_loader, test_labels)  # Test metrics.

            results_epochs['epochs'].append(max_epochs)  # Store epochs.
            results_epochs['train_acc'].append(history['train_acc'][-1])  # Store train acc.
            results_epochs['val_acc'].append(history['val_acc'][-1])  # Store val acc.
            results_epochs['test_acc'].append(test_metrics['accuracy'])  # Store test acc.
            results_epochs['test_f1'].append(test_metrics['f1_score'])  # Store test F1.

            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")  # Log.

            # Release memory before next epoch setting.
            del model, train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Part 2: Varying sample size
        print("\n--- Varying Training Sample Size ---")
        sample_fractions = [0.25, 0.5, 0.75, 1.0]  # Fractions to test.

        for fraction in sample_fractions:  # Iterate sample fractions.
            n_samples = int(len(train_images) * fraction)  # Compute sample size.
            print(f"\nTraining with {n_samples} samples ({fraction*100:.0f}%)...")  # Log.

            # Subsample training data.
            indices = np.random.choice(len(train_images), n_samples, replace=False)  # Subsample.
            train_images_subset = train_images[indices]  # Subset images.
            train_labels_subset = train_labels[indices]  # Subset labels.

            model = ResNetClassifier(num_classes=2, pretrained=True, freeze_backbone=False)  # Full fine-tune.

            train_dataset = BreastMNISTDataset(train_images_subset, train_labels_subset, transform=train_transform)  # Train.
            val_dataset = BreastMNISTDataset(val_images, val_labels, transform=val_transform)  # Val.
            test_dataset = BreastMNISTDataset(test_images, test_labels, transform=val_transform)  # Test.

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Train.
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Val.
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test.

            history = model.train_model(train_loader, val_loader, epochs=15, patience=5)  # Train.
            test_metrics = model.evaluate(test_loader, test_labels)  # Test metrics.

            results_samples['n_samples'].append(n_samples)  # Store sample count.
            results_samples['fraction'].append(fraction)  # Store fraction.
            results_samples['train_acc'].append(history['train_acc'][-1])  # Store train acc.
            results_samples['val_acc'].append(history['val_acc'][-1])  # Store val acc.
            results_samples['test_acc'].append(test_metrics['accuracy'])  # Store test acc.
            results_samples['test_f1'].append(test_metrics['f1_score'])  # Store test F1.

            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")  # Log.

            # Release memory before next sample size.
            del model, train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.results['training_budget_epochs'] = results_epochs  # Cache results.
        self.results['training_budget_samples'] = results_samples  # Cache results.
        return {'epochs': results_epochs, 'samples': results_samples}  # Return both.


def run_lite_benchmark(data_loader, save_dir: str = 'Code/B') -> Dict:
    """
    Run a lightweight ResNet training (no exhaustive benchmarks).
    Trains a single model with sensible defaults to save memory.

    Args:
        data_loader: BreastMNISTLoader instance
        save_dir: Directory to save results

    Returns:
        Dictionary with training results
    """
    import gc

    print("\n" + "="*70)  # Section divider.
    print("RESNET-18 LITE MODE")  # Section title.
    print("="*70)  # Divider.

    # Load data
    train_images, train_labels = data_loader.get_train_data()  # Training split.
    val_images, val_labels = data_loader.get_val_data()  # Validation split.
    test_images, test_labels = data_loader.get_test_data()  # Test split.

    print(f"\nDataset sizes:")  # Header.
    print(f"  Train: {len(train_images)}")  # Train size.
    print(f"  Val: {len(val_images)}")  # Val size.
    print(f"  Test: {len(test_images)}")  # Test size.
    print(f"  Device: {device}")  # Device info.

    # Use smaller batch size to reduce memory.
    batch_size = 16  # Smaller batch for memory.

    # Create model with full fine-tuning and standard augmentation.
    print("\nCreating ResNet-18 model (pretrained, full fine-tuning)...")  # Status.
    model = ResNetClassifier(num_classes=2, pretrained=True, freeze_backbone=False)  # Model.

    train_transform = get_augmentation_transforms('standard')  # Train transform.
    val_transform = get_val_transforms()  # Val/test transform.

    train_dataset = BreastMNISTDataset(train_images, train_labels, transform=train_transform)  # Train.
    val_dataset = BreastMNISTDataset(val_images, val_labels, transform=val_transform)  # Val.
    test_dataset = BreastMNISTDataset(test_images, test_labels, transform=val_transform)  # Test.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Train.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Val.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test.

    # Train with fewer epochs for a faster baseline.
    print("\nTraining model...")  # Status.
    final_history = model.train_model(train_loader, val_loader, epochs=15, patience=5)  # Train.

    # Evaluate
    final_metrics = model.evaluate(test_loader, test_labels)  # Test metrics.

    print(f"\nFinal Test Results:")  # Header.
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")  # Accuracy.
    print(f"  Precision: {final_metrics['precision']:.4f}")  # Precision.
    print(f"  Recall:    {final_metrics['recall']:.4f}")  # Recall.
    print(f"  F1 Score:  {final_metrics['f1_score']:.4f}")  # F1 score.
    print(f"\nConfusion Matrix:")  # Header.
    print(final_metrics['confusion_matrix'])  # Matrix values.

    # Save model for reuse.
    os.makedirs(save_dir, exist_ok=True)  # Ensure output dir exists.
    model_path = os.path.join(save_dir, 'resnet_best_model.pth')  # Output path.
    model.save_model(model_path)  # Save model.

    # Clean up to free memory.
    del train_loader, val_loader, test_loader  # Drop loaders.
    del train_dataset, val_dataset, test_dataset  # Drop datasets.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache.
    gc.collect()  # Run GC.

    return {
        'final_training_history': final_history,
        'final_metrics': final_metrics
    }  # Return results.


def run_full_benchmark(data_loader, save_dir: str = 'Code/B',
                       quick_mode: bool = False) -> Dict:
    """
    Run the complete ResNet benchmarking suite.

    Args:
        data_loader: BreastMNISTLoader instance
        save_dir: Directory to save results
        quick_mode: If True, run fewer experiments for faster testing

    Returns:
        Dictionary with all benchmark results
    """
    print("\n" + "="*70)  # Section divider.
    print("RESNET-18 COMPREHENSIVE BENCHMARK")  # Section title.
    print("="*70)  # Divider.

    # Load data
    train_images, train_labels = data_loader.get_train_data()  # Training split.
    val_images, val_labels = data_loader.get_val_data()  # Validation split.
    test_images, test_labels = data_loader.get_test_data()  # Test split.

    print(f"\nDataset sizes:")  # Header.
    print(f"  Train: {len(train_images)}")  # Train size.
    print(f"  Val: {len(val_images)}")  # Val size.
    print(f"  Test: {len(test_images)}")  # Test size.
    print(f"  Device: {device}")  # Device info.

    benchmark = ResNetBenchmark()  # Benchmark helper.

    epochs = 10 if quick_mode else 15  # Shorter for quick mode.
    batch_size = 8  # Smaller batch size for memory safety.

    # 1. Model capacity analysis.
    capacity_results = benchmark.benchmark_model_capacity(
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels,
        epochs=epochs, batch_size=batch_size
    )

    # 2. Augmentation analysis.
    aug_results = benchmark.benchmark_augmentation(
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels,
        epochs=epochs, batch_size=batch_size
    )

    # 3. Training budget analysis.
    budget_results = benchmark.benchmark_training_budget(
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels,
        batch_size=batch_size
    )

    # 4. Train final best model.
    print("\n" + "="*60)  # Section divider.
    print("TRAINING FINAL BEST MODEL")  # Section title.
    print("="*60)  # Divider.

    # Use full fine-tuning with standard augmentation.
    final_model = ResNetClassifier(num_classes=2, pretrained=True, freeze_backbone=False)  # Final model.

    train_transform = get_augmentation_transforms('standard')  # Train transform.
    val_transform = get_val_transforms()  # Val/test transform.

    train_dataset = BreastMNISTDataset(train_images, train_labels, transform=train_transform)  # Train.
    val_dataset = BreastMNISTDataset(val_images, val_labels, transform=val_transform)  # Val.
    test_dataset = BreastMNISTDataset(test_images, test_labels, transform=val_transform)  # Test.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Train.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Val.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Test.

    final_history = final_model.train_model(train_loader, val_loader, epochs=20, patience=7)  # Train.
    final_metrics = final_model.evaluate(test_loader, test_labels)  # Evaluate.

    print(f"\nFinal Test Results:")  # Header.
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")  # Accuracy.
    print(f"  Precision: {final_metrics['precision']:.4f}")  # Precision.
    print(f"  Recall:    {final_metrics['recall']:.4f}")  # Recall.
    print(f"  F1 Score:  {final_metrics['f1_score']:.4f}")  # F1.
    print(f"\nConfusion Matrix:")  # Header.
    print(final_metrics['confusion_matrix'])  # Matrix values.

    # Save model.
    os.makedirs(save_dir, exist_ok=True)  # Ensure output dir exists.
    model_path = os.path.join(save_dir, 'resnet_best_model.pth')  # Output path.
    final_model.save_model(model_path)  # Save model.

    # Compile all results for downstream plotting/reporting.
    all_results = {
        'model_capacity': capacity_results,
        'augmentation': aug_results,
        'training_budget': budget_results,
        'final_training_history': final_history,
        'final_metrics': final_metrics
    }

    return all_results  # Return full benchmark suite.
