"""
Evaluation and Visualization Module

This module provides:
- Performance metrics computation
- Visualization of results (learning curves, confusion matrices, etc.)
- Comparative analysis between models
- Report generation utilities
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np  # NumPy for numerical array operations
import matplotlib.pyplot as plt  # Matplotlib for creating plots and figures
import seaborn as sns  # Seaborn for enhanced statistical visualizations
from typing import Dict, List, Optional  # Type hints for better documentation
import os  # OS module for file path operations


# ============================================================================
# LEARNING CURVES VISUALIZATION
# ============================================================================

def plot_learning_curves(history: Dict, title: str = "Learning Curves",
                         save_path: Optional[str] = None) -> None:
    """
    Plot training and validation learning curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: Plot title
        save_path: Path to save the figure
    """
    # Create a figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create epoch numbers starting from 1
    epochs = range(1, len(history['train_loss']) + 1)

    # ========================================================================
    # LEFT SUBPLOT: Loss curves
    # ========================================================================
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)  # X-axis label
    axes[0].set_ylabel('Loss', fontsize=12)  # Y-axis label
    axes[0].set_title('Loss Curves', fontsize=14)  # Subplot title
    axes[0].legend(fontsize=10)  # Add legend
    axes[0].grid(True, alpha=0.3)  # Add semi-transparent grid

    # ========================================================================
    # RIGHT SUBPLOT: Accuracy curves
    # ========================================================================
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Add overall title to the figure
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()  # Adjust spacing to prevent overlap

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()  # Display the figure


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        labels: Class labels
        title: Plot title
        save_path: Path to save the figure
    """
    # Set default labels for binary classification (BreastMNIST)
    if labels is None:
        labels = ['Benign', 'Malignant']

    # Create figure for the heatmap
    plt.figure(figsize=(8, 6))

    # Create heatmap using seaborn
    # annot=True shows the counts in each cell
    # fmt='d' formats numbers as integers
    # cmap='Blues' uses blue color gradient
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14})  # Font size for annotations

    plt.xlabel('Predicted Label', fontsize=12)  # X-axis: model predictions
    plt.ylabel('True Label', fontsize=12)  # Y-axis: actual labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# K-NN K VALUE ANALYSIS VISUALIZATION
# ============================================================================

def plot_k_analysis(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot k-NN model capacity analysis (varying k values).

    Args:
        results: Dictionary with k_values and accuracies
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract k values from results
    k_values = results['k_values']

    # ========================================================================
    # LEFT SUBPLOT: Accuracy vs k
    # ========================================================================
    # Plot lines for train, validation, and test accuracy
    axes[0].plot(k_values, results['train_acc'], 'b-o', label='Train', linewidth=2, markersize=6)
    axes[0].plot(k_values, results['val_acc'], 'g-s', label='Validation', linewidth=2, markersize=6)
    axes[0].plot(k_values, results['test_acc'], 'r-^', label='Test', linewidth=2, markersize=6)
    axes[0].set_xlabel('k (Number of Neighbors)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs k', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')  # Use logarithmic scale for k values

    # ========================================================================
    # RIGHT SUBPLOT: F1 Score vs k
    # ========================================================================
    axes[1].plot(k_values, results['train_f1'], 'b-o', label='Train', linewidth=2, markersize=6)
    axes[1].plot(k_values, results['val_f1'], 'g-s', label='Validation', linewidth=2, markersize=6)
    axes[1].plot(k_values, results['test_f1'], 'r-^', label='Test', linewidth=2, markersize=6)
    axes[1].set_xlabel('k (Number of Neighbors)', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score vs k', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')  # Use logarithmic scale for k values

    # Add overall title
    plt.suptitle('k-NN Model Capacity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# FEATURE EXTRACTION METHOD COMPARISON
# ============================================================================

def plot_feature_comparison(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot comparison of different feature extraction methods.

    Args:
        results: Dictionary with feature method results
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract feature method names
    methods = results['method']
    x = np.arange(len(methods))  # Create x positions for bars
    width = 0.25  # Width of each bar

    # ========================================================================
    # LEFT SUBPLOT: Accuracy comparison by feature method
    # ========================================================================
    # Grouped bar chart: train, validation, test accuracy
    axes[0].bar(x - width, results['train_acc'], width, label='Train', color='steelblue')
    axes[0].bar(x, results['val_acc'], width, label='Validation', color='forestgreen')
    axes[0].bar(x + width, results['test_acc'], width, label='Test', color='indianred')
    axes[0].set_xlabel('Feature Method', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy by Feature Method', fontsize=14)
    axes[0].set_xticks(x)  # Set x-tick positions
    axes[0].set_xticklabels(methods, rotation=45, ha='right')  # Rotate labels for readability
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines

    # ========================================================================
    # RIGHT SUBPLOT: F1 Score and Feature Dimension (dual y-axis)
    # ========================================================================
    ax2 = axes[1]
    # Bar chart for F1 scores
    bars = ax2.bar(x, results['f1_score'], color='steelblue', alpha=0.7, label='F1 Score')
    ax2.set_xlabel('Feature Method', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12, color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='steelblue')  # Color y-axis labels

    # Create secondary y-axis for feature dimension
    ax3 = ax2.twinx()  # Share x-axis with ax2
    ax3.plot(x, results['feature_dim'], 'r-o', linewidth=2, markersize=8, label='Feature Dim')
    ax3.set_ylabel('Feature Dimension', fontsize=12, color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    axes[1].set_title('F1 Score and Feature Dimension', fontsize=14)

    # Add overall title
    plt.suptitle('Feature Extraction Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# TRAINING BUDGET ANALYSIS (SAMPLE SIZE EFFECT)
# ============================================================================

def plot_training_budget(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot training budget analysis (sample size effect).

    Args:
        results: Dictionary with training budget results
        save_path: Path to save the figure
    """
    # Create single figure
    plt.figure(figsize=(10, 6))

    # Extract number of samples
    n_samples = results['n_samples']

    # Plot all metrics against number of training samples
    plt.plot(n_samples, results['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=8)
    plt.plot(n_samples, results['val_acc'], 'g-s', label='Validation Accuracy', linewidth=2, markersize=8)
    plt.plot(n_samples, results['test_acc'], 'r-^', label='Test Accuracy', linewidth=2, markersize=8)
    plt.plot(n_samples, results['f1_score'], 'm-d', label='Test F1', linewidth=2, markersize=8)

    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Training Budget Analysis: Effect of Sample Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# DATA AUGMENTATION COMPARISON
# ============================================================================

def plot_augmentation_comparison(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot data augmentation comparison.

    Args:
        results: Dictionary with augmentation results
        save_path: Path to save the figure
    """
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract augmentation levels
    levels = results['augment_level']
    x = np.arange(len(levels))  # X positions for bars
    width = 0.2  # Width of each bar (4 bars, so narrower)

    # Create grouped bar chart with 4 metrics
    ax.bar(x - 1.5*width, results['train_acc'], width, label='Train Acc', color='steelblue')
    ax.bar(x - 0.5*width, results['val_acc'], width, label='Val Acc', color='forestgreen')
    ax.bar(x + 0.5*width, results['test_acc'], width, label='Test Acc', color='indianred')
    ax.bar(x + 1.5*width, results['test_f1'], width, label='Test F1', color='darkorange')

    ax.set_xlabel('Augmentation Level', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Data Augmentation Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)  # Labels: 'none', 'light', 'heavy'
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# RESNET MODEL CAPACITY COMPARISON
# ============================================================================

def plot_model_capacity_comparison(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot ResNet model capacity comparison.

    Args:
        results: Dictionary with model capacity results
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract model configurations
    configs = results['config']
    x = np.arange(len(configs))  # X positions
    width = 0.25  # Bar width

    # ========================================================================
    # LEFT SUBPLOT: Accuracy by model configuration
    # ========================================================================
    axes[0].bar(x - width, results['train_acc'], width, label='Train', color='steelblue')
    axes[0].bar(x, results['val_acc'], width, label='Validation', color='forestgreen')
    axes[0].bar(x + width, results['test_acc'], width, label='Test', color='indianred')
    axes[0].set_xlabel('Model Configuration', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy by Model Configuration', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # RIGHT SUBPLOT: F1 Score vs Model Complexity (dual y-axis)
    # ========================================================================
    ax2 = axes[1]
    # Bar chart for F1 scores
    ax2.bar(x, results['test_f1'], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Model Configuration', fontsize=12)
    ax2.set_ylabel('Test F1 Score', fontsize=12, color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    # Create secondary y-axis for trainable parameters
    ax3 = ax2.twinx()
    # Convert parameters to millions for readability
    ax3.plot(x, np.array(results['trainable_params'])/1e6, 'r-o', linewidth=2, markersize=8)
    ax3.set_ylabel('Trainable Parameters (M)', fontsize=12, color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    axes[1].set_title('F1 Score vs Model Complexity', fontsize=14)

    # Add overall title
    plt.suptitle('ResNet Model Capacity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# EPOCHS ANALYSIS VISUALIZATION
# ============================================================================

def plot_epochs_analysis(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot effect of number of epochs.

    Args:
        results: Dictionary with epoch analysis results
        save_path: Path to save the figure
    """
    # Create single figure
    plt.figure(figsize=(10, 6))

    # Extract epoch counts
    epochs = results['epochs']

    # Plot all metrics against number of epochs
    plt.plot(epochs, results['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=8)
    plt.plot(epochs, results['val_acc'], 'g-s', label='Validation Accuracy', linewidth=2, markersize=8)
    plt.plot(epochs, results['test_acc'], 'r-^', label='Test Accuracy', linewidth=2, markersize=8)
    plt.plot(epochs, results['test_f1'], 'm-d', label='Test F1', linewidth=2, markersize=8)

    plt.xlabel('Number of Training Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Training Budget Analysis: Effect of Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# FINAL MODEL COMPARISON (K-NN VS RESNET)
# ============================================================================

def plot_model_comparison(knn_metrics: Dict, resnet_metrics: Dict,
                          save_path: Optional[str] = None) -> None:
    """
    Plot final comparison between k-NN and ResNet models.

    Args:
        knn_metrics: k-NN final metrics
        resnet_metrics: ResNet final metrics
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ========================================================================
    # LEFT SUBPLOT: Metrics comparison bar chart
    # ========================================================================
    # Define metrics to compare
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    # Extract values for each model
    knn_values = [knn_metrics[m] for m in metrics]
    resnet_values = [resnet_metrics[m] for m in metrics]

    x = np.arange(len(metrics))  # X positions
    width = 0.35  # Bar width

    # Create grouped bar chart
    axes[0].bar(x - width/2, knn_values, width, label='k-NN', color='steelblue')
    axes[0].bar(x + width/2, resnet_values, width, label='ResNet-18', color='indianred')
    axes[0].set_xlabel('Metric', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Performance Metrics Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1)  # Metrics are between 0 and 1

    # ========================================================================
    # RIGHT SUBPLOT: Confusion matrices display (text-based)
    # ========================================================================
    # Display confusion matrices as text (for quick comparison)
    axes[1].text(0.25, 0.5, f"k-NN\nConfusion Matrix:\n{knn_metrics['confusion_matrix']}",
                 ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.75, 0.5, f"ResNet-18\nConfusion Matrix:\n{resnet_metrics['confusion_matrix']}",
                 ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    axes[1].set_title('Confusion Matrices', fontsize=14)
    axes[1].axis('off')  # Hide axes for text display

    # Add overall title
    plt.suptitle('Model Comparison: k-NN vs ResNet-18', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# METRICS SUMMARY PRINTING
# ============================================================================

def print_metrics_summary(metrics: Dict, model_name: str = "Model") -> None:
    """
    Print a formatted summary of evaluation metrics.

    Args:
        metrics: Dictionary with evaluation metrics
        model_name: Name of the model for display
    """
    # Print header with model name
    print(f"\n{'='*50}")
    print(f"{model_name} - Final Evaluation Results")
    print('='*50)

    # Print each metric with 4 decimal places
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")  # Overall correct predictions
    print(f"  Precision: {metrics['precision']:.4f}")  # True positives / predicted positives
    print(f"  Recall:    {metrics['recall']:.4f}")  # True positives / actual positives
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")  # Harmonic mean of precision and recall

    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print('='*50)


# ============================================================================
# GENERATE ALL PLOTS FOR REPORT
# ============================================================================

def generate_all_plots(knn_results: Dict, resnet_results: Dict,
                       output_dir: str = 'results') -> None:
    """
    Generate all plots for the report.

    Args:
        knn_results: Complete k-NN benchmark results
        resnet_results: Complete ResNet benchmark results
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating plots...")

    # ========================================================================
    # K-NN PLOTS
    # ========================================================================

    # Plot k value analysis if available
    if 'k_value_analysis' in knn_results:
        plot_k_analysis(knn_results['k_value_analysis'],
                        save_path=os.path.join(output_dir, 'knn_k_analysis.png'))

    # Plot feature comparison if available
    if 'feature_comparison' in knn_results:
        plot_feature_comparison(knn_results['feature_comparison'],
                                save_path=os.path.join(output_dir, 'knn_feature_comparison.png'))

    # Plot training budget (sample size) analysis if available
    if 'training_budget' in knn_results:
        plot_training_budget(knn_results['training_budget'],
                             save_path=os.path.join(output_dir, 'knn_training_budget.png'))

    # ========================================================================
    # RESNET PLOTS
    # ========================================================================

    # Plot learning curves if available
    if 'final_training_history' in resnet_results:
        plot_learning_curves(resnet_results['final_training_history'],
                             title='ResNet-18 Learning Curves',
                             save_path=os.path.join(output_dir, 'resnet_learning_curves.png'))

    # Plot model capacity comparison if available
    if 'model_capacity' in resnet_results:
        plot_model_capacity_comparison(resnet_results['model_capacity'],
                                        save_path=os.path.join(output_dir, 'resnet_capacity.png'))

    # Plot augmentation comparison if available
    if 'augmentation' in resnet_results:
        plot_augmentation_comparison(resnet_results['augmentation'],
                                      save_path=os.path.join(output_dir, 'resnet_augmentation.png'))

    # Plot training budget analysis if available
    if 'training_budget' in resnet_results:
        # Plot epochs analysis
        if 'epochs' in resnet_results['training_budget']:
            plot_epochs_analysis(resnet_results['training_budget']['epochs'],
                                 save_path=os.path.join(output_dir, 'resnet_epochs.png'))
        # Plot sample size analysis
        if 'samples' in resnet_results['training_budget']:
            plot_training_budget(resnet_results['training_budget']['samples'],
                                 save_path=os.path.join(output_dir, 'resnet_samples.png'))

    # ========================================================================
    # CONFUSION MATRICES
    # ========================================================================

    # Plot k-NN confusion matrix if available
    if 'final_metrics' in knn_results:
        plot_confusion_matrix(knn_results['final_metrics']['confusion_matrix'],
                              title='k-NN Confusion Matrix',
                              save_path=os.path.join(output_dir, 'knn_confusion_matrix.png'))

    # Plot ResNet confusion matrix if available
    if 'final_metrics' in resnet_results:
        plot_confusion_matrix(resnet_results['final_metrics']['confusion_matrix'],
                              title='ResNet-18 Confusion Matrix',
                              save_path=os.path.join(output_dir, 'resnet_confusion_matrix.png'))

    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================

    # Plot final model comparison if both models have results
    if 'final_metrics' in knn_results and 'final_metrics' in resnet_results:
        plot_model_comparison(knn_results['final_metrics'],
                              resnet_results['final_metrics'],
                              save_path=os.path.join(output_dir, 'model_comparison.png'))

    print(f"\nAll plots saved to {output_dir}/")
