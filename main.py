"""
AMLS Assignment 2025-2026: BreastMNIST Classification

Main entry point for running the complete benchmarking pipeline.

This script:
1. Loads the BreastMNIST dataset
2. Runs comprehensive benchmarks on k-NN (Model A) and ResNet-18 (Model B)
3. Generates evaluation metrics and visualizations
4. Saves trained models and results

Usage:
    python main.py [--model {all,knn,resnet}] [--quick]

Author: Student Number 25048282
Course: ELEC0134 Applied Machine Learning Systems
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse  # For parsing command-line arguments (e.g., --model, --quick)
import os  # For file and directory operations (creating folders, joining paths)
import sys  # For system-level operations (modifying Python path, exiting program)
import json  # For saving results as JSON files
import gc  # Garbage collector for manually freeing memory after benchmarks
import numpy as np  # NumPy for numerical operations and array handling
from datetime import datetime  # For timestamping the benchmark run

# Add Code directory to Python path so we can import our custom modules
# __file__ is the current script's path; we add 'Code' subdirectory to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code'))

# Import data loader class that handles loading BreastMNIST dataset
from Code.data_loader import BreastMNISTLoader

# Import benchmark functions for k-NN model (Model A)
# run_full_benchmark: Runs comprehensive experiments (feature methods, k values, hyperparameters)
# run_lite_benchmark: Runs minimal training for memory-constrained environments
from Code.A.knn_model import run_full_benchmark as run_knn_benchmark
from Code.A.knn_model import run_lite_benchmark as run_knn_lite

# Import benchmark functions for ResNet model (Model B)
# run_full_benchmark: Runs comprehensive experiments (capacity, augmentation, epochs)
# run_lite_benchmark: Runs minimal training for memory-constrained environments
from Code.B.resnet_model import run_full_benchmark as run_resnet_benchmark
from Code.B.resnet_model import run_lite_benchmark as run_resnet_lite

# Import evaluation utilities for generating plots and printing metrics
from Code.evaluation import (
    generate_all_plots,  # Creates all visualization plots and saves them
    print_metrics_summary,  # Prints formatted evaluation metrics to console
    plot_model_comparison  # Creates comparison plot between k-NN and ResNet
)


# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

def parse_args():
    """
    Parse command line arguments.

    This function sets up the argument parser to handle:
    - --model: Which model(s) to run (all, knn, or resnet)
    - --quick: Run fewer experiments for faster testing
    - --lite: Run minimal experiments to save memory
    - --no-plots: Skip generating visualization plots
    - --output-dir: Where to save results

    Returns:
        argparse.Namespace: Parsed arguments object with all settings
    """
    # Create ArgumentParser with description shown in help text
    parser = argparse.ArgumentParser(
        description='AMLS Assignment: BreastMNIST Classification Benchmark'
    )

    # --model argument: Choose which model(s) to benchmark
    # choices restricts input to valid options; default='all' runs both models
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'knn', 'resnet'],  # Only these values are allowed
        default='all',  # If not specified, run both models
        help='Which model(s) to run (default: all)'
    )

    # --quick argument: Boolean flag for faster testing (fewer experiments)
    # action='store_true' means: if flag present -> True, else -> False
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick mode with fewer experiments (for testing)'
    )

    # --lite argument: Boolean flag for memory-constrained runs
    # Only trains final model without exhaustive benchmarking
    parser.add_argument(
        '--lite',
        action='store_true',
        help='Run in lite mode: trains only the final model for each (no exhaustive benchmarks). Use this if you run out of memory.'
    )

    # --no-plots argument: Skip plot generation (useful for headless servers)
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    # --output-dir argument: Specify where to save results
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',  # Default saves to 'results/' folder
        help='Directory to save results (default: results)'
    )

    # Parse and return the arguments
    return parser.parse_args()


# ============================================================================
# RESULTS SERIALIZATION
# ============================================================================

def save_results(results: dict, filepath: str) -> None:
    """
    Save results to JSON file.

    This function handles the conversion of NumPy arrays and types to
    JSON-serializable Python types, then saves to a JSON file.

    Args:
        results: Dictionary of results to save (may contain numpy arrays)
        filepath: Path to save the JSON file
    """

    # Nested helper function to recursively convert numpy types to Python types
    # JSON cannot serialize numpy arrays, integers, or floats directly
    def convert_to_serializable(obj):
        """Recursively convert numpy types to JSON-serializable Python types."""

        # Convert numpy array to Python list
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Convert numpy integer types (int32, int64, etc.) to Python int
        elif isinstance(obj, np.integer):
            return int(obj)

        # Convert numpy float types (float32, float64, etc.) to Python float
        elif isinstance(obj, np.floating):
            return float(obj)

        # Recursively process dictionaries (convert all values)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}

        # Recursively process lists (convert all items)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]

        # Return unchanged if already JSON-serializable (str, int, float, bool, None)
        return obj

    # Convert the entire results dictionary
    serializable_results = convert_to_serializable(results)

    # Write to JSON file with pretty-printing (indent=2 for readability)
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Confirm save location to user
    print(f"Results saved to {filepath}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main function to run the complete benchmark pipeline.

    This function orchestrates the entire benchmarking process:
    1. Parse command line arguments
    2. Load the BreastMNIST dataset
    3. Run k-NN benchmark (if selected)
    4. Run ResNet benchmark (if selected)
    5. Generate plots and save results
    6. Print final summary
    """

    # Parse command line arguments to get user preferences
    args = parse_args()

    # ========================================================================
    # PRINT HEADER AND CONFIGURATION
    # ========================================================================

    # Print banner with assignment info
    print("="*70)
    print("AMLS ASSIGNMENT 2025-2026: BREASTMNIST CLASSIFICATION")
    print("="*70)

    # Print current configuration for reproducibility and logging
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # Timestamp
    print(f"Model(s) to run: {args.model}")  # Which models selected
    print(f"Quick mode: {args.quick}")  # Whether using quick mode
    print(f"Lite mode: {args.lite}")  # Whether using lite mode (memory-saving)
    print("="*70)

    # ========================================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================================

    # Create output directory if it doesn't exist
    # exist_ok=True prevents error if directory already exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================
    # STEP 1/4: LOAD DATASET
    # ========================================================================

    print("\n[1/4] Loading BreastMNIST Dataset...")

    try:
        # Initialize data loader with path to dataset
        # BreastMNISTLoader handles loading the .npz file and splitting data
        data_loader = BreastMNISTLoader(data_path='Datasets/BreastMNIST')

        # Get dataset statistics for display
        dataset_info = data_loader.get_dataset_info()

        # Print dataset information
        print("\nDataset Information:")
        print(f"  Training samples:   {dataset_info['train_size']}")      # Number of training images
        print(f"  Validation samples: {dataset_info['val_size']}")        # Number of validation images
        print(f"  Test samples:       {dataset_info['test_size']}")       # Number of test images
        print(f"  Image shape:        {dataset_info['image_shape']}")     # Image dimensions (28, 28)
        print(f"  Number of classes:  {dataset_info['num_classes']}")     # Binary: 0=benign, 1=malignant
        print(f"  Class distribution (train): {dataset_info['class_distribution']['train']}")  # Class balance

    except FileNotFoundError as e:
        # Handle case where dataset file is not found
        print(f"\nError: {e}")
        print("\nPlease ensure the BreastMNIST dataset is placed in:")
        print("  Datasets/BreastMNIST/breastmnist.npz")
        print("\nDownload from: https://medmnist.com/")
        sys.exit(1)  # Exit with error code 1

    # Dictionary to store results from both models
    results = {}

    # ========================================================================
    # STEP 2/4: RUN K-NN (MODEL A) BENCHMARK
    # ========================================================================

    # Check if k-NN should be run (either 'all' or 'knn' selected)
    if args.model in ['all', 'knn']:
        print("\n[2/4] Running k-NN (Model A) Benchmark...")
        print("-"*50)

        try:
            # Choose between lite mode (memory-saving) or full benchmark
            if args.lite:
                # Lite mode: Train single model with default hyperparameters
                # Uses HOG features, k=5, distance weighting
                knn_results = run_knn_lite(data_loader, save_dir='Code/A')
            else:
                # Full mode: Run comprehensive experiments
                # Tests multiple feature methods, k values, hyperparameters
                knn_results = run_knn_benchmark(data_loader, save_dir='Code/A')

            # Store results for later analysis
            results['knn'] = knn_results

            # Print formatted summary of final metrics
            print_metrics_summary(knn_results['final_metrics'], 'k-NN (Model A)')

            # Force garbage collection to free memory before next benchmark
            gc.collect()

        except Exception as e:
            # Handle any errors during k-NN training
            print(f"Error running k-NN benchmark: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
    else:
        # k-NN not selected, skip this step
        print("\n[2/4] Skipping k-NN benchmark")

    # ========================================================================
    # STEP 3/4: RUN RESNET (MODEL B) BENCHMARK
    # ========================================================================

    # Check if ResNet should be run (either 'all' or 'resnet' selected)
    if args.model in ['all', 'resnet']:
        print("\n[3/4] Running ResNet-18 (Model B) Benchmark...")
        print("-"*50)

        try:
            # Choose between lite mode (memory-saving) or full benchmark
            if args.lite:
                # Lite mode: Train single model with smaller batch size
                # Uses pretrained weights, standard augmentation, 15 epochs
                resnet_results = run_resnet_lite(data_loader, save_dir='Code/B')
            else:
                # Full mode: Run comprehensive experiments
                # Tests model capacity, augmentation levels, epoch counts, sample sizes
                resnet_results = run_resnet_benchmark(
                    data_loader,
                    save_dir='Code/B',
                    quick_mode=args.quick  # If quick, use fewer epochs
                )

            # Store results for later analysis
            results['resnet'] = resnet_results

            # Print formatted summary of final metrics
            print_metrics_summary(resnet_results['final_metrics'], 'ResNet-18 (Model B)')

            # Force garbage collection to free memory (especially important for GPU)
            gc.collect()

        except Exception as e:
            # Handle any errors during ResNet training
            print(f"Error running ResNet benchmark: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
    else:
        # ResNet not selected, skip this step
        print("\n[3/4] Skipping ResNet benchmark")

    # ========================================================================
    # STEP 4/4: GENERATE PLOTS AND SAVE RESULTS
    # ========================================================================

    print("\n[4/4] Generating Results and Visualizations...")
    print("-"*50)

    # Generate plots if not disabled and we have results
    if not args.no_plots and len(results) > 0:
        try:
            # Get results for each model (empty dict if not run)
            knn_res = results.get('knn', {})
            resnet_res = results.get('resnet', {})

            # Generate all plots and save to output directory
            # This creates confusion matrices, learning curves, comparison plots, etc.
            generate_all_plots(knn_res, resnet_res, output_dir=args.output_dir)

        except Exception as e:
            # Non-fatal: warn but continue if plot generation fails
            print(f"Warning: Could not generate some plots: {e}")

    # Save results to JSON file if we have any
    if len(results) > 0:
        # Construct path for results JSON file
        results_path = os.path.join(args.output_dir, 'benchmark_results.json')

        # Save results (handles numpy type conversion)
        save_results(results, results_path)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    # Print final summary banner
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # Print k-NN results if available
    if 'knn' in results and 'final_metrics' in results['knn']:
        knn_metrics = results['knn']['final_metrics']
        print(f"\nk-NN (Model A):")
        print(f"  Test Accuracy: {knn_metrics['accuracy']:.4f}")   # Overall accuracy
        print(f"  Test F1 Score: {knn_metrics['f1_score']:.4f}")   # F1 score (harmonic mean of precision/recall)

        # Print best hyperparameters if available
        if 'best_params' in results['knn']:
            print(f"  Best Parameters: {results['knn']['best_params']}")

    # Print ResNet results if available
    if 'resnet' in results and 'final_metrics' in results['resnet']:
        resnet_metrics = results['resnet']['final_metrics']
        print(f"\nResNet-18 (Model B):")
        print(f"  Test Accuracy: {resnet_metrics['accuracy']:.4f}")  # Overall accuracy
        print(f"  Test F1 Score: {resnet_metrics['f1_score']:.4f}")  # F1 score

    # Print locations where results are saved
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"Models saved to: Code/A/ and Code/B/")
    print("="*70)
    print("Benchmark complete!")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Standard Python idiom: only run main() if this script is executed directly
# (not when imported as a module)
if __name__ == '__main__':
    main()
