"""
k-Nearest Neighbors (k-NN) Model for BreastMNIST Classification

This module implements:
- k-NN classifier with multiple distance metrics
- Hyperparameter tuning (k values, distance metrics, weights)
- Model capacity analysis through varying k
- Training budget analysis through varying sample sizes
- Feature extraction comparison (raw, HOG, PCA)
"""

import numpy as np  # Numerical arrays and vectorized ops.
import time  # Timing for training/extraction benchmarks.
from typing import Dict, List, Tuple, Optional  # Type hints for clarity.
from sklearn.neighbors import KNeighborsClassifier  # Core k-NN implementation.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Metrics.
from sklearn.metrics import confusion_matrix, classification_report  # Diagnostic reports.
from sklearn.model_selection import cross_val_score, GridSearchCV  # CV utilities.
import joblib  # Model persistence for sklearn.
import os  # Path utilities.
import sys  # sys.path manipulation for local imports.

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Local import path.
from data_loader import FeatureExtractor, get_sample_subset  # Feature and sampling helpers.


class KNNClassifier:
    """
    k-Nearest Neighbors classifier wrapper with comprehensive benchmarking.

    Attributes:
        k: Number of neighbors
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
        weights: Weight function ('uniform', 'distance')
        model: Underlying sklearn KNeighborsClassifier
    """

    def __init__(self, k: int = 5, metric: str = 'euclidean',
                 weights: str = 'uniform', p: int = 2):
        """
        Initialize k-NN classifier.

        Args:
            k: Number of neighbors to use
            metric: Distance metric for neighbor search
            weights: Weight function for prediction ('uniform' or 'distance')
            p: Power parameter for Minkowski metric
        """
        self.k = k  # Neighbor count.
        self.metric = metric  # Distance metric for neighbor search.
        self.weights = weights  # Voting scheme.
        self.p = p  # Minkowski power parameter.
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            p=p,
            n_jobs=-1  # Use all CPU cores for neighbor search
        )
        self.training_time = 0  # Wall-clock fit time.
        self.is_fitted = False  # Fit status flag.

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the k-NN model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self
        """
        start_time = time.time()  # Start timer.
        self.model.fit(X, y)  # Fit k-NN on features.
        self.training_time = time.time() - start_time  # Record duration.
        self.is_fitted = True  # Mark model as fitted.
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict (n_samples, n_features)

        Returns:
            Predicted labels
        """
        return self.model.predict(X)  # Delegate to sklearn.

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict (n_samples, n_features)

        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)  # Delegate to sklearn.

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Args:
            X: Test features
            y: True labels

        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)  # Class predictions.
        # For binary problems, take the positive class probability.
        y_proba = self.predict_proba(X)[:, 1]  # Probabilities for class 1.

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }

    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, filepath)  # Serialize sklearn model.
        print(f"Model saved to {filepath}")  # User feedback.

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)  # Load model from disk.
        self.is_fitted = True  # Mark as ready for inference.
        print(f"Model loaded from {filepath}")  # User feedback.


class KNNBenchmark:
    """
    Comprehensive benchmarking suite for k-NN classifier.

    Performs experiments on:
    - Model capacity (varying k)
    - Feature extraction methods
    - Training budget (sample size)
    - Distance metrics and weights
    """

    def __init__(self, random_state: int = 42):
        """Initialize benchmark suite."""
        self.random_state = random_state  # Random seed for reproducibility.
        self.results = {}  # Storage for benchmark outputs.

    def benchmark_k_values(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           k_values: List[int] = None) -> Dict:
        """
        Benchmark different k values (model capacity analysis).

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            k_values: List of k values to test

        Returns:
            Dictionary with results for each k value
        """
        if k_values is None:
            k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]  # Default sweep.

        results = {'k_values': [], 'train_acc': [], 'val_acc': [], 'test_acc': [],
                   'train_f1': [], 'val_f1': [], 'test_f1': [], 'training_time': []}  # Accumulators.

        print("\n" + "="*60)  # Section divider.
        print("MODEL CAPACITY ANALYSIS: Varying k")  # Section title.
        print("="*60)  # Divider.

        for k in k_values:  # Iterate over k values.
            print(f"\nTraining k-NN with k={k}...")
            model = KNNClassifier(k=k)  # Instantiate model.
            model.fit(X_train, y_train)  # Fit on training set.

            train_metrics = model.evaluate(X_train, y_train)  # Train metrics.
            val_metrics = model.evaluate(X_val, y_val)  # Validation metrics.
            test_metrics = model.evaluate(X_test, y_test)  # Test metrics.

            results['k_values'].append(k)  # Store k value.
            results['train_acc'].append(train_metrics['accuracy'])  # Store train accuracy.
            results['val_acc'].append(val_metrics['accuracy'])  # Store val accuracy.
            results['test_acc'].append(test_metrics['accuracy'])  # Store test accuracy.
            results['train_f1'].append(train_metrics['f1_score'])  # Store train F1.
            results['val_f1'].append(val_metrics['f1_score'])  # Store val F1.
            results['test_f1'].append(test_metrics['f1_score'])  # Store test F1.
            results['training_time'].append(model.training_time)  # Store training time.

            print(f"  k={k}: Train Acc={train_metrics['accuracy']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Test Acc={test_metrics['accuracy']:.4f}")

        self.results['k_values'] = results  # Cache for later use.
        return results  # Return to caller.

    def benchmark_feature_methods(self, train_images: np.ndarray, train_labels: np.ndarray,
                                   val_images: np.ndarray, val_labels: np.ndarray,
                                   test_images: np.ndarray, test_labels: np.ndarray,
                                   k: int = 5) -> Dict:
        """
        Benchmark different feature extraction methods.

        Args:
            train_images, train_labels: Training data (raw images)
            val_images, val_labels: Validation data
            test_images, test_labels: Test data
            k: Number of neighbors to use

        Returns:
            Dictionary with results for each feature method
        """
        # Keep a mix of raw pixels and handcrafted features for comparison.
        methods = ['raw', 'hog', 'pca', 'hog_pca']  # Feature extraction variants.
        pca_components = [30, 50, 100]  # PCA sizes to test.

        results = {'method': [], 'feature_dim': [], 'train_acc': [], 'val_acc': [],
                   'test_acc': [], 'f1_score': [], 'extraction_time': [],
                   'explained_variance': []}  # Accumulators.

        print("\n" + "="*60)  # Section divider.
        print("FEATURE EXTRACTION COMPARISON")  # Section title.
        print("="*60)  # Divider.

        for method in methods:  # Iterate over feature methods.
            if method in ['pca', 'hog_pca']:
                # Test multiple PCA component values
                for n_comp in pca_components:
                    self._evaluate_feature_method(
                        method, n_comp, train_images, train_labels,
                        val_images, val_labels, test_images, test_labels,
                        k, results
                    )
            else:
                self._evaluate_feature_method(
                    method, None, train_images, train_labels,
                    val_images, val_labels, test_images, test_labels,
                    k, results
                )

        self.results['feature_methods'] = results  # Cache for later use.
        return results  # Return to caller.

    def _evaluate_feature_method(self, method: str, n_components: Optional[int],
                                  train_images: np.ndarray, train_labels: np.ndarray,
                                  val_images: np.ndarray, val_labels: np.ndarray,
                                  test_images: np.ndarray, test_labels: np.ndarray,
                                  k: int, results: Dict) -> None:
        """Helper to evaluate a single feature extraction method."""
        method_name = f"{method}" if n_components is None else f"{method}_{n_components}"  # Label.
        print(f"\nEvaluating {method_name}...")  # Progress output.

        start_time = time.time()  # Start timing extraction.
        extractor = FeatureExtractor(method=method, n_components=n_components)  # Build extractor.

        # Fit on training set, then transform val/test with the same extractor.
        X_train = extractor.fit_transform(train_images)  # Fit + transform train.
        X_val = extractor.transform(val_images)  # Transform validation.
        X_test = extractor.transform(test_images)  # Transform test.
        extraction_time = time.time() - start_time  # Elapsed extraction time.

        model = KNNClassifier(k=k)  # Instantiate model.
        model.fit(X_train, train_labels)  # Fit on extracted features.

        train_metrics = model.evaluate(X_train, train_labels)  # Train metrics.
        val_metrics = model.evaluate(X_val, val_labels)  # Val metrics.
        test_metrics = model.evaluate(X_test, test_labels)  # Test metrics.

        results['method'].append(method_name)  # Record method name.
        results['feature_dim'].append(X_train.shape[1])  # Record feature size.
        results['train_acc'].append(train_metrics['accuracy'])  # Record train accuracy.
        results['val_acc'].append(val_metrics['accuracy'])  # Record val accuracy.
        results['test_acc'].append(test_metrics['accuracy'])  # Record test accuracy.
        results['f1_score'].append(test_metrics['f1_score'])  # Record test F1.
        results['extraction_time'].append(extraction_time)  # Record extraction time.
        results['explained_variance'].append(extractor.get_explained_variance())  # PCA info.

        print(f"  Feature dim: {X_train.shape[1]}, "
              f"Test Acc: {test_metrics['accuracy']:.4f}, "
              f"F1: {test_metrics['f1_score']:.4f}")

    def benchmark_training_budget(self, train_images: np.ndarray, train_labels: np.ndarray,
                                   val_images: np.ndarray, val_labels: np.ndarray,
                                   test_images: np.ndarray, test_labels: np.ndarray,
                                   feature_method: str = 'hog',
                                   k: int = 5,
                                   sample_fractions: List[float] = None) -> Dict:
        """
        Benchmark effect of training set size (training budget analysis).

        Args:
            train_images, train_labels: Full training data
            val_images, val_labels: Validation data
            test_images, test_labels: Test data
            feature_method: Feature extraction method to use
            k: Number of neighbors
            sample_fractions: Fractions of training data to use

        Returns:
            Dictionary with results for each sample size
        """
        if sample_fractions is None:
            sample_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Default fractions.

        results = {'n_samples': [], 'fraction': [], 'train_acc': [], 'val_acc': [],
                   'test_acc': [], 'f1_score': [], 'training_time': []}  # Accumulators.

        print("\n" + "="*60)  # Section divider.
        print("TRAINING BUDGET ANALYSIS: Varying sample size")  # Section title.
        print("="*60)  # Divider.

        # Extract features once for efficiency; subsets reuse the same features.
        extractor = FeatureExtractor(method=feature_method)  # Build extractor.
        X_train_full = extractor.fit_transform(train_images)  # Fit on full train.
        X_val = extractor.transform(val_images)  # Transform val.
        X_test = extractor.transform(test_images)  # Transform test.

        for fraction in sample_fractions:  # Iterate over sample fractions.
            n_samples = int(len(train_images) * fraction)  # Compute sample size.
            print(f"\nTraining with {n_samples} samples ({fraction*100:.0f}%)...")  # Log.

            # Get subset (use features directly)
            if fraction < 1.0:
                indices = np.random.choice(len(X_train_full), n_samples, replace=False)  # Subsample.
                X_train_subset = X_train_full[indices]  # Subset features.
                y_train_subset = train_labels[indices]  # Subset labels.
            else:
                X_train_subset = X_train_full  # Use full features.
                y_train_subset = train_labels  # Use full labels.

            model = KNNClassifier(k=k)  # Instantiate model.
            model.fit(X_train_subset, y_train_subset)  # Fit on subset.

            train_metrics = model.evaluate(X_train_subset, y_train_subset)  # Train metrics.
            val_metrics = model.evaluate(X_val, val_labels)  # Val metrics.
            test_metrics = model.evaluate(X_test, test_labels)  # Test metrics.

            results['n_samples'].append(n_samples)  # Store sample count.
            results['fraction'].append(fraction)  # Store fraction.
            results['train_acc'].append(train_metrics['accuracy'])  # Store train acc.
            results['val_acc'].append(val_metrics['accuracy'])  # Store val acc.
            results['test_acc'].append(test_metrics['accuracy'])  # Store test acc.
            results['f1_score'].append(test_metrics['f1_score'])  # Store test F1.
            results['training_time'].append(model.training_time)  # Store time.

            print(f"  n={n_samples}: Test Acc={test_metrics['accuracy']:.4f}, "
                  f"F1={test_metrics['f1_score']:.4f}")

        self.results['training_budget'] = results  # Cache for later use.
        return results  # Return to caller.

    def benchmark_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Grid search over hyperparameters.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Dictionary with best parameters and CV results
        """
        print("\n" + "="*60)  # Section divider.
        print("HYPERPARAMETER GRID SEARCH")  # Section title.
        print("="*60)  # Divider.

        # Use a smaller grid to reduce memory/CPU load on modest machines.
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }

        base_model = KNeighborsClassifier(n_jobs=1)  # Limit parallelism to reduce memory.
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3,
            scoring='f1', n_jobs=1, verbose=1
        )

        # Combine train and val for cross-validation to maximize data per fold.
        X_combined = np.vstack([X_train, X_val])  # Stack features.
        y_combined = np.concatenate([y_train, y_val])  # Stack labels.

        grid_search.fit(X_combined, y_combined)  # Run grid search.

        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        self.results['hyperparameter_search'] = results  # Cache for later use.
        return results  # Return to caller.

    def get_best_model(self, X_train: np.ndarray, y_train: np.ndarray) -> KNNClassifier:
        """
        Get the best model based on hyperparameter search.

        Args:
            X_train, y_train: Training data

        Returns:
            Fitted KNNClassifier with best parameters
        """
        if 'hyperparameter_search' not in self.results:
            raise RuntimeError("Run hyperparameter search first")  # Guard.

        best_params = self.results['hyperparameter_search']['best_params']  # Extract best params.
        model = KNNClassifier(
            k=best_params['n_neighbors'],
            metric=best_params['metric'],
            weights=best_params['weights'],
            p=best_params.get('p', 2)
        )
        model.fit(X_train, y_train)  # Fit best model on full train.
        return model  # Return fitted classifier.


def run_lite_benchmark(data_loader, save_dir: str = 'Code/A') -> Dict:
    """
    Run a lightweight k-NN training (no exhaustive benchmarks).
    Trains a single model with sensible defaults to save memory.

    Args:
        data_loader: BreastMNISTLoader instance
        save_dir: Directory to save results

    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)  # Section divider.
    print("K-NEAREST NEIGHBORS LITE MODE")  # Section title.
    print("="*70)  # Divider.

    # Load data
    train_images, train_labels = data_loader.get_train_data()  # Training split.
    val_images, val_labels = data_loader.get_val_data()  # Validation split.
    test_images, test_labels = data_loader.get_test_data()  # Test split.

    print(f"\nDataset sizes:")  # Header.
    print(f"  Train: {len(train_images)}")  # Train size.
    print(f"  Val: {len(val_images)}")  # Val size.
    print(f"  Test: {len(test_images)}")  # Test size.

    # Compare feature methods to find the best one.
    print("\nComparing feature extraction methods...")  # Status.
    feature_methods = ['raw', 'hog']  # Methods to compare.
    best_val_acc = -1
    best_feature = None
    best_extractor = None
    best_X_train, best_X_val, best_X_test = None, None, None

    for method in feature_methods:
        print(f"  Evaluating {method}...")  # Progress.
        extractor = FeatureExtractor(method=method)  # Build extractor.
        X_train = extractor.fit_transform(train_images)  # Fit + transform.
        X_val = extractor.transform(val_images)  # Transform val.
        X_test = extractor.transform(test_images)  # Transform test.

        # Quick evaluation with k=5.
        model = KNNClassifier(k=5)
        model.fit(X_train, train_labels)
        val_metrics = model.evaluate(X_val, val_labels)

        print(f"    Val Acc: {val_metrics['accuracy']:.4f}")  # Log result.
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_feature = method
            best_extractor = extractor
            best_X_train, best_X_val, best_X_test = X_train, X_val, X_test

    print(f"\nBest feature method: {best_feature} (Val Acc: {best_val_acc:.4f})")

    # Run hyperparameter search on best feature method.
    print("\nRunning hyperparameter search...")  # Status.
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }

    base_model = KNeighborsClassifier(n_jobs=1)  # Base sklearn model.
    grid_search = GridSearchCV(
        base_model, param_grid, cv=3,
        scoring='f1', n_jobs=1, verbose=1
    )

    # Combine train and val for cross-validation.
    X_combined = np.vstack([best_X_train, best_X_val])  # Stack features.
    y_combined = np.concatenate([train_labels, val_labels])  # Stack labels.
    grid_search.fit(X_combined, y_combined)  # Run grid search.

    best_params = grid_search.best_params_  # Extract best params.
    print(f"\nBest parameters: {best_params}")  # Log result.
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")  # Log score.

    # Train final model with best parameters.
    print("\nTraining k-NN with best parameters...")  # Status.
    model = KNNClassifier(
        k=best_params['n_neighbors'],
        metric=best_params['metric'],
        weights=best_params['weights']
    )
    model.fit(best_X_train, train_labels)  # Fit model.

    # Evaluate
    val_metrics = model.evaluate(best_X_val, val_labels)  # Validation metrics.
    final_metrics = model.evaluate(best_X_test, test_labels)  # Test metrics.

    print(f"\nValidation Results:")  # Header.
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")  # Val accuracy.
    print(f"  F1 Score:  {val_metrics['f1_score']:.4f}")  # Val F1.

    print(f"\nFinal Test Results:")  # Header.
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")  # Test accuracy.
    print(f"  Precision: {final_metrics['precision']:.4f}")  # Test precision.
    print(f"  Recall:    {final_metrics['recall']:.4f}")  # Test recall.
    print(f"  F1 Score:  {final_metrics['f1_score']:.4f}")  # Test F1.
    print(f"\nConfusion Matrix:")  # Header.
    print(final_metrics['confusion_matrix'])  # Matrix values.

    # Save model for later inference or comparison.
    os.makedirs(save_dir, exist_ok=True)  # Ensure output directory exists.
    model_path = os.path.join(save_dir, 'knn_best_model.pkl')  # Output path.
    model.save_model(model_path)  # Persist model.

    return {
        'final_metrics': final_metrics,
        'best_params': best_params,
        'best_feature_method': best_feature
    }


def run_full_benchmark(data_loader, save_dir: str = 'Code/A') -> Dict:
    """
    Run the complete k-NN benchmarking suite.

    Args:
        data_loader: BreastMNISTLoader instance
        save_dir: Directory to save results

    Returns:
        Dictionary with all benchmark results
    """
    print("\n" + "="*70)  # Section divider.
    print("K-NEAREST NEIGHBORS COMPREHENSIVE BENCHMARK")  # Section title.
    print("="*70)  # Divider.

    # Load data
    train_images, train_labels = data_loader.get_train_data()  # Training split.
    val_images, val_labels = data_loader.get_val_data()  # Validation split.
    test_images, test_labels = data_loader.get_test_data()  # Test split.

    print(f"\nDataset sizes:")  # Header.
    print(f"  Train: {len(train_images)}")  # Train size.
    print(f"  Val: {len(val_images)}")  # Val size.
    print(f"  Test: {len(test_images)}")  # Test size.

    benchmark = KNNBenchmark()  # Benchmark helper.

    # 1. Feature extraction comparison
    feature_results = benchmark.benchmark_feature_methods(
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels,
        k=5
    )

    # Find best feature method based on validation accuracy.
    best_feature_idx = np.argmax(feature_results['val_acc'])  # Best by val accuracy.
    best_feature = feature_results['method'][best_feature_idx]  # Method name.
    print(f"\nBest feature method: {best_feature}")  # Log result.

    # 2. Extract features using best method for remaining experiments.
    if 'pca' in best_feature or 'hog_pca' in best_feature:
        method = best_feature.rsplit('_', 1)[0]  # Base method name.
        n_comp = int(best_feature.rsplit('_', 1)[1])  # PCA components.
        extractor = FeatureExtractor(method=method, n_components=n_comp)  # Config extractor.
    else:
        extractor = FeatureExtractor(method=best_feature)  # Config extractor.

    X_train = extractor.fit_transform(train_images)  # Fit + transform train.
    X_val = extractor.transform(val_images)  # Transform val.
    X_test = extractor.transform(test_images)  # Transform test.

    # 3. Model capacity analysis (varying k)
    k_results = benchmark.benchmark_k_values(
        X_train, train_labels,
        X_val, val_labels,
        X_test, test_labels,
        k_values=[1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    )

    # 4. Hyperparameter grid search
    hp_results = benchmark.benchmark_hyperparameters(
        X_train, train_labels,
        X_val, val_labels
    )

    # 5. Training budget analysis
    budget_results = benchmark.benchmark_training_budget(
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels,
        feature_method='hog',  # Use HOG for consistency
        k=hp_results['best_params']['n_neighbors']
    )

    # 6. Final model evaluation
    print("\n" + "="*60)  # Section divider.
    print("FINAL MODEL EVALUATION")  # Section title.
    print("="*60)  # Divider.

    best_model = benchmark.get_best_model(X_train, train_labels)  # Refit best model.
    final_metrics = best_model.evaluate(X_test, test_labels)  # Final test metrics.

    print(f"\nFinal Test Results:")  # Header.
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")  # Test accuracy.
    print(f"  Precision: {final_metrics['precision']:.4f}")  # Test precision.
    print(f"  Recall:    {final_metrics['recall']:.4f}")  # Test recall.
    print(f"  F1 Score:  {final_metrics['f1_score']:.4f}")  # Test F1.
    print(f"\nConfusion Matrix:")  # Header.
    print(final_metrics['confusion_matrix'])  # Matrix values.

    # Save model
    os.makedirs(save_dir, exist_ok=True)  # Ensure output directory exists.
    model_path = os.path.join(save_dir, 'knn_best_model.pkl')  # Output path.
    best_model.save_model(model_path)  # Persist model.

    # Compile all results
    all_results = {
        'feature_comparison': feature_results,
        'k_value_analysis': k_results,
        'hyperparameter_search': hp_results,
        'training_budget': budget_results,
        'final_metrics': final_metrics,
        'best_feature_method': best_feature,
        'best_params': hp_results['best_params']
    }

    return all_results  # Return full benchmark suite.
