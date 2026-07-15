"""
Data Loading Utilities
======================

Provides functions to generate synthetic datasets and load standard
classification benchmarks for training and evaluating neural networks.

All loaders return normalized features with train/test splits ready
for immediate use with the network API.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_synthetic_regression(n_samples=500, noise_std=0.1, test_size=0.2, seed=42):
    """Generate a sine-wave regression dataset.

    Creates random input points in [-2pi, 2pi] and computes
    ``y = sin(x) + eps`` where eps ~ N(0, noise_std2).

    Parameters
    ----------
    n_samples : int, default=500
        Total number of data points to generate.
    noise_std : float, default=0.1
        Standard deviation of additive Gaussian noise on the targets.
    test_size : float, default=0.2
        Fraction of data reserved for the test split.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, 1)
        Normalized training inputs.
    X_test : np.ndarray, shape (n_test, 1)
        Normalized test inputs (using the *training* scaler).
    y_train : np.ndarray, shape (n_train, 1)
        Training targets (un-normalized).
    y_test : np.ndarray, shape (n_test, 1)
        Test targets (un-normalized).
    """
    rng = np.random.RandomState(seed)

    # Generate random x values uniformly in [-2pi, 2pi]
    X = rng.uniform(-2 * np.pi, 2 * np.pi, size=(n_samples, 1))

    # Compute noisy sine targets
    y = np.sin(X) + rng.normal(0, noise_std, size=(n_samples, 1))

    # Train / test split (stratify is not applicable for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Normalize inputs using StandardScaler (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_classification(dataset='iris', test_size=0.2, seed=42):
    """Load a standard classification dataset with one-hot encoded labels.

    Currently supports the Iris dataset via scikit-learn.

    Parameters
    ----------
    dataset : str, default='iris'
        Name of the dataset to load.  Only ``'iris'`` is supported.
    test_size : float, default=0.2
        Fraction of data reserved for the test split.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, n_features)
        Normalized training features.
    X_test : np.ndarray, shape (n_test, n_features)
        Normalized test features.
    y_train : np.ndarray, shape (n_train, n_classes)
        One-hot encoded training labels.
    y_test : np.ndarray, shape (n_test, n_classes)
        One-hot encoded test labels.

    Raises
    ------
    ValueError
        If an unsupported dataset name is provided.
    """
    if dataset.lower() != 'iris':
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Currently only 'iris' is available."
        )

    # Load raw data
    data = load_iris()
    X, y = data.data, data.target  # X: (150,4), y: (150,)

    # Determine number of classes
    num_classes = len(np.unique(y))

    # One-hot encode the integer labels
    y_onehot = one_hot_encode(y, num_classes)

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=seed, stratify=y
    )

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def one_hot_encode(y, num_classes):
    """Convert an array of integer class labels to one-hot vectors.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Integer class labels in [0, num_classes).
    num_classes : int
        Total number of classes.

    Returns
    -------
    one_hot : np.ndarray, shape (n, num_classes)
        One-hot encoded matrix.
    """
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot
