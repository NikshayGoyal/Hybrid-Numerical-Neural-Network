"""
Evaluation Metrics
==================

Provides standard evaluation metrics for regression and classification tasks:
  - Mean Squared Error (MSE)
  - R-squared (R2) coefficient of determination
  - Accuracy (classification argmax or regression R2)
"""

import numpy as np


def compute_mse(y_pred, y_true):
    """Compute the Mean Squared Error between predictions and targets.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, any shape that broadcasts with *y_true*.
    y_true : np.ndarray
        Ground-truth values.

    Returns
    -------
    float
        Scalar MSE value.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_r2(y_pred, y_true):
    """Compute the R-squared (coefficient of determination) score.

    R2 = 1 - SS_res / SS_tot

    A perfect model yields R2 = 1.0.  A model that always predicts the
    mean of *y_true* yields R2 = 0.0.  Negative values indicate that the
    model is worse than predicting the mean.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    y_true : np.ndarray
        Ground-truth values.

    Returns
    -------
    float
        Scalar R2 value.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Guard against zero variance (constant target)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - ss_res / ss_tot)


def compute_accuracy(y_pred, y_true):
    """Compute an accuracy-like metric, automatically adapting to the task.

    Classification (multi-column targets)
        Returns the fraction of samples where ``argmax(y_pred)`` matches
        ``argmax(y_true)`` - i.e. top-1 classification accuracy.

    Regression (single-column or 1-D targets)
        Returns the R2 score, which can be interpreted as the proportion
        of variance explained by the model.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    y_true : np.ndarray
        Ground-truth values.

    Returns
    -------
    float
        Classification accuracy in [0, 1] **or** R2 score (can be negative).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Heuristic: if the last dimension has more than 1 column,
    # treat the task as classification.
    is_classification = y_true.ndim >= 2 and y_true.shape[-1] > 1

    if is_classification:
        pred_labels = np.argmax(y_pred, axis=-1)
        true_labels = np.argmax(y_true, axis=-1)
        return float(np.mean(pred_labels == true_labels))
    else:
        # Regression - return R2 score
        return compute_r2(y_pred, y_true)
