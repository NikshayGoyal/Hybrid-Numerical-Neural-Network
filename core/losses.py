"""
core/losses.py - Loss function implementations for the MLP.

Each loss is a class with two methods:
    forward(y_pred, y_true)  - compute the scalar loss value.
    backward(y_pred, y_true) - compute the gradient of the loss w.r.t. y_pred,
                                returned as an ndarray with the same shape as y_pred.

A factory function ``get_loss(name)`` returns an instance by string name.
"""

import numpy as np


# =============================================================================
# Mean Squared Error Loss
# =============================================================================
class MSELoss:
    """Mean Squared Error loss for regression tasks.

    L = (1 / 2m) * Sum (y_pred - y_true)2

    The factor of 1/2 is included so that the gradient is simply
    (y_pred - y_true) / m, which keeps downstream code cleaner.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the MSE loss.

        Parameters
        ----------
        y_pred : np.ndarray, shape (m, d)
            Network predictions.
        y_true : np.ndarray, shape (m, d)
            Ground-truth targets (same shape as y_pred).

        Returns
        -------
        float
            Scalar loss value  (1/2m) * ||y_pred - y_true||2_F.
        """
        m = y_pred.shape[0]
        return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradient of MSE w.r.t. y_pred.

        Parameters
        ----------
        y_pred : np.ndarray, shape (m, d)
            Network predictions.
        y_true : np.ndarray, shape (m, d)
            Ground-truth targets.

        Returns
        -------
        np.ndarray, shape (m, d)
            dL/d(y_pred) = (y_pred - y_true) / m.
        """
        m = y_pred.shape[0]
        return (y_pred - y_true) / m


# =============================================================================
# Cross-Entropy Loss (with built-in softmax)
# =============================================================================
class CrossEntropyLoss:
    """Categorical cross-entropy loss with built-in softmax.

    Expects raw logits as ``y_pred`` and one-hot encoded labels as ``y_true``.
    Applies softmax internally for numerical stability (log-sum-exp trick)
    and returns the combined gradient (softmax_output - y_true) / m directly,
    bypassing the need for a separate softmax backward pass.

    L = -(1/m) * Sum_i Sum_c  y_true[i,c] * log(softmax(y_pred)[i,c])
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the cross-entropy loss from logits.

        Parameters
        ----------
        y_pred : np.ndarray, shape (m, C)
            Raw logits (pre-softmax) for C classes.
        y_true : np.ndarray, shape (m, C)
            One-hot encoded ground-truth labels.

        Returns
        -------
        float
            Scalar cross-entropy loss.
        """
        m = y_pred.shape[0]

        # Numerically stable log-softmax via the log-sum-exp trick.
        shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp  # shape (m, C)

        # Cross-entropy: -mean of log-probabilities at the true class.
        loss = -np.sum(y_true * log_probs) / m
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Combined softmax + cross-entropy gradient w.r.t. logits.

        Parameters
        ----------
        y_pred : np.ndarray, shape (m, C)
            Raw logits (pre-softmax).
        y_true : np.ndarray, shape (m, C)
            One-hot encoded ground-truth labels.

        Returns
        -------
        np.ndarray, shape (m, C)
            dL/d(logits) = (softmax(y_pred) - y_true) / m.
        """
        m = y_pred.shape[0]

        # Compute softmax probabilities.
        shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        softmax_probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        return (softmax_probs - y_true) / m


# =============================================================================
# Factory
# =============================================================================
# Registry mapping canonical lowercase names to loss classes.
_LOSS_REGISTRY = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "crossentropy": CrossEntropyLoss,
}


def get_loss(name: str):
    """Return a loss function instance by name.

    Parameters
    ----------
    name : str
        One of 'mse', 'cross_entropy' / 'crossentropy'
        (case-insensitive).

    Returns
    -------
    object
        A loss instance with ``forward`` and ``backward`` methods.

    Raises
    ------
    ValueError
        If ``name`` is not found in the registry.
    """
    key = name.strip().lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. "
            f"Available: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[key]()
