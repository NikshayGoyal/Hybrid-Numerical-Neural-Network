"""
core/activations.py - Activation function implementations for the MLP.

Each activation is a class with two methods:
    forward(x)   - element-wise activation applied during the forward pass.
    backward(x)  - element-wise derivative (evaluated at the *pre-activation* z),
                    used during back-propagation to compute dZ = dA * act.backward(Z).

A factory function ``get_activation(name)`` returns an instance by string name.
"""

import numpy as np


# =============================================================================
# ReLU
# =============================================================================
class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU element-wise.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values (any shape).

        Returns
        -------
        np.ndarray
            Element-wise max(0, x), same shape as input.
        """
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU evaluated at the pre-activation values.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values (the same Z stored during forward pass).

        Returns
        -------
        np.ndarray
            1 where x > 0, 0 elsewhere (sub-gradient at x=0 is 0).
        """
        return (x > 0).astype(x.dtype)


# =============================================================================
# Sigmoid
# =============================================================================
class Sigmoid:
    """Logistic sigmoid: f(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the sigmoid function element-wise.

        Uses a numerically stable formulation that avoids overflow in exp()
        by splitting into positive and negative branches.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Sigmoid activations in (0, 1), same shape as input.
        """
        # Clip to prevent overflow warnings; values beyond +-500 saturate anyway.
        x_safe = np.clip(x, -500, 500)
        return np.where(
            x_safe >= 0,
            1.0 / (1.0 + np.exp(-x_safe)),
            np.exp(x_safe) / (1.0 + np.exp(x_safe)),
        )

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative: sigmoid(x) * (1 - sigmoid(x)).

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Element-wise derivative of the sigmoid.
        """
        s = self.forward(x)
        return s * (1.0 - s)


# =============================================================================
# Tanh
# =============================================================================
class Tanh:
    """Hyperbolic tangent activation: f(x) = tanh(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply tanh element-wise.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Tanh activations in (-1, 1).
        """
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative: 1 - tanh(x)^2.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Element-wise derivative of tanh.
        """
        t = np.tanh(x)
        return 1.0 - t ** 2


# =============================================================================
# Softmax
# =============================================================================
class Softmax:
    """Softmax activation for multi-class classification output layers.

    The backward pass for softmax is handled jointly with cross-entropy loss
    (the combined gradient simplifies to y_pred - y_true), so ``backward``
    raises NotImplementedError to prevent accidental misuse.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax row-wise (each sample is a row).

        Subtracts the row-wise maximum for numerical stability before
        exponentiating.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_classes)
            Pre-activation logits.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Probability distributions (rows sum to 1).
        """
        # Shift for numerical stability.
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Not implemented - use the combined softmax + cross-entropy gradient.

        Raises
        ------
        NotImplementedError
            Always; backward is handled inside CrossEntropyLoss.
        """
        raise NotImplementedError(
            "Softmax backward is handled jointly with CrossEntropyLoss. "
            "Do not call this method directly."
        )


# =============================================================================
# Linear (Identity)
# =============================================================================
class Linear:
    """Identity activation: f(x) = x.

    Used as the output-layer activation for regression tasks so that the
    network can produce unbounded predictions.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the input unchanged.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Same array (identity mapping).
        """
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the identity is 1 everywhere.

        Parameters
        ----------
        x : np.ndarray
            Pre-activation values (unused, but kept for API consistency).

        Returns
        -------
        np.ndarray
            Array of ones with the same shape as x.
        """
        return np.ones_like(x)


# =============================================================================
# Factory
# =============================================================================
# Registry mapping canonical lowercase names to activation classes.
_ACTIVATION_REGISTRY = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "linear": Linear,
}


def get_activation(name: str):
    """Return an activation function instance by name.

    Parameters
    ----------
    name : str
        One of 'relu', 'sigmoid', 'tanh', 'softmax', 'linear'
        (case-insensitive).

    Returns
    -------
    object
        An activation instance with ``forward`` and ``backward`` methods.

    Raises
    ------
    ValueError
        If ``name`` is not found in the registry.
    """
    key = name.strip().lower()
    if key not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(_ACTIVATION_REGISTRY.keys())}"
        )
    return _ACTIVATION_REGISTRY[key]()
