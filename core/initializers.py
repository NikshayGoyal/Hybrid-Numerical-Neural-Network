"""
core/initializers.py - Weight initialization strategies for the MLP.

Each initializer is a plain function with signature:
    init_fn(fan_in, fan_out) -> np.ndarray of shape (fan_in, fan_out)

The returned matrix contains the initial weight values for a single layer.
Biases are always initialized to zeros elsewhere (in NeuralNetwork.__init__).

A factory function ``get_initializer(name)`` returns the appropriate callable.
"""

import numpy as np


def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier / Glorot uniform initialization.

    Draws weights from U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
    Designed to keep variance roughly constant across layers when using
    sigmoid or tanh activations (Glorot & Bengio, 2010).

    Parameters
    ----------
    fan_in : int
        Number of input units (rows of the weight matrix).
    fan_out : int
        Number of output units (columns of the weight matrix).

    Returns
    -------
    np.ndarray, shape (fan_in, fan_out)
        Initialized weight matrix.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He (Kaiming) normal initialization.

    Draws weights from N(0, sqrt(2 / fan_in)).  Designed for layers using
    ReLU activations so that the variance of activations is preserved
    through the forward pass (He et al., 2015).

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.

    Returns
    -------
    np.ndarray, shape (fan_in, fan_out)
        Initialized weight matrix.
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


def random_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Simple random normal initialization with small standard deviation.

    Draws weights from N(0, 0.01).  This is a baseline initializer; it
    works for shallow networks but may cause vanishing/exploding gradients
    in deeper architectures.

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.

    Returns
    -------
    np.ndarray, shape (fan_in, fan_out)
        Initialized weight matrix.
    """
    return np.random.randn(fan_in, fan_out) * 0.01


# =============================================================================
# Factory
# =============================================================================
# Registry mapping canonical lowercase names to initializer functions.
_INITIALIZER_REGISTRY = {
    "xavier": xavier_init,
    "glorot": xavier_init,       # alias
    "he": he_init,
    "kaiming": he_init,          # alias
    "random": random_init,
}


def get_initializer(name: str):
    """Return a weight initialization function by name.

    Parameters
    ----------
    name : str
        One of 'xavier'/'glorot', 'he'/'kaiming', 'random'
        (case-insensitive).

    Returns
    -------
    callable
        A function with signature ``fn(fan_in, fan_out) -> np.ndarray``.

    Raises
    ------
    ValueError
        If ``name`` is not found in the registry.
    """
    key = name.strip().lower()
    if key not in _INITIALIZER_REGISTRY:
        raise ValueError(
            f"Unknown initializer '{name}'. "
            f"Available: {list(_INITIALIZER_REGISTRY.keys())}"
        )
    return _INITIALIZER_REGISTRY[key]
