"""
Gradient Descent solver with optional Nesterov-style momentum.

This is the baseline optimiser for the Hybrid Numerical Neural Network
project.  It supports:
  * Vanilla SGD  (momentum=0.0)
  * Classical momentum  (momentum > 0.0)

Momentum update rule (per-parameter):
    v_t  = momentum * v_{t-1} - lr * grad_t
    theta_t  = theta_{t-1} + v_t

where theta represents any weight matrix W or bias vector b.
"""

import numpy as np


class GradientDescent:
    """Stochastic Gradient Descent with optional momentum.

    Parameters
    ----------
    lr : float, default 0.01
        Learning rate (step size).
    momentum : float, default 0.0
        Momentum coefficient in [0, 1). Set to 0 for vanilla SGD.

    Attributes
    ----------
    velocity : list of dict or None
        Per-layer velocity buffers; lazily initialised on the first call
        to :meth:`step`.
    name : str
        Human-readable solver name (used in logs and plots).
    """

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.name = 'Gradient Descent'
        self.velocity = None  # initialised lazily on first step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, network, X, y):
        """Perform one gradient-descent parameter update.

        Parameters
        ----------
        network : NeuralNetwork
            The network whose parameters will be updated in-place.
        X : ndarray, shape (n_samples, n_features)
            Input batch.
        y : ndarray
            Target values corresponding to *X*.

        Returns
        -------
        loss : float
            Scalar loss **before** the parameter update (the value returned
            by ``network.backward``).
        """
        # 1. Forward + backward -> fills network.grads and returns loss
        loss = network.backward(X, y)

        # 2. Lazy-initialise velocity buffers (zeros matching each param)
        if self.velocity is None:
            self._init_velocity(network)

        # 3. Update every layer
        for idx, layer in enumerate(network.layers):
            for key in ('W', 'b'):
                grad = network.grads[idx]['d' + key]

                # Classical momentum: v = mu*v - lr*nabla
                self.velocity[idx][key] = (
                    self.momentum * self.velocity[idx][key] - self.lr * grad
                )

                # Parameter update: theta += v
                layer[key] += self.velocity[idx][key]

        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_velocity(self, network):
        """Create zero-filled velocity buffers that mirror the network params."""
        self.velocity = []
        for layer in network.layers:
            self.velocity.append({
                'W': np.zeros_like(layer['W']),
                'b': np.zeros_like(layer['b']),
            })

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"GradientDescent(lr={self.lr}, momentum={self.momentum})"
        )
