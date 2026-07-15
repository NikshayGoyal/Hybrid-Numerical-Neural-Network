"""
core/network.py - Fully-connected Multi-Layer Perceptron built from scratch in NumPy.

This is the central module of the Hybrid Numerical Neural Network project.
It provides a ``NeuralNetwork`` class that supports:
    * Forward propagation with cached intermediate values.
    * Back-propagation to compute parameter gradients.
    * Flat parameter / gradient vectors for second-order optimizers.
    * Full Hessian computation via centered finite differences.
    * Deep-copy for safe trial updates (used by line-search, trust-region, etc.).

Design decisions
----------------
* Weight matrices have shape (fan_in, fan_out) so that the forward pass is
  simply  Z = A_prev @ W + b  (no transposes needed, and data stays in
  row-major / C-contiguous layout for cache efficiency).
* Biases are stored as row vectors of shape (1, fan_out).
* Hidden layers use the user-specified activation; the output layer uses
  Linear (identity) for regression (MSE loss) or is left as logits for
  classification (CrossEntropyLoss applies softmax internally).
"""

import copy
from typing import Dict, List, Optional

import numpy as np

from .activations import Linear, get_activation
from .initializers import get_initializer
from .losses import get_loss


class NeuralNetwork:
    """NumPy-only MLP with full forward/backward, Hessian, and flat-param API.

    Parameters
    ----------
    layer_sizes : list of int
        Widths of each layer including input and output,
        e.g. [2, 64, 32, 1].
    activation : str, default 'relu'
        Activation function name for **hidden** layers.
    loss : str, default 'mse'
        Loss function name ('mse' or 'cross_entropy').
    initializer : str, default 'he'
        Weight initialization strategy ('he', 'xavier', 'random').

    Attributes
    ----------
    layers : list of dict
        Each element is ``{'W': ndarray(fan_in, fan_out),
                           'b': ndarray(1, fan_out)}``.
    activation : object
        Activation instance with ``forward`` / ``backward`` methods.
    output_activation : object
        Activation used on the output layer (Linear for MSE, identity
        passthrough for cross-entropy whose softmax lives in the loss).
    loss_fn : object
        Loss instance with ``forward`` / ``backward`` methods.
    cache : dict or None
        Populated during ``forward()`` with keys 'Z' and 'A'.
    grads : list of dict or None
        Populated during ``backward()`` - same structure as ``layers``.
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        loss: str = "mse",
        initializer: str = "he",
    ) -> None:
        """Build the network and initialize all parameters.

        Parameters
        ----------
        layer_sizes : list of int
            e.g. [2, 64, 32, 1].  First element is the input dimension,
            last element is the output dimension.
        activation : str
            Activation function name for hidden layers.
        loss : str
            Loss function name.
        initializer : str
            Weight initialization strategy.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least an input and output size.")

        self.layer_sizes = list(layer_sizes)
        self.activation = get_activation(activation)
        self.loss_fn = get_loss(loss)

        # Output layer always uses Linear (identity) activation.
        # For cross-entropy, the softmax is baked into the loss's backward().
        self.output_activation = Linear()

        # ---- Initialize parameters ---- #
        init_fn = get_initializer(initializer)
        self.layers: List[Dict[str, np.ndarray]] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self.layers.append(
                {
                    "W": init_fn(fan_in, fan_out),
                    "b": np.zeros((1, fan_out)),
                }
            )

        # Placeholders - populated by forward() and backward().
        self.cache: Optional[Dict[str, List[np.ndarray]]] = None
        self.grads: Optional[List[Dict[str, np.ndarray]]] = None

    # --------------------------------------------------------------------- #
    #  Forward Pass
    # --------------------------------------------------------------------- #
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through every layer.

        Caches pre-activation (Z) and post-activation (A) tensors so that
        ``backward()`` can compute gradients without re-computing them.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            Input data batch.

        Returns
        -------
        np.ndarray, shape (m, output_dim)
            Network predictions (linear output for regression, logits for
            classification).
        """
        # A[0] is the network input.
        Z_list: List[np.ndarray] = []
        A_list: List[np.ndarray] = [X]

        A = X
        n_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            Z = A @ layer["W"] + layer["b"]   # (m, fan_out)
            Z_list.append(Z)

            if i < n_layers - 1:
                # Hidden layer -> use the specified activation.
                A = self.activation.forward(Z)
            else:
                # Output layer -> linear (identity) activation.
                A = self.output_activation.forward(Z)

            A_list.append(A)

        # Store for backprop.
        self.cache = {"Z": Z_list, "A": A_list}
        return A

    # --------------------------------------------------------------------- #
    #  Backward Pass
    # --------------------------------------------------------------------- #
    def backward(self, X: np.ndarray, y: np.ndarray) -> float:
        """Back-propagation: compute gradients for all parameters.

        Must be called immediately after ``forward(X)`` so that ``self.cache``
        is populated.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            The same input batch used in the preceding ``forward()`` call.
        y : np.ndarray, shape (m, output_dim)
            Ground-truth targets.

        Returns
        -------
        float
            Scalar loss value for this batch.

        Side Effects
        ------------
        Populates ``self.grads`` - a list of dicts with keys 'dW' and 'db',
        one per layer, aligned with ``self.layers``.
        """
        # Always re-run forward pass so activations match current weights.
        self.forward(X)

        Z_list = self.cache["Z"]
        A_list = self.cache["A"]   # A[0] = X, A[i+1] = post-activation of layer i
        m = X.shape[0]
        n_layers = len(self.layers)

        # Get the network output (last element of A_list).
        y_pred = A_list[-1]

        # Compute loss value.
        loss_val = self.loss_fn.forward(y_pred, y)

        # Gradient of the loss w.r.t. the output activations.
        # For MSE  : dL/dA_out = (y_pred - y) / m
        # For CE   : dL/d(logits) = (softmax(y_pred) - y) / m  (combined)
        dZ = self.loss_fn.backward(y_pred, y)

        # For the output layer with Linear activation, the derivative is 1,
        # so dZ already equals dL/dZ_last.  Multiply anyway for generality:
        dZ = dZ * self.output_activation.backward(Z_list[-1])

        # Allocate gradient storage - each element must be its own dict.
        grads: List[Dict[str, np.ndarray]] = [{} for _ in range(n_layers)]

        for i in reversed(range(n_layers)):
            A_prev = A_list[i]  # activation *entering* layer i

            # loss.backward already includes 1/m, so no extra division.
            dW = A_prev.T @ dZ                          # (fan_in, fan_out)
            db = np.sum(dZ, axis=0, keepdims=True)      # (1, fan_out)

            grads[i] = {"dW": dW, "db": db}

            if i > 0:
                # Propagate gradient to the previous layer.
                dA_prev = dZ @ self.layers[i]["W"].T       # (m, fan_in)
                dZ = dA_prev * self.activation.backward(Z_list[i - 1])

        self.grads = grads
        return float(loss_val)

    # --------------------------------------------------------------------- #
    #  Flat Parameter Interface
    # --------------------------------------------------------------------- #
    def get_flat_params(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D vector.

        Order: W_0, b_0, W_1, b_1, ... (each sub-array is row-major flattened).

        Returns
        -------
        np.ndarray, shape (num_params,)
            Concatenated parameter vector.
        """
        parts: List[np.ndarray] = []
        for layer in self.layers:
            parts.append(layer["W"].ravel())
            parts.append(layer["b"].ravel())
        return np.concatenate(parts)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Restore weights and biases from a flat 1-D vector.

        Parameters
        ----------
        flat_params : np.ndarray, shape (num_params,)
            Must have the same length as ``self.num_params()``.
        """
        if flat_params.size != self.num_params():
            raise ValueError(
                f"Expected {self.num_params()} parameters, "
                f"got {flat_params.size}."
            )

        offset = 0
        for layer in self.layers:
            # Weights
            w_size = layer["W"].size
            layer["W"] = flat_params[offset : offset + w_size].reshape(layer["W"].shape)
            offset += w_size
            # Biases
            b_size = layer["b"].size
            layer["b"] = flat_params[offset : offset + b_size].reshape(layer["b"].shape)
            offset += b_size

    def get_flat_grads(self) -> np.ndarray:
        """Flatten all gradients into a single 1-D vector.

        Same ordering convention as ``get_flat_params()``.

        Returns
        -------
        np.ndarray, shape (num_params,)
            Concatenated gradient vector.

        Raises
        ------
        RuntimeError
            If ``backward()`` has not been called yet.
        """
        if self.grads is None:
            raise RuntimeError("backward() must be called before get_flat_grads().")

        parts: List[np.ndarray] = []
        for g in self.grads:
            parts.append(g["dW"].ravel())
            parts.append(g["db"].ravel())
        return np.concatenate(parts)

    def num_params(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
            Sum of all weight and bias elements across every layer.
        """
        total = 0
        for layer in self.layers:
            total += layer["W"].size + layer["b"].size
        return total

    # --------------------------------------------------------------------- #
    #  Hessian Computation (Finite Differences)
    # --------------------------------------------------------------------- #
    def compute_hessian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Approximate the full Hessian matrix via centered finite differences.

        For each parameter index j, perturb p_j by +-eps, recompute the full
        gradient vector, and build the j-th column of the Hessian as:

            H[:, j] = (grad(theta + eps*e_j) - grad(theta - eps*e_j)) / (2eps)

        This is an O(P2*m) operation - feasible only for small networks.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            Input data.
        y : np.ndarray, shape (m, output_dim)
            Target data.

        Returns
        -------
        np.ndarray, shape (P, P)
            Symmetric Hessian matrix where P = ``self.num_params()``.
        """
        eps = 1e-5
        P = self.num_params()
        theta = self.get_flat_params().copy()
        H = np.zeros((P, P))

        for j in range(P):
            # --- Perturb +eps ---
            theta_plus = theta.copy()
            theta_plus[j] += eps
            self.set_flat_params(theta_plus)
            self.forward(X)
            self.backward(X, y)
            grad_plus = self.get_flat_grads().copy()

            # --- Perturb -eps ---
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            self.set_flat_params(theta_minus)
            self.forward(X)
            self.backward(X, y)
            grad_minus = self.get_flat_grads().copy()

            # --- Centered difference ---
            H[:, j] = (grad_plus - grad_minus) / (2.0 * eps)

        # Restore original parameters.
        self.set_flat_params(theta)

        # Symmetrize to reduce floating-point asymmetry.
        H = 0.5 * (H + H.T)
        return H

    # --------------------------------------------------------------------- #
    #  Inference & Utilities
    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass **without** caching intermediates.

        Use this for inference / evaluation where back-propagation is not
        needed.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            Input data.

        Returns
        -------
        np.ndarray, shape (m, output_dim)
            Network predictions.
        """
        A = X
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            Z = A @ layer["W"] + layer["b"]
            if i < n_layers - 1:
                A = self.activation.forward(Z)
            else:
                A = self.output_activation.forward(Z)
        return A

    def copy(self) -> "NeuralNetwork":
        """Return an independent deep copy of this network.

        Useful for trial parameter updates (e.g. line-search) that should
        not modify the original network until the update is accepted.

        Returns
        -------
        NeuralNetwork
            A new instance with identical architecture and parameter values,
            sharing no mutable state with the original.
        """
        return copy.deepcopy(self)

    # --------------------------------------------------------------------- #
    #  Representation
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        """Concise string representation for debugging."""
        arch = " -> ".join(str(s) for s in self.layer_sizes)
        return (
            f"NeuralNetwork(arch=[{arch}], "
            f"params={self.num_params():,}, "
            f"activation={self.activation.__class__.__name__}, "
            f"loss={self.loss_fn.__class__.__name__})"
        )
