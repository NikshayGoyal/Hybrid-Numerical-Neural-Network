"""
Block-wise iterative solver for neural-network weight updates.

Strategy
--------
Each **layer** is treated as a separate optimisation block.  Within each
outer step the solver cycles through every layer and performs a local
gradient update while keeping all other layers frozen.  Because later layers
see the updated weights from earlier layers, this is mathematically a
**block Gauss-Seidel** scheme applied at the layer granularity.

The procedure for one outer step (with *cycles* = C):
    for c in 1 ... C:
        for each layer L_i in [L_1, L_2, ..., L_K]:
            1. Run forward + backward with current parameters
            2. Update only L_i:  W_i -= lr * dW_i,  b_i -= lr * db_i

Returning the loss from the **last** forward/backward pass.

This is especially useful for investigating layer-wise training dynamics
and serves as the "block Gauss-Seidel" baseline in comparisons against
the full Gauss-Seidel and Jacobi solvers.
"""

import numpy as np


class BlockIterative:
    """Block-wise (layer-level) iterative solver.

    Parameters
    ----------
    lr : float, default 0.01
        Learning rate applied to each per-layer gradient update.
    cycles : int, default 1
        Number of full passes over all layers per outer step.
    """

    def __init__(self, lr=0.01, cycles=1):
        self.lr = lr
        self.cycles = cycles
        self.name = 'Block-wise Iterative'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, network, X, y):
        """One block-iterative parameter update.

        Parameters
        ----------
        network : NeuralNetwork
        X : ndarray - input batch
        y : ndarray - targets

        Returns
        -------
        loss : float
            Loss from the **last** forward/backward pass executed during
            this step (i.e. after all block updates).
        """
        n_layers = len(network.layers)
        loss = None

        for _cycle in range(self.cycles):
            for layer_idx in range(n_layers):
                # -- 1. Full forward + backward with current params ----
                loss = network.backward(X, y)

                # -- 2. Update ONLY the current layer ------------------
                layer = network.layers[layer_idx]
                grads = network.grads[layer_idx]

                layer['W'] = layer['W'] - self.lr * grads['dW']
                layer['b'] = layer['b'] - self.lr * grads['db']

                # Other layers remain frozen for this sub-step.
                # On the next iteration of the inner loop the forward pass
                # will pick up the updated weights from this layer,
                # implementing the sequential Gauss-Seidel coupling.

        # Optionally run one final forward to report the up-to-date loss
        # after all blocks have been updated.
        if loss is None:
            # Edge case: zero layers (should never happen in practice)
            loss = network.backward(X, y)

        return loss

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"BlockIterative(lr={self.lr}, cycles={self.cycles})"
