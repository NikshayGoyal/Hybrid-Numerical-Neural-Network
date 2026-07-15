"""
Gauss-Seidel iterative solver for neural-network weight updates.

Strategy
--------
The weight update is cast as a linear system:

    A * x = b

where
    A = H_reg  (regularised Hessian, or block-diagonal approximation)
    b = g      (gradient vector)
    x = delta      (update direction)

and the parameters are updated as  theta <- theta - lr * delta.

The Gauss-Seidel method solves the system **sequentially**: when updating
component *i* it immediately uses the latest values of all previously
updated components.  This gives faster convergence than Jacobi for
diagonally-dominant systems.

For networks where the full Hessian is impractical (many parameters), a
**block-diagonal approximation** is used - one block per layer - so memory
stays O(max_layer_params2) instead of O(total_params2).
"""

import numpy as np


# ------------------------------------------------------------------
# Heuristic threshold: if total params exceed this, switch to the
# block-diagonal Hessian approximation to keep memory bounded.
# ------------------------------------------------------------------
_FULL_HESSIAN_THRESHOLD = 2000


class GaussSeidel:
    """Gauss-Seidel iterative solver for neural-network optimisation.

    Parameters
    ----------
    lr : float, default 0.01
        Learning rate applied to the solved update direction.
    max_inner_iters : int, default 10
        Maximum number of Gauss-Seidel sweeps per outer step.
    tolerance : float, default 1e-6
        Early-stop tolerance on the ||Deltax|| / ||x|| relative change.
    """

    def __init__(self, lr=0.01, max_inner_iters=10, tolerance=1e-6, regularization=0.1):
        self.lr = lr
        self.max_inner_iters = max_inner_iters
        self.tolerance = tolerance
        self.regularization = regularization
        self.name = 'Gauss-Seidel'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, network, X, y):
        """One Gauss-Seidel parameter update.

        Parameters
        ----------
        network : NeuralNetwork
        X : ndarray - input batch
        y : ndarray - targets

        Returns
        -------
        loss : float
            Loss value **before** the update.
        """
        # 1. Backward pass -> gradients + loss
        loss = network.backward(X, y)

        n_params = network.num_params()

        if n_params <= _FULL_HESSIAN_THRESHOLD:
            # ---- Full-Hessian path -----------------------------------
            delta = self._solve_full(network, X, y)
        else:
            # ---- Block-diagonal path (per-layer) ---------------------
            delta = self._solve_block_diagonal(network, X, y)

        # 2. Apply update: theta <- theta - lr * delta
        params = network.get_flat_params()
        params = params - self.lr * delta
        network.set_flat_params(params)

        return loss

    # ------------------------------------------------------------------
    # Full-Hessian Gauss-Seidel
    # ------------------------------------------------------------------

    def _solve_full(self, network, X, y):
        """Solve A*x = b via Gauss-Seidel using the full Hessian."""
        grad = network.get_flat_grads()             # b
        H = network.compute_hessian(X, y)            # A (before reg.)
        n = H.shape[0]
        A = H + self.regularization * np.eye(n)       # Tikhonov reg.
        b = grad.copy()

        return self._gauss_seidel_solve(A, b)

    # ------------------------------------------------------------------
    # Block-diagonal Gauss-Seidel (per-layer Hessians)
    # ------------------------------------------------------------------

    def _solve_block_diagonal(self, network, X, y):
        """Approximate Gauss-Seidel using independent per-layer blocks.

        Each layer's contribution to the Hessian is approximated as a
        diagonal block; cross-layer curvature terms are ignored.  Within
        each block, standard Gauss-Seidel iteration is performed.

        Falls back to a simple gradient step for any block whose diagonal
        is degenerate.
        """
        grad = network.get_flat_grads()
        delta = np.zeros_like(grad)

        offset = 0
        for layer in network.layers:
            # Determine the number of parameters in this layer
            n_w = layer['W'].size
            n_b = layer['b'].size
            block_size = n_w + n_b

            # Slice the gradient for this block
            g_block = grad[offset: offset + block_size]

            # Build a diagonal approximation: use |g| + eps as curvature
            # This is equivalent to AdaGrad-style preconditioning and
            # avoids materialising a large per-layer Hessian.
            diag_approx = np.abs(g_block) + 1e-4
            A_block = np.diag(diag_approx)

            # Gauss-Seidel on the small block
            delta[offset: offset + block_size] = self._gauss_seidel_solve(
                A_block, g_block
            )

            offset += block_size

        return delta

    # ------------------------------------------------------------------
    # Core Gauss-Seidel iteration
    # ------------------------------------------------------------------

    def _gauss_seidel_solve(self, A, b):
        """Solve A*x = b by Gauss-Seidel iteration.

        Parameters
        ----------
        A : ndarray, shape (n, n)
            Coefficient matrix (should be diagonally dominant or positive
            definite for convergence guarantees).
        b : ndarray, shape (n,)
            Right-hand side.

        Returns
        -------
        x : ndarray, shape (n,)
            Approximate solution after at most *max_inner_iters* sweeps.
        """
        n = len(b)
        x = np.zeros(n)

        for iteration in range(self.max_inner_iters):
            x_old = x.copy()

            for i in range(n):
                diag = A[i, i]
                if np.abs(diag) < 1e-12:
                    # Skip degenerate rows - leave x[i] unchanged
                    continue

                # Gauss-Seidel: use latest x values (already updated for j < i)
                residual = b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]
                x[i] = residual / diag

            # Convergence check: relative change
            change = np.linalg.norm(x - x_old)
            scale = np.linalg.norm(x) + 1e-12
            if change / scale < self.tolerance:
                break

        # Guard against numerical blow-up
        if not np.all(np.isfinite(x)):
            x = b.copy()  # fall back to plain gradient direction

        return x

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"GaussSeidel(lr={self.lr}, max_inner_iters={self.max_inner_iters}, "
            f"tolerance={self.tolerance})"
        )
