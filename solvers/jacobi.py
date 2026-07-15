"""
Jacobi iterative solver for neural-network weight updates.

Strategy
--------
Same formulation as the Gauss-Seidel solver - the weight update is cast as
a linear system  A * x = b  - but the Jacobi method updates **all**
components simultaneously using the values from the **previous** iteration.

This makes Jacobi naturally parallelisable (each component is independent
within an iteration), at the cost of typically slower convergence than
Gauss-Seidel for diagonally-dominant systems.

Update per iteration
    x_i^{k+1} = (b_i - Sum_{j!=i} A_{i,j} * x_j^{k}) / A_{i,i}

After convergence (or hitting *max_inner_iters*):
    theta <- theta - lr * x

For large networks the same block-diagonal Hessian approximation used by
the Gauss-Seidel solver is applied.
"""

import numpy as np


# ------------------------------------------------------------------
# Heuristic threshold for switching to block-diagonal approximation
# ------------------------------------------------------------------
_FULL_HESSIAN_THRESHOLD = 2000


class Jacobi:
    """Jacobi iterative solver for neural-network optimisation.

    Parameters
    ----------
    lr : float, default 0.01
        Learning rate applied to the solved update direction.
    max_inner_iters : int, default 10
        Maximum number of Jacobi sweeps per outer step.
    tolerance : float, default 1e-6
        Early-stop tolerance on ||x^{k+1} - x^{k}|| / ||x^{k+1}||.
    """

    def __init__(self, lr=0.01, max_inner_iters=10, tolerance=1e-6, regularization=0.1):
        self.lr = lr
        self.max_inner_iters = max_inner_iters
        self.tolerance = tolerance
        self.regularization = regularization
        self.name = 'Jacobi'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, network, X, y):
        """One Jacobi parameter update.

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
            delta = self._solve_full(network, X, y)
        else:
            delta = self._solve_block_diagonal(network, X, y)

        # 2. Apply update: theta <- theta - lr * delta
        params = network.get_flat_params()
        params = params - self.lr * delta
        network.set_flat_params(params)

        return loss

    # ------------------------------------------------------------------
    # Full-Hessian Jacobi
    # ------------------------------------------------------------------

    def _solve_full(self, network, X, y):
        """Solve A*x = b via Jacobi using the full Hessian."""
        grad = network.get_flat_grads()
        H = network.compute_hessian(X, y)
        n = H.shape[0]
        A = H + self.regularization * np.eye(n)   # Tikhonov regularisation
        b = grad.copy()

        return self._jacobi_solve(A, b)

    # ------------------------------------------------------------------
    # Block-diagonal Jacobi (per-layer)
    # ------------------------------------------------------------------

    def _solve_block_diagonal(self, network, X, y):
        """Approximate Jacobi using independent per-layer blocks.

        Same block-diagonal strategy as the Gauss-Seidel solver: each
        layer is treated as an independent block and cross-layer curvature
        is ignored.
        """
        grad = network.get_flat_grads()
        delta = np.zeros_like(grad)

        offset = 0
        for layer in network.layers:
            n_w = layer['W'].size
            n_b = layer['b'].size
            block_size = n_w + n_b

            g_block = grad[offset: offset + block_size]

            # Diagonal curvature approximation (|g| + eps)
            diag_approx = np.abs(g_block) + 1e-4
            A_block = np.diag(diag_approx)

            delta[offset: offset + block_size] = self._jacobi_solve(
                A_block, g_block
            )

            offset += block_size

        return delta

    # ------------------------------------------------------------------
    # Core Jacobi iteration
    # ------------------------------------------------------------------

    def _jacobi_solve(self, A, b):
        """Solve A*x = b by Jacobi iteration.

        Parameters
        ----------
        A : ndarray, shape (n, n)
            Coefficient matrix.
        b : ndarray, shape (n,)
            Right-hand side.

        Returns
        -------
        x : ndarray, shape (n,)
            Approximate solution.
        """
        n = len(b)
        x = np.zeros(n)

        # Pre-extract the diagonal for efficiency
        diag = np.diag(A).copy()
        # Mask near-zero diagonals to avoid division issues
        safe_diag = np.where(np.abs(diag) < 1e-12, 1.0, diag)

        for iteration in range(self.max_inner_iters):
            # Compute all new values from the OLD x (Jacobi property)
            # x_new[i] = (b[i] - sum_{j!=i} A[i,j]*x[j]) / A[i,i]
            #           = (b[i] - A[i,:] @ x + A[i,i]*x[i]) / A[i,i]
            # Vectorised form:
            Ax = A @ x                       # shape (n,)
            x_new = (b - Ax + diag * x) / safe_diag

            # For rows with degenerate diagonal, keep old value
            degenerate = np.abs(diag) < 1e-12
            x_new[degenerate] = x[degenerate]

            # Convergence check
            change = np.linalg.norm(x_new - x)
            scale = np.linalg.norm(x_new) + 1e-12
            if change / scale < self.tolerance:
                x = x_new
                break

            x = x_new

        # Guard against numerical blow-up
        if not np.all(np.isfinite(x)):
            x = b.copy()

        return x

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Jacobi(lr={self.lr}, max_inner_iters={self.max_inner_iters}, "
            f"tolerance={self.tolerance})"
        )
