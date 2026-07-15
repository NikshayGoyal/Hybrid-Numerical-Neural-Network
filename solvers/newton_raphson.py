"""
Newton-Raphson and Damped-Newton solvers.

These second-order methods use curvature (Hessian) information to compute
a Newton direction and update the network parameters accordingly.

Update rule
-----------
    delta = H_reg-^(-1) * g          (solved via np.linalg.solve for stability)
    theta <- theta - lr * delta

where
    H_reg = H + lambda*I          (Tikhonov / ridge regularisation)
    g     = flattened gradient vector
    lr    = 1.0  for pure Newton,  0.1-0.5  for damped Newton

If the regularised Hessian is (near-)singular the solver falls back to a
plain gradient step so that training never stalls.
"""

import numpy as np


# ======================================================================
# Base mixin with shared logic
# ======================================================================

class _NewtonBase:
    """Shared implementation for both Newton variants."""

    def _newton_step(self, network, X, y):
        """Core Newton update used by both NewtonRaphson and DampedNewton.

        Parameters
        ----------
        network : NeuralNetwork
        X : ndarray  - input batch
        y : ndarray  - targets

        Returns
        -------
        loss : float
        """
        # 1. Backward pass -> gradients + loss
        loss = network.backward(X, y)
        grad = network.get_flat_grads()            # shape (N,)

        # 2. Hessian
        H = network.compute_hessian(X, y)           # shape (N, N)
        n_params = H.shape[0]

        # 3. Tikhonov regularisation:  H_reg = H + lambda*I
        H_reg = H + self.regularization * np.eye(n_params)

        # 4. Solve for Newton direction: H_reg * delta = g
        try:
            delta = np.linalg.solve(H_reg, grad)
        except np.linalg.LinAlgError:
            # Singular matrix -> fall back to gradient step
            delta = grad

        # 5. Guard against NaN / Inf from ill-conditioned systems
        if not np.all(np.isfinite(delta)):
            delta = grad

        # 6. Parameter update:  theta <- theta - lr * delta
        params = network.get_flat_params()
        params = params - self.lr * delta
        network.set_flat_params(params)

        return loss


# ======================================================================
# Full (pure) Newton-Raphson
# ======================================================================

class NewtonRaphson(_NewtonBase):
    """Full Newton-Raphson optimiser.

    Uses `lr = 1.0` by default, giving the classic quadratic-convergence
    Newton step.  A small Tikhonov term is added to the Hessian for
    numerical safety.

    Parameters
    ----------
    lr : float, default 1.0
        Step-size multiplier (1.0 = pure Newton).
    regularization : float, default 1e-4
        Coefficient lambda for Tikhonov regularisation of the Hessian.
    """

    def __init__(self, lr=1.0, regularization=1e-4):
        self.lr = lr
        self.regularization = regularization
        self.name = 'Newton-Raphson'

    def step(self, network, X, y):
        """One Newton-Raphson parameter update.

        Parameters
        ----------
        network : NeuralNetwork
        X : ndarray  - input batch
        y : ndarray  - targets

        Returns
        -------
        loss : float
            Loss value **before** the update.
        """
        return self._newton_step(network, X, y)

    def __repr__(self):
        return (
            f"NewtonRaphson(lr={self.lr}, "
            f"regularization={self.regularization})"
        )


# ======================================================================
# Damped Newton
# ======================================================================

class DampedNewton(_NewtonBase):
    """Damped Newton optimiser.

    Identical to :class:`NewtonRaphson` except the default step size is
    smaller (``lr = 0.1``), which stabilises training on non-convex loss
    surfaces at the cost of slower convergence near the optimum.

    Parameters
    ----------
    lr : float, default 0.1
        Damping factor (typically 0.1-0.5).
    regularization : float, default 1e-4
        Coefficient lambda for Tikhonov regularisation of the Hessian.
    """

    def __init__(self, lr=0.1, regularization=1e-4):
        self.lr = lr
        self.regularization = regularization
        self.name = 'Damped Newton'

    def step(self, network, X, y):
        """One damped-Newton parameter update.

        Parameters
        ----------
        network : NeuralNetwork
        X : ndarray  - input batch
        y : ndarray  - targets

        Returns
        -------
        loss : float
            Loss value **before** the update.
        """
        return self._newton_step(network, X, y)

    def __repr__(self):
        return (
            f"DampedNewton(lr={self.lr}, "
            f"regularization={self.regularization})"
        )
