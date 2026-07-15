"""
Solver modules for the Hybrid Numerical Neural Network project.

Provides a unified interface to multiple optimization strategies:
  - GradientDescent:  Standard SGD with optional momentum
  - NewtonRaphson:    Full Newton step with Tikhonov regularization
  - DampedNewton:     Newton with conservative step size (damping)
  - GaussSeidel:      Sequential coordinate-wise linear solve
  - Jacobi:           Simultaneous coordinate-wise linear solve
  - BlockIterative:   Layer-wise (block) Gauss-Seidel updates

Usage:
    solver = get_solver('gd', lr=0.01, momentum=0.9)
    loss = solver.step(network, X, y)
"""

from .gradient_descent import GradientDescent
from .newton_raphson import NewtonRaphson, DampedNewton
from .gauss_seidel import GaussSeidel
from .jacobi import Jacobi
from .block_iterative import BlockIterative


def get_solver(name, **kwargs):
    """Factory function that returns an instantiated solver by short name.

    Parameters
    ----------
    name : str
        One of 'gd', 'newton', 'damped_newton', 'gauss_seidel', 'jacobi',
        'block'.
    **kwargs
        Keyword arguments forwarded to the solver constructor.

    Returns
    -------
    solver : object
        An instantiated solver with a `step(network, X, y)` method.

    Raises
    ------
    KeyError
        If *name* is not a recognised solver alias.
    """
    solvers = {
        'gd': GradientDescent,
        'newton': NewtonRaphson,
        'damped_newton': DampedNewton,
        'gauss_seidel': GaussSeidel,
        'jacobi': Jacobi,
        'block': BlockIterative,
    }
    if name not in solvers:
        raise KeyError(
            f"Unknown solver '{name}'. Choose from: {list(solvers.keys())}"
        )
    return solvers[name](**kwargs)
