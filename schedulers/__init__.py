"""
Learning-rate schedulers for the Hybrid Numerical Neural Network project.

Currently provides:
  - PolynomialLR: polynomial-decay schedule with configurable power and
    minimum learning rate.
"""

from .polynomial_lr import PolynomialLR

__all__ = ['PolynomialLR']
