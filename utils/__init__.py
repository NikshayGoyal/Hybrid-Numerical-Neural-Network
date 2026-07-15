"""
Utility modules for the Hybrid Numerical Neural Network project.
================================================================

Provides data loading, evaluation metrics, and visualization tools.
"""

from .data_loader import load_synthetic_regression, load_classification
from .metrics import compute_accuracy, compute_mse
from .visualization import plot_convergence, plot_solver_comparison, plot_lr_schedules
