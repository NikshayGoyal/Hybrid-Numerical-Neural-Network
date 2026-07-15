"""
config.py - Central configuration for the Hybrid Numerical Neural Network project.

All hyperparameters, architecture settings, solver configurations, learning rate
schedule parameters, data generation settings, and output paths are defined here
as module-level dictionaries. Import this module wherever configuration is needed.
"""

import os

# =============================================================================
# Network Architecture Configuration
# =============================================================================
# Defines the MLP topology and the default activation, loss, and weight
# initializer used when constructing a NeuralNetwork instance.
NETWORK_CONFIG = {
    "layer_sizes": [2, 16, 8, 1],   # input -> hidden1 -> hidden2 -> output
    "activation": "tanh",             # tanh is smoother than relu -> better Hessians for 2nd-order solvers
    "loss": "mse",                    # loss function (mean squared error)
    "initializer": "he",             # weight initialization strategy
}

# =============================================================================
# Training Configuration
# =============================================================================
# Global training settings shared across all solvers unless overridden.
TRAINING_CONFIG = {
    "epochs": 500,          # maximum number of training iterations
    "batch_size": None,     # None -> full-batch gradient descent
    "seed": 42,             # random seed for reproducibility
}

# =============================================================================
# Solver-Specific Configurations
# =============================================================================
# Each solver (optimizer) has its own dictionary of hyperparameters.
# The keys match the solver class names used in the solvers/ module.
SOLVER_CONFIGS = [
    {
        "name": "gd",
        "display": "Gradient Descent",
        "lr": 0.01,
        "momentum": 0.9,
        "epochs": 500,
    },
    {
        "name": "newton",
        "display": "Newton-Raphson",
        "lr": 1.0,
        "regularization": 0.5,
        "epochs": 50,
    },
    {
        "name": "damped_newton",
        "display": "Damped Newton",
        "lr": 0.3,
        "regularization": 0.1,
        "epochs": 50,
    },
    {
        "name": "gauss_seidel",
        "display": "Gauss-Seidel",
        "lr": 0.001,
        "max_inner_iters": 20,
        "tolerance": 1e-6,
        "regularization": 10.0,
        "epochs": 50,
    },
    {
        "name": "jacobi",
        "display": "Jacobi",
        "lr": 0.001,
        "max_inner_iters": 20,
        "tolerance": 1e-6,
        "regularization": 10.0,
        "epochs": 50,
    },
    {
        "name": "block",
        "display": "Block-wise",
        "lr": 0.01,
        "cycles": 1,
        "epochs": 500,
    },
]

# =============================================================================
# Learning Rate Schedule Configuration
# =============================================================================
# Polynomial decay schedules: lr(t) = initial_lr * (1 - t/T)^power
# Multiple powers are evaluated for comparative analysis.
LR_SCHEDULE_CONFIG = {
    "initial_lr": 0.01,                  # starting learning rate
    "polynomial_powers": [0.5, 1.0, 2.0],  # sqrt-decay, linear-decay, quadratic-decay
}

# =============================================================================
# Data Generation Configuration
# =============================================================================
# Settings for synthetic dataset creation (regression / classification tasks).
DATA_CONFIG = {
    "n_samples": 500,       # total number of data points
    "noise_std": 0.1,       # standard deviation of Gaussian noise added to targets
    "test_split": 0.2,      # fraction of data reserved for testing
}

# =============================================================================
# Output Paths
# =============================================================================
# Directory where training logs, plots, and saved models are written.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
