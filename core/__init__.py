"""
core - Neural network building blocks for the Hybrid Numerical NN project.

This package exposes the primary classes and factory functions needed to
construct, train, and evaluate a fully-connected multilayer perceptron (MLP)
implemented from scratch in NumPy.

Public API
----------
NeuralNetwork     : The MLP class with forward/backward/Hessian support.
get_activation    : Factory that returns an activation function object by name.
get_loss          : Factory that returns a loss function object by name.
get_initializer   : Factory that returns a weight-init callable by name.
"""

from .network import NeuralNetwork
from .activations import get_activation
from .losses import get_loss
from .initializers import get_initializer
