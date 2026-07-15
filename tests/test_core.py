"""
Unit Tests for Hybrid Numerical Neural Network
================================================

Tests core components: network forward/backward, activations, losses,
initializers, solvers, and schedulers.

Run with:  python -m pytest tests/ -v
    or:    python -m tests.test_core
"""

import numpy as np
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuralNetwork
from core.activations import get_activation
from core.losses import get_loss
from core.initializers import get_initializer
from solvers import get_solver
from schedulers.polynomial_lr import PolynomialLR


# =====================================================================
# Activation Tests
# =====================================================================

def test_relu():
    relu = get_activation('relu')
    x = np.array([-2, -1, 0, 1, 2], dtype=float)
    out = relu.forward(x)
    assert np.allclose(out, [0, 0, 0, 1, 2]), f"ReLU forward failed: {out}"
    grad = relu.backward(x)
    assert np.allclose(grad, [0, 0, 0, 1, 1]), f"ReLU backward failed: {grad}"
    print("  [PASS] ReLU activation")


def test_sigmoid():
    sig = get_activation('sigmoid')
    x = np.array([0.0])
    out = sig.forward(x)
    assert np.allclose(out, [0.5]), f"Sigmoid(0) should be 0.5, got {out}"
    grad = sig.backward(x)
    assert np.allclose(grad, [0.25]), f"Sigmoid'(0) should be 0.25, got {grad}"
    print("  [PASS] Sigmoid activation")


def test_tanh():
    tanh = get_activation('tanh')
    x = np.array([0.0])
    out = tanh.forward(x)
    assert np.allclose(out, [0.0]), f"Tanh(0) should be 0.0, got {out}"
    grad = tanh.backward(x)
    assert np.allclose(grad, [1.0]), f"Tanh'(0) should be 1.0, got {grad}"
    print("  [PASS] Tanh activation")


# =====================================================================
# Loss Tests
# =====================================================================

def test_mse_loss():
    mse = get_loss('mse')
    y_pred = np.array([[1.0], [2.0], [3.0]])
    y_true = np.array([[1.0], [2.0], [3.0]])
    loss = mse.forward(y_pred, y_true)
    assert np.isclose(loss, 0.0), f"MSE of identical arrays should be 0, got {loss}"

    y_pred2 = np.array([[2.0], [3.0], [4.0]])
    loss2 = mse.forward(y_pred2, y_true)
    assert loss2 > 0, f"MSE of different arrays should be > 0, got {loss2}"

    grad = mse.backward(y_pred2, y_true)
    assert grad.shape == y_pred2.shape, f"Gradient shape mismatch: {grad.shape}"
    print("  [PASS] MSE loss")


# =====================================================================
# Initializer Tests
# =====================================================================

def test_initializers():
    for name in ['he', 'xavier', 'random']:
        init_fn = get_initializer(name)
        W = init_fn(64, 32)
        assert W.shape == (64, 32), f"{name} init shape wrong: {W.shape}"
        assert np.all(np.isfinite(W)), f"{name} init has non-finite values"
    print("  [PASS] All initializers (he, xavier, random)")


# =====================================================================
# Network Tests
# =====================================================================

def test_network_forward():
    net = NeuralNetwork([2, 8, 4, 1], activation='tanh', loss='mse')
    X = np.random.randn(10, 2)
    y_pred = net.forward(X)
    assert y_pred.shape == (10, 1), f"Forward output shape wrong: {y_pred.shape}"
    assert np.all(np.isfinite(y_pred)), "Forward produced non-finite values"
    print("  [PASS] Network forward pass")


def test_network_backward():
    net = NeuralNetwork([2, 8, 4, 1], activation='tanh', loss='mse')
    X = np.random.randn(10, 2)
    y = np.random.randn(10, 1)
    loss = net.backward(X, y)
    assert np.isfinite(loss), f"Backward returned non-finite loss: {loss}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    grads = net.get_flat_grads()
    assert len(grads) == net.num_params(), "Gradient size != param count"
    assert np.all(np.isfinite(grads)), "Gradients contain non-finite values"
    print("  [PASS] Network backward pass")


def test_network_flat_params():
    net = NeuralNetwork([2, 8, 1], activation='tanh', loss='mse')
    params = net.get_flat_params()
    n = net.num_params()
    assert len(params) == n, f"Flat params length {len(params)} != num_params {n}"

    # Set and get should be identity
    new_params = np.random.randn(n)
    net.set_flat_params(new_params)
    retrieved = net.get_flat_params()
    assert np.allclose(new_params, retrieved), "set/get_flat_params not consistent"
    print("  [PASS] Network flat params round-trip")


def test_network_copy():
    net = NeuralNetwork([2, 4, 1], activation='tanh', loss='mse')
    net_copy = net.copy()
    assert np.allclose(net.get_flat_params(), net_copy.get_flat_params())

    # Modifying copy should not affect original
    net_copy.set_flat_params(np.zeros(net.num_params()))
    assert not np.allclose(net.get_flat_params(), net_copy.get_flat_params())
    print("  [PASS] Network deep copy")


# =====================================================================
# Solver Tests
# =====================================================================

def test_gradient_descent_step():
    np.random.seed(42)
    net = NeuralNetwork([2, 4, 1], activation='tanh', loss='mse')
    X = np.random.randn(20, 2)
    y = np.random.randn(20, 1)
    solver = get_solver('gd', lr=0.01, momentum=0.0)

    loss_before = solver.step(net, X, y)
    loss_after = net.backward(X, y)
    assert loss_after <= loss_before + 1e-6, "GD step did not reduce loss"
    print("  [PASS] Gradient Descent solver")


def test_block_solver_step():
    np.random.seed(42)
    net = NeuralNetwork([2, 4, 1], activation='tanh', loss='mse')
    X = np.random.randn(20, 2)
    y = np.random.randn(20, 1)
    solver = get_solver('block', lr=0.01, cycles=1)

    loss = solver.step(net, X, y)
    assert np.isfinite(loss), f"Block solver returned non-finite loss: {loss}"
    print("  [PASS] Block-wise solver")


# =====================================================================
# Scheduler Tests
# =====================================================================

def test_polynomial_lr():
    scheduler = PolynomialLR(initial_lr=0.1, total_steps=100, power=1.0)
    lr_start = scheduler.get_lr()
    assert np.isclose(lr_start, 0.1), f"Initial LR should be 0.1, got {lr_start}"

    for _ in range(50):
        scheduler.step()
    lr_mid = scheduler.get_lr()
    assert lr_mid < lr_start, f"LR should decay: start={lr_start}, mid={lr_mid}"

    for _ in range(50):
        scheduler.step()
    lr_end = scheduler.get_lr()
    assert lr_end <= lr_mid, f"LR should keep decaying: mid={lr_mid}, end={lr_end}"

    scheduler.reset()
    assert np.isclose(scheduler.get_lr(), 0.1), "Reset did not restore initial LR"
    print("  [PASS] Polynomial LR scheduler")


# =====================================================================
# Run all tests
# =====================================================================

def run_all_tests():
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)

    tests = [
        test_relu, test_sigmoid, test_tanh,
        test_mse_loss,
        test_initializers,
        test_network_forward, test_network_backward,
        test_network_flat_params, test_network_copy,
        test_gradient_descent_step, test_block_solver_step,
        test_polynomial_lr,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
