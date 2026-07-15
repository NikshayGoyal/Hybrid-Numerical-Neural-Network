"""
Hybrid Numerical Neural Network - Experiment Runner
====================================================

Runs all numerical solvers on the same synthetic regression dataset,
compares convergence behaviour, wall-clock cost, and predictive accuracy,
then generates publication-quality plots and a summary table.

Usage
-----
    python -m experiments.run_experiments
"""

import sys
import os

# Ensure the project root is on sys.path so that sibling packages
# (core, solvers, schedulers, utils, config) can be imported when
# running as ``python -m experiments.run_experiments``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import traceback

from core import NeuralNetwork
from solvers import get_solver
from schedulers import PolynomialLR
from utils.data_loader import load_synthetic_regression
from utils.metrics import compute_mse, compute_r2
from utils.visualization import (
    plot_convergence,
    plot_time_vs_accuracy,
    plot_solver_comparison,
    plot_predictions,
    plot_lr_schedules,
    create_summary_table,
)
from config import NETWORK_CONFIG, TRAINING_CONFIG, RESULTS_DIR

# Optional extended configs - may not exist in config.py yet.
try:
    from config import SOLVER_CONFIGS
except ImportError:
    SOLVER_CONFIGS = None

try:
    from config import DATA_CONFIG
except ImportError:
    DATA_CONFIG = None


# -- Default solver configurations ------------------------------------------

DEFAULT_SOLVER_CONFIGS = [
    {
        'name': 'gd',
        'display': 'Gradient Descent',
        'lr': 0.01,
        'momentum': 0.9,
        'epochs': TRAINING_CONFIG.get('epochs', 500),
    },
    {
        'name': 'newton',
        'display': 'Newton-Raphson',
        'lr': 1.0,
        'regularization': 1e-4,
        'epochs': min(50, TRAINING_CONFIG.get('epochs', 500)),
    },
    {
        'name': 'damped_newton',
        'display': 'Damped Newton',
        'lr': 0.1,
        'regularization': 1e-4,
        'epochs': min(50, TRAINING_CONFIG.get('epochs', 500)),
    },
    {
        'name': 'gauss_seidel',
        'display': 'Gauss-Seidel',
        'lr': 0.01,
        'max_inner_iters': 10,
        'epochs': TRAINING_CONFIG.get('epochs', 500),
    },
    {
        'name': 'jacobi',
        'display': 'Jacobi',
        'lr': 0.01,
        'max_inner_iters': 10,
        'epochs': TRAINING_CONFIG.get('epochs', 500),
    },
    {
        'name': 'block',
        'display': 'Block-wise',
        'lr': 0.01,
        'cycles': 1,
        'epochs': TRAINING_CONFIG.get('epochs', 500),
    },
]


# -- Training loop ----------------------------------------------------------

def train_with_solver(network, solver, X_train, y_train, epochs,
                      scheduler=None, verbose_every=50):
    """Train *network* with *solver* for a fixed number of epochs.

    Parameters
    ----------
    network : NeuralNetwork
        The network instance to train **in-place**.
    solver : object
        A solver exposing ``step(network, X, y) -> loss``.
    X_train : np.ndarray
        Training inputs.
    y_train : np.ndarray
        Training targets.
    epochs : int
        Number of training epochs.
    scheduler : PolynomialLR or None
        If provided, the solver's learning rate is updated each epoch via
        ``solver.lr = scheduler.step()``.
    verbose_every : int
        Print a progress line every *verbose_every* epochs.  Set to 0 to
        suppress progress output.

    Returns
    -------
    dict
        ``{'losses': [...], 'times': [...], 'epoch_times': [...]}``

        - **losses** - training loss at the end of each epoch.
        - **times** - cumulative wall-clock time after each epoch (seconds).
        - **epoch_times** - wall-clock duration of each individual epoch.
    """
    losses = []
    cumulative_times = []
    epoch_times = []
    cumulative = 0.0

    solver_name = getattr(solver, 'name', solver.__class__.__name__)

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        # -- Single optimisation step --
        loss = solver.step(network, X_train, y_train)

        t_elapsed = time.time() - t_start
        cumulative += t_elapsed

        losses.append(float(loss))
        cumulative_times.append(cumulative)
        epoch_times.append(t_elapsed)

        # Update learning rate if a scheduler is provided
        if scheduler is not None:
            solver.lr = scheduler.step()

        # Progress logging
        if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
            lr_str = f"  lr={solver.lr:.6f}" if scheduler else ""
            print(f"    Epoch {epoch:>5d}/{epochs}  loss={loss:.6f}"
                  f"  time={cumulative:.2f}s{lr_str}")

        # Early divergence guard - stop if loss explodes
        if np.isnan(loss) or np.isinf(loss):
            print(f"    [!] {solver_name} diverged at epoch {epoch}. Stopping early.")
            break

    return {
        'losses': losses,
        'times': cumulative_times,
        'epoch_times': epoch_times,
    }


# -- Experiment 1: Solver Comparison ---------------------------------------

def run_solver_comparison():
    """Train every solver on synthetic sine-wave regression and compare.

    Workflow
    --------
    1. Load synthetic regression data.
    2. Build a base ``NeuralNetwork`` whose input dimension matches the data.
    3. For each solver: deep-copy the base network, train, collect metrics.
    4. Compute R2 on the held-out test set.
    5. Generate convergence, time-vs-accuracy, bar-chart, and prediction
       plots.
    6. Print a summary table to the console.
    """
    print("\n  Loading synthetic regression data...")
    X_train, X_test, y_train, y_test = load_synthetic_regression(
        n_samples=500, noise_std=0.1, test_size=0.2,
        seed=TRAINING_CONFIG.get('seed', 42),
    )
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # Build base network - override input size to match data
    net_cfg = dict(NETWORK_CONFIG)  # shallow copy
    layer_sizes = list(net_cfg.get('layer_sizes', [2, 64, 32, 1]))
    layer_sizes[0] = X_train.shape[1]
    net_cfg['layer_sizes'] = layer_sizes

    base_network = NeuralNetwork(**net_cfg)
    print(f"  Base network: {layer_sizes}")

    # Resolve solver configs
    solver_configs = SOLVER_CONFIGS if SOLVER_CONFIGS is not None else DEFAULT_SOLVER_CONFIGS

    # Accumulators
    convergence_histories = {}      # name -> loss list
    time_accuracy_data = {}         # name -> {'times': [...], 'accuracies': [...]}
    comparison_results = {}         # name -> summary dict
    prediction_outputs = {}         # name -> y_pred on test set

    for cfg in solver_configs:
        solver_key = cfg['name']
        display_name = cfg.get('display', solver_key)
        num_epochs = cfg.get('epochs', TRAINING_CONFIG.get('epochs', 500))

        # Build kwargs for get_solver (exclude meta keys)
        solver_kwargs = {k: v for k, v in cfg.items()
                         if k not in ('name', 'display', 'epochs')}

        print(f"\n  -- {display_name} {'-' * (40 - len(display_name))}")
        try:
            solver = get_solver(solver_key, **solver_kwargs)
            net = base_network.copy()

            result = train_with_solver(
                net, solver, X_train, y_train,
                epochs=num_epochs, verbose_every=50,
            )

            # Evaluate on test set
            y_pred_test = net.predict(X_test)
            test_mse = compute_mse(y_pred_test, y_test)
            test_r2 = compute_r2(y_pred_test, y_test)

            print(f"    > Test MSE = {test_mse:.6f}  |  R2 = {test_r2:.6f}")

            # Store results
            convergence_histories[display_name] = result['losses']

            # Build per-epoch R2 for time-vs-accuracy plot (approximate
            # using training loss -> R2 on train set is expensive, so we
            # just record final R2 as a flat line endpoint).
            time_accuracy_data[display_name] = {
                'times': result['times'],
                'accuracies': [compute_r2(
                    net.predict(X_test) if i == len(result['losses']) - 1
                    else np.zeros_like(y_test),
                    y_test
                ) if i == len(result['losses']) - 1
                else None
                for i in range(len(result['losses']))],
            }
            # Simplified: compute R2 at every checkpoint (every 10 epochs)
            # to keep cost reasonable but still show a curve.
            accuracies_curve = []
            _net_copy = base_network.copy()
            _solver_copy = get_solver(solver_key, **solver_kwargs)
            for ep_idx in range(len(result['losses'])):
                # We already trained the real network; just use the loss
                # as a proxy for the curve, and place the final R2 at end.
                # A simple heuristic: R2 ~= 1 - loss / var(y_test)
                var_y = float(np.var(y_test))
                proxy_r2 = 1.0 - result['losses'][ep_idx] / var_y if var_y > 0 else 0.0
                accuracies_curve.append(proxy_r2)
            time_accuracy_data[display_name] = {
                'times': result['times'],
                'accuracies': accuracies_curve,
            }

            comparison_results[display_name] = {
                'final_loss': result['losses'][-1] if result['losses'] else float('nan'),
                'final_accuracy': test_r2,
                'total_time': result['times'][-1] if result['times'] else 0.0,
                'epochs': len(result['losses']),
            }

            prediction_outputs[display_name] = y_pred_test

        except Exception as exc:
            print(f"    [!] {display_name} failed: {exc}")
            traceback.print_exc()
            continue

    # -- Generate plots ------------------------------------------------
    if not convergence_histories:
        print("\n  [!] No solvers completed successfully.  Skipping plots.")
        return

    print("\n  Generating plots...")
    plot_convergence(
        convergence_histories,
        save_path=os.path.join(RESULTS_DIR, 'convergence_curves.png'),
    )
    plot_time_vs_accuracy(
        time_accuracy_data,
        save_path=os.path.join(RESULTS_DIR, 'time_vs_accuracy.png'),
    )
    plot_solver_comparison(
        comparison_results,
        save_path=os.path.join(RESULTS_DIR, 'solver_comparison.png'),
    )
    plot_predictions(
        X_test, y_test, prediction_outputs,
        save_path=os.path.join(RESULTS_DIR, 'predictions.png'),
    )

    # -- Summary table -------------------------------------------------
    create_summary_table(comparison_results)


# -- Experiment 2: LR Schedule Comparison ----------------------------------

def run_lr_schedule_comparison():
    """Train with Gradient Descent under different polynomial LR powers.

    Generates a plot of the LR schedules and a convergence comparison.
    """
    print("\n  Loading synthetic regression data...")
    X_train, X_test, y_train, y_test = load_synthetic_regression(
        n_samples=500, noise_std=0.1, test_size=0.2,
        seed=TRAINING_CONFIG.get('seed', 42),
    )

    # Build base network
    net_cfg = dict(NETWORK_CONFIG)
    layer_sizes = list(net_cfg.get('layer_sizes', [2, 64, 32, 1]))
    layer_sizes[0] = X_train.shape[1]
    net_cfg['layer_sizes'] = layer_sizes

    base_network = NeuralNetwork(**net_cfg)
    num_epochs = TRAINING_CONFIG.get('epochs', 500)
    initial_lr = 0.01

    powers = [0.5, 1.0, 2.0, 3.0]
    lr_schedule_curves = {}   # label -> list of LR values
    convergence_curves = {}   # label -> list of loss values

    for power in powers:
        label = f"power={power}"
        print(f"\n  -- GD with polynomial LR ({label}) {'-' * 20}")

        try:
            net = base_network.copy()
            solver = get_solver('gd', lr=initial_lr, momentum=0.9)
            scheduler = PolynomialLR(
                initial_lr=initial_lr,
                total_steps=num_epochs,
                power=power,
                min_lr=1e-6,
            )

            # Record LR schedule values
            scheduler_copy = PolynomialLR(
                initial_lr=initial_lr,
                total_steps=num_epochs,
                power=power,
                min_lr=1e-6,
            )
            lr_values = []
            for _ in range(num_epochs):
                lr_values.append(scheduler_copy.get_lr())
                scheduler_copy.step()
            lr_schedule_curves[label] = lr_values

            # Train
            result = train_with_solver(
                net, solver, X_train, y_train,
                epochs=num_epochs, scheduler=scheduler,
                verbose_every=100,
            )
            convergence_curves[label] = result['losses']

            # Final evaluation
            y_pred_test = net.predict(X_test)
            test_r2 = compute_r2(y_pred_test, y_test)
            print(f"    > Final R2 = {test_r2:.6f}")

        except Exception as exc:
            print(f"    [!] Failed with {label}: {exc}")
            traceback.print_exc()
            continue

    # -- Generate plots ------------------------------------------------
    if lr_schedule_curves:
        print("\n  Generating LR schedule plots...")
        plot_lr_schedules(
            lr_schedule_curves,
            save_path=os.path.join(RESULTS_DIR, 'lr_schedules.png'),
        )
    if convergence_curves:
        plot_convergence(
            convergence_curves,
            save_path=os.path.join(RESULTS_DIR, 'lr_convergence_comparison.png'),
        )


# -- Entry point -----------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('Hybrid Numerical Neural Network - Experiments')
    print('=' * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(TRAINING_CONFIG.get('seed', 42))

    print('\n[1/2] Running Solver Comparison...')
    run_solver_comparison()

    print('\n[2/2] Running LR Schedule Comparison...')
    run_lr_schedule_comparison()

    print('\n' + '=' * 60)
    print('All experiments complete!  Results saved to:', RESULTS_DIR)
    print('=' * 60)
