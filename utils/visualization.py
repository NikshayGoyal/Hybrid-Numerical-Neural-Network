"""
Visualization Utilities
=======================

Publication-quality Matplotlib plots for comparing solver convergence,
accuracy, learning-rate schedules, and predictions.

All plotting functions accept pre-computed data structures and save
figures to disk.  A consistent color palette is used across every plot
so that each solver is instantly recognizable.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import os


# ---------------------------------------------------------------------------
# Solver color palette - consistent across ALL plots
# ---------------------------------------------------------------------------

SOLVER_COLORS = {
    # Canonical names
    'Gradient Descent':  '#2196F3',
    'Newton-Raphson':    '#F44336',
    'Damped Newton':     '#FF9800',
    'Gauss-Seidel':      '#4CAF50',
    'Jacobi':            '#9C27B0',
    'Block-wise':        '#00BCD4',
    # Short aliases (as returned by get_solver().name or config keys)
    'GD':                '#2196F3',
    'Newton':            '#F44336',
    'Block':             '#00BCD4',
}

# Fallback cycle when a solver name is not in the palette
DEFAULT_COLORS = [
    '#2196F3', '#F44336', '#FF9800', '#4CAF50',
    '#9C27B0', '#00BCD4', '#E91E63', '#795548',
    '#607D8B', '#CDDC39',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_color(name, idx=0):
    """Return a consistent hex color for *name*.

    Checks ``SOLVER_COLORS`` first; falls back to ``DEFAULT_COLORS``
    indexed by *idx* (mod length).
    """
    if name in SOLVER_COLORS:
        return SOLVER_COLORS[name]
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


def _setup_style():
    """Apply a clean, modern Matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')


def _ensure_dir(path):
    """Create parent directories for *path* if they don't exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_convergence(histories, save_path='results/convergence_curves.png'):
    """Plot training loss vs. epoch for multiple solvers.

    Parameters
    ----------
    histories : dict[str, list[float]]
        Mapping of solver name -> list of loss values (one per epoch).
    save_path : str
        File path where the figure is saved.
    """
    _setup_style()
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, losses) in enumerate(histories.items()):
        epochs = np.arange(1, len(losses) + 1)
        color = _get_color(name, idx)
        ax.plot(epochs, losses, label=name, color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training Loss Convergence', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Convergence plot saved -> {save_path}")


def plot_time_vs_accuracy(results, save_path='results/time_vs_accuracy.png'):
    """Plot cumulative wall-clock time vs. accuracy / R2 for each solver.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of solver name -> ``{'times': [...], 'accuracies': [...]}``.
        ``times`` is cumulative wall-clock seconds and ``accuracies`` is the
        corresponding metric at each recorded point.
    save_path : str
        File path where the figure is saved.
    """
    _setup_style()
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, data) in enumerate(results.items()):
        times = np.asarray(data['times'])
        accs = np.asarray(data['accuracies'])
        color = _get_color(name, idx)
        ax.plot(times, accs, label=name, color=color, linewidth=2, alpha=0.9)
        # Mark final point with a dot
        if len(times) > 0:
            ax.scatter(times[-1], accs[-1], color=color, s=60, zorder=5,
                       edgecolors='white', linewidths=1.2)

    ax.set_xlabel('Cumulative Wall-Clock Time (s)', fontsize=13)
    ax.set_ylabel('Accuracy / R2', fontsize=13)
    ax.set_title('Time vs. Accuracy', fontsize=15, fontweight='bold')
    ax.set_ylim(bottom=max(ax.get_ylim()[0], -0.5))  # Clip extreme negatives
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Time-vs-accuracy plot saved -> {save_path}")


def plot_solver_comparison(results, save_path='results/solver_comparison.png'):
    """Bar chart comparing final metrics across solvers (2x2 subplots).

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of solver name -> ``{'final_loss': float,
        'final_accuracy': float, 'total_time': float, 'epochs': int}``.
    save_path : str
        File path where the figure is saved.
    """
    _setup_style()
    _ensure_dir(save_path)

    names = list(results.keys())
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    final_loss = [results[n]['final_loss'] for n in names]
    final_acc = [results[n]['final_accuracy'] for n in names]
    total_time = [results[n]['total_time'] for n in names]
    epochs = [results[n]['epochs'] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Final Loss ---
    ax = axes[0, 0]
    bars = ax.bar(names, final_loss, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Training Loss', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=25, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    # Annotate bars
    for bar, val in zip(bars, final_loss):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # --- Final Accuracy / R2 ---
    ax = axes[0, 1]
    bars = ax.bar(names, final_acc, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Accuracy / R2', fontsize=12)
    ax.set_title('Final Accuracy / R2', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=25, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars, final_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # --- Total Time ---
    ax = axes[1, 0]
    bars = ax.bar(names, total_time, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Total Training Time', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=25, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars, total_time):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9)

    # --- Epochs ---
    ax = axes[1, 1]
    bars = ax.bar(names, epochs, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Epochs', fontsize=12)
    ax.set_title('Training Epochs', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=25, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for bar, val in zip(bars, epochs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Solver Comparison Dashboard', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Solver comparison plot saved -> {save_path}")


def plot_lr_schedules(schedules, save_path='results/lr_schedules.png'):
    """Plot learning-rate curves for different polynomial decay powers.

    Parameters
    ----------
    schedules : dict[str, list[float]]
        Mapping of power label (e.g. ``'power=1.0'``) -> list of LR values
        (one per epoch/step).
    save_path : str
        File path where the figure is saved.
    """
    _setup_style()
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a curated set of colors for schedule curves
    schedule_colors = ['#2196F3', '#F44336', '#FF9800', '#4CAF50',
                       '#9C27B0', '#00BCD4', '#E91E63', '#795548']

    for idx, (label, lr_values) in enumerate(schedules.items()):
        steps = np.arange(1, len(lr_values) + 1)
        color = schedule_colors[idx % len(schedule_colors)]
        ax.plot(steps, lr_values, label=label, color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel('Step / Epoch', fontsize=13)
    ax.set_ylabel('Learning Rate', fontsize=13)
    ax.set_title('Polynomial Learning-Rate Schedules', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] LR schedule plot saved -> {save_path}")


def plot_predictions(X, y_true, predictions, save_path='results/predictions.png'):
    """Scatter plot of true data overlaid with predicted curves per solver.

    Parameters
    ----------
    X : np.ndarray, shape (n,) or (n, 1)
        Input feature (1-D regression).
    y_true : np.ndarray, shape (n,) or (n, 1)
        Ground-truth targets.
    predictions : dict[str, np.ndarray]
        Mapping of solver name -> predicted y values (same shape as *y_true*).
    save_path : str
        File path where the figure is saved.
    """
    _setup_style()
    _ensure_dir(save_path)

    # Flatten for plotting
    X_flat = np.asarray(X).ravel()
    y_flat = np.asarray(y_true).ravel()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by X for clean line plots
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y_flat[sort_idx]

    # True data as a scatter
    ax.scatter(X_sorted, y_sorted, color='#B0BEC5', s=15, alpha=0.5,
               label='True data', zorder=1)

    for idx, (name, y_pred) in enumerate(predictions.items()):
        y_pred_flat = np.asarray(y_pred).ravel()
        y_pred_sorted = y_pred_flat[sort_idx]
        color = _get_color(name, idx)
        ax.plot(X_sorted, y_pred_sorted, label=name, color=color,
                linewidth=2, alpha=0.85, zorder=2)

    ax.set_xlabel('X (normalized)', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title('Predictions vs. Ground Truth', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Predictions plot saved -> {save_path}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def create_summary_table(results):
    """Print a nicely formatted comparison table to the console.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of solver name -> ``{'epochs': int, 'final_loss': float,
        'final_accuracy': float, 'total_time': float}``.
    """
    # Column headers
    header = (f"{'Solver':<20} {'Epochs':>7} {'Final Loss':>12} "
              f"{'Accuracy/R2':>12} {'Time (s)':>10}")
    separator = '-' * len(header)

    print()
    print(separator)
    print(header)
    print(separator)

    for name, metrics in results.items():
        epochs = metrics.get('epochs', '-')
        loss = metrics.get('final_loss', float('nan'))
        acc = metrics.get('final_accuracy', float('nan'))
        t = metrics.get('total_time', float('nan'))
        print(f"{name:<20} {epochs:>7} {loss:>12.6f} {acc:>12.6f} {t:>10.3f}")

    print(separator)
    print()
