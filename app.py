"""
Streamlit Interactive Demo
==========================

Launch with:  streamlit run app.py

Provides an interactive dashboard to:
  - Select solvers and configure hyperparameters
  - Train the network and watch convergence live
  - Compare solver performance side-by-side
"""

import streamlit as st
import numpy as np
import time
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.network import NeuralNetwork
from solvers import get_solver
from schedulers.polynomial_lr import PolynomialLR
from utils.data_loader import load_synthetic_regression
from utils.metrics import compute_mse, compute_r2

# =====================================================================
# Page Config
# =====================================================================

st.set_page_config(
    page_title="Hybrid Numerical Neural Network",
    page_icon="🧠",
    layout="wide",
)

# =====================================================================
# Custom CSS
# =====================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Header
# =====================================================================

st.markdown("# 🧠 Hybrid Numerical Neural Network")
st.markdown("*Interactive demo comparing gradient-based and classical numerical solvers for neural network training.*")
st.markdown("---")

# =====================================================================
# Sidebar - Configuration
# =====================================================================

st.sidebar.header("Network Configuration")

layer_input = st.sidebar.text_input("Hidden layers (comma-separated)", "16, 8")
try:
    hidden = [int(x.strip()) for x in layer_input.split(",")]
except ValueError:
    hidden = [16, 8]

activation = st.sidebar.selectbox("Activation", ["tanh", "relu", "sigmoid"], index=0)
epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)
n_samples = st.sidebar.slider("Data samples", 100, 1000, 500, step=100)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Solver Selection")

SOLVER_OPTIONS = {
    "Gradient Descent": {"name": "gd", "params": {"lr": 0.01, "momentum": 0.9}},
    "Newton-Raphson": {"name": "newton", "params": {"lr": 1.0, "regularization": 0.5}},
    "Damped Newton": {"name": "damped_newton", "params": {"lr": 0.3, "regularization": 0.1}},
    "Gauss-Seidel": {"name": "gauss_seidel", "params": {"lr": 0.001, "max_inner_iters": 20, "regularization": 10.0}},
    "Jacobi": {"name": "jacobi", "params": {"lr": 0.001, "max_inner_iters": 20, "regularization": 10.0}},
    "Block-wise": {"name": "block", "params": {"lr": 0.01, "cycles": 1}},
}

selected_solvers = st.sidebar.multiselect(
    "Choose solvers to compare",
    list(SOLVER_OPTIONS.keys()),
    default=["Gradient Descent", "Damped Newton", "Block-wise"],
)

# Cap epochs for Hessian-based solvers
HESSIAN_SOLVERS = {"Newton-Raphson", "Damped Newton", "Gauss-Seidel", "Jacobi"}

st.sidebar.markdown("---")
st.sidebar.header("Learning Rate")
lr_override = st.sidebar.slider("Learning rate (for GD/Block)", 0.001, 0.1, 0.01, step=0.001, format="%.3f")

# =====================================================================
# Main - Run Training
# =====================================================================

if st.button("🚀 Train & Compare", type="primary", use_container_width=True):
    if not selected_solvers:
        st.warning("Please select at least one solver.")
    else:
        np.random.seed(seed)

        # Generate data
        layer_sizes = [1] + hidden + [1]
        X_train, X_test, y_train, y_test = load_synthetic_regression(
            n_samples=n_samples, noise_std=0.1, test_split=0.2, seed=seed
        )

        st.markdown("### Training Progress")

        # Create a base network and copy for each solver
        base_net = NeuralNetwork(layer_sizes, activation=activation, loss='mse', initializer='he')

        all_results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_solvers = len(selected_solvers)

        for solver_idx, solver_name in enumerate(selected_solvers):
            status_text.text(f"Training with {solver_name}...")

            config = SOLVER_OPTIONS[solver_name]
            params = config["params"].copy()

            # Apply LR override for GD and Block
            if solver_name in ("Gradient Descent", "Block-wise"):
                params["lr"] = lr_override

            solver = get_solver(config["name"], **params)
            net = base_net.copy()

            # Cap epochs for expensive solvers
            solver_epochs = min(epochs, 50) if solver_name in HESSIAN_SOLVERS else epochs

            losses = []
            start_time = time.time()

            for epoch in range(solver_epochs):
                loss = solver.step(net, X_train, y_train)
                losses.append(float(loss))

                # Divergence check
                if not np.isfinite(loss) or loss > 1e10:
                    break

            elapsed = time.time() - start_time

            # Evaluate
            y_pred = net.predict(X_test)
            test_mse = compute_mse(y_test, y_pred)
            test_r2 = compute_r2(y_test, y_pred)

            all_results[solver_name] = {
                "losses": losses,
                "test_mse": float(test_mse),
                "test_r2": float(test_r2),
                "time": elapsed,
                "epochs_run": len(losses),
                "network": net,
            }

            progress_bar.progress((solver_idx + 1) / total_solvers)

        status_text.text("Training complete!")
        progress_bar.empty()

        # =============================================================
        # Results Display
        # =============================================================

        st.markdown("---")
        st.markdown("### Results")

        # Metric cards
        cols = st.columns(len(all_results))
        for col, (name, res) in zip(cols, all_results.items()):
            with col:
                r2_display = f"{res['test_r2']:.4f}" if np.isfinite(res['test_r2']) else "N/A"
                st.metric(label=name, value=f"R2 = {r2_display}", delta=f"{res['time']:.2f}s")

        # Convergence plot
        st.markdown("### Convergence Curves")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4']
        for idx, (name, res) in enumerate(all_results.items()):
            ax.plot(range(1, len(res["losses"]) + 1), res["losses"],
                    label=name, color=colors[idx % len(colors)], linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss Convergence", fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Comparison table
        st.markdown("### Detailed Comparison")
        table_data = []
        for name, res in all_results.items():
            table_data.append({
                "Solver": name,
                "Epochs": res["epochs_run"],
                "Final Loss": f"{res['losses'][-1]:.6f}" if res["losses"] else "N/A",
                "Test MSE": f"{res['test_mse']:.6f}" if np.isfinite(res['test_mse']) else "Diverged",
                "Test R2": f"{res['test_r2']:.4f}" if np.isfinite(res['test_r2']) else "Diverged",
                "Time (s)": f"{res['time']:.3f}",
            })
        st.table(table_data)

        # Predictions plot
        st.markdown("### Prediction Quality")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sort_idx = np.argsort(X_test.flatten())
        X_sorted = X_test[sort_idx]
        ax2.scatter(X_test.flatten(), y_test.flatten(), alpha=0.3, s=15,
                    color='gray', label='True data')
        for idx, (name, res) in enumerate(all_results.items()):
            y_pred = res["network"].predict(X_sorted)
            ax2.plot(X_sorted.flatten(), y_pred.flatten(),
                     label=name, color=colors[idx % len(colors)], linewidth=2)
        ax2.set_xlabel("X (normalized)", fontsize=12)
        ax2.set_ylabel("y", fontsize=12)
        ax2.set_title("Predictions vs. Ground Truth", fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

else:
    st.info("Configure settings in the sidebar and click **Train & Compare** to start.")

    # Show project info when idle
    st.markdown("### About This Project")
    st.markdown("""
    This project implements a neural network **from scratch using only NumPy** and trains it
    using six different numerical optimization methods:

    | Solver | Type | Key Property |
    |:---|:---|:---|
    | **Gradient Descent** | First-order | Fast per epoch, uses only gradients |
    | **Newton-Raphson** | Second-order | Uses full Hessian for curvature-aware updates |
    | **Damped Newton** | Second-order | Newton with step-size damping for stability |
    | **Gauss-Seidel** | Iterative | Sequential parameter updates using Hessian system |
    | **Jacobi** | Iterative | Parallel parameter updates (all at once) |
    | **Block-wise** | Coordinate descent | Optimizes one layer at a time |

    Use the sidebar to select solvers, configure the network, and hit **Train & Compare**!
    """)
