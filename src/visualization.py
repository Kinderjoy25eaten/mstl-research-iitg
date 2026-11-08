"""
Visualization Module
====================
Publication-quality plots for the MSTL research project.

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

STYLE = {
    'figure.figsize': (14, 4),
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.0,
}
plt.rcParams.update(STYLE)
sns.set_theme(style='whitegrid', palette='deep')


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_original_series(series, title="Victoria Electricity Demand (Hourly)",
                         ylabel="Demand (MWh)", filename="fig_01_original_series.png"):
    """Plot full original time series."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(series.index, series.values, color='steelblue', linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, filename)


def plot_two_weeks(series, title="Victoria Electricity Demand — First Two Weeks",
                   ylabel="Demand (MWh)", filename="fig_02_two_weeks.png"):
    """Plot first two weeks of hourly data."""
    two_weeks = series.iloc[:336]   # 2 × 168 h
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(two_weeks.index, two_weeks.values, color='darkorange', linewidth=1.0)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, filename)


def plot_mstl_decomposition(series, result, periods=(24, 168),
                             title_prefix="Custom MSTL",
                             filename="fig_03_mstl_decomposition.png"):
    """
    Plot full MSTL decomposition: original, trend, each seasonal, remainder.
    """
    n_panels = 3 + len(periods)   # original + trend + N seasonals + remainder
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels),
                             sharex=True)

    idx = series.index if hasattr(series, 'index') else np.arange(len(result['original']))

    axes[0].plot(idx, result['original'], color='steelblue', linewidth=0.8)
    axes[0].set_ylabel("Original")
    axes[0].set_title(f"{title_prefix} Decomposition", fontsize=14, fontweight='bold')

    axes[1].plot(idx, result['trend'], color='crimson', linewidth=1.0)
    axes[1].set_ylabel("Trend")

    for i, m in enumerate(periods):
        if m in result['seasonal']:
            axes[2 + i].plot(idx, result['seasonal'][m], color='seagreen', linewidth=0.8)
            axes[2 + i].set_ylabel(f"S({m}h)")

    axes[-1].plot(idx, result['remainder'], color='purple', linewidth=0.6, alpha=0.8)
    axes[-1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[-1].set_ylabel("Remainder")
    axes[-1].set_xlabel("Time")

    if hasattr(idx, 'dtype') and np.issubdtype(idx.dtype, np.datetime64):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        fig.autofmt_xdate()

    fig.tight_layout()
    return _save(fig, filename)


def plot_component_comparison(idx, comp_custom, comp_library,
                              comp_name="Trend",
                              filename="fig_04_component_comparison.png"):
    """Compare a single component (e.g. trend) between custom and library MSTL."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(idx, comp_custom, label='Custom MSTL', color='crimson', linewidth=1.0)
    ax.plot(idx, comp_library, label='Library MSTL', color='navy',
            linewidth=1.0, linestyle='--', alpha=0.8)
    ax.set_title(f"{comp_name}: Custom vs Library MSTL", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel(comp_name)
    ax.legend()
    if hasattr(idx, 'dtype') and np.issubdtype(idx.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, filename)


def plot_residual_diagnostics(residuals, method_name="Custom MSTL",
                               filename="fig_residual_diag.png"):
    """Plot residual time series, ACF, and histogram."""
    from statsmodels.graphics.tsaplots import plot_acf

    r = np.asarray(residuals)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(r, color='purple', linewidth=0.6, alpha=0.8)
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_title(f"{method_name} — Residuals")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Residual")

    plot_acf(r, lags=min(100, len(r) // 4), ax=axes[1], alpha=0.05)
    axes[1].set_title("Residual ACF")

    axes[2].hist(r, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Residual Value")
    axes[2].set_ylabel("Frequency")

    fig.suptitle(f"Residual Diagnostics — {method_name}", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, filename)


def plot_sensitivity_heatmap(rmse_table, title="Seasonal Window Sensitivity (RMSE)",
                              filename="fig_ext1_sensitivity_heatmap.png"):
    """Plot RMSE heatmap for seasonal window × iteration grid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(rmse_table, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Outer Iterations")
    ax.set_ylabel("Seasonal Window (s)")
    fig.tight_layout()
    return _save(fig, filename)


def plot_ablation(results_dict, metric='reconstruction_rmse',
                  filename="fig_ext2_ablation.png"):
    """Bar chart comparing MSTL configurations in the ablation study."""
    labels = list(results_dict.keys())
    values = [results_dict[k][metric] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='white', width=0.5)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_title("Ablation Study: Seasonality Configurations", fontsize=13, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xlabel("Configuration")
    fig.tight_layout()
    return _save(fig, filename)


def plot_scalability(sizes, times, labels=None, filename="fig_ext4_scalability.png"):
    """Plot execution time vs dataset size."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if labels is None:
        labels = ['Method'] * len(times[0]) if isinstance(times[0], list) else ['MSTL']
    if isinstance(times[0], (list, np.ndarray)):
        for j, lbl in enumerate(labels):
            ax.plot(sizes, [t[j] for t in times], marker='o', label=lbl)
        ax.legend()
    else:
        ax.plot(sizes, times, marker='o', color='steelblue', label='Custom MSTL')
    ax.set_title("Computational Scalability", fontsize=13, fontweight='bold')
    ax.set_xlabel("Dataset Size (observations)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return _save(fig, filename)


def plot_synthetic_rmse(rmse_dict, filename="fig_synthetic_rmse.png"):
    """Bar chart of component-wise RMSE on synthetic data."""
    components = list(next(iter(rmse_dict.values())).keys())
    methods = list(rmse_dict.keys())
    x = np.arange(len(components))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = [rmse_dict[method].get(c, 0) or 0 for c in components]
        ax.bar(x + i * width, vals, width, label=method)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(components)
    ax.set_title("Component-wise RMSE on Synthetic Data", fontsize=13, fontweight='bold')
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    return _save(fig, filename)


def plot_robustness(original, corrupted, clean_result, corrupted_result,
                    filename="fig_ext3_robustness.png"):
    """Compare MSTL trend on original vs corrupted series."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    n = len(original)
    idx = np.arange(n)

    axes[0, 0].plot(idx, original, color='steelblue', linewidth=0.7)
    axes[0, 0].set_title("Original Series")
    axes[0, 0].set_ylabel("Demand")

    axes[0, 1].plot(idx, corrupted, color='darkorange', linewidth=0.7)
    axes[0, 1].set_title("Corrupted Series (Outliers/Missing)")
    axes[0, 1].set_ylabel("Demand")

    axes[1, 0].plot(idx, clean_result['trend'], color='crimson', linewidth=1.0)
    axes[1, 0].set_title("MSTL Trend — Original")
    axes[1, 0].set_ylabel("Trend")
    axes[1, 0].set_xlabel("Index")

    axes[1, 1].plot(idx, corrupted_result['trend'], color='purple', linewidth=1.0)
    axes[1, 1].set_title("MSTL Trend — Corrupted")
    axes[1, 1].set_ylabel("Trend")
    axes[1, 1].set_xlabel("Index")

    fig.suptitle("Robustness to Outliers and Missing Data",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save(fig, filename)

