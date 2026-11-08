"""
Evaluation Metrics Module
=========================
Provides RMSE and other evaluation utilities.

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error (avoid division by zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


def reconstruction_error(original, trend, seasonals, remainder):
    """
    Verify that trend + sum(seasonals) + remainder ≈ original.
    Returns max absolute error and mean absolute error.
    """
    if isinstance(seasonals, dict):
        sum_s = np.sum(list(seasonals.values()), axis=0)
    else:
        sum_s = np.asarray(seasonals)
    reconstructed = trend + sum_s + remainder
    diff = np.abs(np.asarray(original) - reconstructed)
    return {'max_abs_error': np.max(diff), 'mean_abs_error': np.mean(diff)}


def component_wise_rmse(true_components, estimated_components):
    """
    Calculate RMSE for each named component.

    Parameters
    ----------
    true_components : dict  {name: array}
    estimated_components : dict  {name: array}

    Returns
    -------
    rmse_dict : dict  {name: float}
    """
    result = {}
    for name in true_components:
        if name in estimated_components:
            result[name] = rmse(true_components[name], estimated_components[name])
        else:
            result[name] = None
    return result


def residual_diagnostics(residuals):
    """
    Compute residual summary statistics and normality test.

    Returns
    -------
    dict with mean, std, skew, kurtosis, shapiro_stat, shapiro_p
    """
    r = np.asarray(residuals, dtype=float)
    # Shapiro-Wilk only valid for n <= 5000; subsample if larger
    n = len(r)
    if n > 5000:
        rng = np.random.default_rng(42)
        r_sample = rng.choice(r, size=5000, replace=False)
    else:
        r_sample = r

    shapiro_stat, shapiro_p = stats.shapiro(r_sample)
    return {
        'mean': float(np.mean(r)),
        'std': float(np.std(r)),
        'skewness': float(stats.skew(r)),
        'kurtosis': float(stats.kurtosis(r)),
        'shapiro_stat': float(shapiro_stat),
        'shapiro_p': float(shapiro_p),
        'n': n,
    }


def summarize_results(results_dict):
    """
    Convert a dict of evaluation results to a pandas DataFrame.

    Parameters
    ----------
    results_dict : dict  {method_name: {metric: value}}

    Returns
    -------
    df : pd.DataFrame
    """
    rows = []
    for method, metrics in results_dict.items():
        row = {'Method': method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Method')

