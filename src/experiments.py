"""
Experiments Module
==================
Implements all four extensions and the synthetic-data experiment.

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

from mstl_custom import MSTLCustom, mstl_decompose
from evaluation import rmse, reconstruction_error, component_wise_rmse


# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------

def generate_synthetic(n=1680, seed=42):
    """
    Generate a synthetic hourly time series with known components.

    X(t) = T(t) + S_daily(t) + S_weekly(t) + R(t)

    Parameters
    ----------
    n    : int   number of hourly observations (default = 70 days)
    seed : int   random seed

    Returns
    -------
    dict with keys: x, trend, seasonal_daily, seasonal_weekly, remainder, t
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Trend: smooth quadratic + slow sine
    trend = 500 + 0.05 * t + 200 * np.sin(2 * np.pi * t / (24 * 365))

    # Daily seasonality (period=24)
    s_daily = 300 * np.sin(2 * np.pi * t / 24 + 1.0) + \
              100 * np.cos(4 * np.pi * t / 24)

    # Weekly seasonality (period=168)
    s_weekly = 150 * np.sin(2 * np.pi * t / 168) + \
               50 * np.cos(4 * np.pi * t / 168 + 0.5)

    # Remainder: Gaussian noise
    remainder = rng.normal(0, 30, n)

    x = trend + s_daily + s_weekly + remainder

    return {
        'x': x,
        'trend': trend,
        'seasonal_daily': s_daily,
        'seasonal_weekly': s_weekly,
        'remainder': remainder,
        't': t,
        'n': n,
        'seed': seed,
    }


def evaluate_on_synthetic(synth, periods=(24, 168), iterations=2):
    """
    Decompose the synthetic series and compute component-wise RMSE.

    Returns
    -------
    result dict: estimated components + rmse_dict
    """
    result = mstl_decompose(synth['x'], periods=list(periods),
                            iterations=iterations)

    true = {
        'trend': synth['trend'],
        'seasonal_daily': synth['seasonal_daily'],
        'seasonal_weekly': synth['seasonal_weekly'],
        'remainder': synth['remainder'],
    }
    est = {
        'trend': result['trend'],
        'seasonal_daily': result['seasonal'].get(24, np.zeros(synth['n'])),
        'seasonal_weekly': result['seasonal'].get(168, np.zeros(synth['n'])),
        'remainder': result['remainder'],
    }
    rmse_dict = component_wise_rmse(true, est)

    return {'result': result, 'rmse': rmse_dict, 'true': true, 'estimated': est}


# ---------------------------------------------------------------------------
# EXTENSION 1 — Seasonal Window & Iteration Sensitivity
# ---------------------------------------------------------------------------

def extension1_sensitivity(x, periods=(24, 168),
                            windows_to_try=(7, 11, 15, 21, 25),
                            iterations_to_try=(1, 2, 3)):
    """
    Grid search over seasonal windows and outer iterations.
    Uses reconstruction RMSE as proxy metric.

    Returns
    -------
    df_rmse  : pd.DataFrame  (windows × iterations)
    df_time  : pd.DataFrame  (windows × iterations)
    """
    x = np.asarray(x, dtype=float)
    rmse_rows = []
    time_rows = []

    for s_win in windows_to_try:
        # Ensure window is odd and > 1
        if s_win % 2 == 0:
            s_win += 1
        rmse_row = []
        time_row = []
        for n_iter in iterations_to_try:
            try:
                # Apply same window to all periods
                wins = [s_win] * len(periods)
                t0 = time.perf_counter()
                result = mstl_decompose(x, periods=list(periods),
                                        windows=wins,
                                        iterations=n_iter)
                elapsed = time.perf_counter() - t0

                # Reconstruction RMSE
                sum_s = np.sum(list(result['seasonal'].values()), axis=0)
                reconstructed = result['trend'] + sum_s + result['remainder']
                recon_rmse = rmse(result['original'], reconstructed)
                rmse_row.append(recon_rmse)
                time_row.append(elapsed)
            except Exception as e:
                rmse_row.append(np.nan)
                time_row.append(np.nan)
        rmse_rows.append(rmse_row)
        time_rows.append(time_row)

    index = [str(w) for w in windows_to_try]
    cols = [str(it) for it in iterations_to_try]
    df_rmse = pd.DataFrame(rmse_rows, index=index, columns=cols)
    df_time = pd.DataFrame(time_rows, index=index, columns=cols)
    df_rmse.index.name = "SeasonalWindow"
    df_time.index.name = "SeasonalWindow"
    return df_rmse, df_time


# ---------------------------------------------------------------------------
# EXTENSION 2 — Ablation Study of Multiple Seasonalities
# ---------------------------------------------------------------------------

def extension2_ablation(x):
    """
    Compare three MSTL configurations:
    1. Daily only (period=24)
    2. Weekly only (period=168)
    3. Both (periods=24, 168)

    Returns
    -------
    dict  {config_name: {'result': ..., 'recon_rmse': float, 'time_sec': float}}
    """
    configs = {
        'Daily only (24h)': [24],
        'Weekly only (168h)': [168],
        'Both (24h + 168h)': [24, 168],
    }
    results = {}
    for name, periods in configs.items():
        try:
            t0 = time.perf_counter()
            result = mstl_decompose(x, periods=periods, iterations=2)
            elapsed = time.perf_counter() - t0

            sum_s = np.sum(list(result['seasonal'].values()), axis=0)
            reconstructed = result['trend'] + sum_s + result['remainder']
            recon_rmse = rmse(result['original'], reconstructed)

            results[name] = {
                'result': result,
                'recon_rmse': recon_rmse,
                'reconstruction_rmse': recon_rmse,
                'time_sec': elapsed,
                'periods': periods,
                'remainder_std': float(np.std(result['remainder'])),
                'remainder_mean': float(np.mean(result['remainder'])),
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    return results


# ---------------------------------------------------------------------------
# EXTENSION 3 — Robustness to Outliers and Missing Data
# ---------------------------------------------------------------------------

def _add_outliers(x, frac=0.02, magnitude=5.0, seed=43):
    """Introduce random spikes into a copy of x."""
    rng = np.random.default_rng(seed)
    xc = x.copy()
    n = len(xc)
    n_outliers = int(n * frac)
    idx = rng.choice(n, size=n_outliers, replace=False)
    xc[idx] += magnitude * np.std(x) * rng.choice([-1, 1], size=n_outliers)
    return xc, idx


def _add_missing(x, frac=0.05, seed=44):
    """Randomly remove observations (set to NaN)."""
    rng = np.random.default_rng(seed)
    xc = x.copy().astype(float)
    n = len(xc)
    idx = rng.choice(n, size=int(n * frac), replace=False)
    xc[idx] = np.nan
    return xc, idx


def extension3_robustness(x, periods=(24, 168),
                           outlier_fracs=(0.02, 0.05, 0.10),
                           missing_fracs=(0.02, 0.05, 0.10)):
    """
    Test MSTL under outliers and missing data.

    Returns
    -------
    df_outlier  : pd.DataFrame  summary per outlier fraction
    df_missing  : pd.DataFrame  summary per missing fraction
    corrupted_examples : dict  (for plotting)
    """
    x = np.asarray(x, dtype=float)
    # Baseline
    base_result = mstl_decompose(x, periods=list(periods), iterations=2)
    base_trend_std = np.std(base_result['trend'])

    outlier_rows = []
    corrupted_examples = {}

    for frac in outlier_fracs:
        xc, _ = _add_outliers(x, frac=frac)
        t0 = time.perf_counter()
        try:
            result = mstl_decompose(xc, periods=list(periods), iterations=2)
            elapsed = time.perf_counter() - t0
            trend_rmse = rmse(base_result['trend'], result['trend'])
            rem_std = np.std(result['remainder'])
            outlier_rows.append({
                'OutlierFrac': frac,
                'TrendRMSE': trend_rmse,
                'TrendRMSE_pct': trend_rmse / base_trend_std * 100,
                'RemainderStd': rem_std,
                'TimeSec': elapsed,
            })
            if frac == outlier_fracs[0]:
                corrupted_examples['outlier'] = {
                    'corrupted': xc,
                    'corrupted_result': result,
                    'clean_result': base_result,
                }
        except Exception as e:
            outlier_rows.append({'OutlierFrac': frac, 'error': str(e)})

    missing_rows = []
    for frac in missing_fracs:
        xc, _ = _add_missing(x, frac=frac)
        t0 = time.perf_counter()
        try:
            result = mstl_decompose(xc, periods=list(periods), iterations=2)
            elapsed = time.perf_counter() - t0
            trend_rmse = rmse(base_result['trend'], result['trend'])
            rem_std = np.std(result['remainder'])
            missing_rows.append({
                'MissingFrac': frac,
                'TrendRMSE': trend_rmse,
                'TrendRMSE_pct': trend_rmse / base_trend_std * 100,
                'RemainderStd': rem_std,
                'TimeSec': elapsed,
            })
            if frac == missing_fracs[0]:
                corrupted_examples['missing'] = {
                    'corrupted': xc,
                    'corrupted_result': result,
                    'clean_result': base_result,
                }
        except Exception as e:
            missing_rows.append({'MissingFrac': frac, 'error': str(e)})

    df_outlier = pd.DataFrame(outlier_rows)
    df_missing = pd.DataFrame(missing_rows)
    return df_outlier, df_missing, corrupted_examples


# ---------------------------------------------------------------------------
# EXTENSION 4 — Computational Scalability
# ---------------------------------------------------------------------------

def extension4_scalability(x, periods=(24, 168),
                            sizes_hours=(168, 720, 2160, 4320, None)):
    """
    Measure execution time and memory for different dataset sizes.

    Parameters
    ----------
    x            : full array
    periods      : seasonal periods
    sizes_hours  : list of ints (None = use full array)

    Returns
    -------
    df : pd.DataFrame with n_obs, time_sec, memory_mb per size
    """
    x = np.asarray(x, dtype=float)
    rows = []
    for sz in sizes_hours:
        xsub = x if sz is None else x[:sz]
        n = len(xsub)
        label = f"{sz}h" if sz else f"{n}h (full)"

        # Time and memory
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            result = mstl_decompose(xsub, periods=list(periods), iterations=2)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_mb = peak / 1024 / 1024
            rows.append({
                'Label': label,
                'n_obs': n,
                'time_sec': elapsed,
                'memory_mb': mem_mb,
            })
        except Exception as e:
            tracemalloc.stop()
            rows.append({'Label': label, 'n_obs': n, 'error': str(e)})

    return pd.DataFrame(rows)

