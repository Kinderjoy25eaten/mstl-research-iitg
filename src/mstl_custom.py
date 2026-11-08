"""
Custom MSTL Implementation
==========================
Implements the MSTL algorithm from:
  Bandara, K., Hyndman, R.J., & Bergmeir, C. (2021).
  "MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series
   with Multiple Seasonal Patterns." arXiv:2107.13462

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")


def _boxcox(x, lam):
    """Apply Box-Cox transformation."""
    if lam == 0:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def _inv_boxcox(x, lam):
    """Invert Box-Cox transformation."""
    if lam == 0:
        return np.exp(x)
    return np.power(lam * x + 1.0, 1.0 / lam)


def _interpolate_missing(x):
    """
    Linearly interpolate NaN values in a 1-D array.
    Returns the interpolated array and a boolean mask of original NaN positions.
    """
    x = np.array(x, dtype=float)
    nan_mask = np.isnan(x)
    if nan_mask.any():
        t = np.arange(len(x))
        valid = ~nan_mask
        f = interpolate.interp1d(t[valid], x[valid],
                                 kind='linear',
                                 bounds_error=False,
                                 fill_value=(x[valid][0], x[valid][-1]))
        x[nan_mask] = f(t[nan_mask])
    return x, nan_mask


def _filter_seasonal_periods(periods, n):
    """
    Remove seasonal periods that are too long to exhibit seasonality.
    Following the paper: a period m is kept only if 2*m < n, i.e. n > 2*m.
    """
    return [m for m in periods if 2 * m < n]


class MSTLCustom:
    """
    Custom implementation of MSTL (Multiple Seasonal-Trend decomposition
    using LOESS) following Algorithm 1 of Bandara et al. (2021).

    Parameters
    ----------
    periods : list of int
        Seasonal periods, e.g. [24, 168] for hourly data with
        daily (24h) and weekly (168h) seasonality.
    windows : list of int or None
        Seasonal LOESS window widths for each period (must be odd).
        If None, defaults to 2*periods[i] + 1 for each period.
    iterations : int
        Number of outer MSTL iterations (K in the paper). Default 2.
    stl_kwargs : dict
        Additional keyword arguments passed to STL for every period.
    lambda_boxcox : float or None
        Box-Cox lambda. None = no transformation. 0 = log transform.
    """

    def __init__(self, periods, windows=None, iterations=2,
                 stl_kwargs=None, lambda_boxcox=None):
        self.periods = sorted(periods)       # ascending order per paper
        self.iterations = iterations
        self.lambda_boxcox = lambda_boxcox
        self.stl_kwargs = stl_kwargs or {}

        if windows is None:
            # Default: next odd integer >= 2*period + 1
            self.windows = []
            for m in self.periods:
                w = 2 * m + 1
                if w % 2 == 0:
                    w += 1
                self.windows.append(w)
        else:
            self.windows = windows
            if len(self.windows) != len(self.periods):
                raise ValueError("Length of windows must match length of periods.")

    def fit(self, x):
        """
        Decompose time series x into trend, seasonal components, and remainder.

        Parameters
        ----------
        x : array-like, shape (n,)

        Returns
        -------
        result : dict with keys:
            'trend'      : ndarray, estimated trend T(t)
            'seasonal'   : dict  {period: ndarray}, each seasonal component S_i(t)
            'remainder'  : ndarray, R(t)
            'fitted'     : ndarray, reconstructed series (T + ΣS + R)
            'original'   : ndarray, original x (after interpolation)
            'nan_mask'   : boolean ndarray, True where x was NaN
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        # --- Step 1: Identify valid seasonal periods ---
        valid_periods = _filter_seasonal_periods(self.periods, n)
        if len(valid_periods) == 0:
            raise ValueError(
                f"No seasonal period satisfies 2*m < n (n={n}). "
                f"Use longer series or smaller periods."
            )
        # Align windows with valid periods
        valid_windows = [
            self.windows[self.periods.index(m)] for m in valid_periods
        ]

        # --- Step 2: Interpolate missing observations ---
        x_interp, nan_mask = _interpolate_missing(x.copy())

        # --- Step 3: Box-Cox transformation (optional) ---
        if self.lambda_boxcox is not None:
            if np.any(x_interp <= 0):
                raise ValueError(
                    "Box-Cox requires all values > 0. "
                    "Check the series or the lambda parameter."
                )
            x_work = _boxcox(x_interp, self.lambda_boxcox)
        else:
            x_work = x_interp.copy()

        # --- Step 4: Initialise seasonal components to zero ---
        S = {m: np.zeros(n) for m in valid_periods}

        # --- Step 5: Iterative MSTL (Algorithm 1) ---
        trend = np.zeros(n)

        for iteration in range(self.iterations):
            for idx, (m, w) in enumerate(zip(valid_periods, valid_windows)):
                # De-seasonalise using all *other* seasonal components
                x_adj = x_work.copy()
                for m2, s2 in S.items():
                    if m2 != m:
                        x_adj = x_adj - s2

                # Apply STL to the adjusted series for seasonal period m
                stl = STL(
                    x_adj,
                    period=m,
                    seasonal=w,
                    **self.stl_kwargs
                )
                res = stl.fit(inner_iter=1, outer_iter=0)

                S[m] = res.seasonal
                trend = res.trend  # update trend from each STL call

        # --- Step 6: Trend from last STL ---
        # (Already stored as trend from the last STL call above)

        # --- Step 7: Remainder ---
        sum_seasonal = np.sum([S[m] for m in valid_periods], axis=0)
        remainder = x_work - trend - sum_seasonal

        # --- Step 8: Inverse Box-Cox (if applied) ---
        if self.lambda_boxcox is not None:
            trend = _inv_boxcox(trend, self.lambda_boxcox)
            for m in valid_periods:
                S[m] = _inv_boxcox(S[m], self.lambda_boxcox)
            remainder = _inv_boxcox(remainder, self.lambda_boxcox)

        # --- Step 9: Verification (reconstruction) ---
        reconstructed = trend + sum(S[m] for m in valid_periods) + remainder
        recon_error = np.max(np.abs(reconstructed - x_interp))

        self.result_ = {
            'trend': trend,
            'seasonal': S,
            'remainder': remainder,
            'fitted': reconstructed,
            'original': x_interp,
            'nan_mask': nan_mask,
            'reconstruction_max_error': recon_error,
            'valid_periods': valid_periods,
        }
        return self

    def get_components(self):
        """Return the decomposition result dict."""
        if not hasattr(self, 'result_'):
            raise RuntimeError("Call fit() first.")
        return self.result_

    def reconstruction_error(self):
        """Maximum absolute difference between reconstructed and original."""
        return self.result_['reconstruction_max_error']

    def summary(self):
        """Print a brief summary of the decomposition."""
        r = self.result_
        print("=" * 55)
        print("  Custom MSTL Decomposition Summary")
        print("=" * 55)
        print(f"  Periods used   : {r['valid_periods']}")
        print(f"  Windows used   : {self.windows}")
        print(f"  Iterations     : {self.iterations}")
        print(f"  Series length  : {len(r['original'])}")
        print(f"  Missing obs.   : {r['nan_mask'].sum()}")
        print(f"  Recon. max err : {r['reconstruction_max_error']:.6e}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Convenience function (mirrors the statsmodels API style)
# ---------------------------------------------------------------------------

def mstl_decompose(series, periods, windows=None, iterations=2,
                   stl_kwargs=None, lambda_boxcox=None):
    """
    Convenience wrapper around MSTLCustom.

    Parameters
    ----------
    series : array-like or pd.Series
    periods : list of int
    windows : list of int or None
    iterations : int
    stl_kwargs : dict or None
    lambda_boxcox : float or None

    Returns
    -------
    result dict (see MSTLCustom.fit docstring)
    """
    model = MSTLCustom(
        periods=periods,
        windows=windows,
        iterations=iterations,
        stl_kwargs=stl_kwargs,
        lambda_boxcox=lambda_boxcox,
    )
    if isinstance(series, pd.Series):
        x = series.values.astype(float)
    else:
        x = np.asarray(series, dtype=float)
    model.fit(x)
    return model.get_components()

