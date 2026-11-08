"""
Benchmark Methods Module
========================
Implements or wraps benchmark time-series decomposition methods:
  - Traditional STL (statsmodels)
  - Library MSTL (statsmodels)
  - TBATS
  - Prophet (optional, if installed)

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import numpy as np
import pandas as pd
import time
import traceback
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# STL (single seasonal period)
# ---------------------------------------------------------------------------

def run_stl(series, period=24, seasonal=25, **kwargs):
    """
    Run statsmodels STL with a single seasonal period.

    Returns
    -------
    result dict: trend, seasonal, remainder, time_sec
    """
    from statsmodels.tsa.seasonal import STL

    x = np.asarray(series, dtype=float)
    t0 = time.perf_counter()
    stl = STL(x, period=period, seasonal=seasonal, **kwargs)
    res = stl.fit()
    elapsed = time.perf_counter() - t0

    return {
        'method': 'STL',
        'trend': res.trend,
        'seasonal': res.seasonal,
        'remainder': res.resid,
        'time_sec': elapsed,
        'period': period,
    }


# ---------------------------------------------------------------------------
# Library MSTL (statsmodels)
# ---------------------------------------------------------------------------

def run_mstl_library(series, periods=(24, 168), windows=None, iterations=2):
    """
    Run statsmodels MSTL.

    Parameters
    ----------
    series   : array-like
    periods  : tuple/list of int
    windows  : tuple/list of int or None (defaults to statsmodels default)
    iterations : int

    Returns
    -------
    result dict
    """
    from statsmodels.tsa.seasonal import MSTL

    x = np.asarray(series, dtype=float)

    kwargs = dict(periods=periods, iterate=iterations)
    if windows is not None:
        kwargs['windows'] = windows

    t0 = time.perf_counter()
    model = MSTL(x, **kwargs)
    res = model.fit()
    elapsed = time.perf_counter() - t0

    # Extract seasonal components per period
    seasonal_dict = {}
    if hasattr(res.seasonal, 'shape') and res.seasonal.ndim == 2:
        for i, m in enumerate(periods):
            seasonal_dict[m] = res.seasonal[:, i]
    else:
        # Single period or already 1-D
        seasonal_dict[periods[0]] = np.asarray(res.seasonal)

    return {
        'method': 'MSTL_library',
        'trend': np.asarray(res.trend),
        'seasonal': seasonal_dict,
        'remainder': np.asarray(res.resid),
        'time_sec': elapsed,
        'periods': list(periods),
    }


# ---------------------------------------------------------------------------
# TBATS
# ---------------------------------------------------------------------------

def run_tbats(series, seasonal_periods=(24, 168), use_box_cox=False,
              max_obs=2000, timeout_sec=600):
    """
    Fit TBATS and extract approximate trend and seasonal components.

    TBATS does not natively export STL-style components, so we extract
    the fitted values and residuals only.

    Parameters
    ----------
    series         : array-like
    seasonal_periods : tuple of int
    use_box_cox    : bool
    max_obs        : int   – truncate series for speed (TBATS is O(n^2))
    timeout_sec    : float – not enforced; documented as a warning

    Returns
    -------
    result dict or error dict
    """
    try:
        from tbats import TBATS as TBATSEstimator

        x = np.asarray(series, dtype=float)
        if len(x) > max_obs:
            print(f"[TBATS] Truncating series to {max_obs} obs for speed.")
            x = x[:max_obs]

        t0 = time.perf_counter()
        estimator = TBATSEstimator(
            seasonal_periods=list(seasonal_periods),
            use_box_cox=use_box_cox,
            use_trend=True,
            use_damped_trend=False,
            use_arma_errors=False,
            n_jobs=1,
        )
        model = estimator.fit(x)
        elapsed = time.perf_counter() - t0

        # tbats model returns in-sample fitted values in y_hat and residuals in resid
        if hasattr(model, 'y_hat') and model.y_hat is not None:
            fitted = model.y_hat
        elif hasattr(model, 'fitted_values') and model.fitted_values is not None:
            fitted = model.fitted_values
        else:
            fitted = x - model.resid if hasattr(model, 'resid') else x
        remainder = model.resid if hasattr(model, 'resid') else (x - fitted)

        return {
            'method': 'TBATS',
            'fitted': fitted,
            'remainder': remainder,
            'time_sec': elapsed,
            'aic': model.aic,
            'n_obs': len(x),
        }

    except ImportError:
        return {'method': 'TBATS', 'error': 'tbats package not installed.'}
    except Exception as e:
        return {'method': 'TBATS', 'error': str(e), 'traceback': traceback.format_exc()}


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def run_prophet(series, freq='h', periods=(24, 168)):
    """
    Fit Prophet model with daily and weekly seasonality.

    Prophet is an additive decomposition model from Meta/Facebook.

    Returns
    -------
    result dict or error dict
    """
    try:
        from prophet import Prophet

        x = np.asarray(series, dtype=float)
        if isinstance(series, pd.Series) and series.index.dtype == 'datetime64[ns]':
            ds = series.index.tz_localize(None) if series.index.tz else series.index
        else:
            ds = pd.date_range(start='2014-01-01', periods=len(x), freq=freq)

        df_prophet = pd.DataFrame({'ds': ds, 'y': x})

        t0 = time.perf_counter()
        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='additive',
        )
        m.fit(df_prophet)
        forecast = m.predict(df_prophet)
        elapsed = time.perf_counter() - t0

        trend = forecast['trend'].values
        seasonal = forecast.get('daily', pd.Series(np.zeros(len(x)))).values + \
                   forecast.get('weekly', pd.Series(np.zeros(len(x)))).values
        fitted = forecast['yhat'].values
        remainder = x - fitted

        return {
            'method': 'Prophet',
            'trend': trend,
            'seasonal': seasonal,
            'fitted': fitted,
            'remainder': remainder,
            'time_sec': elapsed,
        }

    except ImportError:
        return {'method': 'Prophet', 'error': 'prophet package not installed.'}
    except Exception as e:
        return {'method': 'Prophet', 'error': str(e)}
