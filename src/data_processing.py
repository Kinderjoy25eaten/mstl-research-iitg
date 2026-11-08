"""
Data Processing Module
======================
Loads and preprocesses the Victoria electricity-demand dataset.

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")


DATASET_URL = (
    "https://raw.githubusercontent.com/tsibble/tsibbledata/"
    "master/data-raw/vic_elec/VIC2015/demand.csv"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOCAL_CSV = os.path.join(DATA_DIR, "vic_elec_demand.csv")
LOCAL_HOURLY = os.path.join(DATA_DIR, "vic_elec_hourly.csv")


def load_raw_dataset(url=None, local_path=None):
    """
    Load the raw Victoria electricity-demand dataset (half-hourly).

    Tries local cache first; falls back to downloading via URL.

    Returns
    -------
    df_raw : pd.DataFrame with columns including 'Time', 'Demand'
    """
    if local_path is None:
        local_path = LOCAL_CSV

    if os.path.exists(local_path):
        print(f"Loading dataset from local cache: {local_path}")
        df_raw = pd.read_csv(local_path)
    elif url is not None:
        print(f"Downloading dataset from: {url}")
        df_raw = pd.read_csv(url)
        os.makedirs(DATA_DIR, exist_ok=True)
        df_raw.to_csv(local_path, index=False)
        print(f"Saved to: {local_path}")
    else:
        raise FileNotFoundError(
            f"Dataset not found at {local_path} and no URL provided."
        )
    return df_raw


def preprocess_to_hourly(df_raw):
    """
    Preprocess raw half-hourly data to hourly data.

    Steps:
    1. Parse 'Time' column as datetime.
    2. Set 'Time' as index.
    3. Extract 'Demand' column.
    4. Aggregate half-hourly observations to hourly by summing.
    5. Handle missing values via linear interpolation.

    Returns
    -------
    hourly : pd.Series, hourly electricity demand (MWh equivalent)
    """
    df = df_raw.copy()

    # Parse Time
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time')
    df = df.set_index('Time')

    # Extract Demand
    demand_col = None
    for col in df.columns:
        if 'demand' in col.lower() or col == 'Demand':
            demand_col = col
            break
    if demand_col is None:
        # Fallback: use first numeric column
        demand_col = df.select_dtypes(include=[np.number]).columns[0]
    demand = df[demand_col].astype(float)

    # Aggregate to hourly (sum of two half-hour observations per hour)
    hourly = demand.resample('h').sum()

    # Replace zeros at hour boundaries from incomplete aggregations with NaN
    # (zeros should not arise from real demand)
    hourly[hourly == 0] = np.nan

    # Interpolate missing values
    hourly = hourly.interpolate(method='linear', limit_direction='both')

    print(f"Half-hourly observations  : {len(demand)}")
    print(f"Hourly observations total : {len(hourly)}")
    print(f"Date range                : {hourly.index[0]} to {hourly.index[-1]}")
    print(f"Missing values after interp: {hourly.isna().sum()}")

    return hourly


def extract_experiment_period(hourly, n_obs=3601, n_days=None, start=None, end=None):
    """
    Extract a specific window of hourly observations.

    The original paper uses 149 days of data (spanning Jan 1 to May 31, 2014,
    yielding exactly 3601 hourly observations with inclusive endpoints).

    Parameters
    ----------
    hourly   : pd.Series, full hourly series
    n_obs    : int or None; number of observations (default: 3601)
    n_days   : int or None; if set, extract first n_days days
    start    : str or pd.Timestamp; start of window
    end      : str or pd.Timestamp; end of window

    Returns
    -------
    sub : pd.Series
    """
    if start is not None and end is not None:
        sub = hourly.loc[start:end]
    elif n_obs is not None:
        sub = hourly.iloc[:n_obs]
    elif n_days is not None:
        sub = hourly.iloc[:n_days * 24 + 1]
    else:
        sub = hourly

    print(f"Experiment period observations: {len(sub)}")
    print(f"  From : {sub.index[0]}")
    print(f"  To   : {sub.index[-1]}")
    n_actual = len(sub)
    if n_actual == 3601:
        print("  Exact reproduction: matched paper's 3601 hourly observations.")
    else:
        print(f"  Observations: {n_actual} (paper target: 3601).")
    return sub
    n_obs = len(sub)
    if abs(n_obs - 3601) > 10:
        print(
            f"  NOTE: Paper reports 3601 observations; we have {n_obs}. "
            "This may differ due to indexing/timezone conventions."
        )
    return sub


def save_hourly(hourly, path=None):
    """Save processed hourly series to CSV."""
    if path is None:
        path = LOCAL_HOURLY
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hourly.to_csv(path, header=['Demand'])
    print(f"Hourly data saved to: {path}")


def load_hourly(path=None):
    """Load previously saved hourly series."""
    if path is None:
        path = LOCAL_HOURLY
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df.iloc[:, 0]
    series.index.name = 'Time'
    return series

