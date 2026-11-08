"""
Export experimental results and decomposition series to JSON for the Web Application.

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
Affiliation: Mehta Family School of Data Science and AI, IIT Guwahati
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_DIR = os.path.join(BASE_DIR, "web")
WEB_DATA_DIR = os.path.join(WEB_DIR, "data")
os.makedirs(WEB_DATA_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from data_processing import load_hourly, extract_experiment_period
from mstl_custom import mstl_decompose
from experiments import generate_synthetic, evaluate_on_synthetic

print("Exporting data for web application...")

# 1. Victoria Electricity Demand (3601 obs)
hourly = load_hourly(os.path.join(BASE_DIR, "data", "vic_elec_hourly.csv"))
series = extract_experiment_period(hourly, n_obs=3601)
x = series.values.astype(float)
dates = [str(t) for t in series.index]

res = mstl_decompose(x, periods=[24, 168], iterations=2)

vic_data = {
    "timestamps": dates,
    "original": [round(float(v), 2) for v in res['original']],
    "trend": [round(float(v), 2) for v in res['trend']],
    "seasonal_24": [round(float(v), 2) for v in res['seasonal'][24]],
    "seasonal_168": [round(float(v), 2) for v in res['seasonal'][168]],
    "remainder": [round(float(v), 2) for v in res['remainder']],
    "n_obs": len(x),
    "reconstruction_error": float(res['reconstruction_max_error'])
}

with open(os.path.join(WEB_DATA_DIR, "vic_elec_decomposition.json"), "w") as f:
    json.dump(vic_data, f)
print("Saved vic_elec_decomposition.json")

# 2. Synthetic Data (1680 obs)
synth = generate_synthetic(n=1680, seed=42)
synth_eval = evaluate_on_synthetic(synth, periods=(24, 168), iterations=2)
syn_res = synth_eval['result']

synth_data = {
    "hours": list(range(synth['n'])),
    "observed": [round(float(v), 2) for v in synth['x']],
    "true_trend": [round(float(v), 2) for v in synth['trend']],
    "true_daily": [round(float(v), 2) for v in synth['seasonal_daily']],
    "true_weekly": [round(float(v), 2) for v in synth['seasonal_weekly']],
    "true_noise": [round(float(v), 2) for v in synth['remainder']],
    "mstl_trend": [round(float(v), 2) for v in syn_res['trend']],
    "mstl_daily": [round(float(v), 2) for v in syn_res['seasonal'][24]],
    "mstl_weekly": [round(float(v), 2) for v in syn_res['seasonal'][168]],
    "mstl_remainder": [round(float(v), 2) for v in syn_res['remainder']],
    "rmse": {k: round(float(v), 4) for k, v in synth_eval['rmse'].items()}
}

with open(os.path.join(WEB_DATA_DIR, "synthetic_decomposition.json"), "w") as f:
    json.dump(synth_data, f)
print("Saved synthetic_decomposition.json")

# 3. Experiment Log & Summary
log_path = os.path.join(BASE_DIR, "results", "experiment_logs", "experiment_log.json")
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log_data = json.load(f)
    with open(os.path.join(WEB_DATA_DIR, "experiment_summary.json"), "w") as f:
        json.dump(log_data, f, indent=2)
    print("Saved experiment_summary.json")

print("All web datasets exported successfully!")

