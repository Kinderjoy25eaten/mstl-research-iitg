"""
Master Experiment Runner
========================
Runs all experiments end-to-end and saves results.

Usage:
    cd MSTL_Project
    python src/run_all_experiments.py

Authors: Rup Narayan Jha (240150032), Jaypal Malviya (240150013),
         Anuj Kumar (240150003), Ravi Kiran (240150029)
"""

import sys
import os

# Ensure src is on path
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import time
import json
import warnings
warnings.filterwarnings("ignore")

from data_processing import (load_raw_dataset, preprocess_to_hourly,
                              extract_experiment_period, save_hourly, load_hourly)
from mstl_custom import MSTLCustom, mstl_decompose
from benchmarks import run_stl, run_mstl_library, run_tbats
from evaluation import (rmse, reconstruction_error, residual_diagnostics,
                        component_wise_rmse, summarize_results)
from experiments import (generate_synthetic, evaluate_on_synthetic,
                          extension1_sensitivity, extension2_ablation,
                          extension3_robustness, extension4_scalability)
from visualization import (plot_original_series, plot_two_weeks,
                            plot_mstl_decomposition, plot_component_comparison,
                            plot_residual_diagnostics, plot_sensitivity_heatmap,
                            plot_ablation, plot_scalability,
                            plot_synthetic_rmse, plot_robustness, FIGURES_DIR)

TABLES_DIR = os.path.join(SRC, '..', 'results', 'tables')
LOGS_DIR = os.path.join(SRC, '..', 'results', 'experiment_logs')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

PERIODS = [24, 168]

LOG = {}


def log_section(name):
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)


# ===========================================================================
# STAGE 1  —  Data
# ===========================================================================
log_section("STAGE 1: Data Preprocessing")

DATASET_URL = (
    "https://raw.githubusercontent.com/tsibble/tsibbledata/"
    "master/data-raw/vic_elec/VIC2015/demand.csv"
)
LOCAL_CSV = os.path.join(SRC, '..', 'data', 'vic_elec_demand.csv')
LOCAL_HOURLY = os.path.join(SRC, '..', 'data', 'vic_elec_hourly.csv')

if os.path.exists(LOCAL_HOURLY):
    print("Loading cached hourly data...")
    hourly = load_hourly(LOCAL_HOURLY)
else:
    df_raw = load_raw_dataset(url=DATASET_URL, local_path=LOCAL_CSV)
    print(f"\nRaw dataset shape: {df_raw.shape}")
    print(df_raw.head())
    hourly = preprocess_to_hourly(df_raw)
    save_hourly(hourly, LOCAL_HOURLY)

# Extract experiment window (3601 observations as reported in paper: 2014-01-01 to 2014-05-31)
series = extract_experiment_period(hourly, n_obs=3601)
x = series.values.astype(float)

LOG['n_obs'] = len(x)
LOG['start'] = str(series.index[0])
LOG['end'] = str(series.index[-1])

print(f"\nFinal series length: {len(x)}")

# Visualise
print("\nGenerating visualizations...")
plot_original_series(series)
plot_two_weeks(series)


# ===========================================================================
# STAGE 2  —  Custom MSTL
# ===========================================================================
log_section("STAGE 2: Custom MSTL Implementation")

t0 = time.perf_counter()
custom_result = mstl_decompose(x, periods=PERIODS, iterations=2)
custom_time = time.perf_counter() - t0

print(f"Custom MSTL time: {custom_time:.3f}s")
print(f"Reconstruction max error: {custom_result['reconstruction_max_error']:.4e}")

plot_mstl_decomposition(series, custom_result, periods=PERIODS,
                         title_prefix="Custom MSTL",
                         filename="fig_03_mstl_custom_decomposition.png")

rem_diag = residual_diagnostics(custom_result['remainder'])
LOG['custom_mstl'] = {
    'time_sec': custom_time,
    'recon_max_error': custom_result['reconstruction_max_error'],
    'residual': rem_diag,
}
print("Residual diagnostics (Custom MSTL):", rem_diag)

plot_residual_diagnostics(custom_result['remainder'], method_name="Custom MSTL",
                           filename="fig_12_residual_custom_mstl.png")


# ===========================================================================
# STAGE 3  —  Library MSTL
# ===========================================================================
log_section("STAGE 3: Library MSTL (statsmodels)")

lib_result = run_mstl_library(x, periods=PERIODS, iterations=2)
print(f"Library MSTL time: {lib_result['time_sec']:.3f}s")

plot_mstl_decomposition(
    series,
    {'trend': lib_result['trend'],
     'seasonal': lib_result['seasonal'],
     'remainder': lib_result['remainder'],
     'original': x,
     'reconstruction_max_error': 0},
    periods=PERIODS,
    title_prefix="Library MSTL (statsmodels)",
    filename="fig_04_mstl_library_decomposition.png"
)

plot_component_comparison(series.index,
                           custom_result['trend'],
                           lib_result['trend'],
                           comp_name="Trend",
                           filename="fig_05_trend_comparison.png")

LOG['library_mstl'] = {'time_sec': lib_result['time_sec']}


# ===========================================================================
# STAGE 4  —  Benchmarks: STL and TBATS
# ===========================================================================
log_section("STAGE 4: Benchmark Methods")

# Traditional STL (daily period=24)
stl_result = run_stl(x, period=24, seasonal=25)
print(f"STL (period=24) time: {stl_result['time_sec']:.3f}s")
LOG['stl'] = {'time_sec': stl_result['time_sec']}

plot_mstl_decomposition(
    series,
    {'trend': stl_result['trend'],
     'seasonal': {24: stl_result['seasonal']},
     'remainder': stl_result['remainder'],
     'original': x,
     'reconstruction_max_error': 0},
    periods=[24],
    title_prefix="STL (period=24)",
    filename="fig_06_stl_decomposition.png"
)

# TBATS (truncated to 500 observations for tractable execution time)
print("\nRunning TBATS (500 observations)...")
tbats_result = run_tbats(x, seasonal_periods=(24, 168), max_obs=500)
if 'error' not in tbats_result:
    print(f"TBATS time: {tbats_result['time_sec']:.3f}s  |  AIC: {tbats_result.get('aic','N/A')}")
    LOG['tbats'] = {
        'time_sec': tbats_result['time_sec'],
        'aic': float(tbats_result.get('aic', 0)),
        'n_obs': tbats_result['n_obs'],
    }
else:
    print(f"TBATS error: {tbats_result['error']}")
    LOG['tbats'] = {'error': tbats_result['error']}


# ===========================================================================
# STAGE 5  —  Synthetic Data Experiment
# ===========================================================================
log_section("STAGE 5: Synthetic Data Experiment")

synth = generate_synthetic(n=1680, seed=42)
print(f"Synthetic series length: {synth['n']}")

synth_eval = evaluate_on_synthetic(synth, periods=(24, 168), iterations=2)
print("Component-wise RMSE (Custom MSTL on synthetic):")
for k, v in synth_eval['rmse'].items():
    print(f"  {k}: {v:.4f}")

LOG['synthetic_rmse'] = synth_eval['rmse']

# Visualise synthetic decomposition
syn_result = synth_eval['result']
n_synth = synth['n']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
axes[0].plot(synth['x'], color='steelblue', linewidth=0.7, label='Synthetic X(t)')
axes[0].set_ylabel("X(t)")
axes[0].set_title("Synthetic Series Decomposition — Custom MSTL", fontsize=13, fontweight='bold')

axes[1].plot(synth['trend'], color='gray', linewidth=1.0, linestyle='--', label='True Trend')
axes[1].plot(syn_result['trend'], color='crimson', linewidth=1.0, label='MSTL Trend')
axes[1].set_ylabel("Trend")
axes[1].legend(loc='best', fontsize=8)

axes[2].plot(synth['seasonal_daily'], color='gray', linestyle='--', linewidth=0.8, label='True S(24h)')
axes[2].plot(syn_result['seasonal'].get(24, []), color='seagreen', linewidth=0.8, label='MSTL S(24h)')
axes[2].set_ylabel("S(24h)")
axes[2].legend(loc='best', fontsize=8)

axes[3].plot(synth['seasonal_weekly'], color='gray', linestyle='--', linewidth=0.8, label='True S(168h)')
axes[3].plot(syn_result['seasonal'].get(168, []), color='darkorange', linewidth=0.8, label='MSTL S(168h)')
axes[3].set_ylabel("S(168h)")
axes[3].legend(loc='best', fontsize=8)

axes[4].plot(synth['remainder'], color='gray', linestyle='--', linewidth=0.6, label='True Remainder')
axes[4].plot(syn_result['remainder'], color='purple', linewidth=0.6, label='MSTL Remainder')
axes[4].set_ylabel("Remainder")
axes[4].set_xlabel("Index")
axes[4].legend(loc='best', fontsize=8)

fig.tight_layout()
fname = os.path.join(FIGURES_DIR, "fig_07_synthetic_decomposition.png")
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fname}")

# RMSE bar chart
rmse_dict_for_plot = {
    'Custom MSTL': synth_eval['rmse'],
}
plot_synthetic_rmse(rmse_dict_for_plot, filename="fig_11_synthetic_rmse.png")


# ===========================================================================
# STAGE 6  —  EXTENSION 1: Sensitivity Analysis
# ===========================================================================
log_section("EXTENSION 1: Seasonal Window & Iteration Sensitivity")

df_rmse_sens, df_time_sens = extension1_sensitivity(
    x,
    periods=PERIODS,
    windows_to_try=[7, 11, 15, 21, 25],
    iterations_to_try=[1, 2, 3],
)
print("\nSensitivity RMSE table:")
print(df_rmse_sens.to_string())
print("\nSensitivity Time (sec) table:")
print(df_time_sens.to_string())

df_rmse_sens.to_csv(os.path.join(TABLES_DIR, "table_ext1_sensitivity_rmse.csv"))
df_time_sens.to_csv(os.path.join(TABLES_DIR, "table_ext1_sensitivity_time.csv"))

plot_sensitivity_heatmap(df_rmse_sens, filename="fig_ext1_sensitivity_heatmap.png")

LOG['ext1_sensitivity'] = {
    'best_window': df_rmse_sens.min().idxmin(),
    'rmse_table': df_rmse_sens.to_dict(),
}


# ===========================================================================
# STAGE 7  —  EXTENSION 2: Ablation Study
# ===========================================================================
log_section("EXTENSION 2: Ablation Study")

ablation_results = extension2_ablation(x)
for name, res in ablation_results.items():
    if 'error' not in res:
        print(f"  {name}: recon_rmse={res['recon_rmse']:.4f}  "
              f"time={res['time_sec']:.3f}s  "
              f"remainder_std={res['remainder_std']:.4f}")
    else:
        print(f"  {name}: ERROR — {res['error']}")

ablation_summary = {
    name: {
        'recon_rmse': res.get('recon_rmse', None),
        'time_sec': res.get('time_sec', None),
        'remainder_std': res.get('remainder_std', None),
    }
    for name, res in ablation_results.items() if 'error' not in res
}
pd.DataFrame(ablation_summary).T.to_csv(
    os.path.join(TABLES_DIR, "table_ext2_ablation.csv")
)
plot_ablation(
    {k: {'reconstruction_rmse': v['recon_rmse']}
     for k, v in ablation_results.items() if 'error' not in v},
    filename="fig_ext2_ablation.png"
)

LOG['ext2_ablation'] = ablation_summary

# Visualise ablation components
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
colors_ = ['steelblue', 'darkorange', 'seagreen']
for i, (name, res) in enumerate(ablation_results.items()):
    if 'error' not in res:
        axes[i].plot(res['result']['trend'], color=colors_[i], linewidth=0.9)
        axes[i].set_title(f"Trend — {name}", fontsize=11)
        axes[i].set_ylabel("Trend")
axes[-1].set_xlabel("Index")
fig.suptitle("Ablation Study: Estimated Trends", fontsize=13, fontweight='bold')
fig.tight_layout()
abl_fig = os.path.join(FIGURES_DIR, "fig_ext2_ablation_trends.png")
fig.savefig(abl_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {abl_fig}")


# ===========================================================================
# STAGE 8  —  EXTENSION 3: Robustness
# ===========================================================================
log_section("EXTENSION 3: Robustness to Outliers and Missing Data")

df_outlier, df_missing, examples = extension3_robustness(
    x, periods=PERIODS,
    outlier_fracs=[0.02, 0.05, 0.10],
    missing_fracs=[0.02, 0.05, 0.10],
)
print("\nOutlier robustness:")
print(df_outlier.to_string(index=False))
print("\nMissing data robustness:")
print(df_missing.to_string(index=False))

df_outlier.to_csv(os.path.join(TABLES_DIR, "table_ext3_outlier_robustness.csv"), index=False)
df_missing.to_csv(os.path.join(TABLES_DIR, "table_ext3_missing_robustness.csv"), index=False)

if 'outlier' in examples:
    ex = examples['outlier']
    plot_robustness(
        x, ex['corrupted'],
        ex['clean_result'], ex['corrupted_result'],
        filename="fig_ext3_outlier_robustness.png"
    )
if 'missing' in examples:
    ex = examples['missing']
    plot_robustness(
        x, ex['corrupted'],
        ex['clean_result'], ex['corrupted_result'],
        filename="fig_ext3_missing_robustness.png"
    )

LOG['ext3_outlier'] = df_outlier.to_dict(orient='records')
LOG['ext3_missing'] = df_missing.to_dict(orient='records')


# ===========================================================================
# STAGE 9  —  EXTENSION 4: Scalability
# ===========================================================================
log_section("EXTENSION 4: Computational Scalability")

df_scale = extension4_scalability(
    x, periods=PERIODS,
    sizes_hours=[168, 720, 1440, 2160, 3601],
)
print(df_scale.to_string(index=False))
df_scale.to_csv(os.path.join(TABLES_DIR, "table_ext4_scalability.csv"), index=False)

valid_rows = df_scale[df_scale['time_sec'].notna()]
plot_scalability(
    valid_rows['n_obs'].tolist(),
    valid_rows['time_sec'].tolist(),
    filename="fig_ext4_scalability.png"
)

LOG['ext4_scalability'] = df_scale.to_dict(orient='records')


# ===========================================================================
# STAGE 10  —  Save log
# ===========================================================================
log_section("Saving experiment log")

log_path = os.path.join(LOGS_DIR, "experiment_log.json")

# Convert numpy types for JSON
def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(log_path, 'w') as f:
    json.dump(_to_serialisable(LOG), f, indent=2)
print(f"Experiment log saved to: {log_path}")

print("\n" + "=" * 60)
print("  ALL EXPERIMENTS COMPLETE")
print("=" * 60)
print(f"  Figures saved in : {FIGURES_DIR}")
print(f"  Tables saved in  : {TABLES_DIR}")
print(f"  Logs saved in    : {LOGS_DIR}")

