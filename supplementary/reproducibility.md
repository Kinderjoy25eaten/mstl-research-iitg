# Reproducibility Guide: Step-by-Step Instructions

**Project**: MSTL Research Implementation and Extensions  
**Affiliation**: Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati  
**Program**: B.Tech in Data Science and Artificial Intelligence  
**Authors**:
- Rup Narayan Jha (240150032)
- Jaypal Malviya (240150013)
- Anuj Kumar (240150003)
- Ravi Kiran (240150029)

---

## 1. Prerequisites and Environment Setup

The codebase is written in standard Python (3.9+) and is fully compatible with Windows, macOS, Linux, and Google Colab environments.

### 1.1 Clone / Access Codebase
Ensure your current working directory is set to the project root:
```bash
cd MSTL_Project
```

### 1.2 Install Required Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```
Key packages installed:
- `numpy`, `pandas`, `scipy`
- `statsmodels` (provides STL and library MSTL)
- `tbats` (provides state-space TBATS benchmark)
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## 2. Re-executing All Experiments from Scratch

To re-run the entire research pipeline end-to-end (data loading, custom MSTL, library comparisons, synthetic evaluations, all 4 extensions, and plot/table generation):

```bash
python src/run_all_experiments.py
```

### Expected Output Logs:
- Data preprocessed to `data/vic_elec_hourly.csv`
- Exactly **3,601 hourly observations** extracted
- 15 publication figures generated under `results/figures/`
- 6 CSV tables generated under `results/tables/`
- Full experiment log saved to `results/experiment_logs/experiment_log.json`

---

## 3. Interactive Notebook Walkthrough

To inspect and interactively execute each stage:

```bash
jupyter lab notebooks/
```

Execute notebooks sequentially:
1. `01_data_preprocessing.ipynb`: Raw dataset inspection and hourly aggregation.
2. `02_custom_mstl_implementation.ipynb`: Step-by-step custom MSTL decomposition and residual checks.
3. `03_library_and_benchmark_comparison.ipynb`: Statsmodels MSTL, STL, and TBATS comparisons.
4. `04_synthetic_data_experiments.ipynb`: Synthetic signal generation and ground-truth RMSE validation.
5. `05_extensions_and_ablation.ipynb`: Seasonal window sensitivity and seasonality ablation.
6. `06_robustness_and_scalability.ipynb`: Outlier/missing robustness and computational scaling.

---

## 4. Verification Checklist

Before concluding reproduction:
- [x] Input series length is exactly 3,601 observations (2014-01-01 to 2014-05-31).
- [x] Custom MSTL reconstruction max error is near zero ($< 10^{-11}$).
- [x] Synthetic trend RMSE is $< 2.0$.
- [x] All 15 figures exist under `results/figures/`.
- [x] All 6 tables exist under `results/tables/`.
- [x] TBATS benchmark completes and reports valid AIC and execution time.

