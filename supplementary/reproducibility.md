# Reproducibility Guide

**Project:** MSTL Implementation and Extensions  
**Affiliation:** Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati  
**Program:** B.Tech in Data Science and Artificial Intelligence  
**Authors:** Rup Narayan Jha (240150032), Jaypal Malviya (240150013), Anuj Kumar (240150003), Ravi Kiran (240150029)

---

## 1. Setup

The code is plain Python and runs on Windows, macOS, Linux, and Google Colab. The reported results were produced with Python 3.14.3 and the package versions listed in `experiment_details.md`. Other versions may work but could change timings slightly.

### 1.1 Get the code
From the project root:
```bash
cd MSTL_Project
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
```
The main packages are:

- `numpy`, `pandas`, `scipy`
- `statsmodels` (STL and the library MSTL)
- `tbats` (the state-space benchmark)
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## 2. Re-running All Experiments

One command runs the whole pipeline: loading the data, our MSTL, the library comparisons, the synthetic test, all four extensions, and the figures and tables.

```bash
python src/run_all_experiments.py
```

Two optional follow-up commands:
```bash
python src/create_notebooks.py   # generate the notebooks
python src/compile_pdfs.py       # compile the documentation PDFs
```

### What to expect
- The processed data is written to `data/vic_elec_hourly.csv` and has exactly **3,601 hourly rows**.
- 15 figures appear in `results/figures/`.
- 6 CSV tables appear in `results/tables/`.
- A full log is saved to `results/experiment_logs/experiment_log.json`.

---

## 3. Notebooks

To step through the stages interactively:
```bash
jupyter lab notebooks/
```
Run them in this order:

1. `01_data_preprocessing.ipynb`: inspect the raw data and aggregate to hourly values.
2. `02_custom_mstl_implementation.ipynb`: run our MSTL step by step and check residuals.
3. `03_library_and_benchmark_comparison.ipynb`: compare with `statsmodels` MSTL, STL, and TBATS.
4. `04_synthetic_data_experiments.ipynb`: build the synthetic series and compute error against ground truth.
5. `05_extensions_and_ablation.ipynb`: window sensitivity and seasonal ablation.
6. `06_robustness_and_scalability.ipynb`: outlier and missing-data tests, and scaling.

---

## 4. Checklist

Before treating a reproduction as complete, confirm that:

- [x] the input has exactly 3,601 observations (2014-01-01 to 2014-05-31);
- [x] the reconstruction error of our MSTL is essentially zero (below $10^{-11}$);
- [x] the synthetic trend RMSE is below 2.0;
- [x] all 15 figures exist in `results/figures/`;
- [x] all 6 tables exist in `results/tables/`;
- [x] the TBATS benchmark finishes and reports a valid AIC and run time.
