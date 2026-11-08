# MSTL Research Project

**MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns**

A B.Tech research project implementing and extending the MSTL algorithm by Bandara, Hyndman & Bergmeir (2021).

## Research Team & Affiliation

**Institution**: Indian Institute of Technology Guwahati (IIT Guwahati)  
**School**: Mehta Family School of Data Science and Artificial Intelligence  
**Program**: B.Tech in Data Science and Artificial Intelligence  

| Name | Roll No | Department / School |
| :--- | :---: | :--- |
| **Rup Narayan Jha** | 240150032 | Mehta Family School of Data Science & AI, IIT Guwahati |
| **Jaypal Malviya** | 240150013 | Mehta Family School of Data Science & AI, IIT Guwahati |
| **Anuj Kumar** | 240150003 | Mehta Family School of Data Science & AI, IIT Guwahati |
| **Ravi Kiran** | 240150029 | Mehta Family School of Data Science & AI, IIT Guwahati |

---

## Project Overview

This project reproduces the MSTL decomposition methodology from the paper:

> Bandara, K., Hyndman, R.J., & Bergmeir, C. (2021). *MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns.* arXiv:2107.13462. Published in *International Journal of Operational Research*, 52(1), 79–98.

We implement:
1. **Custom MSTL** — full Python implementation following Algorithm 1 of the paper
2. **Library MSTL** — using statsmodels for comparison
3. **Benchmarks** — STL and TBATS
4. **Four extensions** beyond the paper's scope

---

## Repository Structure

```
MSTL_Project/
├── data/                       # Datasets (auto-downloaded)
│   ├── vic_elec_demand.csv     # Raw half-hourly Victoria demand
│   └── vic_elec_hourly.csv     # Processed hourly series
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_custom_mstl_implementation.ipynb
│   ├── 03_library_and_benchmark_comparison.ipynb
│   ├── 04_synthetic_data_experiments.ipynb
│   ├── 05_extensions_and_ablation.ipynb
│   └── 06_robustness_and_scalability.ipynb
├── src/
│   ├── mstl_custom.py          # Custom MSTL implementation
│   ├── data_processing.py      # Data loading & preprocessing
│   ├── benchmarks.py           # STL, MSTL library, TBATS, Prophet
│   ├── experiments.py          # All four extensions + synthetic
│   ├── evaluation.py           # RMSE and other metrics
│   ├── visualization.py        # Plot generation
│   └── run_all_experiments.py  # Master runner script
├── results/
│   ├── figures/                # All generated plots (PNG)
│   ├── tables/                 # CSV result tables
│   └── experiment_logs/        # JSON experiment logs
├── report/
│   ├── final_report.tex        # ICML-format LaTeX report
│   ├── references.bib          # BibTeX references
│   └── final_report.pdf        # Compiled PDF
├── supplementary/
│   ├── Supplementary_Material.md  # Detailed supplementary doc
│   └── Supplementary_Material.pdf # Compiled PDF
├── requirements.txt
└── README.md
```

---

## Setup & Reproducibility

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

The Victoria electricity-demand dataset is downloaded automatically when you run the experiments. It originates from the `tsibbledata` R package and is publicly available on GitHub.

Manual download (if needed):

```bash
# Raw half-hourly CSV
wget https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv -O data/vic_elec_demand.csv
```

### 3. Run all experiments

```bash
cd MSTL_Project
python src/run_all_experiments.py
```

This will:
- Preprocess the Victoria electricity-demand dataset
- Run our custom MSTL implementation
- Run the statsmodels library MSTL for comparison
- Run STL and TBATS benchmarks
- Run all four extensions
- Save all figures, tables, and logs

### 4. Run individual notebooks

```bash
jupyter lab
```

Open any notebook from the `notebooks/` directory.

---

## Dataset

**Victoria Electricity Demand** — half-hourly electricity demand (MWh) for the state of Victoria, Australia.

- **Source**: Australian Energy Market Operator (AEMO) via `tsibbledata` R package
- **Raw data**: 230,784 half-hourly observations (2002–2015)
- **Experiment period**: 3,601 hourly observations (2014-01-01 00:00 to 2014-05-31 00:00)
- **Seasonal periods**: Daily (24h) and Weekly (168h)

---

## MSTL Algorithm

The decomposition model is:

```
X(t) = S₁(t) + S₂(t) + ... + Sₙ(t) + T(t) + R(t)
```

**Algorithm 1 (custom implementation)**:
1. Filter periods where 2m < n
2. Sort periods in ascending order
3. Interpolate missing observations
4. Apply Box-Cox transformation (optional)
5. Initialise all seasonal components to zero
6. For each outer iteration:
   - For each seasonal period mᵢ:
     - Remove all other seasonal components from the series
     - Apply STL to the adjusted series with period mᵢ
     - Update seasonal component Sᵢ(t) and trend T(t)
7. Compute remainder: R(t) = X(t) − T(t) − ΣSᵢ(t)

---

## Extensions

| Extension | Description |
|-----------|-------------|
| **Ext 1** | Seasonal window and iteration sensitivity analysis |
| **Ext 2** | Ablation study of single vs multiple seasonalities |
| **Ext 3** | Robustness to outliers and missing data |
| **Ext 4** | Computational scalability analysis |

---

## Key Results

- **Reconstruction Error**: Exact mathematical reconstruction (`max |X - X̂| = 0.0000`).
- **Computational Speed**: Custom MSTL executes in **0.230 s** on 3,601 observations (4.5× faster than `statsmodels.tsa.seasonal.MSTL` at 1.037 s, and 120× faster than state-space TBATS at 28.49 s on 500 obs).
- **Synthetic Ground-Truth Evaluation ($n=1680$)**:
  - Trend RMSE: **1.854** (1.24% relative error over signal range [500, 650])
  - Daily Seasonality RMSE: **6.260** (1.96% relative error over signal range [-320, +320])
  - Weekly Seasonality RMSE: **12.972** (7.86% relative error over signal range [-165, +165])
  - Remainder RMSE: **14.315** (accurately captures true noise $\sigma=30$)
- **Ablation Study ($n=3601$)**:
  - Daily only ($m=24$): Remainder Std = 599.38 MWh (absorbs weekly cycles into trend)
  - Weekly only ($m=168$): Remainder Std = 861.03 MWh (fails to model diurnal cycles)
  - Full Dual Seasonality ($m=[24, 168]$): Remainder Std = **809.77 MWh** (accurately separates both frequencies)
- **Robustness Asymmetry**:
  - Linear interpolation handles up to 10% missing observations with only **0.89%** trend degradation.
  - Artificial outliers ($\pm 5\sigma$) cause up to **33.56%** trend degradation at 10% contamination, indicating the necessity of outlier preprocessing in practice.
- **Computational Scalability**: Strict $O(n)$ linear runtime scaling ($R^2 > 0.998$) from 0.0029 s at $n=168$ to 0.2338 s at $n=3601$, with peak memory usage below 0.75 MB.

---

## References

1. Bandara, K., Hyndman, R.J., & Bergmeir, C. (2021). MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns. *arXiv:2107.13462*.
2. Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on LOESS. *Journal of Official Statistics*, 6(1), 3–73.

---

## License

This project is for educational purposes as part of a B.Tech research course.

