# MSTL Research Project

**MSTL for Multiple Seasonal Patterns: Implementation, Reproduction, and Extensions**

This B.Tech research project re-implements and extends the MSTL decomposition method of Bandara, Hyndman, and Bergmeir (2021).

## Team and Affiliation

**Institution:** Indian Institute of Technology Guwahati (IIT Guwahati)  
**School:** Mehta Family School of Data Science and Artificial Intelligence  
**Program:** B.Tech in Data Science and Artificial Intelligence

| Name | Roll no. | School |
| :--- | :---: | :--- |
| Rup Narayan Jha | 240150032 | Mehta Family School of Data Science & AI, IIT Guwahati |
| Jaypal Malviya | 240150013 | Mehta Family School of Data Science & AI, IIT Guwahati |
| Anuj Kumar | 240150003 | Mehta Family School of Data Science & AI, IIT Guwahati |
| Ravi Kiran | 240150029 | Mehta Family School of Data Science & AI, IIT Guwahati |

---

## About the Project

We study the method from:

> Bandara, K., Hyndman, R. J., and Bergmeir, C. (2021). *MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns.* arXiv:2107.13462. Published in *International Journal of Operational Research*, 52(1), 79-98.

The project contains:

1. **Our own MSTL** in Python, following Algorithm 1 of the paper.
2. **The library MSTL** from `statsmodels`, used as a reference.
3. **Benchmarks** against single-period STL and TBATS.
4. **Four extension studies** that go beyond the original paper.

---

## Repository Layout

```
MSTL_Project/
├── data/
│   ├── vic_elec_demand.csv        # raw half-hourly Victoria demand
│   └── vic_elec_hourly.csv        # processed hourly series
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_custom_mstl_implementation.ipynb
│   ├── 03_library_and_benchmark_comparison.ipynb
│   ├── 04_synthetic_data_experiments.ipynb
│   ├── 05_extensions_and_ablation.ipynb
│   └── 06_robustness_and_scalability.ipynb
├── src/
│   ├── mstl_custom.py             # our MSTL implementation
│   ├── data_processing.py         # loading and preprocessing
│   ├── benchmarks.py              # STL, library MSTL, TBATS, Prophet
│   ├── experiments.py             # synthetic test and the four extensions
│   ├── evaluation.py              # RMSE and other metrics
│   ├── visualization.py           # plotting
│   ├── create_notebooks.py        # builds the notebooks
│   ├── compile_pdfs.py            # builds the documentation PDFs
│   └── run_all_experiments.py     # runs everything
├── results/
│   ├── figures/                   # generated plots (PNG)
│   ├── tables/                    # result tables (CSV)
│   └── experiment_logs/           # JSON logs
├── report/
│   ├── final_report.tex           # ICML-style LaTeX report
│   └── final_report.pdf           # compiled report
├── supplementary/
│   ├── Supplementary_Material.md
│   ├── Supplementary_Material.pdf
│   ├── experiment_details.md
│   ├── implementation_explanation.md
│   └── reproducibility.md
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

The experiments download the Victoria electricity-demand data automatically. It comes from the `tsibbledata` R package, which is public on GitHub. To fetch it by hand:

```bash
wget https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv -O data/vic_elec_demand.csv
```

### 3. Run every experiment

```bash
cd MSTL_Project
python src/run_all_experiments.py
```

This script will:
- preprocess the electricity data;
- run our MSTL;
- run the `statsmodels` MSTL for comparison;
- run the STL and TBATS benchmarks;
- run the synthetic test and all four extensions;
- save the figures, tables, and logs.

### 4. Use the notebooks

```bash
jupyter lab
```

Open any notebook in `notebooks/`. They are meant to be run in numerical order.

More detail, including software versions, is in `supplementary/reproducibility.md` and `supplementary/experiment_details.md`.

---

## Data

**Victoria electricity demand:** half-hourly demand (MWh) for the Australian state of Victoria.

- **Source:** Australian Energy Market Operator (AEMO), through the `tsibbledata` R package
- **Raw data:** 230,784 half-hourly records, 2002-2015
- **Window used:** 3,601 hourly values, 1 January 2014 00:00 to 31 May 2014 00:00
- **Seasonal periods:** daily (24 h) and weekly (168 h)

---

## The Algorithm

MSTL writes a series as seasonal parts, a trend, and a remainder:

```
X(t) = S1(t) + S2(t) + ... + SK(t) + T(t) + R(t)
```

**Algorithm 1 as we implemented it:**
1. Keep only periods with 2m < n.
2. Sort the periods from shortest to longest.
3. Fill missing values by linear interpolation.
4. Optionally apply a Box-Cox transform.
5. Set every seasonal component to zero.
6. For each outer iteration, and for each period m_i:
   - subtract all the other seasonal components from the series,
   - apply STL with period m_i to what is left,
   - update the seasonal component S_i(t) and the trend T(t).
7. Compute the remainder: R(t) = X(t) - T(t) - sum of the S_i(t).

---

## Extension Studies

| Study | What it examines |
| :--- | :--- |
| Extension 1 | How the seasonal window and number of outer iterations affect results |
| Extension 2 | Single versus multiple seasonal periods (ablation) |
| Extension 3 | Robustness to outliers and to missing values |
| Extension 4 | Run time and memory as the series gets longer |

---

## Interactive Website

We also built a small website that lets you explore the decomposition and our results:

**https://mstl-research-iitg-web.vercel.app/**

On it you can:
- see a series split into trend, daily cycle, weekly cycle, and remainder;
- browse the Victoria electricity data;
- look at the benchmark and synthetic-data results;
- read about the extension studies.

The website is a visual companion to this repository and is not needed to reproduce anything.

---

## Main Results

- **Reconstruction:** The components add back to the input exactly (max error 0.0000). This is true by construction, since the remainder is computed by subtraction.
- **Speed:** Our MSTL takes 0.230 s on 3,601 points, about 4.5 times faster than `statsmodels` MSTL (1.037 s). TBATS took 28.49 s on only 500 points, so it is far slower, though that run used a shorter series and the ratio is not like-for-like.
- **Synthetic test ($n = 1{,}680$, known components):**
  - Trend RMSE 1.854 (1.24% of the range [500, 650])
  - Daily cycle RMSE 6.260 (1.96% of [-320, 320])
  - Weekly cycle RMSE 12.972 (7.86% of [-165, 165])
  - Remainder RMSE 14.315, measured against true noise with $\sigma = 30$
- **Ablation ($n = 3{,}601$):**
  - Daily only: remainder std 599.38 MWh, but the weekly pattern is absorbed into the trend
  - Weekly only: remainder std 861.03 MWh, because the daily cycle stays in the residual
  - Both periods: remainder std 809.77 MWh, with the two cycles kept separate
  - A smaller remainder is not automatically better when models carry different numbers of seasonal terms.
- **Robustness:** With 10% of values missing, trend error rises by only 0.89%. With 10% of values replaced by $\pm 5\sigma$ spikes, it rises by 33.56%, so outliers should be filtered before decomposition.
- **Scalability:** Run time grows linearly with length ($R^2 > 0.998$), from 0.0029 s at $n = 168$ to 0.2338 s at $n = 3{,}601$, with peak memory under 0.75 MB.

---

## References

1. Bandara, K., Hyndman, R. J., and Bergmeir, C. (2021). MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns. *arXiv:2107.13462*.
2. Cleveland, R. B., Cleveland, W. S., McRae, J. E., and Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on LOESS. *Journal of Official Statistics*, 6(1), 3-73.

---

## License

This project was created for educational purposes as part of a B.Tech research course.
---
