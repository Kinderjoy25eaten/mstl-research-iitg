# Experiment Details: Configurations, Parameters, and Results

**Project:** MSTL Implementation and Extensions  
**Affiliation:** Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati  
**Program:** B.Tech in Data Science and Artificial Intelligence  
**Authors:** Rup Narayan Jha (240150032), Jaypal Malviya (240150013), Anuj Kumar (240150003), Ravi Kiran (240150029)

---

## 1. Environment

- **Operating system:** Windows 11, x86_64
- **Python:** 3.14.3
- **Libraries:**

| Package | Version |
| :--- | :--- |
| `numpy` | 2.4.3 |
| `pandas` | 3.0.1 |
| `scipy` | 1.17.1 |
| `statsmodels` | 0.15.0 |
| `tbats` | 1.2.0 |
| `matplotlib` | 3.10.8 |
| `seaborn` | 0.13.2 |
| `scikit-learn` | 1.8.0 |

---

## 2. Dataset

- **Name:** Victoria Electricity Demand (`vic_elec`)
- **Source:** Australian Energy Market Operator (AEMO), obtained through the `tsibbledata` repository
- **Raw size:** 230,784 half-hourly records, 2002 to 2015
- **Window used:** 1 January 2014 00:00:00 to 31 May 2014 00:00:00 (149 days, both end points included)
- **Sample size after hourly aggregation:** $n = 3{,}601$
- **Seasonal periods:** daily $m_1 = 24$ hours and weekly $m_2 = 168$ hours

---

## 3. Results Tables

### 3.1 Extension 1: seasonal window ($n_s$) and outer iterations ($I$)

**Table E1.1. Reconstruction RMSE**

| Window $n_s$ | $I = 1$ | $I = 2$ | $I = 3$ |
| :---: | :---: | :---: | :---: |
| 7 | $4.55 \times 10^{-14}$ | $4.29 \times 10^{-14}$ | $4.29 \times 10^{-14}$ |
| 11 | $7.42 \times 10^{-14}$ | $5.25 \times 10^{-14}$ | $7.42 \times 10^{-14}$ |
| 15 | $6.25 \times 10^{-14}$ | $8.02 \times 10^{-14}$ | $6.78 \times 10^{-14}$ |
| 21 | $7.58 \times 10^{-14}$ | $6.78 \times 10^{-14}$ | $5.67 \times 10^{-14}$ |
| 25 | $6.95 \times 10^{-14}$ | $3.39 \times 10^{-14}$ | $6.95 \times 10^{-14}$ |

These values are floating-point rounding, as expected, since the remainder is computed by subtraction.

**Table E1.2. Wall-clock time (seconds)**

| Window $n_s$ | $I = 1$ | $I = 2$ | $I = 3$ |
| :---: | :---: | :---: | :---: |
| 7 | 0.111 | 0.252 | 0.341 |
| 11 | 0.103 | 0.205 | 0.317 |
| 15 | 0.101 | 0.200 | 0.308 |
| 21 | 0.105 | 0.217 | 0.314 |
| 25 | 0.110 | 0.208 | 0.310 |

### 3.2 Extension 2: seasonal ablation

**Table E2. Electricity demand, $n = 3{,}601$**

| Configuration | Periods | Remainder std (MWh) | Remainder mean (MWh) | Time (s) | Reconstruction error |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Daily only | [24] | 599.38 | -0.00 | 0.041 | $0.00 \times 10^{0}$ |
| Weekly only | [168] | 861.03 | -0.00 | 0.165 | $0.00 \times 10^{0}$ |
| Both (full MSTL) | [24, 168] | 809.77 | 1.29 | 0.252 | $0.00 \times 10^{0}$ |

The daily-only remainder is smaller, but the weekly pattern is then absorbed by the trend. Compare remainders only between models with the same seasonal terms.

### 3.3 Extension 3: corrupted data

**Table E3.1. Spikes of $\pm 5\sigma$**

| Contamination | Trend RMSE (MWh) | Degradation (% of series std) | Remainder std (MWh) | Time (s) |
| :---: | ---: | ---: | ---: | ---: |
| 2% | 117.15 | 13.50% | 1,556.33 | 0.248 |
| 5% | 200.11 | 23.05% | 2,226.83 | 0.214 |
| 10% | 291.34 | 33.56% | 3,047.87 | 0.221 |

**Table E3.2. Randomly missing values**

| Missing | Trend RMSE (MWh) | Degradation (% of series std) | Remainder std (MWh) | Time (s) |
| :---: | ---: | ---: | ---: | ---: |
| 2% | 3.92 | 0.45% | 811.28 | 0.221 |
| 5% | 4.04 | 0.47% | 810.38 | 0.224 |
| 10% | 7.72 | 0.89% | 809.13 | 0.206 |

### 3.4 Extension 4: scalability

**Table E4. Time and peak memory by series length**

| Span | Observations $n$ | Time (s) | Peak memory (MB) |
| :--- | :---: | ---: | ---: |
| 168 h (1 week) | 168 | 0.0029 | 0.0465 |
| 720 h (1 month) | 720 | 0.0451 | 0.1743 |
| 1,440 h (2 months) | 1,440 | 0.0901 | 0.3178 |
| 2,160 h (3 months) | 2,160 | 0.1240 | 0.4613 |
| 3,601 h (5 months, full) | 3,601 | 0.2338 | 0.7485 |

### 3.5 Synthetic ground-truth test ($n = 1{,}680$)

**Table E5. Component error against the known signal**

| Component | Equation | True range | RMSE |
| :--- | :--- | :---: | ---: |
| Trend $T(t)$ | $500 + 0.05t + 200\sin(2\pi t/8760)$ | $[500, 650]$ | 1.8541 |
| Daily $S_{24}(t)$ | $300\sin(2\pi t/24 + 1) + 100\cos(4\pi t/24)$ | $[-320, 320]$ | 6.2602 |
| Weekly $S_{168}(t)$ | $150\sin(2\pi t/168) + 50\cos(4\pi t/168 + 0.5)$ | $[-165, 165]$ | 12.9723 |
| Remainder $R(t)$ | $\mathcal{N}(0, 30^2)$ | $\sigma = 30$ | 14.3147 |

