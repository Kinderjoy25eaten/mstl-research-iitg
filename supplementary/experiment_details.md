# Experiment Details: Complete Configurations, Parameters, and Results

**Project**: MSTL Research Implementation and Extensions  
**Affiliation**: Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati  
**Program**: B.Tech in Data Science and Artificial Intelligence  
**Authors**:
- Rup Narayan Jha (240150032)
- Jaypal Malviya (240150013)
- Anuj Kumar (240150003)
- Ravi Kiran (240150029)

---

## 1. Experimental Environment & System Specifications

- **Operating System**: Windows 11 / x86_64
- **Python Runtime**: Python 3.14.3
- **Primary Numerical Libraries**:
  - `numpy`: 2.4.3
  - `pandas`: 3.0.1
  - `scipy`: 1.17.1
  - `statsmodels`: 0.15.0
  - `tbats`: 1.2.0
  - `matplotlib`: 3.10.8
  - `seaborn`: 0.13.2
  - `scikit-learn`: 1.8.0

---

## 2. Dataset Specifications

- **Dataset**: Victoria Electricity Demand (`vic_elec`)
- **Source**: Australian Energy Market Operator (AEMO) via `tsibbledata` repository
- **Raw Observations**: 230,784 half-hourly records (2002 to 2015)
- **Experimental Span**: January 1, 2014 00:00:00 to May 31, 2014 00:00:00 (149 days, inclusive endpoints)
- **Aggregated Sample Size ($n$)**: **3,601 hourly observations**
- **Seasonal Periods Tested**:
  - Diurnal (Daily): $m_1 = 24$ hours
  - Circaseptan (Weekly): $m_2 = 168$ hours

---

## 3. Comprehensive Experimental Tables

### 3.1 Extension 1: Seasonal Window ($n_s$) and Iteration ($I$) Sensitivity

#### Table E1.1: Reconstruction RMSE Matrix
| Seasonal Window ($n_s$) | 1 Outer Iteration | 2 Outer Iterations | 3 Outer Iterations |
| :---: | :---: | :---: | :---: |
| **7** | $4.55 \\times 10^{-14}$ | $4.29 \\times 10^{-14}$ | $4.29 \\times 10^{-14}$ |
| **11** | $7.42 \\times 10^{-14}$ | $5.25 \\times 10^{-14}$ | $7.42 \\times 10^{-14}$ |
| **15** | $6.25 \\times 10^{-14}$ | $8.02 \\times 10^{-14}$ | $6.78 \\times 10^{-14}$ |
| **21** | $7.58 \\times 10^{-14}$ | $6.78 \\times 10^{-14}$ | $5.67 \\times 10^{-14}$ |
| **25** | $6.95 \\times 10^{-14}$ | $3.39 \\times 10^{-14}$ | $6.95 \\times 10^{-14}$ |

#### Table E1.2: Wall-Clock Execution Time (seconds)
| Seasonal Window ($n_s$) | 1 Outer Iteration | 2 Outer Iterations | 3 Outer Iterations |
| :---: | :---: | :---: | :---: |
| **7** | 0.111 s | 0.252 s | 0.341 s |
| **11** | 0.103 s | 0.205 s | 0.317 s |
| **15** | 0.101 s | 0.200 s | 0.308 s |
| **21** | 0.105 s | 0.217 s | 0.314 s |
| **25** | 0.110 s | 0.208 s | 0.310 s |

---

### 3.2 Extension 2: Seasonality Ablation Results

#### Table E2: Ablation Study on Victoria Electricity Demand ($n = 3{,}601$)
| Model Configuration | Seasonal Periods | Remainder Std (MWh) | Remainder Mean (MWh) | Runtime (s) | Recon. Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Daily Only** | [24] | 599.38 MWh | -0.00 MWh | 0.041 s | $0.00 \\times 10^0$ |
| **Weekly Only** | [168] | 861.03 MWh | -0.00 MWh | 0.165 s | $0.00 \\times 10^0$ |
| **Combined (Full MSTL)** | [24, 168] | 809.77 MWh | 1.29 MWh | 0.252 s | $0.00 \\times 10^0$ |

---

### 3.3 Extension 3: Outlier and Missing Data Robustness

#### Table E3.1: Artificial Outlier Spikes ($\pm 5\\sigma$)
| Contamination Fraction | Trend RMSE (MWh) | Trend Degradation (% of Std) | Remainder Std (MWh) | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: |
| **2%** | 117.15 MWh | 13.50% | 1,556.33 MWh | 0.248 s |
| **5%** | 200.11 MWh | 23.05% | 2,226.83 MWh | 0.214 s |
| **10%** | 291.34 MWh | 33.56% | 3,047.87 MWh | 0.221 s |

#### Table E3.2: Random Missing Observations (NaNs)
| Missing Fraction | Trend RMSE (MWh) | Trend Degradation (% of Std) | Remainder Std (MWh) | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: |
| **2%** | 3.92 MWh | 0.45% | 811.28 MWh | 0.221 s |
| **5%** | 4.04 MWh | 0.47% | 810.38 MWh | 0.224 s |
| **10%** | 7.72 MWh | 0.89% | 809.13 MWh | 0.206 s |

---

### 3.4 Extension 4: Computational Scalability Profiling

#### Table E4: Execution Time & Memory by Series Length
| Span Label | Observations ($n$) | Runtime (seconds) | Peak Memory (MB) |
| :--- | :---: | :---: | :---: |
| **168h (1 week)** | 168 | 0.0029 s | 0.0465 MB |
| **720h (1 month)** | 720 | 0.0451 s | 0.1743 MB |
| **1440h (2 months)** | 1,440 | 0.0901 s | 0.3178 MB |
| **2160h (3 months)** | 2,160 | 0.1240 s | 0.4613 MB |
| **3601h (5 months, full)** | 3,601 | 0.2338 s | 0.7485 MB |

---

### 3.5 Synthetic Ground-Truth Evaluation ($n = 1{,}680$)

#### Table E5: Component-wise RMSE against Known Synthetic Signal
| Component | Analytical Signal Equation | True Range | MSTL RMSE |
| :--- | :--- | :---: | :---: |
| **Trend $T(t)$** | $500 + 0.05t + 200\\sin(2\\pi t / 8760)$ | $[500, 650]$ | **1.8541** |
| **Daily Seasonal $S_{24}(t)$** | $300\\sin(2\\pi t / 24 + 1) + 100\\cos(4\\pi t / 24)$ | $[-320, +320]$ | **6.2602** |
| **Weekly Seasonal $S_{168}(t)$** | $150\\sin(2\\pi t / 168) + 50\\cos(4\\pi t / 168 + 0.5)$ | $[-165, +165]$ | **12.9723** |
| **Remainder $R(t)$** | $\\mathcal{N}(0, 30^2)$ | $\\sigma = 30$ | **14.3147** |

