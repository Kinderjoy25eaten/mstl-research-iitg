# Supplementary Material: MSTL Implementation and Extensions
## Seasonal-Trend Decomposition for Time Series with Multiple Seasonal Patterns

**Affiliation:** Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati (IIT Guwahati)  
**Program:** B.Tech in Data Science and Artificial Intelligence  
**Submission date:** 15 November 2026  
**Scope:** Supporting documentation for the main report. Source code is not included, in line with the submission guidelines.

### Team

| Name | Roll number | Department and institute |
| :--- | :--- | :--- |
| Rup Narayan Jha | 240150032 | Mehta Family School of Data Science & AI, IIT Guwahati |
| Jaypal Malviya | 240150013 | Mehta Family School of Data Science & AI, IIT Guwahati |
| Anuj Kumar | 240150003 | Mehta Family School of Data Science & AI, IIT Guwahati |
| Ravi Kiran | 240150029 | Mehta Family School of Data Science & AI, IIT Guwahati |

---

## Contents
1. [Theory and the MSTL Algorithm](#1-theory-and-the-mstl-algorithm)
2. [Our Implementation](#2-our-implementation)
3. [Victoria Electricity Data and Preprocessing](#3-victoria-electricity-data-and-preprocessing)
4. [Synthetic Experiments with Known Ground Truth](#4-synthetic-experiments-with-known-ground-truth)
5. [The Four Extension Studies](#5-the-four-extension-studies)
6. [Comparison with Other Methods](#6-comparison-with-other-methods)
7. [Experimental Configuration](#7-experimental-configuration)
8. [Reproducibility and Code Access](#8-reproducibility-and-code-access)

---

## 1. Theory and the MSTL Algorithm

### 1.1 The problem
Sensors, energy markets, and transport systems often record data every few minutes or hours. Such series usually contain several cycles at different speeds. Hourly electricity demand, for example, shows:

- a **daily** cycle of $m_1 = 24$ hours, shaped by the sun and by routines such as the morning ramp-up and the evening residential peak;
- a **weekly** cycle of $m_2 = 168$ hours, shaped by the contrast between working days and weekends;
- a **yearly** cycle of $m_3 = 8760$ hours (8766 in leap years), shaped by the weather across seasons.

STL, introduced by Cleveland et al. (1990), can model only one seasonal period. On a series with several cycles it leaves the others in the remainder, which hurts forecasting and makes structural diagnosis harder.

### 1.2 The MSTL model
Bandara, Hyndman, and Bergmeir (2021) extend STL to any number $K \ge 1$ of seasonal components through additive back-fitting:

$$X(t) = \sum_{k=1}^{K} S_k(t) + T(t) + R(t)$$

where

- $X(t)$ is the observation at time $t \in \{1, \dots, n\}$;
- $S_k(t)$ is the seasonal component for period $m_k$, with $m_1 < m_2 < \dots < m_K$;
- $T(t)$ is the slowly varying trend;
- $R(t)$ is the remainder.

If the variance of the series changes over time, a Box-Cox transform with parameter $\lambda$ can be applied before decomposing:

$$f_\lambda(X(t)) = \begin{cases} \dfrac{X(t)^\lambda - 1}{\lambda}, & \lambda \neq 0 \\[2mm] \ln X(t), & \lambda = 0 \end{cases}$$

### 1.3 Algorithm 1
The procedure runs in these stages:

1. **Check periods.** A period is kept only if the series contains at least two full cycles of it, that is, $2m_k < n$.
2. **Sort periods.** Remaining periods are ordered from shortest to longest, so fast cycles are handled before slow ones.
3. **Fill gaps.** Missing values are filled by linear interpolation over the valid time points.
4. **Optional transform.** If $\lambda$ is given, the Box-Cox transform is applied.
5. **Initialise.** Every seasonal component starts at zero: $S_k^{(0)}(t) = 0$.
6. **Back-fit.** For each outer iteration $i = 1, \dots, I$ and each period $k = 1, \dots, K$:
   - subtract the current estimates of all other seasonal components,
     $$\tilde{X}_k(t) = X(t) - \sum_{j \neq k} S_j(t);$$
   - run single-period STL on $\tilde{X}_k$ with period $m_k$ and seasonal window $n_{s,k}$,
     $$(S_k, T) \leftarrow \mathrm{STL}\big(\tilde{X}_k,\ \text{period}=m_k,\ \text{seasonal}=n_{s,k}\big);$$
   - store the new $S_k$ and the running trend $T$.
7. **Trend and remainder.** The trend is taken from the very last STL call, and the remainder is found by subtraction:
   $$R(t) = X(t) - T(t) - \sum_{k=1}^{K} S_k(t).$$
8. **Undo the transform.** If Box-Cox was used, the inverse transform brings every component back to the original scale.

Two points are worth stating plainly. Processing short periods first keeps a slow cycle from contaminating the fast-cycle smoother. And because $R(t)$ is defined by subtraction, the components add back to the input exactly, up to floating-point rounding. That second property follows from the definition and does not by itself show the decomposition is statistically good.

---

## 2. Our Implementation

### 2.1 Structure
The code lives in `src/mstl_custom.py`, as a class `MSTLCustom` plus a function wrapper `mstl_decompose`. The main pieces are:

- **`fit(x)`** takes a one-dimensional array-like series, runs the whole pipeline above, and returns a dictionary of components.
- **Reconstruction check.** After decomposing, we rebuild $\hat{X}(t) = T(t) + \sum_k S_k(t) + R(t)$ and record $\max_t |X(t) - \hat{X}(t)|$. In all benchmark runs this was $0.0000 \times 10^{0}$.
- **Seasonal windows.** Each period has its own LOESS seasonal window $n_{s,k}$, which the caller can set. The sensitivity study in Section 5.1 covers odd widths from 7 to 25.

---

## 3. Victoria Electricity Data and Preprocessing

### 3.1 Source
We used the `vic_elec` dataset from the Australian Energy Market Operator, distributed in the R package `tsibbledata`. It has 230,784 half-hourly records covering 2002 to 2015.

### 3.2 Steps
1. **Rebuild timestamps.** Dates are stored as Excel serial numbers (day zero is 1899-12-30), and a period index from 1 to 48 marks the half-hour within the day:
   $$\text{Timestamp} = \text{ExcelDate} + (\text{Period} - 1) \times 30\ \text{min}.$$
2. **Compute demand.** $\text{Demand} = \text{OperationalLessIndustrial} + \text{Industrial}$.
3. **Aggregate to hours.** Half-hourly values are combined into hourly totals with a forward-looking sum.
4. **Select the window.** Bandara et al. (2021) use 149 days of 2014, from 1 January to 31 May. Counting both end points (`2014-01-01 00:00:00` to `2014-05-31 00:00:00`) gives exactly **3,601 hourly observations**.

### 3.3 Summary statistics

| Statistic | Value |
| :--- | ---: |
| Observations ($n$) | 3,601 |
| Mean demand | 11,278.2 MWh |
| Standard deviation | 1,996.8 MWh |
| Minimum | 7,858.0 MWh |
| Maximum | 20,595.0 MWh |

---

## 4. Synthetic Experiments with Known Ground Truth

### 4.1 Why synthetic data
Real demand data has no known true components, so on it we can only inspect residuals and reconstruction. To measure estimation error directly, we built an hourly series of $n = 1{,}680$ hours (70 days) from known parts:

$$X(t) = T(t) + S_{24}(t) + S_{168}(t) + R(t)$$

- Trend: $T(t) = 500 + 0.05t + 200\sin\!\left(\dfrac{2\pi t}{8760}\right)$
- Daily: $S_{24}(t) = 300\sin\!\left(\dfrac{2\pi t}{24} + 1\right) + 100\cos\!\left(\dfrac{4\pi t}{24}\right)$
- Weekly: $S_{168}(t) = 150\sin\!\left(\dfrac{2\pi t}{168}\right) + 50\cos\!\left(\dfrac{4\pi t}{168} + 0.5\right)$
- Remainder: $R(t) \sim \mathcal{N}(0, 30^2)$
- Random seed: 42

### 4.2 Results
Custom MSTL with $I = 2$ outer iterations gave:

| Component | True range | RMSE | Relative error |
| :--- | :--- | ---: | ---: |
| Trend $T(t)$ | $[500, 650]$ | 1.8541 | 1.24% |
| Daily $S_{24}(t)$ | $[-320, 320]$ | 6.2602 | 1.96% |
| Weekly $S_{168}(t)$ | $[-165, 165]$ | 12.9723 | 7.86% |
| Remainder $R(t)$ | $\sigma = 30$ | 14.3147 | n/a |

The trend is recovered closely, and the daily cycle is extracted with under 2% relative error. The weekly cycle has the largest error. We think this is because 70 days hold only ten full weekly cycles, which leaves each smoother with little data.

---

## 5. The Four Extension Studies

### 5.1 Extension 1: seasonal window and iterations
We ran every combination of seasonal window $n_s \in \{7, 11, 15, 21, 25\}$ and outer iterations $I \in \{1, 2, 3\}$ and recorded run time (seconds):

| Window $n_s$ | 1 iteration | 2 iterations | 3 iterations |
| :---: | :---: | :---: | :---: |
| 7 | 0.111 | 0.252 | 0.341 |
| 11 | 0.103 | 0.205 | 0.317 |
| 15 | 0.101 | 0.200 | 0.308 |
| 21 | 0.105 | 0.217 | 0.314 |
| 25 | 0.110 | 0.208 | 0.310 |

**Finding.** Reconstruction error stayed below $10^{-13}$ in every setting, as expected from the subtraction step. Run time grew roughly in proportion to the number of outer iterations. The window width changes how smooth the seasonal estimates are but does not affect reconstruction.

### 5.2 Extension 2: which seasonal periods to include
To see what each period contributes, we compared three configurations on the 3,601-point electricity series:

| Configuration | Periods | Remainder std (MWh) | Remainder mean (MWh) | Time (s) |
| :--- | :--- | ---: | ---: | ---: |
| Daily only | $[24]$ | 599.38 | -0.00 | 0.041 |
| Weekly only | $[168]$ | 861.03 | -0.00 | 0.165 |
| Both (full MSTL) | $[24, 168]$ | 809.77 | 1.29 | 0.252 |

**Finding.** The weekly-only model leaves the strong daily cycle in the remainder, so its residual spread is the largest (861.03 MWh). The daily-only model has a smaller remainder (599.38 MWh) than the full model, but this does not make it the better decomposition. A spectral check showed that without a weekly term the weekly pattern is absorbed by the trend instead of being removed. The remainder standard deviation alone is therefore a poor yardstick when the number of seasonal terms differs, and the full model is preferable because it keeps trend and weekly structure separate.

### 5.3 Extension 3: outliers and missing values
We corrupted the electricity series in two ways: spikes of $\pm 5\sigma$ added to 2%, 5%, and 10% of points, and 2%, 5%, and 10% of points removed at random.

| Corruption rate | Spike case: trend RMSE (MWh) | Degradation | Missing case: trend RMSE (MWh) | Degradation |
| :---: | ---: | ---: | ---: | ---: |
| 2% | 117.15 | 13.50% | 3.92 | 0.45% |
| 5% | 200.11 | 23.05% | 4.04 | 0.47% |
| 10% | 291.34 | 33.56% | 7.72 | 0.89% |

Degradation is the trend RMSE as a percentage of the series standard deviation.

**Finding.** The two kinds of damage behave very differently. Linear interpolation repairs gaps well, so trend error stays under 0.9% even with 10% missing. Spikes, in contrast, pass straight into the trend and seasonal smoothers because the basic method has no robust reweighting, and the trend error reaches 33.56% at 10% contamination. In practice, outlier filtering should come before decomposition.

### 5.4 Extension 4: run time and memory
We measured wall-clock time and peak memory at five series lengths:

| Span | Observations $n$ | Time (s) | Peak memory (MB) |
| :--- | :---: | ---: | ---: |
| 1 week | 168 | 0.0029 | 0.046 |
| 1 month | 720 | 0.0451 | 0.174 |
| 2 months | 1,440 | 0.0901 | 0.318 |
| 3 months | 2,160 | 0.1240 | 0.461 |
| 5 months (full) | 3,601 | 0.2338 | 0.749 |

**Finding.** Run time grows linearly with series length ($R^2 > 0.998$), and peak memory stays below 1 MB.

---

## 6. Comparison with Other Methods

| Method | Type | Periods | Time on 3,601 points | Remainder std |
| :--- | :--- | :--- | ---: | ---: |
| Custom MSTL | LOESS, non-parametric | Several | 0.230 s | 809.77 MWh |
| statsmodels MSTL | LOESS, non-parametric | Several | 1.037 s | 809.77 MWh |
| Standard STL | LOESS, non-parametric | One ($m_1$ only) | 0.078 s | 599.38 MWh |
| TBATS | Exponential state space | Several | 28.492 s (500 points) | 1,280.40 MWh |

**Finding.** Our implementation matches the library decomposition and ran about 4.5 times faster; we put this down to how LOESS parameters are passed but did not profile it. TBATS needed 28.49 s on only 500 points, so the gap to MSTL is at least two orders of magnitude, though the two timings use different series lengths and the ratio is not like-for-like.

---

## 7. Experimental Configuration

- **Python:** 3.14.3 (x86_64)
- **Libraries:** `numpy` 2.4.3, `pandas` 3.0.1, `scipy` 1.17.1, `statsmodels` 0.15.0, `tbats` 1.2.0, `matplotlib` 3.10.8, `seaborn` 0.13.2
- **Hardware:** Intel/AMD x86_64, Windows, 16 GB RAM
- **Random seed:** 42 for the synthetic series and for every stochastic corruption experiment

---

## 8. Reproducibility and Code Access

### 8.1 Running everything
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full experiment suite
python src/run_all_experiments.py

# 3. Build the notebooks
python src/create_notebooks.py
```
A detailed walkthrough is in `reproducibility.md`.

### 8.2 Source code
The complete Python source is kept in the project codebase and will be shared through the institute's GitHub repository once the project is archived. It is left out of this document on purpose.
