# Implementation Explanation: MSTL Custom Algorithm

**Project**: MSTL Research Implementation and Extensions  
**Affiliation**: Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati  
**Program**: B.Tech in Data Science and Artificial Intelligence  
**Authors**:
- Rup Narayan Jha (240150032)
- Jaypal Malviya (240150013)
- Anuj Kumar (240150003)
- Ravi Kiran (240150029)

---

## 1. Algorithmic Foundation

The Multiple Seasonal-Trend decomposition using LOESS (MSTL) algorithm expands upon Cleveland et al.'s (1990) STL procedure. While STL is strictly designed for single-period seasonality, MSTL adopts an iterative Gauss-Seidel-type back-fitting scheme to estimate $K$ simultaneous seasonal components alongside a smooth non-linear trend.

### 1.1 The Additive Model
$$X(t) = \\sum_{k=1}^K S_k(t) + T(t) + R(t), \\quad t = 1, \\dots, n$$

Where:
- $X(t)$: Observed time series at time $t$
- $S_k(t)$: Seasonal component for period $m_k$, where $m_1 < m_2 < \\dots < m_K$
- $T(t)$: Trend-cycle component
- $R(t)$: Irregular remainder component

---

## 2. Step-by-Step Execution of Algorithm 1

### Step 1: Seasonal Period Validation
- An input time series of length $n$ cannot reliably estimate a seasonal cycle of period $m$ unless at least two complete cycles are present.
- MSTL enforces the strict criterion $2m_k < n$. Any period failing this check is pruned from the decomposition queue.

### Step 2: Sorting Seasonal Periods
- Seasonal periods are sorted in ascending numerical order:
  $$m_1 < m_2 < \\dots < m_K$$
- This ensures high-frequency seasonal patterns (e.g., daily $m_1=24$) are extracted and smoothed prior to lower-frequency, longer cycles (e.g., weekly $m_2=168$).

### Step 3: Imputation of Missing Values
- Missing observations (NaNs) interfere with LOESS local polynomial estimation.
- MSTL performs 1D linear interpolation across observed data points. Boundary values outside the first and last observed indices are filled with nearest valid endpoints.
- A boolean mask `nan_mask` is preserved to verify missing value locations.

### Step 4: Optional Box-Cox Variance Stabilization
- Non-constant seasonal variance is handled via Box-Cox transformation with parameter $\\lambda$:
  $$f_\\lambda(x) = \\begin{cases} \\frac{x^\\lambda - 1}{\\lambda}, & \\lambda \\neq 0 \\\\ \\ln(x), & \\lambda = 0 \\end{cases}$$

### Step 5: Seasonal Component Initialization
- All seasonal components are initialized to zero vectors of length $n$:
  $$S_k^{(0)}(t) = 0, \\quad \\forall k \\in \\{1, \\dots, K\\}, \\ t \\in \\{1, \\dots, n\\}$$

### Step 6: The Back-fitting Loop
- For each outer iteration $i = 1, \\dots, I$ (default $I = 2$):
  - For each seasonal period index $k = 1, \\dots, K$:
    1. **De-seasonalization**: Compute the adjusted series:
       $$\\tilde{X}_k(t) = X(t) - \\sum_{j \\neq k} S_j(t)$$
    2. **STL Extraction**: Apply standard single-period STL to $\\tilde{X}_k(t)$ with period $m_k$ and seasonal LOESS window $n_{s,k}$:
       $$(S_k, T) \\leftarrow \\text{STL}(\\tilde{X}_k, \\text{period}=m_k, \\text{seasonal}=n_{s,k})$$
    3. **Update**: Update $S_k(t)$ and update the running trend estimate $T(t)$.

### Step 7: Trend Assignment
- Following the final outer iteration and final seasonal period $m_K$, the trend component $T(t)$ produced by the final STL fit is retained as the authoritative trend estimate.

### Step 8: Remainder Extraction
- The remainder $R(t)$ is computed algebraically to satisfy exact additivity:
  $$R(t) = X(t) - T(t) - \\sum_{k=1}^K S_k(t)$$

### Step 9: Inversion of Variance Transformation
- If Box-Cox transformation was applied, the components are mapped back to original scale using $f_\\lambda^{-1}(\\cdot)$.

---

## 3. LOESS Window Parameter Selection

In standard STL, the seasonal window $n_s$ governs the smoothness of cycle-subseries.
- Smaller $n_s$ values allow seasonal patterns to adapt rapidly over time, but risk absorbing noise.
- Larger $n_s$ values enforce rigid, slowly evolving seasonal patterns.
- Bandara et al. recommend setting $n_{s,k} = 2m_k + 1$ (the next odd integer exceeding $2m_k$) for high-frequency time series. Our custom implementation implements this default, providing smoother, more stable seasonal components on hourly data.

---

## 4. Numerical Reconstruction Verification

A critical quality-assurance step is ensuring mathematical consistency:
$$\\epsilon_{\\text{recon}} = \\max_{1 \\le t \\le n} \\left| X(t) - \\left( T(t) + \\sum_{k=1}^K S_k(t) + R(t) \\right) \\right|$$
In our evaluation on the 3,601-observation Victoria electricity series, $\\epsilon_{\\text{recon}} = 0.0000 \\times 10^0$, confirming exact additivity.

