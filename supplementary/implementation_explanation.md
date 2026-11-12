# Implementation Notes: Our MSTL Code

**Project:** MSTL Implementation and Extensions  
**Affiliation:** Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati  
**Program:** B.Tech in Data Science and Artificial Intelligence  
**Authors:** Rup Narayan Jha (240150032), Jaypal Malviya (240150013), Anuj Kumar (240150003), Ravi Kiran (240150029)

---

## 1. Idea Behind the Algorithm

MSTL builds on the STL procedure of Cleveland et al. (1990). STL handles a single seasonal period. MSTL handles several by repeatedly applying STL inside a Gauss-Seidel style back-fitting loop: each seasonal component is re-estimated while the others are held fixed.

### 1.1 Model
$$X(t) = \sum_{k=1}^{K} S_k(t) + T(t) + R(t), \qquad t = 1, \dots, n$$

- $X(t)$: observed series
- $S_k(t)$: seasonal component for period $m_k$, with $m_1 < m_2 < \dots < m_K$
- $T(t)$: trend
- $R(t)$: remainder

---

## 2. The Procedure, Step by Step

**Step 1: validate periods.** A cycle cannot be estimated reliably from fewer than two repetitions, so any period with $2m_k \ge n$ is dropped.

**Step 2: sort periods.** Periods are put in ascending order, $m_1 < \dots < m_K$. Fast cycles (daily, $m_1 = 24$) are then extracted before slow ones (weekly, $m_2 = 168$).

**Step 3: fill missing values.** Gaps disturb the local regression inside LOESS, so missing points are filled by one-dimensional linear interpolation. Values before the first and after the last valid point take the nearest valid value. A boolean `nan_mask` records which positions were missing.

**Step 4: optional Box-Cox transform.** When the variance changes with the level of the series, we can stabilise it with
$$f_\lambda(x) = \begin{cases} (x^\lambda - 1)/\lambda, & \lambda \neq 0 \\ \ln x, & \lambda = 0. \end{cases}$$

**Step 5: initialise.** Every seasonal component starts as a zero vector of length $n$: $S_k^{(0)}(t) = 0$.

**Step 6: back-fitting.** For each outer iteration $i = 1, \dots, I$ (the default is $I = 2$) and each period index $k = 1, \dots, K$:

1. Remove the other seasonal components:
   $$\tilde{X}_k(t) = X(t) - \sum_{j \neq k} S_j(t).$$
2. Apply single-period STL with period $m_k$ and seasonal LOESS window $n_{s,k}$:
   $$(S_k, T) \leftarrow \mathrm{STL}\big(\tilde{X}_k,\ \text{period}=m_k,\ \text{seasonal}=n_{s,k}\big).$$
3. Update $S_k$ and the running trend $T$.

**Step 7: take the trend.** After the last outer iteration and the last period $m_K$, the trend from that final STL call is kept as the trend estimate.

**Step 8: compute the remainder.** The remainder is defined by subtraction, so the components add back to the input:
$$R(t) = X(t) - T(t) - \sum_{k=1}^{K} S_k(t).$$

**Step 9: undo the transform.** If Box-Cox was applied, each component is mapped back to the original scale with the inverse transform $f_\lambda^{-1}$.

---

## 3. Choosing the Seasonal Window

In STL the seasonal window $n_s$ controls how smooth each cycle-subseries is:

- a **small** $n_s$ lets the seasonal shape change quickly but can start fitting noise;
- a **large** $n_s$ forces a rigid, slowly changing seasonal shape.

Each period has its own window $n_{s,k}$, which can be set by the caller. Section 5.1 of `Supplementary_Material.md` and Tables E1.1 and E1.2 of `experiment_details.md` show the effect of odd widths from 7 to 25.

---

## 4. Checking Reconstruction

As a basic quality check we measure
$$\epsilon_{\text{recon}} = \max_{1 \le t \le n} \left| X(t) - \Big( T(t) + \sum_{k=1}^{K} S_k(t) + R(t) \Big) \right|.$$
On the 3,601-point Victoria series, $\epsilon_{\text{recon}} = 0.0000 \times 10^{0}$. Because $R(t)$ is defined by subtraction, this confirms that the code is consistent. It does not say how accurate the individual components are; the synthetic test in `experiment_details.md` (Table E5) is what measures that.
