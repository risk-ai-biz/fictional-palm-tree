# Market Impact Model — Data & Fitting Plan (EU, Continuous + Auctions)

**Goal:** Build an industrial‑strength market impact model from parent/child executions, trades, and quotes across European **primaries + MTFs**. The model will feed a **Mosek‑based** multi‑period portfolio construction framework. We emphasize scalable **data prep**, **robust estimation**, and **optimization‑ready** forms.

---

## Executive summary

- **Impact structure:** impulses (signed, normalized volume) convolved with a **decay kernel**, plus an **instantaneous** term.  
- **Bucketization:** **VolumeTime** buckets with **end‑of‑bucket** prices; evaluate at multiple **lags** (minutes).  
- **Residualization:** returns vs market/sector to isolate execution‑driven impact.  
- **Parameters:** per **microstructure cluster** (continuous vs auctions treated separately): \(\{\gamma,\alpha,\beta\}\) for continuous; \(\{\eta,\kappa,\rho,\zeta\}\) for auctions.  
- **Optimization:** keep **instantaneous** cost **inside Mosek** via **power‑cone epigraphs**; compute **transient/decay** via **FFT** **outside** the cone model with 1–2 outer iterations. This yields **5–50×** speedups vs modeling full decay inside the solver with negligible loss in optimality.

---

## Question checklist (decide up front)

### Objectives & scope
- Primary use: **pre‑trade** cost curves, **post‑trade** attribution, **execution scheduling**?
- Lags of interest \(L\) (minutes): \(\{1,5,10,30,60\}\)? Overnight excluded or modeled separately?
- Separate **continuous** trading and **auctions** (open/close/re‑opens)? **Yes** (recommended).

### Units & normalization
- **Volume:** \(%\)ADV (choose horizon: e.g., 20‑day rolling; consolidated or venue‑specific as needed).  
- **Prices/impact:** **bps** vs **EBBO mid**.  
- **Lags:** **minutes**, measured from **end‑of‑bucket** timestamp.

### Benchmarks & residualization
- Benchmarks per stock (e.g., **STOXX 600** + sector index by ICB/GICS).  
- Residualize stock returns with rolling beta:  
  \[
  r^{\text{stock}}_t = \beta_m r^{\text{mkt}}_t + \beta_s r^{\text{sect}}_t + \varepsilon_t,\quad
  \text{use } \varepsilon_t \text{ (in bps) for fitting.}
  \]

### Microstructure choices
- Bucketization: **VolumeTime** (1–2% ADV) primary; **ClockTime** backup for TOD analysis.  
- Price source: **EBBO mid** across primaries+MTFs (L1).  
- Aggressiveness: classify child fills (marketable vs passive), venue, and flags.

### Data inclusion/exclusion
- EBBO venues (MICs) included? Treatment of off‑book, auctions, dark, RFQ?  
- Filters: halts, crossed quotes, extreme spreads, broken prints. Corporate action adjustments.

### Intermediate activity & confounding
- Handle other flow in lag windows: **net cumulative signed volume** as control (preferred), or exclusion thresholds.

### Clustering & pooling
- Monthly clustering by liquidity (ADV/turnover), spread (bps), tick‑to‑spread, volatility, venue mix, auction share.  
- Separate clusters for **continuous** vs **auctions**; symbol overrides where data‑rich.

### Estimation strategy
- Bounds: \(\alpha\in[0.3,1],\ \beta>0,\ \gamma>0\). Robust loss preferred.  
- Fit instantaneous (L=0) + transient (L>0) coherently; **FFT** for transient evaluation.  
- Validation: time‑split cross‑validation by month/quarter; OOS diagnostics by cluster/venue/TOD.

### Optimization integration (Mosek)
- Inside solver: instantaneous cost via **power‑cone** epigraph for \(|u|^\alpha\); quadratic risk.  
- Outside solver: transient decay via **FFT**; optional 1–2 outer iterations to reflect decay cost.

---

## Data preparation — key points

1. **Time & venue hygiene**  
   Normalize all timestamps to **UTC**; retain local venue times for session boundaries & DST. Enforce monotone time per ISIN. De‑dup & merge feeds across primaries/MTFs; consistent ISIN mapping.

2. **Consolidated quotes (EBBO)**  
   Build **EBBO** mid (best bid/ask across lit venues). Store spread (bps). Tag auction phases (call/uncross).

3. **Corporate actions**  
   Adjust historical prices for splits/dividends; maintain clean continuous mid series.

4. **Trade/quote sanity**  
   Drop off‑book/OTC/special condition prints from **continuous** model; keep **auction** uncross & indicative separately. Remove crossed/broken quotes; winsorize extreme returns.

5. **Parent–child mapping**  
   Reliable join keys; compute **signed volume**; attribute venue & aggressiveness at child level.

6. **Residualization**  
   Regress stock vs market + sector; store **residual (bps)** series for fitting.

7. **Bucketization & lags**  
   **VolumeTime** buckets reset daily. Record **end‑of‑bucket** timestamp. Lags start at bucket end. Use multiple lags per bucket/execution: \(L\in\{1,5,10,30,60\}\) minutes.

8. **Intermediate executions**  
   Compute **cumulative net signed flow** within \([t, t+L]\); include as control or net into impulse signal.

9. **Controls**  
   Spread (bps), realized vol, time‑of‑day, venue mix, book imbalance proxy, depth proxy (if available).

10. **Storage & compute**  
    Parquet partitioned by **date/ISIN/MIC**; indexes on (ISIN,time). Precompute rolling ADV/vol/spread. Use **Polars/PySpark** for billion‑row transforms.

---

## Model overview

### Continuous trading (propagator / convolution view)

- **Instantaneous (L=0):**  
  \[
  \Delta p_{t,0} = \gamma\,\operatorname{sign}(v_t)\,|v_t|^{\alpha} \;+\; \mathbf{z}_{t,0}^\top \delta \;+\; \varepsilon_{t,0}
  \]

- **Transient (L>0):**  
  \[
  \Delta p_{t,L} = \gamma\,\operatorname{sign}(v_t)\,|v_t|^{\alpha}\,(1+L)^{-\beta}
  \;+\; \mathbf{z}_{t,L}^\top \delta \;+\; \varepsilon_{t,L}
  \]

- **Convolution view:** with impulse \(f_t=\operatorname{sign}(v_t)|v_t|^\alpha\) and kernel \(K_L=(1+L)^{-\beta}\),  
  \[
  I(t) \;=\; (f * K)(t) \;=\; \sum_{\tau\le t} f(\tau)\,K(t-\tau).
  \]

**Units:** \(v_t\) in **%ADV**, returns \(\Delta p\) in **bps**, \(L\) in **minutes**; prices vs **EBBO mid**.

### Auctions

- **Slippage at uncross:**  
  \[
  S \;=\; \eta\,\operatorname{sign}(\pi)\,|\pi|^{\kappa} \;+\; \mathbf{w}^\top\theta \;+\; \varepsilon
  \qquad (\pi=\tfrac{\text{parent notional}}{\text{auction notional}})
  \]

- **Post‑reopen drift (reversion):**  
  \[
  D_L \;=\; \rho\,\operatorname{sign}(\pi)\,|\pi|^{\kappa}\,(1+L)^{-\zeta} \;+\; \mathbf{w}^\top\theta' \;+\; \varepsilon.
  \]

---

## Microstructure clustering

- Monthly feature vectors per instrument: ADV/turnover, spread (bps), tick‑to‑spread, realized vol, venue mix (% MTF), trade size distribution, sign autocorr, auction share/imbalance stats.  
- Cluster into ~**12–20** groups (k‑means/GMM). Separate sets for **continuous** vs **auctions**.  
- Maintain **time‑varying** cluster IDs; fallback to symbol‑level fits where data‑rich.

---

## Observation construction

### Continuous
Per parent order:
- VolumeTime buckets (1–2% ADV). For each bucket \(t\) and lags \(L\):  
  \(v_t\) (signed, %ADV), **end‑of‑bucket EBBO mid**, **residualized** return, controls.  
- Net cumulative signed volume between \(t\) and \(t+L\).

### Auctions
Per parent:
- Participation \(\pi\), indicative imbalance trajectory (if available), reference mid, uncross price, post‑reopen returns at \(L\in\{5,15,30\}\) minutes.

---

## Estimation approaches & trade‑offs

### 1) **Bounded nonlinear least squares** (`scipy.optimize.least_squares`)
- Objective (continuous): for observations \((t,L)\)
  \[
  \min_{\gamma,\alpha,\beta,\delta}\ \sum (\Delta p_{t,L} - \gamma\,\operatorname{sign}(v_t)|v_t|^\alpha (1+L)^{-\beta} - \mathbf{z}_{t,L}^\top\delta)^2
  \]
  with bounds \(\alpha\in[0.3,1],\,\beta>0,\,\gamma>0\).  
- Use robust loss (`loss="huber"`, tuned `f_scale`) for outliers.  
- **Pros:** simple, fast, supports bounds & robust loss. **Cons:** point estimates; sensitive if lags sparse.

### 2) **Coordinate descent + FFT acceleration**
- Grid/coarse search \(\alpha\). For each \(\alpha\), form \(f_t=\operatorname{sign}(v)|v|^\alpha\) and **convolve** with \(K\) via **FFT** to evaluate \(\beta,\gamma\).  
- Complexity per evaluation \(O(T\log T)\).  
- **Pros:** scales to billions of rows (per‑parent shards), stable; separates hard nonlinearity.  
- **Cons:** needs sensible \(\alpha\) grid; but trivially parallel.

### 3) **Robust regression & M‑estimators**
- Huber/Tukey losses or **quantile regression** for heavy tails. Combine with #1/#2.

### 4) **Hierarchical Bayes** (PyMC/Stan)
- Partial pooling within clusters: symbol‑level \(\gamma_i\) shrinks to cluster mean; \(\alpha,\beta\) at cluster‑level with priors.  
- **Pros:** uncertainty quantification, stability with sparse symbols. **Cons:** heavier compute (use monthly refresh).

### 5) **Convex surrogates for instantaneous term (Fusion+Mosek)**
- Fit epigraph variable \(c\ge |u|^\alpha\) via **power cone**; calibrate \(\gamma\) by convex regression / constrained LS on \((u,c)\).  
- **Pros:** convex, clean integration. **Cons:** transient decay not convex—handle via FFT outside.

### 6) **Stochastic / mini‑batch first‑order**
- For web‑scale, SGD/Adam on parameterized kernels with FFT per batch. **Pros:** linear scalability. **Cons:** more engineering for convergence.

### 7) **Other `scipy.optimize` solvers (comparison)**
- `least_squares` (TRF/LM): **best default** (bounds + robust loss).  
- `L‑BFGS‑B` on a custom objective: quick in low‑dim, good with bounds; ensure good scaling and gradients.  
- `trust‑constr`: supports constraints; heavier per‑iteration.  
- Global heuristics (`differential_evolution`, `dual_annealing`): robust seeding but slow; use to seed then switch to `least_squares`.
- **Bottom line:** `least_squares` + bounds + robust loss + FFT evaluation outperforms generic global methods here.

---

## Integration with Mosek (portfolio construction)

### Inside Mosek (convex cone program)
- Inventory dynamics: \(x_{t+1}=x_t-u_t\); terminal \(x_T=0\).  
- Quadratic risk: \(\lambda \sum_t x_t^\top \Sigma x_t\).  
- **Instantaneous impact:** add variables \(c_{t,i}\) with **power‑cone epigraph** \(c_{t,i}\ge |u_{t,i}|^{\alpha_i}\); objective adds \(\gamma_i c_{t,i}\).

### Outside solver (fast transient pricing)
- Given a trade path \(u_t\), compute transient impact with **FFT** on \(f_t=\operatorname{sign}(u_t)|u_t|^\alpha\) convolved with \(K\).  
- Optionally **iterate** 1–2 times: reprice decay, update linearized penalties, re‑solve. Usually converges quickly.

---

## Validation & monitoring

- **OOS** (time‑split by month/quarter). Metrics: **MAE**, **RMSE** (bps), **signed bias** (≈ 0), by cluster/venue/TOD and by lag.  
- **Stability:** rolling parameter bands; drift alerts on \(\alpha,\beta,\gamma\).  
- **Calibration:** predicted vs realized plots; decile calibration by %ADV and participation.  
- **Auctions:** error vs participation deciles; imbalance conditioning; **post‑reopen** reversion check.  
- **Ops metrics:** Mosek solve time vs \(n,T\), cone counts, memory; FFT runtime; end‑to‑end latency.

---

## Phased roadmap (prioritized for maximum benefit)

1. **Data foundation (weeks 1–3)**  
   EBBO + clean trades/quotes (primaries & MTFs), corporate actions, parent↔child linkage, residualization.  
   **Deliverable:** daily panels ready for bucketization.

2. **Baseline instantaneous model (weeks 4–5)**  
   VolumeTime buckets; **L=0 fit** \((\gamma,\alpha)\) per cluster with robust loss. Integrate with **Mosek** (power cones + quadratic risk).  
   **Win:** deployable optimization with immediate impact costs.

3. **Add transient decay via FFT (weeks 6–8)**  
   Multi‑lag construction; estimate \(\beta\) (and refine \(\gamma,\alpha\)) using **FFT‑accelerated** evaluation. Hybrid outer‑loop with Mosek.  
   **Win:** higher‑accuracy costs with minimal runtime impact.

4. **Clustering / partial pooling (weeks 9–10)**  
   Microstructure clusters (continuous & auctions). Robust aggregation or HB for stability; symbol overrides where rich.

5. **Auctions (weeks 11–12)**  
   Fit slippage vs participation/imbalance; model post‑reopen drift. Produce auction cost curves by cluster.

6. **Validation & monitoring (ongoing)**  
   OOS backtests, dashboards, parameter drift checks, monthly auto‑recalibration.

---

## Appendix A — Convex instantaneous + FFT decay (iterative hybrid)

### A1. Data setup (per parent, per day)
- Build **VolumeTime** buckets \(t=1,\dots,T\) (1–2% ADV).  
- For each bucket: signed \(v_t\) in %ADV; **end‑of‑bucket EBBO mid**; residual return in bps.  
- For lags \(L\in\{1,5,10,30,60\}\) minutes: create observations \(\Delta p_{t,L}\).  
- Compute **cumulative net signed flow** in \([t, t+L]\) for confounding control \(\mathbf{z}_{t,L}\).

### A2. Instantaneous epigraph inside Mosek
- Variables: holdings \(x_{t,i}\), trades \(u_{t,i}\), epigraphs \(c_{t,i}\).  
- Constraints: inventory dynamics; **power‑cone** epigraph \(c_{t,i}\ge |u_{t,i}|^{\alpha_i}\).  
- Objective part: \(\sum_{t,i} \gamma_i\,c_{t,i}\) \(+\) quadratic risk \(\lambda \sum_t x_t^\top \Sigma x_t\).

### A3. FFT decay outside
- Fix \(\alpha\). Form impulse \(f_t=\operatorname{sign}(u_t)|u_t|^\alpha\).  
- Kernel \(K=(1+L)^{-\beta}\) (power‑law) or \(K=\exp(-\lambda L)\) (exponential).  
- **Linear convolution** via FFT with zero‑padding:  
  \[
  I = \mathcal{F}^{-1}\{\mathcal{F}(f)\cdot \mathcal{F}(K)\}\ (\text{take first }T).
  \]
- Transient cost proxy: \(\sum_t \tilde{\gamma}\,|I_t|\) or dot with signs as appropriate; use for pricing or linearized penalty.

### A4. Iterative algorithm (fixed‑point / successive approximation)
1. **Solve** cone program (instantaneous only) → trade path \(u^{(0)}\).  
2. **FFT‑price** transient on \(u^{(k)}\) → get decay cost gradient/penalty \(\phi^{(k)}_t\).  
3. **Update** the cone objective with a **linearized** penalty \(\sum_t \phi^{(k)}_t\,|u_t|\) (or adjust \(\gamma\) locally).  
4. **Resolve** to get \(u^{(k+1)}\).  
5. **Stop** when objective change \(<\varepsilon\) or 2–3 iterations reached (empirically sufficient).

**Notes:** keep penalties non‑negative to preserve convexity; optionally damp updates with \(\eta\in(0,1)\).

### A5. Metrics & diagnostics
- **Fit metrics:** MAE/RMSE (bps) by cluster & lag; signed bias; AIC/BIC comparing kernels (power‑law vs exponential).  
- **Optimization metrics:** Mosek solve time, cone counts, KKT time, memory; FFT runtime.  
- **Convergence:** outer‑loop objective reduction, number of iterations to tolerance; stability of \(u_t\).  
- **Business KPIs:** realized slippage reduction vs baseline; impact forecasting error by %ADV decile.

### A6. Practical defaults
- Bucket size: **1–2% ADV**. Lags: \(\{1,5,10,30,60\}\) minutes.  
- Bounds: \(\alpha\in[0.3,1]\), \(\beta\in[0.2,2.0]\), \(\gamma>0\) (scaled by median spread bps).  
- Residualization: daily rolling OLS to market + sector (store residuals in bps).  
- Filters: spread \(<100\) bps, remove halts/crossed markets, winsorize top/bottom 0.1% returns.  
- Parameter pooling: estimate \(\alpha,\beta\) by **cluster**; \(\gamma\) by **symbol** (pool if sparse).

---

## Appendix B — Mosek Fusion (power‑cone) snippet

```python
import mosek.fusion as mf

with mf.Model('liquidate') as M:
    x = M.variable([T+1, n])
    u = M.variable([T, n])
    c = M.variable([T, n])  # epigraph for |u|^alpha

    # inventory dynamics
    for t in range(T):
        M.constraint(x.slice(t,t+1,None) - x.slice(t+1,t+2,None), mf.Domain.equals(u.slice(t,t+1,None)))

    # initial/terminal inventory
    M.constraint(x.slice(0,1,None), mf.Domain.equals(x0))
    M.constraint(x.slice(T,T+1,None), mf.Domain.equals(0.0))

    one = M.parameter('one', 1.0)
    p = alpha/(alpha+1)  # vector per symbol; broadcast as needed

    for t in range(T):
        for i in range(n):
            M.constraint(mf.Expr.vstack(c.index(t,i), one, u.index(t,i)), mf.Domain.powerCone(p[i]))

    # risk + impact
    impact = mf.Expr.dot(gamma.reshape(1,n).repeat(T,axis=0).flatten(), c.flatten())
    # add quadratic risk: sum_t x_t' Sigma x_t  (use quadratic cones or matrix PSD term)
    M.objective(mf.ObjectiveSense.Minimize, impact)  # + risk term
    M.solve()
```

---

## Appendix C — SciPy fitting: least_squares vs alternatives

- **`scipy.optimize.least_squares`** (TRF/LM): supports **bounds** and **robust loss**; best default for cluster‑level nonlinear LS.  
- **`L‑BFGS‑B`**: fast for smooth low‑dim objectives with bounds; use good scaling; requires custom loss.  
- **`trust‑constr`**: handles nonlinear constraints; heavier per‑iteration.  
- **Global heuristics** (differential evolution, dual annealing): robust seeding but slow; use only to seed `least_squares`.  
- **Recommendation:** use `least_squares` + bounds + **Huber** loss, with **FFT‑accelerated** kernel evaluation; fall back to L‑BFGS‑B for smooth variants; reserve global heuristics for seeding tough clusters.
