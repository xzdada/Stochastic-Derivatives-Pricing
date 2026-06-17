# Stochastic Derivative Pricing

A from-scratch quantitative finance library for derivative pricing, risk management, and model calibration, built on stochastic calculus and financial mathematics.

## Table of Contents
- Project Structure
- Installation
- Data Sources
- Theory
    - BSM Model Assumptions
    - Pricing Formulas
    - Greeks
- Modules
- Build Phases

## Project Structure

``` bash
stochastic-derivative-pricing/
├── data/
│ ├── raw/ # option chains, spot prices, rates (CSV/JSON)
│ └── processed/ # cleaned vol surfaces, bootstrapped curves
├── src/
│ ├── models/ # stochastic process definitions
│ │ ├── equity/
│ │ │ ├── gbm.py # GBM base class, exact simulation
│ │ │ ├── heston.py # Heston SDE + char. fn
│ │ │ ├── sabr.py # SABR vol backbone
│ │ │ ├── jump_diffusion.py # Merton / Kou
│ │ │ ├── local_vol.py # Dupire local vol surface
│ │ │ └── garch.py # GARCH(1,1) realized vol
│ │ ├── rates/
│ │ │ ├── base_short_rate.py # abstract ShortRateModel
│ │ │ ├── vasicek.py
│ │ │ ├── cir.py
│ │ │ ├── ho_lee.py
│ │ │ ├── hull_white.py
│ │ │ ├── hjm.py
│ ├── engines/ # numerical pricing engines
│ │ ├── analytical.py # BSM, Black-76, Bachelier
│ │ ├── monte_carlo.py # base MC engine + path generator
│ │ ├── variance_reduction.py # antithetic, control var, importance samp.
│ │ ├── quasi_mc.py # Sobol / Halton sequences
│ │ ├── lsm.py # Longstaff-Schwartz for American
│ │ ├── trees.py # binomial CRR, trinomial, rate trees
│ │ ├── finite_diff.py # explicit, implicit, Crank-Nicolson, ADI
│ │ └── fft_pricer.py # Carr-Madan FFT (Heston, char. fn models)
│ ├── products/ # payoff definitions (data classes)
│ │ ├── base_option.py # abstract Option: S, K, T, r, sigma, type
│ │ ├── european.py
│ │ ├── american.py
│ │ ├── asian.py # avg price + avg strike variants
│ │ ├── barrier.py # knock-in/out, discrete/continuous
│ │ ├── lookback.py
│ │ ├── digital.py
│ │ ├── compound.py
│ │ └── rates_products.py # bond, cap, floor, swaption, swap
│ ├── calibration/
│ │ ├── implied_vol.py # Newton-Raphson + bisection IV solver
│ │ ├── vol_surface.py # build + interpolate vol surface (SVI)
│ │ ├── heston_calib.py # minimize MSE vs market prices
│ │ ├── rate_calib.py # bootstrap yield curve, fit short rate
│ │ └── garch_fit.py # MLE estimation of GARCH params
│ ├── risk/
│ │ ├── greeks.py # analytical + finite-diff Greeks
│ │ ├── delta_hedge.py # discrete hedging simulation + PnL
│ │ ├── portfolio.py # VaR, CVaR, Greeks aggregation
│ │ └── parity.py # put-call parity, boundary checks
│ └── utils/
│ │ ├── data_loader.py # yfinance / CBOE data fetch
│ │ ├── random.py # RNG seeding, Sobol init
│ │ └── plotting.py # vol surface 3D, Greeks heatmap, payoffs
├── notebooks/ 
│ ├── 01_bsm_european.ipynb
│ ├── 02_monte_carlo.ipynb
│ ├── 03_american_options.ipynb
│ ├── 04_exotic_options.ipynb
│ ├── 05_heston_model.ipynb
│ ├── 06_interest_rate_models.ipynb
│ ├── 07_greeks_and_hedging.ipynb
│ ├── 08_implied_vol_surface.ipynb
│ └── 09_model_comparison.ipynb
├── tests/ # unit + integration tests
│ ├── test_analytical.py # BSM vs known values
│ ├── test_mc_convergence.py # MC price -> analytical as N approach inf
│ ├── test_greeks.py # bump-and-reprice vs closed form
│ └── test_parity.py # put-call parity violations
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<xzdada>/stochastic-derivative-pricing.git
cd stochastic-derivative-pricing
pip install -r requirements.txt
```

*A free FRED API key is required for yield curve data. Register at: https://fred.stlouisfed.org/docs/api/api_key.html*

## Data Sources

| Data | Source | Method |
|------------------------------|---------------|---------|
| Equity spot prices & history | Yahoo Finance | yfinance|
| Option chains (calls & puts) | Yahoo Finance | yfinance|
| US Treasury yield curve | FRED API | fredapi|
| Dividend yield | Yahoo Finance | yfinance|

All data access is centralised in `src/utils/data_loader.py`:

```python
from src.utils.data_loader import MarketDataLoader

loader = MarketDataLoader(ticker="SPY", fred_api_key="FRED_API_KEY")

spot = loader.get_spot_price()
hist = loader.get_price_history(period="2y")
chain = loader.get_option_chain(expiry="2026-06-01")
r = loader.get_risk_free_rate(maturity=1.0) # continuously compounded
curve = loader.get_yield_curve() # full term structure
```

**Note**: FRED DGS series report nominal (bond-equivalent) yields. `get_risk_free_rate()` converts to continuously compounded via `r_cont = ln(1 + r_nominal)` before returning, matching the convention used throughout all pricing formulas.

## Theory

### BSM Model Assumptions

The Black-Scholes-Merton framework rests on seven assumptions.
1. **Geometric Brownian Motion**. The asset price follows GBM under the real-world probability measure P:

$$
dS = \mu ~ S ~ dt + \sigma ~ S ~ dW_t = \mu ~ S ~ dt + \sigma ~ S ~ \epsilon ~ \sqrt{dt}
$$

implying log-returns are normally distributed:

$$
ln (\frac{S_T}{S_0}) \sim N(\frac{\mu - \sigma^2}{2} T, ~ \sigma^2 T)
$$

*Limitation*: Real return distributions exhibit fat tails (leptokurtosis) and negative skew, which GBM cannot capture.
*Extensions in this project*: This project will try to use Jump-diffusion models, which add a Poisson jump component; and the Heston model, which introduces stochastic volatility, producing heavier tails endogenously.

2. **Constant volatility**. Volatility is assumed to be a fixed, known constant over the life of the option.

*Limitation*: Implied volatilities extracted from market option prices vary across strikes (volatility smile / skew) and maturities (term structure of volatility), directly contradicting this assumption.
*Extensions in this project*: Local volatility calibrates $\sigma(S, t)$ directly from the market surface. The Heston model lets $\sigma$ evolve stochastically. SABR provides an analytical approximation widely used for interest rate options.

3. **Constant risk-free rate**. The risk-free rate is assumed to be deterministic and constant over [0, T].

*Limitation*: For long-dated options or interest rate derivatives, the stochastic nature of rates contributes meaningfully to pricing.
*Extensions in this project*: Short rate models (Vasicek, CIR, Ho-Lee, Hull-White) model r as a stochastic process and price bonds, caps, floors, and swaptions consistently with the yield curve.

4. **No arbitrage and complete markets**. The market is assumed to be complete, which means any contingent claim can be replicated by a self-financing portfolio of the stock and risk-free bond. Under this condition, the unique risk-neutral measure Q exists and the no-arbitrage price of any derivative is:

$$
V(t) = e^{-r(T-t)} ~ \mathbb{E}^{ \mathbb{Q} (\text{Payoff}(S_T) | F(t))}
$$

Under Q, the drift of S is replaced by (r - q):

$$
dS = (r - q)~ S ~ dt + \sigma ~ S ~ dW_t^\mathbb{Q}
$$

5. **Continuous trading, no transaction costs**. The replicating portfolio can be rebalanced continuously at zero cost.

*Limitation*: In practice, hedging is discrete and incurs bid-ask spreads and transaction costs, producing a hedging error that grows with the rebalancing interval.
*Implementation in this project*: `src/risk/delta_hedge.py` simulates discrete delta hedging and decomposes the resulting PnL into delta PnL, gamma PnL, and vega PnL components, quantifying the hedging error as a function of rebalancing frequency.

6. **European-style option exercise only**. The option can only be exercised at expiry T, no early exercise is permitted.

*Limitation*: American options embed an early exercise right that has positive value, particularly for deep ITM puts and dividend-paying stocks.
*Extensions in this project*: Several alternative pricing methods provided in this project. Binomial and trinomial trees (`src/engines/trees.py`), Crank-Nicolson finite differences (`src/engines/finite_diff.py`), and the Longstaff-Schwartz regression-based Monte Carlo (`src/engines/lsm.py`) all handle early exercise.

7. **Continuous dividend yield q**. The original Black-Scholes (1973) model assumes no dividends. Merton (1973) extended the model to a continuous dividend yield q, replacing the spot price S with the dividend-adjusted forward price S e^{-qT}. This implementation follows Merton's extension throughout.

### Pricing Formula

Under the risk-neutral measure $\mathbb{Q}$ with continuous dividend yield q:

$$
C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
$$
$$
P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
$$
$$
d1 = \frac{[ ln(S / K) + (r - q + \frac{1}{2} \sigma^) T ]}{(\sigma \sqrt{T})}
$$
$$
d2 = d1 - \sigma \sqrt{T}
$$

**N(d2)** is risk-neutral probability that $S_T > K$, which means option expires in the money. **N(d1)** is the delta of the call option, the shares of stock in the replicating portfolio. $K e^{-rT} N(d2)$ is the present value of paying strike K conditional on exercise, whereas $S e^{-qT} N(d1)$ is the present value of receiving the stock conditional on exercise.

**Put-Call Parity**: This relationship holds regardless of the model. Violations in market data indicate arbitrage opportunities or data errors.

$$
C - P = S e^{-qT} - K e^{-rT}
$$

### Greeks

All Greeks are implemented analytically in `src/engines/analytical.py`.

| First Order | Second Order | Higher Order |
|------------|-------------|-------------|
|   Delta    |    Gamma    |    Vanna   |
|   Vega     |    Volga   |     Charm   |
|   Theta   |            |            |
|   Rho     |            |            |


| Greek | Formula | Interpretation |
|--------|--------------------------|------------------------------------------------|
| Delta | **$\frac{\partial V}{\partial S}$** | Change in option price for a \$1 move in spot |
| Gamma | **$\frac{\partial^2 V}{\partial S^2}$** | Rate of change of Delta; measures convexity |
| Vega | **$\frac{\partial V}{\partial \sigma}$** | Change in option value with respect to volatility |
| Theta | **$\frac{\partial V}{\partial t}$** | Time decay of option value |
| Rho | **$\frac{\partial V}{\partial r}$** | Sensitivity to risk-free rate |
| Vanna | **$\frac{\partial^2 V}{\partial S \partial \sigma}$** | Cross-sensitivity of Delta to volatility |
| Volga | **$\frac{\partial^2 V}{\partial \sigma^2}$** | Rate of change of Vega |
| Charm | **$\frac{\partial^2 V}{\partial S \partial t}$** | Rate of change of Delta over time |

### Monte Carlo Simulation

Monte Carlo prices a derivative by simulating a large number of asset price paths under the risk-neutral measure $ \mathbb{Q} $, computing the payoff on each path, and taking the discounted average:

$$
V = e^{-rT} \cdot (\frac{1}{N}) \sum{\text{Payoff}({S_T}^i)}
$$

By the law of large numbers, this converges to the true price as $N \rightarrow \infty$. The standard error shrinks at the rate **$\frac{1}{\sqrt{N}}$**.

Note that Geometric Brownian Motion (GBM) has a closed-form solution, so paths are simulated without any discretisation error regardless of the number of steps:

$$
S(t + \Delta t) = S(t) · \exp( (r - q - \frac{\sigma ^2}{2}) \Delta t  +  \sigma \sqrt{\Delta t} \cdot Z ), ~~  Z \sim N(0,1)
$$

#### Variance Reduction

The **$1 / \sqrt{N}$** convergence rate of standard MC is slow. Specifically, halving the error requires 4 times as many paths. This project will incorporate several variance reduction techniques to achieve lower error.

- Antithetic variates
For each random draw Z, we also simulate with −Z. The two paths are negatively correlated, so averaging their payoffs cancels much of the noise:

$$
\text{payoff}_i = \frac{ \text{Payoff}(Z_i) + \text{Payoff}(−Z_i)}{2}
$$

By using antithetic variates technique, the variance reduction is greatest when the payoff is a monotone function of the terminal price (vanilla calls and puts).

- Control variates
The technique uses a correlated quantity with a known expected value to correct the MC estimate. For a European option under GBM, the BSM price is the natural control:

$$
Y^* = Y − \beta ^* (X − \mu_X)
$$

$$
\beta ^* = Cov(Y, X) / Var(X), ~~~~~~~~~~ \text{(estimated from the same paths)}
$$

where Y is the simulated discounted payoff, X is the BSM payoff on the same paths, and $\mu_X$ is the analytical BSM price. The variance of $Y^*$ is:

$$
Var(Y^*) = Var(Y) \cdot (1 − \rho_{XY}^2)
$$

For GBM + European option, $\rho=1$ and the estimator collapses exactly to the BSM analytical price, confirming correctness. The technique shows its value for path-dependent products (Asian, barrier) where the target payoff and BSM control are highly but not perfectly correlated.

### American Options & Early Exercise

An American option gives the holder the right to exercise at any time $t \in [0, T]$. This creates an optimal stopping problem:

$$
V(t, S) = \sup_{\tau \in [t,T]} E^\mathbb{Q}\!\left[ e^{-r(\tau-t)} \mathrm{Payoff}(S_\tau) \mid S_t \right]
$$

The American option price always satisfies $V_{\text{am}} \ge V_{\text{eu}}$; the difference is the early exercise premium (EEP):

$$
\mathrm{EEP} = V_{\text{American}} - V_{\text{European}} \ge 0
$$

Early exercise optimality depends on the contract:

- **American call (no dividends):** EEP = 0. Exercising early forfeits the option's remaining time value; it is always better to sell than exercise.
- **American put:** EEP > 0. Receiving the strike early and earning interest on the cash dominates waiting.
- **American call (with dividends, $q > 0$):** EEP may be > 0. Not holding the stock means forfeiting dividend income, which can make early exercise worthwhile.

No closed-form solution exists for American options. Three numerical methods are implemented and cross-validated in this project:

**Method 1 — Binomial CRR Tree** ([src/engines/trees.py](src/engines/trees.py))

Cox, Ross and Rubinstein discretise GBM onto a recombining lattice. At each step $\Delta t = T/N$:

$$
u = \exp(\sigma \sqrt{\Delta t}), \qquad d = 1/u
$$

$$
p = \frac{e^{(r-q)\Delta t} - d}{u - d} \quad \text{(risk-neutral up probability)}
$$

Backward induction with early exercise check at every node:

$$
V_{n,j} = \max\!\left( \mathrm{Intrinsic}_{n,j},\ e^{-r\Delta t}\left[p\cdot V_{n+1,j+1} + (1-p)\cdot V_{n+1,j}\right] \right)
$$

The tree converges to BSM as $N \to \infty$ with oscillating $O(1/N)$ error. Greeks are computed via bump-and-reprice (central differences).

**Method 2 — Longstaff-Schwartz LSM** ([src/engines/lsm.py](src/engines/lsm.py))

Longstaff and Schwartz estimate the continuation value at each exercise date by regressing discounted future cash flows onto basis functions of the current stock price (polynomial basis by default):

$$
E^Q\!\left[e^{-r\Delta t} V_{k+1} \mid S_k\right] \approx \sum_i \alpha_i \cdot S_k^i
$$

Only in-the-money paths are used for regression because out-of-the-money paths trivially continue. The optimal exercise policy is found by comparing the regression estimate against the immediate exercise value.

LSM produces a low-biased price estimate — the regression policy is sub-optimal relative to the true exercise boundary.

**Method 3 — Crank-Nicolson Finite Differences** ([src/engines/finite_diff.py](src/engines/finite_diff.py))

Discretises the BSM PDE on a uniform $(S, t)$ grid and solves backward in time using the Crank-Nicolson scheme, which averages the explicit and implicit steps to yield a tridiagonal system at each time step:

$$
\frac{1}{2}\mathcal{L}^{\text{explicit}} + \frac{1}{2}\mathcal{L}^{\text{implicit}} \;\longrightarrow\; A\mathbf{V}^{j} = \mathbf{b}^{j}
$$

American early exercise is enforced via projected SOR (PSOR): after each linear solve, values are projected onto the free-boundary constraint:

$$
V_i^j = \max\!\left(V_i^j,\ \mathrm{Intrinsic}_i^j\right)
$$

Crank-Nicolson is unconditionally stable and second-order accurate in both $S$ and $t$. It is the most accurate of the three methods for smooth payoffs.

### Heston Stochastic Volatility Model

BSM assumes constant volatility, but market implied vols vary across strikes and maturities (the volatility smile/skew). The Heston model let volatility itself follow a stochastic process, producing a richer implied volatility surface that better matches market prices.

Under the risk-neutral measure Q, asset price S and instantaneous variance v follow coupled SDEs:

$$
dS_t = (r - q) S_t dt + \sqrt(v_t) S_t dW_t^S
$$
$$
dv_t = \text{kappa} * (\theta - v_t) dt + \xi * \sqrt(v_t) dW_t^v
$$
$$
Corr(dW_t^S, dW_t^v) = \rho
$$

#### Parameters

|                      |    Symbol     |   Interpretation   |
|----------------------|---------------|-----------------------|
| Mean-reversion Speed | $ \Kappa $  | How fast $v_t$ reverts to theta|
| Long-run Variance | $ \theta $ | Long-run mean of $ v_t \cdot sqrt(theta) $ |
| Vol-of-vol | $ \xi $ | Volatility of the variance process. Higher means more pronounced smile. |
| Correlation | $ \rho $ | Correkation between spot price and volatility. Negative for equities (leverage effect). |
| Initial variance | $ v_0 $ | Starting value of $ v_t \cdot sqrt(v0) $, current implied vol. |

#### Simulation: Euler Discretisation with Full Truncation

**Feller condition.** The variance process remains strictly positive almost surely if and only if:

$$2\kappa\theta > \xi^2$$

When this fails, $v_t$ can reach zero. We apply **full truncation** — replacing $v_t$ with $v^+ = \max(v_t, 0)$ before each update — to keep the simulation numerically stable regardless.

**Correlated shocks** via Cholesky decomposition:

$$\varepsilon_S = Z_S, \qquad \varepsilon_v = \rho\,Z_S + \sqrt{1-\rho^2}\,Z_v, \qquad Z_S, Z_v \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$$

**Variance update** (Euler, with full truncation):

$$v_{t+\Delta t} = v_t + \kappa(\theta - v^+)\,\Delta t + \xi\sqrt{v^+\,\Delta t}\;\varepsilon_v$$

**Asset price update** (exact log-normal conditioned on $v_t$, no discretisation bias):

$$S_{t+\Delta t} = S_t \exp\!\left[\left(r - q - \frac{v^+}{2}\right)\Delta t + \sqrt{v^+\,\Delta t}\;\varepsilon_S\right]$$

#### Volatility Smile Intuition

$\rho$ and $\xi$ are the primary drivers of the implied vol surface shape.

- **$\rho < 0$ (typical for equities):** when variance spikes, the stock tends to fall due to the leverage effect. This creates a negative skew — OTM puts carry higher implied vol than OTM calls.
- **Large $\xi$ (vol-of-vol):** high variance-of-variance fattens the tails symmetrically, raising implied vol for deep OTM options on both sides and producing more pronounced smile curvature.

### Heston Calibration

Given market implied vols across strikes and maturities, we find the 5 Heston parameters that best reproduce the observed smile:

$$
\min_{\kappa,\,\theta,\,\xi,\,\rho,\,v_0} \sum_i w_i \left(\sigma_i^{\text{model}} - \sigma_i^{\text{mkt}}\right)^2
$$

Minimising implied volatility error rather than price error is market convention — it treats each point on the smile equally regardless of moneyness.

This project uses **two-stage optimisation**:

- **Stage 1 — Differential Evolution (global):** stochastic population-based search over the 5-dimensional parameter box. Avoids local minima on the non-convex loss surface. Slower but robust.
- **Stage 2 — L-BFGS-B (local):** gradient-based refinement starting from the DE result. Fast convergence to high precision once within the basin of the global minimum.

Note that the Heston loss surface has multiple near-equivalent minima, particularly in the $(\kappa, \theta)$ subspace — a high $\kappa$ with low $\theta$ can produce a similar smile to a low $\kappa$ with high $\theta$. This is a fundamental identifiability issue, not a numerical one.

