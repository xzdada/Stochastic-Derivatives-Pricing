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
    - Monte Carlo Simulation
    - American Options & Early Exercise
    - Exotic Options: Asian and Barrier
    - Heston Stochastic Volatility Model
    - Interest Rate Models
    - Greeks
    - Hedging Simulation
    - Portfolio Risk (VaR, ES)
    - Implied Volatility Extraction
    - Volatility Surface (SVI)
    - GARCH(1,1) and the Volatility Risk Premium
- Modules

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
| | | └── gaussian_plus.py
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
│ │ ├── portfolio.py # VaR, CVaR
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


### Exotic Options: Asian and Barrier

#### Asian options

Payoff depends on the average price along the path rather than the terminal price alone:

$$
\text{average price call}:  \max(A - K, 0)  
$$

$$
\text{average price put}:   \max(K - A, 0)  \qquad    \text{A = average of the path}
$$

$$
\text{average strike call}: \max(S_T - A, 0)
$$
$$
\text{average strike put}:  \max(A - S_T, 0)  \qquad   \text{A used as a floating strike}
$$

The arithmetic average of correlated log-normal draws is not itself log-normal, so there is no closed form for the standard (arithmetic) Asian option — it is priced via Monte Carlo. The geometric average, however, is log-normal, giving an exact Kemna & Vorst formula used here purely as a validation benchmark:

$$
\sigma_{avg} = \frac{\sigma}{\sqrt{3}}  \qquad  b_{avg} = 0.5*(\frac{r - q - \sigma^2}{6})
$$

A BSM-style formula then applies with these adjusted drift/vol terms.

Validation result: Kemna-Vorst closed-form vs Monte Carlo on the same simulated paths agrees to within 0.0411 (300k paths, 252 steps).

Asian vs. European price comparison (S=K=100, T=1y, r=5%, sigma=20%):
|    Product        |      Price           |
|------------------|------------------|
| European call (BSM) | 10.4506 |
| Asian average price call | 5.7215 |
| Asian average strike call | 5.8576 |

Both Asian variants are cheaper than the European option because averaging thr path reduces the effective volatility of the payoff-determining quantity (the average of many correlated draws has lower variance than a single terminal draw, so the embedded optionality is worth less.

#### Barrier options (knock-in / knock-out)

Payoff depends on whether the path touches a barrier level B at any point before expiry, in addition to the vanilla payoff at maturity. There are four variants based on the condition and direction: up-and-out, up-and-in, down-and-out, down-and-in.

In-out parity (model-free, holds exactly for any barrier level):

```
knock-in price + knock-out price = vanilla price
```

This parity holds because every path either touches the barrier or doesn't, the knock-in and knock-out are mutually exclusive and exhaustive, so summing their payoffs recovers the vanilla payoff exactly. This is one of the strongest available sanity checks for barrier code, because it doesn't depend on comparing against a second pricing metohod, it's an exact identity that any correct implementation must satisfy.

Validation result (S=100, K=100, T=1y, up-barrier=120, down-barrier=80):

|   Variant     |     Price        |
|---------------|------------------|
| Up-and-out (call) | 1.3362 |
| Up-and=in (call) | 9.0748 |
| Sum | 10.4109 (vanilla call = 10.4506, error = 0.0396) |
| Down-and-out (put) | 1.7521 |
| Down-and-in (put) | 3.8291 |
| Sum | 5.5812 (vanilla put = 5.5735, error = 0.0077)|

Both parity checks hold within Monte Carlo noise (300k paths), and the relationship holds across a full sweep of barrier levels, not just at one point, confirmed by tracking Knock-Out + Knock-In against the vanilla price as B varies from 105 to 160.

Note: barrier touch detection is evaluated at the resolution of the simulated path (n_steps), introducing a small discrete-monitoring bias relative to true continuous monitoring. The Broadie-Glasserman-Kou continuity correction would reduce this bias further but is not implemented here


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

### Interest Rate Models (Short Rate Models)

BSM assumes a constant risk-free rate. For long-dated options and any interest-rate-sensitive instrument — bonds, swaps, caps, swaptions — the stochastic evolution of rates matters. Short rate models specify the dynamics of the instantaneous rate $r(t)$ and derive the entire yield curve and derivative prices under a single no-arbitrage framework.

#### Vasicek Model

$$
dr_t = a(b - r_t)\,dt + \sigma\,dW_t
$$

$a > 0$ is the mean-reversion speed; $b$ is the long-run level the rate reverts to; $\sigma$ is the constant volatility. The drift term $a(b - r_t)$ is self-correcting: it is positive when $r_t < b$ and negative when $r_t > b$, pulling the rate back toward $b$ at a rate proportional to both the displacement and $a$.

Like BSM, $r_t$ is Gaussian under Vasicek, which means rates can become negative — a known limitation.

The closed-form zero-coupon bond price is:

$$
P(0,T) = A(T)\,\exp(-B(T)\,r_0)
$$
$$
B(T) = \frac{1 - e^{-aT}}{a}, \qquad A(T) = \exp\!\left[\left(b - \frac{\sigma^2}{2a^2}\right)(B(T) - T) - \frac{\sigma^2 B(T)^2}{4a}\right]
$$

#### CIR — Cox-Ingersoll-Ross Model

CIR replaces Vasicek's constant diffusion with a square-root term, ensuring $r_t$ remains non-negative:

$$
dr_t = a(b - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t
$$

**Feller condition:** $2ab \ge \sigma^2$ guarantees $r_t > 0$ almost surely. When violated, $r_t$ can reach zero; full truncation ($\max(r_t, 0)$ before the square root) is used for simulation — identical in spirit to the Heston variance process.

At $r_t = 0$ the diffusion vanishes while the drift $ab\,dt > 0$ remains positive, so zero is a reflecting boundary rather than an absorbing one.

The closed-form bond price retains the $A(T)\exp(-B(T)\,r_0)$ structure, with $A$ and $B$ solving the CIR Riccati ODEs:

$$
B(T) = \frac{2(e^{\gamma T}-1)}{(\gamma+a)(e^{\gamma T}-1)+2\gamma}, \qquad \gamma = \sqrt{a^2 + 2\sigma^2}
$$

#### Ho-Lee Model

$$
dr_t = \lambda(t)\,dt + \sigma\,dW_t
$$

$\lambda(t)$ is a deterministic, time-dependent drift calibrated to exactly fit the initial market yield curve. With no mean reversion, $r_t$ is a pure drifted random walk and can wander arbitrarily far from $r_0$ over long horizons — the main limitation of this model.

#### Hull-White — Extended Vasicek

Hull-White adds mean reversion to Ho-Lee, inheriting exact yield-curve fitting while keeping rates anchored:

$$
dr_t = [\lambda(t) - a\,r_t]\,dt + \sigma\,dW_t
$$

When $\lambda(t) = ab$ (constant), Hull-White reduces exactly to Vasicek$(r_0, a, b, \sigma)$.

#### Three-Factor Cascade Gaussian+ Model

Single-factor short rate models share a structural limitation: every point on the yield curve is driven by the same random shock, so all maturities move in lockstep and are perfectly correlated. Real yield curves do not behave this way. Empirically, a principal component analysis of historical yield curve changes typically reveals three dominant factors: short-term, medium-term, and long-term moves. This model captures that structure with a hierarchical (cascade) dependency: the short rate reverts toward a medium-term factor, which in turn reverts toward a long-term factor, which reverts toward a fixed long-run mean $r \leftarrow m \leftarrow L \leftarrow \mu$.

$$
dr = a_r(m - r) dt \qquad \text{(no diffusion term)}
$$
$$
dm = a_m(L - m) dt + \sigma_m \left(\rho dW_1 + \sqrt{1-\rho^2} dW_2\right)
$$
$$
dL = a_L(\mu - L) dt + \sigma_L dW_1
$$

$W_1$ and $W_2$ are independent Brownian motions.

| Symbol | Role |
|---|---|
| $r$ | Short-term rate |
| $m$ | Medium-term factor |
| $L$ | Long-term factor |
| $\mu$ | Fixed long-run mean that $L$ reverts to |
| $a_r, a_m, a_L$ | Mean-reversion speeds for $r$, $m$, $L$ |
| $\sigma_m, \sigma_L$ | Volatilities of the medium- and long-term factors |
| $\rho$ | Correlation loading of $dW_1$ in the $m$ dynamics |

Unlike a parallel multi-factor model where each factor has its own diffusion, $r$'s only source of randomness is inherited through its mean-reversion drift toward $m$. This keeps short-rate volatility low and matches the empirically observed humped-shaped term structure of volatility. $r$ is effectively a smoothed version of $m$, which is itself a smoothed version of $L$ — producing a genuine hierarchical term-structure model rather than three independently-shocked factors summed together.

Because each factor's drift depends only on the factor below it, the system is simulated sequentially per time step:

- **Update $L$** using its exact one-factor OU transition (Gaussian, no bias).
- **Update $m$** using its exact OU transition toward the just-updated $L$, with diffusion driven by the correlated pair $(dW_1, dW_2)$.
- **Update $r$** using exact deterministic relaxation toward the just-updated $m$ (no diffusion of its own).

Because $r(t)$ is not a simple linear combination of independent Gaussian factors with constant coefficients (the cascade drift makes $r$'s distribution depend on nested time-integrals of $m$ and $L$), bond prices are computed via Monte Carlo:

$$
P(0,T) = \mathbb{E}^{\mathbb{Q}}\ [\exp \left(-\int_0^T r_s ds\right)]
$$

### Greeks & PnL Attribution

Greeks measure the sensiticity of option price to changes in its inputs. They are the primary tools for risk management and dynamic hedging.

#### Analytical Greeks (BSM)

For European options under BSM, all Greeks have exact closed-form expressions as partial derivatives of the pricing formula. These are implemented in `src/engines/analytical.py` and exposed through the unified `GreeksCalculator` in `src/risk/greeks.py`.

| Greek | Formula (with continuous dividend yield $q$) | Interpretation |
|-------|----------------------------------------------|----------------|
| Delta | Call: $e^{-qT}N(d_1)$ &nbsp; Put: $e^{-qT}(N(d_1)-1)$ | $\partial V/\partial S$ — price change per \$1 spot move; call $\in(0,1)$, put $\in(-1,0)$ |
| Gamma | $\dfrac{e^{-qT}\,n(d_1)}{S\,\sigma\sqrt{T}}$ | $\partial^2 V/\partial S^2$ — convexity; rate of change of delta (identical for calls & puts) |
| Vega | $S\,e^{-qT}\,n(d_1)\sqrt{T}$ | $\partial V/\partial\sigma$ — price change per unit vol move (identical for calls & puts) |
| Theta | Call: $-\dfrac{S e^{-qT}n(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$ &nbsp;  | $\partial V/\partial t$ — time decay; typically negative as expiry approaches |
| Rho | Call: $KTe^{-rT}N(d_2)$ &nbsp; Put: $-KTe^{-rT}N(-d_2)$ | $\partial V/\partial r$ — sensitivity to the risk-free rate |
| Vanna | $-e^{-qT}\,n(d_1)\,\dfrac{d_2}{\sigma}$ | $\partial^2 V/\partial S\,\partial\sigma$ — cross-sensitivity of delta to vol (identical for calls & puts) |
| Volga | $S\,e^{-qT}\,n(d_1)\sqrt{T}\,\dfrac{d_1 d_2}{\sigma}$ | $\partial^2 V/\partial\sigma^2$ — convexity of price in vol space; vol-of-vega (identical for calls & puts) |
| Charm | Call: $-e^{-qT}\!\left[n(d_1)\dfrac{2(r-q)T - d_2\sigma\sqrt{T}}{2T\sigma\sqrt{T}} - q\,N(d_1)\right]$ &nbsp;  | $\partial^2 V/\partial S\,\partial t$ — daily decay of delta; important for overnight hedgers |

#### Bump-and-reprice (universal)

For models without closed-form Greeks like the American options, Heston, etc., all Greeks are computed via central finite differences:

$$
\delta = [V(S+dS) - V(S-dS)] / (2*dS)  \qquad    dS = 1\% \text{ of S}
$$
$$
\gamma = [V(S+dS) - 2V(S) + V(S-dS)] / dS^2
$$
$$
\nu  = [V(\sigma + d\sigma) - V(\sigma - d\sigma)] / (2*d\sigma)  \qquad   d\sigma = 1\% \text{ of } \sigma
$$
$$
\theta = [V(T-dt) - V(T)] / dt / 365   \qquad     dt = 1 day
$$
$$
\rho   = [V(r+dr) - V(r-dr)] / (2*dr)  \qquad    dr = 1 bp
$$


In `GreeksCalculator.greeks_surface()`, this project computes a 2D array of any Greek across a range of spot prices and maturities, useful for visualising how delta/gamma change as the option moves through moneyness.

### Hedging Simulation

Hedging a long call position means continuously selling delta shares of the underlying to stay delta-neutral. In practice rebalancing is discrete with daily, weekly, monthly manual, which introduces hedging error.

Based on the Taylor expansion, the option value change can be decomposed as:

$$
dV \approx \delta dS  +  \frac{1}{2} \gamma {dS}^2  +  \nu d\sigma  +  \theta dt  +  \varepsilon
$$

Each term is accumulated (discounted) across all rebalancing steps:
- Delta PnL: linear exposure to spot moves — cancelled by the hedge
- Gamma PnL: convexity benefit from large spot moves (long option = long gamma)
- Vega PnL: exposure to vol changes (zero under constant-vol GBM)
- Theta PnL: time decay cost of holding the option
- Residual: discrete hedging error (higher-order terms, rebalancing lag)

### Portfolio Risk (VaR & CVaR)

Value at Risk (VaR) answersthe question of 'what is the maximum loss not exceeded with probability p over a horizon?'

$$ \mathbb{P}(\text{loss} > VaR) = 1 - \text{confidence} $$

Conditional VaR (CVaR / Expected Shortfall) is the expected loss given that the loss exceeds VaR:

$$ \text{CVaR} = \mathbb{E}[\text{loss} | \text{loss} > \text{VaR}] $$

CVaR is always $\ge$ VaR and is a coherent risk measure, satisfies monotonicity, and sub-additivity, which VaR does not hold.

#### Three estimation methods

**Historical simulation**: sort observed P&L, and take the corresponding confidence quantile. No distributional assumption. Captures fat tails and skewness, but
limited by available history.

**Parametric (Gaussian)**： Tractable and transparent. Underestimates tail risk when returns are non-normal (fat tails, skew). The sqrt(horizon) scaling is the
square-root-of-time rule, validated to machine precision in this project.

$$
\text{VaR}  = -(\mu - z_\alpha * \sigma) * \sqrt{\text{horizon}}
$$
$$
\text{CVaR} = -(\mu - \sigma * \phi(z_\alpha) / (1-conf)) * \sqrt{\text{horizon}}
$$

**Monte Carlo**: simulate forward P&L under GBM, apply the empirical quantile. Incorporates model dynamics and can be extended to non-linear portfolios such as option positions.

#### Square-root rule of time

Under i.i.d. returns, h-day VaR = 1-day VaR * sqrt(h).

### Implied Volatility Extraction

Raw market option quotes contain noise like wide bid-ask spreads on illiquid strikes, stale last-traded prices, and occasional violations of no-arbitrage bounds from data lags. Inverting BSM on unfiltered data produces unreliable or NaN implied vols. This project uses a cleaning pipeline in `src/calibration/implied_vol.py`, filtered out min price, max spread percentage, no-arbitrage bounds, and dropped failed inversions after inverting BSM.

### Volatility Surface (SVI)

$$
w(k) = a + b * ( \rho * (k-m) + \sqrt{(k-m)^2 + \sigma^2} )
$$
$$
k = \ln(\frac{K}{S})
$$
$$
w(k) = \sigma_{BS}(k)^2 * T
$$

A raw grid of market implied vols is noisy and has gaps between strikes. The Stochastic Volatility Inspired parameterization, introduced by Gatheral, fits a smooth 5 parameter curve to each maturity slice that captures the smile/skew shape shile enabling clean interpolation and extrapolation.