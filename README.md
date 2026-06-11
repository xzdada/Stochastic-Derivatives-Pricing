


Repository Structure

``` bash
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
│ │ └── rates/
│ │ ├── base_short_rate.py # abstract ShortRateModel
│ │ ├── vasicek.py
│ │ ├── cir.py
│ │ ├── ho_lee.py
│ │ ├── hull_white.py
│ │ └── hjm.py
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
│ ├── test_mc_convergence.py # MC price → analytical as N→inf
│ ├── test_greeks.py # bump-and-reprice vs closed form
│ └── test_parity.py # put-call parity violations
├── requirements.txt
└── README.md
```

