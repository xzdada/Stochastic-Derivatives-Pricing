"""
GARCH(1,1) model for realized (historical) volatility estimation.

Fits a GARCH(1,1) model to historical log returns via maximum likelihood, producing a time-varying conditional volatility estimate that can be
compared against the market's option-implied volatility, and used for analysing the volatility risk premium.

See README.md — Theory for the full model specification and the economic interpretation of the realized-vs-implied volatility spread.

Usage
-----
    from src.models.equity.garch import GARCH11
    import numpy as np

    returns = np.diff(np.log(prices))
    model = GARCH11()
    model.fit(returns)

    cond_vol = model.conditional_vol()  # in-sample fitted vol path
    forecast = model.forecast(horizon=30)   # forward vol forecast
    long_run = model.long_run_vol()   # unconditional long-run vol
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class GARCHFitResult:
    """
    omega, alpha, beta : Fitted GARCH(1,1) parameters.
    log_likelihood : float
        Maximised log-likelihood.
    n_obs : Number of return observations used.
    persistence : (alpha + beta) must be < 1 for stationarity.
    long_run_variance : Unconditional variance: omega / (1 - alpha - beta).
    converged : bool
        Whether the optimiser reported successful convergence.
    """
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    n_obs: int
    persistence: float
    long_run_variance: float
    converged: bool

    def __str__(self) -> str:
        return (
            f"GARCH(1,1) Fit | omega={self.omega:.2e}  alpha={self.alpha:.4f}  beta={self.beta:.4f}\n"
            f" persistence (a+b)  = {self.persistence:.4f} (must be < 1 for stationarity)\n"
            f" long-run vol (ann) = {np.sqrt(self.long_run_variance*252)*100:.2f}%\n"
            f" log-likelihood     = {self.log_likelihood:.2f}\n"
            f" n_obs = {self.n_obs}  converged = {self.converged}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class GARCH11:

    def __init__(self):
        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta: float | None = None
        self.mu: float | None = None
        self._returns: np.ndarray | None = None
        self._sigma2:  np.ndarray | None = None
        self.fit_result: GARCHFitResult | None = None


    def fit(
        self,
        returns: np.ndarray,
        demean: bool = True,
    ) -> GARCHFitResult:

        r = np.asarray(returns, dtype=float)
        self.mu = float(np.mean(r)) if demean else 0.0
        eps = r - self.mu
        self._returns = eps
        n = len(eps)

        # Initial guess
        var0 = float(np.var(eps))
        x0 = [0.1 * var0, 0.05, 0.90]   # omega, alpha, beta

        bounds = [
            (1e-12, var0 * 10),  # omega > 0
            (1e-6,  0.5), 
            (1e-6,  0.999),
        ]

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if alpha + beta >= 1.0:
                return 1e10
            sigma2 = self._recursion(eps, omega, alpha, beta)

            ll = -0.5 * np.sum(np.log(sigma2) + eps ** 2 / sigma2)
            return -ll

        result = minimize(
            neg_log_likelihood, x0=x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        omega, alpha, beta = result.x
        self.omega, self.alpha, self.beta = omega, alpha, beta
        self._sigma2 = self._recursion(eps, omega, alpha, beta)

        persistence = alpha + beta
        long_run_var = omega / (1 - persistence) if persistence < 1 else np.nan

        self.fit_result = GARCHFitResult(
            omega=float(omega), alpha=float(alpha), beta=float(beta),
            log_likelihood=float(-result.fun),
            n_obs=n,
            persistence=float(persistence),
            long_run_variance=float(long_run_var),
            converged=bool(result.success),
        )
        return self.fit_result

    @staticmethod
    def _recursion(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        """
        Compute the GARCH(1,1) conditional variance recursion.

            sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}

        Initialised at the unconditional (sample) variance.
        """
        n = len(eps)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(eps)   # initialise at sample variance
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t-1] ** 2 + beta * sigma2[t-1]
        return sigma2

    # ------------------------------------------------------------------
    # Querying the fitted model
    # ------------------------------------------------------------------

    def conditional_vol(self, annualize: bool = True) -> np.ndarray:
        if self._sigma2 is None:
            raise RuntimeError("Model not fitted.")
        vol = np.sqrt(self._sigma2)
        return vol * np.sqrt(252) if annualize else vol

    def long_run_vol(self, annualize: bool = True) -> float:
        """
        Unconditional (long-run) volatility implied by the fitted model.
        """
        if self.fit_result is None:
            raise RuntimeError("Model not fitted.")
        var = self.fit_result.long_run_variance
        vol = np.sqrt(var)
        return float(vol * np.sqrt(252)) if annualize else float(vol)

    def forecast(self, horizon: int, annualize: bool = True) -> np.ndarray:
        """
        Forecast conditional variance forward using the GARCH recursion's mean-reverting property.
        """
        if self.fit_result is None:
            raise RuntimeError("Model not fitted.")

        sigma2_inf = self.fit_result.long_run_variance
        sigma2_t = self._sigma2[-1]
        persistence = self.fit_result.persistence

        h = np.arange(1, horizon + 1)
        sigma2_fcast = sigma2_inf + (persistence ** h) * (sigma2_t - sigma2_inf)
        vol_fcast = np.sqrt(np.maximum(sigma2_fcast, 1e-12))

        return vol_fcast * np.sqrt(252) if annualize else vol_fcast

    def half_life(self) -> float:
        """
        Volatility shock half-life in days: ln(0.5) / ln(alpha+beta).

        Measures how quickly a volatility shock decays back toward the long-run level.
        """
        if self.fit_result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        p = self.fit_result.persistence
        if p <= 0 or p >= 1:
            return np.inf
        return float(np.log(0.5) / np.log(p))