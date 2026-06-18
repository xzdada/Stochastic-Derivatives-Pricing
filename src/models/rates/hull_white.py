"""
Hull-White short rate model (extended Vasicek).

Adds a deterministic, time-dependent drift lambda(t) to the Vasicek model, allowing exact calibration to the initial market yield curve while
retaining mean reversion. 
See README.md — Theory — Interest Rate Models — Hull-White for the full mathematical derivation.

This implementation supports the constant-lambda case, equivalent to Vasicek with a generalised drift, and accepts a custom lambda_func for full market-curve fitting.
"""

from __future__ import annotations

import numpy as np

from src.models.rates.base_short_rate import ShortRateModel


class HullWhiteModel(ShortRateModel):
    """
    Hull-White one-factor short rate model.

        dr_t = [lambda(t) - a*r_t] dt + sigma dW_t

    Parameters
    ----------
    r0 : float
        Initial short rate.
    a : float
        Mean-reversion speed. Must be > 0.
    sigma : float
        Volatility of the short rate. Must be > 0.
    lambda_func : callable, optional
        Deterministic time-dependent drift lambda(t). If None (default), a constant lambda(t) = a*b is used (reduces to Vasicek with long-run mean b).
    long_run_mean : float
        Used only when lambda_func is None: sets lambda(t) = a * long_run_mean,
        making this equivalent to Vasicek(r0, a, long_run_mean, sigma).
    """

    def __init__(
        self,
        r0: float,
        a: float,
        sigma: float,
        lambda_func: callable | None = None,
        long_run_mean: float = 0.04,
    ):
        super().__init__(r0)
        if a <= 0:
            raise ValueError("Mean reversion speed (a) must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive.")
        self.a = float(a)
        self.sigma = float(sigma)
        self.long_run_mean = float(long_run_mean)
        self.lambda_func = (
            lambda_func if lambda_func is not None
            else (lambda t: self.a * self.long_run_mean)
        )

    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:

        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        exp_a_dt = np.exp(-self.a * dt)
        var = (self.sigma ** 2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * dt))
        std = np.sqrt(var)

        r = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        Z = rng.standard_normal((n_paths, n_steps))
        for k in range(n_steps):
            t_k = k * dt
            lambda_k = self.lambda_func(t_k)
            mean = r[:, k] * exp_a_dt + (lambda_k / self.a) * (1 - exp_a_dt)
            r[:, k + 1] = mean + std * Z[:, k]

        return r

    def bond_price(self, T: float) -> float:
        a, sigma, b = self.a, self.sigma, self.long_run_mean
        B = (1 - np.exp(-a * T)) / a
        A = np.exp((b - sigma ** 2 / (2 *a ** 2)) * (B - T) - (sigma ** 2 * B ** 2) / (4 * a))

        return float(A * np.exp(-B * self.r0))

    def params(self) -> dict:
        return {"model": "Hull-White", "r0": self.r0, "a": self.a, "sigma": self.sigma, "long_run_mean": self.long_run_mean}
