"""
Ho-Lee short rate model.

Short rate model with a Gaussian diffusion term and a deterministic, time-dependent drift lambda(t) calibrated to reproduce the initial market yield curve.

See README.md — Theory — Interest Rate Models — Ho-Lee Model for the full mathematical derivation.
"""

from __future__ import annotations

import numpy as np

from src.models.rates.base_short_rate import ShortRateModel


class HoLeeModel(ShortRateModel):
    """
    Ho-Lee (1986) short rate model.

        dr_t = lambda(t) dt + sigma dW_t

    Parameters
    ----------
    r0 : float
        Initial short rate.
    sigma : float
        Volatility of the short rate (constant). Must be > 0.
    lambda_func : callable, optional
        Deterministic drift function lambda(t). 
        If None (default), a constant drift lambda(t) = 0 is used, which reduces to driftless Brownian motion plus r0 (Model 1).

    Notes
    -----
    Like Vasicek, r_t is Gaussian and can become negative. Ho-Lee has no mean reversion, which means the short rate is a pure drifted random walk, so rate paths can wander arbitrarily far from r0 over long horizons.
    """

    def __init__(
        self,
        r0: float,
        sigma: float,
        lambda_func: callable | None = None,
    ):
        super().__init__(r0)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)
        self.lambda_func = lambda_func if lambda_func is not None else (lambda t: 0.0)

    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate r(t) paths.

            r_{t+dt} = r_t + lambda(t) dt + sigma * sqrt(dt) * Z

        This discretisation has no bias because the SDE has constant volatility and a deterministic drift.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        r = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        Z = rng.standard_normal((n_paths, n_steps))
        for k in range(n_steps):
            t_k = k * dt
            r[:, k + 1] = (r[:, k] + self.lambda_func(t_k) * dt + self.sigma * sqrt_dt * Z[:, k])

        return r

    def bond_price(self, T: float) -> float:
        """
        Closed-form zero-coupon bond price under constant lambda(t) = lambda_0:

            P(0,T) = exp[ -r0*T - lambda_0 * T^2/2 + sigma^2 * T^3/6 ]

        For the default lambda(t) = 0 case this simplifies to:

            P(0,T) = exp[ -r0*T + sigma^2 * T^3/6 ]
        """
        lambda_0 = self.lambda_func(0.0)
        log_P = (-self.r0 * T - lambda_0 * T ** 2 / 2 + (self.sigma ** 2 ) * ( T ** 3 ) / 6)

        return float(np.exp(log_P))

    def params(self) -> dict:
        return {"model": "Ho-Lee", "r0": self.r0, "sigma": self.sigma}
