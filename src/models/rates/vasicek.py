"""
Vasicek short rate model.

Mean-reverting Gaussian short rate model with closed-form zero-coupon bond pricing. 

See README.md — Theory — Interest Rate Models — Vasicek for the full mathematical derivation.
"""

from __future__ import annotations

import numpy as np

from src.models.rates.base_short_rate import ShortRateModel


class VasicekModel(ShortRateModel):
    """
        dr_t = a(b - r_t) dt + sigma dW_t

    Parameters
    ----------
    r0 : float
        Initial short rate.
    a : float
        Mean-reversion speed. Must be > 0.
    b : float
        Long-run mean short rate.
    sigma : float
        Volatility of the short rate. Must be > 0.
    """

    def __init__(self, r0: float, a: float, b: float, sigma: float):
        super().__init__(r0)
        if a <= 0:
            raise ValueError("Mean-reversion speed must be positive.")
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)

    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate r(t) paths using the exact transition distribution.

        Conditional on r_t, r_{t+dt} is Gaussian with known mean and variance, no discretisation bias
        """
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
            mean = r[:, k] * exp_a_dt + self.b * (1 - exp_a_dt)
            r[:, k + 1] = mean + std * Z[:, k]

        return r

    def bond_price(self, T: float) -> float:
        """
        Closed-form zero-coupon bond price:

            P(0,T) = A(T) * exp(-B(T) * r0)

            B(T) = (1 - e^{-aT}) / a
            A(T) = exp[ (b - sigma^2/(2a^2))(B(T) - T) - sigma^2 B(T)^2 / (4a) ]
        """
        a, b, sigma = self.a, self.b, self.sigma
        B = (1 - np.exp(-a * T)) / a
        A = np.exp((b - sigma ** 2 / (2 * a ** 2)) * (B - T) - (sigma ** 2 * B ** 2) / (4 * a))
        return float(A * np.exp(-B * self.r0))

    def params(self) -> dict:
        return {"model": "Vasicek", "r0": self.r0, "a": self.a, "b": self.b, "sigma": self.sigma}