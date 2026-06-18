"""
Cox-Ingersoll-Ross short rate model.

Mean-reverting short rate model with a square-root diffusion term that assures r_t non-negative. 
See README.md — Theory — Interest Rate Models — CIR for the full mathematical derivation.
"""

from __future__ import annotations

import numpy as np

from src.models.rates.base_short_rate import ShortRateModel


class CIRModel(ShortRateModel):
    """
    Cox-Ingersoll-Ross (1985) short rate model.

        dr_t = a(b - r_t) dt + sigma sqrt(r_t) dW_t
    """

    def __init__(self, r0: float, a: float, b: float, sigma: float):
        super().__init__(r0)
        if r0 <= 0:
            raise ValueError("r0 must be positive for CIR.")
        if a <= 0:
            raise ValueError("mean reversion speed must be positive.")
        if b <= 0:
            raise ValueError("long term mean must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
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

        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        r = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        Z = rng.standard_normal((n_paths, n_steps))
        for k in range(n_steps):
            r_pos = np.maximum(r[:, k], 0.0)
            r[:, k + 1] = (r[:, k] + self.a * (self.b - r_pos) * dt + self.sigma * np.sqrt(r_pos) * sqrt_dt * Z[:, k])

        return r

    def bond_price(self, T: float) -> float:
        """
        Closed-form zero-coupon bond price:
        """
        a, b, sigma = self.a, self.b, self.sigma
        h = np.sqrt(a ** 2 + 2 * sigma ** 2)
        exp_hT = np.exp(h * T)

        denom = (a + h) * (exp_hT - 1) + 2 * h
        B = (2 * (exp_hT - 1)) / denom
        A = ((2 * h * np.exp((a + h)* T / 2)) / denom) ** (2 * a * b / sigma ** 2)

        return float(A * np.exp(-B * self.r0))

    def feller_condition(self) -> dict:
        """
        Check Feller condition
        """
        lhs = 2 * self.a * self.b
        rhs = self.sigma ** 2
        return {"lhs": lhs, "rhs": rhs, "satisfied": lhs >= rhs, "ratio": lhs / rhs}

    def params(self) -> dict:
        return {"model": "CIR", "r0": self.r0, "a": self.a, "b": self.b, "sigma": self.sigma}