"""
Three-factor cascade Gaussian+ short rate model.

See README.md — Theory for the full mathematical derivation.
"""

from __future__ import annotations

import numpy as np

from src.models.rates.base_short_rate import ShortRateModel


class CascadeGaussianPlusModel(ShortRateModel):
    """
    Three-factor cascade Gaussian+ short rate model.

    Parameters
    ----------
    r0, m0, L0 : float
        Initial values of the short rate, medium-term factor, and long-term factor.
    mu : float
        Fixed long-run mean that the long-term factor L reverts to.
    a_r, a_m, a_L : float
        Mean-reversion speeds for r, m, L respectively. Must all be > 0. Typically a_r > a_m > a_L because short rate reverts fastest to its
        target, the long-term factor reverts slowest to mu.
    sigma_m, sigma_L : float
        Volatility of the medium-term and long-term factors. Must be > 0.
        Note: r has NO sigma_r because it reflects the central bank policy and has no independent diffusion term.
    rho : float
        Correlation coefficient between dW1 and dW2. Must be in (-1, 1).
    """

    def __init__(
        self,
        r0: float,
        m0: float,
        L0: float,
        mu: float,
        a_r: float,
        a_m: float,
        a_L: float,
        sigma_m: float,
        sigma_L: float,
        rho: float,
    ):
        super().__init__(r0)
        self._validate(a_r, a_m, a_L, sigma_m, sigma_L, rho)

        self.m0 = float(m0)
        self.L0 = float(L0)
        self.mu = float(mu)
        self.a_r = float(a_r)
        self.a_m = float(a_m)
        self.a_L = float(a_L)
        self.sigma_m = float(sigma_m)
        self.sigma_L = float(sigma_L)
        self.rho = float(rho)

    @staticmethod
    def _validate(a_r, a_m, a_L, sigma_m, sigma_L, rho) -> None:
        if a_r <= 0 or a_m <= 0 or a_L <= 0:
            raise ValueError("a_r, a_m, a_L must all be positive.")
        if sigma_m <= 0 or sigma_L <= 0:
            raise ValueError("sigma_m and sigma_L must be positive.")
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must be in (-1, 1).")


    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        return_factors: bool = False,
    ) -> np.ndarray:
        """
        Simulate the cascade system via exact-per-step discretisation.

        L is a standard one-factor OU process — simulated with its exact Gaussian transition (no discretisation bias). 
        m's drift depends on the current L value at each step. 
        r's evolution is a pure ODE driven by m and there is no diffusion term.
        """

        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        rho_bar = np.sqrt(1.0 - self.rho ** 2)

        r = np.empty((n_paths, n_steps + 1))
        m = np.empty((n_paths, n_steps + 1))
        L = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.r0
        m[:, 0] = self.m0
        L[:, 0] = self.L0

        # Exact per-step OU transition coefficients for L
        exp_aL_dt = np.exp(-self.a_L * dt)
        var_L = (self.sigma_L ** 2 / (2 * self.a_L)) * (1 - np.exp(-2 * self.a_L * dt))
        std_L = np.sqrt(var_L)

        # Exact per-step OU transition coefficients for m
        exp_am_dt = np.exp(-self.a_m * dt)
        var_m = (self.sigma_m ** 2 / (2 * self.a_m)) * (1 - np.exp(-2 * self.a_m * dt))
        std_m = np.sqrt(var_m)

        # Exact per-step deterministic relaxation for r toward m (piecewise-constant m)
        exp_ar_dt = np.exp(-self.a_r * dt)

        for k in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            # Step L: 
            L[:, k + 1] = (L[:, k] * exp_aL_dt + self.mu * (1 - exp_aL_dt) + std_L * Z1)

            # Step m: OU toward current L[k], driven by correlated (Z1, Z2)
            eps_m = self.rho * Z1 + rho_bar * Z2
            m[:, k + 1] = (m[:, k] * exp_am_dt + L[:, k] * (1 - exp_am_dt) + std_m * eps_m)

            # Step r: pure relaxation toward current m[k] (no own diffusion) 
            r[:, k + 1] = r[:, k] * exp_ar_dt + m[:, k] * (1 - exp_ar_dt)

        if return_factors:
            return r, (m, L)
        return r


    def bond_price(
        self,
        T: float,
        n_paths: int = 200_000,
        n_steps: int | None = None,
        seed: int = 42,
    ) -> float:
        """
        Zero-coupon bond price via Monte Carlo integration of r(t):

            P(0,T) = E^Q[ exp(-integral_0^T r_s ds) ]
        """
        if n_steps is None:
            n_steps = max(100, int(50 * T))

        rng = np.random.default_rng(seed)
        r_paths = self.simulate(T=T, n_paths=n_paths, n_steps=n_steps, rng=rng)

        dt = T / n_steps
        integral = np.trapezoid(r_paths, dx=dt, axis=1)
        return float(np.mean(np.exp(-integral)))


    def half_lives(self) -> dict:
        """
        Mean-reversion half-lives for each layer of the cascade:
            r: ln(2)/a_r  (relaxation toward m)
            m: ln(2)/a_m (relaxation toward L)
            L: ln(2)/a_L (relaxation toward mu)
        """
        return {
            "r": np.log(2) / self.a_r,
            "m": np.log(2) / self.a_m,
            "L": np.log(2) / self.a_L,
        }

    def params(self) -> dict:
        return {
            "model":   "CascadeGaussianPlus",
            "r0": self.r0,
            "m0": self.m0,
            "L0": self.L0,
            "mu": self.mu,
            "a_r": self.a_r,
            "a_m": self.a_m,
            "a_L": self.a_L,
            "sigma_m": self.sigma_m,
            "sigma_L": self.sigma_L,
            "rho": self.rho,
        }

    def __repr__(self) -> str:
        hl = self.half_lives()
        return (
            f"CascadeGaussianPlusModel(r0={self.r0}, m0={self.m0}, L0={self.L0}, mu={self.mu}, half_lives={{'r':{hl['r']:.2f}, 'm':{hl['m']:.2f}, 'L':{hl['L']:.2f}}})"
        )