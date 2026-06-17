"""
Heston stochastic volatility model path simulator.

Simulates joint dynamics for asset price S and instantaneous variance v under the risk-neutral measure Q. Uses Euler discretisation with full
truncation for the variance process.
 
See README.md — Theory — Heston Model for full mathematical derivation, parameter descriptions, Feller condition, and smile intuition.
"""

from __future__ import annotations

import numpy as np


class HestonModel:
    """
    Heston stochastic volatility model under risk-neutral measure Q.

    Parameters
    ----------
    S0 : float, initial asset price.
    v0 : float, initial instantaneous variance, v_0 = sigma_0^2.
    r : float, continuously compounded risk-free rate.
    q : float, continuous dividend yield.
    kappa : float, mean-reversion speed of variance. Must be > 0.
    theta : float, long-run mean of variance. Must be > 0.
    xi : float, volatility of variance process. Must be > 0.
    rho : float, correlation between asset and variance Brownian motions,  -1 < rho < 1.
    """

    def __init__(
        self,
        S0: float,
        v0: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ):
        self._validate(S0, v0, kappa, theta, xi, rho)
        self.S0 = float(S0)
        self.v0 = float(v0)
        self.r = float(r)
        self.q = float(q)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)

    @staticmethod
    def _validate(S0, v0, kappa, theta, xi, rho) -> None:
        if S0 <= 0:
            raise ValueError("S0 must be positive.")
        if v0 <= 0:
            raise ValueError("v0 must be positive.")
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if xi <= 0:
            raise ValueError("xi must be positive.")
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must be in (-1, 1).")


    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        return_var: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Simulate Heston paths via Euler discretisation with full truncation.

        Parameters
        ----------
        T : float
            Time horizon in years.
        n_paths : int
            Number of Monte Carlo paths.
        n_steps : int
            Number of time steps. More steps means less discretisation bias.
        seed : int, optional
        rng : np.random.Generator, optional
        return_var : bool
            If True (default), return (paths, var_paths).
            If False, return only asset price paths.

        Returns
        -------
        paths : ndarray, shape (n_paths, n_steps + 1)
            Simulated asset price paths. Column 0 = S0.
        var_paths : ndarray, shape (n_paths, n_steps + 1)
            Simulated variance paths. Column 0 = v0.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        rho_bar = np.sqrt(1.0 - self.rho ** 2)

        log_S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))

        log_S[:, 0] = np.log(self.S0)
        v[:, 0] = self.v0

        Z = rng.standard_normal((2, n_paths, n_steps))
        Z_S = Z[0] 
        Z_v = self.rho * Z[0] + rho_bar * Z[1]

        for k in range(n_steps):
            v_k = v[:, k]
            v_pos = np.maximum(v_k, 0.0) 
            sqrt_v = np.sqrt(v_pos)

            # Variance update
            v[:, k + 1] = (v_k + self.kappa * (self.theta - v_pos) * dt + self.xi * sqrt_v * sqrt_dt * Z_v[:, k])

            # Asset price update
            log_S[:, k + 1] = (log_S[:, k]
                                + (self.r - self.q - 0.5 * v_pos) * dt
                                + sqrt_v * sqrt_dt * Z_S[:, k])

        paths = np.exp(log_S)

        if return_var:
            return paths, v
        return paths

    def simulate_terminal(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate only terminal prices S_T for European options.

        Returns
        -------
        ndarray, shape (n_paths,)
            Terminal asset prices S_T.
        """
        paths = self.simulate(T, n_paths, n_steps, seed=seed, rng=rng, return_var=False)
        return paths[:, -1]

    def simulate_antithetic(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate antithetic path pairs for variance reduction.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        rho_bar = np.sqrt(1.0 - self.rho ** 2)

        Z = rng.standard_normal((2, n_paths, n_steps))

        def _build(sign: float) -> np.ndarray:
            Z_S = sign * Z[0]
            Z_v = self.rho * sign * Z[0] + rho_bar * sign * Z[1]

            log_S = np.empty((n_paths, n_steps + 1))
            v = np.empty((n_paths, n_steps + 1))
            log_S[:, 0] = np.log(self.S0)
            v[:, 0] = self.v0

            for k in range(n_steps):
                v_pos = np.maximum(v[:, k], 0.0)
                sqrt_v = np.sqrt(v_pos)
                v[:, k+1] = (v[:, k] + self.kappa * (self.theta - v_pos) * dt + self.xi * sqrt_v * sqrt_dt * Z_v[:, k])
                log_S[:, k+1] = (log_S[:, k] + (self.r - self.q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z_S[:, k])
            return np.exp(log_S)

        return _build(+1.0), _build(-1.0)

    def feller_condition(self) -> dict:
        """
        Check whether the Feller condition is satisfied.

        When violated, the variance process can reach zero, and Euler discretisation introduces additional bias.
        """
        lhs = 2.0 * self.kappa * self.theta
        rhs = self.xi ** 2
        return {
            "lhs": lhs,
            "rhs": rhs,
            "satisfied": lhs > rhs,
            "ratio": lhs / rhs,
        }

    def params(self) -> dict:
        return {
            "model": "Heston",
            "S0": self.S0,
            "v0": self.v0,
            "r": self.r,
            "q": self.q,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }

    def __repr__(self) -> str:
        fc = self.feller_condition()
        return (
            f"HestonModel(S0={self.S0}, v0={self.v0}, r={self.r}, q={self.q}, κ={self.kappa}, θ={self.theta}, ξ={self.xi}, ρ={self.rho}) "
            f"[Feller: {'Satisfied' if fc['satisfied'] else 'Violated'} {fc['ratio']:.2f}x]"
        )