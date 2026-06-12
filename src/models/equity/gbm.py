"""
Geometric Brownian Motion (GBM) path simulator.

Mathematical foundation
-----------------------
Under the risk-neutral measure Q, the asset price follows:

    dS = (r - q) S dt + sigma S dW_t^Q

Applying Ito's lemma to ln(S):

    d(ln S) = (r - q - sigma^2 / 2) dt + sigma dW_t^Q

This SDE has an exact solution — no discretisation error:

    S(t + Delta(t)) = S(t) * exp( (r - q - sigma^2 /2) Delta(t)  +  sigma sqrt{Delta(t)} * Z )

where Z sim N(0, 1) i.i.d.
"""

from __future__ import annotations

import numpy as np


class GBMModel:
    """
    Geometric Brownian Motion under the risk-neutral measure Q.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    r : float
        Continuously compounded risk-free rate (annualized decimal).
    sigma : float
        Volatility (annualized decimal). Must be > 0.
    q : float
        Continuous dividend yield (annualized decimal). Default 0.
    """

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ):
        if S0 <= 0:
            raise ValueError("S0 must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if q < 0:
            raise ValueError("q must be non-negative.")

        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths using the exact log-normal update.

        Parameters
        ----------
        T : float
            Total time horizon in years.
        n_paths : int
            Number of Monte Carlo paths.
        n_steps : int
            Number of time steps per path. For European options, n_steps=1.
            n_steps > 1 for path-dependent products or to record intermediate prices.
        seed : int, optional
            Default seed is 926.Ignored if rng is supplied.
        rng : np.random.Generator, optional
            Pre-constructed random number generator. Takes priority over seed.

        Returns
        -------
        np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths. Column 0 is S_0 for every path.
            Column j is S_{j * Delta(t)}. Final column is S_T.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps

        # Drift and diffusion coefficients (per step)
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Draw all random increments at once: shape (n_paths, n_steps)
        Z = rng.standard_normal((n_paths, n_steps))

        # Log-returns per step
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)

        # Prepend 0 for S_0, then exponentiate
        log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
        paths = self.S0 * np.exp(log_paths)

        return paths

    def simulate_terminal(
        self,
        T: float,
        n_paths: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate only the terminal price S_T (single-step).

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Terminal asset prices S_T.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * T
        diffusion = self.sigma * np.sqrt(T)
        Z = rng.standard_normal(n_paths)
        return self.S0 * np.exp(drift + diffusion * Z)

    def simulate_antithetic(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate paired antithetic paths.

        For each random draw Z, also simulate with -Z. 
        The two resulting path matrices are negatively correlated, which reduces variance when averaging their payoffs.

        Parameters
        ----------
        n_paths : int
            Number of path pairs. Total paths simulated = 2 * n_paths.

        Returns
        -------
        (paths, paths_anti) : tuple of ndarray, each shape (n_paths, n_steps+1)
            Original paths and their antithetic counterparts.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        dt = T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))

        def _build_paths(z: np.ndarray) -> np.ndarray:
            log_r = drift + diffusion * z
            log_p = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_r, axis=1)])
            return self.S0 * np.exp(log_p)

        return _build_paths(Z), _build_paths(-Z)

    def params(self) -> dict:
        return {
            "model":  "GBM",
            "S0": self.S0,
            "r": self.r,
            "sigma": self.sigma,
            "q": self.q,
        }

    def __repr__(self) -> str:
        return (f"GBMModel(S0={self.S0}, r={self.r}, sigma={self.sigma}, q={self.q})")