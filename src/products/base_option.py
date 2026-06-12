"""
Abstract base class for all option products.

Every concrete option (European, American, Asian, Barrier, ......) inherits from Option and must implement:
  - payoff(paths) : compute payoff from a matrix of simulated paths
  - description() : human-readable summary string

The base class stores the common contract parameters shared by all vanilla and exotic options.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Option(ABC):
    """
    Abstract base class for all option products.

    Parameters
    ----------
    S : float
        Current spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualized decimal).
    sigma : float
        Volatility of the underlying (annualized decimal).
    q : float
        Continuous dividend yield (annualized decimal). Default 0.
    option_type : {"call", "put"}
        Type of option.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ):
        if S <= 0:
            raise ValueError("Spot price S must be positive.")
        if K <= 0:
            raise ValueError("Strike K must be positive.")
        if T <= 0:
            raise ValueError("Time to expiry T must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive.")
        if q < 0:
            raise ValueError("Dividend yield q must be non-negative.")
        if option_type.lower() not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type.lower()

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the option payoff from simulated price paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated asset price paths. 
            Column 0 is S_0, column -1 is S_T.
            Intermediate columns are available for path-dependent products.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoff for each path.
        """

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the contract."""

    def discount_factor(self) -> float:
        """Return the continuous discount factor e^{-rT}."""
        return float(np.exp(-self.r * self.T))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (S={self.S}, K={self.K}, T={self.T}, r={self.r}, sigma={self.sigma}, q={self.q}, type={self.option_type})")
