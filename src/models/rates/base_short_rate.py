"""
Abstract base class for short rate models.

All short rate models (Vasicek, CIR, Ho-Lee, Hull-White) inherit from ShortRateModel and must implement:
  - simulate(T, n_paths, n_steps, rng) : simulate r(t) paths
  - bond_price(T) : analytical zero-coupon bond price

See README.md — Theory — Interest Rate Models for the mathematical foundation of each model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ShortRateModel(ABC):
    """
    Abstract base class for short rate models.

    Parameters
    ----------
    r0 : Current short rate.
    """

    def __init__(self, r0: float):
        self.r0 = float(r0)

    @abstractmethod
    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate short rate paths r(t).
        """

    @abstractmethod
    def bond_price(self, T: float) -> float:
        """
        Analytical price of a zero-coupon bond paying $1 at maturity T, given the current short rate r0.
        """

    def bond_yield(self, T: float) -> float:
        """
        Continuously compounded yield implied by the bond price:
            y(T) = -ln(P(0,T)) / T
        """
        P = self.bond_price(T)
        return float(-np.log(P) / T)

    def yield_curve(self, maturities: np.ndarray) -> np.ndarray:
        """Return the model-implied yield curve across a range of maturities."""
        return np.array([self.bond_yield(T) for T in maturities])

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.__class__.__name__}({params})"

    @abstractmethod
    def params(self) -> dict:
        """Return model parameters as a dictionary."""