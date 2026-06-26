"""
4 Variants of Barrier options

A barrier option's payoff depends on whether the underlying path touches a barrier level B at any point before expiry, in addition to the usual vanilla payoff at maturity.

Variants
--------
    up-and-out : pays the vanilla payoff unless the path ever goes >= B, in which case it pays 0 and typically expires worthless immediately.
    up-and-in : pays the vanilla payoff only if the path ever goes >= B.
    down-and-out : pays the vanilla payoff unless the path ever goes <= B.
    down-and-in : pays the vanilla payoff only if the path ever goes <= B.

In-out parity (model-free, holds exactly for any barrier level): knock-in price + knock-out price = vanilla price

This is because every path either knocks in or knocks out (mutually exclusive and exhaustive), so summing the two payoffs recovers the vanilla payoff. 
"""

from __future__ import annotations

import numpy as np

from src.products.base_option import Option


class BarrierOption(Option):
    """
    Barrier call or put option.

    Parameters
    ----------
    barrier : float
        The barrier level B.
    barrier_type : {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
    rebate : float
        Cash rebate paid if a knock-out option is extinguished or if a knock-in option never activates. Default 0.
    """

    VALID_TYPES = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        barrier_type: str,
        q: float = 0.0,
        option_type: str = "call",
        rebate: float = 0.0,
    ):
        super().__init__(S, K, T, r, sigma, q, option_type)
        if barrier_type not in self.VALID_TYPES:
            raise ValueError(f"barrier_type must be one of {self.VALID_TYPES}.")
        if barrier <= 0:
            raise ValueError("barrier must be positive.")

        if barrier_type.startswith("up") and barrier <= S:
            raise ValueError("up barrier must be above spot S.")
        if barrier_type.startswith("down") and barrier >= S:
            raise ValueError("down barrier must be below spot S.")

        self.barrier = float(barrier)
        self.barrier_type = barrier_type
        self.rebate = float(rebate)


    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the barrier option payoff from simulated paths.
        """
        S_T = paths[:, -1]

        if self.option_type == "call":
            vanilla_payoff = np.maximum(S_T - self.K, 0.0)
        else:
            vanilla_payoff = np.maximum(self.K - S_T, 0.0)

        if self.barrier_type.startswith("up"):
            touched = np.any(paths >= self.barrier, axis=1)
        else:
            touched = np.any(paths <= self.barrier, axis=1)

        if self.barrier_type.endswith("out"):
            # Pays vanilla unless touched
            payoff = np.where(touched, self.rebate, vanilla_payoff)
        else: 
            # Pays vanilla only if touched; pays rebate if never touched
            payoff = np.where(touched, vanilla_payoff, self.rebate)

        return payoff

    def description(self) -> str:
        return (
            f"Barrier {self.option_type.capitalize()} ({self.barrier_type}) | S={self.S}, K={self.K}, B={self.barrier}, T={self.T}y, r={self.r:.2%}, sigma={self.sigma:.2%}, q={self.q:.2%}"
        )


    def complementary_type(self) -> str:
        """
        Return the barrier_type of the complementary option such that they follows the in-out parity.
        """
        if self.barrier_type.endswith("out"):
            return self.barrier_type.replace("out", "in")
        return self.barrier_type.replace("in", "out")

    def analytical_price(self) -> float:
        """
        barrier_price(this) + barrier_price(complementary) == analytical_price()
        """
        from src.engines.analytical import BSMModel
        return float(
            BSMModel(self.S, self.K, self.T, self.r, self.sigma, self.q).price(self.option_type)
        )