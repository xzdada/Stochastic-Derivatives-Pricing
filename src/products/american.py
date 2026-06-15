"""
American call and put option.

Payoff is identical to European at expiry, but the holder has the right to exercise early at any time at or before expiration.

Early exercise premium
----------------------
The American price always satisfies: V_American >= V_European

For calls on non-dividend-paying stock: early exercise is never optimal,
so V_American_call = V_European_call (BSM applies exactly).

For puts (and calls on dividend-paying stock): early exercise can be optimal when the option is sufficiently deep ITM. The early exercise
premium is the difference V_American - V_European > 0.

The optimal exercise boundary S*(t) separates the continuation region (hold the option) from the stopping region (exercise immediately).

Pricing methods
---------------
All three numerical methods in this project are used and compared:
  - Binomial CRR tree: src/engines/trees.py
  - Longstaff-Schwartz MC: src/engines/lsm.py
  - Crank-Nicolson finite diff: src/engines/finite_diff.py
"""

from __future__ import annotations

import numpy as np

from src.products.base_option import Option


class AmericanOption(Option):
    """
    American vanilla call or put option.

    Identical contract parameters to EuropeanOption. The difference is how the pricing engine handles early exercise — the payoff
    method here returns the immediate exercise value at any node/step, which the engines compare against the continuation value.
    """

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Immediate exercise payoff at terminal date.

        For intermediate steps, engines call intrinsic_value() directly.
        """
        return self.intrinsic_value(paths[:, -1])

    def intrinsic_value(self, S: np.ndarray) -> np.ndarray:
        """
        Immediate exercise (intrinsic) value at spot price S.

        Returns
        -------
        ndarray
            max(S - K, 0) for call, max(K - S, 0) for put.
        """
        S = np.asarray(S, dtype=float)
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        else:
            return np.maximum(self.K - S, 0.0)

    def description(self) -> str:
        return (
            f"American {self.option_type.capitalize()} | S={self.S}, K={self.K}, T={self.T}y, r={self.r:.2%}, σ={self.sigma:.2%}, q={self.q:.2%}"
        )

    def european_price(self) -> float:
        """
        Return the European BSM price as a lower bound benchmark.

        The American price must always be >= this value.
        """
        from src.engines.analytical import BSMModel
        return float(
            BSMModel(self.S, self.K, self.T, self.r, self.sigma, self.q).price(self.option_type)
        )