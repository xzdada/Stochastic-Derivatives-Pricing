"""
European call and put option.

European option payoff depends only on the terminal price S_T:
    Call payoff: max(S_T - K, 0)
    Put payoff: max(K - S_T, 0)

This product will serve as the benchmark for validating all numerical engines against the BSM analytical price.
"""

from __future__ import annotations

import numpy as np

from src.products.base_option import Option


class EuropeanOption(Option):
    """
    European vanilla call / put option.

    Payoff is determined solely by the terminal asset price S_T. No early exercise. No path dependency.

    Parameters
    ----------
    See base_option.Option for full parameter documentation.
    """

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute terminal payoff from simulated paths.

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Only the final column S_T is used.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Undiscounted payoff: max(S_T - K, 0) for call, max(K - S_T, 0) for put.
        """
        S_T = paths[:, -1]
        if self.option_type == "call":
            return np.maximum(S_T - self.K, 0)
        else:
            return np.maximum(self.K - S_T, 0)

    def description(self) -> str:
        return (f"European {self.option_type.capitalize()} | S={self.S}, K={self.K}, T={self.T}y, r={self.r:.2%}, sigma={self.sigma:.2%}, q={self.q:.2%}")

    def analytical_price(self) -> float:
        """
        Return the exact BSM price as a benchmark.
        """
        from src.engines.analytical import BSMModel
        return float(
            BSMModel(self.S, self.K, self.T, self.r, self.sigma, self.q).price(self.option_type))