"""
Asian (average-rate) options.

Two variants:
  - Average price: payoff depends on the average of the path replacing the terminal price S_T.
        Call: max(A - K, 0)        Put: max(K - A, 0)
  - Average strike: payoff uses the average of the path as a floating strike, compared against the terminal price S_T.
        Call: max(S_T - A, 0)      Put: max(A - S_T, 0)

where A is the arithmetic average of the simulated path.

Asian options are path-dependent, and have no general closed-form solution under GBM (the arithmetic average of log-normal variables is not log-normal), so
they are priced via Monte Carlo. Control variates using the geometric average (which have a closed form) are a natural variance reduction technique.

Usage
-----
    from src.products.asian import AsianOption

    opt = AsianOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, averaging="price", option_type="call")
    # then price with MCEngine, e.g.:
    #   engine = MCEngine(model=GBMModel(...), product=opt, n_steps=252)
    #   engine.price()
"""

from __future__ import annotations

import numpy as np

from src.products.base_option import Option


class AsianOption(Option):
    """
    Asian (average-rate) call or put option.

    Parameters
    ----------
    averaging : {"price", "strike"}
        "price"  -> fixed strike K
        "strike" -> floating strike
    average_type : {"arithmetic", "geometric"}
        Arithmetic (default) is standard market convention but has no closed form. Geometric has a closed-form BSM-style price and is mainly used as a control
        variate for the arithmetic case.
    monitoring : {"continuous", "discrete"}
        "continuous" treats every step in the simulated path as an averaging point (the usual MC approximation). "discrete" can be
        extended to use only specific monitoring dates if needed.
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
        averaging: str = "price",
        average_type: str = "arithmetic",
    ):
        super().__init__(S, K, T, r, sigma, q, option_type)
        if averaging not in ("price", "strike"):
            raise ValueError("averaging must be 'price' or 'strike'.")
        if average_type not in ("arithmetic", "geometric"):
            raise ValueError("average_type must be 'arithmetic' or 'geometric'.")
        self.averaging = averaging
        self.average_type = average_type


    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the Asian option payoff from simulated paths.

        paths : np.ndarray, shape (n_paths, n_steps + 1)
            The average is computed over all columns, including S_0.
        """
        A = self._average(paths)
        S_T = paths[:, -1]

        if self.averaging == "price":
            if self.option_type == "call":
                return np.maximum(A - self.K, 0.0)
            return np.maximum(self.K - A, 0.0)
        else:
            if self.option_type == "call":
                return np.maximum(S_T - A, 0.0)
            return np.maximum(A - S_T, 0.0)

    def _average(self, paths: np.ndarray) -> np.ndarray:
        """Compute the arithmetic or geometric average along each path."""
        if self.average_type == "arithmetic":
            return np.mean(paths, axis=1)
        else:
            return np.exp(np.mean(np.log(paths), axis=1))

    def description(self) -> str:
        return (
            f"Asian {self.option_type.capitalize()} ({self.averaging}, {self.average_type}) | S={self.S}, K={self.K}, T={self.T}y, r={self.r:.2%}, sigma={self.sigma:.2%}, q={self.q:.2%}"
        )


    def analytical_price_geometric(self) -> float:
        """
        Closed-form price for a geometric-average, fixed-strike Asian option under GBM.
        The geometric average of a GBM path is itself log-normal, so a BSM-style formula applies with adjusted volatility and drift.
        """
        if self.averaging != "price":
            raise NotImplementedError(
                "Closed-form geometric Asian price is only implemented for averaging='price'."
            )

        from scipy.stats import norm

        sigma_avg = self.sigma / np.sqrt(3.0)
        b_avg = 0.5 * (self.r - self.q - self.sigma ** 2 / 6.0)

        d1 = (np.log(self.S / self.K) + (b_avg + 0.5 * sigma_avg ** 2) * self.T) / (sigma_avg * np.sqrt(self.T))
        d2 = d1 - sigma_avg * np.sqrt(self.T)

        disc_r = np.exp(-self.r * self.T)
        fwd_adj = np.exp((b_avg - self.r) * self.T)

        if self.option_type == "call":
            price = self.S * fwd_adj * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        else:
            price = self.K * disc_r * norm.cdf(-d2) - self.S * fwd_adj * norm.cdf(-d1)

        return float(price)