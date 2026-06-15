"""
Longstaff-Schwartz least-squares Monte Carlo for American options.

Mathematical foundation
-----------------------
American option pricing requires solving an optimal stopping problem:

    V(t, S) = sup_{τ ∈ [t,T]} E^Q[ e^{-r(τ-t)} Payoff(S_τ) | S_t ]

where τ is a stopping time (exercise decision).

LSM Algorithm (Longstaff & Schwartz)
------------------------------------------
1. Simulate N paths of the underlying S over M time steps.

2. At expiry T: exercise value = intrinsic payoff (no choice).

3. Work backwards from t_{M-1} to t_1:
   At each step t_k, for paths that are in-the-money (ITM):
   a. Regress the discounted future cash flows onto basis functions of S_k
   b. Compare the regression estimate (continuation value) with the immediate exercise value:
      - Exercise if intrinsic > continuation (update cash flow to now)
      - Otherwise continue

4. The LSM price is the average discounted cash flow across all paths.

Why ITM only for regression?
    Regressing on all paths includes deep OTM options where the exercise decision is trivially "continue". Including them adds noise without improving the estimate of the exercise boundary.

Bias properties
    LSM produces a low-biased estimate of the American price 
    (the exercise policy found by regression is sub-optimal relative to the true policy).
    The binomial tree and finite difference methods are used to cross-validate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.products.base_option import Option
from src.models.equity.gbm import GBMModel

@dataclass
class LSMResult:
    """
    Container for Longstaff-Schwartz pricing result.

    Attributes
    ----------
    price : float
        LSM price estimate (low-biased).
    stderr : float
        Standard error of the mean across paths.
    ci_low, ci_high : float
        95% confidence interval.
    n_paths : int
    n_steps : int
    early_exercise_premium : float or None
        V_American - V_European (approximated using BSM for European).
    """
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int
    early_exercise_premium: float | None = None

    def __str__(self) -> str:
        eep = (f" | EEP={self.early_exercise_premium:.4f}"
               if self.early_exercise_premium is not None else "")
        return (
            f"LSMResult | price={self.price:.6f} | "
            f"stderr={self.stderr:.6f} | "
            f"95% CI=({self.ci_low:.6f}, {self.ci_high:.6f}) | "
            f"n_paths={self.n_paths:,} | n_steps={self.n_steps}{eep}"
        )

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# LSM Engine
# ---------------------------------------------------------------------------

class LSMEngine:
    """
    Longstaff-Schwartz least-squares Monte Carlo for American options.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of exercise opportunities (time steps).
    seed : int, optional
        Default RNG seed for reproducibility.
    poly_degree : int
        Degree of polynomial basis for regression.
    """

    def __init__(
        self,
        n_paths: int = 100000,
        n_steps: int = 50,
        seed: int | None = 926,
        poly_degree: int = 2,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.poly_degree = poly_degree

    # ------------------------------------------------------------------
    # Pricing method
    # ------------------------------------------------------------------

    def price(
        self,
        product: Option,
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = "default",
        rng: np.random.Generator | None = None,
    ) -> LSMResult:
        """
        Price an American option using the LSM algorithm.
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps

        if rng is None:
            s = self.seed if seed == "default" else seed
            rng = np.random.default_rng(s)

        # Simulate full paths
        model = GBMModel(
            S0=product.S, r=product.r, sigma=product.sigma, q=product.q,
        )
        paths = model.simulate(T=product.T, n_paths=n_paths, n_steps=n_steps, rng=rng)

        dt = product.T / n_steps
        disc = np.exp(-product.r * dt)

        # Terminal payoffs
        cash_flows = self._intrinsic(product, paths[:, -1])

        # Backward induction
        american = _is_american(product)

        for k in range(n_steps - 1, 0, -1):
            cash_flows = cash_flows * disc

            if not american:
                # European: no early exercise
                continue

            S_k = paths[:, k]
            intrinsic = self._intrinsic(product, S_k)
            itm = intrinsic > 0

            if itm.sum() < self.poly_degree + 1:
                # Too few ITM paths to regress
                continue

            # Regression:
            Y = cash_flows[itm]
            X = self._basis(S_k[itm])
            try:
                coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
                continuation = X @ coeffs
            except np.linalg.LinAlgError:
                continue

            # Early exercise decision for ITM paths
            exercise_now = intrinsic[itm] >= continuation
            idx_itm = np.where(itm)[0]
            cash_flows[idx_itm[exercise_now]] = intrinsic[itm][exercise_now]

        # Discount back to t_0
        cash_flows = cash_flows * disc

        # Summary statistics 
        mean = float(np.mean(cash_flows))
        stderr = float(np.std(cash_flows, ddof=1) / np.sqrt(n_paths))
        z = 1.96
        ci_low = mean - z * stderr
        ci_hi = mean + z * stderr

        # Early exercise premium vs BSM European
        from src.engines.analytical import BSMModel
        eu_price = float(
            BSMModel(product.S, product.K, product.T, product.r, product.sigma, product.q).price(product.option_type)
        )
        eep = mean - eu_price

        return LSMResult(
            price=mean,
            stderr=stderr,
            ci_low=ci_low,
            ci_high=ci_hi,
            n_paths=n_paths,
            n_steps=n_steps,
            early_exercise_premium=eep,
        )

    def _intrinsic(self, product: Option, S: np.ndarray) -> np.ndarray:
        """Return intrinsic (immediate exercise) value for spot prices S."""
        S = np.asarray(S, dtype=float)
        if product.option_type == "call":
            return np.maximum(S - product.K, 0.0)
        else:
            return np.maximum(product.K - S, 0.0)

    def _basis(self, S: np.ndarray) -> np.ndarray:
        """
        Build polynomial basis matrix for regression.

        Uses normalised spot (S / mean(S)) for numerical stability.
        """
        S_norm = S / np.mean(S)
        return np.column_stack(
            [S_norm ** i for i in range(self.poly_degree + 1)]
        )


def _is_american(product) -> bool:
    from src.products.american import AmericanOption
    return isinstance(product, AmericanOption)