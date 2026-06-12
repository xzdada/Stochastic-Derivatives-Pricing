"""
Monte Carlo pricing engine.

Design
------
MCEngine is deliberately depends only on two abstractions:

  model: any object with a .simulate(T, n_paths, n_steps, rng) method that returns an ndarray of shape (n_paths, n_steps + 1)

  product: any Option subclass with a .payoff(paths) method that returns an ndarray of shape (n_paths,)

This means any combination of (model * product) can be priced without changing the engine — GBM + European, Heston + Asian, etc.

Random number management (two-layer design)
-------------------------------------------
Layer 1 — instance seed: set at construction time, used as the default for every price() call. Ensures reproducibility across a session without requiring the caller to track seeds.

Layer 2 — call-time seed: an optional seed (or rng) passed directly to price(). When provided, it overrides the instance seed for that single call only. 

Usage
-----
    from src.models.equity.gbm import GBMModel
    from src.products.european import EuropeanOption
    from src.engines.monte_carlo import MCEngine

    product = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    model = GBMModel(S0=100, r=0.05, sigma=0.20)

    engine = MCEngine(model=model, product=product, n_paths=100_000, seed=926)
    result = engine.price()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from src.products.base_option import Option


# ---------------------------------------------------------------------------
# Protocol, defines the interface any stochastic model must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class StochasticModel(Protocol):
    """
    Protocol (structural interface) for stochastic process simulators.

    Any class implementing simulate() with this signature is compatible with MCEngine, regardless of inheritance.
    """

    def simulate(
        self,
        T: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate price/rate paths.

        Parameters
        ----------
        T : time horizon in years
        n_paths : number of Monte Carlo paths
        n_steps : number of time steps per path
        rng : numpy random Generator

        Returns
        -------
        ndarray, shape (n_paths, n_steps + 1)
        """
        ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """
    Container for a single Monte Carlo pricing result.

    Attributes
    ----------
    price : float
        Discounted Monte Carlo price estimate.
    stderr : float
        Standard error of the mean: std(payoffs) / sqrt(n_paths).
    ci_low, ci_high : float
        95% confidence interval bounds: price ± 1.96 * stderr.
    n_paths : int
        Number of paths used.
    n_steps : int
        Number of time steps per path.
    method : str
        Label describing the variance reduction method used.
    """
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int
    method: str = "standard"

    def __str__(self) -> str:
        return (f"MCResult | method={self.method} | price={self.price:.6f} | stderr={self.stderr:.6f} | 95% CI=({self.ci_low:.6f}, {self.ci_high:.6f}) | n_paths={self.n_paths:,}")

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class MCEngine:
    """
    Model-independent Monte Carlo pricing engine.

    Parameters
    ----------
    model : StochasticModel
        Any object implementing simulate(T, n_paths, n_steps, rng).
    product : Option
        Any Option subclass implementing payoff.
    n_paths : int
        Default number of simulation paths.
    n_steps : int
        Default number of time steps per path.
        For European options under GBM, n_steps=1 is exact. 
        For path-dependent products, use n_steps=252 (daily) or higher.
    seed : int, optional
        Instance-level default seed. 926
    confidence : float
        Confidence level for the interval. Default 0.95.
    """

    def __init__(
        self,
        model: StochasticModel,
        product: Option,
        n_paths: int = 100_000,
        n_steps: int = 1,
        seed: int | None = 926,
        confidence: float = 0.95,
    ):
        self.model = model
        self.product = product
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.confidence = confidence

        self._z_score = self._confidence_to_z(confidence)

    # ------------------------------------------------------------------
    # Main pricing methods
    # ------------------------------------------------------------------

    def price(
        self,
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = "default",
        rng: np.random.Generator | None = None,
    ) -> MCResult:
        """
        Price the product using standard Monte Carlo.

        Parameters
        ----------
        n_paths : int, optional
            Override the instance default for this call.
        n_steps : int, optional
            Override the instance default for this call.
        seed : int or None, optional
        rng : np.random.Generator, optional
            Provide a pre-built RNG. Takes priority over seed.

        Returns
        -------
        MCResult
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        rng = self._resolve_rng(seed, rng)

        paths = self.model.simulate(T=self.product.T, n_paths=n_paths, n_steps=n_steps, rng=rng)
        payoffs = self.product.payoff(paths)
        return self._summarise(payoffs, n_paths, n_steps, method="standard")

    def price_antithetic(
        self,
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = "default",
        rng: np.random.Generator | None = None,
    ) -> MCResult:
        """
        Price using antithetic variates variance reduction.

        Simulates n_paths pairs of (Z, -Z) paths. The payoff estimate for each pair is averaged before discounting:

            payoff_i = (payoff(Z_i) + payoff(-Z_i)) / 2

        This halves the variance when the payoff is a monotone function of the terminal price, as for vanilla calls and puts, at the cost of zero extra model evaluations.

        Total paths simulated: 2 * n_paths.
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        rng = self._resolve_rng(seed, rng)

        # Model must expose simulate_antithetic
        if not hasattr(self.model, "simulate_antithetic"):
            raise NotImplementedError(f"{type(self.model).__name__} does not implement simulate_antithetic(). Use standard price() instead.")

        paths, paths_anti = self.model.simulate_antithetic(
            T=self.product.T,
            n_paths=n_paths,
            n_steps=n_steps,
            rng=rng,
        )
        payoffs = self.product.payoff(paths)
        payoffs_anti = self.product.payoff(paths_anti)
        paired = (payoffs + payoffs_anti) / 2.0

        return self._summarise(
            paired, n_paths, n_steps, method="antithetic",
            note=f"(total paths simulated: {2 * n_paths:,})"
        )

    def price_control_variate(
        self,
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = "default",
        rng: np.random.Generator | None = None,
    ) -> MCResult:
        """
        Price using control variate variance reduction.

        Uses the BSM analytical price as the control variate. 
        For a European option under GBM this gives a large variance reduction because the BSM payoff is highly correlated with the simulated payoffs.

        Method
        ------
        Let Y  = simulated discounted payoff
            X  = discounted payoff of the control (BSM)
            μ_X = known analytical price of the control

        Controlled estimator:
            Y_hat = Y - beta *(X - μ_X)

        Optimal beta_hat = Cov(Y, X) / Var(X)

        The variance of Y_hat is: Var(Y) * (1 - rho_{XY}**2)
        where rho_{XY} is the correlation between the MC and control payoffs.

        Requirement
        -----------
        product must have an analytical_price() method.
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        rng     = self._resolve_rng(seed, rng)

        if not hasattr(self.product, "analytical_price"):
            raise NotImplementedError(f"{type(self.product).__name__} does not implement analytical_price(). "
                                      f"Control variate requires a known closed-form price as the control.")

        mu_X = self.product.analytical_price()

        paths   = self.model.simulate(
            T=self.product.T,
            n_paths=n_paths,
            n_steps=n_steps,
            rng=rng,
        )

        # Simulated payoffs
        disc = self.product.discount_factor()
        Y = disc * self.product.payoff(paths)

        # Control payoffs from the same paths (terminal price only)
        S_T  = paths[:, -1]
        if self.product.option_type == "call":
            X_raw = np.maximum(S_T - self.product.K, 0.0)
        else:
            X_raw = np.maximum(self.product.K - S_T, 0.0)
        X = disc * X_raw 

        # Optimal beta
        cov_matrix = np.cov(Y, X)
        beta_star = cov_matrix[0, 1] / cov_matrix[1, 1]

        # Controlled payoffs
        Y_star = Y - beta_star * (X - mu_X)

        # Correlation
        corr = np.corrcoef(Y, X)[0, 1]

        result = self._summarise(Y_star, n_paths, n_steps, method="control_variate", pre_discounted=True)
        result.method = (
            f"control_variate (beta_hat={beta_star:.4f}, rho={corr:.4f}, var_reduction={(1 - corr**2)*100:.1f}%)")
        return result

    # ------------------------------------------------------------------
    # Convergence study
    # ------------------------------------------------------------------

    def convergence_study(
        self,
        path_counts: list[int] | None = None,
        n_steps: int | None = None,
        seed: int | None = "default",
        methods: list[str] | None = None,
    ) -> list[dict]:
        """
        Run price() across increasing path counts to study convergence.

        Parameters
        ----------
        path_counts : list of int 
        methods : list of str
            Subset of {"standard", "antithetic", "control_variate"}.

        Returns
        -------
        list of dict
            Each dict has keys: n_paths, method, price, stderr, ci_low, ci_high.
        """
        if path_counts is None:
            path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
        if methods is None:
            methods = ["standard", "antithetic", "control_variate"]

        n_steps = n_steps or self.n_steps
        results = []

        for n in path_counts:
            for method in methods:
                try:
                    if method == "standard":
                        r = self.price(n_paths=n, n_steps=n_steps, seed=seed)
                    elif method == "antithetic":
                        r = self.price_antithetic(n_paths=n, n_steps=n_steps, seed=seed)
                    elif method == "control_variate":
                        r = self.price_control_variate(n_paths=n, n_steps=n_steps, seed=seed)
                    else:
                        continue
                    results.append({
                        "n_paths": n,
                        "method": method,
                        "price": r.price,
                        "stderr": r.stderr,
                        "ci_low": r.ci_low,
                        "ci_high": r.ci_high,
                    })
                except NotImplementedError:
                    pass

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_rng(
        self,
        seed: int | None | str,
        rng: np.random.Generator | None,
    ) -> np.random.Generator:
        """
        Resolve the RNG to use for a price() call.

        Priority:
          1. Explicit rng object (highest)
          2. Call-time seed (overrides instance seed)
          3. Instance seed (default 926)
        """
        if rng is not None:
            return rng
        if seed == "default":
            return np.random.default_rng(self.seed)
        return np.random.default_rng(seed)

    def _summarise(
        self,
        payoffs: np.ndarray,
        n_paths: int,
        n_steps: int,
        method: str = "standard",
        note: str = "",
        pre_discounted: bool = False,
    ) -> MCResult:
        """
        Compute price, stderr, and CI from a vector of payoffs.

        Parameters
        ----------
        pre_discounted : bool
            If True, payoffs are already discounted (e.g. control variate builds discounted quantities internally). 
            If False (default), apply e^{-rT} discounting here.
        """
        if pre_discounted:
            discounted = payoffs
        else:
            discounted = self.product.discount_factor() * payoffs

        mean = float(np.mean(discounted))
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
        ci_low = mean - self._z_score * stderr
        ci_high = mean + self._z_score * stderr

        return MCResult(
            price = mean,
            stderr = stderr,
            ci_low = ci_low,
            ci_high = ci_high,
            n_paths = n_paths,
            n_steps = n_steps,
            method = method + (f" {note}" if note else ""),
        )

    @staticmethod
    def _confidence_to_z(confidence: float) -> float:
        from scipy.stats import norm
        return float(norm.ppf((1 + confidence) / 2))