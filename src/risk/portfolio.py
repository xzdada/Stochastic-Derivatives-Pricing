"""
Portfolio risk measures: VaR and CVaR (Expected Shortfall).

Implements three VaR/CVaR estimation methods:
  1. Historical simulation  — non-parametric, uses empirical return distribution
  2. Parametric (Gaussian)  — assumes normally distributed returns
  3. Monte Carlo simulation — simulates forward P&L under GBM

See README.md — Theory for definitions, assumptions, and limitations of each method.

Usage
-----
    from src.risk.portfolio import PortfolioRisk
    import numpy as np

    returns = np.random.normal(0, 0.01, 1000)   # daily log returns
    risk = PortfolioRisk(returns, confidence=0.95, horizon=1)

    var = risk.var_historical()
    cvar = risk.cvar_historical()
    print(risk.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class RiskResult:
    """
    Attributes
    ----------
    var : float
        Value at Risk — maximum loss at given confidence level. Reported as a positive number (loss convention).
    cvar : float
        Conditional VaR (Expected Shortfall) — expected loss given that the loss exceeds VaR. Always >= VaR.
    confidence : float
        Confidence level used (e.g. 0.95).
    horizon : int
        Risk horizon in days.
    method : str
    n_obs : int
        Number of paths used.
    """
    var: float
    cvar: float
    confidence: float
    horizon: int
    method: str
    n_obs: int

    def __str__(self) -> str:
        return (
            f"RiskResult ({self.method}) | conf={self.confidence:.0%} | horizon={self.horizon}d | "
            f"VaR={self.var:.4f} | CVaR={self.cvar:.4f} | n={self.n_obs:,}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class PortfolioRisk:
    """
    VaR and CVaR calculator for a single asset or portfolio.
    """

    def __init__(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
        portfolio_value: float = 1.0,
    ):
        self.returns = np.asarray(returns, dtype=float)
        self.confidence = confidence
        self.horizon = horizon
        self.portfolio_value = portfolio_value

        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1).")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

    # ------------------------------------------------------------------
    # Method 1: Historical Simulation
    # ------------------------------------------------------------------

    def var_historical(self) -> float:
        """
        Historical simulation VaR.

        Sort observed P&L and find the (1-confidence) quantile.
        No distributional assumption. Captures fat tails and skewness.
        """
        scaled = self._scale_returns()
        var = float(-np.quantile(scaled, 1 - self.confidence))
        return var * self.portfolio_value

    def cvar_historical(self) -> float:
        """
        Historical simulation CVaR (Expected Shortfall).

        Average of losses beyond the VaR threshold.
        """
        scaled = self._scale_returns()
        threshold = np.quantile(scaled, 1 - self.confidence)
        tail_losses = scaled[scaled <= threshold]
        cvar = float(-np.mean(tail_losses))
        return cvar * self.portfolio_value

    def historical(self) -> RiskResult:
        """Return full RiskResult for historical simulation method."""
        return RiskResult(
            var = self.var_historical(),
            cvar = self.cvar_historical(),
            confidence = self.confidence,
            horizon = self.horizon,
            method = "historical",
            n_obs = len(self.returns),
        )

    # ------------------------------------------------------------------
    # Method 2: Parametric (Gaussian)
    # ------------------------------------------------------------------

    def var_parametric(self) -> float:
        """
        Parametric VaR assuming normally distributed returns.

            VaR = -(mu - z_alpha * sigma) * sqrt(horizon)

        where z_alpha = norm.ppf(1 - confidence).

        Underestimates tail risk for fat-tailed return distributions.
        """
        mu, sigma = float(np.mean(self.returns)), float(np.std(self.returns, ddof=1))
        z_alpha = stats.norm.ppf(1 - self.confidence)
        var_daily = -(mu + z_alpha * sigma)
        var_scaled = var_daily * np.sqrt(self.horizon)
        return max(var_scaled * self.portfolio_value, 0.0)

    def cvar_parametric(self) -> float:
        """
        Parametric CVaR (closed form under Gaussian assumption).

            CVaR = -(mu - sigma * phi(z_alpha) / (1 - confidence)) * sqrt(horizon)

        where phi is the standard normal PDF.

        """
        mu, sigma = float(np.mean(self.returns)), float(np.std(self.returns, ddof=1))
        z_alpha = stats.norm.ppf(1 - self.confidence)
        phi_z = stats.norm.pdf(z_alpha)
        cvar_daily = -(mu - sigma * phi_z / (1 - self.confidence))
        cvar_scaled = cvar_daily * np.sqrt(self.horizon)
        return max(cvar_scaled * self.portfolio_value, 0.0)

    def parametric(self) -> RiskResult:
        """Return full RiskResult for parametric (Gaussian) method."""
        return RiskResult(
            var = self.var_parametric(),
            cvar = self.cvar_parametric(),
            confidence = self.confidence,
            horizon = self.horizon,
            method = "parametric",
            n_obs = len(self.returns),
        )

    # ------------------------------------------------------------------
    # Method 3: Monte Carlo
    # ------------------------------------------------------------------

    def var_montecarlo(
        self,
        S0: float,
        mu: float | None = None,
        sigma: float | None = None,
        n_sims: int = 100_000,
        seed: int = 42,
    ) -> float:
        """
        Monte Carlo VaR — simulate forward P&L under GBM.

        Parameters
        ----------
        S0 : float
            Current asset price.
        mu : float, optional
            Drift (annualised). Defaults to mean(returns) * 252.
        sigma : float, optional
            Volatility (annualised). Defaults to std(returns) * sqrt(252).
        n_sims : int
            Number of simulated paths.
        seed : int

        Returns
        -------
        float  (positive = loss)
        """
        mu_ann = mu    or float(np.mean(self.returns) * 252)
        sigma_ann = sigma or float(np.std(self.returns, ddof=1) * np.sqrt(252))
        T_horizon = self.horizon / 252.0

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n_sims)
        S_T = S0 * np.exp((mu_ann - 0.5 * sigma_ann ** 2) * T_horizon + sigma_ann * np.sqrt(T_horizon) * Z)
        pnl = (S_T - S0) * self.portfolio_value / S0 
        var = float(-np.quantile(pnl, 1 - self.confidence))
        return var

    def cvar_montecarlo(
        self,
        S0: float,
        mu: float | None = None,
        sigma: float | None = None,
        n_sims: int = 100_000,
        seed: int = 42,
    ) -> float:
        """Monte Carlo CVaR — average loss in the tail beyond MC VaR."""
        mu_ann = mu    or float(np.mean(self.returns) * 252)
        sigma_ann = sigma or float(np.std(self.returns, ddof=1) * np.sqrt(252))
        T_horizon = self.horizon / 252.0

        rng  = np.random.default_rng(seed)
        Z = rng.standard_normal(n_sims)
        S_T  = S0 * np.exp((mu_ann - 0.5 * sigma_ann ** 2) * T_horizon + sigma_ann * np.sqrt(T_horizon) * Z)

        pnl = (S_T - S0) * self.portfolio_value / S0
        threshold = np.quantile(pnl, 1 - self.confidence)
        tail = pnl[pnl <= threshold]
        return float(-np.mean(tail))

    def montecarlo(
        self,
        S0: float,
        n_sims: int = 100_000,
        seed: int = 42,
    ) -> RiskResult:
        """Return full RiskResult for Monte Carlo method."""
        return RiskResult(
            var = self.var_montecarlo(S0, n_sims=n_sims, seed=seed),
            cvar = self.cvar_montecarlo(S0, n_sims=n_sims, seed=seed),
            confidence = self.confidence,
            horizon = self.horizon,
            method = "monte_carlo",
            n_obs = n_sims,
        )


    def summary(self, S0: float | None = None) -> str:
        """
        Print a comparison table of all three VaR/CVaR methods.
        """
        r_hist  = self.historical()
        r_param = self.parametric()

        lines = [
            f"Portfolio Risk Summary | conf={self.confidence:.0%} | "
            f"horizon={self.horizon}d | portfolio_value={self.portfolio_value:,.2f}",
            f" {'Method':<20} {'VaR':>10} {'CVaR':>10}",
            f" {'-'*42}",
            f" {'Historical':<20} {r_hist.var:>10.4f} {r_hist.cvar:>10.4f}",
            f" {'Parametric (Gaussian)':<20} {r_param.var:>10.4f} {r_param.cvar:>10.4f}",
        ]

        if S0 is not None:
            r_mc = self.montecarlo(S0)
            lines.append(
                f"  {'Monte Carlo':<20} {r_mc.var:>10.4f} {r_mc.cvar:>10.4f}"
            )

        lines += [
            f"  {'-'*42}",
            f"  n_obs = {len(self.returns):,}",
            f"  Note: CVaR >= VaR, and CVaR is a coherent risk measure.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    def stress_test(
        self,
        scenarios: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply instantaneous stress scenarios to the portfolio.
        """
        return {
            name: shock * self.portfolio_value
            for name, shock in scenarios.items()
        }

    def _scale_returns(self) -> np.ndarray:
        """Scale daily returns to the requested horizon (square-root rule)."""
        if self.horizon == 1:
            return self.returns
        return self.returns * np.sqrt(self.horizon)