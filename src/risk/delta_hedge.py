"""
Delta, Gamma, and Vega hedging simulation with PnL attribution.

Simulates discrete dynamic hedging of an option position over its lifetime, decomposing the resulting PnL into its Greek components.

See README.md — Theory for the full derivation of the PnL attribution formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.products.base_option import Option
from src.engines.analytical import BSMModel
from src.models.equity.gbm import GBMModel

@dataclass
class HedgingResult:

    total_pnl: np.ndarray
    delta_pnl: np.ndarray
    gamma_pnl: np.ndarray
    vega_pnl: np.ndarray
    theta_pnl: np.ndarray
    hedge_error: np.ndarray
    rebalance_freq: str
    n_steps: int
    n_paths: int

    def summary(self) -> dict:
        """Return mean and std of each PnL component."""
        components = ("total_pnl", "delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "hedge_error")

        return {
            c: {"mean": float(np.mean(getattr(self, c))), "std":  float(np.std(getattr(self, c)))}
            for c in components
        }

    def __str__(self) -> str:
        s = self.summary()
        lines = [f"HedgingResult | freq={self.rebalance_freq} | steps={self.n_steps} | paths={self.n_paths:,}", f"{'Component':<14} {'Mean PnL':>10} {'Std PnL':>10} {'-'*36}" ]
        for name, label in [
            ("total_pnl", "Total"),
            ("delta_pnl", "Delta"),
            ("gamma_pnl", "Gamma"),
            ("vega_pnl", "Vega"),
            ("theta_pnl", "Theta"),
            ("hedge_error", "Residual"),
        ]:
            lines.append(
                f"  {label:<14} {s[name]['mean']:>+10.4f} {s[name]['std']:>10.4f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class HedgingSimulator:
    """
    Discrete dynamic hedging simulator.

    Simulates hedging an option position by rebalancing a portfolio of the underlying and risk-free bond at fixed intervals.
    Decomposes the resulting PnL into:
        total_pnl ≈ delta_pnl + gamma_pnl + vega_pnl + theta_pnl + residual
    """

    FREQ_MAP = {"daily": 252, "weekly": 52, "monthly": 12}

    def __init__(
        self,
        product: Option,
        rebalance_freq: str | int = "daily",
        n_paths: int = 10_000,
        seed: int | None = 42,
    ):
        self.product = product
        self.n_paths = n_paths
        self.seed    = seed

        if isinstance(rebalance_freq, str):
            if rebalance_freq not in self.FREQ_MAP:
                raise ValueError(f"rebalance_freq must be one of {list(self.FREQ_MAP)} or an int.")
            annual_steps = self.FREQ_MAP[rebalance_freq]
            self.n_steps = max(1, int(self.product.T * annual_steps))
            self.rebalance_freq = rebalance_freq
        else:
            self.n_steps = int(rebalance_freq)
            self.rebalance_freq = f"{rebalance_freq}_steps"


    def run(
        self,
        seed: int | None = "default",
    ) -> HedgingResult:

        if seed == "default":
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng(seed)

        p  = self.product
        N  = self.n_steps
        dt = p.T / N

        # Simulate GBM paths
        model = GBMModel(S0=p.S, r=p.r, sigma=p.sigma, q=p.q)
        paths = model.simulate(T=p.T, n_paths=self.n_paths, n_steps=N, rng=rng)

        # Accumulate PnL components across steps
        total_pnl = np.zeros(self.n_paths)
        delta_pnl = np.zeros(self.n_paths)
        gamma_pnl = np.zeros(self.n_paths)
        vega_pnl = np.zeros(self.n_paths)
        theta_pnl  = np.zeros(self.n_paths)

        for k in range(N):
            S_k = paths[:, k]
            S_k1 = paths[:, k + 1]
            dS   = S_k1 - S_k
            t_k  = k * dt
            T_rem = p.T - t_k 

            if T_rem <= 1e-8:
                break

            # Greeks at (S_k, T_rem) via BSM
            bsm_k = BSMModel(
                S=1.0, K=p.K, T=T_rem, r=p.r, sigma=p.sigma, q=p.q
            )
            # Vectorised over S_k
            d1, d2 = bsm_k._d1_d2_vec(S_k)
            delta_k = _delta_vec(d1, p.option_type, p.q, T_rem)
            gamma_k = _gamma_vec(S_k, d1, p.sigma, T_rem, p.q)
            vega_k  = _vega_vec(S_k, d1, T_rem, p.q)
            theta_k = _theta_vec(S_k, d1, d2, p.K, p.r, p.q, p.sigma,T_rem, p.option_type)

            disc = np.exp(-p.r * dt)

            # PnL attribution per step (Taylor expansion of dV)
            dp = delta_k * dS  
            gp = 0.5 * gamma_k * dS ** 2 
            vp = np.zeros(self.n_paths)
            tp = theta_k * dt 

            delta_pnl += disc ** k * dp
            gamma_pnl += disc ** k * gp
            vega_pnl += disc ** k * vp
            theta_pnl += disc ** k * tp

        # Total PnL
        option_value_0 = float(
            BSMModel(p.S, p.K, p.T, p.r, p.sigma, p.q).price(p.option_type)
        )
        payoff_T = p.payoff(paths) 
        disc_T = np.exp(-p.r * p.T)
        total_pnl = disc_T * payoff_T - option_value_0

        # Hedging error: difference between total PnL and sum of attribution terms
        hedge_error = total_pnl - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)

        return HedgingResult(
            total_pnl = total_pnl,
            delta_pnl = delta_pnl,
            gamma_pnl = gamma_pnl,
            vega_pnl = vega_pnl,
            theta_pnl = theta_pnl,
            hedge_error = hedge_error,
            rebalance_freq = self.rebalance_freq,
            n_steps = N,
            n_paths = self.n_paths,
        )

    def convergence(
        self,
        freq_list: list | None = None,
        seed: int = 42,
    ) -> list[dict]:

        if freq_list is None:
            freq_list = ["monthly", "weekly", "daily"]
        results = []
        for freq in freq_list:
            sim = HedgingSimulator(
                product=self.product,
                rebalance_freq=freq,
                n_paths=self.n_paths,
                seed=seed,
            )
            r = sim.run()
            s = r.summary()
            results.append({
                "freq": freq,
                "n_steps": r.n_steps,
                "hedge_error_mean": s["hedge_error"]["mean"],
                "hedge_error_std": s["hedge_error"]["std"],
                "total_pnl_std": s["total_pnl"]["std"],
            })
        return results


# ---------------------------------------------------------------------------
# Vectorised BSM helpers
# ---------------------------------------------------------------------------

def _d1_d2_vec(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _delta_vec(d1, option_type, q, T):
    from scipy.stats import norm
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1)


def _gamma_vec(S, d1, sigma, T, q):
    from scipy.stats import norm
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _vega_vec(S, d1, T, q):
    from scipy.stats import norm
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def _theta_vec(S, d1, d2, K, r, q, sigma, T, option_type):
    from scipy.stats import norm
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    sqrt_T = np.sqrt(T)
    common = -(S * disc_q * norm.pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == "call":
        return (common - r * K * disc_r * norm.cdf(d2)
                + q * S * disc_q * norm.cdf(d1))
    return (common + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1))


def _bsm_d1_d2_vec(self, S_vec):
    """Compute d1, d2 for a vector of spot prices S_vec."""
    sqrt_T = np.sqrt(self.T)
    d1 = (np.log(S_vec / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
    d2 = d1 - self.sigma * sqrt_T
    return d1, d2


BSMModel._d1_d2_vec = _bsm_d1_d2_vec