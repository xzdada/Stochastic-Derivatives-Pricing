"""
Unified Greeks calculator supporting all pricing models.

Two computation methods:
  1. Analytical  — exact BSM closed-form derivatives (fast, zero error). Available when the product is European and the model is BSM/GBM.
  2. Bump-and-reprice — central finite differences via any pricing engine. Works for any model * product combination (American, Heston, etc.).

See README.md — Theory — Greeks for definitions and sign conventions.

Usage
-----
    from src.risk.greeks import GreeksCalculator
    from src.products.european import EuropeanOption
    from src.engines.analytical import BSMModel

    opt = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.013)

    # Analytical (BSM)
    calc = GreeksCalculator(opt)
    g = calc.analytical()

    # Bump-and-reprice via any callable pricer
    pricer = lambda o: BSMModel(o.S, o.K, o.T, o.r, o.sigma, o.q).price(o.option_type)
    g = calc.bump_and_reprice(pricer)

    # Greeks surface across strikes
    surface = calc.greeks_surface(pricer, S_range=(80, 120), n_S=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.products.base_option import Option

@dataclass
class GreeksResult:
    delta: float
    gamma: float
    vega:  float
    theta: float
    rho: float
    vanna: float | None = None
    volga: float | None = None
    charm: float | None = None
    method: str = "unknown"
    option_type: str = "call"

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if v is not None and k not in ("method", "option_type")}

    def __str__(self) -> str:
        lines = [
            f"GreeksResult ({self.method}, {self.option_type})",
            f" delta : {self.delta:+.6f}",
            f" gamma : {self.gamma:+.6f}",
            f" vega : {self.vega:+.6f}",
            f" theta : {self.theta:+.6f}  (per day)",
            f" rho : {self.rho:+.6f}",
        ]
        if self.vanna is not None:
            lines.append(f"  vanna : {self.vanna:+.6f}")
        if self.volga is not None:
            lines.append(f"  volga : {self.volga:+.6f}")
        if self.charm is not None:
            lines.append(f"  charm : {self.charm:+.6f}  (per day)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class GreeksCalculator:
    """
    Unified Greeks calculator for any option product and pricing model.
    """

    def __init__(self, product: Option):
        self.product = product

    # ------------------------------------------------------------------
    # Method 1: Analytical (BSM closed-form)
    # ------------------------------------------------------------------

    def analytical(self) -> GreeksResult:
        """
        Compute all Greeks analytically using the BSM closed-form formulas.

        Includes higher-order Greeks (vanna, volga, charm).

        Available only for European options where BSM applies.
        For American or exotic options, use bump_and_reprice() instead.
        """
        from src.engines.analytical import BSMModel
        p = self.product
        m = BSMModel(p.S, p.K, p.T, p.r, p.sigma, p.q)
        g = m.all_greeks(p.option_type)
        return GreeksResult(
            delta = float(g["delta"]),
            gamma = float(g["gamma"]),
            vega = float(g["vega"]),
            theta = float(g["theta"]),
            rho = float(g["rho"]),
            vanna = float(g["vanna"]),
            volga = float(g["volga"]),
            charm = float(g["charm"]),
            method = "analytical",
            option_type = p.option_type,
        )

    # ------------------------------------------------------------------
    # Method 2: Bump-and-reprice 
    # ------------------------------------------------------------------

    def bump_and_reprice(
        self,
        pricer: callable,
        dS_pct: float = 0.01,
        dsigma_pct: float = 0.01,
        dr: float = 1e-4,
        dt_days: float = 1.0,
    ) -> GreeksResult:

        p = self.product
        S = p.S
        sig = p.sigma
        r = p.r
        T = p.T

        dS = S * dS_pct
        dsig = sig * dsigma_pct
        dt = dt_days / 365.0

        V0 = pricer(p)

        def _p(**kw) -> float:
            return float(pricer(_clone(p, **kw)))

        delta = (_p(S=S + dS) - _p(S=S - dS)) / (2 * dS)

        gamma = (_p(S=S + dS) - 2 * V0 + _p(S=S - dS)) / dS ** 2

        vega = (_p(sigma=sig + dsig) - _p(sigma=sig - dsig)) / (2 * dsig)

        T_bumped = max(T - dt, 1e-6)
        theta = (_p(T=T_bumped) - V0) / dt / 365.0   # per calendar day

        rho = (_p(r=r + dr) - _p(r=r - dr)) / (2 * dr)

        return GreeksResult(
            delta = float(delta),
            gamma = float(gamma),
            vega = float(vega),
            theta = float(theta),
            rho = float(rho),
            method = "bump_and_reprice",
            option_type = p.option_type,
        )

    def greeks_surface(
        self,
        pricer: callable,
        S_range: tuple[float, float] | None = None,
        n_S: int = 20,
        T_range: tuple[float, float] | None = None,
        n_T: int = 10,
        method: str = "analytical",
    ) -> dict:
        """
        Compute a 2D Greeks surface across a range of spot prices and/or maturities.
        maturities (one axis at a time or both combined).
        """
        p = self.product
        S_lo, S_hi = S_range or (0.5 * p.S, 1.5 * p.S)
        S_grid = np.linspace(S_lo, S_hi, n_S)

        if T_range is not None:
            T_grid = np.linspace(T_range[0], T_range[1], n_T)
        else:
            T_grid = np.array([p.T])

        shape = (len(T_grid), len(S_grid))
        surface = {name: np.empty(shape)
                   for name in ("delta", "gamma", "vega", "theta", "rho")}

        for i, T in enumerate(T_grid):
            for j, S in enumerate(S_grid):
                prod_ij = _clone(p, S=S, T=T)
                calc_ij = GreeksCalculator(prod_ij)
                if method == "analytical":
                    g = calc_ij.analytical()
                else:
                    g = calc_ij.bump_and_reprice(pricer)
                for name in surface:
                    surface[name][i, j] = getattr(g, name)

        surface["S_grid"] = S_grid
        surface["T_grid"] = T_grid
        return surface


    def compare_methods(self, pricer: callable) -> dict:
        """
        Compute Greeks by both methods and return side-by-side comparison.
        """
        g_ana  = self.analytical()
        g_bump = self.bump_and_reprice(pricer)

        greek_names = ("delta", "gamma", "vega", "theta", "rho")
        diffs = {
            name: abs(getattr(g_ana, name) - getattr(g_bump, name))
            for name in greek_names
        }
        return {
            "analytical": g_ana,
            "bump_and_reprice": g_bump,
            "abs_diff": diffs,
            "max_abs_diff": max(diffs.values()),
        }


def _clone(product: Option, **overrides) -> Option:
    """Return a new option instance with selected parameters overridden."""
    params = {
        "S": product.S,
        "K": product.K,
        "T": product.T,
        "r": product.r,
        "sigma": product.sigma,
        "q": product.q,
        "option_type": product.option_type,
    }
    params.update(overrides)
    return product.__class__(**params)