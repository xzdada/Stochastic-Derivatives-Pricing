"""
Binomial (CRR) and Trinomial tree pricing engines.

Mathematical foundation — CRR Binomial Tree
--------------------------------------------
Cox, Ross and Rubinstein discretise the GBM log-normal dynamics onto a recombining binomial tree.

At each time step dt = T/N, the stock price moves:
    up by factor u = exp(sigma sqrt{dt})
    down by factor d = 1/u = exp(-sigma sqrt{dt})

Risk-neutral probabilities (ensure E^Q[S_{t+dt}] = S_t · e^{(r - q)dt}):
    up move probability p = (e^{(r - q)dt} - d) / (u - d)
    down move probability 1 - p

No-arbitrage requires: d < e^{(r - q)dt} < u

Terminal stock prices at step N (recombining):
    S_{N,j} = S_0 * u^j * d^{N - j},   j = 0, 1,..., N, j is the number of up moves

Backward induction (European):
    V_{n,j} = e^{-rdt} [p * V_{n+1,j+1} + (1-p) * V_{n+1,j}]

American early exercise check:
    V_{n,j} = max( Intrinsic_{n,j},  e^{-r dt} [p*V_{n+1,j+1} + (1 - p) * V_{n+1,j}] )

The tree converges to the BSM price as N goes to infinity, with oscillating error O(1/N).

Greeks via bump-and-reprice
---------------------------
All Greeks are computed by finite differences on the tree price:
    Delta : [V(S + ε) - V(S - ε )] / (2ε)    (central difference)
    Gamma : [V(S + ε) - 2V(S) + V(S - ε)] / ε^2  (second central diff)
    Theta : [V(T - Δt) - V(T)] / Δt (forward diff (one step))
    Vega : [V(σ + ε) - V(σ - ε)] / (2ε)  (central diff)
    Rho : [V(r + ε) - V(r - ε)] / (2ε)  (central diff)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.products.base_option import Option


@dataclass
class TreeResult:
    """
    Container for a binomial tree pricing result.

    Attributes
    ----------
    price : float
        Option price from backward induction.
    option_type : str
    exercise : str
        "european" or "american".
    n_steps : int
        Number of time steps used.
    delta, gamma, vega, theta, rho : float or None
        Greeks computed via bump-and-reprice (None if not computed).
    early_exercise_premium : float or None
        V_American - V_European (None for European options).
    """
    price: float
    option_type: str
    exercise: str
    n_steps: int
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    early_exercise_premium: float | None = None

    def __str__(self) -> str:
        g = ""
        if self.delta is not None:
            g = (f" | Δ={self.delta:.4f} Γ={self.gamma:.6f} ν={self.vega:.4f} Θ={self.theta:.6f} ρ={self.rho:.4f}")
        eep = (f" | EEP={self.early_exercise_premium:.4f}"
               if self.early_exercise_premium is not None else "")
        return (f"TreeResult({self.exercise} {self.option_type}) | price={self.price:.6f} | steps={self.n_steps}{g}{eep}")

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Binomial CRR Tree
# ---------------------------------------------------------------------------

class BinomialTree:
    """
    Cox-Ross-Rubinstein (CRR) binomial tree pricer.

    Handles European and American options with continuous dividend yield.

    Parameters
    ----------
    n_steps : int
        Number of time steps. Higher N means more accurate but slower.
    """

    def __init__(self, n_steps: int = 500):
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        self.n_steps = n_steps

    def price(self, product: Option, n_steps: int | None = None) -> TreeResult:
        """
        Price a European or American option via backward induction.

        Parameters
        ----------
        product : Option
            EuropeanOption or AmericanOption instance.
        n_steps : int, optional
            Override instance n_steps for this call.
        """
        n = n_steps or self.n_steps
        val = self._price_scalar(product, n)

        exercise = "american" if _is_american(product) else "european"
        eep = None
        if exercise == "american":
            eu_val = self._price_scalar(product, n, force_european=True)
            eep = val - eu_val

        return TreeResult(
            price=val,
            option_type=product.option_type,
            exercise=exercise,
            n_steps=n,
            early_exercise_premium=eep,
        )

    def greeks(self, product: Option, n_steps: int | None = None) -> TreeResult:
        """
        Compute price and all Greeks via bump-and-reprice.

        Parameters
        ----------
        product : Option
            EuropeanOption or AmericanOption instance.

        Returns
        -------
        TreeResult with delta, gamma, vega, theta, rho.
        """
        n = n_steps or self.n_steps
        result = self.price(product, n_steps=n)

        S = product.S
        sigma = product.sigma
        r = product.r
        T = product.T

        # bump
        dS = S * 0.01      # 1% of spot
        dsig = sigma * 0.01      # 1% of vol
        dr = 0.0001           # 1 basis point
        dt = T / n        # one tree step

        def _p(s=S, sig=sigma, rate=r, t=T) -> float:
            return self._price_scalar(
                _clone(product, S=s, sigma=sig, r=rate, T=t), n
            )

        # Delta
        delta = (_p(s=S + dS) - _p(s=S - dS)) / (2 * dS)

        # Gamma
        gamma = (_p(s=S + dS) - 2 * _p() + _p(s=S - dS)) / dS ** 2

        # Vega
        vega = (_p(sig=sigma + dsig) - _p(sig=sigma - dsig)) / (2 * dsig)

        # Theta: one step shorter maturity because option losses value over time.
        # For most options Θ < 0 but American deep-ITM puts can have Θ > 0.
        t_bumped = max(T - dt, 1e-6)
        theta = (_p(t=t_bumped) - _p()) / dt / 365

        # Rho
        rho = (_p(rate=r + dr) - _p(rate=r - dr)) / (2 * dr)

        result.delta = float(delta)
        result.gamma = float(gamma)
        result.vega = float(vega)
        result.theta = float(theta)
        result.rho = float(rho)
        return result

    def convergence(
        self,
        product: Option,
        step_range=None,
    ) -> list[dict]:
        """
        Price the option across increasing step counts to check convergence.
        """
        if step_range is None:
            step_range = range(10, 301, 10)

        # European BSM benchmark
        from src.engines.analytical import BSMModel
        bsm_price = float(
            BSMModel(product.S, product.K, product.T, product.r, product.sigma, product.q).price(product.option_type))

        results = []
        for n in step_range:
            r = self.price(product, n_steps=n)
            results.append({
                "n_steps": n,
                "price": r.price,
                "bsm_price": bsm_price,
                "error": abs(r.price - bsm_price),
                "early_exercise_premium": r.early_exercise_premium,
            })
        return results

    # ------------------------------------------------------------------
    # Backward induction
    # ------------------------------------------------------------------

    def _price_scalar(
        self,
        product: Option,
        n: int,
        force_european: bool = False,
    ) -> float:
        """
        Core CRR backward induction. Returns scalar option price.

        Parameters
        ----------
        force_european : bool
            If True, skip early exercise check.
        """
        S, K, T, r, sigma, q = (
            product.S, product.K, product.T, product.r, product.sigma, product.q,
        )
        dt = T / n

        # CRR parameters
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)
        pu = (np.exp((r - q) * dt) - d) / (u - d)
        pd = 1.0 - pu

        if not (0 < pu < 1):
            raise ValueError(
                f"Risk-neutral probability p={pu:.4f} out of (0,1)."
            )

        # Terminal stock prices: S_0 * u^j * d^(n-j)
        j = np.arange(n + 1, dtype=float)
        S_T = S * (u ** j) * (d ** (n - j))

        # Terminal option values
        if product.option_type == "call":
            V = np.maximum(S_T - K, 0.0)
        else:
            V = np.maximum(K - S_T, 0.0)

        american = _is_american(product) and not force_european

        # Backward induction
        for i in range(n - 1, -1, -1):
            # Continuation values (discounted expected value)
            V = disc * (pu * V[1:] + pd * V[:-1])

            if american:
                # Stock prices at this node
                j_i = np.arange(i + 1, dtype=float)
                S_i = S * (u ** j_i) * (d ** (i - j_i))
                if product.option_type == "call":
                    intrinsic = np.maximum(S_i - K, 0.0)
                else:
                    intrinsic = np.maximum(K - S_i, 0.0)
                V = np.maximum(V, intrinsic)

        return float(V[0])

def _is_american(product: Option) -> bool:
    """Return True if product is an AmericanOption."""
    from src.products.american import AmericanOption
    return isinstance(product, AmericanOption)


def _clone(product: Option, **overrides) -> Option:
    """
    Return a new option instance with selected parameters overridden.
    Used for bump-and-reprice Greeks.
    """
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