"""
Black-Scholes-Merton (BSM) analytical pricing engine.

Implements the Merton continuous-dividend extension of BSM.

All inputs and outputs are in consistent units:
  - Prices / strikes in the same currency unit (absolute)
  - Rates and volatility as annualized decimals (e.g. 0.05 for 5%)
  - Time in years

Mathematical foundation
-----------------------

Under risk-neutral measure Q, the stock follows:
    dS = (r - q) S dt + sigma S dW_t

Call price:  C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put  price:  P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

where:
    d1 = [ln(S/K) + (r - q + σ²/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

All Greeks are derived analytically from these formulas.

Usage
-----
    from src.engines.analytical import BSMModel

    bsm = BSMModel(S=500, K=500, T=1.0, r=0.05, sigma=0.20, q=0.013)

    price  = bsm.price("call")
    greeks = bsm.all_greeks("call")

    # Vectorized over strikes
    bsm_vec = BSMModel(S=500, K=np.arange(450, 551, 5), T=1.0, r=0.05, sigma=0.20)
    prices = bsm_vec.price("call")

    # Implied volatility
    iv = BSMModel.implied_vol(market_price=15.0, S=500, K=500, T=1.0, r=0.05, q=0.013, option_type="call")
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from scipy.stats import norm

# Type alias
ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# BSM Model
# ---------------------------------------------------------------------------

class BSMModel:
    """
    Black-Scholes-Merton analytical engine with continuous dividend yield.

    Parameters
    ----------
    S : float or array-like
        Current spot price of the underlying.
    K : float or array-like
        Strike price.
    T : float or array-like
        Time to expiry in years, Must be > 0.
    r : float
        Continuously compounded risk-free rate (annualized decimal).
    sigma : float or array-like
        Volatility of the underlying (annualized decimal). Must be > 0.
    q : float
        Continuous dividend yield (annualized decimal). Default 0.
    """

    def __init__(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: float,
        sigma: ArrayLike,
        q: float = 0.0,
    ):
        self.S = np.asarray(S, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.r = float(r)
        self.sigma = np.asarray(sigma, dtype=float)
        self.q = float(q)
        self._validate()

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate(self) -> None:
        if np.any(self.S <= 0):
            raise ValueError("Spot price S must be positive.")
        if np.any(self.K <= 0):
            raise ValueError("Strike K must be positive.")
        if np.any(self.T <= 0):
            raise ValueError("Time to expiry T must be positive.")
        if np.any(self.sigma <= 0):
            raise ValueError("Volatility sigma must be positive.")
        if self.q < 0:
            raise ValueError("Dividend yield q must be non-negative.")

    # -----------------------------------------------------------------------
    # Core d1 / d2
    # -----------------------------------------------------------------------

    def _d1_d2(self) -> tuple[np.ndarray, np.ndarray]:
 
        sqrt_T = np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    # -----------------------------------------------------------------------
    # Price
    # -----------------------------------------------------------------------

    def price(self, option_type: str = "call") -> np.ndarray:
        """
        Compute the BSM option price.
        """
        d1, d2 = self._d1_d2()
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)

        if option_type.lower() == "call":
            price = (self.S * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2))

        elif option_type.lower() == "put":
            price = (self.K * disc_r * norm.cdf(-d2) - self.S * disc_q * norm.cdf(-d1))

        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")

        return _scalar_or_array(price)

    # -----------------------------------------------------------------------
    # Greeks
    # -----------------------------------------------------------------------

    def delta(self, option_type: str = "call") -> np.ndarray:
        """
        Sensitivity of the option price to a $1 move in the spot price.

        Call delta \in (0, 1)  |  Put delta \in (-1, 0)
        """
        d1, _ = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        if option_type.lower() == "call":
            delta = disc_q * norm.cdf(d1)
        elif option_type.lower() == "put":
            delta = disc_q * (norm.cdf(d1) - 1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")
        return _scalar_or_array(delta)

    def gamma(self) -> np.ndarray:
        """
        Rate of change of delta per $1 move in spot. Identical for calls and puts.
        Measures convexity — large gamma means delta changes rapidly.
        """
        d1, _ = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        gamma = (disc_q * norm.pdf(d1)) / (self.S * self.sigma * np.sqrt(self.T))
        return _scalar_or_array(gamma)

    def vega(self) -> np.ndarray:
        """
        Sensitivity to a 1-unit (100%) change in volatility. Identical for calls and puts.

        In practice, vega is often quoted per 1% move in vol (divide by 100).
        """
        d1, _ = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        vega = self.S * disc_q * norm.pdf(d1) * np.sqrt(self.T)
        return _scalar_or_array(vega)

    def theta(self, option_type: str = "call", annualized: bool = False) -> np.ndarray:
        """
        Sensitivity to the passage of time. Typically negative because options lose value as expiry approaches. Returned as change per calendar day by default (divide by 365); 
        Set annualized = True for raw theta.
        """
        d1, d2 = self._d1_d2()
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)
        sqrt_T = np.sqrt(self.T)

        common = (-(self.S * disc_q * norm.pdf(d1) * self.sigma) / (2 * sqrt_T))

        if option_type.lower() == "call":
            theta_ann = (
                common
                - self.r * self.K * disc_r * norm.cdf(d2)
                + self.q * self.S * disc_q * norm.cdf(d1)
            )
        elif option_type.lower() == "put":
            theta_ann = (
                common
                + self.r * self.K * disc_r * norm.cdf(-d2)
                - self.q * self.S * disc_q * norm.cdf(-d1)
            )
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")

        theta_out = theta_ann if annualized else theta_ann / 365.0
        return _scalar_or_array(theta_out)

    def rho(self, option_type: str = "call") -> np.ndarray:
        """
        Sensitivity to a 1-unit (100%) change in the risk-free rate.
        In practice often quoted per 1 basis point (divide by 10000).
        """
        _, d2 = self._d1_d2()
        disc_r = np.exp(-self.r * self.T)

        if option_type.lower() == "call":
            rho = self.K * self.T * disc_r * norm.cdf(d2)
        elif option_type.lower() == "put":
            rho = -self.K * self.T * disc_r * norm.cdf(-d2)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")
        return _scalar_or_array(rho)

    def vanna(self) -> np.ndarray:
        """
        Cross-Greek between delta and vega. Useful for hedging vol-spot correlation risk. Identical for calls and puts.
        """
        d1, d2 = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        vanna = -disc_q * norm.pdf(d1) * d2 / self.sigma
        return _scalar_or_array(vanna)

    def volga(self) -> np.ndarray:
        """
        Rate of change of vega with respect to volatility. Measures the convexity of the option price in volatility space. Identical for calls and puts.
        """
        d1, d2 = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        volga = self.S * disc_q * norm.pdf(d1) * np.sqrt(self.T) * d1 * d2 / self.sigma
        return _scalar_or_array(volga)

    def charm(self, option_type: str = "call") -> np.ndarray:
        """
        Rate of change of delta over time. Important for overnight delta hedgers. Returned per calendar day.
        """
        d1, d2 = self._d1_d2()
        disc_q = np.exp(-self.q * self.T)
        sqrt_T = np.sqrt(self.T)

        pdf_d1 = norm.pdf(d1)
        term = (2 * (self.r - self.q) * self.T - d2 * self.sigma * sqrt_T) / (2 * self.T * self.sigma * sqrt_T)

        if option_type.lower() == "call":
            charm = -disc_q * (pdf_d1 * term - self.q * norm.cdf(d1))
        elif option_type.lower() == "put":
            charm = -disc_q * (pdf_d1 * term + self.q * norm.cdf(-d1))
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")

        return _scalar_or_array(charm / 365.0)  # per calendar day

    # -----------------------------------------------------------------------
    # All Greeks in one call
    # -----------------------------------------------------------------------

    def all_greeks(self, option_type: str = "call") -> dict:
        """
        Return all Greeks in a single dictionary.
        """
        return {
            "delta": self.delta(option_type),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(option_type),
            "rho": self.rho(option_type),
            "vanna": self.vanna(),
            "volga": self.volga(),
            "charm": self.charm(option_type),
        }

    # -----------------------------------------------------------------------
    # Put-call parity check
    # -----------------------------------------------------------------------

    def put_call_parity_check(self, tol: float = 1e-8) -> bool:
        """
        Verify put-call parity: C - P = S e^{-qT} - K e^{-rT}

        Returns True if parity holds within tolerance.
        """
        C = self.price("call")
        P = self.price("put")
        lhs = C - P
        rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        holds = bool(np.all(np.abs(lhs - rhs) < tol))
        if not holds:
            max_err = float(np.max(np.abs(lhs - rhs)))
            warnings.warn(f"Put-call parity violated. Max error: {max_err:.2e}")
        return holds

    # -----------------------------------------------------------------------
    # Implied volatility (class method)
    # -----------------------------------------------------------------------

    @classmethod
    def implied_vol(
        cls,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
        method: str = "newton",
        tol: float = 1e-8,
        max_iter: int = 200,
        sigma_init: float = 0.20,
        sigma_bounds: tuple[float, float] = (1e-6, 10.0),
    ) -> float:
        """
        Invert the BSM formula to recover implied volatility.

        Tries Newton-Raphson first (fast, quadratic convergence near ATM).
        Falls back to Brent's method (robust, guaranteed convergence) if Newton diverges or the vega is too small.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        S, K, T, r, q : float
            Same as BSMModel constructor. All scalars.
        option_type : {"call", "put"}
        method : {"newton", "brent"}
            Force a specific solver. Default "newton" with brent fallback.
        tol : float
            Convergence tolerance on |BSM(sigma) - market_price|.
        max_iter : int
            Maximum iterations for Newton-Raphson.
        sigma_init : float
            Starting guess for Newton-Raphson (default 20%).
        sigma_bounds : tuple
            (lower, upper) bound for Brent's method.
        """
        # No-arbitrage bounds check 
        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q * T)

        intrinsic_call = max(S * disc_q - K * disc_r, 0.0)
        intrinsic_put = max(K * disc_r - S * disc_q, 0.0)
        upper_call = S * disc_q
        upper_put = K * disc_r

        if option_type.lower() == "call":
            lower = intrinsic_call
            upper = upper_call
        else:
            lower = intrinsic_put
            upper = upper_put

        if market_price < lower - tol:
            raise ValueError(f"market_price {market_price:.4f} below intrinsic value {lower:.4f}.")
        if market_price > upper + tol:
            raise ValueError(f"market_price {market_price:.4f} above theoretical upper bound {upper:.4f}.")

        # If price essentially equals intrinsic, vol is 0
        if market_price <= lower + tol:
            return 0.0

        def bsm_price(sigma: float) -> float:
            return float(cls(S, K, T, r, sigma, q).price(option_type))

        def bsm_vega(sigma: float) -> float:
            return float(cls(S, K, T, r, sigma, q).vega())

        # Newton-Raphson
        if method in ("newton", "auto"):
            sigma = sigma_init
            for _ in range(max_iter):
                price_err = bsm_price(sigma) - market_price
                v = bsm_vega(sigma)
                if abs(v) < 1e-12:
                    break  # vega too small, fall back to Brent
                sigma_new = sigma - price_err / v
                # Keep sigma in valid range
                sigma_new = np.clip(sigma_new, sigma_bounds[0], sigma_bounds[1])
                if abs(sigma_new - sigma) < tol:
                    return float(sigma_new)
                sigma = sigma_new

        # Brent's method fallback
        from scipy.optimize import brentq
        try:
            lo, hi = sigma_bounds
            f_lo = bsm_price(lo) - market_price
            f_hi = bsm_price(hi) - market_price
            if f_lo * f_hi > 0:
                return np.nan
            sigma_brent = brentq(
                lambda s: bsm_price(s) - market_price,
                lo, hi,
                xtol=tol,
                maxiter=max_iter,
            )
            return float(sigma_brent)
        except Exception:
            return np.nan

    @classmethod
    def implied_vol_vectorized(
        cls,
        market_prices: np.ndarray,
        S: float,
        strikes: np.ndarray,
        T: float,
        r: float,
        q: float = 0.0,
        option_types: Union[str, list] = "call",
        **iv_kwargs, # Additional keyword arguments to pass to implied_vol
    ) -> np.ndarray:
        """
        Compute implied volatility for a vector of strikes and market prices.

        Parameters
        ----------
        market_prices : array of float
            Market prices aligned with strikes.
        strikes : array of float
            Strike prices.
        option_types : str or list of str
            Either a single string applied to all, or one per strike.

        Returns
        -------
        np.ndarray
            Implied volatilities, np.nan where no solution found.
        """
        strikes = np.asarray(strikes, dtype=float)
        market_prices = np.asarray(market_prices, dtype=float)
        n = len(strikes)

        if isinstance(option_types, str):
            option_types = [option_types] * n

        ivs = np.empty(n)
        for i in range(n):
            try:
                ivs[i] = cls.implied_vol(
                    market_price=market_prices[i],
                    S=S, K=strikes[i], T=T, r=r, q=q,
                    option_type=option_types[i],
                    **iv_kwargs,
                )
            except (ValueError, Exception):
                ivs[i] = np.nan
        return ivs

def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> np.ndarray:
    """Instantiate BSMModel and return price."""
    return BSMModel(S, K, T, r, sigma, q).price(option_type)


def bsm_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> dict:
    """Instantiate BSMModel and return all Greeks."""
    return BSMModel(S, K, T, r, sigma, q).all_greeks(option_type)


def _scalar_or_array(x: np.ndarray) -> Union[float, np.ndarray]:
    """
    Return a Python float if x is a 0-d array, else return the array.
    """
    if x.ndim == 0:
        return float(x)
    return x