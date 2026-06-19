"""
Full implied volatility surface: strike * maturity grid with interpolation.

A continuous implied vol surface from discrete market quotes using two complementary techniques: 
  1. Stochastic Volatility Inspired parameterization per maturity slice — a 5-parameter functional form that fits the smile/skew shape at each expiry while enforcing no static arbitrage within the slice.
  2. 2D interpolation across maturities — linear or cubic interpolation of SVI-fitted (or raw) implied vols across the time dimension.

See README.md — Theory  for the full SVI formula, fitting methodology, and arbitrage considerations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import griddata


# ---------------------------------------------------------------------------
# SVI slice fit result
# ---------------------------------------------------------------------------

@dataclass
class SVISlice:
    """
    SVI parameters for a single maturity slice.

    Total variance parameterization:

        w(k) = a + b * ( rho*(k-m) + sqrt{(k-m)^2 + sigma^2} )

    where k = log-moneyness = ln(K/F), w(k) = sigma_BS(k)^2 * T.
    """
    T: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse: float
    n_points: int

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Evaluate SVI total variance w(k) at log-moneyness k."""
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """Evaluate Black-Scholes implied vol at log-moneyness k."""
        w = np.maximum(self.total_variance(k), 1e-10)
        return np.sqrt(w / self.T)


# ---------------------------------------------------------------------------
# Vol Surface
# ---------------------------------------------------------------------------

class VolSurface:
    """
    Implied volatility surface: SVI fit per maturity + cross-maturity interpolation.
    """

    def __init__(self, spot: float):
        self.spot   = float(spot)
        self.slices: dict[float, SVISlice] = {}
        self._raw_data: pd.DataFrame | None = None


    def fit(
        self,
        iv_table: pd.DataFrame,
        min_points_per_slice: int = 5,
    ) -> dict[float, SVISlice]:
        """
        Fit an SVI slice to each unique maturity in the IV table.

        Parameters
        ----------
        iv_table : pd.DataFrame
            Columns: T, log_moneyness (or strike), iv_model.
            The output of ImpliedVolExtractor.extract().
        min_points_per_slice : int
            Skip maturities with fewer than this many quotes.
        """

        self._raw_data = iv_table.copy()
        self.slices = {}

        for T in sorted(iv_table["T"].unique()):
            sl = iv_table[iv_table["T"] == T]
            if len(sl) < min_points_per_slice:
                continue

            k = sl["log_moneyness"].values if "log_moneyness" in sl.columns \
                else np.log(sl["strike"].values / self.spot)
            iv = sl["iv_model"].values

            svi_slice = self._fit_svi_slice(T, k, iv)
            self.slices[T] = svi_slice

        return self.slices

    def _fit_svi_slice(
        self,
        T: float,
        k: np.ndarray,
        iv: np.ndarray,
    ) -> SVISlice:
        """
        Fit SVI raw parameters (a, b, rho, m, sigma) by least squares on total variance w(k) = sigma_BS^2 * T.
        """
        w_market = (iv ** 2) * T

        def loss(params):
            a, b, rho, m, sigma = params
            w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
            return np.sum((w_model - w_market) ** 2)

        # Initial guess
        a0 = np.percentile(w_market, 10)
        b0 = 0.1
        rho0 = -0.3
        m0 = k[np.argmin(np.abs(iv - np.median(iv)))]
        sigma0 = 0.1

        bounds = [
            (1e-6, np.max(w_market) * 2),  # min variance a must be small positive
            (1e-4, 5.0), # slope b must be positive
            (-0.999, 0.999), # correlation-like rho in (-1,1)
            (k.min() - 1.0, k.max() + 1.0), # shift m near the data range
            (1e-4, 5.0) # sigma must be positive
        ]

        result = minimize(
            loss, x0=[a0, b0, rho0, m0, sigma0],
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-14}
        )
        a, b, rho, m, sigma = result.x
        rmse = float(np.sqrt(result.fun / len(k)))

        return SVISlice(T=T, a=a, b=b, rho=rho, m=m, sigma=sigma, rmse=rmse, n_points=len(k))

    # ------------------------------------------------------------------
    # Querying the surface
    # ------------------------------------------------------------------

    def smile(self, T: float, k_range: tuple[float, float] = (-0.5, 0.5), n: int = 100) -> pd.DataFrame:
        """
        Return the fitted SVI smile at a given maturity T.
        """
        if T not in self.slices:
            raise ValueError(f"No fitted slice at T={T}. Available: {list(self.slices.keys())}")

        sl = self.slices[T]
        k  = np.linspace(k_range[0], k_range[1], n)
        iv = sl.implied_vol(k)
        strikes = self.spot * np.exp(k)

        return pd.DataFrame({"log_moneyness": k, "strike": strikes, "iv": iv})

    def interpolate(self, K: float, T: float) -> float:
        """
        Interpolate implied vol at an arbitrary (strike, maturity) point.
        """
        if not self.slices:
            raise RuntimeError("No slices fitted. Call fit() first.")

        mats = np.array(sorted(self.slices.keys()))
        k = np.log(K / self.spot)

        if T <= mats[0]:
            return float(self.slices[mats[0]].implied_vol(np.array([k]))[0])
        if T >= mats[-1]:
            return float(self.slices[mats[-1]].implied_vol(np.array([k]))[0])

        # Break T between two fitted maturities
        idx_hi = np.searchsorted(mats, T)
        T_lo, T_hi = mats[idx_hi - 1], mats[idx_hi]

        w_lo = self.slices[T_lo].total_variance(np.array([k]))[0]
        w_hi = self.slices[T_hi].total_variance(np.array([k]))[0]

        # Linear interpolation in total variance across maturity
        weight = (T - T_lo) / (T_hi - T_lo)
        w_interp = w_lo + weight * (w_hi - w_lo)

        return float(np.sqrt(max(w_interp, 1e-10) / T))

    def grid(
        self,
        K_range: tuple[float, float],
        T_range: tuple[float, float],
        n_K: int = 30,
        n_T: int = 30,
    ) -> dict:
        """
        Evaluate the full volatility surface on a regular (K, T) grid.
        """
        K_grid = np.linspace(K_range[0], K_range[1], n_K)
        T_grid = np.linspace(T_range[0], T_range[1], n_T)

        iv_grid = np.empty((n_T, n_K))
        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                iv_grid[i, j] = self.interpolate(K, T)

        return {"K_grid": K_grid, "T_grid": T_grid, "iv_grid": iv_grid}


    def fit_quality(self) -> pd.DataFrame:
        """
        Check SVI fit quality per maturity.
        """

        rows = []
        for T, sl in sorted(self.slices.items()):
            rows.append({
                "T": T, "n_points": sl.n_points, "rmse": sl.rmse, "a": sl.a, "b": sl.b, "rho": sl.rho, "m": sl.m, "sigma": sl.sigma
            })
        return pd.DataFrame(rows)

    def atm_term_structure(self) -> pd.DataFrame:
        """
        Return the at-the-money (K=spot, k=0) implied vol term structure.
        """
        rows = []
        for T, sl in sorted(self.slices.items()):
            atm_iv = float(sl.implied_vol(np.array([0.0]))[0])
            rows.append({"T": T, "atm_iv": atm_iv})
        return pd.DataFrame(rows)

    def skew(self, T: float, k_offset: float = 0.1) -> float:
        """
        Compute the implied vol skew at maturity T:
            skew = iv(k = -offset) - iv(k = +offset)

        Positive skew (puts > calls) is typical for equity indices.
        """
        if T not in self.slices:
            raise ValueError(f"No fitted slice at T={T}.")
        sl = self.slices[T]
        iv_put = float(sl.implied_vol(np.array([-k_offset]))[0])
        iv_call = float(sl.implied_vol(np.array([+k_offset]))[0])
        return iv_put - iv_call