"""
Implied volatility extraction from market option chain data.

Takes a raw option chain and produces a clean table of implied volatilities. Handles data quality issues common in real option chain data like wide bid-ask spreads, stale quotes, arbitrage violations, and failed IV inversions.

See README.md — Theory for the full methodology and data-cleaning rationale.

Usage
-----
    from src.utils.data_loader import MarketDataLoader
    from src.calibration.implied_vol import ImpliedVolExtractor

    loader = MarketDataLoader(ticker="SPY", fred_api_key="...")
    chain = loader.get_option_chain()
    spot = loader.get_spot_price()
    r = loader.get_risk_free_rate(maturity=1.0)
    q = loader.get_dividend_yield()

    extractor = ImpliedVolExtractor(spot=spot, r=r, q=q)
    iv_table  = extractor.extract(chain)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engines.analytical import BSMModel


class ImpliedVolExtractor:
    """
    Extract implied volatilities from a market option chain DataFrame.

    Parameters
    ----------
    spot : Current underlying spot price.
    r : Continuously compounded risk-free rate.
    q : Continuous dividend yield.
    use_mid : bool
        If True (default), use the bid-ask midpoint as the market price.
        If False, use lastPrice.
    max_spread_pct : float
        Filter out quotes where (ask-bid)/mid exceeds this fraction.
        Default 0.50 (50%) — wide spreads indicate illiquid, unreliable quotes.
    min_price : float
        Filter out quotes priced below this (near-worthless options have unstable / noisy implied vols). Default 0.05.
    """

    def __init__(
        self,
        spot: float,
        r: float,
        q: float = 0.0,
        use_mid: bool = True,
        max_spread_pct: float = 0.50,
        min_price: float = 0.05,
    ):
        self.spot = float(spot)
        self.r = float(r)
        self.q = float(q)
        self.use_mid = use_mid
        self.max_spread_pct = max_spread_pct
        self.min_price = min_price


    def extract(self, chain: pd.DataFrame) -> pd.DataFrame:
        """
        Extract implied vols from a raw option chain DataFrame.

        Expected input columns (matches MarketDataLoader.get_option_chain()):
            strike, expiration, T, optionType, bid, ask, mid, lastPrice, volume, openInterest, impliedVolatility (yfinance's own estimate)
        """
        df = chain.copy()
        n_initial = len(df)

        # Select market price 
        if self.use_mid and "mid" in df.columns:
            df["market_price"] = df["mid"]
        else:
            df["market_price"] = df["lastPrice"]

        # Data quality filters 
        df = df[df["market_price"] >= self.min_price]
        n_after_price = len(df)

        if "bid" in df.columns and "ask" in df.columns:
            spread_pct = (df["ask"] - df["bid"]) / df["market_price"].replace(0, np.nan)
            df = df[spread_pct <= self.max_spread_pct] # filter out wide spread
        n_after_spread = len(df)

        # No-arbitrage bound pre-check (intrinsic <= price <= upper bound)
        df = df[df.apply(self._within_noarb_bounds, axis=1)]
        n_after_arb = len(df)

        # Compute implied volatility row-by-row 
        ivs = np.full(len(df), np.nan)
        for i, (_, row) in enumerate(df.iterrows()):
            opt_type = "call" if row["optionType"] == "call" else "put"
            try:
                iv = BSMModel.implied_vol(
                    market_price=row["market_price"], S=self.spot, K=row["strike"], T=row["T"], r=self.r, q=self.q, option_type=opt_type,
                )
                ivs[i] = iv
            except ValueError:
                ivs[i] = np.nan

        df = df.copy()
        df["iv_model"] = ivs
        df = df.dropna(subset=["iv_model"])
        n_final = len(df)

        # Derived columns for surface construction 
        df["moneyness"] = df["strike"] / self.spot
        df["log_moneyness"] = np.log(df["moneyness"])
        df["option_type"] = df["optionType"]

        if "impliedVolatility" in df.columns:
            df["iv_source"] = df["impliedVolatility"]

        cols = ["strike", "T", "expiration", "option_type", "market_price", "iv_model", "moneyness", "log_moneyness"]
        if "iv_source" in df.columns:
            cols.append("iv_source")
        if "volume" in df.columns:
            cols.append("volume")
        if "openInterest" in df.columns:
            cols.append("openInterest")

        result = df[cols].reset_index(drop=True)
        result.attrs["n_initial"] = n_initial
        result.attrs["n_after_price"] = n_after_price
        result.attrs["n_after_spread"] = n_after_spread
        result.attrs["n_after_arb"] = n_after_arb
        result.attrs["n_final"] = n_final
        result.attrs["filter_summary"]  = (
            f"{n_initial} -> {n_after_price} (price) -> {n_after_spread} "
            f"(spread) -> {n_after_arb} (no-arb) -> {n_final} (IV converged)"
        )
        return result


    def _within_noarb_bounds(self, row) -> bool:
        """Check market price is within no-arbitrage bounds before inverting."""
        K, T = row["strike"], row["T"]
        price = row["market_price"]
        disc_r = np.exp(-self.r * T)
        disc_q = np.exp(-self.q * T)

        if row["optionType"] == "call":
            lower = max(self.spot * disc_q - K * disc_r, 0)
            upper = self.spot * disc_q
        else:
            lower = max(K * disc_r - self.spot * disc_q, 0)
            upper = K * disc_r

        return lower - 1e-6 <= price <= upper + 1e-6


    def extract_smile(
        self,
        chain: pd.DataFrame,
        expiry: str,
    ) -> pd.DataFrame:
        """
        Extract implied vol smile for a single expiry date.
        """
        full = self.extract(chain)
        target = pd.Timestamp(expiry)
        smile = full[full["expiration"] == target].sort_values("strike")
        return smile.reset_index(drop=True)