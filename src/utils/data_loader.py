"""
Unified market data loader.

Provides three main data sources:
  - Equity prices : yfinance  (spot, historical OHLCV, dividends)
  - Options chains : yfinance  (calls + puts across all expiries)
  - Yield curve / rates: FRED API (US Treasury constant-maturity rates)

FRED API Key is required. 

Usage
-----
    from src.utils.data_loader import MarketDataLoader

    loader = MarketDataLoader(ticker="SPY", fred_api_key="FRED_API_KEY_HERE")

    # Equity
    spot = loader.get_spot_price()
    hist = loader.get_price_history(period="2y")
    rets = loader.get_log_returns(period="1y")
    hist_vol = loader.get_historical_vol(window=21)

    # Options
    chain = loader.get_option_chain() # all expiries
    chain1 = loader.get_option_chain(expiry="YYYY-MM-DD")  # single expiry

    # Rates
    curve = loader.get_yield_curve() # full term structure
    r = loader.get_risk_free_rate(maturity=1.0) # interpolated scalar

    # Save / load
    loader.save(path="data/processed/spx_data.pkl")
    loader2 = MarketDataLoader.load("data/processed/spx_data.pkl", fred_api_key="FRED_API_KEY_HERE")
"""

from __future__ import annotations

import pickle
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# FRED series IDs for US Treasury constant-maturity yields
_FRED_SERIES: dict[str, str] = {
    "1m": "DGS1MO",
    "3m": "DGS3MO",
    "6m": "DGS6MO",
    "1y": "DGS1",
    "2y": "DGS2",
    "3y": "DGS3",
    "5y": "DGS5",
    "7y": "DGS7",
    "10y": "DGS10",
    "20y": "DGS20",
    "30y": "DGS30",
}

# Corresponding maturities in years
_FRED_MATURITIES: dict[str, float] = {
    "1m": 1/12, "3m": 3/12, "6m": 6/12,
    "1y": 1.0,  "2y": 2.0,  "3y": 3.0,
    "5y": 5.0,  "7y": 7.0,  "10y": 10.0,
    "20y": 20.0, "30y": 30.0,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketDataLoader:
    """
    Central data access object for one equity underlying.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "SPY", "AAPL", "^SPX".
    fred_api_key : str
    cache_dir : str or Path, optional
        Directory to cache downloaded data to avoid repeated API calls.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        fred_api_key: str = "",
        cache_dir: Optional[str | Path] = None,
    ):
        if not fred_api_key:
            raise ValueError("fred_api_key is required. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")

        self.ticker = ticker.upper()
        self.fred_api_key = fred_api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._yf_ticker = yf.Ticker(self.ticker)
        self._fred = None

        # In-memory caches
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._chain_cache: dict[str, pd.DataFrame] = {}
        self._curve_cache: Optional[pd.Series] = None

    # -----------------------------------------------------------------------
    # EQUITY — SPOT & HISTORY
    # -----------------------------------------------------------------------

    def get_spot_price(self) -> float:
        """
        Return the most recent closing price.
        """
        hist = self._yf_ticker.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data returned for {self.ticker}.")
        return float(hist["Close"].iloc[-1])

    def get_price_history(
        self,
        period: str = "2y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download Open, High, Low, Close, and Volume price history.

        Parameters
        ----------
        period : str
            yfinance period string: "1y", "2y", "5y", "max".
            Ignored if both start and end are provided.
        start, end : str, optional
            Date strings "YYYY-MM-DD". Override period when both given.
        interval : str
            Data frequency: "1d", "1wk", "1mo".

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume, Dividends, Stock Splits.
            Index: DatetimeIndex (UTC).
        """
        cache_key = f"{period}_{start}_{end}_{interval}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        kwargs: dict = {"interval": interval, "auto_adjust": True}
        if start and end:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = period

        hist = self._yf_ticker.history(**kwargs)
        if hist.empty:
            raise ValueError(f"No price history for {self.ticker} with {kwargs}.")

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        self._price_cache[cache_key] = hist
        return hist

    def get_log_returns(
        self,
        period: str = "2y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        
        hist = self.get_price_history(period=period, start=start, end=end)
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        returns.name = f"{self.ticker}_log_return"
        return returns

    def get_historical_vol(
        self,
        window: int = 21,
        period: str = "2y",
        annualize: bool = True,
    ) -> pd.Series:
        """
        Rolling historical (realized) volatility.

        Parameters
        ----------
        window : int
            Rolling window in trading days (21 = 1 month, 63 = 1 quarter).
        annualize : bool
            If True, multiply by sqrt(252) to annualize.

        Returns
        -------
        pd.Series
            Rolling volatility estimate.
        """
        rets = self.get_log_returns(period=period)
        vol = rets.rolling(window).std()
        if annualize:
            vol = vol * np.sqrt(252)
        vol.name = f"{self.ticker}_hist_vol_{window}d"
        return vol.dropna()

    def get_dividend_yield(self) -> float:
        """
        Return the trailing dividend yield (used as continuous dividend rate q).

        Returns
        -------
        float
            Annual dividend yield as a decimal (e.g. 0.013 for 1.3%).
        """
        info = self._yf_ticker.info
        return float(info.get("dividendYield") or 0.0)

    # -----------------------------------------------------------------------
    # OPTIONS CHAIN
    # -----------------------------------------------------------------------

    def get_option_chain(
        self,
        expiry: Optional[str] = None,
        option_type: str = "both",
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> pd.DataFrame:
        """
        Download option chain data.

        Parameters
        ----------
        expiry : str, optional
            Expiration date "YYYY-MM-DD". If None, fetches all available expiries and concatenates them into a single DataFrame.
        option_type : {"call", "put", "both"} 
            Which leg(s) to return.
        min_volume : int
            Filter out contracts with volume below this threshold.
        min_open_interest : int
            Filter out contracts with open interest below this threshold.

        Returns
        -------
        pd.DataFrame
            Columns include: contractSymbol, strike, lastPrice, bid, ask, impliedVolatility, volume, openInterest, inTheMoney, expiration, optionType, mid, spread, T (years to expiry).
        """
        expiries = self._yf_ticker.options
        if not expiries:
            raise ValueError(f"No options data available for {self.ticker}.")

        if expiry:
            # Validate and select closest available expiry
            target = pd.Timestamp(expiry)
            avail = pd.to_datetime(list(expiries))
            closest = avail[np.argmin(np.abs(avail - target))].strftime("%Y-%m-%d")
            if closest != expiry:
                print(f"[data_loader] Expiry {expiry} not found; using {closest}.")
            expiry = closest
            expiry_list = [expiry]
        else:
            expiry_list = list(expiries)

        frames: list[pd.DataFrame] = []
        for exp in expiry_list:
            cache_key = f"{exp}_{option_type}"
            if cache_key in self._chain_cache:
                frames.append(self._chain_cache[cache_key])
                continue

            try:
                chain = self._yf_ticker.option_chain(exp)
            except Exception as e:
                print(f"[data_loader] Skipping {exp}: {e}")
                continue

            dfs: list[pd.DataFrame] = []
            if option_type in ("call", "both"):
                df_call = chain.calls.copy()
                df_call["optionType"] = "call"
                dfs.append(df_call)
            if option_type in ("put", "both"):
                df_put = chain.puts.copy()
                df_put["optionType"] = "put"
                dfs.append(df_put)

            df = pd.concat(dfs, ignore_index=True)
            df["expiration"] = pd.Timestamp(exp)

            # Derived columns
            df["mid"] = (df["bid"] + df["ask"]) / 2
            df["spread"] = df["ask"] - df["bid"]

            today = pd.Timestamp(date.today())
            df["T"] = (df["expiration"] - today).dt.days / 365.0

            # Clean up
            df = df.dropna(subset=["strike", "impliedVolatility"])
            df = df[df["impliedVolatility"] > 0]
            df = df[df["T"] > 0]

            if min_volume > 0:
                df = df[df["volume"].fillna(0) >= min_volume]
            if min_open_interest > 0:
                df = df[df["openInterest"].fillna(0) >= min_open_interest]

            self._chain_cache[cache_key] = df
            frames.append(df)

        if not frames:
            raise ValueError("No option data returned after filtering.")

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["expiration", "optionType", "strike"]).reset_index(drop=True)
        return result

    def get_available_expiries(self) -> list[str]:
        """Return list of available option expiry dates as 'YYYY-MM-DD' strings."""
        return list(self._yf_ticker.options)

    def get_otm_options(
        self,
        expiry: Optional[str] = None,
        min_volume: int = 10,
        min_open_interest: int = 100,
    ) -> pd.DataFrame:
        """
        Return OTM calls and puts only.

        Calls where K > S (OTM call), puts where K < S (OTM put).
        At-the-money options are included in both sides.
        """
        spot = self.get_spot_price()
        chain = self.get_option_chain(
            expiry=expiry,
            option_type="both",
            min_volume=min_volume,
            min_open_interest=min_open_interest,
        )
        otm_calls = chain[(chain["optionType"] == "call") & (chain["strike"] >= spot)]
        otm_puts  = chain[(chain["optionType"] == "put")  & (chain["strike"] <= spot)]
        return pd.concat([otm_puts, otm_calls]).sort_values(
            ["expiration", "strike"]
        ).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # YIELD CURVE & RISK-FREE RATE
    # -----------------------------------------------------------------------

    def get_yield_curve(self, as_of: Optional[str] = None) -> pd.Series:
        """
        Fetch the US Treasury yield curve (constant-maturity rates from FRED).

        Parameters
        ----------
        as_of : str, optional
            Date "YYYY-MM-DD" for a historical snapshot. Defaults to most recent.

        Returns
        -------
        pd.Series
            Index: maturity in years (float). Values: annualized yield (decimal).
            e.g. Series({0.25: 0.0532, 0.5: 0.0538, 1.0: 0.0510, ...})
        """
        if self._curve_cache is not None and as_of is None:
            return self._curve_cache

        curve = self._fetch_curve_fred(as_of=as_of)

        if as_of is None:
            self._curve_cache = curve
        return curve

    def _fetch_curve_fred(self, as_of: Optional[str] = None) -> pd.Series:
        """Fetch yield curve via FRED API."""
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("pip install fredapi to use FRED data source.")

        if self._fred is None:
            self._fred = Fred(api_key=self.fred_api_key)

        end_date = as_of or datetime.today().strftime("%Y-%m-%d")
        start_date = (
            pd.Timestamp(end_date) - timedelta(days=10)
        ).strftime("%Y-%m-%d")

        rates: dict[float, float] = {}
        for label, series_id in _FRED_SERIES.items():
            try:
                s = self._fred.get_series(
                    series_id, observation_start=start_date, observation_end=end_date
                )
                val = s.dropna().iloc[-1] if not s.dropna().empty else np.nan
                if not np.isnan(val):
                    rates[_FRED_MATURITIES[label]] = val / 100.0
            except Exception:
                pass

        if not rates:
            raise ValueError("FRED returned no yield data. Check API key.")

        curve = pd.Series(rates).sort_index()
        curve.index.name = "maturity_years"
        curve.name = "yield"
        return curve

    def get_risk_free_rate(self, maturity: float = 1.0) -> float:
        """
        Return a single risk-free rate for a given maturity via linear interpolation.

        FRED DGS series report nominal (bond-equivalent) annualized yields. Convert to continuously compounded via:
            r_cont = ln(1 + r_nominal)

        Parameters
        ----------
        maturity : float
            Time to maturity in years (e.g. 0.5, 1.0, 2.0).

        Returns
        -------
        float
            Continuously compounded risk-free rate as a decimal.
        """
        curve = self.get_yield_curve()
        maturities = np.array(curve.index, dtype=float)
        yields = np.array(curve.values, dtype=float)

        if maturity <= maturities[0]:
            return float(yields[0])
        if maturity >= maturities[-1]:
            return float(yields[-1])

        # Linear interpolation
        r_nominal = float(np.interp(maturity, maturities, yields))

        # Convert from annualized yield to continuous compounding
        r_cont = float(np.log(1 + r_nominal))
        return r_cont

    # -----------------------------------------------------------------------
    # FULL DATA SNAPSHOT
    # -----------------------------------------------------------------------

    def get_market_snapshot(
        self,
        history_period: str = "2y",
        hist_vol_window: int = 21,
    ) -> dict:
        """
        Return a dict with all key market data for the ticker. Kicking off a pricing session with one call.
        """
        spot = self.get_spot_price()
        hist = self.get_price_history(period=history_period)
        rets = self.get_log_returns(period=history_period)
        hvol = self.get_historical_vol(window=hist_vol_window, period=history_period)
        curve = self.get_yield_curve()
        r1y = self.get_risk_free_rate(maturity=1.0)
        q = self.get_dividend_yield()
        expiries = self.get_available_expiries()

        print(f"\n{'='*50}")
        print(f" Market Snapshot: {self.ticker}")
        print(f"{'='*50}")
        print(f" Spot price : {spot:.2f}")
        print(f" Dividend yield : {q*100:.2f}%")
        print(f" Hist. vol ({hist_vol_window}d) : {hvol.iloc[-1]*100:.2f}%")
        print(f" Risk-free rate (1y) : {r1y*100:.2f}%")
        print(f" Available expiries : {len(expiries)} dates")
        print(f" Price history : {hist.index[0].date()} → {hist.index[-1].date()}")
        print(f"{'='*50}\n")

        return {
            "ticker": self.ticker,
            "spot": spot,
            "dividend_yield": q,
            "hist_vol_current": float(hvol.iloc[-1]),
            "price_history": hist,
            "log_returns": rets,
            "hist_vol_series": hvol,
            "yield_curve": curve,
            "risk_free_1y": r1y,
            "expiries": expiries,
        }

    # -----------------------------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle all cached data to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": self.ticker,
            "price_cache": self._price_cache,
            "chain_cache": self._chain_cache,
            "curve_cache": self._curve_cache,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[data_loader] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path, fred_api_key: Optional[str] = None) -> "MarketDataLoader":
        """Load a previously saved MarketDataLoader from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        loader = cls(ticker=payload["ticker"], fred_api_key=fred_api_key)
        loader._price_cache = payload["price_cache"]
        loader._chain_cache = payload["chain_cache"]
        loader._curve_cache = payload["curve_cache"]
        print(f"[data_loader] Loaded from {path}")
        return loader