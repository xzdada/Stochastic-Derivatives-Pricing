"""
Tests for put-call parity and no-arbitrage boundary conditions across pricing engines (BSM analytical, Monte Carlo, Binomial tree).

Put-call parity:  C - P = S*e^(-qT) - K*e^(-rT)

Put-call parity is a model-free, no-arbitrage relationship, which must hold under every pricing method. Violations indicate arbitrage opportunity in real market data.
"""

import numpy as np
import pytest

from src.engines.analytical import BSMModel
from src.engines.monte_carlo import MCEngine
from src.engines.trees import BinomialTree
from src.models.equity.gbm import GBMModel
from src.products.european import EuropeanOption


def parity_rhs(S, K, T, r, q):
    """Right-hand side of put-call parity."""
    return S * np.exp(-q * T) - K * np.exp(-r * T)


class TestParityBSM:
    """Put-call parity under the BSM analytical engine."""

    def test_parity_no_dividend(self, standard_params):
        bsm = BSMModel(**standard_params)
        assert bsm.put_call_parity_check()

    def test_parity_with_dividend(self, dividend_params):
        bsm = BSMModel(**dividend_params)
        assert bsm.put_call_parity_check()

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_parity_across_strikes(self, K):
        p = dict(S=100, T=1.0, r=0.05, sigma=0.20, q=0.013)
        bsm = BSMModel(K=K, **p)
        C = bsm.price("call")
        P = bsm.price("put")
        rhs = parity_rhs(p["S"], K, p["T"], p["r"], p["q"])
        assert (C - P) == pytest.approx(rhs, abs=1e-8)

    @pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_parity_across_maturities(self, T):
        p = dict(S=100, K=100, r=0.05, sigma=0.20, q=0.013)
        bsm = BSMModel(T=T, **p)
        C = bsm.price("call")
        P = bsm.price("put")
        rhs = parity_rhs(p["S"], p["K"], T, p["r"], p["q"])
        assert (C - P) == pytest.approx(rhs, abs=1e-8)


class TestParityMonteCarlo:
    """Put-call parity should hold (within MC noise) under simulation."""

    def test_parity_via_standard_mc(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        call = EuropeanOption(**p, option_type="call")
        put  = EuropeanOption(**p, option_type="put")

        engine_c = MCEngine(model=model, product=call, n_paths=300_000, seed=926)
        engine_p = MCEngine(model=model, product=put,  n_paths=300_000, seed=926)

        C = engine_c.price().price
        P = engine_p.price().price
        rhs = parity_rhs(p["S"], p["K"], p["T"], p["r"], p["q"])

        assert (C - P) == pytest.approx(rhs, abs=0.1)


class TestParityBinomialTree:
    """Put-call parity should hold (within discretisation error) on the tree."""

    def test_parity_via_european_tree(self, dividend_params):
        p = dividend_params
        call = EuropeanOption(**p, option_type="call")
        put  = EuropeanOption(**p, option_type="put")
        tree = BinomialTree(n_steps=500)

        C = tree.price(call).price
        P = tree.price(put).price
        rhs = parity_rhs(p["S"], p["K"], p["T"], p["r"], p["q"])

        assert (C - P) == pytest.approx(rhs, abs=0.01)


class TestNoArbitrageBounds:
    """Option prices must respect fundamental no-arbitrage bounds."""

    def test_call_price_bounded_by_spot(self, standard_params):
        """A call can never be worth more than the dividend-adjusted spot."""
        bsm = BSMModel(**standard_params)
        C = bsm.price("call")
        upper_bound = standard_params["S"] * np.exp(-standard_params["q"] * standard_params["T"])
        assert C <= upper_bound

    def test_call_price_exceeds_intrinsic_forward_value(self, standard_params):
        """Call price must be >= max(S*e^-qT - K*e^-rT, 0)."""
        p = standard_params
        bsm = BSMModel(**p)
        C = bsm.price("call")
        intrinsic = max(p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"]), 0.0)
        assert C >= intrinsic - 1e-8

    def test_put_price_bounded_by_discounted_strike(self, standard_params):
        """A put can never be worth more than the discounted strike."""
        p = standard_params
        bsm = BSMModel(**p)
        P = bsm.price("put")
        upper_bound = p["K"] * np.exp(-p["r"] * p["T"])
        assert P <= upper_bound

    def test_american_geq_european(self, dividend_params):
        """American option price must always be >= European."""
        from src.products.american import AmericanOption

        am_put = AmericanOption(**dividend_params, option_type="put")
        eu_put = EuropeanOption(**dividend_params, option_type="put")
        tree = BinomialTree(n_steps=500)

        am_price = tree.price(am_put).price
        eu_price = tree.price(eu_put).price
        assert am_price >= eu_price - 1e-8

    def test_american_call_no_dividend_equals_european(self, standard_params):
        """With q=0, American call should never be exercised early, EEP should be approx 0."""
        from src.products.american import AmericanOption

        am_call = AmericanOption(**standard_params, option_type="call")
        eu_call = EuropeanOption(**standard_params, option_type="call")
        tree = BinomialTree(n_steps=500)

        am_price = tree.price(am_call).price
        eu_price = tree.price(eu_call).price
        assert abs(am_price - eu_price) < 0.001


class TestMonotonicityInStrike:
    """Call prices decrease and put prices increase monotonically in K."""

    def test_call_prices_strictly_decreasing(self):
        strikes = np.arange(80, 121, 5, dtype=float)
        prices = BSMModel(S=100, K=strikes, T=1.0, r=0.05, sigma=0.20).price("call")
        assert np.all(np.diff(prices) < 0)

    def test_put_prices_strictly_increasing(self):
        strikes = np.arange(80, 121, 5, dtype=float)
        prices = BSMModel(S=100, K=strikes, T=1.0, r=0.05, sigma=0.20).price("put")
        assert np.all(np.diff(prices) > 0)


class TestMonotonicityInMaturity:
    """For non-dividend-paying assets, longer-tenor options are worth more (time value is non-negative)."""

    def test_call_value_increases_with_maturity(self):
        maturities = [0.25, 0.5, 1.0, 2.0, 5.0]
        prices = [
            BSMModel(S=100, K=100, T=T, r=0.05, sigma=0.20, q=0.0).price("call")
            for T in maturities
        ]
        assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))