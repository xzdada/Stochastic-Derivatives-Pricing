"""
Tests for src/risk/greeks.py — analytical vs bump-and-reprice Greeks, and cross-validation against the Binomial tree engine for American options.
"""

import numpy as np
import pytest

from src.engines.analytical import BSMModel
from src.engines.trees import BinomialTree
from src.products.american import AmericanOption
from src.products.european import EuropeanOption
from src.risk.greeks import GreeksCalculator, GreeksResult


@pytest.fixture
def bsm_pricer():
    def _pricer(option):
        return float(
            BSMModel(option.S, option.K, option.T, option.r, option.sigma, option.q)
            .price(option.option_type)
        )
    return _pricer


class TestAnalyticalGreeks:
    """GreeksCalculator.analytical() for European options."""

    def test_returns_all_fields(self, standard_params):
        opt = EuropeanOption(**standard_params, option_type="call")
        g = GreeksCalculator(opt).analytical()
        assert isinstance(g, GreeksResult)
        assert g.method == "analytical"
        for field in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"):
            assert getattr(g, field) is not None

    def test_call_greeks_signs(self, standard_params):
        opt = EuropeanOption(**standard_params, option_type="call")
        g = GreeksCalculator(opt).analytical()
        assert g.delta > 0
        assert g.gamma > 0
        assert g.vega > 0
        assert g.theta < 0
        assert g.rho > 0

    def test_put_greeks_signs(self, standard_params):
        opt = EuropeanOption(**standard_params, option_type="put")
        g = GreeksCalculator(opt).analytical()
        assert g.delta < 0
        assert g.gamma > 0
        assert g.rho < 0


class TestBumpAndRepriceGreeks:
    """GreeksCalculator.bump_and_reprice() must agree with analytical."""

    def test_matches_analytical_within_tolerance(self, standard_params, bsm_pricer):
        opt = EuropeanOption(**standard_params, option_type="call")
        calc = GreeksCalculator(opt)
        cmp = calc.compare_methods(bsm_pricer)
        assert cmp["max_abs_diff"] < 0.001

    @pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
    def test_each_greek_close_to_analytical(self, standard_params, bsm_pricer, greek):
        opt = EuropeanOption(**standard_params, option_type="call")
        calc = GreeksCalculator(opt)
        g_ana  = calc.analytical()
        g_bump = calc.bump_and_reprice(bsm_pricer)
        assert getattr(g_bump, greek) == pytest.approx(getattr(g_ana, greek), abs=1e-3)


class TestAmericanOptionGreeks:
    """Bump-and-reprice is required for American options (no closed form)."""

    def test_american_put_greeks_via_tree(self, dividend_params):
        opt = AmericanOption(**dividend_params, option_type="put")
        tree = BinomialTree(n_steps=200)
        pricer = lambda o: tree.price(o).price

        calc = GreeksCalculator(opt)
        g = calc.bump_and_reprice(pricer)

        assert g.delta < 0
        assert g.gamma > 0
        assert g.vega > 0
        assert g.theta < 0

    def test_american_put_delta_more_negative_than_european(self, dividend_params):
        """American puts with early exercise tend to have larger |delta| near the money due to the exercise boundary effect."""
        am_opt = AmericanOption(**dividend_params, option_type="put")
        eu_opt = EuropeanOption(**dividend_params, option_type="put")
        tree = BinomialTree(n_steps=200)

        pricer_tree = lambda o: tree.price(o).price
        pricer_bsm  = lambda o: float(
            BSMModel(o.S, o.K, o.T, o.r, o.sigma, o.q).price(o.option_type)
        )

        g_am = GreeksCalculator(am_opt).bump_and_reprice(pricer_tree)
        g_eu = GreeksCalculator(eu_opt).bump_and_reprice(pricer_bsm)

        assert g_am.delta <= g_eu.delta + 1e-6


class TestGreeksSurface:
    """Greeks surface across a range of spot prices."""

    def test_surface_shape(self, standard_params, bsm_pricer):
        opt = EuropeanOption(**standard_params, option_type="call")
        calc = GreeksCalculator(opt)
        surface = calc.greeks_surface(bsm_pricer, S_range=(80, 120), n_S=10)
        assert surface["delta"].shape == (1, 10)

    def test_call_delta_increasing_in_spot(self, standard_params, bsm_pricer):
        opt = EuropeanOption(**standard_params, option_type="call")
        calc = GreeksCalculator(opt)
        surface = calc.greeks_surface(bsm_pricer, S_range=(80, 120), n_S=10)
        assert np.all(np.diff(surface["delta"][0]) > 0)

    def test_gamma_always_positive_across_surface(self, standard_params, bsm_pricer):
        opt = EuropeanOption(**standard_params, option_type="call")
        calc = GreeksCalculator(opt)
        surface = calc.greeks_surface(bsm_pricer, S_range=(50, 150), n_S=20)
        assert np.all(surface["gamma"] > 0)


class TestVannaFiniteDifferenceCheck:
    
    def test_vanna_matches_delta_bump(self, standard_params):
        opt = EuropeanOption(**standard_params, option_type="call")
        g = GreeksCalculator(opt).analytical()

        eps = 1e-4
        p = standard_params
        delta_up = BSMModel(
            p["S"], p["K"], p["T"], p["r"], p["sigma"] + eps, p["q"]
        ).delta("call")
        delta_down = BSMModel(
            p["S"], p["K"], p["T"], p["r"], p["sigma"] - eps, p["q"]
        ).delta("call")
        vanna_fd = float(delta_up - delta_down) / (2 * eps)

        assert g.vanna == pytest.approx(vanna_fd, abs=1e-4)