"""
Unit tests for src/engines/analytical.py — BSM pricing, Greeks, and implied volatility inversion.
"""

import numpy as np
import pytest

from src.engines.analytical import BSMModel, bsm_price, bsm_greeks


class TestBSMPricing:
    """Verify BSM prices against known reference values."""

    def test_atm_call_matches_reference(self, standard_params):
        bsm = BSMModel(**standard_params)
        price = bsm.price("call")
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_atm_put_matches_reference(self, standard_params):
        bsm = BSMModel(**standard_params)
        price = bsm.price("put")
        assert price == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity_holds(self, standard_params):
        bsm = BSMModel(**standard_params)
        assert bsm.put_call_parity_check()

    def test_put_call_parity_holds_with_dividend(self, dividend_params):
        bsm = BSMModel(**dividend_params)
        assert bsm.put_call_parity_check()

    def test_dividend_reduces_call_price(self, standard_params, dividend_params):
        c_no_div = BSMModel(**standard_params).price("call")
        c_div = BSMModel(**dividend_params).price("call")
        assert c_div < c_no_div

    def test_dividend_increases_put_price(self, standard_params, dividend_params):
        p_no_div = BSMModel(**standard_params).price("put")
        p_div = BSMModel(**dividend_params).price("put")
        assert p_div > p_no_div

    def test_deep_otm_call_near_zero(self):
        bsm = BSMModel(S=100, K=200, T=0.1, r=0.05, sigma=0.20, q=0.0)
        assert bsm.price("call") < 0.001

    def test_deep_itm_call_near_intrinsic_forward(self):
        bsm = BSMModel(S=100, K=10, T=1.0, r=0.05, sigma=0.20, q=0.0)
        approx = 100 - 10 * np.exp(-0.05)
        assert bsm.price("call") == pytest.approx(approx, abs=0.01)

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_call_price_decreasing_in_strike(self, K):
        prices = BSMModel(
            S=100, K=np.array([80, 90, 100, 110, 120]),
            T=1.0, r=0.05, sigma=0.20
        ).price("call")
        assert np.all(np.diff(prices) < 0)

    def test_vectorized_strikes_shape(self):
        strikes = np.arange(80, 121, 5, dtype=float)
        bsm = BSMModel(S=100, K=strikes, T=1.0, r=0.05, sigma=0.20)
        prices = bsm.price("call")
        assert prices.shape == strikes.shape


class TestBSMValidation:

    @pytest.mark.parametrize("bad_kwargs", [
        dict(S=-1,  K=100, T=1.0, r=0.05, sigma=0.2),
        dict(S=100, K=0, T=1.0, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=0, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=1.0, r=0.05, sigma=0),
        dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=-0.01),
    ])
    def test_invalid_inputs_raise(self, bad_kwargs):
        with pytest.raises(ValueError):
            BSMModel(**bad_kwargs)

    def test_invalid_option_type_raises(self, standard_params):
        bsm = BSMModel(**standard_params)
        with pytest.raises(ValueError):
            bsm.price("invalid_type")


class TestBSMGreeks:
    """Verify Greeks signs and known relationships."""

    def test_call_delta_in_range(self, standard_params):
        delta = BSMModel(**standard_params).delta("call")
        assert 0 < delta < 1

    def test_put_delta_in_range(self, standard_params):
        delta = BSMModel(**standard_params).delta("put")
        assert -1 < delta < 0

    def test_gamma_positive(self, standard_params):
        assert BSMModel(**standard_params).gamma() > 0

    def test_vega_positive(self, standard_params):
        assert BSMModel(**standard_params).vega() > 0

    def test_call_theta_negative(self, standard_params):
        assert BSMModel(**standard_params).theta("call") < 0

    def test_call_rho_positive(self, standard_params):
        assert BSMModel(**standard_params).rho("call") > 0

    def test_put_rho_negative(self, standard_params):
        assert BSMModel(**standard_params).rho("put") < 0

    def test_vanna_matches_finite_difference(self, standard_params):
        """Vanna = d(delta)/d(sigma) — verify against bump-and-reprice."""
        eps = 1e-4
        p_up = {**standard_params, "sigma": standard_params["sigma"] + eps}
        p_down = {**standard_params, "sigma": standard_params["sigma"] - eps}
        delta_up = BSMModel(**p_up).delta("call")
        delta_down = BSMModel(**p_down).delta("call")
        vanna_fd = (delta_up - delta_down) / (2 * eps)
        vanna_analytical = BSMModel(**standard_params).vanna()
        assert vanna_analytical == pytest.approx(vanna_fd, abs=1e-5)

    def test_all_greeks_keys_present(self, standard_params):
        g = BSMModel(**standard_params).all_greeks("call")
        expected_keys = {"delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"}
        assert set(g.keys()) == expected_keys


class TestImpliedVolatility:
    """Verify IV inversion recovers the volatility used to generate prices."""

    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.35, 0.50, 0.80])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_iv_roundtrip(self, sigma_true, option_type):
        market_price = float(
            BSMModel(S=100, K=100, T=1.0, r=0.05, sigma=sigma_true, q=0.0).price(option_type)
        )
        iv = BSMModel.implied_vol(
            market_price, S=100, K=100, T=1.0, r=0.05, q=0.0, option_type=option_type,
        )
        assert iv == pytest.approx(sigma_true, abs=1e-6)

    def test_iv_vectorized_over_strikes(self):
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        sigma_true = 0.25
        prices = bsm_price(S=100, K=strikes, T=0.5, r=0.05, sigma=sigma_true, q=0.0, option_type="call")
        ivs = BSMModel.implied_vol_vectorized(
            prices, S=100, strikes=strikes, T=0.5, r=0.05, option_types="call"
        )
        assert np.allclose(ivs, sigma_true, atol=1e-5)

    def test_iv_raises_below_intrinsic(self):
        with pytest.raises(ValueError):
            BSMModel.implied_vol(
                market_price=0.0001, S=100, K=50, T=1.0, r=0.05, q=0.0, option_type="call",
            )


class TestModuleLevelHelpers:

    def test_bsm_price_wrapper(self, standard_params):
        price = bsm_price(**standard_params, option_type="call")
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_bsm_greeks_wrapper(self, standard_params):
        g = bsm_greeks(**standard_params, option_type="call")
        assert "delta" in g and "gamma" in g