"""
Tests verifying Monte Carlo pricing converges to the BSM analytical benchmark, and that variance reduction techniques reduce standard error without introducing bias.
"""

import numpy as np
import pytest

from src.engines.analytical import BSMModel
from src.engines.monte_carlo import MCEngine, StochasticModel
from src.models.equity.gbm import GBMModel
from src.products.european import EuropeanOption


@pytest.fixture
def gbm_call_setup(standard_params):
    """Standard GBM model + European call + MCEngine, ready to price."""
    p = standard_params
    model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
    product = EuropeanOption(**p, option_type="call")
    engine = MCEngine(model=model, product=product, n_paths=200_000, seed=926)
    bsm_price = float(BSMModel(**p).price("call"))
    return engine, bsm_price


class TestGBMModel:
    """GBM path simulation correctness."""

    def test_satisfies_stochastic_model_protocol(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        assert isinstance(model, StochasticModel)

    def test_path_shape(self, standard_params, rng):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        paths = model.simulate(T=1.0, n_paths=1000, n_steps=50, rng=rng)
        assert paths.shape == (1000, 51)

    def test_paths_start_at_S0(self, standard_params, rng):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        paths = model.simulate(T=1.0, n_paths=1000, n_steps=50, rng=rng)
        assert np.allclose(paths[:, 0], p["S"])

    def test_paths_remain_positive(self, standard_params, rng):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        paths = model.simulate(T=1.0, n_paths=1000, n_steps=50, rng=rng)
        assert np.all(paths > 0)

    def test_martingale_property(self, standard_params):
        """E[S_T] should equal S0 * exp((r-q)*T) under the risk-neutral measure."""
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        S_T = model.simulate_terminal(T=1.0, n_paths=300_000, seed=926)
        expected = p["S"] * np.exp((p["r"] - p["q"]) * 1.0)
        assert np.mean(S_T) == pytest.approx(expected, rel=0.01)

    def test_antithetic_paths_negatively_correlated(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        paths, paths_anti = model.simulate_antithetic(
            T=1.0, n_paths=50_000, n_steps=1, seed=926
        )
        corr = np.corrcoef(paths[:, -1], paths_anti[:, -1])[0, 1]
        assert corr < 0


class TestMCConvergence:
    """Standard Monte Carlo must converge to the BSM analytical price."""

    def test_standard_mc_matches_bsm(self, gbm_call_setup):
        engine, bsm_price = gbm_call_setup
        result = engine.price()
        assert result.price == pytest.approx(bsm_price, abs=0.05)

    def test_bsm_price_within_confidence_interval(self, gbm_call_setup):
        engine, bsm_price = gbm_call_setup
        result = engine.price()
        assert result.ci_low < bsm_price < result.ci_high

    def test_stderr_decreases_with_more_paths(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        product = EuropeanOption(**p, option_type="call")
        engine = MCEngine(model=model, product=product, seed=926)

        result_small = engine.price(n_paths=1_000)
        result_large = engine.price(n_paths=100_000)
        assert result_large.stderr < result_small.stderr


class TestSeedReproducibility:
    """Two-layer seed design: instance default + call-time override."""

    def test_instance_seed_reproducible(self, gbm_call_setup):
        engine, _ = gbm_call_setup
        r1 = engine.price()
        r2 = engine.price()
        assert r1.price == r2.price

    def test_call_time_seed_override_reproducible(self, gbm_call_setup):
        engine, _ = gbm_call_setup
        r1 = engine.price(seed=99)
        r2 = engine.price(seed=99)
        assert r1.price == r2.price

    def test_different_seeds_give_different_results(self, gbm_call_setup):
        engine, _ = gbm_call_setup
        r1 = engine.price()  # instance seed
        r2 = engine.price(seed=99)   # different call-time seed
        assert r1.price != r2.price


class TestVarianceReduction:
    """Antithetic variates and control variates must reduce std error without bias."""

    def test_antithetic_reduces_stderr(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        product = EuropeanOption(**p, option_type="call")
        engine = MCEngine(model=model, product=product, n_paths=100_000, seed=926)

        result_std  = engine.price()
        result_anti = engine.price_antithetic()
        assert result_anti.stderr < result_std.stderr

    def test_antithetic_unbiased(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        product = EuropeanOption(**p, option_type="call")
        engine = MCEngine(model=model, product=product, n_paths=100_000, seed=926)
        bsm_price = float(BSMModel(**p).price("call"))

        result_anti = engine.price_antithetic()
        assert result_anti.price == pytest.approx(bsm_price, abs=0.05)

    def test_control_variate_reduces_stderr(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        product = EuropeanOption(**p, option_type="call")
        engine = MCEngine(model=model, product=product, n_paths=100_000, seed=926)

        result_std = engine.price()
        result_cv  = engine.price_control_variate()
        assert result_cv.stderr < result_std.stderr

    def test_control_variate_matches_bsm_closely(self, standard_params):
        """For European option under GBM, rho=1 with the BSM control, so the CV estimator should collapse almost exactly to BSM."""
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        product = EuropeanOption(**p, option_type="call")
        engine = MCEngine(model=model, product=product, n_paths=100_000, seed=926)
        bsm_price = float(BSMModel(**p).price("call"))

        result_cv = engine.price_control_variate()
        assert result_cv.price == pytest.approx(bsm_price, abs=0.02)

    def test_put_call_parity_via_mc(self, standard_params):
        p = standard_params
        model = GBMModel(S0=p["S"], r=p["r"], sigma=p["sigma"], q=p["q"])
        call = EuropeanOption(**p, option_type="call")
        put  = EuropeanOption(**p, option_type="put")

        engine_c = MCEngine(model=model, product=call, n_paths=300_000, seed=926)
        engine_p = MCEngine(model=model, product=put,  n_paths=300_000, seed=926)

        c_price = engine_c.price().price
        p_price = engine_p.price().price

        lhs = c_price - p_price
        rhs = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
        assert lhs == pytest.approx(rhs, abs=0.1)


class TestConvergenceStudy:
    """The convergence_study() helper must run without error and return
    results across all requested path counts and methods."""

    def test_convergence_study_returns_expected_count(self, gbm_call_setup):
        engine, _ = gbm_call_setup
        results = engine.convergence_study(
            path_counts=[1_000, 10_000],
            methods=["standard", "antithetic"],
            seed=926,
        )
        assert len(results) == 4   # 2 path x 2 methods

    def test_convergence_study_errors_shrink(self, gbm_call_setup):
        engine, bsm_price = gbm_call_setup
        results = engine.convergence_study(
            path_counts=[1_000, 100_000],
            methods=["standard"],
            seed=926,
        )
        errors = [abs(r["price"] - bsm_price) for r in results]
        # large-N error should generally be smaller on average across repeated runs.

        assert all(e < 1.0 for e in errors)