"""
Calibrate the Heston model to market-observed implied volatilities.

Finds the 5 Heston parameters (kappa, theta, xi, rho, v0) that minimise the weighted IV RMSE between model and market implied vols, using a
two-stage optimiser: Differential Evolution (global) then L-BFGS-B (local).

See README.md — Theory — Heston Calibration for full methodology, objective function derivation.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.models.equity.heston import HestonModel
from src.engines.analytical import BSMModel


@dataclass
class CalibrationResult:
    """
    Container for Heston calibration output.

    Attributes
    ----------
    kappa, theta, xi, rho, v0 : float
        Calibrated Heston parameters.
    rmse : float
        Root mean squared error between model and market implied vols.
    max_error : float
        Maximum absolute IV error across all strikes/maturities.
    n_evaluations : int
        Total number of objective function calls.
    elapsed_sec : float
        Wall-clock time for calibration.
    feller_satisfied : bool
        Whether the Feller condition holds for calibrated parameters.
    model_ivs : np.ndarray
        Model implied vols at calibrated parameters.
    market_ivs : np.ndarray
        Market implied vols used for calibration.
    strikes : np.ndarray
    maturities : np.ndarray
    """
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    rmse: float
    max_error: float
    n_evaluations: int
    elapsed_sec: float
    feller_satisfied: bool
    model_ivs: np.ndarray
    market_ivs: np.ndarray
    strikes: np.ndarray
    maturities: np.ndarray
    weights: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def params(self) -> dict:
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho":self.rho,
            "v0": self.v0,
        }

    def __str__(self) -> str:
        fc = "Satisfied" if self.feller_satisfied else "Violated"
        lines = [
            "── Heston Calibration Result ──────────────────────",
            f" κ (kappa) = {self.kappa:.6f}",
            f" θ (theta) = {self.theta:.6f}  (long-run vol ≈ {np.sqrt(self.theta)*100:.2f}%)",
            f" ξ (xi) = {self.xi:.6f}",
            f" ρ (rho) = {self.rho:.6f}",
            f" v0 = {self.v0:.6f}  (spot vol = {np.sqrt(self.v0)*100:.2f}%)",
            f" Feller condition  {fc}  (2κθ/ξ^2 = {2*self.kappa*self.theta/self.xi**2:.3f})",
            f" IV RMSE = {self.rmse*100:.4f}%",
            f" IV max error = {self.max_error*100:.4f}%",
            f" Evaluations = {self.n_evaluations:,}",
            f" Time = {self.elapsed_sec:.1f}s"
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class HestonCalibrator:
    """
    Calibrate Heston model parameters to market implied volatilities.
    """

    # Default parameter bounds
    DEFAULT_BOUNDS = {
        "kappa": (0.1, 10.0),
        "theta": (0.001, 1.0),
        "xi": (0.01, 2.0),
        "rho": (-0.99, 0.99),
        "v0": (0.001, 1.0),
    }

    def __init__(
        self,
        S: float,
        r: float,
        q: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_ivs: np.ndarray,
        option_types: str | list = "call",
        weights: np.ndarray | None = None,
        n_paths: int = 20_000,
        n_steps: int = 50,
        seed: int = 926,
        bounds: dict | None = None,
    ):
        self.S = float(S)
        self.r = float(r)
        self.q = float(q)
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.market_ivs = np.asarray(market_ivs, dtype=float)
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

        N = len(self.strikes)
        if isinstance(option_types, str):
            self.option_types = [option_types] * N
        else:
            self.option_types = list(option_types)

        self.weights = (np.ones(N) if weights is None
                        else np.asarray(weights, dtype=float))
        self.weights = self.weights / self.weights.sum()   # normalise

        # Merge default bounds with any overrides
        b = dict(self.DEFAULT_BOUNDS)
        if bounds:
            b.update(bounds)
        self._bounds = [b["kappa"], b["theta"], b["xi"], b["rho"], b["v0"]]

        self._n_evals = 0

    def calibrate(
        self,
        de_maxiter: int = 300,
        de_popsize: int = 12,
        de_tol: float = 1e-5,
        refine: bool = True,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Run two-stage calibration

        Parameters
        ----------
        de_maxiter : int
            Max generations for differential evolution.
        de_popsize : int
            Population size multiplier for DE (total = popsize * 5 params).
        de_tol : float
            Convergence tolerance for DE.
        refine : bool
            If True, run L-BFGS-B local refinement after DE.
        verbose : bool
            Print progress during calibration.
        """
        self._n_evals = 0
        t0 = time.time()

        if verbose:
            print("Stage 1: Differential Evolution (global search)")

        de_result = differential_evolution(
            func=self._objective,
            bounds=self._bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=de_tol,
            seed=self.seed,
            polish=False,
            workers=1,
        )

        x_best = de_result.x

        if verbose:
            print(f" DE done: RMSE={np.sqrt(de_result.fun)*100:.4f}%  evals={self._n_evals:,}")

        if refine:
            if verbose:
                print("Stage 2: L-BFGS-B local refinement")

            local_result = minimize(
                fun=self._objective,
                x0=x_best,
                method="L-BFGS-B",
                bounds=self._bounds,
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
            )
            x_best = local_result.x

            if verbose:
                print(f"  L-BFGS-B done: RMSE={np.sqrt(self._objective(x_best))*100:.4f}%  total evals={self._n_evals:,}")

        elapsed = time.time() - t0

        # Final evaluation at calibrated params
        kappa, theta, xi, rho, v0 = x_best
        model_ivs = self._model_ivs(kappa, theta, xi, rho, v0)
        iv_errors = model_ivs - self.market_ivs
        rmse = float(np.sqrt(np.average(iv_errors**2, weights=self.weights)))
        max_error = float(np.max(np.abs(iv_errors[~np.isnan(iv_errors)])))
        feller = 2 * kappa * theta > xi ** 2

        if verbose:
            print(f"\nCalibration complete in {elapsed:.1f}s")
            print(f" RMSE={rmse*100:.4f}%  MaxErr={max_error*100:.4f}%")
            fc = "satisfied" if feller else "violated"
            print(f"Feller: {fc}  (2κθ/ξ^2 = {2*kappa*theta/xi**2:.3f})")

        return CalibrationResult(
            kappa=float(kappa),
            theta=float(theta),
            xi=float(xi),
            rho=float(rho),
            v0=float(v0),
            rmse=rmse,
            max_error=max_error,
            n_evaluations=self._n_evals,
            elapsed_sec=elapsed,
            feller_satisfied=feller,
            model_ivs=model_ivs,
            market_ivs=self.market_ivs.copy(),
            strikes=self.strikes.copy(),
            maturities=self.maturities.copy(),
            weights=self.weights.copy(),
        )

    def iv_error_table(self, result: CalibrationResult) -> None:
        """Print a strike-by-strike implied vol comparison table."""
        print(f"\n{'Strike':>8} {'Mat':>6} {'Type':>5} {'Mkt IV':>8} {'Mdl IV':>8} {'Error (bp)':>11}")
        print("─" * 52)
        for i in range(len(self.strikes)):
            mkt = self.market_ivs[i]
            mdl = result.model_ivs[i]
            err_bp = (mdl - mkt) * 10_000
            print(f"{self.strikes[i]:>8.1f} {self.maturities[i]:>6.3f} {self.option_types[i]:>5} "
                  f"{mkt*100:>7.3f}% {mdl*100:>7.3f}% {err_bp:>+10.1f}")
        print(f"\n RMSE = {result.rmse*100:.4f}% MaxErr = {result.max_error*100:.4f}%")

    def _objective(self, params: np.ndarray) -> float:
        """
        Weighted IV RMSE objective function.

        Returns squared RMSE (scipy minimises the raw function value).
        """
        self._n_evals += 1
        kappa, theta, xi, rho, v0 = params
        try:
            model_ivs = self._model_ivs(kappa, theta, xi, rho, v0)
        except Exception:
            return 1e6

        iv_errors = model_ivs - self.market_ivs

        iv_errors = np.where(np.isnan(iv_errors), 1.0, iv_errors)
        return float(np.average(iv_errors ** 2, weights=self.weights))

    def _model_ivs(
        self,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
    ) -> np.ndarray:
        """
        Compute Heston model implied vols for all (K, T) pairs.

        Steps:
          1. Simulate Heston paths.
          2. Compute discounted payoff.
          3. Invert BSM to get model implied vol.
        """
        model = HestonModel(
            S0=self.S, v0=v0, r=self.r, q=self.q, kappa=kappa, theta=theta, xi=xi, rho=rho,
        )

        ivs = np.full(len(self.strikes), np.nan)
        unique_mats = np.unique(self.maturities)

        for T in unique_mats:
            mask = self.maturities == T
            K_set = self.strikes[mask]
            types = [self.option_types[i]
                     for i in range(len(self.strikes)) if mask[i]]

            rng = np.random.default_rng(self.seed)
            S_T = model.simulate_terminal(T=T, n_paths=self.n_paths, n_steps=self.n_steps, rng=rng)
            disc   = np.exp(-self.r * T)

            for j, (K, opt_type) in enumerate(zip(K_set, types)):
                if opt_type == "call":
                    payoffs = np.maximum(S_T - K, 0.0)
                else:
                    payoffs = np.maximum(K - S_T, 0.0)

                mc_price = disc * float(np.mean(payoffs))

                iv = BSMModel.implied_vol(
                    market_price=mc_price, S=self.S, K=K, T=T,
                    r=self.r, q=self.q, option_type=opt_type,
                )

                global_idx = np.where(mask)[0][j]
                ivs[global_idx] = iv

        return ivs