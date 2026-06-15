"""
Finite difference PDE solver for European and American options.

Mathematical foundation
-----------------------
By Ito's lemma, the BSM PDE for option value V(S, t) is:

    ∂V/∂t + (r-q)S ∂V/∂S + 0.5 σ^2 S^2 ∂²V/∂S² - rV = 0

with terminal condition V(S, T) = Payoff(S).

Discretisation
--------------
Define a uniform grid:
  - S-axis : S_min = 0, S_max = 4·K  (sufficient for OTM options)
  - t-axis : t_0 = 0, …, t_M = T, step Δt = T/M

The PDE is discretised at interior nodes (i=1…I-1, j=1…M):

Coefficients at node i:
    a_i = 0.5 Δt( σ^2 i^2 - (r-q)i )        sub-diagonal
    b_i = 1 - Δt( σ^2 i^2 + r )       diagonal
    c_i = 0.5 Δt( σ^2 i^2 + (r-q)i )        super-diagonal

Explicit scheme (unstable for large Δt):
    V_i^{j} = a_i V_{i-1}^{j+1} + b_i^{exp} V_i^{j+1} + c_i V_{i+1}^{j+1}

Implicit scheme (unconditionally stable, first-order in t):
    a_i V_{i-1}^j + b_i^{imp} V_i^j + c_i V_{i+1}^j = V_i^{j+1}

Crank-Nicolson (second-order in both S and t, unconditionally stable):
    0.5 explicit + 0.5 implicit at each time step. Requires solving a tridiagonal system at every step (Thomas algorithm, O(I) per step).

Boundary conditions
    S = 0 : call -> 0,  put -> K e^{-r(T-t)}
    S = S_max: call -> S_max - K e^{-r(T-t)},  put -> 0

American free boundary (PSOR)
    At each time step, after solving the tridiagonal system, enforce:
        V_i^j = max(V_i^j, Intrinsic_i^j)
    This is the projected SOR (PSOR) approach — project onto the early exercise constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from src.products.base_option import Option


@dataclass
class FDResult:
    """
    Container for a finite difference pricing result.

    Attributes
    ----------
    price : float
        Interpolated option price at (S0, t=0).
    scheme : str
        Finite difference scheme used.
    S_steps, T_steps : int
        Grid dimensions.
    early_exercise_premium : float or None
    """
    price: float
    scheme: str
    S_steps: int
    T_steps: int
    early_exercise_premium: float | None = None

    def __str__(self) -> str:
        eep = (f" | EEP={self.early_exercise_premium:.4f}"
               if self.early_exercise_premium is not None else "")
        return (
            f"FDResult({self.scheme}) | price={self.price:.6f} | grid={self.S_steps}*{self.T_steps}{eep}"
        )

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Crank-Nicolson solver
# ---------------------------------------------------------------------------

class CrankNicolson:
    """
    Crank-Nicolson finite difference solver for European and American options.

    Solves the BSM PDE backward in time on a uniform (S, t) grid. American early exercise is handled via the projection (PSOR) method.

    Parameters
    ----------
    S_steps : int
        Number of spatial grid intervals. More → finer S resolution.
    T_steps : int
        Number of time steps. More → finer t resolution.
    S_max_mult : float
        S_max = S_max_mult * K.
    """

    def __init__(
        self,
        S_steps: int = 400,
        T_steps: int = 400,
        S_max_mult: float = 4.0,
    ):
        self.S_steps = S_steps
        self.T_steps = T_steps
        self.S_max_mult = S_max_mult


    def price(self, product: Option) -> FDResult:
        val = self._solve(product, american=_is_american(product))
        eep = None

        if _is_american(product):
            eu_val = self._solve(product, american=False)
            eep = val - eu_val

        return FDResult(
            price=val,
            scheme="Crank-Nicolson",
            S_steps=self.S_steps,
            T_steps=self.T_steps,
            early_exercise_premium=eep,
        )

    def _solve(self, product: Option, american: bool) -> float:
        """
        Core backward PDE solve. Returns interpolated price at S0.
        """
        S0, K, T, r, sigma, q = (
            product.S, product.K, product.T, product.r, product.sigma, product.q,
        )
        is_call = product.option_type == "call"

        S_max = self.S_max_mult * K
        I = self.S_steps
        M = self.T_steps
        dS = S_max / I
        dt = T / M

        # S grid: 0, dS, 2dS, …, S_max  (length I+1)
        S = np.linspace(0, S_max, I + 1)
        i = np.arange(I + 1, dtype=float)

        # Terminal condition
        if is_call:
            V = np.maximum(S - K, 0.0)
        else:
            V = np.maximum(K - S, 0.0)

        # PDE coefficients (interior nodes 1..I-1)
        i_int = i[1:I]

        alpha = 0.25 * dt * (sigma**2 * i_int**2 - (r - q) * i_int)
        beta = -0.5 * dt * (sigma**2 * i_int**2 + r)
        gamma = 0.25 * dt * (sigma**2 * i_int**2 + (r - q) * i_int)

        # Tridiagonal system:
        n_int = I - 1
        ab = np.zeros((3, n_int))
        ab[0, 1:] = -gamma[:-1]   # super-diagonal
        ab[1, :] =  1.0 - beta     # diagonal
        ab[2, :-1] = -alpha[1:]   # sub-diagonal

        # Backward time stepping
        for _ in range(M):
            # Current time boundary conditions
            t_curr = _ * dt 
            tau = T - t_curr  

            if is_call:
                V[0] = 0.0
                V[I] = S_max - K * np.exp(-r * tau)
            else:
                V[0] = K * np.exp(-r * tau)
                V[I] = 0.0

            # RHS vector (explicit half, interior nodes only)
            V_int = V[1:I]
            rhs   = (alpha * V[0:I-1] + (1.0 + beta) * V_int + gamma * V[2:I+1])

            # Adjust RHS for boundary contributions from LHS
            rhs[0] += alpha[0] * V[0]
            rhs[-1] += gamma[-1] * V[I]

            # Solve tridiagonal system
            V_int_new = solve_banded((1, 1), ab, rhs)

            # American early exercise projection
            if american:
                if is_call:
                    intrinsic = np.maximum(S[1:I] - K, 0.0)
                else:
                    intrinsic = np.maximum(K - S[1:I], 0.0)
                V_int_new = np.maximum(V_int_new, intrinsic)

            V[1:I] = V_int_new

        # Interpolate at S0
        return float(np.interp(S0, S, V))

def _is_american(product: Option) -> bool:
    from src.products.american import AmericanOption
    return isinstance(product, AmericanOption)