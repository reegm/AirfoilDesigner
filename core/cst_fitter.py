from __future__ import annotations

import itertools
import math
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from core import config

__all__ = [
    "CSTFitter",
    "fit_airfoil_cst",
    "generate_cst_airfoil_data",
]

# ---------------------------------------------------------------------------
#  Core CST fitter
# ---------------------------------------------------------------------------

class CSTFitter:
    """Analytical CST (Class‑Shape Transformation) fitter."""

    def __init__(self, *, n1: float = 0.5, n2: float = 1.0, degree: int = 8):
        self.n1 = float(n1)
        self.n2 = float(n2)
        self.degree = int(degree)
        self.num_coefficients = self.degree + 1  # shape coeffs (ΔTE optional)

    # ------------------------- helpers ------------------------------------
    def class_function(self, x: np.ndarray) -> np.ndarray:
        return np.power(x, self.n1) * np.power(1 - x, self.n2)

    def bernstein_polynomials(self, x: np.ndarray) -> np.ndarray:
        n = self.degree
        B = np.empty((len(x), n + 1))
        for i in range(n + 1):
            B[:, i] = math.comb(n, i) * np.power(x, i) * np.power(1 - x, n - i)
        return B

    # ----------------------- forward model ---------------------------------
    def cst_function(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        coeffs = np.asarray(coeffs, float)
        shape = coeffs[: self.num_coefficients]
        y = self.class_function(x) * (self.bernstein_polynomials(x) @ shape)
        if coeffs.size == self.num_coefficients + 1:  # ΔTE term present
            y += x * coeffs[-1] / 2.0
        return y
    # ------------------- analytical least‑squares --------------------------
    def lsq(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        fit_te_thickness: bool = False,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, float]:
        """Return (coeffs, max_abs_residual)"""
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        C = self.class_function(x)
        B = self.bernstein_polynomials(x)

        edges = (x < 1e-4) | (x > 1-1e-4)      # drop rows where C ≈ 0
        x_fit, y_fit, C_fit = x[~edges], y[~edges], C[~edges]

        y_rhs = y_fit / C_fit                   # true shape values
        Phi   = self.bernstein_polynomials(x_fit)

        if fit_te_thickness:
            Phi = np.hstack([Phi, x_fit[:, None]])

        # Tikhonov term, not doing any good so far
        # lam = 1e-6 if self.degree > 10 else 0.0
        # A   = Phi.T @ Phi + lam*np.eye(Phi.shape[1])
        # b   = Phi.T @ y_rhs
        # coeffs = np.linalg.solve(A, b)

        coeffs, *_ = np.linalg.lstsq(Phi, y_rhs, rcond=None)
        phys_err = np.max(np.abs(self.class_function(x_fit)*(Phi @ coeffs) - y_fit))
        return coeffs, phys_err

    # -------------------------- metrics ------------------------------------
    def metrics(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> dict:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        y_pred = self.cst_function(x, coeffs)
        res = y - y_pred
        mse = float(np.mean(res ** 2))
        xs = np.linspace(0, 1, 20000)
        tree = cKDTree(np.column_stack([xs, self.cst_function(xs, coeffs)]))
        ortho, _ = tree.query(np.column_stack([x, y]))
        return {
            "rmse": math.sqrt(mse),
            "max_error": np.max(np.abs(res)),
            "orthogonal_rmse": math.sqrt(np.mean(ortho ** 2)),
            "orthogonal_max_error": np.max(ortho),
        }

    # -------------------- sampling helper ----------------------------------
    def sample(self, coeffs: np.ndarray, n: int = 400):
        x = np.linspace(0, 1, n)
        return x, self.cst_function(x, coeffs)

# =============================================================================
#  High‑level API with adaptive refinement
# =============================================================================

def fit_airfoil_cst(
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    *,
    degree: int = config.CST_DEFAULT_DEGREE,
    fit_te_thickness: bool = False,
    logger_func: Optional[Callable[[str], None]] = None,
):
    ux, uy = upper_data[:, 0], upper_data[:, 1]
    lx, ly = lower_data[:, 0], lower_data[:, 1]

    fitter = CSTFitter(degree=degree)

    uc, umerr = fitter.lsq(
        ux,
        uy,
        fit_te_thickness=fit_te_thickness,
    )
    lc, lmerr = fitter.lsq(
        lx,
        ly,
        fit_te_thickness=fit_te_thickness,
    )

    worst_err = max(umerr, lmerr)
    if logger_func:
        logger_func(
            f"max={worst_err:.3e}"
        )

    
    umetrics = fitter.metrics(ux, uy, uc)
    lmetrics = fitter.metrics(lx, ly, lc)

    if logger_func:
        logger_func(
            f"Selected degree={fitter.degree}, n1={fitter.n1:.3f}, n2={fitter.n2:.3f}"
        )
        logger_func(
            f"Upper CST: RMSE={umetrics['rmse']:.6e}, max={umetrics['max_error']:.6e} | "
            f"Lower CST: RMSE={lmetrics['rmse']:.6e}, max={lmetrics['max_error']:.6e}"
        )

    return {
        "upper_coefficients": uc,
        "lower_coefficients": lc,
        "upper_metrics": umetrics,
        "lower_metrics": lmetrics,
        "fitter": fitter,
    }

# ---------------------------------------------------------------------------
#  Sampling helper (unchanged)
# ---------------------------------------------------------------------------

def generate_cst_airfoil_data(
    *,
    upper_coefficients: Optional[np.ndarray] = None,
    lower_coefficients: Optional[np.ndarray] = None,
    fitter: CSTFitter,
    num_points: int = 400,
):
    if upper_coefficients is None or lower_coefficients is None:
        raise TypeError("upper_coefficients and lower_coefficients are required")

    xu, yu = fitter.sample(upper_coefficients, n=num_points)
    xl, yl = fitter.sample(lower_coefficients, n=num_points)
    return np.column_stack([xu, yu]), np.column_stack([xl, yl])
