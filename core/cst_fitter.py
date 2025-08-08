from __future__ import annotations

import itertools
import math
import types
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from core import config

__all__ = [
    "CSTFitter",
    "fit_airfoil_cst",
    "generate_cst_airfoil_data",
    "elevate_bernstein_coefficients",
    "elevate_cst_coefficients",
]

# ---------------------------------------------------------------------------
#  Core CST fitter – identical maths, but packaged for reuse
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
        # If we fitted an additional trailing-edge thickness parameter, add the
        # linear TE term so that y(1) = ΔTE for that surface.
        if coeffs.size == self.num_coefficients + 1:  # ΔTE term present
            y += x * coeffs[-1]
        return y

    # ------------------- analytical least‑squares --------------------------
    def lsq(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        fit_te_thickness: bool = False,
        eps: float = 1e-12,
        lambda_reg: float | None = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve the linear least-squares system for CST coefficients.

        Returns (coeffs, metrics_dict) where coeffs has length (degree+1)
        when fit_te_thickness is False, or (degree+2) when True.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        C = self.class_function(x)
        B = self.bernstein_polynomials(x)

        # Solve directly in physical space (Kulfan formulation)
        b = y
        A = C[:, None] * B
        if fit_te_thickness:
            A = np.hstack([A, x[:, None]])

        # Optional second-difference Tikhonov regularization on shape coeffs
        lam = config.CST_DEFAULT_LAMBDA_REG if lambda_reg is None else float(lambda_reg)
        if lam > 0.0:
            # Build D for second differences on the shape coefficients only
            n = self.num_coefficients
            D = np.zeros((n - 2, n)) if n >= 3 else np.zeros((0, n))
            for i in range(max(0, n - 2)):
                D[i, i] = 1.0
                D[i, i + 1] = -2.0
                D[i, i + 2] = 1.0
            if fit_te_thickness:
                # Extend with a zero column for the TE term so it is not penalized
                D = np.hstack([D, np.zeros((D.shape[0], 1))])
            AtA = A.T @ A + (lam * (D.T @ D)) + (eps * np.eye(A.shape[1]))
            Atb = A.T @ b
            coeffs = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        else:
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

        # Compute metrics in physical space using the forward model
        metrics = self.metrics(x, y, coeffs)
        return coeffs, metrics

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
    def sample(self, coeffs: np.ndarray, n: int = 10000):
        x = np.linspace(0, 1, n)
        return x, self.cst_function(x, coeffs)


# ---------------------- Degree promotion helpers ---------------------------
def elevate_bernstein_coefficients(
    shape_coefficients: np.ndarray, from_degree: int, to_degree: int
) -> np.ndarray:
    """Elevate Bernstein polynomial coefficients from one degree to another.

    This performs exact degree elevation for coefficients of a function expressed
    in the Bernstein basis on [0,1]. Repeats single-step elevation until the
    target degree is reached.
    """
    shape = np.asarray(shape_coefficients, float)
    if to_degree < from_degree:
        raise ValueError("to_degree must be >= from_degree for elevation")
    if shape.size != from_degree + 1:
        raise ValueError("shape_coefficients size must equal from_degree + 1")
    if to_degree == from_degree:
        return shape.copy()

    coeffs = shape.copy()
    n = from_degree
    while n < to_degree:
        new = np.empty(n + 2, dtype=float)
        new[0] = coeffs[0]
        for i in range(1, n + 1):
            alpha = i / (n + 1)
            new[i] = alpha * coeffs[i - 1] + (1.0 - alpha) * coeffs[i]
        new[n + 1] = coeffs[n]
        coeffs = new
        n += 1
    return coeffs


def elevate_cst_coefficients(
    coeffs: np.ndarray, from_degree: int, to_degree: int
) -> np.ndarray:
    """Elevate CST coefficients, preserving optional TE thickness term.

    coeffs layout: [shape(0..from_degree), (optional te_thickness)]
    """
    coeffs = np.asarray(coeffs, float)
    has_te = coeffs.size == (from_degree + 2)
    shape = coeffs[: from_degree + 1]
    shape_elev = elevate_bernstein_coefficients(shape, from_degree, to_degree)
    if has_te:
        return np.concatenate([shape_elev, coeffs[-1:]])
    return shape_elev


def fit_airfoil_cst(
    upper_data: np.ndarray,
    lower_data: np.ndarray,
    *,
    degree: int = 8,
    n1: float = 0.5,
    n2: float = 1.0,
    fit_te_thickness: bool = False,
    logger_func: Optional[Callable[[str], None]] = None,
):
    
    def _chebyshev_nodes01(m: int) -> np.ndarray:
        """Chebyshev nodes of the first kind mapped to [0, 1]."""
        k = np.arange(1, m + 1, dtype=float)
        return 0.5 * (1.0 - np.cos((2.0 * k - 1.0) * np.pi / (2.0 * m)))

    def _resample_to_chebyshev(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample y(x) onto Chebyshev-spaced x in [0,1], including endpoints.

        Keeps the number of samples the same by adding 0 and 1 to the Chebyshev set.
        Assumes x is already in [0,1]. Handles non-strictly-monotone input by sorting
        and uniquing x before interpolation.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = x.size
        if m <= 1:
            return x, y
        inner = max(0, m - 2)
        if inner > 0:
            x_new = np.concatenate([[0.0], _chebyshev_nodes01(inner), [1.0]])
        else:
            x_new = np.array([0.0, 1.0])
        # Prepare monotone unique source for interpolation
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        x_unique, idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[idx]
        # Clamp x_new to available range just in case
        xmin = float(x_unique[0])
        xmax = float(x_unique[-1])
        x_new_clamped = np.clip(x_new, xmin, xmax)
        y_new = np.interp(x_new_clamped, x_unique, y_unique)
        return x_new_clamped, y_new

    # Original data
    ux_raw, uy_raw = upper_data[:, 0], upper_data[:, 1]
    lx_raw, ly_raw = lower_data[:, 0], lower_data[:, 1]

    # Resample to Chebyshev-distributed x including endpoints
    ux, uy = _resample_to_chebyshev(ux_raw, uy_raw)
    lx, ly = _resample_to_chebyshev(lx_raw, ly_raw)

    fitter = CSTFitter(n1=n1, n2=n2, degree=degree)

    uc, _ = fitter.lsq(
        ux,
        uy,
        fit_te_thickness=fit_te_thickness,
        lambda_reg=config.CST_DEFAULT_LAMBDA_REG,
    )
    lc, _ = fitter.lsq(
        lx,
        ly,
        fit_te_thickness=fit_te_thickness,
        lambda_reg=config.CST_DEFAULT_LAMBDA_REG,
    )

    umerr = fitter.metrics(ux, uy, uc)
    lmerr = fitter.metrics(lx, ly, lc)

    if logger_func:
        logger_func(
            f"Selected degree={fitter.degree}, n1={fitter.n1:.3f}, n2={fitter.n2:.3f}"
        )
        logger_func("CST fit using Chebyshev x-sampling of the target.")
        logger_func(
            f"Upper CST: RMSE={umerr['rmse']:.6e}, max={umerr['max_error']:.6e} | "
            f"Lower CST: RMSE={lmerr['rmse']:.6e}, max={lmerr['max_error']:.6e}"
        )

    return {
        "upper_coefficients": uc,
        "lower_coefficients": lc,
        "upper_metrics": umerr,
        "lower_metrics": lmerr,
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
