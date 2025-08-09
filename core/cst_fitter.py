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

    def bernstein_polynomials_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of Bernstein basis w.r.t x for degree n on [0,1]."""
        n = self.degree
        dB = np.empty((len(x), n + 1))
        # Using d/dx [C(n,i) x^i (1-x)^{n-i}] = C(n,i)[ i x^{i-1}(1-x)^{n-i} - (n-i) x^i (1-x)^{n-i-1} ]
        for i in range(n + 1):
            term1 = 0.0 if i == 0 else i * np.power(x, i - 1) * np.power(1 - x, n - i)
            term2 = 0.0 if (n - i) == 0 else (n - i) * np.power(x, i) * np.power(1 - x, n - i - 1)
            dB[:, i] = math.comb(n, i) * (term1 - term2)
        return dB

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

    def cst_derivative(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Compute dy/dx for the CST curve y(x)."""
        x = np.asarray(x, float)
        # Avoid singularities at the endpoints for dC/dx
        x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
        coeffs = np.asarray(coeffs, float)
        shape = coeffs[: self.num_coefficients]
        C = self.class_function(x)
        dC = (
            self.n1 * np.power(x_safe, self.n1 - 1) * np.power(1 - x_safe, self.n2)
            - self.n2 * np.power(x_safe, self.n1) * np.power(1 - x_safe, self.n2 - 1)
        )
        B = self.bernstein_polynomials(x)
        dB = self.bernstein_polynomials_derivative(x)
        poly = B @ shape
        dpoly = dB @ shape
        dydx = dC * poly + C * dpoly
        if coeffs.size == self.num_coefficients + 1:
            dydx += coeffs[-1]
        return dydx

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
    def sample(self, coeffs: np.ndarray, n: int = 10000, method: str | None = None):
        """Sample the CST curve with optional endpoint-clustered spacing.

        method:
          - "uniform" (default): x = linspace(0, 1, n)
          - "cosine": x = 0.5 * (1 - cos(pi * t)), clusters near 0 and 1
          - "sqrt-le": x = t**2, clusters near leading edge (x ≈ 0)
        """
        if method is None:
            method = getattr(config, "CST_SAMPLING_METHOD", "uniform")

        t = np.linspace(0.0, 1.0, n)
        if method == "cosine":
            x = 0.5 * (1.0 - np.cos(np.pi * t))
        elif method == "sqrt-le":
            x = t ** 2
        else:
            x = t
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
    te_slope_upper: Optional[float] = None,
    te_slope_lower: Optional[float] = None,
    logger_func: Optional[Callable[[str], None]] = None,
) -> dict:
    """Coupled CST fit with fixed TE points, LE continuity penalty, and orthogonal reweighting.

    Solver options and class exponents are taken from core.config.
    """

    # Extract arrays
    ux, uy = np.asarray(upper_data[:, 0], float), np.asarray(upper_data[:, 1], float)
    lx, ly = np.asarray(lower_data[:, 0], float), np.asarray(lower_data[:, 1], float)

    # Determine trailing edge y-values from the max-x endpoints
    u_te_idx = int(np.argmax(ux)); dte_u = float(uy[u_te_idx])
    l_te_idx = int(np.argmax(lx)); dte_l = float(ly[l_te_idx])

    # Fitter reads exponents from config
    fitter = CSTFitter(n1=config.CST_N1, n2=config.CST_N2, degree=degree)
    nu = fitter.num_coefficients

    # Build design with TE fixed via RHS shift
    C_u = fitter.class_function(ux)
    B_u = fitter.bernstein_polynomials(ux)
    A_u = C_u[:, None] * B_u
    b_u = uy - (ux * dte_u)

    C_l = fitter.class_function(lx)
    B_l = fitter.bernstein_polynomials(lx)
    A_l = C_l[:, None] * B_l
    b_l = ly - (lx * dte_l)

    # Stack coupled system with independent row counts
    A_top = np.hstack([A_u, np.zeros((A_u.shape[0], nu))])
    A_bot = np.hstack([np.zeros((A_l.shape[0], nu)), A_l])
    A = np.vstack([A_top, A_bot])
    bvec = np.concatenate([b_u, b_l])

    # LE radius continuity penalty (always applied, weight from config)
    pen = float(getattr(config, "CST_LE_RADIUS_PENALTY", 0.0) or 0.0)
    AtA = A.T @ A + 1e-12 * np.eye(A.shape[1])
    Atb = A.T @ bvec
    if pen > 0.0:
        L = np.zeros((1, 2 * nu))
        L[0, 0] = 1.0
        L[0, nu + 0] = 1.0
        AtA += pen * (L.T @ L)

    # TE slope equality constraints at x = 1 - eps using provided slopes
    E = []
    d_eq = []
    eps = 1e-6
    t_eq = 1.0 - eps
    if te_slope_upper is not None:
        Ceq = (t_eq ** fitter.n1) * ((1.0 - t_eq) ** fitter.n2)
        dCeq = (
            fitter.n1 * (t_eq ** (fitter.n1 - 1)) * ((1.0 - t_eq) ** fitter.n2)
            - fitter.n2 * (t_eq ** fitter.n1) * ((1.0 - t_eq) ** (fitter.n2 - 1))
        )
        Bu = fitter.bernstein_polynomials(np.array([t_eq]))[0]
        dB = fitter.bernstein_polynomials_derivative(np.array([t_eq]))[0]
        v = dCeq * Bu + Ceq * dB
        row = np.zeros(2 * nu)
        row[0:nu] = v
        E.append(row)
        d_eq.append(te_slope_upper - dte_u)
    if te_slope_lower is not None:
        Ceq = (t_eq ** fitter.n1) * ((1.0 - t_eq) ** fitter.n2)
        dCeq = (
            fitter.n1 * (t_eq ** (fitter.n1 - 1)) * ((1.0 - t_eq) ** fitter.n2)
            - fitter.n2 * (t_eq ** fitter.n1) * ((1.0 - t_eq) ** (fitter.n2 - 1))
        )
        Bl = fitter.bernstein_polynomials(np.array([t_eq]))[0]
        dB = fitter.bernstein_polynomials_derivative(np.array([t_eq]))[0]
        v = dCeq * Bl + Ceq * dB
        row = np.zeros(2 * nu)
        row[nu:2 * nu] = v
        E.append(row)
        d_eq.append(te_slope_lower - dte_l)

    if E:
        E = np.vstack(E)
        d_eq = np.asarray(d_eq, float)
        # KKT solve
        K = np.block([[AtA, E.T], [E, np.zeros((E.shape[0], E.shape[0]))]])
        rhsK = np.concatenate([Atb, d_eq])
        sol = np.linalg.lstsq(K, rhsK, rcond=None)[0]
        xshape = sol[: 2 * nu]
    else:
        xshape = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

    # Orthogonal reweighting (iterations from config)
    orth_iters = int(getattr(config, "CST_ORTHOGONAL_REWEIGHT_ITERS", 0) or 0)
    if orth_iters > 0:
        for _ in range(orth_iters):
            Pu = xshape[0:nu]
            Pl = xshape[nu:2 * nu]
            uc = np.concatenate([Pu, [dte_u]])
            lc = np.concatenate([Pl, [dte_l]])
            su = fitter.cst_derivative(ux, uc)
            sl = fitter.cst_derivative(lx, lc)
            w_u = 1.0 / np.sqrt(1.0 + su * su)
            w_l = 1.0 / np.sqrt(1.0 + sl * sl)
            W = np.diag(np.concatenate([w_u, w_l]))
            AtA_w = (A.T @ W) @ (W @ A) + 1e-12 * np.eye(A.shape[1])
            Atb_w = (A.T @ W) @ (W @ bvec)
            if pen > 0.0:
                AtA_w += pen * (L.T @ L)
            if E:
                K = np.block([[AtA_w, E.T], [E, np.zeros((E.shape[0], E.shape[0]))]])
                rhsK = np.concatenate([Atb_w, d_eq])
                sol = np.linalg.lstsq(K, rhsK, rcond=None)[0]
                xshape = sol[: 2 * nu]
            else:
                xshape = np.linalg.lstsq(AtA_w, Atb_w, rcond=None)[0]

    Pu = xshape[0:nu]
    Pl = xshape[nu:2 * nu]
    uc = np.concatenate([Pu, [dte_u]])
    lc = np.concatenate([Pl, [dte_l]])

    umerr = fitter.metrics(ux, uy, uc)
    lmerr = fitter.metrics(lx, ly, lc)

    if logger_func:
        logger_func(
            f"Selected degree={fitter.degree}, n1={fitter.n1:.3f}, n2={fitter.n2:.3f}"
        )
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
    num_points: int = 20000,
    sampling_method: str | None = None,
):
    if upper_coefficients is None or lower_coefficients is None:
        raise TypeError("upper_coefficients and lower_coefficients are required")

    xu, yu = fitter.sample(upper_coefficients, n=num_points, method=sampling_method)
    xl, yl = fitter.sample(lower_coefficients, n=num_points, method=sampling_method)
    return np.column_stack([xu, yu]), np.column_stack([xl, yl])
