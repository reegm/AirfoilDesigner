"""
CST Processor for integrating CST fitting with the existing airfoil processing pipeline.

This module provides CST fitting as an intermediary step before Bezier optimization.
"""

import numpy as np
import math
from typing import Optional, Dict, Any, Tuple
from PySide6.QtCore import QObject, Signal

from core import config
from core.cst_fitter import (
    CSTFitter,
    fit_airfoil_cst,
    generate_cst_airfoil_data,
    elevate_cst_coefficients,
)
from core.error_functions import calculate_single_bezier_fitting_error


class CSTProcessor(QObject):
    """
    Processor for CST fitting that integrates with the existing airfoil processing pipeline.
    
    This acts as an intermediary step between raw airfoil data and Bezier optimization.
    """
    
    # Signals for GUI communication
    log_message = Signal(str)
    plot_update_requested = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # CST fitting parameters
        self.degree = 15
        self.n1 = config.CST_N1
        self.n2 = config.CST_N2
        self.bluntTE = False
        
        # Fitting results
        self.cst_fitter = None
        self.upper_coefficients = None
        self.lower_coefficients = None
        self.upper_metrics = None
        self.lower_metrics = None
        
        # Generated CST data
        self.cst_upper_data = None
        self.cst_lower_data = None
        
        # Original data (for comparison)
        self.original_upper_data = None
        self.original_lower_data = None
        
    def set_parameters(self, degree: int = 8, n1: float = 0.5, n2: float = 1.0, blunt_TE = False):
        """
        Set CST fitting parameters.
        
        Args:
            degree: Degree of Bernstein polynomials
            n1: Class function parameter for leading edge
            n2: Class function parameter for trailing edge
        """
        self.degree = degree
        self.n1 = n1
        self.n2 = n2
        self.blunt_TE = blunt_TE
        
    def fit_airfoil(self, upper_data: np.ndarray, lower_data: np.ndarray, blunt_TE: bool) -> bool:
        """
        Fit CST functions to airfoil data.
        
        Args:
            upper_data: Upper surface coordinates (N, 2)
            lower_data: Lower surface coordinates (N, 2)
            
        Returns:
            True if fitting was successful, False otherwise
        """
        try:
            self.original_upper_data = upper_data.copy()
            self.original_lower_data = lower_data.copy()
            self.blunt_TE = blunt_TE
            
            # Perform CST fitting
            # result = fit_airfoil_cst(
            #     upper_data=upper_data,
            #     lower_data=lower_data,
            #     degree=self.degree,
            #     n1=self.n1,
            #     n2=self.n2,
            #     logger_func=self.log_message.emit
            # )

            # If we already have coefficients and only degree increased, elevate instead of refitting
            can_promote = (
                self.cst_fitter is not None and
                self.upper_coefficients is not None and
                self.lower_coefficients is not None and
                self.cst_fitter.degree <= self.degree
            )

            if can_promote and self.cst_fitter.degree < self.degree:
                old_deg = self.cst_fitter.degree
                # Elevate shape coefficients to new degree (preserve optional TE term)
                uc_new = elevate_cst_coefficients(self.upper_coefficients, old_deg, self.degree)
                lc_new = elevate_cst_coefficients(self.lower_coefficients, old_deg, self.degree)
                # Update fitter and evaluate metrics on original data
                self.cst_fitter = CSTFitter(n1=self.n1, n2=self.n2, degree=self.degree)
                self.upper_coefficients = uc_new
                self.lower_coefficients = lc_new
                ux, uy = upper_data[:, 0], upper_data[:, 1]
                lx, ly = lower_data[:, 0], lower_data[:, 1]
                self.upper_metrics = self.cst_fitter.metrics(ux, uy, self.upper_coefficients)
                self.lower_metrics = self.cst_fitter.metrics(lx, ly, self.lower_coefficients)
                self.log_message.emit(f"Promoted CST degree from {old_deg} to {self.degree} without refit.")
            else:
                result = fit_airfoil_cst(
                    upper_data=upper_data,
                    lower_data=lower_data,
                    degree=self.degree,           
                    fit_te_thickness=blunt_TE,       # leave as-is unless you need open TE
                    logger_func=self.log_message.emit
                )

                # Store results
                self.cst_fitter = result['fitter']
                self.upper_coefficients = result['upper_coefficients']
                self.lower_coefficients = result['lower_coefficients']
                self.upper_metrics = result['upper_metrics']
                self.lower_metrics = result['lower_metrics']

            # Generate CST data points
            self.cst_upper_data, self.cst_lower_data = generate_cst_airfoil_data(
                upper_coefficients=self.upper_coefficients,
                lower_coefficients=self.lower_coefficients,
                fitter=self.cst_fitter,
                num_points=1000  # Generate many more points for high-fidelity representation
            )
            
            self.log_message.emit("CST fitting completed successfully.")
            return True
            
        except Exception as e:
            self.log_message.emit(f"CST fitting failed: {e}")
            return False
    
    def get_cst_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the generated CST airfoil data.
        
        Returns:
            Tuple of (upper_data, lower_data) as (N, 2) arrays
        """
        if self.cst_upper_data is None or self.cst_lower_data is None:
            raise ValueError("CST data not available. Run fit_airfoil() first.")
        
        return self.cst_upper_data, self.cst_lower_data
    
    def get_fitting_metrics(self) -> Dict[str, Any]:
        """
        Get the fitting metrics for both surfaces.
        
        Returns:
            Dictionary containing upper and lower surface metrics
        """
        if self.upper_metrics is None or self.lower_metrics is None:
            raise ValueError("Fitting metrics not available. Run fit_airfoil() first.")
        
        return {
            'upper': self.upper_metrics,
            'lower': self.lower_metrics
        }

    # ------------------------------------------------------------------
    # Degree-9 Bézier approximation of CST (orthogonal-weighted, constrained)
    # ------------------------------------------------------------------
    def _bernstein_row(self, u: float, n: int) -> np.ndarray:
        """Bernstein basis row of degree n evaluated at u."""
        row = np.empty(n + 1, dtype=float)
        one_minus_u = 1.0 - u
        for i in range(n + 1):
            row[i] = math.comb(n, i) * (u ** i) * (one_minus_u ** (n - i))
        return row

    def _chebyshev_nodes01(self, m: int) -> np.ndarray:
        """Chebyshev nodes of the first kind mapped to [0, 1]."""
        k = np.arange(1, m + 1, dtype=float)
        return 0.5 * (1.0 - np.cos((2.0 * k - 1.0) * math.pi / (2.0 * m)))

    def _cst_eval(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        return self.cst_fitter.cst_function(np.asarray(x, float), coeffs)

    def _cst_slope(self, x: np.ndarray, coeffs: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Numerical dy/dx for CST curve using finite differences on [0,1]."""
        x = np.asarray(x, float)
        xh1 = np.clip(x + h, 0.0, 1.0)
        xh0 = np.clip(x - h, 0.0, 1.0)
        yh1 = self._cst_eval(xh1, coeffs)
        yh0 = self._cst_eval(xh0, coeffs)
        denom = np.maximum(xh1 - xh0, 1e-12)
        return (yh1 - yh0) / denom

    def _build_degree9_bezier_for_surface(self, coeffs: np.ndarray, p: int = 2, samples: int = 200, lambda_reg: float = 0.0) -> np.ndarray:
        """
        Build a single-span degree-9 Bézier polygon approximating the CST curve
        using x(u) = u^p (fixed in x), and solving for y(u) with orthogonal-weighted
        constrained least squares enforcing y(0), y(1), and TE slope.

        Returns: (10,2) array of control points.
        """
        n = 9
        # Fixed x-control points to represent x(u) = u^p exactly in degree-9 Bernstein basis
        X = np.zeros(n + 1, dtype=float)
        if not (0 <= p <= n):
            raise ValueError("p must be between 0 and 9 for degree-9 Bézier")
        denom = math.comb(n, p)
        for i in range(p, n + 1):
            X[i] = math.comb(i, p) / denom

        # Targets and weights
        m = int(samples)
        u = self._chebyshev_nodes01(m)
        x = u ** p
        y = self._cst_eval(x, coeffs)
        slopes = self._cst_slope(x, coeffs)
        w_ortho = 1.0 / np.sqrt(1.0 + slopes ** 2)  # project vertical to approximate orthogonal

        # Build design matrix Phi for y(u) in Bernstein basis
        Phi = np.vstack([self._bernstein_row(ui, n) for ui in u])  # (m, n+1)

        # Equality constraints: y(0), y(1), and TE slope at u=1
        y0 = float(self._cst_eval(np.array([0.0]), coeffs)[0])
        y1 = float(self._cst_eval(np.array([1.0]), coeffs)[0])
        dy_dx_te = float(self._cst_slope(np.array([1.0]), coeffs)[0])
        dy_du_te = dy_dx_te * p  # since x(u)=u^p -> dx/du = p at u=1

        C = np.zeros((3, n + 1), dtype=float)
        d = np.zeros(3, dtype=float)
        # y(0) = Y_0
        C[0, 0] = 1.0
        d[0] = y0
        # y(1) = Y_n
        C[1, n] = 1.0
        d[1] = y1
        # n*(Y_n - Y_{n-1}) = dy_du_te
        C[2, n] = float(n)
        C[2, n - 1] = float(-n)
        d[2] = dy_du_te

        # Weighted least squares with equality constraints via KKT system
        W = np.diag(w_ortho)
        A = W @ Phi
        b = W @ y
        # Regularization for stability
        # Add a mild second-difference penalty on Y to suppress zig-zag
        D = np.zeros((n - 1, n + 1))
        for i in range(n - 1):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        AtA = A.T @ A + (1e-12 * np.eye(n + 1)) + (float(lambda_reg) * (D.T @ D))
        Atb = A.T @ b
        KKT_top = np.hstack([AtA, C.T])
        KKT_bottom = np.hstack([C, np.zeros((C.shape[0], C.shape[0]))])
        KKT = np.vstack([KKT_top, KKT_bottom])
        rhs = np.concatenate([Atb, d])
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        Y = sol[: n + 1]

        ctrl = np.column_stack([X, Y])
        return ctrl

    def build_degree9_beziers_from_cst(self, p: Any = 2, samples: int = 200, lambda_reg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Build upper and lower degree-9 Bézier polygons from current CST fit.
        
        p can be:
        - int: same power-law for both surfaces
        - tuple[int, int]: (p_upper, p_lower)
        - 'auto' or None: choose p per surface from a small candidate set by minimizing orthogonal max error vs original data
        """
        if self.cst_fitter is None or self.upper_coefficients is None or self.lower_coefficients is None:
            raise RuntimeError("CST fit not available. Run fit_airfoil() first.")

        # Resolve p per surface
        def _choose_p(coeffs: np.ndarray, original_data: np.ndarray) -> tuple[int, np.ndarray, float]:
            candidates = [1, 2, 3]
            best = None
            best_ctrl = None
            best_err = float("inf")
            for cand in candidates:
                ctrl = self._build_degree9_bezier_for_surface(coeffs, p=cand, samples=samples, lambda_reg=lambda_reg)
                # Evaluate orthogonal max error against original data
                _, max_err, _ = calculate_single_bezier_fitting_error(
                    bezier_poly=ctrl,
                    original_data=original_data,
                    error_function="orthogonal",
                    return_max_error=True,
                )
                if max_err < best_err:
                    best = cand
                    best_ctrl = ctrl
                    best_err = max_err
            return best, best_ctrl, best_err

        if p is None or p == 'auto':
            pu, upper_poly, upper_err = _choose_p(self.upper_coefficients, self.original_upper_data)
            pl, lower_poly, lower_err = _choose_p(self.lower_coefficients, self.original_lower_data)
            self.log_message.emit(f"Auto-selected p: upper={pu}, lower={pl} (max orth err: {upper_err:.3e} / {lower_err:.3e})")
        elif isinstance(p, tuple) and len(p) == 2:
            pu, pl = int(p[0]), int(p[1])
            upper_poly = self._build_degree9_bezier_for_surface(self.upper_coefficients, p=pu, samples=samples, lambda_reg=lambda_reg)
            lower_poly = self._build_degree9_bezier_for_surface(self.lower_coefficients, p=pl, samples=samples, lambda_reg=lambda_reg)
        else:
            pu = pl = int(p)
            upper_poly = self._build_degree9_bezier_for_surface(self.upper_coefficients, p=pu, samples=samples, lambda_reg=lambda_reg)
            lower_poly = self._build_degree9_bezier_for_surface(self.lower_coefficients, p=pl, samples=samples, lambda_reg=lambda_reg)

        # Debug dump to terminal if enabled
        if config.DEBUG_WORKER_LOGGING:
            try:
                import numpy as _np
                _np.set_printoptions(precision=6, suppress=True)
                def _summarize(ctrl: _np.ndarray, label: str):
                    sec_diff_y = _np.diff(ctrl[:, 1], n=2)
                    max_sec = float(_np.max(_np.abs(sec_diff_y))) if sec_diff_y.size else 0.0
                    print(f"[Deg9] {label} control points (x, y):\n{ctrl}")
                    print(f"[Deg9] {label} Y second-difference max |Δ²Y| = {max_sec:.6e}")
                _summarize(upper_poly, "Upper")
                _summarize(lower_poly, "Lower")
            except Exception as _e:
                # Keep GUI robust even if terminal print fails
                self.log_message.emit(f"Debug dump failed: {_e}")

        self.log_message.emit(f"Built degree-9 Bézier from CST with p_upper={pu}, p_lower={pl}, lambda={lambda_reg}.")
        return upper_poly, lower_poly
    
    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the CST coefficients for both surfaces.
        
        Returns:
            Tuple of (upper_coefficients, lower_coefficients)
        """
        if self.upper_coefficients is None or self.lower_coefficients is None:
            raise ValueError("CST coefficients not available. Run fit_airfoil() first.")
        
        return self.upper_coefficients, self.lower_coefficients
    
    def request_plot_update(self):
        """
        Request a plot update showing the CST fit results.
        """
        if self.cst_upper_data is None:
            self.log_message.emit("No CST data available for plotting.")
            return
        
        plot_data = {}
        
        if self.original_upper_data is not None:
            plot_data['original_upper'] = self.original_upper_data
        if self.original_lower_data is not None:
            plot_data['original_lower'] = self.original_lower_data
        
        if self.cst_upper_data is not None:
            plot_data['cst_upper'] = self.cst_upper_data
            plot_data['cst_lower'] = self.cst_lower_data
        
        # Add metrics to plot data
        if self.upper_metrics and self.lower_metrics:
            plot_data['cst_metrics'] = {
                'upper_rmse': self.upper_metrics['rmse'],
                'upper_max_error': self.upper_metrics['max_error'],
                'upper_orthogonal_max_error': self.upper_metrics['orthogonal_max_error'],
                'lower_rmse': self.lower_metrics['rmse'],
                'lower_max_error': self.lower_metrics['max_error'],
                'lower_orthogonal_max_error': self.lower_metrics['orthogonal_max_error']
            }
        
        self.plot_update_requested.emit(plot_data)
    
    def clear_data(self):
        """Clear all stored data."""
        self.cst_fitter = None
        self.upper_coefficients = None
        self.lower_coefficients = None
        self.upper_metrics = None
        self.lower_metrics = None
        self.cst_upper_data = None
        self.cst_lower_data = None
        self.original_upper_data = None
        self.original_lower_data = None
    
    def is_fitted(self) -> bool:
        """Check if CST fitting has been performed."""
        return (self.cst_upper_data is not None and 
                self.cst_lower_data is not None and
                self.cst_fitter is not None) 