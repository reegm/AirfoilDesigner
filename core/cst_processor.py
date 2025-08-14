"""
CST Processor for integrating CST fitting with the existing airfoil processing pipeline.

This module provides CST fitting as an intermediary step before Bezier optimization.
"""

import numpy as np
import math
from typing import Optional, Dict, Any, Tuple
from PySide6.QtCore import QObject, Signal, QThread, QTimer
import multiprocessing as mp
from multiprocessing import Queue
import time
from scipy.spatial import cKDTree

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
        # CST comb parameters
        self._cst_comb_scale = 0.05
        self._cst_comb_density = 40
        
        # Original data (for comparison)
        self.original_upper_data = None
        self.original_lower_data = None
        
        # Worker process management
        self._worker_process = None
        self._worker_queue = None
        self._worker_timer = None
        self._fitting_in_progress = False
        
    def set_parameters(self, degree: int = 8, n1: float = 0.5, n2: float = 1.0, blunt_TE = False):
        """
        Set CST fitting parameters.
        
        Args:
            degree: Degree of Bernstein polynomials
            n1: Class function parameter for leading edge
            n2: Class function parameter for trailing edge
        """
        self.degree = degree
        # n1/n2 are now taken from config inside the fitter; keep local for UI display only
        self.n1 = n1
        self.n2 = n2
        self.blunt_TE = blunt_TE
        
    def fit_airfoil(self, upper_data: np.ndarray, lower_data: np.ndarray, blunt_TE: bool) -> bool:
        """
        Fit CST functions to airfoil data using worker process.
        
        Args:
            upper_data: Upper surface coordinates (N, 2)
            lower_data: Lower surface coordinates (N, 2)
            blunt_TE: Whether trailing edge is blunt
            
        Returns:
            True if fitting was successful, False otherwise
        """
        if self._fitting_in_progress:
            self.log_message.emit("CST fitting already in progress...")
            return False
            
        try:
            self.original_upper_data = upper_data.copy()
            self.original_lower_data = lower_data.copy()
            self.blunt_TE = blunt_TE
            
            # Clear previous results
            self.cst_fitter = None
            self.upper_coefficients = None
            self.lower_coefficients = None
            self.upper_metrics = None
            self.lower_metrics = None
            self.cst_upper_data = None
            self.cst_lower_data = None
            
            # Compute TE slopes from stored unit tangent vectors if available
            te_slope_upper = None
            te_slope_lower = None
            try:
                ut = getattr(self.parent(), 'processor', None)
                if ut is not None:
                    uvec = getattr(ut, 'upper_te_tangent_vector', None)
                    lvec = getattr(ut, 'lower_te_tangent_vector', None)
                    if uvec is not None and abs(float(uvec[0])) > 1e-12:
                        te_slope_upper = float(uvec[1]) / float(uvec[0])
                    if lvec is not None and abs(float(lvec[0])) > 1e-12:
                        te_slope_lower = float(lvec[1]) / float(lvec[0])
            except Exception:
                te_slope_upper = None
                te_slope_lower = None

            # Start worker process
            self._start_cst_fitting_worker(
                upper_data=upper_data,
                lower_data=lower_data,
                degree=self.degree,
                te_slope_upper=te_slope_upper,
                te_slope_lower=te_slope_lower
            )
            
            self.log_message.emit("CST fitting started in background...")
            return True
            
        except Exception as e:
            self.log_message.emit(f"Failed to start CST fitting: {e}")
            self._fitting_in_progress = False
            return False
    
    def _start_cst_fitting_worker(self, upper_data, lower_data, degree, te_slope_upper=None, te_slope_lower=None, 
                                  override_te_upper=None, override_te_lower=None):
        """Start the CST fitting worker process."""
        from gui.controllers.optimization_worker import _cst_fitting_worker
        
        # Clean up any existing worker
        self._cleanup_worker()
        
        # Create queue for communication
        self._worker_queue = Queue()
        
        # Prepare arguments for worker
        args = (
            upper_data,
            lower_data,
            degree,
            te_slope_upper,
            te_slope_lower,
            override_te_upper,
            override_te_lower,
            []  # logger_messages placeholder
        )
        
        # Start worker process
        self._worker_process = mp.Process(target=_cst_fitting_worker, args=(args, self._worker_queue))
        self._worker_process.start()
        self._fitting_in_progress = True
        
        # Set up timer to check for results
        self._worker_timer = QTimer()
        self._worker_timer.timeout.connect(self._check_worker_results)
        self._worker_timer.start(100)  # Check every 100ms
        
        self.log_message.emit("CST fitting worker started...")
    
    def _check_worker_results(self):
        """Check for results from the worker process."""
        if self._worker_queue is None:
            return
            
        try:
            # Check if there are any messages in the queue
            while True:
                try:
                    message = self._worker_queue.get_nowait()
                    
                    if message["type"] == "log":
                        self.log_message.emit(message["message"])
                    elif message["type"] == "result":
                        self._handle_worker_result(message)
                except:
                    # No more messages in queue
                    break
                    
        except Exception as e:
            if self._worker_queue is not None:  # Only log if queue should exist
                self.log_message.emit(f"Error checking worker results: {e}")
    
    def _handle_worker_result(self, message):
        """Handle the final result from the worker process."""
        try:
            if message["success"]:
                result = message["result"]
                
                # Store results
                self.cst_fitter = result['fitter']
                self.upper_coefficients = result['upper_coefficients']
                self.lower_coefficients = result['lower_coefficients']
                self.upper_metrics = result['upper_metrics']
                self.lower_metrics = result['lower_metrics']

                # Generate CST data points
                num_pts = int(getattr(config, "CST_SAMPLING_POINTS", 4000))
                self.cst_upper_data, self.cst_lower_data = generate_cst_airfoil_data(
                    upper_coefficients=self.upper_coefficients,
                    lower_coefficients=self.lower_coefficients,
                    fitter=self.cst_fitter,
                    num_points=num_pts,
                    sampling_method=getattr(config, "CST_SAMPLING_METHOD", "cosine"),
                )
                
                self.log_message.emit("CST fitting completed successfully.")
                
                # Request plot update to show results
                self.request_plot_update()
                
            else:
                self.log_message.emit(f"CST fitting failed: {message['error']}")
                
        except Exception as e:
            self.log_message.emit(f"Error handling worker result: {e}")
        finally:
            self._cleanup_worker()
    
    def _cleanup_worker(self):
        """Clean up worker process and related resources."""
        if self._worker_timer:
            self._worker_timer.stop()
            self._worker_timer = None
            
        if self._worker_process and self._worker_process.is_alive():
            self._worker_process.terminate()
            self._worker_process.join(timeout=1.0)
            if self._worker_process.is_alive():
                self._worker_process.kill()
            self._worker_process = None
            
        self._worker_queue = None
        self._fitting_in_progress = False
    
    def thicken_cst_with_worker(self, te_mm: float, chord_mm: float, degree: int):
        """Thicken CST using worker process with TE override."""
        if self._fitting_in_progress:
            self.log_message.emit("CST fitting already in progress...")
            return False
            
        try:
            te_chord = te_mm / chord_mm
            
            # Compute per-surface TE targets from original data sign at TE
            udata = self.original_upper_data
            ldata = self.original_lower_data
            if udata is None or ldata is None:
                raise ValueError("No original data available")
                
            ux, uy = udata[:, 0], udata[:, 1]
            lx, ly = ldata[:, 0], ldata[:, 1]
            u_te_idx = int(np.argmax(ux))
            l_te_idx = int(np.argmax(lx))
            sign_u = 1.0 if uy[u_te_idx] >= 0.0 else -1.0
            sign_l = -1.0 if ly[l_te_idx] <= 0.0 else 1.0
            te_u = sign_u * (0.5 * te_chord)
            te_l = sign_l * (0.5 * te_chord)

            # Get TE slopes
            te_slope_upper = None
            te_slope_lower = None
            try:
                ut = getattr(self.parent(), 'processor', None)
                if ut is not None:
                    uvec = getattr(ut, 'upper_te_tangent_vector', None)
                    lvec = getattr(ut, 'lower_te_tangent_vector', None)
                    if uvec is not None and abs(float(uvec[0])) > 1e-12:
                        te_slope_upper = float(uvec[1]) / float(uvec[0])
                    if lvec is not None and abs(float(lvec[0])) > 1e-12:
                        te_slope_lower = float(lvec[1]) / float(lvec[0])
            except Exception:
                te_slope_upper = None
                te_slope_lower = None

            # Start worker with overrides
            self._start_cst_fitting_worker(
                upper_data=udata,
                lower_data=ldata,
                degree=degree,
                te_slope_upper=te_slope_upper,
                te_slope_lower=te_slope_lower,
                override_te_upper=te_u,
                override_te_lower=te_l
            )
            
            self.log_message.emit(f"CST thickening started: TE={te_mm:.3f} mm (±{te_mm/2:.3f} per surface)")
            return True
            
        except Exception as e:
            self.log_message.emit(f"Failed to start CST thickening: {e}")
            self._fitting_in_progress = False
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

    def _build_degree9_bezier_for_surface(self, coeffs: np.ndarray, p: int = 2, samples: int = 20000, lambda_reg: float = 0.0) -> np.ndarray:
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
    
    def cst_to_exact_bezier_control_points(self, coeffs: np.ndarray, delta_zle: float = 0.0) -> np.ndarray:
        """
        Convert CST coefficients to exact Bézier control points using Marshall's method.
        
        CRITICAL FIX: The Marshall paper creates a curve in parameter s, but we need
        the final curve in the original parameter t where t² = s.
        
        Args:
            coeffs: CST shape function coefficients (length n+1)  
            delta_zle: Trailing edge offset (typically 0.0 for airfoils)
            
        Returns:
            (m+1, 2) array of Bézier control points where m = 2*n + 3
        """
        # Determine CST shape degree and extract optional TE thickness from coeffs
        # coeffs layout from fitter: [shape_0..shape_n, (optional ΔTE)]
        if self.cst_fitter is not None:
            n = int(self.cst_fitter.degree)
        else:
            # Fallback: infer assuming no TE thickness term
            n = len(coeffs) - 1
        shape_len = n + 1
        has_te = len(coeffs) == (shape_len + 1)
        te_thickness = float(coeffs[-1]) if has_te else float(delta_zle or 0.0)

        m = 2 * n + 3        # Bézier degree

        # Step 1: Convert CST shape function S(u) from Bernstein to power basis
        # S(u) = sum_{i=0}^n coeffs[i] * B_{i,n}(u) = sum_{k=0}^n a_k u^k
        # a_k = sum_{i=0}^k coeffs[i] * C(n,i) * C(n-i, k-i) * (-1)^(k-i)
        a = np.zeros(n + 1)
        for k in range(n + 1):
            s = 0.0
            for i in range(0, k + 1):
                s += float(coeffs[i]) * math.comb(n, i) * math.comb(n - i, k - i) * ((-1) ** (k - i))
            a[k] = s

        # Step 2: Build polynomial coefficients in s-domain (Marshall Eq. 11b)
        # ζ(s) = a0*s + s^2*Δζ_te + Σ_{i=1..n} (a_i - a_{i-1}) s^(2i+1) - a_n s^(2n+3)
        b = np.zeros(m + 1)
        if m >= 1:
            b[1] = a[0]
        if m >= 2:
            b[2] = te_thickness
        for i in range(1, n + 1):
            power = 2 * i + 1
            if power <= m:
                b[power] = a[i] - a[i - 1]
        final_power = 2 * n + 3
        if final_power <= m:
            b[final_power] -= a[n]

        # Step 3: Keep the polynomial in the s-parameter.
        # Build monomial coefficients for ξ(s) and ζ(s) directly.
        # ξ(s) = s^2 -> d_xi[2] = 1
        d_xi = np.zeros(m + 1)
        d_xi[2] = 1.0
        # ζ(s) = Σ b_i s^i
        d_zeta = b.copy()

        # Step 4: Convert monomial coefficients to Bézier control points of degree m
        # Using t^j = Σ_{i=j..m} [C(i,j)/C(m,j)] B_{i,m}(t) ⇒ q_i = Σ_{j=0..i} d_j * C(i,j)/C(m,j)
        control_points = np.zeros((m + 1, 2))
        for i in range(m + 1):
            # x-coordinate
            x_sum = 0.0
            for j in range(0, i + 1):
                if d_xi[j] != 0.0:
                    x_sum += d_xi[j] * (math.comb(i, j) / math.comb(m, j))
            control_points[i, 0] = x_sum
            # y-coordinate
            y_sum = 0.0
            for j in range(0, i + 1):
                if d_zeta[j] != 0.0:
                    y_sum += d_zeta[j] * (math.comb(i, j) / math.comb(m, j))
            control_points[i, 1] = y_sum

        # Enforce exact trailing-edge slope in x-space: dy/dx at x=1
        # Using Marshall relations, dy/dx|_{x=1} = ΔTE - Σ a_k
        # For Bézier in parameter s: y'(1) = m (Y_m - Y_{m-1}) and x'(1) = m (X_m - X_{m-1}) = 2
        # ⇒ Y_{m-1} = Y_m - (2/m) * (dy/dx)|_{x=1}
        # Prefer using the fitter's analytical dy/dx near x=1 for robustness
        try:
            if self.cst_fitter is not None:
                eps = 1e-9
                te_slope_x = float(self.cst_fitter.cst_derivative(np.array([1.0 - eps]), np.asarray(coeffs, float))[0])
            else:
                raise RuntimeError
        except Exception:
            # Fallback to Marshall closed form
            te_slope_x = (te_thickness - float(np.sum(a)))
        y_m = float(control_points[-1, 1])
        control_points[-2, 1] = y_m - (2.0 * te_slope_x) / float(m)

        return control_points
        
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
            # Build curvature comb for CST curves using current UI params
            plot_data['comb_cst'] = self._build_cst_curvature_comb(
                self.cst_upper_data,
                self.cst_lower_data,
                scale_factor=self._cst_comb_scale,
                density=self._cst_comb_density,
            )
            # Compute worst orthogonal error markers against original data
            try:
                if self.original_upper_data is not None and self.cst_upper_data is not None:
                    tree_u = cKDTree(self.cst_upper_data)
                    d_u, _ = tree_u.query(self.original_upper_data)
                    idx_u = int(np.argmax(d_u))
                    plot_data['worst_cst_upper_max_error'] = float(d_u[idx_u])
                    plot_data['worst_cst_upper_max_error_idx'] = idx_u
                if self.original_lower_data is not None and self.cst_lower_data is not None:
                    tree_l = cKDTree(self.cst_lower_data)
                    d_l, _ = tree_l.query(self.original_lower_data)
                    idx_l = int(np.argmax(d_l))
                    plot_data['worst_cst_lower_max_error'] = float(d_l[idx_l])
                    plot_data['worst_cst_lower_max_error_idx'] = idx_l
            except Exception:
                # Keep plotting robust; markers are optional
                pass
        
      
        self.plot_update_requested.emit(plot_data)

    def _build_cst_curvature_comb(self, upper_curve: np.ndarray, lower_curve: np.ndarray, *, scale_factor: float = 0.05, density: int = 40):
        """Compute curvature comb hairs for sampled CST curves (upper and lower).

        Returns a list [hairs_upper, hairs_lower], where each entry is a list of
        2-point segments (np.ndarray shape (2,2)).
        """
        def comb_from_polyline(curve_xy: np.ndarray):
            if curve_xy is None or len(curve_xy) < 3:
                return []
            
            xy = np.asarray(curve_xy, float)
            # Central differences for first and second derivatives in arc-length parameterization approximation
            # Use chordwise x as parameter; stable enough for visualization.
            x = xy[:, 0]
            y = xy[:, 1]
            # First derivatives
            dx = np.gradient(x)
            dy = np.gradient(y)
            # Second derivatives
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            # Signed curvature κ = (x'y'' − y'x'') / (x'^2 + y'^2)^(3/2)
            denom_base = np.maximum(dx * dx + dy * dy, 1e-16)
            denom = np.power(denom_base, 1.5)
            kappa = (dx * ddy - dy * ddx) / denom
            # Unit normals from tangents (rotate by +90 degrees)
            norm = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-16))
            tx = dx / norm
            ty = dy / norm
            nx = -ty
            ny = tx
            # Comb end points; choose lengths so convex regions point inward
            # inward ≈ along −sign(κ)·n (since sign(κ)·n gives outward for our parameterization)
            lengths = -kappa * scale_factor
            ends = xy + np.column_stack([nx, ny]) * lengths[:, None]
            # Downsample to requested density by uniform index sampling
            m = int(max(2, density))
            idx = np.linspace(0, xy.shape[0] - 1, m).astype(int)
            idx = np.unique(idx)
            hairs = []
            for j in idx:
                hairs.append(np.array([xy[j], ends[j]]))
            return hairs

    def request_cst_comb_update(self, scale: float, density: int) -> None:
        """Update CST comb parameters and re-emit plot with updated comb."""
        try:
            self._cst_comb_scale = float(scale)
            self._cst_comb_density = int(density)
        except Exception:
            pass
        # Re-emit using current stored CST data and coefficients
        self.request_plot_update()
    
    def clear_data(self):
        """Clear all stored data."""
        # Clean up worker first
        self._cleanup_worker()
        
        self.cst_fitter = None
        self.upper_coefficients = None
        self.lower_coefficients = None
        self.upper_metrics = None
        self.lower_metrics = None
        self.cst_upper_data = None
        self.cst_lower_data = None
        self.original_upper_data = None
        self.original_lower_data = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self._cleanup_worker()
        except Exception:
            pass
    
    def is_fitted(self) -> bool:
        """Check if CST fitting has been performed."""
        return (self.cst_upper_data is not None and 
                self.cst_lower_data is not None and
                self.cst_fitter is not None)
    
    def is_fitting_in_progress(self) -> bool:
        """Check if CST fitting is currently in progress."""
        return self._fitting_in_progress 