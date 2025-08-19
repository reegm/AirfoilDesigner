from __future__ import annotations

import numpy as np
from scipy import interpolate, optimize
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import BSpline

from core import config

class BSplineProcessor:
    """
    B-spline processor with true G2 constraint that maintains G1 continuity.
    Works with arbitrary degree B-splines.
    """

    def __init__(self, degree: int = config.DEFAULT_BSPLINE_DEGREE):
        self.upper_control_points: np.ndarray | None = None
        self.lower_control_points: np.ndarray | None = None
        self.upper_knot_vector: np.ndarray | None = None
        self.lower_knot_vector: np.ndarray | None = None
        self.upper_curve: interpolate.BSpline | None = None
        self.lower_curve: interpolate.BSpline | None = None
        self.degree: int = int(degree)
        self.fitted: bool = False
        self.is_sharp_te: bool = False
        self.enforce_g2: bool = True
        self.g2_weight: float = 100.0  # Weight for G2 constraint in optimization

    def fit_bspline(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int,
        is_thickened: bool = False,
        upper_te_tangent_vector: np.ndarray | None = None,
        lower_te_tangent_vector: np.ndarray | None = None,
        enforce_g2: bool = False,
        enforce_te_tangency: bool = True,
    ) -> bool:
        """
        Fit B-splines with G1 and optional G2 constraints at leading edge.
        """
        try:
            self.enforce_g2 = enforce_g2
            num_spans = num_control_points - self.degree
            print(f"[DEBUG] Constrained B-spline: {len(upper_data)} + {len(lower_data)} points -> {num_control_points} CP per surface")
            print(f"[DEBUG] B-spline degree: {self.degree}, spans: {num_spans}")
            print(f"[DEBUG] Using x = u² parametrization with P1.x = 0 constraint (G1)")
            if self.enforce_g2:
                print(f"[DEBUG] Enforcing G2 continuity with single-pass optimization")
            
            # Set trailing edge type
            self.is_sharp_te = not is_thickened
            print(f"[DEBUG] Trailing edge type: {'Sharp' if self.is_sharp_te else 'Blunt'}")
            print(f"[DEBUG] Enforce TE tangency: {enforce_te_tangency}")
            
            # Ensure both surfaces start at the same point
            le_point = (upper_data[0] + lower_data[0]) / 2
            upper_data_corrected = upper_data.copy()
            lower_data_corrected = lower_data.copy()
            upper_data_corrected[0] = le_point
            lower_data_corrected[0] = le_point
            
            # For sharp trailing edge, ensure they end at the same point
            if self.is_sharp_te:
                te_point = np.array([1.0, 0.0])
                upper_data_corrected[-1] = te_point
                lower_data_corrected[-1] = te_point
            else:
                # For blunt trailing edge, use the actual endpoints from input data
                te_point_upper = upper_data_corrected[-1]
                te_point_lower = lower_data_corrected[-1]
            
            # Normalize TE tangent vectors
            upper_te_dir = self._normalize_vector(upper_te_tangent_vector)
            lower_te_dir = self._normalize_vector(lower_te_tangent_vector)

            if self.enforce_g2:
                # Use optimization-based approach for G2
                success = self._fit_with_g2_optimization(
                    upper_data_corrected, lower_data_corrected,
                    num_control_points, upper_te_dir, lower_te_dir, enforce_te_tangency
                )
                if not success:
                    print(f"[DEBUG] G2 optimization failed, falling back to G1-only")
                    self.enforce_g2 = False
            
            if not self.enforce_g2:
                # Use original G1-only fitting
                self._fit_g1_independent(
                    upper_data_corrected, lower_data_corrected,
                    num_control_points, upper_te_dir, lower_te_dir, enforce_te_tangency
                )
            
            # Final cleanup and validation
            self._finalize_curves()
            self.fitted = True
            self._validate_continuity()
            # Validate trailing edge tangents if they were used in fitting
            if upper_te_dir is not None and lower_te_dir is not None and enforce_te_tangency:
                self._validate_trailing_edge_tangents(upper_te_dir, lower_te_dir)
            
            return True
            
        except Exception as e:
            print(f"[DEBUG] B-spline fitting failed: {e}")
            import traceback
            traceback.print_exc()
            self.fitted = False
            return False

    def _fit_with_g2_optimization(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int,
        upper_te_dir: np.ndarray | None,
        lower_te_dir: np.ndarray | None,
        enforce_te_tangency: bool = True,
    ) -> bool:
        """
        Fit both surfaces with G2 continuity using constrained optimization.
        This maintains exact G1 constraints while achieving G2.
        """
        print(f"[DEBUG] G2 Optimization-based fitting")
        print(f"[DEBUG] Enforce TE tangency: {enforce_te_tangency}")
        
        # Get trailing edge points from input data
        te_point_upper = upper_data[-1]
        te_point_lower = lower_data[-1]
        
        print(f"[DEBUG] Upper TE point from input: {te_point_upper}")
        print(f"[DEBUG] Lower TE point from input: {te_point_lower}")
        
        # Create parameter values
        u_params_upper = self._create_parameter_from_x_coords(upper_data)
        u_params_lower = self._create_parameter_from_x_coords(lower_data)
        
        # Create knot vectors
        knot_vector = self._create_knot_vector(num_control_points)
        self.upper_knot_vector = knot_vector.copy()
        self.lower_knot_vector = knot_vector.copy()
        
        # Build basis matrices
        basis_upper = self._build_basis_matrix(u_params_upper, knot_vector)
        basis_lower = self._build_basis_matrix(u_params_lower, knot_vector)
        
        # Number of free variables per surface
        n_fixed = 3  # P0, P1, P2 are partially constrained
        n_free = num_control_points - n_fixed
        
        # Initial guess from G1-only fit
        self._fit_g1_independent(upper_data, lower_data, num_control_points, upper_te_dir, lower_te_dir, enforce_te_tangency)
        
        # Extract initial values for optimization variables
        initial_vars = []
        initial_vars.append(self.upper_control_points[1, 1])  # P1.y_upper
        initial_vars.append(self.lower_control_points[1, 1])  # P1.y_lower
        initial_vars.append(self.upper_control_points[2, 0])  # P2.x_upper
        initial_vars.append(self.upper_control_points[2, 1])  # P2.y_upper
        initial_vars.append(self.lower_control_points[2, 0])  # P2.x_lower
        initial_vars.append(self.lower_control_points[2, 1])  # P2.y_lower
        
        # Add remaining control points
        for i in range(3, num_control_points):
            initial_vars.extend([self.upper_control_points[i, 0], self.upper_control_points[i, 1]])
        for i in range(3, num_control_points):
            initial_vars.extend([self.lower_control_points[i, 0], self.lower_control_points[i, 1]])
        
        initial_vars = np.array(initial_vars)
        
        def objective(vars):
            """Minimize fitting error."""
            # Reconstruct control points from variables
            cp_upper, cp_lower = self._vars_to_control_points(vars, num_control_points)
            
            # Evaluate fitted curves at data points
            curve_upper = interpolate.BSpline(self.upper_knot_vector, cp_upper, self.degree)
            curve_lower = interpolate.BSpline(self.lower_knot_vector, cp_lower, self.degree)
            
            fitted_upper = np.array([curve_upper(u) for u in u_params_upper])
            fitted_lower = np.array([curve_lower(u) for u in u_params_lower])
            
            # Compute fitting error
            error_upper = np.sum((upper_data - fitted_upper)**2)
            error_lower = np.sum((lower_data - fitted_lower)**2)
            
            return error_upper + error_lower
        
        def curvature_constraint(vars):
            """G2 constraint: equal curvatures at leading edge."""
            # Reconstruct control points
            cp_upper, cp_lower = self._vars_to_control_points(vars, num_control_points)
            
            # Compute curvatures at u=0
            kappa_upper = self._compute_curvature_at_zero(cp_upper, self.upper_knot_vector)
            kappa_lower = self._compute_curvature_at_zero(cp_lower, self.lower_knot_vector)
            
            return kappa_upper - kappa_lower
        
        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': curvature_constraint}
        ]
        
        # Add TE endpoint constraints for both sharp and blunt trailing edges
        def te_constraint_upper(vars):
            cp_upper, _ = self._vars_to_control_points(vars, num_control_points)
            return cp_upper[-1] - te_point_upper
        
        def te_constraint_lower(vars):
            _, cp_lower = self._vars_to_control_points(vars, num_control_points)
            return cp_lower[-1] - te_point_lower
        
        constraints.extend([
            {'type': 'eq', 'fun': te_constraint_upper},
            {'type': 'eq', 'fun': te_constraint_lower}
        ])
        
        print(f"[DEBUG] Added trailing edge endpoint constraints")
        print(f"[DEBUG] Upper TE target: {te_point_upper}")
        print(f"[DEBUG] Lower TE target: {te_point_lower}")
        
        # Add trailing edge tangent constraints if selected
        if upper_te_dir is not None and lower_te_dir is not None and enforce_te_tangency:
            def te_tangent_constraint_upper(vars):
                """Constraint for upper surface trailing edge tangent."""
                cp_upper, _ = self._vars_to_control_points(vars, num_control_points)
                computed_tangent = self._compute_tangent_at_trailing_edge(cp_upper, self.upper_knot_vector)
                # Return the difference between computed and desired tangent
                return computed_tangent - upper_te_dir
            
            def te_tangent_constraint_lower(vars):
                """Constraint for lower surface trailing edge tangent."""
                _, cp_lower = self._vars_to_control_points(vars, num_control_points)
                computed_tangent = self._compute_tangent_at_trailing_edge(cp_lower, self.lower_knot_vector)
                # Return the difference between computed and desired tangent
                return computed_tangent - lower_te_dir
            
            constraints.extend([
                {'type': 'eq', 'fun': te_tangent_constraint_upper},
                {'type': 'eq', 'fun': te_tangent_constraint_lower}
            ])
            
            print(f"[DEBUG] Added trailing edge tangent constraints")
            print(f"[DEBUG] Upper TE tangent target: {upper_te_dir}")
            print(f"[DEBUG] Lower TE tangent target: {lower_te_dir}")
        elif upper_te_dir is not None and lower_te_dir is not None and not enforce_te_tangency:
            print(f"[DEBUG] Skipping trailing edge tangent constraints (disabled by user)")
            print(f"[DEBUG] Available TE vectors - Upper: {upper_te_dir}, Lower: {lower_te_dir}")
        
        # Bounds
        bounds = []
        bounds.append((0.001, 0.1))   # P1.y_upper (positive)
        bounds.append((-0.1, -0.001)) # P1.y_lower (negative)
        bounds.append((0.001, 0.5))   # P2.x_upper
        bounds.append((-0.2, 0.2))    # P2.y_upper
        bounds.append((0.001, 0.5))   # P2.x_lower
        bounds.append((-0.2, 0.2))    # P2.y_lower
        
        # Bounds for remaining control points
        for _ in range(n_free):
            bounds.append((None, None))  # x component for upper
            bounds.append((None, None))  # y component for upper
        for _ in range(n_free):
            bounds.append((None, None))  # x component for lower
            bounds.append((None, None))  # y component for lower
        
        # Optimize
        result = optimize.minimize(
            objective, initial_vars,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 200, 'disp': False}
        )
        
        if result.success or result.status == 0:  # Sometimes status 0 is good enough
            print(f"[DEBUG] G2 optimization converged: {result.message}")
            # Extract final control points
            self.upper_control_points, self.lower_control_points = \
                self._vars_to_control_points(result.x, num_control_points)
            
            # Verify G1 constraints are maintained
            print(f"[DEBUG] Final P0_upper: ({self.upper_control_points[0,0]:.6e}, {self.upper_control_points[0,1]:.6e})")
            print(f"[DEBUG] Final P1_upper: ({self.upper_control_points[1,0]:.6e}, {self.upper_control_points[1,1]:.6e})")
            print(f"[DEBUG] Final P0_lower: ({self.lower_control_points[0,0]:.6e}, {self.lower_control_points[0,1]:.6e})")
            print(f"[DEBUG] Final P1_lower: ({self.lower_control_points[1,0]:.6e}, {self.lower_control_points[1,1]:.6e})")
            
            return True
        else:
            print(f"[DEBUG] G2 optimization failed: {result.message}")
            return False

    def _vars_to_control_points(self, vars: np.ndarray, num_control_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert optimization variables to control points maintaining G1 constraints."""
        cp_upper = np.zeros((num_control_points, 2))
        cp_lower = np.zeros((num_control_points, 2))
        
        # Fixed constraints
        cp_upper[0] = [0.0, 0.0]  # P0
        cp_lower[0] = [0.0, 0.0]  # P0 (same as upper)
        
        cp_upper[1] = [0.0, vars[0]]  # P1: x=0 (G1), y from vars
        cp_lower[1] = [0.0, vars[1]]  # P1: x=0 (G1), y from vars
        
        cp_upper[2] = [vars[2], vars[3]]  # P2 from vars
        cp_lower[2] = [vars[4], vars[5]]  # P2 from vars
        
        # Remaining control points
        idx = 6
        for i in range(3, num_control_points):
            cp_upper[i] = vars[idx:idx+2]
            idx += 2
        
        for i in range(3, num_control_points):
            cp_lower[i] = vars[idx:idx+2]
            idx += 2
        
        return cp_upper, cp_lower

    def _compute_curvature_at_zero(self, control_points: np.ndarray, knot_vector: np.ndarray) -> float:
        """Compute curvature at u=0 for given control points."""
        # Create temporary curve
        curve = interpolate.BSpline(knot_vector, control_points, self.degree)
        
        # Get derivatives at u=0
        d1 = curve.derivative(1)(0.0)
        d2 = curve.derivative(2)(0.0)
        
        # Compute curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        norm_d1 = np.linalg.norm(d1)
        
        if norm_d1 > 1e-12:
            return abs(cross) / (norm_d1 ** 3)
        else:
            return 0.0

    def _compute_tangent_at_trailing_edge(self, control_points: np.ndarray, knot_vector: np.ndarray) -> np.ndarray:
        """
        Compute the tangent vector at the trailing edge (u=1) of a B-spline curve.
        
        For a B-spline of degree p, the tangent at u=1 is:
        tangent = p * (P_n - P_{n-1}) / (t_{n+1} - t_{n-p+1})
        
        Args:
            control_points: Control points of the B-spline
            knot_vector: Knot vector of the B-spline
            
        Returns:
            np.ndarray: Tangent vector at the trailing edge
        """
        n = len(control_points) - 1  # Number of control points minus 1
        p = self.degree
        
        # Get the last two control points
        P_n = control_points[-1]
        P_n_minus_1 = control_points[-2]
        
        # Get the relevant knot values
        t_n_plus_1 = knot_vector[n + 1]
        t_n_minus_p_plus_1 = knot_vector[n - p + 1]
        
        # Compute the denominator
        denominator = t_n_plus_1 - t_n_minus_p_plus_1
        
        if abs(denominator) < 1e-12:
            # Fallback: use simple difference
            tangent = P_n - P_n_minus_1
        else:
            # Compute tangent using B-spline formula
            tangent = p * (P_n - P_n_minus_1) / denominator
        
        # Normalize the tangent vector
        norm = np.linalg.norm(tangent)
        if norm > 1e-12:
            tangent = tangent / norm
        
        return tangent

    def _fit_g1_independent(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int,
        upper_te_dir: np.ndarray | None,
        lower_te_dir: np.ndarray | None,
        enforce_te_tangency: bool = True,
    ):
        """Fit surfaces independently with G1 constraint only."""
        print(f"[DEBUG] G1-only independent fitting")
        print(f"[DEBUG] Enforce TE tangency: {enforce_te_tangency}")
        
        # Get trailing edge points from input data
        te_point_upper = upper_data[-1]
        te_point_lower = lower_data[-1]
        
        print(f"[DEBUG] Upper TE point from input: {te_point_upper}")
        print(f"[DEBUG] Lower TE point from input: {te_point_lower}")
        
        # Create parameter values
        u_params_upper = self._create_parameter_from_x_coords(upper_data)
        u_params_lower = self._create_parameter_from_x_coords(lower_data)
        
        # Create knot vectors
        knot_vector = self._create_knot_vector(num_control_points)
        self.upper_knot_vector = knot_vector.copy()
        self.lower_knot_vector = knot_vector.copy()
        
        # Build basis matrices
        basis_upper = self._build_basis_matrix(u_params_upper, knot_vector)
        basis_lower = self._build_basis_matrix(u_params_lower, knot_vector)
        
        # Fit upper surface
        self.upper_control_points = self._fit_single_surface_g1(
            basis_upper, upper_data, num_control_points, is_upper=True, 
            te_tangent_vector=upper_te_dir if enforce_te_tangency else None, te_point=te_point_upper
        )
        
        # Fit lower surface
        self.lower_control_points = self._fit_single_surface_g1(
            basis_lower, lower_data, num_control_points, is_upper=False, 
            te_tangent_vector=lower_te_dir if enforce_te_tangency else None, te_point=te_point_lower
        )

    def _fit_single_surface_g1(
        self,
        basis_matrix: np.ndarray,
        surface_data: np.ndarray,
        num_control_points: int,
        is_upper: bool,
        te_tangent_vector: np.ndarray | None = None,
        te_point: np.ndarray | None = None
    ) -> np.ndarray:
        """Fit single surface with G1 constraint (P0 = origin, P1.x = 0) and optional TE tangent constraint."""
        # Build the constrained least squares system
        # We'll solve for all control points but with constraints
        
        # Data fitting equations
        A_data = np.zeros((2 * len(surface_data), 2 * num_control_points))
        b_data = np.zeros(2 * len(surface_data))
        
        # X-coordinate equations
        A_data[:len(surface_data), :num_control_points] = basis_matrix
        b_data[:len(surface_data)] = surface_data[:, 0]
        
        # Y-coordinate equations
        A_data[len(surface_data):, num_control_points:] = basis_matrix
        b_data[len(surface_data):] = surface_data[:, 1]
        
        # Constraint equations
        constraints = []
        constraint_rhs = []
        
        # P0 = (0, 0)
        row = np.zeros(2 * num_control_points)
        row[0] = 1.0  # P0.x = 0
        constraints.append(row)
        constraint_rhs.append(0.0)
        
        row = np.zeros(2 * num_control_points)
        row[num_control_points] = 1.0  # P0.y = 0
        constraints.append(row)
        constraint_rhs.append(0.0)
        
        # P1.x = 0 (G1 constraint)
        row = np.zeros(2 * num_control_points)
        row[1] = 1.0
        constraints.append(row)
        constraint_rhs.append(0.0)
        
        # Add trailing edge tangent constraint if provided
        if te_tangent_vector is not None:
            
            row = np.zeros(2 * num_control_points)
            # P_n terms
            row[num_control_points - 1] = -te_tangent_vector[1]  # P_n.x coefficient
            row[2 * num_control_points - 1] = te_tangent_vector[0]  # P_n.y coefficient
            # P_{n-1} terms  
            row[num_control_points - 2] = te_tangent_vector[1]  # P_{n-1}.x coefficient
            row[2 * num_control_points - 2] = -te_tangent_vector[0]  # P_{n-1}.y coefficient
            
            constraints.append(row)
            constraint_rhs.append(0.0)
        
        # Add trailing edge endpoint constraint if provided
        if te_point is not None:
            # Constrain the last control point to be equal to te_point
            # X-coordinate constraint
            row_x = np.zeros(2 * num_control_points)
            row_x[num_control_points - 1] = 1.0  # P_n.x coefficient
            constraints.append(row_x)
            constraint_rhs.append(te_point[0])  # x component
            
            # Y-coordinate constraint
            row_y = np.zeros(2 * num_control_points)
            row_y[2 * num_control_points - 1] = 1.0  # P_n.y coefficient
            constraints.append(row_y)
            constraint_rhs.append(te_point[1])  # y component
        
        # Weight for constraints (make them strong)
        constraint_weight = 1000.0
        
        # Build augmented system
        A_constraints = np.array(constraints) * constraint_weight
        b_constraints = np.array(constraint_rhs) * constraint_weight
        
        A_all = np.vstack([A_data, A_constraints])
        b_all = np.hstack([b_data, b_constraints])
        
        # Solve
        solution = np.linalg.lstsq(A_all, b_all, rcond=None)[0]
        
        # Extract control points
        control_points = np.zeros((num_control_points, 2))
        control_points[:, 0] = solution[:num_control_points]
        control_points[:, 1] = solution[num_control_points:]
        
        # Enforce constraints exactly
        control_points[0] = [0.0, 0.0]
        control_points[1, 0] = 0.0
        
        # Ensure correct sign for P1.y
        if is_upper and control_points[1, 1] < 0:
            control_points[1, 1] = abs(control_points[1, 1])
        elif not is_upper and control_points[1, 1] > 0:
            control_points[1, 1] = -abs(control_points[1, 1])
        
        return control_points

    def _finalize_curves(self):
        """Final cleanup and curve rebuilding."""
        # Ensure shared leading edge
        if self.upper_control_points is not None and self.lower_control_points is not None:
            shared_p0 = (self.upper_control_points[0] + self.lower_control_points[0]) / 2
            self.upper_control_points[0] = shared_p0
            self.lower_control_points[0] = shared_p0
            
            # Enforce G1 constraints exactly
            self.upper_control_points[0] = [0.0, 0.0]
            self.lower_control_points[0] = [0.0, 0.0]
            self.upper_control_points[1, 0] = 0.0
            self.lower_control_points[1, 0] = 0.0
            
            # Handle trailing edge - the optimization should have already set the correct endpoints
            # Just verify that they match the expected values
            if self.is_sharp_te:
                # For sharp trailing edge, both should end at (1.0, 0.0)
                te_point = np.array([1.0, 0.0])
                self.upper_control_points[-1] = te_point
                self.lower_control_points[-1] = te_point
            # For blunt trailing edge, the optimization should have already set the correct endpoints
            # from the input data, so we don't need to modify them
        
        # Rebuild curves
        if self.upper_control_points is not None and self.upper_knot_vector is not None:
            self.upper_curve = interpolate.BSpline(
                self.upper_knot_vector, self.upper_control_points, self.degree
            )
        
        if self.lower_control_points is not None and self.lower_knot_vector is not None:
            self.lower_curve = interpolate.BSpline(
                self.lower_knot_vector, self.lower_control_points, self.degree
            )

    def _validate_trailing_edge_tangents(self, upper_te_dir: np.ndarray | None, lower_te_dir: np.ndarray | None) -> None:
        """Validate that the fitted B-splines satisfy the trailing edge tangent constraints."""
        if not self.fitted or self.upper_control_points is None or self.lower_control_points is None:
            return
        
        if upper_te_dir is not None and self.upper_knot_vector is not None:
            computed_upper_tangent = self._compute_tangent_at_trailing_edge(self.upper_control_points, self.upper_knot_vector)
            upper_error = np.linalg.norm(computed_upper_tangent - upper_te_dir)
            print(f"[DEBUG] Upper TE tangent validation:")
            print(f"[DEBUG]   Target: {upper_te_dir}")
            print(f"[DEBUG]   Computed: {computed_upper_tangent}")
            print(f"[DEBUG]   Error: {upper_error:.6f}")
            print(f"[DEBUG]   Satisfied: {upper_error < 1e-3}")
        
        if lower_te_dir is not None and self.lower_knot_vector is not None:
            computed_lower_tangent = self._compute_tangent_at_trailing_edge(self.lower_control_points, self.lower_knot_vector)
            lower_error = np.linalg.norm(computed_lower_tangent - lower_te_dir)
            print(f"[DEBUG] Lower TE tangent validation:")
            print(f"[DEBUG]   Target: {lower_te_dir}")
            print(f"[DEBUG]   Computed: {computed_lower_tangent}")
            print(f"[DEBUG]   Error: {lower_error:.6f}")
            print(f"[DEBUG]   Satisfied: {lower_error < 1e-3}")

    def _validate_continuity(self):
        """Validate G0, G1, and G2 continuity at the leading edge."""
        if not self.fitted or self.upper_curve is None or self.lower_curve is None:
            return
        
        # G0: Position continuity
        pos_upper = self.upper_curve(0.0)
        pos_lower = self.lower_curve(0.0)
        pos_error = np.linalg.norm(pos_upper - pos_lower)
        
        # G1: Tangent continuity
        dt = 1e-8
        tangent_upper = (self.upper_curve(dt) - self.upper_curve(0.0)) / dt
        tangent_lower = (self.lower_curve(dt) - self.lower_curve(0.0)) / dt
        
        tangent_x_error = max(abs(tangent_upper[0]), abs(tangent_lower[0]))
        cross_product = tangent_upper[0] * tangent_lower[1] - tangent_upper[1] * tangent_lower[0]
        
        # G2: Curvature continuity
        kappa_upper = self._compute_curvature_at_zero(self.upper_control_points, self.upper_knot_vector)
        kappa_lower = self._compute_curvature_at_zero(self.lower_control_points, self.lower_knot_vector)
        
        curvature_error = abs(kappa_upper - kappa_lower)
        curvature_relative_error = curvature_error / max(kappa_upper, kappa_lower) if max(kappa_upper, kappa_lower) > 1e-12 else 0
        
        le_radius_upper = 1/kappa_upper if kappa_upper > 1e-12 else float('inf')
        le_radius_lower = 1/kappa_lower if kappa_lower > 1e-12 else float('inf')
        
        print(f"[DEBUG] Continuity validation:")
        print(f"[DEBUG]   G0 (position) error: {pos_error:.2e}")
        print(f"[DEBUG]   G1 (tangent) x-error: {tangent_x_error:.2e}, cross product: {abs(cross_product):.2e}")
        print(f"[DEBUG]   G1 satisfied: {pos_error < 1e-10 and tangent_x_error < 1e-6}")
        
        if self.enforce_g2:
            print(f"[DEBUG]   G2 (curvature) validation:")
            print(f"[DEBUG]     Upper: κ = {kappa_upper:.6f}, r_LE = {le_radius_upper:.6f}")
            print(f"[DEBUG]     Lower: κ = {kappa_lower:.6f}, r_LE = {le_radius_lower:.6f}")
            print(f"[DEBUG]     Absolute error: {curvature_error:.2e}")
            print(f"[DEBUG]     Relative error: {curvature_relative_error:.2%}")
            print(f"[DEBUG]     G2 satisfied: {curvature_error < 1e-4 or curvature_relative_error < 0.01}")
        
        # Show control points
        print(f"[DEBUG] Control points (first 3):")
        for i in range(min(3, len(self.upper_control_points))):
            print(f"[DEBUG]   Upper P{i}: ({self.upper_control_points[i,0]:.6f}, {self.upper_control_points[i,1]:.6f})")
        for i in range(min(3, len(self.lower_control_points))):
            print(f"[DEBUG]   Lower P{i}: ({self.lower_control_points[i,0]:.6f}, {self.lower_control_points[i,1]:.6f})")

    def _normalize_vector(self, vec: np.ndarray | None) -> np.ndarray | None:
        """Normalize a vector."""
        if vec is None:
            return None
        try:
            v = np.asarray(vec, dtype=float)
            n = float(np.hypot(v[0], v[1]))
            if n <= 1e-12:
                return None
            return v / n
        except Exception:
            return None

    def _create_parameter_from_x_coords(self, surface_data: np.ndarray) -> np.ndarray:
        """Create parameter values with blending."""
        x_coords = surface_data[:, 0]
        x_coords = np.clip(x_coords, 0.0, 1.0 - 1e-12)
        u0 = np.sqrt(x_coords)
        #u1 = np.arccos(1.0 - 2.0 * x_coords) / np.pi
        u_params = u0
        return u_params
    
    def _create_knot_vector(self, num_control_points: int) -> np.ndarray:
        """Create clamped knot vector."""
        n = num_control_points - 1
        p = self.degree
        num_interior = n - p
        
        if num_interior <= 0:
            knot_vector = np.concatenate([
                np.zeros(p + 1),
                np.ones(p + 1)
            ])
        else:
            uniform_knots = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]
            knot_vector = np.concatenate([
                np.zeros(p + 1),
                uniform_knots,
                np.ones(p + 1)
            ])
        return knot_vector
    
    def _build_basis_matrix(self, t_values: np.ndarray, knot_vector: np.ndarray) -> np.ndarray:
        """Build B-spline basis matrix."""
        num_points = len(t_values)
        num_basis = len(knot_vector) - self.degree - 1
        basis_matrix = np.zeros((num_points, num_basis))
        
        for i in range(num_basis):
            for j, t in enumerate(t_values):
                basis_matrix[j, i] = self._evaluate_basis_function(i, self.degree, t, knot_vector)
        
        return basis_matrix
    
    def _evaluate_basis_function(self, i: int, degree: int, t: float, knots: np.ndarray) -> float:
        """Evaluate B-spline basis function."""
        t = max(knots[0], min(knots[-1] - 1e-12, t))
        
        if degree == 0:
            if i < len(knots) - 1:
                if knots[i] <= t < knots[i + 1]:
                    return 1.0
                elif i == len(knots) - 2 and abs(t - knots[-1]) < 1e-12:
                    return 1.0
            return 0.0
        
        result = 0.0
        
        if i + degree < len(knots) and abs(knots[i + degree] - knots[i]) > 1e-15:
            alpha1 = (t - knots[i]) / (knots[i + degree] - knots[i])
            if 0 <= i < len(knots) - degree:
                left_basis = self._evaluate_basis_function(i, degree - 1, t, knots)
                result += alpha1 * left_basis
        
        if i + degree + 1 < len(knots) and abs(knots[i + degree + 1] - knots[i + 1]) > 1e-15:
            alpha2 = (knots[i + degree + 1] - t) / (knots[i + degree + 1] - knots[i + 1])
            if 0 <= i + 1 < len(knots) - degree:
                right_basis = self._evaluate_basis_function(i + 1, degree - 1, t, knots)
                result += alpha2 * right_basis
        
        return result
    
    def is_fitted(self) -> bool:
        """Check if B-splines have been successfully fitted."""
        return self.fitted and self.upper_curve is not None and self.lower_curve is not None
    
    def calculate_curvature_comb_data(self, num_points_per_segment=200, scale_factor=0.050):
        """Calculate curvature comb visualization data."""
        if not self.is_fitted():
            return None
        
        all_curves_combs = []
        curves = [self.upper_curve, self.lower_curve]
        
        for curve in curves:
            curve_comb_hairs = []
            t_start = curve.t[curve.k]
            t_end = curve.t[-(curve.k + 1)]
            t_vals = np.linspace(t_start, t_end, num_points_per_segment)
            curve_points = curve(t_vals)
            derivatives_1 = curve.derivative(1)(t_vals)
            derivatives_2 = curve.derivative(2)(t_vals)
            
            x_prime = derivatives_1[:, 0]
            y_prime = derivatives_1[:, 1]
            x_double_prime = derivatives_2[:, 0]
            y_double_prime = derivatives_2[:, 1]
            
            numerator = x_prime * y_double_prime - y_prime * x_double_prime
            denominator = (x_prime**2 + y_prime**2)**(3/2)
            curvatures = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
            
            tangent_norms = np.sqrt(x_prime**2 + y_prime**2)
            unit_tangents = np.zeros_like(derivatives_1)
            valid_tangents = tangent_norms > 1e-12
            unit_tangents[valid_tangents] = derivatives_1[valid_tangents] / tangent_norms[valid_tangents, np.newaxis]
            
            normals = np.zeros_like(unit_tangents)
            normals[:, 0] = -unit_tangents[:, 1]
            normals[:, 1] = unit_tangents[:, 0]
            
            comb_lengths = -curvatures * scale_factor
            end_points = curve_points + normals * comb_lengths[:, np.newaxis]
            
            for j in range(num_points_per_segment):
                hair_segment = np.array([curve_points[j], end_points[j]])
                curve_comb_hairs.append(hair_segment)
            
            all_curves_combs.append(curve_comb_hairs)
        
        return all_curves_combs if all_curves_combs else None

    def apply_te_thickening(self, te_thickness: float) -> bool:
        """
        Apply trailing edge thickening to fitted B-splines.
        Offsets the last control point of each surface by te_thickness/2 in y (upper +, lower -).
        
        Args:
            te_thickness: The thickness to apply at the trailing edge (0.0 to 1.0)
            
        Returns:
            bool: True if thickening was applied successfully, False otherwise
        """
        if not self.fitted or self.upper_control_points is None or self.lower_control_points is None:
            print("[DEBUG] Cannot apply TE thickening: B-splines not fitted")
            return False
        
        if te_thickness <= 0.0:
            print("[DEBUG] TE thickness must be positive")
            return False
        
        try:
            # Create copies of control points
            upper_cp = self.upper_control_points.copy()
            lower_cp = self.lower_control_points.copy()
            
            # Get current trailing edge points
            current_upper_te = upper_cp[-1]
            current_lower_te = lower_cp[-1]
            
            # For blunt trailing edges, we need to be more careful about thickening
            # The thickening should be applied relative to the current endpoints
            if self.is_sharp_te:
                # Sharp trailing edge: both surfaces end at (1.0, 0.0)
                # Simply offset in y direction
                upper_cp[-1, 1] += te_thickness / 2.0
                lower_cp[-1, 1] -= te_thickness / 2.0
            else:
                # Blunt trailing edge: surfaces end at different points
                # Apply thickening by moving both surfaces outward from the centerline
                # Calculate the centerline at the trailing edge
                te_center = (current_upper_te + current_lower_te) / 2.0
                
                # Move upper surface up and lower surface down from the centerline
                upper_cp[-1] = te_center + np.array([0.0, te_thickness / 2.0])
                lower_cp[-1] = te_center - np.array([0.0, te_thickness / 2.0])
            
            # Update the control points
            self.upper_control_points = upper_cp
            self.lower_control_points = lower_cp
            
            # Rebuild the curves with the new control points
            if self.upper_knot_vector is not None:
                self.upper_curve = interpolate.BSpline(
                    self.upper_knot_vector, self.upper_control_points, self.degree
                )
            
            if self.lower_knot_vector is not None:
                self.lower_curve = interpolate.BSpline(
                    self.lower_knot_vector, self.lower_control_points, self.degree
                )
            
            # Update trailing edge type
            self.is_sharp_te = False
            
            print(f"[DEBUG] Applied trailing edge thickening: {te_thickness:.4f}")
            print(f"[DEBUG] Upper TE after thickening: {self.upper_control_points[-1]}")
            print(f"[DEBUG] Lower TE after thickening: {self.lower_control_points[-1]}")
            return True
            
        except Exception as e:
            print(f"[DEBUG] Error applying TE thickening: {e}")
            return False

    def remove_te_thickening(self) -> bool:
        """
        Remove trailing edge thickening by ensuring both surfaces end at the same point.
        
        Returns:
            bool: True if thickening was removed successfully, False otherwise
        """
        if not self.fitted or self.upper_control_points is None or self.lower_control_points is None:
            print("[DEBUG] Cannot remove TE thickening: B-splines not fitted")
            return False
        
        try:
            # Create copies of control points
            upper_cp = self.upper_control_points.copy()
            lower_cp = self.lower_control_points.copy()
            
            # For sharp trailing edges, set both to (1.0, 0.0)
            # For blunt trailing edges, we need to restore the original endpoints
            # Since we don't store the original endpoints, we'll need to refit the B-splines
            # This is a limitation of the current implementation
            
            # For now, assume sharp trailing edge
            te_point = np.array([1.0, 0.0])
            upper_cp[-1] = te_point
            lower_cp[-1] = te_point
            
            # Update the control points
            self.upper_control_points = upper_cp
            self.lower_control_points = lower_cp
            
            # Rebuild the curves with the new control points
            if self.upper_knot_vector is not None:
                self.upper_curve = interpolate.BSpline(
                    self.upper_knot_vector, self.upper_control_points, self.degree
                )
            
            if self.lower_knot_vector is not None:
                self.lower_curve = interpolate.BSpline(
                    self.lower_knot_vector, self.lower_control_points, self.degree
                )
            
            # Update trailing edge type
            self.is_sharp_te = True
            
            print("[DEBUG] Removed trailing edge thickening")
            print(f"[DEBUG] Upper TE after removal: {self.upper_control_points[-1]}")
            print(f"[DEBUG] Lower TE after removal: {self.lower_control_points[-1]}")
            return True
            
        except Exception as e:
            print(f"[DEBUG] Error removing TE thickening: {e}")
            return False

    # Keep all the original fallback methods that might be called
    def _fit_single_surface_constrained(self, surface_data: np.ndarray, num_control_points: int, is_upper_surface: bool = True, te_direction_unit: np.ndarray | None = None):
        """Fallback to single surface fitting if needed."""
        return self._fit_single_surface_g1(
            self._build_basis_matrix(
                self._create_parameter_from_x_coords(surface_data),
                self._create_knot_vector(num_control_points)
            ),
            surface_data,
            num_control_points,
            is_upper_surface
        )