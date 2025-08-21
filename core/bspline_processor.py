from __future__ import annotations

import numpy as np
from scipy import interpolate, optimize
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import BSpline

from core import config
from utils import bspline_helper

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
        # Backup for undoing TE thickening
        self._backup_upper_control_points: np.ndarray | None = None
        self._backup_lower_control_points: np.ndarray | None = None
        self._backup_upper_knot_vector: np.ndarray | None = None
        self._backup_lower_knot_vector: np.ndarray | None = None


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
            upper_te_dir = bspline_helper.normalize_vector(upper_te_tangent_vector)
            lower_te_dir = bspline_helper.normalize_vector(lower_te_tangent_vector)

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
        u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data)
        u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data)
        
        # Create knot vectors
        knot_vector = bspline_helper.create_knot_vector(num_control_points, self.degree)
        self.upper_knot_vector = knot_vector.copy()
        self.lower_knot_vector = knot_vector.copy()
        
        # Build basis matrices
        basis_upper = bspline_helper.build_basis_matrix(u_params_upper, knot_vector, self.degree)
        basis_lower = bspline_helper.build_basis_matrix(u_params_lower, knot_vector, self.degree)
        
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
            kappa_upper = bspline_helper.compute_curvature_at_zero(cp_upper, self.upper_knot_vector, self.degree)
            kappa_lower = bspline_helper.compute_curvature_at_zero(cp_lower, self.lower_knot_vector, self.degree)
            
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
                computed_tangent = bspline_helper.compute_tangent_at_trailing_edge(cp_upper, self.upper_knot_vector, self.degree)
                # Return the difference between computed and desired tangent
                return computed_tangent - upper_te_dir
            
            def te_tangent_constraint_lower(vars):
                """Constraint for lower surface trailing edge tangent."""
                _, cp_lower = self._vars_to_control_points(vars, num_control_points)
                computed_tangent = bspline_helper.compute_tangent_at_trailing_edge(cp_lower, self.lower_knot_vector, self.degree)
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
        u_params_upper = bspline_helper.create_parameter_from_x_coords(upper_data)
        u_params_lower = bspline_helper.create_parameter_from_x_coords(lower_data)
        
        # Create knot vectors
        knot_vector = bspline_helper.create_knot_vector(num_control_points, self.degree)
        self.upper_knot_vector = knot_vector.copy()
        self.lower_knot_vector = knot_vector.copy()
        
        # Build basis matrices
        basis_upper = bspline_helper.build_basis_matrix(u_params_upper, knot_vector, self.degree)
        basis_lower = bspline_helper.build_basis_matrix(u_params_lower, knot_vector, self.degree)
        
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
            computed_upper_tangent = bspline_helper.compute_tangent_at_trailing_edge(self.upper_control_points, self.upper_knot_vector, self.degree)
            upper_error = np.linalg.norm(computed_upper_tangent - upper_te_dir)
            print(f"[DEBUG] Upper TE tangent validation:")
            print(f"[DEBUG]   Target: {upper_te_dir}")
            print(f"[DEBUG]   Computed: {computed_upper_tangent}")
            print(f"[DEBUG]   Error: {upper_error:.6f}")
            print(f"[DEBUG]   Satisfied: {upper_error < 1e-3}")
        
        if lower_te_dir is not None and self.lower_knot_vector is not None:
            computed_lower_tangent = bspline_helper.compute_tangent_at_trailing_edge(self.lower_control_points, self.lower_knot_vector, self.degree)
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
        kappa_upper = bspline_helper.compute_curvature_at_zero(self.upper_control_points, self.upper_knot_vector, self.degree)
        kappa_lower = bspline_helper.compute_curvature_at_zero(self.lower_control_points, self.lower_knot_vector, self.degree)
        
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

    def is_fitted(self) -> bool:
        """Check if B-splines have been successfully fitted."""
        return self.fitted and self.upper_curve is not None and self.lower_curve is not None

    def calculate_curvature_comb_data(self, num_points_per_segment=200, scale_factor=0.050):
        """Calculate curvature comb visualization data."""
        if not self.is_fitted():
            return None
        
        return bspline_helper.calculate_curvature_comb_data(
            self.upper_curve, self.lower_curve, num_points_per_segment, scale_factor
        )

    def apply_te_thickening(self, te_thickness: float) -> bool:
        """
        Apply trailing edge thickening as a post-processing step to the entire airfoil.
        Uses a smooth C2 blend along the chord so that the offset is 0 at the
        leading edge (with zero slope and curvature) and reaches the requested
        thickness at the trailing edge. This preserves the leading-edge tangency
        and changes the curvature distribution as little as possible.
        
        Args:
            te_thickness: The thickness to apply at the trailing edge (0.0 to 1.0)
            
        Returns:
            bool: True if thickening was applied successfully, False otherwise
        """
        if not self.fitted or self.upper_control_points is None or self.lower_control_points is None:
            print("[DEBUG] Cannot apply TE thickening: B-splines not fitted")
            return False
        
        # Allow zero thickness (applies the UI value even if 0)
        if te_thickness < 0.0:
            print("[DEBUG] TE thickness cannot be negative")
            return False
        
        try:
            # Save backups to allow remove_te_thickening to restore
            self._backup_upper_control_points = self.upper_control_points.copy()
            self._backup_lower_control_points = self.lower_control_points.copy()
            self._backup_upper_knot_vector = None if self.upper_knot_vector is None else self.upper_knot_vector.copy()
            self._backup_lower_knot_vector = None if self.lower_knot_vector is None else self.lower_knot_vector.copy()


            # Sample both curves densely in parameter domain
            if self.upper_curve is None or self.lower_curve is None:
                print("[DEBUG] Cannot apply TE thickening: curves not built")
                return False

            num_samples = max(200, int(config.PLOT_POINTS_PER_SURFACE))

            _, upper_pts = bspline_helper.sample_curve(self.upper_curve, num_samples)
            _, lower_pts = bspline_helper.sample_curve(self.lower_curve, num_samples)

            # Blend based on x (already normalized to chord in data loader)
            upper_x = np.clip(upper_pts[:, 0], 0.0, 1.0)
            lower_x = np.clip(lower_pts[:, 0], 0.0, 1.0)
            f_upper = bspline_helper.smoothstep_quintic(upper_x)
            f_lower = bspline_helper.smoothstep_quintic(lower_x)

            half_thickness = 0.5 * te_thickness

            # Apply vertical, smoothly varying offsets; x unchanged
            thick_upper = upper_pts.copy()
            thick_lower = lower_pts.copy()
            thick_upper[:, 1] = thick_upper[:, 1] + half_thickness * f_upper
            thick_lower[:, 1] = thick_lower[:, 1] - half_thickness * f_lower

            # Refit using current number of control points, preserving G1 at LE
            num_control_points = int(self.upper_control_points.shape[0])
            self._fit_g1_independent(
                thick_upper, thick_lower, num_control_points,
                upper_te_dir=None, lower_te_dir=None, enforce_te_tangency=False
            )

            # Mark as blunt before finalizing so finalize does not force sharp TE
            self.is_sharp_te = False
            
            # Finalize and rebuild curves
            self._finalize_curves()
            self.fitted = True

            print(f"[DEBUG] Applied trailing edge thickening (smooth, C2 blend): {te_thickness:.4f}")
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
            # If we have a backup from before thickening, restore it
            if self._backup_upper_control_points is not None and self._backup_lower_control_points is not None:
                self.upper_control_points = self._backup_upper_control_points
                self.lower_control_points = self._backup_lower_control_points
                # Restore knot vectors when available
                if self._backup_upper_knot_vector is not None:
                    self.upper_knot_vector = self._backup_upper_knot_vector
                if self._backup_lower_knot_vector is not None:
                    self.lower_knot_vector = self._backup_lower_knot_vector

                # Rebuild curves
                if self.upper_knot_vector is not None:
                    self.upper_curve = interpolate.BSpline(
                        self.upper_knot_vector, self.upper_control_points, self.degree
                    )
                if self.lower_knot_vector is not None:
                    self.lower_curve = interpolate.BSpline(
                        self.lower_knot_vector, self.lower_control_points, self.degree
                    )

                # After restoring, clear backups
                self._backup_upper_control_points = None
                self._backup_lower_control_points = None
                self._backup_upper_knot_vector = None
                self._backup_lower_knot_vector = None

                # Reset TE type to whatever the restored geometry implies. If last CPs match in y, it's sharp.
                self.is_sharp_te = bool(np.allclose(self.upper_control_points[-1], self.lower_control_points[-1], atol=1e-12))

                print("[DEBUG] Removed trailing edge thickening (restored from backup)")
                print(f"[DEBUG] Upper TE after removal: {self.upper_control_points[-1]}")
                print(f"[DEBUG] Lower TE after removal: {self.lower_control_points[-1]}")
                return True

                        # Fallback: if no backup available, we can't restore
            print("[DEBUG] Cannot remove thickening: no backup available")
            return False
            
        except Exception as e:
            print(f"[DEBUG] Error removing TE thickening: {e}")
            return False