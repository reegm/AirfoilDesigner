from __future__ import annotations

import numpy as np
from scipy import interpolate
from scipy.interpolate import LSQUnivariateSpline


class BSplineProcessor:
    """
    B-spline processor with G1 constraint: second control point at x=0.
    This ensures vertical tangents at leading edge, giving automatic G1 continuity.
    """

    def __init__(self, degree: int = 5):
        self.upper_control_points: np.ndarray | None = None
        self.lower_control_points: np.ndarray | None = None
        self.upper_knot_vector: np.ndarray | None = None
        self.lower_knot_vector: np.ndarray | None = None
        self.upper_curve: interpolate.BSpline | None = None
        self.lower_curve: interpolate.BSpline | None = None
        self.degree: int = int(degree)
        self.fitted: bool = False
        self.is_sharp_te: bool = False  # Track trailing edge type (will be set from thickened parameter)



    def fit_bspline(
        self,
        upper_data: np.ndarray,
        lower_data: np.ndarray,
        num_control_points: int = 10,
        thickened: bool = False,
    ) -> bool:
        """
        Fit B-splines with G1 constraint at leading edge and proper trailing edge constraints.
        """
        try:
            print(f"[DEBUG] Constrained B-spline: {len(upper_data)} + {len(lower_data)} points -> {num_control_points} CP per surface")
            print(f"[DEBUG] Using x = u² parametrization with P1.x = 0 constraint")
            
            # Set trailing edge type from thickened parameter
            self.is_sharp_te = not thickened
            print(f"[DEBUG] Trailing edge type: {'Sharp' if self.is_sharp_te else 'Blunt'} (thickened={thickened})")
            
            # Ensure both surfaces start at the same point
            # Average the leading edge points
            le_point = (upper_data[0] + lower_data[0]) / 2
            upper_data_corrected = upper_data.copy()
            lower_data_corrected = lower_data.copy()
            upper_data_corrected[0] = le_point
            lower_data_corrected[0] = le_point
            
            # For sharp trailing edge, also ensure they end at the same point
            if self.is_sharp_te:
                te_point = np.array([1.0, 0.0])  # Sharp trailing edge at (1, 0)
                upper_data_corrected[-1] = te_point
                lower_data_corrected[-1] = te_point
                print(f"[DEBUG] Sharp TE: Both surfaces end at (1.0, 0.0)")
            else:
                print(f"[DEBUG] Blunt TE: Upper ends at ({upper_data[-1, 0]:.6f}, {upper_data[-1, 1]:.6f})")
                print(f"[DEBUG] Blunt TE: Lower ends at ({lower_data[-1, 0]:.6f}, {lower_data[-1, 1]:.6f})")
            
            # Fit each surface with the constraint
            self.upper_control_points, self.upper_knot_vector, self.upper_curve = \
                self._fit_single_surface_constrained(upper_data_corrected, num_control_points, is_upper_surface=True)
                
            self.lower_control_points, self.lower_knot_vector, self.lower_curve = \
                self._fit_single_surface_constrained(lower_data_corrected, num_control_points, is_upper_surface=False)
            
            # Ensure both P0 points are identical (should already be, but enforce)
            shared_p0 = (self.upper_control_points[0] + self.lower_control_points[0]) / 2
            self.upper_control_points[0] = shared_p0
            self.lower_control_points[0] = shared_p0
            
            # For sharp trailing edge, ensure both last points are identical at (1, 0)
            if self.is_sharp_te:
                te_point = np.array([1.0, 0.0])
                self.upper_control_points[-1] = te_point
                self.lower_control_points[-1] = te_point
            else:
                # For blunt trailing edge, set last control points to actual trailing edge positions
                self.upper_control_points[-1] = upper_data[-1]
                self.lower_control_points[-1] = lower_data[-1]
                print(f"[DEBUG] Blunt TE: Upper Pn set to ({self.upper_control_points[-1, 0]:.6f}, {self.upper_control_points[-1, 1]:.6f})")
                print(f"[DEBUG] Blunt TE: Lower Pn set to ({self.lower_control_points[-1, 0]:.6f}, {self.lower_control_points[-1, 1]:.6f})")
            
            # Rebuild curves with final control points
            self.upper_curve = interpolate.BSpline(self.upper_knot_vector, self.upper_control_points, self.degree)
            self.lower_curve = interpolate.BSpline(self.lower_knot_vector, self.lower_control_points, self.degree)
            
            self.fitted = True
            self._validate_g1_continuity()
            
            return True
            
        except Exception as e:
            print(f"[DEBUG] Constrained B-spline fitting failed: {e}")
            self.fitted = False
            return False

    def _fit_single_surface_constrained(self, surface_data: np.ndarray, num_control_points: int, is_upper_surface: bool = True):
        """
        Fit single surface with built-in constraint: P1 has x=0 (vertical tangent) and trailing edge constraints.
        """
        print(f"[DEBUG] Fitting {'upper' if is_upper_surface else 'lower'} surface with {len(surface_data)} points using x = u² parametrization...")
        
        try:
            return self._fit_with_built_in_constraint(surface_data, num_control_points, is_upper_surface)
        except Exception as e:
            print(f"[DEBUG] Constrained fitting failed: {e}")
            print(f"[DEBUG] Falling back to scipy method with post-constraint...")
            return self._fit_scipy_with_constraint(surface_data, num_control_points, is_upper_surface)

    def _fit_with_built_in_constraint(self, surface_data: np.ndarray, num_control_points: int, is_upper_surface: bool = True):
        """
        Fit B-spline with P1.x = 0 constraint built into the system using x = u² parametrization.
        """
        # Use x = u² parametrization
        u_params = self._create_parameter_from_x_coords(surface_data)
        
        # Create knot vector
        knot_vector = self._create_knot_vector(num_control_points)
        
        # Build basis matrix
        basis_matrix = self._build_basis_matrix(u_params, knot_vector)
        
        # Solve for x-coordinates with constraint
        x_control = self._solve_x_coordinates_constrained(basis_matrix, surface_data[:, 0], num_control_points, surface_data)
        
        # Solve for y-coordinates with tangent direction constraint
        y_control = self._solve_y_coordinates_constrained(basis_matrix, surface_data[:, 1], num_control_points, is_upper_surface, surface_data)
        
        control_points = np.column_stack([x_control, y_control])
        
        # Apply trailing edge constraint
        if self.is_sharp_te:
            control_points[-1] = np.array([1.0, 0.0])
        else:
            # For blunt trailing edge, set last control point to actual trailing edge position
            control_points[-1] = surface_data[-1]
        
        curve = interpolate.BSpline(knot_vector, control_points, self.degree)
        
        print(f"[DEBUG] Built-in constraint method: P0 = ({control_points[0, 0]:.6f}, {control_points[0, 1]:.6f})")
        print(f"[DEBUG] Built-in constraint method: P1 = ({control_points[1, 0]:.6f}, {control_points[1, 1]:.6f})")
        print(f"[DEBUG] Built-in constraint method: Pn = ({control_points[-1, 0]:.6f}, {control_points[-1, 1]:.6f})")
        
        return control_points, knot_vector, curve

    def _create_parameter_from_x_coords(self, surface_data: np.ndarray) -> np.ndarray:
        """
        Create parameter values using x = u² relationship.
        For airfoil data with x ∈ [0,1], we have u = √x.
        """
        x_coords = surface_data[:, 0]
        # Ensure x coordinates are in [0,1] and handle numerical precision
        x_coords = np.clip(x_coords, 0.0, 1.0 - 1e-12)
        # u = √x, so parameter goes from 0 to 1
        u_params = np.sqrt(x_coords)
        return u_params

    def _solve_x_coordinates_constrained(self, basis_matrix: np.ndarray, x_data: np.ndarray, num_control_points: int, surface_data: np.ndarray = None):
        """
        Solve for x-coordinates with P1.x = 0 constraint and trailing edge constraint.
        For x = u² parametrization with cubic B-splines.
        """
        if self.degree == 3:
            # For cubic B-spline with x = u², control points should follow u² pattern
            # but with P1.x = 0 constraint for vertical tangent
            x_control = np.zeros(num_control_points)
            x_control[0] = 0.0  # Leading edge
            x_control[1] = 0.0  # Constraint: vertical tangent
            
            # For remaining points, use the u² relationship
            # Distribute points to approximate u² mapping
            for i in range(2, num_control_points):
                # Distribute remaining points according to u² pattern
                u_val = i / (num_control_points - 1)
                x_control[i] = u_val * u_val
            
            # Fine-tune with constrained least squares to better fit the data
            x_control = self._refine_with_constrained_least_squares(basis_matrix, x_data, x_control)
            
            # Apply trailing edge constraint
            if surface_data is not None:
                if self.is_sharp_te:
                    x_control[-1] = 1.0  # Sharp trailing edge at x=1
                else:
                    x_control[-1] = surface_data[-1, 0]  # Blunt trailing edge at actual position
            
            return x_control
        else:
            # For other degrees, use constrained least squares
            x_control = self._solve_constrained_least_squares(basis_matrix, x_data, num_control_points)
            
            # Apply trailing edge constraint
            if surface_data is not None:
                if self.is_sharp_te:
                    x_control[-1] = 1.0  # Sharp trailing edge at x=1
                else:
                    x_control[-1] = surface_data[-1, 0]  # Blunt trailing edge at actual position
            
            return x_control

    def _refine_with_constrained_least_squares(self, basis_matrix: np.ndarray, x_data: np.ndarray, initial_x_control: np.ndarray):
        """
        Refine the initial x_control points using constrained least squares.
        """
        num_control_points = len(initial_x_control)
        
        # Set up constraint matrix: P0.x = 0, P1.x = 0
        A = basis_matrix.copy()
        b = x_data.copy()
        
        # Add constraint equations
        constraint_matrix = np.zeros((2, num_control_points))
        constraint_matrix[0, 0] = 1.0  # P0.x = 0
        constraint_matrix[1, 1] = 1.0  # P1.x = 0
        
        # Combine data fitting and constraints
        A_constrained = np.vstack([A, constraint_matrix])
        b_constrained = np.hstack([b, [0.0, 0.0]])
        
        # Solve the overdetermined system
        x_control = np.linalg.lstsq(A_constrained, b_constrained, rcond=None)[0]
        
        # Ensure constraints are exactly satisfied
        x_control[0] = 0.0
        x_control[1] = 0.0
        
        return x_control

    def _solve_y_coordinates_constrained(self, basis_matrix: np.ndarray, y_data: np.ndarray, num_control_points: int, is_upper_surface: bool, surface_data: np.ndarray = None):
        """
        Solve for y-coordinates ensuring correct tangent direction and trailing edge constraint.
        Upper surface P1 should have y > 0, lower surface P1 should have y < 0.
        """
        # First solve unconstrained
        y_control = np.linalg.lstsq(basis_matrix, y_data, rcond=None)[0]
        
        # Constrain P0.y = 0 (leading edge at origin)
        y_control[0] = 0.0
        
        # Check and correct P1 tangent direction
        if is_upper_surface:
            if y_control[1] < 0:
                print(f"[DEBUG] Correcting upper surface tangent direction: P1.y {y_control[1]:.6f} -> {abs(y_control[1]):.6f}")
                y_control[1] = abs(y_control[1])
        else:
            if y_control[1] > 0:
                print(f"[DEBUG] Correcting lower surface tangent direction: P1.y {y_control[1]:.6f} -> {-abs(y_control[1]):.6f}")
                y_control[1] = -abs(y_control[1])
        
        # Apply trailing edge constraint
        if surface_data is not None:
            if self.is_sharp_te:
                y_control[-1] = 0.0  # Sharp trailing edge at y=0
            else:
                y_control[-1] = surface_data[-1, 1]  # Blunt trailing edge at actual position
        
        return y_control

    def _solve_constrained_least_squares(self, basis_matrix: np.ndarray, x_data: np.ndarray, num_control_points: int):
        """
        Solve constrained least squares with P0.x = 0 and P1.x = 0.
        """
        # Set up constraint: P0.x = 0, P1.x = 0
        constraint_matrix = np.zeros((2, num_control_points))
        constraint_matrix[0, 0] = 1.0  # P0.x = 0
        constraint_matrix[1, 1] = 1.0  # P1.x = 0
        
        # Add constraints as additional equations
        A_constrained = np.vstack([basis_matrix, constraint_matrix])
        b_constrained = np.hstack([x_data, [0.0, 0.0]])
        
        # Solve the overdetermined system
        x_control = np.linalg.lstsq(A_constrained, b_constrained, rcond=None)[0]
        
        # Ensure constraints are exactly satisfied
        x_control[0] = 0.0
        x_control[1] = 0.0
        
        return x_control

    def _fit_scipy_with_constraint(self, surface_data: np.ndarray, num_control_points: int, is_upper_surface: bool = True):
        """
        Fallback: Use scipy LSQUnivariateSpline and then adjust control points for all constraints.
        """
        # Use x = u² parametrization
        u_params = self._create_parameter_from_x_coords(surface_data)
        
        num_interior_knots = max(0, num_control_points - self.degree - 1)
        if num_interior_knots > 0:
            interior_knots = np.linspace(0.0, 1.0, num_interior_knots + 2)[1:-1]
        else:
            interior_knots = []
        
        # Fit using scipy
        spline_x = LSQUnivariateSpline(u_params, surface_data[:, 0], interior_knots, k=self.degree)
        spline_y = LSQUnivariateSpline(u_params, surface_data[:, 1], interior_knots, k=self.degree)
        
        # Extract and build proper knot vector
        coeffs_x = spline_x.get_coeffs()
        coeffs_y = spline_y.get_coeffs()
        scipy_knots = spline_x.get_knots()
        
        full_knot_vector = np.concatenate([
            np.zeros(self.degree + 1),
            scipy_knots[1:-1],
            np.ones(self.degree + 1)
        ])
        
        control_points = np.column_stack([coeffs_x, coeffs_y])
        
        # Get trailing edge constraints
        te_x = surface_data[-1, 0] if not self.is_sharp_te else 1.0
        te_y = surface_data[-1, 1] if not self.is_sharp_te else 0.0
        
        # Apply constraints: P0.x = 0, P0.y = 0, P1.x = 0, Pn.x = te_x, Pn.y = te_y
        print(f"[DEBUG] Original P0: ({control_points[0, 0]:.6f}, {control_points[0, 1]:.6f})")
        print(f"[DEBUG] Original P1: ({control_points[1, 0]:.6f}, {control_points[1, 1]:.6f})")
        print(f"[DEBUG] Original Pn: ({control_points[-1, 0]:.6f}, {control_points[-1, 1]:.6f})")
        
        control_points[0, 0] = 0.0   # Force x=0 for first control point
        control_points[0, 1] = 0.0   # Force y=0 for first control point
        control_points[1, 0] = 0.0   # Force x=0 for second control point
        control_points[-1, 0] = te_x # Trailing edge x constraint
        control_points[-1, 1] = te_y # Trailing edge y constraint
        
        # Enforce correct tangent direction
        if is_upper_surface:
            if control_points[1, 1] < 0:
                print(f"[DEBUG] Correcting upper surface tangent direction: P1.y {control_points[1, 1]:.6f} -> {abs(control_points[1, 1]):.6f}")
                control_points[1, 1] = abs(control_points[1, 1])
        else:
            if control_points[1, 1] > 0:
                print(f"[DEBUG] Correcting lower surface tangent direction: P1.y {control_points[1, 1]:.6f} -> {-abs(control_points[1, 1]):.6f}")
                control_points[1, 1] = -abs(control_points[1, 1])
        
        print(f"[DEBUG] Constrained P0: ({control_points[0, 0]:.6f}, {control_points[0, 1]:.6f})")
        print(f"[DEBUG] Constrained P1: ({control_points[1, 0]:.6f}, {control_points[1, 1]:.6f})")
        print(f"[DEBUG] Constrained Pn: ({control_points[-1, 0]:.6f}, {control_points[-1, 1]:.6f})")
        
        # Rebuild curve
        bspline_curve = interpolate.BSpline(full_knot_vector, control_points, self.degree)
        
        return control_points, full_knot_vector, bspline_curve

    def _validate_g1_continuity(self):
        """Validate that G1 continuity is achieved and trailing edge constraints are satisfied."""
        if not self.fitted:
            return
            
        # Check position continuity at leading edge
        pos_upper = self.upper_curve(0.0)
        pos_lower = self.lower_curve(0.0)
        pos_error = np.linalg.norm(pos_upper - pos_lower)
        
        # Check tangent continuity (should be automatic since both P1 have x=0)
        dt = 1e-8
        tangent_upper = (self.upper_curve(dt) - self.upper_curve(0.0)) / dt
        tangent_lower = (self.lower_curve(dt) - self.lower_curve(0.0)) / dt
        
        # For vertical tangents, x-components should be ~0
        tangent_x_error = max(abs(tangent_upper[0]), abs(tangent_lower[0]))
        
        # Cross product should be 0 (both tangents are vertical)
        cross_product = tangent_upper[0] * tangent_lower[1] - tangent_upper[1] * tangent_lower[0]
        
        # Check trailing edge constraints
        te_upper = self.upper_curve(1.0)
        te_lower = self.lower_curve(1.0)
        
        if self.is_sharp_te:
            # For sharp TE, both surfaces should end at (1, 0)
            te_error_upper = np.linalg.norm(te_upper - np.array([1.0, 0.0]))
            te_error_lower = np.linalg.norm(te_lower - np.array([1.0, 0.0]))
            te_position_error = np.linalg.norm(te_upper - te_lower)
        else:
            # For blunt TE, check if control points match the expected trailing edge positions
            te_error_upper = np.linalg.norm(te_upper - self.upper_control_points[-1])
            te_error_lower = np.linalg.norm(te_lower - self.lower_control_points[-1])
            te_position_error = np.linalg.norm(te_upper - te_lower)  # This is expected to be > 0 for blunt TE
        
        print(f"[DEBUG] G1 continuity validation:")
        print(f"[DEBUG]   Position continuity error: {pos_error:.2e}")
        print(f"[DEBUG]   Tangent x-component error: {tangent_x_error:.2e}")
        print(f"[DEBUG]   Cross product: {abs(cross_product):.2e}")
        print(f"[DEBUG]   G1 satisfied: {pos_error < 1e-10 and tangent_x_error < 1e-6}")
        
        print(f"[DEBUG] Trailing edge validation:")
        print(f"[DEBUG]   TE type: {'Sharp' if self.is_sharp_te else 'Blunt'}")
        print(f"[DEBUG]   Upper TE error: {te_error_upper:.2e}")
        print(f"[DEBUG]   Lower TE error: {te_error_lower:.2e}")
        if self.is_sharp_te:
            print(f"[DEBUG]   TE position error: {te_position_error:.2e}")
            print(f"[DEBUG]   Sharp TE satisfied: {te_position_error < 1e-10}")
        
        # Show control points for verification
        print(f"[DEBUG] Control points:")
        print(f"[DEBUG]   Upper P0: ({self.upper_control_points[0, 0]:.6f}, {self.upper_control_points[0, 1]:.6f})")
        print(f"[DEBUG]   Upper P1: ({self.upper_control_points[1, 0]:.6f}, {self.upper_control_points[1, 1]:.6f})")
        print(f"[DEBUG]   Upper Pn: ({self.upper_control_points[-1, 0]:.6f}, {self.upper_control_points[-1, 1]:.6f})")
        print(f"[DEBUG]   Lower P0: ({self.lower_control_points[0, 0]:.6f}, {self.lower_control_points[0, 1]:.6f})")
        print(f"[DEBUG]   Lower P1: ({self.lower_control_points[1, 0]:.6f}, {self.lower_control_points[1, 1]:.6f})")
        print(f"[DEBUG]   Lower Pn: ({self.lower_control_points[-1, 0]:.6f}, {self.lower_control_points[-1, 1]:.6f})")
        
        # Show complete control point arrays for comparison with STEP export
        print(f"[DEBUG] Complete upper control points ({len(self.upper_control_points)} points):")
        for i, cp in enumerate(self.upper_control_points):
            print(f"[DEBUG]   Upper P{i}: ({cp[0]:.6f}, {cp[1]:.6f})")
        
        print(f"[DEBUG] Complete lower control points ({len(self.lower_control_points)} points):")
        for i, cp in enumerate(self.lower_control_points):
            print(f"[DEBUG]   Lower P{i}: ({cp[0]:.6f}, {cp[1]:.6f})")

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
            interior_knots = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]
            knot_vector = np.concatenate([
                np.zeros(p + 1),
                interior_knots,
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