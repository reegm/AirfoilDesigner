import numpy as np
from scipy.optimize import minimize, minimize_scalar
from utils.bezier_utils import general_bezier_curve, leading_edge_curvature, bezier_curvature
from scipy.special import comb
import logging
from core import config
from functools import lru_cache
import weakref

# Cache for binomial coefficients to avoid repeated calculations
_binomial_cache = {}

def get_binomial_coeff(n, k):
    """Get binomial coefficient with caching for performance."""
    key = (n, k)
    if key not in _binomial_cache:
        _binomial_cache[key] = comb(n, k, exact=True)
    return _binomial_cache[key]

# Pre-compute common binomial coefficients for typical Bezier orders
def _precompute_binomials():
    """Pre-compute binomial coefficients for common Bezier orders."""
    for n in range(1, 12):  # Support up to order 11
        for k in range(n + 1):
            get_binomial_coeff(n, k)

_precompute_binomials()

class BezierEvaluator:
    """Cached Bezier curve evaluator for performance optimization."""
    
    def __init__(self, control_points):
        self.control_points = np.asarray(control_points)
        self.n = len(control_points) - 1
        self._bernstein_cache = {}
        self._derivative_cache = {}
        
    def _get_bernstein_coeffs(self, t):
        """Get Bernstein coefficients for given t with caching."""
        if t not in self._bernstein_cache:
            coeffs = np.zeros(self.n + 1)
            for i in range(self.n + 1):
                coeffs[i] = get_binomial_coeff(self.n, i) * (1 - t)**(self.n - i) * t**i
            self._bernstein_cache[t] = coeffs
        return self._bernstein_cache[t]
    
    def _get_derivative_coeffs(self, t):
        """Get derivative coefficients for given t with caching."""
        if t not in self._derivative_cache:
            coeffs = np.zeros(self.n)
            for i in range(self.n):
                coeffs[i] = get_binomial_coeff(self.n - 1, i) * (1 - t)**(self.n - 1 - i) * t**i
            self._derivative_cache[t] = coeffs
        return self._derivative_cache[t]
    
    def evaluate(self, t):
        """Evaluate Bezier curve at parameter t."""
        coeffs = self._get_bernstein_coeffs(t)
        return np.sum(coeffs[:, None] * self.control_points, axis=0)
    
    def evaluate_derivative(self, t):
        """Evaluate first derivative at parameter t."""
        coeffs = self._get_derivative_coeffs(t)
        diff_points = np.diff(self.control_points, axis=0)
        return self.n * np.sum(coeffs[:, None] * diff_points, axis=0)
    
    def evaluate_second_derivative(self, t):
        """Evaluate second derivative at parameter t."""
        if self.n < 2:
            return np.zeros(2)
        
        coeffs = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            coeffs[i] = get_binomial_coeff(self.n - 2, i) * (1 - t)**(self.n - 2 - i) * t**i
        
        diff2_points = np.diff(self.control_points, n=2, axis=0)
        return self.n * (self.n - 1) * np.sum(coeffs[:, None] * diff2_points, axis=0)

def calculate_orthogonal_distance_to_bezier_optimized(point, control_points, initial_t_guess=None, max_iterations=15, tolerance=1e-8):
    """
    Optimized version of orthogonal distance calculation with caching and better convergence.
    """
    evaluator = BezierEvaluator(control_points)
    point = np.asarray(point)
    
    def try_newton_raphson(t_init):
        """Optimized Newton-Raphson with early termination."""
        t = np.clip(t_init, 0, 1)
        
        for iteration in range(max_iterations):
            curve_point = evaluator.evaluate(t)
            first_derivative = evaluator.evaluate_derivative(t)
            second_derivative = evaluator.evaluate_second_derivative(t)
            
            # Vector from curve point to target point
            diff_vector = point - curve_point
            
            # Function: f(t) = (P - B(t)) · B'(t) = 0 for orthogonal distance
            f = np.dot(diff_vector, first_derivative)
            
            # Derivative: f'(t) = -B'(t) · B'(t) + (P - B(t)) · B''(t)
            f_prime = -np.dot(first_derivative, first_derivative) + np.dot(diff_vector, second_derivative)
            
            # Avoid division by zero
            if abs(f_prime) < 1e-12:
                break
            
            # Newton step with adaptive damping
            step = f / f_prime
            damping = min(1.0, 0.5 / max(abs(step), 1e-6))
            
            t_new = t - damping * step
            t_new = np.clip(t_new, 0, 1)
            
            # Check convergence
            if abs(t_new - t) < tolerance:
                t = t_new
                break
            
            t = t_new
        
        # Evaluate final distance
        curve_point = evaluator.evaluate(t)
        distance = np.linalg.norm(point - curve_point)
        return distance, t, curve_point
    
    # Smart initial guess based on x-coordinate
    if initial_t_guess is not None:
        candidates = [initial_t_guess]
    else:
        x_min, x_max = control_points[0, 0], control_points[-1, 0]
        if x_max > x_min:
            x_based_guess = np.clip((point[0] - x_min) / (x_max - x_min), 0, 1)
            candidates = [x_based_guess]
        else:
            candidates = [0.5]
    
    # Reduced number of candidates for better performance
    candidates.extend([0.0, 0.5, 1.0])
    candidates = list(set(candidates))  # Remove duplicates
    
    # Try each candidate and keep the best result
    best_distance = float('inf')
    best_t = 0.5
    best_curve_point = None
    
    for t_init in candidates:
        try:
            dist, t_opt, curve_pt = try_newton_raphson(t_init)
            if dist < best_distance:
                best_distance = dist
                best_t = t_opt
                best_curve_point = curve_pt
        except:
            continue
    
    # Fallback: coarse sampling if Newton-Raphson fails
    if best_distance == float('inf') or best_distance > 1.0:
        t_samples = np.linspace(0, 1, 50)  # Reduced from 100
        distances = []
        for t_sample in t_samples:
            curve_pt = evaluator.evaluate(t_sample)
            dist = np.linalg.norm(point - curve_pt)
            distances.append(dist)
        
        min_idx = np.argmin(distances)
        best_distance = distances[min_idx]
        best_t = t_samples[min_idx]
        best_curve_point = evaluator.evaluate(best_t)
    
    return best_distance, best_t, best_curve_point

def calculate_all_orthogonal_distances_optimized(data_points, control_points):
    """
    Optimized version that calculates all orthogonal distances with better initial guesses.
    """
    evaluator = BezierEvaluator(control_points)
    data_points = np.asarray(data_points)
    n_points = len(data_points)
    
    distances = np.zeros(n_points)
    t_values = np.zeros(n_points)
    curve_points = np.zeros((n_points, 2))
    
    # Pre-compute x-range for better initial guesses
    x_min, x_max = control_points[0, 0], control_points[-1, 0]
    x_range = x_max - x_min
    
    for i, point in enumerate(data_points):
        # Smart initial guess based on x-coordinate and point index
        if x_range > 0:
            initial_t = np.clip((point[0] - x_min) / x_range, 0, 1)
        else:
            initial_t = i / max(n_points - 1, 1)
        
        distance, t_opt, curve_point = calculate_orthogonal_distance_to_bezier_optimized(
            point, control_points, initial_t_guess=initial_t
        )
        
        distances[i] = distance
        t_values[i] = t_opt
        curve_points[i] = curve_point
    
    max_distance = np.max(distances)
    max_distance_idx = np.argmax(distances)
    
    return distances, max_distance, max_distance_idx, t_values, curve_points

def calculate_orthogonal_distance_to_bezier(point, control_points, initial_t_guess=None, max_iterations=20, tolerance=1e-8):
    """
    Calculate the orthogonal distance from a point to a Bezier curve using multiple starting points
    for robustness.
    
    Args:
        point: np.array of shape (2,) - the point to measure distance from
        control_points: np.array of shape (n+1, 2) - Bezier control points
        initial_t_guess: float - initial guess for parameter t (0 <= t <= 1)
        max_iterations: int - maximum Newton-Raphson iterations
        tolerance: float - convergence tolerance
    
    Returns:
        tuple: (distance, optimal_t, curve_point)
    """
    return calculate_orthogonal_distance_to_bezier_optimized(point, control_points, initial_t_guess, max_iterations, tolerance)

def calculate_all_orthogonal_distances(data_points, control_points):
    """
    Calculate orthogonal distances from all data points to a Bezier curve.
    
    Args:
        data_points: np.array of shape (N, 2) - points to measure distances from
        control_points: np.array of shape (n+1, 2) - Bezier control points
    
    Returns:
        tuple: (distances, max_distance, max_distance_idx, t_values, curve_points)
    """
    return calculate_all_orthogonal_distances_optimized(data_points, control_points)

def calculate_orthogonal_error_minmax(control_points, original_data):
    """
    Calculate the maximum orthogonal distance error for minmax optimization.
    
    Args:
        control_points: np.array of shape (n+1, 2) - Bezier control points
        original_data: np.array of shape (N, 2) - original airfoil data points
    
    Returns:
        tuple: (max_error, max_error_idx) or float (max_error) depending on usage
    """
    distances, max_distance, max_distance_idx, _, _ = calculate_all_orthogonal_distances(
        original_data, control_points
    )
    return max_distance, max_distance_idx

def calculate_icp_error(data_points, curve_points, return_max_error=False):
    """
    Calculates the sum of squared Euclidean distances from each data point to the closest point on the curve.
    If return_max_error is True, also returns the maximum pointwise error (not squared) and its index.
    Args:
        data_points (np.ndarray): (N, 2) array of data points.
        curve_points (np.ndarray): (M, 2) array of points sampled along the curve.
        return_max_error (bool): If True, also return the maximum pointwise error and its index.
    Returns:
        float or (float, float, int): Sum of squared distances, and optionally the max pointwise error and its index.
    """
    # Vectorized distance calculation
    dists = np.linalg.norm(data_points[:, None, :] - curve_points[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    sum_sq = np.sum(min_dists ** 2)
    if return_max_error:
        max_error = np.max(min_dists)
        max_error_idx = np.argmax(min_dists)
        return sum_sq, max_error, max_error_idx
    return sum_sq

def calculate_single_bezier_fitting_error(bezier_poly, original_data, error_function="icp", return_max_error=False, num_points_curve_error=None, use_curvature_sampling=False):
    """
    Calculates the fitting error between a Bezier curve and the original data using the specified error function.
    Supported error functions: 'icp', 'orthogonal_minmax'
    If return_max_error is True, returns (sum, max_error, max_error_idx).
    """
    if error_function == "orthogonal_minmax":
        max_error, max_error_idx = calculate_orthogonal_error_minmax(bezier_poly, original_data)
        if return_max_error:
            return max_error, max_error, max_error_idx  # For minmax, sum and max are the same
        return max_error
    
    else:  # ICP (default)
        if num_points_curve_error is None:
            num_points_curve = config.NUM_POINTS_CURVE_ERROR
        else:
            num_points_curve = num_points_curve_error
        
        t_samples = np.linspace(0, 1, num_points_curve)
        
        curve_points = general_bezier_curve(t_samples, bezier_poly)
        curve_sorted = curve_points[np.argsort(curve_points[:, 0])]
        return calculate_icp_error(original_data, curve_sorted, return_max_error=return_max_error)

def resample_points_by_curvature(points, num_samples=200):
    """
    Resample a 2D point set (Nx2) so that points are more densely distributed in regions of high curvature.
    Uses chord-length parameterization and Bezier curvature as a proxy for local curvature.
    Returns a new (num_samples, 2) array.
    """
    points = np.asarray(points)
    if len(points) < 3 or num_samples <= len(points):
        # Not enough points or already dense, just return original (or linearly interpolated)
        t_orig = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, num_samples)
        return np.column_stack([
            np.interp(t_new, t_orig, points[:, 0]),
            np.interp(t_new, t_orig, points[:, 1])
        ])

    # Chord-length parameterization
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.zeros(len(points))
    t[1:] = np.cumsum(dists)
    t /= t[-1]

    # Fit a Bezier curve to the points for curvature estimation
    # Use degree = min(7, len(points)-1) for stability
    from numpy.polynomial.polynomial import Polynomial
    degree = min(7, len(points)-1)
    # Fit x(t) and y(t) polynomials (not a true Bezier, but sufficient for curvature proxy)
    px = Polynomial.fit(t, points[:, 0], degree)
    py = Polynomial.fit(t, points[:, 1], degree)
    t_dense = np.linspace(0, 1, max(400, 4*num_samples))
    x_dense = px(t_dense)
    y_dense = py(t_dense)
    curve_dense = np.column_stack([x_dense, y_dense])

    # Estimate curvature using finite differences
    dx = np.gradient(x_dense, t_dense)
    dy = np.gradient(y_dense, t_dense)
    ddx = np.gradient(dx, t_dense)
    ddy = np.gradient(dy, t_dense)
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5 + 1e-12
    curvature = np.abs(numerator / denominator)
    curvature += np.finfo(float).eps

    # PDF and CDF for adaptive sampling
    pdf = curvature / np.sum(curvature)
    cdf = np.cumsum(pdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    u_values = np.linspace(0, 1, num_samples)
    t_resampled = np.interp(u_values, cdf, t_dense)
    # Interpolate original points at these t values
    x_resampled = np.interp(t_resampled, t, points[:, 0])
    y_resampled = np.interp(t_resampled, t, points[:, 1])
    return np.column_stack([x_resampled, y_resampled])

# Cache for fixed x-coordinates to avoid repeated array creation
_paper_fixed_x_coords_cache = {
    'upper': np.array([0.0, 0.0, 0.11422, 0.25294, 0.37581, 0.49671, 0.61942, 0.74701, 0.88058, 1.0]),
    'lower': np.array([0.0, 0.0, 0.12325, 0.25314, 0.37519, 0.49569, 0.61975, 0.74391, 0.87391, 1.0])
}

def median_x_control_points(original_data, num_control_points):
    """
    Compute the x-locations for Bezier control points as the median x-value of each segment of the airfoil data.
    The first, second, second-to-last, and last control points are fixed at x=0 and x=1 as appropriate to preserve LE/TE tangency.
    Returns an array of x-locations (length = num_control_points).
    """
    data = np.asarray(original_data)
    x_data = data[:, 0]
    n = num_control_points
    indices = np.linspace(0, len(x_data), n+1, dtype=int)
    x_medians = []
    for i in range(n):
        seg = x_data[indices[i]:indices[i+1]]
        if len(seg) == 0:
            x_medians.append(x_medians[-1] if x_medians else 0.0)
        else:
            x_medians.append(np.median(seg))
    # Ensure first and last are exactly 0 and 1 (LE and TE)
    x_medians[0] = 0.0
    x_medians[-1] = 1.0
    # Fix second point to x=0 for LE tangency
    if n > 2:
        x_medians[1] = 0.0
    return np.array(x_medians)

def build_single_venkatamaran_bezier(original_data, num_control_points_new,
                                 start_point, end_point, is_upper_surface,
                                 le_tangent_vector, te_tangent_vector, regularization_weight=0.01, error_function="icp", num_points_curve_error=None, use_curvature_sampling=False):
    """
    Builds a single Bezier curve using the Venkataraman method.
    Optimizes only the y-coordinates of the inner control points.
    error_function: "mse" or "icp" or "venkat_median_x"
    """
    _ = le_tangent_vector

    # Choose x-coordinates for control points
    if error_function.startswith("venkat_median_x"):
        paper_fixed_x_coords = median_x_control_points(original_data, num_control_points_new)
    else:
        paper_fixed_x_coords = _paper_fixed_x_coords_cache['upper' if is_upper_surface else 'lower']

    if num_control_points_new != len(paper_fixed_x_coords):
        num_control_points_new = len(paper_fixed_x_coords)

    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]

    def objective_build(variables_y):
        control_points = np.zeros((len(variables_y) + 2, 2))
        control_points[0] = start_point
        control_points[1:-1, 0] = fixed_inner_x_coords
        control_points[1:-1, 1] = variables_y
        control_points[-1] = end_point
        # Use orthogonal error if requested
        if error_function == "venkat_median_x_orth":
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function="orthogonal_minmax", num_points_curve_error=num_points_curve_error, use_curvature_sampling=use_curvature_sampling)
        else:
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function=error_function, num_points_curve_error=num_points_curve_error, use_curvature_sampling=use_curvature_sampling)
        if isinstance(fitting_error, tuple):
            fitting_error = fitting_error[0]
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        return fitting_error + regularization_weight * smoothness_penalty

    initial_guess_inner_y = np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])

    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = end_point
    px_n_minus_1 = fixed_inner_x_coords[-1]
    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    if not np.isclose(tx_te, 0.0):
         constraints.append({'type': 'eq', 'fun': te_tangent_constraint})

    result = minimize(
        objective_build,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options={'disp': False, 'maxiter': config.SLSQP_OPTIONS['maxiter'], 'ftol': config.SLSQP_OPTIONS['ftol']}
    )

    if not result.success:
        logging.warning(
            "Single Bezier build failed. Using initial guess. Reason: %s",
            result.message,
        )
        variables_y = initial_guess_inner_y
    else:
        variables_y = result.x

    control_points = np.zeros((len(variables_y) + 2, 2))
    control_points[0] = start_point
    control_points[1:-1, 0] = fixed_inner_x_coords
    control_points[1:-1, 1] = variables_y
    control_points[-1] = end_point
    return control_points

def build_single_venkatamaran_bezier_minmax(original_data, num_control_points_new,
                                          start_point, end_point, is_upper_surface,
                                          le_tangent_vector, te_tangent_vector, 
                                          regularization_weight=0.01, error_function="orthogonal_minmax", 
                                          num_points_curve_error=None, use_curvature_sampling=False,
                                          num_points_curvature_resample=10000):
    """
    Builds a single Bezier curve using minmax optimization with orthogonal distance.
    Optimizes to minimize the maximum orthogonal distance error.
    First runs a full ICP optimization to get a good initial guess.
    """
    
    # Currently, the leading-edge tangent vector is not used by this implementation,
    # but the parameter is retained for future extensions and API stability.
    _ = le_tangent_vector

    paper_fixed_x_coords = _paper_fixed_x_coords_cache['upper' if is_upper_surface else 'lower']
    
    if num_control_points_new != len(paper_fixed_x_coords):
        num_control_points_new = len(paper_fixed_x_coords)

    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]

    def objective_minmax(variables_y):
        """
        Minmax objective function: minimize maximum orthogonal distance error.
        """
        # Pre-allocate control points array for better performance
        control_points = np.zeros((len(variables_y) + 2, 2))
        control_points[0] = start_point
        control_points[1:-1, 0] = fixed_inner_x_coords
        control_points[1:-1, 1] = variables_y
        control_points[-1] = end_point
        
        # Calculate maximum orthogonal distance
        try:
            max_distance, max_idx = calculate_orthogonal_error_minmax(control_points, original_data)
        except Exception as e:
            # If orthogonal distance calculation fails, return a large penalty
            logging.warning(f"Orthogonal distance calculation failed: {e}")
            return 1e6
        
        # Smoothness penalty (second derivative of control polygon)
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        
        total_objective = max_distance + regularization_weight * smoothness_penalty
        
        return total_objective

    # Constraints
    constraints = []
    
    # TE Tangency constraint
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = end_point
    px_n_minus_1 = fixed_inner_x_coords[-1]
    
    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
        
    if not np.isclose(tx_te, 0.0):
         constraints.append({'type': 'eq', 'fun': te_tangent_constraint})

    # Stage 1: Run full ICP optimization to get a good initial guess
    logging.info(f"Stage 1: Running full ICP optimization for {is_upper_surface and 'upper' or 'lower'} surface...")
    
    # Run the regular ICP optimization
    icp_result = build_single_venkatamaran_bezier(
        original_data=original_data,
        num_control_points_new=num_control_points_new,
        start_point=start_point,
        end_point=end_point,
        is_upper_surface=is_upper_surface,
        le_tangent_vector=le_tangent_vector,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        error_function="icp",
        num_points_curve_error=num_points_curve_error,
        use_curvature_sampling=use_curvature_sampling
    )
    
    # Extract the y-coordinates of the inner control points from ICP result
    icp_inner_y = icp_result[1:-1, 1]  # Skip first and last points (start/end), take y-coordinates
    
    logging.info(f"Stage 1 complete: ICP optimization finished for {is_upper_surface and 'upper' or 'lower'} surface")

    # Stage 2: Use ICP result as initial guess for minmax optimization
    logging.info(f"Stage 2: Starting minmax optimization using ICP result as initial guess...")
    
    # Reset the call counter for this stage
    if hasattr(objective_minmax, 'call_count'):
        del objective_minmax.call_count
    
    # Resample original_data by curvature if requested (for minimax only)
    if use_curvature_sampling:
        original_data = resample_points_by_curvature(original_data, num_points_curvature_resample)

    result = minimize(
        objective_minmax,
        icp_inner_y,  # Use ICP result as initial guess
        method='SLSQP',
        constraints=constraints,
        options={'disp': False, 'maxiter': 300, 'ftol': 1e-10}
    )

    if not result.success:
        logging.warning(
            "Minmax Bezier build failed. Using ICP solution. Reason: %s",
            result.message,
        )
        variables_y = icp_inner_y
    else:
        variables_y = result.x
        logging.info("Minmax optimization successful")

    # Build final control points
    control_points = np.zeros((len(variables_y) + 2, 2))
    control_points[0] = start_point
    control_points[1:-1, 0] = fixed_inner_x_coords
    control_points[1:-1, 1] = variables_y
    control_points[-1] = end_point
    return control_points

# -----------------------------------------------------------------------------
# Coupled optimiser enforcing G2 continuity at the leading edge
# -----------------------------------------------------------------------------

def build_coupled_venkatamaran_beziers(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    start_point_upper,
    end_point_upper,
    start_point_lower,
    end_point_lower,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="icp",
    use_curvature_sampling=False,
    num_points_curve_error=None,
):
    """Build upper and lower single-segment Bézier curves simultaneously while
    enforcing G2 continuity (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.
    """
    logging.info("Building coupled G2 Bezier curves using the " + error_function + " error function")

    # Fixed abscissae from the paper
    paper_fixed_x_upper = _paper_fixed_x_coords_cache['upper']
    paper_fixed_x_lower = _paper_fixed_x_coords_cache['lower']

    inner_x_upper = paper_fixed_x_upper[1:-1]  # 8 values
    inner_x_lower = paper_fixed_x_lower[1:-1]
    n_inner = len(inner_x_upper)

    # Initial guesses by interpolation of raw data
    init_y_upper = np.interp(inner_x_upper, original_upper_data[:, 0], original_upper_data[:, 1])
    init_y_lower = np.interp(inner_x_lower, original_lower_data[:, 0], original_lower_data[:, 1])
    initial_guess = np.concatenate([init_y_upper, init_y_lower])

    # --- Helper to build full control polygons from variable vector ----------
    def _assemble_polygons(var_y):
        y_u = var_y[:n_inner]
        y_l = var_y[n_inner:]

        # Pre-allocate arrays for better performance
        ctrl_upper = np.zeros((n_inner + 2, 2))
        ctrl_lower = np.zeros((n_inner + 2, 2))
        
        # Upper polygon
        ctrl_upper[0] = start_point_upper
        ctrl_upper[1:-1, 0] = inner_x_upper
        ctrl_upper[1:-1, 1] = y_u
        ctrl_upper[-1] = end_point_upper

        # Lower polygon
        ctrl_lower[0] = start_point_lower
        ctrl_lower[1:-1, 0] = inner_x_lower
        ctrl_lower[1:-1, 1] = y_l
        ctrl_lower[-1] = end_point_lower

        return ctrl_upper, ctrl_lower

    # --- Objective -----------------------------------------------------------
    def objective(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        err_u = calculate_single_bezier_fitting_error(ctrl_u, original_upper_data, error_function=error_function, use_curvature_sampling=use_curvature_sampling, num_points_curve_error=num_points_curve_error)
        err_l = calculate_single_bezier_fitting_error(ctrl_l, original_lower_data, error_function=error_function, use_curvature_sampling=use_curvature_sampling, num_points_curve_error=num_points_curve_error)

        # Extract just the error value if tuple is returned
        if isinstance(err_u, tuple):
            err_u = err_u[0]
        if isinstance(err_l, tuple):
            err_l = err_l[0]

        # Smoothness (second diff) penalty
        def _smooth(ctrl):
            if len(ctrl) <= 2:
                return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)

        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return err_u + err_l + regularization_weight * smooth

    # --- Constraints ---------------------------------------------------------
    constraints = []

    # Trailing-edge tangency (upper)
    tx_u, ty_u = te_tangent_vector_upper
    px_n_u, py_n_u = end_point_upper
    px_n1_u = inner_x_upper[-1]

    if not np.isclose(tx_u, 0.0):
        def _te_tan_upper(var_y):
            y_nm1 = var_y[n_inner - 1]  # last inner y of upper
            return y_nm1 * tx_u - (py_n_u * tx_u - (px_n_u - px_n1_u) * ty_u)
        constraints.append({"type": "eq", "fun": _te_tan_upper})

    # Trailing-edge tangency (lower)
    tx_l, ty_l = te_tangent_vector_lower
    px_n_l, py_n_l = end_point_lower
    px_n1_l = inner_x_lower[-1]

    if not np.isclose(tx_l, 0.0):
        def _te_tan_lower(var_y):
            y_nm1_l = var_y[-1]  # last element corresponds to lower inner trailing point
            return y_nm1_l * tx_l - (py_n_l * tx_l - (px_n_l - px_n1_l) * ty_l)
        constraints.append({"type": "eq", "fun": _te_tan_lower})

    # G2 continuity at LE: curvature_upper + curvature_lower == 0
    def _g2_constraint(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)

    constraints.append({"type": "eq", "fun": _g2_constraint})

    # --- Optimise ------------------------------------------------------------
    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS,
    )

    if not result.success:
        logging.warning("Coupled Bezier build failed. Using initial guess. Reason: %s", result.message)
        var_y_final = initial_guess
    else:
        var_y_final = result.x

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final

def build_coupled_venkatamaran_beziers_minmax(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    start_point_upper,
    end_point_upper,
    start_point_lower,
    end_point_lower,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="orthogonal_minmax",
    use_curvature_sampling=False,
    num_points_curve_error=None,
    num_points_curvature_resample=200,
):
    """Build upper and lower single-segment Bézier curves simultaneously using
    minmax optimization with orthogonal distance while enforcing G2 continuity
    (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.
    
    First runs a full ICP optimization to get a good initial guess.
    """
    logging.info("Building coupled G2 Bezier curves using minmax " + error_function + " optimization")

    # Fixed abscissae from the paper
    paper_fixed_x_upper = _paper_fixed_x_coords_cache['upper']
    paper_fixed_x_lower = _paper_fixed_x_coords_cache['lower']

    inner_x_upper = paper_fixed_x_upper[1:-1]  # 8 values
    inner_x_lower = paper_fixed_x_lower[1:-1]
    n_inner = len(inner_x_upper)

    # --- Helper to build full control polygons from variable vector ----------
    def _assemble_polygons(var_y):
        y_u = var_y[:n_inner]
        y_l = var_y[n_inner:]

        # Pre-allocate arrays for better performance
        ctrl_upper = np.zeros((n_inner + 2, 2))
        ctrl_lower = np.zeros((n_inner + 2, 2))
        
        # Upper polygon
        ctrl_upper[0] = start_point_upper
        ctrl_upper[1:-1, 0] = inner_x_upper
        ctrl_upper[1:-1, 1] = y_u
        ctrl_upper[-1] = end_point_upper

        # Lower polygon
        ctrl_lower[0] = start_point_lower
        ctrl_lower[1:-1, 0] = inner_x_lower
        ctrl_lower[1:-1, 1] = y_l
        ctrl_lower[-1] = end_point_lower

        return ctrl_upper, ctrl_lower

    # --- Minmax Objective ----------------------------------------------------
    def objective_minmax(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        
        # Calculate maximum orthogonal distance for both surfaces
        max_err_u, _ = calculate_orthogonal_error_minmax(ctrl_u, original_upper_data)
        max_err_l, _ = calculate_orthogonal_error_minmax(ctrl_l, original_lower_data)
        
        # For minmax, we want to minimize the worst error across both surfaces
        combined_max_error = max(float(max_err_u), float(max_err_l))

        # Smoothness (second diff) penalty
        def _smooth(ctrl):
            if len(ctrl) <= 2:
                return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)

        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return combined_max_error + regularization_weight * smooth

    # --- Constraints ---------------------------------------------------------
    constraints = []

    # Trailing-edge tangency (upper)
    tx_u, ty_u = te_tangent_vector_upper
    px_n_u, py_n_u = end_point_upper
    px_n1_u = inner_x_upper[-1]

    if not np.isclose(tx_u, 0.0):
        def _te_tan_upper(var_y):
            y_nm1 = var_y[n_inner - 1]  # last inner y of upper
            return y_nm1 * tx_u - (py_n_u * tx_u - (px_n_u - px_n1_u) * ty_u)
        constraints.append({"type": "eq", "fun": _te_tan_upper})

    # Trailing-edge tangency (lower)
    tx_l, ty_l = te_tangent_vector_lower
    px_n_l, py_n_l = end_point_lower
    px_n1_l = inner_x_lower[-1]

    if not np.isclose(tx_l, 0.0):
        def _te_tan_lower(var_y):
            y_nm1_l = var_y[-1]  # last element corresponds to lower inner trailing point
            return y_nm1_l * tx_l - (py_n_l * tx_l - (px_n_l - px_n1_l) * ty_l)
        constraints.append({"type": "eq", "fun": _te_tan_lower})

    # G2 continuity at LE: curvature_upper + curvature_lower == 0
    def _g2_constraint(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)

    constraints.append({"type": "eq", "fun": _g2_constraint})

    # --- Two-stage Optimisation ----------------------------------------------
    # Stage 1: Run full ICP optimization to get a good initial guess
    logging.info("Stage 1: Running full ICP optimization for coupled surfaces...")
    
    # Run the regular ICP optimization for coupled surfaces
    icp_upper, icp_lower = build_coupled_venkatamaran_beziers(
        original_upper_data=original_upper_data,
        original_lower_data=original_lower_data,
        regularization_weight=regularization_weight,
        start_point_upper=start_point_upper,
        end_point_upper=end_point_upper,
        start_point_lower=start_point_lower,
        end_point_lower=end_point_lower,
        te_tangent_vector_upper=te_tangent_vector_upper,
        te_tangent_vector_lower=te_tangent_vector_lower,
        error_function="icp",
        use_curvature_sampling=use_curvature_sampling,
        num_points_curve_error=num_points_curve_error,
    )
    
    # Extract the y-coordinates of the inner control points from ICP results
    icp_upper_inner_y = icp_upper[1:-1, 1]  # Skip first and last points, take y-coordinates
    icp_lower_inner_y = icp_lower[1:-1, 1]
    
    # Combine into single vector for minmax optimization
    improved_initial_guess = np.concatenate([icp_upper_inner_y, icp_lower_inner_y])
    
    logging.info("Stage 1 complete: ICP optimization finished for coupled surfaces")

    # Stage 2: Refine using minmax optimization
    logging.info("Stage 2: Starting minmax optimization using ICP result as initial guess...")
    
    # Resample original_data by curvature if requested (for minimax only)
    if use_curvature_sampling:
        original_upper_data = resample_points_by_curvature(original_upper_data, num_points_curvature_resample)
        original_lower_data = resample_points_by_curvature(original_lower_data, num_points_curvature_resample)

    result = minimize(
        objective_minmax,
        improved_initial_guess,
        method="SLSQP",
        constraints=constraints,
        options={'disp': False, 'maxiter': 500, 'ftol': 1e-10}
    )

    if not result.success:
        logging.warning("Coupled minmax Bezier build failed. Using ICP solution. Reason: %s", result.message)
        var_y_final = improved_initial_guess
    else:
        var_y_final = result.x
        logging.info("Minmax optimization successful")

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final
