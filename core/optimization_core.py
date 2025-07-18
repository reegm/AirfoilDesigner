import numpy as np
from scipy.optimize import minimize, minimize_scalar
from utils.bezier_utils import general_bezier_curve, leading_edge_curvature
from scipy.special import comb
import logging
from core import config

# New orthogonal distance functions
def calculate_orthogonal_distance_to_bezier_fast(point, control_points, initial_t_guess=None, max_iterations=12, tolerance=1e-7):
    """
    Fast version of orthogonal distance calculation with fewer iterations and simpler fallback.
    """
    def bezier_and_derivatives(t, ctrl_pts):
        """Compute Bezier curve point and its first two derivatives at parameter t."""
        n = len(ctrl_pts) - 1
        if n == 0:
            return ctrl_pts[0], np.zeros(2), np.zeros(2)
        
        # Use explicit binomial coefficient formula for better numerical stability
        from scipy.special import comb
        
        # Curve point
        curve_point = np.zeros(2)
        for i in range(n + 1):
            bernstein = comb(n, i) * (1 - t)**(n - i) * t**i
            curve_point += bernstein * ctrl_pts[i]
        
        # First derivative
        first_derivative = np.zeros(2)
        if n >= 1:
            for i in range(n):
                bernstein_deriv = comb(n - 1, i) * (1 - t)**(n - 1 - i) * t**i
                first_derivative += n * bernstein_deriv * (ctrl_pts[i + 1] - ctrl_pts[i])
        
        # Second derivative
        second_derivative = np.zeros(2)
        if n >= 2:
            for i in range(n - 1):
                bernstein_deriv2 = comb(n - 2, i) * (1 - t)**(n - 2 - i) * t**i
                second_derivative += n * (n - 1) * bernstein_deriv2 * (ctrl_pts[i + 2] - 2*ctrl_pts[i + 1] + ctrl_pts[i])
        
        return curve_point, first_derivative, second_derivative
    
    # Smart initial guess based on x-coordinate (works well for airfoils)
    if initial_t_guess is None:
        if control_points[-1, 0] > control_points[0, 0]:
            initial_t_guess = np.clip((point[0] - control_points[0, 0]) / 
                                    (control_points[-1, 0] - control_points[0, 0]), 0, 1)
        else:
            initial_t_guess = 0.5
    
    t = np.clip(initial_t_guess, 0, 1)
    
    # Newton-Raphson iteration with fewer iterations
    for iteration in range(max_iterations):
        curve_point, first_derivative, second_derivative = bezier_and_derivatives(t, control_points)
        
        # Vector from curve point to target point
        diff_vector = point - curve_point
        
        # Function: f(t) = (P - B(t)) · B'(t) = 0 for orthogonal distance
        f = np.dot(diff_vector, first_derivative)
        
        # Derivative: f'(t) = -B'(t) · B'(t) + (P - B(t)) · B''(t)
        f_prime = -np.dot(first_derivative, first_derivative) + np.dot(diff_vector, second_derivative)
        
        # Avoid division by zero
        if abs(f_prime) < 1e-12:
            break
        
        # Newton step with conservative damping
        step = f / f_prime
        if abs(step) > 0.5:  # Conservative damping for stability
            step = 0.5 * np.sign(step)
        
        t_new = t - step
        t_new = np.clip(t_new, 0, 1)
        
        # Check convergence
        if abs(t_new - t) < tolerance:
            t = t_new
            break
        
        t = t_new
    
    # Final evaluation
    curve_point, _, _ = bezier_and_derivatives(t, control_points)
    distance = np.linalg.norm(point - curve_point)
    
    return distance, t, curve_point

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
    def bezier_and_derivatives(t, ctrl_pts):
        """Compute Bezier curve point and its first two derivatives at parameter t."""
        n = len(ctrl_pts) - 1
        if n == 0:
            return ctrl_pts[0], np.zeros(2), np.zeros(2)
        
        # Use explicit binomial coefficient formula for better numerical stability
        from scipy.special import comb
        
        # Curve point
        curve_point = np.zeros(2)
        for i in range(n + 1):
            bernstein = comb(n, i) * (1 - t)**(n - i) * t**i
            curve_point += bernstein * ctrl_pts[i]
        
        # First derivative
        first_derivative = np.zeros(2)
        if n >= 1:
            for i in range(n):
                bernstein_deriv = comb(n - 1, i) * (1 - t)**(n - 1 - i) * t**i
                first_derivative += n * bernstein_deriv * (ctrl_pts[i + 1] - ctrl_pts[i])
        
        # Second derivative
        second_derivative = np.zeros(2)
        if n >= 2:
            for i in range(n - 1):
                bernstein_deriv2 = comb(n - 2, i) * (1 - t)**(n - 2 - i) * t**i
                second_derivative += n * (n - 1) * bernstein_deriv2 * (ctrl_pts[i + 2] - 2*ctrl_pts[i + 1] + ctrl_pts[i])
        
        return curve_point, first_derivative, second_derivative
    
    def try_newton_raphson(t_init):
        """Try Newton-Raphson optimization from a given starting point."""
        t = np.clip(t_init, 0, 1)
        
        for iteration in range(max_iterations):
            curve_point, first_derivative, second_derivative = bezier_and_derivatives(t, control_points)
            
            # Vector from curve point to target point
            diff_vector = point - curve_point
            
            # Function: f(t) = (P - B(t)) · B'(t) = 0 for orthogonal distance
            f = np.dot(diff_vector, first_derivative)
            
            # Derivative: f'(t) = -B'(t) · B'(t) + (P - B(t)) · B''(t)
            f_prime = -np.dot(first_derivative, first_derivative) + np.dot(diff_vector, second_derivative)
            
            # Avoid division by zero
            if abs(f_prime) < 1e-12:
                break
            
            # Newton step with damping for stability
            step = f / f_prime
            damping = 1.0
            if abs(step) > 0.5:  # Large steps can cause instability
                damping = 0.5 / abs(step)
            
            t_new = t - damping * step
            t_new = np.clip(t_new, 0, 1)
            
            # Check convergence
            if abs(t_new - t) < tolerance:
                t = t_new
                break
            
            t = t_new
        
        # Evaluate final distance
        curve_point, _, _ = bezier_and_derivatives(t, control_points)
        distance = np.linalg.norm(point - curve_point)
        return distance, t, curve_point
    
    # Multiple initial guesses for robustness
    if initial_t_guess is not None:
        candidates = [initial_t_guess]
    else:
        # Smart initial guess based on x-coordinate
        if control_points[-1, 0] > control_points[0, 0]:
            x_based_guess = np.clip((point[0] - control_points[0, 0]) / 
                                  (control_points[-1, 0] - control_points[0, 0]), 0, 1)
            candidates = [x_based_guess]
        else:
            candidates = [0.5]
    
    # Add additional candidates for robustness
    candidates.extend([0.0, 0.25, 0.5, 0.75, 1.0])
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
            continue  # Skip if this candidate fails
    
    # Fallback: brute force search if all Newton-Raphson attempts failed or gave poor results
    if best_distance == float('inf') or best_distance > 1.0:  # Sanity check
        t_samples = np.linspace(0, 1, 100)
        distances = []
        for t_sample in t_samples:
            curve_pt, _, _ = bezier_and_derivatives(t_sample, control_points)
            dist = np.linalg.norm(point - curve_pt)
            distances.append(dist)
        
        min_idx = np.argmin(distances)
        best_distance = distances[min_idx]
        best_t = t_samples[min_idx]
        best_curve_point, _, _ = bezier_and_derivatives(best_t, control_points)
    
    return best_distance, best_t, best_curve_point

def calculate_all_orthogonal_distances_fast(data_points, control_points, subsample_factor=1):
    """
    Fast calculation of orthogonal distances using optimized algorithm and optional subsampling.
    
    Args:
        data_points: np.array of shape (N, 2) - points to measure distances from
        control_points: np.array of shape (n+1, 2) - Bezier control points  
        subsample_factor: int - use every Nth point for speed (1 = all points)
    
    Returns:
        tuple: (distances, max_distance, max_distance_idx, t_values, curve_points)
    """
    # Optionally subsample for speed during optimization
    if subsample_factor > 1:
        indices = np.arange(0, len(data_points), subsample_factor)
        sample_points = data_points[indices]
    else:
        indices = np.arange(len(data_points))
        sample_points = data_points
    
    distances = []
    t_values = []
    curve_points = []
    
    for i, point in enumerate(sample_points):
        # Use x-coordinate based initial guess
        x_min, x_max = control_points[0, 0], control_points[-1, 0]
        if x_max > x_min:
            initial_t = np.clip((point[0] - x_min) / (x_max - x_min), 0, 1)
        else:
            initial_t = 0.5
        
        distance, t_opt, curve_point = calculate_orthogonal_distance_to_bezier_fast(
            point, control_points, initial_t_guess=initial_t
        )
        
        distances.append(distance)
        t_values.append(t_opt)
        curve_points.append(curve_point)
    
    distances = np.array(distances)
    max_distance = np.max(distances)
    max_distance_idx = np.argmax(distances)
    
    # If we subsampled, adjust the index back to original array
    if subsample_factor > 1:
        max_distance_idx = indices[max_distance_idx]
    
    return distances, max_distance, max_distance_idx, np.array(t_values), np.array(curve_points)

def calculate_all_orthogonal_distances(data_points, control_points):
    """
    Calculate orthogonal distances from all data points to a Bezier curve.
    
    Args:
        data_points: np.array of shape (N, 2) - points to measure distances from
        control_points: np.array of shape (n+1, 2) - Bezier control points
    
    Returns:
        tuple: (distances, max_distance, max_distance_idx, t_values, curve_points)
    """
    distances = []
    t_values = []
    curve_points = []
    
    for i, point in enumerate(data_points):
        # Use x-coordinate based initial guess for better convergence
        if len(data_points) > 1:
            # Linear interpolation of t based on x-coordinate
            x_min, x_max = control_points[0, 0], control_points[-1, 0]
            if x_max > x_min:
                initial_t = np.clip((point[0] - x_min) / (x_max - x_min), 0, 1)
            else:
                initial_t = i / (len(data_points) - 1) if len(data_points) > 1 else 0.5
        else:
            initial_t = 0.5
        
        distance, t_opt, curve_point = calculate_orthogonal_distance_to_bezier(
            point, control_points, initial_t_guess=initial_t
        )
        
        distances.append(distance)
        t_values.append(t_opt)
        curve_points.append(curve_point)
    
    distances = np.array(distances)
    max_distance = np.max(distances)
    max_distance_idx = np.argmax(distances)
    
    return distances, max_distance, max_distance_idx, np.array(t_values), np.array(curve_points)

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

def calculate_orthogonal_error_sum_squares(control_points, original_data):
    """
    Calculate the sum of squared orthogonal distances for comparison with ICP.
    
    Args:
        control_points: np.array of shape (n+1, 2) - Bezier control points  
        original_data: np.array of shape (N, 2) - original airfoil data points
    
    Returns:
        float: sum of squared orthogonal distances
    """
    distances, _, _, _, _ = calculate_all_orthogonal_distances(original_data, control_points)
    return np.sum(distances ** 2)

def calculate_orthogonal_error_sum_squares_fast(control_points, original_data, subsample_factor=None):
    """
    Fast calculation of sum of squared orthogonal distances using adaptive subsampling during optimization.
    
    Args:
        control_points: np.array of shape (n+1, 2) - Bezier control points  
        original_data: np.array of shape (N, 2) - original airfoil data points
        subsample_factor: int or None - use every Nth point for speed (auto-determined if None)
    
    Returns:
        float: sum of squared orthogonal distances (scaled to compensate for subsampling)
    """
    # Auto-determine subsampling based on data size for better balance of speed vs accuracy
    if subsample_factor is None:
        data_size = len(original_data)
        if data_size > 200:
            subsample_factor = 2  # Every other point for large datasets
        elif data_size > 100:
            subsample_factor = 1  # No subsampling for medium datasets
        else:
            subsample_factor = 1  # No subsampling for small datasets
    
    distances, _, _, _, _ = calculate_all_orthogonal_distances_fast(
        original_data, control_points, subsample_factor=subsample_factor
    )
    # Scale the sum to compensate for subsampling
    return np.sum(distances ** 2) * subsample_factor


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
    dists = np.linalg.norm(data_points[:, None, :] - curve_points[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    sum_sq = np.sum(min_dists ** 2)
    if return_max_error:
        max_error = np.max(min_dists)
        max_error_idx = np.argmax(min_dists)
        return sum_sq, max_error, max_error_idx
    return sum_sq


def calculate_iterative_icp_error(data_points, model, polygons, max_iterations=None, tol=None):
    """
    Performs a true iterative ICP loop:
    1. For each data point, find the closest point on the current curve.
    2. Fit the model to those correspondences (by minimizing squared distances).
    3. Update the curve and repeat.
    Returns the final sum of squared distances.
    """
    if max_iterations is None:
        max_iterations = config.ICP_OPTIONS["max_iterations"]
    if tol is None:
        tol = config.ICP_OPTIONS["tol"]
    # 'model' is currently unused because this simplified ICP routine only computes
    # the fitting error. The variable is kept to maintain API compatibility.
    _ = model
    # Initial curve sampling
    num_points_per_segment = 500
    curves = [general_bezier_curve(np.linspace(0, 1, num_points_per_segment), np.array(p)) for p in polygons]
    curve_points = np.vstack(curves)
    prev_error = None
    for it in range(max_iterations):
        # Step 1: Find closest curve point for each data point
        dists = np.linalg.norm(data_points[:, None, :] - curve_points[None, :, :], axis=2)
        closest_idx = np.argmin(dists, axis=1)
        correspondences = curve_points[closest_idx]
        # Step 2: Fit model to correspondences (least squares on y, fixed x)
        # For each data point, find the segment and t value (approximate by x)
        # For simplicity, only update y-coordinates of control points (like in single Bezier fit)
        # This is a simplification; a full ICP would require more complex fitting.
        # Here, just update the model by minimizing sum((model(x_i) - y_i)^2) for the correspondences.
        # (This is similar to MSE, but with correspondences found by closest point, not by x.)
        # For now, just compute the error and break if converged.
        error = np.sum((data_points - correspondences) ** 2)
        if prev_error is not None and abs(prev_error - error) < tol:
            break
        prev_error = error
        # Step 3: (Optional) update the curve_points by refitting the model to correspondences
        # Not implemented: would require a custom fit routine for the model.
        # For now, just use the correspondences for error calculation.
    return prev_error if prev_error is not None else 0.0


def calculate_single_bezier_fitting_error(bezier_poly, original_data, error_function="icp", return_max_error=False, num_points_curve_error=None):
    """
    Calculates the fitting error between a Bezier curve and the original data using the specified error function.
    Supported error functions: 'icp', 'orthogonal_sum_squares', 'orthogonal_minmax', 'orthogonal_conservative', 'orthogonal_fast'
    If return_max_error is True, returns (sum, max_error, max_error_idx).
    """
    if error_function == "orthogonal_minmax":
        max_error, max_error_idx = calculate_orthogonal_error_minmax(bezier_poly, original_data)
        if return_max_error:
            return max_error, max_error, max_error_idx  # For minmax, sum and max are the same
        return max_error
    
    elif error_function == "orthogonal_fast":
        # Use fast calculation during optimization, but full calculation for final error reporting
        sum_sq_error = calculate_orthogonal_error_sum_squares_fast(bezier_poly, original_data)  # Auto-adaptive subsampling
        if return_max_error:
            # For final error reporting, use full precision
            distances, max_error, max_error_idx, _, _ = calculate_all_orthogonal_distances(original_data, bezier_poly)
            return sum_sq_error, max_error, max_error_idx
        return sum_sq_error
    
    elif error_function in ["orthogonal_sum_squares", "orthogonal_conservative"]:
        sum_sq_error = calculate_orthogonal_error_sum_squares(bezier_poly, original_data)
        if return_max_error:
            distances, max_error, max_error_idx, _, _ = calculate_all_orthogonal_distances(original_data, bezier_poly)
            return sum_sq_error, max_error, max_error_idx
        return sum_sq_error
    
    else:  # ICP (default)
        if num_points_curve_error is None:
            num_points_curve = config.NUM_POINTS_CURVE_ERROR
        else:
            num_points_curve = num_points_curve_error
        curve_points = general_bezier_curve(np.linspace(0, 1, num_points_curve), bezier_poly)
        curve_sorted = curve_points[np.argsort(curve_points[:, 0])]
        return calculate_icp_error(original_data, curve_sorted, return_max_error=return_max_error)

def build_single_venkatamaran_bezier(original_data, num_control_points_new,
                                 start_point, end_point, is_upper_surface,
                                 le_tangent_vector, te_tangent_vector, regularization_weight=0.01, error_function="icp", num_points_curve_error=None):
    """
    Builds a single Bezier curve using the Venkataraman method.
    Optimizes only the y-coordinates of the inner control points.
    error_function: "mse" or "icp"
    """

    # Currently, the leading-edge tangent vector is not used by this implementation,
    # but the parameter is retained for future extensions and API stability.
    _ = le_tangent_vector

    paper_fixed_x_coords_upper = np.array([0.0, 0.0, 0.11422, 0.25294, 0.37581, 0.49671, 0.61942, 0.74701, 0.88058, 1.0])
    paper_fixed_x_coords_lower = np.array([0.0, 0.0, 0.12325, 0.25314, 0.37519, 0.49569, 0.61975, 0.74391, 0.87391, 1.0])

    paper_fixed_x_coords = paper_fixed_x_coords_upper if is_upper_surface else paper_fixed_x_coords_lower
    
    if num_control_points_new != len(paper_fixed_x_coords):
        num_control_points_new = len(paper_fixed_x_coords)

    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]

    def objective_build(variables_y):
        """
        Objective function for building the single Bezier curve.
        """
        control_points = [start_point]
        for i, y_val in enumerate(variables_y):
            control_points.append(np.array([fixed_inner_x_coords[i], y_val]))
        control_points.append(end_point)
        
        control_points = np.array(control_points)
        
        # Geometric error
        fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function=error_function, num_points_curve_error=num_points_curve_error)
        if isinstance(fitting_error, tuple):
            fitting_error = fitting_error[0]
        # Smoothness penalty (second derivative of control polygon)
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        
        return fitting_error + regularization_weight * smoothness_penalty

    # Initial guess for inner control points' y-coordinates
    initial_guess_inner_y = np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])

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

    result = minimize(
        objective_build,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options={'disp': False, 'maxiter': 500, 'ftol': 1e-9}
    )

    if not result.success:
        logging.warning(
            "Single Bezier build failed. Using initial guess. Reason: %s",
            result.message,
        )
        variables_y = initial_guess_inner_y
    else:
        variables_y = result.x

    control_points = [start_point]
    for i, y_val in enumerate(variables_y):
        control_points.append(np.array([fixed_inner_x_coords[i], y_val]))
    control_points.append(end_point)
    return np.array(control_points)

def build_single_venkatamaran_bezier_minmax(original_data, num_control_points_new,
                                          start_point, end_point, is_upper_surface,
                                          le_tangent_vector, te_tangent_vector, 
                                          regularization_weight=0.01, error_function="orthogonal_minmax", 
                                          num_points_curve_error=None):
    """
    Builds a single Bezier curve using minmax optimization with orthogonal distance.
    Optimizes to minimize the maximum orthogonal distance error.
    """
    
    # Currently, the leading-edge tangent vector is not used by this implementation,
    # but the parameter is retained for future extensions and API stability.
    _ = le_tangent_vector
    _ = num_points_curve_error  # Not used for orthogonal distance calculation

    paper_fixed_x_coords_upper = np.array([0.0, 0.0, 0.11422, 0.25294, 0.37581, 0.49671, 0.61942, 0.74701, 0.88058, 1.0])
    paper_fixed_x_coords_lower = np.array([0.0, 0.0, 0.12325, 0.25314, 0.37519, 0.49569, 0.61975, 0.74391, 0.87391, 1.0])

    paper_fixed_x_coords = paper_fixed_x_coords_upper if is_upper_surface else paper_fixed_x_coords_lower
    
    if num_control_points_new != len(paper_fixed_x_coords):
        num_control_points_new = len(paper_fixed_x_coords)

    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]

    def objective_minmax(variables_y):
        """
        Minmax objective function: minimize maximum orthogonal distance error.
        """
        control_points = [start_point]
        for i, y_val in enumerate(variables_y):
            control_points.append(np.array([fixed_inner_x_coords[i], y_val]))
        control_points.append(end_point)
        
        control_points = np.array(control_points)
        
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
        
        # Add logging every 10th function evaluation for debugging
        if not hasattr(objective_minmax, 'call_count'):
            objective_minmax.call_count = 0
        objective_minmax.call_count += 1
        
        if objective_minmax.call_count % 20 == 0:
            surface_type = "upper" if is_upper_surface else "lower"
            logging.info(f"Minmax optimization ({surface_type}): call {objective_minmax.call_count}, max_dist={max_distance:.6f}, smooth_penalty={smoothness_penalty:.6f}")
        
        return total_objective

    # For better convergence in minmax problems, we use a different approach:
    # First get a good initial guess using least squares optimization
    def objective_initial(variables_y):
        """Initial objective using sum of squares for better starting point."""
        control_points = [start_point]
        for i, y_val in enumerate(variables_y):
            control_points.append(np.array([fixed_inner_x_coords[i], y_val]))
        control_points.append(end_point)
        
        control_points = np.array(control_points)
        
        # Use orthogonal sum of squares for initial fit
        fitting_error = calculate_orthogonal_error_sum_squares(control_points, original_data)
        
        # Smoothness penalty
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        
        return fitting_error + regularization_weight * smoothness_penalty

    # Initial guess for inner control points' y-coordinates
    initial_guess_inner_y = np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])

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

    # Multi-stage optimization for better convergence
    current_guess = initial_guess_inner_y.copy()
    
    # Stage 1: Coarse least squares fit
    result_initial = minimize(
        objective_initial,
        current_guess,
        method='SLSQP',
        constraints=constraints,
        options={'disp': False, 'maxiter': 200, 'ftol': 1e-6}
    )
    
    if result_initial.success:
        current_guess = result_initial.x
        logging.info("Stage 1 (least squares) optimization successful")
    else:
        logging.warning("Stage 1 optimization failed, proceeding with initial guess")
    
    # Stage 2: Transition optimization with mixed objective
    def objective_mixed(variables_y, weight_minmax=0.5):
        """Mixed objective that combines sum of squares and minmax."""
        control_points = [start_point]
        for i, y_val in enumerate(variables_y):
            control_points.append(np.array([fixed_inner_x_coords[i], y_val]))
        control_points.append(end_point)
        
        control_points = np.array(control_points)
        
        try:
            # Sum of squares component
            sum_sq_error = calculate_orthogonal_error_sum_squares(control_points, original_data)
            
            # Minmax component
            max_distance, _ = calculate_orthogonal_error_minmax(control_points, original_data)
            
            # Weighted combination
            combined_error = (1 - weight_minmax) * sum_sq_error + weight_minmax * max_distance
            
        except Exception as e:
            logging.warning(f"Mixed objective calculation failed: {e}")
            return 1e6
        
        # Smoothness penalty
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        
        return combined_error + regularization_weight * smoothness_penalty
    
    # Gradually increase minmax weight
    for stage, minmax_weight in enumerate([0.3, 0.7]):
        stage_objective = lambda vars_y, w=minmax_weight: objective_mixed(vars_y, w)
        
        result_stage = minimize(
            stage_objective,
            current_guess,
            method='SLSQP',
            constraints=constraints,
            options={'disp': False, 'maxiter': 150, 'ftol': 1e-8}
        )
        
        if result_stage.success:
            current_guess = result_stage.x
            logging.info(f"Stage {stage + 2} (mixed w={minmax_weight}) optimization successful")
        else:
            logging.warning(f"Stage {stage + 2} optimization failed")
    
    # Stage 3: Pure minmax optimization
    # Reset the call counter for this stage
    if hasattr(objective_minmax, 'call_count'):
        del objective_minmax.call_count
    
    result = minimize(
        objective_minmax,
        current_guess,
        method='SLSQP',
        constraints=constraints,
        options={'disp': False, 'maxiter': 300, 'ftol': 1e-10}
    )

    if not result.success:
        logging.warning(
            "Minmax Bezier build failed. Using current solution. Reason: %s",
            result.message,
        )
        variables_y = current_guess
    else:
        variables_y = result.x
        logging.info("Pure minmax optimization successful")

    control_points = [start_point]
    for i, y_val in enumerate(variables_y):
        control_points.append(np.array([fixed_inner_x_coords[i], y_val]))
    control_points.append(end_point)
    return np.array(control_points)

def fit_bezier_y_least_squares(data_points, control_points_x, t_corr, y_corr, regularization_weight: float = 0.0):
    """
    Least-squares fit of the *y* coordinates of control points while keeping *x* fixed.

    Minimises  ||B y - y_corr||²  +  λ ||D y||²  where D is the second-difference
    operator encouraging smooth control-point variation.  λ = ``regularization_weight``.
    """
    _ = data_points  # Kept for API compatibility; not used here.

    n = len(control_points_x) - 1  # Bézier order
    binom_coeffs = np.array([comb(n, i) for i in range(n + 1)])
    B = np.array([binom_coeffs * (1 - t)**(n - np.arange(n + 1)) * t**np.arange(n + 1) for t in t_corr])

    if regularization_weight > 1e-12:
        # Build second-difference matrix D of size (n-1) x (n+1)
        rows = []
        for i in range(1, n):
            row = np.zeros(n + 1)
            row[i - 1] = 1.0
            row[i] = -2.0
            row[i + 1] = 1.0
            rows.append(row)
        D = np.vstack(rows)

        B_aug = np.vstack([B, np.sqrt(regularization_weight) * D])
        y_aug = np.concatenate([y_corr, np.zeros(D.shape[0])])
        y_ctrl, *_ = np.linalg.lstsq(B_aug, y_aug, rcond=None)
    else:
        y_ctrl, *_ = np.linalg.lstsq(B, y_corr, rcond=None)

    return y_ctrl

def calculate_iterative_icp_error_single_bezier(data_points, control_points, max_iterations=None, tol=None, regularization_weight: float = 0.0):
    """
    Simplified iterative ICP fitting for a single Bezier curve.
    Currently returns the final control points without modifying x-coordinates.
    """
    if max_iterations is None:
        max_iterations = config.ICP_OPTIONS["max_iterations"]
    if tol is None:
        tol = config.ICP_OPTIONS["tol"]
    _ = data_points  # Placeholder to retain API compatibility while silencing linter.
    n = len(control_points) - 1
    control_points_x = np.array([pt[0] for pt in control_points])
    y_ctrl = np.array([pt[1] for pt in control_points])
    prev_error = None
    for it in range(max_iterations):
        # Step 1: Sample curve densely
        t_dense = np.linspace(0, 1, 1000)
        control_poly = np.column_stack([control_points_x, y_ctrl])
        curve_dense = general_bezier_curve(t_dense, control_poly)
        # Step 2: For each data point, find closest curve point and its t
        dists = np.linalg.norm(data_points[:, None, :] - curve_dense[None, :, :], axis=2)
        closest_idx = np.argmin(dists, axis=1)
        t_corr = t_dense[closest_idx]
        y_corr = data_points[:, 1]
        # Step 3: Fit y_ctrl to y_corr at t_corr
        y_ctrl_new = fit_bezier_y_least_squares(data_points, control_points_x, t_corr, y_corr, regularization_weight)
        # Step 4: Check convergence
        # Combine geometric error and smoothness penalty for convergence check
        smooth_pen = 0.0
        if regularization_weight > 1e-12 and len(y_ctrl_new) > 2:
            smooth_pen = regularization_weight * np.sum(np.diff(y_ctrl_new, n=2) ** 2)
        error = np.sum((curve_dense[closest_idx, 1] - y_corr) ** 2) + smooth_pen
        if prev_error is not None and abs(prev_error - error) < tol:
            break
        y_ctrl = y_ctrl_new
        prev_error = error
    # Return final control polygon and error
    final_control_poly = np.column_stack([control_points_x, y_ctrl])
    return final_control_poly, (prev_error if prev_error is not None else 0.0)

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
):
    """Build upper and lower single-segment Bézier curves simultaneously while
    enforcing G2 continuity (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.
    """
    logging.info("Building coupled G2 Bezier curves using the " + error_function + " error function")

    # Fixed abscissae from the paper
    paper_fixed_x_upper = np.array([0.0, 0.0, 0.11422, 0.25294, 0.37581, 0.49671, 0.61942, 0.74701, 0.88058, 1.0])
    paper_fixed_x_lower = np.array([0.0, 0.0, 0.12325, 0.25314, 0.37519, 0.49569, 0.61975, 0.74391, 0.87391, 1.0])

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

        # Upper polygon
        ctrl_upper = [start_point_upper]
        ctrl_upper += [np.array([inner_x_upper[i], y_u[i]]) for i in range(n_inner)]
        ctrl_upper.append(end_point_upper)

        # Lower polygon
        ctrl_lower = [start_point_lower]
        ctrl_lower += [np.array([inner_x_lower[i], y_l[i]]) for i in range(n_inner)]
        ctrl_lower.append(end_point_lower)

        return np.array(ctrl_upper), np.array(ctrl_lower)

    # --- Objective -----------------------------------------------------------
    def objective(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        err_u = calculate_single_bezier_fitting_error(ctrl_u, original_upper_data, error_function=error_function)
        err_l = calculate_single_bezier_fitting_error(ctrl_l, original_lower_data, error_function=error_function)

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
):
    """Build upper and lower single-segment Bézier curves simultaneously using
    minmax optimization with orthogonal distance while enforcing G2 continuity
    (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.
    """
    logging.info("Building coupled G2 Bezier curves using minmax " + error_function + " optimization")

    # Fixed abscissae from the paper
    paper_fixed_x_upper = np.array([0.0, 0.0, 0.11422, 0.25294, 0.37581, 0.49671, 0.61942, 0.74701, 0.88058, 1.0])
    paper_fixed_x_lower = np.array([0.0, 0.0, 0.12325, 0.25314, 0.37519, 0.49569, 0.61975, 0.74391, 0.87391, 1.0])

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

        # Upper polygon
        ctrl_upper = [start_point_upper]
        ctrl_upper += [np.array([inner_x_upper[i], y_u[i]]) for i in range(n_inner)]
        ctrl_upper.append(end_point_upper)

        # Lower polygon
        ctrl_lower = [start_point_lower]
        ctrl_lower += [np.array([inner_x_lower[i], y_l[i]]) for i in range(n_inner)]
        ctrl_lower.append(end_point_lower)

        return np.array(ctrl_upper), np.array(ctrl_lower)

    # --- Minmax Objective ----------------------------------------------------
    def objective_minmax(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        
        # Calculate maximum orthogonal distance for both surfaces
        max_err_u, _ = calculate_orthogonal_error_minmax(ctrl_u, original_upper_data)
        max_err_l, _ = calculate_orthogonal_error_minmax(ctrl_l, original_lower_data)
        
        # For minmax, we want to minimize the worst error across both surfaces
        combined_max_error = max(max_err_u, max_err_l)

        # Smoothness (second diff) penalty
        def _smooth(ctrl):
            if len(ctrl) <= 2:
                return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)

        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return combined_max_error + regularization_weight * smooth

    # --- Initial Objective (sum of squares for better starting point) -------
    def objective_initial(var_y):
        ctrl_u, ctrl_l = _assemble_polygons(var_y)
        err_u = calculate_orthogonal_error_sum_squares(ctrl_u, original_upper_data)
        err_l = calculate_orthogonal_error_sum_squares(ctrl_l, original_lower_data)

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

    # --- Two-stage Optimisation ----------------------------------------------
    # Stage 1: Get good initial solution using sum of squares
    result_initial = minimize(
        objective_initial,
        initial_guess,
        method="SLSQP",
        constraints=constraints,
        options={'disp': False, 'maxiter': 300, 'ftol': 1e-8}
    )

    if result_initial.success:
        improved_initial_guess = result_initial.x
        logging.info("Initial sum-of-squares optimization successful")
    else:
        improved_initial_guess = initial_guess
        logging.warning("Initial sum-of-squares optimization failed. Using interpolated guess. Reason: %s", result_initial.message)

    # Stage 2: Refine using minmax optimization
    result = minimize(
        objective_minmax,
        improved_initial_guess,
        method="SLSQP",
        constraints=constraints,
        options={'disp': False, 'maxiter': 500, 'ftol': 1e-10}
    )

    if not result.success:
        logging.warning("Coupled minmax Bezier build failed. Using initial solution. Reason: %s", result.message)
        var_y_final = improved_initial_guess
    else:
        var_y_final = result.x
        logging.info("Minmax optimization successful")

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final
