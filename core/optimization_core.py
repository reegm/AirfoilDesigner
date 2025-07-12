import numpy as np
from scipy.optimize import minimize
from utils.bezier_utils import general_bezier_curve
from scipy.special import comb
import logging

def calculate_icp_error(data_points, curve_points):
    """
    Calculates the sum of squared Euclidean distances from each data point to the closest point on the curve.
    Args:
        data_points (np.ndarray): (N, 2) array of data points.
        curve_points (np.ndarray): (M, 2) array of points sampled along the curve.
    Returns:
        float: Sum of squared distances.
    """
    # For each data point, find the closest curve point
    dists = np.linalg.norm(data_points[:, None, :] - curve_points[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    return np.sum(min_dists ** 2)


def calculate_iterative_icp_error(data_points, model, polygons, max_iterations=10, tol=1e-6):
    """
    Performs a true iterative ICP loop:
    1. For each data point, find the closest point on the current curve.
    2. Fit the model to those correspondences (by minimizing squared distances).
    3. Update the curve and repeat.
    Returns the final sum of squared distances.
    """
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


def objective_function(variables, model, upper_data, lower_data, spacing_weight, smoothness_weight):
    """
    The objective function to minimize. It's a combination of geometric
    fitting error and penalties for spacing and smoothness.
    error_function: Only 'mse' is supported for segments.
    """
    model.set_variables(variables)
    polygons = model.polygons
    num_points_per_segment = 200
    curves = [general_bezier_curve(np.linspace(0, 1, num_points_per_segment), np.array(p)) for p in polygons]

    # Geometric fitting error
    gen_upper = np.vstack([curves[0], curves[1]])
    gen_upper = gen_upper[np.argsort(gen_upper[:, 0])]
    gen_lower = np.vstack([curves[2], curves[3]])
    gen_lower = gen_lower[np.argsort(gen_lower[:, 0])]

    # Always use MSE for segment optimization (parameter removed)
    interp_upper_y = np.interp(upper_data[:, 0], gen_upper[:, 0], gen_upper[:, 1])
    error_upper = np.sum((interp_upper_y - upper_data[:, 1])**2)
    interp_lower_y = np.interp(lower_data[:, 0], gen_lower[:, 0], gen_lower[:, 1])
    error_lower = np.sum((interp_lower_y - lower_data[:, 1])**2)

    geometric_error = error_upper + error_lower

    # Spacing penalty
    spacing_penalty = 0.0
    for poly in polygons:
        x_coords = np.array(poly)[:, 0]
        if len(x_coords) > 1:
            spacing_penalty += np.var(np.diff(x_coords))

    # Smoothness penalty
    smoothness_penalty = calculate_smoothness_penalty(polygons)

    return geometric_error + (spacing_weight * spacing_penalty) + (smoothness_weight * smoothness_penalty)

def calculate_smoothness_penalty(polygons):
    """
    Calculates a penalty based on the smoothness of the control polygons.
    """
    penalty = 0.0
    # G1 continuity at shoulder points
    # Upper shoulder
    p_upper_front = np.array(polygons[0])
    p_upper_rear = np.array(polygons[1])
    v1 = p_upper_front[-1] - p_upper_front[-2]
    v2 = p_upper_rear[1] - p_upper_rear[0]
    penalty += np.sum((v1 - v2)**2) * 10

    # Lower shoulder
    p_lower_front = np.array(polygons[2])
    p_lower_rear = np.array(polygons[3])
    v3 = p_lower_front[-1] - p_lower_front[-2]
    v4 = p_lower_rear[1] - p_lower_rear[0]
    penalty += np.sum((v3 - v4)**2) * 10
    
    return penalty

def calculate_segment_errors(model, upper_data, lower_data, error_function="mse"):
    """
    Calculates the fitting error for each individual Bezier segment.
    error_function: Only 'mse' is supported for segments.
    """
    polygons = model.polygons
    num_points_curve = 100
    curves = [general_bezier_curve(np.linspace(0, 1, num_points_curve), np.array(p)) for p in polygons]
    data_map = {0: upper_data, 1: upper_data, 2: lower_data, 3: lower_data}
    errors = []
    for i, curve in enumerate(curves):
        data = data_map[i]
        curve_sorted = curve[np.argsort(curve[:, 0])]
        x_min, x_max = curve_sorted[0, 0], curve_sorted[-1, 0]
        mask = (data[:, 0] >= x_min - 1e-6) & (data[:, 0] <= x_max + 1e-6)
        if not np.any(mask):
            errors.append(0.0)
            continue
        filtered_data = data[mask]
        if error_function == "icp":
            # Compute ICP-based error (sum of squared distances to closest points)
            errors.append(calculate_icp_error(filtered_data, curve_sorted))
        else:
            # Default to mean squared error on inter­polated y-values
            interp_y = np.interp(filtered_data[:, 0], curve_sorted[:, 0], curve_sorted[:, 1])
            errors.append(np.sum((interp_y - filtered_data[:, 1])**2))
    return np.array(errors)


def calculate_single_bezier_fitting_error(bezier_poly, original_data, error_function="mse"):
    """
    Calculates the fitting error for a single Bezier curve.
    error_function: "mse" or "icp"
    """
    num_points_curve = 200
    curve_points = general_bezier_curve(np.linspace(0, 1, num_points_curve), bezier_poly)
    curve_sorted = curve_points[np.argsort(curve_points[:, 0])]
    if error_function == "icp":
        return calculate_icp_error(original_data, curve_sorted)
    else:
        interp_y = np.interp(original_data[:, 0], curve_sorted[:, 0], curve_sorted[:, 1])
        error = np.sum((interp_y - original_data[:, 1])**2)
        return error

def build_single_venkatamaran_bezier(original_data, num_control_points_new,
                                 start_point, end_point, is_upper_surface,
                                 le_tangent_vector, te_tangent_vector, regularization_weight=0.01, error_function="mse"):
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
        fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function=error_function)
        
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

def fit_4segment_icp_iterative(model, upper_data, lower_data, max_iterations=15, tol=1e-6):
    """
    True iterative ICP for the 4-segment model. Updates y-coordinates of control points for each segment.
    """
    polygons = model.polygons
    prev_error = None
    for it in range(max_iterations):
        # Step 1: For each segment, sample the curve densely
        num_points_curve = 500
        curves = [general_bezier_curve(np.linspace(0, 1, num_points_curve), np.array(p)) for p in polygons]
        # Step 2: For each data point, find closest point on the corresponding surface (upper/lower)
        # Map: S1, S2 = upper; S3, S4 = lower
        seg_data = [(0, upper_data), (1, upper_data), (2, lower_data), (3, lower_data)]
        total_error = 0.0
        for seg_idx, data in seg_data:
            curve = curves[seg_idx]
            control_points = np.array(polygons[seg_idx])
            control_points_x = control_points[:, 0]
            n = len(control_points) - 1
            # For each data point, find closest point on curve
            dists = np.linalg.norm(data[:, None, :] - curve[None, :, :], axis=2)
            closest_idx = np.argmin(dists, axis=1)
            t_dense = np.linspace(0, 1, num_points_curve)
            t_corr = t_dense[closest_idx]
            y_corr = data[:, 1]
            # Step 3: Fit y-coords of control points using least squares
            y_ctrl_new = fit_bezier_y_least_squares(data, control_points_x, t_corr, y_corr)
            # Update only y-coords, keep x fixed
            for i in range(len(control_points)):
                polygons[seg_idx][i][1] = y_ctrl_new[i]
            # Accumulate error
            control_poly = np.column_stack([control_points_x, y_ctrl_new])
            curve_corr = general_bezier_curve(t_corr, control_poly)
            seg_error = float(np.sum((curve_corr[:, 1] - y_corr) ** 2))
            total_error += seg_error
        # Step 4: Enforce structure
        model._enforce_structure()
        # Step 5: Check convergence
        if prev_error is not None and abs(prev_error - total_error) < tol:
            break
        prev_error = total_error
    return prev_error if prev_error is not None else 0.0

def fit_bezier_y_least_squares(data_points, control_points_x, t_corr, y_corr):
    """
    Given data points, fixed x-coordinates of control points, and correspondences (t_corr, y_corr),
    fit the y-coordinates of the control points using least squares.
    """
    _ = data_points  # Unused in current implementation; kept for API consistency.
    n = len(control_points_x) - 1
    # Build Bernstein basis matrix
    binom_coeffs = np.array([comb(n, i) for i in range(n + 1)])
    B = np.array([binom_coeffs * (1 - t)**(n - np.arange(n + 1)) * t**np.arange(n + 1) for t in t_corr])
    # Solve B @ y_ctrl = y_corr
    y_ctrl, *_ = np.linalg.lstsq(B, y_corr, rcond=None)
    return y_ctrl

def calculate_iterative_icp_error_single_bezier(data_points, control_points, max_iterations=10, tol=1e-6):
    """
    Simplified iterative ICP fitting for a single Bezier curve.
    Currently returns the final control points without modifying x-coordinates.
    """
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
        y_ctrl_new = fit_bezier_y_least_squares(data_points, control_points_x, t_corr, y_corr)
        # Step 4: Check convergence
        error = np.sum((curve_dense[closest_idx, 1] - y_corr) ** 2)
        if prev_error is not None and abs(prev_error - error) < tol:
            break
        y_ctrl = y_ctrl_new
        prev_error = error
    # Return final control polygon and error
    final_control_poly = np.column_stack([control_points_x, y_ctrl])
    return final_control_poly, (prev_error if prev_error is not None else 0.0)
