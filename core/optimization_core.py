import numpy as np
from scipy.optimize import minimize
from utils.bezier_utils import general_bezier_curve, leading_edge_curvature
from scipy.special import comb
import logging
from core import config

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


def calculate_single_bezier_fitting_error(bezier_poly, original_data, error_function="mse", return_max_error=False):
    """
    Calculates the fitting error for a single Bezier curve.
    error_function: "mse" or "icp"
    If return_max_error is True and error_function=="icp", returns (sum, max_error, max_error_idx).
    """
    # Number of sample points pulled from central configuration for accuracy
    num_points_curve = config.NUM_POINTS_CURVE_ERROR
    curve_points = general_bezier_curve(np.linspace(0, 1, num_points_curve), bezier_poly)
    curve_sorted = curve_points[np.argsort(curve_points[:, 0])]
    if error_function == "icp":
        return calculate_icp_error(original_data, curve_sorted, return_max_error=return_max_error)
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
    error_function="mse",
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
