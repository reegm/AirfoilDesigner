import numpy as np
from scipy.optimize import minimize, minimize_scalar
from utils.bezier_utils import general_bezier_curve, leading_edge_curvature, bezier_curvature
import logging
from core import config
from utils.bezier_optimization_utils import calculate_all_orthogonal_distances
from utils.error_calculators import calculate_single_bezier_fitting_error, resample_points_by_curvature, calculate_orthogonal_error_minmax
from utils.control_point_utils import median_x_control_points, get_paper_fixed_x_coords

def _log_message(message, logger_func=None):
    """Helper function to log messages using either the provided logger function or standard logging."""
    if logger_func is not None:
        logger_func(message)
    else:
        logging.info(message)


def map_gui_strategy_to_internal(gui_strategy: str) -> dict:
    """Map GUI fitting strategy selection to internal configuration"""
    mapping = {
        "Standard ICP (Fast)": {
            "method": "standard_icp",
            "control_point_strategy": "paper_fixed",
            "error_metric": "icp"
        },
        "MinMax Orthogonal (Experimental)": {
            "method": "minmax_orthogonal",
            "control_point_strategy": "paper_fixed",
            "error_metric": "orthogonal_minmax"
        },
        "Median-X ICP (Venkataraman 2017)": {
            "method": "median_x_icp",
            "control_point_strategy": "median_x",
            "error_metric": "icp"
        },
        "Median-X Orthogonal": {
            "method": "median_x_orthogonal",
            "control_point_strategy": "median_x",
            "error_metric": "orthogonal_minmax"
        }
    }
    return mapping.get(gui_strategy, mapping["Standard ICP (Fast)"])







def build_single_venkatamaran_bezier(original_data, num_control_points_new,
                                 is_upper_surface,
                                 le_tangent_vector, te_tangent_vector, regularization_weight=0.01, optimization_method="standard_icp", num_points_curve_error=None, use_curvature_sampling=False, logger_func=None):
    """
    Builds a single Bezier curve using the Venkataraman method.
    Optimizes only the y-coordinates of the inner control points.
    optimization_method: "standard_icp", "median_x_icp", "median_x_orthogonal"
    """
    _ = le_tangent_vector

    # Choose x-coordinates for control points
    if optimization_method.startswith("median_x"):
        paper_fixed_x_coords = median_x_control_points(original_data, num_control_points_new)
    else:
        paper_fixed_x_coords = get_paper_fixed_x_coords(is_upper_surface)

    if num_control_points_new != len(paper_fixed_x_coords):
        num_control_points_new = len(paper_fixed_x_coords)

    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]

    def objective_build(variables_y):
        control_points = np.zeros((len(variables_y) + 2, 2))
        control_points[0] = np.array([0.0, 0.0])  # LE at (0,0)
        control_points[1:-1, 0] = fixed_inner_x_coords
        control_points[1:-1, 1] = variables_y
        control_points[-1] = np.array([1.0, 0.0])  # TE at (1,0)
        # Use orthogonal error if requested
        if optimization_method == "median_x_orthogonal":
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function="orthogonal_minmax", num_points_curve_error=num_points_curve_error, use_curvature_sampling=use_curvature_sampling)
        else:
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function="icp", num_points_curve_error=num_points_curve_error, use_curvature_sampling=use_curvature_sampling)
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
    px_n, py_n = 1.0, 0.0  # TE at (1,0)
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
        _log_message(
            f"Single Bezier build failed. Using initial guess. Reason: {result.message}",
            logger_func
        )
        variables_y = initial_guess_inner_y
    else:
        variables_y = result.x

    control_points = np.zeros((len(variables_y) + 2, 2))
    control_points[0] = np.array([0.0, 0.0])  # LE at (0,0)
    control_points[1:-1, 0] = fixed_inner_x_coords
    control_points[1:-1, 1] = variables_y
    control_points[-1] = np.array([1.0, 0.0])  # TE at (1,0)
    return control_points

def build_single_venkatamaran_bezier_minmax(original_data, num_control_points_new,
                                          is_upper_surface,
                                          le_tangent_vector, te_tangent_vector, 
                                          regularization_weight=0.01, optimization_method="minmax_orthogonal", 
                                          num_points_curve_error=None, use_curvature_sampling=False,
                                          num_points_curvature_resample=config.DEFAULT_NUM_POINTS_CURVATURE_RESAMPLE, logger_func=None):
    """
    Builds a single Bezier curve using minmax optimization with orthogonal distance.
    Optimizes to minimize the maximum orthogonal distance error.
    First runs a full ICP optimization to get a good initial guess.
    """

    # Currently, the leading-edge tangent vector is not used by this implementation,
    # but the parameter is retained for future extensions and API stability.
    _ = le_tangent_vector

    paper_fixed_x_coords = get_paper_fixed_x_coords(is_upper_surface)

    if num_control_points_new != len(paper_fixed_x_coords):
        num_control_points_new = len(paper_fixed_x_coords)

    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]

    def objective_minmax(variables_y):
        """
        Minmax objective function: minimize maximum orthogonal distance error.
        """
        # Pre-allocate control points array for better performance
        control_points = np.zeros((len(variables_y) + 2, 2))
        control_points[0] = np.array([0.0, 0.0])  # LE at (0,0)
        control_points[1:-1, 0] = fixed_inner_x_coords
        control_points[1:-1, 1] = variables_y
        control_points[-1] = np.array([1.0, 0.0])  # TE at (1,0)

        # Calculate maximum orthogonal distance
        try:
            max_distance, max_idx = calculate_orthogonal_error_minmax(control_points, original_data)
        except Exception as e:
            # If orthogonal distance calculation fails, return a large penalty
            _log_message(f"Orthogonal distance calculation failed: {e}", logger_func)
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
    px_n, py_n = 1.0, 0.0  # TE at (1,0)
    px_n_minus_1 = fixed_inner_x_coords[-1]

    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)

    if not np.isclose(tx_te, 0.0):
         constraints.append({'type': 'eq', 'fun': te_tangent_constraint})

    # Stage 1: Run full ICP optimization to get a good initial guess
    _log_message(f"Stage 1: Running full ICP optimization for {is_upper_surface and 'upper' or 'lower'} surface...", logger_func)

    # Run the regular ICP optimization
    icp_result = build_single_venkatamaran_bezier(
        original_data=original_data,
        num_control_points_new=num_control_points_new,
        is_upper_surface=is_upper_surface,
        le_tangent_vector=le_tangent_vector,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        error_function="icp",
        num_points_curve_error=num_points_curve_error,
        use_curvature_sampling=use_curvature_sampling,
        logger_func=logger_func
    )

    # Extract the y-coordinates of the inner control points from ICP result
    icp_inner_y = icp_result[1:-1, 1]  # Skip first and last points (start/end), take y-coordinates

    _log_message(f"Stage 1 complete: ICP optimization finished for {is_upper_surface and 'upper' or 'lower'} surface", logger_func)

    # Stage 2: Use ICP result as initial guess for minmax optimization
    _log_message(f"Stage 2: Starting minmax optimization using ICP result as initial guess...", logger_func)

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
        _log_message(
            f"Minmax Bezier build failed. Using ICP solution. Reason: {result.message}",
            logger_func
        )
        variables_y = icp_inner_y
    else:
        variables_y = result.x
        _log_message("Minmax optimization successful", logger_func)

    # Build final control points
    control_points = np.zeros((len(variables_y) + 2, 2))
    control_points[0] = np.array([0.0, 0.0])  # LE at (0,0)
    control_points[1:-1, 0] = fixed_inner_x_coords
    control_points[1:-1, 1] = variables_y
    control_points[-1] = np.array([1.0, 0.0])  # TE at (1,0)
    return control_points

# -----------------------------------------------------------------------------
# Coupled optimiser enforcing G2 continuity at the leading edge
# -----------------------------------------------------------------------------

def build_coupled_venkatamaran_beziers(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    optimization_method="standard_icp",
    use_curvature_sampling=False,
    num_points_curve_error=None,
    logger_func=None,
):
    """Build upper and lower single-segment Bézier curves simultaneously while
    enforcing G2 continuity (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.
    """
    _log_message("Building coupled G2 Bezier curves using " + optimization_method + " optimization", logger_func)

    # Fixed abscissae from the paper
    paper_fixed_x_upper = get_paper_fixed_x_coords(True)  # upper
    paper_fixed_x_lower = get_paper_fixed_x_coords(False)  # lower

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
        err_u = calculate_single_bezier_fitting_error(ctrl_u, original_upper_data, error_function="icp", use_curvature_sampling=use_curvature_sampling, num_points_curve_error=num_points_curve_error)
        err_l = calculate_single_bezier_fitting_error(ctrl_l, original_lower_data, error_function="icp", use_curvature_sampling=use_curvature_sampling, num_points_curve_error=num_points_curve_error)

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
        _log_message(f"Coupled Bezier build failed. Using initial guess. Reason: {result.message}", logger_func)
        var_y_final = initial_guess
    else:
        var_y_final = result.x

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final

def build_coupled_venkatamaran_beziers_minmax(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    optimization_method="minmax_orthogonal",
    use_curvature_sampling=False,
    num_points_curve_error=None,
    num_points_curvature_resample=config.DEFAULT_NUM_POINTS_CURVATURE_RESAMPLE,
    logger_func=None,
):
    """Build upper and lower single-segment Bézier curves simultaneously using
    minmax optimization with orthogonal distance while enforcing G2 continuity
    (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.

    First runs a full ICP optimization to get a good initial guess.
    """
    _log_message("Building coupled G2 Bezier curves using minmax " + optimization_method + " optimization", logger_func)

    # Fixed abscissae from the paper
    paper_fixed_x_upper = get_paper_fixed_x_coords(True)  # upper
    paper_fixed_x_lower = get_paper_fixed_x_coords(False)  # lower

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
    _log_message("Stage 1: Running full ICP optimization for coupled surfaces...", logger_func)

    # Run the regular ICP optimization for coupled surfaces
    icp_upper, icp_lower = build_coupled_venkatamaran_beziers(
        original_upper_data=original_upper_data,
        original_lower_data=original_lower_data,
        regularization_weight=regularization_weight,
        te_tangent_vector_upper=te_tangent_vector_upper,
        te_tangent_vector_lower=te_tangent_vector_lower,
        optimization_method="standard_icp",
        use_curvature_sampling=use_curvature_sampling,
        num_points_curve_error=num_points_curve_error,
        logger_func=logger_func,
    )

    # Extract the y-coordinates of the inner control points from ICP results
    icp_upper_inner_y = icp_upper[1:-1, 1]  # Skip first and last points, take y-coordinates
    icp_lower_inner_y = icp_lower[1:-1, 1]

    # Combine into single vector for minmax optimization
    improved_initial_guess = np.concatenate([icp_upper_inner_y, icp_lower_inner_y])

    _log_message("Stage 1 complete: ICP optimization finished for coupled surfaces", logger_func)

    # Stage 2: Refine using minmax optimization
    _log_message("Stage 2: Starting minmax optimization using ICP result as initial guess...", logger_func)

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
        _log_message(f"Coupled minmax Bezier build failed. Using ICP solution. Reason: {result.message}", logger_func)
        var_y_final = improved_initial_guess
    else:
        var_y_final = result.x
        _log_message("Minmax optimization successful", logger_func)

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final
