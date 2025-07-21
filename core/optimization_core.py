import numpy as np
from scipy.optimize import minimize
from utils.bezier_utils import leading_edge_curvature
import logging
from core import config
from utils.error_calculators import calculate_single_bezier_fitting_error, calculate_orthogonal_error_minmax
from utils.control_point_utils import variable_x_control_points, get_paper_fixed_x_coords

def _log_message(message, logger_func=None):
    """Helper function to log messages using either the provided logger function or standard logging."""
    if logger_func is not None:
        logger_func(message)
    else:
        logging.info(message)


def map_gui_strategy_to_internal(gui_strategy: str, enforce_g2: bool = False, error_function: str = "Euclidean") -> dict:
    """Map GUI strategy and error function selection to internal configuration"""
    
    # Map strategy to control point strategy and base method
    strategy_mapping = {
        "Fixed-x": {
            "control_point_strategy": "paper_fixed",
            "base_method": "fixed_x"
        },
        "Variable-x": {
            "control_point_strategy": "variable_x", 
            "base_method": "variable_x"
        },
        "Minmax": {
            "control_point_strategy": "paper_fixed",
            "base_method": "minmax"
        }
    }
    
    strategy_config = strategy_mapping.get(gui_strategy, strategy_mapping["Fixed-x"])
    
    # Map error function to error metric and sampling type
    error_mapping = {
        "Euclidean": {
            "error_metric": "euclidean",
            "use_curvature_sampling": False
        },
        "Orthogonal": {
            "error_metric": "orthogonal_minmax" if gui_strategy == "Minmax" else "orthogonal_icp",
            "use_curvature_sampling": True
        }
    }
    
    error_config = error_mapping.get(error_function, error_mapping["Euclidean"])
    
    # Combine strategy and error function to determine final method
    method = strategy_config["base_method"]
    
    # Apply error function modifications
    if error_function == "Orthogonal":
        if gui_strategy == "Fixed-x":
            method = "fixed_x_orthogonal"
        elif gui_strategy == "Variable-x":
            method = "variable_x_orthogonal"
        elif gui_strategy == "Minmax":
            method = "minmax"
    
    # If G2 is enabled, map variable-x methods to their G2 equivalents
    if enforce_g2:
        if method == "variable_x":
            method = "variable_x_g2"
        elif method == "variable_x_orthogonal":
            method = "variable_x_orthogonal_g2"
    
    config = {
        "method": method,
        "control_point_strategy": strategy_config["control_point_strategy"],
        "error_metric": error_config["error_metric"],
        "use_curvature_sampling": error_config["use_curvature_sampling"]
    }
    
    return config







def build_single_venkatamaran_bezier(
                                original_data,
                                num_control_points_new,
                                is_upper_surface,
                                le_tangent_vector,
                                te_tangent_vector,
                                regularization_weight=0.01,
                                optimization_method="fixed_x",
                                logger_func=None,
                                ):
    """
    Builds a single Bezier curve using the Venkataraman method.
    Optimizes only the y-coordinates of the inner control points.
    optimization_method: "fixed_x", "variable_x", "variable_x_orthogonal", "fixed_x_orthogonal"
    """
    _ = le_tangent_vector
        # Choose x-coordinates for control points
    if optimization_method.startswith("variable_x"):
        paper_fixed_x_coords = variable_x_control_points(original_data, num_control_points_new)
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
        # Use appropriate error function based on optimization method
        if optimization_method == "variable_x_orthogonal":
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function="orthogonal_icp", return_max_error=False)
        elif optimization_method == "fixed_x_orthogonal":
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function="orthogonal_icp", return_max_error=False)
        else:
            fitting_error = calculate_single_bezier_fitting_error(control_points, original_data, error_function="euclidean", return_max_error=False)
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
        options=config.SLSQP_OPTIONS
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
                                          regularization_weight=0.01, optimization_method="minmax", 
                                          logger_func=None):
    """
    Builds a single Bezier curve using minmax optimization with orthogonal distance.
    Optimizes to minimize the maximum orthogonal distance error.
    First runs a full fixed-x optimization to get a good initial guess.
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
            max_distance, _ = calculate_orthogonal_error_minmax(control_points, original_data)
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

    # Stage 1: Run full fixed-x optimization to get a good initial guess
    _log_message(f"Stage 1: Running full fixed-x optimization for {is_upper_surface and 'upper' or 'lower'} surface...", logger_func)

    # Run the regular fixed-x optimization
    icp_result = build_single_venkatamaran_bezier(
        original_data=original_data,
        num_control_points_new=num_control_points_new,
        is_upper_surface=is_upper_surface,
        le_tangent_vector=le_tangent_vector,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        optimization_method="fixed_x",
        logger_func=logger_func
    )

    # Extract the y-coordinates of the inner control points from fixed-x result
    icp_inner_y = icp_result[1:-1, 1]  # Skip first and last points (start/end), take y-coordinates

    _log_message(f"Stage 1 complete: Fixed-x optimization finished for {is_upper_surface and 'upper' or 'lower'} surface", logger_func)

    # Stage 2: Use fixed-x result as initial guess for minmax optimization
    _log_message(f"Stage 2: Starting minmax optimization using fixed-x result as initial guess...", logger_func)

    # Reset the call counter for this stage
    if hasattr(objective_minmax, 'call_count'):
        del objective_minmax.call_count

    result = minimize(
        objective_minmax,
        icp_inner_y,  # Use fixed-x result as initial guess
        method='SLSQP',
        
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )

    if not result.success:
        _log_message(
            f"Minmax Bezier build failed. Using fixed-x solution. Reason: {result.message}",
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
    optimization_method="fixed_x",
    logger_func=None,
):
    """Build upper and lower single-segment Bézier curves simultaneously while
    enforcing G2 continuity (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.
    
    Supports optimization_method: "fixed_x", "fixed_x_orthogonal"
    """
    _log_message("Building coupled G2 Bezier curves using " + optimization_method + " optimization", logger_func)

    # Fixed abscissae from the paper
    paper_fixed_x_upper = get_paper_fixed_x_coords(True)  # upper
    paper_fixed_x_lower = get_paper_fixed_x_coords(False)  # lower

    inner_x_upper = paper_fixed_x_upper[1:-1]  # 8 values
    inner_x_lower = paper_fixed_x_lower[1:-1]
    n_inner = len(inner_x_upper)

    # Define start and end points
    start_point_upper = np.array([0.0, 0.0])  # LE at (0,0)
    end_point_upper = np.array([1.0, 0.0])    # TE at (1,0)
    start_point_lower = np.array([0.0, 0.0])  # LE at (0,0)
    end_point_lower = np.array([1.0, 0.0])    # TE at (1,0)

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
        
        # Choose error function based on optimization method
        if optimization_method == "fixed_x_orthogonal":
            error_func = "orthogonal_icp"
        else:
            error_func = "euclidean"
            
        err_u = calculate_single_bezier_fitting_error(ctrl_u, original_upper_data, error_function=error_func)
        err_l = calculate_single_bezier_fitting_error(ctrl_l, original_lower_data, error_function=error_func)

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
        method='SLSQP',
        
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )

    if not result.success:
        _log_message(f"Coupled Bezier build failed. Using initial guess. Reason: {result.message}", logger_func)
        var_y_final = initial_guess
    else:
        var_y_final = result.x

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final


def build_coupled_venkatamaran_beziers_variable_x(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    optimization_method="variable_x_g2",
    logger_func=None,
):
    """Build upper and lower single-segment Bézier curves simultaneously while
    enforcing G2 continuity (equal curvature) at the leading edge.
    
    Uses variable x-coordinates for control points instead of fixed paper coordinates.
    Supports both euclidean and orthogonal error metrics.
    """
    _log_message("Building coupled G2 Bezier curves with variable-x control points using " + optimization_method + " optimization", logger_func)

    # Choose x-coordinates for control points using variable-x strategy
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    paper_fixed_x_upper = variable_x_control_points(original_upper_data, num_control_points)
    paper_fixed_x_lower = variable_x_control_points(original_lower_data, num_control_points)

    inner_x_upper = paper_fixed_x_upper[1:-1]  # Skip first and last points
    inner_x_lower = paper_fixed_x_lower[1:-1]
    n_inner = len(inner_x_upper)

    # Define start and end points
    start_point_upper = np.array([0.0, 0.0])  # LE at (0,0)
    end_point_upper = np.array([1.0, 0.0])    # TE at (1,0)
    start_point_lower = np.array([0.0, 0.0])  # LE at (0,0)
    end_point_lower = np.array([1.0, 0.0])    # TE at (1,0)

    # Initial guesses by interpolation of raw data
    init_y_upper = np.interp(inner_x_upper, original_upper_data[:, 0], original_upper_data[:, 1])
    init_y_lower = np.interp(inner_x_lower, original_lower_data[:, 0], original_lower_data[:, 1])
    initial_guess = np.concatenate([init_y_upper, init_y_lower])

    # For orthogonal methods, use a two-stage approach to improve convergence
    if optimization_method == "variable_x_orthogonal_g2":
        _log_message("Using two-stage optimization for orthogonal variable-x G2 method...", logger_func)
        
        # Stage 1: Run Fixed-x optimization to get a good initial guess
        icp_upper, icp_lower = build_coupled_venkatamaran_beziers_variable_x(
            original_upper_data=original_upper_data,
            original_lower_data=original_lower_data,
            regularization_weight=regularization_weight,
            te_tangent_vector_upper=te_tangent_vector_upper,
            te_tangent_vector_lower=te_tangent_vector_lower,
            optimization_method="variable_x_g2",
            logger_func=logger_func,
        )
        
        # Extract the y-coordinates of the inner control points from fixed-x results
        icp_upper_inner_y = icp_upper[1:-1, 1]  # Skip first and last points, take y-coordinates
        icp_lower_inner_y = icp_lower[1:-1, 1]
        
        # Use Fixed-x result as improved initial guess
        initial_guess = np.concatenate([icp_upper_inner_y, icp_lower_inner_y])
        _log_message("Stage 1 complete: Fixed-x optimization finished as initial guess for variable-x G2", logger_func)

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
        
        # Use orthogonal error if requested
        if optimization_method == "variable_x_orthogonal_g2":
            err_u = calculate_single_bezier_fitting_error(ctrl_u, original_upper_data, error_function="orthogonal_minmax")
            err_l = calculate_single_bezier_fitting_error(ctrl_l, original_lower_data, error_function="orthogonal_minmax")
        else:  # variable_x_g2
            err_u = calculate_single_bezier_fitting_error(ctrl_u, original_upper_data, error_function="euclidean")
            err_l = calculate_single_bezier_fitting_error(ctrl_l, original_lower_data, error_function="euclidean")

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
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )

    if not result.success:
        if optimization_method == "variable_x_orthogonal_g2":
            _log_message(f"Orthogonal variable-x G2 optimization failed. Falling back to fixed-x method. Reason: {result.message}", logger_func)
            # Fall back to fixed-x method
            return build_coupled_venkatamaran_beziers_variable_x(
                original_upper_data=original_upper_data,
                original_lower_data=original_lower_data,
                regularization_weight=regularization_weight,
                te_tangent_vector_upper=te_tangent_vector_upper,
                te_tangent_vector_lower=te_tangent_vector_lower,
                optimization_method="variable_x_g2",
                logger_func=logger_func,
            )
        else:
            _log_message(f"Coupled variable-x Bezier build failed. Using initial guess. Reason: {result.message}", logger_func)
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
    optimization_method="minmax",
    logger_func=None,
):
    """Build upper and lower single-segment Bézier curves simultaneously using
    minmax optimization with orthogonal distance while enforcing G2 continuity
    (equal curvature) at the leading edge.

    Only the *y*-coordinates of the inner control points are optimised; *x*-values
    remain fixed to the Venkataraman paper.  The function returns a tuple
    ``(upper_ctrl_pts, lower_ctrl_pts)``.

    First runs a full fixed-x optimization to get a good initial guess.
    """
    _log_message("Building coupled G2 Bezier curves using minmax " + optimization_method + " optimization", logger_func)

    # Fixed abscissae from the paper
    paper_fixed_x_upper = get_paper_fixed_x_coords(True)  # upper
    paper_fixed_x_lower = get_paper_fixed_x_coords(False)  # lower

    inner_x_upper = paper_fixed_x_upper[1:-1]  # 8 values
    inner_x_lower = paper_fixed_x_lower[1:-1]
    n_inner = len(inner_x_upper)

    # Define start and end points
    start_point_upper = np.array([0.0, 0.0])  # LE at (0,0)
    end_point_upper = np.array([1.0, 0.0])    # TE at (1,0)
    start_point_lower = np.array([0.0, 0.0])  # LE at (0,0)
    end_point_lower = np.array([1.0, 0.0])    # TE at (1,0)

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
    # Stage 1: Run full fixed-x optimization to get a good initial guess
    _log_message("Stage 1: Running full fixed-x optimization for coupled surfaces...", logger_func)

    # Run the regular fixed-x optimization for coupled surfaces
    icp_upper, icp_lower = build_coupled_venkatamaran_beziers(
        original_upper_data=original_upper_data,
        original_lower_data=original_lower_data,
        regularization_weight=regularization_weight,
        te_tangent_vector_upper=te_tangent_vector_upper,
        te_tangent_vector_lower=te_tangent_vector_lower,
        optimization_method="fixed_x",
        logger_func=logger_func,
    )

    # Extract the y-coordinates of the inner control points from fixed-x results
    icp_upper_inner_y = icp_upper[1:-1, 1]  # Skip first and last points, take y-coordinates
    icp_lower_inner_y = icp_lower[1:-1, 1]

    # Combine into single vector for minmax optimization
    improved_initial_guess = np.concatenate([icp_upper_inner_y, icp_lower_inner_y])

    _log_message("Stage 1 complete: fixed-x optimization finished for coupled surfaces", logger_func)

    # Stage 2: Refine using minmax optimization
    _log_message("Stage 2: Starting minmax optimization using fixed-x result as initial guess...", logger_func)

    result = minimize(
        objective_minmax,
        improved_initial_guess,
        method='SLSQP',
        
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )

    if not result.success:
        _log_message(f"Coupled minmax Bezier build failed. Using fixed-x solution. Reason: {result.message}", logger_func)
        var_y_final = improved_initial_guess
    else:
        var_y_final = result.x
        _log_message("Minmax optimization successful", logger_func)

    ctrl_upper_final, ctrl_lower_final = _assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final
