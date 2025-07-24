import numpy as np
from scipy.optimize import minimize
from core import config
from core.solver.error_functions import calculate_euclidean_error, calculate_orthogonal_error
from core.solver.error_functions import calculate_single_bezier_fitting_error
from utils.bezier_utils import leading_edge_curvature
from core.solver.solver_helpers import get_fixed_inner_x_coords, get_initial_guess_inner_y, assemble_polygons, smoothness_penalty, te_tangent_constraint_factory, g2_constraint_factory

def build_coupled_bezier_fixed_x_msr(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
):
    """
    Coupled fixed-x Bezier optimizer (G2 at LE, tangency at TE, MSR objective).
    """
    # 1. Setup x-coordinates
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    paper_fixed_x_upper = get_fixed_inner_x_coords(True, num_control_points)
    paper_fixed_x_lower = get_fixed_inner_x_coords(False, num_control_points)
    inner_x_upper = paper_fixed_x_upper
    inner_x_lower = paper_fixed_x_lower
    n_inner = len(inner_x_upper)

    # 2. Initial guess: interpolate y at fixed x for both surfaces, concatenate
    init_y_upper = get_initial_guess_inner_y(original_upper_data, inner_x_upper)
    init_y_lower = get_initial_guess_inner_y(original_lower_data, inner_x_lower)
    initial_guess = np.concatenate([init_y_upper, init_y_lower])

    # 4. Objective
    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error

    def objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y, inner_x_upper, inner_x_lower, original_upper_data, original_lower_data)
        err_u = error_func(original_upper_data, ctrl_u)
        err_l = error_func(original_lower_data, ctrl_l)
        if isinstance(err_u, tuple): err_u = err_u[0]
        if isinstance(err_l, tuple): err_l = err_l[0]
        smooth = smoothness_penalty(ctrl_u) + smoothness_penalty(ctrl_l)
        return err_u + err_l + regularization_weight * smooth

    # 5. Constraints
    constraints = []
    # TE tangency (upper)
    tx_u, ty_u = te_tangent_vector_upper
    px_n_u, py_n_u = 1.0, 0.0
    px_n1_u = inner_x_upper[-1]
    if not np.isclose(tx_u, 0.0):
        constraints.append({"type": "eq", "fun": te_tangent_constraint_factory(tx_u, ty_u, px_n_u, py_n_u, px_n1_u, n_inner - 1)})
    # TE tangency (lower)
    tx_l, ty_l = te_tangent_vector_lower
    px_n_l, py_n_l = 1.0, 0.0
    px_n1_l = inner_x_lower[-1]
    if not np.isclose(tx_l, 0.0):
        constraints.append({"type": "eq", "fun": te_tangent_constraint_factory(tx_l, ty_l, px_n_l, py_n_l, px_n1_l, -1)})
    # G2 at LE
    constraints.append({"type": "eq", "fun": g2_constraint_factory(inner_x_upper, inner_x_lower, original_upper_data, original_lower_data)})

    # 6. Minimize
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not result.success and logger_func:
        logger_func(f"Coupled fixed-x MSR optimization failed. Using initial guess. Reason: {result.message}")
        var_y_final = initial_guess
    else:
        var_y_final = result.x

    ctrl_upper_final, ctrl_lower_final = assemble_polygons(var_y_final, inner_x_upper, inner_x_lower, original_upper_data, original_lower_data)
    return ctrl_upper_final, ctrl_lower_final 

def build_coupled_bezier_fixed_x_minmax(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
):
    """
    Coupled fixed-x Bezier optimizer (G2 at LE, tangency at TE, minmax objective).
    Two-stage: MSR for initial guess, then true minmax optimization.
    """
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    paper_fixed_x_upper = get_fixed_inner_x_coords(True, num_control_points)
    paper_fixed_x_lower = get_fixed_inner_x_coords(False, num_control_points)
    inner_x_upper = paper_fixed_x_upper
    inner_x_lower = paper_fixed_x_lower
    n_inner = len(inner_x_upper)
    init_y_upper = get_initial_guess_inner_y(original_upper_data, inner_x_upper)
    init_y_lower = get_initial_guess_inner_y(original_lower_data, inner_x_lower)
    initial_guess = np.concatenate([init_y_upper, init_y_lower])
    def assemble_polygons(var_y):
        y_u = var_y[:n_inner]
        y_l = var_y[n_inner:]
        ctrl_upper = np.zeros((n_inner + 2, 2))
        ctrl_lower = np.zeros((n_inner + 2, 2))
        ctrl_upper[0] = [0.0, 0.0]
        ctrl_upper[1:-1, 0] = inner_x_upper
        ctrl_upper[1:-1, 1] = y_u
        ctrl_upper[-1] = [1.0, float(original_upper_data[-1, 1])]
        ctrl_lower[0] = [0.0, 0.0]
        ctrl_lower[1:-1, 0] = inner_x_lower
        ctrl_lower[1:-1, 1] = y_l
        ctrl_lower[-1] = [1.0, float(original_lower_data[-1, 1])]
        return ctrl_upper, ctrl_lower
    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error
    def msr_objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        err_u = error_func(original_upper_data, ctrl_u)
        err_l = error_func(original_lower_data, ctrl_l)
        if isinstance(err_u, tuple): err_u = err_u[0]
        if isinstance(err_l, tuple): err_l = err_l[0]
        def _smooth(ctrl):
            if len(ctrl) <= 2: return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)
        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return err_u + err_l + regularization_weight * smooth
    def minmax_objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        _, max_err_u, _ = calculate_single_bezier_fitting_error(
            ctrl_u, original_upper_data, error_function=error_function, return_max_error=True)
        _, max_err_l, _ = calculate_single_bezier_fitting_error(
            ctrl_l, original_lower_data, error_function=error_function, return_max_error=True)
        combined_max_error = max(float(max_err_u), float(max_err_l))
        def _smooth(ctrl):
            if len(ctrl) <= 2: return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)
        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return combined_max_error + regularization_weight * smooth
    constraints = []
    tx_u, ty_u = te_tangent_vector_upper
    px_n_u, py_n_u = 1.0, 0.0
    px_n1_u = inner_x_upper[-1]
    if not np.isclose(tx_u, 0.0):
        def te_tan_upper(var_y):
            y_nm1 = var_y[n_inner - 1]
            return y_nm1 * tx_u - (py_n_u * tx_u - (px_n_u - px_n1_u) * ty_u)
        constraints.append({"type": "eq", "fun": te_tan_upper})
    tx_l, ty_l = te_tangent_vector_lower
    px_n_l, py_n_l = 1.0, 0.0
    px_n1_l = inner_x_lower[-1]
    if not np.isclose(tx_l, 0.0):
        def te_tan_lower(var_y):
            y_nm1_l = var_y[-1]
            return y_nm1_l * tx_l - (py_n_l * tx_l - (px_n_l - px_n1_l) * ty_l)
        constraints.append({"type": "eq", "fun": te_tan_lower})
    def g2_constraint(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)
    constraints.append({"type": "eq", "fun": g2_constraint})
    if logger_func:
        logger_func("Stage 1: Running coupled fixed-x MSR optimization for initial guess...")
    msr_result = minimize(
        msr_objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not msr_result.success:
        if logger_func:
            logger_func(f"Coupled fixed-x MSR optimization failed. Using initial guess. Reason: {msr_result.message}")
        msr_inner = initial_guess
    else:
        msr_inner = msr_result.x
    if logger_func:
        logger_func("Stage 2: Running coupled fixed-x minmax optimization...")
    minmax_result = minimize(
        minmax_objective,
        msr_inner,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not minmax_result.success:
        if logger_func:
            logger_func(f"Coupled fixed-x minmax optimization failed. Using MSR solution. Reason: {minmax_result.message}")
        final_inner = msr_inner
    else:
        final_inner = minmax_result.x
    ctrl_upper_final, ctrl_lower_final = assemble_polygons(final_inner)
    return ctrl_upper_final, ctrl_lower_final 

def build_coupled_bezier_variable_x_msr(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
):
    """
    Coupled variable-x Bezier optimizer (G2 at LE, tangency at TE, MSR objective).
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    """
    from utils.control_point_utils import variable_x_control_points
    paper_x_upper = variable_x_control_points(original_upper_data, config.NUM_CONTROL_POINTS_SINGLE_BEZIER)
    paper_x_lower = variable_x_control_points(original_lower_data, config.NUM_CONTROL_POINTS_SINGLE_BEZIER)
    inner_x_upper = paper_x_upper[1:-1]
    inner_x_lower = paper_x_lower[1:-1]
    n_inner = len(inner_x_upper)
    init_y_upper = np.interp(inner_x_upper, original_upper_data[:, 0], original_upper_data[:, 1])
    init_y_lower = np.interp(inner_x_lower, original_lower_data[:, 0], original_lower_data[:, 1])
    initial_guess = np.concatenate([init_y_upper, init_y_lower])
    def assemble_polygons(var_y):
        y_u = var_y[:n_inner]
        y_l = var_y[n_inner:]
        ctrl_upper = np.zeros((n_inner + 2, 2))
        ctrl_lower = np.zeros((n_inner + 2, 2))
        ctrl_upper[0] = [0.0, 0.0]
        ctrl_upper[1:-1, 0] = inner_x_upper
        ctrl_upper[1:-1, 1] = y_u
        ctrl_upper[-1] = [1.0, float(original_upper_data[-1, 1])]
        ctrl_lower[0] = [0.0, 0.0]
        ctrl_lower[1:-1, 0] = inner_x_lower
        ctrl_lower[1:-1, 1] = y_l
        ctrl_lower[-1] = [1.0, float(original_lower_data[-1, 1])]
        return ctrl_upper, ctrl_lower
    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error
    def objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        err_u = error_func(original_upper_data, ctrl_u)
        err_l = error_func(original_lower_data, ctrl_l)
        if isinstance(err_u, tuple): err_u = err_u[0]
        if isinstance(err_l, tuple): err_l = err_l[0]
        def _smooth(ctrl):
            if len(ctrl) <= 2: return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)
        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return err_u + err_l + regularization_weight * smooth
    constraints = []
    tx_u, ty_u = te_tangent_vector_upper
    px_n_u, py_n_u = 1.0, 0.0
    px_n1_u = inner_x_upper[-1]
    if not np.isclose(tx_u, 0.0):
        def te_tan_upper(var_y):
            y_nm1 = var_y[n_inner - 1]
            return y_nm1 * tx_u - (py_n_u * tx_u - (px_n_u - px_n1_u) * ty_u)
        constraints.append({"type": "eq", "fun": te_tan_upper})
    tx_l, ty_l = te_tangent_vector_lower
    px_n_l, py_n_l = 1.0, 0.0
    px_n1_l = inner_x_lower[-1]
    if not np.isclose(tx_l, 0.0):
        def te_tan_lower(var_y):
            y_nm1_l = var_y[-1]
            return y_nm1_l * tx_l - (py_n_l * tx_l - (px_n_l - px_n1_l) * ty_l)
        constraints.append({"type": "eq", "fun": te_tan_lower})
    def g2_constraint(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)
    constraints.append({"type": "eq", "fun": g2_constraint})
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not result.success and logger_func:
        logger_func(f"Coupled variable-x MSR optimization failed. Using initial guess. Reason: {result.message}")
        var_y_final = initial_guess
    else:
        var_y_final = result.x
    ctrl_upper_final, ctrl_lower_final = assemble_polygons(var_y_final)
    return ctrl_upper_final, ctrl_lower_final 

def build_coupled_bezier_variable_x_minmax(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
):
    """
    Coupled variable-x Bezier optimizer (G2 at LE, tangency at TE, minmax objective).
    Two-stage: MSR for initial guess, then true minmax optimization.
    """
    from utils.control_point_utils import variable_x_control_points
    paper_x_upper = variable_x_control_points(original_upper_data, config.NUM_CONTROL_POINTS_SINGLE_BEZIER)
    paper_x_lower = variable_x_control_points(original_lower_data, config.NUM_CONTROL_POINTS_SINGLE_BEZIER)
    inner_x_upper = paper_x_upper[1:-1]
    inner_x_lower = paper_x_lower[1:-1]
    n_inner = len(inner_x_upper)
    init_y_upper = np.interp(inner_x_upper, original_upper_data[:, 0], original_upper_data[:, 1])
    init_y_lower = np.interp(inner_x_lower, original_lower_data[:, 0], original_lower_data[:, 1])
    initial_guess = np.concatenate([init_y_upper, init_y_lower])
    def assemble_polygons(var_y):
        y_u = var_y[:n_inner]
        y_l = var_y[n_inner:]
        ctrl_upper = np.zeros((n_inner + 2, 2))
        ctrl_lower = np.zeros((n_inner + 2, 2))
        ctrl_upper[0] = [0.0, 0.0]
        ctrl_upper[1:-1, 0] = inner_x_upper
        ctrl_upper[1:-1, 1] = y_u
        ctrl_upper[-1] = [1.0, float(original_upper_data[-1, 1])]
        ctrl_lower[0] = [0.0, 0.0]
        ctrl_lower[1:-1, 0] = inner_x_lower
        ctrl_lower[1:-1, 1] = y_l
        ctrl_lower[-1] = [1.0, float(original_lower_data[-1, 1])]
        return ctrl_upper, ctrl_lower
    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error
    def msr_objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        err_u = error_func(original_upper_data, ctrl_u)
        err_l = error_func(original_lower_data, ctrl_l)
        if isinstance(err_u, tuple): err_u = err_u[0]
        if isinstance(err_l, tuple): err_l = err_l[0]
        def _smooth(ctrl):
            if len(ctrl) <= 2: return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)
        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return err_u + err_l + regularization_weight * smooth
    def minmax_objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        _, max_err_u, _ = calculate_single_bezier_fitting_error(
            ctrl_u, original_upper_data, error_function=error_function, return_max_error=True)
        _, max_err_l, _ = calculate_single_bezier_fitting_error(
            ctrl_l, original_lower_data, error_function=error_function, return_max_error=True)
        combined_max_error = max(float(max_err_u), float(max_err_l))
        def _smooth(ctrl):
            if len(ctrl) <= 2: return 0.0
            return np.sum(np.diff(ctrl[:, 1], n=2) ** 2)
        smooth = _smooth(ctrl_u) + _smooth(ctrl_l)
        return combined_max_error + regularization_weight * smooth
    constraints = []
    tx_u, ty_u = te_tangent_vector_upper
    px_n_u, py_n_u = 1.0, 0.0
    px_n1_u = inner_x_upper[-1]
    if not np.isclose(tx_u, 0.0):
        def te_tan_upper(var_y):
            y_nm1 = var_y[n_inner - 1]
            return y_nm1 * tx_u - (py_n_u * tx_u - (px_n_u - px_n1_u) * ty_u)
        constraints.append({"type": "eq", "fun": te_tan_upper})
    tx_l, ty_l = te_tangent_vector_lower
    px_n_l, py_n_l = 1.0, 0.0
    px_n1_l = inner_x_lower[-1]
    if not np.isclose(tx_l, 0.0):
        def te_tan_lower(var_y):
            y_nm1_l = var_y[-1]
            return y_nm1_l * tx_l - (py_n_l * tx_l - (px_n_l - px_n1_l) * ty_l)
        constraints.append({"type": "eq", "fun": te_tan_lower})
    def g2_constraint(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)
    constraints.append({"type": "eq", "fun": g2_constraint})
    if logger_func:
        logger_func("Stage 1: Running coupled fixed-x MSR optimization for initial guess...")
    msr_result = minimize(
        msr_objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not msr_result.success:
        if logger_func:
            logger_func(f"Coupled variable-x MSR optimization failed. Using initial guess. Reason: {msr_result.message}")
        msr_inner = initial_guess
    else:
        msr_inner = msr_result.x
    if logger_func:
        logger_func("Stage 2: Running coupled variable-x minmax optimization...")
    minmax_result = minimize(
        minmax_objective,
        msr_inner,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not minmax_result.success:
        if logger_func:
            logger_func(f"Coupled variable-x minmax optimization failed. Using MSR solution. Reason: {minmax_result.message}")
        final_inner = msr_inner
    else:
        final_inner = minmax_result.x
    ctrl_upper_final, ctrl_lower_final = assemble_polygons(final_inner)
    return ctrl_upper_final, ctrl_lower_final 