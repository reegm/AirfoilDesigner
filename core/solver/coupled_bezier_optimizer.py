import numpy as np
from scipy.optimize import minimize
from core import config
# from core.solver.error_functions import calculate_euclidean_error, calculate_orthogonal_error  # Deprecated
from core.solver.error_functions import calculate_single_bezier_fitting_error
from utils.bezier_utils import leading_edge_curvature
from core.solver.solver_helpers import (
    get_fixed_inner_x_partition,
    build_control_points_with_fixed,
    smoothness_penalty,
    extract_free_y_from_ctrl,
    make_build_ctrl_fn,
    make_residuals_fn,
    run_minmax_stage,
    get_initial_guess_inner_y,
    g2_constraint_factory_fixed_x
)
from core.solver.bezier_optimizer import build_single_bezier_msr

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
    The pre-trailing-edge control point is fixed and not optimized for both surfaces.
    TE tangency constraints are not needed and are removed.
    """
    if logger_func:
        logger_func("Running coupled fixed-x MSR optimization...")
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    # Partition for upper and lower
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u = get_fixed_inner_x_partition(
        True, num_control_points, original_upper_data, te_tangent_vector_upper, te_y_upper)
    inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l = get_fixed_inner_x_partition(
        False, num_control_points, original_lower_data, te_tangent_vector_lower, te_y_lower)
    n_free_u = len(free_idx_u)
    n_free_l = len(free_idx_l)
    init_y_upper_full = get_initial_guess_inner_y(original_upper_data, inner_x_upper)
    init_y_lower_full = get_initial_guess_inner_y(original_lower_data, inner_x_lower)
    init_y_upper = init_y_upper_full[free_idx_u]
    init_y_lower = init_y_lower_full[free_idx_l]
    initial_guess = np.concatenate([init_y_upper, init_y_lower])
    if error_function == "orthogonal":
        def error_func(data, ctrl):
            return calculate_single_bezier_fitting_error(ctrl, data, error_function="orthogonal")
    else:
        def error_func(data, ctrl):
            return calculate_single_bezier_fitting_error(ctrl, data, error_function="euclidean")
    def assemble_polygons(var_y):
        y_u = var_y[:n_free_u]
        y_l = var_y[n_free_u:]
        ctrl_upper = build_control_points_with_fixed(y_u, inner_x_upper, float(original_upper_data[-1, 1]), free_idx_u, fixed_idx_u, fixed_y_u)
        ctrl_lower = build_control_points_with_fixed(y_l, inner_x_lower, float(original_lower_data[-1, 1]), free_idx_l, fixed_idx_l, fixed_y_l)
        return ctrl_upper, ctrl_lower
    def objective(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y)
        err_u = error_func(original_upper_data, ctrl_u)
        err_l = error_func(original_lower_data, ctrl_l)
        if isinstance(err_u, tuple): err_u = err_u[0]
        if isinstance(err_l, tuple): err_l = err_l[0]
        smooth = smoothness_penalty(ctrl_u) + smoothness_penalty(ctrl_l)
        return err_u + err_l + regularization_weight * smooth
    constraints = []  # TE tangency constraints removed
    constraints.append({"type": "eq", "fun": g2_constraint_factory_fixed_x(
        inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u, te_y_upper,
        inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l, te_y_lower,
        original_upper_data, original_lower_data)})
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
        if logger_func:
            logger_func("Coupled fixed-x MSR optimization succeeded.")
        var_y_final = result.x
    ctrl_upper_final, ctrl_lower_final = assemble_polygons(var_y_final)
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
    The pre-trailing-edge control point is fixed and not optimized for both surfaces.
    TE tangency constraints are not needed and are removed.
    """
    if logger_func:
        logger_func("Stage 1: Running uncoupled fixed-x MSR optimization for initial guess...")
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    # Stage 1: Use build_single_bezier_msr for both upper and lower surfaces
    ctrl_upper = build_single_bezier_msr(
        original_upper_data,
        num_control_points,
        True,
        None,  # le_tangent_vector not used in fixed-x
        te_tangent_vector_upper,
        regularization_weight=0.001,
        error_function="euclidean",
        logger_func=logger_func,
    )
    ctrl_lower = build_single_bezier_msr(
        original_lower_data,
        num_control_points,
        False,
        None,  # le_tangent_vector not used in fixed-x
        te_tangent_vector_lower,
        regularization_weight=0.0,
        error_function="euclidean",
        logger_func=logger_func,
    )
    if logger_func:
        logger_func("Stage 2: Running coupled fixed-x minmax optimization...")
    # Extract free y variables for minmax stage
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u = get_fixed_inner_x_partition(
        True, num_control_points, original_upper_data, te_tangent_vector_upper, te_y_upper)
    inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l = get_fixed_inner_x_partition(
        False, num_control_points, original_lower_data, te_tangent_vector_lower, te_y_lower)
    msr_y_upper = extract_free_y_from_ctrl(ctrl_upper, free_idx_u)
    msr_y_lower = extract_free_y_from_ctrl(ctrl_lower, free_idx_l)
    msr_y = np.concatenate([msr_y_upper, msr_y_lower])
    n_free_u = len(free_idx_u)
    n_free_l = len(free_idx_l)
    def build_ctrls(y):
        y_u = y[:n_free_u]
        y_l = y[n_free_u:]
        ctrl_upper = build_control_points_with_fixed(y_u, inner_x_upper, te_y_upper, free_idx_u, fixed_idx_u, fixed_y_u)
        ctrl_lower = build_control_points_with_fixed(y_l, inner_x_lower, te_y_lower, free_idx_l, fixed_idx_l, fixed_y_l)
        return ctrl_upper, ctrl_lower
    def smooth_coupled(ctrls):
        ctrl_u, ctrl_l = ctrls
        return smoothness_penalty(ctrl_u) + smoothness_penalty(ctrl_l)
    def residuals_fn_coupled(ctrls):
        ctrl_u, ctrl_l = ctrls
        efun = error_function
        if not efun.endswith('_minmax'):
            efun = efun + '_minmax'
        res_u = make_residuals_fn(original_upper_data, error_function, minmax=True)(ctrl_u)
        res_l = make_residuals_fn(original_lower_data, error_function, minmax=True)(ctrl_l)
        return np.concatenate([res_u, res_l])
    def g2_constraint(y_aug):
        ctrls = build_ctrls(y_aug[:-1])
        from utils.bezier_utils import leading_edge_curvature
        return leading_edge_curvature(ctrls[0]) + leading_edge_curvature(ctrls[1])
    constraints_fns = [("eq", g2_constraint)]
    control_polys = run_minmax_stage(
        initial_y=msr_y,
        build_control_points_fn=build_ctrls,
        residuals_fn=residuals_fn_coupled,
        regularization_weight=regularization_weight,
        smoothness_fn=smooth_coupled,
        constraints_fns=constraints_fns,
        logger_func=logger_func
    )
    if logger_func:
        logger_func("Stage 2: Coupled fixed-x minmax optimization completed.")
    return control_polys

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
    Coupled free-x Bezier optimizer (G2 at LE, tangency at TE, MSR objective).
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    The initial guess for the pre-TE y is set using the TE vector.
    """
    from utils.control_point_utils import variable_x_control_points
    if logger_func:
        logger_func("Running coupled free-x MSR optimization...")
    paper_x_upper = variable_x_control_points(original_upper_data, config.NUM_CONTROL_POINTS_SINGLE_BEZIER)
    paper_x_lower = variable_x_control_points(original_lower_data, config.NUM_CONTROL_POINTS_SINGLE_BEZIER)
    inner_x_upper = paper_x_upper[1:-1]
    inner_x_lower = paper_x_lower[1:-1]
    n_inner = len(inner_x_upper)
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    init_y_upper = np.interp(inner_x_upper, original_upper_data[:, 0], original_upper_data[:, 1])
    init_y_lower = np.interp(inner_x_lower, original_lower_data[:, 0], original_lower_data[:, 1])
    # Set pre-TE y using TE vector
    x_n_minus_1_u = inner_x_upper[-1]
    x_te = 1.0
    tx_te_u, ty_te_u = te_tangent_vector_upper
    if abs(tx_te_u) >= 1e-12:
        init_y_upper[-1] = te_y_upper - (x_te - x_n_minus_1_u) * (ty_te_u / tx_te_u)
    x_n_minus_1_l = inner_x_lower[-1]
    tx_te_l, ty_te_l = te_tangent_vector_lower
    if abs(tx_te_l) >= 1e-12:
        init_y_lower[-1] = te_y_lower - (x_te - x_n_minus_1_l) * (ty_te_l / tx_te_l)
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
        def error_func(data, ctrl):
            return calculate_single_bezier_fitting_error(ctrl, data, error_function="orthogonal")
    else:
        def error_func(data, ctrl):
            return calculate_single_bezier_fitting_error(ctrl, data, error_function="euclidean")
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
        logger_func(f"Coupled free-x MSR optimization failed. Using initial guess. Reason: {result.message}")
        var_y_final = initial_guess
    else:
        if logger_func:
            logger_func("Coupled free-x MSR optimization succeeded.")
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
    Coupled free-x Bezier optimizer (G2 at LE, tangency at TE, minmax objective).
    Two-stage: MSR for initial guess, then true minmax optimization.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    The initial guess for the pre-TE y is set using the TE vector.
    """
    from utils.control_point_utils import variable_x_control_points
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    if logger_func:
        logger_func("Stage 1: Running uncoupled fixed-x MSR optimization for initial guess...")
    # Stage 1: Use build_single_bezier_msr for both upper and lower surfaces
    ctrl_upper = build_single_bezier_msr(
        original_upper_data,
        num_control_points,
        True,
        None,  # le_tangent_vector not used in fixed-x
        te_tangent_vector_upper,
        regularization_weight=0.001,
        error_function="euclidean",
        logger_func=logger_func,
    )
    ctrl_lower = build_single_bezier_msr(
        original_lower_data,
        num_control_points,
        False,
        None,  # le_tangent_vector not used in fixed-x
        te_tangent_vector_lower,
        regularization_weight=0.0,
        error_function="euclidean",
        logger_func=logger_func,
    )
    if logger_func:
        logger_func("Stage 2: Running coupled free-x minmax optimization...")
    # Extract free y variables for minmax stage
    paper_x_upper = variable_x_control_points(original_upper_data, num_control_points)
    paper_x_lower = variable_x_control_points(original_lower_data, num_control_points)
    inner_x_upper = paper_x_upper[1:-1]
    inner_x_lower = paper_x_lower[1:-1]
    n_inner = len(inner_x_upper)
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    msr_y_upper = ctrl_upper[1:-1, 1]
    msr_y_lower = ctrl_lower[1:-1, 1]
    msr_y = np.concatenate([msr_y_upper, msr_y_lower])
    def build_ctrls(y):
        y_u = y[:n_inner]
        y_l = y[n_inner:]
        ctrl_upper = np.zeros((n_inner + 2, 2))
        ctrl_lower = np.zeros((n_inner + 2, 2))
        ctrl_upper[0] = [0.0, 0.0]
        ctrl_upper[1:-1, 0] = inner_x_upper
        ctrl_upper[1:-1, 1] = y_u
        ctrl_upper[-1] = [1.0, te_y_upper]
        ctrl_lower[0] = [0.0, 0.0]
        ctrl_lower[1:-1, 0] = inner_x_lower
        ctrl_lower[1:-1, 1] = y_l
        ctrl_lower[-1] = [1.0, te_y_lower]
        return ctrl_upper, ctrl_lower
    def smooth_coupled(ctrls):
        ctrl_u, ctrl_l = ctrls
        return smoothness_penalty(ctrl_u) + smoothness_penalty(ctrl_l)
    def residuals_fn_coupled(ctrls):
        ctrl_u, ctrl_l = ctrls
        efun = error_function
        if not efun.endswith('_minmax'):
            efun = efun + '_minmax'
        res_u = make_residuals_fn(original_upper_data, error_function, minmax=True)(ctrl_u)
        res_l = make_residuals_fn(original_lower_data, error_function, minmax=True)(ctrl_l)
        return np.concatenate([res_u, res_l])
    def g2_constraint(y_aug):
        ctrls = build_ctrls(y_aug[:-1])
        from utils.bezier_utils import leading_edge_curvature
        return leading_edge_curvature(ctrls[0]) + leading_edge_curvature(ctrls[1])
    constraints_fns = [("eq", g2_constraint)]
    control_polys = run_minmax_stage(
        initial_y=msr_y,
        build_control_points_fn=build_ctrls,
        residuals_fn=residuals_fn_coupled,
        regularization_weight=regularization_weight,
        smoothness_fn=smooth_coupled,
        constraints_fns=constraints_fns,
        logger_func=logger_func
    )
    if logger_func:
        logger_func("Stage 2: Coupled free-x minmax optimization completed.")
    return control_polys 