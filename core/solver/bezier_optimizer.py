import numpy as np
from scipy.optimize import minimize
from core import config
from core.solver.error_functions import calculate_single_bezier_fitting_error
from core.solver.solver_helpers import (
    get_fixed_inner_x_partition,
    build_control_points_with_fixed,
    smoothness_penalty,
    extract_free_y_from_ctrl,
    make_build_ctrl_fn,
    make_residuals_fn,
    run_minmax_stage,
    get_initial_guess_inner_y
)

# --- MSR (least-squares) objective ---
def msr_objective(variables_y, build_control_points, original_data, error_function, regularization_weight):
    control_points = build_control_points(variables_y)
    errors = error_function(original_data, control_points, return_max_error=False)
    if isinstance(errors, tuple):
        errors = errors[0]
    smoothness_penalty = 0.0
    if len(control_points) > 2:
        diffs = np.diff(control_points[:, 1], n=2)
        smoothness_penalty = np.sum(diffs ** 2)
    return errors + regularization_weight * smoothness_penalty

# --- Minmax objective ---
def minmax_objective(variables_y, build_control_points, original_data, error_function, regularization_weight):
    control_points = build_control_points(variables_y)
    _, max_error, _ = calculate_single_bezier_fitting_error(
        control_points, original_data, error_function=error_function, return_max_error=True
    )
    smoothness_penalty = 0.0
    if len(control_points) > 2:
        diffs = np.diff(control_points[:, 1], n=2)
        smoothness_penalty = np.sum(diffs ** 2)
    return max_error + regularization_weight * smoothness_penalty

# --- MSR (least-squares) optimizer ---
def build_single_bezier_msr(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=0.00,
    error_function="euclidean",
    logger_func=None,
):
    """
    Fixed-x single Bezier optimizer using mean squared residual (least-squares) objective.
    The pre-trailing-edge control point is fixed and not optimized.
    TE tangency constraint is not needed and is removed.
    """
    if logger_func:
        logger_func("Running fixed-x MSR optimization...")
    te_y = float(original_data[-1, 1])
    fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, original_data, te_tangent_vector, te_y)
    initial_guess_inner_y_full = get_initial_guess_inner_y(original_data, fixed_inner_x_coords)
    initial_guess_inner_y = initial_guess_inner_y_full[free_indices]
    build_ctrl = make_build_ctrl_fn(fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    if error_function == "orthogonal":
        def error_func(ctrl):
            return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="orthogonal", return_max_error=False)
    else:
        def error_func(ctrl):
            return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="euclidean", return_max_error=False)
    def obj(variables_y):
        ctrl = build_ctrl(variables_y)
        errors = error_func(ctrl)
        if isinstance(errors, tuple):
            errors = errors[0]
        return errors + regularization_weight * smoothness_penalty(ctrl)
    constraints = []  # TE tangency constraint removed
    result = minimize(
        obj,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not result.success:
        if logger_func:
            logger_func(f"Fixed-x MSR optimization failed. Using initial guess. Reason: {result.message}")
        final_inner_y = initial_guess_inner_y
    else:
        if logger_func:
            logger_func("Fixed-x MSR optimization succeeded.")
        final_inner_y = result.x
    control_points = build_ctrl(final_inner_y)
    return control_points

def constrained_minmax_fit(
    initial_y,
    build_control_points_fn,
    error_function_fn,
    regularization_weight=0.0,
    smoothness_fn=None,
):
    n_y = len(initial_y)

    def full_objective(y_augmented):
        y = y_augmented[:-1]
        t = y_augmented[-1]
        ctrl = build_control_points_fn(y)
        smooth = smoothness_fn(ctrl) if smoothness_fn else 0.0
        val = t + regularization_weight * smooth
        # print(f"[OBJ] t = {t:.6e}, smooth = {smooth:.6e}, total = {val:.6e}")
        return val

    def constraint_factory(i):
        def constr(y_aug):
            y = y_aug[:-1]
            t = y_aug[-1]
            ctrl = build_control_points_fn(y)
            residuals = error_function_fn(ctrl)
            ci = t - abs(residuals[i])
            # print(f"[C{i}] |e[{i}]| = {abs(residuals[i]):.6e}, t = {t:.6e}, c = {ci:.6e}")
            return ci
        return constr

    ctrl0 = build_control_points_fn(initial_y)
    residuals0 = error_function_fn(ctrl0)
    if not isinstance(residuals0, np.ndarray):
        raise ValueError("error_function_fn must return a vector of residuals")

    t0 = np.max(np.abs(residuals0)) * 1.05
    y0_augmented = np.concatenate([initial_y, [t0]])

    constraints = [{"type": "ineq", "fun": constraint_factory(i)} for i in range(len(residuals0))]

    result = minimize(
        full_objective,
        y0_augmented,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )

    return result


#--- Minmax optimizer ---
def build_single_bezier_minmax(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=0.001,
    error_function="euclidean",
    logger_func=None,
):
    """
    Fixed-x single Bezier optimizer using minmax objective.
    The pre-trailing-edge control point is fixed and not optimized.
    TE tangency constraint is not needed and is removed.
    """
    # Stage 1: Use build_single_bezier_msr for initial guess (always fixed-x, euclidean)
    control_points = build_single_bezier_msr(
        original_data,
        num_control_points_new,
        is_upper_surface,
        le_tangent_vector,
        te_tangent_vector,
        regularization_weight=0.001,
        error_function="euclidean",
        logger_func=logger_func,
    )
    if logger_func:
        logger_func("Running fixed-x minmax optimization...")
    te_y = float(original_data[-1, 1])
    fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, original_data, te_tangent_vector, te_y)
    msr_inner_y = extract_free_y_from_ctrl(control_points, free_indices)
    build_ctrl = make_build_ctrl_fn(fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    residuals_fn = make_residuals_fn(original_data, error_function, minmax=True)
    control_points = run_minmax_stage(
        initial_y=msr_inner_y,
        build_control_points_fn=build_ctrl,
        residuals_fn=residuals_fn,
        regularization_weight=regularization_weight,
        smoothness_fn=smoothness_penalty,
        constraints_fns=None,
        logger_func=logger_func
    )
    if logger_func:
        logger_func("Fixed-x minmax optimization completed.")
    return control_points

def build_single_bezier_variable_x_msr(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=config.DEFAULT_REGULARIZATION_WEIGHT,
    error_function="euclidean",
    logger_func=None,
):
    """
    Uncoupled free-x single Bezier optimizer using mean squared residual (least-squares) objective.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    The initial guess for the pre-TE y is set using the TE vector.
    """
    from utils.control_point_utils import variable_x_control_points
    if logger_func:
        logger_func("Running free-x MSR optimization for initial guess...")
    te_y = float(original_data[-1, 1])
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    if num_control_points_new != len(paper_x_coords):
        num_control_points_new = len(paper_x_coords)
    fixed_inner_x_coords = paper_x_coords[1:-1]
    initial_guess_inner_y = np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])
    # Set pre-TE y using TE vector
    x_n_minus_1 = fixed_inner_x_coords[-1]
    x_te = 1.0
    tx_te, ty_te = te_tangent_vector
    if abs(tx_te) >= 1e-12:
        initial_guess_inner_y[-1] = te_y - (x_te - x_n_minus_1) * (ty_te / tx_te)
    def build_ctrl(variables_y):
        ctrl = np.zeros((len(variables_y) + 2, 2))
        ctrl[0] = np.array([0.0, 0.0])
        ctrl[1:-1, 0] = fixed_inner_x_coords
        ctrl[1:-1, 1] = variables_y
        ctrl[-1] = np.array([1.0, te_y])
        return ctrl
    if error_function == "orthogonal":
        def error_func(ctrl):
            return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="orthogonal", return_max_error=False)
    else:
        def error_func(ctrl):
            return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="euclidean", return_max_error=False)
    def obj(variables_y):
        ctrl = build_ctrl(variables_y)
        errors = error_func(ctrl)
        if isinstance(errors, tuple):
            errors = errors[0]
        return errors + regularization_weight * smoothness_penalty(ctrl)
    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x_coords[-1]
    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    if not np.isclose(tx_te, 0.0):
        constraints.append({'type': 'eq', 'fun': te_tangent_constraint})
    result = minimize(
        obj,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not result.success:
        if logger_func:
            logger_func(f"Free-x MSR optimization failed. Using initial guess. Reason: {result.message}")
        final_inner_y = initial_guess_inner_y
    else:
        if logger_func:
            logger_func("Free-x MSR optimization succeeded.")
        final_inner_y = result.x
    control_points = build_ctrl(final_inner_y)
    return control_points

def build_single_bezier_variable_x_minmax(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=config.DEFAULT_REGULARIZATION_WEIGHT,
    error_function="euclidean",
    logger_func=None,
):
    """
    Uncoupled free-x single Bezier optimizer using minmax objective.
    Two-stage: MSR for initial guess, then true minmax optimization.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    The initial guess for the pre-TE y is set using the TE vector.
    """
    from utils.control_point_utils import variable_x_control_points
    # Stage 1: Use build_single_bezier_msr for initial guess (always fixed-x, euclidean)
    control_points = build_single_bezier_msr(
        original_data,
        num_control_points_new,
        is_upper_surface,
        le_tangent_vector,
        te_tangent_vector,
        regularization_weight=0.001,  # No regularization for initial guess
        error_function="euclidean",
        logger_func=logger_func,
    )
    if logger_func:
        logger_func("Running free-x minmax optimization...")
    te_y = float(original_data[-1, 1])
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    fixed_inner_x_coords = paper_x_coords[1:-1]
    initial_guess_inner_y = control_points[1:-1, 1]
    def build_control_points(variables_y):
        ctrl = np.zeros((len(variables_y) + 2, 2))
        ctrl[0] = np.array([0.0, 0.0])
        ctrl[1:-1, 0] = fixed_inner_x_coords
        ctrl[1:-1, 1] = variables_y
        ctrl[-1] = np.array([1.0, te_y])
        return ctrl
    residuals_fn = make_residuals_fn(original_data, error_function, minmax=True)
    control_points = run_minmax_stage(
        initial_y=initial_guess_inner_y,
        build_control_points_fn=build_control_points,
        residuals_fn=residuals_fn,
        regularization_weight=regularization_weight,
        smoothness_fn=smoothness_penalty,
        constraints_fns=None,
        logger_func=logger_func
    )
    if logger_func:
        logger_func("Free-x minmax optimization completed.")
    return control_points