import numpy as np
from scipy.optimize import minimize
from core import config
from core.solver.error_functions import calculate_euclidean_error, calculate_orthogonal_error
from core.solver.solver_helpers import get_fixed_inner_x_coords, get_initial_guess_inner_y, build_control_points, smoothness_penalty, te_tangent_constraint_factory
from core.solver.error_functions import calculate_single_bezier_fitting_error

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

# --- Minmax (Chebyshev) objective ---
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
    """
    te_y = float(original_data[-1, 1])
    fixed_inner_x_coords = get_fixed_inner_x_coords(is_upper_surface, num_control_points_new)
    initial_guess_inner_y = get_initial_guess_inner_y(original_data, fixed_inner_x_coords)

    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error

    def obj(variables_y):
        return msr_objective(
            variables_y,
            lambda y: build_control_points(y, fixed_inner_x_coords, te_y),
            original_data,
            error_func,
            regularization_weight
        )

    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x_coords[-1]
    if not np.isclose(tx_te, 0.0):
        constraints.append({'type': 'eq', 'fun': te_tangent_constraint_factory(tx_te, ty_te, px_n, py_n, px_n_minus_1, -1)})

    result = minimize(
        obj,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not result.success:
        if logger_func:
            logger_func(f"MSR optimization failed. Using initial guess. Reason: {result.message}")
        final_inner_y = initial_guess_inner_y
    else:
        final_inner_y = result.x
    control_points = build_control_points(final_inner_y, fixed_inner_x_coords, te_y)
    return control_points

# --- Minmax (Chebyshev) optimizer ---
def build_single_bezier_minmax(
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
    Fixed-x single Bezier optimizer using Chebyshev (minmax) objective.
    Two-stage: MSR for initial guess, then true minmax optimization.
    """
    te_y = float(original_data[-1, 1])
    fixed_inner_x_coords = get_fixed_inner_x_coords(is_upper_surface, num_control_points_new)
    initial_guess_inner_y = get_initial_guess_inner_y(original_data, fixed_inner_x_coords)

    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error

    def msr_obj(variables_y):
        return msr_objective(
            variables_y,
            lambda y: build_control_points(y, fixed_inner_x_coords, te_y),
            original_data,
            error_func,
            regularization_weight
        )

    def minmax_obj(variables_y):
        control_points = build_control_points(variables_y, fixed_inner_x_coords, te_y)
        # Use the new error function for orthogonal error
        _, max_error, max_error_idx = calculate_single_bezier_fitting_error(
            control_points, original_data, error_function=error_function, return_max_error=True
        )
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        return max_error + regularization_weight * smoothness_penalty

    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x_coords[-1]
    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    constraint_added = False
    if not np.isclose(tx_te, 0.0):
        constraints.append({'type': 'eq', 'fun': te_tangent_constraint})
        constraint_added = True

    # Stage 1: MSR for initial guess
    if logger_func:
        logger_func("Stage 1: Running MSR optimization for initial guess...")
    msr_result = minimize(
        msr_obj,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not msr_result.success:
        if logger_func:
            logger_func(f"MSR optimization failed. Using initial guess. Reason: {msr_result.message}")
        msr_inner_y = initial_guess_inner_y
    else:
        msr_inner_y = msr_result.x
    # Stage 2: minmax (Chebyshev)
    if logger_func:
        logger_func("Stage 2: Running minmax (Chebyshev) optimization...")
    minmax_result = minimize(
        minmax_obj,
        msr_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not minmax_result.success:
        if logger_func:
            logger_func(f"Minmax optimization failed. Using msr solution. Reason: {minmax_result.message}")
        final_inner_y = msr_inner_y
    else:
        final_inner_y = minmax_result.x
    control_points = build_control_points(final_inner_y, fixed_inner_x_coords, te_y)
    return control_points

# --- Deprecated: unified function, no longer used ---
def build_single_bezier_model_optimized(*args, **kwargs):
    raise NotImplementedError("Use build_single_bezier_msr or build_single_bezier_minmax instead.") 

def build_single_bezier_variable_x_msr(
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
    Uncoupled variable-x single Bezier optimizer using mean squared residual (least-squares) objective.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    """
    from utils.control_point_utils import variable_x_control_points
    te_y = float(original_data[-1, 1])
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    if num_control_points_new != len(paper_x_coords):
        num_control_points_new = len(paper_x_coords)
    fixed_inner_x_coords = paper_x_coords[1:-1]
    # Initial guess for y: interpolate at variable x
    initial_guess_inner_y = np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])
    # Error function
    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error
    def build_control_points(variables_y):
        control_points = np.zeros((len(variables_y) + 2, 2))
        control_points[0] = np.array([0.0, 0.0])
        control_points[1:-1, 0] = fixed_inner_x_coords
        control_points[1:-1, 1] = variables_y
        control_points[-1] = np.array([1.0, te_y])
        return control_points
    def obj(variables_y):
        control_points = build_control_points(variables_y)
        errors = error_func(original_data, control_points, return_max_error=False)
        if isinstance(errors, tuple):
            errors = errors[0]
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        return errors + regularization_weight * smoothness_penalty
    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x_coords[-1]
    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    if not np.isclose(tx_te, 0.0):
        constraints.append({'type': 'eq', 'fun': te_tangent_constraint})
    if logger_func:
        logger_func("Stage 1: Running variable-x MSR optimization for initial guess...")
    result = minimize(
        obj,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not result.success:
        if logger_func:
            logger_func(f"Variable-x MSR optimization failed. Using initial guess. Reason: {result.message}")
        final_inner_y = initial_guess_inner_y
    else:
        final_inner_y = result.x
    control_points = build_control_points(final_inner_y)
    return control_points 

def build_single_bezier_variable_x_minmax(
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
    Uncoupled variable-x single Bezier optimizer using Chebyshev (minmax) objective.
    Two-stage: MSR for initial guess, then true minmax optimization.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are set by variable_x_control_points.
    """
    from utils.control_point_utils import variable_x_control_points
    te_y = float(original_data[-1, 1])
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    if num_control_points_new != len(paper_x_coords):
        num_control_points_new = len(paper_x_coords)
    fixed_inner_x_coords = paper_x_coords[1:-1]
    initial_guess_inner_y = np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])
    # Error function selection
    if error_function == "orthogonal":
        error_func = calculate_orthogonal_error
    else:
        error_func = calculate_euclidean_error
    def build_control_points(variables_y):
        control_points = np.zeros((len(variables_y) + 2, 2))
        control_points[0] = np.array([0.0, 0.0])
        control_points[1:-1, 0] = fixed_inner_x_coords
        control_points[1:-1, 1] = variables_y
        control_points[-1] = np.array([1.0, te_y])
        return control_points
    def msr_obj(variables_y):
        control_points = build_control_points(variables_y)
        errors = error_func(original_data, control_points, return_max_error=False)
        if isinstance(errors, tuple):
            errors = errors[0]
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        return errors + regularization_weight * smoothness_penalty
    def minmax_obj(variables_y):
        control_points = build_control_points(variables_y)
        _, max_error, max_error_idx = calculate_single_bezier_fitting_error(
            control_points, original_data, error_function=error_function, return_max_error=True
        )
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        return max_error + regularization_weight * smoothness_penalty
    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x_coords[-1]
    def te_tangent_constraint(variables_y):
        y_n_minus_1 = variables_y[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    if not np.isclose(tx_te, 0.0):
        constraints.append({'type': 'eq', 'fun': te_tangent_constraint})
    # Stage 1: MSR for initial guess
    if logger_func:
        logger_func("Stage 1: Running variable-x MSR optimization for initial guess...")
    msr_result = minimize(
        msr_obj,
        initial_guess_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not msr_result.success:
        if logger_func:
            logger_func(f"Variable-x MSR optimization failed. Using initial guess. Reason: {msr_result.message}")
        msr_inner_y = initial_guess_inner_y
    else:
        msr_inner_y = msr_result.x
    # Stage 2: minmax
    if logger_func:
        logger_func("Stage 2: Running variable-x minmax optimization...")
    minmax_result = minimize(
        minmax_obj,
        msr_inner_y,
        method='SLSQP',
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    if not minmax_result.success:
        if logger_func:
            logger_func(f"Variable-x minmax optimization failed. Using MSR solution. Reason: {minmax_result.message}")
        final_inner_y = msr_inner_y
    else:
        final_inner_y = minmax_result.x
    control_points = build_control_points(final_inner_y)
    return control_points 