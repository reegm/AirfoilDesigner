import numpy as np
from core import config
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
    minimize_with_debug_with_abort
)

# --- MSR (least-squares) optimizer ---
def build_bezier_single_fixed_x_msr(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=0.00,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Uncoupled fixed-x single Bezier optimizer using mean squared residual (least-squares) objective.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are fixed.
    """
    if logger_func:
        logger_func("Running fixed-x MSR optimization...")
    te_y = float(original_data[-1, 1])
    fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, original_data, te_tangent_vector, te_y)
    initial_guess_inner_y_full = get_initial_guess_inner_y(original_data, fixed_inner_x_coords)
    initial_guess_inner_y = initial_guess_inner_y_full[free_indices]
    if error_function == "orthogonal":
        def error_func(ctrl):
            return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="orthogonal", return_max_error=False)
    else:
        def error_func(ctrl):
            return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="euclidean", return_max_error=False)
    def obj(variables_y):
        ctrl = build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
        errors = error_func(ctrl)
        if isinstance(errors, tuple):
            errors = errors[0]
        return errors + regularization_weight * smoothness_penalty(ctrl)
    
    # Attach debug functions for logging
    if config.DEBUG_WORKER_LOGGING:
        def get_residuals_with_debug(variables_y):
            ctrl = build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
            return error_func(ctrl)
        
        obj.__get_residuals__ = get_residuals_with_debug
        obj.__get_max_error__ = lambda variables_y: np.max(np.abs(error_func(build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values))))
        obj.__build_ctrl__ = lambda variables_y: build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    
    constraints = []
    result, iteration_data = minimize_with_debug_with_abort(
        fun=obj,
        x0=initial_guess_inner_y,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS,
        abort_flag=abort_flag
    )
    if logger_func:
        if result.success:
            logger_func("Fixed-x MSR optimization succeeded.")
        else:
            logger_func(f"Fixed-x MSR optimization failed: {result.message}")
        final_inner_y = result.x
    control_points = build_control_points_with_fixed(final_inner_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    return control_points

def constrained_minmax_fit(
    initial_y,
    build_control_points_fn,
    error_function_fn,
    regularization_weight=0.0,
    smoothness_fn=None,
):
    """
    Legacy minmax optimization function - kept for compatibility.
    """
    def full_objective(y_augmented):
        y = y_augmented[:-1]
        t = y_augmented[-1]
        ctrl = build_control_points_fn(y)
        errors = error_function_fn(ctrl)
        if isinstance(errors, tuple):
            errors = errors[0]
        return t + regularization_weight * (smoothness_fn(ctrl) if smoothness_fn else 0.0)
    
    def constraint_factory(i):
        def constr(y_aug):
            y = y_aug[:-1]
            t = y_aug[-1]
            ctrl = build_control_points_fn(y)
            errors = error_function_fn(ctrl)
            if isinstance(errors, tuple):
                errors = errors[0]
            return t - abs(errors[i])
        return constr

    constraints = [{"type": "ineq", "fun": constraint_factory(i)} for i in range(len(error_function_fn(build_control_points_fn(initial_y))))]
    
    y0_augmented = np.concatenate([initial_y, [np.max(np.abs(error_function_fn(build_control_points_fn(initial_y)))) * 1.05]])
    
    result, iteration_data = minimize_with_debug_with_abort(
        fun=full_objective,
        x0=y0_augmented,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS
    )
    
    if result.success:
        return build_control_points_fn(result.x[:-1])
    else:
        return build_control_points_fn(initial_y)

#--- Minmax optimizer ---
def build_bezier_single_fixed_x_minmax(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Uncoupled fixed-x single Bezier optimizer using minmax objective.
    Two-stage: MSR for initial guess, then true minmax optimization.
    Only the y-coordinates of the inner control points are optimized; x-coordinates are fixed.
    TE tangency constraint is not needed and is removed.
    """
    # Stage 1: Use build_bezier_single_fixed_x_msr for initial guess (always fixed-x, euclidean)
    control_points = build_bezier_single_fixed_x_msr(
        original_data,
        num_control_points_new,
        is_upper_surface,
        le_tangent_vector,
        te_tangent_vector,
        regularization_weight=0,
        error_function="euclidean",
        logger_func=logger_func,
        abort_flag=abort_flag,
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
        logger_func=logger_func,
        abort_flag=abort_flag
    )
    if logger_func:
        logger_func("Fixed-x minmax optimization completed.")
    return control_points

def build_bezier_single_free_x_msr(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=config.DEFAULT_REGULARIZATION_WEIGHT,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
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
    
    # Attach debug functions for logging
    if config.DEBUG_WORKER_LOGGING:
        def get_residuals_with_debug(variables_y):
            ctrl = build_ctrl(variables_y)
            return error_func(ctrl)
        
        obj.__get_residuals__ = get_residuals_with_debug
        obj.__get_max_error__ = lambda variables_y: np.max(np.abs(error_func(build_ctrl(variables_y))))
        obj.__build_ctrl__ = build_ctrl
    
    constraints = []
    tx_te, ty_te = te_tangent_vector
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x_coords[-1]
    py_n_minus_1 = initial_guess_inner_y[-1]
    def te_tangent_constraint(variables_y):
        ctrl = build_ctrl(variables_y)
        py_n_minus_1 = ctrl[-2, 1]
        dx = px_n - px_n_minus_1
        dy = py_n - py_n_minus_1
        return dx * ty_te - dy * tx_te
    constraints.append({"type": "eq", "fun": te_tangent_constraint})
    result, iteration_data = minimize_with_debug_with_abort(
        fun=obj,
        x0=initial_guess_inner_y,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS,
        abort_flag=abort_flag
    )
    if logger_func:
        if result.success:
            logger_func("Free-x MSR optimization succeeded.")
        else:
            logger_func(f"Free-x MSR optimization failed: {result.message}")
        final_inner_y = result.x
    control_points = build_ctrl(final_inner_y)
    return control_points

def build_bezier_single_free_x_minmax(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=config.DEFAULT_REGULARIZATION_WEIGHT,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Uncoupled free-x single Bezier optimizer using minmax objective.
    Now uses true free-x + y optimization via run_xy_minmax.
    """
    from utils.control_point_utils import variable_x_control_points
    from core.solver.solver_helpers import run_xy_minmax

    # Stage 1: MSR for initial guess
    control_points = build_bezier_single_fixed_x_msr(
        original_data,
        num_control_points_new,
        is_upper_surface,
        le_tangent_vector,
        te_tangent_vector,
        regularization_weight=0,
        error_function="euclidean",
        logger_func=logger_func,
    )

    if logger_func:
        logger_func("Running free-x minmax optimization (true XY)...")

    te_y = float(original_data[-1, 1])

    control_points = run_xy_minmax(
        initial_ctrl=control_points,
        original_data=original_data,
        error_fn=error_function + "_minmax",
        te_y=te_y,
        te_tangent_vector=te_tangent_vector,
        bounds_x=None,
        bounds_y=None,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
    )

    if logger_func:
        logger_func("Free-x minmax optimization completed.")

    return control_points

def build_bezier_single_fixed_x_minmax_xy(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Fixed-x single Bezier optimizer using xy_minmax approach with softmax objective.
    This provides the benefits of the softmax optimization while maintaining the speed of fixed-x.
    """
    if logger_func:
        logger_func("Running fixed-x xy_minmax optimization...")
    
    te_y = float(original_data[-1, 1])
    fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, original_data, te_tangent_vector, te_y)
    initial_guess_inner_y_full = get_initial_guess_inner_y(original_data, fixed_inner_x_coords)
    initial_guess_inner_y = initial_guess_inner_y_full[free_indices]
    
    # Build initial control points
    build_ctrl = make_build_ctrl_fn(fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    initial_ctrl = build_ctrl(initial_guess_inner_y)
    
    control_points = run_fixed_x_xy_minmax(
        initial_ctrl=initial_ctrl,
        original_data=original_data,
        error_fn=error_function,
        te_y=te_y,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag
    )
    
    if logger_func:
        logger_func("Fixed-x xy_minmax optimization completed.")
    return control_points

def build_bezier_single_free_x_msr_xy(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Free-x single Bezier optimizer using xy_minmax approach with MSR objective.
    This provides the benefits of xy optimization while using MSR instead of softmax.
    """
    if logger_func:
        logger_func("Running free-x xy MSR optimization...")
    
    te_y = float(original_data[-1, 1])
    from utils.control_point_utils import variable_x_control_points
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    if num_control_points_new != len(paper_x_coords):
        num_control_points_new = len(paper_x_coords)
    
    # Build initial control points
    initial_ctrl = np.zeros((num_control_points_new, 2))
    initial_ctrl[:, 0] = paper_x_coords
    initial_ctrl[:, 1] = np.interp(paper_x_coords, original_data[:, 0], original_data[:, 1])
    
    # Set LE and TE
    initial_ctrl[0] = [0.0, 0.0]  # LE
    initial_ctrl[-1] = [1.0, te_y]  # TE
    
    control_points = run_xy_minmax(
        initial_ctrl=initial_ctrl,
        original_data=original_data,
        error_fn=error_function,
        te_y=te_y,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag
    )
    
    if logger_func:
        logger_func("Free-x xy MSR optimization completed.")
    return control_points

# --- XY Optimization Functions (moved from solver_helpers.py) ---

def run_xy_minmax(
    initial_ctrl,
    original_data,
    error_fn,
    te_y,
    te_tangent_vector,
    bounds_x=None,
    bounds_y=None,
    regularization_weight=0.0,
    logger_func=None,
    abort_flag=None
):
    from core.solver.error_functions import calculate_single_bezier_fitting_error
    
    n = len(initial_ctrl)
    # For LE constraint: second control point x is fixed to 0, so we exclude it from variables
    # Remove TE constraint: let second-to-last point move freely
    x0 = initial_ctrl[2:-1, 0]  # exclude LE, second point, and TE
    y0 = initial_ctrl[1:-1, 1]  # exclude LE and TE, but include second point y
    xy0 = np.concatenate([x0, y0])

    # Initial residuals to set t0
    def build_ctrl(xy):
        n_x_vars = n - 3  # number of free x variables (exclude LE, second point, and TE)
        n_y_vars = n - 2  # number of y variables (exclude LE and TE, but include second point y)
        
        x_inner = xy[:n_x_vars]
        y_inner = xy[n_x_vars:]
        
        ctrl = np.zeros((n, 2))
        ctrl[0] = [0.0, 0.0]  # LE
        ctrl[1] = [0.0, y_inner[0]]  # Second point: x=0, y free
        
        # Remaining inner points (including second-to-last)
        ctrl[2:-1, 0] = x_inner  # Remaining inner x-coordinates
        ctrl[2:-1, 1] = y_inner[1:]  # Remaining inner y-coordinates
        
        ctrl[-1] = [1.0, te_y]  # TE
        return ctrl

    def residuals_fn(ctrl):
        residuals, _, _ = calculate_single_bezier_fitting_error(
            ctrl, original_data, error_function=error_fn, return_max_error=False, return_all=True
        )
        return residuals

    res0 = residuals_fn(initial_ctrl)
    t0 = np.max(np.abs(res0)) * 1.05
    vars0 = xy0

    # Track best configuration during optimization
    best_max_error = float("inf")
    best_xy = None
    
    def full_obj(xy):
        nonlocal best_max_error, best_xy
        
        ctrl = build_ctrl(xy)
        residuals = residuals_fn(ctrl)
        
        # Calculate true max error for tracking
        true_max_error = np.max(np.abs(residuals))
        
        # Update best configuration if we found a better max error
        if true_max_error < best_max_error:
            best_max_error = true_max_error
            best_xy = np.copy(xy)
        
        # Optimize softmax for numerical stability
        alpha = config.SOFTMAX_ALPHA
        softmax_error = np.log(np.sum(np.exp(alpha * np.abs(residuals)))) / alpha
        
        # Add smoothness penalty
        smoothness_penalty_val = smoothness_penalty(ctrl) if regularization_weight > 0 else 0.0
        
        # Calculate final objective components for debugging
        # When smoothness is 0, minimize position penalty to focus on pure fitting
        # When smoothness > 0, use moderate position penalty to prevent extreme movements
        position_weight = 100 if regularization_weight == 0.0 else 1
        position_term = position_weight #* position_penalty
        smoothness_term = regularization_weight * smoothness_penalty_val
        total_obj = softmax_error + position_term + smoothness_term
        
        return total_obj
    
    # Attach debug functions for logging
    if config.DEBUG_WORKER_LOGGING:
        def get_residuals_with_debug(xy):
            return residuals_fn(build_ctrl(xy))
        
        full_obj.__get_residuals__ = get_residuals_with_debug
        full_obj.__get_max_error__ = lambda xy: np.max(np.abs(residuals_fn(build_ctrl(xy))))
        full_obj.__build_ctrl__ = build_ctrl
        full_obj.__te_y__ = te_y
        full_obj.__original_data__ = original_data

    constraints = []

    if te_tangent_vector is not None and te_y is not None:
        def make_te_vector_constraint(te_tangent_vector, te_y, n_x_vars, n_y_vars):
            tx_te, ty_te = te_tangent_vector
            norm = np.sqrt(tx_te**2 + ty_te**2) or 1e-12

            def constraint(xy):
                x_nm1 = xy[n_x_vars - 1]
                y_nm1 = xy[n_x_vars + n_y_vars - 1]
                dx = 1.0 - x_nm1
                dy = te_y - y_nm1
                return (dx * ty_te - dy * tx_te) / norm

            return constraint

        constraints.append({
            "type": "eq",
            "fun": make_te_vector_constraint(te_tangent_vector, te_y, len(initial_ctrl) -3, len(initial_ctrl) -2)
        })

    if bounds_x is None:
        bounds_x = [(0.0, 1.0)] * (n - 3)  # exclude LE, second point, and TE
    if bounds_y is None:
        bounds_y = [(-1.0, 1.0)] * (n - 2)  # exclude LE and TE, but include second point y
    bounds = bounds_x + bounds_y
    success_threshold = config.MAX_ERROR_THRESHOLD
    result, trace = minimize_with_debug_with_abort(
        fun=full_obj,
        x0=vars0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=config.SLSQP_OPTIONS,
        success_threshold=success_threshold,
        abort_flag=abort_flag
    )

    if logger_func:
        if result.success:
            logger_func("XY minmax optimization succeeded.")
        else:
            logger_func(f"XY minmax optimization failed: {result.message}")

    # Use the best configuration found during optimization (if available)
    if best_xy is not None:
        final_ctrl = build_ctrl(best_xy)
        if logger_func:
            logger_func(f"Using best configuration found (max error: {best_max_error:.6e})")
    else:
        final_ctrl = build_ctrl(result.x) if result.success else initial_ctrl
    
    return final_ctrl

def run_fixed_x_xy_minmax(
    initial_ctrl,
    original_data,
    error_fn,
    te_y,
    te_tangent_vector,
    regularization_weight=0.0,
    logger_func=None,
    abort_flag=None
):
    """
    Fixed-x version of xy_minmax that uses the softmax approach but with fixed x-coordinates.
    This provides the benefits of the softmax optimization while maintaining the speed of fixed-x.
    """
    from core.solver.error_functions import calculate_single_bezier_fitting_error
    
    n = len(initial_ctrl)
    # For fixed-x, we only optimize y-coordinates (except LE and TE which are fixed)
    y0 = initial_ctrl[1:-1, 1]  # exclude LE and TE, but include all inner y-coordinates

    def build_ctrl(y):
        ctrl = np.zeros((n, 2))
        ctrl[0] = [0.0, 0.0]  # LE fixed
        
        # Inner points: x-coordinates are fixed, y-coordinates are optimized
        ctrl[1:-1, 0] = initial_ctrl[1:-1, 0]  # Fixed x-coordinates
        ctrl[1:-1, 1] = y  # Optimized y-coordinates
        
        ctrl[-1] = [1.0, te_y]  # TE fixed
        return ctrl

    def residuals_fn(ctrl):
        residuals, _, _ = calculate_single_bezier_fitting_error(
            ctrl, original_data, error_function=error_fn, return_max_error=False, return_all=True
        )
        return residuals

    res0 = residuals_fn(initial_ctrl)
    t0 = np.max(np.abs(res0)) * 1.05

    # Track best configuration during optimization
    best_max_error = float("inf")
    best_y = None
    
    def full_obj(y):
        nonlocal best_max_error, best_y
        
        ctrl = build_ctrl(y)
        residuals = residuals_fn(ctrl)
        
        # Calculate true max error for tracking
        true_max_error = np.max(np.abs(residuals))
        
        # Update best configuration if we found a better max error
        if true_max_error < best_max_error:
            best_max_error = true_max_error
            best_y = np.copy(y)
        
        # Optimize softmax for numerical stability
        alpha = config.SOFTMAX_ALPHA
        softmax_error = np.log(np.sum(np.exp(alpha * np.abs(residuals)))) / alpha
        
        # Add smoothness penalty
        smoothness_penalty_val = smoothness_penalty(ctrl) if regularization_weight > 0 else 0.0
        
        # Calculate final objective
        position_weight = 100 if regularization_weight == 0.0 else 1
        position_term = position_weight
        smoothness_term = regularization_weight * smoothness_penalty_val
        total_obj = softmax_error + position_term + smoothness_term
        
        return total_obj
    
    # Attach debug functions for logging
    if config.DEBUG_WORKER_LOGGING:
        def get_residuals_with_debug(y):
            return residuals_fn(build_ctrl(y))
        
        full_obj.__get_residuals__ = get_residuals_with_debug
        full_obj.__get_max_error__ = lambda y: np.max(np.abs(residuals_fn(build_ctrl(y))))
        full_obj.__build_ctrl__ = build_ctrl

    constraints = []

    result, iteration_data = minimize_with_debug_with_abort(
        fun=full_obj,
        x0=y0,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS,
        success_threshold=config.MAX_ERROR_THRESHOLD,
        abort_flag=abort_flag
    )

    if logger_func:
        if result.success:
            logger_func("Fixed-x softmax optimization succeeded.")
        else:
            logger_func(f"Fixed-x softmax optimization failed: {result.message}")

    # Use the best configuration found during optimization (if available)
    if best_y is not None:
        final_ctrl = build_ctrl(best_y)
        if logger_func:
            logger_func(f"Using best configuration found (max error: {best_max_error:.6e})")
    else:
        final_ctrl = build_ctrl(result.x) if result.success else build_ctrl(y0)
    
    return final_ctrl