import numpy as np
from core import config
from core.error_functions import calculate_single_bezier_fitting_error
from utils.bezier_utils import leading_edge_curvature
from core.solver_helpers import (
    get_fixed_inner_x_partition,
    build_control_points_with_fixed,
    smoothness_penalty,
    extract_free_y_from_ctrl,
    make_build_ctrl_fn,
    make_residuals_fn,
    run_softmax_stage,
    get_initial_guess_inner_y,
    minimize_with_debug_with_abort
)
from core.bezier_unified_optimizer import optimize_bezier


def build_bezier_staged_uncoupled(
    original_data,
    num_control_points_new,
    is_upper_surface,
    le_tangent_vector,
    te_tangent_vector,
    regularization_weight=config.DEFAULT_REGULARIZATION_WEIGHT,
    *,
    error_function: str = "euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Uncoupled staged optimizer pipeline (euclidean error only):
    1) Basin-hopping style fixed-x MSR with bounded SLSQP restarts
    2) Switch to free-x softmax (softmax objective)

    Notes:
    - We deliberately ignore any orthogonal error variants for now and use euclidean only.
    - Regularization weight is applied in Stage 2 softmax consistent with existing design.
    - Uses plateau detection from minimize_with_debug_with_abort via progress to determine stalling implicitly by
      limiting per-stage max iterations and checking no best_true_max improvement across hops.
    """
    import numpy as np

    # Stage 1: Fixed-x MSR with basin-hopping like restarts to escape poor initializations
    # Build an initial fixed-x MSR solution as a baseline
    base_ctrl = optimize_bezier(
        initial_ctrl=None,
        original_data=original_data,
        mode="fixed-x",
        coupled=False,
        error_function="euclidean",
        objective="msr",
        te_y=float(original_data[-1, 1]),
        te_tangent_vector=te_tangent_vector,
        regularization_weight=0.0,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new,
    )

    if abort_flag is not None and abort_flag.value:
        return base_ctrl

    # Prepare inner-y vector for perturbations
    from core.solver_helpers import get_fixed_inner_x_partition, build_control_points_with_fixed, get_initial_guess_inner_y
    fixed_inner_x, free_idx, fixed_idx, fixed_y_vals = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, original_data, te_tangent_vector, float(original_data[-1, 1])
    )
    # Reconstruct inner y from base_ctrl for perturbation center
    def extract_inner_y_from_ctrl(ctrl):
        # ctrl has size n, with fixed inner x per paper. We need free inner y variables order matching free_idx
        full_y = np.interp(fixed_inner_x, ctrl[:, 0], ctrl[:, 1])
        return full_y[free_idx]

    center_y = extract_inner_y_from_ctrl(base_ctrl)

    rng = np.random.default_rng(12345)
    best_ctrl = base_ctrl
    # Evaluate best using euclidean max error for consistency
    from core.error_functions import calculate_single_bezier_fitting_error
    _, best_max_err, _ = calculate_single_bezier_fitting_error(best_ctrl, original_data, error_function="euclidean", return_max_error=True)

    # Local MSR optimization options with tighter iteration budget
    local_opts = dict(config.SLSQP_OPTIONS)
    local_opts["maxiter"] = config.HYBRID_LOCAL_MAXITER_MSR

    # Local objective on fixed-x MSR for inner y variables
    def msr_obj(inner_y):
        ctrl = build_control_points_with_fixed(inner_y, fixed_inner_x, float(original_data[-1, 1]), free_idx, fixed_idx, fixed_y_vals)
        # Return scalar sum of squares
        res = calculate_single_bezier_fitting_error(ctrl, original_data, error_function="euclidean", return_max_error=False)
        if isinstance(res, tuple):
            res = res[0]
        # Attach hooks for progress reporting
        msr_obj.__build_ctrl__ = lambda y: build_control_points_with_fixed(y, fixed_inner_x, float(original_data[-1, 1]), free_idx, fixed_idx, fixed_y_vals)
        return res

    # Run basin-hopping style restarts
    # Stage-specific hop counts
    hops_msr = max(0, int(config.HYBRID_BH_HOPS_MSR))
    hops_free = max(0, int(config.HYBRID_BH_HOPS_FREE_MINMAX))
    perturb_std = float(config.HYBRID_BH_PERTURB_STD)
    current_best_y = center_y.copy()
    # Stage 1 logging
    if logger_func:
        logger_func(f"Stage 1 (fixed-x MSR) hops={hops_msr}")
    for hop in range(hops_msr):
        if logger_func:
            logger_func(f"Stage 1 hop {hop+1}/{hops_msr}")
        if abort_flag is not None and abort_flag.value:
            break
        trial_y = current_best_y + rng.normal(0.0, perturb_std, size=current_best_y.shape)
        # Clamp y to a reasonable range
        trial_y = np.clip(trial_y, -1.0, 1.0)
        # Optimize this trial locally
        result, _ = minimize_with_debug_with_abort(
            fun=msr_obj,
            x0=trial_y,
            method="SLSQP",
            options=local_opts,
            abort_flag=abort_flag,
            progress_callback=logger_func,
        )
        candidate_y = result.x if result is not None else trial_y
        candidate_ctrl = build_control_points_with_fixed(candidate_y, fixed_inner_x, float(original_data[-1, 1]), free_idx, fixed_idx, fixed_y_vals)
        _, candidate_max_err, _ = calculate_single_bezier_fitting_error(candidate_ctrl, original_data, error_function="euclidean", return_max_error=True)
        if candidate_max_err < best_max_err:
            best_max_err = candidate_max_err
            best_ctrl = candidate_ctrl
            current_best_y = candidate_y
            if logger_func:
                logger_func(f"Stage 1 hop {hop+1}/{hops_msr} improved best max error to {best_max_err:.6e}")

    # Stage 2: Free-x softmax (softmax objective) with basin-hopping restarts if not aborted
    if abort_flag is not None and abort_flag.value:
        return best_ctrl

    if logger_func:
        logger_func("Stage 2 (free-x softmax) starting")
    free_opts = dict(config.SLSQP_OPTIONS)
    free_opts["maxiter"] = config.HYBRID_LOCAL_MAXITER_MINMAX_FREE
    final_ctrl = optimize_bezier(
        initial_ctrl=best_ctrl,
        original_data=original_data,
        mode="free-x",
        coupled=False,
        error_function="euclidean",
        objective="softmax",
        te_y=float(original_data[-1, 1]),
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new,
    )
    _, free_softmax_max, _ = calculate_single_bezier_fitting_error(final_ctrl, original_data, error_function="euclidean", return_max_error=True)
    if free_softmax_max < best_max_err:
        best_ctrl = final_ctrl
        best_max_err = free_softmax_max

    # Basin-hopping restarts for free-x: perturb inner x and y slightly
    n = len(best_ctrl)
    x_inner0 = best_ctrl[2:-1, 0]
    y_inner0 = best_ctrl[1:-1, 1]
    if logger_func:
        logger_func(f"Stage 2 (free-x softmax) hops={hops_free}")
    for hop in range(hops_free):
        if logger_func:
            logger_func(f"Stage 2 hop {hop+1}/{hops_free}")
        if abort_flag is not None and abort_flag.value:
            break
        x_trial = np.clip(x_inner0 + rng.normal(0.0, perturb_std, size=x_inner0.shape), 0.0, 1.0)
        # Keep x monotone increasing
        x_trial = np.maximum.accumulate(x_trial)
        y_trial = np.clip(y_inner0 + rng.normal(0.0, perturb_std, size=y_inner0.shape), -1.0, 1.0)
        trial_ctrl = np.zeros_like(best_ctrl)
        trial_ctrl[0] = [0.0, 0.0]
        trial_ctrl[1] = [0.0, y_trial[0]]
        trial_ctrl[2:-1, 0] = x_trial
        trial_ctrl[2:-1, 1] = y_trial[1:]
        trial_ctrl[-1] = [1.0, float(original_data[-1, 1])]
        trial_local = optimize_bezier(
            initial_ctrl=trial_ctrl,
            original_data=original_data,
            mode="free-x",
            coupled=False,
            error_function="euclidean",
            objective="softmax",
            te_y=float(original_data[-1, 1]),
            te_tangent_vector=te_tangent_vector,
            regularization_weight=regularization_weight,
            logger_func=logger_func,
            abort_flag=abort_flag,
            is_upper_surface=is_upper_surface,
            num_control_points_new=num_control_points_new,
        )
        _, trial_max, _ = calculate_single_bezier_fitting_error(trial_local, original_data, error_function="euclidean", return_max_error=True)
        if trial_max < best_max_err:
            best_ctrl = trial_local
            best_max_err = trial_max
            x_inner0 = trial_local[2:-1, 0]
            y_inner0 = trial_local[1:-1, 1]
            if logger_func:
                logger_func(f"Stage 2 hop {hop+1}/{hops_free} improved best max error to {best_max_err:.6e}")

    return best_ctrl

# --- MSR (least-squares) optimizer ---

def build_bezier_fixed_x_msr(
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
    Uncoupled fixed-x single Bezier optimizer using mean squared residual (least-squares) objective.
    Uses the new unified optimizer logic.
    """
    from utils.control_point_utils import variable_x_control_points
    te_y = float(original_data[-1, 1])
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    if num_control_points_new != len(paper_x_coords):
        num_control_points_new = len(paper_x_coords)
    initial_ctrl = np.zeros((num_control_points_new, 2))
    initial_ctrl[:, 0] = paper_x_coords
    initial_ctrl[:, 1] = np.interp(paper_x_coords, original_data[:, 0], original_data[:, 1])
    
    # Set LE and TE
    initial_ctrl[0] = [0.0, 0.0]  # LE
    initial_ctrl[-1] = [1.0, te_y]  # TE

    control_points = optimize_bezier(
        initial_ctrl=initial_ctrl,
        original_data=original_data,
        mode="fixed-x",
        coupled=False,
        error_function=error_function,
        objective="msr",
        te_y=te_y,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new
    )
    return control_points





def build_bezier_free_x_msr(
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
    Uses the new unified optimizer logic.
    """
    from utils.control_point_utils import variable_x_control_points
    te_y = float(original_data[-1, 1])
    paper_x_coords = variable_x_control_points(original_data, num_control_points_new)
    if num_control_points_new != len(paper_x_coords):
        num_control_points_new = len(paper_x_coords)
    initial_ctrl = np.zeros((num_control_points_new, 2))
    initial_ctrl[:, 0] = paper_x_coords
    initial_ctrl[:, 1] = np.interp(paper_x_coords, original_data[:, 0], original_data[:, 1])
    
    # Set LE and TE
    initial_ctrl[0] = [0.0, 0.0]  # LE
    initial_ctrl[-1] = [1.0, te_y]  # TE

    control_points = optimize_bezier(
        initial_ctrl=initial_ctrl,
        original_data=original_data,
        mode="free-x",
        coupled=False,
        error_function=error_function,
        objective="msr",
        te_y=te_y,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new
    )
    return control_points



def build_bezier_free_x_softmax(
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
    Uncoupled free-x single Bezier optimizer using softmax objective with softmax.
    Uses the new unified optimizer logic with MSR initial guess stage.
    """
    # Stage 1: MSR for initial guess using unified optimizer
    control_points = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_data,
        mode="fixed-x",
        coupled=False,
        error_function="euclidean",
        objective="msr",
        te_y=float(original_data[-1, 1]),
        te_tangent_vector=te_tangent_vector,
        regularization_weight=0,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new
    )

    if logger_func:
        logger_func("Running free-x softmax optimization...")

    # Stage 2: Softmax optimization using unified optimizer
    control_points = optimize_bezier(
        initial_ctrl=control_points,
        original_data=original_data,
        mode="free-x",
        coupled=False,
        error_function=error_function,
        objective="softmax",
        te_y=float(original_data[-1, 1]),
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new
    )
    
    if logger_func:
        logger_func("free-x softmax optimization completed.")
    
    return control_points

def build_bezier_fixed_x_softmax(
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
    Uncoupled fixed-x single Bezier optimizer using softmax objective with softmax.
    Uses the unified optimizer directly (no preliminary MSR stage needed).
    """
    if logger_func:
        logger_func("Running fixed-x softmax optimization...")

    control_points = optimize_bezier(
        initial_ctrl=None,  # Build internally
        original_data=original_data,
        mode="fixed-x",
        coupled=False,
        error_function=error_function,
        objective="softmax",
        te_y=float(original_data[-1, 1]),
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new
    )
    
    if logger_func:
        logger_func("fixed-x softmax optimization completed.")
    
    return control_points





















def build_bezier_unified_peak_curvature_free_x_msr(
    upper_data,
    lower_data,
    num_control_points_new,
    te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
    search_region=0.1,
):
    """
    Unified peak curvature-based free-x MSR optimizer for both surfaces.
    Uses a single split point determined by the surface with tighter curvature.
    Reorganizes data so that points with x < split_x from the tighter surface
    are moved to the other surface, creating non-monotonic x sequences.
    Allows control points to move freely in both x and y directions.
    """
    from utils.bezier_utils import find_optimal_split_point_for_both_surfaces, reorganize_airfoil_data_for_split_point
    from utils.control_point_utils import variable_x_control_points
    from core.solver_helpers import minimize_with_debug_with_abort, smoothness_penalty
    from core.error_functions import calculate_single_bezier_fitting_error
    from core import config
    
    if logger_func:
        logger_func("Finding optimal split point for both surfaces...")
    
    # Find the optimal split point for both surfaces
    split_x, split_y, upper_tangent, lower_tangent, tighter_surface_is_upper = find_optimal_split_point_for_both_surfaces(
        upper_data, lower_data, search_region
    )
    
    if logger_func:
        surface_name = "upper" if tighter_surface_is_upper else "lower"
        logger_func(f"Using split point from {surface_name} surface: ({split_x:.4f}, {split_y:.4f})")
        logger_func(f"Upper tangent: {upper_tangent}")
        logger_func(f"Lower tangent: {lower_tangent}")
    
    # Reorganize the airfoil data based on the split point
    if logger_func:
        logger_func("Reorganizing airfoil data for unified split point...")
    
    new_upper_data, new_lower_data = reorganize_airfoil_data_for_split_point(
        upper_data, lower_data, split_x, tighter_surface_is_upper
    )
    
    if logger_func:
        logger_func(f"Data reorganization complete. Upper: {len(new_upper_data)} points, Lower: {len(new_lower_data)} points")
        logger_func("Starting unified peak curvature optimization...")
    
    # Get initial control points for both surfaces (use per-surface TE y)
    te_y_upper = float(new_upper_data[-1, 1])
    te_y_lower = float(new_lower_data[-1, 1])
    
    # Upper surface
    upper_paper_x_coords = variable_x_control_points(new_upper_data, num_control_points_new)
    if num_control_points_new != len(upper_paper_x_coords):
        num_control_points_new = len(upper_paper_x_coords)
    upper_initial_ctrl = np.zeros((num_control_points_new, 2))
    upper_initial_ctrl[:, 0] = upper_paper_x_coords
    upper_initial_ctrl[:, 1] = np.interp(upper_paper_x_coords, new_upper_data[:, 0], new_upper_data[:, 1])
    
    # Set split point as first control point for upper surface
    upper_initial_ctrl[0] = [split_x, split_y]
    upper_initial_ctrl[-1] = [1.0, te_y_upper]  # TE
    
    # Lower surface
    lower_paper_x_coords = variable_x_control_points(new_lower_data, num_control_points_new)
    lower_initial_ctrl = np.zeros((num_control_points_new, 2))
    lower_initial_ctrl[:, 0] = lower_paper_x_coords
    lower_initial_ctrl[:, 1] = np.interp(lower_paper_x_coords, new_lower_data[:, 0], new_lower_data[:, 1])
    
    # Set split point as first control point for lower surface
    lower_initial_ctrl[0] = [split_x, split_y]
    lower_initial_ctrl[-1] = [1.0, te_y_lower]  # TE
    
    # Define objective function for upper surface
    def upper_objective_function(variables):
        # variables: [x1, y1, x2, y2, ..., xn-1, yn-1] (excluding split point and TE)
        n_vars = len(variables)
        n_inner = n_vars // 2
        
        # Reconstruct control points
        control_points = np.zeros((num_control_points_new, 2))
        control_points[0] = [split_x, split_y]  # Split point
        
        # Set inner control points
        for i in range(n_inner):
            control_points[i + 1] = [variables[2*i], variables[2*i + 1]]
        
        control_points[-1] = [1.0, te_y_upper]  # TE
        
        # Apply tangent constraint to second control point (P1)
        if n_inner > 0:
            # Calculate the vector from split point to P1
            p0_to_p1 = control_points[1] - control_points[0]
            
            # Project P1 onto the line defined by split point and upper tangent
            # P1 should lie on: P0 + t * upper_tangent
            tx, ty = upper_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:  # Avoid division by zero
                # Calculate the parameter t that minimizes distance to current P1
                # t = (P1 - P0) · tangent / |tangent|²
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, upper_tangent) / tangent_norm_sq
                
                # Project P1 onto the tangent line
                projected_p1 = np.array([split_x, split_y]) + t * np.array(upper_tangent)
                control_points[1] = projected_p1
        
        # Calculate error
        error = calculate_single_bezier_fitting_error(control_points, new_upper_data, error_function=error_function, return_max_error=False)
        if isinstance(error, tuple):
            error = error[0]
        return error
    
    # Define objective function for lower surface
    def lower_objective_function(variables):
        # variables: [x1, y1, x2, y2, ..., xn-1, yn-1] (excluding split point and TE)
        n_vars = len(variables)
        n_inner = n_vars // 2
        
        # Reconstruct control points
        control_points = np.zeros((num_control_points_new, 2))
        control_points[0] = [split_x, split_y]  # Split point
        
        # Set inner control points
        for i in range(n_inner):
            control_points[i + 1] = [variables[2*i], variables[2*i + 1]]
        
        control_points[-1] = [1.0, te_y_upper]  # TE
        
        # Apply tangent constraint to second control point (P1)
        if n_inner > 0:
            # Calculate the vector from split point to P1
            p0_to_p1 = control_points[1] - control_points[0]
            
            # Project P1 onto the line defined by split point and lower tangent
            # P1 should lie on: P0 + t * lower_tangent
            tx, ty = lower_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:  # Avoid division by zero
                # Calculate the parameter t that minimizes distance to current P1
                # t = (P1 - P0) · tangent / |tangent|²
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, lower_tangent) / tangent_norm_sq
                
                # Project P1 onto the tangent line
                projected_p1 = np.array([split_x, split_y]) + t * np.array(lower_tangent)
                control_points[1] = projected_p1
        
        # Calculate error
        error = calculate_single_bezier_fitting_error(control_points, new_lower_data, error_function=error_function, return_max_error=False)
        if isinstance(error, tuple):
            error = error[0]
        return error
    
    # Prepare initial variables for optimization
    # Extract inner control points (excluding split point and TE)
    upper_inner_vars = []
    for i in range(1, num_control_points_new - 1):
        upper_inner_vars.extend([upper_initial_ctrl[i, 0], upper_initial_ctrl[i, 1]])
    
    lower_inner_vars = []
    for i in range(1, num_control_points_new - 1):
        lower_inner_vars.extend([lower_initial_ctrl[i, 0], lower_initial_ctrl[i, 1]])
    
    # Add build_ctrl functions for progress updates
    def upper_build_ctrl(variables):
        n_vars = len(variables)
        n_inner = n_vars // 2
        control_points = np.zeros((num_control_points_new, 2))
        control_points[0] = [split_x, split_y]
        for i in range(n_inner):
            control_points[i + 1] = [variables[2*i], variables[2*i + 1]]
        control_points[-1] = [1.0, te_y_upper]
        
        # Apply tangent constraint
        if n_inner > 0:
            p0_to_p1 = control_points[1] - control_points[0]
            tx, ty = upper_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, upper_tangent) / tangent_norm_sq
                projected_p1 = np.array([split_x, split_y]) + t * np.array(upper_tangent)
                control_points[1] = projected_p1
        return control_points
    
    def lower_build_ctrl(variables):
        n_vars = len(variables)
        n_inner = n_vars // 2
        control_points = np.zeros((num_control_points_new, 2))
        control_points[0] = [split_x, split_y]
        for i in range(n_inner):
            control_points[i + 1] = [variables[2*i], variables[2*i + 1]]
        control_points[-1] = [1.0, te_y_lower]
        
        # Apply tangent constraint
        if n_inner > 0:
            p0_to_p1 = control_points[1] - control_points[0]
            tx, ty = lower_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, lower_tangent) / tangent_norm_sq
                projected_p1 = np.array([split_x, split_y]) + t * np.array(lower_tangent)
                control_points[1] = projected_p1
        return control_points
    
    upper_objective_function.__build_ctrl__ = upper_build_ctrl
    lower_objective_function.__build_ctrl__ = lower_build_ctrl
    
    # Run optimization for upper surface
    if logger_func:
        logger_func("Optimizing upper surface...")
    
    upper_result, _ = minimize_with_debug_with_abort(
        fun=upper_objective_function,
        x0=upper_inner_vars,
        method="SLSQP",
        options=config.SLSQP_OPTIONS,
        progress_callback=logger_func,
        abort_flag=abort_flag
    )
    
    if not upper_result.success:
        if logger_func:
            logger_func(f"Warning: Upper surface optimization did not converge: {upper_result.message}")
    
    # Run optimization for lower surface
    if logger_func:
        logger_func("Optimizing lower surface...")
    
    lower_result, _ = minimize_with_debug_with_abort(
        fun=lower_objective_function,
        x0=lower_inner_vars,
        method="SLSQP",
        options=config.SLSQP_OPTIONS,
        progress_callback=logger_func,
        abort_flag=abort_flag
    )
    
    if not lower_result.success:
        if logger_func:
            logger_func(f"Warning: Lower surface optimization did not converge: {lower_result.message}")
    
    # Build final control points
    upper_final_ctrl = upper_build_ctrl(upper_result.x)
    lower_final_ctrl = lower_build_ctrl(lower_result.x)
    
    if logger_func:
        logger_func("Unified peak curvature optimization completed.")
    
    # Return both control points and reorganized data for proper error calculation
    return (upper_final_ctrl, lower_final_ctrl), (new_upper_data, new_lower_data)


def build_bezier_unified_peak_curvature_fixed_x_msr(
    upper_data,
    lower_data,
    num_control_points_new,
    te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
    search_region=0.1,
):
    """
    Unified peak curvature-based fixed-x MSR optimizer for both surfaces.
    Uses a single split point determined by the surface with tighter curvature.
    Reorganizes data so that points with x < split_x from the tighter surface
    are moved to the other surface, creating non-monotonic x sequences.
    """
    from utils.bezier_utils import find_optimal_split_point_for_both_surfaces, reorganize_airfoil_data_for_split_point
    from core.solver_helpers import get_unified_peak_curvature_fixed_x_partition, build_control_points_with_peak_curvature_split, get_initial_guess_inner_y, minimize_with_debug_with_abort
    from core.error_functions import calculate_single_bezier_fitting_error
    
    if logger_func:
        logger_func("Finding optimal split point for both surfaces...")
    
    # Find the optimal split point for both surfaces
    split_x, split_y, upper_tangent, lower_tangent, tighter_surface_is_upper = find_optimal_split_point_for_both_surfaces(
        upper_data, lower_data, search_region
    )
    
    if logger_func:
        surface_name = "upper" if tighter_surface_is_upper else "lower"
        logger_func(f"Using split point from {surface_name} surface: ({split_x:.4f}, {split_y:.4f})")
        logger_func(f"Upper tangent: {upper_tangent}")
        logger_func(f"Lower tangent: {lower_tangent}")
    
    # Reorganize the airfoil data
    if logger_func:
        logger_func("Reorganizing airfoil data for unified split point...")
    
    new_upper_data, new_lower_data = reorganize_airfoil_data_for_split_point(
        upper_data, lower_data, split_x, tighter_surface_is_upper
    )
    
    if logger_func:
        logger_func(f"Data reorganization complete. Upper: {len(new_upper_data)} points, Lower: {len(new_lower_data)} points")
    
    # Get unified peak curvature-based partitioning
    (upper_fixed_inner_x_coords, lower_fixed_inner_x_coords,
     upper_free_indices, lower_free_indices,
     upper_fixed_indices, lower_fixed_indices,
     upper_fixed_y_values, lower_fixed_y_values, split_info) = get_unified_peak_curvature_fixed_x_partition(
        new_upper_data, new_lower_data, num_control_points_new, te_tangent_vector, 
        te_y=float(new_upper_data[-1, 1]), search_region=search_region
    )
    
    # Get initial guesses for free y variables
    upper_initial_y = get_initial_guess_inner_y(new_upper_data, upper_fixed_inner_x_coords)
    lower_initial_y = get_initial_guess_inner_y(new_lower_data, lower_fixed_inner_x_coords)
    
    upper_initial_y = upper_initial_y[upper_free_indices]
    lower_initial_y = lower_initial_y[lower_free_indices]
    
    # Combine all variables for optimization
    all_variables = np.concatenate([upper_initial_y, lower_initial_y])
    
    # Define the objective function for both surfaces
    def objective_function(variables):
        # Split variables back to upper and lower
        n_upper_free = len(upper_free_indices)
        upper_vars = variables[:n_upper_free]
        lower_vars = variables[n_upper_free:]
        
        # Build control points for both surfaces
        upper_ctrl = build_control_points_with_peak_curvature_split(
            upper_vars, upper_fixed_inner_x_coords, split_x, split_y, upper_tangent,
            float(new_upper_data[-1, 1]), upper_free_indices, upper_fixed_indices, upper_fixed_y_values
        )
        lower_ctrl = build_control_points_with_peak_curvature_split(
            lower_vars, lower_fixed_inner_x_coords, split_x, split_y, lower_tangent,
            float(new_lower_data[-1, 1]), lower_free_indices, lower_fixed_indices, lower_fixed_y_values
        )
        
        # Calculate combined error
        upper_error = calculate_single_bezier_fitting_error(upper_ctrl, new_upper_data, error_function=error_function, return_max_error=False)
        lower_error = calculate_single_bezier_fitting_error(lower_ctrl, new_lower_data, error_function=error_function, return_max_error=False)
        
        if isinstance(upper_error, tuple):
            upper_error = upper_error[0]
        if isinstance(lower_error, tuple):
            lower_error = lower_error[0]
        
        total_error = upper_error + lower_error
        return total_error
    
    # Add build_ctrl function for progress updates
    def build_ctrl_for_progress(variables):
        n_upper_free = len(upper_free_indices)
        upper_vars = variables[:n_upper_free]
        lower_vars = variables[n_upper_free:]
        
        upper_ctrl = build_control_points_with_peak_curvature_split(
            upper_vars, upper_fixed_inner_x_coords, split_x, split_y, upper_tangent,
            float(new_upper_data[-1, 1]), upper_free_indices, upper_fixed_indices, upper_fixed_y_values
        )
        lower_ctrl = build_control_points_with_peak_curvature_split(
            lower_vars, lower_fixed_inner_x_coords, split_x, split_y, lower_tangent,
            float(new_lower_data[-1, 1]), lower_free_indices, lower_fixed_indices, lower_fixed_y_values
        )
        
        return upper_ctrl, lower_ctrl
    
    objective_function.__build_ctrl__ = build_ctrl_for_progress
    
    # Run optimization
    if logger_func:
        logger_func("Starting unified peak curvature optimization...")
    
    result, _ = minimize_with_debug_with_abort(
        fun=objective_function,
        x0=all_variables,
        method="SLSQP",
        options=config.SLSQP_OPTIONS,
        progress_callback=logger_func,
        abort_flag=abort_flag
    )
    
    if not result.success:
        if logger_func:
            logger_func(f"Warning: Unified peak curvature optimization did not converge: {result.message}")
    
    # Build final control points
    final_upper_ctrl, final_lower_ctrl = build_ctrl_for_progress(result.x)
    
    if logger_func:
        logger_func("Unified peak curvature optimization completed.")
    
    # Return both control points and reorganized data for proper error calculation
    return (final_upper_ctrl, final_lower_ctrl), (new_upper_data, new_lower_data)


def build_bezier_unified_peak_curvature_free_x_softmax(
    upper_data,
    lower_data,
    num_control_points_new,
    te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
    search_region=0.1,
):
    """
    Unified peak curvature-based free-x softmax optimizer for both surfaces.
    Uses a single split point determined by the surface with tighter curvature.
    Reorganizes data so that points with x < split_x from the tighter surface
    are moved to the other surface, creating non-monotonic x sequences.
    Allows control points to move freely in both x and y directions.
    """
    from utils.bezier_utils import find_optimal_split_point_for_both_surfaces, reorganize_airfoil_data_for_split_point
    from utils.control_point_utils import variable_x_control_points
    from core.solver_helpers import minimize_with_debug_with_abort
    from core.error_functions import calculate_single_bezier_fitting_error
    from core import config
    
    if logger_func:
        logger_func("Finding optimal split point for both surfaces...")
    
    # Find the optimal split point for both surfaces
    split_x, split_y, upper_tangent, lower_tangent, tighter_surface_is_upper = find_optimal_split_point_for_both_surfaces(
        upper_data, lower_data, search_region
    )
    
    if logger_func:
        surface_name = "upper" if tighter_surface_is_upper else "lower"
        logger_func(f"Using split point from {surface_name} surface: ({split_x:.4f}, {split_y:.4f})")
        logger_func(f"Upper tangent: {upper_tangent}")
        logger_func(f"Lower tangent: {lower_tangent}")
    
    # Reorganize the airfoil data based on the split point
    if logger_func:
        logger_func("Reorganizing airfoil data for unified split point...")
    
    new_upper_data, new_lower_data = reorganize_airfoil_data_for_split_point(
        upper_data, lower_data, split_x, tighter_surface_is_upper
    )
    
    if logger_func:
        logger_func(f"Data reorganization complete. Upper: {len(new_upper_data)} points, Lower: {len(new_lower_data)} points")
        logger_func("Starting unified peak curvature optimization...")
    
    # Helper: per-surface progress logger wrappers to tag surface for GUI plot updates
    def make_surface_logger(surface_tag):
        def _surface_logger(*args):
            # Strings or other messages pass-through
            if len(args) == 1 and isinstance(args[0], str):
                if logger_func:
                    logger_func(args[0])
                return
            # Progress tuples from minimize_with_debug_with_abort
            if len(args) == 7:
                # iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl
                if logger_func:
                    try:
                        logger_func(*args, surface_tag)
                    except Exception:
                        # Fallback if upstream can't handle 8-arg form
                        logger_func(*args)
                return
            # Fallback
            if logger_func:
                logger_func(*args)
        return _surface_logger

    upper_logger = make_surface_logger("upper")
    lower_logger = make_surface_logger("lower")

    # Helper: bounds and monotonic constraints for free-x inner variables [x1,y1,x2,y2,...]
    def build_bounds_and_constraints(data_array, n_inner, split_x_local):
        y_vals = data_array[:, 1]
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        margin = 0.25 * (y_max - y_min + 1e-6)
        y_lo = y_min - margin
        y_hi = y_max + margin
        delta = 1e-4
        bounds = []
        for i in range(n_inner):
            # x_i bounds increasing from split_x to 1.0
            x_lo = min(max(split_x_local + i * delta, 0.0), 1.0)
            x_hi = max(min(1.0 - (n_inner - 1 - i) * delta, 1.0), x_lo + 1e-12)
            bounds.append((x_lo, x_hi))
            bounds.append((y_lo, y_hi))
        # Monotonic constraints: x_{i+1} - x_i - delta >= 0
        constraints = []
        for i in range(n_inner - 1):
            idx_i = 2 * i
            idx_ip1 = 2 * (i + 1)
            constraints.append({
                "type": "ineq",
                "fun": (lambda ii=idx_i, jj=idx_ip1: lambda vec: vec[jj] - vec[ii] - delta)()
            })
        return bounds, constraints

    # Helper: anchored, tighter bounds around MSR baseline to prevent collapse in softmax stage
    def build_anchored_bounds(data_array, msr_ctrl, split_x_local):
        n_inner = len(msr_ctrl) - 2
        y_vals = data_array[:, 1]
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        global_margin = 0.25 * (y_max - y_min + 1e-6)
        delta = 1e-4
        # Radii around MSR positions
        radius_x = 0.05  # 5% chord by default
        radius_y = 0.20 * (y_max - y_min + 1e-6)
        bounds = []
        for i in range(n_inner):
            x0 = float(msr_ctrl[i + 1, 0])
            y0 = float(msr_ctrl[i + 1, 1])
            x_lo_geom = max(split_x_local + i * delta, 0.0)
            x_hi_geom = min(1.0 - (n_inner - 1 - i) * delta, 1.0)
            x_lo = max(x_lo_geom, x0 - radius_x)
            x_hi = min(x_hi_geom, x0 + radius_x)
            if x_hi <= x_lo:
                x_hi = x_lo + 1e-6
            y_lo = max(y_min - global_margin, y0 - radius_y)
            y_hi = min(y_max + global_margin, y0 + radius_y)
            if y_hi <= y_lo:
                y_hi = y_lo + 1e-6
            bounds.append((x_lo, x_hi))
            bounds.append((y_lo, y_hi))
        # Monotonic constraints: x_{i+1} - x_i - delta >= 0
        constraints = []
        for i in range(n_inner - 1):
            idx_i = 2 * i
            idx_ip1 = 2 * (i + 1)
            constraints.append({
                "type": "ineq",
                "fun": (lambda ii=idx_i, jj=idx_ip1: lambda vec: vec[jj] - vec[ii] - delta)()
            })
        return bounds, constraints

    # Stage 1: Use the existing MSR builder to produce initial control polygons
    if logger_func:
        logger_func("Stage 1: MSR optimization for initial guess...")
        logger_func("Delegating Stage 1 to build_bezier_unified_peak_curvature_free_x_msr(...)")

    from core.bezier_optimizer import build_bezier_unified_peak_curvature_free_x_msr as _msr_builder
    (upper_msr_ctrl, lower_msr_ctrl), (new_upper_data, new_lower_data) = _msr_builder(
        upper_data,
        lower_data,
        num_control_points_new,
        te_tangent_vector,
        regularization_weight=regularization_weight,
        error_function=error_function,
        logger_func=logger_func,
        abort_flag=abort_flag,
        search_region=search_region,
    )

    # Per-surface TE for downstream steps
    te_y_upper = float(new_upper_data[-1, 1])
    te_y_lower = float(new_lower_data[-1, 1])
    # Bounds/constraints for Stage 2
    split_x = float(upper_msr_ctrl[0, 0])
    n_inner_upper = len(upper_msr_ctrl) - 2
    n_inner_lower = len(lower_msr_ctrl) - 2
    upper_bounds, upper_constraints = build_bounds_and_constraints(new_upper_data, n_inner_upper, split_x)
    lower_bounds, lower_constraints = build_bounds_and_constraints(new_lower_data, n_inner_lower, split_x)

    # Optional: Log MSR stage ctrl
    if logger_func:
        logger_func("MSR stage upper control points:")
        logger_func(str(upper_msr_ctrl))
        logger_func("MSR stage lower control points:")
        logger_func(str(lower_msr_ctrl))
    
    # Stage 2: Softmax optimization
    if logger_func:
        logger_func("Stage 2: Softmax optimization...")
    
    # Optimize upper surface with softmax
    if logger_func:
        logger_func("Optimizing upper surface (softmax)...")
    
    # Define softmax objective function for upper surface
    def upper_softmax_objective_function(variables):
        # variables: [x1, y1, x2, y2, ..., xn-1, yn-1] (excluding split point and TE)
        n_vars = len(variables)
        n_inner = n_vars // 2
        
        # Reconstruct control points
        control_points = np.zeros((num_control_points_new, 2))
        control_points[0] = [split_x, split_y]  # Split point
        
        # Set inner control points
        for i in range(n_inner):
            control_points[i + 1] = [variables[2*i], variables[2*i + 1]]
        
        control_points[-1] = [1.0, te_y_upper]  # TE
        
        # Apply tangent constraint to second control point (P1)
        if n_inner > 0:
            p0_to_p1 = control_points[1] - control_points[0]
            tx, ty = upper_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, upper_tangent) / tangent_norm_sq
                projected_p1 = np.array([split_x, split_y]) + t * np.array(upper_tangent)
                control_points[1] = projected_p1
        
        # Calculate residuals for softmax
        efun = error_function if error_function.endswith("_softmax") else (error_function + "_softmax")
        error_result = calculate_single_bezier_fitting_error(control_points, new_upper_data, error_function=efun, return_all=True)
        
        # Extract residuals from the return value
        if isinstance(error_result, tuple) and len(error_result) >= 1:
            residuals = error_result[0]  # First element is the residuals array
        else:
            # Fallback if the return format is unexpected
            residuals = np.array([error_result])
        
        # Apply softmax
        alpha = config.SOFTMAX_ALPHA
        abs_res = np.abs(residuals)
        # Prevent overflow by clipping large values
        max_val = np.max(abs_res)
        if max_val > 100:  # Threshold to prevent overflow
            abs_res = np.clip(abs_res, 0, 100)
        softmax_val = np.log(np.sum(np.exp(alpha * abs_res))) / alpha
        
        if regularization_weight and regularization_weight != 0:
            softmax_val = softmax_val + regularization_weight * smoothness_penalty(control_points)
        return softmax_val

    # Hooks for progress and residuals
    def _upper_softmax_build_ctrl(vars_vec):
        n_vars_local = len(vars_vec)
        n_inner_local = n_vars_local // 2
        ctrl = np.zeros((num_control_points_new, 2))
        ctrl[0] = [split_x, split_y]
        for i in range(n_inner_local):
            ctrl[i + 1] = [vars_vec[2*i], vars_vec[2*i + 1]]
        ctrl[-1] = [1.0, te_y_upper]
        if n_inner_local > 0:
            p0_to_p1 = ctrl[1] - ctrl[0]
            tx, ty = upper_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, upper_tangent) / tangent_norm_sq
                ctrl[1] = np.array([split_x, split_y]) + t * np.array(upper_tangent)
        return ctrl
    def _upper_softmax_residuals(vars_vec):
        ctrl = _upper_softmax_build_ctrl(vars_vec)
        res_tuple = calculate_single_bezier_fitting_error(ctrl, new_upper_data, error_function=error_function, return_all=True)
        return res_tuple[0] if isinstance(res_tuple, tuple) else np.array([res_tuple])
    upper_softmax_objective_function.__build_ctrl__ = _upper_softmax_build_ctrl
    upper_softmax_objective_function.__get_residuals__ = _upper_softmax_residuals
    
    # Optimize upper surface with softmax
    # Softmax stage: anchored bounds/constraints around MSR
    upper_bounds, upper_constraints = build_anchored_bounds(new_upper_data, upper_msr_ctrl, split_x)
    result, _ = minimize_with_debug_with_abort(
        fun=upper_softmax_objective_function,
        x0=np.concatenate([upper_msr_ctrl[1:-1, 0], upper_msr_ctrl[1:-1, 1]]),  # Use MSR result as initial guess
        method="SLSQP",
        options=config.SLSQP_OPTIONS,
        abort_flag=abort_flag,
        bounds=upper_bounds,
        constraints=upper_constraints,
        progress_callback=upper_logger,
    )
    
    if result is None:
        if logger_func:
            logger_func("Upper surface softmax optimization failed")
        return None, None
    
    # Reconstruct upper control points from softmax result
    n_inner = (len(result.x) // 2)
    upper_final_ctrl = np.zeros((num_control_points_new, 2))
    upper_final_ctrl[0] = [split_x, split_y]  # Split point
    for i in range(n_inner):
        upper_final_ctrl[i + 1] = [result.x[2*i], result.x[2*i + 1]]
    upper_final_ctrl[-1] = [1.0, te_y_upper]  # TE
    
    # Apply tangent constraint to second control point
    if n_inner > 0:
        p0_to_p1 = upper_final_ctrl[1] - upper_final_ctrl[0]
        tx, ty = upper_tangent
        if abs(tx) > 1e-12 or abs(ty) > 1e-12:
            tangent_norm_sq = tx*tx + ty*ty
            t = np.dot(p0_to_p1, upper_tangent) / tangent_norm_sq
            projected_p1 = np.array([split_x, split_y]) + t * np.array(upper_tangent)
            upper_final_ctrl[1] = projected_p1

    # Log softmax-stage upper control points
    if logger_func:
        import numpy as _np
        _np.set_printoptions(precision=6, suppress=True)
        logger_func("Softmax stage upper control points:")
        logger_func(str(upper_final_ctrl))
    
    # Optimize lower surface with softmax
    if logger_func:
        logger_func("Optimizing lower surface (softmax)...")
    
    # Define softmax objective function for lower surface
    def lower_softmax_objective_function(variables):
        # variables: [x1, y1, x2, y2, ..., xn-1, yn-1] (excluding split point and TE)
        n_vars = len(variables)
        n_inner = n_vars // 2
        
        # Reconstruct control points
        control_points = np.zeros((num_control_points_new, 2))
        control_points[0] = [split_x, split_y]  # Split point
        
        # Set inner control points
        for i in range(n_inner):
            control_points[i + 1] = [variables[2*i], variables[2*i + 1]]
        
        control_points[-1] = [1.0, te_y_lower]  # TE
        
        # Apply tangent constraint to second control point (P1)
        if n_inner > 0:
            p0_to_p1 = control_points[1] - control_points[0]
            tx, ty = lower_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, lower_tangent) / tangent_norm_sq
                projected_p1 = np.array([split_x, split_y]) + t * np.array(lower_tangent)
                control_points[1] = projected_p1
        
        # Calculate residuals for softmax
        efun = error_function if error_function.endswith("_softmax") else (error_function + "_softmax")
        error_result = calculate_single_bezier_fitting_error(control_points, new_lower_data, error_function=efun, return_all=True)
        
        # Extract residuals from the return value
        if isinstance(error_result, tuple) and len(error_result) >= 1:
            residuals = error_result[0]  # First element is the residuals array
        else:
            # Fallback if the return format is unexpected
            residuals = np.array([error_result])
        
        # Apply softmax
        alpha = config.SOFTMAX_ALPHA
        abs_res = np.abs(residuals)
        # Prevent overflow by clipping large values
        max_val = np.max(abs_res)
        if max_val > 100:  # Threshold to prevent overflow
            abs_res = np.clip(abs_res, 0, 100)
        softmax_val = np.log(np.sum(np.exp(alpha * abs_res))) / alpha
        
        if regularization_weight and regularization_weight != 0:
            softmax_val = softmax_val + regularization_weight * smoothness_penalty(control_points)
        return softmax_val

    # Hooks for progress and residuals (lower)
    def _lower_softmax_build_ctrl(vars_vec):
        n_vars_local = len(vars_vec)
        n_inner_local = n_vars_local // 2
        ctrl = np.zeros((num_control_points_new, 2))
        ctrl[0] = [split_x, split_y]
        for i in range(n_inner_local):
            ctrl[i + 1] = [vars_vec[2*i], vars_vec[2*i + 1]]
        ctrl[-1] = [1.0, te_y_lower]
        if n_inner_local > 0:
            p0_to_p1 = ctrl[1] - ctrl[0]
            tx, ty = lower_tangent
            if abs(tx) > 1e-12 or abs(ty) > 1e-12:
                tangent_norm_sq = tx*tx + ty*ty
                t = np.dot(p0_to_p1, lower_tangent) / tangent_norm_sq
                ctrl[1] = np.array([split_x, split_y]) + t * np.array(lower_tangent)
        return ctrl
    def _lower_softmax_residuals(vars_vec):
        ctrl = _lower_softmax_build_ctrl(vars_vec)
        res_tuple = calculate_single_bezier_fitting_error(ctrl, new_lower_data, error_function=error_function, return_all=True)
        return res_tuple[0] if isinstance(res_tuple, tuple) else np.array([res_tuple])
    lower_softmax_objective_function.__build_ctrl__ = _lower_softmax_build_ctrl
    lower_softmax_objective_function.__get_residuals__ = _lower_softmax_residuals
    
    # Optimize lower surface with softmax
    # Anchored bounds/constraints around lower MSR
    lower_bounds, lower_constraints = build_anchored_bounds(new_lower_data, lower_msr_ctrl, split_x)
    result, _ = minimize_with_debug_with_abort(
        fun=lower_softmax_objective_function,
        x0=np.concatenate([lower_msr_ctrl[1:-1, 0], lower_msr_ctrl[1:-1, 1]]),  # Use MSR result as initial guess
        method="SLSQP",
        options=config.SLSQP_OPTIONS,
        abort_flag=abort_flag,
        bounds=lower_bounds,
        constraints=lower_constraints,
        progress_callback=lower_logger,
    )
    
    if result is None:
        if logger_func:
            logger_func("Lower surface softmax optimization failed")
        return None, None
    
    # Reconstruct lower control points from softmax result
    n_inner = (len(result.x) // 2)
    lower_final_ctrl = np.zeros((num_control_points_new, 2))
    lower_final_ctrl[0] = [split_x, split_y]  # Split point
    for i in range(n_inner):
        lower_final_ctrl[i + 1] = [result.x[2*i], result.x[2*i + 1]]
    lower_final_ctrl[-1] = [1.0, te_y_lower]  # TE
    
    # Apply tangent constraint to second control point
    if n_inner > 0:
        p0_to_p1 = lower_final_ctrl[1] - lower_final_ctrl[0]
        tx, ty = lower_tangent
        if abs(tx) > 1e-12 or abs(ty) > 1e-12:
            tangent_norm_sq = tx*tx + ty*ty
            t = np.dot(p0_to_p1, lower_tangent) / tangent_norm_sq
            projected_p1 = np.array([split_x, split_y]) + t * np.array(lower_tangent)
            lower_final_ctrl[1] = projected_p1

    # Log softmax-stage lower control points
    if logger_func:
        import numpy as _np
        _np.set_printoptions(precision=6, suppress=True)
        logger_func("Softmax stage lower control points:")
        logger_func(str(lower_final_ctrl))
    
    if logger_func:
        logger_func("Unified peak curvature optimization completed.")
    
    return (upper_final_ctrl, lower_final_ctrl), (new_upper_data, new_lower_data)









