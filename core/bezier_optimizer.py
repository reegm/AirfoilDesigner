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
    run_minmax_stage,
    get_initial_guess_inner_y,
    minimize_with_debug_with_abort
)
from core.bezier_unified_optimizer import optimize_bezier


def build_bezier_hybrid_uncoupled(
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
    Uncoupled hybrid optimizer pipeline (euclidean error only):
    1) Basin-hopping style fixed-x MSR with bounded SLSQP restarts
    2) Switch to minmax (softmax objective) while still fixed-x
    3) If stalled, switch to free-x minmax

    Notes:
    - We deliberately ignore any orthogonal error variants for now and use euclidean only.
    - Regularization weight is applied in minmax stages (fixed-x and free-x) consistent with existing design.
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
    hops_fixed = max(0, int(config.HYBRID_BH_HOPS_FIXED_MINMAX))
    hops_free = max(0, int(config.HYBRID_BH_HOPS_FREE_MINMAX))
    perturb_std = float(config.HYBRID_BH_PERTURB_STD)
    current_best_y = center_y.copy()
    # Stage 1 logging
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func(f"Hybrid: Stage 1 (fixed-x MSR) hops={hops_msr}")
    for hop in range(hops_msr):
        if logger_func and config.DEBUG_WORKER_LOGGING:
            logger_func(f"Hybrid: Stage 1 hop {hop+1}/{hops_msr}")
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
                logger_func(f"Hybrid: Stage 1 hop {hop+1}/{hops_msr} improved best max error to {best_max_err:.6e}")

    # Stage 2: Fixed-x minmax (softmax objective) with basin-hopping restarts
    if abort_flag is not None and abort_flag.value:
        return best_ctrl
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func("Hybrid: Stage 2 (fixed-x softmax) starting")
    softmax_opts = dict(config.SLSQP_OPTIONS)
    softmax_opts["maxiter"] = config.HYBRID_LOCAL_MAXITER_MINMAX_FIXED
    # First local softmax
    ctrl_fixed_minmax = optimize_bezier(
        initial_ctrl=best_ctrl,
        original_data=original_data,
        mode="fixed-x",
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
    _, fixed_minmax_max, _ = calculate_single_bezier_fitting_error(ctrl_fixed_minmax, original_data, error_function="euclidean", return_max_error=True)
    if fixed_minmax_max < best_max_err:
        best_ctrl = ctrl_fixed_minmax
        best_max_err = fixed_minmax_max
    # Basin-hopping restarts around fixed-x minmax
    current_ctrl = best_ctrl
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func(f"Hybrid: Stage 2 (fixed-x softmax) hops={hops_fixed}")
    for hop in range(hops_fixed):
        if logger_func and config.DEBUG_WORKER_LOGGING:
            logger_func(f"Hybrid: Stage 2 hop {hop+1}/{hops_fixed}")
        if abort_flag is not None and abort_flag.value:
            break
        # Perturb current best inner y only (keep fixed-x)
        y_full = np.interp(fixed_inner_x, current_ctrl[:, 0], current_ctrl[:, 1])
        y_free = y_full[free_idx]
        y_trial = np.clip(y_free + rng.normal(0.0, perturb_std, size=y_free.shape), -1.0, 1.0)
        ctrl_trial = build_control_points_with_fixed(y_trial, fixed_inner_x, float(original_data[-1, 1]), free_idx, fixed_idx, fixed_y_vals)
        # Local softmax from this trial
        ctrl_trial_minmax = optimize_bezier(
            initial_ctrl=ctrl_trial,
            original_data=original_data,
            mode="fixed-x",
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
        _, trial_max, _ = calculate_single_bezier_fitting_error(ctrl_trial_minmax, original_data, error_function="euclidean", return_max_error=True)
        if trial_max < best_max_err:
            best_max_err = trial_max
            best_ctrl = ctrl_trial_minmax
            current_ctrl = ctrl_trial_minmax
            if logger_func:
                logger_func(f"Hybrid: Stage 2 hop {hop+1}/{hops_fixed} improved best max error to {best_max_err:.6e}")

    # Stage 3: Free-x minmax (softmax objective) with basin-hopping restarts if not aborted
    if abort_flag is not None and abort_flag.value:
        return best_ctrl

    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func("Hybrid: Stage 3 (free-x softmax) starting")
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
    _, free_minmax_max, _ = calculate_single_bezier_fitting_error(final_ctrl, original_data, error_function="euclidean", return_max_error=True)
    if free_minmax_max < best_max_err:
        best_ctrl = final_ctrl
        best_max_err = free_minmax_max

    # Basin-hopping restarts for free-x: perturb inner x and y slightly
    n = len(best_ctrl)
    x_inner0 = best_ctrl[2:-1, 0]
    y_inner0 = best_ctrl[1:-1, 1]
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func(f"Hybrid: Stage 3 (free-x softmax) hops={hops_free}")
    for hop in range(hops_free):
        if logger_func and config.DEBUG_WORKER_LOGGING:
            logger_func(f"Hybrid: Stage 3 hop {hop+1}/{hops_free}")
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
                logger_func(f"Hybrid: Stage 3 hop {hop+1}/{hops_free} improved best max error to {best_max_err:.6e}")

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



def build_bezier_free_x_minmax(
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
    Uncoupled free-x single Bezier optimizer using minmax objective with softmax.
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
        logger_func("Running unified free-x minmax optimization...")

    # Stage 2: Minmax optimization using unified optimizer
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
        logger_func("Unified free-x minmax optimization completed.")
    
    return control_points

def build_bezier_fixed_x_minmax(
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
    Uncoupled fixed-x single Bezier optimizer using minmax objective with softmax.
    Uses the unified optimizer directly (no preliminary MSR stage needed).
    """
    if logger_func:
        logger_func("Running unified fixed-x minmax optimization...")

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
        logger_func("Unified fixed-x minmax optimization completed.")
    
    return control_points









