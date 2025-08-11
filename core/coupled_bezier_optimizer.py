import numpy as np
from core import config
# from core.error_functions import calculate_euclidean_error, calculate_orthogonal_error  # Deprecated
from core.error_functions import calculate_single_bezier_fitting_error
from core.solver_helpers import (
    smoothness_penalty,
)
from utils.bezier_utils import leading_edge_curvature



# --- Unified Bezier Builder Functions ---

def build_coupled_bezier_fixed_x_msr(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Unified coupled fixed-x Bezier optimizer (G2 at LE, tangency at TE, MSR objective).
    Uses the unified optimizer to reproduce legacy behavior.
    """
    from core.bezier_unified_optimizer import optimize_bezier
    
    if logger_func:
        logger_func("Running unified coupled fixed-x MSR optimization...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Use the unified optimizer for coupled fixed-x MSR
    result = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,
        error_function=error_function,
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,  # For upper surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    # The unified optimizer returns a tuple (upper_ctrl, lower_ctrl) for coupled mode
    return result 

def build_coupled_bezier_fixed_x_softmax(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Unified coupled fixed-x Bezier optimizer (G2 at LE, tangency at TE, softmax objective).
    Uses the unified optimizer with 2-stage approach: MSR initial guess followed by softmax optimization.
    """
    from core.bezier_unified_optimizer import optimize_bezier
    
    if logger_func:
        logger_func("Stage 1: Running coupled fixed-x MSR optimization for initial guess...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Run coupled fixed-x MSR to get good initial guess
    initial_result = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,  # Use coupled for better initial guess
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=0,  # No regularization for initial guess
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,  # Required for coupled fixed-x mode
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    # Extract upper and lower control points from coupled result
    initial_upper, initial_lower = initial_result
    
    if logger_func:
        logger_func("Stage 2: Running unified coupled fixed-x softmax (softmax) optimization...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Use the unified optimizer for coupled fixed-x softmax (second stage)
    result = optimize_bezier(
        initial_ctrl=initial_upper,  # Use Stage 1 result as initial guess
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,
        error_function=error_function,
        objective="softmax",  # Use softmax for softmax optimization
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_initial_ctrl=initial_lower,  # Use Stage 1 result as initial guess for lower surface
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,  # Required for coupled fixed-x mode
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,  # Required for coupled fixed-x mode
    )
    
    if logger_func:
        logger_func("Stage 2: Unified coupled fixed-x softmax (softmax) optimization completed.")
    
    # The unified optimizer returns a tuple (upper_ctrl, lower_ctrl) for coupled mode
    return result 

def build_coupled_bezier_free_x_msr(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Unified coupled free-x Bezier optimizer (G2 at LE, tangency at TE, MSR objective).
    Uses the unified optimizer with 2-stage approach: fixed-x MSR initial guess followed by free-x MSR optimization.
    """
    from core.bezier_unified_optimizer import optimize_bezier
    from utils.control_point_utils import variable_x_control_points
    
    if logger_func:
        logger_func("Stage 1: Running coupled fixed-x MSR optimization for initial guess...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Run coupled fixed-x MSR to get good initial guess
    initial_result = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,  # Use coupled for better initial guess
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,  # Required for coupled fixed-x mode
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    # Extract upper and lower control points from coupled result
    initial_upper, initial_lower = initial_result
    
    if logger_func:
        logger_func("Stage 2: Running unified coupled free-x MSR optimization...")
    
    # Stage 2: Run free-x MSR using the fixed-x result as initial guess
    result = optimize_bezier(
        initial_ctrl=initial_upper,  # Use fixed-x result as initial guess
        original_data=original_upper_data,
        mode="free-x",
        coupled=True,
        error_function=error_function,
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_initial_ctrl=initial_lower,  # Use fixed-x result as initial guess for lower surface
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
    )
    
    # The unified optimizer returns a tuple (upper_ctrl, lower_ctrl) for coupled mode
    return result 

def build_coupled_bezier_free_x_softmax(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Unified coupled free-x Bezier optimizer (G2 at LE, tangency at TE, softmax objective).
    Uses the unified optimizer with 2-stage approach: fixed-x MSR initial guess followed by free-x softmax optimization.
    """
    from core.bezier_unified_optimizer import optimize_bezier
    from utils.control_point_utils import variable_x_control_points
    
    if logger_func:
        logger_func("Stage 1: Running coupled fixed-x MSR optimization for initial guess...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Run coupled fixed-x MSR to get good initial guess
    initial_result = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,  # Use coupled for better initial guess
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,  # Required for coupled fixed-x mode
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    # Extract upper and lower control points from coupled result
    initial_upper, initial_lower = initial_result
    
    if logger_func:
        logger_func("Stage 2: Running unified coupled free-x softmax (softmax) optimization...")
    
    # Stage 2: Run free-x softmax using the fixed-x result as initial guess
    result = optimize_bezier(
        initial_ctrl=initial_upper,  # Use fixed-x result as initial guess
        original_data=original_upper_data,
        mode="free-x",
        coupled=True,
        error_function=error_function,
        objective="softmax",  # Use softmax for softmax optimization
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_initial_ctrl=initial_lower,  # Use fixed-x result as initial guess for lower surface
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
    )
    
    # The unified optimizer returns a tuple (upper_ctrl, lower_ctrl) for coupled mode
    return result 

def build_coupled_bezier_staged(
    original_upper_data,
    original_lower_data,
    regularization_weight,
    te_tangent_vector_upper,
    te_tangent_vector_lower,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None,
):
    """
    Coupled staged optimizer pipeline (G2 enabled, euclidean error only):
    1) Basin-hopping style coupled fixed-x MSR with bounded SLSQP restarts
    2) Switch to coupled softmax (softmax objective) while still fixed-x
    3) If stalled, switch to coupled free-x softmax

    Notes:
    - We deliberately ignore any orthogonal error variants for now and use euclidean only.
    - Regularization weight is applied in softmax stages (fixed-x and free-x) consistent with existing design.
    - Uses plateau detection from minimize_with_debug_with_abort via progress to determine stalling implicitly by
      limiting per-stage max iterations and checking no best_true_max improvement across hops.
    - G2 constraint is maintained throughout all stages.
    """
    import numpy as np
    from core.bezier_unified_optimizer import optimize_bezier
    from core.error_functions import calculate_single_bezier_fitting_error
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Coupled fixed-x MSR with basin-hopping like restarts to escape poor initializations
    if logger_func:
        logger_func("Stage 1 (coupled fixed-x MSR) starting")
    
    # Build an initial coupled fixed-x MSR solution as a baseline
    base_result = optimize_bezier(
        initial_ctrl=None,
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,
        error_function="euclidean",
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=0.0,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )

    if abort_flag is not None and abort_flag.value:
        return base_result

    # Extract upper and lower control points from coupled result
    best_upper, best_lower = base_result
    
    # Evaluate best using euclidean max error for consistency
    upper_distances, upper_rms, upper_max_error = calculate_single_bezier_fitting_error(
        best_upper, original_upper_data, error_function="euclidean", return_all=True
    )
    lower_distances, lower_rms, lower_max_error = calculate_single_bezier_fitting_error(
        best_lower, original_lower_data, error_function="euclidean", return_all=True
    )
    best_max_err = max(upper_max_error, lower_max_error)

    # Basin-hopping restarts for Stage 1
    rng = np.random.default_rng(12345)
    hops_msr = max(0, int(config.HYBRID_BH_HOPS_MSR))
    perturb_std = float(config.HYBRID_BH_PERTURB_STD)
    
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func(f"Stage 1 (coupled fixed-x MSR) hops={hops_msr}")
    
    for hop in range(hops_msr):
        if logger_func and config.DEBUG_WORKER_LOGGING:
            logger_func(f"Stage 1 hop {hop+1}/{hops_msr}")
        if abort_flag is not None and abort_flag.value:
            break
            
        # Perturb the current best control points slightly
        trial_upper = best_upper.copy()
        trial_lower = best_lower.copy()
        
        # Perturb inner control points (skip LE and TE)
        for i in range(1, len(trial_upper) - 1):
            trial_upper[i, 1] += rng.normal(0.0, perturb_std)
            trial_lower[i, 1] += rng.normal(0.0, perturb_std)
        
        # Clamp y values to reasonable range
        trial_upper[:, 1] = np.clip(trial_upper[:, 1], -1.0, 1.0)
        trial_lower[:, 1] = np.clip(trial_lower[:, 1], -1.0, 1.0)
        
        # Optimize this trial locally with coupled fixed-x MSR
        trial_result = optimize_bezier(
            initial_ctrl=trial_upper,
            original_data=original_upper_data,
            mode="fixed-x",
            coupled=True,
            error_function="euclidean",
            objective="msr",
            te_y=te_y_upper,
            te_tangent_vector=te_tangent_vector_upper,
            regularization_weight=0.0,
            logger_func=logger_func,
            abort_flag=abort_flag,
            g2_constraint=True,
            lower_data=original_lower_data,
            lower_initial_ctrl=trial_lower,
            lower_te_y=te_y_lower,
            lower_te_tangent_vector=te_tangent_vector_lower,
            is_upper_surface=True,
            num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
        )
        
        if trial_result is not None:
            candidate_upper, candidate_lower = trial_result
            _, _, candidate_upper_max = calculate_single_bezier_fitting_error(
                candidate_upper, original_upper_data, error_function="euclidean", return_all=True
            )
            _, _, candidate_lower_max = calculate_single_bezier_fitting_error(
                candidate_lower, original_lower_data, error_function="euclidean", return_all=True
            )
            candidate_max_err = max(candidate_upper_max, candidate_lower_max)
            
            if candidate_max_err < best_max_err:
                best_max_err = candidate_max_err
                best_upper = candidate_upper
                best_lower = candidate_lower
                if logger_func:
                    logger_func(f"Stage 1 hop {hop+1}/{hops_msr} improved best max error to {best_max_err:.6e}")

    # Stage 2: Coupled fixed-x softmax (softmax objective) with basin-hopping restarts
    if abort_flag is not None and abort_flag.value:
        return (best_upper, best_lower)
        
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func("Stage 2 (coupled fixed-x softmax) starting")
    
    # First local coupled fixed-x softmax
    fixed_softmax_result = optimize_bezier(
        initial_ctrl=best_upper,
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=True,
        error_function="euclidean",
        objective="softmax",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_initial_ctrl=best_lower,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
        is_upper_surface=True,
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    if fixed_softmax_result is not None:
        fixed_upper, fixed_lower = fixed_softmax_result
        _, _, fixed_upper_max = calculate_single_bezier_fitting_error(
            fixed_upper, original_upper_data, error_function="euclidean", return_all=True
        )
        _, _, fixed_lower_max = calculate_single_bezier_fitting_error(
            fixed_lower, original_lower_data, error_function="euclidean", return_all=True
        )
        fixed_max_err = max(fixed_upper_max, fixed_lower_max)
        
        if fixed_max_err < best_max_err:
            best_upper = fixed_upper
            best_lower = fixed_lower
            best_max_err = fixed_max_err
    
    # Basin-hopping restarts around coupled fixed-x softmax
    hops_fixed = max(0, int(config.HYBRID_BH_HOPS_FIXED_MINMAX))
    current_upper = best_upper.copy()
    current_lower = best_lower.copy()
    
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func(f"Stage 2 (coupled fixed-x softmax) hops={hops_fixed}")
    
    for hop in range(hops_fixed):
        if logger_func and config.DEBUG_WORKER_LOGGING:
            logger_func(f"Stage 2 hop {hop+1}/{hops_fixed}")
        if abort_flag is not None and abort_flag.value:
            break
            
        # Perturb current best inner y only (keep fixed-x)
        trial_upper = current_upper.copy()
        trial_lower = current_lower.copy()
        
        for i in range(1, len(trial_upper) - 1):
            trial_upper[i, 1] += rng.normal(0.0, perturb_std)
            trial_lower[i, 1] += rng.normal(0.0, perturb_std)
        
        # Clamp y values to reasonable range
        trial_upper[:, 1] = np.clip(trial_upper[:, 1], -1.0, 1.0)
        trial_lower[:, 1] = np.clip(trial_lower[:, 1], -1.0, 1.0)
        
        # Local coupled softmax from this trial
        trial_result = optimize_bezier(
            initial_ctrl=trial_upper,
            original_data=original_upper_data,
            mode="fixed-x",
            coupled=True,
            error_function="euclidean",
            objective="softmax",
            te_y=te_y_upper,
            te_tangent_vector=te_tangent_vector_upper,
            regularization_weight=regularization_weight,
            logger_func=logger_func,
            abort_flag=abort_flag,
            g2_constraint=True,
            lower_data=original_lower_data,
            lower_initial_ctrl=trial_lower,
            lower_te_y=te_y_lower,
            lower_te_tangent_vector=te_tangent_vector_lower,
            is_upper_surface=True,
            num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
        )
        
        if trial_result is not None:
            trial_upper_opt, trial_lower_opt = trial_result
            _, _, trial_upper_max = calculate_single_bezier_fitting_error(
                trial_upper_opt, original_upper_data, error_function="euclidean", return_all=True
            )
            _, _, trial_lower_max = calculate_single_bezier_fitting_error(
                trial_lower_opt, original_lower_data, error_function="euclidean", return_all=True
            )
            trial_max_err = max(trial_upper_max, trial_lower_max)
            
            if trial_max_err < best_max_err:
                best_max_err = trial_max_err
                best_upper = trial_upper_opt
                best_lower = trial_lower_opt
                current_upper = trial_upper_opt
                current_lower = trial_lower_opt
                if logger_func:
                    logger_func(f"Stage 2 hop {hop+1}/{hops_fixed} improved best max error to {best_max_err:.6e}")

    # Stage 3: Coupled free-x softmax (softmax objective) with basin-hopping restarts if not aborted
    if abort_flag is not None and abort_flag.value:
        return (best_upper, best_lower)

    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func("Stage 3 (coupled free-x softmax) starting")
    
    # Final coupled free-x softmax
    final_result = optimize_bezier(
        initial_ctrl=best_upper,
        original_data=original_upper_data,
        mode="free-x",
        coupled=True,
        error_function="euclidean",
        objective="softmax",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        g2_constraint=True,
        lower_data=original_lower_data,
        lower_initial_ctrl=best_lower,
        lower_te_y=te_y_lower,
        lower_te_tangent_vector=te_tangent_vector_lower,
    )
    
    if final_result is not None:
        final_upper, final_lower = final_result
        _, _, final_upper_max = calculate_single_bezier_fitting_error(
            final_upper, original_upper_data, error_function="euclidean", return_all=True
        )
        _, _, final_lower_max = calculate_single_bezier_fitting_error(
            final_lower, original_lower_data, error_function="euclidean", return_all=True
        )
        final_max_err = max(final_upper_max, final_lower_max)
        
        if final_max_err < best_max_err:
            best_upper = final_upper
            best_lower = final_lower
            best_max_err = final_max_err

    # Basin-hopping restarts for coupled free-x: perturb inner x and y slightly
    hops_free = max(0, int(config.HYBRID_BH_HOPS_FREE_MINMAX))
    n_upper = len(best_upper)
    n_lower = len(best_lower)
    x_inner_upper = best_upper[2:-1, 0]
    y_inner_upper = best_upper[1:-1, 1]
    x_inner_lower = best_lower[2:-1, 0]
    y_inner_lower = best_lower[1:-1, 1]
    
    if logger_func and config.DEBUG_WORKER_LOGGING:
        logger_func(f"Stage 3 (coupled free-x softmax) hops={hops_free}")
    
    for hop in range(hops_free):
        if logger_func and config.DEBUG_WORKER_LOGGING:
            logger_func(f"Stage 3 hop {hop+1}/{hops_free}")
        if abort_flag is not None and abort_flag.value:
            break
            
        # Perturb inner x and y for both surfaces
        x_trial_upper = np.clip(x_inner_upper + rng.normal(0.0, perturb_std, size=x_inner_upper.shape), 0.0, 1.0)
        x_trial_lower = np.clip(x_inner_lower + rng.normal(0.0, perturb_std, size=x_inner_lower.shape), 0.0, 1.0)
        
        # Keep x monotone increasing
        x_trial_upper = np.maximum.accumulate(x_trial_upper)
        x_trial_lower = np.maximum.accumulate(x_trial_lower)
        
        y_trial_upper = np.clip(y_inner_upper + rng.normal(0.0, perturb_std, size=y_inner_upper.shape), -1.0, 1.0)
        y_trial_lower = np.clip(y_inner_lower + rng.normal(0.0, perturb_std, size=y_inner_lower.shape), -1.0, 1.0)
        
        # Build trial control points
        trial_upper = np.zeros_like(best_upper)
        trial_lower = np.zeros_like(best_lower)
        
        # Upper surface
        trial_upper[0] = [0.0, 0.0]  # LE
        trial_upper[1] = [0.0, y_trial_upper[0]]  # First control point
        trial_upper[2:-1, 0] = x_trial_upper
        trial_upper[2:-1, 1] = y_trial_upper[1:]
        trial_upper[-1] = [1.0, te_y_upper]  # TE
        
        # Lower surface
        trial_lower[0] = [0.0, 0.0]  # LE
        trial_lower[1] = [0.0, y_trial_lower[0]]  # First control point
        trial_lower[2:-1, 0] = x_trial_lower
        trial_lower[2:-1, 1] = y_trial_lower[1:]
        trial_lower[-1] = [1.0, te_y_lower]  # TE
        
        # Local coupled free-x softmax from this trial
        trial_result = optimize_bezier(
            initial_ctrl=trial_upper,
            original_data=original_upper_data,
            mode="free-x",
            coupled=True,
            error_function="euclidean",
            objective="softmax",
            te_y=te_y_upper,
            te_tangent_vector=te_tangent_vector_upper,
            regularization_weight=regularization_weight,
            logger_func=logger_func,
            abort_flag=abort_flag,
            g2_constraint=True,
            lower_data=original_lower_data,
            lower_initial_ctrl=trial_lower,
            lower_te_y=te_y_lower,
            lower_te_tangent_vector=te_tangent_vector_lower,
        )
        
        if trial_result is not None:
            trial_upper_opt, trial_lower_opt = trial_result
            _, _, trial_upper_max = calculate_single_bezier_fitting_error(
                trial_upper_opt, original_upper_data, error_function="euclidean", return_all=True
            )
            _, _, trial_lower_max = calculate_single_bezier_fitting_error(
                trial_lower_opt, original_lower_data, error_function="euclidean", return_all=True
            )
            trial_max_err = max(trial_upper_max, trial_lower_max)
            
            if trial_max_err < best_max_err:
                best_upper = trial_upper_opt
                best_lower = trial_lower_opt
                best_max_err = trial_max_err
                x_inner_upper = trial_upper_opt[2:-1, 0]
                y_inner_upper = trial_upper_opt[1:-1, 1]
                x_inner_lower = trial_lower_opt[2:-1, 0]
                y_inner_lower = trial_lower_opt[1:-1, 1]
                if logger_func:
                    logger_func(f"Stage 3 hop {hop+1}/{hops_free} improved best max error to {best_max_err:.6e}")

    return (best_upper, best_lower) 