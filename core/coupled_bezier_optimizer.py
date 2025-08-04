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

def build_coupled_bezier_fixed_x_minmax(
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
        logger_func("Stage 1: Running uncoupled fixed-x MSR optimization for initial guess...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Run uncoupled fixed-x MSR to get good initial guess (much faster than coupled)
    # This matches the legacy implementation which uses uncoupled for Stage 1
    initial_upper = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=False,  # Use uncoupled for speed
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=0,  # No regularization for initial guess
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=True,  # For upper surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    initial_lower = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_lower_data,
        mode="fixed-x",
        coupled=False,  # Use uncoupled for speed
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_lower,
        te_tangent_vector=te_tangent_vector_lower,
        regularization_weight=0,  # No regularization for initial guess
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=False,  # For lower surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    if logger_func:
        logger_func("Stage 2: Running unified coupled fixed-x minmax (softmax) optimization...")
    
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
        objective="softmax",  # Use softmax for minmax optimization
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
        logger_func("Stage 2: Unified coupled fixed-x minmax (softmax) optimization completed.")
    
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
        logger_func("Stage 1: Running uncoupled fixed-x MSR optimization for initial guess...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Run uncoupled fixed-x MSR to get good initial guess (much faster than coupled)
    # This matches the legacy implementation which uses uncoupled for Stage 1
    initial_upper = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=False,  # Use uncoupled for speed
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=True,  # For upper surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    initial_lower = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_lower_data,
        mode="fixed-x",
        coupled=False,  # Use uncoupled for speed
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_lower,
        te_tangent_vector=te_tangent_vector_lower,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=False,  # For lower surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
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

def build_coupled_bezier_free_x_minmax(
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
        logger_func("Stage 1: Running uncoupled fixed-x MSR optimization for initial guess...")
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Stage 1: Run uncoupled fixed-x MSR to get good initial guess (much faster than coupled)
    # This matches the legacy implementation which uses uncoupled for Stage 1
    initial_upper = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_upper_data,
        mode="fixed-x",
        coupled=False,  # Use uncoupled for speed
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_upper,
        te_tangent_vector=te_tangent_vector_upper,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=True,  # For upper surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    initial_lower = optimize_bezier(
        initial_ctrl=None,  # Will be built internally
        original_data=original_lower_data,
        mode="fixed-x",
        coupled=False,  # Use uncoupled for speed
        error_function="euclidean",  # Force euclidean for Stage 1 to save time
        objective="msr",
        te_y=te_y_lower,
        te_tangent_vector=te_tangent_vector_lower,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=False,  # For lower surface
        num_control_points_new=config.NUM_CONTROL_POINTS_SINGLE_BEZIER,
    )
    
    if logger_func:
        logger_func("Stage 2: Running unified coupled free-x minmax (softmax) optimization...")
    
    # Stage 2: Run free-x softmax using the fixed-x result as initial guess
    result = optimize_bezier(
        initial_ctrl=initial_upper,  # Use fixed-x result as initial guess
        original_data=original_upper_data,
        mode="free-x",
        coupled=True,
        error_function=error_function,
        objective="softmax",  # Use softmax for minmax optimization
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