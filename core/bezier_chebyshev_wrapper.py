"""
Wrapper functions for Chebyshev LP optimization to match the interface of existing MSR/Softmax functions.
"""

from core.bezier_unified_optimizer import optimize_bezier


def build_bezier_fixed_x_chebyshev(
    original_data, 
    num_control_points_new, 
    is_upper_surface, 
    le_tangent_vector, 
    te_tangent_vector, 
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None
):
    """
    Fixed-x Chebyshev LP optimizer wrapper.
    
    This provides the same interface as build_bezier_fixed_x_msr and build_bezier_fixed_x_softmax,
    but uses the Chebyshev LP solver for optimal minimax approximation.
    """
    te_y = float(original_data[-1, 1])
    
    return optimize_bezier(
        initial_ctrl=None,
        original_data=original_data,
        mode="fixed-x",
        coupled=False,
        error_function=error_function,
        objective="chebyshev",  # Use our new LP solver
        te_y=te_y,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag,
        is_upper_surface=is_upper_surface,
        num_control_points_new=num_control_points_new
    )


def build_coupled_bezier_fixed_x_chebyshev(
    upper_data,
    lower_data,
    num_control_points,
    upper_le_tangent_vector,
    lower_le_tangent_vector,
    upper_te_tangent_vector,
    lower_te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None
):
    """
    Coupled fixed-x Chebyshev LP optimizer wrapper.
    
    NOTE: Currently, the LP solver only supports uncoupled mode.
    This function will fall back to softmax for coupled optimization.
    """
    if logger_func:
        logger_func("WARNING: Chebyshev LP solver does not support coupled mode yet. Falling back to softmax.")
    
    # Import here to avoid circular imports
    from core.coupled_bezier_optimizer import build_coupled_bezier_fixed_x_softmax
    
    return build_coupled_bezier_fixed_x_softmax(
        upper_data=upper_data,
        lower_data=lower_data,
        num_control_points=num_control_points,
        upper_le_tangent_vector=upper_le_tangent_vector,
        lower_le_tangent_vector=lower_le_tangent_vector,
        upper_te_tangent_vector=upper_te_tangent_vector,
        lower_te_tangent_vector=lower_te_tangent_vector,
        regularization_weight=regularization_weight,
        error_function=error_function,
        logger_func=logger_func,
        abort_flag=abort_flag
    )


# Note: Free-x mode is not supported by the LP solver since it requires fixed x-coordinates
# The LP solver will automatically fall back to softmax for free-x mode via the unified optimizer