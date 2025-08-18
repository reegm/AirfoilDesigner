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
    upper_te_tangent_vector,
    lower_te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None
):
    """
    Coupled fixed-x Chebyshev LP optimizer wrapper.
    
    This uses the coupled Chebyshev LP solver for optimal minimax approximation
    with G2 continuity constraints at the leading edge.
    """
    from core.chebyshev_lp_optimizer import optimize_coupled_fixed_x_chebyshev_lp
    
    if logger_func:
        logger_func("Using coupled Chebyshev LP optimizer...")
        logger_func(f"Input data shapes - upper: {upper_data.shape}, lower: {lower_data.shape}")
    
    result = optimize_coupled_fixed_x_chebyshev_lp(
        original_upper_data=upper_data,
        original_lower_data=lower_data,
        regularization_weight=regularization_weight,
        te_tangent_vector_upper=upper_te_tangent_vector,
        te_tangent_vector_lower=lower_te_tangent_vector,
        error_function=error_function,
        logger_func=logger_func,
        abort_flag=abort_flag
    )
    
    if logger_func:
        if result is not None:
            upper_ctrl, lower_ctrl = result
            logger_func(f"Wrapper returning result - upper: {upper_ctrl.shape}, lower: {lower_ctrl.shape}")
        else:
            logger_func("Wrapper got None result from optimizer")
    
    return result


# Note: Free-x mode is not supported by the LP solver since it requires fixed x-coordinates
# The LP solver will automatically fall back to softmax for free-x mode via the unified optimizer