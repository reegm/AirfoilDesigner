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









