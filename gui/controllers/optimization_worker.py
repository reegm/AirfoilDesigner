"""Optimization worker functions for background processing.

Contains the worker functions that run in separate processes to avoid blocking the GUI.
"""

import numpy as np


def calculate_te_tangent(upper_data, lower_data, te_vector_points):
    """
    Calculate trailing edge tangent vectors for upper and lower surfaces using the last N points.
    Returns (upper_te_tangent_vector, lower_te_tangent_vector)
    """
    def tangent(data, n):
        # Use the last n points to estimate the tangent at the trailing edge
        if n < 2 or len(data) < n:
            n = min(3, len(data))
        pts = data[-n:]
        # Fit a line and return the direction vector (normalized)
        dx = pts[-1, 0] - pts[0, 0]
        dy = pts[-1, 1] - pts[0, 1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            return np.array([1.0, 0.0])
        return np.array([dx, dy]) / norm
    
    upper_te_tangent = tangent(upper_data, te_vector_points)
    lower_te_tangent = tangent(lower_data, te_vector_points)
    return upper_te_tangent, lower_te_tangent


def _generation_worker(args, queue):
    """
    This worker function runs in a separate process to avoid blocking the GUI.
    It operates on the core logic without Qt dependencies.
    """
    import traceback
    import numpy as np
    from core import config
    # Import new optimizer
    from core.bezier_optimizer import (
        build_bezier_fixed_x_msr,
        build_bezier_fixed_x_softmax,
        build_bezier_free_x_msr,
        build_bezier_free_x_softmax,
        build_bezier_staged_uncoupled,
    )
    from core.coupled_bezier_optimizer import (
        build_coupled_bezier_fixed_x_msr,
        build_coupled_bezier_fixed_x_softmax,
        build_coupled_bezier_free_x_msr,
        build_coupled_bezier_free_x_softmax,
        build_coupled_bezier_staged
    )
    from core.error_functions import calculate_single_bezier_fitting_error

    (
        upper_data,
        lower_data,
        regularization_weight,
        g2_flag,
        te_vector_points,
        gui_strategy,
        error_function,
        objective_type,
        enforce_te_tangency,
        abort_flag
    ) = args

    debug_logging_enabled = config.DEBUG_WORKER_LOGGING
    update_plot_enabled = config.UPDATE_PLOT
    last_progress_time = 0
    progress_interval = config.PROGRESS_UPDATE_INTERVAL  # Configurable progress update interval
    
    def worker_logger(message):
        # Always send to GUI text window
        queue.put({"type": "log", "message": message})
        # Only print to terminal if debug logging is enabled
        if debug_logging_enabled:
            print(f"[DEBUG] {message}")
    
    def progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, surface_info=None):
        """Progress callback that sends updates through the queue with rate limiting.
        Always sends updates so the controller can cache best-so-far, regardless of plot updates setting.
        """
        nonlocal last_progress_time
        import time
        current_time = time.time()
        
        # Only send progress updates at most every progress_interval seconds
        if current_time - last_progress_time >= progress_interval:
            last_progress_time = current_time
            
            # Send progress update through queue
            queue.put({
                "type": "progress",
                "iteration": iteration,
                "elapsed": elapsed,
                "val": val,
                "true_max": true_max,
                "best_true_max": best_true_max,
                "best_x": best_x,
                "current_ctrl": current_ctrl,
                "surface_info": surface_info
            })
    
    def combined_logger(*args):
        """Combined logger that handles both simple log messages and progress callbacks"""
        if len(args) == 1 and isinstance(args[0], str):
            # Simple log message - always send to GUI, only print to terminal if debug enabled
            queue.put({"type": "log", "message": args[0]})
            if debug_logging_enabled:
                print(f"[DEBUG] {args[0]}")
        elif len(args) == 6:
            # Progress callback data with 6 arguments (old format)
            iteration, elapsed, val, true_max, best_true_max, best_x = args
            progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, None)
        elif len(args) == 7:
            # Progress callback data with 7 arguments (new format with control points)
            iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
            progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl)
        elif len(args) == 8:
            # Progress callback data with 8 arguments (new format with control points and surface info)
            iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, surface_info = args
            progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, surface_info)
        else:
            # Fallback for unexpected argument patterns - always send to GUI, only print to terminal if debug enabled
            queue.put({"type": "log", "message": f"Unexpected logger call with {len(args)} arguments: {args}"})
            if debug_logging_enabled:
                print(f"[DEBUG] Unexpected logger call with {len(args)} arguments: {args}")

    # Calculate TE tangent vectors only if TE tangency is enforced
    if enforce_te_tangency:
        upper_te_tangent_vector, lower_te_tangent_vector = calculate_te_tangent(upper_data, lower_data, te_vector_points)
    else:
        upper_te_tangent_vector = None
        lower_te_tangent_vector = None

    try:
        # Dispatch based on strategy and g2_flag (coupled/uncoupled)
        if gui_strategy == 'fixed-x' and not g2_flag:
            # Uncoupled fixed-x (implemented)
            worker_logger(f"Starting uncoupled fixed-x {objective_type} optimization...")
            le_tangent_upper = np.array([0.0, 1.0])
            le_tangent_lower = np.array([0.0, -1.0])
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            
            # Create surface-specific loggers for uncoupled operations
            def upper_logger(*args):
                if len(args) == 7:
                    # Add surface info to progress callback
                    iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
                    progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, "upper")
                else:
                    # Pass through other calls unchanged
                    combined_logger(*args)
            
            def lower_logger(*args):
                if len(args) == 7:
                    # Add surface info to progress callback
                    iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
                    progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, "lower")
                else:
                    # Pass through other calls unchanged
                    combined_logger(*args)
            
            if objective_type == 'msr':
                worker_logger("Processing upper surface (fixed-x MSR)...")
                upper_poly = build_bezier_fixed_x_msr(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=upper_logger,
                    abort_flag=abort_flag
                )
                worker_logger("Processing lower surface (fixed-x MSR)...")
                lower_poly = build_bezier_fixed_x_msr(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=lower_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'softmax':
                worker_logger("Processing upper surface (fixed-x softmax)...")
                upper_poly = build_bezier_fixed_x_softmax(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=upper_logger,
                    abort_flag=abort_flag
                )
                worker_logger("Processing lower surface (fixed-x softmax)...")
                lower_poly = build_bezier_fixed_x_softmax(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=lower_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Unknown objective_type: {objective_type}"})
                return
            # Error calculation (for reporting)
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'fixed-x' and g2_flag:
            # Coupled fixed-x (implemented)
            worker_logger(f"Starting coupled fixed-x {objective_type} optimization...")
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            if objective_type == 'msr':
                upper_poly, lower_poly = build_coupled_bezier_fixed_x_msr(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=combined_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'softmax':
                upper_poly, lower_poly = build_coupled_bezier_fixed_x_softmax(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=combined_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Coupled fixed-x objective not yet implemented: {objective_type}"})
                return
            # Error calculation (for reporting)
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'free-x' and not g2_flag:
            # Uncoupled free-x (implemented)
            worker_logger(f"Starting uncoupled free-x {objective_type} optimization...")
            le_tangent_upper = np.array([0.0, 1.0])
            le_tangent_lower = np.array([0.0, -1.0])
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            
            # Create surface-specific loggers for uncoupled operations
            def upper_logger(*args):
                if len(args) == 7:
                    # Add surface info to progress callback
                    iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
                    progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, "upper")
                else:
                    # Pass through other calls unchanged
                    combined_logger(*args)
            
            def lower_logger(*args):
                if len(args) == 7:
                    # Add surface info to progress callback
                    iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
                    progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, "lower")
                else:
                    # Pass through other calls unchanged
                    combined_logger(*args)
            
            if objective_type == 'msr':
                worker_logger("Processing upper surface (free-x MSR)...")
                upper_poly = build_bezier_free_x_msr(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=upper_logger,
                    abort_flag=abort_flag
                )
                worker_logger("Processing lower surface (free-x MSR)...")
                lower_poly = build_bezier_free_x_msr(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=lower_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'softmax':
                worker_logger("Processing upper surface (free-x softmax)...")
                upper_poly = build_bezier_free_x_softmax(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=upper_logger,
                    abort_flag=abort_flag
                )
                worker_logger("Processing lower surface (free-x softmax)...")
                lower_poly = build_bezier_free_x_softmax(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=lower_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Free-x objective not yet implemented: {objective_type}"})
                return
            
            # Error calculation (for reporting)
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'staged' and not g2_flag:
            # Uncoupled staged pipeline (euclidean only)
            worker_logger("Starting uncoupled staged optimization...")
            le_tangent_upper = np.array([0.0, 1.0])
            le_tangent_lower = np.array([0.0, -1.0])
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER

            def upper_logger(*args):
                if len(args) == 7:
                    iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
                    progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, "upper")
                else:
                    combined_logger(*args)

            def lower_logger(*args):
                if len(args) == 7:
                    iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl = args
                    progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl, "lower")
                else:
                    combined_logger(*args)

            # Force euclidean error regardless of UI selection for staged per spec
            worker_logger("Processing upper surface (staged optimization)...")
            upper_poly = build_bezier_staged_uncoupled(
                upper_data,
                num_control_points,
                True,
                le_tangent_upper,
                upper_te_tangent_vector,
                regularization_weight=regularization_weight,
                error_function='euclidean',
                logger_func=upper_logger,
                abort_flag=abort_flag,
            )
            worker_logger("Processing lower surface (staged optimization)...")
            lower_poly = build_bezier_staged_uncoupled(
                lower_data,
                num_control_points,
                False,
                le_tangent_lower,
                lower_te_tangent_vector,
                regularization_weight=regularization_weight,
                error_function='euclidean',
                logger_func=lower_logger,
                abort_flag=abort_flag,
            )

            # Error calculation (for reporting)
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'free-x' and g2_flag:
            # Coupled free-x (implemented)
            worker_logger(f"Starting coupled free-x {objective_type} optimization...")
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            if objective_type == 'msr':
                upper_poly, lower_poly = build_coupled_bezier_free_x_msr(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=combined_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'softmax':
                upper_poly, lower_poly = build_coupled_bezier_free_x_softmax(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=combined_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Coupled free-x objective not yet implemented: {objective_type}"})
                return
            # Error calculation (for reporting)
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'staged' and g2_flag:
            # Coupled staged pipeline (G2 enabled, euclidean only)
            worker_logger("Starting coupled staged optimization...")
            
            # Force euclidean error regardless of UI selection for staged per spec
            upper_poly, lower_poly = build_coupled_bezier_staged(
                upper_data,
                lower_data,
                regularization_weight,
                upper_te_tangent_vector,
                lower_te_tangent_vector,
                error_function='euclidean',
                logger_func=combined_logger,
                abort_flag=abort_flag
            )
            
            # Error calculation (for reporting)
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        else:
            queue.put({"success": False, "error": f"Unknown strategy/g2_flag combination: {gui_strategy}, g2={g2_flag}"})
    except Exception as e:
        queue.put({"success": False, "error": f"Exception in worker: {e}\n{traceback.format_exc()}"}) 