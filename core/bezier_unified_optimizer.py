import numpy as np
from core.error_functions import calculate_single_bezier_fitting_error
from core.solver_helpers import smoothness_penalty, get_fixed_inner_x_partition, get_initial_guess_inner_y, build_control_points_with_fixed, minimize_with_debug_with_abort, g2_constraint_factory_fixed_x, leading_edge_curvature
from core import config


def optimize_bezier(
    initial_ctrl,
    original_data,
    *,
    mode="free-x",  # or "fixed-x"
    coupled=False,
    error_function="euclidean",
    objective="softmax",  # or "msr", "minmax"
    te_y=None,
    te_tangent_vector=None,
    regularization_weight=0.0,
    bounds_x=None,
    bounds_y=None,
    logger_func=None,
    abort_flag=None,
    g2_constraint=False,
    lower_data=None,
    lower_initial_ctrl=None,
    lower_te_y=None,
    lower_te_tangent_vector=None,
    lower_bounds_x=None,
    lower_bounds_y=None,
    is_upper_surface=None,  # Required for fixed-x mode
    num_control_points_new=None,  # Required for fixed-x mode
):
    """
    Unified Bezier optimizer for both single and coupled (G2) paths.
    Handles fixed/free-x, softmax/msr/minmax, and constraints.
    Returns optimized control points (single or tuple for coupled).
    """
    # Helper: select objective function
    def get_objective_fn(obj_type):
        if obj_type == "softmax":
            def softmax_obj(residuals):
                alpha = config.SOFTMAX_ALPHA
                abs_res = np.abs(residuals)
                # Prevent overflow by clipping large values
                max_val = np.max(abs_res)
                if max_val > 100:  # Threshold to prevent overflow
                    abs_res = np.clip(abs_res, 0, 100)
                return np.log(np.sum(np.exp(alpha * abs_res))) / alpha
            return softmax_obj
        elif obj_type == "msr":
            return lambda residuals: np.sum(residuals ** 2)  # Sum of squares, not mean
        elif obj_type == "minmax":
            return lambda residuals: np.max(np.abs(residuals))
        else:
            raise ValueError(f"Unknown objective type: {obj_type}")

    objective_fn = get_objective_fn(objective)

    if not coupled:
        # Single surface
        if mode == "fixed-x":
            # Use legacy fixed-x logic exactly
            if is_upper_surface is None or num_control_points_new is None:
                raise ValueError("is_upper_surface and num_control_points_new required for fixed-x mode")
            
            fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
                is_upper_surface, num_control_points_new, original_data, te_tangent_vector, te_y)
            initial_guess_inner_y_full = get_initial_guess_inner_y(original_data, fixed_inner_x_coords)
            initial_guess_inner_y = initial_guess_inner_y_full[free_indices]
            
            def error_func(ctrl):
                if error_function == "orthogonal":
                    return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="orthogonal", return_max_error=False)
                else:
                    return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="euclidean", return_max_error=False)
            
            # Track best configuration during optimization (matching legacy behavior)
            best_max_error = float("inf")
            best_vars = None
            
            def obj(variables_y):
                nonlocal best_max_error, best_vars
                ctrl = build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
                errors = error_func(ctrl)
                if isinstance(errors, tuple):
                    errors = errors[0]
                
                # Calculate true max error for tracking (matching legacy behavior)
                residuals = calculate_single_bezier_fitting_error(ctrl, original_data, error_function=error_function, return_max_error=False, return_all=True)[0]
                true_max_error = np.max(np.abs(residuals))
                
                # Update best configuration if we found a better max error (matching legacy behavior)
                if true_max_error < best_max_error:
                    best_max_error = true_max_error
                    best_vars = np.copy(variables_y)
                
                return errors + regularization_weight * smoothness_penalty(ctrl)
            
            # Always add build_ctrl function for progress updates (regardless of debug flag)
            obj.__build_ctrl__ = lambda variables_y: build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
            
            # Attach debug functions for logging (only if debug logging is enabled)
            if config.DEBUG_WORKER_LOGGING:
                def get_residuals_with_debug(variables_y):
                    ctrl = build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
                    return error_func(ctrl)
                
                obj.__get_residuals__ = get_residuals_with_debug
                # Use the same max error calculation as in the obj function
                obj.__get_max_error__ = lambda variables_y: np.max(np.abs(calculate_single_bezier_fitting_error(
                    build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values),
                    original_data, error_function=error_function, return_max_error=False, return_all=True)[0]))
            
            constraints = []
            # Only apply success threshold for free-x minmax/softmax objectives, not for fixed-x or MSR
            # Fixed-x should run to completion like MSR, while free-x can use the threshold
            success_threshold = config.MAX_ERROR_THRESHOLD if (mode == "free-x" and objective in ["softmax", "minmax"]) else None
            
            result, iteration_data = minimize_with_debug_with_abort(
                fun=obj,
                x0=initial_guess_inner_y,
                method="SLSQP",
                constraints=constraints,
                options=config.SLSQP_OPTIONS,
                success_threshold=success_threshold,
                abort_flag=abort_flag,
                progress_callback=logger_func
            )
            
            if logger_func:
                if result.success:
                    logger_func("Unified optimization succeeded.")
                else:
                    logger_func(f"Unified optimization failed: {result.message}")
            
            # Use the best configuration found during optimization (matching legacy behavior)
            if best_vars is not None:
                final_inner_y = best_vars
                if logger_func:
                    logger_func(f"Using best configuration found (max error: {best_max_error:.6e})")
            else:
                final_inner_y = result.x
            
            final_ctrl = build_control_points_with_fixed(final_inner_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
            return final_ctrl
            
        elif mode == "free-x":
            # Free-x logic (existing implementation)
            n = len(initial_ctrl)
            x0 = initial_ctrl[2:-1, 0]
            y0 = initial_ctrl[1:-1, 1]
            vars0 = np.concatenate([x0, y0])
            n_x_vars = n - 3
            n_y_vars = n - 2
            def build_ctrl(xy):
                x_inner = xy[:n_x_vars]
                y_inner = xy[n_x_vars:]
                ctrl = np.zeros((n, 2))
                ctrl[0] = [0.0, 0.0]
                ctrl[1] = [0.0, y_inner[0]]
                ctrl[2:-1, 0] = x_inner
                ctrl[2:-1, 1] = y_inner[1:]
                ctrl[-1] = [1.0, te_y]
                return ctrl
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if mode == "free-x":
            def residuals_fn(ctrl):
                residuals, _, _ = calculate_single_bezier_fitting_error(
                    ctrl, original_data, error_function=error_function, return_max_error=False, return_all=True
                )
                return residuals

            # Track best configuration during optimization (matching legacy behavior)
            best_max_error = float("inf")
            best_vars = None
            
            def full_obj(vars):
                nonlocal best_max_error, best_vars
                ctrl = build_ctrl(vars)
                residuals = residuals_fn(ctrl)
                
                # Calculate true max error for tracking (matching legacy behavior)
                true_max_error = np.max(np.abs(residuals))
                
                # Update best configuration if we found a better max error (matching legacy behavior)
                if true_max_error < best_max_error:
                    best_max_error = true_max_error
                    best_vars = np.copy(vars)
                
                # Calculate objective value
                obj_val = objective_fn(residuals)
                
                # Add smoothness penalty
                smoothness_penalty_val = smoothness_penalty(ctrl) if regularization_weight > 0 else 0.0
                
                # Add position penalty (matching legacy behavior)
                # When smoothness is 0, minimize position penalty to focus on pure fitting
                # When smoothness > 0, use moderate position penalty to prevent extreme movements
                position_weight = 100 if regularization_weight == 0.0 else 1
                position_term = position_weight  # * position_penalty (legacy has this commented out)
                smoothness_term = regularization_weight * smoothness_penalty_val
                total_obj = obj_val + position_term + smoothness_term
                
                return total_obj

            constraints = []
            if te_tangent_vector is not None and te_y is not None:
                def make_te_vector_constraint(te_tangent_vector, te_y, n_x_vars, n_y_vars):
                    tx_te, ty_te = te_tangent_vector
                    norm = np.sqrt(tx_te ** 2 + ty_te ** 2) or 1e-12
                    def constraint(vars):
                        x_nm1 = vars[n_x_vars - 1]
                        y_nm1 = vars[n_x_vars + n_y_vars - 1]
                        dx = 1.0 - x_nm1
                        dy = te_y - y_nm1
                        return (dx * ty_te - dy * tx_te) / norm
                    return constraint
                constraints.append({
                    "type": "eq",
                    "fun": make_te_vector_constraint(te_tangent_vector, te_y, n_x_vars, n_y_vars)
                })

            bounds_x = bounds_x or [(0.0, 1.0)] * (n - 3)
            bounds_y = bounds_y or [(-1.0, 1.0)] * (n - 2)
            bounds = bounds_x + bounds_y
            x0 = vars0

            # Always add build_ctrl function for progress updates (regardless of debug flag)
            full_obj.__build_ctrl__ = build_ctrl
            
            # Attach debug functions for logging (only if debug logging is enabled)
            if config.DEBUG_WORKER_LOGGING:
                def get_residuals_with_debug(xy):
                    return residuals_fn(build_ctrl(xy))
                
                full_obj.__get_residuals__ = get_residuals_with_debug
                full_obj.__get_max_error__ = lambda xy: np.max(np.abs(residuals_fn(build_ctrl(xy))))
                full_obj.__te_y__ = te_y
                full_obj.__original_data__ = original_data

            # Only apply success threshold for free-x minmax/softmax objectives, not for fixed-x or MSR
            # Fixed-x should run to completion like MSR, while free-x can use the threshold
            success_threshold = config.MAX_ERROR_THRESHOLD if (mode == "free-x" and objective in ["softmax", "minmax"]) else None
            

            result, trace = minimize_with_debug_with_abort(
                fun=full_obj,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=config.SLSQP_OPTIONS,
                success_threshold=success_threshold,
                abort_flag=abort_flag,
                progress_callback=logger_func
            )
            if logger_func:
                if result.success:
                    logger_func("Unified optimization succeeded.")
                else:
                    logger_func(f"Unified optimization failed: {result.message}")
            # Use the best configuration found during optimization (matching legacy behavior)
            if best_vars is not None:
                final_ctrl = build_ctrl(best_vars)
                if logger_func:
                    logger_func(f"Using best configuration found (max error: {best_max_error:.6e})")
            else:
                if result.success:
                    final_ctrl = build_ctrl(result.x)
                else:
                    final_ctrl = initial_ctrl
            return final_ctrl

    else:
        # Coupled (G2) path
        if mode == "fixed-x":
            # Coupled fixed-x optimization
            if is_upper_surface is None or num_control_points_new is None:
                raise ValueError("is_upper_surface and num_control_points_new required for coupled fixed-x mode")
            
            # Build initial control points for both surfaces
            
            # Upper surface
            inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u = get_fixed_inner_x_partition(
                True, num_control_points_new, original_data, te_tangent_vector, te_y)
            init_y_upper_full = get_initial_guess_inner_y(original_data, inner_x_upper)
            init_y_upper = init_y_upper_full[free_idx_u]
            
            # Lower surface
            inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l = get_fixed_inner_x_partition(
                False, num_control_points_new, lower_data, lower_te_tangent_vector, lower_te_y)
            init_y_lower_full = get_initial_guess_inner_y(lower_data, inner_x_lower)
            init_y_lower = init_y_lower_full[free_idx_l]
            
            # Combine initial guesses
            vars0 = np.concatenate([init_y_upper, init_y_lower])
            n_free_u = len(free_idx_u)
            n_free_l = len(free_idx_l)
            
            def error_func_upper(ctrl):
                if error_function == "orthogonal":
                    return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="orthogonal", return_max_error=False)
                else:
                    return calculate_single_bezier_fitting_error(ctrl, original_data, error_function="euclidean", return_max_error=False)
            
            def error_func_lower(ctrl):
                if error_function == "orthogonal":
                    return calculate_single_bezier_fitting_error(ctrl, lower_data, error_function="orthogonal", return_max_error=False)
                else:
                    return calculate_single_bezier_fitting_error(ctrl, lower_data, error_function="euclidean", return_max_error=False)
            
            def assemble_polygons(var_y):
                y_u = var_y[:n_free_u]
                y_l = var_y[n_free_u:]
                ctrl_upper = build_control_points_with_fixed(y_u, inner_x_upper, te_y, free_idx_u, fixed_idx_u, fixed_y_u)
                ctrl_lower = build_control_points_with_fixed(y_l, inner_x_lower, lower_te_y, free_idx_l, fixed_idx_l, fixed_y_l)
                return ctrl_upper, ctrl_lower
            
            # Track best configuration during optimization (matching legacy behavior)
            best_max_error = float("inf")
            best_vars = None
            
            def obj(var_y):
                nonlocal best_max_error, best_vars
                ctrl_u, ctrl_l = assemble_polygons(var_y)
                err_u = error_func_upper(ctrl_u)
                err_l = error_func_lower(ctrl_l)
                if isinstance(err_u, tuple): err_u = err_u[0]
                if isinstance(err_l, tuple): err_l = err_l[0]
                
                # Calculate true max error for tracking (matching legacy behavior)
                residuals_u = calculate_single_bezier_fitting_error(ctrl_u, original_data, error_function=error_function, return_max_error=False, return_all=True)[0]
                residuals_l = calculate_single_bezier_fitting_error(ctrl_l, lower_data, error_function=error_function, return_max_error=False, return_all=True)[0]
                true_max_error = max(np.max(np.abs(residuals_u)), np.max(np.abs(residuals_l)))
                
                # Update best configuration if we found a better max error (matching legacy behavior)
                if true_max_error < best_max_error:
                    best_max_error = true_max_error
                    best_vars = np.copy(var_y)
                
                smooth = smoothness_penalty(ctrl_u) + smoothness_penalty(ctrl_l)
                return err_u + err_l + regularization_weight * smooth
            
            # Always add build_ctrl function for progress updates (regardless of debug flag)
            obj.__build_ctrl__ = assemble_polygons
            
            # Attach debug functions for logging (only if debug logging is enabled)
            if config.DEBUG_WORKER_LOGGING:
                def get_residuals_with_debug(var_y):
                    ctrl_u, ctrl_l = assemble_polygons(var_y)
                    err_u = error_func_upper(ctrl_u)
                    err_l = error_func_lower(ctrl_l)
                    if isinstance(err_u, tuple): err_u = err_u[0]
                    if isinstance(err_l, tuple): err_l = err_l[0]
                    # For MSR optimization, we return scalar errors, not residual arrays
                    return np.array([err_u, err_l])
                
                obj.__get_residuals__ = get_residuals_with_debug
                # Use the same max error calculation as in the obj function
                obj.__get_max_error__ = lambda var_y: max(
                    np.max(np.abs(calculate_single_bezier_fitting_error(
                        assemble_polygons(var_y)[0], original_data, 
                        error_function=error_function, return_max_error=False, return_all=True)[0])),
                    np.max(np.abs(calculate_single_bezier_fitting_error(
                        assemble_polygons(var_y)[1], lower_data, 
                        error_function=error_function, return_max_error=False, return_all=True)[0]))
                )
            
            # G2 constraint

            constraints = []
            constraints.append({"type": "eq", "fun": g2_constraint_factory_fixed_x(
                inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u, te_y,
                inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l, lower_te_y,
                original_data, lower_data)})
            result, _ = minimize_with_debug_with_abort(
                fun=obj,
                x0=vars0,
                method='SLSQP',
                constraints=constraints,
                options=config.SLSQP_OPTIONS,
                success_threshold=config.MAX_ERROR_THRESHOLD if (mode == "free-x" and objective in ["softmax", "minmax"]) else None,  # Only use threshold for free-x minmax objectives
                abort_flag=abort_flag,
                progress_callback=logger_func
            )
            
            if logger_func:
                if result.success:
                    logger_func("Unified coupled fixed-x MSR optimization succeeded.")
                else:
                    logger_func(f"Unified coupled fixed-x MSR optimization failed: {result.message}")
            
            if not result.success and logger_func:
                logger_func(f"Unified coupled fixed-x MSR optimization failed. Using best result found. Reason: {result.message}")
            var_y_final = result.x
            
            # Use best configuration if available (matching legacy behavior)
            if best_vars is not None:
                var_y_final = best_vars
                if logger_func:
                    logger_func(f"Using best configuration found (max error: {best_max_error:.6e})")
            
            ctrl_upper_final, ctrl_lower_final = assemble_polygons(var_y_final)
            return ctrl_upper_final, ctrl_lower_final
        
        else:
            # Coupled free-x optimization
            if initial_ctrl is None or lower_initial_ctrl is None:
                raise ValueError("initial_ctrl and lower_initial_ctrl required for coupled free-x optimization")
            
            n_upper = len(initial_ctrl)
            n_lower = len(lower_initial_ctrl)
        n_x_vars_upper = n_upper - 3
        n_y_vars_upper = n_upper - 2
        n_x_vars_lower = n_lower - 3
        n_y_vars_lower = n_lower - 2
        x0_upper = initial_ctrl[2:-1, 0]
        y0_upper = initial_ctrl[1:-1, 1]
        x0_lower = lower_initial_ctrl[2:-1, 0]
        y0_lower = lower_initial_ctrl[1:-1, 1]
        vars0 = np.concatenate([x0_upper, y0_upper, x0_lower, y0_lower])
        def build_ctrl_upper(xy):
            x_inner = xy[:n_x_vars_upper]
            y_inner = xy[n_x_vars_upper:n_x_vars_upper + n_y_vars_upper]
            ctrl = np.zeros((n_upper, 2))
            ctrl[0] = [0.0, 0.0]
            ctrl[1] = [0.0, y_inner[0]]
            ctrl[2:-1, 0] = x_inner
            ctrl[2:-1, 1] = y_inner[1:]
            ctrl[-1] = [1.0, te_y]
            return ctrl
        def build_ctrl_lower(xy):
            start_idx = n_x_vars_upper + n_y_vars_upper
            x_inner = xy[start_idx:start_idx + n_x_vars_lower]
            y_inner = xy[start_idx + n_x_vars_lower:]
            ctrl = np.zeros((n_lower, 2))
            ctrl[0] = [0.0, 0.0]
            ctrl[1] = [0.0, y_inner[0]]
            ctrl[2:-1, 0] = x_inner
            ctrl[2:-1, 1] = y_inner[1:]
            ctrl[-1] = [1.0, lower_te_y]
            return ctrl
        def residuals_fn_upper(ctrl):
            residuals, _, _ = calculate_single_bezier_fitting_error(
                ctrl, original_data, error_function=error_function, return_max_error=False, return_all=True
            )
            return residuals
        def residuals_fn_lower(ctrl):
            residuals, _, _ = calculate_single_bezier_fitting_error(
                ctrl, lower_data, error_function=error_function, return_max_error=False, return_all=True
            )
            return residuals
        best_obj = float("inf")
        best_xy = None
        def full_obj(xy):
            nonlocal best_obj, best_xy
            ctrl_upper = build_ctrl_upper(xy)
            ctrl_lower = build_ctrl_lower(xy)
            residuals_upper = residuals_fn_upper(ctrl_upper)
            residuals_lower = residuals_fn_lower(ctrl_lower)
            all_residuals = np.concatenate([residuals_upper, residuals_lower])
            obj_val = objective_fn(all_residuals)
            if obj_val < best_obj:
                best_obj = obj_val
                best_xy = np.copy(xy)
            smoothness_penalty_val = (smoothness_penalty(ctrl_upper) + smoothness_penalty(ctrl_lower)) if regularization_weight > 0 else 0.0
            total_obj = obj_val + regularization_weight * smoothness_penalty_val
            return total_obj
        constraints = []
        if te_tangent_vector is not None and te_y is not None:
            def make_te_vector_constraint_upper(te_tangent_vector, te_y, n_x_vars, n_y_vars):
                tx_te, ty_te = te_tangent_vector
                norm = np.sqrt(tx_te ** 2 + ty_te ** 2) or 1e-12
                def constraint(xy):
                    x_nm1 = xy[n_x_vars - 1]
                    y_nm1 = xy[n_x_vars + n_y_vars - 1]
                    dx = 1.0 - x_nm1
                    dy = te_y - y_nm1
                    return (dx * ty_te - dy * tx_te) / norm
                return constraint
            constraints.append({
                "type": "eq",
                "fun": make_te_vector_constraint_upper(te_tangent_vector, te_y, n_x_vars_upper, n_y_vars_upper)
            })
        if lower_te_tangent_vector is not None and lower_te_y is not None:
            def make_te_vector_constraint_lower(te_tangent_vector, te_y, n_x_vars_upper, n_y_vars_upper, n_x_vars_lower, n_y_vars_lower):
                tx_te, ty_te = te_tangent_vector
                norm = np.sqrt(tx_te ** 2 + ty_te ** 2) or 1e-12
                def constraint(xy):
                    start_idx = n_x_vars_upper + n_y_vars_upper
                    x_nm1 = xy[start_idx + n_x_vars_lower - 1]
                    y_nm1 = xy[start_idx + n_x_vars_lower + n_y_vars_lower - 1]
                    dx = 1.0 - x_nm1
                    dy = te_y - y_nm1
                    return (dx * ty_te - dy * tx_te) / norm
                return constraint
            constraints.append({
                "type": "eq",
                "fun": make_te_vector_constraint_lower(lower_te_tangent_vector, lower_te_y, n_x_vars_upper, n_y_vars_upper, n_x_vars_lower, n_y_vars_lower)
            })
        if g2_constraint:
            def g2_constraint_fn(xy):
                ctrl_upper = build_ctrl_upper(xy)
                ctrl_lower = build_ctrl_lower(xy)
    
                return leading_edge_curvature(ctrl_upper) + leading_edge_curvature(ctrl_lower)
            constraints.append({
                "type": "eq",
                "fun": g2_constraint_fn
            })
        bounds_x_upper = bounds_x or [(0.0, 1.0)] * n_x_vars_upper
        bounds_y_upper = bounds_y or [(-1.0, 1.0)] * n_y_vars_upper
        bounds_x_lower = lower_bounds_x or [(0.0, 1.0)] * n_x_vars_lower
        bounds_y_lower = lower_bounds_y or [(-1.0, 1.0)] * n_y_vars_lower
        bounds = bounds_x_upper + bounds_y_upper + bounds_x_lower + bounds_y_lower
        # Always add build_ctrl function for progress updates
        def build_ctrl_combined(xy):
            return build_ctrl_upper(xy), build_ctrl_lower(xy)
        
        # Always add build_ctrl function for progress updates (regardless of debug flag)
        objective_function = full_obj
        objective_function.__build_ctrl__ = build_ctrl_combined
        
        # Attach debug functions for logging (only if debug logging is enabled)
        if config.DEBUG_WORKER_LOGGING and logger_func:
            def get_residuals_with_debug(xy):
                ctrl_upper = build_ctrl_upper(xy)
                ctrl_lower = build_ctrl_lower(xy)
                residuals_upper = residuals_fn_upper(ctrl_upper)
                residuals_lower = residuals_fn_lower(ctrl_lower)
                return np.concatenate([residuals_upper, residuals_lower])
            
            def get_max_error_with_debug(xy):
                residuals = get_residuals_with_debug(xy)
                return np.max(np.abs(residuals))
            
            # Create a wrapper function that has the debug attributes and preserves best tracking
            def full_obj_with_debug(xy):
                nonlocal best_obj, best_xy
                result = full_obj(xy)
                return result
            
            full_obj_with_debug.__get_residuals__ = get_residuals_with_debug
            full_obj_with_debug.__get_max_error__ = get_max_error_with_debug
            full_obj_with_debug.__build_ctrl__ = build_ctrl_combined  # Use the same build_ctrl function
            
            # Use the wrapper for optimization
            objective_function = full_obj_with_debug
        
        
        result, trace = minimize_with_debug_with_abort(
            fun=objective_function,
            x0=vars0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=config.SLSQP_OPTIONS,
            success_threshold=config.MAX_ERROR_THRESHOLD if (mode == "free-x" and objective in ["softmax", "minmax"]) else None,
            abort_flag=abort_flag,
            progress_callback=logger_func
        )
        if logger_func:
            if result.success:
                logger_func("Unified coupled optimization succeeded.")
            else:
                logger_func(f"Unified coupled optimization failed: {result.message}")
        if best_xy is not None:
            final_ctrl_upper = build_ctrl_upper(best_xy)
            final_ctrl_lower = build_ctrl_lower(best_xy)
            if logger_func:
                logger_func(f"Using best configuration found (obj: {best_obj:.6e})")
        else:
            if result.success:
                final_ctrl_upper = build_ctrl_upper(result.x)
                final_ctrl_lower = build_ctrl_lower(result.x)
            else:
                final_ctrl_upper = initial_ctrl
                final_ctrl_lower = lower_initial_ctrl
        return final_ctrl_upper, final_ctrl_lower