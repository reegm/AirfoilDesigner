import time
import numpy as np
from scipy.optimize import minimize
from utils.control_point_utils import get_paper_fixed_x_coords
from utils.bezier_utils import leading_edge_curvature
from core import config

def minimize_with_debug_with_abort(fun, x0, args=(), method="SLSQP", jac=None, bounds=None, constraints=(), options=None, abort_flag=None, success_threshold=None, progress_callback=None):
    """
    Standard optimization function with debug logging and abort capability.
    This is the unified function that all optimizers should use.
    
    Args:
        progress_callback: Optional callback function(iteration, elapsed, val, true_max, best_true_max, best_x) 
                          called on each iteration for progress updates
    """
    from scipy.optimize import minimize, OptimizeResult

    iteration_data = []
    best_error = float("inf")
    best_true_max = float("inf")
    best_x = None
    no_improve_counter = 0
    plateau_threshold = config.PLATEAU_THRESHOLD
    plateau_patience = config.PLATEAU_PATIENCE
    start_time = time.time()

    get_residuals = getattr(fun, "__get_residuals__", None)

    def detect_stall(current_iter):
        if no_improve_counter >= plateau_patience:
            print("WARNING: Optimizer appears to be stalled.")
            # Abort early on plateau and return best-so-far from the exception handler
            raise EarlyStopException("plateau detected")
        # Manual maxiter enforcement since scipy's SLSQP doesn't always respect it
        if options and current_iter >= options.get('maxiter', 10000):
            print(f"WARNING: Manual maxiter limit reached: {current_iter}")
            raise EarlyStopException(f"Manual maxiter limit reached: {current_iter}")

    class EarlyStopException(Exception):
        pass

    def wrapped_fun(x):
        nonlocal best_error, best_x, best_true_max, no_improve_counter
        iteration = len(iteration_data)
        elapsed = time.time() - start_time

        true_max = None
        alpha = config.SOFTMAX_ALPHA
        current_ctrl = None
        

        try:
            if get_residuals is not None:
                # Get residuals for logging
                residuals = get_residuals(x)
                abs_res = np.abs(residuals)
                true_max = np.max(abs_res)
                softmax_val = np.log(np.sum(np.exp(alpha * abs_res))) / alpha
                
                # Also call the full objective function to include smoothness penalty
                val = fun(x)
                print(f"Iter {iteration:03d} | t = {elapsed:6.2f}s | softmax = {softmax_val:.6e} | max = {true_max:.6e} | best max: = {best_true_max:.6e}")
            else:
                val = fun(x)
                true_max = None
                # print(f"Iter {iteration:03d} | t = {elapsed:6.2f}s | error = {val:.6e}")
            
            # Get current control points if available
            get_ctrl = getattr(fun, "__build_ctrl__", None)
            if get_ctrl is not None:
                current_ctrl = get_ctrl(x)
        except Exception as e:
            val = float("inf")
            true_max = None
            current_ctrl = None
            print(f"Iter {iteration:03d} | t = {elapsed:6.2f}s | error = inf | (evaluation failed: {e})")

        iteration_data.append((iteration, elapsed, val))

        # Update best configuration if either softmax improves OR true_max improves
        should_update = False
        if val + plateau_threshold < best_error:
            best_error = val
            should_update = True
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        # Also update if we found a better true_max (even if softmax didn't improve)
        # Use __get_max_error__ function if available for more accurate tracking
        get_max_error = getattr(fun, "__get_max_error__", None)
        current_max_error = None
        if get_max_error is not None:
            current_max_error = get_max_error(x)
        elif true_max is not None:
            current_max_error = true_max
            
        if current_max_error is not None and current_max_error < best_true_max:
            best_true_max = current_max_error
            should_update = True
            no_improve_counter = 0  # Reset counter when we find a better true_max
            print(f"  -> New best true_max: {current_max_error:.6e}")
            
        if should_update:
            best_x = np.copy(x)

        # Call progress callback if provided
        if progress_callback is not None:
            try:
                progress_callback(iteration, elapsed, val, true_max, best_true_max, best_x, current_ctrl)
            except Exception as e:
                print(f"Progress callback failed: {e}")

        detect_stall(iteration)

        # Check for abort flag periodically
        if abort_flag is not None and abort_flag.value:
            print("Optimization aborted by user.")
            raise EarlyStopException("Optimization aborted by user")

        # Check for success threshold (only if it's provided)
        # Use __get_max_error__ function if available for more accurate threshold comparison
        get_max_error = getattr(fun, "__get_max_error__", None)
        if success_threshold is not None:
            if get_max_error is not None:
                # Use the dedicated max error function for threshold comparison
                threshold_max_error = get_max_error(x)
                if threshold_max_error < success_threshold:
                    raise EarlyStopException()
            elif true_max is not None:
                # Fallback to using the raw residuals max
                if true_max < success_threshold:
                    raise EarlyStopException()

        return val

    try:
        result = minimize(
            wrapped_fun,
            x0,
            args=args,
            method=method,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=options
        )
    except EarlyStopException as e:
        msg = str(e).lower()
        if "aborted by user" in msg:
            result = OptimizeResult(
                x=best_x,
                fun=best_true_max,
                success=True,
                message="Optimization aborted by user - returning best result found",
                nit=len(iteration_data)
            )
        elif "plateau" in msg:
            result = OptimizeResult(
                x=best_x,
                fun=best_true_max,
                success=True,
                message="Early stop: plateau detected",
                nit=len(iteration_data)
            )
        elif "maxiter" in msg:
            result = OptimizeResult(
                x=best_x,
                fun=best_true_max,
                success=True,
                message=str(e),
                nit=len(iteration_data)
            )
        else:
            result = OptimizeResult(
                x=best_x,
                fun=best_true_max,
                success=True,
                message="Early stop",
                nit=len(iteration_data)
            )

    # Note: plateau is handled inside the loop via EarlyStopException now

    # If debug logging is enabled, print final control point configuration(s) to the terminal
    try:
        if config.DEBUG_WORKER_LOGGING:
            print("\nFinal result:")
            print(f"  Success: {result.success}")
            print(f"  Status:  {result.message}")
        if config.DEBUG_WORKER_LOGGING:
            get_ctrl = getattr(fun, "__build_ctrl__", None)
            x_vec = result.x if result is not None and getattr(result, 'x', None) is not None else best_x
            if get_ctrl is not None and x_vec is not None:
                final_ctrl = get_ctrl(x_vec)
                import numpy as _np
                _np.set_printoptions(precision=6, suppress=True)
                if isinstance(final_ctrl, tuple) and len(final_ctrl) == 2:
                    upper_ctrl, lower_ctrl = final_ctrl
                    print("  Final control points (upper):")
                    print(upper_ctrl)
                    print("  Final control points (lower):")
                    print(lower_ctrl)
                else:
                    print("  Final control points:")
                    print(final_ctrl)
    except Exception:
        # Avoid interrupting program flow if debug printing fails
        pass

    return result, iteration_data

def get_fixed_inner_x_coords(is_upper_surface, num_control_points):
    """
    Returns the fixed inner x-coordinates for the given surface and number of control points.
    """
    paper_fixed_x_coords = get_paper_fixed_x_coords(is_upper_surface)
    if num_control_points != len(paper_fixed_x_coords):
        num_control_points = len(paper_fixed_x_coords)
    return paper_fixed_x_coords[1:-1]

def get_initial_guess_inner_y(original_data, fixed_inner_x_coords):
    """
    Returns the initial guess for the inner y-coordinates by interpolating the original data at the fixed x-coordinates.
    """
    return np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])

def get_fixed_inner_x_partition(is_upper_surface, num_control_points, original_data, te_tangent_vector, te_y=None):
    """
    Returns (fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values)
    For fixed-x: the last inner control point (pre-trailing-edge) is fixed.
    The y-value is set so the vector to the TE matches te_tangent_vector.
    """
    paper_fixed_x_coords = get_paper_fixed_x_coords(is_upper_surface)
    if num_control_points != len(paper_fixed_x_coords):
        num_control_points = len(paper_fixed_x_coords)
    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]
    n = len(fixed_inner_x_coords)
    free_indices = list(range(n-1))
    fixed_indices = [n-1]
    # Compute y for pre-TE so that (y_TE - y_n-1)/(x_TE - x_n-1) = ty_TE/tx_TE
    x_n_minus_1 = fixed_inner_x_coords[-1]
    if te_y is None:
        te_y = float(original_data[-1, 1])
    x_te = 1.0
    tx_te, ty_te = te_tangent_vector
    if abs(tx_te) < 1e-12:
        # Avoid division by zero, fallback to interpolation
        y_n_minus_1 = np.interp(x_n_minus_1, original_data[:, 0], original_data[:, 1])
    else:
        y_n_minus_1 = te_y - (x_te - x_n_minus_1) * (ty_te / tx_te)
    fixed_y_values = [y_n_minus_1]
    return fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values



def get_unified_peak_curvature_fixed_x_partition(upper_data, lower_data, num_control_points, te_tangent_vector, te_y=None, search_region=0.1):
    """
    Returns unified peak curvature-based fixed-x partitioning for both surfaces.
    Uses a single split point determined by the surface with tighter curvature.
    
    Args:
        upper_data (np.ndarray): Upper surface data (may be reorganized)
        lower_data (np.ndarray): Lower surface data (may be reorganized)
        num_control_points (int): Number of control points
        te_tangent_vector (tuple): Trailing edge tangent vector
        te_y (float): Trailing edge y-coordinate
        search_region (float): Fraction of chord to search for peak curvature
        
    Returns:
        tuple: (upper_fixed_inner_x_coords, lower_fixed_inner_x_coords, upper_free_indices, lower_free_indices, 
                upper_fixed_indices, lower_fixed_indices, upper_fixed_y_values, lower_fixed_y_values, split_info)
    """
    from utils.bezier_utils import find_optimal_split_point_for_both_surfaces
    
    # Find the optimal split point for both surfaces
    split_x, split_y, upper_tangent, lower_tangent, tighter_surface_is_upper = find_optimal_split_point_for_both_surfaces(
        upper_data, lower_data, search_region
    )
    
    # Get the paper fixed x-coordinates
    upper_paper_fixed_x_coords = get_paper_fixed_x_coords(True)
    lower_paper_fixed_x_coords = get_paper_fixed_x_coords(False)
    
    if num_control_points != len(upper_paper_fixed_x_coords):
        num_control_points = len(upper_paper_fixed_x_coords)
    
    # Create new fixed x-coordinates with the split point
    upper_fixed_inner_x_coords = upper_paper_fixed_x_coords[1:-1].copy()
    lower_fixed_inner_x_coords = lower_paper_fixed_x_coords[1:-1].copy()
    
    # Replace the first inner point with split point for both surfaces
    upper_fixed_inner_x_coords[0] = split_x
    lower_fixed_inner_x_coords[0] = split_x
    

    
    # Set up partitioning for both surfaces
    n_upper = len(upper_fixed_inner_x_coords)
    n_lower = len(lower_fixed_inner_x_coords)
    
    # Free indices: all except first (split point) and last (pre-TE)
    # Note: index 1 (second control point) is now included in free indices
    upper_free_indices = list(range(1, n_upper-1))
    lower_free_indices = list(range(1, n_lower-1))
    
    # Fixed indices: first (split point) and last (pre-TE)
    upper_fixed_indices = [0, n_upper-1]
    lower_fixed_indices = [0, n_lower-1]
    
    # Fixed y-values: split point y and pre-TE y for both surfaces
    if te_y is None:
        te_y = float(upper_data[-1, 1])  # Use upper surface TE y
    
    # Pre-TE y calculation for upper surface
    upper_x_n_minus_1 = upper_fixed_inner_x_coords[-1]
    x_te = 1.0
    tx_te, ty_te = te_tangent_vector
    if abs(tx_te) < 1e-12:
        upper_y_n_minus_1 = np.interp(upper_x_n_minus_1, upper_data[:, 0], upper_data[:, 1])
    else:
        upper_y_n_minus_1 = te_y - (x_te - upper_x_n_minus_1) * (ty_te / tx_te)
    
    # Pre-TE y calculation for lower surface
    lower_x_n_minus_1 = lower_fixed_inner_x_coords[-1]
    if abs(tx_te) < 1e-12:
        lower_y_n_minus_1 = np.interp(lower_x_n_minus_1, lower_data[:, 0], lower_data[:, 1])
    else:
        lower_y_n_minus_1 = te_y - (x_te - lower_x_n_minus_1) * (ty_te / tx_te)
    
    upper_fixed_y_values = [split_y, upper_y_n_minus_1]
    lower_fixed_y_values = [split_y, lower_y_n_minus_1]
    
    # Split info for debugging/logging
    split_info = {
        'split_x': split_x,
        'split_y': split_y,
        'upper_tangent': upper_tangent,
        'lower_tangent': lower_tangent,
        'tighter_surface_is_upper': tighter_surface_is_upper
    }
    
    return (upper_fixed_inner_x_coords, lower_fixed_inner_x_coords, 
            upper_free_indices, lower_free_indices,
            upper_fixed_indices, lower_fixed_indices,
            upper_fixed_y_values, lower_fixed_y_values, split_info)

def build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values):
    """
    Assemble full control points array, inserting fixed y-values at fixed_indices.
    """
    n = len(fixed_inner_x_coords)
    control_points = np.zeros((n + 2, 2))
    control_points[0] = np.array([0.0, 0.0])
    control_points[1:-1, 0] = fixed_inner_x_coords
    y_vals = np.zeros(n)
    y_vals[free_indices] = variables_y
    for idx, y in zip(fixed_indices, fixed_y_values):
        y_vals[idx] = y
    control_points[1:-1, 1] = y_vals
    control_points[-1] = np.array([1.0, te_y])
    return control_points

def build_control_points_with_peak_curvature_split(variables_y, fixed_inner_x_coords, split_x, split_y, split_tangent, te_y, free_indices, fixed_indices, fixed_y_values):
    """
    Assemble full control points array with peak curvature split point.
    The second control point is optimized but constrained to follow the tangent direction.
    """
    n = len(fixed_inner_x_coords)
    control_points = np.zeros((n + 2, 2))
    
    # Set the split point as the first control point
    control_points[0] = np.array([split_x, split_y])
    
    # Set inner control points
    control_points[1:-1, 0] = fixed_inner_x_coords
    y_vals = np.zeros(n)
    y_vals[free_indices] = variables_y
    for idx, y in zip(fixed_indices, fixed_y_values):
        y_vals[idx] = y
    control_points[1:-1, 1] = y_vals
    
    # For the second control point, we want it to be optimized but constrained to the tangent
    # We'll use the optimization variable for the y-coordinate, but calculate x based on tangent
    if 1 in free_indices:  # If the second control point is in free indices
        # Get the optimized y-coordinate for the second control point
        p1_y = y_vals[1]
        
        # Calculate x-coordinate based on tangent direction
        # We want P1 to lie on the line through P0 in the direction of the tangent
        # P1 = P0 + t * tangent, where t is determined by the y-coordinate
        tx, ty = split_tangent
        if abs(ty) > 1e-12:  # Avoid division by zero
            # Solve for t: split_y + t * ty = p1_y
            t = (p1_y - split_y) / ty
            p1_x = split_x + t * tx
        else:
            # If tangent is horizontal, use the fixed x-coordinate
            p1_x = fixed_inner_x_coords[1]
        
        # Update the second control point
        control_points[1] = np.array([p1_x, p1_y])
    
    # Set trailing edge
    control_points[-1] = np.array([1.0, te_y])
    
    return control_points

def assemble_polygons(var_y, inner_x_upper, inner_x_lower, original_upper_data, original_lower_data):
    """
    Assemble full control polygons for coupled Bezier optimization.
    """
    n_inner = len(inner_x_upper)
    y_u = var_y[:n_inner]
    y_l = var_y[n_inner:]
    ctrl_upper = np.zeros((n_inner + 2, 2))
    ctrl_lower = np.zeros((n_inner + 2, 2))
    ctrl_upper[0] = [0.0, 0.0]
    ctrl_upper[1:-1, 0] = inner_x_upper
    ctrl_upper[1:-1, 1] = y_u
    ctrl_upper[-1] = [1.0, float(original_upper_data[-1, 1])]
    ctrl_lower[0] = [0.0, 0.0]
    ctrl_lower[1:-1, 0] = inner_x_lower
    ctrl_lower[1:-1, 1] = y_l
    ctrl_lower[-1] = [1.0, float(original_lower_data[-1, 1])]
    return ctrl_upper, ctrl_lower

def smoothness_penalty(ctrl):
    # penalty = 0.0
    # for i in range(1, len(ctrl) - 1):
    #     v1 = ctrl[i] - ctrl[i - 1]
    #     v2 = ctrl[i + 1] - ctrl[i]
    #     norm1 = np.linalg.norm(v1)
    #     norm2 = np.linalg.norm(v2)
    #     if norm1 < 1e-8 or norm2 < 1e-8:
    #         continue
    #     cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    #     angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    #     penalty += angle**2
    # return penalty / (len(ctrl) - 2)

    deltas = np.diff(ctrl, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_length = np.sum(segment_lengths)
    chord = np.linalg.norm(ctrl[-1] - ctrl[0])
    return (total_length - chord)**2 / len(ctrl)

def te_tangent_constraint_factory(tx_te, ty_te, px_n, py_n, px_n_minus_1, idx):
    """
    Returns a constraint function for trailing edge tangency for the given index.
    """
    def constraint(variables_y):
        y_n_minus_1 = variables_y[idx]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    return constraint

def g2_constraint_factory(inner_x_upper, inner_x_lower, original_upper_data, original_lower_data):
    """
    Returns a constraint function for G2 continuity at the leading edge.
    """
    def constraint(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y, inner_x_upper, inner_x_lower, original_upper_data, original_lower_data)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)
    return constraint

def g2_constraint_factory_fixed_x(inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u, te_y_upper,
                                 inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l, te_y_lower,
                                 original_upper_data, original_lower_data):
    """
    Returns a constraint function for G2 continuity at the leading edge for fixed-x coupled paths,
    reconstructing the full polygons from free variables and fixed pre-TE y-values.
    """
    def constraint(var_y):
        n_free_u = len(free_idx_u)
        n_free_l = len(free_idx_l)
        y_u = var_y[:n_free_u]
        y_l = var_y[n_free_u:]
        ctrl_upper = build_control_points_with_fixed(y_u, inner_x_upper, te_y_upper, free_idx_u, fixed_idx_u, fixed_y_u)
        ctrl_lower = build_control_points_with_fixed(y_l, inner_x_lower, te_y_lower, free_idx_l, fixed_idx_l, fixed_y_l)
        return leading_edge_curvature(ctrl_upper) + leading_edge_curvature(ctrl_lower)
    return constraint

def extract_free_y_from_ctrl(ctrl, free_indices):
    """
    Extract the free y-variables from a control polygon given the free indices.
    """
    return ctrl[1:-1, 1][free_indices]

def make_build_ctrl_fn(fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values):
    """
    Returns a function that builds control points from y-variables for single Bezier.
    """
    def build_ctrl(y):
        return build_control_points_with_fixed(y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    return build_ctrl

def make_residuals_fn(original_data, error_function, softmax=False):
    """
    Returns a function that computes residuals for a given control polygon.
    """
    def residuals(ctrl):
        efun = error_function
        if softmax:
            efun = efun + "_softmax" if not efun.endswith("_softmax") else efun
        residuals, _, _ = __import__('core.error_functions', fromlist=['calculate_single_bezier_fitting_error']).calculate_single_bezier_fitting_error(
            ctrl, original_data, error_function=efun, return_max_error=False, return_all=True)
        return residuals
    return residuals

def detect_stall(iteration_data, threshold=1e-7, window=5):
    if len(iteration_data) < window:
        return False
    recent = iteration_data[-window:]
    diffs = [abs(recent[i][2] - recent[i-1][2]) for i in range(1, window)]
    return all(d < threshold for d in diffs)

def report_constraint_violations(constraints, x):
    print("\nConstraint diagnostics:")
    for i, con in enumerate(constraints):
        try:
            fval = con['fun'](x)
            if isinstance(fval, (list, np.ndarray)):
                for j, val in enumerate(fval):
                    if val < -1e-8:
                        print(f"  Constraint {i}[{j}] violated: {val:.3e}")
            else:
                if fval < -1e-8:
                    print(f"  Constraint {i} violated: {fval:.3e}")
        except Exception as e:
            print(f"  Constraint {i} evaluation failed: {e}")
    print("  (Only violations > 1e-8 are shown.)")

def log_residuals(ctrl, residual_fn, max_display=10):
    """
    Prints a sorted list of residuals (by absolute magnitude).
    Useful to see which points dominate the max error.
    """
    try:
        res = residual_fn(ctrl)
        abs_res = [(i, r, abs(r)) for i, r in enumerate(res)]
        abs_res.sort(key=lambda x: -x[2])
        print("\nTop residuals (by magnitude):")
        for i, val, mag in abs_res[:max_display]:
            y = ctrl[i][1] if i < len(ctrl) else float("nan")
            x = ctrl[i][0] if i < len(ctrl) else float("nan")
            print(f"  #{i:3d}: {val:+.6e} (|r| = {mag:.6e}) at x = {x:.5f}, y = {y:.5f}")
        print("")
    except Exception as e:
        print(f"  Failed to evaluate residuals: {e}")




def run_softmax_stage(
    initial_y,
    build_control_points_fn,
    residuals_fn,
    regularization_weight,
    smoothness_fn,
    constraints_fns=None,
    logger_func=None,
    abort_flag=None
):
    """
    Generic softmax optimizer for single or coupled Bezier, given modular helpers.
    constraints_fns: list of (type, fn) tuples, e.g. [("ineq", fn1), ("eq", fn2)]
    """
    
    # Track best configuration during optimization
    best_max_error = float("inf")
    best_y = None
    
    def full_obj(y):
        nonlocal best_max_error, best_y
        
        ctrl = build_control_points_fn(y)
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
        smoothness_penalty_val = smoothness_fn(ctrl) if smoothness_fn and regularization_weight > 0 else 0.0
        
        # Calculate final objective
        position_weight = 100 if regularization_weight == 0.0 else 1
        position_term = position_weight
        smoothness_term = regularization_weight * smoothness_penalty_val
        total_obj = softmax_error + position_term + smoothness_term
        
        return total_obj
    
    # Attach debug functions for logging
    if config.DEBUG_WORKER_LOGGING:
        def get_residuals_with_debug(y):
            return residuals_fn(build_control_points_fn(y))
        
        full_obj.__get_residuals__ = get_residuals_with_debug
        full_obj.__get_max_error__ = lambda y: np.max(np.abs(residuals_fn(build_control_points_fn(y))))
        full_obj.__build_ctrl__ = build_control_points_fn

    constraints = []
    
    if constraints_fns:
        for ctype, fn in constraints_fns:
            constraints.append({"type": ctype, "fun": fn})

    result, iteration_data = minimize_with_debug_with_abort(
        fun=full_obj,
        x0=initial_y,
        method="SLSQP",
        constraints=constraints,
        options=config.SLSQP_OPTIONS,
        success_threshold=config.MAX_ERROR_THRESHOLD,
        abort_flag=abort_flag
    )

    if logger_func:
        if result.success:
            logger_func("Softmax optimization succeeded.")
        else:
            logger_func(f"Softmax optimization failed: {result.message}")

    # Use the best configuration found during optimization (if available)
    if best_y is not None:
        final_ctrl = build_control_points_fn(best_y)
        if logger_func:
            logger_func(f"Using best configuration found (max error: {best_max_error:.6e})")
    else:
        # Always use result.x since minimize_with_debug_with_abort returns the best result found
        final_ctrl = build_control_points_fn(result.x)
    
    return final_ctrl




