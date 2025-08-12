"""
Chebyshev (minimax) optimizer using Linear Programming for fixed-x Bézier curves.

For fixed control point x-coordinates, the Bézier curve evaluation is linear in y-coordinates,
making the minimax problem a Linear Program:

minimize t
subject to: -t ≤ residual_i(y) ≤ t  for all i

This is much more efficient than the softmax approximation and finds the true minimax solution.
"""

import numpy as np
from scipy.optimize import linprog
from core.error_functions import calculate_single_bezier_fitting_error
from core.solver_helpers import build_control_points_with_fixed
from utils.bezier_utils import general_bezier_curve
from core import config


def solve_chebyshev_lp_fixed_x(
    original_data,
    fixed_inner_x_coords, 
    te_y,
    free_indices,
    fixed_indices, 
    fixed_y_values,
    error_function="euclidean",
    regularization_weight=0.0,
    logger_func=None,
    abort_flag=None
):
    """
    Solve the Chebyshev (minimax) approximation problem using Linear Programming.
    
    For fixed x-coordinates, this finds the exact solution to:
    min max_i |residual_i(y)|
    
    Args:
        original_data: Target points to fit
        fixed_inner_x_coords: Fixed x-coordinates for inner control points  
        te_y: Trailing edge y-coordinate
        free_indices: Indices of free y-variables in the inner control points
        fixed_indices: Indices of fixed y-variables in the inner control points
        fixed_y_values: Values for fixed y-variables
        error_function: "euclidean" (others not supported for LP)
        regularization_weight: Smoothness penalty (approximated with L1 norm)
        logger_func: Optional logging callback
        abort_flag: Optional abort flag
        
    Returns:
        Optimal y-coordinates for free variables, or None if failed
    """
    if error_function != "euclidean":
        raise ValueError("LP solver only supports euclidean error function")
    
    if logger_func:
        logger_func("Setting up Chebyshev LP problem...")
    
    n_free_vars = len(free_indices)
    
    # We need to evaluate the Bézier curve at many points to get residuals
    # Use moderate resolution - LP doesn't need as many points as iterative methods
    num_eval_points = max(20000, len(original_data) * 2)  # Reasonable resolution for LP
    
    def build_residual_matrix():
        """
        Build the constraint matrix A where A @ y gives residuals.
        
        The key insight: For each data point (x_i, y_i), we want to minimize |y_i - curve(x_i)|.
        Since x-coordinates are fixed, we can directly evaluate the curve at each data x-coordinate
        using the Bézier basis functions.
        """
        A_rows = []
        b_values = []
        
        if logger_func:
            logger_func(f"Building constraint matrix for {len(original_data)} data points...")
        
        for i, (data_x, data_y) in enumerate(original_data):
            # For each data point, find the parameter t such that curve_x(t) ≈ data_x
            # Since x-coordinates are fixed, we can solve this directly
            
            # Build baseline curve with all free y-variables = 0
            y_baseline = np.zeros(len(free_indices))
            ctrl_baseline = build_control_points_with_fixed(
                y_baseline, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
            
            # Find parameter t where curve x-coordinate matches data_x
            # Use a simple search since this is precomputation
            best_t = 0.0
            best_x_diff = float('inf')
            
            for t_candidate in np.linspace(0.0, 1.0, 100):  # Coarse search
                curve_point = general_bezier_curve(np.array([t_candidate]), ctrl_baseline)
                curve_x = curve_point[0, 0]
                x_diff = abs(curve_x - data_x)
                if x_diff < best_x_diff:
                    best_x_diff = x_diff
                    best_t = t_candidate
            
            # Fine search around best_t
            t_range = np.linspace(max(0.0, best_t - 0.02), min(1.0, best_t + 0.02), 50)
            for t_candidate in t_range:
                curve_point = general_bezier_curve(np.array([t_candidate]), ctrl_baseline)
                curve_x = curve_point[0, 0]
                x_diff = abs(curve_x - data_x)
                if x_diff < best_x_diff:
                    best_x_diff = x_diff
                    best_t = t_candidate
            
            # Now we have the parameter t that gives us x ≈ data_x
            # Compute how each free y-variable affects the curve y-value at this t
            row = np.zeros(n_free_vars)
            
            # Baseline y-value at this t
            baseline_y = general_bezier_curve(np.array([best_t]), ctrl_baseline)[0, 1]
            
            # Compute the sensitivity to each free y-variable
            for j in range(n_free_vars):
                y_perturb = np.zeros(len(free_indices))
                y_perturb[j] = 1.0  # Unit perturbation
                
                ctrl_perturb = build_control_points_with_fixed(
                    y_perturb, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
                
                perturb_y = general_bezier_curve(np.array([best_t]), ctrl_perturb)[0, 1]
                
                # Sensitivity: how much does curve_y change when y_j changes by 1
                row[j] = perturb_y - baseline_y
            
            A_rows.append(row)
            
            # The residual target: data_y - baseline_y
            # We want: data_y = baseline_y + A[i,:] @ y
            # So: residual = data_y - (baseline_y + A[i,:] @ y) = (data_y - baseline_y) - A[i,:] @ y
            b_values.append(data_y - baseline_y)
        
        return np.array(A_rows), np.array(b_values)
    
    # Build the linear system: residuals = A @ y + b
    A, b = build_residual_matrix()
    n_constraints = len(b)
    
    if logger_func:
        logger_func(f"LP problem: {n_free_vars} variables, {n_constraints} evaluation points")
    
    # Set up the LP problem with regularization:
    # Variables: [y_0, y_1, ..., y_{n-1}, t, s] where s is smoothness penalty
    # Minimize: t + regularization_weight * s
    # Subject to: -t ≤ b[i] - A[i,:] @ y ≤ t  for all i
    #            smoothness_penalty(y) ≤ s
    #
    # The smoothness constraint is approximated using the total variation penalty
    
    if regularization_weight > 0:
        n_vars = n_free_vars + 2  # y-variables + t + s (smoothness variable)
    else:
        n_vars = n_free_vars + 1  # y-variables + t only
    
    # Objective: minimize t + regularization_weight * s
    c = np.zeros(n_vars)
    if regularization_weight > 0:
        c[-2] = 1.0  # Coefficient for t (second to last variable)
        # Scale regularization more aggressively to be more effective
        # The user's regularization weights are tuned for MSR, but LP needs stronger smoothness
        scaled_regularization = regularization_weight * 50.0  # Much stronger scaling
        c[-1] = scaled_regularization  # Coefficient for s (smoothness penalty)
    else:
        c[-1] = 1.0  # Coefficient for t (last variable)
        # Add small implicit smoothness even without regularization to reduce wobbles
        # This helps numerical stability and reduces oscillatory solutions
        implicit_smoothness = 1e-4  # Slightly larger penalty to have some effect
        for i in range(n_free_vars):
            c[i] = implicit_smoothness
    
    # Inequality constraints: A_ub @ x ≤ b_ub
    # Base residual constraints: 2 * n_constraints inequalities
    # Plus smoothness constraints if regularization > 0
    num_base_constraints = 2 * n_constraints
    if regularization_weight > 0:
        if n_free_vars >= 2:
            # Curvature-based smoothness: 2 constraints per second derivative
            # We have (n_free_vars - 1) second derivative terms
            num_smoothness_constraints = 2 * (n_free_vars - 1)
        else:
            # Fallback to first derivative smoothness
            num_smoothness_constraints = 2 * max(0, n_free_vars - 1)
    else:
        num_smoothness_constraints = 0
    total_constraints = num_base_constraints + num_smoothness_constraints
    
    A_ub = np.zeros((total_constraints, n_vars))
    b_ub = np.zeros(total_constraints)
    
    # Base residual constraints
    for i in range(n_constraints):
        if regularization_weight > 0:
            # With regularization: variables are [y_0, ..., y_{n-1}, t, s]
            # Constraint: -A[i,:] @ y - t ≤ -b[i]
            A_ub[2*i, :n_free_vars] = -A[i, :]  # coefficients for y
            A_ub[2*i, -2] = -1.0                # coefficient for t
            b_ub[2*i] = -b[i]
            
            # Constraint: A[i,:] @ y - t ≤ b[i]  
            A_ub[2*i+1, :n_free_vars] = A[i, :] # coefficients for y
            A_ub[2*i+1, -2] = -1.0              # coefficient for t
            b_ub[2*i+1] = b[i]
        else:
            # Without regularization: variables are [y_0, ..., y_{n-1}, t]
            # Constraint: -A[i,:] @ y - t ≤ -b[i]
            A_ub[2*i, :n_free_vars] = -A[i, :]  # coefficients for y
            A_ub[2*i, -1] = -1.0                # coefficient for t
            b_ub[2*i] = -b[i]
            
            # Constraint: A[i,:] @ y - t ≤ b[i]  
            A_ub[2*i+1, :n_free_vars] = A[i, :] # coefficients for y
            A_ub[2*i+1, -1] = -1.0              # coefficient for t
            b_ub[2*i+1] = b[i]
    
    # Add smoothness constraints if regularization is enabled
    if regularization_weight > 0:
        # Better smoothness formulation: approximate the path-length penalty used by MSR
        # The MSR smoothness penalty is (total_length - chord)^2 / len(ctrl)
        # For LP, we approximate this by minimizing sum of |segment_i| - avg_segment_length
        # where avg_segment_length = chord / (n_segments)
        
        # We have n_free_vars+1 segments (LE to first, between free points, last to TE)
        # Let chord length ≈ 1.0 for normalized airfoils
        chord_length = 1.0
        n_segments = n_free_vars + 1
        avg_segment_length = chord_length / n_segments
        
        # For simplicity, penalize large y-coordinate differences (curvature-based smoothness)
        # This is more relevant for Bézier curves than total path length
        constraint_idx = num_base_constraints
        
        # Add weaker smoothness: penalize large second derivatives (curvature)
        if n_free_vars >= 2:
            for i in range(n_free_vars - 1):
                # Approximate second derivative: y[i-1] - 2*y[i] + y[i+1]
                # Add constraints: s ≥ |y[i] - 2*y[i+1] + y[i+2]| (for i=0,1,...)
                # But we need to handle boundaries carefully
                
                if i == 0:
                    # First point: assume y[-1] ≈ 0 (LE constraint)
                    # Second derivative ≈ 0 - 2*y[0] + y[1] = y[1] - 2*y[0]
                    # s ≥ |y[1] - 2*y[0]|
                    
                    # s ≥ (y[1] - 2*y[0])
                    A_ub[constraint_idx, 0] = -2.0     # -2*y[0]  
                    A_ub[constraint_idx, 1] = 1.0      # +y[1]
                    A_ub[constraint_idx, -1] = -1.0    # -s
                    b_ub[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    # s ≥ -(y[1] - 2*y[0])  
                    A_ub[constraint_idx, 0] = 2.0      # +2*y[0]
                    A_ub[constraint_idx, 1] = -1.0     # -y[1]
                    A_ub[constraint_idx, -1] = -1.0    # -s
                    b_ub[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                elif i == n_free_vars - 2:
                    # Last valid second derivative: y[n-3] - 2*y[n-2] + y[n-1]
                    # where y[n-1] is the last free variable
                    
                    # s ≥ (y[i-1] - 2*y[i] + y[i+1])
                    A_ub[constraint_idx, i-1] = 1.0    # +y[i-1]
                    A_ub[constraint_idx, i] = -2.0     # -2*y[i]
                    A_ub[constraint_idx, i+1] = 1.0    # +y[i+1]
                    A_ub[constraint_idx, -1] = -1.0    # -s
                    b_ub[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    # s ≥ -(y[i-1] - 2*y[i] + y[i+1])
                    A_ub[constraint_idx, i-1] = -1.0   # -y[i-1]
                    A_ub[constraint_idx, i] = 2.0      # +2*y[i]
                    A_ub[constraint_idx, i+1] = -1.0   # -y[i+1]
                    A_ub[constraint_idx, -1] = -1.0    # -s
                    b_ub[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                else:
                    # Middle points: regular second derivative
                    # s ≥ |y[i-1] - 2*y[i] + y[i+1]|
                    
                    # s ≥ (y[i-1] - 2*y[i] + y[i+1])
                    A_ub[constraint_idx, i-1] = 1.0    # +y[i-1]
                    A_ub[constraint_idx, i] = -2.0     # -2*y[i]  
                    A_ub[constraint_idx, i+1] = 1.0    # +y[i+1]
                    A_ub[constraint_idx, -1] = -1.0    # -s
                    b_ub[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    # s ≥ -(y[i-1] - 2*y[i] + y[i+1])
                    A_ub[constraint_idx, i-1] = -1.0   # -y[i-1]
                    A_ub[constraint_idx, i] = 2.0      # +2*y[i]
                    A_ub[constraint_idx, i+1] = -1.0   # -y[i+1]
                    A_ub[constraint_idx, -1] = -1.0    # -s
                    b_ub[constraint_idx] = 0.0
                    constraint_idx += 1
        else:
            # Fallback to simple first derivative smoothness for very few variables
            for i in range(n_free_vars - 1):
                # s ≥ |y[i+1] - y[i]|
                A_ub[constraint_idx, i] = -1.0      
                A_ub[constraint_idx, i+1] = 1.0     
                A_ub[constraint_idx, -1] = -1.0     
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, i] = 1.0       
                A_ub[constraint_idx, i+1] = -1.0    
                A_ub[constraint_idx, -1] = -1.0     
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
        
    
    # Bounds on variables
    # Reasonable bounds on y-coordinates
    y_bounds = [(-1.0, 1.0)] * n_free_vars  # y-coordinates bounded
    t_bounds = [(0.0, None)]                 # t >= 0 (absolute value bound)
    if regularization_weight > 0:
        s_bounds = [(0.0, None)]             # s >= 0 (smoothness penalty)
        bounds = y_bounds + t_bounds + s_bounds
    else:
        bounds = y_bounds + t_bounds
    
    if abort_flag is not None and abort_flag.value:
        return None
    
    if logger_func:
        logger_func("Solving LP problem...")
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                        method='highs', options={'disp': False})
        
        if result.success:
            optimal_y = result.x[:n_free_vars]
            if regularization_weight > 0:
                optimal_t = result.x[-2]  # t is second to last
                optimal_s = result.x[-1]  # s is last
            else:
                optimal_t = result.x[-1]  # t is last
                optimal_s = 0.0
            
            # Verify the solution by checking actual residuals
            if logger_func:
                # Build control points with the optimal solution
                ctrl_optimal = build_control_points_with_fixed(
                    optimal_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
                
                # Check residuals at actual data points (the correct way)
                sample_residuals = []
                for data_x, data_y in original_data[:5]:  # Check first 5 points
                    # Find curve y-value at this x-coordinate
                    # (This should match how we built the constraints)
                    best_t = 0.0
                    best_x_diff = float('inf')
                    for t_candidate in np.linspace(0.0, 1.0, 100):
                        curve_point = general_bezier_curve(np.array([t_candidate]), ctrl_optimal)
                        curve_x = curve_point[0, 0]
                        x_diff = abs(curve_x - data_x)
                        if x_diff < best_x_diff:
                            best_x_diff = x_diff
                            best_t = t_candidate
                    
                    curve_y = general_bezier_curve(np.array([best_t]), ctrl_optimal)[0, 1]
                    residual = abs(data_y - curve_y)
                    sample_residuals.append(residual)
                
                actual_max = max(sample_residuals)
                if regularization_weight > 0:
                    logger_func(f"LP solved successfully. LP optimal t: {optimal_t:.6e}, smoothness s: {optimal_s:.6e}, Sample check max: {actual_max:.6e}")
                else:
                    logger_func(f"LP solved successfully. LP optimal t: {optimal_t:.6e}, Sample check max: {actual_max:.6e}")
            
            return optimal_y
        else:
            if logger_func:
                logger_func(f"LP solver failed: {result.message}")
            return None
            
    except Exception as e:
        if logger_func:
            logger_func(f"LP solver error: {str(e)}")
        return None


def optimize_fixed_x_chebyshev_lp(
    original_data,
    num_control_points_new,
    is_upper_surface,
    te_tangent_vector,
    regularization_weight=0.0,
    error_function="euclidean", 
    logger_func=None,
    abort_flag=None
):
    """
    High-level interface for Chebyshev LP optimization with fixed x-coordinates.
    
    This is a drop-in replacement for the softmax-based fixed-x optimizer.
    """
    from core.solver_helpers import get_fixed_inner_x_partition
    
    te_y = float(original_data[-1, 1])
    
    # Get the fixed-x partition (same as current approach)
    fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, original_data, te_tangent_vector, te_y)
    
    # Solve using LP
    optimal_y = solve_chebyshev_lp_fixed_x(
        original_data=original_data,
        fixed_inner_x_coords=fixed_inner_x_coords,
        te_y=te_y,
        free_indices=free_indices,
        fixed_indices=fixed_indices,
        fixed_y_values=fixed_y_values,
        error_function=error_function,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag
    )
    
    if optimal_y is None:
        return None
    
    # Build final control points
    final_ctrl = build_control_points_with_fixed(
        optimal_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    
    return final_ctrl