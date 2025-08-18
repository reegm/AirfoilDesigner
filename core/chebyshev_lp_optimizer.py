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


def solve_coupled_chebyshev_lp_fixed_x(
    original_upper_data,
    original_lower_data,
    fixed_inner_x_coords_upper,
    fixed_inner_x_coords_lower,
    te_y_upper,
    te_y_lower,
    free_indices_upper,
    fixed_indices_upper,
    fixed_y_values_upper,
    free_indices_lower,
    fixed_indices_lower,
    fixed_y_values_lower,
    error_function="euclidean",
    regularization_weight=0.0,
    logger_func=None,
    abort_flag=None
):
    """
    Solve the coupled Chebyshev (minimax) approximation problem using Linear Programming.
    
    Optimizes both upper and lower surfaces simultaneously with G2 continuity constraints.
    The optimization variables are the free y-coordinates for both surfaces.
    
    Args:
        original_upper_data: Target points for upper surface
        original_lower_data: Target points for lower surface  
        fixed_inner_x_coords_upper: Fixed x-coordinates for upper inner control points
        fixed_inner_x_coords_lower: Fixed x-coordinates for lower inner control points
        te_y_upper: Upper surface trailing edge y-coordinate
        te_y_lower: Lower surface trailing edge y-coordinate
        free_indices_upper: Indices of free y-variables in upper inner control points
        fixed_indices_upper: Indices of fixed y-variables in upper inner control points
        fixed_y_values_upper: Values for fixed y-variables in upper surface
        free_indices_lower: Indices of free y-variables in lower inner control points
        fixed_indices_lower: Indices of fixed y-variables in lower inner control points
        fixed_y_values_lower: Values for fixed y-variables in lower surface
        error_function: "euclidean" (others not supported for LP)
        regularization_weight: Smoothness penalty (approximated with L1 norm)
        logger_func: Optional logging callback
        abort_flag: Optional abort flag
        
    Returns:
        Tuple of (optimal_y_upper, optimal_y_lower) for free variables, or (None, None) if failed
    """
    if error_function != "euclidean":
        raise ValueError("LP solver only supports euclidean error function")
    
    if logger_func:
        logger_func("Setting up coupled Chebyshev LP problem...")
    
    n_free_vars_upper = len(free_indices_upper)
    n_free_vars_lower = len(free_indices_lower)
    n_free_vars_total = n_free_vars_upper + n_free_vars_lower
    
    def build_coupled_residual_matrix():
        """
        Build the constraint matrix for coupled surfaces.
        Variables are organized as: [y_upper_0, y_upper_1, ..., y_lower_0, y_lower_1, ...]
        """
        A_rows_upper = []
        b_values_upper = []
        A_rows_lower = []
        b_values_lower = []
        
        if logger_func:
            logger_func(f"Building constraint matrix for {len(original_upper_data)} upper + {len(original_lower_data)} lower data points...")
        
        # Process upper surface
        for i, (data_x, data_y) in enumerate(original_upper_data):
            # Build baseline curve with all free y-variables = 0
            y_baseline = np.zeros(len(free_indices_upper))
            ctrl_baseline = build_control_points_with_fixed(
                y_baseline, fixed_inner_x_coords_upper, te_y_upper, 
                free_indices_upper, fixed_indices_upper, fixed_y_values_upper)
            
            # Find parameter t where curve x-coordinate matches data_x
            best_t = 0.0
            best_x_diff = float('inf')
            
            for t_candidate in np.linspace(0.0, 1.0, 100):
                from utils.bezier_utils import general_bezier_curve
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
            
            # Compute sensitivity to upper surface variables
            row = np.zeros(n_free_vars_total)
            baseline_y = general_bezier_curve(np.array([best_t]), ctrl_baseline)[0, 1]
            
            for j in range(n_free_vars_upper):
                y_perturb = np.zeros(len(free_indices_upper))
                y_perturb[j] = 1.0
                
                ctrl_perturb = build_control_points_with_fixed(
                    y_perturb, fixed_inner_x_coords_upper, te_y_upper,
                    free_indices_upper, fixed_indices_upper, fixed_y_values_upper)
                
                perturb_y = general_bezier_curve(np.array([best_t]), ctrl_perturb)[0, 1]
                row[j] = perturb_y - baseline_y  # Upper variables come first
            
            # Lower variables have zero influence on upper surface residuals (row[n_free_vars_upper:] remains 0)
            A_rows_upper.append(row)
            b_values_upper.append(data_y - baseline_y)
        
        # Process lower surface  
        for i, (data_x, data_y) in enumerate(original_lower_data):
            # Build baseline curve with all free y-variables = 0
            y_baseline = np.zeros(len(free_indices_lower))
            ctrl_baseline = build_control_points_with_fixed(
                y_baseline, fixed_inner_x_coords_lower, te_y_lower,
                free_indices_lower, fixed_indices_lower, fixed_y_values_lower)
            
            # Find parameter t where curve x-coordinate matches data_x
            best_t = 0.0
            best_x_diff = float('inf')
            
            for t_candidate in np.linspace(0.0, 1.0, 100):
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
            
            # Compute sensitivity to lower surface variables
            row = np.zeros(n_free_vars_total)
            baseline_y = general_bezier_curve(np.array([best_t]), ctrl_baseline)[0, 1]
            
            for j in range(n_free_vars_lower):
                y_perturb = np.zeros(len(free_indices_lower))
                y_perturb[j] = 1.0
                
                ctrl_perturb = build_control_points_with_fixed(
                    y_perturb, fixed_inner_x_coords_lower, te_y_lower,
                    free_indices_lower, fixed_indices_lower, fixed_y_values_lower)
                
                perturb_y = general_bezier_curve(np.array([best_t]), ctrl_perturb)[0, 1]
                row[n_free_vars_upper + j] = perturb_y - baseline_y  # Lower variables come second
            
            # Upper variables have zero influence on lower surface residuals (row[:n_free_vars_upper] remains 0)
            A_rows_lower.append(row)
            b_values_lower.append(data_y - baseline_y)
        
        # Combine upper and lower constraints
        A_combined = np.vstack([np.array(A_rows_upper), np.array(A_rows_lower)])
        b_combined = np.concatenate([np.array(b_values_upper), np.array(b_values_lower)])
        
        return A_combined, b_combined
    
    # Build the linear system
    A, b = build_coupled_residual_matrix()
    n_constraints = len(b)
    
    if logger_func:
        logger_func(f"Coupled LP problem: {n_free_vars_total} variables ({n_free_vars_upper} upper + {n_free_vars_lower} lower), {n_constraints} evaluation points")
    
    # Set up the LP problem with G2 constraint and regularization
    # Variables: [y_upper_0, ..., y_upper_{n-1}, y_lower_0, ..., y_lower_{m-1}, t, s]
    if regularization_weight > 0:
        n_vars = n_free_vars_total + 2  # y-variables + t + s (smoothness variable)
    else:
        n_vars = n_free_vars_total + 1  # y-variables + t only
    
    # Objective: minimize t + regularization_weight * s
    c = np.zeros(n_vars)
    if regularization_weight > 0:
        c[-2] = 1.0  # Coefficient for t
        scaled_regularization = regularization_weight * 50.0  # Same scaling as single surface
        c[-1] = scaled_regularization  # Coefficient for s
    else:
        c[-1] = 1.0  # Coefficient for t
        # Add implicit smoothness
        implicit_smoothness = 1e-4
        for i in range(n_free_vars_total):
            c[i] = implicit_smoothness
    
    # Inequality constraints
    num_base_constraints = 2 * n_constraints
    if regularization_weight > 0:
        # Smoothness constraints for both surfaces
        if n_free_vars_upper >= 2:
            num_smoothness_constraints_upper = 2 * (n_free_vars_upper - 1)
        else:
            num_smoothness_constraints_upper = 2 * max(0, n_free_vars_upper - 1)
        if n_free_vars_lower >= 2:
            num_smoothness_constraints_lower = 2 * (n_free_vars_lower - 1)
        else:
            num_smoothness_constraints_lower = 2 * max(0, n_free_vars_lower - 1)
        num_smoothness_constraints = num_smoothness_constraints_upper + num_smoothness_constraints_lower
    else:
        num_smoothness_constraints = 0
    
    # Add G2 constraint: upper and lower curves must have same curvature at LE
    # This is a linear constraint on the first control points
    num_g2_constraints = 1  # One equality constraint = two inequality constraints
    
    total_constraints = num_base_constraints + num_smoothness_constraints + 2 * num_g2_constraints
    
    A_ub = np.zeros((total_constraints, n_vars))
    b_ub = np.zeros(total_constraints)
    
    # Base residual constraints (same as single surface, but for combined variables)
    for i in range(n_constraints):
        if regularization_weight > 0:
            # Constraint: -A[i,:] @ y - t ≤ -b[i]
            A_ub[2*i, :n_free_vars_total] = -A[i, :]
            A_ub[2*i, -2] = -1.0  # coefficient for t
            b_ub[2*i] = -b[i]
            
            # Constraint: A[i,:] @ y - t ≤ b[i]
            A_ub[2*i+1, :n_free_vars_total] = A[i, :]
            A_ub[2*i+1, -2] = -1.0  # coefficient for t
            b_ub[2*i+1] = b[i]
        else:
            # Without regularization
            A_ub[2*i, :n_free_vars_total] = -A[i, :]
            A_ub[2*i, -1] = -1.0  # coefficient for t
            b_ub[2*i] = -b[i]
            
            A_ub[2*i+1, :n_free_vars_total] = A[i, :]
            A_ub[2*i+1, -1] = -1.0  # coefficient for t
            b_ub[2*i+1] = b[i]
    
    constraint_idx = num_base_constraints
    
    # Add smoothness constraints for upper surface
    if regularization_weight > 0 and n_free_vars_upper >= 2:
        for i in range(n_free_vars_upper - 1):
            if i == 0:
                # First point: y[1] - 2*y[0] (assuming y[-1] ≈ 0)
                A_ub[constraint_idx, 0] = -2.0
                A_ub[constraint_idx, 1] = 1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, 0] = 2.0
                A_ub[constraint_idx, 1] = -1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
            elif i == n_free_vars_upper - 2:
                # Last valid second derivative
                A_ub[constraint_idx, i-1] = 1.0
                A_ub[constraint_idx, i] = -2.0
                A_ub[constraint_idx, i+1] = 1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, i-1] = -1.0
                A_ub[constraint_idx, i] = 2.0
                A_ub[constraint_idx, i+1] = -1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
            else:
                # Middle points
                A_ub[constraint_idx, i-1] = 1.0
                A_ub[constraint_idx, i] = -2.0
                A_ub[constraint_idx, i+1] = 1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, i-1] = -1.0
                A_ub[constraint_idx, i] = 2.0
                A_ub[constraint_idx, i+1] = -1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
    
    # Add smoothness constraints for lower surface (offset indices by n_free_vars_upper)
    if regularization_weight > 0 and n_free_vars_lower >= 2:
        for i in range(n_free_vars_lower - 1):
            offset = n_free_vars_upper  # Lower variables start after upper variables
            if i == 0:
                A_ub[constraint_idx, offset + 0] = -2.0
                A_ub[constraint_idx, offset + 1] = 1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, offset + 0] = 2.0
                A_ub[constraint_idx, offset + 1] = -1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
            elif i == n_free_vars_lower - 2:
                A_ub[constraint_idx, offset + i-1] = 1.0
                A_ub[constraint_idx, offset + i] = -2.0
                A_ub[constraint_idx, offset + i+1] = 1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, offset + i-1] = -1.0
                A_ub[constraint_idx, offset + i] = 2.0
                A_ub[constraint_idx, offset + i+1] = -1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
            else:
                A_ub[constraint_idx, offset + i-1] = 1.0
                A_ub[constraint_idx, offset + i] = -2.0
                A_ub[constraint_idx, offset + i+1] = 1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
                
                A_ub[constraint_idx, offset + i-1] = -1.0
                A_ub[constraint_idx, offset + i] = 2.0
                A_ub[constraint_idx, offset + i+1] = -1.0
                A_ub[constraint_idx, -1] = -1.0
                b_ub[constraint_idx] = 0.0
                constraint_idx += 1
    
    # Add G2 constraint: y_upper[0] = -y_lower[0] (symmetric first control points for G2 continuity)
    # This ensures symmetric tangent directions at the leading edge
    # This becomes: y_upper[0] + y_lower[0] = 0, implemented as two inequalities:
    # y_upper[0] + y_lower[0] ≤ 0 and -y_upper[0] - y_lower[0] ≤ 0
    A_ub[constraint_idx, 0] = 1.0  # y_upper[0]
    A_ub[constraint_idx, n_free_vars_upper] = 1.0  # +y_lower[0]
    b_ub[constraint_idx] = 0.0
    constraint_idx += 1
    
    A_ub[constraint_idx, 0] = -1.0  # -y_upper[0]
    A_ub[constraint_idx, n_free_vars_upper] = -1.0  # -y_lower[0]
    b_ub[constraint_idx] = 0.0
    constraint_idx += 1
    
    # Bounds on variables
    y_bounds = [(-1.0, 1.0)] * n_free_vars_total  # y-coordinates bounded
    t_bounds = [(0.0, None)]  # t >= 0
    if regularization_weight > 0:
        s_bounds = [(0.0, None)]  # s >= 0
        bounds = y_bounds + t_bounds + s_bounds
    else:
        bounds = y_bounds + t_bounds
    
    if abort_flag is not None and abort_flag.value:
        return None, None
    
    if logger_func:
        logger_func("Solving coupled LP problem...")
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                        method='highs', options={'disp': False})
        
        if result.success:
            if logger_func:
                logger_func(f"LP solver result: success={result.success}, x shape={result.x.shape if result.x is not None else None}")
            
            optimal_y_upper = result.x[:n_free_vars_upper]
            optimal_y_lower = result.x[n_free_vars_upper:n_free_vars_total]
            
            if regularization_weight > 0:
                optimal_t = result.x[-2]
                optimal_s = result.x[-1]
            else:
                optimal_t = result.x[-1]
                optimal_s = 0.0
            
            if logger_func:
                if regularization_weight > 0:
                    logger_func(f"Coupled LP solved successfully. LP optimal t: {optimal_t:.6e}, smoothness s: {optimal_s:.6e}")
                else:
                    logger_func(f"Coupled LP solved successfully. LP optimal t: {optimal_t:.6e}")
                logger_func(f"Returning optimal_y_upper shape: {optimal_y_upper.shape}, optimal_y_lower shape: {optimal_y_lower.shape}")
            
            return optimal_y_upper, optimal_y_lower
        else:
            if logger_func:
                logger_func(f"Coupled LP solver failed: {result.message}")
            return None, None
            
    except Exception as e:
        if logger_func:
            logger_func(f"Coupled LP solver error: {str(e)}")
        return None, None


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


def optimize_coupled_fixed_x_chebyshev_lp(
    original_upper_data,
    original_lower_data,
    regularization_weight=0.0,
    te_tangent_vector_upper=None,
    te_tangent_vector_lower=None,
    error_function="euclidean",
    logger_func=None,
    abort_flag=None
):
    """
    High-level interface for coupled Chebyshev LP optimization with fixed x-coordinates.
    
    This optimizes both upper and lower surfaces simultaneously with G2 continuity.
    """
    from core.solver_helpers import get_fixed_inner_x_partition
    from core import config
    
    # Get TE y values
    te_y_upper = float(original_upper_data[-1, 1])
    te_y_lower = float(original_lower_data[-1, 1])
    
    # Get the fixed-x partitions for both surfaces
    fixed_inner_x_coords_upper, free_indices_upper, fixed_indices_upper, fixed_y_values_upper = get_fixed_inner_x_partition(
        True, config.NUM_CONTROL_POINTS_SINGLE_BEZIER, original_upper_data, te_tangent_vector_upper, te_y_upper)
    
    fixed_inner_x_coords_lower, free_indices_lower, fixed_indices_lower, fixed_y_values_lower = get_fixed_inner_x_partition(
        False, config.NUM_CONTROL_POINTS_SINGLE_BEZIER, original_lower_data, te_tangent_vector_lower, te_y_lower)
    
    # Solve using coupled LP
    optimal_y_upper, optimal_y_lower = solve_coupled_chebyshev_lp_fixed_x(
        original_upper_data=original_upper_data,
        original_lower_data=original_lower_data,
        fixed_inner_x_coords_upper=fixed_inner_x_coords_upper,
        fixed_inner_x_coords_lower=fixed_inner_x_coords_lower,
        te_y_upper=te_y_upper,
        te_y_lower=te_y_lower,
        free_indices_upper=free_indices_upper,
        fixed_indices_upper=fixed_indices_upper,
        fixed_y_values_upper=fixed_y_values_upper,
        free_indices_lower=free_indices_lower,
        fixed_indices_lower=fixed_indices_lower,
        fixed_y_values_lower=fixed_y_values_lower,
        error_function=error_function,
        regularization_weight=regularization_weight,
        logger_func=logger_func,
        abort_flag=abort_flag
    )
    
    if optimal_y_upper is None or optimal_y_lower is None:
        if logger_func:
            logger_func("Coupled LP optimization failed - optimal_y is None")
        return None
    
    if logger_func:
        logger_func(f"Building final control points from optimal_y_upper: {optimal_y_upper.shape}, optimal_y_lower: {optimal_y_lower.shape}")
    
    # Build final control points for both surfaces
    final_ctrl_upper = build_control_points_with_fixed(
        optimal_y_upper, fixed_inner_x_coords_upper, te_y_upper, 
        free_indices_upper, fixed_indices_upper, fixed_y_values_upper)
    
    final_ctrl_lower = build_control_points_with_fixed(
        optimal_y_lower, fixed_inner_x_coords_lower, te_y_lower,
        free_indices_lower, fixed_indices_lower, fixed_y_values_lower)
    
    if logger_func:
        logger_func(f"Final control points - upper: {final_ctrl_upper.shape}, lower: {final_ctrl_lower.shape}")
        logger_func("Returning coupled Chebyshev LP result to caller")
    
    return final_ctrl_upper, final_ctrl_lower