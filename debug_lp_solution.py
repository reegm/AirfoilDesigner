#!/usr/bin/env python3
"""
Debug script to understand why LP solution reports low error but actual evaluation is bad.
"""

import numpy as np
from core.chebyshev_lp_optimizer import solve_chebyshev_lp_fixed_x
from core.solver_helpers import get_fixed_inner_x_partition, build_control_points_with_fixed
from core.error_functions import calculate_single_bezier_fitting_error
from utils.bezier_utils import general_bezier_curve


def debug_lp_solution():
    """Debug what's going wrong with the LP solution."""
    
    # Load the same airfoil data that failed
    print("=== Debugging LP Solution ===")
    
    # Load the actual FX67K170 airfoil data
    try:
        from utils.data_loader import load_airfoil_data
        upper_surface, lower_surface, airfoil_name, thickened = load_airfoil_data("res/fx67k170.DAT", logger_func=lambda x: None)
        
        # Use upper surface for testing (same as the failed case)
        test_data = upper_surface
        print(f"Loaded {airfoil_name} upper surface: {len(test_data)} points")
        
    except Exception as e:
        print(f"Failed to load airfoil: {e}")
        print("Using fallback simple test data")
        # Fallback to simple test data
        x = np.linspace(0, 1, 50)
        y_true = 0.05 * x * (1 - x) * 4  # Simple parabolic shape
        test_data = np.column_stack([x, y_true])
    
    print(f"Test data: {len(test_data)} points")
    print(f"Data range: x=[{test_data[:, 0].min():.3f}, {test_data[:, 0].max():.3f}], y=[{test_data[:, 1].min():.3f}, {test_data[:, 1].max():.3f}]")
    
    # Set up the LP problem the same way as the real solver
    num_control_points_new = 8
    is_upper_surface = True
    te_tangent_vector = np.array([1.0, 0.0])
    te_y = float(test_data[-1, 1])
    
    print(f"Control points: {num_control_points_new}, te_y: {te_y:.6f}")
    
    # Get fixed-x partition
    fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values = get_fixed_inner_x_partition(
        is_upper_surface, num_control_points_new, test_data, te_tangent_vector, te_y)
    
    print(f"Fixed inner x coords: {fixed_inner_x_coords}")
    print(f"Free indices: {free_indices}, Fixed indices: {fixed_indices}")
    print(f"Fixed y values: {fixed_y_values}")
    
    # Solve using LP
    print("\nSolving LP...")
    optimal_y = solve_chebyshev_lp_fixed_x(
        original_data=test_data,
        fixed_inner_x_coords=fixed_inner_x_coords,
        te_y=te_y,
        free_indices=free_indices,
        fixed_indices=fixed_indices,
        fixed_y_values=fixed_y_values,
        error_function="euclidean",
        regularization_weight=0.0,  # Test improved implicit smoothness
        logger_func=lambda msg: print(f"   {msg}"),
        abort_flag=None
    )
    
    if optimal_y is None:
        print("LP failed!")
        return
    
    print(f"Optimal y: {optimal_y}")
    
    # Build the final control points
    final_ctrl = build_control_points_with_fixed(
        optimal_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    
    print(f"Final control points:\n{final_ctrl}")
    
    # Evaluate the error the same way the GUI does
    print("\nEvaluating final solution...")
    residuals, rms_error, _ = calculate_single_bezier_fitting_error(
        final_ctrl, test_data, error_function="euclidean", return_all=True)
    
    max_error_eval = np.max(np.abs(residuals))
    max_error_idx = np.argmax(np.abs(residuals))
    
    print(f"RMS error: {rms_error:.6e}")
    print(f"Max error (evaluation): {max_error_eval:.6e}")
    print(f"Max error at data point {max_error_idx}: x={test_data[max_error_idx, 0]:.3f}, y={test_data[max_error_idx, 1]:.3f}")
    print(f"Residual at max: {residuals[max_error_idx]:.6e}")
    
    # Now let's manually verify the LP constraint evaluation
    print(f"\nManual verification:")
    
    # Evaluate curve at same points used in LP constraint
    num_eval_points = 200  # Smaller than LP for debug
    t_vals = np.linspace(0.0, 1.0, num_eval_points)
    curve_points = general_bezier_curve(t_vals, final_ctrl)
    
    lp_residuals = []
    for i, t in enumerate(t_vals):
        curve_x = curve_points[i, 0] 
        curve_y = curve_points[i, 1]
        
        # Find target y by interpolation (same as LP)
        target_y = np.interp(curve_x, test_data[:, 0], test_data[:, 1])
        residual = target_y - curve_y
        lp_residuals.append(residual)
    
    lp_residuals = np.array(lp_residuals)
    lp_max_error = np.max(np.abs(lp_residuals))
    lp_max_idx = np.argmax(np.abs(lp_residuals))
    
    print(f"Manual LP-style max error: {lp_max_error:.6e}")
    print(f"Manual max error at t={t_vals[lp_max_idx]:.3f}: curve_x={curve_points[lp_max_idx, 0]:.3f}, residual={lp_residuals[lp_max_idx]:.6e}")
    
    # Compare with direct data point errors
    print(f"\nDirect data evaluation:")
    curve_at_data = general_bezier_curve(np.linspace(0, 1, len(test_data)), final_ctrl)
    direct_residuals = test_data[:, 1] - curve_at_data[:, 1] 
    direct_max = np.max(np.abs(direct_residuals))
    print(f"Direct max error: {direct_max:.6e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"LP reported optimal:     {lp_max_error:.6e} (LP constraints)")
    print(f"Actual evaluation:       {max_error_eval:.6e} (standard error calc)")
    print(f"Direct curve evaluation: {direct_max:.6e}")
    print(f"Ratio (actual/LP):       {max_error_eval/lp_max_error:.1f}x")


if __name__ == "__main__":
    debug_lp_solution()