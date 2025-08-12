#!/usr/bin/env python3
"""
Test the integrated Chebyshev LP solver in the unified optimizer.
"""

import numpy as np
import time
from core.bezier_unified_optimizer import optimize_bezier
from core.error_functions import calculate_single_bezier_fitting_error


def test_unified_chebyshev():
    """Test the unified optimizer with Chebyshev objective."""
    
    print("=== Testing Unified Optimizer with Chebyshev LP ===")
    
    # Generate test data
    x = np.linspace(0, 1, 30)
    y = 0.1 * np.sin(2 * np.pi * x) + 0.05 * x
    test_data = np.column_stack([x, y])
    
    num_control_points = 6
    is_upper_surface = True
    te_tangent_vector = np.array([1.0, 0.0])
    te_y = float(test_data[-1, 1])
    
    print(f"Test data: {len(test_data)} points, {num_control_points} control points")
    
    # Test 1: Chebyshev LP solver
    print("\n1. Testing Chebyshev (LP) via unified optimizer...")
    start_time = time.time()
    
    def simple_logger(msg):
        print(f"   {msg}")
    
    try:
        ctrl_lp = optimize_bezier(
            initial_ctrl=None,
            original_data=test_data,
            mode="fixed-x",
            coupled=False,
            error_function="euclidean",
            objective="chebyshev",  # Use our new LP solver
            te_y=te_y,
            te_tangent_vector=te_tangent_vector,
            regularization_weight=0.0,
            logger_func=simple_logger,
            abort_flag=None,
            is_upper_surface=is_upper_surface,
            num_control_points_new=num_control_points
        )
        
        lp_time = time.time() - start_time
        
        if ctrl_lp is not None:
            lp_residuals, lp_rms, _ = calculate_single_bezier_fitting_error(
                ctrl_lp, test_data, error_function="euclidean", return_all=True)
            lp_max_error = np.max(np.abs(lp_residuals))
            print(f"   LP Success in {lp_time:.3f}s")
            print(f"   LP Max error: {lp_max_error:.6e}")
            print(f"   LP RMS error: {lp_rms:.6e}")
        else:
            print(f"   LP Failed in {lp_time:.3f}s")
            return
            
    except Exception as e:
        print(f"   LP Error: {e}")
        return
    
    # Test 2: Softmax for comparison
    print("\n2. Testing Softmax via unified optimizer...")
    start_time = time.time()
    
    try:
        ctrl_softmax = optimize_bezier(
            initial_ctrl=None,
            original_data=test_data,
            mode="fixed-x",
            coupled=False,
            error_function="euclidean",
            objective="softmax",  # Traditional softmax
            te_y=te_y,
            te_tangent_vector=te_tangent_vector,
            regularization_weight=0.0,
            logger_func=None,  # Disable logging
            abort_flag=None,
            is_upper_surface=is_upper_surface,
            num_control_points_new=num_control_points
        )
        
        softmax_time = time.time() - start_time
        
        if ctrl_softmax is not None:
            softmax_residuals, softmax_rms, _ = calculate_single_bezier_fitting_error(
                ctrl_softmax, test_data, error_function="euclidean", return_all=True)
            softmax_max_error = np.max(np.abs(softmax_residuals))
            print(f"   Softmax Success in {softmax_time:.3f}s")
            print(f"   Softmax Max error: {softmax_max_error:.6e}")
            print(f"   Softmax RMS error: {softmax_rms:.6e}")
        else:
            print(f"   Softmax Failed in {softmax_time:.3f}s")
            return
            
    except Exception as e:
        print(f"   Softmax Error: {e}")
        return
    
    print(f"\n=== COMPARISON ===")
    print(f"LP Time:        {lp_time:.3f}s")
    print(f"Softmax Time:   {softmax_time:.3f}s")  
    print(f"Speedup:        {softmax_time/lp_time:.2f}x")
    print()
    print(f"LP Max Error:      {lp_max_error:.6e}")
    print(f"Softmax Max Error: {softmax_max_error:.6e}")
    print(f"LP Better by:      {((softmax_max_error/lp_max_error - 1)*100):+.1f}%")
    print()
    print("SUCCESS: Chebyshev LP integrated into unified optimizer!")


if __name__ == "__main__":
    test_unified_chebyshev()