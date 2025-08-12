#!/usr/bin/env python3
"""
Simple test of the Chebyshev LP solver
"""

import numpy as np
import time
from core.chebyshev_lp_optimizer import optimize_fixed_x_chebyshev_lp
from core.bezier_optimizer import build_bezier_fixed_x_softmax
from core.error_functions import calculate_single_bezier_fitting_error


def generate_simple_test_data():
    """Generate simple test data."""
    x = np.linspace(0, 1, 50)
    y = 0.1 * np.sin(2 * np.pi * x) + 0.05 * x
    return np.column_stack([x, y])


def simple_test():
    """Simple comparison test."""
    
    print("=== Simple Chebyshev LP Test ===")
    
    test_data = generate_simple_test_data()
    num_control_points = 6
    is_upper_surface = True
    te_tangent_vector = np.array([1.0, 0.0])
    
    print(f"Test data: {len(test_data)} points, {num_control_points} control points")
    
    # Test LP solver
    print("\n1. Testing LP solver...")
    start_time = time.time()
    
    def simple_logger(msg):
        print(f"   {msg}")
    
    ctrl_lp = optimize_fixed_x_chebyshev_lp(
        original_data=test_data,
        num_control_points_new=num_control_points,
        is_upper_surface=is_upper_surface,
        te_tangent_vector=te_tangent_vector,
        regularization_weight=0.0,  # No regularization for simplicity
        logger_func=simple_logger
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
    
    # Test softmax solver  
    print("\n2. Testing Softmax solver...")
    start_time = time.time()
    
    ctrl_softmax = build_bezier_fixed_x_softmax(
        original_data=test_data,
        num_control_points_new=num_control_points,
        is_upper_surface=is_upper_surface,
        le_tangent_vector=np.array([0.0, 1.0]),
        te_tangent_vector=te_tangent_vector,
        regularization_weight=0.0,
        logger_func=None  # Disable logging for softmax
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
    
    print(f"\n=== Results ===")
    print(f"LP Time:        {lp_time:.3f}s")
    print(f"Softmax Time:   {softmax_time:.3f}s")
    print(f"Speedup:        {softmax_time/lp_time:.2f}x")
    print()
    print(f"LP Max Error:      {lp_max_error:.6e}")
    print(f"Softmax Max Error: {softmax_max_error:.6e}")  
    print(f"LP Better by:      {((softmax_max_error/lp_max_error - 1)*100):+.1f}%")
    print()
    print("Notes:")
    print("- LP finds the TRUE optimal Chebyshev solution")
    print("- Softmax approximates the Chebyshev solution")
    print("- LP should have equal or better max error")


if __name__ == "__main__":
    simple_test()