#!/usr/bin/env python3
"""
Test script to compare Chebyshev LP solver with current softmax approach.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from core.chebyshev_lp_optimizer import optimize_fixed_x_chebyshev_lp
from core.bezier_optimizer import build_bezier_fixed_x_softmax
from core.error_functions import calculate_single_bezier_fitting_error
from core import config


def generate_test_airfoil_data():
    """Generate synthetic airfoil data for testing."""
    # Simple NACA-like airfoil shape
    x = np.linspace(0, 1, 100)
    thickness = 0.12
    y_upper = thickness * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    # Add some noise to make it realistic
    np.random.seed(42)
    y_upper += np.random.normal(0, 0.001, len(y_upper))
    
    return np.column_stack([x, y_upper])


def test_comparison():
    """Compare LP solver vs softmax approach."""
    
    print("=== Chebyshev LP vs Softmax Comparison ===\n")
    
    # Generate test data
    test_data = generate_test_airfoil_data()
    num_control_points = 8
    is_upper_surface = True
    te_tangent_vector = np.array([1.0, 0.0])
    regularization_weight = 0.01
    
    print(f"Test data: {len(test_data)} points")
    print(f"Control points: {num_control_points}")
    print(f"Regularization weight: {regularization_weight}")
    print()
    
    # Test LP solver
    print("Testing Chebyshev LP solver...")
    start_time = time.time()
    
    try:
        ctrl_lp = optimize_fixed_x_chebyshev_lp(
            original_data=test_data,
            num_control_points_new=num_control_points,
            is_upper_surface=is_upper_surface,
            te_tangent_vector=te_tangent_vector,
            regularization_weight=regularization_weight,
            logger_func=lambda msg: print(f"  LP: {msg}")
        )
        lp_time = time.time() - start_time
        
        if ctrl_lp is not None:
            # Evaluate LP result
            _, lp_rms, lp_max_info = calculate_single_bezier_fitting_error(
                ctrl_lp, test_data, error_function="euclidean", return_all=True)
            lp_max_error = np.max(np.abs(_))
            
            print(f"  + LP Success: {lp_time:.3f}s")
            print(f"  LP RMS error: {lp_rms:.6e}")
            print(f"  LP Max error: {lp_max_error:.6e}")
        else:
            print("  - LP Failed")
            lp_rms, lp_max_error = float('inf'), float('inf')
            
    except Exception as e:
        print(f"  - LP Error: {e}")
        lp_time = float('inf')
        lp_rms, lp_max_error = float('inf'), float('inf')
        ctrl_lp = None
    
    print()
    
    # Test current softmax approach
    print("Testing current softmax approach...")
    start_time = time.time()
    
    try:
        ctrl_softmax = build_bezier_fixed_x_softmax(
            original_data=test_data,
            num_control_points_new=num_control_points,
            is_upper_surface=is_upper_surface,
            le_tangent_vector=np.array([0.0, 1.0]),  # Not used in fixed-x
            te_tangent_vector=te_tangent_vector,
            regularization_weight=regularization_weight,
            logger_func=lambda msg: print(f"  Softmax: {msg}")
        )
        softmax_time = time.time() - start_time
        
        if ctrl_softmax is not None:
            # Evaluate softmax result
            _, softmax_rms, softmax_max_info = calculate_single_bezier_fitting_error(
                ctrl_softmax, test_data, error_function="euclidean", return_all=True)
            softmax_max_error = np.max(np.abs(_))
            
            print(f"  + Softmax Success: {softmax_time:.3f}s")  
            print(f"  Softmax RMS error: {softmax_rms:.6e}")
            print(f"  Softmax Max error: {softmax_max_error:.6e}")
        else:
            print("  - Softmax Failed")
            softmax_rms, softmax_max_error = float('inf'), float('inf')
            
    except Exception as e:
        print(f"  - Softmax Error: {e}")
        softmax_time = float('inf')
        softmax_rms, softmax_max_error = float('inf'), float('inf')
        ctrl_softmax = None
    
    print()
    print("=== COMPARISON RESULTS ===")
    print(f"LP Time:       {lp_time:.3f}s")
    print(f"Softmax Time:  {softmax_time:.3f}s")
    print(f"Speedup:       {softmax_time/lp_time:.1f}x" if lp_time != float('inf') else "N/A")
    print()
    print(f"LP Max Error:      {lp_max_error:.6e}")
    print(f"Softmax Max Error: {softmax_max_error:.6e}")
    print(f"LP Better by:      {(softmax_max_error/lp_max_error - 1)*100:.1f}%" if lp_max_error != float('inf') else "N/A")
    print()
    print(f"LP RMS Error:      {lp_rms:.6e}")
    print(f"Softmax RMS Error: {softmax_rms:.6e}")
    print(f"RMS Difference:    {(softmax_rms/lp_rms - 1)*100:.1f}%" if lp_rms != float('inf') else "N/A")
    
    # Plot comparison if both succeeded
    if ctrl_lp is not None and ctrl_softmax is not None:
        plt.figure(figsize=(12, 8))
        
        # Plot original data
        plt.subplot(2, 2, 1)
        plt.plot(test_data[:, 0], test_data[:, 1], 'k.', markersize=2, label='Original Data')
        
        # Plot LP result
        from utils.bezier_utils import general_bezier_curve
        t_vals = np.linspace(0, 1, 200)
        lp_curve = general_bezier_curve(t_vals, ctrl_lp)
        plt.plot(lp_curve[:, 0], lp_curve[:, 1], 'b-', linewidth=2, label='LP Solution')
        plt.plot(ctrl_lp[:, 0], ctrl_lp[:, 1], 'bo', markersize=4, label='LP Control Points')
        
        plt.title(f'Chebyshev LP Solution (Max Error: {lp_max_error:.6e})')
        plt.xlabel('x')
        plt.ylabel('y') 
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot softmax result
        plt.subplot(2, 2, 2)
        plt.plot(test_data[:, 0], test_data[:, 1], 'k.', markersize=2, label='Original Data')
        
        softmax_curve = general_bezier_curve(t_vals, ctrl_softmax)
        plt.plot(softmax_curve[:, 0], softmax_curve[:, 1], 'r-', linewidth=2, label='Softmax Solution')
        plt.plot(ctrl_softmax[:, 0], ctrl_softmax[:, 1], 'ro', markersize=4, label='Softmax Control Points')
        
        plt.title(f'Softmax Solution (Max Error: {softmax_max_error:.6e})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot error comparison
        plt.subplot(2, 2, 3)
        lp_residuals, _, _ = calculate_single_bezier_fitting_error(
            ctrl_lp, test_data, error_function="euclidean", return_all=True)
        softmax_residuals, _, _ = calculate_single_bezier_fitting_error(
            ctrl_softmax, test_data, error_function="euclidean", return_all=True)
        
        plt.plot(test_data[:, 0], np.abs(lp_residuals), 'b-', label=f'LP |Error|', alpha=0.7)
        plt.plot(test_data[:, 0], np.abs(softmax_residuals), 'r-', label=f'Softmax |Error|', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('|Error|')
        plt.title('Absolute Error Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot both curves together
        plt.subplot(2, 2, 4)
        plt.plot(test_data[:, 0], test_data[:, 1], 'k.', markersize=2, label='Original Data')
        plt.plot(lp_curve[:, 0], lp_curve[:, 1], 'b-', linewidth=2, label='LP Solution', alpha=0.8)
        plt.plot(softmax_curve[:, 0], softmax_curve[:, 1], 'r--', linewidth=2, label='Softmax Solution', alpha=0.8)
        
        plt.title('Both Solutions Overlaid')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chebyshev_lp_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved as 'chebyshev_lp_comparison.png'")


if __name__ == "__main__":
    test_comparison()