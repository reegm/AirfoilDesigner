#!/usr/bin/env python3
"""
Test script to demonstrate the new orthogonal distance + minmax optimization approach.
This script shows how to use the new error functions with your airfoil designer.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.core_logic import CoreProcessor
import os

def test_minmax_optimization():
    """Test the new minmax optimization with orthogonal distance."""
    
    # Initialize the core processor
    processor = CoreProcessor(logger_func=print)
    
    # You'll need to provide a path to an airfoil .dat file
    # For testing, let's assume you have a NACA airfoil or similar
    airfoil_file = "res/hqa-25-12.dat"  # Replace with actual path
    
    if not os.path.exists(airfoil_file):
        print("Please update the airfoil_file path in this script to test the functionality")
        print("Available error functions:")
        print("- 'icp': Original ICP approach (default)")
        print("- 'orthogonal_sum_squares': Orthogonal distance with sum of squares")
        print("- 'orthogonal_minmax': Orthogonal distance with minmax optimization")
        return
    
    # Load the airfoil data
    if not processor.load_airfoil_data_and_initialize_model(airfoil_file):
        print("Failed to load airfoil data")
        return
    
    # Test different error functions
    error_functions = ["icp", "orthogonal_sum_squares", "orthogonal_minmax"]
    results = {}
    
    for error_func in error_functions:
        print(f"\n{'='*50}")
        print(f"Testing with error function: {error_func}")
        print(f"{'='*50}")
        
        # Build single Bezier model with the current error function
        success = processor.build_single_bezier_model(
            regularization_weight=0.01,
            error_function=error_func,
            enforce_g2=False,  # Test without G2 first
            num_points_curve_error=500,
            te_vector_points=3
        )
        
        if success:
            # Get the results
            upper_poly = processor.single_bezier_upper_poly_sharp
            lower_poly = processor.single_bezier_lower_poly_sharp
            
            # Calculate error metrics for comparison
            from core.optimization_core import (
                calculate_all_orthogonal_distances, 
                calculate_single_bezier_fitting_error
            )
            
            # Orthogonal distances
            upper_distances, upper_max, upper_max_idx, _, _ = calculate_all_orthogonal_distances(
                processor.upper_data, upper_poly
            )
            lower_distances, lower_max, lower_max_idx, _, _ = calculate_all_orthogonal_distances(
                processor.lower_data, lower_poly
            )
            
            # ICP error for comparison
            icp_upper = calculate_single_bezier_fitting_error(
                upper_poly, processor.upper_data, error_function="icp", return_max_error=True
            )
            icp_lower = calculate_single_bezier_fitting_error(
                lower_poly, processor.lower_data, error_function="icp", return_max_error=True
            )
            
            # Store results
            results[error_func] = {
                'upper_poly': upper_poly,
                'lower_poly': lower_poly,
                'upper_ortho_max': upper_max,
                'lower_ortho_max': lower_max,
                'upper_ortho_mean': np.mean(upper_distances),
                'lower_ortho_mean': np.mean(lower_distances),
                'upper_icp_max': icp_upper[1] if isinstance(icp_upper, tuple) else icp_upper,
                'lower_icp_max': icp_lower[1] if isinstance(icp_lower, tuple) else icp_lower,
            }
            
            print(f"Upper surface - Max orthogonal distance: {upper_max:.6f}")
            print(f"Upper surface - Mean orthogonal distance: {np.mean(upper_distances):.6f}")
            print(f"Lower surface - Max orthogonal distance: {lower_max:.6f}")
            print(f"Lower surface - Mean orthogonal distance: {np.mean(lower_distances):.6f}")
            
        else:
            print(f"Failed to build model with {error_func}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON OF ERROR FUNCTIONS")
    print(f"{'='*60}")
    
    if results:
        print(f"{'Error Function':<20} {'Upper Max':<12} {'Lower Max':<12} {'Upper Mean':<12} {'Lower Mean':<12}")
        print("-" * 70)
        
        for error_func, data in results.items():
            print(f"{error_func:<20} {data['upper_ortho_max']:<12.6f} {data['lower_ortho_max']:<12.6f} "
                  f"{data['upper_ortho_mean']:<12.6f} {data['lower_ortho_mean']:<12.6f}")
        
        # Highlight the benefits of minmax
        if 'orthogonal_minmax' in results and 'icp' in results:
            minmax_upper_max = results['orthogonal_minmax']['upper_ortho_max']
            minmax_lower_max = results['orthogonal_minmax']['lower_ortho_max']
            icp_upper_max = results['icp']['upper_ortho_max']
            icp_lower_max = results['icp']['lower_ortho_max']
            
            print(f"\nMinmax vs ICP maximum error improvement:")
            print(f"Upper: {((icp_upper_max - minmax_upper_max) / icp_upper_max * 100):+.2f}%")
            print(f"Lower: {((icp_lower_max - minmax_lower_max) / icp_lower_max * 100):+.2f}%")

def test_g2_minmax():
    """Test G2 continuity with minmax optimization."""
    print(f"\n{'='*60}")
    print("TESTING G2 CONTINUITY WITH MINMAX")
    print(f"{'='*60}")
    
    processor = CoreProcessor(logger_func=print)
    airfoil_file = "path/to/your/airfoil.dat"  # Replace with actual path
    
    if not os.path.exists(airfoil_file):
        print("Please update the airfoil_file path to test G2 functionality")
        return
    
    if not processor.load_airfoil_data_and_initialize_model(airfoil_file):
        print("Failed to load airfoil data")
        return
    
    # Test G2 with minmax
    success = processor.build_single_bezier_model(
        regularization_weight=0.01,
        error_function="orthogonal_minmax",
        enforce_g2=True,  # Enable G2 continuity
        num_points_curve_error=500,
        te_vector_points=3
    )
    
    if success:
        print("G2 + Minmax optimization completed successfully!")
        # You could add curvature analysis here
    else:
        print("G2 + Minmax optimization failed")

if __name__ == "__main__":
    print("Orthogonal Distance + Minmax Optimization Test")
    print("=" * 60)
    
    test_minmax_optimization()
    test_g2_minmax()
    
    print(f"\n{'='*60}")
    print("USAGE SUMMARY")
    print(f"{'='*60}")
    print("To use the new optimization in your GUI:")
    print("1. Set error_function='orthogonal_minmax' for true minmax optimization")
    print("2. Set error_function='orthogonal_sum_squares' for orthogonal distance with sum of squares")
    print("3. All existing constraints (TE vectors, G2, etc.) are preserved")
    print("4. The minmax approach should give more uniform error distribution")
    print("5. Runtime may be longer but fitting quality should be better") 