#!/usr/bin/env python3
"""
Simple test script for orthogonal distance optimization.
This demonstrates the different error functions available.
"""

import os
from core.core_logic import CoreProcessor

def test_different_error_functions(airfoil_file="res/hqa-25-12.dat"):
    """Test different error functions and compare results."""
    
    if not os.path.exists(airfoil_file):
        print(f"Airfoil file '{airfoil_file}' not found.")
        print("Please update the path or use your own .dat file.")
        return
    
    # Available error functions in order of speed vs accuracy
    error_functions = [
        ("icp", "ICP (Original - 🌟 RECOMMENDED)"),  # Fastest and most reliable
        ("orthogonal_fast", "Orthogonal Distance (Fast - Experimental)"),
        ("orthogonal_conservative", "Orthogonal Distance (Conservative)"),
        ("orthogonal_sum_squares", "Orthogonal Distance (Sum of Squares)"),
        ("orthogonal_minmax", "Orthogonal Distance (Pure Minmax - Experimental)")
    ]
    
    print("Testing Different Error Functions")
    print("=" * 50)
    
    for error_func, description in error_functions:
        print(f"\n{'-' * 30}")
        print(f"Testing: {description}")
        print(f"Error function: '{error_func}'")
        print(f"{'-' * 30}")
        
        processor = CoreProcessor(logger_func=print)
        
        # Load airfoil
        if not processor.load_airfoil_data_and_initialize_model(airfoil_file):
            print(f"Failed to load airfoil for {error_func}")
            continue
        
        # Build model
        success = processor.build_single_bezier_model(
            regularization_weight=0.01,
            error_function=error_func,
            enforce_g2=False,
            num_points_curve_error=500,
            te_vector_points=3
        )
        
        if success:
            print(f"✓ {description} completed successfully")
            print(f"  Upper surface max error: {processor.last_single_bezier_upper_max_error:.6f}")
            print(f"  Lower surface max error: {processor.last_single_bezier_lower_max_error:.6f}")
        else:
            print(f"✗ {description} failed")
    
    print(f"\n{'=' * 50}")
    print("RECOMMENDATIONS:")
    print("=" * 50)
    print("1. Use 'icp' for fastest, most reliable results 🌟")
    print("2. Use 'orthogonal_conservative' for better accuracy (slower)")
    print("3. Use 'orthogonal_sum_squares' for best accuracy (very slow)")
    print("4. Use 'orthogonal_fast' and 'orthogonal_minmax' are experimental")
    print("\nIn your GUI, simply change the error_function parameter:")
    print("processor.build_single_bezier_model(error_function='icp')  # Recommended")
    print("\nCurrent performance observations:")
    print("- 'icp': ~5-10 seconds, good results")
    print("- 'orthogonal_conservative': ~10-30 seconds, better accuracy")
    print("- 'orthogonal_sum_squares': ~100-200 seconds, best accuracy")
    print("- Run benchmark_error_functions.py for detailed comparison")

if __name__ == "__main__":
    test_different_error_functions() 