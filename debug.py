#!/usr/bin/env python3
"""
Test script to compare different leading edge enforcement modes and their effect on fit quality.
This should help achieve the target <1e-5 errors by following the papers' approach.
"""

import numpy as np
from core.bspline_processor import BSplineProcessor
from core.error_functions import calculate_bspline_fitting_error

def test_le_enforcement_modes():
    """Test all leading edge enforcement modes and compare fit quality."""
    print("="*70)
    print("TESTING LEADING EDGE ENFORCEMENT MODES")
    print("="*70)
    
    # Create test data (NACA 0012-like)
    x = np.linspace(0, 1, 50)
    thickness = 0.12
    yt = 5 * thickness * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    upper_data = np.column_stack([x, yt])
    lower_data = np.column_stack([x, -yt])
    
    print(f"Test data: {len(upper_data)} points per surface")
    print(f"Target: Errors < 1e-5 (following Venkataraman paper results)")
    
    # Test different control point counts and enforcement modes
    control_point_counts = [8, 10, 12, 15]
    enforcement_modes = ["none", "position", "tangent"]
    
    results = {}
    
    for cp_count in control_point_counts:
        print(f"\n" + "-"*50)
        print(f"TESTING {cp_count} CONTROL POINTS")
        print("-"*50)
        
        results[cp_count] = {}
        
        for mode in enforcement_modes:
            print(f"\n🔧 Mode: {mode}")
            
            # Create processor and set mode
            processor = BSplineProcessor()
            processor.set_leading_edge_enforcement_mode(mode)
            
            # Fit B-splines
            success = processor.fit_bspline(upper_data, lower_data, cp_count)
            
            if success:
                # Calculate errors
                upper_error = calculate_bspline_fitting_error(
                    processor.upper_curve, upper_data,
                    error_function="euclidean", return_max_error=True
                )
                lower_error = calculate_bspline_fitting_error(
                    processor.lower_curve, lower_data,
                    error_function="euclidean", return_max_error=True
                )
                
                # Store results
                results[cp_count][mode] = {
                    "upper_max": upper_error[1],
                    "lower_max": lower_error[1],
                    "upper_rms": np.sqrt(upper_error[0] / len(upper_data)),
                    "lower_rms": np.sqrt(lower_error[0] / len(lower_data)),
                }
                
                # Print results
                print(f"   ✅ SUCCESS")
                print(f"   📊 Upper max error: {upper_error[1]:.6e}")
                print(f"   📊 Lower max error: {lower_error[1]:.6e}")
                print(f"   📊 Upper RMS error: {results[cp_count][mode]['upper_rms']:.6e}")
                print(f"   📊 Lower RMS error: {results[cp_count][mode]['lower_rms']:.6e}")
                
                # Check if we hit target
                max_error = max(upper_error[1], lower_error[1])
                if max_error < 1e-5:
                    print(f"   🎯 TARGET ACHIEVED! (< 1e-5)")
                elif max_error < 1e-4:
                    print(f"   🟡 Close to target (< 1e-4)")
                else:
                    print(f"   🔴 Above target (> 1e-4)")
                
                # Show leading edge positions
                le_upper = processor.upper_control_points[0]
                le_lower = processor.lower_control_points[0]
                print(f"   📍 LE positions: Upper({le_upper[0]:.6f}, {le_upper[1]:.6f}), Lower({le_lower[0]:.6f}, {le_lower[1]:.6f})")
                
            else:
                print(f"   ❌ FAILED")
                results[cp_count][mode] = {"error": "Fitting failed"}
    
    # Summary table
    print(f"\n" + "="*70)
    print("SUMMARY TABLE - MAX ERRORS")
    print("="*70)
    print(f"{'Control Points':<15} {'None':<12} {'Position':<12} {'Tangent':<12} {'Best Mode'}")
    print("-"*70)
    
    for cp_count in control_point_counts:
        row = f"{cp_count:<15}"
        best_error = float('inf')
        best_mode = "N/A"
        
        for mode in enforcement_modes:
            if mode in results[cp_count] and "upper_max" in results[cp_count][mode]:
                max_err = max(results[cp_count][mode]["upper_max"], results[cp_count][mode]["lower_max"])
                row += f"{max_err:<12.2e}"
                
                if max_err < best_error:
                    best_error = max_err
                    best_mode = mode
            else:
                row += f"{'FAILED':<12}"
        
        if best_error < 1e-5:
            best_indicator = f"🎯 {best_mode}"
        elif best_error < 1e-4:
            best_indicator = f"🟡 {best_mode}"
        else:
            best_indicator = f"🔴 {best_mode}"
            
        row += f"{best_indicator}"
        print(row)
    
    # Recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find best overall combination
    best_overall_error = float('inf')
    best_overall_cp = None
    best_overall_mode = None
    
    for cp_count in control_point_counts:
        for mode in enforcement_modes:
            if mode in results[cp_count] and "upper_max" in results[cp_count][mode]:
                max_err = max(results[cp_count][mode]["upper_max"], results[cp_count][mode]["lower_max"])
                if max_err < best_overall_error:
                    best_overall_error = max_err
                    best_overall_cp = cp_count
                    best_overall_mode = mode
    
    if best_overall_error < 1e-5:
        print(f"🎯 EXCELLENT: Use {best_overall_cp} control points with '{best_overall_mode}' mode")
        print(f"   Achieves {best_overall_error:.2e} error (target: < 1e-5)")
    elif best_overall_error < 1e-4:
        print(f"🟡 GOOD: Use {best_overall_cp} control points with '{best_overall_mode}' mode")
        print(f"   Achieves {best_overall_error:.2e} error (close to target)")
        print(f"   Consider increasing control points further for better accuracy")
    else:
        print(f"🔴 NEEDS IMPROVEMENT: Best is {best_overall_cp} control points with '{best_overall_mode}' mode")
        print(f"   Achieves {best_overall_error:.2e} error (target: < 1e-5)")
        print(f"   Try: More control points, different airfoil data, or algorithm improvements")
    
    print(f"\n📝 OBSERVATIONS:")
    print(f"   • 'none' mode: Pure least-squares fit (usually best accuracy)")
    print(f"   • 'position' mode: Minimal constraints (good balance)")
    print(f"   • 'tangent' mode: Full constraints (may reduce accuracy)")
    print(f"   • More control points generally improve accuracy")
    print(f"   • Venkataraman paper achieved 2-3e-05 errors with similar methods")

def test_with_real_airfoil_data():
    """Test with actual airfoil data if available."""
    print(f"\n" + "="*70)
    print("TESTING WITH REAL AIRFOIL DATA (if available)")
    print("="*70)
    
    # This would test with actual loaded airfoil data
    # You can replace this with your actual data loading code
    try:
        # Example: load your fx67k170.DAT or similar
        print("To test with real data:")
        print("1. Load your airfoil data (fx67k170.DAT)")
        print("2. Use processor.set_leading_edge_enforcement_mode('none')")
        print("3. Check if errors drop to < 1e-5 range")
        print("4. If not, try increasing control points to 15-20")
        
    except Exception as e:
        print(f"Real data test not available: {e}")

if __name__ == "__main__":
    test_le_enforcement_modes()
    test_with_real_airfoil_data()