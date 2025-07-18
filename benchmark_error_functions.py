#!/usr/bin/env python3
"""
Benchmark script to systematically compare error functions.
This provides detailed timing and error analysis for each method.
"""

import time
import numpy as np
import os
from core.core_logic import CoreProcessor
from core.optimization_core import calculate_all_orthogonal_distances

def benchmark_error_functions(airfoil_file="res/mh42.dat", regularization_weight=0.001):
    """
    Comprehensive benchmark of all error functions.
    
    Args:
        airfoil_file: Path to airfoil .dat file
        regularization_weight: Regularization weight to use for all tests
    """
    
    if not os.path.exists(airfoil_file):
        print(f"Airfoil file '{airfoil_file}' not found.")
        print("Please update the path or copy a .dat file to the specified location.")
        return
    
    # Test methods in order of expected speed
    methods = [
        ("icp", "ICP (Original)"),
        ("orthogonal_fast", "Orthogonal Fast"),
        ("orthogonal_conservative", "Orthogonal Conservative"), 
        ("orthogonal_sum_squares", "Orthogonal Sum Squares"),
    ]
    
    results = {}
    
    print("=" * 80)
    print(f"BENCHMARKING ERROR FUNCTIONS")
    print(f"Airfoil: {os.path.basename(airfoil_file)}")
    print(f"Regularization weight: {regularization_weight}")
    print("=" * 80)
    
    for method_id, method_name in methods:
        print(f"\n{'-' * 60}")
        print(f"Testing: {method_name}")
        print(f"Method ID: '{method_id}'")
        print(f"{'-' * 60}")
        
        processor = CoreProcessor(logger_func=lambda msg: None)  # Silent processor
        
        # Load airfoil
        load_success = processor.load_airfoil_data_and_initialize_model(airfoil_file)
        if not load_success:
            print(f"❌ Failed to load airfoil for {method_name}")
            results[method_id] = {"success": False, "error": "Failed to load airfoil"}
            continue
        
        # Time the optimization
        start_time = time.time()
        
        success = processor.build_single_bezier_model(
            regularization_weight=regularization_weight,
            error_function=method_id,
            enforce_g2=False,
            num_points_curve_error=500,
            te_vector_points=3
        )
        
        optimization_time = time.time() - start_time
        
        if success:
            # Calculate comprehensive error metrics
            upper_poly = processor.single_bezier_upper_poly_sharp
            lower_poly = processor.single_bezier_lower_poly_sharp
            
            try:
                # Calculate orthogonal distances for all methods (for fair comparison)
                print("  Calculating orthogonal distance metrics...")
                
                upper_distances, upper_max, upper_max_idx, _, _ = calculate_all_orthogonal_distances(
                    processor.upper_data, upper_poly
                )
                lower_distances, lower_max, lower_max_idx, _, _ = calculate_all_orthogonal_distances(
                    processor.lower_data, lower_poly
                )
                
                # Calculate statistics
                upper_mean = np.mean(upper_distances)
                upper_std = np.std(upper_distances)
                upper_95th = np.percentile(upper_distances, 95)
                
                lower_mean = np.mean(lower_distances)
                lower_std = np.std(lower_distances)
                lower_95th = np.percentile(lower_distances, 95)
                
                # Overall statistics
                overall_max = max(upper_max, lower_max)
                all_distances = np.concatenate([upper_distances, lower_distances])
                overall_mean = np.mean(all_distances)
                overall_95th = np.percentile(all_distances, 95)
                
                results[method_id] = {
                    "success": True,
                    "time": optimization_time,
                    "upper_max": upper_max,
                    "upper_mean": upper_mean,
                    "upper_std": upper_std,
                    "upper_95th": upper_95th,
                    "lower_max": lower_max,
                    "lower_mean": lower_mean,
                    "lower_std": lower_std,
                    "lower_95th": lower_95th,
                    "overall_max": overall_max,
                    "overall_mean": overall_mean,
                    "overall_95th": overall_95th,
                    "upper_poly": upper_poly,
                    "lower_poly": lower_poly,
                }
                
                print(f"  ✅ {method_name} completed successfully")
                print(f"     Time: {optimization_time:.2f}s")
                print(f"     Overall max error: {overall_max:.6f}")
                print(f"     Overall mean error: {overall_mean:.6f}")
                print(f"     Overall 95th percentile: {overall_95th:.6f}")
                
            except Exception as e:
                print(f"  ❌ Error calculating metrics for {method_name}: {e}")
                results[method_id] = {
                    "success": False, 
                    "error": f"Metrics calculation failed: {e}",
                    "time": optimization_time
                }
        else:
            print(f"  ❌ {method_name} optimization failed")
            results[method_id] = {
                "success": False, 
                "error": "Optimization failed",
                "time": optimization_time
            }
    
    # Summary comparison
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    successful_methods = [(k, v) for k, v in results.items() if v.get("success", False)]
    
    if successful_methods:
        # Sort by time for performance ranking
        successful_methods.sort(key=lambda x: x[1]["time"])
        
        print(f"\n{'Method':<20} {'Time (s)':<10} {'Max Error':<12} {'Mean Error':<12} {'95th %ile':<12}")
        print("-" * 75)
        
        baseline_time = None
        baseline_max_error = None
        
        for method_id, data in successful_methods:
            method_name = next(name for mid, name in methods if mid == method_id)
            
            time_val = data["time"]
            max_error = data["overall_max"]
            mean_error = data["overall_mean"]
            p95_error = data["overall_95th"]
            
            # Track baseline for comparison
            if method_id == "icp":
                baseline_time = time_val
                baseline_max_error = max_error
            
            print(f"{method_name:<20} {time_val:<10.2f} {max_error:<12.6f} {mean_error:<12.6f} {p95_error:<12.6f}")
        
        # Performance vs accuracy analysis
        if baseline_time and baseline_max_error:
            print(f"\n{'-' * 60}")
            print("PERFORMANCE vs ACCURACY ANALYSIS")
            print(f"{'-' * 60}")
            
            for method_id, data in successful_methods:
                if method_id == "icp":
                    continue
                    
                method_name = next(name for mid, name in methods if mid == method_id)
                
                time_ratio = data["time"] / baseline_time
                error_improvement = (baseline_max_error - data["overall_max"]) / baseline_max_error * 100
                
                print(f"{method_name}:")
                print(f"  Time vs ICP: {time_ratio:.1f}x slower")
                if error_improvement > 0:
                    print(f"  Error vs ICP: {error_improvement:.1f}% better")
                else:
                    print(f"  Error vs ICP: {abs(error_improvement):.1f}% worse")
                print()
    
    # Recommendations
    print(f"{'-' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'-' * 60}")
    
    if successful_methods:
        # Find best speed
        fastest = min(successful_methods, key=lambda x: x[1]["time"])
        fastest_name = next(name for mid, name in methods if mid == fastest[0])
        
        # Find best accuracy
        most_accurate = min(successful_methods, key=lambda x: x[1]["overall_max"])
        most_accurate_name = next(name for mid, name in methods if mid == most_accurate[0])
        
        # Find best balance (simple scoring)
        def balance_score(data):
            # Normalize time and error (lower is better for both)
            time_scores = [d["time"] for _, d in successful_methods]
            error_scores = [d["overall_max"] for _, d in successful_methods]
            
            norm_time = (data["time"] - min(time_scores)) / (max(time_scores) - min(time_scores))
            norm_error = (data["overall_max"] - min(error_scores)) / (max(error_scores) - min(error_scores))
            
            return norm_time + norm_error  # Lower is better
        
        best_balance = min(successful_methods, key=lambda x: balance_score(x[1]))
        best_balance_name = next(name for mid, name in methods if mid == best_balance[0])
        
        print(f"🏆 Fastest: {fastest_name} ({fastest[1]['time']:.2f}s)")
        print(f"🎯 Most Accurate: {most_accurate_name} (max error: {most_accurate[1]['overall_max']:.6f})")
        print(f"⚖️  Best Balance: {best_balance_name}")
        
        print(f"\n💡 For your GUI, try:")
        print(f"   - Use '{best_balance[0]}' for best overall balance")
        print(f"   - Use '{fastest[0]}' if speed is critical")
        print(f"   - Use '{most_accurate[0]}' if accuracy is critical")
    
    return results

if __name__ == "__main__":
    # Default test
    benchmark_error_functions()
    
    print(f"\n{'=' * 80}")
    print("To run with your own airfoil:")
    print("  python benchmark_error_functions.py")
    print("  # Then edit the airfoil_file path in the script")
    print(f"{'=' * 80}") 