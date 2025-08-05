import numpy as np
from utils.bezier_utils import general_bezier_curve, bezier_curvature


def curvature_based_sampling(points, num_points, curvature_weight=1.0):
    """
    Sample a Bezier curve with more points in high-curvature regions.
    
    Args:
        points (np.ndarray): Control points of the Bezier curve (N, 2).
        num_points (int): Total number of points to sample.
        curvature_weight (float): Weight factor for curvature influence (0.0 = uniform, 1.0 = full curvature-based).
    
    Returns:
        np.ndarray: Sampled points on the curve (num_points, 2).
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    
    # Generate a fine uniform sampling to calculate curvature
    fine_t = np.linspace(0, 1, max(100, num_points * 4))
    fine_curvature = np.abs(bezier_curvature(fine_t, points))
    
    # Normalize curvature to [0, 1] range
    if np.max(fine_curvature) > 0:
        normalized_curvature = fine_curvature / np.max(fine_curvature)
    else:
        normalized_curvature = np.ones_like(fine_curvature)
    
    # Create a cumulative distribution function based on curvature
    # Add a small constant to ensure we can sample everywhere
    curvature_density = normalized_curvature + 0.1
    cdf = np.cumsum(curvature_density)
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]
    
    # Generate non-uniform t values based on curvature
    if curvature_weight > 0:
        # Interpolate to find t values that give uniform spacing in the CDF
        uniform_cdf = np.linspace(0, 1, num_points)
        t_curvature = np.interp(uniform_cdf, cdf, fine_t)
        
        # Blend with uniform sampling based on curvature_weight
        t_uniform = np.linspace(0, 1, num_points)
        t_final = (1 - curvature_weight) * t_uniform + curvature_weight * t_curvature
    else:
        t_final = np.linspace(0, 1, num_points)
    
    # Sample the curve at the calculated t values
    sampled_points = general_bezier_curve(t_final, points)
    
    return sampled_points


def sample_airfoil_surfaces(upper_poly, lower_poly, points_per_surface, curvature_weight=0.7):
    """
    Sample both upper and lower airfoil surfaces using curvature-based sampling.
    
    Args:
        upper_poly (np.ndarray): Upper surface control points.
        lower_poly (np.ndarray): Lower surface control points.
        points_per_surface (int): Number of points to sample per surface.
        curvature_weight (float): Weight factor for curvature influence.
    
    Returns:
        tuple: (upper_sampled, lower_sampled) - Sampled points for each surface.
    """
    upper_sampled = curvature_based_sampling(upper_poly, points_per_surface, curvature_weight)
    lower_sampled = curvature_based_sampling(lower_poly, points_per_surface, curvature_weight)
    
    return upper_sampled, lower_sampled
