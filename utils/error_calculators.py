import numpy as np
from core import config
from utils.bezier_utils import general_bezier_curve
from utils.bezier_optimization_utils import calculate_all_orthogonal_distances

def calculate_euclidean_error(data_points, curve_points, return_max_error=False):
    """
    Calculates the sum of squared Euclidean distances from each data point to the closest point on the curve.
    If return_max_error is True, also returns the maximum pointwise error (not squared) and its index.
    Args:
        data_points (np.ndarray): (N, 2) array of data points.
        curve_points (np.ndarray): (M, 2) array of points sampled along the curve.
        return_max_error (bool): If True, also return the maximum pointwise error and its index.
    Returns:
        float or (float, float, int): Sum of squared distances, and optionally the max pointwise error and its index.
    """
    # Sample points along the Bezier curve for error calculation
    num_points_curve = config.NUM_POINTS_CURVE_ERROR
    t_samples = np.linspace(0, 1, num_points_curve)
    resampled_curve_points = general_bezier_curve(t_samples, curve_points)
    resampled_curve_points = resampled_curve_points[np.argsort(resampled_curve_points[:, 0])]

    # Vectorized distance calculation
    dists = np.linalg.norm(data_points[:, None, :] - resampled_curve_points[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    sum_sq = np.sum(min_dists ** 2)
    if return_max_error:
        max_error = np.max(min_dists)
        max_error_idx = np.argmax(min_dists)
        return sum_sq, max_error, max_error_idx
    return sum_sq

def calculate_orthogonal_error_minmax(control_points, original_data):
    """
    Calculate the maximum orthogonal distance error for minmax optimization.
    
    Args:
        control_points: np.array of shape (n+1, 2) - Bezier control points
        original_data: np.array of shape (N, 2) - original airfoil data points
    
    Returns:
        tuple: (max_error, max_error_idx) or float (max_error) depending on usage
    """
    # Calculate orthogonal distances from original_data to the curve
    distances, max_distance, max_distance_idx, _, _ = calculate_all_orthogonal_distances(
        original_data, control_points
    )
    return max_distance, max_distance_idx

def calculate_orthogonal_error_icp(control_points, original_data):
    """
    Calculate the sum of squared orthogonal distances for ICP-style optimization.
    
    Args:
        control_points: np.array of shape (n+1, 2) - Bezier control points
        original_data: np.array of shape (N, 2) - original airfoil data points
    
    Returns:
        float: Sum of squared orthogonal distances
    """
    distances, _, _, _, _ = calculate_all_orthogonal_distances(
        original_data, control_points
    )
    return np.sum(distances ** 2)

def calculate_single_bezier_fitting_error(bezier_poly, original_data, error_function="euclidean", return_max_error=False):
    """
    Calculates the fitting error between a Bezier curve and the original data using the specified error function.
    Supported error functions: 'euclidean', 'orthogonal_minmax', 'orthogonal_icp'
    If return_max_error is True, returns (sum, max_error, max_error_idx).
    """
    
    if error_function == "orthogonal_minmax":
        max_error, max_error_idx = calculate_orthogonal_error_minmax(bezier_poly, original_data)
        if return_max_error:
            return max_error, max_error, max_error_idx  # For minmax, sum and max are the same
        return max_error
    
    elif error_function == "orthogonal_icp":
        # Use sum of squared orthogonal distances (ICP-style with orthogonal distance)
        sum_sq_orthogonal = calculate_orthogonal_error_icp(bezier_poly, original_data)
        if return_max_error:
            # Also calculate max error for consistency
            max_error, max_error_idx = calculate_orthogonal_error_minmax(bezier_poly, original_data)
            return sum_sq_orthogonal, max_error, max_error_idx
        return sum_sq_orthogonal
    
    else:  # Euclidean (default) - euclidean distance
        return calculate_euclidean_error(original_data, bezier_poly, return_max_error=return_max_error)