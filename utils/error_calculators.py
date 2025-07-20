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
    # Vectorized distance calculation
    dists = np.linalg.norm(data_points[:, None, :] - curve_points[None, :, :], axis=2)
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
    # Automatically determine sampling type based on error function
    use_curvature_sampling = error_function in ["orthogonal_minmax", "orthogonal_icp"]
    
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
        num_points_curve = config.NUM_POINTS_CURVE_ERROR
        t_samples = np.linspace(0, 1, num_points_curve)
        
        curve_points = general_bezier_curve(t_samples, bezier_poly)
        curve_sorted = curve_points[np.argsort(curve_points[:, 0])]
        return calculate_euclidean_error(original_data, curve_sorted, return_max_error=return_max_error)

def resample_points_by_curvature(points, num_samples=200):
    """
    Resample a 2D point set (Nx2) so that points are more densely distributed in regions of high curvature.
    Uses chord-length parameterization and Bezier curvature as a proxy for local curvature.
    Returns a new (num_samples, 2) array.
    """
    points = np.asarray(points)
    if len(points) < 3 or num_samples <= len(points):
        # Not enough points or already dense, just return original (or linearly interpolated)
        t_orig = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, num_samples)
        return np.column_stack([
            np.interp(t_new, t_orig, points[:, 0]),
            np.interp(t_new, t_orig, points[:, 1])
        ])

    # Chord-length parameterization
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.zeros(len(points))
    t[1:] = np.cumsum(dists)
    t /= t[-1]

    # Fit a Bezier curve to the points for curvature estimation
    # Use degree = min(7, len(points)-1) for stability
    from numpy.polynomial.polynomial import Polynomial
    degree = min(7, len(points)-1)
    # Fit x(t) and y(t) polynomials (not a true Bezier, but sufficient for curvature proxy)
    px = Polynomial.fit(t, points[:, 0], degree)
    py = Polynomial.fit(t, points[:, 1], degree)
    t_dense = np.linspace(0, 1, max(400, 4*num_samples))
    x_dense = px(t_dense)
    y_dense = py(t_dense)
    curve_dense = np.column_stack([x_dense, y_dense])

    # Estimate curvature using finite differences
    dx = np.gradient(x_dense, t_dense)
    dy = np.gradient(y_dense, t_dense)
    ddx = np.gradient(dx, t_dense)
    ddy = np.gradient(dy, t_dense)
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5 + 1e-12
    curvature = np.abs(numerator / denominator)
    curvature += np.finfo(float).eps

    # PDF and CDF for adaptive sampling
    pdf = curvature / np.sum(curvature)
    cdf = np.cumsum(pdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    u_values = np.linspace(0, 1, num_samples)
    t_resampled = np.interp(u_values, cdf, t_dense)
    # Interpolate original points at these t values
    x_resampled = np.interp(t_resampled, t, points[:, 0])
    y_resampled = np.interp(t_resampled, t, points[:, 1])
    return np.column_stack([x_resampled, y_resampled]) 