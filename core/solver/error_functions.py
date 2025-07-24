import numpy as np
from scipy.spatial import cKDTree
from core import config
from utils.bezier_utils import general_bezier_curve
from utils.bezier_optimization_utils import calculate_all_orthogonal_distances_optimized

def calculate_euclidean_error(data_points, control_points, return_max_error=False):
    """
    Calculates the sum of squared Euclidean distances from each data point to the closest point on the Bezier curve.
    The Bezier curve is sampled at NUM_POINTS_CURVE_ERROR points.
    If return_max_error is True, also returns the maximum pointwise error (not squared) and its index.
    Args:
        data_points (np.ndarray): (N, 2) array of data points.
        control_points (np.ndarray): (M, 2) array of Bezier control points.
        return_max_error (bool): If True, also return the maximum pointwise error and its index.
    Returns:
        float or (float, float, int): Sum of squared distances, and optionally the max pointwise error and its index.
    """
    num_points_curve = config.NUM_POINTS_CURVE_ERROR
    t_samples = np.linspace(0, 1, num_points_curve)
    sampled_curve_points = general_bezier_curve(t_samples, control_points)
    sampled_curve_points = sampled_curve_points[np.argsort(sampled_curve_points[:, 0])]
    tree = cKDTree(sampled_curve_points)
    min_dists, min_idxs = tree.query(data_points, k=1)
    sum_sq = np.sum(min_dists ** 2)
    if return_max_error:
        max_error = np.max(min_dists)
        max_error_idx = int(np.argmax(min_dists))
        return sum_sq, max_error, max_error_idx
    return sum_sq

def calculate_orthogonal_error(data_points, control_points, return_max_error=False):
    """
    Calculates the sum of squared orthogonal distances from each data point to the Bezier curve.
    If return_max_error is True, also returns the maximum orthogonal error and its index.
    Args:
        data_points (np.ndarray): (N, 2) array of data points.
        control_points (np.ndarray): (M, 2) array of Bezier control points.
        return_max_error (bool): If True, also return the maximum orthogonal error and its index.
    Returns:
        float or (float, float, int): Sum of squared orthogonal distances, and optionally the max orthogonal error and its index.
    """
    distances, max_distance, max_distance_idx, _, _ = calculate_all_orthogonal_distances_optimized(data_points, control_points)
    sum_sq = np.sum(distances ** 2)
    if return_max_error:
        return sum_sq, max_distance, max_distance_idx
    return sum_sq

def calculate_single_bezier_fitting_error(
        bezier_poly: np.ndarray,
        original_data: np.ndarray,
        *,
        error_function: str = "euclidean",
        return_max_error: bool = False,
        return_all: bool = False):
    """
    Unified error calculator.

    Parameters
    ----------
    bezier_poly : (N, 2) array
    original_data : (M, 2) array
    error_function : {'euclidean', 'orthogonal_minmax', 'orthogonal_icp'}
    return_max_error : bool, optional
        If True, also return (max_abs_error, max_idx).
    return_all : bool, optional
        If True, return the *signed* residual vector (orthogonal_minmax only).

    Returns
    -------
    float
        sum-of-squares (euclidean / icp)   *or*
        max_abs_error   (orthogonal_minmax)
    When return_max_error=True:
        (float, int) appended  →  (metric, max_err, max_idx)
    When return_all=True **and** error_function=='orthogonal_minmax':
        np.ndarray (signed residuals), rms, (max_err, max_idx)
    """
    if error_function == "orthogonal_minmax":
        # Signed orthogonal distances for minmax
        # (legacy: _orthogonal_signed_distances)
        distances, max_distance, max_distance_idx, proj_pts, normals = calculate_all_orthogonal_distances_optimized(
            original_data, bezier_poly
        )
        abs_vals = np.abs(distances)
        max_idx = int(np.argmax(abs_vals))
        max_err = float(abs_vals[max_idx])
        if return_all:
            rms = float(np.sqrt(np.mean(abs_vals ** 2)))
            return distances, rms, (max_err, max_idx)
        if return_max_error:
            return max_err, max_err, max_idx
        return max_err
    elif error_function == "orthogonal_icp":
        distances, _, _, _, _ = calculate_all_orthogonal_distances_optimized(
            original_data, bezier_poly
        )
        sum_sq = np.sum(distances ** 2)
        if return_max_error:
            max_err = np.max(np.abs(distances))
            max_idx = int(np.argmax(np.abs(distances)))
            return sum_sq, max_err, max_idx
        return sum_sq
    else:
        # Default: use modular error functions
        if error_function == "orthogonal":
            return calculate_orthogonal_error(original_data, bezier_poly, return_max_error=return_max_error)
        else:
            return calculate_euclidean_error(original_data, bezier_poly, return_max_error=return_max_error) 