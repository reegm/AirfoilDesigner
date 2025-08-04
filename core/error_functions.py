import numpy as np
from scipy.spatial import cKDTree
from core import config
from utils.bezier_utils import general_bezier_curve
from utils.bezier_optimization_utils import calculate_all_orthogonal_distances_optimized

def calculate_single_bezier_fitting_error(
        bezier_poly: np.ndarray,
        original_data: np.ndarray,
        *,
        error_function: str = "euclidean",
        return_max_error: bool = False,
        return_all: bool = False):
    """
    Unified error calculator.
    """
    def soft_max(errors: np.ndarray, alpha: float = config.SOFTMAX_ALPHA):
        abs_errors = np.abs(errors)
        return np.log(np.sum(np.exp(alpha * abs_errors))) / alpha

    if error_function == "euclidean":
        num_points_curve = config.NUM_POINTS_CURVE_ERROR
        t_samples = np.linspace(0, 1, num_points_curve)
        sampled_curve_points = general_bezier_curve(t_samples, bezier_poly)
        sampled_curve_points = sampled_curve_points[np.argsort(sampled_curve_points[:, 0])]
        tree = cKDTree(sampled_curve_points)
        min_dists, min_idxs = tree.query(original_data, k=1)
        sum_sq = np.sum(min_dists ** 2)
        if return_all:
            rms = float(np.sqrt(np.mean(min_dists ** 2)))
            return min_dists, rms, (sum_sq, int(np.argmax(min_dists)))
        if return_max_error:
            max_error = np.max(min_dists)
            max_error_idx = int(np.argmax(min_dists))
            return sum_sq, max_error, max_error_idx
        return sum_sq

    elif error_function == "orthogonal_minmax":
        distances, max_distance, max_distance_idx, proj_pts, normals = calculate_all_orthogonal_distances_optimized(
            original_data, bezier_poly
        )
        abs_vals = np.abs(distances)
        soft_max_err = soft_max(distances)
        max_idx = int(np.argmax(abs_vals))
        max_err = float(abs_vals[max_idx])
        if return_all:
            rms = float(np.sqrt(np.mean(abs_vals ** 2)))
            return distances, rms, (soft_max_err, max_idx)
        if return_max_error:
            return soft_max_err, max_err, max_idx
        return soft_max_err

    elif error_function == "orthogonal":
        distances, _, _, _, _ = calculate_all_orthogonal_distances_optimized(
            original_data, bezier_poly
        )
        sum_sq = np.sum(distances ** 2)
        if return_all:
            rms = float(np.sqrt(np.mean(distances ** 2)))
            return distances, rms, (sum_sq, int(np.argmax(np.abs(distances))))
        if return_max_error:
            max_err = np.max(np.abs(distances))
            max_idx = int(np.argmax(np.abs(distances)))
            return sum_sq, max_err, max_idx
        return sum_sq

    elif error_function == "euclidean_minmax":
        num_points_curve = config.NUM_POINTS_CURVE_ERROR
        t_samples = np.linspace(0, 1, num_points_curve)
        sampled_curve_points = general_bezier_curve(t_samples, bezier_poly)
        sampled_curve_points = sampled_curve_points[np.argsort(sampled_curve_points[:, 0])]
        tree = cKDTree(sampled_curve_points)
        min_dists, min_idxs = tree.query(original_data, k=1)
        signed_dists = original_data[:, 1] - sampled_curve_points[min_idxs, 1]
        abs_vals = np.abs(signed_dists)
        soft_max_err = soft_max(signed_dists)
        max_idx = int(np.argmax(abs_vals))
        max_err = float(abs_vals[max_idx])
        if return_all:
            rms = float(np.sqrt(np.mean(abs_vals ** 2)))
            return signed_dists, rms, (soft_max_err, max_idx)
        if return_max_error:
            return soft_max_err, max_err, max_idx
        return soft_max_err

    else:
        return calculate_single_bezier_fitting_error(original_data, bezier_poly, return_max_error=return_max_error)
