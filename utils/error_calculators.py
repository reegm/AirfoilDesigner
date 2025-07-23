import numpy as np
from core import config
from utils.bezier_utils import general_bezier_curve
from utils.bezier_optimization_utils import calculate_all_orthogonal_distances
from scipy.spatial import cKDTree

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

    # Use KDTree for efficient nearest neighbor search
    tree = cKDTree(resampled_curve_points)
    min_dists, min_idxs = tree.query(data_points, k=1)
    sum_sq = np.sum(min_dists ** 2)
    if return_max_error:
        max_error = np.max(min_dists)
        max_error_idx = int(np.argmax(min_dists))
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

# ----------------------------------------------------------------------------- 
# helpers: orthogonal (signed) distances  – update for shape (N,) -> (N,2)
# -----------------------------------------------------------------------------
def _orthogonal_signed_distances(bezier_poly: np.ndarray,
                                 data: np.ndarray) -> np.ndarray:
    """
    Signed orthogonal distance from each sample in *data* to *bezier_poly*.

    Works with the current signature of `calculate_all_orthogonal_distances`
    which returns five items:
        distances, max_distance, max_idx, projection_points, normals
    where *projection_points* and *normals* are 1-D object arrays of length N.
    """
    distances, _, _, proj_pts, normals = calculate_all_orthogonal_distances(
        data, bezier_poly
    )

    # --- ensure proj_pts is (N,2) float -------------------------------------
    proj_pts = np.asarray(proj_pts)
    if proj_pts.ndim == 1:                       # (N,) → stack to (N,2)
        proj_pts = np.vstack(proj_pts)

    # --- ensure normals is (N,2) float --------------------------------------
    normals = np.asarray(normals)
    if normals.ndim == 1:                        # may be (N,) magnitudes
        if normals.dtype == object or normals.size != distances.size:
            normals = np.vstack(normals)
        else:
            # Only magnitudes → fallback sign from y-difference
            vec_y = data[:, 1] - proj_pts[:, 1]
            return distances * np.sign(vec_y + 1e-16)

    # signed distance from dot-product with normals
    vec = data - proj_pts                       # (N,2)
    sign = np.sign(np.einsum("ij,ij->i", vec, normals))
    sign[sign == 0.0] = 1.0                     # avoid zeros
    return distances * sign                     # (N,)


# ----------------------------------------------------------------------------- 
# calculate_single_bezier_fitting_error  (NEW) --------------------------------
# -----------------------------------------------------------------------------
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
        signed = _orthogonal_signed_distances(bezier_poly, original_data)
        abs_vals = np.abs(signed)
        max_idx = int(np.argmax(abs_vals))
        max_err = float(abs_vals[max_idx])

        if return_all:
            rms = float(np.sqrt(np.mean(abs_vals ** 2)))
            return signed, rms, (max_err, max_idx)

        if return_max_error:
            # For pure min–max the metric IS the max error
            return max_err, max_err, max_idx

        return max_err

    elif error_function == "orthogonal_icp":
        # existing orthogonal-ICP code path (unchanged) -----------------------
        sum_sq_orthogonal = calculate_orthogonal_error_icp(bezier_poly,
                                                           original_data)
        if return_max_error:
            max_err, max_idx = calculate_orthogonal_error_minmax(bezier_poly,
                                                                 original_data)
            return sum_sq_orthogonal, max_err, max_idx
        return sum_sq_orthogonal

    else:  # 'euclidean' ------------------------------------------------------
        return calculate_euclidean_error(original_data,
                                         bezier_poly,
                                         return_max_error=return_max_error)
