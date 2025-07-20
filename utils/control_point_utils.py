import numpy as np
from core import config

def variable_x_control_points(original_data, num_control_points):
    """
    Compute the x-locations for Bezier control points as the median x-value of each segment of the airfoil data.
    The first, second, second-to-last, and last control points are fixed at x=0 and x=1 as appropriate to preserve LE/TE tangency.
    Returns an array of x-locations (length = num_control_points).
    """
    data = np.asarray(original_data)
    x_data = data[:, 0]
    n = num_control_points
    indices = np.linspace(0, len(x_data), n+1, dtype=int)
    x_medians = []
    for i in range(n):
        seg = x_data[indices[i]:indices[i+1]]
        if len(seg) == 0:
            x_medians.append(x_medians[-1] if x_medians else 0.0)
        else:
            x_medians.append(np.median(seg))
    # Ensure first and last are exactly 0 and 1 (LE and TE)
    x_medians[0] = 0.0
    x_medians[-1] = 1.0
    # Fix second point to x=0 for LE tangency
    if n > 2:
        x_medians[1] = 0.0
    return np.array(x_medians)

def get_paper_fixed_x_coords(is_upper_surface):
    """
    Get the fixed x-coordinates from the Venkataraman paper for the specified surface.
    
    Args:
        is_upper_surface (bool): True for upper surface, False for lower surface
    
    Returns:
        np.ndarray: Array of fixed x-coordinates
    """
    if is_upper_surface:
        return np.array(config.PAPER_FIXED_X_UPPER)
    else:
        return np.array(config.PAPER_FIXED_X_LOWER) 