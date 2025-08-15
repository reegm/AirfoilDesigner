import numpy as np
from scipy.special import comb

def general_bezier_curve(t, points):
    """
    Calculates points on a Bezier curve of any order.

    Args:
        t (np.ndarray or float): Parameter value(s) between 0 and 1.
        points (np.ndarray): 2D array of control points (num_points, 2).

    Returns:
        np.ndarray: Points on the Bezier curve corresponding to t values.
    """
    n = len(points) - 1
    # Ensure t is a 1D NumPy array for vectorized operations
    if not isinstance(t, np.ndarray):
        t = np.array([t])
    elif t.ndim > 1:
        t = t.flatten()

    # Calculate binomial coefficients for Bernstein basis functions
    binom_coeffs = comb(n, np.arange(n + 1))

    # Compute Bernstein basis functions for all t values: B_i,n(t) = C(n,i) * (1-t)^(n-i) * t^i
    basis_functions = np.array([binom_coeffs[i] * (1 - t)**(n - i) * t**i for i in range(n + 1)]).T

    # Compute the curve points as a dot product of basis functions and control points
    return basis_functions @ points

def bezier_derivative(t, points, order=1):
    """
    Calculates the derivative of a Bezier curve at parameter t.

    Args:
        t (np.ndarray or float): Parameter value(s) between 0 and 1.
        points (np.ndarray): 2D array of control points (num_points, 2).
        order (int): The order of the derivative (1 for first, 2 for second).

    Returns:
        np.ndarray: Derivative vector(s) at the specified t value(s).
    """
    n = len(points) - 1

    if order == 0:
        # If order is 0, return the curve itself
        return general_bezier_curve(t, points)
    elif order == 1:
        # First derivative control points P_i' = n * (P_i+1 - P_i)
        if n < 1:
            # Not enough points for a derivative, return zero vectors
            return np.zeros_like(points[0]) if isinstance(t, (float, int)) else np.zeros((len(t), 2))
        derived_points = n * (points[1:] - points[:-1])
        # The derivative curve is a Bezier curve of order n-1 defined by derived_points
        return general_bezier_curve(t, derived_points)
    elif order == 2:
        # Second derivative control points P_i'' = n * (n-1) * (P_i+2 - 2*P_i+1 + P_i)
        if n < 2:
            # Not enough points for a second derivative, return zero vectors
            return np.zeros_like(points[0]) if isinstance(t, (float, int)) else np.zeros((len(t), 2))
        derived_points = n * (n - 1) * (points[2:] - 2 * points[1:-1] + points[:-2])
        # The second derivative curve is a Bezier curve of order n-2 defined by derived_points
        return general_bezier_curve(t, derived_points)
    else:
        raise ValueError("Derivative order not supported for Bezier curves. Only 0, 1, or 2.")

def bezier_curvature(t, points):
    """
    Calculates the signed curvature of a 2D Bezier curve at parameter t.
    The sign indicates the direction of curvature (e.g., convex vs. concave).

    Args:
        t (np.ndarray or float): Parameter value(s) between 0 and 1.
        points (np.ndarray): 2D array of control points (num_points, 2).

    Returns:
        np.ndarray: Curvature value(s) at the specified t value(s).
    """
    # Ensure t is a NumPy array for vectorized operations
    if not isinstance(t, np.ndarray):
        t = np.array([t])

    # Calculate first and second derivatives at each t
    P_prime = bezier_derivative(t, points, order=1)
    P_double_prime = bezier_derivative(t, points, order=2)

    # Ensure derivatives are 2D arrays (num_t_points, 2)
    if P_prime.ndim == 1: P_prime = P_prime[np.newaxis, :]
    if P_double_prime.ndim == 1: P_double_prime = P_double_prime[np.newaxis, :]

    # Extract x and y components of the derivatives
    x_prime, y_prime = P_prime[:, 0], P_prime[:, 1]
    x_double_prime, y_double_prime = P_double_prime[:, 0], P_double_prime[:, 1]

    # Calculate the numerator (with sign) and denominator of the curvature formula
    numerator = x_prime * y_double_prime - y_prime * x_double_prime
    denominator = (x_prime**2 + y_prime**2)**(3/2)

    # Handle division by zero: curvature is 0 where the denominator is 0 (e.g., at stationary points)
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    return curvature

def leading_edge_curvature(points):
    """Return the curvature at the leading edge (``t = 0``) of a Bézier curve.

    The helper wraps :func:`bezier_curvature` while ensuring that the return
    value is always a plain ``float`` regardless of whether the underlying
    function returns a scalar or a 1-element array.
    """
    curv = bezier_curvature(0.0, points)
    # ``bezier_curvature`` may return either a scalar or a (1,) ndarray.
    if np.isscalar(curv):
        return float(curv)
    return float(np.asarray(curv).ravel()[0])

def find_peak_curvature_split_point(data_points, search_region=0.1, num_samples=1000, other_surface_data=None):
    """
    Find the point of peak curvature near the leading edge to use as a better split point.
    Enhanced version that uses 3 points before and 3 points after the split point for tangent calculation.
    
    Args:
        data_points (np.ndarray): 2D array of airfoil points (x, y)
        search_region (float): Fraction of chord length to search from leading edge (default 0.1 = 10%)
        num_samples (int): Number of samples for curvature calculation
        other_surface_data (np.ndarray): 2D array of the other surface points (x, y) for enhanced tangent calculation
        
    Returns:
        tuple: (split_point_x, split_point_y, tangent_vector)
    """
    import numpy as np
    
    # Ensure data is sorted by x-coordinate
    sorted_indices = np.argsort(data_points[:, 0])
    sorted_data = data_points[sorted_indices]
    
    # Find the leading edge (minimum x-coordinate)
    le_idx = np.argmin(sorted_data[:, 0])
    le_x = sorted_data[le_idx, 0]
    le_y = sorted_data[le_idx, 1]
    
    # Define search region
    search_end_x = le_x + search_region
    
    # Find points within the search region
    mask = sorted_data[:, 0] <= search_end_x
    search_points = sorted_data[mask]
    
    if len(search_points) < 3:
        # Fallback to leading edge if not enough points
        return le_x, le_y, np.array([0.0, 1.0])
    
    # Calculate curvature at each point using finite differences
    curvatures = []
    valid_indices = []
    
    for i in range(1, len(search_points) - 1):
        p_prev = search_points[i - 1]
        p_curr = search_points[i]
        p_next = search_points[i + 1]
        
        # Calculate first and second derivatives using finite differences
        dx1 = p_curr[0] - p_prev[0]
        dy1 = p_curr[1] - p_prev[1]
        dx2 = p_next[0] - p_curr[0]
        dy2 = p_next[1] - p_curr[1]
        
        # Average the derivatives
        dx_avg = (dx1 + dx2) / 2
        dy_avg = (dy1 + dy2) / 2
        
        # Second derivative
        ddx = dx2 - dx1
        ddy = dy2 - dy1
        
        # Calculate curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_avg * ddy - dy_avg * ddx
        denominator = (dx_avg**2 + dy_avg**2)**(3/2)
        
        if abs(denominator) > 1e-12:
            curvature = abs(numerator / denominator)  # Use absolute value for peak detection
            curvatures.append(curvature)
            valid_indices.append(i)
    
    if not curvatures:
        # Fallback to leading edge if curvature calculation failed
        return le_x, le_y, np.array([0.0, 1.0])
    
    # Find the point with maximum curvature
    max_curv_idx = valid_indices[np.argmax(curvatures)]
    split_point = search_points[max_curv_idx]
    
    # Enhanced tangent calculation using 3 points before and 3 points after
    tangent = calculate_enhanced_tangent(split_point, data_points, other_surface_data)
    
    return split_point[0], split_point[1], tangent


def calculate_enhanced_tangent(split_point, primary_data, other_surface_data=None, return_debug=False):
    """
    Calculate enhanced tangent vector using exactly 3 points before and 3 points after the split point.
    Points are gathered from the primary surface first; if not enough are available on one side, points
    from the other surface are used. A quadratic polynomial is fit in least squares sense to these 6 points
    and the derivative at the split point gives the tangent direction.

    Args:
        split_point (np.ndarray): The split point coordinates [x, y]
        primary_data (np.ndarray): Primary surface data points
        other_surface_data (np.ndarray): Other surface data points (optional)
        return_debug (bool): If True, also return (selected_points, fit_mode, poly_coeffs)

    Returns:
        np.ndarray | Tuple[np.ndarray, np.ndarray, str, np.ndarray]: Normalized tangent vector [dx, dy],
        and optionally the selected points, fit mode ('y(x)' or 'x(y)'), and polynomial coefficients.
    """
    import numpy as np
    
    # Combine all available data points
    all_data = primary_data.copy()
    if other_surface_data is not None:
        all_data = np.vstack([all_data, other_surface_data])
    
    # Sort all data by x-coordinate
    sorted_indices = np.argsort(all_data[:, 0])
    sorted_all_data = all_data[sorted_indices]
    
    # Find the index of the split point in the combined data
    # Use closest point if exact match not found
    distances = np.sqrt((sorted_all_data[:, 0] - split_point[0])**2 + 
                       (sorted_all_data[:, 1] - split_point[1])**2)
    split_idx = np.argmin(distances)
    
    # Collect exactly 3 points before and 3 points after the split point (by x order)
    points_before = []
    points_after = []

    # Walk backward for before-points
    i = 1
    while len(points_before) < 3 and (split_idx - i) >= 0:
        points_before.append(sorted_all_data[split_idx - i])
        i += 1

    # Walk forward for after-points
    i = 1
    while len(points_after) < 3 and (split_idx + i) < len(sorted_all_data):
        points_after.append(sorted_all_data[split_idx + i])
        i += 1

    # If still lacking on either side (unlikely after combining both surfaces), just return vertical fallback
    if len(points_before) < 3 or len(points_after) < 3:
        tangent_vec = np.array([0.0, 1.0])
        if return_debug:
            selected_points = np.array(points_before[::-1] + [split_point] + points_after)
            return tangent_vec, selected_points, 'y(x)', np.array([0.0, 1.0, 0.0])
        return tangent_vec

    # Build the selected 6 points around the split point, sorted by x
    selected_points = np.array(points_before[::-1] + [split_point] + points_after)
    # Determine whether to fit y(x) or x(y) based on local orientation
    x_min, x_max = np.min(selected_points[:, 0]), np.max(selected_points[:, 0])
    y_min, y_max = np.min(selected_points[:, 1]), np.max(selected_points[:, 1])
    dx_range = x_max - x_min
    dy_range = y_max - y_min

    fit_mode = 'y(x)'
    # If the x-range is very small compared to y-range, fit x(y)
    if dx_range < 1e-8 or dx_range < 0.5 * dy_range:
        fit_mode = 'x(y)'

    # Fit a quadratic polynomial in least squares sense
    if fit_mode == 'y(x)':
        coeffs = np.polyfit(selected_points[:, 0], selected_points[:, 1], deg=2)
        # y' = 2ax + b
        a, b, c = coeffs
        dy_dx = 2.0 * a * split_point[0] + b
        tangent_vec = np.array([1.0, dy_dx])
    else:
        coeffs = np.polyfit(selected_points[:, 1], selected_points[:, 0], deg=2)
        # x' = 2ay + b  => dy/dx = 1 / x'
        a, b, c = coeffs
        dx_dy = 2.0 * a * split_point[1] + b
        if abs(dx_dy) < 1e-12:
            tangent_vec = np.array([0.0, 1.0])
        else:
            dy_dx = 1.0 / dx_dy
            tangent_vec = np.array([1.0, dy_dx])

    # Normalize tangent
    norm = np.linalg.norm(tangent_vec)
    if norm > 1e-12:
        tangent_vec = tangent_vec / norm
    else:
        tangent_vec = np.array([0.0, 1.0])

    if return_debug:
        return tangent_vec, selected_points, fit_mode, coeffs

    return tangent_vec

def find_optimal_split_point_for_both_surfaces(upper_data, lower_data, search_region=0.1):
    """
    Find the optimal split point for both surfaces based on the surface with tighter curvature.
    Returns the split point coordinates and two tangent vectors - one for each surface.
    
    Args:
        upper_data (np.ndarray): 2D array of upper surface points (x, y)
        lower_data (np.ndarray): 2D array of lower surface points (x, y)
        search_region (float): Fraction of chord length to search from leading edge (default 0.1 = 10%)
        
    Returns:
        tuple: (split_point_x, split_point_y, upper_tangent_vector, lower_tangent_vector, tighter_surface_is_upper)
    """
    import numpy as np
    
    # Maximum allowed x-distance from leading edge (0,0)
    max_x_distance = 0.002
    
    # Find peak curvature for both surfaces (with enhanced tangent calculation using both surfaces)
    upper_split_x, upper_split_y, upper_tangent = find_peak_curvature_split_point(upper_data, search_region, other_surface_data=lower_data)
    lower_split_x, lower_split_y, lower_tangent = find_peak_curvature_split_point(lower_data, search_region, other_surface_data=upper_data)
    
    # Calculate curvature at the split points to determine which surface is tighter
    def calculate_curvature_at_point(data, target_x, target_y):
        """Calculate curvature at a specific point using nearby data."""
        # Find points near the target
        distances = np.sqrt((data[:, 0] - target_x)**2 + (data[:, 1] - target_y)**2)
        near_indices = np.where(distances < search_region * 0.5)[0]
        
        if len(near_indices) < 3:
            return 0.0
        
        near_data = data[near_indices]
        # Sort by x to ensure proper ordering
        sort_idx = np.argsort(near_data[:, 0])
        near_data = near_data[sort_idx]
        
        # Calculate curvature using finite differences
        curvatures = []
        for i in range(1, len(near_data) - 1):
            p_prev = near_data[i - 1]
            p_curr = near_data[i]
            p_next = near_data[i + 1]
            
            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]
            
            dx_avg = (dx1 + dx2) / 2
            dy_avg = (dy1 + dy2) / 2
            ddx = dx2 - dx1
            ddy = dy2 - dy1
            
            numerator = dx_avg * ddy - dy_avg * ddx
            denominator = (dx_avg**2 + dy_avg**2)**(3/2)
            
            if abs(denominator) > 1e-12:
                curvature = abs(numerator / denominator)
                curvatures.append(curvature)
        
        return np.max(curvatures) if curvatures else 0.0
    
    # Calculate curvature at both split points
    upper_curvature = calculate_curvature_at_point(upper_data, upper_split_x, upper_split_y)
    lower_curvature = calculate_curvature_at_point(lower_data, lower_split_x, lower_split_y)
    
    # Determine which surface has tighter curvature
    if lower_curvature > upper_curvature:
        # Lower surface has tighter curvature - use its split point
        split_x = lower_split_x
        split_y = lower_split_y
        tighter_surface_is_upper = False
    else:
        # Upper surface has tighter curvature - use its split point
        split_x = upper_split_x
        split_y = upper_split_y
        tighter_surface_is_upper = True
    
    # Check if the selected split point is within the allowed x-distance from leading edge
    if abs(split_x) > max_x_distance:
        # Split point is too far from leading edge, default to (0,0)
        split_x = 0.0
        split_y = 0.0
        # When split point is at (0,0), use vertical tangents (standard leading edge behavior)
        reference_tangent = np.array([0.0, 1.0])  # Vertical upward as reference
    else:
        # Use the tighter surface's tangent as reference
        if tighter_surface_is_upper:
            reference_tangent = upper_tangent
        else:
            reference_tangent = lower_tangent
    
    # Create separate tangent vectors for each surface
    # We want them to point in opposite directions: one forward, one backward
    # Use the reference tangent (from tighter surface or default at (0,0))
    
    # For upper surface: ensure it points upward and in the reference direction
    upper_tangent_adjusted = reference_tangent.copy()
    if upper_tangent_adjusted[1] < 0:
        upper_tangent_adjusted = -upper_tangent_adjusted
    
    # For lower surface: ensure it points downward and in the opposite direction
    lower_tangent_adjusted = -reference_tangent.copy()  # Opposite direction
    if lower_tangent_adjusted[1] > 0:
        lower_tangent_adjusted = -lower_tangent_adjusted
    
    return split_x, split_y, upper_tangent_adjusted, lower_tangent_adjusted, tighter_surface_is_upper




def reorganize_airfoil_data_for_split_point(upper_data, lower_data, split_x, tighter_surface_is_upper):
    """
    Reorganize airfoil data to use a single split point for both surfaces.
    Points with x < split_x from the tighter surface are moved to the other surface.
    
    Args:
        upper_data (np.ndarray): Original upper surface data
        lower_data (np.ndarray): Original lower surface data
        split_x (float): The split point x-coordinate
        tighter_surface_is_upper (bool): Whether the upper surface has tighter curvature
        
    Returns:
        tuple: (new_upper_data, new_lower_data)
    """
    import numpy as np
    
    # Determine which surface is the source (tighter) and which is the target
    if tighter_surface_is_upper:
        source_data = upper_data
        target_data = lower_data
        source_name = "upper"
        target_name = "lower"
    else:
        source_data = lower_data
        target_data = upper_data
        source_name = "lower"
        target_name = "upper"
    
    # Find points from source surface that should be moved (x < split_x)
    source_points_to_move = source_data[source_data[:, 0] < split_x]
    
    # Remove those points from source surface
    source_points_to_keep = source_data[source_data[:, 0] >= split_x]
    
    # Add the moved points to target surface
    # We need to ensure proper ordering: moved points should come before existing target points
    target_points = target_data.copy()
    
    # Combine moved points with target points and sort by x
    combined_target = np.vstack([source_points_to_move, target_points])
    sort_idx = np.argsort(combined_target[:, 0])
    new_target_data = combined_target[sort_idx]
    
    # Return the reorganized data
    if tighter_surface_is_upper:
        new_upper_data = source_points_to_keep
        new_lower_data = new_target_data
    else:
        new_upper_data = new_target_data
        new_lower_data = source_points_to_keep
    
    return new_upper_data, new_lower_data
