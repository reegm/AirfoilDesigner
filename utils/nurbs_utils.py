import numpy as np
from scipy.special import comb

def basis_function(i, k, u, knots):
    """
    Calculate the i-th basis function of degree k at parameter u.
    
    Args:
        i (int): Index of the basis function
        k (int): Degree of the basis function
        u (float): Parameter value
        knots (np.ndarray): Knot vector
        
    Returns:
        float: Value of the basis function
    """
    if k == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    
    # Handle the case where u is at the end of the knot vector
    if u == knots[-1] and i == len(knots) - k - 2:
        return 1.0
    
    # Handle the case where u is at the start of the knot vector (for clamped splines)
    if u == knots[k] and i == 0:
        return 1.0
    
    # Avoid division by zero
    d1 = knots[i + k] - knots[i]
    d2 = knots[i + k + 1] - knots[i + 1]
    
    c1 = 0.0 if d1 == 0.0 else (u - knots[i]) / d1
    c2 = 0.0 if d2 == 0.0 else (knots[i + k + 1] - u) / d2
    
    return c1 * basis_function(i, k - 1, u, knots) + c2 * basis_function(i + 1, k - 1, u, knots)

def generate_knot_vector(n, p, clamped=True):
    """
    Generate a knot vector for a NURBS curve.
    
    Args:
        n (int): Number of control points - 1
        p (int): Degree of the curve
        clamped (bool): Whether to create a clamped (open) knot vector
        
    Returns:
        np.ndarray: Knot vector
    """
    m = n + p + 1
    
    if clamped:
        # Clamped knot vector: first p+1 knots are 0, last p+1 knots are 1
        knots = np.zeros(m + 1)
        
        # Handle the case where there are no interior knots
        if m - 2*p <= 0:
            # All knots are either 0 or 1
            knots[:p+1] = 0.0
            knots[p+1:] = 1.0
        else:
            # Calculate the correct number of interior knots
            num_interior = m - 2*p
            interior_knots = np.linspace(0, 1, num_interior)
            knots[p+1:p+1+num_interior] = interior_knots
            knots[p+1+num_interior:] = 1.0
    else:
        # Uniform knot vector
        knots = np.linspace(0, 1, m + 1)
    
    return knots

def evaluate_nurbs_curve(u, control_points, weights, knots, degree):
    """
    Evaluate a NURBS curve at parameter u.
    
    Args:
        u (float): Parameter value between 0 and 1
        control_points (np.ndarray): Control points (n+1, 2)
        weights (np.ndarray): Weights for each control point (n+1,)
        knots (np.ndarray): Knot vector
        degree (int): Degree of the curve
        
    Returns:
        np.ndarray: Point on the curve
    """
    n = len(control_points) - 1
    
    # Calculate weighted control points
    weighted_points = control_points * weights[:, np.newaxis]
    
    # Calculate the numerator (weighted sum of basis functions)
    numerator = np.zeros(2)
    denominator = 0.0
    
    for i in range(n + 1):
        basis_val = basis_function(i, degree, u, knots)
        numerator += basis_val * weighted_points[i]
        denominator += basis_val * weights[i]
    
    # Avoid division by zero
    if denominator == 0:
        return np.array([0.0, 0.0])
    
    return numerator / denominator

def sample_nurbs_curve(control_points, weights=None, degree=3, num_samples=100):
    """
    Sample a NURBS curve at evenly spaced parameter values.
    
    Args:
        control_points (np.ndarray): Control points (n+1, 2)
        weights (np.ndarray, optional): Weights for each control point. If None, all weights are 1.0
        degree (int): Degree of the curve
        num_samples (int): Number of sample points
        
    Returns:
        tuple: (sample_points, knots) where sample_points is (num_samples, 2)
    """
    n = len(control_points) - 1
    
    # Ensure degree doesn't exceed maximum allowed
    max_degree = n
    if degree > max_degree:
        degree = max_degree
    
    if weights is None:
        weights = np.ones(n + 1)
    
    # Generate knot vector
    knots = generate_knot_vector(n, degree, clamped=True)
    
    # Sample the curve with better coverage of the parameter range
    # Use a slightly extended range to ensure we capture the full curve
    u_values = np.linspace(0, 1, num_samples)
    sample_points = np.array([evaluate_nurbs_curve(u, control_points, weights, knots, degree) 
                             for u in u_values])
    
    # For clamped splines, ensure we have the exact control points at start and end
    if len(sample_points) > 0:
        # Force the first and last points to be the actual control points
        sample_points[0] = control_points[0]
        sample_points[-1] = control_points[-1]
    
    return sample_points, knots

def bezier_to_nurbs_weights(control_points, degree):
    """
    Convert Bezier control points to NURBS weights.
    For Bezier curves, all weights are typically 1.0.
    
    Args:
        control_points (np.ndarray): Bezier control points
        degree (int): Degree of the curve
        
    Returns:
        np.ndarray: Weights for NURBS representation
    """
    return np.ones(len(control_points))

def create_nurbs_curve_from_bezier(bezier_control_points, degree=3, num_samples=100):
    """
    Create a NURBS curve representation from Bezier control points.
    
    Args:
        bezier_control_points (np.ndarray): Bezier control points (n+1, 2)
        degree (int): Degree of the NURBS curve
        num_samples (int): Number of sample points for evaluation
        
    Returns:
        tuple: (sample_points, knots, weights) for NURBS representation
    """
    weights = bezier_to_nurbs_weights(bezier_control_points, degree)
    sample_points, knots = sample_nurbs_curve(bezier_control_points, weights, degree, num_samples)
    
    return sample_points, knots, weights 