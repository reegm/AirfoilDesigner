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
