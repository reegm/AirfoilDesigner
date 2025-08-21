"""
B-spline helper functions for airfoil processing.

This module contains utility functions for B-spline operations including:
- Parameter creation and knot vector generation
- Basis function evaluation
- Curvature and tangent computations
- Vector normalization
- Smoothstep functions for blending
"""

import numpy as np
from scipy import interpolate


def normalize_vector(vec: np.ndarray | None) -> np.ndarray | None:
    """
    Normalize a vector.
    
    Args:
        vec: Input vector to normalize
        
    Returns:
        Normalized vector or None if normalization fails
    """
    if vec is None:
        return None
    try:
        v = np.asarray(vec, dtype=float)
        n = float(np.hypot(v[0], v[1]))
        if n <= 1e-12:
            return None
        return v / n
    except Exception:
        return None


def create_parameter_from_x_coords(surface_data: np.ndarray) -> np.ndarray:
    """
    Create parameter values with blending.
    
    Args:
        surface_data: Surface data points
        
    Returns:
        Parameter values for B-spline fitting
    """
    x_coords = surface_data[:, 0]
    x_coords = np.clip(x_coords, 0.0, 1.0 - 1e-12)
    u0 = np.sqrt(x_coords)
    u_params = u0
    return u_params


def create_knot_vector(num_control_points: int, degree: int) -> np.ndarray:
    """
    Create clamped knot vector.
    
    Args:
        num_control_points: Number of control points
        degree: B-spline degree
        
    Returns:
        Clamped knot vector
    """
    n = num_control_points - 1
    p = degree
    num_interior = n - p
    
    if num_interior <= 0:
        knot_vector = np.concatenate([
            np.zeros(p + 1),
            np.ones(p + 1)
        ])
    else:
        uniform_knots = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]
        knot_vector = np.concatenate([
            np.zeros(p + 1),
            uniform_knots,
            np.ones(p + 1)
        ])
    return knot_vector


def build_basis_matrix(t_values: np.ndarray, knot_vector: np.ndarray, degree: int) -> np.ndarray:
    """
    Build B-spline basis matrix.
    
    Args:
        t_values: Parameter values
        knot_vector: B-spline knot vector
        degree: B-spline degree
        
    Returns:
        Basis matrix for B-spline fitting
    """
    num_points = len(t_values)
    num_basis = len(knot_vector) - degree - 1
    basis_matrix = np.zeros((num_points, num_basis))
    
    for i in range(num_basis):
        for j, t in enumerate(t_values):
            basis_matrix[j, i] = evaluate_basis_function(i, degree, t, knot_vector)
    
    return basis_matrix


def evaluate_basis_function(i: int, degree: int, t: float, knots: np.ndarray) -> float:
    """
    Evaluate B-spline basis function.
    
    Args:
        i: Basis function index
        degree: B-spline degree
        t: Parameter value
        knots: Knot vector
        
    Returns:
        Basis function value
    """
    t = max(knots[0], min(knots[-1] - 1e-12, t))
    
    if degree == 0:
        if i < len(knots) - 1:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            elif i == len(knots) - 2 and abs(t - knots[-1]) < 1e-12:
                return 1.0
        return 0.0
    
    result = 0.0
    
    if i + degree < len(knots) and abs(knots[i + degree] - knots[i]) > 1e-15:
        alpha1 = (t - knots[i]) / (knots[i + degree] - knots[i])
        if 0 <= i < len(knots) - degree:
            left_basis = evaluate_basis_function(i, degree - 1, t, knots)
            result += alpha1 * left_basis
    
    if i + degree + 1 < len(knots) and abs(knots[i + degree + 1] - knots[i + 1]) > 1e-15:
        alpha2 = (knots[i + degree + 1] - t) / (knots[i + degree + 1] - knots[i + 1])
        if 0 <= i + 1 < len(knots) - degree:
            right_basis = evaluate_basis_function(i + 1, degree - 1, t, knots)
            result += alpha2 * right_basis
    
    return result


def compute_curvature_at_zero(control_points: np.ndarray, knot_vector: np.ndarray, degree: int) -> float:
    """
    Compute curvature at u=0 for given control points.
    
    Args:
        control_points: B-spline control points
        knot_vector: B-spline knot vector
        degree: B-spline degree
        
    Returns:
        Curvature value at u=0
    """
    # Create temporary curve
    curve = interpolate.BSpline(knot_vector, control_points, degree)
    
    # Get derivatives at u=0
    d1 = curve.derivative(1)(0.0)
    d2 = curve.derivative(2)(0.0)
    
    # Compute curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    norm_d1 = np.linalg.norm(d1)
    
    if norm_d1 > 1e-12:
        return abs(cross) / (norm_d1 ** 3)
    else:
        return 0.0


def compute_tangent_at_trailing_edge(control_points: np.ndarray, knot_vector: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute the tangent vector at the trailing edge (u=1) of a B-spline curve.
    
    For a B-spline of degree p, the tangent at u=1 is:
    tangent = p * (P_n - P_{n-1}) / (t_{n+1} - t_{n-p+1})
    
    Args:
        control_points: Control points of the B-spline
        knot_vector: Knot vector of the B-spline
        degree: B-spline degree
        
    Returns:
        Tangent vector at the trailing edge
    """
    n = len(control_points) - 1  # Number of control points minus 1
    p = degree
    
    # Get the last two control points
    P_n = control_points[-1]
    P_n_minus_1 = control_points[-2]
    
    # Get the relevant knot values
    t_n_plus_1 = knot_vector[n + 1]
    t_n_minus_p_plus_1 = knot_vector[n - p + 1]
    
    # Compute the denominator
    denominator = t_n_plus_1 - t_n_minus_p_plus_1
    
    if abs(denominator) < 1e-12:
        # Fallback: use simple difference
        tangent = P_n - P_n_minus_1
    else:
        # Compute tangent using B-spline formula
        tangent = p * (P_n - P_n_minus_1) / denominator
    
    # Normalize the tangent vector
    norm = np.linalg.norm(tangent)
    if norm > 1e-12:
        tangent = tangent / norm
    
    return tangent


def smoothstep_quintic(u: np.ndarray) -> np.ndarray:
    """
    C2 smoothstep (quintic) function.
    
    Ensures f(0)=0, f'(0)=f''(0)=0 and f(1)=1, f'(1)=f''(1)=0
    
    Args:
        u: Input parameter values
        
    Returns:
        Smoothstep values
    """
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15*u + 6*u*u)


def sample_curve(curve: interpolate.BSpline, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a B-spline curve densely in parameter domain.
    
    Args:
        curve: B-spline curve to sample
        num_samples: Number of samples to take
        
    Returns:
        Tuple of (parameter values, curve points)
    """
    t_start = curve.t[curve.k]
    t_end = curve.t[-(curve.k + 1)]
    t_vals = np.linspace(t_start, t_end, num_samples)
    # Avoid exact end to keep derivatives well behaved
    if len(t_vals) > 0:
        t_vals[-1] = min(t_vals[-1], t_end - 1e-12)
    pts = curve(t_vals)
    return t_vals, pts


def calculate_curvature_comb_data(upper_curve: interpolate.BSpline, lower_curve: interpolate.BSpline, 
                                 num_points_per_segment: int = 200, scale_factor: float = 0.050):
    """
    Calculate curvature comb visualization data.
    
    Args:
        upper_curve: Upper surface B-spline curve
        lower_curve: Lower surface B-spline curve
        num_points_per_segment: Number of points per curve segment
        scale_factor: Scale factor for comb visualization
        
    Returns:
        List of curvature comb data for both curves
    """
    all_curves_combs = []
    curves = [upper_curve, lower_curve]
    
    for curve in curves:
        curve_comb_hairs = []
        t_start = curve.t[curve.k]
        t_end = curve.t[-(curve.k + 1)]
        t_vals = np.linspace(t_start, t_end, num_points_per_segment)
        curve_points = curve(t_vals)
        derivatives_1 = curve.derivative(1)(t_vals)
        derivatives_2 = curve.derivative(2)(t_vals)
        
        x_prime = derivatives_1[:, 0]
        y_prime = derivatives_1[:, 1]
        x_double_prime = derivatives_2[:, 0]
        y_double_prime = derivatives_2[:, 1]
        
        numerator = x_prime * y_double_prime - y_prime * x_double_prime
        denominator = (x_prime**2 + y_prime**2)**(3/2)
        curvatures = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
        
        tangent_norms = np.sqrt(x_prime**2 + y_prime**2)
        unit_tangents = np.zeros_like(derivatives_1)
        valid_tangents = tangent_norms > 1e-12
        unit_tangents[valid_tangents] = derivatives_1[valid_tangents] / tangent_norms[valid_tangents, np.newaxis]
        
        normals = np.zeros_like(unit_tangents)
        normals[:, 0] = -unit_tangents[:, 1]
        normals[:, 1] = unit_tangents[:, 0]
        
        comb_lengths = -curvatures * scale_factor
        end_points = curve_points + normals * comb_lengths[:, np.newaxis]
        
        for j in range(num_points_per_segment):
            hair_segment = np.array([curve_points[j], end_points[j]])
            curve_comb_hairs.append(hair_segment)
        
        all_curves_combs.append(curve_comb_hairs)
    
    return all_curves_combs if all_curves_combs else None
