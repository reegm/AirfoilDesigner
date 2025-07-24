import numpy as np
from utils.control_point_utils import get_paper_fixed_x_coords
from utils.bezier_utils import leading_edge_curvature

def get_fixed_inner_x_coords(is_upper_surface, num_control_points):
    """
    Returns the fixed inner x-coordinates for the given surface and number of control points.
    """
    paper_fixed_x_coords = get_paper_fixed_x_coords(is_upper_surface)
    if num_control_points != len(paper_fixed_x_coords):
        num_control_points = len(paper_fixed_x_coords)
    return paper_fixed_x_coords[1:-1]

def get_initial_guess_inner_y(original_data, fixed_inner_x_coords):
    """
    Returns the initial guess for the inner y-coordinates by interpolating the original data at the fixed x-coordinates.
    """
    return np.interp(fixed_inner_x_coords, original_data[:, 0], original_data[:, 1])

def build_control_points(variables_y, fixed_inner_x_coords, te_y):
    """
    Assemble full control points array for a single Bezier curve.
    """
    control_points = np.zeros((len(variables_y) + 2, 2))
    control_points[0] = np.array([0.0, 0.0])
    control_points[1:-1, 0] = fixed_inner_x_coords
    control_points[1:-1, 1] = variables_y
    control_points[-1] = np.array([1.0, te_y])
    return control_points

def assemble_polygons(var_y, inner_x_upper, inner_x_lower, original_upper_data, original_lower_data):
    """
    Assemble full control polygons for coupled Bezier optimization.
    """
    n_inner = len(inner_x_upper)
    y_u = var_y[:n_inner]
    y_l = var_y[n_inner:]
    ctrl_upper = np.zeros((n_inner + 2, 2))
    ctrl_lower = np.zeros((n_inner + 2, 2))
    ctrl_upper[0] = [0.0, 0.0]
    ctrl_upper[1:-1, 0] = inner_x_upper
    ctrl_upper[1:-1, 1] = y_u
    ctrl_upper[-1] = [1.0, float(original_upper_data[-1, 1])]
    ctrl_lower[0] = [0.0, 0.0]
    ctrl_lower[1:-1, 0] = inner_x_lower
    ctrl_lower[1:-1, 1] = y_l
    ctrl_lower[-1] = [1.0, float(original_lower_data[-1, 1])]
    return ctrl_upper, ctrl_lower

def smoothness_penalty(control_points):
    """
    Compute the smoothness penalty (second derivative of y) for a control polygon.
    """
    if len(control_points) <= 2:
        return 0.0
    return np.sum(np.diff(control_points[:, 1], n=2) ** 2)

def te_tangent_constraint_factory(tx_te, ty_te, px_n, py_n, px_n_minus_1, idx):
    """
    Returns a constraint function for trailing edge tangency for the given index.
    """
    def constraint(variables_y):
        y_n_minus_1 = variables_y[idx]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    return constraint

def g2_constraint_factory(inner_x_upper, inner_x_lower, original_upper_data, original_lower_data):
    """
    Returns a constraint function for G2 continuity at the leading edge.
    """
    def constraint(var_y):
        ctrl_u, ctrl_l = assemble_polygons(var_y, inner_x_upper, inner_x_lower, original_upper_data, original_lower_data)
        return leading_edge_curvature(ctrl_u) + leading_edge_curvature(ctrl_l)
    return constraint 