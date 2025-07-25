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

def get_fixed_inner_x_partition(is_upper_surface, num_control_points, original_data, te_tangent_vector, te_y=None):
    """
    Returns (fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values)
    For fixed-x: the last inner control point (pre-trailing-edge) is fixed.
    The y-value is set so the vector to the TE matches te_tangent_vector.
    """
    paper_fixed_x_coords = get_paper_fixed_x_coords(is_upper_surface)
    if num_control_points != len(paper_fixed_x_coords):
        num_control_points = len(paper_fixed_x_coords)
    fixed_inner_x_coords = paper_fixed_x_coords[1:-1]
    n = len(fixed_inner_x_coords)
    free_indices = list(range(n-1))
    fixed_indices = [n-1]
    # Compute y for pre-TE so that (y_TE - y_n-1)/(x_TE - x_n-1) = ty_TE/tx_TE
    x_n_minus_1 = fixed_inner_x_coords[-1]
    if te_y is None:
        te_y = float(original_data[-1, 1])
    x_te = 1.0
    tx_te, ty_te = te_tangent_vector
    if abs(tx_te) < 1e-12:
        # Avoid division by zero, fallback to interpolation
        y_n_minus_1 = np.interp(x_n_minus_1, original_data[:, 0], original_data[:, 1])
    else:
        y_n_minus_1 = te_y - (x_te - x_n_minus_1) * (ty_te / tx_te)
    fixed_y_values = [y_n_minus_1]
    return fixed_inner_x_coords, free_indices, fixed_indices, fixed_y_values

def build_control_points_with_fixed(variables_y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values):
    """
    Assemble full control points array, inserting fixed y-values at fixed_indices.
    """
    n = len(fixed_inner_x_coords)
    control_points = np.zeros((n + 2, 2))
    control_points[0] = np.array([0.0, 0.0])
    control_points[1:-1, 0] = fixed_inner_x_coords
    y_vals = np.zeros(n)
    y_vals[free_indices] = variables_y
    for idx, y in zip(fixed_indices, fixed_y_values):
        y_vals[idx] = y
    control_points[1:-1, 1] = y_vals
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

def g2_constraint_factory_fixed_x(inner_x_upper, free_idx_u, fixed_idx_u, fixed_y_u, te_y_upper,
                                 inner_x_lower, free_idx_l, fixed_idx_l, fixed_y_l, te_y_lower,
                                 original_upper_data, original_lower_data):
    """
    Returns a constraint function for G2 continuity at the leading edge for fixed-x coupled paths,
    reconstructing the full polygons from free variables and fixed pre-TE y-values.
    """
    def constraint(var_y):
        n_free_u = len(free_idx_u)
        n_free_l = len(free_idx_l)
        y_u = var_y[:n_free_u]
        y_l = var_y[n_free_u:]
        ctrl_upper = build_control_points_with_fixed(y_u, inner_x_upper, te_y_upper, free_idx_u, fixed_idx_u, fixed_y_u)
        ctrl_lower = build_control_points_with_fixed(y_l, inner_x_lower, te_y_lower, free_idx_l, fixed_idx_l, fixed_y_l)
        return leading_edge_curvature(ctrl_upper) + leading_edge_curvature(ctrl_lower)
    return constraint

def extract_free_y_from_ctrl(ctrl, free_indices):
    """
    Extract the free y-variables from a control polygon given the free indices.
    """
    return ctrl[1:-1, 1][free_indices]

def make_build_ctrl_fn(fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values):
    """
    Returns a function that builds control points from y-variables for single Bezier.
    """
    def build_ctrl(y):
        return build_control_points_with_fixed(y, fixed_inner_x_coords, te_y, free_indices, fixed_indices, fixed_y_values)
    return build_ctrl

def make_residuals_fn(original_data, error_function, minmax=False):
    """
    Returns a function that computes residuals for a given control polygon.
    """
    def residuals(ctrl):
        efun = error_function
        if minmax:
            efun = efun + "_minmax" if not efun.endswith("_minmax") else efun
        residuals, _, _ = __import__('core.solver.error_functions', fromlist=['calculate_single_bezier_fitting_error']).calculate_single_bezier_fitting_error(
            ctrl, original_data, error_function=efun, return_max_error=False, return_all=True)
        return residuals
    return residuals

def run_minmax_stage(
    initial_y,
    build_control_points_fn,
    residuals_fn,
    regularization_weight,
    smoothness_fn,
    constraints_fns=None,
    logger_func=None
):
    """
    Generic minmax optimizer for single or coupled Bezier, given modular helpers.
    constraints_fns: list of (type, fn) tuples, e.g. [("ineq", fn1), ("eq", fn2)]
    """
    import numpy as np
    from scipy.optimize import minimize
    # Initial residuals and t0
    ctrl0 = build_control_points_fn(initial_y)
    residuals0 = residuals_fn(ctrl0)
    t0 = np.max(np.abs(residuals0)) * 1.05
    y0_aug = np.concatenate([initial_y, [t0]])
    def full_obj(y_aug):
        y = y_aug[:-1]
        t = y_aug[-1]
        ctrl = build_control_points_fn(y)
        val = t + regularization_weight * (smoothness_fn(ctrl) if smoothness_fn else 0.0)
        return val
    constraints = []
    if constraints_fns:
        for ctype, fn in constraints_fns:
            constraints.append({"type": ctype, "fun": fn})
    # Per-residual constraints (minmax)
    for i in range(len(residuals0)):
        def constr_i(i):
            return lambda y_aug: y_aug[-1] - abs(residuals_fn(build_control_points_fn(y_aug[:-1]))[i])
        constraints.append({"type": "ineq", "fun": constr_i(i)})
    if logger_func:
        logger_func("Stage 2: Running modular minmax optimization...")
    result = minimize(
        full_obj,
        y0_aug,
        method="SLSQP",
        constraints=constraints,
        options=getattr(__import__('core.config', fromlist=['SLSQP_OPTIONS']), 'SLSQP_OPTIONS')
    )
    if logger_func:
        if result.success:
            logger_func("Stage 2: Minmax optimization succeeded.")
        else:
            logger_func(f"Stage 2: Minmax optimization failed. Using initial guess. Reason: {result.message}")
    final_y = result.x[:-1] if result.success else initial_y
    return build_control_points_fn(final_y) 