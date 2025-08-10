import numpy as np
from core import config
from utils.bezier_utils import general_bezier_curve, bezier_derivative, bezier_curvature


def compute_thickness_and_camber_percent(
    upper_control_points: np.ndarray,
    lower_control_points: np.ndarray,
    num_samples: int = config.NUM_POINTS_CURVE_ERROR,
) -> tuple[float, float, float, float]:
    """Return maximum thickness, camber, and x-locations of max thickness/camber (all as %).

    Assumes chord is normalized to 1. Samples both Bezier curves uniformly in t
    and uses the y-separation as a thickness proxy. The x-location is taken as
    the average x of the upper and lower curve at the argmax thickness.
    """
    t_values = np.linspace(0.0, 1.0, max(10, int(num_samples)))
    upper_curve = general_bezier_curve(t_values, np.asarray(upper_control_points))
    lower_curve = general_bezier_curve(t_values, np.asarray(lower_control_points))

    y_thickness = upper_curve[:, 1] - lower_curve[:, 1]
    if y_thickness.size:
        idx_max = int(np.argmax(y_thickness))
        max_thickness = float(y_thickness[idx_max])
        x_at_max = 0.5 * float(upper_curve[idx_max, 0] + lower_curve[idx_max, 0])
    else:
        max_thickness = 0.0
        x_at_max = 0.0

    camber_line_y = 0.5 * (upper_curve[:, 1] + lower_curve[:, 1])
    if camber_line_y.size:
        idx_camber = int(np.argmax(np.abs(camber_line_y)))
        max_camber = float(np.abs(camber_line_y[idx_camber]))
        x_at_camber = 0.5 * float(upper_curve[idx_camber, 0] + lower_curve[idx_camber, 0])
    else:
        max_camber = 0.0
        x_at_camber = 0.0

    return 100.0 * max_thickness, 100.0 * max_camber, 100.0 * x_at_max, 100.0 * x_at_camber


def compute_te_wedge_angle_deg(
    upper_control_points: np.ndarray,
    lower_control_points: np.ndarray,
) -> float:
    """Return trailing-edge wedge angle in degrees using Bezier tangents at t=1.

    The wedge angle is computed between the upper tangent and the negative of the
    lower tangent to represent the opening angle at the TE.
    """
    t_te = np.array([1.0])
    du = bezier_derivative(t_te, np.asarray(upper_control_points), order=1)
    dl = bezier_derivative(t_te, np.asarray(lower_control_points), order=1)

    if du.ndim == 2:
        du = du[0]
    if dl.ndim == 2:
        dl = dl[0]

    norm_u = float(np.hypot(du[0], du[1]))
    norm_l = float(np.hypot(dl[0], dl[1]))
    if norm_u == 0.0 or norm_l == 0.0:
        return 0.0

    u_hat = du / norm_u
    l_hat = dl / norm_l

    # Angle between u_hat and -l_hat
    dot_val = float(np.clip(np.dot(u_hat, -l_hat), -1.0, 1.0))
    angle_rad = float(np.arccos(dot_val))
    return float(180-np.degrees(angle_rad))


def compute_le_radius_percent(
    upper_control_points: np.ndarray,
    lower_control_points: np.ndarray,
) -> float:
    """Estimate leading-edge radius as percentage of chord length.

    Uses the average absolute curvature at t=0 across upper and lower curves,
    with radius R = 1 / kappa_avg. Returns 0 if curvature is degenerate.
    """
    k_upper = bezier_curvature(0.0, np.asarray(upper_control_points))
    k_lower = bezier_curvature(0.0, np.asarray(lower_control_points))

    # Ensure scalars
    k_upper = float(k_upper if np.isscalar(k_upper) else np.asarray(k_upper).ravel()[0])
    k_lower = float(k_lower if np.isscalar(k_lower) else np.asarray(k_lower).ravel()[0])

    k_avg = 0.5 * (abs(k_upper) + abs(k_lower))
    if k_avg <= 0.0:
        return 0.0
    radius = 1.0 / k_avg
    return 100.0 * float(radius)

