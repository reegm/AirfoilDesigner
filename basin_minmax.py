import numpy as np
from scipy.optimize import basinhopping, minimize
import time
import os
from pathlib import Path

from utils.data_loader import load_airfoil_data
from utils.control_point_utils import get_paper_fixed_x_coords
from core.solver.error_functions import calculate_single_bezier_fitting_error
from core import config
# from core.optimization_core import build_single_venkatamaran_bezier_minmax

# -------------------------------------------------------------------------
# Proposal generator: small, normally-distributed bumps on Y only
# -------------------------------------------------------------------------
class SmartYStep:
    """
    Propose   y_new = y_old + N(0, σ_y)   for each inner control-point y.
    A single σ_y ≈ 2–3 % of chord height keeps most moves plausible.
    """
    def __init__(self, sigma_y=0.02):
        self.sigma = sigma_y          # chord-normalised units (0–1)

    def __call__(self, y_vec):
        return y_vec + np.random.normal(0.0, self.sigma, size=y_vec.size)

# -------------------------------------------------------------------------
# Early-reject filter: skip the expensive local solver if max-err is huge
# -------------------------------------------------------------------------
class MaxErrEarlyReject:
    """
    Reject a trial immediately if its raw max-error exceeds
    (cap_factor × best_error_so_far).  No SLSQP call needed.
    """
    def __init__(self, obj_fn, cap_factor=3.0):
        self.obj = obj_fn
        self.best = np.inf
        self.cap = cap_factor

    def __call__(self, f_new, x_new, f_old, x_old):
        # obj_fn.last_max_error has already been populated by the objective
        max_err = self.obj.last_max_error
        if max_err < self.best:
            self.best = max_err
        return max_err < self.best * self.cap



# --- Trailing edge tangent calculation (from core_logic.py) ---
def calculate_te_tangent(surface_data, num_points_avg=None):
    if num_points_avg is None:
        num_points_avg = config.DEFAULT_TE_VECTOR_POINTS
    n_pts = len(surface_data)
    if n_pts < 2:
        return np.array([1.0, 0.0])
    num_fit = min(num_points_avg + 1, n_pts)
    pts = surface_data[-num_fit:]
    x_vals, y_vals = pts[:, 0], pts[:, 1]
    if np.allclose(x_vals, x_vals[0]):
        return np.array([1.0, 0.0])
    try:
        slope, _ = np.polyfit(x_vals, y_vals, 1)
    except Exception:
        return np.array([1.0, 0.0])
    vec = np.array([1.0, slope])
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        return vec / norm
    else:
        return np.array([1.0, 0.0])

# --- Main optimization logic for one surface ---
def run_basin_hopping_for_surface(surface_data, is_upper, label, initial_guess=None):
    """
    Run basin-hopping minmax optimization for a single surface.
    Args:
        surface_data (np.ndarray): Airfoil surface data.
        is_upper (bool): True for upper surface, False for lower.
        label (str): Label for logging.
        initial_guess (np.ndarray or None): Optional initial guess for inner y-coordinates.
    """
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    fixed_x = get_paper_fixed_x_coords(is_upper)
    fixed_inner_x = fixed_x[1:-1]
    te_y = float(surface_data[-1, 1])
    if initial_guess is None:
        initial_guess = np.interp(fixed_inner_x, surface_data[:, 0], surface_data[:, 1])
    te_tangent = calculate_te_tangent(surface_data)
    regularization_weight = config.DEFAULT_REGULARIZATION_WEIGHT
    le_tangent = np.array([0.0, 1.0]) if is_upper else np.array([0.0, -1.0])

    class BestMaxErrorTracker:
        def __init__(self):
            self.best_max_error = float('inf')
            self.best_control_points = None
        def update(self, max_error, control_points):
            if max_error < self.best_max_error:
                self.best_max_error = max_error
                self.best_control_points = np.copy(control_points)

    best_tracker = BestMaxErrorTracker()

    def objective(y_inner):
        control_points = build_single_venkatamaran_bezier_minmax(
            original_data=surface_data,
            num_control_points_new=num_control_points,
            is_upper_surface=is_upper,
            le_tangent_vector=le_tangent,
            te_tangent_vector=te_tangent,
            regularization_weight=regularization_weight,
            optimization_method="minmax",
            logger_func=None,
            initial_guess_inner_y=y_inner
        )
        _, max_error, _ = calculate_single_bezier_fitting_error(control_points, surface_data, error_function="euclidean", return_max_error=True)
        best_tracker.update(max_error, control_points)
        print(f"[Objective] Current max error: {max_error:.6e} | Best so far: {best_tracker.best_max_error:.6e}")
        return max_error
    objective.last_max_error = None

    tx_te, ty_te = te_tangent
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_inner_x[-1]
    def te_tangent_constraint(y_inner):
        y_n_minus_1 = y_inner[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - px_n_minus_1) * ty_te)
    constraints = []
    if not np.isclose(tx_te, 0.0):
        constraints.append({'type': 'eq', 'fun': te_tangent_constraint})

    minimizer_kwargs = {
        "method": "SLSQP",
        "constraints": constraints,
        "options": config.SLSQP_OPTIONS,
    }

    print(f"\n--- Starting Basin-Hopping for {label} Surface (using core minmax) ---")
    take_step   = SmartYStep(sigma_y=0.02)   # adjust if moves feel too timid
    accept_test = MaxErrEarlyReject(objective, cap_factor=1.5)

    result = basinhopping(
        func=objective,
        x0=initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=50,                 # or however long you like
        T=0.2,                     # lower temperature: fewer bad uphill moves
        stepsize=1.0,              # ignored by SmartYStep but required by API
        take_step=take_step,
        accept_test=accept_test,
        seed=42,
    )

    print(f"\n--- {label} Surface Optimization Finished ---")
    print(f"Lowest max error found during run: {best_tracker.best_max_error:.6e}")
    print(f"Optimal control points (y-coords) for lowest max error:\n{best_tracker.best_control_points}")
    return best_tracker.best_control_points


def run_basin_hopping_test():
    print("--- Starting Basin-Hopping Optimization Test (Both Surfaces) ---")
    airfoil_file_path = os.path.join("res", "NACA64-720.DAT")
    print(f"Loading airfoil data from: {airfoil_file_path}")
    try:
        upper, lower, *_ = load_airfoil_data(airfoil_file_path)
    except FileNotFoundError:
        print(f"ERROR: Airfoil data file not found at '{airfoil_file_path}'.")
        print("Please ensure the 'res/rg14.DAT' file exists.")
        return

    upper_control_points = run_basin_hopping_for_surface(upper, is_upper=True, label="Upper")
    lower_control_points = run_basin_hopping_for_surface(lower, is_upper=False, label="Lower")
    print("\n--- Basin-Hopping Optimization Test Complete ---")

if __name__ == "__main__":
    run_basin_hopping_test()
