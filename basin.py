import numpy as np
from scipy.optimize import basinhopping, minimize
import time
import os
from pathlib import Path

from utils.data_loader import load_airfoil_data
from utils.control_point_utils import get_paper_fixed_x_coords
from core.error_functions import calculate_single_bezier_fitting_error
from core import config

# -------------------------------------------------------------------------
# Proposal generator: small, normally-distributed bumps on Y only
# -------------------------------------------------------------------------
class SmartYStep:
    """
    Propose y_new and *reject immediately* if its raw max-error is worse than
    best_so_far * cap.  This prevents the expensive SLSQP call.
    """
    def __init__(self, objective_fn, surface_data,
                 fixed_inner_x, best_tracker,
                 sigma_y=0.02, cap=1.5):
        self.obj_fn   = objective_fn        # for cheap error eval
        self.data     = surface_data
        self.x_inner  = fixed_inner_x
        self.best     = best_tracker       # has .best_max
        self.sigma    = sigma_y
        self.cap      = cap

    def __call__(self, y_vec):
        # 1️⃣ propose a candidate --------------------------------------------
        trial_y = y_vec + np.random.normal(0.0, self.sigma, size=y_vec.size)
        

        # 2️⃣ cheap error estimate (single evaluation, no SLSQP) ------------
        ctrl = np.zeros((len(trial_y) + 2, 2))
        ctrl[0]           = (0.0, 0.0)
        ctrl[1:-1, 0]     = self.x_inner
        ctrl[1:-1, 1]     = trial_y
        ctrl[-1]          = (1.0, self.data[-1, 1])
        max_err = calculate_single_bezier_fitting_error(
            ctrl, self.data,
            error_function="euclidean",  # fast
            return_max_error=False         # only need metric
        )

        if max_err > self.cap * self.best.best_max_error:
            # Reject early: return the *current* point so BH treats it as no-move
            return y_vec.copy()

        return trial_y

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

# --- Basin-hopping callback for progress ---
class BasinHoppingCallback:
    def __init__(self, label, objective_fn):
        self.start_time = time.time()
        self.iteration = 0
        self.label = label
        self.objective_fn = objective_fn
    def __call__(self, x, f, accept):
        self.iteration += 1
        elapsed_time = time.time() - self.start_time
        max_error = self.objective_fn.last_max_error
        print(f"{self.label} | Iter: {self.iteration:4d} | Accepted: {str(accept):<5} | Max Error: {max_error:.6e} | Elapsed Time: {elapsed_time:.2f}s")

# --- Main optimization logic for one surface ---
def run_basin_hopping_for_surface(surface_data, is_upper, label):
    num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    fixed_x = get_paper_fixed_x_coords(is_upper)
    fixed_inner_x = fixed_x[1:-1]
    te_y = float(surface_data[-1, 1])
    initial_guess = np.interp(fixed_inner_x, surface_data[:, 0], surface_data[:, 1])
    te_tangent = calculate_te_tangent(surface_data)
    regularization_weight = config.DEFAULT_REGULARIZATION_WEIGHT

    class BestMaxErrorTracker:
        def __init__(self):
            self.best_max_error = float('inf')
            self.best_y_inner = None
        def update(self, max_error, y_inner):
            if max_error < self.best_max_error:
                self.best_max_error = max_error
                self.best_y_inner = np.copy(y_inner)

    best_tracker = BestMaxErrorTracker()

    def objective(y_inner):
        control_points = np.zeros((len(y_inner) + 2, 2))
        control_points[0] = np.array([0.0, 0.0])
        control_points[1:-1, 0] = fixed_inner_x
        control_points[1:-1, 1] = y_inner
        control_points[-1] = np.array([1.0, te_y])
        # Get both sum of squares and max error
        fit_result = calculate_single_bezier_fitting_error(control_points, surface_data, error_function="euclidean", return_max_error=True)
        if isinstance(fit_result, tuple):
            sum_sq, max_error, max_error_idx = fit_result
        else:
            max_error = fit_result
        smoothness_penalty = 0.0
        if len(control_points) > 2:
            diffs = np.diff(control_points[:, 1], n=2)
            smoothness_penalty = np.sum(diffs ** 2)
        # Store max error for callback
        objective.last_max_error = max_error
        # Update best tracker
        best_tracker.update(max_error, y_inner)
        return max_error + regularization_weight * smoothness_penalty  # Minimize max error!
    # Attach attribute for callback access
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

    print(f"\n--- Starting Basin-Hopping for {label} Surface ---")
    callback = BasinHoppingCallback(label, objective)
    take_step = SmartYStep(objective, surface_data, fixed_inner_x, best_tracker, sigma_y=0.02, cap=1.5)
    accept_test = MaxErrEarlyReject(objective, cap_factor=3.0)

    result = basinhopping(
        func=objective,
        x0=initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=50,                 # or however long you like
        T=0.2,                     # lower temperature: fewer bad uphill moves
        stepsize=1.0,              # ignored by SmartYStep but required by API
        take_step=take_step,
        accept_test=accept_test,
        callback=callback,
        seed=42,
    )

    print(f"\n--- {label} Surface Optimization Finished ---")
    # Evaluate max error for the final result
    y_opt = result.x
    control_points = np.zeros((len(y_opt) + 2, 2))
    control_points[0] = np.array([0.0, 0.0])
    control_points[1:-1, 0] = fixed_inner_x
    control_points[1:-1, 1] = y_opt
    control_points[-1] = np.array([1.0, te_y])
    _, max_error, _ = calculate_single_bezier_fitting_error(control_points, surface_data, error_function="euclidean", return_max_error=True)
    print(f"Success: {result.lowest_optimization_result.success}")
    print(f"Message: {result.lowest_optimization_result.message}")
    print(f"Lowest max error found during run: {best_tracker.best_max_error:.6e}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"Optimal control points (y-coords) for lowest max error:\n{np.vstack(([[0.0, 0.0]], np.column_stack((fixed_inner_x, best_tracker.best_y_inner)), [[1.0, te_y]]))}")
    print(f"Optimal control points (y-coords) for optimizer's final result:\n{control_points}")
    return control_points


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
