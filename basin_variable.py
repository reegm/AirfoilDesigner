import numpy as np
import time
from scipy.optimize import basinhopping, minimize

from utils.data_loader import load_airfoil_data
from utils.control_point_utils import get_paper_fixed_x_coords
from utils.error_calculators import calculate_single_bezier_fitting_error
from core import config

# ------------------------------------------------------------------------- #
# Smarter proposal + guardrail accept test                                  #
# ------------------------------------------------------------------------- #
class SmartStep:
    """Anisotropic proposal:  Δy ~ N(0, σ_y),  Δx ~ N(0, σ_x)  (σ_x << σ_y).
    X's are re-sorted so monotonicity holds without throwing the point away.
    """
    def __init__(self, n_inner, dy=0.03, dx=0.01):
        self.n = n_inner
        self.dy = dy
        self.dx = dx

    def __call__(self, x):
        y = x[:self.n] + np.random.normal(0.0, self.dy,  self.n)
        x_inner = x[self.n:] + np.random.normal(0.0, self.dx,  self.n)
        x_inner = np.maximum.accumulate(np.clip(x_inner, 1e-3, 1 - 1e-3))
        return np.concatenate([y, x_inner])

class MaxErrorFilter:
    """Reject if proposed max-error is ridiculously high (cheap eval)."""
    def __init__(self, data, build_ctrl, cap_factor=3.0):
        self.data = data
        self.build_ctrl = build_ctrl
        self.cap_factor = cap_factor
        self.best = np.inf

    def __call__(self, f_new, x_new, f_old, x_old):
        # update running best
        if f_new < self.best:
            self.best = f_new
        # quick drop if new is much worse than best so far
        return f_new < self.best * self.cap_factor


# -----------------------------------------------------------------------------
# Utility: trailing‑edge tangent identical to original -------------------------
# -----------------------------------------------------------------------------

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
    return vec / norm if norm > 1e-9 else np.array([1.0, 0.0])

# -----------------------------------------------------------------------------
# Basin‑hopping progress printer ----------------------------------------------
# -----------------------------------------------------------------------------

class BasinHoppingCallback:
    def __init__(self, label, objective_fn):
        self.start_time = time.time()
        self.iteration = 0
        self.label = label
        self.objective_fn = objective_fn

    def __call__(self, x, f, accept):
        self.iteration += 1
        elapsed = time.time() - self.start_time
        max_err = self.objective_fn.last_max_error
        print(f"{self.label} | iter {self.iteration:4d} | accepted {str(accept):5} | "
              f"max‑err {max_err:.6e} | elapsed {elapsed:.1f}s")

# -----------------------------------------------------------------------------
# Main routine – variable‑X optimisation for one surface -----------------------
# -----------------------------------------------------------------------------

def run_basin_hopping_for_surface_varx(surface_data, *, is_upper: bool, label: str):
    """Minimise *maximum* Euclidean fitting error of a single Bézier curve **with
    variable inner X and Y** positions.  Leading edge is fixed at (0,0);
    trailing edge at (1, y_TE).

    Decision vector = [ y₁..y₈ , x₁..x₈ ]  (for default 10‑pt control polygon)
    where Xs are strictly increasing between (0,1).
    """
    n_ctrl = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
    if n_ctrl < 4:
        raise ValueError("Need ≥4 control points for variable‑X implementation.")

    n_inner = n_ctrl - 2                        # exclude LE and TE
    fixed_x_default = get_paper_fixed_x_coords(is_upper)[1:-1]
    te_y = float(surface_data[-1, 1])

    # ---------------- Initial guess -----------------------------------------
    init_y = np.interp(fixed_x_default, surface_data[:, 0], surface_data[:, 1])
    init_x = fixed_x_default.copy()             # start from paper positions
    x0 = np.concatenate([init_y, init_x])       # length 2*n_inner

    # ---------------- Helper: build control polygon ------------------------
    def build_ctrl(var_vec):
        y_inner = var_vec[:n_inner]
        x_inner = var_vec[n_inner:]
        ctrl = np.zeros((n_ctrl, 2))
        ctrl[0] = (0.0, 0.0)                    # leading edge
        ctrl[1:-1, 0] = x_inner
        ctrl[1:-1, 1] = y_inner
        ctrl[-1] = (1.0, te_y)                 # trailing edge
        return ctrl

    # ---------------- Objective function -----------------------------------
    reg_weight = config.DEFAULT_REGULARIZATION_WEIGHT

    class BestTracker:
        def __init__(self):
            self.best_max = np.inf
            self.best_vec = None
    best = BestTracker()

    def objective(var_vec):
        ctrl = build_ctrl(var_vec)
        fit_res = calculate_single_bezier_fitting_error(
            ctrl, surface_data, error_function="orthogonal", return_max_error=True
        )
        if isinstance(fit_res, tuple):
            _, max_err, _ = fit_res
        else:
            max_err = fit_res

        # Regularisation: keep X roughly spaced like the initial template to
        # discourage collapse (second finite diff of X)
        diffs_y = np.diff(ctrl[:, 1], n=2) if n_ctrl > 4 else 0.0
        smooth_y = np.sum(diffs_y ** 2)
        diffs_x = np.diff(ctrl[:, 0], n=2) if n_ctrl > 4 else 0.0
        smooth_x = np.sum(diffs_x ** 2)
        penalty = reg_weight * (smooth_y + smooth_x)

        objective.last_max_error = max_err
        if max_err < best.best_max:
            best.best_max = max_err
            best.best_vec = var_vec.copy()
        return max_err + penalty  # *** minimise max error ***

    objective.last_max_error = np.inf

    # ---------------- Constraints ------------------------------------------
    min_spacing = 1e-2                        # ensure strict monotonicity

    def monotone_constraints(var_vec):
        x_inner = var_vec[n_inner:]
        return np.diff(x_inner) - min_spacing   # ≥0 for all entries

    def x0_constraint(var_vec):
        x_inner = var_vec[n_inner:]
        return x_inner[0]  # should be zero

    te_tangent = calculate_te_tangent(surface_data)
    tx_te, ty_te = te_tangent
    px_n, py_n = 1.0, 0.0
    px_n_minus_1 = fixed_x_default[-1]
    def te_tangent_constraint(var_vec):
        y_inner = var_vec[:n_inner]
        x_inner = var_vec[n_inner:]
        y_n_minus_1 = y_inner[-1]
        x_n_minus_1 = x_inner[-1]
        return y_n_minus_1 * tx_te - (py_n * tx_te - (px_n - x_n_minus_1) * ty_te)
    constraints = (
        [{"type": "ineq", "fun": monotone_constraints},
         {"type": "eq", "fun": x0_constraint}]
        + ([{"type": "eq", "fun": te_tangent_constraint}] if not np.isclose(tx_te, 0.0) else [])
    )

    minim_kwargs = {
        "method": "SLSQP",
        "constraints": constraints,
        "options": config.SLSQP_OPTIONS,
    }

    # ---------------- Basin‑hopping ----------------------------------------
    print(f"\n--- Variable‑X Basin‑hopping for {label} surface ---")
    callback = BasinHoppingCallback(label, objective)

    result = basinhopping(
        func=objective,
        x0=x0,
        minimizer_kwargs=minim_kwargs,
        niter=100,
        T=1.0,
        stepsize=0.05,
        callback=callback,
        seed=42,
    )

    # ---------------- Report ------------------------------------------------
    ctrl_final = build_ctrl(result.x)
    ctrl_best = build_ctrl(best.best_vec)
    print(f"Best max‑error encountered: {best.best_max:.6e}")
    print("Control points (best):\n", ctrl_best)
    print("Control points (optimizer's final):\n", ctrl_final)
    return ctrl_best

# -----------------------------------------------------------------------------
# Driver to optimise both surfaces -------------------------------------------
# -----------------------------------------------------------------------------

def run_variable_x_basin_hopping():
    airfoil_file = "res/NACA64-720.DAT"
    print(f"Loading airfoil: {airfoil_file}")
    upper, lower, *_ = load_airfoil_data(airfoil_file)

    ctrl_upper = run_basin_hopping_for_surface_varx(upper, is_upper=True, label="Upper")
    ctrl_lower = run_basin_hopping_for_surface_varx(lower, is_upper=False, label="Lower")

    print("\n--- Variable‑X Basin‑hopping complete ---")
    return ctrl_upper, ctrl_lower

if __name__ == "__main__":
    run_variable_x_basin_hopping()
