import numpy as np
import copy
import logging
import os
from scipy.optimize import minimize

# Central configuration constants
from core import config
from core.optimization_core import (
    build_single_venkatamaran_bezier,
    build_coupled_venkatamaran_beziers,
    build_single_venkatamaran_bezier_minmax,
    build_coupled_venkatamaran_beziers_minmax,
    build_coupled_venkatamaran_beziers_variable_x,
)
from utils.error_calculators import calculate_single_bezier_fitting_error
from utils.data_loader import load_airfoil_data, find_shoulder_x_coords
from utils.dxf_exporter import export_curves_to_dxf
from utils.bezier_utils import bezier_curvature, general_bezier_curve


def get_venkat_bezier_builder(enforce_g2, optimization_method):
    """
    Return the correct Bezier builder function and a type string.
    """
    if enforce_g2:
        if optimization_method == "minmax":
            return build_coupled_venkatamaran_beziers_minmax, "coupled"
        elif optimization_method in ("variable_x_g2", "variable_x_orthogonal_g2"):
            return build_coupled_venkatamaran_beziers_variable_x, "coupled"
        else:
            return build_coupled_venkatamaran_beziers, "coupled"
    else:
        if optimization_method == "minmax":
            return build_single_venkatamaran_bezier_minmax, "single"
        else:
            return build_single_venkatamaran_bezier, "single"


class CoreProcessor:
    """
    Handles all the core logic for airfoil processing, independent of any UI.
    """
    def __init__(self, logger_func=None):
        self.logger = logger_func if logger_func is not None else logging.info
        self.upper_data = None
        self.lower_data = None
        self.thickened = False
        self.single_bezier_upper_poly_sharp = None
        self.single_bezier_lower_poly_sharp = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self.airfoil_name = None
        self.last_single_bezier_upper_max_error = None
        self.last_single_bezier_lower_max_error = None
        self.last_single_bezier_upper_max_error_idx = None
        self.last_single_bezier_lower_max_error_idx = None

    def log_message(self, message):
        self.logger(message)

    def load_airfoil_data_and_initialize_model(self, file_path):
        try:
            self.upper_data, self.lower_data, self.airfoil_name, self.thickened = load_airfoil_data(file_path, logger_func=self.log_message)
            self.log_message(f"Successfully loaded airfoil data from '{os.path.basename(file_path)}'.")
            initial_upper_shoulder_x, initial_lower_shoulder_x = find_shoulder_x_coords(self.upper_data, self.lower_data)
            self.log_message(f"Detected initial upper shoulder X-coordinate: {initial_upper_shoulder_x:.4f}")
            self.log_message(f"Detected initial lower shoulder X-coordinate: {initial_lower_shoulder_x:.4f}")
            self.upper_te_tangent_vector, self.lower_te_tangent_vector = self._calculate_te_tangent(self.upper_data, self.lower_data)
            return True
        except Exception as e:
            self.log_message(f"Error loading or initializing airfoil: {e}")
            self._reset_state()
            return False

    def _reset_state(self):
        self.upper_data = None
        self.lower_data = None
        self.single_bezier_upper_poly_sharp = None
        self.single_bezier_lower_poly_sharp = None
        self.last_single_bezier_upper_max_error = None
        self.last_single_bezier_lower_max_error = None
        self.last_single_bezier_upper_max_error_idx = None
        self.last_single_bezier_lower_max_error_idx = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None

    def _calculate_te_tangent(self, upper_data, lower_data, num_points_avg=None):
        if num_points_avg is None:
            from core import config
            num_points_avg = config.DEFAULT_TE_VECTOR_POINTS
        def _surface_tangent(surface_data: np.ndarray, label: str):
            n_pts = len(surface_data)
            if n_pts < 2:
                self.log_message(f"Warning: Not enough {label.lower()} data points to calculate TE tangent. Defaulting to horizontal.")
                return np.array([1.0, 0.0])
            num_fit = min(num_points_avg + 1, n_pts)
            pts = surface_data[-num_fit:]
            x_vals, y_vals = pts[:, 0], pts[:, 1]
            if np.allclose(x_vals, x_vals[0]):
                self.log_message(f"Warning: Degenerate TE x-values for {label.lower()} surface. Defaulting to horizontal.")
                return np.array([1.0, 0.0])
            try:
                slope, _ = np.polyfit(x_vals, y_vals, 1)
            except Exception as _e:
                self.log_message(f"Warning: Polyfit failed for {label.lower()} TE tangent ( {_e} ). Defaulting to horizontal.")
                return np.array([1.0, 0.0])
            vec = np.array([1.0, slope])
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                return vec / norm
            else:
                self.log_message(f"Warning: {label} TE tangent vector from data is near zero after fit. Defaulting to horizontal.")
                return np.array([1.0, 0.0])
        upper_te_tangent_vector = _surface_tangent(upper_data, "Upper")
        lower_te_tangent_vector = _surface_tangent(lower_data, "Lower")
        return upper_te_tangent_vector, lower_te_tangent_vector

    def _compute_discrete_curvature(self, control_poly):
        control_poly = np.asarray(control_poly)
        n_pts = len(control_poly)
        curvatures = np.zeros(n_pts)
        for i in range(1, n_pts - 1):
            v_prev = control_poly[i] - control_poly[i - 1]
            v_next = control_poly[i + 1] - control_poly[i]
            len_prev = np.linalg.norm(v_prev)
            len_next = np.linalg.norm(v_next)
            if len_prev < 1e-12 or len_next < 1e-12:
                curvatures[i] = 0.0
                continue
            v_prev /= len_prev
            v_next /= len_next
            angle = np.arccos(np.clip(np.dot(v_prev, v_next), -1.0, 1.0))
            curvatures[i] = angle
        if n_pts >= 3:
            curvatures[0] = curvatures[1]
            curvatures[-1] = curvatures[-2]
        return curvatures

    def apply_te_thickening_to_single_bezier(self, single_bezier_polygons_copy, te_thickness):
        import copy as _copy
        if te_thickness < 1e-9:
            return single_bezier_polygons_copy
        upper_poly, lower_poly = _copy.deepcopy(single_bezier_polygons_copy[0]), _copy.deepcopy(single_bezier_polygons_copy[1])
        upper_curvatures = self._compute_discrete_curvature(upper_poly)
        lower_curvatures = self._compute_discrete_curvature(lower_poly)
        max_curv_up = np.max(upper_curvatures) + 1e-9
        max_curv_low = np.max(lower_curvatures) + 1e-9
        upper_delta_y_at_te = (te_thickness / 2.0) - upper_poly[-1][1]
        for i in range(1, len(upper_poly) - 1):
            x_chord = upper_poly[i][0]
            if x_chord <= 1e-9:
                continue
            curvature_weight = 1.0 - (upper_curvatures[i] / max_curv_up)
            curvature_weight = np.clip(curvature_weight, 0.0, 1.0)
            scaling_factor = curvature_weight * x_chord
            upper_poly[i][1] += scaling_factor * upper_delta_y_at_te
        upper_poly[-1][1] = te_thickness / 2.0
        lower_delta_y_at_te = (-te_thickness / 2.0) - lower_poly[-1][1]
        for i in range(1, len(lower_poly) - 1):
            x_chord = lower_poly[i][0]
            if x_chord <= 1e-9:
                continue
            curvature_weight = 1.0 - (lower_curvatures[i] / max_curv_low)
            curvature_weight = np.clip(curvature_weight, 0.0, 1.0)
            scaling_factor = curvature_weight * x_chord
            lower_poly[i][1] += scaling_factor * lower_delta_y_at_te
        lower_poly[-1][1] = -te_thickness / 2.0
        try:
            self._log_max_curvature_difference(single_bezier_polygons_copy, [upper_poly, lower_poly])
        except Exception as _e:
            self.log_message(f"Warning: Curvature comparison skipped due to error: {_e}")
        return [upper_poly, lower_poly]

    def _log_max_curvature_difference(self, sharp_polygons, thick_polygons, num_samples: int = 200):
        t_vals = np.linspace(0.0, 1.0, num_samples)
        max_diff = -np.inf
        max_x_pos = None
        max_surface = None
        for surf_idx, surf_name in enumerate(["upper", "lower"]):
            sharp_poly = np.asarray(sharp_polygons[surf_idx])
            thick_poly = np.asarray(thick_polygons[surf_idx])
            curv_sharp = bezier_curvature(t_vals, sharp_poly)
            curv_thick = bezier_curvature(t_vals, thick_poly)
            diff = np.abs(curv_sharp - curv_thick)
            local_max_idx = int(np.argmax(diff))
            local_max_val = float(diff[local_max_idx])
            if local_max_val > max_diff:
                max_diff = local_max_val
                max_surface = surf_name
                t_at_max = t_vals[local_max_idx]
                curve_pt = general_bezier_curve(t_at_max, thick_poly)
                curve_pt = np.asarray(curve_pt).reshape(-1, 2)[0]
                x_at_max = float(curve_pt[0])
                max_x_pos = x_at_max
        if max_diff < 0 or max_surface is None:
            self.log_message("Curvature comparison failed – no valid maximum found.")
            return
        self.log_message(
            (
                f"Max curvature difference after thickening: {max_diff:.3e} at x = "
                f"{max_x_pos:.4f} (surface: {max_surface})"
            )
        )

    def build_single_bezier_model(self, regularization_weight, optimization_method="fixed_x", enforce_g2=False, te_vector_points=None, num_points_curvature_resample: int = config.DEFAULT_NUM_POINTS_CURVATURE_RESAMPLE):
        if self.upper_data is None or self.lower_data is None:
            self.log_message("Error: Original airfoil data not loaded. Cannot build single Bezier model.")
            return False
        use_curvature_sampling = optimization_method in ["minmax", "variable_x_orthogonal", "fixed_x_orthogonal", "variable_x_orthogonal_g2"]
        if use_curvature_sampling:
            self.log_message("Using dynamic sampling for orthogonal error methods.")
        else:
            self.log_message("Using linear sampling for euclidean error methods.")
        num_control_points_single_bezier = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
        if self.thickened:
            y_te_upper = float(self.upper_data[-1, 1])
            y_te_lower = float(self.lower_data[-1, 1])
            te_thickness_for_build = abs(y_te_upper - y_te_lower)
            if te_thickness_for_build < 1e-9:
                te_thickness_for_build = 0.0
                self.log_message("Warning: Detected TE marked as thickened but thickness ≈ 0. Building sharp TE instead.")
            else:
                self.log_message(
                    f"Detected thickened trailing edge in input (thickness = {te_thickness_for_build:.6f}). Building model with thick TE."
                )
        else:
            te_thickness_for_build = 0.0
        self.log_message("-" * 20)
        te_desc = "thick" if te_thickness_for_build > 1e-9 else "sharp"
        self.log_message(f"Building {te_desc} single Bezier curves (Order {num_control_points_single_bezier - 1})...")
        if te_vector_points is not None:
            upper_te_tangent_vector, lower_te_tangent_vector = self._calculate_te_tangent(self.upper_data, self.lower_data, te_vector_points)
            self.upper_te_tangent_vector = upper_te_tangent_vector
            self.lower_te_tangent_vector = lower_te_tangent_vector
        else:
            upper_te_tangent_vector = self.upper_te_tangent_vector
            lower_te_tangent_vector = self.lower_te_tangent_vector
        self.log_message(f"Using TE tangent for single Bezier upper: {upper_te_tangent_vector}")
        self.log_message(f"Using TE tangent for single Bezier lower: {lower_te_tangent_vector}")
        le_tangent_upper = np.array([0.0, 1.0])
        le_tangent_lower = np.array([0.0, -1.0])
        try:
            builder_fn, builder_type = get_venkat_bezier_builder(enforce_g2, optimization_method)
            if builder_type == "coupled":
                # Check if the builder function accepts num_points_curvature_resample parameter
                import inspect
                sig = inspect.signature(builder_fn)
                if 'num_points_curvature_resample' in sig.parameters:
                    upper_poly, lower_poly = builder_fn(
                        original_upper_data=self.upper_data,
                        original_lower_data=self.lower_data,
                        regularization_weight=regularization_weight,
                        te_tangent_vector_upper=upper_te_tangent_vector,
                        te_tangent_vector_lower=lower_te_tangent_vector,
                        optimization_method=optimization_method,
                        num_points_curvature_resample=num_points_curvature_resample,
                        logger_func=self.log_message,
                    )
                else:
                    upper_poly, lower_poly = builder_fn(
                        original_upper_data=self.upper_data,
                        original_lower_data=self.lower_data,
                        regularization_weight=regularization_weight,
                        te_tangent_vector_upper=upper_te_tangent_vector,
                        te_tangent_vector_lower=lower_te_tangent_vector,
                        optimization_method=optimization_method,
                        logger_func=self.log_message,
                    )
                self.single_bezier_upper_poly_sharp = upper_poly
                self.single_bezier_lower_poly_sharp = lower_poly
            else:
                for surf, is_upper, tangent, assign in [
                    (self.upper_data, True, upper_te_tangent_vector, "upper"),
                    (self.lower_data, False, lower_te_tangent_vector, "lower")
                ]:
                    # Check if the builder function accepts num_points_curvature_resample parameter
                    import inspect
                    sig = inspect.signature(builder_fn)
                    if 'num_points_curvature_resample' in sig.parameters:
                        poly = builder_fn(
                            original_data=surf,
                            num_control_points_new=num_control_points_single_bezier,
                            is_upper_surface=is_upper,
                            le_tangent_vector=le_tangent_upper if is_upper else le_tangent_lower,
                            te_tangent_vector=tangent,
                            regularization_weight=regularization_weight,
                            optimization_method=optimization_method,
                            num_points_curvature_resample=num_points_curvature_resample,
                            logger_func=self.log_message
                        )
                    else:
                        poly = builder_fn(
                            original_data=surf,
                            num_control_points_new=num_control_points_single_bezier,
                            is_upper_surface=is_upper,
                            le_tangent_vector=le_tangent_upper if is_upper else le_tangent_lower,
                            te_tangent_vector=tangent,
                            regularization_weight=regularization_weight,
                            optimization_method=optimization_method,
                            logger_func=self.log_message
                        )
                    if assign == "upper":
                        self.single_bezier_upper_poly_sharp = poly
                    else:
                        self.single_bezier_lower_poly_sharp = poly
            # Error calculation (kept DRY)
            for assign, control_poly, orig_data in [
                ("upper", self.single_bezier_upper_poly_sharp, self.upper_data),
                ("lower", self.single_bezier_lower_poly_sharp, self.lower_data)
            ]:
                error_result = calculate_single_bezier_fitting_error(
                    np.array(control_poly), orig_data, error_function="euclidean", return_max_error=True
                )
                if isinstance(error_result, tuple) and len(error_result) == 3:
                    _, max_err, max_err_idx = error_result
                    if assign == "upper":
                        self.last_single_bezier_upper_max_error = max_err
                        self.last_single_bezier_upper_max_error_idx = max_err_idx
                    else:
                        self.last_single_bezier_lower_max_error = max_err
                        self.last_single_bezier_lower_max_error_idx = max_err_idx
                else:
                    self.log_message(f"Warning: Unexpected return format from {assign} error calculation")
                    if assign == "upper":
                        self.last_single_bezier_upper_max_error = None
                        self.last_single_bezier_upper_max_error_idx = None
                    else:
                        self.last_single_bezier_lower_max_error = None
                        self.last_single_bezier_lower_max_error_idx = None
            self.log_message("Sharp single Bezier curves built successfully.")
            return True
        except Exception as e:
            self.log_message(f"Error during sharp single Bezier curve building: {e}")
            self.single_bezier_upper_poly_sharp = None
            self.single_bezier_lower_poly_sharp = None
            self.last_single_bezier_upper_max_error = None
            self.last_single_bezier_lower_max_error = None
            return False

    def export_to_dxf(self, polygons_to_export, chord_length_mm, merged_flag):
        dxf_doc = export_curves_to_dxf(
            polygons_to_export,
            chord_length_mm,
            self.log_message
        )
        if dxf_doc:
            self.log_message("DXF document created successfully.")
            return dxf_doc
        else:
            self.log_message("DXF export failed during document creation.")
            return None
