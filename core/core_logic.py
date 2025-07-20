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

class CoreProcessor:
    """
    Handles all the core logic for airfoil processing, independent of any UI.
    """
    def __init__(self, logger_func=None):
        # self.logger will be the callable function (e.g., self.log_message.emit from AirfoilProcessor)
        self.logger = logger_func if logger_func is not None else logging.info

        # Remove self.model and any segmented model state
        self.upper_data = None # Stores original upper airfoil data
        self.lower_data = None # Stores original lower airfoil data
        self.thickened = False # Stores whether the trailing edge is thickened in the original data
        self.single_bezier_upper_poly_sharp = None # Stores the sharp single Bezier upper control polygon
        self.single_bezier_lower_poly_sharp = None # Stores the sharp single Bezier lower control polygon
        self.upper_te_tangent_vector = None  # Store calculated TE tangent vector for upper surface
        self.lower_te_tangent_vector = None  # Store calculated TE tangent vector for lower surface

        self.airfoil_name = None  # Stores profile name from .dat
        self.last_single_bezier_upper_max_error = None # Stores error for single upper Bezier
        self.last_single_bezier_lower_max_error = None # Stores error for single lower Bezier
        self.last_single_bezier_upper_max_error_idx = None # Stores index of worst fit for single upper Bezier
        self.last_single_bezier_lower_max_error_idx = None # Stores index of worst fit for single lower Bezier


    def log_message(self, message):
        """Logs a message using the configured logger."""
        self.logger(message)

    def load_airfoil_data_and_initialize_model(self, file_path):
        """
        Loads airfoil data and initializes the airfoil data for single Bezier model only.
        """
        try:
            self.upper_data, self.lower_data, self.airfoil_name, self.thickened = load_airfoil_data(file_path, logger_func=self.log_message)
            self.log_message(f"Successfully loaded airfoil data from '{os.path.basename(file_path)}'.")

            initial_upper_shoulder_x, initial_lower_shoulder_x = find_shoulder_x_coords(self.upper_data, self.lower_data)
            self.log_message(f"Detected initial upper shoulder X-coordinate: {initial_upper_shoulder_x:.4f}")
            self.log_message(f"Detected initial lower shoulder X-coordinate: {initial_lower_shoulder_x:.4f}")

            # Calculate and store TE tangent vectors
            self.upper_te_tangent_vector, self.lower_te_tangent_vector = self._calculate_te_tangent(self.upper_data, self.lower_data)

            # No model initialization needed for single Bezier
            return True
        except Exception as e:
            self.log_message(f"Error loading or initializing airfoil: {e}")
            self._reset_state()
            return False

    def _reset_state(self):
        """Resets the internal state of the core processor."""
        self.upper_data = None
        self.lower_data = None
        self.single_bezier_upper_poly_sharp = None
        self.single_bezier_lower_poly_sharp = None
        self.last_single_bezier_upper_max_error = None # Reset
        self.last_single_bezier_lower_max_error = None # Reset
        self.last_single_bezier_upper_max_error_idx = None # Reset
        self.last_single_bezier_lower_max_error_idx = None # Reset
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None

    def _calculate_te_tangent(self, upper_data, lower_data, num_points_avg=None):
        """
        Calculates the approximate trailing edge tangent vectors from the original data.
        Instead of relying on a single segment (last two points), this implementation
        averages the direction vectors obtained from the last *num_points_avg* segments
        (TE minus the preceding points).  This provides a tangent estimate that is
        less sensitive to noise in any individual data point.

        Parameters
        ----------
        upper_data : np.ndarray
            Upper surface data points
        lower_data : np.ndarray
            Lower surface data points
        num_points_avg : int, optional
            Number of points *before* the TE to include in the average. Defaults to config.DEFAULT_TE_VECTOR_POINTS.

        Returns two normalised vectors: (upper_te_tangent_vector, lower_te_tangent_vector).
        """
        if num_points_avg is None:
            from core import config
            num_points_avg = config.DEFAULT_TE_VECTOR_POINTS

        def _surface_tangent(surface_data: np.ndarray, label: str):
            """Compute TE tangent using a linear fit through the last few points."""
            n_pts = len(surface_data)
            if n_pts < 2:
                self.log_message(f"Warning: Not enough {label.lower()} data points to calculate TE tangent. Defaulting to horizontal.")
                return np.array([1.0, 0.0])

            # Use the last num_points_avg+1 points (including TE) for a robust straight-line fit
            num_fit = min(num_points_avg + 1, n_pts)
            pts = surface_data[-num_fit:]
            x_vals, y_vals = pts[:, 0], pts[:, 1]

            # Guard against duplicate x values which would break polyfit
            if np.allclose(x_vals, x_vals[0]):
                self.log_message(f"Warning: Degenerate TE x-values for {label.lower()} surface. Defaulting to horizontal.")
                return np.array([1.0, 0.0])

            try:
                slope, _ = np.polyfit(x_vals, y_vals, 1)
            except Exception as _e:
                self.log_message(f"Warning: Polyfit failed for {label.lower()} TE tangent ( {_e} ). Defaulting to horizontal.")
                return np.array([1.0, 0.0])

            vec = np.array([1.0, slope])  # Vector with guaranteed x-component
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                return vec / norm
            else:
                self.log_message(f"Warning: {label} TE tangent vector from data is near zero after fit. Defaulting to horizontal.")
                return np.array([1.0, 0.0])

        upper_te_tangent_vector = _surface_tangent(upper_data, "Upper")
        lower_te_tangent_vector = _surface_tangent(lower_data, "Lower")

        return upper_te_tangent_vector, lower_te_tangent_vector

    # ------------------------------------------------------------------
    # Trailing-edge thickening helpers (shared by CLI and GUI)
    # ------------------------------------------------------------------

    def _compute_discrete_curvature(self, control_poly):
        """
        Approximates the curvature magnitude at each control point of a Bezier
        control polygon using the turning angle between consecutive polygon
        segments.  A larger value corresponds to a tighter turn (higher local
        curvature).  The first and last control points inherit the curvature
        of their immediate neighbour so that the returned array has the same
        length as *control_poly*.

        Parameters
        ----------
        control_poly : array-like, shape (N, 2)
            The x-y coordinates of the control polygon.

        Returns
        -------
        np.ndarray
            Discrete curvature estimate for every control point (radians).
        """
        control_poly = np.asarray(control_poly)
        n_pts = len(control_poly)
        curvatures = np.zeros(n_pts)

        # Skip end-points when computing turning angle – we need both neighbours
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
            # Turning angle between the two normalised edge vectors
            angle = np.arccos(np.clip(np.dot(v_prev, v_next), -1.0, 1.0))
            curvatures[i] = angle  # 0 for straight edge, π for 180° turn

        # Propagate neighbour curvature to the end-points for continuity
        if n_pts >= 3:
            curvatures[0] = curvatures[1]
            curvatures[-1] = curvatures[-2]

        return curvatures

    def apply_te_thickening_to_single_bezier(self, single_bezier_polygons_copy, te_thickness):
        """Return copies of single-Bezier polygons with TE thickening applied."""
        import copy as _copy

        if te_thickness < 1e-9:
            return single_bezier_polygons_copy

        upper_poly, lower_poly = _copy.deepcopy(single_bezier_polygons_copy[0]), _copy.deepcopy(
            single_bezier_polygons_copy[1]
        )

        # Compute discrete curvature distributions once for efficiency
        upper_curvatures = self._compute_discrete_curvature(upper_poly)
        lower_curvatures = self._compute_discrete_curvature(lower_poly)

        # Avoid division by zero by adding a small epsilon when normalising
        max_curv_up = np.max(upper_curvatures) + 1e-9
        max_curv_low = np.max(lower_curvatures) + 1e-9

        # Upper surface – shift control points towards +y based on inverse curvature
        upper_delta_y_at_te = (te_thickness / 2.0) - upper_poly[-1][1]
        for i in range(1, len(upper_poly) - 1):
            x_chord = upper_poly[i][0]
            if x_chord <= 1e-9:
                continue  # Skip LE point

            curvature_weight = 1.0 - (upper_curvatures[i] / max_curv_up)  # 0 at high-curvature, 1 at low-curvature
            curvature_weight = np.clip(curvature_weight, 0.0, 1.0)

            # Combine curvature weighting with linear chordwise weighting for smooth progression
            scaling_factor = curvature_weight * x_chord
            upper_poly[i][1] += scaling_factor * upper_delta_y_at_te

        # Pin the TE end-point exactly to the required thickness
        upper_poly[-1][1] = te_thickness / 2.0

        # Lower surface – mirror the logic with negative y-direction shift
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

        # ------------------------------------------------
        try:
            self._log_max_curvature_difference(single_bezier_polygons_copy, [upper_poly, lower_poly])
        except Exception as _e:
            # Silently ignore curvature comparison failures to avoid
            # disrupting workflow if something unforeseen happens.
            self.log_message(f"Warning: Curvature comparison skipped due to error: {_e}")

        return [upper_poly, lower_poly]

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _log_max_curvature_difference(self, sharp_polygons, thick_polygons, num_samples: int = 200):
        """Compute curvature difference between *sharp_polygons* and *thick_polygons*.

        The two arguments are expected to be ``[upper_poly, lower_poly]`` lists.
        The function examines both surfaces, samples *num_samples* points
        uniformly in the Bézier parameter (t), and logs the maximum
        absolute curvature difference along with its chordwise x-position and
        surface identifier.
        """

        t_vals = np.linspace(0.0, 1.0, num_samples)

        max_diff = -np.inf
        max_x_pos = None
        max_surface = None  # "upper" or "lower"

        for surf_idx, surf_name in enumerate(["upper", "lower"]):
            sharp_poly = np.asarray(sharp_polygons[surf_idx])
            thick_poly = np.asarray(thick_polygons[surf_idx])

            # Signed curvature arrays (shape (num_samples,))
            curv_sharp = bezier_curvature(t_vals, sharp_poly)
            curv_thick = bezier_curvature(t_vals, thick_poly)

            diff = np.abs(curv_sharp - curv_thick)

            local_max_idx = int(np.argmax(diff))
            local_max_val = float(diff[local_max_idx])

            if local_max_val > max_diff:
                max_diff = local_max_val
                max_surface = surf_name
                t_at_max = t_vals[local_max_idx]
                # Use thickened geometry for x-coordinate (nearly identical)
                curve_pt = general_bezier_curve(t_at_max, thick_poly)
                # ``general_bezier_curve`` returns shape (1, 2) for scalar *t*
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
        """
        Builds the single-span Bezier curves for upper and lower surfaces based on the 2017 Venkataraman paper.
        This method always builds a sharp (thickness 0) single Bezier curve.
        Thickening is applied separately for display.
        Supports optimization methods: 'fixed_x', 'minmax', 'variable_x', 'variable_x_orthogonal'.
        """
        if self.upper_data is None or self.lower_data is None:
            self.log_message("Error: Original airfoil data not loaded. Cannot build single Bezier model.")
            return False

        # Automatically determine sampling type based on optimization method
        use_curvature_sampling = optimization_method in ["minmax", "variable_x_orthogonal", "fixed_x_orthogonal", "variable_x_orthogonal_g2"]
        if use_curvature_sampling:
            self.log_message("Using dynamic sampling for orthogonal error methods.")
        else:
            self.log_message("Using linear sampling for euclidean error methods.")

        num_control_points_single_bezier = config.NUM_CONTROL_POINTS_SINGLE_BEZIER  # From central config
        # Preserve original trailing-edge thickness if the loaded airfoil is thickened.
        if self.thickened:
            # Thickness is the vertical gap between final upper and lower data points.
            y_te_upper = float(self.upper_data[-1, 1])
            y_te_lower = float(self.lower_data[-1, 1])
            te_thickness_for_build = abs(y_te_upper - y_te_lower)
            # Guard against numerical noise
            if te_thickness_for_build < 1e-9:
                te_thickness_for_build = 0.0
                self.log_message("Warning: Detected TE marked as thickened but thickness ≈ 0. Building sharp TE instead.")
            else:
                self.log_message(
                    f"Detected thickened trailing edge in input (thickness = {te_thickness_for_build:.6f}). Building model with thick TE."
                )
        else:
            te_thickness_for_build = 0.0  # Build sharp trailing edge when input is sharp

        self.log_message("-" * 20)
        te_desc = "thick" if te_thickness_for_build > 1e-9 else "sharp"
        self.log_message(f"Building {te_desc} single Bezier curves (Order {num_control_points_single_bezier - 1})...")

        # Use stored TE tangent vectors unless te_vector_points is explicitly provided
        if te_vector_points is not None:
            upper_te_tangent_vector, lower_te_tangent_vector = self._calculate_te_tangent(self.upper_data, self.lower_data, te_vector_points)
            # Optionally update stored values if you want to always keep the latest
            self.upper_te_tangent_vector = upper_te_tangent_vector
            self.lower_te_tangent_vector = lower_te_tangent_vector
        else:
            upper_te_tangent_vector = self.upper_te_tangent_vector
            lower_te_tangent_vector = self.lower_te_tangent_vector

        self.log_message(f"Using TE tangent for single Bezier upper: {upper_te_tangent_vector}")
        self.log_message(f"Using TE tangent for single Bezier lower: {lower_te_tangent_vector}")

        le_tangent_upper = np.array([0.0, 1.0]) # Vertical tangent at LE for upper surface
        le_tangent_lower = np.array([0.0, -1.0]) # Vertical tangent at LE for lower surface

        try:
            if enforce_g2:
                # --- Coupled optimisation with G2 continuity ---
                if optimization_method == "minmax":
                    self.log_message("Building coupled G2 Bezier curves with minmax optimization...")
                    self.single_bezier_upper_poly_sharp, self.single_bezier_lower_poly_sharp = build_coupled_venkatamaran_beziers_minmax(
                        original_upper_data=self.upper_data,
                        original_lower_data=self.lower_data,
                        regularization_weight=regularization_weight,
                        te_tangent_vector_upper=upper_te_tangent_vector,
                        te_tangent_vector_lower=lower_te_tangent_vector,
                        optimization_method=optimization_method,
                        logger_func=self.log_message,
                    )
                elif optimization_method in ["variable_x_g2", "variable_x_orthogonal_g2"]:
                    self.log_message("Building coupled G2 Bezier curves with variable-x control points...")
                    self.single_bezier_upper_poly_sharp, self.single_bezier_lower_poly_sharp = build_coupled_venkatamaran_beziers_variable_x(
                        original_upper_data=self.upper_data,
                        original_lower_data=self.lower_data,
                        regularization_weight=regularization_weight,
                        te_tangent_vector_upper=upper_te_tangent_vector,
                        te_tangent_vector_lower=lower_te_tangent_vector,
                        optimization_method=optimization_method,
                        logger_func=self.log_message,
                    )
                else:
                    self.log_message("Building coupled G2 Bezier curves...")
                    self.single_bezier_upper_poly_sharp, self.single_bezier_lower_poly_sharp = build_coupled_venkatamaran_beziers(
                        original_upper_data=self.upper_data,
                        original_lower_data=self.lower_data,
                        regularization_weight=regularization_weight,
                        te_tangent_vector_upper=upper_te_tangent_vector,
                        te_tangent_vector_lower=lower_te_tangent_vector,
                        optimization_method=optimization_method,
                        logger_func=self.log_message,
                    )


            else:
                if optimization_method == "minmax":
                    self.log_message("Building single Bezier curves with minmax optimization...")
                    self.single_bezier_upper_poly_sharp = build_single_venkatamaran_bezier_minmax(
                        original_data=self.upper_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=True,
                        le_tangent_vector=le_tangent_upper,
                        te_tangent_vector=upper_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        num_points_curvature_resample=num_points_curvature_resample,
                        logger_func=self.log_message
                    )

                    self.single_bezier_lower_poly_sharp = build_single_venkatamaran_bezier_minmax(
                        original_data=self.lower_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=False,
                        le_tangent_vector=le_tangent_lower,
                        te_tangent_vector=lower_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        num_points_curvature_resample=num_points_curvature_resample,
                        logger_func=self.log_message
                    )

                elif optimization_method == "variable_x":
                    self.log_message("Building single Bezier curves with median-x control points...")
                    self.single_bezier_upper_poly_sharp = build_single_venkatamaran_bezier(
                        original_data=self.upper_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=True,
                        le_tangent_vector=le_tangent_upper,
                        te_tangent_vector=upper_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        logger_func=self.log_message
                    )
                    self.single_bezier_lower_poly_sharp = build_single_venkatamaran_bezier(
                        original_data=self.lower_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=False,
                        le_tangent_vector=le_tangent_lower,
                        te_tangent_vector=lower_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        logger_func=self.log_message
                    )
                elif optimization_method == "variable_x_orthogonal":
                    self.log_message("Building single Bezier curves with median-x control points and orthogonal error...")
                    self.single_bezier_upper_poly_sharp = build_single_venkatamaran_bezier(
                        original_data=self.upper_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=True,
                        le_tangent_vector=le_tangent_upper,
                        te_tangent_vector=upper_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        logger_func=self.log_message
                    )
                    self.single_bezier_lower_poly_sharp = build_single_venkatamaran_bezier(
                        original_data=self.lower_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=False,
                        le_tangent_vector=le_tangent_lower,
                        te_tangent_vector=lower_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        logger_func=self.log_message
                    )
                else:  # fixed_x
                    self.log_message("Building single Bezier curves with fixed-x optimization...")
                    self.single_bezier_upper_poly_sharp = build_single_venkatamaran_bezier(
                        original_data=self.upper_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=True,
                        le_tangent_vector=le_tangent_upper,
                        te_tangent_vector=upper_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        logger_func=self.log_message
                    )
                    self.single_bezier_lower_poly_sharp = build_single_venkatamaran_bezier(
                        original_data=self.lower_data,
                        num_control_points_new=num_control_points_single_bezier,
                        is_upper_surface=False,
                        le_tangent_vector=le_tangent_lower,
                        te_tangent_vector=lower_te_tangent_vector,
                        regularization_weight=regularization_weight,
                        optimization_method=optimization_method,
                        logger_func=self.log_message
                    )

            # Calculate and store both MSE and Euclidean errors for single Bezier curves
            upper_error_result = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_upper_poly_sharp), self.upper_data, error_function="euclidean", return_max_error=True
            )
            # Ensure we have a tuple result for Euclidean with return_max_error=True
            if isinstance(upper_error_result, tuple) and len(upper_error_result) == 3:
                euclidean_sum_upper, euclidean_max_upper, euclidean_max_upper_idx = upper_error_result
                self.last_single_bezier_upper_max_error = euclidean_max_upper
                self.last_single_bezier_upper_max_error_idx = euclidean_max_upper_idx
            else:
                self.log_message("Warning: Unexpected return format from upper error calculation")
                self.last_single_bezier_upper_max_error = None
                self.last_single_bezier_upper_max_error_idx = None
            
            lower_error_result = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_lower_poly_sharp), self.lower_data, error_function="euclidean", return_max_error=True
            )
            # Ensure we have a tuple result for Euclidean with return_max_error=True
            if isinstance(lower_error_result, tuple) and len(lower_error_result) == 3:
                euclidean_sum_lower, euclidean_max_lower, euclidean_max_lower_idx = lower_error_result
                self.last_single_bezier_lower_max_error = euclidean_max_lower
                self.last_single_bezier_lower_max_error_idx = euclidean_max_lower_idx
            else:
                self.log_message("Warning: Unexpected return format from lower error calculation")
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
        """
        Exports the given airfoil curves to a DXF file.
        The 'merged_flag' determines how LE/TE connections are drawn in DXF.
        """
        dxf_doc = export_curves_to_dxf(
            polygons_to_export,
            chord_length_mm,
            self.log_message
        )

        if dxf_doc:
            self.log_message("DXF document created successfully.")
            return dxf_doc # Return the document for saving in GUI
        else:
            self.log_message("DXF export failed during document creation.")
            return None

    