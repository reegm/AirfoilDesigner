import numpy as np
import copy
import logging
import os
from scipy.optimize import minimize

# Central configuration constants
from core import config
from core.optimization_core import build_single_venkatamaran_bezier, calculate_single_bezier_fitting_error
from utils.data_loader import load_airfoil_data, find_shoulder_x_coords
from utils.dxf_exporter import export_curves_to_dxf

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

        self.airfoil_name = None  # Stores profile name from .dat
        self.last_single_bezier_upper_error = None # Stores error for single upper Bezier
        self.last_single_bezier_lower_error = None # Stores error for single lower Bezier


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
        self.last_single_bezier_upper_error = None # Reset
        self.last_single_bezier_lower_error = None # Reset


    def _perform_optimization(self, spacing_weight=0.01, smoothness_weight=0.005):
        """
        Helper method to run the optimization on the current single Bezier model.
        """
        if self.upper_data is None or self.lower_data is None:
            self.log_message("Error: Model or data not loaded for single Bezier optimization.")
            return False

        initial_guess = self.single_bezier_upper_poly_sharp

        upper_te_tangent_vector, lower_te_tangent_vector = self._calculate_te_tangent(self.upper_data, self.lower_data)
        self.log_message(f"Calculated original data TE tangent for upper surface: {upper_te_tangent_vector}")
        self.log_message(f"Calculated original data TE tangent for lower surface: {lower_te_tangent_vector}")

        # Always use SLSQP with MSE for segment optimization
        self.log_message("Running optimization with SLSQP (MSE error)...")
        constraints = None # No constraints for single Bezier

        result = minimize(
            objective_function,
            initial_guess,
            args=(
                self.single_bezier_upper_poly_sharp,
                self.upper_data,
                self.lower_data,
                spacing_weight,
                smoothness_weight,
            ),
            constraints=constraints,
            options=config.SLSQP_OPTIONS,
        )

        if not result.success:
            self.log_message(f"Optimization of single Bezier model failed with SLSQP: {result.message}")
            self.last_single_bezier_upper_error = None # Set to None on failure
            self.last_single_bezier_upper_error_mse = None
            self.last_single_bezier_upper_error_icp = None
            return False
        else:
            self.single_bezier_upper_poly_sharp = result.x
            # Calculate both MSE and ICP errors for display
            self.last_single_bezier_upper_error_mse = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_upper_poly_sharp), self.upper_data, error_function="mse"
            )
            self.last_single_bezier_upper_error_icp = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_upper_poly_sharp), self.upper_data, error_function="icp"
            )
            self.log_message("Final fitting errors for single Bezier curve (sum of squared differences):")
            self.log_message(f"  - Upper: MSE: {self.last_single_bezier_upper_error_mse:.2e}, ICP: {self.last_single_bezier_upper_error_icp:.2e}")
            self.log_message("Single Bezier model optimization complete with SLSQP.")
            return True

    def _calculate_te_tangent(self, upper_data, lower_data):
        """
        Calculates the approximate trailing edge tangent vectors from the original data.
        Returns two normalized vectors: (upper_te_tangent_vector, lower_te_tangent_vector).
        """
        upper_te_tangent_vector = np.array([1.0, 0.0]) # Default to horizontal
        lower_te_tangent_vector = np.array([1.0, 0.0]) # Default to horizontal

        if len(upper_data) >= 2:
            upper_te_vec = upper_data[-1] - upper_data[-2]
            norm_upper = np.linalg.norm(upper_te_vec)
            if norm_upper > 1e-9:
                upper_te_tangent_vector = upper_te_vec / norm_upper
            else:
                self.log_message("Warning: Upper TE tangent vector from data is near zero. Defaulting to horizontal.")
        else:
            self.log_message("Warning: Not enough upper data points to calculate TE tangent. Defaulting to horizontal.")

        if len(lower_data) >= 2:
            lower_te_vec = lower_data[-1] - lower_data[-2]
            norm_lower = np.linalg.norm(lower_te_vec)
            if norm_lower > 1e-9:
                lower_te_tangent_vector = lower_te_vec / norm_lower
            else:
                self.log_message("Warning: Lower TE tangent vector from data is near zero. Defaulting to horizontal.")
        else:
            self.log_message("Warning: Not enough lower data points to calculate TE tangent. Defaulting to horizontal.")

        return upper_te_tangent_vector, lower_te_tangent_vector

    # ------------------------------------------------------------------
    # Trailing-edge thickening helpers (shared by CLI and GUI)
    # ------------------------------------------------------------------

    def apply_te_thickening_to_single_bezier(self, single_bezier_polygons_copy, te_thickness):
        """Return copies of single-Bezier polygons with TE thickening applied."""
        import copy as _copy

        if te_thickness < 1e-9:
            return single_bezier_polygons_copy

        upper_poly, lower_poly = _copy.deepcopy(single_bezier_polygons_copy[0]), _copy.deepcopy(
            single_bezier_polygons_copy[1]
        )

        # Upper surface
        upper_delta_y_at_te = (te_thickness / 2.0) - upper_poly[-1][1]
        for i in range(1, len(upper_poly) - 1):
            if upper_poly[i][0] > 1e-9:
                upper_poly[i][1] += (upper_poly[i][0]) * upper_delta_y_at_te  # chord length 1
        upper_poly[-1][1] = te_thickness / 2.0

        # Lower surface
        lower_delta_y_at_te = (-te_thickness / 2.0) - lower_poly[-1][1]
        for i in range(1, len(lower_poly) - 1):
            if lower_poly[i][0] > 1e-9:
                lower_poly[i][1] += (lower_poly[i][0]) * lower_delta_y_at_te
        lower_poly[-1][1] = -te_thickness / 2.0

        return [upper_poly, lower_poly]

    def build_single_bezier_model(self, regularization_weight, error_function="mse"):
        """
        Builds the single-span Bezier curves for upper and lower surfaces based on the 2017 Venkataraman paper.
        This method always builds a sharp (thickness 0) single Bezier curve.
        Thickening is applied separately for display.
        """
        if self.upper_data is None or self.lower_data is None:
            self.log_message("Error: Original airfoil data not loaded. Cannot build single Bezier model.")
            return False

        num_control_points_single_bezier = config.NUM_CONTROL_POINTS_SINGLE_BEZIER  # From central config
        te_thickness_for_build = 0.0 # Always build a sharp version initially

        self.log_message("-" * 20)
        self.log_message(f"Building sharp single Bezier curves (Order {num_control_points_single_bezier - 1})...")

        upper_te_tangent_vector, lower_te_tangent_vector = self._calculate_te_tangent(self.upper_data, self.lower_data)
        self.log_message(f"Calculated original data TE tangent for single Bezier upper: {upper_te_tangent_vector}")
        self.log_message(f"Calculated original data TE tangent for single Bezier lower: {lower_te_tangent_vector}")

        le_tangent_upper = np.array([0.0, 1.0]) # Vertical tangent at LE for upper surface
        le_tangent_lower = np.array([0.0, -1.0]) # Vertical tangent at LE for lower surface

        upper_le_point = np.array([0.0, 0.0])
        upper_te_point_final = np.array([1.0, te_thickness_for_build / 2.0])

        lower_le_point = np.array([0.0, 0.0])
        lower_te_point_final = np.array([1.0, -te_thickness_for_build / 2.0])

        try:
            if error_function == "icp_iter_single":
                from core.optimization_core import calculate_iterative_icp_error_single_bezier
                self.log_message("Running true iterative ICP for single Bezier upper curve...")
                paper_fixed_x_coords_upper = np.array([0.0, 0.0, 0.11422, 0.25294, 0.37581, 0.49671, 0.61942, 0.74701, 0.88058, 1.0])
                upper_poly = [upper_le_point]
                upper_poly += [np.array([x, 0.0]) for x in paper_fixed_x_coords_upper[1:-1]]
                upper_poly.append(upper_te_point_final)
                upper_poly = np.array(upper_poly)
                # Fit y-coords using ICP
                final_upper_poly, icp_error_upper = calculate_iterative_icp_error_single_bezier(self.upper_data, upper_poly)
                self.single_bezier_upper_poly_sharp = final_upper_poly
                self.log_message(f"Final ICP error (upper) after iterative fit: {icp_error_upper:.2e}")

                self.log_message("Running true iterative ICP for single Bezier lower curve...")
                paper_fixed_x_coords_lower = np.array([0.0, 0.0, 0.12325, 0.25314, 0.37519, 0.49569, 0.61975, 0.74391, 0.87391, 1.0])
                lower_poly = [lower_le_point]
                lower_poly += [np.array([x, 0.0]) for x in paper_fixed_x_coords_lower[1:-1]]
                lower_poly.append(lower_te_point_final)
                lower_poly = np.array(lower_poly)
                final_lower_poly, icp_error_lower = calculate_iterative_icp_error_single_bezier(self.lower_data, lower_poly)
                self.single_bezier_lower_poly_sharp = final_lower_poly
                self.log_message(f"Final ICP error (lower) after iterative fit: {icp_error_lower:.2e}")
            else:
                self.single_bezier_upper_poly_sharp = build_single_venkatamaran_bezier(
                    original_data=self.upper_data,
                    num_control_points_new=num_control_points_single_bezier,
                    start_point=upper_le_point,
                    end_point=upper_te_point_final,
                    is_upper_surface=True,
                    le_tangent_vector=le_tangent_upper,
                    te_tangent_vector=upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function
                )

                self.single_bezier_lower_poly_sharp = build_single_venkatamaran_bezier(
                    original_data=self.lower_data,
                    num_control_points_new=num_control_points_single_bezier,
                    start_point=lower_le_point,
                    end_point=lower_te_point_final,
                    is_upper_surface=False,
                    le_tangent_vector=le_tangent_lower,
                    te_tangent_vector=lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function
                )

            # Calculate and store both MSE and ICP errors for single Bezier curves
            self.last_single_bezier_upper_error_mse = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_upper_poly_sharp), self.upper_data, error_function="mse"
            )
            self.last_single_bezier_upper_error_icp = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_upper_poly_sharp), self.upper_data, error_function="icp"
            )
            self.last_single_bezier_lower_error_mse = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_lower_poly_sharp), self.lower_data, error_function="mse"
            )
            self.last_single_bezier_lower_error_icp = calculate_single_bezier_fitting_error(
                np.array(self.single_bezier_lower_poly_sharp), self.lower_data, error_function="icp"
            )
            # For backward compatibility, keep the last used error as the default
            if error_function == "mse":
                self.last_single_bezier_upper_error = self.last_single_bezier_upper_error_mse
                self.last_single_bezier_lower_error = self.last_single_bezier_lower_error_mse
            else:
                self.last_single_bezier_upper_error = self.last_single_bezier_upper_error_icp
                self.last_single_bezier_lower_error = self.last_single_bezier_lower_error_icp
            self.log_message("Sharp single Bezier curves built successfully.")
            return True
        except Exception as e:
            self.log_message(f"Error during sharp single Bezier curve building: {e}")
            self.single_bezier_upper_poly_sharp = None
            self.single_bezier_lower_poly_sharp = None
            self.last_single_bezier_upper_error = None # Set to None on failure
            self.last_single_bezier_lower_error = None # Set to None on failure
            self.last_single_bezier_upper_error_mse = None
            self.last_single_bezier_upper_error_icp = None
            self.last_single_bezier_lower_error_mse = None
            self.last_single_bezier_lower_error_icp = None
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

    