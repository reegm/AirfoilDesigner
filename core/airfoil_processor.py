import numpy as np
import copy
from PySide6.QtCore import QObject, Signal
import logging

from core import config
from utils.bezier_utils import general_bezier_curve, bezier_derivative, bezier_curvature
# --- New import for refactored optimizer ---
from core.solver.bezier_optimizer import build_bezier_single_fixed_x_msr, build_bezier_single_fixed_x_minmax_xy
from core.solver.error_functions import calculate_single_bezier_fitting_error
from utils.data_loader import load_airfoil_data
from utils.dxf_exporter import export_curves_to_dxf


class SignalLogHandler(logging.Handler):
    """A logging handler that emits a Qt signal."""
    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        msg = self.format(record)
        self.signal_emitter.emit(msg)

class AirfoilProcessor(QObject):
    """
    Acts as a bridge between the GUI and the CoreProcessor.
    It holds an instance of the CoreProcessor and uses Qt Signals
    to communicate with the GUI.
    """
    log_message = Signal(str)
    plot_update_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Legacy core processor removed. Use direct attributes.
        self.upper_data = None
        self.lower_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._is_thickened = False # True if thickening is currently applied
        self._thickened_single_bezier_polygons = None # Stores thickened single Bezier polygons
        self._last_plot_data = None # Cache for the last plot data dictionary
        self._current_te_vector_points = None  # Store current TE vector points setting
        self._is_trailing_edge_thickened = False # True if original airfoil has thickened TE
        self.upper_poly_sharp = None
        self.lower_poly_sharp = None
        self.last_single_bezier_upper_max_error = None
        self.last_single_bezier_upper_max_error_idx = None
        self.last_single_bezier_lower_max_error = None
        self.last_single_bezier_lower_max_error_idx = None


    def load_airfoil_data_and_initialize_model(self, file_path):
        """
        Loads airfoil data and initializes the AirfoilModel.
        Resets internal flags and state.
        """
        self._is_thickened = False
        self._thickened_single_bezier_polygons = None
        self._last_plot_data = None
        self.upper_data = None
        self.lower_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._is_trailing_edge_thickened = False
        self.last_single_bezier_upper_max_error = None
        self.last_single_bezier_upper_max_error_idx = None
        self.last_single_bezier_lower_max_error = None
        self.last_single_bezier_lower_max_error_idx = None
        self.upper_poly_sharp = None
        self.lower_poly_sharp = None

        try:
            upper, lower, airfoil_name, thickened = load_airfoil_data(file_path, logger_func=self.log_message.emit)
            self.upper_data = upper
            self.lower_data = lower
            self.airfoil_name = airfoil_name
            self._is_trailing_edge_thickened = thickened
            # Recalculate TE tangent vectors using default (last 3 points)
            te_vector_points = 3
            self.upper_te_tangent_vector, self.lower_te_tangent_vector = self._calculate_te_tangent(
                self.upper_data, self.lower_data, te_vector_points)
            self.log_message.emit("Airfoil data loaded.")
            self._request_plot_update()
            return True
        except Exception as e:
            self.log_message.emit(f"Failed to load or initialize airfoil data: {e}")
            return False

    def is_trailing_edge_thickened(self):
        """Returns True if the loaded airfoil has a thickened trailing edge."""
        return self._is_trailing_edge_thickened

    def toggle_thickening(self, te_thickness_percent):
        """
        Applies or removes trailing edge thickening.
        """
        if self.upper_data is None or self.lower_data is None:
            self.log_message.emit("Error: Please load an airfoil file first to apply/remove thickening.")
            return False

        if not self._is_thickened:
            # Apply thickening
            te_thickness = te_thickness_percent / 100.0
            self.log_message.emit(f"Applying {te_thickness_percent:.2f}% trailing edge thickness...")

            # Thickening for single Bezier model
            if self.upper_poly_sharp is not None and \
               self.lower_poly_sharp is not None:
                single_bezier_polygons_copy = [
                    copy.deepcopy(self.upper_poly_sharp),
                    copy.deepcopy(self.lower_poly_sharp),
                ]
                self._thickened_single_bezier_polygons = self.apply_te_thickening_to_single_bezier(
                    single_bezier_polygons_copy,
                    te_thickness,
                )
            else:
                self.log_message.emit("Warning: Sharp single Bezier model not built, cannot apply thickening to it.")

            if self._thickened_single_bezier_polygons:
                self._is_thickened = True
                self.log_message.emit("Thickening applied successfully.")
            else:
                self.log_message.emit("No models available to apply thickening to.")

        else:
            # Remove thickening
            self.log_message.emit("Removing trailing edge thickening...")
            self._is_thickened = False
            self._thickened_single_bezier_polygons = None
            self.log_message.emit("Thickening removed. Displaying sharp models.")

        self._request_plot_update()
        return True

    def export_to_dxf(self, file_path, chord_length_mm):
        """
        Export the current Bezier model(s) as a DXF file.
        """
        if self._is_thickened and self._thickened_single_bezier_polygons:
            polygons_to_export = self._thickened_single_bezier_polygons
            self.log_message.emit("Preparing to export thickened single Bezier model.")
        elif self.upper_poly_sharp is not None and self.lower_poly_sharp is not None:
            polygons_to_export = [self.upper_poly_sharp, self.lower_poly_sharp]
            self.log_message.emit("Preparing to export sharp single Bezier model.")
        else:
            self.log_message.emit("Error: Single Bezier model not available for export. Please build it first.")
            return False

        dxf_doc = export_curves_to_dxf(
            polygons_to_export,
            chord_length_mm,
            self.log_message.emit,
        )

        if not dxf_doc:
            self.log_message.emit("Single Bezier DXF export failed during document creation.")
            return False

        try:
            dxf_doc.saveas(file_path)
            self.log_message.emit(f"DXF file exported to: {file_path}")
            return True
        except Exception as e:
            self.log_message.emit(f"Error saving DXF file: {e}")
            return False

    def apply_te_thickening_to_single_bezier(self, polygons, te_thickness):
        """
        Apply trailing edge thickening to a pair of Bezier polygons.
        Offsets the last control point of each polygon by te_thickness/2 in y (upper +, lower -).
        """
        if not polygons or len(polygons) != 2:
            self.log_message.emit("[TE Thickening] Invalid polygons input.")
            return polygons
        upper_poly = np.array(polygons[0], copy=True)
        lower_poly = np.array(polygons[1], copy=True)
        # Offset the last control point in y
        upper_poly[-1, 1] += te_thickness / 2.0
        lower_poly[-1, 1] -= te_thickness / 2.0
        self.log_message.emit(f"Applied trailing edge thickening: {te_thickness:.4f}")
        return [upper_poly, lower_poly]

    def request_plot_update_with_comb_params(self, scale, density):
        """Public method to trigger a plot update with specific comb parameters."""
        if self._last_plot_data is None:
            self.log_message.emit("Cannot update comb parameters: no model has been generated yet.")
            return

        updated_plot_data = self._last_plot_data.copy()

        # Use stored TE tangent vectors from core_processor
        upper_te_tangent_vector = self.upper_te_tangent_vector
        lower_te_tangent_vector = self.lower_te_tangent_vector

        # Determine which polygons to use for the single Bezier comb
        polygons_single_bezier = []
        
        upper_poly = updated_plot_data.get('thickened_single_bezier_upper_poly')
        if upper_poly is None:
            upper_poly = updated_plot_data.get('single_bezier_upper_poly')

        lower_poly = updated_plot_data.get('thickened_single_bezier_lower_poly')
        if lower_poly is None:
            lower_poly = updated_plot_data.get('single_bezier_lower_poly')

        if upper_poly is not None and lower_poly is not None:
            polygons_single_bezier = [upper_poly, lower_poly]
        
        if polygons_single_bezier:
            updated_plot_data['comb_single_bezier'] = self._calculate_curvature_comb_data(
                polygons_single_bezier,
                num_points_per_segment=density,
                scale_factor=scale
            )
        else:
            updated_plot_data['comb_single_bezier'] = None # Ensure it's cleared if no model

        # Emit signal with the newly calculated comb data, but don't update the cache with this
        # The cache should only hold data from a full model generation/update
        self.plot_update_requested.emit(updated_plot_data)


    def _calculate_curvature_comb_data(self, polygons, num_points_per_segment=40, scale_factor=0.05):
        """
        Calculates the curvature comb lines for a set of Bezier polygons.
        Returns a list of lists, where each inner list contains the hair segments for one polygon.
        """
        if not polygons:
            return None

        all_polygons_combs = []

        for poly in polygons:
            poly_comb_hairs = []
            poly = np.array(poly)
            if poly.shape[0] < 2:
                all_polygons_combs.append([]) # Append empty list for this polygon
                continue

            t_vals = np.linspace(0, 1, num_points_per_segment)
            curve_points = general_bezier_curve(t_vals, poly)
            derivatives = bezier_derivative(t_vals, poly, order=1)
            curvatures = bezier_curvature(t_vals, poly)

            # Normalize tangents to get unit tangents
            norms = np.linalg.norm(derivatives, axis=1)
            unit_tangents = np.divide(derivatives, norms[:, np.newaxis], out=np.zeros_like(derivatives), where=norms[:, np.newaxis] != 0)

            # Get normal vectors (rotate tangent by 90 degrees)
            # The direction of the comb is now handled by the sign of the curvature.
            # We consistently use the normal that points "outward" from the tangent's direction of travel.
            normals = np.zeros_like(unit_tangents)
            normals[:, 0] = -unit_tangents[:, 1]
            normals[:, 1] = unit_tangents[:, 0]

            # Invert the curvature so that combs point outwards for convex and inwards for concave
            comb_lengths = -curvatures * scale_factor
            end_points = curve_points + normals * comb_lengths[:, np.newaxis]

            # Create individual hair segments as separate line data
            for j in range(num_points_per_segment):
                hair_segment = np.array([curve_points[j], end_points[j]])
                poly_comb_hairs.append(hair_segment)
            
            all_polygons_combs.append(poly_comb_hairs)

        return all_polygons_combs if all_polygons_combs else None


    def _request_plot_update(self):
        """Emits a signal to request a plot update with current model data from the core."""
        if self.upper_data is None or self.lower_data is None:
            self.log_message.emit("No airfoil data available to plot.")
            return

        # Use stored TE tangent vectors from core_processor
        upper_te_tangent_vector = self.upper_te_tangent_vector
        lower_te_tangent_vector = self.lower_te_tangent_vector

        plot_data = {
            'upper_data': self.upper_data,
            'lower_data': self.lower_data,
            'upper_te_tangent_vector': upper_te_tangent_vector,
            'lower_te_tangent_vector': lower_te_tangent_vector,
            'worst_single_bezier_upper_max_error': None,
            'worst_single_bezier_lower_max_error': None,
            'single_bezier_upper_poly': None,
            'single_bezier_lower_poly': None,
            'thickened_single_bezier_upper_poly': None,
            'thickened_single_bezier_lower_poly': None,
            'comb_single_bezier': None,
        }

        # --- Populate Single Bezier Model Data ---
        if self.upper_poly_sharp is not None:
            plot_data['single_bezier_upper_poly'] = self.upper_poly_sharp
            plot_data['single_bezier_lower_poly'] = self.lower_poly_sharp
            plot_data['worst_single_bezier_upper_max_error'] = getattr(self, 'last_single_bezier_upper_max_error', None)
            plot_data['worst_single_bezier_upper_max_error_idx'] = getattr(self, 'last_single_bezier_upper_max_error_idx', None)
            plot_data['worst_single_bezier_lower_max_error'] = getattr(self, 'last_single_bezier_lower_max_error', None)
            plot_data['worst_single_bezier_lower_max_error_idx'] = getattr(self, 'last_single_bezier_lower_max_error_idx', None)

            if self._is_thickened and self._thickened_single_bezier_polygons:
                plot_data['thickened_single_bezier_upper_poly'] = self._thickened_single_bezier_polygons[0]
                plot_data['thickened_single_bezier_lower_poly'] = self._thickened_single_bezier_polygons[1]
                plot_data['comb_single_bezier'] = self._calculate_curvature_comb_data(self._thickened_single_bezier_polygons)
            else:
                plot_data['comb_single_bezier'] = self._calculate_curvature_comb_data([
                    self.upper_poly_sharp,
                    self.lower_poly_sharp
                ])

        self._last_plot_data = plot_data.copy()
        self.plot_update_requested.emit(plot_data)

    def recalculate_te_vectors_and_update_plot(self, te_vector_points):
        """
        Recalculate the trailing edge tangent vectors using the specified number of points
        and update the plot. Does not rebuild the Bezier model.
        """
        if self.upper_data is None or self.lower_data is None:
            self.log_message.emit("Error: No airfoil data loaded. Cannot recalculate TE vectors.")
            return
        upper_te_tangent_vector, lower_te_tangent_vector = self._calculate_te_tangent(
            self.upper_data, self.lower_data, te_vector_points
        )
        # Update the stored TE tangent vectors in CoreProcessor
        self.upper_te_tangent_vector = upper_te_tangent_vector
        self.lower_te_tangent_vector = lower_te_tangent_vector
        plot_data = {
            'upper_data': self.upper_data,
            'lower_data': self.lower_data,
            'upper_te_tangent_vector': upper_te_tangent_vector,
            'lower_te_tangent_vector': lower_te_tangent_vector,
            # The rest of the plot data is left as None or not updated
        }
        self.plot_update_requested.emit(plot_data)
        self.log_message.emit(f"Trailing edge vectors recalculated with {te_vector_points} points.")

    def _calculate_te_tangent(self, upper_data, lower_data, te_vector_points):
        """
        Calculate trailing edge tangent vectors for upper and lower surfaces using the last N points.
        Returns (upper_te_tangent_vector, lower_te_tangent_vector)
        """
        def tangent(data, n):
            # Use the last n points to estimate the tangent at the trailing edge
            if n < 2 or len(data) < n:
                n = min(3, len(data))
            pts = data[-n:]
            dx = pts[-1, 0] - pts[0, 0]
            dy = pts[-1, 1] - pts[0, 1]
            norm = np.hypot(dx, dy)
            if norm == 0:
                return np.array([1.0, 0.0])
            return np.array([dx, dy]) / norm
        upper_te_tangent = tangent(upper_data, te_vector_points)
        lower_te_tangent = tangent(lower_data, te_vector_points)
        return upper_te_tangent, lower_te_tangent
