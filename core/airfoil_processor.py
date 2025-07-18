import numpy as np
import copy
from PySide6.QtCore import QObject, Signal
import logging

from core.core_logic import CoreProcessor
from utils.bezier_utils import general_bezier_curve, bezier_derivative, bezier_curvature


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

        self.core_processor = CoreProcessor(self.log_message.emit)
        self._is_thickened = False # True if thickening is currently applied
        self._thickened_single_bezier_polygons = None # Stores thickened single Bezier polygons
        self._last_plot_data = None # Cache for the last plot data dictionary
        self._current_te_vector_points = None  # Store current TE vector points setting
        self._is_trailing_edge_thickened = False # True if original airfoil has thickened TE


    def load_airfoil_data_and_initialize_model(self, file_path):
        """
        Loads airfoil data and initializes the AirfoilModel.
        Resets internal flags and state.
        """
        self._is_thickened = False
        self._thickened_single_bezier_polygons = None
        self._last_plot_data = None
        self.core_processor._reset_state()
        self._is_trailing_edge_thickened = False

        if self.core_processor.load_airfoil_data_and_initialize_model(file_path):
            # After loading, check if the trailing edge is thickened
            self._is_trailing_edge_thickened = getattr(self.core_processor, 'thickened', False)
            self.log_message.emit("Airfoil data loaded.")
            self._request_plot_update()
            return True
        else:
            self.log_message.emit("Failed to load or initialize airfoil data.")
            return False

    def is_trailing_edge_thickened(self):
        """Returns True if the loaded airfoil has a thickened trailing edge."""
        return self._is_trailing_edge_thickened

    def build_single_bezier_model(self, regularization_weight, error_function="orthogonal_minmax", enforce_g2=False, num_points_curve_error=None, te_vector_points=None):
        """
        Builds the single-span Bezier curves for upper and lower surfaces based on the 2017 Venkataraman paper.
        This method always builds a sharp (thickness 0) single Bezier curve.
        Thickening is applied separately for display.
        Only 'icp' error function is supported.
        """
        import time  # <-- Add import for timing
        if self.core_processor.upper_data is None or self.core_processor.lower_data is None:
            self.log_message.emit("Error: Original airfoil data not loaded. Cannot build single Bezier model.")
            return False

        # Store the TE vector points setting for future plot updates
        self._current_te_vector_points = te_vector_points

        self.log_message.emit("Building single Bezier model...")
        start_time = time.perf_counter()  # Start timing
        result = self.core_processor.build_single_bezier_model(
            regularization_weight,
            error_function=error_function,
            enforce_g2=enforce_g2,
            num_points_curve_error=num_points_curve_error,
            te_vector_points=te_vector_points
        )
        elapsed = time.perf_counter() - start_time  # End timing
        if result:
            self.log_message.emit(f"Single Bezier model built successfully. ({elapsed:.3f} seconds)")
            # Log error values displayed in the graph
            upper_err = getattr(self.core_processor, 'last_single_bezier_upper_max_error', None)
            upper_idx = getattr(self.core_processor, 'last_single_bezier_upper_max_error_idx', None)
            lower_err = getattr(self.core_processor, 'last_single_bezier_lower_max_error', None)
            lower_idx = getattr(self.core_processor, 'last_single_bezier_lower_max_error_idx', None)
            if upper_err is not None and upper_idx is not None:
                self.log_message.emit(f"Upper max error: {upper_err:.4e} at index {upper_idx}")
            if lower_err is not None and lower_idx is not None:
                self.log_message.emit(f"Lower max error: {lower_err:.4e} at index {lower_idx}")
            self._request_plot_update()
            return True
        else:
            self.log_message.emit("Failed to build single Bezier model.")
            return False

    def toggle_thickening(self, te_thickness_percent):
        """
        Applies or removes trailing edge thickening.
        """
        if self.core_processor.upper_data is None or self.core_processor.lower_data is None:
            self.log_message.emit("Error: Please load an airfoil file first to apply/remove thickening.")
            return False

        if not self._is_thickened:
            # Apply thickening
            te_thickness = te_thickness_percent / 100.0
            self.log_message.emit(f"Applying {te_thickness_percent:.2f}% trailing edge thickness...")

            # Thickening for single Bezier model
            if self.core_processor.single_bezier_upper_poly_sharp is not None and \
               self.core_processor.single_bezier_lower_poly_sharp is not None:
                single_bezier_polygons_copy = [
                    copy.deepcopy(self.core_processor.single_bezier_upper_poly_sharp),
                    copy.deepcopy(self.core_processor.single_bezier_lower_poly_sharp),
                ]
                self._thickened_single_bezier_polygons = self.core_processor.apply_te_thickening_to_single_bezier(
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


    def request_plot_update_with_comb_params(self, scale, density):
        """Public method to trigger a plot update with specific comb parameters."""
        if self._last_plot_data is None:
            self.log_message.emit("Cannot update comb parameters: no model has been generated yet.")
            return

        updated_plot_data = self._last_plot_data.copy()

        # Use stored TE tangent vectors from core_processor
        upper_te_tangent_vector = self.core_processor.upper_te_tangent_vector
        lower_te_tangent_vector = self.core_processor.lower_te_tangent_vector

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
        if self.core_processor.upper_data is None or self.core_processor.lower_data is None:
            self.log_message.emit("No airfoil data available to plot.")
            return

        # Use stored TE tangent vectors from core_processor
        upper_te_tangent_vector = self.core_processor.upper_te_tangent_vector
        lower_te_tangent_vector = self.core_processor.lower_te_tangent_vector

        plot_data = {
            'upper_data': self.core_processor.upper_data,
            'lower_data': self.core_processor.lower_data,
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
        if self.core_processor.single_bezier_upper_poly_sharp is not None:
            plot_data['single_bezier_upper_poly'] = self.core_processor.single_bezier_upper_poly_sharp
            plot_data['single_bezier_lower_poly'] = self.core_processor.single_bezier_lower_poly_sharp
            plot_data['worst_single_bezier_upper_max_error'] = getattr(self.core_processor, 'last_single_bezier_upper_max_error', None)
            plot_data['worst_single_bezier_upper_max_error_idx'] = getattr(self.core_processor, 'last_single_bezier_upper_max_error_idx', None)
            plot_data['worst_single_bezier_lower_max_error'] = getattr(self.core_processor, 'last_single_bezier_lower_max_error', None)
            plot_data['worst_single_bezier_lower_max_error_idx'] = getattr(self.core_processor, 'last_single_bezier_lower_max_error_idx', None)

            if self._is_thickened and self._thickened_single_bezier_polygons:
                plot_data['thickened_single_bezier_upper_poly'] = self._thickened_single_bezier_polygons[0]
                plot_data['thickened_single_bezier_lower_poly'] = self._thickened_single_bezier_polygons[1]
                plot_data['comb_single_bezier'] = self._calculate_curvature_comb_data(self._thickened_single_bezier_polygons)
            else:
                plot_data['comb_single_bezier'] = self._calculate_curvature_comb_data([
                    self.core_processor.single_bezier_upper_poly_sharp,
                    self.core_processor.single_bezier_lower_poly_sharp
                ])

        self._last_plot_data = plot_data.copy()
        self.plot_update_requested.emit(plot_data)

    def recalculate_te_vectors_and_update_plot(self, te_vector_points):
        """
        Recalculate the trailing edge tangent vectors using the specified number of points
        and update the plot. Does not rebuild the Bezier model.
        """
        if self.core_processor.upper_data is None or self.core_processor.lower_data is None:
            self.log_message.emit("Error: No airfoil data loaded. Cannot recalculate TE vectors.")
            return
        upper_te_tangent_vector, lower_te_tangent_vector = self.core_processor._calculate_te_tangent(
            self.core_processor.upper_data, self.core_processor.lower_data, te_vector_points
        )
        # Update the stored TE tangent vectors in CoreProcessor
        self.core_processor.upper_te_tangent_vector = upper_te_tangent_vector
        self.core_processor.lower_te_tangent_vector = lower_te_tangent_vector
        plot_data = {
            'upper_data': self.core_processor.upper_data,
            'lower_data': self.core_processor.lower_data,
            'upper_te_tangent_vector': upper_te_tangent_vector,
            'lower_te_tangent_vector': lower_te_tangent_vector,
            # The rest of the plot data is left as None or not updated
        }
        self.plot_update_requested.emit(plot_data)
        self.log_message.emit(f"Trailing edge vectors recalculated with {te_vector_points} points.")
