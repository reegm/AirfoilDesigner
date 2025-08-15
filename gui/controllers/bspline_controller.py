from __future__ import annotations

from typing import Any
import numpy as np

from core.bspline_processor import BSplineProcessor
from core.error_functions import calculate_bspline_fitting_error


class BSplineController:
    """Controller for B-spline operations, following existing architecture."""

    def __init__(self, processor, window: Any):
        self.processor = processor
        self.window = window
        # Reuse the instance created in MainWindow to keep a single source
        self.bspline_processor = getattr(window, "bspline_processor", None) or BSplineProcessor()

    def fit_bspline(self) -> None:
        """Fit B-spline curves to loaded airfoil data."""
        if getattr(self.processor, "upper_data", None) is None or getattr(self.processor, "lower_data", None) is None:
            self.window.status_log.append("No airfoil data loaded. Please load an airfoil first.")
            return

        try:
            # Get control point count from GUI
            num_control_points = int(self.window.optimizer_panel.bspline_cp_spin.value())

            success = self.bspline_processor.fit_bspline(
                self.processor.upper_data,
                self.processor.lower_data,
                num_control_points,
                self.processor.is_trailing_edge_thickened(),
            )

            if success:
                # Calculate and display errors for each surface
                upper_sum_sq, upper_max_err, upper_max_err_idx = calculate_bspline_fitting_error(
                    self.bspline_processor.upper_curve,
                    self.processor.upper_data,
                    error_function="euclidean",
                    return_max_error=True,
                )
                lower_sum_sq, lower_max_err, lower_max_err_idx = calculate_bspline_fitting_error(
                    self.bspline_processor.lower_curve,
                    self.processor.lower_data,
                    error_function="euclidean",
                    return_max_error=True,
                )
                
                # Store max error information for plotting
                self.bspline_processor.last_upper_max_error = upper_max_err
                self.bspline_processor.last_upper_max_error_idx = upper_max_err_idx
                self.bspline_processor.last_lower_max_error = lower_max_err
                self.bspline_processor.last_lower_max_error_idx = lower_max_err_idx
                
                self.window.status_log.append(
                    f"B-spline fit OK. Upper max error: {upper_max_err:.6e}, Lower max error: {lower_max_err:.6e}"
                )
                
                # Trigger plot update with B-spline curves
                self._update_plot_with_bsplines()
            else:
                self.window.status_log.append("B-spline fitting failed.")

        except Exception as e:  # pragma: no cover
            self.window.status_log.append(f"Error during B-spline fitting: {e}")

    def _update_plot_with_bsplines(self) -> None:
        """Update plot to display B-spline curves and control points."""
        # Create plot data with B-spline information
        plot_data = {
            'upper_data': self.processor.upper_data,
            'lower_data': self.processor.lower_data,
            'bspline_upper_curve': self.bspline_processor.upper_curve,
            'bspline_lower_curve': self.bspline_processor.lower_curve,
            'bspline_upper_control_points': self.bspline_processor.upper_control_points,
            'bspline_lower_control_points': self.bspline_processor.lower_control_points,
            'upper_te_tangent_vector': self.processor.upper_te_tangent_vector,
            'lower_te_tangent_vector': self.processor.lower_te_tangent_vector,
        }
        
        # Add B-spline max error information
        if hasattr(self.bspline_processor, 'last_upper_max_error'):
            plot_data['bspline_upper_max_error'] = self.bspline_processor.last_upper_max_error
            plot_data['bspline_upper_max_error_idx'] = self.bspline_processor.last_upper_max_error_idx
            plot_data['bspline_lower_max_error'] = self.bspline_processor.last_lower_max_error
            plot_data['bspline_lower_max_error_idx'] = self.bspline_processor.last_lower_max_error_idx
        
        # Include existing Bezier model data if available
        if hasattr(self.processor, 'upper_poly_sharp') and self.processor.upper_poly_sharp is not None:
            plot_data['single_bezier_upper_poly'] = self.processor.upper_poly_sharp
            plot_data['single_bezier_lower_poly'] = self.processor.lower_poly_sharp
            plot_data['worst_single_bezier_upper_max_error'] = getattr(self.processor, 'last_single_bezier_upper_max_error', None)
            plot_data['worst_single_bezier_upper_max_error_idx'] = getattr(self.processor, 'last_single_bezier_upper_max_error_idx', None)
            plot_data['worst_single_bezier_lower_max_error'] = getattr(self.processor, 'last_single_bezier_lower_max_error', None)
            plot_data['worst_single_bezier_lower_max_error_idx'] = getattr(self.processor, 'last_single_bezier_lower_max_error_idx', None)
        
        # Emit plot update signal
        self.processor.plot_update_requested.emit(plot_data)


