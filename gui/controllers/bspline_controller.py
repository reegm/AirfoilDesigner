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

            # e.g., in your controller before calling fit_bspline(...)
            self.bspline_processor.knot_end_bias = 0.5   # knots cluster more at both ends
            # self.bspline_processor.param_end_bias = 0.3  # parameters mildly cluster at both ends

            success = self.bspline_processor.fit_bspline(
                self.processor.upper_data,
                self.processor.lower_data,
                num_control_points,
                self.processor.is_trailing_edge_thickened(),
                self.processor.upper_te_tangent_vector,
                self.processor.lower_te_tangent_vector,
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
                
                num_spans = num_control_points - self.bspline_processor.degree
                self.window.status_log.append(
                    f"B-spline fit OK (degree {self.bspline_processor.degree}, {num_spans} spans). "
                    f"Upper max error: {upper_max_err:.6e}, Lower max error: {lower_max_err:.6e}"
                )
                
                # Trigger plot update with B-spline curves
                self._update_plot_with_bsplines()
            else:
                self.window.status_log.append("B-spline fitting failed.")

        except Exception as e:  # pragma: no cover
            self.window.status_log.append(f"Error during B-spline fitting: {e}")

    def _update_plot_with_bsplines(self) -> None:
        """Update plot to display B-spline curves and control points."""
        # Get comb parameters from the UI
        comb_scale = self.window.comb_panel.comb_scale_slider.value() / 1000.0
        comb_density = self.window.comb_panel.comb_density_slider.value()
        
        # Calculate B-spline comb data
        comb_bspline = self.bspline_processor.calculate_curvature_comb_data(
            num_points_per_segment=comb_density,
            scale_factor=comb_scale,
        )
        
        # Create plot data with B-spline information
        plot_data = {
            'upper_data': self.processor.upper_data,
            'lower_data': self.processor.lower_data,
            'bspline_upper_curve': self.bspline_processor.upper_curve,
            'bspline_lower_curve': self.bspline_processor.lower_curve,
            'bspline_upper_control_points': self.bspline_processor.upper_control_points,
            'bspline_lower_control_points': self.bspline_processor.lower_control_points,
            'comb_bspline': comb_bspline,
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
            
            # Include Bezier comb data if available (from thickened model)
            if hasattr(self.processor, '_is_thickened') and self.processor._is_thickened:
                if hasattr(self.processor, '_thickened_single_bezier_polygons') and self.processor._thickened_single_bezier_polygons:
                    # Get comb parameters from UI
                    comb_scale = self.window.comb_panel.comb_scale_slider.value() / 1000.0
                    comb_density = self.window.comb_panel.comb_density_slider.value()
                    
                    # Calculate Bezier comb data
                    plot_data['comb_single_bezier'] = self.processor._calculate_curvature_comb_data(
                        self.processor._thickened_single_bezier_polygons,
                        num_points_per_segment=comb_density,
                        scale_factor=comb_scale,
                    )
        
        # Emit plot update signal
        self.processor.plot_update_requested.emit(plot_data)


