from __future__ import annotations

from typing import Any
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from core import config
from core.bspline_processor import BSplineProcessor


class BSplineController:
    """Controller for B-spline operations, following existing architecture."""
    
    def __init__(self, processor, window: Any):
        self.processor = processor
        self.window = window
        # Reuse the instance created in MainWindow to keep a single source
        self.bspline_processor = getattr(window, "bspline_processor", None) or BSplineProcessor()
        # Store the B-spline processor in the window for access by other controllers
        self.window.bspline_processor = self.bspline_processor

    

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

            # Get G2 flag from GUI checkbox
            enforce_g2 = self.window.optimizer_panel.g2_checkbox.isChecked()
            
            # Get TE tangency flag from GUI checkbox
            enforce_te_tangency = self.window.optimizer_panel.enforce_te_tangency_checkbox.isChecked()
            
            print(f"[DEBUG] Controller: enforce_te_tangency checkbox state: {enforce_te_tangency}")
            
            success = self.bspline_processor.fit_bspline(
                self.processor.upper_data,
                self.processor.lower_data,
                num_control_points,
                self.processor.is_trailing_edge_thickened(),
                self.processor.upper_te_tangent_vector,
                self.processor.lower_te_tangent_vector,
                enforce_g2=enforce_g2,
                enforce_te_tangency=enforce_te_tangency,
            )

            if success:
                # Log G2 setting used
                g2_status = "enabled" if enforce_g2 else "disabled"
                te_tangency_status = "enabled" if enforce_te_tangency else "disabled"
                self.window.status_log.append(f"B-spline fitting with G2 continuity {g2_status}, TE tangency {te_tangency_status}")
                
                # Calculate and display errors for each surface
                upper_sum_sq, upper_max_err, upper_max_err_idx = self.calculate_bspline_fitting_error(
                    self.bspline_processor.upper_curve,
                    self.processor.upper_data,
                    return_max_error=True,
                )
                lower_sum_sq, lower_max_err, lower_max_err_idx = self.calculate_bspline_fitting_error(
                    self.bspline_processor.lower_curve,
                    self.processor.lower_data,
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

    def calculate_bspline_fitting_error(
                self,
                bspline_curve: BSpline,
                original_data: np.ndarray,
                *,
                return_max_error: bool = False,
                return_all: bool = False,
            ):
            """
            Calculate fitting error for a B-spline curve against original data.
            """
            # Approximate orthogonal by dense sampling
            num_points_curve = config.NUM_POINTS_CURVE_ERROR
            t_samples = np.linspace(0.0, 1.0, num_points_curve)
            if len(t_samples) > 0:
                t_samples[-1] = min(t_samples[-1], 1.0 - 1e-12)
            sampled_curve_points = bspline_curve(t_samples)
            sampled_curve_points = sampled_curve_points[np.argsort(sampled_curve_points[:, 0])]
            tree = cKDTree(sampled_curve_points)
            min_dists, _ = tree.query(original_data, k=1)
            sum_sq = float(np.sum(min_dists ** 2))
            if return_all:
                rms = float(np.sqrt(np.mean(min_dists ** 2)))
                return min_dists, rms, (sum_sq, int(np.argmax(min_dists)))
            if return_max_error:
                max_error = float(np.max(min_dists))
                max_error_idx = int(np.argmax(min_dists))
                return sum_sq, max_error, max_error_idx
            return sum_sq
    

    def apply_te_thickening(self, te_thickness_percent: float) -> bool:
        """
        Apply trailing edge thickening to fitted B-splines.
        
        Args:
            te_thickness_percent: The thickness percentage to apply (0.0 to 100.0)
            
        Returns:
            bool: True if thickening was applied successfully, False otherwise
        """
        if not self.bspline_processor.is_fitted():
            self.window.status_log.append("No B-spline model fitted. Please fit B-splines first.")
            return False
        
        try:
            # Convert percentage to decimal
            te_thickness = te_thickness_percent / 100.0
            
            success = self.bspline_processor.apply_te_thickening(te_thickness)
            
            if success:
                self.window.status_log.append(f"Applied {te_thickness_percent:.2f}% trailing edge thickness to B-splines.")
                # Update the plot with thickened B-splines
                self._update_plot_with_bsplines()
                return True
            else:
                self.window.status_log.append("Failed to apply trailing edge thickening to B-splines.")
                return False
                
        except Exception as e:
            self.window.status_log.append(f"Error applying trailing edge thickening to B-splines: {e}")
            return False

    def remove_te_thickening(self) -> bool:
        """
        Remove trailing edge thickening from fitted B-splines.
        
        Returns:
            bool: True if thickening was removed successfully, False otherwise
        """
        if not self.bspline_processor.is_fitted():
            self.window.status_log.append("No B-spline model fitted. Please fit B-splines first.")
            return False
        
        try:
            success = self.bspline_processor.remove_te_thickening()
            
            if success:
                self.window.status_log.append("Removed trailing edge thickening from B-splines.")
                # Update the plot with sharp B-splines
                self._update_plot_with_bsplines()
                return True
            else:
                self.window.status_log.append("Failed to remove trailing edge thickening from B-splines.")
                return False
                
        except Exception as e:
            self.window.status_log.append(f"Error removing trailing edge thickening from B-splines: {e}")
            return False

    def is_te_thickened(self) -> bool:
        """
        Check if the B-spline model has trailing edge thickening applied.
        
        Returns:
            bool: True if trailing edge is thickened, False otherwise
        """
        if not self.bspline_processor.is_fitted():
            return False
        return not self.bspline_processor.is_sharp_te

    
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
            'bspline_is_blunt': not self.bspline_processor.is_sharp_te,
        }
        
        # Add B-spline max error information
        if hasattr(self.bspline_processor, 'last_upper_max_error'):
            plot_data['bspline_upper_max_error'] = self.bspline_processor.last_upper_max_error
            plot_data['bspline_upper_max_error_idx'] = self.bspline_processor.last_upper_max_error_idx
            plot_data['bspline_lower_max_error'] = self.bspline_processor.last_lower_max_error
            plot_data['bspline_lower_max_error_idx'] = self.bspline_processor.last_lower_max_error_idx
        

        
        # Emit plot update signal
        self.processor.plot_update_requested.emit(plot_data)


