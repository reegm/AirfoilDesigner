"""UI state controller for the Airfoil Designer GUI.

Handles UI state management, button states, and widget updates.
"""

from __future__ import annotations

from typing import Any

from core.airfoil_processor import AirfoilProcessor


class UIStateController:
    """Handles UI state management and widget updates."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any):
        self.processor = processor
        self.window = window
    
    def update_comb_labels(self) -> None:
        """Update comb scale and density labels."""
        comb = self.window.comb_panel
        scale_val = comb.comb_scale_slider.value() / 1000.0
        comb.comb_scale_label.setText(f"{scale_val:.3f}")

        density_val = comb.comb_density_slider.value()
        comb.comb_density_label.setText(f"{density_val}")
    
    def update_button_states(self) -> None:
        """Enable/disable buttons based on current processor state."""
        fp = self.window.file_panel
        opt = self.window.optimizer_panel
        airfoil = self.window.airfoil_settings_panel
        comb = self.window.comb_panel

        is_file_loaded = self.processor.upper_data is not None
        
        # Check for Bezier model
        is_bezier_model_built = (
            self.processor.upper_poly_sharp is not None
        )
        
        # Check for B-spline model
        bspline_proc = getattr(self.window, "bspline_processor", None)
        is_bspline_model_built = False
        if bspline_proc is not None:
            try:
                is_bspline_model_built = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                is_bspline_model_built = False
        
        # Model is built if either Bezier or B-spline is available
        is_model_built = is_bezier_model_built or is_bspline_model_built
        is_thickened = self.processor._is_thickened
        is_trailing_edge_thickened = False
        if hasattr(self.processor, "is_trailing_edge_thickened"):
            is_trailing_edge_thickened = self.processor.is_trailing_edge_thickened()

        # Build button
        opt.build_single_bezier_button.setEnabled(is_file_loaded)

        # Thickening button
        airfoil.toggle_thickening_button.setEnabled(
            is_model_built and not is_trailing_edge_thickened
        )
        airfoil.toggle_thickening_button.setText(
            "Remove Thickening" if is_thickened else "Apply Thickening"
        )

        # Export buttons
        fp.export_dxf_button.setEnabled(is_model_built)
        fp.export_dat_button.setEnabled(is_model_built)

        # Comb sliders
        comb.comb_scale_slider.setEnabled(is_model_built)
        comb.comb_density_slider.setEnabled(is_model_built)
    
    def handle_comb_params_changed(self) -> None:
        """Handle changes in comb scale/density sliders."""
        comb = self.window.comb_panel
        self.update_comb_labels()

        scale = comb.comb_scale_slider.value() / 1000.0
        density = comb.comb_density_slider.value()

        # Check for any model (Bezier or B-spline)
        is_bezier_model_present = (
            self.processor.upper_poly_sharp is not None
        )
        
        bspline_proc = getattr(self.window, "bspline_processor", None)
        is_bspline_model_present = False
        if bspline_proc is not None:
            try:
                is_bspline_model_present = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                is_bspline_model_present = False
        
        is_model_present = is_bezier_model_present or is_bspline_model_present

        if is_model_present:
            self.processor.request_plot_update_with_comb_params(scale, density)
    
    def handle_toggle_thickening(self) -> None:
        """Apply/remove trailing-edge thickening."""
        airfoil = self.window.airfoil_settings_panel
        try:
            te_thickness_percent = float(airfoil.te_thickness_input.text()) / (
                float(airfoil.chord_length_input.text()) / 100.0
            )
            self.processor.toggle_thickening(te_thickness_percent)
            self.handle_comb_params_changed()
            self.update_button_states()
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid TE Thickness. Please enter a number."
            )
        except Exception as exc:  # pragma: no cover
            self.processor.log_message.emit(
                f"An unexpected error occurred during thickening toggle: {exc}"
            ) 