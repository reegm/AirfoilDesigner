"""UI state controller for the Airfoil Designer GUI.

Handles UI state management, button states, and widget updates.
"""

from __future__ import annotations

from typing import Any

from core.airfoil_processor import AirfoilProcessor
from core import config


class UIStateController:
    """Handles UI state management and widget updates."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any):
        self.processor = processor
        self.window = window
        # Store default values for UI reset
        self._default_comb_scale = config.COMB_SCALE_DEFAULT
        self._default_comb_density = config.COMB_DENSITY_DEFAULT
        self._default_bspline_cp = 10  # Default control points
        self._default_te_vector_points = config.DEFAULT_TE_VECTOR_POINTS
        # Store initial thickness from input data
        self._initial_thickness_mm: float = 0.0
    
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
        
        # Check for B-spline model
        # Prefer the controller's processor reference to avoid mismatches
        bspline_proc = None
        bspline_controller = getattr(self.window, "bspline_controller", None)
        if bspline_controller is not None and getattr(bspline_controller, "bspline_processor", None) is not None:
            bspline_proc = bspline_controller.bspline_processor
        else:
            bspline_proc = getattr(self.window, "bspline_processor", None)
        is_bspline_model_built = False
        if bspline_proc is not None:
            try:
                is_bspline_model_built = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                is_bspline_model_built = False
        
        # Model is built if B-spline is available
        is_model_built = is_bspline_model_built
        
        # Check thickening state for B-spline model
        # The button shows "Remove Thickening" only if thickening has been applied via the UI
        # (i.e., if backup data exists, meaning thickening was applied)
        is_bspline_thickened = False
        if is_bspline_model_built and bspline_proc is not None:
            # Check if thickening was applied via UI (backup exists)
            has_backup = (bspline_proc._backup_upper_control_points is not None and 
                         bspline_proc._backup_lower_control_points is not None)
            is_bspline_thickened = has_backup
        
        # Overall thickening state
        is_thickened = is_bspline_thickened
        
        is_trailing_edge_thickened = False
        if hasattr(self.processor, "is_trailing_edge_thickened"):
            is_trailing_edge_thickened = self.processor.is_trailing_edge_thickened()

        # Calculate current thickness value from UI
        try:
            current_thickness_mm = float(airfoil.te_thickness_input.text())
        except ValueError:
            current_thickness_mm = self._initial_thickness_mm
        
        # Determine if button should be enabled
        # Policy:
        # - Require a fitted B-spline model to apply thickening
        # - Enable "Remove Thickening" if UI-applied thickening exists
        # - Otherwise enable "Thicken" only if the user changed the thickness value
        thickness_differs = abs(current_thickness_mm - self._initial_thickness_mm) > 1e-6

        if is_thickened:
            # UI-applied thickening exists (backups present) -> allow removal
            button_enabled = bool(is_model_built)
        else:
            # No UI-applied thickening -> require fitted model AND a change
            button_enabled = bool(is_model_built and thickness_differs)
        
        # Thickening button - enable/disable based on state and value changes
        airfoil.toggle_thickening_button.setEnabled(button_enabled)
        
        # Set button text based on thickening state
        if is_trailing_edge_thickened and not thickness_differs:
            # Blunt trailing edge loaded from file, no user changes yet
            airfoil.toggle_thickening_button.setText("Thicken")
        elif is_thickened:
            # Thickening has been applied via UI
            airfoil.toggle_thickening_button.setText("Remove Thickening")
        else:
            # No thickening applied
            airfoil.toggle_thickening_button.setText("Thicken")

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

        # Check for B-spline model
        bspline_proc = getattr(self.window, "bspline_processor", None)
        is_bspline_model_present = False
        if bspline_proc is not None:
            try:
                is_bspline_model_present = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                is_bspline_model_present = False
        
        if is_bspline_model_present:
            # Update B-spline comb if B-spline model is present
            bspline_proc = getattr(self.window, "bspline_processor", None)
            if bspline_proc is not None and bspline_proc.is_fitted():
                # Trigger B-spline plot update with new comb parameters
                self.window.bspline_controller._update_plot_with_bsplines()
    
    def handle_toggle_thickening(self) -> None:
        """Apply/remove trailing-edge thickening for B-spline models."""
        airfoil = self.window.airfoil_settings_panel
        try:
            te_thickness_percent = float(airfoil.te_thickness_input.text()) / (
                float(airfoil.chord_length_input.text()) / 100.0
            )
            
            # Check if we have B-spline model to thicken
            bspline_proc = getattr(self.window, "bspline_processor", None)
            is_bspline_model_built = False
            if bspline_proc is not None:
                try:
                    is_bspline_model_built = bool(getattr(bspline_proc, "fitted", False))
                except Exception:
                    is_bspline_model_built = False
            
            if not is_bspline_model_built:
                self.processor.log_message.emit("No B-spline model available to apply thickening to.")
                return
            
            # Determine if UI-applied thickening exists (presence of backups)
            has_ui_applied_thickening = False
            if bspline_proc is not None:
                has_ui_applied_thickening = (
                    getattr(bspline_proc, "_backup_upper_control_points", None) is not None and
                    getattr(bspline_proc, "_backup_lower_control_points", None) is not None
                )
            
            # Handle B-spline thickening based on UI-applied state
            bspline_controller = getattr(self.window, "bspline_controller", None)
            if bspline_controller is not None:
                if not has_ui_applied_thickening:
                    bspline_controller.apply_te_thickening(te_thickness_percent)
                else:
                    bspline_controller.remove_te_thickening()
                    # Reset thickness input to initial value when thickening is removed
                    airfoil.te_thickness_input.setText(f"{self._initial_thickness_mm:.3f}")
            
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

    def reset_ui_for_new_airfoil(self) -> None:
        """Reset stateful controls when a new airfoil is loaded, preserving user settings."""
        # Calculate and store initial thickness from input data
        self._calculate_initial_thickness()
        
        bspline_proc = getattr(self.window, "bspline_processor", None)
        if bspline_proc is not None:
            bspline_proc.fitted = False
            bspline_proc.upper_control_points = None
            bspline_proc.lower_control_points = None
            bspline_proc.upper_curve = None
            bspline_proc.lower_curve = None
            bspline_proc.is_sharp_te = True
            bspline_proc._backup_upper_control_points = None
            bspline_proc._backup_lower_control_points = None
            bspline_proc._backup_upper_knot_vector = None
            bspline_proc._backup_lower_knot_vector = None


        opt = self.window.optimizer_panel
        opt.disable_recalc_button()  # Reset recalc button state

        airfoil = self.window.airfoil_settings_panel
        airfoil.toggle_thickening_button.setEnabled(False)  # Disabled until B-spline is fitted
        airfoil.toggle_thickening_button.setText("Thicken")

        # Update all button states and labels
        self.update_comb_labels()
        self.update_button_states()

        self.processor.log_message.emit("B-spline model reset for new airfoil.")
    
    def _calculate_initial_thickness(self) -> None:
        """Calculate and store the initial thickness from input data, and update the UI."""
        airfoil = self.window.airfoil_settings_panel
        
        if self.processor.upper_data is not None and self.processor.lower_data is not None:
            # Calculate actual thickness from normalized data (chord = 1)
            te_upper_y = self.processor.upper_data[-1, 1]
            te_lower_y = self.processor.lower_data[-1, 1]
            actual_thickness_normalized = abs(te_upper_y - te_lower_y)
            # Convert to mm using default chord length
            initial_thickness_mm_raw = actual_thickness_normalized * config.DEFAULT_CHORD_LENGTH_MM
            # Round to match UI representation to avoid false diffs
            self._initial_thickness_mm = round(initial_thickness_mm_raw, 3)
            
            # Update the input field to show the actual thickness
            airfoil.te_thickness_input.setText(f"{self._initial_thickness_mm:.3f}")
        else:
            # No data loaded, set to default
            self._initial_thickness_mm = float(round(config.DEFAULT_TE_THICKNESS_MM, 3))
            airfoil.te_thickness_input.setText(f"{self._initial_thickness_mm:.3f}")
    
    def handle_thickness_input_changed(self) -> None:
        """Handle changes in the thickness input field."""
        # Update button states when thickness input changes
        self.update_button_states() 