"""Optimization controller for the Airfoil Designer GUI.

Handles TE vector recalculation and related functionality.
"""

from __future__ import annotations

from typing import Any

from core.airfoil_processor import AirfoilProcessor


class OptimizationController:
    """Handles TE vector recalculation and related functionality."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any, ui_state_controller: Any = None):
        self.processor = processor
        self.window = window
        self.ui_state_controller = ui_state_controller
    
    def recalculate_te_vectors(self) -> None:
        """Handle recalculate TE vectors action."""
        opt = self.window.optimizer_panel
        try:
            te_vector_points = int(opt.te_vector_points_combo.currentText())
        except ValueError:
            self.processor.log_message.emit("Error: Invalid input values. Please check all numeric inputs.")
            return

        self.processor.recalculate_te_vectors_and_update_plot(te_vector_points)
        # Update the default TE vector points to the value just used
        opt.set_default_te_vector_points(te_vector_points)
        # Disable the recalculate button until dropdown changes again
        opt.disable_recalc_button()
        self.processor.log_message.emit(f"Recalculating with TE vector points set to: {te_vector_points}")
        
        # Update UI state after TE vector recalculation
        if self.ui_state_controller:
            self.ui_state_controller.update_button_states() 