"""Main controller for the Airfoil Designer GUI.

Orchestrates the other controllers and handles signal routing between GUI and processor.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject

from core.airfoil_processor import AirfoilProcessor
from core.cst_processor import CSTProcessor
from gui.main_window import MainWindow

from .file_controller import FileController
from .optimization_controller import OptimizationController
from .ui_state_controller import UIStateController
from .cst_controller import CSTController


class MainController(QObject):
    """Main controller that orchestrates all other controllers and handles signal routing."""

    def __init__(self, window: MainWindow):
        super().__init__(window)

        self.window = window
        # Provide back-reference for UI components that need to reach controllers
        # (e.g., to access CST processor from UI state controller)
        setattr(self.window, "main_controller", self)
        self.processor = AirfoilProcessor()
        self.cst_processor = CSTProcessor()
        
        # Initialize sub-controllers
        self.ui_state_controller = UIStateController(self.processor, self.window)
        self.file_controller = FileController(self.processor, self.window, self.ui_state_controller)
        self.cst_controller = CSTController(self.cst_processor, self.window, self)
        self.optimization_controller = OptimizationController(self.processor, self.window, self.ui_state_controller)

        # ------------------------------------------------------------------
        # Wire up processor signals
        # ------------------------------------------------------------------
        self.processor.log_message.connect(self.window.status_log.append)
        self.processor.plot_update_requested.connect(self._update_plot_from_processor)

        # ------------------------------------------------------------------
        # Connect widget signals → controller slots
        # ------------------------------------------------------------------
        self._connect_signals()

        # ------------------------------------------------------------------
        # Initial UI state
        # ------------------------------------------------------------------
        self.ui_state_controller.update_comb_labels()
        self.ui_state_controller.update_button_states()

        self.processor.log_message.emit("Application started. Load an airfoil .dat file to begin.")

    def _connect_signals(self) -> None:
        """Connect all GUI signals to their respective controller methods."""
        # File operations
        fp = self.window.file_panel
        fp.load_button.clicked.connect(self.file_controller.load_airfoil_file)
        fp.export_dxf_button.clicked.connect(self.file_controller.export_single_bezier_dxf)
        fp.export_dat_button.clicked.connect(self.file_controller.export_dat_file)

        # Optimization operations
        opt = self.window.optimizer_panel
        opt.build_single_bezier_button.clicked.connect(self.optimization_controller.generate_or_abort)
        opt.recalculate_button.clicked.connect(self.optimization_controller.recalculate_te_vectors)

        # Airfoil settings
        airfoil = self.window.airfoil_settings_panel
        airfoil.toggle_thickening_button.clicked.connect(self.ui_state_controller.handle_toggle_thickening)

        # Comb parameters
        comb = self.window.comb_panel
        comb.comb_scale_slider.valueChanged.connect(self.ui_state_controller.handle_comb_params_changed)
        comb.comb_density_slider.valueChanged.connect(self.ui_state_controller.handle_comb_params_changed)

    def _update_plot_from_processor(self, plot_data: dict[str, Any]) -> None:
        """Receive plot data from the processor and forward to the widget."""
        # Check if this is a progress update by checking if we have current progress info
        is_progress_update = hasattr(self.optimization_controller, '_current_progress_info')
        
        # Cache so we can recompute comb later (but not for progress updates)
        if not is_progress_update:
            self._last_plot_data = plot_data.copy()

        try:
            chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
        except Exception:
            chord_length_mm = None

        # Only clear the plot for non-progress updates to avoid flickering
        if not is_progress_update:
            self.window.plot_widget.clear()
        
        # Remove chord_length_mm from plot_data to avoid duplicate parameter
        plot_data_without_chord = {k: v for k, v in plot_data.items() if k != 'chord_length_mm'}
        
        self.window.plot_widget.plot_airfoil(
            **plot_data_without_chord,
            chord_length_mm=chord_length_mm,
        )

        # Only update button states for non-progress updates
        if not is_progress_update:
            self.ui_state_controller.update_button_states()
        
        # Clear progress info after using it
        if is_progress_update:
            delattr(self.optimization_controller, '_current_progress_info') 