"""Application controller that wires together the Qt widgets and the
underlying *core* processing logic.

The controller owns a :class:`core.airfoil_processor.AirfoilProcessor` instance
and routes GUI events to it, while forwarding log/plot signals back to the
widgets.
"""

from __future__ import annotations

import os
from typing import Any

import multiprocessing
import time
from PySide6.QtCore import Qt, QObject, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QApplication,
    QProgressDialog,
)

from core import config
from core.airfoil_processor import AirfoilProcessor
from utils.dxf_exporter import export_curves_to_dxf

from gui.main_window import MainWindow

import numpy as np

# NOTE: All Bezier builder and optimizer functions have been moved to core/solver/ modules.
# If you need build_single_venkatamaran_bezier, build_single_venkatamaran_bezier_minmax, etc.,
# import from core/solver/bezier_optimizer.py or the relevant new module.

class MainController(QObject):
    """Mediator between :class:`core.airfoil_processor.AirfoilProcessor` and Qt GUI."""

    def __init__(self, window: MainWindow):
        super().__init__(window)

        self.window = window
        self.processor = AirfoilProcessor()
        self._generation_process = None
        self._generation_queue = None
        self._abort_flag = None
        self._generation_timer = QTimer()
        self._generation_timer.setInterval(200)  # ms
        self._generation_timer.timeout.connect(self._check_generation_result)
        self._is_generating = False
        self._generation_start_time = None

        # ------------------------------------------------------------------
        # Wire up processor signals
        # ------------------------------------------------------------------
        self.processor.log_message.connect(self.window.status_log.append)
        self.processor.plot_update_requested.connect(self._update_plot_from_processor)

        # ------------------------------------------------------------------
        # Connect widget signals → controller slots
        # ------------------------------------------------------------------
        fp = self.window.file_panel
        fp.load_button.clicked.connect(self._load_airfoil_file_action)
        fp.export_dxf_button.clicked.connect(self._export_single_bezier_dxf_action)

        opt = self.window.optimizer_panel
        # opt.build_single_bezier_button.clicked.disconnect()
        opt.build_single_bezier_button.clicked.connect(self._generate_or_abort_action)
        opt.recalculate_button.clicked.connect(self._recalculate_te_vectors_action)

        airfoil = self.window.airfoil_settings_panel
        airfoil.toggle_thickening_button.clicked.connect(self._toggle_thickening_action)

        comb = self.window.comb_panel
        comb.comb_scale_slider.valueChanged.connect(self._comb_params_changed)
        comb.comb_density_slider.valueChanged.connect(self._comb_params_changed)

        # ------------------------------------------------------------------
        # Initial UI state
        # ------------------------------------------------------------------
        self._update_comb_labels()
        self._update_button_states()

        self.processor.log_message.emit("Application started. Load an airfoil .dat file to begin.")

    # ------------------------------------------------------------------
    # Slots – processor → GUI
    # ------------------------------------------------------------------
    def _update_plot_from_processor(self, plot_data: dict[str, Any]) -> None:  # noqa: D401
        """Receive plot data from the *processor* and forward to the widget."""
        # Cache so we can recompute comb later
        self._last_plot_data = plot_data.copy()

        try:
            chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
        except Exception:
            chord_length_mm = None

        # Always clear the plot before drawing new data
        self.window.plot_widget.clear()
        self.window.plot_widget.plot_airfoil(
            **plot_data,
            chord_length_mm=chord_length_mm,
        )

        self._update_button_states()

    # ------------------------------------------------------------------
    # Slots – GUI → processor
    # ------------------------------------------------------------------
    def _load_airfoil_file_action(self) -> None:
        """Handle *Load Airfoil File* button."""
        # Clear the plot when loading a new file
        self.window.plot_widget.clear()
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load Airfoil Data",
            "",
            "Airfoil Data Files (*.dat);;All Files (*)",
        )

        if not file_path:
            return

        self.window.file_panel.file_path_label.setText(os.path.basename(file_path))

        try:
            if self.processor.load_airfoil_data_and_initialize_model(file_path):
                self.processor.log_message.emit(
                    f"Successfully loaded '{os.path.basename(file_path)}'."
                )
                # Reset TE vector points dropdown and disable recalc button
                opt = self.window.optimizer_panel
                opt.te_vector_points_combo.setCurrentText(str(opt.default_te_vector_points))
                opt.disable_recalc_button()
            else:
                self.processor.log_message.emit(
                    f"Failed to load '{os.path.basename(file_path)}'. Check file format and content."
                )
        except Exception as exc:  # pragma: no cover – unexpected error path
            self.processor.log_message.emit(
                f"An unexpected error occurred during file loading: {exc}"
            )
        finally:
            self._update_button_states()

    # ------------------------------------------------------------------
    def _generate_or_abort_action(self):
        opt = self.window.optimizer_panel
        if self._is_generating:
            # Abort requested
            if self._abort_flag is not None:
                self._abort_flag.value = True
                self.processor.log_message.emit("Abort requested. Waiting for optimizer to finish up...")
            # Do NOT set self._is_generating = False here
            # Do NOT stop spinner, reset button, or update button states here
            # Let _check_generation_result handle cleanup and GUI updates
            return
        # Start generation in a new process
        try:
            regularization_weight = float(opt.single_bezier_reg_weight_input.text())
            te_vector_points = int(opt.te_vector_points_combo.currentText())
            g2_flag = opt.g2_checkbox.isChecked()
            gui_strategy = opt.strategy_combo.currentText().lower()  # 'fixed-x' or 'free-x'
            error_function = opt.error_function_combo.currentText().lower()  # 'euclidean' or 'orthogonal'
            objective_gui = opt.objective_combo.currentText()
            if objective_gui == 'MSR':
                objective_type = 'msr'
            elif objective_gui == 'Max Error':
                objective_type = 'minmax'
            else:
                self.processor.log_message.emit(f"Error: Unsupported objective '{objective_gui}'.")
                return
            if gui_strategy not in ['fixed-x', 'free-x']:
                self.processor.log_message.emit(f"Error: Unsupported strategy '{gui_strategy}'.")
                return
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid input for regularization weight, curve error points, or TE vector points. Please enter valid numbers."
            )
            return
        self._generation_queue = multiprocessing.Queue()
        self._abort_flag = multiprocessing.Value('b', False)  # Shared boolean flag
        args = (
            self.processor.upper_data,
            self.processor.lower_data,
            regularization_weight,
            g2_flag,
            te_vector_points,
            gui_strategy,
            error_function,
            objective_type,
            self._abort_flag
        )
        self._generation_process = multiprocessing.Process(
            target=_generation_worker,
            args=(args, self._generation_queue)
        )
        self._generation_process.start()
        self._is_generating = True
        self._generation_start_time = time.time()
        opt.build_single_bezier_button.setText("Abort")
        self._generation_timer.start()
        self._update_button_states()
        self.processor.log_message.emit("Started airfoil generation in background process...")
        if not config.DEBUG_WORKER_LOGGING:
            self.window.status_log.start_spinner("Processing...")

    def _check_generation_result(self):
        if not self._is_generating:
            self._generation_timer.stop()
            return
        if self._generation_queue is not None and not self._generation_queue.empty():
            result = self._generation_queue.get()
            
            # Handle log messages from the worker process (only if debug logging is enabled)
            if isinstance(result, dict) and result.get("type") == "log":
                if config.DEBUG_WORKER_LOGGING:
                    self.processor.log_message.emit(result["message"])
                return  # Continue checking for more messages
            
            # Handle final result
            self._generation_timer.stop()
            self._is_generating = False
            opt = self.window.optimizer_panel
            opt.build_single_bezier_button.setText("Generate Airfoil")
            
            # Stop spinner only if it was started (debug mode disabled)
            if not config.DEBUG_WORKER_LOGGING:
                self.window.status_log.stop_spinner()
            
            self._generation_process = None
            self._generation_queue = None
            self._abort_flag = None
            if isinstance(result, dict) and result.get("success") and result.get("upper_poly") is not None:
                # Update processor state with new model
                self.processor.upper_poly_sharp = result["upper_poly"]
                self.processor.lower_poly_sharp = result["lower_poly"]
                self.processor.last_single_bezier_upper_max_error = result.get("upper_max_error")
                self.processor.last_single_bezier_upper_max_error_idx = result.get("upper_max_error_idx")
                self.processor.last_single_bezier_lower_max_error = result.get("lower_max_error")
                self.processor.last_single_bezier_lower_max_error_idx = result.get("lower_max_error_idx")
                elapsed_time = time.time() - self._generation_start_time if self._generation_start_time else 0
                
                # Log success with error information
                base_message = f"Single Bezier model built successfully. (Elapsed time: {elapsed_time:.2f}s)"
                
                # Add error information if available
                upper_error = result.get("upper_max_error")
                lower_error = result.get("lower_max_error")
                if upper_error is not None and lower_error is not None:
                    # Convert to chord percentage and mm if chord length is available
                    try:
                        chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
                        upper_error_mm = upper_error * chord_length_mm
                        lower_error_mm = lower_error * chord_length_mm
                        error_message = f"\n  Upper surface max error: {upper_error:.3e} ({upper_error_mm:.3f}mm @ {chord_length_mm:.0f}mm chord)"
                        error_message += f"\n  Lower surface max error: {lower_error:.3e} ({lower_error_mm:.3f}mm @ {chord_length_mm:.0f}mm chord)"
                    except:
                        # Fallback to just normalized units
                        error_message = f"\n  Upper surface max error: {upper_error:.3e}"
                        error_message += f"\n  Lower surface max error: {lower_error:.3e}"
                    
                    base_message += error_message
                
                self.processor.log_message.emit(base_message)
                self._generation_start_time = None
                self.processor._request_plot_update()
                self._comb_params_changed()
            else:
                self.processor.log_message.emit(result.get("error", "Failed to build single Bezier model."))
            self._update_button_states()
        elif self._generation_process is not None and not self._generation_process.is_alive():
            # Process died without result
            self._generation_timer.stop()
            self._is_generating = False
            opt = self.window.optimizer_panel
            opt.build_single_bezier_button.setText("Generate Airfoil")
            
            # Stop spinner only if it was started (debug mode disabled)
            if not config.DEBUG_WORKER_LOGGING:
                self.window.status_log.stop_spinner()
                
            elapsed_time = time.time() - self._generation_start_time if self._generation_start_time else 0
            self._generation_process = None
            self._generation_queue = None
            self._abort_flag = None
            self.processor.log_message.emit(f"Generation process exited unexpectedly. (Elapsed time: {elapsed_time:.2f}s)")
            self._generation_start_time = None
            self._update_button_states()

    # ------------------------------------------------------------------
    def _recalculate_te_vectors_action(self):
        """Handle *Recalculate* button - only recalculates TE vectors and updates plot."""
        opt = self.window.optimizer_panel
        try:
            te_vector_points = int(opt.te_vector_points_combo.currentText())
            # regularization_weight = float(opt.single_bezier_reg_weight_input.text())
            # error_function = opt.error_function_combo.currentText()
            # g2_flag = opt.g2_checkbox.isChecked()
        except ValueError:
            self.processor.log_message.emit("Error: Invalid input values. Please check all numeric inputs.")
            return

        self.processor.recalculate_te_vectors_and_update_plot(te_vector_points)
        # Update the default TE vector points to the value just used
        opt.set_default_te_vector_points(te_vector_points)
        # Disable the recalculate button until dropdown changes again
        opt.disable_recalc_button()
        self.processor.log_message.emit(f"Recalculating with TE vector points set to: {te_vector_points}")
        
    # ------------------------------------------------------------------
    def _toggle_thickening_action(self) -> None:
        """Apply/remove trailing-edge thickening."""
        airfoil = self.window.airfoil_settings_panel
        try:
            te_thickness_percent = float(airfoil.te_thickness_input.text()) / (
                float(airfoil.chord_length_input.text()) / 100.0
            )
            self.processor.toggle_thickening(te_thickness_percent)
            self._comb_params_changed()
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid TE Thickness. Please enter a number."
            )
        except Exception as exc:  # pragma: no cover
            self.processor.log_message.emit(
                f"An unexpected error occurred during thickening toggle: {exc}"
            )

    # ------------------------------------------------------------------
    def _comb_params_changed(self) -> None:
        """Handle changes in comb scale/density sliders."""
        comb = self.window.comb_panel
        self._update_comb_labels()

        scale = comb.comb_scale_slider.value() / 1000.0
        density = comb.comb_density_slider.value()

        is_model_present = (
            self.processor.upper_poly_sharp is not None
        )

        if is_model_present:
            self.processor.request_plot_update_with_comb_params(scale, density)

    # ------------------------------------------------------------------
    def _export_single_bezier_dxf_action(self) -> None:
        """Export the current Bezier model(s) as a DXF file."""
        polygons_to_export = None
        if self.processor._is_thickened and self.processor._thickened_single_bezier_polygons:
            polygons_to_export = self.processor._thickened_single_bezier_polygons
            self.processor.log_message.emit(
                "Preparing to export thickened single Bezier model."
            )
        elif self.processor.upper_poly_sharp is not None:
            polygons_to_export = [
                self.processor.upper_poly_sharp,
                self.processor.lower_poly_sharp,
            ]
            self.processor.log_message.emit(
                "Preparing to export sharp single Bezier model."
            )

        if polygons_to_export is None:
            self.processor.log_message.emit(
                "Error: Single Bezier model not available for export. Please build it first."
            )
            return

        try:
            chord_length_mm = float(
                self.window.airfoil_settings_panel.chord_length_input.text()
            )
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid chord length. Please enter a number."
            )
            return

        dxf_doc = export_curves_to_dxf(
            polygons_to_export,
            chord_length_mm,
            self.processor.log_message.emit,
        )

        if not dxf_doc:
            self.processor.log_message.emit(
                "Single Bezier DXF export failed during document creation."
            )
            return

        default_filename = self._get_default_dxf_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save Single Bezier DXF File",
            default_filename,
            "DXF Files (*.dxf)",
        )
        if not file_path:
            self.processor.log_message.emit("Single Bezier DXF export cancelled by user.")
            return

        try:
            dxf_doc.saveas(file_path)
            self.processor.log_message.emit(
                f"Single Bezier DXF export successful to '{os.path.basename(file_path)}'."
            )
            self.processor.log_message.emit(
                "Note: For correct scale in CAD software, ensure import settings are configured for millimeters."
            )
        except IOError as exc:
            self.processor.log_message.emit(f"Could not save DXF file: {exc}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _update_comb_labels(self) -> None:
        comb = self.window.comb_panel
        scale_val = comb.comb_scale_slider.value() / 1000.0
        comb.comb_scale_label.setText(f"{scale_val:.3f}")

        density_val = comb.comb_density_slider.value()
        comb.comb_density_label.setText(f"{density_val}")

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on current processor state."""
        fp = self.window.file_panel
        opt = self.window.optimizer_panel
        airfoil = self.window.airfoil_settings_panel
        comb = self.window.comb_panel

        is_file_loaded = self.processor.upper_data is not None
        is_model_built = (
            self.processor.upper_poly_sharp is not None
        )
        is_thickened = self.processor._is_thickened
        is_trailing_edge_thickened = False
        if hasattr(self.processor, "is_trailing_edge_thickened"):
            is_trailing_edge_thickened = self.processor.is_trailing_edge_thickened()

        # Build button
        opt.build_single_bezier_button.setEnabled(is_file_loaded)

        # Recalculate button
        # Remove: opt.recalculate_button.setEnabled(is_file_loaded)

        # Thickening button
        airfoil.toggle_thickening_button.setEnabled(
            is_model_built and not is_trailing_edge_thickened
        )
        airfoil.toggle_thickening_button.setText(
            "Remove Thickening" if is_thickened else "Apply Thickening"
        )

        # Export button
        fp.export_dxf_button.setEnabled(is_model_built)

        # Comb sliders
        comb.comb_scale_slider.setEnabled(is_model_built)
        comb.comb_density_slider.setEnabled(is_model_built)

    # ------------------------------------------------------------------
    def _get_default_dxf_filename(self) -> str:
        """Return a safe default filename based on the loaded profile."""
        import re

        profile_name = getattr(self.processor, "airfoil_name", None)
        if profile_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", profile_name)
            if sanitized:
                return f"{sanitized}.dxf"
        return "airfoil.dxf"

def calculate_te_tangent(upper_data, lower_data, te_vector_points):
    """
    Calculate trailing edge tangent vectors for upper and lower surfaces using the last N points.
    Returns (upper_te_tangent_vector, lower_te_tangent_vector)
    """
    def tangent(data, n):
        # Use the last n points to estimate the tangent at the trailing edge
        if n < 2 or len(data) < n:
            n = min(3, len(data))
        pts = data[-n:]
        # Fit a line and return the direction vector (normalized)
        dx = pts[-1, 0] - pts[0, 0]
        dy = pts[-1, 1] - pts[0, 1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            return np.array([1.0, 0.0])
        return np.array([dx, dy]) / norm
    upper_te_tangent = tangent(upper_data, te_vector_points)
    lower_te_tangent = tangent(lower_data, te_vector_points)
    return upper_te_tangent, lower_te_tangent

def _generation_worker(args, queue):
    """
    This worker function runs in a separate process to avoid blocking the GUI.
    It operates on the core logic without Qt dependencies.
    """
    import traceback
    import numpy as np
    from core import config
    # from utils.data_loader import load_airfoil_data, find_shoulder_x_coords
    # from utils.dxf_exporter import export_curves_to_dxf
    # from utils.bezier_utils import bezier_curvature, general_bezier_curve
    # from utils.control_point_utils import variable_x_control_points, get_paper_fixed_x_coords
    # Import new optimizer
    from core.solver.bezier_optimizer import (
            build_bezier_single_fixed_x_msr,
    build_bezier_single_fixed_x_minmax_xy,
    build_bezier_single_free_x_msr_xy,
    build_bezier_single_free_x_minmax
    )
    from core.solver.coupled_bezier_optimizer import (
            build_bezier_coupled_fixed_x_msr,
    build_bezier_coupled_fixed_x_minmax_xy,
    build_bezier_coupled_free_x_msr_xy,
    build_bezier_coupled_free_x_minmax_xy
    )

    (
        upper_data,
        lower_data,
        regularization_weight,
        g2_flag,
        te_vector_points,
        gui_strategy,
        error_function,
        objective_type,
        abort_flag
    ) = args

    debug_logging_enabled = config.DEBUG_WORKER_LOGGING
    def worker_logger(message):
        if debug_logging_enabled:
            queue.put({"type": "log", "message": message})

    upper_te_tangent_vector, lower_te_tangent_vector = calculate_te_tangent(upper_data, lower_data, te_vector_points)

    try:
        # Dispatch based on strategy and g2_flag (coupled/uncoupled)
        if gui_strategy == 'fixed-x' and not g2_flag:
            # Uncoupled fixed-x (implemented)
            le_tangent_upper = np.array([0.0, 1.0])
            le_tangent_lower = np.array([0.0, -1.0])
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            if objective_type == 'msr':
                upper_poly = build_bezier_single_fixed_x_msr(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
                lower_poly = build_bezier_single_fixed_x_msr(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'minmax':
                upper_poly = build_bezier_single_fixed_x_minmax_xy(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
                lower_poly = build_bezier_single_fixed_x_minmax_xy(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Unknown objective_type: {objective_type}"})
                return
            # Error calculation (for reporting)
            from core.solver.error_functions import calculate_single_bezier_fitting_error
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'fixed-x' and g2_flag:
            # Coupled fixed-x (implemented)
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            if objective_type == 'msr':
                upper_poly, lower_poly = build_bezier_coupled_fixed_x_msr(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'minmax':
                upper_poly, lower_poly = build_bezier_coupled_fixed_x_minmax_xy(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Coupled fixed-x objective not yet implemented: {objective_type}"})
                return
            # Error calculation (for reporting)
            from core.solver.error_functions import calculate_single_bezier_fitting_error
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'free-x' and not g2_flag:
            # Uncoupled free-x (implemented)
            le_tangent_upper = np.array([0.0, 1.0])
            le_tangent_lower = np.array([0.0, -1.0])
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            if objective_type == 'msr':
                upper_poly = build_bezier_single_free_x_msr_xy(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
                lower_poly = build_bezier_single_free_x_msr_xy(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'minmax':
                upper_poly = build_bezier_single_free_x_minmax(
                    upper_data,
                    num_control_points,
                    True,
                    le_tangent_upper,
                    upper_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
                lower_poly = build_bezier_single_free_x_minmax(
                    lower_data,
                    num_control_points,
                    False,
                    le_tangent_lower,
                    lower_te_tangent_vector,
                    regularization_weight=regularization_weight,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Free-x objective not yet implemented: {objective_type}"})
                return
            # Error calculation (for reporting)
            from core.solver.error_functions import calculate_single_bezier_fitting_error
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        elif gui_strategy == 'free-x' and g2_flag:
            # Coupled free-x (implemented)
            num_control_points = config.NUM_CONTROL_POINTS_SINGLE_BEZIER
            if objective_type == 'msr':
                upper_poly, lower_poly = build_bezier_coupled_free_x_msr_xy(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            elif objective_type == 'minmax':
                upper_poly, lower_poly = build_bezier_coupled_free_x_minmax_xy(
                    upper_data,
                    lower_data,
                    regularization_weight,
                    upper_te_tangent_vector,
                    lower_te_tangent_vector,
                    error_function=error_function,
                    logger_func=worker_logger,
                    abort_flag=abort_flag
                )
            else:
                queue.put({"success": False, "error": f"Coupled free-x objective not yet implemented: {objective_type}"})
                return
            # Error calculation (for reporting)
            from core.solver.error_functions import calculate_single_bezier_fitting_error
            if error_function == 'orthogonal':
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='orthogonal', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='orthogonal', return_max_error=True)
            else:
                upper_error_result = calculate_single_bezier_fitting_error(upper_poly, upper_data, error_function='euclidean', return_max_error=True)
                lower_error_result = calculate_single_bezier_fitting_error(lower_poly, lower_data, error_function='euclidean', return_max_error=True)
            _, upper_max_error, upper_max_error_idx = upper_error_result
            _, lower_max_error, lower_max_error_idx = lower_error_result
            queue.put({
                "success": True,
                "upper_poly": upper_poly,
                "lower_poly": lower_poly,
                "upper_max_error": upper_max_error,
                "upper_max_error_idx": upper_max_error_idx,
                "lower_max_error": lower_max_error,
                "lower_max_error_idx": lower_max_error_idx,
            })
        else:
            queue.put({"success": False, "error": f"Unknown strategy/g2_flag combination: {gui_strategy}, g2={g2_flag}"})
    except Exception as e:
        queue.put({"success": False, "error": f"Exception in worker: {e}\n{traceback.format_exc()}"})
