"""Application controller that wires together the Qt widgets and the
underlying *core* processing logic.

The controller owns a :class:`core.airfoil_processor.AirfoilProcessor` instance
and routes GUI events to it, while forwarding log/plot signals back to the
widgets.
"""

from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import Qt, QObject
from PySide6.QtWidgets import (
    QFileDialog,
    QApplication,
    QProgressDialog,
)

from core import config
from core.airfoil_processor import AirfoilProcessor
from utils.dxf_exporter import export_curves_to_dxf

from gui.main_window import MainWindow


class MainController(QObject):
    """Mediator between :class:`core.airfoil_processor.AirfoilProcessor` and Qt GUI."""

    def __init__(self, window: MainWindow):
        super().__init__(window)

        self.window = window
        self.processor = AirfoilProcessor()

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
        opt.build_single_bezier_button.clicked.connect(self._build_single_bezier_action)

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
    def _build_single_bezier_action(self) -> None:
        """Handle *Generate Airfoil* button with modal progress dialog."""
        progress = QProgressDialog("Generating airfoil...", "", 0, 0, self.window)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        opt = self.window.optimizer_panel
        try:
            regularization_weight = float(opt.single_bezier_reg_weight_input.text())
            num_points_curve_error = int(opt.curve_error_points_input.text())
            g2_flag = opt.g2_checkbox.isChecked()

            self.processor.build_single_bezier_model(
                regularization_weight,
                error_function="icp",
                enforce_g2=g2_flag,
                num_points_curve_error=num_points_curve_error,
            )
            self._comb_params_changed()
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid input for regularization weight or curve error points. "
                "Please enter valid numbers."
            )
        except Exception as exc:  # pragma: no cover – unexpected
            self.processor.log_message.emit(f"Error building single Bezier model: {exc}")
        finally:
            progress.close()

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
            self.processor.core_processor.single_bezier_upper_poly_sharp is not None
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
        elif self.processor.core_processor.single_bezier_upper_poly_sharp is not None:
            polygons_to_export = [
                self.processor.core_processor.single_bezier_upper_poly_sharp,
                self.processor.core_processor.single_bezier_lower_poly_sharp,
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

        is_file_loaded = self.processor.core_processor.upper_data is not None
        is_model_built = (
            self.processor.core_processor.single_bezier_upper_poly_sharp is not None
        )
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

        # Export button
        fp.export_dxf_button.setEnabled(is_model_built)

        # Comb sliders
        comb.comb_scale_slider.setEnabled(is_model_built)
        comb.comb_density_slider.setEnabled(is_model_built)

    # ------------------------------------------------------------------
    def _get_default_dxf_filename(self) -> str:
        """Return a safe default filename based on the loaded profile."""
        import re

        profile_name = getattr(self.processor.core_processor, "airfoil_name", None)
        if profile_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", profile_name)
            if sanitized:
                return f"{sanitized}.dxf"
        return "airfoil.dxf" 