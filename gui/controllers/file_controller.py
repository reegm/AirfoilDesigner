"""File operations controller for the Airfoil Designer GUI.

Handles loading airfoil data files and exporting DXF files.
"""

from __future__ import annotations

import os
from typing import Any

from PySide6.QtWidgets import QFileDialog

from core.airfoil_processor import AirfoilProcessor
from utils.dxf_exporter import export_curves_to_dxf
from utils.sampling_utils import sample_airfoil_surfaces
from utils.data_loader import export_airfoil_to_selig_format
import numpy as np


class FileController:
    """Handles file loading and export operations."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any, ui_state_controller: Any = None):
        self.processor = processor
        self.window = window
        self.ui_state_controller = ui_state_controller
    
    def load_airfoil_file(self) -> None:
        """Handle loading an airfoil data file."""
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
                
                # Update UI state after successful file load
                if self.ui_state_controller:
                    self.ui_state_controller.update_button_states()
            else:
                self.processor.log_message.emit(
                    f"Failed to load '{os.path.basename(file_path)}'. Check file format and content."
                )
        except Exception as exc:  # pragma: no cover – unexpected error path
            self.processor.log_message.emit(
                f"An unexpected error occurred during file loading: {exc}"
            )
    

    
    def export_single_bezier_dxf(self) -> None:
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
    
    def _get_default_dxf_filename(self) -> str:
        """Return a safe default filename based on the loaded profile."""
        import re

        profile_name = getattr(self.processor, "airfoil_name", None)
        if profile_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", profile_name)
            if sanitized:
                return f"{sanitized}.dxf"
        return "airfoil.dxf" 

    def export_dat_file(self) -> None:
        """Export the current model as a high-resolution .dat file using the shared points-per-surface setting.
        Exports Bézier if available; otherwise exports CST if a fit exists.
        """
        # Get the number of points per surface from the UI
        try:
            points_per_surface = int(self.window.file_panel.points_per_surface_input.value())
        except Exception:
            self.processor.log_message.emit(
                "Error: Invalid number of points. Please enter a valid number."
            )
            return

        # Get the current airfoil name
        airfoil_name = getattr(self.processor, "airfoil_name", "airfoil") or "airfoil"

        # Check availability
        has_bezier = (
            (self.processor._is_thickened and bool(self.processor._thickened_single_bezier_polygons)) or
            (self.processor.upper_poly_sharp is not None and self.processor.lower_poly_sharp is not None)
        )
        try:
            has_cst = bool(self.window.main_controller.cst_processor.is_fitted())
        except Exception:
            has_cst = False

        upper_sampled = None
        lower_sampled = None
        export_mode = None  # 'bezier' or 'cst'

        if has_bezier:
            # Resolve polygons
            if self.processor._is_thickened and self.processor._thickened_single_bezier_polygons:
                polygons = self.processor._thickened_single_bezier_polygons
                upper_poly = polygons[0] if len(polygons) >= 1 else None
                lower_poly = polygons[1] if len(polygons) >= 2 else None
                self.processor.log_message.emit(
                    "Preparing to export thickened single Bézier model as .dat file."
                )
            else:
                upper_poly = self.processor.upper_poly_sharp
                lower_poly = self.processor.lower_poly_sharp
                self.processor.log_message.emit(
                    "Preparing to export sharp single Bézier model as .dat file."
                )

            if upper_poly is None or lower_poly is None:
                self.processor.log_message.emit(
                    "Error: Single Bézier model not available for export. Please build it first."
                )
                return

            # Sample the surfaces using curvature-based sampling
            try:
                upper_sampled, lower_sampled = sample_airfoil_surfaces(
                    upper_poly, lower_poly, points_per_surface, curvature_weight=0.7
                )
                self.processor.log_message.emit(
                    f"Sampled {points_per_surface} points per surface using curvature-based sampling."
                )
            except Exception as exc:
                self.processor.log_message.emit(
                    f"Error during curvature-based sampling: {exc}"
                )
                return
            export_mode = 'bezier'
        elif has_cst:
            # Sample dense CST curves using curvature-based sampling method
            try:
                fitter = self.window.main_controller.cst_processor.cst_fitter
                uc = self.window.main_controller.cst_processor.upper_coefficients
                lc = self.window.main_controller.cst_processor.lower_coefficients
                
                # Use the CST fitter's built-in sampling method from config (e.g., "cosine" for curvature distribution)
                from core import config
                sampling_method = getattr(config, "CST_SAMPLING_METHOD", "cosine")
                
                xu, yu = fitter.sample(uc, n=int(points_per_surface), method=sampling_method)
                xl, yl = fitter.sample(lc, n=int(points_per_surface), method=sampling_method)
                
                upper_sampled = np.column_stack([xu, yu])
                lower_sampled = np.column_stack([xl, yl])
                
                self.processor.log_message.emit(
                    f"Sampled {points_per_surface} points per surface from CST curves using '{sampling_method}' distribution."
                )
            except Exception as exc:
                self.processor.log_message.emit(
                    f"Error during CST sampling: {exc}"
                )
                return
            export_mode = 'cst'
        else:
            self.processor.log_message.emit(
                "Error: Nothing to export. Build a Bézier model or fit CST first."
            )
            return

        # Get default filename
        default_filename = self._get_default_dat_filename(airfoil_name, is_cst=(export_mode == 'cst'))

        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save High-Resolution .dat File",
            default_filename,
            "DAT Files (*.dat);;All Files (*)",
        )

        if not file_path:
            self.processor.log_message.emit(".dat export cancelled by user.")
            return

        try:
            # Export to Selig format
            export_airfoil_to_selig_format(upper_sampled, lower_sampled, airfoil_name, file_path)
            mode_label = "CST" if export_mode == 'cst' else "Bézier"
            self.processor.log_message.emit(
                f"{mode_label} high-resolution .dat export successful to '{os.path.basename(file_path)}'."
            )
            self.processor.log_message.emit(
                f"Exported {len(upper_sampled)} points per surface in Selig format."
            )
        except IOError as exc:
            self.processor.log_message.emit(f"Could not save .dat file: {exc}")
        except Exception as exc:
            self.processor.log_message.emit(f"An unexpected error occurred during .dat export: {exc}")

    def _get_default_dat_filename(self, airfoil_name: str, is_cst: bool = False) -> str:
        """Return a safe default filename for .dat export based on the loaded profile.
        Uses a different suffix when exporting CST.
        """
        import re

        if airfoil_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", airfoil_name)
            if sanitized:
                return f"{sanitized}_{'cst_' if is_cst else ''}highres.dat"
        return f"airfoil_{'cst_' if is_cst else ''}highres.dat" 