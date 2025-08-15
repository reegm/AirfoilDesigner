"""File operations controller for the Airfoil Designer GUI.

Handles loading airfoil data files and exporting DXF files.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np

from PySide6.QtWidgets import QFileDialog

from core.airfoil_processor import AirfoilProcessor
from utils.dxf_exporter import export_curves_to_dxf, export_bspline_to_dxf
from utils.sampling_utils import sample_airfoil_surfaces
from utils.data_loader import export_airfoil_to_selig_format
from utils.cadquery_exporter import export_bspline_separate_surfaces_to_step


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
    

    def export_dxf(self) -> None:
        """Export the current model as a DXF file. Automatically chooses between B-spline and Bezier export."""
        # First check if B-spline is available and fitted
        bspline_proc = getattr(self.window, "bspline_processor", None)
        bspline_fitted = False
        if bspline_proc is not None:
            try:
                bspline_fitted = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                bspline_fitted = False

        if bspline_fitted and getattr(bspline_proc, "upper_control_points", None) is not None and \
           getattr(bspline_proc, "lower_control_points", None) is not None:
            # Use B-spline export
            self.export_bspline_dxf()
        else:
            # Fall back to Bezier export
            self.export_single_bezier_dxf()
    
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
    
    def export_bspline_dxf(self) -> None:
        """Export the current B-spline model as a DXF file."""
        # Check if B-spline processor is available and fitted
        bspline_proc = getattr(self.window, "bspline_processor", None)
        if bspline_proc is None or not getattr(bspline_proc, "fitted", False):
            self.processor.log_message.emit(
                "Error: B-spline model not available for export. Please fit B-spline first."
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

        dxf_doc = export_bspline_to_dxf(
            bspline_proc,
            chord_length_mm,
            self.processor.log_message.emit,
        )

        if not dxf_doc:
            self.processor.log_message.emit(
                "B-spline DXF export failed during document creation."
            )
            return

        default_filename = self._get_default_dxf_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save B-spline DXF File",
            default_filename,
            "DXF Files (*.dxf)",
        )
        if not file_path:
            self.processor.log_message.emit("B-spline DXF export cancelled by user.")
            return

        try:
            dxf_doc.saveas(file_path)
            self.processor.log_message.emit(
                f"B-spline DXF export successful to '{os.path.basename(file_path)}'."
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
        """Export the current model as a high-resolution .dat file.
        If a B-spline fit is available, export the sampled B-spline curves.
        Otherwise, fall back to exporting the Bezier model using curvature-based sampling.
        """
        # Get the number of points per surface from the UI
        try:
            points_per_surface = self.window.file_panel.points_per_surface_input.value()
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid number of points. Please enter a valid number."
            )
            return

        # Get the current airfoil name
        airfoil_name = getattr(self.processor, "airfoil_name", "airfoil")
        if not airfoil_name:
            airfoil_name = "airfoil"

        # ------------------------------------------------------------------
        # Prefer B-spline export if available
        # ------------------------------------------------------------------
        bspline_proc = getattr(self.window, "bspline_processor", None)
        try:
            bspline_fitted = bool(getattr(bspline_proc, "fitted", False))
        except Exception:
            bspline_fitted = False

        if bspline_proc is not None and bspline_fitted and \
           getattr(bspline_proc, "upper_curve", None) is not None and \
           getattr(bspline_proc, "lower_curve", None) is not None:
            try:
                t_values = np.linspace(0.0, 1.0, points_per_surface)
                if len(t_values) > 0:
                    t_values[-1] = min(t_values[-1], 1.0 - 1e-12)
                upper_points = bspline_proc.upper_curve(t_values)
                lower_points = bspline_proc.lower_curve(t_values)

                # Default filename
                default_filename = self._get_default_dat_filename(f"{airfoil_name}_bspline")

                file_path, _ = QFileDialog.getSaveFileName(
                    self.window,
                    "Save B-spline High-Resolution .dat File",
                    default_filename,
                    "DAT Files (*.dat);;All Files (*)",
                )
                if not file_path:
                    self.processor.log_message.emit(".dat export cancelled by user.")
                    return

                export_airfoil_to_selig_format(upper_points, lower_points, airfoil_name, file_path)
                self.processor.log_message.emit(
                    f"B-spline .dat export successful to '{os.path.basename(file_path)}'."
                )
                self.processor.log_message.emit(
                    f"Exported {len(upper_points)} points per surface in Selig format."
                )
                return
            except Exception as exc:
                self.processor.log_message.emit(
                    f"Error during B-spline .dat export, falling back to Bezier: {exc}"
                )

        # Get the current Bezier polygons
        upper_poly = None
        lower_poly = None
        
        if self.processor._is_thickened and self.processor._thickened_single_bezier_polygons:
            # Use thickened model if available
            polygons = self.processor._thickened_single_bezier_polygons
            if len(polygons) >= 2:
                upper_poly = polygons[0]
                lower_poly = polygons[1]
            self.processor.log_message.emit(
                "Preparing to export thickened single Bezier model as .dat file."
            )
        elif self.processor.upper_poly_sharp is not None and self.processor.lower_poly_sharp is not None:
            # Use sharp model
            upper_poly = self.processor.upper_poly_sharp
            lower_poly = self.processor.lower_poly_sharp
            self.processor.log_message.emit(
                "Preparing to export sharp single Bezier model as .dat file."
            )

        if upper_poly is None or lower_poly is None:
            self.processor.log_message.emit(
                "Error: Single Bezier model not available for export. Please build it first."
            )
            return

        try:
            # Sample the surfaces using curvature-based sampling
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

        # Get default filename
        default_filename = self._get_default_dat_filename(airfoil_name)
        
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
            
            self.processor.log_message.emit(
                f"High-resolution .dat export successful to '{os.path.basename(file_path)}'."
            )
            self.processor.log_message.emit(
                f"Exported {len(upper_sampled)} points per surface in Selig format."
            )
            
        except IOError as exc:
            self.processor.log_message.emit(f"Could not save .dat file: {exc}")
        except Exception as exc:
            self.processor.log_message.emit(f"An unexpected error occurred during .dat export: {exc}")

    def _get_default_dat_filename(self, airfoil_name: str) -> str:
        """Return a safe default filename for .dat export based on the loaded profile."""
        import re

        if airfoil_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", airfoil_name)
            if sanitized:
                return f"{sanitized}_highres.dat"
        return "airfoil_highres.dat"

    def export_step_surface(self) -> None:
        """Export B-spline airfoil as separate 1mm STEP surfaces using CadQuery."""
        # First check if B-spline is available and fitted
        bspline_proc = getattr(self.window, "bspline_processor", None)
        bspline_fitted = False
        if bspline_proc is not None:
            try:
                bspline_fitted = bool(getattr(bspline_proc, "fitted", False))
            except Exception:
                bspline_fitted = False

        if bspline_fitted and getattr(bspline_proc, "upper_control_points", None) is not None and \
           getattr(bspline_proc, "lower_control_points", None) is not None:
            # Use B-spline export
            self._export_bspline_step()
        else:
            # Fall back to Bezier export
            self._export_bezier_step()

    def _export_bspline_step(self) -> None:
        """Export the current B-spline model as STEP surfaces."""
        bspline_proc = getattr(self.window, "bspline_processor", None)
        if bspline_proc is None or not getattr(bspline_proc, "fitted", False):
            self.processor.log_message.emit(
                "Error: B-spline model not available for STEP export. Please fit B-spline first."
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

        # File dialog
        default_filename = self._get_default_step_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save STEP Surface File",
            default_filename,
            "STEP Files (*.step *.stp);;All Files (*)",
        )

        if not file_path:
            self.processor.log_message.emit("STEP export cancelled.")
            return

        # Perform export with hardcoded 1mm surfaces, passing knots/degree to preserve control-pole BSplines
        success = export_bspline_separate_surfaces_to_step(
            upper_control_points=bspline_proc.upper_control_points,
            lower_control_points=bspline_proc.lower_control_points,
            output_path=file_path,
            chord_length_mm=chord_length_mm,
            logger_func=self.processor.log_message.emit,
            upper_knot_vector=getattr(bspline_proc, 'upper_knot_vector', None),
            lower_knot_vector=getattr(bspline_proc, 'lower_knot_vector', None),
            degree=getattr(bspline_proc, 'degree', 5),
        )

        if success:
            self.processor.log_message.emit(f"STEP surface export completed: '{os.path.basename(file_path)}'")
        else:
            self.processor.log_message.emit("STEP surface export failed.")

    def _export_bezier_step(self) -> None:
        """Export the current Bezier model as STEP surfaces."""
        polygons_to_export = None
        if self.processor._is_thickened and self.processor._thickened_single_bezier_polygons:
            polygons_to_export = self.processor._thickened_single_bezier_polygons
            self.processor.log_message.emit("Preparing to export thickened Bézier model as STEP surfaces.")
        elif self.processor.upper_poly_sharp is not None and self.processor.lower_poly_sharp is not None:
            polygons_to_export = [self.processor.upper_poly_sharp, self.processor.lower_poly_sharp]
            self.processor.log_message.emit("Preparing to export sharp Bézier model as STEP surfaces.")
        else:
            self.processor.log_message.emit("Error: No Bézier model available for STEP export.")
            return

        if len(polygons_to_export) < 2:
            self.processor.log_message.emit("Error: Need both upper and lower curves for STEP export.")
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

        # File dialog
        default_filename = self._get_default_step_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save STEP Surface File",
            default_filename,
            "STEP Files (*.step *.stp);;All Files (*)",
        )

        if not file_path:
            self.processor.log_message.emit("STEP export cancelled.")
            return

        # Perform export with hardcoded 1mm surfaces
        success = export_bspline_separate_surfaces_to_step(
            upper_control_points=polygons_to_export[0],
            lower_control_points=polygons_to_export[1],
            output_path=file_path,
            chord_length_mm=chord_length_mm,
            logger_func=self.processor.log_message.emit
        )

        if success:
            self.processor.log_message.emit(f"STEP surface export completed: '{os.path.basename(file_path)}'")
        else:
            self.processor.log_message.emit("STEP surface export failed.")

    def _get_default_step_filename(self) -> str:
        """Generate default filename for STEP export."""
        import re
        profile_name = getattr(self.processor, "airfoil_name", None)
        if profile_name:
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", profile_name)
            if sanitized:
                return f"{sanitized}_surface.step"
        return "airfoil_surface.step" 