"""Controller for CST fitting functionality."""

from __future__ import annotations

from PySide6.QtCore import QObject
import numpy as np
from typing import Any

from core.cst_processor import CSTProcessor
from gui.main_window import MainWindow
from core.error_functions import calculate_single_bezier_fitting_error


class CSTController(QObject):
    """Controller for CST fitting functionality."""

    def __init__(self, cst_processor: CSTProcessor, window: MainWindow, main_controller=None):
        super().__init__(window)
        
        self.cst_processor = cst_processor
        self.window = window
        self.main_controller = main_controller
        
        # Connect CST processor signals
        self.cst_processor.log_message.connect(self.window.status_log.append)
        self.cst_processor.plot_update_requested.connect(self._update_plot_from_cst)
        
        # Connect GUI signals
        self._connect_signals()
        
    def _connect_signals(self) -> None:
        """Connect GUI signals to controller methods."""
        cst_panel = self.window.cst_panel
        
        # CST fitting buttons
        cst_panel.fit_cst_button.clicked.connect(self.fit_cst)
    
    def fit_cst(self) -> None:
        """Perform CST fitting on the loaded airfoil data."""
        if self.main_controller.processor.upper_data is None or self.main_controller.processor.lower_data is None:
            self.window.status_log.append("No airfoil data loaded. Please load an airfoil file first.")
            return
        
        # Get CST parameters from GUI
        params = self.window.cst_panel.get_parameters()
        
        # Set parameters in CST processor
        self.cst_processor.set_parameters(
            degree=params['degree']
        )
        
        # Perform CST fitting
        success = self.cst_processor.fit_airfoil(
            upper_data=self.main_controller.processor.upper_data,
            lower_data=self.main_controller.processor.lower_data,
            blunt_TE=self.main_controller.processor.blunt_TE()
        )
        
        if success:
            # Update GUI state
            self.window.cst_panel.set_cst_fitted(True)
            
            # Update plot
            self._update_display()
            # Ensure UI elements reflect that CST results (and comb) are available
            try:
                self.main_controller.ui_state_controller.update_button_states()
            except Exception:
                pass
            
            # Log metrics
            try:
                metrics = self.cst_processor.get_fitting_metrics()
                self.window.status_log.append(
                    f"CST Fit Complete - Upper RMSE: {metrics['upper']['rmse']:.3e}, "
                    f"Lower RMSE: {metrics['lower']['rmse']:.3e}"
                )
                self.window.status_log.append(
                    f"CST Orthogonal Errors - Upper Max: {metrics['upper']['orthogonal_max_error']:.3e}, "
                    f"Lower Max: {metrics['lower']['orthogonal_max_error']:.3e}"
                )
            except Exception as e:
                self.window.status_log.append(f"Error getting CST metrics: {e}")
        else:
            self.window.cst_panel.set_cst_fitted(False)

    def thicken_cst(self) -> None:
        """If a sharp TE CST exists, refit with a specified TE thickness (mm)."""
        if not self.cst_processor.is_fitted():
            self.window.status_log.append("No CST fit available. Please fit CST first.")
            return
        # If a Bezier model is present, defer to legacy thickening behavior
        if self.main_controller.processor.upper_poly_sharp is not None and self.main_controller.processor.lower_poly_sharp is not None:
            try:
                # Use existing UI thickening entry and legacy thickening
                te_mm = float(self.window.airfoil_settings_panel.te_thickness_input.text())
                chord_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
                te_percent = te_mm / chord_mm
                self.main_controller.processor.toggle_thickening(te_percent * 100.0)
                self.window.status_log.append("Applied thickening to Bezier model (legacy path).")
                return
            except Exception as e:
                self.window.status_log.append(f"Error applying legacy thickening: {e}")
                return
        # Otherwise, refit CST with TE override
        try:
            # Read TE thickness in mm from CST panel; fallback to general airfoil panel
            try:
                te_mm = float(self.window.cst_panel.cst_te_thickness_input.text())
            except Exception:
                te_mm = float(self.window.airfoil_settings_panel.te_thickness_input.text())
            chord_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
            te_chord = te_mm / chord_mm
        except Exception:
            self.window.status_log.append("Invalid TE thickness/chord length. Check inputs.")
            return

        # Refit using the same degree, but override trailing-edge values
        params = self.window.cst_panel.get_parameters()
        self.cst_processor.set_parameters(degree=params['degree'])

        # Re-run fitter with override TE per surface
        try:
            # Compute per-surface TE targets from original data sign at TE
            udata = self.main_controller.processor.upper_data
            ldata = self.main_controller.processor.lower_data
            ux, uy = udata[:, 0], udata[:, 1]
            lx, ly = ldata[:, 0], ldata[:, 1]
            u_te_idx = int(np.argmax(ux)); l_te_idx = int(np.argmax(lx))
            sign_u = 1.0 if uy[u_te_idx] >= 0.0 else -1.0
            sign_l = -1.0 if ly[l_te_idx] <= 0.0 else 1.0
            te_u = sign_u * (0.5 * te_chord)
            te_l = sign_l * (0.5 * te_chord)

            # Call underlying fit with overrides
            from core.cst_fitter import fit_airfoil_cst
            result = fit_airfoil_cst(
                upper_data=udata,
                lower_data=ldata,
                degree=params['degree'],
                override_te_upper=te_u,
                override_te_lower=te_l,
                logger_func=self.cst_processor.log_message.emit,
            )
            self.cst_processor.cst_fitter = result['fitter']
            self.cst_processor.upper_coefficients = result['upper_coefficients']
            self.cst_processor.lower_coefficients = result['lower_coefficients']
            self.cst_processor.upper_metrics = result['upper_metrics']
            self.cst_processor.lower_metrics = result['lower_metrics']
            # Resample
            num_pts = int(getattr(self.cst_processor.__class__, "CST_SAMPLES_FOR_PLOT", 4000) if hasattr(self.cst_processor.__class__, "CST_SAMPLES_FOR_PLOT") else 4000)
            from core.cst_fitter import generate_cst_airfoil_data
            self.cst_processor.cst_upper_data, self.cst_processor.cst_lower_data = generate_cst_airfoil_data(
                upper_coefficients=self.cst_processor.upper_coefficients,
                lower_coefficients=self.cst_processor.lower_coefficients,
                fitter=self.cst_processor.cst_fitter,
                num_points=num_pts,
                sampling_method=None,
            )
            self.window.status_log.append(f"CST thickening applied: TE={te_mm:.3f} mm (±{te_mm/2:.3f} per surface)")
            self.cst_processor.request_plot_update()
        except Exception as e:
            self.window.status_log.append(f"Error during CST thickening: {e}")
    
    def clear_cst(self) -> None:
        """Clear CST fitting results."""
        self.cst_processor.clear_data()
        self.window.cst_panel.set_cst_fitted(False)
        
        # Update plot to show only original data
        self._update_display()
        
        self.window.status_log.append("CST fitting results cleared.")
    
    def _update_display(self) -> None:
        """Update the plot display based on current settings."""
        if not self.cst_processor.is_fitted():
            # If no CST data, just show original data
            self.main_controller.processor._request_plot_update()
            return
        
        # Request plot update from CST processor
        self.cst_processor.request_plot_update()
    
    def _update_plot_from_cst(self, plot_data: dict[str, Any]) -> None:
        """Receive plot data from CST processor and forward to the widget."""
        try:
            chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
        except Exception:
            chord_length_mm = None
        
        # Clear the plot
        self.window.plot_widget.clear()
        
        # Map CST data to expected plot widget parameters
        plot_kwargs = {}
        
        # Handle original data
        if 'original_upper' in plot_data and 'original_lower' in plot_data:
            plot_kwargs['upper_data'] = plot_data['original_upper']
            plot_kwargs['lower_data'] = plot_data['original_lower']
        
        # Handle CST data (will be plotted separately in the plot widget)
        if 'cst_upper' in plot_data:
            plot_kwargs['cst_upper'] = plot_data['cst_upper']
        if 'cst_lower' in plot_data:
            plot_kwargs['cst_lower'] = plot_data['cst_lower']
        if 'comb_cst' in plot_data:
            plot_kwargs['comb_cst'] = plot_data['comb_cst']
        if 'cst_metrics' in plot_data:
            plot_kwargs['cst_metrics'] = plot_data['cst_metrics']
        # Add CST worst-error markers for visualization
        if 'worst_cst_upper_max_error' in plot_data:
            plot_kwargs['worst_cst_upper_max_error'] = plot_data['worst_cst_upper_max_error']
        if 'worst_cst_lower_max_error' in plot_data:
            plot_kwargs['worst_cst_lower_max_error'] = plot_data['worst_cst_lower_max_error']
        if 'worst_cst_upper_max_error_idx' in plot_data:
            plot_kwargs['worst_cst_upper_max_error_idx'] = plot_data['worst_cst_upper_max_error_idx']
        if 'worst_cst_lower_max_error_idx' in plot_data:
            plot_kwargs['worst_cst_lower_max_error_idx'] = plot_data['worst_cst_lower_max_error_idx']
        
        # Add chord length
        if chord_length_mm is not None:
            plot_kwargs['chord_length_mm'] = chord_length_mm
        
        # Plot the data
        self.window.plot_widget.plot_airfoil(**plot_kwargs)
    
    def get_cst_data(self):
        """Get CST data for use in Bezier optimization."""
        if self.cst_processor.is_fitted():
            return self.cst_processor.get_cst_data()
        else:
            # Return original data if no CST fit available
            return (self.main_controller.processor.upper_data, self.main_controller.processor.lower_data)
    
    def is_cst_available(self) -> bool:
        """Check if CST data is available for optimization."""
        return self.cst_processor.is_fitted() 

    def export_cst_dat(self) -> None:
        """Export current CST fit as high-resolution .dat file (Selig format)."""
        if not self.cst_processor.is_fitted():
            self.window.status_log.append("No CST fit available. Please fit CST first.")
            return
        try:
            # Use same dialog/flow as Bezier DAT export, but with CST samples
            from PySide6.QtWidgets import QFileDialog
            from utils.data_loader import export_airfoil_to_selig_format
            # Points per surface reuse file panel control if available; default 400
            try:
                points_per_surface = self.window.file_panel.points_per_surface_input.value()
            except Exception:
                points_per_surface = 400

            # Sample dense CST curves
            fitter = self.cst_processor.cst_fitter
            uc = self.cst_processor.upper_coefficients
            lc = self.cst_processor.lower_coefficients
            xs = np.linspace(0.0, 1.0, int(points_per_surface))
            upper = np.column_stack([xs, fitter.cst_function(xs, uc)])
            lower = np.column_stack([xs, fitter.cst_function(xs, lc)])

            # File dialog
            default_name = getattr(self.main_controller.processor, "airfoil_name", "airfoil") or "airfoil"
            default_path = f"{default_name}_cst_highres.dat"
            file_path, _ = QFileDialog.getSaveFileName(
                self.window,
                "Save CST High-Resolution .dat File",
                default_path,
                "DAT Files (*.dat);;All Files (*)",
            )
            if not file_path:
                self.window.status_log.append("CST .dat export cancelled by user.")
                return

            export_airfoil_to_selig_format(upper, lower, default_name, file_path)
            self.window.status_log.append(f"CST .dat export successful to '{file_path}'.")
            self.window.status_log.append(f"Exported {len(xs)} points per surface in Selig format.")

        except Exception as e:
            self.window.status_log.append(f"Error during CST .dat export: {e}")