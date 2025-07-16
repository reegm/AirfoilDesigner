import sys
import os
import copy
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, QSizePolicy,
    QSlider, QComboBox, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Central configuration
from core import config
from core.airfoil_processor import AirfoilProcessor
from gui.airfoil_plot_widget import AirfoilPlotWidget
from utils.dxf_exporter import export_curves_to_dxf

class AirfoilDesignerApp(QMainWindow):
    """Main application window for the Airfoil Designer."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airfoil Designer")
        self.setGeometry(100, 100, 1200, 800)

        self.processor = AirfoilProcessor()
        self.processor.log_message.connect(self._log_to_display)
        self.processor.plot_update_requested.connect(self._update_plot_from_processor)

        self.init_ui()

        self.comb_scale_slider.valueChanged.connect(self._comb_params_changed)
        self.comb_density_slider.valueChanged.connect(self._comb_params_changed)
        self._update_comb_labels()

        self._update_button_states() # Set initial button states
        self.processor.log_message.emit("Application started. Load an airfoil .dat file to begin.")

    def init_ui(self):
        """Initializes the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Control Panel (Left Side) ---
        control_panel_layout = QVBoxLayout()
        control_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- File Operations Panel ---
        file_group = QGroupBox("File Operations")
        file_group_layout = QVBoxLayout()
        file_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Airfoil File")
        self.load_button.clicked.connect(self._load_airfoil_file_action)
        file_layout.addWidget(self.load_button)
        self.file_path_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_path_label)
        file_group_layout.addLayout(file_layout)
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_single_bezier_dxf_button = QPushButton("Export Single Bezier DXF")
        self.export_single_bezier_dxf_button.clicked.connect(self._export_single_bezier_dxf_action)
        export_layout.addWidget(self.export_single_bezier_dxf_button)
        file_group_layout.addLayout(export_layout)
        file_group.setLayout(file_group_layout)
        control_panel_layout.addWidget(file_group)

        # --- Single Bezier Settings Panel ---
        single_group = QGroupBox("Single Bezier Settings")
        single_group_layout = QVBoxLayout()
        # Regularization Weight
        single_bezier_reg_weight_layout = QHBoxLayout()
        single_bezier_reg_weight_layout.addWidget(QLabel("Single Bezier Reg. Weight:"))
        self.single_bezier_reg_weight_input = QLineEdit(str(config.DEFAULT_REGULARIZATION_WEIGHT))
        self.single_bezier_reg_weight_input.setFixedWidth(80)
        single_bezier_reg_weight_layout.addWidget(self.single_bezier_reg_weight_input)
        single_bezier_reg_weight_layout.addStretch(1)
        single_group_layout.addLayout(single_bezier_reg_weight_layout)
        # Error Function
        error_func_layout = QHBoxLayout()
        error_func_layout.addWidget(QLabel("Single Bezier Error Function:"))
        self.error_func_dropdown = QComboBox()
        self.error_func_dropdown.addItems(["icp_iter_single", "mse" ])
        self.error_func_dropdown.setCurrentIndex(0)
        error_func_layout.addWidget(self.error_func_dropdown)
        error_func_layout.addStretch(1)
        single_group_layout.addLayout(error_func_layout)

        # G2 Continuity checkbox
        g2_layout = QHBoxLayout()
        self.g2_checkbox = QCheckBox("Enforce G2 at LE")
        g2_layout.addWidget(self.g2_checkbox)
        g2_layout.addStretch(1)
        single_group_layout.addLayout(g2_layout)
        # Single Bezier button
        self.build_single_bezier_button = QPushButton("Build Single Bezier Model")
        self.build_single_bezier_button.clicked.connect(self._build_single_bezier_action)
        single_group_layout.addWidget(self.build_single_bezier_button)
        single_group.setLayout(single_group_layout)
        control_panel_layout.addWidget(single_group)

        # --- General Parameters Panel ---
        general_group = QGroupBox("General Parameters")
        general_group_layout = QVBoxLayout()
        # Chord Length
        chord_length_layout = QHBoxLayout()
        chord_length_layout.addWidget(QLabel("Chord Length (mm):"))
        self.chord_length_input = QLineEdit(str(config.DEFAULT_CHORD_LENGTH_MM))
        self.chord_length_input.setFixedWidth(80)
        chord_length_layout.addWidget(self.chord_length_input)
        chord_length_layout.addStretch(1)
        general_group_layout.addLayout(chord_length_layout)
        # TE Thickness
        te_thickness_layout = QHBoxLayout()
        te_thickness_layout.addWidget(QLabel("TE Thickness (mm):"))
        self.te_thickness_input = QLineEdit(str(config.DEFAULT_TE_THICKNESS_MM))
        self.te_thickness_input.setFixedWidth(80)
        te_thickness_layout.addWidget(self.te_thickness_input)
        te_thickness_layout.addStretch(1)
        general_group_layout.addLayout(te_thickness_layout)
        # Thickening button
        self.toggle_thickening_button = QPushButton("Apply Thickening")
        self.toggle_thickening_button.clicked.connect(self._toggle_thickening_action)
        general_group_layout.addWidget(self.toggle_thickening_button)
        general_group.setLayout(general_group_layout)
        control_panel_layout.addWidget(general_group)

        # --- Curvature Comb Panel ---
        comb_group = QGroupBox("Curvature Comb")
        comb_group_layout = QVBoxLayout()
        # Comb Scale
        comb_scale_layout = QHBoxLayout()
        comb_scale_layout.addWidget(QLabel("Comb Scale:"))
        self.comb_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.comb_scale_slider.setMinimum(1)
        self.comb_scale_slider.setMaximum(100)
        self.comb_scale_slider.setValue(50)
        self.comb_scale_slider.setFixedWidth(120)
        comb_scale_layout.addWidget(self.comb_scale_slider)
        self.comb_scale_label = QLabel("")
        self.comb_scale_label.setFixedWidth(50)
        comb_scale_layout.addWidget(self.comb_scale_label)
        comb_group_layout.addLayout(comb_scale_layout)
        # Comb Density
        comb_density_layout = QHBoxLayout()
        comb_density_layout.addWidget(QLabel("Comb Density:"))
        self.comb_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.comb_density_slider.setMinimum(10)
        self.comb_density_slider.setMaximum(100)
        self.comb_density_slider.setValue(40)
        self.comb_density_slider.setFixedWidth(120)
        comb_density_layout.addWidget(self.comb_density_slider)
        self.comb_density_label = QLabel("")
        self.comb_density_label.setFixedWidth(50)
        comb_density_layout.addWidget(self.comb_density_label)
        comb_group_layout.addLayout(comb_density_layout)
        comb_group.setLayout(comb_group_layout)
        control_panel_layout.addWidget(comb_group)

        # --- Log Display ---
        control_panel_layout.addWidget(QLabel("--- Log ---"))
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Monospace", 9))
        control_panel_layout.addWidget(self.log_display, 1)

        main_layout.addLayout(control_panel_layout, 1)

        # --- Plot Area (Right Side) ---
        plot_area_layout = QVBoxLayout()
        self.plot_canvas = AirfoilPlotWidget(self)
        plot_area_layout.addWidget(self.plot_canvas)
        main_layout.addLayout(plot_area_layout, 3)

    # --- GUI Action Methods ---

    def _log_to_display(self, message):
        """Slot to receive and display log messages."""
        self.log_display.append(message)

    def _update_plot_from_processor(self, plot_data):
        """Slot to receive plot data from the processor and update the canvas and button states."""
        self._last_plot_data = plot_data.copy() # Store for re-plotting

        chord_length_mm = None
        try:
            chord_length_mm = float(self.chord_length_input.text())
        except Exception:
            chord_length_mm = None

        self.plot_canvas.plot_airfoil(
            **plot_data,
            chord_length_mm=chord_length_mm,
        )
        self._update_button_states()

    def _update_button_states(self):
        """Updates the enabled/disabled state and text of buttons based on processor state."""
        is_file_loaded = self.processor.core_processor.upper_data is not None
        is_model_built = self.processor.core_processor.single_bezier_upper_poly_sharp is not None
        is_thickened = self.processor._is_thickened
        is_trailing_edge_thickened = False
        if hasattr(self.processor, 'is_trailing_edge_thickened'):
            is_trailing_edge_thickened = self.processor.is_trailing_edge_thickened()

        # "Build Single Bezier Model" button
        self.build_single_bezier_button.setEnabled(is_file_loaded)
        

        # "Apply Thickening" / "Remove Thickening" button
        # Disable if trailing edge is thickened in the original data
        self.toggle_thickening_button.setEnabled(is_model_built and not is_trailing_edge_thickened)
        if is_thickened:
            self.toggle_thickening_button.setText("Remove Thickening")
        else:
            self.toggle_thickening_button.setText("Apply Thickening")

        # Export DXF buttons
        self.export_single_bezier_dxf_button.setEnabled(is_model_built)

        # Comb sliders
        self.comb_scale_slider.setEnabled(is_model_built)
        self.comb_density_slider.setEnabled(is_model_built)

    def _load_airfoil_file_action(self):
        """Handles the 'Load Airfoil File' button click."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Airfoil Data", "", "Airfoil Data Files (*.dat);;All Files (*)")

        if file_path:
            self.file_path_label.setText(os.path.basename(file_path))
            # self.plot_canvas._first_plot_done = False # Reset zoom for new file
            try:
                if self.processor.load_airfoil_data_and_initialize_model(file_path):
                    self.processor.log_message.emit(f"Successfully loaded '{os.path.basename(file_path)}'.")
                else:
                    self.processor.log_message.emit(f"Failed to load '{os.path.basename(file_path)}'. Check file format and content.")
            except Exception as e:
                self.processor.log_message.emit(f"An unexpected error occurred during file loading: {e}")
            finally:
                self._update_button_states() # Always update button states after attempting load

    def _comb_params_changed(self):
        """Handles changes in the comb scale or density sliders."""
        if self.processor is None:
            return

        self._update_comb_labels()

        scale = self.comb_scale_slider.value() / 1000.0
        density = self.comb_density_slider.value()
        
        # Check if any model has been generated before sending update request
        is_model_present = self.processor.core_processor.single_bezier_upper_poly_sharp is not None

        if is_model_present:
            self.processor.request_plot_update_with_comb_params(scale, density)

    def _update_comb_labels(self):
        """Updates the labels for the comb control sliders."""
        # Scale factor is value / 1000 for finer control
        scale_val = self.comb_scale_slider.value() / 1000.0
        self.comb_scale_label.setText(f"{scale_val:.3f}")

        density_val = self.comb_density_slider.value()
        self.comb_density_label.setText(f"{density_val}")

    def _build_single_bezier_action(self):
        """
        Handles the 'Build Single Bezier Model' button click.
        """
        try:
            regularization_weight = float(self.single_bezier_reg_weight_input.text())
            error_function = self.error_func_dropdown.currentText().lower()
            g2_flag = self.g2_checkbox.isChecked()
            self.processor.build_single_bezier_model(regularization_weight, error_function=error_function, enforce_g2=g2_flag)
            self._comb_params_changed()
        except ValueError:
            self.processor.log_message.emit("Error: Invalid input for regularization weight. Please enter a number.")
        except Exception as e:
            self.processor.log_message.emit(f"Error building single Bezier model: {e}")

    def _toggle_thickening_action(self):
        """Handles the 'Apply Thickening' / 'Remove Thickening' button click."""
        try:
            te_thickness_percent = float(self.te_thickness_input.text()) / (float(self.chord_length_input.text()) / 100.0)
            self.processor.toggle_thickening(te_thickness_percent)
            self._comb_params_changed()  # Ensure comb settings are reapplied after thickening
        except ValueError:
            self.processor.log_message.emit("Error: Invalid TE Thickness. Please enter a number.")
        except Exception as e:
            self.processor.log_message.emit(f"An unexpected error occurred during thickening toggle: {e}")

    def _get_default_dxf_filename(self):
        """Generate default DXF filename based on profile name."""
        import re
        
        # Get profile name from core processor
        profile_name = getattr(self.processor.core_processor, 'airfoil_name', None)
        
        if profile_name:
            # Sanitize profile name: allow letters, numbers, dash, underscore
            sanitized = re.sub(r"[^A-Za-z0-9\-_]+", "_", profile_name)
            if sanitized:
                return f"{sanitized}.dxf"
        
        # Fallback to generic name
        return "airfoil.dxf"

    def _export_single_bezier_dxf_action(self):
        """Handles the 'Export Single Bezier DXF' button click."""
        polygons_to_export = None
        if self.processor._is_thickened and self.processor._thickened_single_bezier_polygons:
            polygons_to_export = self.processor._thickened_single_bezier_polygons
            self.processor.log_message.emit("Preparing to export thickened single Bezier model.")
        elif self.processor.core_processor.single_bezier_upper_poly_sharp is not None:
            polygons_to_export = [self.processor.core_processor.single_bezier_upper_poly_sharp,
                                  self.processor.core_processor.single_bezier_lower_poly_sharp]
            self.processor.log_message.emit("Preparing to export sharp single Bezier model.")

        if polygons_to_export is None:
            self.processor.log_message.emit("Error: Single Bezier model not available for export. Please build it first.")
            return

        is_merged_export = True # Single Bezier model exports as a merged curve

        try:
            chord_length_mm = float(self.chord_length_input.text())
        except ValueError:
            self.processor.log_message.emit("Error: Invalid chord length. Please enter a number.")
            return

        dxf_doc = export_curves_to_dxf(
            polygons_to_export,
            chord_length_mm,
            self.processor.log_message.emit
        )

        if dxf_doc:
            default_filename = self._get_default_dxf_filename()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Single Bezier DXF File", default_filename, "DXF Files (*.dxf)")
            if file_path:
                try:
                    dxf_doc.saveas(file_path)
                    self.processor.log_message.emit(f"Single Bezier DXF export successful to '{os.path.basename(file_path)}'.")
                    self.processor.log_message.emit("Note: For correct scale in CAD software, ensure import settings are configured for millimeters.")
                except IOError as e:
                    self.processor.log_message.emit(f"Could not save DXF file: {e}")
            else:
                self.processor.log_message.emit("Single Bezier DXF export cancelled by user.")
        else:
            self.processor.log_message.emit("Single Bezier DXF export failed during document creation.")