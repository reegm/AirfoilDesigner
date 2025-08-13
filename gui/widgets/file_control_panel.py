"""Widgets related to file operations: loading airfoil data and exporting DXF files."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QSizePolicy,
    QSpinBox,
    QComboBox,
)
from PySide6.QtCore import Qt


class FileControlPanel(QGroupBox):
    """Panel containing *Load* and *Export* actions for airfoil files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("File Operations", parent)

        # --- Widgets -----------------------------------------------------
        self.load_button = QPushButton("Load Airfoil File")
        self.load_button.setMinimumWidth(120)  # Give button a minimum width
        self.file_path_label = QLabel("No file loaded")
        
        self.export_dxf_button = QPushButton("Export DXF")
        self.export_dxf_button.setMinimumWidth(120)  # Give button a minimum width
        
        # DXF export type selection
        self.dxf_export_type_label = QLabel("DXF Type:")
        self.dxf_export_type_combo = QComboBox()
        self.dxf_export_type_combo.addItems(["Clamped Spline", "NURBS Fit", "NURBS Control"])
        self.dxf_export_type_combo.setMinimumWidth(120)
        
        # NURBS parameters
        self.nurbs_degree_label = QLabel("NURBS Degree:")
        self.nurbs_degree_input = QSpinBox()
        self.nurbs_degree_input.setMinimum(1)
        self.nurbs_degree_input.setMaximum(9)
        self.nurbs_degree_input.setValue(3)
        self.nurbs_degree_input.setMinimumWidth(60)
        
        self.nurbs_samples_label = QLabel("Samples:")
        self.nurbs_samples_input = QSpinBox()
        self.nurbs_samples_input.setMinimum(50)
        self.nurbs_samples_input.setMaximum(500)
        self.nurbs_samples_input.setValue(200)
        self.nurbs_samples_input.setMinimumWidth(60)
        
        self.export_dat_button = QPushButton("Export DAT")
        self.export_dat_button.setMinimumWidth(120)  # Give button a minimum width
        
        self.points_per_surface_label = QLabel("Points per surface:")
        self.points_per_surface_input = QSpinBox()
        self.points_per_surface_input.setMinimum(10)
        self.points_per_surface_input.setMaximum(1000)
        self.points_per_surface_input.setValue(100)
        self.points_per_surface_input.setMinimumWidth(80)
        
        # --- Layout ------------------------------------------------------
        file_layout = QHBoxLayout()
        file_layout.setSpacing(10)
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.file_path_label, 1)

        export_layout = QHBoxLayout()
        export_layout.setSpacing(10)
        export_layout.addWidget(self.export_dxf_button)
        export_layout.addWidget(self.dxf_export_type_label)
        export_layout.addWidget(self.dxf_export_type_combo)
        export_layout.addStretch(1)

        nurbs_params_layout = QHBoxLayout()
        nurbs_params_layout.setSpacing(10)
        nurbs_params_layout.addWidget(self.nurbs_degree_label)
        nurbs_params_layout.addWidget(self.nurbs_degree_input)
        nurbs_params_layout.addWidget(self.nurbs_samples_label)
        nurbs_params_layout.addWidget(self.nurbs_samples_input)
        nurbs_params_layout.addStretch(1)

        # dat_export_layout = QHBoxLayout()
        # dat_export_layout.setSpacing(10)
        # dat_export_layout.addWidget(self.export_dat_button)
        # dat_export_layout.addWidget(self.points_per_surface_label)
        # dat_export_layout.addWidget(self.points_per_surface_input)
        # dat_export_layout.addStretch(1)

        main_layout = QVBoxLayout()
        # main_layout.setContentsMargins(0, 0, 0, 0)  # Override QGroupBox margins
        # main_layout.setSpacing(10)
        main_layout.addLayout(file_layout)
        main_layout.addLayout(export_layout)
        main_layout.addLayout(nurbs_params_layout)
        # main_layout.addLayout(dat_export_layout)

        self.setLayout(main_layout) 