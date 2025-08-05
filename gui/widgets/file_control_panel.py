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
        export_layout.addStretch(1)

        dat_export_layout = QHBoxLayout()
        dat_export_layout.setSpacing(10)
        dat_export_layout.addWidget(self.export_dat_button)
        dat_export_layout.addWidget(self.points_per_surface_label)
        dat_export_layout.addWidget(self.points_per_surface_input)
        dat_export_layout.addStretch(1)

        main_layout = QVBoxLayout()
        # main_layout.setContentsMargins(0, 0, 0, 0)  # Override QGroupBox margins
        # main_layout.setSpacing(10)
        main_layout.addLayout(file_layout)
        main_layout.addLayout(export_layout)
        main_layout.addLayout(dat_export_layout)

        self.setLayout(main_layout) 