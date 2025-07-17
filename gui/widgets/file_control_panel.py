"""Widgets related to file operations: loading airfoil data and exporting DXF files."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
)
from PySide6.QtCore import Qt


class FileControlPanel(QGroupBox):
    """Panel containing *Load* and *Export* actions for airfoil files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("File Operations", parent)

        # --- Widgets -----------------------------------------------------
        self.load_button = QPushButton("Load Airfoil File")
        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.export_dxf_button = QPushButton("Export DXF")

        # --- Layout ------------------------------------------------------
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.file_path_label, 1)

        export_layout = QHBoxLayout()
        export_layout.addWidget(self.export_dxf_button)
        export_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(file_layout)
        main_layout.addLayout(export_layout)

        self.setLayout(main_layout) 