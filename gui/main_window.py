"""Main window layout for the Airfoil Designer GUI.

This module purposefully holds *only* the Qt layout code – no business
logic. All interactions are delegated to :pyclass:`gui.controllers.MainController`.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt

from gui.widgets import (
    FileControlPanel,
    OptimizerSettingsWidget,
    AirfoilSettingsWidget,
    CombPanelWidget,
    StatusLogWidget,
    AirfoilPlotWidget,
    CSTSettingsWidget,
)


__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Top-level window that arranges all GUI widgets."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Airfoil Designer")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ------------------------------------------------------------------
        # Left-hand control panel
        # ------------------------------------------------------------------
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.file_panel = FileControlPanel(self)
        self.cst_panel = CSTSettingsWidget(self)
        self.optimizer_panel = OptimizerSettingsWidget(self)
        self.airfoil_settings_panel = AirfoilSettingsWidget(self)
        self.comb_panel = CombPanelWidget(self)
        self.status_log = StatusLogWidget(self)

        control_layout.addWidget(self.file_panel)
        control_layout.addWidget(self.cst_panel)
        control_layout.addWidget(self.optimizer_panel)
        control_layout.addWidget(self.airfoil_settings_panel)
        control_layout.addWidget(self.comb_panel)
        control_layout.addWidget(self.status_log, 1)

        main_layout.addLayout(control_layout, 1)

        # ------------------------------------------------------------------
        # Right-hand plot area
        # ------------------------------------------------------------------
        plot_layout = QVBoxLayout()
        self.plot_widget = AirfoilPlotWidget(self)
        plot_layout.addWidget(self.plot_widget)

        main_layout.addLayout(plot_layout, 3)