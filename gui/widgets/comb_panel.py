"""Controls for curvature comb visualisation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QWidget,
)
from core.config import (
    COMB_DENSITY_MIN,
    COMB_DENSITY_MAX,
    COMB_DENSITY_DEFAULT,
)


class CombPanelWidget(QGroupBox):
    """Panel that tweaks scale and density of the curvature comb."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Curvature Comb", parent)

        # Scale slider (value / 1000)
        self.comb_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.comb_scale_slider.setMinimum(1)
        self.comb_scale_slider.setMaximum(100)
        self.comb_scale_slider.setValue(50)
        self.comb_scale_slider.setFixedWidth(120)
        self.comb_scale_label = QLabel("0.050")
        self.comb_scale_label.setFixedWidth(50)

        # Density slider
        self.comb_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.comb_density_slider.setMinimum(COMB_DENSITY_MIN)
        self.comb_density_slider.setMaximum(COMB_DENSITY_MAX)
        self.comb_density_slider.setValue(COMB_DENSITY_DEFAULT)
        self.comb_density_slider.setFixedWidth(120)
        self.comb_density_label = QLabel(str(COMB_DENSITY_DEFAULT))
        self.comb_density_label.setFixedWidth(50)

        # Layout
        layout = QVBoxLayout()

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Comb Scale:"))
        scale_row.addWidget(self.comb_scale_slider)
        scale_row.addWidget(self.comb_scale_label)
        layout.addLayout(scale_row)

        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Comb Density:"))
        density_row.addWidget(self.comb_density_slider)
        density_row.addWidget(self.comb_density_label)
        layout.addLayout(density_row)

        self.setLayout(layout) 