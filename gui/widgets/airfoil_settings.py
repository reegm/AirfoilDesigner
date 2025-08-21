"""General airfoil parameters panel: chord length, trailing-edge thickness, thickening action."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from core import config


class AirfoilSettingsWidget(QGroupBox):
    """Allows editing chord length and trailing-edge thickening."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Airfoil Parameters", parent)

        # Inputs
        self.chord_length_input = QLineEdit(str(config.DEFAULT_CHORD_LENGTH_MM))
        self.chord_length_input.setFixedWidth(80)

        self.te_thickness_input = QLineEdit(str(config.DEFAULT_TE_THICKNESS_MM))
        self.te_thickness_input.setFixedWidth(80)

        # Action button
        self.toggle_thickening_button = QPushButton("Thicken")

        # Layout
        layout = QVBoxLayout()

        # Chord length row
        chord_row = QHBoxLayout()
        chord_row.addWidget(QLabel("Chord Length (mm):"))
        chord_row.addWidget(self.chord_length_input)
        chord_row.addStretch(1)
        layout.addLayout(chord_row)

        # TE thickness row with thickening button
        te_row = QHBoxLayout()
        te_row.addWidget(QLabel("TE Thickness (mm):"))
        te_row.addWidget(self.te_thickness_input)
        te_row.addWidget(self.toggle_thickening_button)
        te_row.addStretch(1)
        layout.addLayout(te_row)

        self.setLayout(layout) 