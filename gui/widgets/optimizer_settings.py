"""Widget holding settings related to the Bezier optimiser."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QWidget,
)

from core import config


class OptimizerSettingsWidget(QGroupBox):
    """Panel exposing parameters for the single-Bezier airfoil optimiser."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Optimizer Settings", parent)

        # --- Inputs ------------------------------------------------------
        # Regularisation weight
        self.single_bezier_reg_weight_input = QLineEdit(str(config.DEFAULT_REGULARIZATION_WEIGHT))
        self.single_bezier_reg_weight_input.setFixedWidth(80)

        # Curve error sample points
        self.curve_error_points_input = QLineEdit(str(config.NUM_POINTS_CURVE_ERROR))
        self.curve_error_points_input.setFixedWidth(80)

        # Enforce G2 at leading edge
        self.g2_checkbox = QCheckBox("Enforce G2 at leading edge")

        # Action button
        self.build_single_bezier_button = QPushButton("Generate Airfoil")

        # --- Layout ------------------------------------------------------
        layout = QVBoxLayout()

        # Regularisation
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Control Point Smoothing:"))
        reg_row.addWidget(self.single_bezier_reg_weight_input)
        reg_row.addStretch(1)
        layout.addLayout(reg_row)

        # Curve error points
        err_row = QHBoxLayout()
        err_row.addWidget(QLabel("Curve Error Points:"))
        err_row.addWidget(self.curve_error_points_input)
        err_row.addStretch(1)
        layout.addLayout(err_row)

        # G2 checkbox
        g2_row = QHBoxLayout()
        g2_row.addWidget(self.g2_checkbox)
        g2_row.addStretch(1)
        layout.addLayout(g2_row)

        # Generate button
        layout.addWidget(self.build_single_bezier_button)

        self.setLayout(layout) 