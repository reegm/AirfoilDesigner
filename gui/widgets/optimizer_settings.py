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
    QComboBox,
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

        # TE Vector Points dropdown
        self.default_te_vector_points = config.DEFAULT_TE_VECTOR_POINTS  # integer!
        self.te_vector_points_combo = QComboBox()
        self.te_vector_points_combo.addItems([str(i) for i in range(1, 6)])  # 1-5
        self.te_vector_points_combo.setCurrentText(str(self.default_te_vector_points))
        self.te_vector_points_combo.setFixedWidth(80)

        # Enforce G2 at leading edge
        self.g2_checkbox = QCheckBox("Enforce G2 at leading edge")

        # Action buttons
        self.build_single_bezier_button = QPushButton("Generate Airfoil")
        self.recalculate_button = QPushButton("Recalculate TE vectors")
        self.recalculate_button.setEnabled(False)  # Initially disabled

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

        # TE Vector Points
        te_row = QHBoxLayout()
        te_row.addWidget(QLabel("TE Vector Points:"))
        te_row.addWidget(self.te_vector_points_combo)
        te_row.addStretch(1)
        layout.addLayout(te_row)

        # G2 checkbox
        g2_row = QHBoxLayout()
        g2_row.addWidget(self.g2_checkbox)
        g2_row.addStretch(1)
        layout.addLayout(g2_row)

        # Buttons
        button_row = QHBoxLayout()
        button_row.addWidget(self.build_single_bezier_button)
        button_row.addWidget(self.recalculate_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

        # Connect TE vector points dropdown to enable/disable recalc button
        self.te_vector_points_combo.currentIndexChanged.connect(self._update_recalc_button_state)
        self._update_recalc_button_state()  # Set initial state

    def _update_recalc_button_state(self):
        current = int(self.te_vector_points_combo.currentText())
        self.recalculate_button.setEnabled(current != self.default_te_vector_points)

    def set_default_te_vector_points(self, value: int):
        self.default_te_vector_points = value
        self._update_recalc_button_state()

    def disable_recalc_button(self):
        self.recalculate_button.setEnabled(False)
        # Explicitly clear any custom style to match default QPushButton disabled look
        self.recalculate_button.setStyleSheet("") 