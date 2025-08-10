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

        # TE Vector Points dropdown
        self.default_te_vector_points = config.DEFAULT_TE_VECTOR_POINTS  # integer!
        self.te_vector_points_combo = QComboBox()
        self.te_vector_points_combo.addItems([str(i) for i in range(2, 6)])  # 1-5
        self.te_vector_points_combo.setCurrentText(str(self.default_te_vector_points))
        self.te_vector_points_combo.setFixedWidth(80)

        # Strategy dropdown (renamed from Fitting Strategy)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Fixed-x",
            "Free-x"
        ])
        self.strategy_combo.setCurrentText("Fixed-x")  # Default to fastest, most reliable method
        self.strategy_combo.setFixedWidth(120)
        # Tooltips for strategy dropdown
        self.strategy_combo.setToolTip(
            "Fixed-x: Fastest method (~5-10s), good accuracy\n"
            "Free-x: Uses variable x-locations for control points\n"
            "Note: Check 'Enforce G2 at leading edge' to use coupled optimization with G2 continuity\n"
        )

        # Error Function dropdown (new)
        self.error_function_combo = QComboBox()
        self.error_function_combo.addItems([
            "Euclidean",
            "Orthogonal"
        ])
        self.error_function_combo.setCurrentText("Euclidean")  # Default as requested
        self.error_function_combo.setFixedWidth(120)
        # Tooltips for error function dropdown
        self.error_function_combo.setToolTip(
            "Euclidean: Uses linear sampling, measures point-to-point distance\n"
            "Orthogonal: Uses dynamic sampling, measures perpendicular distance to curve\n"
        )

        # Objective dropdown (new)
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "MSR",
            "Max Error"
        ])
        self.objective_combo.setCurrentText("MSR")
        self.objective_combo.setFixedWidth(120)
        self.objective_combo.setToolTip(
            "MSR: Mean squared residual (sum of squared errors)\n"
            "Max Error: Minimize the maximum absolute error (minmax objective)\n"
        )

        # Enforce G2 at leading edge
        self.g2_checkbox = QCheckBox("Enforce G2 at leading edge")

        # Action buttons
        self.build_single_bezier_button = QPushButton("Generate Airfoil")
        self.recalculate_button = QPushButton("Recalculate")
        self.recalculate_button.setEnabled(False)  # Initially disabled

        # --- Layout ------------------------------------------------------
        layout = QVBoxLayout()

        # Regularisation
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Control Point Smoothing:"))
        reg_row.addWidget(self.single_bezier_reg_weight_input)
        reg_row.addStretch(1)
        layout.addLayout(reg_row)

        # TE Vector Points and Recalculate button in same row
        te_row = QHBoxLayout()
        te_row.addWidget(QLabel("TE Vector Points:"))
        te_row.addWidget(self.te_vector_points_combo)
        te_row.addWidget(self.recalculate_button)
        te_row.addStretch(1)
        layout.addLayout(te_row)

        # Strategy (renamed from Fitting Strategy)
        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("Strategy:"))
        strategy_row.addWidget(self.strategy_combo)
        strategy_row.addStretch(1)
        layout.addLayout(strategy_row)

        # Error Function (new)
        error_row = QHBoxLayout()
        error_row.addWidget(QLabel("Error Function:"))
        error_row.addWidget(self.error_function_combo)
        error_row.addStretch(1)
        layout.addLayout(error_row)

        # Objective (new)
        objective_row = QHBoxLayout()
        objective_row.addWidget(QLabel("Objective:"))
        objective_row.addWidget(self.objective_combo)
        objective_row.addStretch(1)
        layout.addLayout(objective_row)

        # G2 checkbox
        g2_row = QHBoxLayout()
        g2_row.addWidget(self.g2_checkbox)
        g2_row.addStretch(1)
        layout.addLayout(g2_row)

        # Buttons
        button_row = QHBoxLayout()
        button_row.addWidget(self.build_single_bezier_button)
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