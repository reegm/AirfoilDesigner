"""Widget holding settings related to the B-spline optimiser."""

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
    QSpinBox,
)

from core import config


class OptimizerSettingsWidget(QGroupBox):
    """Panel exposing parameters for the B-spline airfoil optimiser."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Optimizer Settings", parent)

        # --- Inputs ------------------------------------------------------
        # TE Vector Points dropdown
        self.default_te_vector_points = config.DEFAULT_TE_VECTOR_POINTS  # integer!
        self.te_vector_points_combo = QComboBox()
        self.te_vector_points_combo.addItems([str(i) for i in range(2, 6)])  # 1-5
        self.te_vector_points_combo.setCurrentText(str(self.default_te_vector_points))
        self.te_vector_points_combo.setFixedWidth(80)

        # Enforce G2 at leading edge
        self.g2_checkbox = QCheckBox("Enforce G2 at leading edge")
        self.g2_checkbox.setToolTip(
            "G2 continuity ensures smooth curvature transition at the leading edge.\n"
            "When enabled: Both surfaces share the same leading edge radius.\n"
            "When disabled: Only G1 (tangent) continuity is enforced."
        )

        # Enforce TE vector tangency
        self.enforce_te_tangency_checkbox = QCheckBox("Enforce TE vector tangency")
        self.enforce_te_tangency_checkbox.setChecked(False)  # Default to enabled
        self.enforce_te_tangency_checkbox.setToolTip(
            "When enabled: B-splines are constrained to be tangent to the computed trailing edge vectors.\n"
            "When disabled: Only endpoint constraints are applied, allowing better fit for some airfoils.\n"
            "Note: Disabling may improve fit quality for certain input files."
        )

        # B-spline settings
        self.bspline_cp_label = QLabel("B-spline control points:")
        self.bspline_cp_spin = QSpinBox()
        self.bspline_cp_spin.setMinimum(6)
        self.bspline_cp_spin.setMaximum(50)
        self.bspline_cp_spin.setValue(10)

        # Action buttons
        self.fit_bspline_button = QPushButton("Fit B-spline")
        self.recalculate_button = QPushButton("Recalculate")
        self.recalculate_button.setEnabled(False)  # Initially disabled

        # --- Layout ------------------------------------------------------
        layout = QVBoxLayout()

        # TE Vector Points and Recalculate button in same row
        te_row = QHBoxLayout()
        te_row.addWidget(QLabel("TE Vector Points:"))
        te_row.addWidget(self.te_vector_points_combo)
        te_row.addWidget(self.recalculate_button)
        te_row.addStretch(1)
        layout.addLayout(te_row)

        # B-spline row
        bs_row = QHBoxLayout()
        bs_row.addWidget(self.bspline_cp_label)
        bs_row.addWidget(self.bspline_cp_spin)
        bs_row.addWidget(self.fit_bspline_button)
        bs_row.addStretch(1)
        layout.addLayout(bs_row)

        # G2 checkbox
        g2_row = QHBoxLayout()
        g2_row.addWidget(self.g2_checkbox)
        g2_row.addWidget(self.enforce_te_tangency_checkbox)
        g2_row.addStretch(1)
        layout.addLayout(g2_row)

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