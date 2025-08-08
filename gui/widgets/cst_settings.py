"""Widget holding settings related to CST fitting."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QSpinBox,
)

from core import config


class CSTSettingsWidget(QGroupBox):
    """Panel exposing parameters for CST fitting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("CST Settings", parent)

        # --- Inputs ------------------------------------------------------
        # Degree of Bernstein polynomials
        self.degree_spinbox = QSpinBox()
        self.degree_spinbox.setRange(4, 20)
        self.degree_spinbox.setValue(config.CST_DEFAULT_DEGREE)
        self.degree_spinbox.setFixedWidth(80)
        self.degree_spinbox.setToolTip("Degree of Bernstein polynomials (number of coefficients = degree + 1)")

        # Action buttons
        self.fit_cst_button = QPushButton("Fit CST")
        self.fit_cst_button.setToolTip("Perform CST fitting on loaded airfoil data")
        self.build_deg9_button = QPushButton("Build Deg-9 Bézier from CST")
        self.build_deg9_button.setToolTip("Approximate CST with single-span degree-9 Bézier (orthogonal, constrained)")
        self.export_cst_dat_button = QPushButton("Export CST DAT")
        self.export_cst_dat_button.setToolTip("Export current CST curves as high-resolution .dat (Selig format)")
        
        # --- Layout ------------------------------------------------------
        layout = QVBoxLayout()

        # Degree row
        degree_row = QHBoxLayout()
        degree_row.addWidget(QLabel("Degree:"))
        degree_row.addWidget(self.degree_spinbox)
        degree_row.addStretch(1)
        layout.addLayout(degree_row)

        # Buttons row
        button_row = QHBoxLayout()
        button_row.addWidget(self.fit_cst_button)
        button_row.addWidget(self.build_deg9_button)
        button_row.addWidget(self.export_cst_dat_button)
        layout.addLayout(button_row)

        layout.addStretch(1)
        self.setLayout(layout)

    def get_parameters(self) -> dict:
        """Get current CST parameters."""
        return {
            'degree': self.degree_spinbox.value(),
        }

    def set_parameters(self, degree: int = 8, n1: float = 0.5, n2: float = 1.0):
        """Set CST parameters."""
        self.degree_spinbox.setValue(degree)
        
    def set_cst_fitted(self, fitted: bool):
        """Enable/disable buttons based on CST fitting state."""
        if fitted:
            self.fit_cst_button.setText("Refit CST")
        else:
            self.fit_cst_button.setText("Fit CST") 