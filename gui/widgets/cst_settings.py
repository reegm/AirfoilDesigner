"""Widget holding settings related to CST fitting."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)


class CSTSettingsWidget(QGroupBox):
    """Panel exposing parameters for CST fitting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("CST Settings", parent)

        # --- Inputs ------------------------------------------------------
        # Degree of Bernstein polynomials
        self.degree_spinbox = QSpinBox()
        self.degree_spinbox.setRange(4, 20)
        self.degree_spinbox.setValue(15)
        self.degree_spinbox.setFixedWidth(80)
        self.degree_spinbox.setToolTip("Degree of Bernstein polynomials (number of coefficients = degree + 1)")

        # N1 parameter (leading edge)
        self.n1_spinbox = QDoubleSpinBox()
        self.n1_spinbox.setRange(0.1, 2.0)
        self.n1_spinbox.setValue(0.5)
        self.n1_spinbox.setSingleStep(0.1)
        self.n1_spinbox.setDecimals(2)
        self.n1_spinbox.setFixedWidth(80)
        self.n1_spinbox.setToolTip("Class function parameter for leading edge (typically 0.5 for airfoils)")

        # N2 parameter (trailing edge)
        self.n2_spinbox = QDoubleSpinBox()
        self.n2_spinbox.setRange(0.1, 2.0)
        self.n2_spinbox.setValue(1.0)
        self.n2_spinbox.setSingleStep(0.1)
        self.n2_spinbox.setDecimals(2)
        self.n2_spinbox.setFixedWidth(80)
        self.n2_spinbox.setToolTip("Class function parameter for trailing edge (typically 1.0 for airfoils)")

        # Show original data checkbox
        self.show_original_checkbox = QCheckBox("Show Original")
        self.show_original_checkbox.setChecked(True)
        self.show_original_checkbox.setToolTip("Show original airfoil data on plot")

        # Show CST fit checkbox
        self.show_cst_checkbox = QCheckBox("Show CST Fit")
        self.show_cst_checkbox.setChecked(True)
        self.show_cst_checkbox.setToolTip("Show CST fit data on plot")

        # Action buttons
        self.fit_cst_button = QPushButton("Fit CST")
        self.fit_cst_button.setToolTip("Perform CST fitting on loaded airfoil data")
        
        self.clear_cst_button = QPushButton("Clear CST")
        self.clear_cst_button.setToolTip("Clear CST fitting results")
        self.clear_cst_button.setEnabled(False)

        # --- Layout ------------------------------------------------------
        layout = QVBoxLayout()

        # Degree row
        degree_row = QHBoxLayout()
        degree_row.addWidget(QLabel("Degree:"))
        degree_row.addWidget(self.degree_spinbox)
        degree_row.addStretch(1)
        layout.addLayout(degree_row)

        # N1 row
        n1_row = QHBoxLayout()
        n1_row.addWidget(QLabel("N1 (LE):"))
        n1_row.addWidget(self.n1_spinbox)
        n1_row.addStretch(1)
        layout.addLayout(n1_row)

        # N2 row
        n2_row = QHBoxLayout()
        n2_row.addWidget(QLabel("N2 (TE):"))
        n2_row.addWidget(self.n2_spinbox)
        n2_row.addStretch(1)
        layout.addLayout(n2_row)

        # Display options
        display_row = QHBoxLayout()
        display_row.addWidget(self.show_original_checkbox)
        display_row.addWidget(self.show_cst_checkbox)
        layout.addLayout(display_row)

        # Buttons row
        button_row = QHBoxLayout()
        button_row.addWidget(self.fit_cst_button)
        button_row.addWidget(self.clear_cst_button)
        layout.addLayout(button_row)

        layout.addStretch(1)
        self.setLayout(layout)

    def get_parameters(self) -> dict:
        """Get current CST parameters."""
        return {
            'degree': self.degree_spinbox.value(),
            'n1': self.n1_spinbox.value(),
            'n2': self.n2_spinbox.value(),
            'show_original': self.show_original_checkbox.isChecked(),
            'show_cst': self.show_cst_checkbox.isChecked()
        }

    def set_parameters(self, degree: int = 8, n1: float = 0.5, n2: float = 1.0):
        """Set CST parameters."""
        self.degree_spinbox.setValue(degree)
        self.n1_spinbox.setValue(n1)
        self.n2_spinbox.setValue(n2)

    def set_cst_fitted(self, fitted: bool):
        """Enable/disable buttons based on CST fitting state."""
        self.clear_cst_button.setEnabled(fitted)
        if fitted:
            self.fit_cst_button.setText("Refit CST")
        else:
            self.fit_cst_button.setText("Fit CST") 