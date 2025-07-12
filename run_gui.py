import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import AirfoilDesignerApp


def main() -> None:
    """Launch the Qt GUI application."""
    app = QApplication(sys.argv)
    window = AirfoilDesignerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 