import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.controller import MainController


def main() -> None:
    """Launch the Qt GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    controller = MainController(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 