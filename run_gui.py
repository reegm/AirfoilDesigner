import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.controller import MainController
import multiprocessing # Add this import

def main() -> None:
    """Launch the Qt GUI application."""
    app = QApplication(sys.argv)
    
    try:
        app.setWindowIcon(QIcon('img/favicon.ico'))
    except Exception as e:
        print(f"Warning: Could not set application icon. Ensure '{'img/favicon.ico'}' exists and is a valid icon file. Error: {e}")

    window = MainWindow()
    controller = MainController(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()