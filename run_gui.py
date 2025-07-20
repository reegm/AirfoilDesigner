import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.controller import MainController
import multiprocessing # Add this import

def main() -> None:
    """Launch the Qt GUI application."""
    app = QApplication(sys.argv)
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