"""Read-only log output box used by the application to display messages."""

from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from PySide6.QtGui import QFont
from PySide6.QtCore import QTimer


class StatusLogWidget(QWidget):
    """Simple wrapper around ``QTextEdit`` that defaults to monospaced, read-only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Monospace", 9))

        # Spinner functionality
        self._spinner_timer = QTimer()
        self._spinner_timer.timeout.connect(self._update_spinner)
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0
        self._spinner_active = False
        self._spinner_message = "Processing"

        layout = QVBoxLayout()
        layout.addWidget(self._text_edit)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def append(self, text: str) -> None:  # noqa: D401 (docstring style)
        """Append *text* to the log display."""
        self._text_edit.append(text)

    def clear(self) -> None:
        """Clear the log widget."""
        self._text_edit.clear()

    def widget(self) -> QTextEdit:  # pragma: no cover
        """Return the underlying ``QTextEdit`` instance."""
        return self._text_edit

    # ------------------------------------------------------------------
    # Spinner functionality
    # ------------------------------------------------------------------
    def start_spinner(self, message: str = "Processing") -> None:
        """Start the text-based spinner with the given message."""
        if self._spinner_active:
            return
        
        self._spinner_message = message
        self._spinner_active = True
        self._spinner_index = 0
        self._spinner_timer.start(100)  # Update every 100ms
        self._update_spinner()

    def stop_spinner(self) -> None:
        """Stop the text-based spinner."""
        if not self._spinner_active:
            return
        
        self._spinner_timer.stop()
        self._spinner_active = False
        
        # Remove the spinner line if it exists
        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(cursor.MoveOperation.StartOfLine, cursor.MoveMode.KeepAnchor)
        current_line = cursor.selectedText()
        
        # If the current line contains a spinner frame, remove it
        if any(frame in current_line for frame in self._spinner_frames):
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # Remove the newline

    def _update_spinner(self) -> None:
        """Update the spinner animation."""
        if not self._spinner_active:
            return
        
        # Get the current text
        current_text = self._text_edit.toPlainText()
        lines = current_text.split('\n')
        
        # Check if the last line is a spinner line
        if lines and any(frame in lines[-1] for frame in self._spinner_frames):
            # Replace the last line
            lines[-1] = f"{self._spinner_frames[self._spinner_index]} {self._spinner_message}..."
        else:
            # Add a new spinner line
            lines.append(f"{self._spinner_frames[self._spinner_index]} {self._spinner_message}...")
        
        # Update the text
        self._text_edit.setPlainText('\n'.join(lines))
        
        # Move cursor to end to ensure spinner is visible
        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
        
        # Update spinner frame
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames) 