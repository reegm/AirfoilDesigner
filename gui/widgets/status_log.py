"""Read-only log output box used by the application to display messages."""

from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from PySide6.QtGui import QFont


class StatusLogWidget(QWidget):
    """Simple wrapper around ``QTextEdit`` that defaults to monospaced, read-only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Monospace", 9))

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