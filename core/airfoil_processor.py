import numpy as np
import copy
from PySide6.QtCore import QObject, Signal
import logging

from core import config
from utils.data_loader import load_airfoil_data



class SignalLogHandler(logging.Handler):
    """A logging handler that emits a Qt signal."""
    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record):
        msg = self.format(record)
        self.signal_emitter.emit(msg)

class AirfoilProcessor(QObject):
    """
    Acts as a bridge between the GUI and the CoreProcessor.
    It holds an instance of the CoreProcessor and uses Qt Signals
    to communicate with the GUI.
    """
    log_message = Signal(str)
    plot_update_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Core airfoil data
        self.upper_data = None
        self.lower_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._last_plot_data = None # Cache for the last plot data dictionary
        self._current_te_vector_points = None  # Store current TE vector points setting
        self._is_blunt_TE = False # True if original airfoil has thickened TE


    def load_airfoil_data_and_initialize_model(self, file_path):
        """
        Loads airfoil data and initializes the AirfoilModel.
        Resets internal flags and state.
        """
        self._last_plot_data = None
        self.upper_data = None
        self.lower_data = None
        self.upper_te_tangent_vector = None
        self.lower_te_tangent_vector = None
        self._is_blunt_TE = False

        try:
            upper, lower, airfoil_name, blunt_te = load_airfoil_data(file_path, logger_func=self.log_message.emit)
            self.upper_data = upper
            self.lower_data = lower
            self.airfoil_name = airfoil_name
            self._is_blunt_TE = blunt_te
            # Recalculate TE tangent vectors using configured default
            te_vector_points = config.DEFAULT_TE_VECTOR_POINTS
            self.upper_te_tangent_vector, self.lower_te_tangent_vector = self._calculate_te_tangent(
                self.upper_data, self.lower_data, te_vector_points)
            self.log_message.emit("Airfoil data loaded.")
            self._request_plot_update()
            return True
        except Exception as e:
            self.log_message.emit(f"Failed to load or initialize airfoil data: {e}")
            return False

    def is_trailing_edge_thickened(self):
        """Returns True if the loaded airfoil has a thickened trailing edge."""
        return self._is_blunt_TE

    def _request_plot_update(self):
        """Emits a signal to request a plot update with current model data from the core."""
        if self.upper_data is None or self.lower_data is None:
            self.log_message.emit("No airfoil data available to plot.")
            return

        # Use stored TE tangent vectors from core_processor
        upper_te_tangent_vector = self.upper_te_tangent_vector
        lower_te_tangent_vector = self.lower_te_tangent_vector

        plot_data = {
            'upper_data': self.upper_data,
            'lower_data': self.lower_data,
            'upper_te_tangent_vector': upper_te_tangent_vector,
            'lower_te_tangent_vector': lower_te_tangent_vector,
            'geometry_metrics': None,
        }

        # Geometry metrics calculation removed - was Bezier-specific
        plot_data['geometry_metrics'] = None

        self._last_plot_data = plot_data.copy()
        self.plot_update_requested.emit(plot_data)

    def recalculate_te_vectors_and_update_plot(self, te_vector_points):
        """
        Recalculate the trailing edge tangent vectors using the specified number of points
        and update the plot.
        """
        if self.upper_data is None or self.lower_data is None:
            self.log_message.emit("Error: No airfoil data loaded. Cannot recalculate TE vectors.")
            return
        
        upper_te_tangent_vector, lower_te_tangent_vector = self._calculate_te_tangent(
            self.upper_data, self.lower_data, te_vector_points
        )
        # Update the stored TE tangent vectors
        self.upper_te_tangent_vector = upper_te_tangent_vector
        self.lower_te_tangent_vector = lower_te_tangent_vector
        
        # Create plot data that preserves existing model data while updating TE vectors
        plot_data = {
            'upper_data': self.upper_data,
            'lower_data': self.lower_data,
            'upper_te_tangent_vector': upper_te_tangent_vector,
            'lower_te_tangent_vector': lower_te_tangent_vector,
            'geometry_metrics': None,
        }
        
        plot_data['geometry_metrics'] = None
        
        self.plot_update_requested.emit(plot_data)
        self.log_message.emit(f"Trailing edge vectors recalculated with {te_vector_points} points.")

    def _calculate_te_tangent(self, upper_data, lower_data, te_vector_points):
        """
        Calculate trailing edge tangent vectors for upper and lower surfaces using the last N points.
        Returns (upper_te_tangent_vector, lower_te_tangent_vector)
        """
        def tangent(data, n):
            # Use the last n points to estimate the tangent at the trailing edge
            if n < 2 or len(data) < n:
                n = min(3, len(data))
            pts = data[-n:]
            dx = pts[-1, 0] - pts[0, 0]
            dy = pts[-1, 1] - pts[0, 1]
            norm = np.hypot(dx, dy)
            if norm == 0:
                return np.array([1.0, 0.0])
            return np.array([dx, dy]) / norm
        upper_te_tangent = tangent(upper_data, te_vector_points)
        lower_te_tangent = tangent(lower_data, te_vector_points)
        return upper_te_tangent, lower_te_tangent
