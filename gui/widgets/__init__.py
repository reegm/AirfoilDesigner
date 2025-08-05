from __future__ import annotations

# Re-export widget classes for convenience so callers can do:
#   from gui.widgets import AirfoilPlotWidget, FileControlPanel, OptimizerSettingsWidget, AirfoilSettingsWidget, CombPanelWidget, StatusLogWidget
#
# Individual modules define each widget class. Importing them here avoids deep import paths outside this package.

from .airfoil_plot_widget import AirfoilPlotWidget  # noqa: F401
from .file_control_panel import FileControlPanel  # noqa: F401
from .optimizer_settings import OptimizerSettingsWidget  # noqa: F401
from .airfoil_settings import AirfoilSettingsWidget  # noqa: F401
from .comb_panel import CombPanelWidget  # noqa: F401
from .status_log import StatusLogWidget  # noqa: F401
from .cst_settings import CSTSettingsWidget  # noqa: F401 