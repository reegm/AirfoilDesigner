"""Central project configuration constants.

This module gathers default numeric parameters and tunable hyper-parameters
used across the airfoil processing library so they live in one place.
Import these values instead of hard-coding magic numbers inside
algorithms or UI widgets.
"""
from __future__ import annotations

# Debugging & Logging
DEBUG_WORKER_LOGGING: bool = True

# B-spline settings
DEFAULT_BSPLINE_DEGREE: int = 5  # Degree of B-spline curves (3-7 recommended for airfoils)

# ---- Manufacturing / Export defaults -------------------------------------
DEFAULT_CHORD_LENGTH_MM: float = 200.0
DEFAULT_TE_THICKNESS_MM: float = 0.0

# ---- Sampling & Debugging -----------------------------------------------
NUM_POINTS_CURVE_ERROR: int = 35000

# Plot sampling settings
# Curvature-adaptive sampling improves visual smoothness near the leading edge
# while keeping performance reasonable.
PLOT_POINTS_PER_SURFACE: int = 500
PLOT_CURVATURE_WEIGHT: float = 0.85  # 0 = uniform, 1 = fully curvature-driven

# Curvature comb UI ranges
# Old max density (100) becomes the new minimum. Allow much denser combs.
COMB_DENSITY_MIN: int = 100
COMB_DENSITY_MAX: int = 1000
COMB_DENSITY_DEFAULT: int = 200
COMB_SCALE_DEFAULT: float = 0.050


# Number of points used for trailing edge vector calculations
# Higher numbers provide more robust tangent estimates but may be less sensitive to local geometry
DEFAULT_TE_VECTOR_POINTS: int = 2