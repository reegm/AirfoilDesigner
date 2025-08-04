"""Central project configuration constants.

This module gathers default numeric parameters and tunable hyper-parameters
used across the airfoil processing library so they live in one place.
Import these values instead of hard-coding magic numbers inside
algorithms or UI widgets.
"""
from __future__ import annotations

# ---- Optimisation weights -------------------------------------------------
DEFAULT_REGULARIZATION_WEIGHT: float = 0.0

# ---- Geometry & Model -----------------------------------------------------
NUM_CONTROL_POINTS_SINGLE_BEZIER: int = 10  # Order-9 Bezier (per Venkataraman 2017)

# Fixed x-coordinates from the Venkataraman paper
PAPER_FIXED_X_UPPER = [
    0.0,
    0.0,
    0.11422,
    0.25294,
    0.37581,
    0.49671,
    0.61942,
    0.74701,
    0.88058,
    1.0,
]
PAPER_FIXED_X_LOWER = [
    0.0,
    0.0,
    0.12325,
    0.25314,
    0.37519,
    0.49569,
    0.61975,
    0.74391,
    0.87391,
    1.0,
]

# ---- Manufacturing / Export defaults -------------------------------------
DEFAULT_CHORD_LENGTH_MM: float = 200.0
DEFAULT_TE_THICKNESS_MM: float = 0.0

# ---- Optimiser settings ---------------------------------------------------
SLSQP_OPTIONS = {
    "disp": False,
    "maxiter": 10000,
    "ftol": 1e-12,  # Relaxed from 1e-12 to help escape local minima
    "eps": 1e-6    # Added step size control for better gradient estimation
}

# ---- Sampling & Debugging -----------------------------------------------
# Number of sample points used when evaluating Bezier curves for error
# calculations (e.g. euclidedan error). Higher numbers give more accurate error
# estimates at the cost of performance.
NUM_POINTS_CURVE_ERROR: int = 20000


# Number of points used for trailing edge vector calculations
# Higher numbers provide more robust tangent estimates but may be less sensitive to local geometry
DEFAULT_TE_VECTOR_POINTS: int = 3

# Debugging & Logging
DEBUG_WORKER_LOGGING: bool = False

# Plot update control
UPDATE_PLOT: bool = True  # Whether to update the plot during optimization (can be disabled for performance)
PROGRESS_UPDATE_INTERVAL: float = 0.5  # Seconds between progress updates (0.5 = 2 updates per second)

# Orthogonal distance calculation settings
ORTHOGONAL_DISTANCE_MAX_ITERATIONS: int = 20
ORTHOGONAL_DISTANCE_MAX_TOLERANCE: float = 1e-6

# Softmax Settings
SOFTMAX_ALPHA: float = 2000  # Reduced from 100 to be less aggressive about worst errors

# Plateau detection settings
PLATEAU_THRESHOLD: float = 1e-10
PLATEAU_PATIENCE: int = 30

MAX_ERROR_THRESHOLD: float = 9e-5

# ---- Abort mechanism settings -------------------------------------------
# Time interval (seconds) for checking abort flag during optimization
ABORT_CHECK_INTERVAL: float = 0.1

# Maximum time to wait for graceful shutdown after abort request
ABORT_TIMEOUT: float = 5.0