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
    "ftol": 1e-12,  # Function tolerance for convergence
    "gtol": 1e-8,   # Gradient tolerance for early termination
    "eps": 1e-6     # Step size for gradient estimation
}

# ---- Sampling & Debugging -----------------------------------------------
# Adaptive sampling for error calculations
# Start with coarse sampling, increase resolution as error improves
NUM_POINTS_CURVE_ERROR_COARSE: int = 5000    # Initial optimization phases
NUM_POINTS_CURVE_ERROR_MEDIUM: int = 12000   # Intermediate refinement
NUM_POINTS_CURVE_ERROR_FINE: int = 20000     # Final precision
NUM_POINTS_CURVE_ERROR_ULTRA: int = 35000    # Ultimate precision (rarely used)

# Adaptive sampling thresholds - when to increase resolution
ADAPTIVE_SAMPLING_THRESHOLD_MEDIUM: float = 5e-4  # Switch from coarse to medium
ADAPTIVE_SAMPLING_THRESHOLD_FINE: float = 1e-4    # Switch from medium to fine
ADAPTIVE_SAMPLING_THRESHOLD_ULTRA: float = 7e-5   # Switch from fine to ultra

# Legacy setting for backward compatibility
NUM_POINTS_CURVE_ERROR: int = NUM_POINTS_CURVE_ERROR_FINE


# Plot sampling settings
# Curvature-adaptive sampling improves visual smoothness near the leading edge
# while keeping performance reasonable.
PLOT_POINTS_PER_SURFACE: int = 250
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

# Debugging & Logging
DEBUG_WORKER_LOGGING: bool = True

# Plot update control
UPDATE_PLOT: bool = True  # Whether to update the plot during optimization (can be disabled for performance)
PROGRESS_UPDATE_INTERVAL: float = 1  # Seconds between progress updates (0.5 = 2 updates per second)

# Orthogonal distance calculation settings
ORTHOGONAL_DISTANCE_MAX_ITERATIONS: int = 20
ORTHOGONAL_DISTANCE_MAX_TOLERANCE: float = 1e-12

# Softmax Settings
SOFTMAX_ALPHA: float = 2000  # Reduced from 100 to be less aggressive about worst errors

# Plateau detection settings
PLATEAU_THRESHOLD: float = 1e-10
PLATEAU_PATIENCE: int = 30

MAX_ERROR_THRESHOLD: float = 5e-5  # Ultimate target

# ---- Abort mechanism settings -------------------------------------------
# Time interval (seconds) for checking abort flag during optimization
ABORT_CHECK_INTERVAL: float = 0.1

# Maximum time to wait for graceful shutdown after abort request
ABORT_TIMEOUT: float = 5.0

# ---- Staged optimizer settings -------------------------------------------
# Basin-hopping hop counts per stage
HYBRID_BH_HOPS_MSR: int = 5                 # Enable MSR basin-hopping for priming Stage 2
HYBRID_BH_HOPS_FREE_MINMAX: int = 15        # more aggressive in free-x where gains are largest


# Perturbation scale (normalized coordinates)
HYBRID_BH_PERTURB_STD: float = 0.005

# Per-stage local optimizer iteration budgets
HYBRID_LOCAL_MAXITER_MSR: int = 400
HYBRID_LOCAL_MAXITER_MINMAX_FREE: int = 1500