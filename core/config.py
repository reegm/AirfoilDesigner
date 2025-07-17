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
    "maxiter": 5000,
    "ftol": 1e-12,
} 

# ICP optimizer settings
ICP_OPTIONS = {
    "max_iterations": 10,
    "tol": 1e-9,
} 

# ---- Sampling & Debugging -----------------------------------------------
# Number of sample points used when evaluating Bezier curves for error
# calculations (e.g. ICP error). Higher numbers give more accurate error
# estimates at the cost of performance.
NUM_POINTS_CURVE_ERROR: int = 3000
