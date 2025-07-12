import numpy as np

def normalize_airfoil_data(upper_surface, lower_surface, logger_func=print):
    """
    Normalizes airfoil coordinates to have a chord length of 1, with the
    leading edge at (0,0) and the trailing edge chord-line at (1,0).

    This process preserves the shape of the airfoil by performing translation,
    rotation, and scaling.

    Args:
        upper_surface (np.ndarray): Original upper surface coordinates (N, 2), ordered LE to TE.
        lower_surface (np.ndarray): Original lower surface coordinates (N, 2), ordered LE to TE.
        logger_func (callable, optional): A function for logging messages.

    Returns:
        tuple: (normalized_upper_surface, normalized_lower_surface)
    """
    # The data is assumed to be ordered LE to TE for both surfaces.
    le_point = upper_surface[0].copy()
    te_upper = upper_surface[-1]
    te_lower = lower_surface[-1]
    te_point = (te_upper + te_lower) / 2.0
    
    logger_func(f"Normalizing airfoil. Original LE: {le_point}, Original TE (midpoint): {te_point}")

    # Translate so LE is at the origin (0,0)
    upper_translated = upper_surface - le_point
    lower_translated = lower_surface - le_point
    te_translated = te_point - le_point

    # Rotate so the TE point lies on the positive x-axis.
    if np.allclose(te_translated, [0.0, 0.0]):
        logger_func("Warning: Leading and trailing edges are coincident. Cannot determine rotation.")
        rotation_angle = 0.0
    else:
        rotation_angle = -np.arctan2(te_translated[1], te_translated[0])

    if abs(rotation_angle) > 0:
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle),  np.cos(rotation_angle)]
        ])
        upper_rotated = upper_translated @ rotation_matrix.T
        lower_rotated = lower_translated @ rotation_matrix.T
        te_rotated = te_translated @ rotation_matrix.T
    else:
        upper_rotated = upper_translated
        lower_rotated = lower_translated
        te_rotated = te_translated
    
    chord_length = te_rotated[0]

    if np.isclose(chord_length, 0):
        raise ValueError("Cannot normalize airfoil with zero chord length.")
    
    logger_func(f"Detected chord length: {chord_length:.6f}, rotating by {np.rad2deg(rotation_angle):.4f} degrees.")

    upper_normalized = upper_rotated / chord_length
    lower_normalized = lower_rotated / chord_length
    
    return upper_normalized, lower_normalized

def load_airfoil_data(filename, logger_func=print):
    """
    Loads airfoil coordinates from a file, supporting Selig and Lednicer .dat formats.

    Selig format:
        Line 1: Airfoil name
        Subsequent lines: x y coordinates, usually starting from upper TE, around LE, to lower TE.

    Lednicer format:
        Line 1: Airfoil name
        Line 2: NUM_UPPER NUM_LOWER (number of points on upper and lower surfaces)
        Subsequent lines: Upper surface points (x y), typically LE to TE.
        Then: Lower surface points (x y), typically LE to TE.

    Args:
        filename (str): Path to the airfoil data file.
        logger_func (callable, optional): A function (like print or a signal's emit method) to send log messages to.
                                          Defaults to `print`.

    Returns:
        tuple: (upper_surface_coords, lower_surface_coords)
               Each is a NumPy array (N, 2) with points ordered LE to TE.
    Raises:
        ValueError: If the file is empty or malformed.
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()] # Read lines, strip whitespace, remove empty lines

    if not lines:
        raise ValueError(f"Airfoil data file '{filename}' is empty or contains no valid data.")

    airfoil_name = lines[0]
    coords_start_line = 1

    is_lednicer = False
    num_upper_points = 0
    num_lower_points = 0

    # Attempt to detect Lednicer format
    if len(lines) > 1:
        try:
            parts = lines[1].split()
            if len(parts) == 2:
                num_upper_points = int(parts[0])
                num_lower_points = int(parts[1])
                is_lednicer = True
                coords_start_line = 2
        except ValueError:
            pass # Not integers, so it's likely a Selig-like format

    if is_lednicer:
        logger_func(f"Detected Lednicer format for '{airfoil_name}'.")
        logger_func(f"Expected upper points: {num_upper_points}, lower points: {num_lower_points}.")

        all_coords_raw = np.array([list(map(float, line.split())) for line in lines[coords_start_line:]])

        if len(all_coords_raw) != (num_upper_points + num_lower_points):
            logger_func(f"Warning: Declared points ({num_upper_points + num_lower_points}) do not match actual points read ({len(all_coords_raw)}). Attempting to proceed.")

        # Lednicer typically lists upper surface points first, then lower, both LE to TE.
        upper_surface = all_coords_raw[:num_upper_points]
        lower_surface = all_coords_raw[num_upper_points:]

        # Basic validation: check if LE point is consistent
        if not (np.allclose(upper_surface[0], lower_surface[0]) and np.allclose(upper_surface[0], [0.0, 0.0])):
             logger_func("Warning: Leading edge points are not consistent or not at (0,0) in Lednicer format. Data may need normalization.")

    else: # Assume Selig-like format
        logger_func(f"Detected Selig-like format for '{airfoil_name}'.")
        coords_str = lines[coords_start_line:]

        all_coords = np.array([list(map(float, line.split())) for line in coords_str if line])

        # Find the index of the leading edge (x=0.0)
        le_index = -1
        for i, x in enumerate(all_coords[:, 0]):
            if abs(x - 0.0) < 1e-9:
                le_index = i
                break

        if le_index == -1:
            raise ValueError(f"Leading edge (x=0.0) not found in Selig-like data for '{filename}'. Ensure data is normalized.")

        # Selig-like data usually lists points from upper TE, around LE, to lower TE.
        upper_surface_raw = all_coords[:le_index + 1] # Upper surface (TE to LE order)
        lower_surface = all_coords[le_index:] # Lower surface (LE to TE order)

        # Flip the upper surface to be ordered from LE to TE
        upper_surface = np.flipud(upper_surface_raw)

    # Always normalize the data to ensure consistency.
    upper_surface, lower_surface = normalize_airfoil_data(upper_surface, lower_surface, logger_func)

    return upper_surface, lower_surface, airfoil_name

def find_shoulder_x_coords(upper_data, lower_data):
    """
    Identifies the x-coordinates for the upper and lower shared vertices (shoulder points)
    based on maximum/minimum y-values in the original airfoil data.

    Args:
        upper_data (np.ndarray): Coordinates of the upper surface (N, 2).
        lower_data (np.ndarray): Coordinates of the lower surface (N, 2).

    Returns:
        tuple: (upper_shoulder_x, lower_shoulder_x)
    """
    # Find the x-coordinate corresponding to the maximum y-value on the upper surface
    upper_max_y_idx = np.argmax(upper_data[:, 1])
    upper_shoulder_x = upper_data[upper_max_y_idx, 0]

    # Find the x-coordinate corresponding to the minimum y-value on the lower surface
    lower_min_y_idx = np.argmin(lower_data[:, 1])
    lower_shoulder_x = lower_data[lower_min_y_idx, 0]

    return upper_shoulder_x, lower_shoulder_x
