import numpy as np
from scipy.optimize import LinearConstraint
from utils.bezier_utils import general_bezier_curve, bezier_curvature
import logging

class AirfoilModel:
    """
    Manages the geometry, variables, and constraints of the airfoil model.
    The airfoil is defined by four Bezier control polygons:
    S1: Top-front (LE to shared vertex)
    S2: Top-rear (shared vertex to TE)
    S3: Lower-front (LE to shared vertex)
    S4: Lower-rear (shared vertex to TE)
    """
    def __init__(self, initial_upper_shoulder_x=0.4, initial_lower_shoulder_x=0.4):
        """
        Initializes the AirfoilModel with default or provided shoulder x-coordinates.

        Args:
            initial_upper_shoulder_x (float): Fixed x-coordinate for the upper shared vertex.
            initial_lower_shoulder_x (float): Fixed x-coordinate for the lower shared vertex.
        """
        # Initial control points for the four Bezier polygons, set up for a general airfoil shape.
        s1 = [np.array([0.0, 0.0]), np.array([0.0, 0.1]), np.array([0.3, 0.2]), np.array([initial_upper_shoulder_x, 0.12])]
        s2 = [np.array([initial_upper_shoulder_x, 0.12]), np.array([0.6, 0.10]), np.array([0.9, 0.02]), np.array([1.0, 0.0])]
        s3 = [np.array([0.0, 0.0]), np.array([0.0, -0.02]), np.array([0.3, -0.05]), np.array([initial_lower_shoulder_x, -0.06])]
        s4 = [np.array([initial_lower_shoulder_x, -0.06]), np.array([0.6, -0.05]), np.array([0.9, -0.01]), np.array([1.0, 0.0])]
        self.polygons = [s1, s2, s3, s4]
        self.core_processor = None

        # Store initial shared vertex x-coordinates to fix them during optimization.
        self.initial_upper_shoulder_x = initial_upper_shoulder_x
        self.initial_lower_shoulder_x = initial_lower_shoulder_x

        self._variable_map = [] # Maps flat variable array indices to polygon coordinates
        self._update_variable_map() # Initialize the variable map based on current polygon structure
        self._enforce_structure() # Apply initial geometric constraints

    def _update_variable_map(self):
        """
        Creates a map from the flat variable array to the independent polygon point coordinates.
        This map defines which coordinates are treated as optimization variables.
        Dependent points (e.g., shared points, points fixed by tangency constraints) are excluded.
        """
        self._variable_map = []

        for p_idx, poly in enumerate(self.polygons):
            for pt_idx in range(len(poly)):
                for coord_idx in range(2): # 0 for x, 1 for y
                    is_independent = True

                    # Fixed Leading Edge points (first point of S1 and S3) are (0.0, 0.0)
                    if (p_idx == 0 or p_idx == 2) and pt_idx == 0:
                        is_independent = False

                    # Fixed Trailing Edge points (last point of S2 and S4) are (1.0, 0.0) for sharp airfoil
                    elif (p_idx == 1 or p_idx == 3) and pt_idx == len(poly) - 1:
                        is_independent = False

                    # Shared Vertices (start of S2 and S4) are dependent on the end of S1 and S3.
                    elif (p_idx == 1 and pt_idx == 0) or \
                         (p_idx == 3 and pt_idx == 0):
                        is_independent = False

                    # Leading Edge Tangency (B12 and B32 x-coordinates are 0.0)
                    elif (p_idx == 0 or p_idx == 2) and pt_idx == 1 and coord_idx == 0: # x-coordinate
                        is_independent = False

                    # Shared Vertex X-coordinates are fixed to their initial, data-derived positions.
                    elif ((p_idx == 0 and pt_idx == len(poly) - 1) or \
                          (p_idx == 2 and pt_idx == len(poly) - 1)) and coord_idx == 0: # x-coordinate
                        is_independent = False

                    if is_independent:
                        self._variable_map.append({'poly': p_idx, 'pt': pt_idx, 'coord': coord_idx})

        # Sort for consistent variable order.
        self._variable_map.sort(key=lambda m: (m['poly'], m['pt'], m['coord']))

    def get_variables(self):
        """
        Returns a 1D flat array of the model's independent variables.
        Ensures polygon state is consistent before extraction.
        """
        self._enforce_structure()
        return np.array([self.polygons[m['poly']][m['pt']][m['coord']] for m in self._variable_map])

    def _enforce_structure(self):
        """
        Enforces all geometric constraints on the current polygons,
        maintaining the airfoil's structural integrity and tangency conditions.
        """
        # Fixed Leading Edge point at (0,0)
        self.polygons[0][0] = self.polygons[2][0] = np.array([0.0, 0.0])

        # Shared Vertices: S2 starts where S1 ends, S4 starts where S3 ends.
        self.polygons[1][0] = self.polygons[0][-1]
        self.polygons[3][0] = self.polygons[2][-1]

        # Fixed Trailing Edge points at (1,0) for the sharp airfoil.
        self.polygons[1][-1] = np.array([1.0, 0.0])
        self.polygons[3][-1] = np.array([1.0, 0.0])

        # Shared Vertex Tangency Constraints (Horizontal):
        # Control points around the shared vertex lie on a horizontal line.
        if len(self.polygons[0]) >= 4 and len(self.polygons[1]) >= 2:
            shared_vertex_y = self.polygons[0][-1][1] # y-coord of shared vertex (B14/B21)
            self.polygons[0][-2][1] = shared_vertex_y # B13's y-coord
            self.polygons[1][1][1] = shared_vertex_y   # B22's y-coord

        if len(self.polygons[2]) >= 4 and len(self.polygons[3]) >= 2:
            shared_vertex_y = self.polygons[2][-1][1] # y-coord of shared vertex (B34/B41)
            self.polygons[2][-2][1] = shared_vertex_y # B33's y-coord
            self.polygons[3][1][1] = shared_vertex_y   # B42's y-coord

        # Leading Edge Tangency Constraints (Vertical):
        # The second control point's x-coordinate for S1 (B12) and S3 (B32) is 0.0.
        if len(self.polygons[0]) >= 2:
            self.polygons[0][1][0] = 0.0
        if len(self.polygons[2]) >= 2:
            self.polygons[2][1][0] = 0.0

        # Fixed Shared Vertex X-coordinates:
        # Ensure the x-coordinates of the shared vertices (end of S1 and S3) remain fixed.
        if len(self.polygons[0]) >= 4:
            self.polygons[0][-1][0] = self.initial_upper_shoulder_x
        if len(self.polygons[2]) >= 4:
            self.polygons[2][-1][0] = self.initial_lower_shoulder_x

    def _get_variable_indices_map(self):
        """Helper to create a quick lookup for variable indices."""
        return { (m['poly'], m['pt'], m['coord']): i for i, m in enumerate(self._variable_map) }

    def _get_x_ordering_constraints(self, num_vars, var_indices):
        """Generates constraints to ensure x-coordinates are monotonically increasing."""
        constraints = []
        for p_idx, poly in enumerate(self.polygons):
            for pt_idx in range(len(poly) - 1):
                current_x_idx = var_indices.get((p_idx, pt_idx, 0))
                next_x_idx = var_indices.get((p_idx, pt_idx + 1, 0))

                if current_x_idx is not None and next_x_idx is not None:
                    A = np.zeros(num_vars)
                    A[current_x_idx] = -1
                    A[next_x_idx] = 1
                    constraints.append(LinearConstraint(A, 0, np.inf))
                elif current_x_idx is None and next_x_idx is not None:
                    current_x_val = poly[pt_idx][0]
                    A = np.zeros(num_vars)
                    A[next_x_idx] = 1
                    constraints.append(LinearConstraint(A, current_x_val, np.inf))
                elif current_x_idx is not None and next_x_idx is None:
                    next_x_val = poly[pt_idx + 1][0]
                    A = np.zeros(num_vars)
                    A[current_x_idx] = 1
                    constraints.append(LinearConstraint(A, -np.inf, next_x_val))
        return constraints

    def _get_le_tangency_constraints(self, num_vars, var_indices):
        """Generates constraints for leading edge tangency (x-coord of B12 and B32 is 0)."""
        constraints = []
        idx_s1_p1_x = var_indices.get((0, 1, 0))
        if idx_s1_p1_x is not None:
            A = np.zeros(num_vars); A[idx_s1_p1_x] = 1
            constraints.append(LinearConstraint(A, 0.0, 0.0))

        idx_s3_p1_x = var_indices.get((2, 1, 0))
        if idx_s3_p1_x is not None:
            A = np.zeros(num_vars); A[idx_s3_p1_x] = 1
            constraints.append(LinearConstraint(A, 0.0, 0.0))
        return constraints

    def _get_te_tangency_constraints(self, num_vars, var_indices, upper_te_tangent_vector, lower_te_tangent_vector):
        """
        Generates constraints for trailing edge tangency for the 4-segment model.
        The segment connecting the last two control points (B_N-1 to B_N) is constrained to be parallel
        to the provided tangent vectors.
        """
        constraints = []

        def add_tangency_constraint(poly_idx, te_tangent_vector):
            if te_tangent_vector is None:
                return

            tx, ty = te_tangent_vector[0], te_tangent_vector[1]

            # Ensure B_N-1 and B_N exist (last two points of the polygon)
            if len(self.polygons[poly_idx]) >= 2:
                p_n = self.polygons[poly_idx][-1] # B_N (fixed TE point)
                idx_p_n_minus_1_x = var_indices.get((poly_idx, len(self.polygons[poly_idx]) - 2, 0)) # B_N-1.x
                idx_p_n_minus_1_y = var_indices.get((poly_idx, len(self.polygons[poly_idx]) - 2, 1)) # B_N-1.y

                # Constraint: (P_N.x - P_N-1.x) * ty - (P_N.y - P_N-1.y) * tx = 0
                # Rearranged: -P_N-1.x * ty + P_N-1.y * tx = -P_N.x * ty + P_N.y * tx
                # A * variables = b
                # A = [-ty, tx] for [P_N-1.x, P_N-1.y]
                # b = -P_N.x * ty + P_N.y * tx

                A = np.zeros(num_vars)
                b = -p_n[0] * ty + p_n[1] * tx

                # Adjust 'b' if P_N-1.x or P_N-1.y are fixed (not optimization variables)
                if idx_p_n_minus_1_x is not None and idx_p_n_minus_1_y is not None:
                    A[idx_p_n_minus_1_x] = -ty
                    A[idx_p_n_minus_1_y] = tx
                    constraints.append(LinearConstraint(A, b, b))
                elif idx_p_n_minus_1_x is None and idx_p_n_minus_1_y is not None: # P_N-1.x is fixed
                    fixed_x_val = self.polygons[poly_idx][-2][0]
                    A[idx_p_n_minus_1_y] = tx
                    b -= (-fixed_x_val * ty)
                    constraints.append(LinearConstraint(A, b, b))
                elif idx_p_n_minus_1_x is not None and idx_p_n_minus_1_y is None: # P_N-1.y is fixed
                    fixed_y_val = self.polygons[poly_idx][-2][1]
                    A[idx_p_n_minus_1_x] = -ty
                    b -= (fixed_y_val * tx)
                # If both are fixed, no constraint is needed as their positions are determined.

        add_tangency_constraint(1, upper_te_tangent_vector) # S2 (Upper Rear)
        add_tangency_constraint(3, lower_te_tangent_vector) # S4 (Lower Rear)

        return constraints

    def _get_y_sign_constraints(self, num_vars, var_indices):
        """Generates constraints to ensure upper surface y >= 0 and lower surface y <= 0."""
        constraints = []
        # Upper surface (S1 and S2)
        for p_idx in [0, 1]:
            # Iterate through points, excluding fixed LE/TE and shared vertices handled by other constraints.
            start_pt = 1 # Start from B12 or B22
            end_pt = len(self.polygons[p_idx]) - 1 # End before B14 or B24

            # Adjust end_pt for S2 to avoid re-constraining B23's y-coordinate if it's already handled by TE tangency.
            if p_idx == 1 and len(self.polygons[1]) >= 3 and var_indices.get((1, len(self.polygons[1]) - 2, 1)) is not None:
                end_pt = len(self.polygons[p_idx]) - 2 # Exclude B23 from general y>=0 constraint

            for pt_idx in range(start_pt, end_pt):
                idx_y = var_indices.get((p_idx, pt_idx, 1)) # Get index of y-coordinate
                if idx_y is not None:
                    A = np.zeros(num_vars); A[idx_y] = 1
                    constraints.append(LinearConstraint(A, 0.0, np.inf)) # y >= 0

        # Lower surface (S3 and S4)
        for p_idx in [2, 3]:
            # Iterate through points, excluding fixed LE/TE and shared vertices handled by other constraints.
            start_pt = 1 # Start from B32 or B42
            end_pt = len(self.polygons[p_idx]) - 1 # End before B34 or B44

            # Adjust end_pt for S4 to avoid re-constraining B43's y-coordinate if it's already handled by TE tangency.
            if p_idx == 3 and len(self.polygons[3]) >= 3 and var_indices.get((3, len(self.polygons[3]) - 2, 1)) is not None:
                end_pt = len(self.polygons[p_idx]) - 2 # Exclude B43 from general y<=0 constraint

            for pt_idx in range(start_pt, end_pt):
                idx_y = var_indices.get((p_idx, pt_idx, 1)) # Get index of y-coordinate
                if idx_y is not None:
                    A = np.zeros(num_vars); A[idx_y] = 1
                    constraints.append(LinearConstraint(A, -np.inf, 0.0)) # y <= 0
        return constraints

    def get_constraints(self, upper_te_tangent_vector=None, lower_te_tangent_vector=None):
        """
        Generates all linear constraint objects for the optimizer, ensuring geometric validity.

        Args:
            upper_te_tangent_vector (np.array, optional): Desired upper trailing edge tangent vector.
            lower_te_tangent_vector (np.array, optional): Desired lower trailing edge tangent vector.
        """
        num_vars = len(self._variable_map)
        var_indices = self._get_variable_indices_map()
        all_constraints = []

        all_constraints.extend(self._get_x_ordering_constraints(num_vars, var_indices))
        all_constraints.extend(self._get_le_tangency_constraints(num_vars, var_indices))
        all_constraints.extend(self._get_te_tangency_constraints(num_vars, var_indices, upper_te_tangent_vector, lower_te_tangent_vector))
        all_constraints.extend(self._get_y_sign_constraints(num_vars, var_indices))

        return all_constraints

    def set_variables(self, variables):
        """
        Updates the model's geometry from the optimizer's flat variable array.
        After setting, `_enforce_structure` is called to apply all geometric constraints.
        """
        for i, var in enumerate(variables):
            m = self._variable_map[i]
            self.polygons[m['poly']][m['pt']][m['coord']] = var
        self._enforce_structure() # Re-apply fixed constraints after setting variables

    def add_point_to_segment(self, segment_index):
        """
        Adds a control point to the specified segment.
        The new point is placed by identifying the region of highest curvature on the Bezier curve
        and then splitting the closest control polygon edge.
        """
        poly = self.polygons[segment_index]

        if len(poly) < 2:
            return # Cannot add a point to a line segment or single point (need at least one edge)

        # Generate points along the Bezier curve for curvature evaluation
        t_values = np.linspace(0, 1, 200)
        curvatures = bezier_curvature(t_values, np.array(poly))

        # Find the t-value and corresponding point on the curve where curvature is maximum
        max_curvature_t_idx = np.argmax(curvatures)
        t_at_max_curvature = t_values[max_curvature_t_idx]
        point_on_curve_at_max_curvature = general_bezier_curve(t_at_max_curvature, np.array(poly))

        # Find which control polygon edge is closest to this high-curvature point on the curve.
        min_dist = np.inf
        split_idx = -1 # Index of the control point that starts the edge to split

        # Iterate through all edges of the control polygon
        for i in range(len(poly) - 1):
            p1 = poly[i]
            p2 = poly[i+1]

            segment_vector = p2 - p1
            segment_length_sq = np.dot(segment_vector, segment_vector)

            if segment_length_sq == 0: # Degenerate edge
                dist = np.linalg.norm(point_on_curve_at_max_curvature - p1)
            else:
                point_to_p1_vector = point_on_curve_at_max_curvature - p1
                proj_t = np.dot(point_to_p1_vector, segment_vector) / segment_length_sq
                proj_t_clamped = np.clip(proj_t, 0, 1) # Clamp to segment
                closest_point_on_segment = p1 + proj_t_clamped * segment_vector
                dist = np.linalg.norm(point_on_curve_at_max_curvature - closest_point_on_segment)

            if dist < min_dist:
                min_dist = dist
                split_idx = i

        # Fallback: If no suitable edge found, split the longest edge for robustness.
        if split_idx == -1:
            logging.warning(
                "No suitable edge found for curvature-based split in segment %d. Falling back to longest edge split.",
                segment_index,
            )
            edge_lengths = np.linalg.norm(np.diff(np.array(poly), axis=0), axis=1)
            split_idx = np.argmax(edge_lengths)

        # Calculate the new control point as the midpoint of the identified edge
        new_point = (poly[split_idx] + poly[split_idx + 1]) / 2.0

        # Insert the new point into the polygon list
        new_poly_list = poly.copy()
        new_poly_list.insert(split_idx + 1, new_point)

        self.polygons[segment_index] = new_poly_list # Update the segment's polygon
        self._update_variable_map() # Re-generate the variable map as polygon structure has changed
