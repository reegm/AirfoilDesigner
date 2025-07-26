import numpy as np
from scipy.special import comb
from joblib import Parallel, delayed

from core import config

# Cache for binomial coefficients to avoid repeated calculations
_binomial_cache = {}

def get_binomial_coeff(n, k):
    """Get binomial coefficient with caching for performance."""
    key = (n, k)
    if key not in _binomial_cache:
        _binomial_cache[key] = comb(n, k, exact=True)
    return _binomial_cache[key]

# Pre-compute common binomial coefficients for typical Bezier orders
def _precompute_binomials():
    """Pre-compute binomial coefficients for common Bezier orders."""
    for n in range(1, 12):  # Support up to order 11
        for k in range(n + 1):
            get_binomial_coeff(n, k)

_precompute_binomials()

class BezierEvaluator:
    """Cached Bezier curve evaluator for performance optimization."""
    
    def __init__(self, control_points):
        self.control_points = np.asarray(control_points)
        self.n = len(control_points) - 1
        self._bernstein_cache = {}
        self._derivative_cache = {}
        
    def _get_bernstein_coeffs(self, t):
        """Get Bernstein coefficients for given t with caching."""
        if t not in self._bernstein_cache:
            coeffs = np.zeros(self.n + 1)
            for i in range(self.n + 1):
                coeffs[i] = get_binomial_coeff(self.n, i) * (1 - t)**(self.n - i) * t**i
            self._bernstein_cache[t] = coeffs
        return self._bernstein_cache[t]
    
    def _get_derivative_coeffs(self, t):
        """Get derivative coefficients for given t with caching."""
        if t not in self._derivative_cache:
            coeffs = np.zeros(self.n)
            for i in range(self.n):
                coeffs[i] = get_binomial_coeff(self.n - 1, i) * (1 - t)**(self.n - 1 - i) * t**i
            self._derivative_cache[t] = coeffs
        return self._derivative_cache[t]
    
    def evaluate(self, t):
        """Evaluate Bezier curve at parameter t."""
        coeffs = self._get_bernstein_coeffs(t)
        return np.sum(coeffs[:, None] * self.control_points, axis=0)
    
    def evaluate_derivative(self, t):
        """Evaluate first derivative at parameter t."""
        coeffs = self._get_derivative_coeffs(t)
        diff_points = np.diff(self.control_points, axis=0)
        return self.n * np.sum(coeffs[:, None] * diff_points, axis=0)
    
    def evaluate_second_derivative(self, t):
        """Evaluate second derivative at parameter t."""
        if self.n < 2:
            return np.zeros(2)
        
        coeffs = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            coeffs[i] = get_binomial_coeff(self.n - 2, i) * (1 - t)**(self.n - 2 - i) * t**i
        
        diff2_points = np.diff(self.control_points, n=2, axis=0)
        return self.n * (self.n - 1) * np.sum(coeffs[:, None] * diff2_points, axis=0)

def calculate_orthogonal_distance_to_bezier_optimized(point, control_points, initial_t_guess=None, max_iterations=config.ORTHOGONAL_DISTANCE_MAX_ITERATIONS, tolerance=config.ORTHOGONAL_DISTANCE_MAX_TOLERANCE):
    """
    Optimized version of orthogonal distance calculation with caching and better convergence.
    """
    evaluator = BezierEvaluator(control_points)
    point = np.asarray(point)
    
    def try_newton_raphson(t_init):
        """Optimized Newton-Raphson with early termination."""
        t = np.clip(t_init, 0, 1)
        
        for iteration in range(max_iterations):
            curve_point = evaluator.evaluate(t)
            first_derivative = evaluator.evaluate_derivative(t)
            second_derivative = evaluator.evaluate_second_derivative(t)
            
            # Vector from curve point to target point
            diff_vector = point - curve_point
            
            # Function: f(t) = (P - B(t)) · B'(t) = 0 for orthogonal distance
            f = np.dot(diff_vector, first_derivative)
            
            # Derivative: f'(t) = -B'(t) · B'(t) + (P - B(t)) · B''(t)
            f_prime = -np.dot(first_derivative, first_derivative) + np.dot(diff_vector, second_derivative)
            
            # Avoid division by zero
            if abs(f_prime) < 1e-12:
                break
            
            # Newton step with adaptive damping
            step = f / f_prime
            damping = min(1.0, 0.5 / max(abs(step), 1e-6))
            
            t_new = t - damping * step
            t_new = np.clip(t_new, 0, 1)
            
            # Check convergence
            if abs(t_new - t) < tolerance:
                t = t_new
                break
            
            t = t_new
        
        # Evaluate final distance
        curve_point = evaluator.evaluate(t)
        distance = np.linalg.norm(point - curve_point)
        return distance, t, curve_point
    
    # Smart initial guess based on x-coordinate
    if initial_t_guess is not None:
        candidates = [initial_t_guess]
    else:
        x_min, x_max = control_points[0, 0], control_points[-1, 0]
        if x_max > x_min:
            x_based_guess = np.clip((point[0] - x_min) / (x_max - x_min), 0, 1)
            candidates = [x_based_guess]
        else:
            candidates = [0.5]
    
    # Reduced number of candidates for better performance
    candidates.extend([0.0, 0.25, 0.5, 0.75, 1.0])
    candidates = list(set(candidates))  # Remove duplicates
    
    # Try each candidate and keep the best result
    best_distance = float('inf')
    best_t = 0.5
    best_curve_point = None
    
    for t_init in candidates:
        try:
            dist, t_opt, curve_pt = try_newton_raphson(t_init)
            if dist < best_distance:
                best_distance = dist
                best_t = t_opt
                best_curve_point = curve_pt
        except:
            continue
    # Fallback: coarse sampling if Newton-Raphson fails
    if best_distance == float('inf') or best_distance > 1.0:
        print("Newton-Raphson failed. Falling back to coarse sampling.")
        t_samples = np.linspace(0, 1, 100)
        distances = []
        for t_sample in t_samples:
            curve_pt = evaluator.evaluate(t_sample)
            dist = np.linalg.norm(point - curve_pt)
            distances.append(dist)
        
        min_idx = np.argmin(distances)
        best_distance = distances[min_idx]
        best_t = t_samples[min_idx]
        best_curve_point = evaluator.evaluate(best_t)
    
    return best_distance, best_t, best_curve_point

def calculate_all_orthogonal_distances_optimized(data_points, control_points):
    """
    Optimized version that calculates all orthogonal distances with better initial guesses.
    """
    evaluator = BezierEvaluator(control_points)
    data_points = np.asarray(data_points)
    n_points = len(data_points)
    
    distances = np.zeros(n_points)
    t_values = np.zeros(n_points)
    curve_points = np.zeros((n_points, 2))
    
    # Pre-compute x-range for better initial guesses
    x_min, x_max = control_points[0, 0], control_points[-1, 0]
    x_range = x_max - x_min
    

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(calculate_orthogonal_distance_to_bezier_optimized)(
            point,
            control_points,
            initial_t_guess=np.clip((point[0] - x_min) / x_range, 0, 1) if x_range > 0 else i / max(len(data_points) - 1, 1)
        )
        for i, point in enumerate(data_points)
    )

    # Unpack
    distances, t_values, curve_points = zip(*results)
    distances = np.array(distances)
    t_values = np.array(t_values)
    curve_points = np.vstack(curve_points)
    
    max_distance = np.max(distances)
    max_distance_idx = np.argmax(distances)
    
    return distances, max_distance, max_distance_idx, t_values, curve_points 