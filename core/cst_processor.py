"""
CST Processor for integrating CST fitting with the existing airfoil processing pipeline.

This module provides CST fitting as an intermediary step before Bezier optimization.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from PySide6.QtCore import QObject, Signal

from core.cst_fitter import CSTFitter, fit_airfoil_cst, generate_cst_airfoil_data


class CSTProcessor(QObject):
    """
    Processor for CST fitting that integrates with the existing airfoil processing pipeline.
    
    This acts as an intermediary step between raw airfoil data and Bezier optimization.
    """
    
    # Signals for GUI communication
    log_message = Signal(str)
    plot_update_requested = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # CST fitting parameters
        self.degree = 15
        self.n1 = 0.5
        self.n2 = 1.0
        
        # Fitting results
        self.cst_fitter = None
        self.upper_coefficients = None
        self.lower_coefficients = None
        self.upper_metrics = None
        self.lower_metrics = None
        
        # Generated CST data
        self.cst_upper_data = None
        self.cst_lower_data = None
        
        # Original data (for comparison)
        self.original_upper_data = None
        self.original_lower_data = None
        
    def set_parameters(self, degree: int = 8, n1: float = 0.5, n2: float = 1.0):
        """
        Set CST fitting parameters.
        
        Args:
            degree: Degree of Bernstein polynomials
            n1: Class function parameter for leading edge
            n2: Class function parameter for trailing edge
        """
        self.degree = degree
        self.n1 = n1
        self.n2 = n2
        
    def fit_airfoil(self, upper_data: np.ndarray, lower_data: np.ndarray) -> bool:
        """
        Fit CST functions to airfoil data.
        
        Args:
            upper_data: Upper surface coordinates (N, 2)
            lower_data: Lower surface coordinates (N, 2)
            
        Returns:
            True if fitting was successful, False otherwise
        """
        try:
            self.original_upper_data = upper_data.copy()
            self.original_lower_data = lower_data.copy()
            
            # Perform CST fitting
            # result = fit_airfoil_cst(
            #     upper_data=upper_data,
            #     lower_data=lower_data,
            #     degree=self.degree,
            #     n1=self.n1,
            #     n2=self.n2,
            #     logger_func=self.log_message.emit
            # )

            result = fit_airfoil_cst(
                upper_data=upper_data,
                lower_data=lower_data,
                degree=self.degree,           # initial degree “seed”
                degree_max=max(self.degree, 14),
                auto_degree=True,             # turn on adaptive sweep
                target_max_err=1e-5,          # e-5 band
                tune_n1_n2=True,              # n1/n2 grid-search
                n1=self.n1,
                n2=self.n2,
                fit_te_thickness=False,       # leave as-is unless you need open TE
                logger_func=self.log_message.emit
            )

            # Store results
            self.cst_fitter = result['fitter']
            self.upper_coefficients = result['upper_coefficients']
            self.lower_coefficients = result['lower_coefficients']
            self.upper_metrics = result['upper_metrics']
            self.lower_metrics = result['lower_metrics']
            
            # Generate CST data points
            self.cst_upper_data, self.cst_lower_data = generate_cst_airfoil_data(
                upper_coefficients=self.upper_coefficients,
                lower_coefficients=self.lower_coefficients,
                fitter=self.cst_fitter,
                num_points=1000  # Generate many more points for high-fidelity representation
            )
            
            self.log_message.emit("CST fitting completed successfully.")
            return True
            
        except Exception as e:
            self.log_message.emit(f"CST fitting failed: {e}")
            return False
    
    def get_cst_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the generated CST airfoil data.
        
        Returns:
            Tuple of (upper_data, lower_data) as (N, 2) arrays
        """
        if self.cst_upper_data is None or self.cst_lower_data is None:
            raise ValueError("CST data not available. Run fit_airfoil() first.")
        
        return self.cst_upper_data, self.cst_lower_data
    
    def get_fitting_metrics(self) -> Dict[str, Any]:
        """
        Get the fitting metrics for both surfaces.
        
        Returns:
            Dictionary containing upper and lower surface metrics
        """
        if self.upper_metrics is None or self.lower_metrics is None:
            raise ValueError("Fitting metrics not available. Run fit_airfoil() first.")
        
        return {
            'upper': self.upper_metrics,
            'lower': self.lower_metrics
        }
    
    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the CST coefficients for both surfaces.
        
        Returns:
            Tuple of (upper_coefficients, lower_coefficients)
        """
        if self.upper_coefficients is None or self.lower_coefficients is None:
            raise ValueError("CST coefficients not available. Run fit_airfoil() first.")
        
        return self.upper_coefficients, self.lower_coefficients
    
    def request_plot_update(self, show_original: bool = True, show_cst: bool = True):
        """
        Request a plot update showing the CST fit results.
        
        Args:
            show_original: Whether to show the original airfoil data
            show_cst: Whether to show the CST fit data
        """
        if self.cst_upper_data is None:
            self.log_message.emit("No CST data available for plotting.")
            return
        
        plot_data = {}
        
        if show_original and self.original_upper_data is not None:
            plot_data['original_upper'] = self.original_upper_data
            plot_data['original_lower'] = self.original_lower_data
        
        if show_cst:
            plot_data['cst_upper'] = self.cst_upper_data
            plot_data['cst_lower'] = self.cst_lower_data
        
        # Add metrics to plot data
        if self.upper_metrics and self.lower_metrics:
            plot_data['cst_metrics'] = {
                'upper_rmse': self.upper_metrics['rmse'],
                'upper_max_error': self.upper_metrics['max_error'],
                'upper_orthogonal_max_error': self.upper_metrics['orthogonal_max_error'],
                'lower_rmse': self.lower_metrics['rmse'],
                'lower_max_error': self.lower_metrics['max_error'],
                'lower_orthogonal_max_error': self.lower_metrics['orthogonal_max_error']
            }
        
        self.plot_update_requested.emit(plot_data)
    
    def clear_data(self):
        """Clear all stored data."""
        self.cst_fitter = None
        self.upper_coefficients = None
        self.lower_coefficients = None
        self.upper_metrics = None
        self.lower_metrics = None
        self.cst_upper_data = None
        self.cst_lower_data = None
        self.original_upper_data = None
        self.original_lower_data = None
    
    def is_fitted(self) -> bool:
        """Check if CST fitting has been performed."""
        return (self.cst_upper_data is not None and 
                self.cst_lower_data is not None and
                self.cst_fitter is not None) 