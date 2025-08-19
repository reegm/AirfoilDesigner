"""Optimization controller for the Airfoil Designer GUI.

Handles background optimization processes, progress tracking, and worker management.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any

from PySide6.QtCore import QTimer

from core import config
from core.airfoil_processor import AirfoilProcessor
from core.error_functions import calculate_single_bezier_fitting_error
from .optimization_worker import _generation_worker


class OptimizationController:
    """Handles optimization processes and background worker management."""
    
    def __init__(self, processor: AirfoilProcessor, window: Any, ui_state_controller: Any = None):
        self.processor = processor
        self.window = window
        self.ui_state_controller = ui_state_controller
        
        # Optimization state
        self._generation_process = None
        self._generation_queue = None
        self._abort_flag = None
        self._generation_timer = QTimer()
        self._generation_timer.setInterval(200)  # ms
        self._generation_timer.timeout.connect(self._check_generation_result)
        self._is_generating = False
        self._generation_start_time = None
        self._current_progress_info = None
        # Track best-so-far per surface from progress updates
        self._best_true_max_upper = float('inf')
        self._best_true_max_lower = float('inf')
        self._best_ctrl_upper = None
        self._best_ctrl_lower = None
    
    def generate_or_abort(self) -> None:
        """Handle generate/abort button action."""
        opt = self.window.optimizer_panel
        if self._is_generating:
            # Abort requested
            if self._abort_flag is not None:
                self._abort_flag.value = True
                self.processor.log_message.emit("Abort requested. Waiting for optimizer to finish up...")
            # Do NOT set self._is_generating = False here
            # Do NOT reset button or update button states here
            # Let _check_generation_result handle cleanup and GUI updates
            return
        
        # Start generation in a new process
        try:
            regularization_weight = float(opt.single_bezier_reg_weight_input.text())
            te_vector_points = int(opt.te_vector_points_combo.currentText())
            g2_flag = opt.g2_checkbox.isChecked()
            enforce_te_tangency = opt.enforce_te_tangency_checkbox.isChecked()
            gui_strategy = opt.strategy_combo.currentText().lower()  # 'fixed-x' or 'free-x'
            error_function = "euclidean" #opt.error_function_combo.currentText().lower()  # 'euclidean' or 'orthogonal'
            objective_gui = opt.objective_combo.currentText()
            if objective_gui == 'MSR':
                objective_type = 'msr'
            elif objective_gui == 'Softmax':
                objective_type = 'softmax'
            else:
                self.processor.log_message.emit(f"Error: Unsupported objective '{objective_gui}'.")
                return
            if gui_strategy not in ['fixed-x', 'free-x']:
                self.processor.log_message.emit(f"Error: Unsupported strategy '{gui_strategy}'.")
                return
        except ValueError:
            self.processor.log_message.emit(
                "Error: Invalid input for regularization weight, curve error points, or TE vector points. Please enter valid numbers."
            )
            return
        
        # Clear previous Bezier model results to prevent showing old results during new generation
        self.processor.upper_poly_sharp = None
        self.processor.lower_poly_sharp = None
        self.processor.last_single_bezier_upper_max_error = None
        self.processor.last_single_bezier_upper_max_error_idx = None
        self.processor.last_single_bezier_lower_max_error = None
        self.processor.last_single_bezier_lower_max_error_idx = None
        
        # Clear any progress info from previous runs
        self._current_progress_info = None
        self._best_true_max_upper = float('inf')
        self._best_true_max_lower = float('inf')
        self._best_ctrl_upper = None
        self._best_ctrl_lower = None
        
        # Update the plot to clear previous results
        self.processor._request_plot_update()
        
        self._generation_queue = multiprocessing.Queue()
        self._abort_flag = multiprocessing.Value('b', False)  # Shared boolean flag
        args = (
            self.processor.upper_data,
            self.processor.lower_data,
            regularization_weight,
            g2_flag,
            te_vector_points,
            gui_strategy,
            error_function,
            objective_type,
            enforce_te_tangency,
            self._abort_flag
        )
        self._generation_process = multiprocessing.Process(
            target=_generation_worker,
            args=(args, self._generation_queue)
        )
        self._generation_process.start()
        self._is_generating = True
        self._generation_start_time = time.time()
        opt.build_single_bezier_button.setText("Abort")
        self._generation_timer.start()
        self.processor.log_message.emit("Started airfoil generation in background process...")

    def run_staged_or_abort(self) -> None:
        """Start or abort the staged uncoupled optimization pipeline."""
        opt = self.window.optimizer_panel
        if self._is_generating:
            if self._abort_flag is not None:
                self._abort_flag.value = True
                self.processor.log_message.emit("Abort requested. Waiting for optimizer to finish up...")
            return

        # Start staged in a new process
        try:
            regularization_weight = float(opt.single_bezier_reg_weight_input.text())
            te_vector_points = int(opt.te_vector_points_combo.currentText())
            g2_flag = opt.g2_checkbox.isChecked()
            enforce_te_tangency = opt.enforce_te_tangency_checkbox.isChecked()
        except ValueError:
            self.processor.log_message.emit("Error: Invalid input for regularization weight or TE vector points.")
            return

        # Staged optimization now supports both coupled and uncoupled modes

        # Clear previous results
        self.processor.upper_poly_sharp = None
        self.processor.lower_poly_sharp = None
        self.processor.last_single_bezier_upper_max_error = None
        self.processor.last_single_bezier_upper_max_error_idx = None
        self.processor.last_single_bezier_lower_max_error = None
        self.processor.last_single_bezier_lower_max_error_idx = None
        self._current_progress_info = None
        self._best_true_max_upper = float('inf')
        self._best_true_max_lower = float('inf')
        self._best_ctrl_upper = None
        self._best_ctrl_lower = None
        self.processor._request_plot_update()

        self._generation_queue = multiprocessing.Queue()
        self._abort_flag = multiprocessing.Value('b', False)
        args = (
            self.processor.upper_data,
            self.processor.lower_data,
            regularization_weight,
            g2_flag,                 # Use actual G2 flag value
            te_vector_points,
            'staged',                # gui_strategy for worker
            'euclidean',             # error function forced to euclidean
            'softmax',                # objective placeholder
            enforce_te_tangency,
            self._abort_flag,
        )
        self._generation_process = multiprocessing.Process(
            target=_generation_worker,
            args=(args, self._generation_queue)
        )
        self._generation_process.start()
        self._is_generating = True
        self._generation_start_time = time.time()
        opt.staged_opt_button.setText("Abort")
        self._generation_timer.start()
        self.processor.log_message.emit("Started staged optimization in background process...")
    
    def recalculate_te_vectors(self) -> None:
        """Handle recalculate TE vectors action."""
        opt = self.window.optimizer_panel
        try:
            te_vector_points = int(opt.te_vector_points_combo.currentText())
        except ValueError:
            self.processor.log_message.emit("Error: Invalid input values. Please check all numeric inputs.")
            return

        self.processor.recalculate_te_vectors_and_update_plot(te_vector_points)
        # Update the default TE vector points to the value just used
        opt.set_default_te_vector_points(te_vector_points)
        # Disable the recalculate button until dropdown changes again
        opt.disable_recalc_button()
        self.processor.log_message.emit(f"Recalculating with TE vector points set to: {te_vector_points}")
        
        # Update UI state after TE vector recalculation
        if self.ui_state_controller:
            self.ui_state_controller.update_button_states()
    
    def _check_generation_result(self) -> None:
        """Check for results from the background generation process."""
        if not self._is_generating:
            self._generation_timer.stop()
            return
        if self._generation_queue is not None and not self._generation_queue.empty():
            result = self._generation_queue.get()
            
            # Handle log messages from the worker process (always show in GUI)
            if isinstance(result, dict) and result.get("type") == "log":
                self.processor.log_message.emit(result["message"])
                return  # Continue checking for more messages
            
            # Handle progress updates from the worker process
            if isinstance(result, dict) and result.get("type") == "progress":
                self._handle_progress_update(result)
                return  # Continue checking for more messages
            
            # Handle final result
            self._generation_timer.stop()
            self._is_generating = False
            opt = self.window.optimizer_panel
            # Reset both buttons to default text
            opt.build_single_bezier_button.setText("Generate Airfoil")
            if hasattr(opt, 'staged_opt_button'):
                opt.staged_opt_button.setText("Staged Optimization")
            
            self._generation_process = None
            self._generation_queue = None
            self._abort_flag = None
            if isinstance(result, dict) and result.get("success") and result.get("upper_poly") is not None:
                # Update processor state with new model
                self.processor.upper_poly_sharp = result["upper_poly"]
                self.processor.lower_poly_sharp = result["lower_poly"]
                self.processor.last_single_bezier_upper_max_error = result.get("upper_max_error")
                self.processor.last_single_bezier_upper_max_error_idx = result.get("upper_max_error_idx")
                self.processor.last_single_bezier_lower_max_error = result.get("lower_max_error")
                self.processor.last_single_bezier_lower_max_error_idx = result.get("lower_max_error_idx")
                elapsed_time = time.time() - self._generation_start_time if self._generation_start_time else 0
                
                # Log success with error information
                base_message = f"Single Bezier model built successfully. (Elapsed time: {elapsed_time:.2f}s)"
                
                # Add error information if available
                upper_error = result.get("upper_max_error")
                lower_error = result.get("lower_max_error")
                if upper_error is not None and lower_error is not None:
                    # Convert to chord percentage and mm if chord length is available
                    try:
                        chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
                        upper_error_mm = upper_error * chord_length_mm
                        lower_error_mm = lower_error * chord_length_mm
                        error_message = f"\n  Upper surface max error: {upper_error:.3e} ({upper_error_mm:.3f}mm @ {chord_length_mm:.0f}mm chord)"
                        error_message += f"\n  Lower surface max error: {lower_error:.3e} ({lower_error_mm:.3f}mm @ {chord_length_mm:.0f}mm chord)"
                    except:
                        # Fallback to just normalized units
                        error_message = f"\n  Upper surface max error: {upper_error:.3e}"
                        error_message += f"\n  Lower surface max error: {lower_error:.3e}"
                    
                    base_message += error_message
                
                self.processor.log_message.emit(base_message)
                self._generation_start_time = None
                self.processor._request_plot_update()
                
                # Update UI state after successful generation
                if self.ui_state_controller:
                    self.ui_state_controller.update_button_states()
            else:
                self.processor.log_message.emit(result.get("error", "Failed to build single Bezier model."))
        elif self._generation_process is not None and not self._generation_process.is_alive():
            # Process died without result
            self._generation_timer.stop()
            self._is_generating = False
            opt = self.window.optimizer_panel
            opt.build_single_bezier_button.setText("Generate Airfoil")
            if hasattr(opt, 'staged_opt_button'):
                opt.staged_opt_button.setText("Staged Optimization")
            
            elapsed_time = time.time() - self._generation_start_time if self._generation_start_time else 0
            self._generation_process = None
            self._generation_queue = None
            self._abort_flag = None
            # Fallback: use best-so-far control points from progress updates if available
            use_fallback = (self._best_ctrl_upper is not None and self._best_ctrl_lower is not None)
            if use_fallback:
                error_function = 'euclidean'
                try:
                    upper_result = calculate_single_bezier_fitting_error(
                        self._best_ctrl_upper, self.processor.upper_data, error_function=error_function, return_max_error=True
                    )
                    lower_result = calculate_single_bezier_fitting_error(
                        self._best_ctrl_lower, self.processor.lower_data, error_function=error_function, return_max_error=True
                    )
                    _, upper_max_error, upper_max_error_idx = upper_result
                    _, lower_max_error, lower_max_error_idx = lower_result
                except Exception:
                    upper_max_error = lower_max_error = upper_max_error_idx = lower_max_error_idx = None
                base_message = f"Single Bezier model built successfully (best-so-far after early exit). (Elapsed time: {elapsed_time:.2f}s)"
                if upper_max_error is not None and lower_max_error is not None:
                    try:
                        chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
                        umm = upper_max_error * chord_length_mm
                        lmm = lower_max_error * chord_length_mm
                        base_message += f"\n  Upper surface max error: {upper_max_error:.3e} ({umm:.3f}mm @ {chord_length_mm:.0f}mm chord)"
                        base_message += f"\n  Lower surface max error: {lower_max_error:.3e} ({lmm:.3f}mm @ {chord_length_mm:.0f}mm chord)"
                    except Exception:
                        base_message += f"\n  Upper surface max error: {upper_max_error:.3e}"
                        base_message += f"\n  Lower surface max error: {lower_max_error:.3e}"
                self.processor.log_message.emit(base_message)
                # Ensure final plot update reflects best-so-far control points
                self.processor.upper_poly_sharp = self._best_ctrl_upper
                self.processor.lower_poly_sharp = self._best_ctrl_lower
                self.processor._request_plot_update()
                # Update UI state after successful fallback
                if self.ui_state_controller:
                    self.ui_state_controller.update_button_states()
            else:
                self.processor.log_message.emit(f"Generation process exited unexpectedly. (Elapsed time: {elapsed_time:.2f}s)")
            self._generation_start_time = None
    
    def _handle_progress_update(self, progress_data: dict) -> None:
        """Handle progress updates from the optimization process."""
        iteration = progress_data.get("iteration", 0)
        elapsed = progress_data.get("elapsed", 0)
        val = progress_data.get("val", 0)
        true_max = progress_data.get("true_max")
        best_true_max = progress_data.get("best_true_max")
        current_ctrl = progress_data.get("current_ctrl")
        surface_info = progress_data.get("surface_info")
        
        # Track best-so-far control points per surface
        if current_ctrl is not None and best_true_max is not None:
            if surface_info == "upper" or (surface_info is None and self.processor.lower_poly_sharp is None):
                if best_true_max < self._best_true_max_upper:
                    self._best_true_max_upper = best_true_max
                    self._best_ctrl_upper = current_ctrl
            elif surface_info == "lower" or (surface_info is None and self.processor.upper_poly_sharp is not None):
                if best_true_max < self._best_true_max_lower:
                    self._best_true_max_lower = best_true_max
                    self._best_ctrl_lower = current_ctrl

        # Update the plot only if enabled
        if getattr(config, 'UPDATE_PLOT', False) and current_ctrl is not None:
            self._update_plot_with_progress(current_ctrl, iteration, elapsed, true_max, best_true_max, surface_info)
    
    def _update_plot_with_progress(self, current_ctrl, iteration, elapsed, true_max, best_true_max, surface_info=None) -> None:
        """Update the plot with current optimization progress."""
        # Handle different control point formats
        if isinstance(current_ctrl, tuple):
            # Coupled optimization - we have (upper_ctrl, lower_ctrl)
            upper_ctrl, lower_ctrl = current_ctrl
            # Temporarily store the current progress control points
            self.processor.upper_poly_sharp = upper_ctrl
            self.processor.lower_poly_sharp = lower_ctrl
        else:
            # Single surface optimization - we have just one control point set
            # Use surface_info to determine which surface this is for
            if surface_info == "upper" or (surface_info is None and self.processor.lower_poly_sharp is None):
                # This is the upper surface optimization
                self.processor.upper_poly_sharp = current_ctrl
            elif surface_info == "lower" or (surface_info is None and self.processor.upper_poly_sharp is not None):
                # This is the lower surface optimization
                self.processor.lower_poly_sharp = current_ctrl
        
        # Calculate current error for progress display
        upper_max_error = None
        upper_max_error_idx = None
        lower_max_error = None
        lower_max_error_idx = None
        
        # Get the error function from the optimizer settings
        # opt = self.window.optimizer_panel
        error_function = "euclidean" #opt.error_function_combo.currentText().lower()
        
        if self.processor.upper_poly_sharp is not None and self.processor.upper_data is not None:
            try:
                upper_error_result = calculate_single_bezier_fitting_error(
                    self.processor.upper_poly_sharp, 
                    self.processor.upper_data, 
                    error_function=error_function, 
                    return_max_error=True
                )
                _, upper_max_error, upper_max_error_idx = upper_error_result
            except:
                pass  # Ignore errors during progress updates
        
        if self.processor.lower_poly_sharp is not None and self.processor.lower_data is not None:
            try:
                lower_error_result = calculate_single_bezier_fitting_error(
                    self.processor.lower_poly_sharp, 
                    self.processor.lower_data, 
                    error_function=error_function, 
                    return_max_error=True
                )
                _, lower_max_error, lower_max_error_idx = lower_error_result
            except:
                pass  # Ignore errors during progress updates
        
        # Get chord length for error display in mm
        try:
            chord_length_mm = float(self.window.airfoil_settings_panel.chord_length_input.text())
        except:
            chord_length_mm = None
        
        # Create a progress plot data dictionary (only include valid plot_airfoil arguments)
        progress_plot_data = {
            'upper_data': self.processor.upper_data,
            'lower_data': self.processor.lower_data,
            'upper_te_tangent_vector': self.processor.upper_te_tangent_vector,
            'lower_te_tangent_vector': self.processor.lower_te_tangent_vector,
            'single_bezier_upper_poly': self.processor.upper_poly_sharp,
            'single_bezier_lower_poly': self.processor.lower_poly_sharp,
            'worst_single_bezier_upper_max_error': upper_max_error,
            'worst_single_bezier_lower_max_error': lower_max_error,
            'worst_single_bezier_upper_max_error_idx': upper_max_error_idx,
            'worst_single_bezier_lower_max_error_idx': lower_max_error_idx,
            'comb_single_bezier': None,  # Skip comb calculation for performance
            'chord_length_mm': chord_length_mm,
        }
        
        # Store progress info separately for the controller to use
        self._current_progress_info = {
            'iteration': iteration,
            'elapsed': elapsed,
            'true_max': true_max,
            'best_true_max': best_true_max
        }
        
        # Emit the progress plot update
        self.processor.plot_update_requested.emit(progress_plot_data)
    
    @property
    def is_generating(self) -> bool:
        """Check if optimization is currently running."""
        return self._is_generating 