import ezdxf
import traceback
import numpy as np
from .nurbs_utils import create_nurbs_curve_from_bezier, generate_knot_vector

def export_nurbs_curves_to_dxf(polygons, chord_length_mm, logger_func, degree=3, num_samples=200):
    """
    Exports a list of Bezier curves to a DXF file using NURBS representation.

    Args:
        polygons (list of list of np.array): A list of polygons, where each polygon is a list of control points.
        chord_length_mm (float): The desired chord length in millimeters for scaling.
        logger_func (callable): A function (like print or a signal's emit method) to send log messages to.
        degree (int): Degree of the NURBS curves (default: 3 for cubic)
        num_samples (int): Number of sample points for curve evaluation

    Returns:
        ezdxf.document.Drawing: The created DXF document object, or None if an error occurred.
    """
    try:
        if not polygons:
            logger_func("Error: No polygons provided for DXF export.")
            return None

        if chord_length_mm <= 0:
            logger_func("Error: Chord length must be positive for DXF export.")
            return None

        logger_func(f"Preparing NURBS DXF export with chord length: {chord_length_mm:.2f} mm...")

        # Create DXF document
        doc = ezdxf.new('R2000')
        doc.header["$INSUNITS"] = 4  # 4 = millimetres in DXF spec
        msp = doc.modelspace()

        # Scale all polygons by the desired chord length
        scaled_polygons = [[p * chord_length_mm for p in poly] for poly in polygons]

        # Define colors for different elements
        colors = [1, 5, 2]  # Red, Blue, Yellow

        for i, poly in enumerate(scaled_polygons):
            color = colors[i % len(colors)]
            
            # Convert to numpy array if needed
            control_points = np.array(poly)
            
            # Create NURBS curve from Bezier control points
            sample_points, knots, weights = create_nurbs_curve_from_bezier(
                control_points, degree=degree, num_samples=num_samples
            )
            
            # Convert sample points to tuples for ezdxf
            curve_points = [tuple(pt.tolist()) for pt in sample_points]
            
            # Add NURBS curve using fit points (approximation)
            # ezdxf doesn't have direct NURBS support, so we use fit points
            # Ensure we have enough points for smooth representation
            if len(curve_points) < 3:
                logger_func(f"Warning: Curve {i+1} has too few points ({len(curve_points)}), skipping")
                continue
                
            msp.add_spline(
                fit_points=curve_points,
                dxfattribs={"layer": "AIRFOIL_NURBS_CURVE", "color": color},
            )
            
            logger_func(f"Added NURBS curve {i+1} with {len(curve_points)} fit points")

        # Add trailing edge connector if required
        if len(scaled_polygons) >= 2:
            upper_poly, lower_poly = scaled_polygons[:2]
            # Only if the airfoil is not sharp
            if not np.allclose(upper_poly[-1], lower_poly[-1]):
                msp.add_line(
                    tuple(upper_poly[-1].tolist()), 
                    tuple(lower_poly[-1].tolist()), 
                    dxfattribs={'layer': 'TRAILING_EDGE_CONNECTOR', 'color': 2}
                )

        logger_func(f"NURBS DXF document successfully created in memory.")
        return doc
        
    except Exception as e:
        logger_func(f"Error during NURBS DXF export: {e}")
        logger_func(traceback.format_exc())
        return None

def export_nurbs_curves_to_dxf_with_control_points(polygons, chord_length_mm, logger_func, degree=3):
    """
    Export method that creates B-splines using control points.
    This method preserves the original control point structure and creates proper B-splines.

    Args:
        polygons (list of list of np.array): A list of polygons, where each polygon is a list of control points.
        chord_length_mm (float): The desired chord length in millimeters for scaling.
        logger_func (callable): A function to send log messages to.
        degree (int): Degree of the B-spline curves

    Returns:
        ezdxf.document.Drawing: The created DXF document object, or None if an error occurred.
    """
    try:
        if not polygons:
            logger_func("Error: No polygons provided for DXF export.")
            return None

        if chord_length_mm <= 0:
            logger_func("Error: Chord length must be positive for DXF export.")
            return None

        logger_func(f"Preparing NURBS DXF export (control points) with chord length: {chord_length_mm:.2f} mm...")

        # Create DXF document
        doc = ezdxf.new('R2000')
        doc.header["$INSUNITS"] = 4  # 4 = millimetres in DXF spec
        msp = doc.modelspace()

        # Scale all polygons by the desired chord length
        scaled_polygons = [[p * chord_length_mm for p in poly] for poly in polygons]

        # Define colors for different elements
        colors = [1, 5, 2]  # Red, Blue, Yellow

        for i, poly in enumerate(scaled_polygons):
            color = colors[i % len(colors)]
            
            # Convert to numpy array if needed
            control_points = np.array(poly)
            n = len(control_points) - 1
            
            # Ensure degree doesn't exceed maximum allowed
            max_degree = n
            if degree > max_degree:
                degree = max_degree
                logger_func(f"Reduced B-spline degree to {degree} (max allowed for {n+1} control points)")
            
            # Convert control points to tuples for ezdxf
            control_pts = [tuple(pt.tolist()) for pt in control_points]
            
            # Add B-spline using control points
            # Use add_open_spline for proper B-spline behavior with control points
            msp.add_open_spline(
                control_points=control_pts,
                degree=degree,
                dxfattribs={"layer": "AIRFOIL_NURBS_CONTROL", "color": color},
            )
            
            logger_func(f"Added B-spline curve {i+1} with {len(control_pts)} control points")

        # Add trailing edge connector if required
        if len(scaled_polygons) >= 2:
            upper_poly, lower_poly = scaled_polygons[:2]
            # Only if the airfoil is not sharp
            if not np.allclose(upper_poly[-1], lower_poly[-1]):
                msp.add_line(
                    tuple(upper_poly[-1].tolist()), 
                    tuple(lower_poly[-1].tolist()), 
                    dxfattribs={'layer': 'TRAILING_EDGE_CONNECTOR', 'color': 2}
                )

        logger_func(f"NURBS DXF document (B-spline control points) successfully created in memory.")
        return doc
        
    except Exception as e:
        logger_func(f"Error during NURBS DXF export (control points): {e}")
        logger_func(traceback.format_exc())
        return None 