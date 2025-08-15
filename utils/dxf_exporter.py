import ezdxf
import traceback
import numpy as np

def export_curves_to_dxf(polygons, chord_length_mm, logger_func):
    """
    Exports a list of Bezier curves to a DXF file.

    Args:
        polygons (list of list of np.array): A list of polygons, where each polygon is a list of control points.
        chord_length_mm (float): The desired chord length in millimeters for scaling.
        logger_func (callable): A function (like print or a signal's emit method) to send log messages to.
        thickened (bool): Flag indicating if the trailing edge is thickened or unthickened.

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

        logger_func(f"Preparing DXF export with chord length: {chord_length_mm:.2f} mm...")

        # Only use ezdxf.new('R2000') for DXF creation. Do not import 'new' directly from ezdxf.
        doc = ezdxf.new('R2000')
        # Set drawing units to millimetres, matching the test generator script
        doc.header["$INSUNITS"] = 4  # 4 = millimetres in DXF spec
        msp = doc.modelspace()

        # Scale all polygons by the desired chord length
        scaled_polygons = [[p * chord_length_mm for p in poly] for poly in polygons]

        # Define colors for different elements for better visualization in CAD software
        colors = [1, 5, 2] # Red, Blue, Yellow, Magenta

        for i, poly in enumerate(scaled_polygons):
            color = colors[i % len(colors)]

            # Convert numpy points to plain (x, y) tuples for ezdxf
            control_pts = [tuple(pt.tolist()) for pt in poly]

            # Determine spline degree: #control_points - 1, capped at 9 per DXF spec
            degree = min(len(control_pts) - 1, 9)

            # Add open (clamped) B-spline defined by control points
            msp.add_open_spline(
                control_points=control_pts,
                degree=degree,
                dxfattribs={"layer": "AIRFOIL_CURVE", "color": color},
            )

        # Add trailing edge connector if required
        if len(scaled_polygons):
            upper_poly, lower_poly = scaled_polygons
            # only if the airfoil is not sharp
            if not np.allclose(upper_poly[-1], lower_poly[-1]):
                msp.add_line(upper_poly[-1], lower_poly[-1], dxfattribs={'layer': 'TRAILING_EDGE_CONNECTOR', 'color': 2}) # Yellow

        logger_func(f"DXF document successfully created in memory.")
        return doc
    except Exception as e:
        logger_func(f"Error during DXF export: {e}")
        logger_func(traceback.format_exc()) # Log full traceback for debugging
        return None


def export_bspline_to_dxf(bspline_processor, chord_length_mm, logger_func):
    """
    Export B-spline curves to DXF as NURBS curves.
    
    Args:
        bspline_processor: BSplineProcessor instance with fitted B-spline data
        chord_length_mm (float): The desired chord length in millimeters for scaling
        logger_func (callable): A function to send log messages to
        
    Returns:
        ezdxf.document.Drawing: The created DXF document object, or None if an error occurred
    """
    try:
        if not bspline_processor.fitted:
            logger_func("Error: No B-spline fit available for DXF export.")
            return None
            
        if chord_length_mm <= 0:
            logger_func("Error: Chord length must be positive for DXF export.")
            return None

        logger_func(f"Preparing B-spline DXF export with chord length: {chord_length_mm:.2f} mm...")

        # Get B-spline control points
        upper_ctrl_pts = bspline_processor.upper_control_points
        lower_ctrl_pts = bspline_processor.lower_control_points
        
        if upper_ctrl_pts is None or lower_ctrl_pts is None:
            logger_func("Error: B-spline control points not available for DXF export.")
            return None
        
        # Scale control points by chord length
        upper_ctrl_pts_scaled = upper_ctrl_pts * chord_length_mm
        lower_ctrl_pts_scaled = lower_ctrl_pts * chord_length_mm
        
        # Create DXF document
        doc = ezdxf.new('R2000')
        doc.header["$INSUNITS"] = 4  # millimeters
        msp = doc.modelspace()
        
        # Convert control points to format expected by ezdxf
        upper_points = [tuple(pt.tolist()) for pt in upper_ctrl_pts_scaled]
        lower_points = [tuple(pt.tolist()) for pt in lower_ctrl_pts_scaled]
        
        # Determine degrees
        upper_degree = len(upper_points) - 1
        lower_degree = len(lower_points) - 1
        
        logger_func(f"Creating NURBS curves: upper degree {upper_degree}, lower degree {lower_degree}")
        
        # Add upper surface NURBS curve
        msp.add_open_spline(
            control_points=upper_points,
            degree=upper_degree,
            dxfattribs={"layer": "AIRFOIL_UPPER", "color": 1}  # Red
        )
        logger_func(f"  Upper surface: degree {upper_degree} B-spline with {len(upper_points)} control points")
        
        # Add lower surface NURBS curve  
        msp.add_open_spline(
            control_points=lower_points,
            degree=lower_degree,
            dxfattribs={"layer": "AIRFOIL_LOWER", "color": 5}  # Blue
        )
        logger_func(f"  Lower surface: degree {lower_degree} B-spline with {len(lower_points)} control points")
        
        # Add trailing edge connector if needed (for blunt trailing edge)
        if not bspline_processor.is_sharp_te:
            if not np.allclose(upper_ctrl_pts_scaled[-1], lower_ctrl_pts_scaled[-1]):
                msp.add_line(
                    tuple(upper_ctrl_pts_scaled[-1].tolist()), 
                    tuple(lower_ctrl_pts_scaled[-1].tolist()),
                    dxfattribs={'layer': 'TRAILING_EDGE_CONNECTOR', 'color': 2}  # Yellow
                )
                logger_func("  Added trailing edge connector for blunt trailing edge")
        
        logger_func(f"B-spline DXF export completed successfully.")
        return doc
        
    except Exception as e:
        logger_func(f"Error during B-spline DXF export: {e}")
        logger_func(traceback.format_exc())
        return None
