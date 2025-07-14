import ezdxf
import traceback

def export_curves_to_dxf(polygons, chord_length_mm, logger_func, thickened):
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

        # Define colors for different segments for better visualization in CAD software
        colors = [1, 5, 2, 6] # Red, Blue, Yellow, Magenta

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

        logger_func(f"DXF document successfully created in memory.")
        return doc
    except Exception as e:
        logger_func(f"Error during DXF export: {e}")
        logger_func(traceback.format_exc()) # Log full traceback for debugging
        return None
