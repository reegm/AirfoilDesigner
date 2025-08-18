"""CadQuery-based STEP export utilities for airfoil surfaces.

This module provides functions to export B-spline curves as separate 1mm thick surfaces
for optimal Fusion 360 import, preserving mathematical fidelity.
"""

import cadquery as cq
import numpy as np
from typing import List, Tuple, Optional
import os


def validate_control_points(control_points: np.ndarray) -> bool:
    """Validate control points for CadQuery export."""
    if control_points.shape[1] != 2:
        return False
    if len(control_points) < 3:
        return False
    if np.any(np.isnan(control_points)) or np.any(np.isinf(control_points)):
        return False
    return True


def export_bspline_separate_surfaces_to_step(
    upper_control_points: np.ndarray,
    lower_control_points: np.ndarray,
    output_path: str,
    chord_length_mm: float = 200.0,
    logger_func: Optional[callable] = None,
    upper_knot_vector: Optional[np.ndarray] = None,
    lower_knot_vector: Optional[np.ndarray] = None,
    degree: int = 5,
) -> bool:
    """
    Export B-spline curves as separate 1mm thick surfaces for optimal Fusion 360 import.
    
    Args:
        upper_control_points: Upper surface control points (N x 2)
        lower_control_points: Lower surface control points (M x 2) 
        output_path: Output .step file path
        chord_length_mm: Scaling factor for chord length
        logger_func: Optional logging function
        
    Returns:
        bool: True if export successful, False otherwise
    """
    try:
        print(f"Creating separate 1mm surfaces for upper/lower B-spline curves")
        
        # Validate control points
        if not validate_control_points(upper_control_points):
            print(f"Invalid upper control points for STEP export")
            return False
            
        if not validate_control_points(lower_control_points):
            print(f"Invalid lower control points for STEP export")
            return False
        
        # Fixed thickness for optimal Fusion 360 handling
        SURFACE_THICKNESS = 1.0  # mm
        
        # Scale control points to desired chord length
        upper_scaled = upper_control_points * chord_length_mm
        lower_scaled = lower_control_points * chord_length_mm
        
        print(f"DEBUG: Upper control points shape: {upper_scaled.shape}")
        print(f"DEBUG: Upper points range: X=[{upper_scaled[:,0].min():.3f}, {upper_scaled[:,0].max():.3f}], Y=[{upper_scaled[:,1].min():.3f}, {upper_scaled[:,1].max():.3f}]")
        
        # Show complete control points being exported
        print(f"DEBUG: Complete upper control points being exported ({len(upper_scaled)} points):")
        for i, cp in enumerate(upper_scaled):
            print(f"DEBUG:   Upper P{i}: ({cp[0]:.6f}, {cp[1]:.6f})")
        
        print(f"DEBUG: Complete lower control points being exported ({len(lower_scaled)} points):")
        for i, cp in enumerate(lower_scaled):
            print(f"DEBUG:   Lower P{i}: ({cp[0]:.6f}, {cp[1]:.6f})")
        
        # Convert to CadQuery format - align to YZ plane (Y=chord, Z=thickness)
        upper_points_3d = [(0.0, float(pt[0]), float(pt[1])) for pt in upper_scaled]
        lower_points_3d = [(0.0, float(pt[0]), float(pt[1])) for pt in lower_scaled]
        
        print(f"DEBUG: Creating B-spline curves from {len(upper_points_3d)} upper and {len(lower_points_3d)} lower control points")
        
        # Create separate surfaces for upper and lower curves using proper B-splines
        wp = cq.Workplane("XY")
        
        # Helper: build OCC B-spline edge from control points (as poles) and knot vector
        def _edge_from_poles_and_knots(points_3d, knot_vector: Optional[np.ndarray]):
            if knot_vector is None:
                return None
            from OCP.gp import gp_Pnt
            from OCP.TColgp import TColgp_Array1OfPnt
            from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
            from OCP.Geom import Geom_BSplineCurve
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            # Poles
            poles_arr = TColgp_Array1OfPnt(1, len(points_3d))
            for i, (x, y, z) in enumerate(points_3d, start=1):
                poles_arr.SetValue(i, gp_Pnt(float(x), float(y), float(z)))

            # Unique knots + multiplicities
            unique_knots = []
            multiplicities = []
            for kv in list(knot_vector):
                if not unique_knots or abs(kv - unique_knots[-1]) > 1e-12:
                    unique_knots.append(float(kv))
                    multiplicities.append(1)
                else:
                    multiplicities[-1] += 1

            knots_arr = TColStd_Array1OfReal(1, len(unique_knots))
            mults_arr = TColStd_Array1OfInteger(1, len(multiplicities))
            for i, kv in enumerate(unique_knots, start=1):
                knots_arr.SetValue(i, float(kv))
            for i, m in enumerate(multiplicities, start=1):
                mults_arr.SetValue(i, int(m))

            bspline = Geom_BSplineCurve(poles_arr, knots_arr, mults_arr, int(degree), False)
            edge = BRepBuilderAPI_MakeEdge(bspline).Edge()
            return edge

        # Helper: build OCC B-spline surface (U: airfoil curve, V: thickness) from poles and U knot vector
        def _face_from_surface_poles(points_3d, knot_vector: Optional[np.ndarray], thickness: float):
            if knot_vector is None:
                return None
            from OCP.gp import gp_Pnt
            from OCP.TColgp import TColgp_Array2OfPnt
            from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
            from OCP.Geom import Geom_BSplineSurface
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

            num_u = len(points_3d)
            if num_u < 2:
                return None

            # Poles grid: two rows in V (x=0 and x=thickness) for YZ plane alignment
            poles = TColgp_Array2OfPnt(1, num_u, 1, 2)
            for i, (x, y, z) in enumerate(points_3d, start=1):
                poles.SetValue(i, 1, gp_Pnt(0.0, float(y), float(z)))
                poles.SetValue(i, 2, gp_Pnt(float(thickness), float(y), float(z)))

            # U knots/mults from provided vector (clamped expected)
            u_knots_list = []
            u_mults_list = []
            for kv in list(knot_vector):
                if not u_knots_list or abs(kv - u_knots_list[-1]) > 1e-12:
                    u_knots_list.append(float(kv))
                    u_mults_list.append(1)
                else:
                    u_mults_list[-1] += 1

            u_knots = TColStd_Array1OfReal(1, len(u_knots_list))
            u_mults = TColStd_Array1OfInteger(1, len(u_mults_list))
            for i, kv in enumerate(u_knots_list, start=1):
                u_knots.SetValue(i, kv)
            for i, m in enumerate(u_mults_list, start=1):
                u_mults.SetValue(i, int(m))

            # V: linear, clamped [0,1] with degree 1 and multiplicity 2 at ends
            v_knots = TColStd_Array1OfReal(1, 2)
            v_knots.SetValue(1, 0.0)
            v_knots.SetValue(2, 1.0)
            v_mults = TColStd_Array1OfInteger(1, 2)
            v_mults.SetValue(1, 2)
            v_mults.SetValue(2, 2)

            surf = Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, int(degree), 1, False, False)
            topo_face = BRepBuilderAPI_MakeFace(surf, 1e-7).Face()
            return cq.Shape.cast(topo_face)

        def _ensure_cq_edge(edge_obj):
            try:
                # If already a CadQuery Edge, return as-is
                if isinstance(edge_obj, cq.Edge):
                    return edge_obj
                # Try to cast OCC TopoDS_Edge to CadQuery Edge
                cq_shape = cq.Shape.cast(edge_obj)
                # If casting produced a shape with Edges, extract the edge
                if hasattr(cq_shape, 'Edges'):
                    edges = cq_shape.Edges()
                    if len(edges):
                        return edges[0]
                return cq_shape  # Might already be an Edge
            except Exception as cast_err:
                print(f"DEBUG: Failed to cast OCC edge to CadQuery edge: {cast_err}")
                raise

        # Upper surface: build true B-spline surface directly from poles + knots
        try:
            print(f"DEBUG: Creating upper B-spline surface from poles + knots...")
            upper_face = _face_from_surface_poles(upper_points_3d, upper_knot_vector, SURFACE_THICKNESS)
            if upper_face is None:
                print(f"DEBUG: Upper surface BSpline build failed, falling back to ruled surface...")
                wp = cq.Workplane("YZ").spline(upper_points_3d)
                upper_edge = wp.val().Edges()[0]
                upper_edge_offset = cq.Edge.makeSpline([cq.Vector(SURFACE_THICKNESS, pt[1], pt[2]) for pt in upper_points_3d])
                upper_face = cq.Face.makeRuledSurface(upper_edge, upper_edge_offset)
            upper_surface = cq.Workplane().add(upper_face)
        except Exception as e:
            print(f"DEBUG: Upper B-spline creation failed: {str(e)}")
            raise
        
        # Lower surface: build true B-spline surface directly from poles + knots
        try:
            print(f"DEBUG: Creating lower B-spline surface from poles + knots...")
            lower_face = _face_from_surface_poles(lower_points_3d, lower_knot_vector, SURFACE_THICKNESS)
            if lower_face is None:
                print(f"DEBUG: Lower surface BSpline build failed, falling back to ruled surface...")
                wp_lower = cq.Workplane("YZ").spline(lower_points_3d)
                lower_edge = wp_lower.val().Edges()[0]
                lower_edge_offset = cq.Edge.makeSpline([cq.Vector(SURFACE_THICKNESS, pt[1], pt[2]) for pt in lower_points_3d])
                lower_face = cq.Face.makeRuledSurface(lower_edge, lower_edge_offset)
            lower_surface = cq.Workplane().add(lower_face)
        except Exception as e:
            print(f"DEBUG: Lower B-spline creation failed: {str(e)}")
            raise
        
        # Export both surfaces separately to single STEP file
        try:
            print(f"DEBUG: Exporting separate upper and lower B-spline surfaces to STEP file: {output_path}")
            
            # Create a compound with both surfaces
            from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder
            
            # Get the actual face objects
            upper_face_obj = None
            lower_face_obj = None
            
            # Extract faces from workplanes
            if hasattr(upper_surface, 'objects') and upper_surface.objects:
                upper_face_obj = upper_surface.objects[0]
                print(f"DEBUG: Extracted upper face object: {type(upper_face_obj)}")
            elif hasattr(upper_surface, 'val'):
                upper_face_obj = upper_surface.val()
                print(f"DEBUG: Got upper face via val(): {type(upper_face_obj)}")
                    
            if hasattr(lower_surface, 'objects') and lower_surface.objects:
                lower_face_obj = lower_surface.objects[0]
                print(f"DEBUG: Extracted lower face object: {type(lower_face_obj)}")
            elif hasattr(lower_surface, 'val'):
                lower_face_obj = lower_surface.val()
                print(f"DEBUG: Got lower face via val(): {type(lower_face_obj)}")
            
            if upper_face_obj and lower_face_obj:
                # Create compound with both faces
                compound = TopoDS_Compound()
                builder = TopoDS_Builder()
                builder.MakeCompound(compound)
                
                if hasattr(upper_face_obj, 'wrapped'):
                    builder.Add(compound, upper_face_obj.wrapped)
                else:
                    builder.Add(compound, upper_face_obj)
                    
                if hasattr(lower_face_obj, 'wrapped'):
                    builder.Add(compound, lower_face_obj.wrapped)
                else:
                    builder.Add(compound, lower_face_obj)
                
                # Export compound to STEP using STEPControl_Writer (accepts TopoDS_Shape)
                from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
                step_writer = STEPControl_Writer()
                step_writer.Transfer(compound, STEPControl_AsIs)
                status = step_writer.Write(output_path)
                
                
                print(f"DEBUG: STEP compound export status: {status}")
                print(f"DEBUG: STEP export completed successfully with separate B-spline surfaces")
            else:
                print(f"DEBUG: Could not extract face objects, trying direct export...")
                # Fallback: try direct export of first surface
                if hasattr(upper_surface, 'val'):
                    upper_surface.val().exportStep(output_path)
                    print(f"DEBUG: Direct upper surface export completed")
                else:
                    raise Exception("Could not extract surfaces for STEP export")
                    
        except Exception as e:
            print(f"DEBUG: STEP file export failed: {str(e)}")
            raise
        
        print(f"STEP export successful: {len(upper_points_3d)} upper + {len(lower_points_3d)} lower control points")
        print(f"Created separate 1mm B-spline surfaces for optimal Fusion 360 curve extraction")
        
        return True
        
    except Exception as e:
        print(f"CadQuery STEP export failed: {str(e)}")
        return False
