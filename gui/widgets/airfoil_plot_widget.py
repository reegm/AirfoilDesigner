"""Interactive airfoil visualisation widget.

This is the *actual* implementation of :class:`AirfoilPlotWidget`, moved
from ``gui/airfoil_plot_widget.py`` into the new widgets package as part
of the GUI refactor (2025-07-17).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from core.config import (
    PLOT_POINTS_PER_SURFACE,
)


class AirfoilPlotWidget(pg.PlotWidget):
    """Custom `pyqtgraph.PlotWidget` tailored for airfoil visualisation."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setParent(parent)
        # Ensure the plot expands nicely inside layouts
        self.setSizePolicy(
            pg.QtWidgets.QSizePolicy.Policy.Expanding,
            pg.QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()

        # Initial plot settings
        pg.setConfigOptions(antialias=True)
        self.setAspectLocked(True)  # Maintain aspect ratio
        self.showGrid(x=True, y=True)  # Show grid
        self.addLegend(offset=(30, 10))  # Add legend with some offset
        self.setLabel("bottom", "x/c (Chord)")
        self.setLabel("left", "y/c (Chord)")

        self.plot_items: dict[str, object] = {}
        self._first_plot_done = False  # Flag to control initial zoom
        self.getViewBox().sigRangeChanged.connect(self._update_error_text_positions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_airfoil(
        self,
        upper_data,
        lower_data,
        upper_te_tangent_vector=None,
        lower_te_tangent_vector=None,
        chord_length_mm=None,
        geometry_metrics=None,
        bspline_upper_curve=None,
        bspline_lower_curve=None,
        bspline_upper_control_points=None,
        bspline_lower_control_points=None,
        bspline_upper_max_error=None,
        bspline_lower_max_error=None,
        bspline_upper_max_error_idx=None,
        bspline_lower_max_error_idx=None,
        comb_bspline=None,
        bspline_is_thickened=False,
    ):
        """Render everything supplied in *kwargs* on the canvas."""
        # Clear all items to ensure no remnants
        self.clear()
        self.addLegend(offset=(30, 10))  # Re-add legend after clearing
        self.plot_items = {}

        # Colours -------------------------------------------------------
        COLOR_ORIGINAL_DATA = (135, 206, 250, 200)  # SkyBlue

        COLOR_TE_TANGENT_UPPER = pg.mkPen("red", width=2, style=Qt.PenStyle.SolidLine)
        COLOR_TE_TANGENT_LOWER = pg.mkPen("purple", width=2, style=Qt.PenStyle.SolidLine)

        # Comb colours
        COLOR_BSPLINE_COMB = pg.mkPen((180, 180, 180), width=1)
        COLOR_COMB_OUTLINE = pg.mkPen("yellow", width=2, style=Qt.PenStyle.DotLine)

        # B-spline colors
        COLOR_BSPLINE_CURVE = pg.mkPen("magenta", width=2.5)
        COLOR_BSPLINE_CONTROL_POINTS = pg.mkPen((255, 20, 147), width=1.5, style=Qt.PenStyle.DashLine)
        COLOR_BSPLINE_CONTROL_SYMBOL = pg.mkBrush((255, 20, 147, 200))
        COLOR_BSPLINE_CONTROL_SYMBOL_PEN = pg.mkPen((255, 20, 147), width=1.5)
        
        # Thickened B-spline colors
        COLOR_THICKENED_BSPLINE_CURVE = pg.mkPen("darkmagenta", width=3.0)
        COLOR_THICKENED_BSPLINE_CONTROL_POINTS = pg.mkPen((139, 0, 139), width=1.5, style=Qt.PenStyle.DotLine)
        COLOR_THICKENED_BSPLINE_CONTROL_SYMBOL = pg.mkBrush((139, 0, 139, 200))
        COLOR_THICKENED_BSPLINE_CONTROL_SYMBOL_PEN = pg.mkPen((139, 0, 139), width=1.5)

        # --------------------------------------------------------------
        # 1) Original data
        # --------------------------------------------------------------
        all_original_data = np.concatenate([upper_data, lower_data])
        self.plot_items["Original Data"] = self.plot(
            all_original_data[:, 0],
            all_original_data[:, 1],
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(COLOR_ORIGINAL_DATA),
            name="Original Data",
        )

        # --------------------------------------------------------------
        # 2) B-spline curves and control points
        # --------------------------------------------------------------
        if bspline_upper_curve is not None or bspline_lower_curve is not None:
            # Choose colors based on thickening state
            if bspline_is_thickened:
                curve_color = COLOR_THICKENED_BSPLINE_CURVE
                control_color = COLOR_THICKENED_BSPLINE_CONTROL_POINTS
                control_symbol = COLOR_THICKENED_BSPLINE_CONTROL_SYMBOL
                control_symbol_pen = COLOR_THICKENED_BSPLINE_CONTROL_SYMBOL_PEN
                curve_name_prefix = "Thickened B-spline"
            else:
                curve_color = COLOR_BSPLINE_CURVE
                control_color = COLOR_BSPLINE_CONTROL_POINTS
                control_symbol = COLOR_BSPLINE_CONTROL_SYMBOL
                control_symbol_pen = COLOR_BSPLINE_CONTROL_SYMBOL_PEN
                curve_name_prefix = "B-spline"
            
            self.plot_items["B-spline Curves"] = []
            self.plot_items["B-spline Control Points"] = []

            # Plot upper B-spline curve
            if bspline_upper_curve is not None:
                # Sample the B-spline curve
                t_vals = np.linspace(0, 1, PLOT_POINTS_PER_SURFACE)
                bspline_upper_points = bspline_upper_curve(t_vals)
                
                self.plot_items["B-spline Curves"].append(
                    self.plot(
                        bspline_upper_points[:, 0],
                        bspline_upper_points[:, 1],
                        pen=curve_color,
                        antialias=True,
                        name=f"{curve_name_prefix} Upper",
                    )
                )

            # Plot lower B-spline curve
            if bspline_lower_curve is not None:
                # Sample the B-spline curve
                t_vals = np.linspace(0, 1, PLOT_POINTS_PER_SURFACE)
                bspline_lower_points = bspline_lower_curve(t_vals)
                
                self.plot_items["B-spline Curves"].append(
                    self.plot(
                        bspline_lower_points[:, 0],
                        bspline_lower_points[:, 1],
                        pen=curve_color,
                        antialias=True,
                        name=f"{curve_name_prefix} Lower",
                    )
                )

            # Plot upper control points
            if bspline_upper_control_points is not None:
                self.plot_items["B-spline Control Points"].append(
                    self.plot(
                        bspline_upper_control_points[:, 0],
                        bspline_upper_control_points[:, 1],
                        pen=control_color,
                        symbol="s",
                        symbolSize=6,
                        symbolBrush=control_symbol,
                        symbolPen=control_symbol_pen,
                        name=f"{curve_name_prefix} Control Points Upper",
                    )
                )

            # Plot lower control points
            if bspline_lower_control_points is not None:
                self.plot_items["B-spline Control Points"].append(
                    self.plot(
                        bspline_lower_control_points[:, 0],
                        bspline_lower_control_points[:, 1],
                        pen=control_color,
                        symbol="s",
                        symbolSize=6,
                        symbolBrush=control_symbol,
                        symbolPen=control_symbol_pen,
                        name=f"{curve_name_prefix} Control Points Lower",
                    )
                )

        # --------------------------------------------------------------
        # 3) B-spline Curvature comb
        # --------------------------------------------------------------
        if comb_bspline is not None and any(comb_bspline):
            all_bspline_comb_hairs: list[np.ndarray] = []
            all_bspline_comb_tips_segments: list[np.ndarray] = []

            for comb_segments in comb_bspline:
                if not comb_segments:
                    continue
                all_bspline_comb_hairs.extend(comb_segments)

                comb_tips = np.array([hair[1] for hair in comb_segments])
                for j in range(len(comb_tips) - 1):
                    p1 = comb_tips[j]
                    p2 = comb_tips[j + 1]
                    if p1[1] != 0 or p2[1] != 0:
                        all_bspline_comb_tips_segments.append(p1)
                        all_bspline_comb_tips_segments.append(p2)

            if all_bspline_comb_hairs:
                comb_array = np.concatenate(all_bspline_comb_hairs)
                main_bspline_comb_item = self.plot(
                    comb_array[:, 0],
                    comb_array[:, 1],
                    pen=COLOR_BSPLINE_COMB,
                    name="B-spline Curvature Comb",
                    connect="pairs",
                )
                self.plot_items["B-spline Curvature Comb"] = [main_bspline_comb_item]

                if all_bspline_comb_tips_segments:
                    segments_array = np.array(all_bspline_comb_tips_segments)
                    tips_item = self.plot(
                        segments_array[:, 0],
                        segments_array[:, 1],
                        pen=COLOR_COMB_OUTLINE,
                        connect="pairs",
                    )
                    self.plot_items["B-spline Comb Tips Polyline"] = [tips_item]

                    main_bspline_comb_item.visibleChanged.connect(
                        lambda: tips_item.setVisible(main_bspline_comb_item.isVisible())
                    )
                    tips_item.setVisible(main_bspline_comb_item.isVisible())

        # --------------------------------------------------------------
        # 4) Trailing-edge tangent vectors (only once)
        # --------------------------------------------------------------
        tangent_length = 0.05
        if upper_te_tangent_vector is not None and lower_te_tangent_vector is not None:
            if len(upper_data) > 0 and len(lower_data) > 0:
                upper_te_point = upper_data[-1]
                lower_te_point = lower_data[-1]

                tangent_start_upper = upper_te_point - upper_te_tangent_vector * tangent_length
                tangent_end_upper = upper_te_point + upper_te_tangent_vector * tangent_length
                self.plot_items["TE Tangent (Upper)"] = self.plot(
                    [tangent_start_upper[0], tangent_end_upper[0]],
                    [tangent_start_upper[1], tangent_end_upper[1]],
                    pen=COLOR_TE_TANGENT_UPPER,
                    name="TE Tangent (Upper)",
                )

                tangent_start_lower = lower_te_point - lower_te_tangent_vector * tangent_length
                tangent_end_lower = lower_te_point + lower_te_tangent_vector * tangent_length
                self.plot_items["TE Tangent (Lower)"] = self.plot(
                    [tangent_start_lower[0], tangent_end_lower[0]],
                    [tangent_start_lower[1], tangent_end_lower[1]],
                    pen=COLOR_TE_TANGENT_LOWER,
                    name="TE Tangent (Lower)",
                )

        # --------------------------------------------------------------
        # 5) Error text & markers
        # --------------------------------------------------------------
        # B-spline error tracking
        max_bspline_upper = bspline_upper_max_error
        max_bspline_lower = bspline_lower_max_error
        max_bspline_upper_idx = bspline_upper_max_error_idx
        max_bspline_lower_idx = bspline_lower_max_error_idx
        
        # Show error labels for individual surfaces or both surfaces
        error_texts = []
        
        # B-spline error text
        if chord_length_mm is not None and (max_bspline_upper is not None or max_bspline_lower is not None):
            error_html = '<div style="text-align: right; color: #FF6B6B; font-size: 10pt;">'
            
            if max_bspline_upper is not None and max_bspline_lower is not None:
                # Both surfaces available
                max_upper_mm = max_bspline_upper * chord_length_mm
                max_lower_mm = max_bspline_lower * chord_length_mm
                error_html += f"B-spline Max Error (Upper/Lower): {max_bspline_upper:.2e} ({max_upper_mm:.3f} mm) / {max_bspline_lower:.2e} ({max_lower_mm:.3f} mm)"
            elif max_bspline_upper is not None:
                # Only upper surface available
                max_upper_mm = max_bspline_upper * chord_length_mm
                error_html += f"B-spline Max Error (Upper): {max_bspline_upper:.2e} ({max_upper_mm:.3f} mm)"
            elif max_bspline_lower is not None:
                # Only lower surface available
                max_lower_mm = max_bspline_lower * chord_length_mm
                error_html += f"B-spline Max Error (Lower): {max_bspline_lower:.2e} ({max_lower_mm:.3f} mm)"
            
            error_html += "</div>"
            text_item = pg.TextItem(html=error_html, anchor=(1, 1))
            self.addItem(text_item)
            self.plot_items["B-spline Error Text"] = text_item
            error_texts.append(text_item)

        # Worst-error markers
        x_err: list[float] = []
        y_err: list[float] = []
        marker_colors: list[tuple] = []
        
        # B-spline error markers (orange)
        if max_bspline_upper_idx is not None and 0 <= max_bspline_upper_idx < len(upper_data):
            pt = upper_data[max_bspline_upper_idx]
            x_err.append(pt[0])
            y_err.append(pt[1])
            marker_colors.append((255, 165, 0))  # Orange for B-spline
        if max_bspline_lower_idx is not None and 0 <= max_bspline_lower_idx < len(lower_data):
            pt = lower_data[max_bspline_lower_idx]
            x_err.append(pt[0])
            y_err.append(pt[1])
            marker_colors.append((255, 165, 0))  # Orange for B-spline
        
        if x_err:
            # Create separate marker items for different colors
            orange_x = [x for i, x in enumerate(x_err) if marker_colors[i] == (255, 165, 0)]
            orange_y = [y for i, y in enumerate(y_err) if marker_colors[i] == (255, 165, 0)]
            
            if orange_x:
                bspline_marker_item = self.plot(
                    orange_x,
                    orange_y,
                    pen=None,
                    symbol="s",
                    symbolSize=16,
                    symbolBrush=None,
                    symbolPen=pg.mkPen((255, 165, 0), width=3),
                    name="B-spline Max. Error Markers",
                )
                self.plot_items["B-spline Max. Error Markers"] = bspline_marker_item

        # --------------------------------------------------------------
        # 6) Geometry metrics panel (bottom-right)
        # --------------------------------------------------------------
        # Build and add the text item if available
        if geometry_metrics:
            t_pct = geometry_metrics.get('thickness_percent')
            c_pct = geometry_metrics.get('camber_percent')
            wedge = geometry_metrics.get('te_wedge_angle_deg')
            le_r = geometry_metrics.get('le_radius_percent')
            x_t = geometry_metrics.get('x_t_percent', 0.0)
            x_c = geometry_metrics.get('x_c_percent', 0.0)
            geo_html = (
                '<div style="text-align: right; color: #F0E68C; font-size: 10pt;">'
                f"Thickness: {t_pct:.2f}% (x: {x_t:.1f}%)<br/>"
                f"Camber: {c_pct:.2f}% (x: {x_c:.1f}%)<br/>"
                f"TE wedge: {wedge:.2f}°<br/>"
                f"LE Radius: {le_r:.3f}%"
                "</div>"
            )
            # Anchor at bottom-right so the text stays fully inside the viewport
            geo_item = pg.TextItem(html=geo_html, anchor=(1, 1))
            self.addItem(geo_item)
            self.plot_items["Geometry Metrics Text"] = geo_item

        self._update_error_text_positions()

        # --------------------------------------------------------------
        # 7) Initial view range
        # --------------------------------------------------------------
        if not self._first_plot_done:
            all_x = np.concatenate([upper_data[:, 0], lower_data[:, 0]])
            all_y = np.concatenate([upper_data[:, 1], lower_data[:, 1]])

            x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
            y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.2

            self.setXRange(x_min - x_padding, x_max + x_padding)
            self.setYRange(y_min - y_padding, y_max + y_padding)
            self._first_plot_done = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _update_error_text_positions(self):
        """Keep error text anchored to the top-right corner on zoom/pan."""
        vb = self.getViewBox()
        if not vb:
            return

        x_range, y_range = vb.viewRange()
        x_padding = (x_range[1] - x_range[0]) * 0.04
        y_padding = (y_range[1] - y_range[0]) * 0.08

        top_right_x = x_range[1] - x_padding
        current_y = y_range[1] - y_padding

        text_bspline = self.plot_items.get("B-spline Error Text")
        text_geo = self.plot_items.get("Geometry Metrics Text")

        y_offset = (y_range[1] - y_range[0]) * 0.06
        if text_bspline:
            text_bspline.setPos(top_right_x, current_y)
            current_y -= y_offset
        if text_geo:
            # Position slightly above the bottom to avoid being clipped
            bottom_right_y = y_range[0] + (y_range[1] - y_range[0]) * 0.02
            text_geo.setPos(top_right_x, bottom_right_y)