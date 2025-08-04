"""Interactive airfoil visualisation widget.

This is the *actual* implementation of :class:`AirfoilPlotWidget`, moved
from ``gui/airfoil_plot_widget.py`` into the new widgets package as part
of the GUI refactor (2025-07-17).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from utils.bezier_utils import general_bezier_curve


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
        model_polygons_sharp=None,
        thickened_model_polygons=None,
        single_bezier_upper_poly=None,
        single_bezier_lower_poly=None,
        thickened_single_bezier_upper_poly=None,
        thickened_single_bezier_lower_poly=None,
        upper_te_tangent_vector=None,
        lower_te_tangent_vector=None,
        worst_single_bezier_upper_max_error=None,
        worst_single_bezier_lower_max_error=None,
        worst_single_bezier_upper_max_error_idx=None,
        worst_single_bezier_lower_max_error_idx=None,
        comb_single_bezier=None,
        chord_length_mm=None,
    ):
        """Render everything supplied in *kwargs* on the canvas."""
        # Clear all items to ensure no remnants
        self.clear()
        self.addLegend(offset=(30, 10))  # Re-add legend after clearing
        self.plot_items = {}

        # Colours -------------------------------------------------------
        COLOR_ORIGINAL_DATA = (135, 206, 250, 200)  # SkyBlue
        COLOR_SINGLE_UPPER_CURVE = pg.mkPen("blue", width=2.0)
        COLOR_SINGLE_LOWER_CURVE = pg.mkPen("cyan", width=2.0)

        COLOR_SINGLE_POLYGON = pg.mkPen((0, 150, 255), width=2, style=Qt.PenStyle.DashLine)
        COLOR_SINGLE_SYMBOL = pg.mkBrush((0, 200, 255, 255))
        COLOR_SINGLE_SYMBOL_PEN = pg.mkPen((0, 150, 255), width=2)

        COLOR_SINGLE_LOWER_POLYGON = pg.mkPen((100, 200, 255), width=2, style=Qt.PenStyle.DashLine)
        COLOR_SINGLE_LOWER_SYMBOL = pg.mkBrush((100, 220, 255, 255))
        COLOR_SINGLE_LOWER_SYMBOL_PEN = pg.mkPen((100, 200, 255), width=2)

        COLOR_THICKENED_CURVE = pg.mkPen("darkorange", width=2.5)

        COLOR_THICKENED_POLYGON = pg.mkPen((255, 165, 0), width=2, style=Qt.PenStyle.DotLine)
        COLOR_THICKENED_SYMBOL = pg.mkBrush((255, 165, 0, 255))
        COLOR_THICKENED_SYMBOL_PEN = pg.mkPen((255, 165, 0), width=2)

        COLOR_THICKENED_LOWER_POLYGON = pg.mkPen((255, 200, 0), width=2, style=Qt.PenStyle.DotLine)
        COLOR_THICKENED_LOWER_SYMBOL = pg.mkBrush((255, 200, 0, 255))
        COLOR_THICKENED_LOWER_SYMBOL_PEN = pg.mkPen((255, 200, 0), width=2)

        COLOR_TE_TANGENT_UPPER = pg.mkPen("red", width=2, style=Qt.PenStyle.SolidLine)
        COLOR_TE_TANGENT_LOWER = pg.mkPen("purple", width=2, style=Qt.PenStyle.SolidLine)

        # Comb colours
        COLOR_SINGLE_BEZIER_COMB = pg.mkPen((220, 220, 220), width=1)
        COLOR_COMB_OUTLINE = pg.mkPen("yellow", width=2, style=Qt.PenStyle.DotLine)

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
        # 2) Single-Bezier curves (thickened has priority)
        # --------------------------------------------------------------
        if (
            thickened_single_bezier_upper_poly is not None
            and thickened_single_bezier_lower_poly is not None
        ):
            curves_thickened_single_upper = general_bezier_curve(
                np.linspace(0, 1, 100), np.array(thickened_single_bezier_upper_poly)
            )
            curves_thickened_single_lower = general_bezier_curve(
                np.linspace(0, 1, 100), np.array(thickened_single_bezier_lower_poly)
            )

            self.plot_items["Thickened Airfoil"] = [
                self.plot(
                    curves_thickened_single_upper[:, 0],
                    curves_thickened_single_upper[:, 1],
                    pen=COLOR_THICKENED_CURVE,
                    name="Thickened Airfoil - Upper",
                ),
                self.plot(
                    curves_thickened_single_lower[:, 0],
                    curves_thickened_single_lower[:, 1],
                    pen=COLOR_THICKENED_CURVE,
                    name="Thickened Airfoil - Lower",
                ),
            ]
            self.plot_items["Control Polygons (Thickened)"] = []
            for p_idx, p in enumerate(
                [thickened_single_bezier_upper_poly, thickened_single_bezier_lower_poly]
            ):
                if p_idx == 0:  # Upper
                    pen = COLOR_THICKENED_POLYGON
                    symbol_brush = COLOR_THICKENED_SYMBOL
                    symbol_pen = COLOR_THICKENED_SYMBOL_PEN
                else:  # Lower
                    pen = COLOR_THICKENED_LOWER_POLYGON
                    symbol_brush = COLOR_THICKENED_LOWER_SYMBOL
                    symbol_pen = COLOR_THICKENED_LOWER_SYMBOL_PEN
                item = self.plot(
                    np.array(p)[:, 0],
                    np.array(p)[:, 1],
                    pen=pen,
                    symbol="x",
                    symbolSize=7,
                    symbolBrush=symbol_brush,
                    symbolPen=symbol_pen,
                    name=f"Control Poly Thickened {p_idx + 1}",
                )
                self.plot_items["Control Polygons (Thickened)"].append(item)

        elif single_bezier_upper_poly is not None or single_bezier_lower_poly is not None:
            # Handle individual surfaces or both surfaces
            self.plot_items["Single Bezier Airfoil"] = []
            self.plot_items["Control Polygons (Single Bezier)"] = []
            
            # Plot upper surface if available
            if single_bezier_upper_poly is not None:
                curves_single_upper = general_bezier_curve(
                    np.linspace(0, 1, 100), np.array(single_bezier_upper_poly)
                )
                self.plot_items["Single Bezier Airfoil"].append(
                    self.plot(
                        curves_single_upper[:, 0],
                        curves_single_upper[:, 1],
                        pen=COLOR_SINGLE_UPPER_CURVE,
                        name="Single Bezier Airfoil - Upper",
                    )
                )
                
                # Add control polygon for upper surface
                item = self.plot(
                    np.array(single_bezier_upper_poly)[:, 0],
                    np.array(single_bezier_upper_poly)[:, 1],
                    pen=COLOR_SINGLE_POLYGON,
                    symbol="x",
                    symbolSize=7,
                    symbolBrush=COLOR_SINGLE_SYMBOL,
                    symbolPen=COLOR_SINGLE_SYMBOL_PEN,
                    name="Control Poly Single Upper",
                )
                self.plot_items["Control Polygons (Single Bezier)"].append(item)
            
            # Plot lower surface if available
            if single_bezier_lower_poly is not None:
                curves_single_lower = general_bezier_curve(
                    np.linspace(0, 1, 100), np.array(single_bezier_lower_poly)
                )
                self.plot_items["Single Bezier Airfoil"].append(
                    self.plot(
                        curves_single_lower[:, 0],
                        curves_single_lower[:, 1],
                        pen=COLOR_SINGLE_LOWER_CURVE,
                        name="Single Bezier Airfoil - Lower",
                    )
                )
                
                # Add control polygon for lower surface
                item = self.plot(
                    np.array(single_bezier_lower_poly)[:, 0],
                    np.array(single_bezier_lower_poly)[:, 1],
                    pen=COLOR_SINGLE_LOWER_POLYGON,
                    symbol="x",
                    symbolSize=7,
                    symbolBrush=COLOR_SINGLE_LOWER_SYMBOL,
                    symbolPen=COLOR_SINGLE_LOWER_SYMBOL_PEN,
                    name="Control Poly Single Lower",
                )
                self.plot_items["Control Polygons (Single Bezier)"].append(item)

        # --------------------------------------------------------------
        # 3) Curvature comb
        # --------------------------------------------------------------
        if comb_single_bezier is not None and any(comb_single_bezier):
            all_comb_hairs: list[np.ndarray] = []
            all_comb_tips_segments: list[np.ndarray] = []

            for comb_segments in comb_single_bezier:
                if not comb_segments:
                    continue
                all_comb_hairs.extend(comb_segments)

                comb_tips = np.array([hair[1] for hair in comb_segments])
                for j in range(len(comb_tips) - 1):
                    p1 = comb_tips[j]
                    p2 = comb_tips[j + 1]
                    if p1[1] != 0 or p2[1] != 0:
                        all_comb_tips_segments.append(p1)
                        all_comb_tips_segments.append(p2)

            if all_comb_hairs:
                comb_array = np.concatenate(all_comb_hairs)
                main_comb_item = self.plot(
                    comb_array[:, 0],
                    comb_array[:, 1],
                    pen=COLOR_SINGLE_BEZIER_COMB,
                    name="Curvature Comb",
                    connect="pairs",
                )
                self.plot_items["Curvature Comb"] = [main_comb_item]

                if all_comb_tips_segments:
                    segments_array = np.array(all_comb_tips_segments)
                    tips_item = self.plot(
                        segments_array[:, 0],
                        segments_array[:, 1],
                        pen=COLOR_COMB_OUTLINE,
                        connect="pairs",
                    )
                    self.plot_items["Comb Tips Polyline"] = [tips_item]

                    main_comb_item.visibleChanged.connect(
                        lambda: tips_item.setVisible(main_comb_item.isVisible())
                    )
                    tips_item.setVisible(main_comb_item.isVisible())

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
        max_single_upper = worst_single_bezier_upper_max_error
        max_single_lower = worst_single_bezier_lower_max_error
        max_single_upper_idx = worst_single_bezier_upper_max_error_idx
        max_single_lower_idx = worst_single_bezier_lower_max_error_idx
        
        # Show error labels for individual surfaces or both surfaces
        if chord_length_mm is not None and (max_single_upper is not None or max_single_lower is not None):
            error_html = '<div style="text-align: right; color: #00BFFF; font-size: 10pt;">'
            
            if max_single_upper is not None and max_single_lower is not None:
                # Both surfaces available
                max_upper_mm = max_single_upper * chord_length_mm
                max_lower_mm = max_single_lower * chord_length_mm
                error_html += f"Max Error (Upper/Lower): {max_single_upper:.2e} ({max_upper_mm:.3f} mm) / {max_single_lower:.2e} ({max_lower_mm:.3f} mm)"
            elif max_single_upper is not None:
                # Only upper surface available
                max_upper_mm = max_single_upper * chord_length_mm
                error_html += f"Max Error (Upper): {max_single_upper:.2e} ({max_upper_mm:.3f} mm)"
            elif max_single_lower is not None:
                # Only lower surface available
                max_lower_mm = max_single_lower * chord_length_mm
                error_html += f"Max Error (Lower): {max_single_lower:.2e} ({max_lower_mm:.3f} mm)"
            
            error_html += "</div>"
            text_item = pg.TextItem(html=error_html, anchor=(1, 1))
            self.addItem(text_item)
            self.plot_items["Single Bezier Error Text"] = text_item

        # Worst-error markers
        x_err: list[float] = []
        y_err: list[float] = []
        if max_single_upper_idx is not None and 0 <= max_single_upper_idx < len(upper_data):
            pt = upper_data[max_single_upper_idx]
            x_err.append(pt[0])
            y_err.append(pt[1])
        if max_single_lower_idx is not None and 0 <= max_single_lower_idx < len(lower_data):
            pt = lower_data[max_single_lower_idx]
            x_err.append(pt[0])
            y_err.append(pt[1])
        if x_err:
            marker_item = self.plot(
                x_err,
                y_err,
                pen=None,
                symbol="o",
                symbolSize=16,
                symbolBrush=None,
                symbolPen=pg.mkPen((255, 0, 0), width=3),
                name="Max. Error Markers",
            )
            self.plot_items["Max. Error Markers"] = marker_item

        self._update_error_text_positions()

        # --------------------------------------------------------------
        # 6) Initial view range
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

        text_4_seg = self.plot_items.get("4-Seg Error Text")
        text_single = self.plot_items.get("Single Bezier Error Text")

        y_offset = (y_range[1] - y_range[0]) * 0.06
        if text_4_seg:
            text_4_seg.setPos(top_right_x, current_y)
            current_y -= y_offset
        if text_single:
            text_single.setPos(top_right_x, current_y) 