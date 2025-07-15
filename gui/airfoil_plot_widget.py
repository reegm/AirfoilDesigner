import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from utils.bezier_utils import general_bezier_curve

class AirfoilPlotWidget(pg.PlotWidget):
    """A custom PlotWidget for embedding PyQtGraph plots in a PySide6 application."""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setParent(parent)
        self.setSizePolicy(pg.QtWidgets.QSizePolicy.Policy.Expanding, pg.QtWidgets.QSizePolicy.Policy.Expanding)
        self.updateGeometry()

        # Initial plot settings
        self.setAspectLocked(True) # Maintain aspect ratio
        self.showGrid(x=True, y=True) # Show grid
        self.addLegend(offset=(30, 10)) # Add legend with some offset
        self.setLabel('bottom', "x/c (Chord)")
        self.setLabel('left', "y/c (Chord)")

        self.plot_items = {} # Stores references to all plot items
        self._first_plot_done = False # Flag to control initial zoom
        self.getViewBox().sigRangeChanged.connect(self._update_error_text_positions)

    def plot_airfoil(self, upper_data, lower_data, model_polygons_sharp=None,
                     thickened_model_polygons=None,
                     single_bezier_upper_poly=None, single_bezier_lower_poly=None,
                     thickened_single_bezier_upper_poly=None, thickened_single_bezier_lower_poly=None,
                     upper_te_tangent_vector=None, lower_te_tangent_vector=None,
                     worst_single_bezier_upper_max_error=None, worst_single_bezier_lower_max_error=None,
                     worst_single_bezier_upper_max_error_idx=None, worst_single_bezier_lower_max_error_idx=None,
                     comb_single_bezier=None,
                     chord_length_mm=None):
        self.clear() # Clear all items to ensure no remnants
        self.addLegend(offset=(30, 10)) # Re-add legend after clearing
        self.plot_items = {} # Clear stored items

        # Define colors for plotting
        COLOR_ORIGINAL_DATA = (135, 206, 250, 200) # SkyBlue
        
        COLOR_SINGLE_UPPER_CURVE = pg.mkPen('blue', width=2.0)
        COLOR_SINGLE_LOWER_CURVE = pg.mkPen('cyan', width=2.0)
        COLOR_SINGLE_POLYGON = pg.mkPen((0, 0, 200), width=1, style=Qt.PenStyle.DashLine) # Darker Blue
        COLOR_SINGLE_SYMBOL = pg.mkBrush((100, 100, 255, 200)) # Brighter Blue
        COLOR_SINGLE_SYMBOL_PEN = pg.mkPen((80, 80, 200)) # Darker Blue

        COLOR_THICKENED_CURVE = pg.mkPen('darkorange', width=2.5)
        COLOR_THICKENED_POLYGON = pg.mkPen((255, 165, 0), width=1, style=Qt.PenStyle.DotLine) # Orange
        COLOR_THICKENED_SYMBOL = pg.mkBrush((255, 100, 0, 200)) # Brighter Orange
        COLOR_THICKENED_SYMBOL_PEN = pg.mkPen((200, 80, 0)) # Darker Orange

        COLOR_TE_TANGENT_UPPER = pg.mkPen('red', width=2, style=Qt.PenStyle.SolidLine)
        COLOR_TE_TANGENT_LOWER = pg.mkPen('purple', width=2, style=Qt.PenStyle.SolidLine)

        # Comb colors
        COLOR_SINGLE_BEZIER_COMB = pg.mkPen((220, 220, 220), width=1) # Very light grey
        COLOR_COMB_OUTLINE = pg.mkPen('yellow', width=2, style=Qt.PenStyle.DotLine)

        # Plot Original Data points
        self.plot_items['Original Data'] = [
            self.plot(upper_data[:, 0], upper_data[:, 1], pen=None, symbol='o', symbolSize=5,
                      symbolBrush=pg.mkBrush(COLOR_ORIGINAL_DATA), name='Original Data (Upper)'),
            self.plot(lower_data[:, 0], lower_data[:, 1], pen=None, symbol='o', symbolSize=5,
                      symbolBrush=pg.mkBrush(COLOR_ORIGINAL_DATA), name='Original Data (Lower)')
        ]

        # Plot Single Bezier Curves (prioritize thickened if available)
        if thickened_single_bezier_upper_poly is not None and thickened_single_bezier_lower_poly is not None:
            curves_thickened_single_upper = general_bezier_curve(np.linspace(0, 1, 100), np.array(thickened_single_bezier_upper_poly))
            curves_thickened_single_lower = general_bezier_curve(np.linspace(0, 1, 100), np.array(thickened_single_bezier_lower_poly))

            self.plot_items['Thickened Single Bezier Airfoil'] = [
                self.plot(curves_thickened_single_upper[:, 0], curves_thickened_single_upper[:, 1],
                          pen=COLOR_THICKENED_CURVE, name='Thickened Single Bezier Airfoil - Upper'),
                self.plot(curves_thickened_single_lower[:, 0], curves_thickened_single_lower[:, 1],
                          pen=COLOR_THICKENED_CURVE, name='Thickened Single Bezier Airfoil - Lower')
            ]
            self.plot_items['Control Polygons (Thickened Single)'] = []
            for p_idx, p in enumerate([thickened_single_bezier_upper_poly, thickened_single_bezier_lower_poly]):
                item = self.plot(np.array(p)[:, 0], np.array(p)[:, 1],
                          pen=COLOR_THICKENED_POLYGON, symbol='x', symbolSize=7,
                          symbolBrush=COLOR_THICKENED_SYMBOL, symbolPen=COLOR_THICKENED_SYMBOL_PEN,
                          name=f'Control Poly Thickened Single {p_idx+1}')
                self.plot_items['Control Polygons (Thickened Single)'].append(item)

        elif single_bezier_upper_poly is not None and single_bezier_lower_poly is not None:
            curves_single_upper = general_bezier_curve(np.linspace(0, 1, 100), np.array(single_bezier_upper_poly))
            curves_single_lower = general_bezier_curve(np.linspace(0, 1, 100), np.array(single_bezier_lower_poly))

            self.plot_items['Single Bezier Airfoil'] = [
                self.plot(curves_single_upper[:, 0], curves_single_upper[:, 1],
                          pen=COLOR_SINGLE_UPPER_CURVE, name=f'Single Bezier Airfoil (Order 9) - Upper'),
                self.plot(curves_single_lower[:, 0], curves_single_lower[:, 1],
                          pen=COLOR_SINGLE_LOWER_CURVE, name=f'Single Bezier Airfoil (Order 9) - Lower')
            ]
            self.plot_items['Control Polygons (Single Bezier)'] = []
            for p_idx, p in enumerate([single_bezier_upper_poly, single_bezier_lower_poly]):
                item = self.plot(np.array(p)[:, 0], np.array(p)[:, 1],
                          pen=COLOR_SINGLE_POLYGON, symbol='x', symbolSize=7,
                          symbolBrush=COLOR_SINGLE_SYMBOL, symbolPen=COLOR_SINGLE_SYMBOL_PEN,
                          name=f'Control Poly Single {p_idx+1}')
                self.plot_items['Control Polygons (Single Bezier)'].append(item)

        # Plot Curvature Comb for Single Bezier model
        if comb_single_bezier is not None and any(comb_single_bezier):
            self.plot_items['Single Bezier Curvature Comb'] = []
            self.plot_items['Comb Tips Polyline'] = []

            for i, comb_segments in enumerate(comb_single_bezier):
                if not comb_segments:
                    continue

                # --- Plot comb hairs ---
                comb_array = np.concatenate(comb_segments)
                comb_item = self.plot(
                    comb_array[:, 0], comb_array[:, 1],
                    pen=COLOR_SINGLE_BEZIER_COMB,
                    name=f'Single Bezier Curvature Comb {i+1}',
                    connect='pairs'
                )
                self.plot_items['Single Bezier Curvature Comb'].append(comb_item)

                # --- Plot polyline connecting tips ---
                comb_tips = np.array([hair[1] for hair in comb_segments])
                segments_to_plot = []
                for j in range(len(comb_tips) - 1):
                    p1 = comb_tips[j]
                    p2 = comb_tips[j+1]
                    if p1[1] != 0 or p2[1] != 0:
                        segments_to_plot.append(p1)
                        segments_to_plot.append(p2)
                
                if segments_to_plot:
                    segments_array = np.array(segments_to_plot)
                    comb_tips_item = self.plot(
                        segments_array[:, 0], segments_array[:, 1],
                        pen=COLOR_COMB_OUTLINE,
                        name=f'Comb Tips Polyline {i+1}',
                        connect='pairs'
                    )
                    self.plot_items['Comb Tips Polyline'].append(comb_tips_item)

        # Plot Trailing Edge Tangent Vectors (only once, as they are derived from original data)
        tangent_length = 0.05

        if upper_te_tangent_vector is not None and lower_te_tangent_vector is not None:
            # Use the original data's trailing edge points for tangent visualization
            if len(upper_data) > 0 and len(lower_data) > 0:
                upper_te_point = upper_data[-1]
                lower_te_point = lower_data[-1]

                tangent_start_upper = upper_te_point - upper_te_tangent_vector * tangent_length
                tangent_end_upper = upper_te_point + upper_te_tangent_vector * tangent_length
                self.plot_items['TE Tangent (Upper)'] = self.plot(
                    [tangent_start_upper[0], tangent_end_upper[0]],
                    [tangent_start_upper[1], tangent_end_upper[1]],
                    pen=COLOR_TE_TANGENT_UPPER, name='TE Tangent (Upper)')

                tangent_start_lower = lower_te_point - lower_te_tangent_vector * tangent_length
                tangent_end_lower = lower_te_point + lower_te_tangent_vector * tangent_length
                self.plot_items['TE Tangent (Lower)'] = self.plot(
                    [tangent_start_lower[0], tangent_end_lower[0]],
                    [tangent_start_lower[1], tangent_end_lower[1]],
                    pen=COLOR_TE_TANGENT_LOWER, name='TE Tangent (Lower)')

        # Display errors
        
        # For single Bezier model
        max_single_upper = worst_single_bezier_upper_max_error
        max_single_lower = worst_single_bezier_lower_max_error
        max_single_upper_idx = worst_single_bezier_upper_max_error_idx
        max_single_lower_idx = worst_single_bezier_lower_max_error_idx
        if (max_single_upper is not None and max_single_lower is not None and chord_length_mm is not None):
            error_html = '<div style="text-align: right; color: #00BFFF; font-size: 10pt;">'
            max_upper_mm = max_single_upper * chord_length_mm
            max_lower_mm = max_single_lower * chord_length_mm
            error_html += f'Max Error (U/L): {max_upper_mm:.3f} mm / {max_lower_mm:.3f} mm'
            error_html += '</div>'
            text_item = pg.TextItem(html=error_html, anchor=(1, 1))
            self.addItem(text_item)
            self.plot_items['Single Bezier Error Text'] = text_item
            
        # --- collect worst‑error points ---
        x_err, y_err = [], []
        if max_single_upper_idx is not None:
            pt = upper_data[max_single_upper_idx]
            x_err.append(pt[0]);  y_err.append(pt[1])

        if max_single_lower_idx is not None:
            pt = lower_data[max_single_lower_idx]
            x_err.append(pt[0]);  y_err.append(pt[1])

        # --- one PlotDataItem for both markers ---
        if x_err:
            marker_item = self.plot(
                x_err, y_err,
                pen=None,
                symbol='o', symbolSize=16,
                symbolBrush=None,
                symbolPen=pg.mkPen((255, 0, 0), width=3),
                name='Max Error Markers'      # single legend entry
            )
            self.plot_items['Max Error Markers'] = marker_item

        self._update_error_text_positions()

        # Set view limits only on the first plot or when a new file is loaded
        if not self._first_plot_done:
            all_x = np.concatenate([upper_data[:, 0], lower_data[:, 0]])
            all_y = np.concatenate([upper_data[:, 1], lower_data[:, 1]])

            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)

            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.2

            self.setXRange(x_min - x_padding, x_max + x_padding)
            self.setYRange(y_min - y_padding, y_max + y_padding)
            self._first_plot_done = True

    def _update_error_text_positions(self):
        """Updates the position of the error text items to keep them in the top-right corner."""
        vb = self.getViewBox()
        if not vb: return

        x_range, y_range = vb.viewRange()

        # Define padding from the edge as a fraction of the view range
        x_padding = (x_range[1] - x_range[0]) * 0.04
        y_padding = (y_range[1] - y_range[0]) * 0.08

        # Calculate top-right position
        top_right_x = x_range[1] - x_padding
        current_y = y_range[1] - y_padding

        text_4_seg = self.plot_items.get('4-Seg Error Text')
        text_single = self.plot_items.get('Single Bezier Error Text')

        y_offset_between_texts = (y_range[1] - y_range[0]) * 0.06

        if text_4_seg:
            text_4_seg.setPos(top_right_x, current_y)
            current_y -= y_offset_between_texts

        if text_single:
            text_single.setPos(top_right_x, current_y) 