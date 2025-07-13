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
                     worst_4_seg_error=None, worst_4_seg_error_mse=None, worst_4_seg_error_icp=None,
                     worst_single_bezier_upper_error=None, worst_single_bezier_upper_error_mse=None, worst_single_bezier_upper_error_icp=None,
                     worst_single_bezier_lower_error=None, worst_single_bezier_lower_error_mse=None, worst_single_bezier_lower_error_icp=None,
                     comb_4_segment=None, comb_single_bezier=None):
        self.clear() # Clear all items to ensure no remnants
        self.addLegend(offset=(30, 10)) # Re-add legend after clearing
        self.plot_items = {} # Clear stored items

        # Define colors for plotting
        COLOR_ORIGINAL_DATA = (135, 206, 250, 200) # SkyBlue
        COLOR_UPPER_4SEG_CURVE = pg.mkPen((0, 200, 0), width=1.5) # Bright Green
        COLOR_LOWER_4SEG_CURVE = pg.mkPen((200, 0, 200), width=1.5) # Magenta
        COLOR_4SEG_POLYGON = pg.mkPen((150, 150, 150), width=1, style=Qt.PenStyle.DotLine) # Lighter Gray
        COLOR_4SEG_SYMBOL = pg.mkBrush((255, 255, 0, 200)) # Yellow
        COLOR_4SEG_SYMBOL_PEN = pg.mkPen((200, 200, 0)) # Darker Yellow

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

        COLOR_4SEG_COMB = pg.mkPen((100, 255, 100), width=1) # Light Green
        COLOR_SINGLE_BEZIER_COMB = pg.mkPen((100, 150, 255), width=1) # Light Blue

        # Plot Original Data points
        self.plot_items['Original Data'] = [
            self.plot(upper_data[:, 0], upper_data[:, 1], pen=None, symbol='o', symbolSize=5,
                      symbolBrush=pg.mkBrush(COLOR_ORIGINAL_DATA), name='Original Data (Upper)'),
            self.plot(lower_data[:, 0], lower_data[:, 1], pen=None, symbol='o', symbolSize=5,
                      symbolBrush=pg.mkBrush(COLOR_ORIGINAL_DATA), name='Original Data (Lower)')
        ]

        # Plot 4-segment model (prioritize thickened if available)
        if thickened_model_polygons is not None:
            curves_thickened_upper = general_bezier_curve(np.linspace(0, 1, 100), np.array(thickened_model_polygons[0]))
            curves_thickened_upper_rear = general_bezier_curve(np.linspace(0, 1, 100), np.array(thickened_model_polygons[1]))
            curves_thickened_lower = general_bezier_curve(np.linspace(0, 1, 100), np.array(thickened_model_polygons[2]))
            curves_thickened_lower_rear = general_bezier_curve(np.linspace(0, 1, 100), np.array(thickened_model_polygons[3]))

            self.plot_items['Thickened 4-Segment Airfoil'] = [
                self.plot(np.vstack([curves_thickened_upper, curves_thickened_upper_rear])[:, 0],
                          np.vstack([curves_thickened_upper, curves_thickened_upper_rear])[:, 1],
                          pen=COLOR_THICKENED_CURVE, name='Thickened 4-Segment Airfoil - Upper'),
                self.plot(np.vstack([curves_thickened_lower, curves_thickened_lower_rear])[:, 0],
                          np.vstack([curves_thickened_lower, curves_thickened_lower_rear])[:, 1],
                          pen=COLOR_THICKENED_CURVE, name='Thickened 4-Segment Airfoil - Lower')
            ]
            self.plot_items['Control Polygons (Thickened 4-Seg)'] = []
            for p_idx, p in enumerate(thickened_model_polygons):
                item = self.plot(np.array(p)[:, 0], np.array(p)[:, 1],
                          pen=COLOR_THICKENED_POLYGON, symbol='s', symbolSize=6,
                          symbolBrush=COLOR_THICKENED_SYMBOL, symbolPen=COLOR_THICKENED_SYMBOL_PEN,
                          name=f'Control Poly Thickened 4-Seg {p_idx+1}')
                self.plot_items['Control Polygons (Thickened 4-Seg)'].append(item)

        elif model_polygons_sharp is not None:
            curves_sharp_upper = general_bezier_curve(np.linspace(0, 1, 100), np.array(model_polygons_sharp[0]))
            curves_sharp_upper_rear = general_bezier_curve(np.linspace(0, 1, 100), np.array(model_polygons_sharp[1]))
            curves_sharp_lower = general_bezier_curve(np.linspace(0, 1, 100), np.array(model_polygons_sharp[2]))
            curves_sharp_lower_rear = general_bezier_curve(np.linspace(0, 1, 100), np.array(model_polygons_sharp[3]))

            self.plot_items['Optimized Sharp Airfoil'] = [
                self.plot(np.vstack([curves_sharp_upper, curves_sharp_upper_rear])[:, 0],
                          np.vstack([curves_sharp_upper, curves_sharp_upper_rear])[:, 1],
                          pen=COLOR_UPPER_4SEG_CURVE, name='Optimized Sharp Airfoil (4 segments) - Upper'),
                self.plot(np.vstack([curves_sharp_lower, curves_sharp_lower_rear])[:, 0],
                          np.vstack([curves_sharp_lower, curves_sharp_lower_rear])[:, 1],
                          pen=COLOR_LOWER_4SEG_CURVE, name='Optimized Sharp Airfoil (4 segments) - Lower')
            ]
            self.plot_items['Control Polygons (Sharp)'] = []
            for p_idx, p in enumerate(model_polygons_sharp):
                item = self.plot(np.array(p)[:, 0], np.array(p)[:, 1],
                          pen=COLOR_4SEG_POLYGON, symbol='s', symbolSize=6,
                          symbolBrush=COLOR_4SEG_SYMBOL, symbolPen=COLOR_4SEG_SYMBOL_PEN,
                          name=f'Control Poly Sharp {p_idx+1}')
                self.plot_items['Control Polygons (Sharp)'].append(item)

        # Plot Curvature Comb for 4-segment model
        if comb_4_segment is not None and len(comb_4_segment) > 0:
            # Reshape the list of hair segments into a single array for efficient plotting
            comb_array = np.concatenate(comb_4_segment)
            # Plot all hairs as a single item with disconnected lines
            comb_item = self.plot(
                comb_array[:, 0], comb_array[:, 1],
                pen=COLOR_4SEG_COMB, name='4-Seg Curvature Comb',
                connect='pairs'
            )
            self.plot_items['4-Segment Curvature Comb'] = comb_item

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
        if comb_single_bezier is not None and len(comb_single_bezier) > 0:
            # Reshape the list of hair segments into a single array
            comb_array = np.concatenate(comb_single_bezier)
            # Plot all hairs as a single item with disconnected lines
            comb_item = self.plot(
                comb_array[:, 0], comb_array[:, 1],
                pen=COLOR_SINGLE_BEZIER_COMB, name='Single Bezier Curvature Comb',
                connect='pairs'
            )
            self.plot_items['Single Bezier Curvature Comb'] = comb_item

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
        # Unused parameters are kept for API compatibility but silenced for linters.
        _ = (worst_4_seg_error, worst_single_bezier_upper_error, worst_single_bezier_lower_error)

        # For 4-segment model
        mse_4seg = worst_4_seg_error_mse
        icp_4seg = worst_4_seg_error_icp
        if mse_4seg is not None or icp_4seg is not None:
            error_html = '<div style="text-align: right; color: #00C800; font-size: 10pt;">'
            if mse_4seg is not None:
                error_html += f'MSE 4-Seg: {mse_4seg:.2e}<br>'
            if icp_4seg is not None:
                error_html += f'ICP 4-Seg: {icp_4seg:.2e}'
            error_html += '</div>'
            text_item = pg.TextItem(html=error_html, anchor=(1, 1))
            self.addItem(text_item)
            self.plot_items['4-Seg Error Text'] = text_item

        # For single Bezier model
        mse_single_upper = worst_single_bezier_upper_error_mse
        icp_single_upper = worst_single_bezier_upper_error_icp
        mse_single_lower = worst_single_bezier_lower_error_mse
        icp_single_lower = worst_single_bezier_lower_error_icp
        if (mse_single_upper is not None or icp_single_upper is not None or mse_single_lower is not None or icp_single_lower is not None):
            error_html = '<div style="text-align: right; color: #00BFFF; font-size: 10pt;">'
            if mse_single_upper is not None and mse_single_lower is not None:
                error_html += f'MSE Single Bezier (U/L): {mse_single_upper:.2e} / {mse_single_lower:.2e}<br>'
            if icp_single_upper is not None and icp_single_lower is not None:
                error_html += f'ICP Single Bezier (U/L): {icp_single_upper:.2e} / {icp_single_lower:.2e}'
            error_html += '</div>'
            text_item = pg.TextItem(html=error_html, anchor=(1, 1))
            self.addItem(text_item)
            self.plot_items['Single Bezier Error Text'] = text_item

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