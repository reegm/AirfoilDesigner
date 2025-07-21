# Airfoil Designer

Airfoil Designer is a Python-based utility for importing airfoil .dat files and exporting clean Bézier curves, based on Venkataraman’s work (1995, 2017)  
The graphical front‑end is implemented with Qt 6.

---

## Features

*   **Airfoil Analysis**: Load airfoil data from `.dat` files and visualize the airfoil shape.
*   **Bézier Curve Fitting**: Fit single or multiple Bézier curves to the airfoil data.
*   **Trailing Edge Thickening**: Apply trailing edge thickening to the airfoil.
*   **Curvature Combs**: Visualize the curvature of the airfoil using curvature combs.
*   **DXF Export**: Export the Bézier curves to a DXF file for use in CAD software.
*   **Fixed-x fitting strategy**: Uses the fixed x-coordinates from Venkataraman 2017 for the fastest fit.
*   **Variable-x fitting strategy**: Employs a full optimization loop that includes x-coordinates, often resulting in a better fit but requiring significantly more processing time.
*   **Euclidean and orthogonal error calculation**: Orthogonal usually produces better results, but comes with more processing time.
*   **Minmax optimization**: Optimizes the max error instead of the mean squared error. Should in theory improve the fit. Currently uses fixed-x and orthogonal error only.
*   **Enforce G2**: Combines upper and lower curve in a single optimization run and adds a constraint to enforce G2 across the leading edge. Usually comes with a slight degradation of the fit. Increases processing time considerably.

## Usage

1.  **Load Airfoil Data**: Click the `Load Airfoil` button to load a `.dat` file. Currently Selig and Ledniceer are supported.
2.  **Select Strategy**: Select fixed-x for speed, variable-x for accuracy, minmax for overkill. Results depend highly on the inpuit data, so experimenting with different combinations might improve your result. Higher settings might take several minutes to complete!
3. **Regularization Weight** Set optimizer penalty for uneven control point flow. Set to 0 for the best fit, 0.001 is a good starting point for a smooth control point flow.

4.  **Generate Airfoil**: Click the `Generate Airfoil` button to fit a Bézier curve to the airfoil data.
5.  **Toggle Thickening**: Click the `Toggle Thickening` button to apply or remove trailing edge thickening. 
6.  **Export DXF**: Click the `Export DXF` button to export the Bézier curves to a DXF file.
7.  **Graph Controls**: Click the icons in the graphs legend to toggle visibility. Use mouse to pan and zoom.

## Installation
* Make sure Python is installed on your system. 
* Unpack code .zip.
* Open a terminal in project root.
* Optional: Create virtual environment: 

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

* Install dependencies

```bash
pip install -r requirements.txt   # numpy, scipy, ezdxf, PySide6, pyqtgraph
```
Python 3.10 + is recommended. No compiled extensions are required.

---

## Quick start (GUI)

```bash
python run_gui.py
```
## File formats

* **Airfoil input:** Selig or Lednicer `.dat`, chord and orientation are normalized on import.  
* **CAD output:** DXF (millimetres). Separate layers are assigned to the upper and lower surfaces for clarity.

---

## Roadmap

* Investigate optimizing abcsisse for single segment model
  

Contributions are welcome.

---

## Citation


```text
Venkataraman, P. “A New Procedure for Airfoil Definition.” AIAA‑95‑1875, 1995.

Venkataraman, P. “A Bézier Parameterization Scheme for Flexible Airfoils.” 2017.
```

---

## License

Distributed under the MIT License.

