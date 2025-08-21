# Airfoil Fitter

Airfoil Fitter is a Python-based utility for importing airfoil .dat files and exporting clean B-Splines.  
The graphical front‑end is implemented with Qt 6.

---

## Features

*   **Airfoil Import**: Load airfoil data from `.dat` files and visualize the airfoil shape.
*   **B-Spline Fitting**: Fit B-Splines to the airfoil data.
*   **Trailing Edge Thickening**: Apply trailing edge thickening to the airfoil.
*   **Curvature Combs**: Check the curvature of the airfoil using curvature combs.
*   **DXF Export**: Export the B-Spline to a DXF file for use in CAD software.
*   **Enforce G2**: Combines upper and lower curve in a single optimization run and adds a constraint to enforce G2 across the leading edge. Usually comes with a slight degradation of the fit.

## Usage

1.  **Load Airfoil Data**: Click the `Load Airfoil` button to load a `.dat` file. Currently Selig and Ledniceer are supported.
2.  **Generate Airfoil**: Click the `Generate Airfoil` button to fit a Bézier curve to the airfoil data.
3.  **Toggle Thickening**: Click the `Toggle Thickening` button to apply or remove trailing edge thickening. 
4.  **Export DXF**: Click the `Export DXF` button to export the Bézier curves to a DXF file.
5.  **Graph Controls**: Click the icons in the graphs legend to toggle visibility. Use mouse to pan and zoom.

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

