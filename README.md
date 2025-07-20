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
*   **Curvature-Based Sampling**: Use curvature-based sampling for the ICP algorithm to improve accuracy.

## Usage

1.  **Load Airfoil Data**: Click the `Load Airfoil` button to load a `.dat` file.
2.  **Generate Airfoil**: Click the `Generate Airfoil` button to fit a Bézier curve to the airfoil data.
3.  **Toggle Thickening**: Click the `Toggle Thickening` button to apply or remove trailing edge thickening.
4.  **Export DXF**: Click the `Export DXF` button to export the Bézier curves to a DXF file.
5.  **Curvature-Based Sampling**: Check the `Use curvature-based sampling` checkbox to enable curvature-based sampling for the ICP algorithm.

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

1. **Open** a `.dat` file.  
2. **Generate Airfoil** Generates the single‑segment ninth‑order model.  
3. **Regularization Weight** Set optimizer penalty for uneven control point flow. Set to 0 for the best fit, 0.001 is a good starting point for a smooth control point flow.
4. **Curve Error Points** Number of samples on the curve used for optimization and error calculation. Higher values improve the fit, but increase processing time. 
7. **Enforce G2** Ensure G2 continuity across the leading edge. Comes with a slight penalty in the overall fit.
8. **Export DXF** Using the specified chord length and trailing‑edge thickness, if applied.

---

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

