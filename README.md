# Airfoil Designer

Airfoil Designer is a Python-based utility for importing airfoil .dat files and exporting clean Bézier curves, based on Venkataraman’s work (1995, 2017)  
The graphical front‑end is implemented with Qt 6.

---

## Features

* Import Selig or Lednicer `.dat` airfoil point files.
* Input data is normalised to satisfy constraints.  
* Automatic fitting of a **single ninth‑order Bézier** representation with fixed abscissae, matching the 16‑parameter scheme proposed by Venkataraman (2017).  
* Curvature Comb.
* Trailing‑edge thickening and chord‑length scaling prior to export.
* G2 continuity option for the leading edge (upper & lower curves share curvature)
* Direct export to DXF (millimetres) for CAD/CAM workflows.  
* Cross‑platform GUI built with PySide 6.

---

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

