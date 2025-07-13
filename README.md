# AirfoilDesigner

AirfoilDesigner is a Python toolkit for reverse‑engineering, parameterising, and exporting airfoils using Bézier‐curve formulations derived from the work of Venkataraman (1995, 2017).  
The graphical front‑end is implemented with Qt 6.
The console app still kind of works, but is no longer maintained.

---

## Features

* Import Selig or Lednicer `.dat` airfoil point files.
* Input data is normalised to satisfy constraints.  
* Automatic fitting of a **single ninth‑order Bézier** representation with fixed abscissae, matching the 16‑parameter scheme proposed by Venkataraman (2017).  
* Alternatively fitting of a **four‑segment cubic Bézier** model (two segments per surface) using a constrained SLSQP optimiser.  
* Curvature Comb.
* Trailing‑edge thickening and chord‑length scaling prior to export.  
* Direct export to DXF (millimetres) for CAD/CAM workflows.  
* Cross‑platform GUI built with PySide 6; the legacy console script remains for batch automation.

---

## Installation
* Make sure Python is installed on your system. 
* Unpack Code.
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
2. **Build Single Bezier Model** Generates the single‑segment ninth‑order model.  
3. **Select error function** Chose between minimum squared error or euclidean iterative closest point.
4. **Regularization Weight** Set optimizer penalty for uneven control point flow. 
5. **Generate 4-Segment Model** Generate the four segemnt model.
6. **Refine Model** Add a conterol point to the worst fitting segment of the four‑segment model (up to degree 9).  
7. **Export DXF**, specifying chord length and trailing‑edge thickness.

---

## Command‑line usage (deprecated)

```bash
python run_console.py --input NACA2412.dat --output naca2412.dxf --chord 1.0 --te 0.002
```

The console app is no longer maintained.

---

## Theoretical background

| Model | Key attributes | Reference |
|-------|----------------|-----------|
| Four‑segment cubic Bézier (top and bottom surfaces) | 14–20 design variables; shared vertices at leading and trailing edges | Venkataraman, *AIAA‑95‑1875* (1995) |
| Single ninth‑order Bézier with fixed abscissae | 16 ordinate variables per airfoil; suitable for morphing‑wing applications | Venkataraman, “A Bézier Parameterisation Scheme for Flexible Airfoils” (2017) |

Both formulations are implemented in `core/optimization_core.py`.

---

## File formats

* **Airfoil input:** Selig or Lednicer `.dat`, chord and orientation are normalized on import.  
* **CAD output:** DXF (millimetres). Separate layers are assigned to the upper and lower surfaces for clarity.

---

## Roadmap

* Add G2 constraints for the segmented model
* Investigate ptimizing abcsisse for single segment model
  

Contributions are welcome.

---

## Citation

If this software is used in academic work, please cite the underlying parameterisation papers:

```text
Venkataraman, P. “A New Procedure for Airfoil Definition.” AIAA‑95‑1875, 1995.

Venkataraman, P. “A Bézier Parameterization Scheme for Flexible Airfoils.” 2017.
```

---

## License

Distributed under the MIT License.

