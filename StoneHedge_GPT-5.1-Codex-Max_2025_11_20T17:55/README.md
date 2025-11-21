# StoneHedge prototype (GPT-5.1-Codex-Max)

This folder contains a CPU-friendly research prototype that walks through the StoneHedge prompt: segmenting stones from photos with ArUco markers, calibrating them to real-world units, packing irregular polygons into target patios/walkways, and visualizing the layout.

## Contents
- `requirements.txt` – minimal Python stack (OpenCV + Shapely + Matplotlib).
- `src/` – modular helpers for segmentation, calibration, packing, visualization, and an end-to-end pipeline.
- `prototype.ipynb` – runnable demo notebook (load stones → calibrate → segment → pack → visualize).
- `notes.md` – chronological working log.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Images are read from `data/stonehedge/aruco_stones` relative to the repository root.

## Running the prototype
1. Launch Jupyter or VS Code and open `prototype.ipynb`.
2. Execute the cells in order; they will:
   - Load a handful of sample stone photos.
   - Detect ArUco markers to derive pixels-per-inch scale and produce an optional perspective-corrected view.
   - Extract stone polygons via contour-based segmentation (swap this with SAM/SAM2/SAM3 later).
   - Convert polygons into inch units and feed them to a greedy packing routine for configurable target shapes (square/rectangle, 100–400 sq ft).
   - Render the layout with Matplotlib, labeling stones and showing the target boundary.

Key parameters (tunable in the notebook):
- `image_limit`: number of stone photos to process.
- `target_kind`: `"square"` or `"rectangle"`.
- `target_area_sqft`: target area in square feet (e.g., 100, 200, 400).
- `gap_inches`: visual gap size annotated in the plot.

## Implementation notes
- **Segmentation**: `src/segmentation.py` uses adaptive thresholding + morphology + contour simplification to approximate stone outlines while filtering small square-like ArUco contours. It is intentionally lightweight and ready to be swapped for SAM-family models.
- **Calibration**: `src/calibration.py` detects a 5×5 inch ArUco marker to compute pixels-per-inch scale and exposes a perspective warp helper for deskewing.
- **Packing**: `src/packing.py` implements a greedy random placer using Shapely; it respects containment and disjointness and reports filled area and coverage ratio.
- **Visualization**: `src/visualization.py` renders the target polygon plus placed stones with labels and per-stone coloring.
- **Pipeline**: `src/pipeline.py` wires everything together for the notebook, defaulting to the provided ArUco images and a 100 sq ft square target.

## Limitations and next steps
- Contour-based segmentation is a baseline; swap in SAM2/SAM3 or HF alternatives for higher-quality polygons.
- The packing routine is heuristic and does not yet model gap tolerances explicitly or trimming; exploring physics-based or evolutionary solvers is recommended.
- No batching/parallelism; adding caching for segmentation and calibration would speed iteration.
- Visualization can be extended with texture mapping and cut-line overlays using the warped stone imagery.
- Status: actively iterating on the prototype—running the notebook end-to-end and tuning segmentation/calibration parameters to improve coverage without waiting on further input.
