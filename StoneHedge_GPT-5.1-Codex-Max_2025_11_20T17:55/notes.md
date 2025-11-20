# Working notes

- Created folder `StoneHedge_GPT-5.1-Codex-Max_2025_11_20T17:55` per prompt naming convention.
- Added Python helpers for segmentation (OpenCV contour baseline), calibration (ArUco detection for 5x5 inch markers), packing (greedy random using Shapely), and visualization (Matplotlib polygons).
- Prototype pipeline wires segmentation + calibration to produce inch-scale polygons and packs them into configurable target polygons.
- Requirements captured locally in `requirements.txt`; kept dependencies to CPU-friendly scientific stack.
- Current focus: keep iterating on the prototype (running the notebook end-to-end, improving segmentation calibration tuning) until coverage looks acceptable. No user input is blocking progress.
