# StoneHedge Research Prototype

**Model**: Jules
**Date**: 2025-11-20

This repository contains the research and prototype for an automated stone patio/walkway layout planner.

## Deliverables

1.  **Research Report**: See below.
2.  **Prototype Notebook**: `StoneHedge_Prototype.ipynb` (Jupyter Notebook demonstrating the workflow).
3.  **Modular Code**:
    *   `process_stones.py`: Segmentation and Calibration (generates `stones.json`).
    *   `pack_stones.py`: Packing Algorithm (generates `layout.png`).

## Research Report

### 1. Stone Image Segmentation
We evaluated segmentation approaches suitable for identifying irregular stone shapes from photographs.

*   **Selected Approach**: **MobileSAM** (via Ultralytics).
    *   *Reasoning*: Standard SAM (Segment Anything Model) is too heavy for CPU-only environments. MobileSAM offers a good balance of speed and accuracy.
    *   *Performance*: ~3 minutes per image on CPU. Good boundary adherence.
*   **SAM 3**: While SAM 3 was released recently (Nov 2025), it was deemed too experimental and resource-intensive for this initial CPU-based prototype.
*   **Workflow**: Input Image -> ArUco Detection -> Perspective Warp -> MobileSAM -> Polygon Extraction.

### 2. ArUco Marker Calibration
*   We used 5x5 inch ArUco markers placed in the scene.
*   **Method**: `cv2.aruco` detects the marker corners.
*   **Scale**: We calculate pixels-per-inch based on the marker's physical size.
*   **Perspective Correction**: We compute a Homography matrix to warp the image (or polygon coordinates) to a top-down orthographic view, ensuring accurate area and shape calculations.

### 3. Irregular Polygon Packing Algorithm
This is an NP-hard problem (Nesting). We implemented a heuristic approach:

*   **Algorithm**: "Largest-First Monte Carlo with Anchoring".
*   **Steps**:
    1.  Sort stones by area (Largest to Smallest).
    2.  Define a target rectangular area (e.g., 80x60 inches).
    3.  For each stone, attempt N random positions and rotations.
    4.  **Hardscaping Constraint**: For the largest stones (first 40%), prioritize positions furthest from the center (anchoring the edges). For smaller stones, prioritize positions closer to the center (filling gaps).
    5.  **Validation**: Check for containment in target and non-overlap with existing stones (using `shapely` and a gap buffer).
*   **Findings**: This approach successfully packs irregular stones while respecting the "edge anchoring" aesthetic requirement.

### 4. Visualization
*   The prototype visualizes the final layout using `matplotlib`, drawing the placed stone polygons.

## Setup & Usage

### Dependencies
*   Python 3.12+
*   `opencv-contrib-python`
*   `numpy`
*   `matplotlib`
*   `shapely`
*   `ultralytics` (for MobileSAM)
*   `jupyter`, `nbformat`

### Running the Prototype

1.  **Install dependencies**:
    ```bash
    pip install opencv-contrib-python numpy matplotlib shapely ultralytics jupyter nbformat
    ```

2.  **Process Images (Segmentation & Calibration)**:
    *   Place images in `../data/stonehedge/aruco_stones/` (or configure script).
    *   Run:
        ```bash
        python process_stones.py
        ```
    *   This generates `stones.json` (Stone Library).

3.  **Generate Layout (Packing)**:
    *   Run:
        ```bash
        python pack_stones.py
        ```
    *   This generates `layout.png` showing the packed patio.

4.  **View Notebook**:
    *   Open `StoneHedge_Prototype.ipynb` in Jupyter to see the step-by-step tutorial.

## Future Improvements (V2)
*   **Texture Mapping**: Map the original stone textures onto the polygons in the final visualization.
*   **Optimization**: Use a genetic algorithm or simulated annealing for tighter packing.
*   **Web Interface**: Build a frontend to allow users to upload images and define custom target shapes.
