# StoneHedge Research Prompt

Create a new folder "StoneHedge" + _MODELNAME ( _MODELNAME is the name of the mode you're using) + _DATE ( DATE is "YYYY_MM_DDTHH:MM" ) and research/prototype an application for automated stone patio/walkway layout planning.

## Problem Statement
Build a system that takes a target polygon (representing a patio or walkway area) and a library of stone images, then generates an optimal placement layout. The stones are natural/irregular shapes that need to be segmented from photos, calibrated for real-world sizing, and packed into the target area following hardscaping best practices.

## Core Technical Challenges to Research

### 1. Stone Image Segmentation
Research and prototype polygon extraction from stone images:

- Evaluate Meta's SAM (Segment Anything Model) - both SAM 2 and the newly released SAM 3
- Investigate device-capable/lightweight versions for potential on-device processing
- Explore the RoboFlow ecosystem for workflow automation and annotation tools
- Survey Hugging Face for alternative segmentation models
- Output should be polygon contours (list of vertices) for each stone

### 2. ArUco Marker Calibration
The stone images contain ArUco markers for calibration:
sample can be found in `/data/stonehedge/aruco_stones`

- Detect ArUco markers to establish real-world scale (pixels → inches/cm)
  - the ArUco Markers in the pictures is 5 inches X 5 inches measured along the black edges. Open CV should detect the markers automatically
- Apply perspective correction/de-skewing to get orthographic top-down views
- Transform extracted polygons to real-world coordinates

### 3. Irregular Polygon Packing Algorithm
This is the key algorithmic challenge. Research and implement a packing algorithm that:

- Places irregular polygons (stones) into an arbitrary target polygon
  - there is no target polygon shape provided. Please try a few different basic linear shapes (minimum of 4 sides): Square, Rectangle, etc of varying areas: eg 100 sq ft, 200, sq ft, 400 sq ft.
  - You can choose more specific sizes once you have a total are extrapolated from the provided stone images
  - experiment with sizes that require both a subset of the stones ( in these cases you can provide multiple solutions ) and all of the stones
- Accepts a gap tolerance parameter (e.g., 1" with 0.25" margin on each side)
- Handles overlapping regions as cut lines (stones may need trimming to fit)
- Enforces even size distribution - large and small stones should be distributed throughout, not clustered (this is hardscaping best practice, research other best practices that we need to consider)
- Explore algorithms: bin packing variants, physics-based settling, genetic algorithms, constraint satisfaction

### 4. Visualization

- Render the final layout showing stone polygons in position
- Map original stone textures onto placed polygons (toggleable)
- Highlight cut lines where overlaps occur
Show gap spacing

## Deliverables

### 1. Research report covering:

- Comparison of segmentation approaches (SAM versions, alternatives, on-device options)
- Survey of packing algorithms suitable for irregular polygons
- Recommended tech stack and architecture


### 2. Working prototype demonstrating end-to-end data flow:

- Demonstrated in an .ipynb file
- Input: Sample target polygon + sample stone images with ArUco markers
- Process: Segmentation → Calibration → Packing → Visualization
- Output: Rendered layout image


### 3. Code that is modular and well-documented, with clear separation between:

- Segmentation module
- Calibration/transformation module
- Packing algorithm module
- Visualization/rendering module


## Scope Notes
### V1 (this research):

- Focus on core data flow and algorithm viability
- Geometric size distribution for stone placement
- Desktop/server processing is fine

### V2 (future):

- Color/shading distribution in packing algorithm
- On-device processing optimization
- PWA/mobile considerations

### Technical Constraints

- Python is preferred for prototyping (OpenCV, PyTorch ecosystem)
- Include sample/synthetic test data if real stone images aren't available
- Document any assumptions about input formats

Generate charts or visualizations comparing algorithm performance where relevant. Include a README with setup instructions and example usage.