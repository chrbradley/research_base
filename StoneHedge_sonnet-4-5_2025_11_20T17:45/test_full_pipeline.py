"""Test full pipeline: ArUco detection -> Stone segmentation"""

import cv2
import numpy as np
from aruco_calibration import ArucoCalibrator
from opencv_segmentation import OpenCVSegmenter
import json


def segment_stone_from_image(image_path: str, output_dir: str = "."):
    """
    Full pipeline: Load image, detect ArUco, segment stone, extract polygon.

    Args:
        image_path: Path to original stone image with ArUco marker
        output_dir: Output directory
    """
    # Load image
    image = cv2.imread(image_path)
    print(f"Loaded image: {image.shape}")

    # Step 1: Detect ArUco and get calibration
    calibrator = ArucoCalibrator()
    corners, ids, rejected = calibrator.detect_markers(image)

    if corners is None or len(corners) == 0:
        print("ERROR: No ArUco markers detected!")
        return None

    print(f"Detected {len(corners)} ArUco markers")
    ppi = calibrator.calculate_pixel_scale(corners)
    print(f"Scale: {ppi:.2f} pixels per inch")

    # Step 2: Create mask to exclude ArUco marker region
    h, w = image.shape[:2]
    aruco_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Expand ArUco region slightly and mask it out
    for corner in corners:
        pts = corner[0].astype(np.int32)
        # Expand the polygon slightly
        center = pts.mean(axis=0)
        expanded_pts = center + (pts - center) * 1.2
        cv2.fillPoly(aruco_mask, [expanded_pts.astype(np.int32)], 0)

    # Step 3: Segment stone with custom approach
    segmenter = OpenCVSegmenter(min_area=10000)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply mask to remove ArUco region
    gray_masked = cv2.bitwise_and(gray, gray, mask=aruco_mask)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)

    # Use multiple thresholding techniques
    # 1. Otsu's thresholding
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Adaptive thresholding
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5
    )

    # Combine thresholds
    combined = cv2.bitwise_and(thresh_otsu, thresh_adaptive)

    # Apply mask again
    combined = cv2.bitwise_and(combined, combined, mask=aruco_mask)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("ERROR: No contours found!")
        return None

    # Filter and find largest contour (the stone)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 10000]

    if not valid_contours:
        print("ERROR: No valid contours found!")
        return None

    stone_contour = max(valid_contours, key=cv2.contourArea)

    # Simplify polygon
    stone_contour = segmenter.simplify_polygon(stone_contour, epsilon_factor=0.002)

    # Reshape to Nx2
    if stone_contour.shape[1] == 1:
        stone_contour = stone_contour.reshape(-1, 2)

    # Calculate metrics in pixels
    area_pixels = cv2.contourArea(stone_contour)
    perimeter_pixels = cv2.arcLength(stone_contour.reshape(-1, 1, 2), True)

    # Convert to real-world units (inches)
    area_inches = area_pixels / (ppi ** 2)
    perimeter_inches = perimeter_pixels / ppi

    # Get stone dimensions
    rect = cv2.minAreaRect(stone_contour.reshape(-1, 1, 2))
    width_pixels, height_pixels = rect[1]
    width_inches = width_pixels / ppi
    height_inches = height_pixels / ppi

    print(f"\nStone segmentation results:")
    print(f"  Vertices: {len(stone_contour)}")
    print(f"  Area: {area_inches:.2f} sq inches ({area_pixels:.0f} pixels²)")
    print(f"  Perimeter: {perimeter_inches:.2f} inches ({perimeter_pixels:.0f} pixels)")
    print(f"  Dimensions: {width_inches:.2f} x {height_inches:.2f} inches")

    # Visualize
    vis = image.copy()
    cv2.drawContours(vis, [stone_contour.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 0), 5)
    for i, point in enumerate(stone_contour):
        pt = tuple(point.astype(int))
        cv2.circle(vis, pt, 8, (0, 0, 255), -1)

    # Draw ArUco mask for reference
    cv2.drawContours(vis, [pts.astype(np.int32) for pts in [c[0] for c in corners]], -1, (255, 0, 0), 3)

    # Save visualization
    vis_path = f"{output_dir}/stone_segmentation_full.jpg"
    cv2.imwrite(vis_path, vis)
    print(f"  Saved visualization: {vis_path}")

    # Save data
    data = {
        'polygon_pixels': stone_contour.tolist(),
        'num_vertices': len(stone_contour),
        'area_sq_inches': float(area_inches),
        'area_pixels': float(area_pixels),
        'perimeter_inches': float(perimeter_inches),
        'perimeter_pixels': float(perimeter_pixels),
        'width_inches': float(width_inches),
        'height_inches': float(height_inches),
        'pixels_per_inch': float(ppi),
        'aruco_ids': ids.flatten().tolist()
    }

    data_path = f"{output_dir}/stone_data.json"
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved data: {data_path}")

    return data


if __name__ == "__main__":
    test_image = "/home/user/research_base/data/stonehedge/aruco_stones/IMG_3411.JPEG"
    segment_stone_from_image(test_image, ".")
