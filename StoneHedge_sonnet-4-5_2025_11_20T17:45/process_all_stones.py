"""Process all stone images and extract polygon data."""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from aruco_calibration import ArucoCalibrator
from opencv_segmentation import OpenCVSegmenter


def process_stone_image(image_path: str, calibrator: ArucoCalibrator,
                       segmenter: OpenCVSegmenter) -> dict:
    """
    Process a single stone image.

    Returns:
        Dictionary with stone data or None if failed
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Detect ArUco
        corners, ids, _ = calibrator.detect_markers(image)
        if corners is None or len(corners) == 0:
            print(f"  ⚠ No ArUco marker: {Path(image_path).name}")
            return None

        # Get scale
        ppi = calibrator.calculate_pixel_scale(corners)

        # Create mask to exclude ArUco
        h, w = image.shape[:2]
        aruco_mask = np.ones((h, w), dtype=np.uint8) * 255

        for corner in corners:
            pts = corner[0].astype(np.int32)
            center = pts.mean(axis=0)
            expanded_pts = center + (pts - center) * 1.2
            cv2.fillPoly(aruco_mask, [expanded_pts.astype(np.int32)], 0)

        # Segment stone
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=aruco_mask)
        blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)

        # Thresholding
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        combined = cv2.bitwise_and(thresh_otsu, thresh_adaptive)
        combined = cv2.bitwise_and(combined, combined, mask=aruco_mask)

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10000]

        if not valid_contours:
            print(f"  ⚠ No valid contour: {Path(image_path).name}")
            return None

        # Get largest contour (the stone)
        stone_contour = max(valid_contours, key=cv2.contourArea)
        stone_contour = segmenter.simplify_polygon(stone_contour, epsilon_factor=0.002)

        if stone_contour.shape[1] == 1:
            stone_contour = stone_contour.reshape(-1, 2)

        # Calculate metrics
        area_pixels = cv2.contourArea(stone_contour)
        perimeter_pixels = cv2.arcLength(stone_contour.reshape(-1, 1, 2), True)
        area_inches = area_pixels / (ppi ** 2)
        perimeter_inches = perimeter_pixels / ppi

        rect = cv2.minAreaRect(stone_contour.reshape(-1, 1, 2))
        width_pixels, height_pixels = rect[1]
        width_inches = width_pixels / ppi
        height_inches = height_pixels / ppi

        # Return data
        return {
            'image_name': Path(image_path).name,
            'image_path': image_path,
            'polygon_pixels': stone_contour.tolist(),
            'num_vertices': len(stone_contour),
            'area_sq_inches': float(area_inches),
            'area_pixels': float(area_pixels),
            'perimeter_inches': float(perimeter_inches),
            'width_inches': float(width_inches),
            'height_inches': float(height_inches),
            'pixels_per_inch': float(ppi),
            'aruco_ids': ids.flatten().tolist() if ids is not None else []
        }

    except Exception as e:
        print(f"  ✗ Error processing {Path(image_path).name}: {e}")
        return None


def process_all_stones(input_dir: str, output_file: str = "stone_library.json"):
    """
    Process all stone images in a directory.

    Args:
        input_dir: Directory containing stone images
        output_file: Output JSON file
    """
    print("Processing all stone images...")
    print(f"Input directory: {input_dir}\n")

    # Get all JPEG files
    image_files = sorted(Path(input_dir).glob("*.JPEG"))
    print(f"Found {len(image_files)} images\n")

    # Initialize
    calibrator = ArucoCalibrator()
    segmenter = OpenCVSegmenter(min_area=10000)

    # Process each image
    stone_library = []
    success_count = 0
    fail_count = 0

    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_file.name}...")

        result = process_stone_image(str(image_file), calibrator, segmenter)

        if result:
            stone_library.append(result)
            success_count += 1
            print(f"  ✓ Area: {result['area_sq_inches']:.1f} sq in, "
                  f"Dims: {result['width_inches']:.1f}\" x {result['height_inches']:.1f}\"")
        else:
            fail_count += 1

    # Save library
    with open(output_file, 'w') as f:
        json.dump(stone_library, f, indent=2)

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nStone library saved to: {output_file}")

    if stone_library:
        total_area = sum(s['area_sq_inches'] for s in stone_library)
        avg_area = total_area / len(stone_library)
        min_area = min(s['area_sq_inches'] for s in stone_library)
        max_area = max(s['area_sq_inches'] for s in stone_library)

        print(f"\nArea Statistics:")
        print(f"  Total area: {total_area:.1f} sq inches ({total_area/144:.1f} sq feet)")
        print(f"  Average: {avg_area:.1f} sq inches")
        print(f"  Min: {min_area:.1f} sq inches")
        print(f"  Max: {max_area:.1f} sq inches")

    return stone_library


if __name__ == "__main__":
    input_dir = "/home/user/research_base/data/stonehedge/aruco_stones"
    stone_library = process_all_stones(input_dir, "stone_library.json")
