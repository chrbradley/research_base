"""
ArUco Marker Detection and Calibration Module

This module detects ArUco markers in stone images and performs:
1. Marker detection
2. Scale calibration (pixels to real-world units)
3. Perspective correction to get orthographic top-down view
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import json


class ArucoCalibrator:
    """Handles ArUco marker detection and image calibration."""

    # ArUco marker physical size in inches
    MARKER_SIZE_INCHES = 5.0

    def __init__(self, dictionary_type=cv2.aruco.DICT_6X6_250):
        """
        Initialize the calibrator with specified ArUco dictionary.

        Args:
            dictionary_type: OpenCV ArUco dictionary type (default: DICT_6X6_250)
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_markers(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ArUco markers in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (corners, ids, rejected_points)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected

    def calculate_pixel_scale(self, corners: np.ndarray) -> float:
        """
        Calculate pixels per inch based on detected marker corners.

        Args:
            corners: ArUco marker corners (Nx1x4x2 array)

        Returns:
            Pixels per inch scale factor
        """
        # Take the first marker and calculate its edge lengths
        marker_corners = corners[0][0]  # Shape: (4, 2)

        # Calculate all 4 edge lengths
        edge_lengths = []
        for i in range(4):
            p1 = marker_corners[i]
            p2 = marker_corners[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            edge_lengths.append(length)

        # Average edge length in pixels
        avg_edge_pixels = np.mean(edge_lengths)

        # Calculate scale: pixels per inch
        pixels_per_inch = avg_edge_pixels / self.MARKER_SIZE_INCHES

        return pixels_per_inch

    def get_perspective_transform(self, image: np.ndarray, corners: np.ndarray,
                                   ids: np.ndarray, target_ppi: float = 50.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Calculate perspective transform to get orthographic top-down view.

        Args:
            image: Input image
            corners: Detected ArUco corners
            ids: Detected ArUco IDs
            target_ppi: Target pixels per inch for output image

        Returns:
            Tuple of (warped_image, transform_matrix, metadata)
        """
        if corners is None or len(corners) == 0:
            raise ValueError("No markers detected for perspective transform")

        # Get image dimensions
        h, w = image.shape[:2]

        # Calculate current scale
        current_ppi = self.calculate_pixel_scale(corners)

        # Find bounding box of all detected markers
        all_corners = np.vstack([c[0] for c in corners])
        min_x, min_y = np.min(all_corners, axis=0)
        max_x, max_y = np.max(all_corners, axis=0)

        # Add margin (in pixels based on current scale)
        margin_inches = 2.0
        margin_pixels = margin_inches * current_ppi

        # Define source points (corners of region of interest)
        src_points = np.float32([
            [min_x - margin_pixels, min_y - margin_pixels],
            [max_x + margin_pixels, min_y - margin_pixels],
            [max_x + margin_pixels, max_y + margin_pixels],
            [min_x - margin_pixels, max_y + margin_pixels]
        ])

        # Clamp to image boundaries
        src_points[:, 0] = np.clip(src_points[:, 0], 0, w)
        src_points[:, 1] = np.clip(src_points[:, 1], 0, h)

        # Calculate output dimensions in inches
        width_inches = (max_x - min_x) / current_ppi + 2 * margin_inches
        height_inches = (max_y - min_y) / current_ppi + 2 * margin_inches

        # Calculate output dimensions in pixels
        output_width = int(width_inches * target_ppi)
        output_height = int(height_inches * target_ppi)

        # Define destination points (orthographic view)
        dst_points = np.float32([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ])

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Warp the image
        warped = cv2.warpPerspective(image, matrix, (output_width, output_height))

        # Metadata
        metadata = {
            'original_ppi': float(current_ppi),
            'output_ppi': float(target_ppi),
            'output_width_inches': float(width_inches),
            'output_height_inches': float(height_inches),
            'output_width_pixels': output_width,
            'output_height_pixels': output_height,
            'num_markers_detected': len(corners),
            'marker_ids': ids.flatten().tolist() if ids is not None else []
        }

        return warped, matrix, metadata

    def visualize_detection(self, image: np.ndarray, corners: np.ndarray,
                           ids: np.ndarray) -> np.ndarray:
        """
        Draw detected markers on the image for visualization.

        Args:
            image: Input image
            corners: Detected corners
            ids: Detected IDs

        Returns:
            Image with markers drawn
        """
        output = image.copy()
        if corners is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)

            # Add scale information
            if len(corners) > 0:
                ppi = self.calculate_pixel_scale(corners)
                text = f"Scale: {ppi:.2f} pixels/inch"
                cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

        return output


def test_aruco_detection(image_path: str, output_dir: str = "."):
    """
    Test ArUco detection on a single image.

    Args:
        image_path: Path to input image
        output_dir: Directory to save output images
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Loaded image: {image.shape}")

    # Create calibrator
    calibrator = ArucoCalibrator()

    # Detect markers
    corners, ids, rejected = calibrator.detect_markers(image)

    if corners is None or len(corners) == 0:
        print("No ArUco markers detected!")
        return None

    print(f"Detected {len(corners)} markers with IDs: {ids.flatten()}")

    # Calculate scale
    ppi = calibrator.calculate_pixel_scale(corners)
    print(f"Scale: {ppi:.2f} pixels per inch")

    # Visualize detection
    vis_image = calibrator.visualize_detection(image, corners, ids)
    output_path = f"{output_dir}/aruco_detection_viz.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"Saved visualization: {output_path}")

    # Perform perspective correction
    try:
        warped, matrix, metadata = calibrator.get_perspective_transform(image, corners, ids)
        output_path_warped = f"{output_dir}/aruco_warped.jpg"
        cv2.imwrite(output_path_warped, warped)
        print(f"Saved warped image: {output_path_warped}")

        # Save metadata
        metadata_path = f"{output_dir}/calibration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")

        return calibrator, metadata
    except Exception as e:
        print(f"Error in perspective transform: {e}")
        return calibrator, None


if __name__ == "__main__":
    # Test on first sample image
    test_image = "/home/user/research_base/data/stonehedge/aruco_stones/IMG_3411.JPEG"
    test_aruco_detection(test_image, ".")
